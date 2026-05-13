#include "model.h"
#include "session.h"
#include "backend_model.h"
#include "turboquant.h"
#include "sh_log.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int checked_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t tmp;
    if (checked_mul_size(a, b, &tmp) != 0) return -1;
    return checked_mul_size(tmp, c, out);
}

static int checked_mul4_size(size_t a, size_t b, size_t c, size_t d, size_t *out) {
    size_t tmp;
    if (checked_mul3_size(a, b, c, &tmp) != 0) return -1;
    return checked_mul_size(tmp, d, out);
}

BnSession *bn_session_create(const BnModel *model, BnAllocator *alloc) {
    if (!model) return NULL;
    const BnConfig *c = &model->config;

    // Allocate session struct
    BnSession *s;
    if (alloc) {
        s = (BnSession *)bn_malloc(alloc, sizeof(BnSession));
    } else {
        s = (BnSession *)malloc(sizeof(BnSession));
    }
    if (!s) return NULL;
    memset(s, 0, sizeof(BnSession));
    s->backend = bn_backend_session_create();
    if (!s->backend) {
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    // Create session arena
    size_t arena_size = bn_model_session_arena_size(c, &model->weights);
    if (arena_size > SIZE_MAX / 2) {
        bn_backend_session_free(s->backend);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->arena = sh_arena_create(arena_size);
    if (!s->arena) {
        SH_LOG_ERROR("Failed to allocate session arena");
        bn_backend_session_free(s->backend);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    // Allocate session buffers
    if (bn_model_alloc_session_buffers(c, &model->weights, s->arena,
                                        &s->state, &s->moe_state) != 0) {
        SH_LOG_ERROR("Failed to allocate session buffers");
        sh_arena_free(s->arena);
        bn_backend_session_free(s->backend);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->pos = 0;
    return s;
}

void bn_session_free(BnSession *s, BnAllocator *alloc) {
    if (!s) return;
    bn_backend_session_free(s->backend);
    sh_arena_free(s->arena);
    if (alloc) {
        bn_free(alloc, s, sizeof(BnSession));
    } else {
        free(s);
    }
}

void bn_session_reset(BnSession *s, const BnModel *model) {
    if (!s || !model) return;
    const BnConfig *c = &model->config;
    BnRunState *rs = &s->state;

    // KV cache
    int n_attn = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval : c->n_layers;
    size_t kv_size = 0;
    if (n_attn < 0 ||
        checked_mul3_size((size_t)n_attn, (size_t)c->seq_len,
                          (size_t)c->kv_dim, &kv_size) != 0)
        return;
    size_t kv_elem = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    size_t kv_bytes = 0;
    if (checked_mul_size(kv_size, kv_elem, &kv_bytes) != 0) return;
    memset(rs->key_cache, 0, kv_bytes);
    memset(rs->value_cache, 0, kv_bytes);

    // TQ compressed KV cache
    if (rs->key_cache_tq && rs->value_cache_tq && c->kv_tq_bits > 0 && bn_model_tq_state(model)) {
        int kb = bn_tq_key_bytes(bn_model_tq_state(model));
        int vb = bn_tq_value_bytes(bn_model_tq_state(model));
        size_t tq_key_total = 0, tq_val_total = 0;
        if (checked_mul4_size((size_t)n_attn, (size_t)c->seq_len,
                              (size_t)c->n_kv_heads, (size_t)kb, &tq_key_total) != 0 ||
            checked_mul4_size((size_t)n_attn, (size_t)c->seq_len,
                              (size_t)c->n_kv_heads, (size_t)vb, &tq_val_total) != 0)
            return;
        memset(rs->key_cache_tq, 0, tq_key_total);
        memset(rs->value_cache_tq, 0, tq_val_total);
    }

    // SSM state
    if (rs->ssm_state && c->ssm_time_step_rank > 0) {
        int n_ssm = c->n_layers - n_attn;
        int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
        size_t state_total = 0;
        if (n_ssm < 0 ||
            checked_mul4_size((size_t)n_ssm, (size_t)c->ssm_time_step_rank,
                              (size_t)c->ssm_state_size, (size_t)head_v_dim,
                              &state_total) != 0 ||
            checked_mul_size(state_total, sizeof(float), &state_total) != 0)
            return;
        memset(rs->ssm_state, 0, state_total);
    }
    if (rs->ssm_conv_state) {
        int n_ssm = c->n_layers - n_attn;
        int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        size_t conv_total = 0;
        if (n_ssm < 0 ||
            checked_mul4_size((size_t)n_ssm, (size_t)(c->ssm_conv_kernel - 1),
                              (size_t)conv_dim, sizeof(float), &conv_total) != 0)
            return;
        memset(rs->ssm_conv_state, 0, conv_total);
    }

    s->pos = 0;
}
