#include "session.h"
#include "turboquant.h"
#include "sh_log.h"
#include <stdlib.h>
#include <string.h>

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

    // Create session arena
    size_t arena_size = bn_model_session_arena_size(c, &model->weights);
    if (arena_size > SIZE_MAX / 2) {
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->arena = sh_arena_create(arena_size);
    if (!s->arena) {
        SH_LOG_ERROR("Failed to allocate session arena");
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    // Allocate session buffers
    if (bn_model_alloc_session_buffers(c, &model->weights, s->arena,
                                        &s->state, &s->moe_state) != 0) {
        SH_LOG_ERROR("Failed to allocate session buffers");
        sh_arena_free(s->arena);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->pos = 0;
    return s;
}

void bn_session_free(BnSession *s, BnAllocator *alloc) {
    if (!s) return;
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
    size_t kv_size = (size_t)n_attn * c->seq_len * c->kv_dim;
    size_t kv_elem = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    memset(rs->key_cache, 0, kv_size * kv_elem);
    memset(rs->value_cache, 0, kv_size * kv_elem);

    // TQ compressed KV cache
    if (rs->key_cache_tq && rs->value_cache_tq && c->kv_tq_bits > 0 && model->tq_state) {
        int kb = bn_tq_key_bytes(model->tq_state);
        int vb = bn_tq_value_bytes(model->tq_state);
        size_t tq_key_total = (size_t)n_attn * (size_t)c->seq_len * (size_t)c->n_kv_heads * (size_t)kb;
        size_t tq_val_total = (size_t)n_attn * (size_t)c->seq_len * (size_t)c->n_kv_heads * (size_t)vb;
        memset(rs->key_cache_tq, 0, tq_key_total);
        memset(rs->value_cache_tq, 0, tq_val_total);
    }

    // SSM state
    if (rs->ssm_state && c->ssm_time_step_rank > 0) {
        int n_ssm = c->n_layers - n_attn;
        int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
        size_t state_total = (size_t)n_ssm * c->ssm_time_step_rank *
                             c->ssm_state_size * head_v_dim;
        memset(rs->ssm_state, 0, state_total * sizeof(float));
    }
    if (rs->ssm_conv_state) {
        int n_ssm = c->n_layers - n_attn;
        int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        size_t conv_total = (size_t)n_ssm * (c->ssm_conv_kernel - 1) * conv_dim;
        memset(rs->ssm_conv_state, 0, conv_total * sizeof(float));
    }

    s->pos = 0;
}
