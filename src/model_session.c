#include "model.h"
#include "sh_arena.h"
#include "turboquant.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>

static int checked_add_size(size_t *acc, size_t add) {
    if (*acc > SIZE_MAX - add) return -1;
    *acc += add;
    return 0;
}

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

size_t bn_model_session_arena_size(const BnConfig *c, const BnWeights *w) {
    if (!c || c->dim <= 0 || c->n_layers <= 0 || c->n_heads <= 0 ||
        c->seq_len <= 0 || c->kv_dim <= 0 || c->head_size <= 0 ||
        c->vocab_size <= 0) return 0;

    size_t att_size = 0;
    if (checked_mul_size((size_t)c->n_heads, (size_t)c->seq_len, &att_size) != 0)
        return 0;
    int n_attn_layers = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval : c->n_layers;
    int n_ssm_layers = c->n_layers - n_attn_layers;
    if (n_attn_layers < 0 || n_ssm_layers < 0) return 0;
    size_t kv_cache_size = 0;
    if (checked_mul3_size((size_t)n_attn_layers, (size_t)c->seq_len,
                          (size_t)c->kv_dim, &kv_cache_size) != 0)
        return 0;

    if (c->n_heads > 0 && c->head_size > 0 &&
        c->n_heads > INT_MAX / c->head_size) return 0;
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;
    int q_size = xb_size;
    int x_q_size = c->dim > c->hidden_dim ? c->dim : c->hidden_dim;
    if (q_dim > x_q_size) x_q_size = q_dim;
    int half_head = c->head_size / 2;

    int hb_size = c->hidden_dim;
    int hb2_size = c->hidden_dim;
    int xb2_size = c->dim;
    if (c->full_attn_interval > 0) {
        size_t qkv_tmp = 0;
        if (checked_mul3_size((size_t)c->ssm_group_count,
                              (size_t)c->ssm_state_size, 2, &qkv_tmp) != 0 ||
            qkv_tmp > (size_t)INT_MAX - (size_t)c->ssm_inner_size)
            return 0;
        int qkv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        if (qkv_dim > hb_size) hb_size = qkv_dim;
        if (c->ssm_inner_size > hb2_size) hb2_size = c->ssm_inner_size;
        if (c->ssm_inner_size > xb2_size) xb2_size = c->ssm_inner_size;
        if (c->ssm_inner_size > x_q_size) x_q_size = c->ssm_inner_size;
        int gq = 2 * q_dim;
        if (gq > hb_size) hb_size = gq;
    }
    if (c->has_shared_expert && c->shared_expert_intermediate_size > hb_size)
        hb_size = c->shared_expert_intermediate_size;
    if (c->has_shared_expert && c->shared_expert_intermediate_size > hb2_size)
        hb2_size = c->shared_expert_intermediate_size;
    if (c->n_experts > 0 && c->moe_intermediate_size > x_q_size)
        x_q_size = c->moe_intermediate_size;

    size_t ssm_state_size_total = 0;
    size_t ssm_conv_state_total = 0;
    if (n_ssm_layers > 0 && c->ssm_time_step_rank > 0) {
        int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
        size_t state_per_layer = 0;
        if (checked_mul3_size((size_t)c->ssm_time_step_rank,
                              (size_t)c->ssm_state_size, (size_t)head_v_dim,
                              &state_per_layer) != 0 ||
            checked_mul_size((size_t)n_ssm_layers, state_per_layer,
                             &ssm_state_size_total) != 0)
            return 0;
        size_t conv_prefix = 0;
        if (checked_mul3_size((size_t)c->ssm_group_count,
                              (size_t)c->ssm_state_size, 2, &conv_prefix) != 0 ||
            conv_prefix > (size_t)INT_MAX - (size_t)c->ssm_inner_size)
            return 0;
        int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        if (c->ssm_conv_kernel <= 0 ||
            checked_mul3_size((size_t)n_ssm_layers,
                              (size_t)(c->ssm_conv_kernel - 1),
                              (size_t)conv_dim, &ssm_conv_state_total) != 0)
            return 0;
    }

    size_t moe_arena_bytes = 0;
    if (c->n_experts > 0 && c->n_layers > 0) {
        size_t moe_expert_buf_size = 0;
        if (w && w->layers) {
            BnMoEExpertMap *em0 = &w->layers[0].expert_map;
            moe_expert_buf_size = em0->expert_gate_bytes;
            if (em0->expert_up_bytes > moe_expert_buf_size)
                moe_expert_buf_size = em0->expert_up_bytes;
            if (em0->expert_down_bytes > moe_expert_buf_size)
                moe_expert_buf_size = em0->expert_down_bytes;
        }

        size_t tmp = 0;
        if (checked_add_size(&moe_arena_bytes, sizeof(BnMoEState)) != 0 ||
            checked_mul_size((size_t)c->n_experts, sizeof(float), &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul_size((size_t)c->dim, sizeof(float), &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul_size((size_t)c->n_experts_active, sizeof(float), &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul_size((size_t)c->n_experts_active, sizeof(int), &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul_size((size_t)c->moe_intermediate_size, sizeof(float), &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul_size(5, moe_expert_buf_size, &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0)
            return 0;
        int moe_k = c->n_experts_active;
        if (moe_k > BN_MAX_MOE_K) moe_k = BN_MAX_MOE_K;
        if (checked_mul3_size((size_t)moe_k, (size_t)c->moe_intermediate_size,
                              sizeof(float), &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul3_size((size_t)moe_k, (size_t)c->dim, sizeof(float), &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul_size((size_t)moe_k, (size_t)c->moe_intermediate_size, &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0 ||
            checked_mul_size((size_t)(13 + 3 * moe_k), SH_ARENA_ALIGN, &tmp) != 0 ||
            checked_add_size(&moe_arena_bytes, tmp) != 0)
            return 0;
    }

    size_t arena_size = 0;
    size_t tmp = 0;
    if (checked_add_size(&tmp, (size_t)c->dim) != 0 ||
        checked_add_size(&tmp, (size_t)xb_size) != 0 ||
        checked_add_size(&tmp, (size_t)xb2_size) != 0 ||
        checked_add_size(&tmp, (size_t)q_size) != 0 ||
        checked_mul_size(tmp, sizeof(float), &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0)
        return 0;
    tmp = 0;
    if (checked_add_size(&tmp, (size_t)hb_size) != 0 ||
        checked_add_size(&tmp, (size_t)hb2_size) != 0 ||
        checked_mul_size(tmp, sizeof(float), &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0 ||
        checked_mul_size(att_size, sizeof(float), &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0 ||
        checked_mul_size((size_t)c->vocab_size, sizeof(float), &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0)
        return 0;
    size_t kv_elem_size = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    if (checked_mul3_size(2, kv_cache_size, kv_elem_size, &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0 ||
        checked_mul_size((size_t)x_q_size, sizeof(int8_t), &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0 ||
        checked_mul_size((size_t)half_head, sizeof(float), &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0)
        return 0;
    tmp = 0;
    if (checked_add_size(&tmp, ssm_state_size_total) != 0 ||
        checked_add_size(&tmp, ssm_conv_state_total) != 0 ||
        checked_mul_size(tmp, sizeof(float), &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0 ||
        checked_add_size(&arena_size, moe_arena_bytes) != 0)
        return 0;

    if (c->kv_tq_bits > 0) {
        BnTQState tq_tmp;
        if (bn_tq_init(&tq_tmp, c->head_size, c->kv_tq_bits, 0) == 0) {
            int key_bytes = bn_tq_key_bytes(&tq_tmp);
            int val_bytes = bn_tq_value_bytes(&tq_tmp);
            size_t tq_keys = 0, tq_vals = 0;
            if (checked_mul4_size((size_t)n_attn_layers, (size_t)c->seq_len,
                                  (size_t)c->n_kv_heads, (size_t)key_bytes, &tq_keys) != 0 ||
                checked_mul4_size((size_t)n_attn_layers, (size_t)c->seq_len,
                                  (size_t)c->n_kv_heads, (size_t)val_bytes, &tq_vals) != 0 ||
                checked_add_size(&arena_size, tq_keys) != 0 ||
                checked_add_size(&arena_size, tq_vals) != 0 ||
                checked_mul3_size((size_t)c->n_heads, (size_t)c->head_size,
                                  sizeof(float), &tmp) != 0 ||
                checked_add_size(&arena_size, tmp) != 0 ||
                checked_mul_size(3, SH_ARENA_ALIGN, &tmp) != 0 ||
                checked_add_size(&arena_size, tmp) != 0) {
                bn_tq_free(&tq_tmp);
                return 0;
            }
            bn_tq_free(&tq_tmp);
        }
    }

    if (checked_mul_size(16, SH_ARENA_ALIGN, &tmp) != 0 ||
        checked_add_size(&arena_size, tmp) != 0)
        return 0;

    return arena_size;
}

static void alloc_moe_pread_bufs(BnMoEState *ms, const BnWeights *w, SHArena *arena) {
    if (!ms || !w || !w->layers) return;
    BnMoEExpertMap *em0 = &w->layers[0].expert_map;
    size_t buf_size = em0->expert_gate_bytes;
    if (em0->expert_up_bytes > buf_size) buf_size = em0->expert_up_bytes;
    if (em0->expert_down_bytes > buf_size) buf_size = em0->expert_down_bytes;

    ms->buf       = (uint8_t *)sh_arena_alloc(arena, buf_size);
    ms->buf_size  = buf_size;
    ms->buf2      = (uint8_t *)sh_arena_alloc(arena, buf_size);
    ms->buf2_size = buf_size;
    ms->buf3      = (uint8_t *)sh_arena_alloc(arena, buf_size);
    ms->buf3_size = buf_size;
    ms->buf4      = (uint8_t *)sh_arena_alloc(arena, buf_size);
    ms->buf4_size = buf_size;
    ms->buf5      = (uint8_t *)sh_arena_alloc(arena, buf_size);
    ms->buf5_size = buf_size;
}

int bn_model_alloc_session_buffers(const BnConfig *c, const BnWeights *w,
                                    SHArena *arena,
                                    BnRunState *state, BnMoEState **moe_out) {
    size_t att_size = 0;
    if (checked_mul_size((size_t)c->n_heads, (size_t)c->seq_len, &att_size) != 0)
        return -1;

    int n_attn_layers = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval : c->n_layers;
    int n_ssm_layers = c->n_layers - n_attn_layers;
    if (n_attn_layers < 0 || n_ssm_layers < 0) return -1;
    size_t kv_cache_size = 0;
    if (checked_mul3_size((size_t)n_attn_layers, (size_t)c->seq_len,
                          (size_t)c->kv_dim, &kv_cache_size) != 0)
        return -1;

    if (c->n_heads > 0 && c->head_size > 0 &&
        c->n_heads > INT_MAX / c->head_size) return -1;
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;
    int q_size = xb_size;
    int x_q_size = c->dim > c->hidden_dim ? c->dim : c->hidden_dim;
    if (q_dim > x_q_size) x_q_size = q_dim;
    int half_head = c->head_size / 2;

    int hb_size = c->hidden_dim;
    int hb2_size = c->hidden_dim;
    int xb2_size = c->dim;
    if (c->full_attn_interval > 0) {
        int qkv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        if (qkv_dim > hb_size) hb_size = qkv_dim;
        if (c->ssm_inner_size > hb2_size) hb2_size = c->ssm_inner_size;
        if (c->ssm_inner_size > xb2_size) xb2_size = c->ssm_inner_size;
        if (c->ssm_inner_size > x_q_size) x_q_size = c->ssm_inner_size;
        int gq = 2 * q_dim;
        if (gq > hb_size) hb_size = gq;
    }
    if (c->has_shared_expert && c->shared_expert_intermediate_size > hb_size)
        hb_size = c->shared_expert_intermediate_size;
    if (c->has_shared_expert && c->shared_expert_intermediate_size > hb2_size)
        hb2_size = c->shared_expert_intermediate_size;
    if (c->n_experts > 0 && c->moe_intermediate_size > x_q_size)
        x_q_size = c->moe_intermediate_size;

    size_t kv_elem_size = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    BnRunState *s = state;

    s->x           = (float *)sh_arena_calloc(arena, c->dim, sizeof(float));
    s->xb          = (float *)sh_arena_calloc(arena, xb_size, sizeof(float));
    s->xb2         = (float *)sh_arena_calloc(arena, xb2_size, sizeof(float));
    s->q           = (float *)sh_arena_calloc(arena, q_size, sizeof(float));
    s->hb          = (float *)sh_arena_calloc(arena, hb_size, sizeof(float));
    s->hb2         = (float *)sh_arena_calloc(arena, hb2_size, sizeof(float));
    s->att         = (float *)sh_arena_calloc(arena, att_size, sizeof(float));
    s->logits      = (float *)sh_arena_calloc(arena, c->vocab_size, sizeof(float));
    s->key_cache   = (float *)sh_arena_calloc(arena, kv_cache_size, kv_elem_size);
    s->value_cache = (float *)sh_arena_calloc(arena, kv_cache_size, kv_elem_size);
    s->x_q         = (int8_t *)sh_arena_calloc(arena, x_q_size, sizeof(int8_t));
    s->rope_freq   = (float *)sh_arena_alloc(arena, half_head * sizeof(float));

    s->ssm_state = NULL;
    s->ssm_conv_state = NULL;
    size_t ssm_state_size_total = 0;
    size_t ssm_conv_state_total = 0;
    if (n_ssm_layers > 0 && c->ssm_time_step_rank > 0) {
        int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
        size_t state_per_layer = 0;
        if (checked_mul3_size((size_t)c->ssm_time_step_rank,
                              (size_t)c->ssm_state_size, (size_t)head_v_dim,
                              &state_per_layer) != 0 ||
            checked_mul_size((size_t)n_ssm_layers, state_per_layer,
                             &ssm_state_size_total) != 0)
            return -1;
        int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        if (checked_mul3_size((size_t)n_ssm_layers,
                              (size_t)(c->ssm_conv_kernel - 1),
                              (size_t)conv_dim, &ssm_conv_state_total) != 0)
            return -1;
        s->ssm_state = (float *)sh_arena_calloc(arena, ssm_state_size_total, sizeof(float));
        s->ssm_conv_state = (float *)sh_arena_calloc(arena, ssm_conv_state_total, sizeof(float));
    }

    s->key_cache_tq = NULL;
    s->value_cache_tq = NULL;
    s->q_rotated = NULL;
    if (c->kv_tq_bits > 0) {
        BnTQState tq_tmp;
        if (bn_tq_init(&tq_tmp, c->head_size, c->kv_tq_bits, 0) == 0) {
            int key_bytes = bn_tq_key_bytes(&tq_tmp);
            int val_bytes = bn_tq_value_bytes(&tq_tmp);
            size_t tq_key_total = 0, tq_val_total = 0, q_rot_total = 0;
            if (checked_mul4_size((size_t)n_attn_layers, (size_t)c->seq_len,
                                  (size_t)c->n_kv_heads, (size_t)key_bytes,
                                  &tq_key_total) != 0 ||
                checked_mul4_size((size_t)n_attn_layers, (size_t)c->seq_len,
                                  (size_t)c->n_kv_heads, (size_t)val_bytes,
                                  &tq_val_total) != 0 ||
                checked_mul_size((size_t)c->n_heads, (size_t)c->head_size,
                                 &q_rot_total) != 0) {
                bn_tq_free(&tq_tmp);
                return -1;
            }
            s->key_cache_tq   = (uint8_t *)sh_arena_calloc(arena, tq_key_total, 1);
            s->value_cache_tq = (uint8_t *)sh_arena_calloc(arena, tq_val_total, 1);
            s->q_rotated      = (float *)sh_arena_calloc(arena, q_rot_total, sizeof(float));
            bn_tq_free(&tq_tmp);
            if (!s->key_cache_tq || !s->value_cache_tq || !s->q_rotated)
                return -1;
        }
    }

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache ||
        !s->x_q || !s->rope_freq)
        return -1;
    if (ssm_state_size_total > 0 && (!s->ssm_state || !s->ssm_conv_state))
        return -1;

    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : c->head_size;
    int half_rope = rope_dims / 2;
    for (int i = 0; i < half_rope; i++)
        s->rope_freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)rope_dims);
    if (c->rope_text_dims > 0) {
        int text_pairs = c->rope_text_dims / 2;
        for (int i = text_pairs; i < half_rope; i++)
            s->rope_freq[i] = 0.0f;
    }

    *moe_out = NULL;
    if (c->n_experts > 0) {
        BnMoEState *ms = (BnMoEState *)sh_arena_calloc(arena, 1, sizeof(BnMoEState));
        if (!ms) return -1;

        ms->router_logits  = (float *)sh_arena_calloc(arena, c->n_experts, sizeof(float));
        ms->expert_out     = (float *)sh_arena_calloc(arena, c->dim, sizeof(float));
        ms->expert_weights = (float *)sh_arena_calloc(arena, c->n_experts_active, sizeof(float));
        ms->expert_indices = (int *)sh_arena_calloc(arena, c->n_experts_active, sizeof(int));
        ms->expert_hb      = (float *)sh_arena_calloc(arena, c->moe_intermediate_size, sizeof(float));
        ms->expert_hb2     = (float *)sh_arena_calloc(arena, c->moe_intermediate_size, sizeof(float));

        alloc_moe_pread_bufs(ms, w, arena);

        if (!ms->router_logits || !ms->expert_out || !ms->expert_weights ||
            !ms->expert_indices || !ms->expert_hb || !ms->expert_hb2)
            return -1;

        int moe_k = c->n_experts_active;
        if (moe_k > BN_MAX_MOE_K) moe_k = BN_MAX_MOE_K;
        for (int k = 0; k < moe_k; k++) {
            ms->expert_hb_batch[k]   = (float *)sh_arena_calloc(arena, c->moe_intermediate_size, sizeof(float));
            ms->expert_hb2_batch[k]  = (float *)sh_arena_calloc(arena, c->moe_intermediate_size, sizeof(float));
            ms->expert_down_batch[k] = (float *)sh_arena_calloc(arena, c->dim, sizeof(float));
            if (!ms->expert_hb_batch[k] || !ms->expert_hb2_batch[k] || !ms->expert_down_batch[k])
                return -1;
        }

        ms->down_x_q_bufs = (int8_t *)sh_arena_alloc(arena,
            (size_t)moe_k * c->moe_intermediate_size);

        *moe_out = ms;
    }

    return 0;
}
