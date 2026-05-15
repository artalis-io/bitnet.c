#ifndef BN_TRANSFORMER_BATCHED_ATTN_INTERNAL_H
#define BN_TRANSFORMER_BATCHED_ATTN_INTERNAL_H

#include "model.h"
#include <stddef.h>

typedef struct {
    const BnConfig *c;
    BnRunState *s;

    float *Q_buf;       // [n_tokens * wq_rows]
    float *K_new;       // [n_tokens * kv_dim]
    float *V_new;       // [n_tokens * kv_dim]
    float *out;         // [n_tokens * wo_cols] (typically reuses Q_buf)

    size_t loff;
    int pos0;
    int n_tokens;

    int n_heads;
    int n_kv_heads;
    int head_size;
    int kv_dim;
    int kv_mul;
    int seq_len;

    int rope_dims;
    float *rope_freq;

    float *rope_cos;    // [n_tokens * half_rope] pre-allocated by caller
    float *rope_sin;    // [n_tokens * half_rope] pre-allocated by caller
    int rope_stride;    // stride between token RoPE rows, in half-rotary elements
    float attention_scale;

    const float *q_norm;
    const float *k_norm;
    const float *q_bias;
    const float *k_bias;
    const float *v_bias;
    int qk_norm_per_head;
    float norm_eps;

    int q_gated;
    int wq_rows;
    int wo_cols;
} BnBatchedAttnCtx;

void bn_transformer_batched_attn_dispatch(BnModel *m, BnBatchedAttnCtx *ctx);

void bn_transformer_batched_attn_naive_avx2_range(void *ctx, int start, int end);
void bn_transformer_batched_attn_naive_scalar_range(void *ctx, int start, int end);
void bn_transformer_batched_attn_flash_avx2_range(void *ctx, int start, int end);
void bn_transformer_batched_attn_flash_scalar_range(void *ctx, int start, int end);

#endif
