#ifndef BN_TRANSFORMER_GQA_INTERNAL_H
#define BN_TRANSFORMER_GQA_INTERNAL_H

#include "model.h"
#include "transformer_math_internal.h"
#include "transformer_simd_internal.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct BnTQState BnTQState;

typedef struct {
    const BnConfig *c;
    BnRunState *s;
    size_t loff;
    int pos;
    int n_kv;       // min(pos+1, seq_len) -- number of valid KV entries
    int kv_mul;
    int head_size;
    int kv_dim;
    int seq_len;    // cache size for modular indexing
} BnGQACtx;

void bn_transformer_cpu_gqa_dispatch(BnModel *m,
                                     BnGQACtx *gctx,
                                     int n_heads,
                                     int kv_mul);
void bn_transformer_cpu_apply_rope_heads(float *buf,
                                         int n_heads,
                                         int head_size,
                                         int rope_dims,
                                         const float *rc,
                                         const float *rs);

typedef struct {
    const BnConfig *c;
    BnRunState *s;
    const BnTQState *tq;
    const uint8_t *tq_keys;    // layer's packed keys base
    const uint8_t *tq_values;  // layer's packed values base
    int key_stride;            // bytes per position (n_kv_heads * key_bytes)
    int val_stride;            // bytes per position (n_kv_heads * val_bytes)
    int key_bytes;             // bytes per single head's key
    int val_bytes;             // bytes per single head's value
    int pos;
    int n_kv;
    int kv_mul;
    int head_size;
    int seq_len;
    int n_kv_heads;
} BnGQATQCtx;

void bn_transformer_gqa_neon_range(void *ctx, int start, int end);
void bn_transformer_gqa_avx2_range(void *ctx, int start, int end);
void bn_transformer_gqa_wasm_range(void *ctx, int start, int end);
void bn_transformer_gqa_scalar_range(void *ctx, int start, int end);
void bn_transformer_gqa_tq_neon_range(void *ctx, int start, int end);
void bn_transformer_gqa_tq_scalar_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_neon_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_avx2_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_wasm_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_scalar_range(void *ctx, int start, int end);

#endif // BN_TRANSFORMER_GQA_INTERNAL_H
