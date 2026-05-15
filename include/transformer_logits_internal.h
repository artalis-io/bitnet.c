#ifndef BN_TRANSFORMER_LOGITS_INTERNAL_H
#define BN_TRANSFORMER_LOGITS_INTERNAL_H

#include "quant.h"
#include "transformer_simd_internal.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    float *logits;
    const int8_t *emb_i8;
    const float *emb_scales;
    const int8_t *x_q;
    float x_scale;
    int dim;
} BnLogitsI8Ctx;

typedef struct {
    float *logits;
    const float *x;
    const void *emb;
    int dim;
} BnLogitsCtx;

void bn_transformer_logits_i8_neon_range(void *ctx, int start, int end);
void bn_transformer_logits_i8_avx2_range(void *ctx, int start, int end);
void bn_transformer_logits_i8_scalar_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_native_neon_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_neon_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_avx2_range(void *ctx, int start, int end);
void bn_transformer_logits_i8_wasm_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_wasm_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_scalar_range(void *ctx, int start, int end);
void bn_transformer_logits_f32_range(void *ctx, int start, int end);

#endif // BN_TRANSFORMER_LOGITS_INTERNAL_H
