#ifndef BN_TRANSFORMER_LOGITS_INTERNAL_H
#define BN_TRANSFORMER_LOGITS_INTERNAL_H

#include "quant.h"
#include "threadpool.h"
#include "transformer_simd_internal.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    void (*rmsnorm)(float *out, const float *x, const float *w,
                    int size, float eps);
    bn_tp_fn i8_logits;
    int i8_uses_standard_quant;
    int supports_q8_refine;
    bn_tp_fn f16_logits;
    void (*prepare_f16_x)(uint16_t *dst, const float *src, int dim);
} BnLogitsBackendOps;

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
const BnLogitsBackendOps *bn_transformer_logits_backend_ops(void);
int bn_transformer_logits_cpu_tied_q6k_refine_top(void);
int bn_transformer_logits_cpu_tied_q6k_hybrid_top(void);
int bn_transformer_logits_cpu_native_tied_quant_enabled(void);

#endif // BN_TRANSFORMER_LOGITS_INTERNAL_H
