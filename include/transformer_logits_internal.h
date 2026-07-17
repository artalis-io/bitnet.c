#ifndef BN_TRANSFORMER_LOGITS_INTERNAL_H
#define BN_TRANSFORMER_LOGITS_INTERNAL_H

#include "gpu_backend.h"
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
int bn_transformer_logits_q8_refine_supported(
    const BnLogitsBackendOps *ops, const BnQWeight *W);
int bn_transformer_logits_q6_refine_supported(const BnQWeight *W);
int bn_transformer_logits_untied_uses_f16_path(int tensor_type);
int bn_transformer_logits_tied_uses_quant_path(int tensor_type);
int bn_transformer_logits_tied_uses_f16_path(int tensor_type);
int bn_transformer_logits_tied_i8_weight_type(void);
int bn_transformer_logits_tied_f16_weight_type(void);
int bn_transformer_logits_tied_f32_weight_type(void);
void bn_transformer_logits_quant_matvec_gpu_buffer_prepared(
    float *out,
    const BnQWeight *W,
    const BnPreparedWeight *prepared,
    void *W_buf,
    const float *x,
    int8_t *x_q_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu);

#endif // BN_TRANSFORMER_LOGITS_INTERNAL_H
