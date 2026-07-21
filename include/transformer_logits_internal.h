#ifndef BN_TRANSFORMER_LOGITS_INTERNAL_H
#define BN_TRANSFORMER_LOGITS_INTERNAL_H

#include "model_config.h"
#include "quant.h"
#include "threadpool.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef BN_GPU_BACKEND_DECLARED
#define BN_GPU_BACKEND_DECLARED
typedef struct BnGPUBackend BnGPUBackend;
#endif

typedef struct {
    void (*rmsnorm)(float *out, const float *x, const float *w,
                    int size, float eps);
    bn_tp_fn i8_logits;
    int i8_uses_standard_quant;
    int supports_native_quant_refine;
    bn_tp_fn f16_logits;
    void (*prepare_f16_x)(uint16_t *dst, const float *src, int dim);
} BnLogitsBackendOps;

typedef struct {
    float *logits;
    const int8_t *emb_i8;
    const float *emb_scales;
    const int8_t *quantized;
    float activation_scale;
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
int bn_transformer_logits_cpu_tied_kquant_refine_top(void);
int bn_transformer_logits_cpu_tied_kquant_hybrid_top(void);
int bn_transformer_logits_cpu_native_tied_quant_enabled(void);
int bn_transformer_logits_backend_refine_supported(
    const BnLogitsBackendOps *ops, const BnQWeight *W);
int bn_transformer_logits_tied_kquant_refine_supported(const BnQWeight *W);
int bn_transformer_logits_native_quant_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnQWeight *W);
int bn_transformer_logits_native_quant_refine_top(void);
int bn_transformer_logits_untied_uses_f16_path(int tensor_type);
int bn_transformer_logits_tied_uses_quant_path(int tensor_type);
int bn_transformer_logits_tied_uses_f16_path(int tensor_type);
int bn_transformer_logits_tied_i8_weight_type(void);
int bn_transformer_logits_tied_f16_weight_type(void);
int bn_transformer_logits_tied_f32_weight_type(void);
float bn_transformer_logits_final_softcap(const BnConfig *c);
uint32_t bn_transformer_logits_native_quant_task_flags(int enabled);
void bn_transformer_logits_quant_matvec_gpu_buffer_prepared(
    float *out,
    const BnQWeight *W,
    const BnPreparedWeight *prepared,
    void *W_buf,
    const float *x,
    int8_t *quantized_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu);

#endif // BN_TRANSFORMER_LOGITS_INTERNAL_H
