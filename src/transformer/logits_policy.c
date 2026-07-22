#include "transformer_logits_internal.h"
#include "backend_quant.h"
#include "gpu_internal.h"
#include "model_internal.h"

#include <stdlib.h>

int bn_transformer_logits_cpu_tied_kquant_refine_top(void) {
    return bn_backend_quant_cpu_tied_kquant_refine_top();
}

int bn_transformer_logits_cpu_tied_kquant_hybrid_top(void) {
    return bn_backend_quant_cpu_tied_kquant_hybrid_top();
}

int bn_transformer_logits_cpu_native_tied_quant_enabled(void) {
    return getenv("BN_CPU_ENABLE_NATIVE_QUANT_TIED_LOGITS") != NULL ||
           getenv("BN_CPU_NATIVE_TIED_LOGITS") != NULL;
}

int bn_transformer_logits_backend_refine_supported(
    const BnLogitsBackendOps *ops, const BnQWeight *W) {
    return ops && ops->supports_native_quant_refine && W && W->data &&
           bn_backend_quant_supports_native_quant_logits_refine(W->type);
}

int bn_transformer_logits_tied_kquant_refine_supported(const BnQWeight *W) {
    return W && W->data &&
           bn_backend_quant_supports_kquant_logits_refine(W->type);
}

int bn_transformer_logits_native_quant_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnQWeight *W) {
    return W &&
           bn_transformer_gpu_native_quant_logits_refine_enabled(
               gpu, c, W->type);
}

int bn_transformer_logits_native_quant_refine_top(void) {
    return bn_transformer_gpu_native_quant_logits_refine_top(1);
}

int bn_transformer_logits_untied_uses_f16_path(int tensor_type) {
    return bn_backend_quant_logits_uses_f16_path(tensor_type);
}

int bn_transformer_logits_tied_uses_quant_path(int tensor_type) {
    return bn_backend_quant_tied_logits_uses_quant_path(tensor_type);
}

int bn_transformer_logits_tied_uses_f16_path(int tensor_type) {
    return bn_backend_quant_tied_logits_uses_f16_path(tensor_type);
}

int bn_transformer_logits_tied_i8_weight_type(void) {
    return bn_backend_quant_tied_logits_i8_weight_type();
}

int bn_transformer_logits_tied_f16_weight_type(void) {
    return bn_backend_quant_tied_logits_f16_weight_type();
}

int bn_transformer_logits_tied_dense_float_weight_type(void) {
    return bn_backend_quant_tied_logits_dense_float_weight_type();
}

float bn_transformer_logits_final_softcap(const BnConfig *c) {
    return bn_model_config_final_logit_softcap(c);
}

uint32_t bn_transformer_logits_native_quant_task_flags(int enabled) {
    return enabled ? BN_MATVEC_TASK_NATIVE_QUANT : 0u;
}

void bn_transformer_logits_quant_matvec_gpu_buffer_prepared(
    float *out,
    const BnQWeight *W,
    const BnPreparedWeight *prepared,
    void *W_buf,
    const float *x,
    int8_t *quantized_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu) {
    bn_backend_quant_matvec_gpu_buf_prepared(out, W, prepared, W_buf, x,
                                             quantized_buf, pool, gpu);
}
