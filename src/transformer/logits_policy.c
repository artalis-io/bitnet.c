#include "transformer_logits_internal.h"
#include "backend_quant.h"

#include <stdlib.h>

static int logits_env_top_n(const char *name, int min_value) {
    const char *env = getenv(name);
    if (!env)
        return 0;
    int top_n = atoi(env);
    if (top_n < min_value)
        return 0;
    return top_n > 128 ? 128 : top_n;
}

int bn_transformer_logits_cpu_tied_q6k_refine_top(void) {
    return logits_env_top_n("BN_CPU_TIED_Q6K_REFINE_TOP", 1);
}

int bn_transformer_logits_cpu_tied_q6k_hybrid_top(void) {
    return logits_env_top_n("BN_CPU_TIED_Q6K_HYBRID_TOP", 2);
}

int bn_transformer_logits_cpu_native_tied_quant_enabled(void) {
    return getenv("BN_CPU_NATIVE_TIED_LOGITS") != NULL;
}

int bn_transformer_logits_q8_refine_supported(
    const BnLogitsBackendOps *ops, const BnQWeight *W) {
    return ops && ops->supports_q8_refine && W && W->data &&
           bn_backend_quant_supports_q8_logits_refine(W->type);
}

int bn_transformer_logits_q6_refine_supported(const BnQWeight *W) {
    return W && W->data &&
           bn_backend_quant_supports_q6k_logits_refine(W->type);
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

int bn_transformer_logits_tied_f32_weight_type(void) {
    return bn_backend_quant_tied_logits_f32_weight_type();
}

void bn_transformer_logits_quant_matvec_gpu_buffer_prepared(
    float *out,
    const BnQWeight *W,
    const BnPreparedWeight *prepared,
    void *W_buf,
    const float *x,
    int8_t *x_q_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu) {
    bn_backend_quant_matvec_gpu_buf_prepared(out, W, prepared, W_buf, x,
                                             x_q_buf, pool, gpu);
}
