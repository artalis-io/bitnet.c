#include "quant_dispatch_internal.h"
#include <stdlib.h>

static int quant_env_enabled(const char *name, const char *compat_name) {
    return getenv(name) != NULL ||
           (compat_name != NULL && getenv(compat_name) != NULL);
}

static int reference_dot_env_enabled(void) {
    return quant_env_enabled("BN_CPU_REFERENCE_DOT", "BN_CPU_LLAMA_DOT");
}

static int reference_q4_dot_env_enabled(void) {
    return quant_env_enabled("BN_CPU_REFERENCE_Q4_DOT",
                             "BN_CPU_LLAMA_Q4_DOT");
}

static int reference_q6_dot_env_enabled(void) {
    return quant_env_enabled("BN_CPU_REFERENCE_Q6_DOT",
                             "BN_CPU_LLAMA_Q6_DOT");
}

int bn_quant_policy_avx512_q5k_vnni_enabled(int rows) {
    const char *v = getenv("BN_AVX512_Q5K_VNNI");
    if (v)
        return v[0] != '\0' && v[0] != '0';
    return rows >= 4096;
}

int bn_quant_policy_avx2_kquant_float_for_tasks(
    const BnMatvecTask *tasks,
    int n_tasks) {
    const char *v = getenv("BN_AVX2_KQUANT_FLOAT");
    if (v && v[0] != '\0' && v[0] != '0')
        return 1;
    for (int i = 0; i < n_tasks; i++) {
        if (tasks[i].flags & BN_MATVEC_TASK_FORCE_FLOAT_KQUANT)
            return 1;
    }
    return 0;
}

int bn_quant_policy_reference_q4_dot_enabled(uint32_t flags) {
    return !(flags & BN_MATVEC_TASK_NATIVE_QUANT) &&
           ((flags & BN_MATVEC_TASK_REFERENCE_DOT) ||
            reference_dot_env_enabled() ||
            reference_q4_dot_env_enabled());
}

int bn_quant_policy_reference_q6_dot_enabled(uint32_t flags) {
    return !(flags & BN_MATVEC_TASK_NATIVE_QUANT) &&
           ((flags & BN_MATVEC_TASK_REFERENCE_DOT) ||
            reference_dot_env_enabled() ||
            reference_q4_dot_env_enabled() ||
            reference_q6_dot_env_enabled());
}

int bn_quant_policy_batch_reference_q4_dot_enabled(
    const BnMatvecTask *tasks,
    int n_tasks) {
    int reference_dot = reference_dot_env_enabled() ||
                        reference_q4_dot_env_enabled();
    for (int t = 0; t < n_tasks; t++)
        reference_dot = reference_dot ||
                        ((tasks[t].flags &
                          BN_MATVEC_TASK_REFERENCE_DOT) != 0);
    for (int t = 0; t < n_tasks; t++)
        if (tasks[t].flags & BN_MATVEC_TASK_NATIVE_QUANT)
            reference_dot = 0;
    return reference_dot;
}

int bn_quant_policy_wasm_q4_canonical4_enabled(void) {
    return getenv("BN_WASM_Q4_CANONICAL4") != NULL;
}

int bn_quant_policy_q8_0_matmul_batch_enabled(void) {
    return getenv("BN_DISABLE_Q8_0_MATMUL_BATCH") == NULL;
}

int bn_quant_format_is_q4k(int type) {
    return type == BN_GGUF_TENSOR_Q4_K;
}

int bn_quant_format_is_bf16(int type) {
    return type == BN_GGUF_TENSOR_BF16;
}

int bn_quant_format_is_q3k(int type) {
    return type == BN_GGUF_TENSOR_Q3_K;
}

int bn_quant_format_is_q5k(int type) {
    return type == BN_GGUF_TENSOR_Q5_K;
}

int bn_quant_format_is_q6k(int type) {
    return type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_is_q8k(int type) {
    return type == BN_GGUF_TENSOR_Q8_K;
}

int bn_quant_format_is_q8_0(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

int bn_quant_format_is_q5_0(int type) {
    return type == BN_GGUF_TENSOR_Q5_0;
}

int bn_quant_format_is_f16_float_cache_matvec_candidate(int type) {
    return type == BN_GGUF_TENSOR_Q3_K ||
           type == BN_GGUF_TENSOR_IQ3_XXS ||
           type == BN_GGUF_TENSOR_IQ4_XS;
}

int bn_quant_format_eager_aux_cache_supported(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_BF16:
        case BN_GGUF_TENSOR_Q8_0:
        case BN_GGUF_TENSOR_Q5_0:
        case BN_GGUF_TENSOR_Q3_K:
        case BN_GGUF_TENSOR_Q4_K:
        case BN_GGUF_TENSOR_Q5_K:
        case BN_GGUF_TENSOR_Q6_K:
            return 1;
        default:
            return 0;
    }
}

int bn_quant_format_metal_q4_q8_matvec_supported(int type) {
    return type == BN_GGUF_TENSOR_Q4_0;
}

int bn_quant_format_metal_q6_q8k_matvec_supported(int type) {
    return type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_gpu_matvec_supported(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_F32:
        case BN_GGUF_TENSOR_F16:
        case BN_GGUF_TENSOR_BF16:
        case BN_GGUF_TENSOR_Q8_0:
        case BN_GGUF_TENSOR_Q4_0:
        case BN_GGUF_TENSOR_Q5_0:
        case BN_GGUF_TENSOR_Q3_K:
        case BN_GGUF_TENSOR_Q4_K:
        case BN_GGUF_TENSOR_Q5_K:
        case BN_GGUF_TENSOR_Q6_K:
        case BN_GGUF_TENSOR_Q8_K:
        case BN_GGUF_TENSOR_IQ3_XXS:
        case BN_GGUF_TENSOR_IQ4_XS:
            return 1;
        default:
            return 0;
    }
}
