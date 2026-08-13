#include "quant_dispatch_internal.h"
static int q4_dot_default_enabled(const BnQuantRuntimePolicy *policy) {
    if (policy && policy->disable_q4_dot)
        return 0;
#if defined(BN_FORCE_SCALAR) || \
    (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    return 1;
#else
    return 0;
#endif
}

static int q6_dot_default_enabled(const BnQuantRuntimePolicy *policy) {
    if (policy && policy->disable_q6_dot)
        return 0;
#if defined(BN_FORCE_SCALAR) || \
    (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    return 1;
#else
    return 0;
#endif
}

int bn_quant_policy_avx512_q5k_vnni_enabled(
    const BnQuantRuntimePolicy *policy, int rows) {
    if (policy && policy->avx512_kquant_vnni >= 0)
        return policy->avx512_kquant_vnni;
    return rows >= 4096;
}

int bn_quant_policy_avx2_kquant_float_for_tasks(
    const BnQuantRuntimePolicy *policy,
    const BnMatvecTask *tasks,
    int n_tasks) {
    if (policy && policy->avx2_kquant_float)
        return 1;
    for (int i = 0; i < n_tasks; i++) {
        if (tasks[i].flags & BN_MATVEC_TASK_FORCE_FLOAT_KQUANT)
            return 1;
    }
    return 0;
}

int bn_quant_policy_reference_q4_dot_enabled(
    const BnQuantRuntimePolicy *policy, uint32_t flags) {
    return !(policy && policy->disable_q4_dot) &&
           !(flags & BN_MATVEC_TASK_NATIVE_QUANT) &&
           (q4_dot_default_enabled(policy) ||
            (flags & BN_MATVEC_TASK_REFERENCE_DOT) ||
            (policy && (policy->reference_dot ||
                        policy->reference_q4_dot)));
}

int bn_quant_policy_reference_q6_dot_enabled(
    const BnQuantRuntimePolicy *policy, uint32_t flags) {
    return !(flags & BN_MATVEC_TASK_NATIVE_QUANT) &&
           (q6_dot_default_enabled(policy) ||
            (flags & BN_MATVEC_TASK_REFERENCE_DOT) ||
            (policy && (policy->reference_dot || policy->reference_q4_dot ||
                        policy->reference_q6_dot)));
}

int bn_quant_policy_batch_reference_q4_dot_enabled(
    const BnQuantRuntimePolicy *policy,
    const BnMatvecTask *tasks,
    int n_tasks) {
    if (policy && policy->disable_q4_dot)
        return 0;
    int reference_dot = (policy && (policy->reference_dot ||
                                    policy->reference_q4_dot)) ||
                        q4_dot_default_enabled(policy);
    for (int t = 0; t < n_tasks; t++)
        reference_dot = reference_dot ||
                        ((tasks[t].flags &
                          BN_MATVEC_TASK_REFERENCE_DOT) != 0);
    for (int t = 0; t < n_tasks; t++)
        if (tasks[t].flags & BN_MATVEC_TASK_NATIVE_QUANT)
            reference_dot = 0;
    return reference_dot;
}

int bn_quant_policy_q4_scalar_dot_requested(
    const BnQuantRuntimePolicy *policy) {
    return policy && policy->q4_scalar_dot;
}

int bn_quant_policy_wasm_q4_canonical4_enabled(
    const BnQuantRuntimePolicy *policy) {
    return policy && policy->wasm_q4_canonical4;
}

int bn_quant_policy_q8_0_matmul_batch_enabled(
    const BnQuantRuntimePolicy *policy) {
    return !(policy && policy->disable_native_quant_matmul_batch);
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

int bn_quant_format_supports_f16_float_cache_matvec(int type) {
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

int bn_quant_format_supports_direct_native_quant_matvec(int type) {
    return type == BN_GGUF_TENSOR_Q4_0;
}

int bn_quant_format_supports_specialized_native_quant_matvec(int type) {
    return type == BN_GGUF_TENSOR_Q4_K || type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_supports_reference_prepared_accumulation(int type) {
    return bn_quant_format_has_cap(
        type, BN_QUANT_CAP_GPU_REFERENCE_PREPARED_ACCUMULATION);
}

int bn_quant_format_prefers_specialized_native_quant_matvec(int type,
                                                             int cols) {
    if (!bn_quant_format_supports_specialized_native_quant_matvec(type))
        return 0;
    (void)cols;
    return 1;
}

int bn_quant_format_prefers_tall_specialized_native_quant_matvec(
    int type, int rows, int cols) {
    return type == BN_GGUF_TENSOR_Q6_K && rows >= 65536 && cols > 0 &&
           (cols % 256) == 0;
}

int bn_quant_format_supports_native_quant_split(int type) {
    return bn_quant_format_has_cap(type,
                                   BN_QUANT_CAP_GPU_NATIVE_QUANT_SPLIT);
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
