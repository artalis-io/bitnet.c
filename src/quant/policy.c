#include "quant_dispatch_internal.h"

int bn_quant_batch_preparation_worthwhile(const BnQWeight *W, int n_tokens,
                                          BnThreadPool *pool) {
    if (!W || W->rows <= 0 || W->cols <= 0 || n_tokens <= 0)
        return 0;
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    const BnQuantRuntimePolicy *policy = bn_tp_quant_policy(pool);
    /* Raw Q4_K batching reuses weights without a transient repack. At low
     * reuse packing dominates; non-x8 layouts also lose batch weight reuse.
     * Keep larger x8 batches and diagnostic fallback routes unchanged. */
    if (W->type == BN_GGUF_TENSOR_Q4_K && W->cols % BN_QK_K == 0 &&
        W->cols / BN_QK_K <= 1024 &&
        bn_quant_policy_native_matmul_batch_enabled(policy) &&
        !bn_quant_policy_avx2_kquant_float_for_tasks(policy, NULL, 0) &&
        !bn_quant_policy_reference_q4k_dot_enabled(policy, 0))
        return W->rows % 8 == 0 && n_tokens >= 128;
#else
    (void)pool;
#endif
    return 1;
}

int bn_quant_policy_f32_matmul_fine(int rows, int cols, int n_tokens,
                                   int n_threads) {
    /* Normal 32-row chunks underfill a pool with few output rows. Only
     * split expensive, long dots: short dots can instead suffer output
     * cache-line contention, and wide matrices already balance well.
     * Division and widened multiplication keep dimension checks safe. */
    return rows > 0 && cols >= 2048 && n_tokens > 1 && n_threads > 1 &&
           (rows - 1) / n_threads < 8 &&
           (uint64_t)cols * (uint64_t)n_tokens >= 32768;
}

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
    if (policy && policy->avx2_kquant_float > 0)
        return 1;
    for (int i = 0; i < n_tasks; i++) {
        if (tasks[i].flags & BN_MATVEC_TASK_FORCE_FLOAT_KQUANT)
            return 1;
    }
    return 0;
}

int bn_quant_policy_avx2_q5k_float_matvec_enabled(
    const BnQuantRuntimePolicy *policy) {
    if (policy && policy->avx2_kquant_float >= 0)
        return policy->avx2_kquant_float;
    /* Q5_K matvec and prefill must use the same Q8_K input arithmetic.
     * Float-input evaluation remains an explicit diagnostic override. */
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

int bn_quant_policy_reference_q4k_dot_enabled(
    const BnQuantRuntimePolicy *policy, uint32_t flags) {
    if (policy && policy->disable_q4_dot)
        return 0;
    if (flags & BN_MATVEC_TASK_NATIVE_QUANT)
        return 0;
    if ((flags & BN_MATVEC_TASK_REFERENCE_DOT) ||
        (policy && (policy->reference_dot || policy->reference_q4_dot)))
        return 1;
    return q4_dot_default_enabled(policy);
}

int bn_quant_policy_reference_q6_dot_enabled(
    const BnQuantRuntimePolicy *policy, uint32_t flags) {
    return !(flags & BN_MATVEC_TASK_NATIVE_QUANT) &&
           (q6_dot_default_enabled(policy) ||
            (flags & BN_MATVEC_TASK_REFERENCE_DOT) ||
            (policy && (policy->reference_dot || policy->reference_q4_dot ||
                        policy->reference_q6_dot)));
}

int bn_quant_matvec_uses_prepared_weight(const BnQWeight *W, uint32_t flags,
                                         BnThreadPool *pool) {
    if (!W || bn_quant_prepared_qweight_size(W, NULL) == 0)
        return 0;
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    /* Both the canonical Q5_K integer dot and its float diagnostic read
     * original weights. The optional x8 kernels still expose preparation
     * explicitly, but public dispatch must not build an unused layout. */
    if (W->type == BN_GGUF_TENSOR_Q5_K)
        return 0;
#endif
    if ((flags & BN_MATVEC_TASK_FORCE_FLOAT_KQUANT) &&
        (W->type == BN_GGUF_TENSOR_Q4_K ||
         W->type == BN_GGUF_TENSOR_Q5_K ||
         W->type == BN_GGUF_TENSOR_Q6_K))
        return 0;
    if (W->type == BN_GGUF_TENSOR_Q4_K &&
        bn_quant_policy_reference_q4k_dot_enabled(
            bn_tp_quant_policy(pool), flags))
        return 0;
    if (W->type == BN_GGUF_TENSOR_Q5_K &&
        bn_quant_policy_avx2_q5k_float_matvec_enabled(
            bn_tp_quant_policy(pool)))
        return 0;
    return 1;
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

int bn_quant_policy_native_matmul_batch_enabled(
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
    return type == BN_GGUF_TENSOR_IQ3_XXS ||
           type == BN_GGUF_TENSOR_IQ4_XS;
}

int bn_quant_format_supports_packed_codebook_matvec(int type) {
    return bn_quant_format_has_cap(type,
                                   BN_QUANT_CAP_GPU_PACKED_CODEBOOK_MATVEC);
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
    if (bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_MMVQ_BLOCK32_E8M0))
        return 1;
    switch (type) {
        case BN_GGUF_TENSOR_F32:
        case BN_GGUF_TENSOR_F16:
        case BN_GGUF_TENSOR_BF16:
        case BN_GGUF_TENSOR_Q8_0:
        case BN_GGUF_TENSOR_Q4_0:
        case BN_GGUF_TENSOR_Q5_0:
        case BN_GGUF_TENSOR_Q5_1:
        case BN_GGUF_TENSOR_Q3_K:
        case BN_GGUF_TENSOR_Q4_K:
        case BN_GGUF_TENSOR_Q5_K:
        case BN_GGUF_TENSOR_Q6_K:
        case BN_GGUF_TENSOR_Q8_K:
        case BN_GGUF_TENSOR_IQ3_XXS:
        case BN_GGUF_TENSOR_IQ3_S:
        case BN_GGUF_TENSOR_IQ4_NL:
        case BN_GGUF_TENSOR_IQ4_XS:
            return 1;
        default:
            return 0;
    }
}
