#include "quant.h"
#include "backend_quant.h"
#include "quant_dispatch_internal.h"
#include "gguf.h"
#include "sh_arena.h"
#include "quant_ctx.h"
#include "quant_kernels_scalar.h"
#if defined(__AVX2__)
#include "quant_kernels_avx2.h"
#include "simd_helpers.h"
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include "quant_kernels_neon.h"
#include "simd_helpers.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

static uint16_t test_fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void test_q8_blocks_scalar_reference(const float *x,
                                            int8_t *q,
                                            float *scales,
                                            int n) {
    for (int b = 0; b < n / 32; b++) {
        const float *xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++)
            amax = fmaxf(amax, fabsf(xb[i]));
        if (amax == 0.0f) {
            memset(q + b * 32, 0, 32);
            scales[b] = 0.0f;
            continue;
        }
        float scale = amax / 127.0f;
        float inv_scale = 1.0f / scale;
        scales[b] = bn_fp16_to_fp32(bn_fp32_to_fp16(scale));
        for (int i = 0; i < 32; i++) {
            int v = (int)lrintf(xb[i] * inv_scale);
            if (v < -127) v = -127;
            if (v > 127) v = 127;
            q[b * 32 + i] = (int8_t)v;
        }
    }
}

static const BnQuantRuntimePolicy *test_quant_policy_from_env(void) {
    static BnQuantRuntimePolicy policy;
    bn_quant_runtime_policy_from_env(&policy);
    return &policy;
}

#define TEST_QUANT_POLICY() test_quant_policy_from_env()

static void test_quant_policy_helpers(void) {
    printf("test_quant_policy_helpers... ");
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_IQ3_S,
        BN_QUANT_CAP_GPU_MMQ_F32_SCALE | BN_QUANT_CAP_GPU_MMQ_SUBBLOCK32 |
        BN_QUANT_CAP_GPU_MMVQ_SUBBLOCK32_INT_SCALE));
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_IQ4_XS,
        BN_QUANT_CAP_GPU_MMVQ_SUBBLOCK32_INT_SCALE));

    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_IQ4_NL,
        BN_QUANT_CAP_GPU_MMVQ_BLOCK32_CODEBOOK_FP16));
    assert(!bn_quant_format_has_cap(BN_GGUF_TENSOR_IQ4_NL,
        BN_QUANT_CAP_GPU_MMVQ_SUBBLOCK32_INT_SCALE));

    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q3_K,
        BN_QUANT_CAP_GPU_MMVQ_SUBBLOCK16_INPUT_SCALE));

    BnQWeight reuse_weight = {NULL, BN_GGUF_TENSOR_Q4_K, 640, 2560, 1.0f};
    assert(!bn_quant_batch_preparation_worthwhile(NULL, 128, NULL));
    assert(!bn_quant_batch_preparation_worthwhile(&reuse_weight, 0, NULL));
    assert(!bn_quant_batch_preparation_worthwhile(&reuse_weight, -1, NULL));
    assert(bn_quant_batch_preparation_worthwhile(&reuse_weight, 128, NULL));
    assert(bn_quant_batch_preparation_worthwhile(&reuse_weight, INT_MAX, NULL));
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    assert(!bn_quant_batch_preparation_worthwhile(&reuse_weight, 1, NULL));
    assert(!bn_quant_batch_preparation_worthwhile(&reuse_weight, 127, NULL));
    reuse_weight.rows = 143;
    assert(!bn_quant_batch_preparation_worthwhile(&reuse_weight, 256, NULL));
#else
    assert(bn_quant_batch_preparation_worthwhile(&reuse_weight, 1, NULL));
#endif
    reuse_weight.type = BN_GGUF_TENSOR_Q6_K;
    assert(bn_quant_batch_preparation_worthwhile(&reuse_weight, 9, NULL));
    reuse_weight.rows = 0;
    assert(!bn_quant_batch_preparation_worthwhile(&reuse_weight, 9, NULL));

    assert(bn_quant_policy_f32_matmul_fine(48, 2560, 128, 8));
    assert(bn_quant_policy_f32_matmul_fine(64, 2048, 16, 8));
    assert(!bn_quant_policy_f32_matmul_fine(65, 2048, 16, 8));
    assert(!bn_quant_policy_f32_matmul_fine(48, 2047, 17, 8));
    assert(!bn_quant_policy_f32_matmul_fine(48, 2048, 15, 8));
    assert(!bn_quant_policy_f32_matmul_fine(48, 2560, 128, 1));
    assert(!bn_quant_policy_f32_matmul_fine(48, 2560, 128, 0));
    assert(!bn_quant_policy_f32_matmul_fine(0, 2560, 128, 8));
    assert(!bn_quant_policy_f32_matmul_fine(-1, 2560, 128, 8));
    assert(!bn_quant_policy_f32_matmul_fine(48, -1, 128, 8));
    assert(!bn_quant_policy_f32_matmul_fine(48, 2560, 0, 8));
    assert(!bn_quant_policy_f32_matmul_fine(48, INT_MAX, 1, 8));
    assert(bn_quant_policy_f32_matmul_fine(
        INT_MAX, INT_MAX, INT_MAX, INT_MAX));

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    assert(bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_IQ3_S));
    assert(bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_IQ4_XS));
    assert(bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_cpu_matmul_matches_matvec(9999));
    assert(!bn_quant_format_cpu_matvec_uses_float_input(
        BN_GGUF_TENSOR_IQ3_S));
    assert(!bn_quant_format_cpu_matvec_uses_float_input(
        BN_GGUF_TENSOR_IQ4_NL));
    assert(bn_quant_format_cpu_matvec_uses_float_input(
        BN_GGUF_TENSOR_IQ4_XS));
    assert(bn_quant_format_cpu_matvec_uses_float_input(
        BN_GGUF_TENSOR_Q5_1));
    assert(!bn_quant_format_cpu_matvec_uses_float_input(
        BN_GGUF_TENSOR_TQ2_0));
    assert(!bn_quant_format_cpu_matvec_uses_float_input(
        BN_GGUF_TENSOR_MXFP4));
#else
    assert(!bn_quant_format_cpu_matmul_matches_matvec(BN_GGUF_TENSOR_Q4_K));
#endif

#if defined(BN_FORCE_SCALAR) || \
    (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    const int q4_dot_default = 1;
#else
    const int q4_dot_default = 0;
#endif
    const int q4k_dot_default = q4_dot_default;
#if defined(BN_FORCE_SCALAR) || \
    (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    const int q6_dot_default = 1;
#else
    const int q6_dot_default = 0;
#endif

    unsetenv("BN_AVX512_KQUANT_VNNI");
    unsetenv("BN_AVX512_Q5K_VNNI");
    assert(!bn_quant_policy_avx512_q5k_vnni_enabled(TEST_QUANT_POLICY(), 1024));
    assert(bn_quant_policy_avx512_q5k_vnni_enabled(TEST_QUANT_POLICY(), 4096));
    setenv("BN_AVX512_KQUANT_VNNI", "0", 1);
    assert(!bn_quant_policy_avx512_q5k_vnni_enabled(TEST_QUANT_POLICY(), 4096));
    setenv("BN_AVX512_KQUANT_VNNI", "1", 1);
    assert(bn_quant_policy_avx512_q5k_vnni_enabled(TEST_QUANT_POLICY(), 1024));
    unsetenv("BN_AVX512_KQUANT_VNNI");
    setenv("BN_AVX512_Q5K_VNNI", "1", 1);
    assert(bn_quant_policy_avx512_q5k_vnni_enabled(TEST_QUANT_POLICY(), 1024));
    unsetenv("BN_AVX512_Q5K_VNNI");

    BnMatvecTask tasks[2];
    memset(tasks, 0, sizeof(tasks));
    unsetenv("BN_AVX2_KQUANT_FLOAT");
    assert(!bn_quant_policy_avx2_kquant_float_for_tasks(TEST_QUANT_POLICY(), tasks, 2));
    tasks[1].flags = BN_MATVEC_TASK_FORCE_FLOAT_KQUANT;
    assert(bn_quant_policy_avx2_kquant_float_for_tasks(TEST_QUANT_POLICY(), tasks, 2));
    tasks[1].flags = 0;
    setenv("BN_AVX2_KQUANT_FLOAT", "1", 1);
    assert(bn_quant_policy_avx2_kquant_float_for_tasks(TEST_QUANT_POLICY(), tasks, 2));
    assert(bn_quant_policy_avx2_q5k_float_matvec_enabled(TEST_QUANT_POLICY()));
    setenv("BN_AVX2_KQUANT_FLOAT", "0", 1);
    assert(!bn_quant_policy_avx2_kquant_float_for_tasks(TEST_QUANT_POLICY(), tasks, 2));
    assert(!bn_quant_policy_avx2_q5k_float_matvec_enabled(TEST_QUANT_POLICY()));
    unsetenv("BN_AVX2_KQUANT_FLOAT");
    assert(!bn_quant_policy_avx2_q5k_float_matvec_enabled(TEST_QUANT_POLICY()));

    unsetenv("BN_CPU_LLAMA_DOT");
    unsetenv("BN_CPU_LLAMA_Q4_DOT");
    unsetenv("BN_CPU_LLAMA_Q6_DOT");
    unsetenv("BN_CPU_REFERENCE_DOT");
    unsetenv("BN_CPU_DISABLE_Q4_DOT");
    unsetenv("BN_CPU_BLOCK_QUANT_FLOAT");
    unsetenv("BN_CPU_REFERENCE_BLOCK_QUANT_DOT");
    unsetenv("BN_CPU_REFERENCE_Q4_DOT");
    unsetenv("BN_CPU_REFERENCE_KQUANT_DOT");
    unsetenv("BN_CPU_REFERENCE_Q6_DOT");
    unsetenv("BN_CPU_DISABLE_Q6_DOT");
    unsetenv("BN_CPU_KQUANT_FLOAT");
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0) == q4_dot_default);
    assert(bn_quant_policy_reference_q4k_dot_enabled(
               TEST_QUANT_POLICY(), 0) == q4k_dot_default);
    assert(bn_quant_policy_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(), BN_MATVEC_TASK_REFERENCE_DOT));
    assert(bn_quant_policy_reference_q4k_dot_enabled(
        TEST_QUANT_POLICY(), BN_MATVEC_TASK_REFERENCE_DOT));
    setenv("BN_CPU_DISABLE_Q4_DOT", "1", 1);
    assert(!bn_quant_policy_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(), BN_MATVEC_TASK_REFERENCE_DOT));
    assert(!bn_quant_policy_reference_q4k_dot_enabled(
        TEST_QUANT_POLICY(), BN_MATVEC_TASK_REFERENCE_DOT));
    unsetenv("BN_CPU_DISABLE_Q4_DOT");
    assert(!bn_quant_policy_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(),
        BN_MATVEC_TASK_REFERENCE_DOT | BN_MATVEC_TASK_NATIVE_QUANT));
    assert(!bn_quant_policy_reference_q4k_dot_enabled(
        TEST_QUANT_POLICY(), BN_MATVEC_TASK_NATIVE_QUANT));
    BnBlockQ4K policy_q4k_data[8] = {0};
    BnQWeight policy_q4k = {
        policy_q4k_data, BN_GGUF_TENSOR_Q4_K, 8, BN_QK_K, 1.0f
    };
    assert(bn_quant_matvec_uses_prepared_weight(
               &policy_q4k, 0, NULL) == !q4k_dot_default);
    assert(bn_quant_matvec_uses_prepared_weight(
        &policy_q4k, BN_MATVEC_TASK_NATIVE_QUANT, NULL));
    assert(!bn_quant_matvec_uses_prepared_weight(
        &policy_q4k, BN_MATVEC_TASK_FORCE_FLOAT_KQUANT, NULL));
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0) == q6_dot_default);
    setenv("BN_CPU_DISABLE_Q6_DOT", "1", 1);
    assert(!bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_DISABLE_Q6_DOT");
    setenv("BN_CPU_KQUANT_FLOAT", "1", 1);
    assert(!bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_KQUANT_FLOAT");
    setenv("BN_CPU_REFERENCE_BLOCK_QUANT_DOT", "1", 1);
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0));
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_REFERENCE_BLOCK_QUANT_DOT");
    setenv("BN_CPU_REFERENCE_Q4_DOT", "1", 1);
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0));
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_REFERENCE_Q4_DOT");
    setenv("BN_CPU_REFERENCE_KQUANT_DOT", "1", 1);
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0) == q4_dot_default);
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_REFERENCE_KQUANT_DOT");
    setenv("BN_CPU_REFERENCE_Q6_DOT", "1", 1);
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0) == q4_dot_default);
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_REFERENCE_Q6_DOT");
    setenv("BN_CPU_LLAMA_DOT", "1", 1);
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0));
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_LLAMA_DOT");
    setenv("BN_CPU_LLAMA_Q4_DOT", "1", 1);
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0));
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_LLAMA_Q4_DOT");
    setenv("BN_CPU_LLAMA_Q6_DOT", "1", 1);
    assert(bn_quant_policy_reference_q4_dot_enabled(TEST_QUANT_POLICY(), 0) == q4_dot_default);
    assert(bn_quant_policy_reference_q6_dot_enabled(TEST_QUANT_POLICY(), 0));
    unsetenv("BN_CPU_LLAMA_Q6_DOT");

    memset(tasks, 0, sizeof(tasks));
    assert(bn_quant_policy_batch_reference_q4_dot_enabled(
               TEST_QUANT_POLICY(), tasks, 2) ==
           q4_dot_default);
    setenv("BN_CPU_REFERENCE_DOT", "1", 1);
    assert(bn_quant_policy_batch_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(), tasks, 2));
    unsetenv("BN_CPU_REFERENCE_DOT");
    tasks[0].flags = BN_MATVEC_TASK_REFERENCE_DOT;
    assert(bn_quant_policy_batch_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(), tasks, 2));
    tasks[1].flags = BN_MATVEC_TASK_NATIVE_QUANT;
    assert(!bn_quant_policy_batch_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(), tasks, 2));
    tasks[0].flags = 0;
    tasks[1].flags = 0;

    unsetenv("BN_WASM_BLOCK_QUANT_CANONICAL4");
    unsetenv("BN_WASM_Q4_CANONICAL4");
    assert(!bn_quant_policy_wasm_q4_canonical4_enabled(TEST_QUANT_POLICY()));
    setenv("BN_WASM_BLOCK_QUANT_CANONICAL4", "1", 1);
    assert(bn_quant_policy_wasm_q4_canonical4_enabled(TEST_QUANT_POLICY()));
    unsetenv("BN_WASM_BLOCK_QUANT_CANONICAL4");
    setenv("BN_WASM_Q4_CANONICAL4", "1", 1);
    assert(bn_quant_policy_wasm_q4_canonical4_enabled(TEST_QUANT_POLICY()));
    unsetenv("BN_WASM_Q4_CANONICAL4");

    unsetenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH");
    unsetenv("BN_DISABLE_Q8_0_MATMUL_BATCH");
    assert(bn_quant_policy_native_matmul_batch_enabled(TEST_QUANT_POLICY()));
    setenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH", "1", 1);
    assert(!bn_quant_policy_native_matmul_batch_enabled(TEST_QUANT_POLICY()));
    unsetenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH");
    setenv("BN_DISABLE_Q8_0_MATMUL_BATCH", "1", 1);
    assert(!bn_quant_policy_native_matmul_batch_enabled(TEST_QUANT_POLICY()));
    unsetenv("BN_DISABLE_Q8_0_MATMUL_BATCH");

    assert(bn_quant_format_is_q4k(BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_is_q4k(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_is_q5k(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_is_q5k(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_is_q6k(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_is_q6k(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_is_q8k(BN_GGUF_TENSOR_Q8_K));
    assert(!bn_quant_format_is_q8k(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_is_q8_0(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_is_q8_0(BN_GGUF_TENSOR_Q8_K));
    assert(bn_quant_format_is_q5_0(BN_GGUF_TENSOR_Q5_0));
    assert(!bn_quant_format_is_q5_0(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_supports_f16_float_cache_matvec(
        BN_GGUF_TENSOR_Q3_K));
    assert(bn_quant_format_supports_f16_float_cache_matvec(
        BN_GGUF_TENSOR_IQ3_XXS));
    assert(bn_quant_format_supports_f16_float_cache_matvec(
        BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_supports_f16_float_cache_matvec(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_is_bf16(BN_GGUF_TENSOR_BF16));
    assert(!bn_quant_format_is_bf16(BN_GGUF_TENSOR_F16));
    assert(bn_quant_format_is_q3k(BN_GGUF_TENSOR_Q3_K));
    assert(!bn_quant_format_is_q3k(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_prepared_kquant_blocks_per_row(BN_QK_K * 2) == 2);
    assert(bn_backend_quant_prepared_kquant_blocks_per_row(BN_QK_K - 1) == 0);
    assert(bn_backend_quant_prepared_kquant_blocks_per_row(0) == 0);
    assert(bn_backend_quant_prepared_kquant_block_sums_per_row(2) == 32);
    assert(bn_backend_quant_prepared_kquant_block_sums_per_row(0) == 0);

    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_F32));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_F16));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_BF16));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q3_K));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_moe_routed_e8m0(BN_GGUF_TENSOR_MXFP4,
        BN_GGUF_TENSOR_MXFP4, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_supports_moe_routed_e8m0(BN_GGUF_TENSOR_Q5_K,
        BN_GGUF_TENSOR_MXFP4, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_supports_moe_routed_e8m0(BN_GGUF_TENSOR_MXFP4,
        BN_GGUF_TENSOR_MXFP4, BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_moe_routed_affine_mmvq(BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_1));
    assert(!bn_quant_format_supports_moe_routed_affine_mmvq(BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_1));
    assert(!bn_quant_format_supports_moe_routed_affine_mmvq(BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_1));
    assert(!bn_quant_format_supports_moe_routed_affine_mmvq(BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_moe_routed_ordered_kquant(BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_supports_moe_routed_ordered_kquant(BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_supports_moe_routed_ordered_kquant(BN_GGUF_TENSOR_MXFP4, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_supports_moe_routed_ordered_kquant(BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q8_K));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_IQ3_XXS));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_IQ4_XS));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q5_1));
    assert(bn_quant_format_supports_packed_codebook_matvec(
        BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_MXFP4));
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_MXFP4,
        BN_QUANT_CAP_GPU_MMVQ_BLOCK32_E8M0 | BN_QUANT_CAP_GPU_MMQ_BLOCK32_FP4));
    assert(!bn_quant_format_has_cap(BN_GGUF_TENSOR_MXFP4,
        BN_QUANT_CAP_GPU_MMQ_F32_SCALE));
    assert(!bn_quant_format_eager_aux_cache_supported(BN_GGUF_TENSOR_MXFP4));
    assert(bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_BF16));
    assert(bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_Q3_K));
    assert(bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_eager_aux_cache_supported(
        BN_GGUF_TENSOR_IQ4_XS));
    assert(bn_quant_format_avoids_quant_matmul_on_f16_input(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_avoids_quant_matmul_on_f16_input(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_supports_requested_quant_matmul(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_supports_requested_quant_matmul(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_supports_requested_quant_matmul(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_supports_direct_native_quant_matvec(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_supports_direct_native_quant_matvec(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_supports_direct_native_quant_matvec(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_gpu_matvec_native_quant_flag(
               BN_GGUF_TENSOR_Q4_0, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT);
    assert(bn_quant_format_gpu_matvec_native_quant_flag(
               BN_GGUF_TENSOR_Q4_0, 0) == 0);
    assert(bn_quant_format_gpu_matvec_block_q8_activation_flag(
               BN_GGUF_TENSOR_Q8_0, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT);
    assert(bn_quant_format_gpu_matvec_block_q8_activation_flag(
               BN_GGUF_TENSOR_Q4_0, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_BLOCK_Q8_ACTIVATION);
    assert(bn_quant_format_gpu_matvec_reference_kquant_flag(
               BN_GGUF_TENSOR_Q4_K, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_REFERENCE_KQUANT);
    assert(bn_quant_format_gpu_matvec_reference_kquant_flag(
               BN_GGUF_TENSOR_Q5_K, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_REFERENCE_KQUANT);
    assert(bn_quant_format_gpu_matvec_reference_kquant_flag(
               BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_quant_format_gpu_matvec_native_quant_flag(
               BN_GGUF_TENSOR_Q6_K, 1) == 0);
    assert(bn_quant_format_supports_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_supports_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_supports_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_supports_reference_prepared_accumulation(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_supports_reference_prepared_accumulation(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_prefers_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q4_K, 2048));
    assert(bn_quant_format_prefers_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q4_K, 4096));
    assert(bn_quant_format_prefers_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q6_K, 4096));
    assert(bn_quant_format_prefers_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q5_K, 4096));
    assert(bn_quant_format_prefers_tall_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q6_K, 151936, 2048));
    assert(!bn_quant_format_prefers_tall_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q6_K, 4096, 2048));
    assert(!bn_quant_format_prefers_tall_specialized_native_quant_matvec(
        BN_GGUF_TENSOR_Q4_K, 151936, 2048));
    assert(bn_quant_format_supports_native_quant_split(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_supports_native_quant_split(
        BN_GGUF_TENSOR_Q6_K));

    assert(strcmp(bn_quant_format_gpu_shader_name(BN_GGUF_TENSOR_Q4_0),
                  "q4") == 0);
    assert(strcmp(bn_quant_format_gpu_shader_name(BN_GGUF_TENSOR_Q6_K),
                  "q6k") == 0);
    assert(bn_quant_format_gpu_shader_name(BN_GGUF_TENSOR_MXFP4) == NULL);
    assert(bn_quant_format_can_gpu_native(BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_can_gpu_native(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_can_gpu_repack(BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_can_gpu_repack(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_gpu_uses_repacked_layout(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_supports_repacked_bias(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_dispatch_tile_rows(BN_GGUF_TENSOR_Q4_0) == 8u);
    assert(!bn_quant_format_gpu_uses_repacked_layout(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_supports_repacked_bias(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_gpu_dispatch_tile_rows(BN_GGUF_TENSOR_Q8_0) == 32u);
    assert(bn_quant_format_gpu_requires_reference_silu(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_requires_reference_silu(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_gpu_fused_gateup_requires_backend_opt_in(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_gpu_fused_gateup_requires_backend_opt_in(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_logits_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_logits_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_all_f16_cache_supported(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_all_f16_cache_supported(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_moe_all_f16_cache_supported(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_moe_all_f16_cache_supported(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_moe_all_f16_cache_supported(BN_GGUF_TENSOR_Q5_0));
    assert(!bn_quant_format_has_embedded_tensor_scale(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_allows_stacked_layout(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_embedded_tensor_scale_offset(BN_GGUF_TENSOR_Q4_0,
                                                 4, 32) == 0);
    assert(bn_quant_format_has_embedded_tensor_scale(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_allows_stacked_layout(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_embedded_tensor_scale_offset(BN_GGUF_TENSOR_I2_S,
                                                 4, 32) == 32);
    assert(bn_quant_format_moe_down_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_moe_down_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_down_cublas_cache_supported(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_moe_down_cublas_cache_supported(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_moe_down_small_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_moe_down_small_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_moe_quant_only_after_cache(BN_GGUF_TENSOR_Q8_0, 1));
    assert(bn_quant_format_moe_quant_only_after_cache(BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_quant_format_moe_quant_only_after_cache(BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_quant_format_supports_lazy_moe_aux_cache(BN_GGUF_TENSOR_Q3_K));
    assert(bn_quant_format_supports_lazy_moe_aux_cache(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_supports_lazy_moe_aux_cache(BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_supports_lazy_moe_aux_cache(BN_GGUF_TENSOR_Q5_0));
    float iq_tmp[BN_QK_K];
    BnBlockIQ3XXS iq3 = {0};
    BnBlockIQ4XS iq4 = {0};
    assert(bn_quant_dequant_lazy_aux_cache_block(
        BN_GGUF_TENSOR_IQ3_XXS, &iq3, 0, iq_tmp) == 0);
    assert(bn_quant_dequant_lazy_aux_cache_block(
        BN_GGUF_TENSOR_IQ4_XS, &iq4, 0, iq_tmp) == 0);
    assert(bn_quant_dequant_lazy_aux_cache_block(
        BN_GGUF_TENSOR_F32, &iq4, 0, iq_tmp) == -1);
    assert(bn_quant_format_moe_prefers_quant_only(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_moe_prefers_quant_only(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_BF16));
    assert(bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_aux_cache_can_use_f16(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_aux_cache_can_use_f16(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_aux_cache_uses_f32(BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_quant_format_aux_cache_uses_f32(BN_GGUF_TENSOR_Q6_K, 1));
    assert(!bn_quant_format_aux_cache_uses_f32(BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_aux_cache_prefers_large_budget(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_aux_cache_prefers_large_budget(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_aux_cache_prefers_large_budget(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_aux_cache_prefers_large_budget(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_uses_f16_logits_path(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_uses_f16_logits_path(BN_GGUF_TENSOR_F32));
    assert(!bn_quant_format_uses_f16_logits_path(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_logits_i8_cache(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_supports_logits_i8_cache(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_tied_logits_uses_f16_path(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_tied_logits_uses_f16_path(BN_GGUF_TENSOR_BF16));
    assert(bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_BF16));
    assert(bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_F32));
    assert(bn_quant_format_supports_shared_gateup_batch(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_supports_shared_gateup_batch(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_supports_shared_gateup_batch(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_supports_shared_gateup_batch(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q8_K));
    assert(bn_quant_format_supports_moe_routed_kquant_gateup(BN_GGUF_TENSOR_Q4_K,
                                                  BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_supports_moe_routed_kquant_gateup(BN_GGUF_TENSOR_Q4_K,
                                                   BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_supports_moe_routed_midbit_kquant_gateup(
        BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_supports_moe_routed_midbit_kquant_gateup(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_supports_moe_direct_routed_down(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_supports_moe_direct_routed_down(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_supports_moe_direct_routed_down(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_supports_moe_direct_routed_down(
        BN_GGUF_TENSOR_Q8_K));
    assert(bn_quant_format_supports_moe_asymmetric_kquant_down_route(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_quant_format_supports_moe_asymmetric_kquant_down_route(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_supports_moe_asymmetric_kquant_down_route(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, 1));
    assert(!bn_quant_format_supports_moe_asymmetric_kquant_down_route(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K, 1));
    assert(bn_quant_format_supports_cpu_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_supports_cpu_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_supports_moe_native_quant_route(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_supports_moe_native_quant_route(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_K));

    int n_without_f32 = bn_quant_format_gpu_shader_type_count(0);
    int n_with_f32 = bn_quant_format_gpu_shader_type_count(1);
    assert(n_with_f32 == n_without_f32 + 1);
    assert(bn_quant_format_gpu_shader_type_at(6, 1) == BN_GGUF_TENSOR_F32);
    assert(bn_quant_format_gpu_shader_type_at(6, 0) == BN_GGUF_TENSOR_F16);
    assert(bn_quant_format_gpu_shader_type_at(-1, 1) == -1);
    assert(bn_quant_format_gpu_shader_type_at(n_with_f32, 1) == -1);
    for (int i = 0; i < n_without_f32; i++)
        assert(bn_quant_format_gpu_shader_name(
                   bn_quant_format_gpu_shader_type_at(i, 0)) != NULL);
    for (int i = 0; i < n_with_f32; i++)
        assert(bn_quant_format_gpu_shader_name(
                   bn_quant_format_gpu_shader_type_at(i, 1)) != NULL);

    printf("PASSED\n");
}

static float ref_dot_f32(const float *w, const float *x, int cols) {
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) sum += w[i] * x[i];
    return sum;
}

static float ref_dot_f16(const uint16_t *w, const float *x, int cols) {
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) sum += bn_fp16_to_fp32(w[i]) * x[i];
    return sum;
}

static void test_fp16_conversion(void) {
    printf("test_fp16_conversion... ");

    assert(bn_fp32_to_fp16(1.0f) == 0x3C00);
    assert(bn_fp32_to_fp16(-0.0f) == 0x8000);
    assert(bn_fp32_to_fp16(0x1.0p-14f) == 0x0400);
    assert(bn_fp32_to_fp16(0x1.0p-24f) == 0x0001);
    assert(bn_fp32_to_fp16(65504.0f) == 0x7BFF);
    assert(bn_fp32_to_fp16(INFINITY) == 0x7C00);
    assert(bn_fp32_to_fp16(NAN) == 0x7E00);

    // Halfway between 0x3C01 and 0x3C02 rounds to the even mantissa.
    assert(bn_fp32_to_fp16(0x1.006p+0f) == 0x3C02);
    assert(fabsf(bn_fp16_to_fp32(0x0001) - 0x1.0p-24f) < 1e-12f);

    printf("PASSED\n");
}

static void test_iq2xxs_dequant_payload(void) {
    BnBlockIQ2XXS block = {0};
    block.d = bn_fp32_to_fp16(1.0f);
    for (size_t i = 0; i < sizeof(block.qs) / sizeof(block.qs[0]); i++)
        block.qs[i] = (uint16_t)(i * 997u);
    float actual[BN_QK_K];
    float again[BN_QK_K];
    bn_quant_dequant_iq2xxs(&block, actual);
    bn_quant_dequant_iq2xxs(&block, again);
    for (int i = 0; i < BN_QK_K; i++)
        assert(actual[i] == again[i]);
}

static void test_iq3s_block_layout_and_matvec(void) {
    printf("test_iq3s_block_layout_and_matvec... ");
    assert(sizeof(BnBlockIQ3S) == 110);
    assert(bn_quant_format_data_size(BN_GGUF_TENSOR_IQ3_S, 2,
                                     BN_QK_K) ==
           2 * sizeof(BnBlockIQ3S));

    BnBlockIQ3S blocks[2] = {0};
    float x[BN_QK_K];
    float dequant[BN_QK_K];
    float expected[2] = {0};
    float scalar[2] = {0};
    float actual[2] = {0};
    int8_t x_q[BN_QK_K];
    for (int i = 0; i < BN_QK_K; i++)
        x[i] = (float)((i % 17) - 8) / 8.0f;
    for (int row = 0; row < 2; row++) {
        blocks[row].d = bn_fp32_to_fp16(0.015625f * (float)(row + 1));
        for (int i = 0; i < BN_QK_K / 4; i++)
            blocks[row].qs[i] = (uint8_t)((i * 7 + row * 19) & 0xff);
        for (int i = 0; i < BN_QK_K / 32; i++)
            blocks[row].qh[i] = (uint8_t)(0x5a ^ (i * 13 + row));
        for (int i = 0; i < BN_QK_K / 8; i++)
            blocks[row].signs[i] = (uint8_t)(0xa5 ^ (i * 9 + row));
        for (int i = 0; i < BN_QK_K / 64; i++)
            blocks[row].scales[i] = (uint8_t)(0x31 + i * 0x22 + row);
        bn_quant_dequant_iq3s(&blocks[row], dequant);
        for (int i = 0; i < BN_QK_K; i++)
            expected[row] += dequant[i] * x[i];
    }

    BnQWeight weight = {
        blocks, BN_GGUF_TENSOR_IQ3_S, 2, BN_QK_K, 1.0f
    };
    BnIQ3SCtx scalar_ctx = { scalar, &weight, x };
    bn_quant_iq3s_scalar_range(&scalar_ctx, 0, 2);
    bn_quant_matvec(actual, &weight, x, x_q, NULL);
    for (int row = 0; row < 2; row++) {
        assert(fabsf(scalar[row] - expected[row]) < 1e-3f);
#if defined(__AVX2__)
        float xd[1];
        int16_t sums[16];
        bn_quant_x_to_q8k(x, x_q, xd, sums, BN_QK_K);
        bn_quant_dequant_iq3s(&blocks[row], dequant);
        float quant_expected = 0.0f;
        for (int i = 0; i < BN_QK_K; i++)
            quant_expected += dequant[i] * ((float)x_q[i] * xd[0]);
        assert(fabsf(actual[row] - quant_expected) < 1e-3f);
#else
        assert(fabsf(actual[row] - expected[row]) < 1e-3f);
#endif
    }

    float batch_x[2 * BN_QK_K];
    float batch_matvec[4] = {0};
    float batch_matmul[4] = {0};
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < BN_QK_K; i++)
            batch_x[t * BN_QK_K + i] = x[i] * (float)(t + 1);
        bn_quant_matvec(batch_matvec + t * 2, &weight,
                        batch_x + t * BN_QK_K, x_q, NULL);
    }
    bn_quant_matmul_prepared(batch_matmul, &weight, NULL, batch_x, 2,
                             x_q, NULL);
    assert(memcmp(batch_matmul, batch_matvec, sizeof(batch_matmul)) == 0);
    printf("PASSED\n");
}

#if defined(__AVX2__)
static void check_quant_batch_matches_matvec(const BnQWeight *w,
                                             const float *x, int8_t *scratch);

static void test_iq3s_panel_matmul(void) {
    printf("test_iq3s_panel_matmul... ");
    enum { rows = 16, blocks_per_row = 3, cols = BN_QK_K * blocks_per_row,
           tokens = 9 };
    BnBlockIQ3S blocks[rows * blocks_per_row] = {0};
    float x[tokens * cols], got[tokens * rows];
    int8_t xq[cols];
    float xd[blocks_per_row], values[BN_QK_K];
    int16_t sums[blocks_per_row * 16];
    for (int b = 0; b < rows * blocks_per_row; b++) {
        blocks[b].d = bn_fp32_to_fp16(0.0013f * (b + 1));
        for (int i = 0; i < BN_QK_K / 4; i++)
            blocks[b].qs[i] = (uint8_t)(i * 37 + b * 19);
        for (int i = 0; i < BN_QK_K / 32; i++)
            blocks[b].qh[i] = (uint8_t)(0x5a ^ (i * 13 + b));
        for (int i = 0; i < BN_QK_K / 8; i++)
            blocks[b].signs[i] = (uint8_t)(0xa5 ^ (i * 9 + b));
        for (int i = 0; i < BN_QK_K / 64; i++)
            blocks[b].scales[i] = (uint8_t)(0x31 + i * 0x22 + b);
    }
    for (int i = 0; i < tokens * cols; i++)
        x[i] = sinf((float)i * 0.13f) * 1.7f;
    BnQWeight weight = {blocks, BN_GGUF_TENSOR_IQ3_S, rows, cols, 1.0f};
    check_quant_batch_matches_matvec(&weight, x, xq);
    bn_quant_matmul_prepared(got, &weight, NULL, x, tokens, xq, NULL);
    for (int t = 0; t < tokens; t++) {
        bn_quant_x_to_q8k(x + t * cols, xq, xd, sums, cols);
        for (int r = 0; r < rows; r++) {
            float expected = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) {
                BnBlockIQ3S unit = blocks[r * blocks_per_row + b];
                float d = bn_fp16_to_fp32(unit.d);
                unit.d = bn_fp32_to_fp16(1.0f);
                bn_quant_dequant_iq3s(&unit, values);
                int32_t dot = 0;
                for (int i = 0; i < BN_QK_K; i++)
                    dot += (int32_t)values[i] * xq[b * BN_QK_K + i];
                /* Panel arithmetic reduces each superblock in integer space. */
                expected = fmaf((float)dot, d * xd[b], expected);
            }
            assert(got[t * rows + r] == expected);
        }
    }
    printf("PASSED\n");
}
#endif

// --- Integration test: dispatch routing ---
// Verifies that bn_quant_matvec dispatches correctly for each format.

static void test_dispatch_routing(void) {
    printf("test_dispatch_routing... ");

    // TQ2_0: all +1, dot with all-ones = 256
    BnBlockTQ2 *tq2 = (BnBlockTQ2 *)calloc(1, sizeof(BnBlockTQ2));
    for (int i = 0; i < 64; i++) tq2->qs[i] = 0xAA;
    tq2->d = 0x3C00;
    BnQWeight W_tq2 = { tq2, BN_GGUF_TENSOR_TQ2_0, 1, 256, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;
    float out;
    int8_t x_q[256];

    bn_quant_matvec(&out, &W_tq2, x, x_q, NULL);
    assert(fabsf(out - 256.0f) < 1e-3f);

    // Q8_0: all qs=1, scale=1 → dot = 32
    BnBlockQ8_0 *q8 = (BnBlockQ8_0 *)calloc(1, sizeof(BnBlockQ8_0));
    q8->d = 0x3C00;
    for (int i = 0; i < 32; i++) q8->qs[i] = 1;
    BnQWeight W_q8 = { q8, BN_GGUF_TENSOR_Q8_0, 1, 32, 1.0f };

    float x32[32];
    for (int i = 0; i < 32; i++) x32[i] = 1.0f;
    int8_t x_q32[32];

    bn_quant_matvec(&out, &W_q8, x32, x_q32, NULL);
    assert(fabsf(out - 32.0f) < 0.1f);

    // Q5_0: all low bits zero, high bits set on second half -> [-16, 0]
    BnBlockQ5_0 *q5_0 = (BnBlockQ5_0 *)calloc(1, sizeof(BnBlockQ5_0));
    q5_0->d = 0x3C00;
    for (int i = 0; i < 4; i++) q5_0->qh[i] = 0;
    q5_0->qh[2] = 0xFF;
    q5_0->qh[3] = 0xFF;
    BnQWeight W_q5_0 = { q5_0, BN_GGUF_TENSOR_Q5_0, 1, 32, 1.0f };

    bn_quant_matvec(&out, &W_q5_0, x32, x_q32, NULL);
    assert(fabsf(out - (-256.0f)) < 0.1f);

    // Q5_1: q=0, scale=1, min=2 -> each dequantized value is 2.
    BnBlockQ5_1 *q5_1 = (BnBlockQ5_1 *)calloc(1, sizeof(BnBlockQ5_1));
    q5_1->d = 0x3C00;
    q5_1->m = 0x4000;
    BnQWeight W_q5_1 = { q5_1, BN_GGUF_TENSOR_Q5_1, 1, 32, 1.0f };

    bn_quant_matvec(&out, &W_q5_1, x32, x_q32, NULL);
    assert(fabsf(out - 64.0f) < 0.1f);

    // Q6_K: all scales=1, all ql/qh=0 → quant=-32, dot = 256*(-32)*1 = -8192
    BnBlockQ6K *q6k = (BnBlockQ6K *)calloc(1, sizeof(BnBlockQ6K));
    q6k->d = 0x3C00;
    for (int i = 0; i < 16; i++) q6k->scales[i] = 1;
    BnQWeight W_q6k = { q6k, BN_GGUF_TENSOR_Q6_K, 1, 256, 1.0f };

    bn_quant_matvec(&out, &W_q6k, x, x_q, NULL);
    assert(fabsf(out - (-8192.0f)) < 1.0f);

    // Q5_K: all scales=1, mins=0, q=1 → dot = 256
    BnBlockQ5K *q5k = (BnBlockQ5K *)calloc(1, sizeof(BnBlockQ5K));
    q5k->d = 0x3C00;
    q5k->dmin = 0;
    for (int i = 0; i < 4; i++) q5k->scales[i] = 1;
    for (int i = 8; i < 12; i++) q5k->scales[i] = 1;
    for (int i = 0; i < 128; i++) q5k->qs[i] = 0x11;
    BnQWeight W_q5k = { q5k, BN_GGUF_TENSOR_Q5_K, 1, 256, 1.0f };

    bn_quant_matvec(&out, &W_q5k, x, x_q, NULL);
    assert(fabsf(out - 256.0f) < 1.0f);

    free(tq2);
    free(q8);
    free(q5_0);
    free(q5_1);
    free(q6k);
    free(q5k);
    printf("PASSED\n");
}

#if defined(__AVX2__)
static float test_q51_dot_reference(const BnBlockQ5_1 *w, const float *x, int nb) {
    float lanes[8] = {0}, offset = 0.0f;
    for (int b = 0; b < nb; b++) {
        float max = 0.0f;
        for (int j = 0; j < 32; j++) max = fmaxf(max, fabsf(x[b * 32 + j]));
        float raw_d = max / 127.0f;
        float inv = max != 0.0f ? 127.0f / max : 0.0f;
        float d = bn_fp16_to_fp32(bn_fp32_to_fp16(raw_d));
        int q[32], sum = 0;
        for (int j = 0; j < 32; j++) {
            q[j] = (int)nearbyintf(x[b * 32 + j] * inv);
            sum += q[j];
        }
        float s = bn_fp16_to_fp32(bn_fp32_to_fp16(raw_d * (float)sum));
        offset = fmaf(bn_fp16_to_fp32(w[b].m), s, offset);
        for (int k = 0; k < 8; k++) {
            int dot = 0;
            for (int j = k * 4; j < k * 4 + 4; j++) {
                int v = ((w[b].qs[j % 16] >> (4 * (j / 16))) & 15) |
                    (((w[b].qh[j / 8] >> (j % 8)) & 1) << 4);
                dot += v * q[j];
            }
            lanes[k] = fmaf((float)dot, bn_fp16_to_fp32(w[b].d) * d, lanes[k]);
        }
    }
    float a = (lanes[0] + lanes[4]) + (lanes[2] + lanes[6]);
    float b = (lanes[1] + lanes[5]) + (lanes[3] + lanes[7]);
    return (a + b) + offset;
}
#endif

static void test_q51_half_metadata(void) {
    printf("test_q51_half_metadata... ");
#if defined(__AVX2__)
    enum { rows = 65536 };
    BnBlockQ5_1 *blocks = calloc(rows, sizeof(*blocks));
    float *out = malloc((rows + 2) * sizeof(float));
    assert(blocks && out);
    float x[32];
    for (int i = 0; i < 32; i++) x[i] = (float)(i % 7 - 2) * 0.25f + 1;
    for (int mode = 0; mode < 2; mode++) {
        for (int h = 0; h < rows; h++) {
            blocks[h].d = mode == 0 ? (uint16_t)h : 0x3c00;
            blocks[h].m = mode == 1 ? (uint16_t)h : 0x3c00;
            memset(blocks[h].qh, 0xa5, sizeof(blocks[h].qh));
            memset(blocks[h].qs, 0x73, sizeof(blocks[h].qs));
        }
        out[0] = out[rows + 1] = NAN;
        BnQWeight w = {blocks, BN_GGUF_TENSOR_Q5_1, rows, 32, 1};
        BnQ5_1Ctx ctx = {out + 1, &w, x};
        bn_quant_q5_1_avx2_range(&ctx, 0, rows);
        for (int h = 0; h < rows; h++) {
            float expected = test_q51_dot_reference(blocks + h, x, 1);
            if (isnan(expected)) assert(isnan(out[h + 1]));
            else assert(memcmp(out + h + 1, &expected, sizeof(float)) == 0);
        }
        assert(isnan(out[0]) && isnan(out[rows + 1]));
    }
    free(out);
    free(blocks);
#endif
    printf("PASSED\n");
}

static void test_q51_x86_exact(void) {
    printf("test_q51_x86_exact... ");
#if defined(__AVX2__)
    enum { rows = 137, max_nb = 257, tokens = 9, max_cols = max_nb * 32 };
    BnBlockQ5_1 *data = malloc((size_t)rows * max_nb * sizeof(*data));
    float *input = malloc(((size_t)tokens * max_cols + 1) * sizeof(float));
    assert(data && input);
    float *x = input + 1;
    float expected[tokens * rows], out[tokens * rows + 2], other[tokens * rows + 2];
    int8_t scratch[2 * max_cols];
    for (int i = 0; i < tokens * max_cols; i++) {
        /* Zero blocks, nearest-even ties, signed and small nonzero sums. */
        int block = i / 32, j = i % 32;
        x[i] = block % 7 == 0 ? 0.0f :
            (j == 0 ? 127.0f : (float)((i * 31) % 253 - 126) + 0.5f) *
            (block % 3 == 0 ? 0.000123f : 1.0f);
    }
    for (int i = 0; i < rows * max_nb; i++) {
        data[i].d = (uint16_t)(0x1800 + (i * 37) % 0x1800);
        data[i].m = (uint16_t)(0x1400 + (i * 53) % 0x1000 + (i % 2 ? 0x8000 : 0));
        for (int j = 0; j < 4; j++) data[i].qh[j] = (uint8_t)(i * 41 + j * 67);
        for (int j = 0; j < 16; j++) data[i].qs[j] = (uint8_t)(i * 31 + j * 17);
    }
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q5_1,
        BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL));
    /* Include the tiled scratch bound and the wide-input fallback. */
    const int counts[] = {1, 3, 20, 80, 256, 257};
    const char *old_env = getenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH");
    char *saved_env = old_env ? malloc(strlen(old_env) + 1) : NULL;
    if (old_env) { assert(saved_env); strcpy(saved_env, old_env); }
    assert(setenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH", "1", 1) == 0);
    BnThreadPool *disabled = bn_tp_create(3);
    assert(disabled && bn_tp_quant_policy(disabled)->disable_native_quant_matmul_batch);
    if (saved_env) {
        assert(setenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH", saved_env, 1) == 0);
        free(saved_env);
    } else {
        assert(unsetenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH") == 0);
    }
    BnThreadPool *pools[] = {NULL, bn_tp_create(0), bn_tp_create(3), disabled};
    assert(pools[1] && pools[2]);
    for (size_t k = 0; k < sizeof(counts) / sizeof(counts[0]); k++) {
        int nb = counts[k], cols = nb * 32;
        BnQWeight w = {data, BN_GGUF_TENSOR_Q5_1, rows, cols, 1.0f};
        for (int t = 0; t < tokens; t++) for (int r = 0; r < rows; r++)
            expected[t * rows + r] = test_q51_dot_reference(data + r * nb, x + t * cols, nb);
        for (int mode = 0; mode < 4; mode++) {
            for (int i = 0; i < tokens * rows + 2; i++) out[i] = other[i] = NAN;
            bn_quant_matvec(out + 1, &w, x, scratch, pools[mode]);
            assert(memcmp(out + 1, expected, rows * sizeof(float)) == 0);
            assert(isnan(out[0]) && isnan(out[rows + 1]));
            BnMatvecTask batch[] = {{out + 1, &w, NULL, 0}, {other + 1, &w, NULL, 0}};
            bn_quant_matvec_batch(batch, 2, x, scratch, pools[mode]);
            assert(memcmp(out + 1, expected, rows * sizeof(float)) == 0);
            assert(memcmp(other + 1, expected, rows * sizeof(float)) == 0);
            BnMatvecMultiTask multi[] = {{out + 1, &w, x, NULL}, {other + 1, &w, x + cols, NULL}};
            bn_quant_matvec_multi(multi, 2, scratch, pools[mode]);
            assert(memcmp(out + 1, expected, rows * sizeof(float)) == 0);
            assert(memcmp(other + 1, expected + rows, rows * sizeof(float)) == 0);
            bn_quant_matmul(out + 1, &w, x, tokens, scratch, pools[mode]);
            assert(memcmp(out + 1, expected, tokens * rows * sizeof(float)) == 0);
            assert(isnan(out[0]) && isnan(out[tokens * rows + 1]));
            float *outputs[] = {out + 1, other + 1};
            const BnQWeight *weights[] = {&w, &w};
            bn_quant_matmul_prepared_multi_gemv(outputs, weights, NULL, 2,
                x, tokens, scratch, pools[mode]);
            assert(memcmp(out + 1, expected, tokens * rows * sizeof(float)) == 0);
            assert(memcmp(other + 1, expected, tokens * rows * sizeof(float)) == 0);
            assert(isnan(other[0]) && isnan(other[tokens * rows + 1]));
        }
        for (int i = 0; i < tokens * rows + 2; i++) out[i] = NAN;
        BnQ5_1Ctx ctx = {out + 1, &w, x};
        bn_quant_q5_1_avx2_range(&ctx, 1, rows - 1);
        assert(memcmp(out + 2, expected + 1, (rows - 2) * sizeof(float)) == 0);
        assert(isnan(out[0]) && isnan(out[1]) && isnan(out[rows]));
        for (int i = 0; i < tokens * rows + 2; i++) out[i] = NAN;
        BnKQuantFloatMatmulCtx matctx = {out + 1, &w, x, tokens, cols};
        bn_quant_q5_1_avx2_matmul_range(&matctx, 1, 1);
        matctx.n_tokens = 0;
        bn_quant_q5_1_avx2_matmul_range(&matctx, 0, rows);
        for (int i = 0; i < tokens * rows + 2; i++) assert(isnan(out[i]));
        matctx.n_tokens = tokens;
        bn_quant_q5_1_avx2_matmul_range(&matctx, 1, rows - 1);
        for (int t = 0; t < tokens; t++) {
            assert(isnan(out[1 + t * rows]));
            assert(isnan(out[(t + 1) * rows]));
            assert(memcmp(out + 2 + t * rows, expected + 1 + t * rows,
                          (rows - 2) * sizeof(float)) == 0);
        }
        assert(isnan(out[0]) && isnan(out[tokens * rows + 1]));
    }
    bn_tp_free(pools[1]); bn_tp_free(pools[2]); bn_tp_free(disabled);
    free(data); free(input);
#endif
    printf("PASSED\n");
}

static void test_logits_refine_rows(void) {
    printf("test_logits_refine_rows... ");

    float x32[32];
    for (int i = 0; i < 32; i++)
        x32[i] = (float)((i % 7) - 3);
    int8_t x_q32[32];
    float x_scales[1];
    bn_quant_x_to_q8_blocks(x32, x_q32, x_scales, 32);

    BnBlockQ8_0 q8[2];
    memset(q8, 0, sizeof(q8));
    for (int row = 0; row < 2; row++) {
        q8[row].d = bn_fp32_to_fp16(row == 0 ? 1.0f : 0.5f);
        for (int i = 0; i < 32; i++)
            q8[row].qs[i] = (int8_t)(row == 0 ? 1 : -2);
    }
    BnQWeight W_q8 = { q8, BN_GGUF_TENSOR_Q8_0, 2, 32, 1.0f };
    float row;
    assert(bn_quant_q8_logits_refine_row(&W_q8, x_q32, x_scales, 0,
                                         &row) == 0);
    float ref = 0.0f;
    for (int i = 0; i < 32; i++)
        ref += (float)x_q32[i] * x_scales[0];
    assert(fabsf(row - ref) < 1e-4f);
    assert(bn_quant_q8_logits_refine_row(&W_q8, x_q32, x_scales, 2,
                                         &row) == -1);

    float x256[256];
    for (int i = 0; i < 256; i++)
        x256[i] = 1.0f;
    BnBlockQ6K q6[2];
    memset(q6, 0, sizeof(q6));
    for (int row_i = 0; row_i < 2; row_i++) {
        q6[row_i].d = bn_fp32_to_fp16(1.0f);
        for (int i = 0; i < 16; i++)
            q6[row_i].scales[i] = (int8_t)(row_i + 1);
    }
    BnQWeight W_q6 = { q6, BN_GGUF_TENSOR_Q6_K, 2, 256, 1.0f };
    assert(bn_quant_q6_logits_refine_row(&W_q6, x256, 0, &row) == 0);
    assert(fabsf(row - (-8192.0f)) < 1.0f);
    assert(bn_quant_q6_logits_refine_row(&W_q6, x256, 1, &row) == 0);
    assert(fabsf(row - (-16384.0f)) < 1.0f);
    assert(bn_quant_q6_logits_refine_row(&W_q6, x256, -1, &row) == -1);

    int8_t x_q8k[256];
    float x_d[1];
    int16_t x_bsums[16];
    bn_quant_x_to_q8k_scalar(x256, x_q8k, x_d, x_bsums, 256);
    assert(bn_quant_q6_logits_refine_q8k_row(
               &W_q6, x_q8k, x_d, x_bsums, 0, &row) == 0);
    assert(fabsf(row - (-8192.0f)) < 1.0f);
    assert(bn_quant_q6_logits_refine_q8k_row(
               &W_q6, x_q8k, x_d, x_bsums, 1, &row) == 0);
    assert(fabsf(row - (-16384.0f)) < 1.0f);
    assert(bn_quant_q6_logits_refine_q8k_row(
               &W_q6, x_q8k, x_d, x_bsums, 2, &row) == -1);

    printf("PASSED\n");
}

// --- Integration test: batch matvec ---

static void test_matvec_batch(void) {
    printf("test_matvec_batch... ");

    BnBlockTQ2 *blocks1 = (BnBlockTQ2 *)calloc(2, sizeof(BnBlockTQ2));
    BnBlockTQ2 *blocks2 = (BnBlockTQ2 *)calloc(2, sizeof(BnBlockTQ2));

    // Matrix 1: all +1
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < 64; i++) blocks1[r].qs[i] = 0xAA;
        blocks1[r].d = 0x3C00;
    }

    // Matrix 2: all -1
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < 64; i++) blocks2[r].qs[i] = 0x00;
        blocks2[r].d = 0x3C00;
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_TQ2_0, 2, 256, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_TQ2_0, 2, 256, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    // Reference: individual calls
    float ref1[2], ref2[2];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    // Batch call
    float out1[2], out2[2];
    int8_t x_q[256];
    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < 2; i++) {
        assert(fabsf(out1[i] - ref1[i]) < 1e-3f);
        assert(fabsf(out2[i] - ref2[i]) < 1e-3f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_q4_large_matvec_batch(void) {
    printf("test_q4_large_matvec_batch... ");

    enum { N_TASKS = 6, ROWS = 7, COLS = 64 };
    const int n_bpr = COLS / 32;
    BnBlockQ4_0 *blocks = (BnBlockQ4_0 *)calloc(
        (size_t)N_TASKS * ROWS * n_bpr, sizeof(BnBlockQ4_0));
    assert(blocks != NULL);

    BnQWeight weights[N_TASKS];
    float refs[N_TASKS][ROWS];
    float outs[N_TASKS][ROWS];
    BnMatvecTask tasks[N_TASKS];
    float x[COLS];
    int8_t x_q[COLS];

    for (int i = 0; i < COLS; i++)
        x[i] = 0.04f * (float)((i * 7) % 23) - 0.44f;
    for (int t = 0; t < N_TASKS; t++) {
        BnBlockQ4_0 *task_blocks =
            blocks + (size_t)t * ROWS * n_bpr;
        for (int r = 0; r < ROWS; r++) {
            for (int b = 0; b < n_bpr; b++) {
                BnBlockQ4_0 *blk = &task_blocks[r * n_bpr + b];
                blk->d = bn_fp32_to_fp16(0.25f + 0.125f * (float)t);
                for (int i = 0; i < 16; i++)
                    blk->qs[i] = (uint8_t)(t * 29 + r * 17 + b * 7 + i * 5);
            }
        }
        weights[t] = (BnQWeight){
            task_blocks, BN_GGUF_TENSOR_Q4_0, ROWS, COLS, 1.0f };
        bn_quant_matvec(refs[t], &weights[t], x, x_q, NULL);
        tasks[t] = (BnMatvecTask){ outs[t], &weights[t], NULL, 0 };
    }

    bn_quant_matvec_batch(tasks, N_TASKS, x, x_q, NULL);
    for (int t = 0; t < N_TASKS; t++) {
        for (int r = 0; r < ROWS; r++) {
            float scale = fabsf(refs[t][r]) + 1e-6f;
            assert(fabsf(outs[t][r] - refs[t][r]) / scale < 0.02f);
        }
    }

    free(blocks);
    printf("PASSED\n");
}

// --- Integration test: threaded matvec ---

static void test_matvec_threaded(void) {
    printf("test_matvec_threaded... ");

    int rows = 8, cols = 256;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data = (uint8_t *)calloc(data_size, 1);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < row_bytes; b++) {
            data[r * row_bytes + b] = 0x25;
        }
    }

    float tensor_scale = 0.5f;
    memcpy(data + (size_t)rows * row_bytes, &tensor_scale, sizeof(float));

    BnQWeight W = { data, BN_GGUF_TENSOR_I2_S, rows, cols, tensor_scale };

    float x[256];
    for (int i = 0; i < cols; i++) x[i] = 0.1f * (i % 13) - 0.6f;

    // Serial reference
    float ref[8];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref, &W, x, x_q_ref, NULL);

    // Threaded
    BnThreadPool *pool = bn_tp_create(3);
    float out[8];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, pool);

    for (int i = 0; i < rows; i++) {
        float err = fabsf(out[i] - ref[i]);
        float mag = fabsf(ref[i]) + 1e-6f;
        assert(err / mag < 0.02f);
    }

    bn_tp_free(pool);
    free(data);
    printf("PASSED\n");
}

// --- Integration test: matmul vs N individual matvecs ---
// Verifies that bn_quant_matmul produces identical results to calling
// bn_quant_matvec N times with different x vectors.
static void test_matmul_correctness(void) {
    printf("test_matmul_correctness... ");

    int rows = 4, cols = 256, n_tokens = 3;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data = (uint8_t *)calloc(data_size, 1);

    // Fill with deterministic ternary pattern
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < row_bytes; b++)
            data[r * row_bytes + b] = (uint8_t)((r * 17 + b * 31) & 0xFF);

    float tensor_scale = 0.25f;
    memcpy(data + (size_t)rows * row_bytes, &tensor_scale, sizeof(float));
    BnQWeight W = { data, BN_GGUF_TENSOR_I2_S, rows, cols, tensor_scale };

    // Create N different x vectors
    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < cols; i++)
            X[t * cols + i] = 0.1f * ((t * 7 + i * 3) % 19) - 0.9f;

    // Reference: N individual matvec calls
    float *ref = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    int8_t x_q[256];
    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(ref + t * rows, &W, X + t * cols, x_q, NULL);

    // Matmul: single call for all N tokens
    float *out = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    bn_quant_matmul(out, &W, X, n_tokens, x_q, NULL);

    // Compare
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float diff = fabsf(out[t * rows + r] - ref[t * rows + r]);
            float mag = fabsf(ref[t * rows + r]) + 1e-6f;
            assert(diff / mag < 0.01f);
        }
    }

    free(data); free(X); free(ref); free(out);
    printf("PASSED\n");
}

static void test_q4_matmul_correctness(void) {
    printf("test_q4_matmul_correctness... ");

    int rows = 8, cols = 96, n_tokens = 5;
    int n_bpr = cols / 32;
    BnBlockQ4_0 *blocks = (BnBlockQ4_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4_0));
    assert(blocks);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ4_0 *blk = &blocks[(size_t)r * n_bpr + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(1 + ((r + b) % 5)));
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (uint8_t)((r * 3 + b * 5 + i * 7) & 0x0F);
                uint8_t hi = (uint8_t)((r * 11 + b * 13 + i * 17 + 3) & 0x0F);
                blk->qs[i] = (uint8_t)(lo | (hi << 4));
            }
        }
    }

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };
    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    float *ref = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_prepared = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_llama_native = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_llama_prepared = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_llama_batch0 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_llama_batch1 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_multi0 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_multi1 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    int8_t *x_q = (int8_t *)malloc((size_t)cols);
    int n_groups = rows / 4;
    BnPreparedWeight prepared = { 0 };
    prepared.qs = (uint8_t *)calloc((size_t)n_groups * n_bpr * 64, 1);
    prepared.scales = (uint16_t *)calloc((size_t)n_groups * n_bpr * 4, sizeof(uint16_t));
    assert(X && ref && out && out_prepared && out_llama_native &&
           out_llama_prepared && out_llama_batch0 && out_llama_batch1 &&
           out_multi0 && out_multi1 && x_q && prepared.qs &&
           prepared.scales);

    for (int g = 0; g < n_groups; g++) {
        for (int b = 0; b < n_bpr; b++) {
            size_t gb = (size_t)g * n_bpr + b;
            for (int r = 0; r < 4; r++) {
                size_t src = (size_t)(g * 4 + r) * n_bpr + b;
                prepared.scales[gb * 4 + r] = blocks[src].d;
            }
            uint8_t *dst = prepared.qs + gb * 64;
            for (int ng = 0; ng < 4; ng++) {
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) * n_bpr + b;
                    const uint8_t *qs = blocks[src].qs + ng * 4;
                    uint8_t *dp = dst + ng * 16 + r * 4;
                    for (int j = 0; j < 4; j++)
                        dp[j] = qs[j] ^ 0x88;
                }
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < cols; i++) {
            int v = (t * 19 + i * 23) % 41;
            X[(size_t)t * cols + i] = 0.075f * (float)(v - 20);
        }
    }

    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(ref + (size_t)t * rows, &W, X + (size_t)t * cols, x_q, NULL);

    bn_quant_matmul(out, &W, X, n_tokens, x_q, NULL);
    bn_quant_matmul_prepared(out_prepared, &W, &prepared, X, n_tokens, x_q, NULL);
    for (int t = 0; t < n_tokens; t++) {
        bn_quant_matvec_prepared_flags(out_llama_native + (size_t)t * rows,
                                       &W, NULL, X + (size_t)t * cols, x_q,
                                       NULL, BN_MATVEC_TASK_REFERENCE_DOT);
        bn_quant_matvec_prepared_flags(out_llama_prepared + (size_t)t * rows,
                                       &W, &prepared, X + (size_t)t * cols,
                                       x_q, NULL, BN_MATVEC_TASK_REFERENCE_DOT);
        BnMatvecTask q4_batch[2] = {
            { out_llama_batch0 + (size_t)t * rows, &W, &prepared,
              BN_MATVEC_TASK_REFERENCE_DOT },
            { out_llama_batch1 + (size_t)t * rows, &W, &prepared,
              BN_MATVEC_TASK_REFERENCE_DOT },
        };
        bn_quant_matvec_batch(q4_batch, 2, X + (size_t)t * cols, x_q, NULL);
    }
    {
        float *multi_out[2] = { out_multi0, out_multi1 };
        const BnQWeight *multi_w[2] = { &W, &W };
        const BnPreparedWeight *multi_prepared[2] = { &prepared, &prepared };
        bn_quant_matmul_prepared_multi(multi_out, multi_w, multi_prepared, 2,
                                       X, n_tokens, x_q, NULL);
    }

    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float diff = fabsf(out[(size_t)t * rows + r] - ref[(size_t)t * rows + r]);
            float mag = fabsf(ref[(size_t)t * rows + r]) + 1e-6f;
            assert(diff / mag < 0.01f || diff < 1e-4f);
            diff = fabsf(out_prepared[(size_t)t * rows + r] - ref[(size_t)t * rows + r]);
            assert(diff / mag < 0.01f || diff < 1e-4f);
            diff = fabsf(out_multi0[(size_t)t * rows + r] - ref[(size_t)t * rows + r]);
            assert(diff / mag < 0.01f || diff < 1e-4f);
            diff = fabsf(out_multi1[(size_t)t * rows + r] - ref[(size_t)t * rows + r]);
            assert(diff / mag < 0.01f || diff < 1e-4f);
            diff = fabsf(out_llama_prepared[(size_t)t * rows + r] -
                         out_llama_native[(size_t)t * rows + r]);
            assert(diff < 1e-5f);
            diff = fabsf(out_llama_batch0[(size_t)t * rows + r] -
                         out_llama_native[(size_t)t * rows + r]);
            assert(diff < 1e-5f);
            diff = fabsf(out_llama_batch1[(size_t)t * rows + r] -
                         out_llama_native[(size_t)t * rows + r]);
            assert(diff < 1e-5f);
        }
    }

    free(blocks); free(X); free(ref); free(out); free(out_prepared);
    free(out_llama_native); free(out_llama_prepared);
    free(out_llama_batch0); free(out_llama_batch1);
    free(out_multi0); free(out_multi1);
    free(x_q); free(prepared.qs); free(prepared.scales);
    printf("PASSED\n");
}

static void test_q8_matmul_correctness(void) {
    printf("test_q8_matmul_correctness... ");

    int rows = 7, cols = 96, n_tokens = 5;
    int n_bpr = cols / 32;
    BnBlockQ8_0 *blocks = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));
    assert(blocks);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ8_0 *blk = &blocks[(size_t)r * n_bpr + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(1 + ((r + b) % 7)));
            for (int i = 0; i < 32; i++)
                blk->qs[i] = (int8_t)(((r * 17 + b * 11 + i * 5) % 63) - 31);
        }
    }

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };
    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    float *ref = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    int8_t *x_q = (int8_t *)malloc((size_t)cols);
    assert(X && ref && out && x_q);

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < cols; i++) {
            int v = (t * 23 + i * 19 + 5) % 53;
            X[(size_t)t * cols + i] = 0.0625f * (float)(v - 26);
        }
    }

    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(ref + (size_t)t * rows, &W,
                        X + (size_t)t * cols, x_q, NULL);

    bn_quant_matmul(out, &W, X, n_tokens, x_q, NULL);

#if defined(__AVX2__) && !defined(__AVX512F__)
    int8_t *xq_all = (int8_t *)malloc((size_t)n_tokens * cols);
    float *xs_all = (float *)malloc(
        (size_t)n_tokens * n_bpr * sizeof(float));
    assert(xq_all && xs_all);
    for (int t = 0; t < n_tokens; t++)
        bn_quant_x_to_q8_blocks(X + (size_t)t * cols,
                                xq_all + (size_t)t * cols,
                                xs_all + (size_t)t * n_bpr, cols);
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            __m256 acc = _mm256_setzero_ps();
            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ8_0 *blk =
                    &blocks[(size_t)r * n_bpr + b];
                __m256i w = _mm256_loadu_si256(
                    (const __m256i *)blk->qs);
                __m256i xq = _mm256_loadu_si256((const __m256i *)(
                    xq_all + (size_t)t * cols + b * 32));
                __m256i dot = bn_avx2_dpbusd(
                    _mm256_setzero_si256(), w, xq);
                __m256 scale = _mm256_set1_ps(
                    bn_fp16_to_fp32(blk->d) *
                    xs_all[(size_t)t * n_bpr + b]);
                acc = _mm256_fmadd_ps(
                    scale, _mm256_cvtepi32_ps(dot), acc);
            }
            assert(out[(size_t)t * rows + r] == bn_avx2_hsum_ps(acc));
        }
    }
    free(xq_all);
    free(xs_all);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    assert(memcmp(out, ref,
                  (size_t)n_tokens * rows * sizeof(float)) == 0);
#endif
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float diff = fabsf(out[(size_t)t * rows + r] -
                               ref[(size_t)t * rows + r]);
            assert(diff < 1e-5f);
        }
    }

    free(blocks);
    free(X);
    free(ref);
    free(out);
    free(x_q);
    printf("PASSED\n");
}

static void test_q5k_matmul_correctness(void) {
    printf("test_q5k_matmul_correctness... ");

    int rows = 2, cols = 256, n_tokens = 3;
    BnBlockQ5K *blocks = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    for (int r = 0; r < rows; r++) {
        blocks[r].d = 0x3C00;
        for (int i = 0; i < 4; i++) blocks[r].scales[i] = 1;
        for (int i = 8; i < 12; i++) blocks[r].scales[i] = 1;
        memset(blocks[r].qs, r == 0 ? 0x11 : 0x22, sizeof(blocks[r].qs));
    }
    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < cols; i++)
            X[t * cols + i] = 0.05f * ((t * 11 + i * 5) % 23) - 0.5f;

    float ref[6];
    int8_t x_q[256];
    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(ref + t * rows, &W, X + t * cols, x_q, NULL);

    float out[6];
    bn_quant_matmul(out, &W, X, n_tokens, x_q, NULL);

    for (int i = 0; i < rows * n_tokens; i++) {
        float diff = fabsf(out[i] - ref[i]);
        float mag = fabsf(ref[i]) + 1e-6f;
        assert(diff / mag < 0.02f);
    }

    free(blocks);
    free(X);
    printf("PASSED\n");
}

static void fill_q4k_blocks(BnBlockQ4K *q4, int rows, int n_bpr, int seed);

static void test_q5k_x8_prepared_correctness(void) {
    printf("test_q5k_x8_prepared_correctness... ");
#ifdef __AVX2__
    enum { rows = 8, cols = BN_QK_K };
    BnBlockQ5K blocks[rows];
    float x[cols];
    int8_t x_q[cols];
    float x_d[1];
    int16_t x_bsums[16];
    float ref[rows], got[rows];
    float panel_x[4 * cols];
    int8_t panel_q[4 * cols];
    float panel_d[4];
    int16_t panel_bsums[4 * 16];
    float panel_ref[4 * rows], panel_got[4 * rows];

    for (int r = 0; r < rows; r++) {
        blocks[r].d = bn_fp32_to_fp16(0.004f * (float)(r + 2));
        blocks[r].dmin = bn_fp32_to_fp16(0.002f * (float)(r + 1));
        for (int i = 0; i < 12; i++)
            blocks[r].scales[i] = (uint8_t)(r * 37 + i * 19 + 5);
        for (int i = 0; i < BN_QK_K / 8; i++)
            blocks[r].qh[i] = (uint8_t)(r * 29 + i * 43 + 11);
        for (int i = 0; i < BN_QK_K / 2; i++)
            blocks[r].qs[i] = (uint8_t)(r * 31 + i * 17 + 7);
    }
    for (int i = 0; i < cols; i++)
        x[i] = 0.0125f * (float)((i * 23 + 9) % 71 - 35);

    BnQWeight weight = {
        blocks, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f
    };
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(&weight, &kind);
    assert(kind == BN_PREPARED_WEIGHT_Q5_K_X8 && bytes > 0);
    SHArena *arena = sh_arena_create(bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = {0};
    assert(bn_quant_prepare_qweight(&prepared, &weight, arena) == 0);
    assert(prepared.aux != NULL);
    assert(prepared.qs == NULL);
    assert(prepared.f32_scales == NULL);

    bn_quant_x_to_q8k(x, x_q, x_d, x_bsums, cols);
    BnKQuantSdotCtx ref_ctx = {
        ref, &weight, x_q, x_d, x_bsums, NULL
    };
    bn_quant_q5k_scalar_sdot_range(&ref_ctx, 0, rows);
    BnKQuantSdotCtx prepared_ctx = {
        got, &weight, x_q, x_d, x_bsums, &prepared
    };
    bn_quant_q5k_avx2_x8_matvec_range(&prepared_ctx, 0, rows / 8);
    for (int r = 0; r < rows; r++)
        assert(fabsf(got[r] - ref[r]) < 2e-4f);

    for (int t = 0; t < 4; t++) {
        for (int i = 0; i < cols; i++)
            panel_x[t * cols + i] =
                0.009375f * (float)((t * 19 + i * 23 + 9) % 71 - 35);
        bn_quant_x_to_q8k(panel_x + t * cols, panel_q + t * cols,
                          panel_d + t, panel_bsums + t * 16, cols);
        BnKQuantSdotCtx panel_ref_ctx = {
            panel_ref + t * rows, &weight, panel_q + t * cols,
            panel_d + t, panel_bsums + t * 16, NULL
        };
        bn_quant_q5k_scalar_sdot_range(&panel_ref_ctx, 0, rows);
    }
    float *panel_out[1] = { panel_got };
    const BnQWeight *panel_weights[1] = { &weight };
    const BnPreparedWeight *panel_prepared[1] = { &prepared };
    bn_quant_matmul_prepared_kquant_input_multi(
        panel_out, panel_weights, panel_prepared, 1, 4, panel_q, panel_d,
        panel_bsums, panel_x, NULL);
    for (int i = 0; i < 4 * rows; i++)
        assert(fabsf(panel_got[i] - panel_ref[i]) < 2e-4f);
    sh_arena_free(arena);
#endif
    printf("PASSED\n");
}

static void test_q4k_avx512_x16_prepared_matmul_tail_order(void) {
    printf("test_q4k_avx512_x16_prepared_matmul_tail_order... ");
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    enum { rows = 32, cols = 3 * BN_QK_K, nb = cols / BN_QK_K, n_tokens = 65 };
    BnBlockQ4K blocks[rows * nb];
    float x[n_tokens * cols];
    int8_t x_q[n_tokens * cols];
    float x_d[n_tokens * nb];
    int16_t x_bsums[n_tokens * nb * 16];
    float ref[n_tokens * rows], got[n_tokens * rows];
    float panel[4 * rows], tail[rows];

    fill_q4k_blocks(blocks, rows, nb, 17);
    BnQWeight weight = {
        .data = blocks, .type = BN_GGUF_TENSOR_Q4_K,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(&weight, &kind);
    assert(kind == BN_PREPARED_WEIGHT_Q4_K_SCALES && bytes > 0);
    SHArena *arena = sh_arena_create(bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = {0};
    assert(bn_quant_prepare_qweight(&prepared, &weight, arena) == 0);

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < cols; i++)
            x[t * cols + i] =
                0.01125f * (float)((t * 31 + i * 17 + 5) % 79 - 39);
        bn_quant_x_to_q8k(x + t * cols, x_q + t * cols, x_d + t * nb,
                          x_bsums + t * nb * 16, cols);
        BnKQuantSdotCtx ref_ctx = {
            ref + t * rows, &weight, x_q + t * cols,
            x_d + t * nb, x_bsums + t * nb * 16, NULL
        };
        bn_quant_q4k_avx2_sdot_range(&ref_ctx, 0, rows);
    }

    float *out[1] = { got };
    const BnQWeight *weights[1] = { &weight };
    const BnPreparedWeight *prepared_weights[1] = { &prepared };
    float *panel_out[1] = { panel };
    const int counts[] = {17, 31, 32, 33, 48, 63, 64, 65};
    for (size_t k = 0; k < sizeof(counts) / sizeof(counts[0]); k++) {
        int count = counts[k];
        for (int i = 0; i < n_tokens * rows; i++) got[i] = NAN;
        bn_quant_matmul_prepared_kquant_input_multi(
            out, weights, prepared_weights, 1, count, x_q, x_d, x_bsums,
            x, NULL);
        for (int i = 0; i < count * rows; i++)
            assert(fabsf(got[i] - ref[i]) < 1e-3f);
        /* All wide tiles must retain the reference four-token FP order,
         * including its separately accumulated minimum correction. */
        for (int t = 0; t < count - count % 4; t += 4) {
            bn_quant_matmul_prepared_kquant_input_multi(
                panel_out, weights, prepared_weights, 1, 4,
                x_q + t * cols, x_d + t * nb, x_bsums + t * nb * 16,
                x + t * cols, NULL);
            assert(memcmp(got + t * rows, panel, sizeof(panel)) == 0);
        }
        for (int t = count - count % 4; t < count; t++) {
            BnKQuantSdotCtx tail_ctx = {
                tail, &weight, x_q + t * cols, x_d + t * nb,
                x_bsums + t * nb * 16, &prepared
            };
            bn_quant_q4k_avx2_x8_matvec_range(&tail_ctx, 0, rows / 8);
            assert(memcmp(got + t * rows, tail, sizeof(tail)) == 0);
        }
        for (int i = count * rows; i < n_tokens * rows; i++)
            assert(isnan(got[i]));
    }
    sh_arena_free(arena);
#endif
    printf("PASSED\n");
}

static void test_q8k_avx2_x4_pack_signed_max_tie(void) {
    printf("test_q8k_avx2_x4_pack_signed_max_tie... ");
#if defined(__AVX2__)
    float x[4 * BN_QK_K] = {0};
    BnBlockQ8Kx4 packed;

    for (int row = 0; row < 4; row++) {
        for (int i = 0; i < BN_QK_K; i++)
            x[row * BN_QK_K + i] =
                0.003f * (float)(((row + 3) * (i + 5)) % 101 - 50);
        x[row * BN_QK_K] = -1.0f;
        x[row * BN_QK_K + 255] = 1.0f;
    }

    bn_quant_q8k_avx2_pack_x4(&packed, x, 4, BN_QK_K);
    for (int row = 0; row < 4; row++)
        assert(packed.d[row] == -1.0f / 127.0f);

    for (int i = 0; i < BN_QK_K; i++) {
        int j = (i / 8) * 32 + (i % 8);
        for (int row = 0; row < 4; row++) {
            int expected = (int)nearbyintf(
                x[row * BN_QK_K + i] * -127.0f);
            assert(packed.qs[j + row * 8] == expected);
        }
    }
    for (int index = 0; index < BN_QK_K / 4; index++) {
        int sum = 0;
        for (int j = 0; j < BN_QK_K * 4; j++) {
            int j_index = (((j & 31) >> 3) << 2) +
                          ((j >> 8) << 4) + ((j >> 6) & 3);
            if (j_index == index)
                sum += packed.qs[j];
        }
        assert(packed.bsums[index] == sum);
    }
#endif
    printf("PASSED\n");
}

static void test_q5k_multi_matches_single_exact(void) {
    printf("test_q5k_multi_matches_single_exact... ");
#ifdef __AVX2__
    enum { max_tasks = 25, max_rows = 137, nb = 3, cols = nb * BN_QK_K };
    const int counts[] = {1, 2, 8, 24, 25};
    BnBlockQ5K *blocks = malloc((size_t)max_tasks * max_rows * nb * sizeof(*blocks));
    float *x = malloc((size_t)max_tasks * cols * sizeof(*x));
    int8_t *scratch = malloc((size_t)max_tasks * cols);
    assert(blocks && x && scratch);
    BnQWeight weights[max_tasks];
    BnMatvecMultiTask tasks[max_tasks];
    float expected[max_tasks][max_rows], actual[max_tasks][max_rows + 2];
    for (int t = 0; t < max_tasks; t++) {
        BnBlockQ5K *data = blocks + (size_t)t * max_rows * nb;
        int rows = max_rows - t % 5;
        for (int b = 0; b < rows * nb; b++) {
            data[b].d = bn_fp32_to_fp16(0.0031f * (1 + (b + t) % 7));
            data[b].dmin = bn_fp32_to_fp16(0.0023f * (1 + (b + t) % 5));
            for (int i = 0; i < 12; i++) data[b].scales[i] = (uint8_t)(b * 13 + i * 19 + t);
            for (int i = 0; i < 32; i++) data[b].qh[i] = (uint8_t)(b * 23 + i * 17 + t);
            for (int i = 0; i < 128; i++) data[b].qs[i] = (uint8_t)(b * 7 + i * 31 + t);
        }
        for (int i = 0; i < cols; i++)
            x[t * cols + i] = sinf((float)(i + 17 * t) * 0.037f) * 1.7f;
        weights[t] = (BnQWeight){data, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f};
        tasks[t] = (BnMatvecMultiTask){actual[t] + 1, &weights[t], x + t * cols, NULL};
        bn_quant_matvec(expected[t], &weights[t], x + t * cols, scratch, NULL);
    }
    BnThreadPool *workers = bn_tp_create(3);
    assert(workers);
    for (int mode = 0; mode < 2; mode++)
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            for (int t = 0; t < max_tasks; t++)
                for (int i = 0; i < max_rows + 2; i++) actual[t][i] = NAN;
            bn_quant_matvec_multi(tasks, counts[c], scratch, mode ? workers : NULL);
            for (int t = 0; t < max_tasks; t++) {
                assert(isnan(actual[t][0]));
                int written = t < counts[c] ? weights[t].rows : 0;
                if (written)
                    assert(memcmp(actual[t] + 1, expected[t], (size_t)written * sizeof(float)) == 0);
                for (int i = written + 1; i < max_rows + 2; i++) assert(isnan(actual[t][i]));
            }
        }
    /* Mixed column counts must use per-task dispatch, not shared strides. */
    weights[0].cols = cols - BN_QK_K;
    bn_quant_matvec(expected[0], &weights[0], x, scratch, NULL);
    for (int mode = 0; mode < 2; mode++) {
        bn_quant_matvec_multi(tasks, 2, scratch, mode ? workers : NULL);
        for (int t = 0; t < 2; t++)
            assert(memcmp(actual[t] + 1, expected[t],
                          (size_t)weights[t].rows * sizeof(float)) == 0);
    }
    bn_tp_free(workers);
    free(blocks);
    free(x);
    free(scratch);
#endif
    printf("PASSED\n");
}

static void test_q5k_matvec_multi_correctness(void) {
    printf("test_q5k_matvec_multi_correctness... ");

    int rows = 3, cols = 256, n_tasks = 2;
    BnBlockQ5K *blocks1 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    BnBlockQ5K *blocks2 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    for (int r = 0; r < rows; r++) {
        blocks1[r].d = 0x3C00;
        blocks2[r].d = 0x3C00;
        for (int i = 0; i < 4; i++) {
            blocks1[r].scales[i] = 1;
            blocks2[r].scales[i] = 1;
        }
        for (int i = 8; i < 12; i++) {
            blocks1[r].scales[i] = 1;
            blocks2[r].scales[i] = 1;
        }
        memset(blocks1[r].qs, 0x11 + r, sizeof(blocks1[r].qs));
        memset(blocks2[r].qs, 0x22 + r, sizeof(blocks2[r].qs));
    }
    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float X1[256], X2[256];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.04f * ((i * 7) % 29) - 0.5f;
        X2[i] = 0.03f * ((i * 5 + 3) % 31) - 0.4f;
    }

    float ref1[3], ref2[3], out1[3], out2[3];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 256];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        float mag1 = fabsf(ref1[i]) + 1e-6f;
        float mag2 = fabsf(ref2[i]) + 1e-6f;
        assert(fabsf(out1[i] - ref1[i]) / mag1 < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / mag2 < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_q5k_matvec_batch_correctness(void) {
    printf("test_q5k_matvec_batch_correctness... ");

    int rows = 7, cols = 256;
    BnBlockQ5K *blocks1 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    BnBlockQ5K *blocks2 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    BnBlockQ6K *blocks6 = (BnBlockQ6K *)calloc((size_t)rows, sizeof(BnBlockQ6K));
    for (int r = 0; r < rows; r++) {
        blocks1[r].d = 0x3C00;
        blocks2[r].d = 0x3C00;
        for (int i = 0; i < 4; i++) {
            blocks1[r].scales[i] = (uint8_t)(1 + ((r + i) % 3));
            blocks2[r].scales[i] = (uint8_t)(1 + ((2 * r + i) % 3));
        }
        for (int i = 8; i < 12; i++) {
            blocks1[r].scales[i] = (uint8_t)(1 + ((r + i) % 3));
            blocks2[r].scales[i] = (uint8_t)(1 + ((2 * r + i) % 3));
        }
        for (int i = 0; i < 128; i++) {
            blocks1[r].qs[i] = (uint8_t)(r * 19 + i * 7);
            blocks2[r].qs[i] = (uint8_t)(r * 23 + i * 5 + 1);
        }
        for (int i = 0; i < 32; i++) {
            blocks1[r].qh[i] = (uint8_t)(r * 11 + i * 3);
            blocks2[r].qh[i] = (uint8_t)(r * 13 + i * 5 + 2);
        }
        blocks6[r].d = 0x3C00;
        for (int i = 0; i < 16; i++)
            blocks6[r].scales[i] = (int8_t)((r * 5 + i * 3) % 17 - 8);
        for (int i = 0; i < 128; i++)
            blocks6[r].ql[i] = (uint8_t)(r * 13 + i * 5);
        for (int i = 0; i < 64; i++)
            blocks6[r].qh[i] = (uint8_t)(r * 19 + i * 7);
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };
    BnQWeight W6 = { blocks6, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < cols; i++)
        x[i] = 0.04f * ((i * 7 + 5) % 31) - 0.6f;

    float ref1[7], ref2[7], out1[7], out2[7];
    float prepared_out1[7], prepared_out2[7];
    float ref6[7], prepared_out6[7];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);
    bn_quant_matvec(ref6, &W6, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    int8_t x_q[256];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    float x_d[1];
    int16_t x_bsums[16];
    bn_quant_x_to_q8k(x, x_q, x_d, x_bsums, cols);
    BnMatvecTask prepared_tasks[3] = {
        { prepared_out1, &W1, NULL, 0 },
        { prepared_out2, &W2, NULL, 0 },
        { prepared_out6, &W6, NULL, 0 },
    };
    bn_quant_matvec_batch_prepared_kquant_input(
        prepared_tasks, 3, x_q, x_d, x_bsums, x, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
        assert(isfinite(prepared_out1[i]));
        assert(isfinite(prepared_out2[i]));
        assert(isfinite(prepared_out6[i]));
        assert(fabsf(prepared_out6[i] - ref6[i]) /
               (fabsf(ref6[i]) + 1e-6f) < 0.05f);
#if defined(__AVX2__) && !(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
        assert(fabsf(prepared_out1[i] - out1[i]) /
               (fabsf(out1[i]) + 1e-6f) < 1e-5f);
        assert(fabsf(prepared_out2[i] - out2[i]) /
               (fabsf(out2[i]) + 1e-6f) < 1e-5f);
#else
        assert(fabsf(prepared_out1[i] - ref1[i]) /
               (fabsf(ref1[i]) + 1e-6f) < 0.05f);
        assert(fabsf(prepared_out2[i] - ref2[i]) /
               (fabsf(ref2[i]) + 1e-6f) < 0.05f);
#endif
    }

    free(blocks1);
    free(blocks2);
    free(blocks6);
    printf("PASSED\n");
}

static void test_i2s_matvec_multi_correctness(void) {
    printf("test_i2s_matvec_multi_correctness... ");

    int rows = 7, cols = 256, n_tasks = 3;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data1 = (uint8_t *)calloc(data_size, 1);
    uint8_t *data2 = (uint8_t *)calloc(data_size, 1);
    uint8_t *data3 = (uint8_t *)calloc(data_size, 1);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < row_bytes; b++) {
            data1[r * row_bytes + b] = (uint8_t)(r * 13 + b * 7);
            data2[r * row_bytes + b] = (uint8_t)(r * 11 + b * 5 + 3);
            data3[r * row_bytes + b] = (uint8_t)(r * 17 + b * 9 + 1);
        }
    }

    float scale = 0.25f;
    memcpy(data1 + (size_t)rows * row_bytes, &scale, sizeof(float));
    memcpy(data2 + (size_t)rows * row_bytes, &scale, sizeof(float));
    memcpy(data3 + (size_t)rows * row_bytes, &scale, sizeof(float));

    BnQWeight W1 = { data1, BN_GGUF_TENSOR_I2_S, rows, cols, scale };
    BnQWeight W2 = { data2, BN_GGUF_TENSOR_I2_S, rows, cols, scale };
    BnQWeight W3 = { data3, BN_GGUF_TENSOR_I2_S, rows, cols, scale };

    float X1[256], X2[256], X3[256];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.03f * ((i * 7) % 31) - 0.45f;
        X2[i] = 0.04f * ((i * 5 + 2) % 29) - 0.55f;
        X3[i] = 0.05f * ((i * 3 + 4) % 23) - 0.50f;
    }

    float ref1[7], ref2[7], ref3[7];
    float out1[7], out2[7], out3[7];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);
    bn_quant_matvec(ref3, &W3, X3, x_q_ref, NULL);

    BnMatvecMultiTask tasks[3] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
         { out3, &W3, X3, NULL },
    };
    int8_t x_q_bufs[3 * 256];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out3[i] - ref3[i]) / (fabsf(ref3[i]) + 1e-6f) < 0.02f);
    }

    free(data1);
    free(data2);
    free(data3);
    printf("PASSED\n");
}

static void test_q4_matvec_multi_correctness(void) {
    printf("test_q4_matvec_multi_correctness... ");

    int rows = 7, cols = 64, n_tasks = 2;
    int n_bpr = cols / 32;
    BnBlockQ4_0 *blocks1 = (BnBlockQ4_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4_0));
    BnBlockQ4_0 *blocks2 = (BnBlockQ4_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4_0));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ4_0 *a = &blocks1[r * n_bpr + b];
            BnBlockQ4_0 *c = &blocks2[r * n_bpr + b];
            a->d = 0x3C00;
            c->d = 0x3C00;
            for (int i = 0; i < 16; i++) {
                a->qs[i] = (uint8_t)(r * 19 + b * 11 + i * 7);
                c->qs[i] = (uint8_t)(r * 23 + b * 13 + i * 5 + 1);
            }
        }
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };

    float X1[64], X2[64];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.05f * ((i * 7) % 17) - 0.40f;
        X2[i] = 0.04f * ((i * 5 + 3) % 19) - 0.35f;
    }

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[64];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 64];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_q4_repacked_neon_reduction_order(void) {
    printf("test_q4_repacked_neon_reduction_order... ");
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    enum { rows = 8, cols = 1024, n_blocks = cols / 32 };
    BnBlockQ4_0 blocks[rows * n_blocks];
    float x[cols];
    int8_t x_q[cols];
    float x_scales[n_blocks];
    float native[rows];
    float repacked[rows];
    float repacked_scalar[rows];
    float scalar[rows];
    float production[rows];
    memset(native, 0, sizeof(native));
    memset(repacked, 0, sizeof(repacked));
    memset(repacked_scalar, 0, sizeof(repacked_scalar));
    memset(scalar, 0, sizeof(scalar));
    memset(production, 0, sizeof(production));
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4_0 *blk = &blocks[r * n_blocks + b];
            blk->d = bn_fp32_to_fp16(0.015625f * (float)(r + b + 1));
            for (int i = 0; i < 16; i++)
                blk->qs[i] = (uint8_t)(r * 29 + b * 17 + i * 11);
        }
    }
    for (int i = 0; i < cols; i++)
        x[i] = 0.03125f * (float)((i * 13 + 5) % 47) - 0.625f;

    BnQWeight W = {
        .data = blocks, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(&W, &kind);
    assert(kind == BN_PREPARED_WEIGHT_Q4_0_REPACK && bytes > 0);
    SHArena *arena = sh_arena_create(bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = {0};
    assert(bn_quant_prepare_qweight(&prepared, &W, arena) == 0);
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, cols);

    BnQ4SdotCtx native_ctx = { native, &W, x_q, x_scales, NULL };
    BnQ4SdotCtx repacked_ctx = {
        repacked, &W, x_q, x_scales, &prepared
    };
    BnQ4SdotCtx repacked_scalar_ctx = {
        repacked_scalar, &W, x_q, x_scales, &prepared
    };
    BnQ4SdotCtx scalar_ctx = { scalar, &W, x_q, x_scales, NULL };
    bn_quant_q4_scalar_sdot_range(&scalar_ctx, 1, 8);
    bn_quant_q4_neon_sdot_range(&native_ctx, 1, 8);
    bn_quant_q4_repacked_neon_sdot_range(&repacked_ctx, 1, 8);
    bn_quant_q4_repacked_scalar_sdot_range(&repacked_scalar_ctx, 1, 8);
    for (int r = 4; r < 8; r++) {
        for (int b = 0; b < n_blocks; b++) {
            const BnBlockQ4_0 *blk = &blocks[r * n_blocks + b];
            int32_t dot = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t q = blk->qs[i];
                dot += ((int32_t)(q & 0x0f) - 8) * x_q[b * 32 + i];
                dot += ((int32_t)(q >> 4) - 8) * x_q[b * 32 + i + 16];
            }
            float scale = bn_fp16_to_fp32(blk->d) * x_scales[b];
            production[r] = fmaf((float)dot, scale, production[r]);
        }
    }
    for (int r = 1; r < rows; r++) {
        assert(native[r] == scalar[r]);
        assert(fabsf(native[r] - repacked[r]) < 1e-4f);
        if (r >= 4)
            assert(repacked_scalar[r] == repacked[r]);
        if (r >= 4)
            assert(production[r] == repacked[r]);
    }
    sh_arena_free(arena);
#endif
    printf("PASSED\n");
}

static void test_q4_repacked_scalar_layout(void) {
    printf("test_q4_repacked_scalar_layout... ");
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || \
    defined(__wasm_relaxed_simd__)
    enum { rows = 8, cols = 64, n_blocks = cols / 32 };
    BnBlockQ4_0 blocks[rows * n_blocks];
    float x[cols];
    int8_t x_q[cols];
    float x_scales[n_blocks];
    float native[rows];
    float prepared_out[rows];

    memset(native, 0, sizeof(native));
    memset(prepared_out, 0, sizeof(prepared_out));
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4_0 *blk = &blocks[r * n_blocks + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(r + b + 1));
            for (int i = 0; i < 16; i++)
                blk->qs[i] = (uint8_t)(r * 31 + b * 19 + i * 7);
        }
    }
    for (int i = 0; i < cols; i++)
        x[i] = 0.0625f * (float)((i * 17 + 3) % 31) - 0.75f;

    BnQWeight weight = {
        .data = blocks, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(&weight, &kind);
    assert(kind == BN_PREPARED_WEIGHT_Q4_0_REPACK && bytes > 0);
    SHArena *arena = sh_arena_create(bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = {0};
    assert(bn_quant_prepare_qweight(&prepared, &weight, arena) == 0);
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, cols);

    BnQ4SdotCtx native_ctx = {
        native, &weight, x_q, x_scales, NULL
    };
    BnQ4SdotCtx prepared_ctx = {
        prepared_out, &weight, x_q, x_scales, &prepared
    };
    bn_quant_q4_scalar_sdot_range(&native_ctx, 0, rows);
    bn_quant_q4_repacked_scalar_sdot_range(&prepared_ctx, 0, rows);
    for (int r = 0; r < rows; r++)
        assert(prepared_out[r] == native[r]);

    sh_arena_free(arena);
#endif
    printf("PASSED\n");
}

static void test_q4_repacked_neon_fused_gateup_silu(void) {
    printf("test_q4_repacked_neon_fused_gateup_silu... ");
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    enum { rows = 8, cols = 96, n_blocks = cols / 32 };
    BnBlockQ4_0 gate_blocks[rows * n_blocks];
    BnBlockQ4_0 up_blocks[rows * n_blocks];
    float x[cols];
    int8_t x_q[cols];
    float x_scales[n_blocks];
    float gate_out[rows], up_out[rows], fused[rows];
    memset(gate_out, 0, sizeof(gate_out));
    memset(up_out, 0, sizeof(up_out));
    memset(fused, 0, sizeof(fused));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4_0 *gate = &gate_blocks[r * n_blocks + b];
            BnBlockQ4_0 *up = &up_blocks[r * n_blocks + b];
            gate->d = bn_fp32_to_fp16(
                0.01171875f * (float)(1 + ((r + b * 3) % 9)));
            up->d = bn_fp32_to_fp16(
                0.0078125f * (float)(1 + ((r * 5 + b) % 11)));
            for (int i = 0; i < 16; i++) {
                gate->qs[i] = (uint8_t)(r * 31 + b * 19 + i * 7 + 3);
                up->qs[i] = (uint8_t)(r * 17 + b * 23 + i * 13 + 5);
            }
        }
    }
    for (int i = 0; i < cols; i++)
        x[i] = 0.01953125f * (float)((i * 11 + 7) % 53) - 0.5f;

    BnQWeight gate = {
        .data = gate_blocks, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    BnQWeight up = {
        .data = up_blocks, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    BnPreparedWeightKind gate_kind = BN_PREPARED_WEIGHT_NONE;
    BnPreparedWeightKind up_kind = BN_PREPARED_WEIGHT_NONE;
    size_t gate_bytes = bn_quant_prepared_qweight_size(&gate, &gate_kind);
    size_t up_bytes = bn_quant_prepared_qweight_size(&up, &up_kind);
    assert(gate_kind == BN_PREPARED_WEIGHT_Q4_0_REPACK);
    assert(up_kind == BN_PREPARED_WEIGHT_Q4_0_REPACK);
    SHArena *arena = sh_arena_create(
        gate_bytes + up_bytes + 2 * SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight gate_prepared = {0};
    BnPreparedWeight up_prepared = {0};
    assert(bn_quant_prepare_qweight(&gate_prepared, &gate, arena) == 0);
    assert(bn_quant_prepare_qweight(&up_prepared, &up, arena) == 0);
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, cols);

    BnQ4SdotCtx gate_ctx = {
        gate_out, &gate, x_q, x_scales, &gate_prepared
    };
    BnQ4SdotCtx up_ctx = {
        up_out, &up, x_q, x_scales, &up_prepared
    };
    BnQ4GateUpCtx fused_ctx = {
        fused, &gate, &up, x_q, x_scales,
        &gate_prepared, &up_prepared
    };
    bn_quant_q4_repacked_neon_sdot_range(&gate_ctx, 1, 7);
    bn_quant_q4_repacked_neon_sdot_range(&up_ctx, 1, 7);
    bn_quant_q4_repacked_gate_up_silu_neon_range(&fused_ctx, 1, 7);
    for (int r = 1; r < 7; r++) {
        float32x4_t silu =
            bn_neon_fast_silu_f32(vdupq_n_f32(gate_out[r]));
        float expected = vgetq_lane_f32(silu, 0) * up_out[r];
        assert(fused[r] == expected);
    }
    sh_arena_free(arena);
#endif
    printf("PASSED\n");
}

static void test_q4_neon_4row_reduction_order(void) {
    printf("test_q4_neon_4row_reduction_order... ");
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    enum { rows = 7, cols = 160, n_blocks = cols / 32 };
    BnBlockQ4_0 blocks[rows * n_blocks];
    float x[cols];
    int8_t x_q[cols];
    float x_scales[n_blocks];
    float single_row[rows];
    float four_row[rows];
    memset(single_row, 0, sizeof(single_row));
    memset(four_row, 0, sizeof(four_row));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4_0 *blk = &blocks[r * n_blocks + b];
            blk->d = bn_fp32_to_fp16(
                0.0078125f * (float)(1 + ((r * 3 + b * 5) % 11)));
            for (int i = 0; i < 16; i++)
                blk->qs[i] = (uint8_t)(r * 37 + b * 19 + i * 13 + 7);
        }
    }
    for (int i = 0; i < cols; i++)
        x[i] = 0.0234375f * (float)((i * 17 + 11) % 61) - 0.703125f;

    BnQWeight W = {
        .data = blocks, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, cols);
    BnQ4SdotCtx single_ctx = { single_row, &W, x_q, x_scales, NULL };
    BnQ4SdotCtx four_ctx = { four_row, &W, x_q, x_scales, NULL };
    bn_quant_q4_neon_sdot_range(&single_ctx, 0, rows);
    bn_quant_q4_neon_sdot_4row_range(&four_ctx, 0, (rows + 3) / 4);
    for (int r = 0; r < rows; r++)
        assert(fabsf(single_row[r] - four_row[r]) < 1e-6f);
#endif
    printf("PASSED\n");
}

static void test_q8_matvec_batch_correctness(void) {
    printf("test_q8_matvec_batch_correctness... ");

    int rows = 7, cols = 64;
    int n_bpr = cols / 32;
    BnBlockQ8_0 *blocks1 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));
    BnBlockQ8_0 *blocks2 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ8_0 *a = &blocks1[r * n_bpr + b];
            BnBlockQ8_0 *c = &blocks2[r * n_bpr + b];
            a->d = 0x3C00;
            c->d = 0x3C00;
            for (int i = 0; i < 32; i++) {
                a->qs[i] = (int8_t)(((r * 17 + b * 11 + i * 5) % 31) - 15);
                c->qs[i] = (int8_t)(((r * 13 + b * 7 + i * 3) % 29) - 14);
            }
        }
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float x[64];
    for (int i = 0; i < cols; i++)
        x[i] = 0.04f * ((i * 7 + 3) % 23) - 0.45f;

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[64];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    int8_t x_q[64];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_q8_neon_reference_reduction_order(void) {
    printf("test_q8_neon_reference_reduction_order... ");
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    enum { rows = 8, cols = 224, n_blocks = cols / 32 };
    BnBlockQ8_0 blocks[rows * n_blocks];
    int8_t x_q[cols];
    float x_scales[n_blocks];
    float reference[rows];
    float scalar_order[rows];
    float single_row[rows];
    float four_row[rows];
    float repacked[rows];

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ8_0 *blk = &blocks[r * n_blocks + b];
            blk->d = bn_fp32_to_fp16(
                0.00371f * (float)(1 + ((r * 5 + b * 7) % 13)));
            for (int i = 0; i < 32; i++)
                blk->qs[i] = (int8_t)(((r * 37 + b * 19 + i * 11) % 255) - 127);
        }
    }
    for (int i = 0; i < cols; i++)
        x_q[i] = (int8_t)(((i * 29 + 17) % 255) - 127);
    for (int b = 0; b < n_blocks; b++)
        x_scales[b] = 0.00731f * (float)(b + 1);

    BnQWeight weight = {
        .data = blocks,
        .type = BN_GGUF_TENSOR_Q8_0,
        .rows = rows,
        .cols = cols,
        .scale = 1.0f,
    };
    for (int r = 0; r < rows; r++) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        int b = 0;
        for (; b + 1 < n_blocks; b += 2) {
            int32_t lanes0[4] = { 0, 0, 0, 0 };
            int32_t lanes1[4] = { 0, 0, 0, 0 };
            const BnBlockQ8_0 *blk0 = &blocks[r * n_blocks + b];
            const BnBlockQ8_0 *blk1 = blk0 + 1;
            for (int lane = 0; lane < 4; lane++) {
                for (int i = 0; i < 4; i++) {
                    int j = lane * 4 + i;
                    lanes0[lane] += (int32_t)blk0->qs[j] * x_q[b * 32 + j];
                    lanes0[lane] += (int32_t)blk0->qs[j + 16] *
                                    x_q[b * 32 + j + 16];
                    lanes1[lane] += (int32_t)blk1->qs[j] *
                                    x_q[(b + 1) * 32 + j];
                    lanes1[lane] += (int32_t)blk1->qs[j + 16] *
                                    x_q[(b + 1) * 32 + j + 16];
                }
            }
            sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(vld1q_s32(lanes0)),
                                bn_fp16_to_fp32(blk0->d) * x_scales[b]);
            sum1 = vmlaq_n_f32(sum1, vcvtq_f32_s32(vld1q_s32(lanes1)),
                                bn_fp16_to_fp32(blk1->d) * x_scales[b + 1]);
        }
        reference[r] = vaddvq_f32(sum0) + vaddvq_f32(sum1);
        for (; b < n_blocks; b++) {
            const BnBlockQ8_0 *blk = &blocks[r * n_blocks + b];
            int32_t dot = 0;
            for (int i = 0; i < 32; i++)
                dot += (int32_t)blk->qs[i] * x_q[b * 32 + i];
            reference[r] += (float)dot * bn_fp16_to_fp32(blk->d) *
                            x_scales[b];
        }
    }

    BnQ8SdotCtx scalar_ctx = { scalar_order, &weight, x_q, x_scales, NULL };
    BnQ8SdotCtx single_ctx = { single_row, &weight, x_q, x_scales, NULL };
    BnQ8SdotCtx four_ctx = { four_row, &weight, x_q, x_scales, NULL };
    bn_quant_q8_scalar_sdot_range(&scalar_ctx, 0, rows);
    bn_quant_q8_neon_sdot_range(&single_ctx, 0, rows);
    bn_quant_q8_neon_sdot_4row_range(&four_ctx, 0, (rows + 3) / 4);
    assert(memcmp(reference, single_row, sizeof(reference)) == 0);
    assert(memcmp(reference, four_row, sizeof(reference)) == 0);
    assert(memcmp(reference, scalar_order, sizeof(reference)) != 0);

    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(&weight, &kind);
    assert(kind == BN_PREPARED_WEIGHT_Q8_0_REPACK && bytes > 0);
    SHArena *arena = sh_arena_create(bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = {0};
    assert(bn_quant_prepare_qweight(&prepared, &weight, arena) == 0);
    assert(prepared.kind == BN_PREPARED_WEIGHT_Q8_0_REPACK);
    BnQ8SdotCtx repacked_ctx = {
        repacked, &weight, x_q, x_scales, &prepared
    };
    bn_quant_q8_repacked_neon_sdot_range(&repacked_ctx, 0, rows);
    for (int r = 0; r < rows; r++)
        assert(fabsf(repacked[r] - scalar_order[r]) < 1e-3f);
    sh_arena_free(arena);
#endif
    printf("PASSED\n");
}

static void test_q8_matvec_multi_correctness(void) {
    printf("test_q8_matvec_multi_correctness... ");

    int rows = 7, cols = 64, n_tasks = 2;
    int n_bpr = cols / 32;
    BnBlockQ8_0 *blocks1 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));
    BnBlockQ8_0 *blocks2 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ8_0 *a = &blocks1[r * n_bpr + b];
            BnBlockQ8_0 *c = &blocks2[r * n_bpr + b];
            a->d = 0x3C00;
            c->d = 0x3C00;
            for (int i = 0; i < 32; i++) {
                a->qs[i] = (int8_t)(((r * 19 + b * 5 + i * 7) % 37) - 18);
                c->qs[i] = (int8_t)(((r * 11 + b * 13 + i * 3) % 35) - 17);
            }
        }
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float X1[64], X2[64];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.04f * ((i * 5 + 1) % 23) - 0.42f;
        X2[i] = 0.03f * ((i * 7 + 4) % 29) - 0.38f;
    }

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[64];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 64];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_bf16_matvec_batch_correctness(void) {
    printf("test_bf16_matvec_batch_correctness... ");

    int rows = 7, cols = 33;
    uint16_t *data1 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));
    uint16_t *data2 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v1 = 0.0625f * ((r * 17 + c * 5) % 23) - 0.625f;
            float v2 = 0.03125f * ((r * 11 + c * 7 + 3) % 29) - 0.4375f;
            data1[(size_t)r * cols + c] = test_fp32_to_bf16(v1);
            data2[(size_t)r * cols + c] = test_fp32_to_bf16(v2);
        }
    }

    BnQWeight W1 = { data1, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };
    BnQWeight W2 = { data2, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };

    float x[33];
    for (int i = 0; i < cols; i++)
        x[i] = 0.05f * ((i * 7 + 2) % 19) - 0.45f;

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[33];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    int8_t x_q[33];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) < 1e-4f);
        assert(fabsf(out2[i] - ref2[i]) < 1e-4f);
    }

    free(data1);
    free(data2);
    printf("PASSED\n");
}

static void test_unquantized_matvec_correctness(void) {
    printf("test_unquantized_matvec_correctness... ");

    int rows = 5, cols = 37;
    float *f32_data = (float *)calloc((size_t)rows * cols, sizeof(float));
    uint16_t *f16_data = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));
    assert(f32_data && f16_data);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v = 0.02f * ((r * 13 + c * 7) % 31) - 0.27f;
            f32_data[(size_t)r * cols + c] = v;
            f16_data[(size_t)r * cols + c] = bn_fp32_to_fp16(v);
        }
    }

    float x[37];
    for (int i = 0; i < cols; i++)
        x[i] = 0.03f * ((i * 11 + 5) % 23) - 0.31f;

    BnQWeight W32 = { f32_data, BN_GGUF_TENSOR_F32, rows, cols, 1.0f };
    BnQWeight W16 = { f16_data, BN_GGUF_TENSOR_F16, rows, cols, 1.0f };
    float out32[5], out16[5];
    int8_t x_q[37];

    bn_quant_matvec(out32, &W32, x, x_q, NULL);
    bn_quant_matvec(out16, &W16, x, x_q, NULL);

    for (int r = 0; r < rows; r++) {
        assert(fabsf(out32[r] - ref_dot_f32(f32_data + (size_t)r * cols, x, cols)) < 1e-5f);
        assert(fabsf(out16[r] - ref_dot_f16(f16_data + (size_t)r * cols, x, cols)) < 1e-4f);
    }

    free(f32_data);
    free(f16_data);
    printf("PASSED\n");
}

static void test_f32_scalar_reduction_order(void) {
    printf("test_f32_scalar_reduction_order... ");
#ifdef __ARM_NEON
    enum { rows = 7, cols = 5120 };
    float *weights = malloc((size_t)rows * cols * sizeof(*weights));
    float *x = malloc((size_t)cols * sizeof(*x));
    assert(weights && x);
    for (int i = 0; i < rows * cols; i++)
        weights[i] = 0.00390625f * (float)((i * 37 + 11) % 257) - 0.5f;
    for (int i = 0; i < cols; i++)
        x[i] = 0.0078125f * (float)((i * 29 + 5) % 193) - 0.75f;

    BnQWeight weight = {
        weights, BN_GGUF_TENSOR_F32, rows, cols, 1.0f
    };
    float scalar[rows], neon[rows];
    BnF32Ctx scalar_ctx = { scalar, &weight, x };
    BnF32Ctx neon_ctx = { neon, &weight, x };
    bn_quant_f32_scalar_range(&scalar_ctx, 0, rows);
    bn_quant_f32_neon_range(&neon_ctx, 0, rows);
    assert(memcmp(scalar, neon, sizeof(scalar)) == 0);

    free(weights);
    free(x);
#endif
    printf("PASSED\n");
}

static void test_q8k_quantizer_matches_scalar(void) {
    printf("test_q8k_quantizer_matches_scalar... ");
#ifdef __AVX2__
    enum { cols = 2 * BN_QK_K };
    float x[cols];
    int8_t q_avx[cols], q_ref[cols];
    float d_avx[2], d_ref[2];
    int16_t sums_avx[32], sums_ref[32];

    for (int i = 0; i < cols; i++) {
        float sign = (i % 3) ? 1.0f : -1.0f;
        x[i] = sign * (0.0037f * (float)(i % 131) +
                       0.000013f * (float)(i / 7));
    }
    x[17] = -3.25f;
    x[BN_QK_K + 93] = 4.75f;

    bn_quant_x_to_q8k(x, q_avx, d_avx, sums_avx, cols);
    bn_quant_x_to_q8k_scalar(x, q_ref, d_ref, sums_ref, cols);
    assert(memcmp(q_avx, q_ref, sizeof(q_avx)) == 0);
    assert(memcmp(d_avx, d_ref, sizeof(d_avx)) == 0);
    assert(memcmp(sums_avx, sums_ref, sizeof(sums_avx)) == 0);
#endif
    printf("PASSED\n");
}

#ifdef __AVX2__
static float test_f32_decode_dot_reference(const float *w, const float *x,
                                          int cols) {
    enum { lanes = 16 };
    float sums[4][lanes] = {{0}};
    int col = 0;
    for (; col + 4 * lanes <= cols; col += 4 * lanes)
        for (int a = 0; a < 4; a++)
            for (int lane = 0; lane < lanes; lane++) {
                int i = col + a * lanes + lane;
                sums[a][lane] = fmaf(w[i], x[i], sums[a][lane]);
            }
    float merged[lanes];
    for (int lane = 0; lane < lanes; lane++)
        merged[lane] = (sums[0][lane] + sums[2][lane]) +
                       (sums[1][lane] + sums[3][lane]);
#ifdef __AVX512F__
    float sum = _mm512_reduce_add_ps(_mm512_loadu_ps(merged));
#else
    float half[8], quarter[4];
    for (int lane = 0; lane < 8; lane++)
        half[lane] = merged[lane + 8] + merged[lane];
    for (int lane = 0; lane < 4; lane++)
        quarter[lane] = half[lane + 4] + half[lane];
    float sum = (quarter[0] + quarter[2]) +
                (quarter[1] + quarter[3]);
#endif
    for (; col < cols; col++) sum = fmaf(w[col], x[col], sum);
    return sum;
}
#endif

static void test_f32_x86_decode_reduction_order(void) {
    printf("test_f32_x86_decode_reduction_order... ");
#ifdef __AVX2__
    enum { rows = 137, max_cols = 5123 };
    const int sizes[] = {1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
                         127, 5120, 5123};
    float *storage = malloc(((size_t)rows * max_cols + 1) * sizeof(float));
    float *input = malloc((max_cols + 1) * sizeof(float));
    int8_t *x_q = malloc(max_cols);
    assert(storage && input && x_q);
    float *weights = storage + 1, *x = input + 1;
    for (int i = 0; i < rows * max_cols; i++)
        weights[i] = 0.00137f * (float)((i * 37 + 11) % 257) - 0.19f;
    for (int i = 0; i < max_cols; i++)
        x[i] = 0.00317f * (float)((i * 29 + 5) % 193) - 0.31f;
    BnThreadPool *workers = bn_tp_create(3);
    assert(workers);
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int cols = sizes[s];
        BnQWeight weight = { weights, BN_GGUF_TENSOR_F32, rows, cols, 1.0f };
        float expected[rows], out[rows + 2], other[rows + 2];
        for (int row = 0; row < rows; row++)
            expected[row] = test_f32_decode_dot_reference(
                weights + (size_t)row * cols, x, cols);
        for (int mode = 0; mode < 2; mode++) {
            BnThreadPool *pool = mode ? workers : NULL;
            for (int i = 0; i < rows + 2; i++) out[i] = other[i] = NAN;
            bn_quant_matvec(out + 1, &weight, x, x_q, pool);
            for (int row = 0; row < rows; row++)
                if (out[row + 1] != expected[row]) {
                    fprintf(stderr, "F32 decode cols=%d row=%d mode=%d: "
                            "got=%a expected=%a\n", cols, row, mode,
                            (double)out[row + 1], (double)expected[row]);
                    break;
                }
            assert(memcmp(out + 1, expected, sizeof(expected)) == 0);
            assert(isnan(out[0]) && isnan(out[rows + 1]));
            bn_quant_matmul(other + 1, &weight, x, 1, x_q, pool);
            assert(memcmp(other + 1, expected, sizeof(expected)) == 0);
            assert(isnan(other[0]) && isnan(other[rows + 1]));
            BnMatvecTask tasks[2] = {
                { out + 1, &weight, NULL, 0 },
                { other + 1, &weight, NULL, 0 }
            };
            for (int i = 0; i < rows + 2; i++) out[i] = other[i] = NAN;
            bn_quant_matvec_batch(tasks, 2, x, x_q, pool);
            assert(memcmp(out + 1, expected, sizeof(expected)) == 0);
            assert(memcmp(other + 1, expected, sizeof(expected)) == 0);
            assert(isnan(out[0]) && isnan(out[rows + 1]));
            assert(isnan(other[0]) && isnan(other[rows + 1]));
        }
        for (int i = 0; i < rows + 2; i++) out[i] = NAN;
        BnF32Ctx ctx = { out + 1, &weight, x };
        bn_quant_f32_avx2_range(&ctx, 3, rows - 2);
        for (int row = 0; row < rows; row++) {
            if (row >= 3 && row < rows - 2)
                assert(out[row + 1] == expected[row]);
            else
                assert(isnan(out[row + 1]));
        }
        assert(isnan(out[0]) && isnan(out[rows + 1]));
    }
    bn_tp_free(workers);
    free(storage);
    free(input);
    free(x_q);
#endif
    printf("PASSED\n");
}

static void test_f32_batch_scheduling(void) {
    printf("test_f32_batch_scheduling... ");
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    const int shapes[][3] = {
        {48, 2560, 128}, {48, 2047, 17}, {48, 2048, 15},
        {48, 2048, 16}, {48, 2049, 16}, {64, 2048, 16},
        {65, 2048, 16}, {48, 16383, 2}, {48, 16384, 2},
        {48, 128, 256}, {137, 33, 9}, {1, 2048, 16}
    };
    BnThreadPool *pools[] = {
        NULL, bn_tp_create(0), bn_tp_create(3), bn_tp_create(7)
    };
    assert(pools[1] && pools[2] && pools[3]);
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        int rows = shapes[si][0], cols = shapes[si][1];
        int n_tokens = shapes[si][2];
        size_t count = (size_t)rows * n_tokens;
        float *w = malloc((size_t)rows * cols * sizeof(*w));
        float *x = malloc((size_t)n_tokens * cols * sizeof(*x));
        float *ref = malloc(count * sizeof(*ref));
        float *out = malloc((count + 2) * sizeof(*out));
        int8_t *x_q = malloc((size_t)cols);
        assert(w && x && ref && out && x_q);
        for (size_t i = 0; i < (size_t)rows * cols; i++)
            w[i] = (float)((int)(i * 37 % 257) - 128) * 0.00391f;
        for (size_t i = 0; i < (size_t)n_tokens * cols; i++)
            x[i] = (float)((int)(i * 29 % 193) - 96) * 0.00783f;
        BnQWeight weight = {w, BN_GGUF_TENSOR_F32, rows, cols, 1.0f};
        BnKQuantFloatMatmulCtx ctx = {ref, &weight, x, n_tokens, cols};
        bn_quant_f32_avx2_matmul_range(&ctx, 0, rows);
        for (size_t p = 0; p < sizeof(pools) / sizeof(pools[0]); p++) {
            for (int repeat = 0; repeat < 3; repeat++) {
                for (size_t i = 0; i < count + 2; i++) out[i] = 12345.0f;
                bn_quant_matmul(out + 1, &weight, x, n_tokens, x_q, pools[p]);
                assert(memcmp(ref, out + 1, count * sizeof(*ref)) == 0);
                assert(out[0] == 12345.0f && out[count + 1] == 12345.0f);
            }
        }
        free(w); free(x); free(ref); free(out); free(x_q);
    }
    for (size_t p = 1; p < sizeof(pools) / sizeof(pools[0]); p++)
        bn_tp_free(pools[p]);
#endif
    printf("PASSED\n");
}

static void test_f32_avx2_batch_reduction_order(void) {
    printf("test_f32_avx2_batch_reduction_order... ");
#if defined(__AVX2__) && !defined(__AVX512F__)
    enum { rows = 5, cols = 5120, n_tokens = 8 };
    float *weights = malloc((size_t)rows * cols * sizeof(*weights));
    float *x = malloc((size_t)n_tokens * cols * sizeof(*x));
    float *expected = malloc((size_t)n_tokens * rows * sizeof(*expected));
    float *actual = malloc((size_t)n_tokens * rows * sizeof(*actual));
    int8_t *x_q = malloc((size_t)cols);
    assert(weights && x && expected && actual && x_q);

    for (int i = 0; i < rows * cols; i++)
        weights[i] = 0.00390625f * (float)((i * 37 + 11) % 257) - 0.5f;
    for (int i = 0; i < n_tokens * cols; i++)
        x[i] = 0.0078125f * (float)((i * 29 + 5) % 193) - 0.75f;

    for (int t = 0; t < n_tokens; t++) {
        for (int row = 0; row < rows; row++) {
            float lanes[8] = {0};
            const float *w = weights + (size_t)row * cols;
            const float *xt = x + (size_t)t * cols;
            for (int col = 0; col < cols; col += 8)
                for (int lane = 0; lane < 8; lane++)
                    lanes[lane] = fmaf(w[col + lane], xt[col + lane],
                                       lanes[lane]);
            float even = (lanes[0] + lanes[4]) +
                         (lanes[2] + lanes[6]);
            float odd = (lanes[1] + lanes[5]) +
                        (lanes[3] + lanes[7]);
            expected[(size_t)t * rows + row] = even + odd;
        }
    }

    BnQWeight weight = {
        weights, BN_GGUF_TENSOR_F32, rows, cols, 1.0f
    };
    bn_quant_matmul(actual, &weight, x, n_tokens, x_q, NULL);
    assert(memcmp(actual, expected,
                  (size_t)n_tokens * rows * sizeof(*actual)) == 0);

    free(weights);
    free(x);
    free(expected);
    free(actual);
    free(x_q);
#endif
    printf("PASSED\n");
}

static void test_bf16_matvec_multi_correctness(void) {
    printf("test_bf16_matvec_multi_correctness... ");

    int rows = 7, cols = 33, n_tasks = 2;
    uint16_t *data1 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));
    uint16_t *data2 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v1 = 0.03125f * ((r * 13 + c * 3) % 31) - 0.5f;
            float v2 = 0.0625f * ((r * 19 + c * 5 + 1) % 17) - 0.375f;
            data1[(size_t)r * cols + c] = test_fp32_to_bf16(v1);
            data2[(size_t)r * cols + c] = test_fp32_to_bf16(v2);
        }
    }

    BnQWeight W1 = { data1, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };
    BnQWeight W2 = { data2, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };

    float X1[33], X2[33];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.04f * ((i * 5 + 1) % 23) - 0.42f;
        X2[i] = 0.03f * ((i * 7 + 4) % 29) - 0.38f;
    }

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[33];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 33];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) < 1e-4f);
        assert(fabsf(out2[i] - ref2[i]) < 1e-4f);
    }

    free(data1);
    free(data2);
    printf("PASSED\n");
}

static void test_mixed_kquant_matvec_batch_correctness(void) {
    printf("test_mixed_kquant_matvec_batch_correctness... ");

    int rows = 5, cols = 256;
    BnBlockQ4K *q4 = (BnBlockQ4K *)calloc((size_t)rows, sizeof(BnBlockQ4K));
    BnBlockQ6K *q6 = (BnBlockQ6K *)calloc((size_t)rows, sizeof(BnBlockQ6K));

    for (int r = 0; r < rows; r++) {
        q4[r].d = 0x3C00;
        q4[r].dmin = 0;
        for (int i = 0; i < 12; i++) q4[r].scales[i] = (uint8_t)((r * 7 + i * 3) & 0x3f);
        for (int i = 0; i < 128; i++) q4[r].qs[i] = (uint8_t)(r * 17 + i * 11);

        q6[r].d = 0x3C00;
        for (int i = 0; i < 16; i++) q6[r].scales[i] = (int8_t)((r * 5 + i * 3) % 17 - 8);
        for (int i = 0; i < 128; i++) q6[r].ql[i] = (uint8_t)(r * 13 + i * 5);
        for (int i = 0; i < 64; i++) q6[r].qh[i] = (uint8_t)(r * 19 + i * 7);
    }

    BnQWeight W4 = { .data = q4, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W6 = { .data = q6, .type = BN_GGUF_TENSOR_Q6_K, .rows = rows, .cols = cols, .scale = 1.0f };

    float x[256];
    for (int i = 0; i < cols; i++)
        x[i] = 0.03125f * ((i * 11 + 3) % 37) - 0.5f;

    float ref4[5], ref6[5], out4[5], out6[5];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref4, &W4, x, x_q_ref, NULL);
    bn_quant_matvec(ref6, &W6, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out4, &W4, NULL, 0 },
         { out6, &W6, NULL, 0 },
    };
    int8_t x_q[256];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out4[i] - ref4[i]) < 1e-4f);
        assert(fabsf(out6[i] - ref6[i]) < 1e-4f);
    }

    free(q4);
    free(q6);
    printf("PASSED\n");
}

static void test_mixed_kquant_matvec_multi_correctness(void) {
    printf("test_mixed_kquant_matvec_multi_correctness... ");

    int rows = 5, cols = 256;
    BnBlockQ4K *q4 = (BnBlockQ4K *)calloc((size_t)rows, sizeof(BnBlockQ4K));
    BnBlockQ6K *q6 = (BnBlockQ6K *)calloc((size_t)rows, sizeof(BnBlockQ6K));

    for (int r = 0; r < rows; r++) {
        q4[r].d = 0x3C00;
        q4[r].dmin = 0;
        for (int i = 0; i < 12; i++) q4[r].scales[i] = (uint8_t)((r * 9 + i * 5) & 0x3f);
        for (int i = 0; i < 128; i++) q4[r].qs[i] = (uint8_t)(r * 23 + i * 3);

        q6[r].d = 0x3C00;
        for (int i = 0; i < 16; i++) q6[r].scales[i] = (int8_t)((r * 7 + i * 5) % 19 - 9);
        for (int i = 0; i < 128; i++) q6[r].ql[i] = (uint8_t)(r * 11 + i * 13);
        for (int i = 0; i < 64; i++) q6[r].qh[i] = (uint8_t)(r * 17 + i * 7);
    }

    BnQWeight W4 = { .data = q4, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W6 = { .data = q6, .type = BN_GGUF_TENSOR_Q6_K, .rows = rows, .cols = cols, .scale = 1.0f };

    float x4[256], x6[256];
    for (int i = 0; i < cols; i++) {
        x4[i] = 0.025f * ((i * 7 + 5) % 41) - 0.45f;
        x6[i] = 0.02f * ((i * 13 + 2) % 43) - 0.42f;
    }

    float ref4[5], ref6[5], out4[5], out6[5];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref4, &W4, x4, x_q_ref, NULL);
    bn_quant_matvec(ref6, &W6, x6, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out4, &W4, x4, NULL },
         { out6, &W6, x6, NULL },
    };
    int8_t x_q_bufs[2 * 256];
    bn_quant_matvec_multi(tasks, 2, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out4[i] - ref4[i]) < 1e-4f);
        assert(fabsf(out6[i] - ref6[i]) < 1e-4f);
    }

    free(q4);
    free(q6);
    printf("PASSED\n");
}

static void fill_q4k_blocks(BnBlockQ4K *q4, int rows, int n_bpr, int seed) {
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ4K *blk = &q4[(size_t)r * n_bpr + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(1 + ((r + b + seed) % 5)));
            blk->dmin = bn_fp32_to_fp16(0.015625f * (float)(1 + ((2 * r + b + seed) % 7)));
            for (int i = 0; i < 12; i++)
                blk->scales[i] = (uint8_t)((r * 13 + b * 17 + i * 5 + seed) & 0x3f);
            for (int i = 0; i < 128; i++)
                blk->qs[i] = (uint8_t)(r * 19 + b * 23 + i * 7 + seed);
        }
    }
}

static void fill_q6k_blocks(BnBlockQ6K *q6, int rows, int n_bpr, int seed) {
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ6K *blk = &q6[(size_t)r * n_bpr + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(1 + ((r + 2 * b + seed) % 5)));
            for (int i = 0; i < 16; i++)
                blk->scales[i] = (int8_t)((r * 7 + b * 11 + i * 3 + seed) % 31 - 15);
            for (int i = 0; i < 128; i++)
                blk->ql[i] = (uint8_t)(r * 11 + b * 13 + i * 5 + seed);
            for (int i = 0; i < 64; i++)
                blk->qh[i] = (uint8_t)(r * 17 + b * 19 + i * 9 + seed);
        }
    }
}

static void test_q6k_scalar_sdot_4row_correctness(void) {
    printf("test_q6k_scalar_sdot_4row_correctness... ");

    const int rows = 7;
    const int cols = 512;
    const int n_bpr = cols / BN_QK_K;
    BnBlockQ6K *blocks =
        (BnBlockQ6K *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    float *x = (float *)malloc((size_t)cols * sizeof(*x));
    int8_t *x_q = (int8_t *)malloc((size_t)cols);
    float *x_d = (float *)malloc((size_t)n_bpr * sizeof(*x_d));
    int16_t *x_bsums =
        (int16_t *)malloc((size_t)n_bpr * 16 * sizeof(*x_bsums));
    assert(blocks && x && x_q && x_d && x_bsums);

    fill_q6k_blocks(blocks, rows, n_bpr, 53);
    for (int i = 0; i < cols; i++)
        x[i] = 0.03125f * (float)((i * 17 + 9) % 47) - 0.625f;
    bn_quant_x_to_q8k_scalar(x, x_q, x_d, x_bsums, cols);

    BnQWeight weight = {
        .data = blocks,
        .type = BN_GGUF_TENSOR_Q6_K,
        .rows = rows,
        .cols = cols,
        .scale = 1.0f,
    };
    float reference[rows];
    float tiled[rows];
    BnKQuantSdotCtx reference_ctx = {
        reference, &weight, x_q, x_d, x_bsums, NULL
    };
    BnKQuantSdotCtx tiled_ctx = {
        tiled, &weight, x_q, x_d, x_bsums, NULL
    };
    bn_quant_q6k_scalar_sdot_range(&reference_ctx, 0, rows);
    bn_quant_q6k_scalar_sdot_4row_range(
        &tiled_ctx, 0, (rows + 3) / 4);
    for (int row = 0; row < rows; row++)
        assert(fabsf(reference[row] - tiled[row]) <
               1e-4f * fmaxf(1.0f, fabsf(reference[row])));

    free(x_bsums);
    free(x_d);
    free(x_q);
    free(x);
    free(blocks);
    printf("PASSED\n");
}

static void test_q6k_neon_pair_reduction_order(void) {
    printf("test_q6k_neon_pair_reduction_order... ");
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int rows = 7;
    const int cols = 768;
    const int n_bpr = cols / BN_QK_K;
    BnBlockQ6K *blocks =
        (BnBlockQ6K *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    float *x = (float *)malloc((size_t)cols * sizeof(*x));
    int8_t *x_q = (int8_t *)malloc((size_t)cols);
    float *x_d = (float *)malloc((size_t)n_bpr * sizeof(*x_d));
    int16_t *x_bsums =
        (int16_t *)malloc((size_t)n_bpr * 16 * sizeof(*x_bsums));
    assert(blocks && x && x_q && x_d && x_bsums);

    fill_q6k_blocks(blocks, rows, n_bpr, 67);
    for (int i = 0; i < cols; i++)
        x[i] = 0.015625f * (float)((i * 29 + 5) % 89) - 0.6875f;
    bn_quant_x_to_q8k(x, x_q, x_d, x_bsums, cols);

    BnQWeight weight = {
        .data = blocks,
        .type = BN_GGUF_TENSOR_Q6_K,
        .rows = rows,
        .cols = cols,
        .scale = 1.0f,
    };
    float paired[rows];
    float single[rows];
    BnKQuantSdotCtx paired_ctx = {
        paired, &weight, x_q, x_d, x_bsums, NULL
    };
    BnKQuantSdotCtx single_ctx = {
        single, &weight, x_q, x_d, x_bsums, NULL
    };
    bn_quant_q6k_neon_sdot_range(&paired_ctx, 0, rows);
    for (int row = 0; row < rows; row++)
        bn_quant_q6k_neon_sdot_range(&single_ctx, row, row + 1);
    assert(memcmp(paired, single, sizeof(paired)) == 0);

    free(x_bsums);
    free(x_d);
    free(x_q);
    free(x);
    free(blocks);
#else
    printf("SKIPPED");
#endif
    printf("PASSED\n");
}

static void test_q4k_unpacked_gemv_batch_policy(void) {
#ifdef __AVX2__
    printf("test_q4k_unpacked_gemv_batch_policy... ");
    /* Exercise a full eight-token weight-reuse tile plus a one-token tail. */
    enum { matrices = 5, max_rows = 143, cols = 768, nb = 3, tokens = 9 };
    BnBlockQ4K blocks[matrices][max_rows * nb];
    BnQWeight weights[matrices];
    const BnQWeight *ptrs[matrices];
    float input[tokens * cols], actual[matrices][tokens * max_rows + 2];
    float expected[matrices][tokens * max_rows], *outputs[matrices];
    int8_t scratch[cols];
    const int rows[matrices] = {137, 143, 1, 4, 7};
    const char *names[] = {"BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH",
        "BN_AVX2_KQUANT_FLOAT", "BN_CPU_REFERENCE_DOT"};
    char *saved[3] = {0};
    for (int i = 0; i < 3; i++) {
        const char *value = getenv(names[i]);
        saved[i] = value ? strdup(value) : NULL;
        assert(!value || saved[i]);
    }
    for (int i = 0; i < matrices; i++) {
        fill_q4k_blocks(blocks[i], rows[i], nb, 13 + i * 17);
        weights[i] = (BnQWeight){blocks[i], BN_GGUF_TENSOR_Q4_K, rows[i], cols, 1.0f};
        ptrs[i] = &weights[i];
        outputs[i] = actual[i] + 1;
    }
    for (int i = 0; i < tokens * cols; i++)
        input[i] = sinf(i * .017f) + cosf(i * .031f);
    const int counts[] = {1, 2, 4, 5};
    for (int mode = -1; mode < 4; mode++) {
        for (int i = 0; i < 3; i++)
            assert(setenv(names[i], mode == i + 1 ? "1" : "0", 1) == 0);
        BnThreadPool *pool = mode < 0 ? NULL : bn_tp_create(3);
        assert(mode < 0 || pool);
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            int count = counts[c];
            for (int i = 0; i < matrices; i++)
                for (int j = 0; j < tokens * max_rows + 2; j++)
                    actual[i][j] = -12345.0f;
            for (int t = 0; t < tokens; t++) {
                BnMatvecTask tasks[matrices];
                for (int i = 0; i < count; i++)
                    tasks[i] = (BnMatvecTask){expected[i] + (size_t)t * rows[i],
                        &weights[i], NULL, 0};
                bn_quant_matvec_batch(tasks, count, input + (size_t)t * cols, scratch, pool);
            }
            bn_quant_matmul_prepared_multi_gemv(outputs, ptrs, NULL,
                count, input, tokens, scratch, pool);
            for (int i = 0; i < count; i++) {
                assert(memcmp(outputs[i], expected[i], (size_t)tokens * rows[i] * sizeof(float)) == 0);
                assert(actual[i][0] == -12345.0f);
                for (int j = tokens * rows[i] + 1; j < tokens * max_rows + 2; j++)
                    assert(actual[i][j] == -12345.0f);
            }
        }
        bn_tp_free(pool);
    }
    for (int i = 0; i < 3; i++) {
        if (saved[i]) {
            assert(setenv(names[i], saved[i], 1) == 0);
            free(saved[i]);
        } else assert(unsetenv(names[i]) == 0);
    }
    printf("PASSED\n");
#endif
}

static void test_q4k_prepared_multi_matches_separate(void) {
#ifdef __AVX2__
    printf("test_q4k_prepared_multi_matches_separate... ");
    enum { count = 3, max_rows = 24, nb = 3, cols = nb * BN_QK_K, max_tokens = 17 };
    const int rows[count] = {16, 24, 15};
    const int token_counts[] = {1, 2, 3, 4, 5, 8, 9, 16, 17};
    BnBlockQ4K blocks[count][max_rows * nb];
    BnQWeight weights[count];
    const BnQWeight *weight_ptrs[count];
    BnPreparedWeight layouts[count] = {{0}};
    SHArena *arenas[count] = {0};
    float input[max_tokens * cols], ref[count][max_tokens * max_rows];
    float actual[count][max_tokens * max_rows];
    float control[count][max_tokens * max_rows];
    float *outputs[count];
    float *controls[count];
    int8_t scratch[cols];
    int8_t canonical_q[max_tokens * cols];
    float canonical_d[max_tokens * nb];
    int16_t canonical_sums[max_tokens * nb * 16];
    for (int i = 0; i < max_tokens * cols; i++)
        input[i] = sinf((float)i * 0.17f) * 1.3f;
    for (int t = 0; t < max_tokens; t++)
        bn_quant_x_to_q8k(input + t * cols, canonical_q + t * cols,
                          canonical_d + t * nb, canonical_sums + t * nb * 16, cols);
    for (int i = 0; i < count; i++) {
        fill_q4k_blocks(blocks[i], rows[i], nb, 3 + i * 19);
        weights[i] = (BnQWeight){blocks[i], BN_GGUF_TENSOR_Q4_K, rows[i], cols, 1.0f};
        weight_ptrs[i] = &weights[i];
        outputs[i] = actual[i];
        controls[i] = control[i];
        size_t bytes = bn_quant_prepared_qweight_size(&weights[i], NULL);
        if (bytes) {
            arenas[i] = sh_arena_create(bytes);
            assert(arenas[i]);
            assert(bn_quant_prepare_qweight(&layouts[i], &weights[i], arenas[i]) == 0);
        }
    }
    BnThreadPool *workers = bn_tp_create(3);
    assert(workers);
    for (int threaded = 0; threaded < 2; threaded++) {
        BnThreadPool *pool = threaded ? workers : NULL;
        for (int mask = 0; mask < 8; mask++) {
            const BnPreparedWeight *prepared[count];
            for (int i = 0; i < count; i++)
                prepared[i] = (mask & (1 << i)) && arenas[i] ? &layouts[i] : NULL;
            for (size_t t = 0; t < sizeof(token_counts) / sizeof(token_counts[0]); t++) {
                int tokens = token_counts[t];
                for (int i = 0; i < count; i++)
                    bn_quant_matmul_prepared(ref[i], &weights[i], prepared[i],
                                             input, tokens, scratch, pool);
                for (int matrices = 1; matrices <= count; matrices++) {
                    bn_quant_matmul_prepared_multi(outputs, weight_ptrs, prepared,
                        matrices, input, tokens, scratch, pool);
                    bn_quant_matmul_prepared_kquant_input_multi(
                        controls, weight_ptrs, prepared, matrices, tokens,
                        canonical_q, canonical_d, canonical_sums, input, pool);
                    for (int i = 0; i < matrices; i++) {
                        size_t bytes = (size_t)tokens * rows[i] * sizeof(float);
                        assert(memcmp(ref[i], actual[i], bytes) == 0);
                        /* Public prepared-input calls have every canonical
                         * row initialized, independently of panel-only setup. */
                        if (tokens > 1)
                            assert(memcmp(control[i], actual[i], bytes) == 0);
                    }
                    for (int token = 0; token < tokens; token++) {
                        BnMatvecTask tasks[count];
                        for (int i = 0; i < matrices; i++)
                            tasks[i] = (BnMatvecTask){
                                control[i] + (size_t)token * rows[i],
                                &weights[i], prepared[i], 0};
                        bn_quant_matvec_batch(tasks, matrices,
                            input + (size_t)token * cols, scratch, pool);
                    }
                    for (int i = 0; i < matrices; i++)
                        for (int j = 0; j < tokens * rows[i]; j++)
                            actual[i][j] = NAN;
                    bn_quant_matmul_prepared_multi_gemv(
                        outputs, weight_ptrs, prepared, matrices,
                        input, tokens, scratch, pool);
                    for (int i = 0; i < matrices; i++)
                        assert(memcmp(control[i], actual[i],
                            (size_t)tokens * rows[i] * sizeof(float)) == 0);
                }
            }
        }
    }
    bn_tp_free(workers);
    for (int i = 0; i < count; i++) sh_arena_free(arenas[i]);
    printf("PASSED\n");
#endif
}

static void test_matmul_gemv_row_scheduling(void) {
#if defined(__AVX2__)
    printf("test_matmul_gemv_row_scheduling... ");
    // 2112 / (4 * 16) = 33: chunk boundaries deliberately split row groups.
    enum { rows = 2112, cols = BN_QK_K, tokens = 5 };
    BnBlockQ4K *blocks = malloc((size_t)rows * sizeof(*blocks));
    float *actual = malloc((size_t)tokens * rows * sizeof(float));
    float *expected = malloc((size_t)tokens * rows * sizeof(float));
    assert(blocks && actual && expected);
    fill_q4k_blocks(blocks, rows, 1, 37);
    BnQWeight weight = {blocks, BN_GGUF_TENSOR_Q4_K, rows, cols, 1.0f};
    BnPreparedWeight layout = {0};
    SHArena *arena = sh_arena_create(bn_quant_prepared_qweight_size(&weight, NULL));
    assert(arena && bn_quant_prepare_qweight(&layout, &weight, arena) == 0);
    float input[tokens * cols];
    int8_t scratch[cols];
    for (int i = 0; i < tokens * cols; i++) input[i] = sinf((float)i * 0.13f);
    for (int t = 0; t < tokens; t++) {
        BnMatvecTask task = {expected + t * rows, &weight, &layout, 0};
        bn_quant_matvec_batch(&task, 1, input + t * cols, scratch, NULL);
    }
    BnThreadPool *pool = bn_tp_create(15);
    assert(pool);
    const BnQWeight *weights[] = {&weight};
    const BnPreparedWeight *prepared[] = {&layout};
    float *outputs[] = {actual};
    for (int iteration = 0; iteration < 4; iteration++) {
        for (int i = 0; i < tokens * rows; i++) actual[i] = NAN;
        bn_quant_matmul_prepared_multi_gemv(outputs, weights, prepared, 1,
            input, tokens, scratch, pool);
        assert(memcmp(actual, expected, (size_t)tokens * rows * sizeof(float)) == 0);
    }
    bn_quant_matmul_prepared(expected, &weight, &layout, input, tokens, scratch, NULL);
    for (int multi = 0; multi < 2; multi++) {
        for (int i = 0; i < tokens * rows; i++) actual[i] = NAN;
        if (multi)
            bn_quant_matmul_prepared_multi(outputs, weights, prepared, 1,
                input, tokens, scratch, pool);
        else
            bn_quant_matmul_prepared(actual, &weight, &layout, input,
                tokens, scratch, pool);
        assert(memcmp(actual, expected, (size_t)tokens * rows * sizeof(float)) == 0);
    }
    bn_tp_free(pool);
    sh_arena_free(arena);
    free(blocks);
    free(actual);
    free(expected);
    printf("PASSED\n");
#endif
}

static void test_q8_matmul_gemv_order(void) {
    printf("test_q8_matmul_gemv_order... ");
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q8_0,
        BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL));
    assert(!bn_quant_format_has_cap(BN_GGUF_TENSOR_Q4_K,
        BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL));
    enum { rows = 19, cols = 512, tokens = 65, matrices = 2 };
    BnBlockQ8_0 blocks[matrices][rows * (cols / 32)];
    float x[tokens * cols], actual[matrices][tokens * rows + 2];
    float expected[matrices][tokens * rows];
    int8_t scratch[cols];
    BnQWeight weights[matrices];
    const BnQWeight *ptrs[matrices];
    float *out[matrices];
    /* Cover finite half-scale boundaries as well as ordinary model scales. */
    static const uint16_t half_scales[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff,
        0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff
    };
    for (int m = 0; m < matrices; m++) {
        for (int b = 0; b < rows * (cols / 32); b++) {
            blocks[m][b].d = bn_fp32_to_fp16(.0013f * (1 + b % 17));
            if (m == 1)
                blocks[m][b].d = half_scales[b %
                    (sizeof(half_scales) / sizeof(half_scales[0]))];
            for (int j = 0; j < 32; j++)
                blocks[m][b].qs[j] = (int8_t)((b * 17 + j * 13 + m) % 255 - 127);
        }
        weights[m] = (BnQWeight){blocks[m], BN_GGUF_TENSOR_Q8_0,
                                rows, cols, 1.0f};
        ptrs[m] = &weights[m];
        out[m] = actual[m] + 1;
    }
    for (int i = 0; i < tokens * cols; i++) x[i] = sinf(i * .017f);
    BnThreadPool *workers = bn_tp_create(3);
    assert(workers);
    const int counts[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        31, 32, 33, 63, 64, 65
    };
    for (int threaded = 0; threaded < 2; threaded++) {
        BnThreadPool *pool = threaded ? workers : NULL;
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            int nt = counts[c];
            for (int m = 0; m < matrices; m++) {
                for (int i = 0; i < tokens * rows + 2; i++) actual[m][i] = NAN;
                for (int t = 0; t < nt; t++) {
                    BnMatvecTask task = {expected[m] + t * rows, &weights[m], NULL, 0};
                    bn_quant_matvec_batch(&task, 1, x + t * cols, scratch, pool);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                    /* Independent non-VNNI reference for every tile/tail. */
                    float scales[cols / 32], avx2[rows];
                    bn_quant_x_to_q8_blocks(x + t * cols, scratch, scales, cols);
                    BnQ8SdotCtx reference = {avx2, &weights[m], scratch, scales, NULL};
                    bn_quant_q8_avx2_4row_range(&reference, 0, (rows + 3) / 4);
                    assert(memcmp(avx2, expected[m] + t * rows, sizeof(avx2)) == 0);
#endif
                }
            }
            bn_quant_matmul_prepared_multi_gemv(out, ptrs, NULL,
                matrices, x, nt, scratch, pool);
            for (int m = 0; m < matrices; m++) {
                assert(memcmp(out[m], expected[m], (size_t)nt * rows * sizeof(float)) == 0);
                assert(isnan(actual[m][0]));
                for (int i = nt * rows + 1; i < tokens * rows + 2; i++)
                    assert(isnan(actual[m][i]));
            }
        }
    }
    bn_tp_free(workers);
    printf("PASSED\n");
}

static void test_q8_matmul_row_scheduling(void) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    printf("test_q8_matmul_row_scheduling... ");
    enum { rows = 2113, cols = 96, tokens = 17 };
    BnBlockQ8_0 *blocks = malloc((size_t)rows * (cols / 32) * sizeof(*blocks));
    float *expected = malloc((size_t)tokens * rows * sizeof(float));
    float *actual = malloc(((size_t)tokens * rows + 2) * sizeof(float));
    float x[tokens * cols];
    int8_t scratch[cols];
    assert(blocks && expected && actual);
    for (int b = 0; b < rows * (cols / 32); b++) {
        blocks[b].d = bn_fp32_to_fp16(.0013f * (1 + b % 17));
        for (int j = 0; j < 32; j++)
            blocks[b].qs[j] = (int8_t)((b * 17 + j * 13) % 255 - 127);
    }
    for (int i = 0; i < tokens * cols; i++) x[i] = sinf(i * .017f);
    BnQWeight weight = {blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f};
    for (int t = 0; t < tokens; t++) {
        BnMatvecTask task = {expected + (size_t)t * rows, &weight, NULL, 0};
        bn_quant_matvec_batch(&task, 1, x + t * cols, scratch, NULL);
    }
    /* 2113 rows with 16 workers produces 33-row chunks, cutting through
     * four-row kernel groups; the final group has only one valid row. */
    BnThreadPool *pool = bn_tp_create(16);
    assert(pool);
    const int counts[] = {2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 17};
    for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
        for (int rep = 0; rep < 3; rep++) {
            for (int i = 0; i < tokens * rows + 2; i++) actual[i] = NAN;
            bn_quant_matmul(actual + 1, &weight, x, counts[c], scratch, pool);
            assert(memcmp(actual + 1, expected,
                (size_t)counts[c] * rows * sizeof(float)) == 0);
            assert(isnan(actual[0]));
            for (int i = counts[c] * rows + 1; i < tokens * rows + 2; i++)
                assert(isnan(actual[i]));
        }
    }
    bn_tp_free(pool);
    free(blocks); free(expected); free(actual);
    printf("PASSED\n");
#endif
}

static void test_matmul_gemv_fallback(void) {
    printf("test_matmul_gemv_fallback... ");
    enum { count = 5, cols = 35, rows = 9, tokens = 5 };
    float data[count][rows * cols], x[tokens * cols];
    float actual[count][tokens * rows], expected[count][tokens * rows];
    BnQWeight weights[count];
    const BnQWeight *ptrs[count];
    float *outputs[count];
    int8_t scratch[cols];
    for (int i = 0; i < tokens * cols; i++)
        x[i] = sinf((float)i * 0.17f);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < rows * cols; j++)
            data[i][j] = cosf((float)(j + i * 37) * 0.23f);
        weights[i] = (BnQWeight){data[i], BN_GGUF_TENSOR_F32, rows, cols, 1.0f};
        ptrs[i] = &weights[i];
        outputs[i] = actual[i];
    }
    BnThreadPool *workers = bn_tp_create(3);
    assert(workers);
    for (int threaded = 0; threaded < 2; threaded++) {
        BnThreadPool *pool = threaded ? workers : NULL;
        for (int n = 1; n <= count; n++) {
            for (int t = 0; t < tokens; t++)
                for (int i = 0; i < n; i++) {
                    BnMatvecTask task = {expected[i] + t * rows, &weights[i], NULL, 0};
                    bn_quant_matvec_batch(&task, 1, x + t * cols, scratch, pool);
                }
            bn_quant_matmul_prepared_multi_gemv(outputs, ptrs, NULL, n,
                x, tokens, scratch, pool);
            for (int i = 0; i < n; i++)
                assert(memcmp(actual[i], expected[i], sizeof(actual[i])) == 0);
        }
    }
    bn_tp_free(workers);
    printf("PASSED\n");
}

static void test_kquant_prepared_kquant_input_matmul_correctness(void) {
    printf("test_kquant_prepared_kquant_input_matmul_correctness... ");

    /* Exercise a four-token repacked GEMM plus the two-token GEMV tail. */
    int rows = 16, cols = 512, n_tokens = 6;
    int n_bpr = cols / BN_QK_K;
    BnBlockQ4K *q4a = (BnBlockQ4K *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4K));
    BnBlockQ4K *q4b = (BnBlockQ4K *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4K));
    BnBlockQ6K *q6 = (BnBlockQ6K *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ6K));
    assert(q4a && q4b && q6);

    fill_q4k_blocks(q4a, rows, n_bpr, 3);
    fill_q4k_blocks(q4b, rows, n_bpr, 29);
    fill_q6k_blocks(q6, rows, n_bpr, 7);

    BnQWeight W4a = { .data = q4a, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W4b = { .data = q4b, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W6 = { .data = q6, .type = BN_GGUF_TENSOR_Q6_K, .rows = rows, .cols = cols, .scale = 1.0f };

    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    int8_t *x_q = (int8_t *)malloc((size_t)n_tokens * cols);
    int8_t *force_x_q = (int8_t *)malloc((size_t)cols);
    float *x_d = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
    int16_t *x_bsums = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
    float *ref4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *ref6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *float4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *float6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *batch_float4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *batch_float6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *force4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *force6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4b = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4_prepared = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4b_prepared = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4_single_prepared =
        (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    assert(X && x_q && force_x_q && x_d && x_bsums && ref4 && ref6 &&
           float4 && float6 && batch_float4 && batch_float6 &&
           force4 && force6 &&
           out4 && out4b && out6 && out4_prepared && out4b_prepared &&
           out4_single_prepared);

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < cols; i++)
            X[(size_t)t * cols + i] = 0.0275f * (float)((t * 17 + i * 11 + 5) % 43) - 0.55f;
        bn_quant_x_to_q8k(X + (size_t)t * cols,
                          x_q + (size_t)t * cols,
                          x_d + (size_t)t * n_bpr,
                          x_bsums + (size_t)t * n_bpr * 16, cols);
        bn_quant_matvec(ref4 + (size_t)t * rows, &W4a, X + (size_t)t * cols,
                        x_q + (size_t)t * cols, NULL);
        bn_quant_matvec(ref6 + (size_t)t * rows, &W6, X + (size_t)t * cols,
                        x_q + (size_t)t * cols, NULL);
        BnQ4KCtx float4_ctx = {
            float4 + (size_t)t * rows, &W4a, X + (size_t)t * cols
        };
        BnQ4KCtx float6_ctx = {
            float6 + (size_t)t * rows, &W6, X + (size_t)t * cols
        };
        bn_quant_get_float_kernel(W4a.type)(&float4_ctx, 0, rows);
        bn_quant_get_float_kernel(W6.type)(&float6_ctx, 0, rows);
        BnMatvecTask force_tasks[2] = {
            { force4 + (size_t)t * rows, &W4a, NULL,
              BN_MATVEC_TASK_FORCE_FLOAT_KQUANT },
            { force6 + (size_t)t * rows, &W6, NULL,
              BN_MATVEC_TASK_FORCE_FLOAT_KQUANT },
        };
        bn_quant_matvec_batch(force_tasks, 2, X + (size_t)t * cols,
                              force_x_q, NULL);
    }

    bn_quant_matmul_float_x(batch_float4, &W4a, X, n_tokens, NULL);
    bn_quant_matmul_float_x(batch_float6, &W6, X, n_tokens, NULL);
    assert(memcmp(batch_float4, float4,
                  (size_t)n_tokens * rows * sizeof(float)) == 0);
    assert(memcmp(batch_float6, float6,
                  (size_t)n_tokens * rows * sizeof(float)) == 0);

    bn_quant_matmul_prepared_kquant_input(
        out4, &W4a, n_tokens, x_q, x_d, x_bsums, X, NULL);
    bn_quant_matmul_prepared_kquant_input(
        out6, &W6, n_tokens, x_q, x_d, x_bsums, X, NULL);

    float prepared_max_diff = 0.0f;
    for (int i = 0; i < rows * n_tokens; i++) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        assert(out4[i] == ref4[i]);
#else
        assert(fabsf(out4[i] - ref4[i]) < 1e-3f);
#endif
        assert(fabsf(out6[i] - ref6[i]) < 1e-3f);
    }

    float *multi_out[3] = { out4, out4b, out6 };
    const BnQWeight *multi_w[3] = { &W4a, &W4b, &W6 };
    bn_quant_matmul_prepared_kquant_input_multi(
        multi_out, multi_w, NULL, 3, n_tokens, x_q, x_d, x_bsums, X, NULL);

#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
    BnPreparedWeight prepared4a = { 0 };
    BnPreparedWeight prepared4b = { 0 };
    BnPreparedWeightKind kind4a = BN_PREPARED_WEIGHT_NONE;
    BnPreparedWeightKind kind4b = BN_PREPARED_WEIGHT_NONE;
    size_t prep_bytes4a = bn_quant_prepared_qweight_size(&W4a, &kind4a);
    size_t prep_bytes4b = bn_quant_prepared_qweight_size(&W4b, &kind4b);
    assert(kind4a == BN_PREPARED_WEIGHT_Q4_K_SCALES);
    assert(kind4b == BN_PREPARED_WEIGHT_Q4_K_SCALES);
    SHArena *prep_arena =
        sh_arena_create(prep_bytes4a + prep_bytes4b + 4 * SH_ARENA_ALIGN);
    assert(prep_arena != NULL);
    assert(bn_quant_prepare_qweight(&prepared4a, &W4a, prep_arena) == 0);
    assert(bn_quant_prepare_qweight(&prepared4b, &W4b, prep_arena) == 0);
    assert(prepared4a.kind == BN_PREPARED_WEIGHT_Q4_K_SCALES);
    assert(prepared4b.kind == BN_PREPARED_WEIGHT_Q4_K_SCALES);
    float *prepared_out[2] = { out4_prepared, out4b_prepared };
    const BnQWeight *prepared_w[2] = { &W4a, &W4b };
    const BnPreparedWeight *prepared[2] = { &prepared4a, &prepared4b };
    bn_quant_matmul_prepared_kquant_input_multi(
        prepared_out, prepared_w, prepared, 2, n_tokens, x_q, x_d, x_bsums,
        X, NULL);
    bn_quant_matmul_prepared(out4_single_prepared, &W4a, &prepared4a, X,
                             4, force_x_q, NULL);
    assert(memcmp(out4_single_prepared, out4_prepared,
                  (size_t)4 * rows * sizeof(float)) == 0);
#endif

    for (int i = 0; i < rows * n_tokens; i++) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        assert(out4[i] == ref4[i]);
#else
        assert(fabsf(out4[i] - ref4[i]) < 1e-3f);
#endif
        assert(fabsf(out6[i] - ref6[i]) < 1e-3f);
        assert(fabsf(float4[i] - force4[i]) < 1e-6f);
        assert(fabsf(float6[i] - force6[i]) < 1e-6f);
#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
        float diff4a = fabsf(out4_prepared[i] - out4[i]);
        float diff4b = fabsf(out4b_prepared[i] - out4b[i]);
        if (diff4a > prepared_max_diff) prepared_max_diff = diff4a;
        if (diff4b > prepared_max_diff) prepared_max_diff = diff4b;
#endif
    }
#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
    assert(prepared_max_diff < 1e-3f);
#endif

    free(q4a); free(q4b); free(q6);
    free(X); free(x_q); free(force_x_q); free(x_d); free(x_bsums);
    free(ref4); free(ref6); free(float4); free(float6);
    free(batch_float4); free(batch_float6);
    free(force4); free(force6);
    free(out4); free(out4b); free(out6);
    free(out4_prepared); free(out4b_prepared); free(out4_single_prepared);
#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
    sh_arena_free(prep_arena);
#endif
    printf("PASSED\n");
}

static void test_q6k_prepared_matmul_correctness(void) {
    printf("test_q6k_prepared_matmul_correctness... ");

#if defined(__AVX2__) || \
    (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
    int rows = 2048, cols = 4096, n_tokens = 4;
    int n_bpr = cols / BN_QK_K;
    BnBlockQ6K *q6 =
        (BnBlockQ6K *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ6K));
    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    int8_t *x_q = (int8_t *)malloc((size_t)n_tokens * cols);
    float *x_d =
        (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
    int16_t *x_bsums =
        (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
    float *raw = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *prepared_out =
        (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *prepared_input_out =
        (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    assert(q6 && X && x_q && x_d && x_bsums &&
           raw && prepared_out && prepared_input_out);

    fill_q6k_blocks(q6, rows, n_bpr, 41);
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < cols; i++) {
            X[(size_t)t * cols + i] =
                0.01875f * (float)((t * 19 + i * 13 + 7) % 47) - 0.42f;
        }
        bn_quant_x_to_q8k(X + (size_t)t * cols,
                          x_q + (size_t)t * cols,
                          x_d + (size_t)t * n_bpr,
                          x_bsums + (size_t)t * n_bpr * 16, cols);
    }

    BnQWeight W6 = {
        .data = q6, .type = BN_GGUF_TENSOR_Q6_K,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t prep_bytes = bn_quant_prepared_qweight_size(&W6, &kind);
    assert(kind == BN_PREPARED_WEIGHT_Q6_K_X8);
    assert(prep_bytes > 0);
    SHArena *arena = sh_arena_create(prep_bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = { 0 };
    assert(bn_quant_prepare_qweight(&prepared, &W6, arena) == 0);
    assert(prepared.kind == BN_PREPARED_WEIGHT_Q6_K_X8);

    bn_quant_matmul_prepared(raw, &W6, NULL, X, n_tokens, x_q, NULL);
    bn_quant_matmul_prepared(prepared_out, &W6, &prepared, X, n_tokens,
                             x_q, NULL);

    float *outs[1] = { prepared_input_out };
    const BnQWeight *weights[1] = { &W6 };
    const BnPreparedWeight *prepared_weights[1] = { &prepared };
    bn_quant_matmul_prepared_kquant_input_multi(
        outs, weights, prepared_weights, 1, n_tokens, x_q, x_d, x_bsums,
        X, NULL);

    for (int i = 0; i < rows * n_tokens; i++) {
        assert(prepared_out[i] == raw[i]);
        float diff = fabsf(prepared_input_out[i] - raw[i]);
        assert(diff < 1e-3f);
    }

    free(q6);
    free(X);
    free(x_q);
    free(x_d);
    free(x_bsums);
    free(raw);
    free(prepared_out);
    free(prepared_input_out);
    sh_arena_free(arena);
    printf("PASSED\n");
#else
    printf("SKIPPED\n");
#endif
}

static void test_q6k_avx512_row_pair_exact(void) {
    printf("test_q6k_avx512_row_pair_exact... ");
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    enum { max_rows = 137, max_nb = 38, max_tokens = 33 };
    const int row_counts[] = {4, 136, 137};
    const int block_counts[] = {1, 3, 38};
    const int counts[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33};
    BnBlockQ6K *data = malloc((size_t)max_rows * max_nb * sizeof(*data));
    int8_t *input = malloc((size_t)max_tokens * max_nb * BN_QK_K + 1);
    float *scales = malloc((size_t)max_tokens * max_nb * sizeof(*scales));
    int16_t *bsums = malloc((size_t)max_tokens * max_nb * 16 * sizeof(*bsums));
    float expected[max_tokens * max_rows], actual[max_tokens * max_rows + 2];
    assert(data && input && scales && bsums);
    int8_t *q = input + 1;
    for (int i = 0; i < max_tokens * max_nb * BN_QK_K; i++)
        q[i] = (int8_t)((i * 17) % 256 - 128);
    for (int i = 0; i < max_tokens * max_nb; i++) scales[i] = .001f * (1 + i % 7);
    for (int i = 0; i < max_tokens * max_nb * 16; i++) {
        int sum = 0;
        for (int j = 0; j < 16; j++) sum += q[i * 16 + j];
        bsums[i] = (int16_t)sum;
    }
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    for (size_t r = 0; r < sizeof(row_counts) / sizeof(row_counts[0]); r++) {
        int rows = row_counts[r];
        for (size_t b = 0; b < sizeof(block_counts) / sizeof(block_counts[0]); b++) {
            int nb = block_counts[b], cols = nb * BN_QK_K;
            fill_q6k_blocks(data, rows, nb, 73);
            for (int j = 0; j < rows * nb; j++)
                for (int s = 0; s < 16; s++)
                    data[j].scales[s] = (int8_t)((j * 17 + s * 31) % 256 - 128);
            BnQWeight w = {data, BN_GGUF_TENSOR_Q6_K, rows, cols, 1};
            for (int t = 0; t < max_tokens; t++) {
                BnKQuantSdotCtx one = {expected + t * rows, &w, q + t * cols,
                    scales + t * nb, bsums + t * nb * 16, NULL};
                bn_quant_q6k_avx2_4row_range(&one, 0, (rows + 3) / 4);
            }
            for (size_t n = 0; n < sizeof(counts) / sizeof(counts[0]); n++) {
                BnKQuantMatmulCtx ctx = {actual + 1, &w, q, scales, bsums,
                    counts[n], cols, NULL, NULL};
                for (int mode = 0; mode < 2; mode++) {
                    for (int i = 0; i < max_tokens * max_rows + 2; i++) actual[i] = NAN;
                    BnTPTask task = {bn_quant_q6k_avx2_sdot_matmul_4row_range,
                        &ctx, (rows + 3) / 4};
                    bn_tp_dispatch(mode ? pool : NULL, &task, 1);
                    int written = counts[n] * rows;
                    assert(memcmp(actual + 1, expected, (size_t)written * sizeof(float)) == 0);
                    assert(isnan(actual[0]));
                    for (int i = written + 1; i < max_tokens * max_rows + 2; i++) assert(isnan(actual[i]));
                }
                if (rows > 8) {
                    for (int i = 0; i < max_tokens * max_rows + 2; i++) actual[i] = NAN;
                    int end_group = rows / 4 - 1;
                    bn_quant_q6k_avx2_sdot_matmul_4row_range(&ctx, 1, end_group);
                    for (int t = 0; t < counts[n]; t++) for (int row = 0; row < rows; row++) {
                        if (row >= 4 && row < end_group * 4)
                            assert(actual[1 + t * rows + row] == expected[t * rows + row]);
                        else assert(isnan(actual[1 + t * rows + row]));
                    }
                }
            }
        }
    }
    bn_tp_free(pool);
    free(data); free(input); free(scales); free(bsums);
#endif
    printf("PASSED\n");
}

static void test_q6k_matmul_row_scheduling(void) {
#if defined(__AVX2__)
    printf("test_q6k_matmul_row_scheduling... ");
    /* 2113 / (4 * 16) = 33: split group boundaries and a one-row tail. */
    enum { rows = 2113, cols = 768, blocks = cols / BN_QK_K, tokens = 17 };
    BnBlockQ6K *data = malloc((size_t)rows * blocks * sizeof(*data));
    float *expected = malloc((size_t)tokens * rows * sizeof(float));
    float *actual = malloc(((size_t)tokens * rows + 1) * sizeof(float));
    float *second = malloc(((size_t)tokens * rows + 1) * sizeof(float));
    assert(data && expected && actual && second);
    fill_q6k_blocks(data, rows, blocks, 59);
    float input[tokens * cols], scales[tokens * blocks];
    int8_t quantized[tokens * cols], scratch[cols];
    int16_t sums[tokens * blocks * 16];
    for (int i = 0; i < tokens * cols; i++)
        input[i] = sinf((float)i * 0.19f);
    for (int t = 0; t < tokens; t++)
        bn_quant_x_to_q8k(input + t * cols, quantized + t * cols,
            scales + t * blocks, sums + t * blocks * 16, cols);
    BnQWeight weight = { data, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f };
    const BnQWeight *weights[] = { &weight, &weight };
    float *outputs[] = { actual, second };
    BnThreadPool *pool = bn_tp_create(15);
    assert(pool);
    const int counts[] = { 2, 5, 16, 17 };
    for (size_t k = 0; k < sizeof(counts) / sizeof(counts[0]); k++) {
        int count = counts[k];
        size_t len = (size_t)count * rows;
        BnKQuantMatmulCtx reference = {
            expected, &weight, quantized, scales, sums, count, cols, NULL, NULL
        };
        bn_quant_q6k_avx2_sdot_matmul_4row_range(
            &reference, 0, (rows + 3) / 4);
        for (int mode = 0; mode < 3; mode++) {
            for (size_t i = 0; i <= len; i++) actual[i] = second[i] = NAN;
            if (mode == 0)
                bn_quant_matmul_prepared(actual, &weight, NULL, input,
                    count, scratch, pool);
            else if (mode == 1)
                bn_quant_matmul_prepared_kquant_input(actual, &weight,
                    count, quantized, scales, sums, input, pool);
            else
                bn_quant_matmul_prepared_kquant_input_multi(outputs, weights,
                    NULL, 2, count, quantized, scales, sums, input, pool);
            assert(memcmp(actual, expected, len * sizeof(float)) == 0);
            assert(isnan(actual[len]));
            if (mode == 2) {
                assert(memcmp(second, expected, len * sizeof(float)) == 0);
                assert(isnan(second[len]));
            }
        }
    }
    bn_tp_free(pool);
    free(second);
    free(actual);
    free(expected);
    free(data);
    printf("PASSED\n");
#endif
}

static void test_activation_quant_rounding(void) {
    printf("test_activation_quant_rounding... ");

    float x[BN_QK_K] = {0};
    int8_t x_i8[BN_QK_K];
    int8_t x_q8[32];
    int8_t x_q8k[BN_QK_K];
    float x_q8_scales[1];
    float x_d[1];
    int16_t x_bsums[16];

    x[0] = 127.0f;
    x[1] = 0.5f;
    x[2] = -0.5f;
    x[3] = 1.5f;
    x[4] = -1.5f;
    x[5] = 2.5f;
    x[6] = -2.5f;

    float scale = bn_quant_x_to_i8(x, x_i8, BN_QK_K);
    assert(fabsf(scale - 1.0f) < 1e-6f);
    assert(x_i8[0] == 127);
    assert(x_i8[1] == 1);
    assert(x_i8[2] == -1);
    assert(x_i8[3] == 2);
    assert(x_i8[4] == -2);
    assert(x_i8[5] == 3);
    assert(x_i8[6] == -3);

    bn_quant_x_to_q8_blocks(x, x_q8, x_q8_scales, 32);
    assert(fabsf(x_q8_scales[0] - 1.0f) < 1e-6f);
    assert(x_q8[0] == 127);
    assert(x_q8[1] == 0);
    assert(x_q8[2] == 0);
    assert(x_q8[3] == 2);
    assert(x_q8[4] == -2);
    assert(x_q8[5] == 2);
    assert(x_q8[6] == -2);

    memset(x, 0, sizeof(x));
    x[0] = 796.28125f;
    x[1] = 310.361572265625f;
    bn_quant_x_to_q8_blocks(x, x_q8, x_q8_scales, 32);
    assert(x_q8[0] == 127);
    /* llama.cpp forms the reciprocal as 127 / amax, making this an exact tie. */
    assert(x_q8[1] == 50);

    memset(x, 0, sizeof(x));
    x[0] = 127.0f;
    x[1] = 0.5f;
    x[2] = -0.5f;
    x[3] = 1.5f;
    x[4] = -1.5f;
    x[5] = 2.5f;
    x[6] = -2.5f;
    bn_quant_x_to_q8k(x, x_q8k, x_d, x_bsums, BN_QK_K);
    assert(fabsf(x_d[0] + 1.0f) < 1e-6f);
    assert(x_q8k[0] == -127);
    assert(x_q8k[1] == 0);
    assert(x_q8k[2] == 0);
    assert(x_q8k[3] == -2);
    assert(x_q8k[4] == 2);
    assert(x_q8k[5] == -2);
    assert(x_q8k[6] == 2);

    enum { test_blocks = 4096, test_values = test_blocks * 32 };
    float *values = (float *)malloc((size_t)test_values * sizeof(float));
    int8_t *actual = (int8_t *)malloc((size_t)test_values);
    int8_t *expected = (int8_t *)malloc((size_t)test_values);
    float *actual_scales =
        (float *)malloc((size_t)test_blocks * sizeof(float));
    float *expected_scales =
        (float *)malloc((size_t)test_blocks * sizeof(float));
    assert(values && actual && expected && actual_scales && expected_scales);
    uint32_t state = 0x6d2b79f5u;
    for (int i = 0; i < test_values; i++) {
        state = state * 1664525u + 1013904223u;
        int mantissa = (int)((state >> 8) & 0xffffu) - 32768;
        int exponent = (int)((state >> 24) & 7u) - 3;
        values[i] = ldexpf((float)mantissa / 257.0f, exponent);
    }
    bn_quant_x_to_q8_blocks(values, actual, actual_scales, test_values);
    test_q8_blocks_scalar_reference(
        values, expected, expected_scales, test_values);
    assert(memcmp(actual, expected, (size_t)test_values) == 0);
    assert(memcmp(actual_scales, expected_scales,
                  (size_t)test_blocks * sizeof(float)) == 0);
    free(values);
    free(actual);
    free(expected);
    free(actual_scales);
    free(expected_scales);

    printf("PASSED\n");
}

static void test_mxfp4_x86_exact(void) {
    printf("test_mxfp4_x86_exact... ");
#ifdef __AVX2__
    enum { rows = 137, max_nb = 129 };
    const int counts[] = {1, 2, 3, 7, 96, 129};
    BnBlockMXFP4 *blocks = malloc((size_t)rows * max_nb * sizeof(*blocks));
    int8_t *input = malloc(max_nb * 32 + 1);
    float scales[max_nb], expected[rows], actual[rows + 2];
    assert(blocks && input);
    int8_t *x = input + 1;
    for (int b = 0; b < rows * max_nb; b++) {
        blocks[b].e = (uint8_t)(b * 37);
        for (int i = 0; i < 16; i++) blocks[b].qs[i] = (uint8_t)(b * 19 + i * 31);
    }
    for (int i = 0; i < max_nb * 32; i++) x[i] = (int8_t)((i * 17) % 256 - 128);
    for (int b = 0; b < max_nb; b++) scales[b] = (float)(1 + b % 7) * 1e-9f;
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    for (size_t n = 0; n < sizeof(counts) / sizeof(counts[0]); n++) {
        BnQWeight w = {blocks, BN_GGUF_TENSOR_MXFP4, rows, counts[n] * 32, 1.0f};
        BnQ4SdotCtx ref = {expected, &w, x, scales, NULL};
        BnQ4SdotCtx got = {actual + 1, &w, x, scales, NULL};
        bn_quant_mxfp4_scalar_sdot_range(&ref, 0, rows);
        for (int mode = 0; mode < 2; mode++) {
            for (int i = 0; i < rows + 2; i++) actual[i] = NAN;
            BnTPTask task = {bn_quant_mxfp4_avx2_sdot_range, &got, rows};
            bn_tp_dispatch(mode ? pool : NULL, &task, 1);
            assert(memcmp(actual + 1, expected, sizeof(expected)) == 0);
            assert(isnan(actual[0]) && isnan(actual[rows + 1]));
        }
        for (int i = 0; i < rows + 2; i++) actual[i] = NAN;
        bn_quant_mxfp4_avx2_sdot_range(&got, 3, rows - 2);
        for (int r = 0; r < rows; r++) {
            if (r >= 3 && r < rows - 2) assert(actual[r + 1] == expected[r]);
            else assert(isnan(actual[r + 1]));
        }
    }
    bn_tp_free(pool);
    free(blocks);
    free(input);
#endif
    printf("PASSED\n");
}

static void test_mxfp4_batch_exact(void) {
    printf("test_mxfp4_batch_exact... ");
#ifdef __AVX2__
    enum { max_tasks = 25, rows = 137, nb = 7, cols = nb * 32 };
    const int counts[] = {0, 1, 2, 8, 16, 24, 25};
    BnBlockMXFP4 *blocks = malloc((size_t)max_tasks * rows * nb * sizeof(*blocks));
    assert(blocks);
    float x[cols + 1], expected[max_tasks][rows], actual[max_tasks][rows + 2];
    int8_t scratch[cols + 2];
    BnQWeight weights[max_tasks];
    BnMatvecTask tasks[max_tasks];
    for (int i = 0; i < cols; i++) x[i + 1] = (float)(i % 239 - 119) * 1e-9f;
    for (int b = 0; b < max_tasks * rows * nb; b++) {
        blocks[b].e = (uint8_t)(b * 37);
        for (int i = 0; i < 16; i++) blocks[b].qs[i] = (uint8_t)(b * 19 + i * 31);
    }
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    bn_quant_matvec_batch(NULL, 0, NULL, NULL, pool);
    for (int shape = 0; shape < 3; shape++) {
        for (int t = 0; t < max_tasks; t++) {
            weights[t] = (BnQWeight){blocks + (size_t)t * rows * nb,
                BN_GGUF_TENSOR_MXFP4, rows - t % 5,
                cols - (shape == 1 && t % 2 ? 32 : 0), 1.0f};
            tasks[t] = (BnMatvecTask){actual[t] + 1, &weights[t], NULL,
                shape == 2 && t == 0 ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT : 0};
            bn_quant_matvec(expected[t], &weights[t], x + 1, scratch + 1, NULL);
        }
        for (int mode = 0; mode < 2; mode++) {
            for (size_t n = 0; n < sizeof(counts) / sizeof(counts[0]); n++) {
                for (int t = 0; t < max_tasks; t++)
                    for (int r = 0; r < rows + 2; r++) actual[t][r] = NAN;
                scratch[0] = scratch[cols + 1] = 123;
                bn_quant_matvec_batch(tasks, counts[n], x + 1, scratch + 1,
                                       mode ? pool : NULL);
                assert(scratch[0] == 123 && scratch[cols + 1] == 123);
                for (int t = 0; t < max_tasks; t++) {
                    assert(isnan(actual[t][0]));
                    int written = t < counts[n] ? weights[t].rows : 0;
                    if (written) assert(memcmp(actual[t] + 1, expected[t],
                                               (size_t)written * sizeof(float)) == 0);
                    for (int r = written + 1; r < rows + 2; r++) assert(isnan(actual[t][r]));
                }
            }
        }
    }
    bn_tp_free(pool);
    free(blocks);
#endif
    printf("PASSED\n");
}

static void test_mxfp4_matmul_exact(void) {
    printf("test_mxfp4_matmul_exact... ");
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_MXFP4,
        BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL));
#ifdef __AVX2__
    enum { rows = 137, nb = 7, cols = nb * 32, max_tokens = 17 };
    const int counts[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17};
    BnBlockMXFP4 blocks[rows * nb];
    float X[max_tokens * cols + 1], scales[max_tokens * nb];
    int8_t xq[max_tokens * cols + 1], scratch[cols];
    float expected[max_tokens * rows], actual[max_tokens * rows + 2];
    for (int b = 0; b < rows * nb; b++) {
        blocks[b].e = (uint8_t)(b * 37);
        for (int i = 0; i < 16; i++) blocks[b].qs[i] = (uint8_t)(b * 19 + i * 31);
    }
    for (int i = 0; i < max_tokens * cols; i++) {
        X[i + 1] = (float)(i % 239 - 119) * 1e-9f;
        xq[i + 1] = (int8_t)((i * 17) % 256 - 128);
    }
    for (int i = 0; i < max_tokens * nb; i++) scales[i] = (float)(1 + i % 7) * 1e-9f;
    BnQWeight w = {blocks, BN_GGUF_TENSOR_MXFP4, rows, cols, 1.0f};
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    for (int source = 0; source < 2; source++) {
        if (source) for (int t = 0; t < max_tokens; t++)
            bn_quant_x_to_q8_blocks(X + 1 + t * cols, xq + 1 + t * cols,
                                    scales + t * nb, cols);
        for (int t = 0; t < max_tokens; t++) {
            BnQ4SdotCtx ref = {expected + t * rows, &w, xq + 1 + t * cols,
                               scales + t * nb, NULL};
            bn_quant_mxfp4_scalar_sdot_range(&ref, 0, rows);
        }
        for (size_t n = 0; n < sizeof(counts) / sizeof(counts[0]); n++) {
            BnQ4MatmulCtx ctx = {actual + 1, &w, xq + 1, scales, NULL,
                                 counts[n], cols, NULL, NULL, 0};
            for (int mode = 0; mode < 2; mode++) {
                for (int i = 0; i < max_tokens * rows + 2; i++) actual[i] = NAN;
                BnTPTask task = {bn_quant_mxfp4_avx2_matmul_range, &ctx, rows};
                bn_tp_dispatch(mode ? pool : NULL, &task, 1);
                assert(memcmp(actual + 1, expected, (size_t)counts[n] * rows * sizeof(float)) == 0);
                if (source) {
                    bn_quant_matmul(actual + 1, &w, X + 1, counts[n], scratch, mode ? pool : NULL);
                    assert(memcmp(actual + 1, expected, (size_t)counts[n] * rows * sizeof(float)) == 0);
                    float second[max_tokens * rows];
                    float *outputs[] = {actual + 1, second};
                    const BnQWeight *weights[] = {&w, &w};
                    bn_quant_matmul_prepared_multi_gemv(outputs, weights, NULL,
                        2, X + 1, counts[n], scratch, mode ? pool : NULL);
                    assert(memcmp(actual + 1, expected, (size_t)counts[n] * rows * sizeof(float)) == 0);
                    assert(memcmp(second, expected, (size_t)counts[n] * rows * sizeof(float)) == 0);
                }
                assert(isnan(actual[0]));
                for (int i = counts[n] * rows + 1; i < max_tokens * rows + 2; i++)
                    assert(isnan(actual[i]));
            }
            for (int i = 0; i < max_tokens * rows + 2; i++) actual[i] = NAN;
            bn_quant_mxfp4_avx2_matmul_range(&ctx, 3, rows - 2);
            for (int t = 0; t < counts[n]; t++) for (int r = 0; r < rows; r++) {
                if (r >= 3 && r < rows - 2) assert(actual[1 + t * rows + r] == expected[t * rows + r]);
                else assert(isnan(actual[1 + t * rows + r]));
            }
        }
    }
    bn_tp_free(pool);
#endif
    printf("PASSED\n");
}

static void test_mxfp4_matvec_correctness(void) {
    printf("test_mxfp4_matvec_correctness... ");
    static const int8_t values[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    BnBlockMXFP4 blocks[2] = {0};
    float x[32] = {0};
    int8_t x_q[32];
    float x_scales[1];
    float out[2] = {0};
    float scalar[2] = {0};
    float ref[2] = {0};

    x[0] = 127.0f;
    for (int i = 1; i < 32; i++) x[i] = (float)((i % 13) - 6);
    for (int row = 0; row < 2; row++) {
        blocks[row].e = 127;
        for (int i = 0; i < 16; i++) {
            uint8_t lo = (uint8_t)((i + row) & 15);
            uint8_t hi = (uint8_t)((15 - i + row) & 15);
            blocks[row].qs[i] = (uint8_t)(lo | (hi << 4));
            ref[row] += 0.5f * (float)values[lo] * x[i];
            ref[row] += 0.5f * (float)values[hi] * x[i + 16];
        }
    }

    BnQWeight W = { blocks, BN_GGUF_TENSOR_MXFP4, 2, 32, 1.0f };
    assert(bn_quant_format_supported(BN_GGUF_TENSOR_MXFP4));
    assert(bn_quant_format_data_size(BN_GGUF_TENSOR_MXFP4, 2, 32) ==
           sizeof(blocks));
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, 32);
    assert(fabsf(x_scales[0] - 1.0f) < 1e-6f);
    BnQ4SdotCtx ctx = { scalar, &W, x_q, x_scales, NULL };
    bn_quant_mxfp4_scalar_sdot_range(&ctx, 0, 2);
    bn_quant_matvec(out, &W, x, x_q, NULL);
    for (int row = 0; row < 2; row++) {
        assert(fabsf(scalar[row] - ref[row]) < 1e-5f);
        assert(fabsf(out[row] - ref[row]) < 1e-5f);
    }
    printf("PASSED\n");
}

static void test_iq4xs_avx2_matmul_matches_matvec(void) {
#if defined(__AVX2__)
    printf("test_iq4xs_avx2_matmul_matches_matvec... ");
    enum { rows = 16, cols = 2 * BN_QK_K, n_tokens = 19 };
    BnBlockIQ4XS blocks[rows * 2];
    float X[n_tokens * cols];
    float matvec[n_tokens * rows];
    float matmul[n_tokens * rows];
    int8_t x_q[cols];

    for (int i = 0; i < rows * 2; i++) {
        blocks[i].d = bn_fp32_to_fp16(0.003f * (float)(i + 1));
        blocks[i].scales_h = (uint16_t)(0x9e37u * (unsigned)(i + 3));
        for (int j = 0; j < 4; j++)
            blocks[i].scales_l[j] = (uint8_t)(i * 29 + j * 47 + 11);
        for (int j = 0; j < BN_QK_K / 2; j++)
            blocks[i].qs[j] = (uint8_t)(i * 17 + j * 31 + 7);
    }
    for (int i = 0; i < n_tokens * cols; i++)
        X[i] = 0.015625f * (float)((i * 23 + 5) % 67 - 33);

    BnQWeight W = { blocks, BN_GGUF_TENSOR_IQ4_XS, rows, cols, 1.0f };
    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(matvec + (size_t)t * rows, &W,
                        X + (size_t)t * cols, x_q, NULL);
    bn_quant_matmul(matmul, &W, X, n_tokens, x_q, NULL);
    float max_diff = 0.0f;
    for (int i = 0; i < n_tokens * rows; i++) {
        float diff = fabsf(matmul[i] - matvec[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-3f);
    assert(memcmp(matmul, matvec, sizeof(matmul)) != 0);
    printf("PASSED\n");
#endif
}

#if defined(__AVX2__)
static void check_quant_batch_matches_matvec_flags(const BnQWeight *w,
        const float *x, int8_t *scratch, uint32_t flags) {
    enum { max_tasks = 25 };
    float expected[w->rows], got[max_tasks * w->rows];
    BnMatvecTask tasks[max_tasks];
    bn_quant_matvec_prepared_flags(expected, w, NULL, x, scratch, NULL, flags);
    for (int i = 0; i < max_tasks; i++)
        tasks[i] = (BnMatvecTask){got + i * w->rows, w, NULL, flags};
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    const int counts[] = {1, 2, max_tasks};
    for (int threaded = 0; threaded < 2; threaded++) {
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            for (int i = 0; i < max_tasks * w->rows; i++) got[i] = NAN;
            bn_quant_matvec_batch(tasks, counts[c], x, scratch,
                                   threaded ? pool : NULL);
            for (int i = 0; i < counts[c]; i++)
                assert(memcmp(got + i * w->rows, expected, sizeof(expected)) == 0);
            for (int i = counts[c] * w->rows; i < max_tasks * w->rows; i++)
                assert(isnan(got[i]));
        }
    }
    bn_tp_free(pool);
}

static void check_quant_batch_matches_matvec(const BnQWeight *w,
                                             const float *x, int8_t *scratch) {
    check_quant_batch_matches_matvec_flags(w, x, scratch, 0);
}
#endif

static void test_q5k_public_gemv_matmul_order(void) {
    printf("test_q5k_public_gemv_matmul_order... ");
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q5_K,
        BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL));
#ifdef __AVX2__
    enum { rows = 137, nb = 3, cols = nb * BN_QK_K, tokens = 33 };
    const int counts[] = {1, 2, 3, 4, 5, 8, 17, 25, 33};
    BnBlockQ5K blocks[2][rows * nb];
    float X[tokens * cols + 1], ref[2][tokens * rows], got[2][tokens * rows + 2];
    int8_t scratch[cols];
    BnQWeight weights[2];
    const BnQWeight *wp[2] = {&weights[0], &weights[1]};
    for (int m = 0; m < 2; m++) {
        weights[m] = (BnQWeight){blocks[m], BN_GGUF_TENSOR_Q5_K, rows - 1 + m, cols, 1};
        for (int b = 0; b < rows * nb; b++) {
            blocks[m][b].d = bn_fp32_to_fp16(0.002f * ((b + m) % 19 + 1));
            blocks[m][b].dmin = bn_fp32_to_fp16(0.001f * ((b + m) % 13 + 1));
            for (int i = 0; i < 12; i++) blocks[m][b].scales[i] = (uint8_t)(b * 17 + i * 23 + m);
            for (int i = 0; i < 32; i++) blocks[m][b].qh[i] = (uint8_t)(b * 19 + i * 29 + m);
            for (int i = 0; i < 128; i++) blocks[m][b].qs[i] = (uint8_t)(b * 31 + i * 37 + m);
        }
    }
    for (int i = 0; i < tokens * cols; i++) X[i + 1] = sinf((float)i * 0.13f);
    SHArena *arena = sh_arena_create(1024 * 1024);
    BnPreparedWeight prepared = {0};
    assert(arena && bn_quant_prepare_qweight(&prepared, &weights[0], arena) == 0);
    assert(prepared.kind == BN_PREPARED_WEIGHT_Q5_K_X8);
    assert(!bn_quant_matvec_uses_prepared_weight(&weights[0], 0, NULL));
    for (int policy = 0; policy < 3; policy++) {
        if (policy == 1) setenv("BN_AVX2_KQUANT_FLOAT", "1", 1);
        else unsetenv("BN_AVX2_KQUANT_FLOAT");
        if (policy == 2) setenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH", "1", 1);
        else unsetenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH");
        for (int threaded = 0; threaded < 2; threaded++) {
            BnThreadPool *pool = bn_tp_create(threaded ? 3 : 0);
            assert(pool);
            const uint32_t flags[] = {0, BN_MATVEC_TASK_REFERENCE_DOT,
                BN_MATVEC_TASK_NATIVE_QUANT, BN_MATVEC_TASK_FORCE_FLOAT_KQUANT};
            for (size_t f = 0; f < sizeof(flags) / sizeof(flags[0]); f++)
                assert(!bn_quant_matvec_uses_prepared_weight(&weights[0], flags[f], pool));
            for (int packed = 0; packed < 2; packed++) {
                const BnPreparedWeight *pp[2] = {packed ? &prepared : NULL, NULL};
                for (int t = 0; t < tokens; t++) {
                    BnMatvecTask tasks[2] = {
                        {ref[0] + t * weights[0].rows, &weights[0], pp[0], 0},
                        {ref[1] + t * weights[1].rows, &weights[1], pp[1], 0}
                    };
                    bn_quant_matvec_batch(tasks, 2, X + 1 + t * cols, scratch, pool);
                }
                for (int matrices = 1; matrices <= 2; matrices++)
                for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
                    for (int m = 0; m < 2; m++)
                        for (int i = 0; i < tokens * rows + 2; i++) got[m][i] = NAN;
                    float *outputs[2] = {got[0] + 1, got[1] + 1};
                    bn_quant_matmul_prepared_multi_gemv(outputs, wp, pp, matrices,
                        X + 1, counts[c], scratch, pool);
                    for (int m = 0; m < 2; m++) {
                        int written = m < matrices ? counts[c] * weights[m].rows : 0;
                        assert(memcmp(outputs[m], ref[m], (size_t)written * sizeof(float)) == 0);
                        assert(isnan(got[m][0]));
                        for (int i = written + 1; i < tokens * rows + 2; i++) assert(isnan(got[m][i]));
                    }
                }
            }
            bn_tp_free(pool);
        }
    }
    unsetenv("BN_AVX2_KQUANT_FLOAT");
    unsetenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH");
    sh_arena_free(arena);
#endif
    printf("PASSED\n");
}

static void test_q5k_batch_gemv_order(void) {
#ifdef __AVX2__
    printf("test_q5k_batch_gemv_order... ");
    enum { rows = 137, nb = 3, cols = nb * BN_QK_K, tokens = 33 };
    BnBlockQ5K blocks[rows * nb];
    float x[cols], d[tokens * nb], ref[tokens * rows], got[tokens * rows];
    int8_t q[tokens * cols];
    int16_t bs[tokens * nb * 16];
    for (int b = 0; b < rows * nb; b++) {
        blocks[b].d = bn_fp32_to_fp16(0.002f * (b % 19 + 1));
        blocks[b].dmin = bn_fp32_to_fp16(0.001f * (b % 13 + 1));
        for (int i = 0; i < 12; i++) blocks[b].scales[i] = (uint8_t)(b * 17 + i * 23);
        for (int i = 0; i < 32; i++) blocks[b].qh[i] = (uint8_t)(b * 19 + i * 29);
        for (int i = 0; i < 128; i++) blocks[b].qs[i] = (uint8_t)(b * 31 + i * 37);
    }
    BnQWeight w = {blocks, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f};
    for (int t = 0; t < tokens; t++) {
        for (int i = 0; i < cols; i++) x[i] = sinf((float)(i + t * cols) * 0.13f);
        bn_quant_x_to_q8k(x, q + t * cols, d + t * nb, bs + t * nb * 16, cols);
        BnKQuantSdotCtx one = {ref + t * rows, &w, q + t * cols,
                               d + t * nb, bs + t * nb * 16, NULL};
        bn_quant_q5k_avx2_sdot_range(&one, 0, rows);
    }
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    const int counts[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,31,32,33};
    for (int threaded = 0; threaded < 2; threaded++) {
        for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
            for (int j = 0; j < tokens * rows; j++) got[j] = NAN;
            BnKQuantMatmulCtx c = {got, &w, q, d, bs, counts[i], cols, NULL, NULL};
            BnTPTask task = {bn_quant_q5k_avx2_sdot_matmul_range, &c, rows};
            bn_tp_dispatch(threaded ? pool : NULL, &task, 1);
            assert(memcmp(got, ref, (size_t)counts[i] * rows * sizeof(float)) == 0);
            for (int j = counts[i] * rows; j < tokens * rows; j++) assert(isnan(got[j]));
        }
    }
    bn_tp_free(pool);
    printf("PASSED\n");
#endif
}

static void test_q4k_avx2_batch_matches_single(void) {
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("test_q4k_avx2_batch_matches_single... ");
    enum { rows = 133, blocks = 3, cols = blocks * BN_QK_K };
    BnBlockQ4K data[rows * blocks];
    fill_q4k_blocks(data, rows, blocks, 83);
    float x[cols];
    int8_t scratch[cols];
    for (int i = 0; i < cols; i++) x[i] = sinf((float)i * 0.13f) * 1.7f;
    BnQWeight w = {data, BN_GGUF_TENSOR_Q4_K, rows, cols, 1.0f};
    check_quant_batch_matches_matvec_flags(&w, x, scratch, 0);
    check_quant_batch_matches_matvec_flags(&w, x, scratch,
                                          BN_MATVEC_TASK_REFERENCE_DOT);
    printf("PASSED\n");
#endif
}

static void test_q6k_reference_batch_matches_single(void) {
#if defined(__AVX2__)
    printf("test_q6k_reference_batch_matches_single... ");
    enum { rows = 133, blocks = 3, cols = blocks * BN_QK_K, max_tasks = 25 };
    BnBlockQ6K data[rows * blocks];
    fill_q6k_blocks(data, rows, blocks, 83);
    float x[cols], expected[rows], got[max_tasks * rows];
    int8_t scratch[cols];
    for (int i = 0; i < cols; i++) x[i] = sinf((float)i * 0.13f) * 1.7f;
    BnQWeight w = {data, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f};
    bn_quant_matvec_prepared_flags(expected, &w, NULL, x, scratch, NULL,
                                   BN_MATVEC_TASK_REFERENCE_DOT);
    BnMatvecTask tasks[max_tasks];
    for (int i = 0; i < max_tasks; i++)
        tasks[i] = (BnMatvecTask){got + i * rows, &w, NULL,
                                 BN_MATVEC_TASK_REFERENCE_DOT};
    const int counts[] = {1, 2, max_tasks};
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    for (int threaded = 0; threaded < 2; threaded++) {
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            for (int i = 0; i < max_tasks * rows; i++) got[i] = NAN;
            bn_quant_matvec_batch(tasks, counts[c], x, scratch,
                                   threaded ? pool : NULL);
            for (int i = 0; i < counts[c]; i++)
                assert(memcmp(got + i * rows, expected, sizeof(expected)) == 0);
            for (int i = counts[c] * rows; i < max_tasks * rows; i++)
                assert(isnan(got[i]));
        }
    }
    bn_tp_free(pool);
    printf("PASSED\n");
#endif
}

static void test_iq4nl_panel_integer_reference(void) {
#if defined(__AVX2__)
    printf("test_iq4nl_panel_integer_reference... ");
    enum { rows = 16, nb = 5, cols = nb * 32, tokens = 9 };
    BnBlockIQ4NL blocks[rows * nb];
    float x[tokens * cols], got[tokens * rows];
    int8_t xq[cols];
    float xd[nb], values[32];
    for (int b = 0; b < rows * nb; b++) {
        blocks[b].d = bn_fp32_to_fp16(0.0013f * (b + 1));
        for (int i = 0; i < 16; i++)
            blocks[b].qs[i] = (uint8_t)(i * 37 + b * 19);
    }
    for (int i = 0; i < tokens * cols; i++)
        x[i] = sinf((float)i * 0.13f) * 1.7f;
    BnQWeight w = {blocks, BN_GGUF_TENSOR_IQ4_NL, rows, cols, 1.0f};
    check_quant_batch_matches_matvec(&w, x, xq);
    bn_quant_matmul_prepared(got, &w, NULL, x, tokens, xq, NULL);
    for (int t = 0; t < tokens; t++) {
        float single[rows], routed[rows];
        bn_quant_matvec(routed, &w, x + t * cols, xq, NULL);
        bn_quant_x_to_q8_blocks(x + t * cols, xq, xd, cols);
        BnQ8SdotCtx ctx = {single, &w, xq, xd, NULL};
        bn_quant_iq4nl_avx2_q8_range(&ctx, 0, rows);
        for (int r = 0; r < rows; r++) {
            float expected = 0.0f;
            for (int b = 0; b < nb; b++) {
                BnBlockIQ4NL unit = blocks[r * nb + b];
                float d = bn_fp16_to_fp32(unit.d);
                unit.d = bn_fp32_to_fp16(1.0f);
                bn_quant_dequant_iq4nl(&unit, values);
                int32_t dot = 0;
                for (int i = 0; i < 32; i++)
                    dot += (int32_t)values[i] * xq[b * 32 + i];
                expected = fmaf((float)dot, d * xd[b], expected);
            }
            assert(got[t * rows + r] == expected);
            assert(single[r] == expected);
            assert(routed[r] == expected);
        }
    }
    printf("PASSED\n");
#endif
}

static void test_iq4nl_unaligned_matmul_matches_matvec(void) {
#if defined(__AVX2__)
    printf("test_iq4nl_unaligned_matmul_matches_matvec... ");
    enum { rows = 9, cols = 5 * 32, n_tokens = 11 };
    BnBlockIQ4NL blocks[rows * 5];
    float X[n_tokens * cols];
    float matvec[n_tokens * rows];
    float matmul[n_tokens * rows];
    int8_t x_q[cols];

    for (int i = 0; i < rows * 5; i++) {
        blocks[i].d = bn_fp32_to_fp16(0.002f * (float)(i + 1));
        for (int j = 0; j < 16; j++)
            blocks[i].qs[j] = (uint8_t)(i * 17 + j * 31 + 7);
    }
    for (int i = 0; i < n_tokens * cols; i++)
        X[i] = 0.015625f * (float)((i * 23 + 5) % 67 - 33);

    BnQWeight W = { blocks, BN_GGUF_TENSOR_IQ4_NL, rows, cols, 1.0f };
    check_quant_batch_matches_matvec(&W, X, x_q);
    for (int t = 0; t < n_tokens; t++) {
        bn_quant_matvec(matvec + (size_t)t * rows, &W,
                        X + (size_t)t * cols, x_q, NULL);
        float scales[5], values[32], candidate[rows];
        bn_quant_x_to_q8_blocks(X + t * cols, x_q, scales, cols);
        BnQ8SdotCtx ctx = {candidate, &W, x_q, scales, NULL};
        bn_quant_iq4nl_avx2_q8_range(&ctx, 0, rows);
        for (int r = 0; r < rows; r++) {
            float lanes[2][8] = {{0}};
            for (int b = 0; b < 4; b++) {
                BnBlockIQ4NL unit = blocks[r * 5 + b];
                float d = bn_fp16_to_fp32(unit.d) * scales[b];
                unit.d = bn_fp32_to_fp16(1.0f);
                bn_quant_dequant_iq4nl(&unit, values);
                for (int lane = 0; lane < 8; lane++) {
                    int dot = 0;
                    for (int k = 0; k < 4; k++)
                        dot += (int)values[lane * 4 + k] * x_q[b * 32 + lane * 4 + k];
                    lanes[b & 1][lane] = fmaf((float)dot, d, lanes[b & 1][lane]);
                }
            }
            float a[8];
            for (int lane = 0; lane < 8; lane++)
                a[lane] = lanes[0][lane] + lanes[1][lane];
            float expected = ((a[0] + a[4]) + (a[2] + a[6])) +
                             ((a[1] + a[5]) + (a[3] + a[7]));
            BnBlockIQ4NL unit = blocks[r * 5 + 4];
            float d = bn_fp16_to_fp32(unit.d) * scales[4];
            unit.d = bn_fp32_to_fp16(1.0f);
            bn_quant_dequant_iq4nl(&unit, values);
            int dot = 0;
            for (int k = 0; k < 32; k++) dot += (int)values[k] * x_q[128 + k];
            volatile float product = (float)dot * d;
            expected += product;
            assert(candidate[r] == expected);
            assert(matvec[t * rows + r] == expected);
        }
    }
    bn_quant_matmul(matmul, &W, X, n_tokens, x_q, NULL);
    assert(memcmp(matmul, matvec, sizeof(matmul)) == 0);
    printf("PASSED\n");
#endif
}

static void test_q3k_avx2_matmul_matches_matvec(void) {
#if defined(__AVX2__)
    printf("test_q3k_avx2_matmul_matches_matvec... ");
    assert(!bn_quant_format_cpu_matvec_uses_float_input(BN_GGUF_TENSOR_Q3_K));
    enum { rows = 9, cols = 2 * BN_QK_K, n_tokens = 11 };
    BnBlockQ3K blocks[rows * 2];
    float X[n_tokens * cols];
    float matvec[n_tokens * rows];
    float matmul[n_tokens * rows];
    int8_t x_q[cols];

    for (int i = 0; i < rows * 2; i++) {
        blocks[i].d = bn_fp32_to_fp16(0.002f * (float)(i + 1));
        for (int j = 0; j < BN_QK_K / 8; j++)
            blocks[i].hmask[j] = (uint8_t)(i * 13 + j * 37 + 3);
        for (int j = 0; j < BN_QK_K / 4; j++)
            blocks[i].qs[j] = (uint8_t)(i * 17 + j * 31 + 7);
        for (int j = 0; j < 12; j++)
            blocks[i].scales[j] = (uint8_t)(i * 29 + j * 43 + 11);
    }
    for (int i = 0; i < n_tokens * cols; i++)
        X[i] = 0.015625f * (float)((i * 23 + 5) % 67 - 33);

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q3_K, rows, cols, 1.0f };
    check_quant_batch_matches_matvec(&W, X, x_q);
    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(matvec + (size_t)t * rows, &W,
                        X + (size_t)t * cols, x_q, NULL);
    BnThreadPool *pool = bn_tp_create(3);
    assert(pool);
    const int counts[] = {1, 3, 8, n_tokens};
    for (int threaded = 0; threaded < 2; threaded++) {
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            for (int i = 0; i < n_tokens * rows; i++) matmul[i] = NAN;
            bn_quant_matmul(matmul, &W, X, counts[c], x_q, threaded ? pool : NULL);
            assert(memcmp(matmul, matvec, (size_t)counts[c] * rows * sizeof(float)) == 0);
            for (int i = counts[c] * rows; i < n_tokens * rows; i++)
                assert(isnan(matmul[i]));
        }
    }
    bn_tp_free(pool);
    printf("PASSED\n");
#endif
}

static void test_q4_x8_packed_arithmetic(void) {
    printf("test_q4_x8_packed_arithmetic... ");
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    // Pinned GGML CPU repack buffer, Q4_0_8x8, MUL_MAT [64,8] x [64,3].
static const float q4_packed_reference[24] = {
-0x1.422f6p-2f,
0x1.a19ccp-3f,
-0x1.a6cf1p+0f,
0x1.592dp-1f,
0x1.bda46p-1f,
0x1.489512p+2f,
0x1.e4b1ep+1f,
0x1.b8ebp-4f,
-0x1.55b62p-1f,
-0x1.4c2f38p+1f,
0x1.3d57c4p+1f,
-0x1.56a8p-5f,
0x1.0c221cp+1f,
0x1.8d7ed4p+1f,
-0x1.b743b8p+1f,
0x1.5e37p-3f,
-0x1.23018p-3f,
-0x1.52679p+0f,
0x1.011228p+0f,
-0x1.267356p+2f,
0x1.4fe29p+0f,
0x1.c3a04p-2f,
0x1.228386p+2f,
0x1.bee6ep-1f
};

    const int widths[] = {32, 64, 256, 704};
    const int token_counts[] = {1, 3, 8, 9};
    for (int wi = 0; wi < 4; wi++) for (int rows = 8; rows <= 24; rows += 8)
    for (int ti = 0; ti < 4; ti++) {
        int cols = widths[wi], tokens = token_counts[ti], blocks = cols / 32;
        BnBlockQ4_0 *raw = malloc((size_t)rows * blocks * sizeof(*raw));
        float *x = malloc((size_t)tokens * cols * sizeof(float));
        int8_t *xq = malloc((size_t)tokens * cols);
        float *xd = malloc((size_t)tokens * blocks * sizeof(float));
        float *ref = malloc((size_t)tokens * rows * sizeof(float));
        float *out = malloc((size_t)tokens * rows * sizeof(float));
        assert(raw && x && xq && xd && ref && out);
        for (int b = 0; b < rows * blocks; b++) {
            raw[b].d = bn_fp32_to_fp16((b % 7 + 1) * .0625f * (b % 3 ? -1.f : 1.f));
            for (int j = 0; j < 16; j++) raw[b].qs[j] = (uint8_t)(b * 17 + j * 23 + 5);
        }
        for (int i = 0; i < tokens * cols; i++) x[i] = ((i * 13) % 41 - 20) / 32.f;
        BnQWeight weight = {raw, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.f};
        BnPreparedWeightKind kind;
        size_t bytes = bn_quant_prepared_qweight_size(&weight, &kind);
        assert(kind == BN_PREPARED_WEIGHT_Q4_0_X8 && bytes > 0);
        SHArena *arena = sh_arena_create(bytes);
        BnPreparedWeight prepared;
        assert(arena && bn_quant_prepare_qweight(&prepared, &weight, arena) == 0);
        assert(prepared.kind == kind && prepared.aux);
        assert(prepared.aux_size == (size_t)(rows / 8) * blocks * sizeof(BnBlockQ4_0x8));
        const BnBlockQ4_0x8 *packed = (const BnBlockQ4_0x8 *)prepared.aux;
        for (int r = 0; r < rows; r++) for (int b = 0; b < blocks; b++) {
            const BnBlockQ4_0x8 *pb = packed + (size_t)(r / 8) * blocks + b;
            assert(pb->d[r % 8] == raw[r * blocks + b].d);
            assert(memcmp(pb->qs[r % 8], raw[r * blocks + b].qs, 16) == 0);
        }
        for (int t = 0; t < tokens; t++) {
            bn_quant_x_to_q8_blocks(x + t * cols, xq + t * cols, xd + t * blocks, cols);
            for (int r = 0; r < rows; r++) {
                float value = 0.f;
                for (int b = 0; b < blocks; b++) {
                    int sum = 0;const BnBlockQ4_0 *q = raw + r * blocks + b;
                    for (int j = 0; j < 16; j++) {
                        sum += ((q->qs[j] & 15) - 8) * xq[t * cols + b * 32 + j];
                        sum += ((q->qs[j] >> 4) - 8) * xq[t * cols + b * 32 + j + 16];
                    }
                    value = fmaf((float)sum, bn_fp16_to_fp32(q->d) * xd[t * blocks + b], value);
                }
                ref[t * rows + r] = value;
            }
        }
        if (rows == 8 && cols == 64 && tokens == 3)
            assert(memcmp(ref, q4_packed_reference, sizeof(q4_packed_reference)) == 0);
        bn_quant_matmul_prepared(out, &weight, &prepared, x, tokens, xq, NULL);
        assert(memcmp(out, ref, (size_t)tokens * rows * sizeof(float)) == 0);
        for (int t = 0; t < tokens; t++) {
            bn_quant_matvec_prepared(out + t * rows, &weight, &prepared, x + t * cols, xq, NULL);
            assert(memcmp(out + t * rows, ref + t * rows, (size_t)rows * sizeof(float)) == 0);
        }
        assert(weight.data == raw);
        BnQWeight ineligible = weight;ineligible.rows--;
        assert(bn_quant_prepared_qweight_size(&ineligible, NULL) == 0);
        ineligible = weight;ineligible.cols--;
        assert(bn_quant_prepared_qweight_size(&ineligible, NULL) == 0);
        sh_arena_free(arena);free(raw);free(x);free(xq);free(xd);free(ref);free(out);
    }
    assert(!bn_quant_format_tied_logits_uses_prepared_weight(BN_GGUF_TENSOR_Q4_0));
#endif
    printf("PASSED\n");
}

int main(void) {
    printf("=== Quant Integration Tests ===\n");
    test_q5k_batch_gemv_order();
    test_q5k_public_gemv_matmul_order();
    test_q4k_avx2_batch_matches_single();
    test_q6k_reference_batch_matches_single();
    test_quant_policy_helpers();
    test_fp16_conversion();
    test_iq2xxs_dequant_payload();
    test_iq3s_block_layout_and_matvec();
#if defined(__AVX2__)
    test_iq3s_panel_matmul();
#endif
    test_dispatch_routing();
    test_q51_x86_exact();
    test_q51_half_metadata();
    test_logits_refine_rows();
    test_matvec_batch();
    test_q4_large_matvec_batch();
    test_matvec_threaded();
    test_matmul_correctness();
    test_q4_matmul_correctness();
    test_q8_matmul_correctness();
    test_q5k_matmul_correctness();
    test_q5k_x8_prepared_correctness();
    test_q4k_avx512_x16_prepared_matmul_tail_order();
    test_q8k_avx2_x4_pack_signed_max_tie();
    test_q5k_multi_matches_single_exact();
    test_q5k_matvec_multi_correctness();
    test_q5k_matvec_batch_correctness();
    test_i2s_matvec_multi_correctness();
    test_q4_matvec_multi_correctness();
    test_q4_repacked_scalar_layout();
    test_q4_repacked_neon_reduction_order();
    test_q4_repacked_neon_fused_gateup_silu();
    test_q4_neon_4row_reduction_order();
    test_q8_matvec_batch_correctness();
    test_q8_neon_reference_reduction_order();
    test_q8_matvec_multi_correctness();
    test_unquantized_matvec_correctness();
    test_f32_scalar_reduction_order();
    test_q8k_quantizer_matches_scalar();
    test_f32_x86_decode_reduction_order();
    test_f32_avx2_batch_reduction_order();
    test_f32_batch_scheduling();
    test_bf16_matvec_batch_correctness();
    test_bf16_matvec_multi_correctness();
    test_mixed_kquant_matvec_batch_correctness();
    test_mixed_kquant_matvec_multi_correctness();
    test_kquant_prepared_kquant_input_matmul_correctness();
    test_q4k_unpacked_gemv_batch_policy();
    test_q4k_prepared_multi_matches_separate();
    test_matmul_gemv_fallback();
    test_q8_matmul_gemv_order();
    test_q8_matmul_row_scheduling();
    test_matmul_gemv_row_scheduling();
    test_q6k_scalar_sdot_4row_correctness();
    test_q6k_neon_pair_reduction_order();
    test_q6k_prepared_matmul_correctness();
    test_q6k_avx512_row_pair_exact();
    test_q6k_matmul_row_scheduling();
    test_activation_quant_rounding();
    test_q4_x8_packed_arithmetic();
    test_mxfp4_x86_exact();
    test_mxfp4_batch_exact();
    test_mxfp4_matmul_exact();
    test_mxfp4_matvec_correctness();
    test_iq4xs_avx2_matmul_matches_matvec();
    test_iq4nl_unaligned_matmul_matches_matvec();
    test_iq4nl_panel_integer_reference();
    test_q3k_avx2_matmul_matches_matvec();
    printf("All quant integration tests passed!\n");
    return 0;
}
