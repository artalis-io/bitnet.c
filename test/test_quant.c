#include "quant.h"
#include "backend_quant.h"
#include "quant_dispatch_internal.h"
#include "gguf.h"
#include "sh_arena.h"
#include "quant_ctx.h"
#include "quant_kernels_scalar.h"
#if defined(__ARM_NEON)
#include "quant_kernels_neon.h"
#include "simd_helpers.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

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

#if defined(BN_FORCE_SCALAR) || \
    (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    const int q4_dot_default = 1;
#else
    const int q4_dot_default = 0;
#endif
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
    setenv("BN_AVX2_KQUANT_FLOAT", "0", 1);
    assert(!bn_quant_policy_avx2_kquant_float_for_tasks(TEST_QUANT_POLICY(), tasks, 2));
    unsetenv("BN_AVX2_KQUANT_FLOAT");

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
    assert(bn_quant_policy_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(), BN_MATVEC_TASK_REFERENCE_DOT));
    setenv("BN_CPU_DISABLE_Q4_DOT", "1", 1);
    assert(!bn_quant_policy_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(), BN_MATVEC_TASK_REFERENCE_DOT));
    unsetenv("BN_CPU_DISABLE_Q4_DOT");
    assert(!bn_quant_policy_reference_q4_dot_enabled(
        TEST_QUANT_POLICY(),
        BN_MATVEC_TASK_REFERENCE_DOT | BN_MATVEC_TASK_NATIVE_QUANT));
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
    assert(bn_quant_policy_q8_0_matmul_batch_enabled(TEST_QUANT_POLICY()));
    setenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH", "1", 1);
    assert(!bn_quant_policy_q8_0_matmul_batch_enabled(TEST_QUANT_POLICY()));
    unsetenv("BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH");
    setenv("BN_DISABLE_Q8_0_MATMUL_BATCH", "1", 1);
    assert(!bn_quant_policy_q8_0_matmul_batch_enabled(TEST_QUANT_POLICY()));
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
    assert(bn_quant_format_supports_f16_float_cache_matvec(
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
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_Q8_K));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_IQ3_XXS));
    assert(bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_gpu_matvec_supported(BN_GGUF_TENSOR_MXFP4));
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
    free(q6k);
    free(q5k);
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
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < cols; i++)
        x[i] = 0.04f * ((i * 7 + 5) % 31) - 0.6f;

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    int8_t x_q[256];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
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

static void test_kquant_prepared_kquant_input_matmul_correctness(void) {
    printf("test_kquant_prepared_kquant_input_matmul_correctness... ");

    int rows = 7, cols = 512, n_tokens = 5;
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
    float *x_d = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
    int16_t *x_bsums = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
    float *ref4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *ref6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *float4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *float6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *force4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *force6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4b = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4_prepared = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4b_prepared = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    assert(X && x_q && x_d && x_bsums && ref4 && ref6 && float4 && float6 &&
           force4 && force6 &&
           out4 && out4b && out6 && out4_prepared && out4b_prepared);

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
                              x_q + (size_t)t * cols, NULL);
    }

    bn_quant_matmul_prepared_kquant_input(
        out4, &W4a, n_tokens, x_q, x_d, x_bsums, X, NULL);
    bn_quant_matmul_prepared_kquant_input(
        out6, &W6, n_tokens, x_q, x_d, x_bsums, X, NULL);

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
#endif

    for (int i = 0; i < rows * n_tokens; i++) {
        assert(fabsf(out4[i] - ref4[i]) < 1e-3f);
        assert(fabsf(out6[i] - ref6[i]) < 1e-3f);
        assert(fabsf(float4[i] - force4[i]) < 1e-6f);
        assert(fabsf(float6[i] - force6[i]) < 1e-6f);
#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
        assert(fabsf(out4_prepared[i] - out4[i]) < 1e-5f);
        assert(fabsf(out4b_prepared[i] - out4b[i]) < 1e-5f);
#endif
    }

    free(q4a); free(q4b); free(q6);
    free(X); free(x_q); free(x_d); free(x_bsums);
    free(ref4); free(ref6); free(float4); free(float6);
    free(force4); free(force6);
    free(out4); free(out4b); free(out6);
    free(out4_prepared); free(out4b_prepared);
#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
    sh_arena_free(prep_arena);
#endif
    printf("PASSED\n");
}

static void test_q6k_prepared_matmul_correctness(void) {
    printf("test_q6k_prepared_matmul_correctness... ");

#if defined(__AVX2__) || \
    (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
    int rows = 2048, cols = 4096, n_tokens = 3;
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
    assert(kind == BN_PREPARED_WEIGHT_Q6_K_EXPANDED);
    assert(prep_bytes > 0);
    SHArena *arena = sh_arena_create(prep_bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = { 0 };
    assert(bn_quant_prepare_qweight(&prepared, &W6, arena) == 0);
    assert(prepared.kind == BN_PREPARED_WEIGHT_Q6_K_EXPANDED);

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
        assert(fabsf(prepared_out[i] - raw[i]) < 1e-4f);
        assert(fabsf(prepared_input_out[i] - raw[i]) < 1e-4f);
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
    assert(x_q8[1] == 49);

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

int main(void) {
    printf("=== Quant Integration Tests ===\n");
    test_quant_policy_helpers();
    test_fp16_conversion();
    test_dispatch_routing();
    test_logits_refine_rows();
    test_matvec_batch();
    test_q4_large_matvec_batch();
    test_matvec_threaded();
    test_matmul_correctness();
    test_q4_matmul_correctness();
    test_q8_matmul_correctness();
    test_q5k_matmul_correctness();
    test_q5k_matvec_multi_correctness();
    test_q5k_matvec_batch_correctness();
    test_i2s_matvec_multi_correctness();
    test_q4_matvec_multi_correctness();
    test_q4_repacked_neon_reduction_order();
    test_q4_repacked_neon_fused_gateup_silu();
    test_q4_neon_4row_reduction_order();
    test_q8_matvec_batch_correctness();
    test_q8_neon_reference_reduction_order();
    test_q8_matvec_multi_correctness();
    test_unquantized_matvec_correctness();
    test_bf16_matvec_batch_correctness();
    test_bf16_matvec_multi_correctness();
    test_mixed_kquant_matvec_batch_correctness();
    test_mixed_kquant_matvec_multi_correctness();
    test_kquant_prepared_kquant_input_matmul_correctness();
    test_q6k_scalar_sdot_4row_correctness();
    test_q6k_neon_pair_reduction_order();
    test_q6k_prepared_matmul_correctness();
    test_activation_quant_rounding();
    test_mxfp4_matvec_correctness();
    printf("All quant integration tests passed!\n");
    return 0;
}
