#include "transformer_ssm_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Simple deterministic pseudo-random float in [-1, 1]
static float prand(uint32_t *state) {
    *state = *state * 1103515245u + 12345u;
    return (float)((int32_t)(*state >> 16) & 0x7FFF) / 16384.0f - 1.0f;
}

static void fill_random(float *buf, int n, uint32_t *state) {
    for (int i = 0; i < n; i++) buf[i] = prand(state);
}

// Compare two buffers with tolerance, return max absolute diff
static float max_diff(const float *a, const float *b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// --- Test conv_silu ---
static void test_conv_silu(void) {
    printf("test_ssm_conv_silu... ");

    const int qkv_dim = 8192;
    const int kern = 4;
    uint32_t seed = 42;

    float *qkv_ref = malloc(qkv_dim * sizeof(float));
    float *qkv_test = malloc(qkv_dim * sizeof(float));
    float *conv_state_ref = malloc((kern - 1) * qkv_dim * sizeof(float));
    float *conv_state_test = malloc((kern - 1) * qkv_dim * sizeof(float));
    float *conv1d_w = malloc(qkv_dim * kern * sizeof(float));

    fill_random(qkv_ref, qkv_dim, &seed);
    fill_random(conv_state_ref, (kern - 1) * qkv_dim, &seed);
    fill_random(conv1d_w, qkv_dim * kern, &seed);
    memcpy(qkv_test, qkv_ref, qkv_dim * sizeof(float));
    memcpy(conv_state_test, conv_state_ref, (kern - 1) * qkv_dim * sizeof(float));

    // Run scalar reference
    BnSSMConvCtx ctx_ref = { qkv_ref, conv_state_ref, conv1d_w, qkv_dim, kern };
    bn_transformer_ssm_conv_silu_scalar_range(&ctx_ref, 0, qkv_dim);

    // Run platform kernel
    BnSSMConvCtx ctx_test = { qkv_test, conv_state_test, conv1d_w, qkv_dim, kern };
#ifdef __ARM_NEON
    bn_transformer_ssm_conv_silu_neon_range(&ctx_test, 0, qkv_dim);
#elif defined(__AVX2__)
    bn_transformer_ssm_conv_silu_avx2_range(&ctx_test, 0, qkv_dim);
#elif defined(__wasm_simd128__)
    bn_transformer_ssm_conv_silu_wasm_range(&ctx_test, 0, qkv_dim);
#else
    bn_transformer_ssm_conv_silu_scalar_range(&ctx_test, 0, qkv_dim);
#endif

    float d = max_diff(qkv_ref, qkv_test, qkv_dim);
    // AVX2 uses fast polynomial exp (~1e-5 relative error)
    assert(d < 1e-4f);

    free(qkv_ref); free(qkv_test);
    free(conv_state_ref); free(conv_state_test);
    free(conv1d_w);
    printf("PASSED (max_diff=%.2e)\n", d);
}

// --- Test l2norm ---
static void test_l2norm(void) {
    printf("test_ssm_l2norm... ");

    const int num_heads = 16;
    const int head_dim = 128;
    const int total = num_heads * head_dim;
    uint32_t seed = 123;

    float *q_ref = malloc(total * sizeof(float));
    float *k_ref = malloc(total * sizeof(float));
    float *q_test = malloc(total * sizeof(float));
    float *k_test = malloc(total * sizeof(float));

    fill_random(q_ref, total, &seed);
    fill_random(k_ref, total, &seed);
    memcpy(q_test, q_ref, total * sizeof(float));
    memcpy(k_test, k_ref, total * sizeof(float));

    BnSSML2NormCtx ctx_ref = { q_ref, k_ref, head_dim };
    bn_transformer_ssm_l2norm_scalar_range(&ctx_ref, 0, num_heads);

    BnSSML2NormCtx ctx_test = { q_test, k_test, head_dim };
#ifdef __ARM_NEON
    bn_transformer_ssm_l2norm_neon_range(&ctx_test, 0, num_heads);
#elif defined(__AVX2__)
    bn_transformer_ssm_l2norm_avx2_range(&ctx_test, 0, num_heads);
#elif defined(__wasm_simd128__)
    bn_transformer_ssm_l2norm_wasm_range(&ctx_test, 0, num_heads);
#else
    bn_transformer_ssm_l2norm_scalar_range(&ctx_test, 0, num_heads);
#endif

    float dq = max_diff(q_ref, q_test, total);
    float dk = max_diff(k_ref, k_test, total);
    assert(dq < 1e-5f);
    assert(dk < 1e-5f);

    free(q_ref); free(k_ref);
    free(q_test); free(k_test);
    printf("PASSED (max_diff q=%.2e k=%.2e)\n", dq, dk);
}

// --- Test delta ---
static void test_delta(void) {
    printf("test_ssm_delta... ");

    const int num_k_heads = 8;
    const int head_k_dim = 128;
    const int num_v_heads = 32;
    const int head_v_dim = 128;
    const int key_dim = num_k_heads * head_k_dim;
    const int value_dim = num_v_heads * head_v_dim;
    const int state_size = num_v_heads * head_k_dim * head_v_dim;
    uint32_t seed = 456;

    float *state_ref = calloc(state_size, sizeof(float));
    float *state_test = calloc(state_size, sizeof(float));
    float *out_ref = malloc(value_dim * sizeof(float));
    float *out_test = malloc(value_dim * sizeof(float));
    float *q = malloc(key_dim * sizeof(float));
    float *k = malloc(key_dim * sizeof(float));
    float *v_ref = malloc(value_dim * sizeof(float));
    float *v_test = malloc(value_dim * sizeof(float));
    float *alpha = malloc(num_v_heads * sizeof(float));
    float *beta_arr = malloc(num_v_heads * sizeof(float));

    fill_random(q, key_dim, &seed);
    fill_random(k, key_dim, &seed);
    fill_random(v_ref, value_dim, &seed);
    memcpy(v_test, v_ref, value_dim * sizeof(float));

    // Use small values for alpha (decay) and beta to keep state bounded
    for (int i = 0; i < num_v_heads; i++) {
        alpha[i] = 0.9f + 0.05f * prand(&seed);
        beta_arr[i] = 0.3f + 0.2f * prand(&seed);
    }

    // Initialize state with small random values
    fill_random(state_ref, state_size, &seed);
    for (int i = 0; i < state_size; i++) state_ref[i] *= 0.1f;
    memcpy(state_test, state_ref, state_size * sizeof(float));

    float q_scale = 1.0f / sqrtf((float)head_k_dim);

    BnSSMDeltaCtx ctx_ref = {
        state_ref, out_ref, q, k, v_ref,
        alpha, beta_arr,
        num_k_heads, head_k_dim, head_v_dim, q_scale
    };
    bn_transformer_ssm_delta_scalar_range(&ctx_ref, 0, num_v_heads);

    BnSSMDeltaCtx ctx_test = {
        state_test, out_test, q, k, v_test,
        alpha, beta_arr,
        num_k_heads, head_k_dim, head_v_dim, q_scale
    };
#ifdef __ARM_NEON
    bn_transformer_ssm_delta_neon_range(&ctx_test, 0, num_v_heads);
#elif defined(__AVX2__)
    bn_transformer_ssm_delta_avx2_range(&ctx_test, 0, num_v_heads);
#elif defined(__wasm_simd128__)
    bn_transformer_ssm_delta_wasm_range(&ctx_test, 0, num_v_heads);
#else
    bn_transformer_ssm_delta_scalar_range(&ctx_test, 0, num_v_heads);
#endif

    float d_out = max_diff(out_ref, out_test, value_dim);
    float d_state = max_diff(state_ref, state_test, state_size);
    // Delta kernel has FMA reordering, so allow 1e-4 tolerance
    assert(d_out < 1e-4f);
    assert(d_state < 1e-4f);

    free(state_ref); free(state_test);
    free(out_ref); free(out_test);
    free(q); free(k);
    free(v_ref); free(v_test);
    free(alpha); free(beta_arr);
    printf("PASSED (max_diff out=%.2e state=%.2e)\n", d_out, d_state);
}

// --- Test gate ---
static void test_gate(void) {
    printf("test_ssm_gate... ");

    const int num_v_heads = 32;
    const int head_v_dim = 128;
    const int total = num_v_heads * head_v_dim;
    uint32_t seed = 789;

    float *out_ref = malloc(total * sizeof(float));
    float *out_test = malloc(total * sizeof(float));
    float *z = malloc(total * sizeof(float));
    float *norm_w = malloc(head_v_dim * sizeof(float));

    fill_random(out_ref, total, &seed);
    fill_random(z, total, &seed);
    fill_random(norm_w, head_v_dim, &seed);
    memcpy(out_test, out_ref, total * sizeof(float));

    BnSSMGateCtx ctx_ref = { out_ref, z, norm_w, 1e-5f, head_v_dim };
    bn_transformer_ssm_gate_scalar_range(&ctx_ref, 0, num_v_heads);

    BnSSMGateCtx ctx_test = { out_test, z, norm_w, 1e-5f, head_v_dim };
#ifdef __ARM_NEON
    bn_transformer_ssm_gate_neon_range(&ctx_test, 0, num_v_heads);
#elif defined(__AVX2__)
    bn_transformer_ssm_gate_avx2_range(&ctx_test, 0, num_v_heads);
#elif defined(__wasm_simd128__)
    bn_transformer_ssm_gate_wasm_range(&ctx_test, 0, num_v_heads);
#else
    bn_transformer_ssm_gate_scalar_range(&ctx_test, 0, num_v_heads);
#endif

    float d = max_diff(out_ref, out_test, total);
    // AVX2 uses fast polynomial exp for SiLU (~1e-5 relative error)
    assert(d < 1e-4f);

    free(out_ref); free(out_test);
    free(z); free(norm_w);
    printf("PASSED (max_diff=%.2e)\n", d);
}

int main(void) {
    printf("=== SSM Kernel Equivalence Tests ===\n");
#ifdef __ARM_NEON
    printf("Platform: NEON vs scalar\n");
#elif defined(__AVX2__)
    printf("Platform: AVX2 vs scalar\n");
#elif defined(__wasm_simd128__)
    printf("Platform: WASM SIMD128 vs scalar\n");
#else
    printf("Platform: scalar vs scalar (no SIMD)\n");
#endif

    test_conv_silu();
    test_l2norm();
    test_delta();
    test_gate();

    printf("All SSM kernel tests passed!\n");
    return 0;
}
