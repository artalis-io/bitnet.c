#include "transformer_ssm_internal.h"
#include "kda_reference.h"
#include "kda_norm_reference.h"
#include "kda_coefficients_reference.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

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

#ifdef __ARM_NEON
static float neon_dot_reference(const float *x, const float *y, int n) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(x + i),
                         vld1q_f32(y + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(x + i + 4),
                         vld1q_f32(y + i + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(x + i + 8),
                         vld1q_f32(y + i + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(x + i + 12),
                         vld1q_f32(y + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum2),
                                      vaddq_f32(sum1, sum3)));
    for (; i < n; i++)
        sum += x[i] * y[i];
    return sum;
}

static void test_delta_neon_reduction_order(void) {
    printf("test_ssm_delta_neon_reduction_order... ");

    enum { head_dim = 128 };
    float state[head_dim];
    float q[head_dim];
    float k[head_dim];
    float v[1] = { 0.0f };
    float out[1] = { 0.0f };
    float alpha[1] = { 1.0f };
    float beta[1] = { 0.0f };
    uint32_t seed = 0x517cc1b7u;
    fill_random(state, head_dim, &seed);
    fill_random(q, head_dim, &seed);
    fill_random(k, head_dim, &seed);

    float expected = neon_dot_reference(state, q, head_dim);
    BnSSMDeltaCtx ctx = {
        state, out, q, k, v, alpha, beta, 1, head_dim, 1, 1.0f
    };
    bn_transformer_ssm_delta_neon_range(&ctx, 0, 1);
    assert(memcmp(out, &expected, sizeof(expected)) == 0);
    printf("PASSED\n");
}
#endif

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
    BnSSMConvCtx ctx_ref = {
        qkv_ref, conv_state_ref, conv1d_w, qkv_dim, kern, 0
    };
    bn_transformer_ssm_conv_silu_scalar_range(&ctx_ref, 0, qkv_dim);

    // Run platform kernel
    BnSSMConvCtx ctx_test = {
        qkv_test, conv_state_test, conv1d_w, qkv_dim, kern, 0
    };
#ifdef __ARM_NEON
    bn_transformer_ssm_conv_silu_neon_range(&ctx_test, 0, qkv_dim);
#elif defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
    bn_transformer_ssm_conv_silu_x86_range(&ctx_test, 0, qkv_dim);
#elif defined(__AVX2__)
    bn_transformer_ssm_conv_silu_avx2_range(&ctx_test, 0, qkv_dim);
#elif defined(__wasm_simd128__)
    bn_transformer_ssm_conv_silu_wasm_range(&ctx_test, 0, qkv_dim);
#else
    bn_transformer_ssm_conv_silu_scalar_range(&ctx_test, 0, qkv_dim);
#endif

    float d = max_diff(qkv_ref, qkv_test, qkv_dim);
    // SIMD kernels use ggml's fast polynomial exp.
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

    BnSSML2NormCtx ctx_ref = { q_ref, k_ref, 1e-6f, head_dim, BN_SSM_QK_NORM_CLAMPED_L2 };
    bn_transformer_ssm_l2norm_scalar_range(&ctx_ref, 0, num_heads);

    BnSSML2NormCtx ctx_test = { q_test, k_test, 1e-6f, head_dim, BN_SSM_QK_NORM_CLAMPED_L2 };
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

// Independent GGML graph fixtures: reference_cpu_gdn_norm.c.
static void test_qk_rmsnorm_reference(void) {
    printf("test_ssm_qk_rmsnorm_reference... ");
    const struct { int hd; float eps; uint64_t hash; } cases[] = {
        {1, 1e-6f, UINT64_C(0xb87c0d60e44cfc61)},
        {7, 1e-6f, UINT64_C(0x478e515efa2ee10c)},
        {16, 1e-6f, UINT64_C(0x5242569b5cee0e98)},
        {128, 1e-6f, UINT64_C(0x2a61c1ae3be88d36)},
        {256, 1e-6f, UINT64_C(0xa1b09d53143339a0)},
        {1, 1e-5f, UINT64_C(0x3bf82f006cb599ec)},
        {7, 1e-5f, UINT64_C(0x9af71c7c6d9d271e)},
        {16, 1e-5f, UINT64_C(0xf1aae4ca5b681b27)},
        {128, 1e-5f, UINT64_C(0xb6cc48d1cf6ba031)},
        {256, 1e-5f, UINT64_C(0x0689be0d3d2b200c)},
    };
    const float scales[] = {0.0f, 1e-8f, 1.0f, 1e10f};
    for (unsigned c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int hd = cases[c].hd;
        for (int backend = 0; backend < 2; backend++) {
            float q[4*256+2], k[4*256+2];
            for (int i = 0; i < 4*256+2; i++) q[i] = k[i] = NAN;
            for (int h = 0; h < 4; h++) for (int d = 0; d < hd; d++)
                q[1+h*hd+d] = k[1+h*hd+d] =
                    sinf((float)(h*hd+d+1)*0.13f) * scales[h];
            BnSSML2NormCtx ctx = {q+1, k+1, cases[c].eps, hd,
                                   BN_SSM_QK_NORM_RMS_SCALED};
            if (!backend) {
                bn_transformer_ssm_l2norm_scalar_range(&ctx, 0, 4);
            } else {
                // Split head ranges checks partial dispatch without changing sums.
#ifdef __ARM_NEON
                bn_transformer_ssm_l2norm_neon_range(&ctx, 0, 1);
                bn_transformer_ssm_l2norm_neon_range(&ctx, 1, 4);
#elif defined(__AVX2__)
                bn_transformer_ssm_l2norm_avx2_range(&ctx, 0, 1);
                bn_transformer_ssm_l2norm_avx2_range(&ctx, 1, 4);
#elif defined(__wasm_simd128__)
                bn_transformer_ssm_l2norm_wasm_range(&ctx, 0, 1);
                bn_transformer_ssm_l2norm_wasm_range(&ctx, 1, 4);
#else
                bn_transformer_ssm_l2norm_scalar_range(&ctx, 0, 1);
                bn_transformer_ssm_l2norm_scalar_range(&ctx, 1, 4);
#endif
            }
            uint64_t hash = UINT64_C(14695981039346656037);
            assert(memcmp(q+1, k+1, (size_t)4*hd*sizeof(float)) == 0);
            for (int i = 1; i <= 4*hd; i++) {
                uint32_t bits; memcpy(&bits, q+i, sizeof(bits));
                hash = (hash ^ bits) * UINT64_C(1099511628211);
            }
            if (hash != cases[c].hash)
                fprintf(stderr, "GDN RMS hd=%d backend=%d hash=%016llx expected=%016llx\n",
                    hd, backend, (unsigned long long)hash, (unsigned long long)cases[c].hash);
            assert(hash == cases[c].hash);
            assert(isnan(q[0]) && isnan(k[0]));
            for (int i = 4*hd+1; i < 4*256+2; i++) assert(isnan(q[i]) && isnan(k[i]));
        }
    }
    printf("PASSED\n");
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

    BnSSMGateCtx ctx_ref = {
        out_ref, z, norm_w, 1e-5f, head_v_dim, 0, num_v_heads
    };
    bn_transformer_ssm_gate_scalar_range(&ctx_ref, 0, num_v_heads);

    BnSSMGateCtx ctx_test = {
        out_test, z, norm_w, 1e-5f, head_v_dim, 0, num_v_heads
    };
#ifdef __ARM_NEON
    bn_transformer_ssm_gate_neon_range(&ctx_test, 0, num_v_heads);
#elif defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
    bn_transformer_ssm_gate_x86_range(&ctx_test, 0, num_v_heads);
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

    seed = 789;
    fill_random(out_ref, total, &seed);
    fill_random(z, total, &seed);
    fill_random(norm_w, head_v_dim, &seed);
    memcpy(out_test, out_ref, total * sizeof(float));
    ctx_ref.sigmoid_gate = 1;
    ctx_test.sigmoid_gate = 1;
    bn_transformer_ssm_gate_scalar_range(&ctx_ref, 0, num_v_heads);
#ifdef __ARM_NEON
    bn_transformer_ssm_gate_neon_range(&ctx_test, 0, num_v_heads);
#elif defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
    bn_transformer_ssm_gate_avx512_range(&ctx_test, 0, num_v_heads);
#elif defined(__AVX2__)
    bn_transformer_ssm_gate_avx2_range(&ctx_test, 0, num_v_heads);
#elif defined(__wasm_simd128__)
    bn_transformer_ssm_gate_wasm_range(&ctx_test, 0, num_v_heads);
#else
    bn_transformer_ssm_gate_scalar_range(&ctx_test, 0, num_v_heads);
#endif
    float sigmoid_d = max_diff(out_ref, out_test, total);
    assert(sigmoid_d < 1e-4f);

    free(out_ref); free(out_test);
    free(z); free(norm_w);
    printf("PASSED (silu_diff=%.2e sigmoid_diff=%.2e)\n", d, sigmoid_d);
}

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
static void test_conv_native_dispatch(void) {
    printf("test_ssm_conv_native_dispatch... ");
    const int dims[] = {1, 15, 16, 17, 33, 8192, 8208};
    uint32_t seed = 137;
    for (size_t n = 0; n < sizeof(dims) / sizeof(dims[0]); n++) {
        int dim = dims[n];
        float *expected = malloc((size_t)dim * sizeof(float));
        float *actual = malloc((size_t)dim * sizeof(float));
        float *state_ref = malloc((size_t)3 * dim * sizeof(float));
        float *state = malloc((size_t)3 * dim * sizeof(float));
        float *weights = malloc((size_t)4 * dim * sizeof(float));
        assert(expected && actual && state_ref && state && weights);
        for (int sigmoid = 0; sigmoid < 2; sigmoid++) {
            fill_random(expected, dim, &seed);
            fill_random(state_ref, 3 * dim, &seed);
            fill_random(weights, 4 * dim, &seed);
            memcpy(actual, expected, (size_t)dim * sizeof(float));
            memcpy(state, state_ref, (size_t)3 * dim * sizeof(float));
            BnSSMConvCtx ref = {expected, state_ref, weights, dim, 4, sigmoid};
            BnSSMConvCtx test = {actual, state, weights, dim, 4, sigmoid};
            /* Dispatch must preserve the native ISA's convolution and SiLU
             * order regardless of the later output gate or tensor shape. */
            bn_transformer_ssm_conv_silu_avx512_range(&ref, 0, dim);
            bn_transformer_ssm_conv_silu_x86_range(&test, 0, dim);
            assert(memcmp(actual, expected, (size_t)dim * sizeof(float)) == 0);
            assert(memcmp(state, state_ref, (size_t)3 * dim * sizeof(float)) == 0);
        }
        free(expected); free(actual); free(state_ref); free(state); free(weights);
    }
    printf("PASSED\n");
}

static void test_gate_native_dispatch(void) {
    printf("test_ssm_gate_native_dispatch... ");
    const int dims[] = {1, 15, 16, 17, 128};
    uint32_t seed = 173;
    for (size_t n = 0; n < sizeof(dims) / sizeof(dims[0]); n++) {
        int dim = dims[n];
        for (int heads = 32; heads <= 64; heads += 32) {
            size_t bytes = (size_t)heads * dim * sizeof(float);
            float *expected = malloc(bytes), *actual = malloc(bytes);
            float *z = malloc(bytes), *weights = malloc((size_t)dim * sizeof(float));
            assert(expected && actual && z && weights);
            for (int sigmoid = 0; sigmoid < 2; sigmoid++) {
                fill_random(expected, heads * dim, &seed);
                fill_random(z, heads * dim, &seed);
                fill_random(weights, dim, &seed);
                memcpy(actual, expected, bytes);
                BnSSMGateCtx ref = {expected, z, weights, 1e-5f, dim, sigmoid, heads};
                BnSSMGateCtx test = {actual, z, weights, 1e-5f, dim, sigmoid, heads};
                bn_transformer_ssm_gate_avx512_range(&ref, 0, heads);
                bn_transformer_ssm_gate_x86_range(&test, 0, heads);
                assert(memcmp(actual, expected, bytes) == 0);
            }
            free(expected); free(actual); free(z); free(weights);
        }
    }
    printf("PASSED\n");
}
#endif

#ifdef __AVX2__
static void test_sigmoid_gate_rounding(void) {
    printf("test_ssm_sigmoid_gate_rounding... ");
    const int dims[] = {1, 7, 8, 15, 16, 17, 31, 32, 33, 128, 257};
    uint32_t seed = 27183;
    for (size_t n = 0; n < sizeof(dims) / sizeof(dims[0]); n++) {
        int dim = dims[n], count = 3 * dim + 2;
        size_t bytes = (size_t)count * sizeof(float);
        float *input = malloc(bytes), *actual = malloc(bytes);
        float *expected = malloc(bytes), *z = malloc(bytes);
        float *weights = malloc((size_t)dim * sizeof(float));
        assert(input && actual && expected && z && weights);
        fill_random(input, count, &seed);
        fill_random(z, count, &seed);
        fill_random(weights, dim, &seed);
        for (int i = 0; i < count; i++) z[i] *= 8;
        z[dim + 1] = -INFINITY;
        if (dim > 1) z[dim + 2] = INFINITY;
        memcpy(expected, input, bytes);
        double ss = 0;
        for (int i = 0; i < dim; i++) {
            volatile float square = input[dim + 1 + i] * input[dim + 1 + i];
            ss += (double)square;
        }
        float scale = 1.0f / sqrtf((float)(ss / dim) + 1e-5f);
        for (int i = 0; i < dim; i++) {
            volatile float norm = input[dim + 1 + i] * scale;
            volatile float weighted = norm * weights[i];
            float sigmoid = 1.0f / (1.0f + expf(-z[dim + 1 + i]));
            expected[dim + 1 + i] = weighted * sigmoid;
        }
        BnSSMGateCtx ctx = {actual + 1, z + 1, weights, 1e-5f, dim, 1, 3};
        memcpy(actual, input, bytes);
        bn_transformer_ssm_gate_avx2_range(&ctx, 1, 2);
        assert(memcmp(actual, expected, bytes) == 0);
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
        memcpy(actual, input, bytes);
        bn_transformer_ssm_gate_avx512_range(&ctx, 1, 2);
        assert(memcmp(actual, expected, bytes) == 0);
#endif
        free(input); free(actual); free(expected); free(z); free(weights);
    }
    printf("PASSED\n");
}
#endif

static void test_kda_reference(void) {
    for (int split = 0; split < 2; split++) {
        float state[18], out[6];
        memcpy(state, kda_ref_initial, sizeof(state));
        for (int t = 0; t < 4; t++) {
            for (int i = 0; i < 6; i++) out[i] = 77.0f;
            BnSSMKDACtx c = {state, out, kda_ref_q + t * 6,
                kda_ref_k + t * 6, kda_ref_v + t * 6,
                kda_ref_g + t * 6, kda_ref_beta + t * 2, 2, 3, 3,
                kda_ref_scale};
            if (split) {
                float saved[9]; memcpy(saved, state + 9, sizeof(saved));
                assert(bn_transformer_ssm_kda_step(&c, 0, 1) == 0);
                assert(memcmp(saved, state + 9, sizeof(saved)) == 0);
                for (int i = 3; i < 6; i++) assert(out[i] == 77.0f);
                assert(bn_transformer_ssm_kda_step(&c, 1, 2) == 0);
            } else assert(bn_transformer_ssm_kda_step(&c, 0, 2) == 0);
            assert(max_diff(state, kda_ref_states + t * 18, 18) < 2e-6f);
            assert(max_diff(out, kda_ref_out + t * 6, 6) < 2e-6f);
        }
    }
    /* Rectangular state and beta=1 have an exact rank-one solution. */
    float state[6] = {0}, out[3] = {0};
    const float q[] = {2, 0}, k[] = {0.5f, 0.25f};
    const float v[] = {1, 2, -3}, gate[] = {0, -5}, beta[] = {1};
    const float expected[] = {0.5f, 0.25f, 1, 0.5f, -1.5f, -0.75f};
    BnSSMKDACtx c = {state, out, q, k, v, gate, beta, 1, 2, 3, 1};
    assert(bn_transformer_ssm_kda_step(&c, 0, 1) == 0);
    assert(memcmp(state, expected, sizeof(state)) == 0);
    assert(memcmp(out, v, sizeof(out)) == 0);
    /* Invalid metadata and missing buffers cannot partially mutate outputs. */
    for (int bad = 0; bad < 10; bad++) {
        BnSSMKDACtx invalid = c;
        int start = 0, end = 1;
        if (bad == 0) invalid.head_k_dim = 0;
        if (bad == 1) invalid.head_k_dim = 257;
        if (bad == 2) invalid.head_v_dim = 0;
        if (bad == 3) invalid.head_v_dim = 257;
        if (bad == 4) invalid.num_heads = 0;
        if (bad == 5) invalid.q_scale = NAN;
        if (bad == 6) invalid.q_scale = 0;
        if (bad == 7) start = -1;
        if (bad == 8) end = 2;
        if (bad == 9) invalid.beta = NULL;
        assert(bn_transformer_ssm_kda_step(&invalid, start, end) == -1);
        assert(memcmp(state, expected, sizeof(state)) == 0);
        assert(memcmp(out, v, sizeof(out)) == 0);
    }
    BnSSMKDACtx empty = {0};
    empty.num_heads = 1; empty.head_k_dim = empty.head_v_dim = 1;
    empty.q_scale = 1;
    assert(bn_transformer_ssm_kda_step(&empty, 0, 0) == 0);
    assert(bn_transformer_ssm_kda_step(NULL, 0, 0) == -1);
    printf("KDA independent reference, head isolation and validation PASSED\n");
}

static void test_kda_coefficients(void) {
    for (int storage = 0; storage < 2; storage++)
    for (int t = 0; t < 7; t++) for (int inplace = 0; inplace < 2; inplace++) {
        float gate[6], beta[2], input[6], raw[2], parameter[2];
        for (int h = 0; h < 2; h++) parameter[h] = storage
            ? -expf(kda_coeff_a_log[h]) : kda_coeff_a_log[h];
        memcpy(input, kda_coeff_gate + t * 6, sizeof(input));
        memcpy(raw, kda_coeff_beta_raw + t * 2, sizeof(raw));
        BnSSMKDACoeffCtx c = {inplace ? input : gate, inplace ? raw : beta,
            input, kda_coeff_bias, parameter, raw, 2, 3, -5,
            (BnSSMKDAParameterStorage)storage};
        if (inplace) {
            float saved[3]; memcpy(saved, input, sizeof(saved));
            float b = raw[0];
            assert(bn_transformer_ssm_kda_coefficients(&c, 1, 2) == 0);
            assert(memcmp(saved, input, sizeof(saved)) == 0 && raw[0] == b);
            assert(bn_transformer_ssm_kda_coefficients(&c, 0, 1) == 0);
        } else assert(bn_transformer_ssm_kda_coefficients(&c, 0, 2) == 0);
        assert(max_diff(c.log_decay, kda_coeff_log_decay + t * 6, 6) < 2e-6f);
        assert(max_diff(c.beta, kda_coeff_beta + t * 2, 2) < 2e-7f);
        for (int j = 0; j < 6; j++) assert(c.log_decay[j] >= -5 && c.log_decay[j] <= 0);
    }
    float gate[6], bias[6], a[2], raw[2], out[6], beta[2];
    for (int bad = 0; bad < 18; bad++) {
        for (int j = 0; j < 6; j++) {gate[j] = 1; bias[j] = 0; out[j] = 7;}
        for (int j = 0; j < 2; j++) {a[j] = 0; raw[j] = 0; beta[j] = 8;}
        BnSSMKDACoeffCtx c = {out, beta, gate, bias, a, raw, 2, 3, -5, BN_SSM_KDA_A_LOG};
        int start = 0, end = 2;
        if (bad == 0) gate[5] = NAN;
        if (bad == 1) bias[5] = INFINITY;
        if (bad == 2) a[1] = 1000;
        if (bad == 3) a[1] = NAN;
        if (bad == 4) raw[1] = INFINITY;
        if (bad == 5) {gate[5] = FLT_MAX; bias[5] = FLT_MAX;}
        if (bad == 6) c.lower_bound = NAN;
        if (bad == 7) c.lower_bound = 0;
        if (bad == 8) c.lower_bound = 1;
        if (bad == 9) c.head_k_dim = 0;
        if (bad == 10) c.head_k_dim = 257;
        if (bad == 11) c.num_heads = 0;
        if (bad == 12) start = -1;
        if (bad == 13) end = 3;
        if (bad == 14) c.log_decay = NULL;
        if (bad == 15) c.bias = NULL;
        if (bad == 16) c.parameter_storage = (BnSSMKDAParameterStorage)9;
        if (bad == 17) {c.parameter_storage = BN_SSM_KDA_A_NEGATIVE_EXP; a[1] = 1;}
        assert(bn_transformer_ssm_kda_coefficients(&c, start, end) == -1);
        for (int j = 0; j < 6; j++) assert(out[j] == 7);
        for (int j = 0; j < 2; j++) assert(beta[j] == 8);
    }
    /* A disjoint head range must not inspect or modify unrelated bad inputs. */
    for (int j = 0; j < 6; j++) {gate[j] = bias[j] = 0; out[j] = 99;}
    a[0] = raw[0] = NAN; a[1] = raw[1] = 0; beta[0] = beta[1] = 99;
    BnSSMKDACoeffCtx isolated = {out, beta, gate, bias, a, raw, 2, 3, -5,
        BN_SSM_KDA_A_LOG};
    assert(bn_transformer_ssm_kda_coefficients(&isolated, 1, 2) == 0);
    for (int j = 0; j < 3; j++) assert(out[j] == 99 && out[j + 3] == -2.5f);
    assert(beta[0] == 99 && beta[1] == 0.5f);
    assert(bn_transformer_ssm_kda_coefficients(&isolated, 0, 2) == -1);
    for (int j = 0; j < 3; j++) assert(out[j] == 99 && out[j + 3] == -2.5f);
    assert(beta[0] == 99 && beta[1] == 0.5f);
    /* Finite amplitudes may saturate their product; underflow to zero is valid. */
    for (int j = 0; j < 6; j++) {gate[j] = 1e30f; bias[j] = 0;}
    a[0] = 80; a[1] = -1000; raw[0] = 1000; raw[1] = -1000;
    BnSSMKDACoeffCtx c = {out, beta, gate, bias, a, raw, 2, 3, -5, BN_SSM_KDA_A_LOG};
    assert(bn_transformer_ssm_kda_coefficients(&c, 0, 2) == 0);
    for (int j = 0; j < 3; j++) assert(out[j] == -5 && out[j + 3] == -2.5f);
    assert(beta[0] == 1 && beta[1] == 0);
    c.log_decay = c.beta = NULL; c.gate = c.bias = c.decay_parameter = c.beta_raw = NULL;
    assert(bn_transformer_ssm_kda_coefficients(&c, 1, 1) == 0);
    assert(bn_transformer_ssm_kda_coefficients(NULL, 0, 0) == -1);
    printf("KDA coefficients reference, aliases, isolation and atomic rejection PASSED\n");
}

static void test_kda_sum_eps_norm(void) {
    void (*functions[])(void *, int, int) = {
        bn_transformer_ssm_l2norm_scalar_range,
#ifdef __AVX2__
        bn_transformer_ssm_l2norm_avx2_range,
#endif
#ifdef __ARM_NEON
        bn_transformer_ssm_l2norm_neon_range,
#endif
#ifdef __wasm_simd128__
        bn_transformer_ssm_l2norm_wasm_range,
#endif
    };
    for (size_t f = 0; f < sizeof(functions) / sizeof(functions[0]); f++)
    for (int t = 0; t < 5; t++) {
        float q[6], k[6], saved_q[3], saved_k[3];
        memcpy(q, kda_norm_q + 6 * t, sizeof(q));
        memcpy(k, kda_norm_k + 6 * t, sizeof(k));
        memcpy(saved_q, q + 3, sizeof(saved_q));
        memcpy(saved_k, k + 3, sizeof(saved_k));
        BnSSML2NormCtx c = {q, k, 1e-6f, 3, BN_SSM_QK_NORM_L2_SUM_EPS};
        functions[f](&c, 0, 1);
        assert(memcmp(saved_q, q + 3, sizeof(saved_q)) == 0);
        assert(memcmp(saved_k, k + 3, sizeof(saved_k)) == 0);
        functions[f](&c, 1, 2);
        for (int j = 0; j < 6; j++) {
            assert(isfinite(q[j]) && isfinite(k[j]));
            assert(fabsf(q[j] - kda_norm_nq[6 * t + j]) < 2e-6f);
            assert(fabsf(k[j] - kda_norm_nk[6 * t + j]) < 2e-6f);
        }
    }
    float q[] = {1, 1}, k[] = {1e-3f, 0};
    BnSSML2NormCtx c = {q, k, 1e-6f, 2, BN_SSM_QK_NORM_L2_SUM_EPS};
    bn_transformer_ssm_l2norm_scalar_range(&c, 0, 1);
    assert(fabsf(k[0] - 0.70710678f) < 2e-6f);
    uint32_t bits; memcpy(&bits, q, sizeof(bits));
    assert((bits & 0xffffu) != 0); /* No hidden BF16 store after normalization. */
    float before_q[2], before_k[2];
    memcpy(before_q, q, sizeof(q)); memcpy(before_k, k, sizeof(k));
    c.eps = NAN; bn_transformer_ssm_qk_sum_eps_range(&c, 0, 1);
    c.eps = 1e-6f; c.head_dim = 100000;
    bn_transformer_ssm_qk_sum_eps_range(&c, 0, 1);
    assert(memcmp(before_q, q, sizeof(q)) == 0 && memcmp(before_k, k, sizeof(k)) == 0);
    bn_transformer_ssm_qk_sum_eps_range(NULL, 0, 0);
    puts("Direct F32 sum-epsilon Q/K normalization and backend routing PASSED");
}

int main(void) {
#ifdef __AVX2__
    test_sigmoid_gate_rounding();
#endif
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
    test_conv_native_dispatch();
    test_gate_native_dispatch();
#endif
    printf("=== SSM Kernel Equivalence Tests ===\n");
#ifdef __ARM_NEON
    printf("Platform: NEON vs scalar\n");
#elif defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
    printf("Platform: AVX512 vs scalar\n");
#elif defined(__AVX2__)
    printf("Platform: AVX2 vs scalar\n");
#elif defined(__wasm_simd128__)
    printf("Platform: WASM SIMD128 vs scalar\n");
#else
    printf("Platform: scalar vs scalar (no SIMD)\n");
#endif

    test_conv_silu();
    test_l2norm();
    test_qk_rmsnorm_reference();
    test_delta();
    test_kda_reference();
    test_kda_sum_eps_norm();
    test_kda_coefficients();
#ifdef __ARM_NEON
    test_delta_neon_reduction_order();
#endif
    test_gate();

    printf("All SSM kernel tests passed!\n");
    return 0;
}
