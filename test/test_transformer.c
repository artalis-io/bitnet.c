#include "quant.h"
#include "simd_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Test the helper functions that would be internal to transformer.c
// We re-implement them here for testing since they're static in transformer.c

static void rmsnorm(float *out, const float *x, const float *w, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void rope(float *vec, int dim, int head_size, int pos, float theta) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(theta, (float)head_dim / (float)head_size);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

// --- Tests ---

static void test_rmsnorm(void) {
    printf("test_rmsnorm... ");

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    rmsnorm(out, x, w, 4, 1e-5f);

    // RMSNorm: x * 1/rms(x), where rms = sqrt(mean(x^2))
    // mean(x^2) = (1+4+9+16)/4 = 7.5
    // rms = sqrt(7.5) ≈ 2.7386
    // scale = 1/rms ≈ 0.3651
    float rms = sqrtf(7.5f + 1e-5f);
    float scale = 1.0f / rms;

    for (int i = 0; i < 4; i++) {
        float expected = x[i] * scale;
        assert(fabsf(out[i] - expected) < 1e-5f);
    }

    printf("PASSED\n");
}

static void test_softmax(void) {
    printf("test_softmax... ");

    float x[] = {1.0f, 2.0f, 3.0f};
    softmax(x, 3);

    // Check probabilities sum to 1
    float sum = x[0] + x[1] + x[2];
    assert(fabsf(sum - 1.0f) < 1e-5f);

    // Check monotonicity
    assert(x[0] < x[1]);
    assert(x[1] < x[2]);

    // Check specific values
    // softmax([1,2,3]) = exp([1,2,3]) / sum(exp([1,2,3]))
    float e1 = expf(1), e2 = expf(2), e3 = expf(3);
    float esum = e1 + e2 + e3;
    assert(fabsf(x[0] - e1/esum) < 1e-5f);
    assert(fabsf(x[1] - e2/esum) < 1e-5f);
    assert(fabsf(x[2] - e3/esum) < 1e-5f);

    printf("PASSED\n");
}

static void test_rope(void) {
    printf("test_rope... ");

    // Test that RoPE at pos=0 is identity
    float vec[] = {1.0f, 0.0f, 0.0f, 1.0f};
    rope(vec, 4, 4, 0, 10000.0f);

    // At pos=0, angle=0, cos=1, sin=0, so output should equal input
    assert(fabsf(vec[0] - 1.0f) < 1e-5f);
    assert(fabsf(vec[1] - 0.0f) < 1e-5f);
    assert(fabsf(vec[2] - 0.0f) < 1e-5f);
    assert(fabsf(vec[3] - 1.0f) < 1e-5f);

    // Test that RoPE preserves vector magnitude
    float vec2[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float mag_before = 0;
    for (int i = 0; i < 4; i++) mag_before += vec2[i] * vec2[i];

    rope(vec2, 4, 4, 5, 10000.0f);

    float mag_after = 0;
    for (int i = 0; i < 4; i++) mag_after += vec2[i] * vec2[i];

    assert(fabsf(mag_before - mag_after) < 1e-4f);

    printf("PASSED\n");
}

static void test_fp16_embed(void) {
    printf("test_fp16_embed... ");

    // Test FP16 → F32 conversion for embedding lookup
    uint16_t fp16_vals[] = {0x3C00, 0x4000, 0xBC00, 0x0000};  // 1.0, 2.0, -1.0, 0.0
    float f32_vals[4];
    for (int i = 0; i < 4; i++) {
        f32_vals[i] = bn_fp16_to_fp32(fp16_vals[i]);
    }

    assert(fabsf(f32_vals[0] - 1.0f) < 1e-6f);
    assert(fabsf(f32_vals[1] - 2.0f) < 1e-6f);
    assert(fabsf(f32_vals[2] - (-1.0f)) < 1e-6f);
    assert(fabsf(f32_vals[3] - 0.0f) < 1e-6f);

    printf("PASSED\n");
}

static void test_fast_silu(void) {
    printf("test_fast_silu... ");

#ifdef __ARM_NEON
    float vals[12] = {
        -8.0f, -4.0f, -1.5f, -0.25f,
         0.0f,  0.25f, 1.0f,  2.0f,
         4.0f,  8.0f, 12.0f, -12.0f
    };
    float out[12];
    for (int i = 0; i < 12; i += 4) {
        float32x4_t v = vld1q_f32(vals + i);
        vst1q_f32(out + i, bn_neon_fast_silu_f32(v));
    }
    for (int i = 0; i < 12; i++) {
        float exact = vals[i] / (1.0f + expf(-vals[i]));
        assert(fabsf(out[i] - exact) < 2e-3f);
    }
#elif defined(__AVX2__)
    float vals[8] = {-8.0f, -4.0f, -1.5f, -0.25f, 0.25f, 1.0f, 4.0f, 8.0f};
    float out[8];
    __m256 v = _mm256_loadu_ps(vals);
    _mm256_storeu_ps(out, bn_avx2_fast_silu_ps(v));
    for (int i = 0; i < 8; i++) {
        float exact = vals[i] / (1.0f + expf(-vals[i]));
        assert(fabsf(out[i] - exact) < 2e-3f);
    }
#endif

    printf("PASSED\n");
}

int main(void) {
    printf("=== Transformer Tests ===\n");
    test_rmsnorm();
    test_softmax();
    test_rope();
    test_fp16_embed();
    test_fast_silu();
    printf("All transformer tests passed!\n");
    return 0;
}
