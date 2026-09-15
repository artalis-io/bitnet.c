#include "transformer_math_internal.h"
#include "transformer_cpu_features_internal.h"
#include "simd_helpers.h"
#include <stddef.h>

void bn_transformer_scaled_silu(float *x, float scale, int size) {
    for (int i = 0; i < size; i++)
        x[i] *= scale;
    int i = 0;
#if BN_TRANSFORMER_CPU_HAS_AVX512
    for (; i + 15 < size; i += 16)
        _mm512_storeu_ps(x + i,
            bn_avx512_fast_silu_ps(_mm512_loadu_ps(x + i)));
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    for (; i + 7 < size; i += 8) {
        const __m256 value = _mm256_loadu_ps(x + i);
        const __m256 exp_neg = bn_avx2_fast_exp_avx512_ps(
            _mm256_sub_ps(_mm256_setzero_ps(), value));
        _mm256_storeu_ps(x + i, _mm256_div_ps(
            value, _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg)));
    }
#endif
    for (; i < size; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

void bn_transformer_dilated_conv_silu(float *out, const float *current,
    const float *history, const float *weights, int size, int kernel,
    int dilation) {
    for (int i = 0; i < size; i++) {
#if BN_TRANSFORMER_CPU_HAS_AVX2
        float sum = 0.0f;
        for (int k = 0; k < kernel; k++) {
            float input = k == kernel - 1 ? current[i] :
                history[(size_t)k * dilation * size + i];
            volatile float product = weights[(size_t)i * kernel + k] * input;
            sum = k == 0 ? product : sum + product;
        }
#else
        float sum = weights[(size_t)i * kernel + kernel - 1] * current[i];
        for (int k = 0; k < kernel - 1; k++)
            sum += weights[(size_t)i * kernel + k] *
                history[(size_t)k * dilation * size + i];
#endif
        out[i] = sum;
    }
    int i = 0;
#if BN_TRANSFORMER_CPU_HAS_AVX512
    for (; i + 15 < size; i += 16)
        _mm512_storeu_ps(out + i,
            bn_avx512_fast_silu_ps(_mm512_loadu_ps(out + i)));
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    for (; i + 7 < size; i += 8) {
        const __m256 value = _mm256_loadu_ps(out + i);
        const __m256 exp_neg = bn_avx2_fast_exp_avx512_ps(
            _mm256_sub_ps(_mm256_setzero_ps(), value));
        _mm256_storeu_ps(out + i, _mm256_div_ps(
            value, _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg)));
    }
#endif
    for (; i < size; i++)
        out[i] = out[i] / (1.0f + expf(-out[i]));
}

void bn_transformer_scaled_branch_add(float *x, const float *value,
    float scale, const float *branch, int size) {
    for (int i = 0; i < size; i++) {
#if BN_TRANSFORMER_CPU_HAS_AVX2
        volatile float product = value[i] * scale;
        x[i] += product + branch[i];
#else
        x[i] += value[i] * scale;
        x[i] += branch[i];
#endif
    }
}

float bn_transformer_sum_products(const float *a, const float *b, int size) {
#if BN_TRANSFORMER_CPU_HAS_AVX2
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        // Match an F32 multiply followed by a separately reduced sum;
        // neither an FMA nor a double-precision product is equivalent.
        volatile float product = a[i] * b[i];
        sum += (double)product;
    }
    return (float)sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
        sum += a[i] * b[i];
    return sum;
#endif
}

void bn_transformer_softmax(float *x, int size) {
#if BN_TRANSFORMER_CPU_HAS_NEON
    bn_transformer_softmax_neon(x, size);
#elif BN_TRANSFORMER_CPU_HAS_AVX512
    bn_transformer_softmax_avx512(x, size);
#elif defined(__AVX2__) && defined(__FMA__)
    bn_transformer_softmax_avx2(x, size);
#else
    bn_transformer_softmax_scalar(x, size);
#endif
}
