#include "transformer_math_internal.h"
#include "simd_helpers.h"
#include <math.h>
#include <stdint.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

float bn_transformer_fast_exp_scalar(float x) {
    const float r = 0x1.8p23f;
    float z = fmaf(x, 0x1.715476p+0f, r);
    float n = z - r;
    float b = fmaf(-n, 0x1.7f7d1cp-20f,
                   fmaf(-n, 0x1.62e4p-1f, x));
    union {
        float f;
        uint32_t u;
    } bits = {z};
    uint32_t e = bits.u << 23;
    bits.u = e + UINT32_C(0x3f800000);
    float k = bits.f;
    float u = b * b;
    float j = fmaf(0x1.573e2ep-5f, 1.0f,
                   0x1.0e4020p-7f * b);
    j = fmaf(j, u,
             fmaf(0x1.555e66p-3f, b, 0x1.fffdb6p-2f));
    j = fmaf(j, u, 0x1.ffffecp-1f * b);

    if (fabsf(n) <= 126.0f)
        return fmaf(k, j, k);

    uint32_t d = n <= 0.0f ? UINT32_C(0x82000000) : 0;
    bits.u = d + UINT32_C(0x7f000000);
    float s1 = bits.f;
    bits.u = e - d;
    float s2 = bits.f;
    if (fabsf(n) > 192.0f)
        return s1 * s1;
    return fmaf(s2, j, s2) * s1;
}

void bn_transformer_softmax_scalar(float *x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float e0 = bn_transformer_fast_exp_scalar(x[i] - max_val);
        float e1 = bn_transformer_fast_exp_scalar(x[i + 1] - max_val);
        float e2 = bn_transformer_fast_exp_scalar(x[i + 2] - max_val);
        float e3 = bn_transformer_fast_exp_scalar(x[i + 3] - max_val);
        x[i] = e0;
        x[i + 1] = e1;
        x[i + 2] = e2;
        x[i + 3] = e3;
        sum += (double)((e0 + e1) + (e2 + e3));
    }
    if (i < size) {
        float tail[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int n = size - i;
        for (int j = 0; j < n; j++) {
            tail[j] = bn_transformer_fast_exp_scalar(x[i + j] - max_val);
            x[i + j] = tail[j];
        }
        sum += (double)((tail[0] + tail[1]) + (tail[2] + tail[3]));
    }
    float inv = (float)(1.0 / sum);
    for (i = 0; i < size; i++) x[i] *= inv;
}

#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 fast_exp_avx2(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x,
        _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(
        n, _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(
        e, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
        _mm256_set1_ps(126.0f), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                            _mm256_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                            _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
        return _mm256_fmadd_ps(j, k, k);
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(
            n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32((int)UINT32_C(0x82000000)));
    const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(
        g, _mm256_set1_epi32((int)UINT32_C(0x7f000000))));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
        _mm256_set1_ps(192.0f), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(d), _mm256_or_ps(
            _mm256_and_ps(_mm256_castsi256_ps(c),
                          _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
            _mm256_andnot_ps(_mm256_castsi256_ps(c),
                             _mm256_fmadd_ps(k, j, k)))));
}

static inline float reduce_add_8_avx2(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(value, 1),
                           _mm256_castps256_ps128(value));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static inline float reduce_add_16_avx2(__m256 low, __m256 high) {
    return reduce_add_8_avx2(_mm256_add_ps(high, low));
}

void bn_transformer_softmax_avx2(float *x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];

    double sum = 0.0;
    int i = 0;
    const __m256 max_v = _mm256_set1_ps(max_val);
    for (; i + 15 < size; i += 16) {
        __m256 low = fast_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(x + i), max_v));
        __m256 high = fast_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(x + i + 8), max_v));
        _mm256_storeu_ps(x + i, low);
        _mm256_storeu_ps(x + i + 8, high);
        sum += (double)reduce_add_16_avx2(low, high);
    }
    if (i < size) {
        float tail[16];
        int count = size - i;
        for (int j = 0; j < 16; j++)
            tail[j] = j < count ? x[i + j] : -INFINITY;
        __m256 low = fast_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(tail), max_v));
        __m256 high = fast_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(tail + 8), max_v));
        _mm256_storeu_ps(tail, low);
        _mm256_storeu_ps(tail + 8, high);
        for (int j = 0; j < count; j++)
            x[i + j] = tail[j];
        sum += (double)reduce_add_16_avx2(low, high);
    }
    float inv = (float)(1.0 / sum);
    for (i = 0; i < size; i++) x[i] *= inv;
}

void bn_transformer_softmax_avx2_batch(float *x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];

    double sum = 0.0;
    int i = 0;
    const __m256 max_v = _mm256_set1_ps(max_val);
    for (; i + 7 < size; i += 8) {
        __m256 value = fast_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(x + i), max_v));
        _mm256_storeu_ps(x + i, value);
        sum += (double)reduce_add_8_avx2(value);
    }
    if (i < size) {
        float tail[8];
        int count = size - i;
        for (int j = 0; j < 8; j++)
            tail[j] = j < count ? x[i + j] : -INFINITY;
        __m256 value = fast_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(tail), max_v));
        _mm256_storeu_ps(tail, value);
        for (int j = 0; j < count; j++) x[i + j] = tail[j];
        sum += (double)reduce_add_8_avx2(value);
    }
    float inv = (float)(1.0 / sum);
    for (i = 0; i < size; i++) x[i] *= inv;
}
#else
void bn_transformer_softmax_avx2(float *x, int size) {
    bn_transformer_softmax_scalar(x, size);
}
void bn_transformer_softmax_avx2_batch(float *x, int size) {
    bn_transformer_softmax_scalar(x, size);
}
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
void bn_transformer_softmax_avx512(float *x, int size) {
    if (size <= 0)
        return;
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val)
            max_val = x[i];

    double sum = 0.0;
    int i = 0;
    const __m512 max_v = _mm512_set1_ps(max_val);
    for (; i + 15 < size; i += 16) {
        __m512 value = bn_avx512_fast_exp_ps(
            _mm512_sub_ps(_mm512_loadu_ps(x + i), max_v));
        _mm512_storeu_ps(x + i, value);
        sum += (double)_mm512_reduce_add_ps(value);
    }
    if (i < size) {
        float tail[16];
        int count = size - i;
        for (int j = 0; j < 16; j++)
            tail[j] = j < count ? x[i + j] : -INFINITY;
        __m512 value = bn_avx512_fast_exp_ps(
            _mm512_sub_ps(_mm512_loadu_ps(tail), max_v));
        _mm512_storeu_ps(tail, value);
        for (int j = 0; j < count; j++)
            x[i + j] = tail[j];
        sum += (double)_mm512_reduce_add_ps(value);
    }

    float inv = (float)(1.0 / sum);
    int vector_end = size & ~63;
    const __m512 inv_v = _mm512_set1_ps(inv);
    for (i = 0; i < vector_end; i += 64) {
        _mm512_storeu_ps(x + i, _mm512_mul_ps(
            _mm512_loadu_ps(x + i), inv_v));
        _mm512_storeu_ps(x + i + 16, _mm512_mul_ps(
            _mm512_loadu_ps(x + i + 16), inv_v));
        _mm512_storeu_ps(x + i + 32, _mm512_mul_ps(
            _mm512_loadu_ps(x + i + 32), inv_v));
        _mm512_storeu_ps(x + i + 48, _mm512_mul_ps(
            _mm512_loadu_ps(x + i + 48), inv_v));
    }
    for (; i < size; i++)
        x[i] *= inv;
}
#else
void bn_transformer_softmax_avx512(float *x, int size) {
    bn_transformer_softmax_avx2(x, size);
}
#endif
