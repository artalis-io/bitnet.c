#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <assert.h>
#include <math.h>

float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n) {
    // Find absolute max via AVX2
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int i = 0;
    for (; i + 31 < n; i += 32) {
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask));
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i + 8), sign_mask));
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i + 16), sign_mask));
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i + 24), sign_mask));
    }
    for (; i + 7 < n; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask));
    float amax = bn_avx2_hmax_ps(vmax);
    for (; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }

    if (amax == 0.0f) {
        memset(x_q, 0, n);
        return 0.0f;
    }

    float scale = amax / (float)BN_I8_MAX;
    float inv_scale = (float)BN_I8_MAX / amax;
    __m256 vinv = _mm256_set1_ps(inv_scale);

    // Lane-crossing fixup permutation for packs: AVX2 packs operates within
    // 128-bit lanes, so after two packs (32->16->8) the order is [0,4,1,5,2,6,3,7]
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vinv));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 16), vinv));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 24), vinv));
        __m256i s01 = _mm256_packs_epi32(i0, i1);   // 32->16, within lanes
        __m256i s23 = _mm256_packs_epi32(i2, i3);
        __m256i b = _mm256_packs_epi16(s01, s23);    // 16->8, within lanes
        b = _mm256_permutevar8x32_epi32(b, perm);    // fix lane crossing
        _mm256_storeu_si256((__m256i *)(x_q + i), b);
    }
    for (; i < n; i++) {
        int v = (int)roundf(x[i] * inv_scale);
        x_q[i] = (int8_t)(v < -BN_I8_MAX ? -BN_I8_MAX : (v > BN_I8_MAX ? BN_I8_MAX : v));
    }
    return scale;
}

void bn_quant_f16_rows_to_i8(const uint16_t *f16, int8_t *i8_out,
                              float *scales_out, int n_rows, int dim) {
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int r = 0; r < n_rows; r++) {
        const uint16_t *row = f16 + (size_t)r * dim;
        int8_t *out = i8_out + (size_t)r * dim;

        // F16C: convert F16->F32 and find amax
        __m256 vmax = _mm256_setzero_ps();
        int d = 0;
        for (; d + 7 < dim; d += 8) {
            __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d)));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, sign_mask));
        }
        float amax = bn_avx2_hmax_ps(vmax);
        for (; d < dim; d++) {
            float v = bn_fp16_to_fp32(row[d]);
            float a = v < 0 ? -v : v;
            if (a > amax) amax = a;
        }

        if (amax == 0.0f) {
            memset(out, 0, dim);
            scales_out[r] = 0.0f;
            continue;
        }

        float scale = amax / (float)BN_I8_MAX;
        float inv_scale = (float)BN_I8_MAX / amax;
        __m256 vinv = _mm256_set1_ps(inv_scale);
        scales_out[r] = scale;

        d = 0;
        for (; d + 31 < dim; d += 32) {
            __m256 f0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d)));
            __m256 f1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d + 8)));
            __m256 f2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d + 16)));
            __m256 f3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d + 24)));
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, vinv));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, vinv));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(f2, vinv));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(f3, vinv));
            __m256i s01 = _mm256_packs_epi32(i0, i1);
            __m256i s23 = _mm256_packs_epi32(i2, i3);
            __m256i b = _mm256_packs_epi16(s01, s23);
            b = _mm256_permutevar8x32_epi32(b, perm);
            _mm256_storeu_si256((__m256i *)(out + d), b);
        }
        for (; d < dim; d++) {
            float v = bn_fp16_to_fp32(row[d]);
            int q = (int)roundf(v * inv_scale);
            out[d] = (int8_t)(q < -BN_I8_MAX ? -BN_I8_MAX : (q > BN_I8_MAX ? BN_I8_MAX : q));
        }
    }
}

// Per-block Q8_0 quantization (AVX2 version)
// Q8_K quantization: 256-element super-blocks with bsums for Q4_K SDOT.
void bn_quant_x_to_q8k(const float *x, int8_t *x_q, float *x_d,
                         int16_t *x_bsums, int n) {
    assert(n % BN_QK_K == 0 && "bn_quant_x_to_q8k: n must be multiple of BN_QK_K");
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    int n_sb = n / BN_QK_K;

    for (int sb = 0; sb < n_sb; sb++) {
        const float *xb = x + sb * BN_QK_K;
        int8_t *qb = x_q + sb * BN_QK_K;

        // Find amax over 256 elements
        __m256 vmax = _mm256_setzero_ps();
        for (int i = 0; i < BN_QK_K; i += 32) {
            __m256 v0 = _mm256_and_ps(_mm256_loadu_ps(xb + i), sign_mask);
            __m256 v1 = _mm256_and_ps(_mm256_loadu_ps(xb + i + 8), sign_mask);
            __m256 v2 = _mm256_and_ps(_mm256_loadu_ps(xb + i + 16), sign_mask);
            __m256 v3 = _mm256_and_ps(_mm256_loadu_ps(xb + i + 24), sign_mask);
            vmax = _mm256_max_ps(vmax, _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3)));
        }
        float amax = bn_avx2_hmax_ps(vmax);

        if (amax == 0.0f) {
            _mm256_storeu_si256((__m256i *)qb, _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(qb + 32), _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(qb + 64), _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(qb + 96), _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(qb + 128), _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(qb + 160), _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(qb + 192), _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(qb + 224), _mm256_setzero_si256());
            x_d[sb] = 0.0f;
            for (int g = 0; g < 16; g++) x_bsums[sb * 16 + g] = 0;
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_d[sb] = amax / 127.0f;
        __m256 vinv = _mm256_set1_ps(inv_scale);

        // Quantize 256 elements in 32-element chunks
        int16_t *bsums = x_bsums + sb * 16;
        for (int g = 0; g < 8; g++) {
            const float *gx = xb + g * 32;
            int8_t *gq = qb + g * 32;

            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(gx), vinv));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(gx + 8), vinv));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(gx + 16), vinv));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(gx + 24), vinv));
            __m256i s01 = _mm256_packs_epi32(i0, i1);
            __m256i s23 = _mm256_packs_epi32(i2, i3);
            __m256i packed = _mm256_packs_epi16(s01, s23);
            packed = _mm256_permutevar8x32_epi32(packed, perm);
            _mm256_storeu_si256((__m256i *)gq, packed);

            // Compute bsums: sum of 16-element halves
            // Sum first 16 bytes
            __m128i lo = _mm256_castsi256_si128(packed);
            __m128i hi = _mm256_extracti128_si256(packed, 1);
            // Widen to i16 and sum pairs
            __m128i lo16_0 = _mm_cvtepi8_epi16(lo);
            __m128i lo16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(lo, 8));
            __m128i sum16_0 = _mm_add_epi16(lo16_0, lo16_1);
            __m128i sum32_0 = _mm_add_epi32(_mm_cvtepi16_epi32(sum16_0),
                                             _mm_cvtepi16_epi32(_mm_srli_si128(sum16_0, 8)));
            int32_t bsum0 = _mm_extract_epi32(sum32_0, 0) + _mm_extract_epi32(sum32_0, 1)
                          + _mm_extract_epi32(sum32_0, 2) + _mm_extract_epi32(sum32_0, 3);
            bsums[g * 2] = (int16_t)bsum0;

            __m128i hi16_0 = _mm_cvtepi8_epi16(hi);
            __m128i hi16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(hi, 8));
            __m128i sum16_1 = _mm_add_epi16(hi16_0, hi16_1);
            __m128i sum32_1 = _mm_add_epi32(_mm_cvtepi16_epi32(sum16_1),
                                             _mm_cvtepi16_epi32(_mm_srli_si128(sum16_1, 8)));
            int32_t bsum1 = _mm_extract_epi32(sum32_1, 0) + _mm_extract_epi32(sum32_1, 1)
                          + _mm_extract_epi32(sum32_1, 2) + _mm_extract_epi32(sum32_1, 3);
            bsums[g * 2 + 1] = (int16_t)bsum1;
        }
    }
}

void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n) {
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    int n_blocks = n / 32;

    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * 32;
        int8_t *qb = x_q + b * 32;

        __m256 v0 = _mm256_and_ps(_mm256_loadu_ps(xb), sign_mask);
        __m256 v1 = _mm256_and_ps(_mm256_loadu_ps(xb + 8), sign_mask);
        __m256 v2 = _mm256_and_ps(_mm256_loadu_ps(xb + 16), sign_mask);
        __m256 v3 = _mm256_and_ps(_mm256_loadu_ps(xb + 24), sign_mask);
        __m256 vmax = _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3));
        float amax = bn_avx2_hmax_ps(vmax);

        if (amax == 0.0f) {
            _mm256_storeu_si256((__m256i *)qb, _mm256_setzero_si256());
            x_scales[b] = 0.0f;
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_scales[b] = amax / 127.0f;

        __m256 vinv = _mm256_set1_ps(inv_scale);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb), vinv));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 8), vinv));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 16), vinv));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 24), vinv));
        __m256i s01 = _mm256_packs_epi32(i0, i1);
        __m256i s23 = _mm256_packs_epi32(i2, i3);
        __m256i packed = _mm256_packs_epi16(s01, s23);
        packed = _mm256_permutevar8x32_epi32(packed, perm);
        _mm256_storeu_si256((__m256i *)qb, packed);
    }
}
