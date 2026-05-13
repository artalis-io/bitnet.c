#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <math.h>
#include <assert.h>

/* Fused RMSNorm + Q8_K quantization (AVX2).
 *
 * Replaces the 4-pass sequence:
 *   1. RMSNorm pass 1: sum(x²) + store x*w
 *   2. RMSNorm pass 2: scale by 1/sqrt(ss)
 *   3. Q8K pass 1: find amax per 256-block
 *   4. Q8K pass 2: quantize to int8
 *
 * With 2 passes:
 *   1. sum(x²) + store x*w (unchanged — need full sum before scaling)
 *   2. scale by ss + find amax + quantize + compute bsums (fused)
 *
 * Saves 2 full reads of the dim-width activation per call (×2 per layer
 * for attention norm + FFN norm, ×32 layers = 128 avoided sweeps/token).
 */
void bn_quant_rmsnorm_q8k_avx2(const float *x, const float *w, int dim, float eps,
                                  float *xb_out, int8_t *x_q, float *x_d,
                                  int16_t *x_bsums) {
    assert(dim % BN_QK_K == 0);

    /* Pass 1: compute sum(x²) and store unscaled x*w into xb_out */
    __m256 sum_sq = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(w + i);
        sum_sq = _mm256_fmadd_ps(xv, xv, sum_sq);
        _mm256_storeu_ps(xb_out + i, _mm256_mul_ps(xv, wv));
    }
    float ss = 1.0f / sqrtf(bn_avx2_hsum_ps(sum_sq) / dim + eps);

    /* Pass 2: scale by ss + Q8K quantize in one sweep.
     * Process in BN_QK_K (256) super-blocks. */
    __m256 ss_v = _mm256_set1_ps(ss);
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    int n_sb = dim / BN_QK_K;

    for (int sb = 0; sb < n_sb; sb++) {
        float *xb = xb_out + sb * BN_QK_K;
        int8_t *qb = x_q + sb * BN_QK_K;

        /* Scale by ss AND find amax in one loop over 256 elements */
        __m256 vmax = _mm256_setzero_ps();
        for (int i = 0; i < BN_QK_K; i += 32) {
            __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(xb + i), ss_v);
            __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(xb + i + 8), ss_v);
            __m256 s2 = _mm256_mul_ps(_mm256_loadu_ps(xb + i + 16), ss_v);
            __m256 s3 = _mm256_mul_ps(_mm256_loadu_ps(xb + i + 24), ss_v);
            _mm256_storeu_ps(xb + i, s0);
            _mm256_storeu_ps(xb + i + 8, s1);
            _mm256_storeu_ps(xb + i + 16, s2);
            _mm256_storeu_ps(xb + i + 24, s3);
            vmax = _mm256_max_ps(vmax, _mm256_max_ps(
                _mm256_max_ps(_mm256_and_ps(s0, sign_mask), _mm256_and_ps(s1, sign_mask)),
                _mm256_max_ps(_mm256_and_ps(s2, sign_mask), _mm256_and_ps(s3, sign_mask))));
        }
        float amax = bn_avx2_hmax_ps(vmax);

        if (amax == 0.0f) {
            for (int i = 0; i < BN_QK_K; i += 32)
                _mm256_storeu_si256((__m256i *)(qb + i), _mm256_setzero_si256());
            x_d[sb] = 0.0f;
            for (int g = 0; g < 16; g++) x_bsums[sb * 16 + g] = 0;
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_d[sb] = amax / 127.0f;
        __m256 vinv = _mm256_set1_ps(inv_scale);

        /* Quantize + compute bsums (8 groups of 32 elements) */
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

            /* Bsums: sum of first 16 bytes and second 16 bytes */
            __m128i lo = _mm256_castsi256_si128(packed);
            __m128i hi = _mm256_extracti128_si256(packed, 1);
            __m128i lo16_0 = _mm_cvtepi8_epi16(lo);
            __m128i lo16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(lo, 8));
            __m128i sum16_0 = _mm_add_epi16(lo16_0, lo16_1);
            __m128i sum32_0 = _mm_add_epi32(_mm_cvtepi16_epi32(sum16_0),
                                             _mm_cvtepi16_epi32(_mm_srli_si128(sum16_0, 8)));
            bsums[g * 2] = (int16_t)(_mm_extract_epi32(sum32_0, 0) + _mm_extract_epi32(sum32_0, 1)
                                   + _mm_extract_epi32(sum32_0, 2) + _mm_extract_epi32(sum32_0, 3));

            __m128i hi16_0 = _mm_cvtepi8_epi16(hi);
            __m128i hi16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(hi, 8));
            __m128i sum16_1 = _mm_add_epi16(hi16_0, hi16_1);
            __m128i sum32_1 = _mm_add_epi32(_mm_cvtepi16_epi32(sum16_1),
                                             _mm_cvtepi16_epi32(_mm_srli_si128(sum16_1, 8)));
            bsums[g * 2 + 1] = (int16_t)(_mm_extract_epi32(sum32_1, 0) + _mm_extract_epi32(sum32_1, 1)
                                        + _mm_extract_epi32(sum32_1, 2) + _mm_extract_epi32(sum32_1, 3));
        }
    }
}
