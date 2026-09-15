#include "quant_ctx.h"
#include "simd_helpers.h"
#include <string.h>
#include <immintrin.h>
#include <assert.h>
#include <math.h>

static inline __m256i bn_avx2_round_half_away_epi32(__m256 v) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 zero = _mm256_setzero_ps();
    __m256 bias = _mm256_blendv_ps(_mm256_sub_ps(zero, half), half,
                                   _mm256_cmp_ps(v, zero, _CMP_GE_OQ));
    return _mm256_cvttps_epi32(_mm256_add_ps(v, bias));
}

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
        __m256i i0 = bn_avx2_round_half_away_epi32(
            _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
        __m256i i1 = bn_avx2_round_half_away_epi32(
            _mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vinv));
        __m256i i2 = bn_avx2_round_half_away_epi32(
            _mm256_mul_ps(_mm256_loadu_ps(x + i + 16), vinv));
        __m256i i3 = bn_avx2_round_half_away_epi32(
            _mm256_mul_ps(_mm256_loadu_ps(x + i + 24), vinv));
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
            __m256i i0 = bn_avx2_round_half_away_epi32(_mm256_mul_ps(f0, vinv));
            __m256i i1 = bn_avx2_round_half_away_epi32(_mm256_mul_ps(f1, vinv));
            __m256i i2 = bn_avx2_round_half_away_epi32(_mm256_mul_ps(f2, vinv));
            __m256i i3 = bn_avx2_round_half_away_epi32(_mm256_mul_ps(f3, vinv));
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
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m128i bsum_bias = _mm_set1_epi8((char)0x80);
    __m128i bsum_zero = _mm_setzero_si128();
    int n_sb = n / BN_QK_K;

    for (int sb = 0; sb < n_sb; sb++) {
        const float *xb = x + sb * BN_QK_K;
        int8_t *qb = x_q + sb * BN_QK_K;

        float max = 0.0f;
        float amax = 0.0f;
        for (int i = 0; i < BN_QK_K; i++) {
            float ax = fabsf(xb[i]);
            if (ax > amax) {
                amax = ax;
                max = xb[i];
            }
        }

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

        float inv_scale = -127.0f / max;
        x_d[sb] = 1.0f / inv_scale;
        __m256 vinv = _mm256_set1_ps(inv_scale);

        // Quantize 256 elements in 32-element chunks
        int16_t *bsums = x_bsums + sb * 16;
        for (int g = 0; g < 8; g++) {
            const float *gx = xb + g * 32;
            int8_t *gq = qb + g * 32;

            __m256i i0 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(gx), vinv));
            __m256i i1 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(gx + 8), vinv));
            __m256i i2 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(gx + 16), vinv));
            __m256i i3 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(gx + 24), vinv));
            __m256i s01 = _mm256_packs_epi32(i0, i1);
            __m256i s23 = _mm256_packs_epi32(i2, i3);
            __m256i packed = _mm256_packs_epi16(s01, s23);
            packed = _mm256_permutevar8x32_epi32(packed, perm);
            _mm256_storeu_si256((__m256i *)gq, packed);

            // Compute bsums: sum of 16-element halves
            __m128i lo = _mm256_castsi256_si128(packed);
            __m128i hi = _mm256_extracti128_si256(packed, 1);
            __m128i lo_sad = _mm_sad_epu8(_mm_xor_si128(lo, bsum_bias), bsum_zero);
            int32_t bsum0 = (int32_t)_mm_cvtsi128_si32(lo_sad)
                          + (int32_t)_mm_extract_epi16(lo_sad, 4)
                          - 16 * 128;
            bsums[g * 2] = (int16_t)bsum0;

            __m128i hi_sad = _mm_sad_epu8(_mm_xor_si128(hi, bsum_bias), bsum_zero);
            int32_t bsum1 = (int32_t)_mm_cvtsi128_si32(hi_sad)
                          + (int32_t)_mm_extract_epi16(hi_sad, 4)
                          - 16 * 128;
            bsums[g * 2 + 1] = (int16_t)bsum1;
        }
    }
}

void bn_quant_q8k_avx2_pack_x4(BnBlockQ8Kx4 *out, const float *x,
                               int n_tokens, int cols) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const int nb = cols / BN_QK_K;

    assert(n_tokens % 4 == 0);
    assert(cols % BN_QK_K == 0);
    for (int panel = 0; panel < n_tokens / 4; panel++) {
        for (int b = 0; b < nb; b++) {
            BnBlockQ8Kx4 *dst = out + (size_t)panel * nb + b;
            int8_t q[4][BN_QK_K];

            for (int row = 0; row < 4; row++) {
                const float *src = x + (size_t)(panel * 4 + row) * cols +
                                   b * BN_QK_K;
                __m256 values[32];
                __m256 max_abs = _mm256_setzero_ps();
                __m256 signed_max_mask = _mm256_setzero_ps();

                for (int j = 0; j < 32; j++) {
                    __m256 v = _mm256_loadu_ps(src + j * 8);
                    __m256 old_max = max_abs;
                    values[j] = v;
                    max_abs = _mm256_max_ps(max_abs,
                                             _mm256_andnot_ps(sign_bit, v));
                    signed_max_mask = _mm256_and_ps(
                        signed_max_mask,
                        _mm256_cmp_ps(old_max, max_abs, _CMP_EQ_OQ));
                    signed_max_mask = _mm256_or_ps(
                        signed_max_mask,
                        _mm256_cmp_ps(max_abs, v, _CMP_EQ_OQ));
                }

                __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1),
                                         _mm256_castps256_ps128(max_abs));
                max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
                max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
                float max_scalar = _mm_cvtss_f32(max4);
                __m256 final_mask = _mm256_and_ps(
                    signed_max_mask,
                    _mm256_cmp_ps(_mm256_set1_ps(max_scalar), max_abs,
                                  _CMP_EQ_OQ));
                float iscale = max_scalar != 0.0f ? 127.0f / max_scalar : 0.0f;
                if (_mm256_movemask_ps(final_mask))
                    iscale = max_scalar != 0.0f ? -127.0f / max_scalar : 0.0f;
                dst->d[row] = max_scalar != 0.0f ? 1.0f / iscale : 0.0f;

                __m256 scale = _mm256_set1_ps(iscale);
                for (int j = 0; j < 32; j += 4) {
                    __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(
                        _mm256_mul_ps(values[j], scale), _MM_ROUND_NEAREST));
                    __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(
                        _mm256_mul_ps(values[j + 1], scale), _MM_ROUND_NEAREST));
                    __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(
                        _mm256_mul_ps(values[j + 2], scale), _MM_ROUND_NEAREST));
                    __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(
                        _mm256_mul_ps(values[j + 3], scale), _MM_ROUND_NEAREST));
                    __m256i packed = _mm256_packs_epi16(
                        _mm256_packs_epi32(i0, i1),
                        _mm256_packs_epi32(i2, i3));
                    packed = _mm256_permutevar8x32_epi32(packed, perm);
                    _mm256_storeu_si256((__m256i *)(q[row] + j * 8), packed);
                }
            }

            memset(dst->bsums, 0, sizeof(dst->bsums));
            for (int j = 0; j < BN_QK_K * 4; j++) {
                int src_offset = (j / 32) * 8 + j % 8;
                int src_id = (j % 32) / 8;
                int index = (((j & 31) >> 3) << 2) +
                            ((j >> 8) << 4) + ((j >> 6) & 3);
                dst->qs[j] = q[src_id][src_offset];
                dst->bsums[index] += dst->qs[j];
            }
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

        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        x_scales[b] = bn_fp16_to_fp32(bn_fp32_to_fp16(scale));

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
