#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

static inline float q4k_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

static inline __m512i q4k_scale_pair_i32(uint8_t lo, uint8_t hi) {
    return _mm512_set_epi32(hi, hi, hi, hi, hi, hi, hi, hi,
                            lo, lo, lo, lo, lo, lo, lo, lo);
}

static inline __m512i q4k_join_256(__m256i lo, __m256i hi) {
    __m512i z = _mm512_castsi256_si512(lo);
    return _mm512_inserti64x4(z, hi, 1);
}

void bn_quant_q4k_avx512_vnni_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo = _mm256_set1_epi8(0x0F);
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        __m512 row_acc[4] = {
            _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps()
        };
        float row_corr[4] = {0};

        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;

            __m512i xv[4];
            const int8_t *xb = x_q + b * BN_QK_K;
            for (int i = 0; i < 4; i++)
                xv[i] = _mm512_loadu_si512((const void *)(xb + i * 64));

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ4K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                float d = q4k_fp16_to_fp32(blk->d);
                float dmin = q4k_fp16_to_fp32(blk->dmin);

                uint32_t utmp[3];
                memcpy(utmp, blk->scales, 12);
                uint32_t m_lo = utmp[1] & kmask1;
                uint32_t m_hi = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[0] &= kmask1;
                const uint8_t *sc = (const uint8_t *)utmp;
                uint8_t mins[8];
                memcpy(mins, &m_lo, 4);
                memcpy(mins + 4, &m_hi, 4);

                __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
                __m128i bsl = _mm256_castsi256_si128(q8sums);
                __m128i bsh = _mm256_extracti128_si256(q8sums, 1);
                __m128i bs_paired = _mm_hadd_epi16(bsl, bsh);
                __m128i mins_v = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mins));
                __m128i corr128 = _mm_madd_epi16(mins_v, bs_paired);
                __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
                __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
                int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

                __m512i sumi_v = _mm512_setzero_si512();
                const uint8_t *qs = blk->qs;
                for (int p = 0; p < 4; p++) {
                    __m256i raw = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
                    __m256i lo = _mm256_and_si256(raw, mask_lo);
                    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4),
                                                  mask_lo);
                    __m512i prod = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), q4k_join_256(lo, hi), xv[p]);
                    prod = _mm512_mullo_epi32(prod, q4k_scale_pair_i32(sc[2 * p], sc[2 * p + 1]));
                    sumi_v = _mm512_add_epi32(sumi_v, prod);
                }

                row_acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sumi_v),
                                             _mm512_set1_ps(dx * d), row_acc[r]);
                row_corr[r] += dx * dmin * (float)bsum_corr;
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = bn_avx512_hsum_ps(row_acc[r]) - row_corr[r];
    }
}

#endif
