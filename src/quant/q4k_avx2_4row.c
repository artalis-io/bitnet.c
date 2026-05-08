#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

static inline float q4k_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

static inline __m256i q4k_scale_all(uint8_t scale) {
    return _mm256_set1_epi16((int16_t)scale);
}

// 4-row Q4_K SDOT matvec: process 4 output rows at once, loading x_q once.
// Amortizes activation vector memory read 4x for DDR bandwidth optimization.
// Deferred hsum: accumulate all DPBUSD results before any horizontal sum
// to avoid scalar dependencies blocking the next DPBUSD.

void bn_quant_q4k_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo = _mm256_set1_epi8(0xF);
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        float row_sums[4] = {0};

#if defined(__GNUC__) || defined(__clang__)
        #pragma GCC unroll 8
#endif
        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;

            /* Load x_q block ONCE for all rows (4 x 64 bytes = 256 bytes) */
            __m256i xv[8];
            const int8_t *xb = x_q + b * BN_QK_K;
            for (int i = 0; i < 4; i++) {
                xv[2*i]     = _mm256_loadu_si256((const __m256i *)(xb + i * 64));
                xv[2*i + 1] = _mm256_loadu_si256((const __m256i *)(xb + i * 64 + 32));
            }

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ4K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                float d    = q4k_fp16_to_fp32(blk->d);
                float dmin = q4k_fp16_to_fp32(blk->dmin);

                /* Decode scales and mins */
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

                /* Min correction via bsums */
                int32_t bsum_corr = 0;
                for (int j = 0; j < 8; j++)
                    bsum_corr += (int32_t)mins[j] * ((int32_t)bsums[2*j] + (int32_t)bsums[2*j + 1]);

                __m256i sumi_v = _mm256_setzero_si256();
                const uint8_t *qs = blk->qs;
                for (int p = 0; p < 4; p++) {
                    __m256i raw = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
                    __m256i lo = _mm256_and_si256(raw, mask_lo);
                    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo);

                    __m256i plo = _mm256_maddubs_epi16(lo, xv[2*p]);
                    __m256i phi = _mm256_maddubs_epi16(hi, xv[2*p + 1]);
                    plo = _mm256_madd_epi16(q4k_scale_all(sc[2*p]), plo);
                    phi = _mm256_madd_epi16(q4k_scale_all(sc[2*p + 1]), phi);
                    sumi_v = _mm256_add_epi32(sumi_v, _mm256_add_epi32(plo, phi));
                }

                int32_t sumi = bn_avx2_hsum_epi32(sumi_v);
                row_sums[r] += dx * (d * (float)sumi - dmin * (float)bsum_corr);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sums[r];
    }
}
