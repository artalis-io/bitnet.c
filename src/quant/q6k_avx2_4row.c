#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

// 4-row Q6_K SDOT matvec: process 4 output rows at once, unpacking weights
// per-row but loading x_q once. Amortizes activation vector memory read 4x.

void bn_quant_q6k_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo4 = _mm256_set1_epi8(0xF);
    const __m256i mask_hi2 = _mm256_set1_epi8(0x30);
    const __m256i zero = _mm256_setzero_si256();

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        float row_sums[4] = {0};

        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;

            /* Load x_q block ONCE for all rows (2 chunks x 128 bytes = 256 bytes) */
            __m256i xv[8];
            const int8_t *xb = x_q + b * BN_QK_K;
            for (int i = 0; i < 8; i++)
                xv[i] = _mm256_loadu_si256((const __m256i *)(xb + i * 32));

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ6K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                float d = bn_fp16_to_fp32(blk->d);
                const uint8_t *ql = blk->ql;
                const uint8_t *qh = blk->qh;
                const int8_t  *sc = blk->scales;

                int32_t sumi = 0;
                int32_t bias_corr = 0;

                for (int chunk = 0; chunk < 2; chunk++) {
                    __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql);
                    __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                    __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh);

                    /* Unpack 4 unsigned weight vectors (0..63).
                     * Mask after slli_epi16 with 0x30 to prevent 16-bit
                     * cross-byte contamination. */
                    __m256i w0 = _mm256_or_si256(
                        _mm256_and_si256(ql0, mask_lo4),
                        _mm256_and_si256(_mm256_slli_epi16(qh0, 4), mask_hi2));
                    __m256i w1 = _mm256_or_si256(
                        _mm256_and_si256(ql1, mask_lo4),
                        _mm256_and_si256(_mm256_slli_epi16(_mm256_srli_epi16(qh0, 2), 4), mask_hi2));
                    __m256i w2 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4),
                        _mm256_and_si256(_mm256_slli_epi16(_mm256_srli_epi16(qh0, 4), 4), mask_hi2));
                    __m256i w3 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4),
                        _mm256_and_si256(_mm256_slli_epi16(_mm256_srli_epi16(qh0, 6), 4), mask_hi2));

                    int base = chunk * 4;

                    /* DPBUSD all 4 dot products first — no scalar dependencies
                     * between them, so CPU can pipeline all 4 in parallel. */
                    __m256i dot0 = bn_avx2_dpbusd(zero, w0, xv[base + 0]);
                    __m256i dot1 = bn_avx2_dpbusd(zero, w1, xv[base + 1]);
                    __m256i dot2 = bn_avx2_dpbusd(zero, w2, xv[base + 2]);
                    __m256i dot3 = bn_avx2_dpbusd(zero, w3, xv[base + 3]);

                    /* Deferred hsum phase: per-lane sums with separate scales.
                     * Each 256-bit dot has 2 sub-blocks (lo/hi 128-bit lanes). */
                    __m128i lo0 = _mm256_castsi256_si128(dot0), hi0 = _mm256_extracti128_si256(dot0, 1);
                    __m128i lo1 = _mm256_castsi256_si128(dot1), hi1 = _mm256_extracti128_si256(dot1, 1);
                    __m128i lo2 = _mm256_castsi256_si128(dot2), hi2 = _mm256_extracti128_si256(dot2, 1);
                    __m128i lo3 = _mm256_castsi256_si128(dot3), hi3 = _mm256_extracti128_si256(dot3, 1);

                    /* Pairwise hadd to get [sum01_lo, sum01_hi, sum23_lo, sum23_hi] */
                    __m128i p01 = _mm_hadd_epi32(_mm_hadd_epi32(lo0, hi0), _mm_hadd_epi32(lo1, hi1));
                    __m128i p23 = _mm_hadd_epi32(_mm_hadd_epi32(lo2, hi2), _mm_hadd_epi32(lo3, hi3));
                    /* p01 = [sum(lo0), sum(hi0), sum(lo1), sum(hi1)]
                     * p23 = [sum(lo2), sum(hi2), sum(lo3), sum(hi3)] */

                    int32_t s01[4], s23[4];
                    _mm_storeu_si128((__m128i *)s01, p01);
                    _mm_storeu_si128((__m128i *)s23, p23);
                    sumi += s01[0] * (int32_t)sc[0] + s01[1] * (int32_t)sc[1]
                          + s01[2] * (int32_t)sc[2] + s01[3] * (int32_t)sc[3]
                          + s23[0] * (int32_t)sc[4] + s23[1] * (int32_t)sc[5]
                          + s23[2] * (int32_t)sc[6] + s23[3] * (int32_t)sc[7];

                    for (int s = 0; s < 8; s++)
                        bias_corr += (int32_t)sc[s] * (int32_t)bsums[chunk * 8 + s];

                    ql += 64; qh += 32; sc += 8;
                }

                row_sums[r] += d * dx * (float)(sumi - 32 * bias_corr);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sums[r];
    }
}
