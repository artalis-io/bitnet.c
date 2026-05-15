#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

void bn_quant_q4k_avx2_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo = _mm256_set1_epi8(0xF);

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        __m256 row_acc = _mm256_setzero_ps();
        float row_corr = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            float dx   = x_d[b];
            const uint8_t *qs = blk->qs;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

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
            __m128i bs_lo = _mm256_castsi256_si128(q8sums);
            __m128i bs_hi = _mm256_extracti128_si256(q8sums, 1);
            __m128i bs_paired = _mm_hadd_epi16(bs_lo, bs_hi);
            __m128i mins_16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mins));
            __m128i corr128 = _mm_madd_epi16(mins_16, bs_paired);
            __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
            __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
            int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

            __m256i sumi_v = _mm256_setzero_si256();
            for (int p = 0; p < 4; p++) {
                __m256i raw = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
                __m256i lo = _mm256_and_si256(raw, mask_lo);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo);
                __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb + p * 64));
                __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + p * 64 + 32));
                __m256i plo = _mm256_maddubs_epi16(lo, xv0);
                __m256i phi = _mm256_maddubs_epi16(hi, xv1);
                plo = _mm256_madd_epi16(_mm256_set1_epi16((int16_t)sc[2 * p]), plo);
                phi = _mm256_madd_epi16(_mm256_set1_epi16((int16_t)sc[2 * p + 1]), phi);
                sumi_v = _mm256_add_epi32(sumi_v, _mm256_add_epi32(plo, phi));
            }

            row_acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi_v),
                                       _mm256_set1_ps(dx * d), row_acc);
            row_corr += dx * dmin * (float)bsum_corr;
        }
        c->out[row] = bn_avx2_hsum_ps(row_acc) - row_corr;
    }
}

// Reordered matmul: row → token_tile → block → token_in_tile.
// Accumulates all blocks into local acc[] per token tile before
// writing to the scattered output, reducing cache misses.
#define Q4K_TILE_T 8

void bn_quant_q4k_avx2_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;

    const __m256i mask_lo = _mm256_set1_epi8(0xF);

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q4K_TILE_T) {
            int tile_n = t0 + Q4K_TILE_T <= n_tokens ? Q4K_TILE_T : n_tokens - t0;
            float acc[Q4K_TILE_T] = {0};

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr + b];
                float d    = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qs = blk->qs;

                uint32_t utmp[3];
                memcpy(utmp, blk->scales, 12);
                uint32_t m_lo_w = utmp[1] & kmask1;
                uint32_t m_hi_w = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[0] &= kmask1;
                const uint8_t *sc = (const uint8_t *)utmp;
                uint8_t mins[8];
                memcpy(mins, &m_lo_w, 4);
                memcpy(mins + 4, &m_hi_w, 4);

                __m256i w_lo[4], w_hi[4];
                {
                    const uint8_t *qp = qs;
                    for (int p = 0; p < 4; p++) {
                        __m256i raw = _mm256_loadu_si256((const __m256i *)qp);
                        w_lo[p] = _mm256_and_si256(raw, mask_lo);
                        w_hi[p] = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo);
                        qp += 32;
                    }
                }

                __m256i sc_v[8];
                for (int p = 0; p < 8; p++)
                    sc_v[p] = _mm256_set1_epi16((int16_t)sc[p]);

                __m128i mins_16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mins));

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                    float dx = c->x_d[(size_t)t * n_bpr + b];
                    const int16_t *bsums = c->x_bsums + ((size_t)t * n_bpr + b) * 16;

                    __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
                    __m128i bs_lo = _mm256_castsi256_si128(q8sums);
                    __m128i bs_hi = _mm256_extracti128_si256(q8sums, 1);
                    __m128i bs_paired = _mm_hadd_epi16(bs_lo, bs_hi);
                    __m128i corr128 = _mm_madd_epi16(mins_16, bs_paired);
                    __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
                    __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
                    int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

                    __m256i sumi_v = _mm256_setzero_si256();
                    for (int p = 0; p < 4; p++) {
                        __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb + p * 64));
                        __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + p * 64 + 32));
                        __m256i plo = _mm256_maddubs_epi16(w_lo[p], xv0);
                        __m256i phi = _mm256_maddubs_epi16(w_hi[p], xv1);
                        plo = _mm256_madd_epi16(sc_v[2 * p], plo);
                        phi = _mm256_madd_epi16(sc_v[2 * p + 1], phi);
                        sumi_v = _mm256_add_epi32(sumi_v, _mm256_add_epi32(plo, phi));
                    }

                    int32_t sumi = bn_avx2_hsum_epi32(sumi_v);
                    acc[ti] += dx * (d * (float)sumi - dmin * (float)bsum_corr);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += acc[ti];
        }
    }
}
