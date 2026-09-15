#include "quant_ctx.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <string.h>
#include <immintrin.h>

void bn_quant_iq3xxs_avx2_range(void *ctx, int row_start, int row_end) {
    BnIQ3XXSCtx *c = (BnIQ3XXSCtx *)ctx;
    const BnBlockIQ3XXS *blocks = (const BnBlockIQ3XXS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ3XXS *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *qs = blk->qs;
            const uint8_t *scales_and_signs = qs + BN_QK_K / 4;
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
                uint32_t aux32;
                memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                __m256 vdb = _mm256_set1_ps(db);

                // Scalar decode to float buffer
                float tmp[32];
                for (int l = 0; l < 4; l++) {
                    const uint8_t signs = bn_ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7F];
                    const uint8_t *grid1 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 0]];
                    const uint8_t *grid2 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 1]];

                    for (int j = 0; j < 4; j++) {
                        float w1 = (float)grid1[j];
                        float w2 = (float)grid2[j];
                        if (signs & bn_kmask_iq2xs[j + 0]) w1 = -w1;
                        if (signs & bn_kmask_iq2xs[j + 4]) w2 = -w2;
                        tmp[l * 8 + j + 0] = w1;
                        tmp[l * 8 + j + 4] = w2;
                    }
                }

                // AVX2: multiply decoded weights by scale, then FMA with x
                for (int g = 0; g < 32; g += 8) {
                    __m256 wf = _mm256_loadu_ps(tmp + g);
                    __m256 xf = _mm256_loadu_ps(xb + g);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(wf, vdb), xf, acc);
                }

                qs += 8;
                xb += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_iq3xxs_avx2_q8k_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    const BnBlockIQ3XXS *blocks = (const BnBlockIQ3XXS *)c->W->data;
    int n_bpr = c->W->cols / BN_QK_K;

    for (int row = row_start; row < row_end; row++) {
        __m256 accumf = _mm256_setzero_ps();
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockIQ3XXS *blk =
                &blocks[(size_t)row * n_bpr + b];
            const uint8_t *q3 = blk->qs;
            const uint8_t *gas = blk->qs + BN_QK_K / 4;
            const int8_t *q8 = c->x_q + b * BN_QK_K;
            __m256i sumi1 = _mm256_setzero_si256();
            __m256i sumi2 = _mm256_setzero_si256();

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32 += 2) {
                uint32_t aux[2];
                uint8_t grid[2][32];
                int8_t signs[2][32];
                memcpy(aux, gas + 4 * ib32, sizeof(aux));
                for (int half = 0; half < 2; half++) {
                    for (int l = 0; l < 4; l++) {
                        uint8_t sign_mask = bn_ksigns_iq2xs[
                            (aux[half] >> (7 * l)) & 0x7f];
                        const uint8_t *g0 = (const uint8_t *)
                            &bn_iq3xxs_grid[q3[half * 8 + 2 * l]];
                        const uint8_t *g1 = (const uint8_t *)
                            &bn_iq3xxs_grid[q3[half * 8 + 2 * l + 1]];
                        for (int j = 0; j < 4; j++) {
                            grid[half][l * 8 + j] = g0[j];
                            grid[half][l * 8 + j + 4] = g1[j];
                            signs[half][l * 8 + j] =
                                (sign_mask & bn_kmask_iq2xs[j]) ? -1 : 1;
                            signs[half][l * 8 + j + 4] =
                                (sign_mask & bn_kmask_iq2xs[j + 4]) ? -1 : 1;
                        }
                    }
                }
                __m256i w1 = _mm256_loadu_si256((const __m256i *)grid[0]);
                __m256i w2 = _mm256_loadu_si256((const __m256i *)grid[1]);
                __m256i x1 = _mm256_loadu_si256(
                    (const __m256i *)(q8 + ib32 * 32));
                __m256i x2 = _mm256_loadu_si256(
                    (const __m256i *)(q8 + (ib32 + 1) * 32));
                __m256i s1 = _mm256_loadu_si256((const __m256i *)signs[0]);
                __m256i s2 = _mm256_loadu_si256((const __m256i *)signs[1]);
                __m256i p1 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(w1, _mm256_sign_epi8(x1, s1)),
                    _mm256_set1_epi16((int16_t)(2 * (aux[0] >> 28) + 1)));
                __m256i p2 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(w2, _mm256_sign_epi8(x2, s2)),
                    _mm256_set1_epi16((int16_t)(2 * (aux[1] >> 28) + 1)));
                sumi1 = _mm256_add_epi32(sumi1, p1);
                sumi2 = _mm256_add_epi32(sumi2, p2);
                q3 += 16;
            }
            float d = bn_fp16_to_fp32(blk->d) * c->x_d[b];
            accumf = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
                accumf);
        }
        c->out[row] = 0.25f * bn_avx2_hsum_ps(accumf);
    }
}
