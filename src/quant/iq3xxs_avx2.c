#include "quant_internal.h"
#include "simd_helpers.h"
#include "iq_tables.h"
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
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(wf, vdb), xf));
                }

                qs += 8;
                xb += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}
