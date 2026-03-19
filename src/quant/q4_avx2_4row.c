#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

// 4-row Q4_0 matvec: process 4 output rows at once, loading x_q once.
// Amortizes activation vector memory read 4x for DDR4 bandwidth optimization.

void bn_quant_q4_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    int rows = c->W->rows;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias = _mm_set1_epi8(8);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        float sums[4] = {0};

        for (int b = 0; b < n_blocks_per_row; b++) {
            // Load x_q block ONCE for all rows
            __m256i xq256 = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));
            float d_q8 = x_scales[b];

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ4_0 *blk = &blocks[(row0 + r) * n_blocks_per_row + b];
                float d_q4 = bn_fp16_to_fp32(blk->d);

                __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
                __m128i lo_128 = _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias);
                __m128i hi_128 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias);
                __m256i w256 = _mm256_set_m128i(hi_128, lo_128);

                __m256i acc = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq256);
                sums[r] += d_q4 * d_q8 * (float)bn_avx2_hsum_epi32(acc);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = sums[r];
    }
}
