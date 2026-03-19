#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q4_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias = _mm_set1_epi8(8);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);
            _mm_prefetch((const char *)(blk + 8), _MM_HINT_T1);
            float d_q4 = bn_fp16_to_fp32(blk->d);
            float d_q8 = x_scales[b];

            __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
            __m128i lo_128 = _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias);
            __m128i hi_128 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias);

            __m256i w256 = _mm256_set_m128i(hi_128, lo_128);
            __m256i xq256 = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));

            __m256i acc = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq256);
            row_sum += d_q4 * d_q8 * (float)bn_avx2_hsum_epi32(acc);
        }
        c->out[row] = row_sum;
    }
}
