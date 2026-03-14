#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q8_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ8Ctx *c = (BnQ8Ctx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            __m256i w_raw = _mm256_loadu_si256((const __m256i *)blk->qs);
            __m128i w_lo = _mm256_castsi256_si128(w_raw);
            __m128i w_hi = _mm256_extracti128_si256(w_raw, 1);
            __m256 acc = _mm256_setzero_ps();
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo)), _mm256_loadu_ps(xb)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_lo, 8))), _mm256_loadu_ps(xb + 8)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi)), _mm256_loadu_ps(xb + 16)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_hi, 8))), _mm256_loadu_ps(xb + 24)));
            row_sum += bn_avx2_hsum_ps(acc) * d;
        }
        c->out[row] = row_sum;
    }
}
