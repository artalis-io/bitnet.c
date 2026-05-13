#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q4_1_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ4_1Ctx *c = (BnQ4_1Ctx *)ctx;
    const BnBlockQ4_1 *blocks = (const BnBlockQ4_1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_1 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            float m = bn_fp16_to_fp32(blk->m);
            const float *xb = x + b * 32;

            __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
            __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0xF));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0xF));

            __m256 acc = _mm256_setzero_ps();
            __m256 xacc = _mm256_setzero_ps();

            // Lo nibbles (elements 0-15): unsigned quants
            __m256i i32_0 = _mm256_cvtepu8_epi32(lo);
            __m256i i32_1 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(lo, 8));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_0), _mm256_loadu_ps(xb), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_1), _mm256_loadu_ps(xb + 8), acc);
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb));
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb + 8));

            // Hi nibbles (elements 16-31)
            __m256i i32_2 = _mm256_cvtepu8_epi32(hi);
            __m256i i32_3 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(hi, 8));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_2), _mm256_loadu_ps(xb + 16), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_3), _mm256_loadu_ps(xb + 24), acc);
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb + 16));
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb + 24));

            row_sum += bn_avx2_hsum_ps(acc) * d + bn_avx2_hsum_ps(xacc) * m;
        }
        c->out[row] = row_sum;
    }
}
