#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q8k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ8KCtx *c = (BnQ8KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ8K *blocks = (const BnBlockQ8K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = blk->d;
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < BN_QK_K; i += 16) {
                __m128i w = _mm_loadu_si128((const __m128i *)(blk->qs + i));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w)), _mm256_loadu_ps(xb + i), acc);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w, 8))), _mm256_loadu_ps(xb + i + 8), acc);
            }
            row_sum += bn_avx2_hsum_ps(acc) * d;
        }
        c->out[row] = row_sum;
    }
}
