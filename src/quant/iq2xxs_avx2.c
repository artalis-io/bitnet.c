#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_iq2xxs_avx2_range(void *ctx, int row_start, int row_end) {
    BnIQ2XXSCtx *c = (BnIQ2XXSCtx *)ctx;
    const BnBlockIQ2XXS *blocks = (const BnBlockIQ2XXS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ2XXS *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float tmp[BN_QK_K];
            bn_quant_dequant_iq2xxs(blk, tmp);
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < BN_QK_K; i += 8) {
                __m256 w = _mm256_loadu_ps(tmp + i);
                __m256 xv = _mm256_loadu_ps(xb + i);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w, xv));
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}
