#include "quant_internal.h"
#include "quant_neon_helpers.h"
#include <arm_neon.h>

void bn_quant_q8k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ8KCtx *c = (BnQ8KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ8K *blocks = (const BnBlockQ8K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = blk->d;
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
            for (int i = 0; i < BN_QK_K; i += 16) {
                int8x16_t w = vld1q_s8(blk->qs + i);
                bn_neon_acc_i8x16_f32(w, xb + i, &acc0, &acc1, &acc2, &acc3);
            }
            row_sum += bn_neon_reduce4(acc0, acc1, acc2, acc3) * d;
        }
        c->out[row] = row_sum;
    }
}
