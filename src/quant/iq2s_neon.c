#include "quant_internal.h"
#include <arm_neon.h>

void bn_quant_iq2s_neon_range(void *ctx, int row_start, int row_end) {
    BnIQ2SCtx *c = (BnIQ2SCtx *)ctx;
    const BnBlockIQ2S *blocks = (const BnBlockIQ2S *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ2S *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float tmp[BN_QK_K];
            bn_quant_dequant_iq2s(blk, tmp);
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
            for (int i = 0; i < BN_QK_K; i += 16) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(tmp + i),      vld1q_f32(xb + i));
                acc1 = vmlaq_f32(acc1, vld1q_f32(tmp + i + 4),  vld1q_f32(xb + i + 4));
                acc2 = vmlaq_f32(acc2, vld1q_f32(tmp + i + 8),  vld1q_f32(xb + i + 8));
                acc3 = vmlaq_f32(acc3, vld1q_f32(tmp + i + 12), vld1q_f32(xb + i + 12));
            }
            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
