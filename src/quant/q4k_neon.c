#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "quant_neon_helpers.h"
#include <arm_neon.h>

void bn_quant_q4k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ4KCtx *c = (BnQ4KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const float *x = c->x;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float ds = d * sc;
                float dm = dmin * m;
                uint8x16_t raw0 = vld1q_u8(qs);
                uint8x16_t raw1 = vld1q_u8(qs + 16);
                int8x16_t w0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo));
                int8x16_t w1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo));
                BN_QK_ACC_SCALED_16(w0, xb + j,      ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 16, ds, dm);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                ds = d * sc;
                dm = dmin * m;
                w0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                w1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                BN_QK_ACC_SCALED_16(w0, xb + j + 32, ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 48, ds, dm);

                qs += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
