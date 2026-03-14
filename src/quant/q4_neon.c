#include "quant_internal.h"
#include "quant_neon_helpers.h"

void bn_quant_q4_neon_range(void *ctx, int row_start, int row_end) {
    BnQ4Ctx *c = (BnQ4Ctx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            uint8x16_t raw = vld1q_u8(blk->qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0xF))), vdupq_n_s8(8));
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), vdupq_n_s8(8));
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
            bn_neon_acc_i8x16_f32(lo, xb, &acc0, &acc1, &acc2, &acc3);
            bn_neon_acc_i8x16_f32(hi, xb + 16, &acc0, &acc1, &acc2, &acc3);
            row_sum += bn_neon_reduce4(acc0, acc1, acc2, acc3) * d;
        }
        c->out[row] = row_sum;
    }
}
