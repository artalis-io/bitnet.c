#include "quant_internal.h"
#include "quant_neon_helpers.h"

void bn_quant_q4_1_neon_range(void *ctx, int row_start, int row_end) {
    BnQ4_1Ctx *c = (BnQ4_1Ctx *)ctx;
    const BnBlockQ4_1 *blocks = (const BnBlockQ4_1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_1 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            float m = bn_fp16_to_fp32(blk->m);
            const float *xb = x + b * 32;

            uint8x16_t raw = vld1q_u8(blk->qs);
            uint8x16_t lo_u8 = vandq_u8(raw, vdupq_n_u8(0xF));
            uint8x16_t hi_u8 = vshrq_n_u8(raw, 4);

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            // Widen unsigned u8 -> u16 -> s32 -> f32 and FMA with x
            int16x8_t lo16_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo_u8)));
            int16x8_t lo16_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo_u8)));
            acc0 = vmlaq_f32(acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))), vld1q_f32(xb));
            acc1 = vmlaq_f32(acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0))), vld1q_f32(xb + 4));
            acc2 = vmlaq_f32(acc2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))), vld1q_f32(xb + 8));
            acc3 = vmlaq_f32(acc3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1))), vld1q_f32(xb + 12));

            int16x8_t hi16_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi_u8)));
            int16x8_t hi16_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi_u8)));
            acc0 = vmlaq_f32(acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))), vld1q_f32(xb + 16));
            acc1 = vmlaq_f32(acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_0))), vld1q_f32(xb + 20));
            acc2 = vmlaq_f32(acc2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))), vld1q_f32(xb + 24));
            acc3 = vmlaq_f32(acc3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_1))), vld1q_f32(xb + 28));

            float dot = bn_neon_reduce4(acc0, acc1, acc2, acc3);

            // Sum of x for min term
            float32x4_t xsum = vdupq_n_f32(0);
            for (int i = 0; i < 32; i += 4)
                xsum = vaddq_f32(xsum, vld1q_f32(xb + i));
            float32x2_t r = vadd_f32(vget_low_f32(xsum), vget_high_f32(xsum));
            float xs = vget_lane_f32(vpadd_f32(r, r), 0);

            row_sum += dot * d + xs * m;
        }
        c->out[row] = row_sum;
    }
}
