#include "quant_ctx.h"
#include <arm_neon.h>

void bn_quant_q2k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ2KCtx *c = (BnQ2KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ2K *blocks = (const BnBlockQ2K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ2K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *q = blk->qs;
            const float *xb = x + b * BN_QK_K;

            const uint8x16_t mask2 = vdupq_n_u8(3);
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            int is = 0, out_idx = 0;
            for (int n = 0; n < BN_QK_K; n += 128) {
                uint8x16_t q0_vec = vld1q_u8(q);
                uint8x16_t q1_vec = vld1q_u8(q + 16);
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int8x16_t neg_shift = vdupq_n_s8(-(int8_t)shift);
                    {
                        uint8_t sc = blk->scales[is++];
                        float32x4_t vds = vdupq_n_f32(d * (sc & 0xF));
                        float32x4_t vdm = vdupq_n_f32(dmin * (sc >> 4));
                        int8x16_t w = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q0_vec, neg_shift), mask2));
                        int16x8_t lo16 = vmovl_s8(vget_low_s8(w));
                        int16x8_t hi16 = vmovl_s8(vget_high_s8(w));
                        const float *xp = xb + out_idx;
                        acc0 = vmlaq_f32(acc0, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vds), vdm), vld1q_f32(xp));
                        acc1 = vmlaq_f32(acc1, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vds), vdm), vld1q_f32(xp + 4));
                        acc2 = vmlaq_f32(acc2, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vds), vdm), vld1q_f32(xp + 8));
                        acc3 = vmlaq_f32(acc3, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vds), vdm), vld1q_f32(xp + 12));
                        out_idx += 16;
                    }
                    {
                        uint8_t sc = blk->scales[is++];
                        float32x4_t vds = vdupq_n_f32(d * (sc & 0xF));
                        float32x4_t vdm = vdupq_n_f32(dmin * (sc >> 4));
                        int8x16_t w = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q1_vec, neg_shift), mask2));
                        int16x8_t lo16 = vmovl_s8(vget_low_s8(w));
                        int16x8_t hi16 = vmovl_s8(vget_high_s8(w));
                        const float *xp = xb + out_idx;
                        acc0 = vmlaq_f32(acc0, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vds), vdm), vld1q_f32(xp));
                        acc1 = vmlaq_f32(acc1, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vds), vdm), vld1q_f32(xp + 4));
                        acc2 = vmlaq_f32(acc2, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vds), vdm), vld1q_f32(xp + 8));
                        acc3 = vmlaq_f32(acc3, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vds), vdm), vld1q_f32(xp + 12));
                        out_idx += 16;
                    }
                    shift += 2;
                }
                q += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
