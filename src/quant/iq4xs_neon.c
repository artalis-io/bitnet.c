#include "quant_ctx.h"
#include "quant_neon_helpers.h"
#include "iq_tables.h"

void bn_quant_iq4xs_neon_range(void *ctx, int row_start, int row_end) {
    BnIQ4XSCtx *c = (BnIQ4XSCtx *)ctx;
    const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4XS *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;
            const uint8_t *qs = blk->qs;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            for (int j = 0; j < 8; j++) {
                // Extract 6-bit scale
                int lo = (blk->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
                int hi = (blk->scales_h >> (j * 2)) & 3;
                float dl = d * ((lo | (hi << 4)) - 32);
                float32x4_t vdl = vdupq_n_f32(dl);

                // Scalar decode through codebook
                int8_t tmp[32];
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = qs[i];
                    tmp[i]      = bn_kvalues_iq4nl[byte & 0xF];
                    tmp[i + 16] = bn_kvalues_iq4nl[byte >> 4];
                }

                // NEON: widen int8 to float, multiply by scale, FMA with x
                int8x16_t w_lo = vld1q_s8(tmp);
                int16x8_t lo16 = vmovl_s8(vget_low_s8(w_lo));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(w_lo));
                acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vdl), vld1q_f32(xb));
                acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vdl), vld1q_f32(xb + 4));
                acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vdl), vld1q_f32(xb + 8));
                acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vdl), vld1q_f32(xb + 12));

                int8x16_t w_hi = vld1q_s8(tmp + 16);
                lo16 = vmovl_s8(vget_low_s8(w_hi));
                hi16 = vmovl_s8(vget_high_s8(w_hi));
                acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vdl), vld1q_f32(xb + 16));
                acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vdl), vld1q_f32(xb + 20));
                acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vdl), vld1q_f32(xb + 24));
                acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vdl), vld1q_f32(xb + 28));

                qs += 16;
                xb += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
