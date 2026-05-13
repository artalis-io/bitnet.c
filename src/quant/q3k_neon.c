#include "quant_ctx.h"
#include "kquant_helpers.h"
#include <arm_neon.h>

void bn_quant_q3k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ3KCtx *c = (BnQ3KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const float *x = c->x;

    const uint8x16_t mask2 = vdupq_n_u8(3);
    const uint8x16_t bias4 = vdupq_n_u8(4);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ3K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);

            uint8_t scales[16];
            bn_q3k_unpack_scales(blk->scales, scales);

            const uint8_t *q  = blk->qs;
            const uint8_t *hm = blk->hmask;
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    uint8x16_t q0 = vld1q_u8(q);
                    uint8x16_t q1 = vld1q_u8(q + 16);
                    uint8x16_t hm0 = vld1q_u8(hm);
                    uint8x16_t hm1 = vld1q_u8(hm + 16);

                    // Extract low 2 bits: (q >> shift) & 3
                    int8x16_t neg_shift = vdupq_n_s8(-(int8_t)shift);
                    uint8x16_t w0_u = vandq_u8(vshlq_u8(q0, neg_shift), mask2);
                    uint8x16_t w1_u = vandq_u8(vshlq_u8(q1, neg_shift), mask2);

                    // High bit: subtract 4 where hmask bit is NOT set
                    uint8x16_t vm = vdupq_n_u8(m);
                    uint8x16_t hm_set0 = vceqq_u8(vandq_u8(hm0, vm), vdupq_n_u8(0));
                    uint8x16_t hm_set1 = vceqq_u8(vandq_u8(hm1, vm), vdupq_n_u8(0));
                    uint8x16_t sub0 = vandq_u8(hm_set0, bias4);
                    uint8x16_t sub1 = vandq_u8(hm_set1, bias4);
                    int8x16_t w0 = vreinterpretq_s8_u8(vsubq_u8(w0_u, sub0));
                    int8x16_t w1 = vreinterpretq_s8_u8(vsubq_u8(w1_u, sub1));

                    float dl0 = d * ((int)scales[is++] - 32);
                    float32x4_t vdl0 = vdupq_n_f32(dl0);
                    {
                        int16x8_t lo16 = vmovl_s8(vget_low_s8(w0));
                        int16x8_t hi16 = vmovl_s8(vget_high_s8(w0));
                        const float *xp = xb + n + j * 32;
                        acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vdl0), vld1q_f32(xp));
                        acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vdl0), vld1q_f32(xp + 4));
                        acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vdl0), vld1q_f32(xp + 8));
                        acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vdl0), vld1q_f32(xp + 12));
                    }

                    float dl1 = d * ((int)scales[is++] - 32);
                    float32x4_t vdl1 = vdupq_n_f32(dl1);
                    {
                        int16x8_t lo16 = vmovl_s8(vget_low_s8(w1));
                        int16x8_t hi16 = vmovl_s8(vget_high_s8(w1));
                        const float *xp = xb + n + j * 32 + 16;
                        acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vdl1), vld1q_f32(xp));
                        acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vdl1), vld1q_f32(xp + 4));
                        acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vdl1), vld1q_f32(xp + 8));
                        acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vdl1), vld1q_f32(xp + 12));
                    }

                    shift += 2;
                    m <<= 1;
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
