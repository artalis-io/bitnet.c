#include "quant_internal.h"
#include <arm_neon.h>

void bn_quant_q3k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ3KCtx *c = (BnQ3KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const float *x = c->x;

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
                    float dl = d * ((int)scales[is++] - 32);
                    float32x4_t vdl = vdupq_n_f32(dl);
                    {
                        int8_t tmp[16];
                        for (int l = 0; l < 16; l++) {
                            tmp[l] = (int8_t)(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                        }
                        int8x16_t w = vld1q_s8(tmp);
                        int16x8_t lo16 = vmovl_s8(vget_low_s8(w));
                        int16x8_t hi16 = vmovl_s8(vget_high_s8(w));
                        const float *xp = xb + n + j * 32;
                        acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vdl), vld1q_f32(xp));
                        acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vdl), vld1q_f32(xp + 4));
                        acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vdl), vld1q_f32(xp + 8));
                        acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vdl), vld1q_f32(xp + 12));
                    }

                    dl = d * ((int)scales[is++] - 32);
                    vdl = vdupq_n_f32(dl);
                    {
                        int8_t tmp[16];
                        for (int l = 0; l < 16; l++) {
                            tmp[l] = (int8_t)(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                        }
                        int8x16_t w = vld1q_s8(tmp);
                        int16x8_t lo16 = vmovl_s8(vget_low_s8(w));
                        int16x8_t hi16 = vmovl_s8(vget_high_s8(w));
                        const float *xp = xb + n + j * 32 + 16;
                        acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vdl), vld1q_f32(xp));
                        acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vdl), vld1q_f32(xp + 4));
                        acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vdl), vld1q_f32(xp + 8));
                        acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vdl), vld1q_f32(xp + 12));
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
