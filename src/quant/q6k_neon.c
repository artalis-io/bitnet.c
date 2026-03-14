#include "quant_internal.h"
#include <arm_neon.h>

void bn_quant_q6k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ6KCtx *c = (BnQ6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    const uint8x16_t mask_lo4 = vdupq_n_u8(0xF);
    const uint8x16_t mask_2 = vdupq_n_u8(3);
    const int8x16_t bias32 = vdupq_n_s8(32);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int chunk = 0; chunk < 2; chunk++) {
                uint8x16_t ql0 = vld1q_u8(ql);
                uint8x16_t ql1 = vld1q_u8(ql + 16);
                uint8x16_t ql2 = vld1q_u8(ql + 32);
                uint8x16_t ql3 = vld1q_u8(ql + 48);
                uint8x16_t qh0 = vld1q_u8(qh);
                uint8x16_t qh1 = vld1q_u8(qh + 16);

                int8x16_t w0a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql0, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh0, mask_2), 4))), bias32);
                int8x16_t w0b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql1, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh1, mask_2), 4))), bias32);
                int8x16_t w1a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql2, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_2), 4))), bias32);
                int8x16_t w1b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql3, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_2), 4))), bias32);
                int8x16_t w2a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql0, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_2), 4))), bias32);
                int8x16_t w2b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql1, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_2), 4))), bias32);
                int8x16_t w3a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql2, 4),
                    vshlq_n_u8(vshrq_n_u8(qh0, 6), 4))), bias32);
                int8x16_t w3b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql3, 4),
                    vshlq_n_u8(vshrq_n_u8(qh1, 6), 4))), bias32);

                float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
                float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

                #define Q6K_ACC_16(w_vec, xp, scale_val) do { \
                    float ds = d * (float)(scale_val); \
                    float32x4_t vds = vdupq_n_f32(ds); \
                    int16x8_t lo16 = vmovl_s8(vget_low_s8(w_vec)); \
                    int16x8_t hi16 = vmovl_s8(vget_high_s8(w_vec)); \
                    acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vds), vld1q_f32(xp)); \
                    acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vds), vld1q_f32(xp + 4)); \
                    acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vds), vld1q_f32(xp + 8)); \
                    acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vds), vld1q_f32(xp + 12)); \
                } while(0)

                Q6K_ACC_16(w0a, xb +  0, sc[0]);
                Q6K_ACC_16(w0b, xb + 16, sc[1]);
                Q6K_ACC_16(w1a, xb + 32, sc[2]);
                Q6K_ACC_16(w1b, xb + 48, sc[3]);
                Q6K_ACC_16(w2a, xb + 64, sc[4]);
                Q6K_ACC_16(w2b, xb + 80, sc[5]);
                Q6K_ACC_16(w3a, xb + 96, sc[6]);
                Q6K_ACC_16(w3b, xb +112, sc[7]);

                #undef Q6K_ACC_16

                float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
                row_sum += vget_lane_f32(vpadd_f32(r, r), 0);

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}
