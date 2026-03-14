#include "quant_internal.h"
#include <arm_neon.h>

#define TQ1_DECODE(qx) vreinterpretq_s8_u8( \
    vshrq_n_u8(vhaddq_u8((qx), vshrq_n_u8((qx), 1)), 6))

void bn_quant_tq1_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnTQ1SdotCtx *c = (BnTQ1SdotCtx *)ctx;
    const BnBlockTQ1 *blocks = (const BnBlockTQ1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float combined_scale = c->combined_scale;
    const int8_t *x_q = c->x_q;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const int8_t *xb = x_q + b * BN_QK_K;

            int32x4_t bacc = vdupq_n_s32(0);
            int16x8_t xsum_acc = vdupq_n_s16(0);

            {
                uint8x16_t r0 = vld1q_u8(blk->qs);
                uint8x16_t r1 = vld1q_u8(blk->qs + 16);
                uint8x16_t m0_1 = vmulq_u8(r0, vdupq_n_u8(3));
                uint8x16_t m0_2 = vmulq_u8(r0, vdupq_n_u8(9));
                uint8x16_t m0_3 = vmulq_u8(r0, vdupq_n_u8(27));
                uint8x16_t m0_4 = vmulq_u8(r0, vdupq_n_u8(81));
                uint8x16_t m1_1 = vmulq_u8(r1, vdupq_n_u8(3));
                uint8x16_t m1_2 = vmulq_u8(r1, vdupq_n_u8(9));
                uint8x16_t m1_3 = vmulq_u8(r1, vdupq_n_u8(27));
                uint8x16_t m1_4 = vmulq_u8(r1, vdupq_n_u8(81));

                #define S1_TRIT(mx0, mx1, off) do { \
                    int8x16_t y0 = vld1q_s8(xb + (off)); \
                    int8x16_t y1 = vld1q_s8(xb + (off) + 16); \
                    xsum_acc = vpadalq_s8(xsum_acc, y0); \
                    xsum_acc = vpadalq_s8(xsum_acc, y1); \
                    bacc = vdotq_s32(bacc, TQ1_DECODE(mx0), y0); \
                    bacc = vdotq_s32(bacc, TQ1_DECODE(mx1), y1); \
                } while(0)
                S1_TRIT(r0,   r1,   0);
                S1_TRIT(m0_1, m1_1, 32);
                S1_TRIT(m0_2, m1_2, 64);
                S1_TRIT(m0_3, m1_3, 96);
                S1_TRIT(m0_4, m1_4, 128);
                #undef S1_TRIT
            }

            {
                uint8x16_t r = vld1q_u8(blk->qs + 32);
                uint8x16_t m_1 = vmulq_u8(r, vdupq_n_u8(3));
                uint8x16_t m_2 = vmulq_u8(r, vdupq_n_u8(9));
                uint8x16_t m_3 = vmulq_u8(r, vdupq_n_u8(27));
                uint8x16_t m_4 = vmulq_u8(r, vdupq_n_u8(81));

                #define S2_TRIT(mx, off) do { \
                    int8x16_t y = vld1q_s8(xb + (off)); \
                    xsum_acc = vpadalq_s8(xsum_acc, y); \
                    bacc = vdotq_s32(bacc, TQ1_DECODE(mx), y); \
                } while(0)
                S2_TRIT(r,   160);
                S2_TRIT(m_1, 176);
                S2_TRIT(m_2, 192);
                S2_TRIT(m_3, 208);
                S2_TRIT(m_4, 224);
                #undef S2_TRIT
            }

            int32_t qh_dot = 0, qh_xsum = 0;
            static const uint8_t pow3s[] = {1, 3, 9, 27};
            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3s[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    int8_t xv = xb[240 + n*4 + m];
                    qh_dot += xi * xv;
                    qh_xsum += xv;
                }
            }

            int32_t x_sum = vaddlvq_s16(xsum_acc) + qh_xsum;
            row_sum += d * (float)(vaddvq_s32(bacc) + qh_dot - x_sum);
        }
        c->out[row] = row_sum * combined_scale;
    }
}

#undef TQ1_DECODE
