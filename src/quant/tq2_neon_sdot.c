#include "quant_ctx.h"
#include <arm_neon.h>

void bn_quant_tq2_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnTQ2SdotCtx *c = (BnTQ2SdotCtx *)ctx;
    const BnBlockTQ2 *blocks = (const BnBlockTQ2 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const int8_t *x_q = c->x_q;

    const uint8x16_t mask3 = vdupq_n_u8(3);
    const int8x16_t one = vdupq_n_s8(1);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ2 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const int8_t *xb = x_q + b * BN_QK_K;

            int32x4_t iaccA = vdupq_n_s32(0), iaccB = vdupq_n_s32(0);
            int32x4_t iaccC = vdupq_n_s32(0), iaccD = vdupq_n_s32(0);

            for (int half = 0; half < 2; half++) {
                const uint8_t *qs = blk->qs + half * 32;
                const int8_t *xh = xb + half * 128;
                for (int i = 0; i < 2; i++) {
                    uint8x16_t raw = vld1q_u8(qs + i * 16);
                    const int8_t *xp = xh + i * 16;
                    int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one);
                    int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one);
                    int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one);
                    int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 6), mask3)), one);
                    iaccA = vdotq_s32(iaccA, t0, vld1q_s8(xp));
                    iaccB = vdotq_s32(iaccB, t1, vld1q_s8(xp + 32));
                    iaccC = vdotq_s32(iaccC, t2, vld1q_s8(xp + 64));
                    iaccD = vdotq_s32(iaccD, t3, vld1q_s8(xp + 96));
                }
            }

            int32x4_t sum4 = vaddq_s32(vaddq_s32(iaccA, iaccB), vaddq_s32(iaccC, iaccD));
            row_sum += (float)vaddvq_s32(sum4) * d;
        }
        c->out[row] = row_sum * c->combined_scale;
    }
}
