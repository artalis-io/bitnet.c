#include "quant_ctx.h"
#include <arm_neon.h>

void bn_quant_i2s_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnI2SCtx *c = (BnI2SCtx *)ctx;
    int row_bytes = c->W->cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float combined_scale = c->combined_scale;
    int cols = c->W->cols;
    const int8_t *x_q = c->x_q;

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;

        int32x4_t iaccA = vdupq_n_s32(0), iaccB = vdupq_n_s32(0);
        int32x4_t iaccC = vdupq_n_s32(0), iaccD = vdupq_n_s32(0);
        const int8x16_t one = vdupq_n_s8(1);
        const uint8x16_t mask3 = vdupq_n_u8(3);

        const uint8_t *rd_end = rd + row_bytes;
        while (done < cols) {
            if (rd + 64 < rd_end) __builtin_prefetch(rd + 64, 0, 0);
            {
                uint8x16_t raw = vld1q_u8(rd);
                const int8_t *xp = x_q + done;
                int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 6)), one);
                int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one);
                int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one);
                int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one);
                iaccA = vdotq_s32(iaccA, t0, vld1q_s8(xp));
                iaccB = vdotq_s32(iaccB, t1, vld1q_s8(xp + 32));
                iaccC = vdotq_s32(iaccC, t2, vld1q_s8(xp + 64));
                iaccD = vdotq_s32(iaccD, t3, vld1q_s8(xp + 96));
            }
            {
                uint8x16_t raw = vld1q_u8(rd + 16);
                const int8_t *xp = x_q + done + 16;
                int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 6)), one);
                int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one);
                int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one);
                int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one);
                iaccA = vdotq_s32(iaccA, t0, vld1q_s8(xp));
                iaccB = vdotq_s32(iaccB, t1, vld1q_s8(xp + 32));
                iaccC = vdotq_s32(iaccC, t2, vld1q_s8(xp + 64));
                iaccD = vdotq_s32(iaccD, t3, vld1q_s8(xp + 96));
            }
            rd += 32;
            done += 128;
        }

        int32x4_t sum4 = vaddq_s32(vaddq_s32(iaccA, iaccB), vaddq_s32(iaccC, iaccD));
        int32_t total = vaddvq_s32(sum4);
        c->out[row] = (float)total * combined_scale;
    }
}
