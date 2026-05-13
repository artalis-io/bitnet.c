#include "quant_ctx.h"
#include "quant_neon_helpers.h"

void bn_quant_i2s_neon_range(void *ctx, int row_start, int row_end) {
    BnI2SFloatCtx *c = (BnI2SFloatCtx *)ctx;
    int cols = c->W->cols;
    int row_bytes = cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float scale = c->W->scale;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;
        float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
        float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
        float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
        float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
        const int8x16_t one = vdupq_n_s8(1);
        const uint8x16_t mask3 = vdupq_n_u8(3);
        const uint8_t *rd_end = rd + row_bytes;
        while (done < cols) {
            if (rd + 192 < rd_end) {
                __builtin_prefetch(rd + 128, 0, 0);
                __builtin_prefetch(rd + 192, 0, 0);
            }
            for (int h = 0; h < 2; h++) {
                uint8x16_t raw = vld1q_u8(rd + h * 16);
                const float *xp = x + done + h * 16;
                int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 6)), one);
                int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one);
                int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one);
                int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one);
                bn_neon_acc_i8x16_f32(t0, xp + 0*32, &accA0, &accA1, &accA2, &accA3);
                bn_neon_acc_i8x16_f32(t1, xp + 1*32, &accB0, &accB1, &accB2, &accB3);
                bn_neon_acc_i8x16_f32(t2, xp + 2*32, &accA0, &accA1, &accA2, &accA3);
                bn_neon_acc_i8x16_f32(t3, xp + 3*32, &accB0, &accB1, &accB2, &accB3);
            }
            rd += 32;
            done += 128;
        }
        c->out[row] = (bn_neon_reduce4(accA0, accA1, accA2, accA3) +
                    bn_neon_reduce4(accB0, accB1, accB2, accB3)) * scale;
    }
}
