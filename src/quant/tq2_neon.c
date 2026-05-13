#include "quant_ctx.h"
#include "quant_neon_helpers.h"

void bn_quant_tq2_neon_range(void *ctx, int row_start, int row_end) {
    BnTQ2Ctx *c = (BnTQ2Ctx *)ctx;
    const BnBlockTQ2 *blocks = (const BnBlockTQ2 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float tensor_scale = c->W->scale;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ2 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;
            float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
            float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
            float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
            float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
            const uint8x16_t mask3 = vdupq_n_u8(3);
            const int8x16_t one_s8 = vdupq_n_s8(1);
            for (int half = 0; half < 2; half++) {
                const uint8_t *qs = blk->qs + half * 32;
                const float *xh = xb + half * 128;
                for (int i = 0; i < 2; i++) {
                    uint8x16_t raw = vld1q_u8(qs + i * 16);
                    const float *xp = xh + i * 16;
                    int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one_s8);
                    int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one_s8);
                    int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one_s8);
                    int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 6), mask3)), one_s8);
                    bn_neon_acc_i8x16_f32(t0, xp + 0*32, &accA0, &accA1, &accA2, &accA3);
                    bn_neon_acc_i8x16_f32(t1, xp + 1*32, &accB0, &accB1, &accB2, &accB3);
                    bn_neon_acc_i8x16_f32(t2, xp + 2*32, &accA0, &accA1, &accA2, &accA3);
                    bn_neon_acc_i8x16_f32(t3, xp + 3*32, &accB0, &accB1, &accB2, &accB3);
                }
            }
            row_sum += (bn_neon_reduce4(accA0, accA1, accA2, accA3) +
                        bn_neon_reduce4(accB0, accB1, accB2, accB3)) * d;
        }
        c->out[row] = row_sum * tensor_scale;
    }
}
