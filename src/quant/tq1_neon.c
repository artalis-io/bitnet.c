#include "quant_internal.h"
#include "quant_neon_helpers.h"

void bn_quant_tq1_neon_range(void *ctx, int row_start, int row_end) {
    BnTQ1Ctx *c = (BnTQ1Ctx *)ctx;
    const BnBlockTQ1 *blocks = (const BnBlockTQ1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float tensor_scale = c->W->scale;
    const float *x = c->x;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            float block_sum = 0.0f;
            const float *xb = x + b * BN_QK_K;
            float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
            float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
            float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
            float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
            const int8x16_t one_s8 = vdupq_n_s8(1);
            const uint8x8_t three_u8 = vdup_n_u8(3);
            int acc_flip = 0;

            for (int n = 0; n < 5; n++) {
                uint8x16_t pow3_vec = vdupq_n_u8(pow3[n]);
                for (int i = 0; i < 2; i++) {
                    uint8x16_t raw = vld1q_u8(blk->qs + i * 16);
                    uint8x16_t q = vmulq_u8(raw, pow3_vec);
                    uint8x8_t xi_lo = vshrn_n_u16(vmull_u8(vget_low_u8(q), three_u8), 8);
                    uint8x8_t xi_hi = vshrn_n_u16(vmull_u8(vget_high_u8(q), three_u8), 8);
                    int8x16_t ternary = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(xi_lo, xi_hi)), one_s8);
                    if (acc_flip++ & 1)
                        bn_neon_acc_i8x16_f32(ternary, xb + n*32 + i*16, &accB0, &accB1, &accB2, &accB3);
                    else
                        bn_neon_acc_i8x16_f32(ternary, xb + n*32 + i*16, &accA0, &accA1, &accA2, &accA3);
                }
            }

            for (int n = 0; n < 5; n++) {
                uint8x16_t raw = vld1q_u8(blk->qs + 32);
                uint8x16_t q = vmulq_u8(raw, vdupq_n_u8(pow3[n]));
                uint8x8_t xi_lo = vshrn_n_u16(vmull_u8(vget_low_u8(q), three_u8), 8);
                uint8x8_t xi_hi = vshrn_n_u16(vmull_u8(vget_high_u8(q), three_u8), 8);
                int8x16_t ternary = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(xi_lo, xi_hi)), one_s8);
                if (acc_flip++ & 1)
                    bn_neon_acc_i8x16_f32(ternary, xb + 160 + n*16, &accB0, &accB1, &accB2, &accB3);
                else
                    bn_neon_acc_i8x16_f32(ternary, xb + 160 + n*16, &accA0, &accA1, &accA2, &accA3);
            }

            block_sum = bn_neon_reduce4(accA0, accA1, accA2, accA3) +
                        bn_neon_reduce4(accB0, accB1, accB2, accB3);

            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[240 + n*4 + m];
                }
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum * tensor_scale;
    }
}
