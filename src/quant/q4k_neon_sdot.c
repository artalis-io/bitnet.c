#include "quant_internal.h"
#include <arm_neon.h>

void bn_quant_q4k_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4KSdotCtx *c = (BnQ4KSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;
    const float *x_sums = c->x_sums;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t bias8 = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;

            // 8 activation blocks per Q4_K super-block (256 / 32 = 8)
            const int8_t *xb = x_q + (b * BN_QK_K);
            const float *xs = x_scales + (b * 8);
            const float *xsm = x_sums + (b * 8);

            for (int j = 0; j < BN_QK_K; j += 64) {
                int sub = j / 32;

                // Sub-block 0: low nibbles (32 elements)
                uint8_t sc, m;
                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float ds = d * (float)sc;
                float corr = ds * 8.0f - dmin * (float)m;

                uint8x16_t raw0 = vld1q_u8(qs);
                uint8x16_t raw1 = vld1q_u8(qs + 16);

                int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo)), bias8);
                int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo)), bias8);

                int32x4_t s0 = vdotq_s32(zero, q0, vld1q_s8(xb + j));
                int32x4_t s1 = vdotq_s32(zero, q1, vld1q_s8(xb + j + 16));
                int32_t sdot_val = vaddvq_s32(s0) + vaddvq_s32(s1);

                float dx = xs[sub];
                row_sum += dx * (ds * (float)sdot_val + corr * xsm[sub]);

                // Sub-block 1: high nibbles (32 elements)
                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                ds = d * (float)sc;
                corr = ds * 8.0f - dmin * (float)m;

                q0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)), bias8);
                q1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)), bias8);

                s0 = vdotq_s32(zero, q0, vld1q_s8(xb + j + 32));
                s1 = vdotq_s32(zero, q1, vld1q_s8(xb + j + 48));
                sdot_val = vaddvq_s32(s0) + vaddvq_s32(s1);

                dx = xs[sub + 1];
                row_sum += dx * (ds * (float)sdot_val + corr * xsm[sub + 1]);

                qs += 32;
            }
        }
        c->out[row] = row_sum;
    }
}
