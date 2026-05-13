#include "quant_ctx.h"
#include "quant_neon_helpers.h"
#include "iq_tables.h"

void bn_quant_iq3s_neon_range(void *ctx, int row_start, int row_end) {
    BnIQ3SCtx *c = (BnIQ3SCtx *)ctx;
    const BnBlockIQ3S *blocks = (const BnBlockIQ3S *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ3S *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
                uint8_t sc_byte = blk->scales[ib32 / 2];
                int sc_nib = (sc_byte >> ((ib32 & 1) * 4)) & 0xF;
                float dl = d * (1 + 2 * sc_nib);
                float32x4_t vdl = vdupq_n_f32(dl);

                float tmp[32];
                for (int l = 0; l < 8; l++) {
                    int idx9 = blk->qs[ib32 * 8 + l] | (((blk->qh[ib32] >> l) & 1) << 8);
                    uint32_t grid_val = bn_iq3s_grid[idx9];
                    const uint8_t *grid = (const uint8_t *)&grid_val;
                    int sign_byte_idx = ib32 * 4 + l / 2;
                    int sign_bit_base = (l % 2) * 4;
                    uint8_t sign_byte = blk->signs[sign_byte_idx];

                    for (int k = 0; k < 4; k++) {
                        float w = (float)grid[k];
                        if ((sign_byte >> (sign_bit_base + k)) & 1) w = -w;
                        tmp[l * 4 + k] = w;
                    }
                }

                for (int g = 0; g < 32; g += 16) {
                    float32x4_t w0 = vmulq_f32(vld1q_f32(tmp + g + 0), vdl);
                    float32x4_t w1 = vmulq_f32(vld1q_f32(tmp + g + 4), vdl);
                    float32x4_t w2 = vmulq_f32(vld1q_f32(tmp + g + 8), vdl);
                    float32x4_t w3 = vmulq_f32(vld1q_f32(tmp + g + 12), vdl);
                    acc0 = vmlaq_f32(acc0, w0, vld1q_f32(xb + g + 0));
                    acc1 = vmlaq_f32(acc1, w1, vld1q_f32(xb + g + 4));
                    acc2 = vmlaq_f32(acc2, w2, vld1q_f32(xb + g + 8));
                    acc3 = vmlaq_f32(acc3, w3, vld1q_f32(xb + g + 12));
                }
                xb += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
