#include "quant_ctx.h"
#include "quant_neon_helpers.h"
#include "iq_tables.h"
#include <string.h>

void bn_quant_iq3xxs_neon_range(void *ctx, int row_start, int row_end) {
    BnIQ3XXSCtx *c = (BnIQ3XXSCtx *)ctx;
    const BnBlockIQ3XXS *blocks = (const BnBlockIQ3XXS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ3XXS *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *qs = blk->qs;
            const uint8_t *scales_and_signs = qs + BN_QK_K / 4;
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
                uint32_t aux32;
                memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                float32x4_t vdb = vdupq_n_f32(db);

                // Scalar decode to float buffer
                float tmp[32];
                for (int l = 0; l < 4; l++) {
                    const uint8_t signs = bn_ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7F];
                    const uint8_t *grid1 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 0]];
                    const uint8_t *grid2 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 1]];

                    for (int j = 0; j < 4; j++) {
                        float w1 = (float)grid1[j];
                        float w2 = (float)grid2[j];
                        if (signs & bn_kmask_iq2xs[j + 0]) w1 = -w1;
                        if (signs & bn_kmask_iq2xs[j + 4]) w2 = -w2;
                        tmp[l * 8 + j + 0] = w1;
                        tmp[l * 8 + j + 4] = w2;
                    }
                }

                // NEON: multiply decoded weights by scale, then FMA with x
                for (int g = 0; g < 32; g += 16) {
                    float32x4_t w0 = vmulq_f32(vld1q_f32(tmp + g + 0), vdb);
                    float32x4_t w1 = vmulq_f32(vld1q_f32(tmp + g + 4), vdb);
                    float32x4_t w2 = vmulq_f32(vld1q_f32(tmp + g + 8), vdb);
                    float32x4_t w3 = vmulq_f32(vld1q_f32(tmp + g + 12), vdb);
                    acc0 = vmlaq_f32(acc0, w0, vld1q_f32(xb + g + 0));
                    acc1 = vmlaq_f32(acc1, w1, vld1q_f32(xb + g + 4));
                    acc2 = vmlaq_f32(acc2, w2, vld1q_f32(xb + g + 8));
                    acc3 = vmlaq_f32(acc3, w3, vld1q_f32(xb + g + 12));
                }

                qs += 8;
                xb += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
