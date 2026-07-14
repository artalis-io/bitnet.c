#include "quant_ctx.h"
#include <arm_neon.h>
#include <string.h>

static float mxfp4_scale_half_neon(uint8_t e) {
    uint32_t bits = e < 2 ? 0x00200000u << e : (uint32_t)(e - 1) << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

void bn_quant_mxfp4_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockMXFP4 *blocks = (const BnBlockMXFP4 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8x16_t values = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockMXFP4 *blk =
                &blocks[(size_t)row * n_blocks_per_row + b];
            uint8x16_t packed = vld1q_u8(blk->qs);
            int8x16_t lo = vqtbl1q_s8(values, vandq_u8(packed, mask));
            int8x16_t hi = vqtbl1q_s8(values, vshrq_n_u8(packed, 4));
            const int8_t *xb = c->x_q + b * 32;
            int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
            row_sum += mxfp4_scale_half_neon(blk->e) * c->x_scales[b] *
                       (float)vaddvq_s32(acc);
        }
        c->out[row] = row_sum;
    }
}
