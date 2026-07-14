#include "quant_ctx.h"
#include <string.h>

static const int8_t mxfp4_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static float mxfp4_scale_half(uint8_t e) {
    uint32_t bits = e < 2 ? 0x00200000u << e : (uint32_t)(e - 1) << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

void bn_quant_mxfp4_scalar_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockMXFP4 *blocks = (const BnBlockMXFP4 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockMXFP4 *blk =
                &blocks[(size_t)row * n_blocks_per_row + b];
            const int8_t *xb = c->x_q + b * 32;
            int32_t sum = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t q = blk->qs[i];
                sum += (int32_t)mxfp4_values[q & 0x0f] * xb[i];
                sum += (int32_t)mxfp4_values[q >> 4] * xb[i + 16];
            }
            row_sum += mxfp4_scale_half(blk->e) * c->x_scales[b] *
                       (float)sum;
        }
        c->out[row] = row_sum;
    }
}
