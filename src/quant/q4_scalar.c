#include "quant_ctx.h"

void bn_quant_q4_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ4Ctx *c = (BnQ4Ctx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            float block_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                block_sum += ((int)(byte & 0xF) - 8) * xb[i];
                block_sum += ((int)(byte >> 4) - 8) * xb[i + 16];
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum;
    }
}
