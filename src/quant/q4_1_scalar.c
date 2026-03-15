#include "quant_internal.h"

void bn_quant_q4_1_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ4_1Ctx *c = (BnQ4_1Ctx *)ctx;
    const BnBlockQ4_1 *blocks = (const BnBlockQ4_1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_1 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float m = bn_fp16_to_fp32(blk->m);
            const float *xb = x + b * 32;
            float block_sum = 0.0f;
            float x_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                block_sum += (byte & 0xF) * xb[i];
                block_sum += (byte >> 4) * xb[i + 16];
                x_sum += xb[i] + xb[i + 16];
            }
            row_sum += block_sum * d + x_sum * m;
        }
        c->out[row] = row_sum;
    }
}
