#include "quant_ctx.h"

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

void bn_quant_q5_1_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ5_1Ctx *c = (BnQ5_1Ctx *)ctx;
    const BnBlockQ5_1 *blocks = (const BnBlockQ5_1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5_1 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float m = bn_fp16_to_fp32(blk->m);
            const float *xb = x + b * 32;
            float block_sum = 0.0f;
            float x_sum = 0.0f;
            uint32_t qh = (uint32_t)blk->qh[0] |
                          ((uint32_t)blk->qh[1] << 8) |
                          ((uint32_t)blk->qh[2] << 16) |
                          ((uint32_t)blk->qh[3] << 24);
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                int q0 = (byte & 0xF) | (((qh >> i) & 1u) << 4);
                int q1 = (byte >> 4) | (((qh >> (i + 16)) & 1u) << 4);
                block_sum += q0 * xb[i] + q1 * xb[i + 16];
                x_sum += xb[i] + xb[i + 16];
            }
            row_sum += block_sum * d + x_sum * m;
        }
        c->out[row] = row_sum;
    }
}
