#include "quant_ctx.h"

void bn_quant_tq2_scalar_range(void *ctx, int row_start, int row_end) {
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
            float block_sum = 0.0f;
            for (int half = 0; half < 2; half++) {
                const uint8_t *qs = blk->qs + half * 32;
                const float *xh = xb + half * 128;
                for (int m = 0; m < 32; m++) {
                    uint8_t byte = qs[m];
                    int8_t q0 = (int8_t)((byte >> 0) & 3) - 1;
                    int8_t q1 = (int8_t)((byte >> 2) & 3) - 1;
                    int8_t q2 = (int8_t)((byte >> 4) & 3) - 1;
                    int8_t q3 = (int8_t)((byte >> 6) & 3) - 1;
                    block_sum += q0 * xh[0*32 + m];
                    block_sum += q1 * xh[1*32 + m];
                    block_sum += q2 * xh[2*32 + m];
                    block_sum += q3 * xh[3*32 + m];
                }
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum * tensor_scale;
    }
}
