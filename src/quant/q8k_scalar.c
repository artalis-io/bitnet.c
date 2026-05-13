#include "quant_ctx.h"

void bn_quant_q8k_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ8KCtx *c = (BnQ8KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ8K *blocks = (const BnBlockQ8K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8K *blk = &blocks[row * n_blocks_per_row + b];
            float d = blk->d;
            const float *xb = x + b * BN_QK_K;
            float block_sum = 0.0f;
            for (int i = 0; i < BN_QK_K; i++) {
                block_sum += blk->qs[i] * xb[i];
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum;
    }
}
