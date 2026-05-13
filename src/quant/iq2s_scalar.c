#include "quant_ctx.h"

void bn_quant_iq2s_scalar_range(void *ctx, int row_start, int row_end) {
    BnIQ2SCtx *c = (BnIQ2SCtx *)ctx;
    const BnBlockIQ2S *blocks = (const BnBlockIQ2S *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            float tmp[BN_QK_K];
            bn_quant_dequant_iq2s(&blocks[row * n_blocks_per_row + b], tmp);
            const float *xb = x + b * BN_QK_K;
            for (int i = 0; i < BN_QK_K; i++)
                row_sum += tmp[i] * xb[i];
        }
        c->out[row] = row_sum;
    }
}
