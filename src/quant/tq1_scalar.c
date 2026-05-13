#include "quant_ctx.h"

void bn_quant_tq1_scalar_range(void *ctx, int row_start, int row_end) {
    BnTQ1Ctx *c = (BnTQ1Ctx *)ctx;
    const BnBlockTQ1 *blocks = (const BnBlockTQ1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float tensor_scale = c->W->scale;
    const float *x = c->x;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            float block_sum = 0.0f;
            const float *xb = x + b * BN_QK_K;
            for (int n = 0; n < 5; n++) {
                for (int m = 0; m < 32; m++) {
                    uint8_t q = blk->qs[m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[n*32 + m];
                }
            }

            for (int n = 0; n < 5; n++) {
                for (int m = 0; m < 16; m++) {
                    uint8_t q = blk->qs[32 + m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[160 + n*16 + m];
                }
            }

            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[240 + n*4 + m];
                }
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum * tensor_scale;
    }
}
