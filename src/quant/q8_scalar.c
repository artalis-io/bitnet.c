#include "quant_internal.h"

void bn_quant_q8_scalar_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            const int8_t *xb = x_q + b * 32;
            int32_t sumi = 0;
            for (int i = 0; i < 32; i++)
                sumi += (int32_t)blk->qs[i] * (int32_t)xb[i];
            row_sum += (float)sumi * bn_fp16_to_fp32(blk->d) * x_scales[b];
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q8_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ8Ctx *c = (BnQ8Ctx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            float block_sum = 0.0f;
            for (int i = 0; i < 32; i++) {
                block_sum += blk->qs[i] * xb[i];
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum;
    }
}
