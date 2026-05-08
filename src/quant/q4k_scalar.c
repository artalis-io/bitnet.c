#include "quant_internal.h"

void bn_quant_q4k_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ4KCtx *c = (BnQ4KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[row * n_blocks_per_row + b];
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const float *xb = x + b * BN_QK_K;

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;
                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float sum_qx = 0.0f;
                float sum_x = 0.0f;
                for (int l = 0; l < 32; l++) {
                    float xv = xb[j + l];
                    sum_qx += (float)(qs[l] & 0xF) * xv;
                    sum_x += xv;
                }
                row_sum += (d * sc) * sum_qx - (dmin * m) * sum_x;

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                sum_qx = 0.0f;
                sum_x = 0.0f;
                for (int l = 0; l < 32; l++) {
                    float xv = xb[j + l + 32];
                    sum_qx += (float)(qs[l] >> 4) * xv;
                    sum_x += xv;
                }
                row_sum += (d * sc) * sum_qx - (dmin * m) * sum_x;
                qs += 32;
            }
        }
        c->out[row] = row_sum;
    }
}
