#include "quant_internal.h"

void bn_quant_q2k_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ2KCtx *c = (BnQ2KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ2K *blocks = (const BnBlockQ2K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ2K *blk = &blocks[row * n_blocks_per_row + b];
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *q = blk->qs;
            const float *xb = x + b * BN_QK_K;

            int is = 0, out_idx = 0;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    uint8_t sc = blk->scales[is++];
                    float dl = d * (sc & 0xF);
                    float ml = dmin * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        row_sum += (dl * ((q[l] >> shift) & 3) - ml) * xb[out_idx++];
                    sc = blk->scales[is++];
                    dl = d * (sc & 0xF);
                    ml = dmin * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        row_sum += (dl * ((q[l + 16] >> shift) & 3) - ml) * xb[out_idx++];
                    shift += 2;
                }
                q += 32;
            }
        }
        c->out[row] = row_sum;
    }
}
