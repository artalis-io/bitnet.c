#include "quant_ctx.h"
#include "kquant_helpers.h"

void bn_quant_q3k_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ3KCtx *c = (BnQ3KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ3K *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);

            uint8_t scales[16];
            bn_q3k_unpack_scales(blk->scales, scales);

            const uint8_t *q  = blk->qs;
            const uint8_t *hm = blk->hmask;
            const float *xb = x + b * BN_QK_K;

            int is = 0;
            uint8_t m = 1;
            int out_idx = 0;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    float dl = d * ((int)scales[is++] - 32);
                    for (int l = 0; l < 16; l++) {
                        int q3 = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                        row_sum += dl * q3 * xb[out_idx++];
                    }
                    dl = d * ((int)scales[is++] - 32);
                    for (int l = 0; l < 16; l++) {
                        int q3 = ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                        row_sum += dl * q3 * xb[out_idx++];
                    }
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
        c->out[row] = row_sum;
    }
}
