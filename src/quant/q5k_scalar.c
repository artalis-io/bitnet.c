#include "quant_internal.h"

void bn_quant_q5k_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[row * n_blocks_per_row + b];
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const uint8_t *qh = blk->qh;
            const float *xb = x + b * BN_QK_K;

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;
                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float ds = d * sc;
                float dm = dmin * m;
                for (int l = 0; l < 32; l++) {
                    int bit_idx = j + l;
                    int q5 = (qs[l] & 0xF) | (((qh[bit_idx / 8] >> (bit_idx % 8)) & 1) << 4);
                    row_sum += (ds * q5 - dm) * xb[j + l];
                }
                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                ds = d * sc;
                dm = dmin * m;
                for (int l = 0; l < 32; l++) {
                    int bit_idx = j + 32 + l;
                    int q5 = (qs[l] >> 4) | (((qh[bit_idx / 8] >> (bit_idx % 8)) & 1) << 4);
                    row_sum += (ds * q5 - dm) * xb[j + l + 32];
                }
                qs += 32;
            }
        }
        c->out[row] = row_sum;
    }
}
