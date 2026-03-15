#include "quant_internal.h"
#include "iq_tables.h"

void bn_quant_iq4xs_scalar_range(void *ctx, int row_start, int row_end) {
    BnIQ4XSCtx *c = (BnIQ4XSCtx *)ctx;
    const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4XS *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;
            const uint8_t *qs = blk->qs;

            for (int j = 0; j < 8; j++) {
                // Extract 6-bit scale: 4 low bits from scales_l, 2 high bits from scales_h
                int lo = (blk->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
                int hi = (blk->scales_h >> (j * 2)) & 3;
                float dl = d * ((lo | (hi << 4)) - 32);

                float sub_sum = 0.0f;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = qs[i];
                    sub_sum += bn_kvalues_iq4nl[byte & 0xF] * xb[i];
                    sub_sum += bn_kvalues_iq4nl[byte >> 4] * xb[i + 16];
                }
                row_sum += sub_sum * dl;
                qs += 16;
                xb += 32;
            }
        }
        c->out[row] = row_sum;
    }
}
