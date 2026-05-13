#include "quant_ctx.h"
#include "iq_tables.h"

void bn_quant_iq3s_scalar_range(void *ctx, int row_start, int row_end) {
    BnIQ3SCtx *c = (BnIQ3SCtx *)ctx;
    const BnBlockIQ3S *blocks = (const BnBlockIQ3S *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ3S *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
                uint8_t sc_byte = blk->scales[ib32 / 2];
                int sc_nib = (sc_byte >> ((ib32 & 1) * 4)) & 0xF;
                float dl = d * (1 + 2 * sc_nib);

                for (int l = 0; l < 8; l++) {
                    int idx9 = blk->qs[ib32 * 8 + l] | (((blk->qh[ib32] >> l) & 1) << 8);
                    uint32_t grid_val = bn_iq3s_grid[idx9];
                    const uint8_t *grid = (const uint8_t *)&grid_val;
                    int sign_byte_idx = ib32 * 4 + l / 2;
                    int sign_bit_base = (l % 2) * 4;
                    uint8_t sign_byte = blk->signs[sign_byte_idx];

                    for (int k = 0; k < 4; k++) {
                        float w = (float)grid[k];
                        if ((sign_byte >> (sign_bit_base + k)) & 1) w = -w;
                        row_sum += dl * w * *xb++;
                    }
                }
            }
        }
        c->out[row] = row_sum;
    }
}
