#include "quant_ctx.h"
#include "kquant_helpers.h"

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
                int group = j / 64;  // 0..3
                int bit_lo = group * 2;      // bits 0,2,4,6
                int bit_hi = group * 2 + 1;  // bits 1,3,5,7
                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float ds = d * sc;
                float dm = dmin * m;
                for (int l = 0; l < 32; l++) {
                    int q5 = (qs[l] & 0xF) | (((qh[l] >> bit_lo) & 1) << 4);
                    row_sum += (ds * q5 - dm) * xb[j + l];
                }
                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                ds = d * sc;
                dm = dmin * m;
                for (int l = 0; l < 32; l++) {
                    int q5 = (qs[l] >> 4) | (((qh[l] >> bit_hi) & 1) << 4);
                    row_sum += (ds * q5 - dm) * xb[j + l + 32];
                }
                qs += 32;
            }
        }
        c->out[row] = row_sum;
    }
}

#define Q5K_SCALAR_TILE_T 4

void bn_quant_q5k_scalar_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_blocks_per_row = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q5K_SCALAR_TILE_T) {
            int tile_n = t0 + Q5K_SCALAR_TILE_T <= n_tokens
                ? Q5K_SCALAR_TILE_T : n_tokens - t0;
            float sums[Q5K_SCALAR_TILE_T] = {0};

            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ5K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                float d = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);

                for (int ti = 0; ti < tile_n; ti++) {
                    const uint8_t *qs = blk->qs;
                    const uint8_t *qh = blk->qh;
                    const float *xb = c->x + (size_t)(t0 + ti) * cols + b * BN_QK_K;

                    for (int j = 0; j < BN_QK_K; j += 64) {
                        uint8_t sc, m;
                        int sub = j / 32;
                        int group = j / 64;
                        int bit_lo = group * 2;
                        int bit_hi = group * 2 + 1;
                        bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                        float ds = d * sc;
                        float dm = dmin * m;
                        for (int l = 0; l < 32; l++) {
                            int q5 = (qs[l] & 0xF) |
                                     (((qh[l] >> bit_lo) & 1) << 4);
                            sums[ti] += (ds * q5 - dm) * xb[j + l];
                        }
                        bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                        ds = d * sc;
                        dm = dmin * m;
                        for (int l = 0; l < 32; l++) {
                            int q5 = (qs[l] >> 4) |
                                     (((qh[l] >> bit_hi) & 1) << 4);
                            sums[ti] += (ds * q5 - dm) * xb[j + l + 32];
                        }
                        qs += 32;
                    }
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += sums[ti];
        }
    }
}
