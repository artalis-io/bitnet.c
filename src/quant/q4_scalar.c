#include "quant_ctx.h"
#include <math.h>

void bn_quant_q4_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ4Ctx *c = (BnQ4Ctx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            float block_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                block_sum += ((int)(byte & 0xF) - 8) * xb[i];
                block_sum += ((int)(byte >> 4) - 8) * xb[i + 16];
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q4_scalar_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk =
                &blocks[(size_t)row * n_blocks_per_row + b];
            const int8_t *xb = c->x_q + b * 32;
            int32_t sum = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t q = blk->qs[i];
                sum += ((int32_t)(q & 0x0f) - 8) * (int32_t)xb[i];
                sum += ((int32_t)(q >> 4) - 8) * (int32_t)xb[i + 16];
            }
            row_sum = fmaf((float)sum,
                           bn_fp16_to_fp32(blk->d) * c->x_scales[b],
                           row_sum);
        }
        c->out[row] = row_sum;
    }
}

static int32_t q4_repacked_scalar_dot_row(const uint8_t *qbase,
                                          const int8_t *x_q,
                                          int row_lane) {
    int32_t sum = 0;
    for (int ng = 0; ng < 4; ng++) {
        const uint8_t *qp = qbase + ng * 16 + row_lane * 4;
        const int8_t *xlo = x_q + ng * 4;
        const int8_t *xhi = x_q + 16 + ng * 4;
        for (int j = 0; j < 4; j++) {
            uint8_t q = qp[j] ^ 0x88u;
            sum += ((int32_t)(q & 0x0f) - 8) * (int32_t)xlo[j];
            sum += ((int32_t)(q >> 4) - 8) * (int32_t)xhi[j];
        }
    }
    return sum;
}

void bn_quant_q4_repacked_scalar_sdot_range(void *ctx,
                                            int row_start,
                                            int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnPreparedWeight *prepared = c->prepared;
    if (!prepared || !prepared->qs || !prepared->scales) {
        bn_quant_q4_scalar_sdot_range(ctx, row_start, row_end);
        return;
    }

    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    const uint8_t *rp_qs = prepared->qs;
    const uint16_t *rp_scales = prepared->scales;
    int n_blocks_per_row = c->W->cols / 32;
    int row = row_start;

    for (; row < row_end && (row & 3); row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk =
                &blocks[(size_t)row * n_blocks_per_row + b];
            const int8_t *xb = c->x_q + b * 32;
            int32_t sum = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t q = blk->qs[i];
                sum += ((int32_t)(q & 0x0f) - 8) * (int32_t)xb[i];
                sum += ((int32_t)(q >> 4) - 8) * (int32_t)xb[i + 16];
            }
            row_sum = fmaf((float)sum,
                           bn_fp16_to_fp32(blk->d) * c->x_scales[b],
                           row_sum);
        }
        c->out[row] = row_sum;
    }

    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t gb = (size_t)group * n_blocks_per_row + b;
            const uint8_t *qbase = rp_qs + gb * 64;
            const int8_t *xb = c->x_q + b * 32;
            float dx = c->x_scales[b];
            for (int r = 0; r < 4; r++) {
                int32_t sum = q4_repacked_scalar_dot_row(qbase, xb, r);
                sums[r] = fmaf(
                    (float)sum,
                    bn_fp16_to_fp32(rp_scales[gb * 4 + r]) * dx,
                    sums[r]);
            }
        }
        for (int r = 0; r < 4; r++)
            c->out[row + r] = sums[r];
    }

    for (; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk =
                &blocks[(size_t)row * n_blocks_per_row + b];
            const int8_t *xb = c->x_q + b * 32;
            int32_t sum = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t q = blk->qs[i];
                sum += ((int32_t)(q & 0x0f) - 8) * (int32_t)xb[i];
                sum += ((int32_t)(q >> 4) - 8) * (int32_t)xb[i + 16];
            }
            row_sum = fmaf((float)sum,
                           bn_fp16_to_fp32(blk->d) * c->x_scales[b],
                           row_sum);
        }
        c->out[row] = row_sum;
    }
}
