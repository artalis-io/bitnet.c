#include "quant_ctx.h"
#include <math.h>

void bn_quant_q4_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ4Ctx *c = (BnQ4Ctx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    int row = row_start;
    for (; row + 3 < row_end; row += 4) {
        float row_sum0 = 0.0f;
        float row_sum1 = 0.0f;
        float row_sum2 = 0.0f;
        float row_sum3 = 0.0f;
        const BnBlockQ4_0 *row_blocks0 =
            blocks + (size_t)(row + 0) * n_blocks_per_row;
        const BnBlockQ4_0 *row_blocks1 =
            blocks + (size_t)(row + 1) * n_blocks_per_row;
        const BnBlockQ4_0 *row_blocks2 =
            blocks + (size_t)(row + 2) * n_blocks_per_row;
        const BnBlockQ4_0 *row_blocks3 =
            blocks + (size_t)(row + 3) * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk0 = row_blocks0 + b;
            const BnBlockQ4_0 *blk1 = row_blocks1 + b;
            const BnBlockQ4_0 *blk2 = row_blocks2 + b;
            const BnBlockQ4_0 *blk3 = row_blocks3 + b;
            const float *xb = x + b * 32;
            float block_sum0 = 0.0f;
            float block_sum1 = 0.0f;
            float block_sum2 = 0.0f;
            float block_sum3 = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t q0 = blk0->qs[i];
                uint8_t q1 = blk1->qs[i];
                uint8_t q2 = blk2->qs[i];
                uint8_t q3 = blk3->qs[i];
                float xlo = xb[i];
                float xhi = xb[i + 16];
                block_sum0 += ((int)(q0 & 0xF) - 8) * xlo;
                block_sum0 += ((int)(q0 >> 4) - 8) * xhi;
                block_sum1 += ((int)(q1 & 0xF) - 8) * xlo;
                block_sum1 += ((int)(q1 >> 4) - 8) * xhi;
                block_sum2 += ((int)(q2 & 0xF) - 8) * xlo;
                block_sum2 += ((int)(q2 >> 4) - 8) * xhi;
                block_sum3 += ((int)(q3 & 0xF) - 8) * xlo;
                block_sum3 += ((int)(q3 >> 4) - 8) * xhi;
            }
            row_sum0 += block_sum0 * bn_fp16_to_fp32(blk0->d);
            row_sum1 += block_sum1 * bn_fp16_to_fp32(blk1->d);
            row_sum2 += block_sum2 * bn_fp16_to_fp32(blk2->d);
            row_sum3 += block_sum3 * bn_fp16_to_fp32(blk3->d);
        }
        c->out[row + 0] = row_sum0;
        c->out[row + 1] = row_sum1;
        c->out[row + 2] = row_sum2;
        c->out[row + 3] = row_sum3;
    }

    for (; row < row_end; row++) {
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
        float sums[2][4] = {{0.0f}};
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk =
                &blocks[(size_t)row * n_blocks_per_row + b];
            const int8_t *xb = c->x_q + b * 32;
            int32_t dot[4] = {0, 0, 0, 0};
            for (int i = 0; i < 16; i++) {
                uint8_t q = blk->qs[i];
                int lane = i >> 2;
                dot[lane] +=
                    ((int32_t)(q & 0x0f) - 8) * (int32_t)xb[i];
                dot[lane] +=
                    ((int32_t)(q >> 4) - 8) * (int32_t)xb[i + 16];
            }
            float scale = bn_fp16_to_fp32(blk->d) * c->x_scales[b];
            int stream = b & 1;
            for (int lane = 0; lane < 4; lane++)
                sums[stream][lane] =
                    fmaf((float)dot[lane], scale,
                         sums[stream][lane]);
        }
        float sum0 = (sums[0][0] + sums[0][1]) +
                     (sums[0][2] + sums[0][3]);
        float sum1 = (sums[1][0] + sums[1][1]) +
                     (sums[1][2] + sums[1][3]);
        c->out[row] = sum0 + sum1;
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
