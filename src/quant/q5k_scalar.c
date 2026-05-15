#include "quant_ctx.h"
#include "kquant_helpers.h"
#include <string.h>

void bn_quant_q5k_scalar_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            float dx = x_d[b];
            const uint8_t *qh = blk->qh;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            uint32_t utmp[3];
            memcpy(utmp, blk->scales, 12);
            uint32_t m_lo = utmp[1] & kmask1;
            uint32_t m_hi = ((utmp[2] >> 4) & kmask2) |
                            (((utmp[1] >> 6) & kmask3) << 4);
            utmp[1] = (utmp[2] & kmask2) |
                      (((utmp[0] >> 6) & kmask3) << 4);
            utmp[0] &= kmask1;
            const uint8_t *sc = (const uint8_t *)utmp;
            uint8_t mins[8];
            memcpy(mins, &m_lo, 4);
            memcpy(mins + 4, &m_hi, 4);

            int32_t bsum_corr = 0;
            for (int j = 0; j < 8; j++)
                bsum_corr += (int32_t)mins[j] *
                    ((int32_t)bsums[2 * j] + (int32_t)bsums[2 * j + 1]);

            int32_t sumi = 0;
            const uint8_t *qs = blk->qs;
            for (int p = 0; p < 4; p++) {
                int bit_lo = p * 2;
                int bit_hi = bit_lo + 1;
                int sc_lo = (int)sc[2 * p];
                int sc_hi = (int)sc[2 * p + 1];
                for (int i = 0; i < 32; i++) {
                    int qlo = (int)(qs[i] & 0x0f) |
                        (int)(((qh[i] >> bit_lo) & 1) << 4);
                    int qhi = (int)(qs[i] >> 4) |
                        (int)(((qh[i] >> bit_hi) & 1) << 4);
                    sumi += qlo * sc_lo * (int32_t)xb[p * 64 + i];
                    sumi += qhi * sc_hi * (int32_t)xb[p * 64 + 32 + i];
                }
                qs += 32;
            }

            row_sum += dx * (d * (float)sumi - dmin * (float)bsum_corr);
        }
        c->out[row] = row_sum;
    }
}

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

#define Q5K_SCALAR_SDOT_TILE_T 4

void bn_quant_q5k_scalar_sdot_matmul_range(void *ctx,
                                            int row_start,
                                            int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_blocks_per_row = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q5K_SCALAR_SDOT_TILE_T) {
            int tile_n = t0 + Q5K_SCALAR_SDOT_TILE_T <= n_tokens
                ? Q5K_SCALAR_SDOT_TILE_T : n_tokens - t0;
            float sums[Q5K_SCALAR_SDOT_TILE_T] = {0};

            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ5K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                float d = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qh = blk->qh;

                uint32_t utmp[3];
                memcpy(utmp, blk->scales, 12);
                uint32_t m_lo = utmp[1] & kmask1;
                uint32_t m_hi = ((utmp[2] >> 4) & kmask2) |
                                (((utmp[1] >> 6) & kmask3) << 4);
                utmp[1] = (utmp[2] & kmask2) |
                          (((utmp[0] >> 6) & kmask3) << 4);
                utmp[0] &= kmask1;
                const uint8_t *sc = (const uint8_t *)utmp;
                uint8_t mins[8];
                memcpy(mins, &m_lo, 4);
                memcpy(mins + 4, &m_hi, 4);

                int16_t qw[BN_QK_K];
                const uint8_t *qs = blk->qs;
                for (int p = 0; p < 4; p++) {
                    int bit_lo = p * 2;
                    int bit_hi = bit_lo + 1;
                    int sc_lo = (int)sc[2 * p];
                    int sc_hi = (int)sc[2 * p + 1];
                    for (int i = 0; i < 32; i++) {
                        int qlo = (int)(qs[i] & 0x0f) |
                            (int)(((qh[i] >> bit_lo) & 1) << 4);
                        int qhi = (int)(qs[i] >> 4) |
                            (int)(((qh[i] >> bit_hi) & 1) << 4);
                        qw[p * 64 + i] = (int16_t)(qlo * sc_lo);
                        qw[p * 64 + 32 + i] = (int16_t)(qhi * sc_hi);
                    }
                    qs += 32;
                }

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    float dx = c->x_d[(size_t)t * n_blocks_per_row + b];
                    const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                    const int16_t *bsums =
                        c->x_bsums + ((size_t)t * n_blocks_per_row + b) * 16;
                    int32_t bsum_corr = 0;
                    for (int j = 0; j < 8; j++)
                        bsum_corr += (int32_t)mins[j] *
                            ((int32_t)bsums[2 * j] + (int32_t)bsums[2 * j + 1]);

                    int32_t sumi = 0;
                    for (int i = 0; i < BN_QK_K; i++)
                        sumi += (int32_t)qw[i] * (int32_t)xb[i];

                    sums[ti] += dx * (d * (float)sumi -
                                       dmin * (float)bsum_corr);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += sums[ti];
        }
    }
}
