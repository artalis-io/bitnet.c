#include "quant_internal.h"
#include <string.h>

void bn_quant_q4k_scalar_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            float dx = x_d[b];
            const uint8_t *qs = blk->qs;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            uint32_t utmp[3];
            memcpy(utmp, blk->scales, 12);
            uint32_t m_lo = utmp[1] & kmask1;
            uint32_t m_hi = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[0] &= kmask1;
            const uint8_t *sc = (const uint8_t *)utmp;
            uint8_t mins[8];
            memcpy(mins, &m_lo, 4);
            memcpy(mins + 4, &m_hi, 4);

            int32_t bsum_corr = 0;
            for (int j = 0; j < 8; j++)
                bsum_corr += (int32_t)mins[j] * ((int32_t)bsums[2 * j] + (int32_t)bsums[2 * j + 1]);

            int32_t sumi = 0;
            for (int p = 0; p < 4; p++) {
                int32_t slo = 0;
                int32_t shi = 0;
                for (int i = 0; i < 32; i++) {
                    uint8_t q = qs[p * 32 + i];
                    slo += (int32_t)(q & 0x0f) * (int32_t)xb[p * 64 + i];
                    shi += (int32_t)(q >> 4) * (int32_t)xb[p * 64 + 32 + i];
                }
                sumi += slo * (int32_t)sc[2 * p] + shi * (int32_t)sc[2 * p + 1];
            }

            row_sum += dx * (d * (float)sumi - dmin * (float)bsum_corr);
        }
        c->out[row] = row_sum;
    }
}

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
