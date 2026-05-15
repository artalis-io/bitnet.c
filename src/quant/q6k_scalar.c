#include "quant_ctx.h"

void bn_quant_q6k_scalar_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float dx = x_d[b];
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t *sc = blk->scales;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            int32_t sumi = 0;
            int32_t bias_corr = 0;
            for (int chunk = 0; chunk < 2; chunk++) {
                for (int is = 0; is < 2; is++) {
                    int l0 = is * 16;
                    int32_t sum1 = 0;
                    int32_t sum2 = 0;
                    int32_t sum3 = 0;
                    int32_t sum4 = 0;
                    for (int i = 0; i < 16; i++) {
                        int l = l0 + i;
                        uint8_t h = qh[l];
                        int q1 = (int)((ql[l]      & 0x0f) | ((h & 0x03) << 4));
                        int q2 = (int)((ql[l + 32] & 0x0f) | (((h >> 2) & 0x03) << 4));
                        int q3 = (int)((ql[l]      >> 4)   | (((h >> 4) & 0x03) << 4));
                        int q4 = (int)((ql[l + 32] >> 4)   | (((h >> 6) & 0x03) << 4));
                        sum1 += q1 * (int32_t)xb[l];
                        sum2 += q2 * (int32_t)xb[l + 32];
                        sum3 += q3 * (int32_t)xb[l + 64];
                        sum4 += q4 * (int32_t)xb[l + 96];
                    }
                    sumi += (int32_t)sc[is + 0] * sum1 +
                            (int32_t)sc[is + 2] * sum2 +
                            (int32_t)sc[is + 4] * sum3 +
                            (int32_t)sc[is + 6] * sum4;
                }
                for (int g = 0; g < 8; g++)
                    bias_corr += (int32_t)sc[g] * (int32_t)bsums[chunk * 8 + g];

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }

            row_sum += d * dx * (float)(sumi - 32 * bias_corr);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q6k_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ6KCtx *c = (BnQ6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int n = 0; n < BN_QK_K; n += 128) {
                for (int is = 0; is < 2; is++) {
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;
                    float sum3 = 0.0f;
                    float sum4 = 0.0f;
                    int l0 = is * 16;
                    for (int i = 0; i < 16; i++) {
                        int l = l0 + i;
                        int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                        sum1 += (float)q1 * xb[l +  0];
                        sum2 += (float)q2 * xb[l + 32];
                        sum3 += (float)q3 * xb[l + 64];
                        sum4 += (float)q4 * xb[l + 96];
                    }
                    row_sum += d * (
                        (float)sc[is + 0] * sum1 +
                        (float)sc[is + 2] * sum2 +
                        (float)sc[is + 4] * sum3 +
                        (float)sc[is + 6] * sum4);
                }
                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}

#define Q6K_SCALAR_TILE_T 4

void bn_quant_q6k_scalar_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_blocks_per_row = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q6K_SCALAR_TILE_T) {
            int tile_n = t0 + Q6K_SCALAR_TILE_T <= n_tokens
                ? Q6K_SCALAR_TILE_T : n_tokens - t0;
            float sums[Q6K_SCALAR_TILE_T] = {0};

            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ6K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                float d = bn_fp16_to_fp32(blk->d);

                for (int ti = 0; ti < tile_n; ti++) {
                    const uint8_t *ql = blk->ql;
                    const uint8_t *qh = blk->qh;
                    const int8_t *sc = blk->scales;
                    const float *xb = c->x + (size_t)(t0 + ti) * cols + b * BN_QK_K;

                    for (int n = 0; n < BN_QK_K; n += 128) {
                        for (int is = 0; is < 2; is++) {
                            float sum1 = 0.0f;
                            float sum2 = 0.0f;
                            float sum3 = 0.0f;
                            float sum4 = 0.0f;
                            int l0 = is * 16;
                            for (int i = 0; i < 16; i++) {
                                int l = l0 + i;
                                int q1 = (int)((ql[l]      & 0xF) |
                                    (((qh[l] >> 0) & 3) << 4)) - 32;
                                int q2 = (int)((ql[l + 32] & 0xF) |
                                    (((qh[l] >> 2) & 3) << 4)) - 32;
                                int q3 = (int)((ql[l]      >> 4)  |
                                    (((qh[l] >> 4) & 3) << 4)) - 32;
                                int q4 = (int)((ql[l + 32] >> 4)  |
                                    (((qh[l] >> 6) & 3) << 4)) - 32;
                                sum1 += (float)q1 * xb[l + 0];
                                sum2 += (float)q2 * xb[l + 32];
                                sum3 += (float)q3 * xb[l + 64];
                                sum4 += (float)q4 * xb[l + 96];
                            }
                            sums[ti] += d * (
                                (float)sc[is + 0] * sum1 +
                                (float)sc[is + 2] * sum2 +
                                (float)sc[is + 4] * sum3 +
                                (float)sc[is + 6] * sum4);
                        }
                        xb += 128;
                        ql += 64;
                        qh += 32;
                        sc += 8;
                    }
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += sums[ti];
        }
    }
}

#define Q6K_SCALAR_SDOT_TILE_T 4

void bn_quant_q6k_scalar_sdot_matmul_range(void *ctx,
                                            int row_start,
                                            int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_blocks_per_row = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q6K_SCALAR_SDOT_TILE_T) {
            int tile_n = t0 + Q6K_SCALAR_SDOT_TILE_T <= n_tokens
                ? Q6K_SCALAR_SDOT_TILE_T : n_tokens - t0;
            float sums[Q6K_SCALAR_SDOT_TILE_T] = {0};

            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ6K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                float d = bn_fp16_to_fp32(blk->d);

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    float dx = c->x_d[(size_t)t * n_blocks_per_row + b];
                    const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                    const int16_t *bsums =
                        c->x_bsums + ((size_t)t * n_blocks_per_row + b) * 16;
                    const uint8_t *ql = blk->ql;
                    const uint8_t *qh = blk->qh;
                    const int8_t *sc = blk->scales;

                    int32_t sumi = 0;
                    int32_t bias_corr = 0;
                    for (int chunk = 0; chunk < 2; chunk++) {
                        for (int is = 0; is < 2; is++) {
                            int l0 = is * 16;
                            int32_t sum1 = 0;
                            int32_t sum2 = 0;
                            int32_t sum3 = 0;
                            int32_t sum4 = 0;
                            for (int i = 0; i < 16; i++) {
                                int l = l0 + i;
                                uint8_t h = qh[l];
                                int q1 = (int)((ql[l] & 0x0f) | ((h & 0x03) << 4));
                                int q2 = (int)((ql[l + 32] & 0x0f) |
                                    (((h >> 2) & 0x03) << 4));
                                int q3 = (int)((ql[l] >> 4) |
                                    (((h >> 4) & 0x03) << 4));
                                int q4 = (int)((ql[l + 32] >> 4) |
                                    (((h >> 6) & 0x03) << 4));
                                sum1 += q1 * (int32_t)xb[l];
                                sum2 += q2 * (int32_t)xb[l + 32];
                                sum3 += q3 * (int32_t)xb[l + 64];
                                sum4 += q4 * (int32_t)xb[l + 96];
                            }
                            sumi += (int32_t)sc[is + 0] * sum1 +
                                    (int32_t)sc[is + 2] * sum2 +
                                    (int32_t)sc[is + 4] * sum3 +
                                    (int32_t)sc[is + 6] * sum4;
                        }
                        for (int g = 0; g < 8; g++)
                            bias_corr += (int32_t)sc[g] *
                                          (int32_t)bsums[chunk * 8 + g];

                        xb += 128;
                        ql += 64;
                        qh += 32;
                        sc += 8;
                    }

                    sums[ti] += d * dx * (float)(sumi - 32 * bias_corr);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += sums[ti];
        }
    }
}
