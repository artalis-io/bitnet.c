#include "quant_internal.h"
#include <arm_neon.h>

// Q6_K SDOT kernel with Q8_K x quantization:
// - Unsigned 6-bit weights (no bias-32 subtract)
// - Integer accumulation within super-block (one x_d per 256 elements)
// - Bias correction via bsums (integer, outside inner loop)
// - Float conversion once per super-block
void bn_quant_q6k_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ6KSdotCtx *c = (BnQ6KSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const uint8x16_t mask_lo4 = vdupq_n_u8(0xF);
    const uint8x16_t mask_2 = vdupq_n_u8(3);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d  = bn_fp16_to_fp32(blk->d);
            float dx = x_d[b];
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            // Integer accumulation
            int32_t sumi = 0;
            int32_t bias_corr = 0;

            for (int chunk = 0; chunk < 2; chunk++) {
                uint8x16_t ql0 = vld1q_u8(ql);
                uint8x16_t ql1 = vld1q_u8(ql + 16);
                uint8x16_t ql2 = vld1q_u8(ql + 32);
                uint8x16_t ql3 = vld1q_u8(ql + 48);
                uint8x16_t qh0 = vld1q_u8(qh);
                uint8x16_t qh1 = vld1q_u8(qh + 16);

                // Unpack 8 weight vectors — UNSIGNED (0..63), no bias subtract
                int8x16_t w0a = vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql0, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh0, mask_2), 4)));
                int8x16_t w0b = vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql1, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh1, mask_2), 4)));
                int8x16_t w1a = vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql2, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_2), 4)));
                int8x16_t w1b = vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql3, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_2), 4)));
                int8x16_t w2a = vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql0, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_2), 4)));
                int8x16_t w2b = vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql1, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_2), 4)));
                int8x16_t w3a = vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql2, 4),
                    vshlq_n_u8(vshrq_n_u8(qh0, 6), 4)));
                int8x16_t w3b = vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql3, 4),
                    vshlq_n_u8(vshrq_n_u8(qh1, 6), 4)));

                // SDOT + integer scale accumulation (4 pairs × 2 sub-blocks each)
                int32x4_t s0a = vdotq_s32(zero, w0a, vld1q_s8(xb));
                int32x4_t s0b = vdotq_s32(zero, w0b, vld1q_s8(xb + 16));
                sumi += vaddvq_s32(s0a) * (int32_t)sc[0]
                      + vaddvq_s32(s0b) * (int32_t)sc[1];

                int32x4_t s1a = vdotq_s32(zero, w1a, vld1q_s8(xb + 32));
                int32x4_t s1b = vdotq_s32(zero, w1b, vld1q_s8(xb + 48));
                sumi += vaddvq_s32(s1a) * (int32_t)sc[2]
                      + vaddvq_s32(s1b) * (int32_t)sc[3];

                int32x4_t s2a = vdotq_s32(zero, w2a, vld1q_s8(xb + 64));
                int32x4_t s2b = vdotq_s32(zero, w2b, vld1q_s8(xb + 80));
                sumi += vaddvq_s32(s2a) * (int32_t)sc[4]
                      + vaddvq_s32(s2b) * (int32_t)sc[5];

                int32x4_t s3a = vdotq_s32(zero, w3a, vld1q_s8(xb + 96));
                int32x4_t s3b = vdotq_s32(zero, w3b, vld1q_s8(xb + 112));
                sumi += vaddvq_s32(s3a) * (int32_t)sc[6]
                      + vaddvq_s32(s3b) * (int32_t)sc[7];

                // Bias correction: sum(sc[g] * bsum[g]) for this chunk's 8 groups
                for (int g = 0; g < 8; g++)
                    bias_corr += (int32_t)sc[g] * (int32_t)bsums[chunk * 8 + g];

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }

            // Single float conversion per super-block
            // val = d * sc * (w_unsigned - 32) * x_q * x_d
            //     = d * x_d * (sc * w_unsigned * x_q - 32 * sc * sum(x_q))
            //     = d * x_d * (sumi - 32 * bias_corr)
            row_sum += d * dx * (float)(sumi - 32 * bias_corr);
        }
        c->out[row] = row_sum;
    }
}

// Fused Q6_K matmul: load weight block once, dot against all n_tokens x vectors.
void bn_quant_q6k_neon_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;

    const uint8x16_t mask_lo4 = vdupq_n_u8(0xF);
    const uint8x16_t mask_2 = vdupq_n_u8(3);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);

            // Pre-unpack weight vectors for both chunks (stays in L1 across tokens)
            int8x16_t W_all[16];  // 2 chunks × 8 vectors
            {
                const uint8_t *ql = blk->ql;
                const uint8_t *qh = blk->qh;
                for (int chunk = 0; chunk < 2; chunk++) {
                    uint8x16_t ql0 = vld1q_u8(ql), ql1 = vld1q_u8(ql + 16);
                    uint8x16_t ql2 = vld1q_u8(ql + 32), ql3 = vld1q_u8(ql + 48);
                    uint8x16_t qh0 = vld1q_u8(qh), qh1 = vld1q_u8(qh + 16);
                    int base = chunk * 8;
                    W_all[base+0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, mask_lo4), vshlq_n_u8(vandq_u8(qh0, mask_2), 4)));
                    W_all[base+1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, mask_lo4), vshlq_n_u8(vandq_u8(qh1, mask_2), 4)));
                    W_all[base+2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, mask_lo4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_2), 4)));
                    W_all[base+3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, mask_lo4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_2), 4)));
                    W_all[base+4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_2), 4)));
                    W_all[base+5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_2), 4)));
                    W_all[base+6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql2, 4), vshlq_n_u8(vshrq_n_u8(qh0, 6), 4)));
                    W_all[base+7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql3, 4), vshlq_n_u8(vshrq_n_u8(qh1, 6), 4)));
                    ql += 64; qh += 32;
                }
            }

            // Pre-read scales (16 int8 values)
            const int8_t *sc_base = blk->scales;

            for (int t = 0; t < n_tokens; t++) {
                const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                float dx = c->x_d[(size_t)t * n_bpr + b];
                const int16_t *bsums = c->x_bsums + ((size_t)t * n_bpr + b) * 16;
                const int8_t *sc = sc_base;

                int32_t sumi = 0, bias_corr = 0;
                for (int chunk = 0; chunk < 2; chunk++) {
                    int base = chunk * 8;
                    const int8_t *xbc = xb + chunk * 128;

                    sumi += vaddvq_s32(vdotq_s32(zero, W_all[base+0], vld1q_s8(xbc))) * (int32_t)sc[0]
                          + vaddvq_s32(vdotq_s32(zero, W_all[base+1], vld1q_s8(xbc + 16))) * (int32_t)sc[1];
                    sumi += vaddvq_s32(vdotq_s32(zero, W_all[base+2], vld1q_s8(xbc + 32))) * (int32_t)sc[2]
                          + vaddvq_s32(vdotq_s32(zero, W_all[base+3], vld1q_s8(xbc + 48))) * (int32_t)sc[3];
                    sumi += vaddvq_s32(vdotq_s32(zero, W_all[base+4], vld1q_s8(xbc + 64))) * (int32_t)sc[4]
                          + vaddvq_s32(vdotq_s32(zero, W_all[base+5], vld1q_s8(xbc + 80))) * (int32_t)sc[5];
                    sumi += vaddvq_s32(vdotq_s32(zero, W_all[base+6], vld1q_s8(xbc + 96))) * (int32_t)sc[6]
                          + vaddvq_s32(vdotq_s32(zero, W_all[base+7], vld1q_s8(xbc + 112))) * (int32_t)sc[7];

                    for (int g = 0; g < 8; g++)
                        bias_corr += (int32_t)sc[g] * (int32_t)bsums[chunk * 8 + g];
                    sc += 8;
                }

                c->out[(size_t)t * rows + row] += d * dx * (float)(sumi - 32 * bias_corr);
            }
        }
    }
}
