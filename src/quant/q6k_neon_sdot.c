#include "quant_internal.h"
#include <arm_neon.h>

void bn_quant_q6k_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ6KSdotCtx *c = (BnQ6KSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo4 = vdupq_n_u8(0xF);
    const uint8x16_t mask_2 = vdupq_n_u8(3);
    const int8x16_t bias32 = vdupq_n_s8(32);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;

            // 8 activation blocks per Q6_K block (256 / 32 = 8)
            const int8_t *xb = x_q + (b * BN_QK_K);
            const float *xs = x_scales + (b * 8);

            for (int chunk = 0; chunk < 2; chunk++) {
                uint8x16_t ql0 = vld1q_u8(ql);
                uint8x16_t ql1 = vld1q_u8(ql + 16);
                uint8x16_t ql2 = vld1q_u8(ql + 32);
                uint8x16_t ql3 = vld1q_u8(ql + 48);
                uint8x16_t qh0 = vld1q_u8(qh);
                uint8x16_t qh1 = vld1q_u8(qh + 16);

                // Unpack 8 weight vectors (identical to q6k_neon.c)
                int8x16_t w0a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql0, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh0, mask_2), 4))), bias32);
                int8x16_t w0b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql1, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh1, mask_2), 4))), bias32);
                int8x16_t w1a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql2, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_2), 4))), bias32);
                int8x16_t w1b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql3, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_2), 4))), bias32);
                int8x16_t w2a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql0, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_2), 4))), bias32);
                int8x16_t w2b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql1, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_2), 4))), bias32);
                int8x16_t w3a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql2, 4),
                    vshlq_n_u8(vshrq_n_u8(qh0, 6), 4))), bias32);
                int8x16_t w3b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql3, 4),
                    vshlq_n_u8(vshrq_n_u8(qh1, 6), 4))), bias32);

                // 4 pairs, each pair = 32 elements = 2 weight sub-blocks sharing 1 activation scale
                // Pair 0: w0a(sc[0]) + w0b(sc[1]), dx = xs[chunk*4 + 0]
                float dx0 = xs[chunk * 4 + 0];
                int32x4_t s0a = vdotq_s32(zero, w0a, vld1q_s8(xb));
                int32x4_t s0b = vdotq_s32(zero, w0b, vld1q_s8(xb + 16));
                row_sum += d * dx0 * ((float)sc[0] * (float)vaddvq_s32(s0a)
                                    + (float)sc[1] * (float)vaddvq_s32(s0b));

                // Pair 1: w1a(sc[2]) + w1b(sc[3]), dx = xs[chunk*4 + 1]
                float dx1 = xs[chunk * 4 + 1];
                int32x4_t s1a = vdotq_s32(zero, w1a, vld1q_s8(xb + 32));
                int32x4_t s1b = vdotq_s32(zero, w1b, vld1q_s8(xb + 48));
                row_sum += d * dx1 * ((float)sc[2] * (float)vaddvq_s32(s1a)
                                    + (float)sc[3] * (float)vaddvq_s32(s1b));

                // Pair 2: w2a(sc[4]) + w2b(sc[5]), dx = xs[chunk*4 + 2]
                float dx2 = xs[chunk * 4 + 2];
                int32x4_t s2a = vdotq_s32(zero, w2a, vld1q_s8(xb + 64));
                int32x4_t s2b = vdotq_s32(zero, w2b, vld1q_s8(xb + 80));
                row_sum += d * dx2 * ((float)sc[4] * (float)vaddvq_s32(s2a)
                                    + (float)sc[5] * (float)vaddvq_s32(s2b));

                // Pair 3: w3a(sc[6]) + w3b(sc[7]), dx = xs[chunk*4 + 3]
                float dx3 = xs[chunk * 4 + 3];
                int32x4_t s3a = vdotq_s32(zero, w3a, vld1q_s8(xb + 96));
                int32x4_t s3b = vdotq_s32(zero, w3b, vld1q_s8(xb + 112));
                row_sum += d * dx3 * ((float)sc[6] * (float)vaddvq_s32(s3a)
                                    + (float)sc[7] * (float)vaddvq_s32(s3b));

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}
