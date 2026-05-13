#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "quant_neon_helpers.h"
#include <string.h>
#include <arm_neon.h>

// Extract high bits for 16 consecutive l-positions from qh, testing bit `bit_pos`.
// qh[l] stores high bits for position l: bits are interleaved as lo/hi pairs per group.
// bit_pos = group*2 (0,2,4,6) for first half, group*2+1 (1,3,5,7) for second half.
// l_offset = starting l position (0 or 16).
// Returns 16 bytes, each 0x00 or 0x10.
static inline uint8x16_t q5k_extract_hb_neon(const uint8_t *qh, int l_offset, int bit_pos) {
    // Load 16 consecutive qh bytes starting at l_offset
    uint8x16_t qh_vec = vld1q_u8(qh + l_offset);
    // Test the target bit using a mask
    uint8x16_t mask = vdupq_n_u8(1 << bit_pos);
    uint8x16_t tested = vandq_u8(qh_vec, mask);
    uint8x16_t nonzero = vcgtq_u8(tested, vdupq_n_u8(0));
    return vandq_u8(nonzero, vdupq_n_u8(0x10));
}

void bn_quant_q5k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const uint8_t *qh = blk->qh;
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;
                int group = j / 64;  // 0..3
                int bit_lo = group * 2;      // bits 0,2,4,6 for first 32
                int bit_hi = group * 2 + 1;  // bits 1,3,5,7 for second 32

                uint8x16_t raw0 = vld1q_u8(qs);
                uint8x16_t raw1 = vld1q_u8(qs + 16);

                uint8x16_t hb0 = q5k_extract_hb_neon(qh, 0,  bit_lo);  // l=0..15, first half
                uint8x16_t hb1 = q5k_extract_hb_neon(qh, 16, bit_lo);  // l=16..31, first half
                uint8x16_t hb2 = q5k_extract_hb_neon(qh, 0,  bit_hi);  // l=0..15, second half
                uint8x16_t hb3 = q5k_extract_hb_neon(qh, 16, bit_hi);  // l=16..31, second half

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float ds = d * sc;
                float dm = dmin * m;
                int8x16_t w0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw0, mask_lo), hb0));
                int8x16_t w1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw1, mask_lo), hb1));
                BN_QK_ACC_SCALED_16(w0, xb + j, ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 16, ds, dm);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                ds = d * sc;
                dm = dmin * m;
                w0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw0, 4), hb2));
                w1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw1, 4), hb3));
                BN_QK_ACC_SCALED_16(w0, xb + j + 32, ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 48, ds, dm);

                qs += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q5k_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ5KSdotCtx *c = (BnQ5KSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int32x4_t zero = vdupq_n_s32(0);

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            float dx   = x_d[b];
            const uint8_t *qs = blk->qs;
            const uint8_t *qh = blk->qh;
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
                bsum_corr += (int32_t)mins[j] *
                              ((int32_t)bsums[2*j] + (int32_t)bsums[2*j + 1]);

            int32x4_t acc = zero;
            for (int j = 0; j < BN_QK_K; j += 64) {
                int sub = j / 32;
                int group = j / 64;
                int bit_lo = group * 2;
                int bit_hi = group * 2 + 1;

                uint8x16_t raw0 = vld1q_u8(qs);
                uint8x16_t raw1 = vld1q_u8(qs + 16);

                uint8x16_t hb0 = q5k_extract_hb_neon(qh, 0,  bit_lo);
                uint8x16_t hb1 = q5k_extract_hb_neon(qh, 16, bit_lo);
                uint8x16_t hb2 = q5k_extract_hb_neon(qh, 0,  bit_hi);
                uint8x16_t hb3 = q5k_extract_hb_neon(qh, 16, bit_hi);

                int8x16_t w0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw0, mask_lo), hb0));
                int8x16_t w1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw1, mask_lo), hb1));
                int32x4_t p0 = vdotq_s32(zero, w0, vld1q_s8(xb + j));
                int32x4_t p1 = vdotq_s32(zero, w1, vld1q_s8(xb + j + 16));
                acc = vmlaq_n_s32(acc, p0, (int32_t)sc[sub]);
                acc = vmlaq_n_s32(acc, p1, (int32_t)sc[sub]);

                w0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw0, 4), hb2));
                w1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw1, 4), hb3));
                p0 = vdotq_s32(zero, w0, vld1q_s8(xb + j + 32));
                p1 = vdotq_s32(zero, w1, vld1q_s8(xb + j + 48));
                acc = vmlaq_n_s32(acc, p0, (int32_t)sc[sub + 1]);
                acc = vmlaq_n_s32(acc, p1, (int32_t)sc[sub + 1]);

                qs += 32;
            }
            int32_t sumi = vaddvq_s32(acc);

            row_sum += dx * (d * (float)sumi - dmin * (float)bsum_corr);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q5k_neon_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int32x4_t zero = vdupq_n_s32(0);

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ5K *blk = &blocks[(size_t)row * n_bpr + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);

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

            int8x16_t w_lo0[4], w_lo1[4], w_hi0[4], w_hi1[4];
            {
                const uint8_t *qs = blk->qs;
                const uint8_t *qh = blk->qh;
                for (int p = 0; p < 4; p++) {
                    int bit_lo = p * 2;
                    int bit_hi = p * 2 + 1;
                    uint8x16_t raw0 = vld1q_u8(qs);
                    uint8x16_t raw1 = vld1q_u8(qs + 16);
                    uint8x16_t hb0 = q5k_extract_hb_neon(qh, 0,  bit_lo);
                    uint8x16_t hb1 = q5k_extract_hb_neon(qh, 16, bit_lo);
                    uint8x16_t hb2 = q5k_extract_hb_neon(qh, 0,  bit_hi);
                    uint8x16_t hb3 = q5k_extract_hb_neon(qh, 16, bit_hi);
                    w_lo0[p] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw0, mask_lo), hb0));
                    w_lo1[p] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(raw1, mask_lo), hb1));
                    w_hi0[p] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw0, 4), hb2));
                    w_hi1[p] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(raw1, 4), hb3));
                    qs += 32;
                }
            }

            for (int t = 0; t < n_tokens; t++) {
                const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                float dx = c->x_d[(size_t)t * n_bpr + b];
                const int16_t *bsums = c->x_bsums + ((size_t)t * n_bpr + b) * 16;

                int32_t bsum_corr = 0;
                for (int j = 0; j < 8; j++)
                    bsum_corr += (int32_t)mins[j] *
                                  ((int32_t)bsums[2*j] + (int32_t)bsums[2*j + 1]);

                int32_t sumi = 0;
                for (int p = 0; p < 4; p++) {
                    int sub = p * 2;
                    int base = p * 64;
                    int32x4_t p0 = vdotq_s32(zero, w_lo0[p], vld1q_s8(xb + base));
                    int32x4_t p1 = vdotq_s32(zero, w_lo1[p], vld1q_s8(xb + base + 16));
                    sumi += (vaddvq_s32(p0) + vaddvq_s32(p1)) * (int32_t)sc[sub];

                    p0 = vdotq_s32(zero, w_hi0[p], vld1q_s8(xb + base + 32));
                    p1 = vdotq_s32(zero, w_hi1[p], vld1q_s8(xb + base + 48));
                    sumi += (vaddvq_s32(p0) + vaddvq_s32(p1)) * (int32_t)sc[sub + 1];
                }

                c->out[(size_t)t * rows + row] +=
                    dx * (d * (float)sumi - dmin * (float)bsum_corr);
            }
        }
    }
}
