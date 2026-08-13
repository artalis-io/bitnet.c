#include "quant_ctx.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

static inline float q4k_fp16_to_f32(uint16_t h) {
    return vgetq_lane_f32(
        vcvt_f32_f16(vld1_dup_f16((const float16_t *)&h)), 0);
}

// Q4_K SDOT kernel with Q8_K x quantization:
// - Unsigned nibbles (no bias subtract)
// - Integer accumulation within super-block (one x_d per 256 elements)
// - Min correction via bsums (integer, outside inner loop)
// - Float conversion once per super-block
void bn_quant_q4k_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4KSdotCtx *c = (BnQ4KSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int32x4_t zero = vdupq_n_s32(0);

    // kmask constants for batch scale decode
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = q4k_fp16_to_f32(blk->d);
            float dmin = q4k_fp16_to_f32(blk->dmin);
            float dx   = x_d[b];
            const uint8_t *qs = blk->qs;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            // Batch-decode all 8 scales and 8 mins (kmask trick)
            uint32_t utmp[3];
            memcpy(utmp, blk->scales, 12);

            // Extract mins before overwriting utmp[1]
            uint32_t m_lo = utmp[1] & kmask1;
            uint32_t m_hi = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);

            // Extract scales
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[0] &= kmask1;
            const uint8_t *sc = (const uint8_t *)utmp;

            // Min correction via bsums (integer):
            // For each sub-block j (8 total), mins[j] maps to bsums[2j] + bsums[2j+1]
            // (each sub-block = 32 elements = 2 bsum groups of 16)
            uint32x2_t mins_u32 = vdup_n_u32(0);
            mins_u32 = vset_lane_u32(m_lo, mins_u32, 0);
            mins_u32 = vset_lane_u32(m_hi, mins_u32, 1);
            const int16x8_t mins = vreinterpretq_s16_u16(
                vmovl_u8(vreinterpret_u8_u32(mins_u32)));
            const int16x8_t bsum_pairs = vpaddq_s16(
                vld1q_s16(bsums), vld1q_s16(bsums + 8));
            const int32x4_t min_prod = vaddq_s32(
                vmull_s16(vget_low_s16(mins), vget_low_s16(bsum_pairs)),
                vmull_s16(vget_high_s16(mins), vget_high_s16(bsum_pairs)));
            const int32_t bsum_corr = vaddvq_s32(min_prod);

            int32_t block_sumi = 0;
            for (int j = 0; j < BN_QK_K; j += 64) {
                int sub = j / 32;

                // Low nibbles (sub-block 'sub'): unsigned 0..15
                uint8x16_t raw0 = vld1q_u8(qs);
                uint8x16_t raw1 = vld1q_u8(qs + 16);

                int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo));
                int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo));

                int32x4_t p0 = vdotq_s32(zero, lo0, vld1q_s8(xb + j));
                int32x4_t p1 = vdotq_s32(zero, lo1, vld1q_s8(xb + j + 16));
                int32_t sumi = (vaddvq_s32(p0) + vaddvq_s32(p1)) *
                               (int32_t)sc[sub];

                // High nibbles (sub-block 'sub+1'): unsigned 0..15
                int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));

                p0 = vdotq_s32(zero, hi0, vld1q_s8(xb + j + 32));
                p1 = vdotq_s32(zero, hi1, vld1q_s8(xb + j + 48));
                sumi += (vaddvq_s32(p0) + vaddvq_s32(p1)) *
                        (int32_t)sc[sub + 1];

                block_sumi += sumi;

                qs += 32;
            }
            float dd = dx * d;
            float ddmin = dx * dmin;
            row_sum -= ddmin * (float)bsum_corr;
            row_sum += dd * (float)block_sumi;
        }
        c->out[row] = row_sum;
    }
}

// Fused Q4_K matmul: load weight block once, dot against all n_tokens x vectors.
// out[t * rows + row] = sum(W[row] * X[t])
void bn_quant_q4k_neon_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int32x4_t zero = vdupq_n_s32(0);

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        for (int b = 0; b < n_bpr; b++) {
            // Load weight block ONCE for all tokens
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = q4k_fp16_to_f32(blk->d);
            float dmin = q4k_fp16_to_f32(blk->dmin);
            const uint8_t *qs = blk->qs;

            // Decode scales and mins once
            uint32_t utmp[3];
            memcpy(utmp, blk->scales, 12);
            uint32_t m_lo_w = utmp[1] & kmask1;
            uint32_t m_hi_w = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[0] &= kmask1;
            const uint8_t *sc = (const uint8_t *)utmp;
            uint32x2_t mins_u32 = vdup_n_u32(0);
            mins_u32 = vset_lane_u32(m_lo_w, mins_u32, 0);
            mins_u32 = vset_lane_u32(m_hi_w, mins_u32, 1);
            const int16x8_t mins = vreinterpretq_s16_u16(
                vmovl_u8(vreinterpret_u8_u32(mins_u32)));

            // Pre-load and unpack weight nibbles (stays in L1 across tokens)
            int8x16_t w_lo0[4], w_lo1[4], w_hi0[4], w_hi1[4];
            {
                const uint8_t *qp = qs;
                for (int p = 0; p < 4; p++) {
                    uint8x16_t raw0 = vld1q_u8(qp);
                    uint8x16_t raw1 = vld1q_u8(qp + 16);
                    w_lo0[p] = vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo));
                    w_lo1[p] = vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo));
                    w_hi0[p] = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                    w_hi1[p] = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                    qp += 32;
                }
            }

            // Iterate tokens: same weights, different x
            for (int t = 0; t < n_tokens; t++) {
                const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                float dx = c->x_d[(size_t)t * n_bpr + b];
                const int16_t *bsums = c->x_bsums + ((size_t)t * n_bpr + b) * 16;

                const int16x8_t bsum_pairs = vpaddq_s16(
                    vld1q_s16(bsums), vld1q_s16(bsums + 8));
                const int32x4_t min_prod = vaddq_s32(
                    vmull_s16(vget_low_s16(mins), vget_low_s16(bsum_pairs)),
                    vmull_s16(vget_high_s16(mins), vget_high_s16(bsum_pairs)));
                const int32_t bsum_corr = vaddvq_s32(min_prod);

                int32_t sumi = 0;
                for (int p = 0; p < 4; p++) {
                    int sub = p * 2;
                    int32x4_t p0 = vdotq_s32(zero, w_lo0[p], vld1q_s8(xb + p * 64));
                    int32x4_t p1 = vdotq_s32(zero, w_lo1[p], vld1q_s8(xb + p * 64 + 16));
                    sumi += (vaddvq_s32(p0) + vaddvq_s32(p1)) * (int32_t)sc[sub];

                    p0 = vdotq_s32(zero, w_hi0[p], vld1q_s8(xb + p * 64 + 32));
                    p1 = vdotq_s32(zero, w_hi1[p], vld1q_s8(xb + p * 64 + 48));
                    sumi += (vaddvq_s32(p0) + vaddvq_s32(p1)) * (int32_t)sc[sub + 1];
                }

                float dd = dx * d;
                float ddmin = dx * dmin;
                c->out[(size_t)t * rows + row] -=
                    ddmin * (float)bsum_corr;
                c->out[(size_t)t * rows + row] += dd * (float)sumi;
            }
        }
    }
}
