#include "quant_internal.h"
#include <arm_neon.h>
#include <string.h>

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
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
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
            uint8_t mins[8];
            memcpy(mins, &m_lo, 4);
            memcpy(mins + 4, &m_hi, 4);

            int32_t bsum_corr = 0;
            for (int j = 0; j < 8; j++)
                bsum_corr += (int32_t)mins[j] * ((int32_t)bsums[2*j] + (int32_t)bsums[2*j + 1]);

            // Integer accumulation: sumi += vaddvq(sdot) * scale_byte
            int32_t sumi = 0;
            for (int j = 0; j < BN_QK_K; j += 64) {
                int sub = j / 32;

                // Low nibbles (sub-block 'sub'): unsigned 0..15
                uint8x16_t raw0 = vld1q_u8(qs);
                uint8x16_t raw1 = vld1q_u8(qs + 16);

                int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo));
                int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo));

                int32x4_t p0 = vdotq_s32(zero, lo0, vld1q_s8(xb + j));
                int32x4_t p1 = vdotq_s32(zero, lo1, vld1q_s8(xb + j + 16));
                sumi += (vaddvq_s32(p0) + vaddvq_s32(p1)) * (int32_t)sc[sub];

                // High nibbles (sub-block 'sub+1'): unsigned 0..15
                int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));

                p0 = vdotq_s32(zero, hi0, vld1q_s8(xb + j + 32));
                p1 = vdotq_s32(zero, hi1, vld1q_s8(xb + j + 48));
                sumi += (vaddvq_s32(p0) + vaddvq_s32(p1)) * (int32_t)sc[sub + 1];

                qs += 32;
            }

            // Single float conversion per super-block
            row_sum += dx * (d * (float)sumi - dmin * (float)bsum_corr);
        }
        c->out[row] = row_sum;
    }
}
