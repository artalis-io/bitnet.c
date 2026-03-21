#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

// Q4_K AVX2 SDOT kernel with Q8_K x quantization:
// Same approach as NEON: unsigned nibbles, integer accumulation,
// bsums min correction, float conversion once per super-block.
// Uses bn_avx2_dpbusd for signed×signed byte dot product.
void bn_quant_q4k_avx2_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo = _mm256_set1_epi8(0xF);
    const __m256i zero = _mm256_setzero_si256();

    // kmask constants for batch scale decode
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            float dx   = x_d[b];
            const uint8_t *qs = blk->qs;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            // Batch-decode all 8 scales and 8 mins (kmask trick)
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

            // Min correction via bsums (integer)
            int32_t bsum_corr = 0;
            for (int j = 0; j < 8; j++)
                bsum_corr += (int32_t)mins[j] * ((int32_t)bsums[2*j] + (int32_t)bsums[2*j + 1]);

            // Integer accumulation: unsigned nibbles × signed x_q
            int32_t sumi = 0;
            for (int j = 0; j < BN_QK_K; j += 64) {
                int sub = j / 32;

                // Load 32 bytes of qs, split into low and high nibbles
                __m256i raw = _mm256_loadu_si256((const __m256i *)qs);
                __m256i lo = _mm256_and_si256(raw, mask_lo);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo);

                // Load 64 bytes of x_q (two 32-byte loads)
                __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb + j));
                __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + j + 32));

                // SDOT: unsigned nibbles × signed x_q
                // Note: bn_avx2_dpbusd handles signed×signed via sign trick.
                // Since nibbles are 0..15 (non-negative in signed int8),
                // the sign trick is a no-op for the weight operand.
                __m256i dot_lo = bn_avx2_dpbusd(zero, lo, xv0);
                __m256i dot_hi = bn_avx2_dpbusd(zero, hi, xv1);

                sumi += bn_avx2_hsum_epi32(dot_lo) * (int32_t)sc[sub]
                      + bn_avx2_hsum_epi32(dot_hi) * (int32_t)sc[sub + 1];

                qs += 32;
            }

            row_sum += dx * (d * (float)sumi - dmin * (float)bsum_corr);
        }
        c->out[row] = row_sum;
    }
}
