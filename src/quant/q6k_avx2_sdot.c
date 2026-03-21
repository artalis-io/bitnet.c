#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

// Q6_K AVX2 SDOT kernel with Q8_K x quantization:
// Unsigned 6-bit weights, integer accumulation, bsums bias correction.
void bn_quant_q6k_avx2_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo4 = _mm256_set1_epi8(0xF);
    const __m256i mask_2 = _mm256_set1_epi8(3);
    const __m256i zero = _mm256_setzero_si256();

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d  = bn_fp16_to_fp32(blk->d);
            float dx = x_d[b];
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            int32_t sumi = 0;
            int32_t bias_corr = 0;

            for (int chunk = 0; chunk < 2; chunk++) {
                // Load ql (64 bytes) and qh (32 bytes) for this chunk
                __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql);
                __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh);

                // Unpack 8 weight vectors — UNSIGNED (0..63)
                // w0: low nibble of ql[0..31] | (qh & 3) << 4
                __m256i w0 = _mm256_or_si256(
                    _mm256_and_si256(ql0, mask_lo4),
                    _mm256_slli_epi16(_mm256_and_si256(qh0, mask_2), 4));
                // w1: low nibble of ql[32..63] | ((qh >> 2) & 3) << 4
                __m256i w1 = _mm256_or_si256(
                    _mm256_and_si256(ql1, mask_lo4),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 2), mask_2), 4));
                // w2: high nibble of ql[0..31] | ((qh >> 4) & 3) << 4
                __m256i w2 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 4), mask_2), 4));
                // w3: high nibble of ql[32..63] | (qh >> 6) << 4
                __m256i w3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4),
                    _mm256_slli_epi16(_mm256_srli_epi16(qh0, 6), 4));

                // SDOT for 4 pairs (8 sub-blocks of 16 elements each = 128 elements)
                // w0 covers sub-blocks 0,1 (32 elements: low 16 + high 16 in 256-bit)
                __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb));
                __m256i dot0 = bn_avx2_dpbusd(zero, w0, xv0);
                // Split dot0 into sub-block halves (lanes 0-3 = sub0, lanes 4-7 = sub1)
                int32_t sum0_lo = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_castsi256_si128(dot0), _mm256_castsi256_si128(dot0)));
                int32_t sum0_hi = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_extracti128_si256(dot0, 1), _mm256_extracti128_si256(dot0, 1)));
                sumi += sum0_lo * (int32_t)sc[0] + sum0_hi * (int32_t)sc[1];

                __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + 32));
                __m256i dot1 = bn_avx2_dpbusd(zero, w1, xv1);
                int32_t sum1_lo = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_castsi256_si128(dot1), _mm256_castsi256_si128(dot1)));
                int32_t sum1_hi = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_extracti128_si256(dot1, 1), _mm256_extracti128_si256(dot1, 1)));
                sumi += sum1_lo * (int32_t)sc[2] + sum1_hi * (int32_t)sc[3];

                __m256i xv2 = _mm256_loadu_si256((const __m256i *)(xb + 64));
                __m256i dot2 = bn_avx2_dpbusd(zero, w2, xv2);
                int32_t sum2_lo = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_castsi256_si128(dot2), _mm256_castsi256_si128(dot2)));
                int32_t sum2_hi = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_extracti128_si256(dot2, 1), _mm256_extracti128_si256(dot2, 1)));
                sumi += sum2_lo * (int32_t)sc[4] + sum2_hi * (int32_t)sc[5];

                __m256i xv3 = _mm256_loadu_si256((const __m256i *)(xb + 96));
                __m256i dot3 = bn_avx2_dpbusd(zero, w3, xv3);
                int32_t sum3_lo = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_castsi256_si128(dot3), _mm256_castsi256_si128(dot3)));
                int32_t sum3_hi = _mm_cvtsi128_si32(_mm_hadd_epi32(
                    _mm256_extracti128_si256(dot3, 1), _mm256_extracti128_si256(dot3, 1)));
                sumi += sum3_lo * (int32_t)sc[6] + sum3_hi * (int32_t)sc[7];

                // Bias correction
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
