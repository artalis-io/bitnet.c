#include "quant_ctx.h"
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
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 6), mask_2), 4));

                // SDOT for 4 pairs (8 sub-blocks of 16 elements each = 128 elements)
                // w0 covers sub-blocks 0,1 (32 elements: low 16 + high 16 in 256-bit)
                /* Horizontal sum of a 128-bit lane (4 x int32 → scalar) */
                #define HSUM128(v128) ({ \
                    __m128i _h1 = _mm_hadd_epi32(v128, v128); \
                    __m128i _h2 = _mm_hadd_epi32(_h1, _h1); \
                    _mm_cvtsi128_si32(_h2); \
                })

                __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb));
                __m256i dot0 = bn_avx2_dpbusd(zero, w0, xv0);
                sumi += HSUM128(_mm256_castsi256_si128(dot0)) * (int32_t)sc[0]
                      + HSUM128(_mm256_extracti128_si256(dot0, 1)) * (int32_t)sc[1];

                __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + 32));
                __m256i dot1 = bn_avx2_dpbusd(zero, w1, xv1);
                sumi += HSUM128(_mm256_castsi256_si128(dot1)) * (int32_t)sc[2]
                      + HSUM128(_mm256_extracti128_si256(dot1, 1)) * (int32_t)sc[3];

                __m256i xv2 = _mm256_loadu_si256((const __m256i *)(xb + 64));
                __m256i dot2 = bn_avx2_dpbusd(zero, w2, xv2);
                sumi += HSUM128(_mm256_castsi256_si128(dot2)) * (int32_t)sc[4]
                      + HSUM128(_mm256_extracti128_si256(dot2, 1)) * (int32_t)sc[5];

                __m256i xv3 = _mm256_loadu_si256((const __m256i *)(xb + 96));
                __m256i dot3 = bn_avx2_dpbusd(zero, w3, xv3);
                sumi += HSUM128(_mm256_castsi256_si128(dot3)) * (int32_t)sc[6]
                      + HSUM128(_mm256_extracti128_si256(dot3, 1)) * (int32_t)sc[7];

                #undef HSUM128

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

// Helper: full horizontal sum of a 128-bit lane (4 x int32 → scalar)
static inline int32_t hsum_lane_lo(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i h1 = _mm_hadd_epi32(lo, lo);
    __m128i h2 = _mm_hadd_epi32(h1, h1);
    return _mm_cvtsi128_si32(h2);
}

static inline int32_t hsum_lane_hi(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i h1 = _mm_hadd_epi32(hi, hi);
    __m128i h2 = _mm_hadd_epi32(h1, h1);
    return _mm_cvtsi128_si32(h2);
}

// Fused Q6_K AVX2 matmul: load weight block once, dot against all n_tokens x vectors.
void bn_quant_q6k_avx2_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;

    const __m256i mask_lo4 = _mm256_set1_epi8(0xF);
    const __m256i mask_2 = _mm256_set1_epi8(3);
    const __m256i zero = _mm256_setzero_si256();

    for (int row = row_start; row < row_end; row++) {
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);

            // Pre-unpack weight vectors for both chunks (stays in L1 across tokens)
            __m256i W_all[8];  // 2 chunks x 4 vectors
            {
                const uint8_t *ql = blk->ql;
                const uint8_t *qh = blk->qh;
                for (int chunk = 0; chunk < 2; chunk++) {
                    __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql);
                    __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                    __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh);
                    int base = chunk * 4;

                    W_all[base+0] = _mm256_or_si256(
                        _mm256_and_si256(ql0, mask_lo4),
                        _mm256_slli_epi16(_mm256_and_si256(qh0, mask_2), 4));
                    W_all[base+1] = _mm256_or_si256(
                        _mm256_and_si256(ql1, mask_lo4),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 2), mask_2), 4));
                    W_all[base+2] = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 4), mask_2), 4));
                    W_all[base+3] = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 6), mask_2), 4));

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
                    int base = chunk * 4;
                    const int8_t *xbc = xb + chunk * 128;

                    /* All 4 DPBUSDs first (deferred hsum for ILP) */
                    __m256i dot0 = bn_avx2_dpbusd(zero, W_all[base+0],
                        _mm256_loadu_si256((const __m256i *)xbc));
                    __m256i dot1 = bn_avx2_dpbusd(zero, W_all[base+1],
                        _mm256_loadu_si256((const __m256i *)(xbc + 32)));
                    __m256i dot2 = bn_avx2_dpbusd(zero, W_all[base+2],
                        _mm256_loadu_si256((const __m256i *)(xbc + 64)));
                    __m256i dot3 = bn_avx2_dpbusd(zero, W_all[base+3],
                        _mm256_loadu_si256((const __m256i *)(xbc + 96)));

                    sumi += hsum_lane_lo(dot0) * (int32_t)sc[0]
                          + hsum_lane_hi(dot0) * (int32_t)sc[1]
                          + hsum_lane_lo(dot1) * (int32_t)sc[2]
                          + hsum_lane_hi(dot1) * (int32_t)sc[3]
                          + hsum_lane_lo(dot2) * (int32_t)sc[4]
                          + hsum_lane_hi(dot2) * (int32_t)sc[5]
                          + hsum_lane_lo(dot3) * (int32_t)sc[6]
                          + hsum_lane_hi(dot3) * (int32_t)sc[7];

                    for (int g = 0; g < 8; g++)
                        bias_corr += (int32_t)sc[g] * (int32_t)bsums[chunk * 8 + g];
                    sc += 8;
                }

                c->out[(size_t)t * rows + row] += d * dx * (float)(sumi - 32 * bias_corr);
            }
        }
    }
}
