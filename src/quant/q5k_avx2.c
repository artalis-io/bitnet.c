#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <immintrin.h>

// Extract high bit `bit_pos` from 16 consecutive qh bytes starting at l_offset.
// qh[l] stores high bits for position l across all groups.
// Returns 16 bytes, each 0x00 or 0x10.
static inline __m128i q5k_extract_hb(const uint8_t *qh, int l_offset, int bit_pos) {
    __m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l_offset));
    __m128i mask = _mm_set1_epi8((char)(1 << bit_pos));
    __m128i tested = _mm_and_si128(qh_vec, mask);
    __m128i is_zero = _mm_cmpeq_epi8(tested, _mm_setzero_si128());
    return _mm_andnot_si128(is_zero, _mm_set1_epi8(0x10));
}

void bn_quant_q5k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo = _mm_set1_epi8(0xF);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const float *xb = x + b * BN_QK_K;

            const uint8_t *qh = blk->qh;
            __m256 acc = _mm256_setzero_ps();

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;
                int group = j / 64;
                __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));

                int bit_lo = group * 2;      // bits 0,2,4,6
                int bit_hi = group * 2 + 1;  // bits 1,3,5,7
                __m128i hb0 = q5k_extract_hb(qh, 0,  bit_lo);   // l=0..15, first half
                __m128i hb1 = q5k_extract_hb(qh, 16, bit_lo);   // l=16..31, first half
                __m128i hb2 = q5k_extract_hb(qh, 0,  bit_hi);   // l=0..15, second half
                __m128i hb3 = q5k_extract_hb(qh, 16, bit_hi);   // l=16..31, second half

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                __m256 vds = _mm256_set1_ps(d * sc);
                __m256 vdm = _mm256_set1_ps(dmin * m);
                __m128i w0 = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                __m128i w1 = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);

                #define Q5K_AVX2_ACC_16(w128, xp) do { \
                    __m256 wf_lo = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds, vdm); \
                    __m256 wf_hi = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds, vdm); \
                    acc = _mm256_fmadd_ps(wf_lo, _mm256_loadu_ps(xp), acc); \
                    acc = _mm256_fmadd_ps(wf_hi, _mm256_loadu_ps(xp + 8), acc); \
                } while(0)

                Q5K_AVX2_ACC_16(w0, xb + j);
                Q5K_AVX2_ACC_16(w1, xb + j + 16);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                vds = _mm256_set1_ps(d * sc);
                vdm = _mm256_set1_ps(dmin * m);
                w0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                w1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);
                Q5K_AVX2_ACC_16(w0, xb + j + 32);
                Q5K_AVX2_ACC_16(w1, xb + j + 48);

                #undef Q5K_AVX2_ACC_16

                qs += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q5k_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo = _mm_set1_epi8(0xF);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;
        float row_sums[4] = {0};

        for (int b = 0; b < n_blocks_per_row; b++) {
            const float *xb = x + b * BN_QK_K;

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ5K *blk = &blocks[(size_t)(row0 + r) * n_blocks_per_row + b];
                _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);
                float d    = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qs = blk->qs;
                const uint8_t *qh = blk->qh;
                __m256 acc = _mm256_setzero_ps();

                for (int j = 0; j < BN_QK_K; j += 64) {
                    uint8_t sc, m;
                    int sub = j / 32;
                    int group = j / 64;
                    __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                    __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));

                    int bit_lo = group * 2;
                    int bit_hi = group * 2 + 1;
                    __m128i hb0 = q5k_extract_hb(qh, 0,  bit_lo);
                    __m128i hb1 = q5k_extract_hb(qh, 16, bit_lo);
                    __m128i hb2 = q5k_extract_hb(qh, 0,  bit_hi);
                    __m128i hb3 = q5k_extract_hb(qh, 16, bit_hi);

                    bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                    __m256 vds = _mm256_set1_ps(d * sc);
                    __m256 vdm = _mm256_set1_ps(dmin * m);
                    __m128i w0 = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                    __m128i w1 = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);

                    #define Q5K_AVX2_ACC_16(w128, xp) do { \
                        __m256 wf_lo = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds, vdm); \
                        __m256 wf_hi = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds, vdm); \
                        acc = _mm256_fmadd_ps(wf_lo, _mm256_loadu_ps(xp), acc); \
                        acc = _mm256_fmadd_ps(wf_hi, _mm256_loadu_ps(xp + 8), acc); \
                    } while(0)

                    Q5K_AVX2_ACC_16(w0, xb + j);
                    Q5K_AVX2_ACC_16(w1, xb + j + 16);

                    bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                    vds = _mm256_set1_ps(d * sc);
                    vdm = _mm256_set1_ps(dmin * m);
                    w0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                    w1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);
                    Q5K_AVX2_ACC_16(w0, xb + j + 32);
                    Q5K_AVX2_ACC_16(w1, xb + j + 48);

                    #undef Q5K_AVX2_ACC_16

                    qs += 32;
                }

                row_sums[r] += bn_avx2_hsum_ps(acc);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sums[r];
    }
}
