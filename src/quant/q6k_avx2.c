#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q6k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ6KCtx *c = (BnQ6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo4 = _mm_set1_epi8(0xF);
    const __m128i mask_2 = _mm_set1_epi8(3);
    const __m128i bias32 = _mm_set1_epi8(32);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int chunk = 0; chunk < 2; chunk++) {
                __m128i ql0 = _mm_loadu_si128((const __m128i *)(ql));
                __m128i ql1 = _mm_loadu_si128((const __m128i *)(ql + 16));
                __m128i ql2 = _mm_loadu_si128((const __m128i *)(ql + 32));
                __m128i ql3 = _mm_loadu_si128((const __m128i *)(ql + 48));
                __m128i qh0 = _mm_loadu_si128((const __m128i *)(qh));
                __m128i qh1 = _mm_loadu_si128((const __m128i *)(qh + 16));

                __m128i w0a = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql0, mask_lo4), _mm_slli_epi16(_mm_and_si128(qh0, mask_2), 4)), bias32);
                __m128i w0b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql1, mask_lo4), _mm_slli_epi16(_mm_and_si128(qh1, mask_2), 4)), bias32);
                __m128i w1a = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql2, mask_lo4), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 2), mask_2), 4)), bias32);
                __m128i w1b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql3, mask_lo4), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 2), mask_2), 4)), bias32);
                __m128i w2a = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_lo4), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 4), mask_2), 4)), bias32);
                __m128i w2b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_lo4), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 4), mask_2), 4)), bias32);
                __m128i w3a = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql2, 4), mask_lo4), _mm_slli_epi16(_mm_srli_epi16(qh0, 6), 4)), bias32);
                __m128i w3b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql3, 4), mask_lo4), _mm_slli_epi16(_mm_srli_epi16(qh1, 6), 4)), bias32);

                __m256 acc = _mm256_setzero_ps();

                #define Q6K_AVX2_ACC_16(w128, xp, scale_val) do { \
                    __m256 vds = _mm256_set1_ps(d * (float)(scale_val)); \
                    __m256 w_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds); \
                    __m256 w_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(w_lo, _mm256_loadu_ps(xp))); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(w_hi, _mm256_loadu_ps(xp + 8))); \
                } while(0)

                Q6K_AVX2_ACC_16(w0a, xb +   0, sc[0]);
                Q6K_AVX2_ACC_16(w0b, xb +  16, sc[1]);
                Q6K_AVX2_ACC_16(w1a, xb +  32, sc[2]);
                Q6K_AVX2_ACC_16(w1b, xb +  48, sc[3]);
                Q6K_AVX2_ACC_16(w2a, xb +  64, sc[4]);
                Q6K_AVX2_ACC_16(w2b, xb +  80, sc[5]);
                Q6K_AVX2_ACC_16(w3a, xb +  96, sc[6]);
                Q6K_AVX2_ACC_16(w3b, xb + 112, sc[7]);

                #undef Q6K_AVX2_ACC_16

                row_sum += bn_avx2_hsum_ps(acc);

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}
