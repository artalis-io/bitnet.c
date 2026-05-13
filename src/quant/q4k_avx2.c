#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q4k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ4KCtx *c = (BnQ4KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo = _mm_set1_epi8(0xF);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            _mm_prefetch((const char *)(blk + 4), _MM_HINT_T1);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                __m256 vds = _mm256_set1_ps(d * sc);
                __m256 vdm = _mm256_set1_ps(dmin * m);
                __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));
                __m128i w0 = _mm_and_si128(raw0, mask_lo);
                __m128i w1 = _mm_and_si128(raw1, mask_lo);

                #define Q4K_AVX2_ACC_16(w128, xp) do { \
                    __m256 wf_lo = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds, vdm); \
                    __m256 wf_hi = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds, vdm); \
                    acc = _mm256_fmadd_ps(wf_lo, _mm256_loadu_ps(xp), acc); \
                    acc = _mm256_fmadd_ps(wf_hi, _mm256_loadu_ps(xp + 8), acc); \
                } while(0)

                Q4K_AVX2_ACC_16(w0, xb + j);
                Q4K_AVX2_ACC_16(w1, xb + j + 16);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                vds = _mm256_set1_ps(d * sc);
                vdm = _mm256_set1_ps(dmin * m);
                w0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo);
                w1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo);
                Q4K_AVX2_ACC_16(w0, xb + j + 32);
                Q4K_AVX2_ACC_16(w1, xb + j + 48);

                #undef Q4K_AVX2_ACC_16

                qs += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}
