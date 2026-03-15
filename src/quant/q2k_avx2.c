#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q2k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ2KCtx *c = (BnQ2KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ2K *blocks = (const BnBlockQ2K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ2K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *q = blk->qs;
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();

            int is = 0, out_idx = 0;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int8_t tmp0[16], tmp1[16];
                    for (int l = 0; l < 16; l++) {
                        tmp0[l] = (int8_t)((q[l] >> shift) & 3);
                        tmp1[l] = (int8_t)((q[l + 16] >> shift) & 3);
                    }

                    #define Q2K_AVX2_ACC_16(w128, xp, vds, vdm) do { \
                        __m256 wf_lo = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds), vdm); \
                        __m256 wf_hi = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds), vdm); \
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(wf_lo, _mm256_loadu_ps(xp))); \
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(wf_hi, _mm256_loadu_ps(xp + 8))); \
                    } while(0)

                    {
                        uint8_t sc = blk->scales[is++];
                        __m256 vds = _mm256_set1_ps(d * (sc & 0xF));
                        __m256 vdm = _mm256_set1_ps(dmin * (sc >> 4));
                        __m128i w0 = _mm_loadu_si128((const __m128i *)tmp0);
                        Q2K_AVX2_ACC_16(w0, xb + out_idx, vds, vdm);
                        out_idx += 16;
                    }
                    {
                        uint8_t sc = blk->scales[is++];
                        __m256 vds = _mm256_set1_ps(d * (sc & 0xF));
                        __m256 vdm = _mm256_set1_ps(dmin * (sc >> 4));
                        __m128i w1 = _mm_loadu_si128((const __m128i *)tmp1);
                        Q2K_AVX2_ACC_16(w1, xb + out_idx, vds, vdm);
                        out_idx += 16;
                    }

                    #undef Q2K_AVX2_ACC_16

                    shift += 2;
                }
                q += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}
