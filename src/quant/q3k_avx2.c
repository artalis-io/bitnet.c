#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q3k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ3KCtx *c = (BnQ3KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ3K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);

            uint8_t scales[16];
            bn_q3k_unpack_scales(blk->scales, scales);

            const uint8_t *q  = blk->qs;
            const uint8_t *hm = blk->hmask;
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();

            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int8_t tmp0[16], tmp1[16];
                    for (int l = 0; l < 16; l++) {
                        tmp0[l] = (int8_t)(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                        tmp1[l] = (int8_t)(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    }

                    float dl0 = d * ((int)scales[is++] - 32);
                    __m256 vds0 = _mm256_set1_ps(dl0);
                    __m128i w0 = _mm_loadu_si128((const __m128i *)tmp0);
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0)), vds0), _mm256_loadu_ps(xb + n + j*32)));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0, 8))), vds0), _mm256_loadu_ps(xb + n + j*32 + 8)));

                    float dl1 = d * ((int)scales[is++] - 32);
                    __m256 vds1 = _mm256_set1_ps(dl1);
                    __m128i w1 = _mm_loadu_si128((const __m128i *)tmp1);
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1)), vds1), _mm256_loadu_ps(xb + n + j*32 + 16)));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1, 8))), vds1), _mm256_loadu_ps(xb + n + j*32 + 24)));

                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}
