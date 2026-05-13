#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q3k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ3KCtx *c = (BnQ3KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const float *x = c->x;

    const __m128i mask2 = _mm_set1_epi8(3);
    const __m128i bias4 = _mm_set1_epi8(4);

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
                    // Vectorized: load 16 bytes of q and hm, extract 2-bit values
                    __m128i q0 = _mm_loadu_si128((const __m128i *)q);
                    __m128i q1 = _mm_loadu_si128((const __m128i *)(q + 16));
                    __m128i hm0 = _mm_loadu_si128((const __m128i *)hm);
                    __m128i hm1 = _mm_loadu_si128((const __m128i *)(hm + 16));

                    // Extract low 2 bits after shift: (q >> shift) & 3
                    __m128i w0, w1;
                    if (shift == 0) {
                        w0 = _mm_and_si128(q0, mask2);
                        w1 = _mm_and_si128(q1, mask2);
                    } else {
                        w0 = _mm_and_si128(_mm_srli_epi16(q0, shift), mask2);
                        w1 = _mm_and_si128(_mm_srli_epi16(q1, shift), mask2);
                    }

                    // High bit: if (hm & m) is set, subtract 0; else subtract 4
                    // i.e., result = (q>>shift)&3 - (hm_bit_set ? 0 : 4)
                    // = (q>>shift)&3 - 4 + (hm_bit_set ? 4 : 0)
                    // Better: bias = 4 where hm bit is NOT set
                    __m128i vm = _mm_set1_epi8((char)m);
                    __m128i hm_set0 = _mm_cmpeq_epi8(_mm_and_si128(hm0, vm), _mm_setzero_si128());
                    __m128i hm_set1 = _mm_cmpeq_epi8(_mm_and_si128(hm1, vm), _mm_setzero_si128());
                    // hm_set is 0xFF where bit NOT set (should subtract 4), 0x00 where set
                    __m128i sub0 = _mm_and_si128(hm_set0, bias4);
                    __m128i sub1 = _mm_and_si128(hm_set1, bias4);

                    // Final: w = (q>>shift)&3 - sub (as signed int8)
                    w0 = _mm_sub_epi8(w0, sub0);
                    w1 = _mm_sub_epi8(w1, sub1);

                    float dl0 = d * ((int)scales[is++] - 32);
                    __m256 vds0 = _mm256_set1_ps(dl0);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0)), vds0), _mm256_loadu_ps(xb + n + j*32), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0, 8))), vds0), _mm256_loadu_ps(xb + n + j*32 + 8), acc);

                    float dl1 = d * ((int)scales[is++] - 32);
                    __m256 vds1 = _mm256_set1_ps(dl1);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1)), vds1), _mm256_loadu_ps(xb + n + j*32 + 16), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1, 8))), vds1), _mm256_loadu_ps(xb + n + j*32 + 24), acc);

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
