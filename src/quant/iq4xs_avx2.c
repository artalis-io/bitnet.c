#include "quant_ctx.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <immintrin.h>

void bn_quant_iq4xs_avx2_range(void *ctx, int row_start, int row_end) {
    BnIQ4XSCtx *c = (BnIQ4XSCtx *)ctx;
    const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4XS *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;
            const uint8_t *qs = blk->qs;

            __m256 acc = _mm256_setzero_ps();

            for (int j = 0; j < 8; j++) {
                // Extract 6-bit scale
                int lo = (blk->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
                int hi = (blk->scales_h >> (j * 2)) & 3;
                float dl = d * ((lo | (hi << 4)) - 32);

                // Scalar decode through codebook
                int8_t tmp[32];
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = qs[i];
                    tmp[i]      = bn_kvalues_iq4nl[byte & 0xF];
                    tmp[i + 16] = bn_kvalues_iq4nl[byte >> 4];
                }

                // AVX2: widen int8 to float, multiply by scale, FMA with x
                __m256 vdl = _mm256_set1_ps(dl);
                for (int g = 0; g < 32; g += 8) {
                    __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64((const __m128i *)(tmp + g))));
                    __m256 xf = _mm256_loadu_ps(xb + g);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(wf, vdl), xf, acc);
                }

                qs += 16;
                xb += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}
