#include "quant_ctx.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <immintrin.h>

void bn_quant_iq4nl_avx2_range(void *ctx, int row_start, int row_end) {
    BnIQ4NLCtx *c = (BnIQ4NLCtx *)ctx;
    const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4NL *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;

            // Scalar decode through codebook into int8 buffer
            int8_t tmp[32];
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                tmp[i]      = bn_kvalues_iq4nl[byte & 0xF];
                tmp[i + 16] = bn_kvalues_iq4nl[byte >> 4];
            }

            // AVX2: widen int8 to int32, convert to float, multiply with x, accumulate
            __m256 acc = _mm256_setzero_ps();
            for (int g = 0; g < 32; g += 8) {
                __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(tmp + g))));
                __m256 xf = _mm256_loadu_ps(xb + g);
                acc = _mm256_fmadd_ps(wf, xf, acc);
            }
            row_sum += bn_avx2_hsum_ps(acc) * d;
        }
        c->out[row] = row_sum;
    }
}
