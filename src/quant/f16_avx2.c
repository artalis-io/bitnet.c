#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_f16_avx2_range(void *ctx, int row_start, int row_end) {
    BnF16Ctx *c = (BnF16Ctx *)ctx;
    const uint16_t *data = (const uint16_t *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint16_t *w = data + (size_t)row * cols;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int col = 0;
        for (; col + 15 < cols; col += 16) {
            _mm_prefetch((const char *)(w + col + 64), _MM_HINT_T0);
            __m128i h0 = _mm_loadu_si128((const __m128i *)(w + col));
            __m128i h1 = _mm_loadu_si128((const __m128i *)(w + col + 8));
            acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(h0),
                                   _mm256_loadu_ps(x + col), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(h1),
                                   _mm256_loadu_ps(x + col + 8), acc1);
        }

        float row_sum = bn_avx2_hsum_ps(_mm256_add_ps(acc0, acc1));
        for (; col < cols; col++) {
            row_sum += bn_fp16_to_fp32(w[col]) * x[col];
        }
        c->out[row] = row_sum;
    }
}
