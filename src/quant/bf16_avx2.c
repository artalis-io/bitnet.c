#include "quant_ctx.h"
#include "simd_helpers.h"
#include <string.h>
#include <immintrin.h>

void bn_quant_bf16_avx2_range(void *ctx, int row_start, int row_end) {
    BnBF16Ctx *c = (BnBF16Ctx *)ctx;
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

            // Load 8 BF16 values, widen u16->u32, shift left 16 to get F32 bits
            __m128i bf_lo = _mm_loadu_si128((const __m128i *)(w + col));
            __m128i bf_hi = _mm_loadu_si128((const __m128i *)(w + col + 8));
            __m256i w32_lo = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf_lo), 16);
            __m256i w32_hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf_hi), 16);
            __m256 wf_lo = _mm256_castsi256_ps(w32_lo);
            __m256 wf_hi = _mm256_castsi256_ps(w32_hi);

            acc0 = _mm256_fmadd_ps(wf_lo, _mm256_loadu_ps(x + col), acc0);
            acc1 = _mm256_fmadd_ps(wf_hi, _mm256_loadu_ps(x + col + 8), acc1);
        }

        float row_sum = bn_avx2_hsum_ps(_mm256_add_ps(acc0, acc1));

        // Scalar tail
        for (; col < cols; col++) {
            uint32_t bits = (uint32_t)w[col] << 16;
            float wf;
            memcpy(&wf, &bits, 4);
            row_sum += wf * x[col];
        }

        c->out[row] = row_sum;
    }
}

void bn_quant_bf16_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnBF16Ctx *c = (BnBF16Ctx *)ctx;
    const uint16_t *data = (const uint16_t *)c->W->data;
    int rows = c->W->rows;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int group = group_start; group < group_end; group++) {
        int row0 = group * 4;
        if (row0 >= rows) break;
        int nrows = rows - row0;
        if (nrows > 4) nrows = 4;

        __m256 acc0[4];
        __m256 acc1[4];
        for (int r = 0; r < nrows; r++) {
            acc0[r] = _mm256_setzero_ps();
            acc1[r] = _mm256_setzero_ps();
        }

        int col = 0;
        for (; col + 15 < cols; col += 16) {
            __m256 x0 = _mm256_loadu_ps(x + col);
            __m256 x1 = _mm256_loadu_ps(x + col + 8);

            for (int r = 0; r < nrows; r++) {
                const uint16_t *w = data + (size_t)(row0 + r) * cols;
                _mm_prefetch((const char *)(w + col + 64), _MM_HINT_T0);

                __m128i bf_lo = _mm_loadu_si128((const __m128i *)(w + col));
                __m128i bf_hi = _mm_loadu_si128((const __m128i *)(w + col + 8));
                __m256i w32_lo = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf_lo), 16);
                __m256i w32_hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf_hi), 16);
                __m256 wf_lo = _mm256_castsi256_ps(w32_lo);
                __m256 wf_hi = _mm256_castsi256_ps(w32_hi);

                acc0[r] = _mm256_fmadd_ps(wf_lo, x0, acc0[r]);
                acc1[r] = _mm256_fmadd_ps(wf_hi, x1, acc1[r]);
            }
        }

        for (int r = 0; r < nrows; r++) {
            const uint16_t *w = data + (size_t)(row0 + r) * cols;
            float row_sum = bn_avx2_hsum_ps(_mm256_add_ps(acc0[r], acc1[r]));

            for (int tail = col; tail < cols; tail++) {
                uint32_t bits = (uint32_t)w[tail] << 16;
                float wf;
                memcpy(&wf, &bits, 4);
                row_sum += wf * x[tail];
            }

            c->out[row0 + r] = row_sum;
        }
    }
}
