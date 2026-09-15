#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <math.h>

#if !defined(__AVX512F__)
static inline float f32_avx2_reduce_16(const float lanes[16]) {
    float half[8];
    float quarter[4];
    for (int lane = 0; lane < 8; lane++)
        half[lane] = lanes[lane + 8] + lanes[lane];
    for (int lane = 0; lane < 4; lane++)
        quarter[lane] = half[lane + 4] + half[lane];
    return (quarter[0] + quarter[2]) + (quarter[1] + quarter[3]);
}

static inline float f32_avx2_hsum(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                            _mm256_extractf128_ps(value, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static inline float f32_avx2_hsum_reference(__m256 value) {
    __m128 half = _mm_add_ps(_mm256_castps256_ps128(value),
                             _mm256_extractf128_ps(value, 1));
    half = _mm_hadd_ps(half, half);
    return _mm_cvtss_f32(_mm_hadd_ps(half, half));
}
#endif

void bn_quant_f32_avx2_range(void *ctx, int row_start, int row_end) {
    BnF32Ctx *c = (BnF32Ctx *)ctx;
    const float *data = (const float *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const float *w = data + (size_t)row * cols;
#if defined(__AVX512F__)
        /* Decode uses four independent dot accumulators. Prefill below
         * deliberately keeps its matrix-path accumulation order. */
        __m512 acc[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                          _mm512_setzero_ps(), _mm512_setzero_ps() };

        int col = 0;
        for (; col + 63 < cols; col += 64) {
            for (int a = 0; a < 4; a++)
                acc[a] = _mm512_fmadd_ps(_mm512_loadu_ps(w + col + a * 16),
                                         _mm512_loadu_ps(x + col + a * 16),
                                         acc[a]);
        }

        float row_sum = _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(acc[0], acc[2]), _mm512_add_ps(acc[1], acc[3])));
#else
        float acc[4][16] = {{0.0f}};
        int col = 0;
        for (; col + 63 < cols; col += 64)
            for (int a = 0; a < 4; a++)
                for (int lane = 0; lane < 16; lane++) {
                    int i = col + a * 16 + lane;
                    acc[a][lane] = fmaf(w[i], x[i], acc[a][lane]);
                }
        float merged[16];
        for (int lane = 0; lane < 16; lane++)
            merged[lane] = (acc[0][lane] + acc[2][lane]) +
                           (acc[1][lane] + acc[3][lane]);
        float row_sum = f32_avx2_reduce_16(merged);
#endif
        for (; col < cols; col++) {
            /* Keep the scalar tail fused across GCC/Clang builds too. */
            row_sum = fmaf(w[col], x[col], row_sum);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_f32_avx2_reference_range(void *ctx, int row_start, int row_end) {
    BnF32Ctx *c = (BnF32Ctx *)ctx;
    const float *data = (const float *)c->W->data;
    int cols = c->W->cols;
    for (int row = row_start; row < row_end; row++) {
        const float *w = data + (size_t)row * cols;
#if defined(__AVX512F__)
        __m512 acc[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                          _mm512_setzero_ps(), _mm512_setzero_ps() };
        int col = 0;
        for (; col + 63 < cols; col += 64)
            for (int lane = 0; lane < 4; lane++)
                acc[lane] = _mm512_fmadd_ps(
                    _mm512_loadu_ps(w + col + lane * 16),
                    _mm512_loadu_ps(c->x + col + lane * 16), acc[lane]);
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(acc[0], acc[2]),
            _mm512_add_ps(acc[1], acc[3])));
#else
        __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                          _mm256_setzero_ps(), _mm256_setzero_ps() };
        int col = 0;
        for (; col + 31 < cols; col += 32)
            for (int lane = 0; lane < 4; lane++)
                acc[lane] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w + col + lane * 8),
                    _mm256_loadu_ps(c->x + col + lane * 8), acc[lane]);
        acc[0] = _mm256_add_ps(acc[0], acc[2]);
        acc[1] = _mm256_add_ps(acc[1], acc[3]);
        float sum = f32_avx2_hsum_reference(
            _mm256_add_ps(acc[0], acc[1]));
#endif
        for (; col < cols; col++)
            sum += w[col] * c->x[col];
        c->out[row] = sum;
    }
}

void bn_quant_f32_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    const float *data = (const float *)c->W->data;
    int rows = c->W->rows;

    for (int row = row_start; row < row_end; row++) {
        const float *w = data + (size_t)row * c->cols;
        for (int t = 0; t < c->n_tokens; t++) {
            const float *x = c->x + (size_t)t * c->cols;
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            int col = 0;
            for (; col + 15 < c->cols; col += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + col),
                                       _mm512_loadu_ps(x + col), acc);
            float sum = _mm512_reduce_add_ps(acc);
#else
            __m256 acc = _mm256_setzero_ps();
            int col = 0;
            for (; col + 7 < c->cols; col += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + col),
                                       _mm256_loadu_ps(x + col), acc);
            float sum = f32_avx2_hsum(acc);
#endif
            for (; col < c->cols; col++)
                sum += w[col] * x[col];
            c->out[(size_t)t * rows + row] = sum;
        }
    }
}
