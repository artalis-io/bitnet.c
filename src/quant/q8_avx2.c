#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_q8_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d_w = bn_fp16_to_fp32(blk->d);
            float d_x = x_scales[b];

            __m256i w256 = _mm256_loadu_si256((const __m256i *)blk->qs);
            __m256i xq256 = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));

            __m256i acc = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq256);
            row_sum += d_w * d_x * (float)bn_avx2_hsum_epi32(acc);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q8_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    int rows = c->W->rows;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;
        float row_sums[4] = {0};

        for (int b = 0; b < n_blocks_per_row; b++) {
            __m256i xq256 = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));
            float d_x = x_scales[b];

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)(row0 + r) * n_blocks_per_row + b];
                _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);

                __m256i w256 = _mm256_loadu_si256((const __m256i *)blk->qs);
                __m256i acc = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq256);
                row_sums[r] += bn_fp16_to_fp32(blk->d) * d_x * (float)bn_avx2_hsum_epi32(acc);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sums[r];
    }
}

#define Q8_MATMUL_TILE_T 8

void bn_quant_q8_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / 32;
    int n_tokens = c->n_tokens;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;
    (void)c->prepared;

    for (int t0 = 0; t0 < n_tokens; t0 += Q8_MATMUL_TILE_T) {
        int t_end = t0 + Q8_MATMUL_TILE_T;
        if (t_end > n_tokens) t_end = n_tokens;
        int tn = t_end - t0;

        for (int row = row_start; row < row_end; row++) {
            __m256 facc[Q8_MATMUL_TILE_T];
            for (int ti = 0; ti < tn; ti++)
                facc[ti] = _mm256_setzero_ps();

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_bpr + b];
                __m256i w = _mm256_loadu_si256((const __m256i *)blk->qs);
                __m256 d_w = _mm256_set1_ps(bn_fp16_to_fp32(blk->d));

                for (int ti = 0; ti < tn; ti++) {
                    int t = t0 + ti;
                    __m256i xq = _mm256_loadu_si256(
                        (const __m256i *)(x_q + (size_t)t * cols + b * 32));
                    __m256 d = _mm256_mul_ps(
                        d_w, _mm256_set1_ps(x_scales[(size_t)t * n_bpr + b]));
                    __m256i dot = bn_avx2_dpbusd(_mm256_setzero_si256(), w, xq);
                    facc[ti] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), d, facc[ti]);
                }
            }

            for (int ti = 0; ti < tn; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += bn_avx2_hsum_ps(facc[ti]);
        }
    }
}
