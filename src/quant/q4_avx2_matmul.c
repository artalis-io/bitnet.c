#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

/* Q4_0 AVX2 tiled matmul: TILE_T=8 token tiling + float accumulation.
 * Inlined dot_i8_float avoids the redundant integer accumulator.
 */

#define Q4_MATMUL_TILE_T 8

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
    int n_tokens;
    int cols;
} BnQ4MatmulCtx;

static inline __m256 dot_i8_float(__m256i w, __m256i x) {
    __m256i aw = _mm256_sign_epi8(w, w);
    __m256i sx = _mm256_sign_epi8(x, w);
    __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(p32);
}

void bn_quant_q4_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / 32;
    int n_tokens = c->n_tokens;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias8 = _mm_set1_epi8(8);

    for (int t0 = 0; t0 < n_tokens; t0 += Q4_MATMUL_TILE_T) {
        int t_end = t0 + Q4_MATMUL_TILE_T;
        if (t_end > n_tokens) t_end = n_tokens;
        int tn = t_end - t0;

        for (int row = row_start; row < row_end; row++) {
            __m256 facc[Q4_MATMUL_TILE_T];
            for (int i = 0; i < tn; i++) facc[i] = _mm256_setzero_ps();

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_bpr + b];
                __m256 d_q4 = _mm256_set1_ps(bn_fp16_to_fp32(blk->d));

                __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
                __m256i w = _mm256_set_m128i(
                    _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias8),
                    _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias8));

                for (int ti = 0; ti < tn; ti++) {
                    int t = t0 + ti;
                    __m256i xq = _mm256_loadu_si256((const __m256i *)(x_q + (size_t)t * cols + b * 32));
                    __m256 d = _mm256_mul_ps(d_q4, _mm256_set1_ps(x_scales[(size_t)t * n_bpr + b]));
                    facc[ti] = _mm256_fmadd_ps(dot_i8_float(w, xq), d, facc[ti]);
                }
            }

            for (int ti = 0; ti < tn; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += bn_avx2_hsum_ps(facc[ti]);
        }
    }
}
