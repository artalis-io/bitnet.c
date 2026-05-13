#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

/* 4-row Q4_0 matvec: signed dot → float accumulation.
 *
 * Inlines the i8×i8→float path (sign trick + maddubs + madd + cvt + FMA)
 * without the redundant integer accumulator that bn_avx2_dpbusd uses.
 * Saves 1 instruction per block vs calling bn_avx2_dpbusd + cvtepi32_ps.
 */

static inline __m256 dot_i8_float(__m256i w, __m256i x) {
    /* Signed × signed via sign trick: abs(w) × sign(x,w) */
    __m256i aw = _mm256_sign_epi8(w, w);
    __m256i sx = _mm256_sign_epi8(x, w);
    __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(p32);
}

void bn_quant_q4_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_bpr = c->W->cols / 32;
    int rows = c->W->rows;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias8 = _mm_set1_epi8(8);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        __m256 facc[4];
        for (int r = 0; r < nrows; r++) facc[r] = _mm256_setzero_ps();

        for (int b = 0; b < n_bpr; b++) {
            __m256i xq = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));
            __m256 d_x = _mm256_set1_ps(x_scales[b]);

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ4_0 *blk = &blocks[(row0 + r) * n_bpr + b];

                __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
                __m256i w = _mm256_set_m128i(
                    _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias8),
                    _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias8));

                __m256 d = _mm256_mul_ps(d_x, _mm256_set1_ps(bn_fp16_to_fp32(blk->d)));
                facc[r] = _mm256_fmadd_ps(dot_i8_float(w, xq), d, facc[r]);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = bn_avx2_hsum_ps(facc[r]);
    }
}
