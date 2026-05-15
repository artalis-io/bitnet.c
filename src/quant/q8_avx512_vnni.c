#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

static inline __m512i q8_join_256_zero(__m256i lo) {
    return _mm512_castsi256_si512(lo);
}

void bn_quant_q8_avx512_vnni_4row_range(void *ctx, int group_start, int group_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_bpr = c->W->cols / 32;
    int rows = c->W->rows;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;
        float row_sums[4] = {0};

        for (int b = 0; b < n_bpr; b++) {
            __m512i xq = q8_join_256_zero(
                _mm256_loadu_si256((const __m256i *)(x_q + b * 32)));
            float d_x = x_scales[b];

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                __m256i w256 = _mm256_loadu_si256((const __m256i *)blk->qs);
                __m256i x256 = _mm512_castsi512_si256(xq);
                __m512i aw = q8_join_256_zero(_mm256_sign_epi8(w256, w256));
                __m512i sx = q8_join_256_zero(_mm256_sign_epi8(x256, w256));
                __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sx);
                row_sums[r] += bn_fp16_to_fp32(blk->d) * d_x *
                               (float)bn_avx512_hsum_epi32(acc);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sums[r];
    }
}

#endif
