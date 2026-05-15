#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

static inline __m512i q4_join_256_zero(__m256i lo) {
    return _mm512_castsi256_si512(lo);
}

void bn_quant_q4_avx512_vnni_4row_range(void *ctx, int group_start, int group_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_bpr = c->W->cols / 32;
    int rows = c->W->rows;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m128i bias8 = _mm_set1_epi8(8);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;
        float row_sums[4] = {0};

        for (int b = 0; b < n_bpr; b++) {
            __m512i xq = q4_join_256_zero(
                _mm256_loadu_si256((const __m256i *)(x_q + b * 32)));
            float d_x = x_scales[b];

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ4_0 *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
                __m256i w256 = _mm256_set_m128i(
                    _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias8),
                    _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias8));
                __m256i x256 = _mm512_castsi512_si256(xq);
                __m512i aw = q4_join_256_zero(_mm256_sign_epi8(w256, w256));
                __m512i sx = q4_join_256_zero(_mm256_sign_epi8(x256, w256));
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
