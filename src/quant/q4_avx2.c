#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

static inline __m256 dot_i8_float(__m256i w, __m256i x) {
    __m256i aw = _mm256_sign_epi8(w, w);
    __m256i sx = _mm256_sign_epi8(x, w);
    __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(p32);
}

void bn_quant_q4_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_bpr = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias8 = _mm_set1_epi8(8);

    for (int row = row_start; row < row_end; row++) {
        __m256 facc = _mm256_setzero_ps();

        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_bpr + b];
            _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);

            __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
            __m256i w = _mm256_set_m128i(
                _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias8),
                _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias8));
            __m256i xq = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));

            __m256 d = _mm256_set1_ps(bn_fp16_to_fp32(blk->d) * x_scales[b]);
            facc = _mm256_fmadd_ps(dot_i8_float(w, xq), d, facc);
        }

        c->out[row] = bn_avx2_hsum_ps(facc);
    }
}
