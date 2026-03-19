#include "transformer_internal.h"

#ifdef __AVX2__

void bn_transformer_logits_i8_avx2_range(void *ctx, int v_start, int v_end) {
    BnLogitsI8Ctx *lc = (BnLogitsI8Ctx *)ctx;
    const int8_t *emb_i8 = lc->emb_i8;
    const float *emb_scales = lc->emb_scales;
    const int8_t *x_q = lc->x_q;
    float x_scale = lc->x_scale;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const int8_t *row = emb_i8 + (size_t)v * dim;
        _mm_prefetch((const char *)(row + 128), _MM_HINT_T0);
        _mm_prefetch((const char *)(row + 256), _MM_HINT_T1);
        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
        for (int d = 0; d < dim; d += 128) {
            _mm_prefetch((const char *)(row + d + 256), _MM_HINT_T0);
            acc0 = bn_avx2_dpbusd(acc0, _mm256_loadu_si256((const __m256i *)(row+d)),    _mm256_loadu_si256((const __m256i *)(x_q+d)));
            acc1 = bn_avx2_dpbusd(acc1, _mm256_loadu_si256((const __m256i *)(row+d+32)), _mm256_loadu_si256((const __m256i *)(x_q+d+32)));
            acc2 = bn_avx2_dpbusd(acc2, _mm256_loadu_si256((const __m256i *)(row+d+64)), _mm256_loadu_si256((const __m256i *)(x_q+d+64)));
            acc3 = bn_avx2_dpbusd(acc3, _mm256_loadu_si256((const __m256i *)(row+d+96)), _mm256_loadu_si256((const __m256i *)(x_q+d+96)));
        }
        __m256i sum4 = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
        int32_t total = bn_avx2_hsum_epi32(sum4);
        lc->logits[v] = (float)total * emb_scales[v] * x_scale;
    }
}

void bn_transformer_logits_f16_avx2_range(void *ctx, int v_start, int v_end) {
    BnLogitsCtx *lc = (BnLogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int d = 0; d < dim; d += 16) {
            __m256 f0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d)));
            __m256 f1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d + 8)));
            acc0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + d), acc0);
            acc1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + d + 8), acc1);
        }
        lc->logits[v] = bn_avx2_hsum_ps(_mm256_add_ps(acc0, acc1));
    }
}

#endif // __AVX2__
