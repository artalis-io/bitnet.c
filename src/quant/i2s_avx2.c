#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_i2s_avx2_range(void *ctx, int row_start, int row_end) {
    BnI2SCtx *c = (BnI2SCtx *)ctx;
    int row_bytes = c->W->cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float combined_scale = c->combined_scale;
    int cols = c->W->cols;
    const int8_t *x_q = c->x_q;

    const __m256i mask3 = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi8(1);

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;

        __m256i iaccA = _mm256_setzero_si256();
        __m256i iaccB = _mm256_setzero_si256();
        __m256i iaccC = _mm256_setzero_si256();
        __m256i iaccD = _mm256_setzero_si256();

        while (done < cols) {
            _mm_prefetch((const char *)(rd + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(rd + 256), _MM_HINT_T1);
            __m256i raw = _mm256_loadu_si256((const __m256i *)rd);

            __m256i t0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3), one);
            __m256i t1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3), one);
            __m256i t2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3), one);
            __m256i t3 = _mm256_sub_epi8(_mm256_and_si256(raw, mask3), one);

            const int8_t *xp = x_q + done;
            iaccA = bn_avx2_dpbusd(iaccA, t0, _mm256_loadu_si256((const __m256i *)xp));
            iaccB = bn_avx2_dpbusd(iaccB, t1, _mm256_loadu_si256((const __m256i *)(xp + 32)));
            iaccC = bn_avx2_dpbusd(iaccC, t2, _mm256_loadu_si256((const __m256i *)(xp + 64)));
            iaccD = bn_avx2_dpbusd(iaccD, t3, _mm256_loadu_si256((const __m256i *)(xp + 96)));

            rd += 32;
            done += 128;
        }

        __m256i sum4 = _mm256_add_epi32(_mm256_add_epi32(iaccA, iaccB),
                                         _mm256_add_epi32(iaccC, iaccD));
        int32_t total = bn_avx2_hsum_epi32(sum4);
        c->out[row] = (float)total * combined_scale;
    }
}
