#include "quant_ctx.h"
#include <immintrin.h>
#include <string.h>

/* Share unpacking across tokens, never across the FP32 block reduction.
 * A fixed four-token tile keeps the independent sums in registers. */
static inline void mxfp4_matmul_tile(BnQ4MatmulCtx *c, int row,
                                    int token, int count) {
    const BnBlockMXFP4 *blocks = (const BnBlockMXFP4 *)c->W->data;
    int nb = c->cols / 32;
    const __m128i values = _mm_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(15);
    float sums[4] = {0};
    for (int b = 0; b < nb; b++) {
        const BnBlockMXFP4 *block = blocks + (size_t)row * nb + b;
        __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
        __m256i lo = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
            values, _mm_and_si128(packed, mask)));
        __m256i hi = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
            values, _mm_and_si128(_mm_srli_epi16(packed, 4), mask)));
        uint32_t bits = block->e < 2 ? 0x00200000u << block->e
                                     : (uint32_t)(block->e - 1) << 23;
        float scale;
        memcpy(&scale, &bits, sizeof(scale));
        for (int t = 0; t < count; t++) {
            const int8_t *x = c->x_q + (size_t)(token + t) * c->cols + b * 32;
            __m256i xl = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)x));
            __m256i xh = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + 16)));
            __m256i pairs = _mm256_add_epi32(_mm256_madd_epi16(lo, xl),
                                            _mm256_madd_epi16(hi, xh));
            __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(pairs),
                                        _mm256_extracti128_si256(pairs, 1));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            sums[t] += scale * c->x_scales[(size_t)(token + t) * nb + b] *
                       (float)_mm_cvtsi128_si32(sum);
        }
    }
    for (int t = 0; t < count; t++)
        c->out[(size_t)(token + t) * c->W->rows + row] = sums[t];
}

void bn_quant_mxfp4_avx2_matmul_range(void *ctx, int start, int end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    for (int row = start; row < end; row++) {
        int t = 0;
        for (; t + 4 <= c->n_tokens; t += 4)
            mxfp4_matmul_tile(c, row, t, 4);
        switch (c->n_tokens - t) {
            case 3: mxfp4_matmul_tile(c, row, t, 3); break;
            case 2: mxfp4_matmul_tile(c, row, t, 2); break;
            case 1: mxfp4_matmul_tile(c, row, t, 1); break;
        }
    }
}

/* Reduce integer products within a block before applying its scale. Keeping
 * the FP32 accumulation between blocks scalar preserves the canonical dot. */
void bn_quant_mxfp4_avx2_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockMXFP4 *blocks = (const BnBlockMXFP4 *)c->W->data;
    int nb = c->W->cols / 32;
    const __m128i values = _mm_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(15);
    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            const BnBlockMXFP4 *block = blocks + (size_t)row * nb + b;
            __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
            __m256i lo = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                values, _mm_and_si128(packed, mask)));
            __m256i hi = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                values, _mm_and_si128(_mm_srli_epi16(packed, 4), mask)));
            const int8_t *x = c->x_q + b * 32;
            __m256i xl = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)x));
            __m256i xh = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + 16)));
            __m256i pairs = _mm256_add_epi32(_mm256_madd_epi16(lo, xl),
                                            _mm256_madd_epi16(hi, xh));
            __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(pairs),
                                        _mm256_extracti128_si256(pairs, 1));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            uint32_t bits = block->e < 2 ? 0x00200000u << block->e
                                         : (uint32_t)(block->e - 1) << 23;
            float scale;
            memcpy(&scale, &bits, sizeof(scale));
            row_sum += scale * c->x_scales[b] * (float)_mm_cvtsi128_si32(sum);
        }
        c->out[row] = row_sum;
    }
}
