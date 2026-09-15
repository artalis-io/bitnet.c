#include "quant_ctx.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <immintrin.h>
#include <stdlib.h>
#include <math.h>

static inline float iq4nl_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

static inline __m256i iq4nl_q8_block_dot(const BnBlockIQ4NL *w,
                                        const int8_t *x) {
    const __m128i lut = _mm_loadu_si128((const __m128i *)bn_kvalues_iq4nl);
    const __m128i mask = _mm_set1_epi8(15);
    __m128i q = _mm_loadu_si128((const __m128i *)w->qs);
    __m256i values = _mm256_set_m128i(
        _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), mask)),
        _mm_shuffle_epi8(lut, _mm_and_si128(q, mask)));
    __m256i a = _mm256_loadu_si256((const __m256i *)x);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(
        _mm256_sign_epi8(values, values), _mm256_sign_epi8(a, values)),
        _mm256_set1_epi16(1));
}

static inline int32_t iq4nl_q8_block_sum(__m256i dot) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(dot),
                               _mm256_extracti128_si256(dot, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

/* Match native Q8 activation quantization and packed/unpacked dot order. */
void bn_quant_iq4nl_avx2_q8_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)c->W->data;
    int nb = c->W->cols / 32;
    for (int r = row_start; r < row_end; r++) {
        const BnBlockIQ4NL *w = blocks + (size_t)r * nb;
        float sum = 0.0f;
        int b = 0;
        if (c->W->rows % 8 != 0) {
            /* Unpacked vector-dot arithmetic uses alternating block lanes. */
            __m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
            for (; b + 1 < nb; b += 2) {
                for (int k = 0; k < 2; k++) {
                    float d = iq4nl_fp16_to_fp32(w[b + k].d) * c->x_scales[b + k];
                    __m256i dot = iq4nl_q8_block_dot(&w[b + k], c->x_q + (b + k) * 32);
                    acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                        _mm256_cvtepi32_ps(dot), acc[k]);
                }
            }
            sum = bn_avx2_hsum_ps(_mm256_add_ps(acc[0], acc[1]));
            if (b < nb) {
                float d = iq4nl_fp16_to_fp32(w[b].d) * c->x_scales[b];
                int32_t dot = iq4nl_q8_block_sum(iq4nl_q8_block_dot(&w[b], c->x_q + b * 32));
                /* The standalone odd-block tail rounds the product before
                 * adding it to the reduced vector accumulator. */
                volatile float product = (float)dot * d;
                sum += product;
            }
            c->out[r] = sum;
            continue;
        }
        /* Eight-row panels reduce each block in integer space. Canonical
         * weights can reproduce that order without allocating a packed copy. */
        for (; b < nb; b++) {
            float d = iq4nl_fp16_to_fp32(w[b].d) * c->x_scales[b];
            int32_t dot = iq4nl_q8_block_sum(iq4nl_q8_block_dot(&w[b], c->x_q + b * 32));
            sum = fmaf((float)dot, d, sum);
        }
        c->out[r] = sum;
    }
}

void bn_quant_iq4nl_avx2_range(void *ctx, int row_start, int row_end) {
    BnIQ4NLCtx *c = (BnIQ4NLCtx *)ctx;
    const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;
    const __m128i codebook =
        _mm_loadu_si128((const __m128i *)bn_kvalues_iq4nl);
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4NL *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d = iq4nl_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;

            __m128i packed =
                _mm_loadu_si128((const __m128i *)blk->qs);
            __m128i weights[2] = {
                _mm_shuffle_epi8(
                    codebook, _mm_and_si128(packed, nibble_mask)),
                _mm_shuffle_epi8(
                    codebook, _mm_and_si128(
                        _mm_srli_epi16(packed, 4), nibble_mask))
            };

            // AVX2: widen int8 to int32, convert to float, multiply with x, accumulate
            __m256 acc = _mm256_setzero_ps();
            for (int g = 0; g < 4; g++) {
                __m128i w8 = g & 1
                    ? _mm_srli_si128(weights[g / 2], 8)
                    : weights[g / 2];
                __m256 wf = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(w8));
                __m256 xf = _mm256_loadu_ps(xb + g * 8);
                acc = _mm256_fmadd_ps(wf, xf, acc);
            }
            row_sum += bn_avx2_hsum_ps(acc) * d;
        }
        c->out[row] = row_sum;
    }
}

#define IQ4NL_MATMUL_TILE_T 8

void bn_quant_iq4nl_avx2_matmul_range(void *ctx,
                                      int row_start,
                                      int row_end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)c->W->data;
    int n_bpr = c->cols / 32;
    int rows = c->W->rows;
    const __m128i codebook =
        _mm_loadu_si128((const __m128i *)bn_kvalues_iq4nl);
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < c->n_tokens; t0 += IQ4NL_MATMUL_TILE_T) {
            int tile_n = t0 + IQ4NL_MATMUL_TILE_T <= c->n_tokens
                ? IQ4NL_MATMUL_TILE_T : c->n_tokens - t0;
            float row_sum[IQ4NL_MATMUL_TILE_T] = {0};

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockIQ4NL *blk = &blocks[(size_t)row * n_bpr + b];
                __m128i packed =
                    _mm_loadu_si128((const __m128i *)blk->qs);
                __m128i weights[2] = {
                    _mm_shuffle_epi8(
                        codebook, _mm_and_si128(packed, nibble_mask)),
                    _mm_shuffle_epi8(
                        codebook, _mm_and_si128(
                            _mm_srli_epi16(packed, 4), nibble_mask))
                };

                __m256 acc[IQ4NL_MATMUL_TILE_T];
                for (int ti = 0; ti < tile_n; ti++)
                    acc[ti] = _mm256_setzero_ps();
                for (int g = 0; g < 4; g++) {
                    __m128i w8 = g & 1
                        ? _mm_srli_si128(weights[g / 2], 8)
                        : weights[g / 2];
                    __m256 wf = _mm256_cvtepi32_ps(
                        _mm256_cvtepi8_epi32(w8));
                    for (int ti = 0; ti < tile_n; ti++) {
                        const float *xb = c->x +
                            (size_t)(t0 + ti) * c->cols + b * 32 + g * 8;
                        acc[ti] = _mm256_fmadd_ps(
                            wf, _mm256_loadu_ps(xb), acc[ti]);
                    }
                }
                float d = iq4nl_fp16_to_fp32(blk->d);
                for (int ti = 0; ti < tile_n; ti++)
                    row_sum[ti] += bn_avx2_hsum_ps(acc[ti]) * d;
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] = row_sum[ti];
        }
    }
}

static void iq4nl_q8_matmul_x8_fallback(void *ctx,
                                             int group_start,
                                             int group_end) {
    BnQ8MatmulCtx *c = (BnQ8MatmulCtx *)ctx;
    const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)c->W->data;
    int n_bpr = c->cols / 32;

    for (int group = group_start; group < group_end; group++) {
        int row0 = group * 8;
        for (int t = 0; t < c->n_tokens; t++) {
            __m256 sumf = _mm256_setzero_ps();
            const int8_t *x = c->x_q + (size_t)t * c->cols;
            const float *x_scales = c->x_scales + (size_t)t * n_bpr;
            for (int b = 0; b < n_bpr; b++) {
                int32_t dots[8] = {0};
                uint16_t scales[8];
                for (int r = 0; r < 8; r++) {
                    const BnBlockIQ4NL *blk =
                        &blocks[(size_t)(row0 + r) * n_bpr + b];
                    scales[r] = blk->d;
                    for (int k = 0; k < 16; k++) {
                        uint8_t q = blk->qs[k];
                        dots[r] += (int32_t)bn_kvalues_iq4nl[q & 15] *
                                   x[b * 32 + k];
                        dots[r] += (int32_t)bn_kvalues_iq4nl[q >> 4] *
                                   x[b * 32 + k + 16];
                    }
                }
                __m256 d = _mm256_set_ps(
                    iq4nl_fp16_to_fp32(scales[7]),
                    iq4nl_fp16_to_fp32(scales[6]),
                    iq4nl_fp16_to_fp32(scales[5]),
                    iq4nl_fp16_to_fp32(scales[4]),
                    iq4nl_fp16_to_fp32(scales[3]),
                    iq4nl_fp16_to_fp32(scales[2]),
                    iq4nl_fp16_to_fp32(scales[1]),
                    iq4nl_fp16_to_fp32(scales[0]));
                __m256i iv = _mm256_loadu_si256((const __m256i *)dots);
                __m256 scale = _mm256_mul_ps(
                    d, _mm256_set1_ps(x_scales[b]));
                sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iv), scale, sumf);
            }
            _mm256_storeu_ps(c->out + (size_t)t * c->W->rows + row0, sumf);
        }
    }
}

typedef struct {
    float d[8];
    int8_t qs[32 * 8];
} BnIQ4NLPanelX8;

void bn_quant_iq4nl_avx2_q8_matmul_x8_range(void *ctx,
                                          int group_start,
                                          int group_end) {
    BnQ8MatmulCtx *c = (BnQ8MatmulCtx *)ctx;
    const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)c->W->data;
    int nb = c->cols / 32;
    BnIQ4NLPanelX8 *panel = malloc((size_t)nb * sizeof(*panel));
    if (!panel) {
        iq4nl_q8_matmul_x8_fallback(ctx, group_start, group_end);
        return;
    }
    for (int group = group_start; group < group_end; group++) {
        int row0 = group * 8;
        /* Decode once for the entire token batch, interleaving four values
         * from each row so each SIMD lane accumulates one output row. */
        for (int b = 0; b < nb; b++) {
            for (int r = 0; r < 8; r++) {
                const BnBlockIQ4NL *w = &blocks[(size_t)(row0 + r) * nb + b];
                panel[b].d[r] = iq4nl_fp16_to_fp32(w->d);
                for (int k = 0; k < 16; k++) {
                    int off = (k / 4) * 32 + r * 4 + k % 4;
                    panel[b].qs[off] = bn_kvalues_iq4nl[w->qs[k] & 15];
                    panel[b].qs[off + 128] = bn_kvalues_iq4nl[w->qs[k] >> 4];
                }
            }
        }
        for (int t = 0; t < c->n_tokens; t++) {
            __m256 sum = _mm256_setzero_ps();
            for (int b = 0; b < nb; b++) {
                const int8_t *x = c->x_q + (size_t)t * c->cols + b * 32;
                __m256i dot = _mm256_setzero_si256();
                for (int h = 0; h < 2; h++) {
                    __m256i xv = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128((const __m128i *)(x + h * 16)));
                    for (int k = 0; k < 4; k++) {
                        __m256i w = _mm256_loadu_si256((const __m256i *)(
                            panel[b].qs + h * 128 + k * 32));
                        __m256i a;
                        switch (k) {
                            case 0: a = _mm256_shuffle_epi32(xv, 0x00); break;
                            case 1: a = _mm256_shuffle_epi32(xv, 0x55); break;
                            case 2: a = _mm256_shuffle_epi32(xv, 0xaa); break;
                            default: a = _mm256_shuffle_epi32(xv, 0xff); break;
                        }
                        __m256i pairs = _mm256_maddubs_epi16(
                            _mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w));
                        dot = _mm256_add_epi32(dot, _mm256_madd_epi16(
                            pairs, _mm256_set1_epi16(1)));
                    }
                }
                __m256 scale = _mm256_mul_ps(_mm256_loadu_ps(panel[b].d),
                    _mm256_set1_ps(c->x_scales[(size_t)t * nb + b]));
                sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), scale, sum);
            }
            _mm256_storeu_ps(c->out + (size_t)t * c->W->rows + row0, sum);
        }
    }
    free(panel);
}
