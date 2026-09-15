#include "quant_ctx.h"
#include "quant_kernels_avx2.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

static inline float iq4xs_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

void bn_quant_iq4xs_avx2_range(void *ctx, int row_start, int row_end) {
    BnIQ4XSCtx *c = (BnIQ4XSCtx *)ctx;
    const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;
    const __m128i codebook =
        _mm_loadu_si128((const __m128i *)bn_kvalues_iq4nl);
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4XS *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = iq4xs_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;
            const uint8_t *qs = blk->qs;

            __m256 acc = _mm256_setzero_ps();

            for (int j = 0; j < 8; j++) {
                // Extract 6-bit scale
                int lo = (blk->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
                int hi = (blk->scales_h >> (j * 2)) & 3;
                float dl = d * ((lo | (hi << 4)) - 32);

                __m128i packed =
                    _mm_loadu_si128((const __m128i *)qs);
                __m128i w_lo = _mm_shuffle_epi8(
                    codebook, _mm_and_si128(packed, nibble_mask));
                __m128i w_hi = _mm_shuffle_epi8(
                    codebook,
                    _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask));

                // AVX2: widen int8 to float, multiply by scale, FMA with x
                __m256 vdl = _mm256_set1_ps(dl);
                __m128i weights[2] = { w_lo, w_hi };
                for (int g = 0; g < 4; g++) {
                    __m128i w8 = g & 1
                        ? _mm_srli_si128(weights[g / 2], 8)
                        : weights[g / 2];
                    __m256 wf = _mm256_cvtepi32_ps(
                        _mm256_cvtepi8_epi32(w8));
                    __m256 xf = _mm256_loadu_ps(xb + g * 8);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(wf, vdl), xf, acc);
                }

                qs += 16;
                xb += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_iq4xs_avx2_q8k_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)c->W->data;
    int n_bpr = c->W->cols / BN_QK_K;
    const __m128i codebook =
        _mm_loadu_si128((const __m128i *)bn_kvalues_iq4nl);
    const __m128i mask = _mm_set1_epi8(0x0f);

    for (int row = row_start; row < row_end; row++) {
        __m256 accumf = _mm256_setzero_ps();
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockIQ4XS *blk =
                &blocks[(size_t)row * n_bpr + b];
            const uint8_t *qs = blk->qs;
            const int8_t *q8 = c->x_q + b * BN_QK_K;
            uint16_t sh = blk->scales_h;
            __m256i sumi1 = _mm256_setzero_si256();
            __m256i sumi2 = _mm256_setzero_si256();

            for (int ib = 0; ib < BN_QK_K / 32; ib += 2) {
                __m128i packed1 =
                    _mm_loadu_si128((const __m128i *)(qs + ib * 16));
                __m128i packed2 =
                    _mm_loadu_si128((const __m128i *)(qs + (ib + 1) * 16));
                __m256i w1 = _mm256_set_m128i(
                    _mm_shuffle_epi8(
                        codebook,
                        _mm_and_si128(_mm_srli_epi16(packed1, 4), mask)),
                    _mm_shuffle_epi8(codebook,
                                     _mm_and_si128(packed1, mask)));
                __m256i w2 = _mm256_set_m128i(
                    _mm_shuffle_epi8(
                        codebook,
                        _mm_and_si128(_mm_srli_epi16(packed2, 4), mask)),
                    _mm_shuffle_epi8(codebook,
                                     _mm_and_si128(packed2, mask)));
                __m256i x1 = _mm256_loadu_si256(
                    (const __m256i *)(q8 + ib * 32));
                __m256i x2 = _mm256_loadu_si256(
                    (const __m256i *)(q8 + (ib + 1) * 32));
                __m256i dot1 = _mm256_maddubs_epi16(
                    _mm256_sign_epi8(w1, w1), _mm256_sign_epi8(x1, w1));
                __m256i dot2 = _mm256_maddubs_epi16(
                    _mm256_sign_epi8(w2, w2), _mm256_sign_epi8(x2, w2));
                int16_t scale1 = (int16_t)(
                    (blk->scales_l[ib / 2] & 0x0f) |
                    ((sh << 4) & 0x30)) - 32;
                int16_t scale2 = (int16_t)(
                    (blk->scales_l[ib / 2] >> 4) |
                    ((sh << 2) & 0x30)) - 32;
                sh >>= 4;
                sumi1 = _mm256_add_epi32(
                    sumi1, _mm256_madd_epi16(
                               dot1, _mm256_set1_epi16(scale1)));
                sumi2 = _mm256_add_epi32(
                    sumi2, _mm256_madd_epi16(
                               dot2, _mm256_set1_epi16(scale2)));
            }
            float d = iq4xs_fp16_to_fp32(blk->d) * c->x_d[b];
            accumf = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
                accumf);
        }
        c->out[row] = bn_avx2_hsum_ps(accumf);
    }
}

typedef struct {
    float d[8];
    int8_t scales[(BN_QK_K / 16) * 8];
    int8_t qs[BN_QK_K * 8];
} BnIQ4XSPanelX8;

static void iq4xs_decode_panel_x8(BnIQ4XSPanelX8 *panel,
                                  const BnQWeight *weight,
                                  int n_bpr, int row0) {
    for (int b = 0; b < n_bpr; b++) {
        BnIQ4XSPanelX8 *dst = &panel[b];
        for (int r = 0; r < 8; r++) {
            size_t index = (size_t)(row0 + r) * n_bpr + b;
            if (weight->type == BN_GGUF_TENSOR_IQ4_XS) {
                const BnBlockIQ4XS *src =
                    &((const BnBlockIQ4XS *)weight->data)[index];
                dst->d[r] = iq4xs_fp16_to_fp32(src->d);
                for (int ib = 0; ib < BN_QK_K / 32; ib++) {
                    int ls =
                        ((src->scales_l[ib / 2] >> (4 * (ib & 1))) & 15) |
                        (((src->scales_h >> (2 * ib)) & 3) << 4);
                    dst->scales[(2 * ib + 0) * 8 + r] =
                        (int8_t)(ls - 32);
                    dst->scales[(2 * ib + 1) * 8 + r] =
                        (int8_t)(ls - 32);
                    for (int k = 0; k < 16; k++) {
                        uint8_t q = src->qs[ib * 16 + k];
                        int g = k / 4;
                        int lane = k & 3;
                        dst->qs[(2 * ib + 0) * 128 + g * 32 + r * 4 + lane] =
                            bn_kvalues_iq4nl[q & 15];
                        dst->qs[(2 * ib + 1) * 128 + g * 32 + r * 4 + lane] =
                            bn_kvalues_iq4nl[q >> 4];
                    }
                }
            } else if (weight->type == BN_GGUF_TENSOR_IQ3_S) {
                const BnBlockIQ3S *src =
                    &((const BnBlockIQ3S *)weight->data)[index];
                dst->d[r] = iq4xs_fp16_to_fp32(src->d);
                for (int ib = 0; ib < BN_QK_K / 32; ib++) {
                    int8_t scale = (int8_t)(1 + 2 *
                        ((src->scales[ib / 2] >> (4 * (ib & 1))) & 15));
                    dst->scales[(2 * ib + 0) * 8 + r] = scale;
                    dst->scales[(2 * ib + 1) * 8 + r] = scale;
                    for (int j = 0; j < 8; j++) {
                        int grid_index = src->qs[ib * 8 + j] |
                            (((src->qh[ib] >> j) & 1) << 8);
                        const uint8_t *grid = (const uint8_t *)
                            &bn_iq3s_grid[grid_index];
                        for (int k = 0; k < 4; k++) {
                            int lane = j * 4 + k;
                            int8_t value = (int8_t)grid[k];
                            if (src->signs[ib * 4 + lane / 8] &
                                (1u << (lane & 7))) value = (int8_t)-value;
                            dst->qs[ib * 256 + j * 32 + r * 4 + k] = value;
                        }
                    }
                }
            } else {
                const BnBlockIQ3XXS *src =
                    &((const BnBlockIQ3XXS *)weight->data)[index];
                const uint8_t *q3 = src->qs;
                const uint8_t *gas = q3 + BN_QK_K / 4;
                dst->d[r] = iq4xs_fp16_to_fp32(src->d) * 0.25f;
                for (int ib = 0; ib < BN_QK_K / 32; ib++) {
                    uint32_t aux;
                    memcpy(&aux, gas + 4 * ib, sizeof(aux));
                    int8_t scale = (int8_t)(2 * (aux >> 28) + 1);
                    dst->scales[(2 * ib + 0) * 8 + r] = scale;
                    dst->scales[(2 * ib + 1) * 8 + r] = scale;
                    for (int l = 0; l < 4; l++) {
                        uint8_t signs = bn_ksigns_iq2xs[
                            (aux >> (7 * l)) & 0x7f];
                        const uint8_t *g0 = (const uint8_t *)
                            &bn_iq3xxs_grid[q3[ib * 8 + 2 * l + 0]];
                        const uint8_t *g1 = (const uint8_t *)
                            &bn_iq3xxs_grid[q3[ib * 8 + 2 * l + 1]];
                        for (int k = 0; k < 4; k++) {
                            int8_t v0 = (int8_t)g0[k];
                            int8_t v1 = (int8_t)g1[k];
                            if (signs & bn_kmask_iq2xs[k]) v0 = (int8_t)-v0;
                            if (signs & bn_kmask_iq2xs[k + 4]) v1 = (int8_t)-v1;
                            int off = ib * 256 + l * 64 + r * 4 + k;
                            dst->qs[off] = v0;
                            dst->qs[off + 32] = v1;
                        }
                    }
                }
            }
        }
    }
}

static inline __m256i iq4xs_panel_dot4(__m256i acc, __m256i weights,
                                       __m256i activations) {
    __m256i abs_weights = _mm256_sign_epi8(weights, weights);
    __m256i signed_x = _mm256_sign_epi8(activations, weights);
    __m256i pairs = _mm256_maddubs_epi16(abs_weights, signed_x);
    return _mm256_add_epi32(
        acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

void bn_quant_iq_panel_avx2_matmul_range(void *ctx,
                                         int group_start,
                                         int group_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int n_bpr = c->cols / BN_QK_K;
    BnIQ4XSPanelX8 *panel =
        (BnIQ4XSPanelX8 *)malloc((size_t)n_bpr * sizeof(*panel));
    if (!panel) {
        for (int group = group_start; group < group_end; group++) {
            int row0 = group * 8;
            for (int t = 0; t < c->n_tokens; t++) {
                BnKQuantSdotCtx fallback = {
                    c->out + (size_t)t * c->W->rows, c->W,
                    c->x_q + (size_t)t * c->cols,
                    c->x_d + (size_t)t * n_bpr,
                    c->x_bsums + (size_t)t * n_bpr * 16, c->prepared
                };
                if (c->W->type == BN_GGUF_TENSOR_IQ3_S)
                    bn_quant_iq3s_avx2_q8k_range(&fallback, row0, row0 + 8);
                else if (c->W->type == BN_GGUF_TENSOR_IQ3_XXS)
                    bn_quant_iq3xxs_avx2_q8k_range(&fallback, row0, row0 + 8);
                else
                    bn_quant_iq4xs_avx2_q8k_range(&fallback, row0, row0 + 8);
            }
        }
        return;
    }

    for (int group = group_start; group < group_end; group++) {
        int row0 = group * 8;
        iq4xs_decode_panel_x8(panel, c->W, n_bpr, row0);
        for (int t = 0; t < c->n_tokens; t++) {
            __m256 sumf = _mm256_setzero_ps();
            for (int b = 0; b < n_bpr; b++) {
                const BnIQ4XSPanelX8 *p = &panel[b];
                const int8_t *x = c->x_q +
                    (size_t)t * c->cols + (size_t)b * BN_QK_K;
                __m256i sumi = _mm256_setzero_si256();
                for (int sb = 0; sb < BN_QK_K / 16; sb++) {
                    __m128i x16 = _mm_loadu_si128(
                        (const __m128i *)(x + sb * 16));
                    __m256i xv = _mm256_broadcastsi128_si256(x16);
                    __m256i isum = _mm256_setzero_si256();
                    const int8_t *w = p->qs + sb * 128;
                    isum = iq4xs_panel_dot4(
                        isum, _mm256_loadu_si256((const __m256i *)(w + 0)),
                        _mm256_shuffle_epi32(xv, 0x00));
                    isum = iq4xs_panel_dot4(
                        isum, _mm256_loadu_si256((const __m256i *)(w + 32)),
                        _mm256_shuffle_epi32(xv, 0x55));
                    isum = iq4xs_panel_dot4(
                        isum, _mm256_loadu_si256((const __m256i *)(w + 64)),
                        _mm256_shuffle_epi32(xv, 0xaa));
                    isum = iq4xs_panel_dot4(
                        isum, _mm256_loadu_si256((const __m256i *)(w + 96)),
                        _mm256_shuffle_epi32(xv, 0xff));
                    __m256i scales = _mm256_cvtepi8_epi32(
                        _mm_loadl_epi64((const __m128i *)(p->scales + sb * 8)));
                    sumi = _mm256_add_epi32(
                        sumi, _mm256_mullo_epi32(isum, scales));
                }
                __m256 scale = _mm256_mul_ps(
                    _mm256_loadu_ps(p->d),
                    _mm256_set1_ps(c->x_d[(size_t)t * n_bpr + b]));
                sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi), scale, sumf);
            }
            _mm256_storeu_ps(c->out + (size_t)t * c->W->rows + row0, sumf);
        }
    }
    free(panel);
}

#define IQ4XS_MATMUL_TILE_T 8

void bn_quant_iq4xs_avx2_matmul_range(void *ctx,
                                      int row_start,
                                      int row_end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)c->W->data;
    int n_bpr = c->cols / BN_QK_K;
    int rows = c->W->rows;
    const __m128i codebook =
        _mm_loadu_si128((const __m128i *)bn_kvalues_iq4nl);
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);

    for (int row = row_start; row < row_end; row += 3) {
        int nrows = row + 3 <= row_end ? 3 : row_end - row;
        for (int t0 = 0; t0 < c->n_tokens; t0 += IQ4XS_MATMUL_TILE_T) {
            int tile_n = t0 + IQ4XS_MATMUL_TILE_T <= c->n_tokens
                ? IQ4XS_MATMUL_TILE_T : c->n_tokens - t0;
            float row_sum[3][IQ4XS_MATMUL_TILE_T] = {{0}};

            if (tile_n == IQ4XS_MATMUL_TILE_T) {
                for (int b = 0; b < n_bpr; b++) {
                  for (int rr = 0; rr < nrows; rr++) {
                    const BnBlockIQ4XS *blk =
                        &blocks[(size_t)(row + rr) * n_bpr + b];
                    float d = iq4xs_fp16_to_fp32(blk->d);
                    const uint8_t *qs = blk->qs;
                    __m256 a0 = _mm256_setzero_ps();
                    __m256 a1 = _mm256_setzero_ps();
                    __m256 a2 = _mm256_setzero_ps();
                    __m256 a3 = _mm256_setzero_ps();
                    __m256 a4 = _mm256_setzero_ps();
                    __m256 a5 = _mm256_setzero_ps();
                    __m256 a6 = _mm256_setzero_ps();
                    __m256 a7 = _mm256_setzero_ps();

                    for (int j = 0; j < 8; j++) {
                        int lo = (blk->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
                        int hi = (blk->scales_h >> (j * 2)) & 3;
                        __m256 vdl = _mm256_set1_ps(
                            d * ((lo | (hi << 4)) - 32));
                        __m128i packed =
                            _mm_loadu_si128((const __m128i *)qs);
                        __m128i weights[2] = {
                            _mm_shuffle_epi8(
                                codebook, _mm_and_si128(packed, nibble_mask)),
                            _mm_shuffle_epi8(
                                codebook, _mm_and_si128(
                                    _mm_srli_epi16(packed, 4), nibble_mask))
                        };

                        for (int g = 0; g < 4; g++) {
                            __m128i w8 = g & 1
                                ? _mm_srli_si128(weights[g / 2], 8)
                                : weights[g / 2];
                            __m256 wf = _mm256_cvtepi32_ps(
                                _mm256_cvtepi8_epi32(w8));
                            __m256 wd = _mm256_mul_ps(wf, vdl);
                            size_t xoff = (size_t)b * BN_QK_K + j * 32 + g * 8;
#define IQ4XS_FMA_TOKEN(N) \
                            a##N = _mm256_fmadd_ps( \
                                wd, _mm256_loadu_ps(c->x + \
                                    (size_t)(t0 + N) * c->cols + xoff), a##N)
                            IQ4XS_FMA_TOKEN(0);
                            IQ4XS_FMA_TOKEN(1);
                            IQ4XS_FMA_TOKEN(2);
                            IQ4XS_FMA_TOKEN(3);
                            IQ4XS_FMA_TOKEN(4);
                            IQ4XS_FMA_TOKEN(5);
                            IQ4XS_FMA_TOKEN(6);
                            IQ4XS_FMA_TOKEN(7);
#undef IQ4XS_FMA_TOKEN
                        }
                        qs += 16;
                    }

                    row_sum[rr][0] += bn_avx2_hsum_ps(a0);
                    row_sum[rr][1] += bn_avx2_hsum_ps(a1);
                    row_sum[rr][2] += bn_avx2_hsum_ps(a2);
                    row_sum[rr][3] += bn_avx2_hsum_ps(a3);
                    row_sum[rr][4] += bn_avx2_hsum_ps(a4);
                    row_sum[rr][5] += bn_avx2_hsum_ps(a5);
                    row_sum[rr][6] += bn_avx2_hsum_ps(a6);
                    row_sum[rr][7] += bn_avx2_hsum_ps(a7);
                  }
                }

                for (int rr = 0; rr < nrows; rr++)
                    for (int ti = 0; ti < IQ4XS_MATMUL_TILE_T; ti++)
                        c->out[(size_t)(t0 + ti) * rows + row + rr] =
                            row_sum[rr][ti];
                continue;
            }

            for (int b = 0; b < n_bpr; b++) {
              for (int rr = 0; rr < nrows; rr++) {
                const BnBlockIQ4XS *blk =
                    &blocks[(size_t)(row + rr) * n_bpr + b];
                float d = iq4xs_fp16_to_fp32(blk->d);
                const uint8_t *qs = blk->qs;
                __m256 acc[IQ4XS_MATMUL_TILE_T];
                for (int ti = 0; ti < tile_n; ti++)
                    acc[ti] = _mm256_setzero_ps();

                for (int j = 0; j < 8; j++) {
                    int lo = (blk->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
                    int hi = (blk->scales_h >> (j * 2)) & 3;
                    __m256 vdl = _mm256_set1_ps(d * ((lo | (hi << 4)) - 32));
                    __m128i packed =
                        _mm_loadu_si128((const __m128i *)qs);
                    __m128i weights[2] = {
                        _mm_shuffle_epi8(
                            codebook, _mm_and_si128(packed, nibble_mask)),
                        _mm_shuffle_epi8(
                            codebook, _mm_and_si128(
                                _mm_srli_epi16(packed, 4), nibble_mask))
                    };

                    for (int g = 0; g < 4; g++) {
                        __m128i w8 = g & 1
                            ? _mm_srli_si128(weights[g / 2], 8)
                            : weights[g / 2];
                        __m256 wf = _mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(w8));
                        __m256 wd = _mm256_mul_ps(wf, vdl);
                        for (int ti = 0; ti < tile_n; ti++) {
                            const float *xb = c->x +
                                (size_t)(t0 + ti) * c->cols +
                                b * BN_QK_K + j * 32 + g * 8;
                            acc[ti] = _mm256_fmadd_ps(
                                wd, _mm256_loadu_ps(xb), acc[ti]);
                        }
                    }
                    qs += 16;
                }

                for (int ti = 0; ti < tile_n; ti++)
                    row_sum[rr][ti] += bn_avx2_hsum_ps(acc[ti]);
              }
            }

            for (int rr = 0; rr < nrows; rr++)
                for (int ti = 0; ti < tile_n; ti++)
                    c->out[(size_t)(t0 + ti) * rows + row + rr] =
                        row_sum[rr][ti];
        }
    }
}
