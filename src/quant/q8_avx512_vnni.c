#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)



/* Keep eight dot lanes. VL avoids computing an unused upper half; the
 * wider intrinsic preserves support for AVX512 builds without VL. */
static inline __m256i q8_vnni_dot(__m256i aw, __m256i sx) {
#if defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, sx);
#else
    __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
        _mm512_castsi256_si512(aw), _mm512_castsi256_si512(sx));
    return _mm512_castsi512_si256(dot);
#endif
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
        __m256 row_acc[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()
        };

        for (int b = 0; b < n_bpr; b++) {
            __m256i xq = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));
            float d_x = x_scales[b];

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                __m256i w256 = _mm256_loadu_si256((const __m256i *)blk->qs);
                __m256i x256 = xq;
                __m256i aw = _mm256_sign_epi8(w256, w256);
                __m256i sx = _mm256_sign_epi8(x256, w256);
                __m256i dot512 = q8_vnni_dot(aw, sx);
                __m256 scale = _mm256_set1_ps(
                    bn_fp16_to_fp32(blk->d) * d_x);
                row_acc[r] = _mm256_fmadd_ps(
                    scale,
                    _mm256_cvtepi32_ps(dot512),
                    row_acc[r]);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = bn_avx2_hsum_ps(row_acc[r]);
    }
}

#define Q8_AVX512_MATMUL_TILE_T 16

/* Called with constant token counts so the short expert-batch accumulators
 * remain independent registers rather than a dynamically indexed array. */
static inline void q8_vnni_short_row(const BnQ4MatmulCtx *c, int row,
                                     int t0, int nt) {
    const int cols = c->cols;
    const int nb = cols / 32;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data + (size_t)row * nb;
    __m256 acc[7];
    for (int ti = 0; ti < nt; ti++) acc[ti] = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i w = _mm256_loadu_si256((const __m256i *)blocks[b].qs);
#if defined(__F16C__)
        float dw = _cvtsh_ss(blocks[b].d);
#else
        float dw = bn_fp16_to_fp32(blocks[b].d);
#endif
        __m256i aw = _mm256_sign_epi8(w, w);
        for (int ti = 0; ti < nt; ti++) {
            int t = t0 + ti;
            __m256i x = _mm256_loadu_si256((const __m256i *)(
                c->x_q + (size_t)t * cols + b * 32));
            __m256i sx = _mm256_sign_epi8(x, w);
            __m256i dot = q8_vnni_dot(aw, sx);
            __m256 scale = _mm256_set1_ps(dw * c->x_scales[(size_t)t * nb + b]);
            acc[ti] = _mm256_fmadd_ps(scale,
                _mm256_cvtepi32_ps(dot), acc[ti]);
        }
    }
    for (int ti = 0; ti < nt; ti++)
        c->out[(size_t)(t0 + ti) * c->W->rows + row] = bn_avx2_hsum_ps(acc[ti]);
}

void bn_quant_q8_avx512_vnni_matmul_4row_range(void *ctx,
                                                int group_start,
                                                int group_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / 32;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = row0 + 4 <= rows ? 4 : rows - row0;

        for (int t0 = 0, tile_n; t0 < c->n_tokens; t0 += tile_n) {
            tile_n = c->n_tokens - t0;
            if (tile_n > Q8_AVX512_MATMUL_TILE_T)
                tile_n = Q8_AVX512_MATMUL_TILE_T;
            if (tile_n < 8) {
                for (int r = 0; r < nrows; r++) {
                    /* Specialize each remainder without rereading weights
                     * in separate 4/2/1-token passes. */
                    switch (tile_n) {
                        case 1: q8_vnni_short_row(c, row0 + r, t0, 1); break;
                        case 2: q8_vnni_short_row(c, row0 + r, t0, 2); break;
                        case 3: q8_vnni_short_row(c, row0 + r, t0, 3); break;
                        case 4: q8_vnni_short_row(c, row0 + r, t0, 4); break;
                        case 5: q8_vnni_short_row(c, row0 + r, t0, 5); break;
                        case 6: q8_vnni_short_row(c, row0 + r, t0, 6); break;
                        case 7: q8_vnni_short_row(c, row0 + r, t0, 7); break;
                    }
                }
                continue;
            }
            if (nrows == 4 && tile_n == Q8_AVX512_MATMUL_TILE_T) {
                /* One row reuses its weights across sixteen tokens while
                 * leaving registers for dot and scale temporaries. */
                for (int r = 0; r < 4; r++) {
                    __m256 acc[Q8_AVX512_MATMUL_TILE_T];
                    for (int ti = 0; ti < Q8_AVX512_MATMUL_TILE_T; ti++)
                        acc[ti] = _mm256_setzero_ps();
                    for (int b = 0; b < n_bpr; b++) {
                        const BnBlockQ8_0 *blk =
                            &blocks[(size_t)(row0 + r) * n_bpr + b];
                        __m256i w = _mm256_loadu_si256((const __m256i *)blk->qs);
#if defined(__F16C__)
                        float d_w = _cvtsh_ss(blk->d);
#else
                        float d_w = bn_fp16_to_fp32(blk->d);
#endif
                        __m256i aw = _mm256_sign_epi8(w, w);
                        for (int ti = 0; ti < Q8_AVX512_MATMUL_TILE_T; ti++) {
                            int t = t0 + ti;
                            __m256i x = _mm256_loadu_si256((const __m256i *)(
                                c->x_q + (size_t)t * cols + b * 32));
                            __m256i dot = q8_vnni_dot(aw, _mm256_sign_epi8(x, w));
                            __m256 scale = _mm256_set1_ps(
                                d_w * c->x_scales[(size_t)t * n_bpr + b]);
                            acc[ti] = _mm256_fmadd_ps(
                                scale, _mm256_cvtepi32_ps(dot), acc[ti]);
                        }
                    }
                    for (int ti = 0; ti < Q8_AVX512_MATMUL_TILE_T; ti++)
                        c->out[(size_t)(t0 + ti) * rows + row0 + r] =
                            bn_avx2_hsum_ps(acc[ti]);
                }
                continue;
            }
            __m256 row_acc[4][Q8_AVX512_MATMUL_TILE_T];
            for (int r = 0; r < nrows; r++)
                for (int ti = 0; ti < tile_n; ti++)
                    row_acc[r][ti] = _mm256_setzero_ps();

            for (int b = 0; b < n_bpr; b++) {
                for (int r = 0; r < nrows; r++) {
                    const BnBlockQ8_0 *blk =
                        &blocks[(size_t)(row0 + r) * n_bpr + b];
                    __m256i w = _mm256_loadu_si256((const __m256i *)blk->qs);
#if defined(__F16C__)
                    float d_w = _cvtsh_ss(blk->d);
#else
                    float d_w = bn_fp16_to_fp32(blk->d);
#endif
                    __m256i aw = _mm256_sign_epi8(w, w);

                    for (int ti = 0; ti < tile_n; ti++) {
                        int t = t0 + ti;
                        __m256i x = _mm256_loadu_si256((const __m256i *)(
                            c->x_q + (size_t)t * cols + b * 32));
                        __m256i sx = _mm256_sign_epi8(x, w);
                        __m256i dot512 = q8_vnni_dot(aw, sx);
                        float d_x =
                            c->x_scales[(size_t)t * n_bpr + b];
                        __m256 scale = _mm256_set1_ps(d_w * d_x);
                        row_acc[r][ti] = _mm256_fmadd_ps(
                            scale,
                            _mm256_cvtepi32_ps(
                                dot512),
                            row_acc[r][ti]);
                    }
                }
            }

            for (int r = 0; r < nrows; r++)
                for (int ti = 0; ti < tile_n; ti++)
                    c->out[(size_t)(t0 + ti) * rows + row0 + r] =
                        bn_avx2_hsum_ps(row_acc[r][ti]);
        }
    }
}

#endif
