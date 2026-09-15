#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

static inline float q6k_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

static inline __m256i q6k_scale_pair(int8_t lo, int8_t hi) {
    return _mm256_set_epi16(hi, hi, hi, hi, hi, hi, hi, hi,
                            lo, lo, lo, lo, lo, lo, lo, lo);
}

static void q6k_avx2_prepared_4row_range(BnKQuantSdotCtx *c,
                                          int group_start,
                                          int group_end) {
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ6KPrepared *blocks =
        (const BnBlockQ6KPrepared *)c->prepared->aux;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        __m256 row_acc[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()
        };

        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;
            const __m256i q8sums =
                _mm256_loadu_si256((const __m256i *)bsums);

            __m256i xv[8];
            const int8_t *xb = x_q + b * BN_QK_K;
            for (int i = 0; i < 8; i++)
                xv[i] = _mm256_loadu_si256((const __m256i *)(xb + i * 32));

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ6KPrepared *blk =
                    &blocks[(size_t)(row0 + r) * n_bpr + b];
                const int8_t *sc = blk->scales;
                const __m128i sc128 = _mm_loadu_si128((const __m128i *)sc);
                const __m256i sc16 = _mm256_cvtepi8_epi16(sc128);
                const __m256i offset =
                    _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc16), 5);
                __m256i sumi = _mm256_setzero_si256();

                for (int p = 0; p < 8; p++) {
                    __m256i wv = _mm256_loadu_si256(
                        (const __m256i *)(blk->qs + p * 32));
                    __m256i prod = _mm256_maddubs_epi16(wv, xv[p]);
                    prod = _mm256_madd_epi16(
                        q6k_scale_pair(sc[p * 2], sc[p * 2 + 1]), prod);
                    sumi = _mm256_add_epi32(sumi, prod);
                }

                sumi = _mm256_sub_epi32(sumi, offset);
                row_acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi),
                                             _mm256_set1_ps(blk->d * dx),
                                             row_acc[r]);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = bn_avx2_hsum_ps(row_acc[r]);
    }
}

// 4-row Q6_K SDOT matvec: process 4 output rows at once, unpacking weights
// per-row but loading x_q once. Amortizes activation vector memory read 4x.

void bn_quant_q6k_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    if (c->prepared &&
        c->prepared->kind == BN_PREPARED_WEIGHT_Q6_K_EXPANDED &&
        c->prepared->aux &&
        c->prepared->aux_size >=
            (size_t)rows * n_bpr * sizeof(BnBlockQ6KPrepared)) {
        q6k_avx2_prepared_4row_range(c, group_start, group_end);
        return;
    }
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i mask_03  = _mm256_set1_epi8(0x03);
    const __m256i mask_0c  = _mm256_set1_epi8(0x0C);
    const __m256i mask_30  = _mm256_set1_epi8(0x30);
    const __m256i mask_c0  = _mm256_set1_epi8((char)0xC0);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        __m256 row_acc[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()
        };

#if defined(__GNUC__) || defined(__clang__)
        #pragma GCC unroll 8
#endif
        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);

            /* Load x_q block ONCE for all rows (2 chunks x 128 bytes = 256 bytes) */
            __m256i xv[8];
            const int8_t *xb = x_q + b * BN_QK_K;
            for (int i = 0; i < 8; i++)
                xv[i] = _mm256_loadu_si256((const __m256i *)(xb + i * 32));

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ6K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                float d = q6k_fp16_to_fp32(blk->d);
                const uint8_t *ql = blk->ql;
                const uint8_t *qh = blk->qh;
                const int8_t  *sc = blk->scales;

                const __m128i sc128 = _mm_loadu_si128((const __m128i *)sc);
                const __m256i sc16 = _mm256_cvtepi8_epi16(sc128);
                const __m256i offset = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc16), 5);
                __m256i sumi = _mm256_setzero_si256();

                for (int chunk = 0; chunk < 2; chunk++) {
                    __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql);
                    __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                    __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh);

                    /* Unpack 4 unsigned weight vectors (0..63).
                     * Mask after slli_epi16 with 0x30 to prevent 16-bit
                     * cross-byte contamination. */
                    __m256i w0 = _mm256_or_si256(
                        _mm256_and_si256(ql0, mask_lo4),
                        _mm256_slli_epi16(_mm256_and_si256(qh0, mask_03), 4));
                    __m256i w1 = _mm256_or_si256(
                        _mm256_and_si256(ql1, mask_lo4),
                        _mm256_slli_epi16(_mm256_and_si256(qh0, mask_0c), 2));
                    __m256i w2 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4),
                        _mm256_and_si256(qh0, mask_30));
                    __m256i w3 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4),
                        _mm256_srli_epi16(_mm256_and_si256(qh0, mask_c0), 2));

                    int base = chunk * 4;

                    __m256i p0 = _mm256_maddubs_epi16(w0, xv[base + 0]);
                    __m256i p1 = _mm256_maddubs_epi16(w1, xv[base + 1]);
                    __m256i p2 = _mm256_maddubs_epi16(w2, xv[base + 2]);
                    __m256i p3 = _mm256_maddubs_epi16(w3, xv[base + 3]);

                    const int8_t *sc_chunk = sc;
                    p0 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[0], sc_chunk[1]), p0);
                    p1 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[2], sc_chunk[3]), p1);
                    p2 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[4], sc_chunk[5]), p2);
                    p3 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[6], sc_chunk[7]), p3);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p2, p3));

                    ql += 64; qh += 32; sc += 8;
                }

                sumi = _mm256_sub_epi32(sumi, offset);
                row_acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi),
                                             _mm256_set1_ps(d * dx), row_acc[r]);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = bn_avx2_hsum_ps(row_acc[r]);
    }
}

#define Q6K_MATMUL_4ROW_TILE_T 16

static void q6k_avx2_prepared_matmul_4row_range(BnKQuantMatmulCtx *c,
                                                 int group_start,
                                                 int group_end) {
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ6KPrepared *blocks =
        (const BnBlockQ6KPrepared *)c->prepared->aux;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        for (int t0 = 0; t0 < n_tokens; t0 += Q6K_MATMUL_4ROW_TILE_T) {
            int tile_n = t0 + Q6K_MATMUL_4ROW_TILE_T <= n_tokens
                ? Q6K_MATMUL_4ROW_TILE_T : n_tokens - t0;
            __m256 row_acc[4][Q6K_MATMUL_4ROW_TILE_T];

            for (int r = 0; r < 4; r++)
                for (int ti = 0; ti < Q6K_MATMUL_4ROW_TILE_T; ti++)
                    row_acc[r][ti] = _mm256_setzero_ps();

            for (int b = 0; b < n_bpr; b++) {
                __m256i xv[Q6K_MATMUL_4ROW_TILE_T][8];
                __m256i q8sums[Q6K_MATMUL_4ROW_TILE_T];
                float dx[Q6K_MATMUL_4ROW_TILE_T];

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb =
                        c->x_q + (size_t)t * cols + b * BN_QK_K;
                    dx[ti] = c->x_d[(size_t)t * n_bpr + b];
                    const int16_t *bsums =
                        c->x_bsums + ((size_t)t * n_bpr + b) * 16;
                    q8sums[ti] =
                        _mm256_loadu_si256((const __m256i *)bsums);
                    for (int i = 0; i < 8; i++)
                        xv[ti][i] = _mm256_loadu_si256(
                            (const __m256i *)(xb + i * 32));
                }

                for (int r = 0; r < nrows; r++) {
                    const BnBlockQ6KPrepared *blk =
                        &blocks[(size_t)(row0 + r) * n_bpr + b];
                    const int8_t *sc = blk->scales;
                    const __m128i sc128 =
                        _mm_loadu_si128((const __m128i *)sc);
                    const __m256i sc16 = _mm256_cvtepi8_epi16(sc128);
                    __m256i sp[8];

                    for (int p = 0; p < 8; p++)
                        sp[p] = q6k_scale_pair(sc[p * 2], sc[p * 2 + 1]);

                    for (int ti = 0; ti < tile_n; ti++) {
                        __m256i offset = _mm256_slli_epi32(
                            _mm256_madd_epi16(q8sums[ti], sc16), 5);
                        __m256i sumi = _mm256_setzero_si256();

                        for (int p = 0; p < 8; p++) {
                            __m256i wv = _mm256_loadu_si256(
                                (const __m256i *)(blk->qs + p * 32));
                            __m256i prod =
                                _mm256_maddubs_epi16(wv, xv[ti][p]);
                            prod = _mm256_madd_epi16(sp[p], prod);
                            sumi = _mm256_add_epi32(sumi, prod);
                        }

                        sumi = _mm256_sub_epi32(sumi, offset);
                        row_acc[r][ti] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(sumi),
                            _mm256_set1_ps(dx[ti] * blk->d),
                            row_acc[r][ti]);
                    }
                }
            }

            for (int r = 0; r < nrows; r++) {
                for (int ti = 0; ti < tile_n; ti++) {
                    c->out[(size_t)(t0 + ti) * rows + row0 + r] =
                        bn_avx2_hsum_ps(row_acc[r][ti]);
                }
            }
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
static inline __m512i q6k_join_rows(const void *a, const void *b) {
    return _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_loadu_si256(a)),
        _mm256_loadu_si256(b), 1);
}

/* Each 256-bit half retains one row's eight accumulation lanes. Pairing
 * rows widens integer work without changing block FMA or final reduction
 * order. Four tokens share weight unpacking and each input feeds both rows. */
static void q6k_avx512_matmul_2row4(BnKQuantMatmulCtx *c,
                                   int group_start, int group_end) {
    int rows = c->W->rows, cols = c->cols, nb = cols / BN_QK_K;
    const BnBlockQ6K *weights = c->W->data;
    const __m512i m15 = _mm512_set1_epi8(15);
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m12 = _mm512_set1_epi8(12);
    const __m512i m48 = _mm512_set1_epi8(48);
    const __m512i m192 = _mm512_set1_epi8((char)192);
    for (int row = group_start * 4; row < group_end * 4; row += 2) {
        for (int t0 = 0; t0 < c->n_tokens; t0 += 4) {
            __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                            _mm512_setzero_ps(), _mm512_setzero_ps()};
            for (int b = 0; b < nb; b++) {
                const BnBlockQ6K *w0 = weights + (size_t)row * nb + b;
                const BnBlockQ6K *w1 = w0 + nb;
                __m512i sums[4] = {
                    _mm512_setzero_si512(), _mm512_setzero_si512(),
                    _mm512_setzero_si512(), _mm512_setzero_si512()
                };
                for (int chunk = 0; chunk < 2; chunk++) {
                    __m512i l0 = q6k_join_rows(w0->ql + chunk * 64,
                                              w1->ql + chunk * 64);
                    __m512i l1 = q6k_join_rows(w0->ql + chunk * 64 + 32,
                                              w1->ql + chunk * 64 + 32);
                    __m512i h = q6k_join_rows(w0->qh + chunk * 32,
                                             w1->qh + chunk * 32);
                    __m512i v[4] = {
                        _mm512_or_si512(_mm512_and_si512(l0, m15),
                            _mm512_slli_epi16(_mm512_and_si512(h, m3), 4)),
                        _mm512_or_si512(_mm512_and_si512(l1, m15),
                            _mm512_slli_epi16(_mm512_and_si512(h, m12), 2)),
                        _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(l0, 4), m15),
                            _mm512_and_si512(h, m48)),
                        _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(l1, 4), m15),
                            _mm512_srli_epi16(_mm512_and_si512(h, m192), 2))
                    };
                    for (int p = 0; p < 4; p++) {
                        int s = chunk * 8 + p * 2;
                        __m512i scale = _mm512_inserti64x4(
                            _mm512_castsi256_si512(q6k_scale_pair(w0->scales[s], w0->scales[s + 1])),
                            q6k_scale_pair(w1->scales[s], w1->scales[s + 1]), 1);
                        for (int t = 0; t < 4; t++) {
                            const int8_t *x = c->x_q + (size_t)(t0 + t) * cols +
                                b * BN_QK_K + chunk * 128 + p * 32;
                            __m512i xv = _mm512_broadcast_i64x4(
                                _mm256_loadu_si256((const __m256i *)x));
                            sums[t] = _mm512_add_epi32(sums[t],
                                _mm512_madd_epi16(scale, _mm512_maddubs_epi16(v[p], xv)));
                        }
                    }
                }
                __m256i sc = _mm256_set_m128i(
                    _mm_loadu_si128((const __m128i *)w1->scales),
                    _mm_loadu_si128((const __m128i *)w0->scales));
                __m512i sc16 = _mm512_cvtepi8_epi16(sc);
                float d0 = q6k_fp16_to_fp32(w0->d);
                float d1 = q6k_fp16_to_fp32(w1->d);
                for (int t = 0; t < 4; t++) {
                    size_t idx = (size_t)(t0 + t) * nb + b;
                    __m512i bs = _mm512_broadcast_i64x4(
                        _mm256_loadu_si256((const __m256i *)(c->x_bsums + idx * 16)));
                    __m512i sum = _mm512_sub_epi32(sums[t],
                        _mm512_slli_epi32(_mm512_madd_epi16(bs, sc16), 5));
                    __m512 d = _mm512_insertf32x8(
                        _mm512_castps256_ps512(_mm256_set1_ps(d0 * c->x_d[idx])),
                        _mm256_set1_ps(d1 * c->x_d[idx]), 1);
                    acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), d, acc[t]);
                }
            }
            for (int t = 0; t < 4; t++) {
                c->out[(size_t)(t0 + t) * rows + row] =
                    bn_avx2_hsum_ps(_mm512_castps512_ps256(acc[t]));
                c->out[(size_t)(t0 + t) * rows + row + 1] =
                    bn_avx2_hsum_ps(_mm512_extractf32x8_ps(acc[t], 1));
            }
        }
    }
}
#endif

void bn_quant_q6k_avx2_sdot_matmul_4row_range(void *ctx,
                                               int group_start,
                                               int group_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    if (c->prepared &&
        c->prepared->kind == BN_PREPARED_WEIGHT_Q6_K_EXPANDED &&
        c->prepared->aux &&
        c->prepared->aux_size >=
            (size_t)rows * n_bpr * sizeof(BnBlockQ6KPrepared)) {
        q6k_avx2_prepared_matmul_4row_range(c, group_start, group_end);
        return;
    }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    if (rows % 4 == 0 && n_tokens % 4 == 0) {
        q6k_avx512_matmul_2row4(c, group_start, group_end);
        return;
    }
#endif
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;

    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i mask_03  = _mm256_set1_epi8(0x03);
    const __m256i mask_0c  = _mm256_set1_epi8(0x0C);
    const __m256i mask_30  = _mm256_set1_epi8(0x30);
    const __m256i mask_c0  = _mm256_set1_epi8((char)0xC0);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        for (int t0 = 0; t0 < n_tokens; t0 += Q6K_MATMUL_4ROW_TILE_T) {
            int tile_n = t0 + Q6K_MATMUL_4ROW_TILE_T <= n_tokens
                ? Q6K_MATMUL_4ROW_TILE_T : n_tokens - t0;
            __m256 row_acc[4][Q6K_MATMUL_4ROW_TILE_T];

            for (int r = 0; r < 4; r++)
                for (int ti = 0; ti < Q6K_MATMUL_4ROW_TILE_T; ti++)
                    row_acc[r][ti] = _mm256_setzero_ps();

            for (int b = 0; b < n_bpr; b++) {
                __m256i xv[Q6K_MATMUL_4ROW_TILE_T][8];
                float dx[Q6K_MATMUL_4ROW_TILE_T];
                const int16_t *bsums[Q6K_MATMUL_4ROW_TILE_T];

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb =
                        c->x_q + (size_t)t * cols + b * BN_QK_K;
                    dx[ti] = c->x_d[(size_t)t * n_bpr + b];
                    bsums[ti] =
                        c->x_bsums + ((size_t)t * n_bpr + b) * 16;
                    for (int i = 0; i < 8; i++)
                        xv[ti][i] = _mm256_loadu_si256(
                            (const __m256i *)(xb + i * 32));
                }

                for (int r = 0; r < nrows; r++) {
                    const BnBlockQ6K *blk =
                        &blocks[(size_t)(row0 + r) * n_bpr + b];
                    float d = q6k_fp16_to_fp32(blk->d);
                    const uint8_t *ql = blk->ql;
                    const uint8_t *qh = blk->qh;
                    const int8_t *sc = blk->scales;

                    __m256i wv[8];
                    __m256i sp[8];
                    for (int chunk = 0; chunk < 2; chunk++) {
                        __m256i ql0 =
                            _mm256_loadu_si256((const __m256i *)ql);
                        __m256i ql1 =
                            _mm256_loadu_si256((const __m256i *)(ql + 32));
                        __m256i qh0 =
                            _mm256_loadu_si256((const __m256i *)qh);
                        int base = chunk * 4;

                        wv[base + 0] = _mm256_or_si256(
                            _mm256_and_si256(ql0, mask_lo4),
                            _mm256_slli_epi16(
                                _mm256_and_si256(qh0, mask_03), 4));
                        wv[base + 1] = _mm256_or_si256(
                            _mm256_and_si256(ql1, mask_lo4),
                            _mm256_slli_epi16(
                                _mm256_and_si256(qh0, mask_0c), 2));
                        wv[base + 2] = _mm256_or_si256(
                            _mm256_and_si256(
                                _mm256_srli_epi16(ql0, 4), mask_lo4),
                            _mm256_and_si256(qh0, mask_30));
                        wv[base + 3] = _mm256_or_si256(
                            _mm256_and_si256(
                                _mm256_srli_epi16(ql1, 4), mask_lo4),
                            _mm256_srli_epi16(
                                _mm256_and_si256(qh0, mask_c0), 2));

                        const int8_t *sc_chunk = sc + chunk * 8;
                        sp[base + 0] =
                            q6k_scale_pair(sc_chunk[0], sc_chunk[1]);
                        sp[base + 1] =
                            q6k_scale_pair(sc_chunk[2], sc_chunk[3]);
                        sp[base + 2] =
                            q6k_scale_pair(sc_chunk[4], sc_chunk[5]);
                        sp[base + 3] =
                            q6k_scale_pair(sc_chunk[6], sc_chunk[7]);

                        ql += 64;
                        qh += 32;
                    }

                    const __m128i sc128 =
                        _mm_loadu_si128((const __m128i *)sc);
                    const __m256i sc16 = _mm256_cvtepi8_epi16(sc128);

                    for (int ti = 0; ti < tile_n; ti++) {
                        __m256i q8sums = _mm256_loadu_si256(
                            (const __m256i *)bsums[ti]);
                        __m256i offset = _mm256_slli_epi32(
                            _mm256_madd_epi16(q8sums, sc16), 5);

                        __m256i sumi = _mm256_setzero_si256();
                        for (int p = 0; p < 8; p++) {
                            __m256i prod =
                                _mm256_maddubs_epi16(wv[p], xv[ti][p]);
                            prod = _mm256_madd_epi16(sp[p], prod);
                            sumi = _mm256_add_epi32(sumi, prod);
                        }
                        sumi = _mm256_sub_epi32(sumi, offset);
                        row_acc[r][ti] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(sumi),
                            _mm256_set1_ps(dx[ti] * d),
                            row_acc[r][ti]);
                    }
                }
            }

            for (int r = 0; r < nrows; r++) {
                for (int ti = 0; ti < tile_n; ti++) {
                    c->out[(size_t)(t0 + ti) * rows + row0 + r] =
                        bn_avx2_hsum_ps(row_acc[r][ti]);
                }
            }
        }
    }
}
