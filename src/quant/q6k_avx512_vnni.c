#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

static inline float q6k_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

static inline __m512i q6k_scale_quad_i32(int8_t s0, int8_t s1, int8_t s2, int8_t s3) {
    return _mm512_set_epi32(s3, s3, s3, s3, s2, s2, s2, s2,
                            s1, s1, s1, s1, s0, s0, s0, s0);
}

static inline __m512i q6k_join_256(__m256i lo, __m256i hi) {
    __m512i z = _mm512_castsi256_si512(lo);
    return _mm512_inserti64x4(z, hi, 1);
}

static void q6k_avx512_prepared_4row_range(BnKQuantSdotCtx *c,
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
        float row_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;

            __m512i xv[4];
            const int8_t *xb = x_q + b * BN_QK_K;
            for (int i = 0; i < 4; i++)
                xv[i] = _mm512_loadu_si512((const void *)(xb + i * 64));

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ6KPrepared *blk =
                    &blocks[(size_t)(row0 + r) * n_bpr + b];
                const int8_t *sc = blk->scales;
                __m512i sumi = _mm512_setzero_si512();

                for (int pair = 0; pair < 4; pair++) {
                    __m256i w0 = _mm256_loadu_si256(
                        (const __m256i *)(blk->qs + pair * 64));
                    __m256i w1 = _mm256_loadu_si256(
                        (const __m256i *)(blk->qs + pair * 64 + 32));
                    __m512i prod = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), q6k_join_256(w0, w1),
                        xv[pair]);
                    prod = _mm512_mullo_epi32(
                        prod, q6k_scale_quad_i32(sc[pair * 4],
                                                 sc[pair * 4 + 1],
                                                 sc[pair * 4 + 2],
                                                 sc[pair * 4 + 3]));
                    sumi = _mm512_add_epi32(sumi, prod);
                }

                int32_t corr = 0;
                for (int i = 0; i < 16; i++)
                    corr += (int32_t)sc[i] * (int32_t)bsums[i];
                int32_t dot = bn_avx512_hsum_epi32(sumi) - 32 * corr;
                row_sum[r] += blk->d * dx * (float)dot;
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sum[r];
    }
}

#define Q6K_AVX512_MATMUL_TILE_T 4

static void q6k_avx512_prepared_matmul_4row_range(
    BnKQuantMatmulCtx *c, int group_start, int group_end) {
    int rows = c->W->rows;
    int n_bpr = c->cols / BN_QK_K;
    const BnBlockQ6KPrepared *blocks =
        (const BnBlockQ6KPrepared *)c->prepared->aux;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = row0 + 4 <= rows ? 4 : rows - row0;
        for (int t0 = 0; t0 < c->n_tokens;
             t0 += Q6K_AVX512_MATMUL_TILE_T) {
            int tile_n = t0 + Q6K_AVX512_MATMUL_TILE_T <= c->n_tokens
                ? Q6K_AVX512_MATMUL_TILE_T : c->n_tokens - t0;
            float acc[4][Q6K_AVX512_MATMUL_TILE_T] = {{0}};

            for (int b = 0; b < n_bpr; b++) {
                for (int r = 0; r < nrows; r++) {
                    const BnBlockQ6KPrepared *blk =
                        &blocks[(size_t)(row0 + r) * n_bpr + b];
                    __m512i wv[4];
                    __m512i sv[4];
                    for (int pair = 0; pair < 4; pair++) {
                        __m256i w0 = _mm256_loadu_si256(
                            (const __m256i *)(blk->qs + pair * 64));
                        __m256i w1 = _mm256_loadu_si256(
                            (const __m256i *)(blk->qs + pair * 64 + 32));
                        wv[pair] = q6k_join_256(w0, w1);
                        sv[pair] = q6k_scale_quad_i32(
                            blk->scales[pair * 4],
                            blk->scales[pair * 4 + 1],
                            blk->scales[pair * 4 + 2],
                            blk->scales[pair * 4 + 3]);
                    }
                    __m256i scales16 = _mm256_cvtepi8_epi16(
                        _mm_loadu_si128((const __m128i *)blk->scales));
                    for (int ti = 0; ti < tile_n; ti++) {
                        int t = t0 + ti;
                        const int8_t *xb = c->x_q +
                            (size_t)t * c->cols + b * BN_QK_K;
                        __m512i sumi = _mm512_setzero_si512();
                        for (int pair = 0; pair < 4; pair++) {
                            __m512i prod = _mm512_dpbusd_epi32(
                                _mm512_setzero_si512(), wv[pair],
                                _mm512_loadu_si512(
                                    (const void *)(xb + pair * 64)));
                            sumi = _mm512_add_epi32(
                                sumi, _mm512_mullo_epi32(prod, sv[pair]));
                        }
                        const int16_t *bsums = c->x_bsums +
                            ((size_t)t * n_bpr + b) * 16;
                        int32_t corr = bn_avx2_hsum_epi32(
                            _mm256_madd_epi16(
                                scales16,
                                _mm256_loadu_si256(
                                    (const __m256i *)bsums)));
                        int32_t dot =
                            bn_avx512_hsum_epi32(sumi) - 32 * corr;
                        float dx = c->x_d[(size_t)t * n_bpr + b];
                        acc[r][ti] += blk->d * dx * (float)dot;
                    }
                }
            }

            for (int r = 0; r < nrows; r++)
                for (int ti = 0; ti < tile_n; ti++)
                    c->out[(size_t)(t0 + ti) * rows + row0 + r] =
                        acc[r][ti];
        }
    }
}

void bn_quant_q6k_avx512_vnni_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    if (c->prepared &&
        c->prepared->kind == BN_PREPARED_WEIGHT_Q6_K_EXPANDED &&
        c->prepared->aux &&
        c->prepared->aux_size >=
            (size_t)rows * n_bpr * sizeof(BnBlockQ6KPrepared)) {
        q6k_avx512_prepared_4row_range(c, group_start, group_end);
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
        float row_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;

            __m512i xv[4];
            const int8_t *xb = x_q + b * BN_QK_K;
            for (int i = 0; i < 4; i++)
                xv[i] = _mm512_loadu_si512((const void *)(xb + i * 64));

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ6K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                float d = q6k_fp16_to_fp32(blk->d);
                const uint8_t *ql = blk->ql;
                const uint8_t *qh = blk->qh;
                const int8_t *sc = blk->scales;

                __m512i sumi = _mm512_setzero_si512();

                for (int chunk = 0; chunk < 2; chunk++) {
                    __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql);
                    __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                    __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh);

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
                    const int8_t *sc_chunk = sc + chunk * 8;

                    __m512i p01 = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), q6k_join_256(w0, w1), xv[base / 2]);
                    __m512i p23 = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), q6k_join_256(w2, w3), xv[base / 2 + 1]);

                    p01 = _mm512_mullo_epi32(
                        p01, q6k_scale_quad_i32(sc_chunk[0], sc_chunk[1],
                                                sc_chunk[2], sc_chunk[3]));
                    p23 = _mm512_mullo_epi32(
                        p23, q6k_scale_quad_i32(sc_chunk[4], sc_chunk[5],
                                                sc_chunk[6], sc_chunk[7]));
                    sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(p01, p23));

                    ql += 64;
                    qh += 32;
                }

                int32_t corr = 0;
                for (int i = 0; i < 16; i++)
                    corr += (int32_t)sc[i] * (int32_t)bsums[i];
                int32_t dot = bn_avx512_hsum_epi32(sumi) - 32 * corr;
                row_sum[r] += d * dx * (float)dot;
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sum[r];
    }
}

void bn_quant_q6k_avx512_vnni_matmul_4row_range(void *ctx,
                                                int group_start,
                                                int group_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int n_bpr = c->cols / BN_QK_K;

    if (c->prepared &&
        c->prepared->kind == BN_PREPARED_WEIGHT_Q6_K_EXPANDED &&
        c->prepared->aux &&
        c->prepared->aux_size >=
            (size_t)c->W->rows * n_bpr * sizeof(BnBlockQ6KPrepared)) {
        q6k_avx512_prepared_matmul_4row_range(
            c, group_start, group_end);
        return;
    }
    for (int g = group_start; g < group_end; g++) {
        for (int t = 0; t < c->n_tokens; t++) {
            BnKQuantSdotCtx matvec = {
                c->out + (size_t)t * c->W->rows,
                c->W,
                c->x_q + (size_t)t * c->cols,
                c->x_d + (size_t)t * n_bpr,
                c->x_bsums + (size_t)t * n_bpr * 16,
                c->prepared
            };
            bn_quant_q6k_avx512_vnni_4row_range(&matvec, g, g + 1);
        }
    }
}

#undef Q6K_AVX512_MATMUL_TILE_T

#endif
