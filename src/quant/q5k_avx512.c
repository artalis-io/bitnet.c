#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512BW__)

static inline __m128i q5k512_extract_hb(const uint8_t *qh, int l_offset, int bit_pos) {
    __m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l_offset));
    __m128i mask = _mm_set1_epi8((char)(1 << bit_pos));
    __m128i tested = _mm_and_si128(qh_vec, mask);
    __m128i is_zero = _mm_cmpeq_epi8(tested, _mm_setzero_si128());
    return _mm_andnot_si128(is_zero, _mm_set1_epi8(0x10));
}

void bn_quant_q5k_avx512_4row_range(void *ctx, int group_start, int group_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo = _mm_set1_epi8(0x0F);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;
        float row_sums[4] = {0};

        for (int b = 0; b < n_bpr; b++) {
            const float *xb = x + b * BN_QK_K;

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ5K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);
                float d = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qs = blk->qs;
                const uint8_t *qh = blk->qh;
                __m512 acc = _mm512_setzero_ps();

                for (int j = 0; j < BN_QK_K; j += 64) {
                    uint8_t sc, m;
                    int sub = j / 32;
                    int group = j / 64;
                    __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                    __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));

                    int bit_lo = group * 2;
                    int bit_hi = group * 2 + 1;
                    __m128i hb0 = q5k512_extract_hb(qh, 0, bit_lo);
                    __m128i hb1 = q5k512_extract_hb(qh, 16, bit_lo);
                    __m128i hb2 = q5k512_extract_hb(qh, 0, bit_hi);
                    __m128i hb3 = q5k512_extract_hb(qh, 16, bit_hi);

#define Q5K_AVX512_ACC_16(w128, xp) do { \
    __m512 wf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w128)); \
    wf = _mm512_fmsub_ps(wf, vds, vdm); \
    acc = _mm512_fmadd_ps(wf, _mm512_loadu_ps(xp), acc); \
} while (0)

                    bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                    __m512 vds = _mm512_set1_ps(d * sc);
                    __m512 vdm = _mm512_set1_ps(dmin * m);
                    __m128i w0 = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                    __m128i w1 = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);
                    Q5K_AVX512_ACC_16(w0, xb + j);
                    Q5K_AVX512_ACC_16(w1, xb + j + 16);

                    bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                    vds = _mm512_set1_ps(d * sc);
                    vdm = _mm512_set1_ps(dmin * m);
                    w0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                    w1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);
                    Q5K_AVX512_ACC_16(w0, xb + j + 32);
                    Q5K_AVX512_ACC_16(w1, xb + j + 48);

#undef Q5K_AVX512_ACC_16
                    qs += 32;
                }

                row_sums[r] += bn_avx512_hsum_ps(acc);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sums[r];
    }
}

#if defined(__AVX512VNNI__)
static inline __m512i q5k512_join_4x128(__m128i a, __m128i b,
                                        __m128i c, __m128i d) {
    __m512i z = _mm512_castsi128_si512(a);
    z = _mm512_inserti32x4(z, b, 1);
    z = _mm512_inserti32x4(z, c, 2);
    return _mm512_inserti32x4(z, d, 3);
}

static inline __m512i q5k512_scale_pair_i32(uint8_t lo, uint8_t hi) {
    return _mm512_set_epi32(hi, hi, hi, hi, hi, hi, hi, hi,
                            lo, lo, lo, lo, lo, lo, lo, lo);
}

void bn_quant_q5k_avx512_vnni_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        __m512 row_acc[4] = {
            _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps()
        };
        float row_corr[4] = {0};

        for (int b = 0; b < n_bpr; b++) {
            float dx = x_d[b];
            const int16_t *bsums = x_bsums + b * 16;
            const int8_t *xb = x_q + b * BN_QK_K;

            __m512i xv[4];
            for (int p = 0; p < 4; p++)
                xv[p] = _mm512_loadu_si512((const void *)(xb + p * 64));

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ5K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);
                float d = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);

                uint32_t utmp[3];
                memcpy(utmp, blk->scales, 12);
                uint32_t m_lo = utmp[1] & kmask1;
                uint32_t m_hi = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[0] &= kmask1;
                const uint8_t *sc = (const uint8_t *)utmp;
                uint8_t mins[8];
                memcpy(mins, &m_lo, 4);
                memcpy(mins + 4, &m_hi, 4);

                __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
                __m128i bsl = _mm256_castsi256_si128(q8sums);
                __m128i bsh = _mm256_extracti128_si256(q8sums, 1);
                __m128i bs_paired = _mm_hadd_epi16(bsl, bsh);
                __m128i mins_v = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mins));
                __m128i corr128 = _mm_madd_epi16(mins_v, bs_paired);
                __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
                __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
                int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

                __m512i sumi_v = _mm512_setzero_si512();
                const uint8_t *qs = blk->qs;
                const uint8_t *qh = blk->qh;
                for (int p = 0; p < 4; p++) {
                    __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                    __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));
                    int bit_lo = p * 2;
                    int bit_hi = p * 2 + 1;

                    __m128i hb0 = q5k512_extract_hb(qh, 0, bit_lo);
                    __m128i hb1 = q5k512_extract_hb(qh, 16, bit_lo);
                    __m128i hb2 = q5k512_extract_hb(qh, 0, bit_hi);
                    __m128i hb3 = q5k512_extract_hb(qh, 16, bit_hi);

                    __m128i w0 = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                    __m128i w1 = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);
                    __m128i w2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                    __m128i w3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);

                    __m512i prod = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(),
                        q5k512_join_4x128(w0, w1, w2, w3),
                        xv[p]);
                    prod = _mm512_mullo_epi32(
                        prod, q5k512_scale_pair_i32(sc[2 * p], sc[2 * p + 1]));
                    sumi_v = _mm512_add_epi32(sumi_v, prod);
                    qs += 32;
                }

                row_acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sumi_v),
                                             _mm512_set1_ps(dx * d), row_acc[r]);
                row_corr[r] += dx * dmin * (float)bsum_corr;
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = bn_avx512_hsum_ps(row_acc[r]) - row_corr[r];
    }
}

#define Q5K_AVX512_TILE_T 4

void bn_quant_q5k_avx512_vnni_matmul_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;

    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        for (int t0 = 0; t0 < n_tokens; t0 += Q5K_AVX512_TILE_T) {
            int tile_n = t0 + Q5K_AVX512_TILE_T <= n_tokens
                ? Q5K_AVX512_TILE_T : n_tokens - t0;
            float acc[4][Q5K_AVX512_TILE_T] = {{0}};

            for (int b = 0; b < n_bpr; b++) {
                __m512i xv[Q5K_AVX512_TILE_T][4];
                for (int ti = 0; ti < tile_n; ti++) {
                    const int8_t *xb = c->x_q + (size_t)(t0 + ti) * cols + b * BN_QK_K;
                    for (int p = 0; p < 4; p++)
                        xv[ti][p] = _mm512_loadu_si512((const void *)(xb + p * 64));
                }

                for (int r = 0; r < nrows; r++) {
                    const BnBlockQ5K *blk = &blocks[(size_t)(row0 + r) * n_bpr + b];
                    float d = bn_fp16_to_fp32(blk->d);
                    float dmin = bn_fp16_to_fp32(blk->dmin);

                    uint32_t utmp[3];
                    memcpy(utmp, blk->scales, 12);
                    uint32_t m_lo = utmp[1] & kmask1;
                    uint32_t m_hi = ((utmp[2] >> 4) & kmask2) |
                                    (((utmp[1] >> 6) & kmask3) << 4);
                    utmp[1] = (utmp[2] & kmask2) |
                              (((utmp[0] >> 6) & kmask3) << 4);
                    utmp[0] &= kmask1;
                    const uint8_t *sc = (const uint8_t *)utmp;
                    uint8_t mins[8];
                    memcpy(mins, &m_lo, 4);
                    memcpy(mins + 4, &m_hi, 4);
                    __m128i mins_v = _mm_cvtepu8_epi16(
                        _mm_loadl_epi64((const __m128i *)mins));

                    __m512i wv[4];
                    const uint8_t *qs = blk->qs;
                    const uint8_t *qh = blk->qh;
                    for (int p = 0; p < 4; p++) {
                        __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                        __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));
                        int bit_lo = p * 2;
                        int bit_hi = p * 2 + 1;
                        __m128i hb0 = q5k512_extract_hb(qh, 0, bit_lo);
                        __m128i hb1 = q5k512_extract_hb(qh, 16, bit_lo);
                        __m128i hb2 = q5k512_extract_hb(qh, 0, bit_hi);
                        __m128i hb3 = q5k512_extract_hb(qh, 16, bit_hi);
                        __m128i w0 = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                        __m128i w1 = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);
                        __m128i w2 = _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                        __m128i w3 = _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);
                        wv[p] = q5k512_join_4x128(w0, w1, w2, w3);
                        qs += 32;
                    }

                    __m512i sv[4];
                    for (int p = 0; p < 4; p++)
                        sv[p] = q5k512_scale_pair_i32(sc[2 * p], sc[2 * p + 1]);

                    for (int ti = 0; ti < tile_n; ti++) {
                        int t = t0 + ti;
                        float dx = c->x_d[(size_t)t * n_bpr + b];
                        const int16_t *bsums =
                            c->x_bsums + ((size_t)t * n_bpr + b) * 16;

                        __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
                        __m128i bsl = _mm256_castsi256_si128(q8sums);
                        __m128i bsh = _mm256_extracti128_si256(q8sums, 1);
                        __m128i bs_paired = _mm_hadd_epi16(bsl, bsh);
                        __m128i corr128 = _mm_madd_epi16(mins_v, bs_paired);
                        __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
                        __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
                        int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

                        __m512i sumi = _mm512_setzero_si512();
                        for (int p = 0; p < 4; p++) {
                            __m512i prod = _mm512_dpbusd_epi32(
                                _mm512_setzero_si512(), wv[p], xv[ti][p]);
                            prod = _mm512_mullo_epi32(prod, sv[p]);
                            sumi = _mm512_add_epi32(sumi, prod);
                        }
                        int32_t dot = bn_avx512_hsum_epi32(sumi);
                        acc[r][ti] += dx * (d * (float)dot -
                                            dmin * (float)bsum_corr);
                    }
                }
            }

            for (int r = 0; r < nrows; r++)
                for (int ti = 0; ti < tile_n; ti++)
                    c->out[(size_t)(t0 + ti) * rows + row0 + r] += acc[r][ti];
        }
    }
}
#endif

#endif
