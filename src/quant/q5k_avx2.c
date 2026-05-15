#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <immintrin.h>

// Extract high bit `bit_pos` from 16 consecutive qh bytes starting at l_offset.
// qh[l] stores high bits for position l across all groups.
// Returns 16 bytes, each 0x00 or 0x10.
static inline __m128i q5k_extract_hb(const uint8_t *qh, int l_offset, int bit_pos) {
    __m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l_offset));
    __m128i mask = _mm_set1_epi8((char)(1 << bit_pos));
    __m128i tested = _mm_and_si128(qh_vec, mask);
    __m128i is_zero = _mm_cmpeq_epi8(tested, _mm_setzero_si128());
    return _mm_andnot_si128(is_zero, _mm_set1_epi8(0x10));
}

void bn_quant_q5k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo = _mm_set1_epi8(0xF);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const float *xb = x + b * BN_QK_K;

            const uint8_t *qh = blk->qh;
            __m256 acc = _mm256_setzero_ps();

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;
                int group = j / 64;
                __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));

                int bit_lo = group * 2;      // bits 0,2,4,6
                int bit_hi = group * 2 + 1;  // bits 1,3,5,7
                __m128i hb0 = q5k_extract_hb(qh, 0,  bit_lo);   // l=0..15, first half
                __m128i hb1 = q5k_extract_hb(qh, 16, bit_lo);   // l=16..31, first half
                __m128i hb2 = q5k_extract_hb(qh, 0,  bit_hi);   // l=0..15, second half
                __m128i hb3 = q5k_extract_hb(qh, 16, bit_hi);   // l=16..31, second half

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                __m256 vds = _mm256_set1_ps(d * sc);
                __m256 vdm = _mm256_set1_ps(dmin * m);
                __m128i w0 = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                __m128i w1 = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);

                #define Q5K_AVX2_ACC_16(w128, xp) do { \
                    __m256 wf_lo = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds, vdm); \
                    __m256 wf_hi = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds, vdm); \
                    acc = _mm256_fmadd_ps(wf_lo, _mm256_loadu_ps(xp), acc); \
                    acc = _mm256_fmadd_ps(wf_hi, _mm256_loadu_ps(xp + 8), acc); \
                } while(0)

                Q5K_AVX2_ACC_16(w0, xb + j);
                Q5K_AVX2_ACC_16(w1, xb + j + 16);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                vds = _mm256_set1_ps(d * sc);
                vdm = _mm256_set1_ps(dmin * m);
                w0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                w1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);
                Q5K_AVX2_ACC_16(w0, xb + j + 32);
                Q5K_AVX2_ACC_16(w1, xb + j + 48);

                #undef Q5K_AVX2_ACC_16

                qs += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q5k_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo = _mm_set1_epi8(0xF);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;
        float row_sums[4] = {0};

        for (int b = 0; b < n_blocks_per_row; b++) {
            const float *xb = x + b * BN_QK_K;

            for (int r = 0; r < nrows; r++) {
                const BnBlockQ5K *blk = &blocks[(size_t)(row0 + r) * n_blocks_per_row + b];
                _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);
                float d    = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qs = blk->qs;
                const uint8_t *qh = blk->qh;
                __m256 acc = _mm256_setzero_ps();

                for (int j = 0; j < BN_QK_K; j += 64) {
                    uint8_t sc, m;
                    int sub = j / 32;
                    int group = j / 64;
                    __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                    __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));

                    int bit_lo = group * 2;
                    int bit_hi = group * 2 + 1;
                    __m128i hb0 = q5k_extract_hb(qh, 0,  bit_lo);
                    __m128i hb1 = q5k_extract_hb(qh, 16, bit_lo);
                    __m128i hb2 = q5k_extract_hb(qh, 0,  bit_hi);
                    __m128i hb3 = q5k_extract_hb(qh, 16, bit_hi);

                    bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                    __m256 vds = _mm256_set1_ps(d * sc);
                    __m256 vdm = _mm256_set1_ps(dmin * m);
                    __m128i w0 = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                    __m128i w1 = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);

                    #define Q5K_AVX2_ACC_16(w128, xp) do { \
                        __m256 wf_lo = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds, vdm); \
                        __m256 wf_hi = _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds, vdm); \
                        acc = _mm256_fmadd_ps(wf_lo, _mm256_loadu_ps(xp), acc); \
                        acc = _mm256_fmadd_ps(wf_hi, _mm256_loadu_ps(xp + 8), acc); \
                    } while(0)

                    Q5K_AVX2_ACC_16(w0, xb + j);
                    Q5K_AVX2_ACC_16(w1, xb + j + 16);

                    bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                    vds = _mm256_set1_ps(d * sc);
                    vdm = _mm256_set1_ps(dmin * m);
                    w0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                    w1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);
                    Q5K_AVX2_ACC_16(w0, xb + j + 32);
                    Q5K_AVX2_ACC_16(w1, xb + j + 48);

                    #undef Q5K_AVX2_ACC_16

                    qs += 32;
                }

                row_sums[r] += bn_avx2_hsum_ps(acc);
            }
        }

        for (int r = 0; r < nrows; r++)
            c->out[row0 + r] = row_sums[r];
    }
}

#define Q5K_TILE_T 8

void bn_quant_q5k_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ5KMatmulCtx *c = (BnQ5KMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;

    const __m128i mask_lo = _mm_set1_epi8(0xF);

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q5K_TILE_T) {
            int tile_n = t0 + Q5K_TILE_T <= n_tokens ? Q5K_TILE_T : n_tokens - t0;
            float sums[Q5K_TILE_T] = {0};

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ5K *blk = &blocks[(size_t)row * n_bpr + b];
                float d = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qh = blk->qh;

                __m128i wv[16] = {0};
                __m256 vds[16] = {0}, vdm[16] = {0};
                const uint8_t *qs = blk->qs;
                for (int j = 0; j < BN_QK_K; j += 64) {
                    uint8_t sc, m;
                    int sub = j / 32;
                    int group = j / 64;
                    int bit_lo = group * 2;
                    int bit_hi = group * 2 + 1;

                    __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                    __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));
                    __m128i hb0 = q5k_extract_hb(qh, 0, bit_lo);
                    __m128i hb1 = q5k_extract_hb(qh, 16, bit_lo);
                    __m128i hb2 = q5k_extract_hb(qh, 0, bit_hi);
                    __m128i hb3 = q5k_extract_hb(qh, 16, bit_hi);

                    int base = j / 16;
                    bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                    vds[base] = _mm256_set1_ps(d * sc);
                    vdm[base] = _mm256_set1_ps(dmin * m);
                    vds[base + 1] = vds[base];
                    vdm[base + 1] = vdm[base];
                    wv[base] = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                    wv[base + 1] = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);

                    bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                    vds[base + 2] = _mm256_set1_ps(d * sc);
                    vdm[base + 2] = _mm256_set1_ps(dmin * m);
                    vds[base + 3] = vds[base + 2];
                    vdm[base + 3] = vdm[base + 2];
                    wv[base + 2] = _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                    wv[base + 3] = _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);

                    qs += 32;
                }

                for (int ti = 0; ti < tile_n; ti++) {
                    const float *xb = c->x + (size_t)(t0 + ti) * cols + b * BN_QK_K;
                    __m256 acc = _mm256_setzero_ps();

#define Q5K_MATMUL_ACC(idx, xp) do { \
                        __m256 wf_lo = _mm256_fmsub_ps( \
                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wv[(idx)])), \
                            vds[(idx)], vdm[(idx)]); \
                        __m256 wf_hi = _mm256_fmsub_ps( \
                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(wv[(idx)], 8))), \
                            vds[(idx)], vdm[(idx)]); \
                        acc = _mm256_fmadd_ps(wf_lo, _mm256_loadu_ps((xp)), acc); \
                        acc = _mm256_fmadd_ps(wf_hi, _mm256_loadu_ps((xp) + 8), acc); \
                    } while (0)

                    Q5K_MATMUL_ACC(0, xb);
                    Q5K_MATMUL_ACC(1, xb + 16);
                    Q5K_MATMUL_ACC(2, xb + 32);
                    Q5K_MATMUL_ACC(3, xb + 48);
                    Q5K_MATMUL_ACC(4, xb + 64);
                    Q5K_MATMUL_ACC(5, xb + 80);
                    Q5K_MATMUL_ACC(6, xb + 96);
                    Q5K_MATMUL_ACC(7, xb + 112);
                    Q5K_MATMUL_ACC(8, xb + 128);
                    Q5K_MATMUL_ACC(9, xb + 144);
                    Q5K_MATMUL_ACC(10, xb + 160);
                    Q5K_MATMUL_ACC(11, xb + 176);
                    Q5K_MATMUL_ACC(12, xb + 192);
                    Q5K_MATMUL_ACC(13, xb + 208);
                    Q5K_MATMUL_ACC(14, xb + 224);
                    Q5K_MATMUL_ACC(15, xb + 240);
#undef Q5K_MATMUL_ACC

                    sums[ti] += bn_avx2_hsum_ps(acc);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += sums[ti];
        }
    }
}

#define Q5K_SDOT_TILE_T 8

void bn_quant_q5k_avx2_sdot_matmul_range(void *ctx, int row_start, int row_end) {
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

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q5K_SDOT_TILE_T) {
            int tile_n = t0 + Q5K_SDOT_TILE_T <= n_tokens
                ? Q5K_SDOT_TILE_T : n_tokens - t0;
            float acc[Q5K_SDOT_TILE_T] = {0};

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ5K *blk = &blocks[(size_t)row * n_bpr + b];
                float d = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qh = blk->qh;

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

                __m128i w_lo128[8], w_hi128[8];
                const uint8_t *qs = blk->qs;
                for (int p = 0; p < 4; p++) {
                    __m128i raw0 = _mm_loadu_si128((const __m128i *)qs);
                    __m128i raw1 = _mm_loadu_si128((const __m128i *)(qs + 16));
                    int bit_lo = p * 2;
                    int bit_hi = p * 2 + 1;
                    __m128i hb0 = q5k_extract_hb(qh, 0, bit_lo);
                    __m128i hb1 = q5k_extract_hb(qh, 16, bit_lo);
                    __m128i hb2 = q5k_extract_hb(qh, 0, bit_hi);
                    __m128i hb3 = q5k_extract_hb(qh, 16, bit_hi);
                    w_lo128[2 * p] = _mm_or_si128(_mm_and_si128(raw0, mask_lo), hb0);
                    w_hi128[2 * p] = _mm_or_si128(_mm_and_si128(raw1, mask_lo), hb1);
                    w_lo128[2 * p + 1] = _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_lo), hb2);
                    w_hi128[2 * p + 1] = _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_lo), hb3);
                    qs += 32;
                }

                __m256i wv[8];
                for (int p = 0; p < 8; p++) {
                    __m256i z = _mm256_castsi128_si256(w_lo128[p]);
                    wv[p] = _mm256_inserti128_si256(z, w_hi128[p], 1);
                }

                __m256i sc_v[8];
                for (int p = 0; p < 8; p++)
                    sc_v[p] = _mm256_set1_epi16((int16_t)sc[p]);
                __m128i mins_v = _mm_cvtepu8_epi16(
                    _mm_loadl_epi64((const __m128i *)mins));

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                    float dx = c->x_d[(size_t)t * n_bpr + b];
                    const int16_t *bsums =
                        c->x_bsums + ((size_t)t * n_bpr + b) * 16;

                    __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
                    __m128i bs_lo = _mm256_castsi256_si128(q8sums);
                    __m128i bs_hi = _mm256_extracti128_si256(q8sums, 1);
                    __m128i bs_paired = _mm_hadd_epi16(bs_lo, bs_hi);
                    __m128i corr128 = _mm_madd_epi16(mins_v, bs_paired);
                    __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
                    __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
                    int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

                    __m256i sumi_v = _mm256_setzero_si256();
                    for (int p = 0; p < 8; p++) {
                        __m256i xv = _mm256_loadu_si256(
                            (const __m256i *)(xb + p * 32));
                        __m256i prod = _mm256_maddubs_epi16(wv[p], xv);
                        prod = _mm256_madd_epi16(sc_v[p], prod);
                        sumi_v = _mm256_add_epi32(sumi_v, prod);
                    }

                    int32_t dot = bn_avx2_hsum_epi32(sumi_v);
                    acc[ti] += dx * (d * (float)dot -
                                      dmin * (float)bsum_corr);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += acc[ti];
        }
    }
}
