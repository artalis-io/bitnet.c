#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <math.h>

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

static inline __m256i q5k_scale_shuffle(int i) {
    static const uint8_t shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)shuffle + i);
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

static inline void q5k_sdot_accumulate_block(const BnBlockQ5K *blk,
                                               const int8_t *xq,
                                               float dx,
                                               const int16_t *bsums,
                                               __m256 *acc,
                                               float *min_sum) {
    const __m256i mask_lo = _mm256_set1_epi8(0x0f);
    const __m256i one = _mm256_set1_epi8(1);
    const __m128i zero = _mm_setzero_si128();
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;
    float d = dx * bn_fp16_to_fp32(blk->d);
    float dmin = -dx * bn_fp16_to_fp32(blk->dmin);

    uint32_t utmp[4] = {0};
    memcpy(utmp, blk->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) |
              (((utmp[1] >> 6) & kmask3) << 4);
    uint32_t aux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) |
              (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = aux;
    utmp[0] &= kmask1;
    const __m128i packed = _mm_set_epi32(
        (int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]);
    const __m256i mins_scales = _mm256_cvtepu8_epi16(packed);
    const __m256i bsum_vec = _mm256_loadu_si256((const __m256i *)bsums);
    const __m128i bsum_pairs = _mm_hadd_epi16(
        _mm256_extracti128_si256(bsum_vec, 0),
        _mm256_extracti128_si256(bsum_vec, 1));
    const __m128i min_prod = _mm_madd_epi16(
        _mm256_extracti128_si256(mins_scales, 1), bsum_pairs);
    const __m128i min_total = _mm_hadd_epi32(
        _mm_hadd_epi32(min_prod, zero), zero);
    *min_sum = fmaf(dmin, (float)_mm_extract_epi32(min_total, 0), *min_sum);

    const __m128i scales128 = _mm256_extracti128_si256(mins_scales, 0);
    const __m256i scales = _mm256_inserti128_si256(
        _mm256_castsi128_si256(scales128), scales128, 1);
    const __m256i hbits = _mm256_loadu_si256((const __m256i *)blk->qh);
    __m256i hmask = one;
    __m256i sumi = _mm256_setzero_si256();
    const uint8_t *qs = blk->qs;
    int bit = 0;
    for (int p = 0; p < 4; p++) {
        const __m256i scale_lo = _mm256_shuffle_epi8(
            scales, q5k_scale_shuffle(2 * p));
        const __m256i scale_hi = _mm256_shuffle_epi8(
            scales, q5k_scale_shuffle(2 * p + 1));
        __m256i raw = _mm256_loadu_si256((const __m256i *)qs);
        qs += 32;
        __m256i lo = _mm256_and_si256(raw, mask_lo);
        __m256i hi_bits = _mm256_slli_epi16(
            _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
        lo = _mm256_add_epi8(lo, hi_bits);
        hmask = _mm256_slli_epi16(hmask, 1);
        __m256i hi = _mm256_and_si256(
            _mm256_srli_epi16(raw, 4), mask_lo);
        hi_bits = _mm256_slli_epi16(
            _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
        hi = _mm256_add_epi8(hi, hi_bits);
        hmask = _mm256_slli_epi16(hmask, 1);
        __m256i prod_lo = _mm256_maddubs_epi16(
            lo, _mm256_loadu_si256((const __m256i *)(xq + p * 64)));
        __m256i prod_hi = _mm256_maddubs_epi16(
            hi, _mm256_loadu_si256((const __m256i *)(xq + p * 64 + 32)));
        prod_lo = _mm256_madd_epi16(scale_lo, prod_lo);
        prod_hi = _mm256_madd_epi16(scale_hi, prod_hi);
        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(prod_lo, prod_hi));
    }
    *acc = _mm256_fmadd_ps(
        _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), *acc);
}

void bn_quant_q5k_avx2_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int n_bpr = c->W->cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    for (int row = row_start; row < row_end; row++) {
        __m256 acc = _mm256_setzero_ps();
        float min_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++)
            q5k_sdot_accumulate_block(
                &blocks[(size_t)row * n_bpr + b],
                c->x_q + (size_t)b * BN_QK_K, c->x_d[b],
                c->x_bsums + b * 16, &acc, &min_sum);
        c->out[row] = bn_avx2_hsum_ps(acc) + min_sum;
    }
}

static void q5k_x8_group(const BnKQuantSdotCtx *c, int group) {
    int n_bpr = c->W->cols / BN_QK_K;
    const BnBlockQ5Kx8 *blocks = (const BnBlockQ5Kx8 *)c->prepared->aux;
    float sumf[8] = {0};
    float sum_minf[8] = {0};
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int b = 0; b < n_bpr; b++) {
        const BnBlockQ5Kx8 *blk = &blocks[(size_t)group * n_bpr + b];
        uint32_t utmp[32];
        for (int sb = 0; sb < 8; sb++) {
            memcpy(utmp + sb * 4, blk->scales + sb * 12, 12);
            utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) |
                (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
            uint32_t aux = utmp[sb * 4 + 1] & kmask1;
            utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) |
                (((utmp[sb * 4] >> 6) & kmask3) << 4);
            utmp[sb * 4 + 2] = aux;
            utmp[sb * 4] &= kmask1;
        }

        const int8_t *xq = c->x_q + (size_t)b * BN_QK_K;
        float xd = c->x_d[b];
        for (int k = 0; k < 16; k++) {
            const uint8_t *sc0 = (const uint8_t *)utmp + (k / 4) * 32;
            const uint8_t *sc1 = sc0 + 16;
            int qh_shift = (k / 4) * 2;
            for (int r = 0; r < 8; r++) {
                int sumi = 0;
                for (int i = 0; i < 8; i++) {
                    int qoff = k * 64 + r * 8 + i;
                    int qh_idx = (k * 8 + i) & 31;
                    int qh_off = (qh_idx / 8) * 64 + r * 8 + (qh_idx & 7);
                    uint8_t hi = blk->qh[qh_off];
                    int v0 = (blk->qs[qoff] & 15) |
                             (((hi >> qh_shift) & 1) << 4);
                    int v1 = (blk->qs[qoff] >> 4) |
                             (((hi >> (qh_shift + 1)) & 1) << 4);
                    int xoff = (k / 4) * 64 + (k & 3) * 8 + i;
                    sumi += v0 * xq[xoff] * sc0[r] +
                            v1 * xq[xoff + 32] * sc1[r];
                }
                sumf[r] += (float)sumi * bn_fp16_to_fp32(blk->d[r]) * xd;
            }
        }
        for (int sb = 0; sb < 8; sb++) {
            const uint8_t *mins = (const uint8_t *)utmp + 8 + sb * 16;
            int bsum = c->x_bsums[b * 16 + sb * 2] +
                       c->x_bsums[b * 16 + sb * 2 + 1];
            for (int r = 0; r < 8; r++)
                sum_minf[r] += (float)(mins[r] * bsum) *
                    bn_fp16_to_fp32(blk->dmin[r]) * xd;
        }
    }
    for (int r = 0; r < 8; r++)
        c->out[group * 8 + r] = sumf[r] - sum_minf[r];
}

void bn_quant_q5k_avx2_x8_matvec_range(void *ctx, int start, int end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    for (int group = start; group < end; group++)
        q5k_x8_group(c, group);
}

void bn_quant_q5k_avx2_x8_matmul_range(void *ctx, int start, int end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int n_bpr = c->cols / BN_QK_K;
    for (int group = start; group < end; group++) {
        for (int token = 0; token < c->n_tokens; token++) {
            BnKQuantSdotCtx one = {
                c->out + (size_t)token * c->W->rows,
                c->W,
                c->x_q + (size_t)token * c->cols,
                c->x_d + (size_t)token * n_bpr,
                c->x_bsums + (size_t)token * n_bpr * 16,
                c->prepared
            };
            q5k_x8_group(&one, group);
        }
    }
}

void bn_quant_q5k_avx2_x8_gemm_range(void *ctx, int start, int end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int nb = c->cols / BN_QK_K;
    int rows = c->W->rows;
    if (!c->prepared || c->prepared->kind != BN_PREPARED_WEIGHT_Q5_K_X8 ||
        !c->prepared->aux || !c->x_q8k_x4 || (c->n_tokens % 4) != 0) {
        bn_quant_q5k_avx2_x8_matmul_range(ctx, start, end);
        return;
    }

    const BnBlockQ5Kx8 *blocks =
        (const BnBlockQ5Kx8 *)c->prepared->aux;
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int panel = 0; panel < c->n_tokens / 4; panel++) {
        const BnBlockQ8Kx4 *a_ptr = c->x_q8k_x4 + (size_t)panel * nb;
        for (int group = start; group < end; group++) {
            float sumf[4][8] = {{0}};
            float sum_minf[4][8] = {{0}};
            const BnBlockQ5Kx8 *b_ptr = blocks + (size_t)group * nb;

            for (int b = 0; b < nb; b++) {
                uint32_t utmp[32];
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[b].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] =
                        ((utmp[sb * 4 + 2] >> 4) & kmask2) |
                        (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    uint32_t aux = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] =
                        (utmp[sb * 4 + 2] & kmask2) |
                        (((utmp[sb * 4] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = aux;
                    utmp[sb * 4] &= kmask1;
                }

                for (int k = 0; k < 16; k++) {
                    const uint8_t *scales_0 =
                        (const uint8_t *)utmp + (k / 4) * 32;
                    const uint8_t *scales_1 = scales_0 + 16;
                    int qh_shift = (k / 4) * 2;
                    for (int m = 0; m < 4; m++) {
                        for (int r = 0; r < 8; r++) {
                            int sumi = 0;
                            for (int i = 0; i < 8; i++) {
                                int qoff = k * 64 + r * 8 + i;
                                int qh_idx = (k * 8 + i) % 32;
                                int qh_off = (qh_idx / 8) * 64 +
                                             r * 8 + qh_idx % 8;
                                uint8_t hi = b_ptr[b].qh[qh_off];
                                int v0 = (b_ptr[b].qs[qoff] & 15) |
                                    (((hi >> qh_shift) & 1) << 4);
                                int v1 = (b_ptr[b].qs[qoff] >> 4) |
                                    (((hi >> (qh_shift + 1)) & 1) << 4);
                                int xoff = (k / 4) * 256 + (k % 4) * 32 +
                                           m * 8 + i;
                                int sumi1 = v0 * a_ptr[b].qs[xoff];
                                int sumi2 = v1 * a_ptr[b].qs[xoff + 128];
                                sumi1 *= scales_0[r];
                                sumi2 *= scales_1[r];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][r] +=
                                sumi * bn_fp16_to_fp32(b_ptr[b].d[r]) *
                                a_ptr[b].d[m];
                        }
                    }
                }

                for (int sb = 0; sb < 8; sb++) {
                    const uint8_t *mins =
                        (const uint8_t *)utmp + 8 + sb * 16;
                    for (int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[b].bsums + sb * 8 +
                                              m * 4 - (sb % 2) * 6;
                        for (int r = 0; r < 8; r++)
                            sum_minf[m][r] +=
                                mins[r] * (bsums[0] + bsums[1]) *
                                bn_fp16_to_fp32(b_ptr[b].dmin[r]) *
                                a_ptr[b].d[m];
                    }
                }
            }

            for (int m = 0; m < 4; m++)
                for (int r = 0; r < 8; r++)
                    c->out[((size_t)panel * 4 + m) * rows + group * 8 + r] =
                        sumf[m][r] - sum_minf[m][r];
        }
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
                c->out[(size_t)(t0 + ti) * rows + row] = sums[ti];
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
/* Four tokens share weight decoding. The wider register file keeps their
 * independent GEMV accumulators live; no horizontal reduction is moved. */
static void q5k_sdot_tile4(const BnKQuantMatmulCtx *c, int token,
                           int row_start, int row_end) {
    const int nb = c->cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const __m256i mask = _mm256_set1_epi8(15);
    const __m128i zero = _mm_setzero_si128();
    for (int row = row_start; row < row_end; row++) {
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                         _mm256_setzero_ps(), _mm256_setzero_ps()};
        float min_sum[4] = {0};
        for (int b = 0; b < nb; b++) {
            const BnBlockQ5K *blk = &blocks[(size_t)row * nb + b];
            uint32_t tmp[4] = {0};
            memcpy(tmp, blk->scales, 12);
            tmp[3] = ((tmp[2] >> 4) & 0x0f0f0f0fu) |
                     (((tmp[1] >> 6) & 0x03030303u) << 4);
            uint32_t aux = tmp[1] & 0x3f3f3f3fu;
            tmp[1] = (tmp[2] & 0x0f0f0f0fu) |
                     (((tmp[0] >> 6) & 0x03030303u) << 4);
            tmp[2] = aux;
            tmp[0] &= 0x3f3f3f3fu;
            __m256i sm = _mm256_cvtepu8_epi16(_mm_set_epi32(
                (int)tmp[3], (int)tmp[2], (int)tmp[1], (int)tmp[0]));
            __m128i mins = _mm256_extracti128_si256(sm, 1);
            __m128i sc = _mm256_castsi256_si128(sm);
            __m256i scales = _mm256_set_m128i(sc, sc);
            __m256i hi = _mm256_loadu_si256((const __m256i *)blk->qh);
            __m256i weights[8], scale[8];
            for (int p = 0; p < 4; p++) {
                __m256i raw = _mm256_loadu_si256((const __m256i *)(blk->qs + p * 32));
                __m256i hb0 = _mm256_and_si256(_mm256_srli_epi16(hi, 2 * p),
                                              _mm256_set1_epi8(1));
                __m256i hb1 = _mm256_and_si256(_mm256_srli_epi16(hi, 2 * p + 1),
                                              _mm256_set1_epi8(1));
                weights[2 * p] = _mm256_add_epi8(_mm256_and_si256(raw, mask),
                                                _mm256_slli_epi16(hb0, 4));
                weights[2 * p + 1] = _mm256_add_epi8(
                    _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask),
                    _mm256_slli_epi16(hb1, 4));
                scale[2 * p] = _mm256_shuffle_epi8(scales, q5k_scale_shuffle(2 * p));
                scale[2 * p + 1] = _mm256_shuffle_epi8(scales, q5k_scale_shuffle(2 * p + 1));
            }
            float wd = bn_fp16_to_fp32(blk->d);
            float wm = bn_fp16_to_fp32(blk->dmin);
            for (int t = 0; t < 4; t++) {
                const int8_t *q = c->x_q + (size_t)(token + t) * c->cols + b * BN_QK_K;
                const int16_t *bs = c->x_bsums + ((size_t)(token + t) * nb + b) * 16;
                float dx = c->x_d[(size_t)(token + t) * nb + b];
                __m256i bv = _mm256_loadu_si256((const __m256i *)bs);
                __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(bv),
                                               _mm256_extracti128_si256(bv, 1));
                __m128i prod = _mm_madd_epi16(mins, pairs);
                __m128i total = _mm_hadd_epi32(_mm_hadd_epi32(prod, zero), zero);
                min_sum[t] = fmaf(-dx * wm, (float)_mm_extract_epi32(total, 0), min_sum[t]);
                __m256i sum = _mm256_setzero_si256();
                for (int p = 0; p < 4; p++) {
                    __m256i lo = _mm256_madd_epi16(scale[2 * p], _mm256_maddubs_epi16(
                        weights[2 * p], _mm256_loadu_si256((const __m256i *)(q + p * 64))));
                    __m256i high = _mm256_madd_epi16(scale[2 * p + 1], _mm256_maddubs_epi16(
                        weights[2 * p + 1], _mm256_loadu_si256((const __m256i *)(q + p * 64 + 32))));
                    sum = _mm256_add_epi32(sum, _mm256_add_epi32(lo, high));
                }
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * wd),
                                        _mm256_cvtepi32_ps(sum), acc[t]);
            }
        }
        for (int t = 0; t < 4; t++)
            c->out[(size_t)(token + t) * c->W->rows + row] =
                bn_avx2_hsum_ps(acc[t]) + min_sum[t];
    }
}
#endif

void bn_quant_q5k_avx2_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int n_bpr = c->cols / BN_QK_K;
    int token = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    for (; token + 4 <= c->n_tokens; token += 4)
        q5k_sdot_tile4(c, token, row_start, row_end);
#endif
    for (; token < c->n_tokens; token++) {
        BnKQuantSdotCtx one = {
            c->out + (size_t)token * c->W->rows, c->W,
            c->x_q + (size_t)token * c->cols,
            c->x_d + (size_t)token * n_bpr,
            c->x_bsums + (size_t)token * n_bpr * 16,
            c->prepared
        };
        bn_quant_q5k_avx2_sdot_range(&one, row_start, row_end);
    }
}
