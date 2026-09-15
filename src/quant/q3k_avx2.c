#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

static inline float q3k_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

static inline __m256i q3k_scale_shuffle(int i) {
    static const uint8_t shuffle[128] = {
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,
        4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,
        6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,
        8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i *)shuffle + i);
}

void bn_quant_q3k_avx2_q8k_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    const int nb = c->W->cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    for (int row = row_start; row < row_end; row++) {
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const BnBlockQ3K *x = &blocks[(size_t)row * nb + b];
            const int8_t *q8 = c->x_q + b * BN_QK_K;
            const uint8_t *q3 = x->qs;
            uint32_t aux[3];
            memcpy(aux, x->scales, sizeof(aux));
            __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) |
                    (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) |
                    (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) |
                    (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) |
                    (((aux[2] >> 0) & kmask1) << 4));
            scales128 = _mm_sub_epi8(scales128, m32);
            __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
            __m128i lo_scales = _mm256_castsi256_si128(all_scales);
            __m128i hi_scales = _mm256_extracti128_si256(all_scales, 1);
            __m256i scales[2] = {
                _mm256_set_m128i(lo_scales, lo_scales),
                _mm256_set_m128i(hi_scales, hi_scales),
            };
            __m256i hbits = _mm256_loadu_si256(
                (const __m256i *)x->hmask);
            __m256i sumi = _mm256_setzero_si256();
            int bit = 0;

            for (int j = 0; j < 2; j++) {
                __m256i q3bits = _mm256_loadu_si256(
                    (const __m256i *)q3);
                q3 += 32;
                __m256i low[4] = {
                    _mm256_and_si256(q3bits, m3),
                    _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3),
                    _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3),
                    _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3),
                };
                __m256i p[4];
                for (int k = 0; k < 4; k++, bit++) {
                    __m256i high = _mm256_slli_epi16(
                        _mm256_srli_epi16(_mm256_andnot_si256(
                            hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                    __m256i q8v = _mm256_loadu_si256(
                        (const __m256i *)q8);
                    q8 += 32;
                    __m256i dot_high = _mm256_maddubs_epi16(high, q8v);
                    __m256i dot_low = _mm256_maddubs_epi16(low[k], q8v);
                    p[k] = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales[j],
                                            q3k_scale_shuffle(k)),
                        _mm256_sub_epi16(dot_low, dot_high));
                }
                sumi = _mm256_add_epi32(
                    sumi, _mm256_add_epi32(
                        _mm256_add_epi32(p[0], p[1]),
                        _mm256_add_epi32(p[2], p[3])));
            }
            float d = c->x_d[b] * q3k_fp16_to_fp32(x->d);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d),
                                  _mm256_cvtepi32_ps(sumi), acc);
        }
        c->out[row] = bn_avx2_hsum_ps(acc);
    }
}

void bn_quant_q3k_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ3KCtx *c = (BnQ3KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const float *x = c->x;

    const __m128i mask2 = _mm_set1_epi8(3);
    const __m128i bias4 = _mm_set1_epi8(4);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ3K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = q3k_fp16_to_fp32(blk->d);

            uint8_t scales[16];
            bn_q3k_unpack_scales(blk->scales, scales);

            const uint8_t *q  = blk->qs;
            const uint8_t *hm = blk->hmask;
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();

            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    // Vectorized: load 16 bytes of q and hm, extract 2-bit values
                    __m128i q0 = _mm_loadu_si128((const __m128i *)q);
                    __m128i q1 = _mm_loadu_si128((const __m128i *)(q + 16));
                    __m128i hm0 = _mm_loadu_si128((const __m128i *)hm);
                    __m128i hm1 = _mm_loadu_si128((const __m128i *)(hm + 16));

                    // Extract low 2 bits after shift: (q >> shift) & 3
                    __m128i w0, w1;
                    if (shift == 0) {
                        w0 = _mm_and_si128(q0, mask2);
                        w1 = _mm_and_si128(q1, mask2);
                    } else {
                        w0 = _mm_and_si128(_mm_srli_epi16(q0, shift), mask2);
                        w1 = _mm_and_si128(_mm_srli_epi16(q1, shift), mask2);
                    }

                    // High bit: if (hm & m) is set, subtract 0; else subtract 4
                    // i.e., result = (q>>shift)&3 - (hm_bit_set ? 0 : 4)
                    // = (q>>shift)&3 - 4 + (hm_bit_set ? 4 : 0)
                    // Better: bias = 4 where hm bit is NOT set
                    __m128i vm = _mm_set1_epi8((char)m);
                    __m128i hm_set0 = _mm_cmpeq_epi8(_mm_and_si128(hm0, vm), _mm_setzero_si128());
                    __m128i hm_set1 = _mm_cmpeq_epi8(_mm_and_si128(hm1, vm), _mm_setzero_si128());
                    // hm_set is 0xFF where bit NOT set (should subtract 4), 0x00 where set
                    __m128i sub0 = _mm_and_si128(hm_set0, bias4);
                    __m128i sub1 = _mm_and_si128(hm_set1, bias4);

                    // Final: w = (q>>shift)&3 - sub (as signed int8)
                    w0 = _mm_sub_epi8(w0, sub0);
                    w1 = _mm_sub_epi8(w1, sub1);

                    float dl0 = d * ((int)scales[is++] - 32);
                    __m256 vds0 = _mm256_set1_ps(dl0);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0)), vds0), _mm256_loadu_ps(xb + n + j*32), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0, 8))), vds0), _mm256_loadu_ps(xb + n + j*32 + 8), acc);

                    float dl1 = d * ((int)scales[is++] - 32);
                    __m256 vds1 = _mm256_set1_ps(dl1);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1)), vds1), _mm256_loadu_ps(xb + n + j*32 + 16), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1, 8))), vds1), _mm256_loadu_ps(xb + n + j*32 + 24), acc);

                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}

#define Q3K_MATMUL_TILE_T 8

void bn_quant_q3k_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    int n_bpr = c->cols / BN_QK_K;
    int rows = c->W->rows;
    const __m128i mask2 = _mm_set1_epi8(3);
    const __m128i bias4 = _mm_set1_epi8(4);

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < c->n_tokens; t0 += Q3K_MATMUL_TILE_T) {
            int tile_n = t0 + Q3K_MATMUL_TILE_T <= c->n_tokens
                ? Q3K_MATMUL_TILE_T : c->n_tokens - t0;
            float row_sum[Q3K_MATMUL_TILE_T] = {0};

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ3K *blk = &blocks[(size_t)row * n_bpr + b];
                uint8_t scales[16];
                bn_q3k_unpack_scales(blk->scales, scales);
                const uint8_t *q = blk->qs;
                const uint8_t *hm = blk->hmask;
                __m256 acc[Q3K_MATMUL_TILE_T];
                for (int ti = 0; ti < tile_n; ti++)
                    acc[ti] = _mm256_setzero_ps();

                int is = 0;
                uint8_t m = 1;
                float d = q3k_fp16_to_fp32(blk->d);
                for (int n = 0; n < BN_QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        __m128i q0 = _mm_loadu_si128((const __m128i *)q);
                        __m128i q1 = _mm_loadu_si128((const __m128i *)(q + 16));
                        __m128i hm0 = _mm_loadu_si128((const __m128i *)hm);
                        __m128i hm1 = _mm_loadu_si128((const __m128i *)(hm + 16));
                        __m128i w0 = shift == 0 ? _mm_and_si128(q0, mask2) :
                            _mm_and_si128(_mm_srli_epi16(q0, shift), mask2);
                        __m128i w1 = shift == 0 ? _mm_and_si128(q1, mask2) :
                            _mm_and_si128(_mm_srli_epi16(q1, shift), mask2);
                        __m128i vm = _mm_set1_epi8((char)m);
                        w0 = _mm_sub_epi8(w0, _mm_and_si128(
                            _mm_cmpeq_epi8(_mm_and_si128(hm0, vm),
                                           _mm_setzero_si128()), bias4));
                        w1 = _mm_sub_epi8(w1, _mm_and_si128(
                            _mm_cmpeq_epi8(_mm_and_si128(hm1, vm),
                                           _mm_setzero_si128()), bias4));

                        __m256 wd[4];
                        __m256 ds0 = _mm256_set1_ps(
                            d * ((int)scales[is++] - 32));
                        __m256 ds1 = _mm256_set1_ps(
                            d * ((int)scales[is++] - 32));
                        wd[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(w0)), ds0);
                        wd[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(_mm_srli_si128(w0, 8))), ds0);
                        wd[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(w1)), ds1);
                        wd[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(_mm_srli_si128(w1, 8))), ds1);

                        for (int ti = 0; ti < tile_n; ti++) {
                            const float *xb = c->x +
                                (size_t)(t0 + ti) * c->cols +
                                b * BN_QK_K + n + j * 32;
                            for (int k = 0; k < 4; k++)
                                acc[ti] = _mm256_fmadd_ps(
                                    wd[k], _mm256_loadu_ps(xb + k * 8), acc[ti]);
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    q += 32;
                }
                for (int ti = 0; ti < tile_n; ti++)
                    row_sum[ti] += bn_avx2_hsum_ps(acc[ti]);
            }
            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] = row_sum[ti];
        }
    }
}
