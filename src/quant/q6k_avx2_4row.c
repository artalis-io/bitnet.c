#include "quant_internal.h"
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

// 4-row Q6_K SDOT matvec: process 4 output rows at once, unpacking weights
// per-row but loading x_q once. Amortizes activation vector memory read 4x.

void bn_quant_q6k_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
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
