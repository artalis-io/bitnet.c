#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

// 4-row I2S matvec: process 4 output rows at once, loading x_q once.
// This amortizes the activation vector memory read 4x — critical on
// bandwidth-limited DDR4 (~50 GB/s vs M1's ~400 GB/s).

void bn_quant_i2s_avx2_4row_range(void *ctx, int group_start, int group_end) {
    BnI2SCtx *c = (BnI2SCtx *)ctx;
    int row_bytes = c->W->cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float combined_scale = c->combined_scale;
    int cols = c->W->cols;
    int rows = c->W->rows;
    const int8_t *x_q = c->x_q;

    const __m256i mask3 = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi8(1);

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 4;
        int nrows = (row0 + 4 <= rows) ? 4 : rows - row0;

        const uint8_t *rd0 = base + (size_t)row0 * row_bytes;
        const uint8_t *rd1 = (nrows > 1) ? base + (size_t)(row0 + 1) * row_bytes : rd0;
        const uint8_t *rd2 = (nrows > 2) ? base + (size_t)(row0 + 2) * row_bytes : rd0;
        const uint8_t *rd3 = (nrows > 3) ? base + (size_t)(row0 + 3) * row_bytes : rd0;

        __m256i acc0A = _mm256_setzero_si256(), acc0B = _mm256_setzero_si256();
        __m256i acc0C = _mm256_setzero_si256(), acc0D = _mm256_setzero_si256();
        __m256i acc1A = _mm256_setzero_si256(), acc1B = _mm256_setzero_si256();
        __m256i acc1C = _mm256_setzero_si256(), acc1D = _mm256_setzero_si256();
        __m256i acc2A = _mm256_setzero_si256(), acc2B = _mm256_setzero_si256();
        __m256i acc2C = _mm256_setzero_si256(), acc2D = _mm256_setzero_si256();
        __m256i acc3A = _mm256_setzero_si256(), acc3B = _mm256_setzero_si256();
        __m256i acc3C = _mm256_setzero_si256(), acc3D = _mm256_setzero_si256();

        for (int done = 0; done < cols; done += 128) {
            // Load x_q ONCE for 4 rows
            const int8_t *xp = x_q + done;
            __m256i x0 = _mm256_loadu_si256((const __m256i *)xp);
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(xp + 32));
            __m256i x2 = _mm256_loadu_si256((const __m256i *)(xp + 64));
            __m256i x3 = _mm256_loadu_si256((const __m256i *)(xp + 96));

            // Prefetch next weight blocks
            _mm_prefetch((const char *)(rd0 + 64), _MM_HINT_T0);
            _mm_prefetch((const char *)(rd1 + 64), _MM_HINT_T0);
            _mm_prefetch((const char *)(rd2 + 64), _MM_HINT_T0);
            _mm_prefetch((const char *)(rd3 + 64), _MM_HINT_T0);

            // Row 0
            {
                __m256i raw = _mm256_loadu_si256((const __m256i *)rd0);
                __m256i t0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3), one);
                __m256i t1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3), one);
                __m256i t2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3), one);
                __m256i t3 = _mm256_sub_epi8(_mm256_and_si256(raw, mask3), one);
                acc0A = bn_avx2_dpbusd(acc0A, t0, x0);
                acc0B = bn_avx2_dpbusd(acc0B, t1, x1);
                acc0C = bn_avx2_dpbusd(acc0C, t2, x2);
                acc0D = bn_avx2_dpbusd(acc0D, t3, x3);
                rd0 += 32;
            }

            // Row 1
            {
                __m256i raw = _mm256_loadu_si256((const __m256i *)rd1);
                __m256i t0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3), one);
                __m256i t1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3), one);
                __m256i t2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3), one);
                __m256i t3 = _mm256_sub_epi8(_mm256_and_si256(raw, mask3), one);
                acc1A = bn_avx2_dpbusd(acc1A, t0, x0);
                acc1B = bn_avx2_dpbusd(acc1B, t1, x1);
                acc1C = bn_avx2_dpbusd(acc1C, t2, x2);
                acc1D = bn_avx2_dpbusd(acc1D, t3, x3);
                rd1 += 32;
            }

            // Row 2
            {
                __m256i raw = _mm256_loadu_si256((const __m256i *)rd2);
                __m256i t0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3), one);
                __m256i t1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3), one);
                __m256i t2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3), one);
                __m256i t3 = _mm256_sub_epi8(_mm256_and_si256(raw, mask3), one);
                acc2A = bn_avx2_dpbusd(acc2A, t0, x0);
                acc2B = bn_avx2_dpbusd(acc2B, t1, x1);
                acc2C = bn_avx2_dpbusd(acc2C, t2, x2);
                acc2D = bn_avx2_dpbusd(acc2D, t3, x3);
                rd2 += 32;
            }

            // Row 3
            {
                __m256i raw = _mm256_loadu_si256((const __m256i *)rd3);
                __m256i t0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3), one);
                __m256i t1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3), one);
                __m256i t2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3), one);
                __m256i t3 = _mm256_sub_epi8(_mm256_and_si256(raw, mask3), one);
                acc3A = bn_avx2_dpbusd(acc3A, t0, x0);
                acc3B = bn_avx2_dpbusd(acc3B, t1, x1);
                acc3C = bn_avx2_dpbusd(acc3C, t2, x2);
                acc3D = bn_avx2_dpbusd(acc3D, t3, x3);
                rd3 += 32;
            }
        }

        // Horizontal sums
        int32_t t0 = bn_avx2_hsum_epi32(_mm256_add_epi32(_mm256_add_epi32(acc0A, acc0B), _mm256_add_epi32(acc0C, acc0D)));
        c->out[row0] = (float)t0 * combined_scale;
        if (nrows > 1) {
            int32_t t1 = bn_avx2_hsum_epi32(_mm256_add_epi32(_mm256_add_epi32(acc1A, acc1B), _mm256_add_epi32(acc1C, acc1D)));
            c->out[row0 + 1] = (float)t1 * combined_scale;
        }
        if (nrows > 2) {
            int32_t t2 = bn_avx2_hsum_epi32(_mm256_add_epi32(_mm256_add_epi32(acc2A, acc2B), _mm256_add_epi32(acc2C, acc2D)));
            c->out[row0 + 2] = (float)t2 * combined_scale;
        }
        if (nrows > 3) {
            int32_t t3 = bn_avx2_hsum_epi32(_mm256_add_epi32(_mm256_add_epi32(acc3A, acc3B), _mm256_add_epi32(acc3C, acc3D)));
            c->out[row0 + 3] = (float)t3 * combined_scale;
        }
    }
}
