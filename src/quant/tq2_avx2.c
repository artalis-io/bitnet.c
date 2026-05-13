#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

void bn_quant_tq2_avx2_range(void *ctx, int row_start, int row_end) {
    BnTQ2SdotCtx *c = (BnTQ2SdotCtx *)ctx;
    const BnBlockTQ2 *blocks = (const BnBlockTQ2 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float combined_scale = c->combined_scale;
    const int8_t *x_q = c->x_q;

    const __m256i mask3 = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi8(1);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ2 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            const int8_t *xb = x_q + b * BN_QK_K;

            __m256i iaccA = _mm256_setzero_si256();
            __m256i iaccB = _mm256_setzero_si256();
            __m256i iaccC = _mm256_setzero_si256();
            __m256i iaccD = _mm256_setzero_si256();

            for (int half = 0; half < 2; half++) {
                const uint8_t *qs = blk->qs + half * 32;
                const int8_t *xh = xb + half * 128;

                for (int i = 0; i < 2; i++) {
                    __m128i raw128 = _mm_loadu_si128((const __m128i *)(qs + i * 16));
                    __m256i raw = _mm256_set_m128i(raw128, raw128);

                    __m256i t0 = _mm256_sub_epi8(_mm256_and_si256(raw, mask3), one);
                    __m256i t1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3), one);
                    __m256i t2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3), one);
                    __m256i t3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3), one);

                    const int8_t *xp = xh + i * 16;
                    // TQ2 layout: each of 16 bytes produces 4 ternary values interleaved
                    // at positions m, m+32, m+64, m+96 within the half
                    // But the x vector is linear, so t0 covers positions [0..15], t1 covers [32..47], etc.
                    // Within each half (128 elements from 32 packed bytes):
                    //   first 16 bytes → elements at offsets 0..15 (t0), 32..47 (t1), 64..79 (t2), 96..111 (t3)
                    //   next 16 bytes  → elements at offsets 16..31 (t0), 48..63 (t1), 80..95 (t2), 112..127 (t3)
                    // Since DPBUSD works on 32 int8, we need to expand 16->32 or handle differently
                    // Actually: raw128 has 16 bytes, each t has 16 elements in lower 128 bits
                    // Use 128-bit dpbusd equivalent: process 16 elements at a time

                    // Extract lower 128 bits (both halves of __m256i are identical, only need lower)
                    __m128i t0_128 = _mm256_castsi256_si128(t0);
                    __m128i t1_128 = _mm256_castsi256_si128(t1);
                    __m128i t2_128 = _mm256_castsi256_si128(t2);
                    __m128i t3_128 = _mm256_castsi256_si128(t3);

                    // Pack pairs of 16-byte ternary vectors into 32-byte vectors for DPBUSD
                    // t0 (16 elts at offset 0) + t0 from next i iteration? No, just zero-extend
                    // Better: combine t0,t1 into one 256-bit and t2,t3 into another
                    __m256i w01 = _mm256_set_m128i(t1_128, t0_128);
                    __m256i w23 = _mm256_set_m128i(t3_128, t2_128);

                    __m256i xv01 = _mm256_set_m128i(
                        _mm_loadu_si128((const __m128i *)(xp + 1*32)),
                        _mm_loadu_si128((const __m128i *)(xp + 0*32)));
                    __m256i xv23 = _mm256_set_m128i(
                        _mm_loadu_si128((const __m128i *)(xp + 3*32)),
                        _mm_loadu_si128((const __m128i *)(xp + 2*32)));

                    iaccA = bn_avx2_dpbusd(iaccA, w01, xv01);
                    iaccB = bn_avx2_dpbusd(iaccB, w23, xv23);
                }
            }

            __m256i sum = _mm256_add_epi32(_mm256_add_epi32(iaccA, iaccB),
                                            _mm256_add_epi32(iaccC, iaccD));
            row_sum += d * (float)bn_avx2_hsum_epi32(sum);
        }
        c->out[row] = row_sum * combined_scale;
    }
}
