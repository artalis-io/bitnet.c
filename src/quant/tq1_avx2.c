#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>

// TQ1 decode: extract top trit {0,1,2} from a product byte.
// Equivalent to NEON: vshrq_n_u8(vhaddq_u8(x, vshrq_n_u8(x, 1)), 6)
// = ((x + (x >> 1)) >> 1) >> 6 = (x + (x >> 1)) >> 7
// AVX2 has no 8-bit shift, so use 16-bit arithmetic.
static inline __m256i tq1_decode_avx2(__m256i x) {
    __m128i lo128 = _mm256_castsi256_si128(x);
    __m128i hi128 = _mm256_extracti128_si256(x, 1);
    __m256i lo16 = _mm256_cvtepu8_epi16(lo128);
    __m256i hi16 = _mm256_cvtepu8_epi16(hi128);
    lo16 = _mm256_srli_epi16(_mm256_add_epi16(lo16, _mm256_srli_epi16(lo16, 1)), 7);
    hi16 = _mm256_srli_epi16(_mm256_add_epi16(hi16, _mm256_srli_epi16(hi16, 1)), 7);
    __m256i packed = _mm256_packus_epi16(lo16, hi16);
    return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
}

// Multiply 32 unsigned bytes by a constant via 16-bit path (result truncated to u8).
static inline __m256i mul_u8_const(__m256i x, uint8_t c) {
    __m128i lo128 = _mm256_castsi256_si128(x);
    __m128i hi128 = _mm256_extracti128_si256(x, 1);
    __m256i lo16 = _mm256_cvtepu8_epi16(lo128);
    __m256i hi16 = _mm256_cvtepu8_epi16(hi128);
    __m256i vc = _mm256_set1_epi16(c);
    lo16 = _mm256_and_si256(_mm256_mullo_epi16(lo16, vc), _mm256_set1_epi16(0xFF));
    hi16 = _mm256_and_si256(_mm256_mullo_epi16(hi16, vc), _mm256_set1_epi16(0xFF));
    __m256i packed = _mm256_packus_epi16(lo16, hi16);
    return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
}

void bn_quant_tq1_avx2_range(void *ctx, int row_start, int row_end) {
    BnTQ1SdotCtx *c = (BnTQ1SdotCtx *)ctx;
    const BnBlockTQ1 *blocks = (const BnBlockTQ1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float combined_scale = c->combined_scale;
    const int8_t *x_q = c->x_q;

    const __m256i one = _mm256_set1_epi8(1);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 4), _MM_HINT_T0);
            _mm_prefetch((const char *)(blk + 8), _MM_HINT_T1);
            float d = bn_fp16_to_fp32(blk->d);
            const int8_t *xb = x_q + b * BN_QK_K;

            __m256i iacc = _mm256_setzero_si256();

            // Section 1: qs[0..31] → 160 values (5 trits × 32 bytes)
            {
                __m256i r = _mm256_loadu_si256((const __m256i *)blk->qs);

                // Trit 0: decode(r) - 1
                iacc = bn_avx2_dpbusd(iacc,
                    _mm256_sub_epi8(tq1_decode_avx2(r), one),
                    _mm256_loadu_si256((const __m256i *)(xb + 0)));

                // Trit 1: decode(r*3) - 1
                __m256i m1 = mul_u8_const(r, 3);
                iacc = bn_avx2_dpbusd(iacc,
                    _mm256_sub_epi8(tq1_decode_avx2(m1), one),
                    _mm256_loadu_si256((const __m256i *)(xb + 32)));

                // Trit 2: decode(r*9) - 1
                __m256i m2 = mul_u8_const(r, 9);
                iacc = bn_avx2_dpbusd(iacc,
                    _mm256_sub_epi8(tq1_decode_avx2(m2), one),
                    _mm256_loadu_si256((const __m256i *)(xb + 64)));

                // Trit 3: decode(r*27) - 1
                __m256i m3 = mul_u8_const(r, 27);
                iacc = bn_avx2_dpbusd(iacc,
                    _mm256_sub_epi8(tq1_decode_avx2(m3), one),
                    _mm256_loadu_si256((const __m256i *)(xb + 96)));

                // Trit 4: decode(r*81) - 1
                __m256i m4 = mul_u8_const(r, 81);
                iacc = bn_avx2_dpbusd(iacc,
                    _mm256_sub_epi8(tq1_decode_avx2(m4), one),
                    _mm256_loadu_si256((const __m256i *)(xb + 128)));
            }

            // Section 2: qs[32..47] → 80 values (5 trits × 16 bytes)
            // Use 128-bit loads padded to 256-bit (upper half zeroed)
            {
                __m128i r128 = _mm_loadu_si128((const __m128i *)(blk->qs + 32));
                __m256i r = _mm256_castsi128_si256(r128);

                #define TQ1_S2(mx, off) do { \
                    __m256i t = _mm256_sub_epi8(tq1_decode_avx2(mx), one); \
                    __m256i xv = _mm256_castsi128_si256( \
                        _mm_loadu_si128((const __m128i *)(xb + (off)))); \
                    iacc = bn_avx2_dpbusd(iacc, t, xv); \
                } while(0)

                TQ1_S2(r, 160);
                TQ1_S2(mul_u8_const(r, 3), 176);
                TQ1_S2(mul_u8_const(r, 9), 192);
                TQ1_S2(mul_u8_const(r, 27), 208);
                TQ1_S2(mul_u8_const(r, 81), 224);

                #undef TQ1_S2
            }

            // Section 3: qh[4] → 16 values (scalar, same as NEON)
            int32_t qh_dot = 0;
            static const uint8_t pow3s[] = {1, 3, 9, 27};
            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3s[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    qh_dot += (xi - 1) * (int)xb[240 + n*4 + m];
                }
            }

            row_sum += d * (float)(bn_avx2_hsum_epi32(iacc) + qh_dot);
        }
        c->out[row] = row_sum * combined_scale;
    }
}
