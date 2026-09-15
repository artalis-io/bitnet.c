#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

void bn_quant_q4_1_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ4_1Ctx *c = (BnQ4_1Ctx *)ctx;
    const BnBlockQ4_1 *blocks = (const BnBlockQ4_1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_1 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            float m = bn_fp16_to_fp32(blk->m);
            const float *xb = x + b * 32;

            __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
            __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0xF));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0xF));

            __m256 acc = _mm256_setzero_ps();
            __m256 xacc = _mm256_setzero_ps();

            // Lo nibbles (elements 0-15): unsigned quants
            __m256i i32_0 = _mm256_cvtepu8_epi32(lo);
            __m256i i32_1 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(lo, 8));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_0), _mm256_loadu_ps(xb), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_1), _mm256_loadu_ps(xb + 8), acc);
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb));
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb + 8));

            // Hi nibbles (elements 16-31)
            __m256i i32_2 = _mm256_cvtepu8_epi32(hi);
            __m256i i32_3 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(hi, 8));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_2), _mm256_loadu_ps(xb + 16), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_3), _mm256_loadu_ps(xb + 24), acc);
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb + 16));
            xacc = _mm256_add_ps(xacc, _mm256_loadu_ps(xb + 24));

            row_sum += bn_avx2_hsum_ps(acc) * d + bn_avx2_hsum_ps(xacc) * m;
        }
        c->out[row] = row_sum;
    }
}

static inline float q51_fp16_to_fp32(uint16_t value) {
#if defined(__F16C__)
    return _cvtsh_ss(value);
#else
    return bn_fp16_to_fp32(value);
#endif
}

static void q51_quantize_input(const float *input, int nb, int8_t *q,
    float *d, float *s) {
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int b = 0; b < nb; b++) {
        const float *x = input + b * 32;
        __m256 v[4], a = _mm256_setzero_ps();
        for (int k = 0; k < 4; k++) {
            v[k] = _mm256_loadu_ps(x + k * 8);
            a = _mm256_max_ps(a, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v[k]));
        }
        float max = bn_avx2_hmax_ps(a);
        float scale = max / 127.0f;
        __m256 inv = _mm256_set1_ps(max != 0.0f ? 127.0f / max : 0.0f);
        __m256i qi[4], sum = _mm256_setzero_si256();
        for (int k = 0; k < 4; k++) {
            qi[k] = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v[k], inv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            sum = _mm256_add_epi32(sum, qi[k]);
        }
        int lanes[8];
        _mm256_storeu_si256((__m256i *)lanes, sum);
        int total = 0;
        for (int k = 0; k < 8; k++) total += lanes[k];
        d[b] = q51_fp16_to_fp32(bn_fp32_to_fp16(scale));
        s[b] = q51_fp16_to_fp32(bn_fp32_to_fp16(scale * (float)total));
        __m256i packed = _mm256_packs_epi16(
            _mm256_packs_epi32(qi[0], qi[1]), _mm256_packs_epi32(qi[2], qi[3]));
        _mm256_storeu_si256((__m256i *)(q + b * 32),
            _mm256_permutevar8x32_epi32(packed, perm));
    }
}

/* Q5_1 is affine: the dot and offset use separately rounded Q8_1
 * metadata. In particular, s is round_fp16(unrounded_d * sum(q)), not
 * round_fp16(d) * sum(q). Keep eight block-FMA lanes on both x86 ISAs. */
void bn_quant_q5_1_avx2_range(void *ctx, int row_start, int row_end) {
    BnQ5_1Ctx *c = ctx;
    if (row_start >= row_end) return;
    int nb = c->W->cols / 32;
    int8_t q[nb * 32 + 1];
    float d[nb + 1], s[nb + 1];
    q51_quantize_input(c->x, nb, q, d, s);
    const BnBlockQ5_1 *weights = c->W->data;
    const __m256i bytes = _mm256_setr_epi64x(0, 0x0101010101010101LL,
        0x0202020202020202LL, 0x0303030303030303LL);
    const __m256i bits = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    for (int row = row_start; row < row_end; row++) {
        __m256 acc = _mm256_setzero_ps();
        float offset = 0.0f;
        for (int b = 0; b < nb; b++) {
            const BnBlockQ5_1 *w = weights + (size_t)row * nb + b;
            __m128i raw = _mm_loadu_si128((const __m128i *)w->qs);
            __m256i low = _mm256_and_si256(_mm256_set_m128i(
                _mm_srli_epi16(raw, 4), raw), _mm256_set1_epi8(15));
            uint32_t high;
            memcpy(&high, w->qh, sizeof(high));
            __m256i hi = _mm256_shuffle_epi8(_mm256_set1_epi32((int)high), bytes);
            hi = _mm256_and_si256(_mm256_cmpeq_epi8(
                _mm256_or_si256(hi, bits), _mm256_set1_epi8(-1)), _mm256_set1_epi8(16));
            __m256i prod = _mm256_maddubs_epi16(_mm256_or_si256(low, hi),
                _mm256_loadu_si256((const __m256i *)(q + b * 32)));
            __m256 value = _mm256_cvtepi32_ps(_mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
            acc = _mm256_fmadd_ps(value,
                _mm256_set1_ps(q51_fp16_to_fp32(w->d) * d[b]), acc);
            offset = fmaf(q51_fp16_to_fp32(w->m), s[b], offset);
        }
        c->out[row] = bn_avx2_hsum_ps(acc) + offset;
    }
}

void bn_quant_q5_1_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantFloatMatmulCtx *c = ctx;
    if (row_start >= row_end || c->n_tokens <= 0) return;
    int nb = c->cols / 32;
    /* Keep tiled scratch below 42 KiB per worker. Wide matrices retain the
     * original one-token scratch footprint and exact arithmetic. */
    if (nb > 256) {
        for (int t = 0; t < c->n_tokens; t++) {
            BnQ5_1Ctx row = {c->out + (size_t)t * c->W->rows, c->W,
                c->x + (size_t)t * c->cols};
            bn_quant_q5_1_avx2_range(&row, row_start, row_end);
        }
        return;
    }
    int8_t q[4][nb * 32 + 1];
    float d[4][nb + 1], s[4][nb + 1];
    const BnBlockQ5_1 *weights = c->W->data;
    const __m256i bytes = _mm256_setr_epi64x(0, 0x0101010101010101LL,
        0x0202020202020202LL, 0x0303030303030303LL);
    const __m256i bits = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    for (int t0 = 0; t0 < c->n_tokens;) {
        int nt = c->n_tokens - t0;
        if (nt > 4) nt = 4;
        for (int t = 0; t < nt; t++)
            q51_quantize_input(c->x + (size_t)(t0 + t) * c->cols,
                nb, q[t], d[t], s[t]);
        for (int row = row_start; row < row_end; row++) {
            __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps()};
            float offset[4] = {0};
            for (int b = 0; b < nb; b++) {
                const BnBlockQ5_1 *w = weights + (size_t)row * nb + b;
                __m128i raw = _mm_loadu_si128((const __m128i *)w->qs);
                __m256i low = _mm256_and_si256(_mm256_set_m128i(
                    _mm_srli_epi16(raw, 4), raw), _mm256_set1_epi8(15));
                uint32_t high;
                memcpy(&high, w->qh, sizeof(high));
                __m256i hi = _mm256_shuffle_epi8(
                    _mm256_set1_epi32((int)high), bytes);
                hi = _mm256_and_si256(_mm256_cmpeq_epi8(
                    _mm256_or_si256(hi, bits), _mm256_set1_epi8(-1)),
                    _mm256_set1_epi8(16));
                __m256i decoded = _mm256_or_si256(low, hi);
                float wd = q51_fp16_to_fp32(w->d);
                float wm = q51_fp16_to_fp32(w->m);
                for (int t = 0; t < nt; t++) {
                    __m256i prod = _mm256_maddubs_epi16(decoded,
                        _mm256_loadu_si256((const __m256i *)(q[t] + b * 32)));
                    __m256 value = _mm256_cvtepi32_ps(
                        _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
                    acc[t] = _mm256_fmadd_ps(value,
                        _mm256_set1_ps(wd * d[t][b]), acc[t]);
                    offset[t] = fmaf(wm, s[t][b], offset[t]);
                }
            }
            for (int t = 0; t < nt; t++)
                c->out[(size_t)(t0 + t) * c->W->rows + row] =
                    bn_avx2_hsum_ps(acc[t]) + offset[t];
        }
        t0 += nt;
    }
}
