#include "quant_ctx.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <immintrin.h>

void bn_quant_iq3s_avx2_q8k_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    const BnBlockIQ3S *blocks = (const BnBlockIQ3S *)c->W->data;
    int nb = c->W->cols / BN_QK_K;
    for (int row = row_start; row < row_end; row++) {
        __m256 total = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const BnBlockIQ3S *w = &blocks[(size_t)row * nb + b];
            __m256i sums[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
            for (int group = 0; group < 8; group++) {
                uint32_t grids[8];
                int8_t signs[32];
                for (int j = 0; j < 8; j++) {
                    int index = w->qs[group * 8 + j] |
                        (((w->qh[group] >> j) & 1) << 8);
                    grids[j] = bn_iq3s_grid[index];
                }
                for (int j = 0; j < 32; j++)
                    signs[j] = (w->signs[group * 4 + j / 8] &
                                (1u << (j % 8))) ? -1 : 1;
                __m256i grid = _mm256_loadu_si256((const __m256i *)grids);
                __m256i sign = _mm256_loadu_si256((const __m256i *)signs);
                __m256i x = _mm256_loadu_si256((const __m256i *)(
                    c->x_q + b * BN_QK_K + group * 32));
                int scale = 1 + 2 * ((w->scales[group / 2] >>
                                      (4 * (group & 1))) & 15);
                __m256i dot = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(grid, _mm256_sign_epi8(x, sign)),
                    _mm256_set1_epi16((int16_t)scale));
                sums[group & 1] = _mm256_add_epi32(sums[group & 1], dot);
            }
            float scale = bn_fp16_to_fp32(w->d) * c->x_d[b];
            total = _mm256_fmadd_ps(_mm256_set1_ps(scale),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sums[0], sums[1])), total);
        }
        c->out[row] = bn_avx2_hsum_ps(total);
    }
}

static inline float iq3s_fp16_to_fp32(uint16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return bn_fp16_to_fp32(h);
#endif
}

void bn_quant_iq3s_avx2_range(void *ctx, int row_start, int row_end) {
    BnIQ3SCtx *c = (BnIQ3SCtx *)ctx;
    const BnBlockIQ3S *blocks = (const BnBlockIQ3S *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;
    const __m256i bit_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i one = _mm256_set1_epi32(1);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ3S *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = iq3s_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;

            __m256 acc = _mm256_setzero_ps();

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
                uint8_t sc_byte = blk->scales[ib32 / 2];
                int sc_nib = (sc_byte >> ((ib32 & 1) * 4)) & 0xF;
                float dl = d * (1 + 2 * sc_nib);
                __m256 vdl = _mm256_set1_ps(dl);

                __m128i q8 = _mm_loadl_epi64(
                    (const __m128i *)(blk->qs + ib32 * 8));
                __m256i indices = _mm256_cvtepu8_epi32(q8);
                __m256i high = _mm256_and_si256(
                    _mm256_srlv_epi32(
                        _mm256_set1_epi32(blk->qh[ib32]), bit_shifts),
                    one);
                indices = _mm256_add_epi32(
                    indices, _mm256_slli_epi32(high, 8));
                __m256i packed_grid = _mm256_i32gather_epi32(
                    (const int *)bn_iq3s_grid, indices, 4);
                __m128i grid_half[2] = {
                    _mm256_castsi256_si128(packed_grid),
                    _mm256_extracti128_si256(packed_grid, 1)
                };

                for (int g = 0; g < 32; g += 8) {
                    int group = g / 8;
                    __m128i grid8 = group & 1
                        ? _mm_srli_si128(grid_half[group / 2], 8)
                        : grid_half[group / 2];
                    __m256i wi = _mm256_cvtepu8_epi32(grid8);
                    __m256i sign_bits = _mm256_and_si256(
                        _mm256_srlv_epi32(
                            _mm256_set1_epi32(
                                blk->signs[ib32 * 4 + group]),
                            bit_shifts),
                        one);
                    __m256i signs = _mm256_sub_epi32(
                        one, _mm256_slli_epi32(sign_bits, 1));
                    __m256 wf = _mm256_cvtepi32_ps(
                        _mm256_sign_epi32(wi, signs));
                    __m256 xf = _mm256_loadu_ps(xb + g);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(wf, vdl), xf, acc);
                }
                xb += 32;
            }
            row_sum += bn_avx2_hsum_ps(acc);
        }
        c->out[row] = row_sum;
    }
}
