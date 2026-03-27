#include "quant_internal.h"
#include "simd_helpers.h"
#include <immintrin.h>

/* Q4_0 AVX2 tiled matmul: Out[n_tokens][rows] += W[rows][cols] × X[n_tokens][cols]^T
 *
 * Tiled in the token dimension: process TILE_T tokens per pass so the
 * activation tile (TILE_T × 32 bytes per block = TILE_T × n_bpr × 32)
 * stays in L1 cache across all rows. Weight blocks stream from DRAM.
 *
 * For cols=2048, n_bpr=64: tile of 8 tokens = 8 × 64 × 32 = 16KB (fits L1).
 * Without tiling, 80 tokens × 64 × 32 = 160KB (spills to L2).
 */

#define Q4_MATMUL_TILE_T 8

typedef struct {
    float *out;               // [n_tokens * rows]
    const BnQWeight *W;
    const int8_t *x_q;       // [n_tokens * cols]
    const float *x_scales;   // [n_tokens * n_blocks]
    int n_tokens;
    int cols;
} BnQ4MatmulCtx;

void bn_quant_q4_avx2_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / 32;
    int n_tokens = c->n_tokens;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias = _mm_set1_epi8(8);

    for (int t0 = 0; t0 < n_tokens; t0 += Q4_MATMUL_TILE_T) {
        int t_end = t0 + Q4_MATMUL_TILE_T;
        if (t_end > n_tokens) t_end = n_tokens;

        for (int row = row_start; row < row_end; row++) {
            for (int b = 0; b < n_bpr; b++) {
                /* Load and unpack weight block ONCE for all tokens in tile */
                const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_bpr + b];
                float d_q4 = bn_fp16_to_fp32(blk->d);

                __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
                __m128i lo_128 = _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias);
                __m128i hi_128 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias);
                __m256i w256 = _mm256_set_m128i(hi_128, lo_128);

                /* Dot against each token in the tile (x_q tile is in L1) */
                for (int t = t0; t < t_end; t++) {
                    __m256i xq = _mm256_loadu_si256((const __m256i *)(x_q + (size_t)t * cols + b * 32));
                    __m256i acc = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq);
                    float d_x = x_scales[(size_t)t * n_bpr + b];
                    c->out[(size_t)t * rows + row] += d_q4 * d_x * (float)bn_avx2_hsum_epi32(acc);
                }
            }
        }
    }
}
