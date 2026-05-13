#include "quant_ctx.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <string.h>
#include <wasm_simd128.h>

void bn_quant_iq3xxs_wasm_range(void *ctx, int row_start, int row_end) {
    BnIQ3XXSCtx *c = (BnIQ3XXSCtx *)ctx;
    const BnBlockIQ3XXS *blocks = (const BnBlockIQ3XXS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ3XXS *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *qs = blk->qs;
            const uint8_t *scales_and_signs = qs + BN_QK_K / 4;
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
                uint32_t aux32;
                memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                v128_t vdb = wasm_f32x4_splat(db);

                // Scalar decode to float buffer
                float tmp[32];
                for (int l = 0; l < 4; l++) {
                    const uint8_t signs = bn_ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7F];
                    const uint8_t *grid1 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 0]];
                    const uint8_t *grid2 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 1]];

                    for (int j = 0; j < 4; j++) {
                        float w1 = (float)grid1[j];
                        float w2 = (float)grid2[j];
                        if (signs & bn_kmask_iq2xs[j + 0]) w1 = -w1;
                        if (signs & bn_kmask_iq2xs[j + 4]) w2 = -w2;
                        tmp[l * 8 + j + 0] = w1;
                        tmp[l * 8 + j + 4] = w2;
                    }
                }

                // WASM SIMD: multiply decoded weights by scale, then FMA with x
                for (int g = 0; g < 32; g += 16) {
                    v128_t w0 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 0), vdb);
                    v128_t w1 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 4), vdb);
                    v128_t w2 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 8), vdb);
                    v128_t w3 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 12), vdb);
#ifdef __wasm_relaxed_simd__
                    acc0 = wasm_f32x4_relaxed_madd(w0, wasm_v128_load(xb + g + 0), acc0);
                    acc1 = wasm_f32x4_relaxed_madd(w1, wasm_v128_load(xb + g + 4), acc1);
                    acc2 = wasm_f32x4_relaxed_madd(w2, wasm_v128_load(xb + g + 8), acc2);
                    acc3 = wasm_f32x4_relaxed_madd(w3, wasm_v128_load(xb + g + 12), acc3);
#else
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(w0, wasm_v128_load(xb + g + 0)));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(w1, wasm_v128_load(xb + g + 4)));
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(w2, wasm_v128_load(xb + g + 8)));
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(w3, wasm_v128_load(xb + g + 12)));
#endif
                }

                qs += 8;
                xb += 32;
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
