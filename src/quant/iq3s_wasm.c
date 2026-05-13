#include "quant_ctx.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <wasm_simd128.h>

void bn_quant_iq3s_wasm_range(void *ctx, int row_start, int row_end) {
    BnIQ3SCtx *c = (BnIQ3SCtx *)ctx;
    const BnBlockIQ3S *blocks = (const BnBlockIQ3S *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ3S *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
                uint8_t sc_byte = blk->scales[ib32 / 2];
                int sc_nib = (sc_byte >> ((ib32 & 1) * 4)) & 0xF;
                float dl = d * (1 + 2 * sc_nib);
                v128_t vdl = wasm_f32x4_splat(dl);

                float tmp[32];
                for (int l = 0; l < 8; l++) {
                    int idx9 = blk->qs[ib32 * 8 + l] | (((blk->qh[ib32] >> l) & 1) << 8);
                    uint32_t grid_val = bn_iq3s_grid[idx9];
                    const uint8_t *grid = (const uint8_t *)&grid_val;
                    int sign_byte_idx = ib32 * 4 + l / 2;
                    int sign_bit_base = (l % 2) * 4;
                    uint8_t sign_byte = blk->signs[sign_byte_idx];

                    for (int k = 0; k < 4; k++) {
                        float w = (float)grid[k];
                        if ((sign_byte >> (sign_bit_base + k)) & 1) w = -w;
                        tmp[l * 4 + k] = w;
                    }
                }

                for (int g = 0; g < 32; g += 16) {
                    v128_t w0 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 0), vdl);
                    v128_t w1 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 4), vdl);
                    v128_t w2 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 8), vdl);
                    v128_t w3 = wasm_f32x4_mul(wasm_v128_load(tmp + g + 12), vdl);
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
                xb += 32;
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
