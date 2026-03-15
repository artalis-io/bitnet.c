#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_iq2s_wasm_range(void *ctx, int row_start, int row_end) {
    BnIQ2SCtx *c = (BnIQ2SCtx *)ctx;
    const BnBlockIQ2S *blocks = (const BnBlockIQ2S *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            float tmp[BN_QK_K];
            bn_quant_dequant_iq2s(&blocks[row * n_blocks_per_row + b], tmp);
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);
            for (int i = 0; i < BN_QK_K; i += 16) {
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(tmp + i),      wasm_v128_load(xb + i)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(tmp + i + 4),  wasm_v128_load(xb + i + 4)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(tmp + i + 8),  wasm_v128_load(xb + i + 8)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(tmp + i + 12), wasm_v128_load(xb + i + 12)));
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
