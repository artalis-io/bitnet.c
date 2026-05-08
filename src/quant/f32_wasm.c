#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_f32_wasm_range(void *ctx, int row_start, int row_end) {
    BnF32Ctx *c = (BnF32Ctx *)ctx;
    const float *data = (const float *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const float *w = data + (size_t)row * cols;
        v128_t acc0 = wasm_f32x4_splat(0);
        v128_t acc1 = wasm_f32x4_splat(0);
        v128_t acc2 = wasm_f32x4_splat(0);
        v128_t acc3 = wasm_f32x4_splat(0);

        int col = 0;
        for (; col + 15 < cols; col += 16) {
#ifdef __wasm_relaxed_simd__
            acc0 = wasm_f32x4_relaxed_madd(wasm_v128_load(w + col),      wasm_v128_load(x + col),      acc0);
            acc1 = wasm_f32x4_relaxed_madd(wasm_v128_load(w + col + 4),  wasm_v128_load(x + col + 4),  acc1);
            acc2 = wasm_f32x4_relaxed_madd(wasm_v128_load(w + col + 8),  wasm_v128_load(x + col + 8),  acc2);
            acc3 = wasm_f32x4_relaxed_madd(wasm_v128_load(w + col + 12), wasm_v128_load(x + col + 12), acc3);
#else
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(w + col),      wasm_v128_load(x + col)));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(w + col + 4),  wasm_v128_load(x + col + 4)));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(w + col + 8),  wasm_v128_load(x + col + 8)));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(w + col + 12), wasm_v128_load(x + col + 12)));
#endif
        }

        v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
        float row_sum = bn_wasm_hsum_f32x4(sum);
        for (; col < cols; col++) {
            row_sum += w[col] * x[col];
        }
        c->out[row] = row_sum;
    }
}
