#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_bf16_wasm_range(void *ctx, int row_start, int row_end) {
    BnBF16Ctx *c = (BnBF16Ctx *)ctx;
    const uint16_t *data = (const uint16_t *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint16_t *w = data + (size_t)row * cols;
        v128_t acc0 = wasm_f32x4_splat(0);
        v128_t acc1 = wasm_f32x4_splat(0);
        v128_t acc2 = wasm_f32x4_splat(0);
        v128_t acc3 = wasm_f32x4_splat(0);

        int col = 0;
        for (; col + 15 < cols; col += 16) {
            // Decode 16 BF16 values to F32 via scalar, then load as SIMD
            float tmp[16];
            for (int i = 0; i < 16; i++) {
                uint32_t bits = (uint32_t)w[col + i] << 16;
                memcpy(&tmp[i], &bits, 4);
            }
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(&tmp[0]),  wasm_v128_load(x + col)));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(&tmp[4]),  wasm_v128_load(x + col + 4)));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(&tmp[8]),  wasm_v128_load(x + col + 8)));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(&tmp[12]), wasm_v128_load(x + col + 12)));
        }

        v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
        float row_sum = bn_wasm_hsum_f32x4(sum);

        // Scalar tail
        for (; col < cols; col++) {
            uint32_t bits = (uint32_t)w[col] << 16;
            float wf;
            memcpy(&wf, &bits, 4);
            row_sum += wf * x[col];
        }

        c->out[row] = row_sum;
    }
}
