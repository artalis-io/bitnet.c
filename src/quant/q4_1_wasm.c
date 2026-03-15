#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q4_1_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ4_1Ctx *c = (BnQ4_1Ctx *)ctx;
    const BnBlockQ4_1 *blocks = (const BnBlockQ4_1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_1 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float m = bn_fp16_to_fp32(blk->m);
            const float *xb = x + b * 32;

            v128_t acc = wasm_f32x4_splat(0);
            v128_t xacc = wasm_f32x4_splat(0);

            float tmp[32];
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                tmp[i]      = (float)(byte & 0xF);
                tmp[i + 16] = (float)(byte >> 4);
            }

            for (int i = 0; i < 32; i += 4) {
                v128_t w = wasm_v128_load(tmp + i);
                v128_t xv = wasm_v128_load(xb + i);
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(w, xv));
                xacc = wasm_f32x4_add(xacc, xv);
            }

            row_sum += bn_wasm_hsum_f32x4(acc) * d + bn_wasm_hsum_f32x4(xacc) * m;
        }
        c->out[row] = row_sum;
    }
}
