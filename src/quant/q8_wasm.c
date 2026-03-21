#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q8_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ8Ctx *c = (BnQ8Ctx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);
            for (int i = 0; i < 2; i++) {
                v128_t w = wasm_v128_load(blk->qs + i * 16);
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(w);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(w);
                const float *xp = xb + i * 16;
#ifdef __wasm_relaxed_simd__
                acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)),  wasm_v128_load(xp),      acc0);
                acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xp + 4),  acc1);
                acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)),  wasm_v128_load(xp + 8),  acc2);
                acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xp + 12), acc3);
#else
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xp)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xp + 4)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xp + 8)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xp + 12)));
#endif
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum) * d;
        }
        c->out[row] = row_sum;
    }
}