#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_tq2_wasm_range(void *ctx, int row_start, int row_end) {
    BnTQ2Ctx *c = (BnTQ2Ctx *)ctx;
    const BnBlockTQ2 *blocks = (const BnBlockTQ2 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float tensor_scale = c->W->scale;
    const float *x = c->x;

    const v128_t mask3 = wasm_i8x16_splat(3);
    const v128_t one = wasm_i8x16_splat(1);

    for (int row = row_start; row < row_end; row++) {
        v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
        v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ2 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;

            v128_t bacc0 = wasm_f32x4_splat(0), bacc1 = wasm_f32x4_splat(0);
            v128_t bacc2 = wasm_f32x4_splat(0), bacc3 = wasm_f32x4_splat(0);

            for (int half = 0; half < 2; half++) {
                const uint8_t *qs = blk->qs + half * 32;
                const float *xh = xb + half * 128;

                for (int i = 0; i < 2; i++) {
                    v128_t raw = wasm_v128_load(qs + i * 16);
                    const float *xp = xh + i * 16;

                    v128_t t0 = wasm_i8x16_sub(wasm_v128_and(raw, mask3), one);
                    v128_t t1 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 2), mask3), one);
                    v128_t t2 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 4), mask3), one);
                    v128_t t3 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 6), mask3), one);

                    // Widen int8 → int32, convert to float, multiply-accumulate
                    #define TQ2_WASM_ACC(ternary, xbase) do { \
                        v128_t lo16 = wasm_i16x8_extend_low_i8x16(ternary);  \
                        v128_t hi16 = wasm_i16x8_extend_high_i8x16(ternary); \
                        v128_t i0_ = wasm_i32x4_extend_low_i16x8(lo16);      \
                        v128_t i1_ = wasm_i32x4_extend_high_i16x8(lo16);     \
                        v128_t i2_ = wasm_i32x4_extend_low_i16x8(hi16);      \
                        v128_t i3_ = wasm_i32x4_extend_high_i16x8(hi16);     \
                        bacc0 = wasm_f32x4_add(bacc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i0_), wasm_v128_load(xbase)));     \
                        bacc1 = wasm_f32x4_add(bacc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i1_), wasm_v128_load(xbase + 4))); \
                        bacc2 = wasm_f32x4_add(bacc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i2_), wasm_v128_load(xbase + 8))); \
                        bacc3 = wasm_f32x4_add(bacc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i3_), wasm_v128_load(xbase + 12))); \
                    } while(0)

                    TQ2_WASM_ACC(t0, xp + 0*32);
                    TQ2_WASM_ACC(t1, xp + 1*32);
                    TQ2_WASM_ACC(t2, xp + 2*32);
                    TQ2_WASM_ACC(t3, xp + 3*32);
                    #undef TQ2_WASM_ACC
                }
            }

            // Scale by block d and accumulate
            v128_t vd = wasm_f32x4_splat(d);
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(bacc0, vd));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(bacc1, vd));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(bacc2, vd));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(bacc3, vd));
        }

        v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
        c->out[row] = bn_wasm_hsum_f32x4(sum) * tensor_scale;
    }
}
