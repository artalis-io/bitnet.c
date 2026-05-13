#include "quant_ctx.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_i2s_wasm_range(void *ctx, int row_start, int row_end) {
    BnI2SFloatCtx *c = (BnI2SFloatCtx *)ctx;
    int cols = c->W->cols;
    int row_bytes = cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float scale = c->W->scale;
    const float *x = c->x;

    const v128_t mask3 = wasm_i8x16_splat(3);
    const v128_t one = wasm_i8x16_splat(1);

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;

        v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
        v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

        while (done < cols) {
            for (int h = 0; h < 2; h++) {
                v128_t raw = wasm_v128_load(rd + h * 16);

                v128_t t0 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 6), mask3), one);
                v128_t t1 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 4), mask3), one);
                v128_t t2 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 2), mask3), one);
                v128_t t3 = wasm_i8x16_sub(wasm_v128_and(raw, mask3), one);

                const float *xp = x + done + h * 16;
#ifdef __wasm_relaxed_simd__
                #define WASM_ACC_I8x16(ternary, xbase, facc0, facc1, facc2, facc3) do { \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(ternary);  \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(ternary); \
                    v128_t i0 = wasm_i32x4_extend_low_i16x8(lo16);      \
                    v128_t i1 = wasm_i32x4_extend_high_i16x8(lo16);     \
                    v128_t i2 = wasm_i32x4_extend_low_i16x8(hi16);      \
                    v128_t i3 = wasm_i32x4_extend_high_i16x8(hi16);     \
                    facc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(i0), wasm_v128_load(xbase),      facc0); \
                    facc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(i1), wasm_v128_load(xbase + 4),  facc1); \
                    facc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(i2), wasm_v128_load(xbase + 8),  facc2); \
                    facc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(i3), wasm_v128_load(xbase + 12), facc3); \
                } while(0)
#else
                #define WASM_ACC_I8x16(ternary, xbase, facc0, facc1, facc2, facc3) do { \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(ternary);  \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(ternary); \
                    v128_t i0 = wasm_i32x4_extend_low_i16x8(lo16);      \
                    v128_t i1 = wasm_i32x4_extend_high_i16x8(lo16);     \
                    v128_t i2 = wasm_i32x4_extend_low_i16x8(hi16);      \
                    v128_t i3 = wasm_i32x4_extend_high_i16x8(hi16);     \
                    facc0 = wasm_f32x4_add(facc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i0), wasm_v128_load(xbase)));     \
                    facc1 = wasm_f32x4_add(facc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i1), wasm_v128_load(xbase + 4))); \
                    facc2 = wasm_f32x4_add(facc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i2), wasm_v128_load(xbase + 8))); \
                    facc3 = wasm_f32x4_add(facc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i3), wasm_v128_load(xbase + 12))); \
                } while(0)
#endif

                WASM_ACC_I8x16(t0, xp + 0*32,  acc0, acc1, acc2, acc3);
                WASM_ACC_I8x16(t1, xp + 1*32,  acc0, acc1, acc2, acc3);
                WASM_ACC_I8x16(t2, xp + 2*32,  acc0, acc1, acc2, acc3);
                WASM_ACC_I8x16(t3, xp + 3*32,  acc0, acc1, acc2, acc3);
                #undef WASM_ACC_I8x16
            }
            rd += 32;
            done += 128;
        }

        v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
        c->out[row] = bn_wasm_hsum_f32x4(sum) * scale;
    }
}

// Relaxed SIMD SDOT path — mirrors NEON SDOT: integer dot product on int8 ternary × int8 x
#ifdef __wasm_relaxed_simd__
void bn_quant_i2s_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnI2SCtx *c = (BnI2SCtx *)ctx;
    int cols = c->W->cols;
    int row_bytes = cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float combined_scale = c->combined_scale;
    const int8_t *x_q = c->x_q;

    const v128_t mask3 = wasm_i8x16_splat(3);
    const v128_t one = wasm_i8x16_splat(1);

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;

        v128_t iaccA = wasm_i32x4_splat(0), iaccB = wasm_i32x4_splat(0);
        v128_t iaccC = wasm_i32x4_splat(0), iaccD = wasm_i32x4_splat(0);

        while (done < cols) {
            for (int h = 0; h < 2; h++) {
                v128_t raw = wasm_v128_load(rd + h * 16);
                const int8_t *xp = x_q + done + h * 16;

                v128_t t0 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 6), mask3), one);
                v128_t t1 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 4), mask3), one);
                v128_t t2 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 2), mask3), one);
                v128_t t3 = wasm_i8x16_sub(wasm_v128_and(raw, mask3), one);

                // relaxed_dot: s8×s8 → 4 i32 sums (like ARM SDOT)
                iaccA = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(t0, wasm_v128_load(xp), iaccA);
                iaccB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(t1, wasm_v128_load(xp + 32), iaccB);
                iaccC = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(t2, wasm_v128_load(xp + 64), iaccC);
                iaccD = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(t3, wasm_v128_load(xp + 96), iaccD);
            }
            rd += 32;
            done += 128;
        }

        v128_t sum4 = wasm_i32x4_add(wasm_i32x4_add(iaccA, iaccB), wasm_i32x4_add(iaccC, iaccD));
        // Horizontal sum of 4 int32 lanes
        v128_t shuf = wasm_i32x4_shuffle(sum4, sum4, 2, 3, 0, 1);
        sum4 = wasm_i32x4_add(sum4, shuf);
        shuf = wasm_i32x4_shuffle(sum4, sum4, 1, 0, 3, 2);
        sum4 = wasm_i32x4_add(sum4, shuf);
        int32_t total = wasm_i32x4_extract_lane(sum4, 0);

        c->out[row] = (float)total * combined_scale;
    }
}
#else
// Stub when relaxed SIMD not available (function declared but not called)
void bn_quant_i2s_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}
#endif
