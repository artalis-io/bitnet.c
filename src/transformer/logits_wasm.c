#include "transformer_logits_internal.h"

#ifdef __wasm_simd128__

// Vectorized F16→F32: process 4 half-precision values at once.
// Assumes normal values (no subnormals/NaN) — safe for model weights.
static inline void bn_wasm_f16x4_to_f32x4(const uint16_t *src, float *dst) {
    // Load 4 u16 values into low 64 bits of v128
    uint64_t raw;
    memcpy(&raw, src, 8);
    v128_t h = wasm_i64x2_splat((int64_t)raw);

    // Unpack 4 u16 to 4 u32 (zero-extend)
    v128_t u32 = wasm_u32x4_extend_low_u16x8(h);

    // Extract sign, exponent, mantissa
    v128_t sign = wasm_i32x4_shl(wasm_v128_and(u32, wasm_i32x4_splat(0x8000)), 16);
    v128_t exp  = wasm_v128_and(wasm_u32x4_shr(u32, 10), wasm_i32x4_splat(0x1F));
    v128_t mant = wasm_v128_and(u32, wasm_i32x4_splat(0x03FF));

    // Re-bias exponent: FP16 bias=15, FP32 bias=127, delta=112
    v128_t f32_exp = wasm_i32x4_shl(wasm_i32x4_add(exp, wasm_i32x4_splat(112)), 23);
    v128_t f32_mant = wasm_i32x4_shl(mant, 13);

    // Handle zero: if exp==0 && mant==0, result should be sign|0
    v128_t is_zero = wasm_v128_and(wasm_i32x4_eq(exp, wasm_i32x4_splat(0)),
                                    wasm_i32x4_eq(mant, wasm_i32x4_splat(0)));

    v128_t result = wasm_v128_or(sign, wasm_v128_or(f32_exp, f32_mant));
    result = wasm_v128_andnot(result, is_zero);  // zero out non-sign bits if zero
    result = wasm_v128_or(result, wasm_v128_and(sign, is_zero));  // keep sign for zero

    wasm_v128_store(dst, result);
}

void bn_transformer_logits_f16_wasm_range(void *ctx, int v_start, int v_end) {
    BnLogitsCtx *lc = (BnLogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;
    if (dim % 8 != 0) return;  // SIMD alignment guard

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
        for (int d = 0; d < dim; d += 8) {
            float f0[4], f1[4];
            bn_wasm_f16x4_to_f32x4(row + d, f0);
            bn_wasm_f16x4_to_f32x4(row + d + 4, f1);
#ifdef __wasm_relaxed_simd__
            acc0 = wasm_f32x4_relaxed_madd(wasm_v128_load(f0), wasm_v128_load(x + d),     acc0);
            acc1 = wasm_f32x4_relaxed_madd(wasm_v128_load(f1), wasm_v128_load(x + d + 4), acc1);
#else
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(f0), wasm_v128_load(x + d)));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(f1), wasm_v128_load(x + d + 4)));
#endif
        }
        lc->logits[v] = bn_wasm_hsum_f32x4(wasm_f32x4_add(acc0, acc1));
    }
}

#ifdef __wasm_relaxed_simd__
void bn_transformer_logits_i8_wasm_range(void *ctx, int v_start, int v_end) {
    BnLogitsI8Ctx *lc = (BnLogitsI8Ctx *)ctx;
    const int8_t *emb_i8 = lc->emb_i8;
    const float *emb_scales = lc->emb_scales;
    const int8_t *x_q = lc->x_q;
    float x_scale = lc->x_scale;
    int dim = lc->dim;
    if (dim % 64 != 0) return;  // SIMD alignment guard

    for (int v = v_start; v < v_end; v++) {
        const int8_t *row = emb_i8 + (size_t)v * dim;
        v128_t acc0 = wasm_i32x4_splat(0), acc1 = wasm_i32x4_splat(0);
        v128_t acc2 = wasm_i32x4_splat(0), acc3 = wasm_i32x4_splat(0);
        for (int d = 0; d < dim; d += 64) {
            acc0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(row+d),    wasm_v128_load(x_q+d),    acc0);
            acc1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(row+d+16), wasm_v128_load(x_q+d+16), acc1);
            acc2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(row+d+32), wasm_v128_load(x_q+d+32), acc2);
            acc3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(row+d+48), wasm_v128_load(x_q+d+48), acc3);
        }
        v128_t sum4 = wasm_i32x4_add(wasm_i32x4_add(acc0, acc1), wasm_i32x4_add(acc2, acc3));
        int32_t total = wasm_i32x4_extract_lane(sum4, 0) + wasm_i32x4_extract_lane(sum4, 1) +
                        wasm_i32x4_extract_lane(sum4, 2) + wasm_i32x4_extract_lane(sum4, 3);
        lc->logits[v] = (float)total * emb_scales[v] * x_scale;
    }
}
#endif // __wasm_relaxed_simd__

#endif // __wasm_simd128__
