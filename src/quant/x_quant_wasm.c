#include "quant_internal.h"
#include <wasm_simd128.h>
#include <math.h>
#include <assert.h>

#ifdef __wasm_relaxed_simd__

// Quantize float vector x[n] to int8, returning scale = amax/127.
float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n) {
    // Find absolute max
    v128_t vmax = wasm_f32x4_splat(0);
    v128_t sign_mask = wasm_i32x4_splat(0x7FFFFFFF);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i), sign_mask));
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i + 4), sign_mask));
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i + 8), sign_mask));
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i + 12), sign_mask));
    }
    for (; i + 3 < n; i += 4)
        vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(x + i), sign_mask));

    // Horizontal max
    v128_t shuf = wasm_i32x4_shuffle(vmax, vmax, 2, 3, 0, 1);
    vmax = wasm_f32x4_max(vmax, shuf);
    shuf = wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2);
    vmax = wasm_f32x4_max(vmax, shuf);
    float amax = wasm_f32x4_extract_lane(vmax, 0);

    for (; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }

    if (amax == 0.0f) {
        memset(x_q, 0, n);
        return 0.0f;
    }

    float scale = amax / (float)BN_I8_MAX;
    float inv_scale = (float)BN_I8_MAX / amax;
    v128_t vinv = wasm_f32x4_splat(inv_scale);

    i = 0;
    for (; i + 15 < n; i += 16) {
        // Convert float to int32 (round to nearest)
        v128_t f0 = wasm_f32x4_mul(wasm_v128_load(x + i), vinv);
        v128_t f1 = wasm_f32x4_mul(wasm_v128_load(x + i + 4), vinv);
        v128_t f2 = wasm_f32x4_mul(wasm_v128_load(x + i + 8), vinv);
        v128_t f3 = wasm_f32x4_mul(wasm_v128_load(x + i + 12), vinv);
        v128_t i0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f0));
        v128_t i1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f1));
        v128_t i2 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f2));
        v128_t i3 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f3));
        // Narrow i32 -> i16 -> i8 with saturation
        v128_t s01 = wasm_i16x8_narrow_i32x4(i0, i1);
        v128_t s23 = wasm_i16x8_narrow_i32x4(i2, i3);
        v128_t bytes = wasm_i8x16_narrow_i16x8(s01, s23);
        wasm_v128_store(x_q + i, bytes);
    }
    for (; i < n; i++) {
        int v = (int)roundf(x[i] * inv_scale);
        x_q[i] = (int8_t)(v < -BN_I8_MAX ? -BN_I8_MAX : (v > BN_I8_MAX ? BN_I8_MAX : v));
    }
    return scale;
}

// Per-block Q8_0 quantization for Q4_0 integer dot product path.
void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n) {
    int n_blocks = n / 32;
    v128_t sign_mask = wasm_i32x4_splat(0x7FFFFFFF);
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * 32;
        int8_t *qb = x_q + b * 32;

        v128_t v0 = wasm_v128_and(wasm_v128_load(xb), sign_mask);
        v128_t v1 = wasm_v128_and(wasm_v128_load(xb + 4), sign_mask);
        v128_t v2 = wasm_v128_and(wasm_v128_load(xb + 8), sign_mask);
        v128_t v3 = wasm_v128_and(wasm_v128_load(xb + 12), sign_mask);
        v128_t v4 = wasm_v128_and(wasm_v128_load(xb + 16), sign_mask);
        v128_t v5 = wasm_v128_and(wasm_v128_load(xb + 20), sign_mask);
        v128_t v6 = wasm_v128_and(wasm_v128_load(xb + 24), sign_mask);
        v128_t v7 = wasm_v128_and(wasm_v128_load(xb + 28), sign_mask);
        v128_t vmax = wasm_f32x4_max(wasm_f32x4_max(wasm_f32x4_max(v0, v1), wasm_f32x4_max(v2, v3)),
                                      wasm_f32x4_max(wasm_f32x4_max(v4, v5), wasm_f32x4_max(v6, v7)));
        v128_t shuf = wasm_i32x4_shuffle(vmax, vmax, 2, 3, 0, 1);
        vmax = wasm_f32x4_max(vmax, shuf);
        shuf = wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2);
        vmax = wasm_f32x4_max(vmax, shuf);
        float amax = wasm_f32x4_extract_lane(vmax, 0);

        if (amax == 0.0f) {
            memset(qb, 0, 32);
            x_scales[b] = 0.0f;
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_scales[b] = amax / 127.0f;
        v128_t vinv = wasm_f32x4_splat(inv_scale);

        v128_t i0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb), vinv)));
        v128_t i1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 4), vinv)));
        v128_t i2 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 8), vinv)));
        v128_t i3 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 12), vinv)));
        v128_t i4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 16), vinv)));
        v128_t i5 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 20), vinv)));
        v128_t i6 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 24), vinv)));
        v128_t i7 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + 28), vinv)));

        v128_t s01 = wasm_i16x8_narrow_i32x4(i0, i1);
        v128_t s23 = wasm_i16x8_narrow_i32x4(i2, i3);
        v128_t s45 = wasm_i16x8_narrow_i32x4(i4, i5);
        v128_t s67 = wasm_i16x8_narrow_i32x4(i6, i7);
        wasm_v128_store(qb, wasm_i8x16_narrow_i16x8(s01, s23));
        wasm_v128_store(qb + 16, wasm_i8x16_narrow_i16x8(s45, s67));
    }
}

// Quantize n_rows of F16 data to INT8 + per-row float scale.
void bn_quant_f16_rows_to_i8(const uint16_t *f16, int8_t *i8_out,
                              float *scales_out, int n_rows, int dim) {
    for (int r = 0; r < n_rows; r++) {
        const uint16_t *row = f16 + (size_t)r * dim;
        int8_t *out = i8_out + (size_t)r * dim;

        // Convert F16->F32 and find amax
        v128_t vmax = wasm_f32x4_splat(0);
        v128_t sign_mask = wasm_i32x4_splat(0x7FFFFFFF);
        int d = 0;
        for (; d + 7 < dim; d += 8) {
            float f0 = bn_fp16_to_fp32(row[d]);
            float f1 = bn_fp16_to_fp32(row[d + 1]);
            float f2 = bn_fp16_to_fp32(row[d + 2]);
            float f3 = bn_fp16_to_fp32(row[d + 3]);
            float f4 = bn_fp16_to_fp32(row[d + 4]);
            float f5 = bn_fp16_to_fp32(row[d + 5]);
            float f6 = bn_fp16_to_fp32(row[d + 6]);
            float f7 = bn_fp16_to_fp32(row[d + 7]);
            v128_t lo = wasm_f32x4_make(f0, f1, f2, f3);
            v128_t hi = wasm_f32x4_make(f4, f5, f6, f7);
            vmax = wasm_f32x4_max(vmax, wasm_v128_and(lo, sign_mask));
            vmax = wasm_f32x4_max(vmax, wasm_v128_and(hi, sign_mask));
        }
        // Horizontal max
        v128_t shuf = wasm_i32x4_shuffle(vmax, vmax, 2, 3, 0, 1);
        vmax = wasm_f32x4_max(vmax, shuf);
        shuf = wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2);
        vmax = wasm_f32x4_max(vmax, shuf);
        float amax = wasm_f32x4_extract_lane(vmax, 0);
        for (; d < dim; d++) {
            float v = bn_fp16_to_fp32(row[d]);
            float a = v < 0 ? -v : v;
            if (a > amax) amax = a;
        }

        if (amax == 0.0f) {
            memset(out, 0, dim);
            scales_out[r] = 0.0f;
            continue;
        }

        float scale = amax / (float)BN_I8_MAX;
        float inv_scale = (float)BN_I8_MAX / amax;
        v128_t vinv = wasm_f32x4_splat(inv_scale);
        scales_out[r] = scale;

        d = 0;
        for (; d + 15 < dim; d += 16) {
            float fa[16];
            for (int k = 0; k < 16; k++)
                fa[k] = bn_fp16_to_fp32(row[d + k]);
            v128_t f0 = wasm_f32x4_mul(wasm_v128_load(fa), vinv);
            v128_t f1 = wasm_f32x4_mul(wasm_v128_load(fa + 4), vinv);
            v128_t f2 = wasm_f32x4_mul(wasm_v128_load(fa + 8), vinv);
            v128_t f3 = wasm_f32x4_mul(wasm_v128_load(fa + 12), vinv);
            v128_t i0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f0));
            v128_t i1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f1));
            v128_t i2 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f2));
            v128_t i3 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f3));
            v128_t s01 = wasm_i16x8_narrow_i32x4(i0, i1);
            v128_t s23 = wasm_i16x8_narrow_i32x4(i2, i3);
            v128_t bytes = wasm_i8x16_narrow_i16x8(s01, s23);
            wasm_v128_store(out + d, bytes);
        }
        for (; d < dim; d++) {
            float v = bn_fp16_to_fp32(row[d]);
            int q = (int)roundf(v * inv_scale);
            out[d] = (int8_t)(q < -BN_I8_MAX ? -BN_I8_MAX : (q > BN_I8_MAX ? BN_I8_MAX : q));
        }
    }
}

// Q8_K quantization: 256-element super-blocks with bsums for K-quant SDOT.
// x_d[n/256]: one float scale per super-block
// x_bsums[n/256 * 16]: int16 sum per 16-element group (for min correction)
void bn_quant_x_to_q8k(const float *x, int8_t *x_q, float *x_d,
                         int16_t *x_bsums, int n) {
    assert(n % BN_QK_K == 0 && "bn_quant_x_to_q8k: n must be multiple of BN_QK_K");
    int n_sb = n / BN_QK_K;
    v128_t sign_mask = wasm_i32x4_splat(0x7FFFFFFF);

    for (int sb = 0; sb < n_sb; sb++) {
        const float *xb = x + sb * BN_QK_K;
        int8_t *qb = x_q + sb * BN_QK_K;

        // Find amax over 256 elements
        v128_t vmax = wasm_f32x4_splat(0);
        for (int i = 0; i < BN_QK_K; i += 16) {
            vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(xb + i), sign_mask));
            vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(xb + i + 4), sign_mask));
            vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(xb + i + 8), sign_mask));
            vmax = wasm_f32x4_max(vmax, wasm_v128_and(wasm_v128_load(xb + i + 12), sign_mask));
        }
        // Horizontal max
        v128_t shuf = wasm_i32x4_shuffle(vmax, vmax, 2, 3, 0, 1);
        vmax = wasm_f32x4_max(vmax, shuf);
        shuf = wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2);
        vmax = wasm_f32x4_max(vmax, shuf);
        float amax = wasm_f32x4_extract_lane(vmax, 0);

        if (amax == 0.0f) {
            memset(qb, 0, BN_QK_K);
            x_d[sb] = 0.0f;
            memset(x_bsums + sb * 16, 0, 16 * sizeof(int16_t));
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_d[sb] = amax / 127.0f;
        v128_t vinv = wasm_f32x4_splat(inv_scale);

        // Quantize 256 elements and compute bsums in one pass
        int16_t *bsums = x_bsums + sb * 16;
        for (int g = 0; g < 16; g++) {
            const float *gx = xb + g * 16;
            int8_t *gq = qb + g * 16;

            v128_t i0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(gx), vinv)));
            v128_t i1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(gx + 4), vinv)));
            v128_t i2 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(gx + 8), vinv)));
            v128_t i3 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(gx + 12), vinv)));

            // Narrow i32 -> i16 -> i8 with saturation
            v128_t s01 = wasm_i16x8_narrow_i32x4(i0, i1);
            v128_t s23 = wasm_i16x8_narrow_i32x4(i2, i3);
            v128_t bytes = wasm_i8x16_narrow_i16x8(s01, s23);
            wasm_v128_store(gq, bytes);

            // bsum: sum of 16 int8 values (widen to i16 and sum)
            v128_t lo16 = wasm_i16x8_extend_low_i8x16(bytes);
            v128_t hi16 = wasm_i16x8_extend_high_i8x16(bytes);
            v128_t sum16 = wasm_i16x8_add(lo16, hi16);
            // Horizontal sum of 8 int16 lanes
            v128_t shuf1 = wasm_i32x4_shuffle(sum16, sum16, 2, 3, 0, 1);
            v128_t sum2 = wasm_i16x8_add(sum16, shuf1);
            v128_t shuf2 = wasm_i32x4_shuffle(sum2, sum2, 1, 0, 3, 2);
            v128_t sum3 = wasm_i16x8_add(sum2, shuf2);
            int32_t pair = wasm_i32x4_extract_lane(sum3, 0);
            bsums[g] = (int16_t)((pair & 0xFFFF) + (pair >> 16));
        }
    }
}

#endif // __wasm_relaxed_simd__
