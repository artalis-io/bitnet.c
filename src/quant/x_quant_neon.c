#include "quant_internal.h"
#include <arm_neon.h>
#include <assert.h>
#include <math.h>

// Quantize float vector x[n] to int8, returning scale = amax/127.
// x_q[n] = round(x[i] / scale), clamped to [-127, 127].
float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n) {
    // Find absolute max via NEON
    float32x4_t vmax = vdupq_n_f32(0);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i)));
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i + 4)));
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i + 8)));
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i + 12)));
    }
    for (; i + 3 < n; i += 4)
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i)));
    float amax = vmaxvq_f32(vmax);
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
    float32x4_t vinv = vdupq_n_f32(inv_scale);

    i = 0;
    for (; i + 15 < n; i += 16) {
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i),      vinv));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4),  vinv));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8),  vinv));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12), vinv));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x4_t s2 = vqmovn_s32(i2);
        int16x4_t s3 = vqmovn_s32(i3);
        int8x8_t  b0 = vqmovn_s16(vcombine_s16(s0, s1));
        int8x8_t  b1 = vqmovn_s16(vcombine_s16(s2, s3));
        vst1_s8(x_q + i,     b0);
        vst1_s8(x_q + i + 8, b1);
    }
    for (; i < n; i++) {
        int v = (int)roundf(x[i] * inv_scale);
        x_q[i] = (int8_t)(v < -BN_I8_MAX ? -BN_I8_MAX : (v > BN_I8_MAX ? BN_I8_MAX : v));
    }
    return scale;
}

// Quantize n_rows of F16 data to INT8 + per-row float scale.
void bn_quant_f16_rows_to_i8(const uint16_t *f16, int8_t *i8_out,
                              float *scales_out, int n_rows, int dim) {
    for (int r = 0; r < n_rows; r++) {
        const uint16_t *row = f16 + (size_t)r * dim;
        int8_t *out = i8_out + (size_t)r * dim;

        // Convert F16->F32 and find amax via NEON
        float32x4_t vmax = vdupq_n_f32(0);
        int d = 0;
        for (; d + 7 < dim; d += 8) {
            float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(row + d));
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
            vmax = vmaxq_f32(vmax, vabsq_f32(lo));
            vmax = vmaxq_f32(vmax, vabsq_f32(hi));
        }
        float amax = vmaxvq_f32(vmax);
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
        float32x4_t vinv = vdupq_n_f32(inv_scale);
        scales_out[r] = scale;

        d = 0;
        for (; d + 15 < dim; d += 16) {
            float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(row + d));
            float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(row + d + 8));
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(h0)), vinv));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(h0)), vinv));
            int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(h1)), vinv));
            int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(h1)), vinv));
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int16x4_t s2 = vqmovn_s32(i2);
            int16x4_t s3 = vqmovn_s32(i3);
            int8x8_t b0 = vqmovn_s16(vcombine_s16(s0, s1));
            int8x8_t b1 = vqmovn_s16(vcombine_s16(s2, s3));
            vst1_s8(out + d, b0);
            vst1_s8(out + d + 8, b1);
        }
        for (; d < dim; d++) {
            float v = bn_fp16_to_fp32(row[d]);
            int q = (int)roundf(v * inv_scale);
            out[d] = (int8_t)(q < -BN_I8_MAX ? -BN_I8_MAX : (q > BN_I8_MAX ? BN_I8_MAX : q));
        }
    }
}

_Static_assert(BN_QK_K % 16 == 0, "BN_QK_K must be a multiple of 16 for NEON");

// Q8_K quantization: 256-element super-blocks with bsums for Q4_K SDOT.
// x_d[n/256]: one float scale per super-block
// x_bsums[n/256 * 16]: int16 sum per 16-element group (for min correction)
void bn_quant_x_to_q8k(const float *x, int8_t *x_q, float *x_d,
                         int16_t *x_bsums, int n) {
    assert(n % BN_QK_K == 0 && "bn_quant_x_to_q8k: n must be multiple of BN_QK_K");
    int n_sb = n / BN_QK_K;
    for (int sb = 0; sb < n_sb; sb++) {
        const float *xb = x + sb * BN_QK_K;
        int8_t *qb = x_q + sb * BN_QK_K;

        // Find amax over 256 elements
        float32x4_t vmax = vdupq_n_f32(0);
        for (int i = 0; i < BN_QK_K; i += 16) {
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xb + i)));
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xb + i + 4)));
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xb + i + 8)));
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xb + i + 12)));
        }
        float amax = vmaxvq_f32(vmax);

        if (amax == 0.0f) {
            memset(qb, 0, BN_QK_K);
            x_d[sb] = 0.0f;
            memset(x_bsums + sb * 16, 0, 16 * sizeof(int16_t));
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_d[sb] = amax / 127.0f;
        float32x4_t vinv = vdupq_n_f32(inv_scale);

        // Quantize 256 elements and compute bsums in one pass
        int16_t *bsums = x_bsums + sb * 16;
        for (int g = 0; g < 16; g++) {
            const float *gx = xb + g * 16;
            int8_t *gq = qb + g * 16;

            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gx),      vinv));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gx + 4),  vinv));
            int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gx + 8),  vinv));
            int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(gx + 12), vinv));

            int8x8_t r0 = vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1)));
            int8x8_t r1 = vqmovn_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3)));
            vst1_s8(gq, r0);
            vst1_s8(gq + 8, r1);

            // bsum: sum of 16 int8 values
            int8x16_t qv = vld1q_s8(gq);
            bsums[g] = (int16_t)vaddlvq_s8(qv);
        }
    }
}

// Per-block Q8_0 quantization for Q4_0 integer dot product path.
// Quantizes each 32-element block independently with its own scale.
void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n) {
    assert(n % 32 == 0 && "bn_quant_x_to_q8_blocks: n must be multiple of 32");
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * 32;
        int8_t *qb = x_q + b * 32;

        float32x4_t v0 = vabsq_f32(vld1q_f32(xb));
        float32x4_t v1 = vabsq_f32(vld1q_f32(xb + 4));
        float32x4_t v2 = vabsq_f32(vld1q_f32(xb + 8));
        float32x4_t v3 = vabsq_f32(vld1q_f32(xb + 12));
        float32x4_t v4 = vabsq_f32(vld1q_f32(xb + 16));
        float32x4_t v5 = vabsq_f32(vld1q_f32(xb + 20));
        float32x4_t v6 = vabsq_f32(vld1q_f32(xb + 24));
        float32x4_t v7 = vabsq_f32(vld1q_f32(xb + 28));
        float32x4_t vmax = vmaxq_f32(vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)),
                                      vmaxq_f32(vmaxq_f32(v4, v5), vmaxq_f32(v6, v7)));
        float amax = vmaxvq_f32(vmax);

        if (amax == 0.0f) {
            memset(qb, 0, 32);
            x_scales[b] = 0.0f;
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_scales[b] = amax / 127.0f;

        float32x4_t vinv = vdupq_n_f32(inv_scale);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb),      vinv));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + 4),  vinv));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + 8),  vinv));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + 12), vinv));
        int32x4_t i4 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + 16), vinv));
        int32x4_t i5 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + 20), vinv));
        int32x4_t i6 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + 24), vinv));
        int32x4_t i7 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + 28), vinv));

        int8x8_t r0 = vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1)));
        int8x8_t r1 = vqmovn_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3)));
        int8x8_t r2 = vqmovn_s16(vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5)));
        int8x8_t r3 = vqmovn_s16(vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7)));
        vst1_s8(qb,      r0);
        vst1_s8(qb + 8,  r1);
        vst1_s8(qb + 16, r2);
        vst1_s8(qb + 24, r3);
    }
}
