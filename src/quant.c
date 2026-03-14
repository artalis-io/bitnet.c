#include "quant.h"
#include "simd_helpers.h"
#include "gguf.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

// Widen 16 int8 ternary values (-1/0/+1) to float, FMA with 16 floats from x,
// accumulate into 4 float32x4 accumulators.
static inline void neon_acc_i8x16_f32(int8x16_t t, const float *x,
    float32x4_t *a0, float32x4_t *a1, float32x4_t *a2, float32x4_t *a3) {
    int16x8_t lo16 = vmovl_s8(vget_low_s8(t));
    int16x8_t hi16 = vmovl_s8(vget_high_s8(t));
    *a0 = vmlaq_f32(*a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))),  vld1q_f32(x + 0));
    *a1 = vmlaq_f32(*a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vld1q_f32(x + 4));
    *a2 = vmlaq_f32(*a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))),  vld1q_f32(x + 8));
    *a3 = vmlaq_f32(*a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vld1q_f32(x + 12));
}

// Reduce 4 float32x4 accumulators to a single scalar sum (ARMv7-compatible).
static inline float neon_reduce4(float32x4_t a, float32x4_t b,
                                  float32x4_t c, float32x4_t d) {
    float32x4_t s = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d));
    float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#endif // __ARM_NEON

// --- FP16 <-> FP32 conversion ---

float bn_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & BN_FP16_SIGN_MASK) << 16;
    uint32_t exp  = (h >> 10) & BN_FP16_EXP_MASK;
    uint32_t mant = h & BN_FP16_MANT_MASK;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // +/-0
        } else {
            // #37: Subnormal: normalize by shifting mantissa left until hidden bit appears.
            exp = 1;
            while (!(mant & BN_FP16_HIDDEN_BIT)) { mant <<= 1; exp--; }
            mant &= BN_FP16_MANT_MASK;
            f = sign | ((uint32_t)(exp + BN_FP16_EXP_REBIAS) << 23) | ((uint32_t)mant << 13);
        }
    } else if (exp == 31) {
        f = sign | BN_FP32_EXP_INF | ((uint32_t)mant << 13);  // Inf/NaN
    } else {
        f = sign | ((uint32_t)(exp + BN_FP16_EXP_REBIAS) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

uint16_t bn_fp32_to_fp16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);

    uint32_t sign = (f >> 16) & BN_FP16_SIGN_MASK;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & BN_FP32_MANT_MASK;

    if (exp > 15) {
        return (uint16_t)(sign | BN_FP16_INF);  // Inf
    } else if (exp < -14) {
        return (uint16_t)sign;  // Zero (flush subnormals)
    } else {
        return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (mant >> 13));
    }
}

// --- TQ2_0 dequantization ---
// 2-bit packing: 4 values per byte, map {0,1,2} -> {-1,0,+1}

void bn_quant_dequant_tq2(const BnBlockTQ2 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    int idx = 0;

    // Two groups of 32 bytes
    for (int j = 0; j < 64; j += 32) {
        for (int l = 0; l < 4; l++) {
            for (int m = 0; m < 32; m++) {
                int8_t q = (block->qs[j + m] >> (l * 2)) & 3;
                out[idx++] = (float)(q - 1) * d;
            }
        }
    }
}

// --- TQ1_0 dequantization ---
// Base-3 packing: 5 values per byte in qs (240 values), 4 values per byte in qh (16 values)

void bn_quant_dequant_tq1(const BnBlockTQ1 *block, float *out) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    float d = bn_fp16_to_fp32(block->d);
    int idx = 0;

    // Process qs: 48 bytes, in two chunks (32 + 16)
    // First chunk: bytes 0..31, 5 trits each -> 160 values
    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 32; m++) {
            uint8_t q = block->qs[m] * pow3[n];  // uint8 overflow is intentional
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // Second chunk: bytes 32..47, 5 trits each -> 80 values
    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 16; m++) {
            uint8_t q = block->qs[32 + m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // Process qh: 4 bytes, 4 trits each -> 16 values
    for (int n = 0; n < 4; n++) {
        for (int m = 0; m < 4; m++) {
            uint8_t q = block->qh[m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // #35: Assert we produced exactly BN_QK_K values (160 + 80 + 16 = 256)
    assert(idx == BN_QK_K);
}

// --- I2_S dequantization (Microsoft BitNet format) ---
// 2-bit ternary, interleaved byte layout, single per-tensor scale
// Each byte: bits 7-6=subrow0, 5-4=subrow1, 3-2=subrow2, 1-0=subrow3
// Processes 128 elements (4 x 32) per 32-byte chunk

// #36: I2_S uses an interleaved byte layout where each byte contains 2-bit values
// from 4 sub-rows of 32 elements. This means each 128-element chunk always uses
// exactly 32 bytes. Model dimensions are always multiples of 128 in practice.
void bn_quant_dequant_i2s(const uint8_t *data, float *out, int n, float scale) {
    static const float map2bit[4] = { -1.0f, 0.0f, +1.0f, 0.0f };
    int done = 0;

    while (done < n) {
        int blk_e = (n - done >= 128) ? 128 : (n - done);
        int cols0 = blk_e >= 32  ? 32 : blk_e;
        int cols1 = blk_e >= 64  ? 32 : (blk_e > 32  ? blk_e - 32  : 0);
        int cols2 = blk_e >= 96  ? 32 : (blk_e > 64  ? blk_e - 64  : 0);
        int cols3 = blk_e >= 128 ? 32 : (blk_e > 96  ? blk_e - 96  : 0);

        for (int gp = 0; gp < 32; gp++) {
            uint8_t b = data[gp];
            uint8_t c0 = (b >> 6) & 0x3;
            uint8_t c1 = (b >> 4) & 0x3;
            uint8_t c2 = (b >> 2) & 0x3;
            uint8_t c3 = (b >> 0) & 0x3;

            if (gp < cols0) out[done + 0*32 + gp] = scale * map2bit[c0];
            if (gp < cols1) out[done + 1*32 + gp] = scale * map2bit[c1];
            if (gp < cols2) out[done + 2*32 + gp] = scale * map2bit[c2];
            if (gp < cols3) out[done + 3*32 + gp] = scale * map2bit[c3];
        }

        data += 32;
        done += blk_e;
    }
}

// --- Q8_0 dequantization ---
// 32 int8 values per block, FP16 per-block scale

void bn_quant_dequant_q8_0(const BnBlockQ8_0 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    for (int i = 0; i < 32; i++) {
        out[i] = block->qs[i] * d;
    }
}

// --- Q4_0 dequantization ---
// 32 values packed as 16 nibble bytes, FP16 per-block scale
// Low nibble = elements 0-15, high nibble = elements 16-31, centered at 8

void bn_quant_dequant_q4_0(const BnBlockQ4_0 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    for (int i = 0; i < 16; i++) {
        uint8_t b = block->qs[i];
        out[i]      = ((int)(b & 0xF) - 8) * d;
        out[i + 16] = ((int)(b >> 4)  - 8) * d;
    }
}

// --- Q6_K dequantization ---
// 256 values per block: 128 bytes ql (lower 4 bits), 64 bytes qh (upper 2 bits),
// 16 int8 sub-block scales, FP16 super-block scale. 210 bytes total.
// Layout matches ggml: ql/qh/sc pointers advance by 64/32/8 per 128-element chunk.

void bn_quant_dequant_q6k(const BnBlockQ6K *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t  *sc = block->scales;

    for (int n = 0; n < BN_QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l +  0] = d * sc[is + 0] * q1;
            out[l + 32] = d * sc[is + 2] * q2;
            out[l + 64] = d * sc[is + 4] * q3;
            out[l + 96] = d * sc[is + 6] * q4;
        }
        out += 128;
        ql  += 64;
        qh  += 32;
        sc  += 8;
    }
}

// --- SDOT int8 accumulation for I2_S (Apple Silicon / ARMv8.2+ dotprod) ---

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

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
        // Convert 16 floats to 16 int8s
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

        // Convert F16→F32 and find amax via NEON
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

// Context for I2_S SDOT range function
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} I2SCtx;

// I2_S SDOT matvec for a range of rows [row_start, row_end)
static void i2s_sdot_range(void *ctx, int row_start, int row_end) {
    I2SCtx *c = (I2SCtx *)ctx;
    int row_bytes = c->W->cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float combined_scale = c->combined_scale;
    int cols = c->W->cols;
    const int8_t *x_q = c->x_q;

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;

        int32x4_t iaccA = vdupq_n_s32(0), iaccB = vdupq_n_s32(0);
        int32x4_t iaccC = vdupq_n_s32(0), iaccD = vdupq_n_s32(0);
        const int8x16_t one = vdupq_n_s8(1);
        const uint8x16_t mask3 = vdupq_n_u8(3);

        while (done < cols) {
            __builtin_prefetch(rd + 64, 0, 0);
            // First 16 weight bytes -> 64 elements from sub-rows 0-3
            {
                uint8x16_t raw = vld1q_u8(rd);
                const int8_t *xp = x_q + done;
                int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 6)), one);
                int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one);
                int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one);
                int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one);
                iaccA = vdotq_s32(iaccA, t0, vld1q_s8(xp));
                iaccB = vdotq_s32(iaccB, t1, vld1q_s8(xp + 32));
                iaccC = vdotq_s32(iaccC, t2, vld1q_s8(xp + 64));
                iaccD = vdotq_s32(iaccD, t3, vld1q_s8(xp + 96));
            }
            // Second 16 weight bytes -> 64 elements from sub-rows 0-3
            {
                uint8x16_t raw = vld1q_u8(rd + 16);
                const int8_t *xp = x_q + done + 16;
                int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 6)), one);
                int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one);
                int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one);
                int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one);
                iaccA = vdotq_s32(iaccA, t0, vld1q_s8(xp));
                iaccB = vdotq_s32(iaccB, t1, vld1q_s8(xp + 32));
                iaccC = vdotq_s32(iaccC, t2, vld1q_s8(xp + 64));
                iaccD = vdotq_s32(iaccD, t3, vld1q_s8(xp + 96));
            }
            rd += 32;
            done += 128;
        }

        // Reduce int32 accumulators to scalar
        int32x4_t sum4 = vaddq_s32(vaddq_s32(iaccA, iaccB), vaddq_s32(iaccC, iaccD));
        int32_t total = vaddvq_s32(sum4);
        c->out[row] = (float)total * combined_scale;
    }
}

// Per-block Q8_0 quantization for Q4_0 integer dot product path.
// Quantizes each 32-element block independently with its own scale.
void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n) {
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

// Q4_0 SDOT context: integer dot product with per-block scales
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
} Q4SdotCtx;

// Q4_0 SDOT matvec for a range of rows [row_start, row_end)
static void q4_sdot_range(void *ctx, int row_start, int row_end) {
    Q4SdotCtx *c = (Q4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d_q4 = bn_fp16_to_fp32(blk->d);
            float d_q8 = x_scales[b];

            uint8x16_t raw = vld1q_u8(blk->qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);

            const int8_t *xb = x_q + b * 32;
            int32x4_t acc = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));

            row_sum += d_q4 * d_q8 * (float)vaddvq_s32(acc);
        }
        c->out[row] = row_sum;
    }
}

#endif // __ARM_NEON && __ARM_FEATURE_DOTPROD

// --- AVX2 SDOT-equivalent for I2_S ---

#if defined(__AVX2__) && !defined(__ARM_NEON)

float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n) {
    // Find absolute max via AVX2
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int i = 0;
    for (; i + 31 < n; i += 32) {
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask));
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i + 8), sign_mask));
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i + 16), sign_mask));
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i + 24), sign_mask));
    }
    for (; i + 7 < n; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask));
    float amax = bn_avx2_hmax_ps(vmax);
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
    __m256 vinv = _mm256_set1_ps(inv_scale);

    // Lane-crossing fixup permutation for packs: AVX2 packs operates within
    // 128-bit lanes, so after two packs (32→16→8) the order is [0,4,1,5,2,6,3,7]
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vinv));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 16), vinv));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 24), vinv));
        __m256i s01 = _mm256_packs_epi32(i0, i1);   // 32→16, within lanes
        __m256i s23 = _mm256_packs_epi32(i2, i3);
        __m256i b = _mm256_packs_epi16(s01, s23);    // 16→8, within lanes
        b = _mm256_permutevar8x32_epi32(b, perm);    // fix lane crossing
        _mm256_storeu_si256((__m256i *)(x_q + i), b);
    }
    for (; i < n; i++) {
        int v = (int)roundf(x[i] * inv_scale);
        x_q[i] = (int8_t)(v < -BN_I8_MAX ? -BN_I8_MAX : (v > BN_I8_MAX ? BN_I8_MAX : v));
    }
    return scale;
}

void bn_quant_f16_rows_to_i8(const uint16_t *f16, int8_t *i8_out,
                              float *scales_out, int n_rows, int dim) {
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int r = 0; r < n_rows; r++) {
        const uint16_t *row = f16 + (size_t)r * dim;
        int8_t *out = i8_out + (size_t)r * dim;

        // F16C: convert F16→F32 and find amax
        __m256 vmax = _mm256_setzero_ps();
        int d = 0;
        for (; d + 7 < dim; d += 8) {
            __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d)));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, sign_mask));
        }
        float amax = bn_avx2_hmax_ps(vmax);
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
        __m256 vinv = _mm256_set1_ps(inv_scale);
        scales_out[r] = scale;

        d = 0;
        for (; d + 31 < dim; d += 32) {
            __m256 f0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d)));
            __m256 f1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d + 8)));
            __m256 f2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d + 16)));
            __m256 f3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + d + 24)));
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, vinv));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, vinv));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(f2, vinv));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(f3, vinv));
            __m256i s01 = _mm256_packs_epi32(i0, i1);
            __m256i s23 = _mm256_packs_epi32(i2, i3);
            __m256i b = _mm256_packs_epi16(s01, s23);
            b = _mm256_permutevar8x32_epi32(b, perm);
            _mm256_storeu_si256((__m256i *)(out + d), b);
        }
        for (; d < dim; d++) {
            float v = bn_fp16_to_fp32(row[d]);
            int q = (int)roundf(v * inv_scale);
            out[d] = (int8_t)(q < -BN_I8_MAX ? -BN_I8_MAX : (q > BN_I8_MAX ? BN_I8_MAX : q));
        }
    }
}

// Context for I2_S AVX2 range function (same layout as NEON SDOT I2SCtx)
// (I2SCtx is defined inside the NEON DOTPROD guard, so we redefine for AVX2)
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} I2SCtx;

// I2_S AVX2 matvec for a range of rows [row_start, row_end)
static void i2s_avx2_range(void *ctx, int row_start, int row_end) {
    I2SCtx *c = (I2SCtx *)ctx;
    int row_bytes = c->W->cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float combined_scale = c->combined_scale;
    int cols = c->W->cols;
    const int8_t *x_q = c->x_q;

    // I2_S interleaved: each byte has 4 sub-rows × 2 bits
    // Bits 7-6 = sub-row 0, 5-4 = sub-row 1, 3-2 = sub-row 2, 1-0 = sub-row 3
    // Encoding: 0=-1, 1=0, 2=+1 → subtract 1 to get ternary
    const __m256i mask3 = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi8(1);

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;

        __m256i iaccA = _mm256_setzero_si256();
        __m256i iaccB = _mm256_setzero_si256();
        __m256i iaccC = _mm256_setzero_si256();
        __m256i iaccD = _mm256_setzero_si256();

        while (done < cols) {
            _mm_prefetch((const char *)(rd + 64), _MM_HINT_T0);
            // Load 32 packed bytes = 128 ternary values
            __m256i raw = _mm256_loadu_si256((const __m256i *)rd);

            // Extract 4 sub-rows: shift right, mask to 2 bits, subtract 1
            // Note: _mm256_srli_epi16 shifts 16-bit lanes; AND with 0x03
            // cleans any cross-byte contamination from the 16-bit shift.
            __m256i t0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3), one);
            __m256i t1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3), one);
            __m256i t2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3), one);
            __m256i t3 = _mm256_sub_epi8(_mm256_and_si256(raw, mask3), one);

            // Load 4×32 int8 x_q values and accumulate dot products
            const int8_t *xp = x_q + done;
            iaccA = bn_avx2_dpbusd(iaccA, t0, _mm256_loadu_si256((const __m256i *)xp));
            iaccB = bn_avx2_dpbusd(iaccB, t1, _mm256_loadu_si256((const __m256i *)(xp + 32)));
            iaccC = bn_avx2_dpbusd(iaccC, t2, _mm256_loadu_si256((const __m256i *)(xp + 64)));
            iaccD = bn_avx2_dpbusd(iaccD, t3, _mm256_loadu_si256((const __m256i *)(xp + 96)));

            rd += 32;
            done += 128;
        }

        // Reduce int32 accumulators to scalar
        __m256i sum4 = _mm256_add_epi32(_mm256_add_epi32(iaccA, iaccB),
                                         _mm256_add_epi32(iaccC, iaccD));
        int32_t total = bn_avx2_hsum_epi32(sum4);
        c->out[row] = (float)total * combined_scale;
    }
}

// Per-block Q8_0 quantization (AVX2 version)
void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n) {
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    int n_blocks = n / 32;

    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * 32;
        int8_t *qb = x_q + b * 32;

        __m256 v0 = _mm256_and_ps(_mm256_loadu_ps(xb), sign_mask);
        __m256 v1 = _mm256_and_ps(_mm256_loadu_ps(xb + 8), sign_mask);
        __m256 v2 = _mm256_and_ps(_mm256_loadu_ps(xb + 16), sign_mask);
        __m256 v3 = _mm256_and_ps(_mm256_loadu_ps(xb + 24), sign_mask);
        __m256 vmax = _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3));
        float amax = bn_avx2_hmax_ps(vmax);

        if (amax == 0.0f) {
            _mm256_storeu_si256((__m256i *)qb, _mm256_setzero_si256());
            x_scales[b] = 0.0f;
            continue;
        }

        float inv_scale = 127.0f / amax;
        x_scales[b] = amax / 127.0f;

        __m256 vinv = _mm256_set1_ps(inv_scale);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb), vinv));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 8), vinv));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 16), vinv));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 24), vinv));
        __m256i s01 = _mm256_packs_epi32(i0, i1);
        __m256i s23 = _mm256_packs_epi32(i2, i3);
        __m256i packed = _mm256_packs_epi16(s01, s23);
        packed = _mm256_permutevar8x32_epi32(packed, perm);
        _mm256_storeu_si256((__m256i *)qb, packed);
    }
}

// Q4_0 AVX2 DPBUSD context (same layout as NEON Q4SdotCtx)
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
} Q4SdotCtx;

// Q4_0 AVX2 matvec for a range of rows [row_start, row_end)
static void q4_sdot_avx2_range(void *ctx, int row_start, int row_end) {
    Q4SdotCtx *c = (Q4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const __m128i mask_lo = _mm_set1_epi8(0xF);
    const __m128i bias = _mm_set1_epi8(8);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 2), _MM_HINT_T0);
            float d_q4 = bn_fp16_to_fp32(blk->d);
            float d_q8 = x_scales[b];

            __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
            __m128i lo_128 = _mm_sub_epi8(_mm_and_si128(raw, mask_lo), bias);
            __m128i hi_128 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_lo), bias);

            __m256i w256 = _mm256_set_m128i(hi_128, lo_128);
            __m256i xq256 = _mm256_loadu_si256((const __m256i *)(x_q + b * 32));

            __m256i acc = bn_avx2_dpbusd(_mm256_setzero_si256(), w256, xq256);
            row_sum += d_q4 * d_q8 * (float)bn_avx2_hsum_epi32(acc);
        }
        c->out[row] = row_sum;
    }
}

#endif // __AVX2__ && !__ARM_NEON

// --- Range function contexts for non-SDOT paths ---

// I2_S float NEON range context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} I2SFloatCtx;

#if defined(__ARM_NEON) && !defined(__ARM_FEATURE_DOTPROD)
static void i2s_neon_range(void *ctx, int row_start, int row_end) {
    I2SFloatCtx *c = (I2SFloatCtx *)ctx;
    int cols = c->W->cols;
    int row_bytes = cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float scale = c->W->scale;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;
        float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
        float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
        float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
        float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
        const int8x16_t one = vdupq_n_s8(1);
        const uint8x16_t mask3 = vdupq_n_u8(3);
        while (done < cols) {
            __builtin_prefetch(rd + 128, 0, 0);
            __builtin_prefetch(rd + 192, 0, 0);
            for (int h = 0; h < 2; h++) {
                uint8x16_t raw = vld1q_u8(rd + h * 16);
                const float *xp = x + done + h * 16;
                int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 6)), one);
                int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one);
                int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one);
                int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one);
                neon_acc_i8x16_f32(t0, xp + 0*32, &accA0, &accA1, &accA2, &accA3);
                neon_acc_i8x16_f32(t1, xp + 1*32, &accB0, &accB1, &accB2, &accB3);
                neon_acc_i8x16_f32(t2, xp + 2*32, &accA0, &accA1, &accA2, &accA3);
                neon_acc_i8x16_f32(t3, xp + 3*32, &accB0, &accB1, &accB2, &accB3);
            }
            rd += 32;
            done += 128;
        }
        c->out[row] = (neon_reduce4(accA0, accA1, accA2, accA3) +
                    neon_reduce4(accB0, accB1, accB2, accB3)) * scale;
    }
}
#endif // __ARM_NEON && !__ARM_FEATURE_DOTPROD

// I2_S WASM SIMD128 range (float accumulation, 128-bit vectors)
#if defined(__wasm_simd128__)
static void i2s_wasm_range(void *ctx, int row_start, int row_end) {
    I2SFloatCtx *c = (I2SFloatCtx *)ctx;
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
            // Process 16 packed bytes = 64 ternary values at a time
            for (int h = 0; h < 2; h++) {
                v128_t raw = wasm_v128_load(rd + h * 16);

                // Extract 4 sub-rows using byte-granularity shift (no cross-byte issues)
                v128_t t0 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 6), mask3), one);
                v128_t t1 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 4), mask3), one);
                v128_t t2 = wasm_i8x16_sub(wasm_v128_and(wasm_u8x16_shr(raw, 2), mask3), one);
                v128_t t3 = wasm_i8x16_sub(wasm_v128_and(raw, mask3), one);

                // For each sub-row: widen i8→i16→i32→f32, multiply with x, accumulate
                // Process sub-row 0 (16 elements)
                const float *xp = x + done + h * 16;
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
#endif // __wasm_simd128__

// I2_S scalar range
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__wasm_simd128__)
static void i2s_scalar_range(void *ctx, int row_start, int row_end) {
    I2SFloatCtx *c = (I2SFloatCtx *)ctx;
    int cols = c->W->cols;
    int row_bytes = cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float scale = c->W->scale;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;
        const int8_t imap[4] = {-1, 0, 1, 0};
        float sum = 0.0f;
        while (done < cols) {
            for (int gp = 0; gp < 32; gp++) {
                uint8_t b = rd[gp];
                sum += imap[(b >> 6) & 3] * x[done + 0*32 + gp];
                sum += imap[(b >> 4) & 3] * x[done + 1*32 + gp];
                sum += imap[(b >> 2) & 3] * x[done + 2*32 + gp];
                sum += imap[(b >> 0) & 3] * x[done + 3*32 + gp];
            }
            rd += 32;
            done += 128;
        }
        c->out[row] = sum * scale;
    }
}
#endif // !__ARM_NEON && !__AVX2__ && !__wasm_simd128__

// TQ2_0 range context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} TQ2Ctx;

static void tq2_range(void *ctx, int row_start, int row_end) {
    TQ2Ctx *c = (TQ2Ctx *)ctx;
    const BnBlockTQ2 *blocks = (const BnBlockTQ2 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float tensor_scale = c->W->scale;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ2 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;
#ifdef __ARM_NEON
            float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
            float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
            float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
            float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
            const uint8x16_t mask3 = vdupq_n_u8(3);
            const int8x16_t one_s8 = vdupq_n_s8(1);
            for (int half = 0; half < 2; half++) {
                const uint8_t *qs = blk->qs + half * 32;
                const float *xh = xb + half * 128;
                for (int i = 0; i < 2; i++) {
                    uint8x16_t raw = vld1q_u8(qs + i * 16);
                    const float *xp = xh + i * 16;
                    int8x16_t t0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask3)), one_s8);
                    int8x16_t t1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask3)), one_s8);
                    int8x16_t t2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask3)), one_s8);
                    int8x16_t t3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 6), mask3)), one_s8);
                    neon_acc_i8x16_f32(t0, xp + 0*32, &accA0, &accA1, &accA2, &accA3);
                    neon_acc_i8x16_f32(t1, xp + 1*32, &accB0, &accB1, &accB2, &accB3);
                    neon_acc_i8x16_f32(t2, xp + 2*32, &accA0, &accA1, &accA2, &accA3);
                    neon_acc_i8x16_f32(t3, xp + 3*32, &accB0, &accB1, &accB2, &accB3);
                }
            }
            row_sum += (neon_reduce4(accA0, accA1, accA2, accA3) +
                        neon_reduce4(accB0, accB1, accB2, accB3)) * d;
#else
            float block_sum = 0.0f;
            for (int half = 0; half < 2; half++) {
                const uint8_t *qs = blk->qs + half * 32;
                const float *xh = xb + half * 128;
                for (int m = 0; m < 32; m++) {
                    uint8_t byte = qs[m];
                    int8_t q0 = (int8_t)((byte >> 0) & 3) - 1;
                    int8_t q1 = (int8_t)((byte >> 2) & 3) - 1;
                    int8_t q2 = (int8_t)((byte >> 4) & 3) - 1;
                    int8_t q3 = (int8_t)((byte >> 6) & 3) - 1;
                    block_sum += q0 * xh[0*32 + m];
                    block_sum += q1 * xh[1*32 + m];
                    block_sum += q2 * xh[2*32 + m];
                    block_sum += q3 * xh[3*32 + m];
                }
            }
            row_sum += block_sum * d;
#endif
        }
        c->out[row] = row_sum * tensor_scale;
    }
}

// TQ1_0 range context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} TQ1Ctx;

static void tq1_range(void *ctx, int row_start, int row_end) {
    TQ1Ctx *c = (TQ1Ctx *)ctx;
    const BnBlockTQ1 *blocks = (const BnBlockTQ1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float tensor_scale = c->W->scale;
    const float *x = c->x;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            float block_sum = 0.0f;
            const float *xb = x + b * BN_QK_K;
#ifdef __ARM_NEON
            float32x4_t accA0 = vdupq_n_f32(0), accA1 = vdupq_n_f32(0);
            float32x4_t accA2 = vdupq_n_f32(0), accA3 = vdupq_n_f32(0);
            float32x4_t accB0 = vdupq_n_f32(0), accB1 = vdupq_n_f32(0);
            float32x4_t accB2 = vdupq_n_f32(0), accB3 = vdupq_n_f32(0);
            const int8x16_t one_s8 = vdupq_n_s8(1);
            const uint8x8_t three_u8 = vdup_n_u8(3);
            int acc_flip = 0;

            // Section 1: qs[0..31], 5 trits/byte -> 160 values
            for (int n = 0; n < 5; n++) {
                uint8x16_t pow3_vec = vdupq_n_u8(pow3[n]);
                for (int i = 0; i < 2; i++) {
                    uint8x16_t raw = vld1q_u8(blk->qs + i * 16);
                    uint8x16_t q = vmulq_u8(raw, pow3_vec);
                    uint8x8_t xi_lo = vshrn_n_u16(vmull_u8(vget_low_u8(q), three_u8), 8);
                    uint8x8_t xi_hi = vshrn_n_u16(vmull_u8(vget_high_u8(q), three_u8), 8);
                    int8x16_t ternary = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(xi_lo, xi_hi)), one_s8);
                    if (acc_flip++ & 1)
                        neon_acc_i8x16_f32(ternary, xb + n*32 + i*16, &accB0, &accB1, &accB2, &accB3);
                    else
                        neon_acc_i8x16_f32(ternary, xb + n*32 + i*16, &accA0, &accA1, &accA2, &accA3);
                }
            }

            // Section 2: qs[32..47], 5 trits/byte -> 80 values
            for (int n = 0; n < 5; n++) {
                uint8x16_t raw = vld1q_u8(blk->qs + 32);
                uint8x16_t q = vmulq_u8(raw, vdupq_n_u8(pow3[n]));
                uint8x8_t xi_lo = vshrn_n_u16(vmull_u8(vget_low_u8(q), three_u8), 8);
                uint8x8_t xi_hi = vshrn_n_u16(vmull_u8(vget_high_u8(q), three_u8), 8);
                int8x16_t ternary = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(xi_lo, xi_hi)), one_s8);
                if (acc_flip++ & 1)
                    neon_acc_i8x16_f32(ternary, xb + 160 + n*16, &accB0, &accB1, &accB2, &accB3);
                else
                    neon_acc_i8x16_f32(ternary, xb + 160 + n*16, &accA0, &accA1, &accA2, &accA3);
            }

            block_sum = neon_reduce4(accA0, accA1, accA2, accA3) +
                        neon_reduce4(accB0, accB1, accB2, accB3);

            // Section 3: qh[0..3], 4 trits/byte -> 16 values (scalar)
            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[240 + n*4 + m];
                }
            }
#else
            // Section 1: qs[0..31], 5 trits/byte -> 160 values
            for (int n = 0; n < 5; n++) {
                for (int m = 0; m < 32; m++) {
                    uint8_t q = blk->qs[m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[n*32 + m];
                }
            }

            // Section 2: qs[32..47], 5 trits/byte -> 80 values
            for (int n = 0; n < 5; n++) {
                for (int m = 0; m < 16; m++) {
                    uint8_t q = blk->qs[32 + m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[160 + n*16 + m];
                }
            }

            // Section 3: qh[0..3], 4 trits/byte -> 16 values
            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[240 + n*4 + m];
                }
            }
#endif
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum * tensor_scale;
    }
}

// --- Q8_0 matrix-vector multiply ---

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} Q8Ctx;

static void q8_range(void *ctx, int row_start, int row_end) {
    Q8Ctx *c = (Q8Ctx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
#ifdef __ARM_NEON
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
            for (int i = 0; i < 2; i++) {
                int8x16_t w = vld1q_s8(blk->qs + i * 16);
                neon_acc_i8x16_f32(w, xb + i * 16, &acc0, &acc1, &acc2, &acc3);
            }
            row_sum += neon_reduce4(acc0, acc1, acc2, acc3) * d;
#elif defined(__AVX2__)
            __m256i w_raw = _mm256_loadu_si256((const __m256i *)blk->qs);
            __m128i w_lo = _mm256_castsi256_si128(w_raw);
            __m128i w_hi = _mm256_extracti128_si256(w_raw, 1);
            __m256 acc = _mm256_setzero_ps();
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo)), _mm256_loadu_ps(xb)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_lo, 8))), _mm256_loadu_ps(xb + 8)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi)), _mm256_loadu_ps(xb + 16)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_hi, 8))), _mm256_loadu_ps(xb + 24)));
            row_sum += bn_avx2_hsum_ps(acc) * d;
#elif defined(__wasm_simd128__)
            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);
            for (int i = 0; i < 2; i++) {
                v128_t w = wasm_v128_load(blk->qs + i * 16);
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(w);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(w);
                const float *xp = xb + i * 16;
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xp)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xp + 4)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xp + 8)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xp + 12)));
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum) * d;
#else
            float block_sum = 0.0f;
            for (int i = 0; i < 32; i++) {
                block_sum += blk->qs[i] * xb[i];
            }
            row_sum += block_sum * d;
#endif
        }
        c->out[row] = row_sum;
    }
}

// --- Q4_0 matrix-vector multiply (float fallback for WASM/scalar/NEON without dotprod) ---

#if !(defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) && !defined(__AVX2__)

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} Q4Ctx;

static void q4_range(void *ctx, int row_start, int row_end) {
    Q4Ctx *c = (Q4Ctx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
#ifdef __ARM_NEON
            uint8x16_t raw = vld1q_u8(blk->qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0xF))), vdupq_n_s8(8));
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), vdupq_n_s8(8));
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
            neon_acc_i8x16_f32(lo, xb, &acc0, &acc1, &acc2, &acc3);
            neon_acc_i8x16_f32(hi, xb + 16, &acc0, &acc1, &acc2, &acc3);
            row_sum += neon_reduce4(acc0, acc1, acc2, acc3) * d;
#elif defined(__AVX2__)
            __m128i raw = _mm_loadu_si128((const __m128i *)blk->qs);
            __m128i lo_128 = _mm_and_si128(raw, _mm_set1_epi8(0xF));
            __m128i hi_128 = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0xF));
            __m128i bias = _mm_set1_epi8(8);
            lo_128 = _mm_sub_epi8(lo_128, bias);
            hi_128 = _mm_sub_epi8(hi_128, bias);
            __m256 acc = _mm256_setzero_ps();
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_128)), _mm256_loadu_ps(xb)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo_128, 8))), _mm256_loadu_ps(xb + 8)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_128)), _mm256_loadu_ps(xb + 16)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi_128, 8))), _mm256_loadu_ps(xb + 24)));
            row_sum += bn_avx2_hsum_ps(acc) * d;
#elif defined(__wasm_simd128__)
            v128_t raw = wasm_v128_load(blk->qs);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, wasm_i8x16_splat(0xF)), wasm_i8x16_splat(8));
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), wasm_i8x16_splat(8));
            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);
            // Low nibbles (elements 0-15)
            {
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(lo);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(lo);
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xb)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 4)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xb + 8)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 12)));
            }
            // High nibbles (elements 16-31)
            {
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(hi);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(hi);
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xb + 16)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 20)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xb + 24)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 28)));
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum) * d;
#else
            float block_sum = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                block_sum += ((int)(byte & 0xF) - 8) * xb[i];
                block_sum += ((int)(byte >> 4) - 8) * xb[i + 16];
            }
            row_sum += block_sum * d;
#endif
        }
        c->out[row] = row_sum;
    }
}

#endif // !(NEON+DOTPROD) && !AVX2 — Q4_0 float fallback

// --- Q6_K matrix-vector multiply ---
// Layout per 256-element block (2 chunks of 128):
//   ql[0..63]: lower 4 bits (split: ql[0..31] and ql[32..63] per chunk)
//   qh[0..31]: upper 2 bits packed (2 bits per sub-group per element)
//   sc[0..7]: int8 sub-block scales (per 16 elements, 8 per chunk)
//   d: FP16 super-block scale
//
// Per chunk, 4 sub-groups of 32 elements (offsets 0, 32, 64, 96):
//   q1[l] = (ql[l] & 0xF)      | ((qh[l]>>0)&3)<<4 - 32  (elements 0..31)
//   q2[l] = (ql[l+32] & 0xF)   | ((qh[l]>>2)&3)<<4 - 32  (elements 32..63)
//   q3[l] = (ql[l] >> 4)       | ((qh[l]>>4)&3)<<4 - 32  (elements 64..95)
//   q4[l] = (ql[l+32] >> 4)    | ((qh[l]>>6)&3)<<4 - 32  (elements 96..127)
//   Sub-block scale index: sc[l/16 + {0,2,4,6}] per sub-group

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} Q6KCtx;

#ifdef __ARM_NEON
static void q6k_range(void *ctx, int row_start, int row_end) {
    Q6KCtx *c = (Q6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    const uint8x16_t mask_lo4 = vdupq_n_u8(0xF);
    const uint8x16_t mask_2 = vdupq_n_u8(3);
    const int8x16_t bias32 = vdupq_n_s8(32);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int chunk = 0; chunk < 2; chunk++) {
                // Load ql: 64 bytes (ql[0..31] and ql[32..63])
                uint8x16_t ql0 = vld1q_u8(ql);        // ql[0..15]
                uint8x16_t ql1 = vld1q_u8(ql + 16);   // ql[16..31]
                uint8x16_t ql2 = vld1q_u8(ql + 32);   // ql[32..47]
                uint8x16_t ql3 = vld1q_u8(ql + 48);   // ql[48..63]

                // Load qh: 32 bytes
                uint8x16_t qh0 = vld1q_u8(qh);        // qh[0..15]
                uint8x16_t qh1 = vld1q_u8(qh + 16);   // qh[16..31]

                // Sub-group 0 (elements 0..31): lo4 of ql[0..31], qh bits 0-1
                int8x16_t w0a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql0, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh0, mask_2), 4))), bias32);
                int8x16_t w0b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql1, mask_lo4),
                    vshlq_n_u8(vandq_u8(qh1, mask_2), 4))), bias32);

                // Sub-group 1 (elements 32..63): lo4 of ql[32..63], qh bits 2-3
                int8x16_t w1a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql2, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_2), 4))), bias32);
                int8x16_t w1b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql3, mask_lo4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_2), 4))), bias32);

                // Sub-group 2 (elements 64..95): hi4 of ql[0..31], qh bits 4-5
                int8x16_t w2a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql0, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_2), 4))), bias32);
                int8x16_t w2b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql1, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_2), 4))), bias32);

                // Sub-group 3 (elements 96..127): hi4 of ql[32..63], qh bits 6-7
                int8x16_t w3a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql2, 4),
                    vshlq_n_u8(vshrq_n_u8(qh0, 6), 4))), bias32);
                int8x16_t w3b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql3, 4),
                    vshlq_n_u8(vshrq_n_u8(qh1, 6), 4))), bias32);

                // Accumulate: widen int8 weights to float, FMA with x
                // 8 sub-blocks of 16 elements each, with per-16-element scales
                float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
                float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

                #define Q6K_ACC_16(w_vec, xp, scale_val) do { \
                    float ds = d * (float)(scale_val); \
                    float32x4_t vds = vdupq_n_f32(ds); \
                    int16x8_t lo16 = vmovl_s8(vget_low_s8(w_vec)); \
                    int16x8_t hi16 = vmovl_s8(vget_high_s8(w_vec)); \
                    acc0 = vmlaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vds), vld1q_f32(xp)); \
                    acc1 = vmlaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vds), vld1q_f32(xp + 4)); \
                    acc2 = vmlaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vds), vld1q_f32(xp + 8)); \
                    acc3 = vmlaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vds), vld1q_f32(xp + 12)); \
                } while(0)

                Q6K_ACC_16(w0a, xb +  0, sc[0]);
                Q6K_ACC_16(w0b, xb + 16, sc[1]);
                Q6K_ACC_16(w1a, xb + 32, sc[2]);
                Q6K_ACC_16(w1b, xb + 48, sc[3]);
                Q6K_ACC_16(w2a, xb + 64, sc[4]);
                Q6K_ACC_16(w2b, xb + 80, sc[5]);
                Q6K_ACC_16(w3a, xb + 96, sc[6]);
                Q6K_ACC_16(w3b, xb +112, sc[7]);

                #undef Q6K_ACC_16

                float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
                row_sum += vget_lane_f32(vpadd_f32(r, r), 0);

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}

#elif defined(__AVX2__)
static void q6k_range(void *ctx, int row_start, int row_end) {
    Q6KCtx *c = (Q6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    const __m128i mask_lo4 = _mm_set1_epi8(0xF);
    const __m128i mask_2 = _mm_set1_epi8(3);
    const __m128i bias32 = _mm_set1_epi8(32);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int chunk = 0; chunk < 2; chunk++) {
                // Load ql (4×16 bytes) and qh (2×16 bytes)
                __m128i ql0 = _mm_loadu_si128((const __m128i *)(ql));
                __m128i ql1 = _mm_loadu_si128((const __m128i *)(ql + 16));
                __m128i ql2 = _mm_loadu_si128((const __m128i *)(ql + 32));
                __m128i ql3 = _mm_loadu_si128((const __m128i *)(ql + 48));
                __m128i qh0 = _mm_loadu_si128((const __m128i *)(qh));
                __m128i qh1 = _mm_loadu_si128((const __m128i *)(qh + 16));

                // Reconstruct 6-bit weights for 8 sub-blocks of 16 elements
                // Sub-group 0: lo4 of ql[0..31], qh bits 0-1
                __m128i w0a = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(ql0, mask_lo4),
                    _mm_slli_epi16(_mm_and_si128(qh0, mask_2), 4)), bias32);
                __m128i w0b = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(ql1, mask_lo4),
                    _mm_slli_epi16(_mm_and_si128(qh1, mask_2), 4)), bias32);

                // Sub-group 1: lo4 of ql[32..63], qh bits 2-3
                __m128i w1a = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(ql2, mask_lo4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 2), mask_2), 4)), bias32);
                __m128i w1b = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(ql3, mask_lo4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 2), mask_2), 4)), bias32);

                // Sub-group 2: hi4 of ql[0..31], qh bits 4-5
                __m128i w2a = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql0, 4), mask_lo4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 4), mask_2), 4)), bias32);
                __m128i w2b = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql1, 4), mask_lo4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 4), mask_2), 4)), bias32);

                // Sub-group 3: hi4 of ql[32..63], qh bits 6-7
                __m128i w3a = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql2, 4), mask_lo4),
                    _mm_slli_epi16(_mm_srli_epi16(qh0, 6), 4)), bias32);
                __m128i w3b = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql3, 4), mask_lo4),
                    _mm_slli_epi16(_mm_srli_epi16(qh1, 6), 4)), bias32);

                // Accumulate: widen int8 weights to float, FMA with x
                // Each sub-block is 16 elements with its own scale
                __m256 acc = _mm256_setzero_ps();

                #define Q6K_AVX2_ACC_16(w128, xp, scale_val) do { \
                    __m256 vds = _mm256_set1_ps(d * (float)(scale_val)); \
                    __m256 w_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w128)), vds); \
                    __m256 w_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w128, 8))), vds); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(w_lo, _mm256_loadu_ps(xp))); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(w_hi, _mm256_loadu_ps(xp + 8))); \
                } while(0)

                Q6K_AVX2_ACC_16(w0a, xb +   0, sc[0]);
                Q6K_AVX2_ACC_16(w0b, xb +  16, sc[1]);
                Q6K_AVX2_ACC_16(w1a, xb +  32, sc[2]);
                Q6K_AVX2_ACC_16(w1b, xb +  48, sc[3]);
                Q6K_AVX2_ACC_16(w2a, xb +  64, sc[4]);
                Q6K_AVX2_ACC_16(w2b, xb +  80, sc[5]);
                Q6K_AVX2_ACC_16(w3a, xb +  96, sc[6]);
                Q6K_AVX2_ACC_16(w3b, xb + 112, sc[7]);

                #undef Q6K_AVX2_ACC_16

                row_sum += bn_avx2_hsum_ps(acc);

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}

#elif defined(__wasm_simd128__)
static void q6k_range(void *ctx, int row_start, int row_end) {
    Q6KCtx *c = (Q6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    const v128_t mask_lo4 = wasm_i8x16_splat(0xF);
    const v128_t mask_2 = wasm_i8x16_splat(3);
    const v128_t bias32 = wasm_i8x16_splat(32);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int chunk = 0; chunk < 2; chunk++) {
                v128_t ql0 = wasm_v128_load(ql);
                v128_t ql1 = wasm_v128_load(ql + 16);
                v128_t ql2 = wasm_v128_load(ql + 32);
                v128_t ql3 = wasm_v128_load(ql + 48);
                v128_t qh0 = wasm_v128_load(qh);
                v128_t qh1 = wasm_v128_load(qh + 16);

                // Sub-group 0
                v128_t w0a = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(ql0, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(qh0, mask_2), 4)), bias32);
                v128_t w0b = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(ql1, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(qh1, mask_2), 4)), bias32);
                // Sub-group 1
                v128_t w1a = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(ql2, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 2), mask_2), 4)), bias32);
                v128_t w1b = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(ql3, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 2), mask_2), 4)), bias32);
                // Sub-group 2
                v128_t w2a = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(wasm_u8x16_shr(ql0, 4), mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 4), mask_2), 4)), bias32);
                v128_t w2b = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(wasm_u8x16_shr(ql1, 4), mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 4), mask_2), 4)), bias32);
                // Sub-group 3
                v128_t w3a = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(wasm_u8x16_shr(ql2, 4), mask_lo4),
                    wasm_i8x16_shl(wasm_u8x16_shr(qh0, 6), 4)), bias32);
                v128_t w3b = wasm_i8x16_sub(wasm_v128_or(
                    wasm_v128_and(wasm_u8x16_shr(ql3, 4), mask_lo4),
                    wasm_i8x16_shl(wasm_u8x16_shr(qh1, 6), 4)), bias32);

                v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
                v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

                #define Q6K_WASM_ACC_16(w_vec, xp, scale_val) do { \
                    v128_t vds = wasm_f32x4_splat(d * (float)(scale_val)); \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_mul( \
                        wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vds), wasm_v128_load(xp))); \
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_mul( \
                        wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vds), wasm_v128_load(xp + 4))); \
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_mul( \
                        wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vds), wasm_v128_load(xp + 8))); \
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_mul( \
                        wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vds), wasm_v128_load(xp + 12))); \
                } while(0)

                Q6K_WASM_ACC_16(w0a, xb +   0, sc[0]);
                Q6K_WASM_ACC_16(w0b, xb +  16, sc[1]);
                Q6K_WASM_ACC_16(w1a, xb +  32, sc[2]);
                Q6K_WASM_ACC_16(w1b, xb +  48, sc[3]);
                Q6K_WASM_ACC_16(w2a, xb +  64, sc[4]);
                Q6K_WASM_ACC_16(w2b, xb +  80, sc[5]);
                Q6K_WASM_ACC_16(w3a, xb +  96, sc[6]);
                Q6K_WASM_ACC_16(w3b, xb + 112, sc[7]);

                #undef Q6K_WASM_ACC_16

                v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
                row_sum += bn_wasm_hsum_f32x4(sum);

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}

#else
// Scalar fallback
static void q6k_range(void *ctx, int row_start, int row_end) {
    Q6KCtx *c = (Q6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int n = 0; n < BN_QK_K; n += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                    row_sum += d * sc[is + 0] * q1 * xb[l +  0];
                    row_sum += d * sc[is + 2] * q2 * xb[l + 32];
                    row_sum += d * sc[is + 4] * q3 * xb[l + 64];
                    row_sum += d * sc[is + 6] * q4 * xb[l + 96];
                }
                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}
#endif

// --- Quantized matrix-vector multiply ---
// out[rows] = W[rows x cols] @ x[cols]

void bn_quant_matvec(float *out, const BnQWeight *W, const float *x,
                     int8_t *x_q_buf, BnThreadPool *pool) {

    if (W->type == BN_GGUF_TENSOR_I2_S) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        // SDOT path: quantize x once, then use integer dot products
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        I2SCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { i2s_sdot_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__ARM_NEON)
        (void)x_q_buf;
        I2SFloatCtx ctx = { out, W, x };
        BnTPTask task = { i2s_neon_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__AVX2__)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        I2SCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { i2s_avx2_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        I2SFloatCtx ctx = { out, W, x };
        BnTPTask task = { i2s_wasm_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#else
        (void)x_q_buf;
        I2SFloatCtx ctx = { out, W, x };
        BnTPTask task = { i2s_scalar_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#endif
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q8_0) {
        (void)x_q_buf;
        Q8Ctx ctx = { out, W, x };
        BnTPTask task = { q8_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q4_0) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int n_blocks = W->cols / 32;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        Q4SdotCtx ctx = { out, W, x_q_buf, x_scales };
        BnTPTask task = { q4_sdot_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__AVX2__)
        int n_blocks = W->cols / 32;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        Q4SdotCtx ctx = { out, W, x_q_buf, x_scales };
        BnTPTask task = { q4_sdot_avx2_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#else
        (void)x_q_buf;
        Q4Ctx ctx = { out, W, x };
        BnTPTask task = { q4_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#endif
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q6_K) {
        (void)x_q_buf;
        Q6KCtx ctx = { out, W, x };
        BnTPTask task = { q6k_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_TQ2_0) {
        (void)x_q_buf;
        TQ2Ctx ctx = { out, W, x };
        BnTPTask task = { tq2_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    // TQ1_0
    {
        (void)x_q_buf;
        TQ1Ctx ctx = { out, W, x };
        BnTPTask task = { tq1_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
    }
}

// --- Batch ternary matvec ---
// Runs multiple independent matvecs with a single dispatch.

void bn_quant_matvec_batch(const BnMatvecTask *tasks, int n_tasks,
                           const float *x, int8_t *x_q_buf, BnThreadPool *pool) {
    if (n_tasks <= 0) return;

    int cols = tasks[0].W->cols;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int all_i2s = 1, all_q4 = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (!all_i2s && !all_q4) break;
    }

    if (all_i2s) {
        if (n_tasks > 4) {
            for (int t = 0; t < n_tasks; t++)
                bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool);
            return;
        }

        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        I2SCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (I2SCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ i2s_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q4 && n_tasks <= 4) {
        int n_blocks = cols / 32;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, cols);

        Q4SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (Q4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales };
            tp_tasks[t] = (BnTPTask){ q4_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#elif defined(__AVX2__)
    int all_i2s = 1, all_q4 = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (!all_i2s && !all_q4) break;
    }

    if (all_i2s) {
        if (n_tasks > 4) {
            for (int t = 0; t < n_tasks; t++)
                bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool);
            return;
        }

        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        I2SCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (I2SCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ i2s_avx2_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q4 && n_tasks <= 4) {
        int n_blocks = cols / 32;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, cols);

        Q4SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (Q4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales };
            tp_tasks[t] = (BnTPTask){ q4_sdot_avx2_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#else
    (void)x_q_buf;
    (void)cols;
#endif

    // Fallback: use existing per-task matvec
    for (int t = 0; t < n_tasks; t++) {
        bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool);
    }
}
