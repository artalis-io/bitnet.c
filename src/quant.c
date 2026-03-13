#include "quant.h"
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

#endif // __ARM_NEON && __ARM_FEATURE_DOTPROD

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

// I2_S scalar range
#if !defined(__ARM_NEON)
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
#endif // !__ARM_NEON

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

// --- Ternary matrix-vector multiply ---
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
#else
        (void)x_q_buf;
        I2SFloatCtx ctx = { out, W, x };
        BnTPTask task = { i2s_scalar_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#endif
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
    int all_i2s = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) { all_i2s = 0; break; }
    }

    if (all_i2s) {
        assert(n_tasks <= 4 && "bn_quant_matvec_batch: max 4 tasks");

        // Quantize x to int8 once, shared across all tasks
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        // Build one BnTPTask per matvec task
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
#else
    (void)x_q_buf;
    (void)cols;
#endif

    // Fallback: use existing per-task matvec
    for (int t = 0; t < n_tasks; t++) {
        bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool);
    }
}
