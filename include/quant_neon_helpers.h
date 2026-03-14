#ifndef BN_QUANT_NEON_HELPERS_H
#define BN_QUANT_NEON_HELPERS_H

#ifdef __ARM_NEON
#include <arm_neon.h>

// Widen 16 int8 ternary values (-1/0/+1) to float, FMA with 16 floats from x,
// accumulate into 4 float32x4 accumulators.
static inline void bn_neon_acc_i8x16_f32(int8x16_t t, const float *x,
    float32x4_t *a0, float32x4_t *a1, float32x4_t *a2, float32x4_t *a3) {
    int16x8_t lo16 = vmovl_s8(vget_low_s8(t));
    int16x8_t hi16 = vmovl_s8(vget_high_s8(t));
    *a0 = vmlaq_f32(*a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))),  vld1q_f32(x + 0));
    *a1 = vmlaq_f32(*a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vld1q_f32(x + 4));
    *a2 = vmlaq_f32(*a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))),  vld1q_f32(x + 8));
    *a3 = vmlaq_f32(*a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vld1q_f32(x + 12));
}

// Reduce 4 float32x4 accumulators to a single scalar sum (ARMv7-compatible).
static inline float bn_neon_reduce4(float32x4_t a, float32x4_t b,
                                     float32x4_t c, float32x4_t d) {
    float32x4_t s = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d));
    float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}

// Accumulate 16 elements: w_int8 * x_float, scaled by ds, with min subtraction
// Used by Q4_K and Q5_K NEON kernels
// Requires acc0-acc3 to be in scope as float32x4_t variables
#define BN_QK_ACC_SCALED_16(w_vec, xp, ds_val, dm_val) do { \
    float32x4_t vds = vdupq_n_f32(ds_val); \
    float32x4_t vdm = vdupq_n_f32(dm_val); \
    int16x8_t lo16 = vmovl_s8(vget_low_s8(w_vec)); \
    int16x8_t hi16 = vmovl_s8(vget_high_s8(w_vec)); \
    acc0 = vmlaq_f32(acc0, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vds), vdm), vld1q_f32(xp)); \
    acc1 = vmlaq_f32(acc1, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), vds), vdm), vld1q_f32(xp + 4)); \
    acc2 = vmlaq_f32(acc2, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vds), vdm), vld1q_f32(xp + 8)); \
    acc3 = vmlaq_f32(acc3, vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), vds), vdm), vld1q_f32(xp + 12)); \
} while(0)

#endif // __ARM_NEON

#endif // BN_QUANT_NEON_HELPERS_H
