#include "transformer_logits_internal.h"

#ifdef __ARM_NEON

#ifdef __ARM_FEATURE_DOTPROD
void bn_transformer_logits_i8_neon_range(void *ctx, int v_start, int v_end) {
    BnLogitsI8Ctx *lc = (BnLogitsI8Ctx *)ctx;
    const int8_t *emb_i8 = lc->emb_i8;
    const float *emb_scales = lc->emb_scales;
    const int8_t *x_q = lc->x_q;
    float x_scale = lc->x_scale;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const int8_t *row = emb_i8 + (size_t)v * dim;
        __builtin_prefetch(row + (size_t)dim, 0, 0);
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
        for (int d = 0; d < dim; d += 64) {
            __builtin_prefetch(row + d + 128, 0, 0);
            acc0 = vdotq_s32(acc0, vld1q_s8(row+d),    vld1q_s8(x_q+d));
            acc1 = vdotq_s32(acc1, vld1q_s8(row+d+16), vld1q_s8(x_q+d+16));
            acc2 = vdotq_s32(acc2, vld1q_s8(row+d+32), vld1q_s8(x_q+d+32));
            acc3 = vdotq_s32(acc3, vld1q_s8(row+d+48), vld1q_s8(x_q+d+48));
        }
        int32x4_t sum4 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        int32_t total = vaddvq_s32(sum4);
        lc->logits[v] = (float)total * emb_scales[v] * x_scale;
    }
}
#endif // __ARM_FEATURE_DOTPROD

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void bn_transformer_logits_f16_native_neon_range(void *ctx, int v_start, int v_end) {
    BnLogitsCtx *lc = (BnLogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const uint16_t *x_f16 = (const uint16_t *)lc->x;  // pre-converted
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        float32x4_t fsum = vdupq_n_f32(0);
        const float16x8_t fz = vreinterpretq_f16_u16(vdupq_n_u16(0));
        int d = 0;

        #define LDF16(p) vreinterpretq_f16_u16(vld1q_u16(p))
        for (; d + 63 < dim; d += 64) {
            float16x8_t a0 = fz, a1 = fz, a2 = fz, a3 = fz;
            a0 = vfmaq_f16(a0, LDF16(row+d),    LDF16(x_f16+d));
            a1 = vfmaq_f16(a1, LDF16(row+d+8),  LDF16(x_f16+d+8));
            a2 = vfmaq_f16(a2, LDF16(row+d+16), LDF16(x_f16+d+16));
            a3 = vfmaq_f16(a3, LDF16(row+d+24), LDF16(x_f16+d+24));
            a0 = vfmaq_f16(a0, LDF16(row+d+32), LDF16(x_f16+d+32));
            a1 = vfmaq_f16(a1, LDF16(row+d+40), LDF16(x_f16+d+40));
            a2 = vfmaq_f16(a2, LDF16(row+d+48), LDF16(x_f16+d+48));
            a3 = vfmaq_f16(a3, LDF16(row+d+56), LDF16(x_f16+d+56));
            float16x8_t s = vaddq_f16(vaddq_f16(a0, a1), vaddq_f16(a2, a3));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_low_f16(s)));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_high_f16(s)));
        }
        for (; d + 7 < dim; d += 8) {
            float16x8_t p = vmulq_f16(LDF16(row+d), LDF16(x_f16+d));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_low_f16(p)));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_high_f16(p)));
        }
        #undef LDF16

        lc->logits[v] = bn_transformer_neon_hsum_f32(fsum);
    }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void bn_transformer_logits_f16_neon_range(void *ctx, int v_start, int v_end) {
    BnLogitsCtx *lc = (BnLogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        for (int d = 0; d < dim; d += 8) {
            float16x8_t f16 = vreinterpretq_f16_u16(vld1q_u16(row + d));
            acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vget_low_f16(f16)),  vld1q_f32(x + d));
            acc1 = vmlaq_f32(acc1, vcvt_f32_f16(vget_high_f16(f16)), vld1q_f32(x + d + 4));
        }
        lc->logits[v] = bn_transformer_neon_hsum_f32(vaddq_f32(acc0, acc1));
    }
}
#endif // !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#endif // __ARM_NEON
