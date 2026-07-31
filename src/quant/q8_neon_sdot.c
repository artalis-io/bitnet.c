#include "quant_ctx.h"
#include <arm_neon.h>

static inline float q8_fp16_to_f32(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vld1_dup_f16((const float16_t *)&h)), 0);
}

void bn_quant_q8_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int row = row_start; row < row_end; row++) {
        int base = row * n_blocks_per_row;
        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t sumv1 = vdupq_n_f32(0.0f);
        int b = 0;

        for (; b + 1 < n_blocks_per_row; b += 2) {
            const BnBlockQ8_0 *b0 = &blocks[base + b];
            const BnBlockQ8_0 *b1 = &blocks[base + b + 1];
            __builtin_prefetch(b0 + 4, 0, 0);

            float dw0 = q8_fp16_to_f32(b0->d);
            float dw1 = q8_fp16_to_f32(b1->d);
            float dx0 = x_scales[b];
            float dx1 = x_scales[b + 1];
            const int8_t *xb0 = x_q + (b)     * 32;
            const int8_t *xb1 = x_q + (b + 1) * 32;

            int32x4_t a0 = vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), vld1q_s8(b0->qs),
                           vld1q_s8(xb0)),
                vdotq_s32(vdupq_n_s32(0), vld1q_s8(b0->qs + 16),
                           vld1q_s8(xb0 + 16)));
            int32x4_t a1 = vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), vld1q_s8(b1->qs),
                           vld1q_s8(xb1)),
                vdotq_s32(vdupq_n_s32(0), vld1q_s8(b1->qs + 16),
                           vld1q_s8(xb1 + 16)));

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(a0), dw0 * dx0);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(a1), dw1 * dx1);
        }

        float row_sum = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
        for (; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[base + b];
            int sum = 0;
            for (int i = 0; i < 32; i++)
                sum += blk->qs[i] * x_q[b * 32 + i];
            row_sum += (float)sum * q8_fp16_to_f32(blk->d) * x_scales[b];
        }

        c->out[row] = row_sum;
    }
}
