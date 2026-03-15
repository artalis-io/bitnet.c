#include "quant_internal.h"
#include <arm_neon.h>

void bn_quant_bf16_neon_range(void *ctx, int row_start, int row_end) {
    BnBF16Ctx *c = (BnBF16Ctx *)ctx;
    const uint16_t *data = (const uint16_t *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint16_t *w = data + (size_t)row * cols;
        float32x4_t acc0 = vdupq_n_f32(0);
        float32x4_t acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0);
        float32x4_t acc3 = vdupq_n_f32(0);

        int col = 0;
        for (; col + 15 < cols; col += 16) {
            __builtin_prefetch(w + col + 64, 0, 0);
            // Load 4 groups of 4 BF16 values, convert to F32 via shift-left-16
            uint16x4_t bf0 = vld1_u16(w + col);
            uint16x4_t bf1 = vld1_u16(w + col + 4);
            uint16x4_t bf2 = vld1_u16(w + col + 8);
            uint16x4_t bf3 = vld1_u16(w + col + 12);

            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(bf0, 16));
            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(bf1, 16));
            float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(bf2, 16));
            float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(bf3, 16));

            acc0 = vmlaq_f32(acc0, w0, vld1q_f32(x + col));
            acc1 = vmlaq_f32(acc1, w1, vld1q_f32(x + col + 4));
            acc2 = vmlaq_f32(acc2, w2, vld1q_f32(x + col + 8));
            acc3 = vmlaq_f32(acc3, w3, vld1q_f32(x + col + 12));
        }

        float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
        float row_sum = vget_lane_f32(vpadd_f32(r, r), 0);

        // Scalar tail
        for (; col < cols; col++) {
            uint32_t bits = (uint32_t)w[col] << 16;
            float wf;
            memcpy(&wf, &bits, 4);
            row_sum += wf * x[col];
        }

        c->out[row] = row_sum;
    }
}
