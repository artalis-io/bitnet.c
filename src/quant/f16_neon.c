#include "quant_ctx.h"
#include <arm_neon.h>

void bn_quant_f16_neon_range(void *ctx, int row_start, int row_end) {
    BnF16Ctx *c = (BnF16Ctx *)ctx;
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
            float tmp[16];
            for (int i = 0; i < 16; i++) tmp[i] = bn_fp16_to_fp32(w[col + i]);
            acc0 = vmlaq_f32(acc0, vld1q_f32(tmp), vld1q_f32(x + col));
            acc1 = vmlaq_f32(acc1, vld1q_f32(tmp + 4), vld1q_f32(x + col + 4));
            acc2 = vmlaq_f32(acc2, vld1q_f32(tmp + 8), vld1q_f32(x + col + 8));
            acc3 = vmlaq_f32(acc3, vld1q_f32(tmp + 12), vld1q_f32(x + col + 12));
        }

        float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
        float row_sum = vget_lane_f32(vpadd_f32(r, r), 0);
        for (; col < cols; col++) {
            row_sum += bn_fp16_to_fp32(w[col]) * x[col];
        }
        c->out[row] = row_sum;
    }
}
