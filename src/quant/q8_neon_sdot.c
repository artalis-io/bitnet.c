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

    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        int base = row * n_blocks_per_row;
        int b = 0;

        // Process 4 blocks at a time for better ILP
        for (; b + 3 < n_blocks_per_row; b += 4) {
            const BnBlockQ8_0 *b0 = &blocks[base + b];
            const BnBlockQ8_0 *b1 = &blocks[base + b + 1];
            const BnBlockQ8_0 *b2 = &blocks[base + b + 2];
            const BnBlockQ8_0 *b3 = &blocks[base + b + 3];
            __builtin_prefetch(b0 + 6, 0, 0);

            float dw0 = q8_fp16_to_f32(b0->d);
            float dw1 = q8_fp16_to_f32(b1->d);
            float dw2 = q8_fp16_to_f32(b2->d);
            float dw3 = q8_fp16_to_f32(b3->d);

            float dx0 = x_scales[b];
            float dx1 = x_scales[b + 1];
            float dx2 = x_scales[b + 2];
            float dx3 = x_scales[b + 3];

            const int8_t *xb0 = x_q + (b)     * 32;
            const int8_t *xb1 = x_q + (b + 1) * 32;
            const int8_t *xb2 = x_q + (b + 2) * 32;
            const int8_t *xb3 = x_q + (b + 3) * 32;

            // Block 0
            int32x4_t a0 = vdotq_s32(zero, vld1q_s8(b0->qs), vld1q_s8(xb0));
            a0 = vdotq_s32(a0, vld1q_s8(b0->qs + 16), vld1q_s8(xb0 + 16));

            // Block 1
            int32x4_t a1 = vdotq_s32(zero, vld1q_s8(b1->qs), vld1q_s8(xb1));
            a1 = vdotq_s32(a1, vld1q_s8(b1->qs + 16), vld1q_s8(xb1 + 16));

            // Block 2
            int32x4_t a2 = vdotq_s32(zero, vld1q_s8(b2->qs), vld1q_s8(xb2));
            a2 = vdotq_s32(a2, vld1q_s8(b2->qs + 16), vld1q_s8(xb2 + 16));

            // Block 3
            int32x4_t a3 = vdotq_s32(zero, vld1q_s8(b3->qs), vld1q_s8(xb3));
            a3 = vdotq_s32(a3, vld1q_s8(b3->qs + 16), vld1q_s8(xb3 + 16));

            // Deferred reduction
            row_sum += dw0 * dx0 * (float)vaddvq_s32(a0)
                     + dw1 * dx1 * (float)vaddvq_s32(a1)
                     + dw2 * dx2 * (float)vaddvq_s32(a2)
                     + dw3 * dx3 * (float)vaddvq_s32(a3);
        }

        // Tail: remaining blocks
        for (; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[base + b];
            float d_w = q8_fp16_to_f32(blk->d);
            float d_x = x_scales[b];

            int32x4_t acc = vdotq_s32(zero, vld1q_s8(blk->qs), vld1q_s8(x_q + b * 32));
            acc = vdotq_s32(acc, vld1q_s8(blk->qs + 16), vld1q_s8(x_q + b * 32 + 16));

            row_sum += d_w * d_x * (float)vaddvq_s32(acc);
        }

        c->out[row] = row_sum;
    }
}
