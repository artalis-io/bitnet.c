#include "quant_ctx.h"
#include <arm_neon.h>

static inline float q8_fp16_to_f32(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vld1_dup_f16((const float16_t *)&h)), 0);
}

static float q8_neon_row_dot(const BnQWeight *W, int row,
                             const int8_t *x_q, const float *x_scales) {
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)W->data;
    int n_blocks_per_row = W->cols / 32;
    int base = row * n_blocks_per_row;
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    const int32x4_t zero = vdupq_n_s32(0);
    int b = 0;

    for (; b + 1 < n_blocks_per_row; b += 2) {
        const BnBlockQ8_0 *blk0 = &blocks[base + b];
        const BnBlockQ8_0 *blk1 = &blocks[base + b + 1];
        const int8_t *xb0 = x_q + (size_t)b * 32;
        const int8_t *xb1 = xb0 + 32;
        int32x4_t dot0 = vaddq_s32(
            vdotq_s32(zero, vld1q_s8(blk0->qs), vld1q_s8(xb0)),
            vdotq_s32(zero, vld1q_s8(blk0->qs + 16),
                      vld1q_s8(xb0 + 16)));
        int32x4_t dot1 = vaddq_s32(
            vdotq_s32(zero, vld1q_s8(blk1->qs), vld1q_s8(xb1)),
            vdotq_s32(zero, vld1q_s8(blk1->qs + 16),
                      vld1q_s8(xb1 + 16)));
        sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(dot0),
                            q8_fp16_to_f32(blk0->d) * x_scales[b]);
        sum1 = vmlaq_n_f32(sum1, vcvtq_f32_s32(dot1),
                            q8_fp16_to_f32(blk1->d) * x_scales[b + 1]);
    }

    float result = vaddvq_f32(sum0) + vaddvq_f32(sum1);
    for (; b < n_blocks_per_row; b++) {
        const BnBlockQ8_0 *blk = &blocks[base + b];
        const int8_t *xb = x_q + (size_t)b * 32;
        int32_t dot = 0;
        for (int i = 0; i < 32; i++)
            dot += (int32_t)blk->qs[i] * (int32_t)xb[i];
        result += (float)dot * q8_fp16_to_f32(blk->d) * x_scales[b];
    }
    return result;
}

void bn_quant_q8_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    for (int row = row_start; row < row_end; row++)
        c->out[row] = q8_neon_row_dot(
            c->W, row, c->x_q, c->x_scales);
}

void bn_quant_q8_neon_sdot_4row_range(void *ctx, int group_start,
                                      int group_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int32x4_t zero = vdupq_n_s32(0);
    for (int group = group_start; group < group_end; group++) {
        int row0 = group * 4;
        int nr = c->W->rows - row0;
        if (nr > 4) nr = 4;
        if (nr <= 0) continue;

        float32x4_t sum0[4] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
        };
        float32x4_t sum1[4] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
        };
        int b = 0;
        for (; b + 1 < n_blocks_per_row; b += 2) {
            const int8_t *xb0 = c->x_q + (size_t)b * 32;
            const int8_t *xb1 = xb0 + 32;
            int8x16_t x00 = vld1q_s8(xb0);
            int8x16_t x01 = vld1q_s8(xb0 + 16);
            int8x16_t x10 = vld1q_s8(xb1);
            int8x16_t x11 = vld1q_s8(xb1 + 16);
            for (int r = 0; r < nr; r++) {
                const BnBlockQ8_0 *blk0 =
                    &blocks[(size_t)(row0 + r) * n_blocks_per_row + b];
                const BnBlockQ8_0 *blk1 = blk0 + 1;
                int32x4_t dot0 = vaddq_s32(
                    vdotq_s32(zero, vld1q_s8(blk0->qs), x00),
                    vdotq_s32(zero, vld1q_s8(blk0->qs + 16), x01));
                int32x4_t dot1 = vaddq_s32(
                    vdotq_s32(zero, vld1q_s8(blk1->qs), x10),
                    vdotq_s32(zero, vld1q_s8(blk1->qs + 16), x11));
                sum0[r] = vmlaq_n_f32(
                    sum0[r], vcvtq_f32_s32(dot0),
                    q8_fp16_to_f32(blk0->d) * c->x_scales[b]);
                sum1[r] = vmlaq_n_f32(
                    sum1[r], vcvtq_f32_s32(dot1),
                    q8_fp16_to_f32(blk1->d) * c->x_scales[b + 1]);
            }
        }
        for (int r = 0; r < nr; r++) {
            float result = vaddvq_f32(sum0[r]) + vaddvq_f32(sum1[r]);
            for (int tail = b; tail < n_blocks_per_row; tail++) {
                const BnBlockQ8_0 *blk =
                    &blocks[(size_t)(row0 + r) * n_blocks_per_row + tail];
                const int8_t *xb = c->x_q + (size_t)tail * 32;
                int32_t dot = 0;
                for (int i = 0; i < 32; i++)
                    dot += (int32_t)blk->qs[i] * (int32_t)xb[i];
                result += (float)dot * q8_fp16_to_f32(blk->d) *
                          c->x_scales[tail];
            }
            c->out[row0 + r] = result;
        }
    }
}

void bn_quant_q8_repacked_neon_sdot_range(void *ctx,
                                          int row_start,
                                          int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnPreparedWeight *prepared = c->prepared;
    if (!prepared || prepared->kind != BN_PREPARED_WEIGHT_Q8_0_REPACK ||
        !prepared->qs || !prepared->scales) {
        bn_quant_q8_neon_sdot_range(ctx, row_start, row_end);
        return;
    }
    int n_blocks_per_row = c->W->cols / 32;
    int row = row_start;
    for (; row < row_end && (row & 3); row++)
        c->out[row] = q8_neon_row_dot(c->W, row, c->x_q, c->x_scales);
    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t gb = (size_t)group * n_blocks_per_row + b;
            const int8_t *qbase = (const int8_t *)prepared->qs + gb * 128;
            const int8_t *xb = c->x_q + b * 32;
            int8x16_t a0 = vld1q_s8(xb);
            int8x16_t a1 = vld1q_s8(xb + 16);
            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase), a0, 0);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase + 16), a0, 1);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase + 32), a0, 2);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase + 48), a0, 3);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase + 64), a1, 0);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase + 80), a1, 1);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase + 96), a1, 2);
            dot = vdotq_laneq_s32(dot, vld1q_s8(qbase + 112), a1, 3);
            float32x4_t wd = vcvt_f32_f16(vld1_f16(
                (const float16_t *)(prepared->scales + gb * 4)));
            float32x4_t scale = vmulq_n_f32(wd, c->x_scales[b]);
            sum = vfmaq_f32(sum, vcvtq_f32_s32(dot), scale);
        }
        vst1q_f32(c->out + row, sum);
    }
    for (; row < row_end; row++)
        c->out[row] = q8_neon_row_dot(c->W, row, c->x_q, c->x_scales);
}
