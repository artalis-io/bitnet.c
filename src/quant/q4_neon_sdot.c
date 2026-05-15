#include "quant_ctx.h"
#include "simd_helpers.h"
#include <arm_neon.h>
#include <math.h>

static inline int32x4_t q4_repacked_dot4_xor(const uint8_t *qbase,
                                             int8x16_t a0,
                                             int8x16_t a1) {
    const uint8x16_t mask_hi = vdupq_n_u8(0xF0);
    const int32x4_t zero = vdupq_n_s32(0);

    int8x16_t raw0 = vld1q_s8((const int8_t *)qbase);
    int8x16_t raw1 = vld1q_s8((const int8_t *)qbase + 16);
    int8x16_t raw2 = vld1q_s8((const int8_t *)qbase + 32);
    int8x16_t raw3 = vld1q_s8((const int8_t *)qbase + 48);
    int8x16_t lo0 = vshlq_n_s8(raw0, 4);
    int8x16_t lo1 = vshlq_n_s8(raw1, 4);
    int8x16_t lo2 = vshlq_n_s8(raw2, 4);
    int8x16_t lo3 = vshlq_n_s8(raw3, 4);
    int8x16_t hi0 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw0), mask_hi));
    int8x16_t hi1 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw1), mask_hi));
    int8x16_t hi2 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw2), mask_hi));
    int8x16_t hi3 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw3), mask_hi));

    int32x4_t acc02 = vdotq_laneq_s32(zero, lo0, a0, 0);
    int32x4_t acc13 = vdotq_laneq_s32(zero, lo1, a0, 1);
    acc02 = vdotq_laneq_s32(acc02, lo2, a0, 2);
    acc13 = vdotq_laneq_s32(acc13, lo3, a0, 3);
    int32x4_t acc46 = vdotq_laneq_s32(zero, hi0, a1, 0);
    int32x4_t acc57 = vdotq_laneq_s32(zero, hi1, a1, 1);
    acc46 = vdotq_laneq_s32(acc46, hi2, a1, 2);
    acc57 = vdotq_laneq_s32(acc57, hi3, a1, 3);
    return vaddq_s32(vaddq_s32(acc02, acc13), vaddq_s32(acc46, acc57));
}

static inline void q4_repacked_dot4_panel4_xor(const uint8_t *qbase,
                                               const int8_t *abase,
                                               int32x4_t acc[4]) {
    const uint8x16_t mask_hi = vdupq_n_u8(0xF0);
    int8x16_t raw0 = vld1q_s8((const int8_t *)qbase);
    int8x16_t raw1 = vld1q_s8((const int8_t *)qbase + 16);
    int8x16_t raw2 = vld1q_s8((const int8_t *)qbase + 32);
    int8x16_t raw3 = vld1q_s8((const int8_t *)qbase + 48);
    int8x16_t lo0 = vshlq_n_s8(raw0, 4);
    int8x16_t lo1 = vshlq_n_s8(raw1, 4);
    int8x16_t lo2 = vshlq_n_s8(raw2, 4);
    int8x16_t lo3 = vshlq_n_s8(raw3, 4);
    int8x16_t hi0 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw0), mask_hi));
    int8x16_t hi1 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw1), mask_hi));
    int8x16_t hi2 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw2), mask_hi));
    int8x16_t hi3 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(raw3), mask_hi));

    int8x16_t a0 = vld1q_s8(abase);
    int8x16_t a1 = vld1q_s8(abase + 16);
    int8x16_t a2 = vld1q_s8(abase + 32);
    int8x16_t a3 = vld1q_s8(abase + 48);
    int8x16_t a4 = vld1q_s8(abase + 64);
    int8x16_t a5 = vld1q_s8(abase + 80);
    int8x16_t a6 = vld1q_s8(abase + 96);
    int8x16_t a7 = vld1q_s8(abase + 112);

    acc[0] = vdotq_laneq_s32(acc[0], lo0, a0, 0);
    acc[1] = vdotq_laneq_s32(acc[1], lo0, a0, 1);
    acc[2] = vdotq_laneq_s32(acc[2], lo0, a0, 2);
    acc[3] = vdotq_laneq_s32(acc[3], lo0, a0, 3);
    acc[0] = vdotq_laneq_s32(acc[0], lo1, a1, 0);
    acc[1] = vdotq_laneq_s32(acc[1], lo1, a1, 1);
    acc[2] = vdotq_laneq_s32(acc[2], lo1, a1, 2);
    acc[3] = vdotq_laneq_s32(acc[3], lo1, a1, 3);
    acc[0] = vdotq_laneq_s32(acc[0], lo2, a2, 0);
    acc[1] = vdotq_laneq_s32(acc[1], lo2, a2, 1);
    acc[2] = vdotq_laneq_s32(acc[2], lo2, a2, 2);
    acc[3] = vdotq_laneq_s32(acc[3], lo2, a2, 3);
    acc[0] = vdotq_laneq_s32(acc[0], lo3, a3, 0);
    acc[1] = vdotq_laneq_s32(acc[1], lo3, a3, 1);
    acc[2] = vdotq_laneq_s32(acc[2], lo3, a3, 2);
    acc[3] = vdotq_laneq_s32(acc[3], lo3, a3, 3);

    acc[0] = vdotq_laneq_s32(acc[0], hi0, a4, 0);
    acc[1] = vdotq_laneq_s32(acc[1], hi0, a4, 1);
    acc[2] = vdotq_laneq_s32(acc[2], hi0, a4, 2);
    acc[3] = vdotq_laneq_s32(acc[3], hi0, a4, 3);
    acc[0] = vdotq_laneq_s32(acc[0], hi1, a5, 0);
    acc[1] = vdotq_laneq_s32(acc[1], hi1, a5, 1);
    acc[2] = vdotq_laneq_s32(acc[2], hi1, a5, 2);
    acc[3] = vdotq_laneq_s32(acc[3], hi1, a5, 3);
    acc[0] = vdotq_laneq_s32(acc[0], hi2, a6, 0);
    acc[1] = vdotq_laneq_s32(acc[1], hi2, a6, 1);
    acc[2] = vdotq_laneq_s32(acc[2], hi2, a6, 2);
    acc[3] = vdotq_laneq_s32(acc[3], hi2, a6, 3);
    acc[0] = vdotq_laneq_s32(acc[0], hi3, a7, 0);
    acc[1] = vdotq_laneq_s32(acc[1], hi3, a7, 1);
    acc[2] = vdotq_laneq_s32(acc[2], hi3, a7, 2);
    acc[3] = vdotq_laneq_s32(acc[3], hi3, a7, 3);
}

static float q4_native_row_dot(const BnQWeight *W, int row,
                               const int8_t *x_q, const float *x_scales) {
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)W->data;
    int n_blocks_per_row = W->cols / 32;
    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);
    float row_sum = 0.0f;
    for (int b = 0; b < n_blocks_per_row; b++) {
        const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
        uint8x16_t raw = vld1q_u8(blk->qs);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
        const int8_t *xb = x_q + b * 32;
        int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
        acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
        row_sum += bn_fp16_to_fp32(blk->d) * x_scales[b] * (float)vaddvq_s32(acc);
    }
    return row_sum;
}

void bn_quant_q4_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        int base = row * n_blocks_per_row;
        int b = 0;

        // Process 4 blocks at a time for better ILP
        for (; b + 3 < n_blocks_per_row; b += 4) {
            const BnBlockQ4_0 *b0 = &blocks[base + b];
            const BnBlockQ4_0 *b1 = &blocks[base + b + 1];
            const BnBlockQ4_0 *b2 = &blocks[base + b + 2];
            const BnBlockQ4_0 *b3 = &blocks[base + b + 3];
            __builtin_prefetch(b0 + 6, 0, 0);

            float d0 = bn_fp16_to_fp32(b0->d);
            float d1 = bn_fp16_to_fp32(b1->d);
            float d2 = bn_fp16_to_fp32(b2->d);
            float d3 = bn_fp16_to_fp32(b3->d);

            float dx0 = x_scales[b];
            float dx1 = x_scales[b + 1];
            float dx2 = x_scales[b + 2];
            float dx3 = x_scales[b + 3];

            const int8_t *xb0 = x_q + (b)     * 32;
            const int8_t *xb1 = x_q + (b + 1) * 32;
            const int8_t *xb2 = x_q + (b + 2) * 32;
            const int8_t *xb3 = x_q + (b + 3) * 32;

            // Block 0
            uint8x16_t raw0 = vld1q_u8(b0->qs);
            int8x16_t lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo)), eight);
            int8x16_t hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)), eight);
            int32x4_t a0 = vdotq_s32(zero, lo0, vld1q_s8(xb0));
            a0 = vdotq_s32(a0, hi0, vld1q_s8(xb0 + 16));

            // Block 1
            uint8x16_t raw1 = vld1q_u8(b1->qs);
            int8x16_t lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo)), eight);
            int8x16_t hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)), eight);
            int32x4_t a1 = vdotq_s32(zero, lo1, vld1q_s8(xb1));
            a1 = vdotq_s32(a1, hi1, vld1q_s8(xb1 + 16));

            // Block 2
            uint8x16_t raw2 = vld1q_u8(b2->qs);
            int8x16_t lo2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw2, mask_lo)), eight);
            int8x16_t hi2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw2, 4)), eight);
            int32x4_t a2 = vdotq_s32(zero, lo2, vld1q_s8(xb2));
            a2 = vdotq_s32(a2, hi2, vld1q_s8(xb2 + 16));

            // Block 3
            uint8x16_t raw3 = vld1q_u8(b3->qs);
            int8x16_t lo3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw3, mask_lo)), eight);
            int8x16_t hi3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw3, 4)), eight);
            int32x4_t a3 = vdotq_s32(zero, lo3, vld1q_s8(xb3));
            a3 = vdotq_s32(a3, hi3, vld1q_s8(xb3 + 16));

            // Deferred reduction
            row_sum += d0 * dx0 * (float)vaddvq_s32(a0)
                     + d1 * dx1 * (float)vaddvq_s32(a1)
                     + d2 * dx2 * (float)vaddvq_s32(a2)
                     + d3 * dx3 * (float)vaddvq_s32(a3);
        }

        // Tail: remaining blocks
        for (; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[base + b];
            float d_q4 = bn_fp16_to_fp32(blk->d);
            float d_q8 = x_scales[b];

            uint8x16_t raw = vld1q_u8(blk->qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);

            const int8_t *xb = x_q + b * 32;
            int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));

            row_sum += d_q4 * d_q8 * (float)vaddvq_s32(acc);
        }

        c->out[row] = row_sum;
    }
}

#define Q4_NEON_MATMUL_TILE_T 16

void bn_quant_q4_neon_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    int rows = c->W->rows;
    int cols = c->cols;
    int n_blocks_per_row = cols / 32;
    int n_tokens = c->n_tokens;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int t0 = 0; t0 < n_tokens; t0 += Q4_NEON_MATMUL_TILE_T) {
        int t_end = t0 + Q4_NEON_MATMUL_TILE_T;
        if (t_end > n_tokens) t_end = n_tokens;
        int tn = t_end - t0;

        for (int row = row_start; row < row_end; row++) {
            float sums[Q4_NEON_MATMUL_TILE_T] = { 0.0f };
            size_t base = (size_t)row * n_blocks_per_row;

            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ4_0 *blk = &blocks[base + b];
                if (b + 8 < n_blocks_per_row)
                    __builtin_prefetch(blk + 8, 0, 0);

                float wd = bn_fp16_to_fp32(blk->d);
                uint8x16_t raw = vld1q_u8(blk->qs);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);

                for (int ti = 0; ti < tn; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb = x_q + (size_t)t * cols + b * 32;
                    int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
                    acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
                    sums[ti] += wd * x_scales[(size_t)t * n_blocks_per_row + b] *
                                (float)vaddvq_s32(acc);
                }
            }

            for (int ti = 0; ti < tn; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += sums[ti];
        }
    }
}

void bn_quant_q4_repacked_neon_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    const BnPreparedWeight *prepared = c->prepared;
    const uint16_t *rp_scales = prepared ? prepared->scales : NULL;
    const uint8_t *rp_qs = prepared ? prepared->qs : NULL;
    if (!rp_scales || !rp_qs) {
        bn_quant_q4_neon_sdot_matmul_range(ctx, row_start, row_end);
        return;
    }

    int rows = c->W->rows;
    int cols = c->cols;
    int n_blocks_per_row = cols / 32;
    int n_tokens = c->n_tokens;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);

    int row = row_start;
    for (; row < row_end && (row & 3); row++) {
        for (int t = 0; t < n_tokens; t++) {
            float sum = 0.0f;
            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                float wd = bn_fp16_to_fp32(blk->d);
                uint8x16_t raw = vld1q_u8(blk->qs);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
                const int8_t *xb = x_q + (size_t)t * cols + b * 32;
                int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
                acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
                sum += wd * x_scales[(size_t)t * n_blocks_per_row + b] *
                       (float)vaddvq_s32(acc);
            }
            c->out[(size_t)t * rows + row] += sum;
        }
    }

    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;

        for (int t0 = 0; t0 < n_tokens; t0 += Q4_NEON_MATMUL_TILE_T) {
            int t_end = t0 + Q4_NEON_MATMUL_TILE_T;
            if (t_end > n_tokens) t_end = n_tokens;
            int tn = t_end - t0;
            float32x4_t sums[Q4_NEON_MATMUL_TILE_T];
            for (int ti = 0; ti < tn; ti++)
                sums[ti] = vdupq_n_f32(0.0f);

            for (int b = 0; b < n_blocks_per_row; b++) {
                size_t gb = (size_t)group * n_blocks_per_row + b;
                const uint8_t *qbase = rp_qs + gb * 64;
                if (b + 8 < n_blocks_per_row)
                    __builtin_prefetch(rp_qs + (gb + 8) * 64, 0, 0);

                float32x4_t d4 =
                    vcvt_f32_f16(vld1_f16((const float16_t *)(rp_scales + gb * 4)));
                for (int ti = 0; ti < tn; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb = x_q + (size_t)t * cols + b * 32;
                    int8x16_t a0 = vld1q_s8(xb);
                    int8x16_t a1 = vld1q_s8(xb + 16);
                    int32x4_t acc = q4_repacked_dot4_xor(qbase, a0, a1);
                    float32x4_t f = vcvtq_n_f32_s32(acc, 4);
                    float dx = x_scales[(size_t)t * n_blocks_per_row + b];
                    sums[ti] = vfmaq_f32(sums[ti], f, vmulq_n_f32(d4, dx));
                }
            }

            for (int ti = 0; ti < tn; ti++)
                vst1q_f32(c->out + (size_t)(t0 + ti) * rows + row, sums[ti]);
        }
    }

    for (; row < row_end; row++) {
        for (int t = 0; t < n_tokens; t++) {
            float sum = 0.0f;
            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                float wd = bn_fp16_to_fp32(blk->d);
                uint8x16_t raw = vld1q_u8(blk->qs);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
                const int8_t *xb = x_q + (size_t)t * cols + b * 32;
                int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
                acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
                sum += wd * x_scales[(size_t)t * n_blocks_per_row + b] *
                       (float)vaddvq_s32(acc);
            }
            c->out[(size_t)t * rows + row] += sum;
        }
    }
}

void bn_quant_q4_repacked_neon_sdot_matmul_group_range(void *ctx,
                                                        int group_start,
                                                        int group_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    const BnPreparedWeight *prepared = c->prepared;
    const uint16_t *rp_scales = prepared ? prepared->scales : NULL;
    const uint8_t *rp_qs = prepared ? prepared->qs : NULL;
    if (!rp_scales || !rp_qs) {
        bn_quant_q4_neon_sdot_matmul_range(ctx, group_start * 4, group_end * 4);
        return;
    }

    int rows = c->W->rows;
    int cols = c->cols;
    int n_blocks_per_row = cols / 32;
    int n_tokens = c->n_tokens;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int group = group_start; group < group_end; group++) {
        int row = group << 2;

        for (int t0 = 0; t0 < n_tokens; t0 += Q4_NEON_MATMUL_TILE_T) {
            int t_end = t0 + Q4_NEON_MATMUL_TILE_T;
            if (t_end > n_tokens) t_end = n_tokens;
            int tn = t_end - t0;
            float32x4_t sums[Q4_NEON_MATMUL_TILE_T];
            for (int ti = 0; ti < tn; ti++)
                sums[ti] = vdupq_n_f32(0.0f);

            for (int b = 0; b < n_blocks_per_row; b++) {
                size_t gb = (size_t)group * n_blocks_per_row + b;
                const uint8_t *qbase = rp_qs + gb * 64;
                if (b + 8 < n_blocks_per_row)
                    __builtin_prefetch(rp_qs + (gb + 8) * 64, 0, 0);

                float32x4_t d4 =
                    vcvt_f32_f16(vld1_f16((const float16_t *)(rp_scales + gb * 4)));
                for (int ti = 0; ti < tn; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb = x_q + (size_t)t * cols + b * 32;
                    int8x16_t a0 = vld1q_s8(xb);
                    int8x16_t a1 = vld1q_s8(xb + 16);
                    int32x4_t acc = q4_repacked_dot4_xor(qbase, a0, a1);
                    float32x4_t f = vcvtq_n_f32_s32(acc, 4);
                    float dx = x_scales[(size_t)t * n_blocks_per_row + b];
                    sums[ti] = vfmaq_f32(sums[ti], f, vmulq_n_f32(d4, dx));
                }
            }

            for (int ti = 0; ti < tn; ti++)
                vst1q_f32(c->out + (size_t)(t0 + ti) * rows + row, sums[ti]);
        }
    }
}

void bn_quant_q4_repacked_neon_sdot_matmul_panel4_range(void *ctx,
                                                        int group_start,
                                                        int group_end) {
    BnQ4MatmulCtx *c = (BnQ4MatmulCtx *)ctx;
    const BnPreparedWeight *prepared = c->prepared;
    const uint16_t *rp_scales = prepared ? prepared->scales : NULL;
    const uint8_t *rp_qs = prepared ? prepared->qs : NULL;
    const int8_t *x_q4 = c->x_q4;
    const float *x_scales4 = c->x_scales4;
    if (!rp_scales || !rp_qs || !x_q4 || !x_scales4 || c->n_token_panels <= 0) {
        bn_quant_q4_repacked_neon_sdot_matmul_group_range(ctx, group_start, group_end);
        return;
    }

    int rows = c->W->rows;
    int cols = c->cols;
    int n_blocks_per_row = cols / 32;
    int n_tokens = c->n_tokens;
    int n_full_panels = n_tokens / 4;
    const int32x4_t zero = vdupq_n_s32(0);

    for (int group = group_start; group < group_end; group++) {
        int row = group << 2;

        for (int p = 0; p < n_full_panels; p++) {
            float32x4_t sums[4] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
            };
            for (int b = 0; b < n_blocks_per_row; b++) {
                size_t gb = (size_t)group * n_blocks_per_row + b;
                const uint8_t *qbase = rp_qs + gb * 64;
                const int8_t *abase = x_q4 + ((size_t)p * n_blocks_per_row + b) * 128;
                if (b + 8 < n_blocks_per_row) {
                    __builtin_prefetch(rp_qs + (gb + 8) * 64, 0, 0);
                    __builtin_prefetch(x_q4 + ((size_t)p * n_blocks_per_row + b + 8) * 128, 0, 0);
                }

                int32x4_t acc[4] = { zero, zero, zero, zero };
                q4_repacked_dot4_panel4_xor(qbase, abase, acc);
                float32x4_t d4 =
                    vcvt_f32_f16(vld1_f16((const float16_t *)(rp_scales + gb * 4)));
                const float *dx = x_scales4 + ((size_t)p * n_blocks_per_row + b) * 4;
                sums[0] = vfmaq_f32(sums[0], vcvtq_n_f32_s32(acc[0], 4), vmulq_n_f32(d4, dx[0]));
                sums[1] = vfmaq_f32(sums[1], vcvtq_n_f32_s32(acc[1], 4), vmulq_n_f32(d4, dx[1]));
                sums[2] = vfmaq_f32(sums[2], vcvtq_n_f32_s32(acc[2], 4), vmulq_n_f32(d4, dx[2]));
                sums[3] = vfmaq_f32(sums[3], vcvtq_n_f32_s32(acc[3], 4), vmulq_n_f32(d4, dx[3]));
            }

            int t = p * 4;
            vst1q_f32(c->out + (size_t)(t + 0) * rows + row, sums[0]);
            vst1q_f32(c->out + (size_t)(t + 1) * rows + row, sums[1]);
            vst1q_f32(c->out + (size_t)(t + 2) * rows + row, sums[2]);
            vst1q_f32(c->out + (size_t)(t + 3) * rows + row, sums[3]);
        }

        for (int t = n_full_panels * 4; t < n_tokens; t++) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int b = 0; b < n_blocks_per_row; b++) {
                size_t gb = (size_t)group * n_blocks_per_row + b;
                const uint8_t *qbase = rp_qs + gb * 64;
                const int8_t *xb = c->x_q + (size_t)t * cols + b * 32;
                int32x4_t acc = q4_repacked_dot4_xor(qbase,
                                                      vld1q_s8(xb),
                                                      vld1q_s8(xb + 16));
                float32x4_t f = vcvtq_n_f32_s32(acc, 4);
                float32x4_t d4 =
                    vcvt_f32_f16(vld1_f16((const float16_t *)(rp_scales + gb * 4)));
                float dx = c->x_scales[(size_t)t * n_blocks_per_row + b];
                sum = vfmaq_f32(sum, f, vmulq_n_f32(d4, dx));
            }
            vst1q_f32(c->out + (size_t)t * rows + row, sum);
        }
    }
}

void bn_quant_q4_repacked_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const uint16_t *rp_scales = c->prepared ? c->prepared->scales : NULL;
    const uint8_t *rp_qs = c->prepared ? c->prepared->qs : NULL;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);

    int row = row_start;

    // Pre-tail: rows before first 4-aligned boundary
    for (; row < row_end && (row & 3); row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float dx = x_scales[b];
            const int8_t *xb = x_q + b * 32;

            uint8x16_t raw = vld1q_u8(blk->qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
            row_sum += d * dx * (float)vaddvq_s32(acc);
        }
        c->out[row] = row_sum;
    }

    // Main loop: 4 rows at a time (nibble-transposed layout + vdotq_laneq_s32)
    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        float32x4_t row_sums = vdupq_n_f32(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            // Load activation ONCE per block
            int8x16_t a0 = vld1q_s8(x_q + b * 32);
            int8x16_t a1 = vld1q_s8(x_q + b * 32 + 16);
            float dx = x_scales[b];

            size_t gb = (size_t)group * n_blocks_per_row + b;
            const uint8_t *qbase = rp_qs + gb * 64;
            if (b + 8 < n_blocks_per_row)
                __builtin_prefetch(rp_qs + (gb + 8) * 64, 0, 0);

            // Repack XORs 0x88 so signed shift/mask yields centered Q4 values
            // without subtracts.
            int32x4_t acc = q4_repacked_dot4_xor(qbase, a0, a1);

            // Vector scale multiply + accumulate (no per-row reduction!)
            float32x4_t f = vcvtq_n_f32_s32(acc, 4);
            float32x4_t d4 = vcvt_f32_f16(vld1_f16((const float16_t *)(rp_scales + gb * 4)));
            row_sums = vfmaq_f32(row_sums, f, vmulq_n_f32(d4, dx));
        }

        vst1q_f32(&c->out[row], row_sums);
    }

    // Post-tail: remaining 0-3 rows
    for (; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            float dx = x_scales[b];
            const int8_t *xb = x_q + b * 32;

            uint8x16_t raw = vld1q_u8(blk->qs);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
            acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
            row_sum += d * dx * (float)vaddvq_s32(acc);
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q4_repacked_gate_up_silu_neon_range(void *ctx, int row_start, int row_end) {
    BnQ4GateUpCtx *c = (BnQ4GateUpCtx *)ctx;
    const BnQWeight *gate = c->gate;
    const BnQWeight *up = c->up;
    const uint16_t *gate_scales = c->gate_prepared ? c->gate_prepared->scales : NULL;
    const uint16_t *up_scales = c->up_prepared ? c->up_prepared->scales : NULL;
    const uint8_t *gate_qs = c->gate_prepared ? c->gate_prepared->qs : NULL;
    const uint8_t *up_qs = c->up_prepared ? c->up_prepared->qs : NULL;
    int n_blocks_per_row = gate->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    int row = row_start;
    for (; row < row_end && (row & 3); row++) {
        float g = q4_native_row_dot(gate, row, x_q, x_scales);
        float u = q4_native_row_dot(up, row, x_q, x_scales);
        c->out[row] = (g / (1.0f + expf(-g))) * u;
    }

    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        float32x4_t gate_sum = vdupq_n_f32(0.0f);
        float32x4_t up_sum = vdupq_n_f32(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            int8x16_t a0 = vld1q_s8(x_q + b * 32);
            int8x16_t a1 = vld1q_s8(x_q + b * 32 + 16);
            float dx = x_scales[b];
            size_t gb = (size_t)group * n_blocks_per_row + b;
            if (b + 8 < n_blocks_per_row) {
                __builtin_prefetch(gate_qs + (gb + 8) * 64, 0, 0);
                __builtin_prefetch(up_qs + (gb + 8) * 64, 0, 0);
            }

            int32x4_t g_acc = q4_repacked_dot4_xor(gate_qs + gb * 64, a0, a1);
            float32x4_t g_f = vcvtq_n_f32_s32(g_acc, 4);
            float32x4_t g_d = vcvt_f32_f16(vld1_f16((const float16_t *)(gate_scales + gb * 4)));
            gate_sum = vfmaq_f32(gate_sum, g_f, vmulq_n_f32(g_d, dx));

            int32x4_t u_acc = q4_repacked_dot4_xor(up_qs + gb * 64, a0, a1);
            float32x4_t u_f = vcvtq_n_f32_s32(u_acc, 4);
            float32x4_t u_d = vcvt_f32_f16(vld1_f16((const float16_t *)(up_scales + gb * 4)));
            up_sum = vfmaq_f32(up_sum, u_f, vmulq_n_f32(u_d, dx));
        }

        vst1q_f32(c->out + row, vmulq_f32(bn_neon_fast_silu_f32(gate_sum), up_sum));
    }

    for (; row < row_end; row++) {
        float g = q4_native_row_dot(gate, row, x_q, x_scales);
        float u = q4_native_row_dot(up, row, x_q, x_scales);
        c->out[row] = (g / (1.0f + expf(-g))) * u;
    }
}
