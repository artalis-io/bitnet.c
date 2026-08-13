#include "quant_ctx.h"
#include "simd_helpers.h"
#include <arm_neon.h>
#include <math.h>

static inline float q4_neon_fp16_to_fp32(uint16_t bits) {
    float16x4_t half = vreinterpret_f16_u16(vdup_n_u16(bits));
    return vgetq_lane_f32(vcvt_f32_f16(half), 0);
}

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

static inline void q4_repacked_dot4_panel8_xor(const uint8_t *qbase,
                                               const int8_t *abase0,
                                               const int8_t *abase1,
                                               int32x4_t acc[8]) {
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

#define BN_Q4_ACC_PANEL(ai, abase) do {                                      \
        int8x16_t a0 = vld1q_s8((abase));                                    \
        int8x16_t a1 = vld1q_s8((abase) + 16);                               \
        int8x16_t a2 = vld1q_s8((abase) + 32);                               \
        int8x16_t a3 = vld1q_s8((abase) + 48);                               \
        int8x16_t a4 = vld1q_s8((abase) + 64);                               \
        int8x16_t a5 = vld1q_s8((abase) + 80);                               \
        int8x16_t a6 = vld1q_s8((abase) + 96);                               \
        int8x16_t a7 = vld1q_s8((abase) + 112);                              \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], lo0, a0, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], lo0, a0, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], lo0, a0, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], lo0, a0, 3);          \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], lo1, a1, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], lo1, a1, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], lo1, a1, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], lo1, a1, 3);          \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], lo2, a2, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], lo2, a2, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], lo2, a2, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], lo2, a2, 3);          \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], lo3, a3, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], lo3, a3, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], lo3, a3, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], lo3, a3, 3);          \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], hi0, a4, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], hi0, a4, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], hi0, a4, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], hi0, a4, 3);          \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], hi1, a5, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], hi1, a5, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], hi1, a5, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], hi1, a5, 3);          \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], hi2, a6, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], hi2, a6, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], hi2, a6, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], hi2, a6, 3);          \
        acc[(ai) + 0] = vdotq_laneq_s32(acc[(ai) + 0], hi3, a7, 0);          \
        acc[(ai) + 1] = vdotq_laneq_s32(acc[(ai) + 1], hi3, a7, 1);          \
        acc[(ai) + 2] = vdotq_laneq_s32(acc[(ai) + 2], hi3, a7, 2);          \
        acc[(ai) + 3] = vdotq_laneq_s32(acc[(ai) + 3], hi3, a7, 3);          \
    } while (0)

    BN_Q4_ACC_PANEL(0, abase0);
    BN_Q4_ACC_PANEL(4, abase1);
#undef BN_Q4_ACC_PANEL
}

static float q4_native_row_dot(const BnQWeight *W, int row,
                               const int8_t *x_q, const float *x_scales) {
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)W->data;
    int n_blocks_per_row = W->cols / 32;
    const uint8x16_t mask_lo = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t sums[2] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
    };
    for (int b = 0; b < n_blocks_per_row; b++) {
        const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
        uint8x16_t raw = vld1q_u8(blk->qs);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), eight);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
        const int8_t *xb = x_q + b * 32;
        int32x4_t acc = vdotq_s32(zero, lo, vld1q_s8(xb));
        acc = vdotq_s32(acc, hi, vld1q_s8(xb + 16));
        float scale = q4_neon_fp16_to_fp32(blk->d) * x_scales[b];
        sums[b & 1] = vfmaq_n_f32(sums[b & 1], vcvtq_f32_s32(acc), scale);
    }
    return vaddvq_f32(sums[0]) + vaddvq_f32(sums[1]);
}

void bn_quant_q4_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    for (int row = row_start; row < row_end; row++)
        c->out[row] = q4_native_row_dot(
            c->W, row, c->x_q, c->x_scales);
}


void bn_quant_q4_neon_sdot_4row_range(void *ctx,
                                      int group_start,
                                      int group_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    for (int group = group_start; group < group_end; group++) {
        int row0 = group * 4;
        int nr = c->W->rows - row0;
        if (nr > 4) nr = 4;
        if (nr <= 0) continue;

        for (int r = 0; r < nr; r++)
            c->out[row0 + r] = q4_native_row_dot(
                c->W, row0 + r, c->x_q, c->x_scales);
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

                float wd = q4_neon_fp16_to_fp32(blk->d);
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
                float wd = q4_neon_fp16_to_fp32(blk->d);
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
                float wd = q4_neon_fp16_to_fp32(blk->d);
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

        int p = 0;
        for (; p + 1 < n_full_panels; p += 2) {
            float32x4_t sums[8] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
            };
            for (int b = 0; b < n_blocks_per_row; b++) {
                size_t gb = (size_t)group * n_blocks_per_row + b;
                const uint8_t *qbase = rp_qs + gb * 64;
                const int8_t *abase0 = x_q4 + ((size_t)p * n_blocks_per_row + b) * 128;
                const int8_t *abase1 = abase0 + (size_t)n_blocks_per_row * 128;
                if (b + 8 < n_blocks_per_row) {
                    __builtin_prefetch(rp_qs + (gb + 8) * 64, 0, 0);
                    __builtin_prefetch(x_q4 + ((size_t)p * n_blocks_per_row + b + 8) * 128, 0, 0);
                    __builtin_prefetch(x_q4 + ((size_t)(p + 1) * n_blocks_per_row + b + 8) * 128, 0, 0);
                }

                int32x4_t acc[8] = { zero, zero, zero, zero, zero, zero, zero, zero };
                q4_repacked_dot4_panel8_xor(qbase, abase0, abase1, acc);
                float32x4_t d4 =
                    vcvt_f32_f16(vld1_f16((const float16_t *)(rp_scales + gb * 4)));
                const float *dx0 = x_scales4 + ((size_t)p * n_blocks_per_row + b) * 4;
                const float *dx1 = dx0 + (size_t)n_blocks_per_row * 4;
                sums[0] = vfmaq_f32(sums[0], vcvtq_n_f32_s32(acc[0], 4), vmulq_n_f32(d4, dx0[0]));
                sums[1] = vfmaq_f32(sums[1], vcvtq_n_f32_s32(acc[1], 4), vmulq_n_f32(d4, dx0[1]));
                sums[2] = vfmaq_f32(sums[2], vcvtq_n_f32_s32(acc[2], 4), vmulq_n_f32(d4, dx0[2]));
                sums[3] = vfmaq_f32(sums[3], vcvtq_n_f32_s32(acc[3], 4), vmulq_n_f32(d4, dx0[3]));
                sums[4] = vfmaq_f32(sums[4], vcvtq_n_f32_s32(acc[4], 4), vmulq_n_f32(d4, dx1[0]));
                sums[5] = vfmaq_f32(sums[5], vcvtq_n_f32_s32(acc[5], 4), vmulq_n_f32(d4, dx1[1]));
                sums[6] = vfmaq_f32(sums[6], vcvtq_n_f32_s32(acc[6], 4), vmulq_n_f32(d4, dx1[2]));
                sums[7] = vfmaq_f32(sums[7], vcvtq_n_f32_s32(acc[7], 4), vmulq_n_f32(d4, dx1[3]));
            }

            int t = p * 4;
            vst1q_f32(c->out + (size_t)(t + 0) * rows + row, sums[0]);
            vst1q_f32(c->out + (size_t)(t + 1) * rows + row, sums[1]);
            vst1q_f32(c->out + (size_t)(t + 2) * rows + row, sums[2]);
            vst1q_f32(c->out + (size_t)(t + 3) * rows + row, sums[3]);
            vst1q_f32(c->out + (size_t)(t + 4) * rows + row, sums[4]);
            vst1q_f32(c->out + (size_t)(t + 5) * rows + row, sums[5]);
            vst1q_f32(c->out + (size_t)(t + 6) * rows + row, sums[6]);
            vst1q_f32(c->out + (size_t)(t + 7) * rows + row, sums[7]);
        }

        for (; p < n_full_panels; p++) {
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
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    int row = row_start;

    for (; row < row_end && (row & 3); row++) {
        c->out[row] = q4_native_row_dot(c->W, row, x_q, x_scales);
    }

    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        float32x4_t sum = vdupq_n_f32(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            int8x16_t a0 = vld1q_s8(x_q + b * 32);
            int8x16_t a1 = vld1q_s8(x_q + b * 32 + 16);
            float dx = x_scales[b];

            size_t gb = (size_t)group * n_blocks_per_row + b;
            const uint8_t *qbase = rp_qs + gb * 64;
            if (b + 8 < n_blocks_per_row)
                __builtin_prefetch(rp_qs + (gb + 8) * 64, 0, 0);

            int32x4_t dot = q4_repacked_dot4_xor(qbase, a0, a1);
            float32x4_t scales =
                vcvt_f32_f16(vld1_f16((const float16_t *)(rp_scales + gb * 4)));
            float32x4_t scale = vmulq_n_f32(scales, dx);
            sum = vfmaq_f32(sum, vcvtq_n_f32_s32(dot, 4), scale);
        }
        vst1q_f32(c->out + row, sum);
    }

    for (; row < row_end; row++) {
        c->out[row] = q4_native_row_dot(c->W, row, x_q, x_scales);
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
        float32x4_t silu = bn_neon_fast_silu_f32(vdupq_n_f32(g));
        c->out[row] = vgetq_lane_f32(silu, 0) * u;
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

            int32x4_t gate_dot = q4_repacked_dot4_xor(
                gate_qs + gb * 64, a0, a1);
            int32x4_t up_dot = q4_repacked_dot4_xor(
                up_qs + gb * 64, a0, a1);
            float32x4_t gate_d = vcvt_f32_f16(
                vld1_f16((const float16_t *)(gate_scales + gb * 4)));
            float32x4_t up_d = vcvt_f32_f16(
                vld1_f16((const float16_t *)(up_scales + gb * 4)));
            float32x4_t gate_scale = vmulq_n_f32(gate_d, dx);
            float32x4_t up_scale = vmulq_n_f32(up_d, dx);
            gate_sum = vfmaq_f32(
                gate_sum, vcvtq_n_f32_s32(gate_dot, 4), gate_scale);
            up_sum = vfmaq_f32(
                up_sum, vcvtq_n_f32_s32(up_dot, 4), up_scale);
        }
        float32x4_t silu = bn_neon_fast_silu_f32(gate_sum);
        vst1q_f32(c->out + row, vmulq_f32(silu, up_sum));
    }

    for (; row < row_end; row++) {
        float g = q4_native_row_dot(gate, row, x_q, x_scales);
        float u = q4_native_row_dot(up, row, x_q, x_scales);
        float32x4_t silu = bn_neon_fast_silu_f32(vdupq_n_f32(g));
        c->out[row] = vgetq_lane_f32(silu, 0) * u;
    }
}
