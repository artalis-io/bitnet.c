#include "quant_ctx.h"
#include <arm_neon.h>

static inline float q6k_fp16_to_f32(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vld1_dup_f16((const float16_t *)&h)), 0);
}

static inline int32_t q6k_neon_bias_correction(const int8_t *sc,
                                                const int16_t *bsums) {
    const int8x16_t scales = vld1q_s8(sc);
    const int16x8_t scales_lo = vmovl_s8(vget_low_s8(scales));
    const int16x8_t scales_hi = vmovl_s8(vget_high_s8(scales));
    const int16x8_t bsums_lo = vld1q_s16(bsums);
    const int16x8_t bsums_hi = vld1q_s16(bsums + 8);
    const int32x4_t prod = vaddq_s32(
        vaddq_s32(vmull_s16(vget_low_s16(scales_lo),
                            vget_low_s16(bsums_lo)),
                  vmull_s16(vget_high_s16(scales_lo),
                            vget_high_s16(bsums_lo))),
        vaddq_s32(vmull_s16(vget_low_s16(scales_hi),
                            vget_low_s16(bsums_hi)),
                  vmull_s16(vget_high_s16(scales_hi),
                            vget_high_s16(bsums_hi))));
    return vaddvq_s32(prod);
}

// Q6_K SDOT kernel with Q8_K x quantization:
// - Unsigned 6-bit weights (no bias-32 subtract)
// - Integer accumulation within super-block (one x_d per 256 elements)
// - Bias correction via bsums (integer, outside inner loop)
// - Float conversion once per super-block
void bn_quant_q6k_neon_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ6KSdotCtx *c = (BnQ6KSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;
    const uint8x16_t mask_lo4 = vdupq_n_u8(0xF);
    const uint8x16_t mask_2 = vdupq_n_u8(3);
    const int32x4_t zero = vdupq_n_s32(0);

    int row = row_start;
    for (; row + 1 < row_end; row += 2) {
        float row_sum0 = 0.0f;
        float row_sum1 = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk0 =
                &blocks[(size_t)row * n_blocks_per_row + b];
            const BnBlockQ6K *blk1 = blk0 + n_blocks_per_row;
            __builtin_prefetch(blk0 + 1, 0, 0);
            __builtin_prefetch(blk1 + 1, 0, 0);
            float d0 = q6k_fp16_to_f32(blk0->d);
            float d1 = q6k_fp16_to_f32(blk1->d);
            float dx = x_d[b];
            const uint8_t *ql0p = blk0->ql;
            const uint8_t *qh0p = blk0->qh;
            const int8_t *sc0 = blk0->scales;
            const uint8_t *ql1p = blk1->ql;
            const uint8_t *qh1p = blk1->qh;
            const int8_t *sc1 = blk1->scales;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            int32x4_t acc0 = zero;
            int32x4_t acc1 = zero;

            for (int chunk = 0; chunk < 2; chunk++) {
                const uint8x16_t qla0 = vld1q_u8(ql0p);
                const uint8x16_t qla1 = vld1q_u8(ql0p + 16);
                const uint8x16_t qla2 = vld1q_u8(ql0p + 32);
                const uint8x16_t qla3 = vld1q_u8(ql0p + 48);
                const uint8x16_t qha0 = vld1q_u8(qh0p);
                const uint8x16_t qha1 = vld1q_u8(qh0p + 16);
                const uint8x16_t qlb0 = vld1q_u8(ql1p);
                const uint8x16_t qlb1 = vld1q_u8(ql1p + 16);
                const uint8x16_t qlb2 = vld1q_u8(ql1p + 32);
                const uint8x16_t qlb3 = vld1q_u8(ql1p + 48);
                const uint8x16_t qhb0 = vld1q_u8(qh1p);
                const uint8x16_t qhb1 = vld1q_u8(qh1p + 16);

#define Q6_ACC_PAIR(xoff, scale_idx, low_a, high_a, low_b, high_b) do { \
                    const int8x16_t xv = vld1q_s8(xb + (xoff));       \
                    const int8x16_t wa = vreinterpretq_s8_u8(        \
                        vorrq_u8((low_a), (high_a)));                 \
                    const int8x16_t wb = vreinterpretq_s8_u8(        \
                        vorrq_u8((low_b), (high_b)));                 \
                    acc0 = vmlaq_n_s32(                               \
                        acc0, vdotq_s32(zero, wa, xv), sc0[scale_idx]); \
                    acc1 = vmlaq_n_s32(                               \
                        acc1, vdotq_s32(zero, wb, xv), sc1[scale_idx]); \
                } while (0)

                // Keep Q6 values unsigned and apply the -32 correction once
                // after the complete Q8_K super-block has accumulated.
                Q6_ACC_PAIR(0, 0, vandq_u8(qla0, mask_lo4),
                            vshlq_n_u8(vandq_u8(qha0, mask_2), 4),
                            vandq_u8(qlb0, mask_lo4),
                            vshlq_n_u8(vandq_u8(qhb0, mask_2), 4));
                Q6_ACC_PAIR(16, 1, vandq_u8(qla1, mask_lo4),
                            vshlq_n_u8(vandq_u8(qha1, mask_2), 4),
                            vandq_u8(qlb1, mask_lo4),
                            vshlq_n_u8(vandq_u8(qhb1, mask_2), 4));
                Q6_ACC_PAIR(32, 2, vandq_u8(qla2, mask_lo4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qha0, 2), mask_2), 4),
                            vandq_u8(qlb2, mask_lo4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qhb0, 2), mask_2), 4));
                Q6_ACC_PAIR(48, 3, vandq_u8(qla3, mask_lo4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qha1, 2), mask_2), 4),
                            vandq_u8(qlb3, mask_lo4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qhb1, 2), mask_2), 4));
                Q6_ACC_PAIR(64, 4, vshrq_n_u8(qla0, 4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qha0, 4), mask_2), 4),
                            vshrq_n_u8(qlb0, 4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qhb0, 4), mask_2), 4));
                Q6_ACC_PAIR(80, 5, vshrq_n_u8(qla1, 4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qha1, 4), mask_2), 4),
                            vshrq_n_u8(qlb1, 4),
                            vshlq_n_u8(vandq_u8(vshrq_n_u8(qhb1, 4), mask_2), 4));
                Q6_ACC_PAIR(96, 6, vshrq_n_u8(qla2, 4),
                            vshlq_n_u8(vshrq_n_u8(qha0, 6), 4),
                            vshrq_n_u8(qlb2, 4),
                            vshlq_n_u8(vshrq_n_u8(qhb0, 6), 4));
                Q6_ACC_PAIR(112, 7, vshrq_n_u8(qla3, 4),
                            vshlq_n_u8(vshrq_n_u8(qha1, 6), 4),
                            vshrq_n_u8(qlb3, 4),
                            vshlq_n_u8(vshrq_n_u8(qhb1, 6), 4));
#undef Q6_ACC_PAIR
                xb += 128;
                ql0p += 64;
                qh0p += 32;
                sc0 += 8;
                ql1p += 64;
                qh1p += 32;
                sc1 += 8;
            }
            const int32_t bias0 =
                q6k_neon_bias_correction(blk0->scales, bsums);
            const int32_t bias1 =
                q6k_neon_bias_correction(blk1->scales, bsums);
            row_sum0 += d0 * dx * (float)(vaddvq_s32(acc0) - 32 * bias0);
            row_sum1 += d1 * dx * (float)(vaddvq_s32(acc1) - 32 * bias1);
        }
        c->out[row] = row_sum0;
        c->out[row + 1] = row_sum1;
    }

    for (; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk =
                &blocks[(size_t)row * n_blocks_per_row + b];
            float d = q6k_fp16_to_f32(blk->d);
            float dx = x_d[b];
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t *sc = blk->scales;
            const int8_t *xb = x_q + b * BN_QK_K;
            int32x4_t acc = zero;

            for (int chunk = 0; chunk < 2; chunk++) {
                const uint8x16_t ql0 = vld1q_u8(ql);
                const uint8x16_t ql1 = vld1q_u8(ql + 16);
                const uint8x16_t ql2 = vld1q_u8(ql + 32);
                const uint8x16_t ql3 = vld1q_u8(ql + 48);
                const uint8x16_t qh0 = vld1q_u8(qh);
                const uint8x16_t qh1 = vld1q_u8(qh + 16);
#define Q6_ACC_ONE(xoff, scale_idx, low, high) do {                  \
                    const int8x16_t w = vreinterpretq_s8_u8(        \
                        vorrq_u8((low), (high)));                    \
                    acc = vmlaq_n_s32(                               \
                        acc, vdotq_s32(zero, w,                      \
                                       vld1q_s8(xb + (xoff))),       \
                        sc[scale_idx]);                              \
                } while (0)
                Q6_ACC_ONE(0, 0, vandq_u8(ql0, mask_lo4),
                           vshlq_n_u8(vandq_u8(qh0, mask_2), 4));
                Q6_ACC_ONE(16, 1, vandq_u8(ql1, mask_lo4),
                           vshlq_n_u8(vandq_u8(qh1, mask_2), 4));
                Q6_ACC_ONE(32, 2, vandq_u8(ql2, mask_lo4),
                           vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_2), 4));
                Q6_ACC_ONE(48, 3, vandq_u8(ql3, mask_lo4),
                           vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_2), 4));
                Q6_ACC_ONE(64, 4, vshrq_n_u8(ql0, 4),
                           vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_2), 4));
                Q6_ACC_ONE(80, 5, vshrq_n_u8(ql1, 4),
                           vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_2), 4));
                Q6_ACC_ONE(96, 6, vshrq_n_u8(ql2, 4),
                           vshlq_n_u8(vshrq_n_u8(qh0, 6), 4));
                Q6_ACC_ONE(112, 7, vshrq_n_u8(ql3, 4),
                           vshlq_n_u8(vshrq_n_u8(qh1, 6), 4));
#undef Q6_ACC_ONE
                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
            const int32_t bias = q6k_neon_bias_correction(
                blk->scales, x_bsums + b * 16);
            row_sum += d * dx * (float)(vaddvq_s32(acc) - 32 * bias);
        }
        c->out[row] = row_sum;
    }
}

// Fused Q6_K matmul: load weight block once, dot against all n_tokens x vectors.
void bn_quant_q6k_neon_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;

    const uint8x16_t mask_lo4 = vdupq_n_u8(0xF);
    const uint8x16_t mask_2 = vdupq_n_u8(3);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int row = row_start; row < row_end; row++) {
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d = q6k_fp16_to_f32(blk->d);

            // Pre-unpack weight vectors for both chunks (stays in L1 across tokens)
            int8x16_t W_all[16];  // 2 chunks × 8 vectors
            {
                const uint8_t *ql = blk->ql;
                const uint8_t *qh = blk->qh;
                for (int chunk = 0; chunk < 2; chunk++) {
                    uint8x16_t ql0 = vld1q_u8(ql), ql1 = vld1q_u8(ql + 16);
                    uint8x16_t ql2 = vld1q_u8(ql + 32), ql3 = vld1q_u8(ql + 48);
                    uint8x16_t qh0 = vld1q_u8(qh), qh1 = vld1q_u8(qh + 16);
                    int base = chunk * 8;
                    W_all[base+0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, mask_lo4), vshlq_n_u8(vandq_u8(qh0, mask_2), 4)));
                    W_all[base+1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, mask_lo4), vshlq_n_u8(vandq_u8(qh1, mask_2), 4)));
                    W_all[base+2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, mask_lo4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_2), 4)));
                    W_all[base+3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, mask_lo4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), mask_2), 4)));
                    W_all[base+4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_2), 4)));
                    W_all[base+5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), mask_2), 4)));
                    W_all[base+6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql2, 4), vshlq_n_u8(vshrq_n_u8(qh0, 6), 4)));
                    W_all[base+7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql3, 4), vshlq_n_u8(vshrq_n_u8(qh1, 6), 4)));
                    ql += 64; qh += 32;
                }
            }

            // Pre-read scales (16 int8 values)
            const int8_t *sc_base = blk->scales;

            for (int t = 0; t < n_tokens; t++) {
                const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                float dx = c->x_d[(size_t)t * n_bpr + b];
                const int16_t *bsums = c->x_bsums + ((size_t)t * n_bpr + b) * 16;
                const int8_t *sc = sc_base;

                int32x4_t acc = zero;
                for (int chunk = 0; chunk < 2; chunk++) {
                    int base = chunk * 8;
                    const int8_t *xbc = xb + chunk * 128;

                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+0], vld1q_s8(xbc)), (int32_t)sc[0]);
                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+1], vld1q_s8(xbc + 16)), (int32_t)sc[1]);
                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+2], vld1q_s8(xbc + 32)), (int32_t)sc[2]);
                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+3], vld1q_s8(xbc + 48)), (int32_t)sc[3]);
                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+4], vld1q_s8(xbc + 64)), (int32_t)sc[4]);
                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+5], vld1q_s8(xbc + 80)), (int32_t)sc[5]);
                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+6], vld1q_s8(xbc + 96)), (int32_t)sc[6]);
                    acc = vmlaq_n_s32(acc, vdotq_s32(zero, W_all[base+7], vld1q_s8(xbc + 112)), (int32_t)sc[7]);

                    sc += 8;
                }

                const int8x16_t scales = vld1q_s8(sc_base);
                const int16x8_t scales_lo = vmovl_s8(vget_low_s8(scales));
                const int16x8_t scales_hi = vmovl_s8(vget_high_s8(scales));
                const int16x8_t bsums_lo = vld1q_s16(bsums);
                const int16x8_t bsums_hi = vld1q_s16(bsums + 8);
                const int32x4_t bias_prod = vaddq_s32(
                    vaddq_s32(vmull_s16(vget_low_s16(scales_lo), vget_low_s16(bsums_lo)),
                              vmull_s16(vget_high_s16(scales_lo), vget_high_s16(bsums_lo))),
                    vaddq_s32(vmull_s16(vget_low_s16(scales_hi), vget_low_s16(bsums_hi)),
                              vmull_s16(vget_high_s16(scales_hi), vget_high_s16(bsums_hi))));
                const int32_t bias_corr = vaddvq_s32(bias_prod);

                c->out[(size_t)t * rows + row] +=
                    d * dx * (float)(vaddvq_s32(acc) - 32 * bias_corr);
            }
        }
    }
}
