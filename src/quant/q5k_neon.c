#include "quant_internal.h"
#include "quant_neon_helpers.h"
#include <arm_neon.h>

void bn_quant_q5k_neon_range(void *ctx, int row_start, int row_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 1, 0, 0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const uint8_t *qh = blk->qh;
            const float *xb = x + b * BN_QK_K;

            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                float ds = d * sc;
                float dm = dmin * m;
                int8x16_t w0, w1;
                {
                    uint8_t tmp[16];
                    for (int l = 0; l < 16; l++) {
                        int bit_idx = j + l;
                        int hbit = (qh[bit_idx / 8] >> (bit_idx % 8)) & 1;
                        tmp[l] = (qs[l] & 0xF) | (hbit << 4);
                    }
                    w0 = vreinterpretq_s8_u8(vld1q_u8(tmp));
                }
                {
                    uint8_t tmp[16];
                    for (int l = 0; l < 16; l++) {
                        int bit_idx = j + 16 + l;
                        int hbit = (qh[bit_idx / 8] >> (bit_idx % 8)) & 1;
                        tmp[l] = (qs[l + 16] & 0xF) | (hbit << 4);
                    }
                    w1 = vreinterpretq_s8_u8(vld1q_u8(tmp));
                }
                BN_QK_ACC_SCALED_16(w0, xb + j, ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 16, ds, dm);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                ds = d * sc;
                dm = dmin * m;
                {
                    uint8_t tmp[16];
                    for (int l = 0; l < 16; l++) {
                        int bit_idx = j + 32 + l;
                        int hbit = (qh[bit_idx / 8] >> (bit_idx % 8)) & 1;
                        tmp[l] = (qs[l] >> 4) | (hbit << 4);
                    }
                    w0 = vreinterpretq_s8_u8(vld1q_u8(tmp));
                }
                {
                    uint8_t tmp[16];
                    for (int l = 0; l < 16; l++) {
                        int bit_idx = j + 48 + l;
                        int hbit = (qh[bit_idx / 8] >> (bit_idx % 8)) & 1;
                        tmp[l] = (qs[l + 16] >> 4) | (hbit << 4);
                    }
                    w1 = vreinterpretq_s8_u8(vld1q_u8(tmp));
                }
                BN_QK_ACC_SCALED_16(w0, xb + j + 32, ds, dm);
                BN_QK_ACC_SCALED_16(w1, xb + j + 48, ds, dm);

                qs += 32;
            }

            float32x4_t s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            row_sum += vget_lane_f32(vpadd_f32(r, r), 0);
        }
        c->out[row] = row_sum;
    }
}
