#include "quant_ctx.h"
#include "quant_neon_helpers.h"

/* Reuse each weight block across four tokens while retaining the float
 * matvec's per-block reduction and accumulation order. */
void bn_quant_q8_neon_float_matmul_range(void *ctx, int start, int end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks = c->cols / 32;
    for (int row = start; row < end; row++) {
        for (int first = 0; first < c->n_tokens; first += 4) {
            int count = c->n_tokens - first;
            if (count > 4) count = 4;
            float sums[4] = {0};
            for (int b = 0; b < n_blocks; b++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_blocks + b];
                float d = bn_fp16_to_fp32(blk->d);
                float32x4_t a0[4], a1[4], a2[4], a3[4];
                for (int t = 0; t < count; t++) {
                    a0[t] = a1[t] = a2[t] = a3[t] = vdupq_n_f32(0);
                }
                for (int half = 0; half < 2; half++) {
                    int8x16_t w = vld1q_s8(blk->qs + half * 16);
                    for (int t = 0; t < count; t++) {
                        const float *x = c->x + (size_t)(first + t) * c->cols
                                         + b * 32 + half * 16;
                        bn_neon_acc_i8x16_f32(w, x, &a0[t], &a1[t],
                                               &a2[t], &a3[t]);
                    }
                }
                for (int t = 0; t < count; t++)
                    sums[t] += bn_neon_reduce4(a0[t], a1[t], a2[t], a3[t]) * d;
            }
            for (int t = 0; t < count; t++)
                c->out[(size_t)(first + t) * c->W->rows + row] = sums[t];
        }
    }
}

void bn_quant_q8_neon_range(void *ctx, int row_start, int row_end) {
    BnQ8Ctx *c = (BnQ8Ctx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
            for (int i = 0; i < 2; i++) {
                int8x16_t w = vld1q_s8(blk->qs + i * 16);
                bn_neon_acc_i8x16_f32(w, xb + i * 16, &acc0, &acc1, &acc2, &acc3);
            }
            row_sum += bn_neon_reduce4(acc0, acc1, acc2, acc3) * d;
        }
        c->out[row] = row_sum;
    }
}
