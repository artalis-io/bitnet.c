#include "quant_ctx.h"
#include "quant_neon_helpers.h"
#include "iq_tables.h"

void bn_quant_iq4nl_neon_range(void *ctx, int row_start, int row_end) {
    BnIQ4NLCtx *c = (BnIQ4NLCtx *)ctx;
    const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4NL *blk = &blocks[row * n_blocks_per_row + b];
            __builtin_prefetch(blk + 2, 0, 0);
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;

            // Scalar decode through codebook into int8 buffer
            int8_t tmp[32];
            for (int i = 0; i < 16; i++) {
                uint8_t byte = blk->qs[i];
                tmp[i]      = bn_kvalues_iq4nl[byte & 0xF];
                tmp[i + 16] = bn_kvalues_iq4nl[byte >> 4];
            }

            // NEON accumulation
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
            bn_neon_acc_i8x16_f32(vld1q_s8(tmp), xb, &acc0, &acc1, &acc2, &acc3);
            bn_neon_acc_i8x16_f32(vld1q_s8(tmp + 16), xb + 16, &acc0, &acc1, &acc2, &acc3);
            row_sum += bn_neon_reduce4(acc0, acc1, acc2, acc3) * d;
        }
        c->out[row] = row_sum;
    }
}
