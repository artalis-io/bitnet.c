#ifndef BN_QUANT_KERNELS_NEON_H
#define BN_QUANT_KERNELS_NEON_H

#include "quant_ctx.h"

void bn_quant_i2s_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_i2s_neon_range(void *ctx, int start, int end);
void bn_quant_tq2_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_tq2_neon_range(void *ctx, int start, int end);
void bn_quant_tq1_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_tq1_neon_range(void *ctx, int start, int end);
void bn_quant_q8_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q8_neon_range(void *ctx, int start, int end);
void bn_quant_q4_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_repacked_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_repacked_gate_up_silu_neon_range(void *ctx, int start, int end);
void bn_quant_q4_neon_range(void *ctx, int start, int end);
void bn_quant_q6k_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q6k_neon_sdot_matmul_range(void *ctx, int start, int end);
void bn_quant_q6k_neon_range(void *ctx, int start, int end);
void bn_quant_q8k_neon_range(void *ctx, int start, int end);
void bn_quant_q4k_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q4k_neon_sdot_matmul_range(void *ctx, int start, int end);
void bn_quant_q4k_neon_range(void *ctx, int start, int end);
void bn_quant_q5k_neon_range(void *ctx, int start, int end);
void bn_quant_q5k_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q5k_neon_sdot_matmul_range(void *ctx, int start, int end);
void bn_quant_q4_1_neon_range(void *ctx, int start, int end);
void bn_quant_f32_neon_range(void *ctx, int start, int end);
void bn_quant_f16_neon_range(void *ctx, int start, int end);
void bn_quant_bf16_neon_range(void *ctx, int start, int end);
void bn_quant_iq4nl_neon_range(void *ctx, int start, int end);
void bn_quant_iq4xs_neon_range(void *ctx, int start, int end);
void bn_quant_iq3xxs_neon_range(void *ctx, int start, int end);
void bn_quant_iq3s_neon_range(void *ctx, int start, int end);
void bn_quant_iq2xxs_neon_range(void *ctx, int start, int end);
void bn_quant_iq2xs_neon_range(void *ctx, int start, int end);
void bn_quant_iq2s_neon_range(void *ctx, int start, int end);
void bn_quant_q2k_neon_range(void *ctx, int start, int end);
void bn_quant_q3k_neon_range(void *ctx, int start, int end);

#endif // BN_QUANT_KERNELS_NEON_H
