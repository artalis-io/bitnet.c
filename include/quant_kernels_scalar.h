#ifndef BN_QUANT_KERNELS_SCALAR_H
#define BN_QUANT_KERNELS_SCALAR_H

#include "quant_ctx.h"

void bn_quant_i2s_scalar_range(void *ctx, int start, int end);
void bn_quant_tq2_scalar_range(void *ctx, int start, int end);
void bn_quant_tq1_scalar_range(void *ctx, int start, int end);
void bn_quant_q8_scalar_sdot_range(void *ctx, int start, int end);
void bn_quant_q8_scalar_range(void *ctx, int start, int end);
void bn_quant_q4_scalar_range(void *ctx, int start, int end);
void bn_quant_q6k_scalar_sdot_range(void *ctx, int start, int end);
void bn_quant_q6k_scalar_range(void *ctx, int start, int end);
void bn_quant_q6k_scalar_matmul_range(void *ctx, int start, int end);
void bn_quant_q8k_scalar_range(void *ctx, int start, int end);
void bn_quant_q4k_scalar_sdot_range(void *ctx, int start, int end);
void bn_quant_q4k_scalar_range(void *ctx, int start, int end);
void bn_quant_q4k_scalar_matmul_range(void *ctx, int start, int end);
void bn_quant_q5k_scalar_range(void *ctx, int start, int end);
void bn_quant_q5k_scalar_matmul_range(void *ctx, int start, int end);
void bn_quant_q4_1_scalar_range(void *ctx, int start, int end);
void bn_quant_q5_1_scalar_range(void *ctx, int start, int end);
void bn_quant_f32_scalar_range(void *ctx, int start, int end);
void bn_quant_f16_scalar_range(void *ctx, int start, int end);
void bn_quant_bf16_scalar_range(void *ctx, int start, int end);
void bn_quant_iq4nl_scalar_range(void *ctx, int start, int end);
void bn_quant_iq4xs_scalar_range(void *ctx, int start, int end);
void bn_quant_iq3xxs_scalar_range(void *ctx, int start, int end);
void bn_quant_iq3s_scalar_range(void *ctx, int start, int end);
void bn_quant_iq2xxs_scalar_range(void *ctx, int start, int end);
void bn_quant_iq2xs_scalar_range(void *ctx, int start, int end);
void bn_quant_iq2s_scalar_range(void *ctx, int start, int end);
void bn_quant_q2k_scalar_range(void *ctx, int start, int end);
void bn_quant_q3k_scalar_range(void *ctx, int start, int end);

#endif // BN_QUANT_KERNELS_SCALAR_H
