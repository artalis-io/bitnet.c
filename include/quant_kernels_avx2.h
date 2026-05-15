#ifndef BN_QUANT_KERNELS_AVX2_H
#define BN_QUANT_KERNELS_AVX2_H

#include "quant_ctx.h"

void bn_quant_i2s_avx2_range(void *ctx, int start, int end);
void bn_quant_i2s_avx2_4row_range(void *ctx, int start, int end);
void bn_quant_tq2_avx2_range(void *ctx, int start, int end);
void bn_quant_tq1_avx2_range(void *ctx, int start, int end);
void bn_quant_q8_avx2_range(void *ctx, int start, int end);
void bn_quant_q8_avx2_4row_range(void *ctx, int start, int end);
void bn_quant_q4_avx2_range(void *ctx, int start, int end);
void bn_quant_q4_avx2_4row_range(void *ctx, int start, int end);
void bn_quant_q4_avx2_matmul_range(void *ctx, int start, int end);
void bn_quant_q6k_avx2_range(void *ctx, int start, int end);
void bn_quant_q6k_avx2_sdot_range(void *ctx, int start, int end);
void bn_quant_q6k_avx2_4row_range(void *ctx, int start, int end);
void bn_quant_q6k_avx2_sdot_matmul_range(void *ctx, int start, int end);
void bn_quant_q8k_avx2_range(void *ctx, int start, int end);
void bn_quant_q4k_avx2_range(void *ctx, int start, int end);
void bn_quant_q4k_avx2_sdot_range(void *ctx, int start, int end);
void bn_quant_q4k_avx2_4row_range(void *ctx, int start, int end);
void bn_quant_q4k_avx2_sdot_matmul_range(void *ctx, int start, int end);
void bn_quant_q5k_avx2_range(void *ctx, int start, int end);
void bn_quant_q5k_avx2_4row_range(void *ctx, int start, int end);
void bn_quant_q5k_avx2_matmul_range(void *ctx, int start, int end);
void bn_quant_q4_1_avx2_range(void *ctx, int start, int end);
void bn_quant_f32_avx2_range(void *ctx, int start, int end);
void bn_quant_f16_avx2_range(void *ctx, int start, int end);
void bn_quant_bf16_avx2_range(void *ctx, int start, int end);
void bn_quant_bf16_avx2_4row_range(void *ctx, int start, int end);
void bn_quant_iq4nl_avx2_range(void *ctx, int start, int end);
void bn_quant_iq4xs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq3xxs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq3s_avx2_range(void *ctx, int start, int end);
void bn_quant_iq2xxs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq2xs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq2s_avx2_range(void *ctx, int start, int end);
void bn_quant_q2k_avx2_range(void *ctx, int start, int end);
void bn_quant_q3k_avx2_range(void *ctx, int start, int end);

#endif // BN_QUANT_KERNELS_AVX2_H
