#ifndef BN_QUANT_KERNELS_WASM_H
#define BN_QUANT_KERNELS_WASM_H

#include "quant_ctx.h"

void bn_quant_i2s_wasm_range(void *ctx, int start, int end);
void bn_quant_i2s_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_tq2_wasm_range(void *ctx, int start, int end);
void bn_quant_tq2_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_tq1_wasm_range(void *ctx, int start, int end);
void bn_quant_tq1_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_q8_wasm_range(void *ctx, int start, int end);
void bn_quant_q8_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_q8_wasm_sdot_4row_range(void *ctx, int start, int end);
void bn_quant_q4_wasm_range(void *ctx, int start, int end);
void bn_quant_q4_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_wasm_sdot_4row_range(void *ctx, int start, int end);
void bn_quant_q4_repacked_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_repacked_wasm_sdot_8row_range(void *ctx, int start, int end);
void bn_quant_q4_wasm_gate_up_silu_4row_range(void *ctx, int start, int end);
void bn_quant_q4_repacked_gate_up_silu_wasm_range(void *ctx, int start, int end);
void bn_quant_q6k_wasm_range(void *ctx, int start, int end);
void bn_quant_q6k_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_q8k_wasm_range(void *ctx, int start, int end);
void bn_quant_q4k_wasm_range(void *ctx, int start, int end);
void bn_quant_q4k_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_q5k_wasm_range(void *ctx, int start, int end);
void bn_quant_q4_1_wasm_range(void *ctx, int start, int end);
void bn_quant_f32_wasm_range(void *ctx, int start, int end);
void bn_quant_f16_wasm_range(void *ctx, int start, int end);
void bn_quant_bf16_wasm_range(void *ctx, int start, int end);
void bn_quant_iq4nl_wasm_range(void *ctx, int start, int end);
void bn_quant_iq4xs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq3xxs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq3s_wasm_range(void *ctx, int start, int end);
void bn_quant_iq2xxs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq2xs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq2s_wasm_range(void *ctx, int start, int end);
void bn_quant_q2k_wasm_range(void *ctx, int start, int end);
void bn_quant_q3k_wasm_range(void *ctx, int start, int end);

#endif // BN_QUANT_KERNELS_WASM_H
