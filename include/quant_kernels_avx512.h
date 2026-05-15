#ifndef BN_QUANT_KERNELS_AVX512_H
#define BN_QUANT_KERNELS_AVX512_H

#include "quant_ctx.h"

void bn_quant_q8_avx512_vnni_4row_range(void *ctx, int start, int end);
void bn_quant_q4_avx512_vnni_4row_range(void *ctx, int start, int end);
void bn_quant_q4k_avx512_vnni_4row_range(void *ctx, int start, int end);
void bn_quant_q4k_avx512_vnni_matmul_4row_range(void *ctx, int start, int end);
void bn_quant_q5k_avx512_4row_range(void *ctx, int start, int end);
void bn_quant_q5k_avx512_vnni_4row_range(void *ctx, int start, int end);
void bn_quant_q5k_avx512_vnni_matmul_4row_range(void *ctx, int start, int end);
void bn_quant_q6k_avx512_vnni_4row_range(void *ctx, int start, int end);

#endif // BN_QUANT_KERNELS_AVX512_H
