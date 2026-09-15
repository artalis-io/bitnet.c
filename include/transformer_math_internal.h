#ifndef BN_TRANSFORMER_MATH_INTERNAL_H
#define BN_TRANSFORMER_MATH_INTERNAL_H

#include <math.h>

// Max elements for stack VLAs in backend range functions.
// Prevents stack overflow from malicious model configs.
#ifndef BN_MAX_VLA_ELEMS
#define BN_MAX_VLA_ELEMS 8192
#endif

void bn_transformer_softmax(float *x, int size);
void bn_transformer_softmax_scalar(float *x, int size);
void bn_transformer_softmax_avx2(float *x, int size);
void bn_transformer_softmax_avx2_batch(float *x, int size);
void bn_transformer_softmax_avx512(float *x, int size);
void bn_transformer_softmax_neon(float *x, int size);
float bn_transformer_fast_exp_scalar(float x);

// Backend-ordered elementwise product reduction. x86 sums rounded F32
// products in double precision; other backends retain their F32 dot order.
// Inputs may be unaligned. A zero size returns zero without reading inputs.
float bn_transformer_sum_products(const float *a, const float *b, int size);

// In-place F32 scaling followed by SiLU. x86 uses native vector SiLU with
// scalar tails; other backends retain scalar SiLU. Input may be unaligned;
// a zero size does not read or write x.
void bn_transformer_scaled_silu(float *x, float scale, int size);

// Depthwise dilated convolution followed by SiLU. Weights are [size, kernel];
// history is [(kernel - 1) * dilation, size], oldest first. History is read-only.
// kernel/dilation must be positive; size may be zero. out may alias current.
// x86 uses oldest-first rounded products and native SiLU; other backends
// retain current-first accumulation and scalar SiLU.
void bn_transformer_dilated_conv_silu(float *out, const float *current,
    const float *history, const float *weights, int size, int kernel,
    int dilation);

// Add a scaled branch and an already-computed branch to a residual. x86
// groups x + (rounded(value * scale) + branch); other backends retain
// (x + value * scale) + branch. Arrays must not overlap; size may be zero.
void bn_transformer_scaled_branch_add(float *x, const float *value,
    float scale, const float *branch, int size);

#endif // BN_TRANSFORMER_MATH_INTERNAL_H
