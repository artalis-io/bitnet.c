#ifndef BN_TRANSFORMER_SSM_INTERNAL_H
#define BN_TRANSFORMER_SSM_INTERNAL_H

#include "transformer_simd_internal.h"
#include <math.h>
#include <stddef.h>

typedef struct {
    float *qkv;            // [qkv_dim] input/output
    float *conv_state;     // [(kern-1) * qkv_dim]
    const float *conv1d_w; // [qkv_dim * kern]
    int qkv_dim, kern;
} BnSSMConvCtx;

typedef struct {
    float *q, *k;          // [key_dim] each
    int head_dim;
} BnSSML2NormCtx;

typedef struct {
    float *state, *out;
    const float *q, *k;
    float *v;              // also temp for sk
    const float *alpha, *beta;
    int num_k_heads, head_k_dim, head_v_dim;
    float q_scale;
} BnSSMDeltaCtx;

typedef struct {
    float *out;
    const float *z, *norm_w;
    float eps;
    int head_v_dim;
} BnSSMGateCtx;

void bn_transformer_ssm_conv_silu_neon_range(void *ctx, int start, int end);
void bn_transformer_ssm_conv_silu_avx2_range(void *ctx, int start, int end);
void bn_transformer_ssm_conv_silu_wasm_range(void *ctx, int start, int end);
void bn_transformer_ssm_conv_silu_scalar_range(void *ctx, int start, int end);

void bn_transformer_ssm_l2norm_neon_range(void *ctx, int start, int end);
void bn_transformer_ssm_l2norm_avx2_range(void *ctx, int start, int end);
void bn_transformer_ssm_l2norm_wasm_range(void *ctx, int start, int end);
void bn_transformer_ssm_l2norm_scalar_range(void *ctx, int start, int end);

void bn_transformer_ssm_delta_neon_range(void *ctx, int start, int end);
void bn_transformer_ssm_delta_avx2_range(void *ctx, int start, int end);
void bn_transformer_ssm_delta_wasm_range(void *ctx, int start, int end);
void bn_transformer_ssm_delta_scalar_range(void *ctx, int start, int end);

void bn_transformer_ssm_gate_neon_range(void *ctx, int start, int end);
void bn_transformer_ssm_gate_avx2_range(void *ctx, int start, int end);
void bn_transformer_ssm_gate_wasm_range(void *ctx, int start, int end);
void bn_transformer_ssm_gate_scalar_range(void *ctx, int start, int end);

#endif // BN_TRANSFORMER_SSM_INTERNAL_H
