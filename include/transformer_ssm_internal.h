#ifndef BN_TRANSFORMER_SSM_INTERNAL_H
#define BN_TRANSFORMER_SSM_INTERNAL_H

#include "model_config.h"
#include <math.h>
#include <stddef.h>

typedef struct {
    float *qkv;            // [qkv_dim] input/output
    float *conv_state;     // [(kern-1) * qkv_dim]
    const float *conv1d_w; // [qkv_dim * kern]
    int qkv_dim, kern;
    int sigmoid_gate;
} BnSSMConvCtx;

typedef struct {
    float *q, *k;          // [key_dim] each
    float eps;
    int head_dim;
    BnSSMQKNormMode mode;
} BnSSML2NormCtx;

// RMS(eps/head_dim), then separate 1/sqrt(head_dim) scaling.
void bn_transformer_ssm_qk_rmsnorm_range(void *ctx, int start, int end);

/* Direct F32 x/sqrt(sum(x*x)+eps), without intermediate BF16 rounding.
 * Uses a zero-padded binary reduction; padded head size must fit
 * BN_MAX_VLA_ELEMS (default 8192). eps must be positive/finite, Q/K disjoint,
 * and head ranges valid for caller-owned buffers. Invalid metadata is a no-op. */
void bn_transformer_ssm_qk_sum_eps_range(void *ctx, int start, int end);

typedef struct {
    float *state, *out;       // state layout: [v_head][head_v_dim][head_k_dim]
    const float *q, *k;
    float *v;              // also temp for sk
    const float *alpha, *beta;
    int num_k_heads, head_k_dim, head_v_dim;
    float q_scale;
} BnSSMDeltaCtx;

/* KDA recurrence with coordinate-wise log decay. Inputs are already projected,
 * Q/K normalized and beta activated; gate construction and output rounding are
 * separate operations. State is F32 [head][value][key], q/k/log_decay are
 * [head][key], v/out are [head][value], and beta is [head]. Buffers must not
 * overlap. Each call updates only heads [start,end); inputs are read-only.
 * Supports key/value dimensions 1..256. Invalid metadata returns -1 before
 * mutation. A valid empty range permits NULL buffers. No allocation or model
 * state ownership is introduced here. */
typedef struct {
    float *state, *out;
    const float *q, *k, *v, *log_decay, *beta;
    int num_heads, head_k_dim, head_v_dim;
    float q_scale;
} BnSSMKDACtx;

int bn_transformer_ssm_kda_step(const BnSSMKDACtx *ctx, int start, int end);

typedef enum {
    BN_SSM_KDA_A_LOG = 0,
    BN_SSM_KDA_A_NEGATIVE_EXP = 1
} BnSSMKDAParameterStorage;

/* Bounded KDA coefficients, before the recurrence. decay_parameter is either
 * raw per-head A_log or stored -exp(A_log), selected explicitly. The latter
 * avoids log/exp reconstruction of GGUF parameters. gate/bias/log_decay are
 * [head][key]; decay_parameter/beta_raw/beta are [head].
 * amplitude = exp(parameter) or -parameter, according to parameter_storage.
 * log_decay = lower_bound / (1 + exp(-amplitude * (gate + bias)))
 * beta = sigmoid(beta_raw). Inputs and outputs are F32; lower_bound is finite
 * and negative, key dimensions 1..256. Exact log_decay==gate and beta==beta_raw
 * aliases are allowed; other overlap is forbidden. Validate the complete head
 * range before writing: nonfinite inputs, overflowing gate+bias or overflowing
 * exp(A_log) fail without mutation. Stored negative exponentials must be
 * finite and nonpositive. exp(A_log) underflow and saturating sigmoid
 * intermediates are allowed. A valid empty range permits NULL buffers. */
typedef struct {
    float *log_decay, *beta;
    const float *gate, *bias, *decay_parameter, *beta_raw;
    int num_heads, head_k_dim;
    float lower_bound;
    BnSSMKDAParameterStorage parameter_storage;
} BnSSMKDACoeffCtx;

int bn_transformer_ssm_kda_coefficients(const BnSSMKDACoeffCtx *ctx,
    int start, int end);

typedef struct {
    float *out;
    const float *z, *norm_w;
    float eps;
    int head_v_dim;
    int sigmoid_gate;
    int num_v_heads;
} BnSSMGateCtx;

void bn_transformer_ssm_conv_silu_neon_range(void *ctx, int start, int end);
void bn_transformer_ssm_conv_silu_avx2_range(void *ctx, int start, int end);
void bn_transformer_ssm_conv_silu_avx512_range(void *ctx, int start, int end);
void bn_transformer_ssm_conv_silu_x86_range(void *ctx, int start, int end);
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
void bn_transformer_ssm_gate_avx512_range(void *ctx, int start, int end);
void bn_transformer_ssm_gate_x86_range(void *ctx, int start, int end);
void bn_transformer_ssm_gate_wasm_range(void *ctx, int start, int end);
void bn_transformer_ssm_gate_scalar_range(void *ctx, int start, int end);

#endif // BN_TRANSFORMER_SSM_INTERNAL_H
