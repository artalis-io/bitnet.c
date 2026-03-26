#ifndef BN_TRANSFORMER_INTERNAL_H
#define BN_TRANSFORMER_INTERNAL_H

// Internal header for transformer backend files (GQA attention + logits).
// Not part of the public API.

#include "transformer.h"
#include "turboquant.h"
#include "simd_helpers.h"
#include "quant.h"
#include "turboquant.h"
#include <math.h>
#include <string.h>

// --- Context structs ---

typedef struct {
    const BnConfig *c;
    BnRunState *s;
    size_t loff;
    int pos;
    int n_kv;       // min(pos+1, seq_len) -- number of valid KV entries
    int kv_mul;
    int head_size;
    int kv_dim;
    int seq_len;    // cache size for modular indexing
} BnGQACtx;

typedef struct {
    const BnConfig *c;
    BnRunState *s;
    const BnTQState *tq;
    const uint8_t *tq_keys;
    const uint8_t *tq_values;
    int key_stride;
    int val_stride;
    int key_bytes;
    int val_bytes;
    int pos;
    int n_kv;
    int kv_mul;
    int head_size;
    int seq_len;
    int n_kv_heads;
} BnGQATQCtx;

typedef struct {
    float *logits;
    const int8_t *emb_i8;
    const float *emb_scales;
    const int8_t *x_q;
    float x_scale;
    int dim;
} BnLogitsI8Ctx;

typedef struct {
    float *logits;
    const float *x;
    const void *emb;
    int dim;
} BnLogitsCtx;

// Max elements for stack VLAs in backend range functions.
// Prevents stack overflow from malicious model configs.
#ifndef BN_MAX_VLA_ELEMS
#define BN_MAX_VLA_ELEMS 8192
#endif

// --- Shared inline helpers ---

#ifdef __ARM_NEON
#include <arm_neon.h>
static inline float bn_transformer_neon_hsum_f32(float32x4_t v) {
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#endif

static inline void bn_transformer_softmax(float *x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// --- RMSNorm backend declarations ---

void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_wasm(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps);

// --- TurboQuant GQA context ---

typedef struct {
    const BnConfig *c;
    BnRunState *s;
    const BnTQState *tq;
    const uint8_t *tq_keys;    // layer's packed keys base
    const uint8_t *tq_values;  // layer's packed values base
    int key_stride;             // bytes per position (n_kv_heads * key_bytes)
    int val_stride;             // bytes per position (n_kv_heads * val_bytes)
    int key_bytes;              // bytes per single head's key
    int val_bytes;              // bytes per single head's value
    int pos;
    int n_kv;
    int kv_mul;
    int head_size;
    int seq_len;
    int n_kv_heads;
} BnGQATQCtx;

// --- GQA range function declarations ---

void bn_transformer_gqa_neon_range(void *ctx, int start, int end);
void bn_transformer_gqa_avx2_range(void *ctx, int start, int end);
void bn_transformer_gqa_wasm_range(void *ctx, int start, int end);
void bn_transformer_gqa_scalar_range(void *ctx, int start, int end);
void bn_transformer_gqa_tq_neon_range(void *ctx, int start, int end);
void bn_transformer_gqa_tq_scalar_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_neon_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_avx2_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_wasm_range(void *ctx, int start, int end);
void bn_transformer_flash_gqa_scalar_range(void *ctx, int start, int end);

// --- TurboQuant GQA range function declarations ---

void bn_transformer_gqa_tq_scalar_range(void *ctx, int start, int end);
void bn_transformer_gqa_tq_neon_range(void *ctx, int start, int end);

// --- Logits range function declarations ---

void bn_transformer_logits_i8_neon_range(void *ctx, int start, int end);
void bn_transformer_logits_i8_avx2_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_native_neon_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_neon_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_avx2_range(void *ctx, int start, int end);
void bn_transformer_logits_i8_wasm_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_wasm_range(void *ctx, int start, int end);
void bn_transformer_logits_f16_scalar_range(void *ctx, int start, int end);
void bn_transformer_logits_f32_range(void *ctx, int start, int end);

// --- SSM context structs ---

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

// --- SSM range function declarations ---

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

#endif // BN_TRANSFORMER_INTERNAL_H
