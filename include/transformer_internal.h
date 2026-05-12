#ifndef BN_TRANSFORMER_INTERNAL_H
#define BN_TRANSFORMER_INTERNAL_H

// Internal header for transformer backend files (GQA attention + logits).
// Not part of the public API.

#include "transformer.h"
#include "gpu_backend.h"
#include "backend_model.h"
#include "simd_helpers.h"
#include "quant.h"
#include "turboquant.h"
#include <math.h>
#include <string.h>

// --- GPU planning helpers ---

int bn_transformer_gpu_has_cap(const BnGPUBackend *gpu, uint32_t cap);
int bn_transformer_gpu_can_matvec_split(const BnGPUBackend *gpu, int tensor_type);
int bn_transformer_gpu_can_fused_gateup_silu(const BnGPUBackend *gpu,
                                             int tensor_type,
                                             int act_type);
int bn_transformer_gpu_can_flash_attn(const BnGPUBackend *gpu);
void *bn_transformer_backend_handle_or(const BnBackendModel *backend,
                                       int layer,
                                       BnBackendHandleRole role);

// --- Layer-shape planning helpers ---

typedef enum {
    BN_LAYER_ATTN_CLASSIC = 0,
    BN_LAYER_ATTN_GATED_Q = 1,
    BN_LAYER_ATTN_WIDE_Q = 2,
    BN_LAYER_SSM = 3,
} BnLayerKind;

typedef enum {
    BN_KV_FP32 = 0,
    BN_KV_FP16 = 1,
    BN_KV_TQ = 2,
} BnKVMode;

typedef enum {
    BN_EXEC_CPU = 0,
    BN_EXEC_GPU = 1,
    BN_EXEC_CPU_FALLBACK = 2,
} BnExecPlacement;

typedef enum {
    BN_BACKEND_CPU = 0,
    BN_BACKEND_METAL = 1,
    BN_BACKEND_WEBGPU = 2,
    BN_BACKEND_CUDA = 3,
    BN_BACKEND_GPU_UNKNOWN = 4,
} BnBackendPlacement;

typedef enum {
    BN_CPU_BACKEND_SCALAR = 0,
    BN_CPU_BACKEND_NEON = 1,
    BN_CPU_BACKEND_AVX2 = 2,
    BN_CPU_BACKEND_AVX512 = 3,
    BN_CPU_BACKEND_WASM_SIMD = 4,
} BnCPUBackendPlacement;

typedef enum {
    BN_FFN_DENSE_UP = 0,
    BN_FFN_DENSE_GATE_UP = 1,
    BN_FFN_MOE = 2,
} BnFFNKind;

typedef enum {
    BN_LOGITS_TIED_F32 = 0,
    BN_LOGITS_TIED_F16 = 1,
    BN_LOGITS_TIED_I8 = 2,
    BN_LOGITS_UNTIED = 3,
} BnLogitsKind;

typedef enum {
    BN_FUSION_NONE = 0,
    BN_FUSION_QKV_SPLIT = 1u << 0,
    BN_FUSION_FLASH_ATTN = 1u << 1,
    BN_FUSION_ROPE_QK = 1u << 2,
    BN_FUSION_GATEUP_SILU = 1u << 3,
    BN_FUSION_GATEUP_SPLIT = 1u << 4,
    BN_FUSION_RESIDUAL_RMSNORM = 1u << 5,
} BnFusionFlag;

typedef struct {
    int layer;
    int is_attn;
    int attn_idx;
    int ssm_idx;
    BnLayerKind kind;
    BnKVMode kv_mode;
    int q_dim;
    int q_gated;
    int q_wide;
    int head_size;
    int kv_dim;
    int n_kv_heads;
    int kv_mul;
    int qk_stride;
    int has_qk_norm;
    int has_bias;
} BnLayerShapePlan;

typedef struct {
    BnLayerShapePlan shape;
    BnExecPlacement placement;
    BnBackendPlacement backend;
    int use_flash;
    int use_packed_qkv;
    int use_qkv_split;
    int qkv_split_op_code;
    int needs_cpu_fallback;
    uint32_t fusion_flags;
} BnAttentionPlan;

typedef struct {
    int layer;
    BnFFNKind kind;
    BnExecPlacement placement;
    BnBackendPlacement backend;
    int hidden_dim;
    int activation;
    int has_gate;
    int has_sub_norm;
    int use_fused_gateup_silu;
    int use_gateup_split;
    int needs_cpu_fallback;
    uint32_t fusion_flags;
} BnFFNPlan;

typedef struct {
    int layer;
    int ssm_idx;
    BnExecPlacement placement;
    BnBackendPlacement backend;
    int state_size;
    int conv_kernel;
    int inner_size;
    int time_step_rank;
    int group_count;
    int use_qkvz_stack;
    int use_alpha_beta_stack;
    int needs_cpu_fallback;
} BnSSMPlan;

typedef struct {
    int layer;
    BnExecPlacement placement;
    BnBackendPlacement backend;
    int n_experts;
    int n_active;
    int hidden_dim;
    int has_shared_expert;
    int shared_hidden_dim;
    int needs_cpu_fallback;
} BnMoEPlan;

typedef struct {
    BnLogitsKind kind;
    BnExecPlacement placement;
    BnBackendPlacement backend;
    int vocab_size;
    int dim;
    int weight_type;
    int use_i8_output;
    int needs_cpu_fallback;
} BnLogitsPlan;

int bn_transformer_is_attn_layer(const BnConfig *c, int layer);
int bn_transformer_attn_index(const BnConfig *c, int layer);
int bn_transformer_ssm_index(const BnConfig *c, int layer);
BnKVMode bn_transformer_kv_mode(const BnConfig *c, int tq_enabled);
void bn_transformer_tq_write_kv(const BnTQState *tq,
                                BnRunState *s,
                                const float *k_tmp,
                                const float *v_tmp,
                                int n_kv_heads,
                                int head_size,
                                int attn_idx,
                                int cache_pos,
                                int seq_len);
void bn_transformer_tq_gqa_dispatch(BnModel *m,
                                    BnRunState *s,
                                    int attn_idx,
                                    int pos,
                                    int n_heads,
                                    int n_kv_heads,
                                    int head_size,
                                    int kv_mul);
void bn_transformer_write_kv_fp16(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  const float *k_tmp,
                                  const float *v_tmp,
                                  int kv_dim);
void bn_transformer_kv_cache_rows(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  float **key_cache_row,
                                  float **value_cache_row);
void bn_transformer_write_kv_fp32(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  const float *k_tmp,
                                  const float *v_tmp,
                                  int kv_dim);
float *bn_transformer_forward_logits(BnModel *m, BnSession *sess);
void bn_transformer_plan_layer_shape(BnLayerShapePlan *p,
                                     const BnConfig *c,
                                     const BnLayerWeights *lw,
                                     int layer,
                                     int tq_enabled);
BnExecPlacement bn_transformer_preferred_placement(const BnGPUBackend *gpu,
                                                   int prefer_gpu);
BnBackendPlacement bn_transformer_backend_placement(const BnGPUBackend *gpu,
                                                    BnExecPlacement placement);
BnCPUBackendPlacement bn_transformer_cpu_backend_placement(void);
void bn_transformer_cpu_residual_add(float *x, const float *r, int dim);
void bn_transformer_cpu_apply_ffn_activation(BnRunState *s,
                                             const BnFFNPlan *ffn_plan,
                                             int hidden_dim,
                                             int already_activated);
void bn_transformer_cpu_forward_ssm_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          int layer);
void bn_transformer_cpu_forward_ffn_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          const BnFFNPlan *ffn_plan);
int bn_transformer_cpu_forward_layer(BnModel *m,
                                     BnSession *sess,
                                     int layer,
                                     int pos,
                                     int cache_pos,
                                     int rope_dims,
                                     const float *rope_cos,
                                     const float *rope_sin);
float *bn_transformer_gpu_forward(BnModel *m,
                                  BnSession *sess,
                                  int token,
                                  int pos);

void bn_transformer_gpu_finalize_op_kinds(BnGPUOp *ops, int n);
void bn_transformer_gpu_emit_rmsnorm(BnGPUOp *ops, int *n,
                                     void *norm_gpu,
                                     int buf_in,
                                     int buf_out,
                                     int dim,
                                     uint32_t u_eps);
void bn_transformer_gpu_emit_logits(BnGPUOp *ops, int *n,
                                    void *logit_gpu_buf,
                                    int logit_type,
                                    int logit_rows,
                                    int logit_cols);
void bn_transformer_gpu_emit_dense_ffn(BnGPUOp *ops, int *n,
                                       const BnConfig *c,
                                       const BnLayerWeights *lw,
                                       const BnFFNPlan *ffn_plan,
                                       const BnGPUBackend *gpu,
                                       const BnBackendModel *backend,
                                       int layer,
                                       int dim,
                                       uint32_t u_eps,
                                       void *next_norm);
void bn_transformer_gpu_emit_attention(BnGPUOp *ops, int *n,
                                       const BnConfig *c,
                                       const BnLayerWeights *lw,
                                       const BnGPUBackend *gpu,
                                       const BnBackendModel *backend,
                                       int layer,
                                       int pos,
                                       int dim,
                                       int q_dim,
                                       int head_size,
                                       int n_heads,
                                       int kv_dim,
                                       int rope_dims,
                                       int n_kv,
                                       size_t loff,
                                       uint32_t kv_cache_off,
                                       int has_moe,
                                       uint32_t u_eps);
void bn_transformer_gpu_emit_qkv(BnGPUOp *ops, int *n,
                                 const BnConfig *c,
                                 const BnLayerWeights *lw,
                                 const BnLayerShapePlan *plan,
                                 const BnGPUBackend *gpu,
                                 const BnBackendModel *backend,
                                 int layer,
                                 int pos,
                                 int q_dim,
                                 int head_size,
                                 int n_heads,
                                 int kv_dim,
                                 int rope_dims,
                                 uint32_t kv_cache_off,
                                 uint32_t u_eps);
void bn_transformer_gpu_emit_ssm(BnGPUOp *ops, int *n,
                                 const BnConfig *c,
                                 const BnLayerWeights *lw,
                                 const BnLayerShapePlan *plan,
                                 const BnGPUBackend *gpu,
                                 const BnBackendModel *backend,
                                 int layer,
                                 int dim,
                                 uint32_t u_eps);
void bn_transformer_gpu_emit_moe(BnGPUOp *ops, int *n,
                                 BnModel *m,
                                 BnSession *sess,
                                 const BnLayerWeights *lw,
                                 int layer,
                                 int dim,
                                 uint32_t u_eps,
                                 void *next_norm,
                                 void **uncached_bufs,
                                 int *n_uncached);

void bn_transformer_plan_attention(BnAttentionPlan *p,
                                   const BnConfig *c,
                                   const BnLayerWeights *lw,
                                   const BnGPUBackend *gpu,
                                   const BnBackendModel *backend,
                                   int layer,
                                   int tq_enabled,
                                   int prefer_gpu);
void bn_transformer_plan_ffn(BnFFNPlan *p,
                             const BnConfig *c,
                             const BnLayerWeights *lw,
                             const BnGPUBackend *gpu,
                             const BnBackendModel *backend,
                             int layer,
                             int prefer_gpu);
void bn_transformer_plan_ssm(BnSSMPlan *p,
                             const BnConfig *c,
                             const BnLayerWeights *lw,
                             int layer,
                             int prefer_gpu,
                             const BnGPUBackend *gpu,
                             const BnBackendModel *backend);
void bn_transformer_plan_moe(BnMoEPlan *p,
                             const BnConfig *c,
                             const BnLayerWeights *lw,
                             const BnGPUBackend *gpu,
                             int layer,
                             int prefer_gpu);
void bn_transformer_plan_logits(BnLogitsPlan *p,
                                const BnConfig *c,
                                const BnWeights *w,
                                const BnGPUBackend *gpu,
                                int prefer_gpu);

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

void bn_transformer_cpu_gqa_dispatch(BnModel *m,
                                     BnGQACtx *gctx,
                                     int n_heads,
                                     int kv_mul);
void bn_transformer_cpu_apply_rope_heads(float *buf,
                                         int n_heads,
                                         int head_size,
                                         int rope_dims,
                                         const float *rc,
                                         const float *rs);

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
