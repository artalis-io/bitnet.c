#ifndef BN_TRANSFORMER_INTERNAL_H
#define BN_TRANSFORMER_INTERNAL_H

// Internal header for transformer backend files (GQA attention + logits).
// Not part of the public API.

#include "transformer.h"
#include "gpu_backend.h"
#include "simd_helpers.h"
#include "quant.h"
#include "turboquant.h"
#include <math.h>
#include <string.h>

// --- GPU planning helpers ---

static inline int bn_transformer_gpu_has_cap(const BnGPUBackend *gpu, uint32_t cap) {
    return gpu && ((gpu->caps & cap) != 0);
}

static inline int bn_transformer_gpu_can_matvec_split(const BnGPUBackend *gpu, int tensor_type) {
    switch (tensor_type) {
        case BN_GGUF_TENSOR_Q4_0:
            return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_Q4_MATVEC_SPLIT);
        case BN_GGUF_TENSOR_Q8_0:
            return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_Q8_MATVEC_SPLIT);
        case BN_GGUF_TENSOR_Q5_K:
            return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_Q5K_MATVEC_SPLIT);
        default:
            return 0;
    }
}

static inline int bn_transformer_gpu_can_fused_gateup_silu(const BnGPUBackend *gpu,
                                                            int tensor_type,
                                                            int act_type) {
    return tensor_type == BN_GGUF_TENSOR_Q4_0 &&
           act_type != 1 &&
           bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_Q4_FUSED_GATEUP_SILU);
}

static inline int bn_transformer_gpu_can_flash_attn(const BnGPUBackend *gpu) {
    return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_FLASH_ATTN);
}

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
    BN_FUSION_Q8_QKV_SPLIT = 1u << 1,
    BN_FUSION_Q5_QKV_SPLIT = 1u << 2,
    BN_FUSION_FLASH_ATTN = 1u << 3,
    BN_FUSION_ROPE_QK = 1u << 4,
    BN_FUSION_GATEUP_SILU = 1u << 5,
    BN_FUSION_GATEUP_SPLIT = 1u << 6,
    BN_FUSION_RESIDUAL_RMSNORM = 1u << 7,
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
    int use_q8_qkv_split;
    int use_q5_qkv_split;
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

static inline int bn_transformer_is_attn_layer(const BnConfig *c, int layer) {
    return c->full_attn_interval == 0 ||
           ((layer + 1) % c->full_attn_interval == 0);
}

static inline int bn_transformer_attn_index(const BnConfig *c, int layer) {
    return c->full_attn_interval > 0
        ? (layer + 1) / c->full_attn_interval - 1
        : layer;
}

static inline int bn_transformer_ssm_index(const BnConfig *c, int layer) {
    return c->full_attn_interval > 0
        ? layer - (layer + 1) / c->full_attn_interval
        : -1;
}

static inline BnKVMode bn_transformer_kv_mode(const BnConfig *c, int tq_enabled) {
    if (c->kv_tq_bits > 0 && tq_enabled) return BN_KV_TQ;
    if (c->kv_f16) return BN_KV_FP16;
    return BN_KV_FP32;
}

static inline void bn_transformer_plan_layer_shape(BnLayerShapePlan *p,
                                                    const BnConfig *c,
                                                    const BnLayerWeights *lw,
                                                    int layer,
                                                    int tq_enabled) {
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->is_attn = bn_transformer_is_attn_layer(c, layer);
    p->attn_idx = p->is_attn ? bn_transformer_attn_index(c, layer) : -1;
    p->ssm_idx = p->is_attn ? -1 : bn_transformer_ssm_index(c, layer);
    p->head_size = lw->head_size > 0 ? lw->head_size : c->head_size;
    p->kv_dim = lw->kv_dim > 0 ? lw->kv_dim : c->kv_dim;
    p->n_kv_heads = lw->n_kv_heads > 0 ? lw->n_kv_heads : c->n_kv_heads;
    p->kv_mul = lw->kv_mul > 0 ? lw->kv_mul : c->kv_mul;
    p->q_dim = c->n_heads * p->head_size;
    p->q_gated = lw->wq.data && lw->wq.rows > p->q_dim;
    p->q_wide = !p->q_gated && lw->wq.data && lw->wq.rows > c->dim;
    p->qk_stride = c->qk_norm_per_head ? p->head_size : 0;
    p->has_qk_norm = (lw->q_norm || lw->k_norm) ? 1 : 0;
    p->has_bias = (lw->q_bias || lw->k_bias || lw->v_bias) ? 1 : 0;
    p->kv_mode = bn_transformer_kv_mode(c, tq_enabled);
    p->kind = p->is_attn
        ? (p->q_gated ? BN_LAYER_ATTN_GATED_Q
                      : (p->q_wide ? BN_LAYER_ATTN_WIDE_Q : BN_LAYER_ATTN_CLASSIC))
        : BN_LAYER_SSM;
}

static inline BnExecPlacement bn_transformer_preferred_placement(const BnGPUBackend *gpu,
                                                                  int prefer_gpu) {
    return prefer_gpu && gpu ? BN_EXEC_GPU : BN_EXEC_CPU;
}

static inline BnBackendPlacement bn_transformer_backend_placement(const BnGPUBackend *gpu,
                                                                   BnExecPlacement placement) {
    if (placement == BN_EXEC_CPU) return BN_BACKEND_CPU;
    if (placement == BN_EXEC_CPU_FALLBACK) return BN_BACKEND_CPU;
    if (!gpu) return BN_BACKEND_GPU_UNKNOWN;
    switch (gpu->kind) {
        case BN_GPU_BACKEND_METAL: return BN_BACKEND_METAL;
        case BN_GPU_BACKEND_WEBGPU: return BN_BACKEND_WEBGPU;
        case BN_GPU_BACKEND_CUDA: return BN_BACKEND_CUDA;
        default: return BN_BACKEND_GPU_UNKNOWN;
    }
}

static inline void bn_transformer_plan_attention(BnAttentionPlan *p,
                                                  const BnConfig *c,
                                                  const BnLayerWeights *lw,
                                                  const BnGPUBackend *gpu,
                                                  int layer,
                                                  int tq_enabled,
                                                  int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    bn_transformer_plan_layer_shape(&p->shape, c, lw, layer, tq_enabled);
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    if (!p->shape.is_attn) {
        p->needs_cpu_fallback = p->placement == BN_EXEC_GPU;
        if (p->needs_cpu_fallback) {
            p->placement = BN_EXEC_CPU_FALLBACK;
            p->backend = bn_transformer_backend_placement(gpu, p->placement);
        }
        return;
    }
    p->use_flash = c->flash_attn && bn_transformer_gpu_can_flash_attn(gpu);
    p->use_packed_qkv = lw->qkv_stacked_gpu && !p->shape.q_gated &&
                        lw->wq.type == BN_GGUF_TENSOR_Q4_0 &&
                        lw->wk.type == BN_GGUF_TENSOR_Q4_0 &&
                        lw->wv.type == BN_GGUF_TENSOR_Q4_0 &&
                        lw->q_bias_gpu && lw->k_bias_gpu && lw->v_bias_gpu;
    p->use_qkv_split = lw->qkv_stacked_gpu && !p->shape.q_gated &&
                       bn_transformer_gpu_can_matvec_split(gpu, lw->wq.type);
    p->use_q8_qkv_split = p->use_qkv_split && lw->wq.type == BN_GGUF_TENSOR_Q8_0;
    p->use_q5_qkv_split = p->use_qkv_split && lw->wq.type == BN_GGUF_TENSOR_Q5_K;
    if (p->use_qkv_split) p->fusion_flags |= BN_FUSION_QKV_SPLIT;
    if (p->use_q8_qkv_split) p->fusion_flags |= BN_FUSION_Q8_QKV_SPLIT;
    if (p->use_q5_qkv_split) p->fusion_flags |= BN_FUSION_Q5_QKV_SPLIT;
    if (p->use_flash) p->fusion_flags |= BN_FUSION_FLASH_ATTN;
    if (p->placement == BN_EXEC_GPU && !lw->k_bias_gpu)
        p->fusion_flags |= BN_FUSION_ROPE_QK;
}

static inline void bn_transformer_plan_ffn(BnFFNPlan *p,
                                            const BnConfig *c,
                                            const BnLayerWeights *lw,
                                            const BnGPUBackend *gpu,
                                            int layer,
                                            int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    p->kind = lw->router_weight ? BN_FFN_MOE
            : (c->has_ffn_gate ? BN_FFN_DENSE_GATE_UP : BN_FFN_DENSE_UP);
    p->hidden_dim = lw->ffn_up.rows > 0 ? lw->ffn_up.rows : c->hidden_dim;
    p->activation = c->act_type;
    p->has_gate = c->has_ffn_gate;
    p->has_sub_norm = lw->ffn_sub_norm ? 1 : 0;
    p->use_fused_gateup_silu =
        p->placement == BN_EXEC_GPU &&
        c->has_ffn_gate &&
        lw->ffn_gate.type == BN_GGUF_TENSOR_Q4_0 &&
        lw->ffn_up.type == BN_GGUF_TENSOR_Q4_0 &&
        bn_transformer_gpu_can_fused_gateup_silu(gpu, lw->ffn_gate.type, c->act_type);
    p->use_gateup_split =
        p->placement == BN_EXEC_GPU &&
        c->has_ffn_gate &&
        lw->gateup_stacked_gpu &&
        lw->ffn_gate.rows == lw->ffn_up.rows &&
        lw->ffn_gate.cols == lw->ffn_up.cols &&
        ((lw->ffn_gate.type == BN_GGUF_TENSOR_Q4_K && c->act_type != 1) ||
         bn_transformer_gpu_can_matvec_split(gpu, lw->ffn_gate.type));
    if (p->use_fused_gateup_silu) p->fusion_flags |= BN_FUSION_GATEUP_SILU;
    if (p->use_gateup_split) p->fusion_flags |= BN_FUSION_GATEUP_SPLIT;
    if (p->placement == BN_EXEC_GPU) p->fusion_flags |= BN_FUSION_RESIDUAL_RMSNORM;
    if (p->kind == BN_FFN_MOE && p->placement == BN_EXEC_GPU) {
        p->needs_cpu_fallback = 1;
        p->placement = BN_EXEC_CPU_FALLBACK;
        p->backend = bn_transformer_backend_placement(gpu, p->placement);
        p->fusion_flags = BN_FUSION_NONE;
    }
}

static inline void bn_transformer_plan_ssm(BnSSMPlan *p,
                                            const BnConfig *c,
                                            const BnLayerWeights *lw,
                                            int layer,
                                            int prefer_gpu,
                                            const BnGPUBackend *gpu) {
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->ssm_idx = bn_transformer_ssm_index(c, layer);
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    p->state_size = c->ssm_state_size;
    p->conv_kernel = c->ssm_conv_kernel;
    p->inner_size = c->ssm_inner_size;
    p->time_step_rank = c->ssm_time_step_rank;
    p->group_count = c->ssm_group_count;
    p->use_qkvz_stack = p->placement == BN_EXEC_GPU && lw->ssm_qkvz_stacked_gpu;
    p->use_alpha_beta_stack = p->placement == BN_EXEC_GPU && lw->ssm_ab_stacked_gpu;
}

static inline void bn_transformer_plan_moe(BnMoEPlan *p,
                                            const BnConfig *c,
                                            const BnLayerWeights *lw,
                                            const BnGPUBackend *gpu,
                                            int layer,
                                            int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    p->n_experts = c->n_experts;
    p->n_active = c->n_experts_active;
    p->hidden_dim = c->moe_intermediate_size;
    p->has_shared_expert = c->has_shared_expert || lw->shared_expert_gate;
    p->shared_hidden_dim = c->shared_expert_intermediate_size;
    if (p->placement == BN_EXEC_GPU && lw->router_weight) {
        p->needs_cpu_fallback = 1;
        p->placement = BN_EXEC_CPU_FALLBACK;
        p->backend = bn_transformer_backend_placement(gpu, p->placement);
    }
}

static inline void bn_transformer_plan_logits(BnLogitsPlan *p,
                                               const BnConfig *c,
                                               const BnWeights *w,
                                               const BnGPUBackend *gpu,
                                               int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    p->vocab_size = c->vocab_size;
    p->dim = c->dim;
    p->use_i8_output = w->emb_out_i8 != NULL;
    if (w->output_weight.data) {
        p->kind = BN_LOGITS_UNTIED;
        p->weight_type = w->output_weight.type;
    } else if (w->emb_out_i8) {
        p->kind = BN_LOGITS_TIED_I8;
        p->weight_type = BN_GGUF_TENSOR_Q8_0;
    } else if (w->emb_type == BN_GGUF_TENSOR_F16) {
        p->kind = BN_LOGITS_TIED_F16;
        p->weight_type = BN_GGUF_TENSOR_F16;
    } else {
        p->kind = BN_LOGITS_TIED_F32;
        p->weight_type = BN_GGUF_TENSOR_F32;
    }
}

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
