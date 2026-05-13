#ifndef BN_TRANSFORMER_PLAN_INTERNAL_H
#define BN_TRANSFORMER_PLAN_INTERNAL_H

#include "model_config.h"
#include "model_weights.h"
#include <stdint.h>

typedef struct BnBackendModel BnBackendModel;
typedef struct BnGPUBackend BnGPUBackend;

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

int bn_transformer_gpu_has_cap(const BnGPUBackend *gpu, uint32_t cap);
int bn_transformer_gpu_can_matvec_split(const BnGPUBackend *gpu, int tensor_type);
int bn_transformer_gpu_can_fused_gateup_silu(const BnGPUBackend *gpu,
                                             int tensor_type,
                                             int act_type);
int bn_transformer_gpu_can_flash_attn(const BnGPUBackend *gpu);

int bn_transformer_is_attn_layer(const BnConfig *c, int layer);
int bn_transformer_attn_index(const BnConfig *c, int layer);
int bn_transformer_ssm_index(const BnConfig *c, int layer);
BnKVMode bn_transformer_kv_mode(const BnConfig *c, int tq_enabled);
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

#endif // BN_TRANSFORMER_PLAN_INTERNAL_H
