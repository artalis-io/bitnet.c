#ifndef BN_TRANSFORMER_PLAN_INTERNAL_H
#define BN_TRANSFORMER_PLAN_INTERNAL_H

#include "backend_placement.h"
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
    BN_LOGITS_TIED_DENSE_FLOAT = 0,
    BN_LOGITS_TIED_F16 = 1,
    BN_LOGITS_TIED_I8 = 2,
    BN_LOGITS_TIED_QUANT = 3,
    BN_LOGITS_UNTIED_F16 = 4,
    BN_LOGITS_UNTIED_QUANT = 5,
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
    int scalar_exact_activation;
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
int bn_transformer_gpu_can_native_quant_qkv(int q_type, int k_type, int v_type);
int bn_transformer_gpu_can_stack_same_quant_format_qk(int q_type, int k_type);
int bn_transformer_gpu_can_stack_same_quant_format_qk_weights(const BnQWeight *q,
                                                              const BnQWeight *k,
                                                              int q_dim,
                                                              int kv_dim);
int bn_transformer_gpu_can_stack_same_quant_format_gateup(const BnQWeight *gate,
                                                          const BnQWeight *up);
int bn_transformer_gpu_can_matvec_split(const BnGPUBackend *gpu, int tensor_type);
int bn_transformer_gpu_can_fused_gateup_silu(const BnGPUBackend *gpu,
                                             int tensor_type,
                                             int act_type);
int bn_transformer_gpu_can_fused_gateup_silu_pair(const BnGPUBackend *gpu,
                                                  int gate_type,
                                                  int up_type,
                                                  int act_type);
int bn_transformer_gpu_can_gateup_split_activation(const BnGPUBackend *gpu,
                                                   int tensor_type,
                                                   int act_type);
uint32_t bn_transformer_gpu_matvec_kquant_dot_flags(int tensor_type,
                                                 int enabled);
uint32_t bn_transformer_gpu_matvec_reference_kquant_flags(int tensor_type,
                                                         int enabled);
uint32_t bn_transformer_gpu_moe_route_raw_compare_matvec_flags(int tensor_type);
int bn_transformer_gpu_float_buffer_type(void);
uint32_t bn_transformer_gpu_exact_silu_flags(int tensor_type,
                                             int use_silu);
uint32_t bn_transformer_gpu_exact_silu_active_flags(int exact_silu);
int bn_transformer_gpu_prefers_gateup_split(int tensor_type);
int bn_transformer_gpu_same_quant_format_pair_stackable(int left_type,
                                                int right_type);
int bn_transformer_gpu_shared_kquant_gateup_dot_eligible(int gate_type,
                                                         int up_type,
                                                         int cols);
int bn_transformer_gpu_fused_gateup_silu_policy_allows(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_transformer_gpu_small_dense_exact_native_fused_gateup_enabled(
    int use_small_dense_exact_native);
int bn_transformer_gpu_gateup_split_enabled(void);
int bn_transformer_gpu_small_dense_exact_native_down_enabled(
    int use_small_dense_exact_native_down);
int bn_transformer_gpu_qkv_split_enabled(int use_small_dense_exact_native);
int bn_transformer_gpu_qk_split_enabled(void);
int bn_transformer_gpu_ssm_qkvz_split_enabled(void);
int bn_transformer_gpu_ssm_ab_stack_enabled(void);
int bn_transformer_gpu_shared_kquant_dot_enabled(int eligible);
int bn_transformer_gpu_shared_expert_gate_enabled(int eligible);
int bn_transformer_gpu_can_flash_attn(const BnGPUBackend *gpu);
BnBackendPlacement bn_transformer_gpu_backend_placement(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_dense_ffn_fast_path_available(
    const BnGPUBackend *gpu,
    const BnFFNPlan *ffn_plan);
int bn_transformer_gpu_dense_ffn_fast_path_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *x,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type);

int bn_transformer_is_attn_layer(const BnConfig *c, int layer);
int bn_transformer_attn_index(const BnConfig *c, int layer);
int bn_transformer_ssm_index(const BnConfig *c, int layer);
int bn_transformer_attention_layer_count(const BnConfig *c);
int bn_transformer_ssm_layer_count(const BnConfig *c);
int bn_transformer_uses_hybrid_ssm(const BnConfig *c);
int bn_transformer_uses_hybrid_moe(const BnConfig *c);
int bn_transformer_weight_is_packed_qkv(const BnQWeight *qkv,
                                        int input_dim,
                                        int q_dim,
                                        int kv_dim);
int bn_transformer_attention_q_projection_is_gated(const BnQWeight *wq,
                                                   int q_dim);
int bn_transformer_attention_q_projection_is_wide(const BnQWeight *wq,
                                                  int model_dim,
                                                  int q_dim);
int bn_transformer_attention_head_size(const BnConfig *c,
                                       const BnLayerWeights *lw);
int bn_transformer_attention_kv_dim(const BnConfig *c,
                                    const BnLayerWeights *lw);
int bn_transformer_attention_n_kv_heads(const BnConfig *c,
                                        const BnLayerWeights *lw);
int bn_transformer_attention_kv_mul(const BnConfig *c,
                                    const BnLayerWeights *lw);
int bn_transformer_attention_qk_stride(const BnConfig *c,
                                       int head_size);
int bn_transformer_attention_has_qk_norm(const BnLayerWeights *lw);
int bn_transformer_attention_has_bias(const BnLayerWeights *lw);
BnLayerKind bn_transformer_layer_kind(int is_attn,
                                      int q_gated,
                                      int q_wide);
int bn_transformer_attention_requires_cpu_fallback(
    const BnLayerShapePlan *shape,
    BnExecPlacement placement);
int bn_transformer_attention_uses_flash(const BnConfig *c,
                                        const BnGPUBackend *gpu);
int bn_transformer_attention_uses_packed_qkv(
    const BnGPUBackend *gpu,
    const BnLayerShapePlan *shape,
    const BnLayerWeights *lw,
    const void *qkv_stacked,
    const void *q_bias,
    const void *k_bias,
    const void *v_bias);
int bn_transformer_attention_uses_qkv_split(
    const BnGPUBackend *gpu,
    const BnLayerShapePlan *shape,
    const BnLayerWeights *lw,
    const void *qkv_stacked);
int bn_transformer_attention_uses_rope_qk_fusion(
    BnExecPlacement placement,
    const void *k_bias);
BnKVMode bn_transformer_kv_mode(const BnConfig *c, int tq_enabled);
int bn_transformer_kv_mode_stores_host_float_rows(BnKVMode mode);
int bn_transformer_kv_mode_uses_turboquant(BnKVMode mode);
int bn_transformer_kv_mode_uses_fp16(BnKVMode mode);
int bn_transformer_kv_mode_uses_cpu_gqa_cache(BnKVMode mode);
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
uint32_t bn_transformer_cpu_float_kquant_fallback_task_flags(
    const BnConfig *c);
int bn_transformer_cpu_prefill_uses_float_kquant_fallback(
    const BnConfig *c);
int bn_transformer_cpu_prefill_decode_for_parity_enabled(
    const BnConfig *c,
    int gpu_attached);
int bn_transformer_rmsnorm_uses_reference_order(
    const BnConfig *c);
int bn_transformer_rmsnorm_requires_reference_scalar_order(
    const BnConfig *c);
float bn_transformer_attention_scale(
    const BnConfig *c,
    int head_size);
int bn_transformer_attention_value_shares_key(
    const BnConfig *c);
int bn_transformer_attention_uses_post_norm(
    const BnConfig *c);
int bn_transformer_ffn_uses_post_norm(
    const BnConfig *c);
int bn_transformer_uses_layer_output_scale(
    const BnConfig *c);
BnFFNKind bn_transformer_ffn_kind(const BnConfig *c,
                                  const BnLayerWeights *lw);
int bn_transformer_ffn_hidden_dim(const BnConfig *c,
                                  const BnLayerWeights *lw);
int bn_transformer_ffn_has_gate(const BnConfig *c);
int bn_transformer_ffn_has_sub_norm(const BnLayerWeights *lw);
int bn_transformer_ffn_uses_fused_gateup_silu(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    BnExecPlacement placement);
int bn_transformer_ffn_uses_gateup_split(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    BnExecPlacement placement,
    const void *gateup_stacked);
int bn_transformer_ffn_uses_residual_rmsnorm_fusion(
    BnExecPlacement placement);
int bn_transformer_ffn_requires_cpu_fallback(
    BnFFNKind kind,
    BnExecPlacement placement);
int bn_transformer_moe_has_shared_expert(const BnConfig *c,
                                         const BnLayerWeights *lw);
int bn_transformer_moe_layer_has_router(const BnLayerWeights *lw);
int bn_transformer_moe_requires_cpu_fallback(BnExecPlacement placement,
                                             const BnLayerWeights *lw);
int bn_transformer_ssm_uses_qkvz_stack(
    BnExecPlacement placement,
    const void *qkvz_stacked);
int bn_transformer_ssm_uses_alpha_beta_stack(
    BnExecPlacement placement,
    const void *alpha_beta_stacked);
int bn_transformer_logits_uses_i8_output(const BnWeights *w);
int bn_transformer_logits_has_untied_output(const BnWeights *w);
BnLogitsKind bn_transformer_logits_kind(const BnWeights *w);
int bn_transformer_logits_weight_type(const BnWeights *w);
int bn_transformer_per_layer_embedding_dim(
    const BnConfig *c);
int bn_transformer_uses_per_layer_embedding(
    const BnConfig *c);
int bn_transformer_divides_rope_freqs(
    const BnConfig *c,
    int layer);
int bn_transformer_rope_dims_for_head(
    const BnConfig *c,
    int layer_head_size);
float bn_transformer_rope_theta_for_head(
    const BnConfig *c,
    int layer_head_size);
float bn_transformer_rope_base_theta(
    const BnConfig *c);
int bn_transformer_rope_uses_base_frequency(
    const BnConfig *c,
    int layer_head_size);
int bn_transformer_cpu_uses_scalar_hybrid_ssm(
    const BnConfig *c);
int bn_transformer_prefill_uses_exact_activation(
    const BnConfig *c);
int bn_transformer_ffn_uses_exact_scalar_activation(
    const BnConfig *c);
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
