#ifndef BN_TRANSFORMER_GPU_INTERNAL_H
#define BN_TRANSFORMER_GPU_INTERNAL_H

#include "backend_session.h"
#include "backend_model.h"
#include "gpu_backend.h"
#include "gpu_graph_ir.h"
#include "gpu_moe_bridge.h"
#include "model.h"
#include "session.h"
#include "transformer_plan_internal.h"

#define BN_TRANSFORMER_GPU_MAX_VLA_ELEMS 8192

typedef struct {
    void *lowered_ops;
    int n;
    int cap;
    BnGPUValueGraph graph_storage;
    BnGPUValueGraph *graph;
    void *lowering_values;
    int cap_lowering_values;
    int owns_graph_storage;
    int owns_lowering_values;
} BnTransformerGPUEmitContext;

typedef struct {
    const BnGPUBackend *gpu;
    void *gateup_stacked;
    void *ffn_sub_norm;
    void *ffn_gate;
    void *ffn_up;
    void *ffn_down;
} BnTransformerGPUDenseFFNResources;

typedef struct {
    const BnGPUBackend *gpu;
    void *q_bias;
    void *k_bias;
    void *v_bias;
    void *q_norm;
    void *k_norm;
    void *qkv_stacked;
    void *qk_stacked;
    void *packed_qkv;
    void *wq;
    void *wk;
    void *wv;
} BnTransformerGPUQKVResources;

typedef struct {
    const BnGPUBackend *gpu;
    void *k_bias;
    void *attn_sub_norm;
    void *ffn_norm;
    void *wo;
} BnTransformerGPUAttentionResources;

typedef struct {
    const BnGPUBackend *gpu;
    void *ssm_qkvz_stacked;
    void *ssm_ab_stacked;
    void *ssm_conv1d;
    void *ssm_dt_bias;
    void *ssm_a_log;
    void *ssm_norm;
    void *ffn_norm;
    void *wqkv;
    void *wz;
    void *ssm_alpha;
    void *ssm_beta;
    void *ssm_out;
} BnTransformerGPUSSMResources;

typedef struct {
    const BnGPUBackend *gpu;
    void *shared_gate;
    void *shared_up;
    void *shared_down;
    void *shared_expert_gate;
    void *shared_gateup_stacked;
} BnTransformerGPUMoESharedResources;

typedef struct {
    int gate_type;
    int up_type;
    int down_type;
    int gate_rows;
    int up_rows;
    int down_rows;
    int gate_cols;
    int up_cols;
    int down_cols;
} BnTransformerGPUMoESharedProjectionInfo;

typedef struct {
    void *gate;
    void *up;
    void *down;
    void *gate_weight;
    int hidden_dim;
    int gate_type;
    int up_type;
    int down_type;
} BnTransformerGPUMoESharedFFNResources;

typedef struct {
    void *attn_norm;
    void *ffn_norm;
    void *q_norm;
    void *k_norm;
    void *attn_sub_norm;
    void *ffn_sub_norm;
} BnTransformerGPULayerValidationResources;

typedef struct {
    void *gpu_buf;
    int type;
    int rows;
    int cols;
    BnQWeight *cpu_weight;
    BnQWeight tied_weight;
} BnTransformerGPULogitResources;

typedef struct {
    void *output_norm;
    BnTransformerGPULogitResources logits;
    int has_moe;
    int has_ssm;
} BnTransformerGPUForwardPolicy;

int bn_transformer_gpu_activation_uses_silu_path(int activation);
int bn_transformer_gpu_activation_is_relu2(int activation);
BnGPUIRActivationKind bn_transformer_gpu_ffn_activation_kind(int activation);

typedef struct {
    int block_argmax;
} BnTransformerGPUDecodeEntryPolicy;

typedef struct {
    int layer;
    int from_layer;
    int attn_layer;
    int attn_from_layer;
    int ffn_layer;
    int ffn_from_layer;
    int ffn_down_from_layer;
} BnTransformerGPUCPUFallbackPolicy;

typedef struct {
    int keep_backend_matvec;
    int disable_backend_matvec;
} BnTransformerGPUMatvecFallbackPolicy;

typedef struct {
    int from_layer;
    int to_layer;
    int attn_only;
    int ffn_only;
} BnTransformerGPUSmallDenseExactNativeLayerPolicy;

typedef struct {
    int small_dense_exact_native_default;
    int small_dense_exact_native_to_layer;
} BnTransformerGPUSmallDenseExactNativeDecodePolicy;

typedef struct {
    int use_layer;
    int small_dense_exact_native_path;
    int use_attention;
    int use_ffn;
    int use_ffn_down;
} BnTransformerGPUSmallDenseExactNativeLayerUsePolicy;

typedef struct {
    int use_cache;
    int clear_cache;
} BnTransformerGPUCachedDecodePolicy;

typedef struct {
    int resident_moe;
    int graph_cacheable;
} BnTransformerGPUDecodeCacheabilityPolicy;

typedef struct {
    int from_layer;
    int to_layer;
} BnTransformerGPUMoERouteLayerPolicy;

typedef struct {
    int all_active_two_kquant_moe;
    int route_layer_selected;
    int exact_gpu_route;
    int gpu_route_topk;
    int cpu_route_resident_ffn;
    int gpu_routed_ffn;
    uint32_t route_flags;
    void *router;
} BnTransformerGPUMoEDecodeRoutePolicy;

typedef struct {
    int override_cpu_actual;
    int compare_layer;
    int compare_route;
    int compare_input_norm;
    int compare_actual;
    int compare_raw;
    int compare_mid;
    int compare_parts;
    int compare_shared_mid;
    int compare_shared_down;
    int compare_norm;
} BnTransformerGPUMoEDebugPolicy;

typedef struct {
    int enabled;
} BnTransformerGPUMoESharedCPUFallbackPolicy;

typedef struct {
    int enabled;
    void *router_diff;
} BnTransformerGPUMoEDirectRoutePolicy;

typedef struct {
    int enabled;
} BnTransformerGPUMoEAllActiveTwoResourcePolicy;

typedef struct {
    int kquant_default;
    int kquant_enabled;
    int kquant_captures_xb;
    int kquant_refine_top;
    int native_quant_default;
    int native_quant_enabled;
    int native_quant_captures_xb;
    int native_quant_refine_top;
} BnTransformerGPULogitsRefinePolicy;

typedef struct {
    int enabled;
} BnTransformerGPUGenerateArgmaxPolicy;

typedef struct {
    int attention_layer;
    int attention_pos;
    int gqa_layer;
    int gqa_pos;
    int qkv_layer;
    int qkv_pos;
    int ffn_down_layer;
    int ffn_down_pos;
    int ffn_state_layer;
    int ffn_state_pos;
} BnTransformerGPUComparePolicy;

typedef struct {
    int uses_moe;
} BnTransformerGPULayerKindPolicy;

typedef struct {
    int use_kquant_dot;
    int use_fused_gateup;
    int use_gateup_split;
} BnTransformerGPUSharedExpertGateupPolicy;

int bn_transformer_gpu_validate_forward(
    BnTransformerGPUForwardPolicy *out,
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnWeights *w,
    int token,
    int pos,
    const char **reject_reason);
int bn_transformer_gpu_graph_op_capacity(const BnConfig *c);
int bn_transformer_gpu_can_layerwise_rope(const BnGPUBackend *gpu);
int bn_transformer_gpu_uses_small_dense_shape(const BnConfig *c);
int bn_transformer_gpu_uses_large_dense_shape(const BnConfig *c);
int bn_transformer_gpu_uses_large_graph_fallback_shape(const BnConfig *c);
int bn_transformer_gpu_uses_per_layer_embedding(const BnConfig *c);
int bn_transformer_gpu_uses_hybrid_ssm(const BnConfig *c);
int bn_transformer_gpu_uses_large_dense_hybrid_ssm(const BnConfig *c);
int bn_transformer_gpu_uses_non_hybrid_moe(const BnConfig *c);
int bn_transformer_gpu_uses_moe(const BnConfig *c);
int bn_transformer_gpu_uses_dense_attention_only(const BnConfig *c);
int bn_transformer_gpu_uses_small_dense_native_quant_shape(
    const BnConfig *c);
BnTransformerGPULayerKindPolicy
bn_transformer_gpu_layer_kind_policy(const BnLayerWeights *lw);
int bn_transformer_gpu_should_upload_ssm_state(const BnConfig *c);
int bn_transformer_gpu_requires_float_kquant(const BnConfig *c);
int bn_transformer_gpu_dense_batch_prefill_shape_allowed_for_backend(
    const BnConfig *c,
    int supports_large_dense_batch_prefill);
int bn_transformer_gpu_dense_logits_argmax_shape_allowed(
    const BnConfig *c,
    int logits_rows);
int bn_transformer_gpu_moe_logits_mmvq_argmax_shape_allowed(
    const BnConfig *c,
    int logits_cols);
int bn_transformer_gpu_requires_layerwise_rope(const BnConfig *c,
                                               const BnWeights *w);
uint32_t bn_transformer_gpu_moe_gateup_task_flags(const BnConfig *c);
int bn_transformer_gpu_prefill_quant_matmul_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_quant_matmul_batch_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_quant_matmul_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *buf,
    const float *X,
    int rows,
    int cols,
    int n_tokens,
    int tensor_type);
int bn_transformer_gpu_prefill_quant_matmul_batch_backend_run(
    BnGPUBackend *gpu,
    const BnGPUMatvecOp *ops,
    int n_ops,
    const float *X,
    int n_tokens,
    int cols);
int bn_transformer_gpu_prefill_dense_ffn_batch_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_dense_ffn_batch_norm_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_dense_ffn_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type);
int bn_transformer_gpu_prefill_dense_ffn_batch_norm_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps);
int bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps);
int bn_transformer_gpu_prefill_qkv_attention_wo_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_qkv_attention_wo_norm_resid_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_attention_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_attention_wo_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_qkv_attention_wo_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale);
int bn_transformer_gpu_prefill_qkv_attention_wo_norm_resid_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *attn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale);
int bn_transformer_gpu_prefill_attention_backend_run(
    BnGPUBackend *gpu,
    float *out,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    float attention_scale);
int bn_transformer_gpu_prefill_attention_wo_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *wo_buf,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int wo_rows,
    int wo_cols,
    int wo_type,
    float attention_scale);
int bn_transformer_gpu_moe_gateup_split_supported(
    const BnGPUBackend *gpu,
    const BnMoEExpertMap *map,
    int split_op_code);
int bn_transformer_gpu_matvec_split_op_code(int tensor_type);
int bn_transformer_gpu_dense_gateup_exact_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *gate,
    const BnQWeight *up,
    int activation,
    int split_op_code);
int bn_transformer_gpu_dense_ffn_prefers_gateup_split(
    const BnConfig *c,
    int gate_type);
int bn_transformer_gpu_packed_qkv_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *qkv,
    int use_packed_qkv,
    int kv_cache_uses_fp16_rows,
    int split_op_code);
int bn_transformer_gpu_qkv_split_standard_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code);
int bn_transformer_gpu_qkv_split_native_quant_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code);
int bn_transformer_gpu_qkv_split_packed_kquant_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code);
int bn_transformer_gpu_qk_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    const BnQWeight *k,
    int q_dim,
    int kv_dim,
    int split_op_code);
int bn_transformer_gpu_ssm_qkvz_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *qkv,
    int split_op_code);
int bn_transformer_gpu_can_stack_same_quant_format_alpha_beta(
    const BnQWeight *alpha,
    const BnQWeight *beta);
int bn_transformer_gpu_shared_kquant_gateup_dot_eligible(int gate_type,
                                                         int up_type,
                                                         int cols);
int bn_transformer_gpu_shared_kquant_dot_enabled(int eligible);
int bn_transformer_gpu_shared_expert_prefers_gateup_split(int gate_type);
int bn_transformer_gpu_shared_expert_path_available(
    const BnLayerWeights *lw,
    const BnTransformerGPUMoESharedResources *shared);
int bn_transformer_gpu_shared_expert_gate_available(
    const BnLayerWeights *lw,
    const BnTransformerGPUMoESharedResources *shared);
BnTransformerGPUSharedExpertGateupPolicy
bn_transformer_gpu_shared_expert_gateup_policy(
    const BnLayerWeights *lw,
    const BnTransformerGPUMoESharedResources *shared);
int bn_transformer_gpu_logits_needs_cpu_fallback(
    const BnGPUBackend *gpu,
    const BnTransformerGPULogitResources *logits);
int bn_transformer_gpu_all_active_two_kquant_moe_model(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_all_active_two_kquant_moe_layer(
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim);
int bn_transformer_gpu_all_active_two_kquant_moe_layer_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim);
int bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_small_dense_native_quant_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_small_dense_exact_native_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int small_dense_exact_native_from_layer);
int bn_transformer_gpu_small_dense_exact_native_to_layer(
    const BnConfig *c,
    int small_dense_exact_native_default,
    int small_dense_exact_native_to_layer);
int bn_transformer_gpu_small_dense_exact_native_ffn_down_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_large_hybrid_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_large_hybrid_cpu_attn_safe_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_small_dense_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_hybrid_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_moe_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_dense_ffn_batch_tokens_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens);
int bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_backend_matvec_fallback_kept(
    const BnModel *m,
    const BnGPUBackend *gpu);
BnTransformerGPUMatvecFallbackPolicy
bn_transformer_gpu_matvec_fallback_policy(
    const BnModel *m,
    const BnGPUBackend *gpu);
int bn_transformer_gpu_small_dense_exact_native_fused_gateup_enabled(
    int use_small_dense_exact_native);
int bn_transformer_gpu_small_dense_exact_native_down_enabled(
    int use_small_dense_exact_native_down);
int bn_transformer_gpu_qkv_split_enabled(int use_small_dense_exact_native);
int bn_transformer_gpu_qk_split_enabled(void);
int bn_transformer_gpu_qkv_split_debug_enabled(void);
int bn_transformer_gpu_ssm_qkvz_split_enabled(void);
int bn_transformer_gpu_ssm_ab_stack_enabled(void);
int bn_transformer_gpu_split_residual_rmsnorm_enabled(void);
int bn_transformer_gpu_prefill_ssm_layer_disabled(void);
int bn_transformer_gpu_batch_prefill_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_dense_batch_prefill_shape_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_large_hybrid_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_large_hybrid_argmax_blocked(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax);
int bn_transformer_gpu_large_hybrid_prefill_chain_disabled_default(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_prefill_direct_kv_allowed(
    const BnConfig *c,
    const BnWeights *w,
    const BnGPUBackend *gpu,
    int pos0,
    int n_tokens);
int bn_transformer_gpu_prefill_attention_min_tokens(void);
int bn_transformer_gpu_prefill_dense_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_moe_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_moe_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down);
int bn_transformer_gpu_prefill_moe_ffn_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    void *ffn_norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    float norm_eps,
    int norm_topk_prob,
    float expert_weights_scale);
int bn_transformer_gpu_prefill_dense_layer_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_dense_layer_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale);
int bn_transformer_gpu_prefill_moe_layer_backend_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down);
int bn_transformer_gpu_prefill_moe_layer_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int moe_hidden_dim,
    int n_experts,
    int experts_active,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale,
    int norm_topk_prob,
    float expert_weights_scale);
int bn_transformer_gpu_prefill_moe_layer_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int n_tokens);
int bn_transformer_gpu_prefill_ssm_moe_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int n_tokens);
int bn_transformer_gpu_prefill_ssm_layer_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_prefill_ssm_layer_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *wqkv_buf,
    void *wz_buf,
    void *alpha_buf,
    void *beta_buf,
    void *qkvz_stacked_buf,
    void *ab_stacked_buf,
    void *ssm_out_buf,
    void *attn_norm_buf,
    void *conv1d_buf,
    void *dt_bias_buf,
    void *a_log_buf,
    void *ssm_norm_buf,
    void *ffn_gate_buf,
    void *ffn_up_buf,
    void *ffn_down_buf,
    void *ffn_norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int qkv_dim,
    int inner_dim,
    int num_k_heads,
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int conv_kernel,
    int ssm_idx,
    int wqkv_type,
    int wz_type,
    int alpha_type,
    int beta_type,
    int out_type,
    int hidden_dim,
    int ffn_gate_type,
    int ffn_up_type,
    int ffn_down_type,
    int act_type,
    float norm_eps,
    int *did_ffn);
int bn_transformer_gpu_prefill_ssm_dense_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens);
int bn_transformer_gpu_prefill_dense_chain_enabled(void);
int bn_transformer_gpu_prefill_hybrid_chain_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_prefill_attention_enabled(void);
int bn_transformer_gpu_prefill_ssm_run_chain_enabled(void);
int bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed(void);
int bn_transformer_gpu_prefill_moe_chain_debug_enabled(void);
int bn_transformer_gpu_prefill_hybrid_chain_debug_enabled(void);
int bn_transformer_gpu_moe_prefill_enabled(void);
int bn_transformer_gpu_moe_prefill_min_tokens(void);
int bn_transformer_gpu_moe_prefill_backend_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_moe_prefill_tokens_allowed(
    const BnGPUBackend *gpu,
    int n_tokens);
int bn_transformer_gpu_moe_cache_prefill_enabled(void);
int bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int gpu_moe_cache_available);
int bn_transformer_gpu_moe_prefill_shared_fuse_enabled(void);
int bn_transformer_gpu_moe_prefill_shared_batch_available(
    const BnGPUBackend *gpu,
    int n_tokens,
    int backend_available);
int bn_transformer_gpu_moe_prefill_shared_dense_ffn_available(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int backend_available);
int bn_transformer_gpu_moe_route_batch_debug_enabled(void);
int bn_transformer_gpu_moe_prefill_route_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int backend_available);
int bn_transformer_gpu_moe_prefill_route_batch_backend_run(
    BnGPUBackend *gpu,
    int *indices,
    float *weights,
    void *router_buf,
    const float *X,
    int n_tokens,
    int dim,
    int n_experts,
    int k,
    int norm_topk_prob,
    float expert_weights_scale);
int bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down);
int bn_transformer_gpu_moe_prefill_routed_ffn_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int norm_topk_prob,
    float expert_weights_scale);
int bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int prefer_cached_expert_batch);
int bn_transformer_gpu_moe_prefill_resident_expert_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const int *indices,
    const float *weights,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type);
int bn_transformer_gpu_moe_prefill_split_expert_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int used_resident_expert_batch);
int bn_transformer_gpu_moe_prefill_split_expert_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    const BnGPUMoEPrefillExpert *experts,
    int n_experts,
    const int *expert_offsets,
    const int *expert_counts,
    const int *token_ids,
    const float *weights,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type);
int bn_transformer_gpu_moe_prefill_single_expert_batch_available(
    const BnGPUBackend *gpu,
    int n_tokens);
int bn_transformer_gpu_moe_lazy_aux_cache_enabled(void);
int bn_transformer_gpu_moe_quant_only_without_aux_cache(
    const BnGPUBackend *gpu,
    int tensor_type,
    int allow_aux_cache);
int bn_transformer_gpu_large_hybrid_prefill_disabled(void);
int bn_transformer_gpu_native_quant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int tensor_type);
int bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_kquant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int kquant_refine_default);
int bn_transformer_gpu_kquant_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_kquant_logits,
    int kquant_refine_default);
int bn_transformer_gpu_kquant_logits_refine_top(int kquant_refine_default);
int bn_transformer_gpu_kquant_logits_refine_blocks_per_row(int cols);
int bn_transformer_gpu_kquant_logits_refine_block_sums_per_row(
    int blocks_per_row);
int bn_transformer_gpu_native_quant_logits_refine_active(
    const BnGPUBackend *gpu,
    int native_quant_refine_default);
int bn_transformer_gpu_native_quant_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_native_quant_logits);
int bn_transformer_gpu_native_quant_logits_refine_top(
    int native_quant_refine_default);
BnTransformerGPULogitsRefinePolicy bn_transformer_gpu_logits_refine_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    const BnTransformerGPULogitResources *logits,
    int small_dense_exact_native_default);
BnTransformerGPUGenerateArgmaxPolicy
bn_transformer_gpu_generate_argmax_policy(
    const BnGPUBackend *gpu,
    int top_logits,
    float temperature,
    float repeat_penalty);
int bn_transformer_gpu_argmax_available(
    const BnGPUBackend *gpu,
    int want_argmax);
int bn_transformer_gpu_argmax_backend_run(
    BnGPUBackend *gpu,
    int buf_idx,
    int n,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token);
int bn_transformer_gpu_matvec_argmax_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPULogitResources *logits,
    int want_argmax,
    int need_logits,
    int gpu_logits_need_cpu);
int bn_transformer_gpu_matvec_argmax_backend_run(
    BnGPUBackend *gpu,
    void *W_buf,
    int type,
    int rows,
    int cols,
    int buf_idx,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token);
int bn_transformer_gpu_moe_decode_cacheable(
    const BnConfig *c,
    const BnWeights *w,
    const BnBackendModel *backend);
int bn_transformer_gpu_decode_cacheable(
    const BnGPUBackend *gpu,
    int emit_logits,
    int want_argmax,
    int gpu_logits_need_cpu,
    int has_moe,
    int cacheable_resident_moe,
    int kquant_logits_refine_captures_xb,
    int native_quant_logits_refine_captures_xb,
    int need_logits,
    int cpu_fallback_layer,
    int cpu_fallback_from_layer,
    int cpu_fallback_attn_layer,
    int cpu_fallback_attn_from_layer,
    int cpu_fallback_ffn_layer,
    int cpu_fallback_ffn_from_layer,
    int cpu_fallback_ffn_down_from_layer,
    int compare_attention_layer,
    int compare_gqa_layer,
    int compare_qkv_layer,
    int compare_ffn_down_layer,
    int compare_ffn_state_layer);
BnTransformerGPUDecodeCacheabilityPolicy
bn_transformer_gpu_decode_cacheability_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    const BnBackendModel *backend,
    int emit_logits,
    int want_argmax,
    int gpu_logits_need_cpu,
    int has_moe,
    const BnTransformerGPULogitsRefinePolicy *logits_refine,
    int need_logits,
    const BnTransformerGPUCPUFallbackPolicy *cpu_fallback,
    const BnTransformerGPUComparePolicy *compare);
int bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_moe_exact_attention_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_ssm_cpu_fallback_required(
    const BnGPUBackend *gpu);
BnTransformerGPUDecodeEntryPolicy
bn_transformer_gpu_decode_entry_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax);
BnTransformerGPUCPUFallbackPolicy
bn_transformer_gpu_cpu_fallback_policy(void);
BnTransformerGPUCPUFallbackPolicy
bn_transformer_gpu_decode_cpu_attention_fallback_policy(
    BnTransformerGPUCPUFallbackPolicy policy,
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_cpu_fallback_layer_selected(
    int layer,
    int exact_layer,
    int from_layer);
BnTransformerGPUSmallDenseExactNativeLayerPolicy
bn_transformer_gpu_small_dense_exact_native_layer_policy(const BnConfig *c);
BnTransformerGPUSmallDenseExactNativeDecodePolicy
bn_transformer_gpu_small_dense_exact_native_decode_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPUSmallDenseExactNativeLayerPolicy *layer_policy);
BnTransformerGPUSmallDenseExactNativeLayerUsePolicy
bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPUSmallDenseExactNativeLayerPolicy *policy,
    int layer,
    int small_dense_exact_native_default,
    int small_dense_exact_native_to_layer);
BnTransformerGPUCachedDecodePolicy
bn_transformer_gpu_cached_decode_policy(
    int cached_op_count,
    int argmax_requested,
    int cached_has_logits,
    int matvec_argmax_available);
BnTransformerGPUMoERouteLayerPolicy
bn_transformer_gpu_moe_route_layer_policy(void);
BnTransformerGPUComparePolicy
bn_transformer_gpu_compare_policy(void);
int bn_transformer_gpu_flash_attention_enabled(
    const BnGPUBackend *gpu,
    int config_flash_attn,
    int has_moe,
    int n_kv);
int bn_transformer_gpu_moe_routed_kquant_down(const BnMoEExpertMap *map);
int bn_transformer_gpu_moe_routed_kquant_down_allowed(
    const BnMoEExpertMap *map,
    int allow_kquant_down);
int bn_transformer_gpu_moe_routed_native_quant(const BnMoEExpertMap *map);
int bn_transformer_gpu_moe_route_topk_enabled(
    void *moe_router,
    int all_active_two_kquant_moe,
    int all_active_two_kquant_moe_gpu_route_layer_selected);
int bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
    const BnConfig *c,
    int all_active_two_kquant_moe,
    int gpu_route_topk,
    int moe_routed_native_quant);
uint32_t bn_transformer_gpu_moe_route_normalization_flags(const BnConfig *c);
int bn_transformer_gpu_moe_routed_ffn_enabled(
    int gpu_route_topk,
    int cpu_route_resident_ffn,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all,
    const BnMoEExpertMap *map,
    const BnConfig *c,
    int dim);
BnTransformerGPUMoEDecodeRoutePolicy
bn_transformer_gpu_moe_decode_route_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUMoERouteLayerPolicy *layer_policy,
    int layer,
    int dim,
    void *moe_router,
    void *router_diff,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all);
BnTransformerGPUMoEDirectRoutePolicy
bn_transformer_gpu_moe_direct_route_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all);
BnTransformerGPUMoEAllActiveTwoResourcePolicy
bn_transformer_gpu_moe_all_active_two_resource_policy(const BnConfig *c);
int bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all);
int bn_transformer_gpu_all_active_two_kquant_moe_route_layer_selected(
    int layer,
    int route_from_layer,
    int route_to_layer);
void bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
    int *route_from_layer,
    int *route_to_layer);
int bn_transformer_gpu_all_active_two_kquant_moe_exact_gpu_route_enabled(
    int all_active_two_kquant_moe,
    int route_layer_selected);
void *bn_transformer_gpu_all_active_two_kquant_moe_router(
    const BnConfig *c,
    void *moe_router,
    void *router_diff,
    int route_layer_selected,
    int exact_gpu_route);
int bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down);
int bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int layer,
    int cpu_fallback_ffn_layer,
    int cpu_fallback_ffn_from_layer);
int bn_transformer_gpu_moe_routed_ffn_batch_allowed(
    const BnConfig *c);
int bn_transformer_gpu_moe_ffn_disabled(void);
int bn_transformer_gpu_moe_cpu_actual_override_enabled(int safe_default);
BnTransformerGPUMoEDebugPolicy bn_transformer_gpu_moe_debug_policy(
    int cpu_actual_safe_default,
    int compare_layer_selected);
BnTransformerGPUMoEDebugPolicy bn_transformer_gpu_moe_decode_debug_policy(
    const BnConfig *c,
    const BnWeights *w,
    int layer,
    int pos);
int bn_transformer_gpu_moe_compare_layer_selected(int layer, int pos);
int bn_transformer_gpu_moe_compare_input_norm_enabled(void);
int bn_transformer_gpu_moe_compare_actual_enabled(void);
int bn_transformer_gpu_moe_compare_route_enabled(void);
int bn_transformer_gpu_moe_compare_raw_enabled(void);
int bn_transformer_gpu_moe_compare_mid_enabled(void);
int bn_transformer_gpu_moe_compare_parts_enabled(void);
int bn_transformer_gpu_moe_compare_shared_mid_enabled(void);
int bn_transformer_gpu_moe_compare_shared_down_enabled(void);
int bn_transformer_gpu_moe_compare_norm_enabled(void);
int bn_transformer_gpu_moe_shared_cpu_fallback_enabled(int eligible);
int bn_transformer_gpu_moe_has_loaded_shared_expert(
    const BnConfig *c,
    const BnLayerWeights *lw);
BnTransformerGPUMoESharedCPUFallbackPolicy
bn_transformer_gpu_moe_shared_cpu_fallback_policy(
    const BnConfig *c,
    const BnLayerWeights *lw);
int bn_transformer_gpu_moe_gateup_split_enabled(
    const BnGPUBackend *gpu,
    int can_split);
int bn_transformer_gpu_cpu_logits_enabled(int gpu_logits_need_cpu);
int bn_transformer_gpu_debug_argmax_compare_enabled(void);
int bn_transformer_gpu_argmax_debug_enabled(void);
int bn_transformer_gpu_compare_logits_enabled(void);
int bn_transformer_gpu_moe_route_profile_enabled(void);
int bn_transformer_gpu_moe_route_profile_every(void);
int bn_transformer_gpu_profile_level(void);
int bn_transformer_gpu_debug_fallback_enabled(void);
void bn_transformer_gpu_report_fallback(const char *reason);
float *bn_transformer_gpu_reject_forward(
    BnTransformerGPUEmitContext *emit,
    const char *reason);
void *bn_transformer_gpu_resolve_output_norm(
    const BnBackendModel *backend);
void *bn_transformer_gpu_resolve_initial_norm(
    const BnBackendModel *backend);
void *bn_transformer_gpu_resolve_next_norm(
    const BnBackendModel *backend,
    int layer,
    int n_layers,
    void *output_norm);
BnTransformerGPULayerValidationResources
bn_transformer_gpu_resolve_layer_validation_resources(
    const BnBackendModel *backend,
    int layer);
void bn_transformer_gpu_resolve_logit_resources(
    BnTransformerGPULogitResources *out,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnWeights *w);
BnTransformerGPUDenseFFNResources
bn_transformer_gpu_resolve_dense_ffn_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUQKVResources bn_transformer_gpu_resolve_qkv_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUAttentionResources
bn_transformer_gpu_resolve_attention_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUSSMResources bn_transformer_gpu_resolve_ssm_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUMoESharedResources
bn_transformer_gpu_resolve_moe_shared_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
int bn_transformer_gpu_resolve_moe_shared_projection_info(
    BnTransformerGPUMoESharedProjectionInfo *out,
    const BnLayerWeights *lw);
int bn_transformer_gpu_resolve_moe_shared_ffn_resources(
    BnTransformerGPUMoESharedFFNResources *out,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int layer,
    int allow_stacked_gateup);

void bn_transformer_gpu_finalize_op_kinds(void *ops, int n);
void bn_transformer_gpu_emit_context_init(BnTransformerGPUEmitContext *ctx,
                                          void *lowered_ops,
                                          int cap);
int bn_transformer_gpu_emit_context_init_session(
    BnTransformerGPUEmitContext *ctx,
    BnBackendSession *backend,
    void *lowered_ops,
    int cap,
    int cap_values,
    int cap_ops);
int bn_transformer_gpu_emit_context_reserve(
    BnTransformerGPUEmitContext *ctx,
    int cap_values,
    int cap_ops);
void bn_transformer_gpu_emit_context_free(BnTransformerGPUEmitContext *ctx);
int bn_transformer_gpu_emit_context_lower_pending(
    BnTransformerGPUEmitContext *ctx);
int bn_transformer_gpu_emit_context_execute(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu,
    int readback_buf,
    float *readback,
    int readback_count);
int bn_transformer_gpu_emit_context_flush(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu);
int bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
    BnTransformerGPUEmitContext *ctx,
    void *norm_gpu,
    int dim,
    uint32_t u_eps);
int bn_transformer_gpu_emit_context_execute_logits(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu,
    float *logits,
    int vocab_size);
int bn_transformer_gpu_emit_context_rmsnorm(BnTransformerGPUEmitContext *ctx,
                                            void *norm_gpu,
                                            int buf_in,
                                            int buf_out,
                                            int dim,
                                            uint32_t u_eps);
int bn_transformer_gpu_emit_context_logits(BnTransformerGPUEmitContext *ctx,
                                           void *logit_gpu_buf,
                                           int logit_type,
                                           int logit_rows,
                                           int logit_cols);
int bn_transformer_gpu_emit_context_copy(BnTransformerGPUEmitContext *ctx,
                                         int buf_in,
                                         int buf_out,
                                         int src_offset,
                                         int dst_offset,
                                         int count);
int bn_transformer_gpu_emit_context_residual_add(
    BnTransformerGPUEmitContext *ctx,
    int buf_in,
    int buf_aux,
    int count);
int bn_transformer_gpu_emit_context_residual_rmsnorm(
    BnTransformerGPUEmitContext *ctx,
    int x_buf,
    int residual_buf,
    int out_buf,
    int dim,
    uint32_t u_eps,
    void *norm_weight);
int bn_transformer_gpu_emit_context_activation(
    BnTransformerGPUEmitContext *ctx,
    int buf_in,
    int buf_aux,
    int count,
    int param1,
    BnGPUIRActivationKind kind);
int bn_transformer_gpu_emit_context_matvec(BnTransformerGPUEmitContext *ctx,
                                           int type,
                                           void *weight_buf,
                                           int buf_in,
                                           int buf_out,
                                           int rows,
                                           int cols,
                                           int output_offset);
int bn_transformer_gpu_emit_context_matvec_flags(
    BnTransformerGPUEmitContext *ctx,
    int type,
    void *weight_buf,
    int buf_in,
    int buf_out,
    int rows,
    int cols,
    int output_offset,
    uint32_t flags);
int bn_transformer_gpu_emit_context_fused_gateup_silu(
    BnTransformerGPUEmitContext *ctx,
    int type,
    void *weight_buf,
    int buf_in,
    int buf_out,
    int gate_rows,
    int up_rows,
    int cols,
    int use_small_dense_exact_native,
    uint32_t flags);
int bn_transformer_gpu_emit_context_moe_route_topk(
    BnTransformerGPUEmitContext *ctx,
    void *router_buf,
    int buf_in,
    int logits_buf,
    int route_buf,
    int dim,
    int n_experts,
    int k,
    float expert_weights_scale,
    uint32_t flags);
int bn_transformer_gpu_emit_context_moe_routed_ffn(
    BnTransformerGPUEmitContext *ctx,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    int buf_in,
    int route_buf,
    int buf_mid,
    int buf_out,
    int gate_type,
    int down_type,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int exact_silu,
    int layer);
int bn_transformer_gpu_fallback_ssm_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_fallback_moe_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_fallback_cpu_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_fallback_cpu_attention(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_fallback_cpu_ffn(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_fallback_cpu_ffn_down(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int down_input_buf,
    int hidden_dim,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_debug_compare_ffn_down(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int down_input_buf,
    int hidden_dim,
    int dim);
int bn_transformer_gpu_debug_compare_ffn_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    const float *next_norm,
    int layer,
    int pos,
    int dim);
int bn_transformer_gpu_fallback_logits(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    const BnTransformerGPULogitResources *logits,
    int dim);
int bn_transformer_gpu_execute_ops(const BnGPUBackend *gpu,
                                   void *ops,
                                   int n,
                                   int readback_buf,
                                   float *readback,
                                   int readback_count);
int bn_transformer_gpu_write_x(const BnGPUBackend *gpu,
                               const float *x,
                               size_t size_bytes);
int bn_transformer_gpu_write_activation_buf(const BnGPUBackend *gpu,
                                            int buf_idx,
                                            const void *data,
                                            size_t size_bytes);
int bn_transformer_gpu_write_activation_buf_offset(const BnGPUBackend *gpu,
                                                   int buf_idx,
                                                   const void *data,
                                                   size_t size_bytes,
                                                   size_t offset_bytes);
int bn_transformer_gpu_read_x(const BnGPUBackend *gpu,
                              float *x,
                              size_t size_bytes);
int bn_transformer_gpu_read_xb(const BnGPUBackend *gpu,
                               float *xb,
                               size_t size_bytes);
int bn_transformer_gpu_read_xb2(const BnGPUBackend *gpu,
                                float *xb2,
                                size_t size_bytes);
int bn_transformer_gpu_read_activation_buf(const BnGPUBackend *gpu,
                                           int buf_idx,
                                           void *out,
                                           size_t size_bytes);
int bn_transformer_gpu_read_activation_buf_offset(const BnGPUBackend *gpu,
                                                  int buf_idx,
                                                  void *out,
                                                  size_t size_bytes,
                                                  size_t offset_bytes);
int bn_transformer_gpu_debug_compare_attention(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim);
int bn_transformer_gpu_debug_compare_gqa(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim);
int bn_transformer_gpu_debug_compare_qkv(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    uint32_t kv_cache_off,
    int dim,
    int q_dim,
    int kv_dim);
void bn_transformer_gpu_emit_context_dense_ffn(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    const BnTransformerGPUDenseFFNResources *res,
    int dim,
    uint32_t u_eps,
    void *next_norm,
    int skip_down,
    int *down_input_buf,
    int use_small_dense_exact_native,
    int use_small_dense_exact_native_down);
void bn_transformer_gpu_emit_context_attention(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res,
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
    uint32_t u_eps,
    int use_small_dense_exact_native);
void bn_transformer_gpu_emit_context_attention_gqa(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res,
    int pos,
    int q_dim,
    int head_size,
    int n_heads,
    int kv_dim,
    int rope_dims,
    int n_kv,
    size_t loff,
    uint32_t kv_cache_off,
    int has_moe);
void bn_transformer_gpu_emit_context_attention_finish(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res,
    int dim,
    int q_dim,
    int head_size,
    uint32_t u_eps,
    int use_small_dense_exact_native);
void bn_transformer_gpu_emit_context_qkv(BnTransformerGPUEmitContext *ctx,
                                         const BnConfig *c,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         const BnTransformerGPUQKVResources *res,
                                         int pos,
                                         int q_dim,
                                         int head_size,
                                         int n_heads,
                                         int kv_dim,
                                         int rope_dims,
                                         uint32_t kv_cache_off,
                                         uint32_t u_eps,
                                         int use_small_dense_exact_native);
void bn_transformer_gpu_emit_context_ssm(BnTransformerGPUEmitContext *ctx,
                                         const BnConfig *c,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         const BnTransformerGPUSSMResources *res,
                                         int dim,
                                         uint32_t u_eps);
void bn_transformer_gpu_emit_context_moe(BnTransformerGPUEmitContext *ctx,
                                         const BnGPUMoEResources *moe,
                                         const BnTransformerGPUMoESharedResources *shared,
                                         const BnLayerWeights *lw,
                                         int dim,
                                         uint32_t u_eps,
                                         void *next_norm,
                                         int exact_silu);

#endif // BN_TRANSFORMER_GPU_INTERNAL_H
