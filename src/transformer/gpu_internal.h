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
int bn_transformer_gpu_logits_needs_cpu_fallback(
    const BnGPUBackend *gpu,
    const BnTransformerGPULogitResources *logits);
int bn_transformer_gpu_cuda_all2_q4q6_moe_model(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_cuda_all2_q4q6_moe_layer(
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim);
int bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int q4_q8_from_layer);
int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_ffn_down_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_cuda_small_dense_prefill_decode_fallback_requested(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_cuda_large_hybrid_prefill_decode_fallback_default(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_batch_prefill_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_cuda_large_hybrid_prefill_chain_disabled_default(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_cuda_prefill_direct_kv_allowed(
    const BnConfig *c,
    const BnWeights *w,
    const BnGPUBackend *gpu,
    int pos0,
    int n_tokens);
int bn_transformer_gpu_cuda_prefill_attention_min_tokens(void);
int bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu);
int bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu);
int bn_transformer_gpu_cuda_small_dense_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int tensor_type);
int bn_transformer_gpu_cuda_all2_q4q6_moe_q6_logits_refine_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_q6_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q6_refine_default);
int bn_transformer_gpu_q6_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_q6_logits,
    int q6_refine_default);
int bn_transformer_gpu_q6_logits_refine_top(int q6_refine_default);
int bn_transformer_gpu_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q8_refine_default);
int bn_transformer_gpu_q8_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_q8_logits);
int bn_transformer_gpu_q8_logits_refine_top(int q8_refine_default);
int bn_transformer_gpu_matvec_argmax_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPULogitResources *logits,
    int want_argmax,
    int need_logits,
    int gpu_logits_need_cpu);
int bn_transformer_gpu_cuda_moe_decode_cacheable(
    const BnConfig *c,
    const BnWeights *w,
    const BnBackendModel *backend);
int bn_transformer_gpu_cuda_decode_cacheable(
    const BnGPUBackend *gpu,
    int emit_logits,
    int want_argmax,
    int gpu_logits_need_cpu,
    int has_moe,
    int cacheable_resident_moe,
    int q6_logits_refine_captures_xb,
    int q8_logits_refine_captures_xb,
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
int bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_moe_safe_default(
    const BnConfig *c,
    const BnWeights *w);
int bn_transformer_gpu_cuda_moe_exact_attention_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c);
int bn_transformer_gpu_ssm_cpu_fallback_required(
    const BnGPUBackend *gpu);
int bn_transformer_gpu_cuda_large_hybrid_argmax_blocked(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax);
int bn_transformer_gpu_flash_attention_enabled(
    const BnGPUBackend *gpu,
    int config_flash_attn,
    int has_moe,
    int n_kv);
int bn_transformer_gpu_moe_routed_q4(const BnMoEExpertMap *map);
int bn_transformer_gpu_moe_routed_q4_down(const BnMoEExpertMap *map,
                                          int allow_q4_down);
int bn_transformer_gpu_moe_routed_q8(const BnMoEExpertMap *map);
int bn_transformer_gpu_cuda_moe_route_topk_enabled(
    void *moe_router,
    int all2_q4q6_moe,
    int all2_q4q6_moe_gpu_route_layer_selected);
int bn_transformer_gpu_cuda_moe_cpu_route_resident_ffn_enabled(
    int all2_q4q6_moe,
    int gpu_route_topk,
    int moe_routed_q8,
    int n_experts);
int bn_transformer_gpu_cuda_moe_routed_ffn_enabled(
    int gpu_route_topk,
    int cpu_route_resident_ffn,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all,
    const BnMoEExpertMap *map,
    int moe_hidden,
    int dim);
int bn_transformer_gpu_cuda_all2_moe_direct_route_enabled(
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all);
int bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_selected(
    int layer,
    int route_from_layer,
    int route_to_layer);
int bn_transformer_gpu_cuda_all2_q4q6_moe_exact_gpu_route_enabled(
    int all2_q4q6_moe,
    int route_layer_selected);
void *bn_transformer_gpu_cuda_all2_q4q6_moe_router(
    const BnConfig *c,
    void *moe_router,
    void *router_diff,
    int route_layer_selected,
    int exact_gpu_route);
int bn_transformer_gpu_all2_q4_moe_requires_opt_in(
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down);
int bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(int n_experts);
int bn_transformer_gpu_cuda_moe_gateup_split_enabled(
    const BnGPUBackend *gpu,
    int can_split);
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
    int use_q4_q8,
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
                                           float *out,
                                           size_t size_bytes);
int bn_transformer_gpu_read_activation_buf_offset(const BnGPUBackend *gpu,
                                                  int buf_idx,
                                                  float *out,
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
    int use_q4_q8,
    int use_q4_q8_down);
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
    int use_q4_q8);
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
    int use_q4_q8);
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
                                         int use_q4_q8);
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
