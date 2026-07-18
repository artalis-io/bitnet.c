#ifndef BN_TRANSFORMER_PREFILL_INTERNAL_H
#define BN_TRANSFORMER_PREFILL_INTERNAL_H

#include "gpu_backend.h"
#include "model_config.h"
#include "quant.h"
#include "threadpool.h"
#include <stdint.h>

typedef struct {
    float *hb;
    const float *hb2;
    int hidden_dim;
    int act_type;
    int fast_approx;
} BnPrefillFFNActCtx;

typedef struct {
    const char *name;
    void (*rmsnorm)(float *out, const float *x, const float *w,
                    int size, float eps);
    bn_tp_fn ffn_activation;
    bn_tp_fn ssm_conv_silu;
    bn_tp_fn ssm_l2norm;
    bn_tp_fn ssm_delta;
    bn_tp_fn ssm_gate;
    int (*prepare_preq8k)(int8_t *xq, float *xd, int16_t *xbs,
                          int n_bpr, const float *x,
                          int dim, int n_tokens);
    int supports_preq8k;
} BnPrefillCPUOps;

typedef struct {
    int uses_moe;
} BnTransformerPrefillLayerKindPolicy;

typedef struct {
    int enabled;
} BnTransformerPrefillSharedAll2DecodeFallbackPolicy;

typedef struct {
    int uses_hybrid_layer_layout;
    int uses_hybrid_ssm;
    int uses_large_dense_hybrid_ssm;
} BnTransformerPrefillSequencePolicy;

typedef struct {
    int enabled;
} BnTransformerPrefillDenseLayerBatchPolicy;

typedef struct {
    int enabled;
} BnTransformerPrefillDenseLayerChainPolicy;

typedef struct {
    int eligible;
    int fuses_input_norm;
} BnTransformerPrefillRawAttentionPolicy;

typedef enum {
    BN_TRANSFORMER_PREFILL_RAW_ATTENTION_PLAIN = 0,
    BN_TRANSFORMER_PREFILL_RAW_ATTENTION_NORM_RESID
} BnTransformerPrefillRawAttentionCallKind;

typedef struct {
    BnTransformerPrefillRawAttentionCallKind preferred_kind;
} BnTransformerPrefillRawAttentionCallPolicy;

typedef struct {
    int eligible;
    int fuses_output_projection;
} BnTransformerPrefillAttentionBatchPolicy;

typedef enum {
    BN_TRANSFORMER_PREFILL_ATTENTION_BATCH_PLAIN = 0,
    BN_TRANSFORMER_PREFILL_ATTENTION_BATCH_WO
} BnTransformerPrefillAttentionBatchCallKind;

typedef struct {
    BnTransformerPrefillAttentionBatchCallKind preferred_kind;
} BnTransformerPrefillAttentionBatchCallPolicy;

typedef struct {
    int eligible;
    int fuses_norm_residual;
} BnTransformerPrefillFFNBatchPolicy;

typedef enum {
    BN_TRANSFORMER_PREFILL_FFN_BATCH_PLAIN = 0,
    BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM,
    BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM_RESID
} BnTransformerPrefillFFNBatchCallKind;

typedef struct {
    BnTransformerPrefillFFNBatchCallKind kind;
} BnTransformerPrefillFFNBatchCallPolicy;

const BnPrefillCPUOps *bn_transformer_prefill_cpu_ops(void);
int bn_transformer_prefill_profile_enabled(void);
int bn_transformer_prefill_hybrid_batch_allowed(void);
int bn_transformer_prefill_force_token_attention_enabled(void);
BnTransformerPrefillLayerKindPolicy
bn_transformer_prefill_layer_kind_policy(const void *moe_router_weight);
BnTransformerPrefillSharedAll2DecodeFallbackPolicy
bn_transformer_prefill_shared_all2_decode_fallback_policy(
    const BnConfig *c,
    int gpu_available);
BnTransformerPrefillSequencePolicy
bn_transformer_prefill_sequence_policy(const BnConfig *c);
BnTransformerPrefillDenseLayerBatchPolicy
bn_transformer_prefill_dense_layer_batch_policy(
    int gpu_available,
    int tq_state_available,
    int dense_chain_enabled,
    int n_tokens,
    int min_tokens,
    int pos0,
    float layer_rope_theta,
    float config_rope_theta,
    BnTransformerPrefillLayerKindPolicy layer_kind,
    int has_ffn_gate,
    int has_ffn_up,
    int has_q_bias,
    int has_k_bias,
    int has_v_bias,
    int has_attn_sub_norm,
    int has_ffn_sub_norm,
    int has_layer_output_scale,
    int uses_post_norm,
    int has_attn_post_norm,
    int has_ffn_post_norm);
BnTransformerPrefillDenseLayerChainPolicy
bn_transformer_prefill_dense_layer_chain_policy(
    int gpu_available,
    int dense_layer_hook_available,
    int tq_state_available,
    int n_tokens,
    int min_tokens,
    float layer_rope_theta,
    float config_rope_theta,
    int is_attn,
    BnTransformerPrefillLayerKindPolicy layer_kind,
    int has_ffn_gate,
    int has_ffn_up,
    int has_attn_sub_norm,
    int has_ffn_sub_norm,
    int has_layer_output_scale,
    int uses_post_norm,
    int has_attn_post_norm,
    int has_ffn_post_norm);
BnTransformerPrefillRawAttentionPolicy
bn_transformer_prefill_raw_attention_policy(
    int gpu_available,
    int raw_attention_hook_available,
    int norm_resid_hook_available,
    int attn_norm_buffer_available,
    int tq_state_available,
    int q_gated,
    int pos0,
    int n_tokens,
    int min_tokens,
    float layer_rope_theta,
    float config_rope_theta,
    int has_q_bias,
    int has_k_bias,
    int has_v_bias,
    int has_attn_sub_norm,
    int uses_post_norm,
    int has_attn_post_norm);
BnTransformerPrefillRawAttentionCallPolicy
bn_transformer_prefill_raw_attention_call_policy(
    BnTransformerPrefillRawAttentionPolicy policy);
BnTransformerPrefillAttentionBatchPolicy
bn_transformer_prefill_attention_batch_policy(
    int raw_attention_already_used,
    int gpu_available,
    int attention_hook_available,
    int attention_wo_hook_available,
    int attention_feature_enabled,
    int wo_buffer_available,
    int n_tokens,
    int min_tokens,
    int has_attn_sub_norm,
    int uses_post_norm,
    int has_attn_post_norm);
BnTransformerPrefillAttentionBatchCallPolicy
bn_transformer_prefill_attention_batch_call_policy(
    BnTransformerPrefillAttentionBatchPolicy policy);
BnTransformerPrefillFFNBatchPolicy
bn_transformer_prefill_ffn_batch_policy(
    int has_ffn_gate,
    int tokens_allowed,
    int ffn_batch_norm_resid_hook_available,
    int ffn_norm_buffer_available,
    int n_tokens,
    int min_tokens,
    int uses_hybrid_layer_layout,
    int has_ffn_sub_norm,
    int uses_post_norm,
    int has_ffn_post_norm);
BnTransformerPrefillFFNBatchCallPolicy
bn_transformer_prefill_ffn_batch_call_policy(
    int norm_buffer_available,
    int add_residual,
    int ffn_batch_norm_hook_available,
    int ffn_batch_norm_resid_hook_available);
int bn_transformer_prefill_can_preq8k_type(const BnPrefillCPUOps *ops,
                                           int tensor_type);
int bn_transformer_prefill_can_preq8k_pair(const BnPrefillCPUOps *ops,
                                           int left_type,
                                           int right_type);
int bn_transformer_prefill_can_preq8k_triple(const BnPrefillCPUOps *ops,
                                             int first_type,
                                             int second_type,
                                             int third_type);
int bn_transformer_prefill_route_preq8k_type_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int force_float_kquant,
    int dim,
    int tensor_type);
int bn_transformer_prefill_route_preq8k_pair_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int force_float_kquant,
    int dim,
    int left_type,
    int right_type);
int bn_transformer_prefill_route_preq8k_triple_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int force_float_kquant,
    int dim,
    int first_type,
    int second_type,
    int third_type);
bn_tp_fn bn_transformer_prefill_ssm_conv_silu_op(
    const BnConfig *c,
    const BnPrefillCPUOps *ops);
bn_tp_fn bn_transformer_prefill_ssm_l2norm_op(
    const BnConfig *c,
    const BnPrefillCPUOps *ops);
bn_tp_fn bn_transformer_prefill_ssm_delta_op(
    const BnConfig *c,
    const BnPrefillCPUOps *ops);
bn_tp_fn bn_transformer_prefill_ssm_gate_op(
    const BnConfig *c,
    const BnPrefillCPUOps *ops);
int bn_transformer_prefill_stacked_pair_same_format(int left_type,
                                                    int right_type);
int bn_transformer_prefill_qk_stack_compatible(const BnQWeight *q,
                                               const BnQWeight *k,
                                               int q_stride,
                                               int dim);
int bn_transformer_prefill_qkv_stack_batch_compatible(const BnQWeight *q,
                                                      const BnQWeight *k,
                                                      const BnQWeight *v,
                                                      int q_stride,
                                                      int dim);
int bn_transformer_prefill_uses_float_kquant_fallback(int tensor_type);
void bn_transformer_prefill_quant_matmul_gpu_buffer(float *out,
                                                    const BnQWeight *W,
                                                    void *W_buf,
                                                    const float *X,
                                                    int n_tokens,
                                                    int8_t *x_q_buf,
                                                    BnThreadPool *pool,
                                                    BnGPUBackend *gpu);
void bn_transformer_prefill_quant_matmul_batch_gpu_buffers(
    const BnMatvecTask *tasks,
    const void **buffers,
    int n_tasks,
    const float *X,
    int n_tokens,
    int cols,
    int8_t *x_q_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu);

#endif // BN_TRANSFORMER_PREFILL_INTERNAL_H
