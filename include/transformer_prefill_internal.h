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

const BnPrefillCPUOps *bn_transformer_prefill_cpu_ops(void);
int bn_transformer_prefill_profile_enabled(void);
int bn_transformer_prefill_hybrid_batch_allowed(void);
int bn_transformer_prefill_force_token_attention_enabled(void);
BnTransformerPrefillLayerKindPolicy
bn_transformer_prefill_layer_kind_policy(const void *moe_router_weight);
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
