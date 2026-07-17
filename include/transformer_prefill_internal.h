#ifndef BN_TRANSFORMER_PREFILL_INTERNAL_H
#define BN_TRANSFORMER_PREFILL_INTERNAL_H

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

const BnPrefillCPUOps *bn_transformer_prefill_cpu_ops(void);
int bn_transformer_prefill_profile_enabled(void);
int bn_transformer_prefill_hybrid_batch_allowed(void);
int bn_transformer_prefill_force_token_attention_enabled(void);
int bn_transformer_prefill_can_preq8k_type(const BnPrefillCPUOps *ops,
                                           int tensor_type);
int bn_transformer_prefill_can_preq8k_pair(const BnPrefillCPUOps *ops,
                                           int left_type,
                                           int right_type);
int bn_transformer_prefill_can_preq8k_triple(const BnPrefillCPUOps *ops,
                                             int first_type,
                                             int second_type,
                                             int third_type);
int bn_transformer_prefill_stacked_pair_same_format(int left_type,
                                                    int right_type);

#endif // BN_TRANSFORMER_PREFILL_INTERNAL_H
