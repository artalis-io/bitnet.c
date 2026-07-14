#ifndef BN_MODEL_ARCH_H
#define BN_MODEL_ARCH_H

#include "model_config.h"
#include "gguf.h"
#include <stddef.h>

typedef struct {
    const char *name;
    uint32_t flags;
    int (*is_match)(const char *arch);
    const char *(*prefix)(const char *arch);
    int (*activation)(const char *arch);
    int (*attention_value_shares_key)(const char *arch);
    int (*is_ssm_layer)(const BnConfig *c, int layer);
    int (*tensor_name)(char *out, size_t out_size, int layer, int role);
    void (*apply_shapes)(BnConfig *c, int max_head_size, int max_kv_dim);
} BnModelArchOps;

typedef enum {
    BN_MODEL_ARCH_RMSNORM_BACKEND_ORDER = 0,
    BN_MODEL_ARCH_RMSNORM_LLAMA_SCALAR_ORDER,
} BnModelArchRMSNormMode;

typedef enum {
    BN_MODEL_TENSOR_ATTN_NORM = 0,
    BN_MODEL_TENSOR_ATTN_Q,
    BN_MODEL_TENSOR_ATTN_K,
    BN_MODEL_TENSOR_ATTN_V,
    BN_MODEL_TENSOR_ATTN_OUTPUT,
    BN_MODEL_TENSOR_ATTN_Q_BIAS,
    BN_MODEL_TENSOR_ATTN_K_BIAS,
    BN_MODEL_TENSOR_ATTN_V_BIAS,
    BN_MODEL_TENSOR_ATTN_Q_NORM,
    BN_MODEL_TENSOR_ATTN_K_NORM,
    BN_MODEL_TENSOR_ATTN_SUB_NORM,
    BN_MODEL_TENSOR_ATTN_POST_NORM,
    BN_MODEL_TENSOR_SSM_QKV,
    BN_MODEL_TENSOR_SSM_GATE,
    BN_MODEL_TENSOR_SSM_A,
    BN_MODEL_TENSOR_SSM_ALPHA,
    BN_MODEL_TENSOR_SSM_BETA,
    BN_MODEL_TENSOR_SSM_CONV1D,
    BN_MODEL_TENSOR_SSM_DT_BIAS,
    BN_MODEL_TENSOR_SSM_NORM,
    BN_MODEL_TENSOR_SSM_OUT,
    BN_MODEL_TENSOR_FFN_NORM,
    BN_MODEL_TENSOR_FFN_POST_ATTN_NORM,
    BN_MODEL_TENSOR_FFN_SUB_NORM,
    BN_MODEL_TENSOR_FFN_POST_NORM,
    BN_MODEL_TENSOR_FFN_GATE,
    BN_MODEL_TENSOR_FFN_UP,
    BN_MODEL_TENSOR_FFN_DOWN,
    BN_MODEL_TENSOR_MOE_ROUTER,
    BN_MODEL_TENSOR_MOE_GATE_EXPS,
    BN_MODEL_TENSOR_MOE_UP_EXPS,
    BN_MODEL_TENSOR_MOE_GATE_UP_EXPS,
    BN_MODEL_TENSOR_MOE_DOWN_EXPS,
    BN_MODEL_TENSOR_SHARED_FFN_GATE,
    BN_MODEL_TENSOR_SHARED_FFN_UP,
    BN_MODEL_TENSOR_SHARED_FFN_DOWN,
    BN_MODEL_TENSOR_SHARED_FFN_ROUTER,
    BN_MODEL_TENSOR_LAYER_OUTPUT_SCALE,
} BnModelTensorRole;

const char *bn_model_arch_prefix(const char *arch);
int bn_model_arch_is_gemma4(const char *arch);
int bn_model_arch_activation(const char *arch);
int bn_model_arch_attention_value_shares_key(const char *arch);
const char *bn_model_arch_tensor_suffix(BnModelTensorRole role);
int bn_model_arch_default_tensor_name(char *out, size_t out_size,
                                      int layer, int role);
int bn_model_arch_tensor_name_for(const BnModelArchOps *ops,
                                  char *out,
                                  size_t out_size,
                                  int layer,
                                  BnModelTensorRole role);
int bn_model_arch_tensor_scale_name_for(const BnModelArchOps *ops,
                                        char *out,
                                        size_t out_size,
                                        int layer,
                                        BnModelTensorRole role);
int bn_model_arch_requires_large_gpu_graph_fallback(const BnConfig *c);
int bn_model_arch_cpu_force_float_kquant(const BnConfig *c);
float bn_model_arch_attention_scale(const BnConfig *c, int head_size);
BnModelArchRMSNormMode bn_model_arch_rmsnorm_mode(const BnConfig *c);
int bn_model_arch_attention_value_shares_key_config(const BnConfig *c);
int bn_model_arch_uses_per_layer_embedding(const BnConfig *c);
int bn_model_arch_uses_attention_post_norm(const BnConfig *c);
int bn_model_arch_uses_ffn_post_norm(const BnConfig *c);
int bn_model_arch_uses_layer_output_scale(const BnConfig *c);
int bn_model_arch_uses_scalar_hybrid_ssm_cpu(const BnConfig *c);
int bn_model_arch_allows_small_cuda_q8_logit_refine(const BnConfig *c);
int bn_model_arch_rope_text_dims(int rope_dim_count,
                                 const int32_t *sections,
                                 uint64_t n_sections);
int bn_model_arch_is_ssm_layer(const BnConfig *c, int layer);
int bn_model_arch_infer_moe_hidden(BnGGUFFile *f,
                                   const BnModelArchOps *ops);
int bn_model_arch_has_shared_expert(BnGGUFFile *f,
                                    const BnModelArchOps *ops);
int bn_model_arch_infer_shared_expert_hidden(BnGGUFFile *f,
                                             const BnModelArchOps *ops);
void bn_model_arch_load_moe_config(BnConfig *c,
                                   BnGGUFFile *f,
                                   const BnModelArchOps *ops,
                                   const char *prefix);
const BnModelArchOps *bn_model_arch_registry(size_t *count);
const BnModelArchOps *bn_model_arch_ops_for(const char *arch);

#endif // BN_MODEL_ARCH_H
