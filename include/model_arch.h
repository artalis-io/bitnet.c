#ifndef BN_MODEL_ARCH_H
#define BN_MODEL_ARCH_H

#include "model_config.h"
#include "gguf.h"
#include <stddef.h>

#define BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE               (1u << 0)
#define BN_MODEL_ARCH_POLICY_LARGE_GPU_GRAPH_FALLBACK           (1u << 1)
#define BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM               (1u << 2)
#define BN_MODEL_ARCH_POLICY_SCALAR_HYBRID_SSM_CPU \
    BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM
#define BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK     (1u << 3)
#define BN_MODEL_ARCH_POLICY_CPU_FLOAT_KQUANT \
    BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK
#define BN_MODEL_ARCH_POLICY_MOE_EXACT_SILU                     (1u << 4)
#define BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER            (1u << 5)
#define BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY         (1u << 6)
#define BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT                    (1u << 7)
#define BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM                (1u << 8)
#define BN_MODEL_ARCH_POLICY_FFN_POST_NORM                      (1u << 9)
#define BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE                 (1u << 10)
#define BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY              (1u << 11)
#define BN_MODEL_ARCH_POLICY_CPU_PREFILL_DECODE_PARITY \
    BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY
#define BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK (1u << 12)
#define BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP            (1u << 13)
#define BN_MODEL_ARCH_POLICY_MOE_EXACT_GPU_ATTENTION            (1u << 14)
#define BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT            (1u << 15)
#define BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH          (1u << 16)
#define BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE           (1u << 17)
#define BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE    (1u << 18)
#define BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION           (1u << 19)
#define BN_MODEL_ARCH_POLICY_EXACT_SCALAR_FFN_ACTIVATION        (1u << 20)
#define BN_MODEL_ARCH_POLICY_MOE_UNNORMALIZED_TOPK              (1u << 21)
#define BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS                (1u << 22)

typedef struct {
    const char *name;
    uint32_t policy_flags;
    uint32_t moe_policy_flags;
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
    BN_MODEL_ARCH_RMSNORM_REFERENCE_SCALAR_ORDER,
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
int bn_model_arch_activation(const char *arch);
int bn_model_arch_activation_is_silu(int act_type);
int bn_model_arch_activation_is_relu2(int act_type);
int bn_model_arch_activation_is_gelu(int act_type);
int bn_model_arch_activation_uses_silu_path(int act_type);
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
void bn_model_arch_apply_config(BnConfig *c, const BnModelArchOps *ops);
int bn_model_arch_requires_large_gpu_graph_fallback(const BnConfig *c);
int bn_model_arch_requires_float_kquant_fallback(const BnConfig *c);
int bn_model_arch_cpu_force_float_kquant(const BnConfig *c);
float bn_model_arch_attention_scale(const BnConfig *c, int head_size);
BnModelArchRMSNormMode bn_model_arch_rmsnorm_mode(const BnConfig *c);
int bn_model_arch_rmsnorm_requires_reference_scalar_order(const BnConfig *c);
int bn_model_arch_attention_value_shares_key_config(const BnConfig *c);
int bn_model_arch_uses_per_layer_embedding(const BnConfig *c);
int bn_model_arch_uses_attention_post_norm(const BnConfig *c);
int bn_model_arch_uses_ffn_post_norm(const BnConfig *c);
int bn_model_arch_uses_layer_output_scale(const BnConfig *c);
int bn_model_arch_uses_reference_hybrid_ssm(const BnConfig *c);
int bn_model_arch_uses_scalar_hybrid_ssm_cpu(const BnConfig *c);
int bn_model_arch_uses_hybrid_layer_layout(const BnConfig *c);
int bn_model_arch_uses_hybrid_ssm(const BnConfig *c);
int bn_model_arch_uses_large_dense_hybrid_ssm(const BnConfig *c);
int bn_model_arch_uses_dense_attention_only(const BnConfig *c);
int bn_model_arch_uses_large_dense_shape(const BnConfig *c);
int bn_model_arch_uses_large_gpu_graph_fallback_shape(const BnConfig *c);
int bn_model_arch_divides_rope_freqs(const BnConfig *c, int layer);
int bn_model_arch_per_layer_embedding_dim(const BnConfig *c);
int bn_model_arch_allows_small_dense_prefill_decode_fallback(
    const BnConfig *c);
int bn_model_arch_prefill_uses_decode_for_parity(const BnConfig *c);
int bn_model_arch_cpu_prefill_uses_decode_for_parity(const BnConfig *c);
int bn_model_arch_moe_forces_float_kquant_gateup(const BnConfig *c);
int bn_model_arch_moe_prefers_exact_gpu_attention(const BnConfig *c);
int bn_model_arch_moe_uses_scaled_router_input(const BnConfig *c);
int bn_model_arch_moe_uses_dense_residual_branch(const BnConfig *c);
int bn_model_arch_uses_moe(const BnConfig *c);
int bn_model_arch_gguf_u32(BnGGUFFile *f, const char *suffix);
int bn_model_arch_gguf_u32_or_i32_array(BnGGUFFile *f,
                                         const char *suffix,
                                         int idx);
float bn_model_arch_gguf_f32(BnGGUFFile *f, const char *suffix);
uint64_t bn_model_arch_gguf_arr_n(BnGGUFFile *f, const char *suffix);
const void *bn_model_arch_gguf_arr_data(BnGGUFFile *f, const char *suffix);
int bn_model_arch_gguf_bool_array(BnGGUFFile *f,
                                  const char *suffix,
                                  int idx);
int bn_model_arch_gguf_uses_moe(BnGGUFFile *f);
int bn_model_arch_uses_non_hybrid_moe(const BnConfig *c);
int bn_model_arch_uses_hybrid_moe(const BnConfig *c);
int bn_model_arch_uses_two_expert_all_active_moe(const BnConfig *c);
int bn_model_arch_uses_more_than_two_expert_moe(const BnConfig *c);
int bn_model_arch_moe_prefill_forces_matvec(const BnConfig *c);
int bn_model_arch_uses_all_active_two_expert_moe(const BnConfig *c,
                                                 int dim);
int bn_model_arch_loads_extra_metadata(const BnConfig *c);
int bn_model_arch_loads_per_layer_input_weights(const BnConfig *c);
int bn_model_arch_layer_reuses_kv(const BnConfig *c, int layer);
int bn_model_arch_kv_reuse_layer(const BnConfig *c, int layer);
int bn_model_arch_loads_extra_ffn_post_norms(const BnConfig *c);
int bn_model_arch_loads_moe_aux_weights(const BnConfig *c);
int bn_model_arch_config_uses_full_rope_text_dims(const BnConfig *c);
int bn_model_arch_tokenizer_uses_metaspace(const char *tokenizer_model);
int bn_model_arch_allows_small_dense_exact_native(const BnConfig *c);
int bn_model_arch_small_dense_exact_native_to_layer(const BnConfig *c);
int bn_model_arch_allows_small_dense_native_logit_refine(const BnConfig *c);
int bn_model_arch_small_dense_prefill_min_tokens(const BnConfig *c);
int bn_model_arch_uses_small_dense_shape(const BnConfig *c);
int bn_model_arch_uses_small_dense_native_quant_shape(const BnConfig *c);
int bn_model_arch_dense_batch_prefill_shape_allowed(
    const BnConfig *c,
    int supports_large_dense_batch_prefill);
int bn_model_arch_dense_logits_argmax_shape_allowed(const BnConfig *c,
                                                    int logits_rows);
int bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(const BnConfig *c,
                                                       int logits_cols);
int bn_model_arch_prefill_uses_exact_activation(const BnConfig *c);
int bn_model_arch_ffn_uses_exact_scalar_activation(const BnConfig *c);
int bn_model_arch_rope_text_dims(int rope_dim_count,
                                 const int32_t *sections,
                                 uint64_t n_sections);
int bn_model_arch_is_ssm_layer(const BnConfig *c, int layer);
int bn_model_arch_is_attention_layer(const BnConfig *c, int layer);
int bn_model_arch_attention_layer_index(const BnConfig *c, int layer);
int bn_model_arch_ssm_layer_index(const BnConfig *c, int layer);
int bn_model_arch_attention_layer_count(const BnConfig *c);
int bn_model_arch_ssm_layer_count(const BnConfig *c);
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
