#ifndef BN_MODEL_ARCH_H
#define BN_MODEL_ARCH_H

#include "model_config.h"
#include "model_tokenizer_policy.h"
#include "model_anatomy.h"
#include "gguf.h"
#include "json.h"
#include "safetensors.h"
#include <stddef.h>

#define BN_MODEL_ARCH_POLICY_GGUF_ZERO_CENTERED_NORM_SHIFTED             (1u << 1)
#define BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE               (1u << 0)
#define BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK     (1u << 3)
#define BN_MODEL_ARCH_POLICY_MOE_REFERENCE_SILU                 (1u << 4)
#define BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER            (1u << 5)
#define BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY         (1u << 6)
#define BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT                    (1u << 7)
#define BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM                (1u << 8)
#define BN_MODEL_ARCH_POLICY_FFN_POST_NORM                      (1u << 9)
#define BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE                 (1u << 10)
#define BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY              (1u << 11)
#define BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK (1u << 12)
#define BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP_FALLBACK   (1u << 13)
#define BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION            (1u << 14)
#define BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT            (1u << 15)
#define BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH          (1u << 16)
#define BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION  (1u << 17)
#define BN_MODEL_ARCH_POLICY_MOE_SEPARATE_ROUTER_TOPK            (1u << 18)
#define BN_MODEL_ARCH_POLICY_PREFILL_REFERENCE_ACTIVATION       (1u << 19)
#define BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION           (1u << 20)
#define BN_MODEL_ARCH_POLICY_MOE_UNNORMALIZED_TOPK              (1u << 21)
#define BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS                (1u << 22)
#define BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION                (1u << 23)
#define BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT                (1u << 24)
#define BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS                  (1u << 25)
#define BN_MODEL_ARCH_POLICY_QUERY_SPARSE_ATTENTION              (1u << 26)
#define BN_MODEL_ARCH_POLICY_SSM_SIGMOID_GATE                    (1u << 27)
#define BN_MODEL_ARCH_POLICY_POSITIONAL_LAYER_EMBEDDING           (1u << 28)
#define BN_MODEL_ARCH_POLICY_SEPARATE_ROPE_NORM                  (1u << 29)

#define BN_MODEL_ARCH_POLICY_MOE_PREFILL_MATVEC_ROUTER           (1u << 30)
#define BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION_LAST_THIRD      (1u << 31)

struct BnModelAnatomy;

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
    BnSSMQKNormMode ssm_qk_norm;
    BnModelBF16LayoutKind bf16_layout;
    /* Non-NULL for topologies requiring a dedicated binder/executor rather
     * than the legacy GQA/GDN loader. Plans borrow tensor indices only. */
    int (*plan_anatomy)(BnGGUFFile *file, struct BnModelAnatomy *out);
} BnModelArchOps;

typedef enum {
    BN_MODEL_ARCH_RMSNORM_BACKEND_ORDER = 0,
    BN_MODEL_ARCH_RMSNORM_REFERENCE_ORDER,
} BnModelArchRMSNormMode;

/* Architecture-owned metadata/tensor-shape planning only. Does not read tensor
 * payloads or claim executor support. Rejects malformed/missing/ambiguous roles
 * atomically: out remains unchanged on failure. Auxiliary MTP blocks excluded. */
int bn_model_arch_plan_anatomy(BnGGUFFile *file, BnModelAnatomy *out);
/* Validate native main-expert dimensions/quant profile only, not forward
 * topology. Failure preserves out; auxiliary MTP experts are excluded. */
int bn_model_arch_native_expert_config(const BnJSON *document,
    BnModelNativeExpertConfig *out);
/* Exact supported V4.1 numerical policy: rejects missing/unsupported settings
 * atomically. Decimal constants compare exactly, independent of locale and
 * floating-point rounding mode. No tensor reads or executor admission. */
int bn_model_arch_native_numerics(const BnJSON *document,
    BnModelNativeNumerics *out);
/* Pure Engram hash layout query: source config order, prime buckets and seeded
 * NumPy-compatible PCG64 multipliers. Checks all table row sums; preserves out
 * on failure. Bucket primes must fit INT_MAX; array limits are explicit in
 * BnModelNativeEngramLayout. No token map,
 * payload reads or executor admission. */
int bn_model_arch_native_engram_layout(const BnJSON *document,int layer,
    BnModelNativeEngramLayout *out);
/* Source key-normalization profile and immutable map dimensions. */
int bn_model_arch_native_head(const BnModelNativeAnatomy *anatomy,
    const BnModelNativeNumerics *numerics,BnModelNativeHead *out);
int bn_model_arch_native_engram_math(const BnModelNativeAnatomy *anatomy,
    const BnModelNativeNumerics *numerics,int layer,BnModelNativeEngramMath *out);
int bn_model_arch_native_engram_map_spec(const BnModelNativeAnatomy *anatomy,
    const BnModelNativeNumerics *numerics,BnTokenizerMapSpec *out);
/* Stateless, allocation-free source n-gram hash over batch-major compressed
 * token history. Emit last out_len positions, token-major then ngram/head.
 * Layout must be an immutable resolver result. mapped_pad is the COMPRESSED
 * pad ID, not the raw layout pad_token. Structural/input/alias failure leaves
 * output unchanged. Empty out_len is a no-op with out_count=0; no data needed. */
int bn_model_arch_native_engram_hash_rows(const BnModelNativeEngramLayout *layout,uint32_t mapped_pad,
    const uint32_t *mapped_history,size_t batch,size_t history_len,size_t out_len,
    uint64_t *out,size_t out_count);
/* Resolve source scalar metadata under FE_TONEAREST; never changes the caller
 * rounding mode. Rejection leaves out unchanged. */
int bn_model_arch_native_rope_frequency(int rotary_dim,const BnModelNativeNumerics *numerics,
    int compressed,BnModelNativeRopeFrequency *out);
/* Main text storage shapes and cross-layer pool/selection ownership only.
 * Numeric execution settings remain in JSON and are not admitted here.
 * Metadata-only; rejects missing, malformed or unrecognized layer core tensors.
 * Failure leaves out unchanged; auxiliary MTP/vision namespaces excluded. */
int bn_model_arch_native_anatomy(const BnJSON *document,
    const BnSafetensorsIndex *index, BnModelNativeAnatomy *out);

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
typedef enum {
    BN_MODEL_EXPERT_COMPONENT_PACKED = 0,
    BN_MODEL_EXPERT_COMPONENT_PRE,
    BN_MODEL_EXPERT_COMPONENT_POST,
    BN_MODEL_EXPERT_COMPONENT_MARKER,
} BnModelExpertComponent;
/* Native routed-expert checkpoint names, currently deepseek_v41 EXL3 only.
 * Roles are MOE_GATE_EXPS, MOE_UP_EXPS and MOE_DOWN_EXPS. Main layers only;
 * no MTP prefix. Caller validates nonnegative layer/expert against config.
 * No executor admission or payload validation. Output unchanged on failure. */
int bn_model_arch_expert_component_name(const char *architecture,
    char *out, size_t out_size, int layer, int expert,
    BnModelTensorRole role, BnModelExpertComponent component);
int bn_model_arch_tensor_scale_name_for(const BnModelArchOps *ops,
                                        char *out,
                                        size_t out_size,
                                        int layer,
                                        BnModelTensorRole role);
/* Offset stored by GGUF conversion for zero-centered Gemma norms only.
 * Ordinary RMSNorm scales, including SSM norms, are not gamma weights. */
float bn_model_arch_zero_centered_norm_storage_offset(const BnConfig *c);
void bn_model_arch_apply_config(BnConfig *c, const BnModelArchOps *ops);
/* Architecture-planned shared admission/source dimensions only. Does not
 * fabricate legacy attention shapes. Failure leaves config unchanged. */
int bn_model_arch_apply_anatomy_config(BnConfig *config,
    const BnModelAnatomy *anatomy, int max_seq_len);
int bn_model_arch_requires_float_kquant_fallback(const BnConfig *c);
float bn_model_arch_attention_scale(const BnConfig *c, int head_size);
BnModelArchRMSNormMode bn_model_arch_rmsnorm_mode(const BnConfig *c);
int bn_model_arch_rmsnorm_uses_reference_order(const BnConfig *c);
float bn_model_arch_norm_epsilon(const BnConfig *c);
float bn_model_arch_final_logit_softcap(const BnConfig *c);
int bn_model_arch_attention_value_shares_key_config(const BnConfig *c);
int bn_model_arch_attention_qk_norm_stride(const BnConfig *c,
                                           int head_size);
int bn_model_arch_attention_uses_per_head_qk_norm(const BnConfig *c);
int bn_model_arch_uses_per_layer_embedding(const BnConfig *c);
int bn_model_arch_uses_attention_post_norm(const BnConfig *c);
int bn_model_arch_uses_ffn_post_norm(const BnConfig *c);
int bn_model_arch_uses_layer_output_scale(const BnConfig *c);
int bn_model_arch_has_ffn_gate(const BnConfig *c);
int bn_model_arch_config_activation(const BnConfig *c);
int bn_model_arch_uses_hybrid_layer_layout(const BnConfig *c);
int bn_model_arch_uses_hybrid_ssm(const BnConfig *c);
int bn_model_arch_uses_large_dense_hybrid_ssm(const BnConfig *c);
int bn_model_arch_uses_dense_attention_only(const BnConfig *c);
int bn_model_arch_uses_large_dense_shape(const BnConfig *c);
int bn_model_arch_separates_rope_norm(const BnConfig *c);
int bn_model_arch_divides_rope_freqs(const BnConfig *c, int layer);
int bn_model_arch_uses_swa_rope(const BnConfig *c, int layer_head_size);
int bn_model_arch_rope_dims_for_head(const BnConfig *c,
                                     int layer_head_size);
float bn_model_arch_rope_theta_for_head(const BnConfig *c,
                                        int layer_head_size);
float bn_model_arch_rope_base_theta(const BnConfig *c);
int bn_model_arch_rope_uses_base_frequency(const BnConfig *c,
                                           int layer_head_size);
void bn_model_arch_init_rope_frequencies(const BnConfig *c,
                                         float *freqs,
                                         int capacity_pairs);
void bn_model_arch_init_rope_frequencies_for_theta(float theta,
                                                   int rope_dims,
                                                   float *freqs,
                                                   int capacity_pairs);
void bn_model_arch_init_rope_angles_for_theta(float theta,
                                              int rope_dims,
                                              int position,
                                              float *angles,
                                              int capacity_pairs);
int bn_model_arch_per_layer_embedding_dim(const BnConfig *c);
int bn_model_arch_allows_small_dense_prefill_decode_fallback(
    const BnConfig *c);
int bn_model_arch_prefill_uses_decode_for_parity(const BnConfig *c);
int bn_model_arch_moe_requires_float_kquant_gateup_fallback(const BnConfig *c);
int bn_model_arch_moe_requires_reference_attention(const BnConfig *c);
int bn_model_arch_requires_reference_attention(const BnConfig *c);
int bn_model_arch_reference_attention_from_layer(const BnConfig *c);
int bn_model_arch_requires_reference_recurrent(const BnConfig *c);
int bn_model_arch_uses_hyper_connections(const BnConfig *c);
int bn_model_arch_uses_query_sparse_attention(const BnConfig *c);
int bn_model_arch_ssm_uses_sigmoid_gate(const BnConfig *c);
int bn_model_arch_uses_positional_layer_embedding(const BnConfig *c);
int bn_model_arch_moe_uses_scaled_router_input(const BnConfig *c);
int bn_model_arch_moe_uses_reference_router_accumulation(
    const BnConfig *c);
int bn_model_arch_attention_uses_padded_weighted_v_reduction(
    const BnConfig *c);
int bn_model_arch_moe_uses_separate_router_topk(const BnConfig *c);
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
int bn_model_arch_moe_total_experts(const BnConfig *c);
int bn_model_arch_moe_active_experts(const BnConfig *c);
int bn_model_arch_moe_expert_hidden_dim(const BnConfig *c);
int bn_model_arch_moe_route_shape_valid(const BnConfig *c);
int bn_model_arch_moe_normalizes_topk_route_weights(const BnConfig *c);
float bn_model_arch_moe_expert_weights_scale(const BnConfig *c);
int bn_model_arch_moe_uses_reference_silu(const BnConfig *c);
int bn_model_arch_config_has_shared_expert(const BnConfig *c);
int bn_model_arch_shared_expert_hidden_dim(const BnConfig *c);
int bn_model_arch_moe_prefill_uses_matvec_router(const BnConfig *c);
int bn_model_arch_moe_prefill_requires_matvec(const BnConfig *c);
int bn_model_arch_uses_all_active_two_expert_moe(const BnConfig *c,
                                                 int dim);
int bn_model_arch_loads_extra_metadata(const BnConfig *c);
/* Zero for full-attention layers, otherwise includes the current token. */
int bn_model_arch_attention_window(const BnConfig *c, int layer);
int bn_model_arch_loads_per_layer_input_weights(const BnConfig *c);
int bn_model_arch_layer_reuses_kv(const BnConfig *c, int layer);
int bn_model_arch_kv_reuse_layer(const BnConfig *c, int layer);
int bn_model_arch_loads_extra_ffn_post_norms(const BnConfig *c);
int bn_model_arch_loads_moe_aux_weights(const BnConfig *c);
int bn_model_arch_config_uses_full_rope_text_dims(const BnConfig *c);
int bn_model_arch_tokenizer_uses_metaspace(const char *tokenizer_model);
int bn_model_arch_uses_small_dense_shape(const BnConfig *c);
int bn_model_arch_dense_batch_prefill_shape_allowed(
    const BnConfig *c,
    int supports_large_dense_batch_prefill);
int bn_model_arch_dense_logits_argmax_shape_allowed(const BnConfig *c,
                                                    int logits_rows);
int bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(const BnConfig *c,
                                                       int logits_cols);
int bn_model_arch_prefill_uses_reference_activation(const BnConfig *c);
int bn_model_arch_ffn_uses_reference_activation(const BnConfig *c);
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
