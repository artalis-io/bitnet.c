#ifndef BN_MODEL_INTERNAL_H
#define BN_MODEL_INTERNAL_H

#include "model.h"
#include "platform.h"
#include <stddef.h>

struct BnModelRuntime {
    BnThreadPool *pool;
    int owns_pool;
    SHArena *weight_arena;
    BnTQState *tq_state;
    int owns_tq_state;
};

struct BnModelIO {
    BnMappedFile file;
    BnMoEIO moe_io;
};

struct BnModelBackendState {
    BnBackendModel *backend;
};

int bn_model_quant_type_supported(int type);
int bn_model_quant_uses_embedded_block_scale(int type);
int bn_model_quant_uses_embedded_tensor_scale(int type);
size_t bn_model_quant_embedded_tensor_scale_offset(int type,
                                                   int rows,
                                                   int cols);
int bn_model_quant_tied_logits_uses_quant_path(int type);
int bn_model_quant_logits_i8_cache_supported(int type);
void bn_model_quant_prepare_logits_i8_cache(const uint16_t *src,
                                            int8_t *dst,
                                            float *scales,
                                            int rows,
                                            int dim);
int bn_model_quant_uses_dense_float(int type);
int bn_model_quant_can_convert_dense_to_float(int type);
int bn_model_quant_convert_dense_to_float(int type,
                                          const void *src,
                                          float *dst,
                                          int n);
int bn_model_quant_dense_float_type(void);
int bn_model_quant_dequant_row(int type,
                               const void *data,
                               int row,
                               int n,
                               float *out);
int bn_model_activation_is_relu2(int activation);
int bn_model_activation_is_gelu(int activation);
int bn_model_activation_uses_silu_path(int activation);
int bn_model_gguf_uses_moe(BnGGUFFile *file);
int bn_model_gguf_context_length(BnGGUFFile *file);
int bn_model_config_attention_layer_count(const BnConfig *config);
int bn_model_config_ssm_layer_count(const BnConfig *config);
int bn_model_config_is_attention_layer(const BnConfig *config, int layer);
int bn_model_config_attention_layer_index(const BnConfig *config, int layer);
int bn_model_config_ssm_layer_index(const BnConfig *config, int layer);
int bn_model_config_uses_hybrid_layer_layout(const BnConfig *config);
int bn_model_config_uses_hybrid_ssm(const BnConfig *config);
int bn_model_config_uses_hybrid_moe(const BnConfig *config);
int bn_model_config_uses_moe(const BnConfig *config);
int bn_model_config_uses_all_active_two_expert_moe(const BnConfig *config,
                                                   int dim);
int bn_model_config_uses_two_expert_all_active_moe(const BnConfig *config);
int bn_model_config_uses_more_than_two_expert_moe(const BnConfig *config);
int bn_model_config_moe_total_experts(const BnConfig *config);
int bn_model_config_moe_active_experts(const BnConfig *config);
int bn_model_config_moe_expert_hidden_dim(const BnConfig *config);
int bn_model_config_moe_route_shape_valid(const BnConfig *config);
int bn_model_config_moe_normalizes_topk_route_weights(
    const BnConfig *config);
float bn_model_config_moe_expert_weights_scale(const BnConfig *config);
int bn_model_config_moe_uses_reference_silu(const BnConfig *config);
int bn_model_config_has_shared_expert(const BnConfig *config);
int bn_model_config_shared_expert_hidden_dim(const BnConfig *config);
int bn_model_config_moe_requires_float_kquant_gateup_fallback(
    const BnConfig *config);
int bn_model_config_moe_uses_scaled_router_input(const BnConfig *config);
int bn_model_config_moe_uses_dense_residual_branch(const BnConfig *config);
int bn_model_config_moe_prefill_requires_matvec(const BnConfig *config);
int bn_model_config_uses_small_dense_shape(const BnConfig *config);
int bn_model_config_uses_large_dense_shape(const BnConfig *config);
int bn_model_config_uses_large_gpu_graph_fallback_shape(
    const BnConfig *config);
int bn_model_config_uses_per_layer_embedding(const BnConfig *config);
int bn_model_config_uses_large_dense_hybrid_ssm(const BnConfig *config);
int bn_model_config_uses_reference_hybrid_ssm(const BnConfig *config);
int bn_model_config_uses_non_hybrid_moe(const BnConfig *config);
int bn_model_config_uses_dense_attention_only(const BnConfig *config);
int bn_model_config_uses_small_dense_native_quant_shape(
    const BnConfig *config);
int bn_model_config_requires_float_kquant_fallback(const BnConfig *config);
int bn_model_config_prefill_uses_decode_for_parity(const BnConfig *config);
int bn_model_config_rmsnorm_uses_reference_order(const BnConfig *config);
float bn_model_config_attention_scale(const BnConfig *config,
                                      int head_size);
int bn_model_config_attention_value_shares_key(const BnConfig *config);
int bn_model_config_uses_attention_post_norm(const BnConfig *config);
int bn_model_config_uses_ffn_post_norm(const BnConfig *config);
int bn_model_config_uses_layer_output_scale(const BnConfig *config);
int bn_model_config_per_layer_embedding_dim(const BnConfig *config);
int bn_model_config_has_ffn_gate(const BnConfig *config);
int bn_model_config_loads_extra_metadata(const BnConfig *config);
int bn_model_config_loads_per_layer_input_weights(const BnConfig *config);
int bn_model_config_layer_reuses_kv(const BnConfig *config, int layer);
int bn_model_config_kv_reuse_layer(const BnConfig *config, int layer);
int bn_model_config_loads_extra_ffn_post_norms(const BnConfig *config);
int bn_model_config_divides_rope_freqs(const BnConfig *config, int layer);
int bn_model_config_rope_dims_for_head(const BnConfig *config,
                                       int layer_head_size);
float bn_model_config_rope_theta_for_head(const BnConfig *config,
                                          int layer_head_size);
float bn_model_config_rope_base_theta(const BnConfig *config);
int bn_model_config_rope_uses_base_frequency(const BnConfig *config,
                                             int layer_head_size);
void bn_model_config_init_rope_frequencies(const BnConfig *config,
                                           float *freqs,
                                           int capacity_pairs);
float bn_model_config_final_logit_softcap(const BnConfig *config);
int bn_model_config_prefill_uses_reference_activation(const BnConfig *config);
int bn_model_config_ffn_uses_reference_activation(const BnConfig *config);
int bn_model_config_dense_batch_prefill_shape_allowed(
    const BnConfig *config,
    int supports_large_dense_batch_prefill);
int bn_model_config_dense_logits_argmax_shape_allowed(
    const BnConfig *config,
    int logits_rows);
int bn_model_config_moe_logits_mmvq_argmax_shape_allowed(
    const BnConfig *config,
    int logits_cols);
int bn_model_config_allows_small_dense_native_quant(const BnConfig *config);
int bn_model_config_small_dense_native_quant_to_layer(
    const BnConfig *config);
int bn_model_config_allows_small_dense_prefill_decode_fallback(
    const BnConfig *config);
int bn_model_config_small_dense_prefill_min_tokens(const BnConfig *config);
int bn_model_config_allows_small_dense_native_logit_refine(
    const BnConfig *config);
int bn_model_config_moe_prefers_reference_gpu_attention(const BnConfig *config);

#endif // BN_MODEL_INTERNAL_H
