#include "model_internal.h"
#include "model_arch.h"
#include "quant.h"

int bn_model_dequant_qweight_row(const BnQWeight *weight,
                                 int row,
                                 int n,
                                 float *out) {
    if (!weight)
        return -1;
    return bn_quant_dequant_row(weight->type, weight->data, row, n, out);
}

int bn_model_activation_is_relu2(int activation) {
    return bn_model_arch_activation_is_relu2(activation);
}

int bn_model_activation_is_gelu(int activation) {
    return bn_model_arch_activation_is_gelu(activation);
}

int bn_model_activation_uses_silu_path(int activation) {
    return bn_model_arch_activation_uses_silu_path(activation);
}

int bn_model_gguf_uses_moe(BnGGUFFile *file) {
    return bn_model_arch_gguf_uses_moe(file);
}

int bn_model_gguf_context_length(BnGGUFFile *file) {
    return bn_model_arch_gguf_u32(file, "context_length");
}

int bn_model_load_policy_uses_moe(const BnConfig *config) {
    return bn_model_arch_uses_moe(config);
}

int bn_model_load_policy_loads_extra_metadata(const BnConfig *config) {
    return bn_model_arch_loads_extra_metadata(config);
}

int bn_model_load_policy_uses_hybrid_layer_layout(
    const BnConfig *config) {
    return bn_model_arch_uses_hybrid_layer_layout(config);
}

int bn_model_load_policy_moe_total_experts(const BnConfig *config) {
    return bn_model_arch_moe_total_experts(config);
}

int bn_model_load_policy_moe_active_experts(const BnConfig *config) {
    return bn_model_arch_moe_active_experts(config);
}

int bn_model_load_policy_moe_expert_hidden_dim(const BnConfig *config) {
    return bn_model_arch_moe_expert_hidden_dim(config);
}

int bn_model_load_policy_moe_route_shape_valid(const BnConfig *config) {
    return bn_model_arch_moe_route_shape_valid(config);
}

int bn_model_load_policy_loads_per_layer_input_weights(
    const BnConfig *config) {
    return bn_model_arch_loads_per_layer_input_weights(config);
}

int bn_model_load_policy_layer_reuses_kv(const BnConfig *config,
                                         int layer) {
    return bn_model_arch_layer_reuses_kv(config, layer);
}

int bn_model_load_policy_kv_reuse_layer(const BnConfig *config,
                                        int layer) {
    return bn_model_arch_kv_reuse_layer(config, layer);
}

int bn_model_load_policy_uses_ffn_post_norm(const BnConfig *config) {
    return bn_model_arch_uses_ffn_post_norm(config);
}

int bn_model_load_policy_loads_extra_ffn_post_norms(
    const BnConfig *config) {
    return bn_model_arch_loads_extra_ffn_post_norms(config);
}

int bn_model_load_policy_moe_uses_scaled_router_input(
    const BnConfig *config) {
    return bn_model_arch_moe_uses_scaled_router_input(config);
}

int bn_model_load_policy_moe_uses_dense_residual_branch(
    const BnConfig *config) {
    return bn_model_arch_moe_uses_dense_residual_branch(config);
}

int bn_model_load_policy_has_shared_expert(const BnConfig *config) {
    return bn_model_arch_config_has_shared_expert(config);
}

int bn_model_load_policy_weight_type_supported(int type) {
    return bn_quant_format_supported(type);
}

int bn_model_load_policy_weight_uses_embedded_block_scale(int type) {
    return bn_quant_format_uses_embedded_scale(type);
}

int bn_model_load_policy_weight_has_embedded_tensor_scale(int type) {
    return bn_quant_format_has_embedded_tensor_scale(type);
}

size_t bn_model_load_policy_weight_embedded_tensor_scale_offset(int type,
                                                                int rows,
                                                                int cols) {
    return bn_quant_embedded_tensor_scale_offset(type, rows, cols);
}

int bn_model_load_policy_tied_logits_uses_quant_path(int type) {
    return bn_quant_format_tied_logits_uses_quant_path(type);
}

int bn_model_load_policy_logits_i8_cache_supported(int type) {
    return bn_quant_format_supports_logits_i8_cache(type);
}

void bn_model_load_policy_prepare_logits_i8_cache(const uint16_t *src,
                                                  int8_t *dst,
                                                  float *scales,
                                                  int rows,
                                                  int dim) {
    bn_quant_f16_rows_to_i8_dispatch(src, dst, scales, rows, dim);
}

int bn_model_load_policy_shared_expert_gate_uses_dense_float(int type) {
    return bn_quant_format_is_f32(type);
}

int bn_model_load_policy_can_convert_shared_expert_gate_to_dense_float(
    int type) {
    return bn_quant_format_can_convert_dense_to_f32(type);
}

int bn_model_load_policy_convert_shared_expert_gate_to_dense_float(
    int type,
    const void *src,
    float *dst,
    int n) {
    return bn_quant_format_convert_dense_to_f32(type, src, dst, n);
}

int bn_model_load_policy_dense_float_weight_type(void) {
    return bn_quant_format_dense_f32_type();
}

int bn_model_prompt_cache_attention_layer_count(const BnConfig *config) {
    return bn_model_arch_attention_layer_count(config);
}

int bn_model_prompt_cache_supports_kv_snapshot(const BnConfig *config) {
    return config && !bn_model_arch_uses_hybrid_layer_layout(config);
}

int bn_model_session_policy_attention_layer_count(const BnConfig *config) {
    return bn_model_arch_attention_layer_count(config);
}

int bn_model_session_policy_ssm_layer_count(const BnConfig *config) {
    return bn_model_arch_ssm_layer_count(config);
}

int bn_model_session_policy_uses_hybrid_layer_layout(
    const BnConfig *config) {
    return bn_model_arch_uses_hybrid_layer_layout(config);
}

int bn_model_session_policy_shared_expert_hidden_dim(
    const BnConfig *config) {
    return bn_model_arch_shared_expert_hidden_dim(config);
}

int bn_model_session_policy_uses_moe(const BnConfig *config) {
    return bn_model_arch_uses_moe(config);
}

int bn_model_session_policy_per_layer_embedding_dim(
    const BnConfig *config) {
    return bn_model_arch_per_layer_embedding_dim(config);
}

void bn_model_transformer_policy_init_rope_frequencies_for_theta(
    float theta,
    int rope_dims,
    float *freqs,
    int capacity_pairs) {
    bn_model_arch_init_rope_frequencies_for_theta(
        theta, rope_dims, freqs, capacity_pairs);
}

void bn_model_transformer_policy_init_rope_angles_for_theta(
    float theta,
    int rope_dims,
    int position,
    float *angles,
    int capacity_pairs) {
    bn_model_arch_init_rope_angles_for_theta(
        theta, rope_dims, position, angles, capacity_pairs);
}

void bn_model_session_policy_init_rope_frequencies(const BnConfig *config,
                                                   float *freqs,
                                                   int capacity_pairs) {
    bn_model_arch_init_rope_frequencies(config, freqs, capacity_pairs);
}

int bn_model_embed_policy_scales_token_embedding(const BnConfig *config) {
    return bn_model_arch_uses_per_layer_embedding(config);
}

int bn_model_embed_policy_dequant_row(int type,
                                      const void *data,
                                      int row,
                                      int n,
                                      float *out) {
    return bn_quant_dequant_row(type, data, row, n, out);
}

int bn_model_moe_policy_requires_float_kquant_gateup_fallback(
    const BnConfig *config) {
    return bn_model_arch_moe_requires_float_kquant_gateup_fallback(config);
}

int bn_model_moe_policy_uses_scaled_router_input(
    const BnConfig *config) {
    return bn_model_arch_moe_uses_scaled_router_input(config);
}

int bn_model_moe_policy_uses_reference_router_accumulation(
    const BnConfig *config) {
    return bn_model_arch_moe_uses_reference_router_accumulation(config);
}

int bn_model_moe_policy_uses_dense_residual_branch(
    const BnConfig *config) {
    return bn_model_arch_moe_uses_dense_residual_branch(config);
}

int bn_model_backend_policy_ffn_sub_norm_elements(const BnConfig *config) {
    if (!config)
        return 0;
    return bn_model_arch_moe_uses_dense_residual_branch(config)
        ? config->dim : config->hidden_dim;
}

int bn_model_moe_policy_uses_reference_silu(const BnConfig *config) {
    return bn_model_arch_moe_uses_reference_silu(config);
}

int bn_model_moe_policy_activation(const BnConfig *config) {
    return bn_model_arch_config_activation(config);
}

float bn_model_moe_policy_norm_epsilon(const BnConfig *config) {
    return bn_model_arch_norm_epsilon(config);
}

int bn_model_moe_policy_prefill_requires_matvec(const BnConfig *config) {
    return bn_model_arch_moe_prefill_requires_matvec(config);
}

int bn_model_moe_policy_uses_grouped_expert_route(
    const BnConfig *config) {
    return bn_model_arch_uses_more_than_two_expert_moe(config);
}

int bn_model_moe_policy_total_experts(const BnConfig *config) {
    return bn_model_arch_moe_total_experts(config);
}

int bn_model_moe_policy_active_experts(const BnConfig *config) {
    return bn_model_arch_moe_active_experts(config);
}

int bn_model_moe_policy_expert_hidden_dim(const BnConfig *config) {
    return bn_model_arch_moe_expert_hidden_dim(config);
}

int bn_model_moe_policy_normalizes_topk_route_weights(
    const BnConfig *config) {
    return bn_model_arch_moe_normalizes_topk_route_weights(config);
}

float bn_model_moe_policy_expert_weights_scale(const BnConfig *config) {
    return bn_model_arch_moe_expert_weights_scale(config);
}

int bn_model_moe_policy_uses_expert_weights(const BnConfig *config) {
    return bn_model_arch_uses_moe(config);
}

int bn_model_moe_policy_uses_all_active_two_expert_set(
    const BnConfig *config) {
    return bn_model_arch_uses_two_expert_all_active_moe(config);
}

int bn_model_moe_policy_uses_all_active_two_expert_route(
    const BnConfig *config,
    int dim) {
    return bn_model_arch_uses_all_active_two_expert_moe(config, dim);
}

int bn_model_moe_policy_has_shared_expert(const BnConfig *config) {
    return bn_model_arch_config_has_shared_expert(config);
}

int bn_model_moe_policy_shared_expert_hidden_dim(
    const BnConfig *config) {
    return bn_model_arch_shared_expert_hidden_dim(config);
}

int bn_model_transformer_policy_is_attention_layer(
    const BnConfig *config,
    int layer) {
    return bn_model_arch_is_attention_layer(config, layer);
}

int bn_model_transformer_policy_attention_layer_index(
    const BnConfig *config,
    int layer) {
    return bn_model_arch_attention_layer_index(config, layer);
}

int bn_model_transformer_policy_ssm_layer_index(
    const BnConfig *config,
    int layer) {
    return bn_model_arch_ssm_layer_index(config, layer);
}

int bn_model_transformer_policy_attention_layer_count(
    const BnConfig *config) {
    return bn_model_arch_attention_layer_count(config);
}

int bn_model_transformer_policy_ssm_layer_count(const BnConfig *config) {
    return bn_model_arch_ssm_layer_count(config);
}

int bn_model_transformer_policy_uses_hybrid_layer_layout(
    const BnConfig *config) {
    return bn_model_arch_uses_hybrid_layer_layout(config);
}

int bn_model_transformer_policy_uses_hybrid_ssm(const BnConfig *config) {
    return bn_model_arch_uses_hybrid_ssm(config);
}

int bn_model_transformer_policy_uses_hybrid_moe(const BnConfig *config) {
    return bn_model_arch_uses_hybrid_moe(config);
}

int bn_model_transformer_policy_uses_large_dense_hybrid_ssm(
    const BnConfig *config) {
    return bn_model_arch_uses_large_dense_hybrid_ssm(config);
}

int bn_model_transformer_policy_uses_non_hybrid_moe(
    const BnConfig *config) {
    return bn_model_arch_uses_non_hybrid_moe(config);
}

int bn_model_transformer_policy_uses_moe(const BnConfig *config) {
    return bn_model_arch_uses_moe(config);
}

int bn_model_transformer_policy_uses_dense_attention_only(
    const BnConfig *config) {
    return bn_model_arch_uses_dense_attention_only(config);
}

int bn_model_transformer_policy_uses_small_dense_shape(
    const BnConfig *config) {
    return bn_model_arch_uses_small_dense_shape(config);
}

int bn_model_transformer_policy_uses_large_dense_shape(
    const BnConfig *config) {
    return bn_model_arch_uses_large_dense_shape(config);
}

int bn_model_transformer_policy_allows_small_dense_prefill_decode_fallback(
    const BnConfig *config) {
    return bn_model_arch_allows_small_dense_prefill_decode_fallback(config);
}

int bn_model_transformer_policy_dense_batch_prefill_shape_allowed(
    const BnConfig *config,
    int supports_large_dense_batch_prefill) {
    return bn_model_arch_dense_batch_prefill_shape_allowed(
        config, supports_large_dense_batch_prefill);
}

int bn_model_transformer_policy_dense_logits_argmax_shape_allowed(
    const BnConfig *config,
    int logits_rows) {
    return bn_model_arch_dense_logits_argmax_shape_allowed(config,
                                                           logits_rows);
}

int bn_model_transformer_policy_moe_logits_mmvq_argmax_shape_allowed(
    const BnConfig *config,
    int logits_cols) {
    return bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(config,
                                                              logits_cols);
}

int bn_model_transformer_policy_moe_requires_reference_attention(
    const BnConfig *config) {
    return bn_model_arch_moe_requires_reference_attention(config);
}

int bn_model_transformer_policy_requires_reference_attention(
    const BnConfig *config) {
    return bn_model_arch_requires_reference_attention(config);
}

int bn_model_transformer_policy_requires_reference_recurrent(
    const BnConfig *config) {
    return bn_model_arch_requires_reference_recurrent(config);
}

int bn_model_backend_policy_requires_stable_per_layer_input_layout(
    const BnConfig *config) {
    return bn_model_arch_uses_per_layer_embedding(config);
}

float bn_model_transformer_policy_norm_epsilon(const BnConfig *config) {
    return bn_model_arch_norm_epsilon(config);
}

int bn_model_transformer_policy_requires_float_kquant_fallback(
    const BnConfig *config) {
    return bn_model_arch_requires_float_kquant_fallback(config);
}

int bn_model_transformer_policy_activation(const BnConfig *config) {
    return bn_model_arch_config_activation(config);
}

int bn_model_transformer_policy_has_ffn_gate(const BnConfig *config) {
    return bn_model_arch_has_ffn_gate(config);
}

float bn_model_transformer_policy_final_logit_softcap(
    const BnConfig *config) {
    return bn_model_arch_final_logit_softcap(config);
}

int bn_model_transformer_policy_attention_flash_requested(
    const BnConfig *config) {
    return config ? config->flash_attn : 0;
}

int bn_model_transformer_policy_attention_qk_norm_stride(
    const BnConfig *config,
    int head_size) {
    return bn_model_arch_attention_qk_norm_stride(config, head_size);
}

int bn_model_transformer_policy_attention_uses_per_head_qk_norm(
    const BnConfig *config) {
    return bn_model_arch_attention_uses_per_head_qk_norm(config);
}

int bn_model_transformer_policy_prefill_uses_decode_for_parity(
    const BnConfig *config) {
    return bn_model_arch_prefill_uses_decode_for_parity(config);
}

int bn_model_transformer_policy_rmsnorm_uses_reference_order(
    const BnConfig *config) {
    return bn_model_arch_rmsnorm_uses_reference_order(config);
}

float bn_model_transformer_policy_attention_scale(
    const BnConfig *config,
    int head_size) {
    return bn_model_arch_attention_scale(config, head_size);
}

int bn_model_transformer_policy_attention_value_shares_key(
    const BnConfig *config) {
    return bn_model_arch_attention_value_shares_key_config(config);
}

int bn_model_transformer_policy_uses_attention_post_norm(
    const BnConfig *config) {
    return bn_model_arch_uses_attention_post_norm(config);
}

int bn_model_transformer_policy_uses_ffn_post_norm(
    const BnConfig *config) {
    return bn_model_arch_uses_ffn_post_norm(config);
}

int bn_model_transformer_policy_uses_layer_output_scale(
    const BnConfig *config) {
    return bn_model_arch_uses_layer_output_scale(config);
}

int bn_model_transformer_policy_per_layer_embedding_dim(
    const BnConfig *config) {
    return bn_model_arch_per_layer_embedding_dim(config);
}

int bn_model_transformer_policy_uses_per_layer_embedding(
    const BnConfig *config) {
    return bn_model_arch_uses_per_layer_embedding(config);
}

int bn_model_transformer_policy_divides_rope_freqs(
    const BnConfig *config,
    int layer) {
    return bn_model_arch_divides_rope_freqs(config, layer);
}

int bn_model_transformer_policy_rope_dims_for_head(
    const BnConfig *config,
    int layer_head_size) {
    return bn_model_arch_rope_dims_for_head(config, layer_head_size);
}

float bn_model_transformer_policy_rope_theta_for_head(
    const BnConfig *config,
    int layer_head_size) {
    return bn_model_arch_rope_theta_for_head(config, layer_head_size);
}

float bn_model_transformer_policy_rope_base_theta(const BnConfig *config) {
    return bn_model_arch_rope_base_theta(config);
}

int bn_model_transformer_policy_rope_uses_base_frequency(
    const BnConfig *config,
    int layer_head_size) {
    return bn_model_arch_rope_uses_base_frequency(config, layer_head_size);
}

int bn_model_transformer_policy_prefill_uses_reference_activation(
    const BnConfig *config) {
    return bn_model_arch_prefill_uses_reference_activation(config);
}

int bn_model_transformer_policy_ffn_uses_reference_activation(
    const BnConfig *config) {
    return bn_model_arch_ffn_uses_reference_activation(config);
}

int bn_model_activation_plan_attention_layer_count(const BnConfig *config) {
    return bn_model_arch_attention_layer_count(config);
}

int bn_model_activation_plan_ssm_layer_count(const BnConfig *config) {
    return bn_model_arch_ssm_layer_count(config);
}

int bn_model_activation_plan_uses_hybrid_ssm(const BnConfig *config) {
    return bn_model_arch_uses_hybrid_ssm(config);
}

int bn_model_activation_plan_uses_hybrid_moe(const BnConfig *config) {
    return bn_model_arch_uses_hybrid_moe(config);
}

int bn_model_activation_plan_uses_moe(const BnConfig *config) {
    return bn_model_arch_uses_moe(config);
}

int bn_model_activation_plan_rope_dims_for_head(const BnConfig *config,
                                                int layer_head_size) {
    return bn_model_arch_rope_dims_for_head(config, layer_head_size);
}

void bn_model_activation_plan_init_rope_frequencies(
    const BnConfig *config, float *freqs, int capacity_pairs) {
    bn_model_arch_init_rope_frequencies(config, freqs, capacity_pairs);
}
