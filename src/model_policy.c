#include "model_internal.h"
#include "backend_quant.h"
#include "model_arch.h"
#include "quant.h"
#include <math.h>

int bn_model_quant_type_supported(int type) {
    return bn_quant_format_supported(type);
}

int bn_model_quant_uses_embedded_block_scale(int type) {
    return bn_quant_format_uses_embedded_scale(type);
}

int bn_model_quant_uses_embedded_tensor_scale(int type) {
    return bn_quant_format_has_embedded_tensor_scale(type);
}

size_t bn_model_quant_embedded_tensor_scale_offset(int type,
                                                   int rows,
                                                   int cols) {
    return bn_quant_embedded_tensor_scale_offset(type, rows, cols);
}

int bn_model_quant_tied_logits_uses_quant_path(int type) {
    return bn_backend_quant_tied_logits_uses_quant_path(type);
}

int bn_model_quant_logits_i8_cache_supported(int type) {
    return bn_backend_quant_logits_i8_cache_supported(type);
}

void bn_model_quant_prepare_logits_i8_cache(const uint16_t *src,
                                            int8_t *dst,
                                            float *scales,
                                            int rows,
                                            int dim) {
    bn_quant_f16_rows_to_i8_dispatch(src, dst, scales, rows, dim);
}

int bn_model_quant_uses_dense_float(int type) {
    return bn_backend_quant_uses_dense_float(type);
}

int bn_model_quant_can_convert_dense_to_float(int type) {
    return bn_backend_quant_can_convert_dense_to_float(type);
}

int bn_model_quant_convert_dense_to_float(int type,
                                          const void *src,
                                          float *dst,
                                          int n) {
    return bn_backend_quant_convert_dense_to_float(type, src, dst, n);
}

int bn_model_quant_dense_float_type(void) {
    return bn_backend_quant_dense_float_type();
}

int bn_model_quant_dequant_row(int type,
                               const void *data,
                               int row,
                               int n,
                               float *out) {
    return bn_quant_dequant_row(type, data, row, n, out);
}

int bn_model_dequant_qweight_row(const BnQWeight *weight,
                                 int row,
                                 int n,
                                 float *out) {
    if (!weight)
        return -1;
    return bn_model_quant_dequant_row(weight->type, weight->data, row, n, out);
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

int bn_model_config_attention_layer_count(const BnConfig *config) {
    return bn_model_arch_attention_layer_count(config);
}

int bn_model_config_ssm_layer_count(const BnConfig *config) {
    return bn_model_arch_ssm_layer_count(config);
}

int bn_model_config_is_attention_layer(const BnConfig *config, int layer) {
    return bn_model_arch_is_attention_layer(config, layer);
}

int bn_model_config_attention_layer_index(const BnConfig *config,
                                          int layer) {
    return bn_model_arch_attention_layer_index(config, layer);
}

int bn_model_config_ssm_layer_index(const BnConfig *config, int layer) {
    return bn_model_arch_ssm_layer_index(config, layer);
}

int bn_model_config_uses_hybrid_layer_layout(const BnConfig *config) {
    return bn_model_arch_uses_hybrid_layer_layout(config);
}

int bn_model_config_uses_hybrid_ssm(const BnConfig *config) {
    return bn_model_arch_uses_hybrid_ssm(config);
}

int bn_model_config_uses_hybrid_moe(const BnConfig *config) {
    return bn_model_arch_uses_hybrid_moe(config);
}

int bn_model_config_uses_moe(const BnConfig *config) {
    return bn_model_arch_uses_moe(config);
}

int bn_model_config_uses_all_active_two_expert_moe(const BnConfig *config,
                                                   int dim) {
    return bn_model_arch_uses_all_active_two_expert_moe(config, dim);
}

int bn_model_config_uses_two_expert_all_active_moe(const BnConfig *config) {
    return bn_model_arch_uses_two_expert_all_active_moe(config);
}

int bn_model_config_uses_more_than_two_expert_moe(const BnConfig *config) {
    return bn_model_arch_uses_more_than_two_expert_moe(config);
}

int bn_model_config_moe_total_experts(const BnConfig *config) {
    return config ? config->n_experts : 0;
}

int bn_model_config_moe_active_experts(const BnConfig *config) {
    return config ? config->n_experts_active : 0;
}

int bn_model_config_moe_expert_hidden_dim(const BnConfig *config) {
    return config ? config->moe_intermediate_size : 0;
}

int bn_model_config_moe_route_shape_valid(const BnConfig *config) {
    return bn_model_config_moe_active_experts(config) > 0 &&
           bn_model_config_moe_expert_hidden_dim(config) > 0;
}

int bn_model_config_moe_normalizes_topk_route_weights(
    const BnConfig *config) {
    return config ? config->moe_norm_topk_prob : 0;
}

float bn_model_config_moe_expert_weights_scale(const BnConfig *config) {
    return config ? config->moe_expert_weights_scale : 0.0f;
}

int bn_model_config_moe_uses_reference_silu(const BnConfig *config) {
    return config ? config->moe_uses_reference_silu : -1;
}

int bn_model_config_has_shared_expert(const BnConfig *config) {
    return config ? config->has_shared_expert : 0;
}

int bn_model_config_shared_expert_hidden_dim(const BnConfig *config) {
    if (!bn_model_config_has_shared_expert(config) ||
        config->shared_expert_intermediate_size <= 0)
        return 0;
    return config->shared_expert_intermediate_size;
}

int bn_model_config_moe_requires_float_kquant_gateup_fallback(
    const BnConfig *config) {
    return bn_model_arch_moe_requires_float_kquant_gateup_fallback(config);
}

int bn_model_config_moe_uses_scaled_router_input(
    const BnConfig *config) {
    return bn_model_arch_moe_uses_scaled_router_input(config);
}

int bn_model_config_moe_uses_dense_residual_branch(
    const BnConfig *config) {
    return bn_model_arch_moe_uses_dense_residual_branch(config);
}

int bn_model_config_moe_prefill_requires_matvec(const BnConfig *config) {
    return bn_model_arch_moe_prefill_requires_matvec(config);
}

int bn_model_config_uses_small_dense_shape(const BnConfig *config) {
    return bn_model_arch_uses_small_dense_shape(config);
}

int bn_model_config_uses_large_dense_shape(const BnConfig *config) {
    return bn_model_arch_uses_large_dense_shape(config);
}

int bn_model_config_uses_large_gpu_graph_fallback_shape(
    const BnConfig *config) {
    return bn_model_arch_uses_large_gpu_graph_fallback_shape(config);
}

int bn_model_config_uses_per_layer_embedding(const BnConfig *config) {
    return bn_model_arch_uses_per_layer_embedding(config);
}

int bn_model_config_uses_large_dense_hybrid_ssm(const BnConfig *config) {
    return bn_model_arch_uses_large_dense_hybrid_ssm(config);
}

int bn_model_config_uses_reference_hybrid_ssm(const BnConfig *config) {
    return bn_model_arch_uses_reference_hybrid_ssm(config);
}

int bn_model_config_uses_non_hybrid_moe(const BnConfig *config) {
    return bn_model_arch_uses_non_hybrid_moe(config);
}

int bn_model_config_uses_dense_attention_only(const BnConfig *config) {
    return bn_model_arch_uses_dense_attention_only(config);
}

int bn_model_config_uses_small_dense_native_quant_shape(
    const BnConfig *config) {
    return bn_model_arch_uses_small_dense_native_quant_shape(config);
}

int bn_model_config_requires_float_kquant_fallback(const BnConfig *config) {
    return bn_model_arch_requires_float_kquant_fallback(config);
}

int bn_model_config_prefill_uses_decode_for_parity(
    const BnConfig *config) {
    return bn_model_arch_prefill_uses_decode_for_parity(config);
}

int bn_model_config_rmsnorm_uses_reference_order(const BnConfig *config) {
    return bn_model_arch_rmsnorm_uses_reference_order(config);
}

float bn_model_config_attention_scale(const BnConfig *config,
                                      int head_size) {
    return bn_model_arch_attention_scale(config, head_size);
}

int bn_model_config_attention_value_shares_key(const BnConfig *config) {
    return bn_model_arch_attention_value_shares_key_config(config);
}

int bn_model_config_uses_attention_post_norm(const BnConfig *config) {
    return bn_model_arch_uses_attention_post_norm(config);
}

int bn_model_config_uses_ffn_post_norm(const BnConfig *config) {
    return bn_model_arch_uses_ffn_post_norm(config);
}

int bn_model_config_uses_layer_output_scale(const BnConfig *config) {
    return bn_model_arch_uses_layer_output_scale(config);
}

int bn_model_config_per_layer_embedding_dim(const BnConfig *config) {
    return bn_model_arch_per_layer_embedding_dim(config);
}

int bn_model_config_loads_extra_metadata(const BnConfig *config) {
    return bn_model_arch_loads_extra_metadata(config);
}

int bn_model_config_loads_per_layer_input_weights(const BnConfig *config) {
    return bn_model_arch_loads_per_layer_input_weights(config);
}

int bn_model_config_layer_reuses_kv(const BnConfig *config, int layer) {
    return bn_model_arch_layer_reuses_kv(config, layer);
}

int bn_model_config_kv_reuse_layer(const BnConfig *config, int layer) {
    return bn_model_arch_kv_reuse_layer(config, layer);
}

int bn_model_config_loads_extra_ffn_post_norms(const BnConfig *config) {
    return bn_model_arch_loads_extra_ffn_post_norms(config);
}

int bn_model_config_divides_rope_freqs(const BnConfig *config, int layer) {
    return bn_model_arch_divides_rope_freqs(config, layer);
}

static int model_config_uses_swa_rope(const BnConfig *config,
                                      int layer_head_size) {
    return config && config->rope_theta_swa > 0.0f &&
           layer_head_size < config->head_size;
}

int bn_model_config_rope_dims_for_head(const BnConfig *config,
                                       int layer_head_size) {
    if (!config || layer_head_size <= 0)
        return 0;
    int base_rope_dims = config->rope_text_dims > 0
        ? config->rope_text_dims
        : (config->rope_dim_count > 0 ? config->rope_dim_count
                                      : layer_head_size);
    int rope_dims = model_config_uses_swa_rope(config, layer_head_size) &&
                    config->rope_dim_count_swa > 0
        ? config->rope_dim_count_swa
        : base_rope_dims;
    if (rope_dims > layer_head_size)
        rope_dims = layer_head_size;
    return rope_dims > 0 ? rope_dims : 0;
}

float bn_model_config_rope_theta_for_head(const BnConfig *config,
                                          int layer_head_size) {
    if (!config)
        return 0.0f;
    return model_config_uses_swa_rope(config, layer_head_size)
        ? config->rope_theta_swa : config->rope_theta;
}

float bn_model_config_rope_base_theta(const BnConfig *config) {
    return config ? config->rope_theta : 0.0f;
}

int bn_model_config_rope_uses_base_frequency(const BnConfig *config,
                                             int layer_head_size) {
    return !model_config_uses_swa_rope(config, layer_head_size);
}

void bn_model_config_init_rope_frequencies(const BnConfig *config,
                                           float *freqs,
                                           int capacity_pairs) {
    if (!config || !freqs || capacity_pairs <= 0)
        return;
    int rope_dims = config->rope_dim_count > 0
        ? config->rope_dim_count : config->head_size;
    int half_rope = rope_dims / 2;
    if (half_rope > capacity_pairs)
        half_rope = capacity_pairs;
    for (int i = 0; i < half_rope; i++)
        freqs[i] = 1.0f / powf(config->rope_theta,
                               (float)(2 * i) / (float)rope_dims);
    if (config->rope_text_dims > 0) {
        int text_pairs = config->rope_text_dims / 2;
        if (text_pairs < 0)
            text_pairs = 0;
        for (int i = text_pairs; i < half_rope; i++)
            freqs[i] = 0.0f;
    }
}

float bn_model_config_final_logit_softcap(const BnConfig *config) {
    return config ? config->final_logit_softcap : 0.0f;
}

int bn_model_config_prefill_uses_reference_activation(
    const BnConfig *config) {
    return bn_model_arch_prefill_uses_reference_activation(config);
}

int bn_model_config_ffn_uses_reference_activation(
    const BnConfig *config) {
    return bn_model_arch_ffn_uses_reference_activation(config);
}

int bn_model_config_dense_batch_prefill_shape_allowed(
    const BnConfig *config,
    int supports_large_dense_batch_prefill) {
    return bn_model_arch_dense_batch_prefill_shape_allowed(
        config, supports_large_dense_batch_prefill);
}

int bn_model_config_dense_logits_argmax_shape_allowed(
    const BnConfig *config,
    int logits_rows) {
    return bn_model_arch_dense_logits_argmax_shape_allowed(config,
                                                           logits_rows);
}

int bn_model_config_moe_logits_mmvq_argmax_shape_allowed(
    const BnConfig *config,
    int logits_cols) {
    return bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(config,
                                                              logits_cols);
}

int bn_model_config_allows_small_dense_native_quant(
    const BnConfig *config) {
    return bn_model_arch_allows_small_dense_native_quant(config);
}

int bn_model_config_small_dense_native_quant_to_layer(
    const BnConfig *config) {
    return bn_model_arch_small_dense_native_quant_to_layer(config);
}

int bn_model_config_allows_small_dense_prefill_decode_fallback(
    const BnConfig *config) {
    return bn_model_arch_allows_small_dense_prefill_decode_fallback(config);
}

int bn_model_config_small_dense_prefill_min_tokens(
    const BnConfig *config) {
    return bn_model_arch_small_dense_prefill_min_tokens(config);
}

int bn_model_config_allows_small_dense_native_logit_refine(
    const BnConfig *config) {
    return bn_model_arch_allows_small_dense_native_logit_refine(config);
}

int bn_model_config_moe_prefers_reference_gpu_attention(
    const BnConfig *config) {
    return bn_model_arch_moe_prefers_reference_gpu_attention(config);
}
