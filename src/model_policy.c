#include "model_internal.h"
#include "backend_quant.h"
#include "model_arch.h"
#include "quant.h"

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

int bn_model_quant_is_dense_f32(int type) {
    return bn_backend_quant_already_f32(type);
}

int bn_model_quant_can_convert_dense_to_f32(int type) {
    return bn_backend_quant_can_convert_dense_to_f32(type);
}

int bn_model_quant_convert_dense_to_f32(int type,
                                        const void *src,
                                        float *dst,
                                        int n) {
    return bn_backend_quant_convert_dense_to_f32(type, src, dst, n);
}

int bn_model_quant_dense_f32_type(void) {
    return bn_backend_quant_dense_f32_type();
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
