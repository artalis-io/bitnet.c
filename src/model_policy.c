#include "model_internal.h"
#include "backend_quant.h"
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
