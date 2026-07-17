#include "moe_internal.h"
#include "backend_quant.h"

int bn_moe_quant_supports_prepared_q8k(int type) {
    return bn_backend_quant_can_preq8k(type);
}

int bn_moe_quant_uses_embedded_tensor_scale(int type) {
    return bn_quant_format_has_embedded_tensor_scale(type);
}

size_t bn_moe_quant_embedded_tensor_scale_offset(int type, int rows, int cols) {
    return bn_quant_embedded_tensor_scale_offset(type, rows, cols);
}
