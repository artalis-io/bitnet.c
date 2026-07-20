#ifndef BN_GPU_QUANT_LOWERING_INTERNAL_H
#define BN_GPU_QUANT_LOWERING_INTERNAL_H

#include "backend_quant.h"
#include "gpu_shader_ir_internal.h"

static inline int bn_gpu_quant_split_op_code(int type) {
    switch (bn_backend_quant_gpu_split_cap(type)) {
        case BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT:
        case BN_GPU_CAP_MIDBIT_BLOCK32_MATVEC_SPLIT:
            return BN_GPU_CODE_MATVEC_SPLIT;
        case BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT:
            return BN_GPU_CODE_Q4K_MATVEC_SPLIT;
        case BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT:
            return BN_GPU_CODE_Q5K_MATVEC_SPLIT;
        case BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT:
            return BN_GPU_CODE_Q8_MATVEC_SPLIT;
        default: return 0;
    }
}

static inline int bn_gpu_quant_split_op_is_standard(int op_code) {
    return op_code == BN_GPU_CODE_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_is_asymmetric_kquant(int op_code) {
    return op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_is_deinterleaved_kquant(int op_code) {
    return op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_is_native_quant(int op_code) {
    return op_code == BN_GPU_CODE_Q8_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_known(int op_code) {
    return op_code != BN_GPU_CODE_UNKNOWN;
}

#endif // BN_GPU_QUANT_LOWERING_INTERNAL_H
