#ifndef BN_GPU_QUANT_LOWERING_INTERNAL_H
#define BN_GPU_QUANT_LOWERING_INTERNAL_H

#include "backend_quant.h"
#include "gpu_shader_ir_internal.h"

static inline int bn_gpu_quant_split_op_code(int type) {
    switch (bn_backend_quant_gpu_split_cap(type)) {
        case BN_GPU_CAP_Q4_MATVEC_SPLIT:
        case BN_GPU_CAP_Q5_MATVEC_SPLIT:
            return BN_GPU_CODE_MATVEC_SPLIT;
        case BN_GPU_CAP_Q4K_MATVEC_SPLIT:
            return BN_GPU_CODE_Q4K_MATVEC_SPLIT;
        case BN_GPU_CAP_Q5K_MATVEC_SPLIT:
            return BN_GPU_CODE_Q5K_MATVEC_SPLIT;
        case BN_GPU_CAP_Q8_MATVEC_SPLIT:
            return BN_GPU_CODE_Q8_MATVEC_SPLIT;
        default: return 0;
    }
}

static inline int bn_gpu_quant_split_op_is_standard(int op_code) {
    return op_code == BN_GPU_CODE_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_is_q4k(int op_code) {
    return op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_is_q5k(int op_code) {
    return op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_is_q8(int op_code) {
    return op_code == BN_GPU_CODE_Q8_MATVEC_SPLIT;
}

static inline int bn_gpu_quant_split_op_known(int op_code) {
    return op_code != BN_GPU_CODE_UNKNOWN;
}

#endif // BN_GPU_QUANT_LOWERING_INTERNAL_H
