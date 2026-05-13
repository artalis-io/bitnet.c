#ifndef BN_GPU_QUANT_LOWERING_INTERNAL_H
#define BN_GPU_QUANT_LOWERING_INTERNAL_H

#include "gguf.h"
#include "gpu_shader_ir_internal.h"

static inline int bn_gpu_quant_split_op_code(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CODE_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q4_K: return BN_GPU_CODE_Q4K_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q5_K: return BN_GPU_CODE_Q5K_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q8_0: return BN_GPU_CODE_Q8_MATVEC_SPLIT;
        default: return 0;
    }
}

#endif // BN_GPU_QUANT_LOWERING_INTERNAL_H
