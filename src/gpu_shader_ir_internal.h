#ifndef BN_GPU_SHADER_IR_INTERNAL_H
#define BN_GPU_SHADER_IR_INTERNAL_H

#include "gpu_graph_ir.h"
#include <stdint.h>

// Shader uniform parameter count (32 bytes = 8 x u32, matches WGSL Uniforms structs)
#define BN_GPU_OP_PARAMS 8

typedef enum {
    BN_GPU_OP_UNKNOWN = 0,
    BN_GPU_OP_MATVEC = 1,
    BN_GPU_OP_RMSNORM = 2,
    BN_GPU_OP_ROPE = 3,
    BN_GPU_OP_ATTENTION = 4,
    BN_GPU_OP_ACTIVATION = 5,
    BN_GPU_OP_RESIDUAL = 6,
    BN_GPU_OP_COPY = 7,
    BN_GPU_OP_FFN = 8,
    BN_GPU_OP_SSM = 9,
    BN_GPU_OP_LOGITS = 10,
} BnGPUOpKind;

typedef enum {
    BN_GPU_CODE_UNKNOWN = 0,
    BN_GPU_CODE_MATVEC,
    BN_GPU_CODE_RMSNORM,
    BN_GPU_CODE_ROPE,
    BN_GPU_CODE_GQA_SCORES,
    BN_GPU_CODE_SOFTMAX,
    BN_GPU_CODE_GQA_COMBINE,
    BN_GPU_CODE_SILU_GATE,
    BN_GPU_CODE_RELU2_GATE,
    BN_GPU_CODE_RESIDUAL_ADD,
    BN_GPU_CODE_COPY,
    BN_GPU_CODE_BIAS_ADD,
    BN_GPU_CODE_RESIDUAL_RMSNORM,
    BN_GPU_CODE_WEIGHTED_ADD,
    BN_GPU_CODE_SSM_CONV_SILU,
    BN_GPU_CODE_SSM_L2NORM,
    BN_GPU_CODE_SSM_ALPHA_BETA,
    BN_GPU_CODE_SSM_DELTA,
    BN_GPU_CODE_SSM_GATE,
    BN_GPU_CODE_PER_HEAD_RMSNORM,
    BN_GPU_CODE_DEINTERLEAVE_Q,
    BN_GPU_CODE_SIGMOID_GATE,
    BN_GPU_CODE_FLASH_ATTN,
    BN_GPU_CODE_MATVEC_SPLIT,
    BN_GPU_CODE_ROPE_QK,
    BN_GPU_CODE_FUSED_GATEUP_SILU,
    BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT,
    BN_GPU_CODE_Q4K_MATVEC_SPLIT,
    BN_GPU_CODE_Q8_MATVEC_SPLIT,
    BN_GPU_CODE_Q5K_MATVEC_SPLIT,
    BN_GPU_CODE_SILU_ACT,
    BN_GPU_CODE_RELU2_ACT,
    BN_GPU_CODE_WEIGHTED_ADD_SIGMOID,
    BN_GPU_CODE_MOE_ROUTE_TOPK,
    BN_GPU_CODE_MOE_ROUTED_FFN,
} BnGPUOpCode;

// A single backend shader command in the lowered forward pass.
typedef struct BnGPUOp {
    int op_kind;         // BnGPUOpKind semantic op; 0 = infer from op_code
    int op_code;         // BnGPUOpCode concrete shader operation
    int type;            // BN_GGUF_TENSOR_* (matvec only, -1 otherwise)
    void *W_buf;         // primary weight buffer handle
    void *W_buf2;        // optional secondary weight buffer handle
    void *W_buf3;        // optional tertiary weight buffer handle
    int buf_in;          // BN_GPU_VALUE_* primary input
    int buf_out;         // BN_GPU_VALUE_* output
    int buf_aux;         // secondary BN_GPU_VALUE_* (-1 if unused)
    int rows, cols;      // dimensions (matvec: weight dims; others: element count in p0)
    uint32_t flags;      // backend-private lowered flags
    uint32_t p[BN_GPU_OP_PARAMS]; // shader-specific parameters (32 bytes)
} BnGPUOp;

#define BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT 1u
#define BN_GPU_OP_FLAG_MOE_ROUTE_BLOCK 1u
#define BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM 2u
#define BN_GPU_OP_FLAG_REFERENCE_SILU 4u
#define BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT 8u

static inline int bn_gpu_op_code_is_matvec(int code) {
    return code == BN_GPU_CODE_MATVEC;
}

static inline int bn_gpu_op_code_is_split_matvec(int code) {
    switch (code) {
        case BN_GPU_CODE_MATVEC_SPLIT:
        case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
        case BN_GPU_CODE_Q8_MATVEC_SPLIT:
        case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
            return 1;
        default:
            return 0;
    }
}

static inline int bn_gpu_op_code_is_rope(int code) {
    return code == BN_GPU_CODE_ROPE || code == BN_GPU_CODE_ROPE_QK;
}

static inline int bn_gpu_op_code_is_rope_qk(int code) {
    return code == BN_GPU_CODE_ROPE_QK;
}

static inline int bn_gpu_op_code_is_flash_attention(int code) {
    return code == BN_GPU_CODE_FLASH_ATTN;
}

static inline int bn_gpu_op_code_is_per_head_rmsnorm(int code) {
    return code == BN_GPU_CODE_PER_HEAD_RMSNORM;
}

static inline int bn_gpu_op_code_is_copy(int code) {
    return code == BN_GPU_CODE_COPY;
}

static inline BnGPUOpKind bn_gpu_op_kind_from_code(int code) {
    switch (code) {
        case BN_GPU_CODE_MATVEC:
        case BN_GPU_CODE_MATVEC_SPLIT:
        case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
        case BN_GPU_CODE_Q8_MATVEC_SPLIT:
        case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
            return BN_GPU_OP_MATVEC;
        case BN_GPU_CODE_RMSNORM:
        case BN_GPU_CODE_RESIDUAL_RMSNORM:
        case BN_GPU_CODE_PER_HEAD_RMSNORM:
            return BN_GPU_OP_RMSNORM;
        case BN_GPU_CODE_ROPE:
        case BN_GPU_CODE_ROPE_QK:
            return BN_GPU_OP_ROPE;
        case BN_GPU_CODE_GQA_SCORES:
        case BN_GPU_CODE_SOFTMAX:
        case BN_GPU_CODE_GQA_COMBINE:
        case BN_GPU_CODE_FLASH_ATTN:
            return BN_GPU_OP_ATTENTION;
        case BN_GPU_CODE_SILU_GATE:
        case BN_GPU_CODE_RELU2_GATE:
        case BN_GPU_CODE_SIGMOID_GATE:
        case BN_GPU_CODE_SILU_ACT:
        case BN_GPU_CODE_RELU2_ACT:
            return BN_GPU_OP_ACTIVATION;
        case BN_GPU_CODE_RESIDUAL_ADD:
        case BN_GPU_CODE_WEIGHTED_ADD:
        case BN_GPU_CODE_WEIGHTED_ADD_SIGMOID:
        case BN_GPU_CODE_BIAS_ADD:
            return BN_GPU_OP_RESIDUAL;
        case BN_GPU_CODE_COPY:
        case BN_GPU_CODE_DEINTERLEAVE_Q:
            return BN_GPU_OP_COPY;
        case BN_GPU_CODE_FUSED_GATEUP_SILU:
        case BN_GPU_CODE_MOE_ROUTE_TOPK:
        case BN_GPU_CODE_MOE_ROUTED_FFN:
            return BN_GPU_OP_FFN;
        case BN_GPU_CODE_SSM_CONV_SILU:
        case BN_GPU_CODE_SSM_L2NORM:
        case BN_GPU_CODE_SSM_ALPHA_BETA:
        case BN_GPU_CODE_SSM_DELTA:
        case BN_GPU_CODE_SSM_GATE:
        case BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT:
            return BN_GPU_OP_SSM;
        default:
            return BN_GPU_OP_UNKNOWN;
    }
}

static inline BnGPUOpKind bn_gpu_op_kind(const BnGPUOp *op) {
    if (!op) return BN_GPU_OP_UNKNOWN;
    return op->op_kind ? (BnGPUOpKind)op->op_kind
                       : bn_gpu_op_kind_from_code(op->op_code);
}

// Pre-compiled shader command list for dense models (eliminates per-token malloc)
typedef struct {
    BnGPUOp *ops;       // pre-allocated op array
    int cap;            // capacity (max ops)
} BnGPUGraph;

#endif // BN_GPU_SHADER_IR_INTERNAL_H
