#ifndef BN_GPU_SHADER_IR_H
#define BN_GPU_SHADER_IR_H

#include <stdint.h>

// Shared shader IR values used by GPU-resident forward ops.
// Backends lower these values to their own activation buffer slots.
#define BN_GPU_VALUE_X           0
#define BN_GPU_VALUE_XB          1
#define BN_GPU_VALUE_XB2         2
#define BN_GPU_VALUE_Q           3
#define BN_GPU_VALUE_HB          4
#define BN_GPU_VALUE_HB2         5
#define BN_GPU_VALUE_KEY_CACHE   6
#define BN_GPU_VALUE_VALUE_CACHE 7
#define BN_GPU_VALUE_ATT         8
#define BN_GPU_VALUE_LOGITS      9
#define BN_GPU_VALUE_ROPE_FREQ   10
#define BN_GPU_VALUE_SCRATCH     11
#define BN_GPU_VALUE_QKV         12
#define BN_GPU_VALUE_MOE_HB      13
#define BN_GPU_VALUE_MOE_HB2     14
#define BN_GPU_VALUE_MOE_OUT     15
#define BN_GPU_VALUE_SSM_STATE      16
#define BN_GPU_VALUE_SSM_CONV_STATE 17
#define BN_GPU_VALUE_SSM_QKV        18
#define BN_GPU_VALUE_SSM_Z          19
#define BN_GPU_VALUE_SSM_ALPHA      20
#define BN_GPU_VALUE_SSM_BETA       21
#define BN_GPU_VALUE_SSM_V          22
#define BN_GPU_VALUE_COUNT          23

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
} BnGPUOpCode;

// A single shader IR operation in the forward pass.
typedef struct BnGPUOp {
    int op_kind;         // BnGPUOpKind semantic op; 0 = infer from op_code
    int op_code;         // BnGPUOpCode concrete shader operation
    int type;            // BN_GGUF_TENSOR_* (matvec only, -1 otherwise)
    void *W_buf;         // weight buffer handle (matvec only, NULL otherwise)
    int buf_in;          // BN_GPU_VALUE_* primary input
    int buf_out;         // BN_GPU_VALUE_* output
    int buf_aux;         // secondary BN_GPU_VALUE_* (-1 if unused)
    int rows, cols;      // dimensions (matvec: weight dims; others: element count in p0)
    uint32_t p[BN_GPU_OP_PARAMS]; // shader-specific parameters (32 bytes)
} BnGPUOp;

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
        case BN_GPU_CODE_BIAS_ADD:
            return BN_GPU_OP_RESIDUAL;
        case BN_GPU_CODE_COPY:
        case BN_GPU_CODE_DEINTERLEAVE_Q:
            return BN_GPU_OP_COPY;
        case BN_GPU_CODE_FUSED_GATEUP_SILU:
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

// Pre-compiled shader IR list for dense models (eliminates per-token malloc)
typedef struct {
    BnGPUOp *ops;       // pre-allocated op array
    int cap;            // capacity (max ops)
} BnGPUGraph;

#endif // BN_GPU_SHADER_IR_H
