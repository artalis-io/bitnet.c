#ifndef BN_GPU_SHADER_H
#define BN_GPU_SHADER_H

#include "gpu_shader_ir_internal.h"

// Backend-private shader pipeline IDs for current Metal/WGPU dispatch tables.
#define BN_GPU_SHADER_MATVEC       0
#define BN_GPU_SHADER_RMSNORM      1
#define BN_GPU_SHADER_ROPE         2
#define BN_GPU_SHADER_GQA_SCORES   3
#define BN_GPU_SHADER_SOFTMAX      4
#define BN_GPU_SHADER_GQA_COMBINE  5
#define BN_GPU_SHADER_SILU_GATE    6
#define BN_GPU_SHADER_RELU2_GATE   7
#define BN_GPU_SHADER_RESIDUAL_ADD 8
#define BN_GPU_SHADER_COPY         9
#define BN_GPU_SHADER_BIAS_ADD     10
#define BN_GPU_SHADER_RESIDUAL_RMSNORM 11
#define BN_GPU_SHADER_WEIGHTED_ADD   12
#define BN_GPU_SHADER_SSM_CONV_SILU  13
#define BN_GPU_SHADER_SSM_L2NORM     14
#define BN_GPU_SHADER_SSM_ALPHA_BETA 15
#define BN_GPU_SHADER_SSM_DELTA      16
#define BN_GPU_SHADER_SSM_GATE           17
#define BN_GPU_SHADER_PER_HEAD_RMSNORM  18
#define BN_GPU_SHADER_DEINTERLEAVE_Q    19
#define BN_GPU_SHADER_SIGMOID_GATE      20
#define BN_GPU_SHADER_FLASH_ATTN       21
#define BN_GPU_SHADER_MATVEC_SPLIT     22
#define BN_GPU_SHADER_ROPE_QK          23
#define BN_GPU_SHADER_FUSED_GATEUP_SILU 24
#define BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT 25
#define BN_GPU_SHADER_Q4K_MATVEC_SPLIT 26
#define BN_GPU_SHADER_Q8_MATVEC_SPLIT  27
#define BN_GPU_SHADER_Q5K_MATVEC_SPLIT 28
#define BN_GPU_SHADER_SILU_ACT         29
#define BN_GPU_SHADER_RELU2_ACT        30
#define BN_GPU_SHADER_COUNT             31

// Current backend activation slots are identity-lowered from graph values.
#define BN_GPU_BUF_X           BN_GPU_VALUE_X
#define BN_GPU_BUF_XB          BN_GPU_VALUE_XB
#define BN_GPU_BUF_XB2         BN_GPU_VALUE_XB2
#define BN_GPU_BUF_Q           BN_GPU_VALUE_Q
#define BN_GPU_BUF_HB          BN_GPU_VALUE_HB
#define BN_GPU_BUF_HB2         BN_GPU_VALUE_HB2
#define BN_GPU_BUF_KEY_CACHE   BN_GPU_VALUE_KEY_CACHE
#define BN_GPU_BUF_VALUE_CACHE BN_GPU_VALUE_VALUE_CACHE
#define BN_GPU_BUF_ATT         BN_GPU_VALUE_ATT
#define BN_GPU_BUF_LOGITS      BN_GPU_VALUE_LOGITS
#define BN_GPU_BUF_ROPE_FREQ   BN_GPU_VALUE_ROPE_FREQ
#define BN_GPU_BUF_SCRATCH     BN_GPU_VALUE_SCRATCH
#define BN_GPU_BUF_QKV         BN_GPU_VALUE_QKV
#define BN_GPU_BUF_MOE_HB      BN_GPU_VALUE_MOE_HB
#define BN_GPU_BUF_MOE_HB2     BN_GPU_VALUE_MOE_HB2
#define BN_GPU_BUF_MOE_OUT     BN_GPU_VALUE_MOE_OUT
#define BN_GPU_BUF_SSM_STATE      BN_GPU_VALUE_SSM_STATE
#define BN_GPU_BUF_SSM_CONV_STATE BN_GPU_VALUE_SSM_CONV_STATE
#define BN_GPU_BUF_SSM_QKV        BN_GPU_VALUE_SSM_QKV
#define BN_GPU_BUF_SSM_Z          BN_GPU_VALUE_SSM_Z
#define BN_GPU_BUF_SSM_ALPHA      BN_GPU_VALUE_SSM_ALPHA
#define BN_GPU_BUF_SSM_BETA       BN_GPU_VALUE_SSM_BETA
#define BN_GPU_BUF_SSM_V          BN_GPU_VALUE_SSM_V
#define BN_GPU_BUF_COUNT          BN_GPU_VALUE_COUNT

static inline int bn_gpu_shader_from_op_code(int code) {
    switch (code) {
        case BN_GPU_CODE_MATVEC: return BN_GPU_SHADER_MATVEC;
        case BN_GPU_CODE_RMSNORM: return BN_GPU_SHADER_RMSNORM;
        case BN_GPU_CODE_ROPE: return BN_GPU_SHADER_ROPE;
        case BN_GPU_CODE_GQA_SCORES: return BN_GPU_SHADER_GQA_SCORES;
        case BN_GPU_CODE_SOFTMAX: return BN_GPU_SHADER_SOFTMAX;
        case BN_GPU_CODE_GQA_COMBINE: return BN_GPU_SHADER_GQA_COMBINE;
        case BN_GPU_CODE_SILU_GATE: return BN_GPU_SHADER_SILU_GATE;
        case BN_GPU_CODE_RELU2_GATE: return BN_GPU_SHADER_RELU2_GATE;
        case BN_GPU_CODE_RESIDUAL_ADD: return BN_GPU_SHADER_RESIDUAL_ADD;
        case BN_GPU_CODE_COPY: return BN_GPU_SHADER_COPY;
        case BN_GPU_CODE_BIAS_ADD: return BN_GPU_SHADER_BIAS_ADD;
        case BN_GPU_CODE_RESIDUAL_RMSNORM: return BN_GPU_SHADER_RESIDUAL_RMSNORM;
        case BN_GPU_CODE_WEIGHTED_ADD: return BN_GPU_SHADER_WEIGHTED_ADD;
        case BN_GPU_CODE_SSM_CONV_SILU: return BN_GPU_SHADER_SSM_CONV_SILU;
        case BN_GPU_CODE_SSM_L2NORM: return BN_GPU_SHADER_SSM_L2NORM;
        case BN_GPU_CODE_SSM_ALPHA_BETA: return BN_GPU_SHADER_SSM_ALPHA_BETA;
        case BN_GPU_CODE_SSM_DELTA: return BN_GPU_SHADER_SSM_DELTA;
        case BN_GPU_CODE_SSM_GATE: return BN_GPU_SHADER_SSM_GATE;
        case BN_GPU_CODE_PER_HEAD_RMSNORM: return BN_GPU_SHADER_PER_HEAD_RMSNORM;
        case BN_GPU_CODE_DEINTERLEAVE_Q: return BN_GPU_SHADER_DEINTERLEAVE_Q;
        case BN_GPU_CODE_SIGMOID_GATE: return BN_GPU_SHADER_SIGMOID_GATE;
        case BN_GPU_CODE_FLASH_ATTN: return BN_GPU_SHADER_FLASH_ATTN;
        case BN_GPU_CODE_MATVEC_SPLIT: return BN_GPU_SHADER_MATVEC_SPLIT;
        case BN_GPU_CODE_ROPE_QK: return BN_GPU_SHADER_ROPE_QK;
        case BN_GPU_CODE_FUSED_GATEUP_SILU: return BN_GPU_SHADER_FUSED_GATEUP_SILU;
        case BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT: return BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT;
        case BN_GPU_CODE_Q4K_MATVEC_SPLIT: return BN_GPU_SHADER_Q4K_MATVEC_SPLIT;
        case BN_GPU_CODE_Q8_MATVEC_SPLIT: return BN_GPU_SHADER_Q8_MATVEC_SPLIT;
        case BN_GPU_CODE_Q5K_MATVEC_SPLIT: return BN_GPU_SHADER_Q5K_MATVEC_SPLIT;
        case BN_GPU_CODE_SILU_ACT: return BN_GPU_SHADER_SILU_ACT;
        case BN_GPU_CODE_RELU2_ACT: return BN_GPU_SHADER_RELU2_ACT;
        default: return -1;
    }
}

static inline uint32_t bn_gpu_shader_buf_bit(int idx) {
    return (idx >= 0 && idx < 32) ? (1u << (uint32_t)idx) : 0u;
}

static inline int bn_gpu_shader_access_masks(const BnGPUOp *op,
                                             int shader,
                                             uint32_t *reads,
                                             uint32_t *writes) {
    if (!op || !reads || !writes) return -1;
    uint32_t r = 0;
    uint32_t w = 0;

    switch (shader) {
        case BN_GPU_SHADER_MATVEC:
        case BN_GPU_SHADER_RMSNORM:
        case BN_GPU_SHADER_FUSED_GATEUP_SILU:
        case BN_GPU_SHADER_DEINTERLEAVE_Q:
        case BN_GPU_SHADER_COPY:
            r = bn_gpu_shader_buf_bit(op->buf_in);
            w = bn_gpu_shader_buf_bit(op->buf_out);
            break;
        case BN_GPU_SHADER_ROPE:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_ROPE_FREQ);
            w = bn_gpu_shader_buf_bit(op->buf_in);
            break;
        case BN_GPU_SHADER_ROPE_QK:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_aux) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_ROPE_FREQ);
            w = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_aux);
            break;
        case BN_GPU_SHADER_GQA_SCORES:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_KEY_CACHE);
            w = bn_gpu_shader_buf_bit(BN_GPU_BUF_ATT);
            break;
        case BN_GPU_SHADER_SOFTMAX:
            r = bn_gpu_shader_buf_bit(BN_GPU_BUF_ATT);
            w = bn_gpu_shader_buf_bit(BN_GPU_BUF_ATT);
            break;
        case BN_GPU_SHADER_GQA_COMBINE:
            r = bn_gpu_shader_buf_bit(BN_GPU_BUF_ATT) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_VALUE_CACHE);
            w = bn_gpu_shader_buf_bit(op->buf_out);
            break;
        case BN_GPU_SHADER_FLASH_ATTN:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_KEY_CACHE) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_VALUE_CACHE);
            w = bn_gpu_shader_buf_bit(op->buf_out);
            break;
        case BN_GPU_SHADER_SILU_GATE:
        case BN_GPU_SHADER_RELU2_GATE:
        case BN_GPU_SHADER_RESIDUAL_ADD:
        case BN_GPU_SHADER_WEIGHTED_ADD:
        case BN_GPU_SHADER_SSM_GATE:
        case BN_GPU_SHADER_SIGMOID_GATE:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_aux);
            w = bn_gpu_shader_buf_bit(op->buf_in);
            break;
        case BN_GPU_SHADER_BIAS_ADD:
        case BN_GPU_SHADER_PER_HEAD_RMSNORM:
        case BN_GPU_SHADER_SILU_ACT:
        case BN_GPU_SHADER_RELU2_ACT:
            r = bn_gpu_shader_buf_bit(op->buf_in);
            w = bn_gpu_shader_buf_bit(op->buf_in);
            break;
        case BN_GPU_SHADER_RESIDUAL_RMSNORM:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_aux);
            w = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_out);
            break;
        case BN_GPU_SHADER_SSM_CONV_SILU:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_CONV_STATE);
            w = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_CONV_STATE);
            break;
        case BN_GPU_SHADER_SSM_L2NORM:
            r = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_aux);
            w = bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_aux);
            break;
        case BN_GPU_SHADER_SSM_ALPHA_BETA:
            r = bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_ALPHA) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_BETA);
            w = bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_ALPHA) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_BETA);
            break;
        case BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT:
            r = bn_gpu_shader_buf_bit(op->buf_in);
            w = bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_ALPHA) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_BETA);
            break;
        case BN_GPU_SHADER_SSM_DELTA:
            r = bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_STATE) |
                bn_gpu_shader_buf_bit(op->buf_in) |
                bn_gpu_shader_buf_bit(op->buf_aux) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_ALPHA) |
                bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_BETA);
            if (op->p[7] == 0)
                r |= bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_V);
            w = bn_gpu_shader_buf_bit(BN_GPU_BUF_SSM_STATE) |
                bn_gpu_shader_buf_bit(op->buf_out);
            break;
        case BN_GPU_SHADER_MATVEC_SPLIT:
        case BN_GPU_SHADER_Q4K_MATVEC_SPLIT:
        case BN_GPU_SHADER_Q8_MATVEC_SPLIT:
        case BN_GPU_SHADER_Q5K_MATVEC_SPLIT:
            r = bn_gpu_shader_buf_bit(op->buf_in);
            w = bn_gpu_shader_buf_bit(op->buf_out) |
                bn_gpu_shader_buf_bit(op->buf_aux) |
                bn_gpu_shader_buf_bit(op->rows);
            break;
        default:
            return -1;
    }

    *reads = r;
    *writes = w;
    return 0;
}

#endif
