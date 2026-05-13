#ifndef BN_GPU_GRAPH_LOWERING_INTERNAL_H
#define BN_GPU_GRAPH_LOWERING_INTERNAL_H

#include "gpu_graph_ir.h"
#include "gpu_quant_lowering_internal.h"
#include "gpu_shader_ir_internal.h"
#include <string.h>

#define BN_GPU_IR_NO_SHADER_SLOT (-1)

typedef struct {
    int shader_slot;
    void *weight_buf;
    int tensor_type;
} BnGPUIRLoweringValue;

typedef struct {
    const BnGPUIRLoweringValue *values;
    int n_values;
} BnGPUIRLoweringMap;

static inline const BnGPUIRLoweringValue *bn_gpu_ir_lowering_value(
    const BnGPUIRLoweringMap *map,
    int value_id) {
    if (!map || !map->values || value_id < 0 || value_id >= map->n_values)
        return NULL;
    return &map->values[value_id];
}

static inline int bn_gpu_ir_lowering_slot(const BnGPUIRLoweringMap *map,
                                          int value_id) {
    const BnGPUIRLoweringValue *value =
        bn_gpu_ir_lowering_value(map, value_id);
    if (!value || value->shader_slot < 0) return BN_GPU_IR_NO_SHADER_SLOT;
    return value->shader_slot;
}

static inline int bn_gpu_ir_activation_op_code(const BnGPUIROp *op) {
    if (!op || op->n_inputs < 1) return BN_GPU_CODE_UNKNOWN;
    int has_aux = op->n_inputs > 1;
    switch ((BnGPUIRActivationKind)op->aux0) {
        case BN_GPU_IR_ACTIVATION_SILU:
            return has_aux ? BN_GPU_CODE_SILU_GATE : BN_GPU_CODE_SILU_ACT;
        case BN_GPU_IR_ACTIVATION_RELU2:
            return has_aux ? BN_GPU_CODE_RELU2_GATE : BN_GPU_CODE_RELU2_ACT;
        case BN_GPU_IR_ACTIVATION_SIGMOID:
            return has_aux ? BN_GPU_CODE_SIGMOID_GATE : BN_GPU_CODE_UNKNOWN;
        default:
            return BN_GPU_CODE_UNKNOWN;
    }
}

static inline int bn_gpu_ir_op_element_count(const BnGPUIROp *op) {
    if (!op) return 0;
    if (op->rows > 0 && op->cols > 0) return op->rows * op->cols;
    if (op->rows > 0) return op->rows;
    return op->cols;
}

static inline int bn_gpu_ir_ssm_op_code(BnGPUIRSSMKind kind) {
    switch (kind) {
        case BN_GPU_IR_SSM_CONV_SILU: return BN_GPU_CODE_SSM_CONV_SILU;
        case BN_GPU_IR_SSM_L2NORM: return BN_GPU_CODE_SSM_L2NORM;
        case BN_GPU_IR_SSM_ALPHA_BETA: return BN_GPU_CODE_SSM_ALPHA_BETA;
        case BN_GPU_IR_SSM_ALPHA_BETA_SPLIT:
            return BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT;
        case BN_GPU_IR_SSM_DELTA: return BN_GPU_CODE_SSM_DELTA;
        case BN_GPU_IR_SSM_GATE: return BN_GPU_CODE_SSM_GATE;
        default: return BN_GPU_CODE_UNKNOWN;
    }
}

static inline int bn_gpu_ir_utility_op_code(BnGPUIRUtilityKind kind) {
    switch (kind) {
        case BN_GPU_IR_UTILITY_BIAS_ADD: return BN_GPU_CODE_BIAS_ADD;
        case BN_GPU_IR_UTILITY_WEIGHTED_ADD: return BN_GPU_CODE_WEIGHTED_ADD;
        case BN_GPU_IR_UTILITY_RESIDUAL_RMSNORM:
            return BN_GPU_CODE_RESIDUAL_RMSNORM;
        case BN_GPU_IR_UTILITY_PER_HEAD_RMSNORM:
            return BN_GPU_CODE_PER_HEAD_RMSNORM;
        case BN_GPU_IR_UTILITY_DEINTERLEAVE_Q:
            return BN_GPU_CODE_DEINTERLEAVE_Q;
        default:
            return BN_GPU_CODE_UNKNOWN;
    }
}

static inline int bn_gpu_ir_utility_op_kind(BnGPUIRUtilityKind kind) {
    switch (kind) {
        case BN_GPU_IR_UTILITY_RESIDUAL_RMSNORM:
        case BN_GPU_IR_UTILITY_PER_HEAD_RMSNORM:
            return BN_GPU_OP_RMSNORM;
        case BN_GPU_IR_UTILITY_DEINTERLEAVE_Q:
            return BN_GPU_OP_COPY;
        case BN_GPU_IR_UTILITY_BIAS_ADD:
        case BN_GPU_IR_UTILITY_WEIGHTED_ADD:
            return BN_GPU_OP_RESIDUAL;
        default:
            return BN_GPU_OP_UNKNOWN;
    }
}

static inline int bn_gpu_ir_lower_one_to_shader(
    const BnGPUValueGraph *graph,
    const BnGPUIRLoweringMap *map,
    const BnGPUIROp *ir_op,
    BnGPUOp *shader_op) {
    if (!map || !ir_op || !shader_op) return -1;
    memset(shader_op, 0, sizeof(*shader_op));
    shader_op->type = -1;
    shader_op->buf_aux = -1;

    int in0 = ir_op->n_inputs > 0
        ? bn_gpu_ir_lowering_slot(map, ir_op->inputs[0])
        : BN_GPU_IR_NO_SHADER_SLOT;
    int in1 = ir_op->n_inputs > 1
        ? bn_gpu_ir_lowering_slot(map, ir_op->inputs[1])
        : BN_GPU_IR_NO_SHADER_SLOT;
    int out0 = ir_op->n_outputs > 0
        ? bn_gpu_ir_lowering_slot(map, ir_op->outputs[0])
        : BN_GPU_IR_NO_SHADER_SLOT;

    switch (ir_op->kind) {
        case BN_GPU_IR_OP_RMSNORM: {
            const BnGPUIRLoweringValue *weight =
                bn_gpu_ir_lowering_value(map, ir_op->inputs[1]);
            if (in0 < 0 || out0 < 0 || !weight || !weight->weight_buf)
                return -1;
            shader_op->op_kind = BN_GPU_OP_RMSNORM;
            shader_op->op_code = BN_GPU_CODE_RMSNORM;
            shader_op->W_buf = weight->weight_buf;
            shader_op->buf_in = in0;
            shader_op->buf_out = out0;
            shader_op->p[0] = (uint32_t)ir_op->rows;
            shader_op->p[1] = (uint32_t)ir_op->aux0;
            return 0;
        }
        case BN_GPU_IR_OP_COPY:
            if (in0 < 0 || out0 < 0) return -1;
            shader_op->op_kind = BN_GPU_OP_COPY;
            shader_op->op_code = BN_GPU_CODE_COPY;
            shader_op->buf_in = in0;
            shader_op->buf_out = out0;
            shader_op->p[0] = (uint32_t)ir_op->aux0;
            shader_op->p[1] = (uint32_t)ir_op->aux1;
            shader_op->p[2] = (uint32_t)bn_gpu_ir_op_element_count(ir_op);
            return 0;
        case BN_GPU_IR_OP_RESIDUAL_ADD:
            if (in0 < 0 || in1 < 0) return -1;
            shader_op->op_kind = BN_GPU_OP_RESIDUAL;
            shader_op->op_code = BN_GPU_CODE_RESIDUAL_ADD;
            shader_op->buf_in = in0;
            shader_op->buf_out = -1;
            shader_op->buf_aux = in1;
            shader_op->p[0] = (uint32_t)bn_gpu_ir_op_element_count(ir_op);
            return 0;
        case BN_GPU_IR_OP_ACTIVATION: {
            int op_code = bn_gpu_ir_activation_op_code(ir_op);
            if (in0 < 0 || op_code == BN_GPU_CODE_UNKNOWN) return -1;
            if (ir_op->n_inputs > 1 && in1 < 0) return -1;
            shader_op->op_kind = BN_GPU_OP_ACTIVATION;
            shader_op->op_code = op_code;
            shader_op->buf_in = in0;
            shader_op->buf_out = -1;
            shader_op->buf_aux = (ir_op->n_inputs > 1) ? in1 : -1;
            shader_op->p[0] = (uint32_t)bn_gpu_ir_op_element_count(ir_op);
            shader_op->p[1] = (uint32_t)ir_op->aux1;
            return 0;
        }
        case BN_GPU_IR_OP_MATVEC: {
            const BnGPUIRLoweringValue *weight =
                bn_gpu_ir_lowering_value(map, ir_op->inputs[1]);
            if (in0 < 0 || out0 < 0 || !weight || !weight->weight_buf)
                return -1;
            shader_op->op_kind = BN_GPU_OP_MATVEC;
            shader_op->op_code = BN_GPU_CODE_MATVEC;
            shader_op->type = weight->tensor_type;
            shader_op->W_buf = weight->weight_buf;
            shader_op->buf_in = in0;
            shader_op->buf_out = out0;
            shader_op->buf_aux = -1;
            shader_op->rows = ir_op->rows;
            shader_op->cols = ir_op->cols;
            shader_op->p[0] = (uint32_t)ir_op->rows;
            shader_op->p[1] = (uint32_t)ir_op->cols;
            shader_op->p[2] = (uint32_t)(ir_op->aux0 > 0 ? ir_op->aux0 : 1);
            shader_op->p[5] = (uint32_t)ir_op->aux1;
            return 0;
        }
        case BN_GPU_IR_OP_MATVEC_SPLIT: {
            const BnGPUIRLoweringValue *weight =
                bn_gpu_ir_lowering_value(map, ir_op->inputs[1]);
            int out1 = ir_op->n_outputs > 1
                ? bn_gpu_ir_lowering_slot(map, ir_op->outputs[1])
                : BN_GPU_IR_NO_SHADER_SLOT;
            int out2 = ir_op->n_outputs > 2
                ? bn_gpu_ir_lowering_slot(map, ir_op->outputs[2])
                : BN_GPU_IR_NO_SHADER_SLOT;
            int op_code = weight
                ? bn_gpu_quant_split_op_code(weight->tensor_type)
                : BN_GPU_CODE_UNKNOWN;
            if (in0 < 0 || out0 < 0 || out1 < 0 || !weight ||
                !weight->weight_buf || op_code == BN_GPU_CODE_UNKNOWN)
                return -1;
            shader_op->op_kind = BN_GPU_OP_MATVEC;
            shader_op->op_code = op_code;
            shader_op->type = weight->tensor_type;
            shader_op->W_buf = weight->weight_buf;
            shader_op->buf_in = in0;
            shader_op->buf_out = out0;
            shader_op->buf_aux = out1;
            shader_op->rows = out2 >= 0 ? out2 : ir_op->rows;
            shader_op->cols = ir_op->cols;
            shader_op->p[0] = (uint32_t)ir_op->rows;
            shader_op->p[1] = (uint32_t)ir_op->cols;
            shader_op->p[2] = (uint32_t)ir_op->aux0;
            shader_op->p[3] = (uint32_t)ir_op->aux1;
            shader_op->p[6] = (uint32_t)ir_op->output_offsets[1];
            shader_op->p[7] = (uint32_t)ir_op->output_offsets[2];
            return 0;
        }
        case BN_GPU_IR_OP_FFN: {
            const BnGPUIRLoweringValue *weight =
                bn_gpu_ir_lowering_value(map, ir_op->inputs[1]);
            if (in0 < 0 || out0 < 0 || !weight || !weight->weight_buf)
                return -1;
            if ((BnGPUIRActivationKind)ir_op->aux1 !=
                BN_GPU_IR_ACTIVATION_SILU)
                return -1;
            shader_op->op_kind = BN_GPU_OP_FFN;
            shader_op->op_code = BN_GPU_CODE_FUSED_GATEUP_SILU;
            shader_op->type = weight->tensor_type;
            shader_op->W_buf = weight->weight_buf;
            shader_op->buf_in = in0;
            shader_op->buf_out = out0;
            shader_op->buf_aux = -1;
            shader_op->rows = ir_op->rows;
            shader_op->cols = ir_op->cols;
            shader_op->p[0] = (uint32_t)(ir_op->rows + ir_op->aux0);
            shader_op->p[1] = (uint32_t)ir_op->cols;
            shader_op->p[2] = (uint32_t)ir_op->rows;
            return 0;
        }
        case BN_GPU_IR_OP_ROPE:
            if (in0 < 0) return -1;
            if (ir_op->n_inputs > 1 && in1 < 0) return -1;
            shader_op->op_kind = BN_GPU_OP_ROPE;
            shader_op->op_code = ir_op->n_inputs > 1
                ? BN_GPU_CODE_ROPE_QK
                : BN_GPU_CODE_ROPE;
            shader_op->buf_in = in0;
            shader_op->buf_out = -1;
            shader_op->buf_aux = ir_op->n_inputs > 1 ? in1 : -1;
            memcpy(shader_op->p, ir_op->params, sizeof(shader_op->p));
            return 0;
        case BN_GPU_IR_OP_ATTENTION_SCORES:
            if (in0 < 0) return -1;
            shader_op->op_kind = BN_GPU_OP_ATTENTION;
            shader_op->op_code = BN_GPU_CODE_GQA_SCORES;
            shader_op->buf_in = in0;
            shader_op->buf_out = -1;
            shader_op->buf_aux = -1;
            memcpy(shader_op->p, ir_op->params, sizeof(shader_op->p));
            return 0;
        case BN_GPU_IR_OP_SOFTMAX:
            shader_op->op_kind = BN_GPU_OP_ATTENTION;
            shader_op->op_code = BN_GPU_CODE_SOFTMAX;
            shader_op->buf_in = -1;
            shader_op->buf_out = -1;
            shader_op->buf_aux = -1;
            memcpy(shader_op->p, ir_op->params, sizeof(shader_op->p));
            return 0;
        case BN_GPU_IR_OP_ATTENTION_COMBINE:
            if (ir_op->flags & 1u) {
                if (in0 < 0 || out0 < 0) return -1;
                shader_op->op_kind = BN_GPU_OP_ATTENTION;
                shader_op->op_code = BN_GPU_CODE_FLASH_ATTN;
                shader_op->buf_in = in0;
                shader_op->buf_out = out0;
                shader_op->buf_aux = -1;
            } else {
                if (out0 < 0) return -1;
                shader_op->op_kind = BN_GPU_OP_ATTENTION;
                shader_op->op_code = BN_GPU_CODE_GQA_COMBINE;
                shader_op->buf_in = -1;
                shader_op->buf_out = out0;
                shader_op->buf_aux = -1;
            }
            memcpy(shader_op->p, ir_op->params, sizeof(shader_op->p));
            return 0;
        case BN_GPU_IR_OP_SSM: {
            int op_code = bn_gpu_ir_ssm_op_code((BnGPUIRSSMKind)ir_op->aux0);
            int ssm_in = BN_GPU_IR_NO_SHADER_SLOT;
            int ssm_aux = BN_GPU_IR_NO_SHADER_SLOT;
            const BnGPUIRLoweringValue *weight = NULL;
            for (int i = 0; i < ir_op->n_inputs; i++) {
                int value_id = ir_op->inputs[i];
                if (!graph || !bn_gpu_value_graph_has_value(graph, value_id))
                    return -1;
                if (graph->values[value_id].kind == BN_GPU_IR_VALUE_WEIGHT) {
                    weight = bn_gpu_ir_lowering_value(map, value_id);
                } else if (ssm_in == BN_GPU_IR_NO_SHADER_SLOT) {
                    ssm_in = bn_gpu_ir_lowering_slot(map, value_id);
                } else if (ssm_aux == BN_GPU_IR_NO_SHADER_SLOT) {
                    ssm_aux = bn_gpu_ir_lowering_slot(map, value_id);
                }
            }
            if (op_code == BN_GPU_CODE_UNKNOWN) return -1;
            if (ssm_in < 0) return -1;
            if (weight && !weight->weight_buf)
                return -1;
            shader_op->op_kind = BN_GPU_OP_SSM;
            shader_op->op_code = op_code;
            shader_op->W_buf = weight ? weight->weight_buf : NULL;
            shader_op->buf_in = ssm_in;
            shader_op->buf_out = (ir_op->flags & 1u) && out0 >= 0
                ? out0
                : -1;
            shader_op->buf_aux = ssm_aux >= 0 ? ssm_aux : -1;
            shader_op->rows = ir_op->rows;
            memcpy(shader_op->p, ir_op->params, sizeof(shader_op->p));
            return 0;
        }
        case BN_GPU_IR_OP_UTILITY: {
            BnGPUIRUtilityKind kind = (BnGPUIRUtilityKind)ir_op->aux0;
            int op_code = bn_gpu_ir_utility_op_code(kind);
            int op_kind = bn_gpu_ir_utility_op_kind(kind);
            int util_in = BN_GPU_IR_NO_SHADER_SLOT;
            int util_aux = BN_GPU_IR_NO_SHADER_SLOT;
            const BnGPUIRLoweringValue *weight = NULL;
            for (int i = 0; i < ir_op->n_inputs; i++) {
                int value_id = ir_op->inputs[i];
                if (!graph || !bn_gpu_value_graph_has_value(graph, value_id))
                    return -1;
                if (graph->values[value_id].kind == BN_GPU_IR_VALUE_WEIGHT) {
                    weight = bn_gpu_ir_lowering_value(map, value_id);
                } else if (util_in == BN_GPU_IR_NO_SHADER_SLOT) {
                    util_in = bn_gpu_ir_lowering_slot(map, value_id);
                } else if (util_aux == BN_GPU_IR_NO_SHADER_SLOT) {
                    util_aux = bn_gpu_ir_lowering_slot(map, value_id);
                }
            }
            if (op_code == BN_GPU_CODE_UNKNOWN ||
                op_kind == BN_GPU_OP_UNKNOWN ||
                util_in < 0)
                return -1;
            if (weight && !weight->weight_buf)
                return -1;
            shader_op->op_kind = op_kind;
            shader_op->op_code = op_code;
            shader_op->W_buf = weight ? weight->weight_buf : NULL;
            shader_op->buf_in = util_in;
            shader_op->buf_out = (ir_op->flags & 1u) && out0 >= 0
                ? out0
                : -1;
            shader_op->buf_aux = util_aux >= 0 ? util_aux : -1;
            shader_op->rows = ir_op->rows;
            memcpy(shader_op->p, ir_op->params, sizeof(shader_op->p));
            return 0;
        }
        case BN_GPU_IR_OP_LOGITS: {
            const BnGPUIRLoweringValue *weight =
                bn_gpu_ir_lowering_value(map, ir_op->inputs[1]);
            if (in0 < 0 || out0 < 0 || !weight || !weight->weight_buf)
                return -1;
            uint32_t tgs = ((uint32_t)ir_op->rows + 31u) / 32u;
            shader_op->op_kind = BN_GPU_OP_LOGITS;
            shader_op->op_code = BN_GPU_CODE_MATVEC;
            shader_op->type = weight->tensor_type;
            shader_op->W_buf = weight->weight_buf;
            shader_op->buf_in = in0;
            shader_op->buf_out = out0;
            shader_op->rows = ir_op->rows;
            shader_op->cols = ir_op->cols;
            shader_op->p[0] = (uint32_t)ir_op->rows;
            shader_op->p[1] = (uint32_t)ir_op->cols;
            shader_op->p[2] = 1;
            shader_op->p[3] = (tgs > 65535u) ? 65535u : 0u;
            return 0;
        }
        default:
            return -1;
    }
}

static inline int bn_gpu_value_graph_lower_to_shader(
    const BnGPUValueGraph *graph,
    const BnGPUIRLoweringMap *map,
    BnGPUOp *ops,
    int cap_ops,
    int *n_ops) {
    if (!graph || !map || !ops || !n_ops || cap_ops < graph->n_ops)
        return -1;
    int n = 0;
    for (int i = 0; i < graph->n_ops; i++) {
        if (bn_gpu_ir_lower_one_to_shader(graph, map, &graph->ops[i],
                                          &ops[n]) != 0)
            return -1;
        n++;
    }
    *n_ops = n;
    return 0;
}

#endif // BN_GPU_GRAPH_LOWERING_INTERNAL_H
