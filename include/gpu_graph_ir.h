#ifndef BN_GPU_GRAPH_IR_H
#define BN_GPU_GRAPH_IR_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define BN_GPU_IR_INVALID_VALUE (-1)
#define BN_GPU_IR_MAX_OP_INPUTS 4
#define BN_GPU_IR_MAX_OP_OUTPUTS 3

typedef enum {
    BN_GPU_IR_VALUE_TRANSIENT = 0,
    BN_GPU_IR_VALUE_MODEL_INPUT,
    BN_GPU_IR_VALUE_MODEL_OUTPUT,
    BN_GPU_IR_VALUE_WEIGHT,
    BN_GPU_IR_VALUE_KV_CACHE,
    BN_GPU_IR_VALUE_SSM_STATE,
    BN_GPU_IR_VALUE_SCRATCH,
} BnGPUIRValueKind;

typedef enum {
    BN_GPU_IR_VALUE_READABLE = 1u << 0,
    BN_GPU_IR_VALUE_WRITABLE = 1u << 1,
    BN_GPU_IR_VALUE_ALIAS = 1u << 2,
    BN_GPU_IR_VALUE_EXTERNAL = 1u << 3,
} BnGPUIRValueFlags;

typedef enum {
    BN_GPU_IR_OP_UNKNOWN = 0,
    BN_GPU_IR_OP_MATVEC,
    BN_GPU_IR_OP_MATVEC_SPLIT,
    BN_GPU_IR_OP_RMSNORM,
    BN_GPU_IR_OP_ROPE,
    BN_GPU_IR_OP_ATTENTION_SCORES,
    BN_GPU_IR_OP_SOFTMAX,
    BN_GPU_IR_OP_ATTENTION_COMBINE,
    BN_GPU_IR_OP_ACTIVATION,
    BN_GPU_IR_OP_RESIDUAL_ADD,
    BN_GPU_IR_OP_COPY,
    BN_GPU_IR_OP_UTILITY,
    BN_GPU_IR_OP_FFN,
    BN_GPU_IR_OP_SSM,
    BN_GPU_IR_OP_LOGITS,
    BN_GPU_IR_OP_FALLBACK_BOUNDARY,
} BnGPUIROpKind;

typedef enum {
    BN_GPU_IR_ACTIVATION_NONE = 0,
    BN_GPU_IR_ACTIVATION_SILU,
    BN_GPU_IR_ACTIVATION_RELU2,
    BN_GPU_IR_ACTIVATION_SIGMOID,
} BnGPUIRActivationKind;

typedef enum {
    BN_GPU_IR_SSM_UNKNOWN = 0,
    BN_GPU_IR_SSM_CONV_SILU,
    BN_GPU_IR_SSM_L2NORM,
    BN_GPU_IR_SSM_ALPHA_BETA,
    BN_GPU_IR_SSM_ALPHA_BETA_SPLIT,
    BN_GPU_IR_SSM_DELTA,
    BN_GPU_IR_SSM_GATE,
} BnGPUIRSSMKind;

typedef enum {
    BN_GPU_IR_UTILITY_UNKNOWN = 0,
    BN_GPU_IR_UTILITY_BIAS_ADD,
    BN_GPU_IR_UTILITY_WEIGHTED_ADD,
    BN_GPU_IR_UTILITY_RESIDUAL_RMSNORM,
    BN_GPU_IR_UTILITY_PER_HEAD_RMSNORM,
    BN_GPU_IR_UTILITY_DEINTERLEAVE_Q,
} BnGPUIRUtilityKind;

typedef enum {
    BN_GPU_IR_FALLBACK_NONE = 0,
    BN_GPU_IR_FALLBACK_UNSUPPORTED_OP,
    BN_GPU_IR_FALLBACK_UNSUPPORTED_QUANT,
    BN_GPU_IR_FALLBACK_MISSING_BACKEND_HANDLE,
    BN_GPU_IR_FALLBACK_UNSUPPORTED_LAYOUT,
    BN_GPU_IR_FALLBACK_UNSUPPORTED_ARCH,
    BN_GPU_IR_FALLBACK_BACKEND_RESOURCE,
    BN_GPU_IR_FALLBACK_VALIDATION,
} BnGPUIRFallbackReason;

typedef struct {
    int id;
    BnGPUIRValueKind kind;
    int elem_type;
    int rows;
    int cols;
    int alias_of;
    uint32_t flags;
    const char *name;
} BnGPUIRValue;

typedef struct {
    BnGPUIRFallbackReason reason;
    const char *detail;
} BnGPUIRFallback;

typedef struct {
    int id;
    BnGPUIROpKind kind;
    int inputs[BN_GPU_IR_MAX_OP_INPUTS];
    int n_inputs;
    int outputs[BN_GPU_IR_MAX_OP_OUTPUTS];
    int n_outputs;
    int rows;
    int cols;
    int aux0;
    int aux1;
    int output_offsets[BN_GPU_IR_MAX_OP_OUTPUTS];
    uint32_t params[8];
    uint32_t flags;
    BnGPUIRFallback fallback;
    const char *label;
} BnGPUIROp;

typedef struct BnGPUValueGraph {
    BnGPUIRValue *values;
    int n_values;
    int cap_values;
    BnGPUIROp *ops;
    int n_ops;
    int cap_ops;
} BnGPUValueGraph;

static inline void bn_gpu_value_graph_init(BnGPUValueGraph *graph) {
    if (graph) memset(graph, 0, sizeof(*graph));
}

static inline void bn_gpu_value_graph_clear(BnGPUValueGraph *graph) {
    if (!graph) return;
    graph->n_values = 0;
    graph->n_ops = 0;
}

static inline void bn_gpu_value_graph_free(BnGPUValueGraph *graph) {
    if (!graph) return;
    free(graph->values);
    free(graph->ops);
    memset(graph, 0, sizeof(*graph));
}

static inline int bn_gpu_value_graph_reserve_values(BnGPUValueGraph *graph,
                                                    int needed) {
    if (!graph || needed < 0) return -1;
    if (graph->cap_values >= needed) return 0;
    int new_cap = graph->cap_values ? graph->cap_values * 2 : 16;
    while (new_cap < needed) new_cap *= 2;
    BnGPUIRValue *new_values =
        (BnGPUIRValue *)realloc(graph->values,
                                (size_t)new_cap * sizeof(BnGPUIRValue));
    if (!new_values) return -1;
    graph->values = new_values;
    graph->cap_values = new_cap;
    return 0;
}

static inline int bn_gpu_value_graph_reserve_ops(BnGPUValueGraph *graph,
                                                 int needed) {
    if (!graph || needed < 0) return -1;
    if (graph->cap_ops >= needed) return 0;
    int new_cap = graph->cap_ops ? graph->cap_ops * 2 : 32;
    while (new_cap < needed) new_cap *= 2;
    BnGPUIROp *new_ops =
        (BnGPUIROp *)realloc(graph->ops,
                             (size_t)new_cap * sizeof(BnGPUIROp));
    if (!new_ops) return -1;
    graph->ops = new_ops;
    graph->cap_ops = new_cap;
    return 0;
}

static inline int bn_gpu_value_graph_add_value(BnGPUValueGraph *graph,
                                               BnGPUIRValueKind kind,
                                               int elem_type,
                                               int rows,
                                               int cols,
                                               uint32_t flags,
                                               const char *name) {
    if (bn_gpu_value_graph_reserve_values(graph, graph->n_values + 1) != 0)
        return BN_GPU_IR_INVALID_VALUE;
    int id = graph->n_values;
    graph->values[graph->n_values++] = (BnGPUIRValue){
        .id = id,
        .kind = kind,
        .elem_type = elem_type,
        .rows = rows,
        .cols = cols,
        .alias_of = BN_GPU_IR_INVALID_VALUE,
        .flags = flags,
        .name = name,
    };
    return id;
}

static inline int bn_gpu_value_graph_add_alias(BnGPUValueGraph *graph,
                                               int source_value,
                                               const char *name) {
    if (!graph || source_value < 0 || source_value >= graph->n_values)
        return BN_GPU_IR_INVALID_VALUE;
    BnGPUIRValue source = graph->values[source_value];
    int id = bn_gpu_value_graph_add_value(
        graph, source.kind, source.elem_type, source.rows, source.cols,
        source.flags | BN_GPU_IR_VALUE_ALIAS, name);
    if (id != BN_GPU_IR_INVALID_VALUE)
        graph->values[id].alias_of = source_value;
    return id;
}

static inline BnGPUIROp *bn_gpu_value_graph_add_op(BnGPUValueGraph *graph,
                                                   BnGPUIROpKind kind,
                                                   const char *label) {
    if (bn_gpu_value_graph_reserve_ops(graph, graph->n_ops + 1) != 0)
        return NULL;
    int id = graph->n_ops;
    BnGPUIROp *op = &graph->ops[graph->n_ops++];
    memset(op, 0, sizeof(*op));
    op->id = id;
    op->kind = kind;
    op->label = label;
    for (int i = 0; i < BN_GPU_IR_MAX_OP_INPUTS; i++)
        op->inputs[i] = BN_GPU_IR_INVALID_VALUE;
    for (int i = 0; i < BN_GPU_IR_MAX_OP_OUTPUTS; i++)
        op->outputs[i] = BN_GPU_IR_INVALID_VALUE;
    return op;
}

static inline int bn_gpu_ir_op_add_input(BnGPUIROp *op, int value_id) {
    if (!op || op->n_inputs >= BN_GPU_IR_MAX_OP_INPUTS) return -1;
    op->inputs[op->n_inputs++] = value_id;
    return 0;
}

static inline int bn_gpu_ir_op_add_output(BnGPUIROp *op, int value_id) {
    if (!op || op->n_outputs >= BN_GPU_IR_MAX_OP_OUTPUTS) return -1;
    op->outputs[op->n_outputs++] = value_id;
    return 0;
}

static inline void bn_gpu_ir_op_set_fallback(BnGPUIROp *op,
                                             BnGPUIRFallbackReason reason,
                                             const char *detail) {
    if (!op) return;
    op->fallback.reason = reason;
    op->fallback.detail = detail;
}

static inline int bn_gpu_value_graph_has_value(const BnGPUValueGraph *graph,
                                               int value_id) {
    return graph && value_id >= 0 && value_id < graph->n_values;
}

static inline int bn_gpu_value_graph_add_like(BnGPUValueGraph *graph,
                                              int source_value,
                                              BnGPUIRValueKind kind,
                                              uint32_t flags,
                                              const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, source_value))
        return BN_GPU_IR_INVALID_VALUE;
    BnGPUIRValue source = graph->values[source_value];
    return bn_gpu_value_graph_add_value(graph, kind, source.elem_type,
                                        source.rows, source.cols, flags, name);
}

static inline int bn_gpu_value_graph_add_rmsnorm(BnGPUValueGraph *graph,
                                                 int input_value,
                                                 int norm_weight,
                                                 int dim,
                                                 uint32_t eps_bits,
                                                 const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, input_value) ||
        !bn_gpu_value_graph_has_value(graph, norm_weight))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_like(
        graph, input_value, BN_GPU_IR_VALUE_TRANSIENT,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_RMSNORM,
                                              name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, input_value);
    bn_gpu_ir_op_add_input(op, norm_weight);
    bn_gpu_ir_op_add_output(op, output);
    op->rows = dim;
    op->aux0 = (int)eps_bits;
    return output;
}

static inline int bn_gpu_value_graph_add_copy_region(BnGPUValueGraph *graph,
                                                     int input_value,
                                                     int src_offset,
                                                     int dst_offset,
                                                     int count,
                                                     const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, input_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_like(
        graph, input_value, BN_GPU_IR_VALUE_TRANSIENT,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_COPY, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, input_value);
    bn_gpu_ir_op_add_output(op, output);
    op->rows = count ? count : graph->values[input_value].rows;
    op->cols = count ? 0 : graph->values[input_value].cols;
    op->aux0 = src_offset;
    op->aux1 = dst_offset;
    return output;
}

static inline int bn_gpu_value_graph_add_copy(BnGPUValueGraph *graph,
                                              int input_value,
                                              const char *name) {
    return bn_gpu_value_graph_add_copy_region(graph, input_value, 0, 0, 0,
                                              name);
}

static inline int bn_gpu_value_graph_add_residual_add(BnGPUValueGraph *graph,
                                                      int accum_value,
                                                      int residual_value,
                                                      const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, accum_value) ||
        !bn_gpu_value_graph_has_value(graph, residual_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_alias(graph, accum_value, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    graph->values[output].flags |= BN_GPU_IR_VALUE_WRITABLE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(
        graph, BN_GPU_IR_OP_RESIDUAL_ADD, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, accum_value);
    bn_gpu_ir_op_add_input(op, residual_value);
    bn_gpu_ir_op_add_output(op, output);
    op->rows = graph->values[accum_value].rows;
    op->cols = graph->values[accum_value].cols;
    return output;
}

static inline int bn_gpu_value_graph_add_activation(BnGPUValueGraph *graph,
                                                    int input_value,
                                                    int aux_value,
                                                    BnGPUIRActivationKind kind,
                                                    int in_place,
                                                    const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, input_value))
        return BN_GPU_IR_INVALID_VALUE;
    if (aux_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, aux_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = in_place
        ? bn_gpu_value_graph_add_alias(graph, input_value, name)
        : bn_gpu_value_graph_add_like(
              graph, input_value, BN_GPU_IR_VALUE_TRANSIENT,
              BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    graph->values[output].flags |= BN_GPU_IR_VALUE_WRITABLE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_ACTIVATION,
                                              name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, input_value);
    if (aux_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, aux_value);
    bn_gpu_ir_op_add_output(op, output);
    op->rows = graph->values[input_value].rows;
    op->cols = graph->values[input_value].cols;
    op->aux0 = (int)kind;
    op->flags = in_place ? 1u : 0u;
    return output;
}

static inline int bn_gpu_value_graph_add_matvec(BnGPUValueGraph *graph,
                                                int input_value,
                                                int weight_value,
                                                int rows,
                                                int cols,
                                                int batch,
                                                int output_offset,
                                                const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, input_value) ||
        !bn_gpu_value_graph_has_value(graph, weight_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_TRANSIENT, graph->values[input_value].elem_type,
        batch > 0 ? batch : 1, rows,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_MATVEC,
                                              name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, input_value);
    bn_gpu_ir_op_add_input(op, weight_value);
    bn_gpu_ir_op_add_output(op, output);
    op->rows = rows;
    op->cols = cols;
    op->aux0 = batch > 0 ? batch : 1;
    op->aux1 = output_offset;
    return output;
}

static inline BnGPUIROp *bn_gpu_value_graph_add_matvec_split(
    BnGPUValueGraph *graph,
    int input_value,
    int weight_value,
    int total_rows,
    int cols,
    int split0,
    int split1,
    int output0_cols,
    int output1_cols,
    int output2_cols,
    int output1_offset,
    int output2_offset,
    const char *name0,
    const char *name1,
    const char *name2) {
    if (!bn_gpu_value_graph_has_value(graph, input_value) ||
        !bn_gpu_value_graph_has_value(graph, weight_value))
        return NULL;
    int output0 = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_TRANSIENT, graph->values[input_value].elem_type,
        1, output0_cols > 0 ? output0_cols : split0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name0);
    int output1 = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_TRANSIENT, graph->values[input_value].elem_type,
        1, output1_cols > 0 ? output1_cols : total_rows - split0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name1);
    if (output0 == BN_GPU_IR_INVALID_VALUE ||
        output1 == BN_GPU_IR_INVALID_VALUE)
        return NULL;
    int output2 = BN_GPU_IR_INVALID_VALUE;
    if (name2) {
        output2 = bn_gpu_value_graph_add_value(
            graph, BN_GPU_IR_VALUE_TRANSIENT,
            graph->values[input_value].elem_type, 1, output2_cols,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name2);
        if (output2 == BN_GPU_IR_INVALID_VALUE) return NULL;
    }
    BnGPUIROp *op = bn_gpu_value_graph_add_op(
        graph, BN_GPU_IR_OP_MATVEC_SPLIT, name0);
    if (!op) return NULL;
    bn_gpu_ir_op_add_input(op, input_value);
    bn_gpu_ir_op_add_input(op, weight_value);
    bn_gpu_ir_op_add_output(op, output0);
    bn_gpu_ir_op_add_output(op, output1);
    if (output2 != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_output(op, output2);
    op->rows = total_rows;
    op->cols = cols;
    op->aux0 = split0;
    op->aux1 = split1;
    op->output_offsets[1] = output1_offset;
    op->output_offsets[2] = output2_offset;
    return op;
}

static inline int bn_gpu_value_graph_add_fused_gateup(BnGPUValueGraph *graph,
                                                      int input_value,
                                                      int weight_value,
                                                      int gate_rows,
                                                      int up_rows,
                                                      int cols,
                                                      BnGPUIRActivationKind activation,
                                                      const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, input_value) ||
        !bn_gpu_value_graph_has_value(graph, weight_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_TRANSIENT, graph->values[input_value].elem_type,
        1, gate_rows, BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE,
        name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_FFN,
                                              name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, input_value);
    bn_gpu_ir_op_add_input(op, weight_value);
    bn_gpu_ir_op_add_output(op, output);
    op->rows = gate_rows;
    op->cols = cols;
    op->aux0 = up_rows;
    op->aux1 = (int)activation;
    return output;
}

static inline int bn_gpu_value_graph_add_rope(BnGPUValueGraph *graph,
                                              int q_value,
                                              int k_value,
                                              int n_heads,
                                              int head_size,
                                              int pos,
                                              int rope_dims,
                                              int n_kv_heads,
                                              uint32_t kv_cache_off,
                                              const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, q_value))
        return BN_GPU_IR_INVALID_VALUE;
    if (k_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, k_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_alias(graph, q_value, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    graph->values[output].flags |= BN_GPU_IR_VALUE_WRITABLE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_ROPE, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, q_value);
    if (k_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, k_value);
    bn_gpu_ir_op_add_output(op, output);
    op->params[0] = (uint32_t)n_heads;
    op->params[1] = (uint32_t)head_size;
    op->params[2] = (uint32_t)pos;
    op->params[3] = (uint32_t)rope_dims;
    op->params[4] = (uint32_t)n_kv_heads;
    op->params[5] = kv_cache_off;
    return output;
}

static inline int bn_gpu_value_graph_add_flash_attention(
    BnGPUValueGraph *graph,
    int q_value,
    int n_heads,
    int head_size,
    int n_kv,
    int kv_mul,
    int kv_dim,
    int seq_len,
    uint32_t loff,
    uint32_t inv_sqrt_hs,
    const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, q_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_TRANSIENT, graph->values[q_value].elem_type,
        1, n_heads * head_size,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(
        graph, BN_GPU_IR_OP_ATTENTION_COMBINE, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, q_value);
    bn_gpu_ir_op_add_output(op, output);
    op->flags = 1u;
    op->params[0] = (uint32_t)n_heads;
    op->params[1] = (uint32_t)head_size;
    op->params[2] = (uint32_t)n_kv;
    op->params[3] = (uint32_t)kv_mul;
    op->params[4] = (uint32_t)kv_dim;
    op->params[5] = (uint32_t)seq_len;
    op->params[6] = loff;
    op->params[7] = inv_sqrt_hs;
    return output;
}

static inline int bn_gpu_value_graph_add_attention_scores(
    BnGPUValueGraph *graph,
    int q_value,
    int n_heads,
    int head_size,
    int n_kv,
    int kv_mul,
    int kv_dim,
    int seq_len,
    uint32_t loff,
    uint32_t inv_sqrt_hs,
    const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, q_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_TRANSIENT, graph->values[q_value].elem_type,
        n_heads, n_kv, BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE,
        name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(
        graph, BN_GPU_IR_OP_ATTENTION_SCORES, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, q_value);
    bn_gpu_ir_op_add_output(op, output);
    op->params[0] = (uint32_t)n_heads;
    op->params[1] = (uint32_t)head_size;
    op->params[2] = (uint32_t)n_kv;
    op->params[3] = (uint32_t)kv_mul;
    op->params[4] = (uint32_t)kv_dim;
    op->params[5] = (uint32_t)seq_len;
    op->params[6] = loff;
    op->params[7] = inv_sqrt_hs;
    return output;
}

static inline int bn_gpu_value_graph_add_softmax(BnGPUValueGraph *graph,
                                                 int scores_value,
                                                 int n_heads,
                                                 int n_kv,
                                                 int seq_len,
                                                 const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, scores_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_alias(graph, scores_value, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    graph->values[output].flags |= BN_GPU_IR_VALUE_WRITABLE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_SOFTMAX,
                                              name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, scores_value);
    bn_gpu_ir_op_add_output(op, output);
    op->params[0] = (uint32_t)n_heads;
    op->params[1] = (uint32_t)n_kv;
    op->params[2] = (uint32_t)seq_len;
    return output;
}

static inline int bn_gpu_value_graph_add_attention_combine(
    BnGPUValueGraph *graph,
    int weights_value,
    int n_heads,
    int head_size,
    int n_kv,
    int kv_mul,
    int kv_dim,
    int seq_len,
    uint32_t loff,
    const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, weights_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_TRANSIENT, graph->values[weights_value].elem_type,
        1, n_heads * head_size,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(
        graph, BN_GPU_IR_OP_ATTENTION_COMBINE, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, weights_value);
    bn_gpu_ir_op_add_output(op, output);
    op->params[0] = (uint32_t)n_heads;
    op->params[1] = (uint32_t)head_size;
    op->params[2] = (uint32_t)n_kv;
    op->params[3] = (uint32_t)kv_mul;
    op->params[4] = (uint32_t)kv_dim;
    op->params[5] = (uint32_t)seq_len;
    op->params[6] = loff;
    return output;
}

static inline int bn_gpu_value_graph_add_ssm(BnGPUValueGraph *graph,
                                             int input_value,
                                             int aux_value,
                                             int weight_value,
                                             int output_value,
                                             BnGPUIRSSMKind kind,
                                             int rows,
                                             const uint32_t params[8],
                                             const char *name) {
    if (input_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, input_value))
        return BN_GPU_IR_INVALID_VALUE;
    if (aux_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, aux_value))
        return BN_GPU_IR_INVALID_VALUE;
    if (weight_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, weight_value))
        return BN_GPU_IR_INVALID_VALUE;

    int output = output_value;
    int has_explicit_output = output != BN_GPU_IR_INVALID_VALUE;
    if (output == BN_GPU_IR_INVALID_VALUE && input_value != BN_GPU_IR_INVALID_VALUE) {
        output = bn_gpu_value_graph_add_alias(graph, input_value, name);
        if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
        graph->values[output].flags |= BN_GPU_IR_VALUE_WRITABLE;
    } else if (output != BN_GPU_IR_INVALID_VALUE &&
               !bn_gpu_value_graph_has_value(graph, output)) {
        return BN_GPU_IR_INVALID_VALUE;
    }

    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_SSM, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    if (input_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, input_value);
    if (aux_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, aux_value);
    if (weight_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, weight_value);
    if (output != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_output(op, output);
    op->rows = rows;
    op->aux0 = (int)kind;
    op->flags = has_explicit_output ? 1u : 0u;
    if (params)
        memcpy(op->params, params, sizeof(op->params));
    return output;
}

static inline int bn_gpu_value_graph_add_utility(BnGPUValueGraph *graph,
                                                 int input_value,
                                                 int aux_value,
                                                 int weight_value,
                                                 int output_value,
                                                 BnGPUIRUtilityKind kind,
                                                 int rows,
                                                 const uint32_t params[8],
                                                 const char *name) {
    if (input_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, input_value))
        return BN_GPU_IR_INVALID_VALUE;
    if (aux_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, aux_value))
        return BN_GPU_IR_INVALID_VALUE;
    if (weight_value != BN_GPU_IR_INVALID_VALUE &&
        !bn_gpu_value_graph_has_value(graph, weight_value))
        return BN_GPU_IR_INVALID_VALUE;

    int output = output_value;
    int has_explicit_output = output != BN_GPU_IR_INVALID_VALUE;
    if (output == BN_GPU_IR_INVALID_VALUE && input_value != BN_GPU_IR_INVALID_VALUE) {
        output = bn_gpu_value_graph_add_alias(graph, input_value, name);
        if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
        graph->values[output].flags |= BN_GPU_IR_VALUE_WRITABLE;
    } else if (output != BN_GPU_IR_INVALID_VALUE &&
               !bn_gpu_value_graph_has_value(graph, output)) {
        return BN_GPU_IR_INVALID_VALUE;
    }

    BnGPUIROp *op = bn_gpu_value_graph_add_op(
        graph, BN_GPU_IR_OP_UTILITY, name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    if (input_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, input_value);
    if (aux_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, aux_value);
    if (weight_value != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_input(op, weight_value);
    if (output != BN_GPU_IR_INVALID_VALUE)
        bn_gpu_ir_op_add_output(op, output);
    op->rows = rows;
    op->aux0 = (int)kind;
    op->flags = has_explicit_output ? 1u : 0u;
    if (params)
        memcpy(op->params, params, sizeof(op->params));
    return output;
}

static inline int bn_gpu_value_graph_add_logits(BnGPUValueGraph *graph,
                                                int input_value,
                                                int weight_value,
                                                int vocab_size,
                                                int dim,
                                                const char *name) {
    if (!bn_gpu_value_graph_has_value(graph, input_value) ||
        !bn_gpu_value_graph_has_value(graph, weight_value))
        return BN_GPU_IR_INVALID_VALUE;
    int output = bn_gpu_value_graph_add_value(
        graph, BN_GPU_IR_VALUE_MODEL_OUTPUT, graph->values[input_value].elem_type,
        1, vocab_size, BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE,
        name);
    if (output == BN_GPU_IR_INVALID_VALUE) return BN_GPU_IR_INVALID_VALUE;
    BnGPUIROp *op = bn_gpu_value_graph_add_op(graph, BN_GPU_IR_OP_LOGITS,
                                              name);
    if (!op) return BN_GPU_IR_INVALID_VALUE;
    bn_gpu_ir_op_add_input(op, input_value);
    bn_gpu_ir_op_add_input(op, weight_value);
    bn_gpu_ir_op_add_output(op, output);
    op->rows = vocab_size;
    op->cols = dim;
    return output;
}

#endif // BN_GPU_GRAPH_IR_H
