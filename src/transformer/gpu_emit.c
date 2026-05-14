#include "gpu_internal.h"
#include "backend_quant.h"
#include "../gpu_graph_lowering_internal.h"
#include "../gpu_quant_lowering_internal.h"
#include "../gpu_shader_ir_internal.h"
#include "gpu_moe_bridge.h"
#include "gpu_moe_cache.h"
#include "moe.h"
#include "transformer_backend_internal.h"
#include "session.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GPU_OP(code_) \
    .op_code = (code_)

static int emit_context_reserve_lowering(BnTransformerGPUEmitContext *ctx,
                                         int needed);
static int emit_context_utility(BnTransformerGPUEmitContext *ctx,
                                BnGPUIRUtilityKind kind,
                                int buf_in,
                                int buf_aux,
                                int buf_out,
                                int aux_offset,
                                void *weight,
                                const uint32_t params[8]);

void bn_transformer_gpu_finalize_op_kinds(void *ops, int n) {
    BnGPUOp *shader_ops = (BnGPUOp *)ops;
    for (int i = 0; i < n; i++) {
        if (shader_ops[i].op_kind == BN_GPU_OP_UNKNOWN)
            shader_ops[i].op_kind = bn_gpu_op_kind(&shader_ops[i]);
    }
}

void bn_transformer_gpu_emit_context_init(BnTransformerGPUEmitContext *ctx,
                                          void *lowered_ops,
                                          int cap) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->lowered_ops = lowered_ops;
    ctx->cap = cap;
    ctx->graph = &ctx->graph_storage;
    ctx->owns_graph_storage = 1;
    ctx->owns_lowering_values = 1;
    bn_gpu_value_graph_init(ctx->graph);
}

int bn_transformer_gpu_emit_context_init_session(
    BnTransformerGPUEmitContext *ctx,
    BnBackendSession *backend,
    void *lowered_ops,
    int cap,
    int cap_values,
    int cap_ops) {
    if (!ctx || !backend || cap_values < 0 || cap_ops < 0) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->lowered_ops = lowered_ops;
    ctx->cap = cap;
    ctx->graph = bn_backend_session_gpu_value_graph(backend);
    if (!ctx->graph) return -1;
    bn_gpu_value_graph_clear(ctx->graph);
    if (bn_gpu_value_graph_reserve_values(ctx->graph, cap_values) != 0)
        return -1;
    if (bn_gpu_value_graph_reserve_ops(ctx->graph, cap_ops) != 0)
        return -1;
    if (bn_backend_session_ensure_gpu_lowering_values(
            backend, (int)sizeof(BnGPUIRLoweringValue), cap_values,
            &ctx->lowering_values, &ctx->cap_lowering_values) != 0)
        return -1;
    return 0;
}

int bn_transformer_gpu_emit_context_reserve(
    BnTransformerGPUEmitContext *ctx,
    int cap_values,
    int cap_ops) {
    if (!ctx || cap_values < 0 || cap_ops < 0) return -1;
    if (bn_gpu_value_graph_reserve_values(ctx->graph, cap_values) != 0)
        return -1;
    if (bn_gpu_value_graph_reserve_ops(ctx->graph, cap_ops) != 0)
        return -1;
    if (emit_context_reserve_lowering(ctx, cap_values) != 0)
        return -1;
    return 0;
}

void bn_transformer_gpu_emit_context_free(BnTransformerGPUEmitContext *ctx) {
    if (!ctx) return;
    if (ctx->owns_graph_storage)
        bn_gpu_value_graph_free(ctx->graph);
    else
        bn_gpu_value_graph_clear(ctx->graph);
    if (ctx->owns_lowering_values)
        free(ctx->lowering_values);
    memset(ctx, 0, sizeof(*ctx));
}

static int emit_context_reserve_lowering(BnTransformerGPUEmitContext *ctx,
                                         int needed) {
    if (!ctx || needed < 0) return -1;
    if (ctx->cap_lowering_values >= needed) return 0;
    if (!ctx->owns_lowering_values) return -1;
    int new_cap = ctx->cap_lowering_values ? ctx->cap_lowering_values * 2 : 16;
    while (new_cap < needed) new_cap *= 2;
    BnGPUIRLoweringValue *new_values =
        (BnGPUIRLoweringValue *)realloc(
            ctx->lowering_values,
            (size_t)new_cap * sizeof(BnGPUIRLoweringValue));
    if (!new_values) return -1;
    ctx->lowering_values = new_values;
    ctx->cap_lowering_values = new_cap;
    return 0;
}

static BnGPUIRLoweringValue *emit_context_lowering_values(
    BnTransformerGPUEmitContext *ctx) {
    return (BnGPUIRLoweringValue *)ctx->lowering_values;
}

static int emit_context_add_value(BnTransformerGPUEmitContext *ctx,
                                  BnGPUIRValueKind kind,
                                  int elem_type,
                                  int rows,
                                  int cols,
                                  uint32_t flags,
                                  const char *name,
                                  int shader_slot,
                                  void *weight_buf,
                                  int tensor_type) {
    if (!ctx) return BN_GPU_IR_INVALID_VALUE;
    int needed = ctx->graph->n_values + 1;
    if (emit_context_reserve_lowering(ctx, needed) != 0)
        return BN_GPU_IR_INVALID_VALUE;
    int id = bn_gpu_value_graph_add_value(ctx->graph, kind, elem_type,
                                          rows, cols, flags, name);
    if (id != BN_GPU_IR_INVALID_VALUE) {
        BnGPUIRLoweringValue *lowering_values =
            emit_context_lowering_values(ctx);
        lowering_values[id] = (BnGPUIRLoweringValue){
            .shader_slot = shader_slot,
            .weight_buf = weight_buf,
            .tensor_type = tensor_type,
        };
    }
    return id;
}

int bn_transformer_gpu_emit_context_lower_pending(
    BnTransformerGPUEmitContext *ctx) {
    if (!ctx) return -1;
    if (ctx->graph->n_ops == 0) return 0;
    if (!ctx->lowered_ops || ctx->n < 0 || ctx->cap < ctx->n ||
        ctx->cap - ctx->n < ctx->graph->n_ops)
        return -1;
    BnGPUIRLoweringMap map = {
        .values = (BnGPUIRLoweringValue *)ctx->lowering_values,
        .n_values = ctx->graph->n_values,
    };
    int lowered = 0;
    if (bn_gpu_value_graph_lower_to_shader(ctx->graph, &map,
                                           &((BnGPUOp *)ctx->lowered_ops)[ctx->n],
                                           ctx->cap - ctx->n,
                                           &lowered) != 0)
        return -1;
    ctx->n += lowered;
    bn_gpu_value_graph_clear(ctx->graph);
    return 0;
}

int bn_transformer_gpu_emit_context_execute(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu,
    int readback_buf,
    float *readback,
    int readback_count) {
    if (!ctx) return -1;
    if (bn_transformer_gpu_emit_context_lower_pending(ctx) != 0)
        return -1;
    if (ctx->n == 0)
        return 0;
    int rc = bn_transformer_gpu_execute_ops(gpu, ctx->lowered_ops, ctx->n,
                                            readback_buf, readback,
                                            readback_count);
    if (rc == 0)
        ctx->n = 0;
    return rc;
}

int bn_transformer_gpu_emit_context_flush(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu) {
    if (!ctx) return -1;
    if (ctx->n == 0 && ctx->graph->n_ops == 0)
        return 0;
    return bn_transformer_gpu_emit_context_execute(ctx, gpu, -1, NULL, 0);
}

int bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
    BnTransformerGPUEmitContext *ctx,
    void *norm_gpu,
    int dim,
    uint32_t u_eps) {
    return bn_transformer_gpu_emit_context_rmsnorm(
        ctx, norm_gpu, BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps);
}

int bn_transformer_gpu_emit_context_execute_logits(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu,
    float *logits,
    int vocab_size) {
    return bn_transformer_gpu_emit_context_execute(
        ctx, gpu, BN_GPU_VALUE_LOGITS, logits, vocab_size);
}

int bn_transformer_gpu_emit_context_rmsnorm(BnTransformerGPUEmitContext *ctx,
                                            void *norm_gpu,
                                            int buf_in,
                                            int buf_out,
                                            int dim,
                                            uint32_t u_eps) {
    if (!ctx) return -1;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, dim, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "rmsnorm.in",
        buf_in, NULL, -1);
    int weight = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_WEIGHT, -1, dim, 0, BN_GPU_IR_VALUE_READABLE,
        "rmsnorm.weight", BN_GPU_IR_NO_SHADER_SLOT, norm_gpu, -1);
    int output = bn_gpu_value_graph_add_rmsnorm(ctx->graph, input, weight,
                                                dim, u_eps, "rmsnorm.out");
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output] = (BnGPUIRLoweringValue){
        .shader_slot = buf_out,
        .weight_buf = NULL,
        .tensor_type = -1,
    };
    return 0;
}

int bn_transformer_gpu_emit_context_logits(BnTransformerGPUEmitContext *ctx,
                                           void *logit_gpu_buf,
                                           int logit_type,
                                           int logit_rows,
                                           int logit_cols) {
    if (!ctx) return -1;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, logit_cols, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "logits.in",
        BN_GPU_VALUE_XB, NULL, -1);
    int weight = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_WEIGHT, logit_type, logit_rows, logit_cols,
        BN_GPU_IR_VALUE_READABLE, "logits.weight", BN_GPU_IR_NO_SHADER_SLOT,
        logit_gpu_buf, logit_type);
    int output = bn_gpu_value_graph_add_logits(ctx->graph, input, weight,
                                               logit_rows, logit_cols,
                                               "logits.out");
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output] = (BnGPUIRLoweringValue){
        .shader_slot = BN_GPU_VALUE_LOGITS,
        .weight_buf = NULL,
        .tensor_type = -1,
    };
    return 0;
}

int bn_transformer_gpu_emit_context_copy(BnTransformerGPUEmitContext *ctx,
                                         int buf_in,
                                         int buf_out,
                                         int src_offset,
                                         int dst_offset,
                                         int count) {
    if (!ctx) return -1;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, count, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "copy.in",
        buf_in, NULL, -1);
    int output = bn_gpu_value_graph_add_copy_region(
        ctx->graph, input, src_offset, dst_offset, count, "copy.out");
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output] = (BnGPUIRLoweringValue){
        .shader_slot = buf_out,
        .weight_buf = NULL,
        .tensor_type = -1,
    };
    return 0;
}

int bn_transformer_gpu_emit_context_residual_add(
    BnTransformerGPUEmitContext *ctx,
    int buf_in,
    int buf_aux,
    int count) {
    if (!ctx) return -1;
    int accum = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, count, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE,
        "residual.accum", buf_in, NULL, -1);
    int residual = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, count, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
        "residual.add", buf_aux, NULL, -1);
    int output = bn_gpu_value_graph_add_residual_add(
        ctx->graph, accum, residual, "residual.out");
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output] = (BnGPUIRLoweringValue){
        .shader_slot = buf_in,
        .weight_buf = NULL,
        .tensor_type = -1,
    };
    return 0;
}

static void emit_context_residual_rmsnorm(BnTransformerGPUEmitContext *ctx,
                                          int x_buf,
                                          int residual_buf,
                                          int out_buf,
                                          int dim,
                                          uint32_t u_eps,
                                          void *norm_weight) {
    if (getenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM")) {
        bn_transformer_gpu_emit_context_residual_add(
            ctx, x_buf, residual_buf, dim);
        bn_transformer_gpu_emit_context_rmsnorm(
            ctx, norm_weight, x_buf, out_buf, dim, u_eps);
        return;
    }

    uint32_t residual_norm_params[8] = {
        (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0
    };
    emit_context_utility(ctx, BN_GPU_IR_UTILITY_RESIDUAL_RMSNORM,
                         x_buf, residual_buf, out_buf, 0, norm_weight,
                         residual_norm_params);
}

int bn_transformer_gpu_emit_context_activation(
    BnTransformerGPUEmitContext *ctx,
    int buf_in,
    int buf_aux,
    int count,
    int param1,
    BnGPUIRActivationKind kind) {
    if (!ctx) return -1;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, count, 0,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE,
        "activation.in", buf_in, NULL, -1);
    int aux = BN_GPU_IR_INVALID_VALUE;
    if (buf_aux >= 0) {
        aux = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, count, 0,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
            "activation.aux", buf_aux, NULL, -1);
    }
    int output = bn_gpu_value_graph_add_activation(
        ctx->graph, input, aux, kind, 1, "activation.out");
    if (ctx->graph->n_ops > 0)
        ctx->graph->ops[ctx->graph->n_ops - 1].aux1 = param1;
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output] = (BnGPUIRLoweringValue){
        .shader_slot = buf_in,
        .weight_buf = NULL,
        .tensor_type = -1,
    };
    return 0;
}

static int emit_context_matvec_flags(BnTransformerGPUEmitContext *ctx,
                                     int type,
                                     void *weight_buf,
                                     int buf_in,
                                     int buf_out,
                                     int rows,
                                     int cols,
                                     int output_offset,
                                     uint32_t flags) {
    if (!ctx) return -1;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, cols,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "matvec.in",
        buf_in, NULL, -1);
    int weight = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_WEIGHT, type, rows, cols,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "matvec.weight",
        BN_GPU_IR_NO_SHADER_SLOT, weight_buf, type);
    int output = bn_gpu_value_graph_add_matvec(
        ctx->graph, input, weight, rows, cols, 1, output_offset,
        "matvec.out");
    if (output != BN_GPU_IR_INVALID_VALUE && ctx->graph->n_ops > 0)
        ctx->graph->ops[ctx->graph->n_ops - 1].flags |= flags;
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output] = (BnGPUIRLoweringValue){
        .shader_slot = buf_out,
        .weight_buf = NULL,
        .tensor_type = -1,
    };
    return 0;
}

int bn_transformer_gpu_emit_context_matvec(BnTransformerGPUEmitContext *ctx,
                                           int type,
                                           void *weight_buf,
                                           int buf_in,
                                           int buf_out,
                                           int rows,
                                           int cols,
                                           int output_offset) {
    return emit_context_matvec_flags(ctx, type, weight_buf, buf_in, buf_out,
                                     rows, cols, output_offset, 0);
}

int bn_transformer_gpu_emit_context_fused_gateup_silu(
    BnTransformerGPUEmitContext *ctx,
    int type,
    void *weight_buf,
    int buf_in,
    int buf_out,
    int gate_rows,
    int up_rows,
    int cols,
    int use_q4_q8) {
    if (!ctx) return -1;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, cols,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
        "fused_gateup.in", buf_in, NULL, -1);
    int weight = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_WEIGHT, type, gate_rows + up_rows, cols,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
        "fused_gateup.weight", BN_GPU_IR_NO_SHADER_SLOT, weight_buf, type);
    int output = bn_gpu_value_graph_add_fused_gateup(
        ctx->graph, input, weight, gate_rows, up_rows, cols,
        BN_GPU_IR_ACTIVATION_SILU, "fused_gateup.out");
    if (output != BN_GPU_IR_INVALID_VALUE && ctx->graph->n_ops > 0 &&
        use_q4_q8)
        ctx->graph->ops[ctx->graph->n_ops - 1].flags |= 1u;
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output] = (BnGPUIRLoweringValue){
        .shader_slot = buf_out,
        .weight_buf = NULL,
        .tensor_type = -1,
    };
    return 0;
}

static int emit_context_matvec_split(BnTransformerGPUEmitContext *ctx,
                                     int type,
                                     void *weight_buf,
                                     int buf_in,
                                     int buf_out0,
                                     int buf_out1,
                                     int buf_out2,
                                     int total_rows,
                                     int cols,
                                     int split0,
                                     int split1,
                                     int output1_offset,
                                     int output2_offset,
                                     int use_q4_q8) {
    if (!ctx) return -1;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, cols,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
        "matvec_split.in", buf_in, NULL, -1);
    int weight = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_WEIGHT, type, total_rows, cols,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
        "matvec_split.weight", BN_GPU_IR_NO_SHADER_SLOT, weight_buf, type);
    BnGPUIROp *op = bn_gpu_value_graph_add_matvec_split(
        ctx->graph, input, weight, total_rows, cols, split0, split1,
        split0, total_rows - split0,
        split1 > split0 ? total_rows - split1 : 0, output1_offset,
        output2_offset, "matvec_split.out0", "matvec_split.out1",
        buf_out2 >= 0 ? "matvec_split.out2" : NULL);
    if (!op || emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    if (use_q4_q8)
        op->flags |= 1u;
    BnGPUIRLoweringValue *lowering_values = emit_context_lowering_values(ctx);
    lowering_values[op->outputs[0]].shader_slot = buf_out0;
    lowering_values[op->outputs[1]].shader_slot = buf_out1;
    if (op->n_outputs > 2)
        lowering_values[op->outputs[2]].shader_slot = buf_out2;
    return 0;
}

static int emit_context_rope(BnTransformerGPUEmitContext *ctx,
                             int buf_q,
                             int buf_k,
                             int n_heads,
                             int head_size,
                             int pos,
                             int rope_dims,
                             int n_kv_heads,
                             uint32_t kv_cache_off) {
    if (!ctx) return -1;
    int q = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, n_heads * head_size,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "rope.q",
        buf_q, NULL, -1);
    int k = BN_GPU_IR_INVALID_VALUE;
    if (buf_k >= 0) {
        k = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_KV_CACHE, -1, 1, n_kv_heads * head_size,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "rope.k",
            buf_k, NULL, -1);
    }
    int output = bn_gpu_value_graph_add_rope(
        ctx->graph, q, k, n_heads, head_size, pos, rope_dims, n_kv_heads,
        kv_cache_off, "rope.out");
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output].shader_slot = buf_q;
    return 0;
}

static int emit_context_flash_attention(BnTransformerGPUEmitContext *ctx,
                                        int buf_q,
                                        int buf_out,
                                        int n_heads,
                                        int head_size,
                                        int n_kv,
                                        int kv_mul,
                                        int kv_dim,
                                        int seq_len,
                                        size_t loff,
                                        uint32_t u_inv_sqrt_hs) {
    if (!ctx) return -1;
    int q = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, n_heads * head_size,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
        "flash_attn.q", buf_q, NULL, -1);
    int output = bn_gpu_value_graph_add_flash_attention(
        ctx->graph, q, n_heads, head_size, n_kv, kv_mul, kv_dim, seq_len,
        (uint32_t)loff, u_inv_sqrt_hs, "flash_attn.out");
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[output].shader_slot = buf_out;
    return 0;
}

static int emit_context_gqa_attention(BnTransformerGPUEmitContext *ctx,
                                      int buf_q,
                                      int buf_out,
                                      int n_heads,
                                      int head_size,
                                      int n_kv,
                                      int kv_mul,
                                      int kv_dim,
                                      int seq_len,
                                      size_t loff,
                                      uint32_t u_inv_sqrt_hs) {
    if (!ctx) return -1;
    int q = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, n_heads * head_size,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL, "gqa.q",
        buf_q, NULL, -1);
    int scores = bn_gpu_value_graph_add_attention_scores(
        ctx->graph, q, n_heads, head_size, n_kv, kv_mul, kv_dim, seq_len,
        (uint32_t)loff, u_inv_sqrt_hs, "gqa.scores");
    int probs = bn_gpu_value_graph_add_softmax(
        ctx->graph, scores, n_heads, n_kv, seq_len, "gqa.softmax");
    int output = bn_gpu_value_graph_add_attention_combine(
        ctx->graph, probs, n_heads, head_size, n_kv, kv_mul, kv_dim, seq_len,
        (uint32_t)loff, "gqa.out");
    if (output == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    BnGPUIRLoweringValue *lowering_values = emit_context_lowering_values(ctx);
    lowering_values[scores].shader_slot = BN_GPU_VALUE_ATT;
    lowering_values[probs].shader_slot = BN_GPU_VALUE_ATT;
    lowering_values[output].shader_slot = buf_out;
    return 0;
}

static int emit_context_ssm(BnTransformerGPUEmitContext *ctx,
                            BnGPUIRSSMKind kind,
                            int buf_in,
                            int buf_aux,
                            int buf_out,
                            int rows,
                            void *weight_buf,
                            const uint32_t params[8]) {
    if (!ctx) return -1;
    int elem_count = params ? (int)params[0] : 0;
    int input = BN_GPU_IR_INVALID_VALUE;
    if (buf_in >= 0) {
        input = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, elem_count,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "ssm.in",
            buf_in, NULL, -1);
    }
    int aux = BN_GPU_IR_INVALID_VALUE;
    if (buf_aux >= 0) {
        aux = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, elem_count,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "ssm.aux",
            buf_aux, NULL, -1);
    }
    int weight = BN_GPU_IR_INVALID_VALUE;
    if (weight_buf) {
        weight = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_WEIGHT, -1, 1, 0,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
            "ssm.weight", BN_GPU_IR_NO_SHADER_SLOT, weight_buf, -1);
    }
    int output = BN_GPU_IR_INVALID_VALUE;
    if (buf_out >= 0 && buf_out != buf_in) {
        output = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, elem_count,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "ssm.out",
            buf_out, NULL, -1);
    }
    int result = bn_gpu_value_graph_add_ssm(
        ctx->graph, input, aux, weight, output, kind, rows, params, "ssm");
    if (result == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[result].shader_slot =
        buf_out >= 0 ? buf_out : buf_in;
    return 0;
}

static int emit_context_utility(BnTransformerGPUEmitContext *ctx,
                                BnGPUIRUtilityKind kind,
                                int buf_in,
                                int buf_aux,
                                int buf_out,
                                int rows,
                                void *weight_buf,
                                const uint32_t params[8]) {
    if (!ctx) return -1;
    int elem_count = params ? (int)params[0] : 0;
    int input = emit_context_add_value(
        ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, elem_count,
        BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE, "utility.in",
        buf_in, NULL, -1);
    int aux = BN_GPU_IR_INVALID_VALUE;
    if (buf_aux >= 0) {
        aux = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, elem_count,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE,
            "utility.aux", buf_aux, NULL, -1);
    }
    int weight = BN_GPU_IR_INVALID_VALUE;
    if (weight_buf) {
        weight = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_WEIGHT, -1, 1, 0,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_EXTERNAL,
            "utility.weight", BN_GPU_IR_NO_SHADER_SLOT, weight_buf, -1);
    }
    int output = BN_GPU_IR_INVALID_VALUE;
    if (buf_out >= 0 && buf_out != buf_in) {
        output = emit_context_add_value(
            ctx, BN_GPU_IR_VALUE_TRANSIENT, -1, 1, elem_count,
            BN_GPU_IR_VALUE_READABLE | BN_GPU_IR_VALUE_WRITABLE,
            "utility.out", buf_out, NULL, -1);
    }
    int result = bn_gpu_value_graph_add_utility(
        ctx->graph, input, aux, weight, output, kind, rows, params,
        "utility");
    if (result == BN_GPU_IR_INVALID_VALUE ||
        emit_context_reserve_lowering(ctx, ctx->graph->n_values) != 0)
        return -1;
    emit_context_lowering_values(ctx)[result].shader_slot =
        buf_out >= 0 ? buf_out : buf_in;
    return 0;
}

int bn_transformer_gpu_execute_ops(const BnGPUBackend *gpu,
                                   void *ops,
                                   int n,
                                   int readback_buf,
                                   float *readback,
                                   int readback_count) {
    if (!gpu || !gpu->execute || !ops || n < 0) return -1;
    bn_transformer_gpu_finalize_op_kinds(ops, n);
    return gpu->execute(gpu->ctx, ops, n, readback_buf, readback,
                        readback_count);
}

int bn_transformer_gpu_write_x(const BnGPUBackend *gpu,
                               const float *x,
                               size_t size_bytes) {
    if (!gpu || !gpu->write_activation || !x) return -1;
    return gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, x, size_bytes, 0);
}

int bn_transformer_gpu_read_x(const BnGPUBackend *gpu,
                              float *x,
                              size_t size_bytes) {
    if (!gpu || !gpu->read_activation || !x) return -1;
    return gpu->read_activation(gpu->ctx, BN_GPU_VALUE_X, x, size_bytes, 0);
}

int bn_transformer_gpu_read_xb(const BnGPUBackend *gpu,
                               float *xb,
                               size_t size_bytes) {
    if (!gpu || !gpu->read_activation || !xb) return -1;
    return gpu->read_activation(gpu->ctx, BN_GPU_VALUE_XB, xb, size_bytes, 0);
}

int bn_transformer_gpu_read_xb2(const BnGPUBackend *gpu,
                                float *xb2,
                                size_t size_bytes) {
    if (!gpu || !gpu->read_activation || !xb2) return -1;
    return gpu->read_activation(gpu->ctx, BN_GPU_VALUE_XB2, xb2, size_bytes, 0);
}

int bn_transformer_gpu_read_activation_buf(const BnGPUBackend *gpu,
                                           int buf_idx,
                                           float *out,
                                           size_t size_bytes) {
    if (!gpu || !gpu->read_activation || !out) return -1;
    return gpu->read_activation(gpu->ctx, buf_idx, out, size_bytes, 0);
}

int bn_transformer_gpu_read_activation_buf_offset(const BnGPUBackend *gpu,
                                                  int buf_idx,
                                                  float *out,
                                                  size_t size_bytes,
                                                  size_t offset_bytes) {
    if (!gpu || !gpu->read_activation || !out) return -1;
    return gpu->read_activation(gpu->ctx, buf_idx, out, size_bytes,
                                offset_bytes);
}

void bn_transformer_gpu_emit_context_dense_ffn(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    const BnTransformerGPUDenseFFNResources *res,
    int dim,
    uint32_t u_eps,
    void *next_norm,
    int skip_down,
    int *down_input_buf,
    int use_q4_q8) {
    int hidden_dim = ffn_plan->hidden_dim;
    void *gateup_stacked = res ? res->gateup_stacked : NULL;
    void *ffn_sub_norm = res ? res->ffn_sub_norm : NULL;

    if (ffn_plan->has_gate && lw->ffn.ffn_gate.data) {
        int use_fused_gateup = !getenv("BN_GPU_DISABLE_FUSED_GATEUP") &&
                               gateup_stacked &&
                               bn_transformer_gpu_can_fused_gateup_silu(res->gpu, lw->ffn.ffn_gate.type,
                                                                         ffn_plan->activation);
        if (use_fused_gateup) {
            bn_transformer_gpu_emit_context_fused_gateup_silu(
                ctx, lw->ffn.ffn_gate.type, gateup_stacked,
                BN_GPU_VALUE_XB, BN_GPU_VALUE_HB, lw->ffn.ffn_gate.rows,
                lw->ffn.ffn_up.rows, lw->ffn.ffn_gate.cols, use_q4_q8);
        } else if (!getenv("BN_GPU_DISABLE_GATEUP_SPLIT") &&
                   gateup_stacked &&
                   lw->ffn.ffn_gate.rows == lw->ffn.ffn_up.rows &&
                   lw->ffn.ffn_gate.cols == lw->ffn.ffn_up.cols &&
                   ffn_plan->activation != 1 &&
                   bn_gpu_quant_split_op_code(lw->ffn.ffn_gate.type) ==
                       BN_GPU_CODE_Q4K_MATVEC_SPLIT &&
                   bn_transformer_gpu_can_matvec_split(res->gpu, lw->ffn.ffn_gate.type)) {
            int total_rows = lw->ffn.ffn_gate.rows + lw->ffn.ffn_up.rows;
            emit_context_matvec_split(
                ctx, lw->ffn.ffn_gate.type, gateup_stacked,
                BN_GPU_VALUE_XB, BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2, -1,
                total_rows, lw->ffn.ffn_gate.cols, lw->ffn.ffn_gate.rows, 1,
                0, 0, use_q4_q8);
        } else if (!getenv("BN_GPU_DISABLE_GATEUP_SPLIT") &&
                   gateup_stacked &&
                   lw->ffn.ffn_gate.rows == lw->ffn.ffn_up.rows &&
                   lw->ffn.ffn_gate.cols == lw->ffn.ffn_up.cols &&
                   bn_transformer_gpu_can_matvec_split(res->gpu, lw->ffn.ffn_gate.type)) {
            int total_rows = lw->ffn.ffn_gate.rows + lw->ffn.ffn_up.rows;
            emit_context_matvec_split(
                ctx, lw->ffn.ffn_gate.type, gateup_stacked,
                BN_GPU_VALUE_XB, BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2,
                BN_GPU_VALUE_HB2, total_rows, lw->ffn.ffn_gate.cols,
                lw->ffn.ffn_gate.rows, 0, 0, 0, use_q4_q8);
            bn_transformer_gpu_emit_context_activation(
                ctx, BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2, hidden_dim, 0,
                BN_GPU_IR_ACTIVATION_SILU);
        } else {
            emit_context_matvec_flags(
                ctx, lw->ffn.ffn_gate.type,
                res ? res->ffn_gate : NULL,
                BN_GPU_VALUE_XB, BN_GPU_VALUE_HB, lw->ffn.ffn_gate.rows,
                lw->ffn.ffn_gate.cols, 0, use_q4_q8 ? 1u : 0u);
            emit_context_matvec_flags(
                ctx, lw->ffn.ffn_up.type,
                res ? res->ffn_up : NULL,
                BN_GPU_VALUE_XB, BN_GPU_VALUE_HB2, lw->ffn.ffn_up.rows,
                lw->ffn.ffn_up.cols, 0, use_q4_q8 ? 1u : 0u);
            BnGPUIRActivationKind act_kind = (ffn_plan->activation == 1)
                ? BN_GPU_IR_ACTIVATION_RELU2
                : BN_GPU_IR_ACTIVATION_SILU;
            bn_transformer_gpu_emit_context_activation(
                ctx, BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2, hidden_dim, 0,
                act_kind);
        }
    } else {
        emit_context_matvec_flags(
            ctx, lw->ffn.ffn_up.type,
            res ? res->ffn_up : NULL, BN_GPU_VALUE_XB,
            BN_GPU_VALUE_HB, lw->ffn.ffn_up.rows, lw->ffn.ffn_up.cols, 0,
            use_q4_q8 ? 1u : 0u);
        BnGPUIRActivationKind act_kind = (ffn_plan->activation == 1)
            ? BN_GPU_IR_ACTIVATION_RELU2
            : BN_GPU_IR_ACTIVATION_SILU;
        bn_transformer_gpu_emit_context_activation(
            ctx, BN_GPU_VALUE_HB, -1, hidden_dim, 0, act_kind);
    }

    int down_in_buf = BN_GPU_VALUE_HB;
    if (ffn_sub_norm) {
        bn_transformer_gpu_emit_context_rmsnorm(
            ctx, ffn_sub_norm, BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2, hidden_dim,
            u_eps);
        down_in_buf = BN_GPU_VALUE_HB2;
    }
    if (down_input_buf)
        *down_input_buf = down_in_buf;
    if (skip_down) {
        (void)next_norm;
        (void)c;
        return;
    }

    emit_context_matvec_flags(
        ctx, lw->ffn.ffn_down.type,
        res ? res->ffn_down : NULL, down_in_buf,
        BN_GPU_VALUE_XB2, lw->ffn.ffn_down.rows, lw->ffn.ffn_down.cols, 0,
        use_q4_q8 ? 1u : 0u);

    emit_context_residual_rmsnorm(
        ctx, BN_GPU_VALUE_X, BN_GPU_VALUE_XB2, BN_GPU_VALUE_XB, dim, u_eps,
        next_norm);

    (void)c;
}

void bn_transformer_gpu_emit_context_qkv(BnTransformerGPUEmitContext *ctx,
                                         const BnConfig *c,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         const BnTransformerGPUQKVResources *res,
                                         int pos,
                                         int q_dim,
                                         int head_size,
                                         int n_heads,
                                         int kv_dim,
                                         int rope_dims,
                                         uint32_t kv_cache_off,
                                         uint32_t u_eps,
                                         int use_q4_q8) {
    void *q_bias = res ? res->q_bias : NULL;
    void *k_bias = res ? res->k_bias : NULL;
    void *v_bias = res ? res->v_bias : NULL;
    void *q_norm = res ? res->q_norm : NULL;
    void *k_norm = res ? res->k_norm : NULL;
    void *qkv_stacked = res ? res->qkv_stacked : NULL;

    int use_packed_qkv = lw->ssm.wqkv.data &&
                         res && res->packed_qkv &&
                         !q_bias && !k_bias && !v_bias &&
                         lw->ssm.wqkv.rows == q_dim + 2 * kv_dim;
    int q_gated = !use_packed_qkv && plan->q_gated;
    int packed_split_op_code = bn_gpu_quant_split_op_code(lw->ssm.wqkv.type);
    int use_packed_q5_split =
        use_packed_qkv &&
        packed_split_op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT &&
        bn_transformer_gpu_can_matvec_split(res->gpu, lw->ssm.wqkv.type);

    int qkv_split_op_code = bn_gpu_quant_split_op_code(lw->attn.wq.type);
    int qkv_split_disabled = getenv("BN_GPU_DISABLE_QKV_SPLIT") != NULL;
    int use_split = !qkv_split_disabled && qkv_stacked && !q_gated &&
                    !q_bias && !k_bias && !v_bias &&
                    qkv_split_op_code == BN_GPU_CODE_MATVEC_SPLIT &&
                    bn_transformer_gpu_can_matvec_split(res->gpu, lw->attn.wq.type);
    int use_q8_split = !qkv_split_disabled && qkv_stacked && !q_gated &&
                       !q_bias && !k_bias && !v_bias &&
                       qkv_split_op_code == BN_GPU_CODE_Q8_MATVEC_SPLIT &&
                       bn_transformer_gpu_can_matvec_split(res->gpu, lw->attn.wq.type);
    int use_q5_split = !qkv_split_disabled && qkv_stacked && !q_gated &&
                       !q_bias && !k_bias && !v_bias &&
                       qkv_split_op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT &&
                       bn_transformer_gpu_can_matvec_split(res->gpu, lw->attn.wq.type);

    if (use_packed_q5_split) {
        emit_context_matvec_split(
            ctx, lw->ssm.wqkv.type,
            res ? res->packed_qkv : NULL, BN_GPU_VALUE_XB,
            BN_GPU_VALUE_Q, BN_GPU_VALUE_KEY_CACHE,
            BN_GPU_VALUE_VALUE_CACHE, lw->ssm.wqkv.rows, lw->ssm.wqkv.cols,
            q_dim, q_dim + kv_dim, (int)kv_cache_off, (int)kv_cache_off,
            use_q4_q8);
    } else if (use_packed_qkv) {
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->ssm.wqkv.type,
            res ? res->packed_qkv : NULL, BN_GPU_VALUE_XB,
            BN_GPU_VALUE_QKV, lw->ssm.wqkv.rows, lw->ssm.wqkv.cols, 0);
        bn_transformer_gpu_emit_context_copy(
            ctx, BN_GPU_VALUE_QKV, BN_GPU_VALUE_Q, 0, 0, q_dim);
        bn_transformer_gpu_emit_context_copy(
            ctx, BN_GPU_VALUE_QKV, BN_GPU_VALUE_KEY_CACHE, q_dim,
            (int)kv_cache_off, kv_dim);
        bn_transformer_gpu_emit_context_copy(
            ctx, BN_GPU_VALUE_QKV, BN_GPU_VALUE_VALUE_CACHE, q_dim + kv_dim,
            (int)kv_cache_off, kv_dim);
    } else if (use_q5_split || use_q8_split || use_split) {
        int total_rows = lw->attn.wq.rows + lw->attn.wk.rows + lw->attn.wv.rows;
        emit_context_matvec_split(
            ctx, lw->attn.wq.type, qkv_stacked, BN_GPU_VALUE_XB,
            BN_GPU_VALUE_Q, BN_GPU_VALUE_KEY_CACHE,
            BN_GPU_VALUE_VALUE_CACHE, total_rows, lw->attn.wq.cols,
            q_dim, q_dim + kv_dim, (int)kv_cache_off, (int)kv_cache_off,
            use_q4_q8);
    } else {
        if (q_gated) {
            emit_context_matvec_flags(
                ctx, lw->attn.wq.type,
                res ? res->wq : NULL, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_QKV, lw->attn.wq.rows, lw->attn.wq.cols, 0,
                use_q4_q8 ? 1u : 0u);
            uint32_t deinterleave_params[8] = {
                (uint32_t)q_dim, (uint32_t)head_size, 0, 0, 0, 0, 0, 0
            };
            emit_context_utility(ctx, BN_GPU_IR_UTILITY_DEINTERLEAVE_Q,
                                 BN_GPU_VALUE_QKV, -1, BN_GPU_VALUE_Q, 0,
                                 NULL, deinterleave_params);
        } else {
            emit_context_matvec_flags(
                ctx, lw->attn.wq.type,
                res ? res->wq : NULL, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_Q, lw->attn.wq.rows, lw->attn.wq.cols, 0,
                use_q4_q8 ? 1u : 0u);
        }
        if (q_bias) {
            uint32_t bias_params[8] = {
                (uint32_t)q_dim, 0, 0, 0, 0, 0, 0, 0
            };
            emit_context_utility(ctx, BN_GPU_IR_UTILITY_BIAS_ADD,
                                 BN_GPU_VALUE_Q, -1, -1, 0, q_bias,
                                 bias_params);
        }

        if (k_bias) {
            emit_context_matvec_flags(
                ctx, lw->attn.wk.type,
                res ? res->wk : NULL, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_SCRATCH, lw->attn.wk.rows, lw->attn.wk.cols, 0,
                use_q4_q8 ? 1u : 0u);
            uint32_t bias_params[8] = {
                (uint32_t)kv_dim, 0, 0, 0, 0, 0, 0, 0
            };
            emit_context_utility(ctx, BN_GPU_IR_UTILITY_BIAS_ADD,
                                 BN_GPU_VALUE_SCRATCH, -1, -1, 0, k_bias,
                                 bias_params);
            if (k_norm) {
                uint32_t ph_params[8] = {
                    (uint32_t)head_size, u_eps,
                    (uint32_t)c->qk_norm_per_head, 0, 0, 0, 0, 0
                };
                emit_context_utility(ctx,
                                     BN_GPU_IR_UTILITY_PER_HEAD_RMSNORM,
                                     BN_GPU_VALUE_SCRATCH, -1, -1,
                                     c->n_kv_heads, k_norm, ph_params);
            }
            emit_context_rope(ctx, BN_GPU_VALUE_SCRATCH, -1, c->n_kv_heads,
                              head_size, pos, rope_dims, 0, 0);
            bn_transformer_gpu_emit_context_copy(
                ctx, BN_GPU_VALUE_SCRATCH, BN_GPU_VALUE_KEY_CACHE, 0,
                (int)kv_cache_off, kv_dim);
        } else {
            emit_context_matvec_flags(
                ctx, lw->attn.wk.type,
                res ? res->wk : NULL, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_KEY_CACHE, lw->attn.wk.rows, lw->attn.wk.cols,
                (int)kv_cache_off, use_q4_q8 ? 1u : 0u);
        }

        if (v_bias) {
            emit_context_matvec_flags(
                ctx, lw->attn.wv.type,
                res ? res->wv : NULL, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_SCRATCH, lw->attn.wv.rows, lw->attn.wv.cols, 0,
                use_q4_q8 ? 1u : 0u);
            uint32_t bias_params[8] = {
                (uint32_t)kv_dim, 0, 0, 0, 0, 0, 0, 0
            };
            emit_context_utility(ctx, BN_GPU_IR_UTILITY_BIAS_ADD,
                                 BN_GPU_VALUE_SCRATCH, -1, -1, 0, v_bias,
                                 bias_params);
            bn_transformer_gpu_emit_context_copy(
                ctx, BN_GPU_VALUE_SCRATCH, BN_GPU_VALUE_VALUE_CACHE, 0,
                (int)kv_cache_off, kv_dim);
        } else {
            emit_context_matvec_flags(
                ctx, lw->attn.wv.type,
                res ? res->wv : NULL, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_VALUE_CACHE, lw->attn.wv.rows, lw->attn.wv.cols,
                (int)kv_cache_off, use_q4_q8 ? 1u : 0u);
        }
    }

    if (q_norm) {
        uint32_t ph_params[8] = {
            (uint32_t)head_size, u_eps,
            (uint32_t)c->qk_norm_per_head, 0, 0, 0, 0, 0
        };
        emit_context_utility(ctx, BN_GPU_IR_UTILITY_PER_HEAD_RMSNORM,
                             BN_GPU_VALUE_Q, -1, -1, n_heads, q_norm,
                             ph_params);
    }
    if (k_norm && !k_bias) {
        uint32_t ph_params[8] = {
            (uint32_t)head_size, u_eps,
            (uint32_t)c->qk_norm_per_head, kv_cache_off, 0, 0, 0, 0
        };
        emit_context_utility(ctx, BN_GPU_IR_UTILITY_PER_HEAD_RMSNORM,
                             BN_GPU_VALUE_KEY_CACHE, -1, -1, c->n_kv_heads,
                             k_norm, ph_params);
    }
}

void bn_transformer_gpu_emit_context_attention(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res,
    int pos,
    int dim,
    int q_dim,
    int head_size,
    int n_heads,
    int kv_dim,
    int rope_dims,
    int n_kv,
    size_t loff,
    uint32_t kv_cache_off,
    int has_moe,
    uint32_t u_eps,
    int use_q4_q8) {
    bn_transformer_gpu_emit_context_attention_gqa(
        ctx, c, lw, res, pos, q_dim, head_size, n_heads, kv_dim, rope_dims,
        n_kv, loff, kv_cache_off, has_moe);
    bn_transformer_gpu_emit_context_attention_finish(
        ctx, c, lw, res, dim, q_dim, head_size, u_eps, use_q4_q8);
}

void bn_transformer_gpu_emit_context_attention_gqa(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res,
    int pos,
    int q_dim,
    int head_size,
    int n_heads,
    int kv_dim,
    int rope_dims,
    int n_kv,
    size_t loff,
    uint32_t kv_cache_off,
    int has_moe) {
    void *k_bias = res ? res->k_bias : NULL;

    if (!k_bias) {
        emit_context_rope(ctx, BN_GPU_VALUE_Q, BN_GPU_VALUE_KEY_CACHE,
                          n_heads, head_size, pos, rope_dims, c->n_kv_heads,
                          kv_cache_off);
    } else {
        emit_context_rope(ctx, BN_GPU_VALUE_Q, -1, n_heads, head_size, pos,
                          rope_dims, 0, 0);
    }

    {
        float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);
        uint32_t u_inv_sqrt_hs;
        memcpy(&u_inv_sqrt_hs, &inv_sqrt_hs, 4);
        if (bn_transformer_gpu_can_flash_attn(res->gpu) &&
            (has_moe || c->flash_attn)) {
            emit_context_flash_attention(
                ctx, BN_GPU_VALUE_Q, BN_GPU_VALUE_XB, n_heads,
                head_size, n_kv, c->kv_mul, kv_dim, c->seq_len, loff,
                u_inv_sqrt_hs);
        } else {
            emit_context_gqa_attention(
                ctx, BN_GPU_VALUE_Q, BN_GPU_VALUE_XB, n_heads,
                head_size, n_kv, c->kv_mul, kv_dim, c->seq_len, loff,
                u_inv_sqrt_hs);
        }
    }
    (void)lw;
    (void)q_dim;
}

void bn_transformer_gpu_emit_context_attention_finish(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res,
    int dim,
    int q_dim,
    int head_size,
    uint32_t u_eps,
    int use_q4_q8) {
    (void)c;
    void *attn_sub_norm = res ? res->attn_sub_norm : NULL;
    void *ffn_norm = res ? res->ffn_norm : NULL;

    if (lw->attn.wq.rows > q_dim) {
        bn_transformer_gpu_emit_context_activation(
            ctx, BN_GPU_VALUE_XB, BN_GPU_VALUE_QKV, q_dim, head_size,
            BN_GPU_IR_ACTIVATION_SIGMOID);
    }

    int wo_in_buf = BN_GPU_VALUE_XB;
    if (attn_sub_norm) {
        bn_transformer_gpu_emit_context_rmsnorm(
            ctx, attn_sub_norm, BN_GPU_VALUE_XB, BN_GPU_VALUE_SCRATCH, dim,
            u_eps);
        wo_in_buf = BN_GPU_VALUE_SCRATCH;
    }

    emit_context_matvec_flags(
        ctx, lw->attn.wo.type, res ? res->wo : NULL,
        wo_in_buf, BN_GPU_VALUE_XB2, lw->attn.wo.rows, lw->attn.wo.cols, 0,
        use_q4_q8 ? 1u : 0u);

    emit_context_residual_rmsnorm(
        ctx, BN_GPU_VALUE_X, BN_GPU_VALUE_XB2, BN_GPU_VALUE_XB, dim, u_eps,
        ffn_norm);
}

void bn_transformer_gpu_emit_context_ssm(BnTransformerGPUEmitContext *ctx,
                                         const BnConfig *c,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         const BnTransformerGPUSSMResources *res,
                                         int dim,
                                         uint32_t u_eps) {
    int ssm_idx = plan->ssm_idx;
    int num_k_heads = c->ssm_group_count;
    int head_k_dim = c->ssm_state_size;
    int num_v_heads = c->ssm_time_step_rank;
    int head_v_dim = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
    int key_dim = num_k_heads * head_k_dim;
    int value_dim = c->ssm_inner_size;
    int qkv_dim_ssm = key_dim * 2 + value_dim;
    int kern = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
    size_t conv_off = (size_t)ssm_idx * (kern - 1) * qkv_dim_ssm;
    size_t state_per = (size_t)num_v_heads * head_k_dim * head_v_dim;
    size_t state_off = (size_t)ssm_idx * state_per;
    uint32_t u_qscale;
    {
        float qs = 1.0f / sqrtf((float)head_k_dim);
        memcpy(&u_qscale, &qs, 4);
    }

    void *ssm_qkvz_stacked = res ? res->ssm_qkvz_stacked : NULL;
    void *ssm_ab_stacked = res ? res->ssm_ab_stacked : NULL;
    void *ssm_conv1d = res ? res->ssm_conv1d : NULL;
    void *ssm_dt_bias = res ? res->ssm_dt_bias : NULL;
    void *ssm_a_log = res ? res->ssm_a_log : NULL;
    void *ssm_norm = res ? res->ssm_norm : NULL;
    void *ffn_norm = res ? res->ffn_norm : NULL;

    int ssm_split_op_code = bn_gpu_quant_split_op_code(lw->ssm.wqkv.type);
    if (ssm_qkvz_stacked &&
        ssm_split_op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT &&
        bn_transformer_gpu_can_matvec_split(res->gpu, lw->ssm.wqkv.type)) {
        int total_rows = lw->ssm.wqkv.rows + lw->ssm.wz.rows;
        emit_context_matvec_split(
            ctx, lw->ssm.wqkv.type, ssm_qkvz_stacked, BN_GPU_VALUE_XB,
            BN_GPU_VALUE_SSM_QKV, BN_GPU_VALUE_SSM_Z, BN_GPU_VALUE_SSM_Z,
            total_rows, lw->ssm.wqkv.cols, lw->ssm.wqkv.rows, 0, 0, 0, 0);
    } else {
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->ssm.wqkv.type,
            res ? res->wqkv : NULL, BN_GPU_VALUE_XB,
            BN_GPU_VALUE_SSM_QKV, lw->ssm.wqkv.rows, lw->ssm.wqkv.cols, 0);
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->ssm.wz.type, res ? res->wz : NULL,
            BN_GPU_VALUE_XB, BN_GPU_VALUE_SSM_Z, lw->ssm.wz.rows,
            lw->ssm.wz.cols, 0);
    }

    uint32_t ssm_conv_params[8] = {
        (uint32_t)qkv_dim_ssm, (uint32_t)kern, (uint32_t)conv_off,
        (uint32_t)((kern - 1) * qkv_dim_ssm), 0, 0, 0, 0
    };
    emit_context_ssm(ctx, BN_GPU_IR_SSM_CONV_SILU, BN_GPU_VALUE_SSM_QKV,
                     -1, -1, 0, ssm_conv1d, ssm_conv_params);
    uint32_t ssm_l2_params[8] = {
        (uint32_t)head_k_dim, 0, (uint32_t)key_dim, 0, 0, 0, 0, 0
    };
    emit_context_ssm(ctx, BN_GPU_IR_SSM_L2NORM, BN_GPU_VALUE_SSM_QKV,
                     BN_GPU_VALUE_SSM_QKV, -1, num_k_heads, NULL,
                     ssm_l2_params);

    if (ssm_ab_stacked &&
        lw->ssm.ssm_alpha.rows == lw->ssm.ssm_beta.rows &&
        lw->ssm.ssm_alpha.cols == lw->ssm.ssm_beta.cols) {
        int ab_rows = lw->ssm.ssm_alpha.rows + lw->ssm.ssm_beta.rows;
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->ssm.ssm_alpha.type, ssm_ab_stacked, BN_GPU_VALUE_XB,
            BN_GPU_VALUE_SSM_V, ab_rows, lw->ssm.ssm_alpha.cols, 0);
        _Static_assert(sizeof(void*) <= 8, "pointer must fit in 2 x uint32_t");
        uintptr_t a_ptr = (uintptr_t)ssm_a_log;
        uint32_t alpha_beta_split_params[8] = {
            (uint32_t)num_v_heads, (uint32_t)lw->ssm.ssm_alpha.rows,
            0, 0, 0, 0,
            (uint32_t)(a_ptr & 0xFFFFFFFF),
            (uint32_t)((uint64_t)a_ptr >> 32)
        };
        emit_context_ssm(ctx, BN_GPU_IR_SSM_ALPHA_BETA_SPLIT,
                         BN_GPU_VALUE_SSM_V, -1, -1, 0, ssm_dt_bias,
                         alpha_beta_split_params);
    } else {
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->ssm.ssm_alpha.type,
            res ? res->ssm_alpha : NULL,
            BN_GPU_VALUE_XB, BN_GPU_VALUE_SSM_ALPHA, lw->ssm.ssm_alpha.rows,
            lw->ssm.ssm_alpha.cols, 0);
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->ssm.ssm_beta.type,
            res ? res->ssm_beta : NULL,
            BN_GPU_VALUE_XB, BN_GPU_VALUE_SSM_BETA, lw->ssm.ssm_beta.rows,
            lw->ssm.ssm_beta.cols, 0);
        _Static_assert(sizeof(void*) <= 8, "pointer must fit in 2 x uint32_t");
        uintptr_t a_ptr = (uintptr_t)ssm_a_log;
        uint32_t alpha_beta_params[8] = {
            (uint32_t)num_v_heads, 0, 0, 0, 0, 0,
            (uint32_t)(a_ptr & 0xFFFFFFFF),
            (uint32_t)((uint64_t)a_ptr >> 32)
        };
        emit_context_ssm(ctx, BN_GPU_IR_SSM_ALPHA_BETA,
                         BN_GPU_VALUE_SSM_ALPHA, BN_GPU_VALUE_SSM_BETA, -1,
                         0, ssm_dt_bias, alpha_beta_params);
    }

    uint32_t ssm_delta_params[8] = {
        (uint32_t)head_k_dim, (uint32_t)head_v_dim,
        (uint32_t)num_k_heads, u_qscale,
        (uint32_t)(state_off * sizeof(float)),
        (uint32_t)(state_per * sizeof(float)), 0, (uint32_t)key_dim
    };
    emit_context_ssm(ctx, BN_GPU_IR_SSM_DELTA, BN_GPU_VALUE_SSM_QKV,
                     BN_GPU_VALUE_SSM_QKV, BN_GPU_VALUE_XB2, num_v_heads,
                     NULL, ssm_delta_params);
    uint32_t ssm_gate_params[8] = {
        (uint32_t)head_v_dim, u_eps, 0, 0, 0, 0, 0, 0
    };
    emit_context_ssm(ctx, BN_GPU_IR_SSM_GATE, BN_GPU_VALUE_XB2,
                     BN_GPU_VALUE_SSM_Z, -1, num_v_heads, ssm_norm,
                     ssm_gate_params);
    bn_transformer_gpu_emit_context_matvec(
        ctx, lw->ssm.ssm_out.type,
        res ? res->ssm_out : NULL, BN_GPU_VALUE_XB2,
        BN_GPU_VALUE_SCRATCH, lw->ssm.ssm_out.rows, lw->ssm.ssm_out.cols, 0);
    uint32_t residual_norm_params[8] = {
        (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0
    };
    emit_context_utility(ctx, BN_GPU_IR_UTILITY_RESIDUAL_RMSNORM,
                         BN_GPU_VALUE_X, BN_GPU_VALUE_SCRATCH,
                         BN_GPU_VALUE_XB, 0, ffn_norm,
                         residual_norm_params);
}

void bn_transformer_gpu_emit_context_moe(BnTransformerGPUEmitContext *ctx,
                                         const BnGPUMoEResources *moe,
                                         const BnTransformerGPUMoESharedResources *shared,
                                         const BnLayerWeights *lw,
                                         int dim,
                                         uint32_t u_eps,
                                         void *next_norm) {
    if (!ctx || !moe || !lw) return;
    const BnMoEExpertMap *em = moe->expert_map;
    int moe_hidden = moe->moe_hidden;

    for (int k = 0; k < moe->n_experts; k++) {
        const BnGPUMoEResolvedExpert *expert = &moe->experts[k];
        uint32_t u_ew;
        memcpy(&u_ew, &expert->weight, 4);

        if (expert->buffers.use_gateup_split) {
            emit_context_matvec_split(
                ctx, em->gate_type, expert->buffers.gate, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2, -1,
                em->gate_rows + em->up_rows, em->gate_cols,
                em->gate_rows, 0, 0, 0, 0);
        } else {
            bn_transformer_gpu_emit_context_matvec(
                ctx, em->gate_type, expert->buffers.gate, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_MOE_HB, em->gate_rows, em->gate_cols, 0);
            bn_transformer_gpu_emit_context_matvec(
                ctx, em->up_type, expert->buffers.up, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_MOE_HB2, em->up_rows, em->up_cols, 0);
        }
        bn_transformer_gpu_emit_context_activation(
            ctx, BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2, moe_hidden, 0,
            BN_GPU_IR_ACTIVATION_SILU);
        bn_transformer_gpu_emit_context_matvec(
            ctx, em->down_type, expert->buffers.down, BN_GPU_VALUE_MOE_HB,
            BN_GPU_VALUE_XB2, em->down_rows, em->down_cols, 0);
        uint32_t weighted_add_params[8] = {
            (uint32_t)dim, u_ew, 0, 0, 0, 0, 0, 0
        };
        emit_context_utility(ctx, BN_GPU_IR_UTILITY_WEIGHTED_ADD,
                             BN_GPU_VALUE_MOE_OUT, BN_GPU_VALUE_XB2, -1, 0,
                             NULL, weighted_add_params);
    }

    if (lw->shared.shared_gate.data && shared && shared->shared_gate) {
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->shared.shared_gate.type,
            shared->shared_gate,
            BN_GPU_VALUE_XB, BN_GPU_VALUE_HB, lw->shared.shared_gate.rows,
            lw->shared.shared_gate.cols, 0);
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->shared.shared_up.type,
            shared->shared_up,
            BN_GPU_VALUE_XB, BN_GPU_VALUE_HB2, lw->shared.shared_up.rows,
            lw->shared.shared_up.cols, 0);
        bn_transformer_gpu_emit_context_activation(
            ctx, BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2,
            lw->shared.shared_gate.rows, 0, BN_GPU_IR_ACTIVATION_SILU);
        bn_transformer_gpu_emit_context_matvec(
            ctx, lw->shared.shared_down.type,
            shared->shared_down,
            BN_GPU_VALUE_HB, BN_GPU_VALUE_XB2, lw->shared.shared_down.rows,
            lw->shared.shared_down.cols, 0);
        uint32_t u_one;
        { float one = 1.0f; memcpy(&u_one, &one, 4); }
        uint32_t weighted_add_params[8] = {
            (uint32_t)dim, u_one, 0, 0, 0, 0, 0, 0
        };
        emit_context_utility(ctx, BN_GPU_IR_UTILITY_WEIGHTED_ADD,
                             BN_GPU_VALUE_MOE_OUT, BN_GPU_VALUE_XB2, -1, 0,
                             NULL, weighted_add_params);
    }

    bn_transformer_gpu_emit_context_residual_add(
        ctx, BN_GPU_VALUE_X, BN_GPU_VALUE_MOE_OUT, dim);
    bn_transformer_gpu_emit_context_rmsnorm(
        ctx, next_norm, BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps);
}
