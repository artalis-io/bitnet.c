#include "transformer_gpu_internal.h"
#include "backend_quant.h"
#include "gpu_quant_lowering_internal.h"
#include "gpu_moe_bridge.h"
#include "gpu_moe_cache.h"
#include "moe.h"
#include "transformer_backend_internal.h"
#include "session.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void *qweight_backend_buf(const BnBackendModel *backend,
                                 const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

#define GPU_OP(code_) \
    .op_code = (code_)

void bn_transformer_gpu_finalize_op_kinds(BnGPUOp *ops, int n) {
    for (int i = 0; i < n; i++) {
        if (ops[i].op_kind == BN_GPU_OP_UNKNOWN)
            ops[i].op_kind = bn_gpu_op_kind(&ops[i]);
    }
}

void bn_transformer_gpu_emit_rmsnorm(BnGPUOp *ops, int *n,
                                     void *norm_gpu,
                                     int buf_in,
                                     int buf_out,
                                     int dim,
                                     uint32_t u_eps) {
    ops[(*n)++] = (BnGPUOp){
        .op_kind = BN_GPU_OP_RMSNORM,
        GPU_OP(BN_GPU_CODE_RMSNORM), .type = -1,
        .W_buf = norm_gpu,
        .buf_in = buf_in, .buf_out = buf_out, .buf_aux = -1,
        .rows = 0, .cols = 0,
        .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 }
    };
}

void bn_transformer_gpu_emit_dense_ffn(BnGPUOp *ops, int *n,
                                       const BnConfig *c,
                                       const BnLayerWeights *lw,
                                       const BnFFNPlan *ffn_plan,
                                       const BnGPUBackend *gpu,
                                       const BnBackendModel *backend,
                                       int layer,
                                       int dim,
                                       uint32_t u_eps,
                                       void *next_norm) {
    int hidden_dim = ffn_plan->hidden_dim;
    void *gateup_stacked = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED);
    void *ffn_sub_norm = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_FFN_SUB_NORM);

    if (ffn_plan->has_gate && lw->ffn.ffn_gate.data) {
        int use_fused_gateup = gateup_stacked &&
                               bn_transformer_gpu_can_fused_gateup_silu(gpu, lw->ffn.ffn_gate.type,
                                                                         ffn_plan->activation);
        if (use_fused_gateup) {
            int total_rows = lw->ffn.ffn_gate.rows + lw->ffn.ffn_up.rows;
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_FUSED_GATEUP_SILU), .type = lw->ffn.ffn_gate.type,
                .W_buf = gateup_stacked,
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB, .buf_aux = -1,
                .rows = lw->ffn.ffn_gate.rows, .cols = lw->ffn.ffn_gate.cols,
                .p = { (uint32_t)total_rows, (uint32_t)lw->ffn.ffn_gate.cols,
                       (uint32_t)lw->ffn.ffn_gate.rows, 0, 0, 0, 0, 0 }
            };
        } else if (gateup_stacked &&
                   lw->ffn.ffn_gate.rows == lw->ffn.ffn_up.rows &&
                   lw->ffn.ffn_gate.cols == lw->ffn.ffn_up.cols &&
                   ffn_plan->activation != 1 &&
                   bn_gpu_quant_split_op_code(lw->ffn.ffn_gate.type) ==
                       BN_GPU_CODE_Q4K_MATVEC_SPLIT &&
                   bn_transformer_gpu_can_matvec_split(gpu, lw->ffn.ffn_gate.type)) {
            int total_rows = lw->ffn.ffn_gate.rows + lw->ffn.ffn_up.rows;
            int split_op_code = bn_gpu_quant_split_op_code(lw->ffn.ffn_gate.type);
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(split_op_code),
                .type = lw->ffn.ffn_gate.type,
                .W_buf = gateup_stacked,
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB,
                .buf_aux = BN_GPU_VALUE_HB2,
                .rows = total_rows, .cols = lw->ffn.ffn_gate.cols,
                .p = { (uint32_t)total_rows, (uint32_t)lw->ffn.ffn_gate.cols,
                       (uint32_t)lw->ffn.ffn_gate.rows, 1, 0, 0, 0, 0 }
            };
        } else if (gateup_stacked &&
                   lw->ffn.ffn_gate.rows == lw->ffn.ffn_up.rows &&
                   lw->ffn.ffn_gate.cols == lw->ffn.ffn_up.cols &&
                   bn_transformer_gpu_can_matvec_split(gpu, lw->ffn.ffn_gate.type)) {
            int total_rows = lw->ffn.ffn_gate.rows + lw->ffn.ffn_up.rows;
            int split_op_code = bn_gpu_quant_split_op_code(lw->ffn.ffn_gate.type);
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(split_op_code),
                .type = lw->ffn.ffn_gate.type,
                .W_buf = gateup_stacked,
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB,
                .buf_aux = BN_GPU_VALUE_HB2,
                .rows = BN_GPU_VALUE_HB2, .cols = lw->ffn.ffn_gate.cols,
                .p = { (uint32_t)total_rows, (uint32_t)lw->ffn.ffn_gate.cols,
                       (uint32_t)lw->ffn.ffn_gate.rows, 0, 0, 0, 0, 0 }
            };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_SILU_GATE), .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_VALUE_HB, .buf_out = -1,
                .buf_aux = BN_GPU_VALUE_HB2,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)hidden_dim, 0, 0, 0, 0, 0, 0, 0 }
            };
        } else {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->ffn.ffn_gate.type,
                .W_buf = qweight_backend_buf(backend, &lw->ffn.ffn_gate),
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB, .buf_aux = -1,
                .rows = lw->ffn.ffn_gate.rows, .cols = lw->ffn.ffn_gate.cols,
                .p = { (uint32_t)lw->ffn.ffn_gate.rows, (uint32_t)lw->ffn.ffn_gate.cols,
                       1, 0, 0, 0, 0, 0 }
            };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->ffn.ffn_up.type,
                .W_buf = qweight_backend_buf(backend, &lw->ffn.ffn_up),
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB2, .buf_aux = -1,
                .rows = lw->ffn.ffn_up.rows, .cols = lw->ffn.ffn_up.cols,
                .p = { (uint32_t)lw->ffn.ffn_up.rows, (uint32_t)lw->ffn.ffn_up.cols,
                       1, 0, 0, 0, 0, 0 }
            };
            int act_code = (ffn_plan->activation == 1) ? BN_GPU_CODE_RELU2_GATE
                                                       : BN_GPU_CODE_SILU_GATE;
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(act_code),
                .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_VALUE_HB, .buf_out = -1,
                .buf_aux = BN_GPU_VALUE_HB2,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)hidden_dim, 0, 0, 0, 0, 0, 0, 0 }
            };
        }
    } else {
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->ffn.ffn_up.type,
            .W_buf = qweight_backend_buf(backend, &lw->ffn.ffn_up),
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB, .buf_aux = -1,
            .rows = lw->ffn.ffn_up.rows, .cols = lw->ffn.ffn_up.cols,
            .p = { (uint32_t)lw->ffn.ffn_up.rows, (uint32_t)lw->ffn.ffn_up.cols,
                   1, 0, 0, 0, 0, 0 }
        };
        int act_code = (ffn_plan->activation == 1) ? BN_GPU_CODE_RELU2_ACT
                                                   : BN_GPU_CODE_SILU_ACT;
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(act_code),
            .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_VALUE_HB, .buf_out = -1,
            .buf_aux = -1,
            .rows = 0, .cols = 0,
            .p = { (uint32_t)hidden_dim, 0, 0, 0, 0, 0, 0, 0 }
        };
    }

    int down_in_buf = BN_GPU_VALUE_HB;
    if (ffn_sub_norm) {
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_RMSNORM), .type = -1,
            .W_buf = ffn_sub_norm,
            .buf_in = BN_GPU_VALUE_HB, .buf_out = BN_GPU_VALUE_HB2, .buf_aux = -1,
            .p = { (uint32_t)hidden_dim, u_eps, 0, 0, 0, 0, 0, 0 } };
        down_in_buf = BN_GPU_VALUE_HB2;
    }

    ops[(*n)++] = (BnGPUOp){
        GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->ffn.ffn_down.type,
        .W_buf = qweight_backend_buf(backend, &lw->ffn.ffn_down),
        .buf_in = down_in_buf, .buf_out = BN_GPU_VALUE_XB2, .buf_aux = -1,
        .rows = lw->ffn.ffn_down.rows, .cols = lw->ffn.ffn_down.cols,
        .p = { (uint32_t)lw->ffn.ffn_down.rows, (uint32_t)lw->ffn.ffn_down.cols,
               1, 0, 0, 0, 0, 0 }
    };

    ops[(*n)++] = (BnGPUOp){
        GPU_OP(BN_GPU_CODE_RESIDUAL_RMSNORM), .type = -1,
        .W_buf = next_norm,
        .buf_in = BN_GPU_VALUE_X, .buf_out = BN_GPU_VALUE_XB, .buf_aux = BN_GPU_VALUE_XB2,
        .rows = 0, .cols = 0,
        .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 }
    };

    (void)c;
}

void bn_transformer_gpu_emit_qkv(BnGPUOp *ops, int *n,
                                 const BnConfig *c,
                                 const BnLayerWeights *lw,
                                 const BnLayerShapePlan *plan,
                                 const BnGPUBackend *gpu,
                                 const BnBackendModel *backend,
                                 int layer,
                                 int pos,
                                 int q_dim,
                                 int head_size,
                                 int n_heads,
                                 int kv_dim,
                                 int rope_dims,
                                 uint32_t kv_cache_off,
                                 uint32_t u_eps) {
    void *q_bias = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_Q_BIAS);
    void *k_bias = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_K_BIAS);
    void *v_bias = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_V_BIAS);
    void *q_norm = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_Q_NORM);
    void *k_norm = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_K_NORM);
    void *qkv_stacked = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_QKV_STACKED);

    int use_packed_qkv = lw->ssm.wqkv.data &&
                         qweight_backend_buf(backend, &lw->ssm.wqkv) &&
                         !q_bias && !k_bias && !v_bias &&
                         lw->ssm.wqkv.rows == q_dim + 2 * kv_dim;
    int q_gated = !use_packed_qkv && plan->q_gated;
    int packed_split_op_code = bn_gpu_quant_split_op_code(lw->ssm.wqkv.type);
    int use_packed_q5_split =
        use_packed_qkv &&
        packed_split_op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT &&
        bn_transformer_gpu_can_matvec_split(gpu, lw->ssm.wqkv.type);

    int qkv_split_op_code = bn_gpu_quant_split_op_code(lw->attn.wq.type);
    int use_split = qkv_stacked && !q_gated &&
                    !q_bias && !k_bias && !v_bias &&
                    qkv_split_op_code == BN_GPU_CODE_MATVEC_SPLIT &&
                    bn_transformer_gpu_can_matvec_split(gpu, lw->attn.wq.type);
    int use_q8_split = qkv_stacked && !q_gated &&
                       !q_bias && !k_bias && !v_bias &&
                       qkv_split_op_code == BN_GPU_CODE_Q8_MATVEC_SPLIT &&
                       bn_transformer_gpu_can_matvec_split(gpu, lw->attn.wq.type);
    int use_q5_split = qkv_stacked && !q_gated &&
                       !q_bias && !k_bias && !v_bias &&
                       qkv_split_op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT &&
                       bn_transformer_gpu_can_matvec_split(gpu, lw->attn.wq.type);

    if (use_packed_q5_split) {
        ops[(*n)++] = (BnGPUOp){
            .op_code = packed_split_op_code,
            .type = lw->ssm.wqkv.type,
            .W_buf = qweight_backend_buf(backend, &lw->ssm.wqkv),
            .buf_in = BN_GPU_VALUE_XB,
            .buf_out = BN_GPU_VALUE_Q,
            .buf_aux = BN_GPU_VALUE_KEY_CACHE,
            .rows = BN_GPU_VALUE_VALUE_CACHE, .cols = lw->ssm.wqkv.cols,
            .p = { (uint32_t)lw->ssm.wqkv.rows, (uint32_t)lw->ssm.wqkv.cols,
                   (uint32_t)q_dim, (uint32_t)(q_dim + kv_dim),
                   0, 0, kv_cache_off, kv_cache_off }
        };
    } else if (use_packed_qkv) {
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->ssm.wqkv.type,
            .W_buf = qweight_backend_buf(backend, &lw->ssm.wqkv),
            .buf_in = BN_GPU_VALUE_XB,
            .buf_out = BN_GPU_VALUE_QKV,
            .buf_aux = -1,
            .rows = lw->ssm.wqkv.rows, .cols = lw->ssm.wqkv.cols,
            .p = { (uint32_t)lw->ssm.wqkv.rows, (uint32_t)lw->ssm.wqkv.cols,
                   1, 0, 0, 0, 0, 0 }
        };
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_COPY), .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_VALUE_QKV, .buf_out = BN_GPU_VALUE_Q,
            .buf_aux = -1,
            .p = { 0, 0, (uint32_t)q_dim, 0, 0, 0, 0, 0 }
        };
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_COPY), .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_VALUE_QKV, .buf_out = BN_GPU_VALUE_KEY_CACHE,
            .buf_aux = -1,
            .p = { (uint32_t)q_dim, kv_cache_off, (uint32_t)kv_dim,
                   0, 0, 0, 0, 0 }
        };
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_COPY), .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_VALUE_QKV, .buf_out = BN_GPU_VALUE_VALUE_CACHE,
            .buf_aux = -1,
            .p = { (uint32_t)(q_dim + kv_dim), kv_cache_off,
                   (uint32_t)kv_dim, 0, 0, 0, 0, 0 }
        };
    } else if (use_q5_split || use_q8_split || use_split) {
        int total_rows = lw->attn.wq.rows + lw->attn.wk.rows + lw->attn.wv.rows;
        ops[(*n)++] = (BnGPUOp){
            .op_code = qkv_split_op_code,
            .type = lw->attn.wq.type,
            .W_buf = qkv_stacked,
            .buf_in = BN_GPU_VALUE_XB,
            .buf_out = BN_GPU_VALUE_Q,
            .buf_aux = BN_GPU_VALUE_KEY_CACHE,
            .rows = BN_GPU_VALUE_VALUE_CACHE, .cols = lw->attn.wq.cols,
            .p = { (uint32_t)total_rows, (uint32_t)lw->attn.wq.cols,
                   (uint32_t)q_dim, (uint32_t)(q_dim + kv_dim),
                   0, 0, kv_cache_off, kv_cache_off }
        };
    } else {
        if (q_gated) {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->attn.wq.type,
                .W_buf = qweight_backend_buf(backend, &lw->attn.wq),
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_QKV,
                .buf_aux = -1, .rows = lw->attn.wq.rows, .cols = lw->attn.wq.cols,
                .p = { (uint32_t)lw->attn.wq.rows, (uint32_t)lw->attn.wq.cols,
                       1, 0, 0, 0, 0, 0 } };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_DEINTERLEAVE_Q), .type = -1,
                .W_buf = NULL, .buf_in = BN_GPU_VALUE_QKV,
                .buf_out = BN_GPU_VALUE_Q, .buf_aux = -1,
                .p = { (uint32_t)q_dim, (uint32_t)head_size,
                       0, 0, 0, 0, 0, 0 } };
        } else {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->attn.wq.type,
                .W_buf = qweight_backend_buf(backend, &lw->attn.wq),
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_Q,
                .buf_aux = -1, .rows = lw->attn.wq.rows, .cols = lw->attn.wq.cols,
                .p = { (uint32_t)lw->attn.wq.rows, (uint32_t)lw->attn.wq.cols,
                       1, 0, 0, 0, 0, 0 } };
        }
        if (q_bias) {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_BIAS_ADD), .type = -1,
                .W_buf = q_bias,
                .buf_in = BN_GPU_VALUE_Q, .buf_out = -1, .buf_aux = -1,
                .p = { (uint32_t)q_dim, 0, 0, 0, 0, 0, 0, 0 } };
        }

        if (k_bias) {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->attn.wk.type,
                .W_buf = qweight_backend_buf(backend, &lw->attn.wk),
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SCRATCH,
                .buf_aux = -1, .rows = lw->attn.wk.rows, .cols = lw->attn.wk.cols,
                .p = { (uint32_t)lw->attn.wk.rows, (uint32_t)lw->attn.wk.cols,
                       1, 0, 0, 0, 0, 0 } };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_BIAS_ADD), .type = -1,
                .W_buf = k_bias,
                .buf_in = BN_GPU_VALUE_SCRATCH, .buf_out = -1,
                .buf_aux = -1,
                .p = { (uint32_t)kv_dim, 0, 0, 0, 0, 0, 0, 0 } };
            if (k_norm) {
                ops[(*n)++] = (BnGPUOp){
                    GPU_OP(BN_GPU_CODE_PER_HEAD_RMSNORM),
                    .type = -1, .W_buf = k_norm,
                    .buf_in = BN_GPU_VALUE_SCRATCH, .buf_out = -1,
                    .buf_aux = -1, .rows = c->n_kv_heads,
                    .p = { (uint32_t)head_size, u_eps,
                           (uint32_t)c->qk_norm_per_head,
                           0, 0, 0, 0, 0 } };
            }
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_ROPE), .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_VALUE_SCRATCH, .buf_out = -1,
                .buf_aux = -1,
                .p = { (uint32_t)c->n_kv_heads, (uint32_t)head_size,
                       (uint32_t)pos, (uint32_t)rope_dims, 0, 0, 0, 0 } };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_COPY), .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_VALUE_SCRATCH,
                .buf_out = BN_GPU_VALUE_KEY_CACHE, .buf_aux = -1,
                .p = { 0, kv_cache_off, (uint32_t)kv_dim, 0, 0, 0, 0, 0 } };
        } else {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->attn.wk.type,
                .W_buf = qweight_backend_buf(backend, &lw->attn.wk),
                .buf_in = BN_GPU_VALUE_XB,
                .buf_out = BN_GPU_VALUE_KEY_CACHE, .buf_aux = -1,
                .rows = lw->attn.wk.rows, .cols = lw->attn.wk.cols,
                .p = { (uint32_t)lw->attn.wk.rows, (uint32_t)lw->attn.wk.cols,
                       1, 0, 0, kv_cache_off, 0, 0 } };
        }

        if (v_bias) {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->attn.wv.type,
                .W_buf = qweight_backend_buf(backend, &lw->attn.wv),
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SCRATCH,
                .buf_aux = -1, .rows = lw->attn.wv.rows, .cols = lw->attn.wv.cols,
                .p = { (uint32_t)lw->attn.wv.rows, (uint32_t)lw->attn.wv.cols,
                       1, 0, 0, 0, 0, 0 } };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_BIAS_ADD), .type = -1,
                .W_buf = v_bias,
                .buf_in = BN_GPU_VALUE_SCRATCH, .buf_out = -1,
                .buf_aux = -1,
                .p = { (uint32_t)kv_dim, 0, 0, 0, 0, 0, 0, 0 } };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_COPY), .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_VALUE_SCRATCH,
                .buf_out = BN_GPU_VALUE_VALUE_CACHE, .buf_aux = -1,
                .p = { 0, kv_cache_off, (uint32_t)kv_dim, 0, 0, 0, 0, 0 } };
        } else {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->attn.wv.type,
                .W_buf = qweight_backend_buf(backend, &lw->attn.wv),
                .buf_in = BN_GPU_VALUE_XB,
                .buf_out = BN_GPU_VALUE_VALUE_CACHE, .buf_aux = -1,
                .rows = lw->attn.wv.rows, .cols = lw->attn.wv.cols,
                .p = { (uint32_t)lw->attn.wv.rows, (uint32_t)lw->attn.wv.cols,
                       1, 0, 0, kv_cache_off, 0, 0 } };
        }
    }

    if (q_norm) {
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_PER_HEAD_RMSNORM),
            .type = -1, .W_buf = q_norm,
            .buf_in = BN_GPU_VALUE_Q, .buf_out = -1, .buf_aux = -1,
            .rows = n_heads,
            .p = { (uint32_t)head_size, u_eps,
                   (uint32_t)c->qk_norm_per_head, 0, 0, 0, 0, 0 } };
    }
    if (k_norm && !k_bias) {
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_PER_HEAD_RMSNORM),
            .type = -1, .W_buf = k_norm,
            .buf_in = BN_GPU_VALUE_KEY_CACHE, .buf_out = -1, .buf_aux = -1,
            .rows = c->n_kv_heads,
            .p = { (uint32_t)head_size, u_eps,
                   (uint32_t)c->qk_norm_per_head,
                   kv_cache_off, 0, 0, 0, 0 } };
    }
}

void bn_transformer_gpu_emit_attention(BnGPUOp *ops, int *n,
                                       const BnConfig *c,
                                       const BnLayerWeights *lw,
                                       const BnGPUBackend *gpu,
                                       const BnBackendModel *backend,
                                       int layer,
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
                                       uint32_t u_eps) {
    void *k_bias = bn_transformer_backend_handle_or(backend, layer,
                                                    BN_BACKEND_HANDLE_K_BIAS);
    void *attn_sub_norm = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_ATTN_SUB_NORM);
    void *ffn_norm = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);

    if (!k_bias) {
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_ROPE_QK), .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_VALUE_Q, .buf_out = -1, .buf_aux = BN_GPU_VALUE_KEY_CACHE,
            .p = { (uint32_t)n_heads, (uint32_t)head_size,
                   (uint32_t)pos, (uint32_t)rope_dims,
                   (uint32_t)c->n_kv_heads, kv_cache_off, 0, 0 }
        };
    } else {
        ops[(*n)++] = (BnGPUOp){
            GPU_OP(BN_GPU_CODE_ROPE), .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_VALUE_Q, .buf_out = -1, .buf_aux = -1,
            .p = { (uint32_t)n_heads, (uint32_t)head_size,
                   (uint32_t)pos, (uint32_t)rope_dims, 0, 0, 0, 0 }
        };
    }

    {
        float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);
        uint32_t u_inv_sqrt_hs;
        memcpy(&u_inv_sqrt_hs, &inv_sqrt_hs, 4);
        if (bn_transformer_gpu_can_flash_attn(gpu) &&
            (has_moe || c->flash_attn)) {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_FLASH_ATTN), .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_VALUE_Q, .buf_out = BN_GPU_VALUE_XB, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)n_heads, (uint32_t)head_size, (uint32_t)n_kv,
                       (uint32_t)c->kv_mul, (uint32_t)kv_dim, (uint32_t)c->seq_len,
                       (uint32_t)loff, u_inv_sqrt_hs }
            };
        } else {
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_GQA_SCORES), .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_VALUE_Q, .buf_out = -1, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)n_heads, (uint32_t)head_size, (uint32_t)n_kv,
                       (uint32_t)c->kv_mul, (uint32_t)kv_dim, (uint32_t)c->seq_len,
                       (uint32_t)loff, u_inv_sqrt_hs }
            };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_SOFTMAX), .type = -1, .W_buf = NULL,
                .buf_in = -1, .buf_out = -1, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)n_heads, (uint32_t)n_kv, (uint32_t)c->seq_len,
                       0, 0, 0, 0, 0 }
            };
            ops[(*n)++] = (BnGPUOp){
                GPU_OP(BN_GPU_CODE_GQA_COMBINE), .type = -1, .W_buf = NULL,
                .buf_in = -1, .buf_out = BN_GPU_VALUE_XB, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)n_heads, (uint32_t)head_size, (uint32_t)n_kv,
                       (uint32_t)c->kv_mul, (uint32_t)kv_dim, (uint32_t)c->seq_len,
                       (uint32_t)loff, 0 }
            };
        }
    }

    if (lw->attn.wq.rows > q_dim) {
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SIGMOID_GATE), .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_VALUE_XB, .buf_out = -1, .buf_aux = BN_GPU_VALUE_QKV,
            .p = { (uint32_t)q_dim, (uint32_t)head_size, 0, 0, 0, 0, 0, 0 } };
    }

    int wo_in_buf = BN_GPU_VALUE_XB;
    if (attn_sub_norm) {
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_RMSNORM), .type = -1,
            .W_buf = attn_sub_norm,
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SCRATCH, .buf_aux = -1,
            .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 } };
        wo_in_buf = BN_GPU_VALUE_SCRATCH;
    }

    ops[(*n)++] = (BnGPUOp){
        GPU_OP(BN_GPU_CODE_MATVEC), .type = lw->attn.wo.type,
        .W_buf = qweight_backend_buf(backend, &lw->attn.wo),
        .buf_in = wo_in_buf, .buf_out = BN_GPU_VALUE_XB2, .buf_aux = -1,
        .rows = lw->attn.wo.rows, .cols = lw->attn.wo.cols,
        .p = { (uint32_t)lw->attn.wo.rows, (uint32_t)lw->attn.wo.cols, 1, 0, 0, 0, 0, 0 }
    };

    ops[(*n)++] = (BnGPUOp){
        GPU_OP(BN_GPU_CODE_RESIDUAL_RMSNORM), .type = -1,
        .W_buf = ffn_norm,
        .buf_in = BN_GPU_VALUE_X, .buf_out = BN_GPU_VALUE_XB, .buf_aux = BN_GPU_VALUE_XB2,
        .rows = 0, .cols = 0,
        .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 }
    };
}

void bn_transformer_gpu_emit_ssm(BnGPUOp *ops, int *n,
                                 const BnConfig *c,
                                 const BnLayerWeights *lw,
                                 const BnLayerShapePlan *plan,
                                 const BnGPUBackend *gpu,
                                 const BnBackendModel *backend,
                                 int layer,
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

    void *ssm_qkvz_stacked = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED);
    void *ssm_ab_stacked = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_AB_STACKED);
    void *ssm_conv1d = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_CONV1D);
    void *ssm_dt_bias = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_DT_BIAS);
    void *ssm_a_log = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_A_LOG);
    void *ssm_norm = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_NORM);
    void *ffn_norm = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);

    int ssm_split_op_code = bn_gpu_quant_split_op_code(lw->ssm.wqkv.type);
    if (ssm_qkvz_stacked &&
        ssm_split_op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT &&
        bn_transformer_gpu_can_matvec_split(gpu, lw->ssm.wqkv.type)) {
        int total_rows = lw->ssm.wqkv.rows + lw->ssm.wz.rows;
        ops[(*n)++] = (BnGPUOp){ .op_code = ssm_split_op_code,
            .type = lw->ssm.wqkv.type, .W_buf = ssm_qkvz_stacked,
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SSM_QKV,
            .buf_aux = BN_GPU_VALUE_SSM_Z, .rows = BN_GPU_VALUE_SSM_Z,
            .cols = lw->ssm.wqkv.cols,
            .p = { (uint32_t)total_rows, (uint32_t)lw->ssm.wqkv.cols,
                   (uint32_t)lw->ssm.wqkv.rows, 0, 0, 0, 0, 0 } };
    } else {
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->ssm.wqkv.type,
            .W_buf = qweight_backend_buf(backend, &lw->ssm.wqkv),
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SSM_QKV,
            .buf_aux = -1, .rows = lw->ssm.wqkv.rows, .cols = lw->ssm.wqkv.cols,
            .p = { (uint32_t)lw->ssm.wqkv.rows, (uint32_t)lw->ssm.wqkv.cols,
                   1, 0, 0, 0, 0, 0 } };
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->ssm.wz.type,
            .W_buf = qweight_backend_buf(backend, &lw->ssm.wz),
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SSM_Z,
            .buf_aux = -1, .rows = lw->ssm.wz.rows, .cols = lw->ssm.wz.cols,
            .p = { (uint32_t)lw->ssm.wz.rows, (uint32_t)lw->ssm.wz.cols,
                   1, 0, 0, 0, 0, 0 } };
    }

    ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SSM_CONV_SILU),
        .type = -1, .W_buf = ssm_conv1d, .buf_in = BN_GPU_VALUE_SSM_QKV,
        .buf_out = -1, .buf_aux = -1,
        .p = { (uint32_t)qkv_dim_ssm, (uint32_t)kern, (uint32_t)conv_off,
               (uint32_t)((kern - 1) * qkv_dim_ssm), 0, 0, 0, 0 } };
    ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SSM_L2NORM),
        .type = -1, .W_buf = NULL, .buf_in = BN_GPU_VALUE_SSM_QKV,
        .buf_out = -1, .buf_aux = BN_GPU_VALUE_SSM_QKV,
        .rows = num_k_heads,
        .p = { (uint32_t)head_k_dim, 0, (uint32_t)key_dim, 0, 0, 0, 0, 0 } };

    if (ssm_ab_stacked &&
        lw->ssm.ssm_alpha.rows == lw->ssm.ssm_beta.rows &&
        lw->ssm.ssm_alpha.cols == lw->ssm.ssm_beta.cols) {
        int ab_rows = lw->ssm.ssm_alpha.rows + lw->ssm.ssm_beta.rows;
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->ssm.ssm_alpha.type, .W_buf = ssm_ab_stacked,
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SSM_V,
            .buf_aux = -1, .rows = ab_rows, .cols = lw->ssm.ssm_alpha.cols,
            .p = { (uint32_t)ab_rows, (uint32_t)lw->ssm.ssm_alpha.cols,
                   1, 0, 0, 0, 0, 0 } };
        _Static_assert(sizeof(void*) <= 8, "pointer must fit in 2 x uint32_t");
        uintptr_t a_ptr = (uintptr_t)ssm_a_log;
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT),
            .type = -1, .W_buf = ssm_dt_bias, .buf_in = BN_GPU_VALUE_SSM_V,
            .buf_out = -1, .buf_aux = -1,
            .p = { (uint32_t)num_v_heads, (uint32_t)lw->ssm.ssm_alpha.rows,
                   0, 0, 0, 0,
                   (uint32_t)(a_ptr & 0xFFFFFFFF),
                   (uint32_t)((uint64_t)a_ptr >> 32) } };
    } else {
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->ssm.ssm_alpha.type,
            .W_buf = qweight_backend_buf(backend, &lw->ssm.ssm_alpha),
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SSM_ALPHA,
            .buf_aux = -1, .rows = lw->ssm.ssm_alpha.rows, .cols = lw->ssm.ssm_alpha.cols,
            .p = { (uint32_t)lw->ssm.ssm_alpha.rows, (uint32_t)lw->ssm.ssm_alpha.cols,
                   1, 0, 0, 0, 0, 0 } };
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->ssm.ssm_beta.type,
            .W_buf = qweight_backend_buf(backend, &lw->ssm.ssm_beta),
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_SSM_BETA,
            .buf_aux = -1, .rows = lw->ssm.ssm_beta.rows, .cols = lw->ssm.ssm_beta.cols,
            .p = { (uint32_t)lw->ssm.ssm_beta.rows, (uint32_t)lw->ssm.ssm_beta.cols,
                   1, 0, 0, 0, 0, 0 } };
        _Static_assert(sizeof(void*) <= 8, "pointer must fit in 2 x uint32_t");
        uintptr_t a_ptr = (uintptr_t)ssm_a_log;
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SSM_ALPHA_BETA),
            .type = -1, .W_buf = ssm_dt_bias,
            .buf_in = BN_GPU_VALUE_SSM_ALPHA, .buf_out = -1,
            .buf_aux = BN_GPU_VALUE_SSM_BETA,
            .p = { (uint32_t)num_v_heads, 0, 0, 0, 0, 0,
                   (uint32_t)(a_ptr & 0xFFFFFFFF),
                   (uint32_t)((uint64_t)a_ptr >> 32) } };
    }

    ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SSM_DELTA),
        .type = -1, .W_buf = NULL, .buf_in = BN_GPU_VALUE_SSM_QKV,
        .buf_out = BN_GPU_VALUE_XB2, .buf_aux = BN_GPU_VALUE_SSM_QKV,
        .rows = num_v_heads,
        .p = { (uint32_t)head_k_dim, (uint32_t)head_v_dim,
               (uint32_t)num_k_heads, u_qscale,
               (uint32_t)(state_off * sizeof(float)),
               (uint32_t)(state_per * sizeof(float)), 0,
               (uint32_t)key_dim } };
    ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SSM_GATE),
        .type = -1, .W_buf = ssm_norm, .buf_in = BN_GPU_VALUE_XB2,
        .buf_out = -1, .buf_aux = BN_GPU_VALUE_SSM_Z,
        .rows = num_v_heads,
        .p = { (uint32_t)head_v_dim, u_eps, 0, 0, 0, 0, 0, 0 } };
    ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
        .type = lw->ssm.ssm_out.type,
        .W_buf = qweight_backend_buf(backend, &lw->ssm.ssm_out),
        .buf_in = BN_GPU_VALUE_XB2, .buf_out = BN_GPU_VALUE_SCRATCH,
        .buf_aux = -1, .rows = lw->ssm.ssm_out.rows, .cols = lw->ssm.ssm_out.cols,
        .p = { (uint32_t)lw->ssm.ssm_out.rows, (uint32_t)lw->ssm.ssm_out.cols,
               1, 0, 0, 0, 0, 0 } };
    ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_RESIDUAL_RMSNORM),
        .type = -1, .W_buf = ffn_norm,
        .buf_in = BN_GPU_VALUE_X, .buf_out = BN_GPU_VALUE_XB,
        .buf_aux = BN_GPU_VALUE_SCRATCH,
        .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 } };
}

void bn_transformer_gpu_emit_moe(BnGPUOp *ops, int *n,
                                 BnModel *m,
                                 BnSession *sess,
                                 const BnLayerWeights *lw,
                                 int layer,
                                 int dim,
                                 uint32_t u_eps,
                                 void *next_norm,
                                 void **uncached_bufs,
                                 int *n_uncached) {
    BnConfig *c = &m->config;
    const BnBackendModel *backend = bn_model_backend(m);
    BnMoEState *ms = sess->moe_state;
    int K = c->n_experts_active;
    int moe_hidden = c->moe_intermediate_size;
    const BnMoEExpertMap *em = &lw->moe.expert_map;
    *n_uncached = 0;

    for (int k = 0; k < K; k++) {
        int eidx = ms->expert_indices[k];
        if (eidx < 0 || eidx >= c->n_experts) continue;
        float ew = ms->expert_weights[k];
        uint32_t u_ew;
        memcpy(&u_ew, &ew, 4);

        BnGPUMoEExpertBuffers expert;
        if (bn_gpu_moe_bridge_get_expert(m, sess, lw, layer, eidx,
                                         uncached_bufs, n_uncached,
                                         &expert) != 0)
            continue;

        if (expert.use_gateup_split) {
            ops[(*n)++] = (BnGPUOp){ .op_code = expert.gateup_split_op_code,
                .type = em->gate_type, .W_buf = expert.gate,
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_MOE_HB,
                .buf_aux = BN_GPU_VALUE_MOE_HB2,
                .rows = em->gate_rows + em->up_rows, .cols = em->gate_cols,
                .p = { (uint32_t)(em->gate_rows + em->up_rows),
                       (uint32_t)em->gate_cols, (uint32_t)em->gate_rows,
                       0, 0, 0, 0, 0 } };
        } else {
            ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
                .type = em->gate_type, .W_buf = expert.gate,
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_MOE_HB,
                .buf_aux = -1, .rows = em->gate_rows, .cols = em->gate_cols,
                .p = { (uint32_t)em->gate_rows, (uint32_t)em->gate_cols,
                       1, 0, 0, 0, 0, 0 } };
            ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
                .type = em->up_type, .W_buf = expert.up,
                .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_MOE_HB2,
                .buf_aux = -1, .rows = em->up_rows, .cols = em->up_cols,
                .p = { (uint32_t)em->up_rows, (uint32_t)em->up_cols,
                       1, 0, 0, 0, 0, 0 } };
        }
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SILU_GATE),
            .type = -1, .W_buf = NULL, .buf_in = BN_GPU_VALUE_MOE_HB,
            .buf_out = -1, .buf_aux = BN_GPU_VALUE_MOE_HB2,
            .p = { (uint32_t)moe_hidden, 0, 0, 0, 0, 0, 0, 0 } };
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = em->down_type, .W_buf = expert.down,
            .buf_in = BN_GPU_VALUE_MOE_HB, .buf_out = BN_GPU_VALUE_XB2,
            .buf_aux = -1, .rows = em->down_rows, .cols = em->down_cols,
            .p = { (uint32_t)em->down_rows, (uint32_t)em->down_cols,
                   1, 0, 0, 0, 0, 0 } };
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_WEIGHTED_ADD),
            .type = -1, .W_buf = NULL, .buf_in = BN_GPU_VALUE_MOE_OUT,
            .buf_out = -1, .buf_aux = BN_GPU_VALUE_XB2,
            .p = { (uint32_t)dim, u_ew, 0, 0, 0, 0, 0, 0 } };
    }

    if (lw->shared.shared_gate.data && qweight_backend_buf(backend, &lw->shared.shared_gate)) {
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->shared.shared_gate.type,
            .W_buf = qweight_backend_buf(backend, &lw->shared.shared_gate),
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB,
            .buf_aux = -1, .rows = lw->shared.shared_gate.rows,
            .cols = lw->shared.shared_gate.cols,
            .p = { (uint32_t)lw->shared.shared_gate.rows,
                   (uint32_t)lw->shared.shared_gate.cols, 1, 0, 0, 0, 0, 0 } };
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->shared.shared_up.type,
            .W_buf = qweight_backend_buf(backend, &lw->shared.shared_up),
            .buf_in = BN_GPU_VALUE_XB, .buf_out = BN_GPU_VALUE_HB2,
            .buf_aux = -1, .rows = lw->shared.shared_up.rows,
            .cols = lw->shared.shared_up.cols,
            .p = { (uint32_t)lw->shared.shared_up.rows,
                   (uint32_t)lw->shared.shared_up.cols, 1, 0, 0, 0, 0, 0 } };
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_SILU_GATE),
            .type = -1, .W_buf = NULL, .buf_in = BN_GPU_VALUE_HB,
            .buf_out = -1, .buf_aux = BN_GPU_VALUE_HB2,
            .p = { (uint32_t)lw->shared.shared_gate.rows, 0, 0, 0, 0, 0, 0, 0 } };
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_MATVEC),
            .type = lw->shared.shared_down.type,
            .W_buf = qweight_backend_buf(backend, &lw->shared.shared_down),
            .buf_in = BN_GPU_VALUE_HB, .buf_out = BN_GPU_VALUE_XB2,
            .buf_aux = -1, .rows = lw->shared.shared_down.rows,
            .cols = lw->shared.shared_down.cols,
            .p = { (uint32_t)lw->shared.shared_down.rows,
                   (uint32_t)lw->shared.shared_down.cols, 1, 0, 0, 0, 0, 0 } };
        uint32_t u_one;
        { float one = 1.0f; memcpy(&u_one, &one, 4); }
        ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_WEIGHTED_ADD),
            .type = -1, .W_buf = NULL, .buf_in = BN_GPU_VALUE_MOE_OUT,
            .buf_out = -1, .buf_aux = BN_GPU_VALUE_XB2,
            .p = { (uint32_t)dim, u_one, 0, 0, 0, 0, 0, 0 } };
    }

    ops[(*n)++] = (BnGPUOp){ GPU_OP(BN_GPU_CODE_RESIDUAL_ADD),
        .type = -1, .W_buf = NULL, .buf_in = BN_GPU_VALUE_X,
        .buf_out = -1, .buf_aux = BN_GPU_VALUE_MOE_OUT,
        .p = { (uint32_t)dim, 0, 0, 0, 0, 0, 0, 0 } };
    bn_transformer_gpu_emit_rmsnorm(ops, n, next_norm,
                                    BN_GPU_VALUE_X, BN_GPU_VALUE_XB,
                                    dim, u_eps);
}

void bn_transformer_gpu_emit_logits(BnGPUOp *ops, int *n,
                                    void *logit_gpu_buf,
                                    int logit_type,
                                    int logit_rows,
                                    int logit_cols) {
    uint32_t logit_tgs = ((uint32_t)logit_rows + 31) / 32;
    uint32_t tile_x = (logit_tgs > 65535) ? 65535u : 0u;
    ops[(*n)++] = (BnGPUOp){
        .op_kind = BN_GPU_OP_LOGITS,
        GPU_OP(BN_GPU_CODE_MATVEC),
        .type = logit_type,
        .W_buf = logit_gpu_buf,
        .buf_in = BN_GPU_VALUE_XB,
        .buf_out = BN_GPU_VALUE_LOGITS,
        .buf_aux = -1,
        .rows = logit_rows, .cols = logit_cols,
        .p = { (uint32_t)logit_rows, (uint32_t)logit_cols, 1, tile_x, 0, 0, 0, 0 }
    };
}
