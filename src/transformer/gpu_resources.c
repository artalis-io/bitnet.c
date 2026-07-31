#include "gpu_internal.h"
#include "../moe_internal.h"
#include "backend_model.h"
#include "backend_session.h"
#include "model_internal.h"

#include <string.h>

static inline void *qweight_backend_buf(const BnBackendModel *backend,
                                        const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static inline void *backend_handle_or(const BnBackendModel *backend,
                                      int layer,
                                      BnBackendHandleRole role) {
    return bn_backend_model_handle(backend, layer, role);
}

int bn_transformer_gpu_resolve_decode_session_resources(
    BnTransformerGPUDecodeSessionResources *out,
    BnBackendSession *backend,
    int max_ops,
    int include_cached) {
    if (!out || !backend || max_ops <= 0)
        return -1;
    memset(out, 0, sizeof(*out));
    out->command_buffer = bn_backend_session_ensure_gpu_command_buffer(
        backend, max_ops, &out->command_cap);
    if (!out->command_buffer)
        return -1;
    if (include_cached) {
        out->cached_op_count =
            bn_backend_session_gpu_cached_op_count(backend);
        out->cached_has_logits =
            bn_backend_session_gpu_cached_has_logits(backend);
    }
    return 0;
}

void bn_transformer_gpu_clear_decode_session_cache(
    BnBackendSession *backend) {
    bn_backend_session_clear_gpu_cached_ops(backend);
}

void bn_transformer_gpu_store_decode_session_cache(
    BnBackendSession *backend,
    int n_ops,
    int has_logits) {
    bn_backend_session_set_gpu_cached_op_count(backend, n_ops, has_logits);
}

void *bn_transformer_gpu_resolve_output_norm(
    const BnBackendModel *backend) {
    return backend_handle_or(backend, -1, BN_BACKEND_HANDLE_OUTPUT_NORM);
}

void *bn_transformer_gpu_resolve_initial_norm(
    const BnBackendModel *backend) {
    return backend_handle_or(backend, 0, BN_BACKEND_HANDLE_ATTN_NORM);
}

void *bn_transformer_gpu_resolve_next_norm(
    const BnBackendModel *backend,
    int layer,
    int n_layers,
    void *output_norm) {
    return (layer + 1 < n_layers)
        ? backend_handle_or(backend, layer + 1, BN_BACKEND_HANDLE_ATTN_NORM)
        : output_norm;
}

BnTransformerGPULayerValidationResources
bn_transformer_gpu_resolve_layer_validation_resources(
    const BnBackendModel *backend,
    int layer) {
    return (BnTransformerGPULayerValidationResources){
        .attn_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_ATTN_NORM),
        .ffn_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_NORM),
        .q_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_Q_NORM),
        .k_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_K_NORM),
        .attn_sub_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_ATTN_SUB_NORM),
        .ffn_sub_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_SUB_NORM),
    };
}

void bn_transformer_gpu_resolve_logit_resources(
    BnTransformerGPULogitResources *out,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnWeights *w) {
    BnQWeight *ow = (BnQWeight *)&w->output_weight;
    *out = (BnTransformerGPULogitResources){
        .gpu_buf = ow->data ? qweight_backend_buf(backend, ow) : NULL,
        .type = ow->data ? ow->type : -1,
        .rows = ow->data ? ow->rows : c->vocab_size,
        .cols = ow->data ? ow->cols : c->dim,
        .cpu_weight = ow->data ? ow : NULL,
    };
    void *tied_embedding = backend_handle_or(
        backend, -1, BN_BACKEND_HANDLE_TIED_EMBEDDING);
    if (!out->gpu_buf && tied_embedding) {
        out->gpu_buf = tied_embedding;
        out->type = w->emb_type;
        out->rows = c->vocab_size;
        out->cols = c->dim;
    }
    if (!out->cpu_weight && w->token_embedding && out->type >= 0) {
        out->tied_weight.data = w->token_embedding;
        out->tied_weight.type = out->type;
        out->tied_weight.rows = c->vocab_size;
        out->tied_weight.cols = c->dim;
        out->tied_weight.scale = 1.0f;
        out->cpu_weight = &out->tied_weight;
    }
}

void *bn_transformer_gpu_resolve_tied_embedding(
    const BnBackendModel *backend) {
    return backend_handle_or(backend, -1, BN_BACKEND_HANDLE_TIED_EMBEDDING);
}

BnTransformerGPUDenseFFNResources
bn_transformer_gpu_resolve_dense_ffn_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer) {
    return (BnTransformerGPUDenseFFNResources){
        .gpu = gpu,
        .gateup_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED),
        .ffn_sub_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_SUB_NORM),
        .ffn_gate = qweight_backend_buf(backend, &lw->ffn.ffn_gate),
        .ffn_up = qweight_backend_buf(backend, &lw->ffn.ffn_up),
        .ffn_down = qweight_backend_buf(backend, &lw->ffn.ffn_down),
        .ffn_down_prefill = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL),
    };
}

int bn_transformer_gpu_resolve_dense_ffn_projection_layout(
    BnTransformerGPUDenseFFNProjectionLayout *out,
    const BnLayerWeights *lw) {
    if (!out || !lw)
        return 0;
    memset(out, 0, sizeof(*out));
    out->gate_type = lw->ffn.ffn_gate.type;
    out->gate_rows = lw->ffn.ffn_gate.rows;
    out->gate_cols = lw->ffn.ffn_gate.cols;
    out->up_type = lw->ffn.ffn_up.type;
    out->up_rows = lw->ffn.ffn_up.rows;
    out->up_cols = lw->ffn.ffn_up.cols;
    out->down_type = lw->ffn.ffn_down.type;
    out->down_rows = lw->ffn.ffn_down.rows;
    out->down_cols = lw->ffn.ffn_down.cols;
    return 1;
}

BnTransformerGPUQKVResources bn_transformer_gpu_resolve_qkv_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer) {
    return (BnTransformerGPUQKVResources){
        .gpu = gpu,
        .q_bias = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_Q_BIAS),
        .k_bias = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_K_BIAS),
        .v_bias = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_V_BIAS),
        .q_norm = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_Q_NORM),
        .k_norm = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_K_NORM),
        .qkv_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_QKV_STACKED),
        .qk_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_QK_STACKED),
        .packed_qkv = qweight_backend_buf(backend, &lw->ssm.wqkv),
        .wq = qweight_backend_buf(backend, &lw->attn.wq),
        .wk = qweight_backend_buf(backend, &lw->attn.wk),
        .wv = qweight_backend_buf(backend, &lw->attn.wv),
    };
}

int bn_transformer_gpu_resolve_qkv_projection_layout(
    BnTransformerGPUQKVProjectionLayout *out,
    const BnLayerWeights *lw) {
    if (!out || !lw)
        return 0;
    memset(out, 0, sizeof(*out));
    out->packed_type = lw->ssm.wqkv.type;
    out->packed_rows = lw->ssm.wqkv.rows;
    out->packed_cols = lw->ssm.wqkv.cols;
    out->q_type = lw->attn.wq.type;
    out->q_rows = lw->attn.wq.rows;
    out->q_cols = lw->attn.wq.cols;
    out->k_type = lw->attn.wk.type;
    out->k_rows = lw->attn.wk.rows;
    out->k_cols = lw->attn.wk.cols;
    out->v_type = lw->attn.wv.type;
    out->v_rows = lw->attn.wv.rows;
    out->v_cols = lw->attn.wv.cols;
    return 1;
}

BnTransformerGPUAttentionResources
bn_transformer_gpu_resolve_attention_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer) {
    return (BnTransformerGPUAttentionResources){
        .gpu = gpu,
        .k_bias = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_K_BIAS),
        .attn_sub_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_ATTN_SUB_NORM),
        .ffn_norm = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_FFN_NORM),
        .qk_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_QK_STACKED),
        .wv_prefill = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_WV_PREFILL),
        .wv = qweight_backend_buf(backend, &lw->attn.wv),
        .wo_prefill = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_WO_PREFILL),
        .wo = qweight_backend_buf(backend, &lw->attn.wo),
    };
}

int bn_transformer_gpu_resolve_attention_output_projection_layout(
    BnTransformerGPUAttentionOutputProjectionLayout *out,
    const BnLayerWeights *lw) {
    if (!out || !lw)
        return 0;
    memset(out, 0, sizeof(*out));
    out->out_type = lw->attn.wo.type;
    out->out_rows = lw->attn.wo.rows;
    out->out_cols = lw->attn.wo.cols;
    return 1;
}

BnTransformerGPUSSMResources bn_transformer_gpu_resolve_ssm_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer) {
    return (BnTransformerGPUSSMResources){
        .gpu = gpu,
        .ssm_qkvz_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED),
        .ssm_ab_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_SSM_AB_STACKED),
        .ssm_conv1d = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_SSM_CONV1D),
        .ssm_dt_bias = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_SSM_DT_BIAS),
        .ssm_a_log = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_SSM_A_LOG),
        .ssm_norm = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_SSM_NORM),
        .ffn_norm = backend_handle_or(backend, layer, BN_BACKEND_HANDLE_FFN_NORM),
        .wqkv = qweight_backend_buf(backend, &lw->ssm.wqkv),
        .wz = qweight_backend_buf(backend, &lw->ssm.wz),
        .ssm_alpha = qweight_backend_buf(backend, &lw->ssm.ssm_alpha),
        .ssm_beta = qweight_backend_buf(backend, &lw->ssm.ssm_beta),
        .ssm_out = qweight_backend_buf(backend, &lw->ssm.ssm_out),
    };
}

int bn_transformer_gpu_resolve_ssm_projection_layout(
    BnTransformerGPUSSMProjectionLayout *out,
    const BnLayerWeights *lw) {
    if (!out || !lw)
        return 0;
    memset(out, 0, sizeof(*out));
    out->qkv_type = lw->ssm.wqkv.type;
    out->qkv_rows = lw->ssm.wqkv.rows;
    out->qkv_cols = lw->ssm.wqkv.cols;
    out->z_type = lw->ssm.wz.type;
    out->z_rows = lw->ssm.wz.rows;
    out->z_cols = lw->ssm.wz.cols;
    out->alpha_type = lw->ssm.ssm_alpha.type;
    out->alpha_rows = lw->ssm.ssm_alpha.rows;
    out->alpha_cols = lw->ssm.ssm_alpha.cols;
    out->beta_type = lw->ssm.ssm_beta.type;
    out->beta_rows = lw->ssm.ssm_beta.rows;
    out->beta_cols = lw->ssm.ssm_beta.cols;
    out->out_type = lw->ssm.ssm_out.type;
    out->out_rows = lw->ssm.ssm_out.rows;
    out->out_cols = lw->ssm.ssm_out.cols;
    return 1;
}

BnTransformerGPUMoESharedResources
bn_transformer_gpu_resolve_moe_shared_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer) {
    BnMoESharedExpertWeights weights;
    if (!bn_moe_shared_expert_projection_weights(&weights, lw)) {
        return (BnTransformerGPUMoESharedResources){
            .gpu = gpu,
            .shared_expert_gate = backend_handle_or(
                backend, layer, BN_BACKEND_HANDLE_SHARED_EXPERT_GATE),
            .shared_gateup_stacked = backend_handle_or(
                backend, layer, BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED),
        };
    }
    return (BnTransformerGPUMoESharedResources){
        .gpu = gpu,
        .shared_gate = qweight_backend_buf(backend, weights.gate),
        .shared_up = qweight_backend_buf(backend, weights.up),
        .shared_down = qweight_backend_buf(backend, weights.down),
        .shared_expert_gate = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_SHARED_EXPERT_GATE),
        .shared_gateup_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED),
    };
}

int bn_transformer_gpu_resolve_moe_shared_projection_info(
    BnTransformerGPUMoESharedProjectionInfo *out,
    const BnLayerWeights *lw) {
    if (!out)
        return 0;
    memset(out, 0, sizeof(*out));
    BnMoESharedExpertWeights weights;
    if (!bn_moe_shared_expert_projection_weights(&weights, lw))
        return 0;
    out->gate_type = weights.gate->type;
    out->up_type = weights.up->type;
    out->down_type = weights.down->type;
    out->gate_rows = weights.gate->rows;
    out->up_rows = weights.up->rows;
    out->down_rows = weights.down->rows;
    out->gate_cols = weights.gate->cols;
    out->up_cols = weights.up->cols;
    out->down_cols = weights.down->cols;
    return 1;
}

int bn_transformer_gpu_resolve_moe_shared_ffn_resources(
    BnTransformerGPUMoESharedFFNResources *out,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int layer,
    int allow_stacked_gateup) {
    if (!out)
        return 0;
    memset(out, 0, sizeof(*out));
    if (!backend || !bn_transformer_gpu_moe_has_loaded_shared_expert(c, lw))
        return 0;

    BnTransformerGPUMoESharedResources shared =
        bn_transformer_gpu_resolve_moe_shared_resources(
            bn_backend_model_gpu(backend), backend, lw, layer);
    BnTransformerGPUMoESharedProjectionInfo info;
    if (!bn_transformer_gpu_resolve_moe_shared_projection_info(&info, lw))
        return 0;
    void *stacked_gateup = allow_stacked_gateup
        ? shared.shared_gateup_stacked
        : NULL;
    out->gate = stacked_gateup ? stacked_gateup : shared.shared_gate;
    out->up = stacked_gateup ? NULL : shared.shared_up;
    out->down = shared.shared_down;
    out->gate_weight = shared.shared_expert_gate;
    if (!out->gate || !out->down || (!out->up && !stacked_gateup)) {
        memset(out, 0, sizeof(*out));
        return 0;
    }

    BnTransformerGPUMoESharedExpertShapePolicy shared_shape =
        bn_transformer_gpu_moe_shared_expert_shape_policy(c);
    out->hidden_dim = shared_shape.hidden_dim;
    out->gate_type = info.gate_type;
    out->up_type = info.up_type;
    out->down_type = info.down_type;
    return 1;
}

BnTransformerGPUMoEDecodeResources
bn_transformer_gpu_resolve_moe_decode_resources(
    const BnBackendModel *backend,
    int layer) {
    BnTransformerGPUMoEDecodeResources resources = {0};
    if (!backend)
        return resources;

    resources.router =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER);
    resources.router_diff =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER_DIFF);
    resources.gate_all =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL);
    resources.up_all =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL);
    resources.down_all =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
    resources.has_router = resources.router || resources.router_diff;
    resources.resident_valid = resources.has_router && resources.gate_all &&
                               resources.up_all && resources.down_all;
    return resources;
}

BnTransformerGPUMoEPrefillFFNResources
bn_transformer_gpu_resolve_moe_prefill_ffn_resources(
    const BnBackendModel *backend,
    int layer) {
    BnTransformerGPUMoEPrefillFFNResources resources = {0};
    if (!backend)
        return resources;

    resources.router =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER);
    resources.gate_all =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL);
    resources.up_all =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL);
    resources.down_all =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
    resources.ffn_norm =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_FFN_NORM);
    resources.resident_valid = resources.router && resources.gate_all &&
                               resources.up_all && resources.down_all &&
                               resources.ffn_norm;
    return resources;
}

int bn_transformer_gpu_resolve_model_layer_resources(
    BnTransformerGPULayerResources *out,
    const BnModel *model,
    const BnLayerWeights *lw,
    int layer,
    void *output_norm) {
    if (!out || !model || !lw || layer < 0 ||
        layer >= model->config.n_layers)
        return -1;
    const BnBackendModel *backend = bn_model_backend(model);
    const BnGPUBackend *gpu = bn_model_gpu(model);
    *out = (BnTransformerGPULayerResources){0};
    out->next_norm = bn_transformer_gpu_resolve_next_norm(
        backend, layer, model->config.n_layers, output_norm);
    if (bn_transformer_is_attn_layer(&model->config, layer)) {
        out->qkv = bn_transformer_gpu_resolve_qkv_resources(
            gpu, backend, lw, layer);
        out->attention = bn_transformer_gpu_resolve_attention_resources(
            gpu, backend, lw, layer);
    } else {
        out->ssm = bn_transformer_gpu_resolve_ssm_resources(
            gpu, backend, lw, layer);
    }
    if (bn_transformer_gpu_layer_kind_policy(lw).uses_moe) {
        out->moe_shared = bn_transformer_gpu_resolve_moe_shared_resources(
            gpu, backend, lw, layer);
        out->moe_decode = bn_transformer_gpu_resolve_moe_decode_resources(
            backend, layer);
    } else {
        out->dense_ffn = bn_transformer_gpu_resolve_dense_ffn_resources(
            gpu, backend, lw, layer);
    }
    return 0;
}
