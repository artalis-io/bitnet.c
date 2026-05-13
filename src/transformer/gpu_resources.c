#include "gpu_internal.h"

static inline void *qweight_backend_buf(const BnBackendModel *backend,
                                        const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static inline void *backend_handle_or(const BnBackendModel *backend,
                                      int layer,
                                      BnBackendHandleRole role) {
    return bn_backend_model_handle(backend, layer, role);
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
    };
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
        .packed_qkv = qweight_backend_buf(backend, &lw->ssm.wqkv),
        .wq = qweight_backend_buf(backend, &lw->attn.wq),
        .wk = qweight_backend_buf(backend, &lw->attn.wk),
        .wv = qweight_backend_buf(backend, &lw->attn.wv),
    };
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
        .wo = qweight_backend_buf(backend, &lw->attn.wo),
    };
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

BnTransformerGPUMoESharedResources
bn_transformer_gpu_resolve_moe_shared_resources(
    const BnBackendModel *backend,
    const BnLayerWeights *lw) {
    return (BnTransformerGPUMoESharedResources){
        .shared_gate = qweight_backend_buf(backend, &lw->shared.shared_gate),
        .shared_up = qweight_backend_buf(backend, &lw->shared.shared_up),
        .shared_down = qweight_backend_buf(backend, &lw->shared.shared_down),
    };
}
