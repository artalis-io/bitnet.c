#include "gpu_internal.h"
#include "../moe_internal.h"
#include "backend_model.h"
#include "backend_session.h"
#include "model_internal.h"
#include "platform.h"
#include "session_internal.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>

static inline void *qweight_backend_buf(const BnBackendModel *backend,
                                        const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static inline void *backend_handle_or(const BnBackendModel *backend,
                                      int layer,
                                      BnBackendHandleRole role) {
    return bn_backend_model_handle(backend, layer, role);
}

int bn_transformer_gpu_layer_projection_resources_available(
    const BnLayerWeights *lw,
    const BnTransformerGPULayerResources *resources) {
    if (!lw || !resources)
        return 0;

    const BnTransformerGPUQKVResources *qkv = &resources->qkv;
    if (lw->ssm.wqkv.data) {
        if (!qkv->packed_qkv)
            return 0;
    } else if (lw->attn.wq.data) {
        if (!qkv->wq ||
            (lw->attn.has_kv && (!qkv->wk || !qkv->wv)))
            return 0;
    }
    if (lw->attn.wo.data && !resources->attention.wo)
        return 0;

    if (bn_transformer_gpu_layer_kind_policy(lw).uses_moe)
        return 1;

    const BnTransformerGPUDenseFFNResources *ffn = &resources->dense_ffn;
    int dense_available = (!lw->ffn.ffn_gate.data || ffn->ffn_gate) &&
           (!lw->ffn.ffn_up.data || ffn->ffn_up) &&
           (!lw->ffn.ffn_down.data || ffn->ffn_down);
    if (!dense_available)
        return 0;
    if (lw->per_layer.inp_gate.data || lw->per_layer.proj.data ||
        lw->per_layer.post_norm) {
        const BnTransformerGPUPerLayerInputResources *per =
            &resources->per_layer_input;
        return per->inp_gate && per->proj && per->post_norm;
    }
    return 1;
}

int bn_transformer_gpu_resolve_all_active_two_moe_resources(
    BnGPUMoEResources *out,
    BnGPUMoEResolvedExpert *storage,
    BnModel *model,
    BnSession *session,
    const BnLayerWeights *lw,
    int layer,
    void *router_diff,
    BnGPUMoETemporaryBuffers *temporaries) {
    if (!out || !storage || !model || !session || !lw || !router_diff ||
        !temporaries)
        return -1;
    BnTransformerGPUMoEAllActiveTwoResourcePolicy policy =
        bn_transformer_gpu_moe_all_active_two_resource_policy(
            &model->config);
    if (!policy.enabled)
        return -1;

    memset(out, 0, sizeof(*out));
    memset(temporaries, 0, sizeof(*temporaries));
    out->expert_map = &lw->moe.expert_map;
    out->experts = storage;
    out->n_experts = policy.total_experts;
    out->moe_hidden = policy.expert_hidden_dim;
    for (int e = 0; e < policy.total_experts; e++) {
        memset(&storage[e], 0, sizeof(storage[e]));
        if (bn_gpu_moe_bridge_get_expert(
                model, session, lw, layer, e, temporaries,
                &storage[e].buffers) != 0)
            return -1;
        storage[e].weight = 1.0f;
        storage[e].route_gate = router_diff;
        storage[e].route_complement =
            e >= policy.complement_route_from_expert;
    }
    return 0;
}

int bn_transformer_gpu_resolve_routed_moe_resources(
    BnGPUMoEResources *out,
    BnGPUMoEResolvedExpert *storage,
    BnModel *model,
    BnSession *session,
    const BnLayerWeights *lw,
    int layer,
    BnGPUMoETemporaryBuffers *temporaries) {
    return bn_gpu_moe_bridge_resolve_resources(
        out, storage, BN_MAX_MOE_K, model, session, lw, layer, temporaries);
}

int bn_transformer_gpu_resolve_profiled_routed_moe_resources(
    BnGPUMoEResources *out,
    BnGPUMoEResolvedExpert *storage,
    BnModel *model,
    BnSession *session,
    const BnLayerWeights *lw,
    int layer,
    BnGPUMoETemporaryBuffers *temporaries,
    int profile_enabled,
    int dim,
    int n_experts,
    double flush_ms,
    double read_ms,
    double route_ms) {
    double resolve_t0 = profile_enabled ? bn_platform_time_ms() : 0.0;
    int rc = bn_transformer_gpu_resolve_routed_moe_resources(
        out, storage, model, session, lw, layer, temporaries);
    if (rc != 0)
        return rc;
    double resolve_ms = profile_enabled
        ? bn_platform_time_ms() - resolve_t0 : 0.0;
    bn_transformer_gpu_moe_route_profile_add(
        bn_model_gpu(model), dim, n_experts, flush_ms, read_ms, route_ms,
        resolve_ms);
    return 0;
}

void bn_transformer_gpu_release_moe_temporaries(
    BnModel *model,
    BnGPUMoETemporaryBuffers *temporaries) {
    bn_gpu_moe_bridge_release_temporaries(model, temporaries);
}

int bn_transformer_gpu_flush_and_release_moe_temporaries(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnGPUMoETemporaryBuffers *temporaries) {
    if (!emit || !gpu || !model || !temporaries)
        return -1;
    int rc = bn_transformer_gpu_emit_context_flush(emit, gpu);
    bn_transformer_gpu_release_moe_temporaries(model, temporaries);
    return rc;
}

int bn_transformer_gpu_stage_token_input(
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    int token) {
    if (!gpu || !model || !session || model->config.dim <= 0)
        return -1;
    int dim = model->config.dim;
    float embedding[dim];
    bn_model_embed_token(model, embedding, token);
    if (bn_transformer_gpu_write_x(
            gpu, embedding, (size_t)dim * sizeof(float)) != 0)
        return -1;

    int per_dim = bn_transformer_per_layer_embedding_dim(&model->config);
    if (per_dim > 0) {
        size_t count = (size_t)model->config.n_layers * (size_t)per_dim;
        if (!session->state.per_layer_input ||
            bn_gpu_backend_write_activation(
                gpu, BN_GPU_VALUE_PER_LAYER_INPUT,
                session->state.per_layer_input,
                count * sizeof(float), 0) != 0)
            return -1;
    }

    int half_head = model->config.head_size / 2;
    size_t rope_count = (size_t)model->config.n_layers * (size_t)half_head;
    float *rope_freq = (float *)malloc(rope_count * sizeof(float));
    if (!rope_freq)
        return -1;
    for (int layer = 0; layer < model->config.n_layers; layer++) {
        const BnLayerWeights *lw = &model->weights.layers[layer];
        int layer_head_size = lw->attn.head_size > 0
            ? lw->attn.head_size : model->config.head_size;
        int rope_dims = bn_transformer_rope_dims_for_head(
            &model->config, layer_head_size);
        float theta = bn_transformer_rope_theta_for_head(
            &model->config, layer_head_size);
        float layer_freq[half_head];
        bn_model_transformer_policy_init_rope_frequencies_for_theta(
            theta, rope_dims, layer_freq, half_head);
        for (int i = 0; i < half_head; i++) {
            float freq = i < rope_dims / 2
                ? layer_freq[i]
                : 0.0f;
            if (i < rope_dims / 2 &&
                bn_transformer_uses_per_layer_embedding(&model->config) &&
                bn_transformer_rope_uses_base_frequency(
                    &model->config, layer_head_size) &&
                model->weights.rope_freqs) {
                if (bn_transformer_divides_rope_freqs(
                        &model->config, layer))
                    freq /= model->weights.rope_freqs[i];
                else
                    freq *= model->weights.rope_freqs[i];
            }
            rope_freq[(size_t)layer * half_head + i] = freq;
        }
    }
    int rope_rc = bn_gpu_backend_write_activation(
        gpu, BN_GPU_VALUE_ROPE_FREQ, rope_freq,
        rope_count * sizeof(float), 0);
    free(rope_freq);
    if (rope_rc != 0)
        return -1;
    return 0;
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

int bn_transformer_gpu_resolve_session_decode_resources(
    BnTransformerGPUDecodeSessionResources *out,
    const BnSession *session,
    int max_ops,
    int include_cached) {
    return bn_transformer_gpu_resolve_decode_session_resources(
        out, bn_session_backend(session), max_ops, include_cached);
}

void bn_transformer_gpu_clear_session_decode_cache(
    BnSession *session) {
    bn_transformer_gpu_clear_decode_session_cache(
        bn_session_backend(session));
}

void bn_transformer_gpu_store_session_decode_cache(
    BnSession *session,
    int n_ops,
    int has_logits) {
    bn_transformer_gpu_store_decode_session_cache(
        bn_session_backend(session), n_ops, has_logits);
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
    BnTransformerGPUDenseFFNResources resources = {
        .gpu = gpu,
        .gateup_stacked = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED),
        .ffn_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_NORM),
        .ffn_sub_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_SUB_NORM),
        .ffn_gate = qweight_backend_buf(backend, &lw->ffn.ffn_gate),
        .ffn_up = qweight_backend_buf(backend, &lw->ffn.ffn_up),
        .ffn_gate_reference = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_GATE_REFERENCE),
        .ffn_up_reference = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_UP_REFERENCE),
        .ffn_down = qweight_backend_buf(backend, &lw->ffn.ffn_down),
        .ffn_down_prefill = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL),
        .ffn_post_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_POST_NORM),
        .ffn_post_norm_1 = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_POST_NORM_1),
        .ffn_post_norm_2 = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_FFN_POST_NORM_2),
    };
    return resources;
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
        .v_unit_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_V_UNIT_NORM),
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
        .attn_post_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_ATTN_POST_NORM),
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
    resources.router_scale =
        backend_handle_or(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER_SCALE);
    resources.expert_down_scale = backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_MOE_EXPERT_DOWN_SCALE);
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
        BnMoEExecutionPolicy moe_policy =
            bn_moe_execution_policy(&model->config);
        if (moe_policy.uses_dense_residual_branch &&
            lw->norm.ffn_sub_norm) {
            out->attention.ffn_norm = backend_handle_or(
                backend, layer, BN_BACKEND_HANDLE_FFN_SUB_NORM);
        }
    } else {
        out->ssm = bn_transformer_gpu_resolve_ssm_resources(
            gpu, backend, lw, layer);
    }
    if (bn_transformer_gpu_layer_kind_policy(lw).uses_moe) {
        out->moe_shared = bn_transformer_gpu_resolve_moe_shared_resources(
            gpu, backend, lw, layer);
        out->moe_decode = bn_transformer_gpu_resolve_moe_decode_resources(
            backend, layer);
        if (bn_moe_execution_policy(
                &model->config).uses_dense_residual_branch)
            out->dense_ffn = bn_transformer_gpu_resolve_dense_ffn_resources(
                gpu, backend, lw, layer);
    } else {
        out->dense_ffn = bn_transformer_gpu_resolve_dense_ffn_resources(
            gpu, backend, lw, layer);
    }
    out->per_layer_input = (BnTransformerGPUPerLayerInputResources){
        .inp_gate = qweight_backend_buf(backend, &lw->per_layer.inp_gate),
        .proj = qweight_backend_buf(backend, &lw->per_layer.proj),
        .post_norm = backend_handle_or(
            backend, layer, BN_BACKEND_HANDLE_PER_LAYER_POST_NORM),
    };
    return 0;
}
