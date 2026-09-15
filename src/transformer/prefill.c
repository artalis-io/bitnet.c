#include "transformer_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_batched_attn_internal.h"
#include "gpu_internal.h"
#include "gpu_policy.h"
#include "model_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_prefill_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_math_internal.h"
#include "transformer_ssm_internal.h"
#include "moe.h"
#include "session.h"
#include "sh_arena.h"
#include "sh_log.h"
#include "platform.h"
#include "../moe_internal.h"

#include <stdio.h>
#include <stdlib.h>

#define BN_MAX_VLA_ELEMS 8192

static void prefill_quant_matmul_gpu(const BnModel *m, float *out,
                                     const BnQWeight *W, const float *X,
                                     int n_tokens, int8_t *quantized_buf);
static const BnPrefillCPUOps *prefill_cpu_ops(void);

static void prefill_quant_matmul_host(const BnModel *m, float *out,
                                      const BnQWeight *W, const float *X,
                                      int n_tokens, int8_t *quantized_buf) {
    if (!m || !out || !W || !X || n_tokens <= 0)
        return;
    if (n_tokens > 1 &&
        bn_transformer_prefill_quant_requires_native_cpu_prefill(W->type)) {
        bn_transformer_prefill_quant_matmul_prepared(
            out, W, NULL, X, n_tokens, quantized_buf, bn_model_pool(m));
        return;
    }
    for (int t = 0; t < n_tokens; t++)
        bn_transformer_cpu_quant_matvec(
            out + (size_t)t * W->rows, W,
            X + (size_t)t * W->cols, quantized_buf,
            bn_model_pool(m));
}

typedef struct {
    float *x;
    float *hc_residual;
} BnPrefillHCBinding;

typedef struct {
    float *inject;
    float *norm;
    float *low;
    float *gates;
} BnPrefillHCBuffers;

static BnPrefillHCBinding prefill_bind_hc_token(BnRunState *s,
                                                float *x,
                                                float *residual) {
    BnPrefillHCBinding saved = { s->x, s->hc_residual };
    s->x = x;
    s->hc_residual = residual;
    return saved;
}

static void prefill_restore_hc_token(BnRunState *s,
                                     BnPrefillHCBinding saved) {
    s->x = saved.x;
    s->hc_residual = saved.hc_residual;
}

static int prefill_apply_ple_batch(BnModel *m, BnSession *sess,
                                   BnLayerWeights *lw, float *residual,
                                   int n_tokens, int dim, size_t wide,
                                   int pos0) {
    if (!m || !sess || !lw || !residual || n_tokens <= 0 || dim <= 0 ||
        wide == 0 || (size_t)n_tokens > SIZE_MAX / wide ||
        (size_t)n_tokens > SIZE_MAX / (size_t)dim)
        return -1;
    size_t dim_values = (size_t)n_tokens * (size_t)dim;
    size_t wide_values = (size_t)n_tokens * wide;
    if (dim_values > (SIZE_MAX - wide_values) / 2u ||
        wide_values + 2u * dim_values > SIZE_MAX / sizeof(float))
        return -1;
    size_t total = wide_values + 2u * dim_values;
    float *storage = (float *)malloc(total * sizeof(float));
    if (!storage)
        return -1;
    float *embeddings = storage;
    float *keys = embeddings + dim_values;
    float *values = keys + wide_values;
    const int reference_recurrent =
        bn_transformer_prefill_host_reference_enabled(
            bn_model_gpu(m), &m->config);
    int rc = -1;
    for (int t = 0; t < n_tokens; t++) {
        if (bn_transformer_cpu_prepare_positional_layer_embedding(
                m, sess, lw, pos0 + t) != 0)
            goto done;
        memcpy(embeddings + (size_t)t * dim, sess->state.q,
               (size_t)dim * sizeof(float));
    }
    (reference_recurrent ? prefill_quant_matmul_host
                         : prefill_quant_matmul_gpu)(
        m, keys, &lw->ple.key, embeddings, n_tokens,
        reference_recurrent ? sess->state.x_q : NULL);
    (reference_recurrent ? prefill_quant_matmul_host
                         : prefill_quant_matmul_gpu)(
        m, values, &lw->ple.value, embeddings, n_tokens,
        reference_recurrent ? sess->state.x_q : NULL);
    for (int t = 0; t < n_tokens; t++) {
        BnPrefillHCBinding saved = prefill_bind_hc_token(
            &sess->state, sess->state.x,
            residual + (size_t)t * wide);
        const float *key_t = keys + (size_t)t * wide;
        const float *value_t = values + (size_t)t * dim;
        int step_rc = reference_recurrent ? -1 :
            bn_transformer_gpu_fallback_positional_layer_embedding_projected(
                bn_model_gpu(m), m, sess, lw, pos0 + t, key_t, value_t);
        if (step_rc != 0)
            step_rc =
                bn_transformer_cpu_apply_positional_layer_embedding_projected(
                    m, sess, lw, pos0 + t, key_t, value_t);
        prefill_restore_hc_token(&sess->state, saved);
        if (step_rc != 0)
            goto done;
    }
    rc = 0;
done:
    free(storage);
    return rc;
}

typedef struct {
    float *out;
    const float *residual;
    const float *weight;
    int dim;
    int streams;
    float eps;
    void (*rmsnorm)(float *, const float *, const float *, int, float);
} BnPrefillHCNormCtx;

static void prefill_hc_norm_tokens(void *vctx, int start, int end) {
    const BnPrefillHCNormCtx *c = vctx;
    size_t wide = (size_t)c->streams * c->dim;
    for (int t = start; t < end; t++) {
        for (int stream = 0; stream < c->streams; stream++) {
            size_t off = (size_t)t * wide + (size_t)stream * c->dim;
            c->rmsnorm(c->out + off, c->residual + off,
                       c->weight + (size_t)stream * c->dim, c->dim, c->eps);
        }
    }
}

typedef struct {
    float *act;
    const float *norm;
    const float *gates;
    int dim;
    int streams;
    float inv_streams;
} BnPrefillHCMixCtx;

static void prefill_hc_mix_tokens(void *vctx, int start, int end) {
    const BnPrefillHCMixCtx *c = vctx;
    size_t wide = (size_t)c->streams * c->dim;
    /* A task owns complete tokens. Preserve each token's stream-order
     * accumulation; no output element is shared between workers. */
    for (int t = start; t < end; t++) {
        for (int stream = 0; stream < c->streams; stream++) {
            size_t off = (size_t)t * wide + (size_t)stream * c->dim;
            for (int i = 0; i < c->dim; i++) {
                float gate = 1.0f / (1.0f + expf(-c->gates[off + i]));
                c->act[(size_t)t * c->dim + i] +=
                    c->norm[off + i] * gate * c->inv_streams;
            }
        }
    }
}

static int prefill_hc_mix_batch(BnModel *m, BnSession *sess,
                                const BnHyperConnectionWeights *hc,
                                float *act, float *residual,
                                BnPrefillHCBuffers buffers,
                                int n_tokens, int produce_inject,
                                const char *debug_prefix, int layer,
                                int pos0) {
    int dim = m->config.dim;
    int streams = m->config.hyper_connection_count;
    size_t wide = (size_t)streams * dim;
    if (!buffers.inject)
        return -1;
    if (buffers.norm && n_tokens > 1) {
        int rank = m->config.hyper_connection_rank;
        float *low = buffers.low;
        float *gates = buffers.gates;
        if (!low || !gates || !hc || !hc->norm || !hc->down.data ||
            !hc->up.data || (produce_inject && !hc->inject.data))
            return -1;
        float eps = bn_transformer_cpu_norm_epsilon(&m->config);
        int reference_hc =
            bn_model_transformer_policy_requires_reference_recurrent(
                &m->config);
        int used_gpu_norm = 0;
        BnGPUBackend *gpu = bn_model_gpu(m);
        if (!reference_hc &&
            bn_transformer_gpu_hyper_connection_grouped_rmsnorm_batch(
                gpu, m, hc, layer,
                strcmp(debug_prefix, "bitnet_hc_attn_norm") == 0,
                buffers.norm, residual,
                n_tokens, streams, dim, eps) == 0) {
            used_gpu_norm = 1;
        }
        if (!used_gpu_norm) {
            BnPrefillHCNormCtx norm_ctx = {
                buffers.norm, residual, hc->norm, dim, streams, eps,
                bn_transformer_rmsnorm_uses_reference_order(&m->config)
                    ? bn_transformer_rmsnorm_reference
                    : prefill_cpu_ops()->rmsnorm
            };
            BnTPTask norm_task = {
                prefill_hc_norm_tokens, &norm_ctx, n_tokens
            };
            /* This arena-backed batch path is selected by CPU backend
             * capability; other backends retain the CPU fallback. */
            bn_tp_dispatch_fine(bn_model_pool(m), &norm_task, 1);
        }
        const char *norm_weight_tag =
            strcmp(debug_prefix, "bitnet_hc_attn_norm") == 0
                ? "bitnet_hc_attn_norm_weight"
                : "bitnet_hc_ffn_norm_weight";
        bn_transformer_cpu_debug_dump_prefill_values(
            m, hc->norm, (int)wide, norm_weight_tag,
            layer, pos0 + n_tokens - 1);
        for (int t = 0; t < n_tokens; t++)
            bn_transformer_cpu_debug_dump_prefill_values(
                m, buffers.norm + (size_t)t * wide, (int)wide,
                debug_prefix, layer, pos0 + t);
        (reference_hc ? prefill_quant_matmul_host : prefill_quant_matmul_gpu)(
            m, low, &hc->down, buffers.norm, n_tokens,
            reference_hc ? sess->state.x_q : NULL);
        for (int t = 0; t < n_tokens; t++)
            bn_transformer_cpu_debug_dump_prefill_values(
                m, low + (size_t)t * rank, rank,
                "bitnet_hc_low_raw", layer, pos0 + t);
        float inv = 1.0f / (float)streams;
        if (reference_hc ||
            bn_transformer_gpu_hyper_connection_scaled_silu_batch(
                bn_model_gpu(m), low, n_tokens * rank, inv) != 0)
            for (int t = 0; t < n_tokens; t++)
                bn_transformer_scaled_silu(
                    low + (size_t)t * rank, inv, rank);
        for (int t = 0; t < n_tokens; t++)
            bn_transformer_cpu_debug_dump_prefill_values(
                m, low + (size_t)t * rank, rank,
                "bitnet_hc_low", layer, pos0 + t);
        (reference_hc ? prefill_quant_matmul_host : prefill_quant_matmul_gpu)(
            m, gates, &hc->up, low, n_tokens,
            reference_hc ? sess->state.x_q : NULL);
        for (int t = 0; t < n_tokens; t++)
            bn_transformer_cpu_debug_dump_prefill_values(
                m, gates + (size_t)t * wide, (int)wide,
                "bitnet_hc_gate_raw", layer, pos0 + t);
        if (reference_hc || bn_transformer_gpu_hyper_connection_mix_batch(
                bn_model_gpu(m), act, buffers.norm, gates,
                n_tokens, dim, streams) != 0) {
            memset(act, 0, (size_t)n_tokens * dim * sizeof(float));
            BnPrefillHCMixCtx mix_ctx = {
                act, buffers.norm, gates, dim, streams, inv
            };
            BnTPTask mix_task = {
                prefill_hc_mix_tokens, &mix_ctx, n_tokens
            };
            bn_tp_dispatch_fine(bn_model_pool(m), &mix_task, 1);
        }
        const char *mixed_tag = strcmp(debug_prefix, "bitnet_hc_attn_norm") == 0
            ? "bitnet_hc_attn_mixed" : "bitnet_hc_ffn_mixed";
        for (int t = 0; t < n_tokens; t++)
            bn_transformer_cpu_debug_dump_prefill_values(
                m, act + (size_t)t * dim, dim,
                mixed_tag, layer, pos0 + t);
        if (produce_inject)
            (reference_hc ? prefill_quant_matmul_host
                          : prefill_quant_matmul_gpu)(
                m, buffers.inject, &hc->inject, buffers.norm,
                n_tokens, reference_hc ? sess->state.x_q : NULL);
        if (produce_inject) {
            const char *inject_tag =
                strcmp(debug_prefix, "bitnet_hc_attn_norm") == 0
                    ? "bitnet_hc_attn_inject"
                    : "bitnet_hc_ffn_inject";
            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, buffers.inject + (size_t)t * streams, streams,
                    inject_tag, layer, pos0 + t);
        }
        return 0;
    }
    for (int t = 0; t < n_tokens; t++) {
        BnPrefillHCBinding saved = prefill_bind_hc_token(
            &sess->state, act + (size_t)t * dim,
            residual + (size_t)t * wide);
        int rc = bn_transformer_cpu_hyper_connection_mix(
            m, sess, hc, produce_inject && !buffers.norm);
        if (rc == 0 && produce_inject) {
            if (buffers.norm)
                memcpy(buffers.norm + (size_t)t * wide,
                       sess->state.hc_norm, wide * sizeof(float));
            else
                memcpy(buffers.inject + (size_t)t * streams,
                       sess->state.hc_inject, (size_t)streams * sizeof(float));
        }
        prefill_restore_hc_token(&sess->state, saved);
        if (rc != 0)
            return -1;
    }
    if (produce_inject && buffers.norm)
        prefill_quant_matmul_gpu(m, buffers.inject, &hc->inject, buffers.norm,
                                 n_tokens, NULL);
    return 0;
}

static void prefill_hc_combine_batch(BnModel *m, BnSession *sess,
                                     const float *block_out,
                                     const float *inject,
                                     float *residual,
                                     int n_tokens) {
    int dim = m->config.dim;
    int streams = m->config.hyper_connection_count;
    if (!bn_model_transformer_policy_requires_reference_recurrent(
            &m->config) &&
        bn_transformer_gpu_hyper_connection_combine_batch(
            bn_model_gpu(m), residual, block_out, inject,
            n_tokens, dim, streams) == 0)
        return;
    size_t wide = (size_t)m->config.hyper_connection_count * dim;
    for (int t = 0; t < n_tokens; t++) {
        float *token_residual = residual + (size_t)t * wide;
        const float *token_out = block_out + (size_t)t * dim;
        for (int stream = 0; stream < streams; stream++) {
            float scatter = 2.0f /
                (1.0f + expf(-inject[(size_t)t *
                                     m->config.hyper_connection_count + stream] /
                              (float)m->config.hyper_connection_count));
            float *dst = token_residual + (size_t)stream * dim;
            bn_transformer_cpu_scaled_residual_add(
                bn_tp_cpu_policy(bn_model_pool(m)), dst, token_out,
                scatter, dim, sess->state.hc_norm);
        }
    }
}

typedef struct {
    int enabled;
    double embed_ms;
    double attn_norm_ms;
    double qkv_ms;
    double attn_cpu_ms;
    double wo_ms;
    double ffn_norm_ms;
    double ffn_ms;
    double ffn_gateup_ms;
    double ffn_act_ms;
    double ffn_down_ms;
    double residual_ms;
    double logits_ms;
} BnPrefillProfile;

typedef struct {
    const BnModel *model;
    int layer;
    int pos0;
    int active_experts;
    int counts[BN_MOE_OBSERVE_SHARED_GATE_LOGIT + 1];
} BnPrefillMoEDumpCtx;

static void prefill_moe_debug_observe(void *opaque,
                                      BnMoEObservePoint point,
                                      int token,
                                      int slot,
                                      int expert,
                                      const float *values,
                                      int n_values) {
    BnPrefillMoEDumpCtx *ctx = (BnPrefillMoEDumpCtx *)opaque;
    if (!ctx || point < 0 || point > BN_MOE_OBSERVE_SHARED_GATE_LOGIT)
        return;
    char expert_tag[64];
    const char *tag = NULL;
    ctx->counts[point]++;
    int pos = ctx->pos0 + token;
    if (point == BN_MOE_OBSERVE_ROUTED_INPUT)
        tag = "bitnet_ffn_norm_2";
    else if (point == BN_MOE_OBSERVE_ROUTED_OUTPUT)
        tag = "bitnet_moe_raw";
    else if (point == BN_MOE_OBSERVE_ROUTED_SUM)
        tag = "bitnet_moe_routed_sum";
    else if (point == BN_MOE_OBSERVE_SHARED_OUTPUT)
        tag = "bitnet_moe_shared_output";
    else if (point == BN_MOE_OBSERVE_SHARED_GATE)
        tag = "bitnet_moe_shared_gate";
    else if (point == BN_MOE_OBSERVE_SHARED_GATE_LOGIT)
        tag = "bitnet_moe_shared_gate_logit";
    else if (point == BN_MOE_OBSERVE_ROUTED_WEIGHTED_EXPERT &&
             ctx->active_experts > 0) {
        snprintf(expert_tag, sizeof(expert_tag), "bitnet_moe_weighted_%d",
                 slot);
        tag = expert_tag;
    } else if (point == BN_MOE_OBSERVE_ROUTED_POST_NORM && slot < 0) {
        tag = "bitnet_moe_post_norm";
    } else if (point == BN_MOE_OBSERVE_ROUTED_POST_NORM) {
        snprintf(expert_tag, sizeof(expert_tag),
                 "bitnet_moe_down_%d_e%d", slot, expert);
        tag = expert_tag;
    } else if (point == BN_MOE_OBSERVE_DENSE_INPUT)
        tag = "bitnet_moe_dense_input";
    else if (point == BN_MOE_OBSERVE_DENSE_ACTIVATION)
        tag = "bitnet_moe_dense_activation";
    else if (point == BN_MOE_OBSERVE_DENSE_OUTPUT)
        tag = "bitnet_moe_dense_output";
    else if (point == BN_MOE_OBSERVE_COMBINED_OUTPUT)
        tag = "bitnet_moe_combined";
    else if (point == BN_MOE_OBSERVE_FINAL_OUTPUT)
        tag = "bitnet_moe_final";
    else if (point == BN_MOE_OBSERVE_RESIDUAL_OUTPUT)
        tag = "bitnet_moe_residual";
    else if (point == BN_MOE_OBSERVE_ROUTE_WEIGHTS)
        tag = "bitnet_moe_weights";
    else if (point == BN_MOE_OBSERVE_ROUTER_LOGITS)
        tag = "bitnet_moe_logits";
    else if (point == BN_MOE_OBSERVE_ROUTED_ACTIVATION) {
        snprintf(expert_tag, sizeof(expert_tag),
                 "bitnet_moe_swiglu_%d_e%d", slot, expert);
        tag = expert_tag;
    }
    if (!tag)
        return;
    bn_transformer_cpu_debug_dump_prefill_values(
        ctx->model, values, n_values, tag, ctx->layer, pos);
}

static const BnCPURuntimePolicy *prefill_cpu_runtime(const BnModel *m) {
    return bn_tp_cpu_policy(bn_model_pool(m));
}

static inline double prefill_profile_now(const BnPrefillProfile *p) {
    return p && p->enabled ? bn_platform_time_ms() : 0.0;
}

static inline void prefill_profile_add(double *dst, double start) {
    if (start > 0.0)
        *dst += bn_platform_time_ms() - start;
}

static inline void *prefill_qweight_backend_buf(const BnBackendModel *backend,
                                                const BnQWeight *w) {
    return (void *)bn_transformer_prefill_qweight_gpu_buffer_policy(backend,
                                                                    w);
}

static float *prefill_decode_tokens(BnModel *m, BnSession *sess,
                                    const int *tokens, int n_tokens,
                                    int pos0, float *all_logits,
                                    int need_last_logits) {
    BnConfig *c = &m->config;
    float *logits = NULL;
    if (all_logits) {
        for (int t = 0; t < n_tokens; t++) {
            logits = bn_transformer_forward(m, sess, tokens[t], pos0 + t);
            if (!logits)
                return NULL;
            memcpy(all_logits + (size_t)t * c->vocab_size, logits,
                   (size_t)c->vocab_size * sizeof(float));
        }
        if (bn_model_gpu(m))
            sess->gpu_kv_direct_valid = 1;
        return logits;
    }
    if (need_last_logits) {
        for (int t = 0; t + 1 < n_tokens; t++)
            if (bn_transformer_forward_no_logits(
                    m, sess, tokens[t], pos0 + t) != 0)
                return NULL;
        logits = bn_transformer_forward(m, sess, tokens[n_tokens - 1],
                                        pos0 + n_tokens - 1);
        if (logits && bn_model_gpu(m))
            sess->gpu_kv_direct_valid = 1;
        return logits;
    }
    for (int t = 0; t < n_tokens; t++)
        if (bn_transformer_forward_no_logits(m, sess, tokens[t],
                                             pos0 + t) != 0)
            return NULL;
    if (bn_model_gpu(m))
        sess->gpu_kv_direct_valid = 1;
    return sess->state.x;
}

static float *prefill_decode_tokens_with_logits(BnModel *m, BnSession *sess,
                                                const int *tokens,
                                                int n_tokens,
                                                int pos0,
                                                float *all_logits,
                                                int need_last_logits) {
    BnConfig *c = &m->config;
    float *logits = NULL;
    for (int t = 0; t < n_tokens; t++) {
        logits = bn_transformer_forward(m, sess, tokens[t], pos0 + t);
        if (!logits)
            return NULL;
        if (all_logits)
            memcpy(all_logits + (size_t)t * c->vocab_size, logits,
                   (size_t)c->vocab_size * sizeof(float));
    }
    return need_last_logits ? logits : (logits ? sess->state.x : NULL);
}

static int prefill_uses_float_kquant_fallback(const BnModel *m) {
    if (!m)
        return 0;
    BnTransformerPrefillFloatKQuantFallbackPolicy policy =
        bn_transformer_prefill_float_kquant_fallback_policy(&m->config);
    return policy.enabled;
}

static void prefill_quant_matmul_float_kquant_fallback(
        const BnModel *m, float **out, const BnQWeight **W, int n,
        const float *X, int n_tokens, int8_t *quantized_buf) {
    BnTransformerPrefillQuantMatmulResourcePolicy resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            prefill_cpu_runtime(m), bn_model_backend(m), W, n, 4);
    if (!resources.valid)
        return;
    (void)quantized_buf;
    for (int i = 0; i < n; i++)
        bn_transformer_prefill_quant_matmul_float_x(
            out[i], W[i], X, n_tokens, bn_model_pool(m));
}

static void prefill_replay_cpu_projection(const BnModel *m,
                                          float **out,
                                          const BnQWeight **W,
                                          int n,
                                          const float *X,
                                          int n_tokens,
                                          int8_t *quantized_buf) {
    BnCPUPrefillProjectionReplayKind replay =
        bn_transformer_cpu_backend_prefill_projection_replay();
    if (!m || bn_model_gpu(m) || n_tokens <= 1 ||
        replay == BN_CPU_PREFILL_PROJECTION_REPLAY_NONE)
        return;
    for (int i = 0; i < n; i++) {
        if (!W[i])
            continue;
        if (replay == BN_CPU_PREFILL_PROJECTION_REPLAY_NATIVE_ALL_TOKENS) {
            if (bn_transformer_prefill_quant_matmul_matches_matvec(W[i]->type))
                continue;
            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_quant_matvec(
                    out[i] + (size_t)t * W[i]->rows, W[i],
                    X + (size_t)t * W[i]->cols, quantized_buf,
                    bn_model_pool(m));
            continue;
        }
        if (!bn_transformer_prefill_uses_float_kquant_fallback(W[i]->type))
            continue;
        bn_transformer_prefill_quant_matmul_float_x(
            out[i] + (size_t)(n_tokens - 1) * W[i]->rows, W[i],
            X + (size_t)(n_tokens - 1) * W[i]->cols, 1,
            bn_model_pool(m));
    }
}

static void prefill_quant_matmul_gpu(const BnModel *m,
                                     float *out,
                                     const BnQWeight *W,
                                     const float *X,
                                     int n_tokens,
                                     int8_t *quantized_buf) {
    const BnQWeight *weights[1] = { W };
    BnGPUBackend *gpu = bn_model_gpu(m);
    int gpu_available =
        bn_transformer_prefill_quant_matmul_gpu_available(
            gpu, out != NULL, W != NULL,
            prefill_qweight_backend_buf(bn_model_backend(m), W) != NULL,
            X != NULL);
    BnTransformerPrefillQuantMatmulDispatchPolicy dispatch =
        bn_transformer_prefill_quant_matmul_dispatch_policy_for(
            &m->config, weights, 1, 4, gpu_available, 0, 0);
    if (!dispatch.valid)
        return;

    switch (dispatch.path) {
    case BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_FLOAT_KQUANT_FALLBACK: {
        float *outs[1] = { out };
        prefill_quant_matmul_float_kquant_fallback(
            m, outs, weights, 1, X, n_tokens, quantized_buf);
        return;
    }
    case BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_SINGLE:
    case BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_PREPARED_MULTI: {
        BnTransformerPrefillQuantMatmulResourcePolicy resources =
            bn_transformer_prefill_quant_matmul_resource_policy(
                prefill_cpu_runtime(m), bn_model_backend(m), weights, 1, 1);
        if (!resources.valid)
            return;
        bn_transformer_prefill_quant_matmul_prepared(
            out, W, resources.prepared[0],
            X, n_tokens, quantized_buf, bn_model_pool(m));
        float *outs[1] = { out };
        prefill_replay_cpu_projection(
            m, outs, weights, 1, X, n_tokens, quantized_buf);
        return;
    }
    case BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_SINGLE:
        bn_transformer_prefill_quant_matmul_gpu_buffer(
            out, W, prefill_qweight_backend_buf(bn_model_backend(m), W), X,
            n_tokens, quantized_buf, bn_model_pool(m), bn_model_gpu(m));
        return;
    case BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_BATCH:
        return;
    }
}

static int prefill_quant_matmul_gpu_buf(const BnModel *m,
                                        float *out,
                                        const BnQWeight *W,
                                        void *buf,
                                        const float *X,
                                        int n_tokens) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    if (!bn_transformer_prefill_quant_matmul_gpu_available(
            gpu, out != NULL, W != NULL, buf != NULL, X != NULL))
        return -1;
    return bn_transformer_prefill_quant_matmul_gpu_buffer_run(
        out, W, buf, X, n_tokens, gpu);
}

static void prefill_quant_matmul_multi(const BnModel *m,
                                       float **out,
                                       const BnQWeight **W,
                                       int n,
                                       const float *X,
                                       int n_tokens,
    int8_t *quantized_buf) {
    if (!bn_model_gpu(m)) {
        BnTransformerPrefillQuantMatmulDispatchPolicy dispatch =
            bn_transformer_prefill_quant_matmul_dispatch_policy_for(
                &m->config, W, n, 4, 0, 0, 0);
        if (!dispatch.valid)
            return;
        if (dispatch.path ==
            BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_FLOAT_KQUANT_FALLBACK) {
            prefill_quant_matmul_float_kquant_fallback(
                m, out, W, n, X, n_tokens, quantized_buf);
            return;
        }
        if (dispatch.path == BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_SINGLE) {
            for (int i = 0; i < n; i++)
                prefill_quant_matmul_gpu(m, out[i], W[i], X, n_tokens,
                                         quantized_buf);
            return;
        }
        BnTransformerPrefillQuantMatmulResourcePolicy resources =
            bn_transformer_prefill_quant_matmul_resource_policy(
                prefill_cpu_runtime(m), bn_model_backend(m), W, n, 4);
        if (!resources.valid)
            return;
        bn_transformer_prefill_quant_matmul_prepared_multi(
            out, W, resources.prepared, n, X, n_tokens, quantized_buf,
            bn_model_pool(m));
        prefill_replay_cpu_projection(
            m, out, W, n, X, n_tokens, quantized_buf);
        return;
    }
    const BnBackendModel *backend = bn_model_backend(m);
    BnTransformerPrefillQuantMatmulResourcePolicy resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            prefill_cpu_runtime(m), backend, W, n, 16);
    BnMatvecTask tasks[16];
    const void *bufs[16];
    int gpu_batch_available =
        bn_transformer_prefill_quant_matmul_batch_gpu_available(
            bn_model_gpu(m), n, 1, 1, 1, 1);
    if (resources.valid) {
        for (int i = 0; i < n; i++) {
            tasks[i] = (BnMatvecTask){ out[i], W[i], NULL, 0 };
            bufs[i] = resources.gpu_buffers[i];
        }
    }
    BnTransformerPrefillQuantMatmulDispatchPolicy dispatch =
        bn_transformer_prefill_quant_matmul_dispatch_policy_for(
            &m->config, W, n, 4, 1, gpu_batch_available,
            resources.all_gpu_buffers_available);
    if (!dispatch.valid)
        return;
    if (dispatch.path == BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_BATCH &&
        resources.valid) {
        bn_transformer_prefill_quant_matmul_batch_gpu_buffers(
            tasks, bufs, n, X, n_tokens, W[0]->cols, quantized_buf,
            bn_model_pool(m), bn_model_gpu(m));
        return;
    }
    for (int i = 0; i < n; i++)
        prefill_quant_matmul_gpu(m, out[i], W[i], X, n_tokens, quantized_buf);
}

static void prefill_attention_input_norm(
        const BnModel *m, float *out, const float *input,
        const float *weight, void *norm_buf, int n_tokens, int dim,
        float eps, int copy_input) {
    if (copy_input) {
        memcpy(out, input, (size_t)n_tokens * (size_t)dim * sizeof(float));
        return;
    }
    if (norm_buf && bn_transformer_gpu_prefill_rmsnorm_backend_run(
            bn_model_gpu(m), out, norm_buf, input, n_tokens, dim, eps) == 0)
        return;
    for (int t = 0; t < n_tokens; t++)
        prefill_cpu_ops()->rmsnorm(out + (size_t)t * dim,
                                   input + (size_t)t * dim, weight, dim, eps);
}

static int prefill_qk_stacked_gpu(const BnModel *m,
                                  const BnLayerWeights *lw,
                                  float *q_tmp,
                                  float *k_out,
                                  const float *X,
                                  int n_tokens,
                                  int q_stride,
                                  int dim,
                                  int layer) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    if (!bn_transformer_prefill_quant_matmul_gpu_available(
            gpu, q_tmp != NULL, lw != NULL, backend != NULL, X != NULL) ||
        !k_out)
        return -1;
    if (!bn_transformer_prefill_qk_stack_compatible(
            &lw->attn.wq, &lw->attn.wk, q_stride, dim))
        return -1;
    BnTransformerPrefillAttentionProjectionTypes attn_types;
    if (!bn_transformer_prefill_resolve_attention_projection_types(
            &attn_types, lw))
        return -1;
    BnTransformerPrefillStackedAttentionGPUResourcePolicy resources =
        bn_transformer_prefill_stacked_attention_gpu_resource_policy(
            backend, layer, attn_types);
    if (!resources.qk_valid)
        return -1;
    if (!bn_transformer_gpu_prefill_stacked_projection_allowed(
            gpu, resources.qk_type, n_tokens)) return -1;
    int rows = resources.qk_rows;
    if (bn_transformer_gpu_prefill_quant_matmul_backend_run(
            gpu, q_tmp, (void *)resources.qk, X, rows, dim, n_tokens,
            resources.qk_type) != 0)
        return -1;
    for (int t = n_tokens - 1; t >= 0; t--) {
        float *src = q_tmp + (size_t)t * rows;
        memcpy(k_out + (size_t)t * attn_types.k_rows,
               src + attn_types.q_rows,
               (size_t)attn_types.k_rows * sizeof(float));
        memmove(q_tmp + (size_t)t * q_stride, src,
                (size_t)attn_types.q_rows * sizeof(float));
    }
    return 0;
}

static int prefill_qkv_stacked_batch_gpu(const BnModel *m,
                                         const BnLayerWeights *lw,
                                         float *q_tmp,
                                         float *k_out,
                                         float *v_out,
                                         const float *X,
                                         int n_tokens,
                                         int q_stride,
                                         int dim,
                                         int layer) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    if (!bn_transformer_prefill_quant_matmul_batch_gpu_available(
            gpu, 2, q_tmp != NULL && k_out != NULL && v_out != NULL,
            lw != NULL, backend != NULL, X != NULL))
        return -1;
    if (!bn_transformer_prefill_qkv_stack_batch_compatible(
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, q_stride, dim))
        return -1;
    BnTransformerPrefillAttentionProjectionTypes attn_types;
    if (!bn_transformer_prefill_resolve_attention_projection_types(
            &attn_types, lw))
        return -1;
    BnTransformerPrefillStackedAttentionGPUResourcePolicy resources =
        bn_transformer_prefill_stacked_attention_gpu_resource_policy(
            backend, layer, attn_types);
    if (!resources.qkv_valid)
        return -1;

    if (!bn_transformer_gpu_prefill_stacked_projection_allowed(
            gpu, resources.qk_type, n_tokens)) return -1;
    int qk_rows = resources.qk_rows;
    BnGPUMatvecOp ops[2] = {
        {
            .out = q_tmp,
            .W_buf = (void *)resources.qk,
            .rows = qk_rows,
            .cols = dim,
            .type = resources.qk_type,
        },
        {
            .out = v_out,
            .W_buf = (void *)resources.wv,
            .rows = resources.wv_rows,
            .cols = dim,
            .type = resources.wv_type,
        },
    };
    if (bn_transformer_gpu_prefill_quant_matmul_batch_backend_run(
            gpu, ops, 2, X, n_tokens, dim) != 0)
        return -1;

    for (int t = n_tokens - 1; t >= 0; t--) {
        float *src = q_tmp + (size_t)t * qk_rows;
        memcpy(k_out + (size_t)t * attn_types.k_rows,
               src + attn_types.q_rows,
               (size_t)attn_types.k_rows * sizeof(float));
        memmove(q_tmp + (size_t)t * q_stride, src,
                (size_t)attn_types.q_rows * sizeof(float));
    }
    return 0;
}

static int prefill_dense_ffn_gpu_batch(const BnModel *m,
                                       float *out,
                                       const BnLayerWeights *lw,
                                       const float *X,
                                       int n_tokens,
                                       int dim,
                                       int hidden_dim,
                                       int act_type,
                                       int layer,
                                       void *norm_buf,
                                       float norm_eps,
                                       int add_residual) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    BnTransformerPrefillFFNProjectionTypes ffn_types;
    if (!bn_transformer_prefill_resolve_ffn_projection_types(&ffn_types, lw))
        return -1;
    if (!bn_transformer_prefill_dense_ffn_batch_gpu_available(
            gpu, backend != NULL, lw->ffn.ffn_gate.data != NULL,
            lw->ffn.ffn_up.data != NULL, lw->ffn.ffn_down.data != NULL))
        return -1;

    BnTransformerPrefillDenseFFNGPUResourcePolicy ffn_resources =
        bn_transformer_prefill_dense_ffn_gpu_resource_policy(
            backend, layer, lw, ffn_types);
    if (!ffn_resources.valid)
        return -1;

    BnTransformerPrefillFFNBatchCallPolicy call_policy =
        bn_transformer_prefill_ffn_batch_call_policy(
            norm_buf != NULL, add_residual,
            bn_transformer_gpu_prefill_dense_ffn_batch_norm_backend_available(
                gpu),
            bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_available(
                gpu));
    if (call_policy.kind == BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM_RESID) {
        return bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_run(
            gpu, out, (void *)ffn_resources.gate,
            (void *)ffn_resources.up, (void *)ffn_resources.down,
            norm_buf, X, n_tokens, dim, hidden_dim, ffn_types.gate_type,
            ffn_types.up_type, ffn_types.down_type, act_type,
            norm_eps);
    }
    if (call_policy.kind == BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM) {
        return bn_transformer_gpu_prefill_dense_ffn_batch_norm_backend_run(
            gpu, out, (void *)ffn_resources.gate,
            (void *)ffn_resources.up, (void *)ffn_resources.down,
            norm_buf, X, n_tokens, dim, hidden_dim, ffn_types.gate_type,
            ffn_types.up_type, ffn_types.down_type, act_type,
            norm_eps);
    }

    return bn_transformer_gpu_prefill_dense_ffn_batch_backend_run(
        gpu, out, (void *)ffn_resources.gate, (void *)ffn_resources.up,
        (void *)ffn_resources.down, X, n_tokens, dim, hidden_dim,
        ffn_types.gate_type, ffn_types.up_type, ffn_types.down_type,
        act_type);
}

static int prefill_dense_layer_gpu_batch(const BnModel *m,
                                         float *out,
                                         const BnLayerWeights *lw,
                                         const float *X,
                                         float *K_out,
                                         float *V_out,
                                         int n_tokens,
                                         int dim,
                                         int hidden_dim,
                                         int n_heads,
                                         int n_kv_heads,
                                         int head_size,
                                         int kv_mul,
                                         int kv_dim,
                                         int rope_dims,
                                         int layer,
                                         int pos0,
                                         uint32_t kv_cache_off,
                                         int kv_cache_stride,
                                         int qk_norm_per_head,
                                         int normalize_v,
                                         int rope_freq_stride,
                                         float attention_scale,
                                         int final_ffn_last_row_only) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    BnTransformerPrefillFFNProjectionTypes ffn_types;
    if (!bn_transformer_prefill_resolve_ffn_projection_types(&ffn_types, lw))
        return -1;
    BnTransformerPrefillAttentionProjectionTypes attn_types;
    if (!bn_transformer_prefill_resolve_attention_projection_types(
            &attn_types, lw))
        return -1;
    BnTransformerPrefillSSMProjectionTypes ssm_types;
    if (!bn_transformer_prefill_resolve_ssm_projection_types(&ssm_types, lw))
        return -1;
    int q_dim = n_heads * head_size;
    int has_split_qkv =
        lw->attn.wq.data && lw->attn.wk.data && lw->attn.wv.data;
    int has_packed_qkv =
        bn_transformer_weight_is_packed_qkv(&lw->ssm.wqkv, dim,
                                            q_dim, kv_dim);
    if (!bn_transformer_prefill_dense_layer_gpu_available(
            gpu, backend != NULL, has_split_qkv || has_packed_qkv,
            lw->attn.wo.data != NULL, lw->ffn.ffn_gate.data != NULL,
            lw->ffn.ffn_up.data != NULL, lw->ffn.ffn_down.data != NULL))
        return -1;
    int activation = bn_transformer_prefill_config_activation(&m->config);

    BnTransformerPrefillDenseLayerGPUResourcePolicy resources =
        bn_transformer_prefill_dense_layer_gpu_resource_policy(
            backend, layer, lw, has_packed_qkv, attn_types, ssm_types,
            ffn_types);
    if (!resources.valid)
        return -1;

    return bn_transformer_gpu_prefill_dense_layer_backend_run(
        gpu, out, (void *)resources.qk, (void *)resources.wv,
        (void *)resources.wo, (void *)resources.gate,
        (void *)resources.up, (void *)resources.down,
        (void *)resources.attn_norm, (void *)resources.ffn_norm,
        (void *)resources.attn_post_norm,
        (void *)resources.ffn_post_norm,
        (void *)resources.q_norm, (void *)resources.k_norm,
        (void *)resources.q_bias, (void *)resources.k_bias,
        (void *)resources.v_bias, X, K_out, V_out, n_tokens, dim,
        hidden_dim, n_heads, n_kv_heads, head_size, kv_mul, kv_dim,
        resources.qk_rows, resources.qk_type,
        resources.wv_rows, resources.wv_type, attn_types.out_rows,
        attn_types.out_cols, attn_types.out_type, ffn_types.gate_type,
        ffn_types.up_type, ffn_types.down_type, activation,
        qk_norm_per_head,
        normalize_v,
        bn_transformer_prefill_norm_epsilon(&m->config), pos0, rope_dims,
        (size_t)layer * (size_t)rope_freq_stride,
        kv_cache_off, kv_cache_stride, attention_scale,
        lw->norm.layer_output_scale ? lw->norm.layer_output_scale[0] : 1.0f,
        bn_transformer_attention_window(&m->config, layer),
        final_ffn_last_row_only);
}

static int prefill_moe_layer_gpu_batch(const BnModel *m,
                                       float *out,
                                       const BnLayerWeights *lw,
                                       const float *X,
                                       float *K_out,
                                       float *V_out,
                                       int n_tokens,
                                       int dim,
                                       int n_heads,
                                       int n_kv_heads,
                                       int head_size,
                                       int kv_mul,
                                       int kv_dim,
                                       int rope_dims,
                                       int layer,
                                       int pos0,
                                       uint32_t kv_cache_off,
                                       int kv_cache_stride,
                                       int qk_norm_per_head,
                                       float attention_scale) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&m->config);
    BnTransformerPrefillLayerKindPolicy layer_kind =
        bn_transformer_prefill_layer_kind_policy(lw);
    BnTransformerPrefillAttentionProjectionTypes attn_types;
    if (!bn_transformer_prefill_resolve_attention_projection_types(
            &attn_types, lw))
        return -1;
    if (!bn_transformer_prefill_moe_layer_backend_available(
            gpu, &m->config, &lw->moe.expert_map, dim, 0) ||
        !backend ||
        !layer_kind.uses_moe || !lw->attn.wq.data ||
        !lw->attn.wk.data || !lw->attn.wv.data || !lw->attn.wo.data)
        return -1;
    int activation = bn_transformer_prefill_config_activation(&m->config);

    BnTransformerPrefillMoELayerGPUResourcePolicy resources =
        bn_transformer_prefill_moe_layer_gpu_resource_policy(
            backend, &m->config, layer, lw, attn_types);
    if (!resources.valid)
        return -1;
    BnMoERoutedExpertProjectionTypes routed_types;
    if (!bn_moe_routed_expert_projection_types(
            &routed_types, &lw->moe.expert_map))
        return -1;

    return bn_transformer_gpu_prefill_moe_layer_backend_run(
        gpu, out, (void *)resources.qk, (void *)resources.wv,
        (void *)resources.wo, (void *)resources.router,
        (void *)resources.gate_all, (void *)resources.up_all,
        (void *)resources.down_all, (void *)resources.shared_gate,
        (void *)resources.shared_up, (void *)resources.shared_down,
        (void *)resources.shared_gate_weight, (void *)resources.attn_norm,
        (void *)resources.ffn_norm, (void *)resources.q_norm,
        (void *)resources.k_norm, (void *)resources.q_bias,
        (void *)resources.k_bias, (void *)resources.v_bias,
        X, K_out, V_out, n_tokens, dim, route_policy.expert_hidden_dim,
        route_policy.total_experts, route_policy.active_experts, n_heads,
        n_kv_heads, head_size, kv_mul, kv_dim,
        resources.qk_rows, resources.qk_type,
        resources.wv_rows, resources.wv_type, attn_types.out_rows,
        attn_types.out_cols, attn_types.out_type,
        routed_types.gate_type, routed_types.up_type,
        routed_types.down_type, activation,
        resources.shared_hidden_dim, resources.shared_gate_type,
        resources.shared_up_type, resources.shared_down_type,
        qk_norm_per_head,
        bn_transformer_prefill_norm_epsilon(&m->config), pos0, rope_dims,
        kv_cache_off, kv_cache_stride,
        attention_scale, route_policy.norm_topk_prob,
        route_policy.expert_weights_scale, bn_transformer_attention_window(&m->config, layer));
}

static int prefill_moe_ffn_gpu_batch(const BnModel *m,
                                     float *out,
                                     const BnLayerWeights *lw,
                                     const float *X,
                                     int n_tokens,
                                     int dim,
                                     int layer) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!lw)
        return 0;
    BnTransformerPrefillLayerKindPolicy layer_kind =
        bn_transformer_prefill_layer_kind_policy(lw);
    if (!bn_transformer_prefill_moe_ffn_batch_available(
            gpu, c, &lw->moe.expert_map, dim, 0) ||
        !backend ||
        !layer_kind.uses_moe || n_tokens <= 0 || dim <= 0)
        return -1;

    BnTransformerPrefillMoEFFNGPUResourcePolicy resources =
        bn_transformer_prefill_moe_ffn_gpu_resource_policy(
            backend, c, layer, lw);
    if (!resources.valid)
        return -1;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(c);
    int activation = bn_transformer_prefill_config_activation(c);
    BnMoERoutedExpertProjectionTypes routed_types;
    if (!bn_moe_routed_expert_projection_types(
            &routed_types, &lw->moe.expert_map))
        return -1;

    return bn_transformer_gpu_prefill_moe_ffn_batch_backend_run(
        gpu, out, (void *)resources.router, (void *)resources.gate_all,
        (void *)resources.up_all, (void *)resources.down_all,
        (void *)resources.shared_gate, (void *)resources.shared_up,
        (void *)resources.shared_down, (void *)resources.shared_gate_weight,
        (void *)resources.ffn_norm, X, n_tokens, dim,
        route_policy.expert_hidden_dim, route_policy.total_experts,
        route_policy.active_experts,
        routed_types.gate_type, routed_types.up_type,
        routed_types.down_type, activation, resources.shared_hidden_dim,
        resources.shared_gate_type, resources.shared_up_type,
        resources.shared_down_type, bn_transformer_prefill_norm_epsilon(c),
        route_policy.norm_topk_prob,
        route_policy.expert_weights_scale);
}

static int prefill_ssm_moe_layer_chain_ready(const BnModel *m,
                                             const BnLayerWeights *lw,
                                             int layer,
                                             int n_tokens) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!lw)
        return 0;
    if (!bn_transformer_prefill_ssm_gpu_layer_enabled(gpu, c, lw))
        return 0;
    BnTransformerPrefillLayerKindPolicy layer_kind =
        bn_transformer_prefill_layer_kind_policy(lw);
    int use_attn_post_norm =
        bn_transformer_attention_uses_post_norm_layer(c, lw);
    int use_ffn_post_norm =
        bn_transformer_ffn_uses_post_norm_layer(c, lw);
    BnTransformerSSMShapePolicy ssm_shape;
    if (!bn_transformer_ssm_shape_policy(&ssm_shape, c))
        return 0;
    BnTransformerPrefillSSMMoEChainPolicy policy =
        bn_transformer_prefill_ssm_moe_chain_policy(
            bn_transformer_prefill_ssm_moe_chain_available(
                gpu, c, &lw->moe.expert_map, c ? c->dim : 0, 0,
                n_tokens),
            layer_kind,
            lw->norm.ffn_sub_norm != NULL,
            lw->norm.layer_output_scale != NULL,
            use_attn_post_norm || use_ffn_post_norm,
            lw->norm.attn_post_norm != NULL,
            lw->norm.ffn_post_norm != NULL,
            &ssm_shape);
    if (!policy.enabled ||
        !backend ||
        !lw->ssm.wqkv.data || !lw->ssm.wz.data ||
        !lw->ssm.ssm_alpha.data || !lw->ssm.ssm_beta.data ||
        !lw->ssm.ssm_out.data || !lw->norm.attn_norm ||
        !lw->ssm.ssm_conv1d || !lw->ssm.ssm_dt_bias ||
        !lw->ssm.ssm_a || !lw->ssm.ssm_norm)
        return 0;

    BnTransformerPrefillFFNProjectionTypes unused_ffn_types = {0};
    BnTransformerPrefillSSMGPUResourcePolicy ssm_resources =
        bn_transformer_prefill_ssm_gpu_resource_policy(
            backend, layer, lw, 0, unused_ffn_types);
    BnTransformerPrefillMoEFFNGPUResourcePolicy moe_resources =
        bn_transformer_prefill_moe_ffn_gpu_resource_policy(
            backend, c, layer, lw);
    return ssm_resources.valid && moe_resources.valid;
}

static int prefill_dense_layer_chain_ready(const BnModel *m,
                                           const BnLayerWeights *lw,
                                           const BnLayerShapePlan *plan,
                                           int layer,
                                           int n_tokens,
                                           float layer_rope_theta) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!lw)
        return 0;
    BnTransformerPrefillFFNProjectionTypes ffn_types;
    if (!bn_transformer_prefill_resolve_ffn_projection_types(&ffn_types, lw))
        return 0;
    BnTransformerPrefillAttentionProjectionTypes attn_types;
    if (!bn_transformer_prefill_resolve_attention_projection_types(
            &attn_types, lw))
        return 0;
    BnTransformerPrefillSSMProjectionTypes ssm_types;
    if (!bn_transformer_prefill_resolve_ssm_projection_types(&ssm_types, lw))
        return 0;
    BnTransformerPrefillLayerKindPolicy layer_kind =
        bn_transformer_prefill_layer_kind_policy(lw);
    int q_dim = plan->q_dim > 0 ? plan->q_dim
                                : plan->n_heads * plan->head_size;
    int has_split_qkv =
        lw->attn.wq.data && lw->attn.wk.data && lw->attn.wv.data;
    int has_packed_qkv =
        bn_transformer_weight_is_packed_qkv(&lw->ssm.wqkv, c->dim,
                                            q_dim, plan->kv_dim);
    int use_attn_post_norm =
        bn_transformer_attention_uses_post_norm_layer(c, lw);
    int use_ffn_post_norm =
        bn_transformer_ffn_uses_post_norm_layer(c, lw);
    BnTransformerPrefillDenseLayerChainPolicy policy =
        bn_transformer_prefill_dense_layer_chain_policy(
            gpu != NULL,
            bn_transformer_prefill_dense_layer_gpu_available(
                gpu, backend != NULL, has_split_qkv || has_packed_qkv,
                lw->attn.wo.data != NULL, lw->ffn.ffn_gate.data != NULL,
                lw->ffn.ffn_up.data != NULL, lw->ffn.ffn_down.data != NULL),
            bn_model_tq_state(m) != NULL,
            n_tokens,
            bn_transformer_prefill_dense_chain_min_tokens(c, gpu),
            layer_rope_theta,
            bn_transformer_rope_base_theta(c),
            plan->is_attn,
            layer_kind,
            bn_transformer_prefill_has_ffn_gate(c),
            lw->ffn.ffn_up.data != NULL,
            lw->norm.attn_sub_norm != NULL,
            lw->norm.ffn_sub_norm != NULL,
            lw->norm.layer_output_scale != NULL,
            use_attn_post_norm || use_ffn_post_norm,
            lw->norm.attn_post_norm != NULL,
            lw->norm.ffn_post_norm != NULL);
    if (!policy.enabled)
        return 0;

    BnTransformerPrefillDenseLayerGPUResourcePolicy resources =
        bn_transformer_prefill_dense_layer_gpu_resource_policy(
            backend, layer, lw, has_packed_qkv, attn_types, ssm_types,
            ffn_types);
    return resources.valid;
}

static int prefill_moe_layer_chain_ready(const BnModel *m,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         int layer,
                                         int n_tokens,
                                         float layer_rope_theta) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    int moe_layer_backend_available =
        bn_transformer_prefill_moe_layer_backend_available(
            gpu, c, lw ? &lw->moe.expert_map : NULL, c ? c->dim : 0, 0);
    int moe_layer_chain_available =
        bn_transformer_prefill_moe_layer_chain_available(
            gpu, c, lw ? &lw->moe.expert_map : NULL, c ? c->dim : 0, 0,
            n_tokens);
    BnTransformerPrefillLayerKindPolicy layer_kind =
        bn_transformer_prefill_layer_kind_policy(lw);
    BnTransformerMoESharedExpertShapePolicy shared_policy =
        bn_transformer_moe_shared_expert_shape_policy(c, lw);
    if (!moe_layer_chain_available ||
        !backend ||
        bn_model_tq_state(m) != NULL ||
        layer_rope_theta != bn_transformer_rope_base_theta(c) ||
        !plan->is_attn || !layer_kind.uses_moe ||
        lw->norm.attn_sub_norm ||
        lw->norm.layer_output_scale) {
        if (bn_transformer_prefill_moe_chain_debug_enabled(gpu))
            fprintf(stderr,
                    "[bn:prefill:moe-chain] reject layer=%d basic gpu=%d hook=%d backend=%d tq=%d toks=%d min=%d theta=%d attn=%d moe=%d shared=%d bias=%d subnorm=%d scale=%d\n",
                    layer, gpu != NULL,
                    moe_layer_backend_available,
                    backend != NULL, bn_model_tq_state(m) != NULL,
                    n_tokens,
                    bn_transformer_prefill_moe_chain_min_tokens(c, gpu),
                    layer_rope_theta == bn_transformer_rope_base_theta(c),
                    plan->is_attn,
                    layer_kind.uses_moe,
                    shared_policy.has_shared_expert,
                    lw->attn.q_bias || lw->attn.k_bias || lw->attn.v_bias,
                    lw->norm.attn_sub_norm != NULL,
                    lw->norm.layer_output_scale != NULL);
        return 0;
    }
    if (!lw->attn.wq.data || !lw->attn.wk.data || !lw->attn.wv.data ||
        !lw->attn.wo.data)
        return 0;
    BnTransformerPrefillAttentionProjectionTypes attn_types;
    if (!bn_transformer_prefill_resolve_attention_projection_types(
            &attn_types, lw))
        return 0;
    BnTransformerPrefillMoELayerGPUResourcePolicy resources =
        bn_transformer_prefill_moe_layer_gpu_resource_policy(
            backend, c, layer, lw, attn_types);
    int ready = resources.valid;
    if (!ready && bn_transformer_prefill_moe_chain_debug_enabled(gpu))
        fprintf(stderr,
                "[bn:prefill:moe-chain] reject layer=%d handles qk=%d wv=%d wo=%d router=%d gate=%d up=%d down=%d anorm=%d fnorm=%d\n",
                layer,
                resources.qk != NULL,
                resources.wv != NULL,
                resources.wo != NULL,
                resources.router != NULL,
                resources.gate_all != NULL,
                resources.up_all != NULL,
                resources.down_all != NULL,
                resources.attn_norm != NULL,
                resources.ffn_norm != NULL);
    return ready;
}

static int prefill_ssm_layer_gpu(const BnModel *m,
                                 float *out,
                                 const BnLayerWeights *lw,
                                 const float *X,
                                 int n_tokens,
                                 int dim,
                                 int qkv_dim,
                                 int inner_dim,
                                 int num_k_heads,
                                 int head_k_dim,
                                 int num_v_heads,
                                 int head_v_dim,
                                 int conv_kernel,
                                 int ssm_idx,
                                 int layer,
                                 int fuse_ffn,
                                 float norm_eps,
                                 int *did_ffn) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    BnTransformerPrefillFFNProjectionTypes ffn_types;
    if (!bn_transformer_prefill_resolve_ffn_projection_types(&ffn_types, lw))
        return -1;
    BnTransformerPrefillSSMProjectionTypes ssm_types;
    if (!bn_transformer_prefill_resolve_ssm_projection_types(&ssm_types, lw))
        return -1;
    if (!bn_transformer_prefill_ssm_gpu_layer_enabled(
            gpu, &m->config, lw)) {
        if (bn_transformer_prefill_hybrid_chain_debug_enabled(gpu))
            fprintf(stderr,
                    "[bn:prefill:hybrid-chain] ssm layer=%d disabled by policy\n",
                    layer);
        return -1;
    }
    if (!bn_transformer_prefill_ssm_layer_backend_available(gpu) ||
        !backend || !lw ||
        !lw->ssm.wqkv.data || !lw->ssm.wz.data ||
        !lw->ssm.ssm_alpha.data || !lw->ssm.ssm_beta.data ||
        !lw->ssm.ssm_out.data || !lw->norm.attn_norm ||
        !lw->ssm.ssm_conv1d || !lw->ssm.ssm_dt_bias ||
        !lw->ssm.ssm_a || !lw->ssm.ssm_norm) {
        if (bn_transformer_prefill_hybrid_chain_debug_enabled(gpu))
            fprintf(stderr,
                    "[bn:prefill:hybrid-chain] ssm layer=%d missing backend resource"
                    " hook=%d backend=%d wqkv=%d wz=%d alpha=%d beta=%d out=%d"
                    " norm=%d conv=%d dt=%d a=%d ssm_norm=%d\n",
                    layer,
                    bn_transformer_prefill_ssm_layer_backend_available(gpu),
                    backend != NULL, lw && lw->ssm.wqkv.data != NULL,
                    lw && lw->ssm.wz.data != NULL,
                    lw && lw->ssm.ssm_alpha.data != NULL,
                    lw && lw->ssm.ssm_beta.data != NULL,
                    lw && lw->ssm.ssm_out.data != NULL,
                    lw && lw->norm.attn_norm != NULL,
                    lw && lw->ssm.ssm_conv1d != NULL,
                    lw && lw->ssm.ssm_dt_bias != NULL,
                    lw && lw->ssm.ssm_a != NULL,
                    lw && lw->ssm.ssm_norm != NULL);
        return -1;
    }
    BnTransformerPrefillSSMFFNFusePolicy ffn_fuse =
        bn_transformer_prefill_ssm_ffn_fuse_policy(
            fuse_ffn,
            bn_transformer_prefill_ssm_ffn_fuse_allowed(gpu),
            lw->ffn.ffn_gate.data != NULL,
            lw->ffn.ffn_up.data != NULL,
            lw->ffn.ffn_down.data != NULL,
            bn_transformer_prefill_has_ffn_gate(&m->config),
            lw->norm.ffn_sub_norm != NULL,
            lw->norm.layer_output_scale != NULL,
            bn_transformer_ffn_uses_post_norm_layer(&m->config, lw),
            lw->norm.ffn_post_norm != NULL);
    BnTransformerPrefillSSMGPUResourcePolicy resources =
        bn_transformer_prefill_ssm_gpu_resource_policy(
            backend, layer, lw, ffn_fuse.enabled, ffn_types);
    if (!resources.valid) {
        if (bn_transformer_prefill_hybrid_chain_debug_enabled(gpu))
            fprintf(stderr,
                    "[bn:prefill:hybrid-chain] ssm layer=%d resources invalid\n",
                    layer);
        return -1;
    }
    if (did_ffn)
        *did_ffn = 0;
    int activation = bn_transformer_prefill_config_activation(&m->config);
    int rc = bn_transformer_gpu_prefill_ssm_layer_backend_run(
        gpu, out, (void *)resources.wqkv, (void *)resources.wz,
        (void *)resources.alpha, (void *)resources.beta,
        (void *)resources.qkvz_stacked, (void *)resources.ab_stacked,
        (void *)resources.out, (void *)resources.attn_norm,
        (void *)resources.conv1d, (void *)resources.dt_bias,
        (void *)resources.a_log, (void *)resources.ssm_norm,
        (void *)resources.gate, (void *)resources.up,
        (void *)resources.down, (void *)resources.ffn_norm,
        X, n_tokens, dim, qkv_dim, inner_dim,
        num_k_heads, head_k_dim, num_v_heads, head_v_dim,
        conv_kernel, ssm_idx, ssm_types.qkv_type, ssm_types.z_type,
        ssm_types.alpha_type, ssm_types.beta_type, ssm_types.out_type,
        ffn_types.down_cols, ffn_types.gate_type, ffn_types.up_type,
        ffn_types.down_type, activation,
        bn_transformer_ssm_uses_sigmoid_gate(&m->config), norm_eps,
        did_ffn);
    if (rc != 0 && bn_transformer_prefill_hybrid_chain_debug_enabled(gpu))
        fprintf(stderr,
                "[bn:prefill:hybrid-chain] ssm layer=%d backend rc=%d\n",
                layer, rc);
    return rc;
}

static int prefill_ssm_layer_chain_ready(const BnModel *m,
                                         const BnLayerWeights *lw,
                                         int layer,
                                         int n_tokens) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!lw)
        return 0;
    if (!bn_transformer_prefill_ssm_gpu_layer_enabled(gpu, c, lw))
        return 0;
    if (!bn_transformer_prefill_ssm_ffn_fuse_allowed(gpu))
        return 0;
    BnTransformerPrefillFFNProjectionTypes ffn_types;
    if (!bn_transformer_prefill_resolve_ffn_projection_types(&ffn_types, lw))
        return 0;
    BnTransformerPrefillLayerKindPolicy layer_kind =
        bn_transformer_prefill_layer_kind_policy(lw);
    int use_attn_post_norm =
        bn_transformer_attention_uses_post_norm_layer(c, lw);
    int use_ffn_post_norm =
        bn_transformer_ffn_uses_post_norm_layer(c, lw);
    BnTransformerSSMShapePolicy ssm_shape;
    if (!bn_transformer_ssm_shape_policy(&ssm_shape, c))
        return 0;
    BnTransformerPrefillSSMChainPolicy policy =
        bn_transformer_prefill_ssm_chain_policy(
            bn_transformer_prefill_ssm_dense_chain_available(gpu, c,
                                                             n_tokens),
            layer_kind,
            bn_transformer_prefill_has_ffn_gate(c),
            lw->ffn.ffn_up.data != NULL,
            lw->norm.ffn_sub_norm != NULL,
            lw->norm.layer_output_scale != NULL,
            use_attn_post_norm || use_ffn_post_norm,
            lw->norm.attn_post_norm != NULL,
            lw->norm.ffn_post_norm != NULL,
            &ssm_shape);
    if (!policy.enabled ||
        !backend ||
        !lw->ssm.wqkv.data || !lw->ssm.wz.data ||
        !lw->ssm.ssm_alpha.data || !lw->ssm.ssm_beta.data ||
        !lw->ssm.ssm_out.data || !lw->norm.attn_norm ||
        !lw->ssm.ssm_conv1d || !lw->ssm.ssm_dt_bias ||
        !lw->ssm.ssm_a || !lw->ssm.ssm_norm ||
        !lw->ffn.ffn_gate.data || !lw->ffn.ffn_down.data)
        return 0;

    BnTransformerPrefillSSMGPUResourcePolicy resources =
        bn_transformer_prefill_ssm_gpu_resource_policy(
            backend, layer, lw, 1, ffn_types);
    return resources.valid && resources.fuses_ffn;
}

static void prefill_quant_matmul_prepared_kquant_multi(const BnModel *m,
                                                       float **out,
                                                       const BnQWeight **W,
                                                       int n,
                                                       int n_tokens,
                                                       const int8_t *quantized,
                                                       const float *scales,
                                                       const int16_t *block_sums,
                                                       const float *x_float) {
    const BnBackendModel *backend = bn_model_backend(m);
    if (n > 4) {
        bn_transformer_prefill_quant_matmul_prepared_kquant_input_multi(
            out, W, NULL, n, n_tokens, quantized, scales, block_sums, x_float,
            bn_model_pool(m));
        prefill_replay_cpu_projection(
            m, out, W, n, x_float, n_tokens, (int8_t *)quantized);
        return;
    }
    BnTransformerPrefillQuantMatmulResourcePolicy resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            prefill_cpu_runtime(m), backend, W, n, 4);
    bn_transformer_prefill_quant_matmul_prepared_kquant_input_multi(
        out, W, resources.valid ? resources.prepared : NULL, n, n_tokens,
        quantized, scales, block_sums, x_float, bn_model_pool(m));
    prefill_replay_cpu_projection(
        m, out, W, n, x_float, n_tokens, (int8_t *)quantized);
}

typedef struct {
    int8_t *quantized;
    float *scales;
    int16_t *block_sums;
    int blocks_per_row;
    int capacity_blocks_per_row;
} BnPrefillPreparedKQuantBuffers;

typedef enum {
    BN_PREFILL_KQUANT_CANONICAL_WEIGHTS = 0,
    BN_PREFILL_KQUANT_REPACKED_WEIGHTS = 1,
} BnPrefillKQuantWeightLayout;

static const BnPrefillCPUOps *prefill_cpu_ops(void) {
    return bn_transformer_prefill_cpu_ops();
}

static size_t prefill_prepared_kquant_arena_bytes(int dim, int n_tokens) {
    if (!prefill_cpu_ops()->supports_prepared_kquant)
        return 0;
    int blocks_per_row =
        bn_transformer_prefill_prepared_kquant_blocks_per_row(dim);
    if (n_tokens <= 0 || blocks_per_row <= 0)
        return 0;
    int block_sums_per_row =
        bn_transformer_prefill_prepared_kquant_block_sums_per_row(
            blocks_per_row);
    return (size_t)n_tokens * (size_t)dim +
           (size_t)n_tokens * (size_t)blocks_per_row * sizeof(float) +
           (size_t)n_tokens * (size_t)block_sums_per_row * sizeof(int16_t);
}

static BnPrefillPreparedKQuantBuffers
prefill_alloc_prepared_kquant_buffers(SHArena *arena,
                                      int dim,
                                      int n_tokens) {
    BnPrefillPreparedKQuantBuffers b = { NULL, NULL, NULL, 0, 0 };
    if (!prefill_cpu_ops()->supports_prepared_kquant)
        return b;
    if (!arena || n_tokens <= 0)
        return b;
    b.blocks_per_row =
        bn_transformer_prefill_prepared_kquant_blocks_per_row(dim);
    b.capacity_blocks_per_row = b.blocks_per_row;
    if (b.blocks_per_row <= 0)
        return b;
    int block_sums_per_row =
        bn_transformer_prefill_prepared_kquant_block_sums_per_row(
            b.blocks_per_row);
    b.quantized = (int8_t *)sh_arena_alloc(arena,
                                           (size_t)n_tokens * (size_t)dim);
    b.scales = (float *)sh_arena_alloc(
        arena, (size_t)n_tokens * (size_t)b.blocks_per_row * sizeof(float));
    b.block_sums = (int16_t *)sh_arena_alloc(
        arena, (size_t)n_tokens * (size_t)block_sums_per_row * sizeof(int16_t));
    if (!b.quantized || !b.scales || !b.block_sums) {
        b.quantized = NULL;
        b.scales = NULL;
        b.block_sums = NULL;
        b.blocks_per_row = 0;
        b.capacity_blocks_per_row = 0;
    }
    return b;
}

static int prefill_prepare_prepared_kquant(BnPrefillPreparedKQuantBuffers *b,
                                           const float *x,
                                           int dim,
                                           int n_tokens) {
    const BnPrefillCPUOps *ops = prefill_cpu_ops();
    int blocks_per_row =
        bn_transformer_prefill_prepared_kquant_blocks_per_row(dim);
    if (!b || !b->quantized || !b->scales || !b->block_sums || !x || dim <= 0 ||
        n_tokens <= 0 || blocks_per_row <= 0 ||
        blocks_per_row > b->capacity_blocks_per_row ||
        !ops->prepare_prepared_kquant)
        return 0;
    b->blocks_per_row = blocks_per_row;
    return ops->prepare_prepared_kquant(
        b->quantized, b->scales, b->block_sums, blocks_per_row,
        x, dim, n_tokens);
}

static int prefill_try_prepared_kquant_multi(const BnModel *m,
                                             BnPrefillPreparedKQuantBuffers *b,
                                             float **out,
                                             const BnQWeight **W,
                                             int n,
                                             const float *X,
                                             int dim,
                                             int n_tokens,
                                             BnPrefillKQuantWeightLayout layout) {
    const BnPrefillCPUOps *ops = prefill_cpu_ops();
    if (!m || !b || !b->quantized || !out || !W || !X || n <= 0 || n > 4)
        return 0;
    int tensor_types[4];
    for (int i = 0; i < n; i++) {
        if (!W[i])
            return 0;
        tensor_types[i] = W[i]->type;
    }
    if (!bn_transformer_prefill_prepared_kquant_dispatch_policy(
             ops, bn_model_gpu(m), prefill_uses_float_kquant_fallback(m), dim,
             tensor_types, n, 4)
             .enabled)
        return 0;
    if (!prefill_prepare_prepared_kquant(b, X, dim, n_tokens))
        return 0;
    if (layout == BN_PREFILL_KQUANT_REPACKED_WEIGHTS) {
        prefill_quant_matmul_prepared_kquant_multi(
            m, out, W, n, n_tokens, b->quantized, b->scales,
            b->block_sums, X);
    } else {
        bn_transformer_prefill_quant_matmul_prepared_kquant_input_multi(
            out, W, NULL, n, n_tokens, b->quantized, b->scales,
            b->block_sums, X, bn_model_pool(m));
        prefill_replay_cpu_projection(
            m, out, W, n, X, n_tokens, b->quantized);
    }
    return 1;
}

static float *prefill_logits(BnModel *m, BnSession *sess) {
    return bn_transformer_forward_logits(m, sess);
}

static BnTransformerPrefillEntryDispatchPolicy
prefill_entry_dispatch_policy(
        const BnModel *m,
        BnTransformerPrefillRequestKind request) {
    BnTransformerPrefillEntryDispatchPolicy policy = {
        BN_TRANSFORMER_PREFILL_ENTRY_BATCH, 0};
    if (!m)
        return policy;
    return bn_transformer_prefill_entry_dispatch_policy(
        &m->config, bn_model_gpu(m) != NULL, request);
}

static void prefill_rmsnorm_unit(float *out, const float *x, int size, float eps) {
    double ss = 0.0;
    for (int i = 0; i < size; i++)
        ss += (double)(x[i] * x[i]);
    float scale = 1.0f / sqrtf((float)(ss / (double)size) + eps);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * scale;
}

static void prefill_rmsnorm_unit_heads(float *x, int n_heads,
                                       int head_size, float eps) {
    for (int h = 0; h < n_heads; h++)
        prefill_rmsnorm_unit(x + h * head_size, x + h * head_size,
                             head_size, eps);
}

static void prefill_fill_rope(const BnModel *m, int layer,
                              float *rope_cos_buf, float *rope_sin_buf,
                              int rope_stride, int n_tokens, int pos0,
                              int head_size, int rope_dims, float theta) {
    int half_rope = rope_dims / 2;
    float angles[half_rope];
    int adjust_freqs =
        bn_transformer_uses_per_layer_embedding(&m->config) &&
        bn_transformer_rope_uses_base_frequency(&m->config,
                                                 head_size) &&
        m->weights.rope_freqs;
    for (int t = 0; t < n_tokens; t++) {
        int pos = pos0 + t;
        bn_model_transformer_policy_init_rope_angles_for_theta(
            theta, rope_dims, pos, angles, half_rope);
        for (int i = 0; i < half_rope; i++) {
            float angle = angles[i];
            if (adjust_freqs) {
                if (bn_transformer_divides_rope_freqs(&m->config, layer))
                    angle /= m->weights.rope_freqs[i];
                else
                    angle *= m->weights.rope_freqs[i];
            }
            rope_cos_buf[(size_t)t * rope_stride + i] = cosf(angle);
            rope_sin_buf[(size_t)t * rope_stride + i] = sinf(angle);
        }
    }
}

static int prefill_prepare_q_for_gpu_attention(BnBatchedAttnCtx *b) {
    if (!b || b->q_gated || !b->Q_buf || !b->rope_cos || !b->rope_sin)
        return -1;
    if (b->pos0 != 0)
        return -1;
    int head_size = b->head_size;
    int n_heads = b->n_heads;
    int n_tokens = b->n_tokens;
    int q_row_stride = b->q_row_stride > 0 ? b->q_row_stride : b->wq_rows;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : b->rope_dims / 2;
    if (head_size <= 0 || n_heads <= 0 || n_tokens <= 1 ||
        q_row_stride < n_heads * head_size)
        return -1;

    for (int t = 0; t < n_tokens; t++) {
        float *row = b->Q_buf + (size_t)t * q_row_stride;
        if (b->q_bias) {
            for (int i = 0; i < n_heads * head_size; i++)
                row[i] += b->q_bias[i];
        }
        if (b->q_norm) {
            int stride = b->qk_norm_per_head ? head_size : 0;
            for (int h = 0; h < n_heads; h++)
                prefill_cpu_ops()->rmsnorm(row + h * head_size,
                                           row + h * head_size,
                                           b->q_norm + h * stride, head_size,
                                           b->norm_eps);
        }
        bn_transformer_cpu_apply_rope_heads(
            b->runtime, row, n_heads, head_size, b->rope_dims,
            b->rope_cos + (size_t)t * rope_stride,
            b->rope_sin + (size_t)t * rope_stride);
        if (q_row_stride != n_heads * head_size)
            memmove(b->Q_buf + (size_t)t * n_heads * head_size, row,
                    (size_t)n_heads * (size_t)head_size * sizeof(float));
    }
    return 0;
}

/* Every layer and block is checked before the first block changes KV/SSM
 * state. Capability declarations alone do not prove resident resources exist. */
static int prefill_prefix_chain_ready(const BnModel *m, int n_tokens, int pos0) {
    const BnConfig *c = &m->config;
    BnGPUBackend *gpu = bn_model_gpu(m);
    if (!bn_transformer_prefill_prefix_request_allowed(c, gpu, n_tokens, pos0) ||
        !bn_gpu_backend_can_check_prefill_prefix(gpu) || !gpu->write_activation ||
        bn_model_tq_state(m) ||
        !bn_transformer_prefill_hybrid_chain_applicable(gpu, c) ||
        !bn_transformer_prefill_hybrid_chain_enabled(gpu, c) ||
        bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(gpu, c) ||
        bn_gpu_policy_cpu_decode_fallback_requested(gpu))
        return 0;
    /* A singleton tail uses native decode. Two rows ending at the same
     * position preflight its cache extent and the shared projection resources. */
    const int check_tokens = n_tokens == 1 ? 2 : n_tokens;
    const int check_pos = n_tokens == 1 ? pos0 - 1 : pos0;
    int attention = 0, recurrent = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &m->weights.layers[l];
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, lw, l, 0);
        if (plan.is_attn) {
            if (plan.head_size != 256 || plan.kv_mul != 16 ||
                !prefill_moe_layer_chain_ready(m, lw, &plan, l, check_tokens,
                    bn_transformer_rope_theta_for_head(c, plan.head_size)))
                return 0;
            if (check_pos > 0) {
                BnGPUAttentionPrefillPlan span = {0};
                span.n_tokens=check_tokens; span.pos0=check_pos;
                span.n_heads=plan.n_heads; span.n_kv_heads=plan.n_kv_heads;
                span.head_size=plan.head_size;
                span.rope_dims=bn_transformer_rope_dims_for_head(c,plan.head_size);
                span.kv_cache_stride=c->kv_dim;
                span.kv_cache_off=((size_t)plan.attn_idx*c->seq_len+check_pos)*c->kv_dim;
                if (span.kv_cache_off > UINT32_MAX ||
                    !bn_gpu_backend_prefill_prefix_supported(gpu,&span))
                    return 0;
            }
            attention++;
        } else {
            if (!prefill_ssm_moe_layer_chain_ready(m,lw,l,check_tokens) &&
                !prefill_ssm_layer_chain_ready(m,lw,l,check_tokens))
                return 0;
            recurrent++;
        }
    }
    return attention > 0 && recurrent > 0;
}

static int prefill_microbatch_ready(const BnModel *m, int n_tokens,
                                    int pos0, int batch) {
    /* Only this call's completed blocks establish prefix residency. Public
     * resumed requests may have host-only KV and retain their existing path. */
    if (pos0 != 0 || batch <= 1 || n_tokens <= batch ||
        !bn_transformer_prefill_prefix_request_allowed(
            &m->config,bn_model_gpu(m),n_tokens,pos0))
        return 0;
    for (int done=0; done<n_tokens; ) {
        int count=n_tokens-done;
        if (count>batch) count=batch;
        if (!prefill_prefix_chain_ready(m,count,pos0+done)) return 0;
        done+=count;
    }
    return 1;
}

static float *prefill_internal(BnModel *m, BnSession *sess, const int *tokens,
                               int n_tokens, int pos0, float *all_logits,
                               int need_last_logits, int prefix_resident) {
    if (n_tokens <= 0) return NULL;
    const int batch = bn_transformer_prefill_microbatch_tokens(bn_model_gpu(m));
    if (prefill_microbatch_ready(m,n_tokens,pos0,batch)) {
        /* Validate the whole request before any block advances KV/SSM state. */
        for (int t = 0; t < n_tokens; t++) {
            if (tokens[t] < 0 || tokens[t] >= m->config.vocab_size) {
                SH_LOG_ERROR("Token out of range");
                return NULL;
            }
        }
        if (bn_transformer_prefill_profile_enabled(prefill_cpu_runtime(m)))
            fprintf(stderr,"[bn:prefill:microbatch] tokens=%d batch=%d pos=%d\n",
                    n_tokens,batch,pos0);
        for (int done=0; done<n_tokens; ) {
            int count=n_tokens-done;
            if (count>batch) count=batch;
            float *result=prefill_internal(m,sess,tokens+done,count,pos0+done,
                all_logits ? all_logits+(size_t)done*m->config.vocab_size : NULL,
                need_last_logits && done+count==n_tokens, done > 0);
            if (!result) return NULL;
            if (count==1) sess->gpu_kv_direct_valid=1;
            /* Host-returned FP16 rows must reach the backend before the next
             * block attends to them. Generation handles the final upload. */
            if (done+count<n_tokens && !sess->gpu_kv_direct_valid &&
                bn_transformer_gpu_upload_kv_cache(m,sess,pos0+done,count)!=0)
                return NULL;
            done+=count;
            if (done==n_tokens) return result;
        }
    }
    const int request_n_tokens = n_tokens;
    if (sess) sess->gpu_kv_direct_valid = 0;
    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, sess, tokens[0], pos0);
        if (logits && all_logits)
            memcpy(all_logits,logits,(size_t)m->config.vocab_size*sizeof(float));
        return need_last_logits ? logits : (logits ? sess->state.x : NULL);
    }

    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnWeights *w = &m->weights;
    int dim = c->dim;
    float norm_eps = bn_transformer_prefill_norm_epsilon(c);
    BnPrefillProfile prof = {0};
    prof.enabled = bn_transformer_prefill_profile_enabled(
        prefill_cpu_runtime(m));
    double t_prof = prefill_profile_now(&prof);

    int max_head_size = bn_transformer_attention_head_size(c, NULL);
    int max_n_heads = bn_transformer_attention_n_heads(c, NULL);
    int max_q_dim = max_n_heads * max_head_size;
    int max_rope_dims = bn_transformer_rope_dims_for_head(c, max_head_size);
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, &w->layers[l], l,
                                        bn_model_tq_state(m) != NULL);
        if (!plan.is_attn)
            continue;
        if (plan.head_size > max_head_size)
            max_head_size = plan.head_size;
        if (plan.q_dim > max_q_dim)
            max_q_dim = plan.q_dim;
        int layer_rope_dims = bn_transformer_rope_dims_for_head(c, plan.head_size);
        if (layer_rope_dims > max_rope_dims)
            max_rope_dims = layer_rope_dims;
    }

    if (max_head_size > BN_MAX_VLA_ELEMS || dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dimensions too large for stack VLAs");
        return NULL;
    }

    for (int t = 0; t < n_tokens; t++) {
        if (tokens[t] < 0 || tokens[t] >= c->vocab_size) {
            SH_LOG_ERROR("Token out of range");
            return NULL;
        }
    }
    if (pos0 < 0) {
        SH_LOG_ERROR("Position out of range");
        return NULL;
    }

    BnGPUBackend *prefill_gpu = bn_model_gpu(m);
    int gpu_hybrid_prefill =
        bn_transformer_prefill_hybrid_chain_applicable(prefill_gpu, c);
    int gpu_moe_prefill =
        bn_transformer_prefill_moe_chain_applicable(prefill_gpu, c);
    BnTransformerPrefillSequencePolicy sequence_policy =
        bn_transformer_prefill_sequence_policy(c);
    int small_dense_prefill_chain =
        bn_transformer_prefill_small_dense_chain_applicable(prefill_gpu, c);
    BnTransformerPrefillDecodeFallbackPolicy decode_fallback =
        bn_transformer_prefill_decode_fallback_policy(
            sequence_policy, gpu_moe_prefill,
            bn_transformer_prefill_moe_enabled(prefill_gpu), n_tokens,
            bn_transformer_prefill_moe_chain_min_tokens(c, prefill_gpu),
            small_dense_prefill_chain,
            bn_transformer_prefill_dense_chain_min_tokens(c, prefill_gpu),
            gpu_hybrid_prefill,
            bn_transformer_prefill_large_hybrid_disabled(prefill_gpu),
            bn_transformer_prefill_hybrid_batch_allowed(
                prefill_cpu_runtime(m)),
            bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(
                prefill_gpu, c));
    if (decode_fallback.require_logits_decode)
        return prefill_decode_tokens_with_logits(
            m, sess, tokens, n_tokens, pos0, all_logits, need_last_logits);
    if (decode_fallback.decode)
        return prefill_decode_tokens(m, sess, tokens, n_tokens, pos0,
                                     all_logits, need_last_logits);

    size_t act_elems = (size_t)n_tokens * dim;
    if (act_elems / n_tokens != (size_t)dim) {
        SH_LOG_ERROR("Prefill activation buffer size overflow");
        return NULL;
    }

    BnTransformerPrefillBufferShapePolicy buffer_shape;
    if (!bn_transformer_prefill_buffer_shape_policy(
            &buffer_shape, c, sequence_policy, n_tokens, dim, max_q_dim,
            max_rope_dims)) {
        SH_LOG_ERROR("Prefill buffer shape policy failed");
        return NULL;
    }

    int half_rope = buffer_shape.half_rope;
    if (half_rope > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("RoPE dimensions too large for stack VLAs");
        return NULL;
    }

    int kv_dim = buffer_shape.kv_dim;
    int hidden_dim = buffer_shape.hidden_dim;
    int q_buf_stride = buffer_shape.q_buf_stride;
    int xb2_stride = buffer_shape.xb2_stride;
    int hb_stride = buffer_shape.hb_stride;
    size_t nt = (size_t)n_tokens;

    size_t batch_floats = buffer_shape.batch_floats;
    int hyper_connections = bn_transformer_uses_hyper_connections(c);
    size_t hc_wide = hyper_connections
        ? (size_t)c->hyper_connection_count * dim : 0;
    size_t hc_elems = hc_wide * nt;
    size_t hc_norm_elems = hyper_connections &&
        bn_transformer_cpu_backend_supports_hyper_connection_batch_prefill()
        ? hc_elems : 0;
    size_t hc_scratch_elems = hyper_connections
        ? nt * (size_t)c->hyper_connection_count
        : 0;
    if (hyper_connections &&
        (c->hyper_connection_count <= 1 || hc_wide / (size_t)dim !=
         (size_t)c->hyper_connection_count || hc_elems / nt != hc_wide)) {
        SH_LOG_ERROR("Prefill hyper-connection buffer size overflow");
        return NULL;
    }
    size_t arena_size = (act_elems + hc_elems + hc_scratch_elems) * sizeof(float)
                      + batch_floats * sizeof(float)
                      + nt * half_rope * 2 * sizeof(float)
                      + 512;
    if (hc_norm_elems && (c->hyper_connection_rank <= 0 ||
        nt > SIZE_MAX / (size_t)c->hyper_connection_rank)) {
        SH_LOG_ERROR("Prefill hyper-connection rank buffer size overflow");
        return NULL;
    }
    size_t hc_rank_elems = hc_norm_elems
        ? nt * (size_t)c->hyper_connection_rank : 0;
    const size_t hc_work_elems[] = {hc_norm_elems, hc_norm_elems, hc_rank_elems};
    for (size_t i = 0; i < sizeof(hc_work_elems) / sizeof(hc_work_elems[0]); i++) {
        if (hc_work_elems[i] > (SIZE_MAX - arena_size) / sizeof(float)) {
            SH_LOG_ERROR("Prefill hyper-connection workspace size overflow");
            return NULL;
        }
        arena_size += hc_work_elems[i] * sizeof(float);
    }
    int prepared_kquant_dim = dim;
    if (max_q_dim > prepared_kquant_dim)
        prepared_kquant_dim = max_q_dim;
    if (hidden_dim > prepared_kquant_dim)
        prepared_kquant_dim = hidden_dim;
    arena_size += prefill_prepared_kquant_arena_bytes(prepared_kquant_dim,
                                                       n_tokens);

    SHArena *pf_arena = sh_arena_create(arena_size);
    if (!pf_arena) return NULL;

    float *act = (float *)sh_arena_alloc(pf_arena, act_elems * sizeof(float));
    if (!act) { sh_arena_free(pf_arena); return NULL; }

    float *hc_residual = hyper_connections
        ? (float *)sh_arena_alloc(pf_arena, hc_elems * sizeof(float)) : NULL;
    if (hyper_connections && !hc_residual) {
        sh_arena_free(pf_arena);
        return NULL;
    }
    BnPrefillHCBuffers hc_buffers = {0};
    if (hyper_connections) {
        hc_buffers.inject = (float *)sh_arena_alloc(
            pf_arena, nt * (size_t)c->hyper_connection_count * sizeof(float));
        hc_buffers.norm = hc_norm_elems
            ? (float *)sh_arena_alloc(pf_arena, hc_norm_elems * sizeof(float))
            : NULL;
        hc_buffers.gates = hc_norm_elems
            ? (float *)sh_arena_alloc(pf_arena, hc_norm_elems * sizeof(float))
            : NULL;
        hc_buffers.low = hc_rank_elems
            ? (float *)sh_arena_alloc(pf_arena, hc_rank_elems * sizeof(float))
            : NULL;
        if (!hc_buffers.inject || (hc_norm_elems &&
            (!hc_buffers.norm || !hc_buffers.gates || !hc_buffers.low))) {
            sh_arena_free(pf_arena);
            return NULL;
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        bn_model_embed_token(m, act + (size_t)t * dim, tokens[t]);
        if (s->token_history)
            s->token_history[(pos0 + t) % c->seq_len] = tokens[t];
        if (hyper_connections)
            for (int stream = 0; stream < c->hyper_connection_count; stream++)
                memcpy(hc_residual + (size_t)t * hc_wide +
                           (size_t)stream * dim,
                       act + (size_t)t * dim,
                       (size_t)dim * sizeof(float));
    }
    prefill_profile_add(&prof.embed_ms, t_prof);

    float *batch_buf = (float *)sh_arena_alloc(pf_arena, batch_floats * sizeof(float));
    if (!batch_buf) { sh_arena_free(pf_arena); return NULL; }

    BnPrefillPreparedKQuantBuffers prepared_kquant =
        prefill_alloc_prepared_kquant_buffers(pf_arena, prepared_kquant_dim,
                                               n_tokens);

    float *Xb = batch_buf;
    float *Q_buf = Xb + nt * dim;
    float *K_new = Q_buf + nt * q_buf_stride;
    float *V_new = K_new + nt * kv_dim;
    float *Xb2 = V_new + nt * kv_dim;
    float *Hb = Xb2 + nt * xb2_stride;
    float *Hb2 = Hb + nt * hb_stride;

    float *rope_cos_buf = (float *)sh_arena_alloc(pf_arena, nt * half_rope * sizeof(float));
    float *rope_sin_buf = (float *)sh_arena_alloc(pf_arena, nt * half_rope * sizeof(float));
    if (!rope_cos_buf || !rope_sin_buf) { sh_arena_free(pf_arena); return NULL; }

    int rope_cache_dims = -1;
    float rope_cache_theta = 0.0f;

    BnTransformerPrefillDenseModelChainPolicy dense_model_chain =
        bn_transformer_prefill_dense_model_chain_policy(
            bn_transformer_prefill_dense_chain_enabled(prefill_gpu),
            bn_model_gpu(m) != NULL, pos0, c->n_layers);
    if (dense_model_chain.enabled) {
        int chain_ready = 1;
        for (int l = 0; l < c->n_layers; l++) {
            BnLayerWeights *lw = &w->layers[l];
            BnLayerShapePlan plan;
            bn_transformer_plan_layer_shape(
                &plan, c, lw, l, bn_model_tq_state(m) != NULL);
            int layer_head_size = plan.head_size;
            float layer_rope_theta =
                bn_transformer_rope_theta_for_head(c, layer_head_size);
            if (!prefill_moe_layer_chain_ready(
                    m, lw, &plan, l, n_tokens, layer_rope_theta) &&
                !prefill_dense_layer_chain_ready(
                    m, lw, &plan, l, n_tokens, layer_rope_theta)) {
                chain_ready = 0;
                break;
            }
        }
        if (chain_ready) {
            BnTransformerPrefillChainKVPolicy chain_kv =
                bn_transformer_prefill_chain_kv_policy(
                    bn_transformer_prefill_direct_kv_allowed(
                        c, w, prefill_gpu, pos0, n_tokens));
            sess->gpu_kv_direct_valid = 0;
            t_prof = prefill_profile_now(&prof);
            for (int l = 0; l < c->n_layers; l++) {
                BnLayerWeights *lw = &w->layers[l];
                BnLayerShapePlan plan;
                bn_transformer_plan_layer_shape(
                    &plan, c, lw, l, bn_model_tq_state(m) != NULL);
                int layer_head_size = plan.head_size;
                int layer_kv_dim = plan.kv_dim;
                int layer_n_heads = plan.n_heads;
                int layer_n_kv_heads = plan.n_kv_heads;
                int layer_kv_mul = plan.kv_mul;
                int layer_rope_dims =
                    bn_transformer_rope_dims_for_head(c, layer_head_size);
                float layer_rope_theta =
                    bn_transformer_rope_theta_for_head(c, layer_head_size);
                if (rope_cache_dims != layer_rope_dims ||
                    rope_cache_theta != layer_rope_theta) {
                    prefill_fill_rope(m, l, rope_cos_buf, rope_sin_buf,
                                      half_rope, n_tokens, pos0,
                                      layer_head_size, layer_rope_dims,
                                      layer_rope_theta);
                    rope_cache_dims = layer_rope_dims;
                    rope_cache_theta = layer_rope_theta;
                }
                float *layer_out = (l == c->n_layers - 1) ? act : NULL;
                const float *layer_in = (l == 0) ? act : NULL;
                size_t loff = (size_t)plan.attn_idx * c->seq_len * kv_dim;
                uint32_t kv_cache_off =
                    (uint32_t)(loff + (size_t)pos0 * (size_t)kv_dim);
                BnTransformerPrefillLayerKindPolicy layer_kind =
                    bn_transformer_prefill_layer_kind_policy(lw);
                int layer_rc = layer_kind.uses_moe
                    ? prefill_moe_layer_gpu_batch(
                          m, layer_out, lw, layer_in,
                          chain_kv.write_host_kv ? K_new : NULL,
                          chain_kv.write_host_kv ? V_new : NULL,
                          n_tokens, dim, layer_n_heads, layer_n_kv_heads,
                          layer_head_size,
                          layer_kv_mul, layer_kv_dim, layer_rope_dims,
                          l, pos0, kv_cache_off, kv_dim,
                          plan.qk_norm_per_head,
                          bn_transformer_attention_scale(c, layer_head_size))
                    : prefill_dense_layer_gpu_batch(
                          m, layer_out, lw, layer_in,
                          chain_kv.write_host_kv ? K_new : NULL,
                          chain_kv.write_host_kv ? V_new : NULL, n_tokens,
                          dim, hidden_dim, layer_n_heads, layer_n_kv_heads,
                          layer_head_size, layer_kv_mul, layer_kv_dim,
                          layer_rope_dims, l, pos0, kv_cache_off, kv_dim,
                          plan.qk_norm_per_head,
                          plan.value_shares_key,
                          buffer_shape.gpu_rope_freq_stride,
                          bn_transformer_attention_scale(c, layer_head_size),
                          !all_logits && l + 1 == c->n_layers);
                if (layer_rc != 0) {
                    sh_arena_free(pf_arena);
                    return NULL;
                }
                if (chain_kv.write_host_kv) {
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, k_t, layer_kv_dim,
                            "bitnet_attn_k", l, pos);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, v_t, layer_kv_dim,
                            "bitnet_attn_v", l, pos);
                        if (bn_transformer_write_host_kv_cache_row(
                                s, plan.kv_mode, loff, cache_pos, kv_dim,
                                k_t, v_t, layer_kv_dim) != 0) {
                            sh_arena_free(pf_arena);
                            return NULL;
                        }
                    }
                }
            }
            if (chain_kv.mark_direct_valid)
                sess->gpu_kv_direct_valid = 1;
            prefill_profile_add(&prof.qkv_ms, t_prof);
            goto prefill_layers_done;
        }
    }

    BnTransformerPrefillHybridModelChainPolicy hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(
            bn_transformer_prefill_hybrid_chain_enabled(prefill_gpu, c),
            gpu_hybrid_prefill, pos0, c->n_layers,
            bn_model_tq_state(m) != NULL,
            prefix_resident && pos0 > 0 &&
                prefill_prefix_chain_ready(m,n_tokens,pos0));
    if (hybrid_model_chain.enabled) {
        int chain_ready = 1;
        for (int l = 0; l < c->n_layers; l++) {
            BnLayerWeights *lw = &w->layers[l];
            BnLayerShapePlan plan;
            bn_transformer_plan_layer_shape(
                &plan, c, lw, l, bn_model_tq_state(m) != NULL);
            if (plan.is_attn) {
                int layer_head_size = plan.head_size;
                float layer_rope_theta =
                    bn_transformer_rope_theta_for_head(c, layer_head_size);
                if (!prefill_moe_layer_chain_ready(
                        m, lw, &plan, l, n_tokens, layer_rope_theta) &&
                    !prefill_dense_layer_chain_ready(
                        m, lw, &plan, l, n_tokens, layer_rope_theta)) {
                    if (bn_transformer_prefill_hybrid_chain_debug_enabled(
                            prefill_gpu))
                        fprintf(stderr,
                                "[bn:prefill:hybrid-chain] reject attn layer=%d\n",
                                l);
                    chain_ready = 0;
                    break;
                }
            } else if (!prefill_ssm_moe_layer_chain_ready(m, lw, l, n_tokens) &&
                       !prefill_ssm_layer_chain_ready(m, lw, l, n_tokens)) {
                if (bn_transformer_prefill_hybrid_chain_debug_enabled(
                        prefill_gpu))
                    fprintf(stderr,
                            "[bn:prefill:hybrid-chain] reject ssm layer=%d\n",
                            l);
                chain_ready = 0;
                break;
            }
        }
        if (chain_ready) {
            BnTransformerPrefillChainKVPolicy chain_kv =
                bn_transformer_prefill_chain_kv_policy(
                    bn_transformer_prefill_direct_kv_allowed(
                        c, w, prefill_gpu, pos0, n_tokens));
            sess->gpu_kv_direct_valid = 0;
            t_prof = prefill_profile_now(&prof);
            for (int l = 0; l < c->n_layers; l++) {
                BnLayerWeights *lw = &w->layers[l];
                BnLayerShapePlan plan;
                bn_transformer_plan_layer_shape(
                    &plan, c, lw, l, bn_model_tq_state(m) != NULL);
                float *layer_out = (l == c->n_layers - 1) ? act : NULL;
                const float *layer_in = (l == 0) ? act : NULL;
                if (plan.is_attn) {
                    int layer_head_size = plan.head_size;
                    int layer_kv_dim = plan.kv_dim;
                    int layer_n_heads = plan.n_heads;
                    int layer_n_kv_heads = plan.n_kv_heads;
                    int layer_kv_mul = plan.kv_mul;
                    int layer_rope_dims =
                        bn_transformer_rope_dims_for_head(c, layer_head_size);
                    size_t loff =
                        (size_t)plan.attn_idx * c->seq_len * kv_dim;
                    uint32_t kv_cache_off =
                        (uint32_t)(loff + (size_t)pos0 * (size_t)kv_dim);
                    BnTransformerPrefillLayerKindPolicy layer_kind =
                        bn_transformer_prefill_layer_kind_policy(lw);
                    int layer_rc = layer_kind.uses_moe
                        ? prefill_moe_layer_gpu_batch(
                              m, layer_out, lw, layer_in,
                              chain_kv.write_host_kv ? K_new : NULL,
                              chain_kv.write_host_kv ? V_new : NULL,
                              n_tokens, dim, layer_n_heads, layer_n_kv_heads,
                              layer_head_size,
                              layer_kv_mul, layer_kv_dim, layer_rope_dims, l,
                              pos0, kv_cache_off, kv_dim,
                              plan.qk_norm_per_head,
                              bn_transformer_attention_scale(c, layer_head_size))
                        : prefill_dense_layer_gpu_batch(
                              m, layer_out, lw, layer_in,
                              chain_kv.write_host_kv ? K_new : NULL,
                              chain_kv.write_host_kv ? V_new : NULL,
                              n_tokens,
                              dim, hidden_dim, layer_n_heads, layer_n_kv_heads,
                              layer_head_size, layer_kv_mul, layer_kv_dim,
                              layer_rope_dims, l, pos0, kv_cache_off, kv_dim,
                              plan.qk_norm_per_head,
                              plan.value_shares_key,
                              buffer_shape.gpu_rope_freq_stride,
                              bn_transformer_attention_scale(c, layer_head_size),
                              !all_logits && l + 1 == c->n_layers);
                    if (layer_rc != 0) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                    if (chain_kv.write_host_kv) {
                        for (int t = 0; t < n_tokens; t++) {
                            int pos = pos0 + t;
                            int cache_pos = pos % c->seq_len;
                            float *k_t = K_new + (size_t)t * layer_kv_dim;
                            float *v_t = V_new + (size_t)t * layer_kv_dim;
                            if (bn_transformer_write_host_kv_cache_row(
                                    s, plan.kv_mode, loff, cache_pos, kv_dim,
                                    k_t, v_t, layer_kv_dim) != 0) {
                                sh_arena_free(pf_arena);
                                return NULL;
                            }
                        }
                    }
                } else {
                    BnTransformerSSMShapePolicy ssm_shape;
                    if (!bn_transformer_ssm_shape_policy(&ssm_shape, c)) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                    BnTransformerPrefillLayerKindPolicy layer_kind =
                        bn_transformer_prefill_layer_kind_policy(lw);
                    int r_is_moe = layer_kind.uses_moe;
                    float *ssm_out =
                        (!r_is_moe && l == c->n_layers - 1) ? act : NULL;
                    int ssm_did_ffn = 0;
                    if (prefill_ssm_layer_gpu(
                            m, ssm_out, lw, layer_in, n_tokens, dim,
                            ssm_shape.qkv_dim, ssm_shape.value_dim,
                            ssm_shape.num_k_heads, ssm_shape.head_k_dim,
                            ssm_shape.num_v_heads, ssm_shape.head_v_dim,
                            ssm_shape.conv_kernel, plan.ssm_idx, l,
                            !r_is_moe, norm_eps,
                            &ssm_did_ffn) != 0 || (!r_is_moe && !ssm_did_ffn)) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                    if (r_is_moe &&
                        prefill_moe_ffn_gpu_batch(
                            m, layer_out, lw, NULL, n_tokens, dim, l) != 0) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                }
            }
            if (chain_kv.mark_direct_valid)
                sess->gpu_kv_direct_valid = 1;
            prefill_profile_add(&prof.qkv_ms, t_prof);
            goto prefill_layers_done;
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, lw, l, bn_model_tq_state(m) != NULL);
        BnAttentionPlan attn_plan;
        bn_transformer_plan_attention(&attn_plan, c, lw, bn_model_gpu(m),
                                      bn_model_backend(m), l,
                                      bn_model_tq_state(m) != NULL,
                                      bn_model_gpu(m) != NULL);
        BnFFNPlan ffn_plan;
        bn_transformer_plan_ffn(&ffn_plan, c, lw, bn_model_gpu(m),
                                bn_model_backend(m), l,
                                bn_model_gpu(m) != NULL);
        int is_attn = plan.is_attn;

        if (hyper_connections) {
            if (l == c->ple_layer && c->ple_head_count > 0) {
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, hc_residual + (size_t)t * hc_wide,
                        (int)hc_wide, "bitnet_hc_before_ple", l, pos0 + t);
                if (prefill_apply_ple_batch(
                        m, sess, lw, hc_residual, n_tokens, dim,
                        hc_wide, pos0) != 0) {
                    sh_arena_free(pf_arena);
                    return NULL;
                }
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, hc_residual + (size_t)t * hc_wide,
                        (int)hc_wide, "bitnet_hc_after_ple", l, pos0 + t);
            }
            if (prefill_hc_mix_batch(m, sess, &lw->hc_attn, act,
                                     hc_residual, hc_buffers,
                                     n_tokens, 1, "bitnet_hc_attn_norm",
                                     l, pos0) != 0) {
                sh_arena_free(pf_arena);
                return NULL;
            }
        }

        if (is_attn && lw->attn.wq.data) {
            int layer_head_size = plan.head_size;
            int layer_kv_dim = plan.kv_dim;
            int layer_n_heads = plan.n_heads;
            int layer_n_kv_heads = plan.n_kv_heads;
            int layer_kv_mul = plan.kv_mul;
            int layer_q_dim = plan.q_dim;
            int layer_rope_dims = bn_transformer_rope_dims_for_head(c, layer_head_size);
            float layer_rope_theta = bn_transformer_rope_theta_for_head(c, layer_head_size);
            if (rope_cache_dims != layer_rope_dims ||
                rope_cache_theta != layer_rope_theta) {
                prefill_fill_rope(m, l, rope_cos_buf, rope_sin_buf,
                                  half_rope, n_tokens, pos0,
                                  layer_head_size, layer_rope_dims,
                                  layer_rope_theta);
                rope_cache_dims = layer_rope_dims;
                rope_cache_theta = layer_rope_theta;
            }
            BnTransformerPrefillLayerKindPolicy layer_kind =
                bn_transformer_prefill_layer_kind_policy(lw);
            BnTransformerPrefillDenseLayerBatchPolicy dense_layer_batch =
                bn_transformer_prefill_dense_layer_batch_policy(
                    bn_model_gpu(m) != NULL,
                    bn_model_tq_state(m) != NULL,
                    bn_transformer_prefill_dense_chain_enabled(prefill_gpu),
                    n_tokens,
                    bn_transformer_prefill_dense_chain_min_tokens(
                        c, bn_model_gpu(m)),
                    pos0, layer_rope_theta, bn_transformer_rope_base_theta(c),
                    layer_kind,
                    bn_transformer_prefill_has_ffn_gate(c),
                    lw->ffn.ffn_up.data != NULL,
                    lw->attn.q_bias != NULL, lw->attn.k_bias != NULL,
                    lw->attn.v_bias != NULL,
                    lw->norm.attn_sub_norm != NULL,
                    lw->norm.ffn_sub_norm != NULL,
                    lw->norm.layer_output_scale != NULL,
                    attn_plan.use_post_norm || ffn_plan.use_post_norm,
                    lw->norm.attn_post_norm != NULL,
                    lw->norm.ffn_post_norm != NULL);
            if (dense_layer_batch.enabled) {
                t_prof = prefill_profile_now(&prof);
                if (prefill_dense_layer_gpu_batch(
                        m, act, lw, act, K_new, V_new, n_tokens, dim,
                        hidden_dim, layer_n_heads, layer_n_kv_heads,
                        layer_head_size, layer_kv_mul, layer_kv_dim,
                        layer_rope_dims, l, pos0,
                        (uint32_t)((size_t)plan.attn_idx * c->seq_len *
                                   kv_dim + (size_t)pos0 * kv_dim),
                        kv_dim,
                        plan.qk_norm_per_head,
                        plan.value_shares_key,
                        buffer_shape.gpu_rope_freq_stride,
                        bn_transformer_attention_scale(c, layer_head_size),
                        !all_logits && l + 1 == c->n_layers) == 0) {
                    prefill_profile_add(&prof.qkv_ms, t_prof);
                    int attn_idx = plan.attn_idx;
                    size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, k_t, layer_kv_dim, "bitnet_attn_k", l, pos);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, v_t, layer_kv_dim, "bitnet_attn_v", l, pos);
                        if (bn_transformer_write_host_kv_cache_row(
                                s, plan.kv_mode, loff, cache_pos, kv_dim,
                                k_t, v_t, layer_kv_dim) != 0) {
                            sh_arena_free(pf_arena);
                            return NULL;
                        }
                    }
                    continue;
                }
            }
            BnTransformerPrefillAttentionProjectionTypes attn_types;
            if (!bn_transformer_prefill_resolve_attention_projection_types(
                    &attn_types, lw)) {
                sh_arena_free(pf_arena);
                return NULL;
            }
            int q_read_stride = attn_types.q_rows;
            int attn_idx = plan.attn_idx;
            size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
            int kv_read_idx =
                bn_transformer_attention_kv_read_index(c, lw, l);
            size_t read_loff =
                (size_t)kv_read_idx * c->seq_len * kv_dim;
            int has_kv = lw->attn.has_kv;
            int q_gated = plan.q_gated;
            const int reference_attention =
                bn_transformer_prefill_host_reference_enabled(
                    bn_model_gpu(m), c);
            int wo_cols_attn = attn_types.out_cols;
            int used_fused_attn_wo = 0;
            int used_raw_prefill_attn_wo = 0;
            int used_attn_residual = 0;
            BnGPUBackend *gpu = bn_model_gpu(m);
            const BnBackendModel *backend = bn_model_backend(m);
            BnTransformerPrefillRawAttentionGPUResourcePolicy
                raw_attn_resources =
                    bn_transformer_prefill_raw_attention_gpu_resource_policy(
                        backend, l, lw, attn_types);
            BnTransformerPrefillRawAttentionPolicy raw_attn_policy =
                bn_transformer_prefill_raw_attention_policy(
                    gpu != NULL,
                    bn_transformer_prefill_raw_attention_gpu_available(gpu),
                    bn_transformer_prefill_raw_attention_norm_resid_gpu_available(
                        gpu),
                    raw_attn_resources.attn_norm != NULL,
                    bn_model_tq_state(m) != NULL,
                    q_gated,
                    pos0,
                    n_tokens,
                    bn_transformer_prefill_attention_min_tokens(c, gpu),
                    layer_rope_theta,
                    bn_transformer_rope_base_theta(c),
                    lw->attn.q_bias != NULL,
                    lw->attn.k_bias != NULL,
                    lw->attn.v_bias != NULL,
                    lw->norm.attn_sub_norm != NULL,
                    attn_plan.use_post_norm,
                    lw->norm.attn_post_norm != NULL);
            int attn_norm_ready = 0;
            if (!raw_attn_policy.fuses_input_norm) {
                t_prof = prefill_profile_now(&prof);
                prefill_attention_input_norm(
                    m, Xb, act, lw->norm.attn_norm,
                    (void *)raw_attn_resources.attn_norm, n_tokens, dim,
                    norm_eps, hyper_connections);
                prefill_profile_add(&prof.attn_norm_ms, t_prof);
                attn_norm_ready = 1;
            }
            if (raw_attn_policy.eligible) {
                t_prof = prefill_profile_now(&prof);
                int qkv_fused_rc = -1;
                BnTransformerPrefillRawAttentionCallPolicy raw_attn_call =
                    bn_transformer_prefill_raw_attention_call_policy(
                        raw_attn_policy);
                if (raw_attn_resources.valid &&
                    raw_attn_call.preferred_kind ==
                        BN_TRANSFORMER_PREFILL_RAW_ATTENTION_NORM_RESID) {
                    qkv_fused_rc =
                        bn_transformer_gpu_prefill_qkv_attention_wo_norm_resid_backend_run(
                        gpu, act, (void *)raw_attn_resources.qk,
                        (void *)raw_attn_resources.wv,
                        (void *)raw_attn_resources.wo,
                        (void *)raw_attn_resources.attn_norm,
                        (void *)raw_attn_resources.q_norm,
                        (void *)raw_attn_resources.k_norm, act, K_new,
                        V_new, n_tokens, dim, layer_n_heads,
                        layer_n_kv_heads, layer_head_size, layer_kv_mul,
                        kv_dim, raw_attn_resources.qk_rows,
                        raw_attn_resources.qk_type,
                        raw_attn_resources.wv_rows,
                        raw_attn_resources.wv_type, attn_types.out_rows,
                        attn_types.out_cols, attn_types.out_type,
                        plan.qk_norm_per_head,
                        norm_eps, pos0,
                        layer_rope_dims,
                        bn_transformer_attention_scale(c, layer_head_size), bn_transformer_attention_window(c, l));
                    if (qkv_fused_rc == 0)
                        used_attn_residual = 1;
                    if (qkv_fused_rc != 0) {
                        t_prof = prefill_profile_now(&prof);
                        prefill_attention_input_norm(
                            m, Xb, act, lw->norm.attn_norm,
                            (void *)raw_attn_resources.attn_norm, n_tokens, dim,
                            norm_eps, hyper_connections);
                        prefill_profile_add(&prof.attn_norm_ms, t_prof);
                        attn_norm_ready = 1;
                        t_prof = prefill_profile_now(&prof);
                    }
                }
                if (qkv_fused_rc != 0 && raw_attn_resources.valid) {
                    qkv_fused_rc =
                        bn_transformer_gpu_prefill_qkv_attention_wo_backend_run(
                        gpu, Xb2, (void *)raw_attn_resources.qk,
                        (void *)raw_attn_resources.wv,
                        (void *)raw_attn_resources.wo,
                        (void *)raw_attn_resources.q_norm,
                        (void *)raw_attn_resources.k_norm, Xb, K_new, V_new,
                        n_tokens, dim, layer_n_heads, layer_n_kv_heads,
                        layer_head_size, layer_kv_mul, layer_kv_dim,
                        raw_attn_resources.qk_rows,
                        raw_attn_resources.qk_type,
                        raw_attn_resources.wv_rows,
                        raw_attn_resources.wv_type, attn_types.out_rows,
                        attn_types.out_cols, attn_types.out_type,
                        plan.qk_norm_per_head,
                        norm_eps, pos0,
                        layer_rope_dims,
                        bn_transformer_attention_scale(c, layer_head_size), bn_transformer_attention_window(c, l));
                }
                if (qkv_fused_rc == 0) {
                    for (int t = 0; t < n_tokens; t++) {
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, K_new + (size_t)t * layer_kv_dim,
                            layer_kv_dim, "bitnet_attn_k", l, pos0 + t);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, V_new + (size_t)t * layer_kv_dim,
                            layer_kv_dim, "bitnet_attn_v", l, pos0 + t);
                    }
                    used_raw_prefill_attn_wo = 1;
                    used_fused_attn_wo = 1;
                    prefill_profile_add(&prof.qkv_ms, t_prof);
                }
            }
            if (!used_raw_prefill_attn_wo) {
                if (!attn_norm_ready) {
                    t_prof = prefill_profile_now(&prof);
                    prefill_attention_input_norm(
                        m, Xb, act, lw->norm.attn_norm,
                        (void *)raw_attn_resources.attn_norm, n_tokens, dim,
                        norm_eps, hyper_connections);
                    prefill_profile_add(&prof.attn_norm_ms, t_prof);
                    attn_norm_ready = 1;
                }
                for (int t = 0; t < n_tokens; t++) {
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Xb + (size_t)t * dim, dim,
                        "bitnet_attn_norm", l, pos0 + t);
                }
            {
                t_prof = prefill_profile_now(&prof);
                float *qkv_out[3] = { Q_buf, K_new, V_new };
                const BnQWeight *qkv_w[3] = {
                    &lw->attn.wq, &lw->attn.wk, &lw->attn.wv
                };
                int used_prepared_kquant = has_kv && !gpu &&
                    !reference_attention &&
                    prefill_try_prepared_kquant_multi(
                        m, &prepared_kquant, qkv_out, qkv_w, 3, Xb, dim,
                        n_tokens, BN_PREFILL_KQUANT_REPACKED_WEIGHTS);
                if (!has_kv) {
                    float *q_out[1] = { Q_buf };
                    const BnQWeight *q_weight[1] = { &lw->attn.wq };
                    prefill_quant_matmul_multi(m, q_out, q_weight, 1, Xb,
                                               n_tokens, s->x_q);
                } else if (reference_attention) {
                    for (int i = 0; i < 3; i++)
                        prefill_quant_matmul_host(
                            m, qkv_out[i], qkv_w[i], Xb, n_tokens, s->x_q);
                } else if (used_prepared_kquant) {
                    /* handled by prepared K-quant helper */
                } else if (prefill_qkv_stacked_batch_gpu(
                        m, lw, Q_buf, K_new, V_new, Xb, n_tokens,
                        q_buf_stride, dim, l) == 0) {
                    q_read_stride = q_buf_stride;
                } else if (prefill_qk_stacked_gpu(m, lw, Q_buf, K_new, Xb,
                                                  n_tokens, q_buf_stride, dim,
                                                  l) == 0) {
                    q_read_stride = q_buf_stride;
                    BnTransformerPrefillStackedAttentionGPUResourcePolicy
                        stacked_attn_resources =
                            bn_transformer_prefill_stacked_attention_gpu_resource_policy(
                                backend, l, attn_types);
                    if (prefill_quant_matmul_gpu_buf(
                            m, V_new, &lw->attn.wv,
                            (void *)stacked_attn_resources.wv, Xb,
                            n_tokens) != 0)
                        prefill_quant_matmul_gpu(m, V_new, &lw->attn.wv,
                                                 Xb, n_tokens, s->x_q);
                } else {
                    prefill_quant_matmul_multi(m, qkv_out, qkv_w, 3, Xb,
                                               n_tokens, s->x_q);
                }
                prefill_profile_add(&prof.qkv_ms, t_prof);
                for (int t = 0; t < n_tokens; t++) {
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Q_buf + (size_t)t * q_read_stride,
                        q_read_stride, "bitnet_attn_q_matmul", l,
                        pos0 + t);
                    if (has_kv) {
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, K_new + (size_t)t * layer_kv_dim,
                            layer_kv_dim, "bitnet_attn_k_matmul", l,
                            pos0 + t);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, V_new + (size_t)t * layer_kv_dim,
                            layer_kv_dim, "bitnet_attn_v_matmul", l,
                            pos0 + t);
                    }
                }
            }
            }

            BnTransformerPrefillAttentionModePolicy attention_mode =
                bn_transformer_prefill_attention_mode_policy(
                    bn_model_tq_state(m) != NULL,
                    bn_transformer_prefill_requires_token_attention(
                        prefill_cpu_runtime(m)),
                    gpu_hybrid_prefill);
            int used_prepared_gpu_attn = 0;
            int used_backend_qkv_prepare = 0;
            if (has_kv && !used_raw_prefill_attn_wo &&
                bn_model_transformer_policy_requires_reference_attention(c) &&
                plan.value_shares_key && gpu &&
                !q_gated && !lw->attn.q_bias && !lw->attn.k_bias &&
                !lw->attn.v_bias &&
                bn_gpu_backend_can_prefill_qkv_prepared(gpu) &&
                (!lw->attn.q_norm || raw_attn_resources.q_norm) &&
                (!lw->attn.k_norm || raw_attn_resources.k_norm)) {
                BnGPUAttentionPrefillPlan prepare_plan = {
                    .n_tokens = n_tokens, .n_heads = layer_n_heads,
                    .n_kv_heads = layer_n_kv_heads,
                    .head_size = layer_head_size,
                    .attention_window =
                        bn_transformer_attention_window(c, l),
                    .q_row_stride = q_read_stride,
                    .qk_norm_per_head = plan.qk_norm_per_head,
                    .pos0 = pos0, .rope_dims = layer_rope_dims,
                    .rope_freq_offset =
                        (size_t)l * (size_t)(max_head_size / 2),
                    .norm_eps = norm_eps,
                    .attention_scale =
                        bn_transformer_attention_scale(c, layer_head_size),
                };
                used_backend_qkv_prepare =
                    bn_gpu_backend_prefill_qkv_prepared(
                        gpu, Q_buf, K_new,
                        plan.value_shares_key ? V_new : NULL,
                        Q_buf, K_new, V_new,
                        (void *)raw_attn_resources.q_norm,
                        (void *)raw_attn_resources.k_norm,
                        &prepare_plan) == 0;
                if (used_backend_qkv_prepare)
                    q_read_stride = layer_q_dim;
            }
            if (has_kv && !used_raw_prefill_attn_wo && gpu && c->flash_attn &&
                !bn_transformer_prefill_requires_token_attention(prefill_cpu_runtime(m)) &&
                (plan.value_shares_key
                    ? bn_gpu_backend_can_prefill_attention_prepared_v(gpu)
                    : bn_gpu_backend_can_prefill_attention_prepared(gpu)) &&
                bn_transformer_prefill_attention_enabled(gpu) &&
                !bn_model_tq_state(m) &&
                !lw->attn.q_bias && !lw->attn.k_bias && !lw->attn.v_bias &&
                (!lw->attn.q_norm || raw_attn_resources.q_norm) &&
                (!lw->attn.k_norm || raw_attn_resources.k_norm)) {
                BnGPUAttentionPrefillPlan prepared_plan = {
                    .n_tokens = n_tokens, .n_heads = layer_n_heads,
                    .n_kv_heads = layer_n_kv_heads, .head_size = layer_head_size,
                    .attention_window = bn_transformer_attention_window(c, l),
                    .q_row_stride = q_read_stride, .q_gated = q_gated,
                    .qk_norm_per_head = plan.qk_norm_per_head,
                    .reference_rmsnorm_order =
                        bn_transformer_rmsnorm_uses_reference_order(c) &&
                        bn_gpu_backend_has_cap(
                            gpu, BN_GPU_CAP_REFERENCE_RMSNORM_ORDER),
                    .pos0 = pos0, .rope_dims = layer_rope_dims,
                    .rope_freq_offset = (size_t)l * (size_t)(max_head_size / 2),
                    .norm_eps = norm_eps,
                    .attention_scale = bn_transformer_attention_scale(c, layer_head_size),
                };
                int prepared_rc = plan.value_shares_key
                    ? bn_transformer_gpu_prefill_attention_prepared_v_backend_run(
                        gpu, Q_buf, K_new, V_new, Q_buf, K_new, V_new,
                        (void *)raw_attn_resources.q_norm,
                        (void *)raw_attn_resources.k_norm, &prepared_plan)
                    : bn_transformer_gpu_prefill_attention_prepared_backend_run(
                        gpu, Q_buf, K_new, Q_buf, K_new, V_new,
                        (void *)raw_attn_resources.q_norm,
                        (void *)raw_attn_resources.k_norm, &prepared_plan);
                used_prepared_gpu_attn = prepared_rc == 0;
            }
            if (attention_mode.use_batched_attention || used_prepared_gpu_attn) {
                // Phase 1: prepare K/V (bias, norm, RoPE) and write to cache
                t_prof = prefill_profile_now(&prof);
                if (has_kv && !used_raw_prefill_attn_wo &&
                    !used_prepared_gpu_attn &&
                    !used_backend_qkv_prepare) {
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        float *rc = rope_cos_buf + t * half_rope;
                        float *rs = rope_sin_buf + t * half_rope;

                        if (lw->attn.k_bias)
                            for (int i = 0; i < layer_kv_dim; i++) k_t[i] += lw->attn.k_bias[i];
                        if (lw->attn.v_bias)
                            for (int i = 0; i < layer_kv_dim; i++) v_t[i] += lw->attn.v_bias[i];
                        if (lw->attn.k_norm) {
                            int qk_stride = plan.qk_stride;
                            for (int h = 0; h < layer_n_kv_heads; h++)
                                prefill_cpu_ops()->rmsnorm(
                                    k_t + h * layer_head_size,
                                    k_t + h * layer_head_size,
                                    lw->attn.k_norm + h * qk_stride,
                                    layer_head_size, norm_eps);
                        }
                        if (plan.value_shares_key)
                            prefill_rmsnorm_unit_heads(v_t, layer_n_kv_heads,
                                                       layer_head_size, norm_eps);
                        bn_transformer_cpu_apply_rope_heads(
                                                            prefill_cpu_runtime(m),
                                                            k_t, layer_n_kv_heads,
                                                            layer_head_size,
                                                            layer_rope_dims, rc, rs);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, k_t, layer_kv_dim, "bitnet_attn_k", l, pos);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, v_t, layer_kv_dim, "bitnet_attn_v", l, pos);

                        if (bn_transformer_write_host_kv_cache_row(
                                s, plan.kv_mode, loff, cache_pos, kv_dim,
                                k_t, v_t, layer_kv_dim) != 0) {
                            sh_arena_free(pf_arena);
                            return NULL;
                        }
                    }
                } else if (has_kv) {
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        if (used_backend_qkv_prepare)
                            bn_transformer_cpu_debug_dump_prefill_values(
                                m, Q_buf + (size_t)t * q_read_stride,
                                layer_n_heads * layer_head_size,
                                "bitnet_attn_q", l, pos);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, k_t, layer_kv_dim, "bitnet_attn_k", l, pos);
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, v_t, layer_kv_dim, "bitnet_attn_v", l, pos);
                        if (bn_transformer_write_host_kv_cache_row(
                                s, plan.kv_mode, loff, cache_pos, kv_dim,
                                k_t, v_t, layer_kv_dim) != 0) {
                            sh_arena_free(pf_arena);
                            return NULL;
                        }
                    }
                }

                // Phase 2: batched attention (Q processing + attention, parallel over heads)
                BnBatchedAttnCtx bctx = {
                    .c = c, .runtime = prefill_cpu_runtime(m), .s = s,
                    .Q_buf = Q_buf, .K_new = K_new, .V_new = V_new,
                    .out = q_gated ? Xb2 : Q_buf,
                    .loff = has_kv ? loff : read_loff, .layer = l,
                    .pos0 = pos0, .n_tokens = n_tokens,
                    .n_heads = layer_n_heads, .n_kv_heads = layer_n_kv_heads,
                    .head_size = layer_head_size, .kv_dim = kv_dim,
                    .kv_mul = layer_kv_mul, .seq_len = c->seq_len,
                    .rope_dims = used_backend_qkv_prepare ? 0 : layer_rope_dims,
                    .rope_freq = s->rope_freq,
                    .rope_cos = rope_cos_buf, .rope_sin = rope_sin_buf,
                    .rope_stride = half_rope,
                    .attention_scale = bn_transformer_attention_scale(c, layer_head_size),
                    .kv_cache_uses_fp16_rows =
                        bn_transformer_kv_host_cache_uses_fp16_rows(c),
                    .uses_reference_dot_accumulation =
                        bn_transformer_prefill_uses_reference_dot_accumulation(c),
                    .q_norm = used_backend_qkv_prepare ? NULL : lw->attn.q_norm,
                    .k_norm = used_backend_qkv_prepare ? NULL : lw->attn.k_norm,
                    .q_bias = lw->attn.q_bias, .k_bias = lw->attn.k_bias,
                    .v_bias = lw->attn.v_bias,
                    .qk_norm_per_head = plan.qk_norm_per_head,
                    .norm_eps = norm_eps,
                    .q_gated = q_gated,
                    .wq_rows = attn_types.q_rows,
                    .q_row_stride = q_read_stride,
                    .wo_cols = wo_cols_attn,
                };
                t_prof = prefill_profile_now(&prof);
                int used_gpu_attn = used_raw_prefill_attn_wo || used_prepared_gpu_attn;
                void *attn_wo_buf = (void *)raw_attn_resources.wo;
                BnTransformerPrefillAttentionBatchPolicy attn_batch_policy =
                    bn_transformer_prefill_attention_batch_policy(
                        used_gpu_attn,
                        gpu != NULL,
                        has_kv &&
                            bn_transformer_prefill_attention_gpu_available(gpu) &&
                            (!bn_model_transformer_policy_requires_reference_attention(c) ||
                             used_backend_qkv_prepare),
                        bn_transformer_prefill_attention_wo_gpu_available(gpu),
                        bn_transformer_prefill_attention_enabled(prefill_gpu),
                        attn_wo_buf != NULL,
                        n_tokens,
                        bn_transformer_prefill_attention_min_tokens(
                            c, prefill_gpu),
                        lw->norm.attn_sub_norm != NULL,
                        attn_plan.use_post_norm,
                        lw->norm.attn_post_norm != NULL);
                if (attn_batch_policy.eligible &&
                    prefill_prepare_q_for_gpu_attention(&bctx) == 0) {
                    BnTransformerPrefillAttentionBatchCallPolicy
                        attn_call_policy =
                            bn_transformer_prefill_attention_batch_call_policy(
                                attn_batch_policy);
                    if (attn_call_policy.preferred_kind ==
                            BN_TRANSFORMER_PREFILL_ATTENTION_BATCH_WO &&
                        bn_transformer_gpu_prefill_attention_wo_backend_run(
                            gpu, Xb2, attn_wo_buf, Q_buf, K_new, V_new,
                            n_tokens, layer_n_heads, layer_n_kv_heads,
                            layer_head_size, layer_kv_mul, layer_kv_dim,
                            attn_types.out_rows, attn_types.out_cols,
                            attn_types.out_type,
                            bn_transformer_attention_scale(c, layer_head_size), bn_transformer_attention_window(c, l)) == 0) {
                        used_gpu_attn = 1;
                        used_fused_attn_wo = 1;
                    } else if (bn_transformer_gpu_prefill_attention_backend_run(
                            gpu, Q_buf, Q_buf, K_new, V_new, n_tokens,
                            layer_n_heads, layer_n_kv_heads, layer_head_size,
                            layer_kv_mul, layer_kv_dim,
                            bn_transformer_attention_scale(c, layer_head_size), bn_transformer_attention_window(c, l)) == 0) {
                        used_gpu_attn = 1;
                    } else {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                }
                if (!used_gpu_attn && bn_transformer_batched_attn_dispatch(m, &bctx) != 0) {
                    sh_arena_free(pf_arena);
                    return NULL;
                }
                if (!used_gpu_attn && q_gated) {
                    for (int t = 0; t < n_tokens; t++)
                        memcpy(Q_buf + (size_t)t * wo_cols_attn,
                               Xb2 + (size_t)t * wo_cols_attn,
                               (size_t)wo_cols_attn * sizeof(float));
                }
                prefill_profile_add(&prof.attn_cpu_ms, t_prof);
            } else {
                t_prof = prefill_profile_now(&prof);
                for (int t = 0; t < n_tokens; t++) {
                    int pos = pos0 + t;
                    int cache_pos = pos % c->seq_len;
                    float *q_t = Q_buf + (size_t)t * q_read_stride;
                    float *k_t = K_new + (size_t)t * layer_kv_dim;
                    float *v_t = V_new + (size_t)t * layer_kv_dim;

                    if (q_gated) {
                        for (int h = 0; h < layer_n_heads; h++)
                            memcpy(s->q + h * layer_head_size,
                                   q_t + h * 2 * layer_head_size,
                                   layer_head_size * sizeof(float));
                    } else {
                        memcpy(s->q, q_t, layer_q_dim * sizeof(float));
                    }

                    if (lw->attn.q_norm) {
                        int qk_stride = plan.qk_stride;
                        for (int h = 0; h < layer_n_heads; h++)
                            prefill_cpu_ops()->rmsnorm(
                                s->q + h * layer_head_size,
                                s->q + h * layer_head_size,
                                lw->attn.q_norm + h * qk_stride,
                                layer_head_size, norm_eps);
                    }
                    if (lw->attn.k_norm) {
                        int qk_stride = plan.qk_stride;
                        for (int h = 0; h < layer_n_kv_heads; h++)
                            prefill_cpu_ops()->rmsnorm(
                                k_t + h * layer_head_size,
                                k_t + h * layer_head_size,
                                lw->attn.k_norm + h * qk_stride,
                                layer_head_size, norm_eps);
                    }
                    if (plan.value_shares_key)
                        prefill_rmsnorm_unit_heads(v_t, layer_n_kv_heads,
                                                   layer_head_size, norm_eps);

                    if (lw->attn.q_bias) for (int i = 0; i < layer_q_dim; i++) s->q[i] += lw->attn.q_bias[i];
                    if (lw->attn.k_bias) for (int i = 0; i < layer_kv_dim; i++) k_t[i] += lw->attn.k_bias[i];
                    if (lw->attn.v_bias) for (int i = 0; i < layer_kv_dim; i++) v_t[i] += lw->attn.v_bias[i];

                    float *rc = rope_cos_buf + t * half_rope;
                    float *rs = rope_sin_buf + t * half_rope;
                    bn_transformer_cpu_apply_rope_heads(
                                                        prefill_cpu_runtime(m),
                                                        s->q, layer_n_heads,
                                                        layer_head_size,
                                                        layer_rope_dims, rc, rs);
                    bn_transformer_cpu_apply_rope_heads(
                                                        prefill_cpu_runtime(m),
                                                        k_t, layer_n_kv_heads,
                                                        layer_head_size,
                                                        layer_rope_dims, rc, rs);

                    if (bn_transformer_write_host_kv_cache_row(
                            s, plan.kv_mode, loff, cache_pos, kv_dim,
                            k_t, v_t, layer_kv_dim) != 0) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }

                    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                    BnGQACtx gctx = { c, s, loff, pos, n_kv, layer_kv_mul,
                                      layer_head_size, kv_dim, c->seq_len,
                                      bn_transformer_attention_scale(c, layer_head_size),
                                      bn_transformer_kv_host_cache_uses_fp16_rows(c),
                                      NULL, l, 0, 0 };
                    bn_transformer_cpu_gqa_dispatch(m, &gctx, layer_n_heads, layer_kv_mul);

                    if (q_gated) {
                        for (int h = 0; h < layer_n_heads; h++) {
                            float *gate_h = q_t + h * 2 * layer_head_size + layer_head_size;
                            float *xb_h = s->xb + h * layer_head_size;
                            for (int d = 0; d < layer_head_size; d++)
                                xb_h[d] *= 1.0f / (1.0f + expf(-gate_h[d]));
                        }
                    }

                    memcpy(Q_buf + (size_t)t * wo_cols_attn, s->xb,
                           wo_cols_attn * sizeof(float));
            }
                prefill_profile_add(&prof.attn_cpu_ms, t_prof);
                }

            if (!used_fused_attn_wo)
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Q_buf + (size_t)t * attn_types.out_cols,
                        attn_types.out_cols, "bitnet_attn_kqv", l, pos0 + t);

            {
                int wo_cols = attn_types.out_cols;
                t_prof = prefill_profile_now(&prof);
                if (!used_fused_attn_wo && lw->norm.attn_sub_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(
                            Q_buf + (size_t)t * wo_cols,
                            Q_buf + (size_t)t * wo_cols,
                            lw->norm.attn_sub_norm, wo_cols, norm_eps);
                if (!used_fused_attn_wo)
                    for (int t = 0; t < n_tokens; t++)
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, Q_buf + (size_t)t * wo_cols, wo_cols,
                            "bitnet_attn_wo_input", l, pos0 + t);
                if (!used_fused_attn_wo) {
                    void *wo_buf = (void *)raw_attn_resources.wo;
                    float *wo_out[1] = { Xb2 };
                    const BnQWeight *wo_weight[1] = { &lw->attn.wo };
                    int used_prepared_wo = !reference_attention &&
                        wo_cols > dim &&
                        prefill_try_prepared_kquant_multi(
                            m, &prepared_kquant, wo_out, wo_weight, 1,
                            Q_buf, wo_cols, n_tokens,
                            BN_PREFILL_KQUANT_REPACKED_WEIGHTS);
                    if (reference_attention)
                        prefill_quant_matmul_host(
                            m, Xb2, &lw->attn.wo, Q_buf,
                            n_tokens, s->x_q);
                    else if (!used_prepared_wo &&
                        prefill_quant_matmul_gpu_buf(
                            m, Xb2, &lw->attn.wo, wo_buf, Q_buf,
                            n_tokens) != 0)
                        prefill_quant_matmul_gpu(m, Xb2, &lw->attn.wo,
                                                 Q_buf, n_tokens, s->x_q);
                }
                if (!used_fused_attn_wo)
                    for (int t = 0; t < n_tokens; t++)
                        bn_transformer_cpu_debug_dump_prefill_values(
                            m, Xb2 + (size_t)t * dim, dim,
                            "bitnet_attn_wo", l, pos0 + t);
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, act + (size_t)t * dim, dim,
                        "bitnet_attn_residual_input", l, pos0 + t);
                if (!used_fused_attn_wo && attn_plan.use_post_norm &&
                    !hyper_connections && raw_attn_resources.attn_post_norm &&
                    (bn_transformer_gpu_prefill_norm_residual_backend_run(
                        gpu, act, (void *)raw_attn_resources.attn_post_norm,
                        Xb2, act, n_tokens, dim, norm_eps) == 0)) {
                    used_attn_residual = 1;
                } else if (!used_fused_attn_wo && attn_plan.use_post_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(Xb2 + (size_t)t * dim,
                                                   Xb2 + (size_t)t * dim,
                                                   lw->norm.attn_post_norm,
                                                   dim, norm_eps);
                prefill_profile_add(&prof.wo_ms, t_prof);
            }

            t_prof = prefill_profile_now(&prof);
            if (hyper_connections) {
                prefill_hc_combine_batch(m, sess, Xb2, hc_buffers.inject,
                                         hc_residual,
                                         n_tokens);
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, hc_residual + (size_t)t * hc_wide,
                        (int)hc_wide, "bitnet_hc_after_attn", l,
                        pos0 + t);
            } else if (!used_attn_residual)
                for (int t = 0; t < n_tokens; t++)
                    for (int d = 0; d < dim; d++)
                        act[(size_t)t * dim + d] += Xb2[(size_t)t * dim + d];
            prefill_profile_add(&prof.residual_ms, t_prof);
            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, act + (size_t)t * dim, dim,
                    "bitnet_attn_residual", l, pos0 + t);

        } else if (!is_attn) {
            BnTransformerSSMShapePolicy ssm_shape;
            if (!bn_transformer_ssm_shape_policy(&ssm_shape, c)) {
                sh_arena_free(pf_arena);
                return NULL;
            }
            int ssm_idx = plan.ssm_idx;

            if (bn_transformer_prefill_ssm_run_chain_enabled(prefill_gpu) &&
                gpu_hybrid_prefill &&
                (prefill_ssm_layer_chain_ready(m, lw, l, n_tokens) ||
                 prefill_ssm_moe_layer_chain_ready(m, lw, l, n_tokens))) {
                int run_end = l + 1;
                while (run_end < c->n_layers) {
                    BnLayerWeights *rlw = &w->layers[run_end];
                    BnLayerShapePlan rplan;
                    bn_transformer_plan_layer_shape(
                        &rplan, c, rlw, run_end,
                        bn_model_tq_state(m) != NULL);
                    if (rplan.is_attn ||
                        (!prefill_ssm_layer_chain_ready(
                             m, rlw, run_end, n_tokens) &&
                         !prefill_ssm_moe_layer_chain_ready(
                             m, rlw, run_end, n_tokens)))
                        break;
                    run_end++;
                }
                if (run_end - l > 1) {
                    for (int rl = l; rl < run_end; rl++) {
                        BnLayerWeights *rlw = &w->layers[rl];
                        BnLayerShapePlan rplan;
                        bn_transformer_plan_layer_shape(
                            &rplan, c, rlw, rl,
                            bn_model_tq_state(m) != NULL);
                        int r_did_ffn = 0;
                        BnTransformerPrefillLayerKindPolicy r_layer_kind =
                            bn_transformer_prefill_layer_kind_policy(rlw);
                        int r_is_moe = r_layer_kind.uses_moe;
                        float *layer_out =
                            (!r_is_moe && rl == run_end - 1) ? act : NULL;
                        const float *layer_in = (rl == l) ? act : NULL;
                        if (prefill_ssm_layer_gpu(
                                m, layer_out, rlw, layer_in, n_tokens, dim,
                                ssm_shape.qkv_dim, ssm_shape.value_dim,
                                ssm_shape.num_k_heads, ssm_shape.head_k_dim,
                                ssm_shape.num_v_heads, ssm_shape.head_v_dim,
                                ssm_shape.conv_kernel, rplan.ssm_idx, rl,
                                !r_is_moe, norm_eps, &r_did_ffn) != 0 ||
                            (!r_is_moe && !r_did_ffn)) {
                            sh_arena_free(pf_arena);
                            return NULL;
                        }
                        if (r_is_moe) {
                            float *moe_out =
                                (rl == run_end - 1) ? act : NULL;
                            if (prefill_moe_ffn_gpu_batch(
                                    m, moe_out, rlw, NULL, n_tokens, dim,
                                    rl) != 0) {
                                sh_arena_free(pf_arena);
                                return NULL;
                            }
                        }
                    }
                    l = run_end - 1;
                    goto prefill_layer_done;
                }
            }

            size_t state_per_layer = (size_t)ssm_shape.num_v_heads *
                                     ssm_shape.head_k_dim *
                                     ssm_shape.head_v_dim;
            float *ssm_state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
            size_t conv_per_layer =
                (size_t)(ssm_shape.conv_kernel - 1) * ssm_shape.qkv_dim;
            float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;

            int ssm_did_ffn = 0;
            if (prefill_ssm_layer_gpu(m, act, lw, act, n_tokens, dim,
                                      ssm_shape.qkv_dim, ssm_shape.value_dim,
                                      ssm_shape.num_k_heads,
                                      ssm_shape.head_k_dim,
                                      ssm_shape.num_v_heads,
                                      ssm_shape.head_v_dim,
                                      ssm_shape.conv_kernel, ssm_idx, l,
                                      n_tokens >=
                                          bn_transformer_prefill_moe_chain_min_tokens(
                                              c, bn_model_gpu(m)),
                                      norm_eps, &ssm_did_ffn) == 0) {
                if (ssm_did_ffn)
                    goto prefill_layer_done;
                goto prefill_ssm_done;
            }

            if (hyper_connections)
                memcpy(Xb, act, act_elems * sizeof(float));
            else
                for (int t = 0; t < n_tokens; t++)
                    prefill_cpu_ops()->rmsnorm(Xb + (size_t)t * dim,
                                               act + (size_t)t * dim,
                                               lw->norm.attn_norm, dim,
                                               norm_eps);
            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, Xb + (size_t)t * dim, dim,
                    "bitnet_ssm_norm", l, pos0 + t);

            if (q_buf_stride < ssm_shape.qkv_dim) { sh_arena_free(pf_arena); return NULL; }
            float *QKV_all = Q_buf;
            float *Z_all = Xb2;
            float *Out_all = Hb;
            const BnPrefillCPUOps *ssm_cpu_ops = prefill_cpu_ops();
            BnTransformerPrefillSSMProjectionTypes ssm_types;
            if (!bn_transformer_prefill_resolve_ssm_projection_types(
                    &ssm_types, lw)) {
                sh_arena_free(pf_arena);
                return NULL;
            }

            int ssm_prepared_kquant = 0;
            int reference_recurrent =
                bn_model_transformer_policy_requires_reference_recurrent(c);
            float *qz_out[2] = { QKV_all, Z_all };
            const BnQWeight *qz_w[2] = { &lw->ssm.wqkv, &lw->ssm.wz };
            ssm_prepared_kquant = prefill_try_prepared_kquant_multi(
                m, &prepared_kquant, qz_out, qz_w, 2, Xb, dim, n_tokens,
                BN_PREFILL_KQUANT_REPACKED_WEIGHTS);
            if (!ssm_prepared_kquant) {
                void (*project)(const BnModel *, float *, const BnQWeight *,
                                const float *, int, int8_t *) =
                    reference_recurrent ? prefill_quant_matmul_host
                                        : prefill_quant_matmul_gpu;
                project(m, QKV_all, &lw->ssm.wqkv, Xb, n_tokens, s->x_q);
                project(m, Z_all, &lw->ssm.wz, Xb, n_tokens, s->x_q);
            }

            if (hb_stride < 2 * ssm_shape.num_v_heads) {
                sh_arena_free(pf_arena);
                return NULL;
            }
            float *Alpha_all = Hb2;
            float *Beta_all = Alpha_all +
                              (size_t)n_tokens * ssm_shape.num_v_heads;
            {
                float *ab_out[2] = { Alpha_all, Beta_all };
                const BnQWeight *ab_w[2] = {
                    &lw->ssm.ssm_alpha, &lw->ssm.ssm_beta
                };
                if (!prefill_try_prepared_kquant_multi(
                        m, &prepared_kquant, ab_out, ab_w, 2, Xb, dim,
                        n_tokens, BN_PREFILL_KQUANT_REPACKED_WEIGHTS)) {
                    if (reference_recurrent) {
                        prefill_quant_matmul_host(
                            m, Alpha_all, &lw->ssm.ssm_alpha,
                            Xb, n_tokens, s->x_q);
                        prefill_quant_matmul_host(
                            m, Beta_all, &lw->ssm.ssm_beta,
                            Xb, n_tokens, s->x_q);
                    } else {
                        prefill_quant_matmul_multi(m, ab_out, ab_w, 2, Xb,
                                                   n_tokens, s->x_q);
                    }
                }
            }

            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, QKV_all + (size_t)t * ssm_types.qkv_rows,
                    ssm_shape.qkv_dim, "bitnet_ssm_qkv", l, pos0 + t);
            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, Z_all + (size_t)t * ssm_types.z_rows,
                    ssm_shape.value_dim, "bitnet_ssm_z", l, pos0 + t);

            int used_backend_conv_norm =
                !bn_transformer_prefill_host_reference_enabled(
                    bn_model_gpu(m), c) &&
                bn_transformer_gpu_ssm_conv_l2norm_batch(
                    bn_model_gpu(m), m, lw, l, QKV_all, conv_state,
                    n_tokens, ssm_shape.qkv_dim, ssm_shape.conv_kernel,
                    ssm_shape.num_k_heads, ssm_shape.head_k_dim,
                    norm_eps) == 0;

            for (int t = 0; t < n_tokens; t++) {
                float *qkv_t = QKV_all + (size_t)t * ssm_types.qkv_rows;
                if (!used_backend_conv_norm) {
                    BnSSMConvCtx conv_ctx = {
                        qkv_t, conv_state, lw->ssm.ssm_conv1d,
                        ssm_shape.qkv_dim, ssm_shape.conv_kernel,
                        bn_transformer_ssm_uses_sigmoid_gate(c)
                    };
                    BnTPTask conv_task = {
                        bn_transformer_prefill_ssm_conv_silu_op(ssm_cpu_ops),
                        &conv_ctx, ssm_shape.qkv_dim
                    };
                    bn_tp_dispatch(bn_model_pool(m), &conv_task, 1);
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, qkv_t, ssm_shape.qkv_dim,
                        "bitnet_ssm_conv", l, pos0 + t);

                    BnSSML2NormCtx norm_ctx = {
                        qkv_t, qkv_t + ssm_shape.key_dim,
                        norm_eps, ssm_shape.head_k_dim
                    };
                    BnTPTask norm_task = {
                        bn_transformer_prefill_ssm_l2norm_op(ssm_cpu_ops),
                        &norm_ctx, ssm_shape.num_k_heads
                    };
                    bn_tp_dispatch(bn_model_pool(m), &norm_task, 1);
                }
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, qkv_t, ssm_shape.qkv_dim,
                    "bitnet_ssm_qkv_norm", l, pos0 + t);

                float *alpha_arr = Alpha_all +
                                   (size_t)t * ssm_shape.num_v_heads;
                float *beta_arr = Beta_all +
                                  (size_t)t * ssm_shape.num_v_heads;
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, alpha_arr, ssm_shape.num_v_heads,
                    "bitnet_ssm_alpha", l, pos0 + t);
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, beta_arr, ssm_shape.num_v_heads,
                    "bitnet_ssm_beta", l, pos0 + t);
            }

            float q_scale = 1.0f / sqrtf((float)ssm_shape.head_k_dim);
            int used_backend_delta_gate =
                !bn_model_transformer_policy_requires_reference_recurrent(c) &&
                bn_transformer_gpu_ssm_delta_gate_batch(
                    bn_model_gpu(m), m, lw, l, Out_all, ssm_state,
                    QKV_all, Z_all, Alpha_all, Beta_all, n_tokens,
                    ssm_shape.num_k_heads, ssm_shape.head_k_dim,
                    ssm_shape.num_v_heads, ssm_shape.head_v_dim,
                    q_scale, norm_eps,
                    bn_transformer_ssm_uses_sigmoid_gate(c)) == 0;
            for (int t = 0; t < n_tokens; t++) {
                float *alpha_arr = Alpha_all +
                    (size_t)t * ssm_shape.num_v_heads;
                float *beta_arr = Beta_all +
                    (size_t)t * ssm_shape.num_v_heads;
                for (int h = 0; h < ssm_shape.num_v_heads; h++)
                    alpha_arr[h] += lw->ssm.ssm_dt_bias[h];
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, alpha_arr, ssm_shape.num_v_heads,
                    "bitnet_ssm_alpha_biased", l, pos0 + t);
                for (int h = 0; h < ssm_shape.num_v_heads; h++) {
                    float dt = alpha_arr[h];
                    alpha_arr[h] =
                        (dt > 20.0f) ? dt : logf(1.0f + expf(dt));
                }
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, alpha_arr, ssm_shape.num_v_heads,
                    "bitnet_ssm_softplus", l, pos0 + t);
                for (int h = 0; h < ssm_shape.num_v_heads; h++)
                    alpha_arr[h] *= lw->ssm.ssm_a[h];
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, alpha_arr, ssm_shape.num_v_heads,
                    "bitnet_ssm_log_decay", l, pos0 + t);
                for (int h = 0; h < ssm_shape.num_v_heads; h++) {
                    alpha_arr[h] = expf(alpha_arr[h]);
                    beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
                }
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, alpha_arr, ssm_shape.num_v_heads,
                    "bitnet_ssm_decay", l, pos0 + t);
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, beta_arr, ssm_shape.num_v_heads,
                    "bitnet_ssm_beta_sigmoid", l, pos0 + t);
            }
            if (used_backend_delta_gate) {
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, ssm_state, (int)state_per_layer,
                    "bitnet_ssm_state", l,
                    pos0 + n_tokens - 1);
            } else {
                for (int t = 0; t < n_tokens; t++) {
                    float *qkv_t = QKV_all +
                                   (size_t)t * ssm_types.qkv_rows;
                    float *out_t = Out_all +
                                   (size_t)t * ssm_shape.value_dim;
                    float *alpha_arr = Alpha_all +
                        (size_t)t * ssm_shape.num_v_heads;
                    float *beta_arr = Beta_all +
                        (size_t)t * ssm_shape.num_v_heads;
                    BnSSMDeltaCtx delta_ctx = {
                        ssm_state, out_t, qkv_t,
                        qkv_t + ssm_shape.key_dim,
                        qkv_t + 2 * ssm_shape.key_dim,
                        alpha_arr, beta_arr,
                        ssm_shape.num_k_heads, ssm_shape.head_k_dim,
                        ssm_shape.head_v_dim, q_scale
                    };
                    BnTPTask delta_task = {
                        bn_transformer_prefill_ssm_delta_op(ssm_cpu_ops),
                        &delta_ctx, ssm_shape.num_v_heads
                    };
                    ssm_cpu_ops->dispatch_ssm_heads(
                        bn_model_pool(m), &delta_task, 1);
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, out_t, ssm_shape.value_dim,
                        "bitnet_ssm_delta", l, pos0 + t);
                    BnSSMGateCtx gate_ctx = {
                        out_t, Z_all + (size_t)t * ssm_types.z_rows,
                        lw->ssm.ssm_norm, norm_eps,
                        ssm_shape.head_v_dim,
                        bn_transformer_ssm_uses_sigmoid_gate(c),
                        ssm_shape.num_v_heads
                    };
                    BnTPTask gate_task = {
                        bn_transformer_prefill_ssm_gate_op(ssm_cpu_ops),
                        &gate_ctx, ssm_shape.num_v_heads
                    };
                    ssm_cpu_ops->dispatch_ssm_heads(
                        bn_model_pool(m), &gate_task, 1);
                }
            }
            for (int t = 0; t < n_tokens; t++) {
                float *out_t = Out_all +
                               (size_t)t * ssm_shape.value_dim;
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, out_t, ssm_shape.value_dim,
                    "bitnet_ssm_gate", l, pos0 + t);
            }

            if (reference_recurrent) {
                float *ssm_out_values[1] = { Xb };
                const BnQWeight *ssm_out_weights[1] = { &lw->ssm.ssm_out };
                if (!prefill_try_prepared_kquant_multi(
                        m, &prepared_kquant, ssm_out_values,
                        ssm_out_weights, 1, Out_all, ssm_shape.value_dim,
                        n_tokens, BN_PREFILL_KQUANT_REPACKED_WEIGHTS))
                    prefill_quant_matmul_host(
                        m, Xb, &lw->ssm.ssm_out, Out_all, n_tokens, s->x_q);
            } else
                prefill_quant_matmul_gpu(m, Xb, &lw->ssm.ssm_out, Out_all,
                                         n_tokens, s->x_q);

            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, Xb + (size_t)t * dim, dim,
                    "bitnet_ssm_out", l, pos0 + t);

            if (hyper_connections) {
                prefill_hc_combine_batch(m, sess, Xb, hc_buffers.inject,
                                         hc_residual,
                                         n_tokens);
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, hc_residual + (size_t)t * hc_wide,
                        (int)hc_wide, "bitnet_hc_after_attn", l,
                        pos0 + t);
            } else
                for (int t = 0; t < n_tokens; t++)
                    for (int d = 0; d < dim; d++)
                        act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
        }
prefill_ssm_done:
        ;

        /* Persistent token/KV/SSM/PLE state is complete for this layer.
         * Only the requested output row needs the final stateless FFN.
         * Keep all rows for all-logit requests and backend-managed chains. */
        if (!prefill_gpu && !all_logits && l == c->n_layers - 1 &&
            n_tokens > 1 &&
            bn_transformer_cpu_backend_selects_last_prefill_ffn_row()) {
            int last = n_tokens - 1;
            act += (size_t)last * dim;
            if (hyper_connections)
                hc_residual += (size_t)last * hc_wide;
            pos0 += last;
            n_tokens = 1;
        }

        if (hyper_connections &&
            prefill_hc_mix_batch(m, sess, &lw->hc_ffn, act,
                                 hc_residual, hc_buffers,
                                 n_tokens, 1, "bitnet_hc_ffn_norm",
                                 l, pos0) != 0) {
            sh_arena_free(pf_arena);
            return NULL;
        }

        BnTransformerPrefillLayerKindPolicy layer_kind =
            bn_transformer_prefill_layer_kind_policy(lw);
        if (layer_kind.uses_moe) {
            BnPrefillMoEDumpCtx moe_dump = {
                .model = m, .layer = l, .pos0 = pos0,
                .active_experts = bn_moe_route_policy(c).active_experts
            };
            const BnCPURuntimePolicy *runtime = prefill_cpu_runtime(m);
            BnMoEBatchObserveFn observe =
                bn_transformer_cpu_debug_dump_path(runtime) ||
                bn_transformer_cpu_debug_binary_path(runtime)
                ? prefill_moe_debug_observe : NULL;
            if (bn_moe_forward_batch_observed(
                    m, sess, lw, l, act, Xb, n_tokens,
                    hyper_connections
                        ? BN_MOE_BATCH_INPUT_PRENORMALIZED |
                              BN_MOE_BATCH_OUTPUT_RAW
                        : 0,
                    observe, &moe_dump) != 0) {
                sh_arena_free(pf_arena);
                return NULL;
            }
            if (hyper_connections)
                prefill_hc_combine_batch(m, sess, act, hc_buffers.inject,
                                         hc_residual,
                                         n_tokens);
            if (hyper_connections)
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, hc_residual + (size_t)t * hc_wide,
                        (int)hc_wide, "bitnet_hc_after_ffn", l,
                        pos0 + t);
            for (int t = 0; t < n_tokens; t++)
                bn_transformer_cpu_debug_dump_prefill_values(
                    m, Xb + (size_t)t * dim, dim, "bitnet_ffn_norm", l,
                    pos0 + t);
        } else if (lw->ffn.ffn_up.data) {
            BnTransformerPrefillFFNProjectionTypes ffn_types;
            if (!bn_transformer_prefill_resolve_ffn_projection_types(
                    &ffn_types, lw)) {
                sh_arena_free(pf_arena);
                return NULL;
            }
            int used_gpu_batch_ffn = 0;
            int used_ffn_residual = 0;
            t_prof = prefill_profile_now(&prof);
            BnGPUBackend *gpu_ffn = bn_model_gpu(m);
            const BnBackendModel *backend_ffn = bn_model_backend(m);
            int dense_ffn_batch_min_tokens =
                bn_transformer_prefill_dense_chain_min_tokens(c, gpu_ffn);
            int can_use_dense_ffn_batch =
                bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
                    gpu_ffn, c, n_tokens);
            BnTransformerPrefillDenseFFNGPUResourcePolicy ffn_resources =
                bn_transformer_prefill_dense_ffn_gpu_resource_policy(
                    backend_ffn, l, lw, ffn_types);
            void *ffn_norm_buf = (void *)ffn_resources.ffn_norm;
            BnTransformerPrefillFFNBatchPolicy ffn_batch_policy =
                bn_transformer_prefill_ffn_batch_policy(
                    ffn_plan.has_gate,
                    can_use_dense_ffn_batch,
                    bn_transformer_prefill_dense_ffn_batch_norm_resid_gpu_available(
                        gpu_ffn),
                    ffn_norm_buf != NULL,
                    n_tokens,
                    dense_ffn_batch_min_tokens,
                    sequence_policy.uses_hybrid_layer_layout,
                    lw->norm.ffn_sub_norm != NULL,
                    ffn_plan.use_post_norm,
                    lw->norm.ffn_post_norm != NULL);
            if (ffn_batch_policy.fuses_norm_residual &&
                prefill_dense_ffn_gpu_batch(m, act, lw, act, n_tokens,
                                            dim, ffn_plan.hidden_dim,
                                            ffn_plan.activation, l, ffn_norm_buf,
                                            norm_eps, 1) == 0) {
                used_gpu_batch_ffn = 1;
                used_ffn_residual = 1;
            }
            if (!used_gpu_batch_ffn) {
                prefill_profile_add(&prof.ffn_ms, t_prof);
                t_prof = prefill_profile_now(&prof);
                for (int t = 0; t < n_tokens; t++)
                    prefill_cpu_ops()->rmsnorm(Xb + t * dim,
                                               act + (size_t)t * dim,
                                               lw->norm.ffn_norm, dim,
                                               norm_eps);
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Xb + (size_t)t * dim, dim,
                        "bitnet_ffn_norm", l, pos0 + t);
                prefill_profile_add(&prof.ffn_norm_ms, t_prof);

                t_prof = prefill_profile_now(&prof);
                if (ffn_batch_policy.eligible &&
                    prefill_dense_ffn_gpu_batch(m, Xb, lw, Xb, n_tokens,
                                                dim, ffn_plan.hidden_dim,
                                                ffn_plan.activation, l, NULL,
                                                0.0f, 0) == 0) {
                    used_gpu_batch_ffn = 1;
                } else if (ffn_plan.has_gate) {
                    double t_ffn_step = prefill_profile_now(&prof);
                    float *gu_out[2] = { Hb, Hb2 };
                    const BnQWeight *gu_w[2] = {
                        &lw->ffn.ffn_gate, &lw->ffn.ffn_up
                    };
                    if (!prefill_try_prepared_kquant_multi(
                            m, &prepared_kquant, gu_out, gu_w, 2, Xb, dim,
                            n_tokens, BN_PREFILL_KQUANT_REPACKED_WEIGHTS))
                        prefill_quant_matmul_multi(m, gu_out, gu_w, 2, Xb,
                                                   n_tokens, s->x_q);
                for (int t = 0; t < n_tokens; t++) {
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Hb + (size_t)t * ffn_plan.hidden_dim,
                        ffn_plan.hidden_dim, "bitnet_ffn_gate", l, pos0 + t);
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Hb2 + (size_t)t * ffn_plan.hidden_dim,
                        ffn_plan.hidden_dim, "bitnet_ffn_up", l, pos0 + t);
                }
                prefill_profile_add(&prof.ffn_gateup_ms, t_ffn_step);

                t_ffn_step = prefill_profile_now(&prof);
                BnTransformerPrefillActivationPolicy activation_policy =
                    bn_transformer_prefill_activation_policy(
                        gpu_ffn, ffn_plan.activation,
                        ffn_plan.reference_activation);
                BnPrefillFFNActCtx act_ctx = {
                    Hb, Hb2, ffn_plan.hidden_dim,
                    activation_policy.activation,
                    activation_policy.uses_reference_activation
                };
                BnTPTask act_task = {
                    prefill_cpu_ops()->ffn_activation, &act_ctx, n_tokens
                };
                bn_tp_dispatch(bn_model_pool(m), &act_task, 1);
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Hb + (size_t)t * ffn_plan.hidden_dim,
                        ffn_plan.hidden_dim, "bitnet_ffn_swiglu", l,
                        pos0 + t);
                prefill_profile_add(&prof.ffn_act_ms, t_ffn_step);
            } else {
                double t_ffn_step = prefill_profile_now(&prof);
                prefill_quant_matmul_gpu(m, Hb, &lw->ffn.ffn_up, Xb, n_tokens, s->x_q);
                prefill_profile_add(&prof.ffn_gateup_ms, t_ffn_step);
                t_ffn_step = prefill_profile_now(&prof);
                BnTransformerPrefillActivationPolicy activation_policy =
                    bn_transformer_prefill_activation_policy(
                        gpu_ffn, ffn_plan.activation,
                        ffn_plan.reference_activation);
                BnPrefillFFNActCtx act_ctx = {
                    Hb, NULL, ffn_plan.hidden_dim,
                    activation_policy.activation,
                    activation_policy.uses_reference_activation
                };
                BnTPTask act_task = {
                    prefill_cpu_ops()->ffn_activation, &act_ctx, n_tokens
                };
                bn_tp_dispatch(bn_model_pool(m), &act_task, 1);
                prefill_profile_add(&prof.ffn_act_ms, t_ffn_step);
                }
            }

            if (!used_gpu_batch_ffn) {
                if (lw->norm.ffn_sub_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(
                            Hb + (size_t)t * hidden_dim,
                            Hb + (size_t)t * hidden_dim,
                            lw->norm.ffn_sub_norm, hidden_dim, norm_eps);

                double t_ffn_step = prefill_profile_now(&prof);
                float *down_out[1] = { Xb };
                const BnQWeight *down_weight[1] = { &lw->ffn.ffn_down };
                if (!prefill_try_prepared_kquant_multi(
                        m, &prepared_kquant, down_out, down_weight, 1, Hb,
                        hidden_dim, n_tokens,
                        BN_PREFILL_KQUANT_REPACKED_WEIGHTS))
                    prefill_quant_matmul_gpu(m, Xb, &lw->ffn.ffn_down, Hb,
                                             n_tokens, s->x_q);
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Xb + (size_t)t * dim, dim,
                        "bitnet_ffn_down_raw", l, pos0 + t);
                prefill_profile_add(&prof.ffn_down_ms, t_ffn_step);
                if (ffn_plan.use_post_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(Xb + (size_t)t * dim,
                                                   Xb + (size_t)t * dim,
                                                   lw->norm.ffn_post_norm,
                                                   dim, norm_eps);
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, Xb + (size_t)t * dim, dim,
                        "bitnet_ffn_out", l, pos0 + t);
            }
            prefill_profile_add(&prof.ffn_ms, t_prof);

            if (used_ffn_residual) {
                for (int t = 0; t < n_tokens; t++)
                    bn_transformer_cpu_debug_dump_prefill_values(
                        m, act + (size_t)t * dim, dim, "bitnet_lout", l,
                        pos0 + t);
                continue;
            }

            t_prof = prefill_profile_now(&prof);
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
            prefill_profile_add(&prof.residual_ms, t_prof);
        }

        if (ffn_plan.use_layer_output_scale) {
            float scale = lw->norm.layer_output_scale[0];
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] *= scale;
        }
prefill_layer_done:
        for (int t = 0; t < n_tokens; t++)
            bn_transformer_cpu_debug_dump_prefill_values(
                m, act + (size_t)t * dim, dim, "bitnet_lout", l,
                pos0 + t);
    }

prefill_layers_done:
    if (hyper_connections && s->hc_residual)
        memcpy(s->hc_residual,
               hc_residual + (size_t)(n_tokens - 1) * hc_wide,
               hc_wide * sizeof(float));
    if (hyper_connections &&
        prefill_hc_mix_batch(m, sess, &w->hc_output, act,
                             hc_residual, hc_buffers,
                             n_tokens, 0, "bitnet_hc_output_norm",
                             c->n_layers, pos0) != 0) {
        sh_arena_free(pf_arena);
        return NULL;
    }
    if (all_logits) {
        int vocab_size = c->vocab_size;
        t_prof = prefill_profile_now(&prof);
        for (int t = 0; t < n_tokens; t++) {
            memcpy(s->x, act + (size_t)t * dim, dim * sizeof(float));
            float *lg = prefill_logits(m, sess);
            if (!lg) { sh_arena_free(pf_arena); return NULL; }
            memcpy(all_logits + (size_t)t * vocab_size, lg,
                   vocab_size * sizeof(float));
        }
        prefill_profile_add(&prof.logits_ms, t_prof);
        if (prof.enabled) {
            fprintf(stderr,
                    "[bn:prefill:profile] tokens=%d layers=%d embed=%.3f attn_norm=%.3f qkv=%.3f attn_cpu=%.3f wo=%.3f ffn_norm=%.3f ffn=%.3f gateup=%.3f act=%.3f down=%.3f residual=%.3f logits=%.3f\n",
                    request_n_tokens, c->n_layers, prof.embed_ms, prof.attn_norm_ms,
                    prof.qkv_ms, prof.attn_cpu_ms, prof.wo_ms,
                    prof.ffn_norm_ms, prof.ffn_ms, prof.ffn_gateup_ms,
                    prof.ffn_act_ms, prof.ffn_down_ms, prof.residual_ms,
                    prof.logits_ms);
        }
        sh_arena_free(pf_arena);
        return s->logits;
    }

    if (prof.enabled) {
        fprintf(stderr,
                "[bn:prefill:profile] tokens=%d layers=%d embed=%.3f attn_norm=%.3f qkv=%.3f attn_cpu=%.3f wo=%.3f ffn_norm=%.3f ffn=%.3f gateup=%.3f act=%.3f down=%.3f residual=%.3f logits=%.3f\n",
                request_n_tokens, c->n_layers, prof.embed_ms, prof.attn_norm_ms,
                prof.qkv_ms, prof.attn_cpu_ms, prof.wo_ms,
                prof.ffn_norm_ms, prof.ffn_ms, prof.ffn_gateup_ms,
                prof.ffn_act_ms, prof.ffn_down_ms, prof.residual_ms,
                prof.logits_ms);
    }
    memcpy(s->x, act + (size_t)(n_tokens - 1) * dim, dim * sizeof(float));
    sh_arena_free(pf_arena);
    if (need_last_logits)
        return prefill_logits(m, sess);
    return s->x;
}

float *bn_transformer_prefill(BnModel *m, BnSession *s, const int *tokens,
                              int n_tokens, int pos0) {
    BnTransformerPrefillEntryDispatchPolicy dispatch =
        prefill_entry_dispatch_policy(
            m, BN_TRANSFORMER_PREFILL_REQUEST_LAST_LOGITS);
    if (dispatch.path == BN_TRANSFORMER_PREFILL_ENTRY_DECODE_LAST_LOGITS) {
        for (int i = 0; i + 1 < n_tokens; i++)
            if (bn_transformer_forward_no_logits(m, s, tokens[i], pos0 + i) != 0)
                return NULL;
        return n_tokens > 0
            ? bn_transformer_forward(m, s, tokens[n_tokens - 1],
                                     pos0 + n_tokens - 1)
            : NULL;
    }
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL, 1, 0);
}

int bn_transformer_prefill_no_logits(BnModel *m, BnSession *s, const int *tokens,
                                     int n_tokens, int pos0) {
    BnTransformerPrefillEntryDispatchPolicy dispatch =
        prefill_entry_dispatch_policy(
            m, BN_TRANSFORMER_PREFILL_REQUEST_NO_LOGITS);
    if (dispatch.path == BN_TRANSFORMER_PREFILL_ENTRY_DECODE_NO_LOGITS) {
        for (int i = 0; i < n_tokens; i++)
            if (bn_transformer_forward_no_logits(m, s, tokens[i], pos0 + i) != 0)
                return -1;
        return 0;
    }
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL, 0, 0) ? 0 : -1;
}

int bn_transformer_prefill_all(BnModel *m, BnSession *s, const int *tokens,
                               int n_tokens, int pos0, float *all_logits) {
    if (!all_logits || n_tokens <= 0) return -1;

    BnTransformerPrefillEntryDispatchPolicy dispatch =
        prefill_entry_dispatch_policy(
            m, BN_TRANSFORMER_PREFILL_REQUEST_ALL_LOGITS);
    if (dispatch.path == BN_TRANSFORMER_PREFILL_ENTRY_DECODE_ALL_LOGITS) {
        size_t row_bytes = (size_t)m->config.vocab_size * sizeof(float);
        for (int i = 0; i < n_tokens; i++) {
            float *logits = bn_transformer_forward(m, s, tokens[i], pos0 + i);
            if (!logits) return -1;
            memcpy(all_logits + (size_t)i * m->config.vocab_size,
                   logits, row_bytes);
        }
        return 0;
    }

    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, s, tokens[0], pos0);
        if (!logits) return -1;
        memcpy(all_logits, logits, m->config.vocab_size * sizeof(float));
        return 0;
    }

    float *result = prefill_internal(m, s, tokens, n_tokens, pos0, all_logits, 1, 0);
    return result ? 0 : -1;
}
