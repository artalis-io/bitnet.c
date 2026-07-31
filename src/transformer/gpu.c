#include "gpu_internal.h"
#include "platform.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void gpu_cpu_quant_matvec(BnModel *m,
                                 float *out,
                                 const BnQWeight *W,
                                 const float *x,
                                 int8_t *quantized_buf) {
    bn_transformer_gpu_cpu_quant_matvec_model(
        m, out, W, x, quantized_buf);
}

static void gpu_debug_compare_vec_local(const char *label,
                                        int layer,
                                        int pos,
                                        const float *cpu,
                                        const float *gpu,
                                        int n) {
    if (!label || !cpu || !gpu || n <= 0) return;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - gpu[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] %s layer=%d pos=%d max_abs=%.9g max_i=%d "
            "cpu=%.9g gpu=%.9g mean_abs=%.9g rms=%.9g\n",
            label, layer, pos, max_abs, max_i, cpu[max_i], gpu[max_i],
            sum_abs / (double)n, sqrt(sum_sq / (double)n));
}

static void gpu_debug_rmsnorm_scalar_local(float *out,
                                           const float *x,
                                           const float *w,
                                           int n,
                                           float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

static void gpu_moe_route_profile_add(int dim,
                                      int n_experts,
                                      double flush_ms,
                                      double read_ms,
                                      double route_ms,
                                      double resolve_ms) {
    static unsigned long long calls = 0;
    static double total_flush = 0.0;
    static double total_read = 0.0;
    static double total_route = 0.0;
    static double total_resolve = 0.0;
    if (!bn_transformer_gpu_moe_route_profile_enabled())
        return;
    calls++;
    total_flush += flush_ms;
    total_read += read_ms;
    total_route += route_ms;
    total_resolve += resolve_ms;
    int every = bn_transformer_gpu_moe_route_profile_every();
    if ((calls % (unsigned long long)every) == 0) {
        fprintf(stderr,
                "[bn:gpu:moe_route_profile] calls=%llu dim=%d experts=%d "
                "flush=%.3f read=%.3f route=%.3f resolve=%.3f total=%.3f\n",
                calls, dim, n_experts, total_flush, total_read,
                total_route, total_resolve,
                total_flush + total_read + total_route + total_resolve);
        total_flush = 0.0;
        total_read = 0.0;
        total_route = 0.0;
        total_resolve = 0.0;
    }
}

static int gpu_resolve_moe_all_active_two_resources(
    BnGPUMoEResources *out,
    BnGPUMoEResolvedExpert *storage,
    BnModel *m,
    BnSession *sess,
    const BnLayerWeights *lw,
    int layer,
    void *router_diff,
    BnGPUMoETemporaryBuffers *temps) {
    if (!out || !storage || !m || !sess || !lw || !router_diff || !temps)
        return -1;
    BnConfig *c = &m->config;
    BnTransformerGPUMoEAllActiveTwoResourcePolicy policy =
        bn_transformer_gpu_moe_all_active_two_resource_policy(c);
    if (!policy.enabled)
        return -1;
    memset(out, 0, sizeof(*out));
    memset(temps, 0, sizeof(*temps));
    out->expert_map = &lw->moe.expert_map;
    out->experts = storage;
    out->n_experts = policy.total_experts;
    out->moe_hidden = policy.expert_hidden_dim;
    for (int e = 0; e < policy.total_experts; e++) {
        memset(&storage[e], 0, sizeof(storage[e]));
        if (bn_gpu_moe_bridge_get_expert(m, sess, lw, layer, e, temps,
                                         &storage[e].buffers) != 0)
            return -1;
        storage[e].weight = 1.0f;
        storage[e].route_gate = router_diff;
        storage[e].route_complement =
            e >= policy.complement_route_from_expert;
    }
    return 0;
}

// GPU-resident forward pass: one submit per token, reads back logits only.
// Supports classic transformer only (no MoE, no SSM, no gated-Q, no wide-Q,
// no Q/K norms, no sub-norms, no FP16 KV cache).
// Supports attention biases and tied embeddings.
// Returns s->logits on success, NULL to fall back to CPU.
static float *bn_transformer_gpu_forward_impl(BnModel *m, BnSession *sess,
                                              int token, int pos,
                                              int need_logits,
                                              int *argmax_token,
                                              const int *penalty_tokens,
                                              int n_penalty_tokens,
                                              float repeat_penalty) {
    /* no-op */
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
    BnTransformerGPUEmitContext emit;
    bn_transformer_gpu_emit_context_init(&emit, NULL, 0);
    int emit_logits = need_logits || argmax_token != NULL;
    if (!bn_transformer_gpu_model_argmax_available(
            m, argmax_token != NULL))
        return NULL;

    BnTransformerGPUForwardPolicy policy;
    const char *reject_reason = NULL;
    if (bn_transformer_gpu_validate_model_forward(
            &policy, m, token, pos, &reject_reason) != 0)
        return bn_transformer_gpu_reject_forward(&emit, reject_reason);
    BnGPUBackend *gpu = policy.gpu;

    int dim = c->dim;
    int kv_cache_stride = c->kv_dim;
    BnTransformerGPUMoEExecutionPolicy route_policy =
        bn_transformer_gpu_moe_execution_policy(c);
    BnTransformerGPUMoEActivationPolicy moe_activation =
        bn_transformer_gpu_moe_activation_policy(c);
    int max_rope_dims = bn_transformer_rope_dims_for_head(
        c, bn_transformer_attention_head_size(c, NULL));
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerShapePlan shape;
        bn_transformer_plan_layer_shape(&shape, c, &w->layers[l], l,
                                        policy.has_tq);
        if (!shape.is_attn)
            continue;
        int layer_rope_dims =
            bn_transformer_rope_dims_for_head(c, shape.head_size);
        if (layer_rope_dims > max_rope_dims)
            max_rope_dims = layer_rope_dims;
    }
    int half_rope = max_rope_dims / 2;
    float rope_cos[half_rope], rope_sin[half_rope];
    for (int i = 0; i < half_rope; i++) {
        float angle = pos * s->rope_freq[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }
    int cache_pos = pos % c->seq_len;
    int compare_attention_layer = -1;
    int compare_attention_pos = -1;
    int compare_gqa_layer = -1;
    int compare_gqa_pos = -1;
    int compare_qkv_layer = -1;
    int compare_qkv_pos = -1;
    int compare_ffn_down_layer = -1;
    int compare_ffn_down_pos = -1;
    int compare_ffn_state_layer = -1;
    int compare_ffn_state_pos = -1;
    BnTransformerGPUCPUFallbackPolicy cpu_fallback =
        bn_transformer_gpu_cpu_fallback_policy();
    BnTransformerGPUSmallDenseNativeQuantLayerPolicy small_dense_native_quant =
        bn_transformer_gpu_small_dense_native_quant_layer_policy(c);
    BnTransformerGPUComparePolicy compare_policy =
        bn_transformer_gpu_compare_policy();
    BnTransformerGPUMoERouteLayerPolicy moe_route_layer =
        bn_transformer_gpu_moe_route_layer_policy();
    compare_attention_layer = compare_policy.attention_layer;
    compare_attention_pos = compare_policy.attention_pos;
    compare_gqa_layer = compare_policy.gqa_layer;
    compare_gqa_pos = compare_policy.gqa_pos;
    compare_qkv_layer = compare_policy.qkv_layer;
    compare_qkv_pos = compare_policy.qkv_pos;
    compare_ffn_down_layer = compare_policy.ffn_down_layer;
    compare_ffn_down_pos = compare_policy.ffn_down_pos;
    compare_ffn_state_layer = compare_policy.ffn_state_layer;
    compare_ffn_state_pos = compare_policy.ffn_state_pos;
    cpu_fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        cpu_fallback, gpu, c, w);
    BnTransformerGPUDecodeEntryPolicy decode_entry =
        bn_transformer_gpu_decode_entry_policy(
            gpu, c, w, argmax_token != NULL);
    if (decode_entry.block_argmax)
        return NULL;

    // Embed token on CPU, upload to GPU x buffer.
    float emb[dim];
    bn_model_embed_token(m, emb, token);
    if (bn_transformer_gpu_write_x(gpu, emb,
                                   (size_t)dim * sizeof(float)) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "write token embedding failed");

    /* no-op */

    void *output_norm = policy.output_norm;
    BnTransformerGPULogitResources *logit_res = &policy.logits;
    int has_moe = policy.has_moe;

    float norm_eps = bn_transformer_gpu_norm_epsilon(c);

    // Precompute eps as uint32
    uint32_t u_eps;
    { float eps = norm_eps; memcpy(&u_eps, &eps, 4); }

    int max_ops = bn_transformer_gpu_graph_op_capacity(c);

    // Reuse the session-owned GPU IR/lowering storage to avoid per-token malloc.
    BnTransformerGPUDecodeSessionResources decode_session;
    if (bn_transformer_gpu_resolve_session_decode_resources(
            &decode_session, sess, max_ops, 1) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph allocation failed");
    void *command_buffer = decode_session.command_buffer;
    int command_cap = decode_session.command_cap;
    BnTransformerGPULogitsDispatchPolicy logits_dispatch =
        bn_transformer_gpu_logits_dispatch_policy(
            gpu, c, logit_res, argmax_token != NULL, need_logits);
    int gpu_logits_need_cpu = logits_dispatch.needs_cpu_fallback;
    int use_matvec_argmax = logits_dispatch.use_matvec_argmax;
    BnTransformerGPUSmallDenseNativeQuantDecodePolicy small_dense_native_quant_decode =
        bn_transformer_gpu_small_dense_native_quant_decode_policy(gpu, c, &small_dense_native_quant);
    BnTransformerGPULogitsRefinePolicy logits_refine =
        bn_transformer_gpu_logits_refine_policy(
            gpu, c, w, logit_res,
            small_dense_native_quant_decode.small_dense_native_quant_default);
    BnTransformerGPUDecodeCacheabilityPolicy decode_cacheability =
        bn_transformer_gpu_model_decode_cacheability_policy(
            m, emit_logits, argmax_token != NULL,
            gpu_logits_need_cpu, policy.has_moe, &logits_refine, need_logits,
            &cpu_fallback, &compare_policy);
    int cacheable_decode = decode_cacheability.graph_cacheable;
    int cached_n = cacheable_decode ? decode_session.cached_op_count : 0;
    int cached_has_logits =
        cached_n > 0 && decode_session.cached_has_logits;
    BnTransformerGPUCachedDecodePolicy cached_decode =
        bn_transformer_gpu_cached_decode_policy(
            cached_n, argmax_token != NULL, cached_has_logits,
            use_matvec_argmax);
    if (cached_decode.clear_cache) {
        bn_transformer_gpu_clear_session_decode_cache(sess);
        cached_n = 0;
        cached_has_logits = 0;
    }
    if (cached_decode.use_cache && cached_n <= command_cap) {
        if (bn_transformer_gpu_patch_cached_decode_ops(
                command_buffer, cached_n, c, pos) == 0 &&
            bn_transformer_gpu_execute_ops(
                gpu, command_buffer, cached_n,
                need_logits ? BN_GPU_VALUE_LOGITS : -1,
                need_logits ? s->logits : NULL,
                need_logits ? c->vocab_size : 0) == 0) {
            if (argmax_token) {
                int argmax_rc = cached_has_logits
                    ? bn_transformer_gpu_argmax_backend_run(
                          gpu, BN_GPU_VALUE_LOGITS, c->vocab_size,
                          penalty_tokens, n_penalty_tokens, repeat_penalty,
                          argmax_token)
                    : bn_transformer_gpu_matvec_argmax_backend_run(
                          gpu, logit_res->gpu_buf, logit_res->type,
                          logit_res->rows, logit_res->cols, BN_GPU_VALUE_XB,
                          penalty_tokens, n_penalty_tokens, repeat_penalty,
                          argmax_token);
                if (argmax_rc != 0) {
                    bn_transformer_gpu_clear_session_decode_cache(sess);
                    bn_transformer_gpu_emit_context_free(&emit);
                    return NULL;
                }
            }
            bn_transformer_gpu_emit_context_free(&emit);
            return need_logits ? s->logits : s->x;
        }
        bn_transformer_gpu_clear_session_decode_cache(sess);
    }
    if (bn_transformer_gpu_emit_context_init_decode_session(
            &emit, sess, command_buffer, command_cap,
            max_ops * 4, max_ops) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph reserve failed");

    // ---- Initial RMSNorm: x -> xb (using layer 0 attn_norm) ----
    if (bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
            &emit, policy.initial_norm,
            dim, u_eps) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph rmsnorm emit failed");

    /* no-op */

    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        BnTransformerGPULayerResources gpu_layer_res;
        if (bn_transformer_gpu_resolve_model_layer_resources(
                &gpu_layer_res, m, lw, l, output_norm) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu layer resource resolution failed");
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, lw, l, policy.has_tq);
        int is_attn = plan.is_attn;
        BnFFNPlan layer_ffn_plan;
        int layer_ffn_plan_valid = 0;
        BnTransformerGPUDenseFFNResources layer_ffn_res = {0};
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(lw);
        if (!layer_kind.uses_moe) {
            bn_transformer_plan_ffn_resources(
                &layer_ffn_plan, c, lw, gpu,
                &gpu_layer_res.dense_ffn, l, 1);
            layer_ffn_plan_valid = 1;
            layer_ffn_res = gpu_layer_res.dense_ffn;
        }
        BnTransformerGPUSmallDenseNativeQuantLayerUsePolicy small_dense_native_quant_use =
            bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
                gpu, c, &small_dense_native_quant, l,
                small_dense_native_quant_decode.small_dense_native_quant_default,
                small_dense_native_quant_decode.small_dense_native_quant_to_layer);

        // ---- SSM layer ----
        if (!is_attn) {
            BnTransformerGPUSSMFallbackPolicy ssm_fallback =
                bn_transformer_gpu_ssm_fallback_policy(gpu);
            if (ssm_fallback.use_cpu) {
                void *nn = gpu_layer_res.next_norm;
                if (bn_transformer_gpu_fallback_ssm_layer(
                        &emit, gpu, m, sess, lw, l, dim, u_eps, nn) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu ssm cpu fallback failed");
                continue;
            }

            BnTransformerGPUSSMResources ssm_res = gpu_layer_res.ssm;
            bn_transformer_gpu_emit_context_ssm(
                &emit, c, lw, &plan, &ssm_res, dim, u_eps);

            // SSM layer's FFN (dense or MoE) — same as attention layer below
            goto ffn_block;
        }

        // KV cache addressing
        int attn_idx = plan.attn_idx;
        int layer_q_dim = plan.q_dim;
        int layer_head_size = plan.head_size;
        int layer_kv_dim = plan.kv_dim;
        int layer_rope_dims =
            bn_transformer_rope_dims_for_head(c, layer_head_size);
        size_t loff = (size_t)attn_idx * c->seq_len * kv_cache_stride;
        int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
        if (bn_transformer_gpu_cpu_fallback_layer_selected(
                l, cpu_fallback.layer, cpu_fallback.from_layer)) {
            void *next_norm = gpu_layer_res.next_norm;
            if (bn_transformer_gpu_fallback_cpu_layer(
                    &emit, gpu, m, sess, l, pos, cache_pos, layer_rope_dims,
                    rope_cos, rope_sin, dim, u_eps, next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-layer fallback failed");
            continue;
        }

        uint32_t kv_cache_off =
            (uint32_t)(loff + (size_t)cache_pos * kv_cache_stride);
        BnTransformerGPUQKVResources qkv_res = gpu_layer_res.qkv;
        BnTransformerGPUAttentionResources attn_res =
            gpu_layer_res.attention;
        if (!bn_transformer_gpu_layer_projection_resources_available(
                lw, &gpu_layer_res)) {
            void *next_norm = gpu_layer_res.next_norm;
            if (bn_transformer_gpu_fallback_cpu_layer(
                    &emit, gpu, m, sess, l, pos, cache_pos, layer_rope_dims,
                    rope_cos, rope_sin, dim, u_eps, next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu missing-qweight cpu-layer fallback failed");
            continue;
        }
        if (bn_transformer_gpu_cpu_fallback_layer_selected(
                l, cpu_fallback.attn_layer, cpu_fallback.attn_from_layer)) {
            void *ffn_norm = attn_res.ffn_norm;
            if (bn_transformer_gpu_fallback_cpu_attention(
                    &emit, gpu, m, sess, lw, l, pos, cache_pos,
                    layer_rope_dims,
                    rope_cos, rope_sin, dim, u_eps, ffn_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-attention fallback failed");
        } else {
            int compare_attention = compare_attention_layer == l &&
                (compare_attention_pos < 0 || compare_attention_pos == pos);
            int compare_gqa = compare_gqa_layer == l &&
                (compare_gqa_pos < 0 || compare_gqa_pos == pos);
            if (compare_attention || compare_gqa) {
                if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                    bn_transformer_gpu_read_x(gpu, sess->state.x,
                                              (size_t)dim * sizeof(float)) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu attention pre-compare snapshot failed");
            }
            bn_transformer_gpu_emit_context_qkv(
                &emit, c, lw, &plan, &qkv_res, pos, layer_rope_dims,
                kv_cache_off, u_eps,
                small_dense_native_quant_use.use_attention);
            if (!emit_logits && l + 1 == c->n_layers) {
                continue;
            }
            if (compare_qkv_layer == l &&
                (compare_qkv_pos < 0 || compare_qkv_pos == pos)) {
                if (bn_transformer_gpu_debug_compare_qkv(
                        &emit, gpu, m, sess, lw, l, pos, kv_cache_off,
                        dim, layer_q_dim, layer_kv_dim) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu qkv compare failed");
            }
            if (compare_gqa) {
                bn_transformer_gpu_emit_context_attention_gqa(
                    &emit, c, lw, &attn_res, &plan, pos, layer_rope_dims,
                    n_kv, loff,
                    kv_cache_off, has_moe);
                if (bn_transformer_gpu_debug_compare_gqa(
                        &emit, gpu, m, sess, lw, l, pos, cache_pos,
                        layer_rope_dims, rope_cos, rope_sin, dim) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu gqa compare failed");
                bn_transformer_gpu_emit_context_attention_finish(
                    &emit, c, lw, &attn_res, dim, layer_q_dim,
                    layer_head_size, u_eps,
                    small_dense_native_quant_use.use_attention);
            } else {
                bn_transformer_gpu_emit_context_attention(
                    &emit, c, lw, &attn_res, &plan, pos, dim,
                    layer_rope_dims, n_kv, loff,
                    kv_cache_off, has_moe, u_eps,
                    small_dense_native_quant_use.use_attention);
            }
            if (compare_attention) {
                if (bn_transformer_gpu_debug_compare_attention(
                        &emit, gpu, m, sess, lw, l, pos, cache_pos,
                        layer_rope_dims, rope_cos, rope_sin, dim) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu attention compare failed");
            }
        }

        // ---- FFN (MoE or dense) ----
        ffn_block:;
        if (layer_kind.uses_moe) {
            BnTransformerGPUMoEFFNFallbackPolicy moe_ffn_fallback =
                bn_transformer_gpu_moe_ffn_fallback_policy(
                    gpu, c, &lw->moe.expert_map, dim, 1, l,
                    &cpu_fallback);
            if (moe_ffn_fallback.use_cpu) {
                void *moe_next_norm = gpu_layer_res.next_norm;
                if (bn_transformer_gpu_fallback_moe_layer(
                        &emit, gpu, m, sess, lw, l, dim, u_eps,
                        moe_next_norm) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe cpu fallback failed");
                continue;
            }

            BnGPUMoETemporaryBuffers moe_temporaries;
            void *next_norm = gpu_layer_res.next_norm;
            BnGPUMoEResolvedExpert expert_emit[BN_MAX_MOE_K];
            BnGPUMoEResources moe_res;
            BnTransformerGPUMoEDecodeResources moe_decode_res =
                gpu_layer_res.moe_decode;
            BnTransformerGPUMoEDecodeDispatchPolicy moe_dispatch =
                bn_transformer_gpu_moe_decode_dispatch_policy(
                    gpu, c, lw, &moe_route_layer, l, dim,
                    moe_decode_res.router, moe_decode_res.router_diff,
                    moe_decode_res.gate_all, moe_decode_res.up_all,
                    moe_decode_res.down_all);
            if (moe_dispatch.direct_route.enabled) {
                if (gpu_resolve_moe_all_active_two_resources(
                        &moe_res, expert_emit, m, sess, lw, l,
                        moe_dispatch.direct_route.router_diff,
                        &moe_temporaries) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe all-active-two resource resolution failed");
                BnTransformerGPUMoESharedResources moe_shared =
                    gpu_layer_res.moe_shared;
                bn_transformer_gpu_emit_context_moe(
                    &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm,
                    moe_activation.uses_reference_silu);
                if (moe_temporaries.n_buffers > 0) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu execute flush failed");
                    bn_gpu_moe_bridge_release_temporaries(m, &moe_temporaries);
                }
                continue;
            }
            int moe_route_profile = moe_dispatch.route_profile_enabled;
            double moe_prof_t0 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            if (moe_dispatch.requires_session_state && !sess->moe_state)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe session state missing");
            BnTransformerGPUMoEDecodeRoutePolicy moe_route =
                moe_dispatch.decode_route;
            void *moe_router = moe_route.router;
            if (moe_route.gpu_routed_ffn) {
                BnTransformerGPUMoEDebugPolicy moe_debug =
                    bn_transformer_gpu_moe_decode_debug_policy(
                        c, w, l, pos);
                float *moe_cpu_x = NULL;
                float *moe_gpu_x = NULL;
                float *moe_cpu_routed_part = NULL;
                float *moe_cpu_shared_part = NULL;
                float *moe_gpu_routed_part = NULL;
                float *moe_override_x = NULL;
                if (moe_debug.override_cpu_actual) {
                    moe_override_x =
                        (float *)malloc((size_t)dim * sizeof(float));
                    if (!moe_override_x ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_x(
                            gpu, s->x, (size_t)dim * sizeof(float)) != 0 ||
                        bn_transformer_gpu_read_xb(
                            gpu, s->xb, (size_t)dim * sizeof(float)) != 0 ||
                        bn_transformer_gpu_fallback_moe_output_from_state(
                            m, sess, lw, l, dim, moe_override_x) != 0) {
                        free(moe_override_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe cpu override setup failed");
                    }
                }
                if (moe_debug.compare_layer) {
                    moe_cpu_x = (float *)malloc((size_t)dim * sizeof(float));
                    moe_gpu_x = (float *)malloc((size_t)dim * sizeof(float));
                    if (!moe_cpu_x || !moe_gpu_x ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_x(gpu, s->x,
                                                  (size_t)dim * sizeof(float)) != 0 ||
                        bn_transformer_gpu_read_xb(gpu, s->xb,
                                                   (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe compare setup failed");
                    }
                    if (moe_debug.compare_input_norm && lw->norm.ffn_norm) {
                        float *moe_cpu_xb =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (!moe_cpu_xb) {
                            free(moe_cpu_x);
                            free(moe_gpu_x);
                            free(moe_override_x);
                            return bn_transformer_gpu_reject_forward(
                                &emit, "gpu routed moe compare setup failed");
                        }
                        gpu_debug_rmsnorm_scalar_local(
                            moe_cpu_xb, s->x, lw->norm.ffn_norm, dim,
                            norm_eps);
                        gpu_debug_compare_vec_local(
                            "moe_input_norm_compare", l, pos,
                            moe_cpu_xb, s->xb, dim);
                        free(moe_cpu_xb);
                    }
                    if (moe_debug.compare_actual) {
                        if (bn_transformer_gpu_fallback_moe_output_from_state(
                                m, sess, lw, l, dim, moe_cpu_x) != 0) {
                            free(moe_cpu_x);
                            free(moe_gpu_x);
                            free(moe_override_x);
                            return bn_transformer_gpu_reject_forward(
                                &emit, "gpu routed moe compare setup failed");
                        }
                    } else {
                        bn_transformer_gpu_route_model_moe(
                            m, sess->moe_state, s->xb, lw,
                            route_policy.total_experts,
                            route_policy.active_experts,
                            route_policy.normalize_topk,
                            route_policy.expert_weights_scale);
                        if (bn_transformer_gpu_fallback_moe_output(
                                m, sess, lw, dim, s->x, s->xb,
                                moe_cpu_x) != 0) {
                            free(moe_cpu_x);
                            free(moe_gpu_x);
                            free(moe_override_x);
                            return bn_transformer_gpu_reject_forward(
                                &emit, "gpu routed moe compare setup failed");
                        }
                    }
                }
                if (moe_route.cpu_route_resident_ffn) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_xb(gpu, s->xb,
                                                   (size_t)dim * sizeof(float)) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe cpu route input readback failed");
                    bn_transformer_gpu_route_model_moe(
                        m, sess->moe_state, s->xb, lw,
                        route_policy.total_experts,
                        route_policy.active_experts,
                        route_policy.normalize_topk,
                        route_policy.expert_weights_scale);
                    float route_tmp[BN_MAX_MOE_K * 2];
                    int K = route_policy.active_experts;
                    if (K > BN_MAX_MOE_K)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe route K too large");
                    for (int k = 0; k < K; k++) {
                        route_tmp[k] = sess->moe_state->expert_weights[k];
                        route_tmp[K + k] =
                            (float)sess->moe_state->expert_indices[k];
                    }
                    if (bn_transformer_gpu_write_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, route_tmp,
                            (size_t)(2 * K) * sizeof(float)) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe cpu route upload failed");
                } else if (bn_transformer_gpu_emit_context_moe_route_topk(
                               &emit, moe_router, BN_GPU_VALUE_XB,
                               BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2,
                               dim, route_policy.total_experts,
                               route_policy.active_experts,
                               route_policy.expert_weights_scale,
                               moe_route.route_flags) != 0) {
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route emit failed");
                }
                if (moe_debug.compare_route) {
                    float route_tmp[BN_MAX_MOE_K * 2];
                    int K = route_policy.active_experts;
                    if (K > BN_MAX_MOE_K ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, route_tmp,
                            (size_t)(2 * K) * sizeof(float)) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe route compare failed");
                    for (int rk = 0; rk < K; rk++) {
                        fprintf(stderr,
                                "[bn:gpu:debug] moe_route_compare layer=%d pos=%d slot=%d cpu_w=%.9g gpu_w=%.9g cpu_e=%d gpu_e=%d\n",
                                l, pos, rk,
                                sess->moe_state->expert_weights[rk],
                                route_tmp[rk],
                                sess->moe_state->expert_indices[rk],
                                (int)(route_tmp[K + rk] + 0.5f));
                    }
                }
                BnTransformerGPUMoEProjectionPolicy routed_types =
                    bn_transformer_gpu_moe_projection_policy(
                        &lw->moe.expert_map);
                if (!routed_types.valid)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe routed projection types failed");
                if (moe_debug.compare_raw &&
                    moe_decode_res.gate_all && moe_decode_res.up_all &&
                    moe_route.all_active_two_kquant_moe) {
                    int K = route_policy.active_experts;
                    int n_experts = route_policy.total_experts;
                    int moe_hidden = route_policy.expert_hidden_dim;
                    size_t raw_bytes =
                        (size_t)n_experts * (size_t)moe_hidden *
                        sizeof(float);
                    float *cpu_gate = (float *)malloc(raw_bytes);
                    float *cpu_up = (float *)malloc(raw_bytes);
                    float *gpu_gate = (float *)malloc(raw_bytes);
                    float *gpu_up = (float *)malloc(raw_bytes);
                    float route_save[BN_MAX_MOE_K * 2];
                    uint32_t gate_raw_compare_flags =
                        bn_transformer_gpu_moe_expert_projection_matvec_flags(
                            &lw->moe.expert_map, 0, 1);
                    uint32_t up_raw_compare_flags =
                        bn_transformer_gpu_moe_expert_projection_matvec_flags(
                            &lw->moe.expert_map, 1, 1);
                    if (!cpu_gate || !cpu_up || !gpu_gate || !gpu_up ||
                        K > BN_MAX_MOE_K ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, route_save,
                            (size_t)(2 * K) * sizeof(float)) != 0 ||
                        bn_transformer_gpu_fallback_moe_raw_gate_up(
                            m, sess, lw, s->xb, cpu_gate, cpu_up) != 0 ||
                        bn_transformer_gpu_emit_context_matvec_flags(
                            &emit, routed_types.gate_type,
                            moe_decode_res.gate_all, BN_GPU_VALUE_XB,
                            BN_GPU_VALUE_MOE_HB,
                            n_experts * moe_hidden, dim, 0,
                            gate_raw_compare_flags) != 0 ||
                        bn_transformer_gpu_emit_context_matvec_flags(
                            &emit, routed_types.up_type,
                            moe_decode_res.up_all, BN_GPU_VALUE_XB,
                            BN_GPU_VALUE_MOE_HB2,
                            n_experts * moe_hidden, dim, 0,
                            up_raw_compare_flags) != 0 ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB, gpu_gate,
                            raw_bytes) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, gpu_up,
                            raw_bytes) != 0 ||
                        bn_transformer_gpu_write_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, route_save,
                            (size_t)(2 * K) * sizeof(float)) != 0) {
                        free(cpu_gate);
                        free(cpu_up);
                        free(gpu_gate);
                        free(gpu_up);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe raw compare failed");
                    }
                    for (int eidx = 0; eidx < n_experts; eidx++) {
                        char label[64];
                        snprintf(label, sizeof(label),
                                 "moe_raw_gate_compare[%d]", eidx);
                        gpu_debug_compare_vec_local(
                            label, l, pos,
                            cpu_gate + (size_t)eidx * (size_t)moe_hidden,
                            gpu_gate + (size_t)eidx * (size_t)moe_hidden,
                            moe_hidden);
                        snprintf(label, sizeof(label),
                                 "moe_raw_up_compare[%d]", eidx);
                        gpu_debug_compare_vec_local(
                            label, l, pos,
                            cpu_up + (size_t)eidx * (size_t)moe_hidden,
                            gpu_up + (size_t)eidx * (size_t)moe_hidden,
                            moe_hidden);
                    }
                    free(cpu_gate);
                    free(cpu_up);
                    free(gpu_gate);
                    free(gpu_up);
                }
                if (bn_transformer_gpu_emit_context_moe_routed_ffn(
                        &emit, moe_decode_res.gate_all,
                        moe_decode_res.up_all, moe_decode_res.down_all,
                        BN_GPU_VALUE_XB, BN_GPU_VALUE_MOE_HB2,
                        BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_OUT,
                        routed_types.gate_type,
                        routed_types.down_type, dim,
                        route_policy.expert_hidden_dim,
                        route_policy.total_experts,
                        route_policy.active_experts,
                        moe_activation.uses_reference_silu, l) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe routed ffn emit failed");
                if (moe_debug.compare_mid) {
                    int K = route_policy.active_experts;
                    int moe_hidden = route_policy.expert_hidden_dim;
                    size_t mid_bytes =
                        (size_t)K * (size_t)moe_hidden * sizeof(float);
                    float *moe_cpu_mid = (float *)malloc(mid_bytes);
                    float *moe_gpu_mid = (float *)malloc(mid_bytes);
                    if (!moe_cpu_mid || !moe_gpu_mid ||
                        bn_transformer_gpu_fallback_moe_mid(
                            m, sess, lw, s->xb, moe_cpu_mid) != 0 ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB, moe_gpu_mid,
                            mid_bytes) != 0) {
                        free(moe_cpu_mid);
                        free(moe_gpu_mid);
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe mid compare failed");
                    }
                    for (int mk = 0; mk < K; mk++) {
                        char label[64];
                        snprintf(label, sizeof(label), "moe_mid_compare[%d]",
                                 mk);
                        gpu_debug_compare_vec_local(
                            label, l, pos,
                            moe_cpu_mid + (size_t)mk * (size_t)moe_hidden,
                            moe_gpu_mid + (size_t)mk * (size_t)moe_hidden,
                            moe_hidden);
                    }
                    free(moe_cpu_mid);
                    free(moe_gpu_mid);
                }
                if (moe_debug.compare_parts) {
                    moe_cpu_routed_part =
                        (float *)malloc((size_t)dim * sizeof(float));
                    moe_cpu_shared_part =
                        (float *)malloc((size_t)dim * sizeof(float));
                    moe_gpu_routed_part =
                        (float *)malloc((size_t)dim * sizeof(float));
                    if (!moe_cpu_routed_part || !moe_cpu_shared_part ||
                        !moe_gpu_routed_part ||
                        bn_transformer_gpu_fallback_moe_parts(
                            m, sess, lw, dim, s->xb, moe_cpu_routed_part,
                            moe_cpu_shared_part) != 0 ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_OUT, moe_gpu_routed_part,
                            (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_routed_part);
                        free(moe_cpu_shared_part);
                        free(moe_gpu_routed_part);
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe parts compare failed");
                    }
                    gpu_debug_compare_vec_local("moe_routed_part_compare",
                                                l, pos, moe_cpu_routed_part,
                                                moe_gpu_routed_part, dim);
                }
                BnTransformerGPUMoESharedCPUFallbackPolicy
                    shared_cpu_fallback =
                        bn_transformer_gpu_moe_shared_cpu_fallback_policy(
                            c, lw);
                if (shared_cpu_fallback.enabled) {
                    size_t dim_bytes = (size_t)dim * sizeof(float);
                    float *shared_cpu_xb = (float *)malloc(dim_bytes);
                    float *shared_cpu_out = (float *)malloc(dim_bytes);
                    float *shared_gpu_out = (float *)malloc(dim_bytes);
                    if (!shared_cpu_xb || !shared_cpu_out ||
                        !shared_gpu_out ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_xb(
                            gpu, shared_cpu_xb, dim_bytes) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_OUT, shared_gpu_out,
                            dim_bytes) != 0 ||
                        bn_transformer_gpu_fallback_shared_expert_output(
                            m, sess, lw, dim, shared_cpu_xb,
                            shared_cpu_out) != 0) {
                        free(shared_cpu_xb);
                        free(shared_cpu_out);
                        free(shared_gpu_out);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu shared moe cpu fallback failed");
                    }
                    for (int si = 0; si < dim; si++)
                        shared_gpu_out[si] += shared_cpu_out[si];
                    if (bn_transformer_gpu_write_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_OUT,
                            shared_gpu_out, dim_bytes) != 0) {
                        free(shared_cpu_xb);
                        free(shared_cpu_out);
                        free(shared_gpu_out);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu shared moe cpu fallback upload failed");
                    }
                    free(shared_cpu_xb);
                    free(shared_cpu_out);
                    free(shared_gpu_out);
                    bn_transformer_gpu_emit_context_residual_rmsnorm(
                        &emit, BN_GPU_VALUE_X, BN_GPU_VALUE_MOE_OUT,
                        BN_GPU_VALUE_XB, dim, u_eps, next_norm);
                } else if (bn_transformer_gpu_moe_has_loaded_shared_expert(c, lw)) {
                    BnTransformerGPUMoESharedResources moe_shared =
                        gpu_layer_res.moe_shared;
                    BnGPUMoEResources shared_only = {
                        &lw->moe.expert_map, NULL, 1,
                        route_policy.expert_hidden_dim, 1
                    };
                    bn_transformer_gpu_emit_context_moe(
                        &emit, &shared_only, &moe_shared, lw, dim, u_eps,
                        next_norm, moe_activation.uses_reference_silu);
                } else {
                    bn_transformer_gpu_emit_context_residual_rmsnorm(
                        &emit, BN_GPU_VALUE_X, BN_GPU_VALUE_MOE_OUT,
                        BN_GPU_VALUE_XB, dim, u_eps, next_norm);
                }
                if (moe_debug.compare_layer) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_x(gpu, moe_gpu_x,
                                                  (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe compare readback failed");
                    }
                    gpu_debug_compare_vec_local("moe_routed_state_compare",
                                                l, pos, moe_cpu_x, moe_gpu_x,
                                                dim);
                    if (moe_cpu_shared_part && moe_gpu_routed_part) {
                        float *moe_gpu_shared_part =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (moe_gpu_shared_part) {
                            for (int i = 0; i < dim; i++)
                                moe_gpu_shared_part[i] =
                                    moe_gpu_x[i] - s->x[i] -
                                    moe_gpu_routed_part[i];
                            gpu_debug_compare_vec_local(
                                "moe_shared_part_compare", l, pos,
                                moe_cpu_shared_part, moe_gpu_shared_part,
                                dim);
                        }
                        free(moe_gpu_shared_part);
                    }
                    if (moe_debug.compare_shared_mid &&
                        bn_transformer_gpu_moe_has_loaded_shared_expert(c, lw)) {
                        BnTransformerGPUMoESharedExpertShapePolicy shared_shape =
                            bn_transformer_gpu_moe_shared_expert_shape_policy(c);
                        int shared_hidden = shared_shape.hidden_dim;
                        size_t shared_mid_bytes =
                            (size_t)shared_hidden * sizeof(float);
                        float *moe_cpu_shared_mid =
                            (float *)malloc(shared_mid_bytes);
                        float *moe_gpu_shared_mid =
                            (float *)malloc(shared_mid_bytes);
                        if (moe_cpu_shared_mid && moe_gpu_shared_mid &&
                            bn_transformer_gpu_fallback_shared_expert_mid(
                                m, sess, lw, s->xb,
                                moe_cpu_shared_mid) == 0 &&
                            bn_transformer_gpu_read_activation_buf(
                                gpu, BN_GPU_VALUE_HB, moe_gpu_shared_mid,
                                shared_mid_bytes) == 0) {
                            gpu_debug_compare_vec_local(
                                "moe_shared_mid_compare", l, pos,
                                moe_cpu_shared_mid, moe_gpu_shared_mid,
                                shared_hidden);
                        }
                        free(moe_cpu_shared_mid);
                        free(moe_gpu_shared_mid);
                    }
                    if (moe_debug.compare_shared_down &&
                        bn_transformer_gpu_moe_has_loaded_shared_expert(c, lw)) {
                        size_t shared_down_bytes =
                            (size_t)dim * sizeof(float);
                        float *moe_cpu_shared_down =
                            (float *)malloc(shared_down_bytes);
                        float *moe_gpu_shared_down =
                            (float *)malloc(shared_down_bytes);
                        if (moe_cpu_shared_down && moe_gpu_shared_down &&
                            bn_transformer_gpu_fallback_shared_expert_down(
                                m, sess, lw, dim, s->xb,
                                moe_cpu_shared_down) == 0 &&
                            bn_transformer_gpu_read_activation_buf(
                                gpu, BN_GPU_VALUE_XB2, moe_gpu_shared_down,
                                shared_down_bytes) == 0) {
                            gpu_debug_compare_vec_local(
                                "moe_shared_down_compare", l, pos,
                                moe_cpu_shared_down, moe_gpu_shared_down,
                                dim);
                        }
                        free(moe_cpu_shared_down);
                        free(moe_gpu_shared_down);
                    }
                    if (moe_debug.compare_norm) {
                        float *moe_cpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        float *moe_gpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (moe_cpu_norm && moe_gpu_norm &&
                            bn_transformer_gpu_read_xb(
                                gpu, moe_gpu_norm,
                                (size_t)dim * sizeof(float)) == 0) {
                            const float *nw = (l + 1 < c->n_layers)
                                ? w->layers[l + 1].norm.attn_norm
                                : w->output_norm;
                            if (nw) {
                                float ss = 0.0f;
                                for (int i = 0; i < dim; i++)
                                    ss += moe_cpu_x[i] * moe_cpu_x[i];
                                float scale = 1.0f /
                                    sqrtf(ss / (float)dim + norm_eps);
                                for (int i = 0; i < dim; i++)
                                    moe_cpu_norm[i] =
                                        moe_cpu_x[i] * scale * nw[i];
                                gpu_debug_compare_vec_local(
                                    "moe_routed_norm_compare", l, pos,
                                    moe_cpu_norm, moe_gpu_norm, dim);
                            }
                        }
                        free(moe_cpu_norm);
                        free(moe_gpu_norm);
                    }
                    free(moe_cpu_routed_part);
                    free(moe_cpu_shared_part);
                    free(moe_gpu_routed_part);
                    free(moe_cpu_x);
                    free(moe_gpu_x);
                }
                if (moe_debug.override_cpu_actual) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_write_x(
                            gpu, moe_override_x,
                            (size_t)dim * sizeof(float)) != 0 ||
                        bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
                            &emit, next_norm, dim, u_eps) != 0) {
                        free(moe_override_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe cpu override apply failed");
                    }
                }
                free(moe_override_x);
                continue;
            }
            BnTransformerGPUMoEDebugPolicy moe_debug =
                bn_transformer_gpu_moe_debug_policy(
                    0, bn_transformer_gpu_moe_compare_layer_selected(l, pos));
            int did_gpu_route_topk = 0;
            if (moe_route.gpu_route_topk) {
                if (bn_transformer_gpu_emit_context_moe_route_topk(
                        &emit, moe_router, BN_GPU_VALUE_XB,
                        BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2,
                        dim, route_policy.total_experts,
                        route_policy.active_experts,
                        route_policy.expert_weights_scale,
                        moe_route.route_flags) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route emit failed");
                if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route topk failed");
                float route_tmp[BN_MAX_MOE_K * 2];
                int K = route_policy.active_experts;
                if (K > BN_MAX_MOE_K)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route K too large");
                if (bn_transformer_gpu_read_activation_buf(
                        gpu, BN_GPU_VALUE_MOE_HB2, route_tmp,
                        (size_t)(2 * K) * sizeof(float)) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route readback failed");
                for (int k = 0; k < K; k++) {
                    sess->moe_state->expert_weights[k] = route_tmp[k];
                    int eidx = (int)(route_tmp[K + k] + 0.5f);
                    sess->moe_state->expert_indices[k] = eidx;
                }
                if (moe_debug.compare_route) {
                    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                                   (size_t)dim *
                                                   sizeof(float)) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe route compare input failed");
                    float cpu_weights[BN_MAX_MOE_K];
                    int cpu_indices[BN_MAX_MOE_K];
                    bn_transformer_gpu_route_model_moe(
                        m, sess->moe_state, s->xb, lw,
                        route_policy.total_experts,
                        route_policy.active_experts,
                        route_policy.normalize_topk,
                        route_policy.expert_weights_scale);
                    for (int k = 0; k < K; k++) {
                        cpu_weights[k] = sess->moe_state->expert_weights[k];
                        cpu_indices[k] = sess->moe_state->expert_indices[k];
                        sess->moe_state->expert_weights[k] = route_tmp[k];
                        sess->moe_state->expert_indices[k] =
                            (int)(route_tmp[K + k] + 0.5f);
                    }
                    for (int k = 0; k < K; k++) {
                        fprintf(stderr,
                                "[bn:gpu:debug] moe_route_compare layer=%d pos=%d slot=%d cpu_w=%.9g gpu_w=%.9g cpu_e=%d gpu_e=%d\n",
                                l, pos, k, cpu_weights[k], route_tmp[k],
                                cpu_indices[k],
                                (int)(route_tmp[K + k] + 0.5f));
                    }
                }
                did_gpu_route_topk = 1;
            } else if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0) {
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe route input readback failed");
            }
            double moe_prof_t1 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            if (!did_gpu_route_topk &&
                bn_transformer_gpu_read_xb(gpu, s->xb,
                                           (size_t)dim * sizeof(float)) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe route input readback failed");
            double moe_prof_t2 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            if (!did_gpu_route_topk) {
                bn_transformer_gpu_route_model_moe(
                    m, sess->moe_state, s->xb, lw,
                    route_policy.total_experts,
                    route_policy.active_experts,
                    route_policy.normalize_topk,
                    route_policy.expert_weights_scale);
            }
            double moe_prof_t3 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            float *moe_cpu_x = NULL;
            float *moe_gpu_x = NULL;
            if (moe_debug.compare_layer) {
                moe_cpu_x = (float *)malloc((size_t)dim * sizeof(float));
                moe_gpu_x = (float *)malloc((size_t)dim * sizeof(float));
                if (!moe_cpu_x || !moe_gpu_x ||
                    bn_transformer_gpu_read_x(gpu, s->x,
                                              (size_t)dim * sizeof(float)) != 0 ||
                    bn_transformer_gpu_fallback_moe_output(
                        m, sess, lw, dim, s->x, s->xb, moe_cpu_x) != 0) {
                    free(moe_cpu_x);
                    free(moe_gpu_x);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe compare setup failed");
                }
            }
            if (bn_gpu_moe_bridge_resolve_resources(
                    &moe_res, expert_emit, BN_MAX_MOE_K, m, sess, lw, l,
                    &moe_temporaries) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe resource resolution failed");
            double moe_prof_t4 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            gpu_moe_route_profile_add(
                dim, route_policy.total_experts, moe_prof_t1 - moe_prof_t0,
                moe_prof_t2 - moe_prof_t1, moe_prof_t3 - moe_prof_t2,
                moe_prof_t4 - moe_prof_t3);
            BnTransformerGPUMoESharedResources moe_shared =
                gpu_layer_res.moe_shared;
            bn_transformer_gpu_emit_context_moe(
                &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm,
                moe_activation.uses_reference_silu);
            if (moe_temporaries.n_buffers > 0 || moe_debug.compare_layer) {
                if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu execute flush failed");
                if (moe_debug.compare_layer) {
                    if (bn_transformer_gpu_read_x(gpu, moe_gpu_x,
                                                  (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe compare readback failed");
                    }
                    gpu_debug_compare_vec_local("moe_state_compare", l, pos,
                                                moe_cpu_x, moe_gpu_x, dim);
                    if (moe_debug.compare_norm) {
                        float *moe_cpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        float *moe_gpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (moe_cpu_norm && moe_gpu_norm &&
                            bn_transformer_gpu_read_xb(
                                gpu, moe_gpu_norm,
                                (size_t)dim * sizeof(float)) == 0) {
                            const float *nw = (l + 1 < c->n_layers)
                                ? w->layers[l + 1].norm.attn_norm
                                : w->output_norm;
                            if (nw) {
                                float ss = 0.0f;
                                for (int i = 0; i < dim; i++)
                                    ss += moe_cpu_x[i] * moe_cpu_x[i];
                                float scale = 1.0f /
                                    sqrtf(ss / (float)dim + norm_eps);
                                for (int i = 0; i < dim; i++)
                                    moe_cpu_norm[i] =
                                        moe_cpu_x[i] * scale * nw[i];
                                gpu_debug_compare_vec_local(
                                    "moe_norm_compare", l, pos,
                                    moe_cpu_norm, moe_gpu_norm, dim);
                            }
                        }
                        free(moe_cpu_norm);
                        free(moe_gpu_norm);
                    }
                    free(moe_cpu_x);
                    free(moe_gpu_x);
                }
                bn_gpu_moe_bridge_release_temporaries(m, &moe_temporaries);
            }
            continue;  // skip dense FFN below
        }
        void *next_norm = gpu_layer_res.next_norm;
        BnFFNPlan ffn_plan;
        if (layer_ffn_plan_valid)
            ffn_plan = layer_ffn_plan;
        else
            bn_transformer_plan_ffn_resources(
                &ffn_plan, c, lw, gpu, &gpu_layer_res.dense_ffn, l, 1);
        if (bn_transformer_gpu_cpu_fallback_layer_selected(
                l, cpu_fallback.ffn_layer, cpu_fallback.ffn_from_layer)) {
            if (bn_transformer_gpu_fallback_cpu_ffn(
                    &emit, gpu, m, sess, lw, &ffn_plan, dim, u_eps,
                    next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-ffn fallback failed");
            continue;
        }
        BnTransformerGPUDenseFFNResources ffn_res = layer_ffn_res;
        int ffn_down_input_buf = -1;
        int skip_ffn_down = bn_transformer_gpu_cpu_fallback_layer_selected(
            l, -1, cpu_fallback.ffn_down_from_layer);
        int compare_ffn_state = compare_ffn_state_layer == l &&
            (compare_ffn_state_pos < 0 || compare_ffn_state_pos == pos);
        if (compare_ffn_state) {
            if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                bn_transformer_gpu_read_x(gpu, sess->state.x,
                                          (size_t)dim * sizeof(float)) != 0 ||
                bn_transformer_gpu_read_xb(gpu, sess->state.xb,
                                           (size_t)dim * sizeof(float)) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-state pre-compare snapshot failed");
        }
        bn_transformer_gpu_emit_context_dense_ffn(
            &emit, c, lw, &ffn_plan, &ffn_res, dim, u_eps,
            next_norm, skip_ffn_down, &ffn_down_input_buf,
            small_dense_native_quant_use.use_ffn, small_dense_native_quant_use.use_ffn_down);
        if (!skip_ffn_down &&
            compare_ffn_down_layer == l &&
            (compare_ffn_down_pos < 0 || compare_ffn_down_pos == pos)) {
            if (bn_transformer_gpu_debug_compare_ffn_down(
                    &emit, gpu, m, sess, lw, l, pos, ffn_down_input_buf,
                    ffn_plan.hidden_dim, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-down compare failed");
        }
        if (!skip_ffn_down && compare_ffn_state) {
            const float *next_norm_cpu = (l + 1 < c->n_layers)
                ? w->layers[l + 1].norm.attn_norm
                : w->output_norm;
            if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                bn_transformer_gpu_debug_compare_ffn_state(
                    &emit, gpu, m, sess, lw, &ffn_plan, next_norm_cpu,
                    l, pos, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-state compare failed");
        }
        if (skip_ffn_down) {
            if (bn_transformer_gpu_fallback_cpu_ffn_down(
                    &emit, gpu, m, sess, lw, ffn_down_input_buf,
                    ffn_plan.hidden_dim, dim, u_eps, next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-ffn-down fallback failed");
        }
    }

    // ---- Logits matvec: xb -> logits (xb is already normalized) ----
    BnTransformerGPULogitsRefineSnapshotPolicy logits_refine_snapshot =
        bn_transformer_gpu_logits_refine_snapshot_policy(
            need_logits, argmax_token != NULL, &logits_refine);
    int kquant_logits_refine_has_xb_snapshot =
        logits_refine_snapshot.snapshot_satisfies_kquant_refine;
    if (emit_logits && !use_matvec_argmax) {
        if (logits_dispatch.cpu_logits_enabled) {
            if (argmax_token)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu argmax requires gpu logits");
            if (bn_transformer_gpu_fallback_logits(
                    &emit, gpu, m, sess, logit_res, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu logits cpu fallback failed");
            return s->logits;
        }
        if (logits_refine_snapshot.snapshot_before_logits) {
            if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                bn_transformer_gpu_read_xb(gpu, s->xb,
                                           (size_t)dim * sizeof(float)) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu logits pre-refine snapshot failed");
        }
        if (bn_transformer_gpu_emit_context_logits(
                &emit, logit_res->gpu_buf, logit_res->type,
                logit_res->rows, logit_res->cols) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu graph logits emit failed");
    }

    // Safety: verify we didn't overflow the ops array
    if (emit.n + emit.graph->n_ops > max_ops)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu op graph capacity exceeded");

    // Execute final batch (logits + any remaining layer ops).
    if (bn_transformer_gpu_emit_context_lower_pending(&emit) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu final lower failed");
    int final_n = emit.n;
    int rc = bn_transformer_gpu_execute_ops(
        gpu, emit.lowered_ops, emit.n,
        need_logits ? BN_GPU_VALUE_LOGITS : -1,
        need_logits ? s->logits : NULL,
        need_logits ? c->vocab_size : 0);
    if (rc != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu final execute failed");
    if (cacheable_decode && final_n > 0)
        bn_transformer_gpu_store_session_decode_cache(
            sess, final_n,
            emit_logits && !use_matvec_argmax);
    if (argmax_token) {
        if (!use_matvec_argmax &&
            logits_refine.native_quant_captures_xb) {
            int refine_top = logits_refine.native_quant_refine_top;
            if (refine_top > 0 &&
                bn_transformer_gpu_read_activation_buf(
                    gpu, BN_GPU_VALUE_LOGITS, s->logits,
                    (size_t)c->vocab_size * sizeof(float)) == 0 &&
                bn_transformer_gpu_read_xb(gpu, s->xb,
                                           (size_t)dim * sizeof(float)) == 0) {
                bn_transformer_gpu_refine_native_quant_logits_top(
                    s->logits, c->vocab_size, logit_res->cpu_weight,
                    s->xb, s->x_q, refine_top);
                int best = 0;
                float best_v = -INFINITY;
                for (int i = 0; i < c->vocab_size; i++) {
                    float v = s->logits[i];
                    if (repeat_penalty != 1.0f && penalty_tokens &&
                        n_penalty_tokens > 0) {
                        for (int j = 0; j < n_penalty_tokens; j++) {
                            if (penalty_tokens[j] == i) {
                                v = v > 0.0f ? v / repeat_penalty
                                             : v * repeat_penalty;
                                break;
                            }
                        }
                    }
                    if (v > best_v) {
                        best_v = v;
                        best = i;
                    }
                }
                *argmax_token = best;
                bn_transformer_gpu_emit_context_free(&emit);
                return s->x;
            }
        }
        int argmax_rc = use_matvec_argmax
            ? bn_transformer_gpu_matvec_argmax_backend_run(
                  gpu, logit_res->gpu_buf, logit_res->type,
                  logit_res->rows, logit_res->cols, BN_GPU_VALUE_XB,
                  penalty_tokens, n_penalty_tokens, repeat_penalty,
                  argmax_token)
            : bn_transformer_gpu_argmax_backend_run(
                  gpu, BN_GPU_VALUE_LOGITS, c->vocab_size,
                  penalty_tokens, n_penalty_tokens, repeat_penalty,
                  argmax_token);
        if (argmax_rc != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu argmax failed");
        if (bn_transformer_gpu_debug_argmax_compare_enabled() &&
            c->vocab_size > 0) {
            float *dbg_logits =
                (float *)malloc((size_t)c->vocab_size * sizeof(float));
            if (dbg_logits &&
                bn_transformer_gpu_read_activation_buf(
                    gpu, BN_GPU_VALUE_LOGITS, dbg_logits,
                    (size_t)c->vocab_size * sizeof(float)) == 0) {
                int cpu_argmax = 0;
                float cpu_best = -INFINITY;
                for (int i = 0; i < c->vocab_size; i++) {
                    float v = dbg_logits[i];
                    if (repeat_penalty != 1.0f && penalty_tokens &&
                        n_penalty_tokens > 0) {
                        for (int j = 0; j < n_penalty_tokens; j++) {
                            if (penalty_tokens[j] == i) {
                                v = v > 0.0f ? v / repeat_penalty
                                             : v * repeat_penalty;
                                break;
                            }
                        }
                    }
                    if (v > cpu_best) {
                        cpu_best = v;
                        cpu_argmax = i;
                    }
                }
                fprintf(stderr,
                        "[bn:gpu:argmax:cmp] gpu=%d cpu=%d cpu_logit=%.6g\n",
                        *argmax_token, cpu_argmax, cpu_best);
            }
            free(dbg_logits);
        }
        bn_transformer_gpu_emit_context_free(&emit);
        return s->x;
    }
    if (!need_logits) {
        bn_transformer_gpu_emit_context_free(&emit);
        return s->x;
    }
    if (logits_refine.kquant_captures_xb) {
        int refine_top = logits_refine.kquant_refine_top;
        int has_xb = kquant_logits_refine_has_xb_snapshot;
        if (!has_xb && refine_top > 0 &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0)
            has_xb = 1;
        if (refine_top > 0 && has_xb) {
            bn_transformer_gpu_refine_kquant_logits_top(
                s->logits, c->vocab_size, logit_res->cpu_weight,
                s->xb, s->x_q, refine_top);
        }
    }
    if (logits_refine.native_quant_captures_xb) {
        int refine_top = logits_refine.native_quant_refine_top;
        if (refine_top > 0 &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0) {
            bn_transformer_gpu_refine_native_quant_logits_top(
                s->logits, c->vocab_size, logit_res->cpu_weight,
                s->xb, s->x_q, refine_top);
        }
    }
    if (bn_transformer_gpu_compare_logits_enabled()) {
        float *cpu_logits = (float *)malloc((size_t)c->vocab_size *
                                            sizeof(float));
        if (cpu_logits &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0) {
            gpu_cpu_quant_matvec(m, cpu_logits, logit_res->cpu_weight, s->xb, s->x_q);
            double sum_abs = 0.0;
            double sum_sq = 0.0;
            float max_abs = 0.0f;
            int max_i = 0;
            for (int i = 0; i < c->vocab_size; i++) {
                float diff = fabsf(s->logits[i] - cpu_logits[i]);
                sum_abs += (double)diff;
                sum_sq += (double)diff * (double)diff;
                if (diff > max_abs) {
                    max_abs = diff;
                    max_i = i;
                }
            }
            fprintf(stderr,
                    "[bn:gpu:debug] logits_compare pos=%d max_abs=%.9g "
                    "max_i=%d cpu=%.9g gpu=%.9g mean_abs=%.9g rms=%.9g\n",
                    pos, max_abs, max_i, cpu_logits[max_i],
                    s->logits[max_i], sum_abs / (double)c->vocab_size,
                    sqrt(sum_sq / (double)c->vocab_size));
        }
        free(cpu_logits);
    }
    bn_transformer_gpu_emit_context_free(&emit);
    #undef GPU_LEGACY_OPS
    return s->logits;
}

float *bn_transformer_gpu_forward(BnModel *m, BnSession *sess,
                                  int token, int pos) {
    return bn_transformer_gpu_forward_impl(m, sess, token, pos, 1,
                                           NULL, NULL, 0, 1.0f);
}

float *bn_transformer_gpu_forward_no_logits(BnModel *m, BnSession *sess,
                                            int token, int pos) {
    return bn_transformer_gpu_forward_impl(m, sess, token, pos, 0,
                                           NULL, NULL, 0, 1.0f);
}

int bn_transformer_gpu_forward_argmax(BnModel *m, BnSession *sess,
                                      int token, int pos,
                                      const int *penalty_tokens,
                                      int n_penalty_tokens,
                                      float repeat_penalty,
                                      int *out_token) {
    if (!out_token) return -1;
    float *state = bn_transformer_gpu_forward_impl(
        m, sess, token, pos, 0, out_token, penalty_tokens,
        n_penalty_tokens, repeat_penalty);
    return state ? 0 : -1;
}
