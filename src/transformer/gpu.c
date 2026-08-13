#include "gpu_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_logits_internal.h"
#include <math.h>
#include <string.h>

// GPU-resident forward pass. Host access is owned by explicit fallback and
// diagnostic helpers; this function only sequences planned operations.
static float *bn_transformer_gpu_forward_impl(BnModel *m, BnSession *sess,
                                              int token, int pos,
                                              int need_logits,
                                              int *argmax_token,
                                              const int *penalty_tokens,
                                              int n_penalty_tokens,
                                              float repeat_penalty) {
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
    emit.gpu = gpu;

    int dim = c->dim;
    int kv_cache_stride = c->kv_dim;
    BnTransformerGPUMoEExecutionPolicy route_policy =
        bn_transformer_gpu_moe_execution_policy(c);
    BnTransformerGPUMoEActivationPolicy moe_activation =
        bn_transformer_gpu_moe_activation_policy(c);
    int max_head_size = bn_transformer_attention_head_size(c, NULL);
    int max_rope_dims = bn_transformer_rope_dims_for_head(c, max_head_size);
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerShapePlan shape;
        bn_transformer_plan_layer_shape(&shape, c, &w->layers[l], l,
                                        policy.has_tq);
        if (!shape.is_attn)
            continue;
        int layer_rope_dims =
            bn_transformer_rope_dims_for_head(c, shape.head_size);
        if (shape.head_size > max_head_size)
            max_head_size = shape.head_size;
        if (layer_rope_dims > max_rope_dims)
            max_rope_dims = layer_rope_dims;
    }
    int half_rope = max_rope_dims / 2;
    float rope_cos[half_rope], rope_sin[half_rope];
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
    int compare_ssm_layer = -1;
    int compare_ssm_pos = -1;
    BnTransformerGPUCPUFallbackPolicy cpu_fallback =
        bn_transformer_gpu_cpu_fallback_policy(gpu);
    BnTransformerGPUSmallDenseNativeQuantLayerPolicy small_dense_native_quant =
        bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
            gpu, c);
    BnTransformerGPUComparePolicy compare_policy =
        bn_transformer_gpu_compare_policy(gpu);
    BnTransformerGPUMoERouteLayerPolicy moe_route_layer =
        bn_transformer_gpu_moe_route_layer_policy(gpu);
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
    compare_ssm_layer = compare_policy.ssm_layer;
    compare_ssm_pos = compare_policy.ssm_pos;
    cpu_fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        cpu_fallback, gpu, c, w);
    if (cpu_fallback.attn_layer < 0 && cpu_fallback.attn_from_layer < 0 &&
        bn_transformer_gpu_reference_attention_no_logits_cpu_fallback_enabled(
            gpu, c, emit_logits))
        cpu_fallback.attn_from_layer = 0;
    BnTransformerGPUDecodeEntryPolicy decode_entry =
        bn_transformer_gpu_decode_entry_policy(
            gpu, c, w, argmax_token != NULL);
    if (decode_entry.block_argmax || decode_entry.block_forward)
        return NULL;

    if (bn_transformer_gpu_stage_token_input(gpu, m, sess, token) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "write token embedding failed");

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
    emit.gpu = gpu;

    // ---- Initial RMSNorm: x -> xb (using layer 0 attn_norm) ----
    if (bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
            &emit, policy.initial_norm, dim, u_eps) != 0) {
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph rmsnorm emit failed");
    }

    for (int l = 0; l < c->n_layers; l++) {
        if (bn_transformer_gpu_debug_dump_layer_input(
                bn_model_cpu_runtime_policy(m), &emit,
                gpu, l, pos, dim) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu layer-input dump failed");
        BnLayerWeights *lw = &w->layers[l];
        emit.rope_freq_offset = l * (max_head_size / 2);
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
        bn_transformer_plan_ffn_resources(
            &layer_ffn_plan, c, lw, gpu,
            &gpu_layer_res.dense_ffn, l, 1);
        layer_ffn_plan_valid = 1;
        layer_ffn_res = gpu_layer_res.dense_ffn;
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
                int layer_end = l + 1;
                while (!layer_kind.uses_moe && layer_end < c->n_layers) {
                    BnLayerShapePlan next_shape;
                    bn_transformer_plan_layer_shape(
                        &next_shape, c, &w->layers[layer_end], layer_end,
                        policy.has_tq);
                    BnTransformerGPULayerKindPolicy next_kind =
                        bn_transformer_gpu_layer_kind_policy(
                            &w->layers[layer_end]);
                    if (next_shape.is_attn || next_kind.uses_moe)
                        break;
                    layer_end++;
                }
                BnTransformerGPULayerResources final_layer_res;
                if (bn_transformer_gpu_resolve_model_layer_resources(
                        &final_layer_res, m, &w->layers[layer_end - 1],
                        layer_end - 1, output_norm) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu ssm fallback range resource resolution failed");
                void *nn = final_layer_res.next_norm;
                if (bn_transformer_gpu_fallback_ssm_layers(
                        &emit, gpu, m, sess, l, layer_end, dim, u_eps, nn) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu ssm cpu fallback failed");
                l = layer_end - 1;
                continue;
            }

            int compare_ssm = compare_ssm_layer == l &&
                (compare_ssm_pos < 0 || compare_ssm_pos == pos);
            if (compare_ssm &&
                bn_transformer_gpu_debug_snapshot_ssm_state(
                    &emit, gpu, m, sess, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ssm pre-compare snapshot failed");
            BnTransformerGPUSSMResources ssm_res = gpu_layer_res.ssm;
            bn_transformer_gpu_emit_context_ssm(
                &emit, c, lw, &plan, &ssm_res, dim, u_eps,
                small_dense_native_quant_use.use_layer, compare_ssm);
            if (compare_ssm &&
                bn_transformer_gpu_debug_compare_ssm(
                    &emit, gpu, m, sess, lw, &ssm_res,
                    l, pos, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ssm compare failed");

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
        int layer_half_rope = layer_rope_dims / 2;
        float layer_theta =
            bn_transformer_rope_theta_for_head(c, layer_head_size);
        for (int i = 0; i < layer_half_rope; i++) {
            float freq = 1.0f /
                powf(layer_theta,
                     (float)(2 * i) / (float)layer_rope_dims);
            if (bn_transformer_uses_per_layer_embedding(c) &&
                bn_transformer_rope_uses_base_frequency(
                    c, layer_head_size) &&
                w->rope_freqs) {
                if (bn_transformer_divides_rope_freqs(c, l))
                    freq /= w->rope_freqs[i];
                else
                    freq *= w->rope_freqs[i];
            }
            float angle = (float)pos * freq;
            rope_cos[i] = cosf(angle);
            rope_sin[i] = sinf(angle);
        }
        size_t loff = (size_t)attn_idx * c->seq_len * kv_cache_stride;
        int kv_read_idx =
            bn_transformer_attention_kv_read_index(c, lw, l);
        size_t read_loff =
            (size_t)kv_read_idx * c->seq_len * kv_cache_stride;
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
        int cpu_attention_selected =
            bn_transformer_gpu_cpu_fallback_layer_selected(
                l, cpu_fallback.attn_layer, cpu_fallback.attn_from_layer);
        int cpu_ffn_selected =
            bn_transformer_gpu_cpu_fallback_layer_selected(
                l, cpu_fallback.ffn_layer, cpu_fallback.ffn_from_layer);
        if (!cpu_ffn_selected &&
            bn_transformer_gpu_dense_residual_moe_requires_cpu_ffn(
                gpu, c, &lw->moe.expert_map))
            cpu_ffn_selected = 1;
        if (cpu_attention_selected && cpu_ffn_selected) {
            if (bn_transformer_gpu_fallback_cpu_layer(
                    &emit, gpu, m, sess, l, pos, cache_pos, layer_rope_dims,
                    rope_cos, rope_sin, dim, u_eps,
                    gpu_layer_res.next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu composed cpu-layer fallback failed");
            continue;
        }
        if (cpu_attention_selected) {
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
                if (bn_transformer_gpu_debug_snapshot_attention_state(
                        &emit, gpu, sess, dim) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu attention pre-compare snapshot failed");
            }
            bn_transformer_gpu_emit_context_qkv(
                &emit, c, lw, &plan, &qkv_res, pos, layer_rope_dims,
                kv_cache_off, u_eps,
                small_dense_native_quant_use.use_attention);
            int reference_uses_float_kquant =
                !small_dense_native_quant_use.use_attention;
            if (!emit_logits && l + 1 == c->n_layers) {
                continue;
            }
            if (compare_qkv_layer == l &&
                (compare_qkv_pos < 0 || compare_qkv_pos == pos)) {
                if (bn_transformer_gpu_debug_compare_qkv(
                        &emit, gpu, m, sess, lw, l, pos, kv_cache_off,
                        dim, layer_q_dim, layer_kv_dim,
                        qkv_res.k_bias != NULL ||
                            bn_transformer_kv_requires_gpu_cache_write_staging(c),
                        reference_uses_float_kquant) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu qkv compare failed");
            }
            if (compare_gqa) {
                bn_transformer_gpu_emit_context_attention_gqa(
                    &emit, c, lw, &attn_res, &plan, pos, layer_rope_dims,
                    n_kv, read_loff,
                    kv_cache_off, kv_cache_stride, has_moe);
                if (bn_transformer_gpu_debug_compare_gqa(
                        &emit, gpu, m, sess, lw, l, pos, cache_pos,
                        layer_rope_dims, rope_cos, rope_sin, dim,
                        reference_uses_float_kquant) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu gqa compare failed");
                bn_transformer_gpu_emit_context_attention_finish(
                    &emit, c, lw, &attn_res, dim, layer_q_dim,
                    layer_head_size, u_eps,
                    bn_transformer_attention_uses_post_norm_layer(c, lw),
                    small_dense_native_quant_use.use_attention);
            } else {
                bn_transformer_gpu_emit_context_attention(
                    &emit, c, lw, &attn_res, &plan, pos, dim,
                    layer_rope_dims, n_kv, read_loff,
                    kv_cache_off, kv_cache_stride, has_moe, u_eps,
                    small_dense_native_quant_use.use_attention);
            }
            if (compare_attention) {
                if (bn_transformer_gpu_debug_compare_attention(
                        &emit, gpu, m, sess, lw, l, pos, cache_pos,
                        layer_rope_dims, rope_cos, rope_sin, dim,
                        reference_uses_float_kquant) != 0)
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
            moe_dispatch.decode_route.router_scale =
                moe_decode_res.router_scale;
            moe_dispatch.decode_route.expert_down_scale =
                moe_decode_res.expert_down_scale;
            if (moe_dispatch.direct_route.enabled) {
                if (bn_transformer_gpu_resolve_all_active_two_moe_resources(
                        &moe_res, expert_emit, m, sess, lw, l,
                        moe_dispatch.direct_route.router_diff,
                        &moe_temporaries) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe all-active-two resource resolution failed");
                BnTransformerGPUMoESharedResources moe_shared =
                    gpu_layer_res.moe_shared;
                bn_transformer_gpu_emit_context_moe(
                    &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm,
                    moe_activation.uses_reference_silu,
                    moe_activation.uses_reference_ffn_activation);
                if (moe_temporaries.n_buffers > 0) {
                    if (bn_transformer_gpu_flush_and_release_moe_temporaries(
                            &emit, gpu, m, &moe_temporaries) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu execute flush failed");
                }
                continue;
            }
            int moe_route_profile = moe_dispatch.route_profile_enabled;
            if (moe_dispatch.requires_session_state && !sess->moe_state)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe session state missing");
            BnTransformerGPUMoEDecodeRoutePolicy moe_route =
                moe_dispatch.decode_route;
            if (moe_route.gpu_routed_ffn) {
                BnTransformerGPUMoEDebugPolicy moe_debug =
                    bn_transformer_gpu_moe_decode_debug_policy(
                        gpu, c, w, l, pos);
                BnTransformerGPURoutedMoEDebugState moe_debug_state;
                int moe_debug_setup_rc =
                    bn_transformer_gpu_prepare_routed_moe_debug_state(
                        &moe_debug_state, &emit, gpu, m, sess, lw,
                        &route_policy, &moe_debug, l, pos, dim, norm_eps);
                if (moe_debug_setup_rc != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, moe_debug_setup_rc == -1
                            ? "gpu routed moe cpu override setup failed"
                            : "gpu routed moe compare setup failed");
                const char *routed_route_reason = NULL;
                if (bn_transformer_gpu_prepare_routed_moe_route(
                        &emit, gpu, m, sess, lw, &route_policy,
                        &moe_route, &moe_debug, l, pos, dim,
                        &routed_route_reason) != 0) {
                    bn_transformer_gpu_discard_routed_moe_debug_state(
                        &moe_debug_state);
                    return bn_transformer_gpu_reject_forward(
                        &emit, routed_route_reason);
                }
                BnTransformerGPUMoEProjectionPolicy routed_types =
                    bn_transformer_gpu_moe_projection_policy(
                        &lw->moe.expert_map);
                if (!routed_types.valid) {
                    bn_transformer_gpu_discard_routed_moe_debug_state(
                        &moe_debug_state);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe routed projection types failed");
                }
                if (bn_transformer_gpu_debug_compare_routed_moe_raw(
                        &emit, gpu, m, sess, lw, &moe_decode_res,
                        &route_policy, &moe_route, &routed_types,
                        &moe_debug, l, pos, dim) != 0) {
                    bn_transformer_gpu_discard_routed_moe_debug_state(
                        &moe_debug_state);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu routed moe raw compare failed");
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
                        moe_activation.activation,
                        moe_activation.uses_reference_silu,
                        lw->moe.expert_map.gate_stride
                            ? lw->moe.expert_map.gate_stride
                            : lw->moe.expert_map.expert_gate_bytes,
                        lw->moe.expert_map.down_stride
                            ? lw->moe.expert_map.down_stride
                            : lw->moe.expert_map.expert_down_bytes,
                        l) != 0) {
                    bn_transformer_gpu_discard_routed_moe_debug_state(
                        &moe_debug_state);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe routed ffn emit failed");
                }
                if (bn_transformer_gpu_debug_compare_routed_moe_mid(
                        &emit, gpu, m, sess, lw, &route_policy,
                        &moe_debug, l, pos) != 0) {
                    bn_transformer_gpu_discard_routed_moe_debug_state(
                        &moe_debug_state);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu routed moe mid compare failed");
                }
                BnTransformerGPUMoEPartsComparison moe_parts_comparison;
                if (bn_transformer_gpu_prepare_routed_moe_parts_comparison(
                        &moe_parts_comparison, &emit, gpu, m, sess, lw,
                        &moe_debug, sess->state.xb, l, pos, dim) != 0) {
                    bn_transformer_gpu_discard_routed_moe_debug_state(
                        &moe_debug_state);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu routed moe parts compare failed");
                }
                BnTransformerGPUMoESharedCPUFallbackPolicy
                    shared_cpu_fallback =
                        bn_transformer_gpu_moe_shared_cpu_fallback_policy(
                            gpu, c, lw);
                if (shared_cpu_fallback.enabled) {
                    int shared_fallback_rc =
                        bn_transformer_gpu_fallback_shared_expert_residual(
                            &emit, gpu, m, sess, lw, dim);
                    if (shared_fallback_rc != 0) {
                        bn_transformer_gpu_discard_routed_moe_parts_comparison(
                            &moe_parts_comparison);
                        bn_transformer_gpu_discard_routed_moe_debug_state(
                            &moe_debug_state);
                        return bn_transformer_gpu_reject_forward(
                            &emit, shared_fallback_rc == -2
                                ? "gpu shared moe cpu fallback upload failed"
                                : "gpu shared moe cpu fallback failed");
                    }
                    if (!moe_activation.uses_dense_residual_branch)
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
                    if (moe_activation.uses_dense_residual_branch)
                        bn_transformer_gpu_emit_context_moe_routed(
                            &emit, &shared_only, &moe_shared, lw, dim,
                            moe_activation.uses_reference_silu,
                            moe_activation.uses_reference_ffn_activation);
                    else
                        bn_transformer_gpu_emit_context_moe(
                            &emit, &shared_only, &moe_shared, lw, dim, u_eps,
                            next_norm, moe_activation.uses_reference_silu,
                            moe_activation.uses_reference_ffn_activation);
                } else if (!moe_activation.uses_dense_residual_branch) {
                    bn_transformer_gpu_emit_context_residual_rmsnorm(
                        &emit, BN_GPU_VALUE_X, BN_GPU_VALUE_MOE_OUT,
                        BN_GPU_VALUE_XB, dim, u_eps, next_norm);
                }
                if (moe_activation.uses_dense_residual_branch) {
                    int dense_residual_rc =
                        bn_gpu_backend_has_cap(
                            gpu,
                            BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32)
                        ? bn_transformer_gpu_emit_context_dense_residual_moe(
                            &emit, c, lw, &layer_ffn_plan, &layer_ffn_res,
                            dim, u_eps, next_norm,
                            small_dense_native_quant_use.use_ffn,
                            small_dense_native_quant_use.use_ffn_down)
                        : bn_transformer_gpu_fallback_moe_dense_residual_branch(
                            &emit, gpu, m, sess, lw, dim);
                    if (dense_residual_rc != 0) {
                        bn_transformer_gpu_discard_routed_moe_parts_comparison(
                            &moe_parts_comparison);
                        bn_transformer_gpu_discard_routed_moe_debug_state(
                            &moe_debug_state);
                        return bn_transformer_gpu_reject_forward(
                            &emit,
                            dense_residual_rc == -2
                                ? "gpu direct routed moe dense residual down missing"
                            : dense_residual_rc == -3
                                ? "gpu direct routed moe dense residual post-norm-1 missing"
                            : dense_residual_rc == -4
                                ? "gpu direct routed moe dense residual post-norm-2 missing"
                            : dense_residual_rc == -5
                                ? "gpu direct routed moe dense residual final post-norm missing"
                                : "gpu direct routed moe dense residual graph failed");
                    }
                    if (!bn_gpu_backend_has_cap(
                            gpu,
                            BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32))
                        bn_transformer_gpu_emit_context_moe_finish(
                            &emit, dim, u_eps, next_norm);
                }
                int moe_debug_complete_rc =
                    bn_transformer_gpu_complete_routed_moe_debug_state(
                        &moe_debug_state, &moe_parts_comparison,
                        &emit, gpu, m, sess, lw, &moe_debug, next_norm,
                        l, pos, dim, u_eps, norm_eps);
                if (moe_debug_complete_rc != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, moe_debug_complete_rc == -1
                            ? "gpu routed moe compare readback failed"
                            : "gpu routed moe cpu override apply failed");
                int routed_per_dim =
                    bn_transformer_per_layer_embedding_dim(c);
                if (routed_per_dim > 0 &&
                    bn_transformer_gpu_emit_context_per_layer_input(
                        &emit, lw, &gpu_layer_res.per_layer_input, l, dim,
                        routed_per_dim, u_eps, next_norm,
                        layer_ffn_plan.use_layer_output_scale,
                        layer_ffn_plan.use_layer_output_scale
                            ? lw->norm.layer_output_scale[0] : 1.0f,
                        small_dense_native_quant_use.use_ffn) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit,
                        "gpu routed moe per-layer input adapter failed");
                if (routed_per_dim <= 0 &&
                    layer_ffn_plan.use_layer_output_scale &&
                    bn_transformer_gpu_emit_context_scale(
                        &emit, BN_GPU_VALUE_X, dim,
                        lw->norm.layer_output_scale[0]) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu routed moe layer output scale failed");
                continue;
            }
            BnTransformerGPUMoEDebugPolicy moe_debug =
                bn_transformer_gpu_moe_debug_policy(
                    gpu, 0,
                    bn_transformer_gpu_moe_compare_layer_selected(
                        gpu, l, pos));
            BnTransformerGPUMoERouteResolution route_resolution;
            const char *route_reason = NULL;
            if (bn_transformer_gpu_resolve_moe_route(
                    &route_resolution, &emit, gpu, m, sess, lw,
                    &route_policy, &moe_route, &moe_debug,
                    l, pos, dim, moe_route_profile, &route_reason) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, route_reason);
            BnTransformerGPUMoELayerComparison moe_comparison;
            if (bn_transformer_gpu_prepare_moe_layer_comparison(
                    &moe_comparison, gpu, m, sess, lw,
                    &moe_debug, l, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe compare setup failed");
            if (bn_transformer_gpu_resolve_profiled_routed_moe_resources(
                    &moe_res, expert_emit, m, sess, lw, l,
                    &moe_temporaries, moe_route_profile, dim,
                    route_policy.total_experts,
                    route_resolution.flush_ms, route_resolution.read_ms,
                    route_resolution.route_ms) != 0) {
                bn_transformer_gpu_discard_moe_layer_comparison(
                    &moe_comparison);
                bn_transformer_gpu_release_moe_temporaries(
                    m, &moe_temporaries);
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe resource resolution failed");
            }
            if (bn_transformer_gpu_debug_compare_cached_moe_gate_up(
                    gpu, m, sess, lw, &moe_res, &moe_debug,
                    l, pos, dim) != 0) {
                bn_transformer_gpu_discard_moe_layer_comparison(
                    &moe_comparison);
                bn_transformer_gpu_release_moe_temporaries(
                    m, &moe_temporaries);
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cached moe gate/up compare failed");
            }
            BnTransformerGPUMoESharedResources moe_shared =
                gpu_layer_res.moe_shared;
            BnTransformerGPUMoESharedCPUFallbackPolicy shared_cpu_fallback =
                bn_transformer_gpu_moe_shared_cpu_fallback_policy(
                    gpu, c, lw);
            int shared_gate_needs_cpu_fallback =
                bn_transformer_gpu_shared_expert_gate_available(
                    lw, &moe_shared) &&
                 (shared_cpu_fallback.enabled ||
                  !bn_transformer_gpu_weighted_add_sigmoid_supported(gpu));
            if (moe_activation.uses_dense_residual_branch &&
                !moe_debug.compare_parts) {
                bn_transformer_gpu_emit_context_moe_routed(
                    &emit, &moe_res, &moe_shared, lw, dim,
                    moe_activation.uses_reference_silu,
                    moe_activation.uses_reference_ffn_activation);
                int dense_residual_rc =
                    bn_gpu_backend_has_cap(
                        gpu, BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32)
                    ? bn_transformer_gpu_emit_context_dense_residual_moe(
                        &emit, c, lw, &layer_ffn_plan, &layer_ffn_res,
                        dim, u_eps, next_norm,
                        small_dense_native_quant_use.use_ffn,
                        small_dense_native_quant_use.use_ffn_down)
                    : bn_transformer_gpu_fallback_moe_dense_residual_branch(
                        &emit, gpu, m, sess, lw, dim);
                if (dense_residual_rc != 0) {
                    bn_transformer_gpu_discard_moe_layer_comparison(
                        &moe_comparison);
                    bn_transformer_gpu_release_moe_temporaries(
                        m, &moe_temporaries);
                    return bn_transformer_gpu_reject_forward(
                        &emit, dense_residual_rc == -2
                            ? "gpu moe dense residual down missing"
                        : dense_residual_rc == -3
                            ? "gpu moe dense residual post-norm-1 missing"
                        : dense_residual_rc == -4
                            ? "gpu moe dense residual post-norm-2 missing"
                        : dense_residual_rc == -5
                            ? "gpu moe dense residual final post-norm missing"
                            : "gpu moe dense residual graph failed");
                }
                if (!bn_gpu_backend_has_cap(
                        gpu, BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32))
                    bn_transformer_gpu_emit_context_moe_finish(
                        &emit, dim, u_eps, next_norm);
            } else if (moe_debug.compare_parts) {
                for (int k = 0; k < moe_res.n_experts; k++) {
                    BnGPUMoEResources one = moe_res;
                    one.experts = &moe_res.experts[k];
                    one.n_experts = 1;
                    one.preserve_output = k > 0;
                    bn_transformer_gpu_emit_context_moe_routed(
                        &emit, &one, &moe_shared, lw, dim,
                        moe_activation.uses_reference_silu,
                        moe_activation.uses_reference_ffn_activation);
                    if (bn_transformer_gpu_debug_compare_cached_moe_expert(
                            &emit, gpu, m, sess, lw,
                            moe_comparison.input_state,
                            k, l, pos, dim) != 0) {
                        bn_transformer_gpu_discard_moe_layer_comparison(
                            &moe_comparison);
                        bn_transformer_gpu_release_moe_temporaries(
                            m, &moe_temporaries);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu cached moe expert compare failed");
                    }
                }
                BnTransformerGPUMoEPartsComparison parts_comparison;
                if (bn_transformer_gpu_prepare_routed_moe_parts_comparison(
                        &parts_comparison, &emit, gpu, m, sess, lw,
                        &moe_debug, moe_comparison.input_state,
                        l, pos, dim) != 0) {
                    bn_transformer_gpu_discard_moe_layer_comparison(
                        &moe_comparison);
                    bn_transformer_gpu_release_moe_temporaries(
                        m, &moe_temporaries);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu cached moe parts compare failed");
                }
                bn_transformer_gpu_discard_routed_moe_parts_comparison(
                    &parts_comparison);
                if (bn_transformer_gpu_moe_has_loaded_shared_expert(c, lw)) {
                    int shared_fallback_rc =
                        bn_transformer_gpu_fallback_shared_expert_residual_from_input(
                            &emit, gpu, m, sess, lw,
                            moe_comparison.input_state, dim);
                    if (shared_fallback_rc != 0) {
                        bn_transformer_gpu_discard_moe_layer_comparison(
                            &moe_comparison);
                        bn_transformer_gpu_release_moe_temporaries(
                            m, &moe_temporaries);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu shared moe compare fallback failed");
                    }
                }
                bn_transformer_gpu_emit_context_moe_finish(
                    &emit, dim, u_eps, next_norm);
            } else if (shared_gate_needs_cpu_fallback) {
                bn_transformer_gpu_emit_context_moe_routed(
                    &emit, &moe_res, &moe_shared, lw, dim,
                    moe_activation.uses_reference_silu,
                    moe_activation.uses_reference_ffn_activation);
                BnTransformerGPUMoEPartsComparison parts_comparison;
                if (bn_transformer_gpu_prepare_routed_moe_parts_comparison(
                        &parts_comparison, &emit, gpu, m, sess, lw,
                        &moe_debug, moe_comparison.input_state,
                        l, pos, dim) != 0) {
                    bn_transformer_gpu_discard_moe_layer_comparison(
                        &moe_comparison);
                    bn_transformer_gpu_release_moe_temporaries(
                        m, &moe_temporaries);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu cached moe parts compare failed");
                }
                bn_transformer_gpu_discard_routed_moe_parts_comparison(
                    &parts_comparison);
                int shared_fallback_rc =
                    bn_transformer_gpu_fallback_shared_expert_residual_from_input(
                        &emit, gpu, m, sess, lw, sess->state.xb, dim);
                if (shared_fallback_rc != 0) {
                    bn_transformer_gpu_discard_moe_layer_comparison(
                        &moe_comparison);
                    bn_transformer_gpu_release_moe_temporaries(
                        m, &moe_temporaries);
                    return bn_transformer_gpu_reject_forward(
                        &emit, shared_fallback_rc == -2
                            ? "gpu shared moe cpu fallback upload failed"
                            : "gpu shared moe cpu fallback failed");
                }
                bn_transformer_gpu_emit_context_moe_finish(
                    &emit, dim, u_eps, next_norm);
            } else {
                bn_transformer_gpu_emit_context_moe(
                    &emit, &moe_res, &moe_shared, lw, dim, u_eps,
                    next_norm, moe_activation.uses_reference_silu,
                    moe_activation.uses_reference_ffn_activation);
                BnTransformerGPUMoEPartsComparison parts_comparison;
                if (bn_transformer_gpu_prepare_routed_moe_parts_comparison(
                        &parts_comparison, &emit, gpu, m, sess, lw,
                        &moe_debug, moe_comparison.input_state,
                        l, pos, dim) != 0) {
                    bn_transformer_gpu_discard_moe_layer_comparison(
                        &moe_comparison);
                    bn_transformer_gpu_release_moe_temporaries(
                        m, &moe_temporaries);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu cached moe parts compare failed");
                }
                bn_transformer_gpu_discard_routed_moe_parts_comparison(
                    &parts_comparison);
            }
            int moe_per_dim = bn_transformer_per_layer_embedding_dim(c);
            if (moe_per_dim > 0 &&
                bn_transformer_gpu_emit_context_per_layer_input(
                    &emit, lw, &gpu_layer_res.per_layer_input, l, dim,
                    moe_per_dim, u_eps, next_norm,
                    layer_ffn_plan.use_layer_output_scale,
                    layer_ffn_plan.use_layer_output_scale
                        ? lw->norm.layer_output_scale[0] : 1.0f,
                    small_dense_native_quant_use.use_ffn) != 0) {
                bn_transformer_gpu_discard_moe_layer_comparison(
                    &moe_comparison);
                bn_transformer_gpu_release_moe_temporaries(
                    m, &moe_temporaries);
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe per-layer input adapter failed");
            }
            if (moe_per_dim <= 0 &&
                layer_ffn_plan.use_layer_output_scale &&
                bn_transformer_gpu_emit_context_scale(
                    &emit, BN_GPU_VALUE_X, dim,
                    lw->norm.layer_output_scale[0]) != 0) {
                bn_transformer_gpu_discard_moe_layer_comparison(
                    &moe_comparison);
                bn_transformer_gpu_release_moe_temporaries(
                    m, &moe_temporaries);
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe layer output scale failed");
            }
            if (moe_temporaries.n_buffers > 0 || moe_comparison.enabled) {
                if (bn_transformer_gpu_flush_and_release_moe_temporaries(
                        &emit, gpu, m, &moe_temporaries) != 0) {
                    bn_transformer_gpu_discard_moe_layer_comparison(
                        &moe_comparison);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu execute flush failed");
                }
                if (moe_comparison.enabled &&
                    bn_transformer_gpu_complete_moe_layer_comparison(
                        &moe_comparison, gpu, m, l, pos,
                        dim, norm_eps) != 0) {
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe compare readback failed");
                }
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
        int compare_ffn_state = compare_ffn_state_layer == l &&
            (compare_ffn_state_pos < 0 || compare_ffn_state_pos == pos);
        if (bn_transformer_gpu_cpu_fallback_layer_selected(
                l, cpu_fallback.ffn_layer, cpu_fallback.ffn_from_layer)) {
            if (bn_transformer_gpu_fallback_cpu_ffn(
                    &emit, gpu, m, sess, lw, &ffn_plan, l, dim, u_eps,
                    next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-ffn fallback failed");
            continue;
        }
        BnTransformerGPUDenseFFNResources ffn_res = layer_ffn_res;
        int ffn_down_input_buf = -1;
        int skip_ffn_down = bn_transformer_gpu_cpu_fallback_layer_selected(
            l, -1, cpu_fallback.ffn_down_from_layer);
        if (compare_ffn_state) {
            if (bn_transformer_gpu_debug_snapshot_ffn_state(
                    &emit, gpu, sess, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-state pre-compare snapshot failed");
        }
        bn_transformer_gpu_emit_context_dense_ffn(
            &emit, c, lw, &ffn_plan, &ffn_res, dim, u_eps,
            next_norm, skip_ffn_down, &ffn_down_input_buf,
            small_dense_native_quant_use.use_ffn,
            small_dense_native_quant_use.use_ffn_down, 0);
        if (!skip_ffn_down &&
            compare_ffn_down_layer == l &&
            (compare_ffn_down_pos < 0 || compare_ffn_down_pos == pos)) {
            if (bn_transformer_gpu_debug_compare_ffn_down(
                    &emit, gpu, m, sess, lw, &ffn_plan, l, pos,
                    ffn_down_input_buf,
                    ffn_plan.hidden_dim, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-down compare failed");
        }
        if (!skip_ffn_down && compare_ffn_state) {
            const float *next_norm_cpu = (l + 1 < c->n_layers)
                ? w->layers[l + 1].norm.attn_norm
                : w->output_norm;
            if (bn_transformer_gpu_debug_compare_ffn_state(
                    &emit, gpu, m, sess, lw, &ffn_plan, next_norm_cpu,
                    l, pos, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-state compare failed");
        }
        if (skip_ffn_down) {
            if (bn_transformer_gpu_fallback_cpu_ffn_down(
                    &emit, gpu, m, sess, lw, &ffn_plan, ffn_down_input_buf,
                    ffn_plan.hidden_dim, dim, u_eps, next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-ffn-down fallback failed");
        }
        int per_dim = bn_transformer_per_layer_embedding_dim(c);
        if (per_dim > 0 &&
            bn_transformer_gpu_emit_context_per_layer_input(
                &emit, lw, &gpu_layer_res.per_layer_input, l, dim, per_dim,
                u_eps, next_norm, ffn_plan.use_layer_output_scale,
                ffn_plan.use_layer_output_scale
                    ? lw->norm.layer_output_scale[0] : 1.0f,
                small_dense_native_quant_use.use_ffn) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu per-layer input adapter emit failed");
        if (per_dim > 0 && compare_ffn_state &&
            bn_transformer_gpu_debug_compare_per_layer_state(
                &emit, gpu, m, sess, lw, l, pos, dim) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu per-layer input adapter compare failed");
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
            if (bn_transformer_gpu_capture_logits_refine_state(
                    &emit, gpu, sess, dim) != 0)
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
            bn_transformer_gpu_try_refined_argmax(
                gpu, m, sess, logit_res, &logits_refine, dim,
                penalty_tokens, n_penalty_tokens, repeat_penalty,
                argmax_token)) {
            bn_transformer_gpu_emit_context_free(&emit);
            return s->x;
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
        bn_transformer_gpu_debug_compare_argmax(
            gpu, c->vocab_size, penalty_tokens, n_penalty_tokens,
            repeat_penalty, *argmax_token);
        bn_transformer_gpu_emit_context_free(&emit);
        return s->x;
    }
    if (!need_logits) {
        bn_transformer_gpu_emit_context_free(&emit);
        return s->x;
    }
    bn_transformer_gpu_refine_output_logits(
        gpu, m, sess, logit_res, &logits_refine, dim,
        kquant_logits_refine_has_xb_snapshot);
    bn_transformer_logits_apply_final_softcap(
        s->logits, c->vocab_size,
        bn_transformer_logits_final_softcap(c));
    bn_transformer_gpu_debug_compare_logits(
        gpu, m, sess, logit_res, pos, dim);
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
