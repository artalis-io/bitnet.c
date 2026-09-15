#include "moe_internal.h"
#include "model_internal.h"
#include "gpu_backend.h"
#include "gpu_moe_bridge.h"
#include "sh_arena.h"
#include "transformer/gpu_internal.h"
#include "transformer_prefill_internal.h"

static SHArena *moe_prepare_projection(BnPreparedWeight *prepared,
                                       const BnQWeight *weight) {
    size_t bytes = bn_moe_quant_prepared_weight_size(weight);
    if (bytes == 0)
        return NULL;
    SHArena *arena = sh_arena_create(bytes);
    if (!arena)
        return NULL;
    if (bn_moe_quant_prepare_weight(prepared, weight, arena) != 0) {
        sh_arena_free(arena);
        return NULL;
    }
    return arena;
}

typedef struct {
    BnPreparedWeight local;
    SHArena *arena;
    const BnPreparedWeight *cached;
} BnMoEPrefillProjection;

static const BnPreparedWeight *moe_acquire_projection(
    BnMoEPrefillProjection *projection, BnModel *model,
    const BnQWeight *weight, int n_tokens, BnThreadPool *pool) {
    const BnCPURuntimePolicy *cpu_policy = bn_tp_cpu_policy(pool);
    if (cpu_policy && !cpu_policy->prepared_qweights)
        return NULL;
    if (!bn_moe_quant_matvec_uses_prepared_weight(weight, 0, pool))
        return NULL;
    projection->cached = bn_moe_acquire_prepared_projection(model, weight);
    if (projection->cached)
        return projection->cached;
    if (!bn_moe_quant_batch_preparation_worthwhile(weight, n_tokens, pool))
        return NULL;
    projection->arena = moe_prepare_projection(&projection->local, weight);
    return projection->arena ? &projection->local : NULL;
}

static void moe_release_projection(BnMoEPrefillProjection *projection,
                                    BnModel *model) {
    bn_moe_release_prepared_projection(model, projection->cached);
    sh_arena_free(projection->arena);
}

static void moe_batch_route(float *logits, int *all_indices, float *all_weights,
                            const float *x, const float *router_w,
                            int n_tokens, int dim, int n_experts, int k,
                            int norm_topk_prob, float expert_weights_scale,
                            int uses_reference_router_accumulation,
                            int uses_matvec_router, BnThreadPool *pool) {
    /* Decode and prefill have independent accumulation contracts. Honor
     * the prefill choice independently of the row-dot arithmetic policy. */
    if (!uses_matvec_router &&
        bn_moe_router_batch_logits(logits, router_w, x, n_tokens,
                                   dim, n_experts, pool)) {
        for (int t = 0; t < n_tokens; t++)
            bn_moe_route_logits(logits + (size_t)t * n_experts,
                all_indices + (size_t)t * k, all_weights + (size_t)t * k,
                n_experts, k, norm_topk_prob, expert_weights_scale);
        return;
    }
    for (int t = 0; t < n_tokens; t++)
        bn_moe_route_buffers(
            logits + (size_t)t * n_experts,
            all_indices + (size_t)t * k,
            all_weights + (size_t)t * k,
            x + (size_t)t * dim, router_w, dim, n_experts, k,
            norm_topk_prob, expert_weights_scale,
            uses_reference_router_accumulation, pool);
}

/* Commit only after the complete composition succeeds. In particular, a
 * backend rejection must leave the routed output and residual available to
 * the CPU implementation. Input is caller-owned scratch. */
static int moe_batch_dense_gpu(BnModel *m, int layer, BnLayerWeights *lw,
                              const BnMoEExecutionPolicy *policy,
                              const float *act, float *input, float *out,
                              int tokens, int raw,
                              BnMoEBatchObserveFn observe, void *observe_ctx) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    BnMoEPrefillDenseGPUResourcePolicy r =
        bn_moe_prefill_dense_gpu_resource_policy(bn_model_backend(m), layer, lw);
    if (!r.gate || !r.up || !r.down || !r.norm || !r.dense_norm ||
        !r.routed_norm || !r.post_norm ||
        !lw->norm.ffn_norm || !lw->norm.ffn_post_norm_1 ||
        !lw->norm.ffn_post_norm_2 || !lw->norm.ffn_post_norm ||
        !bn_gpu_backend_can_dense_ffn_batch(gpu) ||
        !bn_gpu_backend_can_rmsnorm_batch(gpu) ||
        !bn_gpu_backend_can_rmsnorm_residual_batch(gpu))
        return -1;
    int dim = m->config.dim;
    size_t bytes = (size_t)tokens * dim * sizeof(float);
    BnAllocator a = bn_allocator_default();
    float *dense = bn_malloc(&a, bytes);
    float *combined = bn_malloc(&a, bytes);
    float *final = bn_malloc(&a, bytes);
    int rc = -1;
    if (!dense || !combined || !final) goto cleanup;
    if (bn_gpu_backend_rmsnorm_batch(gpu, input, r.norm, act,
            tokens, dim, policy->norm_eps) != 0 ||
        bn_gpu_backend_dense_ffn_batch(gpu, dense, r.gate, r.up, r.down,
            input, tokens, dim, m->config.hidden_dim,
            lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type, lw->ffn.ffn_down.type,
            policy->activation) != 0 ||
        bn_gpu_backend_rmsnorm_batch(gpu, dense, r.dense_norm, dense,
            tokens, dim, policy->norm_eps) != 0 ||
        bn_gpu_backend_rmsnorm_residual_batch(gpu, combined, r.routed_norm,
            out, dense, tokens, dim, policy->norm_eps) != 0)
        goto cleanup;
    rc = raw
        ? bn_gpu_backend_rmsnorm_batch(gpu, final, r.post_norm, combined,
                                      tokens, dim, policy->norm_eps)
        : bn_gpu_backend_rmsnorm_residual_batch(gpu, final, r.post_norm,
                    combined, act, tokens, dim, policy->norm_eps);
    if (rc != 0) goto cleanup;
    if (observe)
        for (int t = 0; t < tokens; t++) {
            observe(observe_ctx, BN_MOE_OBSERVE_DENSE_INPUT, t, -1, -1,
                    input + (size_t)t * dim, dim);
            observe(observe_ctx, BN_MOE_OBSERVE_DENSE_OUTPUT, t, -1, -1,
                    dense + (size_t)t * dim, dim);
            observe(observe_ctx, BN_MOE_OBSERVE_COMBINED_OUTPUT, t, -1, -1,
                    combined + (size_t)t * dim, dim);
        }
    memcpy(out, final, bytes);
cleanup:
    if (dense) bn_free(&a, dense, bytes);
    if (combined) bn_free(&a, combined, bytes);
    if (final) bn_free(&a, final, bytes);
    return rc;
}

/* Complete architectures with separately normalized dense and routed FFNs.
 * Keep the dense projections batched; the caller retains the raw residual
 * input until both branches and their output norms have finished. */
static int moe_batch_dense_residual(BnModel *m, BnLayerWeights *lw,
                                   const BnMoEExecutionPolicy *policy,
                                   const float *act, float *input, float *out,
                                   int tokens, BnMoEBatchObserveFn observe,
                                   void *observe_ctx) {
    int dim = m->config.dim;
    int hidden = m->config.hidden_dim;
    int has_dense = lw->ffn.ffn_gate.data && lw->ffn.ffn_up.data &&
                    lw->ffn.ffn_down.data;
    BnAllocator a = bn_allocator_default();
    size_t hidden_bytes = (size_t)tokens * hidden * sizeof(float);
    size_t output_bytes = (size_t)tokens * dim * sizeof(float);
    size_t quant_bytes = (size_t)tokens * (dim > hidden ? dim : hidden);
    float *gate = has_dense ? bn_malloc(&a, hidden_bytes) : NULL;
    float *up = has_dense ? bn_malloc(&a, hidden_bytes) : NULL;
    float *down = has_dense ? bn_malloc(&a, output_bytes) : NULL;
    int8_t *quant = has_dense ? bn_malloc(&a, quant_bytes) : NULL;
    int rc = -1;
    if (has_dense && (!gate || !up || !down || !quant)) {
        SH_LOG_ERROR("Failed to allocate dense residual MoE batch buffers");
        goto cleanup;
    }
    for (int t = 0; t < tokens; t++) {
        float *routed = out + (size_t)t * dim;
        if (lw->norm.ffn_post_norm_2)
            bn_moe_rmsnorm(routed, routed, lw->norm.ffn_post_norm_2,
                           dim, policy->norm_eps);
        if (observe)
            observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_POST_NORM,
                    t, -1, -1, routed, dim);
        if (has_dense) {
            bn_moe_rmsnorm(input + (size_t)t * dim, act + (size_t)t * dim,
                           lw->norm.ffn_norm, dim, policy->norm_eps);
            if (observe)
                observe(observe_ctx, BN_MOE_OBSERVE_DENSE_INPUT,
                        t, -1, -1, input + (size_t)t * dim, dim);
        }
    }
    if (has_dense) {
        const BnQWeight *weights[2] = {&lw->ffn.ffn_gate, &lw->ffn.ffn_up};
        const BnPreparedWeight *prepared[2] = {
            bn_moe_prepared_dense_projection(m, weights[0]),
            bn_moe_prepared_dense_projection(m, weights[1])
        };
        float *outputs[2] = {gate, up};
        bn_moe_quant_matmul_prepared_multi(outputs, weights, prepared, 2,
                                          input, tokens, quant, bn_model_pool(m));
        bn_moe_swiglu(gate, gate, up, tokens * hidden,
                      policy->uses_reference_silu,
                      policy->uses_reference_ffn_activation);
        if (observe)
            for (int t = 0; t < tokens; t++)
                observe(observe_ctx, BN_MOE_OBSERVE_DENSE_ACTIVATION,
                        t, -1, -1, gate + (size_t)t * hidden, hidden);
        bn_moe_quant_matmul_prepared(down, &lw->ffn.ffn_down,
            bn_moe_prepared_dense_projection(m, &lw->ffn.ffn_down),
            gate, tokens, quant, bn_model_pool(m));
    }
    for (int t = 0; t < tokens; t++) {
        float *combined = out + (size_t)t * dim;
        if (has_dense) {
            float *dense = down + (size_t)t * dim;
            if (lw->norm.ffn_post_norm_1)
                bn_moe_rmsnorm(dense, dense, lw->norm.ffn_post_norm_1,
                               dim, policy->norm_eps);
            if (observe)
                observe(observe_ctx, BN_MOE_OBSERVE_DENSE_OUTPUT,
                        t, -1, -1, dense, dim);
            bn_moe_residual_add(combined, dense, dim);
        }
        if (observe)
            observe(observe_ctx, BN_MOE_OBSERVE_COMBINED_OUTPUT,
                    t, -1, -1, combined, dim);
        if (lw->norm.ffn_post_norm)
            bn_moe_rmsnorm(combined, combined, lw->norm.ffn_post_norm,
                           dim, policy->norm_eps);
    }
    rc = 0;
cleanup:
    if (gate) bn_free(&a, gate, hidden_bytes);
    if (up) bn_free(&a, up, hidden_bytes);
    if (down) bn_free(&a, down, output_bytes);
    if (quant) bn_free(&a, quant, quant_bytes);
    return rc;
}

static int moe_batch_input_norm_gpu(BnModel *m, int layer, float *out,
                                    const float *input, int tokens,
                                    int use_sub_norm, float eps) {
    BnMoEPrefillRoutedGPUResourcePolicy resources =
        bn_moe_prefill_routed_gpu_resource_policy(bn_model_backend(m), layer);
    void *norm = use_sub_norm ? resources.sub_norm : resources.norm;
    if (!norm)
        return -1;
    return bn_transformer_gpu_prefill_rmsnorm_backend_run(
        bn_model_gpu(m), out, norm, input, tokens, m->config.dim, eps);
}

// --- Batch MoE FFN for prefill ---
// Route all n_tokens, group by expert, batch matmul per expert.
int bn_moe_forward_batch_observed(struct BnModel *m, BnSession *sess,
                                  struct BnLayerWeights *lw, int l,
                                  float *act, float *Xb, int n_tokens,
                                  uint32_t flags,
                                  BnMoEBatchObserveFn observe,
                                  void *observe_ctx) {
    double t_compute = bn_moe_time_ms();
    double t0;
    BnConfig *c = &m->config;
    BnMoEState *ms = sess->moe_state;
    int dim = c->dim;
    BnMoEExecutionPolicy exec_policy = bn_moe_execution_policy(c);
    BnMoERoutePolicy route_policy = bn_moe_route_policy(c);
    int moe_hidden = route_policy.expert_hidden_dim;
    int K = route_policy.active_experts;
    int n_experts = route_policy.total_experts;
    BnMoEPrefillPolicy prefill_policy = bn_moe_prefill_policy(c);
    const int force_host =
        bn_transformer_prefill_host_reference_enabled(bn_model_gpu(m), c);
    BnGPUBackend *prefill_gpu = force_host ? NULL : bn_model_gpu(m);
    BnMoELoadedSharedExpertPolicy shared_policy =
        bn_moe_loaded_shared_expert_policy(c, lw);
    const BnMoEExpertMap *map = &lw->moe.expert_map;
    BnMoERoutedExpertProjectionTypes routed_types;
    if (!bn_moe_routed_expert_projection_types(&routed_types, map))
        return -1;

    BnAllocator a = bn_allocator_default();
    int did_input_norm = 0;
    const float *input_norm = lw->norm.ffn_norm;
    if (exec_policy.uses_dense_residual_branch && lw->norm.ffn_sub_norm)
        input_norm = lw->norm.ffn_sub_norm;
    if (flags & BN_MOE_BATCH_INPUT_PRENORMALIZED) {
        memcpy(Xb, act, (size_t)n_tokens * dim * sizeof(float));
        did_input_norm = 1;
    }
    if (prefill_policy.uses_grouped_expert_route &&
        !exec_policy.uses_scaled_router_input &&
        !exec_policy.uses_dense_residual_branch &&
        !lw->moe.expert_down_scale &&
        !(flags & BN_MOE_BATCH_INPUT_PRENORMALIZED)) {
        BnGPUBackend *gpu = prefill_gpu;
        BnBackendModel *backend = bn_model_backend(m);
        if (bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
                gpu, c, &lw->moe.expert_map) &&
            backend) {
            BnMoEPrefillRoutedGPUResourcePolicy routed_resources =
                bn_moe_prefill_routed_gpu_resource_policy(backend, l);
            BnTransformerGPUMoESharedFFNResources shared_gpu;
            bn_transformer_gpu_resolve_moe_shared_ffn_resources(
                &shared_gpu, backend, c, lw, l, 0);
            if (routed_resources.norm_resid_valid) {
                t0 = bn_moe_time_ms();
                if (bn_transformer_gpu_prefill_moe_ffn_batch_backend_run(
                        gpu, act, routed_resources.router,
                        routed_resources.gate_all, routed_resources.up_all,
                        routed_resources.down_all,
                        shared_gpu.gate, shared_gpu.up, shared_gpu.down,
                        shared_gpu.gate_weight, routed_resources.norm, act,
                        n_tokens, dim,
                        moe_hidden, n_experts, K,
                        routed_types.gate_type, routed_types.up_type,
                        routed_types.down_type,
                        exec_policy.activation, shared_gpu.hidden_dim,
                        shared_gpu.gate_type, shared_gpu.up_type,
                        shared_gpu.down_type,
                        exec_policy.norm_eps, route_policy.norm_topk_prob,
                        route_policy.expert_weights_scale) == 0) {
                    ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
                    ms->stats.compute_time_ms +=
                        bn_moe_time_ms() - t_compute;
                    return 0;
                }
            }
        }

        // Batch RMSNorm for the host/GPU split MoE prefill paths below.
        t0 = bn_moe_time_ms();
        if (force_host ||
            moe_batch_input_norm_gpu(m, l, Xb, act, n_tokens,
                exec_policy.uses_dense_residual_branch && lw->norm.ffn_sub_norm,
                exec_policy.norm_eps) != 0)
            for (int t = 0; t < n_tokens; t++)
                bn_moe_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                               input_norm, dim, exec_policy.norm_eps);
        ms->stats.norm_time_ms += bn_moe_time_ms() - t0;
        did_input_norm = 1;

        // Keep diagnostics available when the resident routed callback below
        // consumes the normalized batch and returns before the split path.
        if (observe)
            for (int t = 0; t < n_tokens; t++)
                observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_INPUT,
                        t, -1, -1, Xb + (size_t)t * dim, dim);

        if (bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
                gpu, c, map, dim, 0) &&
            backend) {
            BnMoEPrefillRoutedGPUResourcePolicy routed_resources =
                bn_moe_prefill_routed_gpu_resource_policy(backend, l);
            size_t sz_mout = (size_t)n_tokens * dim * sizeof(float);
            float *moe_out = (float *)bn_malloc(&a, sz_mout);
            if (moe_out && routed_resources.routed_valid) {
                t0 = bn_moe_time_ms();
                if (bn_transformer_gpu_moe_prefill_routed_ffn_batch_backend_run(
                        gpu, moe_out, routed_resources.router,
                        routed_resources.gate_all, routed_resources.up_all,
                        routed_resources.down_all,
                        Xb, n_tokens, dim, moe_hidden, n_experts, K,
                        routed_types.gate_type, routed_types.up_type,
                        routed_types.down_type,
                        exec_policy.activation, route_policy.norm_topk_prob,
                        route_policy.expert_weights_scale) == 0) {
                    ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
                    int shared_ok = 1;
                    if (shared_policy.has_loaded_path) {
                        shared_ok = 0;
                        size_t sz_shd = (size_t)n_tokens * dim * sizeof(float);
                        float *sh_d = (float *)bn_malloc(&a, sz_shd);
                        BnTransformerGPUMoESharedFFNResources shared_gpu;
                        if (sh_d &&
                            bn_transformer_gpu_moe_prefill_shared_dense_ffn_available(
                                gpu) &&
                            bn_transformer_gpu_resolve_moe_shared_ffn_resources(
                                &shared_gpu, backend, c, lw, l, 0)) {
                            double ts = bn_moe_time_ms();
                            if (bn_transformer_gpu_prefill_dense_ffn_batch_backend_run(
                                    gpu, sh_d, shared_gpu.gate, shared_gpu.up,
                                    shared_gpu.down, Xb, n_tokens, dim,
                                    shared_gpu.hidden_dim,
                                    shared_gpu.gate_type,
                                    shared_gpu.up_type,
                                    shared_gpu.down_type,
                                    exec_policy.activation) == 0) {
                                for (int t = 0; t < n_tokens; t++) {
                                    const float *x_t = Xb + (size_t)t * dim;
                                    float gate =
                                        bn_moe_shared_expert_gate_weight(
                                            lw, x_t, dim);
                                    for (int d = 0; d < dim; d++)
                                        moe_out[(size_t)t * dim + d] +=
                                            gate * sh_d[(size_t)t * dim + d];
                                }
                                ms->stats.shared_time_ms += bn_moe_time_ms() - ts;
                                shared_ok = 1;
                            }
                        }
                        if (sh_d)
                            bn_free(&a, sh_d, sz_shd);
                    }
                    if (shared_ok) {
                        if (observe)
                            for (int t = 0; t < n_tokens; t++)
                                observe(observe_ctx,
                                        BN_MOE_OBSERVE_ROUTED_OUTPUT,
                                        t, -1, -1,
                                        moe_out + (size_t)t * dim, dim);
                        for (int t = 0; t < n_tokens; t++)
                            for (int d = 0; d < dim; d++)
                                act[(size_t)t * dim + d] +=
                                    moe_out[(size_t)t * dim + d];
                        bn_free(&a, moe_out, sz_mout);
                        ms->stats.compute_time_ms +=
                            bn_moe_time_ms() - t_compute;
                        return 0;
                    }
                }
            }
            if (moe_out)
                bn_free(&a, moe_out, sz_mout);
        }
    }

    // 1. Batch RMSNorm
    if (!did_input_norm) {
        t0 = bn_moe_time_ms();
        if (force_host ||
            moe_batch_input_norm_gpu(m, l, Xb, act, n_tokens,
                exec_policy.uses_dense_residual_branch && lw->norm.ffn_sub_norm,
                exec_policy.norm_eps) != 0)
            for (int t = 0; t < n_tokens; t++)
                bn_moe_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                               input_norm, dim, exec_policy.norm_eps);
        ms->stats.norm_time_ms += bn_moe_time_ms() - t0;
    }
    if (observe)
        for (int t = 0; t < n_tokens; t++)
            observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_INPUT, t, -1, -1,
                    Xb + (size_t)t * dim, dim);
    // 2. Batch routing: route each token individually (reuse existing router)
    // Allocate routing results: [n_tokens][K] indices and weights
    size_t sz_idx = (size_t)n_tokens * K * sizeof(int);
    size_t sz_wts = (size_t)n_tokens * K * sizeof(float);
    size_t sz_logits = (size_t)n_tokens * n_experts * sizeof(float);
    int *all_indices = (int *)bn_malloc(&a, sz_idx);
    float *all_weights = (float *)bn_malloc(&a, sz_wts);
    float *batch_logits = (float *)bn_malloc(&a, sz_logits);
    size_t sz_router = (size_t)n_tokens * dim * sizeof(float);
    float *scaled_router = exec_policy.uses_scaled_router_input
        ? bn_malloc(&a, sz_router) : NULL;
    if (!all_indices || !all_weights || !batch_logits ||
        (exec_policy.uses_scaled_router_input && !scaled_router)) {
        if (scaled_router) bn_free(&a, scaled_router, sz_router);
        if (all_indices) bn_free(&a, all_indices, sz_idx);
        if (all_weights) bn_free(&a, all_weights, sz_wts);
        if (batch_logits) bn_free(&a, batch_logits, sz_logits);
        return -1;
    }
    memset(all_indices, 0, sz_idx);
    memset(all_weights, 0, sz_wts);

    const float *router_input = Xb;
    if (scaled_router) {
        BnMoEPrefillRoutedGPUResourcePolicy router_resources =
            bn_moe_prefill_routed_gpu_resource_policy(bn_model_backend(m), l);
        if (force_host || !router_resources.router_scale ||
            bn_transformer_gpu_prefill_scaled_rmsnorm_backend_run(
                prefill_gpu, scaled_router, router_resources.router_scale,
                act, n_tokens, dim, exec_policy.norm_eps,
                1.0f / sqrtf((float)dim)) != 0)
            for (int t = 0; t < n_tokens; t++)
                bn_moe_scaled_router_input(scaled_router + (size_t)t * dim,
                    act + (size_t)t * dim, lw->moe.router_scale,
                    dim, exec_policy.norm_eps);
        router_input = scaled_router;
    }
    t0 = bn_moe_time_ms();
    int used_gpu_route = 0;
    BnGPUBackend *route_gpu = prefill_gpu;
    BnBackendModel *route_backend = bn_model_backend(m);
    if (bn_transformer_gpu_moe_prefill_route_batch_available(
            route_gpu, c, route_backend != NULL)) {
        BnMoEPrefillRoutedGPUResourcePolicy route_resources =
            bn_moe_prefill_routed_gpu_resource_policy(route_backend, l);
        int route_rc = route_resources.router
            ? bn_transformer_gpu_moe_prefill_route_batch_backend_run(
                  route_gpu, all_indices, all_weights, route_resources.router,
                  router_input,
                  n_tokens, dim, n_experts, K, route_policy.norm_topk_prob,
                  route_policy.expert_weights_scale)
            : -1;
        if (route_rc == 0)
            used_gpu_route = 1;
        else if (bn_transformer_gpu_moe_route_batch_debug_enabled(route_gpu))
            fprintf(stderr,
                    "[bn:gpu:moe-route-batch] fallback layer=%d handle=%d rc=%d tokens=%d experts=%d k=%d dim=%d\n",
                    l, route_resources.router != NULL, route_rc, n_tokens,
                    n_experts, K, dim);
    }
    if (!used_gpu_route) {
        moe_batch_route(batch_logits, all_indices, all_weights, router_input,
                        lw->moe.router_weight, n_tokens, dim, n_experts, K,
                        route_policy.norm_topk_prob,
                        route_policy.expert_weights_scale,
                        route_policy.uses_reference_router_accumulation,
                        prefill_policy.uses_matvec_router, bn_model_pool(m));
    }
    ms->stats.route_time_ms += bn_moe_time_ms() - t0;
    if (observe && !used_gpu_route)
        for (int t = 0; t < n_tokens; t++)
            observe(observe_ctx, BN_MOE_OBSERVE_ROUTER_LOGITS, t, -1, -1,
                    batch_logits + (size_t)t * n_experts, n_experts);
    if (observe)
        for (int t = 0; t < n_tokens; t++)
            observe(observe_ctx, BN_MOE_OBSERVE_ROUTE_WEIGHTS, t, -1, -1,
                    all_weights + (size_t)t * K, K);

    if (scaled_router) bn_free(&a, scaled_router, sz_router);

    // 3. Build token-expert grouping (two-pass)
    // Pass 1: count tokens per expert
    size_t sz_ecnt = (size_t)n_experts * sizeof(int);
    int *expert_counts = (int *)bn_malloc(&a, sz_ecnt);
    int *expert_offsets = (int *)bn_malloc(&a, sz_ecnt);
    if (!expert_counts || !expert_offsets) {
        if (expert_counts) bn_free(&a, expert_counts, sz_ecnt);
        if (expert_offsets) bn_free(&a, expert_offsets, sz_ecnt);
        bn_free(&a, batch_logits, sz_logits);
        bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
        return -1;
    }
    memset(expert_counts, 0, sz_ecnt);
    memset(expert_offsets, 0, sz_ecnt);

    for (int t = 0; t < n_tokens; t++)
        for (int k = 0; k < K; k++) {
            int eidx = all_indices[(size_t)t * K + k];
            if (eidx >= 0) expert_counts[eidx]++;
        }

    // Prefix sum for offsets
    int total_assignments = 0;
    for (int e = 0; e < n_experts; e++) {
        expert_offsets[e] = total_assignments;
        total_assignments += expert_counts[e];
    }

    // Pass 2: fill flat arrays
    size_t sz_gtid = (size_t)total_assignments * sizeof(int);
    size_t sz_gwts = (size_t)total_assignments * sizeof(float);
    size_t sz_fill = (size_t)n_experts * sizeof(int);
    int *group_token_ids = (int *)bn_malloc(&a, sz_gtid);
    float *group_weights = (float *)bn_malloc(&a, sz_gwts);
    int *fill_pos = (int *)bn_malloc(&a, sz_fill);
    if (!group_token_ids || !group_weights || !fill_pos) {
        if (group_token_ids) bn_free(&a, group_token_ids, sz_gtid);
        if (group_weights) bn_free(&a, group_weights, sz_gwts);
        if (fill_pos) bn_free(&a, fill_pos, sz_fill);
        bn_free(&a, expert_counts, sz_ecnt); bn_free(&a, expert_offsets, sz_ecnt);
        bn_free(&a, batch_logits, sz_logits);
        bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
        return -1;
    }
    memset(group_token_ids, 0, sz_gtid);
    memset(group_weights, 0, sz_gwts);
    memset(fill_pos, 0, sz_fill);

    for (int t = 0; t < n_tokens; t++)
        for (int k = 0; k < K; k++) {
            int eidx = all_indices[(size_t)t * K + k];
            if (eidx < 0) continue;
            int pos = expert_offsets[eidx] + fill_pos[eidx];
            group_token_ids[pos] = t;
            group_weights[pos] = all_weights[(size_t)t * K + k];
            fill_pos[eidx]++;
        }

    // 4. Allocate batch compute buffers
    // T_max = max tokens assigned to any single expert
    int T_max = 0;
    for (int e = 0; e < n_experts; e++)
        if (expert_counts[e] > T_max) T_max = expert_counts[e];

    size_t sz_gather = (size_t)T_max * dim * sizeof(float);
    size_t sz_gate   = (size_t)T_max * moe_hidden * sizeof(float);
    size_t sz_up     = sz_gate;
    size_t sz_down   = sz_gather;
    size_t sz_mout   = (size_t)n_tokens * dim * sizeof(float);
    size_t sz_slots  = (size_t)n_tokens * K * dim * sizeof(float);
    size_t sz_xq     = (size_t)T_max * (size_t)(dim > moe_hidden ? dim : moe_hidden);
    float *gather_buf   = (float *)bn_malloc(&a, sz_gather);
    float *gate_buf     = (float *)bn_malloc(&a, sz_gate);
    float *up_buf       = (float *)bn_malloc(&a, sz_up);
    float *down_buf     = (float *)bn_malloc(&a, sz_down);
    float *moe_out      = (float *)bn_malloc(&a, sz_mout);
    float *slot_out     = (float *)bn_malloc(&a, sz_slots);
    int8_t *x_q_scratch = (int8_t *)bn_malloc(&a, sz_xq);
    if (!gather_buf || !gate_buf || !up_buf || !down_buf || !moe_out ||
        !slot_out || !x_q_scratch) {
        if (gather_buf) bn_free(&a, gather_buf, sz_gather);
        if (gate_buf) bn_free(&a, gate_buf, sz_gate);
        if (up_buf) bn_free(&a, up_buf, sz_up);
        if (down_buf) bn_free(&a, down_buf, sz_down);
        if (moe_out) bn_free(&a, moe_out, sz_mout);
        if (slot_out) bn_free(&a, slot_out, sz_slots);
        if (x_q_scratch) bn_free(&a, x_q_scratch, sz_xq);
        bn_free(&a, group_token_ids, sz_gtid); bn_free(&a, group_weights, sz_gwts);
        bn_free(&a, fill_pos, sz_fill); bn_free(&a, expert_counts, sz_ecnt);
        bn_free(&a, expert_offsets, sz_ecnt);
        bn_free(&a, batch_logits, sz_logits);
        bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
        return -1;
    }
    memset(moe_out, 0, sz_mout);
    memset(slot_out, 0, sz_slots);

    // 5. Per-expert batch compute
    int used_gpu_moe_batch = 0;
    int used_gpu_shared_batch = 0;
    BnGPUBackend *gpu_batch = prefill_gpu;
    int resident_batch_available =
        bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
            gpu_batch, c, map, dim, 0,
            bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
                gpu_batch, c, bn_model_gpu_moe_cache(m) != NULL));
    float *reduction_scales =
        (bn_transformer_gpu_moe_reduce_batch_backend_available(gpu_batch) ||
         resident_batch_available)
            ? bn_malloc(&a, sz_wts) : NULL;
    if (reduction_scales)
        for (int t = 0; t < n_tokens; t++)
            for (int k = 0; k < K; k++) {
                int expert = all_indices[(size_t)t*K + k];
                reduction_scales[(size_t)t*K + k] = expert >= 0
                    ? bn_moe_expert_weight_scale(lw, expert) : 1.0f;
            }
    // GPU batch callbacks currently accept a combined coefficient. Keep
    // those arguments separate from the raw route weights used by the CPU
    // fallback, which scales the projection before applying its route weight.
    float *gpu_scaled_weights = NULL;
    const float *gpu_group_weights = group_weights;
    int gpu_weights_ready = 1;
    if (gpu_batch && lw->moe.expert_down_scale) {
        gpu_scaled_weights = (float *)bn_malloc(&a, sz_wts + sz_gwts);
        gpu_weights_ready = gpu_scaled_weights != NULL;
        if (gpu_scaled_weights) {
            float *group_scaled = gpu_scaled_weights + (size_t)n_tokens * K;
            for (int t = 0; t < n_tokens; t++)
                for (int k = 0; k < K; k++) {
                    size_t slot = (size_t)t * K + k;
                    gpu_scaled_weights[slot] = all_weights[slot] *
                        bn_moe_expert_weight_scale(lw, all_indices[slot]);
                }
            for (int e = 0; e < n_experts; e++)
                for (int i = 0; i < expert_counts[e]; i++) {
                    int slot = expert_offsets[e] + i;
                    group_scaled[slot] = group_weights[slot] *
                        bn_moe_expert_weight_scale(lw, e);
                }
            gpu_group_weights = group_scaled;
        }
    }
    int prefer_cached_expert_batch =
        bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
            gpu_batch, c, bn_model_gpu_moe_cache(m) != NULL);
    if (gpu_weights_ready &&
        bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
            gpu_batch, c, map, dim, 0, prefer_cached_expert_batch)) {
        const BnBackendModel *backend = bn_model_backend(m);
        BnMoEPrefillResidentGPUResourcePolicy resident_resources =
            bn_moe_prefill_resident_gpu_resource_policy(backend, l);
        if (resident_resources.valid) {
            t0 = bn_moe_time_ms();
            if (bn_transformer_gpu_moe_prefill_resident_expert_batch_backend_run(
                    gpu_batch, moe_out, resident_resources.gate_all,
                    resident_resources.up_all, resident_resources.down_all,
                    all_indices, all_weights, reduction_scales, Xb,
                    n_tokens, dim, moe_hidden,
                    n_experts, K, routed_types.gate_type,
                    routed_types.up_type, routed_types.down_type,
                    exec_policy.activation) == 0) {
                used_gpu_moe_batch = 1;
                ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
            }
        }
    }
    if (gpu_weights_ready &&
        bn_transformer_gpu_moe_prefill_split_expert_batch_available(
            gpu_batch, c, map, dim, 0, used_gpu_moe_batch)) {
        BnGPUMoEPrefillExpert *gpu_experts =
            (BnGPUMoEPrefillExpert *)bn_malloc(
                &a, (size_t)n_experts * sizeof(BnGPUMoEPrefillExpert));
        BnGPUMoETemporaryBuffers *temps =
            (BnGPUMoETemporaryBuffers *)bn_malloc(
                &a, (size_t)n_experts * sizeof(BnGPUMoETemporaryBuffers));
        if (gpu_experts && temps) {
            memset(gpu_experts, 0,
                   (size_t)n_experts * sizeof(BnGPUMoEPrefillExpert));
            memset(temps, 0,
                   (size_t)n_experts * sizeof(BnGPUMoETemporaryBuffers));
            int all_resident = 1;
            for (int e = 0; e < n_experts; e++) {
                if (expert_counts[e] <= 0)
                    continue;
                BnGPUMoEExpertBuffers expert_gpu;
                memset(&expert_gpu, 0, sizeof(expert_gpu));
                if (bn_gpu_moe_bridge_get_expert(
                        m, sess, lw, l, e, &temps[e], &expert_gpu) != 0 ||
                    temps[e].n_buffers != 0 ||
                    !expert_gpu.gate || !expert_gpu.down ||
                    (!expert_gpu.up && !expert_gpu.use_gateup_split)) {
                    all_resident = 0;
                    break;
                }
                gpu_experts[e].gate_buf = expert_gpu.gate;
                gpu_experts[e].up_buf = expert_gpu.up;
                gpu_experts[e].down_buf = expert_gpu.down;
                gpu_experts[e].use_gateup_split =
                    expert_gpu.use_gateup_split;
            }
            if (all_resident) {
                const BnBackendModel *backend = bn_model_backend(m);
                BnTransformerGPUMoESharedFFNResources shared_gpu = {0};
                int has_shared_gpu = 0;
                if (bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
                        gpu_batch, c, lw, backend != NULL)) {
                    has_shared_gpu =
                        bn_transformer_gpu_resolve_moe_shared_ffn_resources(
                            &shared_gpu, backend, c, lw, l, 1);
                }
                t0 = bn_moe_time_ms();
                if (bn_transformer_gpu_moe_prefill_split_expert_batch_backend_run(
                        gpu_batch, moe_out, gpu_experts, n_experts,
                        expert_offsets, expert_counts, group_token_ids,
                        gpu_group_weights, Xb, n_tokens, dim, moe_hidden,
                        routed_types.gate_type, routed_types.up_type,
                        routed_types.down_type,
                        exec_policy.activation, shared_gpu.gate, shared_gpu.up,
                        shared_gpu.down, shared_gpu.gate_weight,
                        shared_gpu.hidden_dim, shared_gpu.gate_type,
                        shared_gpu.up_type, shared_gpu.down_type) == 0) {
                    used_gpu_moe_batch = 1;
                    used_gpu_shared_batch = has_shared_gpu;
                    ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
                }
            }
            for (int e = 0; e < n_experts; e++)
                if (temps[e].n_buffers > 0)
                    bn_gpu_moe_bridge_release_temporaries(m, &temps[e]);
        }
        if (gpu_experts)
            bn_free(&a, gpu_experts,
                    (size_t)n_experts * sizeof(BnGPUMoEPrefillExpert));
        if (temps)
            bn_free(&a, temps,
                    (size_t)n_experts * sizeof(BnGPUMoETemporaryBuffers));
    }

    if (gpu_scaled_weights)
        bn_free(&a, gpu_scaled_weights, sz_wts + sz_gwts);

    if (!used_gpu_moe_batch) for (int e = 0; e < n_experts; e++) {
        int T = expert_counts[e];
        if (T == 0) continue;
        int off = expert_offsets[e];

        // Gather: collect this expert's tokens' activations
        for (int i = 0; i < T; i++)
            memcpy(gather_buf + (size_t)i * dim,
                   Xb + (size_t)group_token_ids[off + i] * dim,
                   dim * sizeof(float));

        // Load expert weights (mmap: zero-copy pointer)
        const void *gate_data = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, map, e, 0);
        const void *up_data   = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, map, e, 1);
        const void *down_data = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, map, e, 2);
        if (!gate_data || !up_data || !down_data) continue;

        BnQWeight wgate, wup, wdown;
        if (!bn_moe_expert_projection_weight(&wgate, gate_data, map, 0) ||
            !bn_moe_expert_projection_weight(&wup, up_data, map, 1) ||
            !bn_moe_expert_projection_weight(&wdown, down_data, map, 2))
            continue;

        BnMoEPrefillProjection gate_prepared = {0};
        BnMoEPrefillProjection up_prepared = {0};
        BnMoEPrefillProjection down_prepared = {0};
        BnThreadPool *pool = bn_model_pool(m);
        const BnPreparedWeight *gate_prepared_ptr =
            moe_acquire_projection(&gate_prepared, m, &wgate, T, pool);
        const BnPreparedWeight *up_prepared_ptr =
            moe_acquire_projection(&up_prepared, m, &wup, T, pool);
        const BnPreparedWeight *down_prepared_ptr =
            moe_acquire_projection(&down_prepared, m, &wdown, T, pool);

        t0 = bn_moe_time_ms();
        int used_gpu_expert = 0;
        BnGPUBackend *gpu = prefill_gpu;
        if (bn_transformer_gpu_moe_prefill_single_expert_batch_available(
                gpu, T)) {
            BnGPUMoEExpertBatchPlan expert_plan = {
                n_tokens, n_experts, e, map->gate_up_fused};
            BnGPUMoETemporaryBuffers temps;
            BnGPUMoEExpertBuffers expert_gpu;
            memset(&temps, 0, sizeof(temps));
            memset(&expert_gpu, 0, sizeof(expert_gpu));
            if (bn_gpu_moe_bridge_get_expert(
                    m, sess, lw, l, e, &temps, &expert_gpu) == 0 &&
                expert_gpu.gate && expert_gpu.down &&
                (expert_gpu.up || expert_gpu.use_gateup_split) &&
                bn_transformer_gpu_prefill_expert_ffn_batch_backend_run(
                    gpu, down_buf, expert_gpu.gate,
                    expert_gpu.use_gateup_split ? NULL : expert_gpu.up,
                    expert_gpu.down, gather_buf, T, dim, moe_hidden,
                    routed_types.gate_type, routed_types.up_type,
                    routed_types.down_type,
                    exec_policy.activation, &expert_plan) == 0) {
                used_gpu_expert = 1;
                if (temps.n_buffers > 0)
                    bn_gpu_moe_bridge_release_temporaries(m, &temps);
            } else if (temps.n_buffers > 0) {
                bn_gpu_moe_bridge_release_temporaries(m, &temps);
            }
        }
        if (used_gpu_expert) {
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
            ms->stats.down_time_ms += 0.0;
        } else if (T == 1) {
            BnMatvecTask gu[2] = {
                 { gate_buf, &wgate, gate_prepared_ptr, 0 },
                 { up_buf,   &wup,   up_prepared_ptr, 0 },
            };
            bn_moe_quant_matvec_batch(gu, 2, gather_buf, x_q_scratch,
                                      bn_model_pool(m));
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            t0 = bn_moe_time_ms();
            bn_moe_swiglu(gate_buf, gate_buf, up_buf, T * moe_hidden,
                          exec_policy.uses_reference_silu,
                          exec_policy.uses_reference_ffn_activation);
            if (observe) {
                int tid = group_token_ids[off];
                int slot = -1;
                for (int k = 0; k < K; k++)
                    if (all_indices[(size_t)tid * K + k] == e) {
                        slot = k;
                        break;
                    }
                observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_ACTIVATION,
                        tid, slot, e, gate_buf, moe_hidden);
            }
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            t0 = bn_moe_time_ms();
            BnMatvecTask down[1] = {{
                down_buf, &wdown, down_prepared_ptr, 0
            }};
            bn_moe_quant_matvec_batch(down, 1, gate_buf, x_q_scratch,
                                      bn_model_pool(m));
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;
        } else {
            float *gu_out[2] = { gate_buf, up_buf };
            const BnQWeight *gu_weights[2] = { &wgate, &wup };
            const BnPreparedWeight *gu_prepared[2] = {
                gate_prepared_ptr, up_prepared_ptr
            };
            bn_moe_quant_matmul_prepared_multi(
                gu_out, gu_weights, gu_prepared, 2, gather_buf, T,
                x_q_scratch, pool);
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            t0 = bn_moe_time_ms();
            bn_moe_swiglu(gate_buf, gate_buf, up_buf, T * moe_hidden,
                          exec_policy.uses_reference_silu,
                          exec_policy.uses_reference_ffn_activation);
            if (observe) for (int i = 0; i < T; i++) {
                float *gate_t = gate_buf + (size_t)i * moe_hidden;
                int tid = group_token_ids[off + i];
                int slot = -1;
                for (int k = 0; k < K; k++)
                    if (all_indices[(size_t)tid * K + k] == e) {
                        slot = k;
                        break;
                    }
                observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_ACTIVATION,
                        tid, slot, e, gate_t, moe_hidden);
            }
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            t0 = bn_moe_time_ms();
            bn_moe_quant_matmul_prepared(
                down_buf, &wdown, down_prepared_ptr, gate_buf, T,
                x_q_scratch, pool);
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;
        }

        moe_release_projection(&gate_prepared, m);
        moe_release_projection(&up_prepared, m);
        moe_release_projection(&down_prepared, m);

        // Scatter-add with routing weights
        t0 = bn_moe_time_ms();
        for (int i = 0; i < T; i++) {
            int tid = group_token_ids[off + i];
            float w = group_weights[off + i];
            float *down_t = down_buf + (size_t)i * dim;
            float *out_t = NULL;
            for (int k = 0; k < K; k++)
                if (all_indices[(size_t)tid * K + k] == e) {
                    out_t = slot_out + ((size_t)tid * K + k) * dim;
                    break;
                }
            if (out_t && reduction_scales)
                memcpy(out_t, down_t, (size_t)dim*sizeof(float));
            if (!reduction_scales || observe)
                bn_moe_scale_expert_output(down_t,
                    bn_moe_expert_weight_scale(lw, e), dim);
            if (observe && out_t) {
                int slot = (int)((out_t - slot_out) / dim) - tid * K;
                observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_POST_NORM,
                        tid, slot, e, down_t, dim);
            }
            if (out_t && !reduction_scales) {
                bn_moe_scale_expert_output(down_t, w, dim);
                bn_moe_residual_add(out_t, down_t, dim);
            }
        }
        ms->stats.accum_time_ms += bn_moe_time_ms() - t0;
    }

    int used_gpu_reduction = 0;
    if (!used_gpu_moe_batch && reduction_scales) {
        t0 = bn_moe_time_ms();
        used_gpu_reduction = bn_transformer_gpu_moe_reduce_batch_backend_run(
            gpu_batch, moe_out, slot_out, all_weights, reduction_scales,
            n_tokens, K, dim) == 0;
        if (!used_gpu_reduction || observe)
            for (int t = 0; t < n_tokens; t++)
                for (int k = 0; k < K; k++) {
                    size_t row = (size_t)t*K + k;
                    float *slot = slot_out + row*dim;
                    memcpy(down_buf, slot, (size_t)dim*sizeof(float));
                    bn_moe_scale_expert_output(down_buf, reduction_scales[row], dim);
                    bn_moe_scale_expert_output(down_buf, all_weights[row], dim);
                    memset(slot, 0, (size_t)dim*sizeof(float));
                    bn_moe_residual_add(slot, down_buf, dim);
                }
        ms->stats.accum_time_ms += bn_moe_time_ms() - t0;
    }

    if (observe && !used_gpu_moe_batch)
        for (int t = 0; t < n_tokens; t++)
            for (int k = 0; k < K; k++)
                observe(observe_ctx,
                        BN_MOE_OBSERVE_ROUTED_WEIGHTED_EXPERT,
                        t, k, all_indices[(size_t)t * K + k],
                        slot_out + ((size_t)t * K + k) * dim, dim);


    if (!used_gpu_moe_batch && !used_gpu_reduction)
        for (int t = 0; t < n_tokens; t++)
            for (int k = 0; k < K; k++)
                bn_moe_residual_add(
                    moe_out + (size_t)t * dim,
                    slot_out + ((size_t)t * K + k) * dim, dim);

    if (observe)
        for (int t = 0; t < n_tokens; t++)
            observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_SUM,
                    t, -1, -1, moe_out + (size_t)t * dim, dim);

    // 6. Shared expert (if present) — batch matmul across all tokens
    if (!used_gpu_shared_batch &&
        shared_policy.has_loaded_path) {
        t0 = bn_moe_time_ms();
        int shared_hidden = shared_policy.hidden_dim;
        size_t sz_shared_gates = (size_t)n_tokens * sizeof(float);
        float *shared_gates = (float *)bn_malloc(&a, sz_shared_gates);
        int shared_gates_on_gpu = 0;
        if (shared_gates && gpu_batch) {
            int projected_on_gpu =
                bn_transformer_gpu_moe_shared_gate_batch(
                    gpu_batch, m, lw, l, shared_gates, Xb,
                    n_tokens, dim) == 0;
            if (!projected_on_gpu) {
                const float *gate_vector =
                    bn_moe_shared_expert_gate_vector(lw);
                if (gate_vector)
                    for (int t = 0; t < n_tokens; t++)
                        shared_gates[t] = bn_moe_dot_row(
                            gate_vector, Xb + (size_t)t * dim, dim);
            }
            if ((projected_on_gpu ||
                 bn_moe_shared_expert_gate_vector(lw)) &&
                bn_gpu_backend_sigmoid_batch(
                    gpu_batch, shared_gates, shared_gates, n_tokens) == 0)
                shared_gates_on_gpu = 1;
        }
        float *sh_gate = gate_buf;  // reuse (T_max >= 1, shared_hidden <= moe_hidden usually)
        float *sh_up = up_buf;
        float *sh_down = down_buf;

        // Need buffers sized for the full shared-expert batch. The reusable
        // routed-expert buffers are only sized by the largest routed group.
        // For safety, allocate if needed.
        int need_sh = ((size_t)n_tokens * shared_hidden > (size_t)T_max * moe_hidden ||
                       n_tokens > T_max);
        size_t sz_shg = (size_t)n_tokens * shared_hidden * sizeof(float);
        size_t sz_shd = (size_t)n_tokens * dim * sizeof(float);
        float *sh_g = need_sh ? (float *)bn_malloc(&a, sz_shg) : sh_gate;
        float *sh_u = need_sh ? (float *)bn_malloc(&a, sz_shg) : sh_up;
        float *sh_d = need_sh ? (float *)bn_malloc(&a, sz_shd) : sh_down;

        BnMoESharedExpertWeights shared_weights = {0};
        if (sh_g && sh_u && sh_d &&
            bn_moe_shared_expert_projection_weights(&shared_weights, lw)) {
            int used_gpu_shared = 0;
            BnGPUBackend *gpu = prefill_gpu;
            BnBackendModel *backend = bn_model_backend(m);
            if (bn_transformer_gpu_moe_prefill_shared_batch_available(
                    gpu, n_tokens, backend != NULL)) {
                BnTransformerGPUMoESharedFFNResources shared_gpu;
                if (bn_transformer_gpu_resolve_moe_shared_ffn_resources(
                        &shared_gpu, backend, c, lw, l, 1) &&
                    bn_transformer_gpu_prefill_dense_ffn_batch_backend_run(
                        gpu, sh_d, shared_gpu.gate, shared_gpu.up,
                        shared_gpu.down, Xb, n_tokens, dim, shared_hidden,
                        shared_gpu.gate_type, shared_gpu.up_type,
                        shared_gpu.down_type,
                        exec_policy.activation) == 0) {
                    used_gpu_shared = 1;
                }
            }

            if (!used_gpu_shared && prefill_policy.requires_matvec_prefill) {
                for (int t = 0; t < n_tokens; t++) {
                    const float *x_t = Xb + (size_t)t * dim;
                    float *gate_t = sh_g + (size_t)t * shared_hidden;
                    float *up_t = sh_u + (size_t)t * shared_hidden;
                    float *down_t = sh_d + (size_t)t * dim;
                    BnMatvecTask shared_gu[2];
                    int n_shared_gu = bn_moe_shared_expert_gateup_tasks(
                        shared_gu, gate_t, up_t, lw, 0);
                    if (n_shared_gu > 0)
                        bn_moe_quant_matvec_batch(shared_gu, n_shared_gu, x_t,
                                                  x_q_scratch, bn_model_pool(m));
                    bn_moe_swiglu(gate_t, gate_t, up_t, shared_hidden,
                                  exec_policy.uses_reference_silu,
                                  exec_policy.uses_reference_ffn_activation);
                    bn_moe_quant_matvec(down_t, shared_weights.down,
                                        gate_t, x_q_scratch, bn_model_pool(m));
                }
            } else if (!used_gpu_shared) {
                bn_moe_quant_matmul(sh_g, shared_weights.gate, Xb,
                                    n_tokens, x_q_scratch, bn_model_pool(m));
                bn_moe_quant_matmul(sh_u, shared_weights.up, Xb,
                                    n_tokens, x_q_scratch, bn_model_pool(m));

                size_t sh_total = (size_t)n_tokens * shared_hidden;
                bn_moe_swiglu(sh_g, sh_g, sh_u, (int)sh_total,
                              exec_policy.uses_reference_silu,
                              exec_policy.uses_reference_ffn_activation);

                bn_moe_quant_matmul(sh_d, shared_weights.down, sh_g,
                                    n_tokens, x_q_scratch, bn_model_pool(m));
            }

            for (int t = 0; t < n_tokens; t++) {
                const float *x_t = Xb + (size_t)t * dim;
                float gate = shared_gates_on_gpu
                    ? shared_gates[t]
                    : bn_moe_shared_expert_gate_weight(lw, x_t, dim);
                if (observe) {
                    float gate_logit = shared_gates_on_gpu
                        ? bn_moe_dot_row(
                            bn_moe_shared_expert_gate_vector(lw), x_t, dim)
                        : bn_moe_dot_row(
                            bn_moe_shared_expert_gate_vector(lw), x_t, dim);
                    observe(observe_ctx, BN_MOE_OBSERVE_SHARED_OUTPUT,
                            t, -1, -1, sh_d + (size_t)t * dim, dim);
                    observe(observe_ctx, BN_MOE_OBSERVE_SHARED_GATE,
                            t, -1, -1, &gate, 1);
                    observe(observe_ctx, BN_MOE_OBSERVE_SHARED_GATE_LOGIT,
                            t, -1, -1, &gate_logit, 1);
                }
                bn_moe_weighted_add(moe_out + (size_t)t * dim,
                                    sh_d + (size_t)t * dim, gate, dim);
            }
        } else if (need_sh) {
            SH_LOG_ERROR("Failed to allocate shared expert batch buffers");
        }

        if (shared_gates)
            bn_free(&a, shared_gates, sz_shared_gates);

        if (need_sh) {
            if (sh_g) bn_free(&a, sh_g, sz_shg);
            if (sh_u) bn_free(&a, sh_u, sz_shg);
            if (sh_d) bn_free(&a, sh_d, sz_shd);
        }
        ms->stats.shared_time_ms += bn_moe_time_ms() - t0;
    }

    int result = 0;
    int composed_on_gpu = 0;
    // 7. Return either the raw block output or its residual sum.
    if (observe)
        for (int t = 0; t < n_tokens; t++)
            observe(observe_ctx, BN_MOE_OBSERVE_ROUTED_OUTPUT, t, -1, -1,
                    moe_out + (size_t)t * dim, dim);
    if (exec_policy.uses_dense_residual_branch) {
        composed_on_gpu = !force_host &&
            moe_batch_dense_gpu(m, l, lw, &exec_policy,
                act, Xb, moe_out, n_tokens, flags & BN_MOE_BATCH_OUTPUT_RAW,
                observe, observe_ctx) == 0;
        if (!composed_on_gpu &&
            moe_batch_dense_residual(m, lw, &exec_policy, act, Xb, moe_out,
                                     n_tokens, observe, observe_ctx) != 0) {
            result = -1;
            goto cleanup;
        }
    }
    if (observe)
        for (int t = 0; t < n_tokens; t++)
            observe(observe_ctx, composed_on_gpu && !(flags & BN_MOE_BATCH_OUTPUT_RAW)
                        ? BN_MOE_OBSERVE_RESIDUAL_OUTPUT : BN_MOE_OBSERVE_FINAL_OUTPUT,
                    t, -1, -1,
                    moe_out + (size_t)t * dim, dim);
    if (composed_on_gpu || (flags & BN_MOE_BATCH_OUTPUT_RAW))
        memcpy(act, moe_out, (size_t)n_tokens * dim * sizeof(float));
    else
        for (int t = 0; t < n_tokens; t++)
            bn_moe_residual_add(act + (size_t)t * dim,
                                moe_out + (size_t)t * dim, dim);

cleanup:
    // Cleanup
    bn_free(&a, gather_buf, sz_gather); bn_free(&a, gate_buf, sz_gate);
    bn_free(&a, up_buf, sz_up); bn_free(&a, down_buf, sz_down);
    bn_free(&a, moe_out, sz_mout); bn_free(&a, x_q_scratch, sz_xq);
    bn_free(&a, slot_out, sz_slots);
    if (reduction_scales) bn_free(&a, reduction_scales, sz_wts);
    bn_free(&a, batch_logits, sz_logits);
    bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
    bn_free(&a, expert_counts, sz_ecnt); bn_free(&a, expert_offsets, sz_ecnt);
    bn_free(&a, group_token_ids, sz_gtid); bn_free(&a, group_weights, sz_gwts);
    bn_free(&a, fill_pos, sz_fill);

    ms->stats.compute_time_ms += bn_moe_time_ms() - t_compute;
    return result;
}

int bn_moe_forward_batch(struct BnModel *m, BnSession *sess,
                         struct BnLayerWeights *lw, int l,
                         float *act, float *Xb, int n_tokens) {
    return bn_moe_forward_batch_observed(
        m, sess, lw, l, act, Xb, n_tokens, 0, NULL, NULL);
}
