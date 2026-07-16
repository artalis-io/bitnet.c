#include "moe_internal.h"
#include "backend_model.h"
#include "gpu_backend.h"
#include "gpu_moe_bridge.h"
#include "transformer/gpu_internal.h"

typedef struct {
    float *logits;
    const float *router_w;
    const float *x;
    int dim;
    int n_experts;
} BnMoEBatchRouterCtx;

static void moe_batch_router_range(void *ctx, int start, int end) {
    BnMoEBatchRouterCtx *c = (BnMoEBatchRouterCtx *)ctx;
    for (int idx = start; idx < end; idx++) {
        int t = idx / c->n_experts;
        int e = idx - t * c->n_experts;
        const float *row = c->router_w + (size_t)e * c->dim;
        const float *x = c->x + (size_t)t * c->dim;
        c->logits[(size_t)t * c->n_experts + e] =
            bn_moe_dot_row(row, x, c->dim);
    }
}

static void moe_batch_route(float *logits, int *all_indices, float *all_weights,
                            const float *x, const float *router_w,
                            int n_tokens, int dim, int n_experts, int k,
                            int norm_topk_prob, float expert_weights_scale,
                            BnThreadPool *pool) {
    BnMoEBatchRouterCtx rctx = { logits, router_w, x, dim, n_experts };
    BnTPTask rtask = { moe_batch_router_range, &rctx, n_tokens * n_experts };
    bn_tp_dispatch(pool, &rtask, 1);

    for (int t = 0; t < n_tokens; t++) {
        float *row = logits + (size_t)t * n_experts;
        float max_val = row[0];
        if (!isfinite(max_val))
            max_val = -INFINITY;
        for (int e = 1; e < n_experts; e++) {
            if (!isfinite(row[e]))
                row[e] = -INFINITY;
            if (row[e] > max_val) max_val = row[e];
        }
        if (!isfinite(max_val))
            max_val = 0.0f;

        float sum = 0.0f;
        for (int e = 0; e < n_experts; e++) {
            row[e] = isfinite(row[e]) ? expf(row[e] - max_val) : 0.0f;
            sum += row[e];
        }

        int picked = 0;
        for (int i = 0; i < k; i++) {
            int best = -1;
            float best_val = -1.0f;
            for (int e = 0; e < n_experts; e++) {
                if (row[e] > best_val) {
                    best_val = row[e];
                    best = e;
                }
            }
            if (best < 0 || best_val <= 0.0f || sum <= 0.0f) {
                all_indices[(size_t)t * k + i] = -1;
                all_weights[(size_t)t * k + i] = 0.0f;
                continue;
            }
            all_indices[(size_t)t * k + i] = best;
            all_weights[(size_t)t * k + i] = best_val / sum;
            row[best] = -1.0f;
            picked++;
        }

        float wsum = 0.0f;
        for (int i = 0; i < k; i++)
            wsum += all_weights[(size_t)t * k + i];
        if (norm_topk_prob && wsum > 0.0f)
            for (int i = 0; i < k; i++)
                all_weights[(size_t)t * k + i] /= wsum;
        else if (picked == 0 && n_experts > 0) {
            all_indices[(size_t)t * k] = 0;
            all_weights[(size_t)t * k] = 1.0f;
        }
        if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
            for (int i = 0; i < k; i++)
                all_weights[(size_t)t * k + i] *= expert_weights_scale;
        }
    }
}

// --- Batch MoE FFN for prefill ---
// Route all n_tokens, group by expert, batch matmul per expert.
int bn_moe_forward_batch(struct BnModel *m, BnSession *sess,
                          struct BnLayerWeights *lw, int l,
                          float *act, float *Xb, int n_tokens) {
    double t_compute = bn_moe_time_ms();
    double t0;
    BnConfig *c = &m->config;
    BnMoEState *ms = sess->moe_state;
    int dim = c->dim;
    int moe_hidden = c->moe_intermediate_size;
    int K = c->n_experts_active;
    int n_experts = c->n_experts;
    int force_matvec_prefill =
        n_experts == 2 && K == 2 && c->has_shared_expert;
    int allow_cuda_route_routed_batch =
        bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(n_experts);
    const BnMoEExpertMap *map = &lw->moe.expert_map;
    int cuda_all2_q4q6_requires_opt_in =
        bn_transformer_gpu_all2_q4_moe_requires_opt_in(c, map, dim, 0);

    BnAllocator a = bn_allocator_default();
    int did_input_norm = 0;
    if (n_experts > 2) {
        BnGPUBackend *gpu = bn_model_gpu(m);
        BnBackendModel *backend = bn_model_backend(m);
        if (allow_cuda_route_routed_batch &&
            bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
            gpu->moe_route_routed_ffn_batch_norm_resid && backend) {
            void *router = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_ROUTER);
            void *gate_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_GATE_ALL);
            void *up_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_UP_ALL);
            void *down_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
            void *norm = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_FFN_NORM);
            void *shared_gate = NULL;
            void *shared_up = NULL;
            void *shared_down = NULL;
            void *shared_gate_weight = NULL;
            int shared_hidden = 0;
            int shared_gate_type = 0;
            int shared_up_type = 0;
            int shared_down_type = 0;
            if (c->has_shared_expert && lw->shared.shared_gate.data) {
                shared_gate = bn_backend_model_qweight_buf(
                    backend, &lw->shared.shared_gate);
                shared_up = bn_backend_model_qweight_buf(
                    backend, &lw->shared.shared_up);
                shared_down = bn_backend_model_qweight_buf(
                    backend, &lw->shared.shared_down);
                shared_gate_weight = bn_backend_model_handle(
                    backend, l, BN_BACKEND_HANDLE_SHARED_EXPERT_GATE);
                if (shared_gate && shared_up && shared_down) {
                    shared_hidden = c->shared_expert_intermediate_size;
                    shared_gate_type = lw->shared.shared_gate.type;
                    shared_up_type = lw->shared.shared_up.type;
                    shared_down_type = lw->shared.shared_down.type;
                } else {
                    shared_gate = NULL;
                    shared_up = NULL;
                    shared_down = NULL;
                    shared_gate_weight = NULL;
                }
            }
            if (router && gate_all && up_all && down_all && norm) {
                t0 = bn_moe_time_ms();
                if (gpu->moe_route_routed_ffn_batch_norm_resid(
                        gpu->ctx, act, router, gate_all, up_all, down_all,
                        shared_gate, shared_up, shared_down,
                        shared_gate_weight, norm, act, n_tokens, dim,
                        moe_hidden, n_experts, K,
                        map->gate_type, map->up_type, map->down_type,
                        c->act_type, shared_hidden, shared_gate_type,
                        shared_up_type, shared_down_type,
                        c->norm_eps, c->moe_norm_topk_prob,
                        c->moe_expert_weights_scale) == 0) {
                    ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
                    ms->stats.compute_time_ms +=
                        bn_moe_time_ms() - t_compute;
                    return 0;
                }
            }
        }

        // Batch RMSNorm for the host/GPU split MoE prefill paths below.
        t0 = bn_moe_time_ms();
        for (int t = 0; t < n_tokens; t++)
            bn_moe_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                           lw->norm.ffn_norm, dim, c->norm_eps);
        ms->stats.norm_time_ms += bn_moe_time_ms() - t0;
        did_input_norm = 1;

        if (!cuda_all2_q4q6_requires_opt_in &&
            allow_cuda_route_routed_batch &&
            bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
            gpu->moe_route_routed_ffn_batch && backend) {
            void *router = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_ROUTER);
            void *gate_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_GATE_ALL);
            void *up_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_UP_ALL);
            void *down_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
            size_t sz_mout = (size_t)n_tokens * dim * sizeof(float);
            float *moe_out = (float *)bn_malloc(&a, sz_mout);
            if (moe_out && router && gate_all && up_all && down_all) {
                t0 = bn_moe_time_ms();
                if (gpu->moe_route_routed_ffn_batch(
                        gpu->ctx, moe_out, router, gate_all, up_all, down_all,
                        Xb, n_tokens, dim, moe_hidden, n_experts, K,
                        map->gate_type, map->up_type, map->down_type,
                        c->act_type, c->moe_norm_topk_prob,
                        c->moe_expert_weights_scale) == 0) {
                    ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
                    int shared_ok = 1;
                    if (c->has_shared_expert && lw->shared.shared_gate.data) {
                        shared_ok = 0;
                        int shared_hidden = c->shared_expert_intermediate_size;
                        size_t sz_shd = (size_t)n_tokens * dim * sizeof(float);
                        float *sh_d = (float *)bn_malloc(&a, sz_shd);
                        void *sg = bn_backend_model_qweight_buf(
                            backend, &lw->shared.shared_gate);
                        void *su = bn_backend_model_qweight_buf(
                            backend, &lw->shared.shared_up);
                        void *sd = bn_backend_model_qweight_buf(
                            backend, &lw->shared.shared_down);
                        if (sh_d && gpu->dense_ffn_batch && sg && su && sd) {
                            double ts = bn_moe_time_ms();
                            if (gpu->dense_ffn_batch(
                                    gpu->ctx, sh_d, sg, su, sd, Xb,
                                    n_tokens, dim, shared_hidden,
                                    lw->shared.shared_gate.type,
                                    lw->shared.shared_up.type,
                                    lw->shared.shared_down.type,
                                    c->act_type) == 0) {
                                if (lw->shared.shared_expert_gate) {
                                    for (int t = 0; t < n_tokens; t++) {
                                        float gate_dot = 0.0f;
                                        for (int d = 0; d < dim; d++)
                                            gate_dot += Xb[(size_t)t * dim + d] *
                                                        lw->shared.shared_expert_gate[d];
                                        float gate = 1.0f / (1.0f + expf(-gate_dot));
                                        for (int d = 0; d < dim; d++)
                                            moe_out[(size_t)t * dim + d] +=
                                                gate * sh_d[(size_t)t * dim + d];
                                    }
                                } else {
                                    for (int t = 0; t < n_tokens; t++)
                                        for (int d = 0; d < dim; d++)
                                            moe_out[(size_t)t * dim + d] +=
                                                sh_d[(size_t)t * dim + d];
                                }
                                ms->stats.shared_time_ms += bn_moe_time_ms() - ts;
                                shared_ok = 1;
                            }
                        }
                        if (sh_d)
                            bn_free(&a, sh_d, sz_shd);
                    }
                    if (shared_ok) {
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
        for (int t = 0; t < n_tokens; t++)
            bn_moe_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                           lw->norm.ffn_norm, dim, c->norm_eps);
        ms->stats.norm_time_ms += bn_moe_time_ms() - t0;
    }

    // 2. Batch routing: route each token individually (reuse existing router)
    // Allocate routing results: [n_tokens][K] indices and weights
    size_t sz_idx = (size_t)n_tokens * K * sizeof(int);
    size_t sz_wts = (size_t)n_tokens * K * sizeof(float);
    size_t sz_logits = (size_t)n_tokens * n_experts * sizeof(float);
    int *all_indices = (int *)bn_malloc(&a, sz_idx);
    float *all_weights = (float *)bn_malloc(&a, sz_wts);
    float *batch_logits = (float *)bn_malloc(&a, sz_logits);
    if (!all_indices || !all_weights || !batch_logits) {
        if (all_indices) bn_free(&a, all_indices, sz_idx);
        if (all_weights) bn_free(&a, all_weights, sz_wts);
        if (batch_logits) bn_free(&a, batch_logits, sz_logits);
        return -1;
    }
    memset(all_indices, 0, sz_idx);
    memset(all_weights, 0, sz_wts);

    t0 = bn_moe_time_ms();
    int used_gpu_route = 0;
    BnGPUBackend *route_gpu = bn_model_gpu(m);
    BnBackendModel *route_backend = bn_model_backend(m);
    if (n_experts > 2 && route_gpu &&
        bn_transformer_gpu_moe_prefill_backend_available(route_gpu) &&
        route_gpu->moe_route_batch && route_backend) {
        void *router_gpu = bn_backend_model_handle(
            route_backend, l, BN_BACKEND_HANDLE_MOE_ROUTER);
        int route_rc = router_gpu
            ? route_gpu->moe_route_batch(route_gpu->ctx, all_indices,
                                         all_weights, router_gpu, Xb,
                                         n_tokens, dim, n_experts, K,
                                         c->moe_norm_topk_prob,
                                         c->moe_expert_weights_scale)
            : -1;
        if (route_rc == 0)
            used_gpu_route = 1;
        else if (bn_transformer_gpu_cuda_moe_route_batch_debug_enabled())
            fprintf(stderr,
                    "[bn:cuda:moe-route-batch] fallback layer=%d handle=%d rc=%d tokens=%d experts=%d k=%d dim=%d\n",
                    l, router_gpu != NULL, route_rc, n_tokens, n_experts, K,
                    dim);
    }
    if (!used_gpu_route) {
        moe_batch_route(batch_logits, all_indices, all_weights, Xb,
                        lw->moe.router_weight, n_tokens, dim, n_experts, K,
                        c->moe_norm_topk_prob,
                        c->moe_expert_weights_scale, bn_model_pool(m));
    }
    ms->stats.route_time_ms += bn_moe_time_ms() - t0;

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
    size_t sz_xq     = (size_t)T_max * (size_t)(dim > moe_hidden ? dim : moe_hidden);
    float *gather_buf   = (float *)bn_malloc(&a, sz_gather);
    float *gate_buf     = (float *)bn_malloc(&a, sz_gate);
    float *up_buf       = (float *)bn_malloc(&a, sz_up);
    float *down_buf     = (float *)bn_malloc(&a, sz_down);
    float *moe_out      = (float *)bn_malloc(&a, sz_mout);
    int8_t *x_q_scratch = (int8_t *)bn_malloc(&a, sz_xq);
    if (!gather_buf || !gate_buf || !up_buf || !down_buf || !moe_out || !x_q_scratch) {
        if (gather_buf) bn_free(&a, gather_buf, sz_gather);
        if (gate_buf) bn_free(&a, gate_buf, sz_gate);
        if (up_buf) bn_free(&a, up_buf, sz_up);
        if (down_buf) bn_free(&a, down_buf, sz_down);
        if (moe_out) bn_free(&a, moe_out, sz_mout);
        if (x_q_scratch) bn_free(&a, x_q_scratch, sz_xq);
        bn_free(&a, group_token_ids, sz_gtid); bn_free(&a, group_weights, sz_gwts);
        bn_free(&a, fill_pos, sz_fill); bn_free(&a, expert_counts, sz_ecnt);
        bn_free(&a, expert_offsets, sz_ecnt);
        bn_free(&a, batch_logits, sz_logits);
        bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
        return -1;
    }
    memset(moe_out, 0, sz_mout);

    // 5. Per-expert batch compute
    int used_gpu_moe_batch = 0;
    int used_gpu_shared_batch = 0;
    BnGPUBackend *gpu_batch = bn_model_gpu(m);
    int prefer_cached_expert_batch =
        bn_model_gpu_moe_cache(m) != NULL &&
        n_experts == 2 && K == 2 &&
        bn_transformer_gpu_cuda_moe_cache_prefill_enabled();
    if (!cuda_all2_q4q6_requires_opt_in &&
        !prefer_cached_expert_batch &&
        bn_transformer_gpu_moe_prefill_backend_available(gpu_batch) &&
        gpu_batch->moe_routed_ffn_batch) {
        const BnBackendModel *backend = bn_model_backend(m);
        void *gate_all = bn_backend_model_handle(
            backend, l, BN_BACKEND_HANDLE_MOE_GATE_ALL);
        void *up_all = bn_backend_model_handle(
            backend, l, BN_BACKEND_HANDLE_MOE_UP_ALL);
        void *down_all = bn_backend_model_handle(
            backend, l, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
        if (gate_all && up_all && down_all) {
            t0 = bn_moe_time_ms();
            if (gpu_batch->moe_routed_ffn_batch(
                    gpu_batch->ctx, moe_out, gate_all, up_all, down_all,
                    all_indices, all_weights, Xb, n_tokens, dim,
                    moe_hidden, n_experts, K, map->gate_type,
                    map->up_type, map->down_type, c->act_type) == 0) {
                used_gpu_moe_batch = 1;
                ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
            }
        }
    }
    if (!cuda_all2_q4q6_requires_opt_in &&
        !used_gpu_moe_batch && gpu_batch &&
        bn_transformer_gpu_moe_prefill_backend_available(gpu_batch) &&
        gpu_batch->moe_ffn_batch) {
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
                void *shared_gate = NULL;
                void *shared_up = NULL;
                void *shared_down = NULL;
                void *shared_gate_weight = NULL;
                int shared_hidden = 0;
                int shared_gate_type = 0;
                int shared_up_type = 0;
                int shared_down_type = 0;
                int can_fuse_shared =
                    c->has_shared_expert && lw->shared.shared_gate.data;
                if (can_fuse_shared && backend &&
                    bn_transformer_gpu_cuda_moe_prefill_shared_fuse_enabled()) {
                    shared_gate = bn_backend_model_handle(
                        backend, l, BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED);
                    if (shared_gate) {
                        shared_up = NULL;
                    } else {
                        shared_gate = bn_backend_model_qweight_buf(
                            backend, &lw->shared.shared_gate);
                        shared_up = bn_backend_model_qweight_buf(
                            backend, &lw->shared.shared_up);
                    }
                    shared_down = bn_backend_model_qweight_buf(
                        backend, &lw->shared.shared_down);
                    shared_gate_weight = bn_backend_model_handle(
                        backend, l, BN_BACKEND_HANDLE_SHARED_EXPERT_GATE);
                    if (shared_gate && shared_down &&
                        (shared_up || bn_backend_model_handle(
                            backend, l,
                            BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED))) {
                        shared_hidden = c->shared_expert_intermediate_size;
                        shared_gate_type = lw->shared.shared_gate.type;
                        shared_up_type = lw->shared.shared_up.type;
                        shared_down_type = lw->shared.shared_down.type;
                    } else {
                        shared_gate = NULL;
                        shared_up = NULL;
                        shared_down = NULL;
                        shared_gate_weight = NULL;
                    }
                }
                t0 = bn_moe_time_ms();
                if (gpu_batch->moe_ffn_batch(
                        gpu_batch->ctx, moe_out, gpu_experts, n_experts,
                        expert_offsets, expert_counts, group_token_ids,
                        group_weights, Xb, n_tokens, dim, moe_hidden,
                        map->gate_type, map->up_type, map->down_type,
                        c->act_type, shared_gate, shared_up, shared_down,
                        shared_gate_weight, shared_hidden, shared_gate_type,
                        shared_up_type, shared_down_type) == 0) {
                    used_gpu_moe_batch = 1;
                    used_gpu_shared_batch =
                        shared_gate && shared_down &&
                        (shared_up || bn_backend_model_handle(
                            backend, l,
                            BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED));
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

        BnQWeight wgate = bn_moe_make_qweight(gate_data, map->gate_type,
                                            map->gate_rows, map->gate_cols);
        BnQWeight wup   = bn_moe_make_qweight(up_data, map->up_type,
                                            map->up_rows, map->up_cols);
        BnQWeight wdown = bn_moe_make_qweight(down_data, map->down_type,
                                            map->down_rows, map->down_cols);

        t0 = bn_moe_time_ms();
        int used_gpu_expert = 0;
        BnGPUBackend *gpu = bn_model_gpu(m);
        if (bn_transformer_gpu_moe_prefill_tokens_allowed(gpu, T) &&
            gpu->dense_ffn_batch) {
            BnGPUMoETemporaryBuffers temps;
            BnGPUMoEExpertBuffers expert_gpu;
            memset(&temps, 0, sizeof(temps));
            memset(&expert_gpu, 0, sizeof(expert_gpu));
            if (bn_gpu_moe_bridge_get_expert(
                    m, sess, lw, l, e, &temps, &expert_gpu) == 0 &&
                expert_gpu.gate && expert_gpu.down &&
                (expert_gpu.up || expert_gpu.use_gateup_split) &&
                gpu->dense_ffn_batch(
                    gpu->ctx, down_buf, expert_gpu.gate,
                    expert_gpu.use_gateup_split ? NULL : expert_gpu.up,
                    expert_gpu.down, gather_buf, T, dim, moe_hidden,
                    map->gate_type, map->up_type, map->down_type,
                    c->act_type) == 0) {
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
            // Single token: use matvec (less overhead)
            BnMatvecTask gu[2] = {
                 { gate_buf, &wgate, NULL, 0 },
                 { up_buf,   &wup  , NULL, 0 },
            };
            bn_quant_matvec_batch(gu, 2, gather_buf, x_q_scratch, bn_model_pool(m));
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            t0 = bn_moe_time_ms();
            for (int i = 0; i < moe_hidden; i++) {
                float g = gate_buf[i];
                gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
            }
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            t0 = bn_moe_time_ms();
            bn_quant_matvec(down_buf, &wdown, gate_buf, x_q_scratch,
                            bn_model_pool(m));
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;
        } else if (force_matvec_prefill) {
            for (int i = 0; i < T; i++) {
                float *x_t = gather_buf + (size_t)i * dim;
                float *gate_t = gate_buf + (size_t)i * moe_hidden;
                float *up_t = up_buf + (size_t)i * moe_hidden;
                float *down_t = down_buf + (size_t)i * dim;
                BnMatvecTask gu[2] = {
                     { gate_t, &wgate, NULL, 0 },
                     { up_t,   &wup,   NULL, 0 },
                };
                bn_quant_matvec_batch(gu, 2, x_t, x_q_scratch,
                                      bn_model_pool(m));
                for (int h = 0; h < moe_hidden; h++) {
                    float g = gate_t[h];
                    gate_t[h] = (g / (1.0f + expf(-g))) * up_t[h];
                }
                bn_quant_matvec(down_t, &wdown, gate_t, x_q_scratch,
                                bn_model_pool(m));
            }
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;
        } else {
            bn_quant_matmul(gate_buf, &wgate, gather_buf, T, x_q_scratch, bn_model_pool(m));
            bn_quant_matmul(up_buf, &wup, gather_buf, T, x_q_scratch, bn_model_pool(m));
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            // SwiGLU activation across T * moe_hidden
            t0 = bn_moe_time_ms();
            {
                size_t swiglu_n = (size_t)T * moe_hidden;
                for (size_t i = 0; i < swiglu_n; i++) {
                    float g = gate_buf[i];
                    gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
                }
            }
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            // Down matmul
            t0 = bn_moe_time_ms();
            bn_quant_matmul(down_buf, &wdown, gate_buf, T, x_q_scratch, bn_model_pool(m));
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;
        }

        // Scatter-add with routing weights
        t0 = bn_moe_time_ms();
        for (int i = 0; i < T; i++) {
            int tid = group_token_ids[off + i];
            float w = group_weights[off + i];
            float *out_t = moe_out + (size_t)tid * dim;
            float *down_t = down_buf + (size_t)i * dim;
            for (int d = 0; d < dim; d++)
                out_t[d] += w * down_t[d];
        }
        ms->stats.accum_time_ms += bn_moe_time_ms() - t0;
    }

    // 6. Shared expert (if present) — batch matmul across all tokens
    if (!used_gpu_shared_batch &&
        c->has_shared_expert && lw->shared.shared_gate.data) {
        t0 = bn_moe_time_ms();
        int shared_hidden = c->shared_expert_intermediate_size;
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

        if (sh_g && sh_u && sh_d) {
            int used_gpu_shared = 0;
            BnGPUBackend *gpu = bn_model_gpu(m);
            BnBackendModel *backend = bn_model_backend(m);
            if (bn_transformer_gpu_moe_prefill_tokens_allowed(gpu, n_tokens) &&
                gpu->dense_ffn_batch && backend) {
                void *gate_gpu = bn_backend_model_handle(
                    backend, l, BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED);
                void *up_gpu = NULL;
                if (!gate_gpu) {
                    gate_gpu = bn_backend_model_qweight_buf(
                        backend, &lw->shared.shared_gate);
                    up_gpu = bn_backend_model_qweight_buf(
                        backend, &lw->shared.shared_up);
                }
                void *down_gpu = bn_backend_model_qweight_buf(
                    backend, &lw->shared.shared_down);
                if (gate_gpu && down_gpu &&
                    (up_gpu || bn_backend_model_handle(
                        backend, l,
                        BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED)) &&
                    gpu->dense_ffn_batch(
                        gpu->ctx, sh_d, gate_gpu, up_gpu, down_gpu, Xb,
                        n_tokens, dim, shared_hidden,
                        lw->shared.shared_gate.type,
                        lw->shared.shared_up.type,
                        lw->shared.shared_down.type,
                        c->act_type) == 0) {
                    used_gpu_shared = 1;
                }
            }

            if (!used_gpu_shared && force_matvec_prefill) {
                for (int t = 0; t < n_tokens; t++) {
                    const float *x_t = Xb + (size_t)t * dim;
                    float *gate_t = sh_g + (size_t)t * shared_hidden;
                    float *up_t = sh_u + (size_t)t * shared_hidden;
                    float *down_t = sh_d + (size_t)t * dim;
                    BnMatvecTask shared_gu[2] = {
                         { gate_t, &lw->shared.shared_gate, NULL, 0 },
                         { up_t,   &lw->shared.shared_up,   NULL, 0 },
                    };
                    bn_quant_matvec_batch(shared_gu, 2, x_t, x_q_scratch,
                                          bn_model_pool(m));
                    for (int h = 0; h < shared_hidden; h++) {
                        float g = gate_t[h];
                        gate_t[h] = (g / (1.0f + expf(-g))) * up_t[h];
                    }
                    bn_quant_matvec(down_t, &lw->shared.shared_down, gate_t,
                                    x_q_scratch, bn_model_pool(m));
                }
            } else if (!used_gpu_shared) {
                bn_quant_matmul(sh_g, &lw->shared.shared_gate, Xb, n_tokens, x_q_scratch, bn_model_pool(m));
                bn_quant_matmul(sh_u, &lw->shared.shared_up, Xb, n_tokens, x_q_scratch, bn_model_pool(m));

                size_t sh_total = (size_t)n_tokens * shared_hidden;
                for (size_t i = 0; i < sh_total; i++) {
                    float g = sh_g[i];
                    sh_g[i] = (g / (1.0f + expf(-g))) * sh_u[i];
                }

                bn_quant_matmul(sh_d, &lw->shared.shared_down, sh_g, n_tokens, x_q_scratch, bn_model_pool(m));
            }

            // Apply shared expert sigmoid gate if present
            if (lw->shared.shared_expert_gate) {
                for (int t = 0; t < n_tokens; t++) {
                    float gate_dot = 0.0f;
                    for (int d = 0; d < dim; d++)
                        gate_dot += Xb[(size_t)t * dim + d] * lw->shared.shared_expert_gate[d];
                    float gate = 1.0f / (1.0f + expf(-gate_dot));
                    for (int d = 0; d < dim; d++)
                        moe_out[(size_t)t * dim + d] += gate * sh_d[(size_t)t * dim + d];
                }
            } else {
                for (int t = 0; t < n_tokens; t++)
                    for (int d = 0; d < dim; d++)
                        moe_out[(size_t)t * dim + d] += sh_d[(size_t)t * dim + d];
            }
        } else if (need_sh) {
            SH_LOG_ERROR("Failed to allocate shared expert batch buffers");
        }

        if (need_sh) {
            if (sh_g) bn_free(&a, sh_g, sz_shg);
            if (sh_u) bn_free(&a, sh_u, sz_shg);
            if (sh_d) bn_free(&a, sh_d, sz_shd);
        }
        ms->stats.shared_time_ms += bn_moe_time_ms() - t0;
    }

    // 7. Residual add: act += moe_out
    for (int t = 0; t < n_tokens; t++)
        for (int d = 0; d < dim; d++)
            act[(size_t)t * dim + d] += moe_out[(size_t)t * dim + d];

    // Cleanup
    bn_free(&a, gather_buf, sz_gather); bn_free(&a, gate_buf, sz_gate);
    bn_free(&a, up_buf, sz_up); bn_free(&a, down_buf, sz_down);
    bn_free(&a, moe_out, sz_mout); bn_free(&a, x_q_scratch, sz_xq);
    bn_free(&a, batch_logits, sz_logits);
    bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
    bn_free(&a, expert_counts, sz_ecnt); bn_free(&a, expert_offsets, sz_ecnt);
    bn_free(&a, group_token_ids, sz_gtid); bn_free(&a, group_weights, sz_gwts);
    bn_free(&a, fill_pos, sz_fill);

    ms->stats.compute_time_ms += bn_moe_time_ms() - t_compute;
    return 0;
}
