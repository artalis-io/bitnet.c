#include "moe_internal.h"

// --- Batch MoE FFN for prefill ---
// Route all n_tokens, group by expert, batch matmul per expert.
int bn_moe_forward_batch(struct BnModel *m, BnSession *sess,
                          struct BnLayerWeights *lw, int l,
                          float *act, float *Xb, int n_tokens) {
    (void)l;  // reserved for pread cache keying in future
    BnConfig *c = &m->config;
    BnMoEState *ms = sess->moe_state;
    int dim = c->dim;
    int moe_hidden = c->moe_intermediate_size;
    int K = c->n_experts_active;
    int n_experts = c->n_experts;
    const BnMoEExpertMap *map = &lw->moe.expert_map;

    // 1. Batch RMSNorm
    for (int t = 0; t < n_tokens; t++)
        bn_moe_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                    lw->norm.ffn_norm, dim, c->norm_eps);

    // 2. Batch routing: route each token individually (reuse existing router)
    // Allocate routing results: [n_tokens][K] indices and weights
    BnAllocator a = bn_allocator_default();
    size_t sz_idx = (size_t)n_tokens * K * sizeof(int);
    size_t sz_wts = (size_t)n_tokens * K * sizeof(float);
    int *all_indices = (int *)bn_malloc(&a, sz_idx);
    float *all_weights = (float *)bn_malloc(&a, sz_wts);
    if (!all_indices || !all_weights) {
        if (all_indices) bn_free(&a, all_indices, sz_idx);
        if (all_weights) bn_free(&a, all_weights, sz_wts);
        return -1;
    }
    memset(all_indices, 0, sz_idx);
    memset(all_weights, 0, sz_wts);

    for (int t = 0; t < n_tokens; t++) {
        bn_moe_route(ms, Xb + (size_t)t * dim, lw->moe.router_weight,
                     dim, n_experts, K, bn_model_pool(m));
        memcpy(all_indices + (size_t)t * K, ms->expert_indices, (size_t)K * sizeof(int));
        memcpy(all_weights + (size_t)t * K, ms->expert_weights, (size_t)K * sizeof(float));
    }

    // 3. Build token-expert grouping (two-pass)
    // Pass 1: count tokens per expert
    size_t sz_ecnt = (size_t)n_experts * sizeof(int);
    int *expert_counts = (int *)bn_malloc(&a, sz_ecnt);
    int *expert_offsets = (int *)bn_malloc(&a, sz_ecnt);
    if (!expert_counts || !expert_offsets) {
        if (expert_counts) bn_free(&a, expert_counts, sz_ecnt);
        if (expert_offsets) bn_free(&a, expert_offsets, sz_ecnt);
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
        bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
        return -1;
    }
    memset(moe_out, 0, sz_mout);

    // 5. Per-expert batch compute
    for (int e = 0; e < n_experts; e++) {
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

        // Gate + Up matmul (T tokens at once)
        if (T == 1) {
            // Single token: use matvec (less overhead)
            BnMatvecTask gu[2] = {
                 { gate_buf, &wgate, NULL },
                 { up_buf,   &wup  , NULL },
            };
            bn_quant_matvec_batch(gu, 2, gather_buf, x_q_scratch, bn_model_pool(m));
        } else {
            bn_quant_matmul(gate_buf, &wgate, gather_buf, T, x_q_scratch, bn_model_pool(m));
            bn_quant_matmul(up_buf, &wup, gather_buf, T, x_q_scratch, bn_model_pool(m));
        }

        // SwiGLU activation across T * moe_hidden
        {
            size_t swiglu_n = (size_t)T * moe_hidden;
            for (size_t i = 0; i < swiglu_n; i++) {
                float g = gate_buf[i];
                gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
            }
        }

        // Down matmul
        if (T == 1) {
            bn_quant_matvec(down_buf, &wdown, gate_buf, x_q_scratch, bn_model_pool(m));
        } else {
            bn_quant_matmul(down_buf, &wdown, gate_buf, T, x_q_scratch, bn_model_pool(m));
        }

        // Scatter-add with routing weights
        for (int i = 0; i < T; i++) {
            int tid = group_token_ids[off + i];
            float w = group_weights[off + i];
            float *out_t = moe_out + (size_t)tid * dim;
            float *down_t = down_buf + (size_t)i * dim;
            for (int d = 0; d < dim; d++)
                out_t[d] += w * down_t[d];
        }
    }

    // 6. Shared expert (if present) — batch matmul across all tokens
    if (c->has_shared_expert && lw->shared.shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;
        float *sh_gate = gate_buf;  // reuse (T_max >= 1, shared_hidden <= moe_hidden usually)
        float *sh_up = up_buf;
        float *sh_down = down_buf;

        // Need buffers sized for n_tokens * shared_hidden
        // If shared_hidden > moe_hidden * T_max, we'd need bigger buffers.
        // For safety, allocate if needed.
        int need_sh = (size_t)n_tokens * shared_hidden > (size_t)T_max * moe_hidden;
        size_t sz_shg = (size_t)n_tokens * shared_hidden * sizeof(float);
        size_t sz_shd = (size_t)n_tokens * dim * sizeof(float);
        float *sh_g = need_sh ? (float *)bn_malloc(&a, sz_shg) : sh_gate;
        float *sh_u = need_sh ? (float *)bn_malloc(&a, sz_shg) : sh_up;
        float *sh_d = need_sh ? (float *)bn_malloc(&a, sz_shd) : sh_down;

        if (sh_g && sh_u && sh_d) {
            bn_quant_matmul(sh_g, &lw->shared.shared_gate, Xb, n_tokens, x_q_scratch, bn_model_pool(m));
            bn_quant_matmul(sh_u, &lw->shared.shared_up, Xb, n_tokens, x_q_scratch, bn_model_pool(m));

            size_t sh_total = (size_t)n_tokens * shared_hidden;
            for (size_t i = 0; i < sh_total; i++) {
                float g = sh_g[i];
                sh_g[i] = (g / (1.0f + expf(-g))) * sh_u[i];
            }

            bn_quant_matmul(sh_d, &lw->shared.shared_down, sh_g, n_tokens, x_q_scratch, bn_model_pool(m));

            // Apply shared expert sigmoid gate if present (Qwen3.5 MoE)
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
    }

    // 7. Residual add: act += moe_out
    for (int t = 0; t < n_tokens; t++)
        for (int d = 0; d < dim; d++)
            act[(size_t)t * dim + d] += moe_out[(size_t)t * dim + d];

    // Cleanup
    bn_free(&a, gather_buf, sz_gather); bn_free(&a, gate_buf, sz_gate);
    bn_free(&a, up_buf, sz_up); bn_free(&a, down_buf, sz_down);
    bn_free(&a, moe_out, sz_mout); bn_free(&a, x_q_scratch, sz_xq);
    bn_free(&a, all_indices, sz_idx); bn_free(&a, all_weights, sz_wts);
    bn_free(&a, expert_counts, sz_ecnt); bn_free(&a, expert_offsets, sz_ecnt);
    bn_free(&a, group_token_ids, sz_gtid); bn_free(&a, group_weights, sz_gwts);
    bn_free(&a, fill_pos, sz_fill);

    return 0;
}

