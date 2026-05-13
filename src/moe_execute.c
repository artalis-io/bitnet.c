#include "moe_internal.h"

// Full MoE FFN block
void bn_moe_forward(struct BnModel *m, BnSession *sess,
                    struct BnLayerWeights *lw, int l) {
#if defined(__EMSCRIPTEN__)
    (void)l;
#endif
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnMoEState *ms = sess->moe_state;
    int dim = c->dim;
    int moe_hidden = c->moe_intermediate_size;
    int K = c->n_experts_active;
    double t0;

    // 1. RMSNorm input
    t0 = bn_moe_time_ms();
    bn_moe_rmsnorm(s->xb, s->x, lw->norm.ffn_norm, dim, c->norm_eps);
    ms->stats.norm_time_ms += bn_moe_time_ms() - t0;

    // 2. Route: select top-K experts (SIMD + threaded)
    t0 = bn_moe_time_ms();
    bn_moe_route(ms, s->xb, lw->moe.router_weight, dim, c->n_experts, K, bn_model_pool(m));
    ms->stats.route_time_ms += bn_moe_time_ms() - t0;

    // 3. Zero output accumulator
    memset(ms->expert_out, 0, dim * sizeof(float));

    // 4. Expert FFN compute
    double t_compute = bn_moe_time_ms();
    int shared_gu_ready = 0;

    if (bn_model_moe_io(m)->mmap_base && K <= BN_MAX_MOE_K) {
        // --- Cross-expert batched dispatch (mmap path) ---
        int valid_k = 0;
        int valid_indices[BN_MAX_MOE_K];
        float valid_weights[BN_MAX_MOE_K];
        BnQWeight wgates[BN_MAX_MOE_K], wups[BN_MAX_MOE_K];

        // Prefetch gate+up pages for all K experts (madvise mode)
#if !defined(__EMSCRIPTEN__)
        if (bn_model_moe_io(m)->madvise_mode) {
            double ta = bn_moe_time_ms();
            bn_moe_madvise_experts(bn_model_moe_io(m), &lw->moe.expert_map, ms->expert_indices, K,
                                MADV_WILLNEED, 0x3 /* gate+up */);
            ms->stats.madvise_time_ms += bn_moe_time_ms() - ta;
        }
#endif

        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            if (eidx < 0) continue;
            const void *gate_data = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, &lw->moe.expert_map, eidx, 0);
            const void *up_data   = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, &lw->moe.expert_map, eidx, 1);
            if (!gate_data || !up_data) {
                SH_LOG_ERROR("Failed to load expert gate/up projection");
                continue;
            }
            wgates[valid_k] = bn_moe_make_qweight(gate_data, lw->moe.expert_map.gate_type,
                                                lw->moe.expert_map.gate_rows, lw->moe.expert_map.gate_cols);
            wups[valid_k]   = bn_moe_make_qweight(up_data, lw->moe.expert_map.up_type,
                                                lw->moe.expert_map.up_rows, lw->moe.expert_map.up_cols);
            valid_indices[valid_k] = eidx;
            valid_weights[valid_k] = ms->expert_weights[k];
            valid_k++;
        }

        if (valid_k > 0) {
            // Gate+up batch
            t0 = bn_moe_time_ms();
            BnMatvecTask gu_tasks[2 * BN_MAX_MOE_K + 2];
            int n_gu = 0;
            for (int k = 0; k < valid_k; k++) {
                gu_tasks[n_gu++] = (BnMatvecTask){ ms->expert_hb_batch[k],  &wgates[k], NULL };
                gu_tasks[n_gu++] = (BnMatvecTask){ ms->expert_hb2_batch[k], &wups[k]  , NULL };
            }
            if (c->has_shared_expert && lw->shared.shared_gate.data) {
                int batch_type = gu_tasks[0].W->type;
                int can_batch_shared = (lw->shared.shared_gate.type == batch_type &&
                                        lw->shared.shared_up.type == batch_type);
                for (int i = 1; can_batch_shared && i < n_gu; i++)
                    can_batch_shared = (gu_tasks[i].W->type == batch_type);
#if defined(__AVX2__)
                if (!can_batch_shared &&
                    bn_quant_format_can_preq8k(lw->shared.shared_gate.type) &&
                    bn_quant_format_can_preq8k(lw->shared.shared_up.type)) {
                    can_batch_shared = 1;
                    for (int i = 0; can_batch_shared && i < n_gu; i++) {
                        int type = gu_tasks[i].W->type;
                        can_batch_shared = bn_quant_format_can_preq8k(type);
                    }
                }
#endif
                if (can_batch_shared) {
                    gu_tasks[n_gu++] = (BnMatvecTask){ s->hb,  &lw->shared.shared_gate, NULL };
                    gu_tasks[n_gu++] = (BnMatvecTask){ s->hb2, &lw->shared.shared_up  , NULL };
                    shared_gu_ready = 1;
                }
            }
            bn_quant_matvec_batch(gu_tasks, n_gu, s->xb, s->x_q, bn_model_pool(m));
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            // Parallel SwiGLU
            t0 = bn_moe_time_ms();
            BnSwiGLUCtx swiglu_ctxs[BN_MAX_MOE_K];
            BnTPTask swiglu_tasks[BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                swiglu_ctxs[k] = (BnSwiGLUCtx){
                    ms->expert_hb_batch[k],
                    ms->expert_hb_batch[k],
                    ms->expert_hb2_batch[k]
                };
                swiglu_tasks[k] = (BnTPTask){ bn_moe_swiglu_range, &swiglu_ctxs[k], moe_hidden };
            }
            bn_tp_dispatch(bn_model_pool(m), swiglu_tasks, valid_k);
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            // Prefetch down projection pages (madvise mode)
#if !defined(__EMSCRIPTEN__)
            if (bn_model_moe_io(m)->madvise_mode) {
                double ta = bn_moe_time_ms();
                bn_moe_madvise_experts(bn_model_moe_io(m), &lw->moe.expert_map, valid_indices, valid_k,
                                    MADV_WILLNEED, 0x4 /* down */);
                ms->stats.madvise_time_ms += bn_moe_time_ms() - ta;
            }
#endif

            // Down projections
            t0 = bn_moe_time_ms();
            BnQWeight wdowns[BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                const void *down_data = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, &lw->moe.expert_map, valid_indices[k], 2);
                if (!down_data) {
                    SH_LOG_ERROR("Failed to load expert down projection");
                    valid_weights[k] = 0.0f;
                    continue;
                }
                wdowns[k] = bn_moe_make_qweight(down_data, lw->moe.expert_map.down_type,
                                              lw->moe.expert_map.down_rows, lw->moe.expert_map.down_cols);
            }

            // Batched down projection: K independent (W, x) pairs in one dispatch.
            // Each expert has different input (hb after SwiGLU) and different weights.
            {
                BnMatvecMultiTask down_tasks[BN_MAX_MOE_K];
                int n_down = 0;
                for (int k = 0; k < valid_k; k++) {
                    if (valid_weights[k] == 0.0f) continue;
                    down_tasks[n_down++] = (BnMatvecMultiTask){
                        ms->expert_down_batch[k], &wdowns[k], ms->expert_hb_batch[k], NULL
                    };
                }
                bn_quant_matvec_multi(down_tasks, n_down, ms->down_x_q_bufs, bn_model_pool(m));
            }
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;

            // Weighted accumulation
            t0 = bn_moe_time_ms();
            for (int k = 0; k < valid_k; k++) {
                float w = valid_weights[k];
                if (w == 0.0f) continue;
                bn_moe_weighted_add(ms->expert_out, ms->expert_down_batch[k], w, dim);
            }
            ms->stats.accum_time_ms += bn_moe_time_ms() - t0;

        }
    }
#if !defined(__EMSCRIPTEN__)
    else if (bn_model_moe_io(m)->fd >= 0 && !bn_model_moe_io(m)->mmap_base) {
        // --- Pipelined pread path with LRU cache ---
        BnMoEPrefetch *pf_gu = (BnMoEPrefetch *)bn_model_moe_io(m)->prefetch;
        BnMoEPrefetch *pf_dn = (BnMoEPrefetch *)bn_model_moe_io(m)->prefetch_down;
        BnMoECache *cache = (BnMoECache *)bn_model_moe_io(m)->cache;
        const BnMoEExpertMap *map = &lw->moe.expert_map;

        // Sort expert indices ascending (insertion sort, K is small)
        // to enable read coalescing on cache misses
        for (int i = 1; i < K; i++) {
            int idx = ms->expert_indices[i];
            float w = ms->expert_weights[i];
            int j = i - 1;
            while (j >= 0 && ms->expert_indices[j] > idx) {
                ms->expert_indices[j + 1] = ms->expert_indices[j];
                ms->expert_weights[j + 1] = ms->expert_weights[j];
                j--;
            }
            ms->expert_indices[j + 1] = idx;
            ms->expert_weights[j + 1] = w;
        }

        // --- Two-phase: separate cache hits from misses ---
        int n_hits = 0, n_misses = 0;
        int miss_indices[BN_MAX_MOE_K];
        float hit_weights[BN_MAX_MOE_K], miss_weights[BN_MAX_MOE_K];
        const uint8_t *hit_ptrs[BN_MAX_MOE_K];  // cache slab pointers for hits

        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            if (eidx < 0) continue;

            if (cache) {
                const uint8_t *cached = bn_moe_cache_lookup_internal(cache, l, eidx);
                if (cached) {
                    hit_weights[n_hits] = ms->expert_weights[k];
                    hit_ptrs[n_hits] = cached;
                    n_hits++;
                    ms->stats.cache_hits++;
                    continue;
                }
                ms->stats.cache_misses++;
            }
            miss_indices[n_misses] = eidx;
            miss_weights[n_misses] = ms->expert_weights[k];
            n_misses++;
        }

        // Start I/O for first miss while we batch-compute hits
        int miss_io_started = 0;
        uint8_t *miss_slot_ptr = NULL;
        uint8_t *miss_g_dst, *miss_u_dst, *miss_d_dst;
        size_t miss_g_off, miss_g_sz, miss_u_off, miss_u_sz, miss_d_off, miss_d_sz;

        if (n_misses > 0) {
            int meidx = miss_indices[0];
            bn_moe_proj_info(map, meidx, 0, &miss_g_off, &miss_g_sz);
            bn_moe_proj_info(map, meidx, 1, &miss_u_off, &miss_u_sz);
            bn_moe_proj_info(map, meidx, 2, &miss_d_off, &miss_d_sz);

            miss_slot_ptr = cache ? bn_moe_cache_insert_internal(cache, l, meidx) : NULL;
            miss_g_dst = miss_slot_ptr ? miss_slot_ptr : ms->buf;
            miss_u_dst = miss_slot_ptr ? miss_slot_ptr + bn_moe_cache_gate_bytes(cache) : ms->buf2;
            miss_d_dst = miss_slot_ptr ? miss_slot_ptr + bn_moe_cache_gate_bytes(cache) + bn_moe_cache_up_bytes(cache) : ms->buf5;

            if (pf_gu) {
                bn_moe_prefetch_start2_internal(pf_gu, miss_g_dst, miss_g_sz, (off_t)miss_g_off,
                                           miss_u_dst, miss_u_sz, (off_t)miss_u_off);
            }
            if (pf_dn) {
                bn_moe_prefetch_start1_internal(pf_dn, miss_d_dst, miss_d_sz, (off_t)miss_d_off);
            }
            miss_io_started = 1;
        }

        // Phase 1: Batch gate+up for all cache hits
        if (n_hits > 0) {
            t0 = bn_moe_time_ms();
            BnQWeight wgates[BN_MAX_MOE_K], wups[BN_MAX_MOE_K];
            BnMatvecTask gu_tasks[2 * BN_MAX_MOE_K];
            for (int h = 0; h < n_hits; h++) {
                const uint8_t *cp = hit_ptrs[h];
                wgates[h] = bn_moe_make_qweight(cp, map->gate_type,
                                              map->gate_rows, map->gate_cols);
                wups[h]   = bn_moe_make_qweight(cp + bn_moe_cache_gate_bytes(cache), map->up_type,
                                              map->up_rows, map->up_cols);
                gu_tasks[2*h]     = (BnMatvecTask){ ms->expert_hb_batch[h],  &wgates[h], NULL };
                gu_tasks[2*h + 1] = (BnMatvecTask){ ms->expert_hb2_batch[h], &wups[h]  , NULL };
            }
            bn_quant_matvec_batch(gu_tasks, 2 * n_hits, s->xb, s->x_q, bn_model_pool(m));
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            // Parallel SwiGLU for hits
            t0 = bn_moe_time_ms();
            BnSwiGLUCtx swiglu_ctxs[BN_MAX_MOE_K];
            BnTPTask swiglu_tasks[BN_MAX_MOE_K];
            for (int h = 0; h < n_hits; h++) {
                swiglu_ctxs[h] = (BnSwiGLUCtx){
                    ms->expert_hb_batch[h],
                    ms->expert_hb_batch[h],
                    ms->expert_hb2_batch[h]
                };
                swiglu_tasks[h] = (BnTPTask){ bn_moe_swiglu_range, &swiglu_ctxs[h], moe_hidden };
            }
            bn_tp_dispatch(bn_model_pool(m), swiglu_tasks, n_hits);
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            // Down projections for hits (data already in cache)
            t0 = bn_moe_time_ms();
            for (int h = 0; h < n_hits; h++) {
                const uint8_t *dp = hit_ptrs[h] + bn_moe_cache_gate_bytes(cache) + bn_moe_cache_up_bytes(cache);
                BnQWeight wdown = bn_moe_make_qweight(dp, map->down_type,
                                                    map->down_rows, map->down_cols);
                bn_quant_matvec(ms->expert_down_batch[h], &wdown,
                                ms->expert_hb_batch[h], s->x_q, bn_model_pool(m));
            }
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;

            // Weighted accumulation for hits
            t0 = bn_moe_time_ms();
            for (int h = 0; h < n_hits; h++) {
                float w = hit_weights[h];
                bn_moe_weighted_add(ms->expert_out, ms->expert_down_batch[h], w, dim);
            }
            ms->stats.accum_time_ms += bn_moe_time_ms() - t0;
        }

        // Phase 2: Process cache misses with I/O overlap
        for (int mi = 0; mi < n_misses; mi++) {
            int eidx = miss_indices[mi];
            float weight = miss_weights[mi];
            const uint8_t *gate_ptr, *up_ptr, *down_ptr;

            if (mi == 0 && miss_io_started) {
                // First miss: I/O was started before phase 1
                if (pf_gu) {
                    double tw = bn_moe_time_ms();
                    int ok = bn_moe_prefetch_wait_internal(pf_gu);
                    ms->stats.prefetch_wait_ms += bn_moe_time_ms() - tw;
                    bn_moe_prefetch_collect_stats(pf_gu, &ms->stats);
                    if (!ok) {
                        if (pread(bn_model_moe_io(m)->fd, miss_g_dst, miss_g_sz, (off_t)miss_g_off) != (ssize_t)miss_g_sz)
                            SH_LOG_ERROR("Fallback gate pread failed");
                        if (pread(bn_model_moe_io(m)->fd, miss_u_dst, miss_u_sz, (off_t)miss_u_off) != (ssize_t)miss_u_sz)
                            SH_LOG_ERROR("Fallback up pread failed");
                    }
                } else {
                    if (pread(bn_model_moe_io(m)->fd, miss_g_dst, miss_g_sz, (off_t)miss_g_off) != (ssize_t)miss_g_sz)
                        SH_LOG_ERROR("Sync gate pread failed");
                    if (pread(bn_model_moe_io(m)->fd, miss_u_dst, miss_u_sz, (off_t)miss_u_off) != (ssize_t)miss_u_sz)
                        SH_LOG_ERROR("Sync up pread failed");
                    if (!pf_dn)
                        if (pread(bn_model_moe_io(m)->fd, miss_d_dst, miss_d_sz, (off_t)miss_d_off) != (ssize_t)miss_d_sz)
                            SH_LOG_ERROR("Sync down pread failed");
                }
                gate_ptr = miss_g_dst;
                up_ptr   = miss_u_dst;
                down_ptr = miss_d_dst;
            } else {
                // Subsequent misses: load gate+up+down
                size_t g_off, g_sz, u_off, u_sz, d_off, d_sz;
                bn_moe_proj_info(map, eidx, 0, &g_off, &g_sz);
                bn_moe_proj_info(map, eidx, 1, &u_off, &u_sz);
                bn_moe_proj_info(map, eidx, 2, &d_off, &d_sz);

                uint8_t *slot = cache ? bn_moe_cache_insert_internal(cache, l, eidx) : NULL;
                uint8_t *g_dst = slot ? slot : ms->buf;
                uint8_t *u_dst = slot ? slot + bn_moe_cache_gate_bytes(cache) : ms->buf2;
                uint8_t *d_dst = slot ? slot + bn_moe_cache_gate_bytes(cache) + bn_moe_cache_up_bytes(cache) : ms->buf5;

                if (pf_gu) {
                    bn_moe_prefetch_start2_internal(pf_gu, g_dst, g_sz, (off_t)g_off,
                                               u_dst, u_sz, (off_t)u_off);
                }
                if (pf_dn) {
                    bn_moe_prefetch_start1_internal(pf_dn, d_dst, d_sz, (off_t)d_off);
                }

                if (pf_gu) {
                    double tw = bn_moe_time_ms();
                    int ok = bn_moe_prefetch_wait_internal(pf_gu);
                    ms->stats.prefetch_wait_ms += bn_moe_time_ms() - tw;
                    bn_moe_prefetch_collect_stats(pf_gu, &ms->stats);
                    if (!ok) {
                        if (pread(bn_model_moe_io(m)->fd, g_dst, g_sz, (off_t)g_off) != (ssize_t)g_sz)
                            SH_LOG_ERROR("Fallback gate pread failed");
                        if (pread(bn_model_moe_io(m)->fd, u_dst, u_sz, (off_t)u_off) != (ssize_t)u_sz)
                            SH_LOG_ERROR("Fallback up pread failed");
                    }
                } else {
                    if (pread(bn_model_moe_io(m)->fd, g_dst, g_sz, (off_t)g_off) != (ssize_t)g_sz)
                        SH_LOG_ERROR("Sync gate pread failed");
                    if (pread(bn_model_moe_io(m)->fd, u_dst, u_sz, (off_t)u_off) != (ssize_t)u_sz)
                        SH_LOG_ERROR("Sync up pread failed");
                    if (pread(bn_model_moe_io(m)->fd, d_dst, d_sz, (off_t)d_off) != (ssize_t)d_sz)
                        SH_LOG_ERROR("Sync down pread failed");
                }

                gate_ptr = g_dst;
                up_ptr   = u_dst;
                down_ptr = d_dst;
            }

            // Gate+up matvec (down I/O may still be in flight)
            t0 = bn_moe_time_ms();
            {
                BnQWeight wgate = bn_moe_make_qweight(gate_ptr, map->gate_type,
                                                    map->gate_rows, map->gate_cols);
                BnQWeight wup = bn_moe_make_qweight(up_ptr, map->up_type,
                                                  map->up_rows, map->up_cols);
                BnMatvecTask gu[2] = {
                     { ms->expert_hb,  &wgate, NULL },
                     { ms->expert_hb2, &wup  , NULL },
                };
                bn_quant_matvec_batch(gu, 2, s->xb, s->x_q, bn_model_pool(m));
            }
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            // SwiGLU
            t0 = bn_moe_time_ms();
            bn_moe_swiglu(ms->expert_hb, ms->expert_hb, ms->expert_hb2, moe_hidden);
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            // Wait for down I/O
            t0 = bn_moe_time_ms();
            if (pf_dn) {
                double tw = bn_moe_time_ms();
                int ok = bn_moe_prefetch_wait_internal(pf_dn);
                ms->stats.prefetch_wait_ms += bn_moe_time_ms() - tw;
                bn_moe_prefetch_collect_stats(pf_dn, &ms->stats);
                if (!ok) {
                    size_t d_off, d_sz;
                    bn_moe_proj_info(map, eidx, 2, &d_off, &d_sz);
                    if (pread(bn_model_moe_io(m)->fd, (void *)(uintptr_t)down_ptr, d_sz, (off_t)d_off) != (ssize_t)d_sz)
                        SH_LOG_ERROR("Fallback down pread failed");
                }
            }

            // Down matvec
            {
                BnQWeight wdown = bn_moe_make_qweight(down_ptr, map->down_type,
                                                    map->down_rows, map->down_cols);
                bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, bn_model_pool(m));
            }
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;

            // Weighted accumulation
            t0 = bn_moe_time_ms();
            bn_moe_weighted_add(ms->expert_out, s->xb2, weight, dim);
            ms->stats.accum_time_ms += bn_moe_time_ms() - t0;
        }

        #undef COLLECT_PF_STATS
    }
#endif
    else {
        // --- Serial fallback (mmap K > BN_MAX_MOE_K or EMSCRIPTEN) ---
        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            float weight = ms->expert_weights[k];
            if (eidx < 0) continue;

            t0 = bn_moe_time_ms();
            const void *gate_data = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, &lw->moe.expert_map, eidx, 0);
            const void *up_data = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, &lw->moe.expert_map, eidx, 1);
            if (!gate_data || !up_data) {
                SH_LOG_ERROR("Failed to load expert gate/up projection");
                continue;
            }
            BnQWeight wgate = bn_moe_make_qweight(gate_data, lw->moe.expert_map.gate_type,
                                                lw->moe.expert_map.gate_rows, lw->moe.expert_map.gate_cols);
            BnQWeight wup = bn_moe_make_qweight(up_data, lw->moe.expert_map.up_type,
                                              lw->moe.expert_map.up_rows, lw->moe.expert_map.up_cols);
            BnMatvecTask gu[2] = {
                 { ms->expert_hb,  &wgate, NULL },
                 { ms->expert_hb2, &wup  , NULL },
            };
            bn_quant_matvec_batch(gu, 2, s->xb, s->x_q, bn_model_pool(m));
            ms->stats.gate_up_time_ms += bn_moe_time_ms() - t0;

            // SwiGLU activation
            t0 = bn_moe_time_ms();
            bn_moe_swiglu(ms->expert_hb, ms->expert_hb, ms->expert_hb2, moe_hidden);
            ms->stats.swiglu_time_ms += bn_moe_time_ms() - t0;

            // Down projection
            t0 = bn_moe_time_ms();
            const void *down_data = bn_moe_load_expert_proj(bn_model_moe_io(m), ms, &lw->moe.expert_map, eidx, 2);
            if (!down_data) {
                SH_LOG_ERROR("Failed to load expert down projection");
                continue;
            }
            BnQWeight wdown = bn_moe_make_qweight(down_data, lw->moe.expert_map.down_type,
                                                lw->moe.expert_map.down_rows, lw->moe.expert_map.down_cols);
            bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, bn_model_pool(m));
            ms->stats.down_time_ms += bn_moe_time_ms() - t0;

            // Weighted accumulation
            t0 = bn_moe_time_ms();
            bn_moe_weighted_add(ms->expert_out, s->xb2, weight, dim);
            ms->stats.accum_time_ms += bn_moe_time_ms() - t0;
        }
    }

    // 5. Shared expert (if present, always resident)
    t0 = bn_moe_time_ms();
    if (c->has_shared_expert && lw->shared.shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;

        if (!shared_gu_ready) {
            BnMatvecTask shared_gu[2] = {
                 { s->hb,  &lw->shared.shared_gate, NULL },
                 { s->hb2, &lw->shared.shared_up  , NULL },
            };
            bn_quant_matvec_batch(shared_gu, 2, s->xb, s->x_q, bn_model_pool(m));
        }
        bn_moe_swiglu(s->hb, s->hb, s->hb2, shared_hidden);
        bn_quant_matvec(s->xb2, &lw->shared.shared_down, s->hb, s->x_q, bn_model_pool(m));

        // Apply shared expert sigmoid gate if present (Qwen3.5 MoE):
        // gate = sigmoid(dot(input, gate_weight)) — scalar per token
        if (lw->shared.shared_expert_gate) {
            float gate_dot = 0.0f;
            for (int d = 0; d < dim; d++)
                gate_dot += s->xb[d] * lw->shared.shared_expert_gate[d];
            float gate = 1.0f / (1.0f + expf(-gate_dot));
            bn_moe_weighted_add(ms->expert_out, s->xb2, gate, dim);
        } else {
            bn_moe_weighted_add(ms->expert_out, s->xb2, 1.0f, dim);
        }
    }
    ms->stats.shared_time_ms += bn_moe_time_ms() - t0;

    ms->stats.compute_time_ms += bn_moe_time_ms() - t_compute;

    // 6. Copy result to xb for residual add by caller
    memcpy(s->xb, ms->expert_out, dim * sizeof(float));

    // 7. Residual add
    bn_moe_residual_add(s->x, s->xb, dim);
}
