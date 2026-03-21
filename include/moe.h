#ifndef BN_MOE_H
#define BN_MOE_H

#include "model.h"

// Router: SIMD matvec + softmax + top-K selection.
// Writes to ms->expert_indices and ms->expert_weights.
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k, BnThreadPool *pool);

// Full MoE FFN block: route -> load -> compute -> combine.
// Reads from s->x (after norm), writes result to s->xb for residual add.
void bn_moe_forward(BnModel *m, BnLayerWeights *lw, int l);

// Print accumulated MoE stats (I/O, routing, compute breakdown).
void bn_moe_print_stats(const BnMoEState *ms, int n_tokens);

// Reset accumulated stats (call between benchmark runs).
void bn_moe_reset_stats(BnMoEState *ms);

// Create expert LRU cache for pread pipeline (no-op on EMSCRIPTEN or mmap).
// budget_bytes: total cache memory budget (0 to disable).
// gate/up/down_bytes: per-expert projection sizes from expert_map.
void *bn_moe_cache_create(size_t budget_bytes, size_t gate_bytes,
                           size_t up_bytes, size_t down_bytes);

// Free expert cache. Safe to call with NULL.
void bn_moe_cache_free(void *cache);

// Print cache hit/miss stats.
void bn_moe_cache_print_stats(const BnMoEState *ms);

// Create I/O prefetch thread for pread pipeline (no-op on EMSCRIPTEN).
// Call after ms->fd is set. Safe to call if mmap_base is set (returns immediately).
void bn_moe_prefetch_create(BnMoEState *ms);

// Destroy I/O prefetch thread. Safe to call if prefetch is NULL.
void bn_moe_prefetch_destroy(BnMoEState *ms);

// Unit test for LRU cache internals. Returns 0 on success, -1 on failure.
int bn_moe_cache_test(void);

#endif // BN_MOE_H
