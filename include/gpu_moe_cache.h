#ifndef BN_GPU_MOE_CACHE_H
#define BN_GPU_MOE_CACHE_H

#include "gpu_backend.h"
#include <stddef.h>

typedef struct {
    int layer;          // -1 = empty slot
    int expert_idx;
    int prev, next;     // intrusive LRU doubly-linked list (indices, -1 = sentinel)
    void *gate_gpu;     // GPU buffer handle for gate projection
    void *up_gpu;       // GPU buffer handle for up projection
    void *down_gpu;     // GPU buffer handle for down projection
} BnGPUMoECacheEntry;

typedef struct {
    BnGPUMoECacheEntry *entries;  // [n_slots]
    int *hash_table;              // [hash_size] open-addressing, slot index or -1
    int hash_size;                // power of 2, >= 2 * n_slots
    int n_slots;

    int lru_head, lru_tail;       // MRU / LRU ends
    int free_head;                // free list head

    size_t entry_bytes;           // gate_bytes + up_bytes + down_bytes (budget accounting)
    BnGPUBackend *gpu;            // needed for buffer_destroy on eviction

    size_t hits, misses;
} BnGPUMoECache;

// Create GPU expert cache. budget_bytes: total GPU memory for cached experts.
// entry_bytes: gate_bytes + up_bytes + down_bytes per expert.
// Returns NULL if budget is 0 or allocation fails.
BnGPUMoECache *bn_gpu_moe_cache_create(size_t budget_bytes, size_t entry_bytes,
                                         BnGPUBackend *gpu);

// Lookup cached GPU buffers for (layer, expert_idx).
// On hit: promotes to MRU, sets *gate_out/*up_out/*down_out, returns 1.
// On miss: returns 0.
int bn_gpu_moe_cache_lookup(BnGPUMoECache *c, int layer, int expert_idx,
                             void **gate_out, void **up_out, void **down_out);

// Insert 3 GPU buffer handles. May evict LRU entry (buffer_destroy on evicted).
// Returns 0 on success.
int bn_gpu_moe_cache_insert(BnGPUMoECache *c, int layer, int expert_idx,
                              void *gate_gpu, void *up_gpu, void *down_gpu);

// Destroy all cached entries + free memory. Safe to call with NULL.
void bn_gpu_moe_cache_free(BnGPUMoECache *c);

// Print hit/miss stats.
void bn_gpu_moe_cache_print_stats(const BnGPUMoECache *c);

#endif
