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
    uint64_t generation;
    unsigned pins;
    int pending, published, retained;
} BnGPUMoECacheEntry;

typedef struct BnGPUMoECache {
    BnGPUMoECacheEntry *entries;  // [n_slots]
    int *hash_table;              // [hash_size] open-addressing, slot index or -1
    unsigned hash_size;           // power of 2, >= 2 * n_slots
    int n_slots;

    int lru_head, lru_tail;       // MRU / LRU ends
    int free_head;                // free list head

    size_t entry_bytes;           // gate_bytes + up_bytes + down_bytes (budget accounting)
    BnGPUBackend *gpu;            // needed for buffer_destroy on eviction

    size_t hits, misses, evictions, aborts;
} BnGPUMoECache;

/* Executor-serialized asynchronous use. Do not copy live leases/reservations.
 * Output storage must not already hold a live lease/reservation.
 * The cache is one immutable model keyspace; never repurpose it for another
 * checkpoint while entries exist. The owner backend must outlive the cache.
 */
typedef struct {
    BnGPUMoECache *cache;
    int slot;
    uint64_t generation;
    void *gate, *up, *down;
} BnGPUMoELease;
typedef struct {
    BnGPUMoECache *cache;
    int slot;
    uint64_t generation;
} BnGPUMoEReservation;
#define BN_GPU_MOE_CACHE_BUSY 1
/* Acquire: 1 hit, 0 miss, -1 invalid. promote=0 preserves decode LRU on a
 * prefill hit. Miss leaves output unchanged. Release only after all queued
 * consumers finish. An empty released lease may be released again. */
int bn_gpu_moe_cache_acquire(BnGPUMoECache *cache,int layer,int expert,
    int promote,BnGPUMoELease *out);
int bn_gpu_moe_cache_release(BnGPUMoELease *lease);
/* Reserve BEFORE allocating/uploading weights. Evicts only unpinned LRU,
 * charges one full slot, and prevents duplicate admission of a pending key.
 * 0 success, BUSY no available slot/key pending, -1 invalid/already published.
 * Rejection leaves output unchanged. Pending entries cannot be acquired.
 * Each committed entry's owned payload must fit entry_bytes; caller accounts
 * format-specific allocations, auxiliary layouts and separate staging budgets.
 */
int bn_gpu_moe_cache_reserve(BnGPUMoECache *cache,int layer,int expert,
    BnGPUMoEReservation *out);
/* Same reservation policy, but retain an unpinned victim's buffers under the
 * pending slot's existing charge. The old key is no longer published. Retired
 * storage remains CACHE-OWNED and exclusive until commit/abort; no live lease
 * is ever overwritten. Regular reserve keeps its destroy-on-eviction policy. */
int bn_gpu_moe_cache_reserve_reuse(BnGPUMoECache *cache,int layer,int expert,
    BnGPUMoEReservation *out);
/* 1 retained buffers returned, 0 none (outputs unchanged), -1 invalid ticket.
 * Finish refill/publication ordering before commit. On transfer failure abort
 * destroys even partially overwritten buffers. Never destroy borrowed handles.
 * To change layout, discard BEFORE allocating replacement storage. */
int bn_gpu_moe_cache_reserved_buffers(const BnGPUMoEReservation *reservation,
    void **gate,void **up,void **down);
int bn_gpu_moe_cache_discard_reserved_buffers(BnGPUMoEReservation *reservation);
/* Successful commit transfers handle ownership and returns a pinned lease.
 * At least one handle required, non-NULL handles distinct. All uploads must
 * finish (or be ordered for every consumer) before publication. Failure keeps
 * reservation/new handles owned by caller. Destroy partial new uploads BEFORE
 * abort. A retained reservation must commit its exact borrowed handle tuple,
 * or discard it first; abort automatically destroys retained storage. */
int bn_gpu_moe_cache_commit(BnGPUMoEReservation *reservation,
    void *gate,void *up,void *down,BnGPUMoELease *out);
int bn_gpu_moe_cache_abort(BnGPUMoEReservation *reservation);
typedef struct {
    size_t capacity_payload, charged_payload;
    size_t published, pending, pinned_entries, leases;
    size_t hits, misses, evictions, aborts;
} BnGPUMoECacheStats;
int bn_gpu_moe_cache_stats(const BnGPUMoECache *cache,BnGPUMoECacheStats *out);
/* Close refuses live leases/reservations with BUSY, changing nothing. */
int bn_gpu_moe_cache_close(BnGPUMoECache *cache);

// Create GPU expert cache. budget_bytes: total GPU memory for cached experts.
// entry_bytes: gate_bytes + up_bytes + down_bytes per expert.
// Returns NULL if budget is 0 or allocation fails.
BnGPUMoECache *bn_gpu_moe_cache_create(size_t budget_bytes, size_t entry_bytes,
                                       int max_entries, BnGPUBackend *gpu);

// Lookup cached GPU buffers for (layer, expert_idx).
// On hit: promotes to MRU, sets *gate_out/*up_out/*down_out, returns 1.
// On miss: returns 0.
int bn_gpu_moe_cache_lookup(BnGPUMoECache *c, int layer, int expert_idx,
                             void **gate_out, void **up_out, void **down_out);

// Prefill lookup counts hits/misses without changing LRU order.
int bn_gpu_moe_cache_lookup_prefill(BnGPUMoECache *c, int layer, int expert_idx,
    void **gate_out, void **up_out, void **down_out);

// Insert 3 GPU buffer handles. May evict LRU entry (buffer_destroy on evicted).
// Returns 0 on success. Already allocated handles can transiently exceed the
// payload budget; asynchronous/budgeted callers must reserve before upload.
int bn_gpu_moe_cache_insert(BnGPUMoECache *c, int layer, int expert_idx,
                              void *gate_gpu, void *up_gpu, void *down_gpu);

// Destroy all cached entries + free memory. Safe to call with NULL.
// Refuses live leases/reservations (logs an error); use close to handle BUSY.
void bn_gpu_moe_cache_free(BnGPUMoECache *c);

// Print hit/miss stats.
void bn_gpu_moe_cache_print_stats(const BnGPUMoECache *c);

#endif
