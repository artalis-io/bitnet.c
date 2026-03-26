#include "gpu_moe_cache.h"
#include "sh_log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- Hash helpers (same algorithm as CPU expert cache in moe.c) ---

static uint32_t cache_hash(int layer, int expert_idx) {
    uint32_t key = (uint32_t)layer * 65537u + (uint32_t)expert_idx;
    key ^= key >> 16; key *= 0x45d9f3bu; key ^= key >> 16;
    return key;
}

static int cache_probe(const BnGPUMoECache *c, int layer, int expert_idx) {
    uint32_t mask = (uint32_t)(c->hash_size - 1);
    uint32_t h = cache_hash(layer, expert_idx) & mask;
    for (int i = 0; i < c->hash_size; i++) {
        int slot = c->hash_table[h];
        if (slot < 0) return -1;
        if (c->entries[slot].layer == layer && c->entries[slot].expert_idx == expert_idx)
            return slot;
        h = (h + 1) & mask;
    }
    return -1;
}

// --- LRU list helpers ---

static void lru_remove(BnGPUMoECache *c, int slot) {
    BnGPUMoECacheEntry *e = &c->entries[slot];
    if (e->prev >= 0) c->entries[e->prev].next = e->next;
    else c->lru_head = e->next;
    if (e->next >= 0) c->entries[e->next].prev = e->prev;
    else c->lru_tail = e->prev;
    e->prev = e->next = -1;
}

static void lru_push_front(BnGPUMoECache *c, int slot) {
    BnGPUMoECacheEntry *e = &c->entries[slot];
    e->prev = -1;
    e->next = c->lru_head;
    if (c->lru_head >= 0) c->entries[c->lru_head].prev = slot;
    c->lru_head = slot;
    if (c->lru_tail < 0) c->lru_tail = slot;
}

// --- Hash table insert/remove ---

static void hash_insert(BnGPUMoECache *c, int layer, int expert_idx, int slot) {
    uint32_t mask = (uint32_t)(c->hash_size - 1);
    uint32_t h = cache_hash(layer, expert_idx) & mask;
    while (c->hash_table[h] >= 0)
        h = (h + 1) & mask;
    c->hash_table[h] = slot;
}

static void hash_remove(BnGPUMoECache *c, int layer, int expert_idx) {
    uint32_t mask = (uint32_t)(c->hash_size - 1);
    uint32_t h = cache_hash(layer, expert_idx) & mask;
    while (c->hash_table[h] >= 0) {
        int slot = c->hash_table[h];
        if (c->entries[slot].layer == layer && c->entries[slot].expert_idx == expert_idx) {
            // Backshift deletion
            c->hash_table[h] = -1;
            uint32_t j = (h + 1) & mask;
            while (c->hash_table[j] >= 0) {
                int s2 = c->hash_table[j];
                uint32_t ideal = cache_hash(c->entries[s2].layer, c->entries[s2].expert_idx) & mask;
                // Check if s2 should be moved to fill the gap
                int gap_ok = (h <= j) ? (ideal <= h || ideal > j) : (ideal <= h && ideal > j);
                if (gap_ok) {
                    c->hash_table[h] = s2;
                    c->hash_table[j] = -1;
                    h = j;
                }
                j = (j + 1) & mask;
            }
            return;
        }
        h = (h + 1) & mask;
    }
}

// --- Eviction: destroy LRU entry's GPU buffers, return slot ---

static int cache_evict(BnGPUMoECache *c) {
    int slot = c->lru_tail;
    if (slot < 0) return -1;

    BnGPUMoECacheEntry *e = &c->entries[slot];
    // Destroy GPU handles
    if (c->gpu && c->gpu->buffer_destroy) {
        if (e->gate_gpu) c->gpu->buffer_destroy(c->gpu->ctx, e->gate_gpu);
        if (e->up_gpu) c->gpu->buffer_destroy(c->gpu->ctx, e->up_gpu);
        if (e->down_gpu) c->gpu->buffer_destroy(c->gpu->ctx, e->down_gpu);
    }

    // Remove from hash + LRU
    hash_remove(c, e->layer, e->expert_idx);
    lru_remove(c, slot);

    e->layer = -1;
    e->gate_gpu = e->up_gpu = e->down_gpu = NULL;
    return slot;
}

// --- Public API ---

BnGPUMoECache *bn_gpu_moe_cache_create(size_t budget_bytes, size_t entry_bytes,
                                         BnGPUBackend *gpu) {
    if (!gpu || budget_bytes == 0 || entry_bytes == 0) return NULL;

    int n_slots = (int)(budget_bytes / entry_bytes);
    if (n_slots < 1) return NULL;

    int hash_size = 1;
    while (hash_size < n_slots * 2) hash_size <<= 1;

    BnGPUMoECache *c = (BnGPUMoECache *)calloc(1, sizeof(BnGPUMoECache));
    if (!c) return NULL;
    c->entries = (BnGPUMoECacheEntry *)calloc((size_t)n_slots, sizeof(BnGPUMoECacheEntry));
    c->hash_table = (int *)malloc((size_t)hash_size * sizeof(int));
    if (!c->entries || !c->hash_table) {
        free(c->entries); free(c->hash_table); free(c);
        return NULL;
    }

    c->n_slots = n_slots;
    c->hash_size = hash_size;
    c->entry_bytes = entry_bytes;
    c->gpu = gpu;
    c->lru_head = c->lru_tail = -1;
    memset(c->hash_table, -1, (size_t)hash_size * sizeof(int));

    // Build free list (singly-linked via .next)
    for (int i = 0; i < n_slots; i++) {
        c->entries[i].layer = -1;
        c->entries[i].prev = -1;
        c->entries[i].next = (i + 1 < n_slots) ? i + 1 : -1;
    }
    c->free_head = 0;

    char slots_str[32], mb_str[32];
    snprintf(slots_str, sizeof(slots_str), "%d", n_slots);
    snprintf(mb_str, sizeof(mb_str), "%zu", budget_bytes / (1024 * 1024));
    SH_LOG_INFO("GPU MoE expert cache", "slots", slots_str, "budget_MB", mb_str);
    return c;
}

int bn_gpu_moe_cache_lookup(BnGPUMoECache *c, int layer, int expert_idx,
                             void **gate_out, void **up_out, void **down_out) {
    if (!c) return 0;
    int slot = cache_probe(c, layer, expert_idx);
    if (slot < 0) { c->misses++; return 0; }

    // Promote to MRU
    lru_remove(c, slot);
    lru_push_front(c, slot);

    BnGPUMoECacheEntry *e = &c->entries[slot];
    *gate_out = e->gate_gpu;
    *up_out = e->up_gpu;
    *down_out = e->down_gpu;
    c->hits++;
    return 1;
}

int bn_gpu_moe_cache_insert(BnGPUMoECache *c, int layer, int expert_idx,
                              void *gate_gpu, void *up_gpu, void *down_gpu) {
    if (!c) return -1;

    int slot;
    if (c->free_head >= 0) {
        slot = c->free_head;
        c->free_head = c->entries[slot].next;
    } else {
        slot = cache_evict(c);
        if (slot < 0) return -1;
    }

    BnGPUMoECacheEntry *e = &c->entries[slot];
    e->layer = layer;
    e->expert_idx = expert_idx;
    e->gate_gpu = gate_gpu;
    e->up_gpu = up_gpu;
    e->down_gpu = down_gpu;

    hash_insert(c, layer, expert_idx, slot);
    lru_push_front(c, slot);

    (void)0; // insert complete
    return 0;
}

void bn_gpu_moe_cache_free(BnGPUMoECache *c) {
    if (!c) return;
    // Destroy all cached GPU handles
    for (int i = 0; i < c->n_slots; i++) {
        BnGPUMoECacheEntry *e = &c->entries[i];
        if (e->layer < 0) continue;
        if (c->gpu && c->gpu->buffer_destroy) {
            if (e->gate_gpu) c->gpu->buffer_destroy(c->gpu->ctx, e->gate_gpu);
            if (e->up_gpu) c->gpu->buffer_destroy(c->gpu->ctx, e->up_gpu);
            if (e->down_gpu) c->gpu->buffer_destroy(c->gpu->ctx, e->down_gpu);
        }
    }
    free(c->entries);
    free(c->hash_table);
    free(c);
}

void bn_gpu_moe_cache_print_stats(const BnGPUMoECache *c) {
    if (!c) return;
    size_t total = c->hits + c->misses;
    float rate = total > 0 ? 100.0f * (float)c->hits / (float)total : 0.0f;
    char hits_str[32], misses_str[32], rate_str[32];
    snprintf(hits_str, sizeof(hits_str), "%zu", c->hits);
    snprintf(misses_str, sizeof(misses_str), "%zu", c->misses);
    snprintf(rate_str, sizeof(rate_str), "%.1f%%", rate);
    SH_LOG_INFO("GPU MoE cache", "hits", hits_str, "misses", misses_str, "hit_rate", rate_str);
}
