#include "moe_internal.h"
#include "model_internal.h"
#include "backend_model.h"

// --- Expert LRU Cache (pread pipeline) ---
#if !defined(__EMSCRIPTEN__)

typedef struct {
    int layer;          // -1 = empty
    int expert_idx;
    int prev, next;     // intrusive LRU doubly-linked list (indices, -1 = sentinel)
    unsigned pins;     // host expert leases held by synchronous GPU uploads
    int published;     // reservations stay invisible until every read succeeds
} BnMoECacheEntry;

struct BnMoECache {
    BnMoEExpertReader reader; /* Non-NULL identity isolates generic reader caches. */
    uint8_t *slab;              // [n_slots * entry_bytes], 32-byte aligned
    size_t entry_bytes;         // gate_bytes + up_bytes + down_bytes
    size_t gate_bytes;          // size of gate projection
    size_t up_bytes;            // size of up projection
    int n_slots;
    int pinned_slots;

    BnMoECacheEntry *entries;   // [n_slots] metadata
    int *hash_table;            // [hash_size] open-addressing, slot index or -1
    int hash_size;              // power of 2, >= 2 * n_slots

    int lru_head, lru_tail;     // MRU / LRU ends
    int free_head;              // free list head
};

static uint32_t moe_cache_hash(int layer, int expert_idx) {
    uint32_t key = (uint32_t)layer * 65537u + (uint32_t)expert_idx;
    // murmurhash3 finalizer
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

static int moe_cache_probe(const BnMoECache *c, int layer, int expert_idx) {
    uint32_t h = moe_cache_hash(layer, expert_idx) & (uint32_t)(c->hash_size - 1);
    for (int i = 0; i < c->hash_size; i++) {
        int idx = (int)((h + (uint32_t)i) & (uint32_t)(c->hash_size - 1));
        int slot = c->hash_table[idx];
        if (slot < 0) return -1;  // empty = not found
        if (c->entries[slot].layer == layer && c->entries[slot].expert_idx == expert_idx)
            return idx;  // return hash table index
    }
    return -1;
}

static void moe_cache_lru_remove(BnMoECache *c, int slot) {
    BnMoECacheEntry *e = &c->entries[slot];
    if (e->prev >= 0) c->entries[e->prev].next = e->next;
    else c->lru_head = e->next;
    if (e->next >= 0) c->entries[e->next].prev = e->prev;
    else c->lru_tail = e->prev;
    e->prev = e->next = -1;
}

static void moe_cache_lru_push_front(BnMoECache *c, int slot) {
    BnMoECacheEntry *e = &c->entries[slot];
    e->prev = -1;
    e->next = c->lru_head;
    if (c->lru_head >= 0) c->entries[c->lru_head].prev = slot;
    c->lru_head = slot;
    if (c->lru_tail < 0) c->lru_tail = slot;
}

// Remove one entry and rebuild the remainder of its linear-probe cluster.
static void moe_cache_hash_remove(BnMoECache *c, int layer, int expert_idx) {
    uint32_t mask = (uint32_t)(c->hash_size - 1);
    uint32_t h = moe_cache_hash(layer, expert_idx) & mask;
    int idx = -1;
    for (int i = 0; i < c->hash_size; i++) {
        int probe = (int)((h + (uint32_t)i) & mask);
        int slot = c->hash_table[probe];
        if (slot < 0) return;  // not found
        if (c->entries[slot].layer == layer && c->entries[slot].expert_idx == expert_idx) {
            idx = probe;
            break;
        }
    }
    if (idx < 0) return;

    c->hash_table[idx] = -1;
    int next = (int)(((uint32_t)idx + 1u) & mask);
    while (c->hash_table[next] >= 0) {
        int slot = c->hash_table[next];
        c->hash_table[next] = -1;
        uint32_t natural = moe_cache_hash(c->entries[slot].layer,
                                          c->entries[slot].expert_idx) & mask;
        while (c->hash_table[natural] >= 0)
            natural = (natural + 1u) & mask;
        c->hash_table[natural] = slot;
        next = (next + 1) & (int)mask;
    }
}

static const uint8_t *moe_cache_lookup(BnMoECache *c, int layer, int expert_idx) {
    int hi = moe_cache_probe(c, layer, expert_idx);
    if (hi < 0) return NULL;
    int slot = c->hash_table[hi];
    // Promote to MRU
    moe_cache_lru_remove(c, slot);
    moe_cache_lru_push_front(c, slot);
    return c->slab + (size_t)slot * c->entry_bytes;
}

static int moe_cache_evict(BnMoECache *c) {
    int slot = c->lru_tail;
    while (slot >= 0 && c->entries[slot].pins)
        slot = c->entries[slot].prev;
    if (slot < 0) return -1;
    // Remove from LRU
    moe_cache_lru_remove(c, slot);
    // Remove from hash table
    moe_cache_hash_remove(c, c->entries[slot].layer, c->entries[slot].expert_idx);
    c->entries[slot].layer = -1;
    return slot;
}

static uint8_t *moe_cache_insert(BnMoECache *c, int layer, int expert_idx) {
    int slot;

    // Try free list first
    if (c->free_head >= 0) {
        slot = c->free_head;
        c->free_head = c->entries[slot].next;
    } else {
        // Evict LRU tail
        slot = moe_cache_evict(c);
        if (slot < 0) return NULL;
    }

    // Set entry metadata
    c->entries[slot].layer = layer;
    c->entries[slot].expert_idx = expert_idx;
    c->entries[slot].published = 1;

    // Insert into hash table
    uint32_t mask = (uint32_t)(c->hash_size - 1);
    uint32_t h = moe_cache_hash(layer, expert_idx) & mask;
    for (int i = 0; i < c->hash_size; i++) {
        int idx = (int)((h + (uint32_t)i) & mask);
        if (c->hash_table[idx] < 0) {
            c->hash_table[idx] = slot;
            break;
        }
    }

    // Push to MRU
    moe_cache_lru_push_front(c, slot);

    return c->slab + (size_t)slot * c->entry_bytes;
}


const uint8_t *bn_moe_cache_lookup_internal(BnMoECache *c, int layer, int expert_idx) {
    return c && !c->reader.identity ? moe_cache_lookup(c, layer, expert_idx) : NULL;
}

uint8_t *bn_moe_cache_insert_internal(BnMoECache *c, int layer, int expert_idx) {
    return c && !c->reader.identity ? moe_cache_insert(c, layer, expert_idx) : NULL;
}

size_t bn_moe_cache_gate_bytes(const BnMoECache *c) {
    return c ? c->gate_bytes : 0;
}

size_t bn_moe_cache_up_bytes(const BnMoECache *c) {
    return c ? c->up_bytes : 0;
}

#endif // !__EMSCRIPTEN__

void *bn_moe_cache_create(size_t budget_bytes, size_t gate_bytes,
                            size_t up_bytes, size_t down_bytes) {
    return bn_moe_cache_create_policy(budget_bytes,gate_bytes,up_bytes,down_bytes,
        BN_MOE_CACHE_PAGES_HUGE_HINT);
}

void *bn_moe_cache_create_policy(size_t budget_bytes,size_t gate_bytes,
    size_t up_bytes,size_t down_bytes,BnMoECachePagePolicy pages) {
    if(pages!=BN_MOE_CACHE_PAGES_HUGE_HINT&&pages!=BN_MOE_CACHE_PAGES_SYSTEM)
        return NULL;
#if !defined(__EMSCRIPTEN__)
    if (budget_bytes == 0) return NULL;
    if (gate_bytes > SIZE_MAX - up_bytes ||
        gate_bytes + up_bytes > SIZE_MAX - down_bytes)
        return NULL;
    size_t entry_bytes = gate_bytes + up_bytes + down_bytes;
    if (entry_bytes == 0) return NULL;

    size_t raw_slots = budget_bytes / entry_bytes;
    if (raw_slots < 1) return NULL;
    /* The next power-of-two hash size (at least 2*slots) must fit int. */
    size_t max_slots = ((size_t)INT_MAX + 1u) / 4u;
    if (raw_slots > max_slots) raw_slots = max_slots;
    int n_slots = (int)raw_slots;

    BnMoECache *c = (BnMoECache *)calloc(1, sizeof(BnMoECache));
    if (!c) return NULL;

    c->entry_bytes = entry_bytes;
    c->gate_bytes = gate_bytes;
    c->up_bytes = up_bytes;
    c->n_slots = n_slots;

    // Hash table: next power of 2 >= 2 * n_slots (unsigned to avoid overflow)
    unsigned hs = 1;
    while (hs < (unsigned)n_slots * 2) hs *= 2;
    c->hash_size = (int)hs;

    // Allocate slab (32-byte aligned)
    size_t slab_size = (size_t)n_slots * entry_bytes;
    c->slab = (uint8_t *)bn_platform_aligned_alloc(32, slab_size);
    if (!c->slab) { free(c); return NULL; }
    if(pages==BN_MOE_CACHE_PAGES_HUGE_HINT)
        bn_platform_advise_huge_pages(c->slab, slab_size);

    c->entries = (BnMoECacheEntry *)calloc((size_t)n_slots, sizeof(BnMoECacheEntry));
    c->hash_table = (int *)malloc((size_t)hs * sizeof(int));
    if (!c->entries || !c->hash_table) {
        bn_platform_aligned_free(c->slab);
        free(c->entries); free(c->hash_table); free(c);
        return NULL;
    }

    // Initialize hash table to -1 (empty)
    for (int i = 0; i < (int)hs; i++) c->hash_table[i] = -1;
    c->lru_head = c->lru_tail = -1;

    // Build free list (singly-linked via .next)
    for (int i = 0; i < n_slots; i++) {
        c->entries[i].layer = -1;
        c->entries[i].expert_idx = -1;
        c->entries[i].prev = -1;
        c->entries[i].next = (i + 1 < n_slots) ? i + 1 : -1;
    }
    c->free_head = 0;

    {
        char slots_s[16], mb_s[16];
        snprintf(slots_s, sizeof(slots_s), "%d", n_slots);
        snprintf(mb_s, sizeof(mb_s), "%.0f", (double)slab_size / (1024.0 * 1024.0));
        SH_LOG_INFO("MoE expert cache", "slots", slots_s, "slab_MB", mb_s);
    }

    return c;
#else
    (void)budget_bytes; (void)gate_bytes; (void)up_bytes; (void)down_bytes;
    return NULL;
#endif
}

void bn_moe_cache_free(void *cache) {
#if !defined(__EMSCRIPTEN__)
    if (!cache) return;
    BnMoECache *c = (BnMoECache *)cache;
    bn_platform_aligned_free(c->slab);
    free(c->entries);
    free(c->hash_table);
    free(c);
#else
    (void)cache;
#endif
}

int bn_moe_cache_capacity_internal(const BnMoECache *c) {
#if !defined(__EMSCRIPTEN__)
    return c ? c->n_slots : 0;
#else
    (void)c;
    return 0;
#endif
}

int bn_moe_cache_available_internal(const BnMoECache *c) {
#if !defined(__EMSCRIPTEN__)
    return c ? c->n_slots - c->pinned_slots : 0;
#else
    (void)c;
    return 0;
#endif
}

int bn_moe_cache_fits_internal(const BnMoECache *c, const size_t bytes[3]) {
#if !defined(__EMSCRIPTEN__)
    return c && bytes && bytes[0] <= c->gate_bytes &&
        bytes[1] <= c->up_bytes &&
        bytes[2] <= c->entry_bytes - c->gate_bytes - c->up_bytes;
#else
    (void)c; (void)bytes;
    return 0;
#endif
}

const uint8_t *bn_moe_cache_pin_internal(BnMoECache *c, int layer,
    int expert, int promote, int *slot) {
#if !defined(__EMSCRIPTEN__)
    if (!c || !slot) return NULL;
    int hi = moe_cache_probe(c, layer, expert);
    if (hi < 0) return NULL;
    int index = c->hash_table[hi];
    if (c->entries[index].pins == UINT_MAX) return NULL;
    if (!c->entries[index].pins) c->pinned_slots++;
    c->entries[index].pins++;
    if (promote) {
        moe_cache_lru_remove(c, index);
        moe_cache_lru_push_front(c, index);
    }
    *slot = index;
    return c->slab + (size_t)index * c->entry_bytes;
#else
    (void)c; (void)layer; (void)expert; (void)promote; (void)slot;
    return NULL;
#endif
}

uint8_t *bn_moe_cache_reserve_internal(BnMoECache *c, int layer,
    int expert, const size_t bytes[3], int *slot) {
#if !defined(__EMSCRIPTEN__)
    if (!slot || !bn_moe_cache_fits_internal(c, bytes) ||
        moe_cache_probe(c, layer, expert) >= 0) return NULL;
    int index;
    if (c->free_head >= 0) {
        index = c->free_head;
        c->free_head = c->entries[index].next;
    } else {
        index = moe_cache_evict(c);
        if (index < 0) return NULL;
    }
    BnMoECacheEntry *e = &c->entries[index];
    e->layer = layer; e->expert_idx = expert;
    e->published = 0; e->pins = 1; c->pinned_slots++;
    moe_cache_lru_push_front(c, index);
    *slot = index;
    return c->slab + (size_t)index * c->entry_bytes;
#else
    (void)c; (void)layer; (void)expert; (void)bytes; (void)slot;
    return NULL;
#endif
}

int bn_moe_cache_commit_internal(BnMoECache *c, int slot) {
#if !defined(__EMSCRIPTEN__)
    if (!c || slot < 0 || slot >= c->n_slots) return -1;
    BnMoECacheEntry *e = &c->entries[slot];
    if (!e->pins || e->published ||
        moe_cache_probe(c, e->layer, e->expert_idx) >= 0) return -1;
    uint32_t mask = (uint32_t)(c->hash_size - 1);
    uint32_t h = moe_cache_hash(e->layer, e->expert_idx) & mask;
    while (c->hash_table[h] >= 0) h = (h + 1u) & mask;
    c->hash_table[h] = slot;
    e->published = 1;
    return 0;
#else
    (void)c; (void)slot;
    return -1;
#endif
}

void bn_moe_cache_unpin_internal(BnMoECache *c, int slot) {
#if !defined(__EMSCRIPTEN__)
    if (c && slot >= 0 && slot < c->n_slots && c->entries[slot].pins) {
        c->entries[slot].pins--;
        if (!c->entries[slot].pins) {
            c->pinned_slots--;
            if (!c->entries[slot].published) {
                moe_cache_lru_remove(c, slot);
                c->entries[slot].layer = c->entries[slot].expert_idx = -1;
                c->entries[slot].next = c->free_head;
                c->free_head = slot;
            }
        }
    }
#else
    (void)c; (void)slot;
#endif
}

void bn_moe_cache_invalidate_internal(BnMoECache *c, int layer, int expert) {
#if !defined(__EMSCRIPTEN__)
    if (!c) return;
    int hi = moe_cache_probe(c, layer, expert);
    if (hi < 0) return;
    int slot = c->hash_table[hi];
    if (c->entries[slot].pins) return;
    moe_cache_lru_remove(c, slot);
    moe_cache_hash_remove(c, layer, expert);
    c->entries[slot].layer = -1;
    c->entries[slot].expert_idx = -1;
    c->entries[slot].next = c->free_head;
    c->free_head = slot;
#else
    (void)c; (void)layer; (void)expert;
#endif
}

void bn_moe_cache_print_stats(const BnMoEState *ms) {
    if (!ms) return;
    size_t total = ms->stats.cache_hits + ms->stats.cache_misses;
    if (total == 0) return;
    char hits_s[16], misses_s[16], rate_s[16];
    snprintf(hits_s, sizeof(hits_s), "%zu", ms->stats.cache_hits);
    snprintf(misses_s, sizeof(misses_s), "%zu", ms->stats.cache_misses);
    snprintf(rate_s, sizeof(rate_s), "%.1f%%", 100.0 * (double)ms->stats.cache_hits / (double)total);
    SH_LOG_INFO("MoE cache", "hits", hits_s, "misses", misses_s, "hit_rate", rate_s);
}

const BnPreparedWeight *bn_moe_acquire_prepared_projection(
    BnModel *model, const BnQWeight *weight) {
    /* Pread staging addresses can identify different experts over time. */
    if (!model || !bn_moe_io_has_mmap(bn_model_moe_io(model)))
        return NULL;
    /* Prefill touches many experts: misses must not evict decode's working
     * set. Locally prepared fallbacks already cover uncached projections. */
    return bn_backend_model_acquire_cached_cpu_prepared(bn_model_backend(model), weight);
}

void bn_moe_release_prepared_projection(
    BnModel *model, const BnPreparedWeight *prepared) {
    if (model)
        bn_backend_model_release_cpu_prepared(bn_model_backend(model), prepared);
}

int bn_moe_prepare_mmap_experts(BnModel *m) {
    if (!m || bn_model_anatomy(m) || !bn_moe_io_has_mmap(bn_model_moe_io(m)) ||
        !bn_model_backend(m))
        return -1;

    BnMoERoutePolicy route_policy = bn_moe_route_policy(&m->config);
    int prepared_count = 0;
    double t0 = bn_platform_time_ms();
    for (int l = 0; l < m->config.n_layers; l++) {
        BnMoEExpertMap *map = &m->weights.layers[l].moe.expert_map;
        for (int expert = 0; expert < route_policy.total_experts; expert++) {
            for (int proj = 0; proj < 3; proj++) {
                size_t offset, bytes;
                if (bn_moe_proj_info(map, expert, proj, &offset, &bytes) != 0 ||
                    bytes == 0)
                    continue;
                const uint8_t *base =
                    bn_moe_mmap_base_for_proj(bn_model_moe_io(m), map, proj);
                if (!base)
                    return -1;
                BnQWeight weight;
                if (!bn_moe_expert_projection_weight(&weight, base + offset,
                                                     map, proj))
                    return -1;
                if (bn_quant_prepared_qweight_size(&weight, NULL) == 0)
                    continue;
                const BnPreparedWeight *prepared =
                    bn_backend_model_acquire_cpu_prepared(
                        bn_model_backend(m), &weight);
                if (!prepared)
                    return -1;
                prepared_count++;
                bn_backend_model_release_cpu_prepared(bn_model_backend(m),
                                                       prepared);
            }
        }
    }
    char count_s[24], ms_s[32];
    snprintf(count_s, sizeof(count_s), "%d", prepared_count);
    snprintf(ms_s, sizeof(ms_s), "%.0f", bn_platform_time_ms() - t0);
    SH_LOG_INFO("CPU expert layouts prepared", "projections", count_s,
                "ms", ms_s);
    return 0;
}

int bn_moe_prefault_mmap(struct BnModel *m) {
    if (!m || bn_model_anatomy(m) || !bn_moe_io_has_mmap(bn_model_moe_io(m)) ||
        !bn_moe_policy_uses_expert_weights(&m->config))
        return -1;

#if defined(__EMSCRIPTEN__)
    return -1;
#else
    long ps = sysconf(_SC_PAGESIZE);
    size_t page = ps > 0 ? (size_t)ps : 4096;
    volatile uint8_t sink = 0;
    size_t touched_pages = 0;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&m->config);
    double t0 = bn_platform_time_ms();

    for (int l = 0; l < m->config.n_layers; l++) {
        BnMoEExpertMap *em = &m->weights.layers[l].moe.expert_map;
        if (em->expert_gate_bytes == 0 && em->expert_up_bytes == 0 &&
            em->expert_down_bytes == 0)
            continue;

        const size_t offsets[3] = { em->gate_offset, em->up_offset, em->down_offset };
        const size_t strides[3] = { em->gate_stride, em->up_stride, em->down_stride };
        const size_t sizes[3] = {
            em->expert_gate_bytes, em->expert_up_bytes, em->expert_down_bytes
        };

        for (int p = 0; p < 3; p++) {
            size_t proj_bytes = sizes[p];
            if (proj_bytes == 0) continue;
            const uint8_t *base =
                bn_moe_mmap_base_for_proj(bn_model_moe_io(m), em, p);
            if (!base) continue;

            for (int e = 0; e < route_policy.total_experts; e++) {
                const uint8_t *ptr = base + offsets[p] +
                    (size_t)e * (strides[p] ? strides[p] : proj_bytes);
                for (size_t off = 0; off < proj_bytes; off += page) {
                    sink ^= ptr[off];
                    touched_pages++;
                }
                sink ^= ptr[proj_bytes - 1];
            }
        }
    }

    char pages_s[32], ms_s[32], mb_s[32];
    snprintf(pages_s, sizeof(pages_s), "%zu", touched_pages);
    snprintf(ms_s, sizeof(ms_s), "%.0f", bn_platform_time_ms() - t0);
    snprintf(mb_s, sizeof(mb_s), "%.1f",
             (double)touched_pages * (double)page / (1024.0 * 1024.0));
    SH_LOG_INFO("MoE mmap prefault complete", "pages", pages_s, "MB", mb_s, "ms", ms_s);
    (void)sink;
    return 0;
#endif
}

// --- Unit test for LRU cache internals ---
int bn_moe_cache_test(void) {
#if !defined(__EMSCRIPTEN__)
    // Create a small cache: 4 slots, entry_bytes = 64
    BnMoECache *c = (BnMoECache *)bn_moe_cache_create(4 * 64, 32, 16, 16);
    if (!c) return -1;
    if (bn_moe_cache_capacity_internal(c) != 4 ||
        bn_moe_cache_capacity_internal(NULL) != 0) {
        bn_moe_cache_free(c);
        return -1;
    }

    // T1: Insert 4 entries (fills free list)
    uint8_t *s0 = moe_cache_insert(c, 0, 10);
    uint8_t *s1 = moe_cache_insert(c, 0, 20);
    uint8_t *s2 = moe_cache_insert(c, 1, 10);
    uint8_t *s3 = moe_cache_insert(c, 1, 20);
    if (!s0 || !s1 || !s2 || !s3) { bn_moe_cache_free(c); return -1; }
    // All 4 unique slab pointers
    if (s0 == s1 || s0 == s2 || s0 == s3 || s1 == s2 || s1 == s3 || s2 == s3)
        { bn_moe_cache_free(c); return -1; }

    // T2: Lookup all 4 — should hit
    if (!moe_cache_lookup(c, 0, 10)) { bn_moe_cache_free(c); return -1; }
    if (!moe_cache_lookup(c, 0, 20)) { bn_moe_cache_free(c); return -1; }
    if (!moe_cache_lookup(c, 1, 10)) { bn_moe_cache_free(c); return -1; }
    if (!moe_cache_lookup(c, 1, 20)) { bn_moe_cache_free(c); return -1; }

    // T3: Lookup non-existent — should miss
    if (moe_cache_lookup(c, 2, 10)) { bn_moe_cache_free(c); return -1; }
    if (moe_cache_lookup(c, 0, 30)) { bn_moe_cache_free(c); return -1; }

    // T4: Insert 5th entry — should evict LRU tail
    // LRU order after T2 lookups: MRU → (1,20) → (1,10) → (0,20) → (0,10) ← LRU
    // So (0,10) should be evicted
    moe_cache_insert(c, 2, 50);
    if (moe_cache_lookup(c, 0, 10)) { bn_moe_cache_free(c); return -1; }  // evicted
    if (!moe_cache_lookup(c, 2, 50)) { bn_moe_cache_free(c); return -1; } // present
    if (!moe_cache_lookup(c, 0, 20)) { bn_moe_cache_free(c); return -1; } // still present

    // T5: Promote (0,20) to MRU by looking it up, then insert 2 more to evict others
    moe_cache_lookup(c, 0, 20);  // promote to MRU
    moe_cache_insert(c, 3, 1);   // evicts LRU
    moe_cache_insert(c, 3, 2);   // evicts next LRU
    // (0,20) should survive (it was promoted to MRU)
    if (!moe_cache_lookup(c, 0, 20)) { bn_moe_cache_free(c); return -1; }

    // T6: Hash collision test — insert many entries with same layer
    // (forces hash collisions and tests backshift deletion)
    bn_moe_cache_free(c);
    c = (BnMoECache *)bn_moe_cache_create(8 * 64, 32, 16, 16);
    if (!c) return -1;
    if (bn_moe_cache_capacity_internal(c) != 8) {
        bn_moe_cache_free(c);
        return -1;
    }
    for (int i = 0; i < 8; i++)
        moe_cache_insert(c, 0, i);
    // All 8 should be present
    for (int i = 0; i < 8; i++) {
        if (!moe_cache_lookup(c, 0, i)) { bn_moe_cache_free(c); return -1; }
    }
    // Insert 9th — evicts LRU (expert 0, since lookups promoted 1-7)
    moe_cache_insert(c, 1, 0);
    if (moe_cache_lookup(c, 0, 0)) { bn_moe_cache_free(c); return -1; }   // evicted
    if (!moe_cache_lookup(c, 1, 0)) { bn_moe_cache_free(c); return -1; }  // present
    // Remaining 1-7 still present
    for (int i = 1; i < 8; i++) {
        if (!moe_cache_lookup(c, 0, i)) { bn_moe_cache_free(c); return -1; }
    }

    // T7: Repeatedly replace the entire resident set. This exercises wrapped
    // probe clusters after many evictions, where every new key must remain
    // reachable through the hash table.
    for (int round = 1; round <= 128; round++) {
        for (int i = 0; i < 8; i++)
            moe_cache_insert(c, round, i);
        for (int i = 0; i < 8; i++) {
            if (!moe_cache_lookup(c, round, i)) {
                bn_moe_cache_free(c);
                return -1;
            }
        }
    }

    bn_moe_cache_free(c);
    return 0;
#else
    return 0;  // no cache on EMSCRIPTEN
#endif
}

const BnPreparedWeight *bn_moe_prepared_dense_projection(
    const BnModel *model, const BnQWeight *weight) {
    return bn_backend_model_prepared_qweight(bn_model_backend(model), weight);
}

static int valid_reader(const BnMoEExpertReader *reader) {
    return reader && reader->identity && reader->read && reader->n_layers>0 &&
        reader->n_experts>0 && reader->bytes[0] && reader->bytes[1] && reader->bytes[2] &&
        reader->bytes[0]<=SIZE_MAX-reader->bytes[1] &&
        reader->bytes[0]+reader->bytes[1]<=SIZE_MAX-reader->bytes[2];
}
void *bn_moe_cache_create_reader(size_t budget,const BnMoEExpertReader *reader) {
    return bn_moe_cache_create_reader_policy(budget,reader,BN_MOE_CACHE_PAGES_HUGE_HINT);
}
void *bn_moe_cache_create_reader_policy(size_t budget,const BnMoEExpertReader *reader,
    BnMoECachePagePolicy pages) {
    if(!valid_reader(reader))return NULL;
    BnMoECache *cache=bn_moe_cache_create_policy(budget,reader->bytes[0],reader->bytes[1],reader->bytes[2],pages);
#ifndef __EMSCRIPTEN__
    if(cache)cache->reader=*reader;
#endif
    return cache;
}
int bn_moe_host_expert_acquire_reader(const BnMoEExpertReader *reader,void *cache_ptr,
    BnMoEStats *stats,int layer,int expert,int admit,void *const staging[3],
    const size_t capacities[3],BnMoEHostExpert *out) {
    if(!out)return -1;
    memset(out,0,sizeof(*out));out->slot=-1;
    if(!valid_reader(reader) || layer<0 || layer>=reader->n_layers ||
       expert<0 || expert>=reader->n_experts || (admit!=0 && admit!=1))return -1;
    const uint8_t *entry=NULL;
    uint8_t *reserved=NULL;
    BnMoECache *cache=(BnMoECache*)cache_ptr;
    size_t total=reader->bytes[0]+reader->bytes[1]+reader->bytes[2];
#ifndef __EMSCRIPTEN__
    if(cache && (cache->reader.identity!=reader->identity || cache->reader.context!=reader->context ||
       cache->reader.read!=reader->read || cache->reader.n_layers!=reader->n_layers ||
       cache->reader.n_experts!=reader->n_experts ||
       memcmp(cache->reader.bytes,reader->bytes,sizeof(reader->bytes))))return -1;
    if(cache)entry=bn_moe_cache_pin_internal(cache,layer,expert,admit,&out->slot);
    if(entry) {
        if(stats){stats->gpu_host_cache_hits++;stats->gpu_host_cache_hit_bytes+=total;}
    } else {
        if(cache && stats){stats->gpu_host_cache_misses++;stats->gpu_host_cache_miss_bytes+=total;}
        if(cache && admit)reserved=bn_moe_cache_reserve_internal(cache,layer,expert,reader->bytes,&out->slot);
        if(reserved)out->cache=cache;
#else
    if(cache)return -1;
    {
#endif
        if(!reserved)for(int p=0;p<3;p++)
            if(!staging || !capacities || !staging[p] || capacities[p]<reader->bytes[p])return -1;
        size_t offset=0;
        for(int p=0;p<3;p++) {
            void *dst=reserved?reserved+offset:staging[p];
            double start=bn_platform_time_ms();
            int rc=reader->read(reader->context,layer,expert,p,dst,reader->bytes[p]);
            if(stats)stats->io_time_ms+=bn_platform_time_ms()-start;
            if(rc){bn_moe_host_expert_release(out);return -1;}
            out->projection[p]=dst;offset+=reader->bytes[p];
            if(stats){stats->io_count++;stats->io_bytes+=reader->bytes[p];}
        }
#ifndef __EMSCRIPTEN__
        if(reserved) {
            if(bn_moe_cache_commit_internal(cache,out->slot)){bn_moe_host_expert_release(out);return -1;}
            entry=reserved;
        }
#endif
    }
    if(entry) {
        out->projection[0]=entry;out->projection[1]=entry+reader->bytes[0];
        out->projection[2]=entry+reader->bytes[0]+reader->bytes[1];out->cache=cache;
    }
    return 0;
}

int bn_moe_cache_is_reader_internal(const BnMoECache *cache) {
#ifndef __EMSCRIPTEN__
    return cache && cache->reader.identity;
#else
    (void)cache;return 0;
#endif
}
