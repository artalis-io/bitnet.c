#include "moe_internal.h"

// --- Expert LRU Cache (pread pipeline) ---
#if !defined(__EMSCRIPTEN__)

typedef struct {
    int layer;          // -1 = empty
    int expert_idx;
    int prev, next;     // intrusive LRU doubly-linked list (indices, -1 = sentinel)
} BnMoECacheEntry;

struct BnMoECache {
    uint8_t *slab;              // [n_slots * entry_bytes], 32-byte aligned
    size_t entry_bytes;         // gate_bytes + up_bytes + down_bytes
    size_t gate_bytes;          // size of gate projection
    size_t up_bytes;            // size of up projection
    int n_slots;

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

// Hash table removal with backshift deletion
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

    // Backshift deletion
    c->hash_table[idx] = -1;
    for (int i = 1; i < c->hash_size; i++) {
        int next = (int)(((uint32_t)idx + (uint32_t)i) & mask);
        int slot = c->hash_table[next];
        if (slot < 0) break;  // chain ended
        uint32_t natural = moe_cache_hash(c->entries[slot].layer,
                                            c->entries[slot].expert_idx) & mask;
        // Check if this element's natural position is at or before the gap
        int gap = idx;
        // Element belongs before the gap if moving it wouldn't break its probe chain
        int should_move;
        if (next >= gap)
            should_move = (int)natural <= gap || (int)natural > next;
        else
            should_move = (int)natural <= gap && (int)natural > next;
        if (should_move) {
            c->hash_table[gap] = slot;
            c->hash_table[next] = -1;
            idx = next;  // new gap
        }
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
    return moe_cache_lookup(c, layer, expert_idx);
}

uint8_t *bn_moe_cache_insert_internal(BnMoECache *c, int layer, int expert_idx) {
    return moe_cache_insert(c, layer, expert_idx);
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
#if !defined(__EMSCRIPTEN__)
    if (budget_bytes == 0) return NULL;
    size_t entry_bytes = gate_bytes + up_bytes + down_bytes;
    if (entry_bytes == 0) return NULL;

    size_t raw_slots = budget_bytes / entry_bytes;
    if (raw_slots < 1) return NULL;
    if (raw_slots > (size_t)INT_MAX / 2) raw_slots = (size_t)INT_MAX / 2;  // cap to avoid overflow
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
#if defined(__APPLE__) || defined(__linux__)
    if (posix_memalign((void **)&c->slab, 32, slab_size) != 0) {
        free(c);
        return NULL;
    }
#else
    c->slab = (uint8_t *)malloc(slab_size);
    if (!c->slab) { free(c); return NULL; }
#endif

    c->entries = (BnMoECacheEntry *)calloc((size_t)n_slots, sizeof(BnMoECacheEntry));
    c->hash_table = (int *)malloc((size_t)hs * sizeof(int));
    if (!c->entries || !c->hash_table) {
        free(c->slab); free(c->entries); free(c->hash_table); free(c);
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
    free(c->slab);
    free(c->entries);
    free(c->hash_table);
    free(c);
#else
    (void)cache;
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

int bn_moe_prefault_mmap(struct BnModel *m) {
    if (!m || !bn_model_moe_io(m)->mmap_base || m->config.n_experts <= 0)
        return -1;

#if defined(__EMSCRIPTEN__)
    return -1;
#else
    long ps = sysconf(_SC_PAGESIZE);
    size_t page = ps > 0 ? (size_t)ps : 4096;
    volatile uint8_t sink = 0;
    size_t touched_pages = 0;
    double t0 = bn_platform_time_ms();

    for (int l = 0; l < m->config.n_layers; l++) {
        BnMoEExpertMap *em = &m->weights.layers[l].moe.expert_map;
        if (em->expert_gate_bytes == 0 && em->expert_up_bytes == 0 &&
            em->expert_down_bytes == 0)
            continue;

        const size_t bases[3] = { em->gate_offset, em->up_offset, em->down_offset };
        const size_t sizes[3] = {
            em->expert_gate_bytes, em->expert_up_bytes, em->expert_down_bytes
        };

        for (int p = 0; p < 3; p++) {
            size_t proj_bytes = sizes[p];
            if (proj_bytes == 0) continue;

            for (int e = 0; e < m->config.n_experts; e++) {
                const uint8_t *ptr = bn_model_moe_io(m)->mmap_base + bases[p] +
                                     (size_t)e * proj_bytes;
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

    bn_moe_cache_free(c);
    return 0;
#else
    return 0;  // no cache on EMSCRIPTEN
#endif
}
