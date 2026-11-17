#include "gpu_moe_cache.h"
#include "sh_log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

// --- Hash helpers (same algorithm as CPU expert cache in moe.c) ---

static uint32_t cache_hash(int layer, int expert_idx) {
    uint32_t key = (uint32_t)layer * 65537u + (uint32_t)expert_idx;
    key ^= key >> 16; key *= 0x45d9f3bu; key ^= key >> 16;
    return key;
}

static int cache_probe(const BnGPUMoECache *c, int layer, int expert_idx) {
    uint32_t mask = (uint32_t)(c->hash_size - 1);
    uint32_t h = cache_hash(layer, expert_idx) & mask;
    for (unsigned i = 0; i < c->hash_size; i++) {
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

static void cache_destroy_handles(BnGPUMoECache *c,BnGPUMoECacheEntry *e) {
    bn_gpu_backend_destroy_buffer(c->gpu,e->gate_gpu);
    bn_gpu_backend_destroy_buffer(c->gpu,e->up_gpu);
    bn_gpu_backend_destroy_buffer(c->gpu,e->down_gpu);
    e->gate_gpu=e->up_gpu=e->down_gpu=NULL;e->retained=0;
}
static int cache_evict(BnGPUMoECache *c,int retain) {
    int slot = c->lru_tail;
    while(slot>=0 && (c->entries[slot].pins || c->entries[slot].generation==UINT64_MAX))
        slot=c->entries[slot].prev;
    if (slot < 0) return -1;

    BnGPUMoECacheEntry *e = &c->entries[slot];
    if(retain)e->retained=1;
    else cache_destroy_handles(c,e);

    // Remove from hash + LRU
    hash_remove(c, e->layer, e->expert_idx);
    lru_remove(c, slot);

    e->layer = -1;
    e->published=0;c->evictions++;
    return slot;
}

static int allocation_fits(size_t count,size_t width) {
    return width && count<=SIZE_MAX/width;
}

// --- Public API ---

BnGPUMoECache *bn_gpu_moe_cache_create(size_t budget_bytes, size_t entry_bytes,
                                       int max_entries, BnGPUBackend *gpu) {
    if (!gpu || !bn_gpu_backend_can_destroy_buffer(gpu) || budget_bytes == 0 || entry_bytes == 0 || max_entries<0) return NULL;
    size_t raw_slots=budget_bytes/entry_bytes;
    size_t limit=((size_t)INT_MAX+1u)/4u;
    if(raw_slots>limit)raw_slots=limit;
    int n_slots = (int)raw_slots;
    if (max_entries > 0 && n_slots > max_entries)
        n_slots = max_entries;
    if (n_slots < 1) return NULL;

    unsigned hash_size = 1;
    while (hash_size < (unsigned)n_slots * 2) hash_size <<= 1;

    if(!allocation_fits((size_t)n_slots,sizeof(BnGPUMoECacheEntry)) ||
       !allocation_fits((size_t)hash_size,sizeof(int)))return NULL;
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

static int cache_lookup(BnGPUMoECache *c, int layer, int expert_idx,
    void **gate_out, void **up_out, void **down_out, int promote) {
    if (!c || layer<0 || expert_idx<0 || !gate_out || !up_out || !down_out) return 0;
    int slot = cache_probe(c, layer, expert_idx);
    if (slot < 0 || !c->entries[slot].published) { c->misses++; return 0; }

    if (promote) {
        lru_remove(c, slot);
        lru_push_front(c, slot);
    }

    BnGPUMoECacheEntry *e = &c->entries[slot];
    *gate_out = e->gate_gpu;
    *up_out = e->up_gpu;
    *down_out = e->down_gpu;
    c->hits++;
    return 1;
}

int bn_gpu_moe_cache_lookup(BnGPUMoECache *c, int layer, int expert_idx,
    void **gate_out, void **up_out, void **down_out) {
    return cache_lookup(c, layer, expert_idx, gate_out, up_out, down_out, 1);
}

int bn_gpu_moe_cache_lookup_prefill(BnGPUMoECache *c, int layer, int expert_idx,
    void **gate_out, void **up_out, void **down_out) {
    return cache_lookup(c, layer, expert_idx, gate_out, up_out, down_out, 0);
}

static int valid_handles(void *g,void *u,void *d) {
    return (g||u||d) && (!g||g!=u) && (!g||g!=d) && (!u||u!=d);
}
int bn_gpu_moe_cache_acquire(BnGPUMoECache *c,int layer,int expert,int promote,BnGPUMoELease *out) {
    if(!c || !out || layer<0 || expert<0 || (promote!=0 && promote!=1))return -1;
    int slot=cache_probe(c,layer,expert);
    if(slot<0 || !c->entries[slot].published){c->misses++;return 0;}
    BnGPUMoECacheEntry *e=&c->entries[slot];if(e->pins==UINT_MAX)return -1;
    if(promote){lru_remove(c,slot);lru_push_front(c,slot);}e->pins++;c->hits++;
    *out=(BnGPUMoELease){c,slot,e->generation,e->gate_gpu,e->up_gpu,e->down_gpu};return 1;
}
int bn_gpu_moe_cache_release(BnGPUMoELease *lease) {
    if(!lease)return -1;
    if(!lease->cache)return 0;
    BnGPUMoECache *c=lease->cache;
    if(lease->slot<0 || lease->slot>=c->n_slots)return -1;
    BnGPUMoECacheEntry *e=&c->entries[lease->slot];
    if(!e->published || !e->pins || e->generation!=lease->generation ||
       e->gate_gpu!=lease->gate || e->up_gpu!=lease->up || e->down_gpu!=lease->down)return -1;
    e->pins--;*lease=(BnGPUMoELease){0};return 0;
}
static int cache_reserve(BnGPUMoECache *c,int layer,int expert,
    BnGPUMoEReservation *out,int retain) {
    if(!c || !out || layer<0 || expert<0)return -1;
    int found=cache_probe(c,layer,expert);
    if(found>=0)return c->entries[found].pending?BN_GPU_MOE_CACHE_BUSY:-1;
    int slot=c->free_head,prev=-1;
    while(slot>=0 && c->entries[slot].generation==UINT64_MAX){prev=slot;slot=c->entries[slot].next;}
    if(slot>=0){if(prev<0)c->free_head=c->entries[slot].next;else c->entries[prev].next=c->entries[slot].next;}
    else slot=cache_evict(c,retain);
    if(slot<0)return BN_GPU_MOE_CACHE_BUSY;
    BnGPUMoECacheEntry *e=&c->entries[slot];e->layer=layer;e->expert_idx=expert;
    e->prev=e->next=-1;e->pending=1;e->published=0;e->pins=0;e->generation++;
    hash_insert(c,layer,expert,slot);
    *out=(BnGPUMoEReservation){c,slot,e->generation};return 0;
}
int bn_gpu_moe_cache_reserve(BnGPUMoECache *c,int layer,int expert,BnGPUMoEReservation *out) {
    return cache_reserve(c,layer,expert,out,0);
}
int bn_gpu_moe_cache_reserve_reuse(BnGPUMoECache *c,int layer,int expert,BnGPUMoEReservation *out) {
    return cache_reserve(c,layer,expert,out,1);
}
static BnGPUMoECacheEntry *reservation_entry(const BnGPUMoEReservation *r) {
    if(!r || !r->cache || r->slot<0 || r->slot>=r->cache->n_slots)return NULL;
    BnGPUMoECacheEntry *e=&r->cache->entries[r->slot];
    return e->pending && e->generation==r->generation?e:NULL;
}
int bn_gpu_moe_cache_reserved_buffers(const BnGPUMoEReservation *r,
    void **g,void **u,void **d) {
    const BnGPUMoECacheEntry *e=reservation_entry(r);
    if(!e||!g||!u||!d)return -1;
    if(!e->retained)return 0;
    *g=e->gate_gpu;*u=e->up_gpu;*d=e->down_gpu;return 1;
}
int bn_gpu_moe_cache_discard_reserved_buffers(BnGPUMoEReservation *r) {
    BnGPUMoECacheEntry *e=reservation_entry(r);if(!e)return -1;
    if(e->retained)cache_destroy_handles(r->cache,e);
    return 0;
}
int bn_gpu_moe_cache_commit(BnGPUMoEReservation *r,void *g,void *u,void *d,BnGPUMoELease *out) {
    BnGPUMoECacheEntry *e=reservation_entry(r);
    if(!e || !out || !valid_handles(g,u,d))return -1;
    if(e->retained&&(g!=e->gate_gpu||u!=e->up_gpu||d!=e->down_gpu))return -1;
    e->gate_gpu=g;e->up_gpu=u;e->down_gpu=d;e->pending=0;e->published=1;e->pins=1;
    e->retained=0;
    lru_push_front(r->cache,r->slot);
    *out=(BnGPUMoELease){r->cache,r->slot,r->generation,g,u,d};*r=(BnGPUMoEReservation){0};return 0;
}
int bn_gpu_moe_cache_abort(BnGPUMoEReservation *r) {
    BnGPUMoECacheEntry *e=reservation_entry(r);if(!e)return -1;
    BnGPUMoECache *c=r->cache;hash_remove(c,e->layer,e->expert_idx);
    if(e->retained)cache_destroy_handles(c,e);
    e->layer=-1;e->pending=0;e->next=c->free_head;e->prev=-1;c->free_head=r->slot;c->aborts++;
    *r=(BnGPUMoEReservation){0};return 0;
}
int bn_gpu_moe_cache_stats(const BnGPUMoECache *c,BnGPUMoECacheStats *out) {
    if(!c || !out)return -1;
    BnGPUMoECacheStats s={0};
    s.capacity_payload=(size_t)c->n_slots*c->entry_bytes;
    for(int i=0;i<c->n_slots;i++) {
        const BnGPUMoECacheEntry *e=&c->entries[i];s.published+=e->published!=0;s.pending+=e->pending!=0;
        s.pinned_entries+=e->pins!=0;if(s.leases>SIZE_MAX-e->pins)return -1;s.leases+=e->pins;
    }
    s.charged_payload=(s.published+s.pending)*c->entry_bytes;
    s.hits=c->hits;s.misses=c->misses;s.evictions=c->evictions;s.aborts=c->aborts;*out=s;return 0;
}
int bn_gpu_moe_cache_insert(BnGPUMoECache *c,int layer,int expert,void *g,void *u,void *d) {
    if(!valid_handles(g,u,d))return -1;
    BnGPUMoEReservation r={0};BnGPUMoELease lease={0};
    if(bn_gpu_moe_cache_reserve(c,layer,expert,&r))return -1;
    if(bn_gpu_moe_cache_commit(&r,g,u,d,&lease)){bn_gpu_moe_cache_abort(&r);return -1;}
    return bn_gpu_moe_cache_release(&lease);
}
int bn_gpu_moe_cache_close(BnGPUMoECache *c) {
    if(!c)return 0;
    for(int i=0;i<c->n_slots;i++)if(c->entries[i].pins || c->entries[i].pending)return BN_GPU_MOE_CACHE_BUSY;
    for(int i=0;i<c->n_slots;i++) {
        BnGPUMoECacheEntry *e=&c->entries[i];if(!e->published)continue;
        bn_gpu_backend_destroy_buffer(c->gpu,e->gate_gpu);
        bn_gpu_backend_destroy_buffer(c->gpu,e->up_gpu);
        bn_gpu_backend_destroy_buffer(c->gpu,e->down_gpu);
    }
    free(c->entries);free(c->hash_table);free(c);return 0;
}
void bn_gpu_moe_cache_free(BnGPUMoECache *c) {
    if(bn_gpu_moe_cache_close(c))fprintf(stderr,"GPU MoE cache still has live leases or reservations\n");
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
