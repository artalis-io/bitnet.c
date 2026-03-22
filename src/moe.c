#include "moe.h"
#include "platform.h"
#include "quant.h"
#include "simd_helpers.h"
#include "sh_log.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#ifndef __EMSCRIPTEN__
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// --- Expert LRU Cache (pread pipeline) ---
#if !defined(__EMSCRIPTEN__)

typedef struct {
    int layer;          // -1 = empty
    int expert_idx;
    int prev, next;     // intrusive LRU doubly-linked list (indices, -1 = sentinel)
} BnMoECacheEntry;

typedef struct {
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
} BnMoECache;

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

// --- I/O Prefetch Thread (pread pipeline) ---
#if !defined(__EMSCRIPTEN__)

typedef struct {
    pthread_t thread;
    pthread_mutex_t mtx;
    pthread_cond_t req_cv;      // I/O thread waits for work
    pthread_cond_t done_cv;     // main thread waits for completion
    int fd;
    int active;                 // request in progress
    int shutdown;
    int success;                // result of last I/O
    double io_time_ms;          // accumulated pread time
    size_t io_bytes;            // accumulated bytes read
    // Request: up to 3 preads per submission (gate, up, down)
    struct { uint8_t *buf; size_t size; off_t offset; } reqs[3];
    int n_reqs;
} BnMoEPrefetch;

static void *moe_prefetch_worker(void *arg) {
    BnMoEPrefetch *pf = (BnMoEPrefetch *)arg;
    pthread_mutex_lock(&pf->mtx);
    while (1) {
        while (!pf->active && !pf->shutdown)
            pthread_cond_wait(&pf->req_cv, &pf->mtx);
        if (pf->shutdown) break;

        // Copy request params under lock
        int n = pf->n_reqs;
        uint8_t *bufs[3]; size_t sizes[3]; off_t offsets[3];
        for (int i = 0; i < n; i++) {
            bufs[i] = pf->reqs[i].buf;
            sizes[i] = pf->reqs[i].size;
            offsets[i] = pf->reqs[i].offset;
        }
        pthread_mutex_unlock(&pf->mtx);

        // Do I/O without holding lock
        double t0 = bn_platform_time_ms();
        int ok = 1;
        size_t bytes = 0;
        for (int i = 0; i < n; i++) {
            ssize_t r = pread(pf->fd, bufs[i], sizes[i], offsets[i]);
            if (r != (ssize_t)sizes[i]) { ok = 0; break; }
            bytes += sizes[i];
        }
        double elapsed = bn_platform_time_ms() - t0;

        pthread_mutex_lock(&pf->mtx);
        pf->success = ok;
        pf->io_time_ms += elapsed;
        pf->io_bytes += bytes;
        pf->active = 0;
        pthread_cond_signal(&pf->done_cv);
    }
    pthread_mutex_unlock(&pf->mtx);
    return NULL;
}

static BnMoEPrefetch *moe_prefetch_init(int fd) {
    BnMoEPrefetch *pf = (BnMoEPrefetch *)calloc(1, sizeof(BnMoEPrefetch));
    if (!pf) return NULL;
    pf->fd = fd;
    pthread_mutex_init(&pf->mtx, NULL);
    pthread_cond_init(&pf->req_cv, NULL);
    pthread_cond_init(&pf->done_cv, NULL);
    if (pthread_create(&pf->thread, NULL, moe_prefetch_worker, pf) != 0) {
        pthread_mutex_destroy(&pf->mtx);
        pthread_cond_destroy(&pf->req_cv);
        pthread_cond_destroy(&pf->done_cv);
        free(pf);
        return NULL;
    }
    return pf;
}

static void moe_prefetch_free(BnMoEPrefetch *pf) {
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    pf->shutdown = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
    pthread_join(pf->thread, NULL);
    pthread_mutex_destroy(&pf->mtx);
    pthread_cond_destroy(&pf->req_cv);
    pthread_cond_destroy(&pf->done_cv);
    free(pf);
}

// Post a 2-read prefetch request (gate+up). Non-blocking.
static void moe_prefetch_start2(BnMoEPrefetch *pf,
                                uint8_t *buf1, size_t size1, off_t off1,
                                uint8_t *buf2, size_t size2, off_t off2) {
    pthread_mutex_lock(&pf->mtx);
    pf->reqs[0].buf = buf1; pf->reqs[0].size = size1; pf->reqs[0].offset = off1;
    pf->reqs[1].buf = buf2; pf->reqs[1].size = size2; pf->reqs[1].offset = off2;
    pf->n_reqs = 2;
    pf->active = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
}

// Post a 1-read prefetch request (down). Non-blocking.
static void moe_prefetch_start1(BnMoEPrefetch *pf,
                                uint8_t *buf1, size_t size1, off_t off1) {
    pthread_mutex_lock(&pf->mtx);
    pf->reqs[0].buf = buf1; pf->reqs[0].size = size1; pf->reqs[0].offset = off1;
    pf->n_reqs = 1;
    pf->active = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
}

// Wait for prefetch to complete. Returns success flag.
static int moe_prefetch_wait(BnMoEPrefetch *pf) {
    pthread_mutex_lock(&pf->mtx);
    while (pf->active)
        pthread_cond_wait(&pf->done_cv, &pf->mtx);
    int ok = pf->success;
    pthread_mutex_unlock(&pf->mtx);
    return ok;
}

#endif // !__EMSCRIPTEN__

// Backend-selected rmsnorm (same selection as transformer.c)
#ifdef __ARM_NEON
extern void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
extern void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
extern void bn_transformer_rmsnorm_wasm(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_wasm
#else
extern void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_scalar
#endif

// --- Phase 4: Vectorized router ---

typedef struct {
    float *logits;
    const float *router_w;
    const float *x;
    int dim;
} BnRouterCtx;

static void moe_router_range(void *ctx, int start, int end) {
    BnRouterCtx *c = (BnRouterCtx *)ctx;
    for (int e = start; e < end; e++) {
        const float *row = c->router_w + (size_t)e * c->dim;
        float sum = 0.0f;
#ifdef __ARM_NEON
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 15 < c->dim; d += 16) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(row + d),      vld1q_f32(c->x + d));
            acc1 = vfmaq_f32(acc1, vld1q_f32(row + d + 4),  vld1q_f32(c->x + d + 4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(row + d + 8),  vld1q_f32(c->x + d + 8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(row + d + 12), vld1q_f32(c->x + d + 12));
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        sum = vaddvq_f32(acc0);
        for (; d < c->dim; d++)
            sum += row[d] * c->x[d];
#elif defined(__AVX2__)
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        int d = 0;
        for (; d + 15 < c->dim; d += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d),     _mm256_loadu_ps(c->x + d),     a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d + 8), _mm256_loadu_ps(c->x + d + 8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 hi = _mm256_extractf128_ps(a0, 1);
        __m128 lo = _mm256_castps256_ps128(a0);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);
        for (; d < c->dim; d++)
            sum += row[d] * c->x[d];
#else
        for (int d = 0; d < c->dim; d++)
            sum += row[d] * c->x[d];
#endif
        c->logits[e] = sum;
    }
}

// Router: SIMD matvec -> softmax -> top-K selection
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k, BnThreadPool *pool) {
    // Router matvec: vectorized + thread-dispatched
    BnRouterCtx rctx = { ms->router_logits, router_w, x, dim };
    BnTPTask rtask = { moe_router_range, &rctx, n_experts };
    bn_tp_dispatch(pool, &rtask, 1);

    // Softmax over all experts
    float max_val = ms->router_logits[0];
    for (int e = 1; e < n_experts; e++)
        if (ms->router_logits[e] > max_val)
            max_val = ms->router_logits[e];

    float sum = 0.0f;
    for (int e = 0; e < n_experts; e++) {
        ms->router_logits[e] = expf(ms->router_logits[e] - max_val);
        sum += ms->router_logits[e];
    }
    for (int e = 0; e < n_experts; e++)
        ms->router_logits[e] /= sum;

    // Top-K selection (partial sort)
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_experts; e++) {
            // Skip already-selected experts
            int skip = 0;
            for (int j = 0; j < i; j++)
                if (ms->expert_indices[j] == e) { skip = 1; break; }
            if (skip) continue;
            if (ms->router_logits[e] > best_val) {
                best_val = ms->router_logits[e];
                best = e;
            }
        }
        ms->expert_indices[i] = best;
        ms->expert_weights[i] = best_val;
    }

    // Normalize selected weights to sum to 1.0
    float wsum = 0.0f;
    for (int i = 0; i < k; i++)
        wsum += ms->expert_weights[i];
    if (wsum > 0.0f) {
        for (int i = 0; i < k; i++)
            ms->expert_weights[i] /= wsum;
    }
}

// Get offset and size for an expert projection.
// proj: 0=gate, 1=up, 2=down
static int moe_proj_info(const BnMoEExpertMap *map, int expert_idx, int proj,
                          size_t *offset, size_t *proj_bytes) {
    switch (proj) {
        case 0:
            *offset = map->gate_offset + (size_t)expert_idx * map->expert_gate_bytes;
            *proj_bytes = map->expert_gate_bytes;
            return 0;
        case 1:
            *offset = map->up_offset + (size_t)expert_idx * map->expert_up_bytes;
            *proj_bytes = map->expert_up_bytes;
            return 0;
        case 2:
            *offset = map->down_offset + (size_t)expert_idx * map->expert_down_bytes;
            *proj_bytes = map->expert_down_bytes;
            return 0;
        default:
            return -1;
    }
}

// madvise helper: issue WILLNEED or DONTNEED for expert projections.
// proj_mask: bitmask (1=gate, 2=up, 4=down), advice: MADV_WILLNEED or MADV_DONTNEED.
#if !defined(__EMSCRIPTEN__)
static void moe_madvise_experts(const BnMoEState *ms, const BnMoEExpertMap *map,
                                 const int *indices, int n, int advice, int proj_mask) {
    if (!ms->io.mmap_base) return;
    long page_size = sysconf(_SC_PAGESIZE);
    for (int k = 0; k < n; k++) {
        int eidx = indices[k];
        if (eidx < 0) continue;
        for (int proj = 0; proj < 3; proj++) {
            if (!((proj_mask >> proj) & 1)) continue;
            size_t offset, proj_bytes;
            moe_proj_info(map, eidx, proj, &offset, &proj_bytes);
            // Page-align: round down start, round up end
            uintptr_t addr = (uintptr_t)ms->io.mmap_base + offset;
            uintptr_t aligned_start = addr & ~((uintptr_t)page_size - 1);
            size_t aligned_len = (addr + proj_bytes - aligned_start + page_size - 1) & ~((size_t)page_size - 1);
            madvise((void *)aligned_start, aligned_len, advice);
        }
    }
}
#endif

// Load one expert projection into a specific buffer.
// Returns pointer to data (mmap pointer or buf), or NULL on error.
static const void *moe_load_expert_proj_into(BnMoEState *ms, const BnMoEExpertMap *map,
                                              int expert_idx, int proj,
                                              uint8_t *buf, size_t buf_size) {
    size_t offset, proj_bytes;
    if (moe_proj_info(map, expert_idx, proj, &offset, &proj_bytes) < 0)
        return NULL;

    ms->stats.io_bytes += proj_bytes;
    ms->stats.io_count++;

    if (ms->io.mmap_base)
        return ms->io.mmap_base + offset;

#if !defined(__EMSCRIPTEN__)
    if (ms->io.fd < 0 || proj_bytes > buf_size) return NULL;
    double t0 = bn_platform_time_ms();
    ssize_t n = pread(ms->io.fd, buf, proj_bytes, (off_t)offset);
    ms->stats.io_time_ms += bn_platform_time_ms() - t0;
    if (n != (ssize_t)proj_bytes) return NULL;
    return buf;
#else
    return NULL;
#endif
}

// Load one expert projection into the default expert_buf.
static const void *moe_load_expert_proj(BnMoEState *ms, const BnMoEExpertMap *map,
                                         int expert_idx, int proj) {
    return moe_load_expert_proj_into(ms, map, expert_idx, proj,
                                      ms->io.buf, ms->io.buf_size);
}

// Build a temporary BnQWeight from pread'd expert data
static BnQWeight moe_make_qweight(const void *data, int type, int rows, int cols) {
    BnQWeight w = {0};
    w.data = data;
    w.type = type;
    w.rows = rows;
    w.cols = cols;
    // Per-block quants have embedded scales
    if (type == BN_GGUF_TENSOR_I2_S) {
        size_t nelements = (size_t)rows * cols;
        const uint8_t *base = (const uint8_t *)data;
        memcpy(&w.scale, base + nelements / 4, sizeof(float));
    } else {
        w.scale = 1.0f;
    }
    return w;
}

// --- Phase 3: SwiGLU range function for parallel dispatch ---

typedef struct {
    float *hb;
    const float *gate;
    const float *up;
} BnSwiGLUCtx;

static void moe_swiglu_range(void *ctx, int start, int end) {
    BnSwiGLUCtx *c = (BnSwiGLUCtx *)ctx;
    int i = start;
#ifdef __AVX2__
    for (; i + 7 < end; i += 8) {
        __m256 g = _mm256_loadu_ps(c->gate + i);
        __m256 u = _mm256_loadu_ps(c->up + i);
        _mm256_storeu_ps(c->hb + i, _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
    }
#endif
    for (; i < end; i++) {
        float g = c->gate[i];
        c->hb[i] = (g / (1.0f + expf(-g))) * c->up[i];
    }
}

// Vectorized SwiGLU for pread path (single expert, no dispatch overhead)
static void moe_swiglu(float *hb, const float *gate, const float *up, int n) {
    int i = 0;
#ifdef __AVX2__
    for (; i + 7 < n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(hb + i, _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
    }
#elif defined(__ARM_NEON)
    // No fast_silu for NEON — use scalar (expf is the bottleneck either way)
#endif
    for (; i < n; i++) {
        float g = gate[i];
        hb[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// Compiler barrier to prevent reordering of timing calls around dispatches
static inline double moe_time_ms(void) {
    double t = bn_platform_time_ms();
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" ::: "memory");
#endif
    return t;
}

// Full MoE FFN block
void bn_moe_forward(BnModel *m, BnLayerWeights *lw, int l) {
#if defined(__EMSCRIPTEN__)
    (void)l;
#endif
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;
    BnMoEState *ms = m->moe_state;
    int dim = c->dim;
    int moe_hidden = c->moe_intermediate_size;
    int K = c->n_experts_active;
    double t0;

    // 1. RMSNorm input
    t0 = moe_time_ms();
    moe_rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);
    ms->stats.norm_time_ms += moe_time_ms() - t0;

    // 2. Route: select top-K experts (SIMD + threaded)
    t0 = moe_time_ms();
    bn_moe_route(ms, s->xb, lw->router_weight, dim, c->n_experts, K, m->pool);
    ms->stats.route_time_ms += moe_time_ms() - t0;

    // 3. Zero output accumulator
    memset(ms->expert_out, 0, dim * sizeof(float));

    // 4. Expert FFN compute
    double t_compute = moe_time_ms();

    if (ms->io.mmap_base && K <= BN_MAX_MOE_K) {
        // --- Cross-expert batched dispatch (mmap path) ---
        int valid_k = 0;
        int valid_indices[BN_MAX_MOE_K];
        float valid_weights[BN_MAX_MOE_K];
        BnQWeight wgates[BN_MAX_MOE_K], wups[BN_MAX_MOE_K];

        // Prefetch gate+up pages for all K experts (madvise mode)
#if !defined(__EMSCRIPTEN__)
        if (ms->io.madvise_mode) {
            double ta = moe_time_ms();
            moe_madvise_experts(ms, &lw->expert_map, ms->expert_indices, K,
                                MADV_WILLNEED, 0x3 /* gate+up */);
            ms->stats.madvise_time_ms += moe_time_ms() - ta;
        }
#endif

        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            if (eidx < 0) continue;
            const void *gate_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 0);
            const void *up_data   = moe_load_expert_proj(ms, &lw->expert_map, eidx, 1);
            if (!gate_data || !up_data) {
                SH_LOG_ERROR("Failed to load expert gate/up projection");
                continue;
            }
            wgates[valid_k] = moe_make_qweight(gate_data, lw->expert_map.gate_type,
                                                lw->expert_map.gate_rows, lw->expert_map.gate_cols);
            wups[valid_k]   = moe_make_qweight(up_data, lw->expert_map.up_type,
                                                lw->expert_map.up_rows, lw->expert_map.up_cols);
            valid_indices[valid_k] = eidx;
            valid_weights[valid_k] = ms->expert_weights[k];
            valid_k++;
        }

        if (valid_k > 0) {
            // Gate+up batch
            t0 = moe_time_ms();
            BnMatvecTask gu_tasks[2 * BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                gu_tasks[2*k]     = (BnMatvecTask){ ms->expert_hb_batch[k],  &wgates[k] };
                gu_tasks[2*k + 1] = (BnMatvecTask){ ms->expert_hb2_batch[k], &wups[k]   };
            }
            bn_quant_matvec_batch(gu_tasks, 2 * valid_k, s->xb, s->x_q, m->pool);
            ms->stats.gate_up_time_ms += moe_time_ms() - t0;

            // Parallel SwiGLU
            t0 = moe_time_ms();
            BnSwiGLUCtx swiglu_ctxs[BN_MAX_MOE_K];
            BnTPTask swiglu_tasks[BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                swiglu_ctxs[k] = (BnSwiGLUCtx){
                    ms->expert_hb_batch[k],
                    ms->expert_hb_batch[k],
                    ms->expert_hb2_batch[k]
                };
                swiglu_tasks[k] = (BnTPTask){ moe_swiglu_range, &swiglu_ctxs[k], moe_hidden };
            }
            bn_tp_dispatch(m->pool, swiglu_tasks, valid_k);
            ms->stats.swiglu_time_ms += moe_time_ms() - t0;

            // Prefetch down projection pages (madvise mode)
#if !defined(__EMSCRIPTEN__)
            if (ms->io.madvise_mode) {
                double ta = moe_time_ms();
                moe_madvise_experts(ms, &lw->expert_map, valid_indices, valid_k,
                                    MADV_WILLNEED, 0x4 /* down */);
                ms->stats.madvise_time_ms += moe_time_ms() - ta;
            }
#endif

            // Down projections
            t0 = moe_time_ms();
            BnQWeight wdowns[BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                const void *down_data = moe_load_expert_proj(ms, &lw->expert_map, valid_indices[k], 2);
                if (!down_data) {
                    SH_LOG_ERROR("Failed to load expert down projection");
                    valid_weights[k] = 0.0f;
                    continue;
                }
                wdowns[k] = moe_make_qweight(down_data, lw->expert_map.down_type,
                                              lw->expert_map.down_rows, lw->expert_map.down_cols);
            }

            // Individual matvec for each expert's down projection.
            // Each expert has different input (hb after SwiGLU), so we can't
            // share x quantization. Individual dispatch uses SDOT automatically.
            for (int k = 0; k < valid_k; k++) {
                if (valid_weights[k] == 0.0f) continue;
                bn_quant_matvec(ms->expert_down_batch[k], &wdowns[k],
                                ms->expert_hb_batch[k], s->x_q, m->pool);
            }
            ms->stats.down_time_ms += moe_time_ms() - t0;

            // Weighted accumulation
            t0 = moe_time_ms();
            for (int k = 0; k < valid_k; k++) {
                float w = valid_weights[k];
                if (w == 0.0f) continue;
                for (int d = 0; d < dim; d++)
                    ms->expert_out[d] += w * ms->expert_down_batch[k][d];
            }
            ms->stats.accum_time_ms += moe_time_ms() - t0;

        }
    }
#if !defined(__EMSCRIPTEN__)
    else if (ms->io.fd >= 0 && !ms->io.mmap_base) {
        // --- Pipelined pread path with LRU cache ---
        BnMoEPrefetch *pf_gu = (BnMoEPrefetch *)ms->io.prefetch;
        BnMoEPrefetch *pf_dn = (BnMoEPrefetch *)ms->io.prefetch_down;
        BnMoECache *cache = (BnMoECache *)ms->io.cache;
        const BnMoEExpertMap *map = &lw->expert_map;

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

        // Helper: collect prefetch stats from a thread
        #define COLLECT_PF_STATS(pf) do { \
            pthread_mutex_lock(&(pf)->mtx); \
            ms->stats.io_time_ms += (pf)->io_time_ms; \
            ms->stats.io_bytes += (pf)->io_bytes; \
            ms->stats.io_count += (pf)->n_reqs; \
            (pf)->io_time_ms = 0; \
            (pf)->io_bytes = 0; \
            pthread_mutex_unlock(&(pf)->mtx); \
        } while(0)

        // --- Two-phase: separate cache hits from misses ---
        int n_hits = 0, n_misses = 0;
        int hit_indices[BN_MAX_MOE_K], miss_indices[BN_MAX_MOE_K];
        float hit_weights[BN_MAX_MOE_K], miss_weights[BN_MAX_MOE_K];
        const uint8_t *hit_ptrs[BN_MAX_MOE_K];  // cache slab pointers for hits

        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            if (eidx < 0) continue;

            if (cache) {
                const uint8_t *cached = moe_cache_lookup(cache, l, eidx);
                if (cached) {
                    hit_indices[n_hits] = eidx;
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
            moe_proj_info(map, meidx, 0, &miss_g_off, &miss_g_sz);
            moe_proj_info(map, meidx, 1, &miss_u_off, &miss_u_sz);
            moe_proj_info(map, meidx, 2, &miss_d_off, &miss_d_sz);

            miss_slot_ptr = cache ? moe_cache_insert(cache, l, meidx) : NULL;
            miss_g_dst = miss_slot_ptr ? miss_slot_ptr : ms->io.buf;
            miss_u_dst = miss_slot_ptr ? miss_slot_ptr + cache->gate_bytes : ms->io.buf2;
            miss_d_dst = miss_slot_ptr ? miss_slot_ptr + cache->gate_bytes + cache->up_bytes : ms->io.buf5;

            if (pf_gu) {
                moe_prefetch_start2(pf_gu, miss_g_dst, miss_g_sz, (off_t)miss_g_off,
                                           miss_u_dst, miss_u_sz, (off_t)miss_u_off);
            }
            if (pf_dn) {
                moe_prefetch_start1(pf_dn, miss_d_dst, miss_d_sz, (off_t)miss_d_off);
            }
            miss_io_started = 1;
        }

        // Phase 1: Batch gate+up for all cache hits
        if (n_hits > 0) {
            t0 = moe_time_ms();
            BnQWeight wgates[BN_MAX_MOE_K], wups[BN_MAX_MOE_K];
            BnMatvecTask gu_tasks[2 * BN_MAX_MOE_K];
            for (int h = 0; h < n_hits; h++) {
                const uint8_t *cp = hit_ptrs[h];
                wgates[h] = moe_make_qweight(cp, map->gate_type,
                                              map->gate_rows, map->gate_cols);
                wups[h]   = moe_make_qweight(cp + cache->gate_bytes, map->up_type,
                                              map->up_rows, map->up_cols);
                gu_tasks[2*h]     = (BnMatvecTask){ ms->expert_hb_batch[h],  &wgates[h] };
                gu_tasks[2*h + 1] = (BnMatvecTask){ ms->expert_hb2_batch[h], &wups[h]   };
            }
            bn_quant_matvec_batch(gu_tasks, 2 * n_hits, s->xb, s->x_q, m->pool);
            ms->stats.gate_up_time_ms += moe_time_ms() - t0;

            // Parallel SwiGLU for hits
            t0 = moe_time_ms();
            BnSwiGLUCtx swiglu_ctxs[BN_MAX_MOE_K];
            BnTPTask swiglu_tasks[BN_MAX_MOE_K];
            for (int h = 0; h < n_hits; h++) {
                swiglu_ctxs[h] = (BnSwiGLUCtx){
                    ms->expert_hb_batch[h],
                    ms->expert_hb_batch[h],
                    ms->expert_hb2_batch[h]
                };
                swiglu_tasks[h] = (BnTPTask){ moe_swiglu_range, &swiglu_ctxs[h], moe_hidden };
            }
            bn_tp_dispatch(m->pool, swiglu_tasks, n_hits);
            ms->stats.swiglu_time_ms += moe_time_ms() - t0;

            // Down projections for hits (data already in cache)
            t0 = moe_time_ms();
            for (int h = 0; h < n_hits; h++) {
                const uint8_t *dp = hit_ptrs[h] + cache->gate_bytes + cache->up_bytes;
                BnQWeight wdown = moe_make_qweight(dp, map->down_type,
                                                    map->down_rows, map->down_cols);
                bn_quant_matvec(ms->expert_down_batch[h], &wdown,
                                ms->expert_hb_batch[h], s->x_q, m->pool);
            }
            ms->stats.down_time_ms += moe_time_ms() - t0;

            // Weighted accumulation for hits
            t0 = moe_time_ms();
            for (int h = 0; h < n_hits; h++) {
                float w = hit_weights[h];
                for (int d = 0; d < dim; d++)
                    ms->expert_out[d] += w * ms->expert_down_batch[h][d];
            }
            ms->stats.accum_time_ms += moe_time_ms() - t0;
        }

        // Phase 2: Process cache misses with I/O overlap
        for (int mi = 0; mi < n_misses; mi++) {
            int eidx = miss_indices[mi];
            float weight = miss_weights[mi];
            const uint8_t *gate_ptr, *up_ptr, *down_ptr;

            if (mi == 0 && miss_io_started) {
                // First miss: I/O was started before phase 1
                if (pf_gu) {
                    double tw = moe_time_ms();
                    int ok = moe_prefetch_wait(pf_gu);
                    ms->stats.prefetch_wait_ms += moe_time_ms() - tw;
                    COLLECT_PF_STATS(pf_gu);
                    if (!ok) {
                        if (pread(ms->io.fd, miss_g_dst, miss_g_sz, (off_t)miss_g_off) != (ssize_t)miss_g_sz)
                            SH_LOG_ERROR("Fallback gate pread failed");
                        if (pread(ms->io.fd, miss_u_dst, miss_u_sz, (off_t)miss_u_off) != (ssize_t)miss_u_sz)
                            SH_LOG_ERROR("Fallback up pread failed");
                    }
                } else {
                    if (pread(ms->io.fd, miss_g_dst, miss_g_sz, (off_t)miss_g_off) != (ssize_t)miss_g_sz)
                        SH_LOG_ERROR("Sync gate pread failed");
                    if (pread(ms->io.fd, miss_u_dst, miss_u_sz, (off_t)miss_u_off) != (ssize_t)miss_u_sz)
                        SH_LOG_ERROR("Sync up pread failed");
                    if (!pf_dn)
                        if (pread(ms->io.fd, miss_d_dst, miss_d_sz, (off_t)miss_d_off) != (ssize_t)miss_d_sz)
                            SH_LOG_ERROR("Sync down pread failed");
                }
                gate_ptr = miss_g_dst;
                up_ptr   = miss_u_dst;
                down_ptr = miss_d_dst;
            } else {
                // Subsequent misses: load gate+up+down
                size_t g_off, g_sz, u_off, u_sz, d_off, d_sz;
                moe_proj_info(map, eidx, 0, &g_off, &g_sz);
                moe_proj_info(map, eidx, 1, &u_off, &u_sz);
                moe_proj_info(map, eidx, 2, &d_off, &d_sz);

                uint8_t *slot = cache ? moe_cache_insert(cache, l, eidx) : NULL;
                uint8_t *g_dst = slot ? slot : ms->io.buf;
                uint8_t *u_dst = slot ? slot + cache->gate_bytes : ms->io.buf2;
                uint8_t *d_dst = slot ? slot + cache->gate_bytes + cache->up_bytes : ms->io.buf5;

                if (pf_gu) {
                    moe_prefetch_start2(pf_gu, g_dst, g_sz, (off_t)g_off,
                                               u_dst, u_sz, (off_t)u_off);
                }
                if (pf_dn) {
                    moe_prefetch_start1(pf_dn, d_dst, d_sz, (off_t)d_off);
                }

                if (pf_gu) {
                    double tw = moe_time_ms();
                    int ok = moe_prefetch_wait(pf_gu);
                    ms->stats.prefetch_wait_ms += moe_time_ms() - tw;
                    COLLECT_PF_STATS(pf_gu);
                    if (!ok) {
                        if (pread(ms->io.fd, g_dst, g_sz, (off_t)g_off) != (ssize_t)g_sz)
                            SH_LOG_ERROR("Fallback gate pread failed");
                        if (pread(ms->io.fd, u_dst, u_sz, (off_t)u_off) != (ssize_t)u_sz)
                            SH_LOG_ERROR("Fallback up pread failed");
                    }
                } else {
                    if (pread(ms->io.fd, g_dst, g_sz, (off_t)g_off) != (ssize_t)g_sz)
                        SH_LOG_ERROR("Sync gate pread failed");
                    if (pread(ms->io.fd, u_dst, u_sz, (off_t)u_off) != (ssize_t)u_sz)
                        SH_LOG_ERROR("Sync up pread failed");
                    if (pread(ms->io.fd, d_dst, d_sz, (off_t)d_off) != (ssize_t)d_sz)
                        SH_LOG_ERROR("Sync down pread failed");
                }

                gate_ptr = g_dst;
                up_ptr   = u_dst;
                down_ptr = d_dst;
            }

            // Gate+up matvec (down I/O may still be in flight)
            t0 = moe_time_ms();
            {
                BnQWeight wgate = moe_make_qweight(gate_ptr, map->gate_type,
                                                    map->gate_rows, map->gate_cols);
                BnQWeight wup = moe_make_qweight(up_ptr, map->up_type,
                                                  map->up_rows, map->up_cols);
                BnMatvecTask gu[2] = {
                    { ms->expert_hb,  &wgate },
                    { ms->expert_hb2, &wup   },
                };
                bn_quant_matvec_batch(gu, 2, s->xb, s->x_q, m->pool);
            }
            ms->stats.gate_up_time_ms += moe_time_ms() - t0;

            // SwiGLU
            t0 = moe_time_ms();
            moe_swiglu(ms->expert_hb, ms->expert_hb, ms->expert_hb2, moe_hidden);
            ms->stats.swiglu_time_ms += moe_time_ms() - t0;

            // Wait for down I/O
            t0 = moe_time_ms();
            if (pf_dn) {
                double tw = moe_time_ms();
                int ok = moe_prefetch_wait(pf_dn);
                ms->stats.prefetch_wait_ms += moe_time_ms() - tw;
                COLLECT_PF_STATS(pf_dn);
                if (!ok) {
                    size_t d_off, d_sz;
                    moe_proj_info(map, eidx, 2, &d_off, &d_sz);
                    if (pread(ms->io.fd, (void *)(uintptr_t)down_ptr, d_sz, (off_t)d_off) != (ssize_t)d_sz)
                        SH_LOG_ERROR("Fallback down pread failed");
                }
            }

            // Down matvec
            {
                BnQWeight wdown = moe_make_qweight(down_ptr, map->down_type,
                                                    map->down_rows, map->down_cols);
                bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, m->pool);
            }
            ms->stats.down_time_ms += moe_time_ms() - t0;

            // Weighted accumulation
            t0 = moe_time_ms();
            for (int d = 0; d < dim; d++)
                ms->expert_out[d] += weight * s->xb2[d];
            ms->stats.accum_time_ms += moe_time_ms() - t0;
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

            t0 = moe_time_ms();
            const void *gate_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 0);
            const void *up_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 1);
            if (!gate_data || !up_data) {
                SH_LOG_ERROR("Failed to load expert gate/up projection");
                continue;
            }
            BnQWeight wgate = moe_make_qweight(gate_data, lw->expert_map.gate_type,
                                                lw->expert_map.gate_rows, lw->expert_map.gate_cols);
            BnQWeight wup = moe_make_qweight(up_data, lw->expert_map.up_type,
                                              lw->expert_map.up_rows, lw->expert_map.up_cols);
            BnMatvecTask gu[2] = {
                { ms->expert_hb,  &wgate },
                { ms->expert_hb2, &wup   },
            };
            bn_quant_matvec_batch(gu, 2, s->xb, s->x_q, m->pool);
            ms->stats.gate_up_time_ms += moe_time_ms() - t0;

            // SwiGLU activation
            t0 = moe_time_ms();
            moe_swiglu(ms->expert_hb, ms->expert_hb, ms->expert_hb2, moe_hidden);
            ms->stats.swiglu_time_ms += moe_time_ms() - t0;

            // Down projection
            t0 = moe_time_ms();
            const void *down_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 2);
            if (!down_data) {
                SH_LOG_ERROR("Failed to load expert down projection");
                continue;
            }
            BnQWeight wdown = moe_make_qweight(down_data, lw->expert_map.down_type,
                                                lw->expert_map.down_rows, lw->expert_map.down_cols);
            bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, m->pool);
            ms->stats.down_time_ms += moe_time_ms() - t0;

            // Weighted accumulation
            t0 = moe_time_ms();
            for (int d = 0; d < dim; d++)
                ms->expert_out[d] += weight * s->xb2[d];
            ms->stats.accum_time_ms += moe_time_ms() - t0;
        }
    }

    // 5. Shared expert (if present, always resident)
    t0 = moe_time_ms();
    if (c->has_shared_expert && lw->shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;

        bn_quant_matvec(s->hb, &lw->shared_gate, s->xb, s->x_q, m->pool);
        bn_quant_matvec(s->hb2, &lw->shared_up, s->xb, s->x_q, m->pool);
        moe_swiglu(s->hb, s->hb, s->hb2, shared_hidden);
        bn_quant_matvec(s->xb2, &lw->shared_down, s->hb, s->x_q, m->pool);

        // Apply shared expert sigmoid gate if present (Qwen3.5 MoE):
        // gate = sigmoid(dot(input, gate_weight)) — scalar per token
        if (lw->shared_expert_gate) {
            float gate_dot = 0.0f;
            for (int d = 0; d < dim; d++)
                gate_dot += s->xb[d] * lw->shared_expert_gate[d];
            float gate = 1.0f / (1.0f + expf(-gate_dot));
            for (int d = 0; d < dim; d++)
                ms->expert_out[d] += gate * s->xb2[d];
        } else {
            for (int d = 0; d < dim; d++)
                ms->expert_out[d] += s->xb2[d];
        }
    }
    ms->stats.shared_time_ms += moe_time_ms() - t0;

    ms->stats.compute_time_ms += moe_time_ms() - t_compute;

    // 6. Copy result to xb for residual add by caller
    memcpy(s->xb, ms->expert_out, dim * sizeof(float));

    // 7. Residual add
    for (int d = 0; d < dim; d++)
        s->x[d] += s->xb[d];
}

// --- Batch MoE FFN for prefill ---
// Route all n_tokens, group by expert, batch matmul per expert.
int bn_moe_forward_batch(BnModel *m, BnLayerWeights *lw, int l,
                          float *act, float *Xb, int n_tokens) {
    (void)l;  // reserved for pread cache keying in future
    BnConfig *c = &m->config;
    BnMoEState *ms = m->moe_state;
    int dim = c->dim;
    int moe_hidden = c->moe_intermediate_size;
    int K = c->n_experts_active;
    int n_experts = c->n_experts;
    const BnMoEExpertMap *map = &lw->expert_map;

    // 1. Batch RMSNorm
    for (int t = 0; t < n_tokens; t++)
        moe_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                    lw->ffn_norm, dim, c->norm_eps);

    // 2. Batch routing: route each token individually (reuse existing router)
    // Allocate routing results: [n_tokens][K] indices and weights
    int *all_indices = (int *)malloc((size_t)n_tokens * K * sizeof(int));
    float *all_weights = (float *)malloc((size_t)n_tokens * K * sizeof(float));
    if (!all_indices || !all_weights) {
        free(all_indices); free(all_weights);
        return -1;
    }

    for (int t = 0; t < n_tokens; t++) {
        bn_moe_route(ms, Xb + (size_t)t * dim, lw->router_weight,
                     dim, n_experts, K, m->pool);
        memcpy(all_indices + t * K, ms->expert_indices, K * sizeof(int));
        memcpy(all_weights + t * K, ms->expert_weights, K * sizeof(float));
    }

    // 3. Build token-expert grouping (two-pass)
    // Pass 1: count tokens per expert
    int *expert_counts = (int *)calloc(n_experts, sizeof(int));
    int *expert_offsets = (int *)malloc(n_experts * sizeof(int));
    if (!expert_counts || !expert_offsets) {
        free(all_indices); free(all_weights); free(expert_counts); free(expert_offsets);
        return -1;
    }

    for (int t = 0; t < n_tokens; t++)
        for (int k = 0; k < K; k++) {
            int eidx = all_indices[t * K + k];
            if (eidx >= 0) expert_counts[eidx]++;
        }

    // Prefix sum for offsets
    int total_assignments = 0;
    for (int e = 0; e < n_experts; e++) {
        expert_offsets[e] = total_assignments;
        total_assignments += expert_counts[e];
    }

    // Pass 2: fill flat arrays
    int *group_token_ids = (int *)malloc(total_assignments * sizeof(int));
    float *group_weights = (float *)malloc(total_assignments * sizeof(float));
    int *fill_pos = (int *)calloc(n_experts, sizeof(int));  // current fill position per expert
    if (!group_token_ids || !group_weights || !fill_pos) {
        free(all_indices); free(all_weights); free(expert_counts);
        free(expert_offsets); free(group_token_ids); free(group_weights); free(fill_pos);
        return -1;
    }

    for (int t = 0; t < n_tokens; t++)
        for (int k = 0; k < K; k++) {
            int eidx = all_indices[t * K + k];
            if (eidx < 0) continue;
            int pos = expert_offsets[eidx] + fill_pos[eidx];
            group_token_ids[pos] = t;
            group_weights[pos] = all_weights[t * K + k];
            fill_pos[eidx]++;
        }

    // 4. Allocate batch compute buffers
    // T_max = max tokens assigned to any single expert
    int T_max = 0;
    for (int e = 0; e < n_experts; e++)
        if (expert_counts[e] > T_max) T_max = expert_counts[e];

    float *gather_buf = (float *)malloc((size_t)T_max * dim * sizeof(float));
    float *gate_buf   = (float *)malloc((size_t)T_max * moe_hidden * sizeof(float));
    float *up_buf     = (float *)malloc((size_t)T_max * moe_hidden * sizeof(float));
    float *down_buf   = (float *)malloc((size_t)T_max * dim * sizeof(float));
    float *moe_out    = (float *)calloc((size_t)n_tokens * dim, sizeof(float));
    int8_t *x_q_scratch = (int8_t *)malloc((size_t)T_max *
        (dim > moe_hidden ? dim : moe_hidden));
    if (!gather_buf || !gate_buf || !up_buf || !down_buf || !moe_out || !x_q_scratch) {
        free(gather_buf); free(gate_buf); free(up_buf); free(down_buf);
        free(moe_out); free(x_q_scratch);
        free(all_indices); free(all_weights); free(expert_counts);
        free(expert_offsets); free(group_token_ids); free(group_weights); free(fill_pos);
        return -1;
    }

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
        const void *gate_data = moe_load_expert_proj(ms, map, e, 0);
        const void *up_data   = moe_load_expert_proj(ms, map, e, 1);
        const void *down_data = moe_load_expert_proj(ms, map, e, 2);
        if (!gate_data || !up_data || !down_data) continue;

        BnQWeight wgate = moe_make_qweight(gate_data, map->gate_type,
                                            map->gate_rows, map->gate_cols);
        BnQWeight wup   = moe_make_qweight(up_data, map->up_type,
                                            map->up_rows, map->up_cols);
        BnQWeight wdown = moe_make_qweight(down_data, map->down_type,
                                            map->down_rows, map->down_cols);

        // Gate + Up matmul (T tokens at once)
        if (T == 1) {
            // Single token: use matvec (less overhead)
            BnMatvecTask gu[2] = {
                { gate_buf, &wgate },
                { up_buf,   &wup   },
            };
            bn_quant_matvec_batch(gu, 2, gather_buf, x_q_scratch, m->pool);
        } else {
            bn_quant_matmul(gate_buf, &wgate, gather_buf, T, x_q_scratch, m->pool);
            bn_quant_matmul(up_buf, &wup, gather_buf, T, x_q_scratch, m->pool);
        }

        // SwiGLU activation across T * moe_hidden
        for (int i = 0; i < T * moe_hidden; i++) {
            float g = gate_buf[i];
            gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
        }

        // Down matmul
        if (T == 1) {
            bn_quant_matvec(down_buf, &wdown, gate_buf, x_q_scratch, m->pool);
        } else {
            bn_quant_matmul(down_buf, &wdown, gate_buf, T, x_q_scratch, m->pool);
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
    if (c->has_shared_expert && lw->shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;
        float *sh_gate = gate_buf;  // reuse (T_max >= 1, shared_hidden <= moe_hidden usually)
        float *sh_up = up_buf;
        float *sh_down = down_buf;

        // Need buffers sized for n_tokens * shared_hidden
        // If shared_hidden > moe_hidden * T_max, we'd need bigger buffers.
        // For safety, allocate if needed.
        int need_sh = (size_t)n_tokens * shared_hidden > (size_t)T_max * moe_hidden;
        float *sh_g = need_sh ? (float *)malloc((size_t)n_tokens * shared_hidden * sizeof(float)) : sh_gate;
        float *sh_u = need_sh ? (float *)malloc((size_t)n_tokens * shared_hidden * sizeof(float)) : sh_up;
        float *sh_d = need_sh ? (float *)malloc((size_t)n_tokens * dim * sizeof(float)) : sh_down;

        if (sh_g && sh_u && sh_d) {
            bn_quant_matmul(sh_g, &lw->shared_gate, Xb, n_tokens, x_q_scratch, m->pool);
            bn_quant_matmul(sh_u, &lw->shared_up, Xb, n_tokens, x_q_scratch, m->pool);

            for (int i = 0; i < n_tokens * shared_hidden; i++) {
                float g = sh_g[i];
                sh_g[i] = (g / (1.0f + expf(-g))) * sh_u[i];
            }

            bn_quant_matmul(sh_d, &lw->shared_down, sh_g, n_tokens, x_q_scratch, m->pool);

            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    moe_out[(size_t)t * dim + d] += sh_d[(size_t)t * dim + d];
        }

        if (need_sh) { free(sh_g); free(sh_u); free(sh_d); }
    }

    // 7. Residual add: act += moe_out
    for (int t = 0; t < n_tokens; t++)
        for (int d = 0; d < dim; d++)
            act[(size_t)t * dim + d] += moe_out[(size_t)t * dim + d];

    // Cleanup
    free(gather_buf); free(gate_buf); free(up_buf); free(down_buf);
    free(moe_out); free(x_q_scratch);
    free(all_indices); free(all_weights); free(expert_counts);
    free(expert_offsets); free(group_token_ids); free(group_weights); free(fill_pos);

    return 0;
}

void bn_moe_print_stats(const BnMoEState *ms, int n_tokens) {
    if (!ms || n_tokens <= 0) return;

    double io_per_tok = (double)ms->stats.io_bytes / (1024.0 * 1024.0) / n_tokens;

    char iot_s[32], bw_s[32], rss_s[32];
    char norm_s[32], rt_s[32], gu_s[32], sw_s[32], dn_s[32], ac_s[32], sh_s[32], ct_s[32];

    snprintf(iot_s, sizeof(iot_s), "%.1f", io_per_tok);

    if (ms->stats.io_time_ms > 0.1)
        snprintf(bw_s, sizeof(bw_s), "%.0f",
                 (double)ms->stats.io_bytes / (1024.0 * 1024.0) / (ms->stats.io_time_ms / 1000.0));
    else
        snprintf(bw_s, sizeof(bw_s), "mmap");

    snprintf(norm_s, sizeof(norm_s), "%.1f", ms->stats.norm_time_ms);
    snprintf(rt_s, sizeof(rt_s), "%.1f", ms->stats.route_time_ms);
    snprintf(gu_s, sizeof(gu_s), "%.1f", ms->stats.gate_up_time_ms);
    snprintf(sw_s, sizeof(sw_s), "%.1f", ms->stats.swiglu_time_ms);
    snprintf(dn_s, sizeof(dn_s), "%.1f", ms->stats.down_time_ms);
    snprintf(ac_s, sizeof(ac_s), "%.1f", ms->stats.accum_time_ms);
    snprintf(sh_s, sizeof(sh_s), "%.1f", ms->stats.shared_time_ms);
    snprintf(ct_s, sizeof(ct_s), "%.1f", ms->stats.compute_time_ms);

    char pw_s[32], ma_s[32];
    snprintf(pw_s, sizeof(pw_s), "%.1f", ms->stats.prefetch_wait_ms);
    snprintf(ma_s, sizeof(ma_s), "%.1f", ms->stats.madvise_time_ms);

    size_t rss = bn_platform_rss_bytes();
    snprintf(rss_s, sizeof(rss_s), "%.2f", (double)rss / (1024.0 * 1024.0 * 1024.0));

    SH_LOG_INFO("MoE stats",
                "MB/tok", iot_s,
                "stream_MB/s", bw_s,
                "rss_GB", rss_s);
    SH_LOG_INFO("MoE breakdown (ms)",
                "norm", norm_s,
                "route", rt_s,
                "gate+up", gu_s,
                "swiglu", sw_s,
                "down", dn_s,
                "accum", ac_s,
                "shared", sh_s,
                "pf_wait", pw_s,
                "madvise", ma_s,
                "total", ct_s);

    bn_moe_cache_print_stats(ms);
}

void bn_moe_reset_stats(BnMoEState *ms) {
    if (!ms) return;
    ms->stats.io_bytes = 0;
    ms->stats.io_time_ms = 0;
    ms->stats.route_time_ms = 0;
    ms->stats.compute_time_ms = 0;
    ms->stats.gate_up_time_ms = 0;
    ms->stats.swiglu_time_ms = 0;
    ms->stats.down_time_ms = 0;
    ms->stats.accum_time_ms = 0;
    ms->stats.shared_time_ms = 0;
    ms->stats.norm_time_ms = 0;
    ms->stats.io_count = 0;
    ms->stats.prefetch_wait_ms = 0;
    ms->stats.madvise_time_ms = 0;
    ms->stats.cache_hits = 0;
    ms->stats.cache_misses = 0;
}

void bn_moe_prefetch_create(BnMoEState *ms) {
    if (!ms || ms->io.prefetch) return;
#if !defined(__EMSCRIPTEN__)
    if (ms->io.fd >= 0 && !ms->io.mmap_base) {
        ms->io.prefetch = moe_prefetch_init(ms->io.fd);
        ms->io.prefetch_down = moe_prefetch_init(ms->io.fd);
        if (ms->io.prefetch && ms->io.prefetch_down) {
            SH_LOG_INFO("MoE I/O prefetch threads", "status", "2 created (gate+up, down)");
        } else {
            // Clean up partial init — free whichever succeeded
            if (ms->io.prefetch) { moe_prefetch_free((BnMoEPrefetch *)ms->io.prefetch); ms->io.prefetch = NULL; }
            if (ms->io.prefetch_down) { moe_prefetch_free((BnMoEPrefetch *)ms->io.prefetch_down); ms->io.prefetch_down = NULL; }
            SH_LOG_WARN("MoE I/O prefetch threads failed to create");
        }
    }
#endif
}

void bn_moe_prefetch_destroy(BnMoEState *ms) {
    if (!ms) return;
#if !defined(__EMSCRIPTEN__)
    if (ms->io.prefetch) {
        moe_prefetch_free((BnMoEPrefetch *)ms->io.prefetch);
        ms->io.prefetch = NULL;
    }
    if (ms->io.prefetch_down) {
        moe_prefetch_free((BnMoEPrefetch *)ms->io.prefetch_down);
        ms->io.prefetch_down = NULL;
    }
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
