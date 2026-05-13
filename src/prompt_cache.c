#include "prompt_cache.h"
#include "model.h"
#include "session.h"
#include "turboquant.h"
#include <string.h>

#if !defined(__EMSCRIPTEN__)
#include <pthread.h>
#endif

// --- FNV-1a hash over int array ---

static uint64_t fnv1a_tokens(const int *tokens, int n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t *p = (const uint8_t *)tokens;
    size_t bytes = (size_t)n * sizeof(int);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// --- Locking helpers ---

#if !defined(__EMSCRIPTEN__)
static void cache_lock(BnPromptCache *c) {
    if (c->mtx) pthread_mutex_lock((pthread_mutex_t *)c->mtx);
}
static void cache_unlock(BnPromptCache *c) {
    if (c->mtx) pthread_mutex_unlock((pthread_mutex_t *)c->mtx);
}
#else
#define cache_lock(c) (void)(c)
#define cache_unlock(c) (void)(c)
#endif

// --- Helpers ---

static BnAllocator resolve_alloc(BnAllocator *a) {
    if (a) return *a;
    return bn_allocator_default();
}

// Compute n_attn_layers from config
static int config_n_attn(const BnConfig *c) {
    return (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval : c->n_layers;
}

// Free a single entry's buffers
static void entry_free(BnPromptCacheEntry *e, BnAllocator *a) {
    if (e->tokens) {
        bn_free(a, e->tokens, (size_t)e->n_tokens * sizeof(int));
        e->tokens = NULL;
    }
    if (e->key_cache) {
        bn_free(a, e->key_cache, e->key_cache_bytes);
        e->key_cache = NULL;
    }
    if (e->value_cache) {
        bn_free(a, e->value_cache, e->val_cache_bytes);
        e->value_cache = NULL;
    }
    e->n_tokens = 0;
    e->key_cache_bytes = 0;
    e->val_cache_bytes = 0;
}

// Evict the oldest entry (index 0, shift left)
static void evict_oldest(BnPromptCache *c) {
    if (c->n_entries == 0) return;
    BnPromptCacheEntry *e = &c->entries[0];
    c->used_bytes -= e->key_cache_bytes + e->val_cache_bytes + (size_t)e->n_tokens * sizeof(int);
    entry_free(e, &c->alloc);
    // Shift remaining entries down
    for (int i = 1; i < c->n_entries; i++)
        c->entries[i - 1] = c->entries[i];
    c->n_entries--;
    memset(&c->entries[c->n_entries], 0, sizeof(BnPromptCacheEntry));
}

// --- Public API ---

BnPromptCache *bn_prompt_cache_create(size_t max_bytes, BnAllocator *alloc) {
    BnAllocator a = resolve_alloc(alloc);
    BnPromptCache *c = (BnPromptCache *)bn_malloc(&a, sizeof(BnPromptCache));
    if (!c) return NULL;
    memset(c, 0, sizeof(BnPromptCache));
    c->max_bytes = max_bytes;
    c->alloc = a;

#if !defined(__EMSCRIPTEN__)
    pthread_mutex_t *mtx = (pthread_mutex_t *)bn_malloc(&a, sizeof(pthread_mutex_t));
    if (mtx) {
        pthread_mutex_init(mtx, NULL);
        c->mtx = mtx;
    }
#endif

    return c;
}

void bn_prompt_cache_free(BnPromptCache *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->n_entries; i++)
        entry_free(&cache->entries[i], &cache->alloc);
#if !defined(__EMSCRIPTEN__)
    if (cache->mtx) {
        pthread_mutex_destroy((pthread_mutex_t *)cache->mtx);
        bn_free(&cache->alloc, cache->mtx, sizeof(pthread_mutex_t));
    }
#endif
    BnAllocator a = cache->alloc;
    bn_free(&a, cache, sizeof(BnPromptCache));
}

int bn_prompt_cache_store(BnPromptCache *cache, const BnModel *model,
                          const BnSession *session,
                          const int *tokens, int n_tokens) {
    if (!cache || !model || !session || !tokens || n_tokens <= 0) return -2;
    const BnConfig *cfg = &model->config;

    // Reject hybrid models (SSM state can't be cheaply snapshotted)
    if (cfg->full_attn_interval > 0) return -1;

    // Validate: n_tokens must not exceed session pos or seq_len
    if (n_tokens > session->pos || n_tokens > cfg->seq_len) return -2;

    int n_attn = config_n_attn(cfg);
    int kv_dim = cfg->kv_dim;
    int tq_bits = cfg->kv_tq_bits;

    // Compute per-cache byte sizes
    size_t key_bytes, val_bytes;
    if (tq_bits > 0 && model->tq_state) {
        // TQ path: packed bytes per head, n_kv_heads heads per position
        int kb = bn_tq_key_bytes(model->tq_state);
        int vb = bn_tq_value_bytes(model->tq_state);
        key_bytes = (size_t)n_attn * (size_t)n_tokens * (size_t)cfg->n_kv_heads * (size_t)kb;
        val_bytes = (size_t)n_attn * (size_t)n_tokens * (size_t)cfg->n_kv_heads * (size_t)vb;
    } else {
        // FP32/FP16 path
        size_t elem_size = cfg->kv_f16 ? sizeof(uint16_t) : sizeof(float);
        key_bytes = (size_t)n_attn * (size_t)n_tokens * (size_t)kv_dim * elem_size;
        val_bytes = key_bytes;
    }
    size_t tok_bytes = (size_t)n_tokens * sizeof(int);
    size_t entry_total = key_bytes + val_bytes + tok_bytes;

    cache_lock(cache);

    // Evict until we have space (if max_bytes is set)
    if (cache->max_bytes > 0) {
        while (cache->used_bytes + entry_total > cache->max_bytes && cache->n_entries > 0)
            evict_oldest(cache);
    }

    // Evict if at max entries
    while (cache->n_entries >= BN_PROMPT_CACHE_MAX_ENTRIES)
        evict_oldest(cache);

    // Allocate entry buffers
    int *tok_copy = (int *)bn_malloc(&cache->alloc, tok_bytes);
    void *kc = bn_malloc(&cache->alloc, key_bytes);
    void *vc = bn_malloc(&cache->alloc, val_bytes);
    if (!tok_copy || !kc || !vc) {
        if (tok_copy) bn_free(&cache->alloc, tok_copy, tok_bytes);
        if (kc) bn_free(&cache->alloc, kc, key_bytes);
        if (vc) bn_free(&cache->alloc, vc, val_bytes);
        cache_unlock(cache);
        return -2;
    }

    // Copy token sequence
    memcpy(tok_copy, tokens, tok_bytes);

    if (tq_bits > 0 && model->tq_state) {
        // TQ path: copy from session's TQ packed caches
        int kb = bn_tq_key_bytes(model->tq_state);
        int vb = bn_tq_value_bytes(model->tq_state);
        size_t pos_stride_k = (size_t)cfg->n_kv_heads * kb;
        size_t pos_stride_v = (size_t)cfg->n_kv_heads * vb;
        size_t layer_stride_src_k = (size_t)cfg->seq_len * pos_stride_k;
        size_t layer_stride_src_v = (size_t)cfg->seq_len * pos_stride_v;
        size_t layer_stride_dst_k = (size_t)n_tokens * pos_stride_k;
        size_t layer_stride_dst_v = (size_t)n_tokens * pos_stride_v;

        const uint8_t *src_k = session->state.key_cache_tq;
        const uint8_t *src_v = session->state.value_cache_tq;
        uint8_t *dst_k = (uint8_t *)kc;
        uint8_t *dst_v = (uint8_t *)vc;

        for (int a = 0; a < n_attn; a++) {
            memcpy(dst_k + a * layer_stride_dst_k, src_k + a * layer_stride_src_k, layer_stride_dst_k);
            memcpy(dst_v + a * layer_stride_dst_v, src_v + a * layer_stride_src_v, layer_stride_dst_v);
        }
    } else {
        // FP32/FP16 path: strided copy from session's [n_attn * seq_len * kv_dim]
        size_t elem_size = cfg->kv_f16 ? sizeof(uint16_t) : sizeof(float);
        size_t layer_stride_src = (size_t)cfg->seq_len * kv_dim * elem_size;
        size_t layer_stride_dst = (size_t)n_tokens * kv_dim * elem_size;
        const uint8_t *src_k = (const uint8_t *)session->state.key_cache;
        const uint8_t *src_v = (const uint8_t *)session->state.value_cache;
        uint8_t *dst_k = (uint8_t *)kc;
        uint8_t *dst_v = (uint8_t *)vc;

        for (int a = 0; a < n_attn; a++) {
            memcpy(dst_k + a * layer_stride_dst, src_k + a * layer_stride_src, layer_stride_dst);
            memcpy(dst_v + a * layer_stride_dst, src_v + a * layer_stride_src, layer_stride_dst);
        }
    }

    // Insert entry
    BnPromptCacheEntry *e = &cache->entries[cache->n_entries];
    e->tokens = tok_copy;
    e->n_tokens = n_tokens;
    e->hash = fnv1a_tokens(tokens, n_tokens);
    e->key_cache = kc;
    e->value_cache = vc;
    e->key_cache_bytes = key_bytes;
    e->val_cache_bytes = val_bytes;
    e->kv_f16 = cfg->kv_f16;
    e->kv_tq_bits = tq_bits;
    e->n_attn_layers = n_attn;
    e->kv_dim = kv_dim;
    cache->n_entries++;
    cache->used_bytes += entry_total;

    cache_unlock(cache);
    return 0;
}

int bn_prompt_cache_restore(BnPromptCache *cache, const BnModel *model,
                            BnSession *session,
                            const int *tokens, int n_tokens) {
    if (!cache || !model || !session || !tokens || n_tokens <= 0) return 0;
    const BnConfig *cfg = &model->config;

    // Reject hybrid models
    if (cfg->full_attn_interval > 0) return 0;

    int n_attn = config_n_attn(cfg);
    int kv_dim = cfg->kv_dim;
    int tq_bits = cfg->kv_tq_bits;

    cache_lock(cache);

    // Find longest prefix match
    int best_idx = -1;
    int best_len = 0;

    for (int i = 0; i < cache->n_entries; i++) {
        BnPromptCacheEntry *e = &cache->entries[i];

        // Config validation: format must match
        if (e->kv_tq_bits != tq_bits) continue;
        if (tq_bits == 0 && e->kv_f16 != cfg->kv_f16) continue;
        if (e->n_attn_layers != n_attn || e->kv_dim != kv_dim) continue;

        // Quick reject: entry can't match more than its own length or query length
        int max_match = e->n_tokens < n_tokens ? e->n_tokens : n_tokens;
        if (max_match <= best_len) continue;

        // Can't exceed seq_len
        if (max_match > cfg->seq_len) max_match = cfg->seq_len;
        if (max_match <= best_len) continue;

        // Compute common prefix length
        int match = 0;
        for (int j = 0; j < max_match; j++) {
            if (e->tokens[j] != tokens[j]) break;
            match++;
        }

        if (match > best_len) {
            best_len = match;
            best_idx = i;
        }
    }

    if (best_idx < 0 || best_len == 0) {
        cache_unlock(cache);
        return 0;
    }

    // Copy KV prefix from cache entry into session
    BnPromptCacheEntry *e = &cache->entries[best_idx];

    if (tq_bits > 0 && model->tq_state) {
        // TQ path: copy packed bytes into session's TQ caches
        int kb = bn_tq_key_bytes(model->tq_state);
        int vb = bn_tq_value_bytes(model->tq_state);
        size_t pos_stride_k = (size_t)cfg->n_kv_heads * kb;
        size_t pos_stride_v = (size_t)cfg->n_kv_heads * vb;
        size_t layer_stride_src_k = (size_t)e->n_tokens * pos_stride_k;
        size_t layer_stride_src_v = (size_t)e->n_tokens * pos_stride_v;
        size_t layer_stride_dst_k = (size_t)cfg->seq_len * pos_stride_k;
        size_t layer_stride_dst_v = (size_t)cfg->seq_len * pos_stride_v;
        size_t copy_per_layer_k = (size_t)best_len * pos_stride_k;
        size_t copy_per_layer_v = (size_t)best_len * pos_stride_v;

        uint8_t *dst_k = session->state.key_cache_tq;
        uint8_t *dst_v = session->state.value_cache_tq;
        const uint8_t *src_k = (const uint8_t *)e->key_cache;
        const uint8_t *src_v = (const uint8_t *)e->value_cache;

        for (int a = 0; a < n_attn; a++) {
            memcpy(dst_k + a * layer_stride_dst_k, src_k + a * layer_stride_src_k, copy_per_layer_k);
            memcpy(dst_v + a * layer_stride_dst_v, src_v + a * layer_stride_src_v, copy_per_layer_v);
        }
    } else {
        // FP32/FP16 path
        size_t elem_size = cfg->kv_f16 ? sizeof(uint16_t) : sizeof(float);
        size_t layer_stride_src = (size_t)e->n_tokens * kv_dim * elem_size;
        size_t layer_stride_dst = (size_t)cfg->seq_len * kv_dim * elem_size;
        size_t copy_per_layer = (size_t)best_len * kv_dim * elem_size;

        uint8_t *dst_k = (uint8_t *)session->state.key_cache;
        uint8_t *dst_v = (uint8_t *)session->state.value_cache;
        const uint8_t *src_k = (const uint8_t *)e->key_cache;
        const uint8_t *src_v = (const uint8_t *)e->value_cache;

        for (int a = 0; a < n_attn; a++) {
            memcpy(dst_k + a * layer_stride_dst, src_k + a * layer_stride_src, copy_per_layer);
            memcpy(dst_v + a * layer_stride_dst, src_v + a * layer_stride_src, copy_per_layer);
        }
    }

    session->pos = best_len;

    cache_unlock(cache);
    return best_len;
}

void bn_prompt_cache_clear(BnPromptCache *cache) {
    if (!cache) return;
    cache_lock(cache);
    for (int i = 0; i < cache->n_entries; i++)
        entry_free(&cache->entries[i], &cache->alloc);
    cache->n_entries = 0;
    cache->used_bytes = 0;
    cache_unlock(cache);
}

int bn_prompt_cache_count(const BnPromptCache *cache) {
    if (!cache) return 0;
    cache_lock((BnPromptCache *)cache);
    int n = cache->n_entries;
    cache_unlock((BnPromptCache *)cache);
    return n;
}

size_t bn_prompt_cache_bytes(const BnPromptCache *cache) {
    if (!cache) return 0;
    cache_lock((BnPromptCache *)cache);
    size_t b = cache->used_bytes;
    cache_unlock((BnPromptCache *)cache);
    return b;
}
