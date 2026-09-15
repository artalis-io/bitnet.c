#include "backend_model.h"
#include "gpu_backend.h"
#include "quant.h"
#include "sh_arena.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct BnBackendCPUPreparedEntry {
    const void *data;
    int type;
    int rows;
    int cols;
    float scale;
    size_t bytes;
    int refs;
    SHArena *arena;
    BnPreparedWeight prepared;
    struct BnBackendCPUPreparedEntry *prev;
    struct BnBackendCPUPreparedEntry *next;
    struct BnBackendCPUPreparedEntry *hash_next;
} BnBackendCPUPreparedEntry;

typedef struct {
    const BnQWeight *weight;
    void *gpu_buf;
    BnPreparedWeight prepared;
    int has_prepared;
} BnBackendQWeightBuf;

typedef struct {
    int layer;
    BnBackendHandleRole role;
    void *handle;
} BnBackendHandle;

struct BnBackendModel {
    BnGPUBackend *gpu;
    int gpu_disabled;
    BnBackendHandle *handles;
    int n_handles;
    int cap_handles;
    BnBackendQWeightBuf *qweights;
    int n_qweights;
    int cap_qweights;
    size_t cpu_prepared_budget;
    size_t cpu_prepared_bytes;
    BnBackendCPUPreparedEntry *cpu_prepared_head;
    BnBackendCPUPreparedEntry *cpu_prepared_tail;
    BnBackendCPUPreparedEntry **cpu_prepared_buckets;
    size_t cpu_prepared_n_buckets;
    atomic_flag cpu_prepared_lock;
};

BnBackendModel *bn_backend_model_create(void) {
    BnBackendModel *backend = (BnBackendModel *)calloc(1, sizeof(*backend));
    if (backend)
        atomic_flag_clear(&backend->cpu_prepared_lock);
    return backend;
}

static void backend_cpu_prepared_lock(BnBackendModel *backend) {
    while (atomic_flag_test_and_set_explicit(&backend->cpu_prepared_lock,
                                              memory_order_acquire)) {
    }
}

static void backend_cpu_prepared_unlock(BnBackendModel *backend) {
    atomic_flag_clear_explicit(&backend->cpu_prepared_lock,
                               memory_order_release);
}

static size_t backend_cpu_prepared_hash(const BnBackendModel *backend,
                                        const void *data) {
    uintptr_t key = (uintptr_t)data;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    return (size_t)key & (backend->cpu_prepared_n_buckets - 1);
}

static void backend_cpu_prepared_hash_remove(
    BnBackendModel *backend, BnBackendCPUPreparedEntry *entry) {
    if (!backend->cpu_prepared_buckets) return;
    size_t bucket = backend_cpu_prepared_hash(backend, entry->data);
    BnBackendCPUPreparedEntry **link =
        &backend->cpu_prepared_buckets[bucket];
    while (*link && *link != entry)
        link = &(*link)->hash_next;
    if (*link)
        *link = entry->hash_next;
}

static void backend_cpu_prepared_unlink(BnBackendModel *backend,
                                        BnBackendCPUPreparedEntry *entry) {
    if (entry->prev) entry->prev->next = entry->next;
    else backend->cpu_prepared_head = entry->next;
    if (entry->next) entry->next->prev = entry->prev;
    else backend->cpu_prepared_tail = entry->prev;
}

static void backend_cpu_prepared_push_front(
    BnBackendModel *backend, BnBackendCPUPreparedEntry *entry) {
    entry->prev = NULL;
    entry->next = backend->cpu_prepared_head;
    if (entry->next) entry->next->prev = entry;
    else backend->cpu_prepared_tail = entry;
    backend->cpu_prepared_head = entry;
}

static void backend_cpu_prepared_free_entry(
    BnBackendModel *backend, BnBackendCPUPreparedEntry *entry) {
    backend_cpu_prepared_hash_remove(backend, entry);
    backend_cpu_prepared_unlink(backend, entry);
    backend->cpu_prepared_bytes -= entry->bytes;
    sh_arena_free(entry->arena);
    free(entry);
}

void bn_backend_model_set_cpu_prepared_cache_budget(BnBackendModel *backend,
                                                     size_t budget_bytes) {
    if (!backend) return;
    backend_cpu_prepared_lock(backend);
    backend->cpu_prepared_budget = budget_bytes;
    if (budget_bytes > 0 && !backend->cpu_prepared_buckets) {
        backend->cpu_prepared_n_buckets = 65536;
        backend->cpu_prepared_buckets = calloc(
            backend->cpu_prepared_n_buckets,
            sizeof(*backend->cpu_prepared_buckets));
        if (!backend->cpu_prepared_buckets) {
            backend->cpu_prepared_n_buckets = 0;
            backend->cpu_prepared_budget = 0;
        }
    }
    BnBackendCPUPreparedEntry *entry = backend->cpu_prepared_tail;
    while (backend->cpu_prepared_bytes > budget_bytes && entry) {
        BnBackendCPUPreparedEntry *prev = entry->prev;
        if (entry->refs == 0)
            backend_cpu_prepared_free_entry(backend, entry);
        entry = prev;
    }
    backend_cpu_prepared_unlock(backend);
}

static const BnPreparedWeight *backend_model_acquire_cpu_prepared(
    BnBackendModel *backend, const BnQWeight *weight, int prepare_missing) {
    if (!backend || !weight) return NULL;
    size_t bytes = bn_quant_prepared_qweight_size(weight, NULL);
    if (bytes == 0) return NULL;

    backend_cpu_prepared_lock(backend);
    if (backend->cpu_prepared_budget == 0 ||
        bytes > backend->cpu_prepared_budget) {
        backend_cpu_prepared_unlock(backend);
        return NULL;
    }
    size_t bucket = backend_cpu_prepared_hash(backend, weight->data);
    for (BnBackendCPUPreparedEntry *entry =
             backend->cpu_prepared_buckets[bucket];
         entry; entry = entry->hash_next) {
        if (entry->data == weight->data && entry->type == weight->type &&
            entry->rows == weight->rows && entry->cols == weight->cols &&
            entry->scale == weight->scale) {
            entry->refs++;
            backend_cpu_prepared_unlink(backend, entry);
            backend_cpu_prepared_push_front(backend, entry);
            backend_cpu_prepared_unlock(backend);
            return &entry->prepared;
        }
    }

    if (!prepare_missing) {
        backend_cpu_prepared_unlock(backend);
        return NULL;
    }

    while (backend->cpu_prepared_bytes + bytes >
           backend->cpu_prepared_budget) {
        BnBackendCPUPreparedEntry *victim = backend->cpu_prepared_tail;
        while (victim && victim->refs != 0)
            victim = victim->prev;
        if (!victim) {
            backend_cpu_prepared_unlock(backend);
            return NULL;
        }
        backend_cpu_prepared_free_entry(backend, victim);
    }

    BnBackendCPUPreparedEntry *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        backend_cpu_prepared_unlock(backend);
        return NULL;
    }
    entry->arena = sh_arena_create(bytes);
    if (!entry->arena ||
        bn_quant_prepare_qweight(&entry->prepared, weight, entry->arena) != 0) {
        sh_arena_free(entry->arena);
        free(entry);
        backend_cpu_prepared_unlock(backend);
        return NULL;
    }
    entry->data = weight->data;
    entry->type = weight->type;
    entry->rows = weight->rows;
    entry->cols = weight->cols;
    entry->scale = weight->scale;
    entry->bytes = bytes;
    entry->refs = 1;
    entry->hash_next = backend->cpu_prepared_buckets[bucket];
    backend->cpu_prepared_buckets[bucket] = entry;
    backend_cpu_prepared_push_front(backend, entry);
    backend->cpu_prepared_bytes += bytes;
    backend_cpu_prepared_unlock(backend);
    return &entry->prepared;
}

const BnPreparedWeight *bn_backend_model_acquire_cpu_prepared(
    BnBackendModel *backend, const BnQWeight *weight) {
    return backend_model_acquire_cpu_prepared(backend, weight, 1);
}

const BnPreparedWeight *bn_backend_model_acquire_cached_cpu_prepared(
    BnBackendModel *backend, const BnQWeight *weight) {
    return backend_model_acquire_cpu_prepared(backend, weight, 0);
}

void bn_backend_model_release_cpu_prepared(BnBackendModel *backend,
                                            const BnPreparedWeight *prepared) {
    if (!backend || !prepared) return;
    backend_cpu_prepared_lock(backend);
    BnBackendCPUPreparedEntry *entry =
        (BnBackendCPUPreparedEntry *)((char *)prepared -
            offsetof(BnBackendCPUPreparedEntry, prepared));
    if (entry->refs > 0) entry->refs--;
    backend_cpu_prepared_unlock(backend);
}

static int backend_handle_seen(void **seen, int n_seen, void *handle) {
    for (int i = 0; i < n_seen; i++) {
        if (seen[i] == handle) return 1;
    }
    return 0;
}

static int backend_destroy_once(BnGPUBackend *gpu, void **seen, int *n_seen,
                                int cap_seen, void *handle) {
    if (!handle) return 0;
    if (backend_handle_seen(seen, *n_seen, handle)) return 0;
    bn_gpu_backend_destroy_buffer(gpu, handle);
    if (*n_seen < cap_seen)
        seen[(*n_seen)++] = handle;
    return 0;
}

static void backend_model_clear_gpu(BnBackendModel *backend) {
    backend->gpu = NULL;
    backend->gpu_disabled = 0;
    backend->n_handles = 0;
    backend->n_qweights = 0;
}

void bn_backend_model_release_gpu(BnBackendModel *backend) {
    if (!backend) return;
    BnGPUBackend *gpu = backend->gpu;
    if (bn_gpu_backend_can_destroy_buffer(gpu)) {
        int cap_seen = backend->n_qweights + backend->n_handles;
        void **seen = NULL;
        if (cap_seen > 0)
            seen = (void **)calloc((size_t)cap_seen, sizeof(void *));
        int n_seen = 0;
        for (int i = 0; i < backend->n_qweights; i++) {
            void *handle = backend->qweights[i].gpu_buf;
            if (seen)
                backend_destroy_once(gpu, seen, &n_seen, cap_seen, handle);
            else if (handle)
                bn_gpu_backend_destroy_buffer(gpu, handle);
        }
        for (int i = 0; i < backend->n_handles; i++) {
            void *handle = backend->handles[i].handle;
            if (seen)
                backend_destroy_once(gpu, seen, &n_seen, cap_seen, handle);
            else if (handle)
                bn_gpu_backend_destroy_buffer(gpu, handle);
        }
        free(seen);
    }
    backend_model_clear_gpu(backend);
}

void bn_backend_model_free(BnBackendModel *backend) {
    if (!backend) return;
    bn_backend_model_release_gpu(backend);
    while (backend->cpu_prepared_tail)
        backend_cpu_prepared_free_entry(backend,
                                        backend->cpu_prepared_tail);
    free(backend->cpu_prepared_buckets);
    free(backend->handles);
    free(backend->qweights);
    free(backend);
}

BnGPUBackend *bn_backend_model_gpu(const BnBackendModel *backend) {
    if (!backend || backend->gpu_disabled) return NULL;
    return backend->gpu;
}

BnGPUBackend *bn_backend_model_gpu_for_cpu_operations(
    const BnBackendModel *backend) {
    return backend ? backend->gpu : NULL;
}

void bn_backend_model_bind_gpu(BnBackendModel *backend, BnGPUBackend *gpu) {
    if (!backend) return;
    backend->gpu = gpu;
    backend->gpu_disabled = 0;
}

void bn_backend_model_set_gpu_disabled(BnBackendModel *backend, int disabled) {
    if (!backend) return;
    backend->gpu_disabled = disabled ? 1 : 0;
}

int bn_backend_model_register_handle(BnBackendModel *backend,
                                     int layer,
                                     BnBackendHandleRole role,
                                     void *handle) {
    if (!backend || role == 0) return -1;
    for (int i = 0; i < backend->n_handles; i++) {
        if (backend->handles[i].layer == layer &&
            backend->handles[i].role == role) {
            backend->handles[i].handle = handle;
            return 0;
        }
    }
    if (backend->n_handles == backend->cap_handles) {
        int new_cap = backend->cap_handles ? backend->cap_handles * 2 : 64;
        BnBackendHandle *new_items = (BnBackendHandle *)realloc(
            backend->handles, (size_t)new_cap * sizeof(BnBackendHandle));
        if (!new_items) return -1;
        backend->handles = new_items;
        backend->cap_handles = new_cap;
    }
    backend->handles[backend->n_handles++] =
        (BnBackendHandle){ layer, role, handle };
    return 0;
}

void *bn_backend_model_handle(const BnBackendModel *backend,
                              int layer,
                              BnBackendHandleRole role) {
    if (!backend || role == 0) return NULL;
    for (int i = 0; i < backend->n_handles; i++) {
        if (backend->handles[i].layer == layer &&
            backend->handles[i].role == role)
            return backend->handles[i].handle;
    }
    return NULL;
}

BnBackendModelMoEPrefillRoutedResources
bn_backend_model_moe_prefill_routed_resources(
    const BnBackendModel *backend,
    int layer) {
    BnBackendModelMoEPrefillRoutedResources resources = {0};
    if (!backend)
        return resources;

    resources.router =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER);
    resources.gate_all =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL);
    resources.up_all =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL);
    resources.down_all =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
    resources.norm =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_FFN_NORM);
    resources.router_scale =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER_SCALE);
    resources.sub_norm =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_FFN_SUB_NORM);
    resources.routed_valid = resources.router && resources.gate_all &&
                             resources.up_all && resources.down_all;
    resources.norm_resid_valid = resources.routed_valid && resources.norm;
    return resources;
}

BnBackendModelMoEPrefillResidentResources
bn_backend_model_moe_prefill_resident_resources(
    const BnBackendModel *backend,
    int layer) {
    BnBackendModelMoEPrefillResidentResources resources = {0};
    if (!backend)
        return resources;

    resources.gate_all =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL);
    resources.up_all =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL);
    resources.down_all =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
    resources.valid = resources.gate_all && resources.up_all &&
                      resources.down_all;
    return resources;
}

int bn_backend_model_register_qweight(BnBackendModel *backend,
                                      const BnQWeight *weight,
                                      void *gpu_buf) {
    if (!backend || !weight) return -1;
    for (int i = 0; i < backend->n_qweights; i++) {
        if (backend->qweights[i].weight == weight) {
            backend->qweights[i].gpu_buf = gpu_buf;
            return 0;
        }
    }
    if (backend->n_qweights == backend->cap_qweights) {
        int new_cap = backend->cap_qweights ? backend->cap_qweights * 2 : 64;
        BnBackendQWeightBuf *new_items = (BnBackendQWeightBuf *)realloc(
            backend->qweights, (size_t)new_cap * sizeof(BnBackendQWeightBuf));
        if (!new_items) return -1;
        backend->qweights = new_items;
        backend->cap_qweights = new_cap;
    }
    backend->qweights[backend->n_qweights++] =
        (BnBackendQWeightBuf){ weight, gpu_buf, { 0 }, 0 };
    return 0;
}

void *bn_backend_model_qweight_buf(const BnBackendModel *backend,
                                   const BnQWeight *weight) {
    if (!backend || !weight) return NULL;
    for (int i = 0; i < backend->n_qweights; i++) {
        if (backend->qweights[i].weight == weight)
            return backend->qweights[i].gpu_buf;
    }
    return NULL;
}

int bn_backend_model_register_prepared_qweight(BnBackendModel *backend,
                                               const BnQWeight *weight,
                                               const BnPreparedWeight *prepared) {
    if (!backend || !weight || !prepared) return -1;
    for (int i = 0; i < backend->n_qweights; i++) {
        if (backend->qweights[i].weight == weight) {
            backend->qweights[i].prepared = *prepared;
            backend->qweights[i].has_prepared = 1;
            return 0;
        }
    }
    if (backend->n_qweights == backend->cap_qweights) {
        int new_cap = backend->cap_qweights ? backend->cap_qweights * 2 : 64;
        BnBackendQWeightBuf *new_items = (BnBackendQWeightBuf *)realloc(
            backend->qweights, (size_t)new_cap * sizeof(BnBackendQWeightBuf));
        if (!new_items) return -1;
        backend->qweights = new_items;
        backend->cap_qweights = new_cap;
    }
    backend->qweights[backend->n_qweights++] =
        (BnBackendQWeightBuf){ weight, NULL, *prepared, 1 };
    return 0;
}

const BnPreparedWeight *bn_backend_model_prepared_qweight(
    const BnBackendModel *backend,
    const BnQWeight *weight) {
    if (!backend || !weight) return NULL;
    for (int i = 0; i < backend->n_qweights; i++) {
        if (backend->qweights[i].weight == weight)
            return backend->qweights[i].has_prepared ? &backend->qweights[i].prepared : NULL;
    }
    return NULL;
}
