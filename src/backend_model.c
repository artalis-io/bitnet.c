#include "backend_model.h"
#include "gpu_backend.h"
#include "quant.h"
#include <stdlib.h>

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
};

BnBackendModel *bn_backend_model_create(void) {
    return (BnBackendModel *)calloc(1, sizeof(BnBackendModel));
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
    gpu->buffer_destroy(gpu->ctx, handle);
    if (*n_seen < cap_seen)
        seen[(*n_seen)++] = handle;
    return 0;
}

void bn_backend_model_release_gpu(BnBackendModel *backend) {
    if (!backend) return;
    BnGPUBackend *gpu = backend->gpu;
    if (gpu && gpu->buffer_destroy) {
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
                gpu->buffer_destroy(gpu->ctx, handle);
        }
        for (int i = 0; i < backend->n_handles; i++) {
            void *handle = backend->handles[i].handle;
            if (seen)
                backend_destroy_once(gpu, seen, &n_seen, cap_seen, handle);
            else if (handle)
                gpu->buffer_destroy(gpu->ctx, handle);
        }
        free(seen);
    }
    bn_backend_model_clear_gpu(backend);
}

void bn_backend_model_free(BnBackendModel *backend) {
    if (!backend) return;
    bn_backend_model_release_gpu(backend);
    free(backend->handles);
    free(backend->qweights);
    free(backend);
}

BnGPUBackend *bn_backend_model_gpu(const BnBackendModel *backend) {
    if (!backend || backend->gpu_disabled) return NULL;
    return backend->gpu;
}

BnGPUBackend *bn_backend_model_raw_gpu(const BnBackendModel *backend) {
    return backend ? backend->gpu : NULL;
}

void bn_backend_model_bind_gpu(BnBackendModel *backend, BnGPUBackend *gpu) {
    if (!backend) return;
    backend->gpu = gpu;
    backend->gpu_disabled = 0;
}

void bn_backend_model_clear_gpu(BnBackendModel *backend) {
    if (!backend) return;
    backend->gpu = NULL;
    backend->gpu_disabled = 0;
    backend->n_handles = 0;
    backend->n_qweights = 0;
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
