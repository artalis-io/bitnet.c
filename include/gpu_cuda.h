#ifndef BN_GPU_CUDA_H
#define BN_GPU_CUDA_H

#ifdef BN_ENABLE_CUDA

#include "gpu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

BnGPUBackend *bn_gpu_cuda_create(void);
BnGPUBackend *bn_gpu_cuda_create_with_policy(
    const BnBackendRuntimePolicy *policy);
void bn_gpu_cuda_destroy(BnGPUBackend *gpu);

/* Backend-local inactive weight storage, separate from the expert cache budget.
 * Calls share the backend's serialized-executor requirement. No active weight
 * is released by trim. Counters are cumulative; bytes/entries are current. */
typedef struct {
    size_t limit_bytes, limit_entries;
    size_t idle_bytes, idle_entries;
    size_t hits, misses, evictions, synchronizations, sync_failures;
} BnCudaWeightPoolStats;
int bn_gpu_cuda_weight_pool_stats(const BnGPUBackend *gpu,
                                  BnCudaWeightPoolStats *stats);
int bn_gpu_cuda_trim_weight_pool(BnGPUBackend *gpu);
/* Diagnostic packed [K,N] readback of a completed native F16 row layout.
 * Invalid handle/owner/capacity rejects before copy. Not an inference path. */
int bn_gpu_cuda_read_f16_rows(const BnGPUBackend *gpu, void *weight,
    uint16_t *out, size_t out_elems);
/* Diagnostic native row-major immutable scalar buffer readback. */
int bn_gpu_cuda_read_raw_rows(const BnGPUBackend *gpu,void *weight,
    void *out,size_t capacity_bytes);

/* Setup/diagnostic-only full derived arena readback; no execution-path use. */
int bn_gpu_cuda_read_router_weight(const BnGPUBackend *gpu,void *weight,
    void *out,size_t capacity);

#ifdef __cplusplus
}
#endif

#endif // BN_ENABLE_CUDA

#endif // BN_GPU_CUDA_H
