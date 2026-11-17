#ifndef BN_BACKEND_MODEL_H
#define BN_BACKEND_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifndef BN_GPU_BACKEND_DECLARED
#define BN_GPU_BACKEND_DECLARED
typedef struct BnGPUBackend BnGPUBackend;
#endif

typedef struct BnBackendModel BnBackendModel;
typedef struct BnQWeight BnQWeight;
typedef struct BnPreparedWeight BnPreparedWeight;
typedef struct BnTrellisRowSource BnTrellisRowSource;
typedef struct BnTrellisRowLayout BnTrellisRowLayout;

/* Backend-owned mutable raw-row cache, separate from immutable CPU weights.
 * Source/index must remain unchanged and outlive the cache. Caller serializes
 * access to each handle, including statistics/free. No borrowed row leases:
 * gather copies into caller-owned packed and scale planes. */
typedef struct BnBackendRowCache { void *state; } BnBackendRowCache;
typedef struct {
    size_t budget_bytes, allocated_bytes, capacity, resident_rows, max_gather_rows;
    uint64_t hits, misses, staging_hits, evictions, io_failures, source_bytes_requested;
    double source_read_ms;
} BnBackendRowCacheStats;
/* Budget includes state, hash/LRU metadata, cached rows, maximum gather staging
 * and one miss scratch row; excludes borrowed source/index, caller output,
 * allocator overhead and OS page cache. Repeated IDs within a gather share
 * staged data even when the persistent LRU cannot hold them all. One
 * allocation at init, none in gather.
 * Init rejection leaves a zeroed handle unchanged. Never copy a live handle. */
int bn_backend_model_row_cache_init(BnBackendRowCache *cache,
    const BnTrellisRowSource *source, size_t budget_bytes, size_t max_gather_rows);
void bn_backend_model_row_cache_free(BnBackendRowCache *cache);
int bn_backend_model_row_cache_stats(const BnBackendRowCache *cache,
    BnBackendRowCacheStats *out);
const BnTrellisRowLayout *bn_backend_model_row_cache_layout(const BnBackendRowCache *cache);
/* Exact output byte counts, disjoint ID/output/cache buffers. Zero count is
 * permitted. Preflights all IDs before cache/I/O changes. On I/O failure output
 * is unchanged; earlier successful fills may remain cached. A failed miss
 * never evicts a valid row. Stats count logical row requests, not disk sectors.
 * Source bytes count requested raw bundles, including failed attempts. */
int bn_backend_model_row_cache_gather(BnBackendRowCache *cache,
    const uint64_t *ids, size_t count, void *weights, size_t weight_bytes,
    void *scales, size_t scale_bytes);

typedef enum {
    BN_BACKEND_HANDLE_OUTPUT_NORM = 1,
    BN_BACKEND_HANDLE_TIED_EMBEDDING = 2,
    BN_BACKEND_HANDLE_ATTN_NORM = 3,
    BN_BACKEND_HANDLE_FFN_NORM = 4,
    BN_BACKEND_HANDLE_Q_BIAS = 5,
    BN_BACKEND_HANDLE_K_BIAS = 6,
    BN_BACKEND_HANDLE_V_BIAS = 7,
    BN_BACKEND_HANDLE_QKV_STACKED = 8,
    BN_BACKEND_HANDLE_GATEUP_STACKED = 9,
    BN_BACKEND_HANDLE_SSM_AB_STACKED = 10,
    BN_BACKEND_HANDLE_SSM_QKVZ_STACKED = 11,
    BN_BACKEND_HANDLE_Q_NORM = 12,
    BN_BACKEND_HANDLE_K_NORM = 13,
    BN_BACKEND_HANDLE_ATTN_SUB_NORM = 14,
    BN_BACKEND_HANDLE_FFN_SUB_NORM = 15,
    BN_BACKEND_HANDLE_SSM_CONV1D = 16,
    BN_BACKEND_HANDLE_SSM_DT_BIAS = 17,
    BN_BACKEND_HANDLE_SSM_A_LOG = 18,
    BN_BACKEND_HANDLE_SSM_NORM = 19,
    BN_BACKEND_HANDLE_QK_STACKED = 20,
    BN_BACKEND_HANDLE_WV_PREFILL = 21,
    BN_BACKEND_HANDLE_WO_PREFILL = 22,
    BN_BACKEND_HANDLE_FFN_DOWN_PREFILL = 23,
    BN_BACKEND_HANDLE_SHARED_EXPERT_GATE = 24,
    BN_BACKEND_HANDLE_MOE_ROUTER_DIFF = 25,
    BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED = 26,
    BN_BACKEND_HANDLE_MOE_ROUTER = 27,
    BN_BACKEND_HANDLE_MOE_GATE_ALL = 28,
    BN_BACKEND_HANDLE_MOE_UP_ALL = 29,
    BN_BACKEND_HANDLE_MOE_DOWN_ALL = 30,
    BN_BACKEND_HANDLE_PER_LAYER_POST_NORM = 31,
    BN_BACKEND_HANDLE_ATTN_POST_NORM = 32,
    BN_BACKEND_HANDLE_FFN_POST_NORM = 33,
    BN_BACKEND_HANDLE_V_UNIT_NORM = 34,
    BN_BACKEND_HANDLE_MOE_ROUTER_SCALE = 35,
    BN_BACKEND_HANDLE_FFN_POST_NORM_1 = 36,
    BN_BACKEND_HANDLE_FFN_POST_NORM_2 = 37,
    BN_BACKEND_HANDLE_MOE_EXPERT_DOWN_SCALE = 38,
    BN_BACKEND_HANDLE_FFN_GATE_REFERENCE = 39,
    BN_BACKEND_HANDLE_FFN_UP_REFERENCE = 40,
    BN_BACKEND_HANDLE_HC_OUTPUT_NORM = 41,
    BN_BACKEND_HANDLE_HC_ATTN_NORM = 42,
    BN_BACKEND_HANDLE_HC_FFN_NORM = 43,
    BN_BACKEND_HANDLE_PLE_NORM_KEY = 44,
    BN_BACKEND_HANDLE_PLE_NORM_QUERY = 45,
    BN_BACKEND_HANDLE_PLE_NORM_CONV = 46,
    BN_BACKEND_HANDLE_PLE_CONV1D = 47,
    BN_BACKEND_HANDLE_PREFILL_QK_PAIR = 48,
} BnBackendHandleRole;

typedef struct {
    void *router;
    void *gate_all;
    void *up_all;
    void *down_all;
    void *norm;
    void *sub_norm;
    void *router_scale;
    int routed_valid;
    int norm_resid_valid;
} BnBackendModelMoEPrefillRoutedResources;

typedef struct {
    void *gate_all;
    void *up_all;
    void *down_all;
    int valid;
} BnBackendModelMoEPrefillResidentResources;

BnBackendModel *bn_backend_model_create(void);
/* One owned CPU resource container, independent of GPU layouts/prepared-cache
 * leases. Release sessions before replacing/freeing model resources. Cleanup
 * carries its own context; no replacement of a live container. */
int bn_backend_model_attach_cpu_resources(BnBackendModel *backend,const void *owner,
    void *resources,void (*release)(void *));
void *bn_backend_model_cpu_resources(const BnBackendModel *backend,const void *owner);
void bn_backend_model_free(BnBackendModel *backend);
BnGPUBackend *bn_backend_model_gpu(const BnBackendModel *backend);
BnGPUBackend *bn_backend_model_gpu_for_cpu_operations(
    const BnBackendModel *backend);
void bn_backend_model_bind_gpu(BnBackendModel *backend, BnGPUBackend *gpu);
void bn_backend_model_release_gpu(BnBackendModel *backend);
/* One owned backend layout container, including precision-specific weights.
 * owner is a non-NULL identity token. Attach transfers ownership on success
 * only and never replaces a live container. Cleanup runs before registered
 * GPU handles are destroyed; borrowed handles must not be freed by cleanup.
 * The callback must carry its cleanup context and finish outstanding work.
 * Callers serialize access and release sessions before model resources.
 */
int bn_backend_model_attach_gpu_resources(BnBackendModel *backend,
    const void *owner, void *resources, void (*release)(void *));
void *bn_backend_model_gpu_resources(const BnBackendModel *backend,
    const void *owner);
/* Publish a staged layout and GPU binding atomically. Both owners must have
 * no registered handles/qweights; destination must have no GPU state and
 * source must own a layout. CPU prepared caches stay with their owners.
 * Failure changes neither owner; success empties source GPU state. */
int bn_backend_model_move_gpu_resources(BnBackendModel *destination,
    BnBackendModel *source);
void bn_backend_model_set_cpu_prepared_cache_budget(BnBackendModel *backend,
                                                     size_t budget_bytes);
const BnPreparedWeight *bn_backend_model_acquire_cpu_prepared(
    BnBackendModel *backend, const BnQWeight *weight);
/* Pin an existing layout without inserting or evicting entries on a miss. */
const BnPreparedWeight *bn_backend_model_acquire_cached_cpu_prepared(
    BnBackendModel *backend, const BnQWeight *weight);
void bn_backend_model_release_cpu_prepared(BnBackendModel *backend,
                                            const BnPreparedWeight *prepared);
void bn_backend_model_set_gpu_disabled(BnBackendModel *backend, int disabled);
int bn_backend_model_register_handle(BnBackendModel *backend,
                                     int layer,
                                     BnBackendHandleRole role,
                                     void *handle);
void *bn_backend_model_handle(const BnBackendModel *backend,
                              int layer,
                              BnBackendHandleRole role);
BnBackendModelMoEPrefillRoutedResources
bn_backend_model_moe_prefill_routed_resources(
    const BnBackendModel *backend,
    int layer);
BnBackendModelMoEPrefillResidentResources
bn_backend_model_moe_prefill_resident_resources(
    const BnBackendModel *backend,
    int layer);
int bn_backend_model_register_qweight(BnBackendModel *backend,
                                      const BnQWeight *weight,
                                      void *gpu_buf);
void *bn_backend_model_qweight_buf(const BnBackendModel *backend,
                                   const BnQWeight *weight);
int bn_backend_model_register_prepared_qweight(BnBackendModel *backend,
                                               const BnQWeight *weight,
                                               const BnPreparedWeight *prepared);
const BnPreparedWeight *bn_backend_model_prepared_qweight(
    const BnBackendModel *backend,
    const BnQWeight *weight);

#endif // BN_BACKEND_MODEL_H
