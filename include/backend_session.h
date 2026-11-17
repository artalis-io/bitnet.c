#ifndef BN_BACKEND_SESSION_H
#define BN_BACKEND_SESSION_H

#include <stddef.h>
#include <stdint.h>
#ifndef BN_GPU_BACKEND_DECLARED
#define BN_GPU_BACKEND_DECLARED
typedef struct BnGPUBackend BnGPUBackend;
#endif
typedef struct BnGPUResidentView BnGPUResidentView;
typedef struct BnBackendRowCache BnBackendRowCache;

/* Embed in a session resource owner, sharing its canonical resident request.
 * The request owns device buffers and pinned slots; this handle owns metadata
 * only. GPU backend/request outlive all calls. Destroy the request before
 * freeing this metadata at owner teardown. Access and shared-cache use are
 * caller-serialized. No model position or KV state is advanced here. */
typedef struct { void *state; } BnBackendRowStage;
typedef struct {
    int format,cols,max_rows,banks; /* banks1..4, two upload slots per bank. */
    size_t host_budget,pinned_budget,device_budget;
} BnBackendRowStageOptions;
typedef struct {
    size_t host_bytes,pinned_bytes,device_bytes,slot_bytes;
    size_t packed_count,scale_count,out_count;
} BnBackendRowStageWorkspace;
typedef struct {
    BnBackendRowStageWorkspace workspace;
    uint64_t batches,rows,upload_bytes,busy_returns,epochs[4];
    int failed;
} BnBackendRowStageInfo;
/* Queue nonempty I32 metadata through the first existing pinned slot. Caller
 * serializes with row submissions. Destination is exact, same-request I32,
 * disjoint from all row-bank buffers. Count*4 must fit slot_bytes. BUSY queues
 * nothing; no allocation/fence/cache I/O or row epoch change. Upload bytes and
 * busy returns include metadata. Backend/lease failure poisons the stage;
 * owner must drain/destroy the request before freeing its metadata. */
int bn_backend_session_row_stage_upload_i32(BnBackendRowStage *stage,
    const int32_t *values,size_t count,const BnGPUResidentView *destination);
/* Pure host query. Budgets cover this controller's metadata and payloads;
 * borrowed cache/index, allocator/context/event overhead and other request
 * resources are separate. Output unchanged on invalid geometry. */
int bn_backend_session_row_stage_plan(const BnBackendRowStageOptions *options,
    BnBackendRowStageWorkspace *out);
/* Setup only, one initialization per request owner. Owns upload slots
 * [0,2*banks) exclusively; an existing reservation must match exactly.
 * Matching live setup is idempotent. No handle publication on failure, but
 * failure after touching GPU resources requires discarding the unpublished
 * request: partial device/pinned allocations remain request-owned. */
int bn_backend_session_row_stage_init(BnBackendRowStage *stage,BnGPUBackend *gpu,
    void *request,const BnBackendRowStageOptions *options);
void bn_backend_session_row_stage_free(BnBackendRowStage *stage);
int bn_backend_session_row_stage_info(const BnBackendRowStage *stage,
    BnBackendRowStageInfo *out);
/* Queues two pinned uploads and one decode. No allocation or GPU completion
 * fence; storage misses may block in the source reader.
 * Returns0 submitted, BN_GPU_UPLOAD_BUSY for retry without cache/I/O work,
 * -1 failure. Zero count is a no-op, permitting NULL ids/out/epoch/cache.
 * Nonzero success publishes a BF16 view and epoch only after enqueueing all
 * work. Preflight, I/O and scale-domain rejection leave prior epochs current;
 * backend failures poison all results and require request teardown.
 * Queue all consumers of a bank BEFORE resubmitting that bank on the same
 * request stream. Returned views expire at its next successful submission.
 * This is submission, not GPU completion; check final completion separately. */
int bn_backend_session_row_stage_submit(BnBackendRowStage *stage,
    BnBackendRowCache *cache,unsigned bank,const uint64_t *ids,size_t count,
    BnGPUResidentView *out,uint64_t *epoch);
/* Same staging transaction, but registry-validates finite F16 after the
 * mandatory BF16 rounding before either DMA is queued. Domain rejection
 * leaves prior views valid; already completed host cache fills may remain. */
int bn_backend_session_row_stage_submit_f16(BnBackendRowStage *stage,
    BnBackendRowCache *cache,unsigned bank,const uint64_t *ids,size_t count,
    BnGPUResidentView *out,uint64_t *epoch);
int bn_backend_session_row_stage_current(const BnBackendRowStage *stage,
    unsigned bank,uint64_t epoch,const BnGPUResidentView *view);

typedef struct BnBackendSession BnBackendSession;
typedef struct BnGPUValueGraph BnGPUValueGraph;

BnBackendSession *bn_backend_session_create(void);
void bn_backend_session_free(BnBackendSession *backend);
/* One backend-owned request resource container (activations, recurrent state,
 * and workspace). owner is a non-NULL identity token, never dereferenced here.
 * Attach transfers ownership only on success and never replaces live storage.
 * release must finish outstanding work before freeing its container; it must
 * carry its own cleanup context rather than borrow model/weight storage.
 * Session reset/free release the container. Access is serialized by the caller.
 */
int bn_backend_session_attach_gpu_resources(BnBackendSession *backend,
    const void *owner, void *resources, void (*release)(void *));
void *bn_backend_session_gpu_resources(const BnBackendSession *backend,
    const void *owner);
void bn_backend_session_release_gpu_resources(BnBackendSession *backend);
int bn_backend_session_has_gpu_resources(const BnBackendSession *backend);
int bn_backend_session_has_execution_resources(const BnBackendSession *backend);
/* Host execution metadata is released AFTER GPU resources finish outstanding
 * work. Its callback may free host memory only; it must not dereference model,
 * backend, or expired device storage. Attach transfers ownership on success
 * only. GPU-resource release/reset/free also detach this metadata exactly once.
 * Caller serializes access; callbacks must not attach new resources.
 */
int bn_backend_session_attach_execution_resources(BnBackendSession *backend,
    const void *owner, void *resources, void (*release)(void *));
void *bn_backend_session_execution_resources(const BnBackendSession *backend,
    const void *owner);

void *bn_backend_session_gpu_graph(const BnBackendSession *backend);
void *bn_backend_session_ensure_gpu_graph(BnBackendSession *backend, int cap_ops);
void *bn_backend_session_ensure_gpu_command_buffer(BnBackendSession *backend,
                                                   int cap_ops,
                                                   int *out_cap);
int bn_backend_session_gpu_cached_op_count(const BnBackendSession *backend);
int bn_backend_session_gpu_cached_has_logits(const BnBackendSession *backend);
void bn_backend_session_set_gpu_cached_op_count(BnBackendSession *backend,
                                                int n_ops,
                                                int has_logits);
void bn_backend_session_clear_gpu_cached_ops(BnBackendSession *backend);
/* A successful GPU decode advanced recurrent buffers beyond host mirrors. */
int bn_backend_session_gpu_decode_ssm_pending(const BnBackendSession *backend);
void bn_backend_session_set_gpu_decode_ssm_pending(BnBackendSession *backend,
                                                   int pending);
BnGPUValueGraph *bn_backend_session_gpu_value_graph(BnBackendSession *backend);
int bn_backend_session_ensure_gpu_lowering_values(BnBackendSession *backend,
                                                  int elem_size,
                                                  int cap_values,
                                                  void **out_values,
                                                  int *out_cap);
void bn_backend_session_release_gpu_graph(BnBackendSession *backend);

#endif // BN_BACKEND_SESSION_H
