#ifndef BN_GPU_MOE_BRIDGE_H
#define BN_GPU_MOE_BRIDGE_H

#include "model.h"
#include "session.h"
#include "gpu_moe_cache.h"

typedef struct {
    void *gate;
    void *up;
    void *down;
    int use_gateup_split;
    int gateup_split_op_code;
} BnGPUMoEExpertBuffers;

/* Serialized expert placement for one immutable native model. The model and
 * backend must outlive the store. Payload budgets exclude metadata, activations
 * and backend context overhead; staging is a separate full expert bundle.
 * Caller owns zeroed storage. Do not copy stores or live leases. */
typedef struct { void *state; } BnGPUMoETrellisStore;
/* Copied at store creation, immutable thereafter. Adds RAM admission without
 * requiring GPU admission; GPU admit=1 retains its existing RAM behavior. Zero preserves both existing LRUs during
 * bypass scans; one admits/promotes RAM misses/hits even on GPU bypass misses.
 * GPU hits still require no host fetch. Enabling this requires a positive RAM payload budget.
 * No additional quota, asynchronous work or routing change is introduced. */
/* Page advice is also copied at construction. SYSTEM requires a positive RAM
 * budget and skips the hint; it does not force base pages or modify the OS. */
typedef struct {
    int ram_admit_misses;
    BnMoECachePagePolicy ram_pages;
} BnGPUMoETrellisPolicy;
typedef struct {
    void *owner;
    BnGPUMoELease cached;
    BnGPUMoEReservation temporary;
    BnGPUMoEExpertBuffers buffers;
} BnGPUMoETrellisLease;
typedef struct {
    BnMoEStats host;
    BnGPUMoECacheStats decode, temporary;
    size_t staging_payload, live_leases, upload_count, upload_bytes;
    /* Reusable temporary allocation includes both idle and leased storage;
     * it shares temporary.capacity_payload, never adds another quota. Cache
     * charged_payload reports active reservations and overlaps this value.
     * Backends without refill support report zero here; their active storage
     * is covered by temporary.charged_payload and released immediately. */
    size_t temporary_allocated_payload, temporary_peak_payload;
    size_t buffer_create_count, buffer_reuse_count;
    /* Pinned staging overlaps staging_payload, not an extra host bundle. */
    size_t pinned_staging_payload, upload_batches;
    double upload_time_ms; /* blocking allocation/upload boundary, not kernel time */
    BnMoECachePagePolicy ram_pages; /* requested advice, not observed page size */
} BnGPUMoETrellisStats;
int bn_gpu_moe_trellis_store_init(BnGPUMoETrellisStore *store,
    const BnModel *model, BnGPUBackend *gpu, size_t ram_budget,
    size_t decode_budget, size_t temporary_budget, size_t staging_budget);
/* Explicit pinned admission. pinned_budget=0 keeps ordinary staging. Positive
 * budget requires backend batch-upload support and one full expert bundle;
 * total staging_budget still applies to the SAME allocation. No live upgrade. */
int bn_gpu_moe_trellis_store_init_pinned(BnGPUMoETrellisStore *store,
    const BnModel *model,BnGPUBackend *gpu,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget);
int bn_gpu_moe_trellis_store_init_policy(BnGPUMoETrellisStore *store,
    const BnModel *model,BnGPUBackend *gpu,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget,const BnGPUMoETrellisPolicy *policy);
/* A failed device upload may leave a valid admitted RAM entry for retry;
 * failed source reads never publish a partial RAM entry. */
/* admit=1 promotes/admit decode misses. admit=0 borrows decode hits without
 * promotion; misses use separately charged temporaries and never enter either
 * RAM or decode LRU with the default store policy. Explicit RAM admission can
 * populate the RAM LRU without admitting GPU entries. BUSY means budget pinned or the same key pending; no I/O
 * or upload is started in that case. Output must be zeroed/empty and is
 * unchanged on rejection. Successful synchronous upload precedes publication.
 * Release does not synchronize: release only after ALL consumers finish,
 * including requests sharing this lease's buffers. Refillable backends retain
 * idle temporary buffers within the SAME temporary budget until store close;
 * release clears the key/reservation but does not necessarily free storage. */
int bn_gpu_moe_trellis_acquire(BnGPUMoETrellisStore *store, int layer,
    int expert, int admit, BnGPUMoETrellisLease *out);
int bn_gpu_moe_trellis_release(BnGPUMoETrellisStore *store,
    BnGPUMoETrellisLease *lease);
int bn_gpu_moe_trellis_stats(const BnGPUMoETrellisStore *store,
    BnGPUMoETrellisStats *out);
/* BUSY leaves everything intact while leases remain. Empty close is safe. */
int bn_gpu_moe_trellis_store_close(BnGPUMoETrellisStore *store);

/* Prepare once after native core setup and before requests/publication. The
 * backend owns the store and frees it before its weights and borrowed model
 * sources. Expert payload/staging budgets are separate from prepared core,
 * parameters, request resources and driver/library overhead. Failure preserves
 * existing backend resources and releases only staged store allocations.
 * The returned store is borrowed: do not close/free it or retain it past
 * backend release. Finish all queued consumers and release their leases before
 * model/backend teardown. This does not admit full-model execution. */
int bn_gpu_moe_trellis_prepare_model(BnModel *model, size_t ram_budget,
    size_t decode_budget, size_t temporary_budget, size_t staging_budget);
int bn_gpu_moe_trellis_prepare_model_pinned(BnModel *model,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget);
/* Same prepare-before-publication ownership and failure rules as above.
 * NULL policy is the default; no live policy change is supported. */
int bn_gpu_moe_trellis_prepare_model_policy(BnModel *model,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget,const BnGPUMoETrellisPolicy *policy);
/* Returns only the store bound to this model/source. Moving a backend
 * container does not implicitly rebind an existing store to another model. */
BnGPUMoETrellisStore *bn_gpu_moe_trellis_model_store(const BnModel *model);

typedef struct {
    BnGPUMoEExpertBuffers buffers;
    float weight; // route weight; output_scale is applied separately when present
    float output_scale;
    int has_output_scale;
    void *route_gate;
    int route_complement;
} BnGPUMoEResolvedExpert;

typedef struct {
    const BnMoEExpertMap *expert_map;
    const BnGPUMoEResolvedExpert *experts;
    int n_experts;
    int moe_hidden;
    int preserve_output;
} BnGPUMoEResources;

typedef struct {
    void *buffers[BN_MAX_MOE_K * 3];
    int n_buffers;
} BnGPUMoETemporaryBuffers;

int bn_gpu_moe_bridge_get_expert(BnModel *m,
                                  BnSession *sess,
                                  const BnLayerWeights *lw,
                                  int layer,
                                  int expert_idx,
                                  BnGPUMoETemporaryBuffers *temporaries,
                                  BnGPUMoEExpertBuffers *out);

// Prefill borrows RAM/VRAM hits without admission or LRU promotion.
// Misses use temporaries, preserving decode's working sets during a prompt scan.
// Gathered consumers may receive raw-only layouts; tokenwise forward uses get_expert.
int bn_gpu_moe_bridge_get_prefill_expert(BnModel *m, BnSession *sess,
    const BnLayerWeights *lw, int layer, int expert,
    BnGPUMoETemporaryBuffers *temporaries, BnGPUMoEExpertBuffers *out);

int bn_gpu_moe_bridge_resolve_resources(BnGPUMoEResources *out,
                                         BnGPUMoEResolvedExpert *expert_storage,
                                         int expert_cap,
                                         BnModel *m,
                                         BnSession *sess,
                                         const BnLayerWeights *lw,
                                         int layer,
                                         BnGPUMoETemporaryBuffers *temporaries);

void bn_gpu_moe_bridge_release_temporaries(
    BnModel *m,
    BnGPUMoETemporaryBuffers *temporaries);

// Conservative per-slot payload charge across every routed MoE layer.
// Includes backend packed/lazy auxiliary layouts; zero on overflow/no experts.
size_t bn_gpu_moe_bridge_cache_entry_bytes(const BnModel *m,
                                           const BnGPUBackend *gpu);

int bn_gpu_moe_bridge_preload_all(BnModel *m);

#endif // BN_GPU_MOE_BRIDGE_H
