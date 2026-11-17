#ifndef BN_TRANSFORMER_BF16_INTERNAL_H
#define BN_TRANSFORMER_BF16_INTERNAL_H

#include "model.h"
#include "backend_session.h"
#include "gpu_backend.h"
#include "backend_layout.h"
#include "moe_types.h"
#include <stddef.h>
#include <stdint.h>

typedef struct BnSession BnSession;
/* Common session dispatch admits complete dense, Flash, or GLM model layouts.
 * Flash and GLM requests use their prepare_context with explicit budgets;
 * partial operator layouts never enter ordinary generation. */
int bn_transformer_bf16_model_active(const BnModel *model);
int bn_transformer_bf16_max_tokens(const BnBackendSession *session);
int bn_transformer_bf16_capacity(const BnBackendSession *session);
/* Serialized setup boundary after successful restart, position/parity zero.
 * Replaces only KV storage; old aliases expire. Failure after mutation poisons
 * the request; callers must discard it. No active-prefix migration. */
int bn_transformer_bf16_resize_idle_kv(const BnModel *model,BnSession *session,int capacity);
/* prefill identifies the operation, including one-token prompt tails; decode
 * and prefill have different GDN arithmetic even at the same token count.
 * Same state transaction as forward; skips final norm/vocabulary projection,
 * completes pending work and preserves host logits before committing position. */
int bn_transformer_bf16_no_logits_session(BnModel *model, BnSession *session,
    const int *tokens, int n_tokens, int position, int prefill);
float *bn_transformer_bf16_forward_session(BnModel *model, BnSession *session,
    const int *tokens, int n_tokens, int position, int prefill);

/* Read-only backend request quote. Includes explicit device tensor/workspace
 * and host/pinned payload, including temporary setup peak. No allocation or
 * resource mutation; size-query callbacks only. Shared model/expert caches,
 * public BnSession logits/history and allocator/driver overhead are separate.
 * Uses exactly the same admission/allocation walk as prepare_context. */
typedef struct {size_t device_bytes,host_bytes,host_peak_bytes;} BnTransformerBF16RequestMemory;
int bn_transformer_bf16_flash_memory_plan(const BnModel *model,BnGPUBackend *gpu,
    int max_tokens,int capacity,BnTransformerBF16RequestMemory *out);
int bn_transformer_bf16_glm_memory_plan(const BnModel *model,int max_tokens,
    int capacity,BnTransformerBF16RequestMemory *out);

typedef struct {
    BnSession *session;
    const int *tokens;
    int n_tokens, position, prefill;
} BnTransformerBF16SessionRow;
typedef struct {
    uint64_t shared_layers, shared_banks, shared_consumers;
    uint64_t joint_kernel_launches, joint_projection_launches, joint_launches_avoided;
    size_t shared_upload_bytes_avoided, busy_fences;
    /* Cohort-owned placement work only; reserved cache payload is shared,
     * reported once as a maximum rather than summed across borrowers. */
    size_t weight_upload_bytes, cache_hits, cache_misses, cache_hit_bytes;
    size_t cache_fill_copy_bytes, cache_reserved_device_bytes;
    size_t cache_miss_bytes, cache_evictions, cache_bypasses;
} BnTransformerBF16SharedInfo;
/* GLM contexts advance layer-by-layer with private routing/attention/state
 * and shared raw expert uploads/joint GPU FFN launches. Private vector/MMA
 * precision buckets and rowwise arithmetic are kept; this is not a change to
 * projection accumulation order through concatenated rows. The executor serializes model source/cache
 * access and cohort calls. All visible inputs are checked before
 * starting any request. Host logits/history/positions publish only after all
 * readbacks/fences succeed. On execution/I/O failure every participant is
 * poisoned and must be discarded. BUSY is drained at an explicit counted
 * backpressure boundary, with bounded retries; no unbounded internal spin. */
int bn_transformer_bf16_glm_forward_group_session(BnModel *model,
    const BnTransformerBF16SessionRow *rows,int count,
    BnTransformerBF16SharedInfo *info);

/* Same cohort publication and bounded backpressure contract as GLM.
 * Flash HC/PLE/GDN/QSA state stays private. Bound GPU expert caches use an
 * owner-issued union lease with prefill borrowing and per-expert decode admission. */
int bn_transformer_bf16_flash_forward_group_session(BnModel *model,
    const BnTransformerBF16SessionRow *rows,int count,
    BnTransformerBF16SharedInfo *info);

/* Explicit resident BF16 execution; no automatic CPU fallback. The model and
 * borrowed rotary buffer must outlive this request's pending work. Preparation
 * requires an empty backend session. Reset/free releases all request storage.
 * A NULL cos_sin selects the model-owned native cache and its capacity.
 * An explicit cache override is for diagnostics; rows are [cos32|sin32] BF16.
 * Only the qualified dense hybrid layout is accepted.
 */
int bn_transformer_bf16_request_init(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, int max_tokens, int capacity,
    void *cos_sin, int rotary_capacity);
typedef enum {
    BN_TRANSFORMER_BF16_ARITHMETIC_EAGER = 0,
    /* Keep SiLU/product and the first of two residual additions in F32.
     * Retains the qualified contiguous norm and GDN/attention profiles; this
     * does not promise bit equality with arbitrary compiler/autotune choices. */
    BN_TRANSFORMER_BF16_ARITHMETIC_FP32_INTERMEDIATES = 1,
    /* Retained F32 activation/residual intermediates plus the fixed tiled
     * 5120-column FMA norm contract. GDN/attention remain native; arbitrary
     * compiler configurations and full-model reference equality are unproven. */
    BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32 = 2,
    /* Tiled FP32 plus explicit SM120 CP GDN prefill with expf decay staging.
     * Attention retains sigmoid in F32; GDN decode keeps its native contract. */
    BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32 = 3,
    /* CP_FP32 with the explicit fixed compiled R2 gated norm. This matches
     * the qualified SM120 Inductor XBLOCK8 reduction, including for small
     * decode cohorts. Other compiler bounds/autotunes may select a different
     * contract; EAGER and existing CP_FP32 retain their original arithmetic. */
    BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2 = 4,
    /* Retained FP32 with the SM120 non-CP rounded-decay inverse. Norm and
     * attention contracts match CP_FP32; prefill state rounding differs. */
    BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32 = 5,
} BnTransformerBF16Arithmetic;
/* Immutable request/coordinator arithmetic plan for the fixed dense64 layout.
 * DEFAULT entries inherit the base arithmetic. Input entries 0..63 select layer input norms; entry64 selects final norm.
 * Post entries select the post-block two-input RMS. Gated norms are unchanged.
 * This records a chosen reduction contract, not model-specific tensor values
 * or an assumption about an arbitrary compiler's autotuner. */
typedef enum {
    BN_TRANSFORMER_BF16_FFN_DEFAULT = 0,
    BN_TRANSFORMER_BF16_FFN_FUSED_RN = 1,
    BN_TRANSFORMER_BF16_FFN_COMPILED = 2,
} BnTransformerBF16FFNArithmetic;
/* Explicit cohort contract: CP for one recurrent prefill sequence, NONCP for
 * multiple. Qualified against the 188-SM SM120 reference (48 state heads);
 * this is not a portable device autotuner. DEFAULT retains fixed arithmetic. */
typedef enum {
    BN_TRANSFORMER_BF16_GDN_PREFILL_DEFAULT = 0,
    BN_TRANSFORMER_BF16_GDN_PREFILL_SINGLE_CP = 1,
} BnTransformerBF16GDNPrefillSelection;
typedef struct {
    BnTransformerBF16Arithmetic arithmetic;
    /* DEFAULT inherits SEPARATE for EAGER and FUSED_RN otherwise. Non-default
     * choices require retained F32 intermediates and explicit backend support.
     * Admitted requests/coordinators own a canonical copy; restart preserves it. */
    BnTransformerBF16FFNArithmetic ffn_arithmetic;
    BnGPURMSNormReduction post_norm_reductions[64];
    BnGPURMSNormReduction input_norm_reductions[65];
    /* Singleton is the legacy default. BATCH matches cohort-wide FA2 partition
     * selection and requires an explicit backend capability. Immutable after
     * admission; restart preserves it and decode participants must agree.
     * All-one-token packed prefill tails share this partition selection. */
    BnGPUAttentionSplitPolicy attention_split_policy;
    /* SINGLE_CP requires NONCP_FP32 plus both backend recurrence capabilities. */
    BnTransformerBF16GDNPrefillSelection gdn_prefill_selection;
} BnTransformerBF16Policy;
int bn_transformer_bf16_request_init_with_policy(const BnModel *model,
    BnGPUBackend *gpu, BnBackendSession *session, int max_tokens, int capacity,
    void *cos_sin, int rotary_capacity, const BnTransformerBF16Policy *policy);
/* Dense request tensor/workspace and host payload; excludes public BnSession
 * arenas, shared model and opaque backend/allocator/driver bookkeeping.
 * Allocation-free; requires backend size queries and preserves out on error. */
int bn_transformer_bf16_dense_memory_plan(const BnModel *model,
    BnGPUBackend *gpu, int max_tokens, int capacity,
    const BnTransformerBF16Policy *policy, BnTransformerBF16RequestMemory *out);
/* Dense hybrid retained KV payload only; zero for invalid topology/capacity. */
size_t bn_transformer_bf16_dense_kv_bytes(const BnModel *model, int capacity);
int bn_transformer_bf16_batch_prepare_with_policy(const BnModel *model,
    BnGPUBackend *gpu, BnBackendSession *coordinator, int max_requests,
    const BnTransformerBF16Policy *policy);
/* Packed coordinator payload, including retained host/pinned descriptors.
 * Group APIs expose combined budgets; each combined group budget is charged
 * conservatively to both host and device. No allocation or submission. */
int bn_transformer_bf16_batch_memory_plan(const BnModel *model,
    BnGPUBackend *gpu, int max_requests, int max_prefill_tokens,
    const BnTransformerBF16Policy *policy, BnTransformerBF16RequestMemory *out);
/* Copy the admitted policy for a healthy dense request or coordinator.
 * DEFAULT entries are resolved to the base reduction when admitted, so
 * semantically equivalent policies can share one decode cohort. */
int bn_transformer_bf16_policy(const BnBackendSession *session,
    BnTransformerBF16Policy *out);
/* Dense hybrid request only. Profile is fixed until reset/free; restart keeps
 * it. Unsupported/invalid profiles reject before allocating request resources.
 * The compatibility initializer above selects EAGER. No shared model mutation. */
int bn_transformer_bf16_request_init_with_arithmetic(const BnModel *model,
    BnGPUBackend *gpu, BnBackendSession *session, int max_tokens, int capacity,
    void *cos_sin, int rotary_capacity, BnTransformerBF16Arithmetic arithmetic);
/* -1 for absent, failed, or non-dense execution state. */
int bn_transformer_bf16_arithmetic(const BnBackendSession *session);
/* -1 when absent, or after an execution failure. Position commits only after
 * logits and all preceding queued work complete successfully. */
int bn_transformer_bf16_position(const BnBackendSession *session);
/* Restart a healthy backend-only session without reallocating request storage.
 * Clears recurrent state and host token history, completes queued clears, then
 * commits position zero. KV beyond the new position is inaccessible. Capacities
 * and borrowed weights/rotary storage are unchanged. Invalid arguments do no
 * work; a device-operation failure invalidates the request and requires ordinary
 * session reset/reinitialization. Caller serializes access. */
int bn_transformer_bf16_session_restart(const BnModel *model, BnSession *session);

/* Serialized dense request pool. Model/GPU outlive the pool and all leases.
 * Capacity, chunk bound and arithmetic are immutable. Allocation is lazy;
 * retained idle slots still count against the KV payload budget. This budget
 * excludes SSM state, activations, backend workspace, model weights and allocator
 * overhead. No paging or total-device-memory guarantee is implied. A lease may
 * be used by ordinary or mixed execution, then must be returned exactly once.
 * Returned pointers must not be retained. Reuse clears recurrent/history state
 * through session_restart; a failed clear discards that slot. Caller serializes
 * all pool and participant operations and completes them before free. */
/* Caller owns a zero-initialized handle; do not copy an initialized handle.
 * Sessions returned by acquire belong to the pool: do not reset/free them. */
typedef struct {void *state;} BnTransformerBF16Pool;
typedef struct {
    size_t kv_budget_bytes, kv_bytes_per_slot, reserved_kv_bytes, peak_kv_bytes;
    int max_slots, allocated_slots, leased_slots;
    uint64_t created, reused, discarded, evicted, resized;
    /* Actual retained capacities, including idle slots; zero means empty.
     * kv_bytes_per_slot remains the configured full-capacity ceiling. */
    int slot_capacity[16], slot_leased[16];
} BnTransformerBF16PoolInfo;
int bn_transformer_bf16_pool_init(BnTransformerBF16Pool *pool, const BnModel *model,
    BnGPUBackend *gpu, int max_slots, int max_tokens, int capacity,
    size_t kv_budget_bytes, const BnTransformerBF16Policy *policy);
/* 0 success, 1 temporarily exhausted, -1 invalid/preparation failure. *out is
 * NULL on failure/exhaustion. Exhaustion queues no GPU work or allocation. */
int bn_transformer_bf16_pool_acquire(BnTransformerBF16Pool *pool, BnSession **out);
/* Request-sized admission. Round up to a power-of-two capacity (at least
 * max_tokens), capped at the configured capacity. Reuse the smallest adequate
 * idle slot, retaining its full charge. Otherwise reclaim idle slots as needed;
 * never move or evict a lease. BUSY leaves all slots unchanged. Preparation
 * failure may have reclaimed idle slots; counters remain accurate. */
int bn_transformer_bf16_pool_acquire_capacity(BnTransformerBF16Pool *pool,
    int required_capacity, BnSession **out);
/* Healthy leases become idle. Poisoned requests are freed and uncharged. */
int bn_transformer_bf16_pool_release(BnTransformerBF16Pool *pool, BnSession *session);
int bn_transformer_bf16_pool_info(const BnTransformerBF16Pool *pool,
    BnTransformerBF16PoolInfo *out);
void bn_transformer_bf16_pool_free(BnTransformerBF16Pool *pool);

typedef enum {
    BN_TRANSFORMER_BF16_INPUT_NORM,
    BN_TRANSFORMER_BF16_FFN_NORM,
    BN_TRANSFORMER_BF16_FFN_OUTPUT,
    BN_TRANSFORMER_BF16_RESIDUAL,
    BN_TRANSFORMER_BF16_FINAL_NORM,
    BN_TRANSFORMER_BF16_LOGITS,
} BnTransformerBF16Stage;
/* Optional diagnostics; NULL in normal execution. A borrowed view is valid
 * only during the callback. No reentry, mutation, or retaining these views.
 * RESIDUAL is a diagnostic BF16 copy after attention; FP32_INTERMEDIATES does
 * not feed this rounded copy into the subsequent three-term residual sum. */
typedef int (*BnTransformerBF16Observe)(void *context, BnTransformerBF16Stage stage,
    int layer, int position, int n_tokens, BnGPUResidentView view);

/* logits_count must equal model vocabulary size. Input rejection queues no
 * work; execution/read/observer errors invalidate the request until reset.
 * Logits are published only after successful completion. Caller serializes
 * access and never mixes this state with the F32 CPU/GPU execution path.
 */
int bn_transformer_bf16_forward(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, const int32_t *tokens, int n_tokens,
    float *logits, size_t logits_count, BnTransformerBF16Observe observe,
    void *observe_context);

/* Experimental dense hybrid decode coordinator. Empty backend session only;
 * fixed capacity1..256 bounded activation/metadata storage, no layer states.
 * Model/GPU outlive it; reset/free releases storage. Participant requests must
 * already be prepared with the same immutable weights and arithmetic profile.
 * Prefill continues through the ordinary per-session path. Caller serializes
 * coordinator and participants; no concurrent or overlapping submissions. */
typedef struct {
    int capacity, failed;
    uint64_t decode_calls, busy_fences;
    int max_prefill_tokens;
    uint64_t prefill_calls, mixed_calls;
} BnTransformerBF16BatchInfo;
typedef struct {
    BnBackendSession *session;
    int32_t token;
    int position;
    float *logits;
    size_t logits_count;
} BnTransformerBF16DecodeRow;
int bn_transformer_bf16_batch_prepare(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,int max_requests,BnTransformerBF16Arithmetic arithmetic);
int bn_transformer_bf16_batch_info(const BnBackendSession *coordinator,
    BnTransformerBF16BatchInfo *out);
/* One token per distinct request, each at its own committed position/parity.
 * Output ranges must be disjoint and exactly vocabulary-sized. Rejection does
 * no GPU work and preserves output/state. All64 layers and vocabulary rows are
 * batched; outputs/positions commit only after coordinator+participant fences.
 * BUSY operator metadata invokes a counted coordinator fence and one retry.
 * Execution failures poison all participants; no partial logits publication.
 * Batch shape may change GEMM arithmetic versus isolated execution. */
int bn_transformer_bf16_batch_decode(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,const BnTransformerBF16DecodeRow *rows,int count);

/* Explicit packed-prefill capacity, separate from decode request capacity.
 * Admission requires a qualified CP/NONCP F32 profile and grouped-prefill capabilities.
 * max_prefill_tokens bounds total packed rows (1..8192), not request count.
 * The old prepare APIs retain decode-only allocation and admission. */
int bn_transformer_bf16_batch_prepare_packed_with_policy(const BnModel *model,
    BnGPUBackend *gpu,BnBackendSession *coordinator,int max_requests,
    int max_prefill_tokens,const BnTransformerBF16Policy *policy);
typedef struct {
    BnBackendSession *session;
    const int32_t *tokens;
    int n_tokens, position;
    float *logits;
    size_t logits_count;
} BnTransformerBF16PrefillRow;
/* Per-request contiguous chunks, packed in caller order. NULL logits/count0
 * suppresses the head for that row; non-NULL outputs must be disjoint and
 * vocabulary-sized. All validation precedes GPU work. Successful completion
 * atomically publishes requested logits and commits all positions/parities.
 * Execution/observer failure poisons all participants and the coordinator.
 * Observer position=-1; activation stages contain all packed tokens, LOGITS
 * contains only requested output rows, in caller order. Borrowed views remain
 * valid only in the callback. Shape changes can change GEMM arithmetic. */
int bn_transformer_bf16_batch_prefill(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,const BnTransformerBF16PrefillRow *rows,int count,
    BnTransformerBF16Observe observe,void *context);

/* One packed graph may combine decode tokens and contiguous prefill chunks.
 * Caller order defines the shared matrix geometry; each row owns independent
 * KV/SSM/history state. Decode rows require exactly one token and a vocabulary
 * output. Prefill outputs may be omitted. Uses packed preparation/token budget.
 * The entire cohort validates before GPU work; output/position publication is
 * atomic on success, and execution failures poison every participant. Kind is
 * explicit: a one-token prefill tail retains prefill recurrence. Kind selects
 * the recurrent operator, independently of caller-owned prompt/sample progress;
 * a caller may select prefill for one-token rows in a contiguous prefill tail.
 * Mixed recurrent roles require the RSQRT decode capability. The immutable
 * policy selects fixed CP/NONCP or the explicit recurrent-prefill cohort contract.
 * Info counters classify all-prefill, all-decode and mixed submissions. */
typedef enum {
    BN_TRANSFORMER_BF16_STEP_PREFILL = 0,
    BN_TRANSFORMER_BF16_STEP_DECODE = 1,
} BnTransformerBF16StepKind;
typedef struct {
    BnTransformerBF16PrefillRow row;
    BnTransformerBF16StepKind kind;
} BnTransformerBF16StepRow;
int bn_transformer_bf16_batch_step(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,const BnTransformerBF16StepRow *rows,int count,
    BnTransformerBF16Observe observe,void *context);

/* Explicit PLE operator request; not admission for a complete Flash model.
 * Uses the session's existing GPU resource + host execution metadata contracts.
 * Prepare requires an empty session; failure releases only this new request.
 * Budgets include tensor/GEMM workspace and pinned/ID payload respectively,
 * excluding shared model weights, driver/context and allocator/event overhead.
 * Model/GPU must outlive this serialized request; session reset/free releases it.
 */
typedef struct {
    int max_tokens, position, pending_tokens, failed;
    size_t tensor_bytes, workspace_bytes, gpu_bytes, host_bytes, staging_bytes;
} BnTransformerBF16PLEInfo;
typedef struct {
    /* Caller may bind upstream GPU output directly to hidden; no host bridge.
     * Other buffers are read-only diagnostic/downstream views. Valid until
     * next submit/reset/free; data becomes valid after successful completion. */
    BnGPUResidentView hidden, embedding, key, value, gated, normed, out, history;
    const int32_t *row_ids;
    size_t row_id_count;
} BnTransformerBF16PLEViews;
int bn_transformer_bf16_ple_prepare(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, int max_tokens, size_t gpu_budget, size_t host_budget);
int bn_transformer_bf16_ple_info(const BnBackendSession *session,
    BnTransformerBF16PLEInfo *out);
int bn_transformer_bf16_ple_views(const BnBackendSession *session,
    int n_tokens, BnTransformerBF16PLEViews *out);
/* Healthy idle request only. NULL context/history imports EOS/zero defaults.
 * Explicit imports are chronological; context length=ngram-1, history length=
 * (kernel-1)*dilation*streams*dim BF16 words. Setup may transfer/synchronize.
 * Invalid arguments leave state intact. GPU failures require session reset. */
int bn_transformer_bf16_ple_reset(BnBackendSession *session, int position,
    const int32_t *context, size_t context_count,
    const uint16_t *history, size_t history_words);
/* Submit reads caller-produced hidden on the request stream and queues the
 * full PLE chain without fences. BUSY queues nothing and changes no state;
 * caller may schedule other work and retry. No blocking staging retry here.
 * Validation rejection preserves state; execution errors poison until reset.
 * Complete is an EXPLICIT completion fence and commits metadata/history only
 * on success. Call it after downstream work, at the outer transaction boundary.
 * No automatic commit or overlapping submissions for this operator request.
 */
int bn_transformer_bf16_ple_submit(BnBackendSession *session,
    const int32_t *tokens, int n_tokens, int position);
int bn_transformer_bf16_ple_complete(BnBackendSession *session);

/* Combined PLE + Flash GDN/QSA operator request, not full-model
 * admission. Prepare requires an empty session and stages all per-layer state
 * plus shared scratch atomically. Budgets have the same payload exclusions as
 * PLE, and include GDN and GEMM workspace once each. Model/GPU outlive session.
 * The existing PLE views/submit/reset APIs address this same metadata owner;
 * PLE submit must be inside a transaction, and PLE complete is rejected. */
typedef struct {
    int max_tokens, active, position, n_tokens, prefill, includes_ple, failed;
    uint64_t valid_layers, expected_layers, submitted_layers;
    size_t tensor_bytes, workspace_bytes, gpu_bytes, host_bytes;
    int max_context;
    uint64_t valid_gdn_layers, valid_qsa_layers;
    size_t preparation_host_peak_bytes;
    uint64_t expected_ffn, submitted_ffn, hc_attention_done;
    int hc_open;
} BnTransformerBF16FlashInfo;
typedef struct {
    int position;
    /* Input accepts upstream GPU output directly. Shared diagnostic scratch
     * expires at the next GDN submission. State/history always name committed
     * banks; output data is valid after successful outer completion. */
    BnGPUResidentView input, qkvz, ba, post, core, norm, out, history, state;
} BnTransformerBF16FlashGDNViews;
int bn_transformer_bf16_flash_prepare(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, int max_tokens, size_t gpu_budget, size_t host_budget);
/* Explicit per-request context cap, independent of model rotary capacity.
 * Prepared QSA model weights enable QSA layers in new requests. GPU budget
 * covers request tensors/workspace; model weights have their separate budget.
 * Host admission includes temporary slot-map setup; info.host_bytes reports
 * retained storage and preparation_host_peak_bytes includes that setup peak. */
int bn_transformer_bf16_flash_prepare_context(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, int max_tokens, int max_context,
    size_t gpu_budget, size_t host_budget);
int bn_transformer_bf16_flash_info(const BnBackendSession *session,
    BnTransformerBF16FlashInfo *out);
int bn_transformer_bf16_flash_gdn_views(const BnBackendSession *session,
    int layer, int n_tokens, BnTransformerBF16FlashGDNViews *out);
typedef struct {
    int position, max_context, pages, groups;
    /* Shared scratch expires at the next layer submit. Cache/history are
     * layer-local; history names the committed bank. Values written during a
     * transaction become valid after successful outer completion. */
    BnGPUResidentView input, projected, q, k, v, gate, index_projected, index_q;
    BnGPUResidentView scores, indices, attention, out, key_cache, value_cache, compressed, history;
} BnTransformerBF16FlashQSAViews;
int bn_transformer_bf16_flash_qsa_views(const BnBackendSession *session,
    int layer, int n_tokens, BnTransformerBF16FlashQSAViews *out);
int bn_transformer_bf16_flash_qsa_submit(BnBackendSession *session, int layer);
/* Select a nonempty subset of components at matching committed positions.
 * A full attention forward selects every admitted GDN/QSA layer plus PLE. A diagnostic subset commits
 * only those components. Decode requires one token; prefill remains explicit
 * for one-token prompt tails. Begin acquires PLE staging BEFORE any GPU work:
 * BUSY leaves all metadata unchanged, with no hash/gather/queue/wait. Caller
 * serializes access; no overlapping transactions on a request. */
int bn_transformer_bf16_flash_begin(BnBackendSession *session, int n_tokens,
    int position, int prefill, uint64_t layers, int include_ple);
int bn_transformer_bf16_flash_gdn_submit(BnBackendSession *session, int layer);
/* Block mode requires HC pre/post for every selected attention layer and
 * accepts caller-produced FFN blocks when no native FFN model layout is present.
 * Prepared native FFN layouts enable flash_ffn_submit and its additional bounded
 * scratch/banks; HC combine then rejects an unfinished FFN producer.
 * HC post swaps the wide residual buffers; views.hidden then names the next
 * input without a copy. PLE apply similarly advances that residual stream. */
int bn_transformer_bf16_flash_begin_blocks(BnBackendSession *session, int n_tokens,
    int position, int prefill, uint64_t attention_layers, uint64_t ffn_layers, int include_ple);
typedef struct {
    int open, layer, ffn;
    BnGPUResidentView hidden, normalized, down, scaled, gate, mixed, inject, block, residual, combined;
} BnTransformerBF16FlashHCViews;
int bn_transformer_bf16_flash_hc_views(const BnBackendSession *session,int n_tokens,
    BnTransformerBF16FlashHCViews *out);
int bn_transformer_bf16_flash_hc_mix(BnBackendSession *session,int layer,int ffn);
int bn_transformer_bf16_flash_hc_combine(BnBackendSession *session);
int bn_transformer_bf16_flash_attention_submit(BnBackendSession *session,int layer);
/* Prepared model FFN layouts enable bounded native expert streaming in new
 * requests. Submit includes HC mix/post and may return BN_GPU_MOE_BANK_BUSY
 * after partial submission; retry the SAME layer on this serialized request.
 * No internal spin, allocation or CPU activation bridge. One packed routing
 * metadata read completes grouping; outer flash_complete commits positions.
 * RAM-cache access across requests sharing a model must be executor-serialized.
 * Prefill borrows without admission/promotion; decode admits. Source leases end
 * after copying into event-protected pinned banks. Optional model-owned VRAM
 * pools use event-protected leases; GPU hits skip source-cache lookups/copies.
 */
int bn_transformer_bf16_flash_ffn_submit(BnBackendSession *session, int layer);
typedef struct {
    BnGPUExpertCachePreloadInfo cache;
    BnMoEStats source;
} BnTransformerBF16FlashPreloadInfo;
/* Setup only, before any request binds the empty full-capacity expert pool.
 * Borrows source RAM entries without admission/promotion. Output unchanged
 * on failure; source read failure leaves the pool empty and retryable. */
int bn_transformer_bf16_flash_preload_experts(const BnModel *model,
    size_t host_budget_bytes, BnTransformerBF16FlashPreloadInfo *out);
typedef struct {
    int pending, layer, next_expert;
    size_t submitted_blocks, metadata_read_bytes;
    /* Read wall time, source fetch/copy wall time, and BUSY-to-retry spans
     * (including caller scheduling delay), not isolated GPU DMA stall time. */
    double metadata_ms, source_ms, bank_retry_ms;
    BnGPUMoEBankStats banks;
    BnGPUExpertCacheStats cache;
    BnMoEStats source;
} BnTransformerBF16FlashFFNInfo;
typedef struct {
    BnGPUResidentView logits, weights, indices, source_rows, gathered, slots, inverse;
    BnGPUResidentView gate, up, down, weighted, routed;
    BnGPUResidentView shared_gate, shared_up, shared_down, shared_logits, sigmoid, shared, block;
} BnTransformerBF16FlashFFNViews;
int bn_transformer_bf16_flash_ffn_info(const BnBackendSession *session,
    BnTransformerBF16FlashFFNInfo *out);
int bn_transformer_bf16_flash_ffn_views(const BnBackendSession *session, int n_tokens,
    BnTransformerBF16FlashFFNViews *out);
/* Optional prepared vocabulary endpoints on the same transaction. Token IDs
 * are copied during submit, then embedding/repetition stay on device. Head
 * consumes the materialized wide hidden state after all selected blocks/PLE
 * have submitted; it applies output HC mix and projects only the last row.
 * No extra output norm. Neither endpoint commits transaction positions.
 * Views reuse HC/PLE scratch and expire at the next operator submission. */
int bn_transformer_bf16_flash_tokens_submit(BnBackendSession *session,
    const int32_t *tokens, int n_tokens);
int bn_transformer_bf16_flash_head_submit(BnBackendSession *session);
typedef struct {
    BnGPUResidentView ids, embedding, repeated, hidden;
    BnGPUResidentView normalized, down, scaled, gate, mixed, logits;
} BnTransformerBF16FlashIOViews;
int bn_transformer_bf16_flash_io_views(const BnBackendSession *session, int n_tokens,
    BnTransformerBF16FlashIOViews *out);
/* Explicit full-model orchestration. Ordinary session dispatch additionally
 * requires the complete-model layout marker and model GPU binding. Begin copies tokens
 * into prepared request storage. Submit resumes at the current layer/phase on
 * BANK_BUSY, preserving prior attention/PLE work. Successful submit leaves the
 * head queued; inspect/read its resident logits, then flash_complete commits
 * every recurrent/PLE/FFN position. Do not mix manual component submissions
 * into a running model transaction. Model/GPU/request access is serialized. */
int bn_transformer_bf16_flash_model_begin(BnBackendSession *session,
    const int32_t *tokens, int n_tokens, int position, int prefill);
int bn_transformer_bf16_flash_model_submit(BnBackendSession *session);
int bn_transformer_bf16_flash_ple_apply(BnBackendSession *session,
    const int32_t *tokens,int n_tokens,int position);
/* Reject incomplete transactions without fencing. Once every selected
 * component is submitted, one explicit stream fence precedes ALL commits.
 * Execution/completion failure poisons the request with old positions/banks
 * preserved; release/reprepare is required. Healthy restart clears all state
 * without allocating and completes the clears before committing zero. */
int bn_transformer_bf16_flash_complete(BnBackendSession *session);
int bn_transformer_bf16_flash_restart(BnBackendSession *session);

/* Complete post-input-normalization KDA attention sublayer. Workspace views
 * are request-owned, preallocated, exact-sized and pairwise disjoint, including
 * input/output and both state/history generations. Execution queues only; no
 * allocation, host transfers or fences. Reserve A8 scratch for tokens *
 * max(dim, heads*128) before submission. Commit next state/history only after
 * successful synchronization. Backend errors can leave uncommitted outputs
 * partially written: discard the request on failure, no rollback guarantee.
 * Metadata preflight cannot validate opaque backend allocation generations. */
typedef struct {
    BnGPUResidentView projected, qkv, tail, fa_ga, beta, fa, ga;
    BnGPUResidentView g1, g2, conv, core, gated;
} BnTransformerBF16KDAWorkspace;
int bn_transformer_bf16_kda_layer(const BnBackendBF16KDAWeights *weights,
    void *request, int n_tokens, const BnTransformerBF16KDAWorkspace *workspace,
    BnGPUResidentView x, BnGPUResidentView initial_history,
    BnGPUResidentView next_history, BnGPUResidentView initial_state,
    BnGPUResidentView next_state, BnGPUResidentView out);

/* Materialized four-stream attention transition: normalized mHC pre -> KDA
 * -> mHC post. This is an explicit arithmetic profile, not a promise that the
 * fused/deferred inter-layer mHC profile has the same rounding. Every view is
 * preallocated and belongs to one request. The normalized pre output is the
 * KDA input; branch_out is its output before residual mixing. No copies,
 * allocation, fences or position/state commits. On execution failure discard
 * the request; opaque backend failures may follow partial submission. */
typedef struct {
    const BnBackendBF16KDAWeights *weights;
    BnGPUMHCPreResidentPlan mhc;
    BnTransformerBF16KDAWorkspace kda;
    BnGPUResidentView initial_history, next_history, initial_state, next_state;
    BnGPUResidentView branch_out, residual_out;
    /* Zero preserves materialized pre/KDA/post. Deferred returns the branch,
     * current residual and pre coefficients for the next fused boundary;
     * residual_out is unused. has_previous selects default fused previous
     * post -> current pre using the four previous views below. */
    int deferred, has_previous;
    BnGPUResidentView previous_residual, previous_branch, previous_post, previous_comb;
} BnTransformerBF16MHCKDAPlan;
int bn_transformer_bf16_mhc_kda_attention(void *request,
    const BnTransformerBF16MHCKDAPlan *plan);

/* Resident dense clamped SwiGLU, with merged gate/up and raw quant weights.
 * Reserve n_tokens * max(dim, hidden_dim) A8 elements before submission.
 * All four exact-sized BF16 views are disjoint and request-owned. No position
 * commits, allocation, copies or fences. Discard request on backend failure. */
int bn_transformer_bf16_dense_ffn(const BnBackendBF16DenseFFNWeights *weights,
    void *request, int n_tokens, BnGPUResidentView x,
    BnGPUResidentView gate_up, BnGPUResidentView activated, BnGPUResidentView out);

/* Complete dense decoder block, returning deferred FFN state (out, ffn.residual,
 * ffn.post, ffn.comb). The attention plan must be deferred; has_previous accepts
 * the preceding block's deferred FFN state. All active views across the entire
 * block are disjoint and metadata-validated before any submission. Backend
 * errors may partially submit; discard request. No state/position commits,
 * copies, allocations or fences. Reserve max(dim, heads*128, hidden_dim)*tokens
 * A8 scratch. This helper does not admit an incomplete whole-model layout. */
typedef struct {
    BnTransformerBF16MHCKDAPlan attention;
    BnGPUMHCPreResidentPlan ffn;
    const BnBackendBF16DenseFFNWeights *weights;
    BnGPUResidentView gate_up, activated, out;
} BnTransformerBF16DenseKDABlockPlan;
int bn_transformer_bf16_dense_kda_block(void *request,
    const BnTransformerBF16DenseKDABlockPlan *plan);

/* Complete routed+shared MoE FFN, after normalization. Caller preallocates
 * exact active views and fixed two-slot bank/quant/BLAS scratch. The CPU source map is
 * borrowed from the immutable prepared weight resource. Source I/O/cache is
 * executor-serialized; host_state and work are request-local. No activation
 * leaves the GPU. One small grouping-metadata read completes routing before
 * streaming banks. No logical position/state commits. */
typedef struct {
    const BnBackendBF16MoEWeights *weights;
    const BnMoEIO *source;
    BnMoEState *host_state;
    int layer,n_tokens,prefill,bank_capacity;
    BnGPUResidentView x,cast,logits,route_weights,indices;
    BnGPUResidentView metadata,cursors,slots,inverse,gathered;
    BnGPUResidentView gate,up,down,weighted,routed;
    BnGPUResidentView shared_gate_up,shared_activated,shared_out,out;
} BnTransformerBF16MoEPlan;
typedef struct {
    int pending,failed,next_expert;
    unsigned bank;
    uint64_t generation;
    int32_t offsets[520];
    size_t submitted_banks,submitted_experts,metadata_read_bytes;
    BnTransformerBF16MoEPlan active;
} BnTransformerBF16MoEWork;
/* Zero work initially. Begin queues projection/routing, reads metadata, then
 * queues shared FFN/input quantization. Invalid visible metadata submits no
 * work; execution or I/O failure poisons work, so discard
 * request. Submit streams until BUSY or done, preserving exact continuation.
 * Retry the SAME work on BUSY. Success queues reduction+shared addition;
 * caller fences before consuming outputs. No rollback on partial failure. */
int bn_transformer_bf16_moe_begin(void *request,const BnTransformerBF16MoEPlan *plan,
    BnTransformerBF16MoEWork *work);
int bn_transformer_bf16_moe_submit(void *request,BnTransformerBF16MoEWork *work);

/* Complete GLM pre-indexer from caller-owned hidden input and normalized MLA
 * q-rank input. Caller preallocates all views plus BLAS/A8 scratch for
 * max(dim,q_rank)*tokens. One global alias/type/count preflight precedes work.
 * Outputs FP8 queries, scaled F32 head weights, newly completed pooled keys
 * and next tail. No allocation, host bridge, fence or position commit.
 * Execution failure may partially submit; discard request without accepting
 * next state. This does not include cache addressing, score/top-k or MLA. */
typedef struct {
    const BnBackendBF16IndexerWeights *weights;
    int layer,n_tokens,position;
    BnGPUResidentView x,qr,initial_tail,next_tail;
    BnGPUResidentView q,key_raw,key,gate,cast,head_raw;
    BnGPUResidentView q_fp8,q_scales,head_weights,compressed,compressed_scales;
} BnTransformerBF16IndexerPlan;
int bn_transformer_bf16_indexer(void *request,const BnTransformerBF16IndexerPlan *plan);

/* Pre-indexer -> append newly completed pools -> gather visible pool history.
 * Caller owns/reserves cache, page table, slots and all activation/scratch
 * views. Slots are scheduler metadata, not host activation intermediates.
 * Cache already contains every pool before floor(position/4); slots match
 * the new logical pool range through block_table, without duplicates.
 * One global preflight covers the complete chain. Status starts at zero and
 * is sticky across gathers; caller fences/checks it before accepting next
 * state or publishing outputs. Errors may partially modify cache/next tail;
 * discard request on failure. No allocations, host copies or position commits.
 * Sequential single-request F32-scale cache profile, physical pages64. */
typedef struct {
    BnTransformerBF16IndexerPlan indexer;
    int num_cache_blocks,logical_blocks;
    BnGPUResidentView cache,block_table,slots,gathered,scales,status;
} BnTransformerBF16IndexerCachePlan;
int bn_transformer_bf16_indexer_cache(void *request,
    const BnTransformerBF16IndexerCachePlan *plan);

/* Pre-indexer -> paged append/gather -> signed FP8 causal scores.
 * Metadata/aliases for the entire chain are checked before any submission.
 * scores is F32[tokens,floor((position+tokens)/4)], empty if no complete pool.
 * All cache-plan reservation/status/failure obligations still apply. Caller
 * supplies x and normalized QR; every subsequent activation/history is native.
 * No allocation, transfer, fence, host activation bridge or position commit.
 * Contiguous gathered-history prefill, not paged decode/top-k/MLA. */
typedef struct {
    BnTransformerBF16IndexerCachePlan cache;
    BnGPUResidentView scores;
} BnTransformerBF16IndexerScoresPlan;
int bn_transformer_bf16_indexer_scores(void *request,
    const BnTransformerBF16IndexerScoresPlan *plan);

/* Complete prefill indexer through pool selection/token expansion, from x/QR.
 * Global preflight covers all outputs/aliases before any submission. In short
 * mode end<=2048, indexed.scores is empty and scoring is bypassed; cache and
 * native pre-indexer still update. Normal mode uses the signed FP8 scorer.
 * pools=I32[tokens,512], out=I32[tokens,2051], no count column. See the GPU
 * select contract for deterministic order/ties and both packing layouts.
 * Existing cache status, reservation and failure obligations apply. Caller
 * checks sticky status after fencing. No allocation/copy/fence/position commit.
 * Paged decode/MLA and downstream order/tie numerical qualification remain. */
typedef struct {
    BnTransformerBF16IndexerScoresPlan indexed;
    int short_prefill;
    BnGPUResidentView pools,out;
} BnTransformerBF16IndexerSelectPlan;
int bn_transformer_bf16_indexer_select(void *request,
    const BnTransformerBF16IndexerSelectPlan *plan);

/* Native sequential decode indexer: pre-indexer -> append newly completed
 * pools -> direct paged FP8 score -> selection. No gather/history staging.
 * Short decode (end<=2048) uses empty scores and bypasses scoring. Views are
 * globally preflighted before work. Cache already contains pools before
 * floor(position/4), slots map newly completed pools through block_table.
 * All reservation/sticky-status/failure obligations of cache/select plans
 * apply. Caller fences/checks status before accepting state. No allocation,
 * transfer, fence or position commit. Multi-request/speculative trees and MLA
 * are not included; selection order/tie downstream qualification remains. */
typedef struct {
    BnTransformerBF16IndexerPlan indexer;
    int num_cache_blocks,logical_blocks,short_decode;
    BnGPUResidentView cache,block_table,slots,scores,pools,out,status;
} BnTransformerBF16IndexerDecodePlan;
int bn_transformer_bf16_indexer_decode(void *request,
    const BnTransformerBF16IndexerDecodePlan *plan);

/* NoPE MLA input preparation (QR1536/KV512, eps1e-5): independent raw QA/KVA, BF16-gamma RMSNorm,
 * raw QB and identity-materialized BF16 per-head query absorption. Caller
 * reserves BLAS/A8 scratch for tokens*max(dim,q_rank,heads*key_dim) and every
 * BF16 activation. Metadata/aliases checked globally before six submissions.
 * Produces normalized QR for the indexer and normalized latent KV for cache.
 * No allocation/copy/fence/position commit. Does not include attention. */
typedef struct {
    const BnBackendBF16MLAWeights *weights;
    int layer,n_tokens;
    BnGPUResidentView x,q_raw,qr,kv_raw,kv,q,query;
} BnTransformerBF16MLAPreparePlan;
int bn_transformer_bf16_mla_prepare(void *request,
    const BnTransformerBF16MLAPreparePlan *plan);
/* Latent attention context -> per-head value expansion -> raw output
 * projection. Caller supplies context BF16[tokens,heads,kv_rank]; reserves
 * scratch for tokens*heads*value_dim plus all outputs. No attention implied. */
typedef struct {
    const BnBackendBF16MLAWeights *weights;
    int layer,n_tokens;
    BnGPUResidentView context,values,out;
} BnTransformerBF16MLAOutputPlan;
int bn_transformer_bf16_mla_output(void *request,
    const BnTransformerBF16MLAOutputPlan *plan);

/* MLA preparation -> NoPE fp8_ds_mla cache append. All aliases/metadata
 * preflight before seven submissions. Caller supplies unique physical slots
 * and cache, preserves committed history and commits position only after its
 * fence succeeds. Failure may partially write activations/cache: discard
 * request. No allocation/host bridge/fence/position commit; no attention. */
typedef struct {
    BnTransformerBF16MLAPreparePlan prepare;
    int num_cache_blocks;
    BnGPUResidentView slots,cache;
} BnTransformerBF16MLACachePlan;
int bn_transformer_bf16_mla_cache(void *request,
    const BnTransformerBF16MLACachePlan *plan);

/* Validated logical selection -> paged NoPE MLA -> value/output projection.
 * All twelve views preflight before three operation submissions. The caller
 * reserves scratch and checks sticky device status before committing state. */
typedef struct {
    const BnBackendBF16MLAWeights *weights;
    int layer;
    BnGPUMlaFP8AttentionResidentPlan attention;
    BnGPUResidentView values,out;
} BnTransformerBF16MLAAttentionPlan;
int bn_transformer_bf16_mla_attention(void *request,
    const BnTransformerBF16MLAAttentionPlan *plan);


/* Resident indexed NoPE MLA from hidden input through output projection.
 * Preparation QR/query/cache, indexer inputs/selection/status and attention
 * are linked exact views. The two cache/table domains remain independent
 * (tokens versus pooled states). Global preflight precedes 20 short or21
 * sparse submissions. All buffers/workspace caller-reserved; no allocation,
 * host bridge, fence or position commit. Failure requires request discard.
 * Selection is the existing chronological policy; its effects versus raw
 * original atomic ordering remain subject to full-model qualification. */
typedef struct {
    BnTransformerBF16MLACachePlan prepare;
    BnTransformerBF16IndexerDecodePlan indexer;
    BnTransformerBF16MLAAttentionPlan output;
} BnTransformerBF16IndexedMLAPlan;
int bn_transformer_bf16_indexed_mla(void *request,
    const BnTransformerBF16IndexedMLAPlan *plan);


/* Complete non-final, deferred mHC -> indexed MLA -> fused FFN entry -> MoE.
 * Incoming deferred state is optional; all active views across the whole
 * block are preflighted before submission. Two control reads complete the
 * 4-byte attention status and 1172-byte routing metadata; no activation leaves
 * the device. Status errors stop before expert I/O. No position/tail commits.
 * The output state is (moe.out,ffn.residual,ffn.post,ffn.comb). Caller fences
 * before accepting it. All buffers, bank/BLAS scratch and host I/O state are
 * caller-owned. BUSY resumes SAME work without replaying attention. On any
 * execution/I/O failure discard the request; metadata rejection changes no work.
 * This does not admit an incomplete whole-model layout or final-layer contract. */
typedef struct {
    BnGPUMHCPreResidentPlan pre,ffn;
    int has_previous;
    BnGPUResidentView previous_residual,previous_branch,previous_post,previous_comb;
    BnTransformerBF16IndexedMLAPlan attention;
    BnTransformerBF16MoEPlan moe;
} BnTransformerBF16MLAMoEBlockPlan;
typedef struct {
    void *request;
    int pending,failed;
    size_t status_read_bytes;
    BnTransformerBF16MoEWork moe;
} BnTransformerBF16MLAMoEBlockWork;
int bn_transformer_bf16_mla_moe_begin(void *request,const BnTransformerBF16MLAMoEBlockPlan *plan,
    BnTransformerBF16MLAMoEBlockWork *work);
int bn_transformer_bf16_mla_moe_submit(void *request,BnTransformerBF16MLAMoEBlockWork *work);


/* KDA attention -> fused FFN pre -> streamed shared/routed MoE. Requires the
 * deferred arithmetic profile and returns (moe.out,ffn.residual,ffn.post,ffn.comb).
 * All 55 possible views are checked globally before submission. Begin performs
 * one small routing-metadata read; no activation bridge or allocation. Zero
 * work initially. BUSY retries the SAME work through submit without replaying
 * attention or changing state generations. Caller fences before committing
 * state/history/positions; any execution or I/O failure poisons work and the
 * request must be discarded. This does not implement final-layer contraction
 * or admit an incomplete whole-model layout. */
typedef struct {
    BnTransformerBF16MHCKDAPlan attention;
    BnGPUMHCPreResidentPlan ffn;
    BnTransformerBF16MoEPlan moe;
} BnTransformerBF16KDAMoEBlockPlan;
typedef struct {
    void *request;
    int pending,failed;
    BnTransformerBF16MoEWork moe;
} BnTransformerBF16KDAMoEBlockWork;
int bn_transformer_bf16_kda_moe_begin(void *request,const BnTransformerBF16KDAMoEBlockPlan *plan,
    BnTransformerBF16KDAMoEBlockWork *work);
int bn_transformer_bf16_kda_moe_submit(void *request,BnTransformerBF16KDAMoEBlockWork *work);

/* Final deferred FFN post -> four-stream mean -> final RMS -> raw Q8 head.
 * All eight exact-sized request views are disjoint and validated before work.
 * Normalize all prompt rows to preserve the reference's256-row RMS launch
 * boundary. last_only selects just the final normalized row for head logits;
 * otherwise head batch geometry is n_tokens. Caller reserves n_tokens*dim
 * A8 scratch and fences before consuming logits; no allocation, host bridge,
 * logical state commit or internal fence. Discard on partial backend failure. */
typedef struct {
    const BnBackendBF16GLMModel *weights;
    int n_tokens,last_only;
    BnGPUResidentView residual,branch,post,comb,completed,collapsed,normalized,logits;
} BnTransformerBF16GLMOutputPlan;
int bn_transformer_bf16_glm_output(void *request,const BnTransformerBF16GLMOutputPlan *plan);

/* Explicit whole GLM execution, currently a qualification API only. Ordinary
 * loader/generation admission remains guarded. Uses fused-recurrent KDA and
 * chronological sparse selection; default original chunk-KDA/atomic top-k
 * ordering equivalence is not implied. Model/GPU/source outlive the request;
 * all access, including shared expert source/cache access, is serialized.
 * Preparation requires an empty session. Budgets cover device payload,
 * pinned banks/descriptors and request host staging/metadata, excluding the
 * already prepared model, shared source cache, driver/allocator overhead.
 * Every execute buffer and expert staging buffer is reserved at preparation.
 * Begin validates/copies IDs and queues embedding. Submit runs all layers,
 * returning BANK_BUSY with its exact continuation retained. Once it returns0,
 * read/check resident logits, then complete fences and commits ALL positions
 * and recurrent generations. Partial execution failure poisons the request;
 * release/reprepare is required. Healthy idle restart clears without resize.
 * Logits are BF16[vocab], final sampled row; all prompt rows are normalized. */
typedef struct {
    int max_tokens,capacity,position,parity,active,failed,layer,head_queued;
    size_t device_bytes,host_bytes;
    void *request;
    BnGPUResidentView logits;
    BnMoEStats source;
} BnTransformerBF16GLMRequestInfo;
int bn_transformer_bf16_glm_prepare(BnBackendSession *session,
    const BnBackendModel *model,const BnMoEIO *source,int max_tokens,int capacity,
    size_t device_budget,size_t host_budget);
/* Bind a shared-model backend-only session to the qualified recurrent-KDA,
 * chronological sparse-index GLM profile. Host readback/ID buffers are included
 * in host_budget. Model, source I/O/cache, and GPU must outlive the request.
 * Setup requires an empty, position-zero session created for this exact model.
 * Common forward/prefill dispatch publishes logits/history only after commit.
 * Calls sharing a model/source I/O are serialized. */
int bn_transformer_bf16_glm_prepare_context(const BnModel *model,
    BnSession *session, int max_tokens, int capacity,
    size_t device_budget, size_t host_budget);
int bn_transformer_bf16_glm_begin(BnBackendSession *session,
    const int32_t *tokens,int n_tokens,int position,int prefill);
int bn_transformer_bf16_glm_submit(BnBackendSession *session);
int bn_transformer_bf16_glm_complete(BnBackendSession *session);
int bn_transformer_bf16_glm_restart(BnBackendSession *session);
int bn_transformer_bf16_glm_info(const BnBackendSession *session,
    BnTransformerBF16GLMRequestInfo *out);

/* Read-only inspection of the committed generation, allowed only while idle
 * and healthy. No queueing, allocation or implicit fence. Exactly one of
 * (history,state) and (latent,index,tail) is populated. Views expire at the
 * next begin/restart or resource release; model/GPU lifetime rules still apply.
 * Callers must not write through these borrowed views. */
typedef struct {
    int layer,position;
    BnGPUResidentView history,state,latent,index,tail;
} BnTransformerBF16GLMLayerStateViews;
int bn_transformer_bf16_glm_layer_views(const BnBackendSession *session,int layer,
    BnTransformerBF16GLMLayerStateViews *out);

#endif
