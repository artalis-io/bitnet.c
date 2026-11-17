#ifndef BN_GENERATE_BATCH_H
#define BN_GENERATE_BATCH_H

#include "generate_batch_plan.h"
#include <stddef.h>
#include <stdint.h>

#define BN_GENERATE_BATCH_QUEUE_CAPACITY 64

typedef struct {void *state;} BnGenerateBatch;
typedef struct {
    void *handle;
    float *logits;
    size_t logits_count;
} BnGenerateBatchLease;
typedef struct {
    uint64_t id;
    BnGenerateBatchLease lease;
    const int32_t *tokens;
    int n_tokens, position, is_prefill, decode_role, emit_token;
} BnGenerateBatchExecutionRow;
typedef struct {
    void *context;
    /* 0 acquires one distinct healthy lease, 1 means temporarily exhausted,
     * -1 means failure. Failure/exhaustion must leave *out zero. */
    int (*acquire)(void *context,BnGenerateBatchLease *out);
    /* Complete and fence the whole cohort before success. Populate every
     * lease's vocabulary-sized logits, including intermediate prefill heads.
     * Token/query/role ordering is defined by the host plan. */
    int (*execute)(void *context,const BnGenerateBatchExecutionRow *rows,int count);
    /* Infallible ownership cleanup. discard requires destroying the execution
     * state, even if it appears healthy. Ordinary returns may be pooled. */
    void (*release)(void *context,BnGenerateBatchLease lease,int discard);
    /* Finite monotonic clock, used for queue/admission/token timestamps. */
    double (*time_ms)(void *context);
} BnGenerateBatchBackend;
typedef struct {
    uint64_t id;
    int prompt_tokens, max_outputs, required_capacity;
} BnGenerateBatchAdmission;
/* Same lease/return contract as backend.acquire. Metadata is immutable and
 * borrowed only for this call. Required capacity includes all prompt inputs
 * and all generated inputs except the final output: prompt+max_outputs-1. */
typedef int (*BnGenerateBatchAcquireRequest)(void *context,
    const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out);
typedef enum {
    BN_GENERATE_BATCH_ADMITTED,
    BN_GENERATE_BATCH_TOKEN,
    BN_GENERATE_BATCH_CANCELLED,
    BN_GENERATE_BATCH_FAILED
} BnGenerateBatchEventKind;
typedef struct {
    BnGenerateBatchEventKind kind;
    uint64_t id;
    int token, output_index, complete;
    double submitted_ms, admitted_ms, timestamp_ms;
} BnGenerateBatchEvent;
typedef void (*BnGenerateBatchNotify)(void *context,const BnGenerateBatchEvent *event);
typedef struct {
    int queued, live, peak_live, failed;
    uint64_t submitted, completed, cancelled, acquired, released;
    uint64_t steps, prefill_tokens, decode_tokens, output_tokens;
} BnGenerateBatchInfo;

/* Bounded, serialized greedy generation. The backend and callback contexts
 * outlive the runtime. Limits are immutable; queue capacity includes live
 * records. No backend work at init. Caller owns a zero-initialized handle;
 * do not copy an initialized handle.
 * No concurrent use or callback reentry, except read-only info(). */
int bn_generate_batch_init(BnGenerateBatch *batch,const BnGenerateBatchLimits *limits,
    int vocab_size,const BnGenerateBatchBackend *backend,
    BnGenerateBatchNotify notify,void *notify_context);
/* Opt-in request-aware acquisition; backend.acquire is ignored and may be
 * NULL. Other callbacks/context have the same contract. No callback at init.
 * The original backend struct and initializer retain their layout/behavior. */
int bn_generate_batch_init_request_aware(BnGenerateBatch *batch,
    const BnGenerateBatchLimits *limits,int vocab_size,
    const BnGenerateBatchBackend *backend,BnGenerateBatchAcquireRequest acquire_request,
    BnGenerateBatchNotify notify,void *notify_context);
/* Configure before the first accepted submission. Default is FIXED. No
 * callbacks; invalid policy, reentry or any prior submission rejects unchanged,
 * including after cancellation/completion. Free/init opens a new configuration. */
int bn_generate_batch_set_prefill_policy(BnGenerateBatch *batch,
    BnGenerateBatchPrefillPolicy policy);
/* Borrow prompt/output arrays until the terminal event or free(). Prompt
 * contents stay immutable while borrowed. Outputs
 * need max_outputs elements and may not overlap any active prompt/output.
 * Submission time starts queueing latency. IDs must be unique while pending.
 * 0 accepted, 1 queue full, -1 invalid/failed runtime; rejection changes nothing.
 * Arrival timing is caller-controlled: enqueue when a request actually arrives. */
int bn_generate_batch_enqueue(BnGenerateBatch *batch,uint64_t id,
    const int32_t *prompt,int prompt_tokens,int32_t *outputs,int max_outputs);
/* Preserve an arrival recorded before the serialized worker could enqueue it.
 * submitted_ms must be finite, in the backend clock's domain, and no later
 * than the current backend time. It is reported unchanged in every event.
 * Queue order remains enqueue order; timestamps do not reorder requests.
 * Ownership and return values are the same as enqueue(). */
int bn_generate_batch_enqueue_at(BnGenerateBatch *batch,uint64_t id,
    const int32_t *prompt,int prompt_tokens,int32_t *outputs,int max_outputs,
    double submitted_ms);
/* Admit queued requests in FIFO order, then run one bounded cohort. 1 means
 * success, 0 no executable work (including temporary lease exhaustion), -1
 * failure. Validate all selected greedy samples before publishing any progress
 * or tokens. All token events share one post-execution timestamp and precede
 * completed-lease reclamation. Failure preserves prior output prefixes, releases
 * every live lease with discard=1, and reports terminal failure to all pending
 * requests. Reinitialize after failure. Scheduler scratch is fixed; backend
 * admission may allocate within its own budget. */
int bn_generate_batch_step(BnGenerateBatch *batch);
/* Cancel one queued/live request between steps. Published tokens stay valid;
 * healthy live leases return normally. Missing IDs and reentry reject. */
int bn_generate_batch_cancel(BnGenerateBatch *batch,uint64_t id);
/* Fail the runtime between steps without executing another cohort. Discard
 * every live lease and report FAILED for all queued/live requests, preserving
 * published output prefixes. 0 on success; -1 for invalid, busy or already
 * failed handles. Reinitialize before accepting more work. */
int bn_generate_batch_abort(BnGenerateBatch *batch);
int bn_generate_batch_info(const BnGenerateBatch *batch,BnGenerateBatchInfo *out);
/* Release all live leases and borrowed arrays, without notifications. */
void bn_generate_batch_free(BnGenerateBatch *batch);

#endif
