#ifndef BN_GENERATE_BATCH_PLAN_H
#define BN_GENERATE_BATCH_PLAN_H
#include <stddef.h>
#include <stdint.h>

#define BN_GENERATE_BATCH_MAX_REQUESTS 16
#define BN_GENERATE_BATCH_MAX_CHUNK_TOKENS 8192
/* Aggregate host scheduling bound. Each backend must separately prepare or
 * reject its selected cohort geometry; this is not a packed-GPU row limit. */
#define BN_GENERATE_BATCH_MAX_TOKENS (BN_GENERATE_BATCH_MAX_REQUESTS * BN_GENERATE_BATCH_MAX_CHUNK_TOKENS)

typedef struct {
    int max_requests, max_prefills, max_chunk, token_budget, capacity;
    /* Zero inherits token_budget. Otherwise max_requests <= value <=
     * token_budget; applied only when at least one logical decode is selected.
     * Prefill-only steps retain the full token_budget. */
    int mixed_token_budget;
} BnGenerateBatchLimits;

/* Candidates are already admitted, in stable admission order. Token storage,
 * sessions, arrival queues and sampling remain owned by the caller. Completed
 * candidates may be present and are ignored. A completed prompt has already
 * published its first generated token. */
typedef struct {
    int prompt_tokens, prefilled, generated, max_outputs;
} BnGenerateBatchCandidate;

typedef struct {
    int candidate, n_tokens, position, is_prefill, emit_token;
    /* Recurrent execution role: 1 only in the contiguous decode prefix.
     * A logical decode after the first prefill retains role zero. */
    int decode_role;
} BnGenerateBatchRow;

typedef struct {
    int count, total_tokens;
    BnGenerateBatchRow rows[BN_GENERATE_BATCH_MAX_REQUESTS];
} BnGenerateBatchPlan;

typedef enum {
    BN_GENERATE_BATCH_PREFILL_FIXED = 0,
    BN_GENERATE_BATCH_PREFILL_RECLAIM = 1
} BnGenerateBatchPrefillPolicy;

/* RECLAIM chooses the largest common quota bounded by max_chunk for which
 * the sum of clipped prompt remainders fits the remaining budget. Selection,
 * decode reservation and stable ordering are identical to FIXED. An indivisible
 * remainder may remain unused. Invalid policy leaves *out unchanged. */
int bn_generate_batch_plan_with_prefill_policy(const BnGenerateBatchLimits *limits,
    const BnGenerateBatchCandidate *candidates, int count,
    BnGenerateBatchPrefillPolicy policy, BnGenerateBatchPlan *out);

/* Reserve one token per selected decode first. Apply mixed_token_budget when
 * decodes were selected. Divide the remaining budget
 * equally among selected prefills, bounded by max_chunk and prompt remainder;
 * unused shares are not redistributed. Then stably sort by query length and
 * derive the recurrent decode prefix. This matches the qualified round-robin
 * mixed cohort contract, rather than promising arbitrary scheduler equivalence.
 * Return 1 for work, 0 when no work is ready, -1 for invalid input. No allocation,
 * callbacks or input mutation. On rejection *out is unchanged. The caller must
 * validate execution and all samples before publishing progress. */
int bn_generate_batch_plan(const BnGenerateBatchLimits *limits,
    const BnGenerateBatchCandidate *candidates, int count, BnGenerateBatchPlan *out);

/* Pure aggregate serving-payload ledger. Shared immutable/cache reservations
 * count once; every retained idle slot continues to count. Host includes
 * pinned payload. Allocator, model-file mappings and driver overhead are not
 * physical RSS/VRAM guarantees. Caller serializes reservations and actual
 * storage operations, and fences/destroys storage BEFORE dropping a ticket. */
typedef struct {size_t host_bytes,device_bytes;} BnGenerateBatchMemory;
typedef struct {BnGenerateBatchMemory retained,peak;} BnGenerateBatchMemoryQuote;
typedef struct {int slot;uint64_t generation;const void *owner;} BnGenerateBatchMemoryTicket;
typedef struct {
    BnGenerateBatchMemoryQuote quote;
    uint64_t generation;
    int state; /* 0 empty, 1 preparing, 2 leased, 3 retained idle. */
} BnGenerateBatchMemorySlot;
typedef struct {
    BnGenerateBatchMemory limit,shared,used,peak;
    BnGenerateBatchMemorySlot slots[BN_GENERATE_BATCH_MAX_REQUESTS];
    uint64_t reservations,reuses,busy_returns;
    int initialized,shared_preparing;
} BnGenerateBatchMemoryLedger;
/* Empty caller-owned ledger; do not copy an initialized ledger. Shared quote includes the setup peak as well as
 * retained state. No allocation/callback. Invalid input preserves everything. */
int bn_generate_batch_memory_init(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemory limits,BnGenerateBatchMemoryQuote shared);
/* Publish shared setup completion before any slot reservation. Setup failure
 * requires releasing shared storage and discarding this caller-owned ledger. */
int bn_generate_batch_memory_shared_commit(BnGenerateBatchMemoryLedger *ledger);
/* Replace the shared reservation between serialized setup stages, before any
 * slot admission. The quote must cover ALL still-live shared storage as well
 * as the next stage's peak. Caller has released obsolete setup temporaries.
 * Error leaves the ledger unchanged; the historical peak never decreases. */
int bn_generate_batch_memory_shared_prepare_as(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryQuote shared);
/* Tighten the admitted shared upper bound after construction. Values may
 * only shrink; no slot admission until this transaction succeeds. */
int bn_generate_batch_memory_shared_commit_as(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemory retained);
/* Reserve the entire preparation peak before allocation. 0 success,1 BUSY
 * because retained/active slots occupy the budget,-1 invalid or impossible
 * even with no slots. *ticket is unchanged on rejection. BUSY does not change
 * reservations; busy_returns is diagnostic. Commit lowers peak to retained;
 * abort/drop releases either preparation or retained reservation. */
int bn_generate_batch_memory_reserve(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryQuote quote,BnGenerateBatchMemoryTicket *ticket);
/* Read-only feasibility after reclaiming all retained IDLE slots. 0 fits,
 * 1 blocked by preparing/live leases or slot count, -1 invalid/impossible.
 * Does not evict, change counters or make a reservation. */
int bn_generate_batch_memory_check_active(const BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryQuote quote);
/* Check a future request against retained shared storage, including while
 * shared setup is still preparing. No reservation or mutation; 0 fits,
 * -1 invalid/impossible. The request may only allocate after shared commit. */
int bn_generate_batch_memory_check_shared(const BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryQuote quote);
int bn_generate_batch_memory_commit(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryTicket ticket);
int bn_generate_batch_memory_idle(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryTicket ticket);
/* Healthy idle reuse changes ownership generation without allocating or
 * releasing any retained reservation. Rejection leaves *renewed unchanged. */
int bn_generate_batch_memory_reuse(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryTicket ticket,BnGenerateBatchMemoryTicket *renewed);
/* Reserve destructive replacement/growth of an idle slot. Conservatively
 * charge its old retained storage PLUS the complete new preparation peak
 * until commit. Success renews generation and enters preparation; failure
 * preserves the old ticket/storage. On execution failure caller destroys the
 * entire old/new storage before drop; there is no implicit rollback/reuse. */
int bn_generate_batch_memory_replace_idle(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryTicket ticket,BnGenerateBatchMemoryQuote replacement,
    BnGenerateBatchMemoryTicket *renewed);
int bn_generate_batch_memory_drop(BnGenerateBatchMemoryLedger *ledger,
    BnGenerateBatchMemoryTicket ticket);

#endif
