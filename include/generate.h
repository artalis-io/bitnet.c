#ifndef BN_GENERATE_H
#define BN_GENERATE_H

#include "tokenizer.h"
#include "sampler.h"
#include "bn_alloc.h"

typedef struct BnModel BnModel;
// Forward declaration — full definition in session.h
typedef struct BnSession BnSession;

// Callback for streaming token output. Return non-zero to stop generation.
// EOG tokens are reported with an empty piece before generation stops.
typedef int (*bn_token_callback)(const char *piece, int token_id, void *user_data);

/* Explicit native raw-text execution, with an already prepared backend-only
 * session at position zero. Shared encoder/decoder and model must describe the
 * same checkpoint. No BOS/EOS/template insertion, automatic EOS stopping,
 * sampling penalties or CPU fallback. Greedy ties choose the first ID.
 * Caller serializes backend access and the session, and supplies disjoint workspace;
 * handles, model and borrowed JSON must outlive the call. Encoding scratch and
 * buffers may change on failure. This orchestration allocates nothing itself;
 * backend expert placement and user callbacks retain their own budgets.
 */
typedef struct {
    const BnTokenizerEncoder *encoder;
    const BnTokenizerDecoder *decoder;
    BnTokenizerEncodeScratch *scratch;
    int *tokens;
    size_t token_capacity;
    unsigned char *piece;
    size_t piece_capacity; /* At least six bytes for any pending UTF8 flush. */
} BnGenerateNativeTextWorkspace;
typedef struct {
    int prefill_tokens; /* Explicit maximum batch size; geometry affects math. */
    int max_new_tokens;
    int admit_decode,skip_special;
} BnGenerateNativeTextOptions;
/* One counted UTF8 chunk per generated token, possibly empty. Commitment
 * timing depends on the entry point below.
 * A final flush uses token_id=-1, including after callback-requested stop.
 * Embedded NUL is supported; chunks are not NUL-terminated. Nonzero stops
 * generation after that token; the final flush is still delivered. */
typedef int (*bn_token_bytes_callback)(const unsigned char *piece,size_t bytes,
    int token_id,void *user_data);
/* Optional read-only completion observer. Borrowed inputs/last-row logits are
 * valid only during the callback. Callbacks must not mutate model, session,
 * tokenizer handles or workspace. */
typedef void (*bn_native_text_forward_callback)(const int *tokens,int count,
    int position,const float *last_logits,int vocabulary,void *user_data);
typedef struct {
    bn_token_bytes_callback token;
    bn_native_text_forward_callback forward;
    void *user_data;
} BnGenerateNativeTextCallbacks;
/* Caller-owned resumable native request. Zero-initialize before init; do not
 * copy or change members while bound. Request storage must be disjoint from
 * model/session/tokenizer/workspace storage. generated is the published token count;
 * all other members are private. No allocation in init/step/cancel/free.
 * The model, prepared session, tokenizer handles and workspace buffers must
 * outlive the request. Each live request needs its own session and workspace.
 * Caller serializes backend access. Do not use/reset/reprepare a bound session
 * externally, mutate workspace, or reenter request operations from callbacks.
 * Only one backend forward attempt is made per step, with no internal BUSY
 * spin. A successful prefill step commits at most prefill_tokens prompt IDs;
 * a decode step commits at most one ID. BUSY preserves staged input, admission,
 * pending UTF8 and publication state. PROGRESS/BUSY may be scheduled again.
 * Terminal step calls return the same status without callbacks/backend work.
 * DONE flushes once. ERROR and CANCELLED never flush: output may already be
 * visible, and the session must be reset/reprepared before reuse. Cancellation
 * is local, does not reset the session or reclaim backend allocations/leases;
 * session reset/free owns that cleanup. free cancels active work and unbinds.
 * predict=0 follows committed-token semantics; predict=1 follows the existing
 * prediction-first semantics, including omission of the final token forward.
 */
typedef enum {
    BN_NATIVE_TEXT_ERROR=-2,
    BN_NATIVE_TEXT_DONE=0,
    BN_NATIVE_TEXT_PROGRESS=1,
    BN_NATIVE_TEXT_BUSY=2,
    BN_NATIVE_TEXT_CANCELLED=3
} BnGenerateNativeTextStatus;
typedef struct {
    const void *owner;
    const BnModel *model;
    BnSession *session;
    BnGenerateNativeTextWorkspace workspace;
    BnGenerateNativeTextOptions options;
    BnGenerateNativeTextCallbacks callbacks;
    BnTokenizerDecodeStream stream,pending;
    size_t prompt_tokens,prefilled,piece_bytes;
    int generated,next,position,rows,phase,predict,stepping;
    BnGenerateNativeTextStatus status;
} BnGenerateNativeTextRequest;
/* init returns 0 or -2, encodes once and performs no forward. Rejection leaves
 * an unbound request unchanged; encoding scratch/token buffers may change.
 * The counted prompt itself need not outlive a successful init. */
int bn_generate_native_text_request_init(BnGenerateNativeTextRequest *request,
    const BnModel *model,BnSession *session,
    const BnGenerateNativeTextWorkspace *workspace,
    const unsigned char *text,size_t bytes,
    const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks,int predict);
BnGenerateNativeTextStatus bn_generate_native_text_request_step(BnGenerateNativeTextRequest *request);
BnGenerateNativeTextStatus bn_generate_native_text_request_cancel(BnGenerateNativeTextRequest *request);
void bn_generate_native_text_request_free(BnGenerateNativeTextRequest *request);

/* Native streaming scheduler. Unlike the encoded-token batch executor, each
 * turn resumes one whole native request forward; it does not merge matrix rows.
 * Caller serializes access, including all backend/callback use. Only info() may
 * reenter. A zero-initialized handle may not be copied after initialization. */
#define BN_NATIVE_TEXT_SCHEDULE_QUEUE 64
#define BN_NATIVE_TEXT_SCHEDULE_LIVE 16
typedef struct {void *state;} BnGenerateNativeTextSchedule;
typedef struct {
    int max_live;
    size_t device_budget,host_budget,pinned_budget;
} BnGenerateNativeTextScheduleLimits;
typedef struct {
    int max_tokens,capacity;
    size_t device_bytes,host_bytes,pinned_bytes;
} BnGenerateNativeTextCharge;
typedef struct {
    BnSession *session;
    BnGenerateNativeTextWorkspace workspace;
} BnGenerateNativeTextLease;
typedef struct {
    void *context;
    /* Plan before allocation. Quote includes request KV/workspace payload, not
     * shared model/expert caches or allocator overhead. 0 quotes, 1 temporarily
     * unavailable, -1 rejects this request. Nonzero leaves *out zero. */
    int (*plan)(void *context,uint64_t id,size_t prompt_bytes,
        const BnGenerateNativeTextOptions *options,int predict,BnGenerateNativeTextCharge *out);
    /* 0 supplies a distinct prepared session and disjoint workspace within the
     * exact quote. 1 temporarily exhausted, -1 failed; nonzero leaves *out zero.
     * Session position/history must be zero. Model/encoder/decoder must match. */
    int (*acquire)(void *context,uint64_t id,const BnGenerateNativeTextCharge *quote,
        BnGenerateNativeTextLease *out);
    /* Infallible, fence/reclaim before returning. discard=1 requires resetting
     * or destroying partially executed state, even if it looks healthy.
     * Charged payload is released only after this callback returns. Any retained
     * idle pool allocations need a separate explicit backend budget. */
    void (*release)(void *context,BnGenerateNativeTextLease lease,int discard);
    /* Finite nonnegative monotonic milliseconds. No callback at init. */
    double (*time_ms)(void *context);
} BnGenerateNativeTextScheduleBackend;
typedef enum {
    BN_NATIVE_TEXT_ADMITTED,BN_NATIVE_TEXT_COMPLETED,
    BN_NATIVE_TEXT_REQUEST_CANCELLED,BN_NATIVE_TEXT_REQUEST_FAILED
} BnGenerateNativeTextEventKind;
typedef struct {
    BnGenerateNativeTextEventKind kind;
    uint64_t id;
    int generated;
    double submitted_ms,admitted_ms,first_token_ms,first_byte_ms,timestamp_ms;
} BnGenerateNativeTextEvent;
typedef void (*bn_native_text_event_callback)(const BnGenerateNativeTextEvent *event,void *context);
typedef struct {
    int queued,live,peak_live,poisoned;
    size_t device_bytes,host_bytes,pinned_bytes;
    size_t peak_device_bytes,peak_host_bytes,peak_pinned_bytes;
    uint64_t submitted,admitted,completed,cancelled,failed,acquired,released;
    uint64_t turns,busy_returns,forwards,output_tokens;
} BnGenerateNativeTextScheduleInfo;
typedef struct {
    int attempted;
    uint64_t id;
    BnGenerateNativeTextStatus status;
} BnGenerateNativeTextScheduleStep;
/* One fixed host allocation at init; queue/step/cancel allocate nothing in the
 * orchestration. Backend admission and user callbacks retain their budgets.
 * Limits account live quoted payload only: they are not total physical VRAM/RSS
 * limits. The immutable model and all callback contexts outlive the schedule. */
int bn_generate_native_text_schedule_init(BnGenerateNativeTextSchedule *schedule,
    const BnModel *model,const BnGenerateNativeTextScheduleLimits *limits,
    const BnGenerateNativeTextScheduleBackend *backend,
    bn_native_text_event_callback notify,void *context);
/* Borrow counted prompt bytes and user callback context until terminal event or
 * free. Options/callback descriptors are copied. IDs are unique while pending.
 * Queue capacity includes live requests. 0 accepted, 1 full, -1 invalid/poisoned.
 * Invalid UTF8 or unsupported prepared geometry fails at admission, with cleanup.
 * submitted_ms is an externally recorded arrival in the backend clock domain;
 * it must be finite, nonnegative and <=now. FIFO remains enqueue order. */
int bn_generate_native_text_schedule_enqueue_at(BnGenerateNativeTextSchedule *schedule,
    uint64_t id,const unsigned char *text,size_t bytes,const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks,int predict,double submitted_ms);
int bn_generate_native_text_schedule_enqueue(BnGenerateNativeTextSchedule *schedule,
    uint64_t id,const unsigned char *text,size_t bytes,const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks,int predict);
/* Admit FIFO within all budgets, then resume one live request, round robin even
 * after BUSY. One request keeps its configured prefill chunk; no wall-time bound
 * or kernel preemption. Permanent oversize/admission/forward errors fail only
 * that request; invalid backend ownership or clock failure poisons the runtime,
 * discards all leases and fails all pending requests. Published prefixes remain.
 * Return 1 for work/events, 0 idle/temporarily blocked, -1 invalid/poisoned.
 * Optional receipt is disjoint caller storage and identifies the attempted
 * request. Terminal notifications follow fenced release and budget reclamation.
 * First-token time includes empty chunks; first-byte time requires nonempty
 * output, including final flush. Missing timestamps are -1. */
int bn_generate_native_text_schedule_step(BnGenerateNativeTextSchedule *schedule,
    BnGenerateNativeTextScheduleStep *receipt);
/* Cancel queued/live work between turns. Live state is discarded/fenced before
 * returning and before its budget becomes available. No UTF8 final flush. */
int bn_generate_native_text_schedule_cancel(BnGenerateNativeTextSchedule *schedule,uint64_t id);
int bn_generate_native_text_schedule_info(const BnGenerateNativeTextSchedule *schedule,
    BnGenerateNativeTextScheduleInfo *out);
/* Discard every live lease, release borrowed prompts, no notifications. */
void bn_generate_native_text_schedule_free(BnGenerateNativeTextSchedule *schedule);

/* Encode the entire counted valid UTF8 prompt once, prefill explicit batches,
 * then forward each selected token before publishing its output. BUSY retries
 * retain the same staged history/admission policy. No reference IDs are used.
 * Returns generated count or -2. A failure may follow committed batches/tokens;
 * no callback is issued for a failed forward. Reset/reprepare the session before
 * retrying. Output capacity rejection precedes staging that generated token.
 * The prepared backend must have sufficient batch/context capacity. */
int bn_generate_native_text(const BnModel *model,BnSession *session,
    const BnGenerateNativeTextWorkspace *workspace,
    const unsigned char *text,size_t bytes,
    const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks);

/* Prediction-first counterpart with the same workspace, validation, greedy
 * policy and return values. Publish each selected token before staging its
 * forward. The final token, including a callback-stopped token, is not
 * forwarded. Successful history ends at prompt_tokens + max(generated-1,0);
 * logits still predict the final reported token. This is not a resumable
 * generation API. Capacity must cover prompt_tokens + max(max_new_tokens-1,0).
 * A later forward failure may follow already published output; returns -2
 * without a final flush. BUSY retries never republish a token. Reset/reprepare
 * before retrying; output is not atomic. No callback implies the same execution
 * and history contract. The existing committed-token entry point is unchanged. */
int bn_generate_native_text_predict(const BnModel *model,BnSession *session,
    const BnGenerateNativeTextWorkspace *workspace,
    const unsigned char *text,size_t bytes,
    const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks);

// Chat template format
typedef enum {
    BN_CHAT_AUTO,    // auto-detect from tokenizer (ChatML if im_start/im_end present, else LLaMA)
    BN_CHAT_CHATML,  // <|im_start|>role\n{content}<|im_end|>\n
    BN_CHAT_LLAMA,   // Role: {content}<|eot_id|>
    BN_CHAT_RAW,     // no wrapping — encode content directly (caller handles template)
} BnChatFormat;

// Chat message roles
typedef enum {
    BN_ROLE_SYSTEM,
    BN_ROLE_USER,
    BN_ROLE_ASSISTANT,
} BnChatRole;

// A single chat message (role + content).
typedef struct {
    BnChatRole role;
    const char *content;
} BnChatMessage;

// Stop string configuration for generation.
typedef struct {
    const char **strings;  // array of stop strings (NULL-terminated content, not token IDs)
    int n;                 // number of stop strings
} BnStopStrings;

// Generate tokens autoregressively from pre-computed logits.
// The model must have logits ready (from bn_prefill or bn_transformer_forward).
// pos is updated to reflect the new position after generation.
// stop: optional stop strings (NULL to disable). Generation halts when any
//       stop string appears in the output. The stop string is NOT included
//       in the callback output.
// alloc is used for internal scratch buffers (NULL = stdlib default).
// Returns: number of tokens generated, -1 on loop detected, -2 on error,
//          -3 on stop string match.
int bn_generate(BnModel *model, BnSession *s, BnTokenizer *tok, BnSampler *sampler,
                int max_tokens, int *pos,
                bn_token_callback cb, void *user_data,
                const BnStopStrings *stop,
                BnAllocator *alloc);

// Speculative decoding: draft K tokens with small model, verify with target.
// Both models must have logits ready from prefill. Greedy only (temperature=0).
// alloc is used for verify_logits buffer (NULL = stdlib default).
// pos is updated. Returns: n_generated, -1 on loop, -2 on error.
int bn_generate_speculative(BnModel *target, BnSession *ts,
                            BnModel *draft, BnSession *ds, int draft_k,
                            BnTokenizer *tok, BnSampler *sampler,
                            int max_tokens, int *pos,
                            bn_token_callback cb, void *user_data,
                            BnAllocator *alloc);

// Prefill prompt tokens through the model. Returns logits for the last token,
// or NULL on error. pos is set to pos0 + n_tokens after return.
// If no_prefill is set, runs tokens one at a time (for debugging).
float *bn_prefill(BnModel *model, BnSession *s, const int *tokens, int n_tokens,
                  int pos0, int no_prefill);
int bn_prefill_no_logits(BnModel *model, BnSession *s, const int *tokens,
                         int n_tokens, int pos0, int no_prefill);

// Encode text into tokens. Returns number of tokens written.
// alloc is used for scratch buffer (NULL = stdlib default).
int bn_count_tokens(const BnTokenizer *tok, const char *text,
                    BnAllocator *alloc);

// Format a single user message into a chat turn (legacy convenience wrapper).
// fmt=BN_CHAT_AUTO uses tokenizer's detected format.
// alloc is used for message formatting buffer (NULL = stdlib default).
// Writes encoded tokens into out_tokens[0..max_tokens-1].
// Returns number of tokens written.
int bn_chat_format_turn(const BnTokenizer *tok, BnChatFormat fmt,
                        const char *user_msg,
                        int *out_tokens, int max_tokens,
                        BnAllocator *alloc);

// Format a multi-turn conversation into tokens.
// Encodes all messages in order, appends assistant prompt at the end.
// fmt=BN_CHAT_AUTO uses tokenizer's detected format.
// alloc is used for formatting buffers (NULL = stdlib default).
// Writes encoded tokens into out_tokens[0..max_tokens-1].
// Returns number of tokens written.
int bn_chat_format_messages(const BnTokenizer *tok, BnChatFormat fmt,
                            const BnChatMessage *messages, int n_messages,
                            int *out_tokens, int max_tokens,
                            BnAllocator *alloc);

// Return the end-of-turn token ID for the given format.
// Used to feed into KV cache after assistant response completes.
// Returns -1 if no end-of-turn token exists for the format.
int bn_chat_turn_end_id(const BnTokenizer *tok, BnChatFormat fmt);

// --- Logprobs API ---

#define BN_LOGPROBS_MAX_TOP_K 20

// A single token's log probability entry.
typedef struct {
    int   token_id;
    float logprob;    // natural log probability (ln)
    const char *text; // decoded token text (pointer into tokenizer vocab, not owned)
} BnLogprobEntry;

// Logprobs result for one generated token.
typedef struct {
    BnLogprobEntry chosen;                        // the sampled token
    BnLogprobEntry top[BN_LOGPROBS_MAX_TOP_K];    // top-K alternatives (sorted by logprob, descending)
    int top_k;                                     // number of valid entries in top[]
} BnLogprobs;

// Compute logprobs from raw logits.
// logits: [vocab_size] raw logits from the forward pass.
// chosen_token: the token that was sampled (its logprob is returned in result->chosen).
// top_k: number of top alternatives to return (clamped to BN_LOGPROBS_MAX_TOP_K).
//        Pass 0 to only compute the chosen token's logprob.
// tok: tokenizer for decoding token text (may be NULL, text fields will be NULL).
// result: output struct, filled by this function.
void bn_logprobs_compute(const float *logits, int vocab_size,
                         int chosen_token, int top_k,
                         const BnTokenizer *tok,
                         BnLogprobs *result);

// --- SSE streaming format (OpenAI-compatible) ---

// Format a token piece into an OpenAI-compatible SSE "data:" line.
// Writes to buf (including trailing \n\n). Returns bytes written (excluding NUL).
// piece: token text (NULL for finish chunks with empty delta).
// id: request ID string (NULL -> "chatcmpl-0").
// model: model name (NULL -> "bitnet").
// finish_reason: NULL for normal chunks, "stop"/"length" for final chunk.
// created: unix timestamp (0 to omit from output).
// Returns -1 if buf_size is insufficient.
int bn_format_sse_chunk(char *buf, int buf_size,
                        const char *piece, const char *id,
                        const char *model, const char *finish_reason,
                        long long created);

// Format the SSE stream terminator: "data: [DONE]\n\n"
// Returns bytes written (excluding NUL), or -1 on insufficient buffer.
int bn_format_sse_done(char *buf, int buf_size);

#endif // BN_GENERATE_H
