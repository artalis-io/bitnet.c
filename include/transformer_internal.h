#ifndef BN_TRANSFORMER_INTERNAL_H
#define BN_TRANSFORMER_INTERNAL_H

// Internal header for top-level transformer orchestration.
// Not part of the public API.

#include "transformer.h"
#include "model.h"
#include "gpu_backend.h"

float *bn_transformer_forward_logits(BnModel *m, BnSession *sess);
float *bn_transformer_gpu_forward(BnModel *m,
                                  BnSession *sess,
                                  int token,
                                  int pos);
float *bn_transformer_gpu_forward_no_logits(BnModel *m,
                                            BnSession *sess,
                                            int token,
                                            int pos);
int bn_transformer_gpu_forward_argmax(BnModel *m,
                                      BnSession *sess,
                                      int token,
                                      int pos,
                                      const int *penalty_tokens,
                                      int n_penalty_tokens,
                                      float repeat_penalty,
                                      int *out_token);
int bn_transformer_gpu_upload_kv_cache(BnModel *m, BnSession *sess,
                                       int pos0, int n_tokens);
int bn_transformer_gpu_upload_ssm_state(BnModel *m, BnSession *sess);
void bn_transformer_gpu_mark_decode_ssm_pending(BnModel *m, BnSession *sess);
void bn_transformer_gpu_mark_host_ssm_current(BnSession *sess);
int bn_transformer_gpu_prefill_ssm_readback_backend_run(BnModel *m,
                                                       BnSession *sess);

/* Explicit learned-router FFN component, not default full-model admission.
 * Selects source BC decode for 1..8 tokens and source grouped prefill above 8.
 * Hash routing is not admitted by this learned-router component.
 * Setup attaches one native request to the existing session GPU container.
 * Model/backend/sources must outlive the session and all consumers. Access is
 * serialized, including expert-store access across sessions. */
typedef struct {
    int max_tokens, fused_window, f16acc_enabled;
    size_t device_budget, host_budget;
} BnTransformerTrellisOptions;
typedef struct {
    size_t device_bytes, host_bytes;
    int batch_cap, slot_rows, group_rows;
} BnTransformerTrellisWorkspace;
/* Conservative payload bound, excluding driver/library/allocator overhead,
 * immutable model weights and expert-store quotas. Shared backend workspaces
 * are counted independently. No checkpoint payload reads or uploads. Budgets
 * in options are ignored by this query. Source grouping uses 256 MiB. */
int bn_transformer_trellis_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *options, BnTransformerTrellisWorkspace *out);
int bn_transformer_trellis_prepare_session(const BnModel *model, BnSession *session,
    const BnTransformerTrellisOptions *options);
/* Borrow input storage on the native request. Exact F16[tokens,D]. No resize;
 * views expire on reset/free. Caller must order writes before execution and
 * finish all external consumers before reuse. Invalid output arguments and
 * shapes leave outputs unchanged. */
int bn_transformer_trellis_session_input(const BnModel *model, BnSession *session,
    int tokens, void **request, BnGPUResidentView *input);
typedef enum {
    BN_TRANSFORMER_TRELLIS_PREFILL=0,
    BN_TRANSFORMER_TRELLIS_DECODE=1
} BnTransformerTrellisProfile;
/* Decode publishes common router/routed/shared/final views; prefill-only
 * order/count/base/kind/slots views are empty and slot_rows is zero. */
typedef struct {
    void *request;
    int slot_rows;
    BnGPUResidentView logits,indices,weights,order,inverse,sorted_weights;
    BnGPUResidentView counts,starts,slot_base,slot_kind,slots,routed;
    BnGPUResidentView shared_gate,shared_up,shared_act,shared,out;
    BnTransformerTrellisProfile profile;
} BnTransformerTrellisResult;
/* Explicit learned routing. Decode borrows all selected nonzero-weight experts
 * at once (up to 48 triples), reads IDs/weights only, and preserves source BC
 * arithmetic. fused_window controls only prefill; decode does not split its
 * working set into windows. Decode-only setup omits prefill workspaces.
 * Reuses setup buffers; no activation allocation or CPU activation math.
 * Counts are read once for host planning; metadata writes, blocking expert
 * uploads and stream completion at lease boundaries are explicit. Placement admission is delegated to the expert store: cached
 * hits can run without temporary capacity; insufficient cold capacity returns
 * BUSY without changing group geometry. BUSY/failure drains consumers and releases leases;
 * a backend completion failure destroys the request and requires reset/setup.
 * All previous results expire at the next execution attempt. Failure leaves
 * result unchanged; discard previous output after any attempt. Success is
 * complete on return and publishes borrowed views until reuse/reset/free. */
int bn_transformer_trellis_run_learned(const BnModel *model, BnSession *session,
    int layer, int tokens, int admit, BnTransformerTrellisResult *result);

/* Optional F16 compressed/index KV pools on the SAME native session request.
 * Setup is transactional with FFN setup; no upgrade/resize of a live request.
 * maxseq is page-aligned (256 tokens) and <= model context. Each architecture
 * KV owner gets one split paged pool, index-key pool and projected-row rings;
 * reader layers borrow that owner's views. Index-selection ownership remains
 * separate. Device quota covers payload; host quota includes setup scratch.
 * No pool quantization, SWA ring or default full-model admission yet. */
typedef struct { int maxseq; size_t device_budget,host_budget; } BnTransformerTrellisKVOptions;
typedef struct { size_t device_bytes,host_bytes; int owners,pages,page_size; } BnTransformerTrellisKVWorkspace;
int bn_transformer_trellis_kv_session_plan(const BnModel *model,
    const BnTransformerTrellisKVOptions *options,BnTransformerTrellisKVWorkspace *out);
int bn_transformer_trellis_prepare_session_kv(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv);
typedef struct {
    void *request;
    int owner,rate,cursor,entries,entries_per_page,pages;
    BnGPUResidentView block_table,pool_c,pool_r,pool_idx,ring_kv,ring_gate;
} BnTransformerTrellisKVViews;
/* Borrows expire on reset/free or any failed publication attempt that reached
 * backend execution. Pool table is immutable setup metadata; pool/ring writes
 * belong only to the owner producer. Caller orders stage production on this
 * request. Visibility is entries=cursor/rate, never the allocation capacity. */
int bn_transformer_trellis_kv_views(const BnModel *model,BnSession *session,
    int layer,BnTransformerTrellisKVViews *out);
/* Publish already rotated F16 latent/key stage rows. Owner layers only. pos0
 * must append at cursor or rewind <=256 tokens; tokens in [1,max_tokens].
 * Source producer must preserve compressor history on rewind. Emitted rows
 * are floor((pos0+tokens)/rate)-floor(pos0/rate); zero rows require canonical
 * empty input views. Completion precedes the owner-local visibility commit;
 * session's whole-model position is not advanced. No hot allocation/upload or
 * readback; one explicit completion fence. Malformed scalar/count arguments
 * reject without changes. Backend rejection/failure conservatively destroys
 * the request, requires session reset/setup, and invalidates ALL old views.
 * Entries beyond committed visibility remain stale and must not be attended. */
int bn_transformer_trellis_kv_publish(const BnModel *model,BnSession *session,
    int layer,int pos0,int tokens,BnGPUResidentView latents,BnGPUResidentView keys);

/* Optional per-layer sliding-window KV rings on the SAME FFN/KV request.
 * Geometry is source window+two256-token pages, rounded up to256. One ring
 * per layer, one shared move scratch buffer (serialized request execution).
 * Quotas are additional to FFN/KV; setup publishes the entire request only
 * after all stages succeed. Matching setup is idempotent; no live upgrades.
 * No attention arithmetic or whole-model admission in these helpers. */
typedef struct {size_t device_budget,host_budget;} BnTransformerTrellisSWAOptions;
typedef struct {size_t device_bytes,host_bytes;int ring_rows,layers;} BnTransformerTrellisSWAWorkspace;
int bn_transformer_trellis_swa_session_plan(const BnModel *model,
    BnTransformerTrellisSWAWorkspace *out);
int bn_transformer_trellis_prepare_session_attention(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa);
typedef struct {
    void *request;
    int cursor,window_beg,window,ring_rows,n_prev,win_floor;
    BnGPUResidentView ring;
} BnTransformerTrellisSWAViews;
/* Pre-update ring for attention at pos0. Borrow expires on reset/failure or
 * next update of this layer. Rewinds must remain at/after retained window_beg
 * and not exceed cursor. Other recurrent components may impose tighter
 * rollback bounds. Only n_prev rows before pos0 are visible as history. */
int bn_transformer_trellis_swa_views(const BnModel *model,BnSession *session,
    int layer,int pos0,BnTransformerTrellisSWAViews *out);
/* Queue AFTER all consumers of the old ring on this request. Source append,
 * page shift or rebase uses preallocated snapshot scratch; completion precedes
 * commit of this layer's window_beg/cursor. One explicit fence, no hot
 * allocation/upload/readback. Backend rejection/failure or completion failure
 * destroys request and invalidates all views until reset/setup. Scalar/count
 * rejection preserves state. Does not advance session whole-model position. */
int bn_transformer_trellis_swa_commit(const BnModel *model,BnSession *session,
    int layer,int pos0,int tokens,BnGPUResidentView kv);

/* Optional native index state on the same FFN/KV request. Plans additional
 * payload/host bytes, reusing the FFN-reserved F16 projection workspace.
 * Setup is transactional; matching setup idempotent, no live upgrades.
 * Architecture metadata supplies sharing/modes; buffers are request-local. */
typedef struct {size_t device_budget,host_budget;} BnTransformerTrellisIndexOptions;
typedef struct {
    size_t device_bytes,host_bytes;
    int max_tokens,max_slab,max_entries,candidate_entries,score_stride;
} BnTransformerTrellisIndexWorkspace;
int bn_transformer_trellis_index_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    BnTransformerTrellisIndexWorkspace *out);
int bn_transformer_trellis_prepare_session_index(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index);
/* Begin once per full chunk before owner pool publication. Invalid arguments
 * preserve state. Success expires all old index results and candidate masks,
 * even for a replay with identical positions. Does not advance model/KV/SWA
 * positions. Each pool must publish after begin before index selection. */
int bn_transformer_trellis_index_begin(const BnModel *model,BnSession *session,int pos0,int tokens);
typedef enum {BN_TRELLIS_INDEX_SLIDING=0,BN_TRELLIS_INDEX_DENSE,
    BN_TRELLIS_INDEX_TOPK,BN_TRELLIS_INDEX_CANDIDATE_SOURCE,
    BN_TRELLIS_INDEX_CANDIDATE_USE} BnTransformerTrellisIndexMode;
typedef struct {
    void *request;
    uint64_t generation;
    BnTransformerTrellisIndexMode mode;
    int source_layer,kv_owner,entries,rate,k_len,borrowed;
    BnGPUResidentView indices;
    uint64_t pool_generation; /* Owner publication, including dense pools. */
    uint64_t selection_revision; /* Nonzero for produced/borrowed gathered IDs. */
} BnTransformerTrellisIndexResult;
/* Full chunk selection. Sources project once then use source256/final-tail16
 * slabs. Readers borrow the latest matching source selection. Dense <=512
 * and sliding paths skip the query/scoring chain; inputs unused there and on
 * shared readers. Source inputs: finite F16 x[tokens,D], normalized q_res
 * [tokens,q_rank], F32 inv_freq[rope_dim/2], exact/disjoint/request-owned.
 * Query RoPE uses actual positions. No activation allocation/upload/readback
 * inside execution; one completion fence for a producing source, none for
 * dense/sliding/shared paths. Output unchanged on rejection/failure. Results
 * carry a selection_revision and expire on next begin/source execution/reset/failure; pool republication
 * invalidates matching cached results. Backend failure destroys request.
 * This component does not admit a whole native forward path. */
int bn_transformer_trellis_index_select(const BnModel *model,BnSession *session,
    int layer,BnGPUResidentView x,BnGPUResidentView q_res,BnGPUResidentView inv_freq,
    BnTransformerTrellisIndexResult *out);

/* Prepared native attention on the same FFN/KV/SWA/index request. Additional
 * quotas cover attention partials, grouped output, concatenated intermediate
 * and F32 projected delta. Requires prepared core/parameter weights. Setup is
 * transactional; no live upgrade. Does not enable whole-model admission. */
typedef struct {size_t device_budget,host_budget;} BnTransformerTrellisAttentionOptions;
typedef struct {
    size_t device_bytes,host_bytes,ml_count,acc_count;
    int max_tokens;
} BnTransformerTrellisAttentionWorkspace;
int bn_transformer_trellis_attention_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    BnTransformerTrellisAttentionWorkspace *out);
int bn_transformer_trellis_prepare_session_projected_attention(const BnModel *model,
    BnSession *session,const BnTransformerTrellisOptions *ffn,
    const BnTransformerTrellisKVOptions *kv,const BnTransformerTrellisSWAOptions *swa,
    const BnTransformerTrellisIndexOptions *index,const BnTransformerTrellisAttentionOptions *attention);
typedef struct {
    void *request;
    uint64_t generation;
    int layer,pos0,tokens;
    BnGPUResidentView delta;
} BnTransformerTrellisAttentionResult;
/* Consume a current index_select result AFTER index_begin and owner pool
 * publication. Queries F16[tokens,64,512] and KV F16[tokens,512] are already
 * rotated; negative_inv_freq is F32[32]. Inputs exact/disjoint/request-owned.
 * Queue attention against PRE-update ring, grouped projection, then ring
 * update on one stream. One completion fence precedes ring cursor commit.
 * Does not advance whole-model position. One success per layer/begin epoch;
 * rewinds require a new begin and retained recurrent history. No hot
 * allocation/upload/readback. Metadata rejection preserves state/output;
 * any backend rejection/failure destroys the request until reset/setup.
 * Borrowed delta expires on next attention attempt reaching execution,
 * reset/free/failure. Result is unchanged on rejection/failure. */
int bn_transformer_trellis_attention_project(const BnModel *model,BnSession *session,
    int layer,const BnTransformerTrellisIndexResult *selection,
    BnGPUResidentView queries,BnGPUResidentView kv,BnGPUResidentView negative_inv_freq,
    BnTransformerTrellisAttentionResult *out);

/* Source QKV/compressor/key production feeding the prepared attention path.
 * Additional request-local scratch and immutable converted compressor norms.
 * Setup generates immutable main/compressed positive/negative frequency tables.
 * The explicit-table entry point remains available for qualified callers. */
typedef struct {size_t device_budget,host_budget;} BnTransformerTrellisProducerOptions;
/* Additional native-request raw-row banks. Architecture/model adapters own
 * table roles and sources; caches are backend-owned and caller-serialized. */
typedef struct {int max_rows,banks;size_t host_budget,pinned_budget,device_budget;} BnTransformerTrellisRowsOptions;
typedef struct {size_t host_bytes,pinned_bytes,device_bytes;int tables,max_rows,banks;} BnTransformerTrellisRowsWorkspace;
typedef struct {void *request;const BnSession *session;int layer;unsigned bank;uint64_t generation,epoch;BnGPUResidentView values;} BnTransformerTrellisRowsResult;
int bn_transformer_trellis_rows_session_plan(const BnModel *model,
    const BnTransformerTrellisRowsOptions *options,BnTransformerTrellisRowsWorkspace *out);
int bn_transformer_trellis_prepare_session_rows(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisRowsOptions *rows);
int bn_transformer_trellis_prepare_session_producer_rows(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index,
    const BnTransformerTrellisAttentionOptions *attention,const BnTransformerTrellisProducerOptions *producer,
    const BnTransformerTrellisRowsOptions *rows);
/* Raw row IDs, not tokenizer IDs. No token position/KV commit. Busy leaves
 * output unchanged. Backend failure releases the whole native request;
 * preflight/storage/domain rejection may be retried while prior results hold. */
int bn_transformer_trellis_rows_submit(const BnModel *model,BnSession *session,
    int layer,unsigned bank,const uint64_t *ids,size_t count,BnTransformerTrellisRowsResult *out);
int bn_transformer_trellis_rows_current(const BnModel *model,BnSession *session,
    const BnTransformerTrellisRowsResult *result);
/* Additional same-request Engram math and hash scratch. Row banks/caches and
 * host history are independently budgeted. Full batches must fit a row bank;
 * GEMM shape is preserved rather than silently splitting math into row tiles. */
typedef struct {size_t device_budget,host_budget;} BnTransformerTrellisEngramOptions;
typedef struct {size_t device_bytes,host_bytes,history_bytes;int max_tokens,tables,heads;} BnTransformerTrellisEngramWorkspace;
typedef struct {
    void *request;const BnSession *session;int layer,tokens;unsigned bank;
    uint64_t generation,epoch,history_epoch,row_epoch;
    BnGPUResidentView lookup,embedding,key,value,normalized_key,normalized_query,dot,delta,out;
} BnTransformerTrellisEngramResult;
int bn_transformer_trellis_engram_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisRowsOptions *rows,
    BnTransformerTrellisEngramWorkspace *out);
/* Prepare host native history first, with max_tokens matching ffn. Setup is
 * transactional for the canonical GPU request; never changes history/position. */
int bn_transformer_trellis_prepare_session_engram(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisRowsOptions *rows,
    const BnTransformerTrellisEngramOptions *engram);
int bn_transformer_trellis_prepare_session_producer_engram(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index,
    const BnTransformerTrellisAttentionOptions *attention,const BnTransformerTrellisProducerOptions *producer,
    const BnTransformerTrellisRowsOptions *rows,const BnTransformerTrellisEngramOptions *engram);
int bn_transformer_trellis_engram_input(const BnModel *model,BnSession *session,
    int tokens,void **request,BnGPUResidentView *stream);
/* Uses the already-staged session mapped history. Queues two pinned row DMAs,
 * row decode and resident math; no allocation, activation readback or fence.
 * Busy/storage/domain rejection preserves old results (cache fills may remain).
 * Backend rejection/failure releases the whole request until reset/setup.
 * Commit host history ONLY after successful WHOLE-model completion. Borrowed
 * result views are not completion proofs and expire after history finalization,
 * later row-bank/math reuse, reset/free or failure. out must be disjoint from
 * session/model storage. Stream is exact F32[tokens,streams,dim], same request. */
int bn_transformer_trellis_engram_submit(const BnModel *model,BnSession *session,
    int layer,unsigned bank,BnGPUResidentView stream,BnTransformerTrellisEngramResult *out);
int bn_transformer_trellis_engram_current(const BnModel *model,BnSession *session,
    const BnTransformerTrellisEngramResult *result);

typedef struct {size_t device_bytes,host_bytes;int max_tokens,max_rate,owners;} BnTransformerTrellisProducerWorkspace;
int bn_transformer_trellis_producer_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    BnTransformerTrellisProducerWorkspace *out);
int bn_transformer_trellis_prepare_session_producer(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index,
    const BnTransformerTrellisAttentionOptions *attention,const BnTransformerTrellisProducerOptions *producer);
/* Begin the full chunk once via index_begin BEFORE running its layers. x is
 * already-normalized F16[tokens,D]. Query/compress/output tables are F32[32];
 * query and compress may be an exact readonly alias. Output table is negative
 * query frequency. Inputs must outlive queued consumers. Owners produce and
 * publish compressed latents/keys; readers require a fresh matching owner and
 * index source. No hot allocation/transfer. Existing publication/index/attention
 * completion fences remain. Backend failure invalidates the entire request;
 * metadata rejection preserves result. Whole-model position is not advanced.
 * Returned F32 delta follows the prepared-attention borrow lifetime. */
int bn_transformer_trellis_attention_forward(const BnModel *model,BnSession *session,int layer,
    BnGPUResidentView x,BnGPUResidentView query_inv_freq,BnGPUResidentView compress_inv_freq,
    BnGPUResidentView negative_query_inv_freq,BnTransformerTrellisAttentionResult *out);

/* Same producer path using setup-generated model frequency tables. No table
 * allocation, generation, upload or readback during execution. */
int bn_transformer_trellis_attention_forward_native(const BnModel *model,BnSession *session,
    int layer,BnGPUResidentView x,BnTransformerTrellisAttentionResult *out);

/* Connected carried-pre HC4 stage. Explicit ordinary-text component, not a
 * full model executor. All 2*layers sites must be consumed in order. It owns
 * stream/carry/scratch state on the canonical native request; normalization
 * writes the existing FFN input view so attention/FFN can consume it directly.
 * Additional payload budget excludes existing FFN/rows/Engram and shared
 * cuBLAS reserves. No live request upgrades. Host history must be prepared. */
typedef struct {size_t device_budget,host_budget;} BnTransformerTrellisHCOptions;
typedef struct {size_t device_bytes,host_bytes;int max_tokens,layers;} BnTransformerTrellisHCWorkspace;
int bn_transformer_trellis_hc_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *ffn,BnTransformerTrellisHCWorkspace *out);
int bn_transformer_trellis_prepare_session_hc(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisRowsOptions *rows,
    const BnTransformerTrellisEngramOptions *engram,const BnTransformerTrellisHCOptions *hc);
int bn_transformer_trellis_prepare_session_producer_hc(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index,
    const BnTransformerTrellisAttentionOptions *attention,const BnTransformerTrellisProducerOptions *producer,
    const BnTransformerTrellisRowsOptions *rows,const BnTransformerTrellisEngramOptions *engram,
    const BnTransformerTrellisHCOptions *hc);
/* Begin from the CURRENT staged raw IDs: one explicit ID upload and resident
 * BF16 embedding -> four F32 streams. Carry resets once here. No fence, host
 * allocation, history/position commit or implicit restart of an active batch.
 * IDs use an existing row-stage pinned slot; BUSY (1) permits unchanged retry.
 * A later begin requires a new staged append after the previous finish. */
int bn_transformer_trellis_hc_begin(const BnModel *model,BnSession *session);
typedef struct {
    void *request;
    const BnSession *session;
    uint64_t generation,epoch,history_epoch;
    int site,tokens;
    BnGPUResidentView streams,partials,post,comb,collapsed,pre,normalized;
} BnTransformerTrellisHCResult;
/* At each attention site run its optional Engram before mixing. BUSY (1)
 * leaves the site ready for retry. Own pre is published for the NEXT site;
 * collapse uses previous pre, or stream zero initially. F32 collapse is
 * normalized directly into F16. Result is current until apply/reset/failure
 * or history finalization; current does not prove GPU completion. */
int bn_transformer_trellis_hc_mix(const BnModel *model,BnSession *session,
    BnTransformerTrellisHCResult *out);
int bn_transformer_trellis_hc_current(const BnModel *model,BnSession *session,
    const BnTransformerTrellisHCResult *result);
/* Apply the matching external F16/F32 branch [tokens,D], then advance exactly
 * one site. Backend rejection/failure destroys the entire canonical request;
 * host history remains pending and caller must reset/replay backend state.
 * Scalar/ticket rejection queues nothing and preserves the pending site. */
int bn_transformer_trellis_hc_apply(const BnModel *model,BnSession *session,
    const BnTransformerTrellisHCResult *result,BnGPUResidentView branch);
/* After every site: final vendor carried-pre collapse -> F32 -> norm -> F16.
 * Returns borrowed head input and F32 collapsed view, without vocabulary
 * projection, completion fence, history/position commit or native admission.
 * Caller must finish the ENTIRE model and fence before publishing/committing.
 * No allocation, activation readback or internal fence in mix/apply/finish. */
int bn_transformer_trellis_hc_finish(const BnModel *model,BnSession *session,
    BnGPUResidentView *collapsed,BnGPUResidentView *head_input);

/* Actual attention + learned-router FFN block on a prepared producer/HC
 * request. Caller stages history, index_begin and hc_begin once per batch.
 * Blocks must follow checkpoint order. BUSY retains the pending HC site:
 * retry SAME layer/admit; attention already completed is not repeated.
 * All branch failures release the WHOLE canonical request, even when an
 * earlier branch committed local KV/index/ring visibility. Reset/replay is
 * required; history/position remain pending. No vocabulary/head execution or
 * model-position commit. This explicit component is not default admission.
 * Caller serializes access and must not interleave lower-level mutations
 * with an active block. Outputs must be disjoint from model/session state. */
typedef struct {
    void *request;
    const BnSession *session;
    uint64_t generation,epoch,history_epoch,index_generation;
    int layer,tokens,position;
    BnGPUResidentView streams;
} BnTransformerTrellisBlockResult;
int bn_transformer_trellis_block_step(const BnModel *model,BnSession *session,
    int layer,int admit,BnTransformerTrellisBlockResult *out);
/* Borrow expires at the next accepted block attempt, history finalization,
 * HC reuse, reset/free or failure. Current is not a completion proof. */
int bn_transformer_trellis_block_current(const BnModel *model,BnSession *session,
    const BnTransformerTrellisBlockResult *result);

/* Explicit native full text forward. Additional quota for F16 vocabulary
 * logits and one host F16 last-row staging buffer. all_logits=0 reproduces
 * source last_tokens_only=1 (head GEMM M1); 1 projects every staged row.
 * This request choice is fixed at setup; large all-row logits require their
 * full explicit quota. Existing producer/HC/base quotas remain separate. */
typedef struct {int all_logits;size_t device_budget,host_budget;} BnTransformerTrellisForwardOptions;
typedef struct {size_t device_bytes,host_bytes;int max_tokens,vocab,logit_rows;} BnTransformerTrellisForwardWorkspace;
int bn_transformer_trellis_forward_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisForwardOptions *options,
    BnTransformerTrellisForwardWorkspace *out);
int bn_transformer_trellis_prepare_session_forward(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *ffn,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index,
    const BnTransformerTrellisAttentionOptions *attention,const BnTransformerTrellisProducerOptions *producer,
    const BnTransformerTrellisRowsOptions *rows,const BnTransformerTrellisEngramOptions *engram,
    const BnTransformerTrellisHCOptions *hc,const BnTransformerTrellisForwardOptions *forward);
typedef struct {
    void *request;
    const BnSession *session;
    uint64_t generation,epoch;
    int position,tokens,rows,vocab;
    BnGPUResidentView logits; /* F16 source head output, same request. */
    const float *host_logits; /* Published LAST row, exact F16 -> F32. */
} BnTransformerTrellisForwardResult;
/* Consume the CURRENT staged history, start embedding/index/carry once, run
 * EVERY block, final carried head collapse/norm, then original F16 head.
 * BUSY retains current block/batch; retry with same admit policy. No layer,
 * history or position replay on retry. Successful last-row read establishes
 * whole-request completion, then finite published-logit preflight and host
 * history/position commit precede publishing last-row F32 logits. No hot
 * activation allocation. Only final logits and existing routing/error metadata
 * cross to host. Partial execution/head/read/commit failure destroys the
 * entire request, preserves prior host logits and requires reset/replay.
 * Scalar/policy rejection queues nothing. Caller serializes all mutations.
 * Explicit API only: does not enable CLI/native CPU admission or text encoding. */
int bn_transformer_trellis_forward_run(const BnModel *model,BnSession *session,
    int admit,BnTransformerTrellisForwardResult *out);
/* Result expires at next accepted forward attempt or reset/free/failure.
 * Host sampling may modify session logits; ticket validates GPU borrow only. */
int bn_transformer_trellis_forward_current(const BnModel *model,BnSession *session,
    const BnTransformerTrellisForwardResult *result);

#define BN_TRANSFORMER_TRELLIS_GROUP_MAX 16
/* Borrow/launch counters cover the shared path; individual_layers counts
 * fallback layers whose internal work retains the existing profiler. */
typedef struct {
    uint64_t calls,shared_layers,individual_layers,expert_borrows,fused_launches,folded_launches,waits;
    /* Fused counts are actual kernels; folded_launches counts private FFN
     * submissions. Joint avoided launches compare against one kernel/consumer. */
    uint64_t joint_fused_launches, fused_launches_avoided, individual_fused_launches;
    /* Successful next-wave borrows before the current explicit drain, and
     * speculative BUSY stops. These count ordering, not actual GPU overlap. */
    uint64_t lookahead_borrows, lookahead_busy_stops;
    /* Coordinated <=8-token layers and bounded lease cohorts. Decode GPU
     * submissions remain private; reuses count duplicate expert bindings
     * served by one lease within a completed cohort, not avoided kernels. */
    uint64_t decode_layers, decode_cohorts, decode_submissions, decode_expert_reuses;
} BnTransformerTrellisGroupStats;
/* Explicit lockstep full forwards for2..16 distinct prepared sessions of one
 * model. Each owns its staged history, positions, activations and KV. Layers
 * with all batches>8 share expert leases across request-owned tiled FFNs;
 * folded GEMMs preserve each original group/row shape. Mixed/decode layers
 * execute existing FFNs separately and are counted. No activation host bridge,
 * new GPU allocation, concatenated-row GEMM or default-path admission.
 * Success commits ALL histories/logits/results after all heads/fences succeed.
 * Rejection before binding queues nothing. Execution failure discards every
 * participant backend, preserves prior host position/logits, and requires replay.
 * BUSY binds the SAME ordered cohort until retry/success/cancel: no individual
 * forward/reset/release/free or external mutation while bound. Caller serializes
 * operations and keeps model/sessions/output storage alive and disjoint.
 * Results unchanged on failure/BUSY. Optional stats report cumulative work for
 * this transaction, including work repeated after BUSY; they are not timings. */
int bn_transformer_trellis_forward_group(const BnModel *model,BnSession *const *sessions,
    int count,int admit,BnTransformerTrellisForwardResult *results,BnTransformerTrellisGroupStats *stats);
/* Cancel a bound cohort: fence/discard ALL participants before return. No host
 * history/logits commit. Reset/free the sessions before reuse. Invalid ordered
 * membership leaves all state intact. */
int bn_transformer_trellis_forward_group_cancel(const BnModel *model,BnSession *const *sessions,int count);

/* Full native request configuration/aggregate planning. Model anatomy owns row
 * head geometry. Plans include all request execution/KV payload and host native
 * history/arena payload; model/cache, allocator and driver context overhead are
 * separate. No allocation, GPU callbacks or storage reads in request_plan. */
typedef struct {
    int max_tokens,maxseq,fused_window,f16acc_enabled,all_logits;
} BnTransformerTrellisRequestConfig;
typedef struct {
    size_t device_bytes,host_bytes,pinned_bytes,kv_bytes,history_bytes;
} BnTransformerTrellisRequestPlan;
int bn_transformer_trellis_request_plan(const BnModel *model,
    const BnTransformerTrellisRequestConfig *config,BnTransformerTrellisRequestPlan *out);
/* Fresh empty backend-only session. Prepares history and the entire forward
 * path within the aggregate plan. Failure may retain host history; reset/free
 * the session before retrying. No automatic model/cache preparation. */
int bn_transformer_trellis_prepare_request(const BnModel *model,BnSession *session,
    const BnTransformerTrellisRequestConfig *config);
/* Healthy FULL-forward boundary only, including a freshly prepared request.
 * Rejects staged/partial execution without GPU work. Completes request work,
 * clears mutable device state and host history/logits, then commits position0
 * and new ownership generation. Preserves capacity, immutable page table,
 * rotary frequencies, norms, weights, pinned staging and all allocations.
 * Old result/selection/history borrows expire. Saturated epochs reject reuse.
 * No allocation/upload/readback or model-global activation reset. Any failure
 * after device work discards the entire request; host publication is preserved
 * until successful restart. Caller ends generation before calling restart. */
int bn_transformer_trellis_session_restart(const BnModel *model,BnSession *session);

/* Serialized exact-shape full native request pool. Retained idle AND leased
 * payload counts against all three budgets. Model/GPU/storage outlive the pool.
 * Request max_tokens is exact; required capacity rounds up to256, at least
 * max_tokens, capped at configured maxseq. Exact idle shapes are reused; other
 * idle slots may be evicted, never live leases. A feasibility BUSY changes no
 * slot or device state. No allocation/GPU work during pool_init itself.
 * Caller owns a zeroed handle, cannot copy it or reset/free leased sessions.
 * Return each lease once after generation ends. The pool does not own codec
 * workspace or shared model/expert-cache budgets. These are payload bounds,
 * not a total physical VRAM/RSS guarantee. */
typedef struct {void *state;} BnTransformerTrellisPool;
typedef struct {
    int max_slots,allocated_slots,leased_slots;
    size_t device_budget,host_budget,pinned_budget;
    size_t reserved_device,reserved_host,reserved_pinned;
    size_t peak_device,peak_host,peak_pinned;
    uint64_t created,reused,released,discarded,evicted,restart_failures;
    int max_tokens[16],capacity[16],leased[16];
} BnTransformerTrellisPoolInfo;
int bn_transformer_trellis_pool_init(BnTransformerTrellisPool *pool,const BnModel *model,
    const BnTransformerTrellisRequestConfig *ceiling,int max_slots,
    size_t device_budget,size_t host_budget,size_t pinned_budget);
/* 0 success, 1 temporarily exhausted, -1 invalid/permanently oversized or
 * preparation failure. *out=NULL on failure. Preparation failure can reclaim
 * idle entries; accounting remains accurate. Successful lease position is0. */
int bn_transformer_trellis_pool_acquire(BnTransformerTrellisPool *pool,
    int max_tokens,int required_capacity,BnSession **out);
/* discard1 frees/fences partial work; discard0 restarts healthy state then
 * retains it idle. A restart failure discards the slot and increments
 * restart_failures, still returning0 for the valid lease cleanup. */
int bn_transformer_trellis_pool_release(BnTransformerTrellisPool *pool,BnSession *session,int discard);
int bn_transformer_trellis_pool_info(const BnTransformerTrellisPool *pool,BnTransformerTrellisPoolInfo *out);
/* Fence/free all slots, invalidate leases; no generation may still use them. */
void bn_transformer_trellis_pool_free(BnTransformerTrellisPool *pool);


#endif // BN_TRANSFORMER_INTERNAL_H
