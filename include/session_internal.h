#ifndef BN_SESSION_INTERNAL_H
#define BN_SESSION_INTERNAL_H

#include "session.h"

typedef struct BnBackendSession BnBackendSession;
typedef struct BnSessionNativeHistory BnSessionNativeHistory;


struct BnSessionBackendState {
    BnBackendSession *backend;
    const BnModel *model;
    int host_vocabulary, host_capacity;
    BnSessionNativeHistory *native_history; /* Optional host transaction storage. */
    uint64_t native_request_generation; /* Survives reset; prevents stale view reuse. */
    const void *native_anatomy; /* Immutable creation identity; never dereferenced. */
};

BnBackendSession *bn_session_backend(const BnSession *session);
/* Creation identity and host-buffer extents; rejects shallow model copies or
 * config mutation before a typed backend request can publish host output. */
int bn_session_matches_model(const BnSession *session, const BnModel *model);

/* Optional native Engram host history. Prepare before execution at position0;
 * budget includes metadata, full committed compressed history, one staged
 * raw batch and its mapped prefix/window. Model map must already be bound.
 * Caller serializes the session and keeps the model immutable/alive. */
typedef struct {
    const uint32_t *mapped; /* Borrowed, read-only; invalid after commit/abort/reset/free. */
    size_t length;
    int prefix, tokens, position;
    uint64_t epoch;
    size_t host_bytes;
    const BnSession *owner;
    const int *raw; /* Staged original IDs[tokens], same borrow lifetime. */
} BnSessionNativeHistoryView;
typedef struct {int max_tokens,capacity,committed,staged;size_t host_bytes;} BnSessionNativeHistoryInfo;
int bn_session_native_history_info(const BnSession *session,const BnModel *model,BnSessionNativeHistoryInfo *out);
int bn_session_native_history_plan(const BnModel *model,int max_tokens,size_t *bytes);
int bn_session_native_history_prepare(BnSession *session,const BnModel *model,int max_tokens,size_t budget);
/* Stage maps/copies raw tokens without changing position/committed history.
 * All layers must consume the same window before finalization. No allocation.
 * Input must be disjoint from this history's storage. Failure preserves state. */
int bn_session_native_history_stage(BnSession *session,const BnModel *model,const int *tokens,int count);
/* out must be disjoint from all session/model storage. */
int bn_session_native_history_view(const BnSession *session,const BnModel *model,BnSessionNativeHistoryView *out);
/* Executor calls commit only after successful completion of the WHOLE batch;
 * it copies raw history and advances position. Abort only discards staged host
 * history: it does not roll back queued GPU work, KV or recurrent state. */
int bn_session_native_history_commit(BnSession *session,const BnModel *model,const BnSessionNativeHistoryView *view);
/* Atomic host publication for1..16 distinct sessions of one immutable model.
 * All tickets validate before any copy/position change; rejection preserves all
 * histories. Executor fences every participant first. No GPU work/allocation. */
int bn_session_native_history_commit_group(const BnModel *model,BnSession *const *sessions,
    const BnSessionNativeHistoryView *views,int count);
int bn_session_native_history_abort(BnSession *session,const BnModel *model,const BnSessionNativeHistoryView *view);
/* History bookkeeping AFTER executor has restored backend compute state and
 * session->pos to position. No active transaction; cannot extend history.
 * This function itself does not restore any backend/KV/recurrent state. */
int bn_session_native_history_rewind(BnSession *session,const BnModel *model,int position);

/* Executor-only host commit after successful fenced backend restart. Requires
 * no staged transaction and matching committed/session position. No GPU work
 * or allocation; clears history and commits position0, preserving storage. */
int bn_session_native_history_restart(BnSession *session,const BnModel *model);

#endif // BN_SESSION_INTERNAL_H
