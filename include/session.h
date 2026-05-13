#ifndef BN_SESSION_H
#define BN_SESSION_H

#include "model_run_state.h"
#include "moe_types.h"
#include "sh_arena.h"
#include "bn_alloc.h"

typedef struct BnModel BnModel;
typedef struct BnBackendSession BnBackendSession;

// Per-request mutable state. Multiple sessions can share one BnModel.
struct BnSession {
    BnRunState state;          // activation buffers + KV cache (arena-allocated)
    BnMoEState *moe_state;     // MoE compute buffers (NULL for dense models)
    SHArena *arena;            // owns all session buffer memory
    BnBackendSession *backend; // per-request backend state
    int pos;                   // generation position
};
typedef struct BnSession BnSession;

// Create a session with its own KV cache and activation buffers.
// alloc: allocator for the session struct itself (NULL = stdlib default).
BnSession *bn_session_create(const BnModel *model, BnAllocator *alloc);

// Free session memory.
void bn_session_free(BnSession *s, BnAllocator *alloc);

// Reset session: clear KV cache, SSM state, reset pos to 0.
void bn_session_reset(BnSession *s, const BnModel *model);

#endif // BN_SESSION_H
