#ifndef BN_MODEL_INTERNAL_H
#define BN_MODEL_INTERNAL_H

#include "model.h"
#include "platform.h"

struct BnModelRuntime {
    BnThreadPool *pool;
    int owns_pool;
    SHArena *weight_arena;
    BnTQState *tq_state;
    int owns_tq_state;
};

struct BnModelIO {
    BnMappedFile file;
    BnMoEIO moe_io;
};

struct BnModelBackendState {
    BnBackendModel *backend;
};

#endif // BN_MODEL_INTERNAL_H
