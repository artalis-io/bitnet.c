#ifndef BN_SESSION_INTERNAL_H
#define BN_SESSION_INTERNAL_H

#include "session.h"

typedef struct BnBackendSession BnBackendSession;

struct BnSessionBackendState {
    BnBackendSession *backend;
};

BnBackendSession *bn_session_backend(const BnSession *session);

#endif // BN_SESSION_INTERNAL_H
