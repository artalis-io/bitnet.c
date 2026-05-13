#ifndef BN_TRANSFORMER_BACKEND_INTERNAL_H
#define BN_TRANSFORMER_BACKEND_INTERNAL_H

#include "backend_model.h"

void *bn_transformer_backend_handle_or(const BnBackendModel *backend,
                                       int layer,
                                       BnBackendHandleRole role);

#endif // BN_TRANSFORMER_BACKEND_INTERNAL_H
