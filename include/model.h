#ifndef BN_MODEL_H
#define BN_MODEL_H

#include "model_config.h"
#include "model_weights.h"
#include "gguf.h"

typedef struct BnGPUBackend BnGPUBackend;
typedef struct BnModelRuntime BnModelRuntime;
typedef struct BnModelIO BnModelIO;
typedef struct BnModelBackendState BnModelBackendState;

typedef struct BnModel {
    BnConfig config;
    BnWeights weights;
    BnModelRuntime *runtime;
    BnModelIO *io;
    BnModelBackendState *backend_state;
} BnModel;

int  bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len, int kv_f16, int kv_tq_bits);
void bn_model_free(BnModel *m);
void bn_model_embed_token(const BnModel *m, float *out, int token);
int bn_model_uses_moe(const BnModel *model);

// Upload all model weights to backend-owned GPU buffers.
// Returns 0 on success. On failure, releases partially uploaded buffers.
int bn_model_upload_weights(BnModel *model, BnGPUBackend *gpu);

// Release all GPU weight buffers. Safe to call if gpu is NULL.
void bn_model_release_gpu(BnModel *model);

#endif // BN_MODEL_H
