#ifndef BN_MODEL_H
#define BN_MODEL_H

#include "model_config.h"
#include "model_weights.h"
#include "model_run_state.h"
#include "platform.h"
#include "gguf.h"
#include "sh_arena.h"
#include "moe_types.h"

typedef struct BnBackendModel BnBackendModel;
typedef struct BnGPUBackend BnGPUBackend;
typedef struct BnThreadPool BnThreadPool;
typedef struct BnTQState BnTQState;
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
void bn_model_set_file(BnModel *model, BnMappedFile file);
BnThreadPool *bn_model_pool(const BnModel *model);
void bn_model_set_thread_pool(BnModel *model, BnThreadPool *pool, int owned);
SHArena *bn_model_weight_arena(const BnModel *model);
BnBackendModel *bn_model_backend(const BnModel *model);
int bn_model_ensure_backend(BnModel *model);
BnTQState *bn_model_tq_state(const BnModel *model);
void bn_model_set_tq_state(BnModel *model, BnTQState *state, int owned);
int bn_model_has_tq(const BnModel *model);
BnMoEIO *bn_model_moe_io(BnModel *model);
const BnMoEIO *bn_model_moe_io_const(const BnModel *model);
void bn_model_set_moe_mmap_base(BnModel *model, const uint8_t *base);
void bn_model_set_moe_fd(BnModel *model, int fd);
void bn_model_set_moe_madvise(BnModel *model, int enabled);
void bn_model_set_moe_cache(BnModel *model, void *cache);
void *bn_model_moe_cache(const BnModel *model);
void bn_model_set_gpu_moe_cache(BnModel *model, void *cache);
void *bn_model_gpu_moe_cache(const BnModel *model);
BnGPUBackend *bn_model_gpu(const BnModel *model);
void bn_model_set_gpu_disabled(BnModel *model, int disabled);

// Upload all model weights to backend-owned GPU buffers.
// Returns 0 on success. On failure, releases partially uploaded buffers.
int bn_model_upload_weights(BnModel *model, BnGPUBackend *gpu);

// Release all GPU weight buffers. Safe to call if gpu is NULL.
void bn_model_release_gpu(BnModel *model);

// Session arena helpers (used by bn_session_create)
size_t bn_model_session_arena_size(const BnConfig *c, const BnWeights *w);
int    bn_model_alloc_session_buffers(const BnConfig *c, const BnWeights *w,
                                       SHArena *arena,
                                       BnRunState *state, BnMoEState **moe_out);

#endif // BN_MODEL_H
