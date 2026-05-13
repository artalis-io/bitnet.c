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

typedef struct BnModel {
    BnConfig config;
    BnWeights weights;
    BnMappedFile file;       // keeps mmap/buffer alive
    BnThreadPool *pool;      // thread pool for parallel dispatch
    SHArena *weight_arena;   // arena for weight transforms (INT8 embeddings, Q4_0 repacking)
    // MoE shared I/O (zero for dense models)
    BnMoEIO moe_io;
    int expert_fd;           // file descriptor for expert pread, -1 if unused
    BnBackendModel *backend; // backend-resident model state/control
    BnTQState *tq_state;     // TurboQuant state (NULL = no TQ compression)
} BnModel;

int  bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len, int kv_f16, int kv_tq_bits);
void bn_model_free(BnModel *m);
void bn_model_embed_token(const BnModel *m, float *out, int token);
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
