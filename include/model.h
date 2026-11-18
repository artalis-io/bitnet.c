#ifndef BN_MODEL_H
#define BN_MODEL_H

#include "model_config.h"
#include "model_anatomy.h"
#include "model_weights.h"
#include "gguf.h"
#include "trellis.h"
#include "json.h"
#include "model_tokenizer_policy.h"

typedef struct BnGPUBackend BnGPUBackend;
typedef struct BnModelRuntime BnModelRuntime;
typedef struct BnModelIO BnModelIO;
typedef struct BnModelBackendState BnModelBackendState;
typedef struct BnCPURuntimePolicy BnCPURuntimePolicy;
typedef struct BnBackendRowCache BnBackendRowCache;

typedef struct BnModel {
    BnConfig config;
    BnWeights weights;
    BnModelRuntime *runtime;
    BnModelIO *io;
    BnModelBackendState *backend_state;
} BnModel;

int  bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len, int kv_f16, int kv_tq_bits);
int  bn_model_load_with_cpu_preparation(BnModel *m, BnGGUFFile *f,
                                        int max_seq_len, int kv_f16,
                                        int kv_tq_bits,
                                        int prepare_cpu_weights);
/* Bind architecture-planned topology and quantized expert source descriptors.
 * model must be empty. Failure leaves it unchanged. file is borrowed and must
 * outlive the model and its backend requests. No tensor payloads are copied and
 * no legacy CPU weights or request state are allocated. Use a backend session;
 * this binding alone does not admit ordinary generation or CPU execution. */
int bn_model_load_topology(BnModel *model, BnGGUFFile *file, int max_seq_len);
const BnModelAnatomy *bn_model_anatomy(const BnModel *model);
/* Bind immutable native routed-expert storage during model setup, before any
 * readers, caches, backend requests or sessions are published. Counts and
 * dimensions come from validated model config; this does not admit a topology
 * or executor. The index and its immutable files must outlive the model.
 * All sources are validated before publication; failure leaves model unchanged.
 * A model may bind once. No expert bodies are loaded (markers are read).
 * Reader/cache leases must expire before model free. Cache ownership stays
 * with the caller; destroy such caches before freeing the model. */
int bn_model_bind_trellis_experts(BnModel *model,
    const BnSafetensorsIndex *index, const char *architecture,
    int n_layers, int n_experts, int dim, int hidden);
/* Config-driven expert-storage load into an empty model. Owns a parsed copy
 * of config; borrows index/files. Atomic failure. Does not admit full topology
 * or generation, and does not populate legacy BnConfig/weights. */
int bn_model_load_trellis_config(BnModel *model, const BnSafetensorsIndex *index,
    const void *config_json, size_t config_bytes);
const BnJSON *bn_model_native_config(const BnModel *model);
const BnModelNativeExpertConfig *bn_model_native_expert_config(const BnModel *model);
/* Optional setup stage after config/storage load and before backend setup or
 * publication. Model owns plan. No full numeric/executor admission. */
int bn_model_prepare_native_anatomy(BnModel *model);
const BnModelNativeAnatomy *bn_model_native_anatomy(const BnModel *model);
/* Optional, one-shot setup after anatomy and before backend publication. */
int bn_model_prepare_native_numerics(BnModel *model);
const BnModelNativeNumerics *bn_model_native_numerics(const BnModel *model);
/* One-shot immutable row binding after anatomy/numerics, before backend setup.
 * Borrows the model index; no table payload read or generation admission. */
int bn_model_prepare_native_rows(BnModel *model);
const BnTrellisRowSource *bn_model_native_row_source(const BnModel *model,int layer);
/* Mutable cache allocation belongs to the existing backend owner. Prepare
 * after core/backend creation and before publishing sessions; shared access
 * is caller-serialized. Cache payload and metadata fit host_budget, using a
 * conservative per-table budget bound. Failure leaves published state intact. */
int bn_model_prepare_native_row_caches(BnModel *model,size_t per_table_budget,
    int max_rows,size_t host_budget);
BnBackendRowCache *bn_model_native_row_cache(const BnModel *model,int layer);
typedef struct {
    int tables,max_rows;
    size_t per_table_budget,host_bound,host_allocated;
} BnModelNativeRowsInfo;
int bn_model_native_row_cache_info(const BnModel *model,BnModelNativeRowsInfo *out);
/* Setup-only immutable source hash metadata; no token map or execution admission. */
int bn_model_native_engram_layout(const BnModel *model,int layer,BnModelNativeEngramLayout *out);
/* Stateless hash of already compressed IDs using a resolved immutable layout.
 * Inputs are batch x history_len; output is batch x out_len x layout.heads.
 * Prefix may be shorter than ngram-1: missing context is mapped-pad filled.
 * Pad ends a gram and resets the following token. No allocation/history or
 * position commit. Full preflight preserves output on rejection; input/output
 * and layout/output must be disjoint. out_count is exact. For out_len=0,
 * out_count=0 and NULL data are permitted (layout/shape still validated). */
int bn_model_native_engram_hash_rows(const BnModelNativeEngramLayout *layout,uint32_t mapped_pad,
    const uint32_t *mapped_history,size_t batch,size_t history_len,size_t out_len,
    uint64_t *out,size_t out_count);
/* Key policy is architecture-owned; downstream tokenizers build the map. */
int bn_model_native_engram_math(const BnModel *model,int layer,BnModelNativeEngramMath *out);
int bn_model_native_head(const BnModel *model,BnModelNativeHead *out);
int bn_model_native_engram_map_spec(const BnModel *model,BnTokenizerMapSpec *out);
/* Copy an immutable first-seen-ordered compressed map before backend setup and
 * publication. Budget covers the uint32 vector; existing model metadata and
 * allocator overhead are separate. Rejection leaves the model unchanged. */
int bn_model_prepare_native_token_map(BnModel *model,const uint32_t *ids,size_t count,size_t host_budget);
const uint32_t *bn_model_native_token_map(const BnModel *model,size_t *count);
typedef struct {int vocab_size,compressed_vocab;uint32_t mapped_pad;size_t host_bytes;} BnModelNativeTokenMapInfo;
int bn_model_native_token_map_info(const BnModel *model,BnModelNativeTokenMapInfo *out);
/* Allocation-free raw-to-compressed batch mapping; no identity fallback or
 * position/history commit. Preflight rejects every invalid ID before writes.
 * Output must be disjoint from input and model storage. Empty count permits
 * NULL data only when a map is bound. */
int bn_model_native_map_tokens(const BnModel *model,const int *tokens,size_t count,uint32_t *out);
/* Architecture-resolved source frequency metadata; no device resources. */
int bn_model_native_rope_frequency(const BnModel *model,int compressed,
    BnModelNativeRopeFrequency *out);
/* One-shot publication of native core projection resources after validated
 * anatomy/numerics and before any backend state. Failure preserves the model.
 * Remaining native parameters/operators are not admitted by this stage. */
int bn_model_prepare_native_core(BnModel *model, BnGPUBackend *gpu,
    size_t device_budget_bytes, size_t host_staging_budget_bytes);
int bn_model_prepare_native_parameters(BnModel *model,
    size_t device_budget_bytes,size_t host_staging_budget_bytes);
int bn_model_prepare_native_routers(BnModel *model,
    size_t device_budget_bytes,size_t host_staging_budget_bytes);
const BnMoEExpertReader *bn_model_trellis_expert_reader(const BnModel *model);
/* Projection 0=gate, 1=up, 2=down; returned source is borrowed from the model. */
const BnTrellisSource *bn_model_trellis_expert_source(const BnModel *model,
    int layer, int expert, int projection);
/* Prepare backend-owned resources from a bound topology. Budgets exclude
 * shared expert source/cache storage and per-request resources. */
int bn_model_prepare_bf16_topology(BnModel *model, BnGPUBackend *gpu,
                                  size_t device_budget, size_t host_budget);
/* Additional model-owned GPU cache payload, separate from these budgets. */
int bn_model_prepare_bf16_topology_with_cache(BnModel *model,BnGPUBackend *gpu,
    size_t device_budget,size_t host_budget,size_t cache_budget);
void bn_model_free(BnModel *m);
void bn_model_embed_token(const BnModel *m, float *out, int token);
int bn_model_uses_moe(const BnModel *model);
const BnCPURuntimePolicy *bn_model_cpu_runtime_policy(const BnModel *model);

// Upload all model weights to backend-owned GPU buffers.
// Returns 0 on success. On failure, releases partially uploaded buffers.
int bn_model_upload_weights(BnModel *model, BnGPUBackend *gpu);
int bn_model_init_gpu_activations(const BnModel *model, BnGPUBackend *gpu);

// Release all GPU weight buffers. Safe to call if gpu is NULL.
void bn_model_release_gpu(BnModel *model);

#endif // BN_MODEL_H
