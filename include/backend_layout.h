#ifndef BN_BACKEND_LAYOUT_H
#define BN_BACKEND_LAYOUT_H

#include "gpu_backend.h"
#include "model_config.h"
#include "model_anatomy.h"
#include "model_weights.h"
#include "quant.h"
#include "sh_arena.h"
#include "safetensors.h"

typedef struct BnBackendModel BnBackendModel;
struct BnGGUFFile;
struct BnModelAnatomy;

/* Immutable backend-owned native F16[K,N] core weight. CPU input index is
 * borrowed only during init. Budgets cover final GPU payload + one GPU tile,
 * and raw/scale/F16 host tiles; context/allocator overhead is separate.
 * GPU outlives the layout and all consumers; empty out required. */
typedef struct {
    BnGPUBackend *gpu;
    void *weight;
    int format, rows, cols;
    size_t bytes, device_peak_bytes, host_staging_bytes;
} BnBackendNativeF16Weight;
int bn_backend_layout_native_f16_init(BnBackendNativeF16Weight *out,
    BnGPUBackend *gpu, const BnSafetensorsIndex *index, int format,
    size_t weight_tensor, size_t scale_tensor,
    size_t device_budget_bytes, size_t host_budget_bytes);
void bn_backend_layout_native_f16_free(BnBackendNativeF16Weight *weight);

typedef struct {
    size_t count, resident_bytes, device_peak_bytes, host_staging_bytes;
} BnBackendNativeCoreStats;
/* All architecture-listed block-scaled core projections, in declared order.
 * Preflight every pair, coverage, byte extent and aggregate budget before
 * creating GPU buffers. Preparation publishes one owned backend container
 * atomically; failure frees staged weights and preserves the destination.
 * Index borrowed only during preparation. GPU and backend outlive consumers;
 * callers finish/release requests before backend/model teardown.
 * Budgets cover weight payload + one device tile and explicit host tiles,
 * excluding metadata, library/context/driver/allocator and request resources.
 * Other BF16/F32 parameters and Engram tables are separate preparation stages. */
int bn_backend_layout_native_core_plan(const BnSafetensorsIndex *index,
    const BnModelNativeAnatomy *anatomy, BnBackendNativeCoreStats *out);
int bn_backend_layout_prepare_native_core(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnSafetensorsIndex *index,
    const BnModelNativeAnatomy *anatomy, size_t device_budget, size_t host_budget);
const BnBackendNativeCoreStats *bn_backend_layout_native_core_stats(
    const BnBackendModel *backend);
const BnBackendNativeF16Weight *bn_backend_layout_native_core_weight(
    const BnBackendModel *backend, int layer, int role);

/* One owned execution-state extension inside the native core container.
 * Setup only, before request publication; serialized with all backend access.
 * Successful attach transfers ownership. Rejection leaves both owners intact.
 * Cleanup runs before core/router/parameter weights, with their GPU and model
 * sources still alive. Callers finish/release requests and leases before
 * backend teardown. The callback owns its cleanup context; this layer has no
 * session, model-family or cache dependency. State moves with the core owner;
 * a move does not rebind borrowed context, whose original sources must outlive
 * the destination owner. */
int bn_backend_layout_attach_native_execution(BnBackendModel *backend,
    const void *owner, void *state, void (*release)(void *));
void *bn_backend_layout_native_execution(const BnBackendModel *backend,
    const void *owner);

typedef struct {
    void *handle;
    int type, rows, cols;
    BnModelNativeParameterLayout layout;
    size_t bytes;
} BnBackendNativeParameter;
typedef struct {
    size_t count, linear_count, bytes, aggregate_resident_bytes;
    size_t device_peak_bytes, host_staging_bytes;
} BnBackendNativeParameterStats;
/* Extend the same core owner once, before request publication. Every source
 * BF16/F32 parameter is finite-checked. Raw embedding/norm/HC/bias/sink bits
 * retain their source dtype; BF16 linear matrices become packed F16 [K,N].
 * Budgets cover aggregate core+parameter payload/tiles and maximum explicit
 * staging across both stages. Metadata/context/library/request memory is
 * excluded. Failure frees only staged parameters; core handles stay stable.
 * Original immutable index/anatomy are required; no Engram table residency. */
int bn_backend_layout_native_parameter_plan(const BnSafetensorsIndex *index,
    const BnModelNativeAnatomy *anatomy,BnBackendNativeParameterStats *out);
int bn_backend_layout_prepare_native_parameters(BnBackendModel *backend,
    const BnSafetensorsIndex *index,const BnModelNativeAnatomy *anatomy,
    size_t device_budget,size_t host_budget);
const BnBackendNativeParameterStats *bn_backend_layout_native_parameter_stats(
    const BnBackendModel *backend);
/* layer=-1 uses BnModelNativeGlobalRole; otherwise native layer tensor role. */
const BnBackendNativeParameter *bn_backend_layout_native_parameter(
    const BnBackendModel *backend,int layer,int role);

typedef struct {
    size_t count,bytes,aggregate_resident_bytes,device_peak_bytes,host_staging_bytes;
} BnBackendNativeRouterStats;
/* One-shot immutable extension after native parameters, before requests.
 * Preflight all layers and aggregate budgets; failure frees staged router
 * handles and preserves existing core/parameters. No tensor host staging. */
int bn_backend_layout_native_router_plan(const BnSafetensorsIndex *index,
    const BnModelNativeAnatomy *anatomy,BnBackendNativeRouterStats *out);
int bn_backend_layout_prepare_native_routers(BnBackendModel *backend,
    const BnSafetensorsIndex *index,const BnModelNativeAnatomy *anatomy,
    size_t device_budget,size_t host_budget);
const BnBackendNativeRouterStats *bn_backend_layout_native_router_stats(const BnBackendModel *backend);
void *bn_backend_layout_native_router_weight(const BnBackendModel *backend,int layer);

/* Immutable GLM KDA sublayer weights, owned by the backend caller. CPU file
 * storage is borrowed only during init. Q/K/V/beta/F_A/G_A remain one Q8_0
 * projection, preserving the checkpoint operator's width-dependent routing.
 * No decoded copies or request state. Zero-initialize before init. */
typedef struct {
    BnGPUBackend *gpu;
    BnGPUBF16LinearWeight in, f_b, g_b, out;
    void *conv, *decay, *bias, *norm;
    int dim, heads, head_dim, rank, kernel, layer;
    float lower_bound, output_eps;
    size_t bytes, host_staging_bytes;
} BnBackendBF16KDAWeights;

/* Qualified Q8_0 projections/F32 parameters, D=rank=128, kernel=4.
 * Budgets cover raw GPU tensor payload and the single reused host staging
 * allocation, excluding mapped file pages and allocator/context overhead.
 * Preflight validates payload bounds in each source shard. Failure leaves out
 * unchanged and releases partial uploads. GPU outlives weights and borrowers.
 * This sublayer preparation does not admit a complete GLM model for inference. */
int bn_backend_layout_bf16_kda_init(BnBackendBF16KDAWeights *out,
    BnGPUBackend *gpu, struct BnGGUFFile *file,
    const struct BnModelAnatomy *anatomy, int layer,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);
void bn_backend_layout_bf16_kda_free(BnBackendBF16KDAWeights *weights);

/* Four-stream mHC backend layouts. Projection is decoded to F32 (not BF16)
 * for TF32 execution; norm gamma is rounded to BF16 once. Caller owns this
 * immutable backend resource; GPU outlives weights and borrowing requests. */
typedef struct {
    BnGPUBackend *gpu;
    void *fn, *scale, *base, *norm;
    int dim, streams, iterations;
    float rms_eps, hc_eps, norm_eps;
    size_t bytes, host_staging_bytes;
} BnBackendBF16MHCWeights;
/* ffn=0 attention, ffn=1 FFN. Q8_0/F32 fn and F32 norm/scale/base profile.
 * Validate every source and finite decoded parameter before uploads. One
 * reused host staging allocation; budgets exclude mapped pages and allocator
 * overhead. Empty out required; failures leave it unchanged and free uploads. */
int bn_backend_layout_bf16_mhc_init(BnBackendBF16MHCWeights *out,
    BnGPUBackend *gpu, struct BnGGUFFile *file,
    const struct BnModelAnatomy *anatomy, int layer, int ffn,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);
void bn_backend_layout_bf16_mhc_free(BnBackendBF16MHCWeights *weights);

/* Immutable Q8_0 dense FFN. Gate/up stay merged in checkpoint order; both
 * linear stages use the registered raw A8 profiles. No BF16 weight decoding.
 * CPU file pages are borrowed only during init. Zero-initialize first. */
typedef struct {
    BnGPUBackend *gpu;
    BnGPUBF16LinearWeight gate_up, down;
    int dim, hidden_dim;
    float clamp_limit;
    size_t bytes, host_staging_bytes;
} BnBackendBF16DenseFFNWeights;
/* Dense non-shared clamp comes from the anatomy's per-layer ordinary FFN
 * clamp (swiglu_clamp_exp); MoE layers are rejected by this dense binder.
 * Budgets cover GPU payload and one raw host concatenation, excluding mapped
 * pages/context/allocator overhead. Failure leaves out unchanged and frees
 * partial uploads; borrowers are released before these weights and the GPU. */
int bn_backend_layout_bf16_dense_ffn_init(BnBackendBF16DenseFFNWeights *out,
    BnGPUBackend *gpu, struct BnGGUFFile *file,
    const struct BnModelAnatomy *anatomy, int layer,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);
void bn_backend_layout_bf16_dense_ffn_free(BnBackendBF16DenseFFNWeights *weights);
/* Shared Q8_0 FFN on a MoE layer. Uses shared_hidden, exactly one shared
 * expert, SHARED_FFN roles and per-layer swiglu_clamp_shexp. Same ownership,
 * budgets and failure contract; release using dense_ffn_free above. Ordinary
 * linear dispatch applies to this branch, independently of routed batches. */
int bn_backend_layout_bf16_shared_ffn_init(BnBackendBF16DenseFFNWeights *out,
    BnGPUBackend *gpu, struct BnGGUFFile *file,
    const struct BnModelAnatomy *anatomy, int layer,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);


/* Immutable GPU routing/shared resources; routed payloads stay in the model's
 * RAM/file source. The borrowed CPU expert map must outlive these resources
 * and requests. Qualified E288/K8, F32 routing, matching Q4_K/Q5_K gate/up, Q5_K/Q6_K down
 * and Q8 shared. */
typedef struct {
    BnGPUBackend *gpu;
    void *router,*bias;
    BnBackendBF16DenseFFNWeights shared;
    const BnMoEExpertMap *source_map;
    int dim,hidden_dim,n_experts,k,gate_type,down_type,layer;
    float route_scale,clamp_limit;
    size_t gate_bytes,down_bytes,bytes,host_staging_bytes;
} BnBackendBF16MoEWeights;
int bn_backend_layout_bf16_moe_init(BnBackendBF16MoEWeights *out,
    BnGPUBackend *gpu,struct BnGGUFFile *file,
    const struct BnModelAnatomy *anatomy,int layer,const BnMoEExpertMap *source_map,
    size_t gpu_budget_bytes,size_t host_staging_budget_bytes);
void bn_backend_layout_bf16_moe_free(BnBackendBF16MoEWeights *weights);

/* GLM pre-indexer, accepting the MLA caller's normalized q-rank input.
 * Independent raw Q8 projections retain their original dispatch widths.
 * Head projection, LayerNorm gamma/bias and pool APE preserve source F32.
 * This explicit raw-GGUF profile is not the default HF merged-BF16 loader.
 * No host staging allocation; GPU budget covers tensor payload only.
 * CPU sources are borrowed only during init. Zero-init; failure leaves out
 * unchanged, freeing partial uploads. GPU outlives resources and borrowers. */
typedef struct {
    BnGPUBackend *gpu;
    BnGPUBF16LinearWeight query,key,gate;
    void *head_weights,*norm,*bias,*ape;
    int dim,q_rank,heads,head_dim,pool_size,layer;
    float norm_eps,weight_scale;
    size_t bytes;
} BnBackendBF16IndexerWeights;
int bn_backend_layout_bf16_indexer_init(BnBackendBF16IndexerWeights *out,
    BnGPUBackend *gpu,struct BnGGUFFile *file,
    const struct BnModelAnatomy *anatomy,int layer,size_t gpu_budget_bytes);
void bn_backend_layout_bf16_indexer_free(BnBackendBF16IndexerWeights *weights);

/* Raw-GGUF NoPE MLA projections. Raw Q8 QA/QB/KVA/O keep original dispatch;
 * norms are decoded BF16 and per-head K/V absorption matrices are BF16 from
 * explicit Q8_1 identity-projection materialization (not direct decode).
 * K_B is [heads,kv_rank,key_dim], V_B [heads,value_dim,kv_rank]. CPU sources
 * remain immutable. bytes=retained payload; peak_bytes includes serial raw
 * conversion staging. Zero-init; failure leaves out unchanged and frees
 * partial resources. GPU outlives resource and borrowing sessions. */
typedef struct {
    BnGPUBackend *gpu;
    BnGPUBF16LinearWeight q_a,q_b,kv_a,output;
    void *q_norm,*kv_norm,*key,*value;
    int dim,q_rank,kv_rank,heads,key_dim,value_dim,layer;
    float norm_eps,scale;
    size_t bytes,peak_bytes;
} BnBackendBF16MLAWeights;
int bn_backend_layout_bf16_mla_init(BnBackendBF16MLAWeights *out,
    BnGPUBackend *gpu,struct BnGGUFFile *file,
    const struct BnModelAnatomy *anatomy,int layer,size_t gpu_budget_bytes);
void bn_backend_layout_bf16_mla_free(BnBackendBF16MLAWeights *weights);

/* Immutable main-model GLM resources. The container owns the anatomy copy,
 * layer array, expert-map descriptors and GPU layouts. Expert source storage
 * (including any repacked data) remains borrowed from the CPU model and must
 * outlive resources/requests. An optional raw expert cache is backend-owned;
 * all borrowing requests must be released before these resources. */
typedef struct {
    BnBackendBF16KDAWeights kda;
    BnBackendBF16MLAWeights mla;
    BnBackendBF16IndexerWeights indexer;
    BnBackendBF16MHCWeights attention_mhc,ffn_mhc;
    BnBackendBF16DenseFFNWeights dense;
    BnBackendBF16MoEWeights moe;
    BnMoEExpertMap expert_map;
    int expert_cache_layer; /* Compact MoE index, or -1 without a cache. */
} BnBackendBF16GLMLayer;
typedef struct {
    BnGPUBackend *gpu;
    int n_layers,dim,vocab_size;
    const struct BnModelAnatomy *anatomy;
    const BnBackendBF16GLMLayer *layers;
    BnGPUBF16LinearWeight embedding,output;
    void *output_norm; /* BF16 gamma decoded from checkpoint F32. */
    void *expert_cache; /* Separate bounded raw payload; no request state. */
    int expert_cache_layers;
    size_t expert_cache_host_bytes;
    size_t bytes,peak_bytes,host_staging_bytes;
} BnBackendBF16GLMModel;
/* Prepare every main layer and separate Q8 vocabulary tables privately, then
 * attach to an empty backend and bind its GPU. Failure leaves the destination
 * unchanged and releases all staged GPU buffers. GPU budget covers aggregate
 * peak tensor payload; host budget covers the largest temporary preparation
 * allocation. Both exclude context/allocator overhead, file mappings, expert
 * source storage and request state. expert_maps has one entry per main layer
 * when any layer uses MoE; descriptors are copied, never repacked here.
 * Preparation is not whole-model execution admission: final contraction/head
 * execution and generation still require their independent qualification. */
int bn_backend_layout_prepare_bf16_glm_model(BnBackendModel *backend,
    BnGPUBackend *gpu,struct BnGGUFFile *file,const struct BnModelAnatomy *anatomy,
    const BnMoEExpertMap *expert_maps,size_t n_expert_maps,
    size_t gpu_budget_bytes,size_t host_staging_budget_bytes);
/* Cache device budget is separate from weights/requests. The host staging
 * budget also bounds persistent cache metadata, exposed in the model. */
int bn_backend_layout_prepare_bf16_glm_model_with_cache(BnBackendModel *backend,
    BnGPUBackend *gpu,struct BnGGUFFile *file,const struct BnModelAnatomy *anatomy,
    const BnMoEExpertMap *expert_maps,size_t n_expert_maps,
    size_t gpu_budget_bytes,size_t host_staging_budget_bytes,size_t cache_budget_bytes);
const BnBackendBF16GLMModel *bn_backend_layout_bf16_glm_model(const BnBackendModel *backend);

/* Backend-owned immutable HC layouts; zero-initialize before init. The GPU
 * must outlive free; release borrowing sessions first. No model mutation. */
typedef struct {
    BnGPUBackend *gpu;
    void *norm;
    BnGPUBF16LinearWeight down, up, inject;
    size_t bytes, peak_bytes;
} BnBackendBF16HCWeights;

/* Complete mix/combine weights. Budget covers peak raw+decoded GPU tensor
 * payload during sequential preparation, excluding allocator/context overhead.
 * Failure leaves out unchanged and releases all partial allocations. Norm
 * contains checkpoint gamma; the execution plan applies the affine +1 offset.
 */
int bn_backend_layout_bf16_hc_init(BnBackendBF16HCWeights *out,
    BnGPUBackend *gpu, const BnHyperConnectionWeights *weights,
    int dim, int streams, int rank, size_t budget_bytes);
void bn_backend_layout_bf16_hc_free(BnBackendBF16HCWeights *weights);
/* Mix-only output HC: injection must be absent, no synthetic weights. */
int bn_backend_layout_bf16_hc_mixer_init(BnBackendBF16HCWeights *out,
    BnGPUBackend *gpu, const BnHyperConnectionWeights *weights,
    int dim, int streams, int rank, size_t budget_bytes);

typedef struct {
    int n_layers, dim, streams, rank;
    const BnBackendBF16HCWeights *attention, *ffn, *output;
    size_t bytes, peak_bytes;
} BnBackendBF16HCModel;

/* Transactionally attach all HC weights to the backend's precision resource
 * container. Model preparation recovers zero-centered gamma from GGUF effective
 * scales in F32 before BF16 conversion; standalone init accepts gamma directly.
 * Shared CPU weights are unchanged. Host scratch is one F32 norm vector
 * (dim * streams elements) released after upload.
 * Budget covers aggregate preparation tensor payload. This alone
 * does not make the model runnable: bn_backend_layout_bf16 remains NULL.
 * Borrowed arrays expire on backend GPU release. No replacement of live layouts.
 */
int bn_backend_layout_prepare_bf16_hc_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t budget_bytes);
const BnBackendBF16HCModel *bn_backend_layout_bf16_hc_model(
    const BnBackendModel *backend);

/* Immutable decoded-BF16 PLE weights. KV is one merged [HC*dim+dim,dim]
 * projection; separate GEMMs do not preserve the qualified arithmetic.
 * The shared host embedding table is deliberately not staged here. */
typedef struct {
    BnGPUBackend *gpu;
    BnGPUBF16LinearWeight kv;
    void *norm_key, *norm_query, *norm_conv, *conv;
    int layer, dim, streams, kernel, dilation;
    size_t bytes, peak_bytes, host_staging_bytes;
    size_t model_bytes, model_peak_bytes; /* Includes existing HC weights. */
} BnBackendBF16PLEWeights;

/* Extend an existing HC precision container for the SAME config/weights/GPU.
 * GPU budget covers aggregate HC+PLE preparation tensor payload; host budget
 * covers temporary raw concatenation, excluding mmap pages/allocator overhead.
 * Only Q8_0 key/value and F32 gamma/conv checkpoint profile is qualified.
 * Failure preserves the existing container. No live replacement; sessions
 * borrow these weights until backend GPU release. Model remains non-runnable.
 */
int bn_backend_layout_prepare_bf16_ple_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);
const BnBackendBF16PLEWeights *bn_backend_layout_bf16_ple_model(
    const BnBackendModel *backend);

/* Host preparation arithmetic, independent of execution precision. */
typedef enum {
    BN_BACKEND_BF16_MATH_LIBC = 0,
    BN_BACKEND_BF16_MATH_MKL_VML_HA = 1,
} BnBackendBF16Math;

/* Natural-head Flash GDN: merged QKVZ/BA, sigmoid norm, no output transpose.
 * Entries for non-SSM layers are zero. Immutable buffers are shared by requests.
 */
typedef struct {
    BnGPUBF16LinearWeight qkvz, ba, out;
    void *conv, *norm, *dt_bias, *a_log;
    int valid;
} BnBackendBF16FlashGDNWeights;
typedef struct {
    int n_layers, n_gdn_layers, dim, hk, hv, dk, dv, kernel;
    float norm_eps;
    BnGPUGatedNormActivation gate_activation;
    BnBackendBF16Math math_profile;
    const BnBackendBF16FlashGDNWeights *layers;
    size_t bytes, peak_bytes, host_staging_bytes;
    size_t model_bytes, model_peak_bytes; /* Includes existing HC and PLE. */
} BnBackendBF16FlashGDNModel;
/* Extend the SAME HC+PLE precision container. Preflight every GDN layer before
 * allocations; failure preserves existing components. Budgets cover aggregate
 * GPU tensor preparation and temporary host concatenation payload, excluding
 * allocator/context/mmap overhead. Requires the qualified MKL VML HA profile.
 * GPU/model outlive borrowing sessions. Does not make Flash runnable. */
int bn_backend_layout_prepare_bf16_flash_gdn_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);
const BnBackendBF16FlashGDNModel *bn_backend_layout_bf16_flash_gdn_model(
    const BnBackendModel *backend);

/* Immutable decoded-BF16 Flash sparse-attention weights. Main Q/K/V and
 * output are Q8_0 in the qualified checkpoint; the indexer is native BF16.
 * Norms hold Gamma recovered in F32 before BF16 rounding. */
typedef struct {
    BnGPUBF16LinearWeight qkv, index, out;
    void *q_norm, *k_norm, *index_q_norm, *index_k_norm;
    int valid;
} BnBackendBF16FlashQSAWeights;
typedef struct {
    int n_layers, n_qsa_layers, dim, heads, kv_heads, head_size;
    int index_heads, index_head_size, compress_ratio, rotary_dim;
    int rotary_capacity;
    float norm_eps, rope_theta;
    const BnBackendBF16FlashQSAWeights *layers;
    void *rotary_cache;
    size_t bytes, peak_bytes, host_staging_bytes;
    size_t model_bytes, model_peak_bytes; /* Includes existing HC, PLE and GDN. */
} BnBackendBF16FlashQSAModel;
/* Extend the SAME prepared HC+PLE+GDN container. All layer geometry and budgets
 * are checked before uploads. Failure preserves existing components; completed
 * resources are immutable until backend release. Budgets cover tensor payload
 * and temporary host concatenation, excluding allocator/context/mmap overhead.
 * Preparation alone does not admit a full Flash model for execution. */
int bn_backend_layout_prepare_bf16_flash_qsa_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);
const BnBackendBF16FlashQSAModel *bn_backend_layout_bf16_flash_qsa_model(
    const BnBackendModel *backend);

/* Shared immutable FFN layouts. Router checkpoint F32 storage is converted to
 * BF16, matching the reference loader; shared Q8 weights remain quantized.
 * Routed expert weights stay in the model's RAM/file storage, not this layout.
 * The expert map borrows any repacked source from the original model. */
typedef struct {
    BnGPUBF16LinearWeight router, shared_router;
    BnGPUBF16FFNWeights shared;
    BnMoEExpertMap expert_map;
} BnBackendBF16FlashFFNWeights;
typedef struct {
    int n_layers, dim, hidden_dim, shared_hidden_dim, n_experts, k;
    int renormalize;
    BnGPUBF16FFNActivationProfile shared_activation;
    const BnBackendBF16FlashFFNWeights *layers;
    size_t bytes, peak_bytes;
    size_t model_bytes, model_peak_bytes;
    void *expert_cache; /* Optional model-owned raw residency; separate budget. */
    size_t expert_cache_host_bytes;
} BnBackendBF16FlashFFNModel;
/* Extend the SAME HC+PLE+GDN+QSA container transactionally. The GPU budget is
 * aggregate preparation tensor payload, including transient F32 router upload;
 * it excludes allocator/context overhead and CPU expert storage. Every layer
 * is validated before upload. Failure preserves earlier components; successful
 * layouts cannot be replaced and expire on backend release. This does not
 * enable whole-model admission. GPU/model outlive all borrowing sessions. */
int bn_backend_layout_prepare_bf16_flash_ffn_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t gpu_budget_bytes);
const BnBackendBF16FlashFFNModel *bn_backend_layout_bf16_flash_ffn_model(
    const BnBackendModel *backend);

/* Separate raw Q8 input/output vocabulary tables. Final normalization belongs
 * to the existing output HC mixer; there is no additional output norm. */
typedef struct {
    int dim, vocab_size;
    void *embedding;
    BnGPUBF16LinearWeight output;
    size_t bytes, peak_bytes, model_bytes, model_peak_bytes;
} BnBackendBF16FlashIOModel;
/* Extend the SAME HC/PLE/GDN/QSA/FFN container, aggregate tensor budget.
 * No decoded vocabulary expansion. Transactional; model/GPU outlive sessions.
 * Does not admit a whole model until its execution gates are qualified. */
int bn_backend_layout_prepare_bf16_flash_io_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t gpu_budget_bytes);
const BnBackendBF16FlashIOModel *bn_backend_layout_bf16_flash_io_model(
    const BnBackendModel *backend);

/* Prepare all qualified Flash components in a private owner, then publish the
 * complete layout and GPU binding. Destination must have no GPU resources.
 * Failure leaves destination unchanged and releases every staged buffer.
 * GPU budget covers aggregate peak tensor payload; host budget covers peak
 * temporary component preparation payload, excluding allocator/mmap overhead.
 * Expert source storage and request state are separate. Component preparation
 * alone never sets the complete-model marker returned by the accessor below.
 */
int bn_backend_layout_prepare_bf16_flash_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes);
/* Same transaction with an additional bounded raw expert pool. Zero preserves
 * the streaming control. Cache storage is separate from immutable tensor bytes;
 * borrowing requests must be released before backend/model teardown. */
int bn_backend_layout_prepare_bf16_flash_model_with_cache(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    size_t gpu_budget_bytes, size_t host_staging_budget_bytes, size_t cache_budget_bytes);
const BnBackendBF16FlashIOModel *bn_backend_layout_bf16_flash_model(
    const BnBackendModel *backend);

/* Immutable backend-owned weights; request KV/history/state live elsewhere. */
typedef struct {
    void *ni, *nf, *qn, *kn, *conv, *an, *dt, *gn;
    BnGPUBF16LinearWeight qkv, z, ba, o, q, k, v;
    BnGPUBF16FFNWeights ffn;
    void *ffn_gate_up; /* Optional decoded [gate;up]; ffn decoded handles borrow its rows. */
    /* Mixed groups keep Q2048 in qkv, K2048 in k and V6144 in v.
     * Uniform groups retain combined QKV10240 in qkv. Logical geometry
     * affects decoded GEMM as well as the small-batch quant cutoff. */
    int split_qk_v;
} BnBackendBF16Layer;

typedef struct {
    size_t device_bytes, device_peak_bytes, host_bytes, host_peak_bytes;
} BnBackendBF16MemoryPlan;

typedef struct {
    int n_layers, max_tokens;
    BnBackendBF16Math math_profile;
    char math_library[256];
    const BnBackendBF16Layer *layers;
    void *final_norm;
    void *rotary_cache;
    int rotary_capacity;
    BnGPUBF16LinearWeight embedding, output;
    int stacked_ffn_layers;
    BnBackendBF16MemoryPlan memory;
} BnBackendBF16Model;

/* Allocation-free dense model payload plan, using the preparation walk.
 * Includes raw/decoded/stacked/rotary device payload and explicit host staging
 * and ownership tables. Excludes borrowed source pages and driver/allocator
 * bookkeeping. Failure leaves out unchanged; callbacks are not invoked. */
int bn_backend_layout_bf16_memory_plan(BnGPUBackend *gpu,
    const BnConfig *config, const BnWeights *weights, int max_tokens,
    BnBackendBF16Math profile, BnBackendBF16MemoryPlan *out);

/* Explicit, qualified dense hybrid shape only. No change to F32 defaults.
 * On failure all newly allocated resources are released and backend is intact.
 * GPU must outlive model teardown; sessions must release borrowed weights first.
 * max_tokens bounds the decoded layouts prepared for subsequent execution.
 */
int bn_backend_layout_prepare_bf16(BnBackendModel *backend, BnGPUBackend *gpu,
    const BnConfig *config, const BnWeights *weights, int max_tokens);
int bn_backend_layout_bf16_math_available(BnBackendBF16Math profile);
int bn_backend_layout_prepare_bf16_with_math(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    int max_tokens, BnBackendBF16Math profile);
/* Same execution/layout, but reject before allocation unless both planned
 * preparation peaks fit. The compatibility initializer has unlimited caps. */
int bn_backend_layout_prepare_bf16_with_limits(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    int max_tokens, BnBackendBF16Math profile,
    size_t device_budget_bytes, size_t host_budget_bytes);
const BnBackendBF16Model *bn_backend_layout_bf16(const BnBackendModel *backend);

typedef enum {
    BN_BACKEND_LAYOUT_OK = 0,
    BN_BACKEND_LAYOUT_NO_GPU,
    BN_BACKEND_LAYOUT_NO_BUFFER_CREATE,
    BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED,
    BN_BACKEND_LAYOUT_MISSING_WEIGHT,
    BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE,
    BN_BACKEND_LAYOUT_I2S_NOT_STACKABLE =
        BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE,
    BN_BACKEND_LAYOUT_TYPE_MISMATCH,
    BN_BACKEND_LAYOUT_COL_MISMATCH,
    BN_BACKEND_LAYOUT_ZERO_SIZE,
    BN_BACKEND_LAYOUT_ALLOC_FAILED,
    BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED,
} BnBackendLayoutReason;

typedef struct {
    size_t lowbit_repack_bytes;
    size_t q8_repack_bytes;
    size_t kquant_scale_table_bytes;
    size_t expanded_kquant_weight_bytes;
    size_t f32_scale_table_bytes;
} BnBackendLayoutPreparedStats;

const char *bn_backend_layout_reason_string(BnBackendLayoutReason reason);

size_t bn_backend_layout_qweight_data_size(const BnQWeight *w);

BnBackendLayoutReason bn_backend_layout_stackable_reason(const BnQWeight *a,
                                                         const BnQWeight *b);

int bn_backend_layout_stackable(const BnQWeight *a, const BnQWeight *b);

int bn_backend_layout_stackable3(const BnQWeight *a,
                                 const BnQWeight *b,
                                 const BnQWeight *c);

/* Borrow existing resident Q/K handles without combining their formats. */
void *bn_backend_layout_create_prefill_qk_pair(BnGPUBackend *gpu,
                                               void *q, void *k);

BnBackendLayoutReason bn_backend_layout_stacked2_reason(const BnGPUBackend *gpu,
                                                        const BnQWeight *a,
                                                        const BnQWeight *b);

BnBackendLayoutReason bn_backend_layout_biased_qweight_reason(const BnGPUBackend *gpu,
                                                              const BnQWeight *w,
                                                              const float *bias);

BnBackendLayoutReason bn_backend_layout_stacked3_qkv_reason(const BnGPUBackend *gpu,
                                                            const BnQWeight *q,
                                                            const BnQWeight *k,
                                                            const BnQWeight *v,
                                                            const float *q_bias,
                                                            const float *k_bias,
                                                            const float *v_bias,
                                                            int q_bias_fused,
                                                            int k_bias_fused,
                                                            int v_bias_fused);

void *bn_backend_layout_upload_stacked2(BnGPUBackend *gpu,
                                        const BnQWeight *a,
                                        const BnQWeight *b);

void *bn_backend_layout_upload_biased_qweight(BnGPUBackend *gpu,
                                              const BnQWeight *w,
                                              const float *bias);

void *bn_backend_layout_upload_stacked3_qkv(BnGPUBackend *gpu,
                                            const BnQWeight *q,
                                            const BnQWeight *k,
                                            const BnQWeight *v,
                                            const float *q_bias,
                                            const float *k_bias,
                                            const float *v_bias,
                                            int q_bias_fused,
                                            int k_bias_fused,
                                            int v_bias_fused);

size_t bn_backend_layout_prepared_qweights_size(const BnConfig *config,
                                                const BnWeights *weights,
                                                BnBackendLayoutPreparedStats *stats);

void bn_backend_layout_prepare_qweights(BnBackendModel *backend,
                                        const BnConfig *config,
                                        const BnWeights *weights,
                                        SHArena *arena,
                                        BnBackendLayoutPreparedStats *stats);

#endif // BN_BACKEND_LAYOUT_H
