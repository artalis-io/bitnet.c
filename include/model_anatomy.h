#ifndef BN_MODEL_ANATOMY_H
#define BN_MODEL_ANATOMY_H
#include <stdint.h>

/* Shared immutable model anatomy values. Family planning APIs stay in
 * model_arch.h; backend binders consume these values without dispatching on
 * architecture names. This header owns no runtime resources. */
typedef enum {
    BN_MODEL_TENSOR_ATTN_NORM = 0,
    BN_MODEL_TENSOR_ATTN_Q,
    BN_MODEL_TENSOR_ATTN_K,
    BN_MODEL_TENSOR_ATTN_V,
    BN_MODEL_TENSOR_ATTN_OUTPUT,
    BN_MODEL_TENSOR_ATTN_Q_BIAS,
    BN_MODEL_TENSOR_ATTN_K_BIAS,
    BN_MODEL_TENSOR_ATTN_V_BIAS,
    BN_MODEL_TENSOR_ATTN_Q_NORM,
    BN_MODEL_TENSOR_ATTN_K_NORM,
    BN_MODEL_TENSOR_ATTN_SUB_NORM,
    BN_MODEL_TENSOR_ATTN_POST_NORM,
    BN_MODEL_TENSOR_SSM_QKV,
    BN_MODEL_TENSOR_SSM_GATE,
    BN_MODEL_TENSOR_SSM_A,
    BN_MODEL_TENSOR_SSM_ALPHA,
    BN_MODEL_TENSOR_SSM_BETA,
    BN_MODEL_TENSOR_SSM_CONV1D,
    BN_MODEL_TENSOR_SSM_DT_BIAS,
    BN_MODEL_TENSOR_SSM_NORM,
    BN_MODEL_TENSOR_SSM_OUT,
    BN_MODEL_TENSOR_FFN_NORM,
    BN_MODEL_TENSOR_FFN_POST_ATTN_NORM,
    BN_MODEL_TENSOR_FFN_SUB_NORM,
    BN_MODEL_TENSOR_FFN_POST_NORM,
    BN_MODEL_TENSOR_FFN_GATE,
    BN_MODEL_TENSOR_FFN_UP,
    BN_MODEL_TENSOR_FFN_DOWN,
    BN_MODEL_TENSOR_MOE_ROUTER,
    BN_MODEL_TENSOR_MOE_GATE_EXPS,
    BN_MODEL_TENSOR_MOE_UP_EXPS,
    BN_MODEL_TENSOR_MOE_GATE_UP_EXPS,
    BN_MODEL_TENSOR_MOE_DOWN_EXPS,
    BN_MODEL_TENSOR_SHARED_FFN_GATE,
    BN_MODEL_TENSOR_SHARED_FFN_UP,
    BN_MODEL_TENSOR_SHARED_FFN_DOWN,
    BN_MODEL_TENSOR_SHARED_FFN_ROUTER,
    BN_MODEL_TENSOR_LAYER_OUTPUT_SCALE,
    BN_MODEL_TENSOR_HC_ATTN_NORM,
    BN_MODEL_TENSOR_HC_ATTN_DOWN,
    BN_MODEL_TENSOR_HC_ATTN_UP,
    BN_MODEL_TENSOR_HC_ATTN_INJECT,
    BN_MODEL_TENSOR_HC_FFN_NORM,
    BN_MODEL_TENSOR_HC_FFN_DOWN,
    BN_MODEL_TENSOR_HC_FFN_UP,
    BN_MODEL_TENSOR_HC_FFN_INJECT,
    BN_MODEL_TENSOR_INDEXER_Q,
    BN_MODEL_TENSOR_INDEXER_K,
    BN_MODEL_TENSOR_INDEXER_Q_NORM,
    BN_MODEL_TENSOR_INDEXER_K_NORM,
    BN_MODEL_TENSOR_HC_OUTPUT_NORM,
    BN_MODEL_TENSOR_HC_OUTPUT_DOWN,
    BN_MODEL_TENSOR_HC_OUTPUT_UP,
    BN_MODEL_TENSOR_PLE_KEY,
    BN_MODEL_TENSOR_PLE_VALUE,
    BN_MODEL_TENSOR_PLE_NORM_KEY,
    BN_MODEL_TENSOR_PLE_NORM_QUERY,
    BN_MODEL_TENSOR_PLE_NORM_CONV,
    BN_MODEL_TENSOR_PLE_CONV1D,
    BN_MODEL_TENSOR_SSM_CONV1D_Q,
    BN_MODEL_TENSOR_SSM_CONV1D_K,
    BN_MODEL_TENSOR_SSM_CONV1D_V,
    BN_MODEL_TENSOR_SSM_F_A,
    BN_MODEL_TENSOR_SSM_F_B,
    BN_MODEL_TENSOR_SSM_G_A,
    BN_MODEL_TENSOR_SSM_G_B,
    BN_MODEL_TENSOR_MHC_ATTN_FN,
    BN_MODEL_TENSOR_MHC_ATTN_BASE,
    BN_MODEL_TENSOR_MHC_ATTN_SCALE,
    BN_MODEL_TENSOR_MHC_FFN_FN,
    BN_MODEL_TENSOR_MHC_FFN_BASE,
    BN_MODEL_TENSOR_MHC_FFN_SCALE,
    BN_MODEL_TENSOR_MLA_Q_A,
    BN_MODEL_TENSOR_MLA_Q_A_NORM,
    BN_MODEL_TENSOR_MLA_Q_B,
    BN_MODEL_TENSOR_MLA_KV_A,
    BN_MODEL_TENSOR_MLA_KV_A_NORM,
    BN_MODEL_TENSOR_MLA_K_B,
    BN_MODEL_TENSOR_MLA_V_B,
    BN_MODEL_TENSOR_INDEXER_K_BIAS,
    BN_MODEL_TENSOR_INDEXER_HEAD_WEIGHTS,
    BN_MODEL_TENSOR_INDEXER_POOL_APE,
    BN_MODEL_TENSOR_INDEXER_POOL_GATE,
    BN_MODEL_TENSOR_MOE_ROUTER_BIAS,
    BN_MODEL_TENSOR_TOKEN_EMBD,
    BN_MODEL_TENSOR_OUTPUT,
    BN_MODEL_TENSOR_OUTPUT_NORM,
    BN_MODEL_TENSOR_ROLE_COUNT,
} BnModelTensorRole;

#define BN_MODEL_ANATOMY_MAX_LAYERS 128

/* Native expert-storage profile only. Other native config fields remain in
 * the immutable JSON document until a complete topology planner admits them. */
typedef struct {
    char architecture[64];
    int format, layers, experts, active_experts, dim, hidden, nextn_layers;
} BnModelNativeExpertConfig;

/* Explicit DeepSeek V4.1 text numerical profile. These settings do not admit
 * an executor, validate tensor payloads, or replace legacy BnConfig. */
typedef struct {
    float rms_epsilon, hc_epsilon, swiglu_limit, routed_scale;
    float rope_theta, compress_rope_theta, yarn_factor, yarn_beta_fast, yarn_beta_slow;
    float rope_attention_factor;
    int main_rope_yarn, compress_rope_yarn, yarn_truncate;
    int hc_streams, hc_iterations, hc_carry_pre, norm_topk;
    int yarn_original_context, context, bos_token, eos_token, pad_token;
    int engram_vocab, engram_compressed_vocab, engram_pad_token;
} BnModelNativeNumerics;

/* Explicit row-storage to math bridge; not original Diffbot full-forward
 * arithmetic and not model execution admission. Architecture selects it. */
typedef enum {BN_NATIVE_ENGRAM_MATH_NONE=0,BN_NATIVE_ENGRAM_MATH_BF16_LOOKUP_F16=1} BnModelNativeEngramMathKind;
typedef struct {
    BnModelNativeEngramMathKind profile;
    int layer,dim,streams,heads,head_dim,embedding_dim;
    float eps,gate_scale;
} BnModelNativeEngramMath;
/* Original ordinary-text native LinearFP16 head: F16 input/weight/output.
 * last_tokens_only is a request choice and changes the head GEMM geometry. */
typedef enum {BN_NATIVE_HEAD_NONE=0,BN_NATIVE_HEAD_F16_LINEAR_F16=1} BnModelNativeHeadKind;
typedef struct {BnModelNativeHeadKind profile;int dim,vocab;} BnModelNativeHead;

/* Resolved source frequency metadata, independent of execution backend.
 * ramp_scale rounds the DOUBLE scalar reciprocal, not reciprocal(F32 width).
 * Plain families use factor1/start0/scale1. Frequency tables do not include
 * an attention magnitude factor. */
typedef struct {
    int rotary_dim,yarn;
    float base,factor,ramp_start,ramp_scale;
} BnModelNativeRopeFrequency;


/* Native text storage/scheduling plan. Tensor IDs refer to the borrowed
 * safetensors index, not GGUF. No payload reads or executor admission. */
typedef enum {
    BN_NATIVE_ATTN_NORM,
    BN_NATIVE_FFN_NORM,
    BN_NATIVE_HC_ATTN_FN,
    BN_NATIVE_HC_ATTN_BASE,
    BN_NATIVE_HC_ATTN_SCALE,
    BN_NATIVE_HC_FFN_FN,
    BN_NATIVE_HC_FFN_BASE,
    BN_NATIVE_HC_FFN_SCALE,
    BN_NATIVE_Q_A,
    BN_NATIVE_Q_A_SCALE,
    BN_NATIVE_Q_B,
    BN_NATIVE_Q_B_SCALE,
    BN_NATIVE_KV,
    BN_NATIVE_KV_SCALE,
    BN_NATIVE_O_A,
    BN_NATIVE_O_A_SCALE,
    BN_NATIVE_O_B,
    BN_NATIVE_O_B_SCALE,
    BN_NATIVE_Q_NORM,
    BN_NATIVE_KV_NORM,
    BN_NATIVE_ATTN_SINK,
    BN_NATIVE_ROUTER,
    BN_NATIVE_ROUTER_BIAS,
    BN_NATIVE_ROUTER_BIAS_VL,
    BN_NATIVE_SHARED_GATE,
    BN_NATIVE_SHARED_GATE_SCALE,
    BN_NATIVE_SHARED_UP,
    BN_NATIVE_SHARED_UP_SCALE,
    BN_NATIVE_SHARED_DOWN,
    BN_NATIVE_SHARED_DOWN_SCALE,
    BN_NATIVE_COMP_KV,
    BN_NATIVE_COMP_GATE,
    BN_NATIVE_COMP_NORM,
    BN_NATIVE_INDEX_Q,
    BN_NATIVE_INDEX_Q_SCALE,
    BN_NATIVE_INDEX_WEIGHTS,
    BN_NATIVE_INDEX_K,
    BN_NATIVE_INDEX_K_NORM,
    BN_NATIVE_ENGRAM_EMBED,
    BN_NATIVE_ENGRAM_SCALE,
    BN_NATIVE_ENGRAM_WKV,
    BN_NATIVE_ENGRAM_WKV_SCALE,
    BN_NATIVE_ENGRAM_K,
    BN_NATIVE_ENGRAM_Q,
    BN_NATIVE_ROLE_COUNT
} BnModelNativeTensorRole;
typedef enum { BN_NATIVE_CANDIDATE_NONE=0, BN_NATIVE_CANDIDATE_SOURCE,
    BN_NATIVE_CANDIDATE_USE } BnModelNativeCandidateMode;
typedef struct {
    int compress_ratio, kv_source, index_source;
    BnModelNativeCandidateMode candidate_mode;
    uint64_t engram_rows;
    int tensor[BN_NATIVE_ROLE_COUNT]; /* -1 for absent roles. */
} BnModelNativeLayerAnatomy;
#define BN_MODEL_NATIVE_MAX_CORE_PROJECTIONS (BN_MODEL_ANATOMY_MAX_LAYERS * 10)
/* Architecture-owned logical projection order, roles and source geometry.
 * Backend preparation consumes these explicit pairs, never family names. */
typedef struct {
    int layer, weight_role, scale_role, weight_tensor, scale_tensor, rows, cols;
} BnModelNativeCoreProjection;
#define BN_MODEL_NATIVE_MAX_PARAMETERS (BN_MODEL_ANATOMY_MAX_LAYERS * BN_NATIVE_ROLE_COUNT + 3)
typedef enum { BN_NATIVE_PARAMETER_RAW=0, BN_NATIVE_PARAMETER_F16_LINEAR } BnModelNativeParameterLayout;
typedef enum { BN_NATIVE_GLOBAL_EMBEDDING=0, BN_NATIVE_GLOBAL_OUTPUT,
    BN_NATIVE_GLOBAL_OUTPUT_NORM, BN_NATIVE_GLOBAL_COUNT } BnModelNativeGlobalRole;
typedef struct {
    int layer, role, tensor, type, rows, cols;
    BnModelNativeParameterLayout layout;
} BnModelNativeParameter;
typedef struct {
    BnModelNativeExpertConfig experts;
    int core_format; /* Registry-owned native block-scaled core format. */
    int vocab, context, heads, head_dim, rope_dim, q_rank, o_groups, o_rank;
    int streams, index_heads, index_dim, index_topk, sliding_window;
    int candidate_source, candidate_blocks, candidate_block_size;
    int engram_ngram, engram_heads, engram_dim, engram_format;
    int embedding, output, output_norm;
    BnModelNativeLayerAnatomy layers[BN_MODEL_ANATOMY_MAX_LAYERS];
    int core_projection_count;
    BnModelNativeCoreProjection core_projections[BN_MODEL_NATIVE_MAX_CORE_PROJECTIONS];
    int parameter_count;
    BnModelNativeParameter parameters[BN_MODEL_NATIVE_MAX_PARAMETERS];
} BnModelNativeAnatomy;

enum {BN_MODEL_NATIVE_ENGRAM_MAX_NGRAM=16,BN_MODEL_NATIVE_ENGRAM_MAX_HEADS=128};
typedef struct {
    int layer,ngram,heads_per_ngram,heads,compressed_vocab;
    int pad_token; /* Original tokenizer ID; map before hash/history use. */
    uint64_t rows;
    uint64_t sizes[BN_MODEL_NATIVE_ENGRAM_MAX_HEADS],offsets[BN_MODEL_NATIVE_ENGRAM_MAX_HEADS];
    uint64_t multipliers[BN_MODEL_NATIVE_ENGRAM_MAX_NGRAM];
} BnModelNativeEngramLayout;

typedef enum {
    BN_MODEL_ANATOMY_NONE = 0,
    BN_MODEL_ANATOMY_KDA,
    BN_MODEL_ANATOMY_INDEXED_MLA,
} BnModelLayerAnatomyKind;

typedef struct {
    BnModelLayerAnatomyKind kind;
    int uses_moe;
    float swiglu_clamp_exp, swiglu_clamp_shexp;
    /* -1 for roles not present. Indices refer to the caller-owned merged GGUF
     * file; no weight copies, backend handles, or request state live here. */
    int tensor[BN_MODEL_TENSOR_ROLE_COUNT];
} BnModelLayerAnatomy;

typedef struct BnModelAnatomy {
    int dim, stored_layers, n_layers, nextn_layers, vocab_size, context_length;
    int dense_hidden, leading_dense_layers;
    int attention_heads, mla_key_dim, mla_value_dim, q_rank, kv_rank, rope_dim;
    int kda_heads, kda_head_dim, kda_gate_rank, conv_kernel;
    int mhc_streams, mhc_iterations;
    int indexer_heads, indexer_dim, indexer_top_k, indexer_pool;
    int experts, active_experts, expert_hidden, shared_hidden, shared_experts;
    int normalize_expert_weights;
    float norm_eps, indexer_norm_eps, mhc_eps, kda_lower_bound;
    float kda_qk_eps, kda_output_eps, expert_weights_scale;
    /* GGUF KDA parameter storage is -exp(A_log); preserve its F32 bits. */
    int kda_parameter_is_negative_exp;
    int embedding_tensor, output_tensor, output_norm_tensor;
    BnModelLayerAnatomy layers[BN_MODEL_ANATOMY_MAX_LAYERS];
} BnModelAnatomy;

#endif
