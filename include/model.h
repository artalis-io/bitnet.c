#ifndef BN_MODEL_H
#define BN_MODEL_H

#include "platform.h"
#include "gguf.h"
#include "quant.h"
#include "threadpool.h"
#include "sh_arena.h"
#include "gpu_backend.h"
#include "turboquant.h"

// Forward declaration for MoE expert map (defined in moe.h)
typedef struct {
    size_t gate_offset, up_offset, down_offset;
    size_t expert_gate_bytes, expert_up_bytes, expert_down_bytes;
    int gate_type, up_type, down_type;
    int gate_rows, gate_cols;
    int up_rows, up_cols;
    int down_rows, down_cols;
    // Repacked: contiguous [gate|up|down] per expert for cache locality.
    // NULL if not repacked (pread mode or insufficient memory).
    uint8_t *repacked;          // [n_experts * expert_total_bytes]
    size_t expert_total_bytes;  // gate_bytes + up_bytes + down_bytes
} BnMoEExpertMap;

#define BN_MAX_MOE_K 16

// Shared MoE I/O control plane (lives on BnModel, shared across sessions)
typedef struct {
    int fd;
    const uint8_t *mmap_base; // mmap'd file base pointer (NULL if using pread)
    int madvise_mode;         // 1 = madvise-guided mmap (WILLNEED/DONTNEED)
    void *prefetch;           // BnMoEPrefetch* for gate+up (opaque, pread only)
    void *prefetch_down;      // BnMoEPrefetch* for down proj (opaque, pread only)
    void *cache;              // BnMoECache* for expert LRU cache (opaque, pread only)
    void *gpu_moe_cache;      // BnGPUMoECache* for GPU expert buffer cache (opaque)
} BnMoEIO;

// Accumulated MoE timing and I/O stats
typedef struct {
    size_t io_bytes;          // total bytes loaded from disk (pread) or touched (mmap)
    double io_time_ms;        // total time spent in expert loading (pread only)
    double route_time_ms;     // total time in routing (router matvec + top-K)
    double compute_time_ms;   // total time in expert FFN compute (all phases)
    double gate_up_time_ms;   // gate+up matvec time
    double swiglu_time_ms;    // SwiGLU activation time
    double down_time_ms;      // down projection matvec time
    double accum_time_ms;     // weighted accumulation time
    double shared_time_ms;    // shared expert time
    double norm_time_ms;      // RMSNorm time
    double prefetch_wait_ms;  // time main thread waited for I/O prefetch
    double madvise_time_ms;   // time spent in madvise calls
    int    io_count;          // number of expert projections loaded
    size_t cache_hits;        // expert cache hits (pread only)
    size_t cache_misses;      // expert cache misses (pread only)
} BnMoEStats;

// MoE per-session state (compute buffers + pread staging + stats)
typedef struct {
    BnMoEStats stats;         // accumulated timing stats
    // Compute buffers (arena-allocated)
    float *router_logits;
    float *expert_out;
    float *expert_weights;
    int   *expert_indices;
    float *expert_hb;
    float *expert_hb2;
    // Batch buffers for cross-expert dispatch (mmap path)
    float *expert_hb_batch[BN_MAX_MOE_K];   // K gate outputs [moe_hidden]
    float *expert_hb2_batch[BN_MAX_MOE_K];  // K up outputs [moe_hidden]
    float *expert_down_batch[BN_MAX_MOE_K]; // K down outputs [dim]
    int8_t *down_x_q_bufs;                 // [K * moe_hidden] int8 scratch for multi-dispatch down
    // Pread staging buffers (arena-allocated, per-session)
    uint8_t *buf;             // gate buffer
    size_t buf_size;
    uint8_t *buf2;            // up buffer / double-buffer
    size_t buf2_size;
    uint8_t *buf3;            // prefetch gate buffer
    size_t buf3_size;
    uint8_t *buf4;            // prefetch up buffer
    size_t buf4_size;
    uint8_t *buf5;            // down buffer
    size_t buf5_size;
} BnMoEState;

#define BN_DEFAULT_ROPE_THETA  10000.0f
#define BN_DEFAULT_NORM_EPS    1e-5f

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len;
    float rope_theta, norm_eps;
    int head_size, kv_dim, kv_mul;  // derived
    int has_ffn_gate, act_type;     // 0=SiLU, 1=ReLU²
    int qk_norm_per_head;           // 1 = per-head separate norms [dim], 0 = shared [head_size]
    int flash_attn;                 // use flash attention (online softmax)
    int kv_f16;                     // store KV cache in FP16 (halves attention DRAM bandwidth)
    // Hybrid SSM + Attention (all zero = pure attention, backward compatible)
    int rope_dim_count;             // partial RoPE dim count (0 = full head_size)
    int rope_text_dims;             // MROPE: dims for text section only (0 = use rope_dim_count)
    int full_attn_interval;         // 0 = all attention, N = every Nth layer is attention
    int ssm_state_size;             // head_k_dim (128)
    int ssm_conv_kernel;            // conv kernel size (4)
    int ssm_inner_size;             // value_dim = num_v_heads * head_v_dim (4096)
    int ssm_time_step_rank;         // num_v_heads (32)
    int ssm_group_count;            // num_k_heads (16)
    // MoE config (all zero = dense FFN, backward compatible)
    int n_experts;              // total experts per layer (e.g. 256 for Qwen3.5-35B)
    int n_experts_active;       // top-K active per token (e.g. 8)
    int moe_intermediate_size;  // per-expert hidden dim
    int has_shared_expert;      // 1 if shared expert exists
    int shared_expert_intermediate_size; // shared expert hidden dim
    // TurboQuant KV compression (0=disabled, 2-4 = bits)
    int kv_tq_bits;
} BnConfig;

typedef struct {
    float *attn_norm, *attn_sub_norm;       // RMSNorm weights [dim]
    BnQWeight wq, wk, wv, wo;                 // attention projection weights (NULL for SSM layers)
    float *q_bias, *k_bias, *v_bias;        // attention biases (NULL if not present)
    float *q_norm, *k_norm;                 // per-head Q/K RMSNorm (NULL if absent)
    float *ffn_norm, *ffn_sub_norm;         // RMSNorm weights
    BnQWeight ffn_gate, ffn_up, ffn_down;     // FFN weights
    // SSM-specific (NULL/zero for attention layers)
    BnQWeight wqkv;                         // fused QKV [dim, qkv_dim]
    BnQWeight wz;                           // Z gate projection [dim, value_dim]
    float *ssm_a;                           // [num_v_heads] F32 — A_log
    BnQWeight ssm_alpha;                    // [dim, num_v_heads] — decay projection
    BnQWeight ssm_beta;                     // [dim, num_v_heads] — update rate projection
    float *ssm_conv1d;                      // [conv_kernel, conv_dim] F32
    float *ssm_dt_bias;                     // [num_v_heads] F32
    float *ssm_norm;                        // [head_v_dim] F32
    BnQWeight ssm_out;                      // [value_dim, dim]
    // MoE weights (NULL/zero for dense layers)
    float *router_weight;                   // [n_experts * dim] F32 — routing gate (always resident)
    BnMoEExpertMap expert_map;              // file offsets for gate/up/down expert tensors
    // Shared expert (always resident, standard QWeight)
    BnQWeight shared_gate, shared_up, shared_down;
    float *shared_expert_gate;   // [dim] sigmoid gate for shared expert output (NULL if absent)
    // GPU handles for F32 norm weights (NULL = not uploaded)
    void *attn_norm_gpu;
    void *ffn_norm_gpu;
    // GPU handles for attention biases (NULL = not uploaded or no biases)
    void *q_bias_gpu;
    void *k_bias_gpu;
    void *v_bias_gpu;
    // Stacked QKV weight buffer for GPU (NULL = use individual Q/K/V)
    void *qkv_stacked_gpu;
    // Stacked gate+up weight buffer for GPU (NULL = use individual gate/up)
    void *gateup_stacked_gpu;
    // Stacked SSM alpha+beta weight buffer for GPU (NULL = use individual alpha/beta)
    void *ssm_ab_stacked_gpu;
    // GPU handles for Q/K norms and sub-norms (NULL = not present)
    void *q_norm_gpu;
    void *k_norm_gpu;
    void *attn_sub_norm_gpu;
    void *ffn_sub_norm_gpu;
    // GPU handles for SSM weights (NULL = not uploaded or not SSM layer)
    void *ssm_conv1d_gpu;       // [kern * qkv_dim] F32
    void *ssm_dt_bias_gpu;      // [nv] F32
    void *ssm_a_log_gpu;        // [nv] F32
    void *ssm_norm_gpu;         // [head_v_dim] F32
} BnLayerWeights;

typedef struct {
    const void *token_embedding;  // raw embedding data (F16, Q4_0, Q8_0, etc.)
    int emb_type;                 // tensor type (F16, Q4_0, Q8_0, etc.)
    int8_t *emb_out_i8;          // [vocab_size * dim] INT8 copy for logits (NULL if unused)
    float  *emb_out_scales;      // [vocab_size] per-row scales (NULL if unused)
    BnQWeight output_weight;      // untied output projection (data=NULL if tied)
    float *output_norm;           // [dim]
    void *output_norm_gpu;        // GPU handle for output_norm (NULL = not uploaded)
    void *emb_gpu_buf;            // GPU handle for tied embedding (NULL if untied or not uploaded)
    BnLayerWeights *layers;         // [n_layers]
} BnWeights;

typedef struct {
    float *x, *xb, *xb2;         // [dim] activation buffers
    float *hb, *hb2;             // [hidden_dim]
    float *q;                     // [dim] query buffer
    float *att;                   // [n_heads * seq_len] attention scores
    float *logits;                // [vocab_size]
    float *key_cache;             // [n_attn_layers * seq_len * kv_dim]
    float *value_cache;           // [n_attn_layers * seq_len * kv_dim]
    int8_t *x_q;                  // [max(dim, hidden_dim)] scratch for int8 quantized x
    float *rope_freq;             // [head_size/2] precomputed RoPE frequencies
    // TurboQuant compressed KV cache (NULL if kv_tq_bits == 0)
    uint8_t *key_cache_tq;        // [n_attn_layers * seq_len * n_kv_heads * key_bytes]
    uint8_t *value_cache_tq;      // [n_attn_layers * seq_len * n_kv_heads * val_bytes]
    float *q_rotated;             // [n_heads * head_size] scratch for rotated queries
    // SSM state (NULL if no SSM layers)
    float *ssm_state;             // [n_ssm * num_v_heads * head_k_dim * head_v_dim]
    float *ssm_conv_state;        // [n_ssm * (conv_kernel-1) * conv_dim]
} BnRunState;

typedef struct {
    BnConfig config;
    BnWeights weights;
    BnMappedFile file;       // keeps mmap/buffer alive
    BnThreadPool *pool;      // thread pool for parallel dispatch
    SHArena *weight_arena;   // arena for weight transforms (INT8 embeddings, Q4_0 repacking)
    // MoE shared I/O (zero for dense models)
    BnMoEIO moe_io;
    int expert_fd;           // file descriptor for expert pread, -1 if unused
    BnGPUBackend *gpu;       // GPU compute backend (NULL = CPU only)
    BnTQState *tq_state;     // TurboQuant state (NULL = no TQ compression)
    // Cached GPU op list (Phase 4: eliminates per-token malloc)
    void *gpu_graph;         // BnGPUGraph* (opaque, allocated by forward_gpu)
} BnModel;

int  bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len, int kv_f16, int kv_tq_bits);
void bn_model_free(BnModel *m);
void bn_model_embed_token(const BnModel *m, float *out, int token);

// Upload all model weights to GPU. Sets gpu_buf on each BnQWeight.
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
