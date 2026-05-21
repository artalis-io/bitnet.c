#ifndef BN_GPU_BACKEND_H
#define BN_GPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    BN_GPU_BACKEND_UNKNOWN = 0,
    BN_GPU_BACKEND_METAL = 1,
    BN_GPU_BACKEND_WEBGPU = 2,
    BN_GPU_BACKEND_CUDA = 3,
} BnGPUBackendKind;

// Descriptor for one operation in a batched matvec submission.
typedef struct {
    float *out;      // host output pointer
    void  *W_buf;    // GPU weight handle (opaque, from buffer_create)
    int rows, cols, type;
} BnGPUMatvecOp;

// GPU compute backend vtable. The caller (e.g., Hull) fills this in
// with their GPU API. bitnet.c calls it for matvec dispatch.
// All function pointers may be NULL (graceful fallback to CPU SIMD).
#ifndef BN_GPU_BACKEND_DECLARED
#define BN_GPU_BACKEND_DECLARED
typedef struct BnGPUBackend BnGPUBackend;
#endif

struct BnGPUBackend {
    // Upload quantized weight data to GPU. Returns opaque buffer handle.
    // type: BN_GGUF_TENSOR_* constant. data/size: raw GGUF tensor bytes.
    // Returns NULL on failure.
    void *(*buffer_create)(void *ctx, const void *data, size_t size,
                           int type, int rows, int cols);
    void  (*buffer_destroy)(void *ctx, void *buffer);

    // Upload quantized weight data with fused bias. Returns opaque buffer handle.
    // The bias data (float[bias_size/4]) is appended to the repacked weight buffer.
    // Returns NULL if not supported for this type, or on failure.
    // Optional (NULL = not supported; caller falls back to separate bias upload).
    void *(*buffer_create_biased)(void *ctx, const void *data, size_t size,
                                   int type, int rows, int cols,
                                   const void *bias, size_t bias_size);

    // Upload two adjacent logical weight tensors as one stacked GPU buffer.
    // Optional (NULL = caller combines data before buffer_create).
    void *(*buffer_create_stacked2)(void *ctx,
                                    const void *data0, size_t size0,
                                    const void *data1, size_t size1,
                                    int type, int rows, int cols);
    void *(*buffer_create_stacked3)(void *ctx,
                                    const void *data0, size_t size0,
                                    const void *data1, size_t size1,
                                    const void *data2, size_t size2,
                                    int type, int rows, int cols);
    void *(*buffer_create_stacked3_biased)(void *ctx,
                                           const void *data0, size_t size0,
                                           const void *data1, size_t size1,
                                           const void *data2, size_t size2,
                                           int type, int rows, int cols,
                                           const void *bias,
                                           size_t bias_size);

    // Quantized matvec: out[rows] = W[rows, cols] @ x[cols]
    // W_buf: opaque handle from buffer_create.
    // x: host float[cols], out: host float[rows] (GPU copies to/from device).
    // Returns 0 on success, nonzero on error (falls back to CPU).
    int (*matvec)(void *ctx, float *out, void *W_buf, const float *x,
                  int rows, int cols, int type);

    // Batch matvec: out[n_tokens * rows] = W @ X[n_tokens * cols]
    // Optional (NULL = repeated single matvec or CPU fallback).
    int (*matmul)(void *ctx, float *out, void *W_buf, const float *X,
                  int rows, int cols, int n_tokens, int type);

    // Batched matmul: multiple W_i @ X projections sharing the same
    // X[n_tokens, x_cols]. Outputs go to host pointers in each op.
    // Optional (NULL = repeated matmul or CPU fallback).
    int (*matmul_batch)(void *ctx, const BnGPUMatvecOp *ops, int n_ops,
                        const float *X, int n_tokens, int x_cols);

    // Batched matvec: encode multiple dispatches in one GPU submission.
    // All ops share the same input x[x_cols]. Outputs go to separate host ptrs.
    // Optional (NULL = fall back to individual matvec calls).
    // Returns 0 on success, -1 on error.
    int (*matvec_batch)(void *ctx, const BnGPUMatvecOp *ops, int n_ops,
                        const float *x, int x_cols);

    // Dense FFN fast path: out[dim] = down(activation(gate(x)) * up(x)).
    // All weight buffers are opaque handles from buffer_create. This is
    // optional and intended for backends that can keep FFN intermediates
    // resident across gate/up activation and down projection.
    int (*dense_ffn)(void *ctx, float *out,
                     void *gate_buf, void *up_buf, void *down_buf,
                     const float *x, int dim, int hidden_dim,
                     int gate_type, int up_type, int down_type,
                     int act_type);

    // Batched dense FFN fast path for prompt processing:
    // out[n_tokens, dim] = down(activation(gate(X)) * up(X)).
    // Optional; backends may keep all gate/up/down intermediates resident.
    int (*dense_ffn_batch)(void *ctx, float *out,
                           void *gate_buf, void *up_buf, void *down_buf,
                           const float *X, int n_tokens,
                           int dim, int hidden_dim,
                           int gate_type, int up_type, int down_type,
                           int act_type);

    // Batched causal attention for prompt processing:
    // out[n_tokens, n_heads * head_size] =
    // attention(Q[n_tokens, n_heads * head_size],
    //           K/V[n_tokens, n_kv_heads * head_size]).
    // Q and K must already include bias, norm, and RoPE. This prompt helper
    // handles only the current prompt window, so callers should use it only
    // when pos0 == 0 unless the backend documents broader cache support.
    int (*prefill_attention)(void *ctx, float *out,
                             const float *Q, const float *K, const float *V,
                             int n_tokens, int n_heads, int n_kv_heads,
                             int head_size, int kv_mul, int kv_dim,
                             float attention_scale);

    // Fused prompt attention + output projection. Optional; backends may keep
    // the attention intermediate resident before applying W_wo.
    int (*prefill_attention_wo)(void *ctx, float *out, void *wo_buf,
                                const float *Q, const float *K,
                                const float *V, int n_tokens,
                                int n_heads, int n_kv_heads, int head_size,
                                int kv_mul, int kv_dim, int wo_rows,
                                int wo_cols, int wo_type,
                                float attention_scale);

    // Fused prompt QK/WV matmul + Q/K norm/RoPE + attention + W_O.
    // Optional CUDA-oriented fast path. Writes processed K/V rows back to
    // K_out/V_out so the existing session KV cache remains authoritative.
    int (*prefill_qkv_attention_wo)(void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int n_heads, int n_kv_heads,
                                    int head_size, int kv_mul, int kv_dim,
                                    int qk_rows, int qk_type,
                                    int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int qk_norm_per_head, float norm_eps,
                                    int pos0, int rope_dims,
                                    float attention_scale);

    // GPU-resident forward pass: execute a backend-private lowered command list
    // as a single submission. All intermediate buffers stay on GPU. Only
    // readback_buf is copied to out_host.
    // Returns 0 on success, -1 on error (caller should fall back to CPU).
    int (*execute)(void *ctx, const void *ops, int n_ops,
                   int readback_buf, float *out_host, int out_len);

    // Initialize GPU-resident activation buffers for a given model config.
    // Must be called after weight upload, before execute().
    // Returns 0 on success.
    int (*init_activations)(void *ctx, const void *config);  // config is BnConfig*

    // Free GPU-resident activation buffers.
    void (*free_activations)(void *ctx);

    // Write host data to a GPU-resident activation buffer.
    // buf_idx: backend activation value slot. offset/size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*write_activation)(void *ctx, int buf_idx, const void *data,
                            size_t size, size_t offset);

    // Read GPU-resident activation buffer to host.
    // buf_idx: backend activation value slot. out: host buffer, size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*read_activation)(void *ctx, int buf_idx, void *out,
                           size_t size, size_t offset);

    // Return argmax over a GPU-resident float buffer, optionally applying the
    // same repeat penalty used by greedy CPU sampling. Optional.
    int (*argmax_activation)(void *ctx, int buf_idx, int n,
                             const int *penalty_tokens, int n_penalty_tokens,
                             float repeat_penalty, int *out_token);

    void *ctx;  // opaque backend context

    // Capability flags (set by backend, checked by transformer)
    uint32_t caps;

    // Concrete backend identity for planning/debugging. Future backends should
    // set this instead of relying on capability inference alone.
    BnGPUBackendKind kind;

    // Maximum storage-buffer binding size in bytes. 0 = unknown; callers
    // should use a conservative fallback when deciding whether to bind a
    // large weight buffer in a GPU-resident graph.
    size_t max_storage_binding_size;
};

// Backend capability bits
#define BN_GPU_CAP_FLASH_ATTN  (1u << 0)  // fused flash attention shader available
#define BN_GPU_CAP_Q8_MATVEC_SPLIT (1u << 1) // stacked Q8_0 split matvec shader available
#define BN_GPU_CAP_Q5K_MATVEC_SPLIT (1u << 2) // Q5_K packed split matvec shader available
#define BN_GPU_CAP_Q4_MATVEC_SPLIT (1u << 3) // stacked Q4_0 split matvec shader available
#define BN_GPU_CAP_Q4_FUSED_GATEUP_SILU (1u << 4) // fused Q4_0 gate/up SiLU shader available
#define BN_GPU_CAP_Q4K_MATVEC_SPLIT (1u << 5) // Q4_K packed split matvec shader available
#define BN_GPU_CAP_Q5_FUSED_GATEUP_SILU (1u << 6) // fused Q5_0 gate/up SiLU shader available
#define BN_GPU_CAP_Q5_MATVEC_SPLIT (1u << 7) // stacked Q5_0 split matvec shader available
#define BN_GPU_CAP_Q8_FUSED_GATEUP_SILU (1u << 8) // fused Q8_0 gate/up SiLU shader available

#endif // BN_GPU_BACKEND_H
