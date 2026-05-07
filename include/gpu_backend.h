#ifndef BN_GPU_BACKEND_H
#define BN_GPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

// --- Phase 4b.3: GPU-resident forward pass ---

// Shader IDs for forward-pass operations
#define BN_GPU_SHADER_MATVEC       0
#define BN_GPU_SHADER_RMSNORM      1
#define BN_GPU_SHADER_ROPE         2
#define BN_GPU_SHADER_GQA_SCORES   3
#define BN_GPU_SHADER_SOFTMAX      4
#define BN_GPU_SHADER_GQA_COMBINE  5
#define BN_GPU_SHADER_SILU_GATE    6
#define BN_GPU_SHADER_RELU2_GATE   7
#define BN_GPU_SHADER_RESIDUAL_ADD 8
#define BN_GPU_SHADER_COPY         9  // compute-shader buffer copy (stays in encoder)
#define BN_GPU_SHADER_BIAS_ADD     10 // x[i] += bias[i], bias from W_buf
#define BN_GPU_SHADER_RESIDUAL_RMSNORM 11 // fused residual_add + rmsnorm
#define BN_GPU_SHADER_WEIGHTED_ADD   12 // x[i] += weight * r[i] (MoE expert accum)
#define BN_GPU_SHADER_SSM_CONV_SILU  13 // Conv1d + SiLU + conv_state shift
#define BN_GPU_SHADER_SSM_L2NORM     14 // Per-head L2 normalization of Q/K
#define BN_GPU_SHADER_SSM_ALPHA_BETA 15 // Softplus + exp/sigmoid for decay/update
#define BN_GPU_SHADER_SSM_DELTA      16 // Delta rule recurrence
#define BN_GPU_SHADER_SSM_GATE           17 // Per-head RMSNorm + SiLU gate
#define BN_GPU_SHADER_PER_HEAD_RMSNORM  18 // Per-head RMSNorm (Q/K norms)
#define BN_GPU_SHADER_DEINTERLEAVE_Q    19 // Extract Q from interleaved [Q,Gate] layout
#define BN_GPU_SHADER_SIGMOID_GATE      20 // out *= sigmoid(gate) for gated Q
#define BN_GPU_SHADER_FLASH_ATTN       21 // Fused Q·K softmax att·V (online softmax)
#define BN_GPU_SHADER_MATVEC_SPLIT     22 // Multi-output matvec (QKV batched, gate+up batched)
#define BN_GPU_SHADER_ROPE_QK          23 // Fused RoPE for Q and K in single dispatch
#define BN_GPU_SHADER_FUSED_GATEUP_SILU 24 // Fused gate+up matvec + SiLU activation (Q4_0)
#define BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT 25 // SSM alpha/beta activation from stacked matvec
#define BN_GPU_SHADER_Q4K_MATVEC_SPLIT 26 // Q4_K two-output matvec split
#define BN_GPU_SHADER_Q8_MATVEC_SPLIT  27 // Q8_0 multi-output matvec split
#define BN_GPU_SHADER_Q5K_MATVEC_SPLIT 28 // Q5_K multi-output matvec split
#define BN_GPU_SHADER_COUNT             29

// GPU-resident activation buffer indices
#define BN_GPU_BUF_X           0
#define BN_GPU_BUF_XB          1
#define BN_GPU_BUF_XB2         2
#define BN_GPU_BUF_Q           3
#define BN_GPU_BUF_HB          4
#define BN_GPU_BUF_HB2         5
#define BN_GPU_BUF_KEY_CACHE   6
#define BN_GPU_BUF_VALUE_CACHE 7
#define BN_GPU_BUF_ATT         8
#define BN_GPU_BUF_LOGITS      9
#define BN_GPU_BUF_ROPE_FREQ   10
#define BN_GPU_BUF_SCRATCH     11  // temp output for KV cache writes
#define BN_GPU_BUF_QKV         12  // stacked QKV matvec output [q_dim + 2*kv_dim]
// MoE buffers
#define BN_GPU_BUF_MOE_HB     13  // expert gate output [moe_hidden_dim]
#define BN_GPU_BUF_MOE_HB2    14  // expert up output [moe_hidden_dim]
#define BN_GPU_BUF_MOE_OUT    15  // accumulated expert output [dim]
// SSM buffers (persistent across tokens)
#define BN_GPU_BUF_SSM_STATE      16  // [n_ssm * num_v_heads * hk * hv]
#define BN_GPU_BUF_SSM_CONV_STATE 17  // [n_ssm * (kern-1) * qkv_dim]
#define BN_GPU_BUF_SSM_QKV        18  // SSM QKV projection output [qkv_dim]
#define BN_GPU_BUF_SSM_Z          19  // Z gate projection output [value_dim]
#define BN_GPU_BUF_SSM_ALPHA      20  // decay rates [num_v_heads]
#define BN_GPU_BUF_SSM_BETA       21  // update rates [num_v_heads]
#define BN_GPU_BUF_SSM_V          22  // V vectors for delta [value_dim]
#define BN_GPU_BUF_COUNT          23

// Shader uniform parameter count (32 bytes = 8 × u32, matches WGSL Uniforms structs)
#define BN_GPU_OP_PARAMS 8

// A single GPU operation in the forward pass
typedef struct {
    int shader;          // BN_GPU_SHADER_* constant
    int type;            // BN_GGUF_TENSOR_* (matvec only, -1 otherwise)
    void *W_buf;         // weight buffer handle (matvec only, NULL otherwise)
    int buf_in;          // activation buffer index for primary input
    int buf_out;         // activation buffer index for output
    int buf_aux;         // secondary input buffer (-1 if unused)
    int rows, cols;      // dimensions (matvec: weight dims; others: element count in p0)
    uint32_t p[BN_GPU_OP_PARAMS]; // shader-specific parameters (32 bytes)
} BnGPUOp;

// Descriptor for one operation in a batched matvec submission.
typedef struct {
    float *out;      // host output pointer
    void  *W_buf;    // GPU weight handle (opaque, from buffer_create)
    int rows, cols, type;
} BnGPUMatvecOp;

// GPU compute backend vtable. The caller (e.g., Hull) fills this in
// with their GPU API. bitnet.c calls it for matvec dispatch.
// All function pointers may be NULL (graceful fallback to CPU SIMD).
typedef struct {
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

    // Batched matvec: encode multiple dispatches in one GPU submission.
    // All ops share the same input x[x_cols]. Outputs go to separate host ptrs.
    // Optional (NULL = fall back to individual matvec calls).
    // Returns 0 on success, -1 on error.
    int (*matvec_batch)(void *ctx, const BnGPUMatvecOp *ops, int n_ops,
                        const float *x, int x_cols);

    // GPU-resident forward pass: execute a sequence of ops as a single submission.
    // All intermediate buffers stay on GPU. Only readback_buf is copied to out_host.
    // Returns 0 on success, -1 on error (caller should fall back to CPU).
    int (*execute)(void *ctx, const BnGPUOp *ops, int n_ops,
                   int readback_buf, float *out_host, int out_len);

    // Initialize GPU-resident activation buffers for a given model config.
    // Must be called after weight upload, before execute().
    // Returns 0 on success.
    int (*init_activations)(void *ctx, const void *config);  // config is BnConfig*

    // Free GPU-resident activation buffers.
    void (*free_activations)(void *ctx);

    // Write host data to a GPU-resident activation buffer.
    // buf_idx: BN_GPU_BUF_* index.  offset/size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*write_activation)(void *ctx, int buf_idx, const void *data,
                            size_t size, size_t offset);

    // Read GPU-resident activation buffer to host.
    // buf_idx: BN_GPU_BUF_* index.  out: host buffer, size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*read_activation)(void *ctx, int buf_idx, void *out,
                           size_t size, size_t offset);

    void *ctx;  // opaque backend context

    // Capability flags (set by backend, checked by transformer)
    uint32_t caps;
} BnGPUBackend;

// Backend capability bits
#define BN_GPU_CAP_FLASH_ATTN  (1u << 0)  // fused flash attention shader available
#define BN_GPU_CAP_Q8_MATVEC_SPLIT (1u << 1) // stacked Q8_0 split matvec shader available
#define BN_GPU_CAP_Q5K_MATVEC_SPLIT (1u << 2) // Q5_K packed split matvec shader available

// Pre-compiled GPU op list for dense models (Phase 4: eliminates per-token malloc)
typedef struct {
    BnGPUOp *ops;       // pre-allocated op array
    int cap;            // capacity (max ops)
} BnGPUGraph;

#endif // BN_GPU_BACKEND_H
