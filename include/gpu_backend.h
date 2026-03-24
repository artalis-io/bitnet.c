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
#define BN_GPU_SHADER_COPY         9  // pseudo-op: buffer-to-buffer copy (no shader)
#define BN_GPU_SHADER_BIAS_ADD     10 // x[i] += bias[i], bias from W_buf
#define BN_GPU_SHADER_COUNT        11

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
#define BN_GPU_BUF_COUNT       12

// A single GPU operation in the forward pass
typedef struct {
    int shader;          // BN_GPU_SHADER_* constant
    int type;            // BN_GGUF_TENSOR_* (matvec only, -1 otherwise)
    void *W_buf;         // weight buffer handle (matvec only, NULL otherwise)
    int buf_in;          // activation buffer index for primary input
    int buf_out;         // activation buffer index for output
    int buf_aux;         // secondary input buffer (-1 if unused)
    int rows, cols;      // dimensions (matvec: weight dims; others: element count in p0)
    uint32_t p[8];       // shader-specific parameters (32 bytes)
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
} BnGPUBackend;

#endif // BN_GPU_BACKEND_H
