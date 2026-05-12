#ifndef BN_GPU_BACKEND_H
#define BN_GPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

// --- Phase 4b.3: GPU-resident forward pass ---

// Backend-neutral graph values used by GPU-resident forward ops.
// Backends lower these values to their own activation buffer slots.
#define BN_GPU_VALUE_X           0
#define BN_GPU_VALUE_XB          1
#define BN_GPU_VALUE_XB2         2
#define BN_GPU_VALUE_Q           3
#define BN_GPU_VALUE_HB          4
#define BN_GPU_VALUE_HB2         5
#define BN_GPU_VALUE_KEY_CACHE   6
#define BN_GPU_VALUE_VALUE_CACHE 7
#define BN_GPU_VALUE_ATT         8
#define BN_GPU_VALUE_LOGITS      9
#define BN_GPU_VALUE_ROPE_FREQ   10
#define BN_GPU_VALUE_SCRATCH     11
#define BN_GPU_VALUE_QKV         12
#define BN_GPU_VALUE_MOE_HB      13
#define BN_GPU_VALUE_MOE_HB2     14
#define BN_GPU_VALUE_MOE_OUT     15
#define BN_GPU_VALUE_SSM_STATE      16
#define BN_GPU_VALUE_SSM_CONV_STATE 17
#define BN_GPU_VALUE_SSM_QKV        18
#define BN_GPU_VALUE_SSM_Z          19
#define BN_GPU_VALUE_SSM_ALPHA      20
#define BN_GPU_VALUE_SSM_BETA       21
#define BN_GPU_VALUE_SSM_V          22
#define BN_GPU_VALUE_COUNT          23

// Shader uniform parameter count (32 bytes = 8 × u32, matches WGSL Uniforms structs)
#define BN_GPU_OP_PARAMS 8

typedef enum {
    BN_GPU_OP_UNKNOWN = 0,
    BN_GPU_OP_MATVEC = 1,
    BN_GPU_OP_RMSNORM = 2,
    BN_GPU_OP_ROPE = 3,
    BN_GPU_OP_ATTENTION = 4,
    BN_GPU_OP_ACTIVATION = 5,
    BN_GPU_OP_RESIDUAL = 6,
    BN_GPU_OP_COPY = 7,
    BN_GPU_OP_FFN = 8,
    BN_GPU_OP_SSM = 9,
    BN_GPU_OP_LOGITS = 10,
} BnGPUOpKind;

typedef enum {
    BN_GPU_CODE_UNKNOWN = 0,
    BN_GPU_CODE_MATVEC,
    BN_GPU_CODE_RMSNORM,
    BN_GPU_CODE_ROPE,
    BN_GPU_CODE_GQA_SCORES,
    BN_GPU_CODE_SOFTMAX,
    BN_GPU_CODE_GQA_COMBINE,
    BN_GPU_CODE_SILU_GATE,
    BN_GPU_CODE_RELU2_GATE,
    BN_GPU_CODE_RESIDUAL_ADD,
    BN_GPU_CODE_COPY,
    BN_GPU_CODE_BIAS_ADD,
    BN_GPU_CODE_RESIDUAL_RMSNORM,
    BN_GPU_CODE_WEIGHTED_ADD,
    BN_GPU_CODE_SSM_CONV_SILU,
    BN_GPU_CODE_SSM_L2NORM,
    BN_GPU_CODE_SSM_ALPHA_BETA,
    BN_GPU_CODE_SSM_DELTA,
    BN_GPU_CODE_SSM_GATE,
    BN_GPU_CODE_PER_HEAD_RMSNORM,
    BN_GPU_CODE_DEINTERLEAVE_Q,
    BN_GPU_CODE_SIGMOID_GATE,
    BN_GPU_CODE_FLASH_ATTN,
    BN_GPU_CODE_MATVEC_SPLIT,
    BN_GPU_CODE_ROPE_QK,
    BN_GPU_CODE_FUSED_GATEUP_SILU,
    BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT,
    BN_GPU_CODE_Q4K_MATVEC_SPLIT,
    BN_GPU_CODE_Q8_MATVEC_SPLIT,
    BN_GPU_CODE_Q5K_MATVEC_SPLIT,
    BN_GPU_CODE_SILU_ACT,
    BN_GPU_CODE_RELU2_ACT,
} BnGPUOpCode;

typedef enum {
    BN_GPU_BACKEND_UNKNOWN = 0,
    BN_GPU_BACKEND_METAL = 1,
    BN_GPU_BACKEND_WEBGPU = 2,
    BN_GPU_BACKEND_CUDA = 3,
} BnGPUBackendKind;

// A single GPU operation in the forward pass
typedef struct {
    int op_kind;         // BnGPUOpKind semantic op; 0 = infer from op_code
    int op_code;         // BnGPUOpCode backend-neutral concrete op
    int type;            // BN_GGUF_TENSOR_* (matvec only, -1 otherwise)
    void *W_buf;         // weight buffer handle (matvec only, NULL otherwise)
    int buf_in;          // BnGPU graph value for primary input
    int buf_out;         // BnGPU graph value for output
    int buf_aux;         // secondary graph value (-1 if unused)
    int rows, cols;      // dimensions (matvec: weight dims; others: element count in p0)
    uint32_t p[BN_GPU_OP_PARAMS]; // shader-specific parameters (32 bytes)
} BnGPUOp;

static inline BnGPUOpKind bn_gpu_op_kind_from_code(int code) {
    switch (code) {
        case BN_GPU_CODE_MATVEC:
        case BN_GPU_CODE_MATVEC_SPLIT:
        case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
        case BN_GPU_CODE_Q8_MATVEC_SPLIT:
        case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
            return BN_GPU_OP_MATVEC;
        case BN_GPU_CODE_RMSNORM:
        case BN_GPU_CODE_RESIDUAL_RMSNORM:
        case BN_GPU_CODE_PER_HEAD_RMSNORM:
            return BN_GPU_OP_RMSNORM;
        case BN_GPU_CODE_ROPE:
        case BN_GPU_CODE_ROPE_QK:
            return BN_GPU_OP_ROPE;
        case BN_GPU_CODE_GQA_SCORES:
        case BN_GPU_CODE_SOFTMAX:
        case BN_GPU_CODE_GQA_COMBINE:
        case BN_GPU_CODE_FLASH_ATTN:
            return BN_GPU_OP_ATTENTION;
        case BN_GPU_CODE_SILU_GATE:
        case BN_GPU_CODE_RELU2_GATE:
        case BN_GPU_CODE_SIGMOID_GATE:
        case BN_GPU_CODE_SILU_ACT:
        case BN_GPU_CODE_RELU2_ACT:
            return BN_GPU_OP_ACTIVATION;
        case BN_GPU_CODE_RESIDUAL_ADD:
        case BN_GPU_CODE_WEIGHTED_ADD:
        case BN_GPU_CODE_BIAS_ADD:
            return BN_GPU_OP_RESIDUAL;
        case BN_GPU_CODE_COPY:
        case BN_GPU_CODE_DEINTERLEAVE_Q:
            return BN_GPU_OP_COPY;
        case BN_GPU_CODE_FUSED_GATEUP_SILU:
            return BN_GPU_OP_FFN;
        case BN_GPU_CODE_SSM_CONV_SILU:
        case BN_GPU_CODE_SSM_L2NORM:
        case BN_GPU_CODE_SSM_ALPHA_BETA:
        case BN_GPU_CODE_SSM_DELTA:
        case BN_GPU_CODE_SSM_GATE:
        case BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT:
            return BN_GPU_OP_SSM;
        default:
            return BN_GPU_OP_UNKNOWN;
    }
}

static inline BnGPUOpKind bn_gpu_op_kind(const BnGPUOp *op) {
    if (!op) return BN_GPU_OP_UNKNOWN;
    return op->op_kind ? (BnGPUOpKind)op->op_kind
                       : bn_gpu_op_kind_from_code(op->op_code);
}

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
    // buf_idx: BN_GPU_VALUE_* graph value. offset/size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*write_activation)(void *ctx, int buf_idx, const void *data,
                            size_t size, size_t offset);

    // Read GPU-resident activation buffer to host.
    // buf_idx: BN_GPU_VALUE_* graph value. out: host buffer, size in bytes.
    // Returns 0 on success, -1 on error.  Optional (NULL = not supported).
    int (*read_activation)(void *ctx, int buf_idx, void *out,
                           size_t size, size_t offset);

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

// Pre-compiled GPU op list for dense models (Phase 4: eliminates per-token malloc)
typedef struct {
    BnGPUOp *ops;       // pre-allocated op array
    int cap;            // capacity (max ops)
} BnGPUGraph;

#endif // BN_GPU_BACKEND_H
