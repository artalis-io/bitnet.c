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

typedef struct {
    void *gate_buf;
    void *up_buf;
    void *down_buf;
    int use_gateup_split;
} BnGPUMoEPrefillExpert;

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
    // Upload quantized weight data without optional backend-side auxiliary
    // caches. Optional; callers use this for memory-sensitive resident caches.
    void *(*buffer_create_quant_only)(void *ctx, const void *data, size_t size,
                                      int type, int rows, int cols);
    // Upload K-quant weight data and request an FP32 auxiliary cache when the
    // backend can fit it. Optional; callers use this for paths that are faster
    // with resident dequantized weights.
    void *(*buffer_create_kquant_f32_cache)(void *ctx, const void *data,
                                            size_t size, int type,
                                            int rows, int cols);
    // Upload quantized weight data and force an FP16 auxiliary cache when the
    // backend can allocate it. Optional; callers use this only after their own
    // memory fit check for resident CUDA layouts.
    void *(*buffer_create_f16_cache)(void *ctx, const void *data,
                                     size_t size, int type,
                                     int rows, int cols);
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

    // Batched dense FFN with input RMSNorm fused into the backend. Optional.
    // This is used by prompt processing to avoid a per-layer CPU norm pass
    // before uploading the prompt activation batch to the backend.
    int (*dense_ffn_batch_norm)(void *ctx, float *out,
                                void *gate_buf, void *up_buf,
                                void *down_buf, void *norm_buf,
                                const float *X, int n_tokens,
                                int dim, int hidden_dim,
                                int gate_type, int up_type, int down_type,
                                int act_type, float norm_eps);

    // Same as dense_ffn_batch_norm, returning X + FFN(norm(X)).
    int (*dense_ffn_batch_norm_resid)(void *ctx, float *out,
                                      void *gate_buf, void *up_buf,
                                      void *down_buf, void *norm_buf,
                                      const float *X, int n_tokens,
                                      int dim, int hidden_dim,
                                      int gate_type, int up_type,
                                      int down_type, int act_type,
                                      float norm_eps);

    // Batched routed MoE FFN for prompt processing. X is the expert input
    // token batch [n_tokens, dim]. expert_offsets/counts describe flat
    // token_ids/weights assignments for each expert. Expert buffers must be
    // resident backend handles. Returns weighted sum in out[n_tokens, dim].
    int (*moe_ffn_batch)(void *ctx, float *out,
                         const BnGPUMoEPrefillExpert *experts,
                         int n_experts,
                         const int *expert_offsets,
                         const int *expert_counts,
                         const int *token_ids,
                         const float *weights,
                         const float *X,
                         int n_tokens, int dim, int hidden_dim,
                         int gate_type, int up_type, int down_type,
                         int act_type,
                         void *shared_gate_buf, void *shared_up_buf,
                         void *shared_down_buf, void *shared_gate_weight_buf,
                         int shared_hidden_dim,
                         int shared_gate_type, int shared_up_type,
                         int shared_down_type);

    // Batched routed MoE FFN using monolithic all-expert resident handles.
    // indices/weights are [n_tokens, k] route results. Returns weighted sum
    // in out[n_tokens, dim].
    int (*moe_routed_ffn_batch)(void *ctx, float *out,
                                void *gate_all_buf, void *up_all_buf,
                                void *down_all_buf,
                                const int *indices,
                                const float *weights,
                                const float *X,
                                int n_tokens, int dim, int hidden_dim,
                                int n_experts, int k,
                                int gate_type, int up_type,
                                int down_type, int act_type);

    // Batched MoE routing for prompt processing. X is [n_tokens, dim].
    // Returns indices/weights as [n_tokens, k] on the host.
    int (*moe_route_batch)(void *ctx, int *indices, float *weights,
                           void *router_buf, const float *X,
                           int n_tokens, int dim, int n_experts, int k,
                           int norm_topk_prob, float expert_weights_scale);

    // Combined batched MoE routing and resident routed FFN for prompt
    // processing. Avoids route readback and re-upload.
    int (*moe_route_routed_ffn_batch)(void *ctx, float *out,
                                      void *router_buf,
                                      void *gate_all_buf,
                                      void *up_all_buf,
                                      void *down_all_buf,
                                      const float *X,
                                      int n_tokens, int dim, int hidden_dim,
                                      int n_experts, int k,
                                      int gate_type, int up_type,
                                      int down_type, int act_type,
                                      int norm_topk_prob,
                                      float expert_weights_scale);

    // Same as moe_route_routed_ffn_batch, with input RMSNorm fused before
    // routing and residual add fused into the returned output:
    // out = X + MoE(norm(X)).
    int (*moe_route_routed_ffn_batch_norm_resid)(
                                      void *ctx, float *out,
                                      void *router_buf,
                                      void *gate_all_buf,
                                      void *up_all_buf,
                                      void *down_all_buf,
                                      void *shared_gate_buf,
                                      void *shared_up_buf,
                                      void *shared_down_buf,
                                      void *shared_gate_weight_buf,
                                      void *norm_buf,
                                      const float *X,
                                      int n_tokens, int dim, int hidden_dim,
                                      int n_experts, int k,
                                      int gate_type, int up_type,
                                      int down_type, int act_type,
                                      int shared_hidden_dim,
                                      int shared_gate_type,
                                      int shared_up_type,
                                      int shared_down_type,
                                      float norm_eps,
                                      int norm_topk_prob,
                                      float expert_weights_scale);

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

    // Same as prefill_qkv_attention_wo, with input RMSNorm fused into the
    // backend before QK/WV projection.
    int (*prefill_qkv_attention_wo_norm)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *attn_norm_buf,
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

    // Same as prefill_qkv_attention_wo_norm, returning X + Attention(norm(X)).
    int (*prefill_qkv_attention_wo_norm_resid)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *attn_norm_buf,
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

    // Full dense transformer layer prefill fast path:
    // X + Attention(norm(X)) + FFN(norm(...)), with K/V rows copied back for
    // the existing CPU-owned session KV cache unless K_out/V_out are NULL. If
    // K_out/V_out are NULL, CUDA may write directly to backend KV activation
    // buffers at kv_cache_off using kv_cache_stride rows. Optional hook.
    // CUDA may accept X == NULL to reuse the previous device-resident output,
    // and out == NULL to leave the new output resident for the next layer.
    int (*prefill_dense_layer)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *gate_buf, void *up_buf,
                                    void *down_buf, void *attn_norm_buf,
                                    void *ffn_norm_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    void *q_bias_buf, void *k_bias_buf,
                                    void *v_bias_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int hidden_dim, int n_heads,
                                    int n_kv_heads, int head_size,
                                    int kv_mul, int kv_dim, int qk_rows,
                                    int qk_type, int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int gate_type, int up_type,
                                    int down_type, int act_type,
                                    int qk_norm_per_head, float norm_eps,
                                    int pos0, int rope_dims,
                                    uint32_t kv_cache_off,
                                    int kv_cache_stride,
                                    float attention_scale);

    // Full MoE transformer layer prefill fast path:
    // X + Attention(norm(X)) + MoE(norm(...)). CUDA may accept X == NULL to
    // reuse the previous device-resident output, and out == NULL to leave the
    // new output resident for the next layer.
    int (*prefill_moe_layer)(
                                    void *ctx, float *out,
                                    void *qk_buf, void *wv_buf, void *wo_buf,
                                    void *router_buf, void *gate_all_buf,
                                    void *up_all_buf, void *down_all_buf,
                                    void *shared_gate_buf,
                                    void *shared_up_buf,
                                    void *shared_down_buf,
                                    void *shared_gate_weight_buf,
                                    void *attn_norm_buf,
                                    void *ffn_norm_buf,
                                    void *q_norm_buf, void *k_norm_buf,
                                    void *q_bias_buf, void *k_bias_buf,
                                    void *v_bias_buf,
                                    const float *X, float *K_out,
                                    float *V_out, int n_tokens, int dim,
                                    int moe_hidden_dim, int n_experts,
                                    int experts_active, int n_heads,
                                    int n_kv_heads, int head_size,
                                    int kv_mul, int kv_dim, int qk_rows,
                                    int qk_type, int wv_rows, int wv_type,
                                    int wo_rows, int wo_cols, int wo_type,
                                    int gate_type, int up_type,
                                    int down_type, int act_type,
                                    int shared_hidden_dim,
                                    int shared_gate_type,
                                    int shared_up_type,
                                    int shared_down_type,
                                    int qk_norm_per_head, float norm_eps,
                                    int pos0, int rope_dims,
                                    uint32_t kv_cache_off,
                                    int kv_cache_stride,
                                    float attention_scale,
                                    int norm_topk_prob,
                                    float expert_weights_scale);

    // Hybrid/SSM prompt block fast path:
    // out[n_tokens, dim] = X + ssm_out(SSM(norm(X))). Backend owns and updates
    // its resident SSM recurrent state. Optional CUDA-oriented hook.
    int (*prefill_ssm_layer)(
                                    void *ctx, float *out,
                                    void *wqkv_buf, void *wz_buf,
                                    void *alpha_buf, void *beta_buf,
                                    void *qkvz_stacked_buf,
                                    void *ab_stacked_buf,
                                    void *ssm_out_buf, void *attn_norm_buf,
                                    void *conv1d_buf, void *dt_bias_buf,
                                    void *a_log_buf, void *ssm_norm_buf,
                                    void *ffn_gate_buf, void *ffn_up_buf,
                                    void *ffn_down_buf, void *ffn_norm_buf,
                                    const float *X, int n_tokens, int dim,
                                    int qkv_dim, int inner_dim,
                                    int num_k_heads, int head_k_dim,
                                    int num_v_heads, int head_v_dim,
                                    int conv_kernel, int ssm_idx,
                                    int wqkv_type, int wz_type,
                                    int alpha_type, int beta_type,
                                    int out_type, int hidden_dim,
                                    int ffn_gate_type, int ffn_up_type,
                                    int ffn_down_type, int act_type,
                                    float norm_eps, int *did_ffn);

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

    // Compute W * activation[buf_idx] and return argmax without materializing
    // host logits. Optional; intended for greedy decode fast paths.
    int (*matvec_argmax_activation)(void *ctx, void *W_buf, int type,
                                    int rows, int cols, int buf_idx,
                                    const int *penalty_tokens,
                                    int n_penalty_tokens,
                                    float repeat_penalty, int *out_token);

    // Return free/total device memory in bytes. Optional.
    int (*memory_info)(void *ctx, size_t *free_bytes, size_t *total_bytes);

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

static inline int bn_gpu_backend_can_create_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create;
}

static inline BnGPUBackendKind bn_gpu_backend_kind(
    const BnGPUBackend *gpu) {
    return gpu ? gpu->kind : BN_GPU_BACKEND_UNKNOWN;
}

static inline int bn_gpu_backend_is_cuda(const BnGPUBackend *gpu) {
    return bn_gpu_backend_kind(gpu) == BN_GPU_BACKEND_CUDA;
}

static inline int bn_gpu_backend_is_metal(const BnGPUBackend *gpu) {
    return bn_gpu_backend_kind(gpu) == BN_GPU_BACKEND_METAL;
}

static inline int bn_gpu_backend_is_webgpu(const BnGPUBackend *gpu) {
    return bn_gpu_backend_kind(gpu) == BN_GPU_BACKEND_WEBGPU;
}

static inline int bn_gpu_backend_can_create_quant_only_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_quant_only;
}

static inline int bn_gpu_backend_can_create_kquant_f32_cache_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_kquant_f32_cache;
}

static inline int bn_gpu_backend_can_create_f16_cache_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_f16_cache;
}

static inline int bn_gpu_backend_can_create_biased_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_biased;
}

static inline int bn_gpu_backend_can_create_stacked2_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_stacked2;
}

static inline int bn_gpu_backend_can_create_stacked3_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_stacked3;
}

static inline int bn_gpu_backend_can_create_stacked3_biased_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_stacked3_biased;
}

static inline int bn_gpu_backend_can_destroy_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_destroy;
}

static inline int bn_gpu_backend_can_execute(
    const BnGPUBackend *gpu) {
    return gpu && gpu->execute;
}

static inline int bn_gpu_backend_can_write_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->write_activation;
}

static inline int bn_gpu_backend_can_read_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->read_activation;
}

static inline int bn_gpu_backend_can_query_memory(
    const BnGPUBackend *gpu) {
    return gpu && gpu->memory_info;
}

static inline int bn_gpu_backend_execute(const BnGPUBackend *gpu,
                                         const void *ops,
                                         int n_ops,
                                         int readback_buf,
                                         float *readback,
                                         int readback_count) {
    if (!bn_gpu_backend_can_execute(gpu))
        return -1;
    return gpu->execute(gpu->ctx, ops, n_ops, readback_buf, readback,
                        readback_count);
}

static inline int bn_gpu_backend_write_activation(const BnGPUBackend *gpu,
                                                  int buf_idx,
                                                  const void *data,
                                                  size_t size,
                                                  size_t offset) {
    if (!bn_gpu_backend_can_write_activation(gpu))
        return -1;
    return gpu->write_activation(gpu->ctx, buf_idx, data, size, offset);
}

static inline int bn_gpu_backend_read_activation(const BnGPUBackend *gpu,
                                                 int buf_idx,
                                                 void *out,
                                                 size_t size,
                                                 size_t offset) {
    if (!bn_gpu_backend_can_read_activation(gpu))
        return -1;
    return gpu->read_activation(gpu->ctx, buf_idx, out, size, offset);
}

static inline int bn_gpu_backend_query_memory(const BnGPUBackend *gpu,
                                              size_t *free_bytes,
                                              size_t *total_bytes) {
    if (!bn_gpu_backend_can_query_memory(gpu))
        return -1;
    return gpu->memory_info(gpu->ctx, free_bytes, total_bytes);
}

static inline void bn_gpu_backend_destroy_buffer(BnGPUBackend *gpu,
                                                 void *buffer) {
    if (bn_gpu_backend_can_destroy_buffer(gpu) && buffer)
        gpu->buffer_destroy(gpu->ctx, buffer);
}

// Backend capability bits
#define BN_GPU_CAP_FLASH_ATTN  (1u << 0)  // fused flash attention shader available
#define BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT (1u << 1) // native-quant split matvec shader available
#define BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT (1u << 2) // deinterleaved K-quant split matvec shader available
#define BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT (1u << 3) // low-bit block32 split matvec shader available
#define BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU (1u << 4) // low-bit block32 fused gate/up SiLU shader available
#define BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT (1u << 5) // asymmetric K-quant split matvec shader available
#define BN_GPU_CAP_MIDBIT_BLOCK32_FUSED_GATEUP_SILU (1u << 6) // mid-bit block32 fused gate/up SiLU shader available
#define BN_GPU_CAP_MIDBIT_BLOCK32_MATVEC_SPLIT (1u << 7) // mid-bit block32 split matvec shader available
#define BN_GPU_CAP_NATIVE_QUANT_FUSED_GATEUP_SILU (1u << 8) // native-quant fused gate/up SiLU shader available
#define BN_GPU_CAP_DEINTERLEAVED_KQUANT_FUSED_GATEUP_SILU (1u << 9) // deinterleaved K-quant fused gate/up SiLU shader available
#define BN_GPU_CAP_LAYERWISE_ROPE (1u << 10) // backend can vary RoPE frequency policy per layer

#endif // BN_GPU_BACKEND_H
