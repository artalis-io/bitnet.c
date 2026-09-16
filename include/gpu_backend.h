#ifndef BN_GPU_BACKEND_H
#define BN_GPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include "runtime_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BN_GPU_BACKEND_UNKNOWN = 0,
    BN_GPU_BACKEND_METAL = 1,
    BN_GPU_BACKEND_WEBGPU = 2,
    BN_GPU_BACKEND_CUDA = 3,
} BnGPUBackendKind;

// Model-independent layout and math for projected prompt Q/K/V values.
// Frequency offsets refer to the backend's prepared activation frequency table.
typedef struct {
    int n_tokens, n_heads, n_kv_heads, head_size;
    int q_row_stride, q_gated, qk_norm_per_head;
    int reference_rmsnorm_order;
    int pos0, rope_dims;
    int attention_window; // 0 = full causal prompt; positive = visible key count
    size_t rope_freq_offset;
    float norm_eps, attention_scale;
    // For pos0 > 0, read earlier KV from the backend's session cache.
    // Offset identifies the first NEW row, in elements; stride is in elements.
    // Prepared attention returns current rows and does not modify the cache.
    size_t kv_cache_off;
    int kv_cache_stride;
} BnGPUAttentionPrefillPlan;

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

// Geometry of a gathered expert batch. Input rows follow original token order,
// with at most one row per token. The runtime owns grouping; backends use the
// full geometry to preserve routed projection arithmetic.
typedef struct {
    int total_tokens;
    int n_experts;
    int expert_index;
    // Source gate/up share one logical projection (0 = independent, 1 = merged).
    // This does not require the backend buffers to share storage.
    int gate_up_fused;
} BnGPUMoEExpertBatchPlan;

// Complete dense-residual MoE composition after routed experts have been
// reduced on the host. Borrowed buffers remain valid for this call only.
typedef struct {
    float *out;
    const float *act;
    const float *routed;
    void *gate_buf;
    void *up_buf;
    void *down_buf;
    void *input_norm_buf;
    void *dense_post_norm_buf;
    void *routed_post_norm_buf;
    void *output_norm_buf;
    int n_tokens;
    int dim;
    int hidden_dim;
    int gate_type;
    int up_type;
    int down_type;
    int act_type;
    float norm_eps;
    int raw_output;
} BnGPUMoEDenseResidualBatch;

typedef enum {
    BN_GPU_ROPE_FACTOR_NONE = 0,
    BN_GPU_ROPE_FACTOR_MULTIPLY,
    BN_GPU_ROPE_FACTOR_DIVIDE,
} BnGPURopeFactorMode;

// Optional construction recipe for a range of the frequency table. Borrowed
// pointers are valid during init_activations only; backends own prepared data.
typedef struct {
    int offset;
    int pair_count;
    int rotary_dims;
    float theta;
    const float *factors;
    BnGPURopeFactorMode factor_mode;
} BnGPURopeFrequencyPlan;

// Backend-neutral description of request-local activation resources. Model and
// runtime planning populate this value; GPU implementations must not inspect
// BnConfig or other model-owned structures.
typedef struct {
    int dim;
    int n_layers;
    int seq_len;
    int kv_dim;
    int n_heads;
    int head_size;
    int vocab_size;
    int per_layer_input_dim;
    int kv_f16;
    int attention_layer_count;
    int ssm_layer_count;
    int uses_hybrid_ssm;
    int uses_hybrid_moe;
    int uses_moe;
    int hyper_connection_count;
    int hyper_connection_rank;
    int xb2_elements;
    int hb_elements;
    int moe_total_experts;
    int moe_active_experts;
    int moe_expert_hidden_dim;
    int ssm_time_step_rank;
    int ssm_state_size;
    int ssm_inner_size;
    int ssm_group_count;
    int ssm_conv_kernel;
    const float *rope_frequencies;
    int rope_frequency_count;
    const BnGPURopeFrequencyPlan *rope_frequency_plans;
    int rope_frequency_plan_count;
    // Preserve standalone RoPE arithmetic after normalization (e.g. IMRoPE).
    // Zero retains the fused normalization/RoPE arithmetic contract.
    int separate_rope_norm;
} BnGPUActivationPlan;

// GPU compute backend vtable. The caller (e.g., Hull) fills this in
// with their GPU API. bitnet.c calls it for matvec dispatch.
// All function pointers may be NULL (graceful fallback to CPU SIMD).
#ifndef BN_GPU_BACKEND_DECLARED
#define BN_GPU_BACKEND_DECLARED
typedef struct BnGPUBackend BnGPUBackend;
#endif

struct BnGPUBackend {
    // Upload quantized weight data to GPU. Returns opaque buffer handle.
    // type: BN_GGUF_TENSOR_* constant, or -1 for untyped auxiliary bytes.
    // data/size: raw tensor bytes; the consuming operation defines auxiliary layout.
    // Returns NULL on failure.
    void *(*buffer_create)(void *ctx, const void *data, size_t size,
                           int type, int rows, int cols);
    // Upload quantized weight data without optional backend-side auxiliary
    // caches. Optional; callers use this for memory-sensitive resident caches.
    void *(*buffer_create_quant_only)(void *ctx, const void *data, size_t size,
                                      int type, int rows, int cols);
    // Wrap immutable host-backed weight storage without copying. Optional;
    // callers must keep data alive for the lifetime of the returned handle.
    void *(*buffer_create_borrowed)(void *ctx, const void *data, size_t size,
                                    int type, int rows, int cols);
    // Wrap immutable host storage in a backend-native layout that can be
    // consumed by ordinary matvec graph operations. Optional.
    void *(*buffer_create_native_matvec_borrowed)(
        void *ctx, const void *data, size_t size, int type, int rows, int cols);
    int (*native_matvec_borrowed_supported)(
        void *ctx, size_t size, int type, int rows, int cols);
    size_t (*buffer_cache_charge)(
        void *ctx, size_t size, int type, int rows, int cols);
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
    // Upper bound on additional resident bytes for buffer_create_f16_cache,
    // excluding raw weights but including packed layouts and companion caches.
    // SIZE_MAX means overflow. Optional; the fallback assumes one FP16 copy.
    size_t (*buffer_f16_cache_extra_bytes)(void *ctx, int type,
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

    // Optional FP32 weighted RMS normalization of X[n_tokens, dim].
    // norm_buf holds at least dim FP32 weights. Supports out == X;
    // rejected calls leave host output unchanged for CPU fallback.
    int (*rmsnorm_batch)(void *ctx, float *out, void *norm_buf,
                         const float *X, int n_tokens, int dim, float eps);

    // Optional grouped RMSNorm for token-major [n_tokens, streams, dim].
    // Each stream has its own contiguous dim-element slice in norm_buf.
    int (*rmsnorm_grouped_batch)(void *ctx, float *out, void *norm_buf,
                                 const float *X, int n_tokens, int streams,
                                 int dim, float eps);

    // Optional elementwise FP32 sigmoid over a host-visible batch.
    // Supports out == X; rejected calls leave host output unchanged.
    int (*sigmoid_batch)(void *ctx, float *out, const float *X, int count);

    // Optional scaled SiLU over a host-visible hyperconnection low-rank batch.
    // Supports in-place operation; rejected calls leave host data unchanged.
    int (*hyper_connection_scaled_silu_batch)(
        void *ctx, float *values, int count, float scale);

    // Optional hyperconnection sigmoid, stream reduction, and mean over
    // host-visible token-major [tokens, streams, dim] inputs.
    int (*hyper_connection_mix_batch)(
        void *ctx, float *out, const float *norm, const float *gates,
        int n_tokens, int dim, int streams);

    // Optional hyperconnection scatter over host-visible prompt rows:
    // residual += block_out * (2 * sigmoid(inject / streams)).
    int (*hyper_connection_combine_batch)(
        void *ctx, float *residual, const float *block_out,
        const float *inject, int n_tokens, int dim, int streams);

    // Optional stateful SSM convolution followed by Q/K L2 normalization.
    // qkv and conv_state are host-visible; convolution weights stay resident.
    int (*ssm_conv_l2norm_batch)(void *ctx, float *qkv, float *conv_state,
        void *conv_weight_buf, int n_tokens, int qkv_dim, int conv_kernel,
        int num_k_heads, int head_k_dim, float norm_eps);

    // Optional recurrent delta update followed by per-head gated RMSNorm.
    // Inputs are already projected, convolved, and Q/K-normalized. Alpha and
    // beta are raw projection results; the backend applies bias, softplus,
    // decay, and sigmoid. Caller-owned recurrent state is copied back.
    int (*ssm_delta_gate_batch)(void *ctx, float *out, float *state,
        const float *qkv, const float *z, const float *alpha,
        const float *beta, void *norm_buf, void *dt_bias_buf,
        void *a_log_buf, int n_tokens,
        int num_k_heads, int head_k_dim, int num_v_heads, int head_v_dim,
        float q_scale, float norm_eps, int sigmoid_gate);

    // Generic positional-mixing primitives. These operate on host-visible
    // arrays while the backend owns all temporary device storage.
    int (*signed_sqrt_gate)(void *ctx, float *gate, float *gated,
                            const float *key, const float *query,
                            const float *value, int dim, int streams,
                            int reduction_threads);
    int (*dilated_conv_silu)(void *ctx, float *out, const float *current,
                             const float *history, void *weight_buf,
                             int channels, int kernel, int dilation);

    // Optional out = ((RMSNorm(X) * post_scale) * norm_weight), each
    // multiplication rounded separately. norm_buf contains raw FP32 weights.
    // Supports out == X; rejected calls leave host output unchanged.
    int (*rmsnorm_scaled_batch)(void *ctx, float *out, void *norm_buf,
        const float *X, int n_tokens, int dim, float eps, float post_scale);

    // out = (RMSNorm(X) * norm_weight) + residual. The backend preserves its
    // model-graph contraction. Host output may alias X or residual; rejection
    // leaves it unchanged. No session cache or model state is modified.
    int (*rmsnorm_residual_batch)(void *ctx, float *out, void *norm_buf,
        const float *X, const float *residual, int n_tokens, int dim, float eps);

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

    // Compose the routed and dense FFN branches on the device in one call.
    int (*moe_dense_residual_batch)(
        void *ctx, const BnGPUMoEDenseResidualBatch *batch);

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

    // Raw gathered expert FFN outputs, before expert scaling and route weights.
    // Optional; rejected calls leave host output unchanged. up_buf may be NULL
    // for stacked gate/up weights, as with dense_ffn_batch.
    int (*moe_expert_ffn_batch)(void *ctx, float *out,
                                void *gate_buf, void *up_buf, void *down_buf,
                                const float *X, int n_tokens, int dim,
                                int hidden_dim, int gate_type, int up_type,
                                int down_type, int act_type,
                                const BnGPUMoEExpertBatchPlan *plan);

    // Reduce raw experts[nt,k,dim] with scales/weights[nt,k] in slot order.
    // Each expert scale multiplication rounds before its route weight.
    // k=2..15 uses FMA after the first slot; k=1/16 uses separate multiply/add.
    // Supports out == experts. Rejection leaves host output unchanged.
    int (*moe_reduce_batch)(void *ctx, float *out, const float *experts,
        const float *weights, const float *scales, int nt, int k, int dim);

    // Batched routed MoE FFN using monolithic all-expert resident handles.
    // indices/weights are [n_tokens, k] route results. Optional output_scales
    // are applied and rounded before route weighting. Returns the weighted sum
    // in out[n_tokens, dim].
    int (*moe_routed_ffn_batch)(void *ctx, float *out,
                                void *gate_all_buf, void *up_all_buf,
                                void *down_all_buf,
                                const int *indices,
                                const float *weights,
                                const float *output_scales,
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

    // All attention-prefill hooks use attention_window == 0 for full causal
    // history; a positive value limits visible keys for each prompt query.
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
                             float attention_scale, int attention_window);

    // Prepare projected Q/K (optional RMSNorm and NeoX RoPE), then causal
    // attention and optional sigmoid query gate. Gated Q has adjacent
    // [query,gate] vectors per head; q_row_stride includes any row padding.
    // Norm buffers contain FP32 vectors (typed F32 or raw auxiliary uploads).
    // No bias or value normalization is performed. Host out may alias Q;
    // K_out may alias K and receives F32 prepared keys before cache rounding.
    // On failure, host outputs remain untouched. No session KV state changes.
    int (*prefill_attention_prepared)(void *ctx, float *out, float *K_out,
        const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // Prepared attention with unit-weight RMS normalization of projected V.
    // Returns F32 prepared K and normalized V before cache rounding; callers
    // own the session cache. out/K_out/V_out may alias Q/K/V respectively.
    // All three output ranges must be distinct. Failure leaves them untouched.
    int (*prefill_attention_prepared_v)(void *ctx, float *out,
        float *K_out, float *V_out, const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // Backend preparation boundary for shared batched attention. Produces
    // compact Q plus prepared K and optional unit-normalized V, without
    // computing attention or changing session cache state.
    int (*prefill_qkv_prepared)(void *ctx, float *Q_out, float *K_out,
        float *V_out, const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // One-token QK scores over the backend-resident prefix. Q/K are raw
    // projected FP32 rows; optional norm buffers request fused RMSNorm,
    // norm-weight multiplication, and RoPE. The backend commits current K/V
    // and returns packed attention probabilities plus the prepared Q/K rows.
    int (*decode_attention_scores_prepared)(void *ctx, float *scores,
        float *Q_out, float *K_out,
        const float *Q, const float *K, const float *V,
        void *q_norm_buf, void *k_norm_buf,
        const BnGPUAttentionPrefillPlan *plan);

    // Read-only preflight of prefix geometry, cache capacity and RoPE resources.
    // Returns 1 when supported, 0 otherwise. Does not require populated KV rows.
    int (*prefill_attention_prefix_supported)(void *ctx,
        const BnGPUAttentionPrefillPlan *plan);

    // Fused prompt attention + output projection. Optional; backends may keep
    // the attention intermediate resident before applying W_wo.
    int (*prefill_attention_wo)(void *ctx, float *out, void *wo_buf,
                                const float *Q, const float *K,
                                const float *V, int n_tokens,
                                int n_heads, int n_kv_heads, int head_size,
                                int kv_mul, int kv_dim, int wo_rows,
                                int wo_cols, int wo_type,
                                float attention_scale, int attention_window);

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
                                    float attention_scale, int attention_window);

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
                                    float attention_scale, int attention_window);

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
                                    float attention_scale, int attention_window);

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
                                    void *attn_post_norm_buf,
                                    void *ffn_post_norm_buf,
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
                                    int qk_norm_per_head, int normalize_v,
                                    float norm_eps,
                                    int pos0, int rope_dims,
                                    size_t rope_freq_offset,
                                    uint32_t kv_cache_off,
                                    int kv_cache_stride,
                                    float attention_scale,
                                    float layer_output_scale,
                                    int attention_window,
                                    int final_ffn_last_row_only);

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
                                    float expert_weights_scale, int attention_window);

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
                                    int sigmoid_gate, float norm_eps,
                                    int *did_ffn);

    // GPU-resident forward pass: execute a backend-private lowered command list
    // as a single submission. All intermediate buffers stay on GPU. Only
    // readback_buf is copied to out_host.
    // Returns 0 on success, -1 on error (caller should fall back to CPU).
    int (*execute)(void *ctx, const void *ops, int n_ops,
                   int readback_buf, float *out_host, int out_len);

    // Initialize GPU-resident activation buffers from a backend-neutral plan.
    // Must be called after weight upload, before execute().
    // Returns 0 on success.
    int (*init_activations)(void *ctx, const BnGPUActivationPlan *plan);

    // Clear request-local mutable activation state while retaining allocated
    // backend buffers and immutable uploaded model data. Optional.
    int (*reset_activations)(void *ctx);

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

    // Prepare backend-owned memory and runtime state used by CPU-orchestrated
    // operations. Optional and idempotent.
    int (*prepare_cpu_operations)(void *ctx);

    // Select the backend's prepared native-quant weight layout before model
    // buffers are uploaded. Optional; the setting is backend-instance local.
    void (*configure_prepared_native_quant)(void *ctx, int enabled);

    void *ctx;  // opaque backend context

    // Capability flags (set by backend, checked by transformer)
    uint64_t caps;

    // Maximum expert count supported by the backend's GPU router. 0 means
    // the backend does not impose a shape limit beyond its capability bit.
    int max_moe_route_experts;

    // Concrete backend identity for planning/debugging. Future backends should
    // set this instead of relying on capability inference alone.
    BnGPUBackendKind kind;
    BnBackendRuntimePolicy runtime_policy;

    // Maximum storage-buffer binding size in bytes. 0 = unknown; callers
    // should use a conservative fallback when deciding whether to bind a
    // large weight buffer in a GPU-resident graph.
    size_t max_storage_binding_size;
};

int bn_gpu_backend_capture_runtime_policy(BnGPUBackend *gpu);
int bn_gpu_backend_runtime_policy_init(BnBackendRuntimePolicy *policy);
int bn_gpu_backend_capture_runtime_policy_from(
    BnGPUBackend *gpu, const BnBackendRuntimePolicy *policy);
void bn_gpu_backend_release_runtime_policy(BnGPUBackend *gpu);

static inline int bn_gpu_backend_can_create_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create;
}

static inline void bn_gpu_backend_configure_prepared_native_quant(
    BnGPUBackend *gpu, int enabled) {
    if (gpu && gpu->configure_prepared_native_quant)
        gpu->configure_prepared_native_quant(gpu->ctx, enabled);
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

static inline int bn_gpu_backend_can_create_borrowed_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_borrowed;
}

static inline int bn_gpu_backend_can_create_native_matvec_borrowed_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_native_matvec_borrowed;
}

static inline int bn_gpu_backend_native_matvec_borrowed_supported(
    const BnGPUBackend *gpu, size_t size, int type, int rows, int cols) {
    return gpu && gpu->buffer_create_native_matvec_borrowed &&
           gpu->native_matvec_borrowed_supported &&
           gpu->native_matvec_borrowed_supported(
               gpu->ctx, size, type, rows, cols);
}

static inline size_t bn_gpu_backend_buffer_cache_charge(
    const BnGPUBackend *gpu, size_t size, int type, int rows, int cols) {
    if (gpu && gpu->buffer_cache_charge)
        return gpu->buffer_cache_charge(gpu->ctx, size, type, rows, cols);
    return size;
}

static inline int bn_gpu_backend_can_create_kquant_f32_cache_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_kquant_f32_cache;
}

static inline int bn_gpu_backend_can_create_f16_cache_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_f16_cache;
}

static inline size_t bn_gpu_backend_f16_cache_extra_bytes(
    const BnGPUBackend *gpu, int type, int rows, int cols) {
    if (gpu && gpu->buffer_f16_cache_extra_bytes)
        return gpu->buffer_f16_cache_extra_bytes(gpu->ctx, type, rows, cols);
    if (rows <= 0 || cols <= 0) return 0;
    if ((size_t)rows > SIZE_MAX / (size_t)cols / sizeof(uint16_t))
        return SIZE_MAX;
    return (size_t)rows * (size_t)cols * sizeof(uint16_t);
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

static inline int bn_gpu_backend_can_matvec(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matvec;
}

static inline int bn_gpu_backend_can_matmul(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matmul;
}

static inline int bn_gpu_backend_can_matvec_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matvec_batch;
}

static inline int bn_gpu_backend_can_matmul_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matmul_batch;
}

static inline int bn_gpu_backend_can_rmsnorm_residual_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_residual_batch;
}

static inline int bn_gpu_backend_can_rmsnorm_scaled_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_scaled_batch;
}

static inline int bn_gpu_backend_can_rmsnorm_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_batch;
}

static inline int bn_gpu_backend_moe_dense_residual_batch(
    const BnGPUBackend *gpu, const BnGPUMoEDenseResidualBatch *batch) {
    if (!gpu || !gpu->moe_dense_residual_batch) return -1;
    return gpu->moe_dense_residual_batch(gpu->ctx, batch);
}

static inline int bn_gpu_backend_can_rmsnorm_grouped_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->rmsnorm_grouped_batch;
}

static inline int bn_gpu_backend_can_sigmoid_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->sigmoid_batch;
}

static inline int bn_gpu_backend_can_hyper_connection_scaled_silu_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->hyper_connection_scaled_silu_batch;
}

static inline int bn_gpu_backend_can_hyper_connection_mix_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->hyper_connection_mix_batch;
}

static inline int bn_gpu_backend_can_hyper_connection_combine_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->hyper_connection_combine_batch;
}

static inline int bn_gpu_backend_can_ssm_delta_gate_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->ssm_delta_gate_batch;
}

static inline int bn_gpu_backend_can_ssm_conv_l2norm_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->ssm_conv_l2norm_batch;
}

static inline int bn_gpu_backend_can_dense_ffn(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn;
}

static inline int bn_gpu_backend_can_moe_reduce_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->moe_reduce_batch;
}

static inline int bn_gpu_backend_can_moe_expert_ffn_batch(const BnGPUBackend *gpu) {
    return gpu && gpu->moe_expert_ffn_batch;
}

static inline int bn_gpu_backend_can_dense_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_batch;
}

static inline int bn_gpu_backend_can_dense_ffn_batch_norm(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_batch_norm;
}

static inline int bn_gpu_backend_can_dense_ffn_batch_norm_resid(
    const BnGPUBackend *gpu) {
    return gpu && gpu->dense_ffn_batch_norm_resid;
}

static inline int bn_gpu_backend_can_moe_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_ffn_batch;
}

static inline int bn_gpu_backend_can_moe_routed_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_routed_ffn_batch;
}

static inline int bn_gpu_backend_can_moe_route_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_route_batch;
}

static inline int bn_gpu_backend_can_moe_route_routed_ffn_batch(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_route_routed_ffn_batch;
}

static inline int bn_gpu_backend_can_moe_route_routed_ffn_batch_norm_resid(
    const BnGPUBackend *gpu) {
    return gpu && gpu->moe_route_routed_ffn_batch_norm_resid;
}

static inline int bn_gpu_backend_can_prefill_attention(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention;
}

static inline int bn_gpu_backend_can_prefill_attention_prepared_v(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_prepared_v;
}

static inline int bn_gpu_backend_can_prefill_attention_prepared(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_prepared;
}

static inline int bn_gpu_backend_can_decode_attention_scores_prepared(
    const BnGPUBackend *gpu) {
    return gpu && gpu->decode_attention_scores_prepared;
}

static inline int bn_gpu_backend_can_check_prefill_prefix(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_prefix_supported;
}

static inline int bn_gpu_backend_prefill_prefix_supported(
    const BnGPUBackend *gpu, const BnGPUAttentionPrefillPlan *plan) {
    return bn_gpu_backend_can_check_prefill_prefix(gpu) && plan &&
           gpu->prefill_attention_prefix_supported(gpu->ctx, plan) == 1;
}

static inline int bn_gpu_backend_can_prefill_attention_wo(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_attention_wo;
}

static inline int bn_gpu_backend_can_prefill_qkv_attention_wo(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_qkv_attention_wo;
}

static inline int bn_gpu_backend_can_prefill_qkv_attention_wo_norm_resid(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_qkv_attention_wo_norm_resid;
}

static inline int bn_gpu_backend_can_prefill_dense_layer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_dense_layer;
}

static inline int bn_gpu_backend_can_prefill_moe_layer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_moe_layer;
}

static inline int bn_gpu_backend_can_prefill_ssm_layer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_ssm_layer;
}

static inline int bn_gpu_backend_can_destroy_buffer(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_destroy;
}

static inline int bn_gpu_backend_can_execute(
    const BnGPUBackend *gpu) {
    return gpu && gpu->execute;
}

static inline int bn_gpu_backend_can_init_activations(
    const BnGPUBackend *gpu) {
    return gpu && gpu->init_activations;
}

static inline int bn_gpu_backend_can_free_activations(
    const BnGPUBackend *gpu) {
    return gpu && gpu->free_activations;
}

static inline int bn_gpu_backend_can_write_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->write_activation;
}

static inline int bn_gpu_backend_can_read_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->read_activation;
}

static inline int bn_gpu_backend_can_argmax_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->argmax_activation;
}

static inline int bn_gpu_backend_can_matvec_argmax_activation(
    const BnGPUBackend *gpu) {
    return gpu && gpu->matvec_argmax_activation;
}

static inline int bn_gpu_backend_can_query_memory(
    const BnGPUBackend *gpu) {
    return gpu && gpu->memory_info;
}

static inline int bn_gpu_backend_prepare_cpu_operations(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prepare_cpu_operations
        ? gpu->prepare_cpu_operations(gpu->ctx) : 0;
}

static inline int bn_gpu_backend_has_cap(const BnGPUBackend *gpu,
                                         uint64_t cap) {
    return gpu && ((gpu->caps & cap) != 0);
}

static inline size_t bn_gpu_backend_max_storage_binding_size(
    const BnGPUBackend *gpu) {
    return gpu ? gpu->max_storage_binding_size : 0;
}

static inline int bn_gpu_backend_moe_route_shape_supported(
    const BnGPUBackend *gpu, int n_experts) {
    return gpu && n_experts > 0 &&
           (gpu->max_moe_route_experts <= 0 ||
            n_experts <= gpu->max_moe_route_experts);
}

static inline void *bn_gpu_backend_create_buffer(BnGPUBackend *gpu,
                                                 const void *data,
                                                 size_t size,
                                                 int type,
                                                 int rows,
                                                 int cols) {
    if (!bn_gpu_backend_can_create_buffer(gpu))
        return NULL;
    return gpu->buffer_create(gpu->ctx, data, size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_quant_only_buffer(BnGPUBackend *gpu,
                                                           const void *data,
                                                           size_t size,
                                                           int type,
                                                           int rows,
                                                           int cols) {
    if (!bn_gpu_backend_can_create_quant_only_buffer(gpu))
        return NULL;
    return gpu->buffer_create_quant_only(gpu->ctx, data, size, type, rows,
                                         cols);
}

static inline void *bn_gpu_backend_create_borrowed_buffer(
    BnGPUBackend *gpu,
    const void *data,
    size_t size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_borrowed_buffer(gpu))
        return NULL;
    return gpu->buffer_create_borrowed(gpu->ctx, data, size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_native_matvec_borrowed_buffer(
    BnGPUBackend *gpu, const void *data, size_t size,
    int type, int rows, int cols) {
    if (!bn_gpu_backend_can_create_native_matvec_borrowed_buffer(gpu))
        return NULL;
    return gpu->buffer_create_native_matvec_borrowed(
        gpu->ctx, data, size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_kquant_f32_cache_buffer(
    BnGPUBackend *gpu,
    const void *data,
    size_t size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_kquant_f32_cache_buffer(gpu))
        return NULL;
    return gpu->buffer_create_kquant_f32_cache(gpu->ctx, data, size, type,
                                               rows, cols);
}

static inline void *bn_gpu_backend_create_f16_cache_buffer(BnGPUBackend *gpu,
                                                          const void *data,
                                                          size_t size,
                                                          int type,
                                                          int rows,
                                                          int cols) {
    if (!bn_gpu_backend_can_create_f16_cache_buffer(gpu))
        return NULL;
    return gpu->buffer_create_f16_cache(gpu->ctx, data, size, type, rows,
                                        cols);
}

static inline void *bn_gpu_backend_create_stacked2_buffer(
    BnGPUBackend *gpu,
    const void *first_data,
    size_t first_size,
    const void *second_data,
    size_t second_size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_stacked2_buffer(gpu))
        return NULL;
    return gpu->buffer_create_stacked2(gpu->ctx, first_data, first_size,
                                       second_data, second_size, type, rows,
                                       cols);
}

static inline void *bn_gpu_backend_create_biased_buffer(BnGPUBackend *gpu,
                                                       const void *data,
                                                       size_t size,
                                                       int type,
                                                       int rows,
                                                       int cols,
                                                       const float *bias,
                                                       size_t bias_size) {
    if (!bn_gpu_backend_can_create_biased_buffer(gpu))
        return NULL;
    return gpu->buffer_create_biased(gpu->ctx, data, size, type, rows, cols,
                                     bias, bias_size);
}

static inline void *bn_gpu_backend_create_stacked3_buffer(
    BnGPUBackend *gpu,
    const void *first_data,
    size_t first_size,
    const void *second_data,
    size_t second_size,
    const void *third_data,
    size_t third_size,
    int type,
    int rows,
    int cols) {
    if (!bn_gpu_backend_can_create_stacked3_buffer(gpu))
        return NULL;
    return gpu->buffer_create_stacked3(gpu->ctx, first_data, first_size,
                                       second_data, second_size, third_data,
                                       third_size, type, rows, cols);
}

static inline void *bn_gpu_backend_create_stacked3_biased_buffer(
    BnGPUBackend *gpu,
    const void *first_data,
    size_t first_size,
    const void *second_data,
    size_t second_size,
    const void *third_data,
    size_t third_size,
    int type,
    int rows,
    int cols,
    const float *bias,
    size_t bias_size) {
    if (!bn_gpu_backend_can_create_stacked3_biased_buffer(gpu))
        return NULL;
    return gpu->buffer_create_stacked3_biased(
        gpu->ctx, first_data, first_size, second_data, second_size,
        third_data, third_size, type, rows, cols, bias, bias_size);
}

static inline int bn_gpu_backend_matvec(const BnGPUBackend *gpu,
                                        float *out,
                                        void *buffer,
                                        const float *x,
                                        int rows,
                                        int cols,
                                        int type) {
    if (!bn_gpu_backend_can_matvec(gpu))
        return -1;
    return gpu->matvec(gpu->ctx, out, buffer, x, rows, cols, type);
}

static inline int bn_gpu_backend_matmul(const BnGPUBackend *gpu,
                                        float *out,
                                        void *buffer,
                                        const float *X,
                                        int rows,
                                        int cols,
                                        int n_tokens,
                                        int type) {
    if (!bn_gpu_backend_can_matmul(gpu))
        return -1;
    return gpu->matmul(gpu->ctx, out, buffer, X, rows, cols, n_tokens,
                       type);
}

static inline int bn_gpu_backend_sigmoid_batch(const BnGPUBackend *gpu,
                                                float *out,
                                                const float *X,
                                                int count) {
    if (!bn_gpu_backend_can_sigmoid_batch(gpu))
        return -1;
    return gpu->sigmoid_batch(gpu->ctx, out, X, count);
}

static inline int bn_gpu_backend_hyper_connection_scaled_silu_batch(
    const BnGPUBackend *gpu, float *values, int count, float scale) {
    if (!bn_gpu_backend_can_hyper_connection_scaled_silu_batch(gpu))
        return -1;
    return gpu->hyper_connection_scaled_silu_batch(
        gpu->ctx, values, count, scale);
}

static inline int bn_gpu_backend_hyper_connection_mix_batch(
    const BnGPUBackend *gpu, float *out, const float *norm,
    const float *gates, int n_tokens, int dim, int streams) {
    if (!bn_gpu_backend_can_hyper_connection_mix_batch(gpu))
        return -1;
    return gpu->hyper_connection_mix_batch(
        gpu->ctx, out, norm, gates, n_tokens, dim, streams);
}

static inline int bn_gpu_backend_hyper_connection_combine_batch(
    const BnGPUBackend *gpu, float *residual, const float *block_out,
    const float *inject, int n_tokens, int dim, int streams) {
    if (!bn_gpu_backend_can_hyper_connection_combine_batch(gpu))
        return -1;
    return gpu->hyper_connection_combine_batch(
        gpu->ctx, residual, block_out, inject, n_tokens, dim, streams);
}

static inline int bn_gpu_backend_ssm_delta_gate_batch(
    const BnGPUBackend *gpu, float *out, float *state, const float *qkv,
    const float *z, const float *alpha, const float *beta, void *norm_buf,
    void *dt_bias_buf, void *a_log_buf, int n_tokens, int num_k_heads,
    int head_k_dim, int num_v_heads,
    int head_v_dim, float q_scale, float norm_eps, int sigmoid_gate) {
    if (!bn_gpu_backend_can_ssm_delta_gate_batch(gpu))
        return -1;
    return gpu->ssm_delta_gate_batch(
        gpu->ctx, out, state, qkv, z, alpha, beta, norm_buf, dt_bias_buf,
        a_log_buf, n_tokens,
        num_k_heads, head_k_dim, num_v_heads, head_v_dim, q_scale,
        norm_eps, sigmoid_gate);
}

static inline int bn_gpu_backend_ssm_conv_l2norm_batch(
    const BnGPUBackend *gpu, float *qkv, float *conv_state,
    void *conv_weight_buf, int n_tokens, int qkv_dim, int conv_kernel,
    int num_k_heads, int head_k_dim, float norm_eps) {
    if (!bn_gpu_backend_can_ssm_conv_l2norm_batch(gpu))
        return -1;
    return gpu->ssm_conv_l2norm_batch(
        gpu->ctx, qkv, conv_state, conv_weight_buf, n_tokens, qkv_dim,
        conv_kernel, num_k_heads, head_k_dim, norm_eps);
}

static inline int bn_gpu_backend_matvec_batch(
    const BnGPUBackend *gpu,
    const BnGPUMatvecOp *ops,
    int n_ops,
    const float *x,
    int x_cols) {
    if (!bn_gpu_backend_can_matvec_batch(gpu))
        return -1;
    return gpu->matvec_batch(gpu->ctx, ops, n_ops, x, x_cols);
}

static inline int bn_gpu_backend_matmul_batch(
    const BnGPUBackend *gpu,
    const BnGPUMatvecOp *ops,
    int n_ops,
    const float *X,
    int n_tokens,
    int x_cols) {
    if (!bn_gpu_backend_can_matmul_batch(gpu))
        return -1;
    return gpu->matmul_batch(gpu->ctx, ops, n_ops, X, n_tokens, x_cols);
}

static inline int bn_gpu_backend_rmsnorm_residual_batch(const BnGPUBackend *gpu,
    float *out, void *norm_buf, const float *X, const float *residual,
    int n_tokens, int dim, float eps) {
    if (!bn_gpu_backend_can_rmsnorm_residual_batch(gpu)) return -1;
    return gpu->rmsnorm_residual_batch(gpu->ctx, out, norm_buf, X, residual,
                                      n_tokens, dim, eps);
}

static inline int bn_gpu_backend_rmsnorm_scaled_batch(const BnGPUBackend *gpu,
    float *out, void *norm, const float *X, int nt, int dim, float eps,
    float post_scale) {
    if (!bn_gpu_backend_can_rmsnorm_scaled_batch(gpu)) return -1;
    return gpu->rmsnorm_scaled_batch(gpu->ctx, out, norm, X, nt, dim, eps,
                                    post_scale);
}

static inline int bn_gpu_backend_rmsnorm_batch(
    const BnGPUBackend *gpu, float *out, void *norm_buf,
    const float *X, int n_tokens, int dim, float eps) {
    if (!bn_gpu_backend_can_rmsnorm_batch(gpu)) return -1;
    return gpu->rmsnorm_batch(gpu->ctx, out, norm_buf, X, n_tokens, dim, eps);
}

static inline int bn_gpu_backend_rmsnorm_grouped_batch(
    const BnGPUBackend *gpu, float *out, void *norm_buf, const float *X,
    int n_tokens, int streams, int dim, float eps) {
    if (!bn_gpu_backend_can_rmsnorm_grouped_batch(gpu)) return -1;
    return gpu->rmsnorm_grouped_batch(gpu->ctx, out, norm_buf, X, n_tokens,
                                      streams, dim, eps);
}

static inline int bn_gpu_backend_dense_ffn(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *x,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    if (!bn_gpu_backend_can_dense_ffn(gpu))
        return -1;
    return gpu->dense_ffn(gpu->ctx, out, gate_buf, up_buf, down_buf, x,
                          dim, hidden_dim, gate_type, up_type, down_type,
                          act_type);
}

static inline int bn_gpu_backend_moe_reduce_batch(const BnGPUBackend *gpu,
    float *out, const float *experts, const float *weights, const float *scales,
    int nt, int k, int dim) {
    if (!bn_gpu_backend_can_moe_reduce_batch(gpu)) return -1;
    return gpu->moe_reduce_batch(gpu->ctx, out, experts, weights, scales, nt, k, dim);
}

static inline int bn_gpu_backend_moe_expert_ffn_batch(
    const BnGPUBackend *gpu, float *out, void *gate, void *up, void *down,
    const float *X, int nt, int dim, int hidden, int gate_type,
    int up_type, int down_type, int act_type, const BnGPUMoEExpertBatchPlan *plan) {
    if (!bn_gpu_backend_can_moe_expert_ffn_batch(gpu)) return -1;
    return gpu->moe_expert_ffn_batch(gpu->ctx, out, gate, up, down, X, nt,
        dim, hidden, gate_type, up_type, down_type, act_type, plan);
}

static inline int bn_gpu_backend_dense_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    if (!bn_gpu_backend_can_dense_ffn_batch(gpu))
        return -1;
    return gpu->dense_ffn_batch(gpu->ctx, out, gate_buf, up_buf, down_buf,
                                X, n_tokens, dim, hidden_dim, gate_type,
                                up_type, down_type, act_type);
}

static inline int bn_gpu_backend_dense_ffn_batch_norm(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps) {
    if (!bn_gpu_backend_can_dense_ffn_batch_norm(gpu))
        return -1;
    return gpu->dense_ffn_batch_norm(
        gpu->ctx, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens,
        dim, hidden_dim, gate_type, up_type, down_type, act_type, norm_eps);
}

static inline int bn_gpu_backend_dense_ffn_batch_norm_resid(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps) {
    if (!bn_gpu_backend_can_dense_ffn_batch_norm_resid(gpu))
        return -1;
    return gpu->dense_ffn_batch_norm_resid(
        gpu->ctx, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens,
        dim, hidden_dim, gate_type, up_type, down_type, act_type, norm_eps);
}

static inline int bn_gpu_backend_moe_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    const BnGPUMoEPrefillExpert *experts,
    int n_experts,
    const int *expert_offsets,
    const int *expert_counts,
    const int *token_ids,
    const float *weights,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type) {
    if (!bn_gpu_backend_can_moe_ffn_batch(gpu))
        return -1;
    return gpu->moe_ffn_batch(
        gpu->ctx, out, experts, n_experts, expert_offsets, expert_counts,
        token_ids, weights, X, n_tokens, dim, hidden_dim, gate_type, up_type,
        down_type, act_type, shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, shared_hidden_dim, shared_gate_type,
        shared_up_type, shared_down_type);
}

static inline int bn_gpu_backend_moe_routed_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const int *indices,
    const float *weights,
    const float *output_scales,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    if (!bn_gpu_backend_can_moe_routed_ffn_batch(gpu))
        return -1;
    return gpu->moe_routed_ffn_batch(
        gpu->ctx, out, gate_all_buf, up_all_buf, down_all_buf, indices,
        weights, output_scales, X, n_tokens, dim, hidden_dim, n_experts, k, gate_type,
        up_type, down_type, act_type);
}

static inline int bn_gpu_backend_moe_route_batch(
    const BnGPUBackend *gpu,
    int *indices,
    float *weights,
    void *router_buf,
    const float *X,
    int n_tokens,
    int dim,
    int n_experts,
    int k,
    int norm_topk_prob,
    float expert_weights_scale) {
    if (!bn_gpu_backend_can_moe_route_batch(gpu))
        return -1;
    return gpu->moe_route_batch(gpu->ctx, indices, weights, router_buf, X,
                                n_tokens, dim, n_experts, k,
                                norm_topk_prob, expert_weights_scale);
}

static inline int bn_gpu_backend_moe_route_routed_ffn_batch(
    const BnGPUBackend *gpu,
    float *out,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int norm_topk_prob,
    float expert_weights_scale) {
    if (!bn_gpu_backend_can_moe_route_routed_ffn_batch(gpu))
        return -1;
    return gpu->moe_route_routed_ffn_batch(
        gpu->ctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        X, n_tokens, dim, hidden_dim, n_experts, k, gate_type, up_type,
        down_type, act_type, norm_topk_prob, expert_weights_scale);
}

static inline int bn_gpu_backend_moe_route_routed_ffn_batch_norm_resid(
    const BnGPUBackend *gpu,
    float *out,
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
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    float norm_eps,
    int norm_topk_prob,
    float expert_weights_scale) {
    if (!bn_gpu_backend_can_moe_route_routed_ffn_batch_norm_resid(gpu))
        return -1;
    return gpu->moe_route_routed_ffn_batch_norm_resid(
        gpu->ctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, norm_buf, X, n_tokens, dim, hidden_dim,
        n_experts, k, gate_type, up_type, down_type, act_type,
        shared_hidden_dim, shared_gate_type, shared_up_type,
        shared_down_type, norm_eps, norm_topk_prob, expert_weights_scale);
}

static inline int bn_gpu_backend_prefill_attention(
    const BnGPUBackend *gpu,
    float *out,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_attention(gpu))
        return -1;
    return gpu->prefill_attention(gpu->ctx, out, Q, K, V, n_tokens,
                                  n_heads, n_kv_heads, head_size, kv_mul,
                                  kv_dim, attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_attention_prepared_v(
    const BnGPUBackend *gpu, float *out, float *K_out, float *V_out,
    const float *Q, const float *K, const float *V, void *q_norm_buf,
    void *k_norm_buf, const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_prefill_attention_prepared_v(gpu)) return -1;
    return gpu->prefill_attention_prepared_v(gpu->ctx, out, K_out, V_out,
        Q, K, V, q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_prefill_attention_prepared(
    const BnGPUBackend *gpu, float *out, float *K_out,
    const float *Q, const float *K, const float *V,
    void *q_norm_buf, void *k_norm_buf, const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_prefill_attention_prepared(gpu)) return -1;
    return gpu->prefill_attention_prepared(gpu->ctx, out, K_out, Q, K, V,
        q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_can_prefill_qkv_prepared(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_qkv_prepared;
}

static inline int bn_gpu_backend_prefill_qkv_prepared(
    const BnGPUBackend *gpu, float *Q_out, float *K_out, float *V_out,
    const float *Q, const float *K, const float *V,
    void *q_norm_buf, void *k_norm_buf,
    const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_prefill_qkv_prepared(gpu)) return -1;
    return gpu->prefill_qkv_prepared(gpu->ctx, Q_out, K_out, V_out,
        Q, K, V, q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_decode_attention_scores_prepared(
    const BnGPUBackend *gpu, float *scores, float *Q_out, float *K_out,
    const float *Q, const float *K, const float *V,
    void *q_norm_buf, void *k_norm_buf,
    const BnGPUAttentionPrefillPlan *plan) {
    if (!bn_gpu_backend_can_decode_attention_scores_prepared(gpu)) return -1;
    return gpu->decode_attention_scores_prepared(
        gpu->ctx, scores, Q_out, K_out, Q, K, V,
        q_norm_buf, k_norm_buf, plan);
}

static inline int bn_gpu_backend_prefill_attention_wo(
    const BnGPUBackend *gpu,
    float *out,
    void *wo_buf,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int wo_rows,
    int wo_cols,
    int wo_type,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_attention_wo(gpu))
        return -1;
    return gpu->prefill_attention_wo(gpu->ctx, out, wo_buf, Q, K, V,
                                     n_tokens, n_heads, n_kv_heads,
                                     head_size, kv_mul, kv_dim, wo_rows,
                                     wo_cols, wo_type, attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_qkv_attention_wo(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_qkv_attention_wo(gpu))
        return -1;
    return gpu->prefill_qkv_attention_wo(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, q_norm_buf, k_norm_buf, X,
        K_out, V_out, n_tokens, dim, n_heads, n_kv_heads, head_size, kv_mul,
        kv_dim, qk_rows, qk_type, wv_rows, wv_type, wo_rows, wo_cols,
        wo_type, qk_norm_per_head, norm_eps, pos0, rope_dims,
        attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_qkv_attention_wo_norm_resid(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *attn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_qkv_attention_wo_norm_resid(gpu))
        return -1;
    return gpu->prefill_qkv_attention_wo_norm_resid(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, attn_norm_buf, q_norm_buf,
        k_norm_buf, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads,
        head_size, kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type,
        wo_rows, wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, attention_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_dense_layer(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *attn_post_norm_buf,
    void *ffn_post_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int qk_norm_per_head,
    int normalize_v,
    float norm_eps,
    int pos0,
    int rope_dims,
    size_t rope_freq_offset,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale,
    float layer_output_scale, int attention_window,
    int final_ffn_last_row_only) {
    if (!bn_gpu_backend_can_prefill_dense_layer(gpu))
        return -1;
    return gpu->prefill_dense_layer(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, gate_buf, up_buf, down_buf,
        attn_norm_buf, ffn_norm_buf, attn_post_norm_buf,
        ffn_post_norm_buf, q_norm_buf, k_norm_buf, q_bias_buf,
        k_bias_buf, v_bias_buf, X, K_out, V_out, n_tokens, dim, hidden_dim,
        n_heads, n_kv_heads, head_size, kv_mul, kv_dim, qk_rows, qk_type,
        wv_rows, wv_type, wo_rows, wo_cols, wo_type, gate_type, up_type,
        down_type, act_type, qk_norm_per_head, normalize_v, norm_eps,
        pos0, rope_dims,
        rope_freq_offset, kv_cache_off, kv_cache_stride, attention_scale,
        layer_output_scale, attention_window, final_ffn_last_row_only);
}

static inline int bn_gpu_backend_prefill_moe_layer(
    const BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int moe_hidden_dim,
    int n_experts,
    int experts_active,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale,
    int norm_topk_prob,
    float expert_weights_scale, int attention_window) {
    if (!bn_gpu_backend_can_prefill_moe_layer(gpu))
        return -1;
    return gpu->prefill_moe_layer(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, router_buf, gate_all_buf,
        up_all_buf, down_all_buf, shared_gate_buf, shared_up_buf,
        shared_down_buf, shared_gate_weight_buf, attn_norm_buf, ffn_norm_buf,
        q_norm_buf, k_norm_buf, q_bias_buf, k_bias_buf, v_bias_buf,
        X, K_out, V_out, n_tokens, dim, moe_hidden_dim, n_experts,
        experts_active, n_heads, n_kv_heads, head_size, kv_mul, kv_dim,
        qk_rows, qk_type, wv_rows, wv_type, wo_rows, wo_cols, wo_type,
        gate_type, up_type, down_type, act_type, shared_hidden_dim,
        shared_gate_type, shared_up_type, shared_down_type, qk_norm_per_head,
        norm_eps, pos0, rope_dims, kv_cache_off, kv_cache_stride,
        attention_scale, norm_topk_prob, expert_weights_scale, attention_window);
}

static inline int bn_gpu_backend_prefill_ssm_layer(
    const BnGPUBackend *gpu,
    float *out,
    void *wqkv_buf,
    void *wz_buf,
    void *alpha_buf,
    void *beta_buf,
    void *qkvz_stacked_buf,
    void *ab_stacked_buf,
    void *ssm_out_buf,
    void *attn_norm_buf,
    void *conv1d_buf,
    void *dt_bias_buf,
    void *a_log_buf,
    void *ssm_norm_buf,
    void *ffn_gate_buf,
    void *ffn_up_buf,
    void *ffn_down_buf,
    void *ffn_norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int qkv_dim,
    int inner_dim,
    int num_k_heads,
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int conv_kernel,
    int ssm_idx,
    int wqkv_type,
    int wz_type,
    int alpha_type,
    int beta_type,
    int out_type,
    int hidden_dim,
    int ffn_gate_type,
    int ffn_up_type,
    int ffn_down_type,
    int act_type,
    int sigmoid_gate,
    float norm_eps,
    int *did_ffn) {
    if (!bn_gpu_backend_can_prefill_ssm_layer(gpu))
        return -1;
    return gpu->prefill_ssm_layer(
        gpu->ctx, out, wqkv_buf, wz_buf, alpha_buf, beta_buf,
        qkvz_stacked_buf, ab_stacked_buf, ssm_out_buf, attn_norm_buf,
        conv1d_buf, dt_bias_buf, a_log_buf, ssm_norm_buf, ffn_gate_buf,
        ffn_up_buf, ffn_down_buf, ffn_norm_buf, X, n_tokens, dim, qkv_dim,
        inner_dim, num_k_heads, head_k_dim, num_v_heads, head_v_dim,
        conv_kernel, ssm_idx, wqkv_type, wz_type, alpha_type, beta_type,
        out_type, hidden_dim, ffn_gate_type, ffn_up_type, ffn_down_type,
        act_type, sigmoid_gate, norm_eps, did_ffn);
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

static inline int bn_gpu_backend_init_activations(const BnGPUBackend *gpu,
                                                  const BnGPUActivationPlan *plan) {
    if (!bn_gpu_backend_can_init_activations(gpu))
        return -1;
    return gpu->init_activations(gpu->ctx, plan);
}

static inline int bn_gpu_backend_reset_activations(const BnGPUBackend *gpu) {
    return gpu && gpu->reset_activations
        ? gpu->reset_activations(gpu->ctx) : 0;
}

static inline void bn_gpu_backend_free_activations(const BnGPUBackend *gpu) {
    if (bn_gpu_backend_can_free_activations(gpu))
        gpu->free_activations(gpu->ctx);
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

static inline int bn_gpu_backend_argmax_activation(
    const BnGPUBackend *gpu,
    int buf_idx,
    int n,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token) {
    if (!bn_gpu_backend_can_argmax_activation(gpu))
        return -1;
    return gpu->argmax_activation(gpu->ctx, buf_idx, n, penalty_tokens,
                                  n_penalty_tokens, repeat_penalty,
                                  out_token);
}

static inline int bn_gpu_backend_matvec_argmax_activation(
    const BnGPUBackend *gpu,
    void *W_buf,
    int type,
    int rows,
    int cols,
    int buf_idx,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token) {
    if (!bn_gpu_backend_can_matvec_argmax_activation(gpu))
        return -1;
    return gpu->matvec_argmax_activation(
        gpu->ctx, W_buf, type, rows, cols, buf_idx, penalty_tokens,
        n_penalty_tokens, repeat_penalty, out_token);
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
#define BN_GPU_CAP_MOE_ROUTED_FFN (1u << 11) // resident decode route + routed FFN graph ops
#define BN_GPU_CAP_BORROWED_WEIGHT_BUFFERS (1u << 12) // weight handles may borrow immutable host-mapped storage
#define BN_GPU_CAP_MOE_ROUTED_KQUANT_DOWN_CACHE (1u << 13) // routed K-quant down cache strategy available
#define BN_GPU_CAP_MOE_ROUTED_NATIVE_QUANT (1u << 14) // native-quant routed FFN strategy available
#define BN_GPU_CAP_DECODE_GRAPH_CACHE (1u << 15) // lowered decode op streams may be replayed
#define BN_GPU_CAP_LARGE_GRAPH_NATIVE (1u << 16) // large dense/hybrid graph execution available
#define BN_GPU_CAP_SSM_GRAPH (1u << 17) // state-space graph operations available
#define BN_GPU_CAP_PER_LAYER_INPUT_GRAPH (1u << 18) // per-layer input adapter graph composition available
#define BN_GPU_CAP_REFERENCE_ATTENTION (1u << 19) // shared-value attention preserves reference trajectory semantics
#define BN_GPU_CAP_REFERENCE_ATTENTION_FALLBACK (1u << 20) // backend supports reference-attention CPU handoff
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT (1u << 21) // backend supports prepared weights with block-quantized activations
#define BN_GPU_CAP_MOE_EXPERT_GRAPH (1u << 22) // CPU-routed selected-expert graph execution available
#define BN_GPU_CAP_HYBRID_SSM_MOE_GRAPH (1u << 23) // combined SSM + routed-MoE graph is coherent
#define BN_GPU_CAP_MOE_ROUTED_LOWBIT_BLOCK32 (1u << 24) // routed block32 low-bit FFN kernels available
#define BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32 (1u << 25) // dense residual low-bit FFN graph available
#define BN_GPU_CAP_REFERENCE_RECURRENT (1u << 26) // recurrent graph preserves reference trajectory semantics
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION (1u << 27) // prepared native-quant attention projections are coherent
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN (1u << 28) // prepared native-quant FFN gate/up is coherent with per-layer input composition
#define BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN (1u << 29) // prepared native-quant FFN down is coherent with per-layer input composition
#define BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH (1u << 30) // ordinary attention graph preserves reference trajectory semantics
#define BN_GPU_CAP_REFERENCE_ATTENTION_TOKEN_FALLBACK (1u << 31) // reference attention requires whole-token CPU orchestration
#define BN_GPU_CAP_HYPER_CONNECTION_GRAPH (UINT64_C(1) << 32) // hyper-connection mix/combine graph operations available
#define BN_GPU_CAP_MOE_ROUTED_MIDBIT_BLOCK32_DOWN (UINT64_C(1) << 33) // routed mid-bit block32 down projection available
#define BN_GPU_CAP_MOE_ROUTED_MIXED_QUANT (UINT64_C(1) << 34) // mixed routed gate/up and down quant formats available
#define BN_GPU_CAP_MOE_ROUTED_MIDBIT_KQUANT_GATEUP (UINT64_C(1) << 35) // routed mid-bit K-quant gate/up projections available
#define BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL (UINT64_C(1) << 36) // batched recurrent prefill preserves reference trajectory semantics
#define BN_GPU_CAP_KQUANT_BLOCK32_LOGITS (UINT64_C(1) << 37) // K-quant logits use backend-native block32 activation quantization
#define BN_GPU_CAP_FP32_GELU (UINT64_C(1) << 38) // GELU retains FP32 input/output, including CPU-executed steps in GPU requests
#define BN_GPU_CAP_MOE_COMBINED_PREFILL_DEFAULT (UINT64_C(1) << 39) // try combined prefill by default; backend checks format eligibility
#define BN_GPU_CAP_KQUANT_BLOCK32_RECURRENT (UINT64_C(1) << 40) // recurrent K-quant projections use backend-native block32 activations

/* RMSNorm graph supports a separately rounded scalar before its weight.
 * Scaled-router uploads retain raw weights when this capability is set. */
#define BN_GPU_CAP_RMSNORM_SEPARATE_SCALE (UINT64_C(1) << 41)
/* WEIGHTED_ADD can round an output scale before route weighting and
 * explicitly choose fused or separate accumulation. */
#define BN_GPU_CAP_WEIGHTED_ADD_SEPARATE_SCALE (UINT64_C(1) << 42)
#define BN_GPU_CAP_PREFILL_LOGICAL_PROJECTION_ROWS (UINT64_C(1) << 43)
#define BN_GPU_CAP_MOE_ROUTED_ORDERED_KQUANT (UINT64_C(1) << 45)
#define BN_GPU_CAP_MOE_ROUTED_E8M0 (UINT64_C(1) << 44) // routed E8M0 gate/up with ordered K-quant down
// Prepared prefix attention: FP16 KV, head256/GQA16, 33..2048 current+prior
// keys (at least two current queries). Backend validates cache bounds.
#define BN_GPU_CAP_PREFILL_PREFIX_KV (UINT64_C(1) << 46)
/* RMSNorm graph can reproduce the model policy's ordered FP32-square/double-sum
 * contract rather than using the backend's parallel reduction. */
#define BN_GPU_CAP_REFERENCE_RMSNORM_ORDER (UINT64_C(1) << 47)

static inline int bn_gpu_backend_prefill_uses_logical_rows(const BnGPUBackend *gpu) {
    return gpu && (gpu->caps & BN_GPU_CAP_PREFILL_LOGICAL_PROJECTION_ROWS) != 0;
}

#ifdef __cplusplus
}
#endif

#endif // BN_GPU_BACKEND_H
