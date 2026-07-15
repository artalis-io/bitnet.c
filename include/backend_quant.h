#ifndef BN_BACKEND_QUANT_H
#define BN_BACKEND_QUANT_H

#include <stdint.h>
#include "gguf.h"
#include "gpu_backend.h"
#include "quant.h"

#define BN_BACKEND_QUANT_GPU_MATVEC_FLAG_Q8K_DOT 1u
#define BN_BACKEND_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K 8u

static inline uint32_t bn_backend_quant_gpu_split_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_Q4_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q5_0: return BN_GPU_CAP_Q5_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q4_K: return BN_GPU_CAP_Q4K_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q5_K: return BN_GPU_CAP_Q5K_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q8_0: return BN_GPU_CAP_Q8_MATVEC_SPLIT;
        default: return 0;
    }
}

static inline int bn_backend_quant_can_gpu_split(int type) {
    return bn_backend_quant_gpu_split_cap(type) != 0;
}

static inline int bn_backend_quant_can_gpu_native(int type) {
    return type == BN_GGUF_TENSOR_Q4_0;
}

static inline int bn_backend_quant_can_gpu_repack(int type) {
    return type == BN_GGUF_TENSOR_Q4_0;
}

static inline int bn_backend_quant_cuda_small_dense_supported(int type) {
    return type == BN_GGUF_TENSOR_F32 ||
           type == BN_GGUF_TENSOR_F16 ||
           type == BN_GGUF_TENSOR_Q8_0 ||
           type == BN_GGUF_TENSOR_Q4_0 ||
           type == BN_GGUF_TENSOR_Q5_0 ||
           type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K ||
           type == BN_GGUF_TENSOR_Q8_K;
}

static inline int bn_backend_quant_cuda_small_dense_q8_supported(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

static inline int bn_backend_quant_is_kquant_float_fallback_candidate(int type) {
    return type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K;
}

static inline int bn_backend_quant_supports_q8_logits_refine(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

static inline int bn_backend_quant_gpu_requires_exact_silu(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

static inline int bn_backend_quant_gpu_prefers_gateup_split(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

static inline int bn_backend_quant_moe_route_q4_down(int gate_type,
                                                     int up_type,
                                                     int down_type,
                                                     int allow_q4_down) {
    return gate_type == BN_GGUF_TENSOR_Q4_K &&
           up_type == BN_GGUF_TENSOR_Q4_K &&
           (down_type == BN_GGUF_TENSOR_Q6_K ||
            (allow_q4_down && down_type == BN_GGUF_TENSOR_Q4_K));
}

static inline int bn_backend_quant_moe_gateup_q4(int gate_type,
                                                 int up_type) {
    return gate_type == BN_GGUF_TENSOR_Q4_K &&
           up_type == BN_GGUF_TENSOR_Q4_K;
}

static inline int bn_backend_quant_moe_route_q8(int gate_type,
                                                int up_type,
                                                int down_type) {
    return gate_type == BN_GGUF_TENSOR_Q8_0 &&
           up_type == BN_GGUF_TENSOR_Q8_0 &&
           down_type == BN_GGUF_TENSOR_Q8_0;
}

static inline int bn_backend_quant_supports_q6k_logits_refine(int type) {
    return type == BN_GGUF_TENSOR_Q6_K;
}

static inline uint32_t bn_backend_quant_gpu_fused_gateup_silu_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q5_0: return BN_GPU_CAP_Q5_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q8_0: return BN_GPU_CAP_Q8_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q4_K: return BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q5_K: return BN_GPU_CAP_Q5K_FUSED_GATEUP_SILU;
        default: return 0;
    }
}

static inline int bn_backend_quant_gpu_fused_gateup_requires_cuda_opt_in(int type) {
    return type == BN_GGUF_TENSOR_Q5_K;
}

static inline int bn_backend_quant_can_gpu_gateup_split_activation(int type,
                                                                  int act_type) {
    return act_type != 1 || type != BN_GGUF_TENSOR_Q4_K;
}

static inline uint32_t bn_backend_quant_gpu_matvec_q8k_dot_flag(int type,
                                                               int enabled) {
    return enabled && type == BN_GGUF_TENSOR_Q4_K
        ? BN_BACKEND_QUANT_GPU_MATVEC_FLAG_Q8K_DOT
        : 0u;
}

static inline uint32_t bn_backend_quant_gpu_matvec_exact_q6k_flag(int type,
                                                                 int enabled) {
    return enabled && type == BN_GGUF_TENSOR_Q6_K
        ? BN_BACKEND_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K
        : 0u;
}

void bn_backend_quant_matvec_gpu(float *out, const BnQWeight *W,
                                 const float *x, int8_t *x_q_buf,
                                 BnThreadPool *pool, BnGPUBackend *gpu);
void bn_backend_quant_matvec_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *x,
                                     int8_t *x_q_buf, BnThreadPool *pool,
                                     BnGPUBackend *gpu);
void bn_backend_quant_matvec_gpu_buf_prepared(float *out, const BnQWeight *W,
                                              const BnPreparedWeight *prepared,
                                              void *W_buf, const float *x,
                                              int8_t *x_q_buf,
                                              BnThreadPool *pool,
                                              BnGPUBackend *gpu);
void bn_backend_quant_matvec_batch_gpu(const BnMatvecTask *tasks, int n_tasks,
                                       const float *x, int8_t *x_q_buf,
                                       BnThreadPool *pool, BnGPUBackend *gpu);
void bn_backend_quant_matvec_batch_gpu_buf(const BnMatvecTask *tasks,
                                           const void *const *W_bufs,
                                           int n_tasks, const float *x,
                                           int8_t *x_q_buf,
                                           BnThreadPool *pool,
                                           BnGPUBackend *gpu);
void bn_backend_quant_matmul_gpu(float *out, const BnQWeight *W,
                                 const float *X, int n_tokens,
                                 int8_t *x_q_buf, BnThreadPool *pool,
                                 BnGPUBackend *gpu);
void bn_backend_quant_matmul_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *X,
                                     int n_tokens, int8_t *x_q_buf,
                                     BnThreadPool *pool, BnGPUBackend *gpu);
void bn_backend_quant_matmul_batch_gpu_buf(const BnMatvecTask *tasks,
                                           const void *const *W_bufs,
                                           int n_tasks, const float *X,
                                           int n_tokens, int x_cols,
                                           int8_t *x_q_buf,
                                           BnThreadPool *pool,
                                           BnGPUBackend *gpu);

#endif // BN_BACKEND_QUANT_H
