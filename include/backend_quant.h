#ifndef BN_BACKEND_QUANT_H
#define BN_BACKEND_QUANT_H

#include <stdint.h>
#include "gguf.h"
#include "gpu_backend.h"
#include "quant.h"

static inline uint32_t bn_backend_quant_gpu_split_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_Q4_MATVEC_SPLIT;
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

static inline uint32_t bn_backend_quant_gpu_fused_gateup_silu_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
        default: return 0;
    }
}

static inline int bn_backend_quant_can_gpu_gateup_split_activation(int type,
                                                                  int act_type) {
    return act_type != 1 || type != BN_GGUF_TENSOR_Q4_K;
}

void bn_backend_quant_matvec_gpu(float *out, const BnQWeight *W,
                                 const float *x, int8_t *x_q_buf,
                                 BnThreadPool *pool, BnGPUBackend *gpu);
void bn_backend_quant_matvec_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *x,
                                     int8_t *x_q_buf, BnThreadPool *pool,
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

#endif // BN_BACKEND_QUANT_H
