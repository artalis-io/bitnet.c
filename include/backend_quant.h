#ifndef BN_BACKEND_QUANT_H
#define BN_BACKEND_QUANT_H

#include <stdint.h>
#include "gguf.h"
#include "gpu_backend.h"
#include "quant.h"

#define BN_BACKEND_QUANT_GPU_MATVEC_FLAG_Q8K_DOT BN_QUANT_GPU_MATVEC_FLAG_Q8K_DOT
#define BN_BACKEND_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K BN_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K

static inline uint32_t bn_backend_quant_gpu_split_cap(int type) {
    return bn_quant_format_gpu_split_cap(type);
}

static inline int bn_backend_quant_can_gpu_split(int type) {
    return bn_quant_format_can_gpu_split(type);
}

static inline int bn_backend_quant_can_gpu_native(int type) {
    return bn_quant_format_can_gpu_native(type);
}

static inline int bn_backend_quant_can_gpu_repack(int type) {
    return bn_quant_format_can_gpu_repack(type);
}

static inline int bn_backend_quant_cuda_small_dense_supported(int type) {
    return bn_quant_format_supports_gpu_small_dense(type);
}

static inline int bn_backend_quant_cuda_small_dense_q8_supported(int type) {
    return bn_quant_format_supports_gpu_small_dense_q8(type);
}

static inline int bn_backend_quant_is_kquant_float_fallback_candidate(int type) {
    return bn_quant_format_is_float_kquant_fallback_candidate(type);
}

static inline int bn_backend_quant_can_preq8k(int type) {
    return bn_quant_format_can_preq8k(type);
}

static inline int bn_backend_quant_supports_q8_logits_refine(int type) {
    return bn_quant_format_supports_q8_logits_refine(type);
}

static inline int bn_backend_quant_logits_uses_f16_path(int type) {
    return bn_quant_format_uses_f16_logits_path(type);
}

static inline int bn_backend_quant_tied_logits_uses_quant_path(int type) {
    return bn_quant_format_tied_logits_uses_quant_path(type);
}

static inline int bn_backend_quant_logits_i8_cache_supported(int type) {
    return bn_quant_format_supports_logits_i8_cache(type);
}

static inline int bn_backend_quant_tied_logits_uses_f16_path(int type) {
    return bn_quant_format_tied_logits_uses_f16_path(type);
}

static inline int bn_backend_quant_tied_logits_i8_weight_type(void) {
    return bn_quant_format_tied_logits_i8_weight_type();
}

static inline int bn_backend_quant_tied_logits_f16_weight_type(void) {
    return bn_quant_format_tied_logits_f16_weight_type();
}

static inline int bn_backend_quant_tied_logits_f32_weight_type(void) {
    return bn_quant_format_tied_logits_f32_weight_type();
}

static inline int bn_backend_quant_dense_f32_type(void) {
    return bn_quant_format_dense_f32_type();
}

static inline int bn_backend_quant_gpu_float_buffer_type(void) {
    return bn_quant_format_gpu_float_buffer_type();
}

static inline int bn_backend_quant_already_f32(int type) {
    return bn_quant_format_is_f32(type);
}

static inline int bn_backend_quant_can_convert_dense_to_f32(int type) {
    return bn_quant_format_can_convert_dense_to_f32(type);
}

static inline int bn_backend_quant_convert_dense_to_f32(
    int type, const void *src, float *dst, int n) {
    return bn_quant_format_convert_dense_to_f32(type, src, dst, n);
}

static inline int bn_backend_quant_gpu_requires_exact_silu(int type) {
    return bn_quant_format_gpu_requires_exact_silu(type);
}

static inline int bn_backend_quant_gpu_prefers_gateup_split(int type) {
    return bn_quant_format_gpu_prefers_gateup_split(type);
}

static inline int bn_backend_quant_moe_route_q4_down(int gate_type,
                                                     int up_type,
                                                     int down_type,
                                                     int allow_q4_down) {
    return bn_quant_format_supports_moe_q4_down_route(gate_type,
                                                      up_type,
                                                      down_type,
                                                      allow_q4_down);
}

static inline int bn_backend_quant_moe_gateup_q4(int gate_type,
                                                 int up_type) {
    return bn_quant_format_supports_moe_q4_gateup(gate_type, up_type);
}

static inline int bn_backend_quant_cpu_fused_q4_gateup_silu(int gate_type,
                                                            int up_type) {
    return bn_quant_format_supports_cpu_fused_q4_gateup_silu(gate_type,
                                                             up_type);
}

static inline int bn_backend_quant_stacked_pair_same_format(int left_type,
                                                            int right_type) {
    return bn_quant_format_pair_same_format(left_type, right_type);
}

static inline int bn_backend_quant_allows_stacked_layout(int type) {
    return bn_quant_format_allows_stacked_layout(type);
}

static inline int bn_backend_quant_moe_route_q8(int gate_type,
                                                int up_type,
                                                int down_type) {
    return bn_quant_format_supports_moe_q8_route(gate_type,
                                                 up_type,
                                                 down_type);
}

static inline int bn_backend_quant_supports_q6k_logits_refine(int type) {
    return bn_quant_format_supports_q6_logits_refine(type);
}

static inline int bn_backend_quant_cuda_logits_q6_f32_cache_supported(int type) {
    return bn_quant_format_cuda_logits_q6_f32_cache_supported(type);
}

static inline int bn_backend_quant_cuda_moe_all_f16_cache_supported(int type) {
    return bn_quant_format_cuda_moe_all_f16_cache_supported(type);
}

static inline int bn_backend_quant_cuda_moe_down_q6_f32_cache_supported(int type) {
    return bn_quant_format_cuda_moe_down_q6_f32_cache_supported(type);
}

static inline int bn_backend_quant_cuda_moe_down_cublas_cache_supported(int type) {
    return bn_quant_format_cuda_moe_down_cublas_cache_supported(type);
}

static inline int bn_backend_quant_cuda_moe_down_cublas_cache_elem_bytes(
    int type, int q6_as_f16) {
    return bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes(type,
                                                                 q6_as_f16);
}

static inline int bn_backend_quant_cuda_moe_down_q4_f32_cache_supported(int type) {
    return bn_quant_format_cuda_moe_down_q4_f32_cache_supported(type);
}

static inline int bn_backend_quant_cuda_moe_quant_only_after_cache(
    int type, int q8_f16_cache) {
    return bn_quant_format_cuda_moe_quant_only_after_cache(type,
                                                           q8_f16_cache);
}

static inline int bn_backend_quant_cuda_lazy_moe_aux_cache_candidate(int type) {
    return bn_quant_format_cuda_lazy_moe_aux_cache_candidate(type);
}

static inline int bn_backend_quant_cuda_moe_prefers_quant_only(int type) {
    return bn_quant_format_cuda_moe_prefers_quant_only(type);
}

static inline int bn_backend_quant_cuda_aux_cache_supported(int type) {
    return bn_quant_format_cuda_aux_cache_supported(type);
}

static inline int bn_backend_quant_cuda_aux_cache_can_use_f16(int type) {
    return bn_quant_format_cuda_aux_cache_can_use_f16(type);
}

static inline int bn_backend_quant_cuda_aux_cache_uses_f32(int type,
                                                          int q6_as_f16) {
    return bn_quant_format_cuda_aux_cache_uses_f32(type, q6_as_f16);
}

static inline int bn_backend_quant_cuda_aux_cache_prefers_large_budget(int type) {
    return bn_quant_format_cuda_aux_cache_prefers_large_budget(type);
}

static inline int bn_backend_quant_cuda_matvec_type_disabled(int type) {
    return bn_quant_policy_cuda_matvec_type_disabled(type);
}

static inline uint32_t bn_backend_quant_gpu_fused_gateup_silu_cap(int type) {
    return bn_quant_format_gpu_fused_gateup_silu_cap(type);
}

static inline int bn_backend_quant_gpu_fused_gateup_requires_cuda_opt_in(int type) {
    return bn_quant_format_gpu_fused_gateup_requires_cuda_opt_in(type);
}

static inline int bn_backend_quant_can_gpu_gateup_split_activation(int type,
                                                                  int act_type) {
    return bn_quant_format_gpu_allows_gateup_split_activation(type, act_type);
}

static inline uint32_t bn_backend_quant_gpu_matvec_q8k_dot_flag(int type,
                                                               int enabled) {
    return bn_quant_format_gpu_matvec_q8k_dot_flag(type, enabled);
}

static inline uint32_t bn_backend_quant_gpu_matvec_exact_q6k_flag(int type,
                                                                 int enabled) {
    return bn_quant_format_gpu_matvec_exact_q6k_flag(type, enabled);
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
