#ifndef BN_GPU_POLICY_H
#define BN_GPU_POLICY_H

#include "gpu_backend.h"

int bn_gpu_policy_cuda_moe_routed_ffn_enabled(int eligible);
int bn_gpu_policy_cuda_moe_all_f16_cache_forced(void);
int bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int q8_f16_cache);
int bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(int eligible);
int bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(int eligible);
int bn_gpu_policy_cuda_moe_fit_debug_enabled(void);
int bn_gpu_policy_cuda_keep_individual_f16_cache_enabled(void);
int bn_gpu_policy_cuda_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_cuda_logits_f16_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced(void);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(int cols);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache);
size_t bn_gpu_policy_cuda_moe_down_q6_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_requires_full_buffer(
    int tensor_type);
int bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_cuda_moe_quant_only_after_cache(int tensor_type,
                                                  int q8_f16_cache);
int bn_gpu_policy_cuda_moe_prefers_quant_only(int tensor_type);
size_t bn_gpu_policy_cuda_layout_reserve_bytes(void);
size_t bn_gpu_policy_cuda_moe_full_reserve_bytes(void);
int bn_gpu_policy_cuda_cublas_matmul_enabled(void);
int bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled(void);
int bn_gpu_policy_cuda_cublas_cache_max_mb(int default_mb,
                                           int large_budget);
size_t bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols);
size_t bn_gpu_policy_cuda_aux_cache_bytes(int tensor_type,
                                          int rows,
                                          int cols);
int bn_gpu_policy_moe_auto_resident_enabled(void);
int bn_gpu_policy_cuda_duplicate_moe_cache_enabled(void);
int bn_gpu_policy_metal_mmap_zero_copy_enabled(void);
int bn_gpu_policy_argmax_debug_enabled(void);

#endif // BN_GPU_POLICY_H
