#include "gpu_policy.h"
#include "quant.h"
#include <stdlib.h>

int bn_gpu_policy_cuda_moe_routed_ffn_enabled(int eligible) {
    return eligible && getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN") == NULL;
}

int bn_gpu_policy_cuda_moe_all_f16_cache_forced(void) {
    return getenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE") != NULL;
}

int bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int q8_f16_cache) {
    if (!gpu || !gpu->buffer_create_f16_cache ||
        getenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE") != NULL)
        return 0;
    if (bn_gpu_policy_cuda_moe_all_f16_cache_forced())
        return 1;
    if (!q8_f16_cache)
        return 0;
    return bn_quant_format_cuda_moe_all_f16_cache_supported(tensor_type);
}

int bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE") == NULL;
}

int bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE") != NULL;
}

int bn_gpu_policy_cuda_moe_fit_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_FIT") != NULL;
}

int bn_gpu_policy_cuda_keep_individual_f16_cache_enabled(void) {
    return getenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE") != NULL;
}

int bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           gpu->buffer_create_q6_f32_cache &&
           bn_quant_format_cuda_logits_q6_f32_cache_supported(tensor_type) &&
           getenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_logits_f16_cache_enabled(const BnGPUBackend *gpu) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           gpu->buffer_create_f16_cache &&
           getenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE") != NULL;
}

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_q6_f32_cache &&
           !bn_gpu_policy_cuda_moe_all_f16_cache_forced() &&
           getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced(void) {
    return getenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE") != NULL;
}

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(int cols) {
    return cols > 1024 ||
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced();
}

int bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu && gpu->buffer_create_q6_f32_cache &&
           bn_quant_format_cuda_moe_down_q4_f32_cache_supported(tensor_type) &&
           getenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_cublas_matmul_enabled(void) {
    return getenv("BN_CUDA_DISABLE_CUBLAS_MATMUL") == NULL;
}

int bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16") == NULL &&
           !bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced();
}

int bn_gpu_policy_cuda_cublas_cache_max_mb(int default_mb,
                                           int large_budget) {
    int max_mb = large_budget ? 512 : default_mb;
    const char *max_env = getenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    if (max_env && *max_env)
        max_mb = atoi(max_env);
    return max_mb;
}

int bn_gpu_policy_moe_auto_resident_enabled(void) {
    return getenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT") == NULL;
}

int bn_gpu_policy_cuda_duplicate_moe_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE") == NULL;
}
