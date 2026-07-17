#include "gpu_policy.h"
#include "backend_quant.h"
#include <stdint.h>
#include <stdlib.h>

int bn_gpu_policy_cuda_moe_routed_ffn_enabled(int eligible) {
    return eligible && getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN") == NULL;
}

int bn_gpu_policy_float_buffer_type(void) {
    return bn_backend_quant_gpu_float_buffer_type();
}

int bn_gpu_policy_cuda_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type) {
    return bn_backend_quant_moe_route_q4_down(gate_type, up_type, down_type,
                                              1) ||
           bn_backend_quant_moe_route_q8(gate_type, up_type, down_type);
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
    return bn_backend_quant_cuda_moe_all_f16_cache_supported(tensor_type);
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

int bn_gpu_policy_cuda_moe_lazy_aux_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE") != NULL;
}

int bn_gpu_policy_cuda_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           gpu->buffer_create_quant_only &&
           !bn_gpu_policy_cuda_keep_individual_f16_cache_enabled();
}

int bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           gpu->buffer_create_q6_f32_cache &&
           bn_backend_quant_cuda_logits_q6_f32_cache_supported(tensor_type) &&
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

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache) {
    return !force_f16_cache &&
           bn_backend_quant_cuda_moe_down_q6_f32_cache_supported(tensor_type) &&
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(cols) &&
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(gpu);
}

size_t bn_gpu_policy_cuda_moe_down_q6_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts) {
    if (!bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
            gpu, tensor_type, cols, 0) ||
        rows <= 0 || cols <= 0 || n_experts <= 0)
        return 0;
    if ((size_t)n_experts > SIZE_MAX / (size_t)rows)
        return SIZE_MAX;
    size_t row_count = (size_t)rows * (size_t)n_experts;
    if (row_count > SIZE_MAX / (size_t)cols)
        return SIZE_MAX;
    size_t elems = row_count * (size_t)cols;
    if (elems > SIZE_MAX / sizeof(float))
        return SIZE_MAX;
    size_t bytes = elems * sizeof(float);

    if (bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced())
        return bytes;

    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(512, 0);
    if (max_mb <= 0)
        return bytes;
    size_t max_bytes = (size_t)max_mb * 1024u * 1024u;
    return bytes <= max_bytes ? bytes : 0;
}

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_requires_full_buffer(
    int tensor_type) {
    return bn_backend_quant_cuda_moe_down_q6_f32_cache_supported(tensor_type) &&
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced();
}

int bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu && gpu->buffer_create_q6_f32_cache &&
           bn_backend_quant_cuda_moe_down_q4_f32_cache_supported(tensor_type) &&
           getenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_moe_quant_only_after_cache(int tensor_type,
                                                  int q8_f16_cache) {
    return bn_backend_quant_cuda_moe_quant_only_after_cache(tensor_type,
                                                            q8_f16_cache);
}

int bn_gpu_policy_cuda_moe_prefers_quant_only(int tensor_type) {
    return bn_backend_quant_cuda_moe_prefers_quant_only(tensor_type);
}

int bn_gpu_policy_cuda_matvec_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MATVEC") != NULL;
}

int bn_gpu_policy_cuda_matvec_type_disabled(int tensor_type) {
    return bn_backend_quant_cuda_matvec_type_disabled(tensor_type);
}

static size_t env_mb_or_default(const char *name, size_t def) {
    const char *s = getenv(name);
    if (!s || !*s)
        return def;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0')
        return def;
    return (size_t)v;
}

static size_t mb_to_bytes_saturating(size_t mb) {
    return mb > SIZE_MAX / (1024u * 1024u)
        ? SIZE_MAX
        : mb * 1024u * 1024u;
}

size_t bn_gpu_policy_cuda_layout_reserve_bytes(void) {
    return mb_to_bytes_saturating(
        env_mb_or_default("BN_CUDA_LAYOUT_RESERVE_MB", 512));
}

size_t bn_gpu_policy_cuda_moe_full_reserve_bytes(void) {
    return mb_to_bytes_saturating(
        env_mb_or_default("BN_CUDA_MOE_FULL_RESERVE_MB", 512));
}

int bn_gpu_policy_cuda_cublas_matmul_enabled(void) {
    return getenv("BN_CUDA_DISABLE_CUBLAS_MATMUL") == NULL;
}

int bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16") == NULL &&
           !bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced();
}

int bn_gpu_policy_cuda_q8_0_quant_matmul_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL") != NULL &&
           getenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL") == NULL;
}

int bn_gpu_policy_cuda_f16_q8_0_matmul_enabled(void) {
    return getenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL") == NULL;
}

int bn_gpu_policy_cuda_q8_0_preq_split_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT") != NULL &&
           getenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT") == NULL;
}

int bn_gpu_policy_cuda_decode_logits_cache_enabled(int gpu_logits_need_cpu) {
    return getenv("BN_CUDA_ENABLE_LOGITS_CACHE") != NULL &&
           !gpu_logits_need_cpu;
}

int bn_gpu_policy_cuda_moe_decode_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_cuda_moe_decode_cache_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_cuda_decode_cache_disabled(void) {
    return getenv("BN_CUDA_DISABLE_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_cuda_q4_q8_decode_cache_disabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_cuda_logits_argmax_disabled(void) {
    return getenv("BN_CUDA_DISABLE_LOGITS_ARGMAX") != NULL;
}

int bn_gpu_policy_cuda_dense_logits_argmax_enabled(void) {
    return getenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX") != NULL;
}

int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX") != NULL;
}

int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX") != NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_layer_disabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER") != NULL;
}

int bn_gpu_policy_cuda_q5k_fused_gateup_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP") != NULL;
}

int bn_gpu_policy_cuda_shared_q4_q8_dot_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT") == NULL;
}

int bn_gpu_policy_cuda_shared_expert_gate_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE") == NULL;
}

static int env_positive_int_or_default(const char *name, int default_tokens) {
    const char *env = getenv(name);
    if (!env || !*env)
        return default_tokens;
    int n = atoi(env);
    return n > 0 ? n : default_tokens;
}

int bn_gpu_policy_cuda_prefill_attention_min_tokens_configured(void) {
    const char *env = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    return env && *env;
}

int bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
    int default_tokens) {
    return env_positive_int_or_default("BN_CUDA_PREFILL_ATTN_MIN_TOKENS",
                                       default_tokens);
}

int bn_gpu_policy_cuda_prefill_dense_chain_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN") == NULL;
}

int bn_gpu_policy_cuda_prefill_hybrid_chain_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN") == NULL;
}

int bn_gpu_policy_cuda_prefill_attention_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_ATTN") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_run_chain_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_ffn_fuse_allowed(void) {
    return getenv("BN_CUDA_DISABLE_SSM_FFN_FUSE") == NULL;
}

int bn_gpu_policy_cuda_prefill_moe_chain_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN") != NULL;
}

int bn_gpu_policy_cuda_prefill_hybrid_chain_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN") != NULL;
}

int bn_gpu_policy_cuda_moe_prefill_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_PREFILL") != NULL;
}

int bn_gpu_policy_cuda_moe_prefill_min_tokens_configured(void) {
    const char *env = getenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    return env && *env;
}

int bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(
    int default_tokens) {
    return env_positive_int_or_default("BN_CUDA_MOE_PREFILL_MIN_TOKENS",
                                       default_tokens);
}

int bn_gpu_policy_cuda_moe_cache_prefill_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL") == NULL;
}

int bn_gpu_policy_cuda_moe_prefill_shared_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE") == NULL;
}

int bn_gpu_policy_cuda_moe_route_batch_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_attention_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_disabled(void) {
    return getenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_forced(void) {
    return getenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_prefill_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_prefill_chain_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_prefill_disabled(void) {
    return getenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL") != NULL;
}

int bn_gpu_policy_cuda_large_hybrid_argmax_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX") != NULL;
}

int bn_gpu_policy_cuda_cublas_cache_max_mb(int default_mb,
                                           int large_budget) {
    int max_mb = large_budget ? 512 : default_mb;
    const char *max_env = getenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    if (max_env && *max_env)
        max_mb = atoi(max_env);
    return max_mb;
}

int bn_gpu_policy_cuda_cublas_aux_cache_max_mb(int tensor_type,
                                               int force_q6_f32,
                                               int force_f16) {
    if (force_f16)
        return 0;

    const char *max_env = getenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    if (max_env && *max_env)
        return atoi(max_env);

    if (force_q6_f32 && bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced())
        return 0;

    return bn_backend_quant_cuda_aux_cache_prefers_large_budget(tensor_type)
        ? 512
        : 128;
}

int bn_gpu_policy_cuda_q6k_f16_cache_adds_f32_down_cache(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL &&
           getenv("BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE") == NULL;
}

size_t bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols) {
    if (!gpu || gpu->kind != BN_GPU_BACKEND_CUDA ||
        rows <= 0 || cols <= 0 ||
        !bn_backend_quant_cuda_moe_down_cublas_cache_supported(tensor_type) ||
        !bn_gpu_policy_cuda_cublas_matmul_enabled())
        return 0;
    size_t elems = (size_t)rows * (size_t)cols;
    int q6_as_f16 = bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled();
    size_t elem_size =
        (size_t)bn_backend_quant_cuda_moe_down_cublas_cache_elem_bytes(
            tensor_type, q6_as_f16);
    if (elem_size != 0 && elems > SIZE_MAX / elem_size)
        return 0;
    return elems * elem_size;
}

size_t bn_gpu_policy_cuda_aux_cache_bytes(int tensor_type,
                                          int rows,
                                          int cols) {
    if (rows <= 0 || cols <= 0 || (cols & 31) != 0 ||
        !bn_gpu_policy_cuda_cublas_matmul_enabled() ||
        !bn_backend_quant_cuda_aux_cache_supported(tensor_type))
        return 0;
    int q6_as_f16 =
        bn_backend_quant_cuda_aux_cache_can_use_f16(tensor_type) &&
        bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled();
    if ((size_t)rows > SIZE_MAX / (size_t)cols)
        return SIZE_MAX;
    size_t elems = (size_t)rows * (size_t)cols;
    size_t elem_size =
        bn_backend_quant_cuda_aux_cache_uses_f32(tensor_type, q6_as_f16)
            ? sizeof(float)
            : sizeof(uint16_t);
    if (elem_size != 0 && elems > SIZE_MAX / elem_size)
        return SIZE_MAX;
    size_t bytes = elems * elem_size;

    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(
        128, bn_backend_quant_cuda_aux_cache_prefers_large_budget(tensor_type));
    if (max_mb > 0 && bytes > (size_t)max_mb * 1024u * 1024u)
        return 0;
    return bytes;
}

int bn_gpu_policy_moe_auto_resident_enabled(void) {
    return getenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT") == NULL;
}

int bn_gpu_policy_auto_caps_sequence(int webgpu,
                                     int cuda,
                                     int metal,
                                     int has_moe,
                                     int model_seq_len,
                                     int cap_seq_len) {
    if (model_seq_len <= cap_seq_len)
        return 0;
    return webgpu || cuda || (metal && has_moe);
}

int bn_gpu_policy_cuda_duplicate_moe_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE") == NULL;
}

int bn_gpu_policy_metal_mmap_zero_copy_enabled(void) {
    return getenv("BN_METAL_ENABLE_MMAP_ZERO_COPY") != NULL;
}

void bn_gpu_policy_metal_apply_q4_q8_default(void) {
    if (!getenv("BN_GPU_Q4_Q8") &&
        !getenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT")) {
        setenv("BN_GPU_Q4_Q8", "1", 1);
        if (!getenv("BN_GPU_Q4_Q8_FROM_LAYER"))
            setenv("BN_GPU_Q4_Q8_FROM_LAYER", "0", 1);
    }
}

int bn_gpu_policy_metal_q4_q8_enabled(void) {
    return getenv("BN_GPU_Q4_Q8") != NULL;
}

int bn_gpu_policy_metal_q4_prepared_enabled(void) {
    return getenv("BN_METAL_Q4_PREPARED") != NULL;
}

int bn_gpu_policy_metal_q4_prepared_upload_enabled(void) {
    const char *from_layer = getenv("BN_GPU_Q4_Q8_FROM_LAYER");
    return bn_gpu_policy_metal_q4_prepared_enabled() &&
           getenv("BN_GPU_Q4_Q8") &&
           (!from_layer || atoi(from_layer) <= 0) &&
           !getenv("BN_GPU_Q4_Q8_ATTN_ONLY") &&
           !getenv("BN_GPU_Q4_Q8_FFN_ONLY");
}

int bn_gpu_policy_metal_shared_weights_enabled(void) {
    return getenv("BN_METAL_SHARED_WEIGHTS") != NULL;
}

int bn_gpu_policy_metal_q6_q8k_enabled(void) {
    return getenv("BN_METAL_ENABLE_Q6_Q8K") != NULL;
}

int bn_gpu_policy_metal_q8_barriers_enabled(void) {
    return getenv("BN_METAL_Q8_BARRIERS") != NULL;
}

int bn_gpu_policy_metal_cpu_order_rmsnorm_enabled(void) {
    return getenv("BN_METAL_CPU_ORDER_RMSNORM") != NULL;
}

int bn_gpu_policy_metal_full_barriers_enabled(void) {
    return getenv("BN_METAL_FULL_BARRIERS") != NULL;
}

int bn_gpu_policy_metal_barriers_enabled(void) {
    return getenv("BN_METAL_ENABLE_BARRIERS") != NULL ||
           bn_gpu_policy_metal_full_barriers_enabled();
}

int bn_gpu_policy_metal_barriers_disabled(void) {
    return getenv("BN_METAL_DISABLE_BARRIERS") != NULL ||
           !bn_gpu_policy_metal_barriers_enabled();
}

int bn_gpu_policy_fused_gateup_enabled(void) {
    return getenv("BN_GPU_DISABLE_FUSED_GATEUP") == NULL;
}

int bn_gpu_policy_q4_q8_fused_gateup_enabled(void) {
    return getenv("BN_GPU_Q4_Q8_DISABLE_GATEUP") == NULL;
}

int bn_gpu_policy_gateup_split_enabled(void) {
    return getenv("BN_GPU_DISABLE_GATEUP_SPLIT") == NULL;
}

int bn_gpu_policy_q4_q8_ffn_down_enabled(void) {
    return getenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN") == NULL;
}

int bn_gpu_policy_qkv_split_enabled(void) {
    return getenv("BN_GPU_DISABLE_QKV_SPLIT") == NULL;
}

int bn_gpu_policy_qkv_split_debug_enabled(void) {
    return getenv("BN_GPU_DEBUG_QKV_SPLIT") != NULL;
}

int bn_gpu_policy_ssm_qkvz_split_enabled(void) {
    return getenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT") == NULL;
}

int bn_gpu_policy_ssm_ab_stack_enabled(void) {
    return getenv("BN_GPU_DISABLE_SSM_AB_STACK") == NULL;
}

int bn_gpu_policy_split_residual_rmsnorm_enabled(void) {
    return getenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM") != NULL;
}

int bn_gpu_policy_argmax_debug_enabled(void) {
    return getenv("BN_GPU_DEBUG_ARGMAX") != NULL;
}

int bn_gpu_policy_cpu_logits_enabled(void) {
    return getenv("BN_GPU_CPU_LOGITS") != NULL;
}

int bn_gpu_policy_compare_logits_enabled(void) {
    return getenv("BN_GPU_COMPARE_LOGITS") != NULL;
}

int bn_gpu_policy_debug_argmax_compare_enabled(void) {
    return getenv("BN_GPU_DEBUG_ARGMAX_COMPARE") != NULL;
}

int bn_gpu_policy_q6_logits_refine_enabled(int cuda_backend,
                                           int q6_refine_default) {
    return q6_refine_default ||
           getenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE") != NULL ||
           (!cuda_backend &&
            getenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE") == NULL);
}

int bn_gpu_policy_q6_logits_refine_top_or_default(int default_top) {
    const char *env = getenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    return env ? atoi(env) : default_top;
}

int bn_gpu_policy_q8_logits_refine_enabled(int cuda_backend,
                                           int q8_refine_default) {
    return getenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE") != NULL ||
           q8_refine_default ||
           (!cuda_backend &&
            getenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE") == NULL);
}

int bn_gpu_policy_q8_logits_refine_top_or_default(int default_top) {
    const char *env = getenv("BN_GPU_Q8_REFINE_TOP");
    return env ? atoi(env) : default_top;
}

int bn_gpu_policy_cuda_moe_ffn_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_FFN") != NULL;
}

int bn_gpu_policy_cuda_moe_cpu_actual_override_enabled(void) {
    return getenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL") != NULL;
}

int bn_gpu_policy_moe_compare_layer_selected(int layer, int pos) {
    const char *compare_moe_env = getenv("BN_GPU_COMPARE_MOE_LAYER");
    if (!compare_moe_env)
        return 0;
    int compare_layer = atoi(compare_moe_env);
    const char *compare_pos_env = getenv("BN_GPU_COMPARE_MOE_POS");
    int compare_pos = compare_pos_env ? atoi(compare_pos_env) : -1;
    return compare_layer == layer && (compare_pos < 0 || compare_pos == pos);
}

int bn_gpu_policy_moe_compare_input_norm_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_INPUT_NORM") != NULL;
}

int bn_gpu_policy_moe_compare_actual_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_ACTUAL") != NULL;
}

int bn_gpu_policy_moe_compare_route_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_ROUTE") != NULL;
}

int bn_gpu_policy_moe_compare_raw_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_RAW") != NULL;
}

int bn_gpu_policy_moe_compare_mid_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_MID") != NULL;
}

int bn_gpu_policy_moe_compare_parts_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_PARTS") != NULL;
}

int bn_gpu_policy_moe_compare_shared_mid_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_SHARED_MID") != NULL;
}

int bn_gpu_policy_moe_compare_shared_down_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_SHARED_DOWN") != NULL;
}

int bn_gpu_policy_moe_compare_norm_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_NORM") != NULL;
}

int bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK") == NULL;
}

int bn_gpu_policy_cuda_moe_gateup_split_enabled(int can_split) {
    return can_split && getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT") == NULL;
}

int bn_gpu_policy_moe_route_profile_enabled(void) {
    return getenv("BN_GPU_MOE_ROUTE_PROFILE") != NULL;
}

int bn_gpu_policy_moe_route_profile_every_or_default(int default_every) {
    int every = default_every;
    const char *env = getenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0)
            every = v;
    }
    return every;
}

int bn_gpu_policy_profile_level(void) {
    const char *profile = getenv("BN_GPU_PROFILE");
    return profile ? atoi(profile) : 0;
}
