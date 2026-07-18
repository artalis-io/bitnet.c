#include "gpu_policy.h"
#include "backend_quant.h"
#include "model_arch.h"
#include <stdint.h>
#include <stdlib.h>

static int gpu_policy_env_enabled(const char *name, const char *compat_name) {
    return getenv(name) != NULL ||
           (compat_name && getenv(compat_name) != NULL);
}

static const char *gpu_policy_env_value(const char *name,
                                        const char *compat_name) {
    const char *env = getenv(name);
    return env ? env : (compat_name ? getenv(compat_name) : NULL);
}

static int gpu_policy_env_layer_selected(const char *name,
                                         const char *compat_name,
                                         int layer) {
    const char *env = gpu_policy_env_value(name, compat_name);
    if (!env || !*env || layer < 0)
        return 0;
    const char *p = env;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',')
            p++;
        if (!*p)
            break;
        int start = atoi(p);
        while (*p && *p != ',' && *p != '-')
            p++;
        int end = start;
        if (*p == '-') {
            p++;
            end = atoi(p);
            while (*p && *p != ',')
                p++;
        }
        if (layer >= start && layer <= end)
            return 1;
    }
    return 0;
}

static float gpu_policy_env_float_or_default(const char *name,
                                             const char *compat_name,
                                             float default_value) {
    const char *env = gpu_policy_env_value(name, compat_name);
    if (!env || !*env) return default_value;
    return (float)atof(env);
}

int bn_gpu_policy_cuda_moe_routed_ffn_enabled(int eligible) {
    return eligible && getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN") == NULL;
}

int bn_gpu_policy_backend_is_cuda(const BnGPUBackend *gpu) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA;
}

int bn_gpu_policy_float_buffer_type(void) {
    return bn_backend_quant_gpu_float_buffer_type();
}

int bn_gpu_policy_attention_layer_count(const BnConfig *c) {
    return bn_model_arch_attention_layer_count(c);
}

int bn_gpu_policy_ssm_layer_count(const BnConfig *c) {
    return bn_model_arch_ssm_layer_count(c);
}

int bn_gpu_policy_uses_hybrid_ssm(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_ssm(c);
}

int bn_gpu_policy_uses_hybrid_moe(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_moe(c);
}

int bn_gpu_policy_uses_moe(const BnConfig *c) {
    return bn_model_arch_uses_moe(c);
}

int bn_gpu_policy_moe_router_diff2_upload_enabled(const BnConfig *c) {
    return bn_model_arch_uses_all_active_two_expert_moe(c, c ? c->dim : 0);
}

int bn_gpu_policy_cuda_moe_f16_aux_cache_auto_enabled(const BnConfig *c) {
    return bn_model_arch_uses_more_than_two_expert_moe(c) ||
           bn_model_arch_uses_two_expert_all_active_moe(c);
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
    return bn_backend_quant_moe_all_f16_cache_supported(tensor_type);
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
    return bn_gpu_policy_backend_is_cuda(gpu) &&
           gpu->buffer_create_quant_only &&
           !bn_gpu_policy_cuda_keep_individual_f16_cache_enabled();
}

int bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return bn_gpu_policy_backend_is_cuda(gpu) &&
           gpu->buffer_create_q6_f32_cache &&
           bn_backend_quant_logits_q6_f32_cache_supported(tensor_type) &&
           getenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_logits_f16_cache_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_backend_is_cuda(gpu) &&
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
           bn_backend_quant_moe_down_q6_f32_cache_supported(tensor_type) &&
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
    return bn_backend_quant_moe_down_q6_f32_cache_supported(tensor_type) &&
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced();
}

int bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu && gpu->buffer_create_q6_f32_cache &&
           bn_backend_quant_moe_down_q4_f32_cache_supported(tensor_type) &&
           getenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_moe_quant_only_after_cache(int tensor_type,
                                                  int q8_f16_cache) {
    return bn_backend_quant_moe_quant_only_after_cache(tensor_type,
                                                            q8_f16_cache);
}

int bn_gpu_policy_cuda_moe_prefers_quant_only(int tensor_type) {
    return bn_backend_quant_moe_prefers_quant_only(tensor_type);
}

int bn_gpu_policy_cuda_matvec_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MATVEC") != NULL;
}

int bn_gpu_policy_cuda_matvec_type_disabled(int tensor_type) {
    return bn_backend_quant_cuda_matvec_type_disabled(tensor_type);
}

int bn_gpu_policy_cuda_small_kquant_native_enabled(int force_float_kquant) {
    if (getenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE"))
        return 1;
    return getenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE") == NULL &&
           !force_float_kquant;
}

int bn_gpu_policy_cuda_small_kquant_native_disabled(void) {
    return getenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE") != NULL;
}

size_t bn_gpu_policy_max_storage_binding_bytes(size_t backend_limit) {
    size_t max_storage_binding = backend_limit;
    if (max_storage_binding == 0)
        max_storage_binding = 128ull * 1024ull * 1024ull;
    const char *override_mb = getenv("BN_GPU_MAX_STORAGE_BINDING_MB");
    if (override_mb) {
        long mb = strtol(override_mb, NULL, 10);
        if (mb >= 0)
            max_storage_binding = (size_t)mb * 1024ull * 1024ull;
    }
    return max_storage_binding;
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

static size_t positive_env_mb_or_default(const char *name, size_t def) {
    const char *s = getenv(name);
    if (!s || !*s)
        return def;
    if (*s == '-')
        return 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0' || v == 0)
        return 0;
    return (size_t)v;
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

static int env_int_or_default(const char *name, int default_value) {
    const char *env = getenv(name);
    return env ? atoi(env) : default_value;
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

int bn_gpu_policy_prefill_matmul_disabled(void) {
    return getenv("BN_GPU_DISABLE_PREFILL_MATMUL") != NULL;
}

int bn_gpu_policy_prefill_matmul_enabled(void) {
    return getenv("BN_GPU_PREFILL_MATMUL") != NULL;
}

int bn_gpu_policy_cuda_prefill_direct_kv_disabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV") != NULL;
}

int bn_gpu_policy_cuda_prefill_direct_kv_with_cpu_fallback_enabled(void) {
    return getenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK") != NULL;
}

int bn_gpu_policy_cpu_decode_fallback_requested(void) {
    return getenv("BN_GPU_CPU_FALLBACK_LAYER") ||
           getenv("BN_GPU_CPU_FALLBACK_FROM_LAYER") ||
           getenv("BN_GPU_CPU_ATTN_LAYER") ||
           getenv("BN_GPU_CPU_ATTN_FROM_LAYER");
}

int bn_gpu_policy_cpu_fallback_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_CPU_FALLBACK_LAYER", default_layer);
}

int bn_gpu_policy_cpu_fallback_from_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_CPU_FALLBACK_FROM_LAYER",
                              default_layer);
}

int bn_gpu_policy_cpu_attention_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_CPU_ATTN_LAYER", default_layer);
}

int bn_gpu_policy_cpu_attention_from_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_CPU_ATTN_FROM_LAYER", default_layer);
}

int bn_gpu_policy_cpu_ffn_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_CPU_FFN_LAYER", default_layer);
}

int bn_gpu_policy_cpu_ffn_from_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_CPU_FFN_FROM_LAYER", default_layer);
}

int bn_gpu_policy_cpu_ffn_down_from_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_CPU_FFN_DOWN_FROM_LAYER",
                              default_layer);
}

int bn_gpu_policy_cuda_ssm_graph_disabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_GRAPH") != NULL;
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

    return bn_backend_quant_aux_cache_prefers_large_budget(tensor_type)
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
    if (!bn_gpu_policy_backend_is_cuda(gpu) ||
        rows <= 0 || cols <= 0 ||
        !bn_backend_quant_moe_down_cublas_cache_supported(tensor_type) ||
        !bn_gpu_policy_cuda_cublas_matmul_enabled())
        return 0;
    size_t elems = (size_t)rows * (size_t)cols;
    int q6_as_f16 = bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled();
    size_t elem_size =
        (size_t)bn_backend_quant_moe_down_cublas_cache_elem_bytes(
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
        !bn_backend_quant_aux_cache_supported(tensor_type))
        return 0;
    int q6_as_f16 =
        bn_backend_quant_aux_cache_can_use_f16(tensor_type) &&
        bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled();
    if ((size_t)rows > SIZE_MAX / (size_t)cols)
        return SIZE_MAX;
    size_t elems = (size_t)rows * (size_t)cols;
    size_t elem_size =
        bn_backend_quant_aux_cache_uses_f32(tensor_type, q6_as_f16)
            ? sizeof(float)
            : sizeof(uint16_t);
    if (elem_size != 0 && elems > SIZE_MAX / elem_size)
        return SIZE_MAX;
    size_t bytes = elems * elem_size;

    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(
        128, bn_backend_quant_aux_cache_prefers_large_budget(tensor_type));
    if (max_mb > 0 && bytes > (size_t)max_mb * 1024u * 1024u)
        return 0;
    return bytes;
}

int bn_gpu_policy_moe_auto_resident_enabled(void) {
    return getenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT") == NULL;
}

size_t bn_gpu_policy_moe_cache_reserve_bytes(void) {
    return mb_to_bytes_saturating(
        positive_env_mb_or_default("BN_GPU_MOE_CACHE_RESERVE_MB", 4096));
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

int bn_gpu_policy_auto_caps_gguf_sequence(int webgpu,
                                          int cuda,
                                          int metal,
                                          BnGGUFFile *gf,
                                          int model_seq_len,
                                          int cap_seq_len) {
    return bn_gpu_policy_auto_caps_sequence(
        webgpu, cuda, metal, bn_model_arch_gguf_uses_moe(gf),
        model_seq_len, cap_seq_len);
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
        if (!getenv("BN_GPU_Q4_Q8_ATTN_ONLY") &&
            !getenv("BN_GPU_Q4_Q8_FFN_ONLY"))
            setenv("BN_GPU_Q4_Q8_FFN_ONLY", "1", 1);
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

int bn_gpu_policy_q4_q8_attn_only_enabled(void) {
    return getenv("BN_GPU_Q4_Q8_ATTN_ONLY") != NULL;
}

int bn_gpu_policy_q4_q8_ffn_only_enabled(void) {
    return getenv("BN_GPU_Q4_Q8_FFN_ONLY") != NULL;
}

int bn_gpu_policy_q4_q8_from_layer_or_default(int n_layers) {
    const char *env = getenv("BN_GPU_Q4_Q8_FROM_LAYER");
    if (env)
        return atoi(env);
    return bn_gpu_policy_metal_q4_q8_enabled() ? n_layers - 1 : -1;
}

int bn_gpu_policy_q4_q8_to_layer_or_default(int n_layers,
                                            int metal_q4_prepared) {
    const char *env = getenv("BN_GPU_Q4_Q8_TO_LAYER");
    if (env)
        return atoi(env);

    env = getenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    if (env) {
        int tail_native = atoi(env);
        if (tail_native > 0) {
            int to_layer = n_layers - tail_native - 1;
            return to_layer < -1 ? -1 : to_layer;
        }
        return -1;
    }

    if (bn_gpu_policy_metal_q4_q8_enabled() && !metal_q4_prepared &&
        n_layers > 33)
        return n_layers - 33 - 1;
    return -1;
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

int bn_gpu_policy_debug_fallback_enabled(void) {
    return getenv("BN_GPU_DEBUG_FALLBACK") != NULL;
}

int bn_gpu_policy_force_graph_enabled(void) {
    return getenv("BN_GPU_FORCE_GRAPH") != NULL;
}

int bn_gpu_policy_flash_min_kv_or_default(int default_min_kv) {
    const char *env = getenv("BN_GPU_FLASH_MIN_KV");
    return env ? atoi(env) : default_min_kv;
}

int bn_gpu_policy_flash_max_kv_or_default(int cuda_backend,
                                          int default_max_kv) {
    const char *env = getenv("BN_GPU_FLASH_MAX_KV");
    if (env)
        return atoi(env);
    return cuda_backend ? 2048 : default_max_kv;
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

int bn_gpu_policy_cuda_moe_router_topk_enabled(int eligible) {
    return eligible && getenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK") == NULL;
}

int bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT") == NULL;
}

int bn_gpu_policy_cuda_moe_router_gpu_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU") == NULL;
}

int bn_gpu_policy_cuda_moe_router_diff2_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2") == NULL;
}

int bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(int large_moe) {
    if (getenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH"))
        return 0;
    return !large_moe ||
           getenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE") != NULL;
}

int bn_gpu_policy_cuda_moe_cpu_actual_override_enabled(void) {
    return getenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL") != NULL;
}

int bn_gpu_policy_small_dense_q8_cpu_attention_safe_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE");
}

int bn_gpu_policy_small_dense_exact_q4_q8_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8",
                                  "BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8");
}

int bn_gpu_policy_small_dense_exact_ffn_down_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN",
                                  "BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
}

int bn_gpu_policy_small_dense_prefill_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL",
                                  "BN_CUDA_DISABLE_SMALL_QWEN_PREFILL");
}

int bn_gpu_policy_small_dense_q8_logits_refine_enabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
        "BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
}

int bn_gpu_policy_small_dense_q8_logits_refine_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
        "BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
}

int bn_gpu_policy_all2_q4q6_moe_fast_ffn_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
                                  "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
}

int bn_gpu_policy_all2_q4q6_moe_fast_graph_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_FAST_GRAPH",
                                  "BN_CUDA_DISABLE_QWEN2MOE_FAST_MOE_GRAPH");
}

int bn_gpu_policy_all2_q4q6_moe_cublas_decode_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_CUBLAS_DECODE",
                                  "BN_CUDA_ENABLE_QWEN2MOE_MOE_CUBLAS_DECODE");
}

int bn_gpu_policy_all2_q4q6_moe_all2_fast_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_ALL2_FAST",
                                  "BN_CUDA_ENABLE_QWEN2MOE_MOE_ALL2_FAST");
}

int bn_gpu_policy_all2_q4q6_moe_q8k_default_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q8K_DEFAULT",
                                  "BN_CUDA_DISABLE_QWEN2MOE_MOE_Q8K_DEFAULT");
}

int bn_gpu_policy_all2_q4q6_route_q8k_default_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_ROUTE_Q8K_DEFAULT",
                                  "BN_CUDA_DISABLE_QWEN2MOE_ROUTE_Q8K_DEFAULT");
}

int bn_gpu_policy_all2_q4q6_route_q8_1_prequant_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREQUANT",
                                  "BN_CUDA_ENABLE_QWEN2MOE_ROUTE_Q8_1_PREQUANT");
}

int bn_gpu_policy_all2_q4q6_fast_q8k_gateup_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP",
                                  "BN_CUDA_ENABLE_QWEN2MOE_FAST_Q8K_GATEUP");
}

int bn_gpu_policy_all2_q4q6_fast_q8k_gateup_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP",
                                  "BN_CUDA_DISABLE_QWEN2MOE_FAST_Q8K_GATEUP");
}

int bn_gpu_policy_all2_q4q6_q6k_pair_down_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN",
                                  "BN_CUDA_ENABLE_QWEN2MOE_Q6K_PAIR_DOWN");
}

int bn_gpu_policy_all2_q4q6_q6k_pair_down_f32_layers_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS");
}

int bn_gpu_policy_all2_q4q6_q6k_pair_down_f32_layer_selected(int layer) {
    return gpu_policy_env_layer_selected(
        "BN_CUDA_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS", layer);
}

int bn_gpu_policy_all2_q4q6_q6k_ordered_down_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN",
                                  "BN_CUDA_ENABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
}

int bn_gpu_policy_all2_q4q6_q6k_ordered_down_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN",
                                  "BN_CUDA_DISABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
}

int bn_gpu_policy_all2_q4q6_q6k_f32_down_default_enabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
}

int bn_gpu_policy_all2_q4q6_q6k_f32_down_default_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
}

int bn_gpu_policy_all2_q4q6_q6k_f32_all2_down_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_ALL2_DOWN",
                                  "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_ALL2_DOWN");
}

int bn_gpu_policy_all2_q4q6_q6k_f32_cache_enabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_CACHE",
                                  "BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_CACHE");
}

int bn_gpu_policy_all2_q4q6_q6k_float_4row_down_default_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN_DEFAULT");
}

int bn_gpu_policy_all2_q4q6_q6k_float_4row_down_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN");
}

int bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_layer_selected(
    int layer) {
    const char *env = gpu_policy_env_value(
        "BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS");
    return !env || !*env ||
           gpu_policy_env_layer_selected(
               "BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS",
               "BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS", layer);
}

int bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_default_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT");
}

int bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN");
}

float bn_gpu_policy_all2_q4q6_down_skip_eps_or_default(float default_eps) {
    return gpu_policy_env_float_or_default("BN_CUDA_ALL2_Q4Q6_DOWN_SKIP_EPS",
                                           "BN_CUDA_QWEN2MOE_DOWN_SKIP_EPS",
                                           default_eps);
}

int bn_gpu_policy_all2_q4q6_moe_cpu_attention_safe_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
}

int bn_gpu_policy_all2_q4q6_moe_q6_logits_refine_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE");
}

int bn_gpu_policy_all2_q4q6_moe_cpu_moe_safe_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE");
}

int bn_gpu_policy_all2_q4q6_moe_exact_attention_disabled(void) {
    return gpu_policy_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN",
                                  "BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN");
}

int bn_gpu_policy_all2_q4q6_moe_cpu_route_resident_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT");
}

int bn_gpu_policy_all2_q4q6_moe_exact_gpu_route_requested(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE");
}

int bn_gpu_policy_all2_q4q6_moe_exact_gpu_route_disabled(void) {
    return gpu_policy_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE");
}

int bn_gpu_policy_all2_q4q6_moe_route_selection_enabled(void) {
    return bn_gpu_policy_cuda_moe_router_gpu_enabled() ||
           bn_gpu_policy_all2_q4q6_moe_exact_gpu_route_requested();
}

void bn_gpu_policy_all2_q4q6_moe_route_layer_range(int *from_layer,
                                                   int *to_layer) {
    const char *env;

    if (from_layer)
        *from_layer = -1;
    if (to_layer)
        *to_layer = -1;

    env = gpu_policy_env_value("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER",
                               "BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
    if (env && from_layer)
        *from_layer = atoi(env);
    env = gpu_policy_env_value("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER",
                               "BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
    if (env && to_layer)
        *to_layer = atoi(env);
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

int bn_gpu_policy_compare_attention_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_COMPARE_ATTENTION_LAYER",
                              default_layer);
}

int bn_gpu_policy_compare_attention_pos_or_default(int default_pos) {
    return env_int_or_default("BN_GPU_COMPARE_ATTENTION_POS", default_pos);
}

int bn_gpu_policy_compare_gqa_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_COMPARE_GQA_LAYER", default_layer);
}

int bn_gpu_policy_compare_gqa_pos_or_default(int default_pos) {
    return env_int_or_default("BN_GPU_COMPARE_GQA_POS", default_pos);
}

int bn_gpu_policy_compare_qkv_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_COMPARE_QKV_LAYER", default_layer);
}

int bn_gpu_policy_compare_qkv_pos_or_default(int default_pos) {
    return env_int_or_default("BN_GPU_COMPARE_QKV_POS", default_pos);
}

int bn_gpu_policy_compare_ffn_down_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_COMPARE_FFN_DOWN_LAYER",
                              default_layer);
}

int bn_gpu_policy_compare_ffn_down_pos_or_default(int default_pos) {
    return env_int_or_default("BN_GPU_COMPARE_FFN_DOWN_POS", default_pos);
}

int bn_gpu_policy_compare_ffn_state_layer_or_default(int default_layer) {
    return env_int_or_default("BN_GPU_COMPARE_FFN_STATE_LAYER",
                              default_layer);
}

int bn_gpu_policy_compare_ffn_state_pos_or_default(int default_pos) {
    return env_int_or_default("BN_GPU_COMPARE_FFN_STATE_POS", default_pos);
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
