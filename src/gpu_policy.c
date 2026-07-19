#include "gpu_policy.h"
#include "backend_quant.h"
#include "model_arch.h"
#include <stdint.h>
#include <stdlib.h>

static int gpu_policy_compat_env_enabled(const char *name,
                                         const char *compat_name) {
    return getenv(name) != NULL ||
           (compat_name && getenv(compat_name) != NULL);
}

static int gpu_policy_compat_env_enabled2(const char *name,
                                          const char *compat_name,
                                          const char *compat_name2) {
    return gpu_policy_compat_env_enabled(name, compat_name) ||
           (compat_name2 && getenv(compat_name2) != NULL);
}

static const char *gpu_policy_compat_env_value(const char *name,
                                               const char *compat_name) {
    const char *env = getenv(name);
    return env ? env : (compat_name ? getenv(compat_name) : NULL);
}

static int gpu_policy_compat_env_layer_selected(const char *name,
                                                const char *compat_name,
                                                int layer) {
    const char *env = gpu_policy_compat_env_value(name, compat_name);
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

static float gpu_policy_compat_env_float_or_default(const char *name,
                                                    const char *compat_name,
                                                    float default_value) {
    const char *env = gpu_policy_compat_env_value(name, compat_name);
    if (!env || !*env) return default_value;
    return (float)atof(env);
}

static int cuda_q8_0_prepared_input_split_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_Q8_0_PREPARED_INPUT_SPLIT",
        "BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
}

static int cuda_q8_0_prepared_input_split_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_Q8_0_PREPARED_INPUT_SPLIT",
        "BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
}

static int cuda_q8_prepared_input_requested(void) {
    return gpu_policy_compat_env_enabled("BN_CUDA_ENABLE_Q8_PREPARED_INPUT",
                                         "BN_CUDA_ENABLE_Q8_PREQ");
}

static int cuda_q8_prepared_input_disabled(void) {
    return gpu_policy_compat_env_enabled("BN_CUDA_DISABLE_Q8_PREPARED_INPUT",
                                         "BN_CUDA_DISABLE_Q8_PREQ");
}

static int cuda_q8_prepared_input_logits_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_Q8_PREPARED_INPUT_LOGITS",
        "BN_CUDA_ENABLE_Q8_PREQ_LOGITS");
}

static int cuda_q8_prepared_input_logits_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_Q8_PREPARED_INPUT_LOGITS",
        "BN_CUDA_DISABLE_Q8_PREQ_LOGITS");
}

static int cuda_moe_route_q8k_prepared_input_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREPARED_INPUT",
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT");
}

static int cuda_moe_route_q8_1_prepared_input_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT",
        "BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT");
}

static int cuda_moe_route_q8_1_prepared_input_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT",
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT");
}

static int cuda_q8_0_ssm_prepared_input_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_Q8_0_SSM_PREPARED_INPUT",
        "BN_CUDA_DISABLE_Q8_0_SSM_PREQ");
}

static int cuda_q8_mixed_prepared_input_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_Q8_MIXED_PREPARED_INPUT",
        "BN_CUDA_ENABLE_Q8_MIXED_PREQ");
}

static int all_active_two_route_q8_1_prepared_input_requested(void) {
    return gpu_policy_compat_env_enabled2(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREPARED_INPUT",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREQUANT",
        "BN_CUDA_ENABLE_QWEN2MOE_ROUTE_Q8_1_PREQUANT");
}

static int small_dense_native_quant_cpu_attention_safe_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE");
}

static int small_dense_exact_native_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8",
        "BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8");
}

static int small_dense_exact_ffn_down_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN",
        "BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
}

static int small_dense_prefill_disabled(void) {
    return gpu_policy_compat_env_enabled("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL",
                                         "BN_CUDA_DISABLE_SMALL_QWEN_PREFILL");
}

static int native_quant_logits_refine_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
        "BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
}

static int native_quant_logits_refine_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
        "BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
}

static int all_active_two_kquant_moe_fast_ffn_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
        "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
}

static int all_active_two_kquant_moe_fast_graph_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_FAST_GRAPH",
        "BN_CUDA_DISABLE_QWEN2MOE_FAST_MOE_GRAPH");
}

static int all_active_two_kquant_moe_cublas_decode_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_CUBLAS_DECODE",
        "BN_CUDA_ENABLE_QWEN2MOE_MOE_CUBLAS_DECODE");
}

static int all_active_two_kquant_moe_fast_route_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_ALL2_FAST",
        "BN_CUDA_ENABLE_QWEN2MOE_MOE_ALL2_FAST");
}

static int all_active_two_kquant_moe_q8k_default_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q8K_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_MOE_Q8K_DEFAULT");
}

static int all_active_two_kquant_route_q8k_default_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_ROUTE_Q8K_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_ROUTE_Q8K_DEFAULT");
}

static int all_active_two_kquant_fast_q8k_gateup_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP",
        "BN_CUDA_ENABLE_QWEN2MOE_FAST_Q8K_GATEUP");
}

static int all_active_two_kquant_fast_q8k_gateup_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP",
        "BN_CUDA_DISABLE_QWEN2MOE_FAST_Q8K_GATEUP");
}

static int all_active_two_kquant_q6k_pair_down_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_PAIR_DOWN");
}

static int all_active_two_kquant_q6k_pair_down_f32_layers_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS");
}

static int all_active_two_kquant_q6k_pair_down_f32_layer_selected(int layer) {
    return gpu_policy_compat_env_layer_selected(
        "BN_CUDA_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS", layer);
}

static int all_active_two_kquant_q6k_ordered_down_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
}

static int all_active_two_kquant_q6k_ordered_down_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
}

static int all_active_two_kquant_q6k_f32_down_default_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
}

static int all_active_two_kquant_q6k_f32_down_default_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
}

static int all_active_two_kquant_q6k_f32_all_active_down_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_ALL2_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_ALL2_DOWN");
}

static int all_active_two_kquant_q6k_f32_cache_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_CACHE",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_CACHE");
}

static int all_active_two_kquant_q6k_float_4row_down_default_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN_DEFAULT");
}

static int all_active_two_kquant_q6k_float_4row_down_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN");
}

static const char *all_active_two_kquant_q6k_f32_exact_4row_down_layers(void) {
    return gpu_policy_compat_env_value(
        "BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS");
}

static int all_active_two_kquant_q6k_f32_exact_4row_down_layer_selected(
    int layer) {
    return gpu_policy_compat_env_layer_selected(
        "BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS", layer);
}

static int all_active_two_kquant_q6k_f32_exact_4row_down_default_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT");
}

static int all_active_two_kquant_q6k_f32_exact_4row_down_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN");
}

static float all_active_two_kquant_down_skip_eps_or_default(float default_eps) {
    return gpu_policy_compat_env_float_or_default(
        "BN_CUDA_ALL2_Q4Q6_DOWN_SKIP_EPS",
        "BN_CUDA_QWEN2MOE_DOWN_SKIP_EPS", default_eps);
}

static int all_active_two_kquant_moe_cpu_attention_safe_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
}

static int all_active_two_kquant_moe_logits_refine_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE");
}

static int all_active_two_kquant_moe_cpu_moe_safe_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE");
}

static int all_active_two_kquant_moe_exact_attention_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN",
        "BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN");
}

static int all_active_two_kquant_moe_cpu_route_resident_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT");
}

static int all_active_two_kquant_moe_exact_gpu_route_requested(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE");
}

static int all_active_two_kquant_moe_exact_gpu_route_disabled(void) {
    return gpu_policy_compat_env_enabled(
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE");
}

static const char *all_active_two_kquant_moe_route_from_layer_value(void) {
    return gpu_policy_compat_env_value(
        "BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER",
        "BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
}

static const char *all_active_two_kquant_moe_route_to_layer_value(void) {
    return gpu_policy_compat_env_value(
        "BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER",
        "BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
}

typedef struct {
    int flash_default;
    int default_flash_max_kv;
    int large_graph_native;
    int small_dense_native;
    int all_active_two_kquant_moe;
    int cpu_attention_fallback;
    int small_dense_exact_native;
    int prefill_decode_fallback;
    int prefill_chain;
    int matvec_fallback;
    int dense_batch_prefill_shape;
    int lazy_moe_aux_cache;
    int native_quant_logits_refine_default;
    int all_active_two_kquant_moe_logits_refine_default;
    int decode_graph_cache;
    int moe_exact_attention;
    int ssm_graph;
    int large_hybrid_argmax;
    int all_active_two_moe_direct_route;
    int resident_moe_ffn;
    int moe_gateup_split;
    int individual_upload_quant_only;
    int logits_q6_f32_cache;
    int logits_f16_cache;
    int moe_prefers_quant_only;
    int fused_gateup_requires_backend_opt_in;
    int moe_down_cublas_cache;
    int suppress_implicit_kquant_logits_refine;
    int suppress_implicit_native_quant_logits_refine;
} BnGPUPolicyBackendCaps;

static const BnGPUPolicyBackendCaps GPU_POLICY_BACKEND_CAPS_NONE = {0};

static const BnGPUPolicyBackendCaps GPU_POLICY_BACKEND_CAPS_CUDA = {
    .flash_default = 1,
    .default_flash_max_kv = 2048,
    .large_graph_native = 1,
    .small_dense_native = 1,
    .all_active_two_kquant_moe = 1,
    .cpu_attention_fallback = 1,
    .small_dense_exact_native = 1,
    .prefill_decode_fallback = 1,
    .prefill_chain = 1,
    .matvec_fallback = 1,
    .dense_batch_prefill_shape = 1,
    .lazy_moe_aux_cache = 1,
    .native_quant_logits_refine_default = 1,
    .all_active_two_kquant_moe_logits_refine_default = 1,
    .decode_graph_cache = 1,
    .moe_exact_attention = 1,
    .ssm_graph = 1,
    .large_hybrid_argmax = 1,
    .all_active_two_moe_direct_route = 1,
    .resident_moe_ffn = 1,
    .moe_gateup_split = 1,
    .individual_upload_quant_only = 1,
    .logits_q6_f32_cache = 1,
    .logits_f16_cache = 1,
    .moe_prefers_quant_only = 1,
    .fused_gateup_requires_backend_opt_in = 1,
    .moe_down_cublas_cache = 1,
    .suppress_implicit_kquant_logits_refine = 1,
    .suppress_implicit_native_quant_logits_refine = 1,
};

static const BnGPUPolicyBackendCaps *
gpu_policy_backend_caps(const BnGPUBackend *gpu) {
    if (!gpu)
        return &GPU_POLICY_BACKEND_CAPS_NONE;
    switch (gpu->kind) {
        case BN_GPU_BACKEND_CUDA:
            return &GPU_POLICY_BACKEND_CAPS_CUDA;
        case BN_GPU_BACKEND_METAL:
        case BN_GPU_BACKEND_WEBGPU:
        default:
            return &GPU_POLICY_BACKEND_CAPS_NONE;
    }
}

static int gpu_policy_cuda_moe_routed_ffn_enabled(int eligible) {
    return eligible && getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN") == NULL;
}

int bn_gpu_policy_moe_resident_routed_ffn_enabled(int eligible) {
    return gpu_policy_cuda_moe_routed_ffn_enabled(eligible);
}

BnBackendPlacement bn_gpu_policy_backend_placement(const BnGPUBackend *gpu) {
    if (!gpu)
        return BN_BACKEND_GPU_UNKNOWN;
    switch (gpu->kind) {
        case BN_GPU_BACKEND_METAL: return BN_BACKEND_METAL;
        case BN_GPU_BACKEND_WEBGPU: return BN_BACKEND_WEBGPU;
        case BN_GPU_BACKEND_CUDA: return BN_BACKEND_CUDA;
        default: return BN_BACKEND_GPU_UNKNOWN;
    }
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

static int gpu_policy_moe_f16_aux_cache_auto_enabled(const BnConfig *c) {
    return bn_model_arch_uses_more_than_two_expert_moe(c) ||
           bn_model_arch_uses_two_expert_all_active_moe(c);
}

int bn_gpu_policy_moe_f16_aux_cache_auto_enabled(const BnConfig *c) {
    return gpu_policy_moe_f16_aux_cache_auto_enabled(c);
}

static int gpu_policy_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type) {
    return bn_backend_quant_moe_route_q4_down(gate_type, up_type, down_type,
                                              1) ||
           bn_backend_quant_moe_route_q8(gate_type, up_type, down_type);
}

int bn_gpu_policy_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type) {
    return gpu_policy_moe_resident_routed_ffn_quant_eligible(gate_type,
                                                             up_type,
                                                             down_type);
}

static int gpu_policy_moe_all_f16_cache_forced(void) {
    return getenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE") != NULL;
}

int bn_gpu_policy_moe_all_f16_cache_forced(void) {
    return gpu_policy_moe_all_f16_cache_forced();
}

static int gpu_policy_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int q8_f16_cache) {
    if (!gpu || !gpu->buffer_create_f16_cache ||
        getenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE") != NULL)
        return 0;
    if (gpu_policy_moe_all_f16_cache_forced())
        return 1;
    if (!q8_f16_cache)
        return 0;
    return bn_backend_quant_moe_all_f16_cache_supported(tensor_type);
}

int bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int q8_f16_cache) {
    return gpu_policy_moe_all_f16_cache_enabled_for_type(gpu, tensor_type,
                                                         q8_f16_cache);
}

static int gpu_policy_moe_gateup_f16_cache_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE") == NULL;
}

int bn_gpu_policy_moe_gateup_f16_cache_enabled(int eligible) {
    return gpu_policy_moe_gateup_f16_cache_enabled(eligible);
}

static int gpu_policy_partial_moe_f16_cache_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE") != NULL;
}

int bn_gpu_policy_partial_moe_f16_cache_enabled(int eligible) {
    return gpu_policy_partial_moe_f16_cache_enabled(eligible);
}

static int gpu_policy_moe_fit_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_FIT") != NULL;
}

int bn_gpu_policy_moe_residency_fit_debug_enabled(void) {
    return gpu_policy_moe_fit_debug_enabled();
}

static int gpu_policy_keep_individual_f16_cache_enabled(void) {
    return getenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE") != NULL;
}

static int gpu_policy_moe_lazy_aux_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE") != NULL;
}

int bn_gpu_policy_moe_lazy_aux_cache_enabled(void) {
    return gpu_policy_moe_lazy_aux_cache_enabled();
}

static int gpu_policy_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->individual_upload_quant_only &&
           gpu &&
           gpu->buffer_create_quant_only &&
           !gpu_policy_keep_individual_f16_cache_enabled();
}

int bn_gpu_policy_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_individual_upload_quant_only_enabled(gpu);
}

static int gpu_policy_logits_q6_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu_policy_backend_caps(gpu)->logits_q6_f32_cache &&
           gpu &&
           gpu->buffer_create_q6_f32_cache &&
           bn_backend_quant_logits_q6_f32_cache_supported(tensor_type) &&
           getenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE") == NULL;
}

int bn_gpu_policy_logits_q6_f32_cache_enabled(const BnGPUBackend *gpu,
                                              int tensor_type) {
    return gpu_policy_logits_q6_f32_cache_enabled(gpu, tensor_type);
}

static int gpu_policy_logits_f16_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->logits_f16_cache &&
           gpu &&
           gpu->buffer_create_f16_cache &&
           getenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE") != NULL;
}

int bn_gpu_policy_logits_f16_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_logits_f16_cache_enabled(gpu);
}

int bn_gpu_policy_cuda_cublas_logits_enabled(void) {
    return getenv("BN_CUDA_ENABLE_CUBLAS_LOGITS") != NULL;
}

int bn_gpu_policy_cuda_f32_logits_matvec_enabled(void) {
    return getenv("BN_CUDA_ENABLE_F32_LOGITS_MATVEC") != NULL &&
           getenv("BN_CUDA_DISABLE_F32_LOGITS_MATVEC") == NULL;
}

int bn_gpu_policy_cuda_f16_logits_matvec_enabled(void) {
    return getenv("BN_CUDA_ENABLE_F16_LOGITS_MATVEC") != NULL;
}

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(
    const BnGPUBackend *gpu) {
    return gpu && gpu->buffer_create_q6_f32_cache &&
           !gpu_policy_moe_all_f16_cache_forced() &&
           getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced(void) {
    return getenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE") != NULL;
}

int bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(int cols) {
    return cols > 1024 ||
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced();
}

static int gpu_policy_moe_down_q6_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache) {
    return !force_f16_cache &&
           bn_backend_quant_moe_down_q6_f32_cache_supported(tensor_type) &&
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(cols) &&
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(gpu);
}

int bn_gpu_policy_moe_down_q6_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache) {
    return gpu_policy_moe_down_q6_f32_cache_preferred(
        gpu, tensor_type, cols, force_f16_cache);
}

static size_t gpu_policy_moe_down_q6_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts) {
    if (!gpu_policy_moe_down_q6_f32_cache_preferred(
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

size_t bn_gpu_policy_moe_down_q6_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts) {
    return gpu_policy_moe_down_q6_f32_cache_bytes(
        gpu, tensor_type, rows, cols, n_experts);
}

static int gpu_policy_moe_down_q6_f32_cache_requires_full_buffer(
    int tensor_type) {
    return bn_backend_quant_moe_down_q6_f32_cache_supported(tensor_type) &&
           bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced();
}

int bn_gpu_policy_moe_down_q6_f32_cache_requires_full_buffer(
    int tensor_type) {
    return gpu_policy_moe_down_q6_f32_cache_requires_full_buffer(tensor_type);
}

static int gpu_policy_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu && gpu->buffer_create_q6_f32_cache &&
           bn_backend_quant_moe_down_q4_f32_cache_supported(tensor_type) &&
           getenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu_policy_moe_down_q4_f32_cache_enabled(gpu, tensor_type);
}

static int gpu_policy_moe_quant_only_after_cache(int tensor_type,
                                                 int q8_f16_cache) {
    return bn_backend_quant_moe_quant_only_after_cache(tensor_type,
                                                       q8_f16_cache);
}

int bn_gpu_policy_moe_quant_only_after_cache(int tensor_type,
                                             int q8_f16_cache) {
    return gpu_policy_moe_quant_only_after_cache(tensor_type, q8_f16_cache);
}

int bn_gpu_policy_moe_prefers_quant_only(const BnGPUBackend *gpu,
                                         int tensor_type) {
    return gpu_policy_backend_caps(gpu)->moe_prefers_quant_only &&
           bn_backend_quant_moe_prefers_quant_only(tensor_type);
}

int bn_gpu_policy_matvec_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MATVEC") != NULL;
}

int bn_gpu_policy_matvec_type_disabled(int tensor_type) {
    if (bn_backend_quant_is_q8_0(tensor_type))
        return getenv("BN_CUDA_DISABLE_Q8_0") != NULL;
    if (bn_backend_quant_is_q5_0(tensor_type))
        return getenv("BN_CUDA_DISABLE_Q5_0") != NULL;
    if (bn_backend_quant_is_q4k(tensor_type))
        return getenv("BN_CUDA_DISABLE_Q4_K") != NULL;
    if (bn_backend_quant_is_q5k(tensor_type))
        return getenv("BN_CUDA_DISABLE_Q5_K") != NULL;
    if (bn_backend_quant_moe_down_is_q6k(tensor_type))
        return getenv("BN_CUDA_DISABLE_Q6_K") != NULL;
    if (bn_backend_quant_is_q8k(tensor_type))
        return getenv("BN_CUDA_DISABLE_Q8_K") != NULL;
    return 0;
}

int bn_gpu_policy_matvec_type_supported(int tensor_type) {
    return bn_backend_quant_gpu_matvec_supported(tensor_type) &&
           !bn_gpu_policy_matvec_type_disabled(tensor_type);
}

int bn_gpu_policy_matmul_batch_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MATMUL_BATCH") == NULL;
}

int bn_gpu_policy_matvec_batch_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MATVEC_BATCH") == NULL;
}

static int small_kquant_native_enabled(int force_float_kquant) {
    if (getenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE"))
        return 1;
    return getenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE") == NULL &&
           !force_float_kquant;
}

int bn_gpu_policy_small_kquant_native_enabled(int force_float_kquant) {
    return small_kquant_native_enabled(force_float_kquant);
}

static int small_kquant_native_disabled(void) {
    return getenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE") != NULL;
}

int bn_gpu_policy_small_kquant_native_disabled(void) {
    return small_kquant_native_disabled();
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

static size_t gpu_policy_layout_reserve_bytes(void) {
    return mb_to_bytes_saturating(
        env_mb_or_default("BN_CUDA_LAYOUT_RESERVE_MB", 512));
}

size_t bn_gpu_policy_layout_reserve_bytes(void) {
    return gpu_policy_layout_reserve_bytes();
}

static size_t gpu_policy_moe_full_reserve_bytes(void) {
    return mb_to_bytes_saturating(
        env_mb_or_default("BN_CUDA_MOE_FULL_RESERVE_MB", 512));
}

size_t bn_gpu_policy_moe_full_reserve_bytes(void) {
    return gpu_policy_moe_full_reserve_bytes();
}

int bn_gpu_policy_cuda_cublas_matmul_enabled(void) {
    return getenv("BN_CUDA_DISABLE_CUBLAS_MATMUL") == NULL;
}

int bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(
    int default_index) {
    const char *env = getenv("BN_CUDA_CUBLAS_GEMM_ALGO");
    if (!env || !env[0])
        return default_index;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || *end != '\0')
        return default_index;
    if (v < 0)
        return -1;
    if (v >= 0 && v <= 23)
        return (int)v;
    return default_index;
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

int bn_gpu_policy_cuda_q8_0_prepared_input_split_enabled(void) {
    return cuda_q8_0_prepared_input_split_requested() &&
           !cuda_q8_0_prepared_input_split_disabled();
}

int bn_gpu_policy_cuda_q8_prepared_input_all_enabled(void) {
    return cuda_q8_prepared_input_requested() &&
           !cuda_q8_prepared_input_disabled();
}

int bn_gpu_policy_cuda_q8_prepared_input_logits_disabled(void) {
    return cuda_q8_prepared_input_logits_disabled();
}

int bn_gpu_policy_cuda_q8_prepared_input_logits_default_enabled(
    int prepared_input_logits_disabled) {
    return !prepared_input_logits_disabled &&
           (cuda_q8_prepared_input_logits_requested() ||
            !cuda_q8_prepared_input_logits_disabled());
}

int bn_gpu_policy_prepared_kquant_input_cache_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q8K_INPUT_CACHE") == NULL;
}

int bn_gpu_policy_cuda_force_quant_matmul_for_type(
    int tensor_type,
    int f16_q8_0_matmul_enabled) {
    return (bn_backend_quant_avoids_quant_matmul_on_f16_input(
                tensor_type) &&
            !f16_q8_0_matmul_enabled) ||
           (bn_backend_quant_force_q4k_quant_matmul_candidate(
                tensor_type) &&
            getenv("BN_CUDA_FORCE_Q4K_QUANT_MATMUL") != NULL) ||
           (bn_backend_quant_force_q6k_quant_matmul_candidate(
                tensor_type) &&
            getenv("BN_CUDA_FORCE_Q6K_QUANT_MATMUL") != NULL);
}

int bn_gpu_policy_cuda_q6k_4warp_long_enabled(int rows, int cols) {
    if (getenv("BN_CUDA_DISABLE_Q6K_4WARP_LONG") != NULL)
        return 0;
    if (rows == 1536 && cols == 8960 &&
        getenv("BN_CUDA_DISABLE_Q6K_4WARP_1536_8960") == NULL)
        return 1;
    if (rows >= 2560 && cols >= 8192 && cols <= 16384)
        return 1;
    return rows >= 2560 && cols >= 5120 && cols < 8192 &&
           getenv("BN_CUDA_ENABLE_Q6K_4WARP_5120") != NULL;
}

int bn_gpu_policy_cuda_q6k_5warp_exact_enabled(int rows, int cols) {
    if (rows == 1536 && cols == 8960 &&
        getenv("BN_CUDA_DISABLE_Q6K_5WARP_1536_8960") == NULL)
        return 1;
    if (rows == 2560 && cols == 9728 &&
        getenv("BN_CUDA_DISABLE_Q6K_5WARP_2560_9728") == NULL)
        return 1;
    return 0;
}

int bn_gpu_policy_cuda_q6k_3warp_exact_enabled(int rows, int cols) {
    if (rows == 1536 && cols == 8960 &&
        getenv("BN_CUDA_DISABLE_Q6K_3WARP_1536_8960") == NULL)
        return 1;
    if (rows == 2560 && cols == 9728 &&
        getenv("BN_CUDA_DISABLE_Q6K_3WARP_2560_9728") == NULL)
        return 1;
    return 0;
}

int bn_gpu_policy_cuda_q6k_2warp_long_enabled(int rows, int cols) {
    return rows >= 2560 && cols >= 8192 && cols <= 12288 &&
           getenv("BN_CUDA_ENABLE_Q6K_2WARP_LONG") != NULL &&
           getenv("BN_CUDA_DISABLE_Q6K_2WARP_LONG") == NULL;
}

int bn_gpu_policy_cuda_q6k_matvec4_shape_disabled(int rows, int cols) {
    if (rows == 1024 && cols == 2560 &&
        getenv("BN_CUDA_ENABLE_Q6K_MATVEC4_1024_2560") == NULL)
        return 1;
    if (rows == 512 && cols == 2048 &&
        getenv("BN_CUDA_ENABLE_Q6K_MATVEC4_512_2048") == NULL)
        return 1;
    return 0;
}

int bn_gpu_policy_cuda_q6k_moe_quant_down_preferred(int routed_q4,
                                                    int down_type,
                                                    int hidden_dim,
                                                    int n_experts,
                                                    int k) {
    return routed_q4 &&
           bn_backend_quant_moe_down_q6_f32_cache_supported(down_type) &&
           hidden_dim <= 1024 && (n_experts > 2 || k > 2) &&
           getenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_f32_cache_path_enabled(
    int routed_q4,
    int down_type,
    int has_f32_data,
    int prefer_quant_down,
    int dim,
    int hidden_dim,
    int n_experts,
    int k) {
    return bn_backend_quant_moe_down_q6_f32_cache_supported(down_type) &&
           has_f32_data && !prefer_quant_down &&
           !(routed_q4 && n_experts == 2 && k == 2 &&
             hidden_dim >= 4096 && dim <= 2048) &&
           getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_moe_down_4row_enabled(int hidden_dim) {
    return hidden_dim <= 1024 &&
           getenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW") == NULL;
}

int bn_gpu_policy_cuda_moe_down_8row_enabled(int hidden_dim) {
    return hidden_dim <= 1024 &&
           getenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_halfwarp_enabled(
    int down_type,
    int prefer_quant_down,
    int n_experts,
    int k) {
    return bn_backend_quant_moe_down_q6_f32_cache_supported(down_type) &&
           (n_experts > 2 || k > 2) &&
           (prefer_quant_down ||
            getenv("BN_CUDA_ENABLE_MOE_DOWN_HALFWARP") != NULL) &&
           getenv("BN_CUDA_DISABLE_MOE_DOWN_HALFWARP") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_split4_enabled(
    int down_type,
    int use_halfwarp,
    int n_experts,
    int k) {
    return !use_halfwarp &&
           bn_backend_quant_moe_down_q6_f32_cache_supported(down_type) &&
           (n_experts > 2 || k > 2) &&
           getenv("BN_CUDA_ENABLE_MOE_DOWN_SPLIT4") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_DOWN_SPLIT4") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_scatter_enabled(
    int down_type,
    int use_halfwarp,
    int use_split4) {
    return !use_halfwarp && !use_split4 &&
           bn_backend_quant_moe_down_q6_f32_cache_supported(down_type) &&
           getenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_scatter_16row_enabled(
    int use_scatter,
    int hidden_dim) {
    return use_scatter && hidden_dim <= 768 &&
           getenv("BN_CUDA_ENABLE_MOE_DOWN_SCATTER_16ROW") != NULL;
}

int bn_gpu_policy_cuda_q6k_moe_float_down_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_pair_down_enabled(
    int f32_down_default,
    int pair_down_f32_layer,
    int all_active_two_disable_pair_down) {
    return !f32_down_default && !pair_down_f32_layer &&
           !all_active_two_disable_pair_down &&
           getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_prefer_f32_down(
    int has_f32_data,
    int hidden_dim,
    int all_active_two_kquant,
    int all_active_two_f32_down) {
    return has_f32_data && hidden_dim >= 4096 &&
           (!all_active_two_kquant || all_active_two_f32_down) &&
           getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_f32_pair2_enabled(int n_experts,
                                                      int k) {
    return n_experts == 2 && k == 2 &&
           getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_f32_pair2_4row_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW") == NULL;
}

int bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(
    int all_active_two_kquant) {
    return all_active_two_kquant &&
           getenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM") == NULL;
}

int bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled(
    int all_active_two_kquant) {
    return all_active_two_kquant &&
           getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_4row_sum_enabled(
    int all_active_two_kquant,
    int k,
    int hidden_dim) {
    return !all_active_two_kquant && k <= 8 && hidden_dim <= 1024 &&
           getenv("BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_8row_sum_enabled(
    int k8_4row_sum,
    int hidden_dim) {
    return k8_4row_sum && hidden_dim <= 1024 &&
           getenv("BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM") != NULL;
}

int bn_gpu_policy_all_active_two_kquant_moe_down_fixed_enabled(
    int all_active_two_kquant) {
    return all_active_two_kquant &&
           getenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_resid_rmsnorm_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_exact_2048_768_enabled(
    int dim,
    int hidden_dim,
    int k) {
    return dim == 2048 && hidden_dim == 768 && k == 8 &&
           getenv("BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768") == NULL;
}

int bn_gpu_policy_all_active_two_kquant_moe_down_accum_4row_enabled(
    void) {
    return getenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW") != NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_q8k_pair_4row_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_f32_cache_enabled(
    int has_f32_data,
    int all_active_two_disable_f32_cache) {
    return has_f32_data && !all_active_two_disable_f32_cache &&
           getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_q6k_moe_down_f16_cache_enabled(int has_f16_data) {
    return has_f16_data &&
           getenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE") == NULL;
}

int bn_gpu_policy_cuda_q4k_moe_down_f32_cache_enabled(int has_f32_data) {
    return has_f32_data &&
           getenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE") == NULL;
}

int bn_gpu_policy_cuda_q4k_moe_pair_down_enabled(int n_experts,
                                                 int k,
                                                 int hidden_dim) {
    return n_experts == 2 && k == 2 && hidden_dim >= 4096 &&
           getenv("BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN") == NULL;
}

int bn_gpu_policy_cuda_q4k_moe_down_8row_enabled(int hidden_dim) {
    return hidden_dim <= 1024 &&
           getenv("BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW") == NULL;
}

int bn_gpu_policy_cuda_q4k_q8k_moe_gateup_enabled(int n_tokens,
                                                  int dim,
                                                  int allow_small_dim) {
    return bn_gpu_policy_kquant_dot_enabled() &&
           (n_tokens <= 1 || (allow_small_dim && dim <= 2048) ||
            getenv("BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP") != NULL);
}

int bn_gpu_policy_cuda_q4k_moe_gateup_8row_enabled(int dim) {
    return dim <= 2048 &&
           getenv("BN_CUDA_DISABLE_MOE_GATEUP_8ROW") == NULL;
}

int bn_gpu_policy_cuda_q4k_moe_gateup_split_enabled(int dim,
                                                    int n_experts) {
    return dim <= 2048 && n_experts > 2 &&
           getenv("BN_CUDA_ENABLE_MOE_GATEUP_SPLIT") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT") == NULL;
}

int bn_gpu_policy_cuda_moe_route_q8k_prepared_input_enabled(
    int dim,
    int all_active_two_kquant) {
    return (dim % BN_QK_K) == 0 &&
           all_active_two_kquant &&
           !bn_gpu_policy_all_active_two_kquant_moe_q8k_default_disabled() &&
           !bn_gpu_policy_all_active_two_kquant_route_q8k_default_disabled() &&
           !cuda_moe_route_q8k_prepared_input_disabled();
}

int bn_gpu_policy_cuda_moe_route_q8_1_prepared_input_enabled(
    int dim,
    int all_active_two_kquant,
    int exact_silu) {
    return (dim % 32) == 0 &&
           all_active_two_kquant &&
           !exact_silu &&
           (bn_gpu_policy_all_active_two_kquant_route_q8_1_prepared_input_enabled() ||
            cuda_moe_route_q8_1_prepared_input_requested()) &&
           getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT") == NULL &&
           getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2") == NULL &&
           !cuda_moe_route_q8_1_prepared_input_disabled();
}

int bn_gpu_policy_cuda_moe_router_fused_topk_enabled(int n_experts,
                                                     int route_block) {
    return n_experts <= 256 &&
           !route_block &&
           getenv("BN_CUDA_ENABLE_MOE_ROUTER_FUSED_TOPK") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_FUSED_TOPK") == NULL;
}

int bn_gpu_policy_cuda_moe_router_warp_disabled(int route_block) {
    return route_block || getenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP") != NULL;
}

int bn_gpu_policy_cuda_moe_router_4warp_enabled(int dim) {
    return dim >= 2048 &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_4WARP") == NULL;
}

int bn_gpu_policy_cuda_moe_router_2warp_enabled(int dim) {
    return dim >= 2048 &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_2WARP") == NULL;
}

int bn_gpu_policy_cuda_moe_router_warp_topk_enabled(int n_experts) {
    return n_experts <= 256 &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP_TOPK") == NULL;
}

int bn_gpu_policy_cuda_q8_moe_q8_1_batch_enabled(int routed_q8) {
    return routed_q8 &&
           getenv("BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1") == NULL;
}

int bn_gpu_policy_cuda_q8_moe_q8x_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q8_MOE_Q8X") == NULL;
}

int bn_gpu_policy_cuda_q8_moe_gateup_2row_enabled(int hidden_dim) {
    return hidden_dim <= 1024 &&
           getenv("BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW") == NULL;
}

int bn_gpu_policy_cuda_q8_moe_down_4row_enabled(int hidden_dim) {
    return hidden_dim <= 1024 &&
           getenv("BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW") != NULL;
}

int bn_gpu_policy_cuda_q8_moe_down_2row_enabled(int hidden_dim) {
    return hidden_dim <= 1024 &&
           getenv("BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW") == NULL;
}

int bn_gpu_policy_cuda_moe_all_active_two_fast_enabled(
    int all_active_two_q4_or_q6) {
    return getenv("BN_CUDA_DISABLE_MOE_ALL2_FAST") == NULL &&
           (!all_active_two_q4_or_q6 ||
            bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled());
}

int bn_gpu_policy_cuda_moe_q4k_q8k_dot_enabled(int use_all_active_two_q8k_default,
                                               int fast_q8k_gateup,
                                               int all_active_two_kquant,
                                               int hidden_dim,
                                               int dim) {
    return (use_all_active_two_q8k_default ||
            fast_q8k_gateup ||
            (all_active_two_kquant &&
             getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2") != NULL) ||
            (hidden_dim > 2048 && dim > 2048) ||
            getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT") != NULL) &&
           getenv("BN_CUDA_DISABLE_MOE_Q4K_Q8K_DOT") == NULL;
}

int bn_gpu_policy_cuda_moe_internal_profile_enabled(int profile) {
    return profile && getenv("BN_CUDA_PROFILE_MOE_INTERNAL") != NULL;
}

int bn_gpu_policy_cuda_moe_q4k_all_active_two_fixed_4row_enabled(
    int prepared_q8k_input,
    int all_active_two_fast_enabled) {
    return prepared_q8k_input &&
           all_active_two_fast_enabled &&
           getenv("BN_CUDA_DISABLE_MOE_Q4K_ALL2_FIXED") == NULL &&
           getenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW") == NULL;
}

int bn_gpu_policy_cuda_moe_q4k_gateup_4row_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW") != NULL;
}

static int gpu_policy_decode_logits_cache_enabled(int gpu_logits_need_cpu) {
    return getenv("BN_CUDA_ENABLE_LOGITS_CACHE") != NULL &&
           !gpu_logits_need_cpu;
}

int bn_gpu_policy_decode_logits_cache_enabled(int gpu_logits_need_cpu) {
    return gpu_policy_decode_logits_cache_enabled(gpu_logits_need_cpu);
}

static int gpu_policy_moe_decode_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_moe_decode_cache_enabled(void) {
    return gpu_policy_moe_decode_cache_enabled();
}

static int gpu_policy_moe_decode_cache_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_moe_decode_cache_disabled(void) {
    return gpu_policy_moe_decode_cache_disabled();
}

static int gpu_policy_decode_cache_disabled(void) {
    return getenv("BN_CUDA_DISABLE_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_decode_cache_disabled(void) {
    return gpu_policy_decode_cache_disabled();
}

static int gpu_policy_native_quant_decode_cache_disabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE") != NULL;
}

int bn_gpu_policy_native_quant_decode_cache_disabled(void) {
    return gpu_policy_native_quant_decode_cache_disabled();
}

static int gpu_policy_logits_argmax_disabled(void) {
    return getenv("BN_CUDA_DISABLE_LOGITS_ARGMAX") != NULL;
}

int bn_gpu_policy_logits_argmax_disabled(void) {
    return gpu_policy_logits_argmax_disabled();
}

static int gpu_policy_dense_logits_argmax_enabled(void) {
    return getenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX") != NULL;
}

int bn_gpu_policy_dense_logits_argmax_enabled(void) {
    return gpu_policy_dense_logits_argmax_enabled();
}

static int gpu_policy_moe_logits_mmvq_argmax_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX") != NULL;
}

int bn_gpu_policy_moe_logits_mmvq_argmax_enabled(void) {
    return gpu_policy_moe_logits_mmvq_argmax_enabled();
}

static int gpu_policy_moe_logits_mmvq_argmax_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX") != NULL;
}

int bn_gpu_policy_moe_logits_mmvq_argmax_disabled(void) {
    return gpu_policy_moe_logits_mmvq_argmax_disabled();
}

int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(int rows,
                                                           int cols) {
    return !gpu_policy_moe_logits_mmvq_argmax_disabled() &&
           rows >= 50000 &&
           (cols == 1536 ||
            gpu_policy_moe_logits_mmvq_argmax_enabled());
}

int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_enabled(int use_mmvq,
                                                           int rows,
                                                           int cols) {
    return use_mmvq && rows == 151936 && cols == 1536 &&
           getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536") == NULL;
}

int bn_gpu_policy_cuda_moe_logits_mmvq_1warp16_1536_enabled(
    int use_1warp8) {
    return use_1warp8 &&
           getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_1WARP16_1536") != NULL;
}

int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_unroll_enabled(
    int use_1warp8,
    int use_1warp16) {
    return use_1warp8 && !use_1warp16 &&
           getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536_UNROLL") ==
               NULL;
}

int bn_gpu_policy_cuda_argmax_fast_enabled(void) {
    return getenv("BN_CUDA_DISABLE_ARGMAX_FAST") == NULL;
}

int bn_gpu_policy_cuda_optimistic_argmax_penalty_enabled(void) {
    return getenv("BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY") != NULL;
}

int bn_gpu_policy_cuda_q5_matvec4_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q5_MATVEC4") != NULL;
}

int bn_gpu_policy_cuda_q5_warp_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q5_WARP") != NULL;
}

int bn_gpu_policy_cuda_q5k_deint_pair_matvec_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q5K_DEINT_PAIR_MATVEC") != NULL;
}

int bn_gpu_policy_cuda_q5k_4warp_enabled(int cols) {
    return cols <= 8192 && getenv("BN_CUDA_DISABLE_Q5K_4WARP") == NULL;
}

int bn_gpu_policy_cuda_q5k_split_4warp_enabled(int cols) {
    return bn_gpu_policy_cuda_q5k_4warp_enabled(cols) &&
           getenv("BN_CUDA_ENABLE_Q5K_SPLIT_4WARP") != NULL;
}

int bn_gpu_policy_cuda_q5k_gateup_2warp_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q5K_GATEUP_2WARP") == NULL;
}

int bn_gpu_policy_cuda_q4k_dot_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4K_DOT") == NULL;
}

int bn_gpu_policy_cuda_q5k_dot_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q5K_DOT") == NULL;
}

int bn_gpu_policy_cuda_q4k_4warp_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4K_4WARP") == NULL;
}

int bn_gpu_policy_cuda_q4k_4warp_shape_enabled(int rows, int cols) {
    return cols <= 8192 ||
           (rows == 1536 && cols == 8960 &&
            getenv("BN_CUDA_DISABLE_Q4K_4WARP_1536_8960") == NULL) ||
           (rows == 2560 && cols == 9728 &&
            getenv("BN_CUDA_DISABLE_Q4K_4WARP_2560_9728") == NULL);
}

int bn_gpu_policy_cuda_q4k_out_residual_rmsnorm_fuse_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE") != NULL;
}

int bn_gpu_policy_cuda_q4k_qkv_mixed_fuse_enabled(int tensor_type) {
    return !bn_backend_quant_is_q4k(tensor_type) ||
           getenv("BN_CUDA_ENABLE_Q4K_QKV_MIXED_FUSE") != NULL;
}

int bn_gpu_policy_cuda_q4k_split_k_rope_cache_fuse_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE") != NULL &&
           getenv("BN_CUDA_DISABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE") == NULL;
}

int bn_gpu_policy_cuda_q4k_split_qk_rope_cache_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4K_SPLIT_QK_ROPE_CACHE_FUSE") == NULL;
}

int bn_gpu_policy_cuda_q4k_split_4warp_enabled(int cols) {
    return cols == 2048 &&
           getenv("BN_CUDA_DISABLE_Q4K_SPLIT_4WARP_2048") == NULL;
}

int bn_gpu_policy_cuda_q4k_split_5warp_enabled(int cols) {
    return cols == 2560 &&
           getenv("BN_CUDA_DISABLE_Q4K_SPLIT_5WARP_2560") == NULL;
}

int bn_gpu_policy_cuda_q4k_split_value_rows(int total_rows, int cols) {
    if (total_rows == 4608 && cols == 2048)
        return 512;
    if (total_rows == 2304 && cols == 2048)
        return 256;
    if (total_rows == 1792 && cols == 1536 &&
        getenv("BN_CUDA_ENABLE_Q4K_SPLIT_VALUE_FUSE_1792") != NULL)
        return 256;
    return 0;
}

int bn_gpu_policy_cuda_q4k_split_value_fuse_enabled(int value_rows) {
    return value_rows > 0 &&
           getenv("BN_CUDA_DISABLE_Q4K_SPLIT_VALUE_FUSE") == NULL;
}

int bn_gpu_policy_kquant_gateup_prepared_path_enabled(int q8k_flag) {
    return q8k_flag || getenv("BN_CUDA_DISABLE_Q4K_GATEUP_Q8_1_FAST") != NULL;
}

int bn_gpu_policy_cuda_q4k_gateup_qwarp4_enabled(int cols) {
    return cols <= 4096 &&
           getenv("BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4") == NULL;
}

int bn_gpu_policy_cuda_q4k_gateup_5warp_enabled(int enable_q4k_4warp,
                                                int cols) {
    return enable_q4k_4warp && cols == 2560 &&
           getenv("BN_CUDA_DISABLE_Q4K_GATEUP_5WARP_2560") == NULL;
}

int bn_gpu_policy_cuda_q4k_gateup_2warp_enabled(int enable_q4k_4warp,
                                                int cols) {
    return enable_q4k_4warp && cols <= 5120 &&
           getenv("BN_CUDA_DISABLE_Q4K_GATEUP_2WARP") == NULL;
}

int bn_gpu_policy_cuda_q4k_gateup_4warp_enabled(int enable_q4k_4warp,
                                                int cols) {
    return enable_q4k_4warp && cols <= 8192;
}

int bn_gpu_policy_cuda_q8_warp_disabled(void) {
    return getenv("BN_CUDA_DISABLE_Q8_WARP") != NULL;
}

int bn_gpu_policy_cuda_q8_0_ssm_matvec_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q8_0_SSM_MATVEC") == NULL;
}

int bn_gpu_policy_cuda_q8_0_ssm_prepared_input_enabled(void) {
    return !cuda_q8_0_ssm_prepared_input_disabled();
}

int bn_gpu_policy_cuda_q8_mixed_prepared_input_enabled(int type_a,
                                                       int type_b,
                                                       int cols) {
    return cuda_q8_mixed_prepared_input_requested() &&
           (bn_backend_quant_is_q8_0(type_a) ||
            bn_backend_quant_is_q8_0(type_b)) &&
           (cols & 31) == 0;
}

int bn_gpu_policy_cuda_f16_q8_0_ssm_matvec_enabled(void) {
    return getenv("BN_CUDA_ENABLE_F16_Q8_0_SSM_MATVEC") != NULL &&
           getenv("BN_CUDA_DISABLE_F16_Q8_0_SSM_MATVEC") == NULL &&
           getenv("BN_CUDA_DISABLE_F16_Q8_0_MATVEC") == NULL;
}

int bn_gpu_policy_cuda_f16_q8_0_matvec_enabled(void) {
    return getenv("BN_CUDA_ENABLE_F16_Q8_0_MATVEC") != NULL &&
           getenv("BN_CUDA_DISABLE_F16_Q8_0_MATVEC") == NULL;
}

int bn_gpu_policy_cuda_f16_q5k_matvec_enabled(void) {
    return getenv("BN_CUDA_ENABLE_F16_Q5K_MATVEC") != NULL;
}

int bn_gpu_policy_cuda_q4k_pair_matvec_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4K_PAIR_MATVEC") == NULL;
}

int bn_gpu_policy_kquant_dot_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL;
}

int bn_gpu_policy_kquant_dot_forced(void) {
    return getenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT") != NULL;
}

int bn_gpu_policy_kquant_matvec4_enabled(int cols) {
    return cols >= 16384 &&
           getenv("BN_CUDA_DISABLE_Q4K_Q8K_MATVEC4") == NULL;
}

int bn_gpu_policy_cuda_q4k_matmul8_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q4K_MATMUL8") != NULL;
}

int bn_gpu_policy_cuda_q4k_sharedx_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH") == NULL;
}

int bn_gpu_policy_cuda_q4k_batch_sharedx_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH") != NULL;
}

int bn_gpu_policy_cuda_q6k_dot_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_DOT") == NULL;
}

int bn_gpu_policy_cuda_q6k_dot_forced(void) {
    return getenv("BN_CUDA_ENABLE_Q6K_DOT") != NULL;
}

int bn_gpu_policy_cuda_q6k_warp_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q6K_WARP") != NULL;
}

int bn_gpu_policy_cuda_q6k_q4k_pair_matvec_enabled(int cols) {
    return (cols < 5120 ||
            getenv("BN_CUDA_ENABLE_Q6K_Q4K_PAIR_MATVEC") != NULL) &&
           getenv("BN_CUDA_DISABLE_Q6K_Q4K_PAIR_MATVEC") == NULL;
}

int bn_gpu_policy_cuda_q6k_q8_1_dot_enabled(int is_logits_op) {
    return getenv("BN_CUDA_ENABLE_Q6K_Q8_1_DOT") != NULL &&
           getenv("BN_CUDA_DISABLE_Q6K_Q8_1_DOT") == NULL &&
           (is_logits_op || getenv("BN_CUDA_ENABLE_Q6K_Q8_1_ALL") != NULL);
}

int bn_gpu_policy_cuda_q6k_mmvq_enabled(int rows,
                                        int cols,
                                        int is_logits_op,
                                        int exact_q6k) {
    return !exact_q6k &&
           getenv("BN_CUDA_DISABLE_Q6K_MMVQ") == NULL &&
           ((cols >= 4096 && rows >= 5120) ||
            (cols >= 2048 && rows >= 50000) ||
            (rows == 512 && cols == 2048 &&
             getenv("BN_CUDA_DISABLE_Q6K_MMVQ_512_2048") == NULL) ||
            (rows == 1536 && cols == 8960 &&
             getenv("BN_CUDA_DISABLE_Q6K_MMVQ_1536_8960") == NULL) ||
            (rows == 2560 && cols == 9728 &&
             getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2560_9728") == NULL) ||
            (is_logits_op && cols == 1536 && rows >= 50000 &&
             getenv("BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536") == NULL));
}

int bn_gpu_policy_cuda_q6k_mmvq_2warp_logits_enabled(int rows,
                                                     int cols,
                                                     int is_logits_op) {
    return is_logits_op && cols <= 2560 && rows >= 50000 &&
           ((cols == 1536 &&
             getenv("BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536") == NULL) ||
            (cols > 1536 &&
             getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_LOGITS_SMALL") ==
                 NULL)) &&
           getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_1536") == NULL;
}

int bn_gpu_policy_cuda_q6k_down_residual_rmsnorm_fuse_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q6K_DOWN_RESID_RMSNORM_FUSE") != NULL;
}

int bn_gpu_policy_cuda_f16_q6k_matvec_enabled(int rows,
                                              int cols,
                                              int exact_q6k) {
    return !exact_q6k &&
           (getenv("BN_CUDA_ENABLE_F16_Q6K_MATVEC") != NULL ||
            (getenv("BN_CUDA_DISABLE_F16_Q6K_MATVEC") == NULL &&
             rows <= 2048 && cols >= 8192));
}

int bn_gpu_policy_cuda_q6k_matmul8_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q6K_MATMUL8") != NULL;
}

int bn_gpu_policy_cuda_q6k_matmul4_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_MATMUL4") == NULL;
}

int bn_gpu_policy_cuda_q6k_matvec4_enabled(void) {
    return getenv("BN_CUDA_DISABLE_Q6K_MATVEC4") == NULL;
}

int bn_gpu_policy_cuda_q6k_batch_warp_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q6K_BATCH_WARP") != NULL;
}

int bn_gpu_policy_cuda_fuse_bias_enabled(void) {
    return getenv("BN_CUDA_DISABLE_FUSE_BIAS") == NULL;
}

int bn_gpu_policy_cuda_rope_flash_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_ROPE_FLASH_FUSE") == NULL;
}

int bn_gpu_policy_cuda_bias_rope_flash_fuse_enabled(void) {
    return getenv("BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE") != NULL;
}

int bn_gpu_policy_cuda_qk_norm_rope_flash_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FLASH_FUSE") == NULL;
}

int bn_gpu_policy_cuda_qk_norm_rope_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FUSE") == NULL;
}

int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_rmsnorm_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_RMSNORM_FUSE") == NULL;
}

int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_FUSE") == NULL;
}

int bn_gpu_policy_cuda_readback_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_READBACK") != NULL;
}

int bn_gpu_policy_cuda_cublas_cache_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_CUBLAS_CACHE") != NULL;
}

int bn_gpu_policy_cuda_nan_verbose_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_NAN_VERBOSE") != NULL;
}

int bn_gpu_policy_cuda_stream_exec_enabled(void) {
    return getenv("BN_CUDA_DISABLE_STREAM_EXEC") == NULL;
}

int bn_gpu_policy_cuda_profile_enabled(void) {
    return getenv("BN_CUDA_PROFILE") != NULL;
}

int bn_gpu_policy_cuda_wall_profile_enabled(void) {
    return getenv("BN_CUDA_PROFILE_WALL") != NULL;
}

int bn_gpu_policy_cuda_profile_shapes_enabled(void) {
    return getenv("BN_CUDA_PROFILE_SHAPES") != NULL;
}

const char *bn_gpu_policy_cuda_device_selector(void) {
    return getenv("BN_CUDA_DEVICE");
}

int bn_gpu_policy_cuda_exec_fail_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_EXEC_FAIL") != NULL;
}

int bn_gpu_policy_cuda_sync_each_op_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_SYNC_EACH_OP") != NULL;
}

int bn_gpu_policy_cuda_nan_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_NAN") != NULL;
}

int bn_gpu_policy_cuda_dump_ops_enabled(void) {
    return getenv("BN_CUDA_DUMP_OPS") != NULL;
}

int bn_gpu_policy_cuda_dump_ops_every_enabled(void) {
    return getenv("BN_CUDA_DUMP_OPS_EVERY") != NULL;
}

int bn_gpu_policy_cuda_prefill_moe_layer_disabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_MOE_LAYER") != NULL;
}

int bn_gpu_policy_cuda_prefill_dense_layer_disabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_DENSE_LAYER") != NULL;
}

int bn_gpu_policy_cuda_prefill_dense_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_DENSE_LAYER") != NULL;
}

int bn_gpu_policy_cuda_prefill_dense_profile_enabled(void) {
    return getenv("BN_CUDA_PREFILL_DENSE_PROFILE") != NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_layer_disabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER") != NULL;
}

int bn_gpu_policy_prefill_ssm_layer_disabled(void) {
    return bn_gpu_policy_cuda_prefill_ssm_layer_disabled();
}

int bn_gpu_policy_cuda_prefill_fused_q4k_gateup_batch_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_fused_q4k_gateup_batch_enabled(void) {
    return getenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH") !=
               NULL &&
           getenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH") ==
               NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_profile_enabled(void) {
    return getenv("BN_CUDA_SSM_PROFILE") != NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_stacked_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_STACKED_PREFILL") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_stream_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_STREAM_PREFILL") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_input_alias_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_PREFILL_INPUT_ALIAS") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_f32_ab_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_F32_AB_PREFILL") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_scan_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_PREFILL_SCAN") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_delta_128_warp_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_DELTA_128_WARP") == NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_ffn_profile_enabled(void) {
    return getenv("BN_CUDA_SSM_FFN_PROFILE") != NULL;
}

int bn_gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_enabled(void) {
    return getenv("BN_CUDA_ENABLE_SSM_FFN_GATEUP_F16_OUT") != NULL &&
           getenv("BN_CUDA_DISABLE_SSM_FFN_GATEUP_F16_OUT") == NULL;
}

int bn_gpu_policy_backend_opt_in_fused_gateup_enabled(void) {
    return getenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP") != NULL;
}

int bn_gpu_policy_fused_gateup_silu_allowed(const BnGPUBackend *gpu,
                                            int tensor_type) {
    if (!bn_gpu_policy_fused_gateup_enabled())
        return 0;
    if (gpu_policy_backend_caps(gpu)->fused_gateup_requires_backend_opt_in &&
        bn_backend_quant_gpu_fused_gateup_requires_backend_opt_in(tensor_type) &&
        !bn_gpu_policy_backend_opt_in_fused_gateup_enabled())
        return 0;
    return 1;
}

int bn_gpu_policy_shared_kquant_dot_enabled(void) {
    return getenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT") == NULL;
}

int bn_gpu_policy_shared_expert_gate_enabled(void) {
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

int bn_gpu_policy_prefill_attention_min_tokens_configured(void) {
    return bn_gpu_policy_cuda_prefill_attention_min_tokens_configured();
}

int bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
    int default_tokens) {
    return env_positive_int_or_default("BN_CUDA_PREFILL_ATTN_MIN_TOKENS",
                                       default_tokens);
}

int bn_gpu_policy_prefill_attention_min_tokens_or_default(
    int default_tokens) {
    return bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
        default_tokens);
}

int bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
    int default_tokens) {
    return env_positive_int_or_default(
        "BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", default_tokens);
}

int bn_gpu_policy_cuda_prefill_gemm_attention_enabled(int n_tokens,
                                                      int max_tokens) {
    if (getenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN") != NULL)
        return 0;
    if (max_tokens > 0 && n_tokens > max_tokens)
        return 0;
    return getenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN") != NULL ||
           n_tokens >=
               bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
                   256);
}

int bn_gpu_policy_cuda_prefill_attention_wo_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_ATTN_WO") == NULL;
}

int bn_gpu_policy_cuda_prefill_qkv_attention_wo_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO") == NULL;
}

int bn_gpu_policy_cuda_prefill_batched_gemm_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM") == NULL;
}

int bn_gpu_policy_cuda_prefill_gemm_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_GEMM") != NULL;
}

int bn_gpu_policy_prefill_dense_chain_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN") == NULL;
}

int bn_gpu_policy_prefill_hybrid_chain_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN") == NULL;
}

int bn_gpu_policy_cuda_prefill_attention_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_ATTN") == NULL;
}

int bn_gpu_policy_prefill_attention_enabled(void) {
    return bn_gpu_policy_cuda_prefill_attention_enabled();
}

int bn_gpu_policy_prefill_ssm_run_chain_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN") == NULL;
}

int bn_gpu_policy_prefill_ssm_ffn_fuse_allowed(void) {
    return getenv("BN_CUDA_DISABLE_SSM_FFN_FUSE") == NULL;
}

int bn_gpu_policy_prefill_moe_chain_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN") != NULL;
}

int bn_gpu_policy_prefill_hybrid_chain_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN") != NULL;
}

int bn_gpu_policy_moe_prefill_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_PREFILL") != NULL;
}

int bn_gpu_policy_moe_prefill_min_tokens_configured(void) {
    const char *env = getenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    return env && *env;
}

int bn_gpu_policy_moe_prefill_min_tokens_or_default(int default_tokens) {
    return env_positive_int_or_default("BN_CUDA_MOE_PREFILL_MIN_TOKENS",
                                       default_tokens);
}

int bn_gpu_policy_cuda_dense_ffn_enabled(void) {
    return getenv("BN_CUDA_ENABLE_DENSE_FFN") != NULL;
}

int bn_gpu_policy_cuda_dense_ffn_batch_enabled(void) {
    return getenv("BN_CUDA_DISABLE_DENSE_FFN_BATCH") == NULL;
}

int bn_gpu_policy_cuda_moe_cublas_gateup_f16_out_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP_F16_OUT") == NULL;
}

int bn_gpu_policy_cuda_moe_cublas_grouped_variable_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_VARIABLE") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED_VARIABLE") == NULL;
}

int bn_gpu_policy_cuda_moe_cublas_grouped_enabled(int routed_q8,
                                                  int routed_q4,
                                                  int gate_f16,
                                                  int up_f16,
                                                  int down_f16,
                                                  int n_experts,
                                                  int k,
                                                  int route_items) {
    int enabled = gate_f16 && up_f16 && down_f16 &&
        ((routed_q8 &&
          getenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GROUPED") == NULL) ||
         (routed_q4 &&
          getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED") == NULL));
    if (enabled && routed_q4 && !(n_experts == 2 && k == 2) &&
        route_items <= 256 &&
        getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_SMALL") == NULL)
        enabled = 0;
    return enabled;
}

int bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(int use_grouped,
                                                      int routed_q8,
                                                      int routed_q4,
                                                      int gate_f16,
                                                      int up_f16,
                                                      int down_f16,
                                                      int n_tokens) {
    return !use_grouped && gate_f16 && up_f16 && !down_f16 && n_tokens > 1 &&
        ((routed_q8 &&
          getenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GATEUP") == NULL) ||
         (routed_q4 &&
          getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GATEUP") != NULL &&
          getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP") == NULL));
}

int bn_gpu_policy_cuda_moe_cublas_all_active_two_fixed_enabled(
    int use_grouped,
    int n_experts,
    int k) {
    return use_grouped && n_experts == 2 && k == 2 &&
           getenv("BN_CUDA_DISABLE_MOE_CUBLAS_ALL2_FIXED") == NULL;
}

int bn_gpu_policy_cuda_moe_sorted_slots_enabled(int routed_q4,
                                                int routed_q8,
                                                int n_tokens,
                                                int use_all_active_two_fixed,
                                                int use_grouped,
                                                int use_gateup_only) {
    return (routed_q4 || routed_q8) && n_tokens > 1 &&
           !use_all_active_two_fixed &&
           (use_grouped || use_gateup_only ||
            getenv("BN_CUDA_ENABLE_MOE_ROUTE_SORT") != NULL);
}

int bn_gpu_policy_cuda_moe_prefill_internal_profile_enabled(void) {
    return getenv("BN_CUDA_PROFILE_MOE_PREFILL_INTERNAL") != NULL;
}

int bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
    int add_norm_resid,
    int out_provided,
    int has_shared,
    int init_out_with_residual) {
    return add_norm_resid && !out_provided && !has_shared &&
           init_out_with_residual &&
           getenv("BN_CUDA_DISABLE_MOE_PREFILL_DIRECT_RESID_OUT") == NULL;
}

int bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(int n_experts) {
    return n_experts <= 256 &&
           getenv("BN_CUDA_ENABLE_MOE_BATCH_FUSED_ROUTE_TOPK") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_BATCH_FUSED_ROUTE_TOPK") == NULL;
}

int bn_gpu_policy_cuda_moe_route_dist_profile_enabled(void) {
    return getenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST") != NULL;
}

int bn_gpu_policy_cuda_moe_route_dist_profile_every_or_default(
    int default_every) {
    return env_positive_int_or_default("BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY",
                                       default_every);
}

int bn_gpu_policy_cuda_moe_cublas_grouped_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED") != NULL;
}

int bn_gpu_policy_cuda_moe_cublas_gateup_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_CUBLAS_GATEUP") != NULL;
}

int bn_gpu_policy_cuda_moe_ffn_batch_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_FFN_BATCH") == NULL;
}

int bn_gpu_policy_cuda_moe_ffn_batch_profile_enabled(void) {
    return getenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_INTERNAL") != NULL;
}

int bn_gpu_policy_moe_cache_prefill_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL") == NULL;
}

int bn_gpu_policy_moe_prefill_shared_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE") == NULL;
}

int bn_gpu_policy_moe_route_batch_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_ROUTE_BATCH") == NULL;
}

int bn_gpu_policy_moe_route_batch_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH") != NULL;
}

int bn_gpu_policy_large_hybrid_attention_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN") != NULL;
}

int bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL;
}

int bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled(void) {
    return getenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL;
}

int bn_gpu_policy_large_hybrid_cpu_attention_safe_forced(void) {
    return getenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL;
}

int bn_gpu_policy_large_hybrid_prefill_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL") != NULL;
}

int bn_gpu_policy_large_hybrid_prefill_chain_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN") != NULL;
}

int bn_gpu_policy_large_hybrid_prefill_disabled(void) {
    return getenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL") != NULL;
}

int bn_gpu_policy_large_hybrid_argmax_enabled(void) {
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX") != NULL;
}

int bn_gpu_policy_prefill_matmul_disabled(void) {
    return getenv("BN_GPU_DISABLE_PREFILL_MATMUL") != NULL;
}

int bn_gpu_policy_prefill_matmul_enabled(void) {
    return getenv("BN_GPU_PREFILL_MATMUL") != NULL;
}

int bn_gpu_policy_prefill_direct_kv_disabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV") != NULL;
}

int bn_gpu_policy_prefill_direct_kv_with_cpu_fallback_enabled(void) {
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

int bn_gpu_policy_ssm_graph_disabled(void) {
    return getenv("BN_CUDA_DISABLE_SSM_GRAPH") != NULL;
}

int bn_gpu_policy_cuda_qkv_mixed_fuse_disabled(void) {
    return getenv("BN_CUDA_DISABLE_QKV_MIXED_FUSE") != NULL;
}

int bn_gpu_policy_cuda_qkv_key_cache_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_QKV_KCACHE_FUSE") == NULL;
}

int bn_gpu_policy_cuda_qkv_kpair_opt_enabled(void) {
    return getenv("BN_CUDA_ENABLE_QKV_KPAIR_OPT") != NULL;
}

int bn_gpu_policy_cuda_q5_gateup_warp_disabled(void) {
    return getenv("BN_CUDA_DISABLE_Q5_GATEUP_WARP") != NULL;
}

int bn_gpu_policy_cuda_q8_gateup_warp_disabled(void) {
    return getenv("BN_CUDA_DISABLE_Q8_GATEUP_WARP") != NULL;
}

int bn_gpu_policy_cuda_graph_exec_requested(void) {
    return getenv("BN_CUDA_ENABLE_GRAPH_EXEC") != NULL ||
           getenv("BN_CUDA_ENABLE_UNSAFE_MOE_FFN") != NULL;
}

int bn_gpu_policy_cuda_moe_graph_max_experts_or_default(
    int default_experts) {
    return env_positive_int_or_default("BN_CUDA_MOE_GRAPH_MAX_EXPERTS",
                                       default_experts);
}

int bn_gpu_policy_cuda_decode_graph_default_enabled(int moe_graph,
                                                    int default_moe_graph) {
    return getenv("BN_CUDA_DISABLE_GRAPH_EXEC") == NULL &&
           getenv("BN_CUDA_ENABLE_MOE_FFN") == NULL &&
           (!moe_graph || default_moe_graph);
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
    if (!gpu_policy_backend_caps(gpu)->moe_down_cublas_cache ||
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

size_t bn_gpu_policy_moe_down_aux_cache_bytes(const BnGPUBackend *gpu,
                                              int tensor_type,
                                              int rows,
                                              int cols) {
    return bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(gpu, tensor_type,
                                                          rows, cols);
}

static size_t gpu_policy_aux_cache_bytes(int tensor_type,
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

size_t bn_gpu_policy_aux_cache_bytes(int tensor_type,
                                     int rows,
                                     int cols) {
    return gpu_policy_aux_cache_bytes(tensor_type, rows, cols);
}

int bn_gpu_policy_cuda_cublas_aux_cache_supported(int tensor_type,
                                                  int cols) {
    return cols > 0 &&
           (cols & 31) == 0 &&
           bn_gpu_policy_cuda_cublas_matmul_enabled() &&
           bn_backend_quant_eager_aux_cache_supported(tensor_type);
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
                                          int cap_seq_len) {
    return bn_gpu_policy_auto_caps_sequence(
        webgpu, cuda, metal, bn_model_arch_gguf_uses_moe(gf),
        bn_model_arch_gguf_u32(gf, "context_length"), cap_seq_len);
}

int bn_gpu_policy_duplicate_moe_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE") != NULL &&
           getenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE") == NULL;
}

int bn_gpu_policy_webgpu_repacked_buffer_supported(int tensor_type) {
    return bn_backend_quant_can_gpu_repack(tensor_type);
}

int bn_gpu_policy_webgpu_repacked_bias_supported(int tensor_type) {
    return bn_backend_quant_gpu_supports_repacked_bias(tensor_type);
}

int bn_gpu_policy_metal_mmap_zero_copy_enabled(void) {
    return getenv("BN_METAL_ENABLE_MMAP_ZERO_COPY") != NULL;
}

void bn_gpu_policy_apply_metal_barrier_disable_override(void) {
    setenv("BN_METAL_DISABLE_BARRIERS", "1", 1);
}

void bn_gpu_policy_apply_specialized_q6_q8k_override(void) {
    setenv("BN_METAL_ENABLE_Q6_Q8K", "1", 1);
}

void bn_gpu_policy_apply_q4_q8_prepared_override(void) {
    setenv("BN_METAL_Q4_PREPARED", "1", 1);
}

void bn_gpu_policy_apply_metal_q4_q8_default_disable_override(void) {
    setenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT", "1", 1);
}

void bn_gpu_policy_apply_metal_private_weights_override(void) {
    setenv("BN_METAL_PRIVATE_WEIGHTS", "1", 1);
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

int bn_gpu_policy_small_dense_native_quant_prepared_layer_default_enabled(void) {
    return bn_gpu_policy_metal_q4_prepared_enabled();
}

int bn_gpu_policy_metal_q4_prepared_upload_enabled(void) {
    const char *from_layer = getenv("BN_GPU_Q4_Q8_FROM_LAYER");
    return bn_gpu_policy_metal_q4_prepared_enabled() &&
           getenv("BN_GPU_Q4_Q8") &&
           (!from_layer || atoi(from_layer) <= 0) &&
           !getenv("BN_GPU_Q4_Q8_ATTN_ONLY") &&
           !getenv("BN_GPU_Q4_Q8_FFN_ONLY");
}

int bn_gpu_policy_metal_repacked_buffer_supported(int tensor_type) {
    return bn_backend_quant_can_gpu_repack(tensor_type);
}

int bn_gpu_policy_metal_repacked_buffer_type(int tensor_type) {
    return bn_gpu_policy_metal_repacked_buffer_supported(tensor_type)
        ? tensor_type
        : -1;
}

int bn_gpu_policy_metal_prepared_stacked_upload_blocked(int tensor_type) {
    return bn_gpu_policy_metal_repacked_buffer_supported(tensor_type) &&
           bn_gpu_policy_metal_q4_prepared_upload_enabled();
}

int bn_gpu_policy_metal_shared_weights_enabled(void) {
    return getenv("BN_METAL_SHARED_WEIGHTS") != NULL;
}

int bn_gpu_policy_metal_q6_q8k_enabled(void) {
    return getenv("BN_METAL_ENABLE_Q6_Q8K") != NULL;
}

int bn_gpu_policy_specialized_native_quant_decode_path_enabled(void) {
    return bn_gpu_policy_metal_q6_q8k_enabled();
}

int bn_gpu_policy_metal_q8_barriers_enabled(void) {
    return getenv("BN_METAL_Q8_BARRIERS") != NULL;
}

int bn_gpu_policy_metal_q4_q8_matvec_supported(
    int tensor_type,
    int q4_q8_enabled,
    int q4_prepared,
    int has_q8_quant_pipeline,
    int has_q4_q8_pipeline,
    int has_q4_prepared_q8_pipeline) {
    if (!q4_q8_enabled ||
        !bn_backend_quant_metal_q4_q8_matvec_supported(tensor_type) ||
        !has_q8_quant_pipeline)
        return 0;
    return q4_prepared
        ? has_q4_prepared_q8_pipeline
        : has_q4_q8_pipeline;
}

int bn_gpu_policy_metal_q4_q8_graph_path_supported(
    int tensor_type,
    int q4_q8_enabled,
    int q4_prepared,
    int prepared_path,
    int has_q8_quant_pipeline,
    int has_pipeline) {
    return q4_prepared == prepared_path &&
           q4_q8_enabled &&
           bn_backend_quant_metal_q4_q8_matvec_supported(tensor_type) &&
           has_q8_quant_pipeline &&
           has_pipeline;
}

int bn_gpu_policy_metal_q6_q8k_matvec_supported(
    int tensor_type,
    int cols,
    int has_q8k_quant_pipeline,
    int has_q6_q8k_pipeline) {
    return bn_gpu_policy_metal_q6_q8k_enabled() &&
           bn_backend_quant_metal_q6_q8k_matvec_supported(tensor_type) &&
           has_q8k_quant_pipeline &&
           has_q6_q8k_pipeline &&
           cols > 0 &&
           (cols % 256) == 0;
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

int bn_gpu_policy_small_dense_exact_native_fused_gateup_enabled(void) {
    return getenv("BN_GPU_Q4_Q8_DISABLE_GATEUP") == NULL;
}

int bn_gpu_policy_small_dense_exact_native_attn_only_enabled(void) {
    return getenv("BN_GPU_Q4_Q8_ATTN_ONLY") != NULL;
}

int bn_gpu_policy_small_dense_exact_native_ffn_only_enabled(void) {
    return getenv("BN_GPU_Q4_Q8_FFN_ONLY") != NULL;
}

int bn_gpu_policy_small_dense_exact_native_from_layer_or_default(int n_layers) {
    const char *env = getenv("BN_GPU_Q4_Q8_FROM_LAYER");
    if (env)
        return atoi(env);
    return bn_gpu_policy_metal_q4_q8_enabled() ? n_layers - 1 : -1;
}

int bn_gpu_policy_small_dense_exact_native_to_layer_or_default(
    int n_layers,
    int native_quant_prepared) {
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

    if (bn_gpu_policy_metal_q4_q8_enabled() && !native_quant_prepared &&
        n_layers > 33)
        return n_layers - 33 - 1;
    return -1;
}

int bn_gpu_policy_gateup_split_enabled(void) {
    return getenv("BN_GPU_DISABLE_GATEUP_SPLIT") == NULL;
}

int bn_gpu_policy_small_dense_exact_native_ffn_down_enabled(void) {
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

int bn_gpu_policy_backend_flash_max_kv_or_default(const BnGPUBackend *gpu,
                                                  int default_max_kv) {
    const BnGPUPolicyBackendCaps *caps = gpu_policy_backend_caps(gpu);
    const char *env = getenv("BN_GPU_FLASH_MAX_KV");
    if (env)
        return atoi(env);
    return caps->default_flash_max_kv ? caps->default_flash_max_kv
                                      : default_max_kv;
}

int bn_gpu_policy_backend_flash_default_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->flash_default;
}

int bn_gpu_policy_backend_large_graph_native_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->large_graph_native;
}

int bn_gpu_policy_backend_small_dense_native_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->small_dense_native;
}

int bn_gpu_policy_backend_all_active_two_kquant_moe_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->all_active_two_kquant_moe;
}

int bn_gpu_policy_backend_cpu_attention_fallback_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->cpu_attention_fallback;
}

int bn_gpu_policy_backend_small_dense_exact_native_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->small_dense_exact_native;
}

int bn_gpu_policy_backend_prefill_decode_fallback_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->prefill_decode_fallback;
}

int bn_gpu_policy_backend_prefill_chain_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->prefill_chain;
}

int bn_gpu_policy_backend_matvec_fallback_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->matvec_fallback;
}

int bn_gpu_policy_backend_dense_batch_prefill_shape_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->dense_batch_prefill_shape;
}

int bn_gpu_policy_backend_lazy_moe_aux_cache_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->lazy_moe_aux_cache;
}

int bn_gpu_policy_backend_native_quant_logits_refine_default_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->native_quant_logits_refine_default;
}

int bn_gpu_policy_backend_all_active_two_kquant_moe_logits_refine_default_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(
        gpu)->all_active_two_kquant_moe_logits_refine_default;
}

int bn_gpu_policy_backend_decode_graph_cache_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->decode_graph_cache;
}

int bn_gpu_policy_backend_moe_exact_attention_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->moe_exact_attention;
}

int bn_gpu_policy_backend_ssm_graph_supported(const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->ssm_graph;
}

int bn_gpu_policy_backend_large_hybrid_argmax_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->large_hybrid_argmax;
}

int bn_gpu_policy_backend_all_active_two_moe_direct_route_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->all_active_two_moe_direct_route;
}

int bn_gpu_policy_backend_resident_moe_ffn_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->resident_moe_ffn;
}

int bn_gpu_policy_backend_moe_gateup_split_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->moe_gateup_split;
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

int bn_gpu_policy_backend_kquant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int kquant_refine_default) {
    const BnGPUPolicyBackendCaps *caps = gpu_policy_backend_caps(gpu);
    return kquant_refine_default ||
           getenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE") != NULL ||
           (!caps->suppress_implicit_kquant_logits_refine &&
            getenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE") == NULL);
}

int bn_gpu_policy_kquant_logits_refine_top_or_default(int default_top) {
    const char *env = getenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    return env ? atoi(env) : default_top;
}

int bn_gpu_policy_backend_native_quant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int native_quant_refine_default) {
    const BnGPUPolicyBackendCaps *caps = gpu_policy_backend_caps(gpu);
    return getenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE") != NULL ||
           native_quant_refine_default ||
           (!caps->suppress_implicit_native_quant_logits_refine &&
            getenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE") == NULL);
}

int bn_gpu_policy_native_quant_logits_refine_top_or_default(int default_top) {
    const char *env = getenv("BN_GPU_Q8_REFINE_TOP");
    return env ? atoi(env) : default_top;
}

int bn_gpu_policy_moe_ffn_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_FFN") != NULL;
}

int bn_gpu_policy_moe_router_topk_enabled(int eligible) {
    return eligible && getenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK") == NULL;
}

int bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT") == NULL;
}

int bn_gpu_policy_moe_router_gpu_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU") == NULL;
}

int bn_gpu_policy_moe_router_diff2_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2") == NULL;
}

int bn_gpu_policy_moe_routed_ffn_batch_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH") == NULL;
}

int bn_gpu_policy_moe_routed_ffn_batch_allowed(int large_moe) {
    if (getenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH"))
        return 0;
    return !large_moe ||
           getenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE") != NULL;
}

int bn_gpu_policy_moe_cpu_actual_override_enabled(void) {
    return getenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL") != NULL;
}

int bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled(void) {
    return small_dense_native_quant_cpu_attention_safe_disabled();
}

int bn_gpu_policy_small_dense_exact_native_disabled(void) {
    return small_dense_exact_native_disabled();
}

int bn_gpu_policy_small_dense_exact_ffn_down_enabled(void) {
    return small_dense_exact_ffn_down_requested();
}

int bn_gpu_policy_small_dense_prefill_disabled(void) {
    return small_dense_prefill_disabled();
}

int bn_gpu_policy_native_quant_logits_refine_requested(void) {
    return native_quant_logits_refine_requested();
}

int bn_gpu_policy_native_quant_logits_refine_disabled(void) {
    return native_quant_logits_refine_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(void) {
    return all_active_two_kquant_moe_fast_ffn_requested();
}

int bn_gpu_policy_all_active_two_kquant_moe_fast_graph_disabled(void) {
    return all_active_two_kquant_moe_fast_graph_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_cublas_decode_enabled(void) {
    return all_active_two_kquant_moe_cublas_decode_requested();
}

int bn_gpu_policy_cuda_moe_cublas_decode_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE") == NULL;
}

int bn_gpu_policy_cuda_moe_cublas_decode_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE") != NULL;
}

int bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled(void) {
    return all_active_two_kquant_moe_fast_route_requested();
}

int bn_gpu_policy_all_active_two_kquant_moe_q8k_default_disabled(void) {
    return all_active_two_kquant_moe_q8k_default_disabled();
}

int bn_gpu_policy_all_active_two_kquant_route_q8k_default_disabled(void) {
    return all_active_two_kquant_route_q8k_default_disabled();
}

int bn_gpu_policy_all_active_two_kquant_route_q8_1_prepared_input_enabled(void) {
    return all_active_two_route_q8_1_prepared_input_requested();
}

int bn_gpu_policy_all_active_two_kquant_fast_q8k_gateup_enabled(void) {
    return all_active_two_kquant_fast_q8k_gateup_requested();
}

int bn_gpu_policy_all_active_two_kquant_fast_q8k_gateup_disabled(void) {
    return all_active_two_kquant_fast_q8k_gateup_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_pair_down_enabled(void) {
    return all_active_two_kquant_q6k_pair_down_requested();
}

int bn_gpu_policy_all_active_two_kquant_q6k_pair_down_f32_layers_disabled(void) {
    return all_active_two_kquant_q6k_pair_down_f32_layers_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_pair_down_f32_layer_selected(int layer) {
    return all_active_two_kquant_q6k_pair_down_f32_layer_selected(layer);
}

int bn_gpu_policy_all_active_two_kquant_q6k_ordered_down_enabled(void) {
    return all_active_two_kquant_q6k_ordered_down_requested();
}

int bn_gpu_policy_all_active_two_kquant_q6k_ordered_down_disabled(void) {
    return all_active_two_kquant_q6k_ordered_down_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_f32_down_default_enabled(void) {
    return all_active_two_kquant_q6k_f32_down_default_requested();
}

int bn_gpu_policy_all_active_two_kquant_q6k_f32_down_default_disabled(void) {
    return all_active_two_kquant_q6k_f32_down_default_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_f32_all_active_down_disabled(void) {
    return all_active_two_kquant_q6k_f32_all_active_down_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_f32_cache_enabled(void) {
    return all_active_two_kquant_q6k_f32_cache_requested();
}

int bn_gpu_policy_all_active_two_kquant_q6k_float_4row_down_default_disabled(void) {
    return all_active_two_kquant_q6k_float_4row_down_default_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_float_4row_down_disabled(void) {
    return all_active_two_kquant_q6k_float_4row_down_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_f32_exact_4row_down_layer_selected(
    int layer) {
    const char *env =
        all_active_two_kquant_q6k_f32_exact_4row_down_layers();
    return !env || !*env ||
           all_active_two_kquant_q6k_f32_exact_4row_down_layer_selected(
               layer);
}

int bn_gpu_policy_all_active_two_kquant_q6k_f32_exact_4row_down_default_disabled(void) {
    return all_active_two_kquant_q6k_f32_exact_4row_down_default_disabled();
}

int bn_gpu_policy_all_active_two_kquant_q6k_f32_exact_4row_down_disabled(void) {
    return all_active_two_kquant_q6k_f32_exact_4row_down_disabled();
}

float bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(float default_eps) {
    return all_active_two_kquant_down_skip_eps_or_default(default_eps);
}

int bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled(void) {
    return all_active_two_kquant_moe_cpu_attention_safe_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled(void) {
    return all_active_two_kquant_moe_logits_refine_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled(void) {
    return all_active_two_kquant_moe_cpu_moe_safe_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_exact_attention_disabled(void) {
    return all_active_two_kquant_moe_exact_attention_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled(void) {
    return all_active_two_kquant_moe_cpu_route_resident_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_requested(void) {
    return all_active_two_kquant_moe_exact_gpu_route_requested();
}

int bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_disabled(void) {
    return all_active_two_kquant_moe_exact_gpu_route_disabled();
}

int bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled(void) {
    return bn_gpu_policy_moe_router_gpu_enabled() ||
           bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_requested();
}

void bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(int *from_layer,
                                                   int *to_layer) {
    const char *env;

    if (from_layer)
        *from_layer = -1;
    if (to_layer)
        *to_layer = -1;

    env = all_active_two_kquant_moe_route_from_layer_value();
    if (env && from_layer)
        *from_layer = atoi(env);
    env = all_active_two_kquant_moe_route_to_layer_value();
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

int bn_gpu_policy_moe_shared_cpu_fallback_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK") == NULL;
}

int bn_gpu_policy_moe_gateup_split_enabled(int can_split) {
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
