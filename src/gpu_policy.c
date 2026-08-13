#include "gpu_policy.h"
#include "backend_quant.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int gpu_runtime_enabled(const BnGPUBackend *gpu, const char *name) {
    return gpu && bn_backend_runtime_policy_enabled(&gpu->runtime_policy,
                                                    name);
}

static const char *gpu_runtime_get(const BnGPUBackend *gpu,
                                   const char *name) {
    return gpu ? bn_backend_runtime_policy_get(&gpu->runtime_policy, name)
               : NULL;
}

static int runtime_int_or_default(const BnBackendRuntimePolicy *policy,
                                  const char *name,
                                  int default_value) {
    const char *value = bn_backend_runtime_policy_get(policy, name);
    if (!value || !*value) return default_value;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    return end && *end == '\0' ? (int)parsed : default_value;
}

static int runtime_positive_int_or_default(
    const BnBackendRuntimePolicy *policy,
    const char *name,
    int default_value) {
    int value = runtime_int_or_default(policy, name, default_value);
    return value > 0 ? value : default_value;
}

static int runtime_compat_enabled(const BnBackendRuntimePolicy *policy,
                                  const char *name,
                                  const char *compat_name) {
    return bn_backend_runtime_policy_enabled(policy, name) ||
           (compat_name &&
            bn_backend_runtime_policy_enabled(policy, compat_name));
}

static int runtime_compat_enabled2(const BnBackendRuntimePolicy *policy,
                                   const char *name,
                                   const char *compat_name,
                                   const char *compat_name2) {
    return runtime_compat_enabled(policy, name, compat_name) ||
           (compat_name2 &&
            bn_backend_runtime_policy_enabled(policy, compat_name2));
}

static int runtime_compat_enabled3(const BnBackendRuntimePolicy *policy,
                                   const char *name,
                                   const char *compat_name,
                                   const char *compat_name2,
                                   const char *compat_name3) {
    return runtime_compat_enabled2(policy, name, compat_name, compat_name2) ||
           (compat_name3 &&
            bn_backend_runtime_policy_enabled(policy, compat_name3));
}

static const char *runtime_compat_value2(
    const BnBackendRuntimePolicy *policy,
    const char *name,
    const char *compat_name,
    const char *compat_name2) {
    const char *value = bn_backend_runtime_policy_get(policy, name);
    if (!value && compat_name)
        value = bn_backend_runtime_policy_get(policy, compat_name);
    if (!value && compat_name2)
        value = bn_backend_runtime_policy_get(policy, compat_name2);
    return value;
}

static const char *runtime_compat_value3(
    const BnBackendRuntimePolicy *policy,
    const char *name,
    const char *compat_name,
    const char *compat_name2,
    const char *compat_name3) {
    const char *value = runtime_compat_value2(policy, name, compat_name,
                                             compat_name2);
    if (!value && compat_name3)
        value = bn_backend_runtime_policy_get(policy, compat_name3);
    return value;
}

static int runtime_layer_selected(const char *env, int layer) {
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

static int runtime_compat_layer_selected2(
    const BnBackendRuntimePolicy *policy,
    const char *name,
    const char *compat_name,
    const char *compat_name2,
    int layer) {
    return runtime_layer_selected(runtime_compat_value2(
        policy, name, compat_name, compat_name2), layer);
}

static int runtime_compat_layer_selected3(
    const BnBackendRuntimePolicy *policy,
    const char *name,
    const char *compat_name,
    const char *compat_name2,
    const char *compat_name3,
    int layer) {
    return runtime_layer_selected(runtime_compat_value3(
        policy, name, compat_name, compat_name2, compat_name3), layer);
}

static float runtime_compat_float_or_default2(
    const BnBackendRuntimePolicy *policy,
    const char *name,
    const char *compat_name,
    const char *compat_name2,
    float default_value) {
    const char *env = runtime_compat_value2(policy, name, compat_name,
                                           compat_name2);
    if (!env || !*env) return default_value;
    return (float)atof(env);
}

static int cuda_native_quant_prepared_input_split_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT",
        "BN_CUDA_ENABLE_Q8_0_PREPARED_INPUT_SPLIT",
        "BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
}

static int cuda_native_quant_prepared_input_split_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT",
        "BN_CUDA_DISABLE_Q8_0_PREPARED_INPUT_SPLIT",
        "BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
}

static int cuda_native_quant_prepared_input_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT",
        "BN_CUDA_ENABLE_Q8_PREPARED_INPUT",
        "BN_CUDA_ENABLE_Q8_PREQ");
}

static int cuda_native_quant_prepared_input_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT",
        "BN_CUDA_DISABLE_Q8_PREPARED_INPUT",
        "BN_CUDA_DISABLE_Q8_PREQ");
}

static int cuda_native_quant_prepared_input_logits_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS",
        "BN_CUDA_ENABLE_Q8_PREPARED_INPUT_LOGITS",
        "BN_CUDA_ENABLE_Q8_PREQ_LOGITS");
}

static int cuda_native_quant_prepared_input_logits_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS",
        "BN_CUDA_DISABLE_Q8_PREPARED_INPUT_LOGITS",
        "BN_CUDA_DISABLE_Q8_PREQ_LOGITS");
}

static int cuda_moe_route_dot_prepared_input_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_MOE_ROUTE_DOT_PREPARED_INPUT",
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREPARED_INPUT",
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT");
}

static int cuda_moe_route_block_prepared_input_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT",
        "BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT",
        "BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT");
}

static int cuda_moe_route_block_prepared_input_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT",
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT",
        "BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT");
}

static int cuda_native_quant_ssm_prepared_input_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_SSM_PREPARED_INPUT",
        "BN_CUDA_DISABLE_Q8_0_SSM_PREPARED_INPUT",
        "BN_CUDA_DISABLE_Q8_0_SSM_PREQ");
}

static int cuda_native_quant_mixed_prepared_input_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_NATIVE_QUANT_MIXED_PREPARED_INPUT",
        "BN_CUDA_ENABLE_Q8_MIXED_PREPARED_INPUT",
        "BN_CUDA_ENABLE_Q8_MIXED_PREQ");
}

static int all_active_two_route_block_prepared_input_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_BLOCK_PREPARED_INPUT",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREPARED_INPUT",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREQUANT") ||
        bn_backend_runtime_policy_enabled(
            policy, "BN_CUDA_ENABLE_QWEN2MOE_ROUTE_Q8_1_PREQUANT");
}

static int small_dense_native_quant_cpu_attention_safe_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE");
}

static int small_dense_native_quant_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT",
        "BN_CUDA_DISABLE_SMALL_DENSE_EXACT_NATIVE",
        "BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8") ||
        bn_backend_runtime_policy_enabled(
            policy, "BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8");
}

static int small_dense_native_quant_ffn_down_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
               policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_DOWN") ||
           runtime_compat_enabled2(
        policy,
        "BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_FFN_DOWN",
        "BN_CUDA_ENABLE_SMALL_DENSE_EXACT_NATIVE_FFN_DOWN",
        "BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN") ||
           bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
}

static int small_dense_native_quant_enabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT",
                                   "BN_GPU_SMALL_DENSE_EXACT_NATIVE",
                                   "BN_GPU_Q4_Q8");
}

static int small_dense_native_quant_gateup_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_GATEUP",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_GATEUP",
        "BN_GPU_Q4_Q8_DISABLE_GATEUP");
}

static int small_dense_native_quant_ffn_down_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_FFN_DOWN",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_FFN_DOWN",
        "BN_GPU_Q4_Q8_DISABLE_FFN_DOWN");
}

static int small_dense_native_quant_attn_only(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_GPU_SMALL_DENSE_NATIVE_QUANT_ATTN_ONLY",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY",
        "BN_GPU_Q4_Q8_ATTN_ONLY");
}

static int small_dense_native_quant_ffn_only(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_ONLY",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY",
        "BN_GPU_Q4_Q8_FFN_ONLY");
}

static const char *small_dense_native_quant_from_layer_env(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_value2(policy,
        "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FROM_LAYER",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER",
        "BN_GPU_Q4_Q8_FROM_LAYER");
}

static const char *small_dense_native_quant_to_layer_env(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_value2(policy,
        "BN_GPU_SMALL_DENSE_NATIVE_QUANT_TO_LAYER",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_TO_LAYER",
        "BN_GPU_Q4_Q8_TO_LAYER");
}

static const char *small_dense_native_quant_tail_layer_env(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_value2(policy,
        "BN_GPU_SMALL_DENSE_NATIVE_QUANT_TAIL_NATIVE",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_TAIL_NATIVE",
        "BN_GPU_Q4_Q8_TAIL_NATIVE");
}

static int small_dense_prefill_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_DISABLE_SMALL_DENSE_PREFILL",
                                  "BN_CUDA_DISABLE_SMALL_QWEN_PREFILL");
}

static int native_quant_logits_refine_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE",
        "BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
        "BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
}

static int native_quant_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE",
        "BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
        "BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
}

static int all_active_two_kquant_moe_fast_ffn_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
        "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
}

static int all_active_two_kquant_moe_fast_graph_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_GRAPH",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_FAST_GRAPH",
        "BN_CUDA_DISABLE_QWEN2MOE_FAST_MOE_GRAPH");
}

static int all_active_two_kquant_moe_cublas_decode_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CUBLAS_DECODE",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_CUBLAS_DECODE",
        "BN_CUDA_ENABLE_QWEN2MOE_MOE_CUBLAS_DECODE");
}

static int all_active_two_kquant_moe_fast_route_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_ROUTE",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_ALL2_FAST",
        "BN_CUDA_ENABLE_QWEN2MOE_MOE_ALL2_FAST");
}

static int all_active_two_kquant_moe_dot_prepared_input_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOT_PREPARED_INPUT_DEFAULT",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q8K_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_MOE_Q8K_DEFAULT");
}

static int all_active_two_kquant_route_dot_prepared_input_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_DOT_PREPARED_INPUT_DEFAULT",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_ROUTE_Q8K_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_ROUTE_Q8K_DEFAULT");
}

static int all_active_two_kquant_fast_prepared_gateup_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP",
        "BN_CUDA_ENABLE_QWEN2MOE_FAST_Q8K_GATEUP");
}

static int all_active_two_kquant_fast_prepared_gateup_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP",
        "BN_CUDA_DISABLE_QWEN2MOE_FAST_Q8K_GATEUP");
}

static int all_active_two_kquant_moe_down_pair_path_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_PAIR_DOWN");
}

static int all_active_two_kquant_moe_down_pair_path_f32_layers_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS");
}

static int all_active_two_kquant_moe_down_pair_path_f32_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer) {
    return runtime_compat_layer_selected2(policy,
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS", layer);
}

static int all_active_two_kquant_moe_down_ordered_quant_path_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
}

static int all_active_two_kquant_moe_down_ordered_quant_path_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
}

static int all_active_two_kquant_moe_down_f32_cache_default_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
}

static int all_active_two_kquant_moe_down_f32_cache_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
}

static int all_active_two_kquant_moe_down_f32_all_active_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_ALL_ACTIVE_DOWN",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_ALL2_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_ALL2_DOWN");
}

static int all_active_two_kquant_moe_down_f32_cache_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_CACHE",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_CACHE",
        "BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_CACHE");
}

static int all_active_two_kquant_moe_down_float_4row_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN_DEFAULT");
}

static int all_active_two_kquant_moe_down_float_4row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN");
}

static const char *all_active_two_kquant_moe_down_f32_4row_layers(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_value3(policy,
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_F32_4ROW_DOWN_LAYERS",
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_LAYERS",
        "BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS");
}

static int all_active_two_kquant_moe_down_f32_4row_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer) {
    return runtime_compat_layer_selected3(policy,
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_F32_4ROW_DOWN_LAYERS",
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_LAYERS",
        "BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS",
        "BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS", layer);
}

static int all_active_two_kquant_moe_down_f32_4row_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled3(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT");
}

static int all_active_two_kquant_moe_down_f32_4row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled3(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_4ROW_DOWN",
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN");
}

static float all_active_two_kquant_down_skip_eps_or_default(
    const BnBackendRuntimePolicy *policy, float default_eps) {
    return runtime_compat_float_or_default2(policy,
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_DOWN_SKIP_EPS",
        "BN_CUDA_ALL2_Q4Q6_DOWN_SKIP_EPS",
        "BN_CUDA_QWEN2MOE_DOWN_SKIP_EPS", default_eps);
}

static int all_active_two_kquant_moe_cpu_attention_safe_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
}

static int all_active_two_kquant_moe_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_LOGITS_REFINE",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE",
        "BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE");
}

static int all_active_two_kquant_moe_cpu_moe_safe_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_MOE_SAFE",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE");
}

static int all_active_two_kquant_moe_reference_attention_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled3(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_REFERENCE_ATTN",
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_ATTN",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN",
        "BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN");
}

static int all_active_two_kquant_moe_cpu_route_resident_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ROUTE_RESIDENT",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT",
        "BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT");
}

static int all_active_two_kquant_moe_reference_gpu_route_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled3(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_REFERENCE_GPU_ROUTE",
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE");
}

static int all_active_two_kquant_moe_reference_gpu_route_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled3(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_REFERENCE_GPU_ROUTE",
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
        "BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE");
}

static const char *all_active_two_kquant_moe_route_from_layer_value(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_value2(policy,
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_FROM_LAYER",
        "BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER",
        "BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
}

static const char *all_active_two_kquant_moe_route_to_layer_value(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_value2(policy,
        "BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_TO_LAYER",
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
    int small_dense_native_quant;
    int prefill_decode_fallback;
    int prefill_chain;
    int matvec_fallback;
    int dense_batch_prefill_shape;
    int lazy_moe_aux_cache;
    int native_quant_logits_refine_default;
    int all_active_two_kquant_moe_logits_refine_default;
    int decode_graph_cache;
    int moe_reference_attention;
    int ssm_graph;
    int large_hybrid_argmax;
    int all_active_two_moe_direct_route;
    int resident_moe_ffn;
    int moe_gateup_split;
    int individual_upload_quant_only;
    int logits_kquant_f32_cache;
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
    .small_dense_native_quant = 1,
    .prefill_decode_fallback = 1,
    .prefill_chain = 1,
    .matvec_fallback = 1,
    .dense_batch_prefill_shape = 1,
    .lazy_moe_aux_cache = 1,
    .native_quant_logits_refine_default = 1,
    .all_active_two_kquant_moe_logits_refine_default = 1,
    .decode_graph_cache = 1,
    .moe_reference_attention = 1,
    .ssm_graph = 1,
    .large_hybrid_argmax = 1,
    .all_active_two_moe_direct_route = 1,
    .resident_moe_ffn = 1,
    .moe_gateup_split = 1,
    .individual_upload_quant_only = 1,
    .logits_kquant_f32_cache = 1,
    .logits_f16_cache = 1,
    .moe_prefers_quant_only = 1,
    .fused_gateup_requires_backend_opt_in = 1,
    .moe_down_cublas_cache = 1,
    .suppress_implicit_kquant_logits_refine = 1,
    .suppress_implicit_native_quant_logits_refine = 1,
};

static const BnGPUPolicyBackendCaps *
gpu_policy_backend_caps(const BnGPUBackend *gpu) {
    if (bn_gpu_backend_is_cuda(gpu))
        return &GPU_POLICY_BACKEND_CAPS_CUDA;
    return &GPU_POLICY_BACKEND_CAPS_NONE;
}

static int gpu_policy_cuda_moe_routed_ffn_disabled(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_ROUTED_FFN");
}

static int gpu_policy_cuda_moe_routed_ffn_enabled(
    const BnGPUBackend *gpu, int eligible) {
    return eligible && !gpu_policy_cuda_moe_routed_ffn_disabled(gpu);
}

int bn_gpu_policy_moe_resident_routed_ffn_enabled(
    const BnGPUBackend *gpu, int eligible) {
    if (gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_MOE_ALL_LAYOUT"))
        return 0;
    if (bn_gpu_backend_is_cuda(gpu))
        return gpu_policy_cuda_moe_routed_ffn_enabled(gpu, eligible);
    if (bn_gpu_backend_is_metal(gpu))
        return eligible &&
               (bn_gpu_policy_metal_routed_moe_decode_enabled(
                    &gpu->runtime_policy) ||
                bn_gpu_policy_metal_cpu_route_resident_moe_enabled(
                    &gpu->runtime_policy));
    return 0;
}

BnBackendPlacement bn_gpu_policy_backend_placement(const BnGPUBackend *gpu) {
    if (bn_gpu_backend_is_metal(gpu))
        return BN_BACKEND_METAL;
    if (bn_gpu_backend_is_webgpu(gpu))
        return BN_BACKEND_WEBGPU;
    if (bn_gpu_backend_is_cuda(gpu))
        return BN_BACKEND_CUDA;
    return BN_BACKEND_GPU_UNKNOWN;
}

static int gpu_policy_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type) {
    return bn_backend_quant_moe_resident_routed_ffn_supported(gate_type,
                                                              up_type,
                                                              down_type);
}

int bn_gpu_policy_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type) {
    return gpu_policy_moe_resident_routed_ffn_quant_eligible(gate_type,
                                                             up_type,
                                                             down_type);
}

int bn_gpu_policy_backend_moe_resident_routed_ffn_eligible(
    const BnGPUBackend *gpu,
    int standard_quant_eligible,
    int metal_quant_eligible,
    int layout_eligible) {
    return layout_eligible &&
           (standard_quant_eligible ||
            (bn_gpu_backend_is_metal(gpu) && metal_quant_eligible));
}

static int gpu_policy_moe_all_f16_cache_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_MOE_ALL_F16_CACHE");
}

static int gpu_policy_moe_all_f16_cache_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_ALL_F16_CACHE");
}

static int gpu_policy_moe_gateup_f16_cache_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE");
}

static int gpu_policy_moe_gateup_f16_cache_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE");
}

static int gpu_policy_partial_moe_f16_cache_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE");
}

static int gpu_policy_moe_fit_debug_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DEBUG_MOE_FIT");
}

static int gpu_policy_keep_individual_f16_cache_requested(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE");
}

static int gpu_policy_moe_lazy_aux_cache_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");
}

static int gpu_policy_logits_kquant_f32_cache_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_LOGITS_KQUANT_F32_CACHE",
        "BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE");
}

static int gpu_policy_logits_kquant_f32_cache_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_LOGITS_KQUANT_F32_CACHE",
        "BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE");
}

static int gpu_policy_logits_f16_cache_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_LOGITS_F16_CACHE");
}

static int gpu_policy_cuda_cublas_logits_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_CUBLAS_LOGITS");
}

static int gpu_policy_cuda_f32_logits_matvec_requested(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_F32_LOGITS_MATVEC");
}

static int gpu_policy_cuda_f32_logits_matvec_disabled(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_F32_LOGITS_MATVEC");
}

static int gpu_policy_cuda_f16_logits_matvec_requested(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_F16_LOGITS_MATVEC");
}

static int gpu_policy_moe_down_kquant_f32_cache_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE",
        "BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
}

static int gpu_policy_moe_down_kquant_f32_cache_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE",
        "BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
}

static int gpu_policy_moe_down_small_expert_f32_cache_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_ENABLE_MOE_DOWN_SMALL_EXPERT_F32_CACHE",
        "BN_CUDA_ENABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE",
        "BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE");
}

static int gpu_policy_moe_down_small_expert_f32_cache_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_SMALL_EXPERT_F32_CACHE",
        "BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE",
        "BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");
}

static int gpu_policy_moe_all_f16_cache_forced(const BnGPUBackend *gpu) {
    return gpu_policy_moe_all_f16_cache_requested(gpu);
}

int bn_gpu_policy_moe_all_f16_cache_forced(const BnGPUBackend *gpu) {
    return gpu_policy_moe_all_f16_cache_forced(gpu);
}

static int gpu_policy_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int native_quant_f16_cache) {
    if (!bn_gpu_backend_can_create_f16_cache_buffer(gpu) ||
        gpu_policy_moe_all_f16_cache_disabled(gpu))
        return 0;
    if (gpu_policy_moe_all_f16_cache_forced(gpu))
        return 1;
    if (!native_quant_f16_cache)
        return 0;
    return bn_backend_quant_moe_all_f16_cache_supported(tensor_type);
}

int bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int native_quant_f16_cache) {
    return gpu_policy_moe_all_f16_cache_enabled_for_type(
        gpu, tensor_type, native_quant_f16_cache);
}

static int gpu_policy_moe_gateup_f16_cache_enabled(const BnGPUBackend *gpu,
                                                    int eligible) {
    return eligible &&
           gpu_policy_moe_gateup_f16_cache_requested(gpu) &&
           !gpu_policy_moe_gateup_f16_cache_disabled(gpu);
}

int bn_gpu_policy_moe_gateup_f16_cache_enabled(const BnGPUBackend *gpu,
                                                int eligible) {
    return gpu_policy_moe_gateup_f16_cache_enabled(gpu, eligible);
}

static int gpu_policy_partial_moe_f16_cache_enabled(const BnGPUBackend *gpu,
                                                     int eligible) {
    return eligible &&
           gpu_policy_partial_moe_f16_cache_requested(gpu);
}

int bn_gpu_policy_partial_moe_f16_cache_enabled(const BnGPUBackend *gpu,
                                                 int eligible) {
    return gpu_policy_partial_moe_f16_cache_enabled(gpu, eligible);
}

static int gpu_policy_moe_fit_debug_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_moe_fit_debug_requested(gpu);
}

int bn_gpu_policy_moe_residency_fit_debug_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_moe_fit_debug_enabled(gpu);
}

static int gpu_policy_keep_individual_f16_cache_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_keep_individual_f16_cache_requested(gpu);
}

static int gpu_policy_moe_lazy_aux_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_moe_lazy_aux_cache_requested(gpu);
}

int bn_gpu_policy_moe_lazy_aux_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_moe_lazy_aux_cache_enabled(gpu);
}

static int gpu_policy_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->individual_upload_quant_only &&
           bn_gpu_backend_can_create_quant_only_buffer(gpu) &&
           !gpu_policy_keep_individual_f16_cache_enabled(gpu);
}

int bn_gpu_policy_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_individual_upload_quant_only_enabled(gpu);
}

static int gpu_policy_logits_kquant_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu_policy_backend_caps(gpu)->logits_kquant_f32_cache &&
           bn_gpu_backend_can_create_kquant_f32_cache_buffer(gpu) &&
           bn_backend_quant_logits_kquant_f32_cache_supported(tensor_type) &&
           gpu_policy_logits_kquant_f32_cache_requested(
               gpu ? &gpu->runtime_policy : NULL) &&
           !gpu_policy_logits_kquant_f32_cache_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_gpu_policy_logits_kquant_f32_cache_enabled(const BnGPUBackend *gpu,
                                                  int tensor_type) {
    return gpu_policy_logits_kquant_f32_cache_enabled(gpu, tensor_type);
}

static int gpu_policy_logits_f16_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->logits_f16_cache &&
           bn_gpu_backend_can_create_f16_cache_buffer(gpu) &&
           gpu_policy_logits_f16_cache_requested(gpu);
}

int bn_gpu_policy_logits_f16_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_logits_f16_cache_enabled(gpu);
}

int bn_gpu_policy_cuda_cublas_logits_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_cuda_cublas_logits_requested(gpu);
}

int bn_gpu_policy_cuda_f32_logits_matvec_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_cuda_f32_logits_matvec_requested(gpu) &&
           !gpu_policy_cuda_f32_logits_matvec_disabled(gpu);
}

int bn_gpu_policy_cuda_f16_logits_matvec_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_cuda_f16_logits_matvec_requested(gpu);
}

int bn_gpu_policy_moe_down_kquant_f32_cache_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_create_kquant_f32_cache_buffer(gpu) &&
           !gpu_policy_moe_all_f16_cache_forced(gpu) &&
           !gpu_policy_moe_down_kquant_f32_cache_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_gpu_policy_moe_down_kquant_f32_cache_forced(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_moe_down_kquant_f32_cache_requested(policy);
}

int bn_gpu_policy_moe_down_kquant_f32_cache_default_for_cols(const BnBackendRuntimePolicy *policy, int cols) {
    return cols > 1024 ||
           bn_gpu_policy_moe_down_kquant_f32_cache_forced(policy);
}

static int gpu_policy_moe_down_kquant_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache) {
    return !force_f16_cache &&
           bn_backend_quant_moe_down_kquant_f32_cache_supported(tensor_type) &&
           bn_gpu_policy_moe_down_kquant_f32_cache_default_for_cols(
               gpu ? &gpu->runtime_policy : NULL, cols) &&
           bn_gpu_policy_moe_down_kquant_f32_cache_enabled(gpu);
}

int bn_gpu_policy_moe_down_kquant_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache) {
    return gpu_policy_moe_down_kquant_f32_cache_preferred(
        gpu, tensor_type, cols, force_f16_cache);
}

static size_t gpu_policy_moe_down_kquant_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts) {
    if (!gpu_policy_moe_down_kquant_f32_cache_preferred(
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

    if (bn_gpu_policy_moe_down_kquant_f32_cache_forced(
            gpu ? &gpu->runtime_policy : NULL))
        return bytes;

    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(
        gpu ? &gpu->runtime_policy : NULL, 512, 0);
    if (max_mb <= 0)
        return bytes;
    size_t max_bytes = (size_t)max_mb * 1024u * 1024u;
    return bytes <= max_bytes ? bytes : 0;
}

size_t bn_gpu_policy_moe_down_kquant_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts) {
    return gpu_policy_moe_down_kquant_f32_cache_bytes(
        gpu, tensor_type, rows, cols, n_experts);
}

static int gpu_policy_moe_down_kquant_f32_cache_requires_full_buffer(
    const BnBackendRuntimePolicy *policy,
    int tensor_type) {
    return bn_backend_quant_moe_down_kquant_f32_cache_supported(tensor_type) &&
           bn_gpu_policy_moe_down_kquant_f32_cache_forced(policy);
}

int bn_gpu_policy_moe_down_kquant_f32_cache_requires_full_buffer(
    const BnBackendRuntimePolicy *policy,
    int tensor_type) {
    return gpu_policy_moe_down_kquant_f32_cache_requires_full_buffer(
        policy, tensor_type);
}

static int gpu_policy_moe_down_small_expert_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return bn_gpu_backend_can_create_kquant_f32_cache_buffer(gpu) &&
           bn_backend_quant_moe_down_small_kquant_f32_cache_supported(
               tensor_type) &&
           gpu_policy_moe_down_small_expert_f32_cache_requested(
               gpu ? &gpu->runtime_policy : NULL) &&
           !gpu_policy_moe_down_small_expert_f32_cache_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_gpu_policy_moe_down_small_expert_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return gpu_policy_moe_down_small_expert_f32_cache_enabled(
        gpu, tensor_type);
}

static int gpu_policy_moe_quant_only_after_cache(int tensor_type,
                                                 int native_quant_f16_cache) {
    return bn_backend_quant_moe_quant_only_after_cache(tensor_type,
                                                       native_quant_f16_cache);
}

int bn_gpu_policy_moe_quant_only_after_cache(int tensor_type,
                                             int native_quant_f16_cache) {
    return gpu_policy_moe_quant_only_after_cache(tensor_type,
                                                 native_quant_f16_cache);
}

int bn_gpu_policy_moe_prefers_quant_only(const BnGPUBackend *gpu,
                                         int tensor_type) {
    return gpu_policy_backend_caps(gpu)->moe_prefers_quant_only &&
           bn_backend_quant_moe_prefers_quant_only(tensor_type);
}

static int gpu_policy_matvec_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DISABLE_MATVEC");
}

static int gpu_policy_native_quant_matvec_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_DISABLE_NATIVE_QUANT_MATVEC",
                                  "BN_CUDA_DISABLE_Q8_0");
}

static int gpu_policy_legacy_5bit_matvec_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_LEGACY_5BIT_MATVEC",
        "BN_CUDA_DISABLE_Q5_0");
}

static int gpu_policy_asymmetric_kquant_matvec_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_MATVEC",
        "BN_CUDA_DISABLE_Q4_K");
}

static int gpu_policy_deinterleaved_kquant_matvec_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_MATVEC",
        "BN_CUDA_DISABLE_Q5_K");
}

static int gpu_policy_down_kquant_matvec_disabled_by_type(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC",
                                  "BN_CUDA_DISABLE_Q6_K");
}

static int gpu_policy_native_kquant_matvec_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_PREPARED_NATIVE_QUANT_MATVEC",
        "BN_CUDA_DISABLE_Q8_K");
}

static int gpu_policy_matmul_batch_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DISABLE_MATMUL_BATCH");
}

static int gpu_policy_matvec_batch_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DISABLE_MATVEC_BATCH");
}

static int gpu_policy_small_state_native_quant_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_SMALL_STATE_NATIVE_QUANT",
        "BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
}

static int gpu_policy_small_state_native_quant_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_SMALL_STATE_NATIVE_QUANT",
        "BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");
}

int bn_gpu_policy_matvec_disabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_matvec_disabled(policy);
}

static int gpu_policy_matvec_type_disabled(
    const BnBackendRuntimePolicy *policy, int tensor_type) {
    if (bn_backend_quant_supports_native_quant_small_state_matvec(tensor_type))
        return gpu_policy_native_quant_matvec_disabled(policy);
    if (bn_backend_quant_supports_legacy_block_matvec(tensor_type))
        return gpu_policy_legacy_5bit_matvec_disabled(policy);
    if (bn_backend_quant_supports_asymmetric_kquant_dot_matvec(tensor_type))
        return gpu_policy_asymmetric_kquant_matvec_disabled(policy);
    if (bn_backend_quant_supports_deinterleaved_kquant_prepared_input_matvec(
            tensor_type))
        return gpu_policy_deinterleaved_kquant_matvec_disabled(policy);
    if (bn_backend_quant_moe_down_uses_down_kquant(tensor_type))
        return gpu_policy_down_kquant_matvec_disabled_by_type(policy);
    if (bn_backend_quant_supports_native_kquant_matvec(tensor_type))
        return gpu_policy_native_kquant_matvec_disabled(policy);
    return 0;
}

int bn_gpu_policy_matvec_type_disabled(
    const BnBackendRuntimePolicy *policy, int tensor_type) {
    return gpu_policy_matvec_type_disabled(policy, tensor_type);
}

int bn_gpu_policy_matvec_type_supported(
    const BnBackendRuntimePolicy *policy, int tensor_type) {
    return bn_backend_quant_gpu_matvec_supported(tensor_type) &&
           !gpu_policy_matvec_type_disabled(policy, tensor_type);
}

int bn_gpu_policy_matmul_batch_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_matmul_batch_disabled(policy);
}

int bn_gpu_policy_matvec_batch_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_matvec_batch_disabled(policy);
}

static int small_state_native_quant_enabled(
    const BnBackendRuntimePolicy *policy,
    int uses_float_kquant_fallback) {
    if (gpu_policy_small_state_native_quant_requested(policy))
        return 1;
    return !gpu_policy_small_state_native_quant_disabled(policy) &&
           !uses_float_kquant_fallback;
}

int bn_gpu_policy_small_state_native_quant_enabled(
    const BnBackendRuntimePolicy *policy,
    int uses_float_kquant_fallback) {
    return small_state_native_quant_enabled(policy,
                                             uses_float_kquant_fallback);
}

static int small_state_native_quant_disabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_small_state_native_quant_disabled(policy);
}

int bn_gpu_policy_small_state_native_quant_disabled(
    const BnBackendRuntimePolicy *policy) {
    return small_state_native_quant_disabled(policy);
}

size_t bn_gpu_policy_max_storage_binding_bytes(
    const BnBackendRuntimePolicy *policy, size_t backend_limit) {
    size_t max_storage_binding = backend_limit;
    if (max_storage_binding == 0)
        max_storage_binding = 128ull * 1024ull * 1024ull;
    const char *override_mb = bn_backend_runtime_policy_get(
        policy, "BN_GPU_MAX_STORAGE_BINDING_MB");
    if (override_mb) {
        long mb = strtol(override_mb, NULL, 10);
        if (mb >= 0)
            max_storage_binding = (size_t)mb * 1024ull * 1024ull;
    }
    return max_storage_binding;
}

static size_t runtime_mb_or_default(const BnBackendRuntimePolicy *policy,
                                    const char *name, size_t def) {
    const char *s = bn_backend_runtime_policy_get(policy, name);
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

static size_t positive_runtime_mb_or_default(
    const BnBackendRuntimePolicy *policy, const char *name, size_t def) {
    const char *s = bn_backend_runtime_policy_get(policy, name);
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

static size_t gpu_policy_layout_reserve_bytes(
    const BnBackendRuntimePolicy *policy) {
    return mb_to_bytes_saturating(
        runtime_mb_or_default(policy, "BN_CUDA_LAYOUT_RESERVE_MB", 512));
}

size_t bn_gpu_policy_layout_reserve_bytes(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_layout_reserve_bytes(policy);
}

static size_t gpu_policy_moe_full_reserve_bytes(
    const BnBackendRuntimePolicy *policy) {
    return mb_to_bytes_saturating(
        runtime_mb_or_default(policy, "BN_CUDA_MOE_FULL_RESERVE_MB", 512));
}

size_t bn_gpu_policy_moe_full_reserve_bytes(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_moe_full_reserve_bytes(policy);
}

static int gpu_policy_cuda_cublas_matmul_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_CUBLAS_MATMUL");
}

static const char *gpu_policy_cuda_cublas_gemm_algo_value(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_get(policy,
                                         "BN_CUDA_CUBLAS_GEMM_ALGO");
}

static int gpu_policy_cuda_down_kquant_cublas_f16_cache_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_CUBLAS_F16_CACHE",
        "BN_CUDA_DISABLE_Q6K_CUBLAS_F16");
}

static int gpu_policy_cuda_native_quant_matmul_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_NATIVE_QUANT_MATMUL",
        "BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL");
}

static int gpu_policy_cuda_native_quant_matmul_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_MATMUL",
        "BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL");
}

static int gpu_policy_cuda_f16_native_quant_matmul_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATMUL",
        "BN_CUDA_DISABLE_F16_Q8_0_MATMUL");
}

static int gpu_policy_prepared_kquant_input_cache_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_PREPARED_KQUANT_INPUT_CACHE",
        "BN_CUDA_DISABLE_Q8K_INPUT_CACHE");
}

static int gpu_policy_force_asymmetric_kquant_quant_matmul_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_FORCE_ASYMMETRIC_KQUANT_QUANT_MATMUL",
        "BN_CUDA_FORCE_Q4K_QUANT_MATMUL");
}

static int gpu_policy_force_down_kquant_quant_matmul_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_FORCE_DOWN_KQUANT_QUANT_MATMUL",
        "BN_CUDA_FORCE_Q6K_QUANT_MATMUL");
}

static int gpu_policy_down_kquant_4warp_long_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_LONG",
        "BN_CUDA_DISABLE_Q6K_4WARP_LONG");
}

static int gpu_policy_down_kquant_4warp_shape_1536x8960_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_1536_8960",
        "BN_CUDA_DISABLE_Q6K_4WARP_1536_8960");
}

static int gpu_policy_down_kquant_4warp_mid_cols_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_4WARP_5120",
        "BN_CUDA_ENABLE_Q6K_4WARP_5120");
}

static int gpu_policy_down_kquant_5warp_shape_1536x8960_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_1536_8960",
        "BN_CUDA_DISABLE_Q6K_5WARP_1536_8960");
}

static int gpu_policy_down_kquant_5warp_shape_2560x9728_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_2560_9728",
        "BN_CUDA_DISABLE_Q6K_5WARP_2560_9728");
}

static int gpu_policy_down_kquant_3warp_shape_1536x8960_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_1536_8960",
        "BN_CUDA_DISABLE_Q6K_3WARP_1536_8960");
}

static int gpu_policy_down_kquant_3warp_shape_2560x9728_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_2560_9728",
        "BN_CUDA_DISABLE_Q6K_3WARP_2560_9728");
}

static int gpu_policy_down_kquant_2warp_long_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_2WARP_LONG",
        "BN_CUDA_ENABLE_Q6K_2WARP_LONG");
}

static int gpu_policy_down_kquant_2warp_long_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_2WARP_LONG",
        "BN_CUDA_DISABLE_Q6K_2WARP_LONG");
}

static int gpu_policy_down_kquant_matvec4_1024x2560_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_1024_2560",
        "BN_CUDA_ENABLE_Q6K_MATVEC4_1024_2560");
}

static int gpu_policy_down_kquant_matvec4_512x2048_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_512_2048",
        "BN_CUDA_ENABLE_Q6K_MATVEC4_512_2048");
}

int bn_gpu_policy_cuda_cublas_matmul_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_cublas_matmul_disabled(policy);
}

int bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(
    const BnBackendRuntimePolicy *policy, int default_index) {
    const char *env = gpu_policy_cuda_cublas_gemm_algo_value(policy);
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

int bn_gpu_policy_cuda_down_kquant_cublas_f16_cache_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_down_kquant_cublas_f16_cache_disabled(policy) &&
           !bn_gpu_policy_moe_down_kquant_f32_cache_forced(policy);
}

int bn_gpu_policy_cuda_native_quant_matmul_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_native_quant_matmul_requested(policy) &&
           !gpu_policy_cuda_native_quant_matmul_disabled(policy);
}

int bn_gpu_policy_cuda_f16_native_quant_matmul_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_f16_native_quant_matmul_disabled(policy);
}

int bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled(
    const BnBackendRuntimePolicy *policy) {
    return cuda_native_quant_prepared_input_split_requested(policy) &&
           !cuda_native_quant_prepared_input_split_disabled(policy);
}

int bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled(
    const BnBackendRuntimePolicy *policy) {
    return cuda_native_quant_prepared_input_requested(policy) &&
           !cuda_native_quant_prepared_input_disabled(policy);
}

int bn_gpu_policy_cuda_native_quant_prepared_input_logits_disabled(
    const BnBackendRuntimePolicy *policy) {
    return cuda_native_quant_prepared_input_logits_disabled(policy);
}

int bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(
    const BnBackendRuntimePolicy *policy,
    int prepared_input_logits_disabled) {
    return !prepared_input_logits_disabled &&
           (cuda_native_quant_prepared_input_logits_requested(policy) ||
            !cuda_native_quant_prepared_input_logits_disabled(policy));
}

int bn_gpu_policy_prepared_kquant_input_cache_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_prepared_kquant_input_cache_disabled(policy);
}

int bn_gpu_policy_cuda_quant_matmul_preferred_for_type(
    const BnBackendRuntimePolicy *policy,
    int tensor_type,
    int f16_native_quant_matmul_enabled) {
    return (bn_backend_quant_avoids_quant_matmul_on_f16_input(
                tensor_type) &&
           !f16_native_quant_matmul_enabled) ||
           (bn_backend_quant_supports_requested_asymmetric_kquant_quant_matmul(
                tensor_type) &&
            gpu_policy_force_asymmetric_kquant_quant_matmul_requested(policy)) ||
           (bn_backend_quant_supports_requested_down_kquant_quant_matmul(
                tensor_type) &&
            gpu_policy_force_down_kquant_quant_matmul_requested(policy));
}

int bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols) {
    if (gpu_policy_down_kquant_4warp_long_disabled(policy))
        return 0;
    if (rows == 1536 && cols == 8960 &&
        !gpu_policy_down_kquant_4warp_shape_1536x8960_disabled(policy))
        return 1;
    if (rows >= 2560 && cols >= 8192 && cols <= 16384)
        return 1;
    return rows >= 2560 && cols >= 5120 && cols < 8192 &&
           gpu_policy_down_kquant_4warp_mid_cols_requested(policy);
}

int bn_gpu_policy_cuda_down_kquant_5warp_shape_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols) {
    if (rows == 1536 && cols == 8960 &&
        !gpu_policy_down_kquant_5warp_shape_1536x8960_disabled(policy))
        return 1;
    if (rows == 2560 && cols == 9728 &&
        !gpu_policy_down_kquant_5warp_shape_2560x9728_disabled(policy))
        return 1;
    return 0;
}

int bn_gpu_policy_cuda_down_kquant_3warp_shape_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols) {
    if (rows == 1536 && cols == 8960 &&
        !gpu_policy_down_kquant_3warp_shape_1536x8960_disabled(policy))
        return 1;
    if (rows == 2560 && cols == 9728 &&
        !gpu_policy_down_kquant_3warp_shape_2560x9728_disabled(policy))
        return 1;
    return 0;
}

int bn_gpu_policy_cuda_down_kquant_2warp_long_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols) {
    return rows >= 2560 && cols >= 8192 && cols <= 12288 &&
           gpu_policy_down_kquant_2warp_long_requested(policy) &&
           !gpu_policy_down_kquant_2warp_long_disabled(policy);
}

int bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(const BnBackendRuntimePolicy *policy, int rows, int cols) {
    if (rows == 1024 && cols == 2560 &&
        !gpu_policy_down_kquant_matvec4_1024x2560_requested(policy))
        return 1;
    if (rows == 512 && cols == 2048 &&
        !gpu_policy_down_kquant_matvec4_512x2048_requested(policy))
        return 1;
    return 0;
}

int bn_gpu_policy_moe_route_all_active_two(int n_experts, int k) {
    return n_experts == 2 && k == 2;
}

int bn_gpu_policy_moe_route_expanded_topk(int n_experts, int k) {
    return n_experts > 2 || k > 2;
}

int bn_gpu_policy_moe_route_all_active_two_large_hidden(int n_experts,
                                                        int k,
                                                        int hidden_dim) {
    return bn_gpu_policy_moe_route_all_active_two(n_experts, k) &&
           hidden_dim >= 4096;
}

static int gpu_policy_moe_down_4row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_DOWN_4ROW");
}

static int gpu_policy_moe_down_8row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_DOWN_8ROW");
}

static int gpu_policy_moe_down_halfwarp_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_MOE_DOWN_HALFWARP");
}

static int gpu_policy_moe_down_halfwarp_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_DOWN_HALFWARP");
}

static int gpu_policy_moe_down_split4_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_MOE_DOWN_SPLIT4");
}

static int gpu_policy_moe_down_split4_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_DOWN_SPLIT4");
}

static int gpu_policy_moe_down_scatter_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_DOWN_SCATTER");
}

static int gpu_policy_moe_down_scatter_16row_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_MOE_DOWN_SCATTER_16ROW");
}

static int gpu_policy_moe_down_float_path_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_FLOAT_PATH",
        "BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN");
}

static int gpu_policy_moe_down_pair_path_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_PATH",
        "BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN");
}

static int gpu_policy_moe_down_f32_pair2_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_PAIR2",
        "BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2");
}

static int gpu_policy_moe_down_f32_pair2_4row_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_4ROW",
        "BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW");
}

static int gpu_policy_all_active_two_kquant_moe_down_accum_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM",
        "BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM");
}

static int gpu_policy_all_active_two_kquant_moe_down_accum_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM",
        "BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM");
}

static int gpu_policy_all_active_two_kquant_moe_down_pair4_sum_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_PAIR4_SUM",
        "BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM");
}

static int gpu_policy_moe_down_prepared_native_quant_4row_sum_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_4ROW_SUM",
        "BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM");
}

static int gpu_policy_moe_down_prepared_native_quant_8row_sum_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_DOWN_KQUANT_K8_8ROW_SUM",
        "BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM");
}

static int gpu_policy_all_active_two_kquant_moe_down_fixed_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_FIXED",
        "BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED");
}

static int gpu_policy_moe_down_resid_rmsnorm_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE");
}

static int gpu_policy_moe_down_prepared_native_quant_shape_2048x768_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_SHAPE_2048_768",
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_EXACT_2048_768",
        "BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768");
}

static int gpu_policy_all_active_two_kquant_moe_down_accum_4row_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM_4ROW",
        "BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW");
}

static int gpu_policy_moe_down_prepared_pair_4row_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_4ROW",
        "BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW");
}

static int gpu_policy_moe_down_f16_cache_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F16_CACHE",
        "BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE");
}

static int gpu_policy_moe_down_f16_cache_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F16_CACHE",
        "BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE");
}

static int gpu_policy_moe_down_prepared_pair8_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN",
        "BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN");
}

static int gpu_policy_moe_down_prepared_pair8_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN",
        "BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN");
}

static int gpu_policy_moe_down_prepared_8row_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW",
        "BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW");
}

static int gpu_policy_moe_down_prepared_8row_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW",
        "BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW");
}

static int gpu_policy_moe_gateup_prepared_dot_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_DOT",
        "BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP");
}

static int gpu_policy_moe_gateup_prepared_8row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_GATEUP_8ROW");
}

static int gpu_policy_moe_gateup_prepared_split_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_MOE_GATEUP_SPLIT");
}

static int gpu_policy_moe_gateup_prepared_split_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
}

static int gpu_policy_moe_dot_input_forced(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOT_INPUT",
        "BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT");
}

static int gpu_policy_all_active_two_moe_dot_input_forced(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_DOT_INPUT",
        "BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2");
}

static int gpu_policy_moe_router_fused_topk_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_MOE_ROUTER_FUSED_TOPK");
}

static int gpu_policy_moe_router_fused_topk_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ROUTER_FUSED_TOPK");
}

static int gpu_policy_moe_router_warp_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ROUTER_WARP");
}

static int gpu_policy_moe_router_4warp_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ROUTER_4WARP");
}

static int gpu_policy_moe_router_2warp_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ROUTER_2WARP");
}

static int gpu_policy_moe_router_warp_topk_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ROUTER_WARP_TOPK");
}

static int gpu_policy_moe_block_prepared_batch_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_BATCH",
        "BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1");
}

static int gpu_policy_moe_block_prepared_decode_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_DECODE",
        "BN_CUDA_DISABLE_Q8_MOE_Q8X");
}

static int gpu_policy_moe_gateup_block_2row_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_GATEUP_2ROW",
        "BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW");
}

static int gpu_policy_moe_down_block_4row_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MOE_NATIVE_QUANT_DOWN_4ROW",
        "BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW");
}

static int gpu_policy_moe_down_block_2row_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_DOWN_2ROW",
        "BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW");
}

static int gpu_policy_moe_all_active_two_fast_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ALL2_FAST");
}

static int gpu_policy_moe_prepared_dot_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOT",
        "BN_CUDA_DISABLE_MOE_Q4K_Q8K_DOT");
}

static int gpu_policy_moe_internal_profile_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_PROFILE_MOE_INTERNAL");
}

static int gpu_policy_moe_all_active_two_fixed_prepared_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_FIXED",
        "BN_CUDA_DISABLE_MOE_Q4K_ALL2_FIXED");
}

static int gpu_policy_moe_gateup_prepared_4row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_4ROW",
        "BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW");
}

int bn_gpu_policy_cuda_moe_down_quant_path_preferred(
    const BnBackendRuntimePolicy *policy,
    int routed_asymmetric_kquant,
    int down_type,
    int hidden_dim,
    int n_experts,
    int k) {
    return routed_asymmetric_kquant &&
           bn_backend_quant_moe_down_kquant_f32_cache_supported(down_type) &&
           hidden_dim <= 1024 &&
           bn_gpu_policy_moe_route_expanded_topk(n_experts, k) &&
           !gpu_policy_moe_down_kquant_f32_cache_requested(policy);
}

int bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
    const BnBackendRuntimePolicy *policy,
    int routed_asymmetric_kquant,
    int down_type,
    int has_f32_data,
    int prefer_quant_down,
    int dim,
    int hidden_dim,
    int n_experts,
    int k) {
    return bn_backend_quant_moe_down_kquant_f32_cache_supported(down_type) &&
           has_f32_data && !prefer_quant_down &&
           !(routed_asymmetric_kquant &&
             bn_gpu_policy_moe_route_all_active_two_large_hidden(n_experts,
                                                                 k,
                                                                 hidden_dim) &&
             dim <= 2048) &&
           !gpu_policy_moe_down_kquant_f32_cache_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_4row_enabled(
    const BnBackendRuntimePolicy *policy, int hidden_dim) {
    return hidden_dim <= 1024 &&
           !gpu_policy_moe_down_4row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_8row_enabled(
    const BnBackendRuntimePolicy *policy, int hidden_dim) {
    return hidden_dim <= 1024 &&
           !gpu_policy_moe_down_8row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_halfwarp_enabled(
    const BnBackendRuntimePolicy *policy,
    int down_type,
    int prefer_quant_down,
    int n_experts,
    int k) {
    return bn_backend_quant_moe_down_kquant_f32_cache_supported(down_type) &&
           bn_gpu_policy_moe_route_expanded_topk(n_experts, k) &&
           (prefer_quant_down ||
            gpu_policy_moe_down_halfwarp_requested(policy)) &&
           !gpu_policy_moe_down_halfwarp_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_split4_enabled(
    const BnBackendRuntimePolicy *policy,
    int down_type,
    int use_halfwarp,
    int n_experts,
    int k) {
    return !use_halfwarp &&
           bn_backend_quant_moe_down_kquant_f32_cache_supported(down_type) &&
           bn_gpu_policy_moe_route_expanded_topk(n_experts, k) &&
           gpu_policy_moe_down_split4_requested(policy) &&
           !gpu_policy_moe_down_split4_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_scatter_enabled(
    const BnBackendRuntimePolicy *policy,
    int down_type,
    int use_halfwarp,
    int use_split4) {
    return !use_halfwarp && !use_split4 &&
           bn_backend_quant_moe_down_kquant_f32_cache_supported(down_type) &&
           !gpu_policy_moe_down_scatter_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_scatter_16row_enabled(
    const BnBackendRuntimePolicy *policy,
    int use_scatter,
    int hidden_dim) {
    return use_scatter && hidden_dim <= 768 &&
           gpu_policy_moe_down_scatter_16row_requested(policy);
}

int bn_gpu_policy_cuda_moe_down_float_path_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_moe_down_float_path_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_pair_path_enabled(
    const BnBackendRuntimePolicy *policy,
    int f32_down_default,
    int pair_down_f32_layer,
    int all_active_two_disable_pair_down) {
    return !f32_down_default && !pair_down_f32_layer &&
           !all_active_two_disable_pair_down &&
           !gpu_policy_moe_down_pair_path_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_prefers_f32_cache(
    const BnBackendRuntimePolicy *policy,
    int has_f32_data,
    int hidden_dim,
    int all_active_two_kquant,
    int all_active_two_f32_down) {
    return has_f32_data && hidden_dim >= 4096 &&
           (!all_active_two_kquant || all_active_two_f32_down) &&
           !gpu_policy_moe_down_kquant_f32_cache_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_f32_pair2_enabled(const BnBackendRuntimePolicy *policy, int n_experts,
                                                      int k) {
    return bn_gpu_policy_moe_route_all_active_two(n_experts, k) &&
           !gpu_policy_moe_down_f32_pair2_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_f32_pair2_4row_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_moe_down_f32_pair2_4row_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_kquant) {
    return all_active_two_kquant &&
           gpu_policy_all_active_two_kquant_moe_down_accum_requested(policy) &&
           !gpu_policy_all_active_two_kquant_moe_down_accum_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_kquant) {
    return all_active_two_kquant &&
           !gpu_policy_all_active_two_kquant_moe_down_pair4_sum_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_prepared_native_quant_4row_sum_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_kquant,
    int k,
    int hidden_dim) {
    return !all_active_two_kquant && k <= 8 && hidden_dim <= 1024 &&
           !gpu_policy_moe_down_prepared_native_quant_4row_sum_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_prepared_native_quant_8row_sum_enabled(
    const BnBackendRuntimePolicy *policy,
    int prepared_native_quant_4row_sum,
    int hidden_dim) {
    return prepared_native_quant_4row_sum && hidden_dim <= 1024 &&
           gpu_policy_moe_down_prepared_native_quant_8row_sum_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_fixed_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_kquant) {
    return all_active_two_kquant &&
           !gpu_policy_all_active_two_kquant_moe_down_fixed_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_resid_rmsnorm_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_moe_down_resid_rmsnorm_fuse_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_prepared_native_quant_shape_2048_768_enabled(
    const BnBackendRuntimePolicy *policy,
    int dim,
    int hidden_dim,
    int k) {
    return dim == 2048 && hidden_dim == 768 && k == 8 &&
           !gpu_policy_moe_down_prepared_native_quant_shape_2048x768_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_accum_4row_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_all_active_two_kquant_moe_down_accum_4row_requested(policy);
}

int bn_gpu_policy_cuda_moe_down_prepared_pair_4row_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_moe_down_prepared_pair_4row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_f32_cache_enabled(
    const BnBackendRuntimePolicy *policy,
    int has_f32_data,
    int all_active_two_disable_f32_cache) {
    return has_f32_data && !all_active_two_disable_f32_cache &&
           !gpu_policy_moe_down_kquant_f32_cache_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_f16_cache_enabled(const BnBackendRuntimePolicy *policy, int has_f16_data) {
    return has_f16_data &&
           gpu_policy_moe_down_f16_cache_requested(policy) &&
           !gpu_policy_moe_down_f16_cache_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_aux_f32_cache_enabled(const BnBackendRuntimePolicy *policy, int has_f32_data) {
    return has_f32_data &&
           !gpu_policy_moe_down_small_expert_f32_cache_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(const BnBackendRuntimePolicy *policy, int n_experts,
                                                       int k,
                                                       int hidden_dim) {
    return bn_gpu_policy_moe_route_all_active_two_large_hidden(n_experts,
                                                               k,
                                                               hidden_dim) &&
           gpu_policy_moe_down_prepared_pair8_requested(policy) &&
           !gpu_policy_moe_down_prepared_pair8_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim) {
    return hidden_dim <= 1024 &&
           gpu_policy_moe_down_prepared_8row_requested(policy) &&
           !gpu_policy_moe_down_prepared_8row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(const BnBackendRuntimePolicy *policy, int n_tokens,
                                                       int dim,
                                                       int allow_small_dim) {
    return bn_gpu_policy_kquant_dot_enabled(policy) &&
           (n_tokens <= 1 || (allow_small_dim && dim <= 2048) ||
            gpu_policy_moe_gateup_prepared_dot_requested(policy));
}

int bn_gpu_policy_cuda_moe_gateup_prepared_8row_enabled(
    const BnBackendRuntimePolicy *policy, int dim) {
    return dim <= 2048 &&
           !gpu_policy_moe_gateup_prepared_8row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_gateup_prepared_split_enabled(
                                                         const BnBackendRuntimePolicy *policy,
                                                         int dim,
                                                         int n_experts) {
    return dim <= 2048 &&
           bn_gpu_policy_moe_route_expanded_topk(n_experts, 0) &&
           gpu_policy_moe_gateup_prepared_split_requested(policy) &&
           !gpu_policy_moe_gateup_prepared_split_disabled(policy);
}

int bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy,
    int dim,
    int all_active_two_kquant) {
    return (dim % BN_QK_K) == 0 &&
           all_active_two_kquant &&
           !bn_gpu_policy_all_active_two_kquant_moe_dot_prepared_input_default_disabled(policy) &&
           !bn_gpu_policy_all_active_two_kquant_route_dot_prepared_input_default_disabled(policy) &&
           !cuda_moe_route_dot_prepared_input_disabled(policy);
}

int bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy,
    int dim,
    int all_active_two_kquant,
    int uses_reference_silu) {
    return (dim % 32) == 0 &&
           all_active_two_kquant &&
           !uses_reference_silu &&
           (bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled(policy) ||
            cuda_moe_route_block_prepared_input_requested(policy)) &&
           !gpu_policy_moe_dot_input_forced(policy) &&
           !gpu_policy_all_active_two_moe_dot_input_forced(policy) &&
           !cuda_moe_route_block_prepared_input_disabled(policy);
}

int bn_gpu_policy_cuda_moe_router_fused_topk_enabled(
    const BnBackendRuntimePolicy *policy, int n_experts, int route_block) {
    return n_experts <= 256 &&
           !route_block &&
           gpu_policy_moe_router_fused_topk_requested(policy) &&
           !gpu_policy_moe_router_fused_topk_disabled(policy);
}

int bn_gpu_policy_cuda_moe_router_warp_disabled(
    const BnBackendRuntimePolicy *policy, int route_block) {
    return route_block || gpu_policy_moe_router_warp_disabled(policy);
}

int bn_gpu_policy_cuda_moe_router_4warp_enabled(
    const BnBackendRuntimePolicy *policy, int dim) {
    return dim >= 2048 &&
           !gpu_policy_moe_router_4warp_disabled(policy);
}

int bn_gpu_policy_cuda_moe_router_2warp_enabled(
    const BnBackendRuntimePolicy *policy, int dim) {
    return dim >= 2048 &&
           !gpu_policy_moe_router_2warp_disabled(policy);
}

int bn_gpu_policy_cuda_moe_router_warp_topk_enabled(
    const BnBackendRuntimePolicy *policy, int n_experts) {
    return n_experts <= 256 &&
           !gpu_policy_moe_router_warp_topk_disabled(policy);
}

int bn_gpu_policy_cuda_moe_block_prepared_batch_enabled(
    const BnBackendRuntimePolicy *policy, int routed_native_quant) {
    return routed_native_quant &&
           !gpu_policy_moe_block_prepared_batch_disabled(policy);
}

int bn_gpu_policy_cuda_moe_block_prepared_decode_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_moe_block_prepared_decode_disabled(policy);
}

int bn_gpu_policy_cuda_moe_gateup_block_2row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim) {
    return hidden_dim <= 1024 &&
           !gpu_policy_moe_gateup_block_2row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_down_block_4row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim) {
    return hidden_dim <= 1024 &&
           gpu_policy_moe_down_block_4row_requested(policy);
}

int bn_gpu_policy_cuda_moe_down_block_2row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim) {
    return hidden_dim <= 1024 &&
           !gpu_policy_moe_down_block_2row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_all_active_two_fast_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_graph_kquant) {
    return !gpu_policy_moe_all_active_two_fast_disabled(policy) &&
           (!all_active_two_graph_kquant ||
            bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled(
                policy));
}

int bn_gpu_policy_cuda_moe_prepared_dot_enabled(
    const BnBackendRuntimePolicy *policy,
    int use_all_active_two_prepared_default,
    int fast_prepared_gateup,
    int all_active_two_kquant,
    int hidden_dim,
    int dim) {
    return (use_all_active_two_prepared_default ||
            fast_prepared_gateup ||
            (all_active_two_kquant &&
             gpu_policy_all_active_two_moe_dot_input_forced(policy)) ||
            (hidden_dim > 2048 && dim > 2048) ||
            gpu_policy_moe_dot_input_forced(policy)) &&
           !gpu_policy_moe_prepared_dot_disabled(policy);
}

int bn_gpu_policy_cuda_moe_internal_profile_enabled(
    const BnBackendRuntimePolicy *policy, int profile) {
    return profile && gpu_policy_moe_internal_profile_requested(policy);
}

int bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(
    const BnBackendRuntimePolicy *policy,
    int prepared_dot_input,
    int all_active_two_fast_enabled) {
    return prepared_dot_input &&
           all_active_two_fast_enabled &&
           !gpu_policy_moe_all_active_two_fixed_prepared_disabled(policy) &&
           !gpu_policy_moe_gateup_prepared_4row_disabled(policy);
}

int bn_gpu_policy_cuda_moe_gateup_prepared_4row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_moe_gateup_prepared_4row_disabled(policy);
}

static int gpu_policy_decode_logits_cache_enabled(
    const BnGPUBackend *gpu, int gpu_logits_need_cpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_LOGITS_CACHE") &&
           !gpu_logits_need_cpu;
}

int bn_gpu_policy_decode_logits_cache_enabled(
    const BnGPUBackend *gpu, int gpu_logits_need_cpu) {
    return gpu_policy_decode_logits_cache_enabled(gpu, gpu_logits_need_cpu);
}

static int gpu_policy_moe_decode_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_MOE_DECODE_CACHE");
}

int bn_gpu_policy_moe_decode_cache_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_moe_decode_cache_enabled(gpu);
}

static int gpu_policy_moe_decode_cache_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_DECODE_CACHE");
}

int bn_gpu_policy_moe_decode_cache_disabled(const BnGPUBackend *gpu) {
    return gpu_policy_moe_decode_cache_disabled(gpu);
}

static int gpu_policy_decode_cache_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_DECODE_CACHE");
}

int bn_gpu_policy_decode_cache_disabled(const BnGPUBackend *gpu) {
    return gpu_policy_decode_cache_disabled(gpu);
}

static int gpu_policy_native_quant_decode_cache_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_DECODE_CACHE",
        "BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE");
}

int bn_gpu_policy_native_quant_decode_cache_disabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_native_quant_decode_cache_disabled(policy);
}

static int gpu_policy_logits_argmax_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_LOGITS_ARGMAX");
}

int bn_gpu_policy_logits_argmax_disabled(const BnGPUBackend *gpu) {
    return gpu_policy_logits_argmax_disabled(gpu);
}

static int gpu_policy_dense_logits_argmax_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
}

int bn_gpu_policy_dense_logits_argmax_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_dense_logits_argmax_enabled(gpu);
}

static int gpu_policy_moe_logits_mmvq_argmax_enabled(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu,
                               "BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX");
}

int bn_gpu_policy_moe_logits_mmvq_argmax_enabled(const BnGPUBackend *gpu) {
    return gpu_policy_moe_logits_mmvq_argmax_enabled(gpu);
}

static int gpu_policy_moe_logits_mmvq_argmax_disabled(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu,
                               "BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
}

static int gpu_policy_moe_logits_mmvq_1warp8_1536_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536");
}

static int gpu_policy_moe_logits_mmvq_1warp16_1536_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_1WARP16_1536");
}

static int gpu_policy_moe_logits_mmvq_1warp8_1536_unroll_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536_UNROLL");
}

static int gpu_policy_argmax_fast_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DISABLE_ARGMAX_FAST");
}

static int gpu_policy_optimistic_argmax_penalty_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY");
}

static int gpu_policy_legacy_block_matvec4_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_LEGACY_BLOCK_MATVEC4",
        "BN_CUDA_ENABLE_Q5_MATVEC4");
}

static int gpu_policy_legacy_block_warp_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_LEGACY_BLOCK_WARP",
        "BN_CUDA_ENABLE_Q5_WARP");
}

static int gpu_policy_deinterleaved_kquant_pair_matvec_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_PAIR_MATVEC",
        "BN_CUDA_ENABLE_Q5K_DEINT_PAIR_MATVEC");
}

static int gpu_policy_deinterleaved_kquant_4warp_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_4WARP",
        "BN_CUDA_DISABLE_Q5K_4WARP");
}

static int gpu_policy_deinterleaved_kquant_split_4warp_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_SPLIT_4WARP",
        "BN_CUDA_ENABLE_Q5K_SPLIT_4WARP");
}

static int gpu_policy_deinterleaved_kquant_gateup_2warp_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_GATEUP_2WARP",
        "BN_CUDA_DISABLE_Q5K_GATEUP_2WARP");
}

static int gpu_policy_symmetric_kquant_dot_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_SYMMETRIC_KQUANT_DOT",
        "BN_CUDA_DISABLE_Q4K_DOT");
}

static int gpu_policy_deinterleaved_kquant_dot_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_DOT",
        "BN_CUDA_DISABLE_Q5K_DOT");
}

static int gpu_policy_asymmetric_kquant_4warp_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP",
        "BN_CUDA_DISABLE_Q4K_4WARP");
}

static int gpu_policy_asymmetric_kquant_4warp_1536x8960_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_1536_8960",
        "BN_CUDA_DISABLE_Q4K_4WARP_1536_8960");
}

static int gpu_policy_asymmetric_kquant_4warp_2560x9728_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_2560_9728",
        "BN_CUDA_DISABLE_Q4K_4WARP_2560_9728");
}

static int gpu_policy_asymmetric_kquant_out_resid_rmsnorm_fuse_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_OUT_RESID_RMSNORM_FUSE",
        "BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE");
}

static int gpu_policy_asymmetric_kquant_qkv_mixed_fuse_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_QKV_MIXED_FUSE",
        "BN_CUDA_ENABLE_Q4K_QKV_MIXED_FUSE");
}

static int gpu_policy_asymmetric_kquant_split_k_rope_cache_fuse_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE",
        "BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE");
}

static int gpu_policy_asymmetric_kquant_split_k_rope_cache_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE",
        "BN_CUDA_DISABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE");
}

static int gpu_policy_asymmetric_kquant_split_qk_rope_cache_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_QK_ROPE_CACHE_FUSE",
        "BN_CUDA_DISABLE_Q4K_SPLIT_QK_ROPE_CACHE_FUSE");
}

static int gpu_policy_asymmetric_kquant_split_4warp_2048_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_4WARP_2048",
        "BN_CUDA_DISABLE_Q4K_SPLIT_4WARP_2048");
}

static int gpu_policy_asymmetric_kquant_split_5warp_2560_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_5WARP_2560",
        "BN_CUDA_DISABLE_Q4K_SPLIT_5WARP_2560");
}

static int gpu_policy_asymmetric_kquant_split_value_1792x1536_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE_1792",
        "BN_CUDA_ENABLE_Q4K_SPLIT_VALUE_FUSE_1792");
}

static int gpu_policy_asymmetric_kquant_split_value_fuse_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE",
        "BN_CUDA_DISABLE_Q4K_SPLIT_VALUE_FUSE");
}

static int gpu_policy_kquant_gateup_fast_path_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_Q8_1_FAST",
        "BN_CUDA_DISABLE_Q4K_GATEUP_Q8_1_FAST");
}

static int gpu_policy_asymmetric_kquant_gateup_qwarp4_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_QWARP4",
        "BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4");
}

static int gpu_policy_asymmetric_kquant_gateup_5warp_2560_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_5WARP_2560",
        "BN_CUDA_DISABLE_Q4K_GATEUP_5WARP_2560");
}

static int gpu_policy_asymmetric_kquant_gateup_2warp_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_2WARP",
        "BN_CUDA_DISABLE_Q4K_GATEUP_2WARP");
}

static int gpu_policy_native_quant_warp_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_WARP",
        "BN_CUDA_DISABLE_Q8_WARP");
}

static int gpu_policy_native_quant_ssm_matvec_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_SSM_MATVEC",
        "BN_CUDA_DISABLE_Q8_0_SSM_MATVEC");
}

static int gpu_policy_f16_native_quant_ssm_matvec_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_F16_NATIVE_QUANT_SSM_MATVEC",
        "BN_CUDA_ENABLE_F16_Q8_0_SSM_MATVEC");
}

static int gpu_policy_f16_native_quant_ssm_matvec_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_F16_NATIVE_QUANT_SSM_MATVEC",
        "BN_CUDA_DISABLE_F16_Q8_0_SSM_MATVEC");
}

static int gpu_policy_f16_native_quant_matvec_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_F16_NATIVE_QUANT_MATVEC",
        "BN_CUDA_ENABLE_F16_Q8_0_MATVEC");
}

static int gpu_policy_f16_native_quant_matvec_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATVEC",
        "BN_CUDA_DISABLE_F16_Q8_0_MATVEC");
}

static int gpu_policy_f16_packed_kquant_matvec_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_F16_PACKED_KQUANT_MATVEC",
        "BN_CUDA_ENABLE_F16_Q5K_MATVEC");
}

static int gpu_policy_symmetric_kquant_pair_matvec_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_SYMMETRIC_KQUANT_PAIR_MATVEC",
        "BN_CUDA_DISABLE_Q4K_PAIR_MATVEC");
}

static int gpu_policy_kquant_dot_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_DOT",
        "BN_CUDA_DISABLE_Q4K_Q8K_DOT");
}

static int gpu_policy_kquant_dot_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_NATIVE_DOT",
        "BN_CUDA_ENABLE_Q4K_Q8K_DOT");
}

static int gpu_policy_kquant_matvec4_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_MATVEC4",
        "BN_CUDA_DISABLE_Q4K_Q8K_MATVEC4");
}

static int gpu_policy_asymmetric_kquant_matmul8_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_MATMUL8",
        "BN_CUDA_ENABLE_Q4K_MATMUL8");
}

static int gpu_policy_asymmetric_kquant_sharedx_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH",
        "BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH");
}

static int gpu_policy_asymmetric_kquant_batch_sharedx_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH",
        "BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH");
}

static int gpu_policy_down_kquant_dot_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_DISABLE_DOWN_KQUANT_DOT",
                                         "BN_CUDA_DISABLE_Q6K_DOT");
}

static int gpu_policy_down_kquant_dot_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_ENABLE_DOWN_KQUANT_DOT",
                                         "BN_CUDA_ENABLE_Q6K_DOT");
}

static int gpu_policy_down_kquant_warp_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_ENABLE_DOWN_KQUANT_WARP",
                                         "BN_CUDA_ENABLE_Q6K_WARP");
}

static int gpu_policy_asymmetric_kquant_pair_matvec_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_MIXED_KQUANT_PAIR_MATVEC",
        "BN_CUDA_ENABLE_Q6K_Q4K_PAIR_MATVEC");
}

static int gpu_policy_asymmetric_kquant_pair_matvec_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_MIXED_KQUANT_PAIR_MATVEC",
        "BN_CUDA_DISABLE_Q6K_Q4K_PAIR_MATVEC");
}

static int gpu_policy_down_kquant_prepared_dot_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT",
        "BN_CUDA_ENABLE_Q6K_Q8_1_DOT");
}

static int gpu_policy_down_kquant_prepared_dot_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_PREPARED_DOT",
        "BN_CUDA_DISABLE_Q6K_Q8_1_DOT");
}

static int gpu_policy_down_kquant_prepared_dot_all_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT_ALL",
        "BN_CUDA_ENABLE_Q6K_Q8_1_ALL");
}

static int gpu_policy_down_kquant_mmvq_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ",
                                         "BN_CUDA_DISABLE_Q6K_MMVQ");
}

static int gpu_policy_down_kquant_mmvq_512x2048_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_512_2048",
        "BN_CUDA_DISABLE_Q6K_MMVQ_512_2048");
}

static int gpu_policy_down_kquant_mmvq_1536x8960_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_1536_8960",
        "BN_CUDA_DISABLE_Q6K_MMVQ_1536_8960");
}

static int gpu_policy_down_kquant_mmvq_2560x9728_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2560_9728",
        "BN_CUDA_DISABLE_Q6K_MMVQ_2560_9728");
}

static int gpu_policy_down_kquant_mmvq_logits_1536_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_LOGITS_1536",
        "BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536");
}

static int gpu_policy_down_kquant_mmvq_2warp_logits_small_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_LOGITS_SMALL",
        "BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_LOGITS_SMALL");
}

static int gpu_policy_down_kquant_mmvq_2warp_1536_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_1536",
        "BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_1536");
}

static int gpu_policy_down_kquant_resid_rmsnorm_fuse_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_RESID_RMSNORM_FUSE",
        "BN_CUDA_ENABLE_Q6K_DOWN_RESID_RMSNORM_FUSE");
}

static int gpu_policy_f16_down_kquant_matvec_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_F16_DOWN_KQUANT_MATVEC",
        "BN_CUDA_ENABLE_F16_Q6K_MATVEC");
}

static int gpu_policy_f16_down_kquant_matvec_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_F16_DOWN_KQUANT_MATVEC",
        "BN_CUDA_DISABLE_F16_Q6K_MATVEC");
}

static int gpu_policy_down_kquant_matmul8_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_ENABLE_DOWN_KQUANT_MATMUL8",
                                         "BN_CUDA_ENABLE_Q6K_MATMUL8");
}

static int gpu_policy_down_kquant_matmul4_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_DISABLE_DOWN_KQUANT_MATMUL4",
                                         "BN_CUDA_DISABLE_Q6K_MATMUL4");
}

static int gpu_policy_down_kquant_matvec4_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC4",
                                         "BN_CUDA_DISABLE_Q6K_MATVEC4");
}

static int gpu_policy_down_kquant_batch_warp_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_DOWN_KQUANT_BATCH_WARP",
        "BN_CUDA_ENABLE_Q6K_BATCH_WARP");
}

static int gpu_policy_cuda_fuse_bias_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DISABLE_FUSE_BIAS");
}

static int gpu_policy_cuda_rope_flash_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_ROPE_FLASH_FUSE");
}

static int gpu_policy_cuda_bias_rope_flash_fuse_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE");
}

static int gpu_policy_cuda_qk_norm_rope_flash_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_QK_NORM_ROPE_FLASH_FUSE");
}

static int gpu_policy_cuda_qk_norm_rope_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_QK_NORM_ROPE_FUSE");
}

static int gpu_policy_cuda_weighted_add_sigmoid_resid_rmsnorm_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy,
        "BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_RMSNORM_FUSE");
}

static int gpu_policy_cuda_weighted_add_sigmoid_resid_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_FUSE");
}

static int gpu_policy_cuda_readback_debug_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DEBUG_READBACK");
}

static int gpu_policy_cuda_cublas_cache_debug_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DEBUG_CUBLAS_CACHE");
}

static int gpu_policy_cuda_nan_verbose_debug_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DEBUG_NAN_VERBOSE");
}

static int gpu_policy_cuda_stream_exec_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DISABLE_STREAM_EXEC");
}

static int gpu_policy_cuda_profile_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy, "BN_CUDA_PROFILE");
}

static int gpu_policy_cuda_wall_profile_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_PROFILE_WALL");
}

static int gpu_policy_cuda_profile_shapes_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_PROFILE_SHAPES");
}

static int gpu_policy_cuda_exec_fail_debug_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DEBUG_EXEC_FAIL");
}

static int gpu_policy_cuda_sync_each_op_debug_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DEBUG_SYNC_EACH_OP");
}

static int gpu_policy_cuda_nan_debug_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy, "BN_CUDA_DEBUG_NAN");
}

static int gpu_policy_cuda_dump_ops_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy, "BN_CUDA_DUMP_OPS");
}

static int gpu_policy_cuda_dump_ops_every_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_DUMP_OPS_EVERY");
}

static int gpu_policy_cuda_prefill_moe_layer_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_PREFILL_MOE_LAYER");
}

static int gpu_policy_cuda_prefill_dense_layer_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_PREFILL_DENSE_LAYER");
}

static int gpu_policy_cuda_prefill_dense_debug_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DEBUG_PREFILL_DENSE_LAYER");
}

static int gpu_policy_cuda_prefill_dense_profile_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_PREFILL_DENSE_PROFILE");
}

static int gpu_policy_cuda_prefill_ssm_layer_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
}

static int gpu_policy_cuda_prefill_fused_asym_kquant_gateup_batch_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(
        policy,
        "BN_CUDA_DISABLE_PREFILL_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH",
        "BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH");
}

static int gpu_policy_cuda_prefill_ssm_fused_asym_kquant_gateup_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(
        policy,
        "BN_CUDA_ENABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH",
        "BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH");
}

static int gpu_policy_cuda_prefill_ssm_fused_asym_kquant_gateup_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(
        policy,
        "BN_CUDA_DISABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH",
        "BN_CUDA_DISABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH");
}

static int gpu_policy_cuda_prefill_ssm_profile_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy, "BN_CUDA_SSM_PROFILE");
}

static int gpu_policy_cuda_prefill_ssm_stacked_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SSM_STACKED_PREFILL");
}

static int gpu_policy_cuda_prefill_ssm_stream_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SSM_STREAM_PREFILL");
}

static int gpu_policy_cuda_prefill_ssm_input_alias_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SSM_PREFILL_INPUT_ALIAS");
}

static int gpu_policy_cuda_prefill_ssm_f32_ab_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SSM_F32_AB_PREFILL");
}

static int gpu_policy_cuda_prefill_ssm_scan_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SSM_PREFILL_SCAN");
}

static int gpu_policy_cuda_prefill_ssm_delta_128_warp_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SSM_DELTA_128_WARP");
}

static int gpu_policy_cuda_prefill_ssm_ffn_profile_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_SSM_FFN_PROFILE");
}

static int gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_SSM_FFN_GATEUP_F16_OUT");
}

static int gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SSM_FFN_GATEUP_F16_OUT");
}

static int gpu_policy_backend_opt_in_fused_gateup_requested(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_ENABLE_PACKED_KQUANT_FUSED_GATEUP",
        "BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
}

static int gpu_policy_shared_kquant_dot_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_SHARED_KQUANT_NATIVE_DOT",
        "BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");
}

static int gpu_policy_shared_expert_gate_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_SHARED_EXPERT_GATE");
}

int bn_gpu_policy_moe_logits_mmvq_argmax_disabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_moe_logits_mmvq_argmax_disabled(gpu);
}

int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(
    const BnBackendRuntimePolicy *policy, int rows, int cols) {
    return !bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX") &&
           rows >= 50000 &&
           (cols == 1536 ||
            bn_backend_runtime_policy_enabled(
                policy, "BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX"));
}

int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_enabled(
    const BnBackendRuntimePolicy *policy, int use_mmvq, int rows, int cols) {
    return use_mmvq && rows == 151936 && cols == 1536 &&
           !gpu_policy_moe_logits_mmvq_1warp8_1536_disabled(policy);
}

int bn_gpu_policy_cuda_moe_logits_mmvq_1warp16_1536_enabled(
    const BnBackendRuntimePolicy *policy, int use_1warp8) {
    return use_1warp8 &&
           gpu_policy_moe_logits_mmvq_1warp16_1536_requested(policy);
}

int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_unroll_enabled(
    const BnBackendRuntimePolicy *policy, int use_1warp8, int use_1warp16) {
    return use_1warp8 && !use_1warp16 &&
           !gpu_policy_moe_logits_mmvq_1warp8_1536_unroll_disabled(policy);
}

int bn_gpu_policy_cuda_argmax_fast_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_argmax_fast_disabled(policy);
}

int bn_gpu_policy_cuda_optimistic_argmax_penalty_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_optimistic_argmax_penalty_requested(policy);
}

int bn_gpu_policy_cuda_legacy_block_matvec4_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_legacy_block_matvec4_requested(policy);
}

int bn_gpu_policy_cuda_legacy_block_warp_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_legacy_block_warp_requested(policy);
}

int bn_gpu_policy_cuda_deinterleaved_kquant_pair_matvec_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_deinterleaved_kquant_pair_matvec_requested(policy);
}

int bn_gpu_policy_cuda_deinterleaved_kquant_4warp_enabled(const BnBackendRuntimePolicy *policy, int cols) {
    return cols <= 8192 && !gpu_policy_deinterleaved_kquant_4warp_disabled(policy);
}

int bn_gpu_policy_cuda_deinterleaved_kquant_split_4warp_enabled(const BnBackendRuntimePolicy *policy, int cols) {
    return bn_gpu_policy_cuda_deinterleaved_kquant_4warp_enabled(policy, cols) &&
           gpu_policy_deinterleaved_kquant_split_4warp_requested(policy);
}

int bn_gpu_policy_cuda_deinterleaved_kquant_gateup_2warp_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_deinterleaved_kquant_gateup_2warp_disabled(policy);
}

int bn_gpu_policy_cuda_symmetric_kquant_dot_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_symmetric_kquant_dot_disabled(policy);
}

int bn_gpu_policy_cuda_deinterleaved_kquant_dot_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_deinterleaved_kquant_dot_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_4warp_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_asymmetric_kquant_4warp_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_4warp_shape_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                             int cols) {
    return cols <= 8192 ||
           (rows == 1536 && cols == 8960 &&
            !gpu_policy_asymmetric_kquant_4warp_1536x8960_disabled(policy)) ||
           (rows == 2560 && cols == 9728 &&
            !gpu_policy_asymmetric_kquant_4warp_2560x9728_disabled(policy));
}

int bn_gpu_policy_cuda_asymmetric_kquant_out_residual_rmsnorm_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_asymmetric_kquant_out_resid_rmsnorm_fuse_requested(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_qkv_mixed_fuse_enabled(
    const BnBackendRuntimePolicy *policy, int tensor_type) {
    return !bn_backend_quant_supports_asymmetric_kquant_prepared_input_split(
               tensor_type) ||
           gpu_policy_asymmetric_kquant_qkv_mixed_fuse_requested(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_split_k_rope_cache_fuse_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_asymmetric_kquant_split_k_rope_cache_fuse_requested(policy) &&
           !gpu_policy_asymmetric_kquant_split_k_rope_cache_fuse_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_split_qk_rope_cache_fuse_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_asymmetric_kquant_split_qk_rope_cache_fuse_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_split_4warp_enabled(const BnBackendRuntimePolicy *policy, int cols) {
    return cols == 2048 &&
           !gpu_policy_asymmetric_kquant_split_4warp_2048_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_split_5warp_enabled(const BnBackendRuntimePolicy *policy, int cols) {
    return cols == 2560 &&
           !gpu_policy_asymmetric_kquant_split_5warp_2560_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_split_value_rows(const BnBackendRuntimePolicy *policy, int total_rows,
                                                         int cols) {
    if (total_rows == 4608 && cols == 2048)
        return 512;
    if (total_rows == 2304 && cols == 2048)
        return 256;
    if (total_rows == 1792 && cols == 1536 &&
        gpu_policy_asymmetric_kquant_split_value_1792x1536_requested(policy))
        return 256;
    return 0;
}

int bn_gpu_policy_cuda_asymmetric_kquant_split_value_fuse_enabled(
    const BnBackendRuntimePolicy *policy, int value_rows) {
    return value_rows > 0 &&
           !gpu_policy_asymmetric_kquant_split_value_fuse_disabled(policy);
}

int bn_gpu_policy_kquant_gateup_prepared_path_enabled(
    const BnBackendRuntimePolicy *policy, int uses_prepared_kquant_input) {
    return uses_prepared_kquant_input ||
           gpu_policy_kquant_gateup_fast_path_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_gateup_qwarp4_enabled(const BnBackendRuntimePolicy *policy, int cols) {
    return cols <= 4096 &&
           !gpu_policy_asymmetric_kquant_gateup_qwarp4_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_gateup_5warp_enabled(
    const BnBackendRuntimePolicy *policy,
    int enable_asymmetric_kquant_4warp,
    int cols) {
    return enable_asymmetric_kquant_4warp && cols == 2560 &&
           !gpu_policy_asymmetric_kquant_gateup_5warp_2560_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_gateup_2warp_enabled(
    const BnBackendRuntimePolicy *policy,
    int enable_asymmetric_kquant_4warp,
    int cols) {
    return enable_asymmetric_kquant_4warp && cols <= 5120 &&
           !gpu_policy_asymmetric_kquant_gateup_2warp_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_gateup_4warp_enabled(
    int enable_asymmetric_kquant_4warp,
    int cols) {
    return enable_asymmetric_kquant_4warp && cols <= 8192;
}

int bn_gpu_policy_cuda_native_quant_warp_disabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_native_quant_warp_disabled(policy);
}

int bn_gpu_policy_cuda_native_quant_ssm_matvec_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_native_quant_ssm_matvec_disabled(policy);
}

int bn_gpu_policy_cuda_native_quant_ssm_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !cuda_native_quant_ssm_prepared_input_disabled(policy);
}

int bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy,
    int type_a,
    int type_b,
    int cols) {
    return cuda_native_quant_mixed_prepared_input_requested(policy) &&
           (bn_backend_quant_supports_native_quant_prepared_input_matvec(
                type_a) ||
            bn_backend_quant_supports_native_quant_prepared_input_matvec(
                type_b)) &&
           (cols & 31) == 0;
}

int bn_gpu_policy_cuda_f16_native_quant_ssm_matvec_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_f16_native_quant_ssm_matvec_requested(policy) &&
           !gpu_policy_f16_native_quant_ssm_matvec_disabled(policy) &&
           !gpu_policy_f16_native_quant_matvec_disabled(policy);
}

int bn_gpu_policy_cuda_f16_native_quant_matvec_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_f16_native_quant_matvec_requested(policy) &&
           !gpu_policy_f16_native_quant_matvec_disabled(policy);
}

int bn_gpu_policy_cuda_f16_packed_kquant_matvec_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_f16_packed_kquant_matvec_requested(policy);
}

int bn_gpu_policy_cuda_symmetric_kquant_pair_matvec_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_symmetric_kquant_pair_matvec_disabled(policy);
}

int bn_gpu_policy_kquant_dot_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_kquant_dot_disabled(policy);
}

int bn_gpu_policy_kquant_dot_forced(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_kquant_dot_requested(policy);
}

int bn_gpu_policy_kquant_matvec4_enabled(const BnBackendRuntimePolicy *policy, int cols) {
    return cols >= 16384 &&
           !gpu_policy_kquant_matvec4_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_matmul8_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_asymmetric_kquant_matmul8_requested(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_sharedx_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_asymmetric_kquant_sharedx_disabled(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_batch_sharedx_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_asymmetric_kquant_batch_sharedx_requested(policy);
}

int bn_gpu_policy_cuda_down_kquant_dot_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_down_kquant_dot_disabled(policy);
}

int bn_gpu_policy_cuda_down_kquant_dot_forced(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_down_kquant_dot_requested(policy);
}

int bn_gpu_policy_cuda_down_kquant_warp_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_down_kquant_warp_requested(policy);
}

int bn_gpu_policy_cuda_asymmetric_kquant_pair_matvec_enabled(const BnBackendRuntimePolicy *policy, int cols) {
    return (cols < 5120 ||
            gpu_policy_asymmetric_kquant_pair_matvec_requested(policy)) &&
           !gpu_policy_asymmetric_kquant_pair_matvec_disabled(policy);
}

int bn_gpu_policy_cuda_down_kquant_prepared_dot_enabled(const BnBackendRuntimePolicy *policy, int is_logits_op) {
    return gpu_policy_down_kquant_prepared_dot_requested(policy) &&
           !gpu_policy_down_kquant_prepared_dot_disabled(policy) &&
           (is_logits_op ||
            gpu_policy_down_kquant_prepared_dot_all_requested(policy));
}

int bn_gpu_policy_cuda_down_kquant_mmvq_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                int cols,
                                                int is_logits_op,
                                                int uses_reference_kquant_matvec) {
    return !uses_reference_kquant_matvec &&
           !gpu_policy_down_kquant_mmvq_disabled(policy) &&
           ((cols >= 4096 && rows >= 5120) ||
            (cols >= 2048 && rows >= 50000) ||
            (rows == 512 && cols == 2048 &&
             !gpu_policy_down_kquant_mmvq_512x2048_disabled(policy)) ||
            (rows == 1536 && cols == 8960 &&
             !gpu_policy_down_kquant_mmvq_1536x8960_disabled(policy)) ||
            (rows == 2560 && cols == 9728 &&
             !gpu_policy_down_kquant_mmvq_2560x9728_disabled(policy)) ||
            (is_logits_op && cols == 1536 && rows >= 50000 &&
             !gpu_policy_down_kquant_mmvq_logits_1536_disabled(policy)));
}

int bn_gpu_policy_cuda_down_kquant_mmvq_2warp_logits_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                             int cols,
                                                             int is_logits_op) {
    return is_logits_op && cols <= 2560 && rows >= 50000 &&
           ((cols == 1536 &&
             !gpu_policy_down_kquant_mmvq_logits_1536_disabled(policy)) ||
            (cols > 1536 &&
             !gpu_policy_down_kquant_mmvq_2warp_logits_small_disabled(policy))) &&
           !gpu_policy_down_kquant_mmvq_2warp_1536_disabled(policy);
}

int bn_gpu_policy_cuda_down_kquant_residual_rmsnorm_fuse_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_down_kquant_resid_rmsnorm_fuse_requested(policy);
}

int bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                      int cols,
                                                      int uses_reference_kquant_matvec) {
    return !uses_reference_kquant_matvec &&
           (gpu_policy_f16_down_kquant_matvec_requested(policy) ||
            (!gpu_policy_f16_down_kquant_matvec_disabled(policy) &&
             rows <= 2048 && cols >= 8192));
}

int bn_gpu_policy_cuda_down_kquant_matmul8_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_down_kquant_matmul8_requested(policy);
}

int bn_gpu_policy_cuda_down_kquant_matmul4_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_down_kquant_matmul4_disabled(policy);
}

int bn_gpu_policy_cuda_down_kquant_matvec4_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_down_kquant_matvec4_disabled(policy);
}

int bn_gpu_policy_cuda_down_kquant_batch_warp_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_down_kquant_batch_warp_requested(policy);
}

int bn_gpu_policy_cuda_fuse_bias_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_fuse_bias_disabled(policy);
}

int bn_gpu_policy_cuda_rope_flash_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_rope_flash_fuse_disabled(policy);
}

int bn_gpu_policy_cuda_bias_rope_flash_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_bias_rope_flash_fuse_requested(policy);
}

int bn_gpu_policy_cuda_qk_norm_rope_flash_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_qk_norm_rope_flash_fuse_disabled(policy);
}

int bn_gpu_policy_cuda_qk_norm_rope_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_qk_norm_rope_fuse_disabled(policy);
}

int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_rmsnorm_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_weighted_add_sigmoid_resid_rmsnorm_fuse_disabled(
        policy);
}

int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_weighted_add_sigmoid_resid_fuse_disabled(policy);
}

int bn_gpu_policy_cuda_readback_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_readback_debug_requested(policy);
}

int bn_gpu_policy_cuda_cublas_cache_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_cublas_cache_debug_requested(policy);
}

int bn_gpu_policy_cuda_cublas_cache_reserve_mb_or_default(
    const BnBackendRuntimePolicy *policy, int default_mb) {
    return runtime_int_or_default(
        policy, "BN_CUDA_CUBLAS_CACHE_RESERVE_MB", default_mb);
}

int bn_gpu_policy_cuda_cublas_workspace_mb_or_default(
    const BnBackendRuntimePolicy *policy, int default_mb) {
    return runtime_int_or_default(
        policy, "BN_CUDA_CUBLAS_WORKSPACE_MB", default_mb);
}

int bn_gpu_policy_cuda_nan_verbose_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_nan_verbose_debug_requested(policy);
}

int bn_gpu_policy_cuda_stream_exec_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_stream_exec_disabled(policy);
}

int bn_gpu_policy_cuda_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_profile_requested(policy);
}

int bn_gpu_policy_cuda_wall_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_wall_profile_requested(policy);
}

int bn_gpu_policy_cuda_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PROFILE_EVERY", default_every);
}

int bn_gpu_policy_cuda_wall_profile_detail_limit_or_default(
    const BnBackendRuntimePolicy *policy, int default_limit) {
    return runtime_int_or_default(
        policy, "BN_CUDA_PROFILE_WALL_DETAIL", default_limit);
}

int bn_gpu_policy_cuda_wall_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PROFILE_WALL_EVERY", default_every);
}

int bn_gpu_policy_cuda_profile_shapes_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_profile_shapes_requested(policy);
}

const char *bn_gpu_policy_cuda_device_selector(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_get(policy, "BN_CUDA_DEVICE");
}

int bn_gpu_policy_cuda_exec_fail_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_exec_fail_debug_requested(policy);
}

int bn_gpu_policy_cuda_sync_each_op_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_sync_each_op_debug_requested(policy);
}

int bn_gpu_policy_cuda_nan_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_nan_debug_requested(policy);
}

int bn_gpu_policy_cuda_dump_ops_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_dump_ops_requested(policy);
}

int bn_gpu_policy_cuda_dump_ops_every_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_dump_ops_every_requested(policy);
}

int bn_gpu_policy_cuda_dump_ops_limit_or_default(
    const BnBackendRuntimePolicy *policy, int default_limit) {
    return runtime_int_or_default(policy, "BN_CUDA_DUMP_OPS_LIMIT",
                                  default_limit);
}

int bn_gpu_policy_cuda_prefill_moe_layer_disabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_moe_layer_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_dense_layer_disabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_dense_layer_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_dense_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_dense_debug_requested(policy);
}

int bn_gpu_policy_cuda_prefill_dense_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_dense_profile_requested(policy);
}

int bn_gpu_policy_cuda_prefill_dense_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PREFILL_DENSE_PROFILE_EVERY", default_every);
}

int bn_gpu_policy_cuda_prefill_ssm_layer_disabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_ssm_layer_disabled(policy);
}

int bn_gpu_policy_prefill_ssm_layer_disabled(
    const BnGPUBackend *gpu) {
    return gpu && gpu_policy_cuda_prefill_ssm_layer_disabled(
                      &gpu->runtime_policy);
}

int bn_gpu_policy_cuda_prefill_fused_asymmetric_kquant_gateup_batch_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_prefill_fused_asym_kquant_gateup_batch_disabled(
        policy);
}

int bn_gpu_policy_cuda_prefill_ssm_fused_asymmetric_kquant_gateup_batch_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_ssm_fused_asym_kquant_gateup_requested(
               policy) &&
           !gpu_policy_cuda_prefill_ssm_fused_asym_kquant_gateup_disabled(
               policy);
}

int bn_gpu_policy_cuda_prefill_ssm_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_ssm_profile_requested(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_stacked_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_prefill_ssm_stacked_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_stream_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_prefill_ssm_stream_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_input_alias_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_prefill_ssm_input_alias_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_f32_ab_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_prefill_ssm_f32_ab_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_scan_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_prefill_ssm_scan_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_delta_128_warp_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_prefill_ssm_delta_128_warp_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_ffn_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_ssm_ffn_profile_requested(policy);
}

int bn_gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_requested(policy) &&
           !gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_disabled(policy);
}

int bn_gpu_policy_backend_opt_in_fused_gateup_enabled(const BnBackendRuntimePolicy *policy) {
    return gpu_policy_backend_opt_in_fused_gateup_requested(policy);
}

int bn_gpu_policy_fused_gateup_silu_allowed(const BnGPUBackend *gpu,
                                            int tensor_type) {
    if (!bn_gpu_policy_fused_gateup_enabled(gpu))
        return 0;
    if (gpu_policy_backend_caps(gpu)->fused_gateup_requires_backend_opt_in &&
        bn_backend_quant_gpu_fused_gateup_requires_backend_opt_in(tensor_type) &&
        !bn_gpu_policy_backend_opt_in_fused_gateup_enabled(
            gpu ? &gpu->runtime_policy : NULL))
        return 0;
    return 1;
}

int bn_gpu_policy_shared_kquant_dot_enabled(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_shared_kquant_dot_disabled(policy);
}

int bn_gpu_policy_shared_expert_gate_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_shared_expert_gate_disabled(policy);
}

int bn_gpu_policy_cuda_prefill_attention_min_tokens_configured(
    const BnBackendRuntimePolicy *policy) {
    const char *value = bn_backend_runtime_policy_get(
        policy, "BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    return value && *value;
}

int bn_gpu_policy_prefill_attention_min_tokens_configured(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           bn_gpu_policy_cuda_prefill_attention_min_tokens_configured(
               &gpu->runtime_policy);
}

int bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
    const BnBackendRuntimePolicy *policy, int default_tokens) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PREFILL_ATTN_MIN_TOKENS", default_tokens);
}

int bn_gpu_policy_prefill_attention_min_tokens_or_default(
    const BnGPUBackend *gpu, int default_tokens) {
    return bn_gpu_backend_is_cuda(gpu)
        ? bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
              &gpu->runtime_policy, default_tokens)
        : default_tokens;
}

int bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
    const BnBackendRuntimePolicy *policy, int default_tokens) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", default_tokens);
}

int bn_gpu_policy_cuda_prefill_gemm_attention_enabled(
    const BnBackendRuntimePolicy *policy, int n_tokens, int max_tokens) {
    if (bn_backend_runtime_policy_enabled(
            policy, "BN_CUDA_DISABLE_PREFILL_GEMM_ATTN"))
        return 0;
    if (max_tokens > 0 && n_tokens > max_tokens)
        return 0;
    return bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_ENABLE_PREFILL_GEMM_ATTN") ||
           n_tokens >=
               bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
                   policy, 256);
}

int bn_gpu_policy_cuda_prefill_attention_wo_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_PREFILL_ATTN_WO");
}

int bn_gpu_policy_cuda_prefill_qkv_attention_wo_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO");
}

int bn_gpu_policy_cuda_prefill_batched_gemm_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM");
}

int bn_gpu_policy_cuda_prefill_gemm_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DEBUG_PREFILL_GEMM");
}

int bn_gpu_policy_prefill_dense_chain_enabled(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           !gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
}

int bn_gpu_policy_prefill_hybrid_chain_enabled(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           !gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN");
}

int bn_gpu_policy_cuda_prefill_attention_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_PREFILL_ATTN");
}

int bn_gpu_policy_prefill_attention_enabled(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           bn_gpu_policy_cuda_prefill_attention_enabled(
               &gpu->runtime_policy);
}

int bn_gpu_policy_prefill_ssm_run_chain_enabled(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           !gpu_runtime_enabled(gpu,
                                "BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
}

int bn_gpu_policy_prefill_ssm_ffn_fuse_allowed(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           !gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_SSM_FFN_FUSE");
}

int bn_gpu_policy_prefill_moe_chain_debug_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu, "BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
}

int bn_gpu_policy_prefill_hybrid_chain_debug_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu, "BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN");
}

int bn_gpu_policy_moe_prefill_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_MOE_PREFILL");
}

int bn_gpu_policy_moe_prefill_min_tokens_configured(
    const BnGPUBackend *gpu) {
    if (!bn_gpu_backend_is_cuda(gpu))
        return 0;
    const char *value = gpu_runtime_get(gpu,
                                        "BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    return value && *value;
}

int bn_gpu_policy_moe_prefill_min_tokens_or_default(
    const BnGPUBackend *gpu, int default_tokens) {
    return bn_gpu_backend_is_cuda(gpu)
        ? runtime_positive_int_or_default(
              &gpu->runtime_policy, "BN_CUDA_MOE_PREFILL_MIN_TOKENS",
              default_tokens)
        : default_tokens;
}

int bn_gpu_policy_cuda_dense_ffn_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_ENABLE_DENSE_FFN");
}

int bn_gpu_policy_cuda_dense_ffn_batch_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_DENSE_FFN_BATCH");
}

int bn_gpu_policy_cuda_moe_cublas_gateup_f16_out_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP_F16_OUT");
}

int bn_gpu_policy_cuda_moe_cublas_grouped_variable_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_VARIABLE") &&
           !bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED_VARIABLE");
}

int bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
                                                  const BnBackendRuntimePolicy *policy,
                                                  int routed_native_quant,
                                                  int routed_asymmetric_kquant,
                                                  int gate_f16,
                                                  int up_f16,
                                                  int down_f16,
                                                  int n_experts,
                                                  int k,
                                                  int route_items) {
    int enabled = gate_f16 && up_f16 && down_f16 &&
        ((routed_native_quant &&
          !runtime_compat_enabled(
              policy,
              "BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GROUPED",
              "BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GROUPED")) ||
         (routed_asymmetric_kquant &&
          !bn_backend_runtime_policy_enabled(
              policy, "BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED")));
    if (enabled && routed_asymmetric_kquant &&
        !bn_gpu_policy_moe_route_all_active_two(n_experts, k) &&
        route_items <= 256 &&
        !bn_backend_runtime_policy_enabled(
            policy, "BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_SMALL"))
        enabled = 0;
    return enabled;
}

int bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
                                                      const BnBackendRuntimePolicy *policy,
                                                      int use_grouped,
                                                      int routed_native_quant,
                                                      int routed_asymmetric_kquant,
                                                      int gate_f16,
                                                      int up_f16,
                                                      int down_f16,
                                                      int n_tokens) {
    return !use_grouped && gate_f16 && up_f16 && !down_f16 && n_tokens > 1 &&
        ((routed_native_quant &&
          !runtime_compat_enabled(
              policy,
              "BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GATEUP",
              "BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GATEUP")) ||
         (routed_asymmetric_kquant &&
          bn_backend_runtime_policy_enabled(
              policy, "BN_CUDA_ENABLE_MOE_CUBLAS_GATEUP") &&
          !bn_backend_runtime_policy_enabled(
              policy, "BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP")));
}

int bn_gpu_policy_cuda_moe_cublas_all_active_two_fixed_enabled(
    const BnBackendRuntimePolicy *policy,
    int use_grouped,
    int n_experts,
    int k) {
    return use_grouped &&
           bn_gpu_policy_moe_route_all_active_two(n_experts, k) &&
           !bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_DISABLE_MOE_CUBLAS_ALL2_FIXED");
}

int bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
    const BnBackendRuntimePolicy *policy,
    int n_tokens,
    int routed_asymmetric_kquant,
    int down_type,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_f16,
    int up_f16,
    int down_f16) {
    return n_tokens == 1 &&
           routed_asymmetric_kquant &&
           bn_gpu_policy_moe_route_all_active_two_large_hidden(n_experts,
                                                               k,
                                                               hidden_dim) &&
           bn_backend_quant_moe_down_uses_down_kquant(down_type) &&
           gate_f16 && up_f16 && down_f16 &&
           bn_gpu_policy_cuda_moe_cublas_decode_enabled(policy);
}

int bn_gpu_policy_cuda_moe_sorted_slots_enabled(
                                                const BnBackendRuntimePolicy *policy,
                                                int routed_asymmetric_kquant,
                                                int routed_native_quant,
                                                int n_tokens,
                                                int use_all_active_two_fixed,
                                                int use_grouped,
                                                int use_gateup_only) {
    return (routed_asymmetric_kquant || routed_native_quant) && n_tokens > 1 &&
           !use_all_active_two_fixed &&
           (use_grouped || use_gateup_only ||
            bn_backend_runtime_policy_enabled(
                policy, "BN_CUDA_ENABLE_MOE_ROUTE_SORT"));
}

int bn_gpu_policy_cuda_moe_prefill_internal_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_PROFILE_MOE_PREFILL_INTERNAL");
}

int bn_gpu_policy_cuda_moe_prefill_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PROFILE_MOE_PREFILL_EVERY", default_every);
}

int bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
    const BnBackendRuntimePolicy *policy,
    int add_norm_resid,
    int out_provided,
    int has_shared,
    int init_out_with_residual) {
    return add_norm_resid && !out_provided && !has_shared &&
           init_out_with_residual &&
           !bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_DISABLE_MOE_PREFILL_DIRECT_RESID_OUT");
}

int bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(
    const BnBackendRuntimePolicy *policy, int n_experts) {
    return n_experts <= 256 &&
           bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_ENABLE_MOE_BATCH_FUSED_ROUTE_TOPK") &&
           !bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_DISABLE_MOE_BATCH_FUSED_ROUTE_TOPK");
}

int bn_gpu_policy_cuda_moe_route_dist_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_PROFILE_MOE_ROUTE_DIST");
}

int bn_gpu_policy_cuda_moe_route_dist_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY", default_every);
}

int bn_gpu_policy_cuda_moe_cublas_grouped_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED");
}

int bn_gpu_policy_cuda_moe_cublas_gateup_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DEBUG_MOE_CUBLAS_GATEUP");
}

int bn_gpu_policy_cuda_moe_ffn_batch_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_FFN_BATCH");
}

int bn_gpu_policy_cuda_moe_ffn_batch_profile_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_PROFILE_MOE_FFN_BATCH_INTERNAL");
}

int bn_gpu_policy_cuda_moe_ffn_batch_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_PROFILE_MOE_FFN_BATCH_EVERY", default_every);
}

int bn_gpu_policy_moe_cache_prefill_enabled(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           !gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
}

int bn_gpu_policy_moe_prefill_shared_fuse_enabled(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           !gpu_runtime_enabled(gpu,
                                "BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
}

int bn_gpu_policy_moe_route_batch_enabled(const BnGPUBackend *gpu) {
    return !bn_gpu_backend_is_cuda(gpu) ||
           !gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_ROUTE_BATCH");
}

int bn_gpu_policy_moe_route_batch_debug_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu, "BN_CUDA_DEBUG_MOE_ROUTE_BATCH");
}

int bn_gpu_policy_large_hybrid_attention_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_LARGE_HYBRID_ATTN");
}

int bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(
               gpu, "BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
}

int bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(
               gpu, "BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
}

int bn_gpu_policy_large_hybrid_cpu_attention_safe_forced(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(
               gpu, "BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE");
}

int bn_gpu_policy_large_hybrid_prefill_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu,
                               "BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL");
}

int bn_gpu_policy_large_hybrid_prefill_chain_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(
               gpu, "BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN");
}

int bn_gpu_policy_large_hybrid_prefill_disabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu,
                               "BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
}

int bn_gpu_policy_large_hybrid_argmax_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu,
                               "BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX");
}

int bn_gpu_policy_prefill_matmul_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_PREFILL_MATMUL");
}

int bn_gpu_policy_prefill_matmul_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_PREFILL_MATMUL");
}

int bn_gpu_policy_prefill_direct_kv_disabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_PREFILL_DIRECT_KV");
}

int bn_gpu_policy_prefill_direct_kv_with_cpu_fallback_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(
               gpu, "BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");
}

int bn_gpu_policy_cpu_decode_fallback_requested(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_CPU_FALLBACK_LAYER") ||
           gpu_runtime_enabled(gpu, "BN_GPU_CPU_FALLBACK_FROM_LAYER") ||
           gpu_runtime_enabled(gpu, "BN_GPU_CPU_ATTN_LAYER") ||
           gpu_runtime_enabled(gpu, "BN_GPU_CPU_ATTN_FROM_LAYER");
}

int bn_gpu_policy_cpu_fallback_layer_or_default(
    const BnGPUBackend *gpu, int default_layer) {
    return runtime_int_or_default(&gpu->runtime_policy,
                                  "BN_GPU_CPU_FALLBACK_LAYER", default_layer);
}

int bn_gpu_policy_cpu_fallback_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer) {
    return runtime_int_or_default(&gpu->runtime_policy,
                                  "BN_GPU_CPU_FALLBACK_FROM_LAYER",
                                  default_layer);
}

int bn_gpu_policy_cpu_attention_layer_or_default(
    const BnGPUBackend *gpu, int default_layer) {
    return runtime_int_or_default(&gpu->runtime_policy,
                                  "BN_GPU_CPU_ATTN_LAYER", default_layer);
}

int bn_gpu_policy_cpu_attention_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer) {
    return runtime_int_or_default(&gpu->runtime_policy,
                                  "BN_GPU_CPU_ATTN_FROM_LAYER",
                                  default_layer);
}

int bn_gpu_policy_cpu_ffn_layer_or_default(
    const BnGPUBackend *gpu, int default_layer) {
    return runtime_int_or_default(&gpu->runtime_policy,
                                  "BN_GPU_CPU_FFN_LAYER", default_layer);
}

int bn_gpu_policy_cpu_ffn_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer) {
    return runtime_int_or_default(&gpu->runtime_policy,
                                  "BN_GPU_CPU_FFN_FROM_LAYER", default_layer);
}

int bn_gpu_policy_cpu_ffn_down_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer) {
    return runtime_int_or_default(&gpu->runtime_policy,
                                  "BN_GPU_CPU_FFN_DOWN_FROM_LAYER",
                                  default_layer);
}

int bn_gpu_policy_ssm_graph_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_SSM_GRAPH") ||
           (bn_gpu_backend_is_cuda(gpu) &&
            gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_SSM_GRAPH"));
}

int bn_gpu_policy_cuda_qkv_mixed_fuse_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_QKV_MIXED_FUSE");
}

int bn_gpu_policy_cuda_qkv_key_cache_fuse_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_QKV_KCACHE_FUSE");
}

int bn_gpu_policy_cuda_qkv_kpair_opt_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_QKV_KPAIR_OPT");
}

int bn_gpu_policy_cuda_legacy_block_gateup_warp_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_LEGACY_BLOCK_GATEUP_WARP",
        "BN_CUDA_DISABLE_Q5_GATEUP_WARP");
}

int bn_gpu_policy_cuda_native_quant_gateup_warp_disabled(const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_GATEUP_WARP",
        "BN_CUDA_DISABLE_Q8_GATEUP_WARP");
}

int bn_gpu_policy_cuda_graph_exec_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_CUDA_ENABLE_GRAPH_EXEC") ||
           bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_ENABLE_UNSAFE_MOE_FFN");
}

int bn_gpu_policy_cuda_moe_graph_max_experts_or_default(
    const BnBackendRuntimePolicy *policy, int default_experts) {
    return runtime_positive_int_or_default(
        policy, "BN_CUDA_MOE_GRAPH_MAX_EXPERTS", default_experts);
}

int bn_gpu_policy_cuda_decode_graph_default_enabled(
    const BnBackendRuntimePolicy *policy, int moe_graph,
    int default_moe_graph) {
    return !bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_DISABLE_GRAPH_EXEC") &&
           !bn_backend_runtime_policy_enabled(
               policy, "BN_CUDA_ENABLE_MOE_FFN") &&
           (!moe_graph || default_moe_graph);
}

int bn_gpu_policy_cuda_cublas_cache_max_mb(
    const BnBackendRuntimePolicy *policy, int default_mb, int large_budget) {
    int max_mb = large_budget ? 512 : default_mb;
    const char *max_env = bn_backend_runtime_policy_get(
        policy, "BN_CUDA_CUBLAS_CACHE_MAX_MB");
    if (max_env && *max_env)
        max_mb = atoi(max_env);
    return max_mb;
}

int bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
                                               const BnBackendRuntimePolicy *policy,
                                               int tensor_type,
                                               int force_down_kquant_f32,
                                               int force_f16) {
    if (force_f16)
        return 0;

    const char *max_env = bn_backend_runtime_policy_get(
        policy, "BN_CUDA_CUBLAS_CACHE_MAX_MB");
    if (max_env && *max_env)
        return atoi(max_env);

    if (force_down_kquant_f32 &&
        bn_gpu_policy_moe_down_kquant_f32_cache_forced(policy))
        return 0;

    return bn_backend_quant_aux_cache_prefers_large_budget(tensor_type)
        ? 512
        : 128;
}

int bn_gpu_policy_cuda_down_kquant_f16_cache_adds_f32_down_cache(const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_moe_down_kquant_f32_cache_disabled(policy) &&
           !runtime_compat_enabled(policy,
               "BN_CUDA_DISABLE_MOE_F16_KQUANT_F32_DOWN_CACHE",
               "BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE");
}

size_t bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols) {
    if (!gpu_policy_backend_caps(gpu)->moe_down_cublas_cache ||
        rows <= 0 || cols <= 0 ||
        !bn_backend_quant_moe_down_cublas_cache_supported(tensor_type) ||
        !bn_gpu_policy_cuda_cublas_matmul_enabled(
            gpu ? &gpu->runtime_policy : NULL))
        return 0;
    size_t elems = (size_t)rows * (size_t)cols;
    int down_kquant_as_f16 =
        bn_gpu_policy_cuda_down_kquant_cublas_f16_cache_enabled(
            gpu ? &gpu->runtime_policy : NULL);
    size_t elem_size =
        (size_t)bn_backend_quant_moe_down_cublas_cache_elem_bytes(
            tensor_type, down_kquant_as_f16);
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

static size_t gpu_policy_aux_cache_bytes(const BnGPUBackend *gpu,
                                         int tensor_type,
                                         int rows,
                                         int cols) {
    if (rows <= 0 || cols <= 0 || (cols & 31) != 0 ||
        !bn_gpu_policy_cuda_cublas_matmul_enabled(
            gpu ? &gpu->runtime_policy : NULL) ||
        !bn_backend_quant_aux_cache_supported(tensor_type))
        return 0;
    int down_kquant_as_f16 =
        bn_backend_quant_aux_cache_can_use_f16(tensor_type) &&
        bn_gpu_policy_cuda_down_kquant_cublas_f16_cache_enabled(
            gpu ? &gpu->runtime_policy : NULL);
    if ((size_t)rows > SIZE_MAX / (size_t)cols)
        return SIZE_MAX;
    size_t elems = (size_t)rows * (size_t)cols;
    size_t elem_size =
        bn_backend_quant_aux_cache_uses_f32(tensor_type, down_kquant_as_f16)
            ? sizeof(float)
            : sizeof(uint16_t);
    if (elem_size != 0 && elems > SIZE_MAX / elem_size)
        return SIZE_MAX;
    size_t bytes = elems * elem_size;

    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(
        gpu ? &gpu->runtime_policy : NULL, 128,
        bn_backend_quant_aux_cache_prefers_large_budget(tensor_type));
    if (max_mb > 0 && bytes > (size_t)max_mb * 1024u * 1024u)
        return 0;
    return bytes;
}

size_t bn_gpu_policy_aux_cache_bytes(const BnGPUBackend *gpu,
                                     int tensor_type,
                                     int rows,
                                     int cols) {
    return gpu_policy_aux_cache_bytes(gpu, tensor_type, rows, cols);
}

int bn_gpu_policy_cuda_cublas_aux_cache_supported(
                                                  const BnBackendRuntimePolicy *policy,
                                                  int tensor_type,
                                                  int cols) {
    return cols > 0 &&
           (cols & 31) == 0 &&
           bn_gpu_policy_cuda_cublas_matmul_enabled(policy) &&
           bn_backend_quant_eager_aux_cache_supported(tensor_type);
}

int bn_gpu_policy_moe_auto_resident_enabled(const BnGPUBackend *gpu) {
    return !gpu_runtime_enabled(gpu, "BN_GPU_MOE_DISABLE_AUTO_RESIDENT");
}

size_t bn_gpu_policy_moe_cache_reserve_bytes(
    const BnBackendRuntimePolicy *policy) {
    return mb_to_bytes_saturating(
        positive_runtime_mb_or_default(
            policy, "BN_GPU_MOE_CACHE_RESERVE_MB", 4096));
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

int bn_gpu_policy_duplicate_moe_cache_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_runtime_enabled(gpu, "BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE") &&
           !gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE");
}

int bn_gpu_policy_webgpu_repacked_buffer_supported(int tensor_type) {
    return bn_backend_quant_can_gpu_repack(tensor_type);
}

int bn_gpu_policy_webgpu_repacked_bias_supported(int tensor_type) {
    return bn_backend_quant_gpu_supports_repacked_bias(tensor_type);
}

int bn_gpu_policy_metal_mmap_zero_copy_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_METAL_DISABLE_MMAP_ZERO_COPY");
}

int bn_gpu_policy_apply_metal_barrier_disable_override(
    BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_set(
        policy, "BN_METAL_DISABLE_BARRIERS", "1", 1);
}

int bn_gpu_policy_apply_specialized_native_quant_decode_override(
    BnBackendRuntimePolicy *policy) {
    if (bn_backend_runtime_policy_set(
            policy, "BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT", "1", 1) != 0)
        return -1;
    bn_backend_runtime_policy_unset(
        policy, "BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT");
    return 0;
}

int bn_gpu_policy_apply_specialized_native_quant_decode_disable_override(
    BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_set(
        policy, "BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT", "1", 1);
}

int bn_gpu_policy_apply_native_quant_prepared_override(
    BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_set(
        policy, "BN_METAL_NATIVE_QUANT_PREPARED", "1", 1);
}

int bn_gpu_policy_apply_metal_small_dense_native_quant_default_disable_override(
    BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_set(
        policy, "BN_METAL_DISABLE_SMALL_DENSE_NATIVE_QUANT_DEFAULT", "1", 1);
}

int bn_gpu_policy_apply_metal_private_weights_override(
    BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_set(
        policy, "BN_METAL_PRIVATE_WEIGHTS", "1", 1);
}

int bn_gpu_policy_apply_metal_cpu_route_resident_moe_override(
    BnBackendRuntimePolicy *policy) {
    static const char *const names[] = {
        "BN_GPU_DISABLE_ROUTED_MOE_DECODE",
        "BN_GPU_DISABLE_MOE_ALL_LAYOUT",
        "BN_METAL_ENABLE_CPU_ROUTE_RESIDENT_MOE",
        "BN_METAL_ENABLE_REFERENCE_ATTENTION",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (bn_backend_runtime_policy_set(policy, names[i], "1", 1) != 0)
            return -1;
    }
    return 0;
}

int bn_gpu_policy_metal_apply_small_dense_native_quant_default(
    BnBackendRuntimePolicy *policy) {
    if (!bn_gpu_policy_metal_small_dense_native_quant_enabled(policy) &&
        !runtime_compat_enabled2(
            policy, "BN_METAL_DISABLE_SMALL_DENSE_NATIVE_QUANT_DEFAULT",
            "BN_METAL_DISABLE_SMALL_DENSE_EXACT_NATIVE_DEFAULT",
            "BN_METAL_DISABLE_Q4_Q8_DEFAULT")) {
        if (bn_backend_runtime_policy_set(
                policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT", "1", 1) != 0)
            return -1;
        if (!runtime_compat_value2(
                policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FROM_LAYER",
                "BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER",
                "BN_GPU_Q4_Q8_FROM_LAYER") &&
            bn_backend_runtime_policy_set(
                policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FROM_LAYER", "0",
                1) != 0)
            return -1;
        if (!runtime_compat_enabled2(
                policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_ATTN_ONLY",
                "BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY",
                "BN_GPU_Q4_Q8_ATTN_ONLY") &&
            !runtime_compat_enabled2(
                policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_ONLY",
                "BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY",
                "BN_GPU_Q4_Q8_FFN_ONLY") &&
            bn_backend_runtime_policy_set(
                policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_ONLY", "1",
                1) != 0)
            return -1;
    }
    return 0;
}

int bn_gpu_policy_metal_small_dense_native_quant_enabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled2(
        policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE", "BN_GPU_Q4_Q8");
}

int bn_gpu_policy_metal_native_quant_prepared_enabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_METAL_NATIVE_QUANT_PREPARED",
                                  "BN_METAL_Q4_PREPARED");
}

int bn_gpu_policy_metal_prepared_f32_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_ENABLE_PREPARED_F32");
}

int bn_gpu_policy_small_dense_native_quant_prepared_layer_default_enabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_METAL_NATIVE_QUANT_PREPARED",
                                  "BN_METAL_Q4_PREPARED");
}

int bn_gpu_policy_metal_native_quant_prepared_upload_enabled(
    const BnBackendRuntimePolicy *policy) {
    const char *from_layer = runtime_compat_value2(
        policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FROM_LAYER",
        "BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER",
        "BN_GPU_Q4_Q8_FROM_LAYER");
    return bn_gpu_policy_metal_native_quant_prepared_enabled(policy) &&
           bn_gpu_policy_metal_small_dense_native_quant_enabled(policy) &&
           (!from_layer || atoi(from_layer) <= 0) &&
           !runtime_compat_enabled2(
               policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_ATTN_ONLY",
               "BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY",
               "BN_GPU_Q4_Q8_ATTN_ONLY") &&
           !runtime_compat_enabled2(
               policy, "BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_ONLY",
               "BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY",
               "BN_GPU_Q4_Q8_FFN_ONLY");
}

int bn_gpu_policy_metal_repacked_buffer_supported(int tensor_type) {
    return bn_backend_quant_can_gpu_repack(tensor_type);
}

int bn_gpu_policy_metal_repacked_buffer_type(int tensor_type) {
    return bn_gpu_policy_metal_repacked_buffer_supported(tensor_type)
        ? tensor_type
        : -1;
}

int bn_gpu_policy_metal_prepared_stacked_upload_blocked(
    const BnBackendRuntimePolicy *policy, int tensor_type) {
    return bn_gpu_policy_metal_repacked_buffer_supported(tensor_type) &&
           bn_gpu_policy_metal_native_quant_prepared_upload_enabled(policy);
}

int bn_gpu_policy_metal_shared_weights_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_SHARED_WEIGHTS");
}

int bn_gpu_policy_metal_specialized_native_quant_enabled(
    const BnBackendRuntimePolicy *policy) {
    if (runtime_compat_enabled(
            policy, "BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT",
            "BN_METAL_DISABLE_Q6_Q8K"))
        return 0;
    return runtime_compat_enabled(
        policy, "BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT",
        "BN_METAL_ENABLE_Q6_Q8K");
}

int bn_gpu_policy_specialized_native_quant_decode_path_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_gpu_policy_metal_specialized_native_quant_enabled(policy);
}

int bn_gpu_policy_metal_native_quant_barriers_enabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy, "BN_METAL_NATIVE_QUANT_BARRIERS",
                                  "BN_METAL_Q8_BARRIERS");
}

int bn_gpu_policy_metal_small_dense_native_quant_matvec_supported(
    int tensor_type,
    int small_dense_native_quant_enabled,
    int native_quant_prepared,
    int has_native_quant_pipeline,
    int has_native_quant_pipeline_unprepared,
    int has_native_quant_prepared_pipeline) {
    if (!small_dense_native_quant_enabled ||
        !bn_backend_quant_supports_direct_native_quant_matvec(tensor_type) ||
        !has_native_quant_pipeline)
        return 0;
    return native_quant_prepared
        ? has_native_quant_prepared_pipeline
        : has_native_quant_pipeline_unprepared;
}

int bn_gpu_policy_metal_small_dense_native_quant_graph_path_supported(
    int tensor_type,
    int small_dense_native_quant_enabled,
    int native_quant_prepared,
    int native_quant_prepared_path,
    int has_native_quant_pipeline,
    int has_pipeline) {
    return native_quant_prepared == native_quant_prepared_path &&
           small_dense_native_quant_enabled &&
           bn_backend_quant_supports_direct_native_quant_matvec(tensor_type) &&
           has_native_quant_pipeline &&
           has_pipeline;
}

int bn_gpu_policy_metal_block_q8_activation_graph_path_supported(
    int tensor_type,
    int block_q8_activation_enabled,
    int has_activation_quant_pipeline,
    int has_matvec_pipeline) {
    return block_q8_activation_enabled &&
           bn_backend_quant_gpu_matvec_block_q8_activation_flag(
               tensor_type, 1) ==
               BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT &&
           has_activation_quant_pipeline &&
           has_matvec_pipeline;
}

int bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
    const BnBackendRuntimePolicy *policy,
    int tensor_type,
    int cols,
    int has_prepared_activation_pipeline,
    int has_specialized_native_pipeline) {
    return bn_gpu_policy_metal_specialized_native_quant_enabled(policy) &&
           bn_backend_quant_prefers_specialized_native_quant_matvec(
               tensor_type, cols) &&
           has_prepared_activation_pipeline &&
           has_specialized_native_pipeline &&
           cols > 0 &&
           (cols % 256) == 0;
}

int bn_gpu_policy_metal_specialized_native_quant_shape_default_enabled(
    const BnBackendRuntimePolicy *policy,
    int tensor_type, int rows, int cols) {
    if (runtime_compat_enabled(
            policy,
            "BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT",
            "BN_METAL_DISABLE_Q6_Q8K"))
        return 0;
    return bn_backend_quant_prefers_tall_specialized_native_quant_matvec(
        tensor_type, rows, cols);
}

int bn_gpu_policy_metal_cpu_order_rmsnorm_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_CPU_ORDER_RMSNORM");
}

int bn_gpu_policy_metal_full_barriers_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_FULL_BARRIERS");
}

int bn_gpu_policy_metal_route_history_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_ROUTE_HISTORY");
}

int bn_gpu_policy_metal_barriers_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_ENABLE_BARRIERS") ||
           bn_gpu_policy_metal_full_barriers_enabled(policy);
}

int bn_gpu_policy_metal_barriers_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_DISABLE_BARRIERS") ||
           !bn_gpu_policy_metal_barriers_enabled(policy);
}

int bn_gpu_policy_fused_gateup_enabled(const BnGPUBackend *gpu) {
    return !gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_FUSED_GATEUP");
}

int bn_gpu_policy_small_dense_native_quant_fused_gateup_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !small_dense_native_quant_gateup_disabled(policy);
}

int bn_gpu_policy_small_dense_native_quant_attn_only_enabled(
    const BnBackendRuntimePolicy *policy) {
    return small_dense_native_quant_attn_only(policy);
}

int bn_gpu_policy_small_dense_native_quant_ffn_only_enabled(
    const BnBackendRuntimePolicy *policy) {
    return small_dense_native_quant_ffn_only(policy);
}

int bn_gpu_policy_small_dense_native_quant_from_layer_or_default(
    const BnBackendRuntimePolicy *policy, int n_layers) {
    const char *env = small_dense_native_quant_from_layer_env(policy);
    if (env)
        return atoi(env);
    return small_dense_native_quant_enabled(policy) ? n_layers - 1 : -1;
}

int bn_gpu_policy_small_dense_native_quant_to_layer_or_default(
    const BnBackendRuntimePolicy *policy,
    int n_layers,
    int native_quant_prepared) {
    const char *env = small_dense_native_quant_to_layer_env(policy);
    if (env)
        return atoi(env);

    env = small_dense_native_quant_tail_layer_env(policy);
    if (env) {
        int tail_native = atoi(env);
        if (tail_native > 0) {
            int to_layer = n_layers - tail_native - 1;
            return to_layer < -1 ? -1 : to_layer;
        }
        return -1;
    }

    if (small_dense_native_quant_enabled(policy) && !native_quant_prepared &&
        n_layers > 33)
        return 0;
    return -1;
}

int bn_gpu_policy_gateup_split_enabled(const BnGPUBackend *gpu) {
    return !gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_GATEUP_SPLIT");
}

int bn_gpu_policy_small_dense_native_quant_ffn_down_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !small_dense_native_quant_ffn_down_disabled(policy);
}

int bn_gpu_policy_qkv_split_enabled(const BnGPUBackend *gpu) {
    return !gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_QKV_SPLIT");
}

int bn_gpu_policy_qkv_split_debug_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_DEBUG_QKV_SPLIT");
}

int bn_gpu_policy_ssm_qkvz_split_enabled(const BnGPUBackend *gpu) {
    return !gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_SSM_QKVZ_SPLIT");
}

int bn_gpu_policy_ssm_ab_stack_enabled(const BnGPUBackend *gpu) {
    return !gpu_runtime_enabled(gpu, "BN_GPU_DISABLE_SSM_AB_STACK");
}

int bn_gpu_policy_split_residual_rmsnorm_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_SPLIT_RESIDUAL_RMSNORM");
}

int bn_gpu_policy_debug_fallback_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_DEBUG_FALLBACK");
}

int bn_gpu_policy_metal_cpu_route_resident_moe_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_METAL_ENABLE_CPU_ROUTE_RESIDENT_MOE");
}

int bn_gpu_policy_metal_dense_residual_graph_diagnostic_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_METAL_ENABLE_DENSE_RESIDUAL_GRAPH_DIAGNOSTIC");
}

int bn_gpu_policy_metal_shared_mmap_buffer_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_METAL_DISABLE_SHARED_MMAP_BUFFER");
}

int bn_gpu_policy_force_graph_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_FORCE_GRAPH");
}

int bn_gpu_policy_flash_min_kv_or_default(const BnGPUBackend *gpu,
                                          int default_min_kv) {
    const char *env = gpu_runtime_get(gpu, "BN_GPU_FLASH_MIN_KV");
    return env ? atoi(env) : default_min_kv;
}

int bn_gpu_policy_backend_flash_max_kv_or_default(const BnGPUBackend *gpu,
                                                  int default_max_kv) {
    const BnGPUPolicyBackendCaps *caps = gpu_policy_backend_caps(gpu);
    const char *env = gpu_runtime_get(gpu, "BN_GPU_FLASH_MAX_KV");
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
    return bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_LARGE_GRAPH_NATIVE);
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

int bn_gpu_policy_backend_reference_attention_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_REFERENCE_ATTENTION);
}

int bn_gpu_policy_backend_reference_attention_native_graph_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_has_cap(
        gpu, BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH);
}

int bn_gpu_policy_backend_reference_attention_token_fallback_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_has_cap(
        gpu, BN_GPU_CAP_REFERENCE_ATTENTION_TOKEN_FALLBACK);
}

int bn_gpu_policy_backend_reference_recurrent_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_REFERENCE_RECURRENT);
}

int bn_gpu_policy_backend_reference_attention_fallback_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_has_cap(
        gpu, BN_GPU_CAP_REFERENCE_ATTENTION_FALLBACK);
}

int bn_gpu_policy_backend_small_dense_native_quant_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->small_dense_native_quant;
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
    return bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_DECODE_GRAPH_CACHE);
}

int bn_gpu_policy_backend_moe_reference_attention_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->moe_reference_attention;
}

int bn_gpu_policy_backend_ssm_graph_supported(const BnGPUBackend *gpu) {
    return bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_SSM_GRAPH);
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
    return gpu_policy_backend_caps(gpu)->resident_moe_ffn ||
           bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_ROUTED_FFN);
}

int bn_gpu_policy_backend_moe_expert_graph_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_backend_resident_moe_ffn_supported(gpu) ||
           bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_EXPERT_GRAPH);
}

int bn_gpu_policy_backend_weighted_add_sigmoid_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) || bn_gpu_backend_is_metal(gpu);
}

int bn_gpu_policy_backend_moe_route_topk_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_ROUTED_FFN);
}

int bn_gpu_policy_metal_routed_moe_decode_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_GPU_DISABLE_ROUTED_MOE_DECODE");
}

int bn_gpu_policy_metal_moe_expert_graph_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_GPU_DISABLE_MOE_EXPERT_GRAPH");
}

int bn_gpu_policy_backend_moe_gateup_split_supported(
    const BnGPUBackend *gpu) {
    return gpu_policy_backend_caps(gpu)->moe_gateup_split;
}

int bn_gpu_policy_argmax_debug_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_DEBUG_ARGMAX");
}

static int gpu_policy_kquant_logits_refine_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_GPU_ENABLE_KQUANT_LOGITS_REFINE",
        "BN_GPU_ENABLE_Q6_LOGITS_REFINE");
}

static int gpu_policy_kquant_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_GPU_DISABLE_KQUANT_LOGITS_REFINE",
        "BN_GPU_DISABLE_Q6_LOGITS_REFINE");
}

static int gpu_policy_kquant_logits_refine_top_or_default(
    const BnBackendRuntimePolicy *policy, int default_top) {
    const char *env = runtime_compat_value2(policy,
        "BN_GPU_KQUANT_LOGITS_REFINE_TOP",
        "BN_GPU_Q6_Q8K_REFINE_TOP", NULL);
    return env ? atoi(env) : default_top;
}

static int gpu_policy_native_quant_logits_refine_requested(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_GPU_ENABLE_NATIVE_QUANT_LOGITS_REFINE",
        "BN_GPU_ENABLE_Q8_LOGITS_REFINE");
}

static int gpu_policy_native_quant_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_GPU_DISABLE_NATIVE_QUANT_LOGITS_REFINE",
        "BN_GPU_DISABLE_Q8_LOGITS_REFINE");
}

static int gpu_policy_native_quant_logits_refine_top_or_default(
    const BnBackendRuntimePolicy *policy, int default_top) {
    const char *env = runtime_compat_value2(policy,
        "BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP",
        "BN_GPU_Q8_REFINE_TOP", NULL);
    return env ? atoi(env) : default_top;
}

static int gpu_policy_cuda_moe_ffn_disabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_FFN");
}

static int gpu_policy_cuda_moe_router_topk_disabled(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_ROUTER_TOPK");
}

static int gpu_policy_cuda_native_quant_moe_cpu_route_resident_disabled(
    const BnBackendRuntimePolicy *policy) {
    return runtime_compat_enabled(policy,
        "BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT",
        "BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
}

static int gpu_policy_cuda_moe_router_gpu_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_GPU_ENABLE_MOE_ROUTER_GPU");
}

static int gpu_policy_cuda_moe_router_gpu_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_GPU_DISABLE_MOE_ROUTER_GPU");
}

static int gpu_policy_cuda_moe_router_diff2_disabled(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
}

static int gpu_policy_cuda_moe_routed_ffn_batch_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH");
}

static int gpu_policy_cuda_moe_route_routed_ffn_batch_disabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
}

static int gpu_policy_cuda_large_moe_route_routed_ffn_batch_requested(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
}

static int gpu_policy_cuda_moe_cpu_actual_override_requested(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu,
                               "BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
}

static int gpu_policy_moe_cpu_actual_override_requested(
    const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_OVERRIDE_MOE_WITH_CPU_ACTUAL");
}

int bn_gpu_policy_cpu_logits_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_CPU_LOGITS");
}

int bn_gpu_policy_compare_logits_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_COMPARE_LOGITS");
}

int bn_gpu_policy_debug_argmax_compare_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_DEBUG_ARGMAX_COMPARE");
}

int bn_gpu_policy_backend_kquant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int kquant_refine_default) {
    const BnGPUPolicyBackendCaps *caps = gpu_policy_backend_caps(gpu);
    const BnBackendRuntimePolicy *policy =
        gpu ? &gpu->runtime_policy : NULL;
    return kquant_refine_default ||
           gpu_policy_kquant_logits_refine_requested(policy) ||
           (!caps->suppress_implicit_kquant_logits_refine &&
            !gpu_policy_kquant_logits_refine_disabled(policy));
}

int bn_gpu_policy_kquant_logits_refine_top_or_default(
    const BnBackendRuntimePolicy *policy, int default_top) {
    return gpu_policy_kquant_logits_refine_top_or_default(policy,
                                                           default_top);
}

int bn_gpu_policy_backend_native_quant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int native_quant_refine_default) {
    const BnGPUPolicyBackendCaps *caps = gpu_policy_backend_caps(gpu);
    const BnBackendRuntimePolicy *policy =
        gpu ? &gpu->runtime_policy : NULL;
    return gpu_policy_native_quant_logits_refine_requested(policy) ||
           native_quant_refine_default ||
           (!caps->suppress_implicit_native_quant_logits_refine &&
            !gpu_policy_native_quant_logits_refine_disabled(policy));
}

int bn_gpu_policy_native_quant_logits_refine_top_or_default(
    const BnBackendRuntimePolicy *policy, int default_top) {
    return gpu_policy_native_quant_logits_refine_top_or_default(policy,
                                                                 default_top);
}

int bn_gpu_policy_moe_ffn_disabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           gpu_policy_cuda_moe_ffn_disabled(gpu);
}

int bn_gpu_policy_moe_router_topk_enabled(const BnGPUBackend *gpu,
                                          int eligible) {
    if (bn_gpu_backend_is_cuda(gpu))
        return eligible && !gpu_policy_cuda_moe_router_topk_disabled(gpu);
    if (bn_gpu_backend_is_metal(gpu))
        return eligible && bn_gpu_policy_metal_routed_moe_decode_enabled(
                               &gpu->runtime_policy);
    return 0;
}

int bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(
    const BnGPUBackend *gpu, int eligible) {
    if (bn_gpu_backend_is_cuda(gpu))
        return eligible &&
               !gpu_policy_cuda_native_quant_moe_cpu_route_resident_disabled(
                   gpu ? &gpu->runtime_policy : NULL);
    if (bn_gpu_backend_is_metal(gpu))
        return eligible &&
               bn_gpu_policy_metal_cpu_route_resident_moe_enabled(
                   &gpu->runtime_policy);
    return 0;
}

int bn_gpu_policy_lowbit_block32_moe_cpu_route_resident_enabled(
    const BnGPUBackend *gpu, int eligible) {
    return bn_gpu_backend_is_metal(gpu) && eligible &&
           bn_gpu_policy_metal_cpu_route_resident_moe_enabled(
               &gpu->runtime_policy);
}

int bn_gpu_policy_moe_router_gpu_enabled(
    const BnBackendRuntimePolicy *policy) {
    return gpu_policy_cuda_moe_router_gpu_requested(policy) &&
           !gpu_policy_cuda_moe_router_gpu_disabled(policy);
}

int bn_gpu_policy_moe_router_diff2_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           !gpu_policy_cuda_moe_router_diff2_disabled(gpu);
}

int bn_gpu_policy_moe_routed_ffn_batch_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !gpu_policy_cuda_moe_routed_ffn_batch_disabled(policy);
}

int bn_gpu_policy_moe_routed_ffn_batch_allowed(
                                                const BnBackendRuntimePolicy *policy,
                                                int large_moe) {
    if (gpu_policy_cuda_moe_route_routed_ffn_batch_disabled(policy))
        return 0;
    return !large_moe ||
           gpu_policy_cuda_large_moe_route_routed_ffn_batch_requested(policy);
}

int bn_gpu_policy_moe_cpu_actual_override_enabled(
    const BnGPUBackend *gpu) {
    return gpu_policy_moe_cpu_actual_override_requested(gpu) ||
           (bn_gpu_backend_is_cuda(gpu) &&
            gpu_policy_cuda_moe_cpu_actual_override_requested(gpu));
}

int bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled(
    const BnBackendRuntimePolicy *policy) {
    return small_dense_native_quant_cpu_attention_safe_disabled(policy);
}

int bn_gpu_policy_small_dense_native_quant_disabled(
    const BnBackendRuntimePolicy *policy) {
    return small_dense_native_quant_disabled(policy);
}

int bn_gpu_policy_small_dense_native_quant_ffn_down_requested(
    const BnBackendRuntimePolicy *policy) {
    return small_dense_native_quant_ffn_down_requested(policy);
}

int bn_gpu_policy_small_dense_prefill_disabled(const BnGPUBackend *gpu) {
    return bn_gpu_backend_is_cuda(gpu) &&
           small_dense_prefill_disabled(gpu ? &gpu->runtime_policy : NULL);
}

int bn_gpu_policy_native_quant_logits_refine_requested(
    const BnBackendRuntimePolicy *policy) {
    return native_quant_logits_refine_requested(policy);
}

int bn_gpu_policy_native_quant_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy) {
    return native_quant_logits_refine_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_fast_ffn_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_fast_graph_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_fast_graph_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_cublas_decode_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_cublas_decode_requested(policy);
}

int bn_gpu_policy_cuda_moe_cublas_decode_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DISABLE_MOE_CUBLAS_DECODE");
}

int bn_gpu_policy_cuda_moe_cublas_decode_debug_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_CUDA_DEBUG_MOE_CUBLAS_DECODE");
}

int bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_fast_route_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_dot_prepared_input_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_dot_prepared_input_default_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_route_dot_prepared_input_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_route_dot_prepared_input_default_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_route_block_prepared_input_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_fast_prepared_gateup_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_fast_prepared_gateup_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_pair_path_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layers_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_pair_path_f32_layers_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer) {
    return all_active_two_kquant_moe_down_pair_path_f32_layer_selected(
        policy, layer);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_ordered_quant_path_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_ordered_quant_path_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_f32_cache_default_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_f32_cache_default_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_f32_all_active_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_f32_all_active_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_enabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_f32_cache_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_float_4row_default_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_float_4row_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_f32_4row_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer) {
    const char *env =
        all_active_two_kquant_moe_down_f32_4row_layers(policy);
    return !env || !*env ||
           all_active_two_kquant_moe_down_f32_4row_layer_selected(
               policy, layer);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_f32_4row_default_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_f32_4row_default_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_down_f32_4row_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_down_f32_4row_disabled(policy);
}

float bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(
    const BnBackendRuntimePolicy *policy, float default_eps) {
    return all_active_two_kquant_down_skip_eps_or_default(policy, default_eps);
}

int bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_cpu_attention_safe_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_logits_refine_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_cpu_moe_safe_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_reference_attention_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_reference_attention_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_cpu_route_resident_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_reference_gpu_route_requested(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_reference_gpu_route_requested(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_reference_gpu_route_disabled(
    const BnBackendRuntimePolicy *policy) {
    return all_active_two_kquant_moe_reference_gpu_route_disabled(policy);
}

int bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_gpu_policy_moe_router_gpu_enabled(policy) ||
           bn_gpu_policy_all_active_two_kquant_moe_reference_gpu_route_requested(
               policy);
}

void bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(
    const BnBackendRuntimePolicy *policy,
    int *from_layer,
    int *to_layer) {
    const char *env;

    if (from_layer)
        *from_layer = -1;
    if (to_layer)
        *to_layer = -1;

    env = all_active_two_kquant_moe_route_from_layer_value(policy);
    if (env && from_layer)
        *from_layer = atoi(env);
    env = all_active_two_kquant_moe_route_to_layer_value(policy);
    if (env && to_layer)
        *to_layer = atoi(env);
}

int bn_gpu_policy_moe_compare_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer, int pos) {
    const char *compare_moe_env = bn_backend_runtime_policy_get(
        policy, "BN_GPU_COMPARE_MOE_LAYER");
    if (!compare_moe_env)
        return 0;
    const char *compare_pos_env = bn_backend_runtime_policy_get(
        policy, "BN_GPU_COMPARE_MOE_POS");
    int compare_pos = compare_pos_env ? atoi(compare_pos_env) : -1;
    int layer_selected = strcmp(compare_moe_env, "all") == 0 ||
                         atoi(compare_moe_env) == layer;
    return layer_selected && (compare_pos < 0 || compare_pos == pos);
}

int bn_gpu_policy_moe_compare_input_norm_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_GPU_COMPARE_MOE_INPUT_NORM");
}

int bn_gpu_policy_moe_compare_actual_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_GPU_COMPARE_MOE_ACTUAL");
}

int bn_gpu_policy_moe_compare_route_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_GPU_COMPARE_MOE_ROUTE");
}

int bn_gpu_policy_moe_compare_raw_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_GPU_COMPARE_MOE_RAW");
}

int bn_gpu_policy_moe_compare_mid_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_GPU_COMPARE_MOE_MID");
}

int bn_gpu_policy_moe_compare_parts_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_GPU_COMPARE_MOE_PARTS");
}

int bn_gpu_policy_moe_compare_shared_mid_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_GPU_COMPARE_MOE_SHARED_MID");
}

int bn_gpu_policy_moe_compare_shared_down_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_GPU_COMPARE_MOE_SHARED_DOWN");
}

int bn_gpu_policy_moe_compare_norm_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_GPU_COMPARE_MOE_NORM");
}

int bn_gpu_policy_compare_attention_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_ATTENTION_LAYER",
                                  default_layer);
}

int bn_gpu_policy_compare_attention_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_ATTENTION_POS",
                                  default_pos);
}

int bn_gpu_policy_compare_gqa_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_GQA_LAYER",
                                  default_layer);
}

int bn_gpu_policy_compare_gqa_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_GQA_POS",
                                  default_pos);
}

int bn_gpu_policy_compare_qkv_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_QKV_LAYER",
                                  default_layer);
}

int bn_gpu_policy_compare_qkv_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_QKV_POS",
                                  default_pos);
}

int bn_gpu_policy_compare_ffn_down_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_FFN_DOWN_LAYER",
                                  default_layer);
}

int bn_gpu_policy_compare_ffn_down_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_FFN_DOWN_POS",
                                  default_pos);
}

int bn_gpu_policy_compare_ffn_state_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_FFN_STATE_LAYER",
                                  default_layer);
}

int bn_gpu_policy_compare_ffn_state_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_FFN_STATE_POS",
                                  default_pos);
}

int bn_gpu_policy_compare_ssm_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_SSM_LAYER",
                                  default_layer);
}

int bn_gpu_policy_compare_ssm_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos) {
    return runtime_int_or_default(policy, "BN_GPU_COMPARE_SSM_POS",
                                  default_pos);
}

int bn_gpu_policy_metal_reference_attention_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(
        policy, "BN_METAL_ENABLE_REFERENCE_ATTENTION");
}

int bn_gpu_policy_metal_prepared_native_quant_attention_enabled(
    const BnBackendRuntimePolicy *policy) {
    return !bn_backend_runtime_policy_enabled(
        policy, "BN_METAL_DISABLE_PREPARED_NATIVE_QUANT_ATTENTION");
}

uint32_t bn_gpu_policy_metal_reference_attention_stage_mask(
    const BnBackendRuntimePolicy *policy) {
    const char *value = bn_backend_runtime_policy_get(
        policy, "BN_METAL_REFERENCE_ATTENTION_STAGES");
    return (uint32_t)(value ? atoi(value) : 7) & 7u;
}

int bn_gpu_policy_metal_ssm_graph_enabled(
    const BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_enabled(policy,
                                             "BN_METAL_ENABLE_SSM_GRAPH");
}

int bn_gpu_policy_moe_shared_cpu_fallback_enabled(
    const BnGPUBackend *gpu, int eligible) {
    if (!eligible)
        return 0;
    if (bn_gpu_backend_is_cuda(gpu))
        return gpu_runtime_enabled(
                   gpu, "BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK") &&
               !gpu_runtime_enabled(
                   gpu, "BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    if (bn_gpu_backend_is_metal(gpu))
        return gpu_runtime_enabled(
                   gpu, "BN_METAL_ENABLE_MOE_SHARED_CPU_FALLBACK") &&
               !gpu_runtime_enabled(
                   gpu, "BN_METAL_DISABLE_MOE_SHARED_CPU_FALLBACK");
    return 0;
}

int bn_gpu_policy_moe_gateup_split_enabled(
    const BnGPUBackend *gpu, int can_split) {
    return can_split &&
           (!bn_gpu_backend_is_cuda(gpu) ||
            !gpu_runtime_enabled(gpu, "BN_CUDA_DISABLE_MOE_GATEUP_SPLIT"));
}

int bn_gpu_policy_moe_route_profile_enabled(const BnGPUBackend *gpu) {
    return gpu_runtime_enabled(gpu, "BN_GPU_MOE_ROUTE_PROFILE");
}

int bn_gpu_policy_moe_route_profile_every_or_default(
    const BnGPUBackend *gpu, int default_every) {
    return gpu
        ? runtime_positive_int_or_default(
              &gpu->runtime_policy, "BN_GPU_MOE_ROUTE_PROFILE_EVERY",
              default_every)
        : default_every;
}

int bn_gpu_policy_profile_level(const BnBackendRuntimePolicy *policy) {
    const char *profile =
        bn_backend_runtime_policy_get(policy, "BN_GPU_PROFILE");
    return profile ? atoi(profile) : 0;
}
