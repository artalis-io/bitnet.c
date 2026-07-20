#include "transformer_cpu_backend_internal.h"
#include "backend_quant.h"
#include "model_arch.h"

#include <stdlib.h>

static int cpu_env_enabled(const char *name, const char *compat_name) {
    return getenv(name) != NULL ||
           (compat_name != NULL && getenv(compat_name) != NULL);
}

static int cpu_reference_dot_env_enabled(void) {
    return cpu_env_enabled("BN_CPU_REFERENCE_DOT", "BN_CPU_LLAMA_DOT");
}

static int cpu_reference_q4_dot_env_enabled(void) {
    return cpu_env_enabled("BN_CPU_REFERENCE_Q4_DOT",
                           "BN_CPU_LLAMA_Q4_DOT");
}

int bn_transformer_cpu_prepared_qweights_enabled(void) {
    return getenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS") == NULL;
}

const char *bn_transformer_cpu_debug_dump_path(void) {
    const char *path = getenv("BN_DUMP_LAYER_INP");
    return path && path[0] ? path : NULL;
}

int bn_transformer_cpu_debug_dump_pos_selected(int pos) {
    const char *pos_env = getenv("BN_DUMP_LAYER_POS");
    return !pos_env || !pos_env[0] || atoi(pos_env) == pos;
}

int bn_transformer_cpu_debug_dump_heads_enabled(void) {
    const char *enabled = getenv("BN_DUMP_ALL_HEADS");
    return enabled && enabled[0];
}

int bn_transformer_cpu_fused_kquant_gateup_silu_allowed(void) {
    return !cpu_reference_dot_env_enabled() &&
           !cpu_reference_q4_dot_env_enabled();
}

int bn_transformer_cpu_can_fused_kquant_gateup_silu(int gate_type, int up_type) {
    return bn_transformer_cpu_fused_kquant_gateup_silu_allowed() &&
           bn_backend_quant_cpu_fused_kquant_gateup_silu(gate_type, up_type);
}

int bn_transformer_cpu_can_prepared_kquant_pair(const BnCPUBackendOps *ops,
                                       int left_type,
                                       int right_type) {
    return ops && ops->supports_prepared_kquant &&
           bn_backend_quant_supports_prepared_kquant(left_type) &&
           bn_backend_quant_supports_prepared_kquant(right_type);
}

int bn_transformer_cpu_can_prepared_kquant_triple(const BnCPUBackendOps *ops,
                                         int first_type,
                                         int second_type,
                                         int third_type) {
    return bn_transformer_cpu_can_prepared_kquant_pair(ops, first_type, second_type) &&
           bn_backend_quant_supports_prepared_kquant(third_type);
}

int bn_transformer_cpu_route_prepared_kquant_pair_enabled(
    const BnCPUBackendOps *ops,
    const BnGPUBackend *gpu,
    int dim,
    int left_type,
    int right_type) {
    return !gpu &&
           dim % BN_QK_K == 0 &&
           bn_transformer_cpu_can_prepared_kquant_pair(ops, left_type, right_type);
}

int bn_transformer_cpu_route_prepared_kquant_triple_enabled(
    const BnCPUBackendOps *ops,
    const BnGPUBackend *gpu,
    int dim,
    int first_type,
    int second_type,
    int third_type) {
    return !gpu &&
           dim % BN_QK_K == 0 &&
           bn_transformer_cpu_can_prepared_kquant_triple(ops, first_type, second_type,
                                                third_type);
}

int bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
    const BnGPUBackend *gpu,
    const BnFFNPlan *ffn_plan,
    int dim,
    int gate_type,
    int up_type) {
    return !gpu &&
           ffn_plan &&
           !ffn_plan->scalar_exact_activation &&
           bn_model_arch_activation_uses_silu_path(ffn_plan->activation) &&
           dim % 32 == 0 &&
           bn_transformer_cpu_can_fused_kquant_gateup_silu(gate_type, up_type);
}

int bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
    const BnGPUBackend *gpu,
    const BnFFNPlan *ffn_plan) {
    return gpu &&
           gpu->dense_ffn &&
           ffn_plan &&
           ffn_plan->has_gate &&
           !ffn_plan->has_sub_norm &&
           bn_model_arch_activation_uses_silu_path(ffn_plan->activation);
}

int bn_transformer_cpu_activation_is_relu2(int activation) {
    return bn_model_arch_activation_is_relu2(activation);
}

int bn_transformer_cpu_activation_is_gelu(int activation) {
    return bn_model_arch_activation_is_gelu(activation);
}

int bn_transformer_cpu_activation_uses_silu_path(int activation) {
    return bn_model_arch_activation_uses_silu_path(activation);
}

uint32_t bn_transformer_cpu_float_kquant_task_flags(int enabled) {
    return enabled ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT : 0u;
}

BnTransformerCPUPostNormPolicy
bn_transformer_cpu_attention_post_norm_policy(int uses_attention_post_norm,
                                              int has_attn_post_norm) {
    BnTransformerCPUPostNormPolicy policy = {0};
    policy.apply = uses_attention_post_norm && has_attn_post_norm;
    return policy;
}

BnTransformerCPUPostNormPolicy
bn_transformer_cpu_ffn_post_norm_policy(int uses_ffn_post_norm,
                                        int has_ffn_post_norm) {
    BnTransformerCPUPostNormPolicy policy = {0};
    policy.apply = uses_ffn_post_norm && has_ffn_post_norm;
    return policy;
}

BnTransformerCPULayerOutputScalePolicy
bn_transformer_cpu_layer_output_scale_policy(int uses_layer_output_scale,
                                             int has_layer_output_scale) {
    BnTransformerCPULayerOutputScalePolicy policy = {0};
    policy.apply = uses_layer_output_scale && has_layer_output_scale;
    return policy;
}

void bn_transformer_cpu_quant_matvec_batch_gpu_buffers(
    const BnMatvecTask *tasks,
    const void **buffers,
    int n_tasks,
    const float *x,
    int8_t *quantized_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu) {
    bn_backend_quant_matvec_batch_gpu_buf(tasks, buffers, n_tasks, x,
                                          quantized_buf, pool, gpu);
}
