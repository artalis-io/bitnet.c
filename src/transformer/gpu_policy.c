#include "gpu_internal.h"
#include "backend_model.h"
#include "transformer_kv_internal.h"
#include "transformer_plan_internal.h"
#include "gpu_policy.h"
#include "../gpu_shader_ir_internal.h"
#include "../gpu_quant_lowering_internal.h"
#include "../moe_internal.h"
#include "backend_layout.h"
#include "backend_quant.h"
#include "model_internal.h"
#include <stdio.h>
#include <stdlib.h>

int bn_transformer_gpu_graph_op_capacity(const BnConfig *c) {
    /* Max ops per batch. MoE/SSM flush between layers, so single-layer max
     * suffices. Approximate flush batch budget:
     * - Attention: ~20 (QKV + norms + RoPE + GQA + sigmoid + Wo + resid)
     * - SSM: ~16 (QKV + Z + conv + splits + L2norm + alpha/beta + delta + gate + out + resid)
     * - MoE: K*5 + shared(5) + residual + rmsnorm = up to BN_MAX_MOE_K*5 + 7
     */
    return 80 * c->n_layers + 5 * BN_MAX_MOE_K + 100;
}

int bn_transformer_gpu_has_cap(const BnGPUBackend *gpu, uint32_t cap) {
    return bn_gpu_backend_has_cap(gpu, cap);
}

int bn_transformer_gpu_can_native_quant_qkv(int q_type, int k_type, int v_type) {
    return bn_backend_quant_can_gpu_native(q_type) &&
           bn_backend_quant_can_gpu_native(k_type) &&
           bn_backend_quant_can_gpu_native(v_type);
}

int bn_transformer_gpu_can_stack_same_quant_format_qk(int q_type, int k_type) {
    return bn_backend_quant_same_quant_format_pair_stackable(q_type, k_type);
}

int bn_transformer_gpu_can_stack_same_quant_format_qk_weights(const BnQWeight *q,
                                                              const BnQWeight *k,
                                                              int q_dim,
                                                              int kv_dim) {
    return q && k &&
           q->rows == q_dim &&
           k->rows == kv_dim &&
           q->cols == k->cols &&
           bn_transformer_gpu_can_stack_same_quant_format_qk(q->type, k->type);
}

int bn_transformer_gpu_can_stack_same_quant_format_gateup(const BnQWeight *gate,
                                                          const BnQWeight *up) {
    return gate && up &&
           gate->rows == up->rows &&
           gate->cols == up->cols &&
           bn_backend_quant_same_quant_format_pair_stackable(gate->type,
                                                             up->type);
}

int bn_transformer_gpu_can_matvec_split(const BnGPUBackend *gpu,
                                        int tensor_type) {
    uint32_t cap = bn_backend_quant_gpu_split_cap(tensor_type);
    return cap != 0 && bn_transformer_gpu_has_cap(gpu, cap);
}

int bn_transformer_gpu_can_fused_gateup_silu(const BnGPUBackend *gpu,
                                             int tensor_type,
                                             int act_type) {
    if (!bn_transformer_gpu_fused_gateup_silu_policy_allows(gpu, tensor_type))
        return 0;
    uint32_t cap = bn_backend_quant_gpu_fused_gateup_silu_cap(tensor_type);
    return cap != 0 &&
           bn_transformer_gpu_activation_uses_silu_path(act_type) &&
           bn_transformer_gpu_has_cap(gpu, cap);
}

int bn_transformer_gpu_can_fused_gateup_silu_pair(const BnGPUBackend *gpu,
                                                  int gate_type,
                                                  int up_type,
                                                  int act_type) {
    uint32_t gate_cap = bn_backend_quant_gpu_fused_gateup_silu_cap(gate_type);
    uint32_t up_cap = bn_backend_quant_gpu_fused_gateup_silu_cap(up_type);
    return gate_cap != 0 && gate_cap == up_cap &&
           bn_transformer_gpu_can_fused_gateup_silu(gpu, gate_type, act_type);
}

int bn_transformer_gpu_can_borrowed_pair_gateup_silu(
    const BnGPUBackend *gpu,
    const BnConfig *config,
    int gate_type,
    int up_type,
    int act_type) {
    return bn_model_transformer_policy_has_auxiliary_prediction_blocks(config) &&
           gate_type == up_type &&
           bn_backend_quant_supports_borrowed_pair_fused_gateup(gate_type) &&
           bn_transformer_gpu_activation_uses_silu_path(act_type) &&
           bn_transformer_gpu_has_cap(
               gpu, BN_GPU_CAP_NATIVE_QUANT_FUSED_GATEUP_SILU);
}

int bn_transformer_gpu_activation_uses_silu_path(int activation) {
    return bn_model_activation_uses_silu_path(activation);
}

int bn_transformer_gpu_activation_is_relu2(int activation) {
    return bn_model_activation_is_relu2(activation);
}

BnGPUIRActivationKind bn_transformer_gpu_ffn_activation_kind(int activation) {
    if (bn_transformer_gpu_activation_is_relu2(activation))
        return BN_GPU_IR_ACTIVATION_RELU2;
    if (bn_model_activation_is_gelu(activation))
        return BN_GPU_IR_ACTIVATION_GELU;
    return BN_GPU_IR_ACTIVATION_SILU;
}

float bn_transformer_gpu_norm_epsilon(const BnConfig *c) {
    return bn_transformer_norm_epsilon(c);
}

int bn_transformer_gpu_can_gateup_split_activation(const BnGPUBackend *gpu,
                                                   int tensor_type,
                                                   int act_type) {
    return bn_transformer_gpu_can_matvec_split(gpu, tensor_type) &&
           bn_backend_quant_can_gpu_gateup_split_activation(tensor_type,
                                                           act_type);
}

int bn_transformer_gpu_dense_ffn_fast_path_available(
    const BnGPUBackend *gpu,
    const BnFFNPlan *ffn_plan) {
    return bn_gpu_backend_can_dense_ffn(gpu) &&
           ffn_plan &&
           ffn_plan->has_gate &&
           !ffn_plan->has_sub_norm &&
           bn_transformer_gpu_activation_uses_silu_path(
               ffn_plan->activation);
}

int bn_transformer_gpu_dense_ffn_fast_path_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *x,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    return bn_gpu_backend_dense_ffn(gpu, out, gate_buf, up_buf, down_buf, x,
                                    dim, hidden_dim, gate_type, up_type,
                                    down_type, act_type);
}

uint32_t bn_transformer_gpu_matvec_kquant_dot_flags(int tensor_type,
                                                 int enabled) {
    return bn_backend_quant_gpu_matvec_kquant_dot_flag(tensor_type, enabled);
}

uint32_t bn_transformer_gpu_matvec_native_quant_flags(int tensor_type,
                                                   int enabled) {
    return bn_backend_quant_gpu_matvec_native_quant_flag(tensor_type, enabled);
}

uint32_t bn_transformer_gpu_matvec_quant_dot_flags(int tensor_type,
                                                   int enabled) {
    return bn_transformer_gpu_matvec_kquant_dot_flags(tensor_type, enabled) |
           bn_transformer_gpu_matvec_native_quant_flags(tensor_type, enabled);
}

uint32_t bn_transformer_gpu_matvec_block_q8_activation_flags(
    int tensor_type,
    int enabled) {
    return bn_backend_quant_gpu_matvec_block_q8_activation_flag(
        tensor_type, enabled);
}

uint32_t bn_transformer_gpu_matvec_reference_kquant_flags(int tensor_type,
                                                         int enabled) {
    return bn_backend_quant_gpu_matvec_reference_kquant_flag(tensor_type,
                                                            enabled);
}

uint32_t bn_transformer_gpu_moe_route_raw_compare_matvec_flags(int tensor_type) {
    return bn_transformer_gpu_matvec_kquant_dot_flags(tensor_type, 1);
}

uint32_t bn_transformer_gpu_moe_expert_projection_matvec_flags(
    const BnMoEExpertMap *map,
    int proj,
    int use_quant_dot) {
    if (!map)
        return 0;
    switch (proj) {
    case 0:
        return bn_transformer_gpu_matvec_quant_dot_flags(map->gate_type,
                                                         use_quant_dot);
    case 1:
        return bn_transformer_gpu_matvec_quant_dot_flags(map->up_type,
                                                         use_quant_dot);
    case 2:
        return bn_transformer_gpu_matvec_quant_dot_flags(map->down_type,
                                                         use_quant_dot);
    default:
        return 0;
    }
}

int bn_transformer_gpu_float_buffer_type(void) {
    return bn_backend_quant_gpu_float_buffer_type();
}

uint32_t bn_transformer_gpu_reference_silu_flags(int tensor_type,
                                             int use_silu) {
    return use_silu && bn_backend_quant_gpu_requires_reference_silu(tensor_type)
        ? BN_GPU_OP_FLAG_REFERENCE_SILU
        : 0u;
}

uint32_t bn_transformer_gpu_reference_activation_flags(
    int reference_activation) {
    return reference_activation ? BN_GPU_OP_FLAG_REFERENCE_ACTIVATION : 0u;
}

uint32_t bn_transformer_gpu_reference_block_accumulation_flags(int enabled) {
    return enabled ? BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION : 0u;
}

uint32_t bn_transformer_gpu_reference_silu_active_flags(int reference_silu) {
    return reference_silu > 0 ? BN_GPU_OP_FLAG_REFERENCE_SILU : 0u;
}

int bn_transformer_gpu_prefers_gateup_split(int tensor_type) {
    return bn_backend_quant_gpu_prefers_gateup_split(tensor_type);
}

int bn_transformer_gpu_dense_ffn_prefers_gateup_split(
    const BnConfig *c,
    int gate_type) {
    return bn_transformer_gpu_prefers_gateup_split(gate_type) &&
           bn_transformer_uses_hybrid_moe(c);
}

int bn_transformer_gpu_same_quant_format_pair_stackable(int left_type,
                                                        int right_type) {
    return bn_backend_quant_same_quant_format_pair_stackable(left_type,
                                                            right_type);
}

int bn_transformer_gpu_shared_kquant_gateup_dot_eligible(int gate_type,
                                                        int up_type,
                                                        int cols) {
    return cols % 256 == 0 &&
           bn_backend_quant_moe_routed_kquant_gateup(gate_type, up_type);
}

int bn_transformer_gpu_can_flash_attn(const BnGPUBackend *gpu) {
    return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_FLASH_ATTN);
}

int bn_transformer_gpu_can_layerwise_rope(const BnGPUBackend *gpu) {
    return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_LAYERWISE_ROPE);
}

int bn_transformer_gpu_uses_small_dense_shape(const BnConfig *c) {
    return bn_transformer_uses_small_dense_shape(c);
}

int bn_transformer_gpu_uses_large_dense_shape(const BnConfig *c) {
    return bn_transformer_uses_large_dense_shape(c);
}

int bn_transformer_gpu_uses_per_layer_embedding(const BnConfig *c) {
    return bn_transformer_uses_per_layer_embedding(c);
}

int bn_transformer_gpu_uses_hybrid_ssm(const BnConfig *c) {
    return bn_transformer_uses_hybrid_ssm(c);
}

int bn_transformer_gpu_uses_large_dense_hybrid_ssm(const BnConfig *c) {
    return bn_transformer_uses_large_dense_hybrid_ssm(c);
}

int bn_transformer_gpu_uses_non_hybrid_moe(const BnConfig *c) {
    return bn_transformer_uses_non_hybrid_moe(c);
}

int bn_transformer_gpu_uses_moe(const BnConfig *c) {
    return bn_transformer_uses_moe(c);
}

int bn_transformer_gpu_uses_dense_attention_only(const BnConfig *c) {
    return bn_transformer_uses_dense_attention_only(c);
}

int bn_transformer_gpu_uses_small_dense_native_quant_shape(
    const BnConfig *c) {
    return bn_transformer_uses_small_dense_native_quant_shape(c);
}

BnTransformerGPULayerKindPolicy
bn_transformer_gpu_layer_kind_policy(const BnLayerWeights *lw) {
    BnTransformerGPULayerKindPolicy policy = {0};
    policy.uses_moe = bn_transformer_moe_layer_has_router(lw);
    return policy;
}

int bn_transformer_gpu_should_upload_ssm_state(const BnConfig *c) {
    return bn_transformer_gpu_uses_hybrid_ssm(c);
}

int bn_transformer_gpu_requires_float_kquant(const BnConfig *c) {
    return bn_transformer_requires_float_kquant_fallback(c);
}

int bn_transformer_gpu_dense_batch_prefill_shape_allowed_for_backend(
    const BnConfig *c,
    int supports_large_dense_batch_prefill) {
    return bn_transformer_dense_batch_prefill_shape_allowed(
        c, supports_large_dense_batch_prefill);
}

int bn_transformer_gpu_dense_logits_argmax_shape_allowed(
    const BnConfig *c,
    int logits_rows) {
    return bn_transformer_dense_logits_argmax_shape_allowed(c, logits_rows);
}

int bn_transformer_gpu_moe_logits_mmvq_argmax_shape_allowed(
    const BnConfig *c,
    int logits_cols) {
    return bn_transformer_moe_logits_mmvq_argmax_shape_allowed(c,
                                                               logits_cols);
}

int bn_transformer_gpu_requires_layerwise_rope(const BnConfig *c,
                                               const BnWeights *w) {
    return c && w &&
           bn_transformer_gpu_uses_per_layer_embedding(c) &&
           w->rope_freqs != NULL;
}

BnBackendPlacement bn_transformer_gpu_backend_placement(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_backend_placement(gpu);
}

int bn_transformer_gpu_prefill_ssm_layer_disabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_ssm_layer_disabled(gpu);
}

int bn_transformer_gpu_fused_gateup_silu_policy_allows(
    const BnGPUBackend *gpu,
    int tensor_type) {
    return bn_gpu_policy_fused_gateup_silu_allowed(gpu, tensor_type);
}

int bn_transformer_gpu_small_dense_native_quant_fused_gateup_enabled(
    const BnGPUBackend *gpu, int use_small_dense_native_quant) {
    return use_small_dense_native_quant &&
           bn_gpu_policy_small_dense_native_quant_fused_gateup_enabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_gateup_split_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_gateup_split_enabled(gpu);
}

int bn_transformer_gpu_small_dense_native_quant_down_enabled(
    const BnGPUBackend *gpu, int use_small_dense_native_quant_down) {
    return use_small_dense_native_quant_down &&
           bn_gpu_policy_small_dense_native_quant_ffn_down_enabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_qkv_split_enabled(
    const BnGPUBackend *gpu, int use_small_dense_native_quant) {
    (void)use_small_dense_native_quant;
    return bn_gpu_policy_qkv_split_enabled(gpu);
}

int bn_transformer_gpu_qk_split_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_qkv_split_enabled(gpu);
}

int bn_transformer_gpu_qkv_split_debug_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_qkv_split_debug_enabled(gpu);
}

int bn_transformer_gpu_ssm_qkvz_split_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_ssm_qkvz_split_enabled(gpu);
}

int bn_transformer_gpu_ssm_ab_stack_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_ssm_ab_stack_enabled(gpu);
}

int bn_transformer_gpu_split_residual_rmsnorm_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_split_residual_rmsnorm_enabled(gpu);
}

int bn_transformer_gpu_shared_kquant_dot_enabled(
    const BnGPUBackend *gpu, int eligible) {
    return eligible &&
           bn_gpu_policy_shared_kquant_dot_enabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_shared_expert_prefers_gateup_split(int gate_type) {
    return bn_transformer_gpu_prefers_gateup_split(gate_type);
}

int bn_transformer_gpu_shared_expert_gate_enabled(
    const BnGPUBackend *gpu, int eligible) {
    return eligible &&
           bn_gpu_policy_shared_expert_gate_enabled(
               gpu ? &gpu->runtime_policy : NULL);
}

BnTransformerGPUSharedExpertGatePolicy
bn_transformer_gpu_shared_expert_gate_policy(const BnLayerWeights *lw) {
    BnTransformerGPUSharedExpertGatePolicy policy = {0};
    BnTransformerMoESharedExpertGatePolicy gate_policy =
        bn_transformer_moe_shared_expert_gate_policy(lw);
    policy.has_gate_vector = gate_policy.has_gate_vector;
    return policy;
}

int bn_transformer_gpu_shared_expert_path_available(
    const BnLayerWeights *lw,
    const BnTransformerGPUMoESharedResources *shared) {
    BnTransformerGPUMoESharedProjectionInfo info;
    return lw &&
           shared &&
           bn_transformer_gpu_resolve_moe_shared_projection_info(&info, lw) &&
           shared->shared_gate;
}

int bn_transformer_gpu_shared_expert_gate_available(
    const BnLayerWeights *lw,
    const BnTransformerGPUMoESharedResources *shared) {
    BnTransformerGPUSharedExpertGatePolicy gate_policy =
        bn_transformer_gpu_shared_expert_gate_policy(lw);
    return lw &&
           shared &&
           gate_policy.has_gate_vector &&
           shared->shared_expert_gate;
}

BnTransformerGPUSharedExpertGateupPolicy
bn_transformer_gpu_shared_expert_gateup_policy(
    const BnGPUBackend *gpu,
    const BnLayerWeights *lw,
    const BnTransformerGPUMoESharedResources *shared) {
    BnTransformerGPUSharedExpertGateupPolicy policy = {0};
    if (!bn_transformer_gpu_shared_expert_path_available(lw, shared))
        return policy;

    BnTransformerGPUMoESharedProjectionInfo info;
    if (!bn_transformer_gpu_resolve_moe_shared_projection_info(&info, lw))
        return policy;
    int shared_kquant_dot_eligible =
        bn_transformer_gpu_shared_kquant_gateup_dot_eligible(
            info.gate_type, info.up_type, info.gate_cols);
    policy.use_kquant_dot =
        bn_transformer_gpu_shared_kquant_dot_enabled(
            gpu, shared_kquant_dot_eligible);

    int prefer_shared_gateup_split =
        bn_transformer_gpu_shared_expert_prefers_gateup_split(
            info.gate_type);
    BnMoESharedExpertWeights weights;
    if (!bn_moe_shared_expert_projection_weights(&weights, lw))
        return policy;
    policy.use_fused_gateup =
        !prefer_shared_gateup_split &&
        shared->shared_gateup_stacked &&
        bn_transformer_gpu_can_stack_same_quant_format_gateup(
            weights.gate, weights.up) &&
        bn_transformer_gpu_can_fused_gateup_silu(
            shared->gpu, info.gate_type, 0);
    policy.use_gateup_split =
        !policy.use_fused_gateup &&
        bn_transformer_gpu_gateup_split_enabled(shared->gpu) &&
        shared->shared_gateup_stacked &&
        bn_transformer_gpu_can_matvec_split(
            shared->gpu, info.gate_type);
    return policy;
}

uint32_t bn_transformer_gpu_moe_gateup_task_flags(const BnConfig *c) {
    return bn_moe_float_kquant_gateup_fallback_task_flags(c);
}

BnTransformerGPUMoEActivationPolicy
bn_transformer_gpu_moe_activation_policy(const BnConfig *c) {
    BnTransformerGPUMoEActivationPolicy policy = {0};
    BnMoEExecutionPolicy exec_policy = bn_moe_execution_policy(c);
    policy.activation = exec_policy.activation;
    policy.uses_reference_silu = exec_policy.uses_reference_silu;
    policy.uses_reference_ffn_activation =
        exec_policy.uses_reference_ffn_activation;
    policy.uses_dense_residual_branch =
        exec_policy.uses_dense_residual_branch;
    return policy;
}

BnTransformerGPUMoESharedExpertShapePolicy
bn_transformer_gpu_moe_shared_expert_shape_policy(const BnConfig *c) {
    BnTransformerGPUMoESharedExpertShapePolicy policy = {0};
    BnTransformerMoESharedExpertShapePolicy shared_policy =
        bn_transformer_moe_shared_expert_shape_policy(c, NULL);
    policy.hidden_dim = shared_policy.hidden_dim;
    return policy;
}

BnTransformerGPUMoEGateupSplitLayoutPolicy
bn_transformer_gpu_moe_gateup_split_layout_policy(
    const BnMoEExpertMap *map) {
    BnTransformerGPUMoEGateupSplitLayoutPolicy policy = {0};
    policy.supported = bn_transformer_moe_supports_gateup_split_layout(map);
    return policy;
}

int bn_transformer_gpu_prefill_quant_matmul_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_matmul(gpu);
}

int bn_transformer_gpu_prefill_quant_matmul_batch_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_matmul_batch(gpu);
}

int bn_transformer_gpu_prefill_quant_matmul_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *buf,
    const float *X,
    int rows,
    int cols,
    int n_tokens,
    int tensor_type) {
    if (!bn_transformer_gpu_prefill_quant_matmul_backend_available(gpu))
        return -1;
    return bn_gpu_backend_matmul(gpu, out, buf, X, rows, cols, n_tokens,
                                 tensor_type);
}

int bn_transformer_gpu_prefill_quant_matmul_batch_backend_run(
    BnGPUBackend *gpu,
    const BnGPUMatvecOp *ops,
    int n_ops,
    const float *X,
    int n_tokens,
    int cols) {
    if (!bn_transformer_gpu_prefill_quant_matmul_batch_backend_available(gpu))
        return -1;
    return bn_gpu_backend_matmul_batch(gpu, ops, n_ops, X, n_tokens, cols);
}

int bn_transformer_gpu_prefill_dense_ffn_batch_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_dense_ffn_batch(gpu);
}

int bn_transformer_gpu_prefill_dense_ffn_batch_norm_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_dense_ffn_batch_norm(gpu);
}

int bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_dense_ffn_batch_norm_resid(gpu);
}

int bn_transformer_gpu_prefill_dense_ffn_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    if (!bn_transformer_gpu_prefill_dense_ffn_batch_backend_available(gpu))
        return -1;
    return bn_gpu_backend_dense_ffn_batch(gpu, out, gate_buf, up_buf,
                                          down_buf, X, n_tokens, dim,
                                          hidden_dim, gate_type, up_type,
                                          down_type, act_type);
}

int bn_transformer_gpu_prefill_dense_ffn_batch_norm_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps) {
    if (!bn_transformer_gpu_prefill_dense_ffn_batch_norm_backend_available(gpu))
        return -1;
    return bn_gpu_backend_dense_ffn_batch_norm(
        gpu, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens, dim,
        hidden_dim, gate_type, up_type, down_type, act_type, norm_eps);
}

int bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    float norm_eps) {
    if (!bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_available(
            gpu))
        return -1;
    return bn_gpu_backend_dense_ffn_batch_norm_resid(
        gpu, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens, dim,
        hidden_dim, gate_type, up_type, down_type, act_type, norm_eps);
}

int bn_transformer_gpu_prefill_qkv_attention_wo_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_prefill_qkv_attention_wo(gpu);
}

int bn_transformer_gpu_prefill_qkv_attention_wo_norm_resid_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_prefill_qkv_attention_wo_norm_resid(gpu);
}

int bn_transformer_gpu_prefill_attention_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_prefill_attention(gpu);
}

int bn_transformer_gpu_prefill_attention_wo_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_prefill_attention_wo(gpu);
}

int bn_transformer_gpu_prefill_qkv_attention_wo_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale) {
    if (!bn_transformer_gpu_prefill_qkv_attention_wo_backend_available(gpu))
        return -1;
    return bn_gpu_backend_prefill_qkv_attention_wo(
        gpu, out, qk_buf, wv_buf, wo_buf, q_norm_buf, k_norm_buf, X, K_out,
        V_out, n_tokens, dim, n_heads, n_kv_heads, head_size, kv_mul, kv_dim,
        qk_rows, qk_type, wv_rows, wv_type, wo_rows, wo_cols, wo_type,
        qk_norm_per_head, norm_eps, pos0, rope_dims, attention_scale);
}

int bn_transformer_gpu_prefill_qkv_attention_wo_norm_resid_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *attn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    float attention_scale) {
    if (!bn_transformer_gpu_prefill_qkv_attention_wo_norm_resid_backend_available(
            gpu))
        return -1;
    return bn_gpu_backend_prefill_qkv_attention_wo_norm_resid(
        gpu, out, qk_buf, wv_buf, wo_buf, attn_norm_buf, q_norm_buf,
        k_norm_buf, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads,
        head_size, kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type,
        wo_rows, wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, attention_scale);
}

int bn_transformer_gpu_prefill_attention_backend_run(
    BnGPUBackend *gpu,
    float *out,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    float attention_scale) {
    if (!bn_transformer_gpu_prefill_attention_backend_available(gpu))
        return -1;
    return bn_gpu_backend_prefill_attention(
        gpu, out, Q, K, V, n_tokens, n_heads, n_kv_heads, head_size, kv_mul,
        kv_dim, attention_scale);
}

int bn_transformer_gpu_prefill_attention_wo_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *wo_buf,
    const float *Q,
    const float *K,
    const float *V,
    int n_tokens,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int wo_rows,
    int wo_cols,
    int wo_type,
    float attention_scale) {
    if (!bn_transformer_gpu_prefill_attention_wo_backend_available(gpu))
        return -1;
    return bn_gpu_backend_prefill_attention_wo(
        gpu, out, wo_buf, Q, K, V, n_tokens, n_heads, n_kv_heads, head_size,
        kv_mul, kv_dim, wo_rows, wo_cols, wo_type, attention_scale);
}

int bn_transformer_gpu_moe_gateup_split_supported(
    const BnGPUBackend *gpu,
    const BnMoEExpertMap *map,
    int split_op_code) {
    if (!map || !bn_gpu_quant_split_op_is_asymmetric_kquant(split_op_code))
        return 0;
    BnTransformerGPUMoEGateupSplitLayoutPolicy layout =
        bn_transformer_gpu_moe_gateup_split_layout_policy(map);
    return bn_transformer_gpu_can_matvec_split(gpu, map->gate_type) &&
           bn_transformer_gpu_same_quant_format_pair_stackable(map->up_type,
                                                       map->gate_type) &&
           layout.supported;
}

int bn_transformer_gpu_matvec_split_op_code(int tensor_type) {
    return bn_gpu_quant_split_op_code(tensor_type);
}

int bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *gate,
    const BnQWeight *up,
    int activation,
    int split_op_code) {
    if (!gate || !up ||
        !bn_transformer_gpu_activation_uses_silu_path(activation) ||
        !bn_gpu_quant_split_op_is_asymmetric_kquant(split_op_code))
        return 0;
    return bn_transformer_gpu_can_stack_same_quant_format_gateup(gate, up) &&
           bn_transformer_gpu_can_matvec_split(gpu, gate->type);
}

int bn_transformer_gpu_packed_qkv_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *qkv,
    int use_packed_qkv,
    int kv_cache_uses_fp16_rows,
    int split_op_code) {
    return qkv && use_packed_qkv && !kv_cache_uses_fp16_rows &&
           bn_gpu_quant_split_op_is_deinterleaved_kquant(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, qkv->type);
}

int bn_transformer_gpu_qkv_split_standard_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code) {
    return q && bn_gpu_quant_split_op_is_standard(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_qkv_split_native_quant_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code) {
    return q && bn_gpu_quant_split_op_is_native_quant(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_qkv_split_packed_kquant_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code) {
    return q && bn_gpu_quant_split_op_is_deinterleaved_kquant(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_qk_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    const BnQWeight *k,
    int q_dim,
    int kv_dim,
    int split_op_code) {
    if (!q || !k || !bn_gpu_quant_split_op_known(split_op_code))
        return 0;
    return q->rows == q_dim &&
           k->rows == kv_dim &&
           q->cols == k->cols &&
           bn_transformer_gpu_same_quant_format_pair_stackable(q->type, k->type) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_ssm_qkvz_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *qkv,
    int split_op_code) {
    return qkv && bn_gpu_quant_split_op_known(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, qkv->type);
}

int bn_transformer_gpu_can_stack_same_quant_format_alpha_beta(
    const BnQWeight *alpha,
    const BnQWeight *beta) {
    return alpha && beta &&
           alpha->rows == beta->rows &&
           alpha->cols == beta->cols &&
           bn_transformer_gpu_same_quant_format_pair_stackable(alpha->type,
                                                       beta->type);
}

int bn_transformer_gpu_logits_needs_cpu_fallback(
    const BnGPUBackend *gpu,
    const BnTransformerGPULogitResources *logits) {
    if (!gpu || !logits || !logits->cpu_weight)
        return 0;

    size_t max_storage_binding =
        bn_gpu_policy_max_storage_binding_bytes(
            &gpu->runtime_policy,
            bn_gpu_backend_max_storage_binding_size(gpu));

    return bn_backend_layout_qweight_data_size(logits->cpu_weight) >
           max_storage_binding;
}

int bn_transformer_gpu_all_active_two_kquant_moe_layer(
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim) {
    if (!lw || !bn_transformer_moe_uses_all_active_two_route(c, dim))
        return 0;
    return bn_transformer_gpu_moe_routed_kquant_down_allowed(
        &lw->moe.expert_map, 0);
}

int bn_transformer_gpu_all_active_two_kquant_moe_layer_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim) {
    return bn_gpu_policy_backend_all_active_two_kquant_moe_supported(gpu) &&
           bn_transformer_gpu_all_active_two_kquant_moe_layer(c, lw, dim);
}

int bn_transformer_gpu_all_active_two_kquant_moe_model(const BnConfig *c,
                                           const BnWeights *w) {
    if (!w || !bn_transformer_moe_uses_configured_all_active_two_route(c))
        return 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(lw);
        if (!layer_kind.uses_moe)
            continue;
        if (bn_transformer_gpu_all_active_two_kquant_moe_layer(c, lw, c->dim))
            return 1;
    }
    return 0;
}

static int all_active_two_kquant_moe_requires_opt_in(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    const BnBackendRuntimePolicy *runtime =
        gpu ? &gpu->runtime_policy : NULL;
    return bn_transformer_gpu_all_active_two_kquant_moe_model(c, w) &&
           !bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(runtime) &&
           bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled(
               runtime);
}

static int small_dense_backend_native_by_default(
    const BnConfig *c,
    const BnWeights *w) {
    if (!c || !w || !bn_transformer_gpu_uses_small_dense_shape(c))
        return 0;
    return bn_backend_quant_dense_graph_model_supported(
        w, c, BN_BACKEND_QUANT_DENSE_GRAPH_ANY);
}

static int small_dense_backend_native_quant_by_default(
    const BnConfig *c,
    const BnWeights *w) {
    if (!c || !w || !bn_transformer_gpu_uses_small_dense_shape(c))
        return 0;
    return bn_backend_quant_dense_graph_model_supported(
        w, c, BN_BACKEND_QUANT_DENSE_GRAPH_NATIVE_QUANT);
}

int bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    const BnBackendRuntimePolicy *runtime =
        gpu ? &gpu->runtime_policy : NULL;
    return bn_transformer_gpu_all_active_two_kquant_moe_model(c, w) &&
           !bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(runtime) &&
           !bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled(
               runtime);
}

int bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_gpu_policy_backend_cpu_attention_fallback_supported(gpu) &&
           bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
               gpu, c, w);
}

int bn_transformer_gpu_small_dense_native_quant_cpu_attn_safe_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_prefill_uses_decode_for_parity(c) &&
           bn_transformer_allows_small_dense_native_quant(c) &&
           small_dense_backend_native_quant_by_default(c, w) &&
           !bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_gpu_policy_backend_cpu_attention_fallback_supported(gpu) &&
           bn_transformer_gpu_small_dense_native_quant_cpu_attn_safe_default(
               gpu, c, w);
}

int bn_transformer_gpu_small_dense_native_quant_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int small_dense_native_quant_from_layer) {
    return small_dense_native_quant_from_layer < 0 &&
           bn_gpu_policy_backend_small_dense_native_quant_supported(gpu) &&
           bn_transformer_allows_small_dense_native_quant(c) &&
           !bn_gpu_policy_small_dense_native_quant_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_small_dense_native_quant_to_layer(
    const BnConfig *c,
    int small_dense_native_quant_default,
    int small_dense_native_quant_to_layer) {
    if (!small_dense_native_quant_default || small_dense_native_quant_to_layer >= 0)
        return small_dense_native_quant_to_layer;
    return bn_transformer_small_dense_native_quant_to_layer(c);
}

int bn_transformer_gpu_small_dense_native_quant_ffn_down_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_small_dense_native_quant_supported(gpu) &&
           bn_transformer_allows_small_dense_native_quant(c) &&
           bn_gpu_policy_small_dense_native_quant_ffn_down_requested(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_large_hybrid_cpu_attn_safe_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    if (!bn_transformer_gpu_uses_large_dense_shape(c) || !w ||
        bn_gpu_policy_large_hybrid_attention_enabled(gpu) ||
        bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled(gpu))
        return 0;
    if (!bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled(gpu) &&
        !bn_gpu_policy_large_hybrid_cpu_attention_safe_forced(gpu))
        return 0;
    if (bn_transformer_gpu_uses_hybrid_ssm(c))
        return 1;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        if (bn_transformer_layer_has_attention_ssm_qkv(lw))
            return 1;
    }
    return 0;
}

int bn_transformer_gpu_large_hybrid_cpu_attn_safe_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_gpu_policy_backend_cpu_attention_fallback_supported(gpu) &&
           bn_transformer_gpu_large_hybrid_cpu_attn_safe_default(gpu, c, w);
}

int bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_reference_attention_fallback_supported(gpu) &&
           bn_model_transformer_policy_requires_reference_attention(c) &&
           !bn_model_transformer_policy_has_auxiliary_prediction_blocks(c) &&
           !bn_transformer_gpu_reference_attention_exact_enabled(gpu, c) &&
           !bn_gpu_policy_backend_reference_attention_native_graph_supported(
               gpu) &&
           !bn_gpu_policy_backend_reference_attention_token_fallback_supported(
               gpu);
}

int bn_transformer_gpu_reference_attention_no_logits_cpu_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int emit_logits) {
    return !emit_logits &&
           bn_model_transformer_policy_requires_reference_attention(c) &&
           bn_gpu_policy_backend_reference_attention_fallback_supported(gpu) &&
           bn_gpu_policy_backend_reference_attention_native_graph_supported(
               gpu) &&
           !bn_transformer_gpu_reference_attention_exact_enabled(gpu, c);
}

int bn_transformer_gpu_reference_attention_exact_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_reference_attention_supported(gpu) &&
           bn_gpu_policy_backend_reference_attention_native_graph_supported(
               gpu) &&
           bn_model_transformer_policy_requires_reference_attention(c);
}

int bn_transformer_gpu_reference_recurrent_exact_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_reference_recurrent_supported(gpu) &&
           bn_model_transformer_policy_requires_reference_recurrent(c);
}

int bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_prefill_decode_fallback_supported(gpu) &&
           bn_transformer_allows_small_dense_prefill_decode_fallback(c) &&
           bn_gpu_policy_small_dense_prefill_disabled(gpu);
}

int bn_transformer_gpu_small_dense_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_prefill_chain_supported(gpu) &&
           bn_transformer_small_dense_prefill_min_tokens(c) > 0;
}

int bn_transformer_gpu_hybrid_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_uses_hybrid_ssm(c) &&
           bn_gpu_policy_backend_prefill_chain_supported(gpu);
}

int bn_transformer_gpu_moe_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_uses_non_hybrid_moe(c) &&
           bn_gpu_policy_backend_prefill_chain_supported(gpu);
}

int bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_prefill_decode_fallback_supported(gpu) &&
           bn_transformer_gpu_uses_large_dense_hybrid_ssm(c) &&
           !bn_gpu_policy_large_hybrid_prefill_enabled(gpu);
}

int bn_transformer_gpu_backend_matvec_fallback_kept(
    const BnModel *m,
    const BnGPUBackend *gpu) {
    if (!m || !bn_gpu_policy_backend_matvec_fallback_supported(gpu) ||
        !bn_gpu_backend_can_execute(gpu))
        return 0;
    const BnConfig *c = &m->config;
    if (!bn_transformer_gpu_uses_dense_attention_only(c))
        return 0;
    if (bn_gpu_policy_small_state_native_quant_enabled(
            gpu ? &gpu->runtime_policy : NULL,
            bn_transformer_gpu_requires_float_kquant(c)))
        return 1;
    if (!bn_transformer_gpu_uses_small_dense_native_quant_shape(c))
        return 1;

    return bn_backend_quant_dense_graph_model_supported(
        &m->weights, c, BN_BACKEND_QUANT_DENSE_GRAPH_NATIVE_QUANT);
}

int bn_transformer_gpu_backend_cpu_operations_kept(
    const BnModel *m,
    const BnGPUBackend *gpu) {
    if (!m || !gpu || !bn_gpu_backend_can_moe_routed_ffn_batch(gpu))
        return 0;
    const BnBackendModel *backend = bn_model_backend(m);
    int moe_layers = 0;
    for (int layer = 0; layer < m->config.n_layers; layer++) {
        if (!bn_moe_policy_layer_has_router(&m->weights.layers[layer]))
            continue;
        moe_layers++;
        BnBackendModelMoEPrefillResidentResources resources =
            bn_backend_model_moe_prefill_resident_resources(backend, layer);
        if (!resources.valid)
            return 0;
    }
    return moe_layers > 0;
}

BnTransformerGPUMatvecFallbackPolicy
bn_transformer_gpu_matvec_fallback_policy(
    const BnModel *m,
    const BnGPUBackend *gpu) {
    BnTransformerGPUMatvecFallbackPolicy policy = {0};
    policy.keep_backend_matvec =
        bn_transformer_gpu_backend_matvec_fallback_kept(m, gpu);
    policy.keep_backend_operations =
        bn_transformer_gpu_backend_cpu_operations_kept(m, gpu);
    policy.disable_backend_matvec =
        !policy.keep_backend_matvec && !policy.keep_backend_operations;
    return policy;
}

int bn_transformer_gpu_dense_batch_prefill_shape_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_dense_batch_prefill_shape_allowed_for_backend(
        c, bn_gpu_policy_backend_dense_batch_prefill_shape_supported(gpu));
}

int bn_transformer_gpu_batch_prefill_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    if (!c)
        return 0;
    if (bn_gpu_policy_prefill_matmul_disabled(gpu))
        return 0;
    if (bn_gpu_policy_prefill_matmul_enabled(gpu))
        return 1;
    if (bn_transformer_prefill_uses_decode_for_parity(c))
        return 0;
    if (bn_transformer_kv_mode_uses_turboquant(
            bn_transformer_kv_mode(c, 1)))
        return 0;
    if (bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
            gpu, c) ||
        bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
            gpu, c))
        return 0;
    if (bn_transformer_gpu_uses_hybrid_ssm(c)) {
        return bn_gpu_policy_backend_prefill_chain_supported(gpu) &&
               bn_gpu_backend_can_prefill_ssm_layer(gpu) &&
               bn_gpu_policy_prefill_hybrid_chain_enabled(gpu) &&
               !bn_transformer_gpu_prefill_ssm_layer_disabled(gpu);
    }
    if (bn_transformer_gpu_uses_moe(c))
        return bn_gpu_policy_backend_prefill_chain_supported(gpu) &&
               bn_gpu_policy_moe_prefill_enabled(gpu);
    return bn_transformer_gpu_dense_batch_prefill_shape_allowed(gpu, c);
}

int bn_transformer_gpu_large_hybrid_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    if (!c || !bn_gpu_policy_backend_cpu_attention_fallback_supported(gpu) ||
        !bn_transformer_gpu_uses_large_dense_hybrid_ssm(c))
        return 0;
    if (bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled(gpu))
        return 1;
    return !bn_gpu_policy_large_hybrid_attention_enabled(gpu) &&
           !bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled(gpu) &&
           bn_gpu_policy_large_hybrid_cpu_attention_safe_forced(gpu);
}

int bn_transformer_gpu_large_hybrid_prefill_chain_disabled_default(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_prefill_chain_supported(gpu) &&
           bn_transformer_gpu_uses_large_dense_hybrid_ssm(c) &&
           !bn_gpu_policy_large_hybrid_prefill_chain_enabled(gpu);
}

int bn_transformer_gpu_prefill_direct_kv_allowed(
    const BnConfig *c,
    const BnWeights *w,
    const BnGPUBackend *gpu,
    int pos0,
    int n_tokens) {
    if (!c || !bn_gpu_policy_backend_prefill_chain_supported(gpu))
        return 0;
    if (bn_gpu_policy_prefill_direct_kv_disabled(gpu))
        return 0;
    if ((bn_gpu_policy_cpu_decode_fallback_requested(gpu) ||
         bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
             gpu, c, w) ||
         bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
             gpu, c, w) ||
         bn_transformer_gpu_large_hybrid_cpu_attn_fallback_enabled(
             gpu, c)) &&
        !bn_gpu_policy_prefill_direct_kv_with_cpu_fallback_enabled(gpu))
        return 0;
    if (bn_transformer_kv_host_cache_uses_fp16_rows(c) ||
        pos0 < 0 || pos0 + n_tokens > c->seq_len)
        return 0;
    return 1;
}

int bn_transformer_gpu_prefill_attention_min_tokens(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_attention_min_tokens_or_default(gpu, 16);
}

int bn_transformer_gpu_prefill_dense_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    if (bn_gpu_policy_prefill_attention_min_tokens_configured(gpu))
        return bn_transformer_gpu_prefill_attention_min_tokens(gpu);
    if (bn_gpu_policy_backend_prefill_chain_supported(gpu) && c) {
        int shape_min = bn_transformer_small_dense_prefill_min_tokens(c);
        if (shape_min > 0)
            return shape_min;
    }
    if (bn_gpu_policy_backend_prefill_chain_supported(gpu) && c)
        return 16;
    return bn_transformer_gpu_prefill_attention_min_tokens(gpu);
}

int bn_transformer_gpu_dense_ffn_batch_tokens_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens) {
    return !bn_gpu_policy_backend_prefill_chain_supported(gpu) ||
           n_tokens >=
               bn_transformer_gpu_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_prefill_moe_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    if (bn_gpu_policy_moe_prefill_min_tokens_configured(gpu))
        return bn_gpu_policy_moe_prefill_min_tokens_or_default(gpu, 1);
    if (bn_gpu_policy_backend_prefill_chain_supported(gpu) && c)
        return bn_gpu_policy_moe_prefill_min_tokens_or_default(gpu, 16);
    return bn_gpu_policy_moe_prefill_min_tokens_or_default(
        gpu, bn_transformer_gpu_prefill_dense_chain_min_tokens(c, gpu));
}

int bn_transformer_gpu_prefill_moe_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down) {
    return bn_gpu_policy_backend_prefill_chain_supported(gpu) &&
           bn_gpu_backend_can_moe_route_routed_ffn_batch_norm_resid(gpu) &&
           bn_transformer_gpu_moe_routed_ffn_batch_allowed(gpu, c) &&
           !bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
               gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_gpu_prefill_moe_ffn_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    void *ffn_norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    float norm_eps,
    int norm_topk_prob,
    float expert_weights_scale) {
    return bn_gpu_backend_moe_route_routed_ffn_batch_norm_resid(
        gpu, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, ffn_norm_buf, X, n_tokens, dim, hidden_dim,
        n_experts, k, gate_type, up_type, down_type, act_type,
        shared_hidden_dim, shared_gate_type, shared_up_type,
        shared_down_type, norm_eps, norm_topk_prob, expert_weights_scale);
}

int bn_transformer_gpu_prefill_dense_layer_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_prefill_dense_layer(gpu);
}

int bn_transformer_gpu_prefill_dense_layer_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *gate_buf,
    void *up_buf,
    void *down_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale) {
    if (!bn_transformer_gpu_prefill_dense_layer_backend_available(gpu))
        return -1;
    return bn_gpu_backend_prefill_dense_layer(
        gpu, out, qk_buf, wv_buf, wo_buf, gate_buf, up_buf, down_buf,
        attn_norm_buf, ffn_norm_buf, q_norm_buf, k_norm_buf, q_bias_buf,
        k_bias_buf, v_bias_buf, X, K_out, V_out, n_tokens, dim, hidden_dim,
        n_heads, n_kv_heads, head_size, kv_mul, kv_dim, qk_rows, qk_type,
        wv_rows, wv_type, wo_rows, wo_cols, wo_type, gate_type, up_type,
        down_type, act_type, qk_norm_per_head, norm_eps, pos0, rope_dims,
        kv_cache_off, kv_cache_stride, attention_scale);
}

int bn_transformer_gpu_prefill_moe_layer_backend_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down) {
    return bn_gpu_backend_can_prefill_moe_layer(gpu) &&
           bn_transformer_gpu_moe_routed_ffn_batch_allowed(gpu, c) &&
           !bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
               gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_gpu_prefill_moe_layer_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *qk_buf,
    void *wv_buf,
    void *wo_buf,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    void *attn_norm_buf,
    void *ffn_norm_buf,
    void *q_norm_buf,
    void *k_norm_buf,
    void *q_bias_buf,
    void *k_bias_buf,
    void *v_bias_buf,
    const float *X,
    float *K_out,
    float *V_out,
    int n_tokens,
    int dim,
    int moe_hidden_dim,
    int n_experts,
    int experts_active,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int qk_rows,
    int qk_type,
    int wv_rows,
    int wv_type,
    int wo_rows,
    int wo_cols,
    int wo_type,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type,
    int qk_norm_per_head,
    float norm_eps,
    int pos0,
    int rope_dims,
    uint32_t kv_cache_off,
    int kv_cache_stride,
    float attention_scale,
    int norm_topk_prob,
    float expert_weights_scale) {
    return bn_gpu_backend_prefill_moe_layer(
        gpu, out, qk_buf, wv_buf, wo_buf, router_buf, gate_all_buf,
        up_all_buf, down_all_buf, shared_gate_buf, shared_up_buf,
        shared_down_buf, shared_gate_weight_buf, attn_norm_buf, ffn_norm_buf,
        q_norm_buf, k_norm_buf, q_bias_buf, k_bias_buf, v_bias_buf, X, K_out,
        V_out, n_tokens, dim, moe_hidden_dim, n_experts, experts_active,
        n_heads, n_kv_heads, head_size, kv_mul, kv_dim, qk_rows, qk_type,
        wv_rows, wv_type, wo_rows, wo_cols, wo_type, gate_type, up_type,
        down_type, act_type, shared_hidden_dim, shared_gate_type,
        shared_up_type, shared_down_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, kv_cache_off, kv_cache_stride, attention_scale,
        norm_topk_prob, expert_weights_scale);
}

int bn_transformer_gpu_prefill_moe_layer_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int n_tokens) {
    return bn_transformer_gpu_prefill_moe_layer_backend_available(
               gpu, c, map, dim, allow_kquant_down) &&
           n_tokens >=
               bn_transformer_gpu_prefill_moe_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_prefill_ssm_moe_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int n_tokens) {
    return bn_gpu_policy_backend_prefill_chain_supported(gpu) &&
           bn_gpu_backend_can_prefill_ssm_layer(gpu) &&
           !bn_transformer_gpu_prefill_ssm_layer_disabled(gpu) &&
           n_tokens >=
               bn_transformer_gpu_prefill_moe_chain_min_tokens(c, gpu) &&
           bn_transformer_gpu_prefill_moe_ffn_batch_available(
               gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_gpu_prefill_ssm_layer_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_prefill_ssm_layer(gpu);
}

int bn_transformer_gpu_prefill_ssm_layer_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *wqkv_buf,
    void *wz_buf,
    void *alpha_buf,
    void *beta_buf,
    void *qkvz_stacked_buf,
    void *ab_stacked_buf,
    void *ssm_out_buf,
    void *attn_norm_buf,
    void *conv1d_buf,
    void *dt_bias_buf,
    void *a_log_buf,
    void *ssm_norm_buf,
    void *ffn_gate_buf,
    void *ffn_up_buf,
    void *ffn_down_buf,
    void *ffn_norm_buf,
    const float *X,
    int n_tokens,
    int dim,
    int qkv_dim,
    int inner_dim,
    int num_k_heads,
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int conv_kernel,
    int ssm_idx,
    int wqkv_type,
    int wz_type,
    int alpha_type,
    int beta_type,
    int out_type,
    int hidden_dim,
    int ffn_gate_type,
    int ffn_up_type,
    int ffn_down_type,
    int act_type,
    float norm_eps,
    int *did_ffn) {
    if (!bn_transformer_gpu_prefill_ssm_layer_backend_available(gpu))
        return -1;
    return bn_gpu_backend_prefill_ssm_layer(
        gpu, out, wqkv_buf, wz_buf, alpha_buf, beta_buf, qkvz_stacked_buf,
        ab_stacked_buf, ssm_out_buf, attn_norm_buf, conv1d_buf, dt_bias_buf,
        a_log_buf, ssm_norm_buf, ffn_gate_buf, ffn_up_buf, ffn_down_buf,
        ffn_norm_buf, X, n_tokens, dim, qkv_dim, inner_dim, num_k_heads,
        head_k_dim, num_v_heads, head_v_dim, conv_kernel, ssm_idx, wqkv_type,
        wz_type, alpha_type, beta_type, out_type, hidden_dim, ffn_gate_type,
        ffn_up_type, ffn_down_type, act_type, norm_eps, did_ffn);
}

int bn_transformer_gpu_prefill_ssm_dense_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens) {
    return bn_gpu_policy_backend_prefill_chain_supported(gpu) &&
           bn_gpu_backend_can_prefill_ssm_layer(gpu) &&
           !bn_transformer_gpu_prefill_ssm_layer_disabled(gpu) &&
           n_tokens >=
               bn_transformer_gpu_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_prefill_dense_chain_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_dense_chain_enabled(gpu);
}

int bn_transformer_gpu_prefill_hybrid_chain_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_prefill_hybrid_chain_enabled(gpu) &&
           !bn_transformer_gpu_large_hybrid_prefill_chain_disabled_default(
               gpu, c);
}

int bn_transformer_gpu_prefill_attention_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_attention_enabled(gpu);
}

int bn_transformer_gpu_prefill_ssm_run_chain_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_ssm_run_chain_enabled(gpu);
}

int bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_ssm_ffn_fuse_allowed(gpu);
}

int bn_transformer_gpu_prefill_moe_chain_debug_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_moe_chain_debug_enabled(gpu);
}

int bn_transformer_gpu_prefill_hybrid_chain_debug_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_prefill_hybrid_chain_debug_enabled(gpu);
}

int bn_transformer_gpu_moe_prefill_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_prefill_enabled(gpu);
}

int bn_transformer_gpu_moe_prefill_min_tokens(const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_prefill_min_tokens_or_default(gpu, 1);
}

int bn_transformer_gpu_moe_prefill_backend_available(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_backend_prefill_chain_supported(gpu);
}

int bn_transformer_gpu_moe_prefill_tokens_allowed(
    const BnGPUBackend *gpu,
    int n_tokens) {
    return bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           n_tokens >= bn_transformer_gpu_moe_prefill_min_tokens(gpu);
}

int bn_transformer_gpu_moe_cache_prefill_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_cache_prefill_enabled(gpu);
}

int bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int gpu_moe_cache_available) {
    return gpu_moe_cache_available &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_transformer_moe_uses_all_active_two_expert_set(c) &&
           bn_transformer_gpu_moe_cache_prefill_enabled(gpu);
}

int bn_transformer_gpu_moe_prefill_shared_fuse_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_prefill_shared_fuse_enabled(gpu);
}

int bn_transformer_gpu_moe_prefill_shared_batch_available(
    const BnGPUBackend *gpu,
    int n_tokens,
    int backend_available) {
    return backend_available &&
           bn_transformer_gpu_moe_prefill_tokens_allowed(gpu, n_tokens) &&
           bn_gpu_backend_can_dense_ffn_batch(gpu) &&
           bn_transformer_gpu_moe_prefill_shared_fuse_enabled(gpu);
}

int bn_transformer_gpu_moe_prefill_shared_dense_ffn_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_gpu_backend_can_dense_ffn_batch(gpu);
}

int bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int backend_available) {
    return backend_available &&
           bn_transformer_gpu_moe_has_loaded_shared_expert(c, lw) &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_transformer_gpu_moe_prefill_shared_fuse_enabled(gpu);
}

int bn_transformer_gpu_moe_route_batch_debug_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_route_batch_debug_enabled(gpu);
}

int bn_transformer_gpu_moe_prefill_route_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int backend_available) {
    return backend_available &&
           bn_transformer_moe_uses_grouped_route(c) &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_gpu_backend_can_moe_route_batch(gpu);
}

int bn_transformer_gpu_moe_prefill_route_batch_backend_run(
    BnGPUBackend *gpu,
    int *indices,
    float *weights,
    void *router_buf,
    const float *X,
    int n_tokens,
    int dim,
    int n_experts,
    int k,
    int norm_topk_prob,
    float expert_weights_scale) {
    return bn_gpu_backend_moe_route_batch(
        gpu, indices, weights, router_buf, X, n_tokens, dim, n_experts, k,
        norm_topk_prob, expert_weights_scale);
}

int bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return c &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_gpu_backend_can_moe_route_routed_ffn_batch_norm_resid(gpu) &&
           bn_transformer_gpu_moe_routed_ffn_batch_allowed(gpu, c);
}

int bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down) {
    return c &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_gpu_backend_can_moe_route_routed_ffn_batch(gpu) &&
           bn_transformer_gpu_moe_routed_ffn_batch_allowed(gpu, c) &&
           !bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
               gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_gpu_moe_prefill_routed_ffn_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *router_buf,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    int norm_topk_prob,
    float expert_weights_scale) {
    return bn_gpu_backend_moe_route_routed_ffn_batch(
        gpu, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        X, n_tokens, dim, hidden_dim, n_experts, k, gate_type, up_type,
        down_type, act_type, norm_topk_prob, expert_weights_scale);
}

int bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int prefer_cached_expert_batch) {
    return !prefer_cached_expert_batch &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_gpu_backend_can_moe_routed_ffn_batch(gpu) &&
           !bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
               gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_gpu_moe_prefill_resident_expert_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    void *gate_all_buf,
    void *up_all_buf,
    void *down_all_buf,
    const int *indices,
    const float *weights,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int act_type) {
    return bn_gpu_backend_moe_routed_ffn_batch(
        gpu, out, gate_all_buf, up_all_buf, down_all_buf, indices,
        weights, X, n_tokens, dim, hidden_dim, n_experts, k, gate_type,
        up_type, down_type, act_type);
}

int bn_transformer_gpu_moe_prefill_split_expert_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int used_resident_expert_batch) {
    return !used_resident_expert_batch &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_gpu_backend_can_moe_ffn_batch(gpu) &&
           !bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
               gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_gpu_moe_prefill_split_expert_batch_backend_run(
    BnGPUBackend *gpu,
    float *out,
    const BnGPUMoEPrefillExpert *experts,
    int n_experts,
    const int *expert_offsets,
    const int *expert_counts,
    const int *token_ids,
    const float *weights,
    const float *X,
    int n_tokens,
    int dim,
    int hidden_dim,
    int gate_type,
    int up_type,
    int down_type,
    int act_type,
    void *shared_gate_buf,
    void *shared_up_buf,
    void *shared_down_buf,
    void *shared_gate_weight_buf,
    int shared_hidden_dim,
    int shared_gate_type,
    int shared_up_type,
    int shared_down_type) {
    return bn_gpu_backend_moe_ffn_batch(
        gpu, out, experts, n_experts, expert_offsets, expert_counts,
        token_ids, weights, X, n_tokens, dim, hidden_dim, gate_type, up_type,
        down_type, act_type, shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, shared_hidden_dim, shared_gate_type,
        shared_up_type, shared_down_type);
}

int bn_transformer_gpu_moe_prefill_single_expert_batch_available(
    const BnGPUBackend *gpu,
    int n_tokens) {
    return bn_transformer_gpu_moe_prefill_tokens_allowed(gpu, n_tokens) &&
           bn_gpu_backend_can_dense_ffn_batch(gpu);
}

int bn_transformer_gpu_moe_lazy_aux_cache_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_lazy_aux_cache_enabled(gpu);
}

int bn_transformer_gpu_moe_quant_only_without_aux_cache(
    const BnGPUBackend *gpu,
    int tensor_type,
    int allow_aux_cache) {
    return bn_gpu_policy_backend_lazy_moe_aux_cache_supported(gpu) &&
           !allow_aux_cache &&
           !bn_transformer_gpu_moe_lazy_aux_cache_enabled(gpu) &&
           bn_backend_quant_supports_lazy_moe_aux_cache(tensor_type);
}

int bn_transformer_gpu_large_hybrid_prefill_disabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_large_hybrid_prefill_disabled(gpu);
}

int bn_transformer_gpu_native_quant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int tensor_type) {
    return bn_gpu_policy_backend_native_quant_logits_refine_default_supported(
               gpu) &&
           bn_backend_quant_supports_native_quant_logits_refine(tensor_type) &&
           bn_transformer_allows_small_dense_native_logit_refine(c) &&
           bn_gpu_policy_native_quant_logits_refine_requested(
               gpu ? &gpu->runtime_policy : NULL) &&
           !bn_gpu_policy_native_quant_logits_refine_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_gpu_policy_backend_all_active_two_kquant_moe_logits_refine_default_supported(
               gpu) &&
           bn_transformer_gpu_all_active_two_kquant_moe_model(c, w) &&
           bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(
               gpu ? &gpu->runtime_policy : NULL) &&
           !bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_kquant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int kquant_refine_default) {
    return bn_gpu_policy_backend_kquant_logits_refine_enabled(
        gpu, kquant_refine_default);
}

int bn_transformer_gpu_kquant_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_kquant_logits,
    int kquant_refine_default) {
    return refine_kquant_logits &&
           kquant_refine_default &&
           logits &&
           bn_backend_quant_supports_kquant_logits_refine(logits->type) &&
           logits->cpu_weight != NULL;
}

int bn_transformer_gpu_kquant_logits_refine_top(
    const BnGPUBackend *gpu, int kquant_refine_default) {
    return bn_gpu_policy_kquant_logits_refine_top_or_default(
        gpu ? &gpu->runtime_policy : NULL,
        kquant_refine_default ? 64 : 8);
}

int bn_transformer_gpu_kquant_logits_refine_blocks_per_row(int cols) {
    return bn_backend_quant_prepared_kquant_blocks_per_row(cols);
}

int bn_transformer_gpu_kquant_logits_refine_block_sums_per_row(
    int blocks_per_row) {
    return bn_backend_quant_prepared_kquant_block_sums_per_row(blocks_per_row);
}

int bn_transformer_gpu_native_quant_logits_refine_active(
    const BnGPUBackend *gpu,
    int native_quant_refine_default) {
    return bn_gpu_policy_backend_native_quant_logits_refine_enabled(
        gpu, native_quant_refine_default);
}

int bn_transformer_gpu_native_quant_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_native_quant_logits) {
    return refine_native_quant_logits &&
           logits &&
           bn_backend_quant_supports_native_quant_logits_refine(logits->type) &&
           logits->cpu_weight != NULL;
}

int bn_transformer_gpu_native_quant_logits_refine_top(
    const BnGPUBackend *gpu, int native_quant_refine_default) {
    return bn_gpu_policy_native_quant_logits_refine_top_or_default(
        gpu ? &gpu->runtime_policy : NULL,
        native_quant_refine_default ? 16 : 8);
}

BnTransformerGPULogitsRefinePolicy bn_transformer_gpu_logits_refine_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    const BnTransformerGPULogitResources *logits,
    int small_dense_native_quant_default) {
    BnTransformerGPULogitsRefinePolicy p = {0};
    p.kquant_default =
        bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
            gpu, c, w);
    p.kquant_enabled = bn_transformer_gpu_kquant_logits_refine_enabled(
        gpu, p.kquant_default);
    p.kquant_captures_xb = bn_transformer_gpu_kquant_logits_refine_captures_xb(
        logits, p.kquant_enabled, p.kquant_default);
    p.kquant_refine_top =
        bn_transformer_gpu_kquant_logits_refine_top(gpu,
                                                     p.kquant_default);

    int tensor_type = logits ? logits->type : -1;
    p.native_quant_default =
        small_dense_native_quant_default &&
        bn_transformer_gpu_native_quant_logits_refine_enabled(
            gpu, c, tensor_type);
    p.native_quant_enabled =
        bn_transformer_gpu_native_quant_logits_refine_active(
            gpu, p.native_quant_default);
    p.native_quant_captures_xb =
        bn_transformer_gpu_native_quant_logits_refine_captures_xb(
            logits, p.native_quant_enabled);
    p.native_quant_refine_top =
        bn_transformer_gpu_native_quant_logits_refine_top(
            gpu, p.native_quant_default);
    return p;
}

BnTransformerGPULogitsRefineSnapshotPolicy
bn_transformer_gpu_logits_refine_snapshot_policy(
    int need_logits,
    int want_argmax,
    const BnTransformerGPULogitsRefinePolicy *logits_refine) {
    BnTransformerGPULogitsRefineSnapshotPolicy policy = {0};
    policy.snapshot_before_logits =
        need_logits &&
        !want_argmax &&
        logits_refine &&
        logits_refine->kquant_captures_xb;
    policy.snapshot_satisfies_kquant_refine =
        policy.snapshot_before_logits;
    return policy;
}

int bn_transformer_gpu_cpu_logits_enabled(const BnGPUBackend *gpu,
                                          int gpu_logits_need_cpu) {
    return gpu_logits_need_cpu || bn_gpu_policy_cpu_logits_enabled(gpu);
}

int bn_transformer_gpu_compare_logits_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_compare_logits_enabled(gpu);
}

int bn_transformer_gpu_debug_argmax_compare_enabled(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_debug_argmax_compare_enabled(gpu);
}

int bn_transformer_gpu_argmax_debug_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_argmax_debug_enabled(gpu);
}

BnTransformerGPUGenerateArgmaxPolicy
bn_transformer_gpu_generate_argmax_policy(
    const BnGPUBackend *gpu,
    int top_logits,
    float temperature,
    float repeat_penalty) {
    BnTransformerGPUGenerateArgmaxPolicy policy = {0};
    policy.enabled =
        bn_gpu_backend_can_argmax_activation(gpu) &&
        top_logits <= 0 &&
        temperature == 0.0f &&
        repeat_penalty >= 1.0f;
    return policy;
}

int bn_transformer_gpu_argmax_available(
    const BnGPUBackend *gpu,
    int want_argmax) {
    return !want_argmax || bn_gpu_backend_can_argmax_activation(gpu);
}

int bn_transformer_gpu_model_argmax_available(
    const BnModel *model,
    int want_argmax) {
    return model &&
        bn_transformer_gpu_argmax_available(
            bn_model_gpu(model), want_argmax);
}

int bn_transformer_gpu_argmax_backend_run(
    BnGPUBackend *gpu,
    int buf_idx,
    int n,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token) {
    return bn_gpu_backend_argmax_activation(gpu, buf_idx, n, penalty_tokens,
                                            n_penalty_tokens, repeat_penalty,
                                            out_token);
}

int bn_transformer_gpu_matvec_argmax_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPULogitResources *logits,
    int want_argmax,
    int need_logits,
    int gpu_logits_need_cpu) {
    if (!gpu || !c || !logits || !want_argmax || need_logits ||
        !bn_gpu_backend_can_matvec_argmax_activation(gpu) ||
        bn_transformer_gpu_cpu_logits_enabled(gpu, gpu_logits_need_cpu) ||
        bn_gpu_policy_logits_argmax_disabled(gpu) ||
        !bn_backend_quant_supports_kquant_logits_refine(logits->type))
        return 0;

    if (!bn_transformer_gpu_uses_moe(c)) {
        return bn_transformer_gpu_dense_logits_argmax_shape_allowed(
                   c, logits->rows) ||
               bn_gpu_policy_dense_logits_argmax_enabled(gpu);
    }
    if (bn_transformer_gpu_uses_configured_all_active_two_kquant_moe_route(c))
        return 1;
    if (bn_gpu_policy_moe_logits_mmvq_argmax_enabled(gpu))
        return 1;
    return bn_transformer_gpu_moe_logits_mmvq_argmax_shape_allowed(
               c, logits->cols) &&
           !bn_gpu_policy_moe_logits_mmvq_argmax_disabled(gpu);
}

BnTransformerGPULogitsDispatchPolicy
bn_transformer_gpu_logits_dispatch_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPULogitResources *logits,
    int want_argmax,
    int need_logits) {
    BnTransformerGPULogitsDispatchPolicy policy = {0};
    policy.needs_cpu_fallback =
        bn_transformer_gpu_logits_needs_cpu_fallback(gpu, logits);
    policy.cpu_logits_enabled =
        bn_transformer_gpu_cpu_logits_enabled(
            gpu, policy.needs_cpu_fallback);
    policy.use_matvec_argmax =
        bn_transformer_gpu_matvec_argmax_enabled(
            gpu, c, logits, want_argmax, need_logits,
            policy.needs_cpu_fallback);
    return policy;
}

int bn_transformer_gpu_matvec_argmax_backend_run(
    BnGPUBackend *gpu,
    void *W_buf,
    int type,
    int rows,
    int cols,
    int buf_idx,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token) {
    return bn_gpu_backend_matvec_argmax_activation(
        gpu, W_buf, type, rows, cols, buf_idx, penalty_tokens,
        n_penalty_tokens, repeat_penalty, out_token);
}

int bn_transformer_gpu_moe_decode_cacheable(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    const BnBackendModel *backend) {
    if (bn_gpu_policy_moe_decode_cache_disabled(gpu) ||
        !c || !w || !backend || !bn_transformer_gpu_uses_moe(c))
        return 0;
    if (bn_moe_execution_policy(c).uses_scaled_router_input)
        return 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(lw);
        if (!layer_kind.uses_moe)
            continue;
        const BnMoEExpertMap *em = &lw->moe.expert_map;
        int routed_kquant_down = bn_transformer_gpu_moe_routed_kquant_down(em);
        int routed_native_quant = bn_transformer_gpu_moe_routed_native_quant(em);
        int routed_lowbit_block32 =
            bn_transformer_gpu_moe_routed_lowbit_block32(em);
        BnTransformerGPUMoEDecodeResources moe_resources =
            bn_transformer_gpu_resolve_moe_decode_resources(backend, l);
        if (!moe_resources.resident_valid ||
            (!routed_kquant_down && !routed_native_quant &&
             !routed_lowbit_block32) ||
            !bn_transformer_moe_supports_resident_routed_ffn_layout(c, em))
            return 0;
    }
    return 1;
}

int bn_transformer_gpu_decode_cacheable(
    const BnGPUBackend *gpu,
    int emit_logits,
    int want_argmax,
    int gpu_logits_need_cpu,
    int has_moe,
    int cacheable_resident_moe,
    int kquant_logits_refine_captures_xb,
    int native_quant_logits_refine_captures_xb,
    int need_logits,
    int cpu_fallback_layer,
    int cpu_fallback_from_layer,
    int cpu_fallback_attn_layer,
    int cpu_fallback_attn_from_layer,
    int cpu_fallback_ffn_layer,
    int cpu_fallback_ffn_from_layer,
    int cpu_fallback_ffn_down_from_layer,
    int compare_attention_layer,
    int compare_gqa_layer,
    int compare_qkv_layer,
    int compare_ffn_down_layer,
    int compare_ffn_state_layer) {
    if ((!emit_logits || want_argmax ||
         bn_gpu_policy_decode_logits_cache_enabled(
             gpu, gpu_logits_need_cpu)) == 0)
        return 0;
    if (!bn_gpu_policy_backend_decode_graph_cache_supported(gpu))
        return 0;
    if (has_moe && !cacheable_resident_moe &&
        !bn_gpu_policy_moe_decode_cache_enabled(gpu))
        return 0;
    if (bn_gpu_policy_decode_cache_disabled(gpu))
        return 0;
    if (kquant_logits_refine_captures_xb && !(want_argmax && !need_logits))
        return 0;
    if (native_quant_logits_refine_captures_xb &&
        !(want_argmax && !need_logits))
        return 0;
    if (cpu_fallback_layer >= 0 || cpu_fallback_from_layer >= 0 ||
        cpu_fallback_attn_layer >= 0 || cpu_fallback_attn_from_layer >= 0 ||
        cpu_fallback_ffn_layer >= 0 || cpu_fallback_ffn_from_layer >= 0 ||
        cpu_fallback_ffn_down_from_layer >= 0)
        return 0;
    if (compare_attention_layer >= 0 || compare_gqa_layer >= 0 ||
        compare_qkv_layer >= 0 || compare_ffn_down_layer >= 0 ||
        compare_ffn_state_layer >= 0)
        return 0;
    if (bn_gpu_policy_native_quant_decode_cache_disabled(
            gpu ? &gpu->runtime_policy : NULL) ||
        bn_transformer_gpu_cpu_logits_enabled(gpu, gpu_logits_need_cpu) ||
        bn_transformer_gpu_compare_logits_enabled(gpu) ||
        bn_gpu_policy_specialized_native_quant_decode_path_enabled(
            gpu ? &gpu->runtime_policy : NULL))
        return 0;
    return 1;
}

BnTransformerGPUDecodeCacheabilityPolicy
bn_transformer_gpu_decode_cacheability_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    const BnBackendModel *backend,
    int emit_logits,
    int want_argmax,
    int gpu_logits_need_cpu,
    int has_moe,
    const BnTransformerGPULogitsRefinePolicy *logits_refine,
    int need_logits,
    const BnTransformerGPUCPUFallbackPolicy *cpu_fallback,
    const BnTransformerGPUComparePolicy *compare) {
    BnTransformerGPUDecodeCacheabilityPolicy policy = {0};
    policy.resident_moe =
        has_moe &&
        bn_transformer_gpu_moe_decode_cacheable(gpu, c, w, backend);
    policy.graph_cacheable =
        bn_transformer_gpu_decode_cacheable(
            gpu, emit_logits, want_argmax, gpu_logits_need_cpu, has_moe,
            policy.resident_moe,
            logits_refine ? logits_refine->kquant_captures_xb : 0,
            logits_refine ? logits_refine->native_quant_captures_xb : 0,
            need_logits,
            cpu_fallback ? cpu_fallback->layer : -1,
            cpu_fallback ? cpu_fallback->from_layer : -1,
            cpu_fallback ? cpu_fallback->attn_layer : -1,
            cpu_fallback ? cpu_fallback->attn_from_layer : -1,
            cpu_fallback ? cpu_fallback->ffn_layer : -1,
            cpu_fallback ? cpu_fallback->ffn_from_layer : -1,
            cpu_fallback ? cpu_fallback->ffn_down_from_layer : -1,
            compare ? compare->attention_layer : -1,
            compare ? compare->gqa_layer : -1,
            compare ? compare->qkv_layer : -1,
            compare ? compare->ffn_down_layer : -1,
            compare ? compare->ffn_state_layer : -1);
    if (compare && compare->ssm_layer >= 0)
        policy.graph_cacheable = 0;
    return policy;
}

BnTransformerGPUDecodeCacheabilityPolicy
bn_transformer_gpu_model_decode_cacheability_policy(
    const BnModel *model,
    int emit_logits,
    int want_argmax,
    int gpu_logits_need_cpu,
    int has_moe,
    const BnTransformerGPULogitsRefinePolicy *logits_refine,
    int need_logits,
    const BnTransformerGPUCPUFallbackPolicy *cpu_fallback,
    const BnTransformerGPUComparePolicy *compare) {
    if (!model)
        return (BnTransformerGPUDecodeCacheabilityPolicy){0};
    return bn_transformer_gpu_decode_cacheability_policy(
        bn_model_gpu(model), &model->config, &model->weights,
        bn_model_backend(model), emit_logits, want_argmax,
        gpu_logits_need_cpu, has_moe, logits_refine, need_logits,
        cpu_fallback, compare);
}

int bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    const BnBackendRuntimePolicy *runtime =
        gpu ? &gpu->runtime_policy : NULL;
    return bn_transformer_gpu_all_active_two_kquant_moe_model(c, w) &&
           !bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(runtime) &&
           !bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled(
               runtime);
}

int bn_transformer_gpu_moe_reference_attention_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_backend_moe_reference_attention_supported(gpu) &&
           bn_transformer_moe_requires_reference_attention(c) &&
           !bn_gpu_policy_all_active_two_kquant_moe_reference_attention_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_ssm_cpu_fallback_required(
    const BnGPUBackend *gpu) {
    return !bn_gpu_policy_backend_ssm_graph_supported(gpu) ||
           bn_gpu_policy_ssm_graph_disabled(gpu);
}

BnTransformerGPUSSMFallbackPolicy
bn_transformer_gpu_ssm_fallback_policy(
    const BnGPUBackend *gpu) {
    BnTransformerGPUSSMFallbackPolicy policy = {0};
    policy.use_cpu = bn_transformer_gpu_ssm_cpu_fallback_required(gpu);
    return policy;
}

int bn_transformer_gpu_large_hybrid_argmax_blocked(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax) {
    return want_argmax &&
           bn_gpu_policy_backend_large_hybrid_argmax_supported(gpu) &&
           bn_transformer_gpu_large_hybrid_cpu_attn_safe_default(
               gpu, c, w) &&
           !bn_gpu_policy_large_hybrid_argmax_enabled(gpu);
}

BnTransformerGPUDecodeEntryPolicy
bn_transformer_gpu_decode_entry_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax) {
    BnTransformerGPUDecodeEntryPolicy policy = {0};
    policy.block_argmax =
        bn_transformer_gpu_large_hybrid_argmax_blocked(
            gpu, c, w, want_argmax);
    policy.block_forward =
        bn_gpu_policy_backend_reference_attention_token_fallback_supported(
            gpu) &&
        bn_model_transformer_policy_requires_reference_attention(c) &&
        !bn_transformer_gpu_reference_attention_exact_enabled(gpu, c) &&
        !bn_gpu_policy_backend_reference_attention_native_graph_supported(gpu);
    return policy;
}

BnTransformerGPUCPUFallbackPolicy
bn_transformer_gpu_cpu_fallback_policy(const BnGPUBackend *gpu) {
    BnTransformerGPUCPUFallbackPolicy policy = {
        .layer = bn_gpu_policy_cpu_fallback_layer_or_default(gpu, -1),
        .from_layer =
            bn_gpu_policy_cpu_fallback_from_layer_or_default(gpu, -1),
        .attn_layer = bn_gpu_policy_cpu_attention_layer_or_default(gpu, -1),
        .attn_from_layer =
            bn_gpu_policy_cpu_attention_from_layer_or_default(gpu, -1),
        .ffn_layer = bn_gpu_policy_cpu_ffn_layer_or_default(gpu, -1),
        .ffn_from_layer =
            bn_gpu_policy_cpu_ffn_from_layer_or_default(gpu, -1),
        .ffn_down_from_layer =
            bn_gpu_policy_cpu_ffn_down_from_layer_or_default(gpu, -1),
    };
    return policy;
}

static int gpu_cpu_attention_fallback_unset(
    const BnTransformerGPUCPUFallbackPolicy *policy) {
    return policy &&
           policy->layer < 0 &&
           policy->from_layer < 0 &&
           policy->attn_layer < 0 &&
           policy->attn_from_layer < 0;
}

BnTransformerGPUCPUFallbackPolicy
bn_transformer_gpu_decode_cpu_attention_fallback_policy(
    BnTransformerGPUCPUFallbackPolicy policy,
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    if (!gpu_cpu_attention_fallback_unset(&policy))
        return policy;
    int default_cpu_attention =
        bn_transformer_gpu_reference_attention_cpu_fallback_enabled(gpu, c) ||
        bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
            gpu, c, w) ||
        bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
            gpu, c, w) ||
        bn_transformer_gpu_large_hybrid_cpu_attn_safe_fallback_enabled(
            gpu, c, w);
    if (default_cpu_attention)
        policy.attn_from_layer = 0;
    return policy;
}

int bn_transformer_gpu_cpu_fallback_layer_selected(
    int layer,
    int selected_layer,
    int from_layer) {
    return (selected_layer >= 0 && layer == selected_layer) ||
           (from_layer >= 0 && layer >= from_layer);
}

BnTransformerGPUSmallDenseNativeQuantLayerPolicy
bn_transformer_gpu_small_dense_native_quant_layer_policy(const BnConfig *c) {
    return bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
        NULL, c);
}

BnTransformerGPUSmallDenseNativeQuantLayerPolicy
bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
    const BnGPUBackend *gpu, const BnConfig *c) {
    const BnBackendRuntimePolicy *runtime =
        gpu ? &gpu->runtime_policy : NULL;
    int n_layers = c ? c->n_layers : 0;
    BnTransformerGPUSmallDenseNativeQuantLayerPolicy policy = {
        .from_layer =
            bn_gpu_policy_small_dense_native_quant_from_layer_or_default(
                runtime, n_layers),
        .to_layer = bn_gpu_policy_small_dense_native_quant_to_layer_or_default(
            runtime, n_layers,
            bn_gpu_policy_small_dense_native_quant_prepared_layer_default_enabled(
                runtime) ||
            (bn_model_transformer_policy_has_auxiliary_prediction_blocks(c) &&
             bn_gpu_backend_has_cap(gpu,
                                    BN_GPU_CAP_PREPARED_NATIVE_QUANT))),
        .attn_only =
            bn_gpu_policy_small_dense_native_quant_attn_only_enabled(runtime),
        .ffn_only =
            bn_gpu_policy_small_dense_native_quant_ffn_only_enabled(runtime),
    };
    return policy;
}

BnTransformerGPUSmallDenseNativeQuantDecodePolicy
bn_transformer_gpu_small_dense_native_quant_decode_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPUSmallDenseNativeQuantLayerPolicy *layer_policy) {
    BnTransformerGPUSmallDenseNativeQuantDecodePolicy policy = {0};
    int from_layer = layer_policy ? layer_policy->from_layer : -1;
    int to_layer = layer_policy ? layer_policy->to_layer : -1;
    policy.small_dense_native_quant_default =
        bn_transformer_gpu_small_dense_native_quant_default(
            gpu, c, from_layer);
    policy.small_dense_native_quant_to_layer =
        bn_transformer_gpu_small_dense_native_quant_to_layer(
            c, policy.small_dense_native_quant_default, to_layer);
    return policy;
}

BnTransformerGPUSmallDenseNativeQuantLayerUsePolicy
bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPUSmallDenseNativeQuantLayerPolicy *policy,
    int layer,
    int small_dense_native_quant_default,
    int small_dense_native_quant_to_layer) {
    BnTransformerGPUSmallDenseNativeQuantLayerUsePolicy use = {0};
    if (!policy)
        return use;

    int use_prepared_reference_attention =
        bn_model_transformer_policy_requires_reference_attention(c) &&
        bn_gpu_policy_backend_reference_attention_native_graph_supported(gpu) &&
        bn_gpu_backend_has_cap(
            gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION);

    use.use_layer = policy->from_layer >= 0 &&
                    layer >= policy->from_layer &&
                    (policy->to_layer < 0 || layer <= policy->to_layer);
    if (bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT) &&
        bn_gpu_backend_has_cap(
            gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN) &&
        bn_model_transformer_policy_uses_per_layer_embedding(c))
        use.use_layer = 1;
    if (use_prepared_reference_attention)
        use.use_layer = 1;
    use.small_dense_native_quant_path =
        small_dense_native_quant_default &&
        (small_dense_native_quant_to_layer < 0 ||
         layer <= small_dense_native_quant_to_layer);
    if (use.small_dense_native_quant_path)
        use.use_layer = 1;

    use.use_attention = use.use_layer && !policy->ffn_only;
    use.use_ffn = use.use_layer && !policy->attn_only;
    if (bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT) &&
        bn_gpu_backend_has_cap(
            gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN) &&
        bn_model_transformer_policy_uses_per_layer_embedding(c)) {
        use.use_attention = bn_gpu_backend_has_cap(
            gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION);
        use.use_ffn = 1;
    }
    if (use_prepared_reference_attention)
        use.use_attention = 1;
    use.use_ffn_down = use.use_ffn;
    if (bn_gpu_backend_has_cap(
            gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN) &&
        bn_model_transformer_policy_uses_per_layer_embedding(c))
        use.use_ffn_down = bn_gpu_backend_has_cap(
            gpu, BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN);
    if (use.small_dense_native_quant_path &&
        !bn_transformer_gpu_small_dense_native_quant_ffn_down_enabled(
            gpu, c))
        use.use_ffn_down = 0;
    return use;
}

BnTransformerGPUCachedDecodePolicy
bn_transformer_gpu_cached_decode_policy(
    int cached_op_count,
    int argmax_requested,
    int cached_has_logits,
    int matvec_argmax_available) {
    BnTransformerGPUCachedDecodePolicy policy = {0};
    policy.use_cache = cached_op_count > 0;
    policy.clear_cache =
        policy.use_cache &&
        argmax_requested &&
        !cached_has_logits &&
        !matvec_argmax_available;
    if (policy.clear_cache)
        policy.use_cache = 0;
    return policy;
}

BnTransformerGPUMoERouteLayerPolicy
bn_transformer_gpu_moe_route_layer_policy(const BnGPUBackend *gpu) {
    BnTransformerGPUMoERouteLayerPolicy policy = {-1, -1};
    bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
        gpu, &policy.from_layer, &policy.to_layer);
    return policy;
}

BnTransformerGPUComparePolicy
bn_transformer_gpu_compare_policy(const BnGPUBackend *gpu) {
    const BnBackendRuntimePolicy *runtime =
        gpu ? &gpu->runtime_policy : NULL;
    BnTransformerGPUComparePolicy policy = {
        .attention_layer =
            bn_gpu_policy_compare_attention_layer_or_default(runtime, -1),
        .attention_pos =
            bn_gpu_policy_compare_attention_pos_or_default(runtime, -1),
        .gqa_layer = bn_gpu_policy_compare_gqa_layer_or_default(runtime, -1),
        .gqa_pos = bn_gpu_policy_compare_gqa_pos_or_default(runtime, -1),
        .qkv_layer = bn_gpu_policy_compare_qkv_layer_or_default(runtime, -1),
        .qkv_pos = bn_gpu_policy_compare_qkv_pos_or_default(runtime, -1),
        .ffn_down_layer =
            bn_gpu_policy_compare_ffn_down_layer_or_default(runtime, -1),
        .ffn_down_pos =
            bn_gpu_policy_compare_ffn_down_pos_or_default(runtime, -1),
        .ffn_state_layer =
            bn_gpu_policy_compare_ffn_state_layer_or_default(runtime, -1),
        .ffn_state_pos =
            bn_gpu_policy_compare_ffn_state_pos_or_default(runtime, -1),
        .ssm_layer = bn_gpu_policy_compare_ssm_layer_or_default(runtime, -1),
        .ssm_pos = bn_gpu_policy_compare_ssm_pos_or_default(runtime, -1),
    };
    return policy;
}

int bn_transformer_gpu_flash_attention_enabled(
    const BnGPUBackend *gpu,
    int flash_requested,
    int has_moe,
    int n_kv) {
    int flash_default = bn_gpu_policy_backend_flash_default_enabled(gpu);
    int flash_min_kv = bn_gpu_policy_flash_min_kv_or_default(gpu, 0);
    int flash_max_kv =
        bn_gpu_policy_backend_flash_max_kv_or_default(gpu, 0);

    return bn_transformer_gpu_can_flash_attn(gpu) &&
           (has_moe || flash_requested || flash_default) &&
           n_kv >= flash_min_kv &&
           (flash_max_kv <= 0 || n_kv <= flash_max_kv);
}

int bn_transformer_gpu_moe_routed_kquant_down(const BnMoEExpertMap *map) {
    return bn_transformer_gpu_moe_routed_kquant_down_allowed(map, 1);
}

int bn_transformer_gpu_moe_routed_kquant_down_allowed(
    const BnMoEExpertMap *map,
    int allow_kquant_down) {
    return map &&
           bn_backend_quant_moe_route_asymmetric_kquant_down(
               map->gate_type,
               map->up_type,
               map->down_type,
               allow_kquant_down);
}

int bn_transformer_gpu_moe_routed_native_quant(const BnMoEExpertMap *map) {
    return map &&
           bn_backend_quant_moe_route_native_quant(map->gate_type,
                                                   map->up_type,
                                                   map->down_type);
}

int bn_transformer_gpu_moe_routed_lowbit_block32(
    const BnMoEExpertMap *map) {
    return map && bn_backend_quant_moe_routed_lowbit_block32(
                      map->gate_type, map->up_type, map->down_type);
}

int bn_transformer_gpu_dense_residual_moe_requires_cpu_ffn(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map) {
    if (!gpu || !c || !map ||
        !bn_moe_execution_policy(c).uses_dense_residual_branch ||
        !bn_transformer_gpu_moe_routed_lowbit_block32(map))
        return 0;
    return !bn_gpu_backend_has_cap(
        gpu, BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32);
}

int bn_transformer_gpu_moe_route_topk_enabled(
    const BnGPUBackend *gpu,
    void *moe_router,
    int all_active_two_kquant_moe,
    int all_active_two_kquant_moe_gpu_route_layer_selected) {
    int eligible = moe_router &&
                   (!all_active_two_kquant_moe ||
                    all_active_two_kquant_moe_gpu_route_layer_selected);
    return bn_gpu_policy_moe_router_topk_enabled(gpu, eligible);
}

int bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int all_active_two_kquant_moe,
    int gpu_route_topk,
    int moe_routed_native_quant,
    int moe_routed_lowbit_block32) {
    if (all_active_two_kquant_moe && !gpu_route_topk &&
        !bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled(
            gpu ? &gpu->runtime_policy : NULL))
        return 1;
    int grouped_cpu_route = !gpu_route_topk &&
        bn_transformer_moe_uses_grouped_route(c);
    return bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(
               gpu, grouped_cpu_route && moe_routed_native_quant) ||
           bn_gpu_policy_lowbit_block32_moe_cpu_route_resident_enabled(
               gpu, grouped_cpu_route && moe_routed_lowbit_block32);
}

int bn_transformer_gpu_moe_routed_ffn_enabled(
    const BnGPUBackend *gpu,
    int gpu_route_topk,
    int cpu_route_resident_ffn,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all,
    const BnMoEExpertMap *map,
    const BnConfig *c,
    int dim) {
    if (!bn_gpu_backend_moe_route_shape_supported(
            gpu, bn_moe_route_policy(c).total_experts))
        return 0;
    int routed_kquant_down_supported =
        bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_ROUTED_KQUANT_DOWN_CACHE) &&
        bn_transformer_gpu_moe_routed_kquant_down(map);
    int routed_native_quant_supported =
        bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_ROUTED_NATIVE_QUANT) &&
        bn_transformer_gpu_moe_routed_native_quant(map);
    int routed_lowbit_block32_supported =
        bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_ROUTED_LOWBIT_BLOCK32) &&
        bn_transformer_gpu_moe_routed_lowbit_block32(map);
    if ((!gpu_route_topk && !cpu_route_resident_ffn) ||
        !moe_gate_all || !moe_up_all || !moe_down_all ||
        (!routed_kquant_down_supported &&
         !routed_native_quant_supported &&
         !routed_lowbit_block32_supported &&
         !(bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_ROUTED_FFN) && map &&
           bn_backend_quant_moe_routed_kquant_gateup(
               map->gate_type, map->up_type) &&
           bn_backend_quant_moe_direct_routed_down(map->down_type))) ||
        !bn_gpu_policy_moe_resident_routed_ffn_enabled(gpu, 1))
        return 0;
    return bn_transformer_moe_supports_resident_routed_ffn_shape(c, map, dim);
}

uint32_t bn_transformer_gpu_moe_route_normalization_flags(const BnConfig *c) {
    return bn_transformer_moe_normalizes_topk_route_weights(c)
        ? 0u
        : BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM;
}

BnTransformerGPUMoEDecodeRoutePolicy
bn_transformer_gpu_moe_decode_route_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUMoERouteLayerPolicy *layer_policy,
    int layer,
    int dim,
    void *moe_router,
    void *router_diff,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all) {
    BnTransformerGPUMoEDecodeRoutePolicy policy = {0};
    int from_layer = layer_policy ? layer_policy->from_layer : -1;
    int to_layer = layer_policy ? layer_policy->to_layer : -1;

    policy.all_active_two_kquant_moe =
        bn_transformer_gpu_all_active_two_kquant_moe_layer_enabled(
            gpu, c, lw, dim);
    policy.route_layer_selected =
        bn_transformer_gpu_all_active_two_kquant_moe_route_layer_selected(
            gpu, layer, from_layer, to_layer);
    policy.reference_gpu_route =
        bn_transformer_gpu_all_active_two_kquant_moe_reference_gpu_route_enabled(
            gpu, policy.all_active_two_kquant_moe,
            policy.route_layer_selected);
    policy.router = bn_transformer_gpu_all_active_two_kquant_moe_router(
        gpu, c, moe_router, router_diff, policy.route_layer_selected,
        policy.reference_gpu_route);
    policy.route_flags |= bn_transformer_gpu_moe_route_normalization_flags(c);

    int routed_native_quant = lw &&
        bn_transformer_gpu_moe_routed_native_quant(&lw->moe.expert_map);
    int routed_lowbit_block32 = lw &&
        bn_transformer_gpu_moe_routed_lowbit_block32(&lw->moe.expert_map);
    BnMoEExecutionPolicy execution = bn_moe_execution_policy(c);
    policy.uses_scaled_router_input = execution.uses_scaled_router_input;
    policy.gpu_route_topk =
        (!execution.uses_scaled_router_input ||
         (lw && lw->moe.router_scale)) &&
        bn_gpu_policy_backend_moe_route_topk_supported(gpu) &&
        bn_gpu_backend_moe_route_shape_supported(
            gpu, bn_moe_route_policy(c).total_experts) &&
        bn_transformer_gpu_moe_route_topk_enabled(
            gpu, policy.router, policy.all_active_two_kquant_moe,
            policy.route_layer_selected);
    policy.cpu_route_resident_ffn =
        bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
            gpu, c, policy.all_active_two_kquant_moe, policy.gpu_route_topk,
            routed_native_quant, routed_lowbit_block32);
    policy.gpu_routed_ffn =
        lw &&
        (!policy.all_active_two_kquant_moe || policy.route_layer_selected) &&
        bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_ROUTED_FFN) &&
        bn_transformer_gpu_moe_routed_ffn_enabled(
            gpu,
            policy.gpu_route_topk, policy.cpu_route_resident_ffn,
            moe_gate_all, moe_up_all, moe_down_all, &lw->moe.expert_map,
            c, dim);
    return policy;
}

BnTransformerGPUMoEDirectRoutePolicy
bn_transformer_gpu_moe_direct_route_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all) {
    BnTransformerGPUMoEDirectRoutePolicy policy = {0};
    policy.router_diff = router_diff;
    policy.enabled =
        bn_gpu_policy_backend_all_active_two_moe_direct_route_supported(gpu) &&
        bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
            gpu, c, router_diff, moe_gate_all);
    return policy;
}

BnTransformerGPUMoEDecodeDispatchPolicy
bn_transformer_gpu_moe_decode_dispatch_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUMoERouteLayerPolicy *layer_policy,
    int layer,
    int dim,
    void *moe_router,
    void *router_diff,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all) {
    BnTransformerGPUMoEDecodeDispatchPolicy policy = {0};
    policy.direct_route = bn_transformer_gpu_moe_direct_route_policy(
        gpu, c, router_diff, moe_gate_all);
    policy.requires_session_state = !policy.direct_route.enabled;
    policy.route_profile_enabled =
        policy.requires_session_state &&
        bn_transformer_gpu_moe_route_profile_enabled(gpu);
    if (policy.requires_session_state) {
        policy.decode_route = bn_transformer_gpu_moe_decode_route_policy(
            gpu, c, lw, layer_policy, layer, dim, moe_router, router_diff,
            moe_gate_all, moe_up_all, moe_down_all);
    }
    return policy;
}

int bn_transformer_gpu_uses_configured_all_active_two_kquant_moe_route(
    const BnConfig *c) {
    return bn_transformer_moe_uses_configured_all_active_two_route(c);
}

BnTransformerGPUMoEAllActiveTwoResourcePolicy
bn_transformer_gpu_moe_all_active_two_resource_policy(const BnConfig *c) {
    BnTransformerGPUMoEAllActiveTwoResourcePolicy policy = {0};
    BnMoEAllActiveTwoRouteResourcePolicy moe_policy =
        bn_moe_all_active_two_route_resource_policy(c);
    policy.enabled = moe_policy.enabled;
    policy.total_experts = moe_policy.total_experts;
    policy.expert_hidden_dim = moe_policy.expert_hidden_dim;
    policy.complement_route_from_expert =
        moe_policy.complement_route_from_expert;
    return policy;
}

BnTransformerGPUMoEExecutionPolicy
bn_transformer_gpu_moe_execution_policy(const BnConfig *c) {
    BnTransformerGPUMoEExecutionPolicy policy = {0};
    BnMoERoutePolicy route = bn_moe_route_policy(c);
    policy.total_experts = route.total_experts;
    policy.active_experts = route.active_experts;
    policy.expert_hidden_dim = route.expert_hidden_dim;
    policy.normalize_topk = route.norm_topk_prob;
    policy.expert_weights_scale = route.expert_weights_scale;
    return policy;
}

BnTransformerGPUMoEProjectionPolicy
bn_transformer_gpu_moe_projection_policy(const BnMoEExpertMap *map) {
    BnTransformerGPUMoEProjectionPolicy policy = {0};
    BnMoERoutedExpertProjectionTypes types;
    if (!bn_moe_routed_expert_projection_types(&types, map))
        return policy;
    policy.valid = 1;
    policy.gate_type = types.gate_type;
    policy.up_type = types.up_type;
    policy.down_type = types.down_type;
    return policy;
}

int bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all) {
    return router_diff &&
           bn_transformer_gpu_uses_configured_all_active_two_kquant_moe_route(
               c) &&
           bn_transformer_moe_normalizes_topk_route_weights(c) &&
           !moe_gate_all &&
           bn_gpu_policy_moe_router_gpu_enabled(
               gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_all_active_two_kquant_moe_route_layer_selected(
    const BnGPUBackend *gpu,
    int layer,
    int route_from_layer,
    int route_to_layer) {
    if (!bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled(
            gpu ? &gpu->runtime_policy : NULL))
        return 0;
    return route_from_layer < 0 ||
           (layer >= route_from_layer &&
            (route_to_layer < 0 || layer <= route_to_layer));
}

void bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
    const BnGPUBackend *gpu,
    int *route_from_layer,
    int *route_to_layer) {
    bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(
        gpu ? &gpu->runtime_policy : NULL, route_from_layer, route_to_layer);
}

int bn_transformer_gpu_all_active_two_kquant_moe_reference_gpu_route_enabled(
    const BnGPUBackend *gpu,
    int all_active_two_kquant_moe,
    int route_layer_selected) {
    return all_active_two_kquant_moe &&
           route_layer_selected &&
           bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(
               gpu ? &gpu->runtime_policy : NULL) &&
           !bn_gpu_policy_all_active_two_kquant_moe_reference_gpu_route_disabled(
               gpu ? &gpu->runtime_policy : NULL);
}

void *bn_transformer_gpu_all_active_two_kquant_moe_router(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    void *moe_router,
    void *router_diff,
    int route_layer_selected,
    int reference_gpu_route) {
    if (router_diff &&
        bn_transformer_gpu_uses_configured_all_active_two_kquant_moe_route(c) &&
        route_layer_selected &&
        bn_gpu_policy_moe_router_diff2_enabled(gpu) &&
        !reference_gpu_route)
        return router_diff;
    return moe_router;
}

int bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down) {
    if (!c || !map ||
        !bn_transformer_moe_uses_all_active_two_route(c, dim) ||
        !bn_transformer_gpu_moe_routed_kquant_down_allowed(
            map, allow_kquant_down) ||
        bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(
            gpu ? &gpu->runtime_policy : NULL))
        return 0;
    return 1;
}

int bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int layer,
    int cpu_fallback_ffn_layer,
    int cpu_fallback_ffn_from_layer) {
    int selected_expert_graph =
        bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_EXPERT_GRAPH);
    int resident_route =
        bn_gpu_policy_backend_resident_moe_ffn_supported(gpu) &&
        bn_gpu_backend_moe_route_shape_supported(
            gpu, bn_moe_route_policy(c).total_experts);
    if (!selected_expert_graph && !resident_route)
        return 1;
    if (bn_transformer_gpu_moe_ffn_disabled(gpu))
        return 1;
    if (bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in(
            gpu, c, map, dim, allow_kquant_down))
        return 1;
    return bn_transformer_gpu_cpu_fallback_layer_selected(
        layer, cpu_fallback_ffn_layer, cpu_fallback_ffn_from_layer);
}

BnTransformerGPUMoEFFNFallbackPolicy
bn_transformer_gpu_moe_ffn_fallback_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int layer,
    const BnTransformerGPUCPUFallbackPolicy *cpu_fallback) {
    BnTransformerGPUMoEFFNFallbackPolicy policy = {0};
    int fallback_layer = cpu_fallback ? cpu_fallback->ffn_layer : -1;
    int fallback_from_layer = cpu_fallback ? cpu_fallback->ffn_from_layer : -1;
    policy.use_cpu = bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        gpu, c, map, dim, allow_kquant_down, layer, fallback_layer,
        fallback_from_layer);
    return policy;
}

int bn_transformer_gpu_moe_routed_ffn_batch_allowed(
    const BnGPUBackend *gpu, const BnConfig *c) {
    return bn_gpu_policy_moe_routed_ffn_batch_allowed(
        gpu ? &gpu->runtime_policy : NULL,
        bn_transformer_moe_uses_grouped_route(c));
}

int bn_transformer_gpu_moe_ffn_disabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_ffn_disabled(gpu);
}

int bn_transformer_gpu_moe_cpu_actual_override_enabled(
    const BnGPUBackend *gpu, int safe_default) {
    return safe_default ||
           bn_gpu_policy_moe_cpu_actual_override_enabled(gpu);
}

BnTransformerGPUMoEDebugPolicy bn_transformer_gpu_moe_debug_policy(
    const BnGPUBackend *gpu,
    int cpu_actual_safe_default,
    int compare_layer_selected) {
    BnTransformerGPUMoEDebugPolicy policy = {0};
    policy.override_cpu_actual =
        bn_transformer_gpu_moe_cpu_actual_override_enabled(
            gpu, cpu_actual_safe_default);
    policy.compare_layer = compare_layer_selected;
    policy.compare_route =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_route_enabled(&gpu->runtime_policy);
    policy.compare_input_norm =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_input_norm_enabled(&gpu->runtime_policy);
    policy.compare_actual =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_actual_enabled(&gpu->runtime_policy);
    policy.compare_raw =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_raw_enabled(&gpu->runtime_policy);
    policy.compare_mid =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_mid_enabled(&gpu->runtime_policy);
    policy.compare_parts =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_parts_enabled(&gpu->runtime_policy);
    policy.compare_shared_mid =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_shared_mid_enabled(&gpu->runtime_policy);
    policy.compare_shared_down =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_shared_down_enabled(&gpu->runtime_policy);
    policy.compare_norm =
        compare_layer_selected &&
        bn_gpu_policy_moe_compare_norm_enabled(&gpu->runtime_policy);
    return policy;
}

BnTransformerGPUMoEDebugPolicy bn_transformer_gpu_moe_decode_debug_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int layer,
    int pos) {
    return bn_transformer_gpu_moe_debug_policy(
        gpu,
        bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
            gpu, c, w),
        bn_gpu_policy_moe_compare_layer_selected(&gpu->runtime_policy,
                                                 layer, pos));
}

int bn_transformer_gpu_moe_compare_layer_selected(
    const BnGPUBackend *gpu, int layer, int pos) {
    return gpu && bn_gpu_policy_moe_compare_layer_selected(
                      &gpu->runtime_policy, layer, pos);
}

int bn_transformer_gpu_moe_compare_actual_enabled(
    const BnGPUBackend *gpu) {
    return gpu && bn_gpu_policy_moe_compare_actual_enabled(
                      &gpu->runtime_policy);
}

int bn_transformer_gpu_moe_shared_cpu_fallback_enabled(
    const BnGPUBackend *gpu, int eligible) {
    return bn_gpu_policy_moe_shared_cpu_fallback_enabled(gpu, eligible);
}

int bn_transformer_gpu_weighted_add_sigmoid_supported(
    const BnGPUBackend *gpu) {
    return bn_gpu_policy_backend_weighted_add_sigmoid_supported(gpu);
}

int bn_transformer_gpu_moe_has_loaded_shared_expert(
    const BnConfig *c,
    const BnLayerWeights *lw) {
    return bn_transformer_moe_has_loaded_shared_expert_path(c, lw);
}

BnTransformerGPUMoESharedCPUFallbackPolicy
bn_transformer_gpu_moe_shared_cpu_fallback_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw) {
    BnTransformerGPUMoESharedCPUFallbackPolicy policy = {0};
    policy.enabled =
        bn_transformer_gpu_moe_shared_cpu_fallback_enabled(
            gpu,
            bn_transformer_gpu_moe_has_loaded_shared_expert(c, lw));
    return policy;
}

int bn_transformer_gpu_moe_gateup_split_enabled(
    const BnGPUBackend *gpu,
    int can_split) {
    return bn_gpu_policy_backend_moe_gateup_split_supported(gpu) &&
           can_split &&
           bn_gpu_policy_moe_gateup_split_enabled(gpu, can_split);
}

int bn_transformer_gpu_moe_route_profile_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_route_profile_enabled(gpu);
}

int bn_transformer_gpu_moe_route_profile_every(const BnGPUBackend *gpu) {
    return bn_gpu_policy_moe_route_profile_every_or_default(gpu, 28);
}

int bn_transformer_gpu_profile_level(const BnGPUBackend *gpu) {
    return bn_gpu_policy_profile_level(gpu ? &gpu->runtime_policy : NULL);
}

int bn_transformer_gpu_debug_fallback_enabled(const BnGPUBackend *gpu) {
    return bn_gpu_policy_debug_fallback_enabled(gpu);
}

void bn_transformer_gpu_report_fallback(const BnGPUBackend *gpu,
                                        const char *reason) {
    if (!bn_transformer_gpu_debug_fallback_enabled(gpu))
        return;
    fprintf(stderr, "[gpu:fallback] %s\n", reason ? reason : "unknown");
}

float *bn_transformer_gpu_reject_forward(
    BnTransformerGPUEmitContext *emit,
    const char *reason) {
    bn_transformer_gpu_report_fallback(emit ? emit->gpu : NULL, reason);
    bn_transformer_gpu_emit_context_free(emit);
    return NULL;
}

int bn_transformer_gpu_validate_forward(
    BnTransformerGPUForwardPolicy *out,
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnWeights *w,
    int token,
    int pos,
    const char **reject_reason) {
    *out = (BnTransformerGPUForwardPolicy){0};
    if (reject_reason)
        *reject_reason = NULL;
#define GPU_POLICY_REJECT(msg) do { \
        if (reject_reason) *reject_reason = (msg); \
        return -1; \
    } while (0)

    if (!gpu)
        GPU_POLICY_REJECT("backend missing");
    if (!bn_gpu_backend_can_execute(gpu))
        GPU_POLICY_REJECT("backend missing execute");
    if (!bn_gpu_backend_can_write_activation(gpu))
        GPU_POLICY_REJECT("backend missing write_activation");

    if (token < 0 || token >= c->vocab_size)
        GPU_POLICY_REJECT("token out of bounds");
    if (pos < 0)
        GPU_POLICY_REJECT("negative position");
    if (!bn_gpu_policy_force_graph_enabled(gpu) &&
        bn_transformer_gpu_uses_hybrid_ssm(c) &&
        !bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_SSM_GRAPH))
        GPU_POLICY_REJECT("hybrid ssm graph unsupported by gpu backend");
    if (!bn_gpu_policy_force_graph_enabled(gpu) &&
        bn_transformer_gpu_uses_hybrid_ssm(c) &&
        bn_transformer_uses_hybrid_moe(c) &&
        !bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_HYBRID_SSM_MOE_GRAPH))
        GPU_POLICY_REJECT("combined hybrid ssm/moe graph unsupported by gpu backend");

    static const BnGPUBackend *cached_gpu = NULL;
    static const BnBackendModel *cached_backend = NULL;
    static const BnConfig *cached_config = NULL;
    static const BnWeights *cached_weights = NULL;
    static BnTransformerGPUForwardPolicy cached_policy;
    static int cached_valid = 0;
    if (cached_valid && cached_gpu == gpu && cached_backend == backend &&
        cached_config == c && cached_weights == w) {
        *out = cached_policy;
        return 0;
    }

    int backend_large_native =
        bn_gpu_policy_backend_large_graph_native_enabled(gpu);
    if (!bn_gpu_policy_force_graph_enabled(gpu) && !backend_large_native &&
        bn_transformer_gpu_uses_large_dense_shape(c))
        GPU_POLICY_REJECT("large dense gpu graph disabled");
    if (bn_transformer_gpu_requires_layerwise_rope(c, w) &&
        !bn_transformer_gpu_can_layerwise_rope(gpu))
        GPU_POLICY_REJECT("layerwise rope unsupported by gpu backend");
    if (bn_transformer_gpu_uses_per_layer_embedding(c) &&
        !bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_PER_LAYER_INPUT_GRAPH))
        GPU_POLICY_REJECT("per-layer input graph unsupported by gpu backend");

    if (bn_gpu_policy_backend_small_dense_native_enabled(gpu) &&
        bn_transformer_gpu_uses_small_dense_shape(c)) {
        if (bn_gpu_policy_small_state_native_quant_disabled(
                gpu ? &gpu->runtime_policy : NULL)) {
            if (!small_dense_backend_native_quant_by_default(c, w))
                GPU_POLICY_REJECT("small dense gpu graph disabled");
        } else if (!small_dense_backend_native_by_default(c, w)) {
            GPU_POLICY_REJECT("small dense gpu graph unsupported");
        }
    }

    if (c->dim > BN_TRANSFORMER_GPU_MAX_VLA_ELEMS)
        GPU_POLICY_REJECT("dim exceeds VLA limit");

    out->gpu = (BnGPUBackend *)gpu;
    out->initial_norm = bn_transformer_gpu_resolve_initial_norm(backend);
    out->output_norm = bn_transformer_gpu_resolve_output_norm(backend);
    if (!out->output_norm)
        GPU_POLICY_REJECT("output norm not uploaded");

    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        BnTransformerGPULayerValidationResources layer_res =
            bn_transformer_gpu_resolve_layer_validation_resources(backend, l);
        int is_attn = bn_transformer_is_attn_layer(c, l);
        if (!is_attn) {
            out->has_ssm = 1;
            continue;
        }
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(lw);
        if (layer_kind.uses_moe)
            out->has_moe = 1;
        if (!lw->attn.wq.data && !lw->ssm.wqkv.data)
            GPU_POLICY_REJECT("attention layer has no wq/wqkv data");
        if (lw->attn.q_norm && !layer_res.q_norm)
            GPU_POLICY_REJECT("q norm not uploaded");
        if (lw->attn.k_norm && !layer_res.k_norm)
            GPU_POLICY_REJECT("k norm not uploaded");
        if (lw->norm.attn_sub_norm && !layer_res.attn_sub_norm)
            GPU_POLICY_REJECT("attention sub norm not uploaded");
        if (lw->norm.ffn_sub_norm && !layer_res.ffn_sub_norm)
            GPU_POLICY_REJECT("ffn sub norm not uploaded");
        if (!layer_res.attn_norm || !layer_res.ffn_norm)
            GPU_POLICY_REJECT("layer norm not uploaded");
    }

    if (out->has_moe) {
        int selected_expert_graph =
            bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_EXPERT_GRAPH);
        int resident_route =
            bn_gpu_policy_backend_resident_moe_ffn_supported(gpu) &&
            bn_gpu_backend_moe_route_shape_supported(
                gpu, bn_moe_route_policy(c).total_experts);
        if ((!selected_expert_graph && !resident_route) ||
            bn_transformer_gpu_moe_ffn_disabled(gpu))
            GPU_POLICY_REJECT("moe gpu forward unsupported");
    }
    if (out->has_moe &&
        bn_gpu_policy_backend_all_active_two_kquant_moe_supported(gpu) &&
        all_active_two_kquant_moe_requires_opt_in(gpu, c, w))
        GPU_POLICY_REJECT("all-active-two K-quant MoE gpu-resident forward requires opt-in");
    if (out->has_ssm &&
        (!bn_gpu_backend_can_read_activation(gpu) ||
         !bn_gpu_backend_can_write_activation(gpu)))
        GPU_POLICY_REJECT("ssm needs read/write activation");

    bn_transformer_gpu_resolve_logit_resources(&out->logits, backend, c, w);
    if (!out->logits.gpu_buf)
        GPU_POLICY_REJECT("logit weight not uploaded");

    cached_gpu = gpu;
    cached_backend = backend;
    cached_config = c;
    cached_weights = w;
    cached_policy = *out;
    cached_valid = 1;
    return 0;
#undef GPU_POLICY_REJECT
}

int bn_transformer_gpu_validate_model_forward(
    BnTransformerGPUForwardPolicy *out,
    const BnModel *model,
    int token,
    int pos,
    const char **reject_reason) {
    if (!model) {
        if (out)
            *out = (BnTransformerGPUForwardPolicy){0};
        if (reject_reason)
            *reject_reason = "model missing";
        return -1;
    }
    int rc = bn_transformer_gpu_validate_forward(
        out, bn_model_gpu(model), bn_model_backend(model),
        &model->config, &model->weights, token, pos, reject_reason);
    if (rc == 0)
        out->has_tq = bn_model_has_tq(model);
    return rc;
}
