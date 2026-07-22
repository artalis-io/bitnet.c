#include "moe_internal.h"
#include "backend_quant.h"
#include "model_internal.h"

uint32_t bn_moe_float_kquant_gateup_fallback_task_flags(const BnConfig *c) {
    return bn_model_config_moe_requires_float_kquant_gateup_fallback(c)
        ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT
        : 0u;
}

BnMoEExecutionPolicy bn_moe_execution_policy(const BnConfig *c) {
    BnMoEExecutionPolicy policy = {0};
    policy.uses_scaled_router_input =
        bn_model_config_moe_uses_scaled_router_input(c);
    policy.uses_dense_residual_branch =
        bn_model_config_moe_uses_dense_residual_branch(c);
    policy.exact_silu =
        policy.uses_dense_residual_branch || !c ? -1 : c->moe_exact_silu;
    return policy;
}

int bn_moe_policy_exact_silu(const BnConfig *c) {
    return bn_moe_execution_policy(c).exact_silu;
}

BnMoEPrefillPolicy bn_moe_prefill_policy(const BnConfig *c) {
    BnMoEPrefillPolicy policy = {0};
    policy.requires_matvec_prefill =
        bn_model_config_moe_prefill_requires_matvec(c);
    policy.uses_grouped_expert_route =
        bn_model_config_uses_more_than_two_expert_moe(c);
    return policy;
}

BnMoERoutePolicy bn_moe_route_policy(const BnConfig *c) {
    BnMoERoutePolicy policy = {0};
    if (!c)
        return policy;
    policy.total_experts = c->n_experts;
    policy.active_experts = c->n_experts_active;
    policy.expert_hidden_dim = c->moe_intermediate_size;
    policy.norm_topk_prob = c->moe_norm_topk_prob;
    policy.expert_weights_scale = c->moe_expert_weights_scale;
    return policy;
}

int bn_moe_policy_uses_expert_weights(const BnConfig *c) {
    return bn_model_config_uses_moe(c);
}

int bn_moe_policy_uses_all_active_two_expert_set(const BnConfig *c) {
    return bn_model_config_uses_two_expert_all_active_moe(c);
}

int bn_moe_policy_uses_all_active_two_expert_route(const BnConfig *c,
                                                   int dim) {
    return bn_model_config_uses_all_active_two_expert_moe(c, dim);
}

int bn_moe_policy_uses_grouped_expert_route(const BnConfig *c) {
    return bn_model_config_uses_more_than_two_expert_moe(c);
}

int bn_moe_policy_normalizes_topk_route_weights(const BnConfig *c) {
    return c ? c->moe_norm_topk_prob : 0;
}

int bn_moe_policy_layer_has_router(const BnLayerWeights *lw) {
    return lw ? lw->moe.router_weight != NULL : 0;
}

int bn_moe_policy_has_shared_expert(const BnConfig *c,
                                    const BnLayerWeights *lw) {
    return (c && c->has_shared_expert) ||
           (lw && lw->shared.shared_expert_gate);
}

int bn_moe_policy_has_shared_expert_gate_vector(
    const BnLayerWeights *lw) {
    return lw && lw->shared.shared_expert_gate;
}

int bn_moe_policy_has_loaded_shared_gate_projection(
    const BnLayerWeights *lw) {
    return lw && lw->shared.shared_gate.data != NULL;
}

int bn_moe_policy_has_loaded_shared_expert_path(const BnConfig *c,
                                                const BnLayerWeights *lw) {
    return bn_moe_policy_has_shared_expert(c, lw) &&
           bn_moe_policy_has_loaded_shared_gate_projection(lw);
}

int bn_moe_policy_has_loaded_shared_expert(const BnConfig *c,
                                           const BnLayerWeights *lw) {
    return c &&
           c->has_shared_expert &&
           lw &&
           lw->shared.shared_gate.data != NULL;
}

int bn_moe_policy_shared_expert_hidden_dim(const BnConfig *c) {
    return bn_model_config_shared_expert_hidden_dim(c);
}

int bn_moe_policy_supports_resident_routed_ffn_shape(
    int dim,
    int expert_hidden_dim,
    const BnMoEExpertMap *em) {
    return em &&
           em->gate_rows == expert_hidden_dim &&
           em->up_rows == expert_hidden_dim &&
           em->gate_cols == dim &&
           em->up_cols == dim &&
           em->down_rows == dim &&
           em->down_cols == expert_hidden_dim;
}

int bn_moe_policy_supports_resident_routed_ffn_layout(
    const BnConfig *c,
    const BnMoEExpertMap *em) {
    BnMoERoutePolicy route_policy = bn_moe_route_policy(c);
    return c &&
           bn_moe_policy_supports_resident_routed_ffn_shape(
               c->dim, route_policy.expert_hidden_dim, em);
}

int bn_moe_policy_supports_gateup_split_layout(const BnMoEExpertMap *em) {
    return em &&
           em->gate_rows == em->up_rows &&
           em->gate_cols == em->up_cols;
}

int bn_moe_policy_supports_shared_gateup_batch_type(int shared_gate_type,
                                                    int shared_up_type,
                                                    int batch_type) {
    return bn_backend_quant_shared_gateup_batch_type_supported(
        shared_gate_type, shared_up_type, batch_type);
}

int bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
    int shared_gate_type,
    int shared_up_type,
    int batch_type,
    int mixed_shared_gateup_supported) {
    if (!bn_moe_policy_supports_shared_gateup_batch_type(
            shared_gate_type, shared_up_type, batch_type))
        return 0;
    if (mixed_shared_gateup_supported)
        return 1;
    return bn_backend_quant_same_quant_format_pair_stackable(shared_gate_type,
                                                     batch_type) &&
           bn_backend_quant_same_quant_format_pair_stackable(shared_up_type,
                                                     batch_type);
}

int bn_moe_policy_can_batch_loaded_shared_gateup(
    const BnMatvecTask *tasks,
    int n_tasks,
    const BnLayerWeights *lw) {
    BnMoESharedExpertWeights weights;
    if (!bn_moe_shared_expert_projection_weights(&weights, lw))
        return 0;
    return bn_moe_can_batch_shared_gateup(
        tasks, n_tasks, weights.gate->type, weights.up->type);
}

int bn_moe_shared_expert_gateup_tasks(BnMatvecTask *tasks,
                                      float *gate_out,
                                      float *up_out,
                                      const BnLayerWeights *lw,
                                      uint32_t flags) {
    BnMoESharedExpertWeights weights;
    if (!tasks || !gate_out || !up_out ||
        !bn_moe_shared_expert_projection_weights(&weights, lw))
        return 0;
    tasks[0] = (BnMatvecTask){
        gate_out, weights.gate, NULL, flags
    };
    tasks[1] = (BnMatvecTask){
        up_out, weights.up, NULL, flags
    };
    return 2;
}

const BnQWeight *bn_moe_shared_expert_down_weight(const BnLayerWeights *lw) {
    BnMoESharedExpertWeights weights;
    return bn_moe_shared_expert_projection_weights(&weights, lw)
        ? weights.down
        : NULL;
}

int bn_moe_shared_expert_projection_weights(
    BnMoESharedExpertWeights *out,
    const BnLayerWeights *lw) {
    if (!out || !bn_moe_policy_has_loaded_shared_gate_projection(lw))
        return 0;
    out->gate = &lw->shared.shared_gate;
    out->up = &lw->shared.shared_up;
    out->down = &lw->shared.shared_down;
    return 1;
}

int bn_moe_quant_uses_embedded_tensor_scale(int type) {
    return bn_backend_quant_has_embedded_tensor_scale(type);
}

size_t bn_moe_quant_embedded_tensor_scale_offset(int type, int rows, int cols) {
    return bn_backend_quant_embedded_tensor_scale_offset(type, rows, cols);
}

void bn_moe_quant_matvec_gateup_gpu_buffers(BnMatvecTask *tasks,
                                            const void **buffers,
                                            int n_tasks,
                                            const float *x,
                                            int8_t *quantized_buf,
                                            BnThreadPool *pool,
                                            BnGPUBackend *gpu) {
    bn_backend_quant_matvec_batch_gpu_buf(tasks, buffers, n_tasks, x,
                                          quantized_buf, pool, gpu);
}

void bn_moe_quant_matvec_down_gpu_buffer(float *out,
                                         const BnQWeight *W,
                                         void *W_buf,
                                         const float *x,
                                         int8_t *quantized_buf,
                                         BnThreadPool *pool,
                                         BnGPUBackend *gpu) {
    bn_backend_quant_matvec_gpu_buf(out, W, W_buf, x, quantized_buf, pool, gpu);
}
