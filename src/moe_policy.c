#include "moe_internal.h"
#include "backend_quant.h"
#include "model_internal.h"

static int moe_requires_float_kquant_gateup_fallback(const BnConfig *c) {
    return bn_model_config_moe_requires_float_kquant_gateup_fallback(c);
}

static int moe_uses_scaled_router_input(const BnConfig *c) {
    return bn_model_config_moe_uses_scaled_router_input(c);
}

static int moe_uses_dense_residual_branch(const BnConfig *c) {
    return bn_model_config_moe_uses_dense_residual_branch(c);
}

static int moe_uses_reference_silu(const BnConfig *c) {
    return bn_model_config_moe_uses_reference_silu(c);
}

static int moe_config_activation(const BnConfig *c) {
    return bn_model_config_activation(c);
}

static float moe_norm_epsilon(const BnConfig *c) {
    return bn_model_config_norm_epsilon(c);
}

static int moe_prefill_requires_matvec(const BnConfig *c) {
    return bn_model_config_moe_prefill_requires_matvec(c);
}

static int moe_uses_grouped_expert_route(const BnConfig *c) {
    return bn_model_config_uses_more_than_two_expert_moe(c);
}

static int moe_total_experts(const BnConfig *c) {
    return bn_model_config_moe_total_experts(c);
}

static int moe_active_experts(const BnConfig *c) {
    return bn_model_config_moe_active_experts(c);
}

static int moe_expert_hidden_dim(const BnConfig *c) {
    return bn_model_config_moe_expert_hidden_dim(c);
}

static int moe_normalizes_topk_route_weights(const BnConfig *c) {
    return bn_model_config_moe_normalizes_topk_route_weights(c);
}

static float moe_expert_weights_scale(const BnConfig *c) {
    return bn_model_config_moe_expert_weights_scale(c);
}

static int moe_uses_expert_weights(const BnConfig *c) {
    return bn_model_config_uses_moe(c);
}

static int moe_uses_all_active_two_expert_set(const BnConfig *c) {
    return bn_model_config_uses_two_expert_all_active_moe(c);
}

static int moe_uses_all_active_two_expert_route(const BnConfig *c,
                                                int dim) {
    return bn_model_config_uses_all_active_two_expert_moe(c, dim);
}

static int moe_has_configured_shared_expert(const BnConfig *c) {
    return bn_model_config_has_shared_expert(c);
}

static int moe_shared_expert_hidden_dim(const BnConfig *c) {
    return bn_model_config_shared_expert_hidden_dim(c);
}

uint32_t bn_moe_float_kquant_gateup_fallback_task_flags(const BnConfig *c) {
    return moe_requires_float_kquant_gateup_fallback(c)
        ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT
        : 0u;
}

BnMoEExecutionPolicy bn_moe_execution_policy(const BnConfig *c) {
    BnMoEExecutionPolicy policy = {0};
    policy.uses_reference_silu = -1;
    if (!c)
        return policy;
    policy.uses_scaled_router_input =
        moe_uses_scaled_router_input(c);
    policy.uses_dense_residual_branch =
        moe_uses_dense_residual_branch(c);
    policy.uses_reference_silu = policy.uses_dense_residual_branch
        ? -1
        : moe_uses_reference_silu(c);
    policy.activation = moe_config_activation(c);
    policy.norm_eps = moe_norm_epsilon(c);
    return policy;
}

int bn_moe_policy_uses_reference_silu(const BnConfig *c) {
    return bn_moe_execution_policy(c).uses_reference_silu;
}

BnMoEPrefillPolicy bn_moe_prefill_policy(const BnConfig *c) {
    BnMoEPrefillPolicy policy = {0};
    policy.requires_matvec_prefill =
        moe_prefill_requires_matvec(c);
    policy.uses_grouped_expert_route =
        moe_uses_grouped_expert_route(c);
    return policy;
}

BnMoERoutePolicy bn_moe_route_policy(const BnConfig *c) {
    BnMoERoutePolicy policy = {0};
    if (!c)
        return policy;
    policy.total_experts = moe_total_experts(c);
    policy.active_experts = moe_active_experts(c);
    policy.expert_hidden_dim = moe_expert_hidden_dim(c);
    policy.norm_topk_prob = moe_normalizes_topk_route_weights(c);
    policy.expert_weights_scale = moe_expert_weights_scale(c);
    return policy;
}

int bn_moe_policy_uses_expert_weights(const BnConfig *c) {
    return moe_uses_expert_weights(c);
}

int bn_moe_policy_uses_all_active_two_expert_set(const BnConfig *c) {
    return moe_uses_all_active_two_expert_set(c);
}

int bn_moe_policy_uses_all_active_two_expert_route(const BnConfig *c,
                                                   int dim) {
    return moe_uses_all_active_two_expert_route(c, dim);
}

int bn_moe_policy_uses_grouped_expert_route(const BnConfig *c) {
    return moe_uses_grouped_expert_route(c);
}

int bn_moe_policy_normalizes_topk_route_weights(const BnConfig *c) {
    return moe_normalizes_topk_route_weights(c);
}

int bn_moe_policy_layer_has_router(const BnLayerWeights *lw) {
    return lw ? lw->moe.router_weight != NULL : 0;
}

int bn_moe_policy_has_shared_expert(const BnConfig *c,
                                    const BnLayerWeights *lw) {
    return moe_has_configured_shared_expert(c) ||
           (lw && lw->shared.shared_expert_gate);
}

int bn_moe_policy_has_shared_expert_gate_vector(
    const BnLayerWeights *lw) {
    return lw && lw->shared.shared_expert_gate;
}

const float *bn_moe_shared_expert_gate_vector(const BnLayerWeights *lw) {
    return bn_moe_policy_has_shared_expert_gate_vector(lw)
        ? lw->shared.shared_expert_gate
        : NULL;
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
    return moe_has_configured_shared_expert(c) &&
           lw &&
           lw->shared.shared_gate.data != NULL;
}

int bn_moe_policy_shared_expert_hidden_dim(const BnConfig *c) {
    return moe_shared_expert_hidden_dim(c);
}

BnMoELoadedSharedExpertPolicy
bn_moe_loaded_shared_expert_policy(const BnConfig *c,
                                   const BnLayerWeights *lw) {
    BnMoELoadedSharedExpertPolicy policy = {0};
    policy.has_loaded_path = bn_moe_policy_has_loaded_shared_expert(c, lw);
    if (policy.has_loaded_path)
        policy.hidden_dim = bn_moe_policy_shared_expert_hidden_dim(c);
    return policy;
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
