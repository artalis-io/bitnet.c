#include "moe_internal.h"
#include "backend_quant.h"
#include "model_internal.h"
#include "transformer_cpu_backend_internal.h"

uint32_t bn_moe_gateup_task_flags(const BnConfig *c) {
    return bn_model_config_moe_forces_float_kquant_gateup(c)
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

BnMoEPrefillPolicy bn_moe_prefill_policy(const BnConfig *c) {
    BnMoEPrefillPolicy policy = {0};
    policy.force_matvec_prefill =
        bn_model_config_moe_prefill_forces_matvec(c);
    policy.uses_grouped_expert_route =
        bn_model_config_uses_more_than_two_expert_moe(c);
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

int bn_moe_policy_layer_has_router(const BnLayerWeights *lw) {
    return lw ? lw->moe.router_weight != NULL : 0;
}

int bn_moe_policy_has_shared_expert(const BnConfig *c,
                                    const BnLayerWeights *lw) {
    return (c && c->has_shared_expert) ||
           (lw && lw->shared.shared_expert_gate);
}

int bn_moe_policy_has_loaded_shared_expert(const BnConfig *c,
                                           const BnLayerWeights *lw) {
    return c &&
           c->has_shared_expert &&
           lw &&
           lw->shared.shared_gate.data != NULL;
}

int bn_moe_policy_supports_resident_routed_ffn_layout(
    const BnConfig *c,
    const BnMoEExpertMap *em) {
    return c && em &&
           em->gate_rows == c->moe_intermediate_size &&
           em->up_rows == c->moe_intermediate_size &&
           em->gate_cols == c->dim &&
           em->up_cols == c->dim &&
           em->down_rows == c->dim &&
           em->down_cols == c->moe_intermediate_size;
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
    int batch_type) {
    if (!bn_moe_policy_supports_shared_gateup_batch_type(
            shared_gate_type, shared_up_type, batch_type))
        return 0;
    if (bn_transformer_cpu_backend_supports_mixed_shared_gateup_batch())
        return 1;
    return bn_backend_quant_same_quant_format_pair_stackable(shared_gate_type,
                                                     batch_type) &&
           bn_backend_quant_same_quant_format_pair_stackable(shared_up_type,
                                                     batch_type);
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
