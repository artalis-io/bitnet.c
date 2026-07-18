#include "transformer_prefill_internal.h"
#include "backend_quant.h"

#include <stdlib.h>

int bn_transformer_prefill_profile_enabled(void) {
    return getenv("BN_PREFILL_PROFILE") != NULL;
}

int bn_transformer_prefill_hybrid_batch_allowed(void) {
    return getenv("BN_PREFILL_ALLOW_HYBRID_BATCH") != NULL;
}

int bn_transformer_prefill_force_token_attention_enabled(void) {
    return getenv("BN_PREFILL_FORCE_TOKEN_ATTN") != NULL;
}

BnTransformerPrefillLayerKindPolicy
bn_transformer_prefill_layer_kind_policy(const void *moe_router_weight) {
    BnTransformerPrefillLayerKindPolicy policy = {0};
    policy.uses_moe = moe_router_weight != NULL;
    return policy;
}

int bn_transformer_prefill_can_preq8k_type(const BnPrefillCPUOps *ops,
                                           int tensor_type) {
    return ops && ops->supports_preq8k &&
           bn_backend_quant_can_preq8k(tensor_type);
}

int bn_transformer_prefill_can_preq8k_pair(const BnPrefillCPUOps *ops,
                                           int left_type,
                                           int right_type) {
    return bn_transformer_prefill_can_preq8k_type(ops, left_type) &&
           bn_backend_quant_can_preq8k(right_type);
}

int bn_transformer_prefill_can_preq8k_triple(const BnPrefillCPUOps *ops,
                                             int first_type,
                                             int second_type,
                                             int third_type) {
    return bn_transformer_prefill_can_preq8k_pair(ops, first_type,
                                                  second_type) &&
           bn_backend_quant_can_preq8k(third_type);
}

int bn_transformer_prefill_route_preq8k_type_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int force_float_kquant,
    int dim,
    int tensor_type) {
    return !gpu &&
           !force_float_kquant &&
           dim > 0 &&
           dim % BN_QK_K == 0 &&
           bn_transformer_prefill_can_preq8k_type(ops, tensor_type);
}

int bn_transformer_prefill_route_preq8k_pair_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int force_float_kquant,
    int dim,
    int left_type,
    int right_type) {
    return bn_transformer_prefill_route_preq8k_type_enabled(
               ops, gpu, force_float_kquant, dim, left_type) &&
           bn_backend_quant_can_preq8k(right_type);
}

int bn_transformer_prefill_route_preq8k_triple_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int force_float_kquant,
    int dim,
    int first_type,
    int second_type,
    int third_type) {
    return bn_transformer_prefill_route_preq8k_pair_enabled(
               ops, gpu, force_float_kquant, dim, first_type, second_type) &&
           bn_backend_quant_can_preq8k(third_type);
}

int bn_transformer_prefill_stacked_pair_same_format(int left_type,
                                                    int right_type) {
    return bn_backend_quant_stacked_pair_same_format(left_type, right_type);
}

int bn_transformer_prefill_uses_float_kquant_fallback(int tensor_type) {
    return bn_backend_quant_is_kquant_float_fallback_candidate(tensor_type);
}

void bn_transformer_prefill_quant_matmul_gpu_buffer(float *out,
                                                    const BnQWeight *W,
                                                    void *W_buf,
                                                    const float *X,
                                                    int n_tokens,
                                                    int8_t *x_q_buf,
                                                    BnThreadPool *pool,
                                                    BnGPUBackend *gpu) {
    bn_backend_quant_matmul_gpu_buf(out, W, W_buf, X, n_tokens, x_q_buf,
                                    pool, gpu);
}

void bn_transformer_prefill_quant_matmul_batch_gpu_buffers(
    const BnMatvecTask *tasks,
    const void **buffers,
    int n_tasks,
    const float *X,
    int n_tokens,
    int cols,
    int8_t *x_q_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu) {
    bn_backend_quant_matmul_batch_gpu_buf(tasks, buffers, n_tasks, X,
                                          n_tokens, cols, x_q_buf, pool, gpu);
}
