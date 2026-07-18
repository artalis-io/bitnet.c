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

BnTransformerPrefillDenseLayerBatchPolicy
bn_transformer_prefill_dense_layer_batch_policy(
    int gpu_available,
    int tq_state_available,
    int dense_chain_enabled,
    int n_tokens,
    int min_tokens,
    int pos0,
    float layer_rope_theta,
    float config_rope_theta,
    BnTransformerPrefillLayerKindPolicy layer_kind,
    int has_ffn_gate,
    int has_ffn_up,
    int has_q_bias,
    int has_k_bias,
    int has_v_bias,
    int has_attn_sub_norm,
    int has_ffn_sub_norm,
    int has_layer_output_scale,
    int uses_post_norm,
    int has_attn_post_norm,
    int has_ffn_post_norm) {
    BnTransformerPrefillDenseLayerBatchPolicy policy = {0};
    policy.enabled =
        gpu_available &&
        !tq_state_available &&
        dense_chain_enabled &&
        n_tokens >= min_tokens &&
        pos0 == 0 &&
        layer_rope_theta == config_rope_theta &&
        !layer_kind.uses_moe &&
        has_ffn_gate &&
        has_ffn_up &&
        !has_q_bias &&
        !has_k_bias &&
        !has_v_bias &&
        !has_attn_sub_norm &&
        !has_ffn_sub_norm &&
        !has_layer_output_scale &&
        !(uses_post_norm && (has_attn_post_norm || has_ffn_post_norm));
    return policy;
}

BnTransformerPrefillRawAttentionPolicy
bn_transformer_prefill_raw_attention_policy(
    int gpu_available,
    int raw_attention_hook_available,
    int norm_resid_hook_available,
    int attn_norm_buffer_available,
    int tq_state_available,
    int q_gated,
    int pos0,
    int n_tokens,
    int min_tokens,
    float layer_rope_theta,
    float config_rope_theta,
    int has_q_bias,
    int has_k_bias,
    int has_v_bias,
    int has_attn_sub_norm,
    int uses_post_norm,
    int has_attn_post_norm) {
    BnTransformerPrefillRawAttentionPolicy policy = {0};
    policy.eligible =
        gpu_available &&
        raw_attention_hook_available &&
        !tq_state_available &&
        !q_gated &&
        pos0 == 0 &&
        layer_rope_theta == config_rope_theta &&
        !has_q_bias &&
        !has_k_bias &&
        !has_v_bias &&
        !has_attn_sub_norm &&
        !(uses_post_norm && has_attn_post_norm);
    policy.fuses_input_norm =
        policy.eligible &&
        norm_resid_hook_available &&
        attn_norm_buffer_available &&
        n_tokens >= min_tokens;
    return policy;
}

BnTransformerPrefillAttentionBatchPolicy
bn_transformer_prefill_attention_batch_policy(
    int raw_attention_already_used,
    int gpu_available,
    int attention_hook_available,
    int attention_wo_hook_available,
    int attention_feature_enabled,
    int wo_buffer_available,
    int n_tokens,
    int min_tokens,
    int has_attn_sub_norm,
    int uses_post_norm,
    int has_attn_post_norm) {
    BnTransformerPrefillAttentionBatchPolicy policy = {0};
    policy.eligible =
        !raw_attention_already_used &&
        gpu_available &&
        attention_hook_available &&
        attention_feature_enabled &&
        n_tokens >= min_tokens;
    policy.fuses_output_projection =
        policy.eligible &&
        attention_wo_hook_available &&
        wo_buffer_available &&
        !has_attn_sub_norm &&
        !(uses_post_norm && has_attn_post_norm);
    return policy;
}

BnTransformerPrefillAttentionBatchCallPolicy
bn_transformer_prefill_attention_batch_call_policy(
    BnTransformerPrefillAttentionBatchPolicy policy) {
    BnTransformerPrefillAttentionBatchCallPolicy call_policy = {0};
    call_policy.preferred_kind = policy.fuses_output_projection
        ? BN_TRANSFORMER_PREFILL_ATTENTION_BATCH_WO
        : BN_TRANSFORMER_PREFILL_ATTENTION_BATCH_PLAIN;
    return call_policy;
}

BnTransformerPrefillFFNBatchPolicy
bn_transformer_prefill_ffn_batch_policy(
    int has_ffn_gate,
    int tokens_allowed,
    int ffn_batch_norm_resid_hook_available,
    int ffn_norm_buffer_available,
    int n_tokens,
    int min_tokens,
    int uses_hybrid_layer_layout,
    int has_ffn_sub_norm,
    int uses_post_norm,
    int has_ffn_post_norm) {
    BnTransformerPrefillFFNBatchPolicy policy = {0};
    int compatible_norms =
        !has_ffn_sub_norm &&
        !(uses_post_norm && has_ffn_post_norm);
    policy.eligible =
        has_ffn_gate &&
        compatible_norms &&
        tokens_allowed &&
        (!uses_hybrid_layer_layout || n_tokens >= min_tokens);
    policy.fuses_norm_residual =
        policy.eligible &&
        ffn_batch_norm_resid_hook_available &&
        ffn_norm_buffer_available &&
        n_tokens >= min_tokens;
    return policy;
}

BnTransformerPrefillFFNBatchCallPolicy
bn_transformer_prefill_ffn_batch_call_policy(
    int norm_buffer_available,
    int add_residual,
    int ffn_batch_norm_hook_available,
    int ffn_batch_norm_resid_hook_available) {
    BnTransformerPrefillFFNBatchCallPolicy policy = {0};
    if (norm_buffer_available && add_residual &&
        ffn_batch_norm_resid_hook_available) {
        policy.kind = BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM_RESID;
    } else if (norm_buffer_available && ffn_batch_norm_hook_available) {
        policy.kind = BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM;
    } else {
        policy.kind = BN_TRANSFORMER_PREFILL_FFN_BATCH_PLAIN;
    }
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
