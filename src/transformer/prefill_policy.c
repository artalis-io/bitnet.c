#include "transformer_prefill_internal.h"
#include "backend_quant.h"
#include "gpu_internal.h"
#include "gpu_policy.h"
#include "model_arch.h"

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

BnTransformerPrefillSharedAll2DecodeFallbackPolicy
bn_transformer_prefill_shared_all2_decode_fallback_policy(
    const BnConfig *c,
    int gpu_available) {
    BnTransformerPrefillSharedAll2DecodeFallbackPolicy policy = {0};
    policy.enabled =
        c &&
        bn_model_arch_uses_all_active_two_expert_moe(c, c->dim) &&
        c->has_shared_expert &&
        !gpu_available;
    return policy;
}

BnTransformerPrefillSequencePolicy
bn_transformer_prefill_sequence_policy(const BnConfig *c) {
    BnTransformerPrefillSequencePolicy policy = {0};
    policy.uses_hybrid_layer_layout =
        bn_model_arch_uses_hybrid_layer_layout(c);
    policy.uses_hybrid_ssm =
        bn_model_arch_uses_hybrid_ssm(c);
    policy.uses_large_dense_hybrid_ssm =
        bn_model_arch_uses_large_dense_hybrid_ssm(c);
    return policy;
}

int bn_transformer_prefill_hybrid_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_hybrid_prefill_chain_applicable(gpu, c);
}

int bn_transformer_prefill_moe_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_moe_prefill_chain_applicable(gpu, c);
}

int bn_transformer_prefill_small_dense_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_cuda_small_dense_prefill_chain_applicable(
        gpu, c);
}

int bn_transformer_prefill_moe_enabled(void) {
    return bn_transformer_gpu_cuda_moe_prefill_enabled();
}

int bn_transformer_prefill_large_hybrid_disabled(void) {
    return bn_transformer_gpu_cuda_large_hybrid_prefill_disabled();
}

BnTransformerPrefillDecodeFallbackPolicy
bn_transformer_prefill_decode_fallback_policy(
    BnTransformerPrefillSequencePolicy sequence,
    int gpu_moe_prefill,
    int moe_prefill_enabled,
    int n_tokens,
    int moe_min_tokens,
    int cuda_small_dense_prefill_chain,
    int small_dense_min_tokens,
    int gpu_hybrid_prefill,
    int large_hybrid_prefill_disabled,
    int hybrid_batch_allowed) {
    BnTransformerPrefillDecodeFallbackPolicy policy = {0};
    int hybrid_batch_decode =
        sequence.uses_hybrid_ssm &&
        !gpu_hybrid_prefill &&
        !hybrid_batch_allowed;
    policy.decode =
        (gpu_moe_prefill &&
         (!moe_prefill_enabled || n_tokens < moe_min_tokens)) ||
        (cuda_small_dense_prefill_chain &&
         n_tokens < small_dense_min_tokens) ||
        (gpu_hybrid_prefill &&
         sequence.uses_large_dense_hybrid_ssm &&
         large_hybrid_prefill_disabled) ||
        hybrid_batch_decode;
    policy.require_logits_decode = hybrid_batch_decode;
    return policy;
}

BnTransformerPrefillDenseModelChainPolicy
bn_transformer_prefill_dense_model_chain_policy(
    int dense_chain_enabled,
    int gpu_available,
    int pos0,
    int n_layers) {
    BnTransformerPrefillDenseModelChainPolicy policy = {0};
    policy.enabled =
        dense_chain_enabled &&
        gpu_available &&
        pos0 == 0 &&
        n_layers > 0;
    return policy;
}

BnTransformerPrefillHybridModelChainPolicy
bn_transformer_prefill_hybrid_model_chain_policy(
    int hybrid_chain_enabled,
    int gpu_hybrid_prefill,
    int pos0,
    int n_layers,
    int tq_state_available) {
    BnTransformerPrefillHybridModelChainPolicy policy = {0};
    policy.enabled =
        hybrid_chain_enabled &&
        gpu_hybrid_prefill &&
        pos0 == 0 &&
        n_layers > 0 &&
        !tq_state_available;
    return policy;
}

int bn_transformer_prefill_hybrid_chain_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_cuda_prefill_hybrid_chain_enabled(gpu, c);
}

int bn_transformer_prefill_hybrid_chain_debug_enabled(void) {
    return bn_transformer_gpu_cuda_prefill_hybrid_chain_debug_enabled();
}

BnTransformerPrefillAttentionModePolicy
bn_transformer_prefill_attention_mode_policy(
    int tq_state_available,
    int force_token_attention_requested,
    int gpu_hybrid_prefill) {
    BnTransformerPrefillAttentionModePolicy policy = {0};
    policy.use_batched_attention =
        !tq_state_available &&
        !force_token_attention_requested &&
        !gpu_hybrid_prefill;
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

BnTransformerPrefillDenseLayerChainPolicy
bn_transformer_prefill_dense_layer_chain_policy(
    int gpu_available,
    int dense_layer_hook_available,
    int tq_state_available,
    int n_tokens,
    int min_tokens,
    float layer_rope_theta,
    float config_rope_theta,
    int is_attn,
    BnTransformerPrefillLayerKindPolicy layer_kind,
    int has_ffn_gate,
    int has_ffn_up,
    int has_attn_sub_norm,
    int has_ffn_sub_norm,
    int has_layer_output_scale,
    int uses_post_norm,
    int has_attn_post_norm,
    int has_ffn_post_norm) {
    BnTransformerPrefillDenseLayerChainPolicy policy = {0};
    policy.enabled =
        gpu_available &&
        dense_layer_hook_available &&
        !tq_state_available &&
        n_tokens >= min_tokens &&
        layer_rope_theta == config_rope_theta &&
        is_attn &&
        !layer_kind.uses_moe &&
        has_ffn_gate &&
        has_ffn_up &&
        !has_attn_sub_norm &&
        !has_ffn_sub_norm &&
        !has_layer_output_scale &&
        !(uses_post_norm && (has_attn_post_norm || has_ffn_post_norm));
    return policy;
}

int bn_transformer_prefill_dense_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_prefill_dense_chain_enabled(void) {
    return bn_transformer_gpu_cuda_prefill_dense_chain_enabled();
}

int bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens) {
    return bn_transformer_gpu_dense_ffn_batch_tokens_allowed(gpu, c,
                                                             n_tokens);
}

int bn_transformer_prefill_attention_min_tokens(void) {
    return bn_transformer_gpu_cuda_prefill_attention_min_tokens();
}

int bn_transformer_prefill_attention_enabled(void) {
    return bn_transformer_gpu_cuda_prefill_attention_enabled();
}

BnTransformerPrefillSSMChainPolicy
bn_transformer_prefill_ssm_chain_policy(
    int chain_available,
    BnTransformerPrefillLayerKindPolicy layer_kind,
    int has_ffn_gate,
    int has_ffn_up,
    int has_ffn_sub_norm,
    int has_layer_output_scale,
    int uses_post_norm,
    int has_attn_post_norm,
    int has_ffn_post_norm,
    int ssm_time_step_rank,
    int ssm_state_size,
    int ssm_inner_size,
    int ssm_group_count) {
    BnTransformerPrefillSSMChainPolicy policy = {0};
    policy.enabled =
        chain_available &&
        !layer_kind.uses_moe &&
        has_ffn_gate &&
        has_ffn_up &&
        !has_ffn_sub_norm &&
        !has_layer_output_scale &&
        !(uses_post_norm && (has_attn_post_norm || has_ffn_post_norm)) &&
        ssm_time_step_rank > 0 &&
        ssm_state_size > 0 &&
        ssm_inner_size > 0 &&
        ssm_group_count > 0;
    return policy;
}

BnTransformerPrefillSSMMoEChainPolicy
bn_transformer_prefill_ssm_moe_chain_policy(
    int chain_available,
    BnTransformerPrefillLayerKindPolicy layer_kind,
    int has_ffn_sub_norm,
    int has_layer_output_scale,
    int uses_post_norm,
    int has_attn_post_norm,
    int has_ffn_post_norm,
    int ssm_time_step_rank,
    int ssm_state_size,
    int ssm_inner_size,
    int ssm_group_count) {
    BnTransformerPrefillSSMMoEChainPolicy policy = {0};
    policy.enabled =
        chain_available &&
        layer_kind.uses_moe &&
        !has_ffn_sub_norm &&
        !has_layer_output_scale &&
        !(uses_post_norm && (has_attn_post_norm || has_ffn_post_norm)) &&
        ssm_time_step_rank > 0 &&
        ssm_state_size > 0 &&
        ssm_inner_size > 0 &&
        ssm_group_count > 0;
    return policy;
}

int bn_transformer_prefill_ssm_layer_backend_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_prefill_ssm_layer_backend_available(gpu);
}

int bn_transformer_prefill_ssm_dense_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens) {
    return bn_transformer_gpu_cuda_prefill_ssm_dense_chain_available(
        gpu, c, n_tokens);
}

int bn_transformer_prefill_ssm_run_chain_enabled(void) {
    return bn_transformer_gpu_cuda_prefill_ssm_run_chain_enabled();
}

int bn_transformer_prefill_moe_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down) {
    return bn_transformer_gpu_cuda_prefill_moe_ffn_batch_available(
        gpu, c, map, dim, allow_q4_down);
}

int bn_transformer_prefill_moe_layer_backend_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down) {
    return bn_transformer_gpu_prefill_moe_layer_backend_available(
        gpu, c, map, dim, allow_q4_down);
}

int bn_transformer_prefill_ssm_moe_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down,
    int n_tokens) {
    return bn_transformer_gpu_cuda_prefill_ssm_moe_chain_available(
        gpu, c, map, dim, allow_q4_down, n_tokens);
}

int bn_transformer_prefill_moe_layer_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down,
    int n_tokens) {
    return bn_transformer_gpu_prefill_moe_layer_chain_available(
        gpu, c, map, dim, allow_q4_down, n_tokens);
}

int bn_transformer_prefill_moe_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(c, gpu);
}

int bn_transformer_prefill_moe_chain_debug_enabled(void) {
    return bn_transformer_gpu_cuda_prefill_moe_chain_debug_enabled();
}

int bn_transformer_prefill_ssm_ffn_fuse_allowed(void) {
    return bn_transformer_gpu_cuda_prefill_ssm_ffn_fuse_allowed();
}

BnTransformerPrefillSSMFFNFusePolicy
bn_transformer_prefill_ssm_ffn_fuse_policy(
    int fuse_requested,
    int fuse_allowed,
    int has_ffn_gate_weight,
    int has_ffn_up,
    int has_ffn_down,
    int has_ffn_gate_config,
    int has_ffn_sub_norm,
    int has_layer_output_scale,
    int uses_ffn_post_norm,
    int has_ffn_post_norm) {
    BnTransformerPrefillSSMFFNFusePolicy policy = {0};
    policy.enabled =
        fuse_requested &&
        fuse_allowed &&
        has_ffn_gate_weight &&
        has_ffn_up &&
        has_ffn_down &&
        has_ffn_gate_config &&
        !has_ffn_sub_norm &&
        !has_layer_output_scale &&
        !(uses_ffn_post_norm && has_ffn_post_norm);
    return policy;
}

BnTransformerPrefillSSMStateUploadPolicy
bn_transformer_prefill_ssm_state_upload_policy(
    const BnConfig *c,
    int gpu_attached) {
    BnTransformerPrefillSSMStateUploadPolicy policy = {0};
    policy.upload = gpu_attached &&
                    bn_model_arch_uses_hybrid_ssm(c) &&
                    bn_gpu_policy_cuda_prefill_ssm_layer_disabled();
    return policy;
}

BnTransformerPrefillEntryPolicy
bn_transformer_prefill_entry_policy(
    int no_prefill,
    int parity_cpu,
    int n_tokens,
    int gpu_attached,
    int gpu_batch_prefill_enabled) {
    BnTransformerPrefillEntryPolicy policy = {0};
    policy.batch =
        !no_prefill &&
        !parity_cpu &&
        n_tokens > 1 &&
        (!gpu_attached || gpu_batch_prefill_enabled);
    return policy;
}

BnTransformerPrefillKVUploadPolicy
bn_transformer_prefill_kv_upload_policy(
    int gpu_attached,
    int gpu_kv_direct_valid) {
    BnTransformerPrefillKVUploadPolicy policy = {0};
    policy.upload = gpu_attached && !gpu_kv_direct_valid;
    return policy;
}

BnTransformerPrefillChainKVPolicy
bn_transformer_prefill_chain_kv_policy(
    int direct_gpu_kv_requested) {
    BnTransformerPrefillChainKVPolicy policy = {0};
    policy.write_host_kv = !direct_gpu_kv_requested;
    policy.mark_direct_valid = direct_gpu_kv_requested;
    return policy;
}

int bn_transformer_prefill_direct_kv_allowed(
    const BnConfig *c,
    const BnWeights *w,
    const BnGPUBackend *gpu,
    int pos0,
    int n_tokens) {
    return bn_transformer_gpu_cuda_prefill_direct_kv_allowed(
        c, w, gpu, pos0, n_tokens);
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

BnTransformerPrefillRawAttentionCallPolicy
bn_transformer_prefill_raw_attention_call_policy(
    BnTransformerPrefillRawAttentionPolicy policy) {
    BnTransformerPrefillRawAttentionCallPolicy call_policy = {0};
    call_policy.preferred_kind = policy.fuses_input_norm
        ? BN_TRANSFORMER_PREFILL_RAW_ATTENTION_NORM_RESID
        : BN_TRANSFORMER_PREFILL_RAW_ATTENTION_PLAIN;
    return call_policy;
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

int bn_transformer_prefill_qk_stack_compatible(const BnQWeight *q,
                                               const BnQWeight *k,
                                               int q_stride,
                                               int dim) {
    return q && k &&
           bn_transformer_prefill_stacked_pair_same_format(q->type,
                                                           k->type) &&
           q->cols == dim &&
           k->cols == dim &&
           q_stride >= q->rows + k->rows;
}

int bn_transformer_prefill_qkv_stack_batch_compatible(const BnQWeight *q,
                                                      const BnQWeight *k,
                                                      const BnQWeight *v,
                                                      int q_stride,
                                                      int dim) {
    return bn_transformer_prefill_qk_stack_compatible(q, k, q_stride, dim) &&
           v &&
           v->cols == dim;
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
