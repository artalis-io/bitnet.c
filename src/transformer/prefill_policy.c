#include "transformer_prefill_internal.h"
#include "backend_quant.h"
#include "gpu_internal.h"
#include "model_internal.h"
#include "transformer_plan_internal.h"
#include "../moe_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

int bn_transformer_prefill_profile_enabled(void) {
    return getenv("BN_PREFILL_PROFILE") != NULL;
}

int bn_transformer_prefill_hybrid_batch_allowed(void) {
    return getenv("BN_PREFILL_ALLOW_HYBRID_BATCH") != NULL;
}

int bn_transformer_prefill_requires_token_attention(void) {
    return getenv("BN_PREFILL_FORCE_TOKEN_ATTN") != NULL;
}

BnTransformerPrefillLayerKindPolicy
bn_transformer_prefill_layer_kind_policy(const BnLayerWeights *lw) {
    BnTransformerPrefillLayerKindPolicy policy = {0};
    policy.uses_moe = bn_transformer_moe_layer_has_router(lw);
    return policy;
}

BnTransformerPrefillSharedAllActiveTwoDecodeFallbackPolicy
bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
    const BnConfig *c,
    int gpu_available) {
    BnTransformerPrefillSharedAllActiveTwoDecodeFallbackPolicy policy = {0};
    BnTransformerMoESharedExpertShapePolicy shared_policy =
        bn_transformer_moe_shared_expert_shape_policy(c, NULL);
    policy.enabled =
        bn_transformer_moe_uses_configured_all_active_two_route(c) &&
        shared_policy.has_shared_expert &&
        !gpu_available;
    return policy;
}

BnTransformerPrefillFloatKQuantFallbackPolicy
bn_transformer_prefill_float_kquant_fallback_policy(const BnConfig *c) {
    BnTransformerPrefillFloatKQuantFallbackPolicy policy = {0};
    policy.enabled =
        bn_transformer_cpu_prefill_uses_float_kquant_fallback(c);
    policy.task_flags =
        bn_transformer_prefill_float_kquant_fallback_task_flags(policy.enabled);
    return policy;
}

BnTransformerPrefillQuantMatmulDispatchPolicy
bn_transformer_prefill_quant_matmul_dispatch_policy(
    int n_tasks,
    int max_cpu_batch_tasks,
    int gpu_available,
    int gpu_batch_available,
    int all_gpu_buffers_available,
    int float_kquant_fallback_enabled,
    int all_weights_float_kquant_fallback) {
    BnTransformerPrefillQuantMatmulDispatchPolicy policy = {0};
    if (n_tasks <= 0 || max_cpu_batch_tasks <= 0)
        return policy;

    policy.valid = 1;
    if (gpu_available) {
        policy.path =
            n_tasks > 1 &&
            gpu_batch_available &&
            all_gpu_buffers_available
                ? BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_BATCH
                : BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_SINGLE;
        return policy;
    }

    if (n_tasks <= max_cpu_batch_tasks &&
        float_kquant_fallback_enabled &&
        all_weights_float_kquant_fallback) {
        policy.path =
            BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_FLOAT_KQUANT_FALLBACK;
        return policy;
    }

    policy.path =
        n_tasks <= max_cpu_batch_tasks
            ? BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_PREPARED_MULTI
            : BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_SINGLE;
    return policy;
}

static int prefill_weight_uses_float_kquant_fallback(const BnQWeight *w) {
    return w && bn_transformer_prefill_uses_float_kquant_fallback(w->type);
}

static int prefill_all_weights_use_float_kquant_fallback(
    const BnQWeight *const *weights,
    int n_tasks) {
    if (!weights || n_tasks <= 0)
        return 0;
    for (int i = 0; i < n_tasks; i++) {
        if (!prefill_weight_uses_float_kquant_fallback(weights[i]))
            return 0;
    }
    return 1;
}

BnTransformerPrefillQuantMatmulDispatchPolicy
bn_transformer_prefill_quant_matmul_dispatch_policy_for(
    const BnConfig *c,
    const BnQWeight *const *weights,
    int n_tasks,
    int max_cpu_batch_tasks,
    int gpu_available,
    int gpu_batch_available,
    int all_gpu_buffers_available) {
    BnTransformerPrefillFloatKQuantFallbackPolicy fallback_policy =
        bn_transformer_prefill_float_kquant_fallback_policy(c);
    return bn_transformer_prefill_quant_matmul_dispatch_policy(
        n_tasks, max_cpu_batch_tasks, gpu_available, gpu_batch_available,
        all_gpu_buffers_available, fallback_policy.enabled,
        prefill_all_weights_use_float_kquant_fallback(weights, n_tasks));
}

BnTransformerPrefillSequencePolicy
bn_transformer_prefill_sequence_policy(const BnConfig *c) {
    BnTransformerPrefillSequencePolicy policy = {0};
    policy.uses_hybrid_layer_layout =
        bn_transformer_prefill_uses_hybrid_layer_layout(c);
    policy.uses_hybrid_ssm =
        bn_transformer_prefill_uses_hybrid_ssm(c);
    policy.uses_large_dense_hybrid_ssm =
        bn_transformer_prefill_uses_large_dense_hybrid_ssm(c);
    return policy;
}

static int prefill_add_mul_size(size_t *out, size_t a, int b) {
    if (!out || b < 0)
        return 0;
    size_t sb = (size_t)b;
    if (a != 0 && sb > (SIZE_MAX - *out) / a)
        return 0;
    *out += a * sb;
    return 1;
}

int bn_transformer_prefill_buffer_shape_policy(
    BnTransformerPrefillBufferShapePolicy *out,
    const BnConfig *c,
    BnTransformerPrefillSequencePolicy sequence_policy,
    int n_tokens,
    int dim,
    int max_q_dim,
    int max_rope_dims) {
    if (!out || !c || n_tokens <= 0 || dim <= 0 || max_q_dim < 0 ||
        max_rope_dims < 0 || c->kv_dim <= 0 || c->hidden_dim <= 0)
        return 0;
    if (max_q_dim > dim && max_q_dim > INT_MAX / 2)
        return 0;
    if (c->kv_dim > INT_MAX / 2)
        return 0;

    BnTransformerPrefillBufferShapePolicy policy = {0};
    policy.kv_dim = c->kv_dim;
    policy.hidden_dim = c->hidden_dim;
    policy.q_buf_stride = max_q_dim > dim ? max_q_dim * 2 : dim;
    policy.xb2_stride = dim;
    policy.hb_stride = c->hidden_dim;
    policy.hb2_stride = c->hidden_dim;
    policy.half_rope = max_rope_dims / 2;

    if (sequence_policy.uses_hybrid_ssm) {
        BnTransformerSSMShapePolicy ssm_shape;
        if (!bn_transformer_ssm_shape_policy(&ssm_shape, c))
            return 0;
        if (ssm_shape.qkv_dim > policy.q_buf_stride)
            policy.q_buf_stride = ssm_shape.qkv_dim;
        if (ssm_shape.value_dim > policy.xb2_stride)
            policy.xb2_stride = ssm_shape.value_dim;
        if (ssm_shape.value_dim > policy.hb_stride)
            policy.hb_stride = ssm_shape.value_dim;
        if (ssm_shape.value_dim > policy.hb2_stride)
            policy.hb2_stride = ssm_shape.value_dim;
    }

    size_t nt = (size_t)n_tokens;
    if (!prefill_add_mul_size(&policy.batch_floats, nt, dim) ||
        !prefill_add_mul_size(&policy.batch_floats, nt,
                              policy.q_buf_stride) ||
        !prefill_add_mul_size(&policy.batch_floats, nt,
                              policy.kv_dim * 2) ||
        !prefill_add_mul_size(&policy.batch_floats, nt,
                              policy.xb2_stride) ||
        !prefill_add_mul_size(&policy.batch_floats, nt,
                              policy.hb_stride) ||
        !prefill_add_mul_size(&policy.batch_floats, nt,
                              policy.hb2_stride))
        return 0;

    *out = policy;
    return 1;
}

int bn_transformer_prefill_uses_hybrid_layer_layout(const BnConfig *c) {
    return bn_transformer_uses_hybrid_layer_layout(c);
}

int bn_transformer_prefill_uses_hybrid_ssm(const BnConfig *c) {
    return bn_transformer_uses_hybrid_ssm(c);
}

int bn_transformer_prefill_uses_large_dense_hybrid_ssm(const BnConfig *c) {
    return bn_transformer_uses_large_dense_hybrid_ssm(c);
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
    return bn_transformer_gpu_small_dense_prefill_chain_applicable(
        gpu, c);
}

int bn_transformer_prefill_moe_enabled(void) {
    return bn_transformer_gpu_moe_prefill_enabled();
}

int bn_transformer_prefill_large_hybrid_disabled(void) {
    return bn_transformer_gpu_large_hybrid_prefill_disabled();
}

BnTransformerPrefillDecodeFallbackPolicy
bn_transformer_prefill_decode_fallback_policy(
    BnTransformerPrefillSequencePolicy sequence,
    int gpu_moe_prefill,
    int moe_prefill_enabled,
    int n_tokens,
    int moe_min_tokens,
    int small_dense_prefill_chain,
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
        (small_dense_prefill_chain &&
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
    return bn_transformer_gpu_prefill_hybrid_chain_enabled(gpu, c);
}

int bn_transformer_prefill_hybrid_chain_debug_enabled(void) {
    return bn_transformer_gpu_prefill_hybrid_chain_debug_enabled();
}

BnTransformerPrefillAttentionModePolicy
bn_transformer_prefill_attention_mode_policy(
    int tq_state_available,
    int requires_token_attention,
    int gpu_hybrid_prefill) {
    BnTransformerPrefillAttentionModePolicy policy = {0};
    policy.use_batched_attention =
        !tq_state_available &&
        !requires_token_attention &&
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
    return bn_transformer_gpu_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_prefill_dense_chain_enabled(void) {
    return bn_transformer_gpu_prefill_dense_chain_enabled();
}

int bn_transformer_prefill_dense_layer_gpu_available(
    const BnGPUBackend *gpu,
    int backend_available,
    int has_qkv,
    int has_wo,
    int has_gate,
    int has_up,
    int has_down) {
    return bn_transformer_gpu_prefill_dense_layer_backend_available(gpu) &&
           backend_available && has_qkv && has_wo && has_gate && has_up &&
           has_down;
}

int bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens) {
    return bn_transformer_gpu_dense_ffn_batch_tokens_allowed(gpu, c,
                                                             n_tokens);
}

int bn_transformer_prefill_dense_ffn_batch_gpu_available(
    const BnGPUBackend *gpu,
    int backend_available,
    int has_gate,
    int has_up,
    int has_down) {
    return bn_transformer_gpu_prefill_dense_ffn_batch_backend_available(gpu) &&
           backend_available && has_gate &&
           has_up && has_down;
}

int bn_transformer_prefill_dense_ffn_batch_norm_resid_gpu_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_prefill_dense_ffn_batch_norm_resid_backend_available(
        gpu);
}

int bn_transformer_prefill_attention_min_tokens(void) {
    return bn_transformer_gpu_prefill_attention_min_tokens();
}

int bn_transformer_prefill_attention_enabled(void) {
    return bn_transformer_gpu_prefill_attention_enabled();
}

int bn_transformer_prefill_raw_attention_gpu_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_prefill_qkv_attention_wo_backend_available(gpu);
}

int bn_transformer_prefill_raw_attention_norm_resid_gpu_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_prefill_qkv_attention_wo_norm_resid_backend_available(
        gpu);
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
    const BnTransformerSSMShapePolicy *ssm_shape) {
    BnTransformerPrefillSSMChainPolicy policy = {0};
    policy.enabled =
        chain_available &&
        !layer_kind.uses_moe &&
        has_ffn_gate &&
        has_ffn_up &&
        !has_ffn_sub_norm &&
        !has_layer_output_scale &&
        !(uses_post_norm && (has_attn_post_norm || has_ffn_post_norm)) &&
        ssm_shape &&
        ssm_shape->num_v_heads > 0 &&
        ssm_shape->head_k_dim > 0 &&
        ssm_shape->head_v_dim > 0 &&
        ssm_shape->num_k_heads > 0 &&
        ssm_shape->qkv_dim > 0 &&
        ssm_shape->conv_kernel > 1;
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
    const BnTransformerSSMShapePolicy *ssm_shape) {
    BnTransformerPrefillSSMMoEChainPolicy policy = {0};
    policy.enabled =
        chain_available &&
        layer_kind.uses_moe &&
        !has_ffn_sub_norm &&
        !has_layer_output_scale &&
        !(uses_post_norm && (has_attn_post_norm || has_ffn_post_norm)) &&
        ssm_shape &&
        ssm_shape->num_v_heads > 0 &&
        ssm_shape->head_k_dim > 0 &&
        ssm_shape->head_v_dim > 0 &&
        ssm_shape->num_k_heads > 0 &&
        ssm_shape->qkv_dim > 0 &&
        ssm_shape->conv_kernel > 1;
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
    return bn_transformer_gpu_prefill_ssm_dense_chain_available(
        gpu, c, n_tokens);
}

int bn_transformer_prefill_ssm_run_chain_enabled(void) {
    return bn_transformer_gpu_prefill_ssm_run_chain_enabled();
}

int bn_transformer_prefill_moe_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down) {
    return bn_transformer_gpu_prefill_moe_ffn_batch_available(
        gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_prefill_moe_layer_backend_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down) {
    return bn_transformer_gpu_prefill_moe_layer_backend_available(
        gpu, c, map, dim, allow_kquant_down);
}

int bn_transformer_prefill_ssm_moe_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int n_tokens) {
    return bn_transformer_gpu_prefill_ssm_moe_chain_available(
        gpu, c, map, dim, allow_kquant_down, n_tokens);
}

int bn_transformer_prefill_moe_layer_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_kquant_down,
    int n_tokens) {
    return bn_transformer_gpu_prefill_moe_layer_chain_available(
        gpu, c, map, dim, allow_kquant_down, n_tokens);
}

int bn_transformer_prefill_moe_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_prefill_moe_chain_min_tokens(c, gpu);
}

int bn_transformer_prefill_moe_chain_debug_enabled(void) {
    return bn_transformer_gpu_prefill_moe_chain_debug_enabled();
}

int bn_transformer_prefill_ssm_ffn_fuse_allowed(void) {
    return bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed();
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
                    bn_transformer_prefill_uses_hybrid_ssm(c) &&
                    bn_transformer_gpu_prefill_ssm_layer_disabled();
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
    return bn_transformer_gpu_prefill_direct_kv_allowed(
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

int bn_transformer_prefill_attention_gpu_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_prefill_attention_backend_available(gpu);
}

int bn_transformer_prefill_attention_wo_gpu_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_prefill_attention_wo_backend_available(gpu);
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

int bn_transformer_prefill_can_prepared_kquant_type(const BnPrefillCPUOps *ops,
                                           int tensor_type) {
    return ops && ops->supports_prepared_kquant &&
           bn_backend_quant_supports_prepared_kquant(tensor_type);
}

int bn_transformer_prefill_can_prepared_kquant_pair(const BnPrefillCPUOps *ops,
                                           int left_type,
                                           int right_type) {
    return bn_transformer_prefill_can_prepared_kquant_type(ops, left_type) &&
           bn_backend_quant_supports_prepared_kquant(right_type);
}

int bn_transformer_prefill_can_prepared_kquant_triple(const BnPrefillCPUOps *ops,
                                             int first_type,
                                             int second_type,
                                             int third_type) {
    return bn_transformer_prefill_can_prepared_kquant_pair(ops, first_type,
                                                  second_type) &&
           bn_backend_quant_supports_prepared_kquant(third_type);
}

int bn_transformer_prefill_prepared_kquant_blocks_per_row(int dim) {
    return bn_backend_quant_prepared_kquant_blocks_per_row(dim);
}

int bn_transformer_prefill_prepared_kquant_block_sums_per_row(
    int blocks_per_row) {
    return bn_backend_quant_prepared_kquant_block_sums_per_row(blocks_per_row);
}

int bn_transformer_prefill_route_prepared_kquant_type_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int uses_float_kquant_fallback,
    int dim,
    int tensor_type) {
    return !gpu &&
           !uses_float_kquant_fallback &&
           bn_transformer_prefill_prepared_kquant_blocks_per_row(dim) > 0 &&
           bn_transformer_prefill_can_prepared_kquant_type(ops, tensor_type);
}

BnTransformerPrefillPreparedKQuantDispatchPolicy
bn_transformer_prefill_prepared_kquant_dispatch_policy(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int uses_float_kquant_fallback,
    int dim,
    const int *tensor_types,
    int n_types,
    int max_types) {
    BnTransformerPrefillPreparedKQuantDispatchPolicy policy = {0};
    if (!tensor_types || n_types <= 0 || max_types <= 0 ||
        n_types > max_types)
        return policy;
    if (gpu || uses_float_kquant_fallback ||
        bn_transformer_prefill_prepared_kquant_blocks_per_row(dim) <= 0)
        return policy;
    for (int i = 0; i < n_types; i++) {
        if (!bn_transformer_prefill_can_prepared_kquant_type(
                ops, tensor_types[i]))
            return policy;
    }
    policy.enabled = 1;
    return policy;
}

int bn_transformer_prefill_route_prepared_kquant_pair_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int uses_float_kquant_fallback,
    int dim,
    int left_type,
    int right_type) {
    int tensor_types[2] = { left_type, right_type };
    return bn_transformer_prefill_prepared_kquant_dispatch_policy(
               ops, gpu, uses_float_kquant_fallback, dim, tensor_types, 2, 2)
        .enabled;
}

int bn_transformer_prefill_route_prepared_kquant_triple_enabled(
    const BnPrefillCPUOps *ops,
    const BnGPUBackend *gpu,
    int uses_float_kquant_fallback,
    int dim,
    int first_type,
    int second_type,
    int third_type) {
    int tensor_types[3] = { first_type, second_type, third_type };
    return bn_transformer_prefill_prepared_kquant_dispatch_policy(
               ops, gpu, uses_float_kquant_fallback, dim, tensor_types, 3, 3)
        .enabled;
}

int bn_transformer_prefill_same_quant_format_pair_stackable(int left_type,
                                                            int right_type) {
    return bn_backend_quant_same_quant_format_pair_stackable(left_type,
                                                            right_type);
}

int bn_transformer_prefill_resolve_attention_projection_types(
    BnTransformerPrefillAttentionProjectionTypes *out,
    const BnLayerWeights *lw) {
    if (!out || !lw)
        return 0;
    memset(out, 0, sizeof(*out));
    out->q_type = lw->attn.wq.type;
    out->q_rows = lw->attn.wq.rows;
    out->q_cols = lw->attn.wq.cols;
    out->k_type = lw->attn.wk.type;
    out->k_rows = lw->attn.wk.rows;
    out->k_cols = lw->attn.wk.cols;
    out->v_type = lw->attn.wv.type;
    out->v_rows = lw->attn.wv.rows;
    out->v_cols = lw->attn.wv.cols;
    out->out_type = lw->attn.wo.type;
    out->out_rows = lw->attn.wo.rows;
    out->out_cols = lw->attn.wo.cols;
    return 1;
}

int bn_transformer_prefill_resolve_ffn_projection_types(
    BnTransformerPrefillFFNProjectionTypes *out,
    const BnLayerWeights *lw) {
    if (!out || !lw)
        return 0;
    memset(out, 0, sizeof(*out));
    out->gate_type = lw->ffn.ffn_gate.type;
    out->gate_rows = lw->ffn.ffn_gate.rows;
    out->gate_cols = lw->ffn.ffn_gate.cols;
    out->up_type = lw->ffn.ffn_up.type;
    out->up_rows = lw->ffn.ffn_up.rows;
    out->up_cols = lw->ffn.ffn_up.cols;
    out->down_type = lw->ffn.ffn_down.type;
    out->down_rows = lw->ffn.ffn_down.rows;
    out->down_cols = lw->ffn.ffn_down.cols;
    return 1;
}

int bn_transformer_prefill_resolve_ssm_projection_types(
    BnTransformerPrefillSSMProjectionTypes *out,
    const BnLayerWeights *lw) {
    if (!out || !lw)
        return 0;
    memset(out, 0, sizeof(*out));
    out->qkv_type = lw->ssm.wqkv.type;
    out->qkv_rows = lw->ssm.wqkv.rows;
    out->qkv_cols = lw->ssm.wqkv.cols;
    out->z_type = lw->ssm.wz.type;
    out->z_rows = lw->ssm.wz.rows;
    out->z_cols = lw->ssm.wz.cols;
    out->alpha_type = lw->ssm.ssm_alpha.type;
    out->alpha_rows = lw->ssm.ssm_alpha.rows;
    out->alpha_cols = lw->ssm.ssm_alpha.cols;
    out->beta_type = lw->ssm.ssm_beta.type;
    out->beta_rows = lw->ssm.ssm_beta.rows;
    out->beta_cols = lw->ssm.ssm_beta.cols;
    out->out_type = lw->ssm.ssm_out.type;
    out->out_rows = lw->ssm.ssm_out.rows;
    out->out_cols = lw->ssm.ssm_out.cols;
    return 1;
}

int bn_transformer_prefill_activation_is_relu2(int activation) {
    return bn_model_activation_is_relu2(activation);
}

int bn_transformer_prefill_activation_is_gelu(int activation) {
    return bn_model_activation_is_gelu(activation);
}

int bn_transformer_prefill_activation_uses_silu_path(int activation) {
    return bn_model_activation_uses_silu_path(activation);
}

int bn_transformer_prefill_config_activation(const BnConfig *c) {
    return bn_transformer_config_activation(c);
}

int bn_transformer_prefill_has_ffn_gate(const BnConfig *c) {
    return bn_transformer_has_ffn_gate(c);
}

float bn_transformer_prefill_norm_epsilon(const BnConfig *c) {
    return bn_transformer_norm_epsilon(c);
}

BnTransformerPrefillActivationPolicy
bn_transformer_prefill_activation_policy(int activation,
                                         int uses_reference_activation) {
    BnTransformerPrefillActivationPolicy policy = {
        activation,
        uses_reference_activation
    };
    return policy;
}

int bn_transformer_prefill_qk_stack_compatible(const BnQWeight *q,
                                               const BnQWeight *k,
                                               int q_stride,
                                               int dim) {
    return q && k &&
           bn_transformer_prefill_same_quant_format_pair_stackable(q->type,
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
    return bn_backend_quant_requires_float_kquant_fallback(tensor_type);
}

uint32_t bn_transformer_prefill_float_kquant_fallback_task_flags(int enabled) {
    return enabled ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT : 0u;
}

int bn_transformer_prefill_quant_matmul_gpu_available(
    const BnGPUBackend *gpu,
    int has_output,
    int has_weight,
    int has_weight_buffer,
    int has_input) {
    return bn_transformer_gpu_prefill_quant_matmul_backend_available(gpu) &&
           has_output && has_weight &&
           has_weight_buffer && has_input;
}

int bn_transformer_prefill_quant_matmul_batch_gpu_available(
    const BnGPUBackend *gpu,
    int n_tasks,
    int has_outputs,
    int has_weights,
    int has_weight_buffers,
    int has_input) {
    return bn_transformer_gpu_prefill_quant_matmul_batch_backend_available(gpu) &&
           n_tasks > 1 && n_tasks <= 16 &&
           has_outputs && has_weights && has_weight_buffers && has_input;
}

int bn_transformer_prefill_quant_matmul_gpu_buffer_run(float *out,
                                                       const BnQWeight *W,
                                                       void *W_buf,
                                                       const float *X,
                                                       int n_tokens,
                                                       BnGPUBackend *gpu) {
    if (!bn_transformer_prefill_quant_matmul_gpu_available(
            gpu, out != NULL, W != NULL, W_buf != NULL, X != NULL))
        return -1;
    return bn_transformer_gpu_prefill_quant_matmul_backend_run(
        gpu, out, W_buf, X, W->rows, W->cols, n_tokens, W->type);
}

void bn_transformer_prefill_quant_matmul_gpu_buffer(float *out,
                                                    const BnQWeight *W,
                                                    void *W_buf,
                                                    const float *X,
                                                    int n_tokens,
                                                    int8_t *quantized_buf,
                                                    BnThreadPool *pool,
                                                    BnGPUBackend *gpu) {
    bn_backend_quant_matmul_gpu_buf(out, W, W_buf, X, n_tokens, quantized_buf,
                                    pool, gpu);
}

void bn_transformer_prefill_quant_matmul_batch_gpu_buffers(
    const BnMatvecTask *tasks,
    const void **buffers,
    int n_tasks,
    const float *X,
    int n_tokens,
    int cols,
    int8_t *quantized_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu) {
    bn_backend_quant_matmul_batch_gpu_buf(tasks, buffers, n_tasks, X,
                                          n_tokens, cols, quantized_buf, pool,
                                          gpu);
}
