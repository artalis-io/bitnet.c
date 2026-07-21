#include "transformer_plan_internal.h"
#include "gpu_internal.h"
#include "model_internal.h"
#include "../moe_internal.h"
#include "transformer_backend_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_logits_internal.h"
#include <stdlib.h>
#include <string.h>

void *bn_transformer_backend_handle_or(const BnBackendModel *backend,
                                       int layer,
                                       BnBackendHandleRole role) {
    return bn_backend_model_handle(backend, layer, role);
}

int bn_transformer_is_attn_layer(const BnConfig *c, int layer) {
    return bn_model_config_is_attention_layer(c, layer);
}

int bn_transformer_attn_index(const BnConfig *c, int layer) {
    return bn_model_config_attention_layer_index(c, layer);
}

int bn_transformer_ssm_index(const BnConfig *c, int layer) {
    return bn_model_config_ssm_layer_index(c, layer);
}

int bn_transformer_attention_layer_count(const BnConfig *c) {
    return bn_model_config_attention_layer_count(c);
}

int bn_transformer_ssm_layer_count(const BnConfig *c) {
    return bn_model_config_ssm_layer_count(c);
}

int bn_transformer_uses_hybrid_ssm(const BnConfig *c) {
    return bn_model_config_uses_hybrid_ssm(c);
}

int bn_transformer_uses_hybrid_moe(const BnConfig *c) {
    return bn_model_config_uses_hybrid_moe(c);
}

int bn_transformer_weight_is_packed_qkv(const BnQWeight *qkv,
                                        int input_dim,
                                        int q_dim,
                                        int kv_dim) {
    return qkv && qkv->data &&
           qkv->cols == input_dim &&
           qkv->rows == q_dim + 2 * kv_dim;
}

int bn_transformer_attention_q_projection_is_gated(const BnQWeight *wq,
                                                   int q_dim) {
    return wq && wq->data && wq->rows > q_dim;
}

int bn_transformer_attention_q_projection_is_wide(const BnQWeight *wq,
                                                  int model_dim,
                                                  int q_dim) {
    return wq && wq->data &&
           wq->rows <= q_dim &&
           wq->rows != model_dim;
}

int bn_transformer_attention_head_size(const BnConfig *c,
                                       const BnLayerWeights *lw) {
    return lw && lw->attn.head_size > 0 ? lw->attn.head_size
                                        : c->head_size;
}

int bn_transformer_attention_kv_dim(const BnConfig *c,
                                    const BnLayerWeights *lw) {
    return lw && lw->attn.kv_dim > 0 ? lw->attn.kv_dim
                                     : c->kv_dim;
}

int bn_transformer_attention_n_kv_heads(const BnConfig *c,
                                        const BnLayerWeights *lw) {
    return lw && lw->attn.n_kv_heads > 0 ? lw->attn.n_kv_heads
                                         : c->n_kv_heads;
}

int bn_transformer_attention_kv_mul(const BnConfig *c,
                                    const BnLayerWeights *lw) {
    return lw && lw->attn.kv_mul > 0 ? lw->attn.kv_mul
                                     : c->kv_mul;
}

int bn_transformer_attention_qk_stride(const BnConfig *c,
                                       int head_size) {
    return c && c->qk_norm_per_head ? head_size : 0;
}

int bn_transformer_attention_has_qk_norm(const BnLayerWeights *lw) {
    return lw && (lw->attn.q_norm || lw->attn.k_norm);
}

int bn_transformer_attention_has_bias(const BnLayerWeights *lw) {
    return lw && (lw->attn.q_bias || lw->attn.k_bias || lw->attn.v_bias);
}

BnLayerKind bn_transformer_layer_kind(int is_attn,
                                      int q_gated,
                                      int q_wide) {
    if (!is_attn) return BN_LAYER_SSM;
    if (q_gated) return BN_LAYER_ATTN_GATED_Q;
    if (q_wide) return BN_LAYER_ATTN_WIDE_Q;
    return BN_LAYER_ATTN_CLASSIC;
}

int bn_transformer_attention_requires_cpu_fallback(
    const BnLayerShapePlan *shape,
    BnExecPlacement placement) {
    return shape && !shape->is_attn && placement == BN_EXEC_GPU;
}

int bn_transformer_attention_uses_flash(const BnConfig *c,
                                        const BnGPUBackend *gpu) {
    return c && c->flash_attn && bn_transformer_gpu_can_flash_attn(gpu);
}

int bn_transformer_attention_uses_packed_qkv(
    const BnGPUBackend *gpu,
    const BnLayerShapePlan *shape,
    const BnLayerWeights *lw,
    const void *qkv_stacked,
    const void *q_bias,
    const void *k_bias,
    const void *v_bias) {
    (void)gpu;
    return qkv_stacked &&
           shape && !shape->q_gated &&
           lw &&
           bn_transformer_gpu_can_native_quant_qkv(
               lw->attn.wq.type, lw->attn.wk.type, lw->attn.wv.type) &&
           q_bias && k_bias && v_bias;
}

int bn_transformer_attention_uses_qkv_split(
    const BnGPUBackend *gpu,
    const BnLayerShapePlan *shape,
    const BnLayerWeights *lw,
    const void *qkv_stacked) {
    return qkv_stacked &&
           shape && !shape->q_gated &&
           lw &&
           bn_transformer_gpu_can_matvec_split(gpu, lw->attn.wq.type);
}

int bn_transformer_attention_uses_rope_qk_fusion(
    BnExecPlacement placement,
    const void *k_bias) {
    return placement == BN_EXEC_GPU && !k_bias;
}

BnKVMode bn_transformer_kv_mode(const BnConfig *c, int tq_enabled) {
    if (c->kv_tq_bits > 0 && tq_enabled) return BN_KV_TQ;
    if (c->kv_f16) return BN_KV_FP16;
    return BN_KV_FP32;
}

void bn_transformer_plan_layer_shape(BnLayerShapePlan *p,
                                     const BnConfig *c,
                                     const BnLayerWeights *lw,
                                     int layer,
                                     int tq_enabled) {
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->is_attn = lw->block_kind == BN_LAYER_BLOCK_ATTENTION;
    p->attn_idx = p->is_attn ? bn_transformer_attn_index(c, layer) : -1;
    p->ssm_idx = p->is_attn ? -1 : bn_transformer_ssm_index(c, layer);
    p->head_size = bn_transformer_attention_head_size(c, lw);
    p->kv_dim = bn_transformer_attention_kv_dim(c, lw);
    p->n_kv_heads = bn_transformer_attention_n_kv_heads(c, lw);
    p->kv_mul = bn_transformer_attention_kv_mul(c, lw);
    p->q_dim = c->n_heads * p->head_size;
    p->q_gated = bn_transformer_attention_q_projection_is_gated(
        &lw->attn.wq, p->q_dim);
    p->q_wide = bn_transformer_attention_q_projection_is_wide(
        &lw->attn.wq, c->dim, p->q_dim);
    p->qk_stride = bn_transformer_attention_qk_stride(c, p->head_size);
    p->has_qk_norm = bn_transformer_attention_has_qk_norm(lw);
    p->has_bias = bn_transformer_attention_has_bias(lw);
    p->kv_mode = bn_transformer_kv_mode(c, tq_enabled);
    p->kind = bn_transformer_layer_kind(p->is_attn, p->q_gated, p->q_wide);
}

BnExecPlacement bn_transformer_preferred_placement(const BnGPUBackend *gpu,
                                                   int prefer_gpu) {
    return prefer_gpu && gpu ? BN_EXEC_GPU : BN_EXEC_CPU;
}

BnBackendPlacement bn_transformer_backend_placement(const BnGPUBackend *gpu,
                                                    BnExecPlacement placement) {
    if (placement == BN_EXEC_CPU) return BN_BACKEND_CPU;
    if (placement == BN_EXEC_CPU_FALLBACK) return BN_BACKEND_CPU;
    return bn_transformer_gpu_backend_placement(gpu);
}

uint32_t bn_transformer_cpu_force_float_kquant_task_flags(
    const BnConfig *c) {
    return bn_transformer_cpu_float_kquant_task_flags(
        bn_model_config_requires_float_kquant_fallback(c));
}

int bn_transformer_cpu_prefill_force_float_kquant_enabled(
    const BnConfig *c) {
    return bn_model_config_requires_float_kquant_fallback(c) &&
           bn_transformer_cpu_backend_supports_float_kquant_prefill();
}

int bn_transformer_cpu_prefill_decode_for_parity_enabled(
    const BnConfig *c,
    int gpu_attached) {
    return !gpu_attached &&
           bn_model_config_prefill_uses_decode_for_parity(c);
}

int bn_transformer_rmsnorm_uses_reference_order(
    const BnConfig *c) {
    return bn_model_config_rmsnorm_uses_reference_order(c);
}

int bn_transformer_rmsnorm_requires_reference_scalar_order(
    const BnConfig *c) {
    return bn_transformer_rmsnorm_uses_reference_order(c);
}

float bn_transformer_attention_scale(
    const BnConfig *c,
    int head_size) {
    return bn_model_config_attention_scale(c, head_size);
}

int bn_transformer_attention_value_shares_key(
    const BnConfig *c) {
    return bn_model_config_attention_value_shares_key(c);
}

int bn_transformer_attention_uses_post_norm(
    const BnConfig *c) {
    return bn_model_config_uses_attention_post_norm(c);
}

int bn_transformer_ffn_uses_post_norm(
    const BnConfig *c) {
    return bn_model_config_uses_ffn_post_norm(c);
}

int bn_transformer_uses_layer_output_scale(
    const BnConfig *c) {
    return bn_model_config_uses_layer_output_scale(c);
}

BnFFNKind bn_transformer_ffn_kind(const BnConfig *c,
                                  const BnLayerWeights *lw) {
    if (lw && lw->ffn_kind == BN_LAYER_FFN_MOE)
        return BN_FFN_MOE;
    return c && c->has_ffn_gate ? BN_FFN_DENSE_GATE_UP
                                : BN_FFN_DENSE_UP;
}

int bn_transformer_ffn_hidden_dim(const BnConfig *c,
                                  const BnLayerWeights *lw) {
    return lw && lw->ffn.ffn_up.rows > 0 ? lw->ffn.ffn_up.rows
                                         : c->hidden_dim;
}

int bn_transformer_ffn_has_gate(const BnConfig *c) {
    return c && c->has_ffn_gate;
}

int bn_transformer_ffn_has_sub_norm(const BnLayerWeights *lw) {
    return lw && lw->norm.ffn_sub_norm;
}

int bn_transformer_ffn_uses_fused_gateup_silu(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    BnExecPlacement placement) {
    return placement == BN_EXEC_GPU &&
           bn_transformer_ffn_has_gate(c) &&
           lw &&
           bn_transformer_gpu_can_fused_gateup_silu_pair(
               gpu, lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type, c->act_type);
}

int bn_transformer_ffn_uses_gateup_split(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    BnExecPlacement placement,
    const void *gateup_stacked) {
    return placement == BN_EXEC_GPU &&
           bn_transformer_ffn_has_gate(c) &&
           gateup_stacked &&
           lw &&
           bn_transformer_gpu_can_stack_same_quant_format_gateup(&lw->ffn.ffn_gate,
                                                     &lw->ffn.ffn_up) &&
           bn_transformer_gpu_can_gateup_split_activation(
               gpu, lw->ffn.ffn_gate.type, c->act_type);
}

int bn_transformer_ffn_uses_residual_rmsnorm_fusion(
    BnExecPlacement placement) {
    return placement == BN_EXEC_GPU;
}

int bn_transformer_ffn_requires_cpu_fallback(
    BnFFNKind kind,
    BnExecPlacement placement) {
    return kind == BN_FFN_MOE && placement == BN_EXEC_GPU;
}

int bn_transformer_moe_has_shared_expert(const BnConfig *c,
                                         const BnLayerWeights *lw) {
    return bn_moe_policy_has_shared_expert(c, lw);
}

int bn_transformer_moe_layer_has_router(const BnLayerWeights *lw) {
    return bn_moe_policy_layer_has_router(lw);
}

int bn_transformer_moe_requires_cpu_fallback(BnExecPlacement placement,
                                             const BnLayerWeights *lw) {
    BnTransformerGPULayerKindPolicy layer_kind =
        bn_transformer_gpu_layer_kind_policy(lw);
    return placement == BN_EXEC_GPU && layer_kind.uses_moe;
}

int bn_transformer_ssm_uses_qkvz_stack(
    BnExecPlacement placement,
    const void *qkvz_stacked) {
    return placement == BN_EXEC_GPU && qkvz_stacked;
}

int bn_transformer_ssm_uses_alpha_beta_stack(
    BnExecPlacement placement,
    const void *alpha_beta_stacked) {
    return placement == BN_EXEC_GPU && alpha_beta_stacked;
}

int bn_transformer_logits_uses_i8_output(const BnWeights *w) {
    return w && w->emb_out_i8 != NULL;
}

int bn_transformer_logits_has_untied_output(const BnWeights *w) {
    return w && w->output_weight.data;
}

BnLogitsKind bn_transformer_logits_kind(const BnWeights *w) {
    if (bn_transformer_logits_has_untied_output(w)) {
        return bn_transformer_logits_untied_uses_f16_path(
                   w->output_weight.type)
            ? BN_LOGITS_UNTIED_F16
            : BN_LOGITS_UNTIED_QUANT;
    }
    if (bn_transformer_logits_uses_i8_output(w))
        return BN_LOGITS_TIED_I8;
    if (w && bn_transformer_logits_tied_uses_quant_path(w->emb_type))
        return BN_LOGITS_TIED_QUANT;
    if (w && bn_transformer_logits_tied_uses_f16_path(w->emb_type))
        return BN_LOGITS_TIED_F16;
    return BN_LOGITS_TIED_F32;
}

int bn_transformer_logits_weight_type(const BnWeights *w) {
    if (bn_transformer_logits_has_untied_output(w))
        return w->output_weight.type;
    if (bn_transformer_logits_uses_i8_output(w))
        return bn_transformer_logits_tied_i8_weight_type();
    if (w && bn_transformer_logits_tied_uses_quant_path(w->emb_type))
        return w->emb_type;
    if (w && bn_transformer_logits_tied_uses_f16_path(w->emb_type))
        return bn_transformer_logits_tied_f16_weight_type();
    return bn_transformer_logits_tied_f32_weight_type();
}

int bn_transformer_per_layer_embedding_dim(
    const BnConfig *c) {
    return bn_model_config_per_layer_embedding_dim(c);
}

int bn_transformer_uses_per_layer_embedding(
    const BnConfig *c) {
    return bn_model_config_uses_per_layer_embedding(c);
}

int bn_transformer_divides_rope_freqs(
    const BnConfig *c,
    int layer) {
    return bn_model_config_divides_rope_freqs(c, layer);
}

int bn_transformer_cpu_uses_scalar_hybrid_ssm(
    const BnConfig *c) {
    return bn_model_config_uses_reference_hybrid_ssm(c);
}

int bn_transformer_prefill_uses_exact_activation(
    const BnConfig *c) {
    return bn_model_config_prefill_uses_exact_activation(c);
}

int bn_transformer_ffn_uses_exact_scalar_activation(
    const BnConfig *c) {
    return bn_model_config_ffn_uses_reference_activation(c);
}

void bn_transformer_plan_attention(BnAttentionPlan *p,
                                   const BnConfig *c,
                                   const BnLayerWeights *lw,
                                   const BnGPUBackend *gpu,
                                   const BnBackendModel *backend,
                                   int layer,
                                   int tq_enabled,
                                   int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    bn_transformer_plan_layer_shape(&p->shape, c, lw, layer, tq_enabled);
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    if (!p->shape.is_attn) {
        p->needs_cpu_fallback =
            bn_transformer_attention_requires_cpu_fallback(
                &p->shape, p->placement);
        if (p->needs_cpu_fallback) {
            p->placement = BN_EXEC_CPU_FALLBACK;
            p->backend = bn_transformer_backend_placement(gpu, p->placement);
        }
        return;
    }

    void *qkv_stacked = bn_transformer_backend_handle_or(backend, layer,
                                                         BN_BACKEND_HANDLE_QKV_STACKED);
    void *q_bias = bn_transformer_backend_handle_or(backend, layer,
                                                    BN_BACKEND_HANDLE_Q_BIAS);
    void *k_bias = bn_transformer_backend_handle_or(backend, layer,
                                                    BN_BACKEND_HANDLE_K_BIAS);
    void *v_bias = bn_transformer_backend_handle_or(backend, layer,
                                                    BN_BACKEND_HANDLE_V_BIAS);

    p->use_flash = bn_transformer_attention_uses_flash(c, gpu);
    p->use_packed_qkv = bn_transformer_attention_uses_packed_qkv(
        gpu, &p->shape, lw, qkv_stacked, q_bias, k_bias, v_bias);
    p->use_qkv_split = bn_transformer_attention_uses_qkv_split(
        gpu, &p->shape, lw, qkv_stacked);
    if (p->use_qkv_split) p->fusion_flags |= BN_FUSION_QKV_SPLIT;
    if (p->use_flash) p->fusion_flags |= BN_FUSION_FLASH_ATTN;
    if (bn_transformer_attention_uses_rope_qk_fusion(p->placement, k_bias))
        p->fusion_flags |= BN_FUSION_ROPE_QK;
}

void bn_transformer_plan_ffn(BnFFNPlan *p,
                             const BnConfig *c,
                             const BnLayerWeights *lw,
                             const BnGPUBackend *gpu,
                             const BnBackendModel *backend,
                             int layer,
                             int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    p->kind = bn_transformer_ffn_kind(c, lw);
    p->hidden_dim = bn_transformer_ffn_hidden_dim(c, lw);
    p->activation = c->act_type;
    p->has_gate = bn_transformer_ffn_has_gate(c);
    p->has_sub_norm = bn_transformer_ffn_has_sub_norm(lw);
    p->scalar_exact_activation =
        bn_transformer_ffn_uses_exact_scalar_activation(c);

    void *gateup_stacked = bn_transformer_backend_handle_or(backend, layer,
                                                            BN_BACKEND_HANDLE_GATEUP_STACKED);

    p->use_fused_gateup_silu = bn_transformer_ffn_uses_fused_gateup_silu(
        gpu, c, lw, p->placement);
    p->use_gateup_split = bn_transformer_ffn_uses_gateup_split(
        gpu, c, lw, p->placement, gateup_stacked);
    if (p->use_fused_gateup_silu) p->fusion_flags |= BN_FUSION_GATEUP_SILU;
    if (p->use_gateup_split) p->fusion_flags |= BN_FUSION_GATEUP_SPLIT;
    if (bn_transformer_ffn_uses_residual_rmsnorm_fusion(p->placement))
        p->fusion_flags |= BN_FUSION_RESIDUAL_RMSNORM;
    if (bn_transformer_ffn_requires_cpu_fallback(p->kind, p->placement)) {
        p->needs_cpu_fallback = 1;
        p->placement = BN_EXEC_CPU_FALLBACK;
        p->backend = bn_transformer_backend_placement(gpu, p->placement);
        p->fusion_flags = BN_FUSION_NONE;
    }
}

void bn_transformer_plan_ssm(BnSSMPlan *p,
                             const BnConfig *c,
                             const BnLayerWeights *lw,
                             int layer,
                             int prefer_gpu,
                             const BnGPUBackend *gpu,
                             const BnBackendModel *backend) {
    (void)lw;
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->ssm_idx = bn_transformer_ssm_index(c, layer);
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    p->state_size = c->ssm_state_size;
    p->conv_kernel = c->ssm_conv_kernel;
    p->inner_size = c->ssm_inner_size;
    p->time_step_rank = c->ssm_time_step_rank;
    p->group_count = c->ssm_group_count;
    void *qkvz_stacked = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED);
    void *alpha_beta_stacked = bn_transformer_backend_handle_or(
        backend, layer, BN_BACKEND_HANDLE_SSM_AB_STACKED);
    p->use_qkvz_stack = bn_transformer_ssm_uses_qkvz_stack(
        p->placement, qkvz_stacked);
    p->use_alpha_beta_stack = bn_transformer_ssm_uses_alpha_beta_stack(
        p->placement, alpha_beta_stacked);
}

void bn_transformer_plan_moe(BnMoEPlan *p,
                             const BnConfig *c,
                             const BnLayerWeights *lw,
                             const BnGPUBackend *gpu,
                             int layer,
                             int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    p->layer = layer;
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    BnMoERoutePolicy route_policy = bn_moe_route_policy(c);
    p->n_experts = route_policy.total_experts;
    p->n_active = route_policy.active_experts;
    p->hidden_dim = route_policy.expert_hidden_dim;
    p->has_shared_expert = bn_transformer_moe_has_shared_expert(c, lw);
    p->shared_hidden_dim = bn_moe_policy_shared_expert_hidden_dim(c);
    if (bn_transformer_moe_requires_cpu_fallback(p->placement, lw)) {
        p->needs_cpu_fallback = 1;
        p->placement = BN_EXEC_CPU_FALLBACK;
        p->backend = bn_transformer_backend_placement(gpu, p->placement);
    }
}

void bn_transformer_plan_logits(BnLogitsPlan *p,
                                const BnConfig *c,
                                const BnWeights *w,
                                const BnGPUBackend *gpu,
                                int prefer_gpu) {
    memset(p, 0, sizeof(*p));
    p->placement = bn_transformer_preferred_placement(gpu, prefer_gpu);
    p->backend = bn_transformer_backend_placement(gpu, p->placement);
    p->vocab_size = c->vocab_size;
    p->dim = c->dim;
    p->use_i8_output = bn_transformer_logits_uses_i8_output(w);
    p->kind = bn_transformer_logits_kind(w);
    p->weight_type = bn_transformer_logits_weight_type(w);
}
