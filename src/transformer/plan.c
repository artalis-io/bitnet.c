#include "transformer_plan_internal.h"
#include "gpu_backend.h"
#include "model_arch.h"
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
    return bn_model_arch_is_attention_layer(c, layer);
}

int bn_transformer_attn_index(const BnConfig *c, int layer) {
    return bn_model_arch_attention_layer_index(c, layer);
}

int bn_transformer_ssm_index(const BnConfig *c, int layer) {
    return bn_model_arch_ssm_layer_index(c, layer);
}

int bn_transformer_attention_layer_count(const BnConfig *c) {
    return bn_model_arch_attention_layer_count(c);
}

int bn_transformer_ssm_layer_count(const BnConfig *c) {
    return bn_model_arch_ssm_layer_count(c);
}

int bn_transformer_uses_hybrid_ssm(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_ssm(c);
}

int bn_transformer_uses_hybrid_moe(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_moe(c);
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
    p->head_size = lw->attn.head_size > 0 ? lw->attn.head_size : c->head_size;
    p->kv_dim = lw->attn.kv_dim > 0 ? lw->attn.kv_dim : c->kv_dim;
    p->n_kv_heads = lw->attn.n_kv_heads > 0 ? lw->attn.n_kv_heads : c->n_kv_heads;
    p->kv_mul = lw->attn.kv_mul > 0 ? lw->attn.kv_mul : c->kv_mul;
    p->q_dim = c->n_heads * p->head_size;
    p->q_gated = bn_transformer_attention_q_projection_is_gated(
        &lw->attn.wq, p->q_dim);
    p->q_wide = bn_transformer_attention_q_projection_is_wide(
        &lw->attn.wq, c->dim, p->q_dim);
    p->qk_stride = c->qk_norm_per_head ? p->head_size : 0;
    p->has_qk_norm = (lw->attn.q_norm || lw->attn.k_norm) ? 1 : 0;
    p->has_bias = (lw->attn.q_bias || lw->attn.k_bias || lw->attn.v_bias) ? 1 : 0;
    p->kv_mode = bn_transformer_kv_mode(c, tq_enabled);
    p->kind = p->is_attn
        ? (p->q_gated ? BN_LAYER_ATTN_GATED_Q
                      : (p->q_wide ? BN_LAYER_ATTN_WIDE_Q : BN_LAYER_ATTN_CLASSIC))
        : BN_LAYER_SSM;
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
    return bn_model_arch_cpu_force_float_kquant(c)
        ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT
        : 0u;
}

int bn_transformer_cpu_prefill_force_float_kquant_enabled(
    const BnConfig *c) {
    return bn_model_arch_cpu_force_float_kquant(c) &&
           bn_transformer_cpu_backend_supports_float_kquant_prefill();
}

int bn_transformer_cpu_prefill_decode_for_parity_enabled(
    const BnConfig *c,
    int gpu_attached) {
    return !gpu_attached &&
           bn_model_arch_cpu_prefill_uses_decode_for_parity(c);
}

int bn_transformer_rmsnorm_requires_reference_scalar_order(
    const BnConfig *c) {
    return c &&
           bn_model_arch_rmsnorm_mode(c) ==
               BN_MODEL_ARCH_RMSNORM_REFERENCE_SCALAR_ORDER;
}

float bn_transformer_attention_scale(
    const BnConfig *c,
    int head_size) {
    return bn_model_arch_attention_scale(c, head_size);
}

int bn_transformer_attention_value_shares_key(
    const BnConfig *c) {
    return bn_model_arch_attention_value_shares_key_config(c);
}

int bn_transformer_attention_uses_post_norm(
    const BnConfig *c) {
    return bn_model_arch_uses_attention_post_norm(c);
}

int bn_transformer_ffn_uses_post_norm(
    const BnConfig *c) {
    return bn_model_arch_uses_ffn_post_norm(c);
}

int bn_transformer_uses_layer_output_scale(
    const BnConfig *c) {
    return bn_model_arch_uses_layer_output_scale(c);
}

int bn_transformer_per_layer_embedding_dim(
    const BnConfig *c) {
    return bn_model_arch_per_layer_embedding_dim(c);
}

int bn_transformer_cpu_uses_scalar_hybrid_ssm(
    const BnConfig *c) {
    return bn_model_arch_uses_scalar_hybrid_ssm_cpu(c);
}

int bn_transformer_prefill_uses_exact_activation(
    const BnConfig *c) {
    return bn_model_arch_prefill_uses_exact_activation(c);
}

int bn_transformer_ffn_uses_exact_scalar_activation(
    const BnConfig *c) {
    return bn_model_arch_ffn_uses_exact_scalar_activation(c);
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
        p->needs_cpu_fallback = p->placement == BN_EXEC_GPU;
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

    p->use_flash = c->flash_attn && bn_transformer_gpu_can_flash_attn(gpu);
    p->use_packed_qkv = qkv_stacked && !p->shape.q_gated &&
                        bn_transformer_gpu_can_native_qkv(
                            lw->attn.wq.type, lw->attn.wk.type,
                            lw->attn.wv.type) &&
                        q_bias && k_bias && v_bias;
    p->use_qkv_split = qkv_stacked && !p->shape.q_gated &&
                       bn_transformer_gpu_can_matvec_split(gpu, lw->attn.wq.type);
    if (p->use_qkv_split) p->fusion_flags |= BN_FUSION_QKV_SPLIT;
    if (p->use_flash) p->fusion_flags |= BN_FUSION_FLASH_ATTN;
    if (p->placement == BN_EXEC_GPU && !k_bias)
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
    p->kind = lw->ffn_kind == BN_LAYER_FFN_MOE ? BN_FFN_MOE
            : (c->has_ffn_gate ? BN_FFN_DENSE_GATE_UP : BN_FFN_DENSE_UP);
    p->hidden_dim = lw->ffn.ffn_up.rows > 0 ? lw->ffn.ffn_up.rows : c->hidden_dim;
    p->activation = c->act_type;
    p->has_gate = c->has_ffn_gate;
    p->has_sub_norm = lw->norm.ffn_sub_norm ? 1 : 0;
    p->scalar_exact_activation =
        bn_transformer_ffn_uses_exact_scalar_activation(c);

    void *gateup_stacked = bn_transformer_backend_handle_or(backend, layer,
                                                            BN_BACKEND_HANDLE_GATEUP_STACKED);

    p->use_fused_gateup_silu =
        p->placement == BN_EXEC_GPU &&
        c->has_ffn_gate &&
        bn_transformer_gpu_can_fused_gateup_silu_pair(
            gpu, lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type, c->act_type);
    p->use_gateup_split =
        p->placement == BN_EXEC_GPU &&
        c->has_ffn_gate &&
        gateup_stacked &&
        bn_transformer_gpu_can_use_stacked_gateup(&lw->ffn.ffn_gate,
                                                  &lw->ffn.ffn_up) &&
        bn_transformer_gpu_can_gateup_split_activation(
            gpu, lw->ffn.ffn_gate.type, c->act_type);
    if (p->use_fused_gateup_silu) p->fusion_flags |= BN_FUSION_GATEUP_SILU;
    if (p->use_gateup_split) p->fusion_flags |= BN_FUSION_GATEUP_SPLIT;
    if (p->placement == BN_EXEC_GPU) p->fusion_flags |= BN_FUSION_RESIDUAL_RMSNORM;
    if (p->kind == BN_FFN_MOE && p->placement == BN_EXEC_GPU) {
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
    p->use_qkvz_stack = p->placement == BN_EXEC_GPU &&
        bn_transformer_backend_handle_or(backend, layer, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED);
    p->use_alpha_beta_stack = p->placement == BN_EXEC_GPU &&
        bn_transformer_backend_handle_or(backend, layer, BN_BACKEND_HANDLE_SSM_AB_STACKED);
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
    p->n_experts = c->n_experts;
    p->n_active = c->n_experts_active;
    p->hidden_dim = c->moe_intermediate_size;
    p->has_shared_expert = c->has_shared_expert || lw->shared.shared_expert_gate;
    p->shared_hidden_dim = c->shared_expert_intermediate_size;
    if (p->placement == BN_EXEC_GPU && lw->moe.router_weight) {
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
    p->use_i8_output = w->emb_out_i8 != NULL;
    if (w->output_weight.data) {
        p->weight_type = w->output_weight.type;
        p->kind = bn_transformer_logits_untied_uses_f16_path(
                      w->output_weight.type)
            ? BN_LOGITS_UNTIED_F16
            : BN_LOGITS_UNTIED_QUANT;
    } else if (w->emb_out_i8) {
        p->kind = BN_LOGITS_TIED_I8;
        p->weight_type = bn_transformer_logits_tied_i8_weight_type();
    } else if (bn_transformer_logits_tied_uses_quant_path(w->emb_type)) {
        p->kind = BN_LOGITS_TIED_QUANT;
        p->weight_type = w->emb_type;
    } else if (bn_transformer_logits_tied_uses_f16_path(w->emb_type)) {
        p->kind = BN_LOGITS_TIED_F16;
        p->weight_type = bn_transformer_logits_tied_f16_weight_type();
    } else {
        p->kind = BN_LOGITS_TIED_F32;
        p->weight_type = bn_transformer_logits_tied_f32_weight_type();
    }
}
