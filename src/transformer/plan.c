#include "transformer_plan_internal.h"
#include "backend_quant.h"
#include "gpu_backend.h"
#include "transformer_backend_internal.h"
#include <stdlib.h>
#include <string.h>

int bn_transformer_gpu_has_cap(const BnGPUBackend *gpu, uint32_t cap) {
    return gpu && ((gpu->caps & cap) != 0);
}

int bn_transformer_gpu_can_matvec_split(const BnGPUBackend *gpu, int tensor_type) {
    uint32_t cap = bn_backend_quant_gpu_split_cap(tensor_type);
    return cap != 0 && bn_transformer_gpu_has_cap(gpu, cap);
}

int bn_transformer_gpu_can_fused_gateup_silu(const BnGPUBackend *gpu,
                                             int tensor_type,
                                             int act_type) {
    if (getenv("BN_GPU_DISABLE_FUSED_GATEUP"))
        return 0;
    uint32_t cap = bn_backend_quant_gpu_fused_gateup_silu_cap(tensor_type);
    return cap != 0 && act_type != 1 && bn_transformer_gpu_has_cap(gpu, cap);
}

int bn_transformer_gpu_can_flash_attn(const BnGPUBackend *gpu) {
    return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_FLASH_ATTN);
}

void *bn_transformer_backend_handle_or(const BnBackendModel *backend,
                                       int layer,
                                       BnBackendHandleRole role) {
    return bn_backend_model_handle(backend, layer, role);
}

int bn_transformer_is_attn_layer(const BnConfig *c, int layer) {
    return c->full_attn_interval == 0 ||
           ((layer + 1) % c->full_attn_interval == 0);
}

int bn_transformer_attn_index(const BnConfig *c, int layer) {
    return c->full_attn_interval > 0
        ? (layer + 1) / c->full_attn_interval - 1
        : layer;
}

int bn_transformer_ssm_index(const BnConfig *c, int layer) {
    return c->full_attn_interval > 0
        ? layer - (layer + 1) / c->full_attn_interval
        : -1;
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
    p->q_gated = lw->attn.wq.data && lw->attn.wq.rows > p->q_dim;
    p->q_wide = !p->q_gated && lw->attn.wq.data && lw->attn.wq.rows > c->dim;
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
    if (!gpu) return BN_BACKEND_GPU_UNKNOWN;
    switch (gpu->kind) {
        case BN_GPU_BACKEND_METAL: return BN_BACKEND_METAL;
        case BN_GPU_BACKEND_WEBGPU: return BN_BACKEND_WEBGPU;
        case BN_GPU_BACKEND_CUDA: return BN_BACKEND_CUDA;
        default: return BN_BACKEND_GPU_UNKNOWN;
    }
}

BnCPUBackendPlacement bn_transformer_cpu_backend_placement(void) {
#ifdef BN_FORCE_SCALAR
    return BN_CPU_BACKEND_SCALAR;
#elif defined(__AVX512F__)
    return BN_CPU_BACKEND_AVX512;
#elif defined(__AVX2__)
    return BN_CPU_BACKEND_AVX2;
#elif defined(__ARM_NEON)
    return BN_CPU_BACKEND_NEON;
#elif defined(__wasm_relaxed_simd__) || defined(__wasm_simd128__)
    return BN_CPU_BACKEND_WASM_SIMD;
#else
    return BN_CPU_BACKEND_SCALAR;
#endif
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
                        bn_backend_quant_can_gpu_native(lw->attn.wq.type) &&
                        bn_backend_quant_can_gpu_native(lw->attn.wk.type) &&
                        bn_backend_quant_can_gpu_native(lw->attn.wv.type) &&
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

    void *gateup_stacked = bn_transformer_backend_handle_or(backend, layer,
                                                            BN_BACKEND_HANDLE_GATEUP_STACKED);

    p->use_fused_gateup_silu =
        p->placement == BN_EXEC_GPU &&
        c->has_ffn_gate &&
        bn_backend_quant_gpu_fused_gateup_silu_cap(lw->ffn.ffn_gate.type) != 0 &&
        bn_backend_quant_gpu_fused_gateup_silu_cap(lw->ffn.ffn_up.type) ==
            bn_backend_quant_gpu_fused_gateup_silu_cap(lw->ffn.ffn_gate.type) &&
        bn_transformer_gpu_can_fused_gateup_silu(gpu, lw->ffn.ffn_gate.type, c->act_type);
    p->use_gateup_split =
        p->placement == BN_EXEC_GPU &&
        c->has_ffn_gate &&
        gateup_stacked &&
        lw->ffn.ffn_gate.rows == lw->ffn.ffn_up.rows &&
        lw->ffn.ffn_gate.cols == lw->ffn.ffn_up.cols &&
        bn_transformer_gpu_can_matvec_split(gpu, lw->ffn.ffn_gate.type) &&
        bn_backend_quant_can_gpu_gateup_split_activation(lw->ffn.ffn_gate.type,
                                                         c->act_type);
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
        p->kind = BN_LOGITS_UNTIED;
        p->weight_type = w->output_weight.type;
    } else if (w->emb_out_i8) {
        p->kind = BN_LOGITS_TIED_I8;
        p->weight_type = BN_GGUF_TENSOR_Q8_0;
    } else if (w->emb_type == BN_GGUF_TENSOR_F16) {
        p->kind = BN_LOGITS_TIED_F16;
        p->weight_type = BN_GGUF_TENSOR_F16;
    } else {
        p->kind = BN_LOGITS_TIED_F32;
        p->weight_type = BN_GGUF_TENSOR_F32;
    }
}
