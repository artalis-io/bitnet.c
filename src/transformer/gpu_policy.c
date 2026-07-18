#include "gpu_internal.h"
#include "gpu_policy.h"
#include "../gpu_shader_ir_internal.h"
#include "../gpu_quant_lowering_internal.h"
#include "backend_quant.h"
#include "model_arch.h"
#include <stdio.h>
#include <stdlib.h>

static int gpu_env_enabled(const char *name, const char *compat_name) {
    return getenv(name) != NULL ||
           (compat_name && getenv(compat_name) != NULL);
}

static const char *gpu_env_value(const char *name, const char *compat_name) {
    const char *env = getenv(name);
    return env ? env : (compat_name ? getenv(compat_name) : NULL);
}

int bn_transformer_gpu_graph_op_capacity(const BnConfig *c) {
    /* Max ops per batch. MoE/SSM flush between layers, so single-layer max
     * suffices. Approximate flush batch budget:
     * - Attention: ~20 (QKV + norms + RoPE + GQA + sigmoid + Wo + resid)
     * - SSM: ~16 (QKV + Z + conv + splits + L2norm + alpha/beta + delta + gate + out + resid)
     * - MoE: K*5 + shared(5) + residual + rmsnorm = up to BN_MAX_MOE_K*5 + 7
     */
    return 80 * c->n_layers + 5 * BN_MAX_MOE_K + 100;
}

int bn_transformer_gpu_backend_is_cuda(const BnGPUBackend *gpu) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA;
}

int bn_transformer_gpu_has_cap(const BnGPUBackend *gpu, uint32_t cap) {
    return gpu && ((gpu->caps & cap) != 0);
}

int bn_transformer_gpu_can_native_qkv(int q_type, int k_type, int v_type) {
    return bn_backend_quant_can_gpu_native(q_type) &&
           bn_backend_quant_can_gpu_native(k_type) &&
           bn_backend_quant_can_gpu_native(v_type);
}

int bn_transformer_gpu_can_use_stacked_qk(int q_type, int k_type) {
    return bn_backend_quant_stacked_pair_same_format(q_type, k_type);
}

int bn_transformer_gpu_can_use_stacked_qk_weights(const BnQWeight *q,
                                                  const BnQWeight *k,
                                                  int q_dim,
                                                  int kv_dim) {
    return q && k &&
           q->rows == q_dim &&
           k->rows == kv_dim &&
           q->cols == k->cols &&
           bn_transformer_gpu_can_use_stacked_qk(q->type, k->type);
}

int bn_transformer_gpu_can_use_stacked_gateup(const BnQWeight *gate,
                                              const BnQWeight *up) {
    return gate && up &&
           gate->rows == up->rows &&
           gate->cols == up->cols &&
           bn_backend_quant_stacked_pair_same_format(gate->type, up->type);
}

int bn_transformer_gpu_can_matvec_split(const BnGPUBackend *gpu,
                                        int tensor_type) {
    uint32_t cap = bn_backend_quant_gpu_split_cap(tensor_type);
    return cap != 0 && bn_transformer_gpu_has_cap(gpu, cap);
}

int bn_transformer_gpu_can_fused_gateup_silu(const BnGPUBackend *gpu,
                                             int tensor_type,
                                             int act_type) {
    if (!bn_transformer_gpu_fused_gateup_silu_policy_allows(gpu, tensor_type))
        return 0;
    uint32_t cap = bn_backend_quant_gpu_fused_gateup_silu_cap(tensor_type);
    return cap != 0 && act_type != 1 && bn_transformer_gpu_has_cap(gpu, cap);
}

int bn_transformer_gpu_can_fused_gateup_silu_pair(const BnGPUBackend *gpu,
                                                  int gate_type,
                                                  int up_type,
                                                  int act_type) {
    uint32_t gate_cap = bn_backend_quant_gpu_fused_gateup_silu_cap(gate_type);
    uint32_t up_cap = bn_backend_quant_gpu_fused_gateup_silu_cap(up_type);
    return gate_cap != 0 && gate_cap == up_cap &&
           bn_transformer_gpu_can_fused_gateup_silu(gpu, gate_type, act_type);
}

int bn_transformer_gpu_can_gateup_split_activation(const BnGPUBackend *gpu,
                                                   int tensor_type,
                                                   int act_type) {
    return bn_transformer_gpu_can_matvec_split(gpu, tensor_type) &&
           bn_backend_quant_can_gpu_gateup_split_activation(tensor_type,
                                                           act_type);
}

uint32_t bn_transformer_gpu_matvec_q8k_dot_flags(int tensor_type,
                                                 int enabled) {
    return bn_backend_quant_gpu_matvec_q8k_dot_flag(tensor_type, enabled);
}

uint32_t bn_transformer_gpu_matvec_exact_q6k_flags(int tensor_type,
                                                   int enabled) {
    return bn_backend_quant_gpu_matvec_exact_q6k_flag(tensor_type, enabled);
}

int bn_transformer_gpu_float_buffer_type(void) {
    return bn_backend_quant_gpu_float_buffer_type();
}

uint32_t bn_transformer_gpu_exact_silu_flags(int tensor_type,
                                             int use_silu) {
    return use_silu && bn_backend_quant_gpu_requires_exact_silu(tensor_type)
        ? BN_GPU_OP_FLAG_EXACT_SILU
        : 0u;
}

int bn_transformer_gpu_prefers_gateup_split(int tensor_type) {
    return bn_backend_quant_gpu_prefers_gateup_split(tensor_type);
}

int bn_transformer_gpu_stacked_pair_same_format(int left_type,
                                                int right_type) {
    return bn_backend_quant_stacked_pair_same_format(left_type, right_type);
}

int bn_transformer_gpu_shared_q4_q8_gateup_dot_eligible(int gate_type,
                                                        int up_type,
                                                        int cols) {
    return cols % 256 == 0 &&
           bn_backend_quant_moe_gateup_q4(gate_type, up_type);
}

int bn_transformer_gpu_can_flash_attn(const BnGPUBackend *gpu) {
    return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_FLASH_ATTN);
}

int bn_transformer_gpu_can_layerwise_rope(const BnGPUBackend *gpu) {
    return bn_transformer_gpu_has_cap(gpu, BN_GPU_CAP_LAYERWISE_ROPE);
}

int bn_transformer_gpu_requires_layerwise_rope(const BnConfig *c,
                                               const BnWeights *w) {
    return c && w &&
           bn_model_arch_uses_per_layer_embedding(c) &&
           w->rope_freqs != NULL;
}

BnBackendPlacement bn_transformer_gpu_backend_placement(
    const BnGPUBackend *gpu) {
    if (!gpu)
        return BN_BACKEND_GPU_UNKNOWN;
    switch (gpu->kind) {
        case BN_GPU_BACKEND_METAL: return BN_BACKEND_METAL;
        case BN_GPU_BACKEND_WEBGPU: return BN_BACKEND_WEBGPU;
        case BN_GPU_BACKEND_CUDA: return BN_BACKEND_CUDA;
        default: return BN_BACKEND_GPU_UNKNOWN;
    }
}

int bn_transformer_gpu_cuda_prefill_ssm_layer_disabled(void) {
    return bn_gpu_policy_cuda_prefill_ssm_layer_disabled();
}

int bn_transformer_gpu_fused_gateup_silu_policy_allows(
    const BnGPUBackend *gpu,
    int tensor_type) {
    if (!bn_gpu_policy_fused_gateup_enabled())
        return 0;
    if (bn_transformer_gpu_backend_is_cuda(gpu) &&
        bn_backend_quant_gpu_fused_gateup_requires_cuda_opt_in(tensor_type) &&
        !bn_gpu_policy_cuda_q5k_fused_gateup_enabled())
        return 0;
    return 1;
}

int bn_transformer_gpu_q4_q8_fused_gateup_enabled(int use_q4_q8) {
    return use_q4_q8 && bn_gpu_policy_q4_q8_fused_gateup_enabled();
}

int bn_transformer_gpu_gateup_split_enabled(void) {
    return bn_gpu_policy_gateup_split_enabled();
}

int bn_transformer_gpu_q4_q8_ffn_down_enabled(int use_q4_q8_down) {
    return use_q4_q8_down &&
           bn_gpu_policy_q4_q8_ffn_down_enabled();
}

int bn_transformer_gpu_qkv_split_enabled(int use_q4_q8) {
    return !use_q4_q8 && bn_gpu_policy_qkv_split_enabled();
}

int bn_transformer_gpu_qk_split_enabled(void) {
    return bn_gpu_policy_qkv_split_enabled();
}

int bn_transformer_gpu_qkv_split_debug_enabled(void) {
    return bn_gpu_policy_qkv_split_debug_enabled();
}

int bn_transformer_gpu_ssm_qkvz_split_enabled(void) {
    return bn_gpu_policy_ssm_qkvz_split_enabled();
}

int bn_transformer_gpu_ssm_ab_stack_enabled(void) {
    return bn_gpu_policy_ssm_ab_stack_enabled();
}

int bn_transformer_gpu_split_residual_rmsnorm_enabled(void) {
    return bn_gpu_policy_split_residual_rmsnorm_enabled();
}

int bn_transformer_gpu_shared_q4_q8_dot_enabled(int eligible) {
    return eligible &&
           bn_gpu_policy_cuda_shared_q4_q8_dot_enabled();
}

int bn_transformer_gpu_shared_expert_gate_enabled(int eligible) {
    return eligible &&
           bn_gpu_policy_cuda_shared_expert_gate_enabled();
}

uint32_t bn_transformer_gpu_moe_gateup_task_flags(const BnConfig *c) {
    return bn_model_arch_moe_forces_float_kquant_gateup(c)
        ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT
        : 0u;
}

int bn_transformer_gpu_moe_gateup_split_supported(
    const BnGPUBackend *gpu,
    const BnMoEExpertMap *map,
    int split_op_code) {
    if (!map || !bn_gpu_quant_split_op_is_q4k(split_op_code))
        return 0;
    return bn_transformer_gpu_can_matvec_split(gpu, map->gate_type) &&
           map->up_type == map->gate_type &&
           map->gate_rows == map->up_rows &&
           map->gate_cols == map->up_cols;
}

int bn_transformer_gpu_matvec_split_op_code(int tensor_type) {
    return bn_gpu_quant_split_op_code(tensor_type);
}

int bn_transformer_gpu_dense_gateup_exact_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *gate,
    const BnQWeight *up,
    int activation,
    int split_op_code) {
    if (!gate || !up || activation == 1 ||
        !bn_gpu_quant_split_op_is_q4k(split_op_code))
        return 0;
    return bn_transformer_gpu_can_use_stacked_gateup(gate, up) &&
           bn_transformer_gpu_can_matvec_split(gpu, gate->type);
}

int bn_transformer_gpu_packed_qkv_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *qkv,
    int use_packed_qkv,
    int kv_f16,
    int split_op_code) {
    return qkv && use_packed_qkv && !kv_f16 &&
           bn_gpu_quant_split_op_is_q5k(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, qkv->type);
}

int bn_transformer_gpu_qkv_split_standard_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code) {
    return q && bn_gpu_quant_split_op_is_standard(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_qkv_split_q8_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code) {
    return q && bn_gpu_quant_split_op_is_q8(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_qkv_split_q5_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    int split_op_code) {
    return q && bn_gpu_quant_split_op_is_q5k(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_qk_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *q,
    const BnQWeight *k,
    int q_dim,
    int kv_dim,
    int split_op_code) {
    if (!q || !k || !bn_gpu_quant_split_op_known(split_op_code))
        return 0;
    return q->rows == q_dim &&
           k->rows == kv_dim &&
           q->cols == k->cols &&
           bn_transformer_gpu_stacked_pair_same_format(q->type, k->type) &&
           bn_transformer_gpu_can_matvec_split(gpu, q->type);
}

int bn_transformer_gpu_ssm_qkvz_split_supported(
    const BnGPUBackend *gpu,
    const BnQWeight *qkv,
    int split_op_code) {
    return qkv && bn_gpu_quant_split_op_known(split_op_code) &&
           bn_transformer_gpu_can_matvec_split(gpu, qkv->type);
}

int bn_transformer_gpu_can_use_stacked_alpha_beta(const BnQWeight *alpha,
                                                  const BnQWeight *beta) {
    return alpha && beta &&
           alpha->rows == beta->rows &&
           alpha->cols == beta->cols &&
           bn_transformer_gpu_stacked_pair_same_format(alpha->type,
                                                       beta->type);
}

int bn_transformer_gpu_logits_needs_cpu_fallback(
    const BnGPUBackend *gpu,
    const BnTransformerGPULogitResources *logits) {
    if (!gpu || !logits || !logits->cpu_weight)
        return 0;

    size_t max_storage_binding =
        bn_gpu_policy_max_storage_binding_bytes(
            gpu->max_storage_binding_size);

    return bn_qweight_data_size(logits->cpu_weight) > max_storage_binding;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_layer(
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim) {
    if (!lw || !bn_model_arch_uses_all_active_two_expert_moe(c, dim))
        return 0;
    return bn_transformer_gpu_moe_routed_q4_down(&lw->moe.expert_map, 0);
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_layer_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_all2_q4q6_moe_layer(c, lw, dim);
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_model(const BnConfig *c,
                                                const BnWeights *w) {
    if (!w || !bn_model_arch_uses_all_active_two_expert_moe(c, c ? c->dim : 0))
        return 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        if (!lw->moe.router_weight)
            continue;
        if (bn_transformer_gpu_cuda_all2_q4q6_moe_layer(c, lw, c->dim))
            return 1;
    }
    return 0;
}

static int cuda_all2_q4q6_moe_requires_opt_in(const BnConfig *c,
                                              const BnWeights *w) {
    return bn_transformer_gpu_cuda_all2_q4q6_moe_model(c, w) &&
           !gpu_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
                            "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") &&
           gpu_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE",
                           "BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
}

static int small_dense_cuda_native_by_default(
    const BnConfig *c,
    const BnWeights *w) {
    if (!c || !w || !bn_model_arch_uses_small_cuda_dense_shape(c))
        return 0;
    if (w->output_weight.data) {
        if (!bn_backend_quant_cuda_small_dense_supported(w->output_weight.type))
            return 0;
    } else if (!bn_backend_quant_cuda_small_dense_supported(w->emb_type)) {
        return 0;
    }
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        const BnQWeight *weights[] = {
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, &lw->attn.wo,
            &lw->ffn.ffn_gate, &lw->ffn.ffn_up, &lw->ffn.ffn_down,
        };
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (weights[i]->data &&
                !bn_backend_quant_cuda_small_dense_supported(weights[i]->type))
                return 0;
        }
    }
    return 1;
}

static int small_dense_cuda_q8_native_by_default(
    const BnConfig *c,
    const BnWeights *w) {
    if (!c || !w || !bn_model_arch_uses_small_cuda_dense_shape(c))
        return 0;
    if (w->output_weight.data) {
        if (!bn_backend_quant_cuda_small_dense_q8_supported(w->output_weight.type))
            return 0;
    } else if (!bn_backend_quant_cuda_small_dense_q8_supported(w->emb_type)) {
        return 0;
    }
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        const BnQWeight *weights[] = {
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, &lw->attn.wo,
            &lw->ffn.ffn_gate, &lw->ffn.ffn_up, &lw->ffn.ffn_down,
        };
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (weights[i]->data &&
                !bn_backend_quant_cuda_small_dense_q8_supported(weights[i]->type))
                return 0;
        }
    }
    return 1;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_cuda_all2_q4q6_moe_model(c, w) &&
           !gpu_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
                            "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") &&
           !gpu_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE",
                            "BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_safe_default(c, w);
}

int bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w) {
    return bn_model_arch_allows_small_cuda_dense_exact_q4_q8(c) &&
           small_dense_cuda_q8_native_by_default(c, w) &&
           !gpu_env_enabled("BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE",
                            "BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE");
}

int bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_safe_default(c, w);
}

int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int q4_q8_from_layer) {
    return q4_q8_from_layer < 0 &&
           bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_allows_small_cuda_dense_exact_q4_q8(c) &&
           !gpu_env_enabled("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8",
                            "BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8");
}

int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_to_layer(
    const BnConfig *c,
    int exact_q4_q8_default,
    int q4_q8_to_layer) {
    if (!exact_q4_q8_default || q4_q8_to_layer >= 0 || !c ||
        c->n_layers <= 33)
        return q4_q8_to_layer;
    return c->n_layers - 33 - 1;
}

int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_ffn_down_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_allows_small_cuda_dense_exact_q4_q8(c) &&
           gpu_env_enabled("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN",
                           "BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
}

int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w) {
    if (!bn_model_arch_uses_large_dense_shape(c) || !w ||
        bn_gpu_policy_cuda_large_hybrid_attention_enabled() ||
        bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_disabled())
        return 0;
    if (!bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_enabled() &&
        !bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_forced())
        return 0;
    if (bn_model_arch_uses_hybrid_ssm(c))
        return 1;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        if (lw->block_kind == BN_LAYER_BLOCK_ATTENTION && lw->ssm.wqkv.data)
            return 1;
    }
    return 0;
}

int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_default(c, w);
}

int bn_transformer_gpu_cuda_small_dense_prefill_decode_fallback_requested(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_allows_small_cuda_prefill_decode_fallback(c) &&
           gpu_env_enabled("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL",
                           "BN_CUDA_DISABLE_SMALL_QWEN_PREFILL");
}

int bn_transformer_gpu_cuda_small_dense_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_small_cuda_dense_prefill_min_tokens(c) > 0;
}

int bn_transformer_gpu_hybrid_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_model_arch_uses_hybrid_ssm(c) &&
           bn_transformer_gpu_backend_is_cuda(gpu);
}

int bn_transformer_gpu_moe_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_model_arch_uses_non_hybrid_moe(c) &&
           bn_transformer_gpu_backend_is_cuda(gpu);
}

int bn_transformer_gpu_cuda_large_hybrid_prefill_decode_fallback_default(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_uses_large_dense_hybrid_ssm(c) &&
           !bn_gpu_policy_cuda_large_hybrid_prefill_enabled();
}

int bn_transformer_gpu_cuda_matvec_fallback_kept(
    const BnModel *m,
    const BnGPUBackend *gpu) {
    if (!m || !bn_transformer_gpu_backend_is_cuda(gpu) || !gpu->execute)
        return 0;
    const BnConfig *c = &m->config;
    if (!bn_model_arch_uses_dense_attention_only(c))
        return 0;
    if (bn_gpu_policy_cuda_small_kquant_native_enabled(
            bn_model_arch_cpu_force_float_kquant(c)))
        return 1;
    if (!bn_model_arch_uses_small_cuda_dense_q8_native_shape(c))
        return 1;

    const BnWeights *w = &m->weights;
    if (w->output_weight.data) {
        if (!bn_backend_quant_cuda_small_dense_q8_supported(
                w->output_weight.type))
            return 0;
    } else if (!bn_backend_quant_cuda_small_dense_q8_supported(w->emb_type)) {
        return 0;
    }
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        const BnQWeight *weights[] = {
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, &lw->attn.wo,
            &lw->ffn.ffn_gate, &lw->ffn.ffn_up, &lw->ffn.ffn_down,
        };
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (weights[i]->data &&
                !bn_backend_quant_cuda_small_dense_q8_supported(
                    weights[i]->type))
                return 0;
        }
    }
    return 1;
}

int bn_transformer_gpu_dense_batch_prefill_shape_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_model_arch_dense_batch_prefill_shape_allowed(
        c, bn_transformer_gpu_backend_is_cuda(gpu));
}

int bn_transformer_gpu_batch_prefill_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    if (!c)
        return 0;
    if (bn_gpu_policy_prefill_matmul_disabled())
        return 0;
    if (bn_gpu_policy_prefill_matmul_enabled())
        return 1;
    if (c->kv_tq_bits != 0)
        return 0;
    if (bn_transformer_gpu_cuda_small_dense_prefill_decode_fallback_requested(
            gpu, c) ||
        bn_transformer_gpu_cuda_large_hybrid_prefill_decode_fallback_default(
            gpu, c))
        return 0;
    if (bn_model_arch_uses_hybrid_ssm(c)) {
        return bn_transformer_gpu_backend_is_cuda(gpu) &&
               gpu->prefill_ssm_layer &&
               bn_gpu_policy_cuda_prefill_hybrid_chain_enabled() &&
               !bn_transformer_gpu_cuda_prefill_ssm_layer_disabled();
    }
    if (bn_model_arch_uses_moe(c))
        return bn_transformer_gpu_backend_is_cuda(gpu) &&
               bn_gpu_policy_cuda_moe_prefill_enabled();
    return bn_transformer_gpu_dense_batch_prefill_shape_allowed(gpu, c);
}

int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    if (!c || !bn_transformer_gpu_backend_is_cuda(gpu) ||
        !bn_model_arch_uses_large_dense_hybrid_ssm(c))
        return 0;
    if (bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_enabled())
        return 1;
    return !bn_gpu_policy_cuda_large_hybrid_attention_enabled() &&
           !bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_disabled() &&
           bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_forced();
}

int bn_transformer_gpu_cuda_large_hybrid_prefill_chain_disabled_default(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_uses_large_dense_hybrid_ssm(c) &&
           !bn_gpu_policy_cuda_large_hybrid_prefill_chain_enabled();
}

int bn_transformer_gpu_cuda_prefill_direct_kv_allowed(
    const BnConfig *c,
    const BnWeights *w,
    const BnGPUBackend *gpu,
    int pos0,
    int n_tokens) {
    if (!c || !bn_transformer_gpu_backend_is_cuda(gpu))
        return 0;
    if (bn_gpu_policy_cuda_prefill_direct_kv_disabled())
        return 0;
    if ((bn_gpu_policy_cpu_decode_fallback_requested() ||
         bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_fallback_enabled(
             gpu, c, w) ||
         bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_fallback_enabled(
             gpu, c, w) ||
         bn_transformer_gpu_cuda_large_hybrid_cpu_attn_fallback_enabled(
             gpu, c)) &&
        !bn_gpu_policy_cuda_prefill_direct_kv_with_cpu_fallback_enabled())
        return 0;
    if (c->kv_f16 || pos0 < 0 || pos0 + n_tokens > c->seq_len)
        return 0;
    return 1;
}

int bn_transformer_gpu_cuda_prefill_attention_min_tokens(void) {
    return bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(16);
}

int bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    if (bn_gpu_policy_cuda_prefill_attention_min_tokens_configured())
        return bn_transformer_gpu_cuda_prefill_attention_min_tokens();
    if (bn_transformer_gpu_backend_is_cuda(gpu) && c) {
        int arch_min = bn_model_arch_small_cuda_dense_prefill_min_tokens(c);
        if (arch_min > 0)
            return arch_min;
    }
    if (bn_transformer_gpu_backend_is_cuda(gpu) && c)
        return 16;
    return bn_transformer_gpu_cuda_prefill_attention_min_tokens();
}

int bn_transformer_gpu_dense_ffn_batch_tokens_allowed(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens) {
    return !bn_transformer_gpu_backend_is_cuda(gpu) ||
           n_tokens >=
               bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    if (bn_gpu_policy_cuda_moe_prefill_min_tokens_configured())
        return bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(1);
    if (bn_transformer_gpu_backend_is_cuda(gpu) && c)
        return bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(16);
    return bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(
        bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(c, gpu));
}

int bn_transformer_gpu_cuda_prefill_moe_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           gpu->moe_route_routed_ffn_batch_norm_resid &&
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(c) &&
           !bn_transformer_gpu_all2_q4_moe_requires_opt_in(
               c, map, dim, allow_q4_down);
}

int bn_transformer_gpu_prefill_moe_layer_backend_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down) {
    return gpu && gpu->prefill_moe_layer &&
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(c) &&
           !bn_transformer_gpu_all2_q4_moe_requires_opt_in(
               c, map, dim, allow_q4_down);
}

int bn_transformer_gpu_prefill_moe_layer_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down,
    int n_tokens) {
    return bn_transformer_gpu_prefill_moe_layer_backend_available(
               gpu, c, map, dim, allow_q4_down) &&
           n_tokens >=
               bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_cuda_prefill_ssm_moe_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down,
    int n_tokens) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           gpu->prefill_ssm_layer &&
           !bn_transformer_gpu_cuda_prefill_ssm_layer_disabled() &&
           n_tokens >=
               bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(c, gpu) &&
           bn_transformer_gpu_cuda_prefill_moe_ffn_batch_available(
               gpu, c, map, dim, allow_q4_down);
}

int bn_transformer_gpu_prefill_ssm_layer_backend_available(
    const BnGPUBackend *gpu) {
    return gpu && gpu->prefill_ssm_layer;
}

int bn_transformer_gpu_cuda_prefill_ssm_dense_chain_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int n_tokens) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           gpu->prefill_ssm_layer &&
           !bn_transformer_gpu_cuda_prefill_ssm_layer_disabled() &&
           n_tokens >=
               bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_cuda_prefill_dense_chain_enabled(void) {
    return bn_gpu_policy_cuda_prefill_dense_chain_enabled();
}

int bn_transformer_gpu_cuda_prefill_hybrid_chain_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_gpu_policy_cuda_prefill_hybrid_chain_enabled() &&
           !bn_transformer_gpu_cuda_large_hybrid_prefill_chain_disabled_default(
               gpu, c);
}

int bn_transformer_gpu_cuda_prefill_attention_enabled(void) {
    return bn_gpu_policy_cuda_prefill_attention_enabled();
}

int bn_transformer_gpu_cuda_prefill_ssm_run_chain_enabled(void) {
    return bn_gpu_policy_cuda_prefill_ssm_run_chain_enabled();
}

int bn_transformer_gpu_cuda_prefill_ssm_ffn_fuse_allowed(void) {
    return bn_gpu_policy_cuda_prefill_ssm_ffn_fuse_allowed();
}

int bn_transformer_gpu_cuda_prefill_moe_chain_debug_enabled(void) {
    return bn_gpu_policy_cuda_prefill_moe_chain_debug_enabled();
}

int bn_transformer_gpu_cuda_prefill_hybrid_chain_debug_enabled(void) {
    return bn_gpu_policy_cuda_prefill_hybrid_chain_debug_enabled();
}

int bn_transformer_gpu_cuda_moe_prefill_enabled(void) {
    return bn_gpu_policy_cuda_moe_prefill_enabled();
}

int bn_transformer_gpu_cuda_moe_prefill_min_tokens(void) {
    return bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(1);
}

int bn_transformer_gpu_moe_prefill_backend_available(
    const BnGPUBackend *gpu) {
    return bn_transformer_gpu_backend_is_cuda(gpu);
}

int bn_transformer_gpu_moe_prefill_tokens_allowed(
    const BnGPUBackend *gpu,
    int n_tokens) {
    return bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           n_tokens >= bn_transformer_gpu_cuda_moe_prefill_min_tokens();
}

int bn_transformer_gpu_cuda_moe_cache_prefill_enabled(void) {
    return bn_gpu_policy_cuda_moe_cache_prefill_enabled();
}

int bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int gpu_moe_cache_available) {
    return gpu_moe_cache_available &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_model_arch_uses_two_expert_all_active_moe(c) &&
           bn_transformer_gpu_cuda_moe_cache_prefill_enabled();
}

int bn_transformer_gpu_cuda_moe_prefill_shared_fuse_enabled(void) {
    return bn_gpu_policy_cuda_moe_prefill_shared_fuse_enabled();
}

int bn_transformer_gpu_moe_prefill_shared_batch_available(
    const BnGPUBackend *gpu,
    int n_tokens,
    int backend_available) {
    return backend_available &&
           bn_transformer_gpu_moe_prefill_tokens_allowed(gpu, n_tokens) &&
           gpu->dense_ffn_batch &&
           bn_transformer_gpu_cuda_moe_prefill_shared_fuse_enabled();
}

int bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    int backend_available) {
    return backend_available &&
           c && lw &&
           c->has_shared_expert &&
           lw->shared.shared_gate.data &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           bn_transformer_gpu_cuda_moe_prefill_shared_fuse_enabled();
}

int bn_transformer_gpu_cuda_moe_route_batch_debug_enabled(void) {
    return bn_gpu_policy_cuda_moe_route_batch_debug_enabled();
}

int bn_transformer_gpu_moe_prefill_route_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int backend_available) {
    return backend_available &&
           bn_model_arch_uses_more_than_two_expert_moe(c) &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           gpu->moe_route_batch;
}

int bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return c &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           gpu->moe_route_routed_ffn_batch_norm_resid &&
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(c);
}

int bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down) {
    return c &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           gpu->moe_route_routed_ffn_batch &&
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(c) &&
           !bn_transformer_gpu_all2_q4_moe_requires_opt_in(
               c, map, dim, allow_q4_down);
}

int bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down,
    int prefer_cached_expert_batch) {
    return !prefer_cached_expert_batch &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           gpu->moe_routed_ffn_batch &&
           !bn_transformer_gpu_all2_q4_moe_requires_opt_in(
               c, map, dim, allow_q4_down);
}

int bn_transformer_gpu_moe_prefill_split_expert_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down,
    int used_resident_expert_batch) {
    return !used_resident_expert_batch &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           gpu->moe_ffn_batch &&
           !bn_transformer_gpu_all2_q4_moe_requires_opt_in(
               c, map, dim, allow_q4_down);
}

int bn_transformer_gpu_moe_prefill_single_expert_batch_available(
    const BnGPUBackend *gpu,
    int n_tokens) {
    return bn_transformer_gpu_moe_prefill_tokens_allowed(gpu, n_tokens) &&
           gpu->dense_ffn_batch;
}

int bn_transformer_gpu_cuda_moe_lazy_aux_cache_enabled(void) {
    return bn_gpu_policy_cuda_moe_lazy_aux_cache_enabled();
}

int bn_transformer_gpu_moe_quant_only_without_aux_cache(
    const BnGPUBackend *gpu,
    int tensor_type,
    int allow_aux_cache) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           !allow_aux_cache &&
           !bn_transformer_gpu_cuda_moe_lazy_aux_cache_enabled() &&
           bn_backend_quant_cuda_lazy_moe_aux_cache_candidate(tensor_type);
}

int bn_transformer_gpu_cuda_large_hybrid_prefill_disabled(void) {
    return bn_gpu_policy_cuda_large_hybrid_prefill_disabled();
}

int bn_transformer_gpu_cuda_small_dense_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int tensor_type) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_backend_quant_supports_q8_logits_refine(tensor_type) &&
           bn_model_arch_allows_small_cuda_q8_logit_refine(c) &&
           gpu_env_enabled("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
                           "BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE") &&
           !gpu_env_enabled("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE",
                            "BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_q6_logits_refine_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_all2_q4q6_moe_model(c, w) &&
           gpu_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
                           "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") &&
           !gpu_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE",
                            "BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE");
}

int bn_transformer_gpu_q6_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q6_refine_default) {
    return bn_gpu_policy_q6_logits_refine_enabled(
        bn_transformer_gpu_backend_is_cuda(gpu), q6_refine_default);
}

int bn_transformer_gpu_q6_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_q6_logits,
    int q6_refine_default) {
    return refine_q6_logits &&
           q6_refine_default &&
           logits &&
           bn_backend_quant_supports_q6k_logits_refine(logits->type) &&
           logits->cpu_weight != NULL;
}

int bn_transformer_gpu_q6_logits_refine_top(int q6_refine_default) {
    return bn_gpu_policy_q6_logits_refine_top_or_default(
        q6_refine_default ? 64 : 8);
}

int bn_transformer_gpu_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q8_refine_default) {
    return bn_gpu_policy_q8_logits_refine_enabled(
        bn_transformer_gpu_backend_is_cuda(gpu), q8_refine_default);
}

int bn_transformer_gpu_q8_logits_refine_captures_xb(
    const BnTransformerGPULogitResources *logits,
    int refine_q8_logits) {
    return refine_q8_logits &&
           logits &&
           bn_backend_quant_supports_q8_logits_refine(logits->type) &&
           logits->cpu_weight != NULL;
}

int bn_transformer_gpu_q8_logits_refine_top(int q8_refine_default) {
    return bn_gpu_policy_q8_logits_refine_top_or_default(
        q8_refine_default ? 16 : 8);
}

BnTransformerGPULogitsRefinePolicy bn_transformer_gpu_logits_refine_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    const BnTransformerGPULogitResources *logits,
    int small_dense_exact_q4_q8_default) {
    BnTransformerGPULogitsRefinePolicy p = {0};
    p.q6_default =
        bn_transformer_gpu_cuda_all2_q4q6_moe_q6_logits_refine_default(
            gpu, c, w);
    p.q6_enabled = bn_transformer_gpu_q6_logits_refine_enabled(
        gpu, p.q6_default);
    p.q6_captures_xb = bn_transformer_gpu_q6_logits_refine_captures_xb(
        logits, p.q6_enabled, p.q6_default);

    int tensor_type = logits ? logits->type : -1;
    p.q8_default =
        small_dense_exact_q4_q8_default &&
        bn_transformer_gpu_cuda_small_dense_q8_logits_refine_enabled(
            gpu, c, tensor_type);
    p.q8_enabled = bn_transformer_gpu_q8_logits_refine_enabled(
        gpu, p.q8_default);
    p.q8_captures_xb = bn_transformer_gpu_q8_logits_refine_captures_xb(
        logits, p.q8_enabled);
    return p;
}

int bn_transformer_gpu_cpu_logits_enabled(int gpu_logits_need_cpu) {
    return gpu_logits_need_cpu || bn_gpu_policy_cpu_logits_enabled();
}

int bn_transformer_gpu_compare_logits_enabled(void) {
    return bn_gpu_policy_compare_logits_enabled();
}

int bn_transformer_gpu_debug_argmax_compare_enabled(void) {
    return bn_gpu_policy_debug_argmax_compare_enabled();
}

int bn_transformer_gpu_argmax_debug_enabled(void) {
    return bn_gpu_policy_argmax_debug_enabled();
}

BnTransformerGPUGenerateArgmaxPolicy
bn_transformer_gpu_generate_argmax_policy(
    const BnGPUBackend *gpu,
    int top_logits,
    float temperature,
    float repeat_penalty) {
    BnTransformerGPUGenerateArgmaxPolicy policy = {0};
    policy.enabled =
        gpu &&
        gpu->argmax_activation &&
        top_logits <= 0 &&
        temperature == 0.0f &&
        repeat_penalty >= 1.0f;
    return policy;
}

int bn_transformer_gpu_matvec_argmax_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPULogitResources *logits,
    int want_argmax,
    int need_logits,
    int gpu_logits_need_cpu) {
    if (!gpu || !c || !logits || !want_argmax || need_logits ||
        !gpu->matvec_argmax_activation ||
        bn_transformer_gpu_cpu_logits_enabled(gpu_logits_need_cpu) ||
        bn_gpu_policy_cuda_logits_argmax_disabled() ||
        !bn_backend_quant_supports_q6k_logits_refine(logits->type))
        return 0;

    if (!bn_model_arch_uses_moe(c)) {
        return logits->rows > 262144 ||
               bn_gpu_policy_cuda_dense_logits_argmax_enabled();
    }
    if (bn_model_arch_uses_all_active_two_expert_moe(c, c->dim))
        return 1;
    if (bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled())
        return 1;
    return logits->cols == 1536 &&
           !bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled();
}

int bn_transformer_gpu_cuda_moe_decode_cacheable(
    const BnConfig *c,
    const BnWeights *w,
    const BnBackendModel *backend) {
    if (bn_gpu_policy_cuda_moe_decode_cache_disabled() ||
        !c || !w || !backend || !bn_model_arch_uses_moe(c))
        return 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        if (!lw->moe.router_weight)
            continue;
        const BnMoEExpertMap *em = &lw->moe.expert_map;
        int routed_q4 = bn_transformer_gpu_moe_routed_q4(em);
        int routed_q8 = bn_transformer_gpu_moe_routed_q8(em);
        int has_router =
            bn_backend_model_handle(backend, l, BN_BACKEND_HANDLE_MOE_ROUTER) ||
            bn_backend_model_handle(backend, l,
                                    BN_BACKEND_HANDLE_MOE_ROUTER_DIFF);
        if (!has_router ||
            !bn_backend_model_handle(backend, l,
                                     BN_BACKEND_HANDLE_MOE_GATE_ALL) ||
            !bn_backend_model_handle(backend, l,
                                     BN_BACKEND_HANDLE_MOE_UP_ALL) ||
            !bn_backend_model_handle(backend, l,
                                     BN_BACKEND_HANDLE_MOE_DOWN_ALL) ||
            (!routed_q4 && !routed_q8) ||
            em->gate_rows != c->moe_intermediate_size ||
            em->up_rows != c->moe_intermediate_size ||
            em->gate_cols != c->dim ||
            em->up_cols != c->dim ||
            em->down_rows != c->dim ||
            em->down_cols != c->moe_intermediate_size)
            return 0;
    }
    return 1;
}

int bn_transformer_gpu_cuda_decode_cacheable(
    const BnGPUBackend *gpu,
    int emit_logits,
    int want_argmax,
    int gpu_logits_need_cpu,
    int has_moe,
    int cacheable_resident_moe,
    int q6_logits_refine_captures_xb,
    int q8_logits_refine_captures_xb,
    int need_logits,
    int cpu_fallback_layer,
    int cpu_fallback_from_layer,
    int cpu_fallback_attn_layer,
    int cpu_fallback_attn_from_layer,
    int cpu_fallback_ffn_layer,
    int cpu_fallback_ffn_from_layer,
    int cpu_fallback_ffn_down_from_layer,
    int compare_attention_layer,
    int compare_gqa_layer,
    int compare_qkv_layer,
    int compare_ffn_down_layer,
    int compare_ffn_state_layer) {
    if ((!emit_logits || want_argmax ||
         bn_gpu_policy_cuda_decode_logits_cache_enabled(
             gpu_logits_need_cpu)) == 0)
        return 0;
    if (!bn_transformer_gpu_backend_is_cuda(gpu))
        return 0;
    if (has_moe && !cacheable_resident_moe &&
        !bn_gpu_policy_cuda_moe_decode_cache_enabled())
        return 0;
    if (bn_gpu_policy_cuda_decode_cache_disabled())
        return 0;
    if (q6_logits_refine_captures_xb && !(want_argmax && !need_logits))
        return 0;
    if (q8_logits_refine_captures_xb && !(want_argmax && !need_logits))
        return 0;
    if (cpu_fallback_layer >= 0 || cpu_fallback_from_layer >= 0 ||
        cpu_fallback_attn_layer >= 0 || cpu_fallback_attn_from_layer >= 0 ||
        cpu_fallback_ffn_layer >= 0 || cpu_fallback_ffn_from_layer >= 0 ||
        cpu_fallback_ffn_down_from_layer >= 0)
        return 0;
    if (compare_attention_layer >= 0 || compare_gqa_layer >= 0 ||
        compare_qkv_layer >= 0 || compare_ffn_down_layer >= 0 ||
        compare_ffn_state_layer >= 0)
        return 0;
    if (bn_gpu_policy_cuda_q4_q8_decode_cache_disabled() ||
        bn_transformer_gpu_cpu_logits_enabled(gpu_logits_need_cpu) ||
        bn_transformer_gpu_compare_logits_enabled() ||
        bn_gpu_policy_metal_q6_q8k_enabled())
        return 0;
    return 1;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_moe_safe_default(
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_cuda_all2_q4q6_moe_model(c, w) &&
           !gpu_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
                            "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") &&
           !gpu_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE",
                            "BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE");
}

int bn_transformer_gpu_cuda_moe_exact_attention_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_moe_prefers_cuda_exact_attention(c) &&
           !gpu_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN",
                            "BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN");
}

int bn_transformer_gpu_ssm_cpu_fallback_required(
    const BnGPUBackend *gpu) {
    return !bn_transformer_gpu_backend_is_cuda(gpu) ||
           bn_gpu_policy_cuda_ssm_graph_disabled();
}

int bn_transformer_gpu_cuda_large_hybrid_argmax_blocked(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax) {
    return want_argmax &&
           bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_default(c, w) &&
           !bn_gpu_policy_cuda_large_hybrid_argmax_enabled();
}

static int gpu_policy_env_int(const char *name, int default_value) {
    const char *env = getenv(name);
    return env ? atoi(env) : default_value;
}

BnTransformerGPUCPUFallbackPolicy
bn_transformer_gpu_cpu_fallback_policy(void) {
    BnTransformerGPUCPUFallbackPolicy policy = {
        .layer = gpu_policy_env_int("BN_GPU_CPU_FALLBACK_LAYER", -1),
        .from_layer =
            gpu_policy_env_int("BN_GPU_CPU_FALLBACK_FROM_LAYER", -1),
        .attn_layer = gpu_policy_env_int("BN_GPU_CPU_ATTN_LAYER", -1),
        .attn_from_layer =
            gpu_policy_env_int("BN_GPU_CPU_ATTN_FROM_LAYER", -1),
        .ffn_layer = gpu_policy_env_int("BN_GPU_CPU_FFN_LAYER", -1),
        .ffn_from_layer =
            gpu_policy_env_int("BN_GPU_CPU_FFN_FROM_LAYER", -1),
        .ffn_down_from_layer =
            gpu_policy_env_int("BN_GPU_CPU_FFN_DOWN_FROM_LAYER", -1),
    };
    return policy;
}

static int gpu_cpu_attention_fallback_unset(
    const BnTransformerGPUCPUFallbackPolicy *policy) {
    return policy &&
           policy->layer < 0 &&
           policy->from_layer < 0 &&
           policy->attn_layer < 0 &&
           policy->attn_from_layer < 0;
}

BnTransformerGPUCPUFallbackPolicy
bn_transformer_gpu_decode_cpu_attention_fallback_policy(
    BnTransformerGPUCPUFallbackPolicy policy,
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    if (!gpu_cpu_attention_fallback_unset(&policy))
        return policy;
    int default_cpu_attention =
        bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_fallback_enabled(
            gpu, c, w) ||
        bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_fallback_enabled(
            gpu, c, w) ||
        bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_fallback_enabled(
            gpu, c, w);
    if (default_cpu_attention)
        policy.attn_from_layer = 0;
    return policy;
}

int bn_transformer_gpu_cpu_fallback_layer_selected(
    int layer,
    int exact_layer,
    int from_layer) {
    return (exact_layer >= 0 && layer == exact_layer) ||
           (from_layer >= 0 && layer >= from_layer);
}

BnTransformerGPUQ4Q8LayerPolicy
bn_transformer_gpu_q4_q8_layer_policy(const BnConfig *c) {
    int n_layers = c ? c->n_layers : 0;
    BnTransformerGPUQ4Q8LayerPolicy policy = {
        .from_layer = bn_gpu_policy_q4_q8_from_layer_or_default(n_layers),
        .to_layer = bn_gpu_policy_q4_q8_to_layer_or_default(
            n_layers, bn_gpu_policy_metal_q4_prepared_enabled()),
        .attn_only = bn_gpu_policy_q4_q8_attn_only_enabled(),
        .ffn_only = bn_gpu_policy_q4_q8_ffn_only_enabled(),
    };
    return policy;
}

BnTransformerGPUQ4Q8LayerUsePolicy
bn_transformer_gpu_q4_q8_layer_use_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnTransformerGPUQ4Q8LayerPolicy *policy,
    int layer,
    int small_dense_exact_q4_q8_default,
    int small_dense_exact_q4_q8_to_layer) {
    BnTransformerGPUQ4Q8LayerUsePolicy use = {0};
    if (!policy)
        return use;

    use.use_layer = policy->from_layer >= 0 &&
                    layer >= policy->from_layer &&
                    (policy->to_layer < 0 || layer <= policy->to_layer);
    use.small_dense_exact_q4_q8 =
        small_dense_exact_q4_q8_default &&
        (small_dense_exact_q4_q8_to_layer < 0 ||
         layer <= small_dense_exact_q4_q8_to_layer);
    if (use.small_dense_exact_q4_q8)
        use.use_layer = 1;

    int exact_attention =
        bn_transformer_gpu_cuda_moe_exact_attention_enabled(gpu, c);
    use.use_attention =
        (use.use_layer || exact_attention) && !policy->ffn_only;
    use.use_ffn = use.use_layer && !policy->attn_only;
    use.use_ffn_down = use.use_ffn;
    if (use.small_dense_exact_q4_q8 &&
        !bn_transformer_gpu_cuda_small_dense_exact_q4_q8_ffn_down_enabled(
            gpu, c))
        use.use_ffn_down = 0;
    return use;
}

BnTransformerGPUCachedDecodePolicy
bn_transformer_gpu_cached_decode_policy(
    int cached_op_count,
    int argmax_requested,
    int cached_has_logits,
    int matvec_argmax_available) {
    BnTransformerGPUCachedDecodePolicy policy = {0};
    policy.use_cache = cached_op_count > 0;
    policy.clear_cache =
        policy.use_cache &&
        argmax_requested &&
        !cached_has_logits &&
        !matvec_argmax_available;
    if (policy.clear_cache)
        policy.use_cache = 0;
    return policy;
}

BnTransformerGPUMoERouteLayerPolicy
bn_transformer_gpu_moe_route_layer_policy(void) {
    static int init = 0;
    static BnTransformerGPUMoERouteLayerPolicy policy = {-1, -1};
    if (!init) {
        bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_range(
            &policy.from_layer, &policy.to_layer);
        init = 1;
    }
    return policy;
}

BnTransformerGPUComparePolicy
bn_transformer_gpu_compare_policy(void) {
    BnTransformerGPUComparePolicy policy = {
        .attention_layer =
            gpu_policy_env_int("BN_GPU_COMPARE_ATTENTION_LAYER", -1),
        .attention_pos =
            gpu_policy_env_int("BN_GPU_COMPARE_ATTENTION_POS", -1),
        .gqa_layer = gpu_policy_env_int("BN_GPU_COMPARE_GQA_LAYER", -1),
        .gqa_pos = gpu_policy_env_int("BN_GPU_COMPARE_GQA_POS", -1),
        .qkv_layer = gpu_policy_env_int("BN_GPU_COMPARE_QKV_LAYER", -1),
        .qkv_pos = gpu_policy_env_int("BN_GPU_COMPARE_QKV_POS", -1),
        .ffn_down_layer =
            gpu_policy_env_int("BN_GPU_COMPARE_FFN_DOWN_LAYER", -1),
        .ffn_down_pos =
            gpu_policy_env_int("BN_GPU_COMPARE_FFN_DOWN_POS", -1),
        .ffn_state_layer =
            gpu_policy_env_int("BN_GPU_COMPARE_FFN_STATE_LAYER", -1),
        .ffn_state_pos =
            gpu_policy_env_int("BN_GPU_COMPARE_FFN_STATE_POS", -1),
    };
    return policy;
}

int bn_transformer_gpu_flash_attention_enabled(
    const BnGPUBackend *gpu,
    int config_flash_attn,
    int has_moe,
    int n_kv) {
    int cuda_backend = bn_transformer_gpu_backend_is_cuda(gpu);
    int flash_min_kv = bn_gpu_policy_flash_min_kv_or_default(0);
    int flash_max_kv =
        bn_gpu_policy_flash_max_kv_or_default(cuda_backend, 0);

    return bn_transformer_gpu_can_flash_attn(gpu) &&
           (has_moe || config_flash_attn || cuda_backend) &&
           n_kv >= flash_min_kv &&
           (flash_max_kv <= 0 || n_kv <= flash_max_kv);
}

int bn_transformer_gpu_moe_routed_q4(const BnMoEExpertMap *map) {
    return bn_transformer_gpu_moe_routed_q4_down(map, 1);
}

int bn_transformer_gpu_moe_routed_q4_down(const BnMoEExpertMap *map,
                                          int allow_q4_down) {
    return map &&
           bn_backend_quant_moe_route_q4_down(map->gate_type,
                                              map->up_type,
                                              map->down_type,
                                              allow_q4_down);
}

int bn_transformer_gpu_moe_routed_q8(const BnMoEExpertMap *map) {
    return map &&
           bn_backend_quant_moe_route_q8(map->gate_type,
                                         map->up_type,
                                         map->down_type);
}

int bn_transformer_gpu_cuda_moe_route_topk_enabled(
    void *moe_router,
    int all2_q4q6_moe,
    int all2_q4q6_moe_gpu_route_layer_selected) {
    int eligible = moe_router &&
                   (!all2_q4q6_moe ||
                    all2_q4q6_moe_gpu_route_layer_selected);
    return bn_gpu_policy_cuda_moe_router_topk_enabled(eligible);
}

int bn_transformer_gpu_cuda_moe_cpu_route_resident_ffn_enabled(
    const BnConfig *c,
    int all2_q4q6_moe,
    int gpu_route_topk,
    int moe_routed_q8) {
    if (all2_q4q6_moe && !gpu_route_topk &&
        !gpu_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT",
                         "BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT"))
        return 1;
    return bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled(
        !gpu_route_topk && moe_routed_q8 &&
        bn_model_arch_uses_more_than_two_expert_moe(c));
}

int bn_transformer_gpu_cuda_moe_routed_ffn_enabled(
    int gpu_route_topk,
    int cpu_route_resident_ffn,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all,
    const BnMoEExpertMap *map,
    int moe_hidden,
    int dim) {
    if ((!gpu_route_topk && !cpu_route_resident_ffn) ||
        !moe_gate_all || !moe_up_all || !moe_down_all ||
        (!bn_transformer_gpu_moe_routed_q4(map) &&
         !bn_transformer_gpu_moe_routed_q8(map)) ||
        !bn_gpu_policy_cuda_moe_routed_ffn_enabled(1))
        return 0;
    return map->gate_rows == moe_hidden &&
           map->up_rows == moe_hidden &&
           map->gate_cols == dim &&
           map->up_cols == dim &&
           map->down_rows == dim &&
           map->down_cols == moe_hidden;
}

BnTransformerGPUMoEDecodeRoutePolicy
bn_transformer_gpu_moe_decode_route_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUMoERouteLayerPolicy *layer_policy,
    int layer,
    int dim,
    void *moe_router,
    void *router_diff,
    void *moe_gate_all,
    void *moe_up_all,
    void *moe_down_all) {
    BnTransformerGPUMoEDecodeRoutePolicy policy = {0};
    int from_layer = layer_policy ? layer_policy->from_layer : -1;
    int to_layer = layer_policy ? layer_policy->to_layer : -1;

    policy.all2_q4q6_moe =
        bn_transformer_gpu_cuda_all2_q4q6_moe_layer_enabled(
            gpu, c, lw, dim);
    policy.route_layer_selected =
        bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_selected(
            layer, from_layer, to_layer);
    policy.exact_gpu_route =
        bn_transformer_gpu_cuda_all2_q4q6_moe_exact_gpu_route_enabled(
            policy.all2_q4q6_moe, policy.route_layer_selected);
    policy.router = bn_transformer_gpu_cuda_all2_q4q6_moe_router(
        c, moe_router, router_diff, policy.route_layer_selected,
        policy.exact_gpu_route);
    if (!c || !c->moe_norm_topk_prob)
        policy.route_flags |= BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM;

    int routed_q8 = lw &&
        bn_transformer_gpu_moe_routed_q8(&lw->moe.expert_map);
    policy.gpu_route_topk =
        bn_transformer_gpu_cuda_moe_route_topk_enabled(
            policy.router, policy.all2_q4q6_moe,
            policy.route_layer_selected);
    policy.cpu_route_resident_ffn =
        bn_transformer_gpu_cuda_moe_cpu_route_resident_ffn_enabled(
            c, policy.all2_q4q6_moe, policy.gpu_route_topk, routed_q8);
    policy.gpu_routed_ffn =
        lw && bn_transformer_gpu_cuda_moe_routed_ffn_enabled(
            policy.gpu_route_topk, policy.cpu_route_resident_ffn,
            moe_gate_all, moe_up_all, moe_down_all, &lw->moe.expert_map,
            c ? c->moe_intermediate_size : 0, dim);
    return policy;
}

BnTransformerGPUMoEDirectRoutePolicy
bn_transformer_gpu_moe_direct_route_policy(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all) {
    BnTransformerGPUMoEDirectRoutePolicy policy = {0};
    policy.router_diff = router_diff;
    policy.enabled =
        bn_transformer_gpu_backend_is_cuda(gpu) &&
        bn_transformer_gpu_cuda_all2_moe_direct_route_enabled(
            c, router_diff, moe_gate_all);
    return policy;
}

BnTransformerGPUMoEAll2ResourcePolicy
bn_transformer_gpu_moe_all2_resource_policy(const BnConfig *c) {
    BnTransformerGPUMoEAll2ResourcePolicy policy = {0};
    policy.enabled =
        bn_model_arch_uses_all_active_two_expert_moe(c, c ? c->dim : 0);
    return policy;
}

int bn_transformer_gpu_cuda_all2_moe_direct_route_enabled(
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all) {
    return router_diff &&
           bn_model_arch_uses_all_active_two_expert_moe(c, c ? c->dim : 0) &&
           c->moe_norm_topk_prob &&
           !moe_gate_all &&
           bn_gpu_policy_cuda_moe_router_gpu_enabled();
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_selected(
    int layer,
    int route_from_layer,
    int route_to_layer) {
    if (!bn_gpu_policy_cuda_moe_router_gpu_enabled() &&
        !gpu_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
                         "BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE"))
        return 0;
    return route_from_layer < 0 ||
           (layer >= route_from_layer &&
            (route_to_layer < 0 || layer <= route_to_layer));
}

void bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_range(
    int *route_from_layer,
    int *route_to_layer) {
    const char *env;

    if (route_from_layer)
        *route_from_layer = -1;
    if (route_to_layer)
        *route_to_layer = -1;

    env = gpu_env_value("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER",
                        "BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
    if (env && route_from_layer)
        *route_from_layer = atoi(env);
    env = gpu_env_value("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER",
                        "BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
    if (env && route_to_layer)
        *route_to_layer = atoi(env);
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_exact_gpu_route_enabled(
    int all2_q4q6_moe,
    int route_layer_selected) {
    return all2_q4q6_moe &&
           route_layer_selected &&
           gpu_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
                           "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") &&
           !gpu_env_enabled("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE",
                            "BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE");
}

void *bn_transformer_gpu_cuda_all2_q4q6_moe_router(
    const BnConfig *c,
    void *moe_router,
    void *router_diff,
    int route_layer_selected,
    int exact_gpu_route) {
    if (router_diff &&
        bn_model_arch_uses_all_active_two_expert_moe(c, c ? c->dim : 0) &&
        route_layer_selected &&
        bn_gpu_policy_cuda_moe_router_diff2_enabled() &&
        !exact_gpu_route)
        return router_diff;
    return moe_router;
}

int bn_transformer_gpu_all2_q4_moe_requires_opt_in(
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down) {
    if (!c || !map ||
        !bn_model_arch_uses_all_active_two_expert_moe(c, dim) ||
        !bn_transformer_gpu_moe_routed_q4_down(map, allow_q4_down) ||
        gpu_env_enabled("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN",
                        "BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN"))
        return 0;
    return 1;
}

int bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down,
    int layer,
    int cpu_fallback_ffn_layer,
    int cpu_fallback_ffn_from_layer) {
    if (!bn_transformer_gpu_backend_is_cuda(gpu))
        return 1;
    if (bn_transformer_gpu_cuda_moe_ffn_disabled())
        return 1;
    if (bn_transformer_gpu_all2_q4_moe_requires_opt_in(
            c, map, dim, allow_q4_down))
        return 1;
    return bn_transformer_gpu_cpu_fallback_layer_selected(
        layer, cpu_fallback_ffn_layer, cpu_fallback_ffn_from_layer);
}

int bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(
    const BnConfig *c) {
    return bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(
        bn_model_arch_uses_more_than_two_expert_moe(c));
}

int bn_transformer_gpu_cuda_moe_ffn_disabled(void) {
    return bn_gpu_policy_cuda_moe_ffn_disabled();
}

int bn_transformer_gpu_cuda_moe_cpu_actual_override_enabled(int safe_default) {
    return safe_default ||
           bn_gpu_policy_cuda_moe_cpu_actual_override_enabled();
}

BnTransformerGPUMoEDebugPolicy bn_transformer_gpu_moe_debug_policy(
    int cpu_actual_safe_default,
    int compare_layer_selected) {
    BnTransformerGPUMoEDebugPolicy policy = {0};
    policy.override_cpu_actual =
        bn_transformer_gpu_cuda_moe_cpu_actual_override_enabled(
            cpu_actual_safe_default);
    policy.compare_layer = compare_layer_selected;
    policy.compare_route =
        compare_layer_selected &&
        bn_transformer_gpu_moe_compare_route_enabled();
    policy.compare_input_norm =
        compare_layer_selected &&
        bn_transformer_gpu_moe_compare_input_norm_enabled();
    policy.compare_actual =
        compare_layer_selected &&
        bn_transformer_gpu_moe_compare_actual_enabled();
    return policy;
}

int bn_transformer_gpu_moe_compare_layer_selected(int layer, int pos) {
    return bn_gpu_policy_moe_compare_layer_selected(layer, pos);
}

int bn_transformer_gpu_moe_compare_input_norm_enabled(void) {
    return bn_gpu_policy_moe_compare_input_norm_enabled();
}

int bn_transformer_gpu_moe_compare_actual_enabled(void) {
    return bn_gpu_policy_moe_compare_actual_enabled();
}

int bn_transformer_gpu_moe_compare_route_enabled(void) {
    return bn_gpu_policy_moe_compare_route_enabled();
}

int bn_transformer_gpu_moe_compare_raw_enabled(void) {
    return bn_gpu_policy_moe_compare_raw_enabled();
}

int bn_transformer_gpu_moe_compare_mid_enabled(void) {
    return bn_gpu_policy_moe_compare_mid_enabled();
}

int bn_transformer_gpu_moe_compare_parts_enabled(void) {
    return bn_gpu_policy_moe_compare_parts_enabled();
}

int bn_transformer_gpu_moe_compare_shared_mid_enabled(void) {
    return bn_gpu_policy_moe_compare_shared_mid_enabled();
}

int bn_transformer_gpu_moe_compare_shared_down_enabled(void) {
    return bn_gpu_policy_moe_compare_shared_down_enabled();
}

int bn_transformer_gpu_moe_compare_norm_enabled(void) {
    return bn_gpu_policy_moe_compare_norm_enabled();
}

int bn_transformer_gpu_cuda_moe_shared_cpu_fallback_enabled(int eligible) {
    return bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled(eligible);
}

int bn_transformer_gpu_cuda_moe_gateup_split_enabled(
    const BnGPUBackend *gpu,
    int can_split) {
    return bn_transformer_gpu_backend_is_cuda(gpu) && can_split &&
           bn_gpu_policy_cuda_moe_gateup_split_enabled(can_split);
}

int bn_transformer_gpu_moe_route_profile_enabled(void) {
    return bn_gpu_policy_moe_route_profile_enabled();
}

int bn_transformer_gpu_moe_route_profile_every(void) {
    return bn_gpu_policy_moe_route_profile_every_or_default(28);
}

int bn_transformer_gpu_profile_level(void) {
    return bn_gpu_policy_profile_level();
}

int bn_transformer_gpu_debug_fallback_enabled(void) {
    return bn_gpu_policy_debug_fallback_enabled();
}

void bn_transformer_gpu_report_fallback(const char *reason) {
    if (!bn_transformer_gpu_debug_fallback_enabled())
        return;
    fprintf(stderr, "[gpu:fallback] %s\n", reason ? reason : "unknown");
}

float *bn_transformer_gpu_reject_forward(
    BnTransformerGPUEmitContext *emit,
    const char *reason) {
    bn_transformer_gpu_report_fallback(reason);
    bn_transformer_gpu_emit_context_free(emit);
    return NULL;
}

int bn_transformer_gpu_validate_forward(
    BnTransformerGPUForwardPolicy *out,
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnWeights *w,
    int token,
    int pos,
    const char **reject_reason) {
    *out = (BnTransformerGPUForwardPolicy){0};
    if (reject_reason)
        *reject_reason = NULL;
#define GPU_POLICY_REJECT(msg) do { \
        if (reject_reason) *reject_reason = (msg); \
        return -1; \
    } while (0)

    if (!gpu)
        GPU_POLICY_REJECT("backend missing");
    if (!gpu->execute)
        GPU_POLICY_REJECT("backend missing execute");
    if (!gpu->write_activation)
        GPU_POLICY_REJECT("backend missing write_activation");

    if (token < 0 || token >= c->vocab_size)
        GPU_POLICY_REJECT("token out of bounds");
    if (pos < 0)
        GPU_POLICY_REJECT("negative position");

    static const BnGPUBackend *cached_gpu = NULL;
    static const BnBackendModel *cached_backend = NULL;
    static const BnConfig *cached_config = NULL;
    static const BnWeights *cached_weights = NULL;
    static BnTransformerGPUForwardPolicy cached_policy;
    static int cached_valid = 0;
    if (cached_valid && cached_gpu == gpu && cached_backend == backend &&
        cached_config == c && cached_weights == w) {
        *out = cached_policy;
        return 0;
    }

    int cuda_large_native = bn_transformer_gpu_backend_is_cuda(gpu);
    if (!bn_gpu_policy_force_graph_enabled() && !cuda_large_native &&
        bn_model_arch_uses_large_gpu_graph_fallback_shape(c))
        GPU_POLICY_REJECT("large arch/hybrid/moe gpu graph disabled");
    if (bn_transformer_gpu_requires_layerwise_rope(c, w) &&
        !bn_transformer_gpu_can_layerwise_rope(gpu))
        GPU_POLICY_REJECT("layerwise rope unsupported by gpu backend");

    if (bn_transformer_gpu_backend_is_cuda(gpu) &&
        bn_model_arch_uses_small_cuda_dense_shape(c)) {
        if (bn_gpu_policy_cuda_small_kquant_native_disabled()) {
            if (!small_dense_cuda_q8_native_by_default(c, w))
                GPU_POLICY_REJECT("small dense cuda graph disabled");
        } else if (!small_dense_cuda_native_by_default(c, w)) {
            GPU_POLICY_REJECT("small dense cuda graph unsupported");
        }
    }

    if (c->dim > BN_TRANSFORMER_GPU_MAX_VLA_ELEMS)
        GPU_POLICY_REJECT("dim exceeds VLA limit");

    out->output_norm = bn_transformer_gpu_resolve_output_norm(backend);
    if (!out->output_norm)
        GPU_POLICY_REJECT("output norm not uploaded");

    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        BnTransformerGPULayerValidationResources layer_res =
            bn_transformer_gpu_resolve_layer_validation_resources(backend, l);
        int is_attn = bn_transformer_is_attn_layer(c, l);
        if (!is_attn) {
            out->has_ssm = 1;
            continue;
        }
        if (lw->moe.router_weight)
            out->has_moe = 1;
        if (!lw->attn.wq.data && !lw->ssm.wqkv.data)
            GPU_POLICY_REJECT("attention layer has no wq/wqkv data");
        if (lw->attn.q_norm && !layer_res.q_norm)
            GPU_POLICY_REJECT("q norm not uploaded");
        if (lw->attn.k_norm && !layer_res.k_norm)
            GPU_POLICY_REJECT("k norm not uploaded");
        if (lw->norm.attn_sub_norm && !layer_res.attn_sub_norm)
            GPU_POLICY_REJECT("attention sub norm not uploaded");
        if (lw->norm.ffn_sub_norm && !layer_res.ffn_sub_norm)
            GPU_POLICY_REJECT("ffn sub norm not uploaded");
        if (!layer_res.attn_norm || !layer_res.ffn_norm)
            GPU_POLICY_REJECT("layer norm not uploaded");
    }

    if (out->has_moe &&
        (!bn_transformer_gpu_backend_is_cuda(gpu) ||
         bn_transformer_gpu_cuda_moe_ffn_disabled()))
        GPU_POLICY_REJECT("moe gpu-resident forward unsupported");
    if (out->has_moe &&
        bn_transformer_gpu_backend_is_cuda(gpu) &&
        cuda_all2_q4q6_moe_requires_opt_in(c, w))
        GPU_POLICY_REJECT("all2 q4/q6 moe gpu-resident forward requires opt-in");
    if (out->has_ssm && (!gpu->read_activation || !gpu->write_activation))
        GPU_POLICY_REJECT("ssm needs read/write activation");

    bn_transformer_gpu_resolve_logit_resources(&out->logits, backend, c, w);
    if (!out->logits.gpu_buf)
        GPU_POLICY_REJECT("logit weight not uploaded");

    cached_gpu = gpu;
    cached_backend = backend;
    cached_config = c;
    cached_weights = w;
    cached_policy = *out;
    cached_valid = 1;
    return 0;
#undef GPU_POLICY_REJECT
}
