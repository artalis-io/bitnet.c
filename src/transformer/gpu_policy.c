#include "gpu_internal.h"
#include "gpu_policy.h"
#include "../gpu_shader_ir_internal.h"
#include "backend_quant.h"
#include "model_arch.h"
#include <stdio.h>
#include <stdlib.h>

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
    return getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER") != NULL;
}

int bn_transformer_gpu_fused_gateup_silu_policy_allows(
    const BnGPUBackend *gpu,
    int tensor_type) {
    if (getenv("BN_GPU_DISABLE_FUSED_GATEUP") != NULL)
        return 0;
    if (bn_transformer_gpu_backend_is_cuda(gpu) &&
        bn_backend_quant_gpu_fused_gateup_requires_cuda_opt_in(tensor_type) &&
        getenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP") == NULL)
        return 0;
    return 1;
}

int bn_transformer_gpu_q4_q8_fused_gateup_enabled(int use_q4_q8) {
    return use_q4_q8 && getenv("BN_GPU_Q4_Q8_DISABLE_GATEUP") == NULL;
}

int bn_transformer_gpu_gateup_split_enabled(void) {
    return getenv("BN_GPU_DISABLE_GATEUP_SPLIT") == NULL;
}

int bn_transformer_gpu_q4_q8_ffn_down_enabled(int use_q4_q8_down) {
    return use_q4_q8_down &&
           getenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN") == NULL;
}

int bn_transformer_gpu_qkv_split_enabled(int use_q4_q8) {
    return !use_q4_q8 && getenv("BN_GPU_DISABLE_QKV_SPLIT") == NULL;
}

int bn_transformer_gpu_qk_split_enabled(void) {
    return getenv("BN_GPU_DISABLE_QKV_SPLIT") == NULL;
}

int bn_transformer_gpu_qkv_split_debug_enabled(void) {
    return getenv("BN_GPU_DEBUG_QKV_SPLIT") != NULL;
}

int bn_transformer_gpu_ssm_qkvz_split_enabled(void) {
    return getenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT") == NULL;
}

int bn_transformer_gpu_ssm_ab_stack_enabled(void) {
    return getenv("BN_GPU_DISABLE_SSM_AB_STACK") == NULL;
}

int bn_transformer_gpu_split_residual_rmsnorm_enabled(void) {
    return getenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM") != NULL;
}

int bn_transformer_gpu_shared_q4_q8_dot_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT") == NULL;
}

int bn_transformer_gpu_shared_expert_gate_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE") == NULL;
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
    if (!map || split_op_code != BN_GPU_CODE_Q4K_MATVEC_SPLIT)
        return 0;
    return bn_transformer_gpu_can_matvec_split(gpu, map->gate_type) &&
           map->up_type == map->gate_type &&
           map->gate_rows == map->up_rows &&
           map->gate_cols == map->up_cols;
}

int bn_transformer_gpu_logits_needs_cpu_fallback(
    const BnGPUBackend *gpu,
    const BnTransformerGPULogitResources *logits) {
    if (!gpu || !logits || !logits->cpu_weight)
        return 0;

    size_t max_storage_binding = gpu->max_storage_binding_size;
    if (max_storage_binding == 0)
        max_storage_binding = 128ull * 1024ull * 1024ull;
    const char *override_mb = getenv("BN_GPU_MAX_STORAGE_BINDING_MB");
    if (override_mb) {
        long mb = strtol(override_mb, NULL, 10);
        if (mb >= 0)
            max_storage_binding = (size_t)mb * 1024ull * 1024ull;
    }

    return bn_qweight_data_size(logits->cpu_weight) > max_storage_binding;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_layer(
    const BnConfig *c,
    const BnLayerWeights *lw,
    int dim) {
    if (!c || !lw ||
        c->n_experts != 2 ||
        c->n_experts_active != 2 ||
        c->moe_intermediate_size < 4096 ||
        dim > 2048)
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
    if (!c || !w || c->n_experts != 2 ||
        c->n_experts_active != 2 ||
        c->moe_intermediate_size < 4096 ||
        c->dim > 2048)
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
           getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") == NULL &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE") != NULL;
}

static int small_dense_cuda_native_by_default(
    const BnConfig *c,
    const BnWeights *w) {
    if (!c || !w || c->n_experts > 0 || c->full_attn_interval > 0 ||
        c->dim > 2560)
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
    if (!c || !w || c->n_experts > 0 || c->full_attn_interval > 0 ||
        c->dim > 2560)
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
           getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") == NULL &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE") == NULL;
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
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE") == NULL;
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
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8") == NULL;
}

int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_ffn_down_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_allows_small_cuda_dense_exact_q4_q8(c) &&
           getenv("BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN") != NULL;
}

int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w) {
    if (!c || !w || c->n_experts > 0 || c->dim < 4096 ||
        getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN") != NULL ||
        getenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL)
        return 0;
    if (getenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE") == NULL &&
        getenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE") == NULL)
        return 0;
    if (c->full_attn_interval > 0)
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
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_PREFILL") != NULL;
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
    return c && c->full_attn_interval > 0 && c->ssm_inner_size > 0 &&
           bn_transformer_gpu_backend_is_cuda(gpu);
}

int bn_transformer_gpu_moe_prefill_chain_applicable(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return c && c->n_experts > 0 && c->full_attn_interval <= 0 &&
           bn_transformer_gpu_backend_is_cuda(gpu);
}

int bn_transformer_gpu_cuda_large_hybrid_prefill_decode_fallback_default(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           c && c->n_experts <= 0 &&
           c->full_attn_interval > 0 &&
           c->ssm_inner_size > 0 &&
           c->dim >= 4096 &&
           getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL") == NULL;
}

int bn_transformer_gpu_cuda_matvec_fallback_kept(
    const BnModel *m,
    const BnGPUBackend *gpu) {
    if (!m || !bn_transformer_gpu_backend_is_cuda(gpu) || !gpu->execute)
        return 0;
    const BnConfig *c = &m->config;
    if (c->n_experts > 0 || c->full_attn_interval > 0)
        return 0;
    if (getenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE"))
        return 1;
    if (getenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE") == NULL &&
        !bn_model_arch_cpu_force_float_kquant(c))
        return 1;
    if (c->dim > 2560 || c->dim <= 1024)
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

int bn_transformer_gpu_batch_prefill_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    if (!c)
        return 0;
    if (getenv("BN_GPU_DISABLE_PREFILL_MATMUL"))
        return 0;
    if (getenv("BN_GPU_PREFILL_MATMUL"))
        return 1;
    if (c->kv_tq_bits != 0)
        return 0;
    if (bn_transformer_gpu_cuda_small_dense_prefill_decode_fallback_requested(
            gpu, c) ||
        bn_transformer_gpu_cuda_large_hybrid_prefill_decode_fallback_default(
            gpu, c))
        return 0;
    if (c->full_attn_interval > 0) {
        return bn_transformer_gpu_backend_is_cuda(gpu) &&
               gpu->prefill_ssm_layer &&
               getenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN") == NULL &&
               !bn_transformer_gpu_cuda_prefill_ssm_layer_disabled();
    }
    if (bn_transformer_gpu_backend_is_cuda(gpu) && c->n_experts > 0)
        return getenv("BN_CUDA_ENABLE_MOE_PREFILL") != NULL;
    if (c->n_experts > 0)
        return 0;
    if (bn_transformer_gpu_backend_is_cuda(gpu))
        return c->dim <= 8192;
    return c->dim <= 2560;
}

int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    if (!c || !bn_transformer_gpu_backend_is_cuda(gpu) ||
        c->n_experts > 0 || c->dim < 4096 ||
        c->full_attn_interval <= 0 || c->ssm_inner_size <= 0)
        return 0;
    if (getenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL)
        return 1;
    return getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN") == NULL &&
           getenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE") == NULL &&
           getenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE") != NULL;
}

int bn_transformer_gpu_cuda_large_hybrid_prefill_chain_disabled_default(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return c && bn_transformer_gpu_backend_is_cuda(gpu) &&
           c->n_experts <= 0 && c->dim >= 4096 &&
           c->full_attn_interval > 0 && c->ssm_inner_size > 0 &&
           getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN") == NULL;
}

static int gpu_cpu_decode_fallback_requested(void) {
    return getenv("BN_GPU_CPU_FALLBACK_LAYER") ||
           getenv("BN_GPU_CPU_FALLBACK_FROM_LAYER") ||
           getenv("BN_GPU_CPU_ATTN_LAYER") ||
           getenv("BN_GPU_CPU_ATTN_FROM_LAYER");
}

int bn_transformer_gpu_cuda_prefill_direct_kv_allowed(
    const BnConfig *c,
    const BnWeights *w,
    const BnGPUBackend *gpu,
    int pos0,
    int n_tokens) {
    if (!c || !bn_transformer_gpu_backend_is_cuda(gpu))
        return 0;
    if (getenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV"))
        return 0;
    if ((gpu_cpu_decode_fallback_requested() ||
         bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_fallback_enabled(
             gpu, c, w) ||
         bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_fallback_enabled(
             gpu, c, w) ||
         bn_transformer_gpu_cuda_large_hybrid_cpu_attn_fallback_enabled(
             gpu, c)) &&
        !getenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK"))
        return 0;
    if (c->kv_f16 || pos0 < 0 || pos0 + n_tokens > c->seq_len)
        return 0;
    return 1;
}

int bn_transformer_gpu_cuda_prefill_attention_min_tokens(void) {
    const char *env = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    if (!env || !*env)
        return 16;
    int n = atoi(env);
    return n > 0 ? n : 16;
}

int bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    const char *env = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    if (env && *env)
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
    const char *env = getenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    if (env && *env) {
        int n = atoi(env);
        return n > 0 ? n : 1;
    }
    if (bn_transformer_gpu_backend_is_cuda(gpu) && c)
        return 16;
    return bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_cuda_prefill_moe_ffn_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnMoEExpertMap *map,
    int dim,
    int allow_q4_down) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           gpu->moe_route_routed_ffn_batch_norm_resid &&
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(
               c ? c->n_experts : 0) &&
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
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(
               c ? c->n_experts : 0) &&
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
    return getenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN") == NULL;
}

int bn_transformer_gpu_cuda_prefill_hybrid_chain_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return getenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN") == NULL &&
           !bn_transformer_gpu_cuda_large_hybrid_prefill_chain_disabled_default(
               gpu, c);
}

int bn_transformer_gpu_cuda_prefill_attention_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_ATTN") == NULL;
}

int bn_transformer_gpu_cuda_prefill_ssm_run_chain_enabled(void) {
    return getenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN") == NULL;
}

int bn_transformer_gpu_cuda_prefill_ssm_ffn_fuse_allowed(void) {
    return getenv("BN_CUDA_DISABLE_SSM_FFN_FUSE") == NULL;
}

int bn_transformer_gpu_cuda_prefill_moe_chain_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN") != NULL;
}

int bn_transformer_gpu_cuda_prefill_hybrid_chain_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN") != NULL;
}

int bn_transformer_gpu_cuda_moe_prefill_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_PREFILL") != NULL;
}

int bn_transformer_gpu_cuda_moe_prefill_min_tokens(void) {
    const char *env = getenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    if (!env || !*env)
        return 1;
    int n = atoi(env);
    return n > 0 ? n : 1;
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
    return getenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL") == NULL;
}

int bn_transformer_gpu_cuda_moe_prefill_shared_fuse_enabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE") == NULL;
}

int bn_transformer_gpu_cuda_moe_route_batch_debug_enabled(void) {
    return getenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH") != NULL;
}

int bn_transformer_gpu_moe_prefill_route_batch_available(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return c && c->n_experts > 2 &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           gpu->moe_route_batch;
}

int bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return c &&
           bn_transformer_gpu_moe_prefill_backend_available(gpu) &&
           gpu->moe_route_routed_ffn_batch_norm_resid &&
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(c->n_experts);
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
           bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(c->n_experts) &&
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

int bn_transformer_gpu_cuda_moe_lazy_aux_cache_enabled(void) {
    return getenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE") != NULL;
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
    return getenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL") != NULL;
}

int bn_transformer_gpu_cuda_small_dense_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int tensor_type) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_backend_quant_supports_q8_logits_refine(tensor_type) &&
           bn_model_arch_allows_small_cuda_q8_logit_refine(c) &&
           getenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE") != NULL &&
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE") == NULL;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_q6_logits_refine_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_all2_q4q6_moe_model(c, w) &&
           getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") != NULL &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE") == NULL;
}

int bn_transformer_gpu_q6_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q6_refine_default) {
    int cuda_backend = bn_transformer_gpu_backend_is_cuda(gpu);
    return q6_refine_default ||
           getenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE") != NULL ||
           (!cuda_backend &&
            getenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE") == NULL);
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
    int refine_top = q6_refine_default ? 64 : 8;
    const char *env = getenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    if (env)
        refine_top = atoi(env);
    return refine_top;
}

int bn_transformer_gpu_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q8_refine_default) {
    int cuda_backend = bn_transformer_gpu_backend_is_cuda(gpu);
    return getenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE") != NULL ||
           q8_refine_default ||
           (!cuda_backend &&
            getenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE") == NULL);
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
    int refine_top = q8_refine_default ? 16 : 8;
    const char *env = getenv("BN_GPU_Q8_REFINE_TOP");
    if (env)
        refine_top = atoi(env);
    return refine_top;
}

int bn_transformer_gpu_cpu_logits_enabled(int gpu_logits_need_cpu) {
    return gpu_logits_need_cpu || getenv("BN_GPU_CPU_LOGITS") != NULL;
}

int bn_transformer_gpu_compare_logits_enabled(void) {
    return getenv("BN_GPU_COMPARE_LOGITS") != NULL;
}

int bn_transformer_gpu_debug_argmax_compare_enabled(void) {
    return getenv("BN_GPU_DEBUG_ARGMAX_COMPARE") != NULL;
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
        getenv("BN_CUDA_DISABLE_LOGITS_ARGMAX") != NULL ||
        !bn_backend_quant_supports_q6k_logits_refine(logits->type))
        return 0;

    if (c->n_experts <= 0) {
        return logits->rows > 262144 ||
               getenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX") != NULL;
    }
    if (c->n_experts == 2 && c->n_experts_active == 2)
        return 1;
    if (getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX") != NULL)
        return 1;
    return logits->cols == 1536 &&
           getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX") == NULL;
}

int bn_transformer_gpu_cuda_moe_decode_cacheable(
    const BnConfig *c,
    const BnWeights *w,
    const BnBackendModel *backend) {
    if (getenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE") != NULL ||
        !c || !w || !backend || c->n_experts <= 0)
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
         (getenv("BN_CUDA_ENABLE_LOGITS_CACHE") &&
          !gpu_logits_need_cpu)) == 0)
        return 0;
    if (!bn_transformer_gpu_backend_is_cuda(gpu))
        return 0;
    if (has_moe && !cacheable_resident_moe &&
        getenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE") == NULL)
        return 0;
    if (getenv("BN_CUDA_DISABLE_DECODE_CACHE") != NULL)
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
    if (getenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE") != NULL ||
        bn_transformer_gpu_cpu_logits_enabled(gpu_logits_need_cpu) ||
        bn_transformer_gpu_compare_logits_enabled() ||
        getenv("BN_METAL_ENABLE_Q6_Q8K") != NULL)
        return 0;
    return 1;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_moe_safe_default(
    const BnConfig *c,
    const BnWeights *w) {
    return bn_transformer_gpu_cuda_all2_q4q6_moe_model(c, w) &&
           getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") == NULL &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE") == NULL;
}

int bn_transformer_gpu_cuda_moe_exact_attention_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_model_arch_moe_prefers_cuda_exact_attention(c) &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN") == NULL;
}

int bn_transformer_gpu_ssm_cpu_fallback_required(
    const BnGPUBackend *gpu) {
    return !bn_transformer_gpu_backend_is_cuda(gpu) ||
           getenv("BN_CUDA_DISABLE_SSM_GRAPH") != NULL;
}

int bn_transformer_gpu_cuda_large_hybrid_argmax_blocked(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax) {
    return want_argmax &&
           bn_transformer_gpu_backend_is_cuda(gpu) &&
           bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_default(c, w) &&
           getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX") == NULL;
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

BnTransformerGPUQ4Q8LayerPolicy
bn_transformer_gpu_q4_q8_layer_policy(const BnConfig *c) {
    int n_layers = c ? c->n_layers : 0;
    BnTransformerGPUQ4Q8LayerPolicy policy = {
        .from_layer = -1,
        .to_layer = -1,
        .attn_only = getenv("BN_GPU_Q4_Q8_ATTN_ONLY") != NULL,
        .ffn_only = getenv("BN_GPU_Q4_Q8_FFN_ONLY") != NULL,
    };

    const char *env = getenv("BN_GPU_Q4_Q8_FROM_LAYER");
    if (env) {
        policy.from_layer = atoi(env);
    } else if (getenv("BN_GPU_Q4_Q8")) {
        policy.from_layer = n_layers - 1;
    }

    env = getenv("BN_GPU_Q4_Q8_TO_LAYER");
    if (env) {
        policy.to_layer = atoi(env);
    } else {
        env = getenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
        if (env) {
            int tail_native = atoi(env);
            if (tail_native > 0) {
                policy.to_layer = n_layers - tail_native - 1;
                if (policy.to_layer < -1)
                    policy.to_layer = -1;
            }
        } else if (getenv("BN_GPU_Q4_Q8") &&
                   getenv("BN_METAL_Q4_PREPARED") == NULL &&
                   n_layers > 33) {
            policy.to_layer = n_layers - 33 - 1;
        }
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
    int flash_min_kv = 0;
    const char *flash_min_env = getenv("BN_GPU_FLASH_MIN_KV");
    if (flash_min_env) flash_min_kv = atoi(flash_min_env);
    int flash_max_kv = 0;
    const char *flash_max_env = getenv("BN_GPU_FLASH_MAX_KV");
    if (flash_max_env)
        flash_max_kv = atoi(flash_max_env);
    else if (bn_transformer_gpu_backend_is_cuda(gpu))
        flash_max_kv = 2048;

    return bn_transformer_gpu_can_flash_attn(gpu) &&
           (has_moe || config_flash_attn ||
            bn_transformer_gpu_backend_is_cuda(gpu)) &&
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
    return moe_router &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK") == NULL &&
           (!all2_q4q6_moe || all2_q4q6_moe_gpu_route_layer_selected);
}

int bn_transformer_gpu_cuda_moe_cpu_route_resident_ffn_enabled(
    int all2_q4q6_moe,
    int gpu_route_topk,
    int moe_routed_q8,
    int n_experts) {
    if (all2_q4q6_moe && !gpu_route_topk &&
        getenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT") == NULL)
        return 1;
    return !gpu_route_topk && moe_routed_q8 && n_experts > 2 &&
           getenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT") == NULL;
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
        getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN") != NULL)
        return 0;
    return map->gate_rows == moe_hidden &&
           map->up_rows == moe_hidden &&
           map->gate_cols == dim &&
           map->up_cols == dim &&
           map->down_rows == dim &&
           map->down_cols == moe_hidden;
}

int bn_transformer_gpu_cuda_all2_moe_direct_route_enabled(
    const BnConfig *c,
    void *router_diff,
    void *moe_gate_all) {
    return router_diff &&
           c && c->n_experts == 2 &&
           c->n_experts_active == 2 &&
           c->moe_norm_topk_prob &&
           !moe_gate_all &&
           getenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU") == NULL;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_selected(
    int layer,
    int route_from_layer,
    int route_to_layer) {
    if (getenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU") == NULL &&
        getenv("BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE") == NULL)
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

    env = getenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
    if (env && route_from_layer)
        *route_from_layer = atoi(env);
    env = getenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
    if (env && route_to_layer)
        *route_to_layer = atoi(env);
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_exact_gpu_route_enabled(
    int all2_q4q6_moe,
    int route_layer_selected) {
    return all2_q4q6_moe &&
           route_layer_selected &&
           getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") != NULL &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE") == NULL;
}

void *bn_transformer_gpu_cuda_all2_q4q6_moe_router(
    const BnConfig *c,
    void *moe_router,
    void *router_diff,
    int route_layer_selected,
    int exact_gpu_route) {
    if (router_diff &&
        c && c->n_experts == 2 &&
        c->n_experts_active == 2 &&
        route_layer_selected &&
        getenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2") == NULL &&
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
        c->n_experts != 2 ||
        c->n_experts_active != 2 ||
        c->moe_intermediate_size < 4096 ||
        dim > 2048 ||
        !bn_transformer_gpu_moe_routed_q4_down(map, allow_q4_down) ||
        getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") != NULL)
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
    return cpu_fallback_ffn_layer == layer ||
           (cpu_fallback_ffn_from_layer >= 0 &&
            layer >= cpu_fallback_ffn_from_layer);
}

int bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(int n_experts) {
    if (getenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH"))
        return 0;
    return n_experts <= 2 ||
           getenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE") != NULL;
}

int bn_transformer_gpu_cuda_moe_ffn_disabled(void) {
    return getenv("BN_CUDA_DISABLE_MOE_FFN") != NULL;
}

int bn_transformer_gpu_cuda_moe_cpu_actual_override_enabled(int safe_default) {
    return safe_default ||
           getenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL") != NULL;
}

int bn_transformer_gpu_moe_compare_layer_selected(int layer, int pos) {
    const char *compare_moe_env = getenv("BN_GPU_COMPARE_MOE_LAYER");
    if (!compare_moe_env)
        return 0;
    int compare_layer = atoi(compare_moe_env);
    const char *compare_pos_env = getenv("BN_GPU_COMPARE_MOE_POS");
    int compare_pos = compare_pos_env ? atoi(compare_pos_env) : -1;
    return compare_layer == layer && (compare_pos < 0 || compare_pos == pos);
}

int bn_transformer_gpu_moe_compare_input_norm_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_INPUT_NORM") != NULL;
}

int bn_transformer_gpu_moe_compare_actual_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_ACTUAL") != NULL;
}

int bn_transformer_gpu_moe_compare_route_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_ROUTE") != NULL;
}

int bn_transformer_gpu_moe_compare_raw_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_RAW") != NULL;
}

int bn_transformer_gpu_moe_compare_mid_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_MID") != NULL;
}

int bn_transformer_gpu_moe_compare_parts_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_PARTS") != NULL;
}

int bn_transformer_gpu_moe_compare_shared_mid_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_SHARED_MID") != NULL;
}

int bn_transformer_gpu_moe_compare_shared_down_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_SHARED_DOWN") != NULL;
}

int bn_transformer_gpu_moe_compare_norm_enabled(void) {
    return getenv("BN_GPU_COMPARE_MOE_NORM") != NULL;
}

int bn_transformer_gpu_cuda_moe_shared_cpu_fallback_enabled(int eligible) {
    return eligible &&
           getenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK") != NULL &&
           getenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK") == NULL;
}

int bn_transformer_gpu_cuda_moe_gateup_split_enabled(
    const BnGPUBackend *gpu,
    int can_split) {
    return bn_transformer_gpu_backend_is_cuda(gpu) && can_split &&
           getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT") == NULL;
}

int bn_transformer_gpu_moe_route_profile_enabled(void) {
    return getenv("BN_GPU_MOE_ROUTE_PROFILE") != NULL;
}

int bn_transformer_gpu_moe_route_profile_every(void) {
    int every = 28;
    const char *env = getenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0)
            every = v;
    }
    return every;
}

int bn_transformer_gpu_profile_level(void) {
    return bn_gpu_policy_profile_level();
}

int bn_transformer_gpu_debug_fallback_enabled(void) {
    return getenv("BN_GPU_DEBUG_FALLBACK") != NULL;
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
    if (!getenv("BN_GPU_FORCE_GRAPH") && c->dim >= 4096 &&
        !cuda_large_native &&
        (bn_model_arch_requires_large_gpu_graph_fallback(c) ||
         c->full_attn_interval > 0 ||
         c->n_experts > 0))
        GPU_POLICY_REJECT("large arch/hybrid/moe gpu graph disabled");

    if (bn_transformer_gpu_backend_is_cuda(gpu) && c->dim <= 2560 &&
        c->n_experts <= 0 && c->full_attn_interval <= 0) {
        if (getenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE")) {
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
