#include "gpu_internal.h"
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

static int small_dense_cuda_qweight_supported(int type) {
    return type == BN_GGUF_TENSOR_F32 || type == BN_GGUF_TENSOR_F16 ||
           type == BN_GGUF_TENSOR_Q8_0 || type == BN_GGUF_TENSOR_Q4_0 ||
           type == BN_GGUF_TENSOR_Q5_0 || type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K || type == BN_GGUF_TENSOR_Q6_K ||
           type == BN_GGUF_TENSOR_Q8_K;
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
    return lw->moe.expert_map.gate_type == BN_GGUF_TENSOR_Q4_K &&
           lw->moe.expert_map.up_type == BN_GGUF_TENSOR_Q4_K &&
           lw->moe.expert_map.down_type == BN_GGUF_TENSOR_Q6_K;
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
        if (!small_dense_cuda_qweight_supported(w->output_weight.type))
            return 0;
    } else if (!small_dense_cuda_qweight_supported(w->emb_type)) {
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
                !small_dense_cuda_qweight_supported(weights[i]->type))
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
        if (w->output_weight.type != BN_GGUF_TENSOR_Q8_0)
            return 0;
    } else if (w->emb_type != BN_GGUF_TENSOR_Q8_0) {
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
            if (weights[i]->data && weights[i]->type != BN_GGUF_TENSOR_Q8_0)
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

int bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_safe_default(
    const BnConfig *c,
    const BnWeights *w) {
    return bn_model_arch_allows_small_cuda_dense_exact_q4_q8(c) &&
           small_dense_cuda_q8_native_by_default(c, w) &&
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE") == NULL;
}

int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int q4_q8_from_layer) {
    return q4_q8_from_layer < 0 &&
           gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           bn_model_arch_allows_small_cuda_dense_exact_q4_q8(c) &&
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8") == NULL;
}

int bn_transformer_gpu_cuda_small_dense_exact_q4_q8_ffn_down_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
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

int bn_transformer_gpu_cuda_small_dense_prefill_decode_fallback_requested(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           bn_model_arch_allows_small_cuda_prefill_decode_fallback(c) &&
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_PREFILL") != NULL;
}

int bn_transformer_gpu_cuda_large_hybrid_prefill_decode_fallback_default(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           c && c->n_experts <= 0 &&
           c->full_attn_interval > 0 &&
           c->ssm_inner_size > 0 &&
           c->dim >= 4096 &&
           getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL") == NULL;
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
        return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
               gpu->prefill_ssm_layer &&
               getenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN") == NULL &&
               getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER") == NULL;
    }
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA && c->n_experts > 0)
        return getenv("BN_CUDA_ENABLE_MOE_PREFILL") != NULL;
    if (c->n_experts > 0)
        return 0;
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA)
        return c->dim <= 8192;
    return c->dim <= 2560;
}

int bn_transformer_gpu_cuda_large_hybrid_cpu_attn_fallback_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c) {
    if (!c || !gpu || gpu->kind != BN_GPU_BACKEND_CUDA ||
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
    return c && gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
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
    if (!c || !gpu || gpu->kind != BN_GPU_BACKEND_CUDA)
        return 0;
    if (getenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV"))
        return 0;
    if ((gpu_cpu_decode_fallback_requested() ||
         bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_safe_default(
             c, w) ||
         bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_safe_default(c, w) ||
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
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA && c) {
        int arch_min = bn_model_arch_small_cuda_dense_prefill_min_tokens(c);
        if (arch_min > 0)
            return arch_min;
    }
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA && c)
        return 16;
    return bn_transformer_gpu_cuda_prefill_attention_min_tokens();
}

int bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(
    const BnConfig *c,
    const BnGPUBackend *gpu) {
    const char *env = getenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    if (env && *env) {
        int n = atoi(env);
        return n > 0 ? n : 1;
    }
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA && c)
        return 16;
    return bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(c, gpu);
}

int bn_transformer_gpu_cuda_small_dense_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    int tensor_type) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           tensor_type == BN_GGUF_TENSOR_Q8_0 &&
           bn_model_arch_allows_small_cuda_q8_logit_refine(c) &&
           getenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE") != NULL &&
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE") == NULL;
}

int bn_transformer_gpu_cuda_all2_q4q6_moe_q6_logits_refine_default(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           bn_transformer_gpu_cuda_all2_q4q6_moe_model(c, w) &&
           getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") != NULL &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE") == NULL;
}

int bn_transformer_gpu_q6_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q6_refine_default) {
    int cuda_backend = gpu && gpu->kind == BN_GPU_BACKEND_CUDA;
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
           logits->type == BN_GGUF_TENSOR_Q6_K &&
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
    int cuda_backend = gpu && gpu->kind == BN_GPU_BACKEND_CUDA;
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
           logits->type == BN_GGUF_TENSOR_Q8_0 &&
           logits->cpu_weight != NULL;
}

int bn_transformer_gpu_q8_logits_refine_top(int q8_refine_default) {
    int refine_top = q8_refine_default ? 16 : 8;
    const char *env = getenv("BN_GPU_Q8_REFINE_TOP");
    if (env)
        refine_top = atoi(env);
    return refine_top;
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
        getenv("BN_GPU_CPU_LOGITS") != NULL ||
        gpu_logits_need_cpu ||
        getenv("BN_CUDA_DISABLE_LOGITS_ARGMAX") != NULL ||
        logits->type != BN_GGUF_TENSOR_Q6_K)
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
    if (!c || !w || !backend || c->n_experts <= 0)
        return 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        if (!lw->moe.router_weight)
            continue;
        const BnMoEExpertMap *em = &lw->moe.expert_map;
        int routed_q4 = em->gate_type == BN_GGUF_TENSOR_Q4_K &&
                        em->up_type == BN_GGUF_TENSOR_Q4_K &&
                        (em->down_type == BN_GGUF_TENSOR_Q6_K ||
                         em->down_type == BN_GGUF_TENSOR_Q4_K);
        int routed_q8 = em->gate_type == BN_GGUF_TENSOR_Q8_0 &&
                        em->up_type == BN_GGUF_TENSOR_Q8_0 &&
                        em->down_type == BN_GGUF_TENSOR_Q8_0;
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
    if (!gpu || gpu->kind != BN_GPU_BACKEND_CUDA)
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
        getenv("BN_GPU_CPU_LOGITS") != NULL ||
        getenv("BN_GPU_COMPARE_LOGITS") != NULL ||
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
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           bn_model_arch_moe_prefers_cuda_exact_attention(c) &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN") == NULL;
}

int bn_transformer_gpu_ssm_cpu_fallback_required(
    const BnGPUBackend *gpu) {
    return !gpu || gpu->kind != BN_GPU_BACKEND_CUDA ||
           getenv("BN_CUDA_DISABLE_SSM_GRAPH") != NULL;
}

int bn_transformer_gpu_cuda_large_hybrid_argmax_blocked(
    const BnGPUBackend *gpu,
    const BnConfig *c,
    const BnWeights *w,
    int want_argmax) {
    return want_argmax &&
           gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_default(c, w) &&
           getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX") == NULL;
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
        map->gate_type != BN_GGUF_TENSOR_Q4_K ||
        map->up_type != BN_GGUF_TENSOR_Q4_K ||
        getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") != NULL)
        return 0;
    return map->down_type == BN_GGUF_TENSOR_Q6_K ||
           (allow_q4_down && map->down_type == BN_GGUF_TENSOR_Q4_K);
}

int bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed(int n_experts) {
    if (getenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH"))
        return 0;
    return n_experts <= 2 ||
           getenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE") != NULL;
}

int bn_transformer_gpu_cuda_moe_gateup_split_enabled(
    const BnGPUBackend *gpu,
    int can_split) {
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA && can_split &&
           getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT") == NULL;
}

void bn_transformer_gpu_report_fallback(const char *reason) {
    if (!getenv("BN_GPU_DEBUG_FALLBACK"))
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

    int cuda_large_native = gpu->kind == BN_GPU_BACKEND_CUDA;
    if (!getenv("BN_GPU_FORCE_GRAPH") && c->dim >= 4096 &&
        !cuda_large_native &&
        (bn_model_arch_requires_large_gpu_graph_fallback(c) ||
         c->full_attn_interval > 0 ||
         c->n_experts > 0))
        GPU_POLICY_REJECT("large arch/hybrid/moe gpu graph disabled");

    if (gpu->kind == BN_GPU_BACKEND_CUDA && c->dim <= 2560 &&
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
        (gpu->kind != BN_GPU_BACKEND_CUDA ||
         getenv("BN_CUDA_DISABLE_MOE_FFN") != NULL))
        GPU_POLICY_REJECT("moe gpu-resident forward unsupported");
    if (out->has_moe &&
        gpu->kind == BN_GPU_BACKEND_CUDA &&
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
