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

static int cuda_all2_q4q6_moe_model(const BnConfig *c,
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
        if (lw->moe.expert_map.gate_type == BN_GGUF_TENSOR_Q4_K &&
            lw->moe.expert_map.up_type == BN_GGUF_TENSOR_Q4_K &&
            lw->moe.expert_map.down_type == BN_GGUF_TENSOR_Q6_K)
            return 1;
    }
    return 0;
}

static int cuda_all2_q4q6_moe_requires_opt_in(const BnConfig *c,
                                              const BnWeights *w) {
    return cuda_all2_q4q6_moe_model(c, w) &&
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
    return cuda_all2_q4q6_moe_model(c, w) &&
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
           cuda_all2_q4q6_moe_model(c, w) &&
           getenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN") != NULL &&
           getenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE") == NULL;
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
