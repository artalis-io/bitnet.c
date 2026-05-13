#include "gpu_internal.h"
#include "backend_session.h"
#include "session.h"
#include <string.h>

// GPU-resident forward pass: one submit per token, reads back logits only.
// Supports classic transformer only (no MoE, no SSM, no gated-Q, no wide-Q,
// no Q/K norms, no sub-norms, no FP16 KV cache).
// Supports attention biases (Qwen2.5) and tied embeddings (BitNet).
// Returns s->logits on success, NULL to fall back to CPU.
float *bn_transformer_gpu_forward(BnModel *m, BnSession *sess, int token, int pos) {
    /* no-op */
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    BnTransformerGPUEmitContext emit;
    bn_transformer_gpu_emit_context_init(&emit, NULL, 0);

#define GPU_REJECT(msg) do { \
        bn_transformer_gpu_report_fallback((msg)); \
        bn_transformer_gpu_emit_context_free(&emit); \
        return NULL; \
    } while (0)

    BnTransformerGPUForwardPolicy policy;
    const char *reject_reason = NULL;
    if (bn_transformer_gpu_validate_forward(
            &policy, gpu, backend, c, w, token, pos, &reject_reason) != 0)
        GPU_REJECT(reject_reason);

    int dim = c->dim;
    int kv_dim = c->kv_dim;
    int head_size = c->head_size;
    int n_heads = c->n_heads;
    int q_dim = n_heads * head_size;
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;

    // Embed token on CPU, upload to GPU x buffer
    float emb[dim];
    bn_model_embed_token(m, emb, token);
    if (bn_transformer_gpu_write_x(gpu, emb,
                                   (size_t)dim * sizeof(float)) != 0)
        GPU_REJECT("write token embedding failed");

    /* no-op */

    void *output_norm = policy.output_norm;
    BnTransformerGPULogitResources *logit_res = &policy.logits;
    int has_moe = policy.has_moe;

    // Precompute eps as uint32
    uint32_t u_eps;
    { float eps = c->norm_eps; memcpy(&u_eps, &eps, 4); }

    int max_ops = bn_transformer_gpu_graph_op_capacity(c);

    // Reuse the session-owned op array to avoid per-token malloc.
    BnGPUGraph *graph =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(sess->backend, max_ops);
    if (!graph) GPU_REJECT("gpu graph allocation failed");
    bn_transformer_gpu_emit_context_init(&emit, graph->ops, graph->cap);

    // Helper: flush current ops (no readback), reset counter
    #define GPU_FLUSH() do { \
        if (emit.n > 0 || emit.graph.n_ops > 0) { \
            if (bn_transformer_gpu_emit_context_execute(&emit, gpu, -1, NULL, 0) != 0) GPU_REJECT("gpu execute flush failed"); \
        } \
    } while(0)

    // ---- Initial RMSNorm: x -> xb (using layer 0 attn_norm) ----
    if (bn_transformer_gpu_emit_context_rmsnorm(
            &emit, bn_transformer_gpu_resolve_initial_norm(backend),
            BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps) != 0)
        GPU_REJECT("gpu graph rmsnorm emit failed");

    /* no-op */

    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, lw, l, bn_model_tq_state(m) != NULL);
        int is_attn = plan.is_attn;

        // ---- SSM layer: CPU fallback until the WebGPU SSM path is token-coherent ----
        if (!is_attn) {
            int use_cpu_ssm_fallback = 1;
            if (use_cpu_ssm_fallback) {
                void *nn = bn_transformer_gpu_resolve_next_norm(
                    backend, l, c->n_layers, output_norm);
                if (bn_transformer_gpu_fallback_ssm_layer(
                        &emit, gpu, m, sess, lw, l, dim, u_eps, nn) != 0)
                    GPU_REJECT("gpu ssm cpu fallback failed");
                continue;
            }

            BnTransformerGPUSSMResources ssm_res =
                bn_transformer_gpu_resolve_ssm_resources(gpu, backend, lw, l);
            bn_transformer_gpu_emit_context_ssm(
                &emit, c, lw, &plan, &ssm_res, dim, u_eps);

            // SSM layer's FFN (dense or MoE) — same as attention layer below
            goto ffn_block;
        }

        // KV cache addressing
        int attn_idx = plan.attn_idx;
        size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
        int cache_pos = pos % c->seq_len;
        int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;

        uint32_t kv_cache_off = (uint32_t)(loff + (size_t)cache_pos * kv_dim);
        BnTransformerGPUQKVResources qkv_res =
            bn_transformer_gpu_resolve_qkv_resources(gpu, backend, lw, l);
        bn_transformer_gpu_emit_context_qkv(
            &emit, c, lw, &plan, &qkv_res, pos, q_dim,
            head_size, n_heads, kv_dim, rope_dims, kv_cache_off, u_eps);
        BnTransformerGPUAttentionResources attn_res =
            bn_transformer_gpu_resolve_attention_resources(gpu, backend, lw, l);
        bn_transformer_gpu_emit_context_attention(
            &emit, c, lw, &attn_res, pos, dim, q_dim,
            head_size, n_heads, kv_dim, rope_dims, n_kv, loff, kv_cache_off,
            has_moe, u_eps);

        // ---- FFN (MoE or dense) ----
        ffn_block:;
        if (lw->moe.router_weight) {
            // MoE FFN: CPU fallback until the WebGPU MoE path is token-coherent
            int use_cpu_moe_fallback = 1;
            if (use_cpu_moe_fallback) {
                void *moe_next_norm = bn_transformer_gpu_resolve_next_norm(
                    backend, l, c->n_layers, output_norm);
                if (bn_transformer_gpu_fallback_moe_layer(
                        &emit, gpu, m, sess, lw, l, dim, u_eps,
                        moe_next_norm) != 0)
                    GPU_REJECT("gpu moe cpu fallback failed");
                continue;
            }

            void *uncached_bufs[BN_MAX_MOE_K * 3];
            int n_uncached = 0;
            void *next_norm = bn_transformer_gpu_resolve_next_norm(
                backend, l, c->n_layers, output_norm);
            BnGPUMoEResolvedExpert expert_emit[BN_MAX_MOE_K];
            BnGPUMoEResources moe_res;
            if (bn_gpu_moe_bridge_resolve_resources(
                    &moe_res, expert_emit, BN_MAX_MOE_K, m, sess, lw, l,
                    uncached_bufs, &n_uncached) != 0)
                GPU_REJECT("gpu moe resource resolution failed");
            BnTransformerGPUMoESharedResources moe_shared =
                bn_transformer_gpu_resolve_moe_shared_resources(backend, lw);
            bn_transformer_gpu_emit_context_moe(
                &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm);
            // If buffers are cached, let the next layer's attention join this
            // submission; the next CPU routing readback will flush it.  Uncached
            // buffers still require an immediate flush before destruction.
            if (n_uncached > 0) {
                GPU_FLUSH();
                for (int ub = 0; ub < n_uncached; ub++)
                    gpu->buffer_destroy(gpu->ctx, uncached_bufs[ub]);
            }
            continue;  // skip dense FFN below
        }
        void *next_norm = bn_transformer_gpu_resolve_next_norm(
            backend, l, c->n_layers, output_norm);
        BnFFNPlan ffn_plan;
        bn_transformer_plan_ffn(&ffn_plan, c, lw, gpu, backend, l, 1);
        BnTransformerGPUDenseFFNResources ffn_res =
            bn_transformer_gpu_resolve_dense_ffn_resources(gpu, backend, lw, l);
        bn_transformer_gpu_emit_context_dense_ffn(
            &emit, c, lw, &ffn_plan, &ffn_res, dim, u_eps,
            next_norm);
    }

    // ---- Logits matvec: xb -> logits (xb is already normalized) ----
    {
        if (bn_transformer_gpu_logits_needs_cpu_fallback(gpu, logit_res)) {
            if (bn_transformer_gpu_fallback_logits(
                    &emit, gpu, m, sess, logit_res, dim) != 0)
                GPU_REJECT("gpu logits cpu fallback failed");
            return s->logits;
        }
        if (bn_transformer_gpu_emit_context_logits(
                &emit, logit_res->gpu_buf, logit_res->type,
                logit_res->rows, logit_res->cols) != 0)
            GPU_REJECT("gpu graph logits emit failed");
    }

    // Safety: verify we didn't overflow the ops array
    if (emit.n + emit.graph.n_ops > max_ops) { GPU_REJECT("gpu op graph capacity exceeded"); }

    // Execute final batch (logits + any remaining layer ops)
    int rc = bn_transformer_gpu_emit_context_execute(
        &emit, gpu, BN_GPU_VALUE_LOGITS, s->logits, c->vocab_size);
    if (rc != 0) GPU_REJECT("gpu final execute failed");
    bn_transformer_gpu_emit_context_free(&emit);
    #undef GPU_LEGACY_OPS
    #undef GPU_FLUSH
    #undef GPU_REJECT
    return s->logits;
}
