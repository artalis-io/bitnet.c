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

    BnTransformerGPUForwardPolicy policy;
    const char *reject_reason = NULL;
    if (bn_transformer_gpu_validate_forward(
            &policy, gpu, backend, c, w, token, pos, &reject_reason) != 0)
        return bn_transformer_gpu_reject_forward(&emit, reject_reason);

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
        return bn_transformer_gpu_reject_forward(
            &emit, "write token embedding failed");

    /* no-op */

    void *output_norm = policy.output_norm;
    BnTransformerGPULogitResources *logit_res = &policy.logits;
    int has_moe = policy.has_moe;

    // Precompute eps as uint32
    uint32_t u_eps;
    { float eps = c->norm_eps; memcpy(&u_eps, &eps, 4); }

    int max_ops = bn_transformer_gpu_graph_op_capacity(c);

    // Reuse the session-owned op array to avoid per-token malloc.
    int command_cap = 0;
    void *command_buffer = bn_backend_session_ensure_gpu_command_buffer(
        sess->backend, max_ops, &command_cap);
    if (!command_buffer)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph allocation failed");
    bn_transformer_gpu_emit_context_init(&emit, command_buffer, command_cap);

    // ---- Initial RMSNorm: x -> xb (using layer 0 attn_norm) ----
    if (bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
            &emit, bn_transformer_gpu_resolve_initial_norm(backend),
            dim, u_eps) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph rmsnorm emit failed");

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
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu ssm cpu fallback failed");
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
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe cpu fallback failed");
                continue;
            }

            BnGPUMoETemporaryBuffers moe_temporaries;
            void *next_norm = bn_transformer_gpu_resolve_next_norm(
                backend, l, c->n_layers, output_norm);
            BnGPUMoEResolvedExpert expert_emit[BN_MAX_MOE_K];
            BnGPUMoEResources moe_res;
            if (bn_gpu_moe_bridge_resolve_resources(
                    &moe_res, expert_emit, BN_MAX_MOE_K, m, sess, lw, l,
                    &moe_temporaries) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe resource resolution failed");
            BnTransformerGPUMoESharedResources moe_shared =
                bn_transformer_gpu_resolve_moe_shared_resources(backend, lw);
            bn_transformer_gpu_emit_context_moe(
                &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm);
            if (moe_temporaries.n_buffers > 0) {
                if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu execute flush failed");
                bn_gpu_moe_bridge_release_temporaries(m, &moe_temporaries);
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
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu logits cpu fallback failed");
            return s->logits;
        }
        if (bn_transformer_gpu_emit_context_logits(
                &emit, logit_res->gpu_buf, logit_res->type,
                logit_res->rows, logit_res->cols) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu graph logits emit failed");
    }

    // Safety: verify we didn't overflow the ops array
    if (emit.n + emit.graph.n_ops > max_ops)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu op graph capacity exceeded");

    // Execute final batch (logits + any remaining layer ops)
    int rc = bn_transformer_gpu_emit_context_execute_logits(
        &emit, gpu, s->logits, c->vocab_size);
    if (rc != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu final execute failed");
    bn_transformer_gpu_emit_context_free(&emit);
    #undef GPU_LEGACY_OPS
    return s->logits;
}
