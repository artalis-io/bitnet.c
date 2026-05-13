#include "transformer_gpu_internal.h"
#include "transformer_cpu_internal.h"
#include "model_arch.h"
#include "backend_session.h"
#include "quant.h"
#include "moe.h"
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BN_MAX_VLA_ELEMS 8192

static inline void *qweight_backend_buf(const BnBackendModel *backend,
                                        const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static inline void *backend_handle_or(const BnBackendModel *backend,
                                      int layer,
                                      BnBackendHandleRole role) {
    return bn_backend_model_handle(backend, layer, role);
}

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
    int debug_fallback = getenv("BN_GPU_DEBUG_FALLBACK") != NULL;
    static int debug_printed = 0;

#define GPU_REJECT(msg) do { \
        if (debug_fallback && !debug_printed) { \
            fprintf(stderr, "[gpu:fallback] %s\n", (msg)); \
            debug_printed = 1; \
        } \
        return NULL; \
    } while (0)

    if (!gpu || !gpu->execute || !gpu->write_activation) GPU_REJECT("backend missing execute/write_activation");

    // Bounds checks
    if (token < 0 || token >= c->vocab_size) GPU_REJECT("token out of bounds");
    if (pos < 0) GPU_REJECT("negative position");

    /* Large hybrid SSM/MoE decode graphs are much slower than CPU on the
     * current WebGPU/dzn stack. Keep an escape hatch for shader profiling. */
    if (!getenv("BN_GPU_FORCE_GRAPH") && c->dim >= 4096 &&
        (bn_model_arch_requires_large_gpu_graph_fallback(c) ||
         c->full_attn_interval > 0 || c->n_experts > 0))
        GPU_REJECT("large arch/hybrid/moe gpu graph disabled");

    // FP16 KV cache not supported on GPU path
    if (c->kv_f16) GPU_REJECT("kv_f16 unsupported");

    int dim = c->dim;
    int kv_dim = c->kv_dim;
    int head_size = c->head_size;
    int n_heads = c->n_heads;
    int q_dim = n_heads * head_size;
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;

    // Embed token on CPU, upload to GPU x buffer
    if (dim > BN_MAX_VLA_ELEMS) GPU_REJECT("dim exceeds VLA limit");
    float emb[dim];
    bn_model_embed_token(m, emb, token);
    if (gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, emb,
                              (size_t)dim * sizeof(float), 0) != 0)
        GPU_REJECT("write token embedding failed");

    /* no-op */

    // Validation: check for unsupported layer configurations.
    // SSM layers are handled via CPU fallback per-layer.
    void *output_norm = backend_handle_or(backend, -1,
                                          BN_BACKEND_HANDLE_OUTPUT_NORM);
    if (!output_norm) GPU_REJECT("output norm not uploaded");
    int has_moe = 0, has_ssm = 0;
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        void *attn_norm = backend_handle_or(backend, l,
                                            BN_BACKEND_HANDLE_ATTN_NORM);
        void *ffn_norm = backend_handle_or(backend, l,
                                           BN_BACKEND_HANDLE_FFN_NORM);
        void *q_norm = backend_handle_or(backend, l,
                                         BN_BACKEND_HANDLE_Q_NORM);
        void *k_norm = backend_handle_or(backend, l,
                                         BN_BACKEND_HANDLE_K_NORM);
        void *attn_sub_norm = backend_handle_or(backend, l,
                                                BN_BACKEND_HANDLE_ATTN_SUB_NORM);
        void *ffn_sub_norm = backend_handle_or(backend, l,
                                               BN_BACKEND_HANDLE_FFN_SUB_NORM);
        int is_attn = bn_transformer_is_attn_layer(c, l);
        if (!is_attn) { has_ssm = 1; continue; }
        if (lw->moe.router_weight) { has_moe = 1; }
        if (!lw->attn.wq.data && !lw->ssm.wqkv.data) GPU_REJECT("attention layer has no wq/wqkv data");
        // Q-gated attention is handled below via DEINTERLEAVE_Q + SIGMOID_GATE.
        if (lw->attn.q_norm && !q_norm) GPU_REJECT("q norm not uploaded");
        if (lw->attn.k_norm && !k_norm) GPU_REJECT("k norm not uploaded");
        if (lw->norm.attn_sub_norm && !attn_sub_norm) GPU_REJECT("attention sub norm not uploaded");
        if (lw->norm.ffn_sub_norm && !ffn_sub_norm) GPU_REJECT("ffn sub norm not uploaded");
        if (!attn_norm || !ffn_norm) GPU_REJECT("layer norm not uploaded");
    }
    if (has_moe) GPU_REJECT("moe gpu-resident forward unsupported");
    // Models with SSM need per-layer CPU-GPU sync via read/write_activation.
    if (has_ssm && (!gpu->read_activation || !gpu->write_activation))
        GPU_REJECT("ssm needs read/write activation");

    // Resolve logits weight
    BnQWeight *ow = &w->output_weight;
    void *logit_gpu_buf = ow->data ? qweight_backend_buf(backend, ow) : NULL;
    int logit_type = ow->data ? ow->type : -1;
    int logit_rows = ow->data ? ow->rows : c->vocab_size;
    int logit_cols = ow->data ? ow->cols : dim;
    void *tied_embedding = backend_handle_or(backend, -1,
                                             BN_BACKEND_HANDLE_TIED_EMBEDDING);
    if (!logit_gpu_buf && tied_embedding) {
        logit_gpu_buf = tied_embedding;
        logit_type = w->emb_type;
        logit_rows = c->vocab_size;
        logit_cols = dim;
    }
    if (!logit_gpu_buf) GPU_REJECT("logit weight not uploaded");

    // Precompute eps as uint32
    uint32_t u_eps;
    { float eps = c->norm_eps; memcpy(&u_eps, &eps, 4); }

    // Max ops per batch. MoE/SSM flush between layers, so single-layer max suffices.
    // Max ops per flush batch:
    // Attention: ~20 (QKV + norms + RoPE + GQA + sigmoid + Wo + resid)
    // SSM: ~16 (QKV + Z + conv + splits + L2norm + alpha/beta + delta + gate + out + resid)
    // MoE: K*5 + shared(5) + residual + rmsnorm = up to BN_MAX_MOE_K*5 + 7
    // Total worst case: 20 + BN_MAX_MOE_K*5 + 7 = 107 for K=16
    int max_ops = 80 * c->n_layers + 5 * BN_MAX_MOE_K + 100;

    // Reuse the session-owned op array to avoid per-token malloc.
    BnGPUGraph *graph =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(sess->backend, max_ops);
    if (!graph) GPU_REJECT("gpu graph allocation failed");
    BnGPUOp *ops = graph->ops;
    int n = 0;

    // Helper: flush current ops (no readback), reset counter
    #define GPU_FLUSH() do { \
        if (n > 0) { \
            bn_transformer_gpu_finalize_op_kinds(ops, n); \
            if (gpu->execute(gpu->ctx, ops, n, -1, NULL, 0) != 0) GPU_REJECT("gpu execute flush failed"); \
            n = 0; \
        } \
    } while(0)

    // ---- Initial RMSNorm: x -> xb (using layer 0 attn_norm) ----
    bn_transformer_gpu_emit_rmsnorm(ops, &n,
                         backend_handle_or(backend, 0, BN_BACKEND_HANDLE_ATTN_NORM),
                         BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps);

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
                GPU_FLUSH();
                if (gpu->read_activation(gpu->ctx, BN_GPU_VALUE_X, s->x,
                                          (size_t)dim * sizeof(float), 0) != 0)
                    { return NULL; }
                bn_transformer_cpu_forward_ssm_block(m, sess, lw, l);
                bn_transformer_cpu_residual_add(s->x, s->xb, dim);
                if (lw->moe.router_weight)
                    bn_moe_forward(m, sess, lw, l);
                else
                    bn_transformer_cpu_forward_ffn_block(m, sess, lw, NULL);
                if (gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, s->x,
                                           (size_t)dim * sizeof(float), 0) != 0)
                    { return NULL; }
                void *nn = (l + 1 < c->n_layers)
                    ? backend_handle_or(backend, l + 1, BN_BACKEND_HANDLE_ATTN_NORM)
                    : output_norm;
                bn_transformer_gpu_emit_rmsnorm(ops, &n, nn, BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps);
                continue;
            }

            bn_transformer_gpu_emit_ssm(ops, &n, c, lw, &plan, gpu,
                                        bn_model_backend(m), l, dim, u_eps);

            // SSM layer's FFN (dense or MoE) — same as attention layer below
            goto ffn_block;
        }

        // KV cache addressing
        int attn_idx = plan.attn_idx;
        size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
        int cache_pos = pos % c->seq_len;
        int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;

        uint32_t kv_cache_off = (uint32_t)(loff + (size_t)cache_pos * kv_dim);
        bn_transformer_gpu_emit_qkv(ops, &n, c, lw, &plan, gpu, bn_model_backend(m),
                                    l, pos, q_dim, head_size, n_heads,
                                    kv_dim, rope_dims, kv_cache_off, u_eps);
        bn_transformer_gpu_emit_attention(ops, &n, c, lw, gpu, bn_model_backend(m),
                                          l, pos, dim, q_dim, head_size,
                                          n_heads, kv_dim, rope_dims, n_kv,
                                          loff, kv_cache_off, has_moe, u_eps);

        // ---- FFN (MoE or dense) ----
        ffn_block:;
        if (lw->moe.router_weight) {
            // MoE FFN: CPU fallback until the WebGPU MoE path is token-coherent
            int use_cpu_moe_fallback = 1;
            if (use_cpu_moe_fallback) {
                GPU_FLUSH();
                if (gpu->read_activation(gpu->ctx, BN_GPU_VALUE_X, s->x,
                                          (size_t)dim * sizeof(float), 0) != 0)
                    { return NULL; }
                bn_moe_forward(m, sess, lw, l);
                if (gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, s->x,
                                           (size_t)dim * sizeof(float), 0) != 0)
                    { return NULL; }
                void *moe_next_norm = (l + 1 < c->n_layers)
                    ? backend_handle_or(backend, l + 1, BN_BACKEND_HANDLE_ATTN_NORM)
                    : output_norm;
                bn_transformer_gpu_emit_rmsnorm(ops, &n, moe_next_norm, BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps);
                continue;
            }

            void *uncached_bufs[BN_MAX_MOE_K * 3];
            int n_uncached = 0;
            void *next_norm = (l + 1 < c->n_layers)
                ? backend_handle_or(backend, l + 1, BN_BACKEND_HANDLE_ATTN_NORM)
                : output_norm;
            bn_transformer_gpu_emit_moe(ops, &n, m, sess, lw, l, dim, u_eps,
                                        next_norm, uncached_bufs, &n_uncached);
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
        void *next_norm = (l + 1 < c->n_layers)
            ? backend_handle_or(backend, l + 1, BN_BACKEND_HANDLE_ATTN_NORM)
            : output_norm;
        BnFFNPlan ffn_plan;
        bn_transformer_plan_ffn(&ffn_plan, c, lw, gpu, backend, l, 1);
        bn_transformer_gpu_emit_dense_ffn(ops, &n, c, lw, &ffn_plan, gpu,
                                          bn_model_backend(m), l, dim, u_eps, next_norm);
    }

    // ---- Logits matvec: xb -> logits (xb is already normalized) ----
    {
        size_t max_gpu_binding = gpu->max_storage_binding_size;
        if (max_gpu_binding == 0)
            max_gpu_binding = 128ull * 1024ull * 1024ull;
        BnQWeight tied = {0};
        BnQWeight *logit_cpu_w = ow->data ? ow : NULL;
        if (!logit_cpu_w && w->token_embedding && logit_type >= 0) {
            tied.data = w->token_embedding;
            tied.type = logit_type;
            tied.rows = c->vocab_size;
            tied.cols = dim;
            tied.scale = 1.0f;
            logit_cpu_w = &tied;
        }
        if (logit_cpu_w && bn_qweight_data_size(logit_cpu_w) > max_gpu_binding) {
            GPU_FLUSH();
            if (gpu->read_activation(gpu->ctx, BN_GPU_VALUE_XB, s->x,
                                      (size_t)dim * sizeof(float), 0) != 0)
                GPU_REJECT("read logits input failed");
            bn_quant_matvec(s->logits, logit_cpu_w, s->x, s->x_q, bn_model_pool(m));
            return s->logits;
        }
        bn_transformer_gpu_emit_logits(ops, &n, logit_gpu_buf, logit_type, logit_rows, logit_cols);
    }

    // Safety: verify we didn't overflow the ops array
    if (n > max_ops) { GPU_REJECT("gpu op graph capacity exceeded"); }

    // Execute final batch (logits + any remaining layer ops)
    bn_transformer_gpu_finalize_op_kinds(ops, n);
    int rc = gpu->execute(gpu->ctx, ops, n, BN_GPU_VALUE_LOGITS,
                          s->logits, c->vocab_size);
    if (rc != 0) GPU_REJECT("gpu final execute failed");
    #undef GPU_FLUSH
    #undef GPU_REJECT
    return s->logits;
}
