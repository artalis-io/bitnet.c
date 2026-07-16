#include "transformer_cpu_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_batched_attn_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "backend_quant.h"
#include "backend_model.h"
#include "quant.h"
#include "moe.h"
#include "session.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline const BnPreparedWeight *cpu_qweight_prepared(
    const BnBackendModel *backend,
    const BnQWeight *w) {
    if (!bn_transformer_cpu_prepared_qweights_enabled())
        return NULL;
    return bn_backend_model_prepared_qweight(backend, w);
}

static BnMatvecTask *cpu_prepare_matvec_tasks(const BnModel *m,
                                              const BnMatvecTask *tasks,
                                              int n_tasks,
                                              BnMatvecTask *inline_tasks,
                                              int inline_cap) {
    BnMatvecTask *prepared = inline_tasks;
    if (n_tasks > inline_cap) {
        prepared = (BnMatvecTask *)malloc((size_t)n_tasks * sizeof(*prepared));
        if (!prepared) return NULL;
    }
    for (int i = 0; i < n_tasks; i++) {
        prepared[i] = tasks[i];
        prepared[i].prepared = cpu_qweight_prepared(bn_model_backend(m), tasks[i].W);
        prepared[i].flags |=
            bn_transformer_cpu_force_float_kquant_task_flags(&m->config);
    }
    return prepared;
}

static void cpu_quant_matvec_batch_prepared(const BnModel *m,
                                            const BnMatvecTask *tasks,
                                            int n_tasks,
                                            const float *x,
                                            int8_t *x_q_buf) {
    BnMatvecTask inline_tasks[8];
    BnMatvecTask *prepared_tasks =
        cpu_prepare_matvec_tasks(m, tasks, n_tasks, inline_tasks, 8);
    if (!prepared_tasks) {
        bn_quant_matvec_batch(tasks, n_tasks, x, x_q_buf, bn_model_pool(m));
        return;
    }
    if (bn_model_gpu(m)) {
        void *bufs_inline[8];
        void **bufs = bufs_inline;
        void **heap_bufs = NULL;
        if (n_tasks > 8) {
            heap_bufs = (void **)malloc((size_t)n_tasks * sizeof(*heap_bufs));
            bufs = heap_bufs;
        }
        if (bufs) {
            const BnBackendModel *backend = bn_model_backend(m);
            for (int i = 0; i < n_tasks; i++)
                bufs[i] = bn_backend_model_qweight_buf(backend,
                                                       prepared_tasks[i].W);
            bn_backend_quant_matvec_batch_gpu_buf(
                prepared_tasks, (const void *const *)bufs, n_tasks, x,
                x_q_buf, bn_model_pool(m), bn_model_gpu(m));
            free(heap_bufs);
            if (prepared_tasks != inline_tasks) free(prepared_tasks);
            return;
        }
    }
    bn_quant_matvec_batch(prepared_tasks, n_tasks, x, x_q_buf, bn_model_pool(m));
    if (prepared_tasks != inline_tasks) free(prepared_tasks);
}

static void cpu_quant_matvec_batch_preq8k(const BnModel *m,
                                          const BnMatvecTask *tasks,
                                          int n_tasks,
                                          const int8_t *x_q,
                                          const float *x_d,
                                          const int16_t *x_bsums,
                                          const float *x_float) {
    if (bn_model_gpu(m)) {
        cpu_quant_matvec_batch_prepared(m, tasks, n_tasks, x_float,
                                        (int8_t *)x_q);
        return;
    }
    if (bn_transformer_cpu_force_float_kquant_task_flags(&m->config)) {
        cpu_quant_matvec_batch_prepared(m, tasks, n_tasks, x_float,
                                        (int8_t *)x_q);
        return;
    }

    BnMatvecTask inline_tasks[8];
    BnMatvecTask *prepared_tasks =
        cpu_prepare_matvec_tasks(m, tasks, n_tasks, inline_tasks, 8);
    if (!prepared_tasks) {
        bn_quant_matvec_batch_preq8k(tasks, n_tasks, x_q, x_d, x_bsums,
                                     x_float, bn_model_pool(m));
        return;
    }
    bn_quant_matvec_batch_preq8k(prepared_tasks, n_tasks, x_q, x_d, x_bsums,
                                 x_float, bn_model_pool(m));
    if (prepared_tasks != inline_tasks) free(prepared_tasks);
}

static int cpu_quant_can_preq8k_pair(int a, int b) {
    return bn_quant_format_can_preq8k(a) && bn_quant_format_can_preq8k(b);
}

static const BnCPUBackendOps *cpu_backend_ops(void) {
    return bn_transformer_cpu_backend_ops();
}

static int cpu_quant_can_preq8k_triple(const BnCPUBackendOps *ops,
                                       int a, int b, int c) {
    return ops->supports_preq8k &&
           cpu_quant_can_preq8k_pair(a, b) &&
           bn_quant_format_can_preq8k(c);
}

static void cpu_rmsnorm_reference_scalar_order(float *out, const float *x,
                                           const float *w, int size, float eps) {
    double ss = 0.0;
    for (int i = 0; i < size; i++)
        ss += (double)(x[i] * x[i]);
    float scale = 1.0f / sqrtf((float)(ss / (double)size) + eps);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * scale * w[i];
}

static inline void cpu_rmsnorm_model(const BnModel *m, float *out,
                                     const float *x, const float *w,
                                     int size, float eps) {
    if (m && bn_transformer_rmsnorm_requires_reference_scalar_order(&m->config)) {
        cpu_rmsnorm_reference_scalar_order(out, x, w, size, eps);
        return;
    }
    cpu_backend_ops()->rmsnorm(out, x, w, size, eps);
}

static void cpu_rmsnorm_unit(float *out, const float *x, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)size + eps);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * ss;
}

static void cpu_rmsnorm_unit_heads(float *x, int n_heads, int head_size, float eps) {
    for (int h = 0; h < n_heads; h++)
        cpu_rmsnorm_unit(x + h * head_size, x + h * head_size, head_size, eps);
}

void bn_transformer_cpu_gqa_dispatch(BnModel *m,
                                     BnGQACtx *gctx,
                                     int n_heads,
                                     int kv_mul) {
    (void)kv_mul;
    const BnCPUBackendOps *ops = cpu_backend_ops();
    if (gctx->attention_scale == 0.0f)
        gctx->attention_scale =
            bn_transformer_attention_scale(&m->config, gctx->head_size);
    bn_tp_fn attn_fn = m->config.flash_attn ? ops->flash_gqa : ops->gqa;
    BnTPTask gqa = { attn_fn, gctx, n_heads };
    bn_tp_dispatch(bn_model_pool(m), &gqa, 1);
}

void bn_transformer_batched_attn_dispatch(BnModel *m,
                                          BnBatchedAttnCtx *ctx) {
    const BnCPUBackendOps *ops = cpu_backend_ops();
    if (ctx->attention_scale == 0.0f)
        ctx->attention_scale =
            bn_transformer_attention_scale(&m->config, ctx->head_size);
    bn_tp_fn fn = m->config.flash_attn
        ? ((ctx->n_tokens > 1 && ops->batched_attn_flash_pair)
            ? ops->batched_attn_flash_pair
            : ops->batched_attn_flash)
        : ops->batched_attn_naive;
    int units = ctx->n_heads;
    if (fn == ops->batched_attn_flash_pair)
        units = ctx->n_heads * ctx->n_tokens;
    BnTPTask task = { fn, ctx, units };
    bn_tp_dispatch(bn_model_pool(m), &task, 1);
}

void bn_transformer_cpu_residual_add(float *x, const float *r, int dim) {
    cpu_backend_ops()->residual_add(x, r, dim);
}

static void cpu_debug_dump_array_n(int n_values,
                                   const float *x,
                                   const char *tag,
                                   int layer,
                                   int pos) {
    const char *path = getenv("BN_DUMP_LAYER_INP");
    if (!path || !path[0]) return;

    const char *pos_env = getenv("BN_DUMP_LAYER_POS");
    if (pos_env && pos_env[0] && atoi(pos_env) != pos) return;

    float sum = 0.0f;
    float ss = 0.0f;
    float minv = x[0];
    float maxv = x[0];
    for (int i = 0; i < n_values; i++) {
        float v = x[i];
        sum += v;
        ss += v * v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }

    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "%s pos=%d layer=%d dim=%d sum=%.9g ss=%.9g min=%.9g max=%.9g first=",
            tag, pos, layer, n_values, sum, ss, minv, maxv);
    int n = n_values < 16 ? n_values : 16;
    for (int i = 0; i < n; i++)
        fprintf(f, "%s%.9g", i ? "," : "", x[i]);
    fputc('\n', f);
    fclose(f);
}

static void cpu_debug_dump_attn_weights(const BnRunState *s,
                                        int n_heads,
                                        int n_kv,
                                        int seq_len,
                                        const char *tag,
                                        int layer,
                                        int pos) {
    const char *path = getenv("BN_DUMP_LAYER_INP");
    if (!path || !path[0]) return;
    if (n_heads <= 0 || n_kv <= 0 || seq_len <= 0) return;

    int n_values = n_heads * n_kv;
    float *tmp = (float *)malloc((size_t)n_values * sizeof(*tmp));
    if (!tmp) return;
    for (int h = 0; h < n_heads; h++)
        memcpy(tmp + (size_t)h * n_kv, s->att + (size_t)h * seq_len,
               (size_t)n_kv * sizeof(*tmp));
    cpu_debug_dump_array_n(n_values, tmp, tag, layer, pos);
    free(tmp);
}

static void cpu_debug_dump_heads(const float *x,
                                 int n_heads,
                                 int head_size,
                                 const char *tag,
                                 int layer,
                                 int pos) {
    const char *enabled = getenv("BN_DUMP_ALL_HEADS");
    if (!enabled || !enabled[0]) return;
    char head_tag[96];
    for (int h = 0; h < n_heads; h++) {
        snprintf(head_tag, sizeof(head_tag), "%s_h%d", tag, h);
        cpu_debug_dump_array_n(head_size, x + (size_t)h * head_size,
                               head_tag, layer, pos);
    }
}

static void cpu_debug_dump_attn_weight_heads(const BnRunState *s,
                                             int n_heads,
                                             int n_kv,
                                             int seq_len,
                                             const char *tag,
                                             int layer,
                                             int pos) {
    const char *enabled = getenv("BN_DUMP_ALL_HEADS");
    if (!enabled || !enabled[0]) return;
    if (n_heads <= 0 || n_kv <= 0 || seq_len <= 0) return;
    char head_tag[96];
    for (int h = 0; h < n_heads; h++) {
        snprintf(head_tag, sizeof(head_tag), "%s_h%d", tag, h);
        cpu_debug_dump_array_n(n_kv, s->att + (size_t)h * seq_len,
                               head_tag, layer, pos);
    }
}

static void cpu_debug_dump_array(const BnConfig *c,
                                 const float *x,
                                 const char *tag,
                                 int layer,
                                 int pos) {
    cpu_debug_dump_array_n(c->dim, x, tag, layer, pos);
}

static void cpu_debug_dump_vector(const BnModel *m,
                                  const BnSession *sess,
                                  const char *tag,
                                  int layer,
                                  int pos) {
    cpu_debug_dump_array(&m->config, sess->state.x, tag, layer, pos);
}

static void cpu_debug_dump_layer_input(const BnModel *m,
                                       const BnSession *sess,
                                       int layer,
                                       int pos) {
    cpu_debug_dump_vector(m, sess, "bitnet_inp", layer, pos);
}

static void cpu_apply_arch_per_layer_embedding(BnModel *m,
                                               BnSession *sess,
                                               BnLayerWeights *lw,
                                               int layer) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int per_dim = bn_transformer_per_layer_embedding_dim(c);
    if (per_dim <= 0 ||
        !s->per_layer_input || !lw->per_layer.inp_gate.data ||
        !lw->per_layer.proj.data || !lw->per_layer.post_norm)
        return;

    memcpy(s->xb2, s->x, (size_t)c->dim * sizeof(float));

    BnMatvecTask gate[1] = {{ s->hb, &lw->per_layer.inp_gate, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, gate, 1, s->x, s->x_q);
    for (int i = 0; i < per_dim; i++) {
        float g = s->hb[i];
        float gelu = 0.5f * g *
                     (1.0f + tanhf(0.7978845608028654f * g *
                                   (1.0f + 0.044715f * g * g)));
        s->hb[i] = gelu * s->per_layer_input[(size_t)layer * per_dim + i];
    }

    BnMatvecTask proj[1] = {{ s->x, &lw->per_layer.proj, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, proj, 1, s->hb, s->x_q);
    cpu_rmsnorm_model(m, s->x, s->x, lw->per_layer.post_norm, c->dim, c->norm_eps);
    bn_transformer_cpu_residual_add(s->x, s->xb2, c->dim);
}

void bn_transformer_cpu_apply_rope_heads(float *buf,
                                         int n_heads,
                                         int head_size,
                                         int rope_dims,
                                         const float *rc,
                                         const float *rs) {
    cpu_backend_ops()->apply_rope_heads(buf, n_heads, head_size,
                                        rope_dims, rc, rs);
}

void bn_transformer_cpu_apply_ffn_activation(BnRunState *s,
                                             const BnFFNPlan *ffn_plan,
                                             int hidden_dim,
                                             int already_activated) {
    if (already_activated)
        return;
    cpu_backend_ops()->apply_ffn_activation(s, ffn_plan, hidden_dim);
}

// Process a single layer (attention/SSM block + FFN). Reads/writes s->x.
// Returns 0 on success.
int bn_transformer_cpu_forward_layer(BnModel *m, BnSession *sess, int l, int pos, int cache_pos,
                                int rope_dims, const float *rope_cos,
                                const float *rope_sin) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
    int dim = c->dim;
    int n_heads = c->n_heads;
    BnLayerWeights *lw = &w->layers[l];
    BnAttentionPlan attn_plan;
    BnFFNPlan ffn_plan;
    BnMoEPlan moe_plan;
    BnSSMPlan ssm_plan;
    bn_transformer_plan_attention(&attn_plan, c, lw, bn_model_gpu(m),
                                  bn_model_backend(m), l, bn_model_tq_state(m) != NULL, 0);
    bn_transformer_plan_ffn(&ffn_plan, c, lw, bn_model_gpu(m),
                            bn_model_backend(m), l, 0);
    bn_transformer_plan_moe(&moe_plan, c, lw, bn_model_gpu(m), l, 0);
    bn_transformer_plan_ssm(&ssm_plan, c, lw, l, 0, bn_model_gpu(m),
                            bn_model_backend(m));
    const BnLayerShapePlan *shape = &attn_plan.shape;
    int head_size = shape->head_size;
    int kv_dim = shape->kv_dim;
    int kv_cache_stride = c->kv_dim;
    int n_kv_heads = shape->n_kv_heads;
    int kv_mul = shape->kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    int qk_stride = shape->qk_stride; // per-head norm offset
    int is_attn = shape->is_attn;
    const BnCPUBackendOps *cpu_ops = cpu_backend_ops();

    cpu_debug_dump_layer_input(m, sess, l, pos);

    if (is_attn) {
        // ---- Attention block ----

        // KV cache offset: contiguous among attention layers only
        int attn_idx = shape->attn_idx;
        int kv_read_idx = attn_idx;
        if (!lw->attn.has_kv && lw->attn.kv_reuse_layer >= 0)
            kv_read_idx = bn_transformer_attn_index(c, lw->attn.kv_reuse_layer);
        size_t loff = (size_t)attn_idx * c->seq_len * kv_cache_stride;
        size_t read_loff = (size_t)kv_read_idx * c->seq_len * kv_cache_stride;

        // Q projection width detection:
        // q_dim = n_heads * head_size (total Q output elements)
        // Gated Q: wq.rows = 2 * q_dim (interleaved [Q, gate] per head)
        // Wide Q: wq.rows = q_dim > dim (head_size > dim/n_heads)
        // Classic: wq.rows = dim = q_dim
        int q_dim = shape->q_dim;
        int q_gated = shape->q_gated;
        int q_wide = shape->q_wide;

        /* Fused attn RMSNorm + Q8K: quantize s->xb once, reuse for Q and K+V */
        int attn_preq8k = 0;
        int attn_kquant = cpu_quant_can_preq8k_triple(
            cpu_ops, lw->attn.wq.type, lw->attn.wk.type, lw->attn.wv.type) &&
            !bn_model_gpu(m) && dim % BN_QK_K == 0;
        int n_sb_attn = dim / BN_QK_K;
        float attn_q8k_d[n_sb_attn > 0 ? n_sb_attn : 1];
        int16_t attn_q8k_bsums[n_sb_attn > 0 ? n_sb_attn * 16 : 1];
        if (attn_kquant) {
            cpu_ops->rmsnorm_q8k(s->x, lw->norm.attn_norm, dim, c->norm_eps,
                                 s->xb, s->x_q, attn_q8k_d,
                                 attn_q8k_bsums);
            attn_preq8k = 1;
        } else
        {
            cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.attn_norm, dim, c->norm_eps);
        }
        cpu_debug_dump_array(c, s->xb, "bitnet_attn_norm", l, pos);

        /* no-op */

        if (q_gated) {
            // --- Gated Q attention path ---
            float *q_full = s->hb;  // [2*dim]
            float *k_tmp = s->hb2;
            float *v_tmp = s->hb2 + kv_dim;

            // Q+K+V matvecs (reuse cached Q8K if available)
            if (!(c->kv_tq_bits > 0 && bn_model_tq_state(m)) && !c->kv_f16) {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
                BnMatvecTask qkv[3] = {
                     { q_full,          &lw->attn.wq, NULL, 0 },
                     { key_cache_row,   &lw->attn.wk, NULL, 0 },
                     { value_cache_row, &lw->attn.wv, NULL, 0 },
                };
                if (attn_preq8k)
                    cpu_quant_matvec_batch_preq8k(m, qkv, 3, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);
                k_tmp = key_cache_row;
                v_tmp = value_cache_row;
            } else {
                BnMatvecTask qkv[3] = {
                     { q_full, &lw->attn.wq, NULL, 0 },
                     { k_tmp, &lw->attn.wk, NULL, 0 },
                     { v_tmp, &lw->attn.wv, NULL, 0 },
                };
                if (attn_preq8k)
                    cpu_quant_matvec_batch_preq8k(m, qkv, 3, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);
            }

            /* Extract Q from interleaved [Q, gate] and optionally apply Q norm.
             * Fused: copy from q_full stride-2hs directly into cpu_rmsnorm if norm exists,
             * avoiding a separate memcpy + reload. */
            if (lw->attn.q_norm) {
                for (int h = 0; h < n_heads; h++)
                    cpu_rmsnorm_model(m, s->q + h*head_size,
                            q_full + h * 2 * head_size,
                            lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
            } else {
                for (int h = 0; h < n_heads; h++)
                    memcpy(s->q + h * head_size,
                           q_full + h * 2 * head_size,
                           head_size * sizeof(float));
            }
            if (lw->attn.k_norm)
                for (int h = 0; h < n_kv_heads; h++)
                    cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                            lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);
            if (bn_transformer_attention_value_shares_key(c))
                cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, c->norm_eps);

            bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);
            bn_transformer_cpu_apply_rope_heads(k_tmp, n_kv_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);

            // Write KV + GQA
            if (c->kv_tq_bits > 0 && bn_model_tq_state(m)) {
                bn_transformer_tq_write_kv(bn_model_tq_state(m), s, k_tmp, v_tmp,
                            n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);
                bn_transformer_tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                n_kv_heads, head_size, kv_mul);
            } else if (c->kv_f16) {
                bn_transformer_write_kv_fp16(s, loff, cache_pos,
                                             kv_cache_stride, k_tmp, v_tmp,
                                             kv_dim);
            }
            // FP32 path already wrote to cache directly

            if (!(c->kv_tq_bits > 0 && bn_model_tq_state(m))) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_cache_stride,
                                  c->seq_len,
                                  bn_transformer_attention_scale(c, head_size) };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            }

            // Sigmoid gate: xb *= sigmoid(gate)
            for (int h = 0; h < n_heads; h++) {
                float *gate_h = q_full + h * 2 * head_size + head_size;
                float *xb_h = s->xb + h * head_size;
                cpu_ops->apply_sigmoid_gate(xb_h, gate_h, head_size);
            }

            // wo projection + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.attn_sub_norm, dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
            if (bn_transformer_attention_uses_post_norm(c) &&
                lw->norm.attn_post_norm)
                cpu_rmsnorm_model(m, s->xb2, s->xb2, lw->norm.attn_post_norm, dim, c->norm_eps);
            bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

        } else if (q_wide) {
            // --- Wide Q attention path: head_size > dim/n_heads, no gate ---
            float *k_tmp = s->hb, *v_tmp = s->hb2;
            int has_kv = lw->attn.has_kv;

            // Q matvec: xb[dim] → q[q_dim]
            {
                BnMatvecTask q_task[1] = {{ s->q, &lw->attn.wq, NULL, 0 }};
                cpu_quant_matvec_batch_prepared(m, q_task, 1, s->xb, s->x_q);
            }
            // K/V matvec: xb[dim] -> kv_dim. Compact KV formats need temp
            // FP32 rows before packing into the cache.
            if (!has_kv) {
                /* Shared-KV layer: Q-only projection, then attend to the
                 * reused cache layer selected by the model-family policy. */
            } else if ((c->kv_tq_bits > 0 && bn_model_tq_state(m)) || c->kv_f16) {
                BnMatvecTask kv[2] = {
                     { k_tmp, &lw->attn.wk, NULL, 0 },
                     { v_tmp, &lw->attn.wv, NULL, 0 },
                };
                cpu_quant_matvec_batch_prepared(m, kv, 2, s->xb, s->x_q);
            } else {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
                BnMatvecTask kv[2] = {
                     { key_cache_row,   &lw->attn.wk, NULL, 0 },
                     { value_cache_row, &lw->attn.wv, NULL, 0 },
                };
                cpu_quant_matvec_batch_prepared(m, kv, 2, s->xb, s->x_q);
                k_tmp = key_cache_row;
                v_tmp = value_cache_row;
            }

            if (lw->attn.q_norm)
                for (int h = 0; h < n_heads; h++)
                    cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                            lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
            if (has_kv && lw->attn.k_norm)
                for (int h = 0; h < n_kv_heads; h++)
                    cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                            lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);
            if (has_kv && bn_transformer_attention_value_shares_key(c))
                cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, c->norm_eps);

            bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);
            if (has_kv)
                bn_transformer_cpu_apply_rope_heads(k_tmp, n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

            if (has_kv && c->kv_tq_bits > 0 && bn_model_tq_state(m)) {
                // TQ write + GQA
                bn_transformer_tq_write_kv(bn_model_tq_state(m), s, k_tmp, v_tmp,
                            n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);
                bn_transformer_tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                n_kv_heads, head_size, kv_mul);
            } else if (has_kv && c->kv_f16) {
                bn_transformer_write_kv_fp16(s, loff, cache_pos,
                                             kv_cache_stride, k_tmp, v_tmp,
                                             kv_dim);
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size,
                                  kv_cache_stride, c->seq_len,
                                  bn_transformer_attention_scale(c, head_size) };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            } else {
                // Standard GQA
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, has_kv ? loff : read_loff, pos, n_kv, kv_mul, head_size, kv_cache_stride,
                                  c->seq_len,
                                  bn_transformer_attention_scale(c, head_size) };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            }

            // wo projection (q_dim → dim) + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.attn_sub_norm, q_dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
            if (bn_transformer_attention_uses_post_norm(c) &&
                lw->norm.attn_post_norm)
                cpu_rmsnorm_model(m, s->xb2, s->xb2, lw->norm.attn_post_norm, dim, c->norm_eps);
            bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

        } else {
            // --- Classic attention path (existing) ---
            float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
            float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;

            if (c->kv_tq_bits > 0 && bn_model_tq_state(m)) {
                // --- TurboQuant KV path ---
                // Use temp buffers for K/V, then quantize into TQ cache
                float *k_tmp = s->hb, *v_tmp = s->hb2;
                BnMatvecTask qkv[3] = {
                     { s->q,  &lw->attn.wq, NULL, 0 },
                     { k_tmp, &lw->attn.wk, NULL, 0 },
                     { v_tmp, &lw->attn.wv, NULL, 0 },
                };
                if (attn_preq8k)
                    cpu_quant_matvec_batch_preq8k(m, qkv, 3, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);

                if (lw->attn.q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
                if (lw->attn.k_bias) for (int i = 0; i < kv_dim; i++) k_tmp[i] += lw->attn.k_bias[i];
                if (lw->attn.v_bias) for (int i = 0; i < kv_dim; i++) v_tmp[i] += lw->attn.v_bias[i];

                if (lw->attn.q_norm)
                    for (int h = 0; h < n_heads; h++)
                        cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);
                if (bn_transformer_attention_value_shares_key(c))
                    cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, c->norm_eps);

                bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(k_tmp, n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

                // Write TQ compressed KV
                bn_transformer_tq_write_kv(bn_model_tq_state(m), s, k_tmp, v_tmp,
                            n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);

                // TQ GQA
                bn_transformer_tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                n_kv_heads, head_size, kv_mul);

            } else if (c->kv_f16) {
                float *k_tmp = s->hb, *v_tmp = s->hb2;
                BnMatvecTask qkv[3] = {
                     { s->q,  &lw->attn.wq, NULL, 0 },
                     { k_tmp, &lw->attn.wk, NULL, 0 },
                     { v_tmp, &lw->attn.wv, NULL, 0 },
                };
                if (attn_preq8k)
                    cpu_quant_matvec_batch_preq8k(m, qkv, 3, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);

                if (lw->attn.q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
                if (lw->attn.k_bias) for (int i = 0; i < kv_dim; i++) k_tmp[i] += lw->attn.k_bias[i];
                if (lw->attn.v_bias) for (int i = 0; i < kv_dim; i++) v_tmp[i] += lw->attn.v_bias[i];

                if (lw->attn.q_norm)
                    for (int h = 0; h < n_heads; h++)
                        cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);
                if (bn_transformer_attention_value_shares_key(c))
                    cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, c->norm_eps);

                bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(k_tmp, n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

                bn_transformer_write_kv_fp16(s, loff, cache_pos,
                                             kv_cache_stride, k_tmp, v_tmp,
                                             kv_dim);
            } else {
                BnMatvecTask qkv[3] = {
                     { s->q,            &lw->attn.wq, NULL, 0 },
                     { key_cache_row,   &lw->attn.wk, NULL, 0 },
                     { value_cache_row, &lw->attn.wv, NULL, 0 },
                };
                if (attn_preq8k)
                    cpu_quant_matvec_batch_preq8k(m, qkv, 3, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);

                if (lw->attn.q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
                if (lw->attn.k_bias) for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
                if (lw->attn.v_bias) for (int i = 0; i < kv_dim; i++) value_cache_row[i] += lw->attn.v_bias[i];

                if (lw->attn.q_norm)
                    for (int h = 0; h < n_heads; h++)
                        cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm_model(m, key_cache_row + h*head_size, key_cache_row + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);
                if (bn_transformer_attention_value_shares_key(c))
                    cpu_rmsnorm_unit_heads(value_cache_row, n_kv_heads, head_size, c->norm_eps);

                bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

                cpu_debug_dump_array_n(q_dim, s->q, "bitnet_attn_q", l, pos);
                cpu_debug_dump_array_n(kv_dim, key_cache_row, "bitnet_attn_k", l, pos);
                cpu_debug_dump_array_n(kv_dim, value_cache_row, "bitnet_attn_v", l, pos);
                cpu_debug_dump_heads(s->q, n_heads, head_size,
                                     "bitnet_attn_q", l, pos);
            }

            // GQA attention (standard path — TQ handled above)
            if (!(c->kv_tq_bits > 0 && bn_model_tq_state(m))) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_cache_stride,
                                  c->seq_len,
                                  bn_transformer_attention_scale(c, head_size) };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
                cpu_debug_dump_attn_weights(s, n_heads, n_kv, c->seq_len,
                                            "bitnet_attn_softmax", l, pos);
                cpu_debug_dump_attn_weight_heads(s, n_heads, n_kv, c->seq_len,
                                                "bitnet_attn_softmax", l, pos);
            }
            cpu_debug_dump_array_n(q_dim, s->xb, "bitnet_attn_out", l, pos);
            cpu_debug_dump_heads(s->xb, n_heads, head_size,
                                 "bitnet_attn_out", l, pos);

            // Attention sub-norm + wo projection + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.attn_sub_norm, dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
            if (bn_transformer_attention_uses_post_norm(c) &&
                lw->norm.attn_post_norm)
                cpu_rmsnorm_model(m, s->xb2, s->xb2, lw->norm.attn_post_norm, dim, c->norm_eps);
            bn_transformer_cpu_residual_add(s->x, s->xb2, dim);
        }

    } else {
        // ---- SSM block ----
        (void)ssm_plan;
        bn_transformer_cpu_forward_ssm_block(m, sess, lw, l);
        bn_transformer_cpu_residual_add(s->x, s->xb, dim);
    }

    // ---- FFN block ---- (shared by both layer types)
    /* no-op */
    cpu_debug_dump_vector(m, sess, "bitnet_ffn_inp", l, pos);
    if (ffn_plan.kind == BN_FFN_MOE) {
        // MoE FFN — route, pread, compute, combine
        (void)moe_plan;
        bn_moe_forward(m, sess, lw, l);
    } else {
        // Dense FFN
        bn_transformer_cpu_forward_ffn_block(m, sess, lw, l, pos, &ffn_plan);
    }

    cpu_apply_arch_per_layer_embedding(m, sess, lw, l);

    if (bn_transformer_uses_layer_output_scale(c) &&
        lw->norm.layer_output_scale) {
        float scale = lw->norm.layer_output_scale[0];
        for (int i = 0; i < dim; i++)
            s->x[i] *= scale;
    }

    (void)is_attn; // used only in debug builds


    return 0;
}


void bn_transformer_cpu_forward_ssm_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          int layer) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int dim = c->dim;
    int num_k_heads = c->ssm_group_count;
    int head_k_dim = c->ssm_state_size;
    int num_v_heads = c->ssm_time_step_rank;
    int head_v_dim = c->ssm_inner_size / num_v_heads;
    int key_dim = num_k_heads * head_k_dim;
    int value_dim = c->ssm_inner_size;
    int qkv_dim = key_dim * 2 + value_dim;
    int kern = c->ssm_conv_kernel;
    int ssm_idx = bn_transformer_ssm_index(c, layer);
    size_t state_per_layer = (size_t)num_v_heads * head_k_dim * head_v_dim;
    float *state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
    size_t conv_per_layer = (size_t)(kern - 1) * qkv_dim;
    float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;
    const BnCPUBackendOps *cpu_ops = cpu_backend_ops();

    int ssm_preq8k = 0;
    int n_sb_ssm = dim / BN_QK_K;
    float ssm_q8k_d[n_sb_ssm > 0 ? n_sb_ssm : 1];
    int16_t ssm_q8k_bsums[n_sb_ssm > 0 ? n_sb_ssm * 16 : 1];
    int ssm_kquant = cpu_ops->supports_preq8k &&
                     !bn_model_gpu(m) && dim % BN_QK_K == 0 &&
                     cpu_quant_can_preq8k_pair(lw->ssm.wqkv.type, lw->ssm.wz.type);
    if (ssm_kquant) {
        cpu_ops->rmsnorm_q8k(s->x, lw->norm.attn_norm, dim, c->norm_eps,
                             s->xb, s->x_q, ssm_q8k_d, ssm_q8k_bsums);
        ssm_preq8k = 1;
    } else
    {
        cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.attn_norm, dim, c->norm_eps);
    }

    float *qkv = s->hb;
    float *z = s->hb2;
    BnMatvecTask qz_tasks[2] = {
         { qkv, &lw->ssm.wqkv, NULL, 0 },
        { z,   &lw->ssm.wz, NULL, 0 },
    };
    if (ssm_preq8k)
        cpu_quant_matvec_batch_preq8k(m, qz_tasks, 2, s->x_q,
                                     ssm_q8k_d, ssm_q8k_bsums, s->xb);
    else
        cpu_quant_matvec_batch_prepared(m, qz_tasks, 2, s->xb, s->x_q);

    int scalar_hybrid_ssm = bn_transformer_cpu_uses_scalar_hybrid_ssm(c);
    BnSSMConvCtx conv_ctx = { qkv, conv_state, lw->ssm.ssm_conv1d, qkv_dim, kern };
    BnTPTask conv_task = {
        scalar_hybrid_ssm ? bn_transformer_ssm_conv_silu_scalar_range
                          : cpu_ops->ssm_conv_silu,
        &conv_ctx, qkv_dim
    };
    bn_tp_dispatch(bn_model_pool(m), &conv_task, 1);

    float *q_raw = qkv;
    float *k_raw = qkv + key_dim;
    float *v_raw = qkv + 2 * key_dim;

    BnSSML2NormCtx norm_ctx = { q_raw, k_raw, c->norm_eps, head_k_dim };
    BnTPTask norm_task = {
        scalar_hybrid_ssm ? bn_transformer_ssm_l2norm_scalar_range
                          : cpu_ops->ssm_l2norm,
        &norm_ctx, num_k_heads
    };
    bn_tp_dispatch(bn_model_pool(m), &norm_task, 1);

    if (num_v_heads > 8192 || head_v_dim > 8192) {
        SH_LOG_ERROR("SSM dimensions too large for stack VLAs");
        return;
    }
    float alpha_arr[num_v_heads], beta_arr[num_v_heads];
    BnMatvecTask ab[2] = {
         { alpha_arr, &lw->ssm.ssm_alpha, NULL, 0 },
        { beta_arr,  &lw->ssm.ssm_beta, NULL, 0 },
    };
    if (ssm_preq8k &&
        cpu_quant_can_preq8k_pair(lw->ssm.ssm_alpha.type, lw->ssm.ssm_beta.type)) {
        cpu_quant_matvec_batch_preq8k(m, ab, 2, s->x_q,
                                     ssm_q8k_d, ssm_q8k_bsums, s->xb);
    } else {
        cpu_quant_matvec_batch_prepared(m, ab, 2, s->xb, s->x_q);
    }

    for (int h = 0; h < num_v_heads; h++) {
        float dt = alpha_arr[h] + lw->ssm.ssm_dt_bias[h];
        float dt_sp = (dt > 20.0f) ? dt : logf(1.0f + expf(dt));
        alpha_arr[h] = expf(dt_sp * lw->ssm.ssm_a[h]);
        beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
    }

    float *out = s->xb2;
    float q_scale = 1.0f / sqrtf((float)head_k_dim);
    BnSSMDeltaCtx delta_ctx = {
        state, out, q_raw, k_raw, v_raw,
        alpha_arr, beta_arr,
        num_k_heads, head_k_dim, head_v_dim, q_scale
    };
    BnTPTask delta_task = {
        scalar_hybrid_ssm ? bn_transformer_ssm_delta_scalar_range
                          : cpu_ops->ssm_delta,
        &delta_ctx, num_v_heads
    };
    bn_tp_dispatch(bn_model_pool(m), &delta_task, 1);

    BnSSMGateCtx gate_ctx = { out, z, lw->ssm.ssm_norm, c->norm_eps, head_v_dim };
    BnTPTask gate_task = {
        scalar_hybrid_ssm ? bn_transformer_ssm_gate_scalar_range
                          : cpu_ops->ssm_gate,
        &gate_ctx, num_v_heads
    };
    bn_tp_dispatch(bn_model_pool(m), &gate_task, 1);

    BnMatvecTask proj[1] = {{ s->xb, &lw->ssm.ssm_out, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, proj, 1, out, s->x_q);
}

void bn_transformer_cpu_forward_ffn_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          int layer,
                                          int pos,
                                          const BnFFNPlan *ffn_plan) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnFFNPlan local_plan;
    if (!ffn_plan) {
        bn_transformer_plan_ffn(&local_plan, c, lw, bn_model_gpu(m),
                                bn_model_backend(m), 0, bn_model_gpu(m) != NULL);
        ffn_plan = &local_plan;
    }
    int dim = c->dim;
    int hidden_dim = ffn_plan->hidden_dim;
    int ffn_activated = 0;
    int fused_gate_up = 0;
    const BnCPUBackendOps *cpu_ops = cpu_backend_ops();

    BnGPUBackend *gpu = bn_model_gpu(m);
    if (gpu && gpu->dense_ffn && ffn_plan->has_gate &&
        !ffn_plan->has_sub_norm && ffn_plan->activation == 0) {
        const BnBackendModel *backend = bn_model_backend(m);
        void *gate_buf = bn_backend_model_qweight_buf(backend, &lw->ffn.ffn_gate);
        void *up_buf = bn_backend_model_qweight_buf(backend, &lw->ffn.ffn_up);
        void *down_buf = bn_backend_model_qweight_buf(backend, &lw->ffn.ffn_down);
        if (gate_buf && up_buf && down_buf) {
            cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.ffn_norm, dim, c->norm_eps);
            if (gpu->dense_ffn(gpu->ctx, s->xb, gate_buf, up_buf, down_buf,
                               s->xb, dim, hidden_dim,
                               lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type,
                               lw->ffn.ffn_down.type, ffn_plan->activation) == 0) {
                if (bn_transformer_ffn_uses_post_norm(c) &&
                    lw->norm.ffn_post_norm)
                    cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.ffn_post_norm, dim,
                                c->norm_eps);
                bn_transformer_cpu_residual_add(s->x, s->xb, dim);
                return;
            }
        }
    }

    if (cpu_ops->supports_preq8k &&
        ffn_plan->has_gate && !bn_model_gpu(m) && dim % BN_QK_K == 0 &&
        cpu_quant_can_preq8k_pair(lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)) {
        int n_sb = dim / BN_QK_K;
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        cpu_ops->rmsnorm_q8k(s->x, lw->norm.ffn_norm, dim, c->norm_eps,
                             s->xb, s->x_q, q8k_d, q8k_bsums);
        BnMatvecTask ffn[2] = {
             { s->hb,  &lw->ffn.ffn_gate, NULL, 0 },
            { s->hb2, &lw->ffn.ffn_up, NULL, 0 },
        };
        cpu_quant_matvec_batch_preq8k(m, ffn, 2, s->x_q, q8k_d, q8k_bsums,
                                     s->xb);
        fused_gate_up = 1;
    }

    if (!fused_gate_up) {
        cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.ffn_norm, dim, c->norm_eps);
        cpu_debug_dump_array(c, s->xb, "bitnet_ffn_norm", layer, pos);

        if (ffn_plan->has_gate) {
            const BnPreparedWeight *gate_prepared =
                cpu_qweight_prepared(bn_model_backend(m), &lw->ffn.ffn_gate);
            const BnPreparedWeight *up_prepared =
                cpu_qweight_prepared(bn_model_backend(m), &lw->ffn.ffn_up);
            if (!bn_model_gpu(m) && !ffn_plan->scalar_exact_activation &&
                !getenv("BN_CPU_LLAMA_DOT") &&
                !getenv("BN_CPU_LLAMA_Q4_DOT") &&
                ffn_plan->activation == 0 &&
                bn_quant_format_supports_cpu_fused_q4_gateup_silu(
                    lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type) &&
                dim % 32 == 0 &&
                bn_quant_q4_gate_up_silu(s->hb, &lw->ffn.ffn_gate, gate_prepared,
                                         &lw->ffn.ffn_up, up_prepared, s->xb,
                                         s->x_q, bn_model_pool(m)) == 0) {
                ffn_activated = 1;
            } else
            {
                BnMatvecTask ffn[2] = {
                     { s->hb,  &lw->ffn.ffn_gate, NULL, 0 },
                    { s->hb2, &lw->ffn.ffn_up, NULL, 0 },
                };
                cpu_quant_matvec_batch_prepared(m, ffn, 2, s->xb, s->x_q);
            }
            cpu_debug_dump_array_n(hidden_dim, s->hb2, "bitnet_ffn_up",
                                   layer, pos);
            cpu_debug_dump_array_n(hidden_dim, s->hb, "bitnet_ffn_gate",
                                   layer, pos);
        } else {
            BnMatvecTask ffn[1] = {{ s->hb, &lw->ffn.ffn_up, NULL, 0 }};
            cpu_quant_matvec_batch_prepared(m, ffn, 1, s->xb, s->x_q);
        }
    }

    bn_transformer_cpu_apply_ffn_activation(s, ffn_plan, hidden_dim, ffn_activated);
    cpu_debug_dump_array_n(hidden_dim, s->hb, "bitnet_ffn_swiglu",
                           layer, pos);

    if (ffn_plan->has_sub_norm)
        cpu_rmsnorm_model(m, s->hb, s->hb, lw->norm.ffn_sub_norm, hidden_dim, c->norm_eps);

    BnMatvecTask down[1] = {{ s->xb, &lw->ffn.ffn_down, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, down, 1, s->hb, s->x_q);
    if (bn_transformer_ffn_uses_post_norm(c) && lw->norm.ffn_post_norm)
        cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.ffn_post_norm, dim, c->norm_eps);
    cpu_debug_dump_array(c, s->xb, "bitnet_ffn_out", layer, pos);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
    cpu_debug_dump_vector(m, sess, "bitnet_lout", layer, pos);
}
