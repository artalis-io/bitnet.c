#include "transformer_cpu_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_batched_attn_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_plan_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "model_internal.h"
#include "moe.h"
#include "../moe_internal.h"
#include "session.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float cpu_reference_gelu(float x) {
    if (x <= -10.0f)
        return 0.0f;
    if (x >= 10.0f)
        return x;
    float rounded_x = bn_fp16_to_fp32(bn_fp32_to_fp16(x));
    float inner = 0.7978845608028654f * rounded_x *
                  (1.0f + 0.044715f * rounded_x * rounded_x);
    float gelu = 0.5f * rounded_x * (1.0f + tanhf(inner));
    return bn_fp16_to_fp32(bn_fp32_to_fp16(gelu));
}

static const BnCPURuntimePolicy *cpu_runtime(const BnModel *m) {
    return bn_tp_cpu_policy(bn_model_pool(m));
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
        BnTransformerCPUMatvecResourcePolicy resource =
            bn_transformer_cpu_matvec_resource_policy(
                cpu_runtime(m), &m->config, bn_model_backend(m), tasks[i].W);
        if (resource.valid) {
            prepared[i].prepared = resource.prepared;
            prepared[i].flags |= resource.task_flags;
        }
    }
    return prepared;
}

static void cpu_quant_matvec_batch_prepared(const BnModel *m,
                                            const BnMatvecTask *tasks,
                                            int n_tasks,
                                            const float *x,
                                            int8_t *quantized_buf) {
    BnMatvecTask inline_tasks[8];
    BnMatvecTask *prepared_tasks =
        cpu_prepare_matvec_tasks(m, tasks, n_tasks, inline_tasks, 8);
    if (!prepared_tasks) {
        bn_transformer_cpu_quant_matvec_batch(tasks, n_tasks, x, quantized_buf,
                                              bn_model_pool(m));
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
            for (int i = 0; i < n_tasks; i++) {
                BnTransformerCPUMatvecResourcePolicy resource =
                    bn_transformer_cpu_matvec_resource_policy(
                        cpu_runtime(m), &m->config, backend,
                        prepared_tasks[i].W);
                bufs[i] = (void *)resource.gpu_buffer;
            }
            bn_transformer_cpu_quant_matvec_batch_gpu_buffers(
                prepared_tasks, (const void **)bufs, n_tasks, x,
                quantized_buf, bn_model_pool(m), bn_model_gpu(m));
            free(heap_bufs);
            if (prepared_tasks != inline_tasks) free(prepared_tasks);
            return;
        }
    }
    bn_transformer_cpu_quant_matvec_batch(prepared_tasks, n_tasks, x,
                                          quantized_buf, bn_model_pool(m));
    if (prepared_tasks != inline_tasks) free(prepared_tasks);
}

static void cpu_quant_matvec_batch_prepared_kquant(const BnModel *m,
                                                   const BnMatvecTask *tasks,
                                                   int n_tasks,
                                                   const int8_t *quantized,
                                                   const float *scales,
                                                   const int16_t *block_sums,
                                                   const float *x_float) {
    BnTransformerCPUPreparedKQuantInputDispatchPolicy dispatch =
        bn_transformer_cpu_prepared_kquant_input_dispatch_policy(
            bn_model_gpu(m),
            bn_transformer_cpu_float_kquant_fallback_task_flags(&m->config) !=
                0);
    if (dispatch.path ==
        BN_TRANSFORMER_CPU_PREPARED_KQUANT_INPUT_FLOAT_FALLBACK) {
        cpu_quant_matvec_batch_prepared(m, tasks, n_tasks, x_float,
                                        (int8_t *)quantized);
        return;
    }

    BnMatvecTask inline_tasks[8];
    BnMatvecTask *prepared_tasks =
        cpu_prepare_matvec_tasks(m, tasks, n_tasks, inline_tasks, 8);
    if (!prepared_tasks) {
        bn_transformer_cpu_quant_matvec_batch_prepared_kquant_input(
            tasks, n_tasks, quantized, scales, block_sums, x_float,
            bn_model_pool(m));
        return;
    }
    bn_transformer_cpu_quant_matvec_batch_prepared_kquant_input(
        prepared_tasks, n_tasks, quantized, scales, block_sums, x_float,
        bn_model_pool(m));
    if (prepared_tasks != inline_tasks) free(prepared_tasks);
}

static const BnCPUBackendOps *cpu_backend_ops(
    const BnCPURuntimePolicy *runtime) {
    return bn_transformer_cpu_backend_ops(runtime);
}

static void cpu_rmsnorm_reference_order(float *out, const float *x,
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
    if (m && bn_transformer_rmsnorm_uses_reference_order(&m->config)) {
        cpu_rmsnorm_reference_order(out, x, w, size, eps);
        return;
    }
    cpu_backend_ops(cpu_runtime(m))->rmsnorm(out, x, w, size, eps);
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
    const BnCPUBackendOps *ops = cpu_backend_ops(cpu_runtime(m));
    if (gctx->attention_scale == 0.0f)
        gctx->attention_scale =
            bn_transformer_attention_scale(&m->config, gctx->head_size);
    bn_tp_fn attn_fn = bn_transformer_attention_uses_cpu_flash(&m->config)
                           ? ops->flash_gqa
                           : ops->gqa;
    BnTPTask gqa = { attn_fn, gctx, n_heads };
    bn_tp_dispatch(bn_model_pool(m), &gqa, 1);
}

void bn_transformer_batched_attn_dispatch(BnModel *m,
                                          BnBatchedAttnCtx *ctx) {
    const BnCPUBackendOps *ops = cpu_backend_ops(cpu_runtime(m));
    if (ctx->attention_scale == 0.0f)
        ctx->attention_scale =
            bn_transformer_attention_scale(&m->config, ctx->head_size);
    bn_tp_fn fn = bn_transformer_attention_uses_cpu_flash(&m->config)
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

void bn_transformer_cpu_residual_add(const BnCPURuntimePolicy *runtime,
                                     float *x, const float *r, int dim) {
    cpu_backend_ops(runtime)->residual_add(x, r, dim);
}

static void cpu_debug_dump_array_n(const BnCPURuntimePolicy *runtime,
                                   int n_values,
                                   const float *x,
                                   const char *tag,
                                   int layer,
                                   int pos) {
    if (!bn_transformer_cpu_debug_dump_pos_selected(runtime, pos)) return;

    const char *binary_path = bn_transformer_cpu_debug_binary_path(runtime);
    if (binary_path &&
        bn_transformer_cpu_debug_binary_selected(runtime, tag, layer)) {
        FILE *binary = fopen(binary_path, "wb");
        if (binary) {
            (void)fwrite(x, sizeof(*x), (size_t)n_values, binary);
            fclose(binary);
        }
    }

    const char *path = bn_transformer_cpu_debug_dump_path(runtime);
    if (!path) return;

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

void bn_transformer_cpu_debug_dump_values(const BnCPURuntimePolicy *runtime,
                                          const float *values, int count,
                                          const char *tag, int layer,
                                          int pos) {
    if (!values || count <= 0 || !tag) return;
    cpu_debug_dump_array_n(runtime, count, values, tag, layer, pos);
}

static void cpu_debug_dump_attn_weights(const BnCPURuntimePolicy *runtime,
                                        const BnRunState *s,
                                        int n_heads,
                                        int n_kv,
                                        int seq_len,
                                        const char *tag,
                                        int layer,
                                        int pos) {
    if (!bn_transformer_cpu_debug_dump_path(runtime)) return;
    if (n_heads <= 0 || n_kv <= 0 || seq_len <= 0) return;

    int n_values = n_heads * n_kv;
    float *tmp = (float *)malloc((size_t)n_values * sizeof(*tmp));
    if (!tmp) return;
    for (int h = 0; h < n_heads; h++)
        memcpy(tmp + (size_t)h * n_kv, s->att + (size_t)h * seq_len,
               (size_t)n_kv * sizeof(*tmp));
    cpu_debug_dump_array_n(runtime, n_values, tmp, tag, layer, pos);
    free(tmp);
}

static void cpu_debug_dump_heads(const BnCPURuntimePolicy *runtime,
                                 const float *x,
                                 int n_heads,
                                 int head_size,
                                 const char *tag,
                                 int layer,
                                 int pos) {
    if (!bn_transformer_cpu_debug_dump_heads_enabled(runtime)) return;
    char head_tag[96];
    for (int h = 0; h < n_heads; h++) {
        snprintf(head_tag, sizeof(head_tag), "%s_h%d", tag, h);
        cpu_debug_dump_array_n(runtime, head_size,
                               x + (size_t)h * head_size,
                               head_tag, layer, pos);
    }
}

static void cpu_debug_dump_attn_weight_heads(
                                             const BnCPURuntimePolicy *runtime,
                                             const BnRunState *s,
                                             int n_heads,
                                             int n_kv,
                                             int seq_len,
                                             const char *tag,
                                             int layer,
                                             int pos) {
    if (!bn_transformer_cpu_debug_dump_heads_enabled(runtime)) return;
    if (n_heads <= 0 || n_kv <= 0 || seq_len <= 0) return;
    char head_tag[96];
    for (int h = 0; h < n_heads; h++) {
        snprintf(head_tag, sizeof(head_tag), "%s_h%d", tag, h);
        cpu_debug_dump_array_n(runtime, n_kv,
                               s->att + (size_t)h * seq_len,
                               head_tag, layer, pos);
    }
}

static void cpu_debug_dump_array(const BnCPURuntimePolicy *runtime,
                                 const BnConfig *c,
                                 const float *x,
                                 const char *tag,
                                 int layer,
                                 int pos) {
    cpu_debug_dump_array_n(runtime, c->dim, x, tag, layer, pos);
}

static void cpu_debug_dump_vector(const BnModel *m,
                                  const BnSession *sess,
                                  const char *tag,
                                  int layer,
                                  int pos) {
    cpu_debug_dump_array(cpu_runtime(m), &m->config, sess->state.x,
                         tag, layer, pos);
}

static void cpu_debug_dump_moe_route(const BnCPURuntimePolicy *runtime,
                                     const BnMoEState *ms,
                                     int n_active,
                                     int layer,
                                     int pos) {
    const char *path = bn_transformer_cpu_debug_dump_path(runtime);
    if (!path || !ms || !ms->expert_indices || !ms->expert_weights ||
        n_active <= 0 ||
        !bn_transformer_cpu_debug_dump_pos_selected(runtime, pos))
        return;

    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "bitnet_moe_route pos=%d layer=%d indices=", pos, layer);
    for (int i = 0; i < n_active; i++)
        fprintf(f, "%s%d", i ? "," : "", ms->expert_indices[i]);
    fputs(" weights=", f);
    for (int i = 0; i < n_active; i++)
        fprintf(f, "%s%.9g", i ? "," : "", ms->expert_weights[i]);
    fputs(" logits=", f);
    for (int i = 0; i < n_active; i++) {
        int expert = ms->expert_indices[i];
        fprintf(f, "%s%.9g", i ? "," : "",
                expert >= 0 ? ms->router_logits[expert] : -INFINITY);
    }
    fputc('\n', f);
    fclose(f);
}

typedef struct {
    const BnCPURuntimePolicy *runtime;
    int layer;
    int pos;
} BnCPUMoEDumpContext;

static void cpu_debug_dump_moe_checkpoint(void *opaque,
                                          BnMoEObservePoint point,
                                          const float *values,
                                          int n_values) {
    static const char *const tags[] = {
        "bitnet_ffn_norm_2",
        "bitnet_moe_raw",
        "bitnet_ffn_moe",
        "bitnet_ffn_norm_1",
        "bitnet_ffn_geglu",
        "bitnet_ffn_mlp",
        "bitnet_ffn_moe_combined",
        "bitnet_ffn_post_norm",
    };
    BnCPUMoEDumpContext *ctx = (BnCPUMoEDumpContext *)opaque;
    if (!ctx || point < 0 || point >= (int)(sizeof(tags) / sizeof(tags[0])))
        return;
    cpu_debug_dump_array_n(ctx->runtime, n_values, values, tags[point],
                           ctx->layer, ctx->pos);
}

static void cpu_debug_dump_layer_input(const BnModel *m,
                                       const BnSession *sess,
                                       int layer,
                                       int pos) {
    cpu_debug_dump_vector(m, sess, "bitnet_inp", layer, pos);
}

static void cpu_apply_per_layer_input_projection(BnModel *m,
                                                 BnSession *sess,
                                                 BnLayerWeights *lw,
                                                 int layer) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int per_dim = bn_transformer_per_layer_embedding_dim(c);
    float norm_eps = bn_transformer_cpu_norm_epsilon(c);
    if (per_dim <= 0 ||
        !s->per_layer_input || !lw->per_layer.inp_gate.data ||
        !lw->per_layer.proj.data || !lw->per_layer.post_norm)
        return;

    memcpy(s->xb2, s->x, (size_t)c->dim * sizeof(float));

    BnMatvecTask gate[1] = {{ s->hb, &lw->per_layer.inp_gate, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, gate, 1, s->x, s->x_q);
    for (int i = 0; i < per_dim; i++) {
        float g = s->hb[i];
        s->hb[i] = cpu_reference_gelu(g) *
                   s->per_layer_input[(size_t)layer * per_dim + i];
    }

    BnMatvecTask proj[1] = {{ s->x, &lw->per_layer.proj, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, proj, 1, s->hb, s->x_q);
    cpu_rmsnorm_model(m, s->x, s->x, lw->per_layer.post_norm, c->dim,
                      norm_eps);
    bn_transformer_cpu_residual_add(cpu_runtime(m), s->x, s->xb2, c->dim);
}

void bn_transformer_cpu_apply_rope_heads(const BnCPURuntimePolicy *runtime,
                                         float *buf,
                                         int n_heads,
                                         int head_size,
                                         int rope_dims,
                                         const float *rc,
                                         const float *rs) {
    cpu_backend_ops(runtime)->apply_rope_heads(buf, n_heads, head_size,
                                               rope_dims, rc, rs);
}

void bn_transformer_cpu_apply_ffn_activation(const BnCPURuntimePolicy *runtime,
                                             BnRunState *s,
                                             const BnFFNPlan *ffn_plan,
                                             int hidden_dim,
                                             int already_activated) {
    if (already_activated)
        return;
    cpu_backend_ops(runtime)->apply_ffn_activation(s, ffn_plan, hidden_dim);
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
    float norm_eps = bn_transformer_cpu_norm_epsilon(c);
    const BnCPUBackendOps *cpu_ops = cpu_backend_ops(cpu_runtime(m));

    cpu_debug_dump_layer_input(m, sess, l, pos);

    if (is_attn) {
        // ---- Attention block ----

        // KV cache offset: contiguous among attention layers only
        int attn_idx = shape->attn_idx;
        int kv_read_idx = bn_transformer_attention_kv_read_index(c, lw, l);
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
        BnKVMode kv_mode = shape->kv_mode;
        BnTransformerCPUAttentionProjectionTypes attn_types;
        if (!bn_transformer_cpu_resolve_attention_projection_types(
                &attn_types, lw))
            return -1;

        /* Prepared K-quant attn RMSNorm: quantize s->xb once, reuse for Q and K+V */
        int attn_prepared_kquant = 0;
        int attn_prepared_kquant_types[3] = {
            attn_types.q_type, attn_types.k_type, attn_types.v_type
        };
        BnTransformerCPUPreparedKQuantRoutePolicy attn_prepared_kquant_route =
            bn_transformer_cpu_prepared_kquant_route_policy(
                cpu_ops, bn_model_gpu(m), dim,
                attn_prepared_kquant_types, 3, 3);
        int n_sb_attn = bn_transformer_cpu_prepared_kquant_blocks_per_row(dim);
        int n_attn_bsums =
            bn_transformer_cpu_prepared_kquant_block_sums_per_row(n_sb_attn);
        float attn_prepared_kquant_scales[n_sb_attn > 0 ? n_sb_attn : 1];
        int16_t attn_prepared_kquant_block_sums[
            n_attn_bsums > 0 ? n_attn_bsums : 1];
        if (attn_prepared_kquant_route.enabled) {
            cpu_ops->rmsnorm_prepared_kquant(s->x, lw->norm.attn_norm, dim, norm_eps,
                                 s->xb, s->x_q, attn_prepared_kquant_scales,
                                 attn_prepared_kquant_block_sums);
            attn_prepared_kquant = 1;
        } else
        {
            cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.attn_norm, dim, norm_eps);
        }
        cpu_debug_dump_array(cpu_runtime(m), c, s->xb,
                             "bitnet_attn_norm", l, pos);

        /* no-op */

        if (q_gated) {
            // --- Gated Q attention path ---
            float *q_full = s->hb;  // [2*dim]
            float *k_tmp = s->hb2;
            float *v_tmp = s->hb2 + kv_dim;

            // Q+K+V matvecs (reuse cached prepared K-quant activation if available)
            if (bn_transformer_kv_mode_stores_host_float_rows(kv_mode)) {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
                BnMatvecTask qkv[3] = {
                     { q_full,          &lw->attn.wq, NULL, 0 },
                     { key_cache_row,   &lw->attn.wk, NULL, 0 },
                     { value_cache_row, &lw->attn.wv, NULL, 0 },
                };
                if (attn_prepared_kquant)
                    cpu_quant_matvec_batch_prepared_kquant(
                        m, qkv, 3, s->x_q, attn_prepared_kquant_scales,
                        attn_prepared_kquant_block_sums, s->xb);
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
                if (attn_prepared_kquant)
                    cpu_quant_matvec_batch_prepared_kquant(
                        m, qkv, 3, s->x_q, attn_prepared_kquant_scales,
                        attn_prepared_kquant_block_sums, s->xb);
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
                            lw->attn.q_norm + h*qk_stride, head_size, norm_eps);
            } else {
                for (int h = 0; h < n_heads; h++)
                    memcpy(s->q + h * head_size,
                           q_full + h * 2 * head_size,
                           head_size * sizeof(float));
            }
            if (lw->attn.k_norm)
                for (int h = 0; h < n_kv_heads; h++)
                    cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                            lw->attn.k_norm + h*qk_stride, head_size, norm_eps);
            if (shape->value_shares_key)
                cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, norm_eps);

            bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), s->q,
                                                n_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);
            bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), k_tmp,
                                                n_kv_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);

            // Write KV + GQA
            if (bn_transformer_kv_mode_uses_turboquant(kv_mode)) {
                bn_transformer_tq_write_kv(bn_model_tq_state(m), s, k_tmp, v_tmp,
                            n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);
                bn_transformer_tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                n_kv_heads, head_size, kv_mul);
            } else if (bn_transformer_kv_mode_uses_fp16(kv_mode)) {
                bn_transformer_write_kv_fp16(s, loff, cache_pos,
                                             kv_cache_stride, k_tmp, v_tmp,
                                             kv_dim);
            }
            // FP32 path already wrote to cache directly

            if (bn_transformer_kv_mode_uses_cpu_gqa_cache(kv_mode)) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_cache_stride,
                                  c->seq_len,
                                  bn_transformer_attention_scale(c, head_size),
                                  bn_transformer_kv_host_cache_uses_fp16_rows(c) };
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
                cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.attn_sub_norm, dim, norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
            if (attn_plan.use_post_norm)
                cpu_rmsnorm_model(m, s->xb2, s->xb2, lw->norm.attn_post_norm, dim, norm_eps);
            bn_transformer_cpu_residual_add(cpu_runtime(m), s->x, s->xb2, dim);

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
            } else if (!bn_transformer_kv_mode_stores_host_float_rows(kv_mode)) {
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

            cpu_debug_dump_array_n(cpu_runtime(m), q_dim, s->q,
                                   "bitnet_attn_q_raw", l, pos);
            if (has_kv) {
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, k_tmp,
                                       "bitnet_attn_k_raw", l, pos);
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, v_tmp,
                                       "bitnet_attn_v_raw", l, pos);
            }

            if (lw->attn.q_norm)
                for (int h = 0; h < n_heads; h++)
                    cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                            lw->attn.q_norm + h*qk_stride, head_size, norm_eps);
            if (has_kv && lw->attn.k_norm)
                for (int h = 0; h < n_kv_heads; h++)
                    cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                            lw->attn.k_norm + h*qk_stride, head_size, norm_eps);
            if (has_kv && shape->value_shares_key)
                cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, norm_eps);

            bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), s->q,
                                                n_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);
            if (has_kv)
                bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), k_tmp,
                                                    n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

            cpu_debug_dump_array_n(cpu_runtime(m), q_dim, s->q,
                                   "bitnet_attn_q", l, pos);
            if (has_kv) {
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, k_tmp,
                                       "bitnet_attn_k", l, pos);
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, v_tmp,
                                       "bitnet_attn_v", l, pos);
            }
            cpu_debug_dump_heads(cpu_runtime(m), s->q, n_heads, head_size,
                                 "bitnet_attn_q", l, pos);

            if (has_kv && bn_transformer_kv_mode_uses_turboquant(kv_mode)) {
                // TQ write + GQA
                bn_transformer_tq_write_kv(bn_model_tq_state(m), s, k_tmp, v_tmp,
                            n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);
                bn_transformer_tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                n_kv_heads, head_size, kv_mul);
            } else if (has_kv && bn_transformer_kv_mode_uses_fp16(kv_mode)) {
                bn_transformer_write_kv_fp16(s, loff, cache_pos,
                                             kv_cache_stride, k_tmp, v_tmp,
                                             kv_dim);
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size,
                                  kv_cache_stride, c->seq_len,
                                  bn_transformer_attention_scale(c, head_size),
                                  bn_transformer_kv_host_cache_uses_fp16_rows(c) };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            } else {
                // Standard GQA
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, has_kv ? loff : read_loff, pos, n_kv, kv_mul, head_size, kv_cache_stride,
                                  c->seq_len,
                                  bn_transformer_attention_scale(c, head_size),
                                  bn_transformer_kv_host_cache_uses_fp16_rows(c) };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            }

            cpu_debug_dump_attn_weights(cpu_runtime(m), s, n_heads,
                                        pos + 1, c->seq_len,
                                        "bitnet_attn_softmax", l, pos);
            cpu_debug_dump_attn_weight_heads(cpu_runtime(m), s, n_heads,
                                             pos + 1, c->seq_len,
                                             "bitnet_attn_softmax", l, pos);
            cpu_debug_dump_array_n(cpu_runtime(m), q_dim, s->xb,
                                   "bitnet_attn_out", l, pos);
            cpu_debug_dump_heads(cpu_runtime(m), s->xb, n_heads, head_size,
                                 "bitnet_attn_out", l, pos);

            // wo projection (q_dim → dim) + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.attn_sub_norm, q_dim, norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
            if (attn_plan.use_post_norm)
                cpu_rmsnorm_model(m, s->xb2, s->xb2, lw->norm.attn_post_norm, dim, norm_eps);
            bn_transformer_cpu_residual_add(cpu_runtime(m), s->x, s->xb2, dim);

        } else {
            // --- Classic attention path (existing) ---
            float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
            float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;

            if (bn_transformer_kv_mode_uses_turboquant(kv_mode)) {
                // --- TurboQuant KV path ---
                // Use temp buffers for K/V, then quantize into TQ cache
                float *k_tmp = s->hb, *v_tmp = s->hb2;
                BnMatvecTask qkv[3] = {
                     { s->q,  &lw->attn.wq, NULL, 0 },
                     { k_tmp, &lw->attn.wk, NULL, 0 },
                     { v_tmp, &lw->attn.wv, NULL, 0 },
                };
                if (attn_prepared_kquant)
                    cpu_quant_matvec_batch_prepared_kquant(
                        m, qkv, 3, s->x_q, attn_prepared_kquant_scales,
                        attn_prepared_kquant_block_sums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);

                if (lw->attn.q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
                if (lw->attn.k_bias) for (int i = 0; i < kv_dim; i++) k_tmp[i] += lw->attn.k_bias[i];
                if (lw->attn.v_bias) for (int i = 0; i < kv_dim; i++) v_tmp[i] += lw->attn.v_bias[i];

                if (lw->attn.q_norm)
                    for (int h = 0; h < n_heads; h++)
                        cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, norm_eps);
                if (shape->value_shares_key)
                    cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, norm_eps);

                bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), s->q,
                                                    n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), k_tmp,
                                                    n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

                // Write TQ compressed KV
                bn_transformer_tq_write_kv(bn_model_tq_state(m), s, k_tmp, v_tmp,
                            n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);

                // TQ GQA
                bn_transformer_tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                n_kv_heads, head_size, kv_mul);

            } else if (bn_transformer_kv_mode_uses_fp16(kv_mode)) {
                float *k_tmp = s->hb, *v_tmp = s->hb2;
                BnMatvecTask qkv[3] = {
                     { s->q,  &lw->attn.wq, NULL, 0 },
                     { k_tmp, &lw->attn.wk, NULL, 0 },
                     { v_tmp, &lw->attn.wv, NULL, 0 },
                };
                if (attn_prepared_kquant)
                    cpu_quant_matvec_batch_prepared_kquant(
                        m, qkv, 3, s->x_q, attn_prepared_kquant_scales,
                        attn_prepared_kquant_block_sums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);

                if (lw->attn.q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
                if (lw->attn.k_bias) for (int i = 0; i < kv_dim; i++) k_tmp[i] += lw->attn.k_bias[i];
                if (lw->attn.v_bias) for (int i = 0; i < kv_dim; i++) v_tmp[i] += lw->attn.v_bias[i];

                if (lw->attn.q_norm)
                    for (int h = 0; h < n_heads; h++)
                        cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm_model(m, k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, norm_eps);
                if (shape->value_shares_key)
                    cpu_rmsnorm_unit_heads(v_tmp, n_kv_heads, head_size, norm_eps);

                bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), s->q,
                                                    n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), k_tmp,
                                                    n_kv_heads, head_size,
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
                if (attn_prepared_kquant)
                    cpu_quant_matvec_batch_prepared_kquant(
                        m, qkv, 3, s->x_q, attn_prepared_kquant_scales,
                        attn_prepared_kquant_block_sums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);

                if (lw->attn.q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
                if (lw->attn.k_bias) for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
                if (lw->attn.v_bias) for (int i = 0; i < kv_dim; i++) value_cache_row[i] += lw->attn.v_bias[i];

                cpu_debug_dump_array_n(cpu_runtime(m), q_dim, s->q,
                                       "bitnet_attn_q_raw", l, pos);
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, key_cache_row,
                                       "bitnet_attn_k_raw", l, pos);
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, value_cache_row,
                                       "bitnet_attn_v_raw", l, pos);

                if (lw->attn.q_norm)
                    for (int h = 0; h < n_heads; h++)
                        cpu_rmsnorm_model(m, s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm_model(m, key_cache_row + h*head_size, key_cache_row + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, norm_eps);
                if (shape->value_shares_key)
                    cpu_rmsnorm_unit_heads(value_cache_row, n_kv_heads, head_size, norm_eps);

                bn_transformer_cpu_apply_rope_heads(cpu_runtime(m), s->q,
                                                    n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(
                    cpu_runtime(m), key_cache_row, n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

                cpu_debug_dump_array_n(cpu_runtime(m), q_dim, s->q,
                                       "bitnet_attn_q", l, pos);
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, key_cache_row,
                                       "bitnet_attn_k", l, pos);
                cpu_debug_dump_array_n(cpu_runtime(m), kv_dim, value_cache_row,
                                       "bitnet_attn_v", l, pos);
                cpu_debug_dump_heads(cpu_runtime(m), s->q, n_heads, head_size,
                                     "bitnet_attn_q", l, pos);
            }

            // GQA attention (standard path — TQ handled above)
            if (bn_transformer_kv_mode_uses_cpu_gqa_cache(kv_mode)) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_cache_stride,
                                  c->seq_len,
                                  bn_transformer_attention_scale(c, head_size),
                                  bn_transformer_kv_host_cache_uses_fp16_rows(c) };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
                cpu_debug_dump_attn_weights(cpu_runtime(m), s, n_heads,
                                            n_kv, c->seq_len,
                                            "bitnet_attn_softmax", l, pos);
                cpu_debug_dump_attn_weight_heads(cpu_runtime(m), s, n_heads,
                                                n_kv, c->seq_len,
                                                "bitnet_attn_softmax", l, pos);
            }
            cpu_debug_dump_array_n(cpu_runtime(m), q_dim, s->xb,
                                   "bitnet_attn_out", l, pos);
            cpu_debug_dump_heads(cpu_runtime(m), s->xb, n_heads, head_size,
                                 "bitnet_attn_out", l, pos);

            // Attention sub-norm + wo projection + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.attn_sub_norm, dim, norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
            if (attn_plan.use_post_norm)
                cpu_rmsnorm_model(m, s->xb2, s->xb2, lw->norm.attn_post_norm, dim, norm_eps);
            bn_transformer_cpu_residual_add(cpu_runtime(m), s->x, s->xb2, dim);
        }

    } else {
        // ---- SSM block ----
        (void)ssm_plan;
        bn_transformer_cpu_forward_ssm_block(m, sess, lw, l, pos);
        bn_transformer_cpu_residual_add(cpu_runtime(m), s->x, s->xb, dim);
    }

    // ---- FFN block ---- (shared by both layer types)
    /* no-op */
    cpu_debug_dump_vector(m, sess, "bitnet_ffn_inp", l, pos);
    if (ffn_plan.kind == BN_FFN_MOE) {
        // MoE FFN — route, pread, compute, combine
        BnCPUMoEDumpContext dump_ctx = { cpu_runtime(m), l, pos };
        BnMoEObserveFn observe = bn_transformer_cpu_debug_dump_path(
                                     cpu_runtime(m))
                                     ? cpu_debug_dump_moe_checkpoint
                                     : NULL;
        bn_moe_forward_observed(m, sess, lw, l, observe, &dump_ctx);
        cpu_debug_dump_array_n(cpu_runtime(m), moe_plan.n_experts,
                               sess->moe_state->router_logits,
                               "bitnet_moe_logits", l, pos);
        cpu_debug_dump_moe_route(cpu_runtime(m), sess->moe_state,
                                 moe_plan.n_active, l, pos);
        cpu_debug_dump_array(cpu_runtime(m), c,
                             sess->moe_state->expert_out,
                             "bitnet_moe_out", l, pos);
    } else {
        // Dense FFN
        bn_transformer_cpu_forward_ffn_block(m, sess, lw, l, pos, &ffn_plan);
    }

    cpu_apply_per_layer_input_projection(m, sess, lw, l);

    if (ffn_plan.use_layer_output_scale) {
        float scale = lw->norm.layer_output_scale[0];
        for (int i = 0; i < dim; i++)
            s->x[i] *= scale;
    }

    cpu_debug_dump_vector(m, sess, "bitnet_lout", l, pos);

    (void)is_attn; // used only in debug builds


    return 0;
}


void bn_transformer_cpu_forward_ssm_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          int layer,
                                          int pos) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int dim = c->dim;
    BnTransformerSSMShapePolicy ssm_shape;
    if (!bn_transformer_ssm_shape_policy(&ssm_shape, c))
        return;
    int ssm_idx = bn_transformer_ssm_index(c, layer);
    float norm_eps = bn_transformer_cpu_norm_epsilon(c);
    size_t state_per_layer = (size_t)ssm_shape.num_v_heads *
                             ssm_shape.head_k_dim *
                             ssm_shape.head_v_dim;
    float *state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
    size_t conv_per_layer =
        (size_t)(ssm_shape.conv_kernel - 1) * ssm_shape.qkv_dim;
    float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;
    const BnCPUBackendOps *cpu_ops = cpu_backend_ops(cpu_runtime(m));
    BnTransformerCPUSSMProjectionTypes ssm_types;
    if (!bn_transformer_cpu_resolve_ssm_projection_types(&ssm_types, lw))
        return;

    int ssm_prepared_kquant = 0;
    int n_sb_ssm = bn_transformer_cpu_prepared_kquant_blocks_per_row(dim);
    int n_ssm_bsums =
        bn_transformer_cpu_prepared_kquant_block_sums_per_row(n_sb_ssm);
    float ssm_prepared_kquant_scales[n_sb_ssm > 0 ? n_sb_ssm : 1];
    int16_t ssm_prepared_kquant_block_sums[
        n_ssm_bsums > 0 ? n_ssm_bsums : 1];
    int ssm_qz_prepared_kquant_types[2] = {
        ssm_types.qkv_type, ssm_types.z_type
    };
    BnTransformerCPUPreparedKQuantRoutePolicy ssm_qz_prepared_kquant_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            cpu_ops, bn_model_gpu(m), dim, ssm_qz_prepared_kquant_types, 2, 2);
    if (ssm_qz_prepared_kquant_route.enabled) {
        cpu_ops->rmsnorm_prepared_kquant(s->x, lw->norm.attn_norm, dim, norm_eps,
                             s->xb, s->x_q, ssm_prepared_kquant_scales,
                             ssm_prepared_kquant_block_sums);
        ssm_prepared_kquant = 1;
    } else
    {
        cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.attn_norm, dim, norm_eps);
    }

    cpu_debug_dump_array_n(cpu_runtime(m), dim, s->xb,
                           "bitnet_ssm_norm", layer, pos);

    float *qkv = s->hb;
    float *z = s->hb2;
    BnMatvecTask qz_tasks[2] = {
         { qkv, &lw->ssm.wqkv, NULL, 0 },
        { z,   &lw->ssm.wz, NULL, 0 },
    };
    if (ssm_prepared_kquant)
        cpu_quant_matvec_batch_prepared_kquant(m, qz_tasks, 2, s->x_q,
                                               ssm_prepared_kquant_scales,
                                               ssm_prepared_kquant_block_sums,
                                               s->xb);
    else
        cpu_quant_matvec_batch_prepared(m, qz_tasks, 2, s->xb, s->x_q);
    cpu_debug_dump_array_n(cpu_runtime(m), ssm_shape.qkv_dim, qkv,
                           "bitnet_ssm_qkv", layer, pos);
    cpu_debug_dump_array_n(cpu_runtime(m),
                           ssm_shape.num_v_heads * ssm_shape.head_v_dim, z,
                           "bitnet_ssm_z", layer, pos);

    BnSSMConvCtx conv_ctx = {
        qkv, conv_state, lw->ssm.ssm_conv1d,
        ssm_shape.qkv_dim, ssm_shape.conv_kernel
    };
    BnTPTask conv_task = {
        bn_transformer_cpu_ssm_conv_silu_op(cpu_ops),
        &conv_ctx, ssm_shape.qkv_dim
    };
    bn_tp_dispatch(bn_model_pool(m), &conv_task, 1);
    cpu_debug_dump_array_n(cpu_runtime(m), ssm_shape.qkv_dim, qkv,
                           "bitnet_ssm_conv", layer, pos);

    float *q_raw = qkv;
    float *k_raw = qkv + ssm_shape.key_dim;
    float *v_raw = qkv + 2 * ssm_shape.key_dim;

    BnSSML2NormCtx norm_ctx = {
        q_raw, k_raw, norm_eps, ssm_shape.head_k_dim
    };
    BnTPTask norm_task = {
        bn_transformer_cpu_ssm_l2norm_op(cpu_ops),
        &norm_ctx, ssm_shape.num_k_heads
    };
    bn_tp_dispatch(bn_model_pool(m), &norm_task, 1);
    cpu_debug_dump_array_n(cpu_runtime(m), ssm_shape.qkv_dim, qkv,
                           "bitnet_ssm_qkv_norm", layer, pos);

    if (ssm_shape.num_v_heads > 8192 || ssm_shape.head_v_dim > 8192) {
        SH_LOG_ERROR("SSM dimensions too large for stack VLAs");
        return;
    }
    float alpha_arr[ssm_shape.num_v_heads];
    float beta_arr[ssm_shape.num_v_heads];
    BnMatvecTask ab[2] = {
         { alpha_arr, &lw->ssm.ssm_alpha, NULL, 0 },
        { beta_arr,  &lw->ssm.ssm_beta, NULL, 0 },
    };
    int ssm_ab_prepared_kquant_types[2] = {
        ssm_types.alpha_type, ssm_types.beta_type
    };
    BnTransformerCPUPreparedKQuantRoutePolicy ssm_ab_prepared_kquant_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            cpu_ops, bn_model_gpu(m), dim, ssm_ab_prepared_kquant_types, 2, 2);
    if (ssm_prepared_kquant && ssm_ab_prepared_kquant_route.enabled) {
        cpu_quant_matvec_batch_prepared_kquant(m, ab, 2, s->x_q,
                                               ssm_prepared_kquant_scales,
                                               ssm_prepared_kquant_block_sums,
                                               s->xb);
    } else {
        cpu_quant_matvec_batch_prepared(m, ab, 2, s->xb, s->x_q);
    }
    cpu_debug_dump_array_n(cpu_runtime(m), ssm_shape.num_v_heads, alpha_arr,
                           "bitnet_ssm_alpha", layer, pos);
    cpu_debug_dump_array_n(cpu_runtime(m), ssm_shape.num_v_heads, beta_arr,
                           "bitnet_ssm_beta", layer, pos);

    for (int h = 0; h < ssm_shape.num_v_heads; h++) {
        float dt = alpha_arr[h] + lw->ssm.ssm_dt_bias[h];
        float dt_sp = (dt > 20.0f) ? dt : logf(1.0f + expf(dt));
        alpha_arr[h] = dt_sp;
        beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
    }
    cpu_debug_dump_array_n(cpu_runtime(m), ssm_shape.num_v_heads, alpha_arr,
                           "bitnet_ssm_softplus", layer, pos);
    cpu_debug_dump_array_n(cpu_runtime(m), ssm_shape.num_v_heads, beta_arr,
                           "bitnet_ssm_beta_sigmoid", layer, pos);
    for (int h = 0; h < ssm_shape.num_v_heads; h++)
        alpha_arr[h] = expf(alpha_arr[h] * lw->ssm.ssm_a[h]);

    float *out = s->xb2;
    float q_scale = 1.0f / sqrtf((float)ssm_shape.head_k_dim);
    BnSSMDeltaCtx delta_ctx = {
        state, out, q_raw, k_raw, v_raw,
        alpha_arr, beta_arr,
        ssm_shape.num_k_heads, ssm_shape.head_k_dim,
        ssm_shape.head_v_dim, q_scale
    };
    BnTPTask delta_task = {
        bn_transformer_cpu_ssm_delta_op(cpu_ops),
        &delta_ctx, ssm_shape.num_v_heads
    };
    bn_tp_dispatch(bn_model_pool(m), &delta_task, 1);
    cpu_debug_dump_array_n(cpu_runtime(m),
                           ssm_shape.num_v_heads * ssm_shape.head_v_dim, out,
                           "bitnet_ssm_delta", layer, pos);

    BnSSMGateCtx gate_ctx = {
        out, z, lw->ssm.ssm_norm, norm_eps, ssm_shape.head_v_dim
    };
    BnTPTask gate_task = {
        bn_transformer_cpu_ssm_gate_op(cpu_ops),
        &gate_ctx, ssm_shape.num_v_heads
    };
    bn_tp_dispatch(bn_model_pool(m), &gate_task, 1);
    cpu_debug_dump_array_n(cpu_runtime(m),
                           ssm_shape.num_v_heads * ssm_shape.head_v_dim, out,
                           "bitnet_ssm_gate", layer, pos);

    BnMatvecTask proj[1] = {{ s->xb, &lw->ssm.ssm_out, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, proj, 1, out, s->x_q);
    cpu_debug_dump_array_n(cpu_runtime(m), dim, s->xb,
                           "bitnet_ssm_out", layer, pos);
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
    float norm_eps = bn_transformer_cpu_norm_epsilon(c);
    int ffn_activated = 0;
    int fused_gate_up = 0;
    const BnCPUBackendOps *cpu_ops = cpu_backend_ops(cpu_runtime(m));
    BnTransformerCPUFFNProjectionTypes ffn_types;
    if (!bn_transformer_cpu_resolve_ffn_projection_types(&ffn_types, lw))
        return;

    BnGPUBackend *gpu = bn_model_gpu(m);
    if (bn_transformer_cpu_gpu_dense_ffn_fast_path_available(gpu, ffn_plan)) {
        const BnBackendModel *backend = bn_model_backend(m);
        BnTransformerCPUMatvecResourcePolicy gate_resource =
            bn_transformer_cpu_matvec_resource_policy(
                cpu_runtime(m), c, backend, &lw->ffn.ffn_gate);
        BnTransformerCPUMatvecResourcePolicy up_resource =
            bn_transformer_cpu_matvec_resource_policy(
                cpu_runtime(m), c, backend, &lw->ffn.ffn_up);
        BnTransformerCPUMatvecResourcePolicy down_resource =
            bn_transformer_cpu_matvec_resource_policy(
                cpu_runtime(m), c, backend, &lw->ffn.ffn_down);
        void *gate_buf = (void *)gate_resource.gpu_buffer;
        void *up_buf = (void *)up_resource.gpu_buffer;
        void *down_buf = (void *)down_resource.gpu_buffer;
        if (gate_buf && up_buf && down_buf) {
            cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.ffn_norm, dim, norm_eps);
            if (bn_transformer_gpu_dense_ffn_fast_path_run(
                    gpu, s->xb, gate_buf, up_buf, down_buf, s->xb,
                    dim, hidden_dim, ffn_types.gate_type,
                    ffn_types.up_type, ffn_types.down_type,
                    ffn_plan->activation) == 0) {
                if (ffn_plan->use_post_norm)
                    cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.ffn_post_norm, dim,
                                norm_eps);
                bn_transformer_cpu_residual_add(
                    cpu_runtime(m), s->x, s->xb, dim);
                return;
            }
        }
    }

    int ffn_prepared_kquant_types[2] = {
        ffn_types.gate_type, ffn_types.up_type
    };
    BnTransformerCPUPreparedKQuantRoutePolicy ffn_prepared_kquant_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            cpu_ops, gpu, dim, ffn_prepared_kquant_types, 2, 2);
    if (ffn_plan->has_gate && ffn_prepared_kquant_route.enabled) {
        int n_sb = bn_transformer_cpu_prepared_kquant_blocks_per_row(dim);
        int n_bsums =
            bn_transformer_cpu_prepared_kquant_block_sums_per_row(n_sb);
        float ffn_prepared_kquant_scales[n_sb];
        int16_t ffn_prepared_kquant_block_sums[n_bsums];
        cpu_ops->rmsnorm_prepared_kquant(s->x, lw->norm.ffn_norm, dim, norm_eps,
                             s->xb, s->x_q, ffn_prepared_kquant_scales,
                             ffn_prepared_kquant_block_sums);
        BnMatvecTask ffn[2] = {
             { s->hb,  &lw->ffn.ffn_gate, NULL, 0 },
            { s->hb2, &lw->ffn.ffn_up, NULL, 0 },
        };
        cpu_quant_matvec_batch_prepared_kquant(m, ffn, 2, s->x_q,
                                               ffn_prepared_kquant_scales,
                                               ffn_prepared_kquant_block_sums,
                                               s->xb);
        fused_gate_up = 1;
    }

    if (!fused_gate_up) {
        cpu_rmsnorm_model(m, s->xb, s->x, lw->norm.ffn_norm, dim, norm_eps);
        cpu_debug_dump_array(cpu_runtime(m), c, s->xb,
                             "bitnet_ffn_norm", layer, pos);

        if (ffn_plan->has_gate) {
            const BnPreparedWeight *gate_prepared =
                bn_transformer_cpu_matvec_resource_policy(
                    cpu_runtime(m), &m->config, bn_model_backend(m),
                    &lw->ffn.ffn_gate).prepared;
            const BnPreparedWeight *up_prepared =
                bn_transformer_cpu_matvec_resource_policy(
                    cpu_runtime(m), &m->config, bn_model_backend(m),
                    &lw->ffn.ffn_up).prepared;
            if (bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
                    cpu_runtime(m), gpu, ffn_plan, dim,
                    ffn_types.gate_type, ffn_types.up_type) &&
                bn_transformer_cpu_fused_kquant_gateup_silu(
                    s->hb, &lw->ffn.ffn_gate, gate_prepared,
                    &lw->ffn.ffn_up, up_prepared, s->xb, s->x_q,
                    bn_model_pool(m)) == 0) {
                ffn_activated = 1;
            } else
            {
                BnMatvecTask ffn[2] = {
                     { s->hb,  &lw->ffn.ffn_gate, NULL, 0 },
                    { s->hb2, &lw->ffn.ffn_up, NULL, 0 },
                };
                cpu_quant_matvec_batch_prepared(m, ffn, 2, s->xb, s->x_q);
            }
            cpu_debug_dump_array_n(cpu_runtime(m), hidden_dim, s->hb2,
                                   "bitnet_ffn_up",
                                   layer, pos);
            cpu_debug_dump_array_n(cpu_runtime(m), hidden_dim, s->hb,
                                   "bitnet_ffn_gate",
                                   layer, pos);
        } else {
            BnMatvecTask ffn[1] = {{ s->hb, &lw->ffn.ffn_up, NULL, 0 }};
            cpu_quant_matvec_batch_prepared(m, ffn, 1, s->xb, s->x_q);
        }
    }

    bn_transformer_cpu_apply_ffn_activation(
        cpu_runtime(m), s, ffn_plan, hidden_dim, ffn_activated);
    cpu_debug_dump_array_n(cpu_runtime(m), hidden_dim, s->hb,
                           "bitnet_ffn_swiglu",
                           layer, pos);

    if (ffn_plan->has_sub_norm)
        cpu_rmsnorm_model(m, s->hb, s->hb, lw->norm.ffn_sub_norm, hidden_dim, norm_eps);

    BnMatvecTask down[1] = {{ s->xb, &lw->ffn.ffn_down, NULL, 0 }};
    cpu_quant_matvec_batch_prepared(m, down, 1, s->hb, s->x_q);
    if (ffn_plan->use_post_norm)
        cpu_rmsnorm_model(m, s->xb, s->xb, lw->norm.ffn_post_norm, dim, norm_eps);
    cpu_debug_dump_array(cpu_runtime(m), c, s->xb,
                         "bitnet_ffn_out", layer, pos);
    bn_transformer_cpu_residual_add(cpu_runtime(m), s->x, s->xb, dim);
}
