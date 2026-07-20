#include "gpu_internal.h"
#include "../gpu_shader.h"
#include "backend_model.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_plan_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "moe.h"
#include "platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fallback_rmsnorm(float *out,
                             const float *x,
                             const float *w,
                             int size,
                             float eps) {
    bn_transformer_rmsnorm_default(out, x, w, size, eps);
}

static const BnPreparedWeight *fallback_cpu_prepared_qweight(
    const BnModel *m,
    const BnQWeight *w) {
    if (!bn_transformer_cpu_prepared_qweights_enabled())
        return NULL;
    return bn_backend_model_prepared_qweight(bn_model_backend(m), w);
}

static void fallback_cpu_matvec_batch(const BnModel *m,
                                      const BnMatvecTask *tasks,
                                      int n_tasks,
                                      const float *x,
                                      int8_t *quantized_buf) {
    BnMatvecTask inline_tasks[8];
    BnMatvecTask *prepared = inline_tasks;
    if (n_tasks > 8) {
        prepared = (BnMatvecTask *)malloc((size_t)n_tasks * sizeof(*prepared));
        if (!prepared) {
            bn_transformer_cpu_quant_matvec_batch(tasks, n_tasks, x,
                                                  quantized_buf,
                                                  bn_model_pool(m));
            return;
        }
    }
    for (int i = 0; i < n_tasks; i++) {
        prepared[i] = tasks[i];
        prepared[i].prepared =
            fallback_cpu_prepared_qweight(m, tasks[i].W);
        prepared[i].flags |=
            bn_transformer_cpu_force_float_kquant_task_flags(&m->config);
    }
    bn_transformer_cpu_quant_matvec_batch(prepared, n_tasks, x, quantized_buf,
                                          bn_model_pool(m));
    if (prepared != inline_tasks)
        free(prepared);
}

static void fallback_cpu_matvec(const BnModel *m,
                                float *out,
                                const BnQWeight *W,
                                const float *x,
                                int8_t *quantized_buf) {
    BnMatvecTask task = { out, W, NULL, 0 };
    fallback_cpu_matvec_batch(m, &task, 1, x, quantized_buf);
}

int bn_transformer_gpu_fallback_ssm_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_transformer_cpu_forward_ssm_block(m, sess, lw, layer);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
    if (lw->moe.router_weight)
        bn_moe_forward(m, sess, lw, layer);
    else
        bn_transformer_cpu_forward_ffn_block(m, sess, lw, layer, sess->pos, NULL);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_moe_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_model_set_gpu_disabled(m, 1);
    bn_moe_forward(m, sess, lw, layer);
    bn_model_set_gpu_disabled(m, 0);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_cpu_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_model_set_gpu_disabled(m, 1);
    int rc = bn_transformer_cpu_forward_layer(m, sess, layer, pos, cache_pos,
                                              rope_dims, rope_cos, rope_sin);
    bn_model_set_gpu_disabled(m, 0);
    if (rc != 0)
        return -1;
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_cpu_attention(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.q_gated ||
        c->kv_tq_bits > 0 || c->kv_f16 || lw->norm.attn_sub_norm)
        return -1;

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;

    int head_size = shape.head_size;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    size_t loff = (size_t)shape.attn_idx * c->seq_len * c->kv_dim;
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim, c->norm_eps);
    {
        BnMatvecTask qkv[3] = {
            { s->q, &lw->attn.wq, NULL, 0 },
            { key_cache_row, &lw->attn.wk, NULL, 0 },
            { value_cache_row, &lw->attn.wv, NULL, 0 },
        };
        fallback_cpu_matvec_batch(m, qkv, 3, s->xb, s->x_q);
    }

    if (lw->attn.q_bias) {
        for (int i = 0; i < shape.q_dim; i++) s->q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++)
            value_cache_row[i] += lw->attn.v_bias[i];
    }
    if (lw->attn.q_norm) {
        for (int h = 0; h < c->n_heads; h++)
            fallback_rmsnorm(s->q + (size_t)h * head_size,
                             s->q + (size_t)h * head_size,
                             lw->attn.q_norm + (size_t)h * shape.qk_stride,
                             head_size, c->norm_eps);
    }
    if (lw->attn.k_norm) {
        for (int h = 0; h < n_kv_heads; h++)
            fallback_rmsnorm(key_cache_row + (size_t)h * head_size,
                             key_cache_row + (size_t)h * head_size,
                             lw->attn.k_norm + (size_t)h * shape.qk_stride,
                             head_size, c->norm_eps);
    }

    bn_transformer_cpu_apply_rope_heads(s->q, c->n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, c->n_heads, kv_mul);

    {
        BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
        fallback_cpu_matvec_batch(m, wo, 1, s->xb, s->x_q);
    }
    bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

    size_t kv_row_bytes = (size_t)c->kv_dim * sizeof(float);
    size_t kv_row_off = (loff + (size_t)cache_pos * (size_t)c->kv_dim) *
                        sizeof(float);
    if (bn_transformer_gpu_write_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, key_cache_row, kv_row_bytes,
            kv_row_off) != 0 ||
        bn_transformer_gpu_write_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, value_cache_row, kv_row_bytes,
            kv_row_off) != 0)
        return -1;

    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

static void fallback_cpu_forward_ffn_from_xb(BnModel *m,
                                             BnSession *sess,
                                             BnLayerWeights *lw,
                                             const BnFFNPlan *ffn_plan,
                                             int dim) {
    BnRunState *s = &sess->state;
    int hidden_dim = ffn_plan->hidden_dim;
    if (ffn_plan->has_gate) {
        BnMatvecTask ffn[2] = {
            { s->hb, &lw->ffn.ffn_gate, NULL, 0 },
            { s->hb2, &lw->ffn.ffn_up, NULL, 0 },
        };
        fallback_cpu_matvec_batch(m, ffn, 2, s->xb, s->x_q);
    } else {
        fallback_cpu_matvec(m, s->hb, &lw->ffn.ffn_up, s->xb, s->x_q);
    }
    bn_transformer_cpu_apply_ffn_activation(s, ffn_plan, hidden_dim, 0);
    if (ffn_plan->has_sub_norm)
        fallback_rmsnorm(s->hb, s->hb, lw->norm.ffn_sub_norm,
                         hidden_dim, m->config.norm_eps);
    fallback_cpu_matvec(m, s->xb, &lw->ffn.ffn_down, s->hb, s->x_q);
    if (bn_transformer_ffn_uses_post_norm(&m->config) &&
        lw->norm.ffn_post_norm)
        fallback_rmsnorm(s->xb, s->xb, lw->norm.ffn_post_norm,
                         dim, m->config.norm_eps);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
}

int bn_transformer_gpu_fallback_cpu_ffn(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    fallback_cpu_forward_ffn_from_xb(m, sess, lw, ffn_plan, dim);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_cpu_ffn_down(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int down_input_buf,
    int hidden_dim,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    if (bn_transformer_gpu_read_activation_buf(
            gpu, down_input_buf, s->hb,
            (size_t)hidden_dim * sizeof(float)) != 0)
        return -1;
    fallback_cpu_matvec(m, s->xb, &lw->ffn.ffn_down, s->hb, s->x_q);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_debug_compare_ffn_down(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int down_input_buf,
    int hidden_dim,
    int dim) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_activation_buf(
            gpu, down_input_buf, s->hb,
            (size_t)hidden_dim * sizeof(float)) != 0)
        return -1;
    if (bn_transformer_gpu_read_xb2(gpu, s->xb2,
                                    (size_t)dim * sizeof(float)) != 0)
        return -1;

    fallback_cpu_matvec(m, s->xb, &lw->ffn.ffn_down, s->hb, s->x_q);

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < dim; i++) {
        float diff = fabsf(s->xb2[i] - s->xb[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] ffn_down_compare layer=%d pos=%d type=%d "
            "rows=%d cols=%d max_abs=%.9g max_i=%d cpu=%.9g gpu=%.9g "
            "mean_abs=%.9g rms=%.9g\n",
            layer, pos, lw->ffn.ffn_down.type, lw->ffn.ffn_down.rows,
            lw->ffn.ffn_down.cols, max_abs, max_i, s->xb[max_i],
            s->xb2[max_i], sum_abs / (double)dim,
            sqrt(sum_sq / (double)dim));
    return 0;
}

static void debug_compare_vec(const char *label,
                              int layer,
                              int pos,
                              const float *cpu,
                              const float *gpu,
                              int n) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] %s layer=%d pos=%d "
            "max_abs=%.9g max_i=%d cpu=%.9g gpu=%.9g "
            "mean_abs=%.9g rms=%.9g\n",
            label, layer, pos, max_abs, max_i, cpu[max_i], gpu[max_i],
           sum_abs / (double)n, sqrt(sum_sq / (double)n));
}

static const BnPreparedWeight *debug_prepared_qweight(BnModel *m,
                                                      const BnQWeight *w) {
    if (!m || !w || !bn_transformer_cpu_prepared_qweights_enabled())
        return NULL;
    return bn_backend_model_prepared_qweight(bn_model_backend(m), w);
}

static void debug_quant_matvec_prepared(BnModel *m,
                                        float *out,
                                        const BnQWeight *W,
                                        const float *x,
                                        int8_t *quantized_buf) {
    bn_transformer_cpu_quant_matvec_prepared_flags(
        out, W, debug_prepared_qweight(m, W), x, quantized_buf,
        bn_model_pool(m), 0);
}

static void debug_compare_native_quant_activation(const BnGPUBackend *gpu,
                                                 int layer,
                                                 int pos,
                                                 const float *x,
                                                 int cols) {
    if (!gpu || !x || cols <= 0 || (cols % 32) != 0 ||
        !bn_transformer_cpu_has_native_quant_activation())
        return;
    int n_blocks = cols / 32;
    int8_t *cpu_q = (int8_t *)malloc((size_t)cols);
    int8_t *gpu_q = (int8_t *)malloc((size_t)cols);
    float *cpu_scales = (float *)malloc((size_t)n_blocks * sizeof(float));
    float *gpu_scales = (float *)malloc((size_t)n_blocks * sizeof(float));
    if (!cpu_q || !gpu_q || !cpu_scales || !gpu_scales) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        return;
    }

    if (bn_transformer_cpu_quantize_native_logits_refine_activation(
            x, cpu_q, cpu_scales, cols) != 0) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        return;
    }
    if (bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_DEBUG_BUF_Q8_ACT, gpu_q, (size_t)cols) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_DEBUG_BUF_Q8_SCALE, gpu_scales,
            (size_t)n_blocks * sizeof(float)) != 0) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        return;
    }

    int max_q_abs = 0;
    int max_q_i = 0;
    long long sum_q_abs = 0;
    int n_q_diff = 0;
    for (int i = 0; i < cols; i++) {
        int diff = (int)gpu_q[i] - (int)cpu_q[i];
        int ad = diff < 0 ? -diff : diff;
        if (ad > max_q_abs) {
            max_q_abs = ad;
            max_q_i = i;
        }
        if (ad) n_q_diff++;
        sum_q_abs += ad;
    }

    double sum_scale_abs = 0.0;
    double sum_scale_sq = 0.0;
    float max_scale_abs = 0.0f;
    int max_scale_i = 0;
    for (int i = 0; i < n_blocks; i++) {
        float diff = fabsf(gpu_scales[i] - cpu_scales[i]);
        sum_scale_abs += (double)diff;
        sum_scale_sq += (double)diff * (double)diff;
        if (diff > max_scale_abs) {
            max_scale_abs = diff;
            max_scale_i = i;
        }
    }

    fprintf(stderr,
            "[bn:gpu:debug] q8_act_compare layer=%d pos=%d "
            "q_max_abs=%d q_max_i=%d cpu_q=%d gpu_q=%d "
            "q_diff_count=%d q_mean_abs=%.9g "
            "scale_max_abs=%.9g scale_max_i=%d cpu_scale=%.9g "
            "gpu_scale=%.9g scale_mean_abs=%.9g scale_rms=%.9g\n",
            layer, pos, max_q_abs, max_q_i, (int)cpu_q[max_q_i],
            (int)gpu_q[max_q_i], n_q_diff,
            (double)sum_q_abs / (double)cols, max_scale_abs, max_scale_i,
            cpu_scales[max_scale_i], gpu_scales[max_scale_i],
            sum_scale_abs / (double)n_blocks,
            sqrt(sum_scale_sq / (double)n_blocks));

    free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
}

int bn_transformer_gpu_debug_compare_ffn_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    const float *next_norm,
    int layer,
    int pos,
    int dim) {
    BnRunState *s = &sess->state;
    float *cpu_x_in = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_xb_in = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_hb = NULL;
    float *cpu_hb2 = NULL;
    float *gpu_hb = NULL;
    int hidden_dim = ffn_plan ? ffn_plan->hidden_dim : 0;
    if (hidden_dim > 0) {
        cpu_hb = (float *)malloc((size_t)hidden_dim * sizeof(float));
        cpu_hb2 = (float *)malloc((size_t)hidden_dim * sizeof(float));
        gpu_hb = (float *)malloc((size_t)hidden_dim * sizeof(float));
    }
    if (!cpu_x_in || !cpu_xb_in || !cpu_x || !cpu_xb || !gpu_x || !gpu_xb ||
        (hidden_dim > 0 && (!cpu_hb || !cpu_hb2 || !gpu_hb))) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
        free(cpu_hb);
        free(cpu_hb2);
        free(gpu_hb);
        return -1;
    }

    memcpy(cpu_x_in, s->x, (size_t)dim * sizeof(float));
    memcpy(cpu_xb_in, s->xb, (size_t)dim * sizeof(float));

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
        free(cpu_hb);
        free(cpu_hb2);
        free(gpu_hb);
        return -1;
    }

    memcpy(s->x, cpu_x_in, (size_t)dim * sizeof(float));
    memcpy(s->xb, cpu_xb_in, (size_t)dim * sizeof(float));
    if (hidden_dim > 0) {
        if (ffn_plan->has_gate) {
            debug_quant_matvec_prepared(m, cpu_hb, &lw->ffn.ffn_gate,
                                        s->xb, s->x_q);
            debug_quant_matvec_prepared(m, cpu_hb2, &lw->ffn.ffn_up,
                                        s->xb, s->x_q);
            for (int i = 0; i < hidden_dim; i++) {
                float g = cpu_hb[i];
                if (bn_transformer_gpu_ffn_activation_kind(
                        ffn_plan->activation) == BN_GPU_IR_ACTIVATION_RELU2) {
                    float r = g > 0.0f ? g : 0.0f;
                    cpu_hb[i] = r * r * cpu_hb2[i];
                } else {
                    cpu_hb[i] = (g / (1.0f + expf(-g))) * cpu_hb2[i];
                }
            }
        } else {
            debug_quant_matvec_prepared(m, cpu_hb, &lw->ffn.ffn_up,
                                        s->xb, s->x_q);
            for (int i = 0; i < hidden_dim; i++) {
                float v = cpu_hb[i];
                if (bn_transformer_gpu_ffn_activation_kind(
                        ffn_plan->activation) == BN_GPU_IR_ACTIVATION_RELU2) {
                    float r = v > 0.0f ? v : 0.0f;
                    cpu_hb[i] = r * r;
                } else {
                    cpu_hb[i] = v / (1.0f + expf(-v));
                }
            }
        }
    }
    fallback_cpu_forward_ffn_from_xb(m, sess, lw, ffn_plan, dim);
    memcpy(cpu_x, s->x, (size_t)dim * sizeof(float));
    if (next_norm)
        fallback_rmsnorm(cpu_xb, cpu_x, next_norm, dim, m->config.norm_eps);

    if ((hidden_dim > 0 &&
         bn_transformer_gpu_read_activation_buf(
             gpu, BN_GPU_VALUE_HB, gpu_hb,
             (size_t)hidden_dim * sizeof(float)) != 0) ||
        bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        (next_norm && bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                                 (size_t)dim * sizeof(float)) != 0)) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
        free(cpu_hb);
        free(cpu_hb2);
        free(gpu_hb);
        return -1;
    }
    if (hidden_dim > 0)
        debug_compare_vec("ffn_hidden_compare", layer, pos, cpu_hb, gpu_hb,
                          hidden_dim);
    debug_compare_vec("ffn_state_compare", layer, pos, cpu_x, gpu_x, dim);
    if (next_norm)
        debug_compare_vec("ffn_next_norm_compare", layer, pos, cpu_xb, gpu_xb,
                          dim);

    memcpy(s->x, gpu_x, (size_t)dim * sizeof(float));
    free(cpu_x_in);
    free(cpu_xb_in);
    free(cpu_x);
    free(cpu_xb);
    free(gpu_x);
    free(gpu_xb);
    free(cpu_hb);
    free(cpu_hb2);
    free(gpu_hb);
    return 0;
}

int bn_transformer_gpu_debug_compare_attention(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.q_gated ||
        c->kv_tq_bits > 0 || c->kv_f16 || lw->norm.attn_sub_norm)
        return -1;

    float *cpu_in = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_x = (float *)malloc((size_t)dim * sizeof(float));
    if (!cpu_in || !gpu_x) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }
    memcpy(cpu_in, s->x, (size_t)dim * sizeof(float));

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }
    if (bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }

    int head_size = shape.head_size;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    size_t loff = (size_t)shape.attn_idx * c->seq_len * c->kv_dim;
    size_t kv_bytes = (size_t)n_kv * c->kv_dim * sizeof(float);
    size_t kv_off = loff * sizeof(float);

    if (bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, s->key_cache + loff, kv_bytes,
            kv_off) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, s->value_cache + loff, kv_bytes,
            kv_off) != 0) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }

    memcpy(s->x, cpu_in, (size_t)dim * sizeof(float));
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim, c->norm_eps);
    fallback_cpu_matvec(m, s->q, &lw->attn.wq, s->xb, s->x_q);
    fallback_cpu_matvec(m, key_cache_row, &lw->attn.wk, s->xb, s->x_q);
    fallback_cpu_matvec(m, value_cache_row, &lw->attn.wv, s->xb, s->x_q);

    if (lw->attn.q_bias) {
        for (int i = 0; i < shape.q_dim; i++) s->q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++)
            value_cache_row[i] += lw->attn.v_bias[i];
    }
    if (lw->attn.q_norm) {
        for (int h = 0; h < c->n_heads; h++)
            fallback_rmsnorm(s->q + (size_t)h * head_size,
                             s->q + (size_t)h * head_size,
                             lw->attn.q_norm + (size_t)h * shape.qk_stride,
                             head_size, c->norm_eps);
    }
    if (lw->attn.k_norm) {
        for (int h = 0; h < n_kv_heads; h++)
            fallback_rmsnorm(key_cache_row + (size_t)h * head_size,
                             key_cache_row + (size_t)h * head_size,
                             lw->attn.k_norm + (size_t)h * shape.qk_stride,
                             head_size, c->norm_eps);
    }

    bn_transformer_cpu_apply_rope_heads(s->q, c->n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, c->n_heads, kv_mul);

    fallback_cpu_matvec(m, s->xb2, &lw->attn.wo, s->xb, s->x_q);
    bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < dim; i++) {
        float diff = fabsf(gpu_x[i] - s->x[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] attention_compare layer=%d pos=%d "
            "max_abs=%.9g max_i=%d cpu=%.9g gpu=%.9g "
            "mean_abs=%.9g rms=%.9g\n",
            layer, pos, max_abs, max_i, s->x[max_i], gpu_x[max_i],
            sum_abs / (double)dim, sqrt(sum_sq / (double)dim));

    free(cpu_in);
    free(gpu_x);
    return 0;
}

int bn_transformer_gpu_debug_compare_gqa(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.q_gated || shape.q_wide ||
        c->kv_tq_bits > 0 || c->kv_f16 ||
        lw->attn.q_norm || lw->attn.k_norm || lw->norm.attn_sub_norm)
        return -1;

    float *cpu_in = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    if (!cpu_in || !gpu_xb) {
        free(cpu_in);
        free(gpu_xb);
        return -1;
    }
    memcpy(cpu_in, s->x, (size_t)dim * sizeof(float));

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                   (size_t)dim * sizeof(float)) != 0) {
        free(cpu_in);
        free(gpu_xb);
        return -1;
    }

    int head_size = shape.head_size;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    size_t loff = (size_t)shape.attn_idx * c->seq_len * c->kv_dim;
    size_t kv_bytes = (size_t)n_kv * c->kv_dim * sizeof(float);
    size_t kv_off = loff * sizeof(float);

    if (bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, s->key_cache + loff, kv_bytes,
            kv_off) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, s->value_cache + loff, kv_bytes,
            kv_off) != 0) {
        free(cpu_in);
        free(gpu_xb);
        return -1;
    }

    memcpy(s->x, cpu_in, (size_t)dim * sizeof(float));
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim, c->norm_eps);
    fallback_cpu_matvec(m, s->q, &lw->attn.wq, s->xb, s->x_q);
    fallback_cpu_matvec(m, key_cache_row, &lw->attn.wk, s->xb, s->x_q);
    fallback_cpu_matvec(m, value_cache_row, &lw->attn.wv, s->xb, s->x_q);

    if (lw->attn.q_bias) {
        for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++)
            value_cache_row[i] += lw->attn.v_bias[i];
    }

    bn_transformer_cpu_apply_rope_heads(s->q, c->n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, c->n_heads, kv_mul);

    debug_compare_vec("gqa_compare", layer, pos, s->xb, gpu_xb, dim);

    free(cpu_in);
    free(gpu_xb);
    return 0;
}

int bn_transformer_gpu_debug_compare_qkv(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    uint32_t kv_cache_off,
    int dim,
    int q_dim,
    int kv_dim) {
    BnRunState *s = &sess->state;
    float *cpu_q = (float *)malloc((size_t)q_dim * sizeof(float));
    float *cpu_k = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *cpu_v = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_q = (float *)malloc((size_t)q_dim * sizeof(float));
    float *gpu_k = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *gpu_v = (float *)malloc((size_t)kv_dim * sizeof(float));
    if (!cpu_q || !cpu_k || !cpu_v || !gpu_xb || !gpu_q || !gpu_k || !gpu_v) {
        free(cpu_q); free(cpu_k); free(cpu_v);
        free(gpu_xb); free(gpu_q); free(gpu_k); free(gpu_v);
        return -1;
    }

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                   (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf(gpu, BN_GPU_VALUE_Q, gpu_q,
                                               (size_t)q_dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, gpu_k,
            (size_t)kv_dim * sizeof(float),
            (size_t)kv_cache_off * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, gpu_v,
            (size_t)kv_dim * sizeof(float),
            (size_t)kv_cache_off * sizeof(float)) != 0) {
        free(cpu_q); free(cpu_k); free(cpu_v);
        free(gpu_xb); free(gpu_q); free(gpu_k); free(gpu_v);
        return -1;
    }

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim, m->config.norm_eps);
    debug_compare_vec("attn_norm_compare", layer, pos, s->xb, gpu_xb, dim);
    debug_compare_native_quant_activation(gpu, layer, pos, s->xb, dim);
    debug_quant_matvec_prepared(m, cpu_q, &lw->attn.wq, s->xb, s->x_q);
    debug_quant_matvec_prepared(m, cpu_k, &lw->attn.wk, s->xb, s->x_q);
    debug_quant_matvec_prepared(m, cpu_v, &lw->attn.wv, s->xb, s->x_q);
    if (lw->attn.q_bias) {
        for (int i = 0; i < q_dim; i++) cpu_q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) cpu_k[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++) cpu_v[i] += lw->attn.v_bias[i];
    }
    if (lw->attn.q_norm) {
        int head_size = m->config.head_size;
        int qk_stride = m->config.qk_norm_per_head ? head_size : 0;
        for (int h = 0; h < m->config.n_heads; h++)
            fallback_rmsnorm(cpu_q + (size_t)h * head_size,
                             cpu_q + (size_t)h * head_size,
                             lw->attn.q_norm + (size_t)h * qk_stride,
                             head_size, m->config.norm_eps);
    }
    if (lw->attn.k_norm) {
        int head_size = m->config.head_size;
        int qk_stride = m->config.qk_norm_per_head ? head_size : 0;
        for (int h = 0; h < m->config.n_kv_heads; h++)
            fallback_rmsnorm(cpu_k + (size_t)h * head_size,
                             cpu_k + (size_t)h * head_size,
                             lw->attn.k_norm + (size_t)h * qk_stride,
                             head_size, m->config.norm_eps);
    }
    if (lw->attn.k_bias) {
        int head_size = m->config.head_size;
        int rope_dims = m->config.rope_dim_count > 0
            ? m->config.rope_dim_count : head_size;
        int half = rope_dims / 2;
        for (int h = 0; h < m->config.n_kv_heads; h++) {
            float *kh = cpu_k + (size_t)h * head_size;
            for (int i = 0; i < half; i++) {
                float angle = (float)pos * s->rope_freq[i];
                float cosv = cosf(angle);
                float sinv = sinf(angle);
                float x0 = kh[i];
                float x1 = kh[i + half];
                kh[i] = x0 * cosv - x1 * sinv;
                kh[i + half] = x0 * sinv + x1 * cosv;
            }
        }
    }

    debug_compare_vec("qkv_q_compare", layer, pos, cpu_q, gpu_q, q_dim);
    debug_compare_vec("qkv_k_compare", layer, pos, cpu_k, gpu_k, kv_dim);
    debug_compare_vec("qkv_v_compare", layer, pos, cpu_v, gpu_v, kv_dim);

    free(cpu_q); free(cpu_k); free(cpu_v);
    free(gpu_xb); free(gpu_q); free(gpu_k); free(gpu_v);
    return 0;
}

int bn_transformer_gpu_fallback_logits(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    const BnTransformerGPULogitResources *logits,
    int dim) {
    BnRunState *s = &sess->state;
    double t0 = bn_platform_time_ms();
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        bn_transformer_gpu_report_fallback("gpu logits cpu fallback flush failed");
        return -1;
    }
    double t_flush = bn_platform_time_ms();
    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                   (size_t)dim * sizeof(float)) != 0) {
        bn_transformer_gpu_report_fallback("gpu logits cpu fallback read_xb failed");
        return -1;
    }
    double t_read = bn_platform_time_ms();
    fallback_cpu_matvec(m, s->logits, logits->cpu_weight, s->xb, s->x_q);
    double t_logits = bn_platform_time_ms();
    if (bn_transformer_gpu_profile_level() >= 3) {
        fprintf(stderr,
                "[gpu:fallback:logits] flush=%.3fms read=%.3fms cpu=%.3fms total=%.3fms\n",
                t_flush - t0, t_read - t_flush, t_logits - t_read,
                t_logits - t0);
    }
    return 0;
}
