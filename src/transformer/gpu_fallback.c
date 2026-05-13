#include "gpu_internal.h"
#include "../gpu_shader_ir_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "moe.h"
#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fallback_rmsnorm(float *out,
                             const float *x,
                             const float *w,
                             int size,
                             float eps) {
#ifdef __ARM_NEON
    bn_transformer_rmsnorm_neon(out, x, w, size, eps);
#elif defined(__AVX2__)
    bn_transformer_rmsnorm_avx2(out, x, w, size, eps);
#elif defined(__wasm_simd128__)
    bn_transformer_rmsnorm_wasm(out, x, w, size, eps);
#else
    bn_transformer_rmsnorm_scalar(out, x, w, size, eps);
#endif
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
        bn_transformer_cpu_forward_ffn_block(m, sess, lw, NULL);
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
    bn_moe_forward(m, sess, lw, layer);
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
    if (bn_transformer_cpu_forward_layer(m, sess, layer, pos, cache_pos,
                                         rope_dims, rope_cos, rope_sin) != 0)
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
    if (!shape.is_attn || shape.q_gated || shape.q_wide ||
        c->kv_tq_bits > 0 || c->kv_f16 ||
        lw->attn.q_norm || lw->attn.k_norm || lw->norm.attn_sub_norm)
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
    bn_quant_matvec(s->q, &lw->attn.wq, s->xb, s->x_q, bn_model_pool(m));
    bn_quant_matvec(key_cache_row, &lw->attn.wk, s->xb, s->x_q,
                    bn_model_pool(m));
    bn_quant_matvec(value_cache_row, &lw->attn.wv, s->xb, s->x_q,
                    bn_model_pool(m));

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

    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, c->n_heads, kv_mul);

    bn_quant_matvec(s->xb2, &lw->attn.wo, s->xb, s->x_q, bn_model_pool(m));
    bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
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
    bn_transformer_cpu_forward_ffn_block(m, sess, lw, ffn_plan);
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
    bn_quant_matvec(s->xb, &lw->ffn.ffn_down, s->hb, s->x_q,
                    bn_model_pool(m));
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

    bn_quant_matvec(s->xb, &lw->ffn.ffn_down, s->hb, s->x_q,
                    bn_model_pool(m));

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
    if (!cpu_x_in || !cpu_xb_in || !cpu_x || !cpu_xb || !gpu_x || !gpu_xb) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
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
        return -1;
    }

    memcpy(s->x, cpu_x_in, (size_t)dim * sizeof(float));
    memcpy(s->xb, cpu_xb_in, (size_t)dim * sizeof(float));
    bn_transformer_cpu_forward_ffn_block(m, sess, lw, ffn_plan);
    memcpy(cpu_x, s->x, (size_t)dim * sizeof(float));
    if (next_norm)
        fallback_rmsnorm(cpu_xb, cpu_x, next_norm, dim, m->config.norm_eps);

    if (bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        (next_norm && bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                                 (size_t)dim * sizeof(float)) != 0)) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
        return -1;
    }
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
    if (!shape.is_attn || shape.q_gated || shape.q_wide ||
        c->kv_tq_bits > 0 || c->kv_f16 ||
        lw->attn.q_norm || lw->attn.k_norm || lw->norm.attn_sub_norm)
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
    bn_quant_matvec(s->q, &lw->attn.wq, s->xb, s->x_q, bn_model_pool(m));
    bn_quant_matvec(key_cache_row, &lw->attn.wk, s->xb, s->x_q,
                    bn_model_pool(m));
    bn_quant_matvec(value_cache_row, &lw->attn.wv, s->xb, s->x_q,
                    bn_model_pool(m));

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
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, c->n_heads, kv_mul);

    bn_quant_matvec(s->xb2, &lw->attn.wo, s->xb, s->x_q, bn_model_pool(m));
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
    float *gpu_q = (float *)malloc((size_t)q_dim * sizeof(float));
    float *gpu_k = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *gpu_v = (float *)malloc((size_t)kv_dim * sizeof(float));
    if (!cpu_q || !cpu_k || !cpu_v || !gpu_q || !gpu_k || !gpu_v) {
        free(cpu_q); free(cpu_k); free(cpu_v);
        free(gpu_q); free(gpu_k); free(gpu_v);
        return -1;
    }

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_x(gpu, s->x,
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
        free(gpu_q); free(gpu_k); free(gpu_v);
        return -1;
    }

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim, m->config.norm_eps);
    bn_quant_matvec(cpu_q, &lw->attn.wq, s->xb, s->x_q, bn_model_pool(m));
    bn_quant_matvec(cpu_k, &lw->attn.wk, s->xb, s->x_q, bn_model_pool(m));
    bn_quant_matvec(cpu_v, &lw->attn.wv, s->xb, s->x_q, bn_model_pool(m));
    if (lw->attn.q_bias) {
        for (int i = 0; i < q_dim; i++) cpu_q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) cpu_k[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++) cpu_v[i] += lw->attn.v_bias[i];
    }

    debug_compare_vec("qkv_q_compare", layer, pos, cpu_q, gpu_q, q_dim);
    debug_compare_vec("qkv_k_compare", layer, pos, cpu_k, gpu_k, kv_dim);
    debug_compare_vec("qkv_v_compare", layer, pos, cpu_v, gpu_v, kv_dim);

    free(cpu_q); free(cpu_k); free(cpu_v);
    free(gpu_q); free(gpu_k); free(gpu_v);
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
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_xb(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_quant_matvec(s->logits, logits->cpu_weight, s->x, s->x_q,
                    bn_model_pool(m));
    return 0;
}
