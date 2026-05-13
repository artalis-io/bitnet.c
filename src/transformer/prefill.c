#include "transformer_internal.h"
#include "backend_quant.h"
#include "moe.h"
#include "session.h"
#include "sh_arena.h"
#include "sh_log.h"

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

#define BN_MAX_VLA_ELEMS 8192

static inline void *prefill_qweight_backend_buf(const BnBackendModel *backend,
                                                const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static void prefill_quant_matmul_gpu(const BnModel *m,
                                     float *out,
                                     const BnQWeight *W,
                                     const float *X,
                                     int n_tokens,
                                     int8_t *x_q_buf) {
    bn_backend_quant_matmul_gpu_buf(out, W,
                                    prefill_qweight_backend_buf(m->backend, W),
                                    X, n_tokens, x_q_buf, m->pool,
                                    bn_model_gpu(m));
}

#ifdef __AVX2__
static int prefill_quant_can_preq8k_pair(int a, int b) {
    return bn_quant_format_can_preq8k(a) && bn_quant_format_can_preq8k(b);
}

static int prefill_quant_can_preq8k_triple(int a, int b, int c) {
    return prefill_quant_can_preq8k_pair(a, b) && bn_quant_format_can_preq8k(c);
}
#endif

#ifdef __ARM_NEON
#define prefill_rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
#define prefill_rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
#define prefill_rmsnorm bn_transformer_rmsnorm_wasm
#else
#define prefill_rmsnorm bn_transformer_rmsnorm_scalar
#endif

#ifdef __ARM_NEON
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_neon_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_neon_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_neon_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_neon_range
#elif defined(__AVX2__)
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_avx2_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_avx2_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_avx2_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_avx2_range
#elif defined(__wasm_simd128__)
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_wasm_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_wasm_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_wasm_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_wasm_range
#else
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_scalar_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_scalar_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_scalar_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_scalar_range
#endif

static float *prefill_logits(BnModel *m, BnSession *sess) {
    return bn_transformer_forward_logits(m, sess);
}

static float *prefill_internal(BnModel *m, BnSession *sess, const int *tokens,
                               int n_tokens, int pos0, float *all_logits) {
    if (n_tokens <= 0) return NULL;
    if (n_tokens == 1) return bn_transformer_forward(m, sess, tokens[0], pos0);

    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int dim = c->dim;
    int head_size = c->head_size;

    if (head_size > BN_MAX_VLA_ELEMS || dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dimensions too large for stack VLAs");
        return NULL;
    }

    for (int t = 0; t < n_tokens; t++) {
        if (tokens[t] < 0 || tokens[t] >= c->vocab_size) {
            SH_LOG_ERROR("Token out of range");
            return NULL;
        }
    }
    if (pos0 < 0) {
        SH_LOG_ERROR("Position out of range");
        return NULL;
    }

    size_t act_elems = (size_t)n_tokens * dim;
    if (act_elems / n_tokens != (size_t)dim) {
        SH_LOG_ERROR("Prefill activation buffer size overflow");
        return NULL;
    }

    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;
    int half_rope = rope_dims / 2;
    if (half_rope > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("RoPE dimensions too large for stack VLAs");
        return NULL;
    }

    int kv_dim = c->kv_dim;
    int hidden_dim = c->hidden_dim;
    int q_dim = c->n_heads * head_size;
    int q_buf_stride = (q_dim > dim ? q_dim * 2 : dim);
    int xb2_stride = dim;
    int hb_stride = hidden_dim;
    int hb2_stride = hidden_dim;
    if (c->full_attn_interval > 0 && c->ssm_inner_size > 0) {
        int ssm_qkv_dim = c->ssm_group_count * c->ssm_state_size * 2 +
                          c->ssm_inner_size;
        if (ssm_qkv_dim > q_buf_stride)
            q_buf_stride = ssm_qkv_dim;
        if (c->ssm_inner_size > xb2_stride)
            xb2_stride = c->ssm_inner_size;
        if (c->ssm_inner_size > hb_stride)
            hb_stride = c->ssm_inner_size;
        if (c->ssm_inner_size > hb2_stride)
            hb2_stride = c->ssm_inner_size;
    }
    size_t nt = (size_t)n_tokens;

    size_t batch_floats = nt * dim
                        + nt * (size_t)q_buf_stride
                        + nt * kv_dim * 2
                        + nt * (size_t)xb2_stride
                        + nt * (size_t)hb_stride
                        + nt * (size_t)hb2_stride;
    size_t arena_size = act_elems * sizeof(float)
                      + batch_floats * sizeof(float);
#ifdef __AVX2__
    int n_bpr_pf = (dim % BN_QK_K == 0) ? dim / BN_QK_K : 0;
    if (n_bpr_pf > 0)
        arena_size += nt * dim
                    + nt * n_bpr_pf * sizeof(float)
                    + nt * n_bpr_pf * 16 * sizeof(int16_t);
#endif

    SHArena *pf_arena = sh_arena_create(arena_size);
    if (!pf_arena) return NULL;

    float *act = (float *)sh_arena_alloc(pf_arena, act_elems * sizeof(float));
    if (!act) { sh_arena_free(pf_arena); return NULL; }

    for (int t = 0; t < n_tokens; t++)
        bn_model_embed_token(m, act + (size_t)t * dim, tokens[t]);

    float *batch_buf = (float *)sh_arena_alloc(pf_arena, batch_floats * sizeof(float));
    if (!batch_buf) { sh_arena_free(pf_arena); return NULL; }

#ifdef __AVX2__
    int8_t *pf_xq = NULL;
    float *pf_xd = NULL;
    int16_t *pf_xbs = NULL;
    if (n_bpr_pf > 0) {
        pf_xq = (int8_t *)sh_arena_alloc(pf_arena, nt * dim);
        pf_xd = (float *)sh_arena_alloc(pf_arena, nt * n_bpr_pf * sizeof(float));
        pf_xbs = (int16_t *)sh_arena_alloc(pf_arena, nt * n_bpr_pf * 16 * sizeof(int16_t));
        if (!pf_xq || !pf_xd || !pf_xbs)
            pf_xq = NULL;
    }
#endif

    float *Xb = batch_buf;
    float *Q_buf = Xb + nt * dim;
    float *K_new = Q_buf + nt * q_buf_stride;
    float *V_new = K_new + nt * kv_dim;
    float *Xb2 = V_new + nt * kv_dim;
    float *Hb = Xb2 + nt * xb2_stride;
    float *Hb2 = Hb + nt * hb_stride;

    BnWeights *w = &m->weights;

    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, lw, l, m->tq_state != NULL);
        int is_attn = plan.is_attn;

        if (is_attn && lw->wq.data) {
            for (int t = 0; t < n_tokens; t++)
                prefill_rmsnorm(Xb + t * dim, act + (size_t)t * dim,
                                lw->attn_norm, dim, c->norm_eps);

#ifdef __AVX2__
            if (pf_xq && !bn_model_gpu(m) &&
                prefill_quant_can_preq8k_triple(lw->wq.type, lw->wk.type, lw->wv.type)) {
                int n_bpr = dim / BN_QK_K;
                for (int t = 0; t < n_tokens; t++)
                    bn_quant_x_to_q8k(Xb + (size_t)t * dim,
                                      pf_xq + (size_t)t * dim,
                                      pf_xd + (size_t)t * n_bpr,
                                      pf_xbs + (size_t)t * n_bpr * 16, dim);
                bn_quant_matmul_preq8k(Q_buf, &lw->wq, n_tokens,
                                       pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                bn_quant_matmul_preq8k(K_new, &lw->wk, n_tokens,
                                       pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                bn_quant_matmul_preq8k(V_new, &lw->wv, n_tokens,
                                       pf_xq, pf_xd, pf_xbs, Xb, m->pool);
            } else
#endif
            {
                prefill_quant_matmul_gpu(m, Q_buf, &lw->wq, Xb, n_tokens, s->x_q);
                prefill_quant_matmul_gpu(m, K_new, &lw->wk, Xb, n_tokens, s->x_q);
                prefill_quant_matmul_gpu(m, V_new, &lw->wv, Xb, n_tokens, s->x_q);
            }

            int attn_idx = plan.attn_idx;
            size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
            int kv_mul = plan.kv_mul;
            int q_gated = plan.q_gated;

            for (int t = 0; t < n_tokens; t++) {
                int pos = pos0 + t;
                int cache_pos = pos % c->seq_len;
                float *q_t = Q_buf + (size_t)t * lw->wq.rows;
                float *k_t = K_new + (size_t)t * kv_dim;
                float *v_t = V_new + (size_t)t * kv_dim;

                if (q_gated) {
                    for (int h = 0; h < c->n_heads; h++)
                        memcpy(s->q + h * head_size, q_t + h * 2 * head_size,
                               head_size * sizeof(float));
                } else {
                    memcpy(s->q, q_t, q_dim * sizeof(float));
                }

                if (lw->q_norm) {
                    int qk_stride = c->qk_norm_per_head ? head_size : 0;
                    for (int h = 0; h < c->n_heads; h++)
                        prefill_rmsnorm(s->q + h * head_size, s->q + h * head_size,
                                        lw->q_norm + h * qk_stride, head_size,
                                        c->norm_eps);
                }
                if (lw->k_norm) {
                    int qk_stride = c->qk_norm_per_head ? head_size : 0;
                    for (int h = 0; h < c->n_kv_heads; h++)
                        prefill_rmsnorm(k_t + h * head_size, k_t + h * head_size,
                                        lw->k_norm + h * qk_stride, head_size,
                                        c->norm_eps);
                }

                if (lw->q_bias) for (int i = 0; i < q_dim; i++) s->q[i] += lw->q_bias[i];
                if (lw->k_bias) for (int i = 0; i < kv_dim; i++) k_t[i] += lw->k_bias[i];
                if (lw->v_bias) for (int i = 0; i < kv_dim; i++) v_t[i] += lw->v_bias[i];

                float rope_cos_t[half_rope], rope_sin_t[half_rope];
                for (int i = 0; i < half_rope; i++) {
                    float angle = pos * s->rope_freq[i];
                    rope_cos_t[i] = cosf(angle);
                    rope_sin_t[i] = sinf(angle);
                }
                bn_transformer_cpu_apply_rope_heads(s->q, c->n_heads, head_size,
                                                    rope_dims, rope_cos_t, rope_sin_t);
                bn_transformer_cpu_apply_rope_heads(k_t, c->n_kv_heads, head_size,
                                                    rope_dims, rope_cos_t, rope_sin_t);

                if (c->kv_f16)
                    bn_transformer_write_kv_fp16(s, loff, cache_pos, kv_dim,
                                                 k_t, v_t, kv_dim);
                else
                    bn_transformer_write_kv_fp32(s, loff, cache_pos, kv_dim,
                                                 k_t, v_t, kv_dim);

                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul,
                                  head_size, kv_dim, c->seq_len };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, c->n_heads, kv_mul);

                if (q_gated) {
                    for (int h = 0; h < c->n_heads; h++) {
                        float *gate_h = q_t + h * 2 * head_size + head_size;
                        float *xb_h = s->xb + h * head_size;
                        for (int d = 0; d < head_size; d++)
                            xb_h[d] *= 1.0f / (1.0f + expf(-gate_h[d]));
                    }
                }

                int wo_cols = lw->wo.cols;
                memcpy(Q_buf + (size_t)t * wo_cols, s->xb, wo_cols * sizeof(float));
            }

            {
                int wo_cols = lw->wo.cols;
                if (lw->attn_sub_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_rmsnorm(Q_buf + (size_t)t * wo_cols,
                                        Q_buf + (size_t)t * wo_cols,
                                        lw->attn_sub_norm, wo_cols, c->norm_eps);
                prefill_quant_matmul_gpu(m, Xb2, &lw->wo, Q_buf, n_tokens, s->x_q);
            }

            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb2[(size_t)t * dim + d];

        } else if (!is_attn) {
            int num_k_heads = c->ssm_group_count;
            int head_k_dim = c->ssm_state_size;
            int num_v_heads = c->ssm_time_step_rank;
            int head_v_dim = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
            int key_dim_ssm = num_k_heads * head_k_dim;
            int value_dim = c->ssm_inner_size;
            int qkv_dim_ssm = key_dim_ssm * 2 + value_dim;
            int kern_ssm = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
            int ssm_idx = plan.ssm_idx;
            size_t state_per_layer = (size_t)num_v_heads * head_k_dim * head_v_dim;
            float *ssm_state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
            size_t conv_per_layer = (size_t)(kern_ssm - 1) * qkv_dim_ssm;
            float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;

            for (int t = 0; t < n_tokens; t++)
                prefill_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                                lw->attn_norm, dim, c->norm_eps);

            if (q_buf_stride < qkv_dim_ssm) { sh_arena_free(pf_arena); return NULL; }
            float *QKV_all = Q_buf;
            float *Z_all = Xb2;
            float *Out_all = Hb;

            prefill_quant_matmul_gpu(m, QKV_all, &lw->wqkv, Xb, n_tokens, s->x_q);
            prefill_quant_matmul_gpu(m, Z_all, &lw->wz, Xb, n_tokens, s->x_q);

            for (int t = 0; t < n_tokens; t++) {
                float *qkv_t = QKV_all + (size_t)t * lw->wqkv.rows;
                float *z_t = Z_all + (size_t)t * lw->wz.rows;
                float *out_t = Out_all + (size_t)t * value_dim;
                float *xb_t = Xb + (size_t)t * dim;

                BnSSMConvCtx conv_ctx = { qkv_t, conv_state, lw->ssm_conv1d,
                                          qkv_dim_ssm, kern_ssm };
                BnTPTask conv_task = { prefill_ssm_conv_silu, &conv_ctx, qkv_dim_ssm };
                bn_tp_dispatch(m->pool, &conv_task, 1);

                float *q_raw = qkv_t;
                float *k_raw = qkv_t + key_dim_ssm;
                float *v_raw = qkv_t + 2 * key_dim_ssm;

                BnSSML2NormCtx norm_ctx = { q_raw, k_raw, head_k_dim };
                BnTPTask norm_task = { prefill_ssm_l2norm, &norm_ctx, num_k_heads };
                bn_tp_dispatch(m->pool, &norm_task, 1);

                if (num_v_heads > BN_MAX_VLA_ELEMS) continue;
                float alpha_arr[num_v_heads > 0 ? num_v_heads : 1];
                float beta_arr[num_v_heads > 0 ? num_v_heads : 1];
                BnMatvecTask ab[2] = {
                    { alpha_arr, &lw->ssm_alpha, NULL },
                    { beta_arr,  &lw->ssm_beta, NULL },
                };
                bn_quant_matvec_batch(ab, 2, xb_t, s->x_q, m->pool);
                for (int h = 0; h < num_v_heads; h++) {
                    float dt = alpha_arr[h] + lw->ssm_dt_bias[h];
                    float dt_sp = (dt > 20.0f) ? dt : logf(1.0f + expf(dt));
                    alpha_arr[h] = expf(dt_sp * lw->ssm_a[h]);
                    beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
                }

                float q_scale = 1.0f / sqrtf((float)head_k_dim);
                BnSSMDeltaCtx delta_ctx = {
                    ssm_state, out_t, q_raw, k_raw, v_raw,
                    alpha_arr, beta_arr,
                    num_k_heads, head_k_dim, head_v_dim, q_scale
                };
                BnTPTask delta_task = { prefill_ssm_delta, &delta_ctx, num_v_heads };
                bn_tp_dispatch(m->pool, &delta_task, 1);

                BnSSMGateCtx gate_ctx = { out_t, z_t, lw->ssm_norm,
                                          c->norm_eps, head_v_dim };
                BnTPTask gate_task = { prefill_ssm_gate, &gate_ctx, num_v_heads };
                bn_tp_dispatch(m->pool, &gate_task, 1);
            }

            prefill_quant_matmul_gpu(m, Xb, &lw->ssm_out, Out_all, n_tokens, s->x_q);

            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
        }

        if (lw->router_weight) {
            if (bn_moe_forward_batch(m, sess, lw, l, act, Xb, n_tokens) != 0) {
                sh_arena_free(pf_arena);
                return NULL;
            }
        } else if (lw->ffn_up.data) {
            for (int t = 0; t < n_tokens; t++)
                prefill_rmsnorm(Xb + t * dim, act + (size_t)t * dim,
                                lw->ffn_norm, dim, c->norm_eps);

            if (c->has_ffn_gate) {
#ifdef __AVX2__
                if (pf_xq && !bn_model_gpu(m) &&
                    prefill_quant_can_preq8k_pair(lw->ffn_gate.type, lw->ffn_up.type)) {
                    int n_bpr = dim / BN_QK_K;
                    for (int t = 0; t < n_tokens; t++)
                        bn_quant_x_to_q8k(Xb + (size_t)t * dim,
                                          pf_xq + (size_t)t * dim,
                                          pf_xd + (size_t)t * n_bpr,
                                          pf_xbs + (size_t)t * n_bpr * 16, dim);
                    bn_quant_matmul_preq8k(Hb, &lw->ffn_gate, n_tokens,
                                           pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                    bn_quant_matmul_preq8k(Hb2, &lw->ffn_up, n_tokens,
                                           pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                } else
#endif
                {
                    prefill_quant_matmul_gpu(m, Hb, &lw->ffn_gate, Xb, n_tokens, s->x_q);
                    prefill_quant_matmul_gpu(m, Hb2, &lw->ffn_up, Xb, n_tokens, s->x_q);
                }

                for (int t = 0; t < n_tokens; t++) {
                    float *hb_t = Hb + (size_t)t * hidden_dim;
                    float *hb2_t = Hb2 + (size_t)t * hidden_dim;
                    if (c->act_type == 1) {
                        for (int i = 0; i < hidden_dim; i++) {
                            float g = hb_t[i] > 0 ? hb_t[i] : 0;
                            hb_t[i] = g * g * hb2_t[i];
                        }
                    } else {
                        for (int i = 0; i < hidden_dim; i++) {
                            float g = hb_t[i];
                            hb_t[i] = (g / (1.0f + expf(-g))) * hb2_t[i];
                        }
                    }
                }
            } else {
                prefill_quant_matmul_gpu(m, Hb, &lw->ffn_up, Xb, n_tokens, s->x_q);
                for (int t = 0; t < n_tokens; t++) {
                    float *hb_t = Hb + (size_t)t * hidden_dim;
                    if (c->act_type == 1) {
                        for (int i = 0; i < hidden_dim; i++) {
                            float v = hb_t[i] > 0 ? hb_t[i] : 0;
                            hb_t[i] = v * v;
                        }
                    } else {
                        for (int i = 0; i < hidden_dim; i++) {
                            float v = hb_t[i];
                            hb_t[i] = v / (1.0f + expf(-v));
                        }
                    }
                }
            }

            if (lw->ffn_sub_norm)
                for (int t = 0; t < n_tokens; t++)
                    prefill_rmsnorm(Hb + (size_t)t * hidden_dim,
                                    Hb + (size_t)t * hidden_dim,
                                    lw->ffn_sub_norm, hidden_dim, c->norm_eps);

            prefill_quant_matmul_gpu(m, Xb, &lw->ffn_down, Hb, n_tokens, s->x_q);

            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
        }
    }

    if (all_logits) {
        int vocab_size = c->vocab_size;
        for (int t = 0; t < n_tokens; t++) {
            memcpy(s->x, act + (size_t)t * dim, dim * sizeof(float));
            float *lg = prefill_logits(m, sess);
            if (!lg) { sh_arena_free(pf_arena); return NULL; }
            memcpy(all_logits + (size_t)t * vocab_size, lg,
                   vocab_size * sizeof(float));
        }
        sh_arena_free(pf_arena);
        return s->logits;
    }

    memcpy(s->x, act + (size_t)(n_tokens - 1) * dim, dim * sizeof(float));
    sh_arena_free(pf_arena);
    return prefill_logits(m, sess);
}

float *bn_transformer_prefill(BnModel *m, BnSession *s, const int *tokens,
                              int n_tokens, int pos0) {
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL);
}

int bn_transformer_prefill_all(BnModel *m, BnSession *s, const int *tokens,
                               int n_tokens, int pos0, float *all_logits) {
    if (!all_logits || n_tokens <= 0) return -1;

    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, s, tokens[0], pos0);
        if (!logits) return -1;
        memcpy(all_logits, logits, m->config.vocab_size * sizeof(float));
        return 0;
    }

    float *result = prefill_internal(m, s, tokens, n_tokens, pos0, all_logits);
    return result ? 0 : -1;
}
