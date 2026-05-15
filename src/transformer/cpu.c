#include "transformer_cpu_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_batched_attn_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "backend_model.h"
#include "quant.h"
#include "moe.h"
#include "session.h"
#include "sh_log.h"
#include <stdlib.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

static inline const BnPreparedWeight *cpu_qweight_prepared(
    const BnBackendModel *backend,
    const BnQWeight *w) {
    if (getenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS"))
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

#ifdef __AVX2__
static int cpu_quant_can_preq8k_triple(int a, int b, int c) {
    return cpu_quant_can_preq8k_pair(a, b) &&
           bn_quant_format_can_preq8k(c);
}
#endif

#ifdef __ARM_NEON
#define cpu_rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
#define cpu_rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
#define cpu_rmsnorm bn_transformer_rmsnorm_wasm
#else
#define cpu_rmsnorm bn_transformer_rmsnorm_scalar
#endif

#ifdef __ARM_NEON
#define cpu_ssm_conv_silu bn_transformer_ssm_conv_silu_neon_range
#define cpu_ssm_l2norm    bn_transformer_ssm_l2norm_neon_range
#define cpu_ssm_delta     bn_transformer_ssm_delta_neon_range
#define cpu_ssm_gate      bn_transformer_ssm_gate_neon_range
#elif defined(__AVX2__)
#define cpu_ssm_conv_silu bn_transformer_ssm_conv_silu_avx2_range
#define cpu_ssm_l2norm    bn_transformer_ssm_l2norm_avx2_range
#define cpu_ssm_delta     bn_transformer_ssm_delta_avx2_range
#define cpu_ssm_gate      bn_transformer_ssm_gate_avx2_range
#elif defined(__wasm_simd128__)
#define cpu_ssm_conv_silu bn_transformer_ssm_conv_silu_wasm_range
#define cpu_ssm_l2norm    bn_transformer_ssm_l2norm_wasm_range
#define cpu_ssm_delta     bn_transformer_ssm_delta_wasm_range
#define cpu_ssm_gate      bn_transformer_ssm_gate_wasm_range
#else
#define cpu_ssm_conv_silu bn_transformer_ssm_conv_silu_scalar_range
#define cpu_ssm_l2norm    bn_transformer_ssm_l2norm_scalar_range
#define cpu_ssm_delta     bn_transformer_ssm_delta_scalar_range
#define cpu_ssm_gate      bn_transformer_ssm_gate_scalar_range
#endif

void bn_transformer_cpu_gqa_dispatch(BnModel *m,
                                     BnGQACtx *gctx,
                                     int n_heads,
                                     int kv_mul) {
    (void)kv_mul;
#ifdef __ARM_NEON
    bn_tp_fn attn_fn = m->config.flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
    bn_tp_fn attn_fn = m->config.flash_attn ? bn_transformer_flash_gqa_avx2_range : bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
    bn_tp_fn attn_fn = m->config.flash_attn ? bn_transformer_flash_gqa_wasm_range : bn_transformer_gqa_wasm_range;
#else
    bn_tp_fn attn_fn = m->config.flash_attn ? bn_transformer_flash_gqa_scalar_range : bn_transformer_gqa_scalar_range;
#endif
    BnTPTask gqa = { attn_fn, gctx, n_heads };
    bn_tp_dispatch(bn_model_pool(m), &gqa, 1);
}

void bn_transformer_batched_attn_dispatch(BnModel *m,
                                          BnBatchedAttnCtx *ctx) {
    bn_tp_fn fn;
    if (m->config.flash_attn) {
#ifdef __AVX2__
        fn = bn_transformer_batched_attn_flash_avx2_range;
#else
        fn = bn_transformer_batched_attn_flash_scalar_range;
#endif
    } else {
#ifdef __AVX2__
        fn = bn_transformer_batched_attn_naive_avx2_range;
#else
        fn = bn_transformer_batched_attn_naive_scalar_range;
#endif
    }
    BnTPTask task = { fn, ctx, ctx->n_heads };
    bn_tp_dispatch(bn_model_pool(m), &task, 1);
}

void bn_transformer_cpu_residual_add(float *x, const float *r, int dim) {
#ifdef __ARM_NEON
    for (int i = 0; i < dim; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(r + i)));
#elif defined(__AVX2__)
    for (int i = 0; i < dim; i += 8)
        _mm256_storeu_ps(x + i,
                         _mm256_add_ps(_mm256_loadu_ps(x + i),
                                       _mm256_loadu_ps(r + i)));
#elif defined(__wasm_simd128__)
    for (int i = 0; i < dim; i += 4)
        wasm_v128_store(x + i,
                        wasm_f32x4_add(wasm_v128_load(x + i),
                                       wasm_v128_load(r + i)));
#else
    for (int i = 0; i < dim; i++)
        x[i] += r[i];
#endif
}

void bn_transformer_cpu_apply_rope_heads(float *buf,
                                         int n_heads,
                                         int head_size,
                                         int rope_dims,
                                         const float *rc,
                                         const float *rs) {
#ifdef __AVX2__
    if (rope_dims >= 8) {
        for (int h = 0; h < n_heads; h++) {
            float *hd = buf + h * head_size;
            int half_rope = rope_dims / 2;
            int i = 0;
            for (; i + 7 < half_rope; i += 8) {
                __m256 v0 = _mm256_loadu_ps(hd + i);
                __m256 v1 = _mm256_loadu_ps(hd + half_rope + i);
                __m256 cos_v = _mm256_loadu_ps(rc + i);
                __m256 sin_v = _mm256_loadu_ps(rs + i);
                __m256 out0 = _mm256_fmsub_ps(v0, cos_v,
                                              _mm256_mul_ps(v1, sin_v));
                __m256 out1 = _mm256_fmadd_ps(v0, sin_v,
                                              _mm256_mul_ps(v1, cos_v));
                _mm256_storeu_ps(hd + i, out0);
                _mm256_storeu_ps(hd + half_rope + i, out1);
            }
            for (; i < half_rope; i++) {
                int j = i + half_rope;
                float v0 = hd[i], v1 = hd[j];
                hd[i] = v0 * rc[i] - v1 * rs[i];
                hd[j] = v0 * rs[i] + v1 * rc[i];
            }
        }
        return;
    }
#endif
    for (int h = 0; h < n_heads; h++) {
        float *hd = buf + h * head_size;
        int half_rope = rope_dims / 2;
        for (int i = 0; i < half_rope; i++) {
            int j = i + half_rope;
            float v0 = hd[i], v1 = hd[j];
            hd[i] = v0 * rc[i] - v1 * rs[i];
            hd[j] = v0 * rs[i] + v1 * rc[i];
        }
    }
}

void bn_transformer_cpu_apply_ffn_activation(BnRunState *s,
                                             const BnFFNPlan *ffn_plan,
                                             int hidden_dim,
                                             int already_activated) {
    if (already_activated)
        return;

    if (ffn_plan->has_gate) {
        if (ffn_plan->activation == 1) {
#ifdef __ARM_NEON
            float32x4_t zero = vdupq_n_f32(0);
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vmaxq_f32(vld1q_f32(s->hb + i), zero);
                vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(g, g),
                                                vld1q_f32(s->hb2 + i)));
            }
#elif defined(__AVX2__)
            __m256 zero = _mm256_setzero_ps();
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_max_ps(_mm256_loadu_ps(s->hb + i), zero);
                _mm256_storeu_ps(s->hb + i,
                    _mm256_mul_ps(_mm256_mul_ps(g, g),
                                  _mm256_loadu_ps(s->hb2 + i)));
            }
#elif defined(__wasm_simd128__)
            v128_t zero = wasm_f32x4_splat(0);
            for (int i = 0; i < hidden_dim; i += 4) {
                v128_t g = wasm_f32x4_max(wasm_v128_load(s->hb + i), zero);
                wasm_v128_store(s->hb + i,
                    wasm_f32x4_mul(wasm_f32x4_mul(g, g),
                                   wasm_v128_load(s->hb2 + i)));
            }
#else
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = g * g * s->hb2[i];
            }
#endif
        } else {
#ifdef __ARM_NEON
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vld1q_f32(s->hb + i);
                float32x4_t u = vld1q_f32(s->hb2 + i);
                vst1q_f32(s->hb + i, vmulq_f32(bn_neon_fast_silu_f32(g), u));
            }
#elif defined(__AVX2__)
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_loadu_ps(s->hb + i);
                __m256 u = _mm256_loadu_ps(s->hb2 + i);
                _mm256_storeu_ps(s->hb + i,
                                 _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
            }
#else
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i];
                s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
            }
#endif
        }
    } else {
        if (ffn_plan->activation == 1) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else {
#ifdef __ARM_NEON
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t v = vld1q_f32(s->hb + i);
                vst1q_f32(s->hb + i, bn_neon_fast_silu_f32(v));
            }
#elif defined(__AVX2__)
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 v = _mm256_loadu_ps(s->hb + i);
                _mm256_storeu_ps(s->hb + i, bn_avx2_fast_silu_ps(v));
            }
#else
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i];
                s->hb[i] = v / (1.0f + expf(-v));
            }
#endif
        }
    }
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

    if (is_attn) {
        // ---- Attention block ----

        // KV cache offset: contiguous among attention layers only
        int attn_idx = shape->attn_idx;
        size_t loff = (size_t)attn_idx * c->seq_len * kv_cache_stride;

        // Q projection width detection:
        // q_dim = n_heads * head_size (total Q output elements)
        // Gated Q (Qwen3.5): wq.rows = 2 * q_dim (interleaved [Q, gate] per head)
        // Wide Q (Qwen3 MoE): wq.rows = q_dim > dim (head_size > dim/n_heads)
        // Classic: wq.rows = dim = q_dim
        int q_dim = shape->q_dim;
        int q_gated = shape->q_gated;
        int q_wide = shape->q_wide;

        /* Fused attn RMSNorm + Q8K: quantize s->xb once, reuse for Q and K+V */
        int attn_preq8k = 0;
#ifdef __AVX2__
        int attn_kquant = cpu_quant_can_preq8k_triple(lw->attn.wq.type, lw->attn.wk.type, lw->attn.wv.type) &&
                          !bn_model_gpu(m) && dim % BN_QK_K == 0;
#endif
        int n_sb_attn = dim / BN_QK_K;
        float attn_q8k_d[n_sb_attn > 0 ? n_sb_attn : 1];
        int16_t attn_q8k_bsums[n_sb_attn > 0 ? n_sb_attn * 16 : 1];
#ifdef __AVX2__
        if (attn_kquant) {
            bn_quant_rmsnorm_q8k_avx2(s->x, lw->norm.attn_norm, dim, c->norm_eps,
                                        s->xb, s->x_q, attn_q8k_d, attn_q8k_bsums);
            attn_preq8k = 1;
        } else
#endif
        {
            cpu_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim, c->norm_eps);
        }

        /* no-op */

        if (q_gated) {
            // --- Gated Q path (Qwen3.5 attention) ---
            float *q_full = s->hb;  // [2*dim]
            float *k_tmp = s->hb2;
            float *v_tmp = s->hb2 + kv_dim;

            // Q+K+V matvecs (reuse cached Q8K if available)
            if (!(c->kv_tq_bits > 0 && bn_model_tq_state(m)) && !c->kv_f16) {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
                BnMatvecTask qkv[3] = {
                     { q_full,          &lw->attn.wq, NULL },
                     { key_cache_row,   &lw->attn.wk, NULL },
                     { value_cache_row, &lw->attn.wv, NULL },
                };
                if (attn_preq8k)
                    cpu_quant_matvec_batch_preq8k(m, qkv, 3, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb);
                else
                    cpu_quant_matvec_batch_prepared(m, qkv, 3, s->xb, s->x_q);
                k_tmp = key_cache_row;
                v_tmp = value_cache_row;
            } else {
                BnMatvecTask qkv[3] = {
                     { q_full, &lw->attn.wq, NULL },
                     { k_tmp, &lw->attn.wk, NULL },
                     { v_tmp, &lw->attn.wv, NULL },
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
                    cpu_rmsnorm(s->q + h*head_size,
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
                    cpu_rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                            lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);

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
                uint16_t *kc = (uint16_t *)s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
                uint16_t *vc = (uint16_t *)s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
#ifdef __ARM_NEON
                for (int i = 0; i < kv_dim; i += 4) {
                    vst1_u16(kc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(k_tmp + i))));
                    vst1_u16(vc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(v_tmp + i))));
                }
#elif defined(__AVX2__)
                for (int i = 0; i < kv_dim; i += 8) {
                    _mm_storeu_si128((__m128i *)(kc + i), _mm256_cvtps_ph(_mm256_loadu_ps(k_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                    _mm_storeu_si128((__m128i *)(vc + i), _mm256_cvtps_ph(_mm256_loadu_ps(v_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                }
#else
                for (int i = 0; i < kv_dim; i++) {
                    kc[i] = bn_fp32_to_fp16(k_tmp[i]);
                    vc[i] = bn_fp32_to_fp16(v_tmp[i]);
                }
#endif
            }
            // FP32 path already wrote to cache directly

            if (!(c->kv_tq_bits > 0 && bn_model_tq_state(m))) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_cache_stride, c->seq_len };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            }

            // Sigmoid gate: xb *= sigmoid(gate)
            for (int h = 0; h < n_heads; h++) {
                float *gate_h = q_full + h * 2 * head_size + head_size;
                float *xb_h = s->xb + h * head_size;
#ifdef __AVX2__
                for (int d = 0; d < head_size; d += 8) {
                    __m256 g = _mm256_loadu_ps(gate_h + d);
                    __m256 xv = _mm256_loadu_ps(xb_h + d);
                    _mm256_storeu_ps(xb_h + d, _mm256_mul_ps(xv, bn_avx2_fast_sigmoid_ps(g)));
                }
#else
                for (int d = 0; d < head_size; d++)
                    xb_h[d] *= 1.0f / (1.0f + expf(-gate_h[d]));
#endif
            }

            // wo projection + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm(s->xb, s->xb, lw->norm.attn_sub_norm, dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
            bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

        } else if (q_wide) {
            // --- Wide Q path (Qwen3 MoE: head_size > dim/n_heads, no gate) ---
            float *k_tmp = s->hb, *v_tmp = s->hb2;

            // Q matvec: xb[dim] → q[q_dim]
            {
                BnMatvecTask q_task[1] = {{ s->q, &lw->attn.wq, NULL }};
                cpu_quant_matvec_batch_prepared(m, q_task, 1, s->xb, s->x_q);
            }
            // K/V matvec: xb[dim] → kv_dim (always to temp buffers for TQ compat)
            if (c->kv_tq_bits > 0 && bn_model_tq_state(m)) {
                BnMatvecTask kv[2] = {
                     { k_tmp, &lw->attn.wk, NULL },
                     { v_tmp, &lw->attn.wv, NULL },
                };
                cpu_quant_matvec_batch_prepared(m, kv, 2, s->xb, s->x_q);
            } else {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
                BnMatvecTask kv[2] = {
                     { key_cache_row,   &lw->attn.wk, NULL },
                     { value_cache_row, &lw->attn.wv, NULL },
                };
                cpu_quant_matvec_batch_prepared(m, kv, 2, s->xb, s->x_q);
                k_tmp = key_cache_row;
                v_tmp = value_cache_row;
            }

            if (lw->attn.q_norm)
                for (int h = 0; h < n_heads; h++)
                    cpu_rmsnorm(s->q + h*head_size, s->q + h*head_size,
                            lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
            if (lw->attn.k_norm)
                for (int h = 0; h < n_kv_heads; h++)
                    cpu_rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                            lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);

            bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);
            bn_transformer_cpu_apply_rope_heads(k_tmp, n_kv_heads, head_size,
                             layer_rope_dims, rope_cos, rope_sin);

            if (c->kv_tq_bits > 0 && bn_model_tq_state(m)) {
                // TQ write + GQA
                bn_transformer_tq_write_kv(bn_model_tq_state(m), s, k_tmp, v_tmp,
                            n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);
                bn_transformer_tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                n_kv_heads, head_size, kv_mul);
            } else {
                // Standard GQA
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_cache_stride, c->seq_len };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            }

            // wo projection (q_dim → dim) + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm(s->xb, s->xb, lw->norm.attn_sub_norm, q_dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
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
                     { s->q,  &lw->attn.wq, NULL },
                     { k_tmp, &lw->attn.wk, NULL },
                     { v_tmp, &lw->attn.wv, NULL },
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
                        cpu_rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);

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
                     { s->q,  &lw->attn.wq, NULL },
                     { k_tmp, &lw->attn.wk, NULL },
                     { v_tmp, &lw->attn.wv, NULL },
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
                        cpu_rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);

                bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(k_tmp, n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);

                uint16_t *kc = (uint16_t *)s->key_cache   + loff + (size_t)cache_pos * kv_cache_stride;
                uint16_t *vc = (uint16_t *)s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
#ifdef __ARM_NEON
                for (int i = 0; i < kv_dim; i += 4) {
                    float32x4_t kv4 = vld1q_f32(k_tmp + i);
                    float16x4_t kh4 = vcvt_f16_f32(kv4);
                    vst1_u16(kc + i, vreinterpret_u16_f16(kh4));
                    float32x4_t vv4 = vld1q_f32(v_tmp + i);
                    float16x4_t vh4 = vcvt_f16_f32(vv4);
                    vst1_u16(vc + i, vreinterpret_u16_f16(vh4));
                }
#elif defined(__AVX2__)
                for (int i = 0; i < kv_dim; i += 8) {
                    _mm_storeu_si128((__m128i *)(kc + i), _mm256_cvtps_ph(_mm256_loadu_ps(k_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                    _mm_storeu_si128((__m128i *)(vc + i), _mm256_cvtps_ph(_mm256_loadu_ps(v_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                }
#else
                for (int i = 0; i < kv_dim; i++) {
                    kc[i] = bn_fp32_to_fp16(k_tmp[i]);
                    vc[i] = bn_fp32_to_fp16(v_tmp[i]);
                }
#endif
            } else {
                BnMatvecTask qkv[3] = {
                     { s->q,            &lw->attn.wq, NULL },
                     { key_cache_row,   &lw->attn.wk, NULL },
                     { value_cache_row, &lw->attn.wv, NULL },
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
                        cpu_rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->attn.q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->attn.k_norm)
                    for (int h = 0; h < n_kv_heads; h++)
                        cpu_rmsnorm(key_cache_row + h*head_size, key_cache_row + h*head_size,
                                lw->attn.k_norm + h*qk_stride, head_size, c->norm_eps);

                bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
                bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                 layer_rope_dims, rope_cos, rope_sin);
            }

            // GQA attention (standard path — TQ handled above)
            if (!(c->kv_tq_bits > 0 && bn_model_tq_state(m))) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_cache_stride, c->seq_len };
                bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
            }

            // Attention sub-norm + wo projection + residual
            if (lw->norm.attn_sub_norm)
                cpu_rmsnorm(s->xb, s->xb, lw->norm.attn_sub_norm, dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL }};
                cpu_quant_matvec_batch_prepared(m, wo, 1, s->xb, s->x_q);
            }
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
    if (ffn_plan.kind == BN_FFN_MOE) {
        // MoE FFN — route, pread, compute, combine
        (void)moe_plan;
        bn_moe_forward(m, sess, lw, l);
    } else {
        // Dense FFN
        bn_transformer_cpu_forward_ffn_block(m, sess, lw, &ffn_plan);
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

    int ssm_preq8k = 0;
    int n_sb_ssm = dim / BN_QK_K;
    float ssm_q8k_d[n_sb_ssm > 0 ? n_sb_ssm : 1];
    int16_t ssm_q8k_bsums[n_sb_ssm > 0 ? n_sb_ssm * 16 : 1];
#ifdef __AVX2__
    int ssm_kquant = !bn_model_gpu(m) && dim % BN_QK_K == 0 &&
                     cpu_quant_can_preq8k_pair(lw->ssm.wqkv.type, lw->ssm.wz.type);
    if (ssm_kquant) {
        bn_quant_rmsnorm_q8k_avx2(s->x, lw->norm.attn_norm, dim, c->norm_eps,
                                  s->xb, s->x_q, ssm_q8k_d, ssm_q8k_bsums);
        ssm_preq8k = 1;
    } else
#endif
    {
        cpu_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim, c->norm_eps);
    }

    float *qkv = s->hb;
    float *z = s->hb2;
    BnMatvecTask qz_tasks[2] = {
         { qkv, &lw->ssm.wqkv, NULL },
        { z,   &lw->ssm.wz, NULL },
    };
    if (ssm_preq8k)
        cpu_quant_matvec_batch_preq8k(m, qz_tasks, 2, s->x_q,
                                     ssm_q8k_d, ssm_q8k_bsums, s->xb);
    else
        cpu_quant_matvec_batch_prepared(m, qz_tasks, 2, s->xb, s->x_q);

    BnSSMConvCtx conv_ctx = { qkv, conv_state, lw->ssm.ssm_conv1d, qkv_dim, kern };
    BnTPTask conv_task = { cpu_ssm_conv_silu, &conv_ctx, qkv_dim };
    bn_tp_dispatch(bn_model_pool(m), &conv_task, 1);

    float *q_raw = qkv;
    float *k_raw = qkv + key_dim;
    float *v_raw = qkv + 2 * key_dim;

    BnSSML2NormCtx norm_ctx = { q_raw, k_raw, head_k_dim };
    BnTPTask norm_task = { cpu_ssm_l2norm, &norm_ctx, num_k_heads };
    bn_tp_dispatch(bn_model_pool(m), &norm_task, 1);

    if (num_v_heads > 8192 || head_v_dim > 8192) {
        SH_LOG_ERROR("SSM dimensions too large for stack VLAs");
        return;
    }
    float alpha_arr[num_v_heads], beta_arr[num_v_heads];
    BnMatvecTask ab[2] = {
         { alpha_arr, &lw->ssm.ssm_alpha, NULL },
        { beta_arr,  &lw->ssm.ssm_beta, NULL },
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
    BnTPTask delta_task = { cpu_ssm_delta, &delta_ctx, num_v_heads };
    bn_tp_dispatch(bn_model_pool(m), &delta_task, 1);

    BnSSMGateCtx gate_ctx = { out, z, lw->ssm.ssm_norm, c->norm_eps, head_v_dim };
    BnTPTask gate_task = { cpu_ssm_gate, &gate_ctx, num_v_heads };
    bn_tp_dispatch(bn_model_pool(m), &gate_task, 1);

    BnMatvecTask proj[1] = {{ s->xb, &lw->ssm.ssm_out, NULL }};
    cpu_quant_matvec_batch_prepared(m, proj, 1, out, s->x_q);
}

void bn_transformer_cpu_forward_ffn_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
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

#ifdef __AVX2__
    if (ffn_plan->has_gate && !bn_model_gpu(m) && dim % BN_QK_K == 0 &&
        cpu_quant_can_preq8k_pair(lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)) {
        int n_sb = dim / BN_QK_K;
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_rmsnorm_q8k_avx2(s->x, lw->norm.ffn_norm, dim, c->norm_eps,
                                  s->xb, s->x_q, q8k_d, q8k_bsums);
        BnMatvecTask ffn[2] = {
             { s->hb,  &lw->ffn.ffn_gate, NULL },
            { s->hb2, &lw->ffn.ffn_up, NULL },
        };
        cpu_quant_matvec_batch_preq8k(m, ffn, 2, s->x_q, q8k_d, q8k_bsums,
                                     s->xb);
        fused_gate_up = 1;
    }
#endif

    if (!fused_gate_up) {
        cpu_rmsnorm(s->xb, s->x, lw->norm.ffn_norm, dim, c->norm_eps);

        if (ffn_plan->has_gate) {
            const BnPreparedWeight *gate_prepared =
                cpu_qweight_prepared(bn_model_backend(m), &lw->ffn.ffn_gate);
            const BnPreparedWeight *up_prepared =
                cpu_qweight_prepared(bn_model_backend(m), &lw->ffn.ffn_up);
            if (!bn_model_gpu(m) && ffn_plan->activation != 1 &&
                lw->ffn.ffn_gate.type == BN_GGUF_TENSOR_Q4_0 &&
                lw->ffn.ffn_up.type == BN_GGUF_TENSOR_Q4_0 &&
                dim % 32 == 0 &&
                bn_quant_q4_gate_up_silu(s->hb, &lw->ffn.ffn_gate, gate_prepared,
                                         &lw->ffn.ffn_up, up_prepared, s->xb,
                                         s->x_q, bn_model_pool(m)) == 0) {
                ffn_activated = 1;
            } else
            {
                BnMatvecTask ffn[2] = {
                     { s->hb,  &lw->ffn.ffn_gate, NULL },
                    { s->hb2, &lw->ffn.ffn_up, NULL },
                };
                cpu_quant_matvec_batch_prepared(m, ffn, 2, s->xb, s->x_q);
            }
        } else {
            BnMatvecTask ffn[1] = {{ s->hb, &lw->ffn.ffn_up, NULL }};
            cpu_quant_matvec_batch_prepared(m, ffn, 1, s->xb, s->x_q);
        }
    }

    bn_transformer_cpu_apply_ffn_activation(s, ffn_plan, hidden_dim, ffn_activated);

    if (ffn_plan->has_sub_norm)
        cpu_rmsnorm(s->hb, s->hb, lw->norm.ffn_sub_norm, hidden_dim, c->norm_eps);

    BnMatvecTask down[1] = {{ s->xb, &lw->ffn.ffn_down, NULL }};
    cpu_quant_matvec_batch_prepared(m, down, 1, s->hb, s->x_q);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
}
