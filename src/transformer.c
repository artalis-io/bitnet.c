#include "transformer.h"
#include "sh_log.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

static inline float neon_hsum_f32(float32x4_t v) {
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#endif

// --- Helper functions ---

static void rmsnorm(float *out, const float *x, const float *w, int size, float eps) {
#ifdef __ARM_NEON
    float32x4_t sum_sq = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        sum_sq = vmlaq_f32(sum_sq, xv, xv);
    }
    float ss = neon_hsum_f32(sum_sq);
    for (; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    float32x4_t ss_v = vdupq_n_f32(ss);
    for (i = 0; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, ss_v), wv));
    }
    for (; i < size; i++) out[i] = x[i] * ss * w[i];
#else
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
#endif
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// --- GQA context (shared by both attention paths) ---

typedef struct {
    const BnConfig *c;
    BnRunState *s;
    size_t loff;
    int pos;
    int kv_mul;
    int head_size;
    int kv_dim;
} GQACtx;

static void gqa_range(void *ctx, int h_start, int h_end) {
    GQACtx *g = (GQACtx *)ctx;
    const BnConfig *c = g->c;
    BnRunState *s = g->s;
    int head_size = g->head_size;
    int kv_dim = g->kv_dim;
    int kv_mul = g->kv_mul;
    int pos = g->pos;
    size_t loff = g->loff;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * c->seq_len;
        int kv_h = h / kv_mul;
        float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);

        for (int t = 0; t <= pos; t++) {
            float *k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
#ifdef __ARM_NEON
            float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
            float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
            for (int d = 0; d < head_size; d += 16) {
                a0 = vmlaq_f32(a0, vld1q_f32(q_h + d),      vld1q_f32(k_t + d));
                a1 = vmlaq_f32(a1, vld1q_f32(q_h + d + 4),  vld1q_f32(k_t + d + 4));
                a2 = vmlaq_f32(a2, vld1q_f32(q_h + d + 8),  vld1q_f32(k_t + d + 8));
                a3 = vmlaq_f32(a3, vld1q_f32(q_h + d + 12), vld1q_f32(k_t + d + 12));
            }
            float32x4_t sum = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            att[t] = neon_hsum_f32(sum) * inv_sqrt_hs;
#else
            float score = 0.0f;
            for (int d = 0; d < head_size; d++) score += q_h[d] * k_t[d];
            att[t] = score * inv_sqrt_hs;
#endif
        }

        softmax(att, pos + 1);

        float *xb_h = s->xb + h * head_size;
        memset(xb_h, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float *v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
            float a = att[t];
#ifdef __ARM_NEON
            float32x4_t a_v = vdupq_n_f32(a);
            for (int d = 0; d < head_size; d += 16) {
                vst1q_f32(xb_h + d,      vmlaq_f32(vld1q_f32(xb_h + d),      a_v, vld1q_f32(v_t + d)));
                vst1q_f32(xb_h + d + 4,  vmlaq_f32(vld1q_f32(xb_h + d + 4),  a_v, vld1q_f32(v_t + d + 4)));
                vst1q_f32(xb_h + d + 8,  vmlaq_f32(vld1q_f32(xb_h + d + 8),  a_v, vld1q_f32(v_t + d + 8)));
                vst1q_f32(xb_h + d + 12, vmlaq_f32(vld1q_f32(xb_h + d + 12), a_v, vld1q_f32(v_t + d + 12)));
            }
#else
            for (int d = 0; d < head_size; d++) xb_h[d] += a * v_t[d];
#endif
        }
    }
}

// --- Flash GQA attention (online softmax, per-head, single-pass) ---

#ifdef __ARM_NEON
#define FLASH_ATTN_TILE 64

static void flash_gqa_range(void *ctx, int h_start, int h_end) {
    GQACtx *g = (GQACtx *)ctx;
    BnRunState *s = g->s;
    int head_size = g->head_size;
    int kv_dim = g->kv_dim;
    int kv_mul = g->kv_mul;
    int pos = g->pos;
    size_t loff = g->loff;
    int n_pos = pos + 1;
    float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        int kv_h = h / kv_mul;

        // Stack-allocated online softmax state
        float out_buf[head_size];
        memset(out_buf, 0, head_size * sizeof(float));
        float running_max = -INFINITY;
        float running_sum = 0.0f;

        // Single pass over KV cache in tiles
        for (int t_start = 0; t_start < n_pos; t_start += FLASH_ATTN_TILE) {
            int t_end = t_start + FLASH_ATTN_TILE;
            if (t_end > n_pos) t_end = n_pos;

            for (int t = t_start; t < t_end; t++) {
                float *k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;

                if (t + 1 < t_end)
                    __builtin_prefetch(s->key_cache + loff + (size_t)(t+1) * kv_dim + kv_h * head_size, 0, 0);

                // Score: dot(Q, K) * scale
                float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
                for (int d = 0; d < head_size; d += 16) {
                    a0 = vmlaq_f32(a0, vld1q_f32(q_h + d),      vld1q_f32(k_t + d));
                    a1 = vmlaq_f32(a1, vld1q_f32(q_h + d + 4),  vld1q_f32(k_t + d + 4));
                    a2 = vmlaq_f32(a2, vld1q_f32(q_h + d + 8),  vld1q_f32(k_t + d + 8));
                    a3 = vmlaq_f32(a3, vld1q_f32(q_h + d + 12), vld1q_f32(k_t + d + 12));
                }
                float score = neon_hsum_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * inv_sqrt_hs;

                // Online softmax update
                float *v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                __builtin_prefetch(v_t, 0, 0);

                float old_max = running_max;
                if (score > old_max) {
                    float rescale = expf(old_max - score);
                    running_max = score;
                    running_sum *= rescale;
                    float32x4_t rs = vdupq_n_f32(rescale);
                    for (int d = 0; d < head_size; d += 4)
                        vst1q_f32(out_buf + d, vmulq_f32(vld1q_f32(out_buf + d), rs));
                }

                float w = expf(score - running_max);
                running_sum += w;
                float32x4_t wv = vdupq_n_f32(w);
                for (int d = 0; d < head_size; d += 4)
                    vst1q_f32(out_buf + d, vmlaq_f32(vld1q_f32(out_buf + d), wv, vld1q_f32(v_t + d)));
            }
        }

        // Finalize: output = out_buf / running_sum
        float *xb_h = s->xb + h * head_size;
        float inv_sum = 1.0f / running_sum;
        float32x4_t is = vdupq_n_f32(inv_sum);
        for (int d = 0; d < head_size; d += 4)
            vst1q_f32(xb_h + d, vmulq_f32(vld1q_f32(out_buf + d), is));
    }
}
#endif // __ARM_NEON

// --- Logits range functions ---

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
typedef struct {
    float *logits;
    const int8_t *emb_i8;
    const float *emb_scales;
    const int8_t *x_q;
    float x_scale;
    int dim;
} LogitsI8Ctx;

static void logits_i8_sdot_range(void *ctx, int v_start, int v_end) {
    LogitsI8Ctx *lc = (LogitsI8Ctx *)ctx;
    const int8_t *emb_i8 = lc->emb_i8;
    const float *emb_scales = lc->emb_scales;
    const int8_t *x_q = lc->x_q;
    float x_scale = lc->x_scale;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const int8_t *row = emb_i8 + (size_t)v * dim;
        __builtin_prefetch(row + (size_t)dim, 0, 0);
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
        for (int d = 0; d < dim; d += 64) {
            __builtin_prefetch(row + d + 128, 0, 0);
            acc0 = vdotq_s32(acc0, vld1q_s8(row+d),    vld1q_s8(x_q+d));
            acc1 = vdotq_s32(acc1, vld1q_s8(row+d+16), vld1q_s8(x_q+d+16));
            acc2 = vdotq_s32(acc2, vld1q_s8(row+d+32), vld1q_s8(x_q+d+32));
            acc3 = vdotq_s32(acc3, vld1q_s8(row+d+48), vld1q_s8(x_q+d+48));
        }
        int32x4_t sum4 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        int32_t total = vaddvq_s32(sum4);
        lc->logits[v] = (float)total * emb_scales[v] * x_scale;
    }
}
#endif // __ARM_NEON && __ARM_FEATURE_DOTPROD

typedef struct {
    float *logits;
    const float *x;
    const void *emb;
    int dim;
} LogitsCtx;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void logits_f16_native_range(void *ctx, int v_start, int v_end) {
    LogitsCtx *lc = (LogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const uint16_t *x_f16 = (const uint16_t *)lc->x;  // pre-converted
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        float32x4_t fsum = vdupq_n_f32(0);
        const float16x8_t fz = vreinterpretq_f16_u16(vdupq_n_u16(0));
        int d = 0;

        #define LDF16(p) vreinterpretq_f16_u16(vld1q_u16(p))
        for (; d + 63 < dim; d += 64) {
            float16x8_t a0 = fz, a1 = fz, a2 = fz, a3 = fz;
            a0 = vfmaq_f16(a0, LDF16(row+d),    LDF16(x_f16+d));
            a1 = vfmaq_f16(a1, LDF16(row+d+8),  LDF16(x_f16+d+8));
            a2 = vfmaq_f16(a2, LDF16(row+d+16), LDF16(x_f16+d+16));
            a3 = vfmaq_f16(a3, LDF16(row+d+24), LDF16(x_f16+d+24));
            a0 = vfmaq_f16(a0, LDF16(row+d+32), LDF16(x_f16+d+32));
            a1 = vfmaq_f16(a1, LDF16(row+d+40), LDF16(x_f16+d+40));
            a2 = vfmaq_f16(a2, LDF16(row+d+48), LDF16(x_f16+d+48));
            a3 = vfmaq_f16(a3, LDF16(row+d+56), LDF16(x_f16+d+56));
            float16x8_t s = vaddq_f16(vaddq_f16(a0, a1), vaddq_f16(a2, a3));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_low_f16(s)));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_high_f16(s)));
        }
        for (; d + 7 < dim; d += 8) {
            float16x8_t p = vmulq_f16(LDF16(row+d), LDF16(x_f16+d));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_low_f16(p)));
            fsum = vaddq_f32(fsum, vcvt_f32_f16(vget_high_f16(p)));
        }
        #undef LDF16

        lc->logits[v] = neon_hsum_f32(fsum);
    }
}
#endif

#if defined(__ARM_NEON) && !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void logits_f16_neon_range(void *ctx, int v_start, int v_end) {
    LogitsCtx *lc = (LogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        for (int d = 0; d < dim; d += 8) {
            float16x8_t f16 = vreinterpretq_f16_u16(vld1q_u16(row + d));
            acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vget_low_f16(f16)),  vld1q_f32(x + d));
            acc1 = vmlaq_f32(acc1, vcvt_f32_f16(vget_high_f16(f16)), vld1q_f32(x + d + 4));
        }
        lc->logits[v] = neon_hsum_f32(vaddq_f32(acc0, acc1));
    }
}
#endif

#if !defined(__ARM_NEON)
static void logits_f16_scalar_range(void *ctx, int v_start, int v_end) {
    LogitsCtx *lc = (LogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += bn_fp16_to_fp32(row[d]) * x[d];
        }
        lc->logits[v] = sum;
    }
}
#endif // !__ARM_NEON

static void logits_f32_range(void *ctx, int v_start, int v_end) {
    LogitsCtx *lc = (LogitsCtx *)ctx;
    const float *emb = (const float *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const float *row = emb + (size_t)v * dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += row[d] * x[d];
        }
        lc->logits[v] = sum;
    }
}

// --- Forward pass ---

float *bn_transformer_forward(BnModel *m, int token, int pos) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul;
    int head_size = c->head_size;

    // #9: Validate token bounds
    if (token < 0 || token >= c->vocab_size) {
        SH_LOG_ERROR("Token out of range");
        return NULL;
    }

    // #10: Validate pos bounds to prevent KV-cache OOB write
    if (pos < 0 || pos >= c->seq_len) {
        SH_LOG_ERROR("Position out of range");
        return NULL;
    }

    // Embed the token
    bn_model_embed_token(m, s->x, token);

    // Precompute RoPE cos/sin for this position (128 trig calls total,
    // vs 96,000 if computed per-head per-layer)
    int half_head = head_size / 2;
    float rope_cos[half_head], rope_sin[half_head];
    for (int i = 0; i < half_head; i++) {
        float angle = pos * s->rope_freq[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }

    // Process each layer
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        size_t loff = (size_t)l * c->seq_len * kv_dim;
        float *key_cache_row   = s->key_cache   + loff + (size_t)pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + (size_t)pos * kv_dim;

        // ---- Attention block ----

        rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);

        // QKV projections (unified path — bn_quant_matvec_batch handles SDOT internally)
        {
            BnMatvecTask qkv[3] = {
                { s->q,            &lw->wq },
                { key_cache_row,   &lw->wk },
                { value_cache_row, &lw->wv },
            };
            bn_quant_matvec_batch(qkv, 3, s->xb, s->x_q, m->pool);
        }

        // RoPE using precomputed cos/sin (no trig calls here)
        for (int i = 0; i < dim; i += 2) {
            int fi = (i / 2) % half_head;
            float v0 = s->q[i], v1 = s->q[i + 1];
            s->q[i]     = v0 * rope_cos[fi] - v1 * rope_sin[fi];
            s->q[i + 1] = v0 * rope_sin[fi] + v1 * rope_cos[fi];
        }
        for (int i = 0; i < kv_dim; i += 2) {
            int fi = (i / 2) % half_head;
            float v0 = key_cache_row[i], v1 = key_cache_row[i + 1];
            key_cache_row[i]     = v0 * rope_cos[fi] - v1 * rope_sin[fi];
            key_cache_row[i + 1] = v0 * rope_sin[fi] + v1 * rope_cos[fi];
        }

        // GQA attention
        {
            GQACtx gctx = { c, s, loff, pos, kv_mul, head_size, kv_dim };
#ifdef __ARM_NEON
            bn_tp_fn attn_fn = c->flash_attn ? flash_gqa_range : gqa_range;
#else
            bn_tp_fn attn_fn = gqa_range;
#endif
            BnTPTask gqa = { attn_fn, &gctx, c->n_heads };
            bn_tp_dispatch(m->pool, &gqa, 1);
        }

        // Attention sub-norm + wo projection
        if (lw->attn_sub_norm)
            rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);

        {
            BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
            bn_quant_matvec_batch(wo, 1, s->xb, s->x_q, m->pool);
        }

        // Residual connection
#ifdef __ARM_NEON
        for (int i = 0; i < dim; i += 4)
            vst1q_f32(s->x + i, vaddq_f32(vld1q_f32(s->x + i), vld1q_f32(s->xb2 + i)));
#else
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
#endif

        // ---- FFN block ----

        rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

        if (c->has_ffn_gate) {
            // Gate + Up in one dispatch
            {
                BnMatvecTask ffn[2] = {
                    { s->hb,  &lw->ffn_gate },
                    { s->hb2, &lw->ffn_up   },
                };
                bn_quant_matvec_batch(ffn, 2, s->xb, s->x_q, m->pool);
            }

            // Activation
            if (c->act_type == 1) {
#ifdef __ARM_NEON
                float32x4_t zero = vdupq_n_f32(0);
                for (int i = 0; i < hidden_dim; i += 4) {
                    float32x4_t g = vmaxq_f32(vld1q_f32(s->hb + i), zero);
                    vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(g, g), vld1q_f32(s->hb2 + i)));
                }
#else
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i] > 0 ? s->hb[i] : 0;
                    s->hb[i] = g * g * s->hb2[i];
                }
#endif
            } else {
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i];
                    s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
                }
            }
        } else {
            {
                BnMatvecTask ffn[1] = {{ s->hb, &lw->ffn_up }};
                bn_quant_matvec_batch(ffn, 1, s->xb, s->x_q, m->pool);
            }

            if (c->act_type == 1) {
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i] > 0 ? s->hb[i] : 0;
                    s->hb[i] = v * v;
                }
            } else {
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i];
                    s->hb[i] = v / (1.0f + expf(-v));
                }
            }
        }

        // FFN sub-norm + down projection
        if (lw->ffn_sub_norm)
            rmsnorm(s->hb, s->hb, lw->ffn_sub_norm, hidden_dim, c->norm_eps);

        {
            BnMatvecTask down[1] = {{ s->xb, &lw->ffn_down }};
            bn_quant_matvec_batch(down, 1, s->hb, s->x_q, m->pool);
        }

        // Residual connection
#ifdef __ARM_NEON
        for (int i = 0; i < dim; i += 4)
            vst1q_f32(s->x + i, vaddq_f32(vld1q_f32(s->x + i), vld1q_f32(s->xb + i)));
#else
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
#endif

        if (l == 0 && pos == 0) {
            char v0[16], v1[16], v2[16], v3[16];
            snprintf(v0, sizeof(v0), "%.6f", s->x[0]);
            snprintf(v1, sizeof(v1), "%.6f", s->x[1]);
            snprintf(v2, sizeof(v2), "%.6f", s->x[2]);
            snprintf(v3, sizeof(v3), "%.6f", s->x[3]);
            SH_LOG_DEBUG("Layer 0 pos 0", "x0", v0, "x1", v1, "x2", v2, "x3", v3);
        }
    }

    // Final RMSNorm
    rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);

    // Tied embeddings: logits = token_embedding^T @ x
    if (m->weights.emb_type == BN_GGUF_TENSOR_F16) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (w->emb_out_i8) {
            // INT8 SDOT path: quantize x once, dot against pre-quantized embeddings
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            LogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { logits_i8_sdot_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#endif
        {
            const uint16_t *emb = (const uint16_t *)w->token_embedding;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            // Convert x to F16 once for native F16 FMA
            uint16_t x_f16[dim];
            for (int d = 0; d < dim; d += 8) {
                float16x4_t lo = vcvt_f16_f32(vld1q_f32(s->x + d));
                float16x4_t hi = vcvt_f16_f32(vld1q_f32(s->x + d + 4));
                vst1q_u16(x_f16 + d, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
            }
            LogitsCtx lctx = { s->logits, (const float *)(void *)x_f16, emb, dim };
            BnTPTask logits_task = { logits_f16_native_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__ARM_NEON)
            LogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { logits_f16_neon_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#else
            LogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { logits_f16_scalar_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#endif
        }
    } else {
        // F32 embeddings
        const float *emb = (const float *)w->token_embedding;
        LogitsCtx lctx = { s->logits, s->x, emb, dim };
        BnTPTask logits_task = { logits_f32_range, &lctx, c->vocab_size };
        bn_tp_dispatch(m->pool, &logits_task, 1);
    }

    return s->logits;
}
