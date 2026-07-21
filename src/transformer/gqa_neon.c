#include "transformer_gqa_internal.h"
#include "transformer_simd_internal.h"

#ifdef __ARM_NEON

static inline float gqa_neon_dot_ggml_order(const float *x,
                                            const float *y,
                                            int n) {
    int i = 0;
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int np = n & ~15;
    for (; i < np; i += 16) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(x + i),      vld1q_f32(y + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(x + i + 4),  vld1q_f32(y + i + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(x + i + 8),  vld1q_f32(y + i + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(x + i + 12), vld1q_f32(y + i + 12));
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(sum0, sum1),
                                vaddq_f32(sum2, sum3));
    float out = vaddvq_f32(sum);
    for (; i < n; i++)
        out = fmaf(x[i], y[i], out);
    return out;
}

void bn_transformer_gqa_neon_range(void *ctx, int h_start, int h_end) {
    BnGQACtx *g = (BnGQACtx *)ctx;
    BnRunState *s = g->s;
    int head_size = g->head_size;
    int kv_dim = g->kv_dim;
    int kv_mul = g->kv_mul;
    int n_kv = g->n_kv;
    int seq_len = g->seq_len;
    int start = g->pos - n_kv + 1;
    size_t loff = g->loff;
    int kv_cache_uses_fp16_rows = g->kv_cache_uses_fp16_rows;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 4 != 0) return;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;
        float attn_scale = g->attention_scale;

        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            if (kv_cache_uses_fp16_rows) {
                const uint16_t *k_f16 = (const uint16_t *)s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                float k_buf[head_size];
                for (int d = 0; d < head_size; d += 4) {
                    vst1q_f32(k_buf + d,
                              vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k_f16 + d))));
                }
                att[i] = gqa_neon_dot_ggml_order(q_h, k_buf, head_size) * attn_scale;
            } else {
                const float *k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                att[i] = gqa_neon_dot_ggml_order(q_h, k_t, head_size) * attn_scale;
            }
        }

        bn_transformer_softmax(att, n_kv);

        float *xb_h = s->xb + h * head_size;
        memset(xb_h, 0, head_size * sizeof(float));
        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float a = att[i];
            float32x4_t a_v = vdupq_n_f32(a);
            if (kv_cache_uses_fp16_rows) {
                const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d += 16) {
                    vst1q_f32(xb_h + d,      vmlaq_f32(vld1q_f32(xb_h + d),      a_v, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v_f16 + d)))));
                    vst1q_f32(xb_h + d + 4,  vmlaq_f32(vld1q_f32(xb_h + d + 4),  a_v, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v_f16 + d + 4)))));
                    vst1q_f32(xb_h + d + 8,  vmlaq_f32(vld1q_f32(xb_h + d + 8),  a_v, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v_f16 + d + 8)))));
                    vst1q_f32(xb_h + d + 12, vmlaq_f32(vld1q_f32(xb_h + d + 12), a_v, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v_f16 + d + 12)))));
                }
            } else {
                const float *v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d += 16) {
                    vst1q_f32(xb_h + d,      vmlaq_f32(vld1q_f32(xb_h + d),      a_v, vld1q_f32(v_t + d)));
                    vst1q_f32(xb_h + d + 4,  vmlaq_f32(vld1q_f32(xb_h + d + 4),  a_v, vld1q_f32(v_t + d + 4)));
                    vst1q_f32(xb_h + d + 8,  vmlaq_f32(vld1q_f32(xb_h + d + 8),  a_v, vld1q_f32(v_t + d + 8)));
                    vst1q_f32(xb_h + d + 12, vmlaq_f32(vld1q_f32(xb_h + d + 12), a_v, vld1q_f32(v_t + d + 12)));
                }
            }
        }
    }
}

// --- Flash GQA attention (online softmax, per-head, single-pass) ---

#define FLASH_ATTN_TILE 64

void bn_transformer_flash_gqa_neon_range(void *ctx, int h_start, int h_end) {
    BnGQACtx *g = (BnGQACtx *)ctx;
    BnRunState *s = g->s;
    int head_size = g->head_size;
    int kv_dim = g->kv_dim;
    int kv_mul = g->kv_mul;
    int n_kv = g->n_kv;
    int seq_len = g->seq_len;
    int start = g->pos - n_kv + 1;
    size_t loff = g->loff;
    int kv_cache_uses_fp16_rows = g->kv_cache_uses_fp16_rows;
    float attn_scale = g->attention_scale;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 4 != 0) return;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        int kv_h = h / kv_mul;

        // Stack-allocated online softmax state
        float out_buf[head_size];
        memset(out_buf, 0, head_size * sizeof(float));
        float running_max = -INFINITY;
        float running_sum = 0.0f;

        // Single pass over KV cache in tiles
        for (int ti_start = 0; ti_start < n_kv; ti_start += FLASH_ATTN_TILE) {
            int ti_end = ti_start + FLASH_ATTN_TILE;
            if (ti_end > n_kv) ti_end = n_kv;

            for (int ti = ti_start; ti < ti_end; ti++) {
                int t = (start + ti) % seq_len;
                float k_buf[head_size];
                const float *k_t;
                if (kv_cache_uses_fp16_rows) {
                    const uint16_t *k_f16 = (const uint16_t *)s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                    for (int d = 0; d < head_size; d += 4) {
                        float16x4_t hv = vreinterpret_f16_u16(vld1_u16(k_f16 + d));
                        vst1q_f32(k_buf + d, vcvt_f32_f16(hv));
                    }
                    k_t = k_buf;
                } else {
                    k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                }

                if (ti + 1 < ti_end) {
                    int t_next = (start + ti + 1) % seq_len;
                    if (kv_cache_uses_fp16_rows)
                        __builtin_prefetch((const uint16_t *)s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size, 0, 0);
                    else
                        __builtin_prefetch(s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size, 0, 0);
                }

                // Score: dot(Q, K) * scale
                float score = gqa_neon_dot_ggml_order(q_h, k_t, head_size) *
                              attn_scale;

                // Online softmax update
                float v_buf[head_size];
                const float *v_t;
                if (kv_cache_uses_fp16_rows) {
                    const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                    for (int d = 0; d < head_size; d += 4) {
                        float16x4_t hv = vreinterpret_f16_u16(vld1_u16(v_f16 + d));
                        vst1q_f32(v_buf + d, vcvt_f32_f16(hv));
                    }
                    v_t = v_buf;
                } else {
                    v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                }
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
        float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        float32x4_t is = vdupq_n_f32(inv_sum);
        for (int d = 0; d < head_size; d += 4)
            vst1q_f32(xb_h + d, vmulq_f32(vld1q_f32(out_buf + d), is));
    }
}

#endif // __ARM_NEON
