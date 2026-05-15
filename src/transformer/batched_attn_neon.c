#include "transformer_batched_attn_internal.h"
#include "transformer_math_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_simd_internal.h"
#include "simd_helpers.h"
#include <math.h>
#include <string.h>

#ifdef __ARM_NEON

#define BATCHED_RMSNORM bn_transformer_rmsnorm_neon
#define FLASH_ATTN_TILE 64

static inline float batched_dot_fp32_neon(const float *q, const float *k,
                                          int head_size) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 15 < head_size; d += 16) {
        a0 = vmlaq_f32(a0, vld1q_f32(q + d),      vld1q_f32(k + d));
        a1 = vmlaq_f32(a1, vld1q_f32(q + d + 4),  vld1q_f32(k + d + 4));
        a2 = vmlaq_f32(a2, vld1q_f32(q + d + 8),  vld1q_f32(k + d + 8));
        a3 = vmlaq_f32(a3, vld1q_f32(q + d + 12), vld1q_f32(k + d + 12));
    }
    for (; d + 3 < head_size; d += 4)
        a0 = vmlaq_f32(a0, vld1q_f32(q + d), vld1q_f32(k + d));
    float sum = bn_transformer_neon_hsum_f32(vaddq_f32(vaddq_f32(a0, a1),
                                                       vaddq_f32(a2, a3)));
    for (; d < head_size; d++) sum += q[d] * k[d];
    return sum;
}

static inline float batched_dot_fp16_neon(const float *q, const uint16_t *k,
                                          int head_size) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 15 < head_size; d += 16) {
        a0 = vmlaq_f32(a0, vld1q_f32(q + d),
                       vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k + d))));
        a1 = vmlaq_f32(a1, vld1q_f32(q + d + 4),
                       vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k + d + 4))));
        a2 = vmlaq_f32(a2, vld1q_f32(q + d + 8),
                       vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k + d + 8))));
        a3 = vmlaq_f32(a3, vld1q_f32(q + d + 12),
                       vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k + d + 12))));
    }
    for (; d + 3 < head_size; d += 4)
        a0 = vmlaq_f32(a0, vld1q_f32(q + d),
                       vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k + d))));
    float sum = bn_transformer_neon_hsum_f32(vaddq_f32(vaddq_f32(a0, a1),
                                                       vaddq_f32(a2, a3)));
    for (; d < head_size; d++) {
        float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(k[d]));
        sum += q[d] * vgetq_lane_f32(vcvt_f32_f16(hv), 0);
    }
    return sum;
}

static inline void batched_acc_v_fp32_neon(float *out, float w,
                                           const float *v, int head_size) {
    float32x4_t wv = vdupq_n_f32(w);
    int d = 0;
    for (; d + 15 < head_size; d += 16) {
        vst1q_f32(out + d,      vmlaq_f32(vld1q_f32(out + d),      wv, vld1q_f32(v + d)));
        vst1q_f32(out + d + 4,  vmlaq_f32(vld1q_f32(out + d + 4),  wv, vld1q_f32(v + d + 4)));
        vst1q_f32(out + d + 8,  vmlaq_f32(vld1q_f32(out + d + 8),  wv, vld1q_f32(v + d + 8)));
        vst1q_f32(out + d + 12, vmlaq_f32(vld1q_f32(out + d + 12), wv, vld1q_f32(v + d + 12)));
    }
    for (; d + 3 < head_size; d += 4)
        vst1q_f32(out + d, vmlaq_f32(vld1q_f32(out + d), wv, vld1q_f32(v + d)));
    for (; d < head_size; d++) out[d] += w * v[d];
}

static inline void batched_acc_v_fp16_neon(float *out, float w,
                                           const uint16_t *v, int head_size) {
    float32x4_t wv = vdupq_n_f32(w);
    int d = 0;
    for (; d + 15 < head_size; d += 16) {
        vst1q_f32(out + d,
                  vmlaq_f32(vld1q_f32(out + d), wv,
                             vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v + d)))));
        vst1q_f32(out + d + 4,
                  vmlaq_f32(vld1q_f32(out + d + 4), wv,
                             vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v + d + 4)))));
        vst1q_f32(out + d + 8,
                  vmlaq_f32(vld1q_f32(out + d + 8), wv,
                             vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v + d + 8)))));
        vst1q_f32(out + d + 12,
                  vmlaq_f32(vld1q_f32(out + d + 12), wv,
                             vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v + d + 12)))));
    }
    for (; d + 3 < head_size; d += 4)
        vst1q_f32(out + d,
                  vmlaq_f32(vld1q_f32(out + d), wv,
                             vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v + d)))));
    for (; d < head_size; d++) {
        float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(v[d]));
        out[d] += w * vgetq_lane_f32(vcvt_f32_f16(hv), 0);
    }
}

static void batched_apply_rope_neon(float *q, int head_size, int rope_dims,
                                    const float *rc, const float *rs) {
    int half_rope = rope_dims / 2;
    int i = 0;
    for (; i + 3 < half_rope; i += 4) {
        float32x4_t v0 = vld1q_f32(q + i);
        float32x4_t v1 = vld1q_f32(q + half_rope + i);
        float32x4_t cv = vld1q_f32(rc + i);
        float32x4_t sv = vld1q_f32(rs + i);
        vst1q_f32(q + i,              vsubq_f32(vmulq_f32(v0, cv), vmulq_f32(v1, sv)));
        vst1q_f32(q + half_rope + i,  vaddq_f32(vmulq_f32(v0, sv), vmulq_f32(v1, cv)));
    }
    for (; i < half_rope; i++) {
        float v0 = q[i];
        float v1 = q[half_rope + i];
        q[i] = v0 * rc[i] - v1 * rs[i];
        q[half_rope + i] = v0 * rs[i] + v1 * rc[i];
    }
    (void)head_size;
}

static inline void batched_prepare_q(BnBatchedAttnCtx *b, int h, int t,
                                     float *q_local) {
    int head_size = b->head_size;
    int half_rope = b->rope_dims / 2;
    int q_head_stride = b->q_gated ? 2 * head_size : head_size;
    float *q_src = b->Q_buf + (size_t)t * b->wq_rows + h * q_head_stride;
    memcpy(q_local, q_src, (size_t)head_size * sizeof(float));
    if (b->q_bias) {
        int d = 0;
        for (; d + 3 < head_size; d += 4) {
            vst1q_f32(q_local + d,
                      vaddq_f32(vld1q_f32(q_local + d),
                                vld1q_f32(b->q_bias + h * head_size + d)));
        }
        for (; d < head_size; d++)
            q_local[d] += b->q_bias[h * head_size + d];
    }
    if (b->q_norm) {
        int stride = b->qk_norm_per_head ? head_size : 0;
        BATCHED_RMSNORM(q_local, q_local, b->q_norm + h * stride,
                        head_size, b->norm_eps);
    }
    batched_apply_rope_neon(q_local, head_size, b->rope_dims,
                            b->rope_cos + (size_t)t * half_rope,
                            b->rope_sin + (size_t)t * half_rope);
}

static inline void batched_store_output(BnBatchedAttnCtx *b, int h, int t,
                                        const float *out_buf, float inv_sum) {
    int head_size = b->head_size;
    int q_head_stride = b->q_gated ? 2 * head_size : head_size;
    float *dst = b->out + (size_t)t * b->wo_cols + h * head_size;
    float32x4_t is = vdupq_n_f32(inv_sum);
    int d = 0;
    if (b->q_gated) {
        float *gate = b->Q_buf + (size_t)t * b->wq_rows + h * q_head_stride + head_size;
        for (; d + 3 < head_size; d += 4) {
            float32x4_t o = vmulq_f32(vld1q_f32(out_buf + d), is);
            vst1q_f32(dst + d, vmulq_f32(o, bn_neon_fast_sigmoid_f32(vld1q_f32(gate + d))));
        }
        for (; d < head_size; d++)
            dst[d] = out_buf[d] * inv_sum / (1.0f + expf(-gate[d]));
    } else {
        for (; d + 3 < head_size; d += 4)
            vst1q_f32(dst + d, vmulq_f32(vld1q_f32(out_buf + d), is));
        for (; d < head_size; d++) dst[d] = out_buf[d] * inv_sum;
    }
}

void bn_transformer_batched_attn_naive_neon_range(void *ctx, int h_start,
                                                  int h_end) {
    BnBatchedAttnCtx *b = (BnBatchedAttnCtx *)ctx;
    BnRunState *s = b->s;
    int head_size = b->head_size;
    int kv_dim = b->kv_dim;
    int kv_mul = b->kv_mul;
    int seq_len = b->seq_len;
    int n_tokens = b->n_tokens;
    int pos0 = b->pos0;
    size_t loff = b->loff;
    int kv_f16 = b->c->kv_f16;
    float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);
    if (head_size > BN_MAX_VLA_ELEMS) return;

    for (int h = h_start; h < h_end; h++) {
        int kv_h = h / kv_mul;
        for (int t = 0; t < n_tokens; t++) {
            int pos = pos0 + t;
            int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
            int kv_start = pos - n_kv + 1;
            if (n_kv > BN_MAX_VLA_ELEMS) return;

            float q_local[head_size];
            float att[n_kv > 0 ? n_kv : 1];
            float xb_local[head_size];
            batched_prepare_q(b, h, t, q_local);

            if (kv_f16) {
                const uint16_t *kc_base = (const uint16_t *)s->key_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    att[i] = batched_dot_fp16_neon(q_local,
                                                   kc_base + (size_t)ki * kv_dim + kv_h * head_size,
                                                   head_size) * inv_sqrt_hs;
                }
            } else {
                const float *kc_base = s->key_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    att[i] = batched_dot_fp32_neon(q_local,
                                                   kc_base + (size_t)ki * kv_dim + kv_h * head_size,
                                                   head_size) * inv_sqrt_hs;
                }
            }

            bn_transformer_softmax(att, n_kv);
            memset(xb_local, 0, (size_t)head_size * sizeof(float));
            if (kv_f16) {
                const uint16_t *vc_base = (const uint16_t *)s->value_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    batched_acc_v_fp16_neon(xb_local, att[i],
                                            vc_base + (size_t)ki * kv_dim + kv_h * head_size,
                                            head_size);
                }
            } else {
                const float *vc_base = s->value_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    batched_acc_v_fp32_neon(xb_local, att[i],
                                            vc_base + (size_t)ki * kv_dim + kv_h * head_size,
                                            head_size);
                }
            }
            batched_store_output(b, h, t, xb_local, 1.0f);
        }
    }
}

void bn_transformer_batched_attn_flash_neon_range(void *ctx, int h_start,
                                                  int h_end) {
    BnBatchedAttnCtx *b = (BnBatchedAttnCtx *)ctx;
    BnRunState *s = b->s;
    int head_size = b->head_size;
    int kv_dim = b->kv_dim;
    int kv_mul = b->kv_mul;
    int seq_len = b->seq_len;
    int n_tokens = b->n_tokens;
    int pos0 = b->pos0;
    size_t loff = b->loff;
    int kv_f16 = b->c->kv_f16;
    float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);
    if (head_size > BN_MAX_VLA_ELEMS) return;

    for (int h = h_start; h < h_end; h++) {
        int kv_h = h / kv_mul;
        for (int t = 0; t < n_tokens; t++) {
            int pos = pos0 + t;
            int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
            int kv_start = pos - n_kv + 1;
            float q_local[head_size];
            float out_buf[head_size];
            batched_prepare_q(b, h, t, q_local);
            memset(out_buf, 0, (size_t)head_size * sizeof(float));
            float running_max = -INFINITY;
            float running_sum = 0.0f;

            for (int ti_start = 0; ti_start < n_kv; ti_start += FLASH_ATTN_TILE) {
                int ti_end = ti_start + FLASH_ATTN_TILE;
                if (ti_end > n_kv) ti_end = n_kv;
                for (int ti = ti_start; ti < ti_end; ti++) {
                    int ki = (kv_start + ti) % seq_len;
                    float score;
                    if (kv_f16) {
                        score = batched_dot_fp16_neon(q_local,
                                                      (const uint16_t *)s->key_cache + loff +
                                                      (size_t)ki * kv_dim + kv_h * head_size,
                                                      head_size) * inv_sqrt_hs;
                    } else {
                        score = batched_dot_fp32_neon(q_local,
                                                      s->key_cache + loff +
                                                      (size_t)ki * kv_dim + kv_h * head_size,
                                                      head_size) * inv_sqrt_hs;
                    }

                    float old_max = running_max;
                    if (score > old_max) {
                        float rescale = expf(old_max - score);
                        running_max = score;
                        running_sum *= rescale;
                        float32x4_t rs = vdupq_n_f32(rescale);
                        int d = 0;
                        for (; d + 3 < head_size; d += 4)
                            vst1q_f32(out_buf + d, vmulq_f32(vld1q_f32(out_buf + d), rs));
                        for (; d < head_size; d++) out_buf[d] *= rescale;
                    }

                    float w = expf(score - running_max);
                    running_sum += w;
                    if (kv_f16) {
                        batched_acc_v_fp16_neon(out_buf, w,
                                                (const uint16_t *)s->value_cache + loff +
                                                (size_t)ki * kv_dim + kv_h * head_size,
                                                head_size);
                    } else {
                        batched_acc_v_fp32_neon(out_buf, w,
                                                s->value_cache + loff +
                                                (size_t)ki * kv_dim + kv_h * head_size,
                                                head_size);
                    }
                }
            }

            float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
            batched_store_output(b, h, t, out_buf, inv_sum);
        }
    }
}

#endif
