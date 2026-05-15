#include "transformer_batched_attn_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_math_internal.h"
#include "quant.h"
#include "threadpool.h"
#include <string.h>
#include <math.h>

#define BATCHED_RMSNORM bn_transformer_rmsnorm_scalar

static inline float batched_dot(const float *a, const float *b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static inline float batched_dot_fp16(const float *a, const uint16_t *b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * bn_fp16_to_fp32(b[i]);
    return sum;
}

static inline void batched_acc_fp16(float *out, float w, const uint16_t *v, int n) {
    for (int i = 0; i < n; i++) out[i] += w * bn_fp16_to_fp32(v[i]);
}

static void batched_rope_token(float *v, int n_heads, int head_size,
                               int rope_dims, const float *rc, const float *rs,
                               int stride) {
    (void)head_size;
    int half = rope_dims / 2;
    for (int h = 0; h < n_heads; h++) {
        float *hd = v + h * stride;
        for (int i = 0; i < half; i++) {
            float v0 = hd[i], v1 = hd[half + i];
            hd[i]      = v0 * rc[i] - v1 * rs[i];
            hd[half + i] = v0 * rs[i] + v1 * rc[i];
        }
    }
}

void bn_transformer_batched_attn_naive_scalar_range(void *ctx, int h_start, int h_end) {
    BnBatchedAttnCtx *b = (BnBatchedAttnCtx *)ctx;
    BnRunState *s = b->s;
    int hs = b->head_size, kv_dim = b->kv_dim, kv_mul = b->kv_mul;
    int seq_len = b->seq_len, n_tokens = b->n_tokens, pos0 = b->pos0;
    size_t loff = b->loff;
    int kv_f16 = b->c->kv_f16;
    int rope_dims = b->rope_dims, half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_stride = b->q_gated ? 2 * hs : hs;
    float attn_scale = b->attention_scale;
    if (hs > BN_MAX_VLA_ELEMS) return;

    for (int h = h_start; h < h_end; h++) {
        int kv_h = h / kv_mul;
        for (int t = 0; t < n_tokens; t++) {
            int pos = pos0 + t;
            float *rc = b->rope_cos + (size_t)t * rope_stride;
            float *rs = b->rope_sin + (size_t)t * rope_stride;
            float *q_src = b->Q_buf + (size_t)t * b->wq_rows + h * q_stride;
            float q[hs];
            memcpy(q, q_src, hs * sizeof(float));
            if (b->q_bias) for (int d = 0; d < hs; d++) q[d] += b->q_bias[h * hs + d];
            if (b->q_norm) {
                int st = b->qk_norm_per_head ? hs : 0;
                BATCHED_RMSNORM(q, q, b->q_norm + h * st, hs, b->norm_eps);
            }
            batched_rope_token(q, 1, hs, rope_dims, rc, rs, hs);

            int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
            int kv_start = pos - n_kv + 1;
            float att[n_kv > BN_MAX_VLA_ELEMS ? 1 : n_kv];
            if (n_kv > BN_MAX_VLA_ELEMS) return;

            if (kv_f16) {
                const uint16_t *kc = (const uint16_t *)s->key_cache + loff;
                for (int i = 0; i < n_kv; i++)
                    att[i] = batched_dot_fp16(q, kc + (size_t)((kv_start + i) % seq_len) * kv_dim + kv_h * hs, hs) * attn_scale;
            } else {
                const float *kc = s->key_cache + loff;
                for (int i = 0; i < n_kv; i++)
                    att[i] = batched_dot(q, kc + (size_t)((kv_start + i) % seq_len) * kv_dim + kv_h * hs, hs) * attn_scale;
            }
            bn_transformer_softmax(att, n_kv);

            float xb[hs];
            memset(xb, 0, hs * sizeof(float));
            if (kv_f16) {
                const uint16_t *vc = (const uint16_t *)s->value_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    const uint16_t *vi = vc + (size_t)((kv_start + i) % seq_len) * kv_dim + kv_h * hs;
                    batched_acc_fp16(xb, att[i], vi, hs);
                }
            } else {
                const float *vc = s->value_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    float a = att[i];
                    const float *vi = vc + (size_t)((kv_start + i) % seq_len) * kv_dim + kv_h * hs;
                    for (int d = 0; d < hs; d++) xb[d] += a * vi[d];
                }
            }

            if (b->q_gated) {
                float *gate = q_src + hs;
                for (int d = 0; d < hs; d++)
                    xb[d] *= 1.0f / (1.0f + expf(-gate[d]));
            }
            memcpy(b->out + (size_t)t * b->wo_cols + h * hs, xb, hs * sizeof(float));
        }
    }
}

void bn_transformer_batched_attn_flash_scalar_range(void *ctx, int h_start, int h_end) {
    BnBatchedAttnCtx *b = (BnBatchedAttnCtx *)ctx;
    BnRunState *s = b->s;
    int hs = b->head_size, kv_dim = b->kv_dim, kv_mul = b->kv_mul;
    int seq_len = b->seq_len, n_tokens = b->n_tokens, pos0 = b->pos0;
    size_t loff = b->loff;
    int kv_f16 = b->c->kv_f16;
    int rope_dims = b->rope_dims, half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_stride = b->q_gated ? 2 * hs : hs;
    float attn_scale = b->attention_scale;
    if (hs > BN_MAX_VLA_ELEMS) return;

    for (int h = h_start; h < h_end; h++) {
        int kv_h = h / kv_mul;
        for (int t = 0; t < n_tokens; t++) {
            int pos = pos0 + t;
            float *rc = b->rope_cos + (size_t)t * rope_stride;
            float *rs = b->rope_sin + (size_t)t * rope_stride;
            float *q_src = b->Q_buf + (size_t)t * b->wq_rows + h * q_stride;
            float q[hs];
            memcpy(q, q_src, hs * sizeof(float));
            if (b->q_bias) for (int d = 0; d < hs; d++) q[d] += b->q_bias[h * hs + d];
            if (b->q_norm) {
                int st = b->qk_norm_per_head ? hs : 0;
                BATCHED_RMSNORM(q, q, b->q_norm + h * st, hs, b->norm_eps);
            }
            batched_rope_token(q, 1, hs, rope_dims, rc, rs, hs);

            int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
            int kv_start = pos - n_kv + 1;
            float out_buf[hs];
            memset(out_buf, 0, hs * sizeof(float));
            float rmax = -INFINITY, rsum = 0;

            for (int i = 0; i < n_kv; i++) {
                int ki = (kv_start + i) % seq_len;
                float score;
                if (kv_f16) {
                    const uint16_t *kc = (const uint16_t *)s->key_cache + loff;
                    score = batched_dot_fp16(q, kc + (size_t)ki * kv_dim + kv_h * hs, hs) * attn_scale;
                } else {
                    const float *kc = s->key_cache + loff;
                    score = batched_dot(q, kc + (size_t)ki * kv_dim + kv_h * hs, hs) * attn_scale;
                }
                if (score > rmax) {
                    float rescale = expf(rmax - score);
                    rmax = score; rsum *= rescale;
                    for (int d = 0; d < hs; d++) out_buf[d] *= rescale;
                }
                float w = expf(score - rmax); rsum += w;
                if (kv_f16) {
                    const uint16_t *vcache = (const uint16_t *)s->value_cache + loff;
                    batched_acc_fp16(out_buf, w, vcache + (size_t)ki * kv_dim + kv_h * hs, hs);
                } else {
                    const float *vcache = s->value_cache + loff;
                    const float *vi = vcache + (size_t)ki * kv_dim + kv_h * hs;
                    for (int d = 0; d < hs; d++) out_buf[d] += w * vi[d];
                }
            }

            float inv_sum = rsum > 0 ? 1.0f / rsum : 0;
            float *dst = b->out + (size_t)t * b->wo_cols + h * hs;
            if (b->q_gated) {
                float *gate = q_src + hs;
                for (int d = 0; d < hs; d++)
                    dst[d] = out_buf[d] * inv_sum / (1.0f + expf(-gate[d]));
            } else {
                for (int d = 0; d < hs; d++) dst[d] = out_buf[d] * inv_sum;
            }
        }
    }
}
