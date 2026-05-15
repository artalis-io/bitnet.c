#include "transformer_batched_attn_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_math_internal.h"
#include "simd_helpers.h"
#include "threadpool.h"
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>

#define BATCHED_RMSNORM bn_transformer_rmsnorm_avx2

static inline float batched_dot_fp32(const float *q, const float *k, int head_size) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 31 < head_size; d += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d),      _mm256_loadu_ps(k + d), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8),  _mm256_loadu_ps(k + d + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 16), _mm256_loadu_ps(k + d + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 24), _mm256_loadu_ps(k + d + 24), a3);
    }
    for (; d + 7 < head_size; d += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(k + d), a0);
    return bn_avx2_hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}

static inline float batched_dot_fp16(const float *q, const uint16_t *k, int head_size) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 31 < head_size; d += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d),      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d))), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8),  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d + 8))), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 16), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d + 16))), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 24), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d + 24))), a3);
    }
    for (; d + 7 < head_size; d += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d))), a0);
    return bn_avx2_hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}

static inline void batched_acc_v_fp32(float *out, float w, const float *v, int head_size) {
    __m256 wv = _mm256_set1_ps(w);
    int d = 0;
    for (; d + 31 < head_size; d += 32) {
        _mm256_storeu_ps(out + d,      _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d),      _mm256_loadu_ps(out + d)));
        _mm256_storeu_ps(out + d + 8,  _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d + 8),  _mm256_loadu_ps(out + d + 8)));
        _mm256_storeu_ps(out + d + 16, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d + 16), _mm256_loadu_ps(out + d + 16)));
        _mm256_storeu_ps(out + d + 24, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d + 24), _mm256_loadu_ps(out + d + 24)));
    }
    for (; d + 7 < head_size; d += 8)
        _mm256_storeu_ps(out + d, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d), _mm256_loadu_ps(out + d)));
}

static inline void batched_acc_v_fp16(float *out, float w, const uint16_t *v, int head_size) {
    __m256 wv = _mm256_set1_ps(w);
    for (int d = 0; d < head_size; d += 8)
        _mm256_storeu_ps(out + d, _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v + d))), _mm256_loadu_ps(out + d)));
}

static void batched_apply_rope_token(float *q, int n_heads, int head_size,
                                     int rope_dims, const float *rc, const float *rs,
                                     int q_stride) {
    (void)head_size;
    int half_rope = rope_dims / 2;
    for (int h = 0; h < n_heads; h++) {
        float *hd = q + h * q_stride;
        int i = 0;
        for (; i + 7 < half_rope; i += 8) {
            __m256 v0 = _mm256_loadu_ps(hd + i);
            __m256 v1 = _mm256_loadu_ps(hd + half_rope + i);
            __m256 cos_v = _mm256_loadu_ps(rc + i);
            __m256 sin_v = _mm256_loadu_ps(rs + i);
            _mm256_storeu_ps(hd + i,            _mm256_fmsub_ps(v0, cos_v, _mm256_mul_ps(v1, sin_v)));
            _mm256_storeu_ps(hd + half_rope + i, _mm256_fmadd_ps(v0, sin_v, _mm256_mul_ps(v1, cos_v)));
        }
        for (; i < half_rope; i++) {
            float v0 = hd[i], v1 = hd[half_rope + i];
            hd[i]            = v0 * rc[i] - v1 * rs[i];
            hd[half_rope + i] = v0 * rs[i] + v1 * rc[i];
        }
    }
}

void bn_transformer_batched_attn_naive_avx2_range(void *ctx, int h_start, int h_end) {
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
    int rope_dims = b->rope_dims;
    int half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_head_stride = b->q_gated ? 2 * head_size : head_size;
    float attn_scale = b->attention_scale;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

    for (int h = h_start; h < h_end; h++) {
        int kv_h = h / kv_mul;

        for (int t = 0; t < n_tokens; t++) {
            int pos = pos0 + t;
            float *rc = b->rope_cos + (size_t)t * rope_stride;
            float *rs = b->rope_sin + (size_t)t * rope_stride;

            float *q_src = b->Q_buf + (size_t)t * b->wq_rows + h * q_head_stride;
            float q_local[head_size];
            memcpy(q_local, q_src, head_size * sizeof(float));

            if (b->q_bias)
                for (int d = 0; d < head_size; d++)
                    q_local[d] += b->q_bias[h * head_size + d];
            if (b->q_norm) {
                int stride = b->qk_norm_per_head ? head_size : 0;
                BATCHED_RMSNORM(q_local, q_local, b->q_norm + h * stride,
                                head_size, b->norm_eps);
            }
            batched_apply_rope_token(q_local, 1, head_size, rope_dims, rc, rs, head_size);

            int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
            int kv_start = pos - n_kv + 1;

            float att[n_kv > BN_MAX_VLA_ELEMS ? 1 : n_kv];
            if (n_kv > BN_MAX_VLA_ELEMS) return;

            if (kv_f16) {
                const uint16_t *kc_base = (const uint16_t *)s->key_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    att[i] = batched_dot_fp16(q_local, kc_base + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;
                }
            } else {
                const float *kc_base = s->key_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    att[i] = batched_dot_fp32(q_local, kc_base + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;
                }
            }

            bn_transformer_softmax(att, n_kv);

            float xb_local[head_size];
            memset(xb_local, 0, head_size * sizeof(float));
            if (kv_f16) {
                const uint16_t *vc_base = (const uint16_t *)s->value_cache + loff;
                for (int i = 0; i < n_kv; i++)
                    batched_acc_v_fp16(xb_local, att[i], vc_base + (size_t)((kv_start + i) % seq_len) * kv_dim + kv_h * head_size, head_size);
            } else {
                const float *vc_base = s->value_cache + loff;
                for (int i = 0; i < n_kv; i++)
                    batched_acc_v_fp32(xb_local, att[i], vc_base + (size_t)((kv_start + i) % seq_len) * kv_dim + kv_h * head_size, head_size);
            }

            // Apply gate if needed
            if (b->q_gated) {
                float *gate = q_src + head_size;
                for (int d = 0; d < head_size; d++)
                    xb_local[d] *= 1.0f / (1.0f + expf(-gate[d]));
            }

            // Write to output: out[t * wo_cols + h * head_size]
            memcpy(b->out + (size_t)t * b->wo_cols + h * head_size,
                   xb_local, head_size * sizeof(float));
        }
    }
}

#define FLASH_ATTN_TILE 64

void bn_transformer_batched_attn_flash_avx2_range(void *ctx, int h_start, int h_end) {
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
    int rope_dims = b->rope_dims;
    int half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_head_stride = b->q_gated ? 2 * head_size : head_size;
    float attn_scale = b->attention_scale;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

    for (int h = h_start; h < h_end; h++) {
        int kv_h = h / kv_mul;

        for (int t = 0; t < n_tokens; t++) {
            int pos = pos0 + t;
            float *rc = b->rope_cos + (size_t)t * rope_stride;
            float *rs = b->rope_sin + (size_t)t * rope_stride;

            float *q_src = b->Q_buf + (size_t)t * b->wq_rows + h * q_head_stride;
            float q_local[head_size];
            memcpy(q_local, q_src, head_size * sizeof(float));

            if (b->q_bias)
                for (int d = 0; d < head_size; d++)
                    q_local[d] += b->q_bias[h * head_size + d];
            if (b->q_norm) {
                int stride = b->qk_norm_per_head ? head_size : 0;
                BATCHED_RMSNORM(q_local, q_local, b->q_norm + h * stride,
                                head_size, b->norm_eps);
            }
            batched_apply_rope_token(q_local, 1, head_size, rope_dims, rc, rs, head_size);

            int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
            int kv_start = pos - n_kv + 1;

            float out_buf[head_size];
            memset(out_buf, 0, head_size * sizeof(float));
            float running_max = -INFINITY;
            float running_sum = 0.0f;

            for (int ti_start = 0; ti_start < n_kv; ti_start += FLASH_ATTN_TILE) {
                int ti_end = ti_start + FLASH_ATTN_TILE;
                if (ti_end > n_kv) ti_end = n_kv;

                for (int ti = ti_start; ti < ti_end; ti++) {
                    int ki = (kv_start + ti) % seq_len;

                    if (ti + 1 < ti_end) {
                        int ki_next = (kv_start + ti + 1) % seq_len;
                        if (kv_f16)
                            _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                        else
                            _mm_prefetch((const char *)(s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                    }

                    float score;
                    if (kv_f16)
                        score = batched_dot_fp16(q_local, (const uint16_t *)s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;
                    else
                        score = batched_dot_fp32(q_local, s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;

                    float old_max = running_max;
                    if (score > old_max) {
                        float rescale = expf(old_max - score);
                        running_max = score;
                        running_sum *= rescale;
                        __m256 rs_v = _mm256_set1_ps(rescale);
                        for (int rd = 0; rd < head_size; rd += 8)
                            _mm256_storeu_ps(out_buf + rd, _mm256_mul_ps(_mm256_loadu_ps(out_buf + rd), rs_v));
                    }

                    float w = expf(score - running_max);
                    running_sum += w;
                    if (kv_f16)
                        batched_acc_v_fp16(out_buf, w, (const uint16_t *)s->value_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size);
                    else
                        batched_acc_v_fp32(out_buf, w, s->value_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size);
                }
            }

            float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
            __m256 is_v = _mm256_set1_ps(inv_sum);

            if (b->q_gated) {
                float *gate = q_src + head_size;
                for (int d = 0; d < head_size; d += 8) {
                    __m256 o = _mm256_mul_ps(_mm256_loadu_ps(out_buf + d), is_v);
                    __m256 g = _mm256_loadu_ps(gate + d);
                    _mm256_storeu_ps(b->out + (size_t)t * b->wo_cols + h * head_size + d,
                                     _mm256_mul_ps(o, bn_avx2_fast_sigmoid_ps(g)));
                }
            } else {
                float *dst = b->out + (size_t)t * b->wo_cols + h * head_size;
                for (int d = 0; d < head_size; d += 8)
                    _mm256_storeu_ps(dst + d, _mm256_mul_ps(_mm256_loadu_ps(out_buf + d), is_v));
            }
        }
    }
}

void bn_transformer_batched_attn_flash_avx2_pair_range(void *ctx, int unit_start, int unit_end) {
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
    int rope_dims = b->rope_dims;
    int half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_head_stride = b->q_gated ? 2 * head_size : head_size;
    float attn_scale = b->attention_scale;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

    for (int unit = unit_start; unit < unit_end; unit++) {
        int h = unit / n_tokens;
        int t = unit - h * n_tokens;
        int kv_h = h / kv_mul;
        int pos = pos0 + t;
        float *rc = b->rope_cos + (size_t)t * rope_stride;
        float *rs = b->rope_sin + (size_t)t * rope_stride;

        float *q_src = b->Q_buf + (size_t)t * b->wq_rows + h * q_head_stride;
        float q_local[head_size];
        memcpy(q_local, q_src, head_size * sizeof(float));

        if (b->q_bias)
            for (int d = 0; d < head_size; d++)
                q_local[d] += b->q_bias[h * head_size + d];
        if (b->q_norm) {
            int stride = b->qk_norm_per_head ? head_size : 0;
            BATCHED_RMSNORM(q_local, q_local, b->q_norm + h * stride,
                            head_size, b->norm_eps);
        }
        batched_apply_rope_token(q_local, 1, head_size, rope_dims, rc, rs, head_size);

        int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
        int kv_start = pos - n_kv + 1;

        float out_buf[head_size];
        memset(out_buf, 0, head_size * sizeof(float));
        float running_max = -INFINITY;
        float running_sum = 0.0f;

        for (int ti_start = 0; ti_start < n_kv; ti_start += FLASH_ATTN_TILE) {
            int ti_end = ti_start + FLASH_ATTN_TILE;
            if (ti_end > n_kv) ti_end = n_kv;

            for (int ti = ti_start; ti < ti_end; ti++) {
                int ki = (kv_start + ti) % seq_len;

                if (ti + 1 < ti_end) {
                    int ki_next = (kv_start + ti + 1) % seq_len;
                    if (kv_f16)
                        _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                    else
                        _mm_prefetch((const char *)(s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }

                float score;
                if (kv_f16)
                    score = batched_dot_fp16(q_local, (const uint16_t *)s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;
                else
                    score = batched_dot_fp32(q_local, s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;

                float old_max = running_max;
                if (score > old_max) {
                    float rescale = expf(old_max - score);
                    running_max = score;
                    running_sum *= rescale;
                    __m256 rs_v = _mm256_set1_ps(rescale);
                    for (int rd = 0; rd < head_size; rd += 8)
                        _mm256_storeu_ps(out_buf + rd, _mm256_mul_ps(_mm256_loadu_ps(out_buf + rd), rs_v));
                }

                float w = expf(score - running_max);
                running_sum += w;
                if (kv_f16)
                    batched_acc_v_fp16(out_buf, w, (const uint16_t *)s->value_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size);
                else
                    batched_acc_v_fp32(out_buf, w, s->value_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size);
            }
        }

        float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        __m256 is_v = _mm256_set1_ps(inv_sum);

        if (b->q_gated) {
            float *gate = q_src + head_size;
            for (int d = 0; d < head_size; d += 8) {
                __m256 o = _mm256_mul_ps(_mm256_loadu_ps(out_buf + d), is_v);
                __m256 g = _mm256_loadu_ps(gate + d);
                _mm256_storeu_ps(b->out + (size_t)t * b->wo_cols + h * head_size + d,
                                 _mm256_mul_ps(o, bn_avx2_fast_sigmoid_ps(g)));
            }
        } else {
            float *dst = b->out + (size_t)t * b->wo_cols + h * head_size;
            for (int d = 0; d < head_size; d += 8)
                _mm256_storeu_ps(dst + d, _mm256_mul_ps(_mm256_loadu_ps(out_buf + d), is_v));
        }
    }
}

#endif // __AVX2__
