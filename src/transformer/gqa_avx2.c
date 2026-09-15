#include "transformer_gqa_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_simd_internal.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __AVX2__

static inline float gqa_avx2_hsum(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                           _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

static float gqa_avx2_dot(const float *q, const void *k,
                          int head_size, int fp16_rows) {
    __m256 sum[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    int d = 0;
    for (; d + 31 < head_size; d += 32) {
        for (int r = 0; r < 4; r++) {
            __m256 kv = fp16_rows
                ? _mm256_cvtph_ps(_mm_loadu_si128(
                      (const __m128i *)((const uint16_t *)k + d + r * 8)))
                : _mm256_loadu_ps((const float *)k + d + r * 8);
            sum[r] = _mm256_fmadd_ps(
                _mm256_loadu_ps(q + d + r * 8), kv, sum[r]);
        }
    }
    sum[0] = _mm256_add_ps(sum[0], sum[2]);
    sum[1] = _mm256_add_ps(sum[1], sum[3]);
    sum[0] = _mm256_add_ps(sum[0], sum[1]);
    float result = gqa_avx2_hsum(sum[0]);
    for (; d < head_size; d++) {
        float kv = fp16_rows
            ? _cvtsh_ss(((const uint16_t *)k)[d])
            : ((const float *)k)[d];
        result += q[d] * kv;
    }
    return result;
}

static float gqa_avx2_weighted_v(const float *att, int n_kv,
                                 const BnRunState *s, size_t loff,
                                 int start, int seq_len, int kv_dim,
                                 int kv_h, int head_size, int d,
                                 int fp16_rows,
                                 int batched_prompt_contract,
                                 int use_padded_reduction) {
    __m256 sum[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    int i = 0;
    for (; i + 31 < n_kv; i += 32) {
        float values[32];
        for (int j = 0; j < 32; j++) {
            int t = (start + i + j) % seq_len;
            if (fp16_rows) {
                const uint16_t *row = (const uint16_t *)s->value_cache +
                    loff + (size_t)t * kv_dim + kv_h * head_size;
                values[j] = _cvtsh_ss(row[d]);
            } else {
                const float *row = s->value_cache + loff +
                    (size_t)t * kv_dim + kv_h * head_size;
                values[j] = row[d];
            }
        }
        for (int r = 0; r < 4; r++)
            sum[r] = _mm256_fmadd_ps(
                _mm256_loadu_ps(att + i + r * 8),
                _mm256_loadu_ps(values + r * 8), sum[r]);
    }
    if (i < n_kv && !batched_prompt_contract && !use_padded_reduction) {
        sum[0] = _mm256_add_ps(sum[0], sum[2]);
        sum[1] = _mm256_add_ps(sum[1], sum[3]);
        float result = gqa_avx2_hsum(_mm256_add_ps(sum[0], sum[1]));
        for (; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float value;
            if (fp16_rows) {
                const uint16_t *row = (const uint16_t *)s->value_cache +
                    loff + (size_t)t * kv_dim + kv_h * head_size;
                value = _cvtsh_ss(row[d]);
            } else {
                const float *row = s->value_cache + loff +
                    (size_t)t * kv_dim + kv_h * head_size;
                value = row[d];
            }
            result = fmaf(att[i], value, result);
        }
        return result;
    }
    if (i < n_kv) {
        float tail_att[32] = {0.0f};
        float tail_values[32] = {0.0f};
        int count = n_kv - i;
        for (int j = 0; j < count; j++) {
            int t = (start + i + j) % seq_len;
            tail_att[j] = att[i + j];
            if (fp16_rows) {
                const uint16_t *row = (const uint16_t *)s->value_cache +
                    loff + (size_t)t * kv_dim + kv_h * head_size;
                tail_values[j] = _cvtsh_ss(row[d]);
            } else {
                const float *row = s->value_cache + loff +
                    (size_t)t * kv_dim + kv_h * head_size;
                tail_values[j] = row[d];
            }
        }
        for (int r = 0; r < 4; r++)
            sum[r] = _mm256_fmadd_ps(
                _mm256_loadu_ps(tail_att + r * 8),
                _mm256_loadu_ps(tail_values + r * 8), sum[r]);
    }
    sum[0] = _mm256_add_ps(sum[0], sum[2]);
    sum[1] = _mm256_add_ps(sum[1], sum[3]);
    return gqa_avx2_hsum(_mm256_add_ps(sum[0], sum[1]));
}

void bn_transformer_gqa_avx2_range(void *ctx, int h_start, int h_end) {
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
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;
        float attn_scale = g->attention_scale;

        for (int i = 0; !g->scores_ready && i < n_kv; i++) {
            int t = (start + i) % seq_len;
            // Prefetch next KV entry
            if (i + 1 < n_kv) {
                int t_next = (start + i + 1) % seq_len;
                if (kv_cache_uses_fp16_rows)
                    _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                else
                    _mm_prefetch((const char *)(s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
            }
            const void *k_row = kv_cache_uses_fp16_rows
                ? (const void *)((const uint16_t *)s->key_cache + loff +
                    (size_t)t * kv_dim + kv_h * head_size)
                : (const void *)(s->key_cache + loff +
                    (size_t)t * kv_dim + kv_h * head_size);
            float dot = gqa_avx2_dot(
                q_h, k_row, head_size, kv_cache_uses_fp16_rows);
            att[i] = dot;
        }

        if (g->runtime &&
            (bn_transformer_cpu_debug_dump_path(g->runtime) ||
             bn_transformer_cpu_debug_binary_path(g->runtime))) {
            char score_tag[48];
            snprintf(score_tag, sizeof(score_tag),
                     "bitnet_attn_scores_h%d", h);
            bn_transformer_cpu_debug_dump_values(
                g->runtime, att, n_kv, score_tag, g->layer, g->pos);
        }

        if (g->scores_ready != 2)
            for (int i = 0; i < n_kv; i++)
                att[i] *= attn_scale;

        if (g->scores_ready != 2) {
            if (s->batched_prompt_contract ||
                g->use_padded_weighted_v_reduction)
                bn_transformer_softmax_avx2_batch(att, n_kv);
            else
                bn_transformer_softmax(att, n_kv);
        }

        if (h == 0 && bn_transformer_cpu_debug_binary_selected(
                g->runtime, "bitnet_attn_values_h0", g->layer)) {
            float *values = malloc((size_t)n_kv * (size_t)head_size *
                                   sizeof(*values));
            if (values) {
                for (int i = 0; i < n_kv; i++) {
                    int t = (start + i) % seq_len;
                    float *dst = values + (size_t)i * head_size;
                    if (kv_cache_uses_fp16_rows) {
                        const uint16_t *src = (const uint16_t *)s->value_cache +
                            loff + (size_t)t * kv_dim + kv_h * head_size;
                        for (int d = 0; d < head_size; d += 8)
                            _mm256_storeu_ps(dst + d, _mm256_cvtph_ps(
                                _mm_loadu_si128((const __m128i *)(src + d))));
                    } else {
                        const float *src = s->value_cache + loff +
                            (size_t)t * kv_dim + kv_h * head_size;
                        memcpy(dst, src, (size_t)head_size * sizeof(*dst));
                    }
                }
                bn_transformer_cpu_debug_dump_values(
                    g->runtime, values, n_kv * head_size,
                    "bitnet_attn_values_h0", g->layer, g->pos);
                free(values);
            }
        }

        float *xb_h = s->xb + h * head_size;
        for (int d = 0; d < head_size; d++)
            xb_h[d] = gqa_avx2_weighted_v(
                att, n_kv, s, loff, start, seq_len, kv_dim, kv_h,
                head_size, d, kv_cache_uses_fp16_rows,
                s->batched_prompt_contract,
                g->use_padded_weighted_v_reduction);
    }
}

// --- Flash GQA attention (online softmax, per-head, single-pass) ---

#define FLASH_ATTN_TILE 64

void bn_transformer_flash_gqa_avx2_range(void *ctx, int h_start, int h_end) {
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
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

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

                // Prefetch next K entry
                if (ti + 1 < ti_end) {
                    int t_next = (start + ti + 1) % seq_len;
                    if (kv_cache_uses_fp16_rows)
                        _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                    else
                        _mm_prefetch((const char *)(s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }

                // Score: dot(Q, K) * scale — inline F16 conversion, no temp buffer
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                if (kv_cache_uses_fp16_rows) {
                    const uint16_t *k_f16 = (const uint16_t *)s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                    int d = 0;
                    for (; d + 31 < head_size; d += 32) {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d),      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_f16 + d))), a0);
                        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 8),  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_f16 + d + 8))), a1);
                        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 16), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_f16 + d + 16))), a2);
                        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 24), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_f16 + d + 24))), a3);
                    }
                    for (; d + 15 < head_size; d += 16) {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d),     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_f16 + d))), a0);
                        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 8), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_f16 + d + 8))), a1);
                    }
                } else {
                    const float *k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                    int d = 0;
                    for (; d + 31 < head_size; d += 32) {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d),      _mm256_loadu_ps(k_t + d), a0);
                        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 8),  _mm256_loadu_ps(k_t + d + 8), a1);
                        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 16), _mm256_loadu_ps(k_t + d + 16), a2);
                        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 24), _mm256_loadu_ps(k_t + d + 24), a3);
                    }
                    for (; d + 15 < head_size; d += 16) {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d),     _mm256_loadu_ps(k_t + d), a0);
                        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d + 8), _mm256_loadu_ps(k_t + d + 8), a1);
                    }
                }
                float score = gqa_avx2_hsum(_mm256_add_ps(
                    _mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3))) *
                    attn_scale;

                // Online softmax update
                float old_max = running_max;
                if (score > old_max) {
                    float rescale = expf(old_max - score);
                    running_max = score;
                    running_sum *= rescale;
                    __m256 rs = _mm256_set1_ps(rescale);
                    for (int rd = 0; rd < head_size; rd += 8)
                        _mm256_storeu_ps(out_buf + rd, _mm256_mul_ps(_mm256_loadu_ps(out_buf + rd), rs));
                }

                float w = expf(score - running_max);
                running_sum += w;
                __m256 wv = _mm256_set1_ps(w);
                // V accumulation with inline F16 conversion (no temp buffer)
                if (kv_cache_uses_fp16_rows) {
                    const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                    for (int vd = 0; vd < head_size; vd += 8)
                        _mm256_storeu_ps(out_buf + vd, _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_f16 + vd))), _mm256_loadu_ps(out_buf + vd)));
                } else {
                    const float *v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                    int vd = 0;
                    for (; vd + 31 < head_size; vd += 32) {
                        _mm256_storeu_ps(out_buf + vd,      _mm256_fmadd_ps(wv, _mm256_loadu_ps(v_t + vd),      _mm256_loadu_ps(out_buf + vd)));
                        _mm256_storeu_ps(out_buf + vd + 8,  _mm256_fmadd_ps(wv, _mm256_loadu_ps(v_t + vd + 8),  _mm256_loadu_ps(out_buf + vd + 8)));
                        _mm256_storeu_ps(out_buf + vd + 16, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v_t + vd + 16), _mm256_loadu_ps(out_buf + vd + 16)));
                        _mm256_storeu_ps(out_buf + vd + 24, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v_t + vd + 24), _mm256_loadu_ps(out_buf + vd + 24)));
                    }
                    for (; vd < head_size; vd += 8)
                        _mm256_storeu_ps(out_buf + vd, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v_t + vd), _mm256_loadu_ps(out_buf + vd)));
                }
            }
        }

        // Finalize: output = out_buf / running_sum
        float *xb_h = s->xb + h * head_size;
        float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        __m256 is = _mm256_set1_ps(inv_sum);
        for (int d = 0; d < head_size; d += 8)
            _mm256_storeu_ps(xb_h + d, _mm256_mul_ps(_mm256_loadu_ps(out_buf + d), is));
    }
}

#endif // __AVX2__

#ifdef __AVX512F__

static float gqa_avx512_weighted_v(const float *att, int n_kv,
                                   const BnRunState *s, size_t loff,
                                   int start, int seq_len, int kv_dim,
                                   int kv_h, int head_size, int d,
                                   int fp16_rows) {
    float lanes[4][16] = {{0.0f}};
    for (int i = 0; i < n_kv; i++) {
        int t = (start + i) % seq_len;
        float value;
        if (fp16_rows) {
            const uint16_t *row = (const uint16_t *)s->value_cache + loff +
                (size_t)t * kv_dim + kv_h * head_size;
            value = _cvtsh_ss(row[d]);
        } else {
            const float *row = s->value_cache + loff +
                (size_t)t * kv_dim + kv_h * head_size;
            value = row[d];
        }
        int reg = (i >> 4) & 3;
        int lane = i & 15;
        lanes[reg][lane] = fmaf(att[i], value, lanes[reg][lane]);
    }
    __m512 a0 = _mm512_loadu_ps(lanes[0]);
    __m512 a1 = _mm512_loadu_ps(lanes[1]);
    __m512 a2 = _mm512_loadu_ps(lanes[2]);
    __m512 a3 = _mm512_loadu_ps(lanes[3]);
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(a0, a2), _mm512_add_ps(a1, a3)));
    return sum;
}

void bn_transformer_gqa_avx512_range(void *ctx, int h_start, int h_end) {
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
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 16 != 0)
        return;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;

        for (int i = 0; !g->scores_ready && i < n_kv; i++) {
            int t = (start + i) % seq_len;
            __m512 a0 = _mm512_setzero_ps();
            __m512 a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps();
            __m512 a3 = _mm512_setzero_ps();
            int d = 0;
            if (kv_cache_uses_fp16_rows) {
                const uint16_t *k = (const uint16_t *)s->key_cache + loff +
                    (size_t)t * kv_dim + kv_h * head_size;
                for (; d + 63 < head_size; d += 64) {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d),
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d))), a0);
                    a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d + 16),
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d + 16))), a1);
                    a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d + 32),
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d + 32))), a2);
                    a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d + 48),
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d + 48))), a3);
                }
                for (; d + 15 < head_size; d += 16)
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d),
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d))), a0);
            } else {
                const float *k = s->key_cache + loff + (size_t)t * kv_dim +
                    kv_h * head_size;
                for (; d + 63 < head_size; d += 64) {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d), _mm512_loadu_ps(k + d), a0);
                    a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d + 16), _mm512_loadu_ps(k + d + 16), a1);
                    a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d + 32), _mm512_loadu_ps(k + d + 32), a2);
                    a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d + 48), _mm512_loadu_ps(k + d + 48), a3);
                }
                for (; d + 15 < head_size; d += 16)
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q_h + d),
                                         _mm512_loadu_ps(k + d), a0);
            }
            att[i] = bn_avx512_hsum_ps(_mm512_add_ps(
                _mm512_add_ps(a0, a2), _mm512_add_ps(a1, a3)));
        }

        if (g->runtime &&
            (bn_transformer_cpu_debug_dump_path(g->runtime) ||
             bn_transformer_cpu_debug_binary_path(g->runtime))) {
            char score_tag[48];
            snprintf(score_tag, sizeof(score_tag),
                     "bitnet_attn_scores_h%d", h);
            bn_transformer_cpu_debug_dump_values(
                g->runtime, att, n_kv, score_tag, g->layer, g->pos);
        }

        if (g->scores_ready != 2)
            for (int i = 0; i < n_kv; i++)
                att[i] *= g->attention_scale;

        if (g->scores_ready != 2)
            bn_transformer_softmax(att, n_kv);

        if (h == 0)
            bn_transformer_cpu_debug_dump_values(
                g->runtime, att, n_kv, "bitnet_attn_softmax_h0",
                g->layer, g->pos);

        if (h == 0 && bn_transformer_cpu_debug_binary_selected(
                g->runtime, "bitnet_attn_values_h0", g->layer)) {
            float *values = malloc((size_t)n_kv * (size_t)head_size *
                                   sizeof(*values));
            if (values) {
                for (int i = 0; i < n_kv; i++) {
                    int t = (start + i) % seq_len;
                    float *dst = values + (size_t)i * head_size;
                    if (kv_cache_uses_fp16_rows) {
                        const uint16_t *src =
                            (const uint16_t *)s->value_cache + loff +
                            (size_t)t * kv_dim + kv_h * head_size;
                        for (int d = 0; d < head_size; d += 16)
                            _mm512_storeu_ps(dst + d, _mm512_cvtph_ps(
                                _mm256_loadu_si256(
                                    (const __m256i *)(src + d))));
                    } else {
                        const float *src = s->value_cache + loff +
                            (size_t)t * kv_dim + kv_h * head_size;
                        memcpy(dst, src, (size_t)head_size * sizeof(*dst));
                    }
                }
                bn_transformer_cpu_debug_dump_values(
                    g->runtime, values, n_kv * head_size,
                    "bitnet_attn_values_h0", g->layer, g->pos);
                free(values);
            }
        }

        float *out = s->xb + h * head_size;
        for (int d = 0; d < head_size; d++)
            out[d] = gqa_avx512_weighted_v(
                att, n_kv, s, loff, start, seq_len, kv_dim, kv_h,
                head_size, d, kv_cache_uses_fp16_rows);
    }
}

#endif // __AVX512F__
