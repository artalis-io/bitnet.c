#include "transformer_gqa_internal.h"

#ifdef __AVX2__

void bn_transformer_gqa_avx2_range(void *ctx, int h_start, int h_end) {
    BnGQACtx *g = (BnGQACtx *)ctx;
    const BnConfig *c = g->c;
    BnRunState *s = g->s;
    int head_size = g->head_size;
    int kv_dim = g->kv_dim;
    int kv_mul = g->kv_mul;
    int n_kv = g->n_kv;
    int seq_len = g->seq_len;
    int start = g->pos - n_kv + 1;
    size_t loff = g->loff;
    int kv_f16 = c->kv_f16;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;
        float attn_scale = g->attention_scale;

        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            // Prefetch next KV entry
            if (i + 1 < n_kv) {
                int t_next = (start + i + 1) % seq_len;
                if (kv_f16)
                    _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                else
                    _mm_prefetch((const char *)(s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
            }
            // Dot product with inline F16→F32 conversion (no intermediate buffer)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            if (kv_f16) {
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
            att[i] = bn_avx2_hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))) * attn_scale;
        }

        bn_transformer_softmax(att, n_kv);

        float *xb_h = s->xb + h * head_size;
        memset(xb_h, 0, head_size * sizeof(float));
        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            // Prefetch next value entry
            if (i + 1 < n_kv) {
                int t_next = (start + i + 1) % seq_len;
                if (kv_f16)
                    _mm_prefetch((const char *)((const uint16_t *)s->value_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                else
                    _mm_prefetch((const char *)(s->value_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
            }
            float a = att[i];
            __m256 av = _mm256_set1_ps(a);
            if (kv_f16) {
                const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d += 8)
                    _mm256_storeu_ps(xb_h + d, _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_f16 + d))), _mm256_loadu_ps(xb_h + d)));
            } else {
                const float *v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d += 8)
                    _mm256_storeu_ps(xb_h + d, _mm256_fmadd_ps(av, _mm256_loadu_ps(v_t + d), _mm256_loadu_ps(xb_h + d)));
            }
        }
    }
}

// --- Flash GQA attention (online softmax, per-head, single-pass) ---

#define FLASH_ATTN_TILE 64

void bn_transformer_flash_gqa_avx2_range(void *ctx, int h_start, int h_end) {
    BnGQACtx *g = (BnGQACtx *)ctx;
    const BnConfig *c = g->c;
    BnRunState *s = g->s;
    int head_size = g->head_size;
    int kv_dim = g->kv_dim;
    int kv_mul = g->kv_mul;
    int n_kv = g->n_kv;
    int seq_len = g->seq_len;
    int start = g->pos - n_kv + 1;
    size_t loff = g->loff;
    int kv_f16 = c->kv_f16;
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
                    if (kv_f16)
                        _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                    else
                        _mm_prefetch((const char *)(s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }

                // Score: dot(Q, K) * scale — inline F16 conversion, no temp buffer
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                if (kv_f16) {
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
                float score = bn_avx2_hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))) * attn_scale;

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
                if (kv_f16) {
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
