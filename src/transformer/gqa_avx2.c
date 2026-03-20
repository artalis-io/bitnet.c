#include "transformer_internal.h"

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

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;
        float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);

        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float k_buf[head_size];
            const float *k_t;
            if (kv_f16) {
                const uint16_t *k_f16 = (const uint16_t *)s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                // Prefetch next KV entry
                if (i + 1 < n_kv) {
                    int t_next = (start + i + 1) % seq_len;
                    _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }
                for (int d = 0; d < head_size; d += 8)
                    _mm256_storeu_ps(k_buf + d, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k_f16 + d))));
                k_t = k_buf;
            } else {
                k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                // Prefetch next KV entry
                if (i + 1 < n_kv) {
                    int t_next = (start + i + 1) % seq_len;
                    _mm_prefetch((const char *)(s->key_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }
            }
            // 4-way unrolled dot product (32 floats per iteration)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
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
            att[i] = bn_avx2_hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))) * inv_sqrt_hs;
        }

        bn_transformer_softmax(att, n_kv);

        float *xb_h = s->xb + h * head_size;
        memset(xb_h, 0, head_size * sizeof(float));
        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float v_buf[head_size];
            const float *v_t;
            if (kv_f16) {
                const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                // Prefetch next value entry
                if (i + 1 < n_kv) {
                    int t_next = (start + i + 1) % seq_len;
                    _mm_prefetch((const char *)((const uint16_t *)s->value_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }
                for (int d = 0; d < head_size; d += 8)
                    _mm256_storeu_ps(v_buf + d, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v_f16 + d))));
                v_t = v_buf;
            } else {
                v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                if (i + 1 < n_kv) {
                    int t_next = (start + i + 1) % seq_len;
                    _mm_prefetch((const char *)(s->value_cache + loff + (size_t)t_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }
            }
            float a = att[i];
            __m256 av = _mm256_set1_ps(a);
            for (int d = 0; d < head_size; d += 8) {
                __m256 cur = _mm256_loadu_ps(xb_h + d);
                _mm256_storeu_ps(xb_h + d, _mm256_fmadd_ps(av, _mm256_loadu_ps(v_t + d), cur));
            }
        }
    }
}

#endif // __AVX2__
