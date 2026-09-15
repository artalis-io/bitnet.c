#include "transformer_batched_attn_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_math_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "simd_helpers.h"
#include "threadpool.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef __AVX2__
#include <immintrin.h>

#define BATCHED_RMSNORM bn_transformer_rmsnorm_avx2

static inline float batched_fast_expf(float x) {
    if (x < -87.3f) x = -87.3f;
    if (x > 88.7f) x = 88.7f;

    float n_f = floorf(x * 1.4426950409f + 0.5f);
    int n = (int)n_f;
    float r = x - n_f * 0.6931471806f;

    float poly = ((0.04166664f * r + 0.16666667f) * r + 0.49999994f) * r + 1.0f;
    poly = poly * r + 1.0f;

    union {
        uint32_t i;
        float f;
    } e2n;
    e2n.i = (uint32_t)(n + 127) << 23;
    return poly * e2n.f;
}

#ifndef __AVX512F__
static inline float batched_dot_hsum(__m256 x) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(x),
                            _mm256_extractf128_ps(x, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

static inline float batched_reduce_4x16(const float lanes[4][16]) {
    float combined[16];
    float half[8];
    float quarter[4];
    for (int lane = 0; lane < 16; lane++)
        combined[lane] = (lanes[0][lane] + lanes[2][lane]) +
                         (lanes[1][lane] + lanes[3][lane]);
    for (int lane = 0; lane < 8; lane++)
        half[lane] = combined[lane + 8] + combined[lane];
    for (int lane = 0; lane < 4; lane++)
        quarter[lane] = half[lane + 4] + half[lane];
    return (quarter[0] + quarter[2]) + (quarter[1] + quarter[3]);
}
#endif

static inline float batched_dot_fp32(const float *q, const float *k, int head_size) {
#ifdef __AVX512F__
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    int d = 0;
    for (; d + 63 < head_size; d += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d),
                             _mm512_loadu_ps(k + d), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 16),
                             _mm512_loadu_ps(k + d + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 32),
                             _mm512_loadu_ps(k + d + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 48),
                             _mm512_loadu_ps(k + d + 48), a3);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(a0, a2), _mm512_add_ps(a1, a3)));
#else
    float lanes[4][16] = {{0.0f}};
    for (int d = 0; d < head_size; d++)
        lanes[(d >> 4) & 3][d & 15] = fmaf(
            q[d], k[d], lanes[(d >> 4) & 3][d & 15]);
    return batched_reduce_4x16(lanes);
#endif
}

static inline float batched_dot_fp32_reference(const float *q, const float *k,
                                                int head_size) {
#ifdef __AVX512F__
    // llama.cpp's reference CPU graph is compiled for the active x86 ISA.
    // Preserve its native AVX512 accumulation and reduction tree here; the
    // architecture policy selects reference attention semantics, not a fixed
    // 256-bit implementation width.
    return batched_dot_fp32(q, k, head_size);
#else
    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    int d = 0;
    for (; d + 31 < head_size; d += 32)
        for (int lane = 0; lane < 4; lane++)
            acc[lane] = _mm256_fmadd_ps(
                _mm256_loadu_ps(q + d + lane * 8),
                _mm256_loadu_ps(k + d + lane * 8), acc[lane]);
    acc[0] = _mm256_add_ps(acc[0], acc[2]);
    acc[1] = _mm256_add_ps(acc[1], acc[3]);
    acc[0] = _mm256_add_ps(acc[0], acc[1]);
    __m128 half = _mm_add_ps(_mm256_castps256_ps128(acc[0]),
                             _mm256_extractf128_ps(acc[0], 1));
    half = _mm_hadd_ps(half, half);
    float sum = _mm_cvtss_f32(_mm_hadd_ps(half, half));
    for (; d < head_size; d++) sum += q[d] * k[d];
    return sum;
#endif
}

static inline float batched_dot_fp16(const float *q, const uint16_t *k, int head_size) {
#ifdef __AVX512F__
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    int d = 0;
    for (; d + 63 < head_size; d += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d),
            _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d))), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 16),
            _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d + 16))), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 32),
            _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d + 32))), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 48),
            _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + d + 48))), a3);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(a0, a2), _mm512_add_ps(a1, a3)));
#else
    float lanes[4][16] = {{0.0f}};
    for (int d = 0; d < head_size; d++)
        lanes[(d >> 4) & 3][d & 15] = fmaf(
            q[d], _cvtsh_ss(k[d]), lanes[(d >> 4) & 3][d & 15]);
    return batched_reduce_4x16(lanes);
#endif
}

static inline void batched_acc_v_fp32(float *out, float w,
                                      const float *v, int head_size) {
    __m256 wv = _mm256_set1_ps(w);
    for (int d = 0; d < head_size; d += 8)
        _mm256_storeu_ps(
            out + d,
            _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d),
                            _mm256_loadu_ps(out + d)));
}

static void batched_weighted_v_fp32(float *out, const float *att,
                                    const float *v_base, int n_kv,
    int kv_start, int seq_len, int kv_dim,
    int kv_head_offset, int head_size) {
    for (int d = 0; d < head_size; d += 8) {
#ifdef __AVX512F__
        // llama.cpp's AVX512 tinyBLAS keeps sixteen accumulation lanes.
        __m256 lanes[16];
        const int lane_mask = 15;
        for (int j = 0; j < 16; j++) lanes[j] = _mm256_setzero_ps();
#else
        // llama.cpp's AVX2 tinyBLAS keeps eight accumulation lanes per
        // output. Transpose those lanes over eight contiguous output
        // dimensions while preserving the per-lane FMA order.
        __m256 lanes[8];
        const int lane_mask = 7;
        for (int j = 0; j < 8; j++) lanes[j] = _mm256_setzero_ps();
#endif
        for (int i = 0; i < n_kv; i++) {
            int token = (kv_start + i) % seq_len;
            int lane = i & lane_mask;
            const float *values = v_base + (size_t)token * kv_dim +
                                  kv_head_offset + d;
            lanes[lane] = _mm256_fmadd_ps(
                _mm256_set1_ps(att[i]), _mm256_loadu_ps(values), lanes[lane]);
        }
#ifdef __AVX512F__
        for (int j = 0; j < 8; j++)
            lanes[j] = _mm256_add_ps(lanes[j + 8], lanes[j]);
#endif
        // Match llama.cpp tinyBLAS hsum for the active SIMD width.
        for (int j = 0; j < 4; j++)
            lanes[j] = _mm256_add_ps(lanes[j + 4], lanes[j]);
        _mm256_storeu_ps(out + d, _mm256_add_ps(
            _mm256_add_ps(lanes[0], lanes[2]),
            _mm256_add_ps(lanes[1], lanes[3])));
    }
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

static void batched_attn_naive_avx2_one(BnBatchedAttnCtx *b, int h,
                                         int t) {
    BnRunState *s = b->s;
    int head_size = b->head_size;
    int kv_dim = b->kv_dim;
    int kv_mul = b->kv_mul;
    int seq_len = b->seq_len;
    int pos0 = b->pos0;
    size_t loff = b->loff;
    int kv_cache_uses_fp16_rows = b->kv_cache_uses_fp16_rows;
    int rope_dims = b->rope_dims;
    int half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_row_stride = b->q_row_stride > 0 ? b->q_row_stride : b->wq_rows;
    int q_head_stride = b->q_gated ? 2 * head_size : head_size;
    float attn_scale = b->attention_scale;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

    int kv_h = h / kv_mul;
            int pos = pos0 + t;
            float *rc = b->rope_cos + (size_t)t * rope_stride;
            float *rs = b->rope_sin + (size_t)t * rope_stride;

            float *q_src = b->Q_buf + (size_t)t * q_row_stride + h * q_head_stride;
            float q_local[head_size];
            memcpy(q_local, q_src, head_size * sizeof(float));
            if (h == 0)
                bn_transformer_cpu_debug_dump_values(
                    b->runtime, q_local, head_size,
                    "bitnet_attn_q_matmul_h0", b->layer, pos);

            if (b->q_bias)
                for (int d = 0; d < head_size; d++)
                    q_local[d] += b->q_bias[h * head_size + d];
            if (b->q_norm) {
                int stride = b->qk_norm_per_head ? head_size : 0;
                BATCHED_RMSNORM(q_local, q_local, b->q_norm + h * stride,
                                head_size, b->norm_eps);
            }
            if (h == 0)
                bn_transformer_cpu_debug_dump_values(
                    b->runtime, q_local, head_size,
                    "bitnet_attn_q_normed_h0", b->layer, pos);
            batched_apply_rope_token(q_local, 1, head_size, rope_dims, rc, rs, head_size);
            if (h == 0)
                bn_transformer_cpu_debug_dump_values(
                    b->runtime, q_local, head_size, "bitnet_attn_q_h0",
                    b->layer, pos);

            int n_kv = (pos + 1 < seq_len) ? pos + 1 : seq_len;
            if (b->attention_window > 0 && n_kv > b->attention_window)
                n_kv = b->attention_window;
            int kv_start = pos - n_kv + 1;

            int att_size = (n_kv + 255) & ~255;
            float att[att_size > BN_MAX_VLA_ELEMS ? 1 : att_size];
            if (att_size > BN_MAX_VLA_ELEMS) return;

            if (kv_cache_uses_fp16_rows) {
                const uint16_t *kc_base = (const uint16_t *)s->key_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    att[i] = batched_dot_fp16(
                        q_local,
                        kc_base + (size_t)ki * kv_dim + kv_h * head_size,
                        head_size);
                }
            } else {
                const float *kc_base = s->key_cache + loff;
                for (int i = 0; i < n_kv; i++) {
                    int ki = (kv_start + i) % seq_len;
                    const float *k = kc_base + (size_t)ki * kv_dim +
                                     kv_h * head_size;
                    att[i] = b->uses_reference_dot_accumulation
                        ? batched_dot_fp32_reference(q_local, k, head_size)
                        : batched_dot_fp32(q_local, k, head_size);
                }
            }
            if (h == 0)
                bn_transformer_cpu_debug_dump_values(
                    b->runtime, att, n_kv, "bitnet_attn_scores_raw_h0",
                    b->layer, pos);
            for (int i = 0; i < n_kv; i++)
                att[i] *= attn_scale;
            if (h == 0)
                bn_transformer_cpu_debug_dump_values(
                    b->runtime, att, n_kv, "bitnet_attn_scores_h0",
                    b->layer, pos);

            for (int i = n_kv; i < att_size; i++)
                att[i] = -INFINITY;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            bn_transformer_softmax(att, att_size);
#else
            bn_transformer_softmax_avx2_batch(att, att_size);
#endif
            if (h == 0)
                bn_transformer_cpu_debug_dump_values(
                    b->runtime, att, att_size, "bitnet_attn_softmax_h0",
                    b->layer, pos);

            float xb_local[head_size];
            memset(xb_local, 0, head_size * sizeof(float));
            if (kv_cache_uses_fp16_rows) {
                const uint16_t *vc_base = (const uint16_t *)s->value_cache + loff;
                for (int i = 0; i < n_kv; i++)
                    batched_acc_v_fp16(xb_local, att[i], vc_base + (size_t)((kv_start + i) % seq_len) * kv_dim + kv_h * head_size, head_size);
            } else {
                const float *vc_base = s->value_cache + loff;
                batched_weighted_v_fp32(
                    xb_local, att, vc_base, n_kv, kv_start, seq_len, kv_dim,
                    kv_h * head_size, head_size);
            }

            // Apply gate if needed
            if (b->q_gated) {
                float *gate = q_src + head_size;
                for (int d = 0; d < head_size; d++)
                    xb_local[d] *= 1.0f / (1.0f + expf(-gate[d]));
            }
            if (h == 0)
                bn_transformer_cpu_debug_dump_values(
                    b->runtime, xb_local, head_size, "bitnet_attn_out_h0",
                    b->layer, pos);

            // Write to output: out[t * wo_cols + h * head_size]
            memcpy(b->out + (size_t)t * b->wo_cols + h * head_size,
                   xb_local, head_size * sizeof(float));
}

void bn_transformer_batched_attn_naive_avx2_range(void *ctx, int h_start,
                                                   int h_end) {
    BnBatchedAttnCtx *b = (BnBatchedAttnCtx *)ctx;
    for (int h = h_start; h < h_end; h++)
        for (int t = 0; t < b->n_tokens; t++)
            batched_attn_naive_avx2_one(b, h, t);
}

void bn_transformer_batched_attn_naive_avx2_pair_range(
    void *ctx, int unit_start, int unit_end) {
    BnBatchedAttnCtx *b = (BnBatchedAttnCtx *)ctx;
    for (int unit = unit_start; unit < unit_end; unit++) {
        int h = unit / b->n_tokens;
        int t = unit - h * b->n_tokens;
        batched_attn_naive_avx2_one(b, h, t);
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
    int kv_cache_uses_fp16_rows = b->kv_cache_uses_fp16_rows;
    int rope_dims = b->rope_dims;
    int half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_row_stride = b->q_row_stride > 0 ? b->q_row_stride : b->wq_rows;
    int q_head_stride = b->q_gated ? 2 * head_size : head_size;
    float attn_scale = b->attention_scale;
    if (head_size > BN_MAX_VLA_ELEMS || head_size % 8 != 0) return;

    for (int h = h_start; h < h_end; h++) {
        int kv_h = h / kv_mul;

        for (int t = 0; t < n_tokens; t++) {
            int pos = pos0 + t;
            float *rc = b->rope_cos + (size_t)t * rope_stride;
            float *rs = b->rope_sin + (size_t)t * rope_stride;

            float *q_src = b->Q_buf + (size_t)t * q_row_stride + h * q_head_stride;
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
            if (b->attention_window > 0 && n_kv > b->attention_window)
                n_kv = b->attention_window;
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
                        if (kv_cache_uses_fp16_rows)
                            _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                        else
                            _mm_prefetch((const char *)(s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                    }

                    float score;
                    if (kv_cache_uses_fp16_rows)
                        score = batched_dot_fp16(q_local, (const uint16_t *)s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;
                    else
                        score = batched_dot_fp32(q_local, s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;

                    float old_max = running_max;
                    if (score > old_max) {
                        float rescale = batched_fast_expf(old_max - score);
                        running_max = score;
                        running_sum *= rescale;
                        __m256 rs_v = _mm256_set1_ps(rescale);
                        for (int rd = 0; rd < head_size; rd += 8)
                            _mm256_storeu_ps(out_buf + rd, _mm256_mul_ps(_mm256_loadu_ps(out_buf + rd), rs_v));
                    }

                    float w = batched_fast_expf(score - running_max);
                    running_sum += w;
                    if (kv_cache_uses_fp16_rows)
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
    int kv_cache_uses_fp16_rows = b->kv_cache_uses_fp16_rows;
    int rope_dims = b->rope_dims;
    int half_rope = rope_dims / 2;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : half_rope;
    int q_row_stride = b->q_row_stride > 0 ? b->q_row_stride : b->wq_rows;
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

        float *q_src = b->Q_buf + (size_t)t * q_row_stride + h * q_head_stride;
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
        if (b->attention_window > 0 && n_kv > b->attention_window)
            n_kv = b->attention_window;
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
                    if (kv_cache_uses_fp16_rows)
                        _mm_prefetch((const char *)((const uint16_t *)s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                    else
                        _mm_prefetch((const char *)(s->key_cache + loff + (size_t)ki_next * kv_dim + kv_h * head_size), _MM_HINT_T0);
                }

                float score;
                if (kv_cache_uses_fp16_rows)
                    score = batched_dot_fp16(q_local, (const uint16_t *)s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;
                else
                    score = batched_dot_fp32(q_local, s->key_cache + loff + (size_t)ki * kv_dim + kv_h * head_size, head_size) * attn_scale;

                float old_max = running_max;
                if (score > old_max) {
                    float rescale = batched_fast_expf(old_max - score);
                    running_max = score;
                    running_sum *= rescale;
                    __m256 rs_v = _mm256_set1_ps(rescale);
                    for (int rd = 0; rd < head_size; rd += 8)
                        _mm256_storeu_ps(out_buf + rd, _mm256_mul_ps(_mm256_loadu_ps(out_buf + rd), rs_v));
                }

                float w = batched_fast_expf(score - running_max);
                running_sum += w;
                if (kv_cache_uses_fp16_rows)
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
