#include "transformer_gqa_internal.h"

static float gqa_scalar_dot_ggml_order(const float *x,
                                       const float *y,
                                       int n) {
    float sums[4][4] = {{0.0f}};
    int i = 0;
    int np = n & ~15;
    for (; i < np; i += 16) {
        for (int group = 0; group < 4; group++) {
            for (int lane = 0; lane < 4; lane++) {
                int d = i + group * 4 + lane;
                sums[group][lane] = fmaf(x[d], y[d], sums[group][lane]);
            }
        }
    }
    float lanes[4];
    for (int lane = 0; lane < 4; lane++)
        lanes[lane] = (sums[0][lane] + sums[2][lane]) +
                      (sums[1][lane] + sums[3][lane]);
    float out = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; i < n; i++)
        out = fmaf(x[i], y[i], out);
    return out;
}

static void gqa_scalar_combine_small_kv_ggml_order(
    float *out,
    const float *att,
    const float *value_cache,
    size_t loff,
    int start,
    int n_kv,
    int seq_len,
    int kv_dim,
    int kv_head_offset,
    int head_size) {
    for (int d = 0; d < head_size; d++) {
        float sums[4][4] = {{0.0f}};
        for (int group = 0; group < 4; group++) {
            for (int lane = 0; lane < 4; lane++) {
                int i = group * 4 + lane;
                if (i >= n_kv)
                    break;
                int t = (start + i) % seq_len;
                sums[group][lane] = fmaf(
                    att[i],
                    value_cache[loff + (size_t)t * kv_dim +
                                kv_head_offset + d],
                    0.0f);
            }
        }
        float lanes[4];
        for (int lane = 0; lane < 4; lane++)
            lanes[lane] = (sums[0][lane] + sums[2][lane]) +
                          (sums[1][lane] + sums[3][lane]);
        out[d] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
}

static void gqa_scalar_combine_mmvf_128_order(
    float *out, const float *att, const float *value_cache, size_t loff,
    int start, int n_kv, int seq_len, int kv_dim, int kv_head_offset,
    int head_size) {
    for (int d = 0; d < head_size; d++) {
        float partial[128] = {0.0f};
        for (int tid = 0; tid < 128; tid++) {
            int t0 = 2 * tid;
            if (t0 < n_kv) {
                int row = (start + t0) % seq_len;
                partial[tid] = fmaf(
                    value_cache[loff + (size_t)row * kv_dim +
                                kv_head_offset + d],
                    att[t0], partial[tid]);
            }
            if (t0 + 1 < n_kv) {
                int row = (start + t0 + 1) % seq_len;
                partial[tid] = fmaf(
                    value_cache[loff + (size_t)row * kv_dim +
                                kv_head_offset + d],
                    att[t0 + 1], partial[tid]);
            }
        }
        for (int warp = 0; warp < 4; warp++) {
            float *w = partial + 32 * warp;
            for (int offset = 16; offset > 0; offset >>= 1) {
                float old[32];
                memcpy(old, w, sizeof(old));
                for (int lane = 0; lane < 32; lane++)
                    w[lane] = old[lane] + old[lane ^ offset];
            }
        }
        float warp_sum[32] = {
            partial[0], partial[32], partial[64], partial[96]
        };
        for (int offset = 16; offset > 0; offset >>= 1) {
            float old[32];
            memcpy(old, warp_sum, sizeof(old));
            for (int lane = 0; lane < 32; lane++)
                warp_sum[lane] = old[lane] + old[lane ^ offset];
        }
        out[d] = warp_sum[0];
    }
}

void bn_transformer_gqa_scalar_range(void *ctx, int h_start, int h_end) {
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
    if (head_size > BN_MAX_VLA_ELEMS) return;

    for (int h = h_start; h < h_end; h++) {
        float *q_h = s->q + h * head_size;
        float *att = s->att + h * seq_len;
        int kv_h = h / kv_mul;
        float attn_scale = g->attention_scale;

        for (int i = 0; !g->scores_ready && i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float k_buf[head_size];
            const float *k_t;
            if (kv_cache_uses_fp16_rows) {
                const uint16_t *k_f16 = (const uint16_t *)s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d++) k_buf[d] = bn_fp16_to_fp32(k_f16[d]);
                k_t = k_buf;
            } else {
                k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
            }
            att[i] = gqa_scalar_dot_ggml_order(q_h, k_t, head_size) *
                     attn_scale;
        }

        if (g->scores_ready != 2)
            bn_transformer_softmax(att, n_kv);

        float *xb_h = s->xb + h * head_size;
        if (g->scores_ready == 2 && !kv_cache_uses_fp16_rows) {
            gqa_scalar_combine_mmvf_128_order(
                xb_h, att, s->value_cache, loff, start, n_kv, seq_len,
                kv_dim, kv_h * head_size, head_size);
            continue;
        }
        if (!kv_cache_uses_fp16_rows && n_kv <= 16) {
            gqa_scalar_combine_small_kv_ggml_order(
                xb_h, att, s->value_cache, loff, start, n_kv, seq_len,
                kv_dim, kv_h * head_size, head_size);
            continue;
        }
        memset(xb_h, 0, head_size * sizeof(float));
        for (int i = 0; i < n_kv; i++) {
            int t = (start + i) % seq_len;
            float v_buf[head_size];
            const float *v_t;
            if (kv_cache_uses_fp16_rows) {
                const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                for (int d = 0; d < head_size; d++) v_buf[d] = bn_fp16_to_fp32(v_f16[d]);
                v_t = v_buf;
            } else {
                v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
            }
            float a = att[i];
            for (int d = 0; d < head_size; d++)
                xb_h[d] = fmaf(a, v_t[d], xb_h[d]);
        }
    }
}

// --- Flash GQA attention (online softmax, per-head, single-pass) ---

#define FLASH_ATTN_TILE 64

void bn_transformer_flash_gqa_scalar_range(void *ctx, int h_start, int h_end) {
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
    if (head_size > BN_MAX_VLA_ELEMS) return;

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
                    for (int d = 0; d < head_size; d++) k_buf[d] = bn_fp16_to_fp32(k_f16[d]);
                    k_t = k_buf;
                } else {
                    k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                }

                // Score: dot(Q, K) * scale
                float score = 0.0f;
                for (int d = 0; d < head_size; d++) score += q_h[d] * k_t[d];
                score *= attn_scale;

                // Online softmax update
                float v_buf[head_size];
                const float *v_t;
                if (kv_cache_uses_fp16_rows) {
                    const uint16_t *v_f16 = (const uint16_t *)s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                    for (int d = 0; d < head_size; d++) v_buf[d] = bn_fp16_to_fp32(v_f16[d]);
                    v_t = v_buf;
                } else {
                    v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                }

                float old_max = running_max;
                float max_scale = 1.0f;
                float weight = 1.0f;
                if (score > old_max) {
                    running_max = score;
                    max_scale = expf(old_max - running_max);
                    for (int d = 0; d < head_size; d++)
                        out_buf[d] *= max_scale;
                } else {
                    weight = expf(score - running_max);
                }

                for (int d = 0; d < head_size; d++)
                    out_buf[d] = fmaf(weight, v_t[d], out_buf[d]);
                running_sum = running_sum * max_scale + weight;
            }
        }

        // Finalize: output = out_buf / running_sum
        float *xb_h = s->xb + h * head_size;
        float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        for (int d = 0; d < head_size; d++) xb_h[d] = out_buf[d] * inv_sum;
    }
}
