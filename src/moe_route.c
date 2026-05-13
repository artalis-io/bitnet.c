#include "moe_internal.h"

// --- Phase 4: Vectorized router ---

typedef struct {
    float *logits;
    const float *router_w;
    const float *x;
    int dim;
} BnRouterCtx;

static void moe_router_range(void *ctx, int start, int end) {
    BnRouterCtx *c = (BnRouterCtx *)ctx;
#if defined(__AVX2__)
    int avx2_e = start;
    for (; avx2_e + 3 < end; avx2_e += 4) {
        const float *row0 = c->router_w + (size_t)(avx2_e + 0) * c->dim;
        const float *row1 = c->router_w + (size_t)(avx2_e + 1) * c->dim;
        const float *row2 = c->router_w + (size_t)(avx2_e + 2) * c->dim;
        const float *row3 = c->router_w + (size_t)(avx2_e + 3) * c->dim;
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        int d = 0;
        for (; d + 15 < c->dim; d += 16) {
            __m256 x0 = _mm256_loadu_ps(c->x + d);
            __m256 x1 = _mm256_loadu_ps(c->x + d + 8);
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + d),     x0, a00);
            a01 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + d + 8), x1, a01);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + d),     x0, a10);
            a11 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + d + 8), x1, a11);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(row2 + d),     x0, a20);
            a21 = _mm256_fmadd_ps(_mm256_loadu_ps(row2 + d + 8), x1, a21);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(row3 + d),     x0, a30);
            a31 = _mm256_fmadd_ps(_mm256_loadu_ps(row3 + d + 8), x1, a31);
        }
        float sum0 = bn_avx2_hsum_ps(_mm256_add_ps(a00, a01));
        float sum1 = bn_avx2_hsum_ps(_mm256_add_ps(a10, a11));
        float sum2 = bn_avx2_hsum_ps(_mm256_add_ps(a20, a21));
        float sum3 = bn_avx2_hsum_ps(_mm256_add_ps(a30, a31));
        for (; d < c->dim; d++) {
            float x = c->x[d];
            sum0 += row0[d] * x;
            sum1 += row1[d] * x;
            sum2 += row2[d] * x;
            sum3 += row3[d] * x;
        }
        c->logits[avx2_e + 0] = sum0;
        c->logits[avx2_e + 1] = sum1;
        c->logits[avx2_e + 2] = sum2;
        c->logits[avx2_e + 3] = sum3;
    }
    start = avx2_e;
#endif
    for (int e = start; e < end; e++) {
        const float *row = c->router_w + (size_t)e * c->dim;
        float sum = 0.0f;
#ifdef __ARM_NEON
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 15 < c->dim; d += 16) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(row + d),      vld1q_f32(c->x + d));
            acc1 = vfmaq_f32(acc1, vld1q_f32(row + d + 4),  vld1q_f32(c->x + d + 4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(row + d + 8),  vld1q_f32(c->x + d + 8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(row + d + 12), vld1q_f32(c->x + d + 12));
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        sum = vaddvq_f32(acc0);
        for (; d < c->dim; d++)
            sum += row[d] * c->x[d];
#elif defined(__AVX2__)
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        int d = 0;
        for (; d + 15 < c->dim; d += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d),     _mm256_loadu_ps(c->x + d),     a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d + 8), _mm256_loadu_ps(c->x + d + 8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 hi = _mm256_extractf128_ps(a0, 1);
        __m128 lo = _mm256_castps256_ps128(a0);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);
        for (; d < c->dim; d++)
            sum += row[d] * c->x[d];
#else
        for (int d = 0; d < c->dim; d++)
            sum += row[d] * c->x[d];
#endif
        c->logits[e] = sum;
    }
}

// Router: SIMD matvec -> softmax -> top-K selection
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k, BnThreadPool *pool) {
    // Router matvec: vectorized + thread-dispatched
    BnRouterCtx rctx = { ms->router_logits, router_w, x, dim };
    BnTPTask rtask = { moe_router_range, &rctx, n_experts };
    bn_tp_dispatch(pool, &rtask, 1);

    // Softmax over all experts
    float max_val = ms->router_logits[0];
    for (int e = 1; e < n_experts; e++)
        if (ms->router_logits[e] > max_val)
            max_val = ms->router_logits[e];

    float sum = 0.0f;
    for (int e = 0; e < n_experts; e++) {
        ms->router_logits[e] = expf(ms->router_logits[e] - max_val);
        sum += ms->router_logits[e];
    }
    // Top-K selection (partial sort). Select over exp(logit - max);
    // the all-expert softmax denominator is common and cancels after
    // the selected weights are normalized below.
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_experts; e++) {
            if (ms->router_logits[e] > best_val) {
                best_val = ms->router_logits[e];
                best = e;
            }
        }
        ms->expert_indices[i] = best;
        ms->expert_weights[i] = best_val / sum;
        ms->router_logits[best] = -1.0f;
    }

    // Normalize selected weights to sum to 1.0
    float wsum = 0.0f;
    for (int i = 0; i < k; i++)
        wsum += ms->expert_weights[i];
    if (wsum > 0.0f) {
        for (int i = 0; i < k; i++)
            ms->expert_weights[i] /= wsum;
    }
}

