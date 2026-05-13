#include "moe_internal.h"

// Build a temporary BnQWeight from pread'd expert data
BnQWeight bn_moe_make_qweight(const void *data, int type, int rows, int cols) {
    BnQWeight w = {0};
    w.data = data;
    w.type = type;
    w.rows = rows;
    w.cols = cols;
    // Per-block quants have embedded scales
    if (type == BN_GGUF_TENSOR_I2_S) {
        size_t nelements = (size_t)rows * cols;
        const uint8_t *base = (const uint8_t *)data;
        memcpy(&w.scale, base + nelements / 4, sizeof(float));
    } else {
        w.scale = 1.0f;
    }
    return w;
}

// --- Phase 3: SwiGLU range function for parallel dispatch ---

void bn_moe_swiglu_range(void *ctx, int start, int end) {
    BnSwiGLUCtx *c = (BnSwiGLUCtx *)ctx;
    int i = start;
#ifdef __AVX2__
    for (; i + 7 < end; i += 8) {
        __m256 g = _mm256_loadu_ps(c->gate + i);
        __m256 u = _mm256_loadu_ps(c->up + i);
        _mm256_storeu_ps(c->hb + i, _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
    }
#endif
    for (; i < end; i++) {
        float g = c->gate[i];
        c->hb[i] = (g / (1.0f + expf(-g))) * c->up[i];
    }
}

// Vectorized SwiGLU for pread path (single expert, no dispatch overhead)
void bn_moe_swiglu(float *hb, const float *gate, const float *up, int n) {
    int i = 0;
#ifdef __AVX2__
    for (; i + 7 < n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(hb + i, _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
    }
#elif defined(__ARM_NEON)
    // No fast_silu for NEON — use scalar (expf is the bottleneck either way)
#endif
    for (; i < n; i++) {
        float g = gate[i];
        hb[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// Compiler barrier to prevent reordering of timing calls around dispatches
double bn_moe_time_ms(void) {
    double t = bn_platform_time_ms();
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" ::: "memory");
#endif
    return t;
}

void bn_moe_weighted_add(float *dst, const float *src, float weight, int n) {
    int i = 0;
#ifdef __AVX2__
    __m256 wv = _mm256_set1_ps(weight);
    for (; i + 7 < n; i += 8) {
        __m256 acc = _mm256_loadu_ps(dst + i);
        __m256 val = _mm256_mul_ps(wv, _mm256_loadu_ps(src + i));
        _mm256_storeu_ps(dst + i, _mm256_add_ps(acc, val));
    }
#elif defined(__ARM_NEON)
    float32x4_t wv = vdupq_n_f32(weight);
    for (; i + 3 < n; i += 4) {
        float32x4_t acc = vld1q_f32(dst + i);
        float32x4_t val = vmulq_f32(wv, vld1q_f32(src + i));
        vst1q_f32(dst + i, vaddq_f32(acc, val));
    }
#endif
    for (; i < n; i++)
        dst[i] += weight * src[i];
}

void bn_moe_residual_add(float *x, const float *r, int n) {
    int i = 0;
#ifdef __AVX2__
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i),
                                              _mm256_loadu_ps(r + i)));
#elif defined(__ARM_NEON)
    for (; i + 3 < n; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(r + i)));
#endif
    for (; i < n; i++)
        x[i] += r[i];
}

