#include "moe_internal.h"
#include "transformer_cpu_features_internal.h"
#include "simd_helpers.h"
#include "transformer_rmsnorm_internal.h"

typedef struct {
    const char *name;
    void (*rmsnorm)(float *out, const float *x, const float *w,
                    int size, float eps);
    float (*dot_row)(const float *row, const float *x, int dim);
    int (*dot4_rows)(float *out, const float *router_w, const float *x,
                     int dim, int start_expert);
    double (*softmax_exp)(float *out, const float *logits, int n,
                          float max_logit);
    void (*swiglu_silu)(float *hb, const float *gate, const float *up,
                        int n, int uses_reference_silu);
    void (*weighted_add)(float *dst, const float *src, float weight, int n);
    void (*residual_add)(float *x, const float *r, int n);
    int supports_mixed_shared_gateup_batch;
    int supports_router_batch;
    int prefers_expert_gemv_batch;
} BnMoECPUOps;

#if BN_TRANSFORMER_CPU_HAS_NEON
static void moe_rmsnorm_neon(float *out, const float *x, const float *w,
                             int size, float eps) {
    bn_transformer_rmsnorm_neon(out, x, w, size, eps);
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_rmsnorm_avx2(float *out, const float *x, const float *w,
                             int size, float eps) {
    bn_transformer_rmsnorm_avx2(out, x, w, size, eps);
}
#endif

#if BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void moe_rmsnorm_wasm(float *out, const float *x, const float *w,
                             int size, float eps) {
    bn_transformer_rmsnorm_wasm(out, x, w, size, eps);
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2 && \
    !BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void moe_rmsnorm_scalar(float *out, const float *x, const float *w,
                               int size, float eps) {
    bn_transformer_rmsnorm_scalar(out, x, w, size, eps);
}
#endif

float bn_moe_dot_row_reference(const float *row, const float *x, int dim) {
#if defined(__AVX512F__)
    __m512 acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(),
    };
    int d = 0;
    for (; d + 63 < dim; d += 64)
        for (int lane = 0; lane < 4; lane++)
            acc[lane] = _mm512_fmadd_ps(
                _mm512_loadu_ps(row + d + lane * 16),
                _mm512_loadu_ps(x + d + lane * 16), acc[lane]);
    acc[0] = _mm512_add_ps(acc[0], acc[2]);
    acc[1] = _mm512_add_ps(acc[1], acc[3]);
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]));
    for (; d < dim; d++)
        sum += row[d] * x[d];
    return sum;
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    int d = 0;
    for (; d + 31 < dim; d += 32)
        for (int lane = 0; lane < 4; lane++)
            acc[lane] = _mm256_fmadd_ps(
                _mm256_loadu_ps(row + d + lane * 8),
                _mm256_loadu_ps(x + d + lane * 8), acc[lane]);
    acc[0] = _mm256_add_ps(acc[0], acc[2]);
    acc[1] = _mm256_add_ps(acc[1], acc[3]);
    acc[0] = _mm256_add_ps(acc[0], acc[1]);
    __m128 half = _mm_add_ps(_mm256_castps256_ps128(acc[0]),
                             _mm256_extractf128_ps(acc[0], 1));
    half = _mm_hadd_ps(half, half);
    float sum = _mm_cvtss_f32(_mm_hadd_ps(half, half));
    for (; d < dim; d++)
        sum += row[d] * x[d];
    return sum;
#else
    volatile float sum = 0.0f;
    for (int d = 0; d < dim; d++)
        sum += row[d] * x[d];
    return sum;
#endif
}

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2
static float moe_dot_row_scalar(const float *row, const float *x, int dim) {
    return bn_moe_dot_row_reference(row, x, dim);
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static float moe_dot_row_neon(const float *row, const float *x, int dim) {
    float sum = 0.0f;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 15 < dim; d += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + d),      vld1q_f32(x + d));
        acc1 = vfmaq_f32(acc1, vld1q_f32(row + d + 4),  vld1q_f32(x + d + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(row + d + 8),  vld1q_f32(x + d + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(row + d + 12), vld1q_f32(x + d + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    sum = vaddvq_f32(acc0);
    for (; d < dim; d++)
        sum += row[d] * x[d];
    return sum;
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static inline float moe_reduce_16_avx2(__m256 low, __m256 high) {
    __m256 half = _mm256_add_ps(high, low);
    __m128 quarter = _mm_add_ps(_mm256_extractf128_ps(half, 1),
                                _mm256_castps256_ps128(half));
    __m128 pairs = _mm_add_ps(_mm_movehl_ps(quarter, quarter), quarter);
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(
        pairs, pairs, _MM_SHUFFLE(1, 1, 1, 1))));
}

static inline float moe_reduce_8_avx2(__m256 value) {
    __m128 half = _mm_add_ps(_mm256_extractf128_ps(value, 1),
                             _mm256_castps256_ps128(value));
    half = _mm_add_ps(half, _mm_movehl_ps(half, half));
    return _mm_cvtss_f32(_mm_add_ss(half, _mm_movehdup_ps(half)));
}

static inline __m256 moe_reference_exp_avx2(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(
        x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(
        n, _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(
        e, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
        _mm256_set1_ps(126.0f), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                            _mm256_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                            _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
        return _mm256_fmadd_ps(j, k, k);
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(
            n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32((int)UINT32_C(0x82000000)));
    const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(
        g, _mm256_set1_epi32((int)UINT32_C(0x7f000000))));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
        _mm256_set1_ps(192.0f), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(_mm256_castsi256_ps(c),
                    _mm256_fmadd_ps(k, j, k)))));
}

static float moe_dot_row_avx2(const float *row, const float *x, int dim) {
    float sum = 0.0f;
#if defined(__AVX512F__)
    __m512 acc[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(),
    };
    int d = 0;
    for (; d + 63 < dim; d += 64)
        for (int lane = 0; lane < 4; lane++)
            acc[lane] = _mm512_fmadd_ps(
                _mm512_loadu_ps(row + d + lane * 16),
                _mm512_loadu_ps(x + d + lane * 16), acc[lane]);
    sum = _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(acc[0], acc[2]), _mm512_add_ps(acc[1], acc[3])));
#else
    __m256 lo[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps() };
    __m256 hi[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps() };
    int d = 0;
    for (; d + 63 < dim; d += 64)
        for (int group = 0; group < 4; group++) {
            int off = d + group * 16;
            lo[group] = _mm256_fmadd_ps(
                _mm256_loadu_ps(row + off), _mm256_loadu_ps(x + off),
                lo[group]);
            hi[group] = _mm256_fmadd_ps(
                _mm256_loadu_ps(row + off + 8),
                _mm256_loadu_ps(x + off + 8), hi[group]);
        }
    __m256 merged_lo = _mm256_add_ps(_mm256_add_ps(lo[0], lo[2]),
                                     _mm256_add_ps(lo[1], lo[3]));
    __m256 merged_hi = _mm256_add_ps(_mm256_add_ps(hi[0], hi[2]),
                                     _mm256_add_ps(hi[1], hi[3]));
    sum = moe_reduce_16_avx2(merged_lo, merged_hi);
#endif
    for (; d < dim; d++)
        sum += row[d] * x[d];
    return sum;
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_AVX2 || BN_TRANSFORMER_CPU_HAS_NEON
static int moe_dot4_rows_scalar(float *out, const float *router_w,
                                const float *x, int dim, int start_expert) {
    (void)out;
    (void)router_w;
    (void)x;
    (void)dim;
    (void)start_expert;
    return 0;
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2
static double moe_softmax_exp_scalar(float *out, const float *logits, int n,
                                     float max_logit) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float value = expf(logits[i] - max_logit);
        out[i] = value;
        sum += (double)value;
    }
    return sum;
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static double moe_softmax_exp_neon(float *out, const float *logits, int n,
                                   float max_logit) {
    double sum = 0.0;
    float32x4_t maxv = vdupq_n_f32(max_logit);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t value = bn_neon_fast_exp_f32(
            vsubq_f32(vld1q_f32(logits + i), maxv));
        vst1q_f32(out + i, value);
        sum += (double)vaddvq_f32(value);
    }
    for (; i < n; i++) {
        float value = expf(logits[i] - max_logit);
        out[i] = value;
        sum += (double)value;
    }
    return sum;
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static double moe_softmax_exp_avx2(float *out, const float *logits, int n,
                                   float max_logit) {
    double sum = 0.0;
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 maxv = _mm512_set1_ps(max_logit);
    for (; i + 15 < n; i += 16) {
        __m512 value = bn_avx512_fast_exp_ps(
            _mm512_sub_ps(_mm512_loadu_ps(logits + i), maxv));
        _mm512_storeu_ps(out + i, value);
        sum += (double)_mm512_reduce_add_ps(value);
    }
#else
    __m256 maxv = _mm256_set1_ps(max_logit);
    for (; i + 15 < n; i += 16) {
        __m256 low = moe_reference_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(logits + i), maxv));
        __m256 high = moe_reference_exp_avx2(
            _mm256_sub_ps(_mm256_loadu_ps(logits + i + 8), maxv));
        _mm256_storeu_ps(out + i, low);
        _mm256_storeu_ps(out + i + 8, high);
        /* ggml_vec_soft_max_f32 reduces each AVX2 vector independently
         * before widening the partial sum to double. */
        sum += (double)moe_reduce_8_avx2(low);
        sum += (double)moe_reduce_8_avx2(high);
    }
#endif
    for (; i < n; i++) {
        float value = expf(logits[i] - max_logit);
        out[i] = value;
        sum += (double)value;
    }
    return sum;
}

static int moe_dot4_rows_avx2(float *out, const float *router_w,
                              const float *x, int dim, int start_expert) {
#if defined(__AVX512F__)
    __m512 acc[4][4];
    for (int r = 0; r < 4; r++)
        for (int j = 0; j < 4; j++)
            acc[r][j] = _mm512_setzero_ps();
    int d = 0;
    for (; d + 63 < dim; d += 64)
        for (int j = 0; j < 4; j++) {
            __m512 xv = _mm512_loadu_ps(x + d + j * 16);
            for (int r = 0; r < 4; r++) {
                const float *w = router_w + (size_t)(start_expert + r) * dim;
                acc[r][j] = _mm512_fmadd_ps(
                    _mm512_loadu_ps(w + d + j * 16), xv, acc[r][j]);
            }
        }
    for (int r = 0; r < 4; r++) {
        const float *w = router_w + (size_t)(start_expert + r) * dim;
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(acc[r][0], acc[r][2]),
            _mm512_add_ps(acc[r][1], acc[r][3])));
        for (int tail = d; tail < dim; tail++) sum += w[tail] * x[tail];
        out[r] = sum;
    }
    return 1;
#else
    const float *rows[4];
    __m256 lo[4][4], hi[4][4];
    for (int r = 0; r < 4; r++) {
        rows[r] = router_w + (size_t)(start_expert + r) * dim;
        for (int group = 0; group < 4; group++) {
            lo[r][group] = _mm256_setzero_ps();
            hi[r][group] = _mm256_setzero_ps();
        }
    }
    int d = 0;
    for (; d + 63 < dim; d += 64) {
        for (int group = 0; group < 4; group++) {
            int off = d + group * 16;
            __m256 xlo = _mm256_loadu_ps(x + off);
            __m256 xhi = _mm256_loadu_ps(x + off + 8);
            for (int r = 0; r < 4; r++) {
                lo[r][group] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(rows[r] + off), xlo, lo[r][group]);
                hi[r][group] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(rows[r] + off + 8), xhi, hi[r][group]);
            }
        }
    }
    float sums[4];
    for (int r = 0; r < 4; r++) {
        __m256 merged_lo = _mm256_add_ps(
            _mm256_add_ps(lo[r][0], lo[r][2]),
            _mm256_add_ps(lo[r][1], lo[r][3]));
        __m256 merged_hi = _mm256_add_ps(
            _mm256_add_ps(hi[r][0], hi[r][2]),
            _mm256_add_ps(hi[r][1], hi[r][3]));
        sums[r] = moe_reduce_16_avx2(merged_lo, merged_hi);
    }
    for (; d < dim; d++) {
        float xv = x[d];
        for (int r = 0; r < 4; r++)
            sums[r] += rows[r][d] * xv;
    }
    for (int r = 0; r < 4; r++) out[r] = sums[r];
    return 1;
#endif
}

#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_swiglu_silu_scalar(float *hb, const float *gate,
                                   const float *up, int n,
                                   int uses_reference_silu) {
    (void)uses_reference_silu;
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        hb[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static void moe_swiglu_silu_neon(float *hb, const float *gate,
                                 const float *up, int n,
                                 int uses_reference_silu) {
    (void)uses_reference_silu;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t g = vld1q_f32(gate + i);
        float32x4_t u = vld1q_f32(up + i);
        vst1q_f32(hb + i,
                  vmulq_f32(bn_neon_fast_silu_f32(g), u));
    }
    for (; i < n; i++) {
        float32x4_t g = vdupq_n_f32(gate[i]);
        hb[i] = vgetq_lane_f32(bn_neon_fast_silu_f32(g), 0) * up[i];
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_swiglu_silu_avx2(float *hb, const float *gate,
                                 const float *up, int n, int uses_reference_silu) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    (void)uses_reference_silu;
#endif
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 g = _mm512_loadu_ps(gate + i);
        __m512 u = _mm512_loadu_ps(up + i);
        _mm512_storeu_ps(hb + i,
                         _mm512_mul_ps(bn_avx512_fast_silu_ps(g), u));
    }
#else
    for (; i + 7 < n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 exp_neg = uses_reference_silu
            ? moe_reference_exp_avx2(neg_g)
            : bn_avx2_fast_exp_avx512_ps(neg_g);
        __m256 silu = _mm256_div_ps(
            g, _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg));
        _mm256_storeu_ps(hb + i, _mm256_mul_ps(silu, u));
    }
#endif
    for (; i < n; i++) {
        float g = gate[i];
        hb[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_weighted_add_scalar(float *dst, const float *src,
                                    float weight, int n) {
    for (int i = 0; i < n; i++)
        dst[i] += weight * src[i];
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static void moe_weighted_add_neon(float *dst, const float *src,
                                  float weight, int n) {
    int i = 0;
    float32x4_t wv = vdupq_n_f32(weight);
    for (; i + 3 < n; i += 4) {
        float32x4_t acc = vld1q_f32(dst + i);
        vst1q_f32(dst + i,
                  vfmaq_f32(acc, wv, vld1q_f32(src + i)));
    }
    for (; i < n; i++)
        dst[i] += weight * src[i];
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_weighted_add_avx2(float *dst, const float *src,
                                  float weight, int n) {
    int i = 0;
    __m256 wv = _mm256_set1_ps(weight);
    for (; i + 7 < n; i += 8) {
        __m256 acc = _mm256_loadu_ps(dst + i);
        __m256 val = _mm256_mul_ps(wv, _mm256_loadu_ps(src + i));
        _mm256_storeu_ps(dst + i, _mm256_add_ps(acc, val));
    }
    for (; i < n; i++)
        dst[i] += weight * src[i];
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_residual_add_scalar(float *x, const float *r, int n) {
    for (int i = 0; i < n; i++)
        x[i] += r[i];
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static void moe_residual_add_neon(float *x, const float *r, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(r + i)));
    for (; i < n; i++)
        x[i] += r[i];
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_residual_add_avx2(float *x, const float *r, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i),
                                              _mm256_loadu_ps(r + i)));
    for (; i < n; i++)
        x[i] += r[i];
}
#endif

static const BnMoECPUOps *bn_moe_cpu_ops(void) {
#if BN_TRANSFORMER_CPU_HAS_NEON
    static const BnMoECPUOps ops = {
        "neon",
        moe_rmsnorm_neon,
        moe_dot_row_neon,
        moe_dot4_rows_scalar,
        moe_softmax_exp_neon,
        moe_swiglu_silu_neon,
        moe_weighted_add_neon,
        moe_residual_add_neon,
        0,
        0,
        0,
    };
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    static const BnMoECPUOps ops = {
        "avx2",
        moe_rmsnorm_avx2,
        moe_dot_row_avx2,
        moe_dot4_rows_avx2,
        moe_softmax_exp_avx2,
        moe_swiglu_silu_avx2,
        moe_weighted_add_avx2,
        moe_residual_add_avx2,
        1,
        1,
        1,
    };
#elif BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
    static const BnMoECPUOps ops = {
        "wasm",
        moe_rmsnorm_wasm,
        moe_dot_row_scalar,
        moe_dot4_rows_scalar,
        moe_softmax_exp_scalar,
        moe_swiglu_silu_scalar,
        moe_weighted_add_scalar,
        moe_residual_add_scalar,
        0,
        0,
        0,
    };
#else
    static const BnMoECPUOps ops = {
        "scalar",
        moe_rmsnorm_scalar,
        moe_dot_row_scalar,
        moe_dot4_rows_scalar,
        moe_softmax_exp_scalar,
        moe_swiglu_silu_scalar,
        moe_weighted_add_scalar,
        moe_residual_add_scalar,
        0,
        0,
        0,
    };
#endif
    return &ops;
}

void bn_moe_rmsnorm(float *out, const float *x, const float *w,
                    int size, float eps) {
    bn_moe_cpu_ops()->rmsnorm(out, x, w, size, eps);
}

float bn_moe_dot_row(const float *row, const float *x, int dim) {
    return bn_moe_cpu_ops()->dot_row(row, x, dim);
}

int bn_moe_dot4_rows(float *out, const float *router_w, const float *x,
                     int dim, int start_expert) {
    return bn_moe_cpu_ops()->dot4_rows(out, router_w, x, dim, start_expert);
}

double bn_moe_softmax_exp(float *out, const float *logits, int n,
                          float max_logit) {
    return bn_moe_cpu_ops()->softmax_exp(out, logits, n, max_logit);
}

void bn_moe_swiglu_silu(float *hb, const float *gate, const float *up,
                        int n, int uses_reference_silu) {
    bn_moe_cpu_ops()->swiglu_silu(hb, gate, up, n, uses_reference_silu);
}

int bn_moe_can_batch_shared_gateup(const BnMatvecTask *tasks, int n_tasks,
                                   int shared_gate_type, int shared_up_type) {
    if (!tasks || n_tasks <= 0)
        return 0;
    int mixed_shared_gateup_supported =
        bn_moe_cpu_ops()->supports_mixed_shared_gateup_batch;
    int batch_type = tasks[0].W->type;
    BnMoESharedGateupBatchPolicy batch_policy =
        bn_moe_shared_gateup_batch_policy(
            shared_gate_type, shared_up_type, batch_type,
            mixed_shared_gateup_supported);
    for (int i = 1; batch_policy.can_batch && i < n_tasks; i++)
        batch_policy = bn_moe_shared_gateup_batch_policy(
            shared_gate_type, shared_up_type, tasks[i].W->type,
            mixed_shared_gateup_supported);
    return batch_policy.can_batch;
}

int bn_moe_router_batch_logits(float *out, const float *weights,
                                const float *x, int tokens, int dim,
                                int experts, BnThreadPool *pool) {
    if (tokens > 1 && bn_moe_cpu_ops()->supports_router_batch) {
        BnQWeight w = {weights, BN_GGUF_TENSOR_F32, experts, dim, 1.0f};
        bn_quant_matmul(out, &w, x, tokens, NULL, pool);
        return 1;
    }
    return 0;
}

void bn_moe_quant_matvec(float *out,
                         const BnQWeight *W,
                         const float *x,
                         int8_t *quantized_buf,
                         BnThreadPool *pool) {
    bn_quant_matvec(out, W, x, quantized_buf, pool);
}

size_t bn_moe_quant_prepared_weight_size(const BnQWeight *weight) {
    return bn_quant_prepared_qweight_size(weight, NULL);
}

int bn_moe_quant_batch_preparation_worthwhile(const BnQWeight *weight,
                                              int n_tokens, BnThreadPool *pool) {
    return bn_quant_batch_preparation_worthwhile(weight, n_tokens, pool);
}

int bn_moe_quant_matvec_uses_prepared_weight(const BnQWeight *weight,
                                             uint32_t flags,
                                             BnThreadPool *pool) {
    return bn_quant_matvec_uses_prepared_weight(weight, flags, pool);
}

int bn_moe_quant_prepare_weight(BnPreparedWeight *prepared,
                                const BnQWeight *weight,
                                SHArena *arena) {
    return bn_quant_prepare_qweight(prepared, weight, arena);
}

void bn_moe_quant_matvec_prepared(float *out,
                                  const BnQWeight *W,
                                  const BnPreparedWeight *prepared,
                                  const float *x,
                                  int8_t *quantized_buf,
                                  BnThreadPool *pool) {
    bn_quant_matvec_prepared(out, W, prepared, x, quantized_buf, pool);
}

void bn_moe_quant_matvec_batch(const BnMatvecTask *tasks,
                               int n_tasks,
                               const float *x,
                               int8_t *quantized_buf,
                               BnThreadPool *pool) {
    bn_quant_matvec_batch(tasks, n_tasks, x, quantized_buf, pool);
}

void bn_moe_quant_matvec_multi(const BnMatvecMultiTask *tasks,
                               int n_tasks,
                               int8_t *quantized_bufs,
                               BnThreadPool *pool) {
    bn_quant_matvec_multi(tasks, n_tasks, quantized_bufs, pool);
}

void bn_moe_quant_matmul(float *out,
                         const BnQWeight *W,
                         const float *x,
                         int n_tokens,
                         int8_t *quantized_buf,
                         BnThreadPool *pool) {
    bn_quant_matmul(out, W, x, n_tokens, quantized_buf, pool);
}

void bn_moe_quant_matmul_prepared(float *out,
                                  const BnQWeight *W,
                                  const BnPreparedWeight *prepared,
                                  const float *x,
                                  int n_tokens,
                                  int8_t *quantized_buf,
                                  BnThreadPool *pool) {
    if (bn_moe_cpu_ops()->prefers_expert_gemv_batch) {
        bn_quant_matmul_prepared_multi_gemv(
            &out, &W, &prepared, 1, x, n_tokens, quantized_buf, pool);
        return;
    }
    bn_quant_matmul_prepared(
        out, W, prepared, x, n_tokens, quantized_buf, pool);
}

void bn_moe_quant_matmul_prepared_multi(
    float **out, const BnQWeight **weights,
    const BnPreparedWeight **prepared, int n,
    const float *x, int n_tokens, int8_t *quantized_buf,
    BnThreadPool *pool) {
    if (bn_moe_cpu_ops()->prefers_expert_gemv_batch) {
        bn_quant_matmul_prepared_multi_gemv(
            out, weights, prepared, n, x, n_tokens, quantized_buf, pool);
        return;
    }
    bn_quant_matmul_prepared_multi(
        out, weights, prepared, n, x, n_tokens, quantized_buf, pool);
}

void bn_moe_weighted_add(float *dst, const float *src, float weight, int n) {
    bn_moe_cpu_ops()->weighted_add(dst, src, weight, n);
}

void bn_moe_residual_add(float *x, const float *r, int n) {
    bn_moe_cpu_ops()->residual_add(x, r, n);
}
