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
    void (*swiglu_silu)(float *hb, const float *gate, const float *up,
                        int n, int uses_reference_silu);
    void (*weighted_add)(float *dst, const float *src, float weight, int n);
    void (*residual_add)(float *x, const float *r, int n);
    int supports_mixed_shared_gateup_batch;
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

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2
static float moe_dot_row_scalar(const float *row, const float *x, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; d++)
        sum += row[d] * x[d];
    return sum;
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
static float moe_dot_row_avx2(const float *row, const float *x, int dim) {
    float sum = 0.0f;
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 15 < dim; d += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d),
                             _mm256_loadu_ps(x + d), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d + 8),
                             _mm256_loadu_ps(x + d + 8), a1);
    }
    sum = bn_avx2_hsum_ps(_mm256_add_ps(a0, a1));
    for (; d < dim; d++)
        sum += row[d] * x[d];
    return sum;
}
#endif

static int moe_dot4_rows_scalar(float *out, const float *router_w,
                                const float *x, int dim, int start_expert) {
    (void)out;
    (void)router_w;
    (void)x;
    (void)dim;
    (void)start_expert;
    return 0;
}

#if BN_TRANSFORMER_CPU_HAS_AVX2
static int moe_dot4_rows_avx2(float *out, const float *router_w,
                              const float *x, int dim, int start_expert) {
    const float *row0 = router_w + (size_t)(start_expert + 0) * dim;
    const float *row1 = router_w + (size_t)(start_expert + 1) * dim;
    const float *row2 = router_w + (size_t)(start_expert + 2) * dim;
    const float *row3 = router_w + (size_t)(start_expert + 3) * dim;
    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 15 < dim; d += 16) {
        __m256 x0 = _mm256_loadu_ps(x + d);
        __m256 x1 = _mm256_loadu_ps(x + d + 8);
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
    for (; d < dim; d++) {
        float xv = x[d];
        sum0 += row0[d] * xv;
        sum1 += row1[d] * xv;
        sum2 += row2[d] * xv;
        sum3 += row3[d] * xv;
    }
    out[0] = sum0;
    out[1] = sum1;
    out[2] = sum2;
    out[3] = sum3;
    return 1;
}
#endif

static void moe_swiglu_silu_scalar(float *hb, const float *gate,
                                   const float *up, int n, int uses_reference_silu) {
    (void)uses_reference_silu;
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        hb[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void moe_swiglu_silu_avx2(float *hb, const float *gate,
                                 const float *up, int n, int uses_reference_silu) {
    if (uses_reference_silu) {
        moe_swiglu_silu_scalar(hb, gate, up, n, uses_reference_silu);
        return;
    }
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(hb + i, _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
    }
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
        float32x4_t val = vmulq_f32(wv, vld1q_f32(src + i));
        vst1q_f32(dst + i, vaddq_f32(acc, val));
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
        moe_swiglu_silu_scalar,
        moe_weighted_add_neon,
        moe_residual_add_neon,
        0,
    };
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    static const BnMoECPUOps ops = {
        "avx2",
        moe_rmsnorm_avx2,
        moe_dot_row_avx2,
        moe_dot4_rows_avx2,
        moe_swiglu_silu_avx2,
        moe_weighted_add_avx2,
        moe_residual_add_avx2,
        1,
    };
#elif BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
    static const BnMoECPUOps ops = {
        "wasm",
        moe_rmsnorm_wasm,
        moe_dot_row_scalar,
        moe_dot4_rows_scalar,
        moe_swiglu_silu_scalar,
        moe_weighted_add_scalar,
        moe_residual_add_scalar,
        0,
    };
#else
    static const BnMoECPUOps ops = {
        "scalar",
        moe_rmsnorm_scalar,
        moe_dot_row_scalar,
        moe_dot4_rows_scalar,
        moe_swiglu_silu_scalar,
        moe_weighted_add_scalar,
        moe_residual_add_scalar,
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
    int can_batch = bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
        shared_gate_type, shared_up_type, batch_type,
        mixed_shared_gateup_supported);
    for (int i = 1; can_batch && i < n_tasks; i++)
        can_batch = bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
            shared_gate_type, shared_up_type, tasks[i].W->type,
            mixed_shared_gateup_supported);
    return can_batch;
}

void bn_moe_quant_matvec(float *out,
                         const BnQWeight *W,
                         const float *x,
                         int8_t *quantized_buf,
                         BnThreadPool *pool) {
    bn_quant_matvec(out, W, x, quantized_buf, pool);
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

void bn_moe_weighted_add(float *dst, const float *src, float weight, int n) {
    bn_moe_cpu_ops()->weighted_add(dst, src, weight, n);
}

void bn_moe_residual_add(float *x, const float *r, int n) {
    bn_moe_cpu_ops()->residual_add(x, r, n);
}
