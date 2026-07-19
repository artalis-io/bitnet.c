#include "transformer_ssm_internal.h"
#include "transformer_simd_internal.h"

#ifdef __AVX2__

static inline float bn_ssm_avx2_hsum_ggml_ps(__m256 v) {
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(v),
                                 _mm256_extractf128_ps(v, 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}

// Conv1d + SiLU over channel range [start, end).
// Keep scalar accumulation order here: recurrent SSM layers are sensitive
// enough that AVX2 FMA regrouping changes greedy token selection vs llama.cpp.
void bn_transformer_ssm_conv_silu_avx2_range(void *ctx, int start, int end) {
    BnSSMConvCtx *c = (BnSSMConvCtx *)ctx;
    float *qkv = c->qkv;
    float *conv_state = c->conv_state;
    const float *conv1d_w = c->conv1d_w;
    int qkv_dim = c->qkv_dim;
    int kern = c->kern;

    for (int ch = start; ch < end; ch++) {
        float sum = 0;
        for (int k = 0; k < kern - 1; k++)
            sum += conv_state[(size_t)k * qkv_dim + ch] *
                   conv1d_w[(size_t)ch * kern + k];
        float cur = qkv[ch];
        sum += cur * conv1d_w[(size_t)ch * kern + (kern - 1)];
        for (int k = 0; k < kern - 2; k++)
            conv_state[(size_t)k * qkv_dim + ch] =
                conv_state[(size_t)(k + 1) * qkv_dim + ch];
        conv_state[(size_t)(kern - 2) * qkv_dim + ch] = cur;
        qkv[ch] = sum / (1.0f + expf(-sum));
    }
}

// L2 normalize Q and K per head, range over heads [start, end)
void bn_transformer_ssm_l2norm_avx2_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = (BnSSML2NormCtx *)ctx;
    int hd = c->head_dim;
    float eps = c->eps;

    for (int h = start; h < end; h++) {
        float *qh = c->q + h * hd;
        float *kh = c->k + h * hd;

        // Sum-of-squares with 2x unroll (128 dims = 16 vectors = 8 iters)
        __m256 qss0 = _mm256_setzero_ps(), qss1 = _mm256_setzero_ps();
        __m256 kss0 = _mm256_setzero_ps(), kss1 = _mm256_setzero_ps();
        for (int d = 0; d < hd; d += 16) {
            __m256 q0 = _mm256_loadu_ps(qh + d);
            __m256 q1 = _mm256_loadu_ps(qh + d + 8);
            qss0 = _mm256_fmadd_ps(q0, q0, qss0);
            qss1 = _mm256_fmadd_ps(q1, q1, qss1);
            __m256 k0 = _mm256_loadu_ps(kh + d);
            __m256 k1 = _mm256_loadu_ps(kh + d + 8);
            kss0 = _mm256_fmadd_ps(k0, k0, kss0);
            kss1 = _mm256_fmadd_ps(k1, k1, kss1);
        }
        float qn = bn_ssm_avx2_hsum_ggml_ps(_mm256_add_ps(qss0, qss1));
        float kn = bn_ssm_avx2_hsum_ggml_ps(_mm256_add_ps(kss0, kss1));
        __m256 qscale = _mm256_set1_ps(1.0f / fmaxf(sqrtf(qn), eps));
        __m256 kscale = _mm256_set1_ps(1.0f / fmaxf(sqrtf(kn), eps));
        for (int d = 0; d < hd; d += 8) {
            _mm256_storeu_ps(qh + d, _mm256_mul_ps(_mm256_loadu_ps(qh + d), qscale));
            _mm256_storeu_ps(kh + d, _mm256_mul_ps(_mm256_loadu_ps(kh + d), kscale));
        }
    }
}

static inline float bn_ssm_avx2_dot_f32(int n, const float *x, const float *y) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    int np = n & ~31;
    for (; i < np; i += 32) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 y0 = _mm256_loadu_ps(y + i);
        s0 = _mm256_fmadd_ps(x0, y0, s0);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        s3 = _mm256_fmadd_ps(x3, y3, s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s2),
                             _mm256_add_ps(s1, s3));
    float sum = bn_ssm_avx2_hsum_ggml_ps(s);
    for (; i < n; i++)
        sum += x[i] * y[i];
    return sum;
}

static inline float bn_ssm_avx2_scale_dot_f32(int n, float *x,
                                              const float *y, float scale) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    int np = n & ~31;
    for (; i < np; i += 32) {
        __m256 x0 = _mm256_mul_ps(_mm256_loadu_ps(x + i), vscale);
        _mm256_storeu_ps(x + i, x0);
        __m256 y0 = _mm256_loadu_ps(y + i);
        s0 = _mm256_fmadd_ps(x0, y0, s0);
        __m256 x1 = _mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vscale);
        _mm256_storeu_ps(x + i + 8, x1);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        __m256 x2 = _mm256_mul_ps(_mm256_loadu_ps(x + i + 16), vscale);
        _mm256_storeu_ps(x + i + 16, x2);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
        __m256 x3 = _mm256_mul_ps(_mm256_loadu_ps(x + i + 24), vscale);
        _mm256_storeu_ps(x + i + 24, x3);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        s3 = _mm256_fmadd_ps(x3, y3, s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s2),
                             _mm256_add_ps(s1, s3));
    float sum = bn_ssm_avx2_hsum_ggml_ps(s);
    for (; i < n; i++) {
        x[i] *= scale;
        sum += x[i] * y[i];
    }
    return sum;
}

static inline void bn_ssm_avx2_mad_f32(int n, float *y, const float *x,
                                       float v) {
    __m256 vv = _mm256_set1_ps(v);
    int i = 0;
    int np = n & ~31;
    for (; i < np; i += 32) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 y0 = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(x0, vv, y0));
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        _mm256_storeu_ps(y + i + 8, _mm256_fmadd_ps(x1, vv, y1));
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        _mm256_storeu_ps(y + i + 16, _mm256_fmadd_ps(x2, vv, y2));
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        _mm256_storeu_ps(y + i + 24, _mm256_fmadd_ps(x3, vv, y3));
    }
    for (; i < n; i++)
        y[i] += x[i] * v;
}

// Delta rule recurrence over V-head range [start, end)
void bn_transformer_ssm_delta_avx2_range(void *ctx, int start, int end) {
    BnSSMDeltaCtx *c = (BnSSMDeltaCtx *)ctx;
    int head_k_dim = c->head_k_dim;
    int head_v_dim = c->head_v_dim;
    int num_k_heads = c->num_k_heads;
    float q_scale = c->q_scale;

    for (int hv = start; hv < end; hv++) {
        int hk = hv % num_k_heads;
        const float *qh = c->q + hk * head_k_dim;
        const float *kh = c->k + hk * head_k_dim;
        float *vh = c->v + hv * head_v_dim;
        float *S = c->state + (size_t)hv * head_k_dim * head_v_dim;
        float decay = c->alpha[hv];
        float beta = c->beta[hv];

        // State is transposed: S[v][k] stores the mathematical state[k][v],
        // matching llama.cpp fused GDN's contiguous dot/mad order.
        float *oh = c->out + hv * head_v_dim;
        for (int v = 0; v < head_v_dim; v++) {
            float *row = S + (size_t)v * head_k_dim;
            float sk = bn_ssm_avx2_scale_dot_f32(head_k_dim, row, kh, decay);
            float delta = (vh[v] - sk) * beta;
            bn_ssm_avx2_mad_f32(head_k_dim, row, kh, delta);
            oh[v] = bn_ssm_avx2_dot_f32(head_k_dim, row, qh) * q_scale;
        }
    }
}

// Per-head RMSNorm + SiLU gate over V-head range [start, end)
void bn_transformer_ssm_gate_avx2_range(void *ctx, int start, int end) {
    BnSSMGateCtx *c = (BnSSMGateCtx *)ctx;
    int hd = c->head_v_dim;
    float eps = c->eps;

    for (int hv = start; hv < end; hv++) {
        float *oh = c->out + hv * hd;
        const float *zh = c->z + hv * hd;
        const float *nw = c->norm_w;

        // RMSNorm: vectorized sum-of-squares
        __m256 ss0 = _mm256_setzero_ps(), ss1 = _mm256_setzero_ps();
        for (int d = 0; d < hd; d += 16) {
            __m256 o0 = _mm256_loadu_ps(oh + d);
            __m256 o1 = _mm256_loadu_ps(oh + d + 8);
            ss0 = _mm256_fmadd_ps(o0, o0, ss0);
            ss1 = _mm256_fmadd_ps(o1, o1, ss1);
        }
        float ss = bn_ssm_avx2_hsum_ggml_ps(_mm256_add_ps(ss0, ss1));
        float scale_s = 1.0f / sqrtf(ss / hd + eps);
        for (int d = 0; d < hd; d++) {
            float g = zh[d];
            oh[d] = oh[d] * scale_s * nw[d] * (g / (1.0f + expf(-g)));
        }
    }
}

#endif // __AVX2__
