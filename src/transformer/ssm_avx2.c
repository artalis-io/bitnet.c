#include "transformer_ssm_internal.h"
#include "transformer_simd_internal.h"

#ifdef __AVX2__

// Sigmoid is a separate expf-based operation in the reference CPU graph;
// using the vector SiLU exponential approximation changes quantization inputs.
static void ssm_sigmoid_gate(float *out, const float *z, const float *weights,
                             float scale, int size) {
    for (int i = 0; i < size; i++) {
        float gate = 1.0f / (1.0f + expf(-z[i]));
        out[i] = out[i] * scale * weights[i] * gate;
    }
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
        float cur = qkv[ch];
        float sum = 0.0f;
        for (int k = 0; k < kern - 1; k++)
            sum = _mm_cvtss_f32(_mm_add_ss(
                _mm_set_ss(sum),
                _mm_mul_ss(
                    _mm_set_ss(conv_state[(size_t)k * qkv_dim + ch]),
                    _mm_set_ss(conv1d_w[(size_t)ch * kern + k]))));
        sum = _mm_cvtss_f32(_mm_add_ss(
            _mm_set_ss(sum),
            _mm_mul_ss(_mm_set_ss(cur),
                       _mm_set_ss(conv1d_w[(size_t)ch * kern +
                                           (kern - 1)]))));
        for (int k = 0; k < kern - 2; k++)
            conv_state[(size_t)k * qkv_dim + ch] =
                conv_state[(size_t)(k + 1) * qkv_dim + ch];
        conv_state[(size_t)(kern - 2) * qkv_dim + ch] = cur;
        qkv[ch] = sum;
    }

    int ch = start;
    for (; ch + 7 < end; ch += 8) {
        __m256 x = _mm256_loadu_ps(qkv + ch);
        __m256 exp_neg = bn_avx2_fast_exp_ps(
            _mm256_sub_ps(_mm256_setzero_ps(), x));
        _mm256_storeu_ps(qkv + ch, _mm256_div_ps(
            x, _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg)));
    }
    for (; ch < end; ch++)
        qkv[ch] = qkv[ch] / (1.0f + expf(-qkv[ch]));
}

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
void bn_transformer_ssm_conv_silu_avx512_range(void *ctx, int start, int end) {
    BnSSMConvCtx *c = (BnSSMConvCtx *)ctx;
    float *qkv = c->qkv;
    float *conv_state = c->conv_state;
    const float *conv1d_w = c->conv1d_w;
    int qkv_dim = c->qkv_dim;
    int kern = c->kern;
    for (int ch = start; ch < end; ch++) {
        __m128 sum = _mm_setzero_ps();
        for (int k = 0; k < kern - 1; k++) {
            sum = _mm_fmadd_ss(
                _mm_set_ss(conv_state[(size_t)k * qkv_dim + ch]),
                _mm_set_ss(conv1d_w[(size_t)ch * kern + k]), sum);
        }
        float cur = qkv[ch];
        sum = _mm_fmadd_ss(
            _mm_set_ss(cur),
            _mm_set_ss(conv1d_w[(size_t)ch * kern + (kern - 1)]), sum);
        for (int k = 0; k < kern - 2; k++)
            conv_state[(size_t)k * qkv_dim + ch] =
                conv_state[(size_t)(k + 1) * qkv_dim + ch];
        conv_state[(size_t)(kern - 2) * qkv_dim + ch] = cur;
        qkv[ch] = _mm_cvtss_f32(sum);
    }

    int ch = start;
    for (; ch + 15 < end; ch += 16) {
        __m512 x = _mm512_loadu_ps(qkv + ch);
        _mm512_storeu_ps(qkv + ch, bn_avx512_fast_silu_ps(x));
    }
    for (; ch < end; ch++)
        qkv[ch] = qkv[ch] / (1.0f + expf(-qkv[ch]));
}

void bn_transformer_ssm_conv_silu_x86_range(void *ctx, int start, int end) {
    bn_transformer_ssm_conv_silu_avx512_range(ctx, start, end);
}

#endif

// L2 normalize Q and K per head, range over heads [start, end)
void bn_transformer_ssm_l2norm_avx2_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = (BnSSML2NormCtx *)ctx;
    int hd = c->head_dim;
    float eps = c->eps;

    for (int h = start; h < end; h++) {
        float *qh = c->q + h * hd;
        float *kh = c->k + h * hd;

        double qn_sum = 0.0;
        double kn_sum = 0.0;
        for (int d = 0; d < hd; d++) {
            qn_sum += (double)(qh[d] * qh[d]);
            kn_sum += (double)(kh[d] * kh[d]);
        }
        __m256 qscale = _mm256_set1_ps(
            1.0f / fmaxf(sqrtf((float)qn_sum), eps));
        __m256 kscale = _mm256_set1_ps(
            1.0f / fmaxf(sqrtf((float)kn_sum), eps));
        for (int d = 0; d < hd; d += 8) {
            _mm256_storeu_ps(qh + d,
                             _mm256_mul_ps(_mm256_loadu_ps(qh + d), qscale));
            _mm256_storeu_ps(kh + d,
                             _mm256_mul_ps(_mm256_loadu_ps(kh + d), kscale));
        }
    }
}

static inline float bn_ssm_avx2_dot_f32(int n, const float *x, const float *y) {
#if defined(__AVX512F__)
    __m512 sumv[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                       _mm512_setzero_ps(), _mm512_setzero_ps() };
    int i = 0;
    int np = n & ~63;
    for (; i < np; i += 64) {
        for (int group = 0; group < 4; group++) {
            int off = i + group * 16;
            sumv[group] = _mm512_fmadd_ps(_mm512_loadu_ps(x + off),
                _mm512_loadu_ps(y + off), sumv[group]);
        }
    }
    sumv[0] = _mm512_add_ps(sumv[0], sumv[2]);
    sumv[1] = _mm512_add_ps(sumv[1], sumv[3]);
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(sumv[0], sumv[1]));
#else
    __m256 sumv[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                       _mm256_setzero_ps(), _mm256_setzero_ps() };
    int i = 0;
    int np = n & ~31;
    for (; i < np; i += 32) {
        for (int group = 0; group < 4; group++) {
            int off = i + group * 8;
            sumv[group] = _mm256_fmadd_ps(_mm256_loadu_ps(x + off),
                _mm256_loadu_ps(y + off), sumv[group]);
        }
    }
    sumv[0] = _mm256_add_ps(sumv[0], sumv[2]);
    sumv[1] = _mm256_add_ps(sumv[1], sumv[3]);
    sumv[0] = _mm256_add_ps(sumv[0], sumv[1]);
    __m128 halves = _mm_add_ps(_mm256_castps256_ps128(sumv[0]),
                               _mm256_extractf128_ps(sumv[0], 1));
    __m128 pairs = _mm_hadd_ps(halves, halves);
    float sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
#endif
    for (; i < n; i++)
        sum += x[i] * y[i];
    return sum;
}

static inline float bn_ssm_avx2_scale_dot_f32(int n, float *x,
                                              const float *y, float scale) {
#if defined(__AVX512F__)
    __m512 vscale = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(x + i,
                         _mm512_mul_ps(_mm512_loadu_ps(x + i), vscale));
#else
    __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i,
                         _mm256_mul_ps(_mm256_loadu_ps(x + i), vscale));
#endif
    for (; i < n; i++)
        x[i] *= scale;
    return bn_ssm_avx2_dot_f32(n, x, y);
}

static inline void bn_ssm_avx2_mad_f32(int n, float *y, const float *x,
                                       float v) {
#if defined(__AVX512F__)
    __m512 vv = _mm512_set1_ps(v);
    int i = 0;
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(
            _mm512_loadu_ps(x + i), vv, _mm512_loadu_ps(y + i)));
#else
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
#endif
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

        double ss = 0.0;
        for (int d = 0; d < hd; d++)
            ss += (double)(oh[d] * oh[d]);
        float scale_s = 1.0f / sqrtf((float)(ss / hd) + eps);
        if (c->sigmoid_gate) {
            ssm_sigmoid_gate(oh, zh, nw, scale_s, hd);
            continue;
        }
        __m256 scale = _mm256_set1_ps(scale_s);
        int d = 0;
        for (; d + 7 < hd; d += 8) {
            __m256 normed = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_loadu_ps(oh + d), scale),
                _mm256_loadu_ps(nw + d));
            __m256 z = _mm256_loadu_ps(zh + d);
            __m256 exp_neg = bn_avx2_fast_exp_ps(
                _mm256_sub_ps(_mm256_setzero_ps(), z));
            __m256 gate = _mm256_div_ps(
                z, _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg));
            _mm256_storeu_ps(oh + d, _mm256_mul_ps(normed, gate));
        }
        for (; d < hd; d++) {
            float g = zh[d];
            float sigmoid = 1.0f / (1.0f + expf(-g));
            oh[d] = oh[d] * scale_s * nw[d] *
                    (g * sigmoid);
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
void bn_transformer_ssm_gate_avx512_range(void *ctx, int start, int end) {
    BnSSMGateCtx *c = (BnSSMGateCtx *)ctx;
    int hd = c->head_v_dim;
    float eps = c->eps;

    for (int hv = start; hv < end; hv++) {
        float *oh = c->out + hv * hd;
        const float *zh = c->z + hv * hd;
        const float *nw = c->norm_w;

        double ss = 0.0;
        for (int d = 0; d < hd; d++)
            ss += (double)(oh[d] * oh[d]);
        float scale_s = 1.0f / sqrtf((float)(ss / hd) + eps);
        if (c->sigmoid_gate) {
            ssm_sigmoid_gate(oh, zh, nw, scale_s, hd);
            continue;
        }
        __m512 scale = _mm512_set1_ps(scale_s);
        int d = 0;
        for (; d + 15 < hd; d += 16) {
            __m512 normed = _mm512_mul_ps(
                _mm512_mul_ps(_mm512_loadu_ps(oh + d), scale),
                _mm512_loadu_ps(nw + d));
            __m512 z = _mm512_loadu_ps(zh + d);
            __m512 gate = bn_avx512_fast_silu_ps(z);
            _mm512_storeu_ps(oh + d, _mm512_mul_ps(normed, gate));
        }
        for (; d < hd; d++) {
            float g = zh[d];
            float sigmoid = 1.0f / (1.0f + expf(-g));
            oh[d] = oh[d] * scale_s * nw[d] *
                    (g * sigmoid);
        }
    }
}

void bn_transformer_ssm_gate_x86_range(void *ctx, int start, int end) {
    bn_transformer_ssm_gate_avx512_range(ctx, start, end);
}
#endif

#endif // __AVX2__
