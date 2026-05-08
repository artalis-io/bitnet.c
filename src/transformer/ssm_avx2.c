#include "transformer_internal.h"

#ifdef __AVX2__

// Conv1d + SiLU over channel range [start, end).
// Keep scalar accumulation order here: recurrent Qwen3.5 layers are sensitive
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
        float qn = bn_avx2_hsum_ps(_mm256_add_ps(qss0, qss1));
        float kn = bn_avx2_hsum_ps(_mm256_add_ps(kss0, kss1));
        __m256 qscale = _mm256_set1_ps(1.0f / (sqrtf(qn) + 1e-6f));
        __m256 kscale = _mm256_set1_ps(1.0f / (sqrtf(kn) + 1e-6f));
        for (int d = 0; d < hd; d += 8) {
            _mm256_storeu_ps(qh + d, _mm256_mul_ps(_mm256_loadu_ps(qh + d), qscale));
            _mm256_storeu_ps(kh + d, _mm256_mul_ps(_mm256_loadu_ps(kh + d), kscale));
        }
    }
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

        // --- Pass 1: decay S in-place while computing sk = S @ k ---
        float sk[head_v_dim] __attribute__((aligned(32)));
        __m256 vdecay = _mm256_set1_ps(decay);
        {
            __m256 kv = _mm256_set1_ps(kh[0]);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 16) {
                __m256 r0 = _mm256_mul_ps(_mm256_loadu_ps(row + v),     vdecay);
                __m256 r1 = _mm256_mul_ps(_mm256_loadu_ps(row + v + 8), vdecay);
                _mm256_storeu_ps(row + v,     r0);
                _mm256_storeu_ps(row + v + 8, r1);
                _mm256_storeu_ps(sk + v,     _mm256_mul_ps(r0, kv));
                _mm256_storeu_ps(sk + v + 8, _mm256_mul_ps(r1, kv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            __m256 kv = _mm256_set1_ps(kh[k]);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                __m256 r0 = _mm256_mul_ps(_mm256_loadu_ps(row + v),     vdecay);
                __m256 r1 = _mm256_mul_ps(_mm256_loadu_ps(row + v + 8), vdecay);
                _mm256_storeu_ps(row + v,     r0);
                _mm256_storeu_ps(row + v + 8, r1);
                _mm256_storeu_ps(sk + v,     _mm256_fmadd_ps(r0, kv, _mm256_loadu_ps(sk + v)));
                _mm256_storeu_ps(sk + v + 8, _mm256_fmadd_ps(r1, kv, _mm256_loadu_ps(sk + v + 8)));
            }
        }

        // --- Compute delta = beta * (v - sk) in-place over sk ---
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int v = 0; v < head_v_dim; v += 16) {
            _mm256_storeu_ps(sk + v,     _mm256_mul_ps(vbeta, _mm256_sub_ps(_mm256_loadu_ps(vh + v),     _mm256_loadu_ps(sk + v))));
            _mm256_storeu_ps(sk + v + 8, _mm256_mul_ps(vbeta, _mm256_sub_ps(_mm256_loadu_ps(vh + v + 8), _mm256_loadu_ps(sk + v + 8))));
        }

        // --- Pass 2: update S and accumulate output o = S^T @ q * q_scale ---
        float *oh = c->out + hv * head_v_dim;
        {
            __m256 kv = _mm256_set1_ps(kh[0]);
            __m256 qv = _mm256_set1_ps(qh[0] * q_scale);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 16) {
                __m256 r0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(sk + v),     _mm256_loadu_ps(row + v));
                __m256 r1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(sk + v + 8), _mm256_loadu_ps(row + v + 8));
                _mm256_storeu_ps(row + v,     r0);
                _mm256_storeu_ps(row + v + 8, r1);
                _mm256_storeu_ps(oh + v,     _mm256_mul_ps(r0, qv));
                _mm256_storeu_ps(oh + v + 8, _mm256_mul_ps(r1, qv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            __m256 kv = _mm256_set1_ps(kh[k]);
            __m256 qv = _mm256_set1_ps(qh[k] * q_scale);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                __m256 r0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(sk + v),     _mm256_loadu_ps(row + v));
                __m256 r1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(sk + v + 8), _mm256_loadu_ps(row + v + 8));
                _mm256_storeu_ps(row + v,     r0);
                _mm256_storeu_ps(row + v + 8, r1);
                _mm256_storeu_ps(oh + v,     _mm256_fmadd_ps(r0, qv, _mm256_loadu_ps(oh + v)));
                _mm256_storeu_ps(oh + v + 8, _mm256_fmadd_ps(r1, qv, _mm256_loadu_ps(oh + v + 8)));
            }
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
        float ss = bn_avx2_hsum_ps(_mm256_add_ps(ss0, ss1));
        float scale_s = 1.0f / sqrtf(ss / hd + eps);
        for (int d = 0; d < hd; d++) {
            float g = zh[d];
            oh[d] = oh[d] * scale_s * nw[d] * (g / (1.0f + expf(-g)));
        }
    }
}

#endif // __AVX2__
