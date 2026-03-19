#include "transformer_internal.h"

#ifdef __AVX2__

// Conv1d + SiLU over channel range [start, end)
// Optimized: process 8 channels at a time using AVX2 + fast SiLU.
void bn_transformer_ssm_conv_silu_avx2_range(void *ctx, int start, int end) {
    BnSSMConvCtx *c = (BnSSMConvCtx *)ctx;
    float *qkv = c->qkv;
    float *conv_state = c->conv_state;
    const float *conv1d_w = c->conv1d_w;
    int qkv_dim = c->qkv_dim;
    int kern = c->kern;

    // Vectorized path: 8 channels at a time (kern=4 specialized)
    int ch = start;
    if (kern == 4) {
        for (; ch + 7 < end; ch += 8) {
            // Gather conv_state and weights for 8 channels
            // conv_state layout: [tap][qkv_dim], weights: [ch][kern]
            float *cs0 = conv_state + ch;
            float *cs1 = conv_state + (size_t)qkv_dim + ch;
            float *cs2 = conv_state + (size_t)2 * qkv_dim + ch;

            // Load conv_state for 3 taps (kern-1 = 3)
            __m256 s0 = _mm256_loadu_ps(cs0);
            __m256 s1 = _mm256_loadu_ps(cs1);
            __m256 s2 = _mm256_loadu_ps(cs2);
            __m256 cur = _mm256_loadu_ps(qkv + ch);

            // Load weights: conv1d_w[ch*4 + k] for 8 channels
            // Weights are contiguous per channel: [w0,w1,w2,w3] for each ch
            // We need to gather w[ch*4+0], w[ch*4+1], w[ch*4+2], w[ch*4+3]
            // Using scalar gathers is fine since weights are likely in L1
            float w0[8], w1[8], w2[8], w3[8];
            for (int i = 0; i < 8; i++) {
                const float *wp = conv1d_w + (size_t)(ch + i) * 4;
                w0[i] = wp[0]; w1[i] = wp[1]; w2[i] = wp[2]; w3[i] = wp[3];
            }

            __m256 sum = _mm256_mul_ps(s0, _mm256_loadu_ps(w0));
            sum = _mm256_fmadd_ps(s1, _mm256_loadu_ps(w1), sum);
            sum = _mm256_fmadd_ps(s2, _mm256_loadu_ps(w2), sum);
            sum = _mm256_fmadd_ps(cur, _mm256_loadu_ps(w3), sum);

            // Shift conv_state: cs0 = cs1, cs1 = cs2, cs2 = cur
            _mm256_storeu_ps(cs0, s1);
            _mm256_storeu_ps(cs1, s2);
            _mm256_storeu_ps(cs2, cur);

            // SiLU: sum * sigmoid(sum) using fast exp
            _mm256_storeu_ps(qkv + ch, bn_avx2_fast_silu_ps(sum));
        }
    }

    // Scalar fallback for remaining channels
    for (; ch < end; ch++) {
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

        // --- Pass 0: Decay state S *= decay ---
        __m256 vdecay = _mm256_set1_ps(decay);
        int n_state = head_k_dim * head_v_dim;
        for (int i = 0; i < n_state; i += 16) {
            _mm256_storeu_ps(S + i,     _mm256_mul_ps(_mm256_loadu_ps(S + i),     vdecay));
            _mm256_storeu_ps(S + i + 8, _mm256_mul_ps(_mm256_loadu_ps(S + i + 8), vdecay));
        }

        // --- Pass 1: sk = S @ k ---
        float sk[head_v_dim] __attribute__((aligned(32)));
        {
            __m256 kv = _mm256_set1_ps(kh[0]);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 16) {
                _mm256_storeu_ps(sk + v,     _mm256_mul_ps(_mm256_loadu_ps(row + v),     kv));
                _mm256_storeu_ps(sk + v + 8, _mm256_mul_ps(_mm256_loadu_ps(row + v + 8), kv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            __m256 kv = _mm256_set1_ps(kh[k]);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                _mm256_storeu_ps(sk + v,     _mm256_fmadd_ps(_mm256_loadu_ps(row + v),     kv, _mm256_loadu_ps(sk + v)));
                _mm256_storeu_ps(sk + v + 8, _mm256_fmadd_ps(_mm256_loadu_ps(row + v + 8), kv, _mm256_loadu_ps(sk + v + 8)));
            }
        }

        // --- Compute delta = beta * (v - sk) in-place over sk ---
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int v = 0; v < head_v_dim; v += 16) {
            _mm256_storeu_ps(sk + v,     _mm256_mul_ps(vbeta, _mm256_sub_ps(_mm256_loadu_ps(vh + v),     _mm256_loadu_ps(sk + v))));
            _mm256_storeu_ps(sk + v + 8, _mm256_mul_ps(vbeta, _mm256_sub_ps(_mm256_loadu_ps(vh + v + 8), _mm256_loadu_ps(sk + v + 8))));
        }

        // --- Pass 2: State update S[k][v] += kh[k] * delta[v] ---
        for (int k = 0; k < head_k_dim; k++) {
            __m256 kv = _mm256_set1_ps(kh[k]);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                _mm256_storeu_ps(row + v,     _mm256_fmadd_ps(kv, _mm256_loadu_ps(sk + v),     _mm256_loadu_ps(row + v)));
                _mm256_storeu_ps(row + v + 8, _mm256_fmadd_ps(kv, _mm256_loadu_ps(sk + v + 8), _mm256_loadu_ps(row + v + 8)));
            }
        }

        // --- Pass 3: Read output o = S^T @ q * q_scale ---
        float *oh = c->out + hv * head_v_dim;
        {
            __m256 qv = _mm256_set1_ps(qh[0] * q_scale);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 16) {
                _mm256_storeu_ps(oh + v,     _mm256_mul_ps(_mm256_loadu_ps(row + v),     qv));
                _mm256_storeu_ps(oh + v + 8, _mm256_mul_ps(_mm256_loadu_ps(row + v + 8), qv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            __m256 qv = _mm256_set1_ps(qh[k] * q_scale);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                _mm256_storeu_ps(oh + v,     _mm256_fmadd_ps(_mm256_loadu_ps(row + v),     qv, _mm256_loadu_ps(oh + v)));
                _mm256_storeu_ps(oh + v + 8, _mm256_fmadd_ps(_mm256_loadu_ps(row + v + 8), qv, _mm256_loadu_ps(oh + v + 8)));
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
        __m256 scale = _mm256_set1_ps(1.0f / sqrtf(ss / hd + eps));

        // Apply norm weight + SiLU gate (fast vectorized exp)
        for (int d = 0; d < hd; d += 8) {
            __m256 o = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(oh + d), scale), _mm256_loadu_ps(nw + d));
            __m256 g = _mm256_loadu_ps(zh + d);
            _mm256_storeu_ps(oh + d, _mm256_mul_ps(o, bn_avx2_fast_silu_ps(g)));
        }
    }
}

#endif // __AVX2__
