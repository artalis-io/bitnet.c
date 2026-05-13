#include "transformer_ssm_internal.h"

#ifdef __wasm_simd128__

// Conv1d + SiLU over channel range [start, end)
// Stays scalar — strided conv_state data, kern=4 taps.
void bn_transformer_ssm_conv_silu_wasm_range(void *ctx, int start, int end) {
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
void bn_transformer_ssm_l2norm_wasm_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = (BnSSML2NormCtx *)ctx;
    int hd = c->head_dim;

    for (int h = start; h < end; h++) {
        float *qh = c->q + h * hd;
        float *kh = c->k + h * hd;

        // Sum-of-squares with 2x unroll (128 dims = 32 vectors = 16 iters)
        v128_t qss0 = wasm_f32x4_splat(0), qss1 = wasm_f32x4_splat(0);
        v128_t kss0 = wasm_f32x4_splat(0), kss1 = wasm_f32x4_splat(0);
        for (int d = 0; d < hd; d += 8) {
            v128_t q0 = wasm_v128_load(qh + d);
            v128_t q1 = wasm_v128_load(qh + d + 4);
#ifdef __wasm_relaxed_simd__
            qss0 = wasm_f32x4_relaxed_madd(q0, q0, qss0);
            qss1 = wasm_f32x4_relaxed_madd(q1, q1, qss1);
#else
            qss0 = wasm_f32x4_add(qss0, wasm_f32x4_mul(q0, q0));
            qss1 = wasm_f32x4_add(qss1, wasm_f32x4_mul(q1, q1));
#endif
            v128_t k0 = wasm_v128_load(kh + d);
            v128_t k1 = wasm_v128_load(kh + d + 4);
#ifdef __wasm_relaxed_simd__
            kss0 = wasm_f32x4_relaxed_madd(k0, k0, kss0);
            kss1 = wasm_f32x4_relaxed_madd(k1, k1, kss1);
#else
            kss0 = wasm_f32x4_add(kss0, wasm_f32x4_mul(k0, k0));
            kss1 = wasm_f32x4_add(kss1, wasm_f32x4_mul(k1, k1));
#endif
        }
        float qn = bn_wasm_hsum_f32x4(wasm_f32x4_add(qss0, qss1));
        float kn = bn_wasm_hsum_f32x4(wasm_f32x4_add(kss0, kss1));
        v128_t qscale = wasm_f32x4_splat(1.0f / (sqrtf(qn) + 1e-6f));
        v128_t kscale = wasm_f32x4_splat(1.0f / (sqrtf(kn) + 1e-6f));
        for (int d = 0; d < hd; d += 4) {
            wasm_v128_store(qh + d, wasm_f32x4_mul(wasm_v128_load(qh + d), qscale));
            wasm_v128_store(kh + d, wasm_f32x4_mul(wasm_v128_load(kh + d), kscale));
        }
    }
}

// Delta rule recurrence over V-head range [start, end)
void bn_transformer_ssm_delta_wasm_range(void *ctx, int start, int end) {
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
        v128_t vdecay = wasm_f32x4_splat(decay);
        int n_state = head_k_dim * head_v_dim;
        for (int i = 0; i < n_state; i += 8) {
            wasm_v128_store(S + i,     wasm_f32x4_mul(wasm_v128_load(S + i),     vdecay));
            wasm_v128_store(S + i + 4, wasm_f32x4_mul(wasm_v128_load(S + i + 4), vdecay));
        }

        // --- Pass 1: sk = S @ k ---
        float sk[head_v_dim];
        {
            v128_t kv = wasm_f32x4_splat(kh[0]);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 8) {
                wasm_v128_store(sk + v,     wasm_f32x4_mul(wasm_v128_load(row + v),     kv));
                wasm_v128_store(sk + v + 4, wasm_f32x4_mul(wasm_v128_load(row + v + 4), kv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            v128_t kv = wasm_f32x4_splat(kh[k]);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 8) {
#ifdef __wasm_relaxed_simd__
                wasm_v128_store(sk + v,     wasm_f32x4_relaxed_madd(wasm_v128_load(row + v),     kv, wasm_v128_load(sk + v)));
                wasm_v128_store(sk + v + 4, wasm_f32x4_relaxed_madd(wasm_v128_load(row + v + 4), kv, wasm_v128_load(sk + v + 4)));
#else
                wasm_v128_store(sk + v,     wasm_f32x4_add(wasm_v128_load(sk + v),     wasm_f32x4_mul(wasm_v128_load(row + v),     kv)));
                wasm_v128_store(sk + v + 4, wasm_f32x4_add(wasm_v128_load(sk + v + 4), wasm_f32x4_mul(wasm_v128_load(row + v + 4), kv)));
#endif
            }
        }

        // --- Compute delta = beta * (v - sk) in-place over sk ---
        v128_t vbeta = wasm_f32x4_splat(beta);
        for (int v = 0; v < head_v_dim; v += 8) {
            wasm_v128_store(sk + v,     wasm_f32x4_mul(vbeta, wasm_f32x4_sub(wasm_v128_load(vh + v),     wasm_v128_load(sk + v))));
            wasm_v128_store(sk + v + 4, wasm_f32x4_mul(vbeta, wasm_f32x4_sub(wasm_v128_load(vh + v + 4), wasm_v128_load(sk + v + 4))));
        }

        // --- Pass 2: State update S[k][v] += kh[k] * delta[v] ---
        for (int k = 0; k < head_k_dim; k++) {
            v128_t kv = wasm_f32x4_splat(kh[k]);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 8) {
#ifdef __wasm_relaxed_simd__
                wasm_v128_store(row + v,     wasm_f32x4_relaxed_madd(kv, wasm_v128_load(sk + v),     wasm_v128_load(row + v)));
                wasm_v128_store(row + v + 4, wasm_f32x4_relaxed_madd(kv, wasm_v128_load(sk + v + 4), wasm_v128_load(row + v + 4)));
#else
                wasm_v128_store(row + v,     wasm_f32x4_add(wasm_v128_load(row + v),     wasm_f32x4_mul(kv, wasm_v128_load(sk + v))));
                wasm_v128_store(row + v + 4, wasm_f32x4_add(wasm_v128_load(row + v + 4), wasm_f32x4_mul(kv, wasm_v128_load(sk + v + 4))));
#endif
            }
        }

        // --- Pass 3: Read output o = S^T @ q * q_scale ---
        float *oh = c->out + hv * head_v_dim;
        {
            v128_t qv = wasm_f32x4_splat(qh[0] * q_scale);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 8) {
                wasm_v128_store(oh + v,     wasm_f32x4_mul(wasm_v128_load(row + v),     qv));
                wasm_v128_store(oh + v + 4, wasm_f32x4_mul(wasm_v128_load(row + v + 4), qv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            v128_t qv = wasm_f32x4_splat(qh[k] * q_scale);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 8) {
#ifdef __wasm_relaxed_simd__
                wasm_v128_store(oh + v,     wasm_f32x4_relaxed_madd(wasm_v128_load(row + v),     qv, wasm_v128_load(oh + v)));
                wasm_v128_store(oh + v + 4, wasm_f32x4_relaxed_madd(wasm_v128_load(row + v + 4), qv, wasm_v128_load(oh + v + 4)));
#else
                wasm_v128_store(oh + v,     wasm_f32x4_add(wasm_v128_load(oh + v),     wasm_f32x4_mul(wasm_v128_load(row + v),     qv)));
                wasm_v128_store(oh + v + 4, wasm_f32x4_add(wasm_v128_load(oh + v + 4), wasm_f32x4_mul(wasm_v128_load(row + v + 4), qv)));
#endif
            }
        }
    }
}

// Per-head RMSNorm + SiLU gate over V-head range [start, end)
void bn_transformer_ssm_gate_wasm_range(void *ctx, int start, int end) {
    BnSSMGateCtx *c = (BnSSMGateCtx *)ctx;
    int hd = c->head_v_dim;
    float eps = c->eps;

    for (int hv = start; hv < end; hv++) {
        float *oh = c->out + hv * hd;
        const float *zh = c->z + hv * hd;
        const float *nw = c->norm_w;

        // RMSNorm: vectorized sum-of-squares
        v128_t ss0 = wasm_f32x4_splat(0), ss1 = wasm_f32x4_splat(0);
        for (int d = 0; d < hd; d += 8) {
            v128_t o0 = wasm_v128_load(oh + d);
            v128_t o1 = wasm_v128_load(oh + d + 4);
#ifdef __wasm_relaxed_simd__
            ss0 = wasm_f32x4_relaxed_madd(o0, o0, ss0);
            ss1 = wasm_f32x4_relaxed_madd(o1, o1, ss1);
#else
            ss0 = wasm_f32x4_add(ss0, wasm_f32x4_mul(o0, o0));
            ss1 = wasm_f32x4_add(ss1, wasm_f32x4_mul(o1, o1));
#endif
        }
        float ss = bn_wasm_hsum_f32x4(wasm_f32x4_add(ss0, ss1));
        v128_t scale = wasm_f32x4_splat(1.0f / sqrtf(ss / hd + eps));

        // Apply norm weight + SiLU gate (per-element expf)
        for (int d = 0; d < hd; d += 4) {
            v128_t o = wasm_f32x4_mul(wasm_f32x4_mul(wasm_v128_load(oh + d), scale), wasm_v128_load(nw + d));
            float g0 = zh[d], g1 = zh[d+1], g2 = zh[d+2], g3 = zh[d+3];
            float sv[4] = {
                g0 / (1.0f + expf(-g0)),
                g1 / (1.0f + expf(-g1)),
                g2 / (1.0f + expf(-g2)),
                g3 / (1.0f + expf(-g3))
            };
            wasm_v128_store(oh + d, wasm_f32x4_mul(o, wasm_v128_load(sv)));
        }
    }
}

#endif // __wasm_simd128__
