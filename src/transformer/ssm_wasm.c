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
    float eps = c->eps;

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
        v128_t qscale = wasm_f32x4_splat(1.0f / fmaxf(sqrtf(qn), eps));
        v128_t kscale = wasm_f32x4_splat(1.0f / fmaxf(sqrtf(kn), eps));
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

        // State is transposed: S[v][k] stores the mathematical state[k][v].
        float *oh = c->out + hv * head_v_dim;
        for (int v = 0; v < head_v_dim; v++) {
            float *row = S + (size_t)v * head_k_dim;
            v128_t acc = wasm_f32x4_splat(0.0f);
            v128_t vdecay = wasm_f32x4_splat(decay);
            int k = 0;
            for (; k + 4 <= head_k_dim; k += 4) {
                v128_t r = wasm_f32x4_mul(wasm_v128_load(row + k), vdecay);
                wasm_v128_store(row + k, r);
#ifdef __wasm_relaxed_simd__
                acc = wasm_f32x4_relaxed_madd(r, wasm_v128_load(kh + k), acc);
#else
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(r, wasm_v128_load(kh + k)));
#endif
            }
            float sk = bn_wasm_hsum_f32x4(acc);
            for (; k < head_k_dim; k++) {
                row[k] *= decay;
                sk += row[k] * kh[k];
            }
            float delta = (vh[v] - sk) * beta;
            v128_t vdelta = wasm_f32x4_splat(delta);
            for (k = 0; k + 4 <= head_k_dim; k += 4) {
#ifdef __wasm_relaxed_simd__
                wasm_v128_store(row + k, wasm_f32x4_relaxed_madd(
                    wasm_v128_load(kh + k), vdelta, wasm_v128_load(row + k)));
#else
                wasm_v128_store(row + k, wasm_f32x4_add(
                    wasm_v128_load(row + k),
                    wasm_f32x4_mul(wasm_v128_load(kh + k), vdelta)));
#endif
            }
            for (; k < head_k_dim; k++)
                row[k] += kh[k] * delta;

            acc = wasm_f32x4_splat(0.0f);
            for (k = 0; k + 4 <= head_k_dim; k += 4) {
#ifdef __wasm_relaxed_simd__
                acc = wasm_f32x4_relaxed_madd(wasm_v128_load(row + k),
                                               wasm_v128_load(qh + k), acc);
#else
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(row + k),
                                                         wasm_v128_load(qh + k)));
#endif
            }
            float sum = bn_wasm_hsum_f32x4(acc);
            for (; k < head_k_dim; k++)
                sum += row[k] * qh[k];
            oh[v] = sum * q_scale;
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
