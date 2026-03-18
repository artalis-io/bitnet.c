#include "transformer_internal.h"

#ifdef __ARM_NEON

// Conv1d + SiLU over channel range [start, end)
// Processes 4 channels at a time where possible.
void bn_transformer_ssm_conv_silu_neon_range(void *ctx, int start, int end) {
    BnSSMConvCtx *c = (BnSSMConvCtx *)ctx;
    float *qkv = c->qkv;
    float *conv_state = c->conv_state;
    const float *conv1d_w = c->conv1d_w;
    int qkv_dim = c->qkv_dim;
    int kern = c->kern;

    // Process channels. Conv state is strided (k * qkv_dim + ch), so we
    // process one channel at a time but vectorize the kern=4 dot product.
    for (int ch = start; ch < end; ch++) {
        // Load weights for this channel: w[ch*kern+0..kern-1]
        float sum = 0;
        for (int k = 0; k < kern - 1; k++)
            sum += conv_state[(size_t)k * qkv_dim + ch] *
                   conv1d_w[(size_t)ch * kern + k];
        float cur = qkv[ch];
        sum += cur * conv1d_w[(size_t)ch * kern + (kern - 1)];
        // Shift conv_state
        for (int k = 0; k < kern - 2; k++)
            conv_state[(size_t)k * qkv_dim + ch] =
                conv_state[(size_t)(k + 1) * qkv_dim + ch];
        conv_state[(size_t)(kern - 2) * qkv_dim + ch] = cur;
        // SiLU: x * sigmoid(x)
        qkv[ch] = sum / (1.0f + expf(-sum));
    }
}

// L2 normalize Q and K per head, range over heads [start, end)
void bn_transformer_ssm_l2norm_neon_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = (BnSSML2NormCtx *)ctx;
    int hd = c->head_dim;

    for (int h = start; h < end; h++) {
        float *qh = c->q + h * hd;
        float *kh = c->k + h * hd;

        // Vectorized sum-of-squares with 4x unroll
        float32x4_t qss0 = vdupq_n_f32(0), qss1 = vdupq_n_f32(0);
        float32x4_t qss2 = vdupq_n_f32(0), qss3 = vdupq_n_f32(0);
        float32x4_t kss0 = vdupq_n_f32(0), kss1 = vdupq_n_f32(0);
        float32x4_t kss2 = vdupq_n_f32(0), kss3 = vdupq_n_f32(0);
        for (int d = 0; d < hd; d += 16) {
            float32x4_t q0 = vld1q_f32(qh + d);
            float32x4_t q1 = vld1q_f32(qh + d + 4);
            float32x4_t q2 = vld1q_f32(qh + d + 8);
            float32x4_t q3 = vld1q_f32(qh + d + 12);
            qss0 = vmlaq_f32(qss0, q0, q0);
            qss1 = vmlaq_f32(qss1, q1, q1);
            qss2 = vmlaq_f32(qss2, q2, q2);
            qss3 = vmlaq_f32(qss3, q3, q3);
            float32x4_t k0 = vld1q_f32(kh + d);
            float32x4_t k1 = vld1q_f32(kh + d + 4);
            float32x4_t k2 = vld1q_f32(kh + d + 8);
            float32x4_t k3 = vld1q_f32(kh + d + 12);
            kss0 = vmlaq_f32(kss0, k0, k0);
            kss1 = vmlaq_f32(kss1, k1, k1);
            kss2 = vmlaq_f32(kss2, k2, k2);
            kss3 = vmlaq_f32(kss3, k3, k3);
        }
        float qn = bn_transformer_neon_hsum_f32(vaddq_f32(vaddq_f32(qss0, qss1), vaddq_f32(qss2, qss3)));
        float kn = bn_transformer_neon_hsum_f32(vaddq_f32(vaddq_f32(kss0, kss1), vaddq_f32(kss2, kss3)));
        float32x4_t qscale = vdupq_n_f32(1.0f / (sqrtf(qn) + 1e-6f));
        float32x4_t kscale = vdupq_n_f32(1.0f / (sqrtf(kn) + 1e-6f));
        for (int d = 0; d < hd; d += 4) {
            vst1q_f32(qh + d, vmulq_f32(vld1q_f32(qh + d), qscale));
            vst1q_f32(kh + d, vmulq_f32(vld1q_f32(kh + d), kscale));
        }
    }
}

// Delta rule recurrence over V-head range [start, end)
void bn_transformer_ssm_delta_neon_range(void *ctx, int start, int end) {
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
        float32x4_t vdecay = vdupq_n_f32(decay);
        int n_state = head_k_dim * head_v_dim;
        for (int i = 0; i < n_state; i += 16) {
            vst1q_f32(S + i,      vmulq_f32(vld1q_f32(S + i),      vdecay));
            vst1q_f32(S + i + 4,  vmulq_f32(vld1q_f32(S + i + 4),  vdecay));
            vst1q_f32(S + i + 8,  vmulq_f32(vld1q_f32(S + i + 8),  vdecay));
            vst1q_f32(S + i + 12, vmulq_f32(vld1q_f32(S + i + 12), vdecay));
        }

        // --- Pass 1: sk = S @ k ---
        // sk[v] = sum_k S[k*head_v_dim + v] * kh[k]
        // Accumulate rows of S weighted by kh[k]
        float sk[head_v_dim] __attribute__((aligned(16)));
        {
            float32x4_t kv = vdupq_n_f32(kh[0]);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 16) {
                vst1q_f32(sk + v,      vmulq_f32(vld1q_f32(row + v),      kv));
                vst1q_f32(sk + v + 4,  vmulq_f32(vld1q_f32(row + v + 4),  kv));
                vst1q_f32(sk + v + 8,  vmulq_f32(vld1q_f32(row + v + 8),  kv));
                vst1q_f32(sk + v + 12, vmulq_f32(vld1q_f32(row + v + 12), kv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            float32x4_t kv = vdupq_n_f32(kh[k]);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                vst1q_f32(sk + v,      vmlaq_f32(vld1q_f32(sk + v),      vld1q_f32(row + v),      kv));
                vst1q_f32(sk + v + 4,  vmlaq_f32(vld1q_f32(sk + v + 4),  vld1q_f32(row + v + 4),  kv));
                vst1q_f32(sk + v + 8,  vmlaq_f32(vld1q_f32(sk + v + 8),  vld1q_f32(row + v + 8),  kv));
                vst1q_f32(sk + v + 12, vmlaq_f32(vld1q_f32(sk + v + 12), vld1q_f32(row + v + 12), kv));
            }
        }

        // --- Compute delta = beta * (v - sk) in-place over sk ---
        float32x4_t vbeta = vdupq_n_f32(beta);
        for (int v = 0; v < head_v_dim; v += 16) {
            vst1q_f32(sk + v,      vmulq_f32(vbeta, vsubq_f32(vld1q_f32(vh + v),      vld1q_f32(sk + v))));
            vst1q_f32(sk + v + 4,  vmulq_f32(vbeta, vsubq_f32(vld1q_f32(vh + v + 4),  vld1q_f32(sk + v + 4))));
            vst1q_f32(sk + v + 8,  vmulq_f32(vbeta, vsubq_f32(vld1q_f32(vh + v + 8),  vld1q_f32(sk + v + 8))));
            vst1q_f32(sk + v + 12, vmulq_f32(vbeta, vsubq_f32(vld1q_f32(vh + v + 12), vld1q_f32(sk + v + 12))));
        }

        // --- Pass 2: State update S[k][v] += kh[k] * delta[v] ---
        for (int k = 0; k < head_k_dim; k++) {
            float32x4_t kv = vdupq_n_f32(kh[k]);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                vst1q_f32(row + v,      vmlaq_f32(vld1q_f32(row + v),      kv, vld1q_f32(sk + v)));
                vst1q_f32(row + v + 4,  vmlaq_f32(vld1q_f32(row + v + 4),  kv, vld1q_f32(sk + v + 4)));
                vst1q_f32(row + v + 8,  vmlaq_f32(vld1q_f32(row + v + 8),  kv, vld1q_f32(sk + v + 8)));
                vst1q_f32(row + v + 12, vmlaq_f32(vld1q_f32(row + v + 12), kv, vld1q_f32(sk + v + 12)));
            }
        }

        // --- Pass 3: Read output o = S^T @ q * q_scale ---
        // o[v] = sum_k S[k*head_v_dim + v] * qh[k]
        float *oh = c->out + hv * head_v_dim;
        {
            float32x4_t qv = vdupq_n_f32(qh[0] * q_scale);
            float *row = S;
            for (int v = 0; v < head_v_dim; v += 16) {
                vst1q_f32(oh + v,      vmulq_f32(vld1q_f32(row + v),      qv));
                vst1q_f32(oh + v + 4,  vmulq_f32(vld1q_f32(row + v + 4),  qv));
                vst1q_f32(oh + v + 8,  vmulq_f32(vld1q_f32(row + v + 8),  qv));
                vst1q_f32(oh + v + 12, vmulq_f32(vld1q_f32(row + v + 12), qv));
            }
        }
        for (int k = 1; k < head_k_dim; k++) {
            float32x4_t qv = vdupq_n_f32(qh[k] * q_scale);
            float *row = S + (size_t)k * head_v_dim;
            for (int v = 0; v < head_v_dim; v += 16) {
                vst1q_f32(oh + v,      vmlaq_f32(vld1q_f32(oh + v),      vld1q_f32(row + v),      qv));
                vst1q_f32(oh + v + 4,  vmlaq_f32(vld1q_f32(oh + v + 4),  vld1q_f32(row + v + 4),  qv));
                vst1q_f32(oh + v + 8,  vmlaq_f32(vld1q_f32(oh + v + 8),  vld1q_f32(row + v + 8),  qv));
                vst1q_f32(oh + v + 12, vmlaq_f32(vld1q_f32(oh + v + 12), vld1q_f32(row + v + 12), qv));
            }
        }
    }
}

// Per-head RMSNorm + SiLU gate over V-head range [start, end)
void bn_transformer_ssm_gate_neon_range(void *ctx, int start, int end) {
    BnSSMGateCtx *c = (BnSSMGateCtx *)ctx;
    int hd = c->head_v_dim;
    float eps = c->eps;

    for (int hv = start; hv < end; hv++) {
        float *oh = c->out + hv * hd;
        const float *zh = c->z + hv * hd;
        const float *nw = c->norm_w;

        // RMSNorm: vectorized sum-of-squares
        float32x4_t ss0 = vdupq_n_f32(0), ss1 = vdupq_n_f32(0);
        float32x4_t ss2 = vdupq_n_f32(0), ss3 = vdupq_n_f32(0);
        for (int d = 0; d < hd; d += 16) {
            float32x4_t o0 = vld1q_f32(oh + d);
            float32x4_t o1 = vld1q_f32(oh + d + 4);
            float32x4_t o2 = vld1q_f32(oh + d + 8);
            float32x4_t o3 = vld1q_f32(oh + d + 12);
            ss0 = vmlaq_f32(ss0, o0, o0);
            ss1 = vmlaq_f32(ss1, o1, o1);
            ss2 = vmlaq_f32(ss2, o2, o2);
            ss3 = vmlaq_f32(ss3, o3, o3);
        }
        float ss = bn_transformer_neon_hsum_f32(vaddq_f32(vaddq_f32(ss0, ss1), vaddq_f32(ss2, ss3)));
        float32x4_t scale = vdupq_n_f32(1.0f / sqrtf(ss / hd + eps));

        // Apply norm weight + SiLU gate: oh = (oh * scale * nw) * silu(z)
        for (int d = 0; d < hd; d += 4) {
            float32x4_t o = vmulq_f32(vmulq_f32(vld1q_f32(oh + d), scale), vld1q_f32(nw + d));
            // SiLU(z) = z * sigmoid(z). Per-element expf needed.
            float g0 = zh[d], g1 = zh[d+1], g2 = zh[d+2], g3 = zh[d+3];
            float32x4_t silu;
            float sv[4] = {
                g0 / (1.0f + expf(-g0)),
                g1 / (1.0f + expf(-g1)),
                g2 / (1.0f + expf(-g2)),
                g3 / (1.0f + expf(-g3))
            };
            silu = vld1q_f32(sv);
            vst1q_f32(oh + d, vmulq_f32(o, silu));
        }
    }
}

#endif // __ARM_NEON
