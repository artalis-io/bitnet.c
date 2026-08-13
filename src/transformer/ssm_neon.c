#include "transformer_ssm_internal.h"
#include "transformer_simd_internal.h"

#ifdef __ARM_NEON

static float ssm_dot_neon(const float *x, const float *y, int n) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(x + i),
                         vld1q_f32(y + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(x + i + 4),
                         vld1q_f32(y + i + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(x + i + 8),
                         vld1q_f32(y + i + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(x + i + 12),
                         vld1q_f32(y + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum2),
                                      vaddq_f32(sum1, sum3)));
    for (; i < n; i++)
        sum += x[i] * y[i];
    return sum;
}

// Conv1d + SiLU over channel range [start, end)
// Processes 4 channels at a time where possible.
void bn_transformer_ssm_conv_silu_neon_range(void *ctx, int start, int end) {
    BnSSMConvCtx *c = (BnSSMConvCtx *)ctx;
    float *qkv = c->qkv;
    float *conv_state = c->conv_state;
    const float *conv1d_w = c->conv1d_w;
    int qkv_dim = c->qkv_dim;
    int kern = c->kern;

    int ch = start;
    for (; ch + 3 < end; ch += 4) {
        float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int lane = 0; lane < 4; lane++) {
            int channel = ch + lane;
            for (int k = 0; k < kern - 1; k++)
                sums[lane] += conv_state[(size_t)k * qkv_dim + channel] *
                              conv1d_w[(size_t)channel * kern + k];
            float cur = qkv[channel];
            sums[lane] += cur * conv1d_w[(size_t)channel * kern + kern - 1];
            for (int k = 0; k < kern - 2; k++)
                conv_state[(size_t)k * qkv_dim + channel] =
                    conv_state[(size_t)(k + 1) * qkv_dim + channel];
            conv_state[(size_t)(kern - 2) * qkv_dim + channel] = cur;
        }
        vst1q_f32(qkv + ch, bn_neon_fast_silu_f32(vld1q_f32(sums)));
    }
    for (; ch < end; ch++) {
        float sum = 0.0f;
        for (int k = 0; k < kern - 1; k++)
            sum += conv_state[(size_t)k * qkv_dim + ch] *
                   conv1d_w[(size_t)ch * kern + k];
        float cur = qkv[ch];
        sum += cur * conv1d_w[(size_t)ch * kern + kern - 1];
        for (int k = 0; k < kern - 2; k++)
            conv_state[(size_t)k * qkv_dim + ch] =
                conv_state[(size_t)(k + 1) * qkv_dim + ch];
        conv_state[(size_t)(kern - 2) * qkv_dim + ch] = cur;
        qkv[ch] = vgetq_lane_f32(
            bn_neon_fast_silu_f32(vdupq_n_f32(sum)), 0);
    }
}

// L2 normalize Q and K per head, range over heads [start, end)
void bn_transformer_ssm_l2norm_neon_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = (BnSSML2NormCtx *)ctx;
    int hd = c->head_dim;
    float eps = c->eps;

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
        float qn = vaddvq_f32(vaddq_f32(vaddq_f32(qss0, qss2),
                                        vaddq_f32(qss1, qss3)));
        float kn = vaddvq_f32(vaddq_f32(vaddq_f32(kss0, kss2),
                                        vaddq_f32(kss1, kss3)));
        float32x4_t qscale = vdupq_n_f32(1.0f / fmaxf(sqrtf(qn), eps));
        float32x4_t kscale = vdupq_n_f32(1.0f / fmaxf(sqrtf(kn), eps));
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

        // State is transposed: S[v][k] stores the mathematical state[k][v].
        float *oh = c->out + hv * head_v_dim;
        for (int v = 0; v < head_v_dim; v++) {
            float *row = S + (size_t)v * head_k_dim;
            float32x4_t vdecay = vdupq_n_f32(decay);
            int k = 0;
            for (; k + 4 <= head_k_dim; k += 4) {
                float32x4_t r = vmulq_f32(vld1q_f32(row + k), vdecay);
                vst1q_f32(row + k, r);
            }
            for (; k < head_k_dim; k++) {
                row[k] *= decay;
            }
            float sk = ssm_dot_neon(row, kh, head_k_dim);
            float delta = (vh[v] - sk) * beta;
            float32x4_t vdelta = vdupq_n_f32(delta);
            for (k = 0; k + 4 <= head_k_dim; k += 4)
                vst1q_f32(row + k, vmlaq_f32(vld1q_f32(row + k),
                                             vld1q_f32(kh + k), vdelta));
            for (; k < head_k_dim; k++)
                row[k] += kh[k] * delta;

            float sum = ssm_dot_neon(row, qh, head_k_dim);
            oh[v] = sum * q_scale;
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

        double sum = 0.0;
        for (int d = 0; d < hd; d++)
            sum += (double)(oh[d] * oh[d]);
        float mean = (float)(sum / hd);
        float32x4_t scale =
            vdupq_n_f32(1.0f / sqrtf(mean + eps));

        // Apply norm weight + SiLU gate: oh = (oh * scale * nw) * silu(z)
        for (int d = 0; d < hd; d += 4) {
            float32x4_t o = vmulq_f32(vmulq_f32(vld1q_f32(oh + d), scale), vld1q_f32(nw + d));
            float32x4_t silu =
                bn_neon_fast_silu_f32(vld1q_f32(zh + d));
            vst1q_f32(oh + d, vmulq_f32(o, silu));
        }
    }
}

#endif // __ARM_NEON
