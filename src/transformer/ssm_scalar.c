#include "transformer_ssm_internal.h"

// Conv1d + SiLU over channel range [start, end)
void bn_transformer_ssm_conv_silu_scalar_range(void *ctx, int start, int end) {
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
        // Shift conv_state for this channel
        for (int k = 0; k < kern - 2; k++)
            conv_state[(size_t)k * qkv_dim + ch] =
                conv_state[(size_t)(k + 1) * qkv_dim + ch];
        conv_state[(size_t)(kern - 2) * qkv_dim + ch] = cur;
        // SiLU
        qkv[ch] = sum / (1.0f + expf(-sum));
    }
}

// L2 normalize Q and K per head, range over heads [start, end)
void bn_transformer_ssm_l2norm_scalar_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = (BnSSML2NormCtx *)ctx;
    int hd = c->head_dim;
    float eps = c->eps;

    for (int h = start; h < end; h++) {
        float *qh = c->q + h * hd;
        float *kh = c->k + h * hd;
        float qn = 0, kn = 0;
        for (int d = 0; d < hd; d++) {
            qn += qh[d] * qh[d];
            kn += kh[d] * kh[d];
        }
        qn = 1.0f / fmaxf(sqrtf(qn), eps);
        kn = 1.0f / fmaxf(sqrtf(kn), eps);
        for (int d = 0; d < hd; d++) {
            qh[d] *= qn;
            kh[d] *= kn;
        }
    }
}

// Delta rule recurrence over V-head range [start, end)
void bn_transformer_ssm_delta_scalar_range(void *ctx, int start, int end) {
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
        float delta[head_v_dim];
        for (int v = 0; v < head_v_dim; v++) {
            float *row = S + (size_t)v * head_k_dim;
            float sum = 0;
            for (int k = 0; k < head_k_dim; k++)
                row[k] *= decay;
            for (int k = 0; k < head_k_dim; k++)
                sum += row[k] * kh[k];
            delta[v] = beta * (vh[v] - sum);
        }

        // State update and read output in llama.cpp fused-GDN dot/mad order.
        float *oh = c->out + hv * head_v_dim;
        for (int v = 0; v < head_v_dim; v++) {
            float *row = S + (size_t)v * head_k_dim;
            float d = delta[v];
            for (int k = 0; k < head_k_dim; k++)
                row[k] += kh[k] * d;
            float sum = 0;
            for (int k = 0; k < head_k_dim; k++)
                sum += row[k] * qh[k];
            oh[v] = sum * q_scale;
        }
    }
}

// Per-head RMSNorm + SiLU gate over V-head range [start, end)
void bn_transformer_ssm_gate_scalar_range(void *ctx, int start, int end) {
    BnSSMGateCtx *c = (BnSSMGateCtx *)ctx;
    int hd = c->head_v_dim;

    for (int hv = start; hv < end; hv++) {
        float *oh = c->out + hv * hd;
        const float *zh = c->z + hv * hd;
        // RMSNorm
        float ss = 0;
        for (int d = 0; d < hd; d++) ss += oh[d] * oh[d];
        ss = 1.0f / sqrtf(ss / hd + c->eps);
        for (int d = 0; d < hd; d++) oh[d] *= ss * c->norm_w[d];
        // SiLU gate
        for (int d = 0; d < hd; d++) {
            float g = zh[d];
            oh[d] *= g / (1.0f + expf(-g));
        }
    }
}
