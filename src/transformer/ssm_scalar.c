#include "transformer_ssm_internal.h"
#include "transformer_math_internal.h"

static float ssm_silu_scalar(float x) {
    return x / (1.0f + bn_transformer_fast_exp_scalar(-x));
}

static float ssm_dot_scalar(const float *x, const float *y, int n) {
    float sum[4][4] = {{0}};
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        for (int group = 0; group < 4; group++)
            for (int lane = 0; lane < 4; lane++)
                sum[group][lane] = fmaf(x[i + 4 * group + lane],
                                        y[i + 4 * group + lane],
                                        sum[group][lane]);
    }
    float lane[4];
    for (int j = 0; j < 4; j++)
        lane[j] = (sum[0][j] + sum[2][j]) +
                  (sum[1][j] + sum[3][j]);
    float result = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    for (; i < n; i++)
        result += x[i] * y[i];
    return result;
}

static float ssm_sum_sq_scalar(const float *x, int n) {
    float sum[4][4] = {{0}};
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        for (int group = 0; group < 4; group++)
            for (int lane = 0; lane < 4; lane++) {
                float value = x[i + 4 * group + lane];
                sum[group][lane] = fmaf(value, value, sum[group][lane]);
            }
    }
    float lane[4];
    for (int j = 0; j < 4; j++)
        lane[j] = (sum[0][j] + sum[2][j]) +
                  (sum[1][j] + sum[3][j]);
    float result = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    for (; i < n; i++)
        result += x[i] * x[i];
    return result;
}

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
        qkv[ch] = ssm_silu_scalar(sum);
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
        float qn = ssm_sum_sq_scalar(qh, hd);
        float kn = ssm_sum_sq_scalar(kh, hd);
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
            for (int k = 0; k < head_k_dim; k++)
                row[k] *= decay;
            float sum = ssm_dot_scalar(row, kh, head_k_dim);
            delta[v] = beta * (vh[v] - sum);
        }

        // State update and read output in llama.cpp fused-GDN dot/mad order.
        float *oh = c->out + hv * head_v_dim;
        for (int v = 0; v < head_v_dim; v++) {
            float *row = S + (size_t)v * head_k_dim;
            float d = delta[v];
            for (int k = 0; k < head_k_dim; k++)
                row[k] = fmaf(kh[k], d, row[k]);
            float sum = ssm_dot_scalar(row, qh, head_k_dim);
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
        double sum = 0.0;
        for (int d = 0; d < hd; d++)
            sum += (double)(oh[d] * oh[d]);
        float mean = (float)(sum / hd);
        float scale = 1.0f / sqrtf(mean + c->eps);
        for (int d = 0; d < hd; d++)
            oh[d] = oh[d] * scale * c->norm_w[d];
        // SiLU gate
        for (int d = 0; d < hd; d++) {
            float g = zh[d];
            oh[d] *= ssm_silu_scalar(g);
        }
    }
}
