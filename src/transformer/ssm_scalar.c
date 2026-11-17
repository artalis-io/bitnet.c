#include "transformer_ssm_internal.h"
#include "transformer_math_internal.h"
#include <stdint.h>
#include <stdio.h>

int bn_transformer_ssm_kda_coefficients(const BnSSMKDACoeffCtx *c,
    int start, int end) {
    if (!c || c->num_heads < 1 || c->head_k_dim < 1 || c->head_k_dim > 256 ||
        start < 0 || end < start || end > c->num_heads ||
        !isfinite(c->lower_bound) || c->lower_bound >= 0.0f ||
        (c->parameter_storage != BN_SSM_KDA_A_LOG &&
         c->parameter_storage != BN_SSM_KDA_A_NEGATIVE_EXP) ||
        (size_t)c->num_heads > SIZE_MAX / sizeof(float) / (size_t)c->head_k_dim) {
        fprintf(stderr, "KDA coefficients: invalid dimensions or range\n");
        return -1;
    }
    if (start == end) return 0;
    if (!c->log_decay || !c->beta || !c->gate || !c->bias ||
        !c->decay_parameter || !c->beta_raw) {
        fprintf(stderr, "KDA coefficients: missing buffers\n");
        return -1;
    }
    /* Preflight the entire range, including later heads, so an invalid input
     * cannot leave partially generated coefficients (also for in-place use). */
    for (int h = start; h < end; h++) {
        if (!isfinite(c->decay_parameter[h]) || !isfinite(c->beta_raw[h]))
            goto invalid_data;
        if (c->parameter_storage == BN_SSM_KDA_A_LOG) {
            if (!isfinite(expf(c->decay_parameter[h]))) goto invalid_data;
        } else if (c->decay_parameter[h] > 0.0f) goto invalid_data;
        for (int k = 0; k < c->head_k_dim; k++) {
            size_t i = (size_t)h * c->head_k_dim + k;
            if (!isfinite(c->gate[i]) || !isfinite(c->bias[i]) ||
                !isfinite(c->gate[i] + c->bias[i])) goto invalid_data;
        }
    }
    for (int h = start; h < end; h++) {
        float amplitude = c->parameter_storage == BN_SSM_KDA_A_LOG
            ? expf(c->decay_parameter[h]) : -c->decay_parameter[h];
        c->beta[h] = 1.0f / (1.0f + expf(-c->beta_raw[h]));
        for (int k = 0; k < c->head_k_dim; k++) {
            size_t i = (size_t)h * c->head_k_dim + k;
            float shifted = c->gate[i] + c->bias[i];
            float scaled = amplitude * shifted;
            c->log_decay[i] = c->lower_bound / (1.0f + expf(-scaled));
        }
    }
    return 0;
invalid_data:
    fprintf(stderr, "KDA coefficients: invalid input values\n");
    return -1;
}

int bn_transformer_ssm_kda_step(const BnSSMKDACtx *c, int start, int end) {
    if (!c || c->num_heads < 1 || c->head_k_dim < 1 || c->head_k_dim > 256 ||
        c->head_v_dim < 1 || c->head_v_dim > 256 || start < 0 || end < start ||
        end > c->num_heads || !isfinite(c->q_scale) || c->q_scale <= 0.0f ||
        (size_t)c->num_heads > SIZE_MAX / sizeof(float) /
            (size_t)c->head_k_dim / (size_t)c->head_v_dim) {
        fprintf(stderr, "KDA state step: invalid dimensions or range\n");
        return -1;
    }
    if (start == end) return 0;
    if (!c->state || !c->out || !c->q || !c->k || !c->v ||
        !c->log_decay || !c->beta) {
        fprintf(stderr, "KDA state step: missing buffers\n");
        return -1;
    }
    for (int h = start; h < end; h++) {
        size_t key_offset = (size_t)h * c->head_k_dim;
        size_t value_offset = (size_t)h * c->head_v_dim;
        float decay[256], query[256];
        for (int k = 0; k < c->head_k_dim; k++) {
            decay[k] = expf(c->log_decay[key_offset + k]);
            query[k] = c->q[key_offset + k] * c->q_scale;
        }
        for (int v = 0; v < c->head_v_dim; v++) {
            float *row = c->state +
                (value_offset + (size_t)v) * c->head_k_dim;
            float prediction = 0.0f;
            for (int k = 0; k < c->head_k_dim; k++) {
                row[k] *= decay[k];
                prediction = fmaf(row[k], c->k[key_offset + k], prediction);
            }
            float delta = (c->v[value_offset + v] - prediction) * c->beta[h];
            float output = 0.0f;
            for (int k = 0; k < c->head_k_dim; k++) {
                row[k] = fmaf(delta, c->k[key_offset + k], row[k]);
                output = fmaf(row[k], query[k], output);
            }
            c->out[value_offset + v] = output;
        }
    }
    return 0;
}

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
void bn_transformer_ssm_qk_rmsnorm_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = ctx;
    int hd = c->head_dim;
    if (hd <= 0) return;
    float head_scale = 1.0f / sqrtf((float)hd);
    float eps = c->eps / (float)hd;
    for (int h = start; h < end; h++) {
        float *vectors[] = {c->q + (size_t)h * hd, c->k + (size_t)h * hd};
        for (int v = 0; v < 2; v++) {
            float *x = vectors[v];
            double sum = 0.0;
            for (int d = 0; d < hd; d++) sum += (double)(x[d] * x[d]);
            float scale = 1.0f / sqrtf((float)(sum / hd) + eps);
            for (int d = 0; d < hd; d++) {
                float normalized = x[d] * scale;
                x[d] = normalized * head_scale;
            }
        }
    }
}

void bn_transformer_ssm_qk_sum_eps_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = ctx;
    if (!c || c->head_dim < 1 || c->head_dim > BN_MAX_VLA_ELEMS ||
        !isfinite(c->eps) || c->eps <= 0 || !c->q || !c->k ||
        start < 0 || end < start) return;
    int padded = 1;
    while (padded < c->head_dim) {
        if (padded > BN_MAX_VLA_ELEMS / 2) return;
        padded *= 2;
    }
    float squares[padded];
    for (int h = start; h < end; h++) {
        float *vectors[] = {c->q + (size_t)h * c->head_dim,
                            c->k + (size_t)h * c->head_dim};
        for (int v = 0; v < 2; v++) {
            float *x = vectors[v];
            for (int d = 0; d < padded; d++)
                squares[d] = d < c->head_dim ? x[d] * x[d] : 0.0f;
            for (int width = padded / 2; width; width /= 2)
                for (int d = 0; d < width; d++) squares[d] += squares[d + width];
            float denominator = sqrtf(squares[0] + c->eps);
            for (int d = 0; d < c->head_dim; d++) x[d] /= denominator;
        }
    }
}

void bn_transformer_ssm_l2norm_scalar_range(void *ctx, int start, int end) {
    BnSSML2NormCtx *c = (BnSSML2NormCtx *)ctx;
    if (c->mode == BN_SSM_QK_NORM_L2_SUM_EPS) {
        bn_transformer_ssm_qk_sum_eps_range(ctx, start, end);
        return;
    }
    if (c->mode == BN_SSM_QK_NORM_RMS_SCALED) {
        bn_transformer_ssm_qk_rmsnorm_range(ctx, start, end);
        return;
    }
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
        // Architecture-selected recurrent output gate.
        for (int d = 0; d < hd; d++) {
            float g = zh[d];
            float sigmoid = 1.0f / (1.0f + bn_transformer_fast_exp_scalar(-g));
            oh[d] *= c->sigmoid_gate ? sigmoid : g * sigmoid;
        }
    }
}
