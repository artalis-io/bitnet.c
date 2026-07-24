#include "transformer_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer/gpu_internal.h"
#include "model_internal.h"
#include "turboquant.h"
#include "moe.h"
#include "session.h"
#include "platform.h"
#include "sh_log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void rmsnorm_per_layer_slice(float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        x[i] *= ss * w[i];
}

static int prepare_per_layer_input_state(BnModel *m, BnSession *sess,
                                         int token) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
    int per_dim = bn_transformer_per_layer_embedding_dim(c);
    if (per_dim <= 0)
        return 0;
    float norm_eps = bn_transformer_norm_epsilon(c);
    int total = per_dim * c->n_layers;
    if (!s->per_layer_input || !w->per_layer_model_proj.data ||
        !w->per_layer_token_embd.data || !w->per_layer_proj_norm)
        return -1;

    bn_transformer_cpu_quant_matvec(s->per_layer_input,
                                    &w->per_layer_model_proj,
                                    s->x, s->x_q, bn_model_pool(m));

    float proj_scale = 1.0f / sqrtf((float)c->dim);
    for (int i = 0; i < total; i++)
        s->per_layer_input[i] *= proj_scale;
    for (int l = 0; l < c->n_layers; l++)
        rmsnorm_per_layer_slice(s->per_layer_input + (size_t)l * per_dim,
                                w->per_layer_proj_norm, per_dim, norm_eps);

    if (bn_model_dequant_qweight_row(&w->per_layer_token_embd,
                                     token, total, s->hb) != 0)
        return -1;
    float tok_scale = sqrtf((float)per_dim);
    float input_scale = 1.0f / sqrtf(2.0f);
    for (int i = 0; i < total; i++)
        s->per_layer_input[i] =
            (s->per_layer_input[i] + s->hb[i] * tok_scale) * input_scale;
    return 0;
}

// Per-layer timing instrumentation (compile with -DBN_BENCH_LAYERS)
#ifdef BN_BENCH_LAYERS
static double bl_rmsnorm_us, bl_matvec_qkv_us, bl_rope_us, bl_gqa_us;
static double bl_logits_us, bl_residual_us, bl_ffn_us;
static double bl_ssm_conv_us, bl_ssm_l2norm_us, bl_ssm_delta_us, bl_ssm_gate_us;
static double bl_sigmoid_gate_us;
static int bl_layer_count;

#define BL_START() double _bl_t0 = bn_platform_time_ms()
#define BL_ACC(var) do { (var) += (bn_platform_time_ms() - _bl_t0) * 1000.0; _bl_t0 = bn_platform_time_ms(); } while(0)

static void bl_print_reset(void) {
    if (bl_layer_count == 0) return;
    double total = bl_rmsnorm_us + bl_matvec_qkv_us + bl_rope_us + bl_gqa_us +
                   bl_logits_us + bl_residual_us + bl_ffn_us +
                   bl_ssm_conv_us + bl_ssm_l2norm_us + bl_ssm_delta_us +
                   bl_ssm_gate_us + bl_sigmoid_gate_us;
    fprintf(stderr, "\n=== Per-layer timing (%d layers) ===\n", bl_layer_count);
    fprintf(stderr, "  RMSNorm:       %8.0f us (%.1f%%)\n", bl_rmsnorm_us, 100*bl_rmsnorm_us/total);
    fprintf(stderr, "  Matvec QKV:    %8.0f us (%.1f%%)\n", bl_matvec_qkv_us, 100*bl_matvec_qkv_us/total);
    fprintf(stderr, "  RoPE:          %8.0f us (%.1f%%)\n", bl_rope_us, 100*bl_rope_us/total);
    fprintf(stderr, "  GQA:           %8.0f us (%.1f%%)\n", bl_gqa_us, 100*bl_gqa_us/total);
    fprintf(stderr, "  Sigmoid gate:  %8.0f us (%.1f%%)\n", bl_sigmoid_gate_us, 100*bl_sigmoid_gate_us/total);
    fprintf(stderr, "  FFN:           %8.0f us (%.1f%%)\n", bl_ffn_us, 100*bl_ffn_us/total);
    fprintf(stderr, "  Residual:      %8.0f us (%.1f%%)\n", bl_residual_us, 100*bl_residual_us/total);
    fprintf(stderr, "  SSM conv:      %8.0f us (%.1f%%)\n", bl_ssm_conv_us, 100*bl_ssm_conv_us/total);
    fprintf(stderr, "  SSM l2norm:    %8.0f us (%.1f%%)\n", bl_ssm_l2norm_us, 100*bl_ssm_l2norm_us/total);
    fprintf(stderr, "  SSM delta:     %8.0f us (%.1f%%)\n", bl_ssm_delta_us, 100*bl_ssm_delta_us/total);
    fprintf(stderr, "  SSM gate:      %8.0f us (%.1f%%)\n", bl_ssm_gate_us, 100*bl_ssm_gate_us/total);
    fprintf(stderr, "  Logits:        %8.0f us (%.1f%%)\n", bl_logits_us, 100*bl_logits_us/total);
    fprintf(stderr, "  TOTAL:         %8.0f us\n", total);
    bl_rmsnorm_us = bl_matvec_qkv_us = bl_rope_us = bl_gqa_us = 0;
    bl_logits_us = bl_residual_us = bl_ffn_us = 0;
    bl_ssm_conv_us = bl_ssm_l2norm_us = bl_ssm_delta_us = bl_ssm_gate_us = 0;
    bl_sigmoid_gate_us = 0;
    bl_layer_count = 0;
}
#else
#define BL_START() (void)0
#define BL_ACC(var) (void)0
#endif

// Max elements for stack VLAs (head_size, dim). Prevents stack overflow
// from malicious model configs. 8192 = 32KB of floats, well within stack.
#define BN_MAX_VLA_ELEMS 8192

// Embed + all layers (attention + FFN). Populates KV cache at `pos`.
// Leaves final activation in s->x. Returns 0 on success, -1 on error.
static int forward_layers(BnModel *m, BnSession *sess, int token, int pos) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int head_size = c->head_size;

    // Guard against stack overflow from VLAs sized by model config
    if (head_size > BN_MAX_VLA_ELEMS || c->dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dimensions too large for stack VLAs");
        return -1;
    }

    // #9: Validate token bounds
    if (token < 0 || token >= c->vocab_size) {
        SH_LOG_ERROR("Token out of range");
        return -1;
    }

    // #10: Validate pos bounds
    if (pos < 0) {
        SH_LOG_ERROR("Position out of range");
        return -1;
    }

    // Embed the token
    bn_model_embed_token(m, s->x, token);
    if (prepare_per_layer_input_state(m, sess, token) != 0) {
        SH_LOG_ERROR("Per-layer input preparation failed");
        return -1;
    }

    // Process each layer
    int cache_pos = pos % c->seq_len;
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &m->weights.layers[l];
        int layer_head_size = lw->attn.head_size > 0 ? lw->attn.head_size : head_size;
        int rope_dims = bn_transformer_rope_dims_for_head(c, layer_head_size);
        int half_rope = rope_dims / 2;
        float rope_cos[half_rope], rope_sin[half_rope];
        float theta = bn_transformer_rope_theta_for_head(c, layer_head_size);
        for (int i = 0; i < half_rope; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dims);
            if (bn_transformer_uses_per_layer_embedding(c) &&
                bn_transformer_rope_uses_base_frequency(c, layer_head_size) &&
                m->weights.rope_freqs) {
                if (bn_transformer_divides_rope_freqs(c, l))
                    freq /= m->weights.rope_freqs[i];
                else
                    freq *= m->weights.rope_freqs[i];
            }
            float angle = pos * freq;
            rope_cos[i] = cosf(angle);
            rope_sin[i] = sinf(angle);
        }
        if (bn_transformer_cpu_forward_layer(m, sess, l, pos, cache_pos, rope_dims,
                                         rope_cos, rope_sin) != 0)
            return -1;
    }

    return 0;
}

// Final RMSNorm + logits computation. Reads s->x, writes s->logits.
// Returns s->logits.
static float *forward_logits(BnModel *m, BnSession *sess) {
    BL_START();
    float *logits = bn_transformer_forward_logits(m, sess);
    BL_ACC(bl_logits_us);
#ifdef BN_BENCH_LAYERS
    bl_print_reset();
#endif
    return logits;
}

float *bn_transformer_forward(BnModel *m, BnSession *s, int token, int pos) {
    // Try GPU-resident forward pass first
    float *gpu_logits = bn_transformer_gpu_forward(m, s, token, pos);
    if (gpu_logits) {
        /* no-op */
        return gpu_logits;
    }

    // Fall back to CPU orchestration. Graph backend policy decides whether
    // per-matvec backend fallback remains useful after full-graph rejection.
    BnGPUBackend *gpu = bn_model_gpu(m);
    BnTransformerGPUMatvecFallbackPolicy matvec_fallback =
        bn_transformer_gpu_matvec_fallback_policy(m, gpu);
    int disable_gpu = matvec_fallback.disable_backend_matvec;
    if (disable_gpu)
        bn_model_set_gpu_disabled(m, 1);
    int rc = forward_layers(m, s, token, pos);
    float *logits = rc == 0 ? forward_logits(m, s) : NULL;
    if (disable_gpu)
        bn_model_set_gpu_disabled(m, 0);
    return logits;
}

int bn_transformer_forward_no_logits(BnModel *m, BnSession *s,
                                     int token, int pos) {
    float *gpu_state = bn_transformer_gpu_forward_no_logits(m, s, token, pos);
    if (gpu_state)
        return 0;

    BnGPUBackend *gpu = bn_model_gpu(m);
    BnTransformerGPUMatvecFallbackPolicy matvec_fallback =
        bn_transformer_gpu_matvec_fallback_policy(m, gpu);
    int disable_gpu = matvec_fallback.disable_backend_matvec;
    if (disable_gpu)
        bn_model_set_gpu_disabled(m, 1);
    int rc = forward_layers(m, s, token, pos);
    if (disable_gpu)
        bn_model_set_gpu_disabled(m, 0);
    return rc == 0 ? 0 : -1;
}
