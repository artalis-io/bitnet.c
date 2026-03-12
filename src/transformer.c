#include "transformer.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// --- Helper functions ---

static void rmsnorm(float *out, const float *x, const float *w, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void rope(float *vec, int dim, int head_size, int pos, float theta) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(theta, (float)head_dim / (float)head_size);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

// --- Forward pass ---

float *transformer_forward(Model *m, int token, int pos) {
    Config *c = &m->config;
    Weights *w = &m->weights;
    RunState *s = &m->state;
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul;
    int head_size = c->head_size;

    // Embed the token
    model_embed_token(m, s->x, token);

    // Process each layer
    for (int l = 0; l < c->n_layers; l++) {
        LayerWeights *lw = &w->layers[l];

        // ---- Attention block ----

        // RMSNorm before attention
        rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);

        // QKV projections (ternary matmul)
        ternary_matvec(s->q, &lw->wq, s->xb);    // q = Wq @ xb
        // KV go directly into cache
        size_t loff = (size_t)l * c->seq_len * kv_dim;
        float *key_cache_row   = s->key_cache   + loff + (size_t)pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + (size_t)pos * kv_dim;

        ternary_matvec(key_cache_row,   &lw->wk, s->xb);  // k = Wk @ xb
        ternary_matvec(value_cache_row, &lw->wv, s->xb);  // v = Wv @ xb

        // RoPE on q and k
        rope(s->q, dim, head_size, pos, c->rope_theta);
        rope(key_cache_row, kv_dim, head_size, pos, c->rope_theta);

        // Grouped Query Attention (GQA)
        for (int h = 0; h < c->n_heads; h++) {
            float *q_h = s->q + h * head_size;
            float *att = s->att + h * c->seq_len;
            int kv_h = h / kv_mul;  // which KV head this query head attends to

            // Attention scores: q · k for all positions up to pos
            for (int t = 0; t <= pos; t++) {
                float *k_t = s->key_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                float score = 0.0f;
                for (int d = 0; d < head_size; d++) {
                    score += q_h[d] * k_t[d];
                }
                att[t] = score / sqrtf((float)head_size);
            }

            // Softmax over attention scores
            softmax(att, pos + 1);

            // Weighted sum of values
            float *xb_h = s->xb + h * head_size;
            memset(xb_h, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v_t = s->value_cache + loff + (size_t)t * kv_dim + kv_h * head_size;
                float a = att[t];
                for (int d = 0; d < head_size; d++) {
                    xb_h[d] += a * v_t[d];
                }
            }
        }

        // Attention sub-norm (BitNet-specific)
        if (lw->attn_sub_norm) {
            rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);
        }

        // Output projection: Wo @ xb → xb2
        ternary_matvec(s->xb2, &lw->wo, s->xb);

        // Residual connection
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        // ---- FFN block ----

        // RMSNorm before FFN
        rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

        if (c->has_ffn_gate) {
            // SwiGLU / Gated: gate * activation(up)
            ternary_matvec(s->hb,  &lw->ffn_gate, s->xb);  // gate
            ternary_matvec(s->hb2, &lw->ffn_up,   s->xb);  // up

            if (c->act_type == 1) {
                // ReLU²: relu(x)² for gate, relu(x)² for up... actually:
                // BitNet uses: relu²(gate) * up  or  gate * relu²(up)?
                // Standard SwiGLU: silu(gate) * up
                // BitNet b1.58 with ReLU²: relu²(gate) * up
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i] > 0 ? s->hb[i] : 0;  // ReLU
                    s->hb[i] = g * g * s->hb2[i];            // ReLU² * up
                }
            } else {
                // SiLU (SwiGLU): silu(gate) * up
                for (int i = 0; i < hidden_dim; i++) {
                    float g = s->hb[i];
                    s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
                }
            }
        } else {
            // No gate: just up + activation
            ternary_matvec(s->hb, &lw->ffn_up, s->xb);
            if (c->act_type == 1) {
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i] > 0 ? s->hb[i] : 0;
                    s->hb[i] = v * v;
                }
            } else {
                for (int i = 0; i < hidden_dim; i++) {
                    float v = s->hb[i];
                    s->hb[i] = v / (1.0f + expf(-v));
                }
            }
        }

        // FFN sub-norm (BitNet-specific)
        if (lw->ffn_sub_norm) {
            rmsnorm(s->hb, s->hb, lw->ffn_sub_norm, hidden_dim, c->norm_eps);
        }

        // Down projection
        ternary_matvec(s->xb, &lw->ffn_down, s->hb);

        // Residual connection
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];

        #ifdef DEBUG
        if (l == 0 && pos == 0) {
            fprintf(stderr, "debug: layer 0 pos 0 x[0..3] = %.6f %.6f %.6f %.6f\n",
                    s->x[0], s->x[1], s->x[2], s->x[3]);
        }
        #endif
    }

    // Final RMSNorm
    rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);

    // Tied embeddings: logits = token_embedding^T @ x
    // Compute logits as dot product of each embedding row with x
    if (m->weights.emb_type == GGUF_TENSOR_F16) {
        const uint16_t *emb = (const uint16_t *)w->token_embedding;
        for (int v = 0; v < c->vocab_size; v++) {
            const uint16_t *row = emb + (size_t)v * dim;
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                sum += fp16_to_fp32(row[d]) * s->x[d];
            }
            s->logits[v] = sum;
        }
    } else {
        // F32 embeddings
        const float *emb = (const float *)w->token_embedding;
        for (int v = 0; v < c->vocab_size; v++) {
            const float *row = emb + (size_t)v * dim;
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                sum += row[d] * s->x[d];
            }
            s->logits[v] = sum;
        }
    }

    return s->logits;
}
