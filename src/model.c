#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Helper: load a QWeight from GGUF tensor + scale tensor ---

static int load_qweight(QWeight *w, GGUFFile *f, const char *weight_name, const char *scale_name) {
    int ti = gguf_find_tensor(f, weight_name);
    if (ti < 0) {
        fprintf(stderr, "model: tensor '%s' not found\n", weight_name);
        return -1;
    }

    GGUFTensorInfo *info = &f->tensors[ti];
    w->data = gguf_tensor_data(f, ti);
    w->type = info->type;
    w->rows = (int)info->dims[1];
    w->cols = (int)info->dims[0];

    // Load companion scale tensor
    int si = gguf_find_tensor(f, scale_name);
    if (si >= 0) {
        // Scale tensor is a single F32 value
        float *scale_ptr = (float *)gguf_tensor_data(f, si);
        w->scale = *scale_ptr;
    } else {
        w->scale = 1.0f;
    }

    return 0;
}

// --- Helper: load F32 norm weights from GGUF ---

static float *load_f32_tensor(GGUFFile *f, const char *name) {
    int ti = gguf_find_tensor(f, name);
    if (ti < 0) return NULL;
    return (float *)gguf_tensor_data(f, ti);
}

// --- Model loading ---

int model_load(Model *m, GGUFFile *f, int max_seq_len) {
    memset(m, 0, sizeof(Model));
    Config *c = &m->config;

    // Try to detect architecture prefix
    const char *arch = gguf_get_str(f, "general.architecture");
    char prefix[64] = "llama";
    if (arch) {
        snprintf(prefix, sizeof(prefix), "%s", arch);
    }

    // Build key names with architecture prefix
    char key[128];

    snprintf(key, sizeof(key), "%s.embedding_length", prefix);
    c->dim = (int)gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.feed_forward_length", prefix);
    c->hidden_dim = (int)gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.block_count", prefix);
    c->n_layers = (int)gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.attention.head_count", prefix);
    c->n_heads = (int)gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.attention.head_count_kv", prefix);
    c->n_kv_heads = (int)gguf_get_u32(f, key);
    if (c->n_kv_heads == 0) c->n_kv_heads = c->n_heads;

    snprintf(key, sizeof(key), "%s.context_length", prefix);
    c->seq_len = (int)gguf_get_u32(f, key);
    if (max_seq_len > 0 && max_seq_len < c->seq_len) c->seq_len = max_seq_len;

    snprintf(key, sizeof(key), "%s.rope.freq_base", prefix);
    c->rope_theta = gguf_get_f32(f, key);
    if (c->rope_theta == 0.0f) c->rope_theta = 10000.0f;

    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", prefix);
    c->norm_eps = gguf_get_f32(f, key);
    if (c->norm_eps == 0.0f) c->norm_eps = 1e-5f;

    // Vocab size from tokenizer metadata
    c->vocab_size = (int)gguf_get_arr_n(f, "tokenizer.ggml.tokens");

    // Derived dimensions
    c->head_size = c->dim / c->n_heads;
    c->kv_dim = c->head_size * c->n_kv_heads;
    c->kv_mul = c->n_heads / c->n_kv_heads;

    // Detect FFN gate and activation type
    c->has_ffn_gate = (gguf_find_tensor(f, "blk.0.ffn_gate.weight") >= 0) ? 1 : 0;

    // Check for activation type: bitnet uses ReLU² (act_type=1)
    // Check architecture-specific key or default based on arch
    if (arch && strcmp(arch, "bitnet") == 0) {
        c->act_type = 1;  // ReLU²
    } else {
        c->act_type = 0;  // SiLU (default for LLaMA-like)
    }

    // Validate
    if (c->dim == 0 || c->n_layers == 0 || c->n_heads == 0 || c->vocab_size == 0) {
        fprintf(stderr, "model: invalid config (dim=%d layers=%d heads=%d vocab=%d)\n",
                c->dim, c->n_layers, c->n_heads, c->vocab_size);
        return -1;
    }

    #ifdef DEBUG
    fprintf(stderr, "model: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d\n",
            c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);
    fprintf(stderr, "model: head_size=%d kv_dim=%d kv_mul=%d rope_theta=%.1f norm_eps=%g\n",
            c->head_size, c->kv_dim, c->kv_mul, c->rope_theta, c->norm_eps);
    fprintf(stderr, "model: has_ffn_gate=%d act_type=%d\n", c->has_ffn_gate, c->act_type);
    #endif

    // --- Load weights ---
    Weights *w = &m->weights;

    // Token embedding
    int emb_idx = gguf_find_tensor(f, "token_embd.weight");
    if (emb_idx < 0) {
        fprintf(stderr, "model: token_embd.weight not found\n");
        return -1;
    }
    w->token_embedding = gguf_tensor_data(f, emb_idx);
    w->emb_type = f->tensors[emb_idx].type;

    // Output norm
    w->output_norm = load_f32_tensor(f, "output_norm.weight");

    // Allocate per-layer weights
    w->layers = (LayerWeights *)calloc(c->n_layers, sizeof(LayerWeights));

    for (int i = 0; i < c->n_layers; i++) {
        LayerWeights *lw = &w->layers[i];
        char wname[128], sname[128];

        // Attention norms
        snprintf(wname, sizeof(wname), "blk.%d.attn_norm.weight", i);
        lw->attn_norm = load_f32_tensor(f, wname);

        snprintf(wname, sizeof(wname), "blk.%d.attn_sub_norm.weight", i);
        lw->attn_sub_norm = load_f32_tensor(f, wname);

        // Attention Q/K/V/O weights
        snprintf(wname, sizeof(wname), "blk.%d.attn_q.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_q.scale", i);
        load_qweight(&lw->wq, f, wname, sname);

        snprintf(wname, sizeof(wname), "blk.%d.attn_k.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_k.scale", i);
        load_qweight(&lw->wk, f, wname, sname);

        snprintf(wname, sizeof(wname), "blk.%d.attn_v.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_v.scale", i);
        load_qweight(&lw->wv, f, wname, sname);

        snprintf(wname, sizeof(wname), "blk.%d.attn_output.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_output.scale", i);
        load_qweight(&lw->wo, f, wname, sname);

        // FFN norms
        snprintf(wname, sizeof(wname), "blk.%d.ffn_norm.weight", i);
        lw->ffn_norm = load_f32_tensor(f, wname);

        snprintf(wname, sizeof(wname), "blk.%d.ffn_sub_norm.weight", i);
        lw->ffn_sub_norm = load_f32_tensor(f, wname);

        // FFN gate/up/down weights
        if (c->has_ffn_gate) {
            snprintf(wname, sizeof(wname), "blk.%d.ffn_gate.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.ffn_gate.scale", i);
            load_qweight(&lw->ffn_gate, f, wname, sname);
        }

        snprintf(wname, sizeof(wname), "blk.%d.ffn_up.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_up.scale", i);
        load_qweight(&lw->ffn_up, f, wname, sname);

        snprintf(wname, sizeof(wname), "blk.%d.ffn_down.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_down.scale", i);
        load_qweight(&lw->ffn_down, f, wname, sname);
    }

    // --- Allocate RunState ---
    RunState *s = &m->state;
    s->x       = (float *)calloc(c->dim, sizeof(float));
    s->xb      = (float *)calloc(c->dim, sizeof(float));
    s->xb2     = (float *)calloc(c->dim, sizeof(float));
    s->hb      = (float *)calloc(c->hidden_dim, sizeof(float));
    s->hb2     = (float *)calloc(c->hidden_dim, sizeof(float));
    s->q       = (float *)calloc(c->dim, sizeof(float));
    s->att     = (float *)calloc(c->n_heads * c->seq_len, sizeof(float));
    s->logits  = (float *)calloc(c->vocab_size, sizeof(float));
    s->key_cache   = (float *)calloc((size_t)c->n_layers * c->seq_len * c->kv_dim, sizeof(float));
    s->value_cache = (float *)calloc((size_t)c->n_layers * c->seq_len * c->kv_dim, sizeof(float));

    return 0;
}

void model_free(Model *m) {
    if (!m) return;
    free(m->weights.layers);
    RunState *s = &m->state;
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void model_embed_token(const Model *m, float *out, int token) {
    int dim = m->config.dim;

    if (m->weights.emb_type == GGUF_TENSOR_F16) {
        // Dequantize one row of F16 embedding
        const uint16_t *emb = (const uint16_t *)m->weights.token_embedding;
        const uint16_t *row = emb + (size_t)token * dim;
        for (int i = 0; i < dim; i++) {
            out[i] = fp16_to_fp32(row[i]);
        }
    } else if (m->weights.emb_type == GGUF_TENSOR_F32) {
        const float *emb = (const float *)m->weights.token_embedding;
        memcpy(out, emb + (size_t)token * dim, dim * sizeof(float));
    } else {
        fprintf(stderr, "model: unsupported embedding type %d\n", m->weights.emb_type);
        memset(out, 0, dim * sizeof(float));
    }
}
