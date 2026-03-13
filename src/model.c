#include "model.h"
#include "sh_arena.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- Helper: load a BnQWeight from GGUF tensor + scale tensor ---

static int load_qweight(BnQWeight *w, BnGGUFFile *f, const char *weight_name, const char *scale_name) {
    int ti = bn_gguf_find_tensor(f, weight_name);
    if (ti < 0) {
        SH_LOG_ERROR("Tensor not found", "name", weight_name);
        return -1;
    }

    BnGGUFTensorInfo *info = &f->tensors[ti];
    w->data = bn_gguf_tensor_data(f, ti);
    if (!w->data) {
        SH_LOG_ERROR("Tensor data out of bounds", "name", weight_name);
        return -1;
    }
    w->type = info->type;
    w->rows = (int)info->dims[1];
    w->cols = (int)info->dims[0];

    if (w->type == BN_GGUF_TENSOR_I2_S) {
        // I2_S: per-tensor scale stored at end of packed data (offset = nelements/4)
        size_t nelements = (size_t)w->rows * w->cols;
        const uint8_t *base = (const uint8_t *)w->data;
        const float *scale_ptr = (const float *)(base + nelements / 4);
        w->scale = *scale_ptr;
    } else {
        // TQ1_0/TQ2_0: companion .scale tensor
        int si = bn_gguf_find_tensor(f, scale_name);
        if (si >= 0) {
            float *scale_ptr = (float *)bn_gguf_tensor_data(f, si);
            w->scale = scale_ptr ? *scale_ptr : 1.0f;
        } else {
            w->scale = 1.0f;
        }
    }

    return 0;
}

// --- Helper: load F32 norm weights from GGUF ---

static float *load_f32_tensor(BnGGUFFile *f, const char *name) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0) return NULL;
    return (float *)bn_gguf_tensor_data(f, ti);
}

// --- Model loading ---

int bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len) {
    memset(m, 0, sizeof(BnModel));
    BnConfig *c = &m->config;

    // Try to detect architecture prefix
    const char *arch = bn_gguf_get_str(f, "general.architecture");
    char prefix[64] = "llama";
    if (arch) {
        snprintf(prefix, sizeof(prefix), "%s", arch);
    }

    // Build key names with architecture prefix
    char key[128];

    snprintf(key, sizeof(key), "%s.embedding_length", prefix);
    c->dim = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.feed_forward_length", prefix);
    c->hidden_dim = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.block_count", prefix);
    c->n_layers = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.attention.head_count", prefix);
    c->n_heads = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.attention.head_count_kv", prefix);
    c->n_kv_heads = (int)bn_gguf_get_u32(f, key);
    if (c->n_kv_heads == 0) c->n_kv_heads = c->n_heads;

    snprintf(key, sizeof(key), "%s.context_length", prefix);
    c->seq_len = (int)bn_gguf_get_u32(f, key);
    if (max_seq_len > 0 && max_seq_len < c->seq_len) c->seq_len = max_seq_len;

    snprintf(key, sizeof(key), "%s.rope.freq_base", prefix);
    c->rope_theta = bn_gguf_get_f32(f, key);
    if (c->rope_theta == 0.0f) c->rope_theta = BN_DEFAULT_ROPE_THETA;

    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", prefix);
    c->norm_eps = bn_gguf_get_f32(f, key);
    if (c->norm_eps == 0.0f) c->norm_eps = BN_DEFAULT_NORM_EPS;

    // Vocab size from tokenizer metadata
    c->vocab_size = (int)bn_gguf_get_arr_n(f, "tokenizer.ggml.tokens");

    // #15, #38: Validate BEFORE computing derived dimensions to avoid division by zero
    if (c->dim <= 0 || c->n_layers <= 0 || c->n_heads <= 0 ||
        c->vocab_size <= 0 || c->n_kv_heads <= 0 || c->hidden_dim <= 0 ||
        c->seq_len <= 0) {
        SH_LOG_ERROR("Invalid model config");
        return -1;
    }

    // Derived dimensions (safe now — denominators validated above)
    c->head_size = c->dim / c->n_heads;
    c->kv_dim = c->head_size * c->n_kv_heads;
    c->kv_mul = c->n_heads / c->n_kv_heads;

    // Validate alignment for NEON vectorized paths
    if (c->dim % c->n_heads != 0) {
        SH_LOG_ERROR("dim not divisible by n_heads");
        return -1;
    }
    if (c->n_heads % c->n_kv_heads != 0) {
        SH_LOG_ERROR("n_heads not divisible by n_kv_heads");
        return -1;
    }

    // Detect FFN gate and activation type
    c->has_ffn_gate = (bn_gguf_find_tensor(f, "blk.0.ffn_gate.weight") >= 0) ? 1 : 0;

    // Check for activation type: bitnet uses ReLU² (act_type=1)
    if (arch && strncmp(arch, "bitnet", 6) == 0) {
        c->act_type = 1;  // ReLU²
    } else {
        c->act_type = 0;  // SiLU (default for LLaMA-like)
    }

    {
        char dim_s[16], layers_s[16], heads_s[16], vocab_s[16];
        snprintf(dim_s, sizeof(dim_s), "%d", c->dim);
        snprintf(layers_s, sizeof(layers_s), "%d", c->n_layers);
        snprintf(heads_s, sizeof(heads_s), "%d", c->n_heads);
        snprintf(vocab_s, sizeof(vocab_s), "%d", c->vocab_size);
        SH_LOG_DEBUG("Model config", "dim", dim_s, "layers", layers_s,
                     "heads", heads_s, "vocab", vocab_s);
    }

    // --- Load weights ---
    BnWeights *w = &m->weights;

    // Token embedding
    int emb_idx = bn_gguf_find_tensor(f, "token_embd.weight");
    if (emb_idx < 0) {
        SH_LOG_ERROR("token_embd.weight not found");
        return -1;
    }
    w->token_embedding = bn_gguf_tensor_data(f, emb_idx);
    if (!w->token_embedding) {
        SH_LOG_ERROR("token_embd.weight data out of bounds");
        return -1;
    }
    w->emb_type = f->tensors[emb_idx].type;

    // #24: Output norm — must exist
    w->output_norm = load_f32_tensor(f, "output_norm.weight");
    if (!w->output_norm) {
        SH_LOG_ERROR("output_norm.weight not found");
        return -1;
    }

    // Allocate per-layer weights
    w->layers = (BnLayerWeights *)calloc(c->n_layers, sizeof(BnLayerWeights));
    if (!w->layers) {
        SH_LOG_ERROR("Failed to allocate layer weights");
        return -1;
    }

    for (int i = 0; i < c->n_layers; i++) {
        BnLayerWeights *lw = &w->layers[i];
        char wname[128], sname[128];

        // #25: Attention norms — must exist
        snprintf(wname, sizeof(wname), "blk.%d.attn_norm.weight", i);
        lw->attn_norm = load_f32_tensor(f, wname);
        if (!lw->attn_norm) {
            SH_LOG_ERROR("Tensor not found", "name", wname);
            goto fail_layers;
        }

        snprintf(wname, sizeof(wname), "blk.%d.attn_sub_norm.weight", i);
        lw->attn_sub_norm = load_f32_tensor(f, wname);  // optional

        // #23: Check load_qweight return values
        snprintf(wname, sizeof(wname), "blk.%d.attn_q.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_q.scale", i);
        if (load_qweight(&lw->wq, f, wname, sname) != 0) goto fail_layers;

        snprintf(wname, sizeof(wname), "blk.%d.attn_k.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_k.scale", i);
        if (load_qweight(&lw->wk, f, wname, sname) != 0) goto fail_layers;

        snprintf(wname, sizeof(wname), "blk.%d.attn_v.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_v.scale", i);
        if (load_qweight(&lw->wv, f, wname, sname) != 0) goto fail_layers;

        snprintf(wname, sizeof(wname), "blk.%d.attn_output.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_output.scale", i);
        if (load_qweight(&lw->wo, f, wname, sname) != 0) goto fail_layers;

        // #25: FFN norms — must exist
        snprintf(wname, sizeof(wname), "blk.%d.ffn_norm.weight", i);
        lw->ffn_norm = load_f32_tensor(f, wname);
        if (!lw->ffn_norm) {
            SH_LOG_ERROR("Tensor not found", "name", wname);
            goto fail_layers;
        }

        snprintf(wname, sizeof(wname), "blk.%d.ffn_sub_norm.weight", i);
        lw->ffn_sub_norm = load_f32_tensor(f, wname);  // optional

        // FFN gate/up/down weights
        if (c->has_ffn_gate) {
            snprintf(wname, sizeof(wname), "blk.%d.ffn_gate.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.ffn_gate.scale", i);
            if (load_qweight(&lw->ffn_gate, f, wname, sname) != 0) goto fail_layers;
        }

        snprintf(wname, sizeof(wname), "blk.%d.ffn_up.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_up.scale", i);
        if (load_qweight(&lw->ffn_up, f, wname, sname) != 0) goto fail_layers;

        snprintf(wname, sizeof(wname), "blk.%d.ffn_down.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_down.scale", i);
        if (load_qweight(&lw->ffn_down, f, wname, sname) != 0) goto fail_layers;
    }

    // --- Allocate BnRunState via arena ---
    // #1, #14: Check all allocations and guard against overflow
    BnRunState *s = &m->state;

    size_t att_size = (size_t)c->n_heads * c->seq_len;
    if (att_size / c->n_heads != (size_t)c->seq_len) {
        SH_LOG_ERROR("Attention buffer size overflow");
        goto fail_state;
    }

    // #14: Check for overflow before large KV cache allocations
    size_t kv_cache_size = (size_t)c->n_layers * c->seq_len * c->kv_dim;
    if (c->n_layers > 0 && c->seq_len > 0 && c->kv_dim > 0 &&
        kv_cache_size / c->n_layers / c->seq_len != (size_t)c->kv_dim) {
        SH_LOG_ERROR("KV cache size overflow");
        goto fail_state;
    }

    int x_q_size = c->dim > c->hidden_dim ? c->dim : c->hidden_dim;
    int half_head = c->head_size / 2;

    // Compute total arena capacity (all RunState buffers)
    size_t arena_size = 0;
    arena_size += 4 * (size_t)c->dim * sizeof(float);         // x, xb, xb2, q
    arena_size += 2 * (size_t)c->hidden_dim * sizeof(float);  // hb, hb2
    arena_size += att_size * sizeof(float);                     // att
    arena_size += (size_t)c->vocab_size * sizeof(float);       // logits
    arena_size += 2 * kv_cache_size * sizeof(float);           // key_cache, value_cache
    arena_size += (size_t)x_q_size * sizeof(int8_t);           // x_q
    arena_size += (size_t)half_head * sizeof(float);           // rope_freq
    arena_size += 12 * SH_ARENA_ALIGN;                         // alignment padding

    m->arena = sh_arena_create(arena_size);
    if (!m->arena) {
        SH_LOG_ERROR("Failed to allocate run state arena");
        goto fail_state;
    }

    s->x           = (float *)sh_arena_calloc(m->arena, c->dim, sizeof(float));
    s->xb          = (float *)sh_arena_calloc(m->arena, c->dim, sizeof(float));
    s->xb2         = (float *)sh_arena_calloc(m->arena, c->dim, sizeof(float));
    s->q           = (float *)sh_arena_calloc(m->arena, c->dim, sizeof(float));
    s->hb          = (float *)sh_arena_calloc(m->arena, c->hidden_dim, sizeof(float));
    s->hb2         = (float *)sh_arena_calloc(m->arena, c->hidden_dim, sizeof(float));
    s->att         = (float *)sh_arena_calloc(m->arena, att_size, sizeof(float));
    s->logits      = (float *)sh_arena_calloc(m->arena, c->vocab_size, sizeof(float));
    s->key_cache   = (float *)sh_arena_calloc(m->arena, kv_cache_size, sizeof(float));
    s->value_cache = (float *)sh_arena_calloc(m->arena, kv_cache_size, sizeof(float));
    s->x_q         = (int8_t *)sh_arena_calloc(m->arena, x_q_size, sizeof(int8_t));
    s->rope_freq   = (float *)sh_arena_alloc(m->arena, half_head * sizeof(float));

    // #1: Check all allocations succeeded
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache ||
        !s->x_q || !s->rope_freq) {
        SH_LOG_ERROR("Failed to allocate run state buffers");
        goto fail_state;
    }

    // Precompute RoPE frequencies: freq[i] = 1/theta^(2i/head_size)
    for (int i = 0; i < half_head; i++) {
        s->rope_freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)c->head_size);
    }

    return 0;

fail_state:
    bn_model_free(m);
    return -1;

fail_layers:
    bn_model_free(m);
    return -1;
}

void bn_model_free(BnModel *m) {
    if (!m) return;
    bn_tp_free(m->pool);
    free(m->weights.layers);
    sh_arena_free(m->arena);
    memset(m, 0, sizeof(BnModel));
}

// #8: Bounds-check token before accessing embedding table
void bn_model_embed_token(const BnModel *m, float *out, int token) {
    int dim = m->config.dim;

    if (token < 0 || token >= m->config.vocab_size) {
        SH_LOG_ERROR("Token out of range");
        memset(out, 0, dim * sizeof(float));
        return;
    }

    if (m->weights.emb_type == BN_GGUF_TENSOR_F16) {
        // Dequantize one row of F16 embedding
        const uint16_t *emb = (const uint16_t *)m->weights.token_embedding;
        const uint16_t *row = emb + (size_t)token * dim;
        for (int i = 0; i < dim; i++) {
            out[i] = bn_fp16_to_fp32(row[i]);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_F32) {
        const float *emb = (const float *)m->weights.token_embedding;
        memcpy(out, emb + (size_t)token * dim, dim * sizeof(float));
    } else {
        SH_LOG_ERROR("Unsupported embedding type");
        memset(out, 0, dim * sizeof(float));
    }
}
