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
    if (info->n_dims < 2) {
        SH_LOG_ERROR("Weight tensor must be 2D", "name", weight_name);
        return -1;
    }
    w->data = bn_gguf_tensor_data(f, ti);
    if (!w->data) {
        SH_LOG_ERROR("Tensor data out of bounds", "name", weight_name);
        return -1;
    }
    w->type = info->type;
    w->rows = (int)info->dims[1];
    w->cols = (int)info->dims[0];

    // Validate supported tensor types
    if (w->type != BN_GGUF_TENSOR_I2_S && w->type != BN_GGUF_TENSOR_Q4_0 &&
        w->type != BN_GGUF_TENSOR_Q4_1 && w->type != BN_GGUF_TENSOR_Q8_0 &&
        w->type != BN_GGUF_TENSOR_TQ1_0 && w->type != BN_GGUF_TENSOR_TQ2_0 &&
        w->type != BN_GGUF_TENSOR_Q2_K && w->type != BN_GGUF_TENSOR_Q3_K &&
        w->type != BN_GGUF_TENSOR_Q4_K && w->type != BN_GGUF_TENSOR_Q5_K &&
        w->type != BN_GGUF_TENSOR_Q6_K && w->type != BN_GGUF_TENSOR_Q8_K &&
        w->type != BN_GGUF_TENSOR_BF16 &&
        w->type != BN_GGUF_TENSOR_IQ4_NL && w->type != BN_GGUF_TENSOR_IQ4_XS &&
        w->type != BN_GGUF_TENSOR_IQ3_XXS && w->type != BN_GGUF_TENSOR_IQ3_S &&
        w->type != BN_GGUF_TENSOR_IQ2_XXS && w->type != BN_GGUF_TENSOR_IQ2_XS &&
        w->type != BN_GGUF_TENSOR_IQ2_S) {
        SH_LOG_ERROR("Unsupported tensor type", "name", weight_name);
        return -1;
    }

    if (w->type == BN_GGUF_TENSOR_I2_S) {
        // I2_S: per-tensor scale stored at end of packed data (offset = nelements/4)
        size_t nelements = (size_t)w->rows * w->cols;
        const uint8_t *base = (const uint8_t *)w->data;
        memcpy(&w->scale, base + nelements / 4, sizeof(float));
    } else if (w->type == BN_GGUF_TENSOR_Q4_0 || w->type == BN_GGUF_TENSOR_Q4_1 ||
               w->type == BN_GGUF_TENSOR_Q8_0 ||
               w->type == BN_GGUF_TENSOR_Q2_K || w->type == BN_GGUF_TENSOR_Q3_K ||
               w->type == BN_GGUF_TENSOR_Q4_K || w->type == BN_GGUF_TENSOR_Q5_K ||
               w->type == BN_GGUF_TENSOR_Q6_K || w->type == BN_GGUF_TENSOR_Q8_K ||
               w->type == BN_GGUF_TENSOR_BF16 ||
               w->type == BN_GGUF_TENSOR_IQ4_NL || w->type == BN_GGUF_TENSOR_IQ4_XS ||
               w->type == BN_GGUF_TENSOR_IQ3_XXS || w->type == BN_GGUF_TENSOR_IQ3_S ||
               w->type == BN_GGUF_TENSOR_IQ2_XXS || w->type == BN_GGUF_TENSOR_IQ2_XS ||
               w->type == BN_GGUF_TENSOR_IQ2_S) {
        // Per-block scales embedded in each block's d field
        w->scale = 1.0f;
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

int bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len, int kv_f16) {
    memset(m, 0, sizeof(BnModel));
    BnConfig *c = &m->config;
    c->kv_f16 = kv_f16;

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
    w->emb_out_i8 = NULL;
    w->emb_out_scales = NULL;

    // Untied output weight (if present)
    memset(&w->output_weight, 0, sizeof(w->output_weight));
    int out_idx = bn_gguf_find_tensor(f, "output.weight");
    if (out_idx >= 0) {
        BnGGUFTensorInfo *out_info = &f->tensors[out_idx];
        int ot = out_info->type;
        if (ot != BN_GGUF_TENSOR_Q4_0 && ot != BN_GGUF_TENSOR_Q4_1 &&
            ot != BN_GGUF_TENSOR_Q8_0 &&
            ot != BN_GGUF_TENSOR_Q2_K && ot != BN_GGUF_TENSOR_Q3_K &&
            ot != BN_GGUF_TENSOR_Q4_K && ot != BN_GGUF_TENSOR_Q5_K &&
            ot != BN_GGUF_TENSOR_Q6_K && ot != BN_GGUF_TENSOR_Q8_K &&
            ot != BN_GGUF_TENSOR_I2_S && ot != BN_GGUF_TENSOR_TQ1_0 &&
            ot != BN_GGUF_TENSOR_TQ2_0 && ot != BN_GGUF_TENSOR_F16 &&
            ot != BN_GGUF_TENSOR_BF16 &&
            ot != BN_GGUF_TENSOR_IQ4_NL && ot != BN_GGUF_TENSOR_IQ4_XS &&
            ot != BN_GGUF_TENSOR_IQ3_XXS && ot != BN_GGUF_TENSOR_IQ3_S &&
            ot != BN_GGUF_TENSOR_IQ2_XXS && ot != BN_GGUF_TENSOR_IQ2_XS &&
            ot != BN_GGUF_TENSOR_IQ2_S) {
            SH_LOG_ERROR("Unsupported output.weight type");
            return -1;
        }
        if (out_info->n_dims < 2) {
            SH_LOG_ERROR("output.weight must be 2D");
            return -1;
        }
        w->output_weight.data = bn_gguf_tensor_data(f, out_idx);
        w->output_weight.type = ot;
        w->output_weight.rows = (int)out_info->dims[1];
        w->output_weight.cols = (int)out_info->dims[0];
        if (ot == BN_GGUF_TENSOR_Q4_0 || ot == BN_GGUF_TENSOR_Q4_1 ||
            ot == BN_GGUF_TENSOR_Q8_0 ||
            ot == BN_GGUF_TENSOR_Q2_K || ot == BN_GGUF_TENSOR_Q3_K ||
            ot == BN_GGUF_TENSOR_Q4_K || ot == BN_GGUF_TENSOR_Q5_K ||
            ot == BN_GGUF_TENSOR_Q6_K || ot == BN_GGUF_TENSOR_Q8_K ||
            ot == BN_GGUF_TENSOR_BF16 ||
            ot == BN_GGUF_TENSOR_IQ4_NL || ot == BN_GGUF_TENSOR_IQ4_XS ||
            ot == BN_GGUF_TENSOR_IQ3_XXS || ot == BN_GGUF_TENSOR_IQ3_S ||
            ot == BN_GGUF_TENSOR_IQ2_XXS || ot == BN_GGUF_TENSOR_IQ2_XS ||
            ot == BN_GGUF_TENSOR_IQ2_S) {
            w->output_weight.scale = 1.0f;
        } else if (ot == BN_GGUF_TENSOR_I2_S) {
            size_t nel = (size_t)w->output_weight.rows * w->output_weight.cols;
            const uint8_t *base = (const uint8_t *)w->output_weight.data;
            memcpy(&w->output_weight.scale, base + nel / 4, sizeof(float));
        } else {
            w->output_weight.scale = 1.0f;
        }
    }

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

        // Attention biases (optional, used by Qwen2)
        snprintf(wname, sizeof(wname), "blk.%d.attn_q.bias", i);
        lw->q_bias = load_f32_tensor(f, wname);
        snprintf(wname, sizeof(wname), "blk.%d.attn_k.bias", i);
        lw->k_bias = load_f32_tensor(f, wname);
        snprintf(wname, sizeof(wname), "blk.%d.attn_v.bias", i);
        lw->v_bias = load_f32_tensor(f, wname);

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

    // INT8 embedding size (DOTPROD + F16 only)
    size_t emb_i8_bytes = 0;
    size_t emb_i8_scales_bytes = 0;
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
    int want_i8_emb = (w->emb_type == BN_GGUF_TENSOR_F16) ||
                       (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16);
    int i8_emb_rows = 0;
    if (want_i8_emb) {
        // For untied F16 output weight, quantize that; otherwise quantize tied embeddings
        i8_emb_rows = (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16)
                       ? w->output_weight.rows : c->vocab_size;
        emb_i8_bytes = (size_t)i8_emb_rows * c->dim;
        emb_i8_scales_bytes = (size_t)i8_emb_rows * sizeof(float);
    }
#endif

    // Compute total arena capacity (all RunState buffers + INT8 embeddings)
    size_t arena_size = 0;
    arena_size += 4 * (size_t)c->dim * sizeof(float);         // x, xb, xb2, q
    arena_size += 2 * (size_t)c->hidden_dim * sizeof(float);  // hb, hb2
    arena_size += att_size * sizeof(float);                     // att
    arena_size += (size_t)c->vocab_size * sizeof(float);       // logits
    size_t kv_elem_size = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    arena_size += 2 * kv_cache_size * kv_elem_size;           // key_cache, value_cache
    arena_size += (size_t)x_q_size * sizeof(int8_t);           // x_q
    arena_size += (size_t)half_head * sizeof(float);           // rope_freq
    arena_size += emb_i8_bytes + emb_i8_scales_bytes;          // INT8 embeddings
    arena_size += 14 * SH_ARENA_ALIGN;                         // alignment padding

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
    s->key_cache   = (float *)sh_arena_calloc(m->arena, kv_cache_size, kv_elem_size);
    s->value_cache = (float *)sh_arena_calloc(m->arena, kv_cache_size, kv_elem_size);
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

    // Quantize F16 embeddings to INT8 for fast SDOT logits kernel
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
    if (want_i8_emb) {
        w->emb_out_i8 = (int8_t *)sh_arena_alloc(m->arena, emb_i8_bytes);
        w->emb_out_scales = (float *)sh_arena_alloc(m->arena, emb_i8_scales_bytes);
        if (w->emb_out_i8 && w->emb_out_scales) {
            const uint16_t *src = (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16)
                                  ? (const uint16_t *)w->output_weight.data
                                  : (const uint16_t *)w->token_embedding;
            bn_quant_f16_rows_to_i8(src, w->emb_out_i8, w->emb_out_scales,
                                    i8_emb_rows, c->dim);
            char i8_mb[16]; snprintf(i8_mb, sizeof(i8_mb), "%.0f", (double)emb_i8_bytes / (1024*1024));
            SH_LOG_INFO("INT8 output embeddings ready", "MB", i8_mb);
        } else {
            w->emb_out_i8 = NULL;
            w->emb_out_scales = NULL;
            SH_LOG_DEBUG("INT8 embedding arena alloc failed, using F16 fallback");
        }
    }
#endif

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
    sh_arena_free(m->arena);  // frees INT8 embeddings too
    bn_platform_unload_file(&m->file);  // safe even if file.data is NULL
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
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q4_0) {
        const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockQ4_0 *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q4_0(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q8_0) {
        const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockQ8_0 *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q8_0(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q2_K) {
        const BnBlockQ2K *blocks = (const BnBlockQ2K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ2K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q2k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q3_K) {
        const BnBlockQ3K *blocks = (const BnBlockQ3K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ3K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q3k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q4_K) {
        const BnBlockQ4K *blocks = (const BnBlockQ4K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ4K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q4k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q5_K) {
        const BnBlockQ5K *blocks = (const BnBlockQ5K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ5K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q5k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q6_K) {
        const BnBlockQ6K *blocks = (const BnBlockQ6K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ6K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q6k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q8_K) {
        const BnBlockQ8K *blocks = (const BnBlockQ8K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ8K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q8k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q4_1) {
        const BnBlockQ4_1 *blocks = (const BnBlockQ4_1 *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockQ4_1 *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q4_1(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_BF16) {
        const uint16_t *emb = (const uint16_t *)m->weights.token_embedding;
        const uint16_t *row = emb + (size_t)token * dim;
        for (int i = 0; i < dim; i++) {
            out[i] = bn_bf16_to_fp32(row[i]);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ4_NL) {
        const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockIQ4NL *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq4nl(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ4_XS) {
        const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ4XS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq4xs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ3_XXS) {
        const BnBlockIQ3XXS *blocks = (const BnBlockIQ3XXS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ3XXS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq3xxs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ3_S) {
        const BnBlockIQ3S *blocks = (const BnBlockIQ3S *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ3S *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq3s(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ2_XXS) {
        const BnBlockIQ2XXS *blocks = (const BnBlockIQ2XXS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ2XXS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq2xxs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ2_XS) {
        const BnBlockIQ2XS *blocks = (const BnBlockIQ2XS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ2XS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq2xs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ2_S) {
        const BnBlockIQ2S *blocks = (const BnBlockIQ2S *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ2S *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq2s(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_F32) {
        const float *emb = (const float *)m->weights.token_embedding;
        memcpy(out, emb + (size_t)token * dim, dim * sizeof(float));
    } else {
        SH_LOG_ERROR("Unsupported embedding type");
        memset(out, 0, dim * sizeof(float));
    }
}
