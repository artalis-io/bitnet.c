#include "model.h"
#include "moe.h"
#include "sh_arena.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
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
    if (info->dims[1] > INT_MAX || info->dims[0] > INT_MAX) {
        SH_LOG_ERROR("Tensor dimensions exceed INT_MAX", "name", weight_name);
        return -1;
    }
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

// --- Q4_0 weight repacking helpers ---

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static size_t q4_repack_bytes(const BnQWeight *w) {
    if (w->type != BN_GGUF_TENSOR_Q4_0 || !w->data) return 0;
    if (w->rows % 4 != 0) return 0;  // need complete 4-row groups
    size_t n_blocks = (size_t)w->rows * (w->cols / 32);
    return n_blocks * sizeof(float) + n_blocks * 16 + 2 * SH_ARENA_ALIGN;
}

static void q4_repack(BnQWeight *w, SHArena *arena) {
    if (w->type != BN_GGUF_TENSOR_Q4_0 || !w->data) return;
    if (w->rows % 4 != 0) return;  // need complete 4-row groups
    int n_blocks_per_row = w->cols / 32;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;

    w->rp_scales = (float *)sh_arena_alloc(arena, n_blocks * sizeof(float));
    w->rp_qs = (uint8_t *)sh_arena_alloc(arena, n_blocks * 16);
    if (!w->rp_scales || !w->rp_qs) {
        w->rp_scales = NULL;
        w->rp_qs = NULL;
        return;
    }

    // Nibble-transposed 4-row interleaved layout for vdotq_laneq_s32:
    // Scales: group g, block b → rp_scales[(g * n_blocks_per_row + b) * 4 + r]
    // Quants: group g, block b → rp_qs[(g * n_blocks_per_row + b) * 64 + ...]
    //   Within each 64-byte chunk, bytes are ordered for lane-select SDOT:
    //   [r0_b0..b3, r1_b0..b3, r2_b0..b3, r3_b0..b3,   <- register 0 (16 bytes)
    //    r0_b4..b7, r1_b4..b7, r2_b4..b7, r3_b4..b7,   <- register 1
    //    r0_b8..b11, r1_b8..b11, r2_b8..b11, r3_b8..b11, <- register 2
    //    r0_b12..b15, r1_b12..b15, r2_b12..b15, r3_b12..b15] <- register 3
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)w->data;
    int n_groups = w->rows / 4;
    for (int g = 0; g < n_groups; g++) {
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t gb = (size_t)g * n_blocks_per_row + b;
            // Scales: same flat interleaved layout
            for (int r = 0; r < 4; r++) {
                size_t src = (size_t)(g * 4 + r) * n_blocks_per_row + b;
                w->rp_scales[gb * 4 + r] = bn_fp16_to_fp32(blocks[src].d);
            }
            // Quants: nibble-transpose within 64-byte chunk
            uint8_t *dst = w->rp_qs + gb * 64;
            for (int ng = 0; ng < 4; ng++) {
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) * n_blocks_per_row + b;
                    memcpy(dst + ng * 16 + r * 4, blocks[src].qs + ng * 4, 4);
                }
            }
        }
    }
}
#endif

// --- Helper: load F32 norm weights from GGUF ---

static float *load_f32_tensor(BnGGUFFile *f, const char *name) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0) return NULL;
    return (float *)bn_gguf_tensor_data(f, ti);
}

// --- Helper: compute byte size for quantized tensor type ---

static size_t bn_tensor_type_size(int type, size_t nelements) {
    switch (type) {
        case BN_GGUF_TENSOR_F32:      return nelements * 4;
        case BN_GGUF_TENSOR_F16:      return nelements * 2;
        case BN_GGUF_TENSOR_BF16:     return nelements * 2;
        case BN_GGUF_TENSOR_Q4_0:     return (nelements / 32) * 18;
        case BN_GGUF_TENSOR_Q4_1:     return (nelements / 32) * 20;
        case BN_GGUF_TENSOR_Q8_0:     return (nelements / 32) * 34;
        case BN_GGUF_TENSOR_I2_S:     return (nelements / 4) + 4;
        case BN_GGUF_TENSOR_TQ1_0:    return (nelements / 256) * 54;
        case BN_GGUF_TENSOR_TQ2_0:    return (nelements / 256) * 66;
        case BN_GGUF_TENSOR_Q2_K:     return (nelements / 256) * 84;
        case BN_GGUF_TENSOR_Q3_K:     return (nelements / 256) * 110;
        case BN_GGUF_TENSOR_Q4_K:     return (nelements / 256) * 144;
        case BN_GGUF_TENSOR_Q5_K:     return (nelements / 256) * 176;
        case BN_GGUF_TENSOR_Q6_K:     return (nelements / 256) * 210;
        case BN_GGUF_TENSOR_Q8_K:     return (nelements / 256) * 292;
        case BN_GGUF_TENSOR_IQ4_NL:   return (nelements / 32) * 18;
        case BN_GGUF_TENSOR_IQ4_XS:   return (nelements / 256) * 136;
        case BN_GGUF_TENSOR_IQ3_XXS:  return (nelements / 256) * 98;
        case BN_GGUF_TENSOR_IQ3_S:    return (nelements / 256) * 114;
        case BN_GGUF_TENSOR_IQ2_XXS:  return (nelements / 256) * 66;
        case BN_GGUF_TENSOR_IQ2_XS:   return (nelements / 256) * 74;
        case BN_GGUF_TENSOR_IQ2_S:    return (nelements / 256) * 82;
        default: return 0;
    }
}

// --- Helper: compute expert map for one fused 3D tensor ---
// Tensor shape: [n_experts, rows_per_expert, cols_per_expert]
// Returns file offset of first expert's data and bytes per expert slice.
static int load_expert_map_proj(BnGGUFFile *f, const char *name,
                                int n_experts, int *type_out,
                                int *rows_out, int *cols_out,
                                size_t *base_offset_out, size_t *expert_bytes_out) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0) return -1;

    BnGGUFTensorInfo *info = &f->tensors[ti];
    if (info->n_dims < 3) {
        SH_LOG_ERROR("Expert tensor must be 3D", "name", name);
        return -1;
    }

    // GGUF dims: [cols, rows, n_experts] (column-major convention)
    int cols = (int)info->dims[0];
    int rows = (int)info->dims[1];
    int n_exp = (int)info->dims[2];
    if (n_exp != n_experts) {
        SH_LOG_ERROR("Expert count mismatch in tensor", "name", name);
        return -1;
    }

    *type_out = (int)info->type;
    *rows_out = rows;
    *cols_out = cols;

    // Compute file offset of tensor data
    size_t tensor_offset = f->data_offset + info->offset;
    *base_offset_out = tensor_offset;

    // Bytes per single expert slice
    size_t expert_elements = (size_t)rows * cols;
    *expert_bytes_out = bn_tensor_type_size((int)info->type, expert_elements);
    if (*expert_bytes_out == 0) {
        SH_LOG_ERROR("Unsupported expert tensor type", "name", name);
        return -1;
    }

    return 0;
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

    // Early MoE expert count read (needed for validation — hidden_dim can be 0 for MoE-only FFN)
    snprintf(key, sizeof(key), "%s.expert_count", prefix);
    int early_n_experts = (int)bn_gguf_get_u32(f, key);

    // #15, #38: Validate BEFORE computing derived dimensions to avoid division by zero
    // hidden_dim may be 0 for pure MoE models (all FFN is expert-based)
    if (c->dim <= 0 || c->n_layers <= 0 || c->n_heads <= 0 ||
        c->vocab_size <= 0 || c->n_kv_heads <= 0 || c->seq_len <= 0 ||
        (c->hidden_dim <= 0 && early_n_experts <= 0)) {
        SH_LOG_ERROR("Invalid model config");
        return -1;
    }

    // Derived dimensions (safe now — denominators validated above)
    // Check for explicit head size (Qwen3 has key_length != dim/n_heads)
    snprintf(key, sizeof(key), "%s.attention.key_length", prefix);
    int explicit_head_size = (int)bn_gguf_get_u32(f, key);
    c->head_size = (explicit_head_size > 0) ? explicit_head_size : (c->dim / c->n_heads);
    c->kv_dim = c->head_size * c->n_kv_heads;
    c->kv_mul = c->n_heads / c->n_kv_heads;

    // Validate alignment for SIMD vectorized paths
    if (explicit_head_size == 0 && c->dim % c->n_heads != 0) {
        SH_LOG_ERROR("dim not divisible by n_heads");
        return -1;
    }
    if (c->n_heads % c->n_kv_heads != 0) {
        SH_LOG_ERROR("n_heads not divisible by n_kv_heads");
        return -1;
    }
    if (c->dim % 128 != 0) {
        SH_LOG_ERROR("dim must be multiple of 128 for SIMD kernels");
        return -1;
    }
    if (c->head_size % 16 != 0) {
        SH_LOG_ERROR("head_size must be multiple of 16 for SIMD kernels");
        return -1;
    }

    // Hybrid SSM + Attention config (all default to 0 for pure attention models)
    snprintf(key, sizeof(key), "%s.rope.dimension_count", prefix);
    c->rope_dim_count = (int)bn_gguf_get_u32(f, key);

    // MROPE: dimension_sections[0] = text-only RoPE pairs (sections 1,2 are vision)
    // For text-only inference, only apply RoPE to the first section's dimensions.
    snprintf(key, sizeof(key), "%s.rope.dimension_sections", prefix);
    {
        uint64_t nsect = bn_gguf_get_arr_n(f, key);
        if (nsect > 0) {
            const int32_t *sections = (const int32_t *)bn_gguf_get_arr_data(f, key);
            if (sections && sections[0] > 0 && c->rope_dim_count > 0) {
                c->rope_text_dims = sections[0] * 2;  // pairs → dims
            }
        }
    }

    snprintf(key, sizeof(key), "%s.full_attention_interval", prefix);
    c->full_attn_interval = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.ssm.state_size", prefix);
    c->ssm_state_size = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.ssm.conv_kernel", prefix);
    c->ssm_conv_kernel = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.ssm.inner_size", prefix);
    c->ssm_inner_size = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.ssm.time_step_rank", prefix);
    c->ssm_time_step_rank = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.ssm.group_count", prefix);
    c->ssm_group_count = (int)bn_gguf_get_u32(f, key);

    // Validate SSM config when hybrid model
    if (c->full_attn_interval > 0) {
        if (c->ssm_time_step_rank <= 0) {
            SH_LOG_ERROR("ssm_time_step_rank must be > 0 for hybrid models");
            return -1;
        }
        if (c->ssm_inner_size <= 0 || c->ssm_inner_size % c->ssm_time_step_rank != 0) {
            SH_LOG_ERROR("ssm_inner_size must be > 0 and divisible by ssm_time_step_rank");
            return -1;
        }
        if (c->ssm_state_size <= 0 || c->ssm_group_count <= 0) {
            SH_LOG_ERROR("ssm_state_size and ssm_group_count must be > 0");
            return -1;
        }
    }

    // MoE config
    snprintf(key, sizeof(key), "%s.expert_count", prefix);
    c->n_experts = (int)bn_gguf_get_u32(f, key);
    snprintf(key, sizeof(key), "%s.expert_used_count", prefix);
    c->n_experts_active = (int)bn_gguf_get_u32(f, key);

    if (c->n_experts > 0) {
        // Per-expert intermediate size (from expert FFN tensors)
        snprintf(key, sizeof(key), "%s.expert_feed_forward_length", prefix);
        c->moe_intermediate_size = (int)bn_gguf_get_u32(f, key);
        if (c->moe_intermediate_size == 0) {
            // Fallback: infer from tensor dims
            int ti = bn_gguf_find_tensor(f, "blk.0.ffn_gate_exps.weight");
            if (ti >= 0 && f->tensors[ti].n_dims >= 3) {
                c->moe_intermediate_size = (int)f->tensors[ti].dims[1];
            }
        }

        // Shared expert
        c->has_shared_expert = (bn_gguf_find_tensor(f, "blk.0.ffn_gate_shexp.weight") >= 0) ? 1 : 0;
        if (c->has_shared_expert) {
            snprintf(key, sizeof(key), "%s.expert_shared_feed_forward_length", prefix);
            c->shared_expert_intermediate_size = (int)bn_gguf_get_u32(f, key);
            if (c->shared_expert_intermediate_size == 0) {
                // Fallback: infer from tensor dims
                int ti = bn_gguf_find_tensor(f, "blk.0.ffn_gate_shexp.weight");
                if (ti >= 0 && f->tensors[ti].n_dims >= 2)
                    c->shared_expert_intermediate_size = (int)f->tensors[ti].dims[1];
            }
        }

        if (c->n_experts_active <= 0 || c->moe_intermediate_size <= 0) {
            SH_LOG_ERROR("Invalid MoE config: n_experts_active and moe_intermediate_size must be > 0");
            return -1;
        }

        {
            char ne_s[16], ka_s[16], mi_s[16];
            snprintf(ne_s, sizeof(ne_s), "%d", c->n_experts);
            snprintf(ka_s, sizeof(ka_s), "%d", c->n_experts_active);
            snprintf(mi_s, sizeof(mi_s), "%d", c->moe_intermediate_size);
            SH_LOG_INFO("MoE config", "experts", ne_s, "active", ka_s, "expert_hidden", mi_s);
        }
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
        if (c->full_attn_interval > 0) {
            char fai_s[16], ssm_s[16];
            snprintf(fai_s, sizeof(fai_s), "%d", c->full_attn_interval);
            snprintf(ssm_s, sizeof(ssm_s), "%d", c->ssm_inner_size);
            SH_LOG_DEBUG("Hybrid SSM+Attn", "attn_interval", fai_s,
                         "ssm_inner", ssm_s);
        }
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
        if (out_info->dims[1] > INT_MAX || out_info->dims[0] > INT_MAX) {
            SH_LOG_ERROR("output.weight dimensions exceed INT_MAX");
            return -1;
        }
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

        // Determine layer type for hybrid models
        int is_ssm = (c->full_attn_interval > 0) &&
                     ((i + 1) % c->full_attn_interval != 0);

        // #25: Attention norms — must exist
        snprintf(wname, sizeof(wname), "blk.%d.attn_norm.weight", i);
        lw->attn_norm = load_f32_tensor(f, wname);
        if (!lw->attn_norm) {
            SH_LOG_ERROR("Tensor not found", "name", wname);
            goto fail_layers;
        }

        snprintf(wname, sizeof(wname), "blk.%d.attn_sub_norm.weight", i);
        lw->attn_sub_norm = load_f32_tensor(f, wname);  // optional

        if (is_ssm) {
            // --- SSM layer weights ---
            snprintf(wname, sizeof(wname), "blk.%d.attn_qkv.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.attn_qkv.scale", i);
            if (load_qweight(&lw->wqkv, f, wname, sname) != 0) goto fail_layers;

            snprintf(wname, sizeof(wname), "blk.%d.attn_gate.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.attn_gate.scale", i);
            if (load_qweight(&lw->wz, f, wname, sname) != 0) goto fail_layers;

            snprintf(wname, sizeof(wname), "blk.%d.ssm_a", i);
            lw->ssm_a = load_f32_tensor(f, wname);

            snprintf(wname, sizeof(wname), "blk.%d.ssm_alpha.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.ssm_alpha.scale", i);
            if (load_qweight(&lw->ssm_alpha, f, wname, sname) != 0) goto fail_layers;

            snprintf(wname, sizeof(wname), "blk.%d.ssm_beta.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.ssm_beta.scale", i);
            if (load_qweight(&lw->ssm_beta, f, wname, sname) != 0) goto fail_layers;

            snprintf(wname, sizeof(wname), "blk.%d.ssm_conv1d.weight", i);
            lw->ssm_conv1d = load_f32_tensor(f, wname);

            snprintf(wname, sizeof(wname), "blk.%d.ssm_dt.bias", i);
            lw->ssm_dt_bias = load_f32_tensor(f, wname);

            snprintf(wname, sizeof(wname), "blk.%d.ssm_norm.weight", i);
            lw->ssm_norm = load_f32_tensor(f, wname);

            snprintf(wname, sizeof(wname), "blk.%d.ssm_out.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.ssm_out.scale", i);
            if (load_qweight(&lw->ssm_out, f, wname, sname) != 0) goto fail_layers;
        } else {
            // --- Attention layer weights ---
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

            // Q/K norms (Qwen3.5 / OLMoE attention)
            snprintf(wname, sizeof(wname), "blk.%d.attn_q_norm.weight", i);
            lw->q_norm = load_f32_tensor(f, wname);
            snprintf(wname, sizeof(wname), "blk.%d.attn_k_norm.weight", i);
            lw->k_norm = load_f32_tensor(f, wname);

            // Detect per-head vs shared norms (layer 0 only)
            if (i == 0 && lw->q_norm) {
                snprintf(wname, sizeof(wname), "blk.0.attn_q_norm.weight");
                int qi = bn_gguf_find_tensor(f, wname);
                if (qi >= 0 && f->tensors[qi].dims[0] == (uint64_t)c->dim)
                    c->qk_norm_per_head = 1;
            }
        }

        // #25: FFN norms — must exist
        snprintf(wname, sizeof(wname), "blk.%d.ffn_norm.weight", i);
        lw->ffn_norm = load_f32_tensor(f, wname);
        if (!lw->ffn_norm) {
            // Qwen3.5 uses post_attention_norm instead of ffn_norm
            snprintf(wname, sizeof(wname), "blk.%d.post_attention_norm.weight", i);
            lw->ffn_norm = load_f32_tensor(f, wname);
        }
        if (!lw->ffn_norm) {
            SH_LOG_ERROR("FFN norm not found for layer");
            goto fail_layers;
        }

        snprintf(wname, sizeof(wname), "blk.%d.ffn_sub_norm.weight", i);
        lw->ffn_sub_norm = load_f32_tensor(f, wname);  // optional

        // FFN weights: MoE or dense
        if (c->n_experts > 0) {
            // --- MoE layer: router + expert offsets + shared expert ---

            // Router weight: [n_experts, dim] F32 — always resident
            snprintf(wname, sizeof(wname), "blk.%d.ffn_gate_inp.weight", i);
            lw->router_weight = (float *)load_f32_tensor(f, wname);
            if (!lw->router_weight) {
                SH_LOG_ERROR("Router weight not found", "name", wname);
                goto fail_layers;
            }

            // Expert tensor offsets (NOT loaded into memory)
            BnMoEExpertMap *em = &lw->expert_map;

            snprintf(wname, sizeof(wname), "blk.%d.ffn_gate_exps.weight", i);
            if (load_expert_map_proj(f, wname, c->n_experts,
                    &em->gate_type, &em->gate_rows, &em->gate_cols,
                    &em->gate_offset, &em->expert_gate_bytes) != 0) goto fail_layers;

            snprintf(wname, sizeof(wname), "blk.%d.ffn_up_exps.weight", i);
            if (load_expert_map_proj(f, wname, c->n_experts,
                    &em->up_type, &em->up_rows, &em->up_cols,
                    &em->up_offset, &em->expert_up_bytes) != 0) goto fail_layers;

            snprintf(wname, sizeof(wname), "blk.%d.ffn_down_exps.weight", i);
            if (load_expert_map_proj(f, wname, c->n_experts,
                    &em->down_type, &em->down_rows, &em->down_cols,
                    &em->down_offset, &em->expert_down_bytes) != 0) goto fail_layers;

            // Shared expert (optional, always resident)
            if (c->has_shared_expert) {
                snprintf(wname, sizeof(wname), "blk.%d.ffn_gate_shexp.weight", i);
                snprintf(sname, sizeof(sname), "blk.%d.ffn_gate_shexp.scale", i);
                if (load_qweight(&lw->shared_gate, f, wname, sname) != 0) goto fail_layers;

                snprintf(wname, sizeof(wname), "blk.%d.ffn_up_shexp.weight", i);
                snprintf(sname, sizeof(sname), "blk.%d.ffn_up_shexp.scale", i);
                if (load_qweight(&lw->shared_up, f, wname, sname) != 0) goto fail_layers;

                snprintf(wname, sizeof(wname), "blk.%d.ffn_down_shexp.weight", i);
                snprintf(sname, sizeof(sname), "blk.%d.ffn_down_shexp.scale", i);
                if (load_qweight(&lw->shared_down, f, wname, sname) != 0) goto fail_layers;

                // Shared expert sigmoid gate (optional, Qwen3.5 MoE)
                snprintf(wname, sizeof(wname), "blk.%d.ffn_gate_inp_shexp.weight", i);
                lw->shared_expert_gate = load_f32_tensor(f, wname);
            }
        } else {
            // --- Dense FFN ---
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
    }

    // --- Allocate BnRunState via arena ---
    // #1, #14: Check all allocations and guard against overflow
    BnRunState *s = &m->state;

    size_t att_size = (size_t)c->n_heads * c->seq_len;
    if (att_size / c->n_heads != (size_t)c->seq_len) {
        SH_LOG_ERROR("Attention buffer size overflow");
        goto fail_state;
    }

    // Only attention layers need KV cache
    int n_attn_layers = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval : c->n_layers;
    int n_ssm_layers = c->n_layers - n_attn_layers;

    // #14: Check for overflow before large KV cache allocations
    size_t kv_cache_size = (size_t)n_attn_layers * c->seq_len * c->kv_dim;
    if (n_attn_layers > 0 && c->seq_len > 0 && c->kv_dim > 0 &&
        kv_cache_size / n_attn_layers / c->seq_len != (size_t)c->kv_dim) {
        SH_LOG_ERROR("KV cache size overflow");
        goto fail_state;
    }

    // q_dim = n_heads * head_size (may differ from dim when attention.key_length is set)
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;  // xb must hold attention output
    int q_size = xb_size;  // q must match xb for attention head access pattern

    int x_q_size = c->dim > c->hidden_dim ? c->dim : c->hidden_dim;
    if (q_dim > x_q_size) x_q_size = q_dim;
    int half_head = c->head_size / 2;

    // Scratch buffer sizes — enlarged for hybrid SSM + gated-Q attention
    int hb_size = c->hidden_dim;
    int hb2_size = c->hidden_dim;
    int xb2_size = c->dim;
    if (c->full_attn_interval > 0) {
        // SSM: qkv projection → hb, z gate → hb2, recurrence output → xb2
        int qkv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        if (qkv_dim > hb_size) hb_size = qkv_dim;
        if (c->ssm_inner_size > hb2_size) hb2_size = c->ssm_inner_size;
        if (c->ssm_inner_size > xb2_size) xb2_size = c->ssm_inner_size;
        if (c->ssm_inner_size > x_q_size) x_q_size = c->ssm_inner_size;
        // Gated Q attention: q_full (Q + gate) → hb
        int gq = 2 * q_dim;
        if (gq > hb_size) hb_size = gq;
    }
    // MoE shared expert may need larger hb/hb2 buffers
    if (c->has_shared_expert && c->shared_expert_intermediate_size > hb_size)
        hb_size = c->shared_expert_intermediate_size;
    if (c->has_shared_expert && c->shared_expert_intermediate_size > hb2_size)
        hb2_size = c->shared_expert_intermediate_size;
    // MoE expert intermediate for x_q scratch
    if (c->n_experts > 0 && c->moe_intermediate_size > x_q_size)
        x_q_size = c->moe_intermediate_size;

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

    // Q4_0 weight repacking size (NEON SDOT only)
    size_t q4_repack_total = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int i = 0; i < c->n_layers; i++) {
        BnLayerWeights *lw = &w->layers[i];
        q4_repack_total += q4_repack_bytes(&lw->wq);
        q4_repack_total += q4_repack_bytes(&lw->wk);
        q4_repack_total += q4_repack_bytes(&lw->wv);
        q4_repack_total += q4_repack_bytes(&lw->wo);
        q4_repack_total += q4_repack_bytes(&lw->wqkv);
        q4_repack_total += q4_repack_bytes(&lw->wz);
        q4_repack_total += q4_repack_bytes(&lw->ssm_out);
        q4_repack_total += q4_repack_bytes(&lw->ffn_gate);
        q4_repack_total += q4_repack_bytes(&lw->ffn_up);
        q4_repack_total += q4_repack_bytes(&lw->ffn_down);
    }
    q4_repack_total += q4_repack_bytes(&w->output_weight);
#endif

    // SSM state sizing
    size_t ssm_state_size_total = 0;
    size_t ssm_conv_state_total = 0;
    if (n_ssm_layers > 0 && c->ssm_time_step_rank > 0) {
        int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
        size_t state_per_layer = (size_t)c->ssm_time_step_rank *
                                  c->ssm_state_size * head_v_dim;
        ssm_state_size_total = (size_t)n_ssm_layers * state_per_layer;
        int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        ssm_conv_state_total = (size_t)n_ssm_layers *
                                (c->ssm_conv_kernel - 1) * conv_dim;
    }

    // MoE buffer sizing
    size_t moe_arena_bytes = 0;
    size_t moe_expert_buf_size = 0;
    if (c->n_experts > 0) {
        // Determine max expert projection size from layer 0
        BnMoEExpertMap *em0 = &w->layers[0].expert_map;
        moe_expert_buf_size = em0->expert_gate_bytes;
        if (em0->expert_up_bytes > moe_expert_buf_size)
            moe_expert_buf_size = em0->expert_up_bytes;
        if (em0->expert_down_bytes > moe_expert_buf_size)
            moe_expert_buf_size = em0->expert_down_bytes;

        moe_arena_bytes += sizeof(BnMoEState);                                    // state struct
        moe_arena_bytes += (size_t)c->n_experts * sizeof(float);                  // router_logits
        moe_arena_bytes += (size_t)c->dim * sizeof(float);                        // expert_out
        moe_arena_bytes += (size_t)c->n_experts_active * sizeof(float);           // expert_weights
        moe_arena_bytes += (size_t)c->n_experts_active * sizeof(int);             // expert_indices
        moe_arena_bytes += (size_t)c->moe_intermediate_size * sizeof(float);      // expert_hb
        moe_arena_bytes += (size_t)c->moe_intermediate_size * sizeof(float);      // expert_hb2
        moe_arena_bytes += moe_expert_buf_size;                                    // expert_buf (pread)
        moe_arena_bytes += moe_expert_buf_size;                                    // expert_buf2 (pread double-buffer)
        moe_arena_bytes += moe_expert_buf_size;                                    // expert_buf3 (prefetch gate)
        moe_arena_bytes += moe_expert_buf_size;                                    // expert_buf4 (prefetch up)
        // Batch buffers for cross-expert dispatch (mmap path)
        int moe_k = c->n_experts_active;
        if (moe_k > BN_MAX_MOE_K) moe_k = BN_MAX_MOE_K;
        moe_arena_bytes += (size_t)moe_k * c->moe_intermediate_size * sizeof(float); // hb_batch
        moe_arena_bytes += (size_t)moe_k * c->moe_intermediate_size * sizeof(float); // hb2_batch
        moe_arena_bytes += (size_t)moe_k * c->dim * sizeof(float);                   // down_batch
        moe_arena_bytes += (12 + 3 * moe_k) * SH_ARENA_ALIGN;                       // alignment
    }

    // Compute total arena capacity (all RunState buffers + INT8 embeddings + Q4 repacking)
    size_t arena_size = 0;
    arena_size += ((size_t)c->dim + (size_t)xb_size + (size_t)xb2_size + (size_t)q_size) * sizeof(float); // x, xb, xb2, q
    arena_size += ((size_t)hb_size + (size_t)hb2_size) * sizeof(float);    // hb, hb2
    arena_size += att_size * sizeof(float);                     // att
    arena_size += (size_t)c->vocab_size * sizeof(float);       // logits
    size_t kv_elem_size = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    arena_size += 2 * kv_cache_size * kv_elem_size;           // key_cache, value_cache
    arena_size += (size_t)x_q_size * sizeof(int8_t);           // x_q
    arena_size += (size_t)half_head * sizeof(float);           // rope_freq
    arena_size += (ssm_state_size_total + ssm_conv_state_total) * sizeof(float); // SSM state
    arena_size += emb_i8_bytes + emb_i8_scales_bytes;          // INT8 embeddings
    arena_size += q4_repack_total;                              // Q4_0 repacked weights
    arena_size += moe_arena_bytes;                              // MoE buffers
    arena_size += 16 * SH_ARENA_ALIGN;                         // alignment padding

    if (arena_size > SIZE_MAX / 2) {
        SH_LOG_ERROR("Arena size overflow");
        goto fail_state;
    }

    m->arena = sh_arena_create(arena_size);
    if (!m->arena) {
        SH_LOG_ERROR("Failed to allocate run state arena");
        goto fail_state;
    }

    s->x           = (float *)sh_arena_calloc(m->arena, c->dim, sizeof(float));
    s->xb          = (float *)sh_arena_calloc(m->arena, xb_size, sizeof(float));
    s->xb2         = (float *)sh_arena_calloc(m->arena, xb2_size, sizeof(float));
    s->q           = (float *)sh_arena_calloc(m->arena, q_size, sizeof(float));
    s->hb          = (float *)sh_arena_calloc(m->arena, hb_size, sizeof(float));
    s->hb2         = (float *)sh_arena_calloc(m->arena, hb2_size, sizeof(float));
    s->att         = (float *)sh_arena_calloc(m->arena, att_size, sizeof(float));
    s->logits      = (float *)sh_arena_calloc(m->arena, c->vocab_size, sizeof(float));
    s->key_cache   = (float *)sh_arena_calloc(m->arena, kv_cache_size, kv_elem_size);
    s->value_cache = (float *)sh_arena_calloc(m->arena, kv_cache_size, kv_elem_size);
    s->x_q         = (int8_t *)sh_arena_calloc(m->arena, x_q_size, sizeof(int8_t));
    s->rope_freq   = (float *)sh_arena_alloc(m->arena, half_head * sizeof(float));

    // SSM state buffers
    s->ssm_state = NULL;
    s->ssm_conv_state = NULL;
    if (ssm_state_size_total > 0) {
        s->ssm_state = (float *)sh_arena_calloc(m->arena, ssm_state_size_total, sizeof(float));
        s->ssm_conv_state = (float *)sh_arena_calloc(m->arena, ssm_conv_state_total, sizeof(float));
    }

    // #1: Check all allocations succeeded
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache ||
        !s->x_q || !s->rope_freq) {
        SH_LOG_ERROR("Failed to allocate run state buffers");
        goto fail_state;
    }
    if (ssm_state_size_total > 0 && (!s->ssm_state || !s->ssm_conv_state)) {
        SH_LOG_ERROR("Failed to allocate SSM state buffers");
        goto fail_state;
    }

    // MoE state buffers
    m->moe_state = NULL;
    m->expert_fd = -1;
    if (c->n_experts > 0) {
        BnMoEState *ms = (BnMoEState *)sh_arena_calloc(m->arena, 1, sizeof(BnMoEState));
        if (!ms) {
            SH_LOG_ERROR("Failed to allocate MoE state");
            goto fail_state;
        }
        ms->io.fd = -1;  // will be set from BnMappedFile.fd after model.file is assigned
        ms->router_logits  = (float *)sh_arena_calloc(m->arena, c->n_experts, sizeof(float));
        ms->expert_out     = (float *)sh_arena_calloc(m->arena, c->dim, sizeof(float));
        ms->expert_weights = (float *)sh_arena_calloc(m->arena, c->n_experts_active, sizeof(float));
        ms->expert_indices = (int *)sh_arena_calloc(m->arena, c->n_experts_active, sizeof(int));
        ms->expert_hb      = (float *)sh_arena_calloc(m->arena, c->moe_intermediate_size, sizeof(float));
        ms->expert_hb2     = (float *)sh_arena_calloc(m->arena, c->moe_intermediate_size, sizeof(float));
        ms->io.buf      = (uint8_t *)sh_arena_alloc(m->arena, moe_expert_buf_size);
        ms->io.buf_size  = moe_expert_buf_size;
        ms->io.buf2     = (uint8_t *)sh_arena_alloc(m->arena, moe_expert_buf_size);
        ms->io.buf2_size = moe_expert_buf_size;
        ms->io.buf3     = (uint8_t *)sh_arena_alloc(m->arena, moe_expert_buf_size);
        ms->io.buf3_size = moe_expert_buf_size;
        ms->io.buf4     = (uint8_t *)sh_arena_alloc(m->arena, moe_expert_buf_size);
        ms->io.buf4_size = moe_expert_buf_size;
        ms->io.prefetch = NULL;

        if (!ms->router_logits || !ms->expert_out || !ms->expert_weights ||
            !ms->expert_indices || !ms->expert_hb || !ms->expert_hb2 ||
            !ms->io.buf || !ms->io.buf2 ||
            !ms->io.buf3 || !ms->io.buf4) {
            SH_LOG_ERROR("Failed to allocate MoE buffers");
            goto fail_state;
        }

        // Batch buffers for cross-expert dispatch
        int moe_k = c->n_experts_active;
        if (moe_k > BN_MAX_MOE_K) moe_k = BN_MAX_MOE_K;
        for (int k = 0; k < moe_k; k++) {
            ms->expert_hb_batch[k]   = (float *)sh_arena_calloc(m->arena, c->moe_intermediate_size, sizeof(float));
            ms->expert_hb2_batch[k]  = (float *)sh_arena_calloc(m->arena, c->moe_intermediate_size, sizeof(float));
            ms->expert_down_batch[k] = (float *)sh_arena_calloc(m->arena, c->dim, sizeof(float));
            if (!ms->expert_hb_batch[k] || !ms->expert_hb2_batch[k] || !ms->expert_down_batch[k]) {
                SH_LOG_ERROR("Failed to allocate MoE batch buffers");
                goto fail_state;
            }
        }

        m->moe_state = ms;

        {
            char buf_s[16];
            snprintf(buf_s, sizeof(buf_s), "%.1f", (double)moe_expert_buf_size / (1024 * 1024));
            SH_LOG_INFO("MoE state allocated", "expert_buf_MB", buf_s);
        }
    }

    // Precompute RoPE frequencies: freq[i] = 1/theta^(2i/rope_dims)
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : c->head_size;
    int half_rope = rope_dims / 2;
    for (int i = 0; i < half_rope; i++) {
        s->rope_freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)rope_dims);
    }
    // MROPE text-only: zero out non-text section frequencies.
    // Sections 1,2 (vision height/width) get position=0 for text tokens → no rotation.
    if (c->rope_text_dims > 0) {
        int text_pairs = c->rope_text_dims / 2;
        for (int i = text_pairs; i < half_rope; i++)
            s->rope_freq[i] = 0.0f;
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

    // Repack Q4_0 weights into split scales/qs layout for NEON SDOT
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (q4_repack_total > 0) {
        for (int i = 0; i < c->n_layers; i++) {
            BnLayerWeights *lw = &w->layers[i];
            q4_repack(&lw->wq, m->arena);
            q4_repack(&lw->wk, m->arena);
            q4_repack(&lw->wv, m->arena);
            q4_repack(&lw->wo, m->arena);
            q4_repack(&lw->wqkv, m->arena);
            q4_repack(&lw->wz, m->arena);
            q4_repack(&lw->ssm_out, m->arena);
            q4_repack(&lw->ffn_gate, m->arena);
            q4_repack(&lw->ffn_up, m->arena);
            q4_repack(&lw->ffn_down, m->arena);
        }
        q4_repack(&w->output_weight, m->arena);
        char rp_mb[16]; snprintf(rp_mb, sizeof(rp_mb), "%.0f", (double)q4_repack_total / (1024*1024));
        SH_LOG_INFO("Q4_0 weights repacked", "MB", rp_mb);
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

void bn_model_reset_state(BnModel *m) {
    if (!m) return;
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;

    // KV cache
    int n_attn = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval : c->n_layers;
    size_t kv_size = (size_t)n_attn * c->seq_len * c->kv_dim;
    size_t kv_elem = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    memset(s->key_cache, 0, kv_size * kv_elem);
    memset(s->value_cache, 0, kv_size * kv_elem);

    // SSM state
    if (s->ssm_state && c->ssm_time_step_rank > 0) {
        int n_ssm = c->n_layers - n_attn;
        int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
        size_t state_total = (size_t)n_ssm * c->ssm_time_step_rank *
                             c->ssm_state_size * head_v_dim;
        memset(s->ssm_state, 0, state_total * sizeof(float));
    }
    if (s->ssm_conv_state) {
        int n_ssm = c->n_layers - n_attn;
        int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        size_t conv_total = (size_t)n_ssm * (c->ssm_conv_kernel - 1) * conv_dim;
        memset(s->ssm_conv_state, 0, conv_total * sizeof(float));
    }
}

void bn_model_free(BnModel *m) {
    if (!m) return;
    if (m->moe_state) bn_moe_prefetch_destroy(m->moe_state);
    bn_tp_free(m->pool);
    free(m->weights.layers);
    sh_arena_free(m->arena);  // frees MoE state, INT8 embeddings too
    bn_platform_unload_file(&m->file);  // safe even if file.data is NULL, also closes fd
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
