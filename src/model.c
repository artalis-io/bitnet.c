#include "model.h"
#include "backend_model.h"
#include "model_arch.h"
#include "moe.h"
#include "sh_arena.h"
#include "sh_log.h"
#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

// --- Helper: load a BnQWeight from GGUF tensor + scale tensor ---

static int qweight_type_supported(int type) {
    return bn_quant_format_supported(type);
}

static int qweight_type_uses_embedded_scale(int type) {
    return bn_quant_format_uses_embedded_scale(type);
}

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

    if (!qweight_type_supported(w->type)) {
        SH_LOG_ERROR("Unsupported tensor type", "name", weight_name);
        return -1;
    }

    if (w->type == BN_GGUF_TENSOR_I2_S) {
        // I2_S: per-tensor scale stored at end of packed data (offset = nelements/4)
        size_t nelements = (size_t)w->rows * w->cols;
        const uint8_t *base = (const uint8_t *)w->data;
        memcpy(&w->scale, base + nelements / 4, sizeof(float));
    } else if (qweight_type_uses_embedded_scale(w->type)) {
        // Per-block scales are embedded for quantized types; plain float types need no scale.
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

#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__wasm_relaxed_simd__)
static size_t q4_repack_bytes(const BnQWeight *w) {
    if (w->type != BN_GGUF_TENSOR_Q4_0 || !w->data) return 0;
    if (w->rows % 4 != 0) return 0;  // need complete 4-row groups
    size_t n_blocks = (size_t)w->rows * (w->cols / 32);
    size_t bytes = n_blocks * 16 + SH_ARENA_ALIGN;
#ifdef __wasm_relaxed_simd__
    bytes += n_blocks * sizeof(float) + SH_ARENA_ALIGN;
#else
    bytes += n_blocks * sizeof(uint16_t) + SH_ARENA_ALIGN;
#endif
    return bytes;
}

static void q4_repack(BnBackendModel *backend, const BnQWeight *w, SHArena *arena) {
    if (w->type != BN_GGUF_TENSOR_Q4_0 || !w->data) return;
    if (w->rows % 4 != 0) return;  // need complete 4-row groups
    int n_blocks_per_row = w->cols / 32;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;
    BnPreparedWeight prepared = { 0 };

    prepared.qs = (uint8_t *)sh_arena_alloc(arena, n_blocks * 16);
#ifdef __wasm_relaxed_simd__
    prepared.f32_scales = (float *)sh_arena_alloc(arena, n_blocks * sizeof(float));
#else
    prepared.scales = (uint16_t *)sh_arena_alloc(arena, n_blocks * sizeof(uint16_t));
#endif
    if (!prepared.qs) return;
#ifdef __wasm_relaxed_simd__
    if (!prepared.f32_scales) return;
#else
    if (!prepared.scales) return;
#endif

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
#ifdef __wasm_relaxed_simd__
                prepared.f32_scales[gb * 4 + r] = bn_fp16_to_fp32(blocks[src].d);
#else
                prepared.scales[gb * 4 + r] = blocks[src].d;
#endif
            }
            // Quants: nibble-transpose within 64-byte chunk
            uint8_t *dst = prepared.qs + gb * 64;
            for (int ng = 0; ng < 4; ng++) {
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) * n_blocks_per_row + b;
                    const uint8_t *qs = blocks[src].qs + ng * 4;
                    uint8_t *dp = dst + ng * 16 + r * 4;
                    for (int j = 0; j < 4; j++)
                        dp[j] = qs[j] ^ 0x88;
                }
            }
        }
    }
    (void)bn_backend_model_register_prepared_qweight(backend, w, &prepared);
}
#endif

#ifdef __wasm_relaxed_simd__
static size_t q8_f32_scale_bytes(const BnQWeight *w) {
    if (w->type != BN_GGUF_TENSOR_Q8_0 || !w->data) return 0;
    if ((w->cols & 31) != 0) return 0;
    size_t n_blocks = (size_t)w->rows * (w->cols / 32);
    return n_blocks * sizeof(float) + SH_ARENA_ALIGN;
}

static void q8_prepare_f32_scales(BnBackendModel *backend, const BnQWeight *w,
                                  SHArena *arena) {
    if (w->type != BN_GGUF_TENSOR_Q8_0 || !w->data) return;
    if ((w->cols & 31) != 0) return;
    int n_blocks_per_row = w->cols / 32;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;
    BnPreparedWeight prepared = { 0 };
    prepared.f32_scales = (float *)sh_arena_alloc(arena, n_blocks * sizeof(float));
    if (!prepared.f32_scales) return;

    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)w->data;
    for (size_t i = 0; i < n_blocks; i++)
        prepared.f32_scales[i] = bn_fp16_to_fp32(blocks[i].d);
    (void)bn_backend_model_register_prepared_qweight(backend, w, &prepared);
}
#endif

// --- Helper: load F32 norm weights from GGUF ---

static float *load_f32_tensor(BnGGUFFile *f, const char *name) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0) return NULL;
    return (float *)bn_gguf_tensor_data(f, ti);
}

static int gguf_get_u32_or_i32_array(BnGGUFFile *f, const char *key, int idx) {
    int ki = bn_gguf_find_key(f, key);
    if (ki < 0) return 0;
    BnGGUFKeyValue *kv = &f->kvs[ki];
    if (kv->type == BN_GGUF_TYPE_UINT32) return (int)kv->value.u32;
    if (kv->type == BN_GGUF_TYPE_INT32) return kv->value.i32;
    if (kv->type != BN_GGUF_TYPE_ARRAY || idx < 0) return 0;
    BnGGUFArray *a = &kv->value.arr;
    if ((uint64_t)idx >= a->n || !a->data) return 0;
    if (a->elem_type == BN_GGUF_TYPE_INT32)
        return ((const int32_t *)a->data)[idx];
    if (a->elem_type == BN_GGUF_TYPE_UINT32)
        return (int)((const uint32_t *)a->data)[idx];
    return 0;
}

static int tensor_dim0(BnGGUFFile *f, const char *name) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0 || f->tensors[ti].n_dims < 1 || f->tensors[ti].dims[0] > INT_MAX)
        return 0;
    return (int)f->tensors[ti].dims[0];
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
    if (info->dims[0] > INT_MAX || info->dims[1] > INT_MAX || info->dims[2] > INT_MAX) {
        SH_LOG_ERROR("Expert tensor dimensions exceed INT_MAX", "name", name);
        return -1;
    }
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
    size_t expert_elements = 0;
    if (checked_mul_size((size_t)rows, (size_t)cols, &expert_elements) != 0 ||
        !bn_gguf_tensor_size(info->type, (uint64_t)expert_elements, expert_bytes_out)) {
        SH_LOG_ERROR("Unsupported expert tensor type", "name", name);
        return -1;
    }

    return 0;
}

static int load_expert_map_gate_up_fused(BnGGUFFile *f, const char *name,
                                         int n_experts, int expert_hidden,
                                         BnMoEExpertMap *em) {
    int ti = bn_gguf_find_tensor(f, name);
    if (ti < 0) return -1;
    BnGGUFTensorInfo *info = &f->tensors[ti];
    if (info->n_dims < 3 || info->dims[0] > INT_MAX ||
        info->dims[1] > INT_MAX || info->dims[2] > INT_MAX)
        return -1;

    int cols = (int)info->dims[0];
    int rows = (int)info->dims[1];
    int n_exp = (int)info->dims[2];
    if (n_exp != n_experts || rows != expert_hidden * 2) {
        SH_LOG_ERROR("Fused gate/up expert tensor shape mismatch", "name", name);
        return -1;
    }

    size_t one_proj_elems = 0, fused_elems = 0;
    size_t one_proj_bytes = 0, fused_bytes = 0;
    if (checked_mul_size((size_t)expert_hidden, (size_t)cols, &one_proj_elems) != 0 ||
        checked_mul_size((size_t)rows, (size_t)cols, &fused_elems) != 0 ||
        !bn_gguf_tensor_size(info->type, (uint64_t)one_proj_elems, &one_proj_bytes) ||
        !bn_gguf_tensor_size(info->type, (uint64_t)fused_elems, &fused_bytes))
        return -1;

    em->gate_type = (int)info->type;
    em->up_type = (int)info->type;
    em->gate_rows = expert_hidden;
    em->up_rows = expert_hidden;
    em->gate_cols = cols;
    em->up_cols = cols;
    em->expert_gate_bytes = one_proj_bytes;
    em->expert_up_bytes = one_proj_bytes;
    em->gate_stride = fused_bytes;
    em->up_stride = fused_bytes;
    em->gate_offset = f->data_offset + info->offset;
    em->up_offset = em->gate_offset + one_proj_bytes;
    return 0;
}

// --- Model loading ---

int bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len, int kv_f16, int kv_tq_bits) {
    memset(m, 0, sizeof(BnModel));
    m->backend = bn_backend_model_create();
    if (!m->backend) return -1;
    BnConfig *c = &m->config;
    c->kv_f16 = kv_f16;
    c->kv_tq_bits = kv_tq_bits;

    // Try to detect architecture prefix
    const char *arch = bn_gguf_get_str(f, "general.architecture");
    const BnModelArchOps *arch_ops = bn_model_arch_ops_for(arch);
    if (!arch_ops) {
        SH_LOG_ERROR("No model architecture ops registered");
        return -1;
    }
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%s", arch_ops->prefix(arch));
    c->arch_flags = arch_ops->flags;

    // Build key names with architecture prefix
    char key[128];

    snprintf(key, sizeof(key), "%s.embedding_length", prefix);
    c->dim = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.feed_forward_length", prefix);
    c->hidden_dim = gguf_get_u32_or_i32_array(f, key, 0);

    snprintf(key, sizeof(key), "%s.block_count", prefix);
    c->n_layers = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.attention.head_count", prefix);
    c->n_heads = (int)bn_gguf_get_u32(f, key);

    snprintf(key, sizeof(key), "%s.attention.head_count_kv", prefix);
    c->n_kv_heads = gguf_get_u32_or_i32_array(f, key, 0);
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
    int max_head_size = c->head_size;
    int max_kv_dim = c->kv_dim;
    int max_q_dim = c->n_heads * c->head_size;

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
            c->rope_text_dims =
                bn_model_arch_rope_text_dims(c->rope_dim_count, sections, nsect);
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
    bn_model_arch_load_moe_config(c, f, arch_ops, prefix);
    if (c->n_experts > 0) {
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
    {
        char ffn_gate_name[128];
        if (bn_model_arch_tensor_name_for(arch_ops, ffn_gate_name,
                                          sizeof(ffn_gate_name), 0,
                                          BN_MODEL_TENSOR_FFN_GATE) != 0)
            return -1;
        c->has_ffn_gate = (bn_gguf_find_tensor(f, ffn_gate_name) >= 0) ? 1 : 0;
    }

    // Check for activation type: bitnet uses ReLU² (act_type=1)
    c->act_type = arch_ops->activation(arch);

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
        if (!qweight_type_supported(ot)) {
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
        if (qweight_type_uses_embedded_scale(ot)) {
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
        int is_ssm = arch_ops->is_ssm_layer(c, i);

        // #25: Attention norms — must exist
        if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                          BN_MODEL_TENSOR_ATTN_NORM) != 0)
            goto fail_layers;
        lw->attn_norm = load_f32_tensor(f, wname);
        if (!lw->attn_norm) {
            SH_LOG_ERROR("Tensor not found", "name", wname);
            goto fail_layers;
        }

        if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                          BN_MODEL_TENSOR_ATTN_SUB_NORM) != 0)
            goto fail_layers;
        lw->attn_sub_norm = load_f32_tensor(f, wname);  // optional

        if (is_ssm) {
            // --- SSM layer weights ---
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_QKV) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_SSM_QKV) != 0)
                goto fail_layers;
            if (load_qweight(&lw->wqkv, f, wname, sname) != 0) goto fail_layers;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_GATE) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_SSM_GATE) != 0)
                goto fail_layers;
            if (load_qweight(&lw->wz, f, wname, sname) != 0) goto fail_layers;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_A) != 0)
                goto fail_layers;
            lw->ssm_a = load_f32_tensor(f, wname);

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_ALPHA) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_SSM_ALPHA) != 0)
                goto fail_layers;
            if (load_qweight(&lw->ssm_alpha, f, wname, sname) != 0) goto fail_layers;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_BETA) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_SSM_BETA) != 0)
                goto fail_layers;
            if (load_qweight(&lw->ssm_beta, f, wname, sname) != 0) goto fail_layers;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_CONV1D) != 0)
                goto fail_layers;
            lw->ssm_conv1d = load_f32_tensor(f, wname);

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_DT_BIAS) != 0)
                goto fail_layers;
            lw->ssm_dt_bias = load_f32_tensor(f, wname);

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_NORM) != 0)
                goto fail_layers;
            lw->ssm_norm = load_f32_tensor(f, wname);

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_SSM_OUT) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_SSM_OUT) != 0)
                goto fail_layers;
            if (load_qweight(&lw->ssm_out, f, wname, sname) != 0) goto fail_layers;
        } else {
            // --- Attention layer weights ---
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_Q) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_ATTN_Q) != 0)
                goto fail_layers;
            if (load_qweight(&lw->wq, f, wname, sname) != 0) goto fail_layers;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_K) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_ATTN_K) != 0)
                goto fail_layers;
            if (load_qweight(&lw->wk, f, wname, sname) != 0) goto fail_layers;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_V) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_ATTN_V) != 0)
                goto fail_layers;
            if (bn_gguf_find_tensor(f, wname) >= 0) {
                if (load_qweight(&lw->wv, f, wname, sname) != 0) goto fail_layers;
            } else if (arch_ops->attention_value_shares_key(arch)) {
                lw->wv = lw->wk;
            } else {
                SH_LOG_ERROR("Tensor not found", "name", wname);
                goto fail_layers;
            }

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_OUTPUT) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_ATTN_OUTPUT) != 0)
                goto fail_layers;
            if (load_qweight(&lw->wo, f, wname, sname) != 0) goto fail_layers;

            // Attention biases (optional, used by Qwen2)
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_Q_BIAS) != 0)
                goto fail_layers;
            lw->q_bias = load_f32_tensor(f, wname);
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_K_BIAS) != 0)
                goto fail_layers;
            lw->k_bias = load_f32_tensor(f, wname);
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_V_BIAS) != 0)
                goto fail_layers;
            lw->v_bias = load_f32_tensor(f, wname);

            // Q/K norms (Qwen3.5 / OLMoE attention)
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_Q_NORM) != 0)
                goto fail_layers;
            lw->q_norm = load_f32_tensor(f, wname);
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_ATTN_K_NORM) != 0)
                goto fail_layers;
            lw->k_norm = load_f32_tensor(f, wname);

            // Detect per-head vs shared norms (layer 0 only)
            if (i == 0 && lw->q_norm) {
                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), 0,
                                                  BN_MODEL_TENSOR_ATTN_Q_NORM) != 0)
                    goto fail_layers;
                int qi = bn_gguf_find_tensor(f, wname);
                if (qi >= 0 && f->tensors[qi].dims[0] == (uint64_t)c->dim)
                    c->qk_norm_per_head = 1;
            }

            lw->q_dim = lw->wq.rows;
            lw->head_size = c->head_size;
            if (lw->q_norm && !c->qk_norm_per_head) {
                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_ATTN_Q_NORM) != 0)
                    goto fail_layers;
                int hs = tensor_dim0(f, wname);
                if (hs > 0) lw->head_size = hs;
            } else if (!lw->q_norm && c->n_heads > 0 && lw->wq.rows > 0 &&
                       lw->wq.rows % c->n_heads == 0) {
                lw->head_size = lw->wq.rows / c->n_heads;
            }
            snprintf(key, sizeof(key), "%s.attention.head_count_kv", prefix);
            lw->n_kv_heads = gguf_get_u32_or_i32_array(f, key, i);
            if (lw->n_kv_heads <= 0 && lw->head_size > 0)
                lw->n_kv_heads = lw->wk.rows / lw->head_size;
            if (lw->n_kv_heads <= 0) lw->n_kv_heads = c->n_kv_heads;
            lw->kv_dim = lw->wk.rows > 0 ? lw->wk.rows : lw->n_kv_heads * lw->head_size;
            lw->kv_mul = (lw->n_kv_heads > 0) ? c->n_heads / lw->n_kv_heads : c->kv_mul;
            if (lw->head_size <= 0 || lw->kv_dim <= 0 || lw->kv_mul <= 0 ||
                c->n_heads % lw->n_kv_heads != 0) {
                SH_LOG_ERROR("Invalid per-layer attention dimensions");
                goto fail_layers;
            }
            if (lw->head_size > max_head_size) max_head_size = lw->head_size;
            if (lw->kv_dim > max_kv_dim) max_kv_dim = lw->kv_dim;
            if (lw->q_dim > max_q_dim) max_q_dim = lw->q_dim;
        }

        // #25: FFN norms — must exist
        if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                          BN_MODEL_TENSOR_FFN_NORM) != 0)
            goto fail_layers;
        lw->ffn_norm = load_f32_tensor(f, wname);
        if (!lw->ffn_norm) {
            // Qwen3.5 uses post_attention_norm instead of ffn_norm
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_FFN_POST_ATTN_NORM) != 0)
                goto fail_layers;
            lw->ffn_norm = load_f32_tensor(f, wname);
        }
        if (!lw->ffn_norm) {
            SH_LOG_ERROR("FFN norm not found for layer");
            goto fail_layers;
        }

        if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                          BN_MODEL_TENSOR_FFN_SUB_NORM) != 0)
            goto fail_layers;
        lw->ffn_sub_norm = load_f32_tensor(f, wname);  // optional

        // FFN weights: MoE or dense
        if (c->n_experts > 0) {
            // --- MoE layer: router + expert offsets + shared expert ---

            // Router weight: [n_experts, dim] F32 — always resident
            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_MOE_ROUTER) != 0)
                goto fail_layers;
            lw->router_weight = (float *)load_f32_tensor(f, wname);
            if (!lw->router_weight) {
                SH_LOG_ERROR("Router weight not found", "name", wname);
                goto fail_layers;
            }

            // Expert tensor offsets (NOT loaded into memory)
            BnMoEExpertMap *em = &lw->expert_map;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_MOE_GATE_EXPS) != 0)
                goto fail_layers;
            if (bn_gguf_find_tensor(f, wname) >= 0) {
                if (load_expert_map_proj(f, wname, c->n_experts,
                        &em->gate_type, &em->gate_rows, &em->gate_cols,
                        &em->gate_offset, &em->expert_gate_bytes) != 0) goto fail_layers;

                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_MOE_UP_EXPS) != 0)
                    goto fail_layers;
                if (load_expert_map_proj(f, wname, c->n_experts,
                        &em->up_type, &em->up_rows, &em->up_cols,
                        &em->up_offset, &em->expert_up_bytes) != 0) goto fail_layers;
            } else {
                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_MOE_GATE_UP_EXPS) != 0)
                    goto fail_layers;
                if (load_expert_map_gate_up_fused(f, wname, c->n_experts,
                        c->moe_intermediate_size, em) != 0) goto fail_layers;
            }

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_MOE_DOWN_EXPS) != 0)
                goto fail_layers;
            if (load_expert_map_proj(f, wname, c->n_experts,
                    &em->down_type, &em->down_rows, &em->down_cols,
                    &em->down_offset, &em->expert_down_bytes) != 0) goto fail_layers;
            em->down_stride = em->expert_down_bytes;

            // Shared expert (optional, always resident)
            if (c->has_shared_expert) {
                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_SHARED_FFN_GATE) != 0 ||
                    bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                        BN_MODEL_TENSOR_SHARED_FFN_GATE) != 0)
                    goto fail_layers;
                if (load_qweight(&lw->shared_gate, f, wname, sname) != 0) goto fail_layers;

                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_SHARED_FFN_UP) != 0 ||
                    bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                        BN_MODEL_TENSOR_SHARED_FFN_UP) != 0)
                    goto fail_layers;
                if (load_qweight(&lw->shared_up, f, wname, sname) != 0) goto fail_layers;

                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_SHARED_FFN_DOWN) != 0 ||
                    bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                        BN_MODEL_TENSOR_SHARED_FFN_DOWN) != 0)
                    goto fail_layers;
                if (load_qweight(&lw->shared_down, f, wname, sname) != 0) goto fail_layers;

                // Shared expert sigmoid gate (optional, Qwen3.5 MoE)
                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_SHARED_FFN_ROUTER) != 0)
                    goto fail_layers;
                lw->shared_expert_gate = load_f32_tensor(f, wname);
            }
        } else {
            // --- Dense FFN ---
            if (c->has_ffn_gate) {
                if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                                  BN_MODEL_TENSOR_FFN_GATE) != 0 ||
                    bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                        BN_MODEL_TENSOR_FFN_GATE) != 0)
                    goto fail_layers;
                if (load_qweight(&lw->ffn_gate, f, wname, sname) != 0) goto fail_layers;
            }

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_FFN_UP) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_FFN_UP) != 0)
                goto fail_layers;
            if (load_qweight(&lw->ffn_up, f, wname, sname) != 0) goto fail_layers;

            if (bn_model_arch_tensor_name_for(arch_ops, wname, sizeof(wname), i,
                                              BN_MODEL_TENSOR_FFN_DOWN) != 0 ||
                bn_model_arch_tensor_scale_name_for(arch_ops, sname, sizeof(sname), i,
                                                    BN_MODEL_TENSOR_FFN_DOWN) != 0)
                goto fail_layers;
            if (load_qweight(&lw->ffn_down, f, wname, sname) != 0) goto fail_layers;
        }
    }

    arch_ops->apply_shapes(c, max_head_size, max_kv_dim);
    (void)max_q_dim;

    // --- Weight arena: INT8 embeddings + Q4_0 repacking ---

    // INT8 embedding size (DOTPROD + F16 only)
    size_t emb_i8_bytes = 0;
    size_t emb_i8_scales_bytes = 0;
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__) || defined(__wasm_relaxed_simd__)
    int want_i8_emb = (w->emb_type == BN_GGUF_TENSOR_F16) ||
                       (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16);
    int i8_emb_rows = 0;
    if (want_i8_emb) {
        i8_emb_rows = (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16)
                       ? w->output_weight.rows : c->vocab_size;
        emb_i8_bytes = (size_t)i8_emb_rows * c->dim;
        emb_i8_scales_bytes = (size_t)i8_emb_rows * sizeof(float);
    }
#endif

    // Q4_0 weight repacking size for lane-dot kernels.
    size_t q4_repack_total = 0;
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__wasm_relaxed_simd__)
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

    size_t q8_scale_total = 0;
#ifdef __wasm_relaxed_simd__
    for (int i = 0; i < c->n_layers; i++) {
        BnLayerWeights *lw = &w->layers[i];
        q8_scale_total += q8_f32_scale_bytes(&lw->wq);
        q8_scale_total += q8_f32_scale_bytes(&lw->wk);
        q8_scale_total += q8_f32_scale_bytes(&lw->wv);
        q8_scale_total += q8_f32_scale_bytes(&lw->wo);
        q8_scale_total += q8_f32_scale_bytes(&lw->wqkv);
        q8_scale_total += q8_f32_scale_bytes(&lw->wz);
        q8_scale_total += q8_f32_scale_bytes(&lw->ssm_out);
        q8_scale_total += q8_f32_scale_bytes(&lw->ffn_gate);
        q8_scale_total += q8_f32_scale_bytes(&lw->ffn_up);
        q8_scale_total += q8_f32_scale_bytes(&lw->ffn_down);
    }
    q8_scale_total += q8_f32_scale_bytes(&w->output_weight);
#endif

    size_t weight_arena_size = emb_i8_bytes + emb_i8_scales_bytes + q4_repack_total + q8_scale_total
                              + 4 * SH_ARENA_ALIGN;
    m->weight_arena = NULL;
    m->expert_fd = -1;
    memset(&m->moe_io, 0, sizeof(m->moe_io));
    m->moe_io.fd = -1;

    if (weight_arena_size > 4 * SH_ARENA_ALIGN) {
        m->weight_arena = sh_arena_create(weight_arena_size);
        if (!m->weight_arena) {
            SH_LOG_ERROR("Failed to allocate weight arena");
            goto fail_state;
        }

        // Quantize F16 embeddings to INT8 for fast SDOT logits kernel
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__) || defined(__wasm_relaxed_simd__)
        if (want_i8_emb) {
            w->emb_out_i8 = (int8_t *)sh_arena_alloc(m->weight_arena, emb_i8_bytes);
            w->emb_out_scales = (float *)sh_arena_alloc(m->weight_arena, emb_i8_scales_bytes);
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

        // Repack Q4_0 weights into split scales/qs layout for lane-dot kernels.
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__wasm_relaxed_simd__)
        if (q4_repack_total > 0) {
            for (int i = 0; i < c->n_layers; i++) {
                BnLayerWeights *lw = &w->layers[i];
                q4_repack(m->backend, &lw->wq, m->weight_arena);
                q4_repack(m->backend, &lw->wk, m->weight_arena);
                q4_repack(m->backend, &lw->wv, m->weight_arena);
                q4_repack(m->backend, &lw->wo, m->weight_arena);
                q4_repack(m->backend, &lw->wqkv, m->weight_arena);
                q4_repack(m->backend, &lw->wz, m->weight_arena);
                q4_repack(m->backend, &lw->ssm_out, m->weight_arena);
                q4_repack(m->backend, &lw->ffn_gate, m->weight_arena);
                q4_repack(m->backend, &lw->ffn_up, m->weight_arena);
                q4_repack(m->backend, &lw->ffn_down, m->weight_arena);
            }
            q4_repack(m->backend, &w->output_weight, m->weight_arena);
            char rp_mb[16]; snprintf(rp_mb, sizeof(rp_mb), "%.0f", (double)q4_repack_total / (1024*1024));
            SH_LOG_INFO("Q4_0 weights repacked", "MB", rp_mb);
        }
#endif

#ifdef __wasm_relaxed_simd__
        if (q8_scale_total > 0) {
            for (int i = 0; i < c->n_layers; i++) {
                BnLayerWeights *lw = &w->layers[i];
                q8_prepare_f32_scales(m->backend, &lw->wq, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->wk, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->wv, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->wo, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->wqkv, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->wz, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->ssm_out, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->ffn_gate, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->ffn_up, m->weight_arena);
                q8_prepare_f32_scales(m->backend, &lw->ffn_down, m->weight_arena);
            }
            q8_prepare_f32_scales(m->backend, &w->output_weight, m->weight_arena);
            char q8_mb[16]; snprintf(q8_mb, sizeof(q8_mb), "%.0f", (double)q8_scale_total / (1024*1024));
            SH_LOG_INFO("Q8_0 FP32 scales ready", "MB", q8_mb);
        }
#endif
    }

    // Initialize TurboQuant state if KV compression is enabled
    if (c->kv_tq_bits > 0) {
        m->tq_state = (BnTQState *)malloc(sizeof(BnTQState));
        if (!m->tq_state) goto fail_state;
        if (bn_tq_init(m->tq_state, c->head_size, c->kv_tq_bits, 0x5451303042ULL) != 0) {
            free(m->tq_state);
            m->tq_state = NULL;
            goto fail_state;
        }
        char tq_bits[4], tq_kb[16], tq_vb[16];
        snprintf(tq_bits, sizeof(tq_bits), "%d", c->kv_tq_bits);
        snprintf(tq_kb, sizeof(tq_kb), "%d", bn_tq_key_bytes(m->tq_state));
        snprintf(tq_vb, sizeof(tq_vb), "%d", bn_tq_value_bytes(m->tq_state));
        SH_LOG_INFO("TurboQuant KV", "bits", tq_bits, "key_bytes", tq_kb, "val_bytes", tq_vb);
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
    bn_model_release_gpu(m);
    bn_moe_prefetch_destroy(&m->moe_io);
    bn_tp_free(m->pool);
    if (m->tq_state) {
        bn_tq_free(m->tq_state);
        free(m->tq_state);
        m->tq_state = NULL;
    }
    free(m->weights.layers);
    sh_arena_free(m->weight_arena);
    bn_backend_model_free(m->backend);
    bn_platform_unload_file(&m->file);
    memset(m, 0, sizeof(BnModel));
}
