#ifndef BN_MODEL_ARCH_H
#define BN_MODEL_ARCH_H

#include "model.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *name;
    uint32_t flags;
    int (*is_match)(const char *arch);
    const char *(*prefix)(const char *arch);
    int (*activation)(const char *arch);
    int (*attention_value_shares_key)(const char *arch);
    int (*is_ssm_layer)(const BnConfig *c, int layer);
    int (*tensor_name)(char *out, size_t out_size, int layer, int role);
    void (*apply_shapes)(BnConfig *c, int max_head_size, int max_kv_dim);
} BnModelArchOps;

#define BN_MODEL_ARCH_FLAG_GEMMA4 1u
#define BN_MODEL_ARCH_FLAG_LARGE_GPU_GRAPH_FALLBACK 2u
#define BN_MODEL_ARCH_FLAG_QWEN 4u
#define BN_MODEL_ARCH_FLAG_BITNET 8u

typedef enum {
    BN_MODEL_TENSOR_ATTN_NORM = 0,
    BN_MODEL_TENSOR_ATTN_Q,
    BN_MODEL_TENSOR_ATTN_K,
    BN_MODEL_TENSOR_ATTN_V,
    BN_MODEL_TENSOR_ATTN_OUTPUT,
    BN_MODEL_TENSOR_ATTN_Q_BIAS,
    BN_MODEL_TENSOR_ATTN_K_BIAS,
    BN_MODEL_TENSOR_ATTN_V_BIAS,
    BN_MODEL_TENSOR_ATTN_Q_NORM,
    BN_MODEL_TENSOR_ATTN_K_NORM,
    BN_MODEL_TENSOR_ATTN_SUB_NORM,
    BN_MODEL_TENSOR_SSM_QKV,
    BN_MODEL_TENSOR_SSM_GATE,
    BN_MODEL_TENSOR_SSM_A,
    BN_MODEL_TENSOR_SSM_ALPHA,
    BN_MODEL_TENSOR_SSM_BETA,
    BN_MODEL_TENSOR_SSM_CONV1D,
    BN_MODEL_TENSOR_SSM_DT_BIAS,
    BN_MODEL_TENSOR_SSM_NORM,
    BN_MODEL_TENSOR_SSM_OUT,
    BN_MODEL_TENSOR_FFN_NORM,
    BN_MODEL_TENSOR_FFN_POST_ATTN_NORM,
    BN_MODEL_TENSOR_FFN_SUB_NORM,
    BN_MODEL_TENSOR_FFN_GATE,
    BN_MODEL_TENSOR_FFN_UP,
    BN_MODEL_TENSOR_FFN_DOWN,
    BN_MODEL_TENSOR_MOE_ROUTER,
    BN_MODEL_TENSOR_MOE_GATE_EXPS,
    BN_MODEL_TENSOR_MOE_UP_EXPS,
    BN_MODEL_TENSOR_MOE_GATE_UP_EXPS,
    BN_MODEL_TENSOR_MOE_DOWN_EXPS,
    BN_MODEL_TENSOR_SHARED_FFN_GATE,
    BN_MODEL_TENSOR_SHARED_FFN_UP,
    BN_MODEL_TENSOR_SHARED_FFN_DOWN,
    BN_MODEL_TENSOR_SHARED_FFN_ROUTER,
} BnModelTensorRole;

static inline const char *bn_model_arch_prefix(const char *arch) {
    return arch && arch[0] ? arch : "llama";
}

static inline int bn_model_arch_is_gemma4(const char *arch) {
    return arch && strcmp(arch, "gemma4") == 0;
}

static inline int bn_model_arch_activation(const char *arch) {
    return (arch && strncmp(arch, "bitnet", 6) == 0) ? 1 : 0;
}

static inline int bn_model_arch_attention_value_shares_key(const char *arch) {
    return bn_model_arch_is_gemma4(arch);
}

static inline const char *bn_model_arch_tensor_suffix(BnModelTensorRole role) {
    switch (role) {
        case BN_MODEL_TENSOR_ATTN_NORM:   return "attn_norm.weight";
        case BN_MODEL_TENSOR_ATTN_Q:      return "attn_q.weight";
        case BN_MODEL_TENSOR_ATTN_K:      return "attn_k.weight";
        case BN_MODEL_TENSOR_ATTN_V:      return "attn_v.weight";
        case BN_MODEL_TENSOR_ATTN_OUTPUT: return "attn_output.weight";
        case BN_MODEL_TENSOR_ATTN_Q_BIAS: return "attn_q.bias";
        case BN_MODEL_TENSOR_ATTN_K_BIAS: return "attn_k.bias";
        case BN_MODEL_TENSOR_ATTN_V_BIAS: return "attn_v.bias";
        case BN_MODEL_TENSOR_ATTN_Q_NORM: return "attn_q_norm.weight";
        case BN_MODEL_TENSOR_ATTN_K_NORM: return "attn_k_norm.weight";
        case BN_MODEL_TENSOR_ATTN_SUB_NORM: return "attn_sub_norm.weight";
        case BN_MODEL_TENSOR_SSM_QKV:     return "attn_qkv.weight";
        case BN_MODEL_TENSOR_SSM_GATE:    return "attn_gate.weight";
        case BN_MODEL_TENSOR_SSM_A:       return "ssm_a";
        case BN_MODEL_TENSOR_SSM_ALPHA:   return "ssm_alpha.weight";
        case BN_MODEL_TENSOR_SSM_BETA:    return "ssm_beta.weight";
        case BN_MODEL_TENSOR_SSM_CONV1D:  return "ssm_conv1d.weight";
        case BN_MODEL_TENSOR_SSM_DT_BIAS: return "ssm_dt.bias";
        case BN_MODEL_TENSOR_SSM_NORM:    return "ssm_norm.weight";
        case BN_MODEL_TENSOR_SSM_OUT:     return "ssm_out.weight";
        case BN_MODEL_TENSOR_FFN_NORM:    return "ffn_norm.weight";
        case BN_MODEL_TENSOR_FFN_POST_ATTN_NORM: return "post_attention_norm.weight";
        case BN_MODEL_TENSOR_FFN_SUB_NORM: return "ffn_sub_norm.weight";
        case BN_MODEL_TENSOR_FFN_GATE:    return "ffn_gate.weight";
        case BN_MODEL_TENSOR_FFN_UP:      return "ffn_up.weight";
        case BN_MODEL_TENSOR_FFN_DOWN:    return "ffn_down.weight";
        case BN_MODEL_TENSOR_MOE_ROUTER:  return "ffn_gate_inp.weight";
        case BN_MODEL_TENSOR_MOE_GATE_EXPS: return "ffn_gate_exps.weight";
        case BN_MODEL_TENSOR_MOE_UP_EXPS: return "ffn_up_exps.weight";
        case BN_MODEL_TENSOR_MOE_GATE_UP_EXPS: return "ffn_gate_up_exps.weight";
        case BN_MODEL_TENSOR_MOE_DOWN_EXPS: return "ffn_down_exps.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_GATE: return "ffn_gate_shexp.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_UP: return "ffn_up_shexp.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_DOWN: return "ffn_down_shexp.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_ROUTER: return "ffn_gate_inp_shexp.weight";
        default:                          return NULL;
    }
}

static inline int bn_model_arch_default_tensor_name(char *out,
                                                    size_t out_size,
                                                    int layer,
                                                    int role) {
    const char *suffix = bn_model_arch_tensor_suffix((BnModelTensorRole)role);
    if (!out || out_size == 0 || layer < 0 || !suffix) return -1;
    int n = snprintf(out, out_size, "blk.%d.%s", layer, suffix);
    return (n < 0 || (size_t)n >= out_size) ? -1 : 0;
}

static inline int bn_model_arch_tensor_name_for(const BnModelArchOps *ops,
                                                char *out,
                                                size_t out_size,
                                                int layer,
                                                BnModelTensorRole role) {
    if (ops && ops->tensor_name)
        return ops->tensor_name(out, out_size, layer, (int)role);
    return bn_model_arch_default_tensor_name(out, out_size, layer, (int)role);
}

static inline int bn_model_arch_tensor_scale_name_for(const BnModelArchOps *ops,
                                                      char *out,
                                                      size_t out_size,
                                                      int layer,
                                                      BnModelTensorRole role) {
    char weight_name[128];
    if (bn_model_arch_tensor_name_for(ops, weight_name, sizeof(weight_name),
                                      layer, role) != 0)
        return -1;
    const char *suffix = ".weight";
    size_t len = strlen(weight_name);
    size_t suffix_len = strlen(suffix);
    if (len < suffix_len ||
        strcmp(weight_name + len - suffix_len, suffix) != 0)
        return -1;
    if (len - suffix_len + strlen(".scale") + 1 > out_size)
        return -1;
    memcpy(out, weight_name, len - suffix_len);
    memcpy(out + len - suffix_len, ".scale", strlen(".scale") + 1);
    return 0;
}

static inline int bn_model_arch_requires_large_gpu_graph_fallback(const BnConfig *c) {
    return c && ((c->arch_flags & BN_MODEL_ARCH_FLAG_LARGE_GPU_GRAPH_FALLBACK) != 0);
}

static inline int bn_model_arch_rope_text_dims(int rope_dim_count,
                                                const int32_t *sections,
                                                uint64_t n_sections) {
    if (!sections || n_sections == 0 || rope_dim_count <= 0 || sections[0] <= 0)
        return 0;
    return sections[0] * 2;
}

static inline int bn_model_arch_is_ssm_layer(const BnConfig *c, int layer) {
    return c && c->full_attn_interval > 0 &&
           ((layer + 1) % c->full_attn_interval != 0);
}

static inline int bn_model_arch_infer_moe_hidden(BnGGUFFile *f,
                                                 const BnModelArchOps *ops) {
    char name[128];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_MOE_GATE_EXPS) != 0)
        return 0;
    int ti = bn_gguf_find_tensor(f, name);
    if (ti >= 0 && f->tensors[ti].n_dims >= 3)
        return (int)f->tensors[ti].dims[1];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_MOE_GATE_UP_EXPS) != 0)
        return 0;
    ti = bn_gguf_find_tensor(f, name);
    if (ti >= 0 && f->tensors[ti].n_dims >= 3)
        return (int)(f->tensors[ti].dims[1] / 2);
    return 0;
}

static inline int bn_model_arch_has_shared_expert(BnGGUFFile *f,
                                                  const BnModelArchOps *ops) {
    char name[128];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_SHARED_FFN_GATE) != 0)
        return 0;
    return bn_gguf_find_tensor(f, name) >= 0;
}

static inline int bn_model_arch_infer_shared_expert_hidden(BnGGUFFile *f,
                                                           const BnModelArchOps *ops) {
    char name[128];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_SHARED_FFN_GATE) != 0)
        return 0;
    int ti = bn_gguf_find_tensor(f, name);
    if (ti >= 0 && f->tensors[ti].n_dims >= 2)
        return (int)f->tensors[ti].dims[1];
    return 0;
}

static inline void bn_model_arch_load_moe_config(BnConfig *c,
                                                  BnGGUFFile *f,
                                                  const BnModelArchOps *ops,
                                                  const char *prefix) {
    char key[128];
    snprintf(key, sizeof(key), "%s.expert_count", prefix);
    c->n_experts = (int)bn_gguf_get_u32(f, key);
    snprintf(key, sizeof(key), "%s.expert_used_count", prefix);
    c->n_experts_active = (int)bn_gguf_get_u32(f, key);

    if (c->n_experts <= 0) return;

    snprintf(key, sizeof(key), "%s.expert_feed_forward_length", prefix);
    c->moe_intermediate_size = (int)bn_gguf_get_u32(f, key);
    if (c->moe_intermediate_size == 0)
        c->moe_intermediate_size = bn_model_arch_infer_moe_hidden(f, ops);

    c->has_shared_expert = bn_model_arch_has_shared_expert(f, ops);
    if (c->has_shared_expert) {
        snprintf(key, sizeof(key), "%s.expert_shared_feed_forward_length", prefix);
        c->shared_expert_intermediate_size = (int)bn_gguf_get_u32(f, key);
        if (c->shared_expert_intermediate_size == 0)
            c->shared_expert_intermediate_size =
                bn_model_arch_infer_shared_expert_hidden(f, ops);
    }
}

static inline void bn_model_arch_apply_gemma4_shapes(BnConfig *c,
                                                      int max_head_size,
                                                      int max_kv_dim) {
    if (!c) return;
    c->head_size = max_head_size;
    c->kv_dim = max_kv_dim;
    c->kv_mul = c->n_heads / c->n_kv_heads;
}

static inline void bn_model_arch_apply_default_shapes(BnConfig *c,
                                                       int max_head_size,
                                                       int max_kv_dim) {
    (void)c;
    (void)max_head_size;
    (void)max_kv_dim;
}

static inline int bn_model_arch_match_gemma4(const char *arch) {
    return bn_model_arch_is_gemma4(arch);
}

static inline int bn_model_arch_match_qwen(const char *arch) {
    return arch && strncmp(arch, "qwen", 4) == 0;
}

static inline int bn_model_arch_match_bitnet(const char *arch) {
    return arch && strncmp(arch, "bitnet", 6) == 0;
}

static inline int bn_model_arch_match_default(const char *arch) {
    (void)arch;
    return 1;
}

static inline const BnModelArchOps *bn_model_arch_registry(size_t *count) {
    static const BnModelArchOps ops[] = {
        {
            "gemma4",
            BN_MODEL_ARCH_FLAG_GEMMA4 | BN_MODEL_ARCH_FLAG_LARGE_GPU_GRAPH_FALLBACK,
            bn_model_arch_match_gemma4,
            bn_model_arch_prefix,
            bn_model_arch_activation,
            bn_model_arch_attention_value_shares_key,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_gemma4_shapes,
        },
        {
            "qwen",
            BN_MODEL_ARCH_FLAG_QWEN,
            bn_model_arch_match_qwen,
            bn_model_arch_prefix,
            bn_model_arch_activation,
            bn_model_arch_attention_value_shares_key,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
        {
            "bitnet",
            BN_MODEL_ARCH_FLAG_BITNET,
            bn_model_arch_match_bitnet,
            bn_model_arch_prefix,
            bn_model_arch_activation,
            bn_model_arch_attention_value_shares_key,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
        {
            "default",
            0,
            bn_model_arch_match_default,
            bn_model_arch_prefix,
            bn_model_arch_activation,
            bn_model_arch_attention_value_shares_key,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
    };
    if (count) *count = sizeof(ops) / sizeof(ops[0]);
    return ops;
}

static inline const BnModelArchOps *bn_model_arch_ops_for(const char *arch) {
    size_t n = 0;
    const BnModelArchOps *ops = bn_model_arch_registry(&n);
    for (size_t i = 0; i < n; i++) {
        if (ops[i].is_match && ops[i].is_match(arch)) return &ops[i];
    }
    return n > 0 ? &ops[n - 1] : NULL;
}

#endif // BN_MODEL_ARCH_H
