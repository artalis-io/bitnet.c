#ifndef BN_MODEL_ARCH_H
#define BN_MODEL_ARCH_H

#include "model.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

static inline int bn_model_arch_infer_moe_hidden(BnGGUFFile *f) {
    int ti = bn_gguf_find_tensor(f, "blk.0.ffn_gate_exps.weight");
    if (ti >= 0 && f->tensors[ti].n_dims >= 3)
        return (int)f->tensors[ti].dims[1];
    ti = bn_gguf_find_tensor(f, "blk.0.ffn_gate_up_exps.weight");
    if (ti >= 0 && f->tensors[ti].n_dims >= 3)
        return (int)(f->tensors[ti].dims[1] / 2);
    return 0;
}

static inline int bn_model_arch_has_shared_expert(BnGGUFFile *f) {
    return bn_gguf_find_tensor(f, "blk.0.ffn_gate_shexp.weight") >= 0;
}

static inline int bn_model_arch_infer_shared_expert_hidden(BnGGUFFile *f) {
    int ti = bn_gguf_find_tensor(f, "blk.0.ffn_gate_shexp.weight");
    if (ti >= 0 && f->tensors[ti].n_dims >= 2)
        return (int)f->tensors[ti].dims[1];
    return 0;
}

static inline void bn_model_arch_load_moe_config(BnConfig *c,
                                                  BnGGUFFile *f,
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
        c->moe_intermediate_size = bn_model_arch_infer_moe_hidden(f);

    c->has_shared_expert = bn_model_arch_has_shared_expert(f);
    if (c->has_shared_expert) {
        snprintf(key, sizeof(key), "%s.expert_shared_feed_forward_length", prefix);
        c->shared_expert_intermediate_size = (int)bn_gguf_get_u32(f, key);
        if (c->shared_expert_intermediate_size == 0)
            c->shared_expert_intermediate_size =
                bn_model_arch_infer_shared_expert_hidden(f);
    }
}

static inline void bn_model_arch_apply_gemma4_shapes(BnConfig *c,
                                                      int max_head_size,
                                                      int max_kv_dim) {
    if (!c || !c->arch_gemma4) return;
    c->head_size = max_head_size;
    c->kv_dim = max_kv_dim;
    c->kv_mul = c->n_heads / c->n_kv_heads;
}

#endif // BN_MODEL_ARCH_H
