#include "model_arch.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

const char *bn_model_arch_prefix(const char *arch) {
    return arch && arch[0] ? arch : "llama";
}

static int bn_model_arch_is_gemma4(const char *arch) {
    return arch && strcmp(arch, "gemma4") == 0;
}

int bn_model_arch_activation(const char *arch) {
    const BnModelArchOps *ops = bn_model_arch_ops_for(arch);
    if (ops && ops->activation)
        return ops->activation(arch);
    return BN_MODEL_ACTIVATION_SILU;
}

int bn_model_arch_activation_is_silu(int act_type) {
    return act_type == BN_MODEL_ACTIVATION_SILU;
}

int bn_model_arch_activation_is_relu2(int act_type) {
    return act_type == BN_MODEL_ACTIVATION_RELU2;
}

int bn_model_arch_activation_is_gelu(int act_type) {
    return act_type == BN_MODEL_ACTIVATION_GELU;
}

int bn_model_arch_activation_uses_silu_path(int act_type) {
    return !bn_model_arch_activation_is_relu2(act_type) &&
           !bn_model_arch_activation_is_gelu(act_type);
}

int bn_model_arch_attention_value_shares_key(const char *arch) {
    const BnModelArchOps *ops = bn_model_arch_ops_for(arch);
    return ops && ops->attention_value_shares_key &&
           ops->attention_value_shares_key(arch);
}

const char *bn_model_arch_tensor_suffix(BnModelTensorRole role) {
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
        case BN_MODEL_TENSOR_ATTN_POST_NORM: return "post_attention_norm.weight";
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
        case BN_MODEL_TENSOR_FFN_POST_NORM: return "post_ffw_norm.weight";
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
        case BN_MODEL_TENSOR_LAYER_OUTPUT_SCALE: return "layer_output_scale.weight";
        default:                          return NULL;
    }
}

int bn_model_arch_default_tensor_name(char *out,
                                      size_t out_size,
                                      int layer,
                                      int role) {
    const char *suffix = bn_model_arch_tensor_suffix((BnModelTensorRole)role);
    if (!out || out_size == 0 || layer < 0 || !suffix) return -1;
    int n = snprintf(out, out_size, "blk.%d.%s", layer, suffix);
    return (n < 0 || (size_t)n >= out_size) ? -1 : 0;
}

int bn_model_arch_tensor_name_for(const BnModelArchOps *ops,
                                  char *out,
                                  size_t out_size,
                                  int layer,
                                  BnModelTensorRole role) {
    if (ops && ops->tensor_name)
        return ops->tensor_name(out, out_size, layer, (int)role);
    return bn_model_arch_default_tensor_name(out, out_size, layer, (int)role);
}

int bn_model_arch_tensor_scale_name_for(const BnModelArchOps *ops,
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

void bn_model_arch_apply_config(BnConfig *c, const BnModelArchOps *ops) {
    if (c)
        c->policy_flags = ops ? ops->policy_flags : 0;
}

int bn_model_arch_requires_large_gpu_graph_fallback(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_LARGE_GPU_GRAPH_FALLBACK) != 0);
}

int bn_model_arch_requires_float_kquant_fallback(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK) != 0);
}

int bn_model_arch_cpu_force_float_kquant(const BnConfig *c) {
    return bn_model_arch_requires_float_kquant_fallback(c);
}

float bn_model_arch_attention_scale(const BnConfig *c, int head_size) {
    if (c && (c->policy_flags & BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE))
        return 1.0f;
    return 1.0f / sqrtf((float)head_size);
}

BnModelArchRMSNormMode bn_model_arch_rmsnorm_mode(const BnConfig *c) {
    if (c && (c->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER))
        return BN_MODEL_ARCH_RMSNORM_REFERENCE_SCALAR_ORDER;
    return BN_MODEL_ARCH_RMSNORM_BACKEND_ORDER;
}

int bn_model_arch_rmsnorm_uses_reference_order(
    const BnConfig *c) {
    return bn_model_arch_rmsnorm_mode(c) ==
           BN_MODEL_ARCH_RMSNORM_REFERENCE_SCALAR_ORDER;
}

int bn_model_arch_rmsnorm_requires_reference_scalar_order(
    const BnConfig *c) {
    return bn_model_arch_rmsnorm_uses_reference_order(c);
}

int bn_model_arch_attention_value_shares_key_config(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY) != 0);
}

int bn_model_arch_uses_per_layer_embedding(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT) != 0);
}

int bn_model_arch_uses_attention_post_norm(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM) != 0);
}

int bn_model_arch_uses_ffn_post_norm(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_FFN_POST_NORM) != 0);
}

int bn_model_arch_uses_layer_output_scale(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE) != 0);
}

int bn_model_arch_uses_reference_hybrid_ssm(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM) != 0) &&
           c->full_attn_interval > 0;
}

int bn_model_arch_uses_scalar_hybrid_ssm_cpu(const BnConfig *c) {
    return bn_model_arch_uses_reference_hybrid_ssm(c);
}

int bn_model_arch_uses_hybrid_layer_layout(const BnConfig *c) {
    return c && c->full_attn_interval > 0;
}

int bn_model_arch_uses_hybrid_ssm(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_layer_layout(c) && c->ssm_inner_size > 0;
}

int bn_model_arch_uses_dense_attention_only(const BnConfig *c) {
    return c &&
           c->n_experts <= 0 &&
           !bn_model_arch_uses_hybrid_layer_layout(c);
}

int bn_model_arch_uses_large_dense_shape(const BnConfig *c) {
    return c &&
           c->n_experts <= 0 &&
           c->dim >= 4096;
}

int bn_model_arch_uses_large_dense_hybrid_ssm(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_ssm(c) &&
           bn_model_arch_uses_large_dense_shape(c);
}

int bn_model_arch_uses_large_gpu_graph_fallback_shape(const BnConfig *c) {
    return c &&
           c->dim >= 4096 &&
           (bn_model_arch_requires_large_gpu_graph_fallback(c) ||
            c->full_attn_interval > 0 ||
            bn_model_arch_uses_moe(c));
}

static int model_arch_gemma4_divides_rope_freqs(const BnConfig *c, int layer) {
    if (!c || ((c->policy_flags & BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT) == 0))
        return 0;
    if (c->per_layer_input_dim > 0)
        return 1;
    if (c->n_experts > 0 && c->n_layers == 30)
        return layer == 5 || layer == 23 || layer == 29;
    return 0;
}

int bn_model_arch_divides_rope_freqs(const BnConfig *c, int layer) {
    return model_arch_gemma4_divides_rope_freqs(c, layer);
}

int bn_model_arch_per_layer_embedding_dim(const BnConfig *c) {
    if (!bn_model_arch_uses_per_layer_embedding(c) ||
        c->per_layer_input_dim <= 0)
        return 0;
    return c->per_layer_input_dim;
}

int bn_model_arch_allows_small_dense_prefill_decode_fallback(
    const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK) != 0) &&
           bn_model_arch_uses_dense_attention_only(c) &&
           c->dim <= 2560;
}

int bn_model_arch_prefill_uses_decode_for_parity(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY) != 0);
}

int bn_model_arch_cpu_prefill_uses_decode_for_parity(const BnConfig *c) {
    return bn_model_arch_prefill_uses_decode_for_parity(c);
}

int bn_model_arch_moe_forces_float_kquant_gateup(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP) != 0);
}

int bn_model_arch_moe_prefers_exact_gpu_attention(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_EXACT_GPU_ATTENTION) != 0);
}

int bn_model_arch_moe_uses_scaled_router_input(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT) != 0);
}

int bn_model_arch_moe_uses_dense_residual_branch(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH) != 0);
}

int bn_model_arch_uses_moe(const BnConfig *c) {
    return c && c->n_experts > 0;
}

static int bn_model_arch_gguf_key(BnGGUFFile *f,
                                  const char *suffix,
                                  char *key,
                                  size_t key_size) {
    if (!f || !suffix || !key || key_size == 0) return -1;
    const char *arch = bn_gguf_get_str(f, "general.architecture");
    const BnModelArchOps *ops = bn_model_arch_ops_for(arch);
    const char *prefix = ops && ops->prefix
        ? ops->prefix(arch)
        : bn_model_arch_prefix(arch);
    int n = snprintf(key, key_size, "%s.%s", prefix, suffix);
    return (n < 0 || (size_t)n >= key_size) ? -1 : 0;
}

int bn_model_arch_gguf_u32(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0;
    return (int)bn_gguf_get_u32(f, key);
}

int bn_model_arch_gguf_u32_or_i32_array(BnGGUFFile *f,
                                        const char *suffix,
                                        int idx) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0;
    int ki = bn_gguf_find_key(f, key);
    if (ki < 0) return 0;
    BnGGUFKeyValue *kv = &f->kvs[ki];
    if (kv->type == BN_GGUF_TYPE_UINT32) return (int)kv->value.u32;
    if (kv->type == BN_GGUF_TYPE_INT32) return kv->value.i32;
    if (kv->type != BN_GGUF_TYPE_ARRAY || idx < 0) return 0;
    BnGGUFArray *a = &kv->value.arr;
    if ((uint64_t)idx >= a->n || !a->data) return 0;
    if (a->elem_type == BN_GGUF_TYPE_INT32) {
        int32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return v;
    }
    if (a->elem_type == BN_GGUF_TYPE_UINT32) {
        uint32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return (int)v;
    }
    return 0;
}

float bn_model_arch_gguf_f32(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0.0f;
    return bn_gguf_get_f32(f, key);
}

uint64_t bn_model_arch_gguf_arr_n(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0;
    return bn_gguf_get_arr_n(f, key);
}

const void *bn_model_arch_gguf_arr_data(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0)
        return NULL;
    return bn_gguf_get_arr_data(f, key);
}

int bn_model_arch_gguf_bool_array(BnGGUFFile *f,
                                  const char *suffix,
                                  int idx) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0 ||
        idx < 0)
        return 0;
    int ki = bn_gguf_find_key(f, key);
    if (ki < 0) return 0;
    BnGGUFKeyValue *kv = &f->kvs[ki];
    if (kv->type == BN_GGUF_TYPE_BOOL) return kv->value.b ? 1 : 0;
    if (kv->type != BN_GGUF_TYPE_ARRAY) return 0;
    BnGGUFArray *a = &kv->value.arr;
    if ((uint64_t)idx >= a->n || !a->data) return 0;
    if (a->elem_type == BN_GGUF_TYPE_BOOL)
        return ((const uint8_t *)a->data)[idx] ? 1 : 0;
    if (a->elem_type == BN_GGUF_TYPE_INT32) {
        int32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return v != 0;
    }
    if (a->elem_type == BN_GGUF_TYPE_UINT32) {
        uint32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return v != 0;
    }
    return 0;
}

int bn_model_arch_gguf_uses_moe(BnGGUFFile *f) {
    return bn_model_arch_gguf_u32(f, "expert_count") > 0;
}

int bn_model_arch_uses_non_hybrid_moe(const BnConfig *c) {
    return bn_model_arch_uses_moe(c) && c->full_attn_interval <= 0;
}

int bn_model_arch_uses_hybrid_moe(const BnConfig *c) {
    return bn_model_arch_uses_moe(c) && c->full_attn_interval > 0;
}

int bn_model_arch_uses_two_expert_all_active_moe(const BnConfig *c) {
    return c &&
           c->n_experts == 2 &&
           c->n_experts_active == 2;
}

int bn_model_arch_uses_more_than_two_expert_moe(const BnConfig *c) {
    return c && c->n_experts > 2;
}

int bn_model_arch_moe_prefill_forces_matvec(const BnConfig *c) {
    return bn_model_arch_uses_two_expert_all_active_moe(c) &&
           c->has_shared_expert;
}

int bn_model_arch_uses_all_active_two_expert_moe(const BnConfig *c,
                                                 int dim) {
    return bn_model_arch_uses_two_expert_all_active_moe(c) &&
           c->moe_intermediate_size >= 4096 &&
           dim <= 2048;
}

int bn_model_arch_loads_extra_metadata(const BnConfig *c) {
    return bn_model_arch_uses_per_layer_embedding(c);
}

int bn_model_arch_loads_per_layer_input_weights(const BnConfig *c) {
    return bn_model_arch_uses_per_layer_embedding(c) &&
           c->per_layer_input_dim > 0;
}

int bn_model_arch_layer_reuses_kv(const BnConfig *c, int layer) {
    return bn_model_arch_attention_value_shares_key_config(c) &&
           c->kv_unique_layer_count > 0 &&
           layer >= c->kv_unique_layer_count;
}

int bn_model_arch_kv_reuse_layer(const BnConfig *c, int layer) {
    if (!bn_model_arch_layer_reuses_kv(c, layer))
        return -1;
    int is_swa = (layer >= 0 &&
                  layer < (int)(sizeof(c->sliding_window_pattern) /
                                sizeof(c->sliding_window_pattern[0])))
        ? c->sliding_window_pattern[layer]
        : 0;
    int reuse_layer = c->kv_unique_layer_count - (is_swa ? 2 : 1);
    return reuse_layer < 0 ? 0 : reuse_layer;
}

int bn_model_arch_loads_extra_ffn_post_norms(const BnConfig *c) {
    return bn_model_arch_uses_ffn_post_norm(c);
}

int bn_model_arch_loads_moe_aux_weights(const BnConfig *c) {
    return bn_model_arch_moe_uses_scaled_router_input(c);
}

int bn_model_arch_config_uses_full_rope_text_dims(const BnConfig *c) {
    return c &&
           ((c->policy_flags & BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS) != 0);
}

int bn_model_arch_tokenizer_uses_metaspace(const char *tokenizer_model) {
    return bn_model_arch_is_gemma4(tokenizer_model);
}

int bn_model_arch_uses_small_dense_shape(const BnConfig *c) {
    return bn_model_arch_uses_dense_attention_only(c) &&
           c->dim <= 2560;
}

int bn_model_arch_uses_small_dense_native_quant_shape(const BnConfig *c) {
    return bn_model_arch_uses_small_dense_shape(c) &&
           c->dim > 1024;
}

int bn_model_arch_dense_batch_prefill_shape_allowed(
    const BnConfig *c,
    int supports_large_dense_batch_prefill) {
    if (!bn_model_arch_uses_dense_attention_only(c))
        return 0;
    return c->dim <= (supports_large_dense_batch_prefill ? 8192 : 2560);
}

int bn_model_arch_dense_logits_argmax_shape_allowed(const BnConfig *c,
                                                    int logits_rows) {
    return bn_model_arch_uses_dense_attention_only(c) &&
           logits_rows > 262144;
}

int bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(const BnConfig *c,
                                                       int logits_cols) {
    return bn_model_arch_uses_moe(c) &&
           logits_cols == 1536;
}

int bn_model_arch_allows_small_dense_exact_native(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE) != 0) &&
           bn_model_arch_uses_small_dense_shape(c);
}

int bn_model_arch_small_dense_exact_native_to_layer(const BnConfig *c) {
    if (!c || c->n_layers <= 33)
        return -1;
    return c->n_layers - 33 - 1;
}

int bn_model_arch_allows_small_dense_native_logit_refine(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE) != 0) &&
           bn_model_arch_allows_small_dense_exact_native(c);
}

int bn_model_arch_small_dense_prefill_min_tokens(const BnConfig *c) {
    if (!bn_model_arch_allows_small_dense_native_logit_refine(c))
        return 0;
    return (c->policy_flags & BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION) ? 7 : 2;
}

int bn_model_arch_prefill_uses_exact_activation(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION) != 0);
}

int bn_model_arch_ffn_uses_reference_activation(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION) != 0);
}

int bn_model_arch_ffn_uses_exact_scalar_activation(const BnConfig *c) {
    return bn_model_arch_ffn_uses_reference_activation(c);
}

int bn_model_arch_rope_text_dims(int rope_dim_count,
                                 const int32_t *sections,
                                 uint64_t n_sections) {
    if (!sections || n_sections == 0 || rope_dim_count <= 0 || sections[0] <= 0)
        return 0;
    return sections[0] * 2;
}

int bn_model_arch_is_ssm_layer(const BnConfig *c, int layer) {
    return bn_model_arch_uses_hybrid_layer_layout(c) &&
           ((layer + 1) % c->full_attn_interval != 0);
}

int bn_model_arch_is_attention_layer(const BnConfig *c, int layer) {
    return c && !bn_model_arch_is_ssm_layer(c, layer);
}

int bn_model_arch_attention_layer_index(const BnConfig *c, int layer) {
    if (!c) return -1;
    return bn_model_arch_uses_hybrid_layer_layout(c)
        ? (layer + 1) / c->full_attn_interval - 1
        : layer;
}

int bn_model_arch_ssm_layer_index(const BnConfig *c, int layer) {
    if (!bn_model_arch_uses_hybrid_layer_layout(c)) return -1;
    return layer - (layer + 1) / c->full_attn_interval;
}

int bn_model_arch_attention_layer_count(const BnConfig *c) {
    if (!c || c->n_layers <= 0) return 0;
    return bn_model_arch_uses_hybrid_layer_layout(c)
        ? c->n_layers / c->full_attn_interval
        : c->n_layers;
}

int bn_model_arch_ssm_layer_count(const BnConfig *c) {
    if (!c || c->n_layers <= 0) return 0;
    int n_attn_layers = bn_model_arch_attention_layer_count(c);
    int n_ssm_layers = c->n_layers - n_attn_layers;
    return n_ssm_layers > 0 ? n_ssm_layers : 0;
}

int bn_model_arch_infer_moe_hidden(BnGGUFFile *f,
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

int bn_model_arch_has_shared_expert(BnGGUFFile *f,
                                    const BnModelArchOps *ops) {
    char name[128];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_SHARED_FFN_GATE) != 0)
        return 0;
    return bn_gguf_find_tensor(f, name) >= 0;
}

int bn_model_arch_infer_shared_expert_hidden(BnGGUFFile *f,
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

void bn_model_arch_load_moe_config(BnConfig *c,
                                   BnGGUFFile *f,
                                   const BnModelArchOps *ops,
                                   const char *prefix) {
    char key[128];
    snprintf(key, sizeof(key), "%s.expert_count", prefix);
    c->n_experts = (int)bn_gguf_get_u32(f, key);
    snprintf(key, sizeof(key), "%s.expert_used_count", prefix);
    c->n_experts_active = (int)bn_gguf_get_u32(f, key);

    if (c->n_experts <= 0) return;
    if (ops)
        c->policy_flags |= ops->moe_policy_flags;
    c->moe_norm_topk_prob =
        (c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_UNNORMALIZED_TOPK) == 0;
    c->moe_exact_silu =
        (c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_EXACT_SILU) != 0;
    snprintf(key, sizeof(key), "%s.expert_weights_scale", prefix);
    c->moe_expert_weights_scale = bn_gguf_get_f32(f, key);

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

static void bn_model_arch_apply_gemma4_shapes(BnConfig *c,
                                              int max_head_size,
                                              int max_kv_dim) {
    if (!c) return;
    c->head_size = max_head_size;
    c->kv_dim = max_kv_dim;
    c->kv_mul = c->n_heads / c->n_kv_heads;
}

static int bn_model_arch_gemma4_tensor_name(char *out, size_t out_size,
                                            int layer, int role) {
    const char *suffix = NULL;
    switch ((BnModelTensorRole)role) {
        case BN_MODEL_TENSOR_FFN_SUB_NORM:
            suffix = "pre_ffw_norm_2.weight";
            break;
        default:
            return bn_model_arch_default_tensor_name(out, out_size, layer, role);
    }
    if (!out || out_size == 0 || layer < 0) return -1;
    int n = snprintf(out, out_size, "blk.%d.%s", layer, suffix);
    return (n < 0 || (size_t)n >= out_size) ? -1 : 0;
}

static void bn_model_arch_apply_default_shapes(BnConfig *c,
                                               int max_head_size,
                                               int max_kv_dim) {
    (void)c;
    (void)max_head_size;
    (void)max_kv_dim;
}

static int bn_model_arch_default_activation(const char *arch) {
    (void)arch;
    return BN_MODEL_ACTIVATION_SILU;
}

static int bn_model_arch_bitnet_activation(const char *arch) {
    (void)arch;
    return BN_MODEL_ACTIVATION_RELU2;
}

static int bn_model_arch_gemma4_activation(const char *arch) {
    (void)arch;
    return BN_MODEL_ACTIVATION_GELU;
}

static int bn_model_arch_attention_value_unique(const char *arch) {
    (void)arch;
    return 0;
}

static int bn_model_arch_attention_value_shared(const char *arch) {
    (void)arch;
    return 1;
}

static int bn_model_arch_match_gemma4(const char *arch) {
    return bn_model_arch_is_gemma4(arch);
}

static int bn_model_arch_match_qwen(const char *arch) {
    return arch && strncmp(arch, "qwen", 4) == 0;
}

static int bn_model_arch_match_qwen2(const char *arch) {
    return arch && strncmp(arch, "qwen2", 5) == 0;
}

static int bn_model_arch_match_qwen3(const char *arch) {
    return arch && strcmp(arch, "qwen3") == 0;
}

static int bn_model_arch_match_qwen35(const char *arch) {
    return arch && (strcmp(arch, "qwen35") == 0 ||
                    strcmp(arch, "qwen35moe") == 0);
}

static int bn_model_arch_match_bitnet(const char *arch) {
    return arch && strncmp(arch, "bitnet", 6) == 0;
}

static int bn_model_arch_match_default(const char *arch) {
    (void)arch;
    return 1;
}

const BnModelArchOps *bn_model_arch_registry(size_t *count) {
    static const BnModelArchOps ops[] = {
        {
            "gemma4",
            BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE |
            BN_MODEL_ARCH_POLICY_LARGE_GPU_GRAPH_FALLBACK |
            BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY |
            BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT |
            BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM |
            BN_MODEL_ARCH_POLICY_FFN_POST_NORM |
            BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT |
            BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH,
            0,
            bn_model_arch_match_gemma4,
            bn_model_arch_prefix,
            bn_model_arch_gemma4_activation,
            bn_model_arch_attention_value_shared,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_gemma4_tensor_name,
            bn_model_arch_apply_gemma4_shapes,
        },
        {
            "qwen3",
            BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM |
            BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE |
            BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION,
            0,
            bn_model_arch_match_qwen3,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
        {
            "qwen35",
            BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION |
            BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS,
            0,
            bn_model_arch_match_qwen35,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
        {
            "qwen2",
            BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM |
            BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION,
            BN_MODEL_ARCH_POLICY_MOE_EXACT_SILU |
            BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP |
            BN_MODEL_ARCH_POLICY_MOE_EXACT_GPU_ATTENTION |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_MOE_UNNORMALIZED_TOPK,
            bn_model_arch_match_qwen2,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
        {
            "qwen",
            BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION,
            0,
            bn_model_arch_match_qwen,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
        {
            "bitnet",
            0,
            0,
            bn_model_arch_match_bitnet,
            bn_model_arch_prefix,
            bn_model_arch_bitnet_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
        {
            "default",
            0,
            0,
            bn_model_arch_match_default,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
        },
    };
    if (count) *count = sizeof(ops) / sizeof(ops[0]);
    return ops;
}

const BnModelArchOps *bn_model_arch_ops_for(const char *arch) {
    size_t n = 0;
    const BnModelArchOps *ops = bn_model_arch_registry(&n);
    for (size_t i = 0; i < n; i++) {
        if (ops[i].is_match && ops[i].is_match(arch)) return &ops[i];
    }
    return n > 0 ? &ops[n - 1] : NULL;
}
