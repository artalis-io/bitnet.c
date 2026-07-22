#include "quant.h"
#include "gguf.h"
#include "gpu_backend.h"

#define BN_QUANT_CAP_CPU_ALL \
    (BN_QUANT_CAP_CPU_MATVEC | BN_QUANT_CAP_CPU_BATCH | BN_QUANT_CAP_CPU_MATMUL)
#define BN_QUANT_CAP_LOADABLE_CPU \
    (BN_QUANT_CAP_LOADABLE | BN_QUANT_CAP_CPU_ALL)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED \
    (BN_QUANT_CAP_LOADABLE_CPU | BN_QUANT_CAP_EMBEDDED_SCALE)
#define BN_QUANT_CAP_LOADABLE_CPU_TIED_LOGITS \
    (BN_QUANT_CAP_LOADABLE_CPU | BN_QUANT_CAP_TIED_LOGITS_QUANT_PATH)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS \
    (BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_TIED_LOGITS_QUANT_PATH)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREPARED_KQUANT \
    (BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_CPU_PREPARED_KQUANT)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREPARED_KQUANT_TIED_LOGITS \
    (BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREPARED_KQUANT | BN_QUANT_CAP_TIED_LOGITS_QUANT_PATH)
#define BN_QUANT_CAP_GPU_DENSE_GRAPH_NATIVE_QUANT_FORMAT \
    (BN_QUANT_CAP_GPU_DENSE_GRAPH | BN_QUANT_CAP_GPU_DENSE_GRAPH_NATIVE_QUANT | \
     BN_QUANT_CAP_NATIVE_QUANT_LOGITS_REFINE | BN_QUANT_CAP_GPU_EXACT_SILU | \
     BN_QUANT_CAP_GPU_GATEUP_SPLIT_PREFERRED)
#define BN_QUANT_CAP_GPU_DENSE_GRAPH_KQUANT \
    (BN_QUANT_CAP_GPU_DENSE_GRAPH | BN_QUANT_CAP_FLOAT_KQUANT_FALLBACK)
#define BN_QUANT_CAP_GPU_DENSE_GRAPH_KQUANT_LOGITS_REFINE \
    (BN_QUANT_CAP_GPU_DENSE_GRAPH_KQUANT | BN_QUANT_CAP_KQUANT_LOGITS_REFINE)
#define BN_QUANT_CAP_LAZY_AUX_CACHE \
    (BN_QUANT_CAP_LAZY_MOE_AUX_CACHE_CANDIDATE | BN_QUANT_CAP_AUX_CACHE)
#define BN_QUANT_CAP_MOE_ALL_F16_LAZY_AUX \
    (BN_QUANT_CAP_MOE_ALL_F16_CACHE | BN_QUANT_CAP_LAZY_AUX_CACHE)
#define BN_QUANT_CAP_MOE_LARGE_AUX_CACHE \
    (BN_QUANT_CAP_MOE_ALL_F16_LAZY_AUX | BN_QUANT_CAP_AUX_CACHE_LARGE_BUDGET)
#define BN_QUANT_CPU_HOOKS bn_quant_matvec, bn_quant_matmul
#define BN_QUANT_NO_CPU_HOOKS NULL, NULL

static const BnQuantFormatOps g_quant_formats[] = {
    { BN_GGUF_TENSOR_F32,      "F32",      BN_QUANT_LAYOUT_DENSE,    1,   4,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_GPU_DENSE_GRAPH, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_F16,      "F16",      BN_QUANT_LAYOUT_DENSE,    1,   2,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_GPU_DENSE_GRAPH | BN_QUANT_CAP_LOGITS_F16_PATH, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_BF16,     "BF16",     BN_QUANT_LAYOUT_DENSE,    1,   2,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_AUX_CACHE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_0,     "Q4_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  18,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_CPU_REPACKED | BN_QUANT_CAP_GPU_DENSE_GRAPH | BN_QUANT_CAP_GPU_NATIVE | BN_QUANT_CAP_GPU_REPACKED | BN_QUANT_CAP_CPU_FUSED_KQUANT_GATEUP_SILU, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_1,     "Q4_1",     BN_QUANT_LAYOUT_BLOCK32,  32,  20,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_0,     "Q5_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  22,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_GPU_DENSE_GRAPH | BN_QUANT_CAP_AUX_CACHE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_1,     "Q5_1",     BN_QUANT_LAYOUT_BLOCK32,  32,  24,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q8_0,     "Q8_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  34,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_GPU_DENSE_GRAPH_NATIVE_QUANT_FORMAT | BN_QUANT_CAP_MOE_ALL_F16_LAZY_AUX | BN_QUANT_CAP_MOE_PREFERS_QUANT_ONLY | BN_QUANT_CAP_MOE_NATIVE_QUANT_ROUTE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_I2_S,     "I2_S",     BN_QUANT_LAYOUT_I2S,      4,   1,   BN_QUANT_CAP_LOADABLE_CPU_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_MXFP4,    "MXFP4",    BN_QUANT_LAYOUT_BLOCK32,  32,  17,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_TQ1_0,    "TQ1_0",    BN_QUANT_LAYOUT_BLOCK256, 256, 54,  BN_QUANT_CAP_LOADABLE_CPU_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_TQ2_0,    "TQ2_0",    BN_QUANT_LAYOUT_BLOCK256, 256, 66,  BN_QUANT_CAP_LOADABLE_CPU_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q2_K,     "Q2_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 84,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q3_K,     "Q3_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 110, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_LAZY_AUX_CACHE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_K,     "Q4_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 144, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREPARED_KQUANT_TIED_LOGITS | BN_QUANT_CAP_GPU_DENSE_GRAPH_KQUANT | BN_QUANT_CAP_MOE_LARGE_AUX_CACHE | BN_QUANT_CAP_MOE_DOWN_SMALL_KQUANT_F32_CACHE | BN_QUANT_CAP_MOE_ROUTED_KQUANT_GATEUP, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_K,     "Q5_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 176, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREPARED_KQUANT_TIED_LOGITS | BN_QUANT_CAP_GPU_DENSE_GRAPH_KQUANT | BN_QUANT_CAP_GPU_FUSED_GATEUP_BACKEND_OPT_IN | BN_QUANT_CAP_MOE_LARGE_AUX_CACHE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q6_K,     "Q6_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 210, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREPARED_KQUANT_TIED_LOGITS | BN_QUANT_CAP_GPU_DENSE_GRAPH_KQUANT_LOGITS_REFINE | BN_QUANT_CAP_MOE_LARGE_AUX_CACHE | BN_QUANT_CAP_LOGITS_KQUANT_F32_CACHE | BN_QUANT_CAP_MOE_DOWN_KQUANT_F32_CACHE | BN_QUANT_CAP_MOE_DOWN_CUBLAS_CACHE | BN_QUANT_CAP_AUX_CACHE_F16, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q8_K,     "Q8_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 292, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_GPU_DENSE_GRAPH, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ4_NL,   "IQ4_NL",   BN_QUANT_LAYOUT_BLOCK32,  32,  18,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ4_XS,   "IQ4_XS",   BN_QUANT_LAYOUT_BLOCK256, 256, 136, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_LAZY_AUX_CACHE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ3_XXS,  "IQ3_XXS",  BN_QUANT_LAYOUT_BLOCK256, 256, 98,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS | BN_QUANT_CAP_LAZY_AUX_CACHE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ3_S,    "IQ3_S",    BN_QUANT_LAYOUT_BLOCK256, 256, 114, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ2_XXS,  "IQ2_XXS",  BN_QUANT_LAYOUT_BLOCK256, 256, 66,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ2_XS,   "IQ2_XS",   BN_QUANT_LAYOUT_BLOCK256, 256, 74,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ2_S,    "IQ2_S",    BN_QUANT_LAYOUT_BLOCK256, 256, 82,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_TIED_LOGITS, BN_QUANT_CPU_HOOKS },
};

const BnQuantFormatOps *bn_quant_format_ops(int type) {
    size_t n = sizeof(g_quant_formats) / sizeof(g_quant_formats[0]);
    for (size_t i = 0; i < n; i++) {
        if (g_quant_formats[i].type == type) return &g_quant_formats[i];
    }
    return NULL;
}

int bn_quant_format_supported(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_LOADABLE);
}

int bn_quant_format_has_cap(int type, uint32_t cap) {
    const BnQuantFormatOps *ops = bn_quant_format_ops(type);
    return ops && ((ops->caps & cap) == cap);
}

int bn_quant_format_uses_embedded_scale(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_EMBEDDED_SCALE);
}

int bn_quant_format_has_embedded_tensor_scale(int type) {
    const BnQuantFormatOps *ops = bn_quant_format_ops(type);
    return ops && ops->layout == BN_QUANT_LAYOUT_I2S;
}

size_t bn_quant_embedded_tensor_scale_offset(int type, int rows, int cols) {
    if (!bn_quant_format_has_embedded_tensor_scale(type))
        return 0;
    return (size_t)rows * (size_t)cols / 4;
}

int bn_quant_format_allows_stacked_layout(int type) {
    return !bn_quant_format_has_embedded_tensor_scale(type);
}

int bn_quant_format_has_cpu_matvec(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_MATVEC);
}

int bn_quant_format_has_cpu_batch(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_BATCH);
}

int bn_quant_format_has_cpu_matmul(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_MATMUL);
}

int bn_quant_format_supports_prepared_kquant(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_PREPARED_KQUANT);
}

int bn_quant_format_can_cpu_repack(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_REPACKED);
}

int bn_quant_format_can_gpu_native(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_NATIVE);
}

int bn_quant_format_can_gpu_repack(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_REPACKED);
}

int bn_quant_format_gpu_uses_repacked_layout(int type) {
    return bn_quant_format_can_gpu_repack(type);
}

int bn_quant_format_gpu_supports_repacked_bias(int type) {
    return bn_quant_format_gpu_uses_repacked_layout(type);
}

uint32_t bn_quant_format_gpu_dispatch_tile_rows(int type) {
    return bn_quant_format_gpu_uses_repacked_layout(type) ? 8u : 32u;
}

int bn_quant_format_supports_gpu_dense_graph(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_DENSE_GRAPH);
}

int bn_quant_format_supports_gpu_dense_graph_native_quant(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_DENSE_GRAPH_NATIVE_QUANT);
}

int bn_quant_format_requires_float_kquant_fallback(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_FLOAT_KQUANT_FALLBACK);
}

int bn_quant_format_supports_native_quant_logits_refine(int type) {
    return bn_quant_format_has_cap(type,
                                   BN_QUANT_CAP_NATIVE_QUANT_LOGITS_REFINE);
}

int bn_quant_format_supports_kquant_logits_refine(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_KQUANT_LOGITS_REFINE);
}

const char *bn_quant_format_gpu_shader_name(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_F32:     return "f32";
        case BN_GGUF_TENSOR_F16:     return "f16";
        case BN_GGUF_TENSOR_I2_S:    return "i2s";
        case BN_GGUF_TENSOR_TQ1_0:   return "tq1";
        case BN_GGUF_TENSOR_TQ2_0:   return "tq2";
        case BN_GGUF_TENSOR_Q4_0:    return "q4";
        case BN_GGUF_TENSOR_Q4_1:    return "q4_1";
        case BN_GGUF_TENSOR_Q8_0:    return "q8";
        case BN_GGUF_TENSOR_BF16:    return "bf16";
        case BN_GGUF_TENSOR_Q2_K:    return "q2k";
        case BN_GGUF_TENSOR_Q3_K:    return "q3k";
        case BN_GGUF_TENSOR_Q4_K:    return "q4k";
        case BN_GGUF_TENSOR_Q5_K:    return "q5k";
        case BN_GGUF_TENSOR_Q6_K:    return "q6k";
        case BN_GGUF_TENSOR_Q8_K:    return "q8k";
        case BN_GGUF_TENSOR_IQ4_NL:  return "iq4nl";
        case BN_GGUF_TENSOR_IQ4_XS:  return "iq4xs";
        case BN_GGUF_TENSOR_IQ3_XXS: return "iq3xxs";
        case BN_GGUF_TENSOR_IQ3_S:   return "iq3s";
        case BN_GGUF_TENSOR_IQ2_XXS: return "iq2xxs";
        case BN_GGUF_TENSOR_IQ2_XS:  return "iq2xs";
        case BN_GGUF_TENSOR_IQ2_S:   return "iq2s";
        default: return NULL;
    }
}

static const int g_gpu_shader_types[] = {
    BN_GGUF_TENSOR_I2_S, BN_GGUF_TENSOR_TQ1_0, BN_GGUF_TENSOR_TQ2_0,
    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_1, BN_GGUF_TENSOR_Q8_0,
    BN_GGUF_TENSOR_F16, BN_GGUF_TENSOR_BF16,
    BN_GGUF_TENSOR_Q2_K, BN_GGUF_TENSOR_Q3_K,
    BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K,
    BN_GGUF_TENSOR_Q8_K, BN_GGUF_TENSOR_IQ4_NL, BN_GGUF_TENSOR_IQ4_XS,
    BN_GGUF_TENSOR_IQ3_XXS, BN_GGUF_TENSOR_IQ3_S, BN_GGUF_TENSOR_IQ2_XXS,
    BN_GGUF_TENSOR_IQ2_XS, BN_GGUF_TENSOR_IQ2_S,
};

int bn_quant_format_gpu_shader_type_count(int include_f32) {
    int n = (int)(sizeof(g_gpu_shader_types) / sizeof(g_gpu_shader_types[0]));
    return n + (include_f32 ? 1 : 0);
}

int bn_quant_format_gpu_shader_type_at(int index, int include_f32) {
    if (index < 0)
        return -1;
    if (include_f32) {
        if (index == 6)
            return BN_GGUF_TENSOR_F32;
        if (index > 6)
            index--;
    }
    int n = (int)(sizeof(g_gpu_shader_types) / sizeof(g_gpu_shader_types[0]));
    if (index >= n)
        return -1;
    return g_gpu_shader_types[index];
}

uint32_t bn_quant_format_gpu_split_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q5_0: return BN_GPU_CAP_MIDBIT_BLOCK32_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q4_K: return BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q5_K: return BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q8_0: return BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT;
        default: return 0;
    }
}

int bn_quant_format_can_gpu_split(int type) {
    return bn_quant_format_gpu_split_cap(type) != 0;
}

int bn_quant_format_gpu_requires_exact_silu(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_EXACT_SILU);
}

int bn_quant_format_gpu_prefers_gateup_split(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_GATEUP_SPLIT_PREFERRED);
}

uint32_t bn_quant_format_gpu_fused_gateup_silu_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q5_0: return BN_GPU_CAP_MIDBIT_BLOCK32_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q8_0: return BN_GPU_CAP_NATIVE_QUANT_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q4_K: return BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q5_K: return BN_GPU_CAP_DEINTERLEAVED_KQUANT_FUSED_GATEUP_SILU;
        default: return 0;
    }
}

int bn_quant_format_gpu_fused_gateup_requires_backend_opt_in(int type) {
    return bn_quant_format_has_cap(
        type, BN_QUANT_CAP_GPU_FUSED_GATEUP_BACKEND_OPT_IN);
}

int bn_quant_format_gpu_allows_gateup_split_activation(int type,
                                                       int act_type) {
    return act_type != 1 || type != BN_GGUF_TENSOR_Q4_K;
}

uint32_t bn_quant_format_gpu_matvec_kquant_dot_flag(int type, int enabled) {
    return enabled && type == BN_GGUF_TENSOR_Q4_K
        ? BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT
        : 0u;
}

uint32_t bn_quant_format_gpu_matvec_exact_kquant_flag(int type, int enabled) {
    return enabled && type == BN_GGUF_TENSOR_Q6_K
        ? BN_QUANT_GPU_MATVEC_FLAG_EXACT_KQUANT
        : 0u;
}

int bn_quant_format_dense_f32_type(void) {
    return BN_GGUF_TENSOR_F32;
}

int bn_quant_format_gpu_float_buffer_type(void) {
    return BN_GGUF_TENSOR_F32;
}

int bn_quant_format_is_f32(int type) {
    return type == BN_GGUF_TENSOR_F32;
}

int bn_quant_format_can_convert_dense_to_f32(int type) {
    return type == BN_GGUF_TENSOR_F16 ||
           type == BN_GGUF_TENSOR_BF16;
}

int bn_quant_format_convert_dense_to_f32(int type, const void *src,
                                         float *dst, int n) {
    if (!src || !dst || n < 0)
        return -1;
    if (type == BN_GGUF_TENSOR_BF16) {
        const uint16_t *s = (const uint16_t *)src;
        for (int i = 0; i < n; i++)
            dst[i] = bn_bf16_to_fp32(s[i]);
        return 0;
    }
    if (type == BN_GGUF_TENSOR_F16) {
        const uint16_t *s = (const uint16_t *)src;
        for (int i = 0; i < n; i++)
            dst[i] = bn_fp16_to_fp32(s[i]);
        return 0;
    }
    return -1;
}

int bn_quant_format_logits_kquant_f32_cache_supported(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_LOGITS_KQUANT_F32_CACHE);
}

int bn_quant_format_moe_all_f16_cache_supported(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_MOE_ALL_F16_CACHE);
}

int bn_quant_format_moe_down_kquant_f32_cache_supported(int type) {
    return bn_quant_format_has_cap(type,
                                   BN_QUANT_CAP_MOE_DOWN_KQUANT_F32_CACHE);
}

int bn_quant_format_moe_down_cublas_cache_supported(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_MOE_DOWN_CUBLAS_CACHE);
}

int bn_quant_format_moe_down_cublas_cache_elem_bytes(int type,
                                                     int down_kquant_f16_cache) {
    if (!bn_quant_format_moe_down_cublas_cache_supported(type))
        return 0;
    return down_kquant_f16_cache ? (int)sizeof(uint16_t) : (int)sizeof(float);
}

int bn_quant_format_moe_down_small_kquant_f32_cache_supported(int type) {
    return bn_quant_format_has_cap(
        type, BN_QUANT_CAP_MOE_DOWN_SMALL_KQUANT_F32_CACHE);
}

int bn_quant_format_avoids_quant_matmul_on_f16_input(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

int bn_quant_format_supports_requested_quant_matmul(int type) {
    return type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_moe_quant_only_after_cache(int type,
                                               int native_quant_f16_cache) {
    return !native_quant_f16_cache ||
           !bn_quant_format_moe_prefers_quant_only(type);
}

int bn_quant_format_supports_lazy_moe_aux_cache(int type) {
    return bn_quant_format_has_cap(type,
                                   BN_QUANT_CAP_LAZY_MOE_AUX_CACHE_CANDIDATE);
}

int bn_quant_format_moe_prefers_quant_only(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_MOE_PREFERS_QUANT_ONLY);
}

int bn_quant_format_aux_cache_supported(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_AUX_CACHE);
}

int bn_quant_format_aux_cache_can_use_f16(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_AUX_CACHE_F16);
}

int bn_quant_format_aux_cache_uses_f32(int type, int down_kquant_f16_cache) {
    return bn_quant_format_aux_cache_can_use_f16(type) &&
           !down_kquant_f16_cache;
}

int bn_quant_format_aux_cache_prefers_large_budget(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_AUX_CACHE_LARGE_BUDGET);
}

int bn_quant_format_uses_f16_logits_path(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_LOGITS_F16_PATH);
}

int bn_quant_format_tied_logits_uses_quant_path(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_TIED_LOGITS_QUANT_PATH);
}

int bn_quant_format_supports_logits_i8_cache(int type) {
    return bn_quant_format_uses_f16_logits_path(type);
}

int bn_quant_format_tied_logits_uses_f16_path(int type) {
    return bn_quant_format_uses_f16_logits_path(type);
}

int bn_quant_format_tied_logits_i8_weight_type(void) {
    return BN_GGUF_TENSOR_Q8_0;
}

int bn_quant_format_tied_logits_f16_weight_type(void) {
    return BN_GGUF_TENSOR_F16;
}

int bn_quant_format_tied_logits_f32_weight_type(void) {
    return BN_GGUF_TENSOR_F32;
}

int bn_quant_format_supports_moe_asymmetric_kquant_down_route(int gate_type,
                                                              int up_type,
                                                              int down_type,
                                                              int allow_asymmetric_kquant_down) {
    return bn_quant_format_has_cap(gate_type,
                                   BN_QUANT_CAP_MOE_ROUTED_KQUANT_GATEUP) &&
           bn_quant_format_has_cap(up_type,
                                   BN_QUANT_CAP_MOE_ROUTED_KQUANT_GATEUP) &&
           (bn_quant_format_has_cap(down_type,
                                    BN_QUANT_CAP_MOE_DOWN_KQUANT_F32_CACHE) ||
            (allow_asymmetric_kquant_down &&
             bn_quant_format_has_cap(down_type,
                                     BN_QUANT_CAP_MOE_DOWN_SMALL_KQUANT_F32_CACHE)));
}

int bn_quant_format_supports_moe_routed_kquant_gateup(int gate_type,
                                                      int up_type) {
    return bn_quant_format_has_cap(gate_type,
                                   BN_QUANT_CAP_MOE_ROUTED_KQUANT_GATEUP) &&
           bn_quant_format_has_cap(up_type,
                                   BN_QUANT_CAP_MOE_ROUTED_KQUANT_GATEUP);
}

int bn_quant_format_supports_cpu_fused_kquant_gateup_silu(int gate_type,
                                                          int up_type) {
    return bn_quant_format_has_cap(gate_type,
                                   BN_QUANT_CAP_CPU_FUSED_KQUANT_GATEUP_SILU) &&
           bn_quant_format_has_cap(up_type,
                                   BN_QUANT_CAP_CPU_FUSED_KQUANT_GATEUP_SILU);
}

int bn_quant_format_same_quant_format_pair_stackable(int left_type,
                                                     int right_type) {
    return left_type == right_type;
}

int bn_quant_format_supports_shared_gateup_batch(int shared_gate_type,
                                                 int shared_up_type,
                                                 int batch_type) {
    if (bn_quant_format_same_quant_format_pair_stackable(shared_gate_type,
                                                         batch_type) &&
        bn_quant_format_same_quant_format_pair_stackable(shared_up_type,
                                                         batch_type))
        return 1;
    return bn_quant_format_supports_prepared_kquant(shared_gate_type) &&
           bn_quant_format_supports_prepared_kquant(shared_up_type) &&
           bn_quant_format_supports_prepared_kquant(batch_type);
}

int bn_quant_format_supports_moe_native_quant_route(int gate_type,
                                                    int up_type,
                                                    int down_type) {
    return bn_quant_format_has_cap(gate_type,
                                   BN_QUANT_CAP_MOE_NATIVE_QUANT_ROUTE) &&
           bn_quant_format_has_cap(up_type,
                                   BN_QUANT_CAP_MOE_NATIVE_QUANT_ROUTE) &&
           bn_quant_format_has_cap(down_type,
                                   BN_QUANT_CAP_MOE_NATIVE_QUANT_ROUTE);
}

BnQuantMatvecFn bn_quant_format_matvec(int type) {
    const BnQuantFormatOps *ops = bn_quant_format_ops(type);
    return ops ? ops->matvec : NULL;
}

BnQuantMatmulFn bn_quant_format_matmul(int type) {
    const BnQuantFormatOps *ops = bn_quant_format_ops(type);
    return ops ? ops->matmul : NULL;
}

size_t bn_quant_format_data_size(int type, int rows, int cols) {
    const BnQuantFormatOps *ops = bn_quant_format_ops(type);
    if (!ops || rows <= 0 || cols <= 0) return 0;

    size_t nelements = (size_t)rows * (size_t)cols;
    if (ops->layout == BN_QUANT_LAYOUT_I2S)
        return nelements / 4 + 4;
    if (ops->layout == BN_QUANT_LAYOUT_DENSE)
        return nelements * ops->bytes_per_block;
    if (ops->block_elems <= 0 || nelements % (size_t)ops->block_elems != 0)
        return 0;
    return (nelements / (size_t)ops->block_elems) * ops->bytes_per_block;
}
