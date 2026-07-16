#include "quant.h"
#include "gguf.h"
#include "gpu_backend.h"

#define BN_QUANT_CAP_CPU_ALL \
    (BN_QUANT_CAP_CPU_MATVEC | BN_QUANT_CAP_CPU_BATCH | BN_QUANT_CAP_CPU_MATMUL)
#define BN_QUANT_CAP_LOADABLE_CPU \
    (BN_QUANT_CAP_LOADABLE | BN_QUANT_CAP_CPU_ALL)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED \
    (BN_QUANT_CAP_LOADABLE_CPU | BN_QUANT_CAP_EMBEDDED_SCALE)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K \
    (BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_CPU_PREQ8K)
#define BN_QUANT_CAP_GPU_SMALL_DENSE_Q8_FORMAT \
    (BN_QUANT_CAP_GPU_SMALL_DENSE | BN_QUANT_CAP_GPU_SMALL_DENSE_Q8 | \
     BN_QUANT_CAP_Q8_LOGITS_REFINE)
#define BN_QUANT_CAP_GPU_SMALL_DENSE_KQUANT \
    (BN_QUANT_CAP_GPU_SMALL_DENSE | BN_QUANT_CAP_FLOAT_KQUANT_FALLBACK)
#define BN_QUANT_CAP_GPU_SMALL_DENSE_Q6K \
    (BN_QUANT_CAP_GPU_SMALL_DENSE_KQUANT | BN_QUANT_CAP_Q6_LOGITS_REFINE)
#define BN_QUANT_CPU_HOOKS bn_quant_matvec, bn_quant_matmul
#define BN_QUANT_NO_CPU_HOOKS NULL, NULL

static const BnQuantFormatOps g_quant_formats[] = {
    { BN_GGUF_TENSOR_F32,      "F32",      BN_QUANT_LAYOUT_DENSE,    1,   4,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_GPU_SMALL_DENSE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_F16,      "F16",      BN_QUANT_LAYOUT_DENSE,    1,   2,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_GPU_SMALL_DENSE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_BF16,     "BF16",     BN_QUANT_LAYOUT_DENSE,    1,   2,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_0,     "Q4_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  18,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_CPU_REPACKED | BN_QUANT_CAP_GPU_SMALL_DENSE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_1,     "Q4_1",     BN_QUANT_LAYOUT_BLOCK32,  32,  20,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_0,     "Q5_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  22,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_GPU_SMALL_DENSE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_1,     "Q5_1",     BN_QUANT_LAYOUT_BLOCK32,  32,  24,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q8_0,     "Q8_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  34,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_GPU_SMALL_DENSE_Q8_FORMAT, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_I2_S,     "I2_S",     BN_QUANT_LAYOUT_I2S,      4,   1,   BN_QUANT_CAP_LOADABLE_CPU, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_MXFP4,    "MXFP4",    BN_QUANT_LAYOUT_BLOCK32,  32,  17,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_TQ1_0,    "TQ1_0",    BN_QUANT_LAYOUT_BLOCK256, 256, 54,  BN_QUANT_CAP_LOADABLE_CPU, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_TQ2_0,    "TQ2_0",    BN_QUANT_LAYOUT_BLOCK256, 256, 66,  BN_QUANT_CAP_LOADABLE_CPU, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q2_K,     "Q2_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 84,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q3_K,     "Q3_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 110, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_K,     "Q4_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 144, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K | BN_QUANT_CAP_GPU_SMALL_DENSE_KQUANT, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_K,     "Q5_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 176, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K | BN_QUANT_CAP_GPU_SMALL_DENSE_KQUANT, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q6_K,     "Q6_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 210, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K | BN_QUANT_CAP_GPU_SMALL_DENSE_Q6K, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q8_K,     "Q8_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 292, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_GPU_SMALL_DENSE, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ4_NL,   "IQ4_NL",   BN_QUANT_LAYOUT_BLOCK32,  32,  18,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ4_XS,   "IQ4_XS",   BN_QUANT_LAYOUT_BLOCK256, 256, 136, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ3_XXS,  "IQ3_XXS",  BN_QUANT_LAYOUT_BLOCK256, 256, 98,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ3_S,    "IQ3_S",    BN_QUANT_LAYOUT_BLOCK256, 256, 114, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ2_XXS,  "IQ2_XXS",  BN_QUANT_LAYOUT_BLOCK256, 256, 66,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ2_XS,   "IQ2_XS",   BN_QUANT_LAYOUT_BLOCK256, 256, 74,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_IQ2_S,    "IQ2_S",    BN_QUANT_LAYOUT_BLOCK256, 256, 82,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
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

int bn_quant_format_has_cpu_matvec(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_MATVEC);
}

int bn_quant_format_has_cpu_batch(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_BATCH);
}

int bn_quant_format_has_cpu_matmul(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_MATMUL);
}

int bn_quant_format_can_preq8k(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_PREQ8K);
}

int bn_quant_format_can_cpu_repack(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_CPU_REPACKED);
}

int bn_quant_format_supports_gpu_small_dense(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_SMALL_DENSE);
}

int bn_quant_format_supports_gpu_small_dense_q8(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_GPU_SMALL_DENSE_Q8);
}

int bn_quant_format_is_float_kquant_fallback_candidate(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_FLOAT_KQUANT_FALLBACK);
}

int bn_quant_format_supports_q8_logits_refine(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_Q8_LOGITS_REFINE);
}

int bn_quant_format_supports_q6_logits_refine(int type) {
    return bn_quant_format_has_cap(type, BN_QUANT_CAP_Q6_LOGITS_REFINE);
}

uint32_t bn_quant_format_gpu_split_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_Q4_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q5_0: return BN_GPU_CAP_Q5_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q4_K: return BN_GPU_CAP_Q4K_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q5_K: return BN_GPU_CAP_Q5K_MATVEC_SPLIT;
        case BN_GGUF_TENSOR_Q8_0: return BN_GPU_CAP_Q8_MATVEC_SPLIT;
        default: return 0;
    }
}

int bn_quant_format_can_gpu_split(int type) {
    return bn_quant_format_gpu_split_cap(type) != 0;
}

int bn_quant_format_gpu_requires_exact_silu(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

int bn_quant_format_gpu_prefers_gateup_split(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

uint32_t bn_quant_format_gpu_fused_gateup_silu_cap(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_Q4_0: return BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q5_0: return BN_GPU_CAP_Q5_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q8_0: return BN_GPU_CAP_Q8_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q4_K: return BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
        case BN_GGUF_TENSOR_Q5_K: return BN_GPU_CAP_Q5K_FUSED_GATEUP_SILU;
        default: return 0;
    }
}

int bn_quant_format_gpu_fused_gateup_requires_cuda_opt_in(int type) {
    return type == BN_GGUF_TENSOR_Q5_K;
}

int bn_quant_format_gpu_allows_gateup_split_activation(int type,
                                                       int act_type) {
    return act_type != 1 || type != BN_GGUF_TENSOR_Q4_K;
}

uint32_t bn_quant_format_gpu_matvec_q8k_dot_flag(int type, int enabled) {
    return enabled && type == BN_GGUF_TENSOR_Q4_K
        ? BN_QUANT_GPU_MATVEC_FLAG_Q8K_DOT
        : 0u;
}

uint32_t bn_quant_format_gpu_matvec_exact_q6k_flag(int type, int enabled) {
    return enabled && type == BN_GGUF_TENSOR_Q6_K
        ? BN_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K
        : 0u;
}

int bn_quant_format_cuda_logits_q6_f32_cache_supported(int type) {
    return type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_cuda_moe_all_f16_cache_supported(int type) {
    return type == BN_GGUF_TENSOR_Q8_0 ||
           type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_cuda_moe_down_q6_f32_cache_supported(int type) {
    return type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_cuda_moe_down_cublas_cache_supported(int type) {
    return type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes(int type,
                                                          int q6_as_f16) {
    if (!bn_quant_format_cuda_moe_down_cublas_cache_supported(type))
        return 0;
    return q6_as_f16 ? (int)sizeof(uint16_t) : (int)sizeof(float);
}

int bn_quant_format_cuda_moe_down_q4_f32_cache_supported(int type) {
    return type == BN_GGUF_TENSOR_Q4_K;
}

int bn_quant_format_cuda_moe_quant_only_after_cache(int type,
                                                    int q8_f16_cache) {
    return type != BN_GGUF_TENSOR_Q8_0 || !q8_f16_cache;
}

int bn_quant_format_cuda_lazy_moe_aux_cache_candidate(int type) {
    return type == BN_GGUF_TENSOR_Q3_K ||
           type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K ||
           type == BN_GGUF_TENSOR_Q8_0 ||
           type == BN_GGUF_TENSOR_IQ3_XXS ||
           type == BN_GGUF_TENSOR_IQ4_XS;
}

int bn_quant_format_cuda_moe_prefers_quant_only(int type) {
    return type == BN_GGUF_TENSOR_Q8_0;
}

int bn_quant_format_cuda_aux_cache_supported(int type) {
    return type == BN_GGUF_TENSOR_Q8_0 ||
           type == BN_GGUF_TENSOR_Q5_0 ||
           type == BN_GGUF_TENSOR_BF16 ||
           type == BN_GGUF_TENSOR_Q3_K ||
           type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K ||
           type == BN_GGUF_TENSOR_IQ3_XXS ||
           type == BN_GGUF_TENSOR_IQ4_XS;
}

int bn_quant_format_cuda_aux_cache_can_use_f16(int type) {
    return type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_cuda_aux_cache_uses_f32(int type, int q6_as_f16) {
    return type == BN_GGUF_TENSOR_Q6_K && !q6_as_f16;
}

int bn_quant_format_cuda_aux_cache_prefers_large_budget(int type) {
    return type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K;
}

int bn_quant_format_uses_f16_logits_path(int type) {
    return type == BN_GGUF_TENSOR_F16;
}

int bn_quant_format_tied_logits_uses_quant_path(int type) {
    return bn_quant_format_supported(type) &&
           type != BN_GGUF_TENSOR_F16 &&
           type != BN_GGUF_TENSOR_F32;
}

int bn_quant_format_supports_logits_i8_cache(int type) {
    return type == BN_GGUF_TENSOR_F16;
}

int bn_quant_format_tied_logits_uses_f16_path(int type) {
    return type == BN_GGUF_TENSOR_F16;
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

int bn_quant_format_supports_moe_q4_down_route(int gate_type,
                                               int up_type,
                                               int down_type,
                                               int allow_q4_down) {
    return gate_type == BN_GGUF_TENSOR_Q4_K &&
           up_type == BN_GGUF_TENSOR_Q4_K &&
           (down_type == BN_GGUF_TENSOR_Q6_K ||
            (allow_q4_down && down_type == BN_GGUF_TENSOR_Q4_K));
}

int bn_quant_format_supports_moe_q4_gateup(int gate_type, int up_type) {
    return gate_type == BN_GGUF_TENSOR_Q4_K &&
           up_type == BN_GGUF_TENSOR_Q4_K;
}

int bn_quant_format_supports_cpu_fused_q4_gateup_silu(int gate_type,
                                                      int up_type) {
    return gate_type == BN_GGUF_TENSOR_Q4_0 &&
           up_type == BN_GGUF_TENSOR_Q4_0;
}

int bn_quant_format_pair_same_format(int left_type, int right_type) {
    return left_type == right_type;
}

int bn_quant_format_supports_moe_q8_route(int gate_type,
                                          int up_type,
                                          int down_type) {
    return gate_type == BN_GGUF_TENSOR_Q8_0 &&
           up_type == BN_GGUF_TENSOR_Q8_0 &&
           down_type == BN_GGUF_TENSOR_Q8_0;
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
