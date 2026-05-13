#include "quant.h"
#include "gguf.h"

#define BN_QUANT_CAP_CPU_ALL \
    (BN_QUANT_CAP_CPU_MATVEC | BN_QUANT_CAP_CPU_BATCH | BN_QUANT_CAP_CPU_MATMUL)
#define BN_QUANT_CAP_LOADABLE_CPU \
    (BN_QUANT_CAP_LOADABLE | BN_QUANT_CAP_CPU_ALL)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED \
    (BN_QUANT_CAP_LOADABLE_CPU | BN_QUANT_CAP_EMBEDDED_SCALE)
#define BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K \
    (BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_CPU_PREQ8K)
#define BN_QUANT_CPU_HOOKS bn_quant_matvec, bn_quant_matmul
#define BN_QUANT_NO_CPU_HOOKS NULL, NULL

static const BnQuantFormatOps g_quant_formats[] = {
    { BN_GGUF_TENSOR_F32,      "F32",      BN_QUANT_LAYOUT_DENSE,    1,   4,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_F16,      "F16",      BN_QUANT_LAYOUT_DENSE,    1,   2,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_BF16,     "BF16",     BN_QUANT_LAYOUT_DENSE,    1,   2,   BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_0,     "Q4_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  18,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED | BN_QUANT_CAP_CPU_REPACKED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_1,     "Q4_1",     BN_QUANT_LAYOUT_BLOCK32,  32,  20,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_0,     "Q5_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  22,  0, BN_QUANT_NO_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_1,     "Q5_1",     BN_QUANT_LAYOUT_BLOCK32,  32,  24,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q8_0,     "Q8_0",     BN_QUANT_LAYOUT_BLOCK32,  32,  34,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_I2_S,     "I2_S",     BN_QUANT_LAYOUT_I2S,      4,   1,   BN_QUANT_CAP_LOADABLE_CPU, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_TQ1_0,    "TQ1_0",    BN_QUANT_LAYOUT_BLOCK256, 256, 54,  BN_QUANT_CAP_LOADABLE_CPU, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_TQ2_0,    "TQ2_0",    BN_QUANT_LAYOUT_BLOCK256, 256, 66,  BN_QUANT_CAP_LOADABLE_CPU, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q2_K,     "Q2_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 84,  BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q3_K,     "Q3_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 110, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q4_K,     "Q4_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 144, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q5_K,     "Q5_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 176, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q6_K,     "Q6_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 210, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K, BN_QUANT_CPU_HOOKS },
    { BN_GGUF_TENSOR_Q8_K,     "Q8_K",     BN_QUANT_LAYOUT_BLOCK256, 256, 292, BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED, BN_QUANT_CPU_HOOKS },
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
