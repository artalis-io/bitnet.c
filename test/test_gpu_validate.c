/*
 * test_gpu_validate.c — comprehensive GPU vs CPU validation for all 22 quant types
 *
 * For each quant type: generates synthetic weight data, runs both GPU matvec
 * and CPU scalar matvec, and verifies the results match within tolerance.
 * Also validates matmul (batch matvec) for a few representative types.
 *
 * Requires: BN_ENABLE_GPU=1 and `make fetch-wgpu` before building.
 */

#ifdef BN_ENABLE_GPU

#include "gpu_wgpu.h"
#include "gpu_backend.h"
#include "quant.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROWS 4
#define TOL 1e-2f

static const char *SHADER_DIR = "shaders";

/* ── Type info table ──────────────────────────────────────────────── */

typedef struct {
    const char *name;
    int type;
    int block_elems;    /* elements per block */
    int cols;           /* test matrix columns (multiple of block_elems) */
} TypeInfo;

/* All 22 quant types.
 * F16 and F32 have no CPU matvec kernel in dispatch.c, so they are
 * listed but will be skipped by the CPU path (bn_quant_matvec does not
 * handle them). BF16 does have a kernel. */
static const TypeInfo ALL_TYPES[] = {
    { "I2_S",      BN_GGUF_TENSOR_I2_S,     128, 256 },
    { "TQ1_0",     BN_GGUF_TENSOR_TQ1_0,    256, 256 },
    { "TQ2_0",     BN_GGUF_TENSOR_TQ2_0,    256, 256 },
    { "Q4_0",      BN_GGUF_TENSOR_Q4_0,      32, 256 },
    { "Q4_1",      BN_GGUF_TENSOR_Q4_1,      32, 256 },
    { "Q8_0",      BN_GGUF_TENSOR_Q8_0,      32, 256 },
    { "BF16",      BN_GGUF_TENSOR_BF16,       1, 256 },
    { "F16",       BN_GGUF_TENSOR_F16,        1, 256 },
    { "F32",       BN_GGUF_TENSOR_F32,        1, 256 },
    { "Q2_K",      BN_GGUF_TENSOR_Q2_K,     256, 256 },
    { "Q3_K",      BN_GGUF_TENSOR_Q3_K,     256, 256 },
    { "Q4_K",      BN_GGUF_TENSOR_Q4_K,     256, 256 },
    { "Q5_K",      BN_GGUF_TENSOR_Q5_K,     256, 256 },
    { "Q6_K",      BN_GGUF_TENSOR_Q6_K,     256, 256 },
    { "Q8_K",      BN_GGUF_TENSOR_Q8_K,     256, 256 },
    { "IQ4_NL",    BN_GGUF_TENSOR_IQ4_NL,    32, 256 },
    { "IQ4_XS",    BN_GGUF_TENSOR_IQ4_XS,  256, 256 },
    { "IQ3_XXS",   BN_GGUF_TENSOR_IQ3_XXS, 256, 256 },
    { "IQ3_S",     BN_GGUF_TENSOR_IQ3_S,   256, 256 },
    { "IQ2_XXS",   BN_GGUF_TENSOR_IQ2_XXS, 256, 256 },
    { "IQ2_XS",    BN_GGUF_TENSOR_IQ2_XS,  256, 256 },
    { "IQ2_S",     BN_GGUF_TENSOR_IQ2_S,   256, 256 },
};
#define N_TYPES (int)(sizeof(ALL_TYPES) / sizeof(ALL_TYPES[0]))

/* ── Helpers ──────────────────────────────────────────────────────── */

/* Convert float to BF16 (truncation, matching hardware behavior) */
static uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

/* ── Weight data generators ───────────────────────────────────────── */

/*
 * make_weight_data: allocate and fill valid synthetic weight data for a
 * given quant type. Returns malloc'd buffer. Sets *tensor_scale for
 * types that need per-tensor scale (I2_S, TQ1_0, TQ2_0). Returns NULL
 * if the type is not supported (F16, F32 have no CPU matvec kernel).
 */
static void *make_weight_data(int type, int rows, int cols, float *tensor_scale) {
    size_t nelements = (size_t)rows * cols;
    *tensor_scale = 1.0f;

    switch (type) {

    /* ── I2_S: 2-bit ternary, 128 elements per interleaved block ──── */
    case BN_GGUF_TENSOR_I2_S: {
        size_t data_size = nelements / 4 + 4;
        uint8_t *data = calloc(1, data_size);
        if (!data) return NULL;
        /* 0xAA = 10101010 binary = [2,2,2,2] per byte = all +1 ternary */
        memset(data, 0xAA, nelements / 4);
        float scale = 1.0f;
        memcpy(data + nelements / 4, &scale, sizeof(float));
        *tensor_scale = scale;
        return data;
    }

    /* ── TQ1_0: base-3 ternary, 256 elements per block, 54 bytes ──── */
    case BN_GGUF_TENSOR_TQ1_0: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockTQ1);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockTQ1 *blk = (BnBlockTQ1 *)(data + (size_t)b * block_size);
            /* qs=0: decode(0) base-3 gives all-zero trits → weight = trit - 1 = -1 */
            memset(blk->qs, 0, sizeof(blk->qs));
            memset(blk->qh, 0, sizeof(blk->qh));
            blk->d = bn_fp32_to_fp16(1.0f);
        }
        return data;
    }

    /* ── TQ2_0: 2-bit ternary, 256 elements per block, 66 bytes ───── */
    case BN_GGUF_TENSOR_TQ2_0: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockTQ2);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockTQ2 *blk = (BnBlockTQ2 *)(data + (size_t)b * block_size);
            /* 0xAA = each pair of bits is 10 = value 2, decoded as 2-1 = +1 */
            memset(blk->qs, 0xAA, sizeof(blk->qs));
            blk->d = bn_fp32_to_fp16(1.0f);
        }
        return data;
    }

    /* ── Q4_0: 4-bit, 32 elements per block, 18 bytes ─────────────── */
    case BN_GGUF_TENSOR_Q4_0: {
        int n_blocks = (int)(nelements / 32);
        size_t block_size = sizeof(BnBlockQ4_0);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        uint16_t fp16_d = bn_fp32_to_fp16(0.5f);
        /* nibble = 12 → value = (12 - 8) * 0.5 = 2.0 */
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4_0 *blk = (BnBlockQ4_0 *)(data + (size_t)b * block_size);
            blk->d = fp16_d;
            memset(blk->qs, 0xCC, 16);  /* nibbles: 0xC=12 both high and low */
        }
        return data;
    }

    /* ── Q4_1: 4-bit with min, 32 elements per block, 20 bytes ────── */
    case BN_GGUF_TENSOR_Q4_1: {
        int n_blocks = (int)(nelements / 32);
        size_t block_size = sizeof(BnBlockQ4_1);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        uint16_t fp16_d = bn_fp32_to_fp16(0.5f);
        uint16_t fp16_m = bn_fp32_to_fp16(1.0f);
        /* nibble = 4 → value = 0.5*4 + 1.0 = 3.0 */
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4_1 *blk = (BnBlockQ4_1 *)(data + (size_t)b * block_size);
            blk->d = fp16_d;
            blk->m = fp16_m;
            memset(blk->qs, 0x44, 16);  /* both nibbles = 4 */
        }
        return data;
    }

    /* ── Q8_0: 8-bit, 32 elements per block, 34 bytes ─────────────── */
    case BN_GGUF_TENSOR_Q8_0: {
        int n_blocks = (int)(nelements / 32);
        size_t block_size = sizeof(BnBlockQ8_0);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        uint16_t fp16_d = bn_fp32_to_fp16(0.5f);
        /* qs = 10 → value = 10 * 0.5 = 5.0 */
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ8_0 *blk = (BnBlockQ8_0 *)(data + (size_t)b * block_size);
            blk->d = fp16_d;
            memset(blk->qs, 10, 32);
        }
        return data;
    }

    /* ── BF16: brain float16, 1 element per "block" ───────────────── */
    case BN_GGUF_TENSOR_BF16: {
        uint16_t *data = calloc(nelements, sizeof(uint16_t));
        if (!data) return NULL;
        uint16_t val = fp32_to_bf16(2.0f);
        for (size_t i = 0; i < nelements; i++)
            data[i] = val;
        return data;
    }

    /* ── F16: IEEE 754 float16, 1 element per "block" ─────────────── */
    case BN_GGUF_TENSOR_F16: {
        uint16_t *data = calloc(nelements, sizeof(uint16_t));
        if (!data) return NULL;
        uint16_t val = bn_fp32_to_fp16(2.0f);
        for (size_t i = 0; i < nelements; i++)
            data[i] = val;
        return data;
    }

    /* ── F32: 32-bit float, 1 element per "block" ─────────────────── */
    case BN_GGUF_TENSOR_F32: {
        float *data = calloc(nelements, sizeof(float));
        if (!data) return NULL;
        for (size_t i = 0; i < nelements; i++)
            data[i] = 2.0f;
        return data;
    }

    /* ── Q2_K: 2-bit k-quant, 256 elements per block, 84 bytes ────── */
    case BN_GGUF_TENSOR_Q2_K: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockQ2K);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ2K *blk = (BnBlockQ2K *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            blk->dmin = bn_fp32_to_fp16(0.0f);
            /* scales: 4-bit scale | 4-bit min per 16-element group.
             * Set scale=1, min=0 → byte = 0x01 */
            memset(blk->scales, 0x01, 16);
            /* qs: all bits = 01 → quant value = 1 per element.
             * 0x55 = 01010101 = [1,1,1,1] per byte */
            memset(blk->qs, 0x55, 64);
        }
        return data;
    }

    /* ── Q3_K: 3-bit k-quant, 256 elements per block, 110 bytes ───── */
    case BN_GGUF_TENSOR_Q3_K: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockQ3K);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ3K *blk = (BnBlockQ3K *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            /* hmask = 0: high bit of each 3-bit quant is 0 */
            memset(blk->hmask, 0, 32);
            /* qs: low 2 bits. 0x55 = [01,01,01,01] = quant low bits = 1 */
            memset(blk->qs, 0x55, 64);
            /* scales: 12 bytes of packed 6-bit values. Set all to small known value */
            memset(blk->scales, 0x01, 12);
        }
        return data;
    }

    /* ── Q4_K: 4-bit k-quant, 256 elements per block, 144 bytes ───── */
    case BN_GGUF_TENSOR_Q4_K: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockQ4K);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4K *blk = (BnBlockQ4K *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            blk->dmin = bn_fp32_to_fp16(0.0f);
            /* scales[0..11]: 8 packed 6-bit scales + 8 packed 6-bit mins.
             * Set all to 0x01 for small known values */
            memset(blk->scales, 0x01, 12);
            /* qs: all nibbles = 8 → value = (8 * sc) * d */
            memset(blk->qs, 0x88, 128);
        }
        return data;
    }

    /* ── Q5_K: 5-bit k-quant, 256 elements per block, 176 bytes ───── */
    case BN_GGUF_TENSOR_Q5_K: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockQ5K);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ5K *blk = (BnBlockQ5K *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            blk->dmin = bn_fp32_to_fp16(0.0f);
            memset(blk->scales, 0x01, 12);
            memset(blk->qh, 0, 32);  /* high bit = 0 for all */
            /* qs: all nibbles = 8 */
            memset(blk->qs, 0x88, 128);
        }
        return data;
    }

    /* ── Q6_K: 6-bit k-quant, 256 elements per block, 210 bytes ───── */
    case BN_GGUF_TENSOR_Q6_K: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockQ6K);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ6K *blk = (BnBlockQ6K *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            /* ql: lower 4 bits of each quant. Set to known value */
            memset(blk->ql, 0x22, 128);  /* nibbles = 2 */
            /* qh: upper 2 bits. 0 = upper bits are 0 */
            memset(blk->qh, 0, 64);
            /* scales: int8 per 16-element sub-block. Set to 1 */
            memset(blk->scales, 1, 16);
        }
        return data;
    }

    /* ── Q8_K: 8-bit k-quant, 256 elements per block, 292 bytes ───── */
    case BN_GGUF_TENSOR_Q8_K: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockQ8K);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ8K *blk = (BnBlockQ8K *)(data + (size_t)b * block_size);
            blk->d = 1.0f;  /* Q8_K uses float32 scale, not FP16 */
            /* qs: all = 10 → value = 10 * 1.0 = 10.0 */
            memset(blk->qs, 10, 256);
            /* bsums: sum of 16 consecutive qs = 16 * 10 = 160 */
            for (int i = 0; i < 16; i++)
                blk->bsums[i] = 160;
        }
        return data;
    }

    /* ── IQ4_NL: 4-bit non-linear codebook, 32 elements, 18 bytes ── */
    case BN_GGUF_TENSOR_IQ4_NL: {
        int n_blocks = (int)(nelements / 32);
        size_t block_size = sizeof(BnBlockIQ4NL);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockIQ4NL *blk = (BnBlockIQ4NL *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            /* nibble = 0 → codebook[0]. Use 0x00 for simplicity */
            memset(blk->qs, 0x00, 16);
        }
        return data;
    }

    /* ── IQ4_XS: 4-bit non-linear + sub-block scales, 256 elems ──── */
    case BN_GGUF_TENSOR_IQ4_XS: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockIQ4XS);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockIQ4XS *blk = (BnBlockIQ4XS *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            blk->scales_h = 0;
            /* scales_l: 4 bytes, low 4 bits of 8 scales. Set to small value */
            memset(blk->scales_l, 0x11, 4);  /* scale nibbles = 1 */
            /* qs: nibble = 0 → codebook[0] */
            memset(blk->qs, 0x00, 128);
        }
        return data;
    }

    /* ── IQ3_XXS: 3-bit codebook, 256 elements per block, 98 bytes ── */
    case BN_GGUF_TENSOR_IQ3_XXS: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockIQ3XXS);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockIQ3XXS *blk = (BnBlockIQ3XXS *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            /* qs: interleaved grid indices + packed signs/scales.
             * All zeros → grid index 0, signs = 0 */
            memset(blk->qs, 0, sizeof(blk->qs));
        }
        return data;
    }

    /* ── IQ3_S: 3-bit codebook with separate signs, 256 elems ──────── */
    case BN_GGUF_TENSOR_IQ3_S: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockIQ3S);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockIQ3S *blk = (BnBlockIQ3S *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            memset(blk->qs, 0, sizeof(blk->qs));
            memset(blk->qh, 0, sizeof(blk->qh));
            memset(blk->signs, 0, sizeof(blk->signs));
            /* scales: 4-bit nibble-packed. Set to small known value */
            memset(blk->scales, 0x11, sizeof(blk->scales));
        }
        return data;
    }

    /* ── IQ2_XXS: 2-bit codebook, 256 elements per block, 66 bytes ── */
    case BN_GGUF_TENSOR_IQ2_XXS: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockIQ2XXS);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockIQ2XXS *blk = (BnBlockIQ2XXS *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            /* qs: packed grid indices + signs/scales. All zeros */
            memset(blk->qs, 0, sizeof(blk->qs));
        }
        return data;
    }

    /* ── IQ2_XS: 2-bit codebook + explicit scales, 256 elems ──────── */
    case BN_GGUF_TENSOR_IQ2_XS: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockIQ2XS);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockIQ2XS *blk = (BnBlockIQ2XS *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            memset(blk->qs, 0, sizeof(blk->qs));
            memset(blk->scales, 0x11, sizeof(blk->scales));
        }
        return data;
    }

    /* ── IQ2_S: 2-bit codebook (1024 grid), 256 elems, 82 bytes ───── */
    case BN_GGUF_TENSOR_IQ2_S: {
        int n_blocks = (int)(nelements / 256);
        size_t block_size = sizeof(BnBlockIQ2S);
        uint8_t *data = calloc((size_t)n_blocks, block_size);
        if (!data) return NULL;
        for (int b = 0; b < n_blocks; b++) {
            BnBlockIQ2S *blk = (BnBlockIQ2S *)(data + (size_t)b * block_size);
            blk->d = bn_fp32_to_fp16(1.0f);
            memset(blk->qs, 0, sizeof(blk->qs));
            memset(blk->qh, 0, sizeof(blk->qh));
            memset(blk->scales, 0x11, sizeof(blk->scales));
        }
        return data;
    }

    default:
        return NULL;
    }
}

/* ── CPU reference matvec ─────────────────────────────────────────── */

static void cpu_matvec(float *out, const BnQWeight *W, const float *x) {
    int max_dim = W->cols > W->rows ? W->cols : W->rows;
    int8_t *scratch = calloc((size_t)max_dim, 1);
    if (!scratch) return;
    bn_quant_matvec(out, W, x, scratch, NULL);
    free(scratch);
}

/* ── Per-type validation ──────────────────────────────────────────── */

/* Returns: 1 = pass, 0 = skip, -1 = fail */
static int validate_type(BnGPUBackend *gpu, const TypeInfo *info) {
    int rows = ROWS;
    int cols = info->cols;

    /* F16 and F32 have no CPU matvec kernel in dispatch.c */
    if (info->type == BN_GGUF_TENSOR_F16 ||
        info->type == BN_GGUF_TENSOR_F32) {
        printf("SKIP (no CPU matvec kernel)\n");
        return 0;
    }

    /* Generate weight data */
    float tensor_scale = 1.0f;
    void *data = make_weight_data(info->type, rows, cols, &tensor_scale);
    if (!data) {
        printf("SKIP (no data generator)\n");
        return 0;
    }

    /* Build BnQWeight descriptor */
    BnQWeight W = {0};
    W.data = data;
    W.type = info->type;
    W.rows = rows;
    W.cols = cols;
    W.scale = tensor_scale;

    /* Compute data size and upload to GPU */
    size_t sz = bn_qweight_data_size(&W);
    if (sz == 0) {
        printf("SKIP (unknown data size)\n");
        free(data);
        return 0;
    }

    W.gpu_buf = gpu->buffer_create(gpu->ctx, data, sz, W.type, W.rows, W.cols);
    if (!W.gpu_buf) {
        printf("SKIP (buffer_create failed)\n");
        free(data);
        return 0;
    }

    /* Input vector: all 1.0 */
    float *x = calloc((size_t)cols, sizeof(float));
    if (!x) { gpu->buffer_destroy(gpu->ctx, W.gpu_buf); free(data); return 0; }
    for (int i = 0; i < cols; i++)
        x[i] = 1.0f;

    /* CPU reference (scalar, no thread pool) */
    float out_cpu[ROWS];
    memset(out_cpu, 0, sizeof(out_cpu));
    cpu_matvec(out_cpu, &W, x);

    /* GPU dispatch */
    float out_gpu[ROWS];
    memset(out_gpu, 0, sizeof(out_gpu));
    int rc = gpu->matvec(gpu->ctx, out_gpu, W.gpu_buf, x, rows, cols, W.type);

    int result;
    if (rc != 0) {
        printf("SKIP (GPU dispatch error %d)\n", rc);
        result = 0;
    } else {
        /* Compare GPU vs CPU */
        float max_diff = 0.0f;
        int pass = 1;
        for (int i = 0; i < rows; i++) {
            float diff = fabsf(out_gpu[i] - out_cpu[i]);
            if (diff > max_diff) max_diff = diff;
            float denom = fabsf(out_cpu[i]);
            float rel = (denom > 1e-6f) ? diff / denom : diff;
            if (rel > TOL && diff > TOL) {
                pass = 0;
            }
        }
        if (pass) {
            printf("PASS (max_diff=%.6f, cpu[0]=%.4f gpu[0]=%.4f)\n",
                   max_diff, out_cpu[0], out_gpu[0]);
            result = 1;
        } else {
            printf("FAIL (max_diff=%.6f)\n", max_diff);
            for (int i = 0; i < rows; i++) {
                printf("  [%d] cpu=%.6f gpu=%.6f diff=%.6f\n",
                       i, out_cpu[i], out_gpu[i],
                       fabsf(out_gpu[i] - out_cpu[i]));
            }
            result = -1;
        }
    }

    /* Cleanup */
    gpu->buffer_destroy(gpu->ctx, W.gpu_buf);
    free(x);
    free(data);
    return result;
}

/* ── Matmul validation (batch matvec) ─────────────────────────────── */

/* Validate matmul for a single type. Returns 1=pass, 0=skip, -1=fail */
static int validate_matmul(BnGPUBackend *gpu, const TypeInfo *info, int n_tokens) {
    int rows = ROWS;
    int cols = info->cols;

    if (info->type == BN_GGUF_TENSOR_F16 ||
        info->type == BN_GGUF_TENSOR_F32) {
        printf("SKIP (no CPU kernel)\n");
        return 0;
    }

    float tensor_scale = 1.0f;
    void *data = make_weight_data(info->type, rows, cols, &tensor_scale);
    if (!data) { printf("SKIP\n"); return 0; }

    BnQWeight W = {0};
    W.data = data;
    W.type = info->type;
    W.rows = rows;
    W.cols = cols;
    W.scale = tensor_scale;

    size_t sz = bn_qweight_data_size(&W);
    W.gpu_buf = gpu->buffer_create(gpu->ctx, data, sz, W.type, W.rows, W.cols);
    if (!W.gpu_buf) { printf("SKIP (buffer)\n"); free(data); return 0; }

    /* X: n_tokens x cols, varied input per token */
    float *X = calloc((size_t)n_tokens * cols, sizeof(float));
    if (!X) { gpu->buffer_destroy(gpu->ctx, W.gpu_buf); free(data); return 0; }
    for (int t = 0; t < n_tokens; t++)
        for (int j = 0; j < cols; j++)
            X[t * cols + j] = (j % 2 == 0) ? 1.0f : 2.0f;

    /* GPU matmul */
    float *out_gpu = calloc((size_t)n_tokens * rows, sizeof(float));
    if (!out_gpu) {
        gpu->buffer_destroy(gpu->ctx, W.gpu_buf);
        free(X); free(data);
        return 0;
    }
    int rc = gpu->matmul(gpu->ctx, out_gpu, W.gpu_buf, X,
                          rows, cols, n_tokens, W.type);
    if (rc != 0) {
        printf("SKIP (matmul dispatch error %d)\n", rc);
        gpu->buffer_destroy(gpu->ctx, W.gpu_buf);
        free(out_gpu); free(X); free(data);
        return 0;
    }

    /* CPU reference: per-token matvec */
    float *out_cpu = calloc((size_t)n_tokens * rows, sizeof(float));
    if (!out_cpu) {
        gpu->buffer_destroy(gpu->ctx, W.gpu_buf);
        free(out_gpu); free(X); free(data);
        return 0;
    }
    for (int t = 0; t < n_tokens; t++)
        cpu_matvec(out_cpu + t * rows, &W, X + t * cols);

    /* Compare */
    float max_diff = 0.0f;
    int pass = 1;
    int total = n_tokens * rows;
    for (int i = 0; i < total; i++) {
        float diff = fabsf(out_gpu[i] - out_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        float denom = fabsf(out_cpu[i]);
        float rel = (denom > 1e-6f) ? diff / denom : diff;
        if (rel > TOL && diff > TOL)
            pass = 0;
    }

    int result;
    if (pass) {
        printf("PASS (n_tokens=%d, max_diff=%.6f)\n", n_tokens, max_diff);
        result = 1;
    } else {
        printf("FAIL (n_tokens=%d, max_diff=%.6f)\n", n_tokens, max_diff);
        for (int t = 0; t < n_tokens; t++)
            for (int i = 0; i < rows; i++)
                printf("  [t=%d][%d] cpu=%.6f gpu=%.6f\n",
                       t, i, out_cpu[t * rows + i], out_gpu[t * rows + i]);
        result = -1;
    }

    gpu->buffer_destroy(gpu->ctx, W.gpu_buf);
    free(out_gpu); free(out_cpu); free(X); free(data);
    return result;
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== GPU Validation Benchmark: %d quant types ===\n\n", N_TYPES);

    BnGPUBackend *gpu = bn_gpu_wgpu_create(SHADER_DIR);
    if (!gpu) {
        printf("No GPU available, skipping all tests\n");
        return 0;
    }

    /* ── Phase 1: matvec validation for all types ─────────────────── */
    printf("--- Phase 1: matvec (rows=%d) ---\n", ROWS);
    int passed = 0, failed = 0, skipped = 0;

    for (int i = 0; i < N_TYPES; i++) {
        printf("  %-10s ", ALL_TYPES[i].name);
        int r = validate_type(gpu, &ALL_TYPES[i]);
        if (r == 1) passed++;
        else if (r == 0) skipped++;
        else failed++;
    }

    printf("\n  Matvec: %d passed, %d failed, %d skipped\n\n", passed, failed, skipped);

    /* ── Phase 2: matmul validation for representative types ──────── */
    printf("--- Phase 2: matmul (n_tokens=4) ---\n");
    int mm_passed = 0, mm_failed = 0, mm_skipped = 0;

    /* Test matmul with 3 representative types: I2_S, Q4_0, Q4_K */
    const TypeInfo *matmul_types[] = {
        &ALL_TYPES[0],   /* I2_S */
        &ALL_TYPES[3],   /* Q4_0 */
        &ALL_TYPES[11],  /* Q4_K */
    };
    int n_mm = (int)(sizeof(matmul_types) / sizeof(matmul_types[0]));

    for (int i = 0; i < n_mm; i++) {
        printf("  %-10s ", matmul_types[i]->name);
        int r = validate_matmul(gpu, matmul_types[i], 4);
        if (r == 1) mm_passed++;
        else if (r == 0) mm_skipped++;
        else mm_failed++;
    }

    printf("\n  Matmul: %d passed, %d failed, %d skipped\n\n",
           mm_passed, mm_failed, mm_skipped);

    /* ── Summary ──────────────────────────────────────────────────── */
    int total_failed = failed + mm_failed;
    printf("=== GPU Validation: %d matvec passed, %d matmul passed, %d total failed ===\n",
           passed, mm_passed, total_failed);

    bn_gpu_wgpu_destroy(gpu);
    return total_failed > 0 ? 1 : 0;
}

#else /* !BN_ENABLE_GPU */

#include <stdio.h>

int main(void) {
    printf("GPU validation skipped (BN_ENABLE_GPU not set)\n");
    return 0;
}

#endif /* BN_ENABLE_GPU */
