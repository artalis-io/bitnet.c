/*
 * test_gpu_wgpu.c — wgpu-native GPU backend integration tests
 *
 * Tests real GPU dispatch using wgpu-native. Validates shader correctness
 * by comparing GPU matvec results against CPU reference (bn_quant_matvec).
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
#include <assert.h>

/* Tolerance for GPU vs CPU comparison */
#define TOL 1e-3f

/* Path to WGSL shaders (relative to project root where tests run) */
static const char *SHADER_DIR = "shaders";

/* ── Helpers ───────────────────────────────────────────────────────── */

/* Create I2_S weight data: 4 values per byte, all +1 (encoding=2).
 * Interleaved byte layout. Per-tensor scale at end as float. */
static uint8_t *make_i2s_data(int rows, int cols, float scale, size_t *out_size)
{
    size_t nelements = (size_t)rows * cols;
    size_t data_size = nelements / 4 + 4;
    uint8_t *data = calloc(1, data_size);
    assert(data);
    /* Encode all +1: 2-bit value = 2, byte = 0xAA (10 10 10 10) */
    for (size_t i = 0; i < nelements / 4; i++)
        data[i] = 0xAA;
    /* Per-tensor scale as float at end */
    memcpy(data + nelements / 4, &scale, sizeof(float));
    if (out_size) *out_size = data_size;
    return data;
}

/* Create Q4_0 weight data: 32 elements per block, 18 bytes per block.
 * All nibbles set to `nib`, FP16 scale set to `scale`. */
static uint8_t *make_q4_data(int rows, int cols, uint8_t nib, float scale,
                              size_t *out_size)
{
    int n_blocks = (rows * cols) / 32;
    size_t block_size = sizeof(BnBlockQ4_0);
    size_t data_size = (size_t)n_blocks * block_size;
    uint8_t *data = calloc(1, data_size);
    assert(data);

    uint16_t fp16_scale = bn_fp32_to_fp16(scale);
    uint8_t packed = (nib & 0xF) | ((nib & 0xF) << 4);

    for (int b = 0; b < n_blocks; b++) {
        BnBlockQ4_0 *blk = (BnBlockQ4_0 *)(data + b * block_size);
        blk->d = fp16_scale;
        memset(blk->qs, packed, 16);
    }
    if (out_size) *out_size = data_size;
    return data;
}

/* Create Q4_K weight data: 256 elements per block, 144 bytes per block.
 * Simple case: all nibbles = nib, scales/mins = 0 except first group. */
static uint8_t *make_q4k_data(int rows, int cols, float scale,
                               size_t *out_size)
{
    int n_blocks = (rows * cols) / 256;
    size_t block_size = sizeof(BnBlockQ4K);
    size_t data_size = (size_t)n_blocks * block_size;
    uint8_t *data = calloc(1, data_size);
    assert(data);

    uint16_t fp16_d = bn_fp32_to_fp16(scale);
    /* Simple: set d, dmin=0, all scales to small values, nibbles to 8 (zero-point) */
    for (int b = 0; b < n_blocks; b++) {
        BnBlockQ4K *blk = (BnBlockQ4K *)(data + b * block_size);
        blk->d = fp16_d;
        blk->dmin = 0;
        /* scales[0..11] encode 8 scale/min pairs. Set all scale=1, min=0 */
        memset(blk->scales, 0x01, 12);
        /* All nibbles = 8 -> value = (8 * sc - min) * d */
        memset(blk->qs, 0x88, 128);
    }
    if (out_size) *out_size = data_size;
    return data;
}

/* Compute CPU reference matvec */
static void cpu_matvec(float *out, const BnQWeight *W, const float *x)
{
    int max_dim = W->cols > W->rows ? W->cols : W->rows;
    int8_t *scratch = calloc((size_t)max_dim, 1);
    assert(scratch);
    bn_quant_matvec(out, W, x, scratch, NULL);
    free(scratch);
}

/* Compare two float arrays within tolerance */
static int compare_floats(const float *a, const float *b, int n, float tol)
{
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tol) {
            fprintf(stderr, "  mismatch at [%d]: gpu=%.6f cpu=%.6f diff=%.6f\n",
                    i, a[i], b[i], diff);
            return 0;
        }
    }
    return 1;
}

/* ── Test 1: create and destroy ────────────────────────────────────── */

static void test_wgpu_create_destroy(void)
{
    printf("test_wgpu_create_destroy... ");

    BnGPUBackend *gpu = bn_gpu_wgpu_create(SHADER_DIR);
    if (!gpu) {
        printf("SKIPPED (no GPU or no shaders)\n");
        return;
    }
    assert(gpu->ctx != NULL);
    assert(gpu->buffer_create != NULL);
    assert(gpu->buffer_destroy != NULL);
    assert(gpu->matvec != NULL);
    assert(gpu->matmul != NULL);

    bn_gpu_wgpu_destroy(gpu);
    printf("PASSED\n");
}

/* ── Test 2: I2_S matvec ──────────────────────────────────────────── */

static void test_wgpu_i2s_matvec(void)
{
    printf("test_wgpu_i2s_matvec... ");

    BnGPUBackend *gpu = bn_gpu_wgpu_create(SHADER_DIR);
    if (!gpu) { printf("SKIPPED\n"); return; }

    int rows = 4, cols = 128;
    float scale = 1.0f;
    size_t data_size;
    uint8_t *data = make_i2s_data(rows, cols, scale, &data_size);

    BnQWeight W = {
        .data = data, .type = BN_GGUF_TENSOR_I2_S,
        .rows = rows, .cols = cols, .scale = scale,
    };

    /* Upload to GPU */
    void *gpu_buf = gpu->buffer_create(gpu->ctx, data, data_size,
                                        W.type, W.rows, W.cols);
    if (!gpu_buf) {
        printf("SKIPPED (buffer_create failed)\n");
        free(data); bn_gpu_wgpu_destroy(gpu); return;
    }

    /* Input: all 1.0 */
    float *x = calloc((size_t)cols, sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = 1.0f;

    /* GPU matvec */
    float *out_gpu = calloc((size_t)rows, sizeof(float));
    int rc = gpu->matvec(gpu->ctx, out_gpu, gpu_buf, x, rows, cols, W.type);
    if (rc != 0) {
        printf("SKIPPED (matvec dispatch failed, no shader?)\n");
        gpu->buffer_destroy(gpu->ctx, gpu_buf);
        free(out_gpu); free(x); free(data);
        bn_gpu_wgpu_destroy(gpu); return;
    }

    /* CPU reference */
    float *out_cpu = calloc((size_t)rows, sizeof(float));
    cpu_matvec(out_cpu, &W, x);

    assert(compare_floats(out_gpu, out_cpu, rows, TOL));

    gpu->buffer_destroy(gpu->ctx, gpu_buf);
    free(out_gpu); free(out_cpu); free(x); free(data);
    bn_gpu_wgpu_destroy(gpu);
    printf("PASSED\n");
}

/* ── Test 3: Q4_0 matvec ─────────────────────────────────────────── */

static void test_wgpu_q4_matvec(void)
{
    printf("test_wgpu_q4_matvec... ");

    BnGPUBackend *gpu = bn_gpu_wgpu_create(SHADER_DIR);
    if (!gpu) { printf("SKIPPED\n"); return; }

    int rows = 4, cols = 256;
    float scale = 0.5f;
    /* nibble=12 -> value = (12 - 8) * scale = 4 * 0.5 = 2.0 */
    size_t data_size;
    uint8_t *data = make_q4_data(rows, cols, 12, scale, &data_size);

    BnQWeight W = {
        .data = data, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = rows, .cols = cols, .scale = 1.0f,
    };

    void *gpu_buf = gpu->buffer_create(gpu->ctx, data, data_size,
                                        W.type, W.rows, W.cols);
    if (!gpu_buf) {
        printf("SKIPPED (buffer_create failed)\n");
        free(data); bn_gpu_wgpu_destroy(gpu); return;
    }

    float *x = calloc((size_t)cols, sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = 1.0f;

    float *out_gpu = calloc((size_t)rows, sizeof(float));
    int rc = gpu->matvec(gpu->ctx, out_gpu, gpu_buf, x, rows, cols, W.type);
    if (rc != 0) {
        printf("SKIPPED (no Q4_0 shader)\n");
        gpu->buffer_destroy(gpu->ctx, gpu_buf);
        free(out_gpu); free(x); free(data);
        bn_gpu_wgpu_destroy(gpu); return;
    }

    float *out_cpu = calloc((size_t)rows, sizeof(float));
    cpu_matvec(out_cpu, &W, x);

    assert(compare_floats(out_gpu, out_cpu, rows, TOL));

    gpu->buffer_destroy(gpu->ctx, gpu_buf);
    free(out_gpu); free(out_cpu); free(x); free(data);
    bn_gpu_wgpu_destroy(gpu);
    printf("PASSED\n");
}

/* ── Test 4: Q4_K matvec ─────────────────────────────────────────── */

static void test_wgpu_q4k_matvec(void)
{
    printf("test_wgpu_q4k_matvec... ");

    BnGPUBackend *gpu = bn_gpu_wgpu_create(SHADER_DIR);
    if (!gpu) { printf("SKIPPED\n"); return; }

    int rows = 2, cols = 256;
    float scale = 1.0f;
    size_t data_size;
    uint8_t *data = make_q4k_data(rows, cols, scale, &data_size);

    BnQWeight W = {
        .data = data, .type = BN_GGUF_TENSOR_Q4_K,
        .rows = rows, .cols = cols, .scale = 1.0f,
    };

    void *gpu_buf = gpu->buffer_create(gpu->ctx, data, data_size,
                                        W.type, W.rows, W.cols);
    if (!gpu_buf) {
        printf("SKIPPED (buffer_create failed)\n");
        free(data); bn_gpu_wgpu_destroy(gpu); return;
    }

    float *x = calloc((size_t)cols, sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = 1.0f;

    float *out_gpu = calloc((size_t)rows, sizeof(float));
    int rc = gpu->matvec(gpu->ctx, out_gpu, gpu_buf, x, rows, cols, W.type);
    if (rc != 0) {
        printf("SKIPPED (no Q4_K shader)\n");
        gpu->buffer_destroy(gpu->ctx, gpu_buf);
        free(out_gpu); free(x); free(data);
        bn_gpu_wgpu_destroy(gpu); return;
    }

    float *out_cpu = calloc((size_t)rows, sizeof(float));
    cpu_matvec(out_cpu, &W, x);

    assert(compare_floats(out_gpu, out_cpu, rows, TOL));

    gpu->buffer_destroy(gpu->ctx, gpu_buf);
    free(out_gpu); free(out_cpu); free(x); free(data);
    bn_gpu_wgpu_destroy(gpu);
    printf("PASSED\n");
}

/* ── Test 5: matmul (batch matvec) ─────────────────────────────────── */

static void test_wgpu_matmul(void)
{
    printf("test_wgpu_matmul... ");

    BnGPUBackend *gpu = bn_gpu_wgpu_create(SHADER_DIR);
    if (!gpu) { printf("SKIPPED\n"); return; }

    int rows = 4, cols = 128, n_tokens = 4;
    float scale = 1.0f;
    size_t data_size;
    uint8_t *data = make_i2s_data(rows, cols, scale, &data_size);

    BnQWeight W = {
        .data = data, .type = BN_GGUF_TENSOR_I2_S,
        .rows = rows, .cols = cols, .scale = scale,
    };

    void *gpu_buf = gpu->buffer_create(gpu->ctx, data, data_size,
                                        W.type, W.rows, W.cols);
    if (!gpu_buf) {
        printf("SKIPPED (buffer_create failed)\n");
        free(data); bn_gpu_wgpu_destroy(gpu); return;
    }

    /* X: n_tokens x cols, each row = [1.0, 2.0, 1.0, 2.0, ...] */
    float *X = calloc((size_t)n_tokens * cols, sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int j = 0; j < cols; j++)
            X[t * cols + j] = (j % 2 == 0) ? 1.0f : 2.0f;

    /* GPU matmul */
    float *out_gpu = calloc((size_t)n_tokens * rows, sizeof(float));
    int rc = gpu->matmul(gpu->ctx, out_gpu, gpu_buf, X,
                          rows, cols, n_tokens, W.type);
    if (rc != 0) {
        printf("SKIPPED (matmul dispatch failed)\n");
        gpu->buffer_destroy(gpu->ctx, gpu_buf);
        free(out_gpu); free(X); free(data);
        bn_gpu_wgpu_destroy(gpu); return;
    }

    /* CPU reference: repeated single matvec per token */
    float *out_cpu = calloc((size_t)n_tokens * rows, sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        cpu_matvec(out_cpu + t * rows, &W, X + t * cols);

    assert(compare_floats(out_gpu, out_cpu, n_tokens * rows, TOL));

    gpu->buffer_destroy(gpu->ctx, gpu_buf);
    free(out_gpu); free(out_cpu); free(X); free(data);
    bn_gpu_wgpu_destroy(gpu);
    printf("PASSED\n");
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    test_wgpu_create_destroy();
    test_wgpu_i2s_matvec();
    test_wgpu_q4_matvec();
    test_wgpu_q4k_matvec();
    test_wgpu_matmul();
    printf("All GPU wgpu tests completed\n");
    return 0;
}

#else /* !BN_ENABLE_GPU */

#include <stdio.h>

int main(void)
{
    printf("GPU tests skipped (BN_ENABLE_GPU not set)\n");
    return 0;
}

#endif /* BN_ENABLE_GPU */
