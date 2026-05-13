// bench_kernels.c — per-kernel matvec benchmark for bitnet.c
// Compilable for native (NEON/AVX2) and WASM (emcc + Node.js).
// Usage: ./bench model.gguf [--iters N] [--threads T]

#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "session.h"
#include "quant.h"
#include "transformer.h"
#include "sampler.h"
#include "threadpool.h"
#ifdef BN_ENABLE_WEBGPU
#include "gpu_wgpu.h"
#endif
#if defined(__wasm_relaxed_simd__)
#include <wasm_simd128.h>
#include "simd_helpers.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *type_name(int type) {
    switch (type) {
        case BN_GGUF_TENSOR_F32:     return "F32";
        case BN_GGUF_TENSOR_F16:     return "F16";
        case BN_GGUF_TENSOR_Q4_0:    return "Q4_0";
        case BN_GGUF_TENSOR_Q4_1:    return "Q4_1";
        case BN_GGUF_TENSOR_Q8_0:    return "Q8_0";
        case BN_GGUF_TENSOR_Q2_K:    return "Q2_K";
        case BN_GGUF_TENSOR_Q3_K:    return "Q3_K";
        case BN_GGUF_TENSOR_Q4_K:    return "Q4_K";
        case BN_GGUF_TENSOR_Q5_K:    return "Q5_K";
        case BN_GGUF_TENSOR_Q6_K:    return "Q6_K";
        case BN_GGUF_TENSOR_Q8_K:    return "Q8_K";
        case BN_GGUF_TENSOR_IQ2_XXS: return "IQ2_XXS";
        case BN_GGUF_TENSOR_IQ2_XS:  return "IQ2_XS";
        case BN_GGUF_TENSOR_IQ2_S:   return "IQ2_S";
        case BN_GGUF_TENSOR_IQ3_XXS: return "IQ3_XXS";
        case BN_GGUF_TENSOR_IQ3_S:   return "IQ3_S";
        case BN_GGUF_TENSOR_IQ4_NL:  return "IQ4_NL";
        case BN_GGUF_TENSOR_IQ4_XS:  return "IQ4_XS";
        case BN_GGUF_TENSOR_BF16:    return "BF16";
        case BN_GGUF_TENSOR_TQ1_0:   return "TQ1_0";
        case BN_GGUF_TENSOR_TQ2_0:   return "TQ2_0";
        case BN_GGUF_TENSOR_I2_S:    return "I2_S";
        default:                     return "???";
    }
}

static const char *backend_name(int webgpu) {
    if (webgpu) return "WebGPU";
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return "ARM NEON + SDOT";
#elif defined(__ARM_NEON)
    return "ARM NEON";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__wasm_simd128__)
  #ifdef __wasm_relaxed_simd__
    return "WASM Relaxed SIMD";
  #else
    return "WASM SIMD128";
  #endif
#else
    return "Scalar";
#endif
}

// Simple LCG for deterministic random fill
static uint32_t bench_rng_state = 42;
static volatile float bench_sink = 0.0f;
static int bench_q4_expand_enabled = 0;

static float bench_randf(void) {
    bench_rng_state = bench_rng_state * 1664525u + 1013904223u;
    return (float)(bench_rng_state >> 16) / 65536.0f - 0.5f;
}

typedef struct {
    const char *name;
    const BnQWeight *W;
} BenchTarget;

#if defined(__wasm_relaxed_simd__)
typedef struct {
    int rows;
    int cols;
    int n_blocks_per_row;
    int8_t *qs;
    uint16_t *scales;
} BenchQ4Expanded;

static void bench_q4_expanded_free(BenchQ4Expanded *e) {
    if (!e) return;
    free(e->qs);
    free(e->scales);
    memset(e, 0, sizeof(*e));
}

static int bench_q4_expanded_build(BenchQ4Expanded *e, const BnQWeight *W) {
    memset(e, 0, sizeof(*e));
    if (!W || W->type != BN_GGUF_TENSOR_Q4_0 || W->cols % 32 != 0)
        return -1;

    int n_blocks_per_row = W->cols / 32;
    size_t n_blocks = (size_t)W->rows * n_blocks_per_row;
    e->qs = (int8_t *)malloc((size_t)W->rows * W->cols);
    e->scales = (uint16_t *)malloc(n_blocks * sizeof(uint16_t));
    if (!e->qs || !e->scales) {
        bench_q4_expanded_free(e);
        return -1;
    }

    e->rows = W->rows;
    e->cols = W->cols;
    e->n_blocks_per_row = n_blocks_per_row;

    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)W->data;
    for (int row = 0; row < W->rows; row++) {
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            int8_t *dst = e->qs + (size_t)row * W->cols + b * 32;
            e->scales[(size_t)row * n_blocks_per_row + b] = blk->d;
            for (int i = 0; i < 16; i++) {
                uint8_t raw = blk->qs[i];
                dst[i] = (int8_t)((raw & 0x0F) - 8);
                dst[i + 16] = (int8_t)((raw >> 4) - 8);
            }
        }
    }
    return 0;
}

static void bench_q4_expanded_matvec(float *out, const BenchQ4Expanded *e,
                                     const int8_t *x_q, const float *x_scales) {
    int row = 0;
    for (; row + 3 < e->rows; row += 4) {
        v128_t accf0 = wasm_f32x4_splat(0.0f);
        v128_t accf1 = wasm_f32x4_splat(0.0f);
        v128_t accf2 = wasm_f32x4_splat(0.0f);
        v128_t accf3 = wasm_f32x4_splat(0.0f);
        const int8_t *row_qs0 = e->qs + (size_t)row * e->cols;
        const int8_t *row_qs1 = row_qs0 + e->cols;
        const int8_t *row_qs2 = row_qs1 + e->cols;
        const int8_t *row_qs3 = row_qs2 + e->cols;
        const uint16_t *row_scales0 = e->scales + (size_t)row * e->n_blocks_per_row;
        const uint16_t *row_scales1 = row_scales0 + e->n_blocks_per_row;
        const uint16_t *row_scales2 = row_scales1 + e->n_blocks_per_row;
        const uint16_t *row_scales3 = row_scales2 + e->n_blocks_per_row;

        for (int b = 0; b < e->n_blocks_per_row; b++) {
            const int8_t *x = x_q + b * 32;
            v128_t x0 = wasm_v128_load(x);
            v128_t x1 = wasm_v128_load(x + 16);
            float dx = x_scales[b];

#define BENCH_Q4_EXP_ACC(accf_, row_qs_, row_scales_) do { \
                const int8_t *w_ = (row_qs_) + b * 32; \
                v128_t acc_ = wasm_i32x4_relaxed_dot_i8x16_i7x16_add( \
                    wasm_v128_load(w_), x0, wasm_i32x4_splat(0)); \
                acc_ = wasm_i32x4_relaxed_dot_i8x16_i7x16_add( \
                    wasm_v128_load(w_ + 16), x1, acc_); \
                v128_t scale_ = wasm_f32x4_splat(bn_fp16_to_fp32((row_scales_)[b]) * dx); \
                accf_ = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(acc_), scale_, accf_); \
            } while (0)

            BENCH_Q4_EXP_ACC(accf0, row_qs0, row_scales0);
            BENCH_Q4_EXP_ACC(accf1, row_qs1, row_scales1);
            BENCH_Q4_EXP_ACC(accf2, row_qs2, row_scales2);
            BENCH_Q4_EXP_ACC(accf3, row_qs3, row_scales3);

#undef BENCH_Q4_EXP_ACC
        }
        out[row] = bn_wasm_hsum_f32x4(accf0);
        out[row + 1] = bn_wasm_hsum_f32x4(accf1);
        out[row + 2] = bn_wasm_hsum_f32x4(accf2);
        out[row + 3] = bn_wasm_hsum_f32x4(accf3);
    }

    for (; row < e->rows; row++) {
        v128_t accf = wasm_f32x4_splat(0.0f);
        const int8_t *row_qs = e->qs + (size_t)row * e->cols;
        const uint16_t *row_scales = e->scales + (size_t)row * e->n_blocks_per_row;
        for (int b = 0; b < e->n_blocks_per_row; b++) {
            const int8_t *w = row_qs + b * 32;
            const int8_t *x = x_q + b * 32;
            v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(w), wasm_v128_load(x), wasm_i32x4_splat(0));
            acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(w + 16), wasm_v128_load(x + 16), acc);
            v128_t scale = wasm_f32x4_splat(bn_fp16_to_fp32(row_scales[b]) * x_scales[b]);
            accf = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(acc), scale, accf);
        }
        out[row] = bn_wasm_hsum_f32x4(accf);
    }
}

static void bench_q4_expanded(const char *name, const BnQWeight *W,
                              const float *x, int8_t *x_q, int n_iters) {
    if (!bench_q4_expand_enabled || W->type != BN_GGUF_TENSOR_Q4_0)
        return;

    BenchQ4Expanded e;
    if (bench_q4_expanded_build(&e, W) != 0)
        return;

    float *out = (float *)calloc((size_t)W->rows, sizeof(float));
    float *x_scales = (float *)malloc((size_t)e.n_blocks_per_row * sizeof(float));
    if (!out || !x_scales) {
        free(out);
        free(x_scales);
        bench_q4_expanded_free(&e);
        return;
    }

    bn_quant_x_to_q8_blocks(x, x_q, x_scales, W->cols);

    for (int i = 0; i < 5; i++)
        bench_q4_expanded_matvec(out, &e, x_q, x_scales);

    double t0 = bn_platform_time_ms();
    for (int i = 0; i < n_iters; i++)
        bench_q4_expanded_matvec(out, &e, x_q, x_scales);
    double elapsed = bn_platform_time_ms() - t0;
    bench_sink += out[0] + out[W->rows / 2] + out[W->rows - 1];

    double us_per_call = (elapsed * 1000.0) / n_iters;
    size_t total_bytes = (size_t)W->rows * W->cols + (size_t)W->cols * sizeof(float);
    double gb_per_s = (total_bytes / 1e9) / (us_per_call / 1e6);
    char expanded_name[16];
    snprintf(expanded_name, sizeof(expanded_name), "%s_x8", name);
    printf("%-8s | %-7s | %5d x %-5d | %8.1f | %5.2f\n",
           expanded_name, "Q4i8", W->rows, W->cols, us_per_call, gb_per_s);

    free(out);
    free(x_scales);
    bench_q4_expanded_free(&e);
}
#else
static void bench_q4_expanded(const char *name, const BnQWeight *W,
                              const float *x, int8_t *x_q, int n_iters) {
    (void)name; (void)W; (void)x; (void)x_q; (void)n_iters;
}
#endif

static void bench_matvec(const char *name, const BnQWeight *W,
                          const float *x, int8_t *x_q, BnThreadPool *pool,
                          int n_iters) {
    float *out = calloc((size_t)W->rows, sizeof(float));
    if (!out) return;

    // Warmup
    for (int i = 0; i < 5; i++)
        bn_quant_matvec(out, W, x, x_q, pool);

    // Timed iterations
    double t0 = bn_platform_time_ms();
    for (int i = 0; i < n_iters; i++)
        bn_quant_matvec(out, W, x, x_q, pool);
    double elapsed = bn_platform_time_ms() - t0;
    bench_sink += out[0] + out[W->rows / 2] + out[W->rows - 1];

    double us_per_call = (elapsed * 1000.0) / n_iters;

    // Compute weight data size for bandwidth
    // For block-quantized formats, compute actual bytes from block sizes
    size_t weight_bytes = 0;
    int cols = W->cols, rows = W->rows;
    switch (W->type) {
        case BN_GGUF_TENSOR_I2_S:     weight_bytes = (size_t)rows * cols / 4; break;
        case BN_GGUF_TENSOR_TQ1_0:    weight_bytes = (size_t)rows * (cols / 256) * 54; break;
        case BN_GGUF_TENSOR_TQ2_0:    weight_bytes = (size_t)rows * (cols / 256) * 66; break;
        case BN_GGUF_TENSOR_Q4_0:     weight_bytes = (size_t)rows * (cols / 32) * 18; break;
        case BN_GGUF_TENSOR_Q4_1:     weight_bytes = (size_t)rows * (cols / 32) * 20; break;
        case BN_GGUF_TENSOR_Q8_0:     weight_bytes = (size_t)rows * (cols / 32) * 34; break;
        case BN_GGUF_TENSOR_Q2_K:     weight_bytes = (size_t)rows * (cols / 256) * 84; break;
        case BN_GGUF_TENSOR_Q3_K:     weight_bytes = (size_t)rows * (cols / 256) * 110; break;
        case BN_GGUF_TENSOR_Q4_K:     weight_bytes = (size_t)rows * (cols / 256) * 144; break;
        case BN_GGUF_TENSOR_Q5_K:     weight_bytes = (size_t)rows * (cols / 256) * 176; break;
        case BN_GGUF_TENSOR_Q6_K:     weight_bytes = (size_t)rows * (cols / 256) * 210; break;
        case BN_GGUF_TENSOR_Q8_K:     weight_bytes = (size_t)rows * (cols / 256) * 292; break;
        case BN_GGUF_TENSOR_BF16:     weight_bytes = (size_t)rows * cols * 2; break;
        case BN_GGUF_TENSOR_F16:      weight_bytes = (size_t)rows * cols * 2; break;
        case BN_GGUF_TENSOR_F32:      weight_bytes = (size_t)rows * cols * 4; break;
        case BN_GGUF_TENSOR_IQ4_NL:   weight_bytes = (size_t)rows * (cols / 32) * 18; break;
        case BN_GGUF_TENSOR_IQ4_XS:   weight_bytes = (size_t)rows * (cols / 256) * 136; break;
        case BN_GGUF_TENSOR_IQ3_XXS:  weight_bytes = (size_t)rows * (cols / 256) * 98; break;
        case BN_GGUF_TENSOR_IQ3_S:    weight_bytes = (size_t)rows * (cols / 256) * 114; break;
        case BN_GGUF_TENSOR_IQ2_XXS:  weight_bytes = (size_t)rows * (cols / 256) * 66; break;
        case BN_GGUF_TENSOR_IQ2_XS:   weight_bytes = (size_t)rows * (cols / 256) * 74; break;
        case BN_GGUF_TENSOR_IQ2_S:    weight_bytes = (size_t)rows * (cols / 256) * 82; break;
        default:                      weight_bytes = (size_t)rows * cols; break;
    }
    // Add activation vector read
    size_t total_bytes = weight_bytes + (size_t)cols * sizeof(float);
    double gb_per_s = (total_bytes / 1e9) / (us_per_call / 1e6);

    printf("%-8s | %-7s | %5d x %-5d | %8.1f | %5.2f\n",
           name, type_name(W->type), W->rows, W->cols, us_per_call, gb_per_s);

    bench_q4_expanded(name, W, x, x_q, n_iters);

    free(out);
}

static void bench_logits_f16(const BnModel *m, const float *x, int n_iters) {
    int vocab = m->config.vocab_size;
    int dim = m->config.dim;
    float *logits = calloc((size_t)vocab, sizeof(float));
    if (!logits) return;

    const uint16_t *emb = (const uint16_t *)m->weights.token_embedding;

    // Use the model's matvec through output_weight if it exists,
    // otherwise benchmark the logits embedding path directly.
    // For F16 embeddings, we do a manual dot product benchmark.

    // Warmup
    for (int i = 0; i < 5; i++) {
        for (int v = 0; v < vocab; v++) {
            const uint16_t *row = emb + (size_t)v * dim;
            float sum = 0.0f;
            for (int d = 0; d < dim; d++)
                sum += bn_fp16_to_fp32(row[d]) * x[d];
            logits[v] = sum;
        }
    }

    double t0 = bn_platform_time_ms();
    for (int iter = 0; iter < n_iters; iter++) {
        for (int v = 0; v < vocab; v++) {
            const uint16_t *row = emb + (size_t)v * dim;
            float sum = 0.0f;
            for (int d = 0; d < dim; d++)
                sum += bn_fp16_to_fp32(row[d]) * x[d];
            logits[v] = sum;
        }
    }
    double elapsed = bn_platform_time_ms() - t0;
    bench_sink += logits[0] + logits[vocab / 2] + logits[vocab - 1];
    double us_per_call = (elapsed * 1000.0) / n_iters;
    size_t total_bytes = (size_t)vocab * dim * 2 + (size_t)dim * sizeof(float);
    double gb_per_s = (total_bytes / 1e9) / (us_per_call / 1e6);

    printf("%-8s | %-7s | %5d x %-5d | %8.1f | %5.2f\n",
           "logits", "F16", vocab, dim, us_per_call, gb_per_s);

    free(logits);
}

static void bench_logits_real(const BnModel *m, int n_iters, BnThreadPool *pool) {
    // Use the actual transformer logits path via a forward pass on token 0
    int dim = m->config.dim;
    int vocab = m->config.vocab_size;

    // Build a fake x vector
    float *x = calloc((size_t)dim, sizeof(float));
    float *logits = calloc((size_t)vocab, sizeof(float));
    int8_t *x_q = calloc((size_t)dim, sizeof(int8_t));
    if (!x || !logits || !x_q) { free(x); free(logits); free(x_q); return; }

    for (int i = 0; i < dim; i++) x[i] = bench_randf();

    // If model has output_weight (untied embeddings), benchmark that
    if (m->weights.output_weight.data) {
        const BnQWeight *W = &m->weights.output_weight;

        // Warmup
        for (int i = 0; i < 5; i++)
            bn_quant_matvec(logits, W, x, x_q, pool);

        double t0 = bn_platform_time_ms();
        for (int i = 0; i < n_iters; i++)
            bn_quant_matvec(logits, W, x, x_q, pool);
        double elapsed = bn_platform_time_ms() - t0;
        bench_sink += logits[0] + logits[vocab / 2] + logits[vocab - 1];
        double us_per_call = (elapsed * 1000.0) / n_iters;

        // Rough bandwidth
        size_t total_bytes = (size_t)vocab * dim / 2; // approximate
        double gb_per_s = (total_bytes / 1e9) / (us_per_call / 1e6);

        printf("%-8s | %-7s | %5d x %-5d | %8.1f | %5.2f\n",
               "logits", type_name(W->type), W->rows, W->cols, us_per_call, gb_per_s);
    }

    free(x);
    free(logits);
    free(x_q);
}

static void bench_toks(BnModel *m, BnSession *s, int n_gen) {
    // Generate tokens and measure throughput
    int warmup = 4;
    int total = warmup + n_gen;

    BnSampler sampler;
    bn_sampler_init(&sampler, m->config.vocab_size, 0.0f, 0.9f, 42);

    // Feed BOS token at pos 0
    float *logits = bn_transformer_forward(m, s, 1, 0);  // token 1 = BOS for most models
    if (!logits) {
        fprintf(stderr, "Forward pass failed\n");
        bn_sampler_free(&sampler);
        return;
    }
    int next = bn_sampler_sample(&sampler, logits);

    // Warmup + timed generation
    double t_start = 0;
    for (int i = 0; i < total; i++) {
        if (i == warmup) t_start = bn_platform_time_ms();
        logits = bn_transformer_forward(m, s, next, i + 1);
        if (!logits) break;
        next = bn_sampler_sample(&sampler, logits);
    }
    double elapsed = bn_platform_time_ms() - t_start;
    double toks_per_sec = (double)n_gen / (elapsed / 1000.0);

    printf("\nThroughput: %.1f tok/s  (%d tokens in %.0f ms, %d warmup)\n",
           toks_per_sec, n_gen, elapsed, warmup);

    bn_sampler_free(&sampler);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [--iters N] [--threads T] [--toks N] [--kv16] [--q4-expand] [--webgpu] [--shader-dir DIR]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int n_iters = 100;
    int n_threads = 1;
    int n_toks = 32;
    int kv_f16 = 0;
    int use_webgpu = 0;
#ifdef BN_ENABLE_WEBGPU
    const char *shader_dir = "shaders/";
#endif

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            n_iters = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            n_threads = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--toks") == 0 && i + 1 < argc)
            n_toks = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--kv16") == 0)
            kv_f16 = 1;
        else if (strcmp(argv[i], "--q4-expand") == 0)
            bench_q4_expand_enabled = 1;
        else if (strcmp(argv[i], "--webgpu") == 0)
            use_webgpu = 1;
        else if (strcmp(argv[i], "--shader-dir") == 0 && i + 1 < argc) {
#ifdef BN_ENABLE_WEBGPU
            shader_dir = argv[++i];
#else
            i++;
#endif
        }
    }

    // Load model
    BnMappedFile mf = bn_platform_load_file(model_path);
    if (!mf.data) {
        fprintf(stderr, "Failed to load %s\n", model_path);
        return 1;
    }

    BnGGUFFile *gf = bn_gguf_open(mf.data, mf.size);
    if (!gf) {
        fprintf(stderr, "Failed to parse GGUF\n");
        bn_platform_unload_file(&mf);
        return 1;
    }

    BnModel model = {0};
    int bench_seq_len = n_toks + 8;
    if (bench_seq_len < 32) bench_seq_len = 32;
    if (bn_model_load(&model, gf, bench_seq_len, kv_f16, 0) != 0) {
        fprintf(stderr, "Failed to load model\n");
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    bn_model_set_file(&model, mf);
    if (model.config.n_experts > 0) {
        if ((mf.is_mmap == 1 || mf.is_mmap == 0) && mf.data)
            bn_model_set_moe_mmap_base(&model, mf.data);
        if (mf.fd >= 0)
            bn_model_set_moe_fd(&model, mf.fd);
    }

#ifdef BN_ENABLE_WEBGPU
    BnGPUBackend *gpu = NULL;
    if (use_webgpu) {
        gpu = bn_gpu_wgpu_create(shader_dir);
        if (!gpu) {
            fprintf(stderr, "Failed to create WebGPU backend\n");
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }
        if (bn_model_upload_weights(&model, gpu) != 0) {
            fprintf(stderr, "Failed to upload model weights to WebGPU\n");
            bn_gpu_wgpu_destroy(gpu);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }
        if (gpu->init_activations &&
            gpu->init_activations(gpu->ctx, &model.config) != 0) {
            fprintf(stderr, "Failed to initialize WebGPU activations\n");
            bn_model_free(&model);
            bn_gpu_wgpu_destroy(gpu);
            bn_gguf_free(gf);
            return 1;
        }
    }
#else
    if (use_webgpu) {
        fprintf(stderr, "--webgpu requires BN_ENABLE_WEBGPU=1 build\n");
        bn_model_free(&model);
        bn_gguf_free(gf);
        return 1;
    }
#endif

    // Extract model name from path
    const char *fname = strrchr(model_path, '/');
    fname = fname ? fname + 1 : model_path;

    // Create thread pool if requested
    BnThreadPool *pool = NULL;
    if (n_threads > 1)
        pool = bn_tp_create(n_threads - 1);

    // Allocate input buffers large enough for the widest loaded projection.
    // Some hybrid/Gemma-family layers have projection input widths that differ
    // from both dim and hidden_dim.
    int dim = model.config.dim;
    int hidden_dim = model.config.hidden_dim;
    int buf_size = dim > hidden_dim ? dim : hidden_dim;
    if (model.config.moe_intermediate_size > buf_size)
        buf_size = model.config.moe_intermediate_size;
    if (model.config.ssm_inner_size > buf_size)
        buf_size = model.config.ssm_inner_size;
    for (int l = 0; l < model.config.n_layers; l++) {
        BnLayerWeights *lw = &model.weights.layers[l];
        const BnQWeight *weights[] = {
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, &lw->attn.wo,
            &lw->ffn.ffn_gate, &lw->ffn.ffn_up, &lw->ffn.ffn_down,
            &lw->ssm.wqkv, &lw->ssm.wz, &lw->ssm.ssm_alpha, &lw->ssm.ssm_beta, &lw->ssm.ssm_out,
            &lw->shared.shared_gate, &lw->shared.shared_up, &lw->shared.shared_down,
        };
        for (size_t i = 0; i < sizeof(weights) / sizeof(weights[0]); i++) {
            if (weights[i]->data && weights[i]->cols > buf_size)
                buf_size = weights[i]->cols;
        }
        if (lw->moe.expert_map.gate_cols > buf_size) buf_size = lw->moe.expert_map.gate_cols;
        if (lw->moe.expert_map.up_cols > buf_size) buf_size = lw->moe.expert_map.up_cols;
        if (lw->moe.expert_map.down_cols > buf_size) buf_size = lw->moe.expert_map.down_cols;
    }
    if (model.weights.output_weight.data && model.weights.output_weight.cols > buf_size)
        buf_size = model.weights.output_weight.cols;

    float *x = calloc((size_t)buf_size, sizeof(float));
    int8_t *x_q = calloc((size_t)buf_size, sizeof(int8_t));
    if (!x || !x_q) {
        fprintf(stderr, "Failed to allocate buffers\n");
        bn_model_free(&model);
        bn_gguf_free(gf);
        return 1;
    }

    // Fill with random data
    for (int i = 0; i < buf_size; i++) x[i] = bench_randf();

    // Print header
    printf("Backend: %-20s | Model: %-30s | Iters: %d | Threads: %d\n",
           backend_name(use_webgpu), fname, n_iters, pool ? bn_tp_num_threads(pool) : 1);
    printf("%-8s | %-7s | %-13s | %8s | %s\n",
           "Matrix", "Format", "Dims", "us/call", "GB/s");
    printf("---------|---------|---------------|----------|------\n");

    // Benchmark layer 0 weight matrices
    BnLayerWeights *L = &model.weights.layers[0];

    BenchTarget targets[] = {
        { "wq",   &L->attn.wq },
        { "wk",   &L->attn.wk },
        { "wv",   &L->attn.wv },
        { "wo",   &L->attn.wo },
        { "up",   &L->ffn.ffn_up },
        { "down",  &L->ffn.ffn_down },
    };
    int n_targets = sizeof(targets) / sizeof(targets[0]);

    // Add gate if present
    BenchTarget gate_target = { "gate", &L->ffn.ffn_gate };

    for (int i = 0; i < n_targets; i++) {
        if (targets[i].W->data)
            bench_matvec(targets[i].name, targets[i].W, x, x_q, pool, n_iters);
    }
    if (model.config.has_ffn_gate && gate_target.W->data)
        bench_matvec(gate_target.name, gate_target.W, x, x_q, pool, n_iters);

    // Benchmark logits
    if (model.weights.output_weight.data) {
        bench_logits_real(&model, n_iters, pool);
    } else if (model.weights.emb_type == BN_GGUF_TENSOR_F16) {
        bench_logits_f16(&model, x, n_iters);
    }

    // Tok/s benchmark (forward pass)
    bn_model_set_thread_pool(&model, pool, 0);
    BnSession *session = bn_session_create(&model, NULL);
    if (session) {
        bench_toks(&model, session, n_toks);
        bn_session_free(session, NULL);
    } else {
        fprintf(stderr, "Failed to create session for tok/s benchmark\n");
    }
    bn_model_set_thread_pool(&model, NULL, 0);

    // Cleanup
    free(x);
    free(x_q);
    if (pool) bn_tp_free(pool);
    bn_model_free(&model);
#ifdef BN_ENABLE_WEBGPU
    if (gpu) bn_gpu_wgpu_destroy(gpu);
#endif
    bn_gguf_free(gf);

    return 0;
}
