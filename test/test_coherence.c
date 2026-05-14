/*
 * test_coherence.c — Cross-backend coherence test for bitnet.c
 *
 * Verifies that GPU forward pass and CPU SIMD backends produce consistent
 * results against scalar baselines. Requires a real GGUF model file.
 *
 * Usage: ./test_coherence <model.gguf> [--webgpu]
 *
 * Phase 1: Forward pass coherence (GPU vs CPU greedy decode)
 * Phase 2: Matvec backend comparison (SIMD vs scalar)
 * Phase 3: GPU standalone matvec vs CPU scalar
 */

#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "session.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "moe.h"
#include "quant.h"
#include "quant_internal.h"
#include "gpu_backend.h"
#include "../src/gpu_shader.h"
#ifdef BN_ENABLE_WEBGPU
#include "gpu_wgpu.h"
#endif
#ifdef BN_ENABLE_METAL
#include "gpu_metal.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_DECODE_STEPS 5
#define N_MATCH_REQUIRED 3
#define MATVEC_TOL 2.0f  /* I2_S SDOT vs scalar can differ ~1.3 for large cols */

/* ── Helpers ──────────────────────────────────────────────────────── */

static const char *type_name(int type) {
    switch (type) {
    case BN_GGUF_TENSOR_I2_S:    return "I2_S";
    case BN_GGUF_TENSOR_TQ1_0:   return "TQ1_0";
    case BN_GGUF_TENSOR_TQ2_0:   return "TQ2_0";
    case BN_GGUF_TENSOR_Q4_0:    return "Q4_0";
    case BN_GGUF_TENSOR_Q4_1:    return "Q4_1";
    case BN_GGUF_TENSOR_Q8_0:    return "Q8_0";
    case BN_GGUF_TENSOR_BF16:    return "BF16";
    case BN_GGUF_TENSOR_F16:     return "F16";
    case BN_GGUF_TENSOR_F32:     return "F32";
    case BN_GGUF_TENSOR_Q2_K:    return "Q2_K";
    case BN_GGUF_TENSOR_Q3_K:    return "Q3_K";
    case BN_GGUF_TENSOR_Q4_K:    return "Q4_K";
    case BN_GGUF_TENSOR_Q5_K:    return "Q5_K";
    case BN_GGUF_TENSOR_Q6_K:    return "Q6_K";
    case BN_GGUF_TENSOR_Q8_K:    return "Q8_K";
    case BN_GGUF_TENSOR_IQ4_NL:  return "IQ4_NL";
    case BN_GGUF_TENSOR_IQ4_XS:  return "IQ4_XS";
    case BN_GGUF_TENSOR_IQ3_XXS: return "IQ3_XXS";
    case BN_GGUF_TENSOR_IQ3_S:   return "IQ3_S";
    case BN_GGUF_TENSOR_IQ2_XXS: return "IQ2_XXS";
    case BN_GGUF_TENSOR_IQ2_XS:  return "IQ2_XS";
    case BN_GGUF_TENSOR_IQ2_S:   return "IQ2_S";
    default:                      return "UNKNOWN";
    }
}

/* Seeded deterministic random float in [-1, 1] */
static float rand_float(uint64_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return (float)((int32_t)(*state & 0xFFFFFF) - (1 << 23)) / (float)(1 << 23);
}

static void print_vec_delta(const char *label,
                            int step,
                            const float *a,
                            const float *b,
                            int n) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    printf("    %s[%d]: max_abs=%.9g max_i=%d cpu=%.9g other=%.9g mean_abs=%.9g rms=%.9g\n",
           label, step, max_abs, max_i, a[max_i], b[max_i],
           sum_abs / (double)n, sqrt(sum_sq / (double)n));
}

#if defined(BN_ENABLE_WEBGPU) || defined(BN_ENABLE_METAL)
static int compare_prefill_kv_cache(BnGPUBackend *gpu,
                                    const BnConfig *config,
                                    const float *cpu_key,
                                    const float *cpu_value,
                                    int n_attn,
                                    int n_prompt) {
    if (!gpu || !gpu->read_activation || !cpu_key || !cpu_value)
        return 0;
    if (config->kv_tq_bits > 0 || config->kv_f16)
        return 0;

    size_t used_layer = (size_t)n_prompt * (size_t)config->kv_dim;
    size_t full_layer = (size_t)config->seq_len * (size_t)config->kv_dim;
    float *gpu_kv = malloc(used_layer * sizeof(float));
    if (!gpu_kv) {
        fprintf(stderr, "Failed to allocate GPU KV scratch\n");
        return -1;
    }

    for (int a = 0; a < n_attn; a++) {
        size_t packed_off = (size_t)a * used_layer;
        size_t gpu_off = (size_t)a * full_layer * sizeof(float);
        if (gpu->read_activation(gpu->ctx, BN_GPU_BUF_KEY_CACHE, gpu_kv,
                                 used_layer * sizeof(float), gpu_off) == 0) {
            print_vec_delta("prefill_key", a, cpu_key + packed_off, gpu_kv,
                            (int)used_layer);
        }
        if (gpu->read_activation(gpu->ctx, BN_GPU_BUF_VALUE_CACHE, gpu_kv,
                                 used_layer * sizeof(float), gpu_off) == 0) {
            print_vec_delta("prefill_value", a, cpu_value + packed_off, gpu_kv,
                            (int)used_layer);
        }
    }

    free(gpu_kv);
    return 0;
}
#endif

/* ── Phase 2: compare SIMD vs scalar matvec for a single weight ──── */

/* Get the explicit scalar range function for a given type.
 * All scalar contexts share the same {out, W, x} layout. */
typedef void (*scalar_fn)(void *ctx, int start, int end);
static scalar_fn get_scalar_fn(int type) {
    switch (type) {
    case BN_GGUF_TENSOR_I2_S:    return bn_quant_i2s_scalar_range;
    case BN_GGUF_TENSOR_TQ1_0:   return bn_quant_tq1_scalar_range;
    case BN_GGUF_TENSOR_TQ2_0:   return bn_quant_tq2_scalar_range;
    case BN_GGUF_TENSOR_Q4_0:    return bn_quant_q4_scalar_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_scalar_range;
    case BN_GGUF_TENSOR_Q8_0:    return bn_quant_q8_scalar_range;
    case BN_GGUF_TENSOR_F32:     return bn_quant_f32_scalar_range;
    case BN_GGUF_TENSOR_F16:     return bn_quant_f16_scalar_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_scalar_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_scalar_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_scalar_range;
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_scalar_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_scalar_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_scalar_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_scalar_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_scalar_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_scalar_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_scalar_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_scalar_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_scalar_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_scalar_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_scalar_range;
    default: return NULL;
    }
}

static const char *simd_backend_name(void) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return "NEON SDOT";
#elif defined(__ARM_NEON)
    return "NEON";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__wasm_relaxed_simd__)
    return "WASM relaxed";
#elif defined(__wasm_simd128__)
    return "WASM SIMD128";
#else
    return "scalar";
#endif
}

static int test_matvec_weight(const char *name, const BnQWeight *W, BnThreadPool *pool) {
    if (!W->data || W->rows == 0 || W->cols == 0) {
        printf("  %-12s SKIP (no data)\n", name);
        return 0; /* not a failure */
    }

    scalar_fn sfn = get_scalar_fn(W->type);
    if (!sfn) {
        printf("  %-12s SKIP (no scalar kernel for type %d)\n", name, W->type);
        return 0;
    }

    int rows = W->rows;
    int cols = W->cols;

    /* Allocate input, outputs, scratch */
    float *x = malloc((size_t)cols * sizeof(float));
    float *out_scalar = calloc((size_t)rows, sizeof(float));
    float *out_simd = calloc((size_t)rows, sizeof(float));
    int max_dim = cols > rows ? cols : rows;
    int8_t *x_q = calloc((size_t)max_dim, 1);
    if (!x || !out_scalar || !out_simd || !x_q) {
        printf("  %-12s SKIP (alloc failed)\n", name);
        free(x); free(out_scalar); free(out_simd); free(x_q);
        return 0;
    }

    /* Fill x with deterministic random values */
    uint64_t rng = 12345;
    for (int i = 0; i < cols; i++)
        x[i] = rand_float(&rng);

    /* Explicit scalar kernel (all scalar contexts share {out, W, x} layout) */
    BnFloatXCtx sctx = { out_scalar, W, x };
    sfn(&sctx, 0, rows);

    /* Compile-time best SIMD backend via bn_quant_matvec */
    bn_quant_matvec(out_simd, W, x, x_q, pool);

    /* Compare */
    float max_diff = 0.0f;
    for (int i = 0; i < rows; i++) {
        float diff = fabsf(out_simd[i] - out_scalar[i]);
        if (diff > max_diff) max_diff = diff;
    }

    int pass = max_diff < MATVEC_TOL;
    printf("  %-12s %-6s type=%-8s rows=%-5d cols=%-5d max_diff=%.4f (scalar vs %s)\n",
           name, pass ? "PASS" : "FAIL", type_name(W->type),
           rows, cols, max_diff, simd_backend_name());

    if (!pass) {
        for (int i = 0; i < rows && i < 8; i++) {
            float diff = fabsf(out_simd[i] - out_scalar[i]);
            if (diff > MATVEC_TOL * 0.1f)
                printf("    [%d] scalar=%.6f simd=%.6f diff=%.6f\n",
                       i, out_scalar[i], out_simd[i], diff);
        }
    }

    free(x); free(out_scalar); free(out_simd); free(x_q);
    return pass ? 1 : -1;
}

#if defined(BN_ENABLE_WEBGPU) || defined(BN_ENABLE_METAL)
static int test_gpu_matvec_weight(const char *backend_name,
                                  BnGPUBackend *gpu,
                                  const char *name,
                                  const BnQWeight *W,
                                  const float *bias) {
    if (!gpu || !W || !W->data || W->rows == 0 || W->cols == 0) {
        printf("  %-12s SKIP (no data)\n", name);
        return 0;
    }

    int rows = W->rows;
    int cols = W->cols;
    size_t sz = bn_qweight_data_size(W);
    void *gpu_buf = NULL;
    if (bias && gpu->buffer_create_biased) {
        gpu_buf = gpu->buffer_create_biased(
            gpu->ctx, W->data, sz, W->type, rows, cols,
            bias, (size_t)rows * sizeof(float));
    }
    if (!gpu_buf)
        gpu_buf = gpu->buffer_create(gpu->ctx, W->data, sz,
                                     W->type, rows, cols);
    if (!gpu_buf) {
        printf("  %-12s SKIP (%s buffer_create failed)\n",
               name, backend_name);
        return 0;
    }

    float *x = malloc((size_t)cols * sizeof(float));
    float *out_cpu = calloc((size_t)rows, sizeof(float));
    float *out_gpu = calloc((size_t)rows, sizeof(float));
    int max_dim = cols > rows ? cols : rows;
    int8_t *x_q = calloc((size_t)max_dim, 1);
    if (!x || !out_cpu || !out_gpu || !x_q) {
        printf("  %-12s SKIP (alloc failed)\n", name);
        free(x); free(out_cpu); free(out_gpu); free(x_q);
        gpu->buffer_destroy(gpu->ctx, gpu_buf);
        return 0;
    }

    uint64_t rng = 99999;
    for (int j = 0; j < cols; j++)
        x[j] = rand_float(&rng);

    bn_quant_matvec(out_cpu, W, x, x_q, NULL);
    if (bias) {
        for (int i = 0; i < rows; i++)
            out_cpu[i] += bias[i];
    }

    int rc = gpu->matvec(gpu->ctx, out_gpu, gpu_buf, x,
                         rows, cols, W->type);
    int result = 0;
    if (rc != 0) {
        printf("  %-12s SKIP (%s matvec dispatch error %d)\n",
               name, backend_name, rc);
    } else {
        float max_diff = 0.0f;
        for (int i = 0; i < rows; i++) {
            float diff = fabsf(out_gpu[i] - out_cpu[i]);
            if (diff > max_diff) max_diff = diff;
        }

        int pass = max_diff < MATVEC_TOL;
        printf("  %-12s %s vs CPU: %-6s max_diff=%.9g (rows=%d cols=%d type=%s)\n",
               name, backend_name, pass ? "PASS" : "FAIL",
               max_diff, rows, cols, type_name(W->type));
        if (pass) {
            result = 1;
        } else {
            result = -1;
            for (int i = 0; i < rows && i < 8; i++) {
                printf("    [%d] cpu=%.6f %s=%.6f diff=%.6f\n",
                       i, out_cpu[i], backend_name, out_gpu[i],
                       fabsf(out_gpu[i] - out_cpu[i]));
            }
        }
    }

    free(x); free(out_cpu); free(out_gpu); free(x_q);
    gpu->buffer_destroy(gpu->ctx, gpu_buf);
    return result;
}
#endif

/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [--webgpu] [--metal] [--prompt <text>] [--cpu-fallback-layer N] [--cpu-fallback-from-layer N] [--cpu-attn-layer N] [--cpu-attn-from-layer N] [--cpu-ffn-layer N] [--cpu-ffn-from-layer N] [--cpu-ffn-down-from-layer N] [--compare-attention-layer N] [--compare-attention-pos N] [--compare-gqa-layer N] [--compare-gqa-pos N] [--compare-qkv-layer N] [--compare-qkv-pos N] [--compare-ffn-down-layer N] [--compare-ffn-down-pos N] [--compare-ffn-state-layer N] [--compare-ffn-state-pos N] [--compare-logits] [--compare-hidden] [--compare-kv-cache] [--cpu-disable-prepared-qweights] [--metal-q4-prepared] [--metal-full-barriers] [--metal-disable-barriers] [--metal-disable-q4-q8] [--metal-cpu-rmsnorm] [--q4-q8-from-layer N] [--q4-q8-to-layer N] [--q4-q8-tail-native N] [--q4-q8-attn-only] [--q4-q8-ffn-only] [--disable-qkv-split] [--disable-gateup-split] [--disable-fused-gateup] [--split-residual-rmsnorm] [--flash]\n", argv[0]);
        fprintf(stderr, "Coherence test: WebGPU/Metal vs CPU forward pass, SIMD vs scalar matvec\n");
        return 1;
    }

    int use_webgpu = 0, use_metal = 0;
    int cpu_fallback_layer = -1;
    int cpu_fallback_from_layer = -1;
    int cpu_attn_layer = -1;
    int cpu_attn_from_layer = -1;
    int cpu_ffn_layer = -1;
    int cpu_ffn_from_layer = -1;
    int cpu_ffn_down_from_layer = -1;
    int compare_attention_layer = -1;
    int compare_attention_pos = -1;
    int compare_gqa_layer = -1;
    int compare_gqa_pos = -1;
    int compare_qkv_layer = -1;
    int compare_qkv_pos = -1;
    int compare_ffn_down_layer = -1;
    int compare_ffn_down_pos = -1;
    int compare_ffn_state_layer = -1;
    int compare_ffn_state_pos = -1;
    int compare_logits = 0;
    int compare_hidden = 0;
    int compare_kv_cache = 0;
    int cpu_disable_prepared_qweights = 0;
    int metal_q4_prepared = 0;
    int metal_full_barriers = 0;
    int metal_disable_barriers = 0;
    int metal_disable_q4_q8 = 0;
    int metal_cpu_rmsnorm = 0;
    int q4_q8_from_layer = -1;
    int q4_q8_to_layer = -1;
    int q4_q8_tail_native = -1;
    int q4_q8_attn_only = 0;
    int q4_q8_ffn_only = 0;
    int use_flash = 0;
    int disable_qkv_split = 0;
    int disable_gateup_split = 0;
    int disable_fused_gateup = 0;
    int split_residual_rmsnorm = 0;
    const char *prompt = "Hello";
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--webgpu") == 0) {
            use_webgpu = 1;
        } else if (strcmp(argv[i], "--metal") == 0) {
            use_metal = 1;
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--cpu-fallback-layer") == 0 && i + 1 < argc) {
            cpu_fallback_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-fallback-from-layer") == 0 && i + 1 < argc) {
            cpu_fallback_from_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-attn-layer") == 0 && i + 1 < argc) {
            cpu_attn_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-attn-from-layer") == 0 && i + 1 < argc) {
            cpu_attn_from_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-ffn-layer") == 0 && i + 1 < argc) {
            cpu_ffn_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-ffn-from-layer") == 0 && i + 1 < argc) {
            cpu_ffn_from_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-ffn-down-from-layer") == 0 && i + 1 < argc) {
            cpu_ffn_down_from_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-attention-layer") == 0 && i + 1 < argc) {
            compare_attention_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-attention-pos") == 0 && i + 1 < argc) {
            compare_attention_pos = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-gqa-layer") == 0 && i + 1 < argc) {
            compare_gqa_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-gqa-pos") == 0 && i + 1 < argc) {
            compare_gqa_pos = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-qkv-layer") == 0 && i + 1 < argc) {
            compare_qkv_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-qkv-pos") == 0 && i + 1 < argc) {
            compare_qkv_pos = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-ffn-down-layer") == 0 && i + 1 < argc) {
            compare_ffn_down_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-ffn-down-pos") == 0 && i + 1 < argc) {
            compare_ffn_down_pos = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-ffn-state-layer") == 0 && i + 1 < argc) {
            compare_ffn_state_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-ffn-state-pos") == 0 && i + 1 < argc) {
            compare_ffn_state_pos = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-logits") == 0) {
            compare_logits = 1;
        } else if (strcmp(argv[i], "--compare-hidden") == 0) {
            compare_hidden = 1;
        } else if (strcmp(argv[i], "--compare-kv-cache") == 0) {
            compare_kv_cache = 1;
        } else if (strcmp(argv[i], "--cpu-disable-prepared-qweights") == 0) {
            cpu_disable_prepared_qweights = 1;
        } else if (strcmp(argv[i], "--metal-q4-prepared") == 0) {
            metal_q4_prepared = 1;
        } else if (strcmp(argv[i], "--metal-full-barriers") == 0) {
            metal_full_barriers = 1;
        } else if (strcmp(argv[i], "--metal-disable-barriers") == 0) {
            metal_disable_barriers = 1;
        } else if (strcmp(argv[i], "--metal-disable-q4-q8") == 0) {
            metal_disable_q4_q8 = 1;
        } else if (strcmp(argv[i], "--metal-cpu-rmsnorm") == 0) {
            metal_cpu_rmsnorm = 1;
        } else if (strcmp(argv[i], "--q4-q8-from-layer") == 0 && i + 1 < argc) {
            q4_q8_from_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-q8-to-layer") == 0 && i + 1 < argc) {
            q4_q8_to_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-q8-tail-native") == 0 && i + 1 < argc) {
            q4_q8_tail_native = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-q8-attn-only") == 0) {
            q4_q8_attn_only = 1;
        } else if (strcmp(argv[i], "--q4-q8-ffn-only") == 0) {
            q4_q8_ffn_only = 1;
        } else if (strcmp(argv[i], "--flash") == 0) {
            use_flash = 1;
        } else if (strcmp(argv[i], "--disable-qkv-split") == 0) {
            disable_qkv_split = 1;
        } else if (strcmp(argv[i], "--disable-gateup-split") == 0) {
            disable_gateup_split = 1;
        } else if (strcmp(argv[i], "--disable-fused-gateup") == 0) {
            disable_fused_gateup = 1;
        } else if (strcmp(argv[i], "--split-residual-rmsnorm") == 0) {
            split_residual_rmsnorm = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }
    if (cpu_fallback_from_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", cpu_fallback_from_layer);
        setenv("BN_GPU_CPU_FALLBACK_FROM_LAYER", layer_env, 1);
    }
    if (cpu_fallback_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", cpu_fallback_layer);
        setenv("BN_GPU_CPU_FALLBACK_LAYER", layer_env, 1);
    }
    if (cpu_attn_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", cpu_attn_layer);
        setenv("BN_GPU_CPU_ATTN_LAYER", layer_env, 1);
    }
    if (cpu_attn_from_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", cpu_attn_from_layer);
        setenv("BN_GPU_CPU_ATTN_FROM_LAYER", layer_env, 1);
    }
    if (cpu_ffn_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", cpu_ffn_layer);
        setenv("BN_GPU_CPU_FFN_LAYER", layer_env, 1);
    }
    if (cpu_ffn_from_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", cpu_ffn_from_layer);
        setenv("BN_GPU_CPU_FFN_FROM_LAYER", layer_env, 1);
    }
    if (cpu_ffn_down_from_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", cpu_ffn_down_from_layer);
        setenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER", layer_env, 1);
    }
    if (compare_attention_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", compare_attention_layer);
        setenv("BN_GPU_COMPARE_ATTENTION_LAYER", layer_env, 1);
    }
    if (compare_attention_pos >= 0) {
        char pos_env[32];
        snprintf(pos_env, sizeof(pos_env), "%d", compare_attention_pos);
        setenv("BN_GPU_COMPARE_ATTENTION_POS", pos_env, 1);
    }
    if (compare_gqa_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", compare_gqa_layer);
        setenv("BN_GPU_COMPARE_GQA_LAYER", layer_env, 1);
    }
    if (compare_gqa_pos >= 0) {
        char pos_env[32];
        snprintf(pos_env, sizeof(pos_env), "%d", compare_gqa_pos);
        setenv("BN_GPU_COMPARE_GQA_POS", pos_env, 1);
    }
    if (compare_qkv_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", compare_qkv_layer);
        setenv("BN_GPU_COMPARE_QKV_LAYER", layer_env, 1);
    }
    if (compare_qkv_pos >= 0) {
        char pos_env[32];
        snprintf(pos_env, sizeof(pos_env), "%d", compare_qkv_pos);
        setenv("BN_GPU_COMPARE_QKV_POS", pos_env, 1);
    }
    if (compare_ffn_down_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", compare_ffn_down_layer);
        setenv("BN_GPU_COMPARE_FFN_DOWN_LAYER", layer_env, 1);
    }
    if (compare_ffn_down_pos >= 0) {
        char pos_env[32];
        snprintf(pos_env, sizeof(pos_env), "%d", compare_ffn_down_pos);
        setenv("BN_GPU_COMPARE_FFN_DOWN_POS", pos_env, 1);
    }
    if (compare_ffn_state_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", compare_ffn_state_layer);
        setenv("BN_GPU_COMPARE_FFN_STATE_LAYER", layer_env, 1);
    }
    if (compare_ffn_state_pos >= 0) {
        char pos_env[32];
        snprintf(pos_env, sizeof(pos_env), "%d", compare_ffn_state_pos);
        setenv("BN_GPU_COMPARE_FFN_STATE_POS", pos_env, 1);
    }
    if (q4_q8_from_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", q4_q8_from_layer);
        setenv("BN_GPU_Q4_Q8", "1", 1);
        setenv("BN_GPU_Q4_Q8_FROM_LAYER", layer_env, 1);
    }
    if (q4_q8_to_layer >= 0) {
        char layer_env[32];
        snprintf(layer_env, sizeof(layer_env), "%d", q4_q8_to_layer);
        setenv("BN_GPU_Q4_Q8_TO_LAYER", layer_env, 1);
    }
    if (q4_q8_tail_native >= 0) {
        char tail_env[32];
        snprintf(tail_env, sizeof(tail_env), "%d", q4_q8_tail_native);
        setenv("BN_GPU_Q4_Q8_TAIL_NATIVE", tail_env, 1);
    }
    if (q4_q8_attn_only)
        setenv("BN_GPU_Q4_Q8_ATTN_ONLY", "1", 1);
    if (q4_q8_ffn_only)
        setenv("BN_GPU_Q4_Q8_FFN_ONLY", "1", 1);
    if (disable_fused_gateup)
        setenv("BN_GPU_DISABLE_FUSED_GATEUP", "1", 1);
    if (disable_qkv_split)
        setenv("BN_GPU_DISABLE_QKV_SPLIT", "1", 1);
    if (disable_gateup_split)
        setenv("BN_GPU_DISABLE_GATEUP_SPLIT", "1", 1);
    if (split_residual_rmsnorm)
        setenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM", "1", 1);
    if (compare_logits)
        setenv("BN_GPU_COMPARE_LOGITS", "1", 1);
    if (cpu_disable_prepared_qweights)
        setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", "1", 1);
    if (metal_q4_prepared)
        setenv("BN_METAL_Q4_PREPARED", "1", 1);
    if (metal_full_barriers)
        setenv("BN_METAL_FULL_BARRIERS", "1", 1);
    if (metal_disable_barriers)
        setenv("BN_METAL_DISABLE_BARRIERS", "1", 1);
    if (metal_disable_q4_q8)
        setenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT", "1", 1);
    if (metal_cpu_rmsnorm)
        setenv("BN_METAL_CPU_ORDER_RMSNORM", "1", 1);

    int total_pass = 0, total_fail = 0, total_skip = 0;

    printf("=== Coherence Test ===\n");
    printf("Model: %s\n", argv[1]);
    printf("GPU:   %s\n\n", use_webgpu ? "webgpu" : use_metal ? "metal" : "off");

    /* ── Load model ──────────────────────────────────────────────── */

    BnMappedFile mf = bn_platform_load_file(argv[1]);
    if (!mf.data) {
        fprintf(stderr, "Failed to load file: %s\n", argv[1]);
        return 1;
    }

    BnGGUFFile *gf = bn_gguf_open(mf.data, mf.size);
    if (!gf) {
        fprintf(stderr, "Failed to parse GGUF\n");
        bn_platform_unload_file(&mf);
        return 1;
    }

    BnModel model;
    if (bn_model_load(&model, gf, 2048, 0, 0) != 0) {
        fprintf(stderr, "Failed to load model\n");
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    if (use_flash)
        model.config.flash_attn = 1;
    bn_model_set_file(&model, mf);
    if (model.config.n_experts > 0) {
        if (mf.is_mmap == 1 && mf.data)
            bn_model_set_moe_mmap_base(&model, mf.data);
        if (mf.fd >= 0)
            bn_model_set_moe_fd(&model, mf.fd);
        bn_moe_prefetch_create(bn_model_moe_io(&model));
    }

    BnTokenizer tok;
    if (bn_tokenizer_init(&tok, gf) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return 1;
    }

    /* ── Encode prompt ───────────────────────────────────────────── */

    int prompt_tokens[64];
    int n_prompt = bn_tokenizer_encode(&tok, prompt, tok.add_bos, prompt_tokens, 64);
    printf("Prompt: \"%s\" -> %d token(s): ", prompt, n_prompt);
    for (int i = 0; i < n_prompt; i++) printf("%d ", prompt_tokens[i]);
    printf("\n\n");

    /* ════════════════════════════════════════════════════════════════
     * Phase 1: Forward pass coherence (GPU vs CPU)
     * ════════════════════════════════════════════════════════════════ */

    printf("--- Phase 1: Forward pass coherence (greedy decode) ---\n");

    int vocab_size = model.config.vocab_size;
    int n_attn = (model.config.full_attn_interval > 0)
        ? model.config.n_layers / model.config.full_attn_interval
        : model.config.n_layers;
    size_t kv_used_count = (size_t)n_attn * (size_t)n_prompt *
                           (size_t)model.config.kv_dim;
    float *cpu_step_logits = calloc((size_t)N_DECODE_STEPS * (size_t)vocab_size,
                                    sizeof(float));
    float *cpu_step_hidden = calloc((size_t)N_DECODE_STEPS *
                                    (size_t)model.config.dim, sizeof(float));
    float *cpu_prefill_key = compare_kv_cache
        ? calloc(kv_used_count, sizeof(float)) : NULL;
    float *cpu_prefill_value = compare_kv_cache
        ? calloc(kv_used_count, sizeof(float)) : NULL;
    if (!cpu_step_logits || !cpu_step_hidden ||
        (compare_kv_cache && (!cpu_prefill_key || !cpu_prefill_value))) {
        fprintf(stderr, "Failed to allocate CPU step logits\n");
        free(cpu_step_logits);
        free(cpu_step_hidden);
        free(cpu_prefill_key);
        free(cpu_prefill_value);
        return 1;
    }

    /* CPU decode */
    int cpu_tokens[N_DECODE_STEPS];
    {
        /* Ensure no GPU for CPU baseline */
        bn_model_set_gpu_disabled(&model, 1);

        BnSession *s = bn_session_create(&model, NULL);
        if (!s) {
            fprintf(stderr, "Failed to create CPU session\n");
            return 1;
        }

        BnSampler sampler;
        bn_sampler_init(&sampler, model.config.vocab_size, 0.0f, 0.0f, 42);

        int token = prompt_tokens[0];
        int pos = 0;
        float *logits = NULL;

        /* Prefill prompt tokens */
        for (int i = 0; i < n_prompt; i++) {
            logits = bn_transformer_forward(&model, s, token, pos);
            if (i < n_prompt - 1)
                token = prompt_tokens[i + 1];
            pos++;
        }

        if (compare_kv_cache && model.config.kv_tq_bits == 0 &&
            !model.config.kv_f16) {
            size_t used_layer = (size_t)n_prompt *
                                (size_t)model.config.kv_dim;
            size_t full_layer = (size_t)model.config.seq_len *
                                (size_t)model.config.kv_dim;
            for (int a = 0; a < n_attn; a++) {
                memcpy(cpu_prefill_key + (size_t)a * used_layer,
                       s->state.key_cache + (size_t)a * full_layer,
                       used_layer * sizeof(float));
                memcpy(cpu_prefill_value + (size_t)a * used_layer,
                       s->state.value_cache + (size_t)a * full_layer,
                       used_layer * sizeof(float));
            }
        }

        /* Greedy decode N_DECODE_STEPS tokens */
        for (int i = 0; i < N_DECODE_STEPS; i++) {
            memcpy(cpu_step_logits + (size_t)i * (size_t)vocab_size,
                   logits, (size_t)vocab_size * sizeof(float));
            memcpy(cpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                   s->state.x, (size_t)model.config.dim * sizeof(float));
            token = bn_sampler_sample(&sampler, logits);
            cpu_tokens[i] = token;
            if (i + 1 < N_DECODE_STEPS) {
                logits = bn_transformer_forward(&model, s, token, pos);
                pos++;
            }
        }

        printf("  CPU tokens: ");
        for (int i = 0; i < N_DECODE_STEPS; i++) {
            const char *piece = bn_tokenizer_decode(&tok, cpu_tokens[i]);
            printf("[%d]=%d(\"%s\") ", i, cpu_tokens[i], piece ? piece : "?");
        }
        printf("\n");

        bn_sampler_free(&sampler);
        bn_session_free(s, NULL);
        bn_model_set_gpu_disabled(&model, 0);
    }

#ifdef BN_ENABLE_WEBGPU
    if (use_webgpu) {
        /* GPU decode */
        BnGPUBackend *gpu = bn_gpu_wgpu_create("shaders/");
        if (!gpu) {
            printf("  GPU: not available, skipping Phase 1 GPU comparison\n");
            total_skip++;
        } else {
            if (bn_model_upload_weights(&model, gpu) != 0) {
                printf("  GPU: weight upload failed, skipping\n");
                bn_gpu_wgpu_destroy(gpu);
                total_skip++;
            } else {
                if (gpu->init_activations)
                    gpu->init_activations(gpu->ctx, &model.config);

                int gpu_tokens[N_DECODE_STEPS];

                BnSession *s = bn_session_create(&model, NULL);
                if (!s) {
                    fprintf(stderr, "Failed to create GPU session\n");
                    return 1;
                }

                BnSampler sampler;
                bn_sampler_init(&sampler, model.config.vocab_size, 0.0f, 0.0f, 42);
                float *gpu_step_logits = calloc((size_t)N_DECODE_STEPS *
                                                (size_t)vocab_size,
                                                sizeof(float));
                float *gpu_step_hidden = calloc((size_t)N_DECODE_STEPS *
                                                (size_t)model.config.dim,
                                                sizeof(float));
                if (!gpu_step_logits || !gpu_step_hidden) {
                    fprintf(stderr, "Failed to allocate GPU step logits\n");
                    free(gpu_step_logits);
                    free(gpu_step_hidden);
                    return 1;
                }

                int token = prompt_tokens[0];
                int pos = 0;
                float *logits = NULL;

                for (int i = 0; i < n_prompt; i++) {
                    logits = bn_transformer_forward(&model, s, token, pos);
                    if (i < n_prompt - 1)
                        token = prompt_tokens[i + 1];
                    pos++;
                }

                if (compare_kv_cache &&
                    compare_prefill_kv_cache(gpu, &model.config,
                                             cpu_prefill_key,
                                             cpu_prefill_value,
                                             n_attn, n_prompt) != 0)
                    return 1;

                for (int i = 0; i < N_DECODE_STEPS; i++) {
                    memcpy(gpu_step_logits + (size_t)i * (size_t)vocab_size,
                           logits, (size_t)vocab_size * sizeof(float));
                    memcpy(gpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                           s->state.x, (size_t)model.config.dim * sizeof(float));
                    token = bn_sampler_sample(&sampler, logits);
                    gpu_tokens[i] = token;
                    if (i + 1 < N_DECODE_STEPS) {
                        logits = bn_transformer_forward(&model, s, token, pos);
                        pos++;
                    }
                }

                printf("  GPU tokens: ");
                for (int i = 0; i < N_DECODE_STEPS; i++) {
                    const char *piece = bn_tokenizer_decode(&tok, gpu_tokens[i]);
                    printf("[%d]=%d(\"%s\") ", i, gpu_tokens[i], piece ? piece : "?");
                }
                printf("\n");

                /* Compare: first N_MATCH_REQUIRED must match */
                for (int i = 0; i < N_DECODE_STEPS; i++) {
                    int match = (cpu_tokens[i] == gpu_tokens[i]);
                    int required = (i < N_MATCH_REQUIRED);
                    if (match) {
                        printf("  token[%d]: PASS (cpu=%d gpu=%d)\n",
                               i, cpu_tokens[i], gpu_tokens[i]);
                        if (compare_hidden) {
                            print_vec_delta(
                                "hidden", i,
                                cpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                                gpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                                model.config.dim);
                        }
                        total_pass++;
                    } else if (required) {
                        printf("  token[%d]: FAIL (cpu=%d gpu=%d) [REQUIRED]\n",
                               i, cpu_tokens[i], gpu_tokens[i]);
                        printf("    logits: cpu[cpu]=%.6f cpu[gpu]=%.6f gpu[cpu]=%.6f gpu[gpu]=%.6f\n",
                               cpu_step_logits[(size_t)i * (size_t)vocab_size + cpu_tokens[i]],
                               cpu_step_logits[(size_t)i * (size_t)vocab_size + gpu_tokens[i]],
                               gpu_step_logits[(size_t)i * (size_t)vocab_size + cpu_tokens[i]],
                               gpu_step_logits[(size_t)i * (size_t)vocab_size + gpu_tokens[i]]);
                        print_vec_delta(
                            "hidden", i,
                            cpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                            gpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                            model.config.dim);
                        total_fail++;
                    } else {
                        printf("  token[%d]: DRIFT (cpu=%d gpu=%d) [allowed]\n",
                               i, cpu_tokens[i], gpu_tokens[i]);
                        total_pass++; /* drift after N_MATCH_REQUIRED is OK */
                    }
                }

                bn_sampler_free(&sampler);
                free(gpu_step_logits);
                free(gpu_step_hidden);
                bn_session_free(s, NULL);

                /* Clean up GPU for Phase 3 reuse */
                if (gpu->free_activations)
                    gpu->free_activations(gpu->ctx);
                bn_model_release_gpu(&model);
                bn_gpu_wgpu_destroy(gpu);
            }
        }
    } else
#endif
#ifdef BN_ENABLE_METAL
    if (use_metal) {
        BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
        if (!gpu) {
            printf("  Metal: not available, skipping Phase 1 Metal comparison\n");
            total_skip++;
        } else {
            if (bn_model_upload_weights(&model, gpu) != 0) {
                printf("  Metal: weight upload failed, skipping\n");
                bn_gpu_metal_destroy(gpu);
                total_skip++;
            } else {
                if (gpu->init_activations)
                    gpu->init_activations(gpu->ctx, &model.config);

                int gpu_tokens[N_DECODE_STEPS];

                BnSession *s = bn_session_create(&model, NULL);
                if (!s) {
                    fprintf(stderr, "Failed to create Metal session\n");
                    return 1;
                }

                BnSampler sampler;
                bn_sampler_init(&sampler, model.config.vocab_size, 0.0f, 0.0f, 42);
                float *gpu_step_logits = calloc((size_t)N_DECODE_STEPS *
                                                (size_t)vocab_size,
                                                sizeof(float));
                float *gpu_step_hidden = calloc((size_t)N_DECODE_STEPS *
                                                (size_t)model.config.dim,
                                                sizeof(float));
                if (!gpu_step_logits || !gpu_step_hidden) {
                    fprintf(stderr, "Failed to allocate Metal step logits\n");
                    free(gpu_step_logits);
                    free(gpu_step_hidden);
                    return 1;
                }

                int token = prompt_tokens[0];
                int pos = 0;
                float *logits = NULL;

                for (int i = 0; i < n_prompt; i++) {
                    logits = bn_transformer_forward(&model, s, token, pos);
                    if (i < n_prompt - 1)
                        token = prompt_tokens[i + 1];
                    pos++;
                }

                if (compare_kv_cache &&
                    compare_prefill_kv_cache(gpu, &model.config,
                                             cpu_prefill_key,
                                             cpu_prefill_value,
                                             n_attn, n_prompt) != 0)
                    return 1;

                for (int i = 0; i < N_DECODE_STEPS; i++) {
                    memcpy(gpu_step_logits + (size_t)i * (size_t)vocab_size,
                           logits, (size_t)vocab_size * sizeof(float));
                    memcpy(gpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                           s->state.x, (size_t)model.config.dim * sizeof(float));
                    token = bn_sampler_sample(&sampler, logits);
                    gpu_tokens[i] = token;
                    if (i + 1 < N_DECODE_STEPS) {
                        logits = bn_transformer_forward(&model, s, token, pos);
                        pos++;
                    }
                }

                printf("  Metal tokens: ");
                for (int i = 0; i < N_DECODE_STEPS; i++) {
                    const char *piece = bn_tokenizer_decode(&tok, gpu_tokens[i]);
                    printf("[%d]=%d(\"%s\") ", i, gpu_tokens[i], piece ? piece : "?");
                }
                printf("\n");

                for (int i = 0; i < N_DECODE_STEPS; i++) {
                    int match = (cpu_tokens[i] == gpu_tokens[i]);
                    int required = (i < N_MATCH_REQUIRED);
                    if (match) {
                        printf("  token[%d]: PASS (cpu=%d metal=%d)\n",
                               i, cpu_tokens[i], gpu_tokens[i]);
                        if (compare_hidden) {
                            print_vec_delta(
                                "hidden", i,
                                cpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                                gpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                                model.config.dim);
                        }
                        total_pass++;
                    } else if (required) {
                        printf("  token[%d]: FAIL (cpu=%d metal=%d) [REQUIRED]\n",
                               i, cpu_tokens[i], gpu_tokens[i]);
                        printf("    logits: cpu[cpu]=%.6f cpu[metal]=%.6f metal[cpu]=%.6f metal[metal]=%.6f\n",
                               cpu_step_logits[(size_t)i * (size_t)vocab_size + cpu_tokens[i]],
                               cpu_step_logits[(size_t)i * (size_t)vocab_size + gpu_tokens[i]],
                               gpu_step_logits[(size_t)i * (size_t)vocab_size + cpu_tokens[i]],
                               gpu_step_logits[(size_t)i * (size_t)vocab_size + gpu_tokens[i]]);
                        print_vec_delta(
                            "hidden", i,
                            cpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                            gpu_step_hidden + (size_t)i * (size_t)model.config.dim,
                            model.config.dim);
                        total_fail++;
                    } else {
                        printf("  token[%d]: DRIFT (cpu=%d metal=%d) [allowed]\n",
                               i, cpu_tokens[i], gpu_tokens[i]);
                        total_pass++;
                    }
                }

                bn_sampler_free(&sampler);
                free(gpu_step_logits);
                free(gpu_step_hidden);
                bn_session_free(s, NULL);

                if (gpu->free_activations)
                    gpu->free_activations(gpu->ctx);
                bn_model_release_gpu(&model);
                bn_gpu_metal_destroy(gpu);
            }
        }
    } else
#endif
    {
        if (use_webgpu || use_metal)
            printf("  GPU: not compiled, skipping\n");
        else
            printf("  GPU: not requested, skipping GPU vs CPU comparison\n");
        total_skip += N_DECODE_STEPS;
    }

    printf("\n");
    free(cpu_step_logits);
    free(cpu_step_hidden);
    free(cpu_prefill_key);
    free(cpu_prefill_value);

    /* ════════════════════════════════════════════════════════════════
     * Phase 2: Matvec backend comparison (SIMD vs scalar)
     * ════════════════════════════════════════════════════════════════ */

    printf("--- Phase 2: Matvec SIMD vs scalar (layer 0 weights) ---\n");

    BnLayerWeights *L0 = &model.weights.layers[0];

    typedef struct { const char *name; const BnQWeight *W; } WeightEntry;
    WeightEntry weights[] = {
        { "wq",       &L0->attn.wq },
        { "wk",       &L0->attn.wk },
        { "wv",       &L0->attn.wv },
        { "wo",       &L0->attn.wo },
        { "ffn_gate", &L0->ffn.ffn_gate },
        { "ffn_up",   &L0->ffn.ffn_up },
        { "ffn_down", &L0->ffn.ffn_down },
    };
    int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));

    for (int i = 0; i < n_weights; i++) {
        int r = test_matvec_weight(weights[i].name, weights[i].W, bn_model_pool(&model));
        if (r == 1) total_pass++;
        else if (r == -1) total_fail++;
        else total_skip++;
    }

    printf("\n");

    /* ════════════════════════════════════════════════════════════════
     * Phase 3: GPU standalone matvec vs CPU scalar
     * ════════════════════════════════════════════════════════════════ */

    printf("--- Phase 3: GPU standalone matvec vs CPU scalar (layer 0 weight) ---\n");

    const BnQWeight *phase3_W = &L0->attn.wq;
    const char *phase3_name = "wq";
    if (!phase3_W->data || phase3_W->rows == 0) {
        if (L0->ssm.wqkv.data && L0->ssm.wqkv.rows > 0) {
            phase3_W = &L0->ssm.wqkv;
            phase3_name = "wqkv";
        } else if (L0->ffn.ffn_gate.data && L0->ffn.ffn_gate.rows > 0) {
            phase3_W = &L0->ffn.ffn_gate;
            phase3_name = "ffn_gate";
        }
    }
#if !defined(BN_ENABLE_WEBGPU) && !defined(BN_ENABLE_METAL)
    (void)phase3_name;
#endif

#ifdef BN_ENABLE_WEBGPU
    if (use_webgpu) {
        BnGPUBackend *gpu = bn_gpu_wgpu_create("shaders/");
        if (!gpu) {
            printf("  GPU: not available, skipping Phase 3\n");
            total_skip++;
        } else {
            const BnQWeight *W = phase3_W;
            if (!W->data || W->rows == 0) {
                printf("  SKIP: no layer 0 matvec weight has data\n");
                total_skip++;
            } else {
                int rows = W->rows;
                int cols = W->cols;

                /* Upload weight to GPU */
                size_t sz = bn_qweight_data_size(W);
                void *gpu_buf = gpu->buffer_create(gpu->ctx, W->data, sz,
                                                    W->type, rows, cols);
                if (!gpu_buf) {
                    printf("  SKIP: buffer_create failed\n");
                    total_skip++;
                } else {
                    /* Random input */
                    float *x = malloc((size_t)cols * sizeof(float));
                    uint64_t rng = 99999;
                    for (int j = 0; j < cols; j++)
                        x[j] = rand_float(&rng);

                    /* CPU scalar */
                    float *out_cpu = calloc((size_t)rows, sizeof(float));
                    int max_dim = cols > rows ? cols : rows;
                    int8_t *x_q = calloc((size_t)max_dim, 1);
                    bn_quant_matvec(out_cpu, W, x, x_q, NULL);

                    /* GPU */
                    float *out_gpu = calloc((size_t)rows, sizeof(float));
                    int rc = gpu->matvec(gpu->ctx, out_gpu, gpu_buf, x,
                                          rows, cols, W->type);
                    if (rc != 0) {
                        printf("  SKIP: GPU matvec dispatch error %d\n", rc);
                        total_skip++;
                    } else {
                        float max_diff = 0.0f;
                        for (int i = 0; i < rows; i++) {
                            float diff = fabsf(out_gpu[i] - out_cpu[i]);
                            if (diff > max_diff) max_diff = diff;
                        }

                        int pass = max_diff < MATVEC_TOL;
                        printf("  %s GPU vs CPU: %-6s max_diff=%.4f (rows=%d cols=%d type=%s)\n",
                               phase3_name, pass ? "PASS" : "FAIL", max_diff, rows, cols, type_name(W->type));
                        if (pass)
                            total_pass++;
                        else {
                            total_fail++;
                            for (int i = 0; i < rows && i < 8; i++) {
                                printf("    [%d] cpu=%.6f gpu=%.6f diff=%.6f\n",
                                       i, out_cpu[i], out_gpu[i],
                                       fabsf(out_gpu[i] - out_cpu[i]));
                            }
                        }
                    }

                    free(x); free(out_cpu); free(out_gpu); free(x_q);
                    gpu->buffer_destroy(gpu->ctx, gpu_buf);
                }
            }

            const BnQWeight *logits_W = NULL;
            BnQWeight tied_logits;
            if (model.weights.output_weight.data) {
                logits_W = &model.weights.output_weight;
            } else if (bn_quant_format_supported(model.weights.emb_type) &&
                       model.weights.emb_type != BN_GGUF_TENSOR_F16 &&
                       model.weights.emb_type != BN_GGUF_TENSOR_F32) {
                tied_logits = (BnQWeight){
                    model.weights.token_embedding,
                    model.weights.emb_type,
                    model.config.vocab_size,
                    model.config.dim,
                    1.0f
                };
                logits_W = &tied_logits;
            }
            if (logits_W && logits_W != W) {
                int r = test_gpu_matvec_weight("GPU", gpu, "logits", logits_W, NULL);
                if (r == 1) total_pass++;
                else if (r == -1) total_fail++;
                else total_skip++;
            }
            {
                int r = test_gpu_matvec_weight("GPU", gpu, "wq+bias",
                                               &L0->attn.wq, L0->attn.q_bias);
                if (r == 1) total_pass++;
                else if (r == -1) total_fail++;
                else total_skip++;
            }
            bn_gpu_wgpu_destroy(gpu);
        }
    } else
#endif
#ifdef BN_ENABLE_METAL
    if (use_metal) {
        BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
        if (!gpu) {
            printf("  Metal: not available, skipping Phase 3\n");
            total_skip++;
        } else {
            const BnQWeight *W = phase3_W;
            if (!W->data || W->rows == 0) {
                printf("  SKIP: no layer 0 matvec weight has data\n");
                total_skip++;
            } else {
                int rows = W->rows;
                int cols = W->cols;

                size_t sz = bn_qweight_data_size(W);
                void *gpu_buf = gpu->buffer_create(gpu->ctx, W->data, sz,
                                                    W->type, rows, cols);
                if (!gpu_buf) {
                    printf("  SKIP: buffer_create failed\n");
                    total_skip++;
                } else {
                    float *x = malloc((size_t)cols * sizeof(float));
                    uint64_t rng = 99999;
                    for (int j = 0; j < cols; j++)
                        x[j] = rand_float(&rng);

                    float *out_cpu = calloc((size_t)rows, sizeof(float));
                    int max_dim = cols > rows ? cols : rows;
                    int8_t *x_q = calloc((size_t)max_dim, 1);
                    bn_quant_matvec(out_cpu, W, x, x_q, NULL);

                    float *out_gpu = calloc((size_t)rows, sizeof(float));
                    int rc = gpu->matvec(gpu->ctx, out_gpu, gpu_buf, x,
                                          rows, cols, W->type);
                    if (rc != 0) {
                        printf("  SKIP: Metal matvec dispatch error %d\n", rc);
                        total_skip++;
                    } else {
                        float max_diff = 0.0f;
                        for (int i = 0; i < rows; i++) {
                            float diff = fabsf(out_gpu[i] - out_cpu[i]);
                            if (diff > max_diff) max_diff = diff;
                        }

                        int pass = max_diff < MATVEC_TOL;
                        printf("  %s Metal vs CPU: %-6s max_diff=%.4f (rows=%d cols=%d type=%s)\n",
                               phase3_name, pass ? "PASS" : "FAIL", max_diff, rows, cols, type_name(W->type));
                        if (pass)
                            total_pass++;
                        else {
                            total_fail++;
                            for (int i = 0; i < rows && i < 8; i++) {
                                printf("    [%d] cpu=%.6f metal=%.6f diff=%.6f\n",
                                       i, out_cpu[i], out_gpu[i],
                                       fabsf(out_gpu[i] - out_cpu[i]));
                            }
                        }
                    }

                    free(x); free(out_cpu); free(out_gpu); free(x_q);
                    gpu->buffer_destroy(gpu->ctx, gpu_buf);
                }
            }

            const BnQWeight *logits_W = NULL;
            BnQWeight tied_logits;
            if (model.weights.output_weight.data) {
                logits_W = &model.weights.output_weight;
            } else if (bn_quant_format_supported(model.weights.emb_type) &&
                       model.weights.emb_type != BN_GGUF_TENSOR_F16 &&
                       model.weights.emb_type != BN_GGUF_TENSOR_F32) {
                tied_logits = (BnQWeight){
                    model.weights.token_embedding,
                    model.weights.emb_type,
                    model.config.vocab_size,
                    model.config.dim,
                    1.0f
                };
                logits_W = &tied_logits;
            }
            if (logits_W && logits_W != W) {
                int r = test_gpu_matvec_weight("Metal", gpu, "logits", logits_W, NULL);
                if (r == 1) total_pass++;
                else if (r == -1) total_fail++;
                else total_skip++;
            }
            {
                int r = test_gpu_matvec_weight("Metal", gpu, "wk",
                                               &L0->attn.wk, L0->attn.k_bias);
                if (r == 1) total_pass++;
                else if (r == -1) total_fail++;
                else total_skip++;
            }
            {
                int r = test_gpu_matvec_weight("Metal", gpu, "wv",
                                               &L0->attn.wv, L0->attn.v_bias);
                if (r == 1) total_pass++;
                else if (r == -1) total_fail++;
                else total_skip++;
            }
            {
                int r = test_gpu_matvec_weight("Metal", gpu, "wq+bias",
                                               &L0->attn.wq, L0->attn.q_bias);
                if (r == 1) total_pass++;
                else if (r == -1) total_fail++;
                else total_skip++;
            }
            bn_gpu_metal_destroy(gpu);
        }
    } else
#endif
    {
        if (use_webgpu || use_metal)
            printf("  GPU: not compiled, skipping\n");
        else
            printf("  GPU: not requested, skipping\n");
        total_skip++;
    }

    printf("\n");

    /* ── Summary ──────────────────────────────────────────────────── */

    printf("=== Coherence Test Summary ===\n");
    printf("  PASS: %d  FAIL: %d  SKIP: %d\n", total_pass, total_fail, total_skip);
    printf("  Result: %s\n", total_fail == 0 ? "PASS" : "FAIL");

    /* ── Cleanup ──────────────────────────────────────────────────── */

    bn_tokenizer_free(&tok);
    bn_model_free(&model);
    bn_gguf_free(gf);

    return total_fail > 0 ? 1 : 0;
}
