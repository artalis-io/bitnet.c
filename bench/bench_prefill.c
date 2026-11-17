#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "model_internal.h"
#include "tokenizer.h"
#include "session.h"
#include "transformer.h"
#include "threadpool.h"
#include "moe.h"
#include "generate.h"
#include "gpu_backend.h"
#include "gpu_graph_ir.h"
#ifdef BN_ENABLE_CUDA
#include "gpu_cuda.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [prompt_tokens] [runs] [threads] [options]\n"
        "Warm-cache benchmark; defaults: CPU, 512 tokens, 5 runs, 8 threads.\n"
        "  --cuda              Use CUDA (requires BN_ENABLE_CUDA=1 build)\n"
        "  --cuda-profile      Capture measured runs with cudaProfilerStart/Stop\n"
        "  --kv32              Use F32 KV (default: F16 KV)\n"
        "  --prompt-file PATH  Use the first prompt_tokens tokens of UTF-8 text\n"
        "                      (default: repeated hello, synthetic locality)\n"
        "  --with-logits       Include final-token logits in the measurement\n"
        "  --tokenwise         Sequential forward baseline, same output contract\n"
        "  --flash             Enable flash attention\n"
        "  --profile           Enable host prefill stage timings and MoE stats\n",
        prog);
}

static int positive_int(const char *arg, int *out) {
    char *end;
    errno = 0;
    long value = strtol(arg, &end, 10);
    if (errno || end == arg || *end || value <= 0 || value > INT_MAX / 2)
        return -1;
    *out = (int)value;
    return 0;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Tokenize the complete source before selecting a prefix. Insufficient text
 * is an error: silently repeating a corpus would distort expert locality. */
static int *prompt_tokens_from_file(BnTokenizer *tok, const char *path, int n) {
    BnMappedFile text = bn_platform_load_file(path);
    if (!text.data) return NULL;
    int *tokens = NULL;
    char *terminated = NULL;
    if (text.size == 0 || text.size > (size_t)INT_MAX - 16 ||
        text.size > SIZE_MAX / sizeof(int) - 16) {
        fprintf(stderr, "prompt file is empty or too large\n");
        goto done;
    }
    terminated = malloc(text.size + 1);
    tokens = malloc((text.size + 16) * sizeof(int));
    if (!terminated || !tokens) goto fail;
    memcpy(terminated, text.data, text.size);
    terminated[text.size] = '\0';
    if (memchr(terminated, '\0', text.size)) {
        fprintf(stderr, "prompt file contains a NUL byte\n");
        goto fail;
    }
    int count = bn_tokenizer_encode(tok, terminated, tok->add_bos, tokens,
                                     (int)text.size + 16);
    if (count < n) {
        fprintf(stderr, "prompt file has %d tokens; need %d\n", count, n);
        goto fail;
    }
    goto done;
fail:
    free(tokens);
    tokens = NULL;
done:
    free(terminated);
    bn_platform_unload_file(&text);
    return tokens;
}

static int *synthetic_tokens(BnTokenizer *tok, int n) {
    if ((size_t)n > (SIZE_MAX - 1) / 6 ||
        (size_t)n > SIZE_MAX / sizeof(int) - 16)
        return NULL;
    size_t bytes = (size_t)n * 6 + 1;
    char *prompt = malloc(bytes);
    int *tokens = malloc(((size_t)n + 16) * sizeof(int));
    if (!prompt || !tokens) {
        free(prompt);
        free(tokens);
        return NULL;
    }
    for (int i = 0; i < n; i++) memcpy(prompt + (size_t)i * 6, "hello ", 6);
    for (int words = n; words > 0; words--) {
        prompt[(size_t)words * 6] = '\0';
        if (bn_tokenizer_encode(tok, prompt, tok->add_bos, tokens, n + 16) == n) {
            free(prompt);
            return tokens;
        }
    }
    fprintf(stderr, "could not construct %d-token synthetic prompt\n", n);
    free(prompt);
    free(tokens);
    return NULL;
}

static int run_prefill(BnModel *model, BnSession *sess, const int *tokens,
                        int n, int tokenwise, int with_logits) {
    BnGPUBackend *gpu = bn_model_gpu(model);
    if (gpu) {
        int rc = with_logits
            ? (bn_prefill(model, sess, tokens, n, 0, tokenwise) ? 0 : -1)
            : bn_prefill_no_logits(model, sess, tokens, n, 0, tokenwise);
        if (rc != 0) return rc;
        /* Include completion of queued device work, including no-logit
         * tokenwise runs. Do not compare enqueue time with CPU wall time. */
        float completion;
        return bn_gpu_backend_read_activation(gpu, BN_GPU_VALUE_X,
                                               &completion, sizeof(completion), 0);
    }
    if (!tokenwise)
        return with_logits
            ? (bn_transformer_prefill(model, sess, tokens, n, 0) ? 0 : -1)
            : bn_transformer_prefill_no_logits(model, sess, tokens, n, 0);
    for (int t = 0; t < n; t++) {
        int rc = with_logits && t == n - 1
            ? (bn_transformer_forward(model, sess, tokens[t], t) ? 0 : -1)
            : bn_transformer_forward_no_logits(model, sess, tokens[t], t);
        if (rc != 0) return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "--help")) {
        usage(argv[0]);
        return argc >= 2 ? 0 : 1;
    }
    int n = 512, runs = 5, threads = 8, positional = 0;
    int with_logits = 0, tokenwise = 0, profile = 0, flash = 0, use_cuda = 0;
    int cuda_profile = 0;
    int kv_f16 = 1;
    const char *prompt_path = NULL;
    int *positionals[] = { &n, &runs, &threads };
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc)
            prompt_path = argv[++i];
        else if (!strcmp(argv[i], "--with-logits")) with_logits = 1;
        else if (!strcmp(argv[i], "--tokenwise")) tokenwise = 1;
        else if (!strcmp(argv[i], "--profile")) profile = 1;
        else if (!strcmp(argv[i], "--flash")) flash = 1;
        else if (!strcmp(argv[i], "--cuda")) use_cuda = 1;
        else if (!strcmp(argv[i], "--cuda-profile")) cuda_profile = 1;
        else if (!strcmp(argv[i], "--kv32")) kv_f16 = 0;
        else if (positional >= 3 || positive_int(argv[i], positionals[positional++])) {
            usage(argv[0]);
            return 1;
        }
    }
    if (cuda_profile && !use_cuda) {
        fprintf(stderr, "--cuda-profile requires --cuda\n");
        return 1;
    }
#ifndef BN_ENABLE_CUDA
    if (use_cuda) {
        fprintf(stderr, "--cuda requires BN_ENABLE_CUDA=1 build\n");
        return 1;
    }
#endif
    if ((size_t)runs > SIZE_MAX / sizeof(double)) return 1;
    if (profile && setenv("BN_PREFILL_PROFILE", "1", 1) != 0) return 1;

    int rc = 1, loaded = 0, tokenizer_ready = 0;
    BnModel model;
    BnTokenizer tok;
    BnSession *sess = NULL;
    BnGPUBackend *gpu = NULL;
    int *tokens = NULL;
    double *times = NULL;
    /* Let GGUF own mappings so split MoE checkpoints resolve all shards. */
    BnGGUFFile *gf = bn_gguf_open_file(argv[1]);
    if (!gf) goto cleanup;
    if (bn_model_load(&model, gf, n * 2, kv_f16, 0) != 0) goto cleanup;
    loaded = 1;
    model.config.flash_attn = flash;
    const BnMappedFile *mf = bn_gguf_primary_file(gf);
    if (mf) bn_model_set_file(&model, *mf);

    /* A zero-worker runtime still captures environment policy at t1. */
    BnThreadPool *pool = bn_tp_create(threads - 1);
    if (!pool) goto cleanup;
    bn_model_set_thread_pool(&model, pool, 1);
#ifdef BN_ENABLE_CUDA
    if (use_cuda) {
        gpu = bn_gpu_cuda_create();
        if (!gpu || bn_model_upload_weights(&model, gpu) != 0 ||
            !bn_gpu_backend_can_init_activations(gpu) ||
            bn_model_init_gpu_activations(&model, gpu) != 0) {
            fprintf(stderr, "CUDA setup failed; benchmark aborted\n");
            goto cleanup;
        }
    }
#endif
    if (bn_tokenizer_init(&tok, gf) != 0) goto cleanup;
    tokenizer_ready = 1;
    tokens = prompt_path ? prompt_tokens_from_file(&tok, prompt_path, n)
                         : synthetic_tokens(&tok, n);
    times = malloc((size_t)runs * sizeof(double));
    sess = bn_session_create(&model, NULL);
    if (!tokens || !times || !sess) {
        fprintf(stderr, "benchmark setup failed\n");
        goto cleanup;
    }
    if (run_prefill(&model, sess, tokens, n, tokenwise, with_logits) != 0) {
        fprintf(stderr, "warmup prefill failed\n");
        goto cleanup;
    }
    for (int r = 0; r < runs; r++) {
        bn_session_reset(sess, &model);
        if (sess->moe_state) bn_moe_reset_stats(sess->moe_state);
#ifdef BN_ENABLE_CUDA
        if (cuda_profile && (cudaDeviceSynchronize() != cudaSuccess ||
                             cudaProfilerStart() != cudaSuccess)) {
            fprintf(stderr, "CUDA profile capture start failed\n");
            goto cleanup;
        }
#endif
        double start = bn_platform_time_ms();
        int run_rc = run_prefill(&model, sess, tokens, n, tokenwise, with_logits);
        times[r] = bn_platform_time_ms() - start;
#ifdef BN_ENABLE_CUDA
        if (cuda_profile && cudaProfilerStop() != cudaSuccess) {
            fprintf(stderr, "CUDA profile capture stop failed\n");
            goto cleanup;
        }
#endif
        if (run_rc != 0) {
            fprintf(stderr, "prefill failed at run %d\n", r);
            goto cleanup;
        }
        if (profile && sess->moe_state) bn_moe_print_stats(sess->moe_state, n);
    }
    qsort(times, (size_t)runs, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int r = 0; r < runs; r++) sum += times[r];
    double median = (runs & 1) ? times[runs / 2]
        : (times[runs / 2 - 1] + times[runs / 2]) * 0.5;
    printf("prompt_tokens=%d runs=%d threads=%d median_ms=%.3f mean_ms=%.3f "
           "tok_s=%.3f backend=%s mode=%s output=%s prompt=%s kv=%s "
           "flash=%d cache=warm reset_timed=0 cuda_profile=%d\n",
           n, runs, threads, median, sum / runs, 1000.0 * n / median,
           gpu ? "cuda" : "cpu",
           tokenwise ? "tokenwise" : "batch", with_logits ? "last" : "none",
           prompt_path ? "file" : "synthetic-hello", kv_f16 ? "f16" : "f32",
           flash, cuda_profile);
    rc = 0;
cleanup:
    bn_session_free(sess, NULL);
    free(tokens);
    free(times);
    if (tokenizer_ready) bn_tokenizer_free(&tok);
    if (loaded) bn_model_free(&model);
#ifdef BN_ENABLE_CUDA
    bn_gpu_cuda_destroy(gpu);
#endif
    bn_gguf_free(gf);
    return rc;
}
