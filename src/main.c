#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "threadpool.h"
#include "sh_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread/qos.h>
#endif

typedef struct {
    const char *model_path;
    const char *prompt;
    int n_tokens;
    float temperature;
    float topp;
    uint64_t seed;
    int max_seq_len;
    int flash_attn;
} CLIArgs;

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.gguf> [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -p <prompt>     Input prompt (default: \"Hello\")\n");
    fprintf(stderr, "  -n <int>        Number of tokens to generate (default: 256)\n");
    fprintf(stderr, "  --temp <float>  Temperature (default: 0.0 = greedy)\n");
    fprintf(stderr, "  --topp <float>  Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --seed <int>    Random seed (default: 42)\n");
    fprintf(stderr, "  --maxseq <int>  Max sequence length (default: model max)\n");
    fprintf(stderr, "  --flash         Use flash attention (online softmax)\n");
}

static CLIArgs parse_args(int argc, char **argv) {
    CLIArgs args = {0};
    args.prompt = "Hello";
    args.n_tokens = 256;
    args.temperature = 0.0f;
    args.topp = 0.9f;
    args.seed = 42;
    args.max_seq_len = 0;

    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }

    args.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            args.n_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            args.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--topp") == 0 && i + 1 < argc) {
            args.topp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--maxseq") == 0 && i + 1 < argc) {
            args.max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--flash") == 0) {
            args.flash_attn = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    return args;
}

int main(int argc, char **argv) {
    sh_log_init(NULL);
    CLIArgs args = parse_args(argc, argv);

    // Detect P-core count on Apple Silicon for thread pool sizing
    int n_workers = 0;
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    {
        int ncores = 0;
        size_t len = sizeof(ncores);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &ncores, &len, NULL, 0) == 0 && ncores > 1)
            n_workers = ncores - 1;  // main thread counts as one
    }
#endif

    // Load model file
    SH_LOG_INFO("Loading model", "path", args.model_path);
    double t0 = bn_platform_time_ms();

    BnMappedFile mf = bn_platform_load_file(args.model_path);
    if (!mf.data) {
        SH_LOG_ERROR("Failed to load file", "path", args.model_path);
        return 1;
    }
    {
        char mb[32], ms[32];
        snprintf(mb, sizeof(mb), "%.1f", mf.size / (1024.0 * 1024.0));
        snprintf(ms, sizeof(ms), "%.0f", bn_platform_time_ms() - t0);
        SH_LOG_INFO("File loaded", "MB", mb, "ms", ms);
    }

    // Parse GGUF
    BnGGUFFile *gf = bn_gguf_open(mf.data, mf.size);
    if (!gf) {
        SH_LOG_ERROR("Failed to parse GGUF");
        bn_platform_unload_file(&mf);
        return 1;
    }
    {
        char ver[8], nt[16], nkv[16];
        snprintf(ver, sizeof(ver), "%u", gf->version);
        snprintf(nt, sizeof(nt), "%llu", (unsigned long long)gf->n_tensors);
        snprintf(nkv, sizeof(nkv), "%llu", (unsigned long long)gf->n_kv);
        SH_LOG_INFO("GGUF parsed", "version", ver, "tensors", nt, "kv", nkv);
    }

    // Load model
    BnModel model;
    if (bn_model_load(&model, gf, args.max_seq_len) != 0) {
        SH_LOG_ERROR("Failed to load model");
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    model.file = mf;  // keep mmap alive
    model.config.flash_attn = args.flash_attn;

    // Create thread pool
    model.pool = bn_tp_create(n_workers);
    if (model.pool) {
        char nt[8];
        snprintf(nt, sizeof(nt), "%d", bn_tp_num_threads(model.pool));
        SH_LOG_INFO("Thread pool created", "threads", nt);
    } else if (n_workers > 0) {
        SH_LOG_WARN("Failed to create thread pool, running single-threaded");
    }

    BnConfig *cfg = &model.config;

    {
        char dim[16], layers[16], heads[16], vocab[16], seq[16];
        snprintf(dim, sizeof(dim), "%d", cfg->dim);
        snprintf(layers, sizeof(layers), "%d", cfg->n_layers);
        snprintf(heads, sizeof(heads), "%d", cfg->n_heads);
        snprintf(vocab, sizeof(vocab), "%d", cfg->vocab_size);
        snprintf(seq, sizeof(seq), "%d", cfg->seq_len);
        SH_LOG_INFO("Model loaded", "dim", dim, "layers", layers, "heads", heads,
                     "vocab", vocab, "seq", seq);
    }

    // Initialize tokenizer
    BnTokenizer tokenizer;
    if (bn_tokenizer_init(&tokenizer, gf) != 0) {
        SH_LOG_ERROR("Failed to init tokenizer");
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    {
        char vs[16]; snprintf(vs, sizeof(vs), "%d", tokenizer.vocab_size);
        SH_LOG_INFO("Tokenizer loaded", "tokens", vs);
    }

    // #30: Encode prompt -- upper bound is 1 token per byte + BOS + margin
    int max_prompt_tokens = (int)strlen(args.prompt) + 3;
    int *prompt_tokens = (int *)malloc(max_prompt_tokens * sizeof(int));
    if (!prompt_tokens) {
        SH_LOG_ERROR("Failed to allocate prompt token buffer");
        bn_tokenizer_free(&tokenizer);
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    int n_prompt = bn_tokenizer_encode(&tokenizer, args.prompt, 1, prompt_tokens,
                                    max_prompt_tokens);
    {
        char np[16]; snprintf(np, sizeof(np), "%d", n_prompt);
        SH_LOG_INFO("Prompt encoded", "n_tokens", np);
    }

    // Initialize sampler
    BnSampler sampler;
    bn_sampler_init(&sampler, cfg->vocab_size, args.temperature, args.topp, args.seed);

    // Generation loop
    {
        char nt[16]; snprintf(nt, sizeof(nt), "%d", args.n_tokens);
        SH_LOG_INFO("Starting generation", "n_tokens", nt);
    }
    double gen_start = bn_platform_time_ms();
    int token = prompt_tokens[0];
    int pos = 0;
    int n_generated = 0;

    for (int i = 0; i < n_prompt + args.n_tokens; i++) {
        float *logits = bn_transformer_forward(&model, token, pos);
        if (!logits) {
            SH_LOG_ERROR("Forward pass returned NULL");
            break;
        }

        int next;
        if (i < n_prompt - 1) {
            // Still in prompt: force-feed next prompt token
            next = prompt_tokens[i + 1];
        } else {
            // Generate
            next = bn_sampler_sample(&sampler, logits);
            n_generated++;

            // Check for EOS/EOT
            if (next == tokenizer.eos_id || next == tokenizer.eot_id) {
                break;
            }
        }

        // Print token (skip BOS)
        if (i >= n_prompt - 1) {
            const char *piece = bn_tokenizer_decode(&tokenizer, next);
            if (!piece) piece = "";
            printf("%s", piece);
            fflush(stdout);
        }

        token = next;
        pos++;

        if (pos >= cfg->seq_len) {
            SH_LOG_WARN("Reached max sequence length");
            break;
        }
    }

    double gen_end = bn_platform_time_ms();
    double gen_time = gen_end - gen_start;
    double total_time = gen_end - t0;

    printf("\n");
    {
        char ng[16], speed[32], total[32];
        snprintf(ng, sizeof(ng), "%d", n_generated);
        if (n_generated > 0) {
            snprintf(speed, sizeof(speed), "%.2f", n_generated / (gen_time / 1000.0));
        } else {
            snprintf(speed, sizeof(speed), "0");
        }
        snprintf(total, sizeof(total), "%.1f", total_time);
        SH_LOG_INFO("Generation complete", "tokens", ng, "tok/s", speed, "total_ms", total);
    }

    // Cleanup
    free(prompt_tokens);
    bn_sampler_free(&sampler);
    bn_tokenizer_free(&tokenizer);
    bn_model_free(&model);
    bn_gguf_free(gf);
    bn_platform_unload_file(&mf);

    return 0;
}
