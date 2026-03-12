#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *model_path;
    const char *prompt;
    int n_tokens;
    float temperature;
    float topp;
    uint64_t seed;
    int max_seq_len;
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
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    return args;
}

int main(int argc, char **argv) {
    CLIArgs args = parse_args(argc, argv);

    // Load model file
    fprintf(stderr, "Loading %s...\n", args.model_path);
    double t0 = platform_time_ms();

    MappedFile mf = platform_load_file(args.model_path);
    if (!mf.data) {
        fprintf(stderr, "Failed to load file: %s\n", args.model_path);
        return 1;
    }
    fprintf(stderr, "File loaded: %.1f MB (%.0f ms)\n",
            mf.size / (1024.0 * 1024.0), platform_time_ms() - t0);

    // Parse GGUF
    GGUFFile *gf = gguf_open(mf.data, mf.size);
    if (!gf) {
        fprintf(stderr, "Failed to parse GGUF\n");
        platform_unload_file(&mf);
        return 1;
    }
    fprintf(stderr, "GGUF v%u: %llu tensors, %llu kv pairs\n",
            gf->version, (unsigned long long)gf->n_tensors,
            (unsigned long long)gf->n_kv);

    // Load model
    Model model;
    if (model_load(&model, gf, args.max_seq_len) != 0) {
        fprintf(stderr, "Failed to load model\n");
        gguf_free(gf);
        platform_unload_file(&mf);
        return 1;
    }
    model.file = mf;  // keep mmap alive
    Config *cfg = &model.config;

    fprintf(stderr, "Model: dim=%d layers=%d heads=%d vocab=%d seq=%d\n",
            cfg->dim, cfg->n_layers, cfg->n_heads, cfg->vocab_size, cfg->seq_len);

    // Initialize tokenizer
    Tokenizer tokenizer;
    if (tokenizer_init(&tokenizer, gf) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        model_free(&model);
        gguf_free(gf);
        platform_unload_file(&mf);
        return 1;
    }
    fprintf(stderr, "Tokenizer: %d tokens\n", tokenizer.vocab_size);

    // Encode prompt
    int *prompt_tokens = (int *)malloc((strlen(args.prompt) + 3) * sizeof(int));
    int n_prompt = tokenizer_encode(&tokenizer, args.prompt, 1, prompt_tokens,
                                    (int)strlen(args.prompt) + 3);
    fprintf(stderr, "Prompt tokens (%d): ", n_prompt);
    for (int i = 0; i < n_prompt; i++) fprintf(stderr, "%d ", prompt_tokens[i]);
    fprintf(stderr, "\n");

    // Initialize sampler
    Sampler sampler;
    sampler_init(&sampler, cfg->vocab_size, args.temperature, args.topp, args.seed);

    // Generation loop
    fprintf(stderr, "Generating %d tokens...\n", args.n_tokens);
    double gen_start = platform_time_ms();
    int token = prompt_tokens[0];
    int pos = 0;
    int n_generated = 0;

    for (int i = 0; i < n_prompt + args.n_tokens; i++) {
        float *logits = transformer_forward(&model, token, pos);

        int next;
        if (i < n_prompt - 1) {
            // Still in prompt: force-feed next prompt token
            next = prompt_tokens[i + 1];
        } else {
            // Generate
            next = sampler_sample(&sampler, logits);
            n_generated++;

            // Check for EOS/EOT
            if (next == tokenizer.eos_id || next == tokenizer.eot_id) {
                break;
            }
        }

        // Print token (skip BOS)
        if (i >= n_prompt - 1) {
            const char *piece = tokenizer_decode(&tokenizer, next);
            printf("%s", piece);
            fflush(stdout);
        }

        token = next;
        pos++;

        if (pos >= cfg->seq_len) {
            fprintf(stderr, "\n[reached max sequence length %d]\n", cfg->seq_len);
            break;
        }
    }

    double gen_end = platform_time_ms();
    double gen_time = gen_end - gen_start;
    double total_time = gen_end - t0;

    printf("\n");
    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Generated: %d tokens\n", n_generated);
    if (n_generated > 0) {
        fprintf(stderr, "Speed: %.2f tok/s (%.1f ms/tok)\n",
                n_generated / (gen_time / 1000.0), gen_time / n_generated);
    }
    fprintf(stderr, "Total time: %.1f ms\n", total_time);

    // Cleanup
    free(prompt_tokens);
    tokenizer_free(&tokenizer);
    model_free(&model);
    gguf_free(gf);
    platform_unload_file(&mf);

    return 0;
}
