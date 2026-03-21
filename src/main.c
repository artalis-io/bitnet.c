#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "moe.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "threadpool.h"
#include "sh_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

// Callback for streaming token output. Return non-zero to stop generation.
typedef int (*bn_token_callback)(const char *piece, int token_id, void *user_data);

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread/qos.h>
#else
#include <unistd.h>
#endif

#if !defined(__EMSCRIPTEN__)
#include <sys/mman.h>
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
    int chat;
    int temp_set;       // whether user explicitly set --temp
    float repeat_penalty;
    int repeat_set;     // whether user explicitly set --repeat-penalty
    int no_prefill;
    int kv_f16;
    int force_pread;    // force pread for expert loading (bypass mmap)
    int cache_mb;       // expert cache budget in MB (default 2048, 0 to disable)
    int cache_mb_set;   // whether user explicitly set --cache-mb
    int force_madvise;  // madvise-guided mmap for low-RSS expert streaming
    const char *draft_path; // --draft <model.gguf> for speculative decoding
    int draft_k;        // --draft-k: number of draft tokens (default 5)
    int threads;        // 0 = auto-detect
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
    fprintf(stderr, "  --chat          Interactive chat REPL mode\n");
    fprintf(stderr, "  --repeat-penalty <float>  Repetition penalty (default: 1.1)\n");
    fprintf(stderr, "  --kv16          Store KV cache in FP16 (halves attention DRAM bandwidth)\n");
    fprintf(stderr, "  --no-prefill    Disable batch prompt prefill (compute logits for every token)\n");
    fprintf(stderr, "  --pread         Force pread for MoE expert loading (measure SSD streaming speed)\n");
    fprintf(stderr, "  --cache-mb <int>  Expert cache budget in MB (default: 4096, 0 to disable)\n");
    fprintf(stderr, "  --madvise         madvise-guided mmap for MoE (low RSS, mmap speed)\n");
    fprintf(stderr, "  --draft <path>  Draft model for speculative decoding\n");
    fprintf(stderr, "  --draft-k <int> Draft tokens per iteration (default: 5)\n");
    fprintf(stderr, "  -t <int>        Number of threads (default: auto-detect)\n");
}

static int parse_int(const char *s, const char *name) {
    char *end;
    long val = strtol(s, &end, 10);
    if (*end != '\0' || val < INT_MIN || val > INT_MAX) {
        fprintf(stderr, "Invalid value for %s: %s\n", name, s);
        exit(1);
    }
    return (int)val;
}

static float parse_float(const char *s, const char *name) {
    char *end;
    float val = strtof(s, &end);
    if (*end != '\0') {
        fprintf(stderr, "Invalid value for %s: %s\n", name, s);
        exit(1);
    }
    return val;
}

static CLIArgs parse_args(int argc, char **argv) {
    CLIArgs args = {0};
    args.prompt = "Hello";
    args.n_tokens = 256;
    args.temperature = 0.0f;
    args.topp = 0.9f;
    args.repeat_penalty = 1.1f;
    args.seed = 42;
    args.max_seq_len = 0;
    args.cache_mb = 4096;
    args.draft_k = 5;

    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }

    args.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            args.n_tokens = parse_int(argv[++i], "-n");
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            args.temperature = parse_float(argv[++i], "--temp");
            args.temp_set = 1;
        } else if (strcmp(argv[i], "--topp") == 0 && i + 1 < argc) {
            args.topp = parse_float(argv[++i], "--topp");
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            char *end;
            args.seed = (uint64_t)strtoull(argv[++i], &end, 10);
            if (*end != '\0') { fprintf(stderr, "Invalid value for --seed: %s\n", argv[i]); exit(1); }
        } else if (strcmp(argv[i], "--maxseq") == 0 && i + 1 < argc) {
            args.max_seq_len = parse_int(argv[++i], "--maxseq");
        } else if (strcmp(argv[i], "--flash") == 0) {
            args.flash_attn = 1;
        } else if (strcmp(argv[i], "--chat") == 0) {
            args.chat = 1;
        } else if (strcmp(argv[i], "--kv16") == 0) {
            args.kv_f16 = 1;
        } else if (strcmp(argv[i], "--no-prefill") == 0) {
            args.no_prefill = 1;
        } else if (strcmp(argv[i], "--pread") == 0) {
            args.force_pread = 1;
        } else if (strcmp(argv[i], "--madvise") == 0) {
            args.force_madvise = 1;
        } else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            args.draft_path = argv[++i];
        } else if (strcmp(argv[i], "--draft-k") == 0 && i + 1 < argc) {
            args.draft_k = parse_int(argv[++i], "--draft-k");
        } else if (strcmp(argv[i], "--cache-mb") == 0 && i + 1 < argc) {
            args.cache_mb = parse_int(argv[++i], "--cache-mb");
            args.cache_mb_set = 1;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            args.threads = parse_int(argv[++i], "-t");
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            args.repeat_penalty = parse_float(argv[++i], "--repeat-penalty");
            args.repeat_set = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    return args;
}

// Loop detection constants
#define LOOP_BUF_SIZE 32
#define LOOP_NGRAM    4

// Generate tokens with callback-based streaming.
// Returns: n_generated, -1 on loop, -2 on error.
static int generate_response(BnModel *model, BnTokenizer *tok, BnSampler *sampler,
                              int max_tokens, int *pos,
                              bn_token_callback cb, void *user_data) {
    int loop_buf[LOOP_BUF_SIZE];
    int loop_idx = 0, gen_count = 0;
    memset(loop_buf, -1, sizeof(loop_buf));

    float *logits = model->state.logits;
    if (!logits) return -2;

    for (int i = 0; i < max_tokens; i++) {
        int next = bn_sampler_sample(sampler, logits);

        if (next == tok->eot_id || next == tok->eos_id ||
            next == tok->im_end_id)
            break;

        // Ring buffer loop detection
        loop_buf[loop_idx] = next;
        loop_idx = (loop_idx + 1) % LOOP_BUF_SIZE;
        gen_count++;

        if (gen_count >= 2 * LOOP_NGRAM) {
            int looping = 1;
            for (int k = 0; k < LOOP_NGRAM; k++) {
                int a = loop_buf[((loop_idx - 1 - k) % LOOP_BUF_SIZE + LOOP_BUF_SIZE) % LOOP_BUF_SIZE];
                int b = loop_buf[((loop_idx - 1 - k - LOOP_NGRAM) % LOOP_BUF_SIZE + LOOP_BUF_SIZE) % LOOP_BUF_SIZE];
                if (a != b) { looping = 0; break; }
            }
            if (looping) return -1;
        }

        bn_sampler_accept(sampler, next);

        const char *piece = bn_tokenizer_decode(tok, next);
        if (piece && cb) {
            if (cb(piece, next, user_data))
                break;
        }

        logits = bn_transformer_forward(model, next, *pos);
        (*pos)++;
        if (!logits) return -2;
    }

    return gen_count;
}

// Speculative decoding: draft K tokens with small model, verify with target.
// Greedy only (temperature=0). Returns n_generated, -1 on loop, -2 on error.
static int generate_response_speculative(
    BnModel *target, BnModel *draft, int draft_k,
    BnTokenizer *tok, BnSampler *sampler,
    int max_tokens, int *pos,
    bn_token_callback cb, void *user_data)
{
    int gen_count = 0;
    int draft_tokens[20];  // max draft_k = 20
    if (draft_k > 20) draft_k = 20;
    int n_accepted_total = 0, n_drafted_total = 0;

    // Both models should have logits ready from the last prompt token
    float *target_logits = target->state.logits;
    float *draft_logits = draft->state.logits;
    if (!target_logits || !draft_logits) return -2;

    while (gen_count < max_tokens) {
        // --- Draft phase: generate K tokens with draft model ---
        int k_actual = 0;
        for (int i = 0; i < draft_k && gen_count + i < max_tokens; i++) {
            int d = bn_sampler_sample(sampler, draft_logits);  // greedy argmax
            if (d == tok->eot_id || d == tok->eos_id || d == tok->im_end_id)
                break;
            draft_tokens[k_actual++] = d;
            draft_logits = bn_transformer_forward(draft, d, *pos + k_actual - 1);
            if (!draft_logits) return -2;
        }

        if (k_actual == 0) {
            // Draft immediately hit EOS — sample from target instead
            int t = bn_sampler_sample(sampler, target_logits);
            if (t == tok->eot_id || t == tok->eos_id || t == tok->im_end_id)
                break;
            bn_sampler_accept(sampler, t);
            const char *piece = bn_tokenizer_decode(tok, t);
            if (piece && cb && cb(piece, t, user_data)) break;
            target_logits = bn_transformer_forward(target, t, *pos);
            draft_logits = bn_transformer_forward(draft, t, *pos);
            (*pos)++;
            gen_count++;
            if (!target_logits || !draft_logits) return -2;
            continue;
        }

        n_drafted_total += k_actual;

        // --- Verify phase: batch prefill all draft tokens through target ---
        // prefill_all runs all k_actual tokens in one pass, returns logits at every position.
        // Logits at position i tell us the target's prediction AFTER seeing draft_tokens[0..i-1].
        // We compare argmax(logits[i]) with draft_tokens[i].
        int vocab_size = target->config.vocab_size;
        float *verify_logits = (float *)malloc((size_t)(k_actual + 1) * vocab_size * sizeof(float));
        if (!verify_logits) return -2;

        // First position: we already have target_logits from the previous iteration
        memcpy(verify_logits, target_logits, vocab_size * sizeof(float));

        // Batch the k_actual draft tokens through target (fills KV cache for all positions)
        if (bn_transformer_prefill_all(target, draft_tokens, k_actual, *pos,
                                        verify_logits + vocab_size) != 0) {
            free(verify_logits);
            return -2;
        }
        // Now verify_logits[0] = logits before draft_tokens[0] (= previous target_logits)
        // verify_logits[(i+1)*vocab] = logits after processing draft_tokens[0..i]

        int n_accepted = 0;
        int corrected = -1;
        for (int i = 0; i < k_actual; i++) {
            float *lg = verify_logits + (size_t)i * vocab_size;
            int t_i = bn_sampler_sample(sampler, lg);  // target's greedy choice at position i
            if (t_i == draft_tokens[i]) {
                n_accepted++;
            } else {
                corrected = t_i;
                break;
            }
        }
        // target_logits now points to logits after last accepted token
        target_logits = target->state.logits;  // prefill_all left last logits here
        free(verify_logits);

        // Stream accepted draft tokens
        for (int i = 0; i < n_accepted; i++) {
            bn_sampler_accept(sampler, draft_tokens[i]);
            const char *piece = bn_tokenizer_decode(tok, draft_tokens[i]);
            if (piece && cb && cb(piece, draft_tokens[i], user_data)) goto done;
            gen_count++;
        }

        if (corrected >= 0) {
            // Stream the corrected token
            if (corrected != tok->eot_id && corrected != tok->eos_id &&
                corrected != tok->im_end_id) {
                bn_sampler_accept(sampler, corrected);
                const char *piece = bn_tokenizer_decode(tok, corrected);
                if (piece && cb && cb(piece, corrected, user_data)) goto done;
                gen_count++;
            } else {
                // Target says stop
                *pos += n_accepted;
                break;
            }
            *pos += n_accepted + 1;

            // Re-sync target: prefill filled KV with wrong token at rejection pos.
            // Overwrite by running corrected token through target at that position.
            target_logits = bn_transformer_forward(target, corrected, *pos - 1);
            if (!target_logits) return -2;

            // Re-sync draft model
            draft_logits = bn_transformer_forward(draft, corrected, *pos - 1);
            if (!draft_logits) return -2;
        } else {
            // All K accepted — bonus token from target's last logits
            *pos += n_accepted;
            int bonus = bn_sampler_sample(sampler, target_logits);
            if (bonus != tok->eot_id && bonus != tok->eos_id &&
                bonus != tok->im_end_id) {
                bn_sampler_accept(sampler, bonus);
                const char *piece = bn_tokenizer_decode(tok, bonus);
                if (piece && cb && cb(piece, bonus, user_data)) goto done;
                gen_count++;
                target_logits = bn_transformer_forward(target, bonus, *pos);
                draft_logits = bn_transformer_forward(draft, bonus, *pos);
                (*pos)++;
                if (!target_logits || !draft_logits) return -2;
            } else {
                break;  // target says stop
            }
        }

        n_accepted_total += n_accepted + (corrected >= 0 ? 1 : 1); // +1 for corrected or bonus
    }

done:
    if (n_drafted_total > 0) {
        char acc_s[32], rate_s[32];
        snprintf(acc_s, sizeof(acc_s), "%d/%d", n_accepted_total, n_drafted_total);
        snprintf(rate_s, sizeof(rate_s), "%.1f%%",
                 100.0 * n_accepted_total / n_drafted_total);
        SH_LOG_INFO("Speculative decoding", "accepted", acc_s, "rate", rate_s);
    }
    return gen_count;
}

static int print_token(const char *piece, int token_id, void *user_data) {
    (void)token_id;
    (void)user_data;
    printf("%s", piece);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    sh_log_init(NULL);
    CLIArgs args = parse_args(argc, argv);

    // Determine thread count: -t flag > Apple P-core detect > sysconf > 1
    int n_workers = 0;
    if (args.threads > 0) {
        n_workers = args.threads - 1;  // main thread counts as one
    } else {
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        {
            int ncores = 0;
            size_t len = sizeof(ncores);
            if (sysctlbyname("hw.perflevel0.logicalcpu", &ncores, &len, NULL, 0) == 0 && ncores > 1)
                n_workers = ncores - 1;
        }
#else
        long ncores = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncores > 1) n_workers = (int)ncores - 1;
#endif
    }

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
    if (bn_model_load(&model, gf, args.max_seq_len, args.kv_f16) != 0) {
        SH_LOG_ERROR("Failed to load model");
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    model.file = mf;  // keep mmap alive
    model.config.flash_attn = args.flash_attn;

    // Set expert I/O for MoE: prefer mmap, fallback to pread
    if (model.moe_state) {
        if (!args.force_pread && mf.is_mmap == 1 && mf.data) {
            model.moe_state->io.mmap_base = mf.data;
        }
        if (mf.fd >= 0) {
            model.moe_state->io.fd = mf.fd;
            model.expert_fd = mf.fd;
        }

        // madvise-guided mmap: use WILLNEED prefetch hints
        if (args.force_madvise && args.force_pread) {
            SH_LOG_WARN("--madvise and --pread are mutually exclusive, ignoring --madvise");
        } else if (args.force_madvise && !model.moe_state->io.mmap_base) {
            SH_LOG_WARN("--madvise requires mmap (file not mmap'd), falling back to pread");
        } else if (args.force_madvise && model.moe_state->io.mmap_base) {
#if !defined(__EMSCRIPTEN__)
            // Only suppress readahead — don't evict. Let page cache manage eviction.
            model.moe_state->io.madvise_mode = 1;
            SH_LOG_INFO("Expert I/O mode", "mode", "madvise");
#endif
        } else if (args.force_pread) {
            SH_LOG_INFO("Expert I/O mode", "mode", "pread (forced)");
        } else if (model.moe_state->io.mmap_base) {
            SH_LOG_INFO("Expert I/O mode", "mode", "mmap");
        }

        // Create I/O prefetch thread for pread pipeline (not needed for madvise)
        if (!model.moe_state->io.madvise_mode)
            bn_moe_prefetch_create(model.moe_state);

        // Create expert LRU cache (pread only, not needed for madvise)
        if (!model.moe_state->io.madvise_mode &&
            args.cache_mb > 0 && !model.moe_state->io.mmap_base && model.moe_state->io.fd >= 0
            && model.config.n_layers > 0) {
            BnMoEExpertMap *em = &model.weights.layers[0].expert_map;
            model.moe_state->io.cache = bn_moe_cache_create(
                (size_t)args.cache_mb * 1024 * 1024,
                em->expert_gate_bytes, em->expert_up_bytes, em->expert_down_bytes);
        }
    }

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
        return 1;
    }
    {
        char vs[16]; snprintf(vs, sizeof(vs), "%d", tokenizer.vocab_size);
        SH_LOG_INFO("Tokenizer loaded", "tokens", vs);
    }

    // Apply chat-mode sampling defaults if user didn't explicitly set temp
    if (args.chat && !args.temp_set) {
        args.temperature = 0.5f;
        args.topp = 0.9f;
    }
    // Initialize sampler
    BnSampler sampler;
    if (bn_sampler_init(&sampler, cfg->vocab_size, args.temperature, args.topp, args.seed) != 0) {
        fprintf(stderr, "Failed to allocate sampler\n");
        bn_tokenizer_free(&tokenizer);
        bn_model_free(&model);
        bn_gguf_free(gf);
        return 1;
    }
    if (args.repeat_penalty > 1.0f)
        bn_sampler_set_repeat_penalty(&sampler, args.repeat_penalty, 64);

    // Load draft model for speculative decoding
    BnModel draft_model = {0};
    BnGGUFFile *draft_gf = NULL;
    BnMappedFile draft_mf = {0};
    int has_draft = 0;
    if (args.draft_path) {
        SH_LOG_INFO("Loading draft model", "path", args.draft_path);
        draft_mf = bn_platform_load_file(args.draft_path);
        if (!draft_mf.data) {
            SH_LOG_ERROR("Failed to load draft model file");
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }
        draft_gf = bn_gguf_open(draft_mf.data, draft_mf.size);
        if (!draft_gf) {
            SH_LOG_ERROR("Failed to parse draft GGUF");
            bn_platform_unload_file(&draft_mf);
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }
        if (bn_model_load(&draft_model, draft_gf, args.max_seq_len, args.kv_f16) != 0) {
            SH_LOG_ERROR("Failed to load draft model");
            bn_gguf_free(draft_gf);
            bn_platform_unload_file(&draft_mf);
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }
        draft_model.file = draft_mf;
        draft_model.config.flash_attn = args.flash_attn;
        // Share thread pool (never run concurrently)
        draft_model.pool = model.pool;

        // Validate vocab compatibility
        if (draft_model.config.vocab_size != cfg->vocab_size) {
            SH_LOG_ERROR("Draft model vocab size mismatch");
            draft_model.pool = NULL;  // don't double-free pool
            bn_model_free(&draft_model);
            bn_gguf_free(draft_gf);
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }

        has_draft = 1;
        {
            char dd[16], dl[16];
            snprintf(dd, sizeof(dd), "%d", draft_model.config.dim);
            snprintf(dl, sizeof(dl), "%d", draft_model.config.n_layers);
            SH_LOG_INFO("Draft model loaded", "dim", dd, "layers", dl,
                         "draft_k", args.draft_k > 9 ? "10+" : "5");
        }
    }

    if (args.chat) {
        // --- Chat REPL mode ---
        int max_tokens = 4096 + 64;  // generous buffer for encoded turns
        int *tokens = (int *)malloc(max_tokens * sizeof(int));
        if (!tokens) {
            SH_LOG_ERROR("Failed to allocate token buffer");
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }

        int pos = 0;
        int seq_len = cfg->seq_len;

        // Feed BOS at pos=0 (skip if model says not to)
        if (tokenizer.add_bos) {
            bn_transformer_forward(&model, tokenizer.bos_id, pos);
            pos++;
        }

        char line[4096];
        while (1) {
            printf("> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;

            // Strip trailing newline
            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = '\0';

            if (len == 0) continue;
            if (strcmp(line, "/quit") == 0) break;
            if (strcmp(line, "/reset") == 0) {
                pos = 0;
                if (tokenizer.add_bos) {
                    bn_transformer_forward(&model, tokenizer.bos_id, pos);
                    pos++;
                }
                bn_sampler_reset_recent(&sampler);
                printf("[conversation reset]\n");
                continue;
            }

            int n = 0;
            if (tokenizer.chatml) {
                // ChatML: <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
                if (n < max_tokens) tokens[n++] = tokenizer.im_start_id;
                char user_buf[4096 + 16];
                snprintf(user_buf, sizeof(user_buf), "user\n%s", line);
                n += bn_tokenizer_encode(&tokenizer, user_buf, 0,
                                         tokens + n, max_tokens - n);
                if (n < max_tokens) tokens[n++] = tokenizer.im_end_id;
                int n2 = bn_tokenizer_encode(&tokenizer, "\n", 0,
                                             tokens + n, max_tokens - n);
                n += n2;
                if (n < max_tokens) tokens[n++] = tokenizer.im_start_id;
                n2 = bn_tokenizer_encode(&tokenizer, "assistant\n", 0,
                                         tokens + n, max_tokens - n);
                n += n2;
            } else {
                // LLaMA/BitNet: User: {msg}<|eot_id|>Assistant:
                char user_buf[4096 + 16];
                snprintf(user_buf, sizeof(user_buf), "User: %s", line);
                n = bn_tokenizer_encode(&tokenizer, user_buf, 0, tokens, max_tokens);
                if (n < max_tokens && tokenizer.eot_id >= 0)
                    tokens[n++] = tokenizer.eot_id;
                int n2 = bn_tokenizer_encode(&tokenizer, "Assistant: ", 0,
                                             tokens + n, max_tokens - n);
                n += n2;
            }

            // Context overflow: reset if prompt won't fit
            if (pos + n > seq_len) {
                printf("[context full — resetting]\n");
                pos = 0;
                if (tokenizer.add_bos) {
                    bn_transformer_forward(&model, tokenizer.bos_id, pos);
                    pos++;
                }
                bn_sampler_reset_recent(&sampler);
            }

            // Feed prompt tokens through forward pass
            float *logits = NULL;
            if (!args.no_prefill && n > 1) {
                logits = bn_transformer_prefill(&model, tokens, n, pos);
                pos += n;
            } else {
                for (int i = 0; i < n; i++) {
                    logits = bn_transformer_forward(&model, tokens[i], pos);
                    pos++;
                }
            }

            if (!logits) {
                SH_LOG_ERROR("Forward pass returned NULL during prompt");
                break;
            }

            // Cap generation to remaining context
            int max_gen = args.n_tokens;
            int remaining = seq_len - pos - 1;
            if (remaining < max_gen) max_gen = remaining;
            if (max_gen < 1) max_gen = 1;

            int gen_count = generate_response(&model, &tokenizer, &sampler,
                                               max_gen, &pos,
                                               print_token, NULL);

            // Feed end-of-turn token into KV cache to close the assistant turn
            int turn_end_id = tokenizer.chatml ? tokenizer.im_end_id : tokenizer.eot_id;
            if (turn_end_id >= 0 && pos < seq_len) {
                bn_transformer_forward(&model, turn_end_id, pos);
                pos++;
            }

            printf("\n");

            if (gen_count < 0) {
                printf("[loop detected]\n");
            }

            // Context usage indicator when >75% full
            if (pos * 4 > seq_len * 3) {
                printf("[%d/%d]\n", pos, seq_len);
            }
        }

        free(tokens);
    } else {
        // --- Single-shot generation ---
        // #30: Encode prompt -- upper bound is 1 token per byte + BOS + margin
        int max_prompt_tokens = (int)strlen(args.prompt) + 3;
        int *prompt_tokens = (int *)malloc(max_prompt_tokens * sizeof(int));
        if (!prompt_tokens) {
            SH_LOG_ERROR("Failed to allocate prompt token buffer");
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }
        int n_prompt = bn_tokenizer_encode(&tokenizer, args.prompt, tokenizer.add_bos,
                                        prompt_tokens, max_prompt_tokens);
        {
            char np[16]; snprintf(np, sizeof(np), "%d", n_prompt);
            SH_LOG_INFO("Prompt encoded", "n_tokens", np);
        }
#ifdef DEBUG
        fprintf(stderr, "DBG tokens:");
        for (int i = 0; i < n_prompt; i++) fprintf(stderr, " %d", prompt_tokens[i]);
        fprintf(stderr, "\n");
#endif

        // Generation loop
        {
            char nt[16]; snprintf(nt, sizeof(nt), "%d", args.n_tokens);
            SH_LOG_INFO("Starting generation", "n_tokens", nt);
        }
        double gen_start = bn_platform_time_ms();
        int pos = 0;
        int n_generated = 0;
        float *logits;

        // Prefill prompt tokens (skip logits for intermediate tokens)
        if (!args.no_prefill && n_prompt > 1) {
            logits = bn_transformer_prefill(&model, prompt_tokens, n_prompt, 0);
            pos = n_prompt;
        } else {
            logits = NULL;
            for (int i = 0; i < n_prompt; i++) {
                logits = bn_transformer_forward(&model, prompt_tokens[i], i);
            }
            pos = n_prompt;
        }
        // Also prefill draft model if speculative decoding
        if (has_draft && logits) {
            if (!args.no_prefill && n_prompt > 1) {
                bn_transformer_prefill(&draft_model, prompt_tokens, n_prompt, 0);
            } else {
                for (int i = 0; i < n_prompt; i++)
                    bn_transformer_forward(&draft_model, prompt_tokens[i], i);
            }
        }

        if (!logits) {
            SH_LOG_ERROR("Forward pass returned NULL during prompt");
        } else {
#ifdef DEBUG
            // Dump top-10 logits after prefill
            {
                int top[10];
                for (int k = 0; k < 10; k++) top[k] = 0;
                for (int v = 0; v < cfg->vocab_size; v++) {
                    for (int k = 0; k < 10; k++) {
                        if (logits[v] > logits[top[k]]) {
                            for (int j = 9; j > k; j--) top[j] = top[j-1];
                            top[k] = v;
                            break;
                        }
                    }
                }
                fprintf(stderr, "DBG top10 after prefill:\n");
                for (int k = 0; k < 10; k++)
                    fprintf(stderr, "  token %6d: %.4f\n", top[k], logits[top[k]]);
            }
#endif
            // Generate tokens — speculative or standard
            if (has_draft) {
                n_generated = generate_response_speculative(
                    &model, &draft_model, args.draft_k,
                    &tokenizer, &sampler, args.n_tokens, &pos,
                    print_token, NULL);
                if (n_generated < 0)
                    SH_LOG_ERROR("Speculative generation failed");
            } else
            for (int i = 0; i < args.n_tokens; i++) {
                int next = bn_sampler_sample(&sampler, logits);
                n_generated++;

                // Check for EOS/EOT
                if (next == tokenizer.eos_id || next == tokenizer.eot_id) {
                    break;
                }

                bn_sampler_accept(&sampler, next);

                const char *piece = bn_tokenizer_decode(&tokenizer, next);
                if (!piece) piece = "";
                printf("%s", piece);
                fflush(stdout);

                logits = bn_transformer_forward(&model, next, pos);
                pos++;
                if (!logits) {
                    SH_LOG_ERROR("Forward pass returned NULL");
                    break;
                }
#ifdef DEBUG
                {
                    int top5[5] = {0};
                    for (int v = 0; v < cfg->vocab_size; v++) {
                        for (int k = 0; k < 5; k++) {
                            if (logits[v] > logits[top5[k]]) {
                                for (int j = 4; j > k; j--) top5[j] = top5[j-1];
                                top5[k] = v;
                                break;
                            }
                        }
                    }
                    fprintf(stderr, "DBG gen %d (token %d, pos %d) top5:", i, next, pos);
                    for (int k = 0; k < 5; k++)
                        fprintf(stderr, " %d(%.3f)", top5[k], logits[top5[k]]);
                    fprintf(stderr, "\n");
                }
#endif
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

        // Print MoE stats if applicable
        if (model.moe_state)
            bn_moe_print_stats(model.moe_state, n_generated + n_prompt);

        free(prompt_tokens);
    }

    // Cleanup
    if (has_draft) {
        draft_model.pool = NULL;  // shared, don't double-free
        bn_model_free(&draft_model);
        bn_gguf_free(draft_gf);
    }
    if (model.moe_state && model.moe_state->io.cache) {
        bn_moe_cache_free(model.moe_state->io.cache);
        model.moe_state->io.cache = NULL;
    }
    bn_sampler_free(&sampler);
    bn_tokenizer_free(&tokenizer);
    bn_model_free(&model);  // also unloads mmap'd file
    bn_gguf_free(gf);

    return 0;
}
