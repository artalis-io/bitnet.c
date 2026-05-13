#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "moe.h"
#include "generate.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "threadpool.h"
#include "session.h"
#include "prompt_cache.h"
#include "sh_log.h"
#ifdef BN_ENABLE_WEBGPU
#include "gpu_wgpu.h"
#endif
#if defined(BN_ENABLE_WEBGPU) || defined(BN_ENABLE_METAL)
#include "gpu_moe_cache.h"
#endif
#ifdef BN_ENABLE_METAL
#include "gpu_metal.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread/qos.h>
#endif
#include <unistd.h>

typedef struct {
    const char *model_path;
    const char *prompt;
    int n_tokens;
    float temperature;
    float topp;
    uint64_t seed;
    int max_seq_len;
    int max_seq_len_set;
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
    int prefault_moe;   // touch all mmap'd MoE expert pages before generation
    int quiet;          // suppress generated token output
    int token_ids;      // print generated token IDs to stderr
    const char *draft_path; // --draft <model.gguf> for speculative decoding
    int draft_k;        // --draft-k: number of draft tokens (default 5)
    int threads;        // 0 = auto-detect
    int webgpu;         // use WebGPU backend for matvec (requires BN_ENABLE_WEBGPU)
    int metal;          // use Metal backend (requires BN_ENABLE_METAL)
    int gpu_cpu_logits; // hidden diagnostic: run final logits on CPU
    int gpu_compare_logits; // hidden diagnostic: compare GPU logits to CPU
    const char *shader_dir; // --shader-dir for WebGPU WGSL shaders
    const char *metal_shader_dir; // --metal-shader-dir for Metal shaders
    int kv_tq_bits;     // TurboQuant KV compression (0=disabled, 2-4=bits)
    int gpu_cache_mb;   // GPU expert buffer cache in MB (default 4096, 0 to disable)
} CLIArgs;

#define BN_GPU_DEFAULT_MAXSEQ 4096

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.gguf> [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -p <prompt>     Input prompt (default: \"Hello\")\n");
    fprintf(stderr, "  -n <int>        Number of tokens to generate (default: 256)\n");
    fprintf(stderr, "  --temp <float>  Temperature (default: 0.0 = greedy)\n");
    fprintf(stderr, "  --topp <float>  Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --seed <int>    Random seed (default: 42)\n");
    fprintf(stderr, "  --maxseq <int>  Max sequence length (default: model max; GPU auto-caps large contexts to 4096)\n");
    fprintf(stderr, "  --flash         Use flash attention (online softmax)\n");
    fprintf(stderr, "  --chat          Interactive chat REPL mode\n");
    fprintf(stderr, "  --repeat-penalty <float>  Repetition penalty (default: 1.1)\n");
    fprintf(stderr, "  --kv16          Store KV cache in FP16 (halves attention DRAM bandwidth)\n");
    fprintf(stderr, "  --kv-tq <bits>  TurboQuant KV compression (2, 3, or 4 bits)\n");
    fprintf(stderr, "  --no-prefill    Disable batch prompt prefill (compute logits for every token)\n");
    fprintf(stderr, "  --pread         Force pread for MoE expert loading (measure SSD streaming speed)\n");
    fprintf(stderr, "  --cache-mb <int>  Expert cache budget in MB (default: 4096, 0 to disable)\n");
    fprintf(stderr, "  --gpu-cache-mb <int>  GPU expert buffer cache in MB (default: 4096, 0 to disable)\n");
    fprintf(stderr, "  --madvise         madvise-guided mmap for MoE (low RSS, mmap speed)\n");
    fprintf(stderr, "  --prefault-moe    Fault all mmap'd MoE expert pages during startup\n");
    fprintf(stderr, "  --quiet           Suppress generated token output\n");
    fprintf(stderr, "  --token-ids       Print generated token IDs to stderr\n");
    fprintf(stderr, "  --draft <path>  Draft model for speculative decoding\n");
    fprintf(stderr, "  --draft-k <int> Draft tokens per iteration (default: 5)\n");
    fprintf(stderr, "  --webgpu        Enable WebGPU backend (requires BN_ENABLE_WEBGPU=1)\n");
    fprintf(stderr, "  --metal         Enable Metal backend (requires BN_ENABLE_METAL=1)\n");
    fprintf(stderr, "  --shader-dir <path>        WebGPU shader directory (default: shaders/)\n");
    fprintf(stderr, "  --metal-shader-dir <path>  Metal shader directory (default: shaders/metal/)\n");
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
    args.gpu_cache_mb = 4096;
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
            args.max_seq_len_set = 1;
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
        } else if (strcmp(argv[i], "--prefault-moe") == 0) {
            args.prefault_moe = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            args.quiet = 1;
        } else if (strcmp(argv[i], "--token-ids") == 0) {
            args.token_ids = 1;
        } else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            args.draft_path = argv[++i];
        } else if (strcmp(argv[i], "--draft-k") == 0 && i + 1 < argc) {
            args.draft_k = parse_int(argv[++i], "--draft-k");
        } else if (strcmp(argv[i], "--cache-mb") == 0 && i + 1 < argc) {
            args.cache_mb = parse_int(argv[++i], "--cache-mb");
            args.cache_mb_set = 1;
        } else if (strcmp(argv[i], "--gpu-cache-mb") == 0 && i + 1 < argc) {
            args.gpu_cache_mb = parse_int(argv[++i], "--gpu-cache-mb");
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            args.threads = parse_int(argv[++i], "-t");
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            args.repeat_penalty = parse_float(argv[++i], "--repeat-penalty");
            args.repeat_set = 1;
        } else if (strcmp(argv[i], "--kv-tq") == 0 && i + 1 < argc) {
            args.kv_tq_bits = parse_int(argv[++i], "--kv-tq");
        } else if (strcmp(argv[i], "--webgpu") == 0) {
            args.webgpu = 1;
        } else if (strcmp(argv[i], "--metal") == 0) {
            args.metal = 1;
        } else if (strcmp(argv[i], "--gpu-cpu-logits") == 0) {
            args.gpu_cpu_logits = 1;
        } else if (strcmp(argv[i], "--gpu-compare-logits") == 0) {
            args.gpu_compare_logits = 1;
        } else if (strcmp(argv[i], "--shader-dir") == 0 && i + 1 < argc) {
            args.shader_dir = argv[++i];
        } else if (strcmp(argv[i], "--metal-shader-dir") == 0 && i + 1 < argc) {
            args.metal_shader_dir = argv[++i];
        } else if (strcmp(argv[i], "--gpu-cpu-fallback-layer") == 0 && i + 1 < argc) {
            char layer_env[32];
            snprintf(layer_env, sizeof(layer_env), "%d",
                     parse_int(argv[++i], "--gpu-cpu-fallback-layer"));
            setenv("BN_GPU_CPU_FALLBACK_LAYER", layer_env, 1);
        } else if (strcmp(argv[i], "--gpu-cpu-fallback-from-layer") == 0 && i + 1 < argc) {
            char layer_env[32];
            snprintf(layer_env, sizeof(layer_env), "%d",
                     parse_int(argv[++i], "--gpu-cpu-fallback-from-layer"));
            setenv("BN_GPU_CPU_FALLBACK_FROM_LAYER", layer_env, 1);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    return args;
}

static int gguf_get_arch_u32(BnGGUFFile *gf, const char *suffix) {
    const char *arch = bn_gguf_get_str(gf, "general.architecture");
    char prefix[64] = "llama";
    char key[128];
    if (arch) {
        snprintf(prefix, sizeof(prefix), "%s", arch);
    }
    snprintf(key, sizeof(key), "%s.%s", prefix, suffix);
    return (int)bn_gguf_get_u32(gf, key);
}

// Chat history tracker for prompt caching
typedef struct {
    int *history;
    int history_len;
    int history_cap;
    int token_ids;
} BnChatHistory;

static int print_token(const char *piece, int token_id, void *user_data) {
    // Track generated tokens in history for prompt cache
    BnChatHistory *h = (BnChatHistory *)user_data;
    if (h && h->history && h->history_len < h->history_cap) {
        h->history[h->history_len++] = token_id;
    }
    if (h && h->token_ids) {
        fprintf(stderr, "token_id=%d\n", token_id);
    }
    printf("%s", piece);
    if (user_data || isatty(STDOUT_FILENO))
        fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    sh_log_init(NULL);
    CLIArgs args = parse_args(argc, argv);

    // Validate --kv-tq
    if (args.kv_tq_bits > 0) {
        if (args.kv_tq_bits < 2 || args.kv_tq_bits > 4) {
            fprintf(stderr, "--kv-tq must be 2, 3, or 4\n");
            return 1;
        }
        if (args.kv_f16) {
            fprintf(stderr, "--kv-tq and --kv16 are mutually exclusive\n");
            return 1;
        }
        if (args.webgpu || args.metal) {
            fprintf(stderr, "--kv-tq and --webgpu/--metal are mutually exclusive (GPU TQ not yet supported)\n");
            return 1;
        }
    }

    // Validate --webgpu and --metal mutual exclusion
    if (args.webgpu && args.metal) {
        fprintf(stderr, "--webgpu and --metal are mutually exclusive\n");
        return 1;
    }

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
        /* Use physical core count, not logical (hyperthreads hurt on
         * memory-bandwidth-bound workloads). Parse /proc/cpuinfo for
         * unique core IDs; fall back to sysconf logical count / 2. */
        int phys_cores = 0;
        {
            FILE *fp = fopen("/proc/cpuinfo", "r");
            if (fp) {
                char buf[256];
                int seen[1024];
                int n_seen = 0;
                while (fgets(buf, sizeof(buf), fp)) {
                    int id;
                    if (sscanf(buf, "core id : %d", &id) == 1) {
                        int dup = 0;
                        for (int i = 0; i < n_seen; i++)
                            if (seen[i] == id) { dup = 1; break; }
                        if (!dup && n_seen < 1024) seen[n_seen++] = id;
                    }
                }
                fclose(fp);
                if (n_seen > 0) phys_cores = n_seen;
            }
            if (phys_cores == 0) {
                long logical = sysconf(_SC_NPROCESSORS_ONLN);
                phys_cores = (logical > 1) ? (int)(logical / 2) : 1;
            }
        }
        /* LLM inference is DRAM-bandwidth-bound: using all cores causes
         * memory bus contention. Half the physical cores is the sweet spot
         * for DDR4/DDR5 systems (verified: 16-core Ryzen peaks at 8 threads). */
        int target = phys_cores / 2;
        if (target < 4) target = (phys_cores > 4) ? 4 : phys_cores;
        if (target > 1) n_workers = target - 1;
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

    if ((args.webgpu || args.metal) && !args.max_seq_len_set) {
        int model_seq_len = gguf_get_arch_u32(gf, "context_length");
        int n_experts = gguf_get_arch_u32(gf, "expert_count");
        if ((args.webgpu || (args.metal && n_experts > 0)) &&
            model_seq_len > BN_GPU_DEFAULT_MAXSEQ) {
            args.max_seq_len = BN_GPU_DEFAULT_MAXSEQ;
            SH_LOG_WARN(args.webgpu ? "Auto-capping WebGPU sequence length" :
                                      "Auto-capping Metal MoE sequence length",
                        "seq", "4096", "override", "--maxseq");
        }
    }

    // Load model
    BnModel model;
    if (bn_model_load(&model, gf, args.max_seq_len, args.kv_f16, args.kv_tq_bits) != 0) {
        SH_LOG_ERROR("Failed to load model");
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    bn_model_set_file(&model, mf);  // keep mmap/buffer alive
    model.config.flash_attn = args.flash_attn;

    // Set expert I/O for MoE: prefer mmap, fallback to pread
    if (model.config.n_experts > 0) {
        BnMoEIO *moe_io = bn_model_moe_io(&model);
        if (!args.force_pread && mf.is_mmap == 1 && mf.data) {
            bn_model_set_moe_mmap_base(&model, mf.data);
        }
        if (mf.fd >= 0) {
            bn_model_set_moe_fd(&model, mf.fd);
        }

        // madvise-guided mmap: use WILLNEED prefetch hints
        if (args.force_madvise && args.force_pread) {
            SH_LOG_WARN("--madvise and --pread are mutually exclusive, ignoring --madvise");
        } else if (args.force_madvise && !moe_io->mmap_base) {
            SH_LOG_WARN("--madvise requires mmap (file not mmap'd), falling back to pread");
        } else if (args.force_madvise && moe_io->mmap_base) {
#if !defined(__EMSCRIPTEN__)
            // Only suppress readahead — don't evict. Let page cache manage eviction.
            bn_model_set_moe_madvise(&model, 1);
            SH_LOG_INFO("Expert I/O mode", "mode", "madvise");
#endif
        } else if (args.force_pread) {
            SH_LOG_INFO("Expert I/O mode", "mode", "pread (forced)");
        } else if (moe_io->mmap_base) {
            SH_LOG_INFO("Expert I/O mode", "mode", "mmap");
        }

        if (args.prefault_moe) {
            if (args.force_pread || !moe_io->mmap_base) {
                SH_LOG_WARN("--prefault-moe requires mmap, ignoring");
            } else {
                bn_moe_prefault_mmap(&model);
            }
        }

        // Create I/O prefetch thread for pread pipeline (not needed for madvise)
        if (!moe_io->madvise_mode)
            bn_moe_prefetch_create(moe_io);

        // Create expert LRU cache (pread only, not needed for madvise)
        if (!moe_io->madvise_mode &&
            args.cache_mb > 0 && !moe_io->mmap_base && moe_io->fd >= 0
            && model.config.n_layers > 0) {
            BnMoEExpertMap *em = &model.weights.layers[0].moe.expert_map;
            bn_model_set_moe_cache(&model, bn_moe_cache_create(
                (size_t)args.cache_mb * 1024 * 1024,
                em->expert_gate_bytes, em->expert_up_bytes, em->expert_down_bytes));
        }
    }

    // Create thread pool
    bn_model_set_thread_pool(&model, bn_tp_create(n_workers), 1);
    if (bn_model_pool(&model)) {
        char nt[8];
        snprintf(nt, sizeof(nt), "%d", bn_tp_num_threads(bn_model_pool(&model)));
        SH_LOG_INFO("Thread pool created", "threads", nt);
    } else if (n_workers > 0) {
        SH_LOG_WARN("Failed to create thread pool, running single-threaded");
    }

    // WebGPU backend (optional)
#ifdef BN_ENABLE_WEBGPU
    if (args.webgpu) {
        const char *sd = args.shader_dir ? args.shader_dir : "shaders/";
        BnGPUBackend *gpu = bn_gpu_wgpu_create(sd);
        if (gpu) {
            double gpu_t0 = bn_platform_time_ms();
            if (bn_model_upload_weights(&model, gpu) == 0) {
                char ms[16];
                snprintf(ms, sizeof(ms), "%.0f", bn_platform_time_ms() - gpu_t0);
                SH_LOG_INFO("WebGPU weights uploaded", "ms", ms);
                // Initialize GPU-resident activation buffers for forward pass
                if (gpu->init_activations) {
                    if (gpu->init_activations(gpu->ctx, &model.config) == 0) {
                        SH_LOG_INFO("GPU forward pass ready");
                        // Initialize GPU slab allocator for MoE weight suballocation
                        if (model.config.n_experts > 0)
                            bn_gpu_wgpu_init_slab(gpu, (size_t)args.gpu_cache_mb);
                        // Create GPU expert buffer cache for MoE
                        if (model.config.n_experts > 0 && args.gpu_cache_mb > 0 &&
                            model.config.n_layers > 0) {
                            BnMoEExpertMap *em0 = &model.weights.layers[0].moe.expert_map;
                            size_t entry_bytes = em0->expert_gate_bytes + em0->expert_up_bytes
                                               + em0->expert_down_bytes;
                            if (entry_bytes > 0) {
                                bn_model_set_gpu_moe_cache(
                                    &model,
                                    bn_gpu_moe_cache_create(
                                        (size_t)args.gpu_cache_mb * 1024 * 1024,
                                        entry_bytes, gpu));
                            }
                        }
                    }
                }
            } else {
                SH_LOG_WARN("WebGPU weight upload failed, falling back to CPU");
                bn_gpu_wgpu_destroy(gpu);
            }
        } else {
            SH_LOG_WARN("No WebGPU adapter available, falling back to CPU");
        }
    }
#else
    if (args.webgpu) {
        SH_LOG_WARN("--webgpu requires BN_ENABLE_WEBGPU=1 build, falling back to CPU");
    }
#endif

    // Metal backend (optional)
#ifdef BN_ENABLE_METAL
    if (args.metal) {
        const char *sd = args.metal_shader_dir ? args.metal_shader_dir : "shaders/metal/";
        BnGPUBackend *gpu = bn_gpu_metal_create(sd);
        if (gpu) {
            // Zero-copy: let Metal wrap mmap'd weight data (Phase 5)
            if (mf.is_mmap && mf.data)
                bn_gpu_metal_set_mmap_range(gpu, mf.data, mf.size);
            double gpu_t0 = bn_platform_time_ms();
            if (bn_model_upload_weights(&model, gpu) == 0) {
                char ms[16];
                snprintf(ms, sizeof(ms), "%.0f", bn_platform_time_ms() - gpu_t0);
                SH_LOG_INFO("Metal weights uploaded", "ms", ms);
                if (gpu->init_activations) {
                    if (gpu->init_activations(gpu->ctx, &model.config) == 0) {
                        SH_LOG_INFO("Metal forward pass ready");
                        if (args.gpu_cpu_logits)
                            setenv("BN_GPU_CPU_LOGITS", "1", 1);
                        if (args.gpu_compare_logits)
                            setenv("BN_GPU_COMPARE_LOGITS", "1", 1);
                        if (model.config.n_experts > 0)
                            bn_gpu_metal_init_slab(gpu, (size_t)args.gpu_cache_mb);
                        if (model.config.n_experts > 0 && args.gpu_cache_mb > 0 &&
                            model.config.n_layers > 0) {
                            BnMoEExpertMap *em0 = &model.weights.layers[0].moe.expert_map;
                            size_t entry_bytes = em0->expert_gate_bytes + em0->expert_up_bytes
                                               + em0->expert_down_bytes;
                            if (entry_bytes > 0) {
                                bn_model_set_gpu_moe_cache(
                                    &model,
                                    bn_gpu_moe_cache_create(
                                        (size_t)args.gpu_cache_mb * 1024 * 1024,
                                        entry_bytes, gpu));
                            }
                        }
                    }
                }
            } else {
                SH_LOG_WARN("Metal weight upload failed, falling back to CPU");
                bn_gpu_metal_destroy(gpu);
            }
        } else {
            SH_LOG_WARN("No Metal device available, falling back to CPU");
        }
    }
#else
    if (args.metal) {
        SH_LOG_WARN("--metal requires BN_ENABLE_METAL=1 build, falling back to CPU");
    }
#endif

    BnSession *session = bn_session_create(&model, NULL);
    if (!session) {
        SH_LOG_ERROR("Failed to create session");
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
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
        if (bn_model_load(&draft_model, draft_gf, args.max_seq_len, args.kv_f16, args.kv_tq_bits) != 0) {
            SH_LOG_ERROR("Failed to load draft model");
            bn_gguf_free(draft_gf);
            bn_platform_unload_file(&draft_mf);
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
        }
        bn_model_set_file(&draft_model, draft_mf);
        draft_model.config.flash_attn = args.flash_attn;
        // Share thread pool (never run concurrently)
        bn_model_set_thread_pool(&draft_model, bn_model_pool(&model), 0);

        // Validate vocab compatibility
        if (draft_model.config.vocab_size != cfg->vocab_size) {
            SH_LOG_ERROR("Draft model vocab size mismatch");
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

    BnSession *draft_session = NULL;
    if (has_draft) {
        draft_session = bn_session_create(&draft_model, NULL);
        if (!draft_session) {
            SH_LOG_ERROR("Failed to create draft session");
            bn_model_free(&draft_model);
            bn_gguf_free(draft_gf);
            bn_session_free(session, NULL);
            bn_sampler_free(&sampler);
            bn_tokenizer_free(&tokenizer);
            bn_model_free(&model);
            bn_gguf_free(gf);
            return 1;
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

        // Prompt cache: saves KV prefix for fast /reset and context overflow recovery
        BnPromptCache *prompt_cache = bn_prompt_cache_create(0, NULL);

        int pos = 0;
        int seq_len = cfg->seq_len;

        // Token history for prompt cache keying
        int *history = (int *)malloc((size_t)seq_len * sizeof(int));
        int history_len = 0;

        // Feed BOS at pos=0 (skip if model says not to)
        if (tokenizer.add_bos) {
            bn_transformer_forward(&model, session, tokenizer.bos_id, pos);
            pos++;
            // Track in history and cache
            if (history) { history[0] = tokenizer.bos_id; history_len = 1; }
            if (prompt_cache) {
                int bos_tok = tokenizer.bos_id;
                bn_prompt_cache_store(prompt_cache, &model, session, &bos_tok, 1);
            }
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
                // Try to restore BOS state from cache (avoids re-prefilling)
                int restored = 0;
                if (prompt_cache && history && history_len > 0) {
                    restored = bn_prompt_cache_restore(prompt_cache, &model, session,
                                                        history, history_len);
                }
                if (restored > 0) {
                    pos = restored;
                    history_len = restored;
                } else {
                    bn_session_reset(session, &model);
                    pos = 0;
                    history_len = 0;
                    if (tokenizer.add_bos) {
                        bn_transformer_forward(&model, session, tokenizer.bos_id, pos);
                        pos++;
                        if (history) { history[0] = tokenizer.bos_id; history_len = 1; }
                    }
                }
                bn_sampler_reset_recent(&sampler);
                printf("[conversation reset]\n");
                continue;
            }

            // Format user message into chat turn tokens
            int n = bn_chat_format_turn(&tokenizer, BN_CHAT_AUTO, line, tokens, max_tokens, NULL);

            // Context overflow: reset and try restoring cached prefix
            if (pos + n > seq_len) {
                printf("[context full — resetting]\n");
                bn_session_reset(session, &model);
                pos = 0;
                history_len = 0;
                if (tokenizer.add_bos) {
                    bn_transformer_forward(&model, session, tokenizer.bos_id, pos);
                    pos++;
                    if (history) { history[0] = tokenizer.bos_id; history_len = 1; }
                }
                bn_sampler_reset_recent(&sampler);
            }

            // Append turn tokens to history
            if (history && history_len + n <= seq_len) {
                memcpy(history + history_len, tokens, (size_t)n * sizeof(int));
            }

            // Feed prompt tokens through forward pass
            float *logits = bn_prefill(&model, session, tokens, n, pos, args.no_prefill);
            pos += n;
            if (history && history_len + n <= seq_len) history_len += n;

            if (!logits) {
                SH_LOG_ERROR("Forward pass returned NULL during prompt");
                break;
            }
            for (int i = 0; i < n; i++) {
                bn_sampler_accept(&sampler, tokens[i]);
            }

            // Cap generation to remaining context
            int max_gen = args.n_tokens;
            int remaining = seq_len - pos - 1;
            if (remaining < max_gen) max_gen = remaining;
            if (max_gen < 1) max_gen = 1;

            BnChatHistory hist_ctx = { history, history_len, seq_len, args.token_ids };
            int gen_count = bn_generate(&model, session, &tokenizer, &sampler,
                                         max_gen, &pos,
                                         print_token, &hist_ctx, NULL, NULL);
            history_len = hist_ctx.history_len;

            // Feed end-of-turn token into KV cache to close the assistant turn
            int turn_end_id = bn_chat_turn_end_id(&tokenizer, BN_CHAT_AUTO);
            if (turn_end_id >= 0 && pos < seq_len) {
                bn_transformer_forward(&model, session, turn_end_id, pos);
                pos++;
                if (history && history_len < seq_len) history[history_len++] = turn_end_id;
            }

            // Cache KV state after this complete turn for prefix reuse
            if (prompt_cache && history && history_len > 1) {
                bn_prompt_cache_store(prompt_cache, &model, session, history, history_len);
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
        free(history);
        bn_prompt_cache_free(prompt_cache);
    } else {
        // --- Single-shot generation ---
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
        double prompt_start = bn_platform_time_ms();
        double gen_start = prompt_start;
        int pos = 0;
        int n_generated = 0;

        // Prefill prompt tokens
        float *logits = bn_prefill(&model, session, prompt_tokens, n_prompt, 0, args.no_prefill);
        pos = n_prompt;

        // Also prefill draft model if speculative decoding
        if (has_draft && logits) {
            bn_prefill(&draft_model, draft_session, prompt_tokens, n_prompt, 0, args.no_prefill);
        }

        if (!logits) {
            SH_LOG_ERROR("Forward pass returned NULL during prompt");
        } else {
            for (int i = 0; i < n_prompt; i++) {
                bn_sampler_accept(&sampler, prompt_tokens[i]);
            }
            gen_start = bn_platform_time_ms();
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
                n_generated = bn_generate_speculative(
                    &model, session, &draft_model, draft_session, args.draft_k,
                    &tokenizer, &sampler, args.n_tokens, &pos,
                    args.quiet ? NULL : print_token, NULL, NULL);
                if (n_generated < 0)
                    SH_LOG_ERROR("Speculative generation failed");
            } else {
                BnChatHistory out_ctx = { NULL, 0, 0, args.token_ids };
                n_generated = bn_generate(&model, session, &tokenizer, &sampler,
                                           args.n_tokens, &pos,
                                           args.quiet ? NULL : print_token,
                                           args.token_ids ? &out_ctx : NULL,
                                           NULL, NULL);
            }
        }

        double gen_end = bn_platform_time_ms();
        double gen_time = gen_end - gen_start;
        double total_time = gen_end - t0;

        printf("\n");
        {
            char ng[16], speed[32], prompt_ms[32], total[32];
            snprintf(ng, sizeof(ng), "%d", n_generated);
            if (n_generated > 0) {
                snprintf(speed, sizeof(speed), "%.2f", n_generated / (gen_time / 1000.0));
            } else {
                snprintf(speed, sizeof(speed), "0");
            }
            snprintf(prompt_ms, sizeof(prompt_ms), "%.1f", gen_start - prompt_start);
            snprintf(total, sizeof(total), "%.1f", total_time);
            SH_LOG_INFO("Generation complete", "tokens", ng, "tok/s", speed,
                        "prompt_ms", prompt_ms, "total_ms", total);
        }

        // Print MoE stats if applicable
        if (session->moe_state)
            bn_moe_print_stats(session->moe_state, n_generated + n_prompt);

        free(prompt_tokens);
    }

    // Cleanup
    if (has_draft) {
        bn_session_free(draft_session, NULL);
        bn_model_free(&draft_model);
        bn_gguf_free(draft_gf);
    }
    if (bn_model_moe_cache(&model)) {
        bn_moe_cache_free(bn_model_moe_cache(&model));
        bn_model_set_moe_cache(&model, NULL);
    }
#if defined(BN_ENABLE_WEBGPU) || defined(BN_ENABLE_METAL)
    if (bn_model_gpu_moe_cache(&model)) {
        bn_gpu_moe_cache_print_stats(bn_model_gpu_moe_cache(&model));
        bn_gpu_moe_cache_free(bn_model_gpu_moe_cache(&model));
        bn_model_set_gpu_moe_cache(&model, NULL);
    }
#endif
    bn_session_free(session, NULL);
    bn_sampler_free(&sampler);
    bn_tokenizer_free(&tokenizer);
    bn_model_free(&model);  // also unloads mmap'd file
    bn_gguf_free(gf);

    return 0;
}
