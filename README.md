# bitnet.c

Minimal, zero-dependency LLM inference in pure C11.

## Why bitnet.c

LLM inference engines are bloated. llama.cpp is 200K+ lines of C++ with CMake, CUDA SDKs, and abstraction layers between you and the SIMD instructions that actually matter. Ollama wraps that in Go. vLLM wraps PyTorch in Python. Each layer adds build complexity, deployment friction, and memory overhead.

bitnet.c takes the opposite approach:

- **Zero dependencies.** Only libc and libm. No C++, no CMake, no Python, no package managers. Optional GPU via wgpu-native — but CPU is the default and first-class citizen.
- **Minimal memory footprint.** [TurboQuant](#turboquant-kv-cache-compression) compresses the KV cache to 3-bit (8.9x smaller), so you serve ~9x more concurrent users in the same RAM. A 35B MoE model at 64K context runs in 8.4 GB — not 26 GB.
- **Flash MoE.** SSD-streamed expert loading via pread + LRU cache means you run 21 GB MoE models on a 16 GB machine. Hot experts stay in cache; cold ones stream from NVMe at 5 GB/s.
- **CPU-first SIMD.** ARM NEON/SDOT, AVX2, WASM SIMD128 — auto-selected at compile time. The forward pass is a flat `for` loop over layers calling SIMD matvec kernels directly. No graph abstraction, no tensor framework.
- **Embeddable.** ~8,000 lines of C11. Compiles to WASM and runs in a browser. Clean build under 3 seconds. Link it as a static library into anything.

**When to use llama.cpp/Ollama instead:** CUDA inference, OpenAI-compatible API serving, or multi-model management.

## Features

- **Pure C11** — no C++, no frameworks, no dependencies beyond libc and libm (GPU backend optional)
- **GGUF model loading** — loads any GGUF file with supported tensor types
- **20+ quantization formats** — ternary, k-quants, imatrix codebook, and unquantized (see table below)
- **Full transformer forward pass** — RoPE, GQA, RMSNorm, sub-norms, tied/untied embeddings
- **Hybrid SSM + Attention** — Gated DeltaNet SSM layers (conv1d, SiLU, delta rule recurrence) alongside standard GQA attention layers
- **Flash GQA attention** — online softmax with KV-head grouping, single-pass over KV cache
- **TurboQuant KV cache compression** — `--kv-tq 3` compresses KV cache to 3-bit (8.9x smaller than FP32) via Randomized Hadamard Transform + Lloyd-Max quantization + QJL residual correction. NEON SIMD vectorized. Dramatically reduces per-session memory: serve ~9x more concurrent users, or fit 256K context in 15 GB for a 35B MoE model.
- **Optional F16 KV cache** — `--kv16` halves attention DRAM bandwidth with minimal precision loss
- **5 SIMD backends** — ARM NEON/SDOT, AVX2, WASM SIMD128, scalar fallback (auto-selected at compile time), plus optional WebGPU
- **Mixture of Experts (MoE)** — sparse MoE with top-K routing, batched expert dispatch, 3 I/O modes (mmap, pread+LRU cache, madvise)
- **Pthread thread pool** — persistent workers with atomic work-stealing dispatch
- **BPE tokenizer** — loaded directly from GGUF metadata
- **Sampling** — greedy (argmax), multinomial, and nucleus (top-p)
- **Native mmap** — zero-copy model loading on macOS/Linux
- **Optional WebGPU backend** — wgpu-native, 41 WGSL shaders, single-submit forward pass
- **Prompt caching** — shared KV prefix cache with longest-prefix matching, FIFO eviction, TQ-aware (8.7x smaller entries with `--kv-tq`)
- **SSE streaming** — OpenAI-compatible server-sent events formatter
- **Logprobs API** — top-K log probabilities from logits
- **WASM build** — runs in the browser via Emscripten with Web Worker streaming

## Quantization Formats

### Ternary / Binary

| Format | bpw | Block | Packing |
|--------|-----|-------|---------|
| I2_S   | 2.0 | 128 | 2-bit interleaved + per-tensor scale |
| TQ1_0  | 1.69 | 256 | Base-3 (5 values/byte) + residual |
| TQ2_0  | 2.06 | 256 | 2-bit fields (4 values/byte) |

### K-Quants

| Format | bpw | Block | Packing |
|--------|-----|-------|---------|
| Q2_K | 2.63 | 256 | 2-bit quants + 4-bit sub-block scales/mins |
| Q3_K | 3.44 | 256 | 3-bit quants (split ql/qh) + 6-bit scales |
| Q4_0 | 4.50 | 32 | 4-bit nibbles + FP16 per-block scale |
| Q4_1 | 5.00 | 32 | 4-bit nibbles + FP16 scale + FP16 min |
| Q4_K | 4.50 | 256 | 4-bit quants + 6-bit sub-block scales/mins |
| Q5_K | 5.50 | 256 | 5-bit quants (split ql/qh) + 6-bit scales/mins |
| Q6_K | 6.56 | 256 | 6-bit quants (split ql/qh) + int8 scales |
| Q8_0 | 8.50 | 32 | 8-bit values + FP16 per-block scale |
| Q8_K | 9.13 | 256 | 8-bit values + float32 scale + int16 sums |

### Importance-Matrix (imatrix) Codebook

| Format | bpw | Block | Packing |
|--------|-----|-------|---------|
| IQ2_XXS | 2.06 | 256 | 2-bit codebook (256-entry grid) + packed signs/scales |
| IQ2_XS  | 2.31 | 256 | 2-bit codebook (512-entry grid) + explicit scales |
| IQ2_S   | 2.56 | 256 | 2-bit codebook (1024-entry grid) + scales |
| IQ3_XXS | 3.06 | 256 | 3-bit codebook (256-entry grid) + packed signs/scales |
| IQ3_S   | 3.56 | 256 | 3-bit codebook (512-entry grid) + separate signs/scales |
| IQ4_NL  | 4.50 | 32 | 4-bit non-linear (16-entry LUT) |
| IQ4_XS  | 4.25 | 256 | 4-bit non-linear + 6-bit sub-block scales |

### Unquantized

| Format | bpw | Packing |
|--------|-----|---------|
| F16  | 16 | IEEE 754 half-precision |
| BF16 | 16 | bfloat16 (truncated float) |
| F32  | 32 | IEEE 754 single-precision |

## SIMD Backends

| Backend | Platform | Notes |
|---------|----------|-------|
| ARM NEON/SDOT | Apple Silicon, ARMv8 | SDOT int8 matvec, native FP16 logits |
| AVX2 | x86-64 (Haswell+) | DPBUSD int8 matvec, F16C conversion |
| WASM SIMD128 | Browser (Emscripten) | Relaxed SIMD SDOT for all types |
| Scalar | Any C11 compiler | Portable fallback |
| WebGPU | GPU (wgpu-native) | 41 WGSL shaders, optional `BN_ENABLE_WEBGPU=1` |
| Metal | macOS GPU | Native Metal backend, optional `BN_ENABLE_METAL=1` |

CPU backends auto-selected at compile time based on target architecture. WebGPU and Metal are opt-in.

## Architecture And Test Matrix

The codebase keeps model loading, quant kernels, transformer kernels, GPU backends, and request state in separate modules:

- Quant formats live in `src/quant/{format}_{backend}.c`; dispatch is centralized in `src/quant/dispatch.c`.
- Transformer kernels live in `src/transformer/{op}_{backend}.c`; the layer loop remains in `src/transformer.c`.
- GPU implementations share the `BnGPUBackend` vtable in `include/gpu_backend.h`; WebGPU and Metal live in `src/gpu_wgpu.c` and `src/gpu_metal.m`.
- `BnModel` is immutable after load; `BnSession` owns per-request mutable KV cache and activations.

Executable harnesses keep those boundaries visible:

```bash
# Static backend/quant/shader coverage matrix
make test_backend_matrix

# Model-family coherence harness. Missing models are skipped by default.
make test_model_matrix

# Require all configured model-family cases to be present
REQUIRE_MODELS=1 make test_model_matrix

# Auto-discover local GGUFs under a root, e.g. /data/models/gguf
BN_MODEL_ROOT=/data/models/gguf make test_model_matrix
```

The model matrix covers Llama 2, Llama 3, Microsoft BitNet 1.58, Qwen 2.5, Qwen 3 dense, Qwen 3 sparse MoE, Qwen 3.5 dense, and Qwen 3.5 sparse MoE. For exact paths, set `BN_MODEL_LLAMA2`, `BN_MODEL_LLAMA3`, `BN_MODEL_BITNET158`, `BN_MODEL_QWEN25`, `BN_MODEL_QWEN3_DENSE`, `BN_MODEL_QWEN3_MOE`, `BN_MODEL_QWEN35_DENSE`, or `BN_MODEL_QWEN35_MOE`.

## Quick Start

```bash
# Build (CPU)
make

# Build with WebGPU backend (optional)
make fetch-wgpu
make BN_ENABLE_WEBGPU=1

# Run inference
./bitnet model.gguf -p "Hello" -n 256

# Run with sampling
./bitnet model.gguf -p "Once upon a time" -n 512 --temp 0.7 --topp 0.9

# Interactive chat
./bitnet model.gguf --chat

# Run tests
make test

# Cross-backend coherence test (GPU vs CPU, SIMD vs scalar)
make BN_ENABLE_WEBGPU=1 test_coherence
./test_coherence model.gguf --webgpu

# Architecture matrix dry run used by CI
make test_architecture
```

### Options

```
Usage: ./bitnet <model.gguf> [options]
  -p <prompt>     Input prompt (default: "Hello")
  -n <int>        Number of tokens to generate (default: 256)
  --temp <float>  Temperature (default: 0.0 = greedy)
  --topp <float>  Top-p sampling (default: 0.9)
  --seed <int>    Random seed (default: 42)
  --maxseq <int>  Max sequence length (default: model max; GPU auto-caps large contexts to 4096)
  --flash         Use flash attention (online softmax)
  --chat          Interactive chat REPL mode
  --repeat-penalty <float>  Repetition penalty (default: 1.0, chat: 1.1)
  --kv16          Store KV cache in FP16 (halves attention DRAM bandwidth)
  --kv-tq <bits>  TurboQuant KV compression (2, 3, or 4 bits; recommended: 3)
  --no-prefill    Disable batch prompt prefill (compute logits for every token)
  --pread         Force pread for MoE expert loading (lower RSS than mmap)
  --cache-mb <N>  Expert LRU cache budget in MB (default: 4096, 0 to disable)
  --gpu-cache-mb <N>  GPU expert buffer cache in MB (default: 4096, 0 to disable)
  --madvise       madvise-guided mmap for MoE (experimental)
  --prefault-moe  Fault all mmap'd MoE expert pages during startup
  --draft <path>  Draft model for speculative decoding (greedy, same tokenizer required, inherits --kv-tq)
  --draft-k <int> Draft tokens per iteration (default: 5)
  --webgpu        Enable WebGPU inference (requires BN_ENABLE_WEBGPU=1 build)
  --metal         Enable Metal inference (requires BN_ENABLE_METAL=1 build)
  --shader-dir <path>        WebGPU shader directory (default: shaders/)
  --metal-shader-dir <path>  Metal shader directory (default: shaders/metal/)
  -t <int>        Number of threads (default: auto-detect)
```

### Chat Mode

`--chat` enters an interactive REPL with multi-turn conversation support. KV cache is reused across turns so context accumulates naturally. Type `/quit` or Ctrl-D to exit. Type `/reset` to clear context (restores cached KV prefix if available).

Chat mode defaults to `--temp 0.5 --topp 0.9 --repeat-penalty 1.1` for more natural conversation. Override with explicit flags.

**Prompt caching**: Chat mode tracks full token history and caches KV state after each complete turn. On `/reset`, the longest matching prefix is restored from cache, skipping re-prefill. On context overflow, the session resets cleanly. With `--kv-tq 3`, cached entries are 8.7x smaller and restore ~10x faster.

### MoE Expert I/O Modes

For Mixture of Experts models (Qwen3-MoE, OLMoE, Mixtral), expert weights can be loaded in three modes:

- **mmap** (default): Direct memory-mapped access. Full model in RAM. Fastest throughput with cross-expert batched dispatch.
- **pread** (`--pread`): SSD streaming via pread syscalls with 2 prefetch threads and an LRU expert cache. Uses `--cache-mb` to control cache budget (default 4096 MB). Lower RSS at the cost of I/O latency.
- **madvise** (`--madvise`): Mmap with `MADV_WILLNEED` prefetch hints per expert. Experimental — syscall overhead currently negates the benefit.

```bash
# Default: mmap (all experts in RAM)
./bitnet models/Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello" -n 64

# Pread with 4GB expert cache (lower RSS)
./bitnet models/Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello" -n 64 --pread

# Pread with custom cache size
./bitnet models/Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello" -n 64 --pread --cache-mb 2048
```

### TurboQuant KV Cache Compression

`--kv-tq 3` compresses the KV cache from FP32 to 3-bit — an **8.9x reduction in per-session memory**. The KV cache is the dominant memory cost in multi-user serving: each concurrent session needs its own KV cache, so compressing it means proportionally more users in the same RAM.

Based on the TurboQuant paper (arXiv 2504.19874), using Randomized Hadamard Transform for O(d log d) rotation, Lloyd-Max scalar quantization, and QJL residual correction for keys. NEON SIMD vectorized on ARM with scalar fallback.

```bash
# Enable 3-bit TQ KV compression
./bitnet model.gguf -p "Hello" -n 256 --kv-tq 3

# Minimal memory: pread + small expert cache + TQ-3
./bitnet models/Qwen3.5-35B-A3B-Q4_K_M.gguf --pread --cache-mb 2048 --kv-tq 3 -p "Hello" -n 256
```

**Per-session KV cache size (Qwen3.5-35B-A3B, 40 layers, 4 KV heads):**

| Context | FP32 KV/session | TQ-3 KV/session | Compression |
|---------|-----------------|-----------------|-------------|
| 4K tokens | 1.2 GB | 0.14 GB | 8.9x |
| 16K tokens | 5.0 GB | 0.56 GB | 8.9x |
| 64K tokens | 20.0 GB | 2.25 GB | 8.9x |

**Concurrent sessions on a 32 GB machine** (pread + 2 GB expert cache, 4.1 GB non-expert weights = 6.1 GB base):

| Context | FP32 sessions | TQ-3 sessions | Multiplier |
|---------|---------------|---------------|------------|
| 4K tokens | 21 | 184 | **8.8x** |
| 16K tokens | 5 | 46 | **9.2x** |
| 64K tokens | 1 | 11 | **11x** |

At 64K context with FP32, you can barely fit **1 session** (20 GB KV + 6.1 GB base = 26.1 GB). With TQ-3, you fit **11 sessions** in the same RAM.

**Total RSS for a single session (pread + 2 GB expert cache):**

| Context | FP32 RSS | TQ-3 RSS |
|---------|----------|----------|
| 4K tokens | 7.4 GB | **6.3 GB** |
| 16K tokens | 11.1 GB | **6.7 GB** |
| 64K tokens | 26.1 GB | **8.4 GB** |
| 256K tokens | 86.1 GB | **15.1 GB** |

**Performance overhead** (Qwen3.5-35B-A3B):

| Context | TQ-3 | Baseline | Overhead |
|---------|------|----------|----------|
| 30 tokens | 3.5 tok/s | 7.2 tok/s | 2.1x |
| 141 tokens | 3.1 tok/s | 3.6 tok/s | 1.2x |
| 561 tokens | 1.1 tok/s | 1.2 tok/s | 1.06x |
| 1401 tokens | 0.45 tok/s | 0.47 tok/s | 1.04x |

TQ overhead vanishes at longer context — at 500+ tokens it's within 5% of baseline. The per-token write cost amortizes as context grows and KV read bandwidth savings dominate.

## Getting a Model

Any GGUF model using supported weight types works. Download from HuggingFace:

```bash
mkdir -p model
pip install huggingface-hub

# Example: Qwen2.5-3B (Q4_K_M quantization)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  --include "qwen2.5-3b-instruct-q4_k_m.gguf" \
  --local-dir model/

# Example: BitNet ternary model
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \
  --include "bitnet-b1.58-2B-4T-TQ2_0.gguf" \
  --local-dir model/
```

Then run:

```bash
./bitnet model/qwen2.5-3b-instruct-q4_k_m.gguf -p "The capital of France is"
```

The `model/` directory is git-ignored — model files won't be committed.

## Project Structure

```
bitnet.c/
├── include/
│   ├── platform.h              # Platform abstraction (mmap, timing)
│   ├── gguf.h                  # GGUF v3 reader API
│   ├── quant.h                 # Public quant API: block structs, matvec, dequant
│   ├── quant_internal.h        # Quant backend context structs + range function decls
│   ├── quant_neon_helpers.h    # Shared NEON helpers for quant kernels
│   ├── iq_tables.h             # IQ codebook lookup tables (shared across backends)
│   ├── turboquant.h            # TurboQuant KV compression: RHT + Lloyd-Max + QJL
│   ├── model.h                 # Config, Weights, model loading, session arena helpers
│   ├── moe.h                   # MoE expert routing, loading, LRU cache
│   ├── session.h               # BnSession: per-request KV cache + activation buffers
│   ├── prompt_cache.h          # BnPromptCache: shared KV prefix cache
│   ├── gpu_backend.h           # BnGPUBackend: GPU compute vtable
│   ├── gpu_wgpu.h              # wgpu-native WebGPU backend API
│   ├── gpu_metal.h             # Native Metal backend API
│   ├── transformer.h           # Forward pass public API
│   ├── transformer_internal.h  # Transformer backend context structs + range function decls
│   ├── tokenizer.h             # BPE tokenizer API
│   ├── sampler.h               # Sampling strategies
│   ├── generate.h              # Library API: generate, prefill, chat, SSE, logprobs
│   ├── bn_alloc.h              # Vtable allocator (Hull-compatible)
│   ├── threadpool.h            # Persistent pthread thread pool
│   ├── simd_helpers.h          # Shared AVX2/WASM SIMD inline helpers
│   ├── sh_arena.h              # Arena allocator
│   └── sh_log.h                # Structured logging
├── src/
│   ├── platform.c              # mmap/fread/timing abstraction
│   ├── gguf.c                  # GGUF binary format parser
│   ├── quant/                  # Per-format per-backend matvec kernels
│   │   ├── dispatch.c          # Format dispatch + batch matvec
│   │   ├── dequant.c           # Dequantization functions
│   │   ├── fp16.c              # FP16/BF16 ↔ FP32 conversion
│   │   └── {format}_{backend}.c # ~97 backend kernel files
│   ├── transformer/            # Per-backend transformer kernels
│   │   ├── rmsnorm_{neon,avx2,wasm,scalar}.c
│   │   ├── gqa_{neon,avx2,wasm,scalar}.c      # Standard GQA attention
│   │   ├── gqa_tq_{neon,scalar}.c              # TurboQuant GQA attention
│   │   ├── logits_{neon,avx2,wasm,scalar}.c
│   │   └── ssm_{neon,avx2,wasm,scalar}.c       # Gated DeltaNet SSM kernels
│   ├── turboquant.c            # TurboQuant: RHT (NEON/scalar), QJL, Lloyd-Max
│   ├── model.c                 # GGUF → Config/Weights mapping, GPU weight upload
│   ├── moe.c                   # MoE routing, pread + LRU cache, expert dispatch
│   ├── session.c               # BnSession create/free/reset
│   ├── prompt_cache.c          # Shared KV prefix cache with FIFO eviction
│   ├── gpu_wgpu.c              # wgpu-native WebGPU backend (optional)
│   ├── gpu_metal.m             # Native Metal backend (optional)
│   ├── generate.c              # Library API: generate, prefill, chat, SSE, logprobs
│   ├── transformer.c           # Forward pass: layer loop, FFN, dispatch
│   ├── tokenizer.c             # BPE encode/decode from GGUF vocab
│   ├── sampler.c               # Argmax, multinomial, top-p sampling
│   ├── threadpool.c            # Thread pool with condvar dispatch
│   └── main.c                  # CLI entry point
├── shaders/                    # WGSL shaders for WebGPU backend
│   ├── {format}_matvec.wgsl    # 23 matvec shaders (all quant types)
│   ├── rmsnorm.wgsl            # Forward-pass shaders
│   ├── rope.wgsl
│   ├── gqa_scores.wgsl
│   ├── gqa_combine.wgsl
│   ├── silu_gate.wgsl
│   ├── relu2_gate.wgsl
│   ├── residual_add.wgsl
│   ├── softmax.wgsl
│   ├── bias_add.wgsl
│   ├── weighted_add.wgsl       # MoE expert accumulation
│   ├── sigmoid_gate.wgsl       # MoE shared expert gate
│   ├── per_head_rmsnorm.wgsl   # Q/K per-head norms
│   ├── deinterleave_q.wgsl     # Q-gated attention deinterleave
│   ├── ssm_conv_silu.wgsl      # SSM: conv1d + SiLU activation
│   ├── ssm_l2norm.wgsl         # SSM: L2-norm Q/K
│   ├── ssm_alpha_beta.wgsl     # SSM: decay/update rates
│   ├── ssm_delta.wgsl          # SSM: delta rule recurrence
│   └── ssm_gate.wgsl           # SSM: SiLU gate + output
├── test/                       # Assert-based unit tests + backend/model harnesses
├── wasm/                       # Emscripten WASM build + browser demo
├── docs/
│   ├── inference.md            # Inference pipeline: math, algorithms, code map
│   └── roadmap.md              # Development roadmap + optimization history
└── Makefile
```

### Module Dependency Graph

Each module depends only on those above it:

```
platform
    ↓
  gguf              ← standalone, testable in isolation
    ↓
  quant             ← standalone, testable with synthetic data
    ↓
turboquant          ← standalone: RHT rotation, Lloyd-Max quantization, QJL
    ↓
  model             ← depends on gguf + quant + turboquant
    ↓
tokenizer           ← depends on gguf
    ↓
  moe               ← depends on model + quant (expert routing, pread + LRU cache)
    ↓
 session            ← depends on model (per-request KV cache + TQ buffers)
    ↓
transformer         ← depends on model + session + quant + moe + turboquant
    ↓
 sampler            ← standalone, testable in isolation
    ↓
prompt_cache        ← depends on model + session + turboquant
    ↓
 generate           ← depends on model + session + transformer + tokenizer + sampler
    ↓
gpu_wgpu            ← optional WebGPU backend (BN_ENABLE_WEBGPU=1)
    ↓
  main              ← wires everything together
```

## Library API

bitnet.c can be used as a library. The model/session split enables concurrent request handling — multiple sessions share one immutable model.

```c
#include "generate.h"
#include "session.h"

// Load model (shared, immutable after load)
// Last arg: kv_tq_bits (0=off, 3=TurboQuant 3-bit KV compression)
BnModel model;
bn_model_load(&model, gf, 0, 0, 3);  // TQ-3 for 8.9x KV compression
model.file = mf;
model.pool = bn_tp_create(7);

// Create independent sessions (each gets its own compressed KV cache)
BnSession *s1 = bn_session_create(&model, NULL);
BnSession *s2 = bn_session_create(&model, NULL);

// Request 1
bn_prefill(&model, s1, tokens1, n1, s1->pos, 0);
s1->pos += n1;
bn_generate(&model, s1, &tok, &sampler, 256, &s1->pos, cb, ud, NULL, NULL);

// Request 2 (concurrent — shared weights, independent KV cache)
bn_prefill(&model, s2, tokens2, n2, s2->pos, 0);
s2->pos += n2;
bn_generate(&model, s2, &tok, &sampler, 256, &s2->pos, cb, ud, NULL, NULL);

// Cleanup
bn_session_free(s1, NULL);
bn_session_free(s2, NULL);
bn_model_free(&model);
```

Sessions are not thread-safe individually, but different sessions can be used from different threads concurrently since they share only immutable model data.

### Prompt Caching

`BnPromptCache` stores KV prefix snapshots with longest-prefix matching and FIFO eviction. When `--kv-tq` is enabled, entries store TQ-compressed packed bytes instead of FP32 — 8.7x smaller entries, ~10x faster store/restore.

```c
BnPromptCache *pc = bn_prompt_cache_create(max_bytes, NULL);
bn_prompt_cache_store(pc, &model, session, tokens, n_tokens);     // snapshot KV state
int restored = bn_prompt_cache_restore(pc, &model, session, tokens, n);  // longest prefix match
bn_prompt_cache_free(pc);
```

Format validation prevents mismatches — a TQ-3 cached entry won't match a FP32 restore request, and vice versa.

### SSE Streaming & Logprobs

`bn_format_sse_chunk` / `bn_format_sse_done` format tokens as OpenAI-compatible server-sent events. `bn_logprobs_compute` returns top-K log probabilities from logits.

## WASM Build

Requires [Emscripten](https://emscripten.org/):

```bash
./wasm/build.sh
# Produces wasm/bitnet.js + wasm/bitnet.wasm
# Open wasm/index.html in a browser
```

## GPU Backend (WebGPU)

Optional WebGPU inference via [wgpu-native](https://github.com/gfx-rs/wgpu-native). Requires `BN_ENABLE_WEBGPU=1` at build time.

```bash
make fetch-wgpu                          # download wgpu-native v27
make BN_ENABLE_WEBGPU=1                  # build with WebGPU support
./bitnet model.gguf -p "Hello" --webgpu  # run on WebGPU
```

The WebGPU backend uses 41 WGSL shaders (23 matvec covering all quantization formats + 10 forward-pass + 3 MoE + 5 SSM operations) with a single-submit forward pass — one command buffer per token for dense transformer models. Hybrid SSM and sparse MoE models use WebGPU for supported attention/dense operations while routing SSM and MoE layers through the CPU path for token-coherent generation until the full WebGPU SSM/MoE paths pass coherence tests. The `BnGPUBackend` vtable in `include/gpu_backend.h` abstracts GPU compute, with the wgpu-native implementation in `src/gpu_wgpu.c`.

For Hull integration, set `WGPU_LIB_DIR` to avoid double-vendoring the wgpu-native library.

### GPU on WSL2

WSL2 does not ship a native NVIDIA Vulkan ICD. WebGPU inference requires Mesa's **dzn** (Dozen) driver, which translates Vulkan calls to D3D12 and routes them to the host GPU. Stock dzn lacks extensions wgpu-native requires, so a patch is included.

```bash
# Prerequisites
sudo apt-get install -y meson libdrm-dev libelf-dev llvm-dev \
  libexpat1-dev directx-headers-dev ninja-build python3-mako libvulkan-dev
pip3 install --user meson   # need meson >= 1.4

# Build patched dzn driver (~1 min)
./patches/build-dzn.sh

# Run with GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
VK_ICD_FILENAMES=/tmp/mesa-dzn/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json \
./bitnet model.gguf --webgpu --maxseq 4096 -p "Hello" -n 64
```

**Note:** WebGPU inference auto-caps large model contexts to 4096 when `--maxseq` is omitted to keep KV cache buffers within device limits. Pass `--maxseq N` to override.

The patch (`patches/mesa-dzn-wgpu-compat.patch`) makes these changes to Mesa's dzn driver:
- Advertises `VK_EXT_robustness2`, `VK_EXT_image_robustness`, `VK_KHR_zero_initialize_workgroup_memory` (D3D12 provides the underlying guarantees)
- Raises `maxStorageBufferRange` from 128 MB to 2 GB-1 (D3D12 supports this)
- Sets robustness2 alignment properties (required by wgpu, D3D12 has no alignment constraint)
- Reports conformance version 1.0.0.0 (wgpu rejects adapters reporting 0.0.0.0)

## Model Support

Tested models with generation quality and performance on Apple M1 Max (8 P-cores, 32 GB), release build, greedy decoding, 8 threads:

| Model | Size | Quant | Architecture | tok/s | Quality |
|-------|------|-------|--------------|-------|---------|
| bitnet-b1.58-2B-4T | 1.1 GB | I2_S (ternary) | Transformer | **29–41** | Factual, correct code |
| Qwen2.5-3B-Instruct | 1.7 GB | Q4_0 | Transformer | **23–25** | Coherent, instruct-following |
| Llama3-8B-1.58 | 3.3 GB | TQ1_0 (ternary) | Transformer | **8–9** | Basic completion |
| Qwen3-30B-A3B | 17.7 GB | Q4_K_M | MoE (128 experts, K=8) | **19–20** | Parity-or-better than llama.cpp at steady state |
| Qwen3.5-35B-A3B | 21.0 GB | Q4_K_M | SSM+MoE (256 experts) | **9–12** | Parity-class CPU throughput with pread/mmap modes |
| Qwen3.5-9B | 5.3 GB | Q4_K_M (mixed) | Hybrid SSM+Attention | **2.8–3.1** | Best quality, chain-of-thought |
| Qwen3.6-27B | 15.7 GB | Q4_K_M | Hybrid SSM+Attention | **2.5–2.8** | llama.cpp parity-class on AVX2 for short greedy runs |
| Qwen3.6-35B-A3B | 20.6 GB | Q4_K_M | SSM+MoE (256 experts) | **1.4–1.8 cold**, higher when warm | Functional on CPU and hybrid WebGPU/CPU fallback |

Any GGUF model using supported weight types works — these are just the tested configurations.

### Hybrid SSM + Attention

bitnet.c supports hybrid architectures that mix SSM (state space model) layers with standard attention layers, such as Qwen3.5's Gated DeltaNet:

- **SSM layers**: Conv1d (kernel=4) → SiLU → L2-norm Q/K → delta rule recurrence → per-head RMSNorm → SiLU gate → output projection
- **Attention layers**: Standard GQA with RoPE, flash attention, KV cache
- **Mixed layout**: The model config specifies which layers use SSM vs attention

## Performance

Measured on Apple M1 Max (8 P-cores, 32 GB), PGO build, greedy decoding, 8 threads. llama.cpp b8320 via Homebrew. Current results are parity-class with llama.cpp on the serving-oriented MoE workloads and competitive on dense CPU workloads; exact winners vary by quant format, page-cache state, and whether llama.cpp routes work to Metal.

| Model | Size | Quant | bitnet.c | llama.cpp CPU | Status |
|-------|------|-------|----------|---------------|--------|
| bitnet-b1.58-2B-4T | 620 MB | I2_S (ternary) | **52 tok/s** | — | Native BitNet path; llama.cpp does not cover this quant |
| Qwen2.5-3B-Instruct | 1.7 GB | Q4_0 | **30 tok/s** | 40 tok/s | Competitive dense CPU path |
| Llama3-8B-1.58 | 3.4 GB | TQ1_0 (ternary) | **14.5 tok/s** | 19 tok/s | Competitive ternary CPU path |
| Qwen3-30B-A3B MoE | 17.7 GB | Q4_K_M | **19–20 tok/s** | 10–12 tok/s | Ahead at steady state |
| Qwen3.5-35B-A3B MoE | 21.0 GB | Q4_K_M | **9–12 tok/s** | 6–8 tok/s | Parity-or-better depending on I/O mode and cache state |
| Qwen3.6-27B hybrid | 15.7 GB | Q4_K_M | **2.5–2.8 tok/s** | 2.5 tok/s observed via llama.cpp CPU completion | Parity-class AVX2; first-token prefill dominates short prompts |
| Qwen3.6-35B-A3B MoE | 20.6 GB | Q4_K_M | **1.4–1.8 tok/s cold** | Comparable CPU behavior, strongly cache-state dependent | Sparse Qwen3.6 is token-coherent; `--webgpu` uses CPU fallback for SSM/MoE layers |

CPU performance is now best described as parity-class rather than a blanket percentage. Dense Q4_0/TQ1_0 rows remain close but still model-specific; MoE serving rows reach parity or exceed llama.cpp once the expert working set is warm. Qwen3.6 uses F32 SSM projection tensors inside otherwise quantized GGUFs, so the unquantized F32/F16 matvec path is required for CPU and GPU backends. llama.cpp may route some nominal CPU MoE work through Metal, so comparisons should note backend and page-cache state. Optional WebGPU (`--webgpu`) and native Metal (`--metal`) backends are available for GPU-accelerated inference; on adapters with 128 MB storage-binding limits, oversized logits weights fall back to CPU while the main forward pass remains on GPU. Hybrid SSM and sparse MoE models currently use CPU fallback for SSM/MoE layers under `--webgpu` because the full WebGPU SSM/MoE paths are not yet token-coherent.

## Known Limitations

- **Small ternary models degrade on multi-turn chat.** The 2B 1.58-bit BitNet model (~400 MB effective weights) struggles with multi-turn conversations, especially after code-heavy responses. This is a model capacity limitation, not an inference engine bug.
- **Long generation degenerates into repetition.** Small models cannot reliably sustain coherent generation beyond ~50-100 tokens. A built-in loop detector stops generation when this is detected.

## Acknowledgments

- **[llama2.c](https://github.com/karpathy/llama2.c)** by Andrej Karpathy — the inspiration for proving that a complete LLM inference engine can be beautifully simple.
- **[BitNet](https://github.com/microsoft/BitNet)** by Microsoft Research — the 1.58-bit quantization research ([paper](https://arxiv.org/abs/2402.17764)) that makes ternary-weight transformers practical.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** / **[GGML](https://github.com/ggml-org/ggml)** by Georgi Gerganov — the GGUF file format, quantization implementations, codebook tables, and the broader ecosystem.

## License

[MIT](LICENSE)
