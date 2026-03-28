# CLAUDE.md

Instructions for Claude Code when working on this project.

## Project Overview

bitnet.c is a pure C11 inference engine for BitNet b1.58 transformer models. It loads GGUF model files and performs autoregressive text generation. Inspired by Karpathy's llama2.c.

## Build

```bash
make                    # build the main binary (CPU)
make BN_ENABLE_GPU=1    # build with GPU backend (requires make fetch-wgpu first)
make debug              # build with -DDEBUG -g -O0
make asan               # sanitizer build
make clean              # remove artifacts
make test               # run all unit tests
make avx2-check         # AVX2 cross-compile check
make fetch-wgpu         # download wgpu-native v27
```

Individual test targets: `make test_gguf`, `make test_quant`, `make test_tokenizer`, `make test_transformer`, `make test_generate`, `make test_session`, `make test_prompt_cache`, `make test_threadpool`, `make test_safety`, `make test_arena`, `make test_ssm`, `make test_gguf_fuzz`, `make test_moe`, `make test_turboquant`.

GPU-specific test targets (require `BN_ENABLE_GPU=1`): `make test_gpu_wgpu`, `make test_gpu_validate`, `make test_gpu_backend`.

Coherence test (requires a real GGUF model file):
```bash
make BN_ENABLE_GPU=1 test_coherence
./test_coherence models/bitnet-b1.58-2B-4T.gguf --gpu   # GPU vs CPU forward pass + matvec
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf   # CPU-only: SIMD vs scalar matvec
```

## Architecture

Modules are organized in strict dependency order — each depends only on those above it:

1. `platform` — mmap/buffer abstraction, timing
2. `gguf` — GGUF v3 binary parser (standalone)
3. `quant` — dequantization + SIMD matvec, 22 types x 5 backends (standalone)
4. `turboquant` — TurboQuant KV cache compression: Randomized Hadamard Transform + Lloyd-Max quantization + QJL residual. NEON SIMD vectorized (FWHT butterfly, vcntq popcount, FMA). Scalar fallback on non-ARM. (standalone)
5. `model` — GGUF → Config/Weights mapping, session arena helpers, GPU weight upload (depends on gguf + quant + turboquant)
6. `tokenizer` — BPE tokenizer (depends on gguf)
7. `moe` — MoE expert routing, loading, caching (depends on model + quant)
8. `session` — per-request mutable state: KV cache, activation buffers, MoE compute buffers (depends on model + bn_alloc)
9. `transformer` — forward pass, CPU SIMD + GPU-resident (depends on model + session + quant + moe + turboquant)
10. `sampler` — sampling strategies (standalone)
11. `threadpool` — persistent pthread pool with atomic work-stealing
12. `bn_alloc` — vtable allocator interface (standalone, Hull-compatible with `KlAllocator`)
13. `prompt_cache` — shared KV prefix cache: longest-prefix matching, FIFO eviction, thread-safe, TQ-aware (depends on model + session + bn_alloc + turboquant)
14. `generate` — library API: generation, prefill, chat formatting, SSE streaming, logprobs, stop strings (depends on model + session + tokenizer + sampler + transformer + bn_alloc)
15. `gpu_wgpu` — wgpu-native WebGPU backend, optional with `BN_ENABLE_GPU=1` (depends on model + gpu_backend)
16. `main` — CLI wiring (depends on generate + all above)

Headers live in `include/`, implementations in `src/`, tests in `test/`.

## Code Style

- C11, compiled with `-Wall -Wextra`
- No external dependencies (only libc + libm; wgpu-native optional for GPU)
- Use `#ifdef __EMSCRIPTEN__` for WASM-specific code paths
- Use `#ifdef DEBUG` for debug-only logging
- Prefix public API functions with module name: `gguf_`, `model_`, `tokenizer_`, etc.
- Static helper functions are file-local
- Memory: caller allocates structs, modules fill them. Use `_init`/`_free` pairs.

## Key Types

- `BnMappedFile` — wraps mmap'd or malloc'd buffer
- `BnGGUFFile` — parsed GGUF header, KV pairs, tensor info
- `BnBlockTQ1` / `BnBlockTQ2` — quantized weight blocks (54 / 66 bytes per 256 elements)
- `BnQWeight` — weight tensor descriptor (zero-copy into GGUF buffer)
- `BnConfig` — model hyperparameters (including MoE: `n_experts`, `n_experts_active`)
- `BnModel` — shared immutable state: config + weights + file + pool + MoE I/O + GPU backend
- `BnSession` — per-request mutable state: KV cache, activation buffers, MoE compute buffers, position
- `BnMoEIO` — shared MoE I/O control plane (fd, mmap_base, prefetch threads, LRU cache) on BnModel
- `BnMoEState` — per-session MoE compute buffers + pread staging + stats
- `BnPromptCache` — shared KV prefix cache with longest-prefix matching and FIFO eviction
- `BnPromptCacheEntry` — cached KV snapshot: token sequence + compact KV data (FP32/FP16 or TQ packed)
- `BnGPUBackend` — GPU compute vtable (buffer_create/destroy, matvec, matmul, matvec_batch, execute, init_activations)
- `BnMoEExpertMap` — file offsets for gate/up/down expert tensors per layer
- `BnTokenizer` — BPE vocab + sorted index for encoding
- `BnSampler` — sampling parameters + RNG state
- `BnAllocator` — vtable allocator (malloc/realloc/free + ctx), compatible with Hull's `KlAllocator`
- `BnChatMessage` — `{role, content}` for multi-turn chat formatting
- `BnStopStrings` — stop string array for generation halting
- `BnTQState` — TurboQuant state: RHT signs (rotation + QJL), Lloyd-Max centroids/boundaries, scale factor
- `BnGQATQCtx` — TurboQuant GQA attention context (packed keys/values, byte strides)

## MoE (Mixture of Experts)

MoE support is in `src/moe.c` + `include/moe.h`. The module handles expert routing, weight loading, and FFN compute for sparse MoE models (Qwen3, OLMoE, Mixtral, etc.).

### Expert I/O Modes

Three modes for loading expert weights, selected by CLI flags:

| Mode | Flag | How it works | RSS | Speed |
|------|------|-------------|-----|-------|
| **mmap** | (default) | Direct mmap'd file access. Cross-expert batched dispatch. | Full model | Fastest |
| **pread + LRU cache** | `--pread` | Pread syscalls with 2 prefetch threads. LRU cache (open-addressing hash + intrusive doubly-linked list) stores hot experts in a contiguous slab. `--cache-mb N` controls budget (default 4096). | Model - cache | Good |
| **madvise** | `--madvise` | Mmap with `MADV_WILLNEED` prefetch hints per expert. Experimental. | ~Model | Slower (syscall overhead) |

Expert I/O is **fully orthogonal** to SIMD dispatch — kernels don't know where weight data came from.

### SIMD Runtimes

5 backends, selected at compile time via `#ifdef` (plus optional GPU):

| Backend | Platforms | Key feature |
|---------|-----------|-------------|
| NEON SDOT | ARM (M1+) | `vdotq_s32` + Q8_K x quantization for Q4_K/Q6_K |
| AVX2 | x86-64 | `bn_avx2_dpbusd` + Q8_K x quantization for Q4_K/Q6_K |
| WASM SIMD128 | Browser/Node.js | Relaxed SIMD SDOT for all types |
| Scalar | Fallback | Pure C, no SIMD |
| WebGPU | GPU (wgpu-native) | 41 WGSL shaders (23 matvec + 10 forward-pass + 3 MoE + 5 SSM), optional `BN_ENABLE_GPU=1` |

Q8_K x quantization (256-element super-blocks with bsums) enables integer accumulation in Q4_K/Q6_K kernels. Unsigned nibbles, no bias subtract, float conversion once per super-block.

### MoE Forward Pass (`bn_moe_forward`)

1. RMSNorm input
2. Route: SIMD matvec → softmax → top-K selection
3. **mmap path**: batch all K experts' gate+up matvecs → parallel SwiGLU → individual down matvecs → weighted accumulation
4. **pread path**: two-phase (batch cache hits, then process misses with I/O overlap)
5. Shared expert (if present)
6. Residual add

### Thread Pool

Persistent pthread pool with atomic work-stealing dispatch (`include/threadpool.h`). Adaptive chunk size (`n / (4 * n_threads)`, min 16) for load balancing. ~2us condvar dispatch latency.

### Speculative Decoding

Optional `--draft <model.gguf>` flag loads a small draft model to generate K candidate tokens (default K=5 via `--draft-k`), then verifies with the target model. Greedy only (temp=0). Draft and target must share the same tokenizer (same vocab_size). Two `BnModel` instances coexist with shared thread pool; each has its own `BnSession` with independent KV cache and activation buffers. No KV cache rollback needed (attention window bounded by pos). Best with dense targets + same-family small draft; MoE targets verify sequentially (no batch speedup yet). `--kv-tq` applies to both target and draft sessions, shrinking the draft's KV cache too.

### TurboQuant KV Cache Compression

`--kv-tq <bits>` enables TurboQuant KV cache compression (2, 3, or 4 bits; recommended: 3). Based on arXiv 2504.19874.

**Algorithm**: Randomized Hadamard Transform (RHT) rotates vectors to Gaussianize coordinates, Lloyd-Max scalar quantization maps each coordinate to the nearest centroid, QJL residual correction preserves key score accuracy via 1-bit sign projections.

**Implementation** (`src/turboquant.c`, `include/turboquant.h`):
- **RHT**: `y = (1/sqrt(d)) * H * D * x` where H is Walsh-Hadamard (FWHT butterfly, O(d log d)) and D is diagonal ±1 signs from seeded PRNG
- **QJL**: Second independent RHT for sign projection (replaces dense O(d²) Gaussian matrix)
- **NEON SIMD** (`#ifdef __ARM_NEON`): vectorized FWHT (vuzp/vzip stride-1/2, vaddq/vsubq stride-4+), XOR sign-bit application, vcntq_u8 XNOR popcount, vfmaq_f32 centroid dot + value accumulate
- **Scalar fallback** (`#else`): pure C, no SIMD — always available on all platforms
- **Precomputed QJL API**: `bn_tq_qjl_precompute` + `bn_tq_score_key_precomputed` eliminate redundant per-key O(d log d) QJL projection

**GQA TQ kernels** (`src/transformer/gqa_tq_scalar.c`, `src/transformer/gqa_tq_neon.c`):
- Dispatch via `#ifdef __ARM_NEON` in `src/transformer.c`
- NEON version delegates to scalar (which internally uses NEON-optimized turboquant.c)
- Precomputes QJL signs once per head, scores all keys, softmax, then weighted combine

**Packed format per head** (d=128, 3-bit): key=68B, value=50B. For d=256: key=132B, value=98B.

**Per-session KV memory** (the main benefit — enables multi-user serving):
- 8.9x compression: 64K context goes from 20 GB/session (FP32) to 2.2 GB/session (TQ-3)
- On a 32 GB machine (pread + 2 GB cache): 1 FP32 session at 64K vs **11 TQ-3 sessions**
- 256K context: 80 GB/session (FP32, impossible) vs 9 GB/session (TQ-3, fits easily)

**Total RSS with `--pread --cache-mb 2048 --kv-tq 3`** (Qwen3.5-35B-A3B, single session):
- Base (non-expert weights + expert cache): ~6.1 GB
- 64K context: **8.4 GB** total (vs 26.1 GB with FP32)
- 256K context: **15.1 GB** total (vs 86.1 GB with FP32)

**Performance overhead**: ~2x at 30 tokens, ~1.2x at 140 tokens, <5% at 500+ tokens. The per-token write cost amortizes as context grows.

### Concurrent Sessions (BnModel/BnSession Split)

`BnModel` is shared and immutable after load — it holds config, weights, the mmap'd file, the thread pool, and shared MoE I/O (fd, mmap_base, prefetch threads, LRU cache).

`BnSession` holds all per-request mutable state — KV cache, activation buffers, MoE compute buffers, pread staging, and generation position. Multiple sessions can share one model for concurrent request handling.

**API:**
```c
BnSession *s = bn_session_create(&model, NULL);  // allocator or NULL for stdlib
bn_prefill(&model, s, tokens, n, 0, 0);
s->pos += n;
bn_generate(&model, s, &tok, &sampler, 256, &s->pos, cb, ud, NULL, NULL);
bn_session_reset(s, &model);   // clear KV cache, reset pos
bn_session_free(s, NULL);
```

All forward pass and generation functions take both `BnModel *` and `BnSession *`. The model provides weights and shared resources; the session provides mutable state.

### Prompt Cache

`BnPromptCache` stores KV prefix snapshots keyed by token sequences. Supports longest-prefix matching for efficient KV reuse across requests or chat turns.

**TQ-aware**: When `--kv-tq` is enabled, prompt cache stores TQ-compressed packed bytes instead of FP32/FP16 data. Entry sizes shrink ~8.7x (TQ-3), store/restore run ~8-10x faster (memcpy-bound). Format validation prevents mismatches (TQ-3 entry won't match FP32 query).

**Chat mode**: Tracks full token history and caches KV state after each complete turn (user prompt + assistant response + end-of-turn token). On `/reset`, restores the longest matching prefix. On context overflow, resets with history tracking for future prefix reuse.

**API:**
```c
BnPromptCache *pc = bn_prompt_cache_create(max_bytes, NULL);
bn_prompt_cache_store(pc, &model, session, tokens, n_tokens);    // snapshot KV
int restored = bn_prompt_cache_restore(pc, &model, session, tokens, n);  // longest prefix
bn_prompt_cache_clear(pc);
bn_prompt_cache_free(pc);
```

**Entry layout**: `BnPromptCacheEntry` holds separate `key_cache_bytes` and `val_cache_bytes` (split for TQ where key and value have different packed sizes), plus `kv_tq_bits` for format validation. Thread-safe via pthread mutex.

## Testing

Tests use assert-based checks with synthetic data — no real model files needed for unit tests. Each test file is self-contained and can be compiled independently with its module dependencies.

`test_e2e.c` requires a real GGUF model file: `./test_e2e model.gguf`

## WASM

WASM build requires Emscripten. Run `./wasm/build.sh`. The API wrapper in `wasm/api.c` exports functions via `EMSCRIPTEN_KEEPALIVE`. The browser demo uses a Web Worker for non-blocking inference.

## GPU Backend (WebGPU)

Optional GPU inference via wgpu-native. Build with `make BN_ENABLE_GPU=1 WGPU_LIB_DIR=vendor/wgpu` (run `make fetch-wgpu` first to download wgpu-native v27).

- 41 WGSL shaders in `shaders/` (23 matvec + 10 forward-pass + 3 MoE + 5 SSM)
- `BnGPUBackend` vtable in `include/gpu_backend.h` — buffer_create/destroy, matvec, matmul, matvec_batch, execute, init_activations
- wgpu-native runtime in `src/gpu_wgpu.c`
- `--gpu` CLI flag enables GPU inference
- Single-submit forward pass: one command buffer per token
- Hull-compatible: `WGPU_LIB_DIR` override avoids double-vendoring
- WSL2 requires a patched Mesa dzn driver — see `patches/` directory and `patches/build-dzn.sh`
- WSL2 GPU run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib VK_ICD_FILENAMES=/tmp/mesa-dzn/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json ./bitnet model.gguf --gpu --maxseq 4096`
- `--maxseq` is recommended on GPU to cap KV cache VRAM (models with 256K context will OOM otherwise)

## Common Tasks

- **Add a new GGUF metadata key**: read it in `model_load()` in `src/model.c`
- **Add a new tensor type**: add block struct to `include/quant.h`, dequant + SIMD kernels in `src/quant/`, dispatch in `src/quant/dispatch.c`
- **Add a new SIMD backend for existing type**: create `src/quant/<type>_<backend>.c`, add to Makefile `QUANT_BACKEND`, wire in `dispatch.c`
- **Modify the forward pass**: edit `transformer_forward()` in `src/transformer.c`
- **Modify MoE expert dispatch**: edit `bn_moe_forward()` in `src/moe.c`
- **Add a new sampling strategy**: extend `sampler_sample()` in `src/sampler.c`
- **Add a new MoE I/O mode**: add flag in `src/main.c`, branch in `bn_moe_forward()` in `src/moe.c`
- **Export a new function to WASM**: add `EMSCRIPTEN_KEEPALIVE` wrapper in `wasm/api.c`, update `build.sh` exported functions list
- **Integrate as a library**: `#include "generate.h"` and `#include "session.h"` — load model with `bn_model_load`, create session with `bn_session_create`, then use `bn_prefill`, `bn_generate`, `bn_chat_format_messages`. Pass custom `BnAllocator` or NULL for stdlib.
- **Add concurrent sessions**: Create multiple `BnSession` from the same `BnModel` — each gets independent KV cache and activation buffers. Sessions are not thread-safe individually, but different sessions can be used from different threads concurrently (they share only immutable model data).
- **Add a chat template**: add case to `BnChatFormat` enum, implement in `encode_*_turn` in `src/generate.c`
- **Prompt caching**: use `bn_prompt_cache_store`/`bn_prompt_cache_restore` for KV prefix reuse across requests. TQ-aware: stores compressed packed bytes when `--kv-tq` is enabled (8.7x smaller entries)
- **GPU inference**: `bn_gpu_wgpu_create`, `bn_model_upload_weights`, `--gpu` flag
- **SSE streaming**: `bn_format_sse_chunk` / `bn_format_sse_done` for OpenAI-compatible server-sent events
- **Logprobs**: `bn_logprobs_compute` for top-K log probabilities from logits
- **Add a GPU shader**: create `shaders/<name>.wgsl`, wire in `src/gpu_wgpu.c`

