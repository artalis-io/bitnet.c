# CLAUDE.md

Instructions for Claude Code when working on this project.

## Project Overview

bitnet.c is a pure C11 inference engine for BitNet b1.58 transformer models. It loads GGUF model files and performs autoregressive text generation. Inspired by Karpathy's llama2.c.

## Build

```bash
make          # build the main binary
make debug    # build with -DDEBUG -g -O0
make clean    # remove artifacts
make test     # run all unit tests
```

Individual test targets: `make test_gguf`, `make test_quant`, `make test_tokenizer`, `make test_transformer`.

## Architecture

Modules are organized in strict dependency order — each depends only on those above it:

1. `platform` — mmap/buffer abstraction, timing
2. `gguf` — GGUF v3 binary parser (standalone)
3. `quant` — dequantization + SIMD matvec for all quant types (standalone)
4. `model` — GGUF → Config/Weights mapping (depends on gguf + quant)
5. `tokenizer` — BPE tokenizer (depends on gguf)
6. `moe` — MoE expert routing, loading, caching (depends on model + quant)
7. `transformer` — forward pass (depends on model + quant + moe)
8. `sampler` — sampling strategies (standalone)
9. `threadpool` — persistent pthread pool with atomic work-stealing
10. `main` — CLI wiring

Headers live in `include/`, implementations in `src/`, tests in `test/`.

## Code Style

- C11, compiled with `-Wall -Wextra`
- No external dependencies (only libc + libm)
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
- `BnModel` — config + weights + runtime state + MoE state
- `BnMoEState` — MoE runtime: I/O buffers, expert cache, prefetch threads, stats
- `BnMoEExpertMap` — file offsets for gate/up/down expert tensors per layer
- `BnTokenizer` — BPE vocab + sorted index for encoding
- `BnSampler` — sampling parameters + RNG state

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

4 backends, selected at compile time via `#ifdef`:

| Backend | Platforms | Key feature |
|---------|-----------|-------------|
| NEON SDOT | ARM (M1+) | `vdotq_s32` + Q8_K x quantization for Q4_K/Q6_K |
| AVX2 | x86-64 | `bn_avx2_dpbusd` + Q8_K x quantization for Q4_K/Q6_K |
| WASM SIMD128 | Browser/Node.js | Relaxed SIMD SDOT for I2_S/Q4_0, float-x for K-quants |
| Scalar | Fallback | Pure C, no SIMD |

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

## Testing

Tests use assert-based checks with synthetic data — no real model files needed for unit tests. Each test file is self-contained and can be compiled independently with its module dependencies.

`test_e2e.c` requires a real GGUF model file: `./test_e2e model.gguf`

## WASM

WASM build requires Emscripten. Run `./wasm/build.sh`. The API wrapper in `wasm/api.c` exports functions via `EMSCRIPTEN_KEEPALIVE`. The browser demo uses a Web Worker for non-blocking inference.

## Common Tasks

- **Add a new GGUF metadata key**: read it in `model_load()` in `src/model.c`
- **Add a new tensor type**: add block struct to `include/quant.h`, dequant + SIMD kernels in `src/quant/`, dispatch in `src/quant/dispatch.c`
- **Add a new SIMD backend for existing type**: create `src/quant/<type>_<backend>.c`, add to Makefile `QUANT_BACKEND`, wire in `dispatch.c`
- **Modify the forward pass**: edit `transformer_forward()` in `src/transformer.c`
- **Modify MoE expert dispatch**: edit `bn_moe_forward()` in `src/moe.c`
- **Add a new sampling strategy**: extend `sampler_sample()` in `src/sampler.c`
- **Add a new MoE I/O mode**: add flag in `src/main.c`, branch in `bn_moe_forward()` in `src/moe.c`
- **Export a new function to WASM**: add `EMSCRIPTEN_KEEPALIVE` wrapper in `wasm/api.c`, update `build.sh` exported functions list

