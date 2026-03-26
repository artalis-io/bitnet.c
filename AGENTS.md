# AGENTS.md

Guidelines for AI agents working on this codebase.

## Repository Structure

```
include/   — public headers (one per module)
src/       — implementations (one per module + main.c)
test/      — unit tests (one per module + e2e)
wasm/      — WASM build, API wrapper, browser demo
docs/      — documentation and roadmap
```

## Module Boundaries

Modules have strict, one-directional dependencies. When modifying a module, only its own files and downstream consumers should be affected. Never introduce circular dependencies.

**Dependency order**: platform → gguf → quant → model → tokenizer → moe → session → transformer → sampler → threadpool → bn_alloc → prompt_cache → generate → gpu_wgpu (optional) → main

## Agent Workflow

### Before Making Changes

1. Read the relevant header file to understand the public API
2. Read the implementation file to understand current behavior
3. Read the corresponding test file to understand expected behavior
4. Run `make test` to confirm tests pass before changing anything

### Making Changes

1. Modify the implementation in `src/`
2. Update the header in `include/` if the API changes
3. Update or add tests in `test/`
4. Run `make clean && make bitnet` — must compile with zero warnings
5. Run `make test` — all tests must pass

### Adding a New Module

1. Create `include/newmodule.h` with the public API
2. Create `src/newmodule.c` with the implementation
3. Create `test/test_newmodule.c` with unit tests
4. Add to `SRCS` in `Makefile`
5. Add a test target in `Makefile`
6. Update the dependency graph in this file and CLAUDE.md

## Testing Strategy

- Unit tests use synthetic data — no model files required
- Each test builds only its module + dependencies (not the whole project)
- Tests use `assert()` — they crash on failure, print "PASSED" on success
- The e2e test (`test/test_e2e.c`) requires a real GGUF model file
- The coherence test (`test/test_coherence.c`) requires a real GGUF model file:
  - Phase 1: GPU vs CPU forward pass — 5 greedy tokens, first 3 must match
  - Phase 2: SIMD vs scalar matvec — all layer 0 weights, max_diff < 1.0
  - Phase 3: GPU standalone matvec vs CPU scalar — layer 0 wq, max_diff < 1.0
  - Run: `make BN_ENABLE_GPU=1 test_coherence && ./test_coherence model.gguf --gpu`

## Code Conventions

- C11 standard, no GNU extensions
- All public functions prefixed with module name: `gguf_open()`, `model_load()`, etc.
- Internal helpers are `static`
- Memory management: `_init`/`_free` pairs; caller owns the struct, module fills it. Exception: `BnSession` is heap-allocated via `bn_session_create`/`bn_session_free`.
- Error handling: return -1 or NULL on failure, print to stderr
- No global mutable state in library modules (only in main.c and wasm/api.c)
- **Model/Session split**: `BnModel` is shared and immutable after load (config, weights, file, pool, MoE I/O, GPU backend). `BnSession` holds per-request mutable state (KV cache, activation buffers, MoE compute buffers, pos). All forward pass and generation functions take both `BnModel *` and `BnSession *`.
- **GPU vtable**: `BnGPUBackend` in `include/gpu_backend.h` abstracts GPU compute (buffer ops, matvec, matmul, execute). The wgpu-native implementation is in `src/gpu_wgpu.c`. GPU code is gated by `BN_ENABLE_GPU`.
- Platform-specific code gated by `#ifdef __EMSCRIPTEN__`

## Performance Considerations

- 5 SIMD backends: NEON SDOT, AVX2, WASM SIMD128, scalar fallback (auto-selected at compile time), WebGPU (optional)
- Q8_K x quantization for Q4_K/Q6_K: integer accumulation, unsigned nibbles, bsums correction
- MoE expert LRU cache with open-addressing hash + intrusive LRU list (pread mode)
- Atomic work-stealing thread dispatch for load balancing
- Batch prefill with fused Q4_K matmul kernel (dense models)
- KV cache pre-allocated for max sequence length
- TurboQuant KV compression: Randomized Hadamard Transform (O(d log d)), NEON vectorized FWHT + popcount + FMA, precomputed QJL to avoid per-key redundancy

## CLI Flags (key ones for agents)

- `--pread` — force pread for MoE expert loading (lower RSS)
- `--cache-mb N` — expert LRU cache budget in MB (default 4096, pread only)
- `--madvise` — madvise-guided mmap (experimental)
- `--draft <path>` — draft model for speculative decoding (greedy, same tokenizer)
- `--draft-k N` — draft tokens per iteration (default 5)
- `--flash` — flash attention (online softmax)
- `--kv16` — FP16 KV cache
- `--kv-tq <bits>` — TurboQuant KV cache compression (2, 3, or 4 bits; recommended: 3). 8.9x per-session KV compression — the key benefit is multi-user serving: ~9x more concurrent sessions in the same RAM. Combine with `--pread --cache-mb 2048` for minimal footprint: 35B MoE + 64K context = 8.4 GB/session (vs 26 GB with FP32 KV).
- `--no-prefill` — disable batch prefill
- `--gpu` — enable GPU inference (requires `BN_ENABLE_GPU=1` build)
- `-t N` — thread count

## WASM Specifics

- WASM builds use `platform_load_buffer()` (no mmap)
- Maximum 2 GB WASM memory — constrains model + KV cache size
- `wasm/api.c` uses global state (single model instance)
- All WASM-exported functions must be listed in `wasm/build.sh`
