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

**Dependency order**: platform → gguf → quant → model → tokenizer → moe → transformer → sampler → threadpool → main

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

## Code Conventions

- C11 standard, no GNU extensions
- All public functions prefixed with module name: `gguf_open()`, `model_load()`, etc.
- Internal helpers are `static`
- Memory management: `_init`/`_free` pairs; caller owns the struct, module fills it
- Error handling: return -1 or NULL on failure, print to stderr
- No global mutable state in library modules (only in main.c and wasm/api.c)
- Platform-specific code gated by `#ifdef __EMSCRIPTEN__`

## Performance Considerations

- 4 SIMD backends: NEON SDOT, AVX2, WASM SIMD128, scalar fallback (auto-selected at compile time)
- Q8_K x quantization for Q4_K/Q6_K: integer accumulation, unsigned nibbles, bsums correction
- MoE expert LRU cache with open-addressing hash + intrusive LRU list (pread mode)
- Atomic work-stealing thread dispatch for load balancing
- Batch prefill with fused Q4_K matmul kernel (dense models)
- KV cache pre-allocated for max sequence length

## CLI Flags (key ones for agents)

- `--pread` — force pread for MoE expert loading (lower RSS)
- `--cache-mb N` — expert LRU cache budget in MB (default 4096, pread only)
- `--madvise` — madvise-guided mmap (experimental)
- `--draft <path>` — draft model for speculative decoding (greedy, same tokenizer)
- `--draft-k N` — draft tokens per iteration (default 5)
- `--flash` — flash attention (online softmax)
- `--kv16` — FP16 KV cache
- `--no-prefill` — disable batch prefill
- `-t N` — thread count

## WASM Specifics

- WASM builds use `platform_load_buffer()` (no mmap)
- Maximum 2 GB WASM memory — constrains model + KV cache size
- `wasm/api.c` uses global state (single model instance)
- All WASM-exported functions must be listed in `wasm/build.sh`
