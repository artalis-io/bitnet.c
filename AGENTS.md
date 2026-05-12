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

**Dependency order**: platform -> gguf -> quant -> turboquant -> model_arch -> model -> backend_layout/backend_model -> tokenizer -> moe -> session -> transformer -> sampler -> threadpool -> bn_alloc -> prompt_cache -> generate -> gpu_wgpu/gpu_metal (optional) -> main

Architecture ownership rules:

- `BnModel` is shared and immutable after load. It owns model anatomy, CPU-visible weights, config, file state, pool, and shared MoE I/O.
- `BnSession` owns request-local mutable state: KV cache, activation buffers, MoE compute buffers, SSM state, and position.
- `BnBackendModel` / backend session state own GPU/backend-resident buffers, packed layouts, stacked/fused tensors, activation buffers, and future CUDA/AVX-512 backend state.
- `BnQuantFormatOps` in `include/quant.h` / `src/quant/registry.c` owns quant metadata and capability declarations. Add quant behavior there before adding backend-specific branches.
- `BnModelArchOps` in `include/model_arch.h` owns model-family shape, tensor-role, activation, norm, SSM, MoE, and naming rules.
- Public GPU graph code uses `BnGPUOpKind`, `BnGPUOpCode`, and `BN_GPU_VALUE_*`. Backend-private shader IDs stay in `src/gpu_shader.h` and backend implementations.

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
  - Phase 2: SIMD vs scalar matvec — layer 0 weights, max_diff < 2.0
  - Phase 3: GPU standalone matvec vs CPU scalar — layer 0 wq when available, max_diff < 2.0
  - Run Metal: `make BN_ENABLE_METAL=1 test_coherence && ./test_coherence model.gguf --metal`
  - Run WebGPU: `make BN_ENABLE_WEBGPU=1 test_coherence && ./test_coherence model.gguf --webgpu`

## Code Conventions

- C11 standard, no GNU extensions
- All public functions prefixed with module name: `gguf_open()`, `model_load()`, etc.
- Internal helpers are `static`
- Memory management: `_init`/`_free` pairs; caller owns the struct, module fills it. Exception: `BnSession` is heap-allocated via `bn_session_create`/`bn_session_free`.
- Error handling: return -1 or NULL on failure, print to stderr
- No global mutable state in library modules (only in main.c and wasm/api.c)
- **Model/Session split**: `BnModel` is shared and immutable after load. `BnSession` holds per-request mutable state. All forward pass and generation functions take both `BnModel *` and `BnSession *`.
- **Backend state split**: GPU handles and packed/fused backend layouts do not live on `BnQWeight` or `BnLayerWeights`; use `BnBackendModel` and backend session state.
- **GPU vtable**: `BnGPUBackend` in `include/gpu_backend.h` abstracts GPU compute. wgpu-native lives in `src/gpu_wgpu.c` and is gated by `BN_ENABLE_WEBGPU`; Metal lives in `src/gpu_metal.m` and is gated by `BN_ENABLE_METAL`. `BN_ENABLE_GPU` is a compatibility alias for WebGPU.
- **Transformer split**: `src/transformer.c` is top-level orchestration. Planning lives in `src/transformer/plan.c`, CPU execution in `src/transformer/cpu.c`, GPU orchestration/emission in `src/transformer/gpu.c` and `src/transformer/gpu_emit.c`, KV/logits/prefill helpers in `src/transformer/{kv,logits,prefill}.c`.
- Platform-specific code gated by `#ifdef __EMSCRIPTEN__`

## Performance Considerations

- CPU backends: NEON SDOT, AVX2, WASM SIMD128, scalar fallback. Optional GPU backends: Metal and WebGPU.
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
- `--metal` — enable Metal inference (requires `BN_ENABLE_METAL=1` build)
- `--webgpu` / `--gpu` — enable WebGPU inference (`--gpu` is compatibility spelling; requires `BN_ENABLE_WEBGPU=1` or `BN_ENABLE_GPU=1` build)
- `--maxseq N` — cap sequence length (recommended on GPU to limit KV cache VRAM)
- `-t N` — thread count

## WASM Specifics

- WASM builds use `platform_load_buffer()` (no mmap)
- Maximum 2 GB WASM memory — constrains model + KV cache size
- `wasm/api.c` uses global state (single model instance)
- All WASM-exported functions must be listed in `wasm/build.sh`

## GPU on WSL2

WSL2 lacks a native NVIDIA Vulkan ICD. GPU inference uses Mesa's **dzn** (Dozen) driver, which translates Vulkan to D3D12. Stock dzn is missing extensions wgpu-native requires (`VK_EXT_robustness2`, etc.), so a patch is provided in `patches/mesa-dzn-wgpu-compat.patch`.

**Setup:**
1. Install deps: `sudo apt-get install -y meson libdrm-dev libelf-dev llvm-dev libexpat1-dev directx-headers-dev ninja-build python3-mako libvulkan-dev && pip3 install --user meson`
2. Run `./patches/build-dzn.sh` (clones Mesa, applies patch, builds dzn — ~1 min)
3. Run with: `LD_LIBRARY_PATH=/usr/lib/wsl/lib VK_ICD_FILENAMES=/tmp/mesa-dzn/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json ./bitnet model.gguf --gpu --maxseq 4096`

**What the patch does:** Adds `VK_EXT_robustness2` + `VK_EXT_image_robustness` (D3D12 provides the guarantees natively), raises `maxStorageBufferRange` to 2GB-1, sets robustness2 alignment properties, and bumps the conformance version so wgpu accepts the adapter.

**Important:** Use `--maxseq` on GPU to cap KV cache size. Models with large context windows (e.g. 256K) will try to allocate far more VRAM than available without this flag.
