# CLAUDE.md

Instructions for Claude Code when working on this project.

## Project Overview

bitnet.c is a C11 GGUF inference engine for dense, MoE, and hybrid
SSM/attention LLMs. It is CPU-first, with optional Metal and wgpu-native WebGPU
backends. The current architecture separates model anatomy, quant formats,
backend-resident state, transformer planning, CPU execution, GPU op emission,
KV/logits helpers, and generation APIs.

## Build And Test

```bash
make clean
make bitnet
make test

make debug
make asan
make avx2-check

make fetch-wgpu
make BN_ENABLE_WEBGPU=1 bitnet test_gpu_wgpu
make BN_ENABLE_METAL=1 bitnet test_coherence
```

`BN_ENABLE_GPU=1` is a compatibility alias for WebGPU. Prefer
`BN_ENABLE_WEBGPU=1` in new docs and commands.

Individual tests include:

```bash
make test_architecture
make test_backend_matrix
make test_model_matrix
make test_gguf
make test_quant
make test_tokenizer
make test_transformer
make test_generate
make test_session
make test_prompt_cache
make test_threadpool
make test_safety
make test_arena
make test_ssm
make test_gguf_fuzz
make test_moe
make test_qwen36
make test_gemma4
make test_turboquant
make test_gpu_backend
```

Coherence tests require a real GGUF model:

```bash
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal

make BN_ENABLE_WEBGPU=1 test_coherence
./test_coherence models/model.gguf --webgpu

make test_coherence
./test_coherence models/model.gguf
```

## Architecture Boundaries

Modules are organized to avoid circular dependencies and cross-product branches:

1. `platform` — mmap/buffer abstraction, timing
2. `gguf` — GGUF parser
3. `quant` — format metadata, dequantization, CPU kernels, backend capability declarations
4. `turboquant` — compressed KV support
5. `model_arch` — model-family rules and tensor-role mapping
6. `model` — config, immutable CPU-visible weights, model loading
7. `backend_layout` / `backend_model` — backend-owned uploads, packed/fused layouts, backend session state
8. `tokenizer` — BPE tokenizer
9. `moe` — expert routing, loading, cache, and sparse FFN compute
10. `session` — per-request mutable KV, activations, SSM, and MoE scratch
11. `transformer` — planning and CPU/GPU execution
12. `sampler` — token sampling
13. `threadpool` — persistent pthread workers
14. `bn_alloc` — allocator vtable
15. `prompt_cache` — shared KV prefix cache
16. `generate` — library API, prefill/generate/chat/SSE/logprobs/stop strings
17. `gpu_wgpu` / `gpu_metal` — optional backend implementations
18. `main` — CLI wiring

Key transformer files:

| File | Responsibility |
|---|---|
| `src/transformer.c` | Top-level forward orchestration only. |
| `src/transformer/plan.c` | Layer/block plans and placement decisions. |
| `src/transformer/cpu.c` | CPU execution for attention, SSM, FFN, MoE, RoPE, residuals. |
| `src/transformer/gpu.c` | GPU-resident execution and CPU fallback boundaries. |
| `src/transformer/gpu_emit.c` | Emits backend-neutral `BnGPUOp` commands. |
| `src/transformer/kv.c` | FP32, FP16, TurboQuant KV helpers. |
| `src/transformer/logits.c` | CPU logits routing. |
| `src/transformer/prefill.c` | Batch prefill. |

## Ownership Rules

- `BnModel` is shared and immutable after load. It owns config, architecture
  metadata, CPU-visible weights, file state, thread pool, and shared MoE I/O.
- `BnSession` is per request. It owns KV cache, activations, SSM state, MoE
  scratch, and generation position.
- `BnBackendModel` and backend session state own GPU/backend-resident buffers,
  stacked QKV/gate-up/SSM layouts, fused buffers, activation buffers, and future
  CUDA/AVX-512 backend state.
- `BnQWeight`, `BnLayerWeights`, and `BnWeights` must not expose backend handles.
- `BnQuantFormatOps` owns quant block geometry, sizing, CPU hooks, repack/native
  layout support, split/fused capability, and backend capability metadata.
- `BnModelArchOps` owns model-family rules: tensor names, tensor roles,
  activation/norm choices, SSM/MoE/shared-expert rules, and architecture flags.
- Public GPU graph code uses `BnGPUOpKind`, `BnGPUOpCode`, and `BN_GPU_VALUE_*`.
  `BN_GPU_SHADER_*` IDs are backend-private in `src/gpu_shader.h`.

## Code Style

- C11, `-Wall -Wextra`, no GNU extensions unless already isolated behind a build guard.
- No external dependencies for CPU builds beyond libc/libm/pthread.
- Public functions use module prefixes.
- Internal helpers are `static`.
- Use `_init` / `_free` pairs for caller-owned structs. `BnSession` is created
  with `bn_session_create` and freed with `bn_session_free`.
- Keep platform-specific code behind feature macros such as `__EMSCRIPTEN__`,
  `BN_ENABLE_WEBGPU`, and `BN_ENABLE_METAL`.
- Avoid global mutable state in library modules. `main.c` and `wasm/api.c` are
  exceptions for application-level state.

## Common Tasks

- Add a model-family rule: update `include/model_arch.h`, then add synthetic
  architecture tests.
- Add a quant format: update `include/quant.h`, `src/quant/registry.c`, format
  kernels in `src/quant/`, Makefile sources, and quant capability tests.
- Add backend layout behavior: update `include/backend_layout.h` and
  `src/backend_layout.c`; keep model load backend-neutral.
- Modify transformer behavior: update the relevant plan/execution module under
  `src/transformer/`, not just `src/transformer.c`.
- Add GPU behavior: emit backend-neutral op codes in `gpu_emit.c`, then lower in
  `src/gpu_metal.m` or `src/gpu_wgpu.c`. Keep shader IDs private.
- Modify MoE expert dispatch: update `include/moe.h` and `src/moe.c`.
- Add a sampling strategy: extend `src/sampler.c`.
- Add a CLI flag: update `src/main.c` and docs.
- Export a WASM API: add the `EMSCRIPTEN_KEEPALIVE` wrapper in `wasm/api.c` and
  update `wasm/build.sh`.
- Integrate as a library: include `generate.h` and `session.h`, load a
  `BnModel`, create one `BnSession` per request, then call `bn_prefill` and
  `bn_generate`.

## Runtime Notes

- MoE I/O modes are mmap, `--pread --cache-mb N`, and experimental `--madvise`.
- `--maxseq` is important on GPU and large-context models because KV allocation
  follows the selected sequence cap.
- `--kv16` halves KV cache storage.
- `--kv-tq 2|3|4` enables TurboQuant compressed KV cache.
- `--draft PATH` enables speculative decoding with a same-tokenizer draft model.
- WebGPU depends on wgpu-native adapter availability. Runtime checks may skip on
  machines with no suitable adapter.
- Metal is macOS-only and uses system Metal/Foundation frameworks.

## WASM

WASM builds use `platform_load_buffer()` rather than mmap and are constrained by
wasm32 memory limits. `wasm/api.c` is allowed to use global application state for
the browser demo. Keep exported functions listed in `wasm/build.sh`.
