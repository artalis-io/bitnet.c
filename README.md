# bitnet.c

Small C11 inference engine for GGUF LLMs, with a bias toward readable module
boundaries, CPU correctness, and optional GPU execution through explicit backend
interfaces.

The current codebase is no longer a single flat transformer loop. Model anatomy,
quant formats, backend-resident state, transformer planning, CPU execution, GPU
op emission, KV helpers, logits, and prefill now live in separate modules.

## What It Supports

- GGUF v3 model loading for dense, MoE, and hybrid SSM/attention families used by
  current Qwen, BitNet, Llama-style, Gemma-family, and related GGUF exports.
- Quantized CPU inference for `I2_S`, `TQ1_0`, `TQ2_0`, `Q4_0`, `Q4_1`,
  `Q8_0`, `Q2_K` through `Q8_K`, `IQ2` through `IQ4`, `F16`, `BF16`, and `F32`.
- CPU backends for scalar, ARM NEON, x86 AVX2, and WASM SIMD where kernels exist.
- Optional native Metal and wgpu-native WebGPU backends.
- MoE expert routing with mmap, pread, and expert LRU cache modes.
- Hybrid SSM/attention execution, including CPU fallback for backend gaps.
- Per-request `BnSession` state with shared immutable `BnModel`.
- Prompt cache, stop strings, logprobs, chat formatting, SSE formatting, batch
  prefill, FP16 KV, and TurboQuant KV compression.

CUDA and AVX-512 are architectural targets, not implemented backends yet.

## Build

```bash
make clean
make bitnet
make test
```

Optional backends:

```bash
# Native Metal on macOS
make BN_ENABLE_METAL=1 bitnet test_coherence

# wgpu-native WebGPU
make fetch-wgpu
make BN_ENABLE_WEBGPU=1 bitnet test_gpu_wgpu
```

`BN_ENABLE_GPU=1` is accepted as a compatibility alias for WebGPU.

## Run

```bash
./bitnet models/model.gguf -p "The capital of France is" -n 64 -t 8
./bitnet models/model.gguf --metal -p "Hello" -n 64
./bitnet models/model.gguf --webgpu --maxseq 4096 -p "Hello" -n 64
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--pread` | Stream MoE expert weights with `pread` instead of relying on mmap. |
| `--cache-mb N` | MoE expert LRU cache budget for pread mode. |
| `--madvise` | Experimental mmap prefetch hints. |
| `--flash` | Use online-softmax attention where supported. |
| `--kv16` | Store KV cache in FP16. |
| `--kv-tq 2|3|4` | Use TurboQuant compressed KV cache. |
| `--draft PATH` | Use speculative decoding with a draft model. |
| `--no-prefill` | Disable batch prefill. |
| `--maxseq N` | Cap context allocation, especially important on GPU. |

## Architecture

The important ownership split is:

```text
GGUF parser       -> raw tensor metadata and bytes
model anatomy     -> architecture rules and tensor roles
model weights     -> immutable CPU-visible weights
quant registry    -> format sizing, kernels, layout capabilities
backend model     -> uploaded weights and backend-specific packed layouts
backend session   -> activation/KV buffers and per-request backend state
transformer plan  -> backend-neutral layer/block decisions
backend lowerer   -> CPU, Metal, WebGPU, future CUDA/AVX-512 execution
```

Key modules:

| Area | Files |
|---|---|
| Public model/session API | `include/model.h`, `include/session.h` |
| Backend-owned model/session state | `include/backend_model.h`, `src/backend_model.c` |
| Backend layout/upload choices | `include/backend_layout.h`, `src/backend_layout.c` |
| Architecture-family rules | `include/model_arch.h` |
| Quant registry and kernels | `include/quant.h`, `src/quant/registry.c`, `src/quant/*` |
| Transformer orchestration | `src/transformer.c` |
| Transformer planning | `src/transformer/plan.c` |
| CPU execution | `src/transformer/cpu.c` |
| GPU execution and op emission | `src/transformer/gpu.c`, `src/transformer/gpu_emit.c` |
| KV/logits/prefill helpers | `src/transformer/kv.c`, `src/transformer/logits.c`, `src/transformer/prefill.c` |
| GPU contract | `include/gpu_backend.h` |
| Backend-private shader lowering | `src/gpu_shader.h`, `src/gpu_metal.m`, `src/gpu_wgpu.c` |

`BnModel` is intended to be shared and immutable after load. It does not expose
GPU handles on weights. Backend handles, stacked buffers, repacked layouts, and
backend graph state live behind `BnBackendModel`; per-request activation and KV
state lives in `BnSession` and backend session state.

`BnGPUOp` is a backend-neutral command contract. Public code uses semantic op
kinds, op codes, and `BN_GPU_VALUE_*` graph values. Metal and WebGPU lower those
op codes to backend-private shader IDs internally.

## Library Shape

Minimal generation flow:

```c
#include "generate.h"
#include "gguf.h"
#include "model.h"
#include "session.h"

BnModel model;
/* open GGUF, load model, create thread pool; see main.c for complete wiring */

BnSession *session = bn_session_create(&model, NULL);
bn_prefill(&model, session, tokens, n_tokens, session->pos, 0);
session->pos += n_tokens;
bn_generate(&model, session, &last_token, &sampler, 128,
            &session->pos, callback, user_data, NULL, NULL);

bn_session_free(session, NULL);
bn_model_free(&model);
```

Different sessions can run independently against the same loaded model. A single
session is not thread-safe by itself.

## Validation

Common gates:

```bash
make test
make clean
make bitnet
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal
make BN_ENABLE_WEBGPU=1 test_gpu_wgpu
./test/backend_matrix.sh
```

`test_coherence` compares GPU versus CPU token generation, SIMD versus scalar
matvecs, and standalone GPU matvecs versus scalar CPU where the selected model
has the needed tensors.

## Performance Notes

Performance is model, quant, backend, page-cache, and context dependent. Treat
the numbers in `docs/benchmarks.md` as local checkpoints, not global claims.

Current strongest areas:

- CPU MoE serving with warm expert working sets.
- Quantized CPU matvecs for the formats used by tested GGUF models.
- Metal/WebGPU graph execution for supported dense and hybrid paths, with clear
  CPU fallback for unsupported blocks.

Current weak or unfinished areas:

- CUDA is not implemented.
- AVX-512 placement exists in architecture tests, but kernels and benchmarks are
  still future work.
- WebGPU availability depends on the adapter and wgpu-native platform support.
- Some hybrid SSM/MoE GPU paths still fall back to CPU when a backend capability
  is missing.

## Docs

| Document | Purpose |
|---|---|
| `docs/inference.md` | Current inference pipeline and module map. |
| `docs/transformer-behavior-map.md` | Planner/executor behavior that should remain visible in tests. |
| `docs/benchmarks.md` | Reproducible benchmark commands and qualified local results. |
| `docs/coherence.md` | GPU/CPU/SIMD coherence test policy. |
| `docs/hull-integration.md` | Integration boundary for Hull or any host embedding bitnet.c. |
| `docs/roadmap.md` | Architecture, backend, quant, and model-family roadmap. |

## WASM

The WASM target is maintained in `wasm/` and uses `platform_load_buffer()` rather
than mmap. It is constrained by wasm32 memory limits and browser/runtime support
for SIMD and shared memory.

```bash
./wasm/build.sh
```

## License

[MIT](LICENSE)
