# Inference Pipeline

This document describes the current inference path after the transformer split.
It is a code map and execution overview, not a line-by-line math tutorial.

## High-Level Flow

```text
prompt text
  -> tokenizer
  -> optional batch prefill
  -> per-token transformer forward
  -> logits
  -> sampler
  -> decoded token text
```

Public entry points are in `include/generate.h` and `include/transformer.h`.
The CLI wiring is in `src/main.c`; the library orchestration is in
`src/generate.c`.

## Model And Session Ownership

`BnModel` is shared and immutable after load. It owns configuration, CPU-visible
weights, model architecture metadata, tokenizer-visible metadata, optional MoE
I/O state, and a `BnBackendModel` pointer for backend-resident state.

`BnSession` is per request. It owns mutable activation buffers, KV cache, MoE
scratch, SSM state, and position. A backend may also attach per-session
activation/KV buffers through backend session state.

This split is required for concurrent serving: many sessions can share one
loaded model without mutating model anatomy or CPU weights.

## Transformer Modules

| File | Responsibility |
|---|---|
| `src/transformer.c` | Token bounds, embedding setup, CPU/GPU top-level route, final timing. |
| `src/transformer/plan.c` | Layer/block shape planning and backend placement decisions. |
| `src/transformer/cpu.c` | CPU layer execution for attention, SSM, dense FFN, MoE, RoPE, residuals, and backend dispatch calls. |
| `src/transformer/gpu.c` | GPU-resident forward orchestration and deterministic CPU fallback boundaries. |
| `src/transformer/gpu_emit.c` | Emits backend-neutral `BnGPUOp` commands for attention, SSM, dense FFN, MoE, and logits. |
| `src/transformer/kv.c` | FP32, FP16, and TurboQuant KV row/write helpers. |
| `src/transformer/logits.c` | CPU logits routing for tied and untied output weights. |
| `src/transformer/prefill.c` | Batch prefill path. |
| `src/transformer/*_{neon,avx2,wasm,scalar}.c` | ISA-specific kernels for RMSNorm, GQA, TurboQuant GQA, SSM, and logits. |

The intended layering is plan first, then execute. New architecture, quant, and
backend features should be represented in plans and capability checks before
they appear as execution branches.

## Planning

The planner turns model configuration and layer weights into explicit decisions:

- attention layer versus SSM layer
- compact attention/SSM indices
- Q/K/V shape, gated Q, wide Q, Q/K norms, and bias presence
- KV mode: FP32, FP16, or TurboQuant
- dense FFN versus MoE versus shared expert variants
- activation: SiLU, ReLU2, or architecture-specific choices
- logits layout: tied embedding or output tensor
- backend placement and fallback reason

Model-family rules come from `BnModelArchOps` in `include/model_arch.h`.
Quant-format behavior comes from `BnQuantFormatOps` in `include/quant.h` and
`src/quant/registry.c`.

## CPU Execution

The CPU path consumes the plan and calls the quant dispatch layer rather than
switching directly on every model/quant/backend combination. The hot operations
are still the expected transformer pieces:

1. token embedding into `session->x`
2. per-layer RMSNorm
3. Q/K/V projection
4. RoPE or architecture-specific positional handling
5. KV cache write
6. GQA or SSM block
7. output projection and residual
8. dense FFN or MoE block
9. final RMSNorm and logits

Quantized matvec and batch matvec dispatch live under `src/quant/dispatch.c`.
Format-specific kernels live under `src/quant/`.

## GPU Execution

The public GPU contract is `include/gpu_backend.h`.

GPU execution uses:

- `BnGPUOpKind` for semantic operation kind
- `BnGPUOpCode` for backend-neutral concrete operation
- `BN_GPU_VALUE_*` for graph-visible buffers
- `BnBackendModel` for uploaded weights and packed/fused backend layouts
- backend-private shader IDs in `src/gpu_shader.h`

`src/transformer/gpu_emit.c` emits op-code-based commands. Metal and WebGPU lower
those op codes to their own shader pipelines in `src/gpu_metal.m` and
`src/gpu_wgpu.c`. Public headers do not expose `BN_GPU_SHADER_*`.

If a backend cannot execute a planned block, the path falls back to CPU at a
deterministic boundary. Fallback decisions should be visible in tests and debug
output, not hidden inside the math loop.

## KV Cache Modes

| Mode | Purpose |
|---|---|
| FP32 | Reference path and highest precision. |
| FP16 (`--kv16`) | Half-size KV cache for lower memory pressure. |
| TurboQuant (`--kv-tq`) | Compressed KV cache for long-context and multi-session serving. |

KV helpers are isolated in `src/transformer/kv.c` so future backends can own
their session-resident KV buffers without changing model weights.

## Prefill And Generation

Prefill processes prompt tokens while avoiding unnecessary logits work for
intermediate prompt positions. The implementation lives in
`src/transformer/prefill.c` and is called through `bn_prefill`.

Decode is autoregressive: each new token runs one forward pass, then the sampler
chooses the next token from logits. Samplers live in `src/sampler.c` and support
greedy, temperature sampling, top-p, and repetition penalty.

## Validation Hooks

Use these gates when changing the inference path:

```bash
make test_transformer
make test
make clean
make bitnet
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal
```

For WebGPU-specific work:

```bash
make fetch-wgpu
make BN_ENABLE_WEBGPU=1 test_gpu_wgpu
```
