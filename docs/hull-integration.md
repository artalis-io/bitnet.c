# Hull Integration

This document describes the current integration boundary between Hull and
bitnet.c. Hull is not a dependency of this repository, and bitnet.c should not
include Hull headers.

## Boundary

```text
Hull app / HTTP / Lua / JS
  -> Hull LLM adapter
  -> bitnet.c public C API
  -> CPU, Metal, WebGPU, or future backend implementation
```

The stable bitnet.c side is:

- `BnModel` for shared immutable model state
- `BnSession` for per-request mutable state
- `bn_prefill` and `bn_generate` for inference
- tokenizer, sampler, stop-string, logprob, chat-format, and SSE helpers
- `BnPromptCache` for shared KV prefix caching
- `BnAllocator` for host-provided allocation
- `BnGPUBackend` for optional host/backend compute

Hull-specific protocol glue, HTTP routing, session pools, and WASM AoT adapters
belong in Hull.

## Current Backend Model

GPU handles no longer live on `BnQWeight` or `BnLayerWeights`. Backend-resident
state is isolated behind:

- `BnBackendModel`: uploaded weights, stacked QKV/gate-up/SSM buffers, norms,
  biases, tied embeddings, backend layout choices, and fallback metadata.
- backend session state: activation buffers, KV buffers, scratch, and other
  per-request backend resources.

This is the integration rule: model anatomy and CPU weights remain immutable;
backend layout and residency are backend-owned.

## GPU Contract

`include/gpu_backend.h` is the public GPU contract. The transformer emits
backend-neutral ops using semantic op kinds, op codes, and `BN_GPU_VALUE_*`
buffer IDs.

Backends lower those op codes privately:

- Metal: `src/gpu_metal.m`
- WebGPU: `src/gpu_wgpu.c`
- private shader mapping: `src/gpu_shader.h`

Do not build host integrations against `BN_GPU_SHADER_*`; those IDs are backend
implementation details.

## Suggested Hull Adapter Shape

Hull can expose its own vendor-neutral interface and implement a bitnet adapter:

```c
typedef struct {
    void *(*load)(const char *path, const HlLlmOpts *opts);
    void  (*unload)(void *model);
    void *(*session_create)(void *model);
    void  (*session_free)(void *session);
    int   (*tokenize)(void *model, const char *text, int *out, int max);
    int   (*prefill)(void *model, void *session, const int *tokens, int n);
    int   (*generate)(void *model, void *session, const HlGenOpts *opts);
} HlLlmBackend;
```

The bitnet adapter should translate Hull options to bitnet.c flags and keep file
descriptors or mapped memory alive for the lifetime of the loaded model. That
preserves mmap, pread, and madvise choices for MoE models.

## Streaming

bitnet.c provides formatting helpers, not an HTTP server:

- `bn_format_sse_chunk`
- `bn_format_sse_done`

Hull should own response writing, backpressure, cancellation, and request
lifetime. The token callback from `bn_generate` can format SSE chunks and pass
them to Hull's response writer.

## WASM

The repository includes a browser-oriented Emscripten build in `wasm/`. A Hull
WASM AoT adapter should live in Hull because the ABI, host calls, shared data
segments, and callback protocol are Hull-specific.

The bitnet.c side should remain ordinary C library code using
`platform_load_buffer()` under Emscripten.

## Integration Checklist

1. Load the GGUF and keep backing storage alive until unload.
2. Create one shared `BnModel`.
3. Create one `BnSession` per concurrent request.
4. Optionally create and bind a `BnGPUBackend`.
5. Use `BnPromptCache` only when requests can safely share prefix KV state.
6. Stream via token callbacks and SSE formatting helpers.
7. Free sessions before freeing the model.
8. Keep backend-specific shader IDs and buffer layouts out of Hull's public API.
