# bitnet.c / Hull Integration Roadmap

How bitnet.c becomes Hull's LLM inference backend — without either project depending on the other.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Hull Application (Lua / JS)                                │
│  app.post("/v1/chat/completions", function(req, res) ... )  │
├─────────────────────────────────────────────────────────────┤
│  Hull LLM Capability (cap/llm.c)                           │
│  HlLlmBackend vtable — vendor-independent                  │
│  Session pool, SSE streaming, prompt caching                │
├──────────────────────┬──────────────────────────────────────┤
│  llm_bitnet.c        │  llm_llamacpp.c / llm_remote.c      │
│  (adapter)           │  (other backends, same vtable)       │
├──────────────────────┴──────────────────────────────────────┤
│  bitnet.c (pure C library, zero Hull dependency)            │
│  BnModel / BnSession / bn_generate / BnPromptCache          │
│  BnGPUBackend vtable — runtime-filled by caller             │
│  WGSL shaders — portable, no GPU runtime dependency         │
├─────────────────────────────────────────────────────────────┤
│  Compute backends (filled by Hull or any other host)        │
│  CPU SIMD (NEON/AVX2/WASM) ← built-in, always available    │
│  WebGPU (WGSL shaders)     ← optional, via BnGPUBackend    │
│  WASM AoT                  ← fallback, via Hull's WAMR     │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **bitnet.c knows nothing about Hull.** No Hull headers, no Hull types, no link dependency. It's a pure C library with vtable extension points.
2. **Hull knows nothing about bitnet.c internals.** It calls the `HlLlmBackend` vtable. Swapping to llama.cpp or a remote API is one line.
3. **Hull doesn't limit bitnet.c's I/O strategy.** The file descriptor and mmap pointer are passed through, so bitnet.c can use mmap, pread+LRU, or madvise for MoE expert loading.
4. **GPU is optional.** bitnet.c's CPU SIMD backends are always available. If Hull provides a `BnGPUBackend` implementation, GPU acceleration kicks in. If not, nothing changes.
5. **WASM AoT is the fallback.** If GPU is unavailable and native linking isn't possible, Hull can run bitnet.c as a WASM AoT module via WAMR — same code, ~1.2x native overhead.

## Phases

### Phase 1: SSE Formatter (bitnet.c)

Add OpenAI-compatible SSE chunk formatting to the library API. No HTTP dependency — just string formatting.

**Files:**
- `include/generate.h` — declare `bn_format_sse_chunk`
- `src/generate.c` — implement

**API:**
```c
// Format a token piece into an OpenAI-compatible SSE data line.
// Writes to buf, returns bytes written. buf must be at least piece_len + 128.
// id: request ID string (may be NULL). model: model name (may be NULL).
// finish_reason: NULL for normal chunks, "stop"/"length" for final chunk.
int bn_format_sse_chunk(char *buf, int buf_size,
                        const char *piece, const char *id,
                        const char *model, const char *finish_reason);
```

Output format:
```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"bitnet-2b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

```

**Why in bitnet.c:** Any host (Hull, standalone server, Node addon) that wants OpenAI-compatible streaming needs this. It's a utility function, not a server feature.

**Test:** `test/test_generate.c` — verify output matches OpenAI spec for normal chunks, empty content, and finish chunks.

### Phase 2: Hull WASM AoT Adapter (bitnet.c)

Add a WASM entry point that follows Hull's `hull_process` ABI convention. This enables bitnet.c to run as a Hull compute module when native linking or GPU isn't available.

**Files:**
- `wasm/hull_adapter.c` — `hull_process` entry point
- Update `wasm/build.sh` — optional `--hull` flag to build the adapter

**ABI:**
```c
// Hull calls this with a binary request, gets a binary response.
// Input:  [4-byte op] [payload]
// Ops:    0x01 = load (path), 0x02 = generate (tokens + params), 0x03 = tokenize
// Output: [4-byte status] [payload]
int hull_process(const uint8_t *in, int in_len, uint8_t *out, int out_max);
```

Binary protocol (not JSON) because this is a hot path — no parsing overhead. Hull's `llm_bitnet.c` adapter handles JSON↔binary translation on the Lua side.

**Streaming via host_call:** For token-by-token streaming, the WASM module calls `host_call(OP_CALLBACK, chunk_id, data, len)` for each token. Hull's Lua handler receives these as callback events and streams them via SSE.

**Why in bitnet.c:** The adapter is bitnet.c-specific (it calls bitnet.c's API). It ships with bitnet.c but is only compiled when targeting WASM. No Hull headers needed — just the ABI convention (a function signature).

### Phase 3: GPU Backend Vtable (bitnet.c)

Define a GPU compute abstraction that any host can fill in. bitnet.c calls it for matvec; the host implements it with whatever GPU API it has.

**Files:**
- `include/gpu_backend.h` — vtable definition
- `src/gpu_dispatch.c` — dispatch logic (GPU if available, else CPU SIMD)

**Vtable:**
```c
typedef struct {
    // Upload weight tensor to GPU. Returns opaque buffer handle.
    void *(*buffer_create)(void *ctx, const void *data, size_t size);
    void  (*buffer_destroy)(void *ctx, void *buffer);

    // Quantized matvec: out[rows] = W[rows, cols] @ x[cols]
    // W is a GPU buffer handle (from buffer_create).
    // x and out are host pointers (copied to/from GPU internally).
    // type: BN_GGUF_TENSOR_* constant for the weight format.
    int (*matvec)(void *ctx, float *out, void *W_buf, const float *x,
                  int rows, int cols, int type);

    // Batch matvec for prefill: out[n_tokens, rows] = W @ X[n_tokens, cols]
    int (*matmul)(void *ctx, float *out, void *W_buf, const float *X,
                  int rows, int cols, int n_tokens, int type);

    void *ctx;  // opaque backend context (e.g., Hull's HlGpuCtx)
} BnGPUBackend;
```

**Integration in forward pass:**
```c
// In bn_quant_matvec (dispatch.c):
if (W->gpu_buf && gpu_backend) {
    gpu_backend->matvec(gpu_backend->ctx, out, W->gpu_buf, x, W->rows, W->cols, W->type);
    return;
}
// ... existing CPU SIMD dispatch
```

**Weight upload:**
```c
// New function: upload model weights to GPU
int bn_model_upload_weights(BnModel *model, BnGPUBackend *gpu);

// Sets W->gpu_buf for each weight tensor. CPU SIMD remains the fallback
// if any weight fails to upload or gpu is NULL.
```

**Why in bitnet.c:** The vtable is bitnet.c's contract. It says "if you give me a GPU that can do matvec, I'll use it." bitnet.c doesn't know it's wgpu, Metal, Vulkan, or a mock. The vtable lives next to the existing SIMD dispatch — it's just another backend tier.

### Phase 4: WGSL Compute Shaders (bitnet.c)

Write portable WGSL shaders for the hot-path quant types. These ship with bitnet.c but are compiled by the host.

**Files:**
```
shaders/
├── i2s_matvec.wgsl      # I2_S ternary (BitNet models)
├── q4_matvec.wgsl        # Q4_0 standard 4-bit
├── q4k_matvec.wgsl       # Q4_K k-quant 4-bit
├── q6k_matvec.wgsl       # Q6_K k-quant 6-bit
└── common.wgsl           # shared: hsum, block decode helpers
```

**Shader structure (example: I2_S):**
```wgsl
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms { rows: u32, cols: u32, scale: f32 }
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.rows) { return; }
    // Decode I2_S 2-bit packed weights, dot with x, write out[row]
}
```

**Why in bitnet.c:** The shaders are tied to bitnet.c's weight formats (I2_S block layout, Q4_K scale encoding, etc.). They're not Hull-specific — any WebGPU host can compile them. Keeping them in bitnet.c means they're versioned with the quant format definitions they depend on.

### Phase 5: Hull LLM Capability (Hull repo)

Hull's side: a vendor-independent LLM capability module.

**Files (in Hull):**
```
hull/
├── include/hull/cap/llm.h      # HlLlmBackend vtable + HlMappedFile
├── src/hull/cap/llm.c          # Session pool, SSE streaming, cache orchestration
├── src/hull/cap/llm_bitnet.c   # bitnet.c adapter (implements HlLlmBackend)
├── src/hull/cap/llm_gpu.c      # Fills BnGPUBackend from Hull's HlGpuCtx
└── vendor/bitnet/               # git subtree of bitnet.c
```

**Hull's vtable:**
```c
typedef struct {
    void *(*load)(const HlMappedFile *file, const HlLlmOpts *opts, KlAllocator *alloc);
    void  (*unload)(void *model);

    void *(*session_create)(void *model, KlAllocator *alloc);
    void  (*session_free)(void *session, KlAllocator *alloc);
    void  (*session_reset)(void *session, void *model);

    int (*tokenize)(void *model, const char *text, int *out, int max);
    const char *(*detokenize)(void *model, int token_id);

    int (*prefill)(void *model, void *session, const int *tokens, int n, int pos);
    int (*step)(void *model, void *session, int token, int pos, float temp, float topp);

    int (*vocab_size)(void *model);
    int (*context_length)(void *model);
    int (*eos_token)(void *model);
} HlLlmBackend;

typedef struct {
    const void *data;
    size_t size;
    int fd;
    int is_mmap;
} HlMappedFile;
```

**HlMappedFile preserves all I/O capabilities:**
- `data` + `is_mmap` → bitnet.c can use mmap path (direct pointer access)
- `fd` → bitnet.c can use pread+LRU cache for MoE expert streaming
- `is_mmap` + `data` → bitnet.c can use madvise for prefetch hints
- Hull opens the file, keeps fd alive until unload — bitnet.c's I/O strategy is unrestricted

**Lua API:**
```lua
local llm = require("hull.llm")

-- Load with full I/O control
llm.load("models/qwen3-30b.gguf", {
    threads = 8,
    gpu = true,          -- upload weights to GPU if available
    io_mode = "auto",    -- "mmap" | "pread" | "madvise" | "auto"
    cache_mb = 4096,     -- MoE expert LRU cache budget
    kv_f16 = true,       -- FP16 KV cache
    prompt_cache = true, -- enable shared KV prefix caching
})

-- OpenAI-compatible endpoint
app.post("/v1/chat/completions", function(req, res)
    local body = json.decode(req.body)
    if body.stream then
        res:header("Content-Type", "text/event-stream")
        llm.chat(body.messages, {
            max_tokens = body.max_tokens,
            temperature = body.temperature,
            stop = body.stop,
            on_token = function(chunk) res:write(chunk) end,
        })
        res:write("data: [DONE]\n\n")
    else
        local result = llm.chat(body.messages, {
            max_tokens = body.max_tokens,
            temperature = body.temperature,
        })
        res:json(result)
    end
end)
```

**Session pool:** `llm.c` pre-creates N sessions. On request: check out a session, restore prompt cache if applicable, prefill, generate with streaming, check session back in. Sessions are reset between requests unless prompt cache allows reuse.

**GPU integration:** `llm_gpu.c` creates a `BnGPUBackend` that routes `matvec` calls to Hull's `hl_cap_gpu_dispatch()`. On model load, it compiles bitnet.c's WGSL shaders via `hl_cap_gpu_compile()` and uploads weights via `hl_cap_gpu_buffer_create()`. This fills in `BnGPUBackend` and passes it to `bn_model_upload_weights()`.

**WASM AoT fallback:** If GPU isn't available and native bitnet.c linking isn't possible (e.g., cross-platform distribution), Hull falls back to running bitnet.c as a WASM AoT module via `compute.call("bitnet", ...)`. Same code, sandboxed, ~1.2x overhead. The `llm_bitnet.c` adapter detects this and routes through the WASM path instead of direct C calls.

## Implementation Order

### In bitnet.c (this repo)

| # | Phase | Effort | Depends on |
|---|-------|--------|------------|
| 1 | SSE chunk formatter | Small | Nothing |
| 2 | Hull WASM AoT adapter | Small | Nothing |
| 3 | GPU backend vtable | Medium | Nothing |
| 4 | WGSL compute shaders | Large | Phase 3 |

### In Hull

| # | Phase | Effort | Depends on |
|---|-------|--------|------------|
| 5 | `HlLlmBackend` vtable + `llm.c` | Medium | Nothing (can start in parallel) |
| 6 | `llm_bitnet.c` adapter | Small | Phase 5 + bitnet.c Phase 1 |
| 7 | `llm_gpu.c` (BnGPUBackend → HlGpuCtx) | Medium | Phase 5 + bitnet.c Phase 3-4 |
| 8 | WASM AoT fallback path | Small | Phase 5 + bitnet.c Phase 2 |

Phases 1-4 (bitnet.c) and Phase 5 (Hull) can proceed in parallel. The adapter (Phase 6) is the integration point.

## What Exists Today

| Component | Status |
|-----------|--------|
| `BnModel` / `BnSession` split | Done |
| `BnPromptCache` (shared KV prefix) | Done |
| `BnAllocator` (compatible with `KlAllocator`) | Done |
| `bn_token_callback` (streaming) | Done |
| `bn_format_sse_chunk` | Not started |
| WASM build (`wasm/api.c`) | Done (not Hull ABI) |
| `BnGPUBackend` vtable | Not started |
| WGSL shaders | Not started |
| Hull `HlLlmBackend` vtable | Not started |
| Hull `llm_bitnet.c` adapter | Not started |
