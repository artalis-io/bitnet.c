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

**API (implemented):**
```c
// Format a token piece into an OpenAI-compatible SSE data line.
// Returns bytes written (excluding NUL), or -1 on insufficient buffer.
int bn_format_sse_chunk(char *buf, int buf_size,
                        const char *piece, const char *id,
                        const char *model, const char *finish_reason,
                        long long created);

// Format the SSE stream terminator: "data: [DONE]\n\n"
int bn_format_sse_done(char *buf, int buf_size);
```

Output format:
```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"bitnet-2b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

```

**Why in bitnet.c:** Any host (Hull, standalone server, Node addon) that wants OpenAI-compatible streaming needs this. It's a utility function, not a server feature.

**Test:** `test/test_generate.c` — verify output matches OpenAI spec for normal chunks, empty content, and finish chunks.

### Phase 2: Hull WASM AoT Adapter (Hull repo — NOT bitnet.c)

The WASM adapter implements Hull's `hull_process(in_ptr, in_len, out_ptr, out_max)` ABI and uses Hull-specific conventions: shared data segments for zero-copy model loading (`host_call(OP_DATA_INFO, ...)`), streaming via `host_call(OP_CALLBACK, ...)`, and Hull's binary op protocol.

**This adapter lives in Hull, not bitnet.c.** It is 100% Hull's ABI — it knows Hull's entry point signature, Hull's host_call opcodes, Hull's shared heap mechanism. Nothing in it is reusable by other WASM hosts. bitnet.c's library API is the integration surface; the adapter is Hull's glue code that calls it.

**Files (in Hull):**
```
hull/
├── src/hull/cap/llm_bitnet_wasm.c   # hull_process adapter (compiled to WASM)
└── build/                            # wamrc AOT compilation of the above
```

**Protocol:** Binary, little-endian (native WASM byte order). First byte is op code.

| Op | Name | Input payload | Output payload | Streaming |
|----|------|---------------|----------------|-----------|
| `0x01` | INIT | `{u32 max_seq_len, u32 kv_f16, f32 temp, f32 topp, u64 seed}` | `{u32 status, u32 vocab_size, u32 seq_len}` | No |
| `0x02` | GENERATE | `{u32 n_tokens, u32 max_gen, f32 temp, f32 topp, u32 n_stop, i32 tokens[], ...stop strings}` | `{u32 n_generated}` | Yes — SSE chunks via host_call |
| `0x03` | TOKENIZE | `{u32 text_len, u8 text[]}` | `{u32 n_tokens, i32 tokens[]}` | No |
| `0x04` | RESET | (empty) | `{u32 status}` | No |

**Model loading:** Via Hull's shared data segments (zero-copy). Hull Lua does `compute.data("bitnet", "model", fs.mmap("model.gguf"))`, then the INIT handler queries the segment address via `host_call(OP_DATA_INFO, 0, 0)` and calls `bn_gguf_open()` on the pointer.

**Streaming:** During GENERATE, the adapter formats each token as an SSE chunk using `bn_format_sse_chunk()` (from bitnet.c's library API), then calls `host_call(OP_CALLBACK, sse_ptr, sse_len)`. Hull's Lua handler writes chunks verbatim to the HTTP response.

**Why in Hull:** The adapter is tightly coupled to Hull's ABI conventions. A different WASM host (Wasmer, Wasmtime, Node WASI) would need a completely different adapter. bitnet.c's responsibility ends at the library API.

### Phase 3: GPU Backend Vtable (bitnet.c)

Define a GPU compute abstraction that any host can fill in. bitnet.c calls it for matvec; the host implements it with whatever GPU API it has.

**Files (implemented):**
- `include/gpu_backend.h` — vtable definition
- `include/quant.h` — `void *gpu_buf` on `BnQWeight`, `bn_quant_matvec_gpu`, `bn_quant_matvec_batch_gpu`
- `include/model.h` — `BnGPUBackend *gpu` on `BnModel`, `bn_model_upload_weights`, `bn_model_release_gpu`
- `src/quant/dispatch.c` — GPU-aware dispatch functions
- `src/transformer.c` — all 19 matvec call sites wired to GPU-aware variants

**Vtable (implemented):**
```c
typedef struct {
    void *(*buffer_create)(void *ctx, const void *data, size_t size,
                           int type, int rows, int cols);
    void  (*buffer_destroy)(void *ctx, void *buffer);
    int (*matvec)(void *ctx, float *out, void *W_buf, const float *x,
                  int rows, int cols, int type);
    int (*matmul)(void *ctx, float *out, void *W_buf, const float *X,
                  int rows, int cols, int n_tokens, int type);  // optional (NULL ok)
    void *ctx;
} BnGPUBackend;
```

**Dispatch (implemented):**
```c
// GPU-aware matvec: tries GPU first, falls back to CPU SIMD
bn_quant_matvec_gpu(out, &W, x, x_q, pool, model->gpu);
bn_quant_matvec_batch_gpu(tasks, n, x, x_q, pool, model->gpu);

// Weight lifecycle
bn_model_upload_weights(model, gpu);  // uploads all BnQWeight tensors
bn_model_release_gpu(model);          // destroys all GPU buffers
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

| # | Phase | Effort | Depends on | Status |
|---|-------|--------|------------|--------|
| 1 | SSE chunk formatter | Small | Nothing | **Done** |
| 3 | GPU backend vtable | Medium | Nothing | **Done** |
| 4 | WGSL compute shaders | Large | Phase 3 | Not started |

### In Hull

| # | Phase | Effort | Depends on | Status |
|---|-------|--------|------------|--------|
| 2 | WASM AoT adapter | Small | bitnet.c library API | Not started |
| 5 | `HlLlmBackend` vtable + `llm.c` | Medium | Nothing (can start in parallel) | Not started |
| 6 | `llm_bitnet.c` native adapter | Small | Phase 5 + bitnet.c Phase 1 | Not started |
| 7 | `llm_gpu.c` (BnGPUBackend → HlGpuCtx) | Medium | Phase 5 + bitnet.c Phase 3-4 | Not started |
| 8 | WASM AoT fallback path | Small | Phase 2 + Phase 5 | Not started |

Phases 1, 3-4 (bitnet.c) and Phases 2, 5 (Hull) can proceed in parallel. The native adapter (Phase 6) is the integration point.

## What Exists Today

### bitnet.c (library API — integration surface)

| Component | Status |
|-----------|--------|
| `BnModel` / `BnSession` split | Done |
| `BnPromptCache` (shared KV prefix) | Done |
| `BnAllocator` (compatible with `KlAllocator`) | Done |
| `bn_token_callback` (streaming) | Done |
| `bn_format_sse_chunk` (OpenAI SSE) | Done |
| `BnGPUBackend` vtable + dispatch | Done |
| `bn_model_upload_weights` / `bn_model_release_gpu` | Done |
| `bn_logprobs_compute` (logprobs API) | Done |
| `bn_chat_format_messages` (multi-turn) | Done |
| `BnStopStrings` (stop string matching) | Done |
| WASM build (`wasm/api.c`) | Done (browser demo) |
| WGSL compute shaders | Not started |

### Hull (server + adapter — not started)

| Component | Status |
|-----------|--------|
| `HlLlmBackend` vtable | Not started |
| `llm.c` (session pool, SSE streaming) | Not started |
| `llm_bitnet.c` (native C adapter) | Not started |
| `llm_bitnet_wasm.c` (WASM AoT adapter) | Not started |
| `llm_gpu.c` (BnGPUBackend → HlGpuCtx) | Not started |
