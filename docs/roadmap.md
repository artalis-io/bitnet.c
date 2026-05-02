# Roadmap

Development roadmap for bitnet.c.

## Phase 1: Core Modules (Naive C, Correct) — Done

- [x] `platform` — mmap/buffer abstraction, timing
- [x] `gguf` — GGUF v3 binary format parser
- [x] `quant` — TQ1_0/TQ2_0 dequantization, ternary matvec
- [x] `model` — GGUF → Config/Weights mapping, RunState allocation
- [x] `transformer` — full forward pass (RoPE, GQA, sub-norms, tied embeddings)
- [x] `tokenizer` — BPE encode/decode from GGUF metadata
- [x] `sampler` — argmax, multinomial, top-p sampling
- [x] `main` — CLI entry point with argument parsing
- [x] Unit tests for all modules
- [x] Makefile (native build)

## Phase 2: WASM Build — Done

- [x] `wasm/api.c` — EMSCRIPTEN_KEEPALIVE wrapper functions
- [x] `wasm/build.sh` — Emscripten build script
- [x] `wasm/worker.js` — Web Worker with streaming token output
- [x] `wasm/index.html` — minimal browser demo

## Phase 3: Validation & Correctness

- [ ] Greedy decode comparison vs `llama-cli` reference output
- [ ] Bit-exact TQ1_0/TQ2_0 dequantization validation against llama.cpp
- [ ] Activation dump mode (`-DDEBUG`) with layer-by-layer checkpoints
- [ ] Test against multiple BitNet GGUF models (2B, 4B variants)
- [ ] Validate tokenizer output matches reference BPE implementation
- [ ] Edge cases: empty prompt, single token, max sequence length

## Phase 4: SIMD Optimization — Done

- [x] ARM NEON kernels for I2_S/TQ1_0/TQ2_0 ternary matvec
- [x] SDOT (vdotq_s32) int8 accumulation for I2_S — 2x speedup over float FMA
- [x] Arithmetic ternary decode `(bits - 1)` — 15% speedup over compare-based
- [x] Batch matvec dispatch (QKV, gate+up grouped)
- [x] Native FP16 logits path via `-mcpu=apple-m1`
- [x] x86 AVX2 kernels for all quant formats (I2_S, Q4_0, Q8_0, Q6_K, Q8_K, Q4_K, Q5_K, Q3_K)
- [x] AVX2 fine-tuning: FMA across all kernels, Q8_0 DPBUSD, TQ1/TQ2 AVX2 kernels, Q5_K/Q3_K vectorization
- [x] WASM SIMD128 kernels for all quant formats

## Phase 5: Memory & Performance — Done

- [x] Pthread thread pool (~2us condvar dispatch, replaces OMP fork/join)
- [x] Arena allocator for RunState (single allocation for all buffers)
- [x] RoPE frequency + cos/sin precomputation
- [x] Preallocated sampler candidates buffer (eliminates per-token malloc)
- [x] Prefetch hints in I2_S SDOT, TQ1_0, TQ2_0 kernels
- [x] KV cache quantization (F16 KV via --kv16)
- [x] Sliding window KV cache (ring buffer, continues past seq_len)
- [x] Batch inference (bn_transformer_prefill)
- [x] Profile-guided optimization (PGO build)
- [x] INT8 output embeddings (~52.5 tok/s)

## Phase 6: Modular Backend Architecture — Done

- [x] Split `quant.c` into per-format per-backend modules (`src/quant/`)
- [x] Split `transformer.c` into per-backend modules (`src/transformer/`)
- [x] Internal headers (`quant_internal.h`, `transformer_internal.h`) with context structs
- [x] Backend selection via Makefile variables (ARM: NEON+scalar, x86: AVX2+scalar)
- [x] AVX2 cross-compile syntax check (`make avx2-check`)

## Phase 7: Extended Quantization Formats — Done

- [x] Q4_0 (4-bit) with SDOT/DPBUSD integer matvec
- [x] Q8_0 (8-bit) with NEON/AVX2/WASM kernels
- [x] Q6_K (6-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q8_K (8-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q4_K (4-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q5_K (5-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q3_K (3-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Non-tied output weights (separate output projection matrix)

## Phase 8: Extended Format Coverage — Done

- [x] Q2_K (2-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q4_1 (4-bit with min) with NEON/AVX2/WASM kernels
- [x] BF16 weight type with NEON/AVX2/WASM kernels
- [x] IQ4_NL (4-bit non-linear codebook) with NEON/AVX2/WASM kernels
- [x] IQ4_XS (4-bit non-linear with sub-block scales) with NEON/AVX2/WASM kernels
- [x] IQ3_XXS (3-bit codebook) with NEON/AVX2/WASM kernels
- [x] IQ3_S (3-bit codebook with separate signs) with NEON/AVX2/WASM kernels
- [x] IQ2_XXS (2-bit codebook) with NEON/AVX2/WASM kernels
- [x] IQ2_XS (2-bit codebook with scales) with NEON/AVX2/WASM kernels
- [x] IQ2_S (2-bit codebook, 1024-entry grid) with NEON/AVX2/WASM kernels

## Performance Analysis (M1 Max, bitnet-b1.58-2B-4T)

### Current: ~52.5 tok/s (8 P-cores)

The workload is **DRAM bandwidth-bound**. Each token reads ~0.83 GB from memory:

| Component | Data Read | % of Total |
|---|---|---|
| 30x layer I2_S weights (Q/K/V/O + gate/up/down) | 497 MB | 60% |
| Logits (INT8 embedding x 128K vocab) | 328 MB | 40% |
| KV cache (pos-dependent) | ~18 MB | <2% |

M1 Max CPU aggregate DRAM bandwidth: ~55 GB/s (CPU-only; the 400 GB/s spec is GPU-inclusive).
At 52.5 tok/s x 0.83 GB = **~43 GB/s sustained** — 79% of max bandwidth.

### Optimization history

| Change | tok/s | Delta |
|---|---|---|
| Baseline (naive C) | ~15.5 | — |
| SDOT int8 accumulation + batch matvec | ~33 | +113% |
| Arithmetic ternary decode + RoPE precompute | ~38 | +15% |
| Pthread thread pool (replace OMP) | ~41 | +8% |
| Arena allocator + FP16 native logits + prefetch | ~46 | +12% |
| INT8 output embeddings + SDOT logits | ~52.5 | +14% |

### Multi-model benchmarks (M1 Max, 8 P-cores, PGO)

| Model | Format | Size | tok/s |
|---|---|---|---|
| bitnet-b1.58-2B-4T | I2_S | 1.1 GB | 52.5 |
| Qwen2.5-3B-Instruct | Q4_0 | 1.7 GB | 30.0 |
| Llama3-8B-1.58 | TQ1_0 | 3.4 GB | 14.5 |

### What would move the needle

Only **reducing data volume** helps at this point:

1. **Speculative decoding** — use a smaller draft model to reduce per-token cost.
2. **KV cache quantization to INT8** — further reduces attention data at long positions.
3. **Weight clustering / pruning** — reduce I2_S weight data below 497 MB.

## Phase 9: Concurrent Sessions & Prompt Cache — Done

- [x] BnModel/BnSession split (shared model + per-request mutable state)
- [x] `bn_session_create/free/reset` API
- [x] BnPromptCache (shared KV prefix with longest-prefix matching, FIFO eviction)
- [x] Thread-safe prompt cache with configurable byte budget
- [x] Prompt cache integrated in chat mode CLI

## Phase 10: SIMD Backend Parity — Done

- [x] AVX2 flash GQA (online softmax attention)
- [x] WASM flash GQA + scalar flash GQA
- [x] WASM I8 SDOT logits + `bn_quant_f16_rows_to_i8` for WASM
- [x] WASM SDOT for TQ1_0, TQ2_0, Q8_0, Q4_K, Q6_K
- [x] WASM `bn_quant_x_to_q8k` (Q8_K super-block quantization)
- [x] AVX2 Q4_K/Q6_K fused matmul (batch prefill)
- [x] VLA guards + SIMD alignment guards in all GQA backends

## Phase 11: GPU Compute Backend (WebGPU) — Done

- [x] `BnGPUBackend` vtable (matvec, matmul, matvec_batch, execute, init_activations)
- [x] GPU dispatch integration in `quant/dispatch.c` (`bn_quant_matvec_gpu`, `bn_quant_matvec_batch_gpu`)
- [x] 22 WGSL matvec shaders for all quant types (I2_S through IQ2_S + F16/F32)
- [x] 9 WGSL forward-pass shaders (rmsnorm, rope, gqa_scores, softmax, gqa_combine, silu_gate, relu2_gate, residual_add, bias_add)
- [x] `bn_model_upload_weights` / `bn_model_release_gpu` for GPU buffer management
- [x] Norm weight + bias + tied embedding GPU upload
- [x] Persistent scratch buffers + batched command encoding
- [x] Single-submit GPU-resident forward pass (one command buffer per token)
- [x] `--gpu` CLI flag, `make BN_ENABLE_GPU=1`, `make fetch-wgpu`
- [x] GPU validation benchmark (20/20 matvec pass, 3/3 matmul pass)
- [x] wgpu-native vendoring with SHA-256 verification
- See [docs/hull-integration.md](hull-integration.md) for the Hull integration design

### Library API — Done
- [x] SSE chunk formatter (`bn_format_sse_chunk`, `bn_format_sse_done`)
- [x] Logprobs API (`bn_logprobs_compute`)
- [x] Multi-turn chat formatting (`bn_chat_format_messages`)
- [x] Stop strings (`BnStopStrings`)
- [x] Allocator vtable (`BnAllocator`, compatible with Keel's `KlAllocator`)

## Future Work

### GPU Optimization
- [ ] Improve GPU forward-pass shader precision (match CPU output beyond first token)
- [ ] FP16 KV cache on GPU
- [ ] Sub-norms, Q/K norms, gated-Q support in GPU forward path
- [ ] MoE expert dispatch on GPU

### Extended Model Support
- [ ] LoRA adapter loading

### Developer Experience
- [x] Interactive mode (--chat REPL with sliding window)
- [ ] Token probability output mode (for debugging/research)
- [ ] JSON output mode (structured generation metadata)
- [ ] Model info dump command (`--info` to print config without inference)

### GPU Performance (Metal/WebGPU)

Current: 290 ops, 289 barriers, ~22.7ms/token on M1 Max (Qwen 2.5 3B Q4_0). llama.cpp: ~9ms.

**Key finding**: Barriers cost zero (22.7ms with 0 barriers vs 22.8ms with 289). The bottleneck is **Q4_0 matvec kernel execution** — the DQ4 dequantization + float4 dot product inner loop. Reducing ops/barriers further has no impact. The path to parity requires a fundamentally faster matvec kernel.

- [x] Eliminate COPY ops: direct-to-destination QKV writes via shader offsets (-108 ops, -108 barriers)
- [x] Batched QKV matvec: single-dispatch split shader for Q4_0 (-72 ops)
- [x] PSO caching: skip redundant setComputePipelineState calls
- [x] Fused RoPE Q+K: single dispatch for Q and K rotation (-36 ops)
- [x] Fused gate+up+SiLU: stacked weight buffer, single Q4_0 kernel (-72 ops, -36 barriers)
- [x] Fix logits tiling: dispatch (4748,1,1) not (65535,1,1) for vocab=151936
- [ ] **Q4_0 matvec kernel rewrite** (THE bottleneck): current DQ4 macro does 4 shifts + 4 masks + 4 subtracts + 4 int-to-float + 1 float4 mul per 4 elements = ~17 ops. Need simdgroup_matrix or packed int8 dot product to reduce ALU pressure. Study llama.cpp's `kernel_mul_mv_q4_0_f32` approach.
- [ ] Interleaved Q4_0 repack: store [scale, nib0..3] per block contiguously instead of split sections. Improves spatial locality but doesn't help if compute-bound.
- [ ] MATVEC_SPLIT for all quant types: extend beyond Q4_0
- [ ] Half-precision intermediates where precision allows

### SIMD Backends
- [ ] AVX-512 VNNI — native `vpdpbusd`, 512-bit vectors (Ice Lake+, Zen 4+)

### Platform Expansion
- [ ] Windows support (VirtualAlloc instead of mmap)
- [ ] iOS/Android builds (static library)
- [ ] Python bindings (ctypes or cffi wrapper)
- [ ] Node.js native addon

## Non-Goals

- Full GGUF compatibility (only supported types listed above)
- Training or fine-tuning
- Multi-GPU / distributed inference
- Replacing llama.cpp for general LLM inference
- HTTP server (that's Hull's job)
