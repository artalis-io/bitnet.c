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

- [x] ARM NEON kernels for I2_S/TQ1_0/TQ2_0 ternary matvec (in `src/quant.c`)
- [x] SDOT (vdotq_s32) int8 accumulation for I2_S — 2× speedup over float FMA
- [x] Arithmetic ternary decode `(bits - 1)` — 15% speedup over compare-based
- [x] Batch matvec dispatch (QKV, gate+up grouped)
- [x] Native FP16 logits path via `-mcpu=apple-m1` (enables `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`)
- [ ] x86 AVX2 kernels for ternary matvec
- [ ] WASM SIMD128 kernels using `wasm_simd128.h` intrinsics

## Phase 5: Memory & Performance — Partially Done

### Completed
- [x] Pthread thread pool (~2μs condvar dispatch, replaces OMP fork/join)
- [x] Arena allocator for RunState (single allocation for all buffers)
- [x] RoPE frequency + cos/sin precomputation
- [x] Preallocated sampler candidates buffer (eliminates per-token malloc)
- [x] Prefetch hints in I2_S SDOT, TQ1_0, TQ2_0 kernels

### Remaining
- [ ] KV cache quantization (reduce from FP32 to FP16 or INT8)
- [ ] Streaming KV cache (sliding window for long sequences)
- [ ] Batch inference (process multiple tokens per forward pass for prompt)
- [ ] Profile-guided optimization (PGO build)
- [ ] INT8 output embeddings (reduce logits data from 656 MB to 328 MB per token)

## Performance Analysis (M1 Max, bitnet-b1.58-2B-4T)

### Current: ~42 tok/s (8 P-cores), ~98% of hardware bandwidth ceiling

The workload is **DRAM bandwidth-bound**. Each token reads ~1.15 GB from memory:

| Component | Data Read | % of Total |
|---|---|---|
| 30× layer I2_S weights (Q/K/V/O + gate/up/down) | 497 MB | 43% |
| Logits (F16 embedding × 128K vocab) | 656 MB | 57% |
| KV cache (pos-dependent) | ~18 MB | <2% |

M1 Max CPU aggregate DRAM bandwidth: ~55 GB/s (CPU-only; the 400 GB/s spec is GPU-inclusive).
At 42 tok/s × 1.15 GB = **48 GB/s sustained** — 87% of max bandwidth.

### Scaling behavior (bandwidth-limited)

| Cores | Est. BW (GB/s) | Est. tok/s | Speedup vs 1 |
|---|---|---|---|
| 1 | 7 | 6 | 1.0× |
| 2 | 14 | 12 | 2.0× |
| 4 | 28 | 24 | 4.0× |
| 8 | 55 | 43 | 7.1× |

Diminishing returns start at ~4 cores as DRAM interface saturates.

### What would move the needle

Only **reducing data volume** helps at this point:

1. **INT8 output embeddings** — halves logits data (656→328 MB). Est. ~55 tok/s (+30%).
2. **KV cache quantization** — reduces attention data at long positions.
3. **Prompt batching** — process multiple prompt tokens per forward pass (amortize logits).

### Optimization history

| Change | tok/s | Δ |
|---|---|---|
| Baseline (naive C) | ~15.5 | — |
| SDOT int8 accumulation + batch matvec | ~33 | +113% |
| Arithmetic ternary decode + RoPE precompute | ~38 | +15% |
| Pthread thread pool (replace OMP) | ~38 | latency improvement |
| Arena allocator + sh_log + FP16 native logits | ~42 | +10% |

## Phase 6: Extended Model Support

- [ ] Q6_K embedding dequantization (some models use Q6_K for token_embd)
- [ ] Non-tied output weights (separate output projection matrix)
- [ ] Multiple architecture support beyond BitNet (detect from GGUF metadata)
- [ ] Chat template support (system/user/assistant formatting)
- [ ] LoRA adapter loading

## Phase 7: Developer Experience

- [ ] Interactive mode (REPL-style multi-turn conversation)
- [ ] Token probability output mode (for debugging/research)
- [ ] JSON output mode (structured generation metadata)
- [ ] Model info dump command (`--info` to print config without inference)
- [ ] Progress bar for model loading
- [ ] Completion callback API (for embedding in other C programs)

## Phase 8: Platform Expansion

- [ ] Windows support (VirtualAlloc instead of mmap)
- [ ] iOS/Android builds (static library)
- [ ] Python bindings (ctypes or cffi wrapper)
- [ ] Node.js native addon
- [ ] WebGPU compute shader backend for browser

## Non-Goals

- Full GGUF compatibility (only BitNet-relevant types)
- Training or fine-tuning
- Multi-GPU / distributed inference
- Replacing llama.cpp for general LLM inference
