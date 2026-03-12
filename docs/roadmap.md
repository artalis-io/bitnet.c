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

## Phase 4: SIMD Optimization

- [ ] ARM NEON kernels for ternary matvec (`src/quant_neon.c`)
- [ ] x86 AVX2 kernels for ternary matvec (`src/quant_avx2.c`)
- [ ] WASM SIMD128 kernels using `wasm_simd128.h` intrinsics
- [ ] Fused dequant+dot (eliminate intermediate float buffer)
- [ ] Compile-time kernel selection via Makefile flags
- [ ] Benchmark harness: tok/s across platforms
- [ ] Target: 3-5x speedup over naive implementation

## Phase 5: Memory & Performance

- [ ] KV cache quantization (reduce from FP32 to FP16 or INT8)
- [ ] Streaming KV cache (sliding window for long sequences)
- [ ] Multi-threaded attention (pthread for native, SharedArrayBuffer for WASM)
- [ ] Batch inference (process multiple tokens per forward pass for prompt)
- [ ] Memory-mapped KV cache for very long contexts
- [ ] Profile-guided optimization (PGO build)

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
