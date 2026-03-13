# bitnet.c

A clean-room, pure C inference engine for [BitNet b1.58](https://arxiv.org/abs/2402.17764) transformer models.

Inspired by Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) — a beautifully minimal LLaMA inference implementation in a single C file — **bitnet.c** takes the same philosophy and applies it to Microsoft's [BitNet](https://github.com/microsoft/BitNet) architecture with its 1.58-bit ternary weights.

Where Microsoft's official BitNet inference framework depends on a modified llama.cpp fork (~100K+ lines of C++), bitnet.c delivers a complete inference pipeline in ~4,100 lines of modular, readable C.

## Features

- **Pure C11** — no C++, no frameworks, no dependencies beyond libc and libm
- **GGUF model loading** — compatible with existing BitNet GGUF files (e.g. `bitnet-b1.58-2B-4T`)
- **I2_S, TQ1_0, & TQ2_0 formats** — native support for Microsoft's I2_S and GGML ternary quantization
- **Full transformer forward pass** — RoPE, GQA, RMSNorm, sub-norms, tied embeddings
- **Flash GQA attention** — online softmax with KV-head grouping, single-pass over KV cache
- **ARM NEON/SDOT optimizations** — SDOT int8 matvec, native FP16 logits, INT8 output embeddings
- **Pthread thread pool** — persistent workers with condvar dispatch (~2us), replaces OpenMP
- **BPE tokenizer** — loaded directly from GGUF metadata
- **Sampling** — greedy (argmax), multinomial, and nucleus (top-p)
- **Native mmap** — zero-copy model loading on macOS/Linux
- **WASM build** — runs in the browser via Emscripten with Web Worker streaming
- **Modular architecture** — orthogonal, composable, separately unit-testable modules

## Quick Start

```bash
# Build
make

# Run inference
./bitnet model.gguf -p "Hello" -n 256

# Run with sampling
./bitnet model.gguf -p "Once upon a time" -n 512 --temp 0.7 --topp 0.9

# Run tests
make test
```

### Options

```
Usage: ./bitnet <model.gguf> [options]
  -p <prompt>     Input prompt (default: "Hello")
  -n <int>        Number of tokens to generate (default: 256)
  --temp <float>  Temperature (default: 0.0 = greedy)
  --topp <float>  Top-p sampling (default: 0.9)
  --seed <int>    Random seed (default: 42)
  --maxseq <int>  Max sequence length (default: model max)
```

## Getting a Model

Download the official Microsoft BitNet b1.58-2B-4T GGUF model into the `model/` directory:

```bash
mkdir -p model

# Using huggingface-cli (recommended)
pip install huggingface-hub
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \
  --include "bitnet-b1.58-2B-4T-TQ2_0.gguf" \
  --local-dir model/

# Or download directly with curl (~780 MB)
curl -L -o model/bitnet-b1.58-2B-4T-TQ2_0.gguf \
  https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/bitnet-b1.58-2B-4T-TQ2_0.gguf
```

Then run:

```bash
./bitnet model/bitnet-b1.58-2B-4T-TQ2_0.gguf -p "The capital of France is"
```

The `model/` directory is git-ignored — model files won't be committed.

## Project Structure

```
bitnet.c/
├── include/
│   ├── platform.h      # Platform abstraction (mmap, timing)
│   ├── gguf.h          # GGUF v3 reader API
│   ├── quant.h         # TQ1_0/TQ2_0/I2_S dequantization + ternary matmul
│   ├── model.h         # Config, Weights, model loading
│   ├── transformer.h   # Forward pass API
│   ├── tokenizer.h     # BPE tokenizer API
│   ├── sampler.h       # Sampling strategies
│   ├── threadpool.h    # Persistent pthread thread pool
│   ├── sh_arena.h      # Arena allocator
│   └── sh_log.h        # Structured logging
├── src/
│   ├── platform.c      # mmap/fread/timing abstraction
│   ├── gguf.c          # GGUF binary format parser
│   ├── quant.c         # FP16 conversion, TQ1/TQ2/I2_S dequant, ternary matvec
│   ├── model.c         # GGUF → Config/Weights mapping
│   ├── transformer.c   # Forward pass: flash attention, FFN, sub-norms
│   ├── tokenizer.c     # BPE encode/decode from GGUF vocab
│   ├── sampler.c       # Argmax, multinomial, top-p sampling
│   ├── threadpool.c    # Thread pool with condvar dispatch
│   ├── sh_arena.c      # Arena allocator implementation
│   ├── sh_log.c        # Structured logging implementation
│   └── main.c          # CLI entry point
├── test/
│   ├── test_gguf.c     # GGUF parser tests
│   ├── test_quant.c    # Dequantization + matvec tests
│   ├── test_transformer.c  # RMSNorm, softmax, RoPE tests
│   ├── test_tokenizer.c    # BPE encode/decode tests
│   ├── test_threadpool.c   # Thread pool dispatch tests
│   ├── test_safety.c       # Safety/bounds-checking regression tests
│   └── test_e2e.c      # End-to-end greedy decode test
├── wasm/
│   ├── api.c           # WASM-exported API wrapper
│   ├── build.sh        # Emscripten build script
│   ├── worker.js       # Web Worker for non-blocking inference
│   └── index.html      # Browser demo
├── docs/
│   └── roadmap.md      # Development roadmap
└── Makefile
```

### Module Dependency Graph

Each module depends only on those above it:

```
platform
    ↓
  gguf        ← standalone, testable in isolation
    ↓
  quant       ← standalone, testable with synthetic data
    ↓
  model       ← depends on gguf + quant
    ↓
tokenizer     ← depends on gguf
    ↓
transformer   ← depends on model + quant
    ↓
 sampler      ← standalone, testable in isolation
    ↓
  main        ← wires everything together
```

## WASM Build

Requires [Emscripten](https://emscripten.org/):

```bash
./wasm/build.sh
# Produces wasm/bitnet.js + wasm/bitnet.wasm
# Open wasm/index.html in a browser
```

## Performance

Benchmarked on Apple M1 Max (8 P-cores, 32 GB), `bitnet-b1.58-2B-4T` (I2_S format), greedy decoding:

**~52 tok/s** (256 tokens generated)

### Optimization History

| Optimization | tok/s | Speedup |
|---|---|---|
| Baseline (scalar C) | ~15.5 | 1.0x |
| + SDOT int8 accumulation + batch matvec | ~33 | 2.1x |
| + Arithmetic ternary decode + RoPE precompute | ~38 | 2.5x |
| + Pthread thread pool (replace OpenMP) | ~38 | 2.5x |
| + Arena allocator + native FP16 logits + prefetch | ~46 | 3.0x |
| + INT8 output embeddings (SDOT logits) | **~52** | **3.4x** |

### Key Optimizations

- **SDOT (vdotq_s32)** — ARM dot product instructions for I2_S ternary matvec. Quantize activations to int8 once, then use integer dot products instead of float FMA. 2x speedup.
- **Batch matvec** — group independent projections (QKV, gate+up) into a single thread pool dispatch, sharing the int8-quantized activation vector.
- **Pthread thread pool** — persistent worker threads with condvar dispatch (~2us), replacing OpenMP fork/join (~15us on macOS).
- **Native FP16 logits** — `-mcpu=apple-m1` enables `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` for native F16 FMA in the logits computation (embedding dot products). Apple clang's `-march=native` misses this.
- **INT8 output embeddings** — pre-quantize F16 embedding table to INT8 with per-row scales at load time. SDOT logits kernel reads 328 MB/token instead of 656 MB.
- **Arena allocator** — all RunState buffers in a single contiguous allocation for better TLB coverage.

### Bandwidth Analysis

At ~52 tok/s the workload is **DRAM bandwidth-bound**, reading ~825 MB per token:

| Component | Data/Token |
|---|---|
| Layer weights (30 layers, I2_S) | 497 MB |
| Logits (INT8 embeddings, 128K vocab) | 328 MB |

This sustains ~43 GB/s of the M1 Max CPU bandwidth ceiling (~55-60 GB/s).

## How It Works

### BitNet b1.58 Architecture

BitNet b1.58 is a transformer variant where all linear layer weights are constrained to ternary values {-1, 0, +1}. This enables:

- **1.58-bit weights** — dramatically smaller models (vs 16-bit or even 4-bit quantization)
- **Multiplication-free inference** — ternary matmul reduces to additions and subtractions
- **Sub-norms** — additional RMSNorm layers after attention and FFN activations

### Quantization Formats

| Format | Bits/Weight | Packing | Block Size |
|--------|-------------|---------|------------|
| TQ1_0  | 1.6875      | Base-3 (5 values/byte) + residual | 256 |
| TQ2_0  | 2.0625      | 2-bit fields (4 values/byte) | 256 |

### Memory Budget (bitnet-b1.58-2B-4T, 2048 context)

| Component | Size |
|-----------|------|
| GGUF buffer (weights + F16 embeddings) | ~620 MB |
| INT8 embedding cache (128K × 2560) | ~329 MB |
| KV cache (30 layers × 2048 × 640 × 4 × 2) | ~298 MB |
| RunState activations | ~3 MB |
| **Total** | **~1,250 MB** |

## Design Decisions

### Why not lookup tables (LUTs) for the ternary kernel?

T-MAC and bitnet.cpp use lookup tables to replace multiply-accumulate with table-indexed addition. The idea: precompute partial sums for each possible weight bit pattern, then look them up instead of doing arithmetic. We evaluated this approach and rejected it.

**The I2_S SDOT kernel doesn't need LUTs.** The ternary decode is 4 shifts + 3 masks + 4 subtracts per 64 values — branchless, no data-dependent control flow, no table lookup. ARM's `vdotq_s32` then does 16 int8×int8 multiply-accumulates in a single instruction (~3 cycles). A LUT replaces one fast hardware instruction with a memory load — trading L1 cache capacity for arithmetic savings on operations that are already free relative to memory stalls.

**The bottleneck is DRAM bandwidth, not decode arithmetic.** At ~52 tok/s the runtime sustains ~43 GB/s reading ~825 MB of weight data per token. The ternary decode costs ~6 cycles per 128 elements; the SDOT instructions cost ~8 cycles; the memory stalls cost everything else. A LUT can't reduce the bytes read from DRAM. It can only consume L1 cache that's currently holding weight prefetch buffers and KV cache tiles.

**Where LUTs would help (but don't apply here).** TQ1_0 base-3 encoding requires 3 multiplies + 1 shift + 1 narrow per 8 elements for trit extraction — genuinely expensive. A 256-entry byte→5-trit LUT (2 KB) would eliminate this. But the target model uses I2_S, not TQ1_0, so this optimization has zero impact on the actual workload.

### Why C

**The inner loop is 8 NEON intrinsics.** The entire I2_S SDOT matvec kernel — the function that consumes >90% of runtime — is 40 lines of C with ARM intrinsics. There is no abstraction to manage, no object lifetime to track, no type system to appease. The code *is* the assembly, minus register allocation.

**Why not Rust?** Rust solves memory safety at the type system level, which is genuinely valuable at scale — large teams, deep dependency trees, long-lived codebases with contributor churn. For a small, focused inference engine with zero dependencies and one or two authors who understand every line, the tradeoffs don't pay off:

- *The hot path is NEON intrinsics.* The matvec kernel, logits computation, and attention scoring are all hand-written SIMD. These are `unsafe` by definition in Rust — you get the borrow checker's overhead without its guarantees where performance matters.
- *Zero-copy GGUF loading.* The model is mmap'd and weights are read directly from the mapped buffer as raw pointers into packed binary data. This is one `mmap` call in C. In Rust it's `unsafe` pointer arithmetic wrapped in lifetime-annotated structs, or a dependency on `memmap2` + `bytemuck` + `zerocopy`.
- *No dependencies to protect against.* This project links libc, libm, and nothing else. No crates, no supply chain, no transitive dependencies. Cargo's safety story is about managing a dependency graph — there is no graph here.
- *Build time.* Clean build: under 2 seconds. A comparable Rust project with `memmap2`, `half`, `rayon`: 20-40 seconds.

**Why not C++?** Everything C gives you here but with a language that actively fights simplicity. The GGUF parser is a flat struct with pointer arithmetic. The thread pool is pthreads + condvar. The transformer forward pass is a `for` loop over layers. In C++ someone would reach for `std::variant` for tensor types, `std::jthread` with `std::barrier`, and a template-based layer abstraction. All objectively worse for 4,100 lines of inference code.

**Why not Zig?** Zig's `comptime`, explicit allocators, and native SIMD vectors are genuinely appealing. For a new project not needing Emscripten WASM support, it would be a strong choice. But Zig's WASM target doesn't support the Emscripten features this project relies on (`EXPORTED_FUNCTIONS`, `MODULARIZE`, `SINGLE_FILE`), and the ecosystem isn't there yet for production inference workloads.

**Why not Python / Go / Java?** An inference engine's job is to read ~825 MB of weights from DRAM per token as fast as the memory bus allows, then do minimal arithmetic on the result. Managed runtimes add GC pauses, prevent manual memory layout, and can't express SIMD intrinsics. Python would need NumPy or ctypes back into C. Go can't do NEON. Java's `Vector` API is experimental and JNI overhead kills the inner loop. The right language for a DRAM-bandwidth-bound SIMD kernel is the one that compiles to the SIMD instructions.

## Acknowledgments

This project stands on the shoulders of giants:

- **[llama2.c](https://github.com/karpathy/llama2.c)** by Andrej Karpathy — the inspiration for proving that a complete LLM inference engine can be beautifully simple. The modular architecture, mmap-based loading, and "make it work in pure C" philosophy all trace directly to Karpathy's pioneering work.

- **[BitNet](https://github.com/microsoft/BitNet)** by Microsoft Research — the breakthrough 1.58-bit quantization research ([paper](https://arxiv.org/abs/2402.17764)) that makes ternary-weight transformers practical. The official framework and model weights make this project possible.

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** / **[GGML](https://github.com/ggml-org/ggml)** by Georgi Gerganov — the GGUF file format, TQ1_0/TQ2_0 quantization implementations, and the broader ecosystem of efficient LLM inference.

## License

[MIT](LICENSE)
