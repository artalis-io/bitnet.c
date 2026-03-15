# bitnet.c

A minimal, embeddable LLM inference engine in pure C11.

Started as a clean-room inference engine for Microsoft's [BitNet b1.58](https://arxiv.org/abs/2402.17764) ternary models, inspired by Karpathy's [llama2.c](https://github.com/karpathy/llama2.c). Now supports standard GGUF quantization formats (Q4_0, Q8_0) alongside ternary (I2_S, TQ1_0, TQ2_0) — covering most small language models on HuggingFace.

Zero dependencies beyond libc and libm, four SIMD backends, compiles to WASM, and fits in ~8,000 lines of modular C.

## Features

- **Pure C11** — no C++, no frameworks, no dependencies beyond libc and libm
- **GGUF model loading** — loads any GGUF file with supported tensor types
- **Quantization formats** — I2_S, TQ1_0, TQ2_0 (ternary), Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K (k-quants), Q8_0, Q8_K
- **Full transformer forward pass** — RoPE, GQA, RMSNorm, sub-norms, tied embeddings
- **Flash GQA attention** — online softmax with KV-head grouping, single-pass over KV cache
- **Optional F16 KV cache** — `--kv16` halves attention DRAM bandwidth with minimal precision loss
- **4 SIMD backends** — ARM NEON/SDOT, AVX2, WASM SIMD128, scalar fallback (auto-selected at compile time)
- **Pthread thread pool** — persistent workers with condvar dispatch (~2us), replaces OpenMP
- **BPE tokenizer** — loaded directly from GGUF metadata
- **Sampling** — greedy (argmax), multinomial, and nucleus (top-p)
- **Native mmap** — zero-copy model loading on macOS/Linux
- **WASM build** — runs in the browser via Emscripten with Web Worker streaming
- **Modular architecture** — orthogonal, composable, separately unit-testable modules

## SIMD Backends

| Backend | Platform | Notes |
|---------|----------|-------|
| ARM NEON/SDOT | Apple Silicon, ARMv8 | SDOT int8 matvec, native FP16 logits |
| AVX2 | x86-64 (Haswell+) | DPBUSD int8 matvec, F16C conversion |
| WASM SIMD128 | Browser (Emscripten) | 128-bit vector ops |
| Scalar | Any C11 compiler | Portable fallback |

Auto-selected at compile time based on target architecture.

## Tested Models

| Model | Params | Format | Status |
|-------|--------|--------|--------|
| [bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | 2B | I2_S / TQ2_0 | Primary development target |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | 0.5B | Q4_0 + Q8_0 | Working |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) | 3B | Q4_0 + Q6_K | Working |

Models must use only supported weight types (I2_S, TQ1_0, TQ2_0, Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K, F16, F32).

## Quick Start

```bash
# Build
make

# Run inference
./bitnet model.gguf -p "Hello" -n 256

# Run with sampling
./bitnet model.gguf -p "Once upon a time" -n 512 --temp 0.7 --topp 0.9

# Interactive chat
./bitnet model.gguf --chat

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
  --flash         Use flash attention (online softmax)
  --chat          Interactive chat REPL mode
  --repeat-penalty <float>  Repetition penalty (default: 1.0, chat: 1.1)
  --kv16          Store KV cache in FP16 (halves attention DRAM bandwidth)
  --no-prefill    Disable batch prompt prefill (compute logits for every token)
```

### Chat Mode

`--chat` enters an interactive REPL with multi-turn conversation support. KV cache is reused across turns so context accumulates naturally. Type `/quit` or Ctrl-D to exit.

Chat mode defaults to `--temp 0.5 --topp 0.9 --repeat-penalty 1.1` for more natural conversation. Override with explicit flags.

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
│   ├── platform.h              # Platform abstraction (mmap, timing)
│   ├── gguf.h                  # GGUF v3 reader API
│   ├── quant.h                 # Public quant API: block structs, matvec, dequant
│   ├── quant_internal.h        # Quant backend context structs + range function decls
│   ├── model.h                 # Config, Weights, model loading
│   ├── transformer.h           # Forward pass public API
│   ├── transformer_internal.h  # Transformer backend context structs + range function decls
│   ├── tokenizer.h             # BPE tokenizer API
│   ├── sampler.h               # Sampling strategies
│   ├── threadpool.h            # Persistent pthread thread pool
│   ├── simd_helpers.h          # Shared AVX2/WASM SIMD inline helpers
│   ├── sh_arena.h              # Arena allocator
│   └── sh_log.h                # Structured logging
├── src/
│   ├── platform.c              # mmap/fread/timing abstraction
│   ├── gguf.c                  # GGUF binary format parser
│   ├── quant/                  # Per-format per-backend matvec kernels
│   │   ├── dispatch.c          # Format dispatch + batch matvec
│   │   ├── dequant.c           # Dequantization functions
│   │   ├── fp16.c              # FP16 ↔ FP32 conversion
│   │   ├── i2s_neon_sdot.c     # I2_S SDOT kernel (ARM dotprod)
│   │   ├── i2s_scalar.c        # I2_S scalar fallback
│   │   ├── q4_neon_sdot.c      # Q4_0 SDOT kernel
│   │   ├── q6k_neon.c          # Q6_K NEON kernel
│   │   └── ...                 # ~50 backend files total
│   ├── transformer/            # Per-backend transformer kernels
│   │   ├── rmsnorm_{neon,avx2,wasm,scalar}.c
│   │   ├── gqa_{neon,avx2,wasm,scalar}.c
│   │   └── logits_{neon,avx2,wasm,scalar}.c
│   ├── model.c                 # GGUF → Config/Weights mapping
│   ├── transformer.c           # Forward pass: layer loop, FFN, dispatch
│   ├── tokenizer.c             # BPE encode/decode from GGUF vocab
│   ├── sampler.c               # Argmax, multinomial, top-p sampling
│   ├── threadpool.c            # Thread pool with condvar dispatch
│   ├── sh_arena.c              # Arena allocator implementation
│   ├── sh_log.c                # Structured logging implementation
│   └── main.c                  # CLI entry point
├── test/                       # Assert-based unit tests (synthetic data, no model needed)
├── wasm/                       # Emscripten WASM build + browser demo
├── docs/
│   ├── inference.md            # Inference pipeline: math, algorithms, code map
│   └── roadmap.md              # Development roadmap + optimization history
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

### Benchmarks

Measured on Apple M1 Max (8 P-cores, 32 GB), greedy decoding (`--temp 0`), 8 threads.

| Model | Size | Quant | bitnet.c | llama.cpp CPU | llama.cpp Metal |
|-------|------|-------|----------|---------------|-----------------|
| bitnet-b1.58-2B-4T | 620 MB | I2_S (ternary) | **52 tok/s** | — | — |
| Qwen2.5-3B-Instruct | 1.7 GB | Q4_0 + Q6_K | **14 tok/s** | 25 tok/s | 112 tok/s |

bitnet.c is a pure CPU engine with no GPU backend. On ternary models it's competitive with anything — the workload is DRAM-bandwidth-bound and GPU dispatch overhead adds latency for zero throughput gain. On standard quant models (Q4_0/Q6_K), llama.cpp's CPU path is ~1.8x faster due to weight repacking and more aggressive integer dot product kernels; its Metal path is ~8x faster via GPU offload.

### Optimization History (bitnet-b1.58-2B-4T)

| Optimization | tok/s | Speedup |
|---|---|---|
| Baseline (scalar C) | ~15.5 | 1.0x |
| + SDOT int8 accumulation + batch matvec | ~33 | 2.1x |
| + Arithmetic ternary decode + RoPE precompute | ~38 | 2.5x |
| + Pthread thread pool (replace OpenMP) | ~41 | 2.6x |
| + Arena allocator + native FP16 logits + prefetch | ~46 | 3.0x |
| + INT8 output embeddings (SDOT logits) | **~52** | **3.4x** |

### Key Optimizations

- **SDOT (vdotq_s32)** — ARM dot product instructions for I2_S ternary matvec. Quantize activations to int8 once, then use integer dot products instead of float FMA. 2x speedup.
- **Batch matvec** — group independent projections (QKV, gate+up) into a single thread pool dispatch, sharing the int8-quantized activation vector.
- **Pthread thread pool** — persistent worker threads with condvar dispatch (~2us), replacing OpenMP fork/join (~15us on macOS).
- **SIMD Q6_K kernels** — vectorized 6-bit dequant + float FMA for NEON, AVX2, and WASM SIMD128. 78% speedup on Q6_K-bottlenecked logits.
- **Q4_0 integer dot product** — per-block Q8_0 activation quantization with SDOT/DPBUSD integer matvec, matching llama.cpp's approach. 6/8 first-word match on greedy decoding.
- **INT8 output embeddings** — pre-quantize F16 embedding table to INT8 with per-row scales at load time. SDOT logits kernel reads 328 MB/token instead of 656 MB.

### Bandwidth Analysis (bitnet-b1.58-2B-4T)

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
| I2_S   | 2.0         | 2-bit interleaved (4 values/byte) + per-tensor scale | 128 |
| TQ1_0  | 1.6875      | Base-3 (5 values/byte) + residual | 256 |
| TQ2_0  | 2.0625      | 2-bit fields (4 values/byte) | 256 |
| Q2_K   | 2.625       | 2-bit quants + 4-bit sub-block scales/mins | 256 |
| Q3_K   | 3.4375      | 3-bit quants (split ql/qh) + 6-bit sub-block scales | 256 |
| Q4_0   | 4.5         | 4-bit nibbles (2 values/byte) + FP16 per-block scale | 32 |
| Q4_K   | 4.5         | 4-bit quants + 6-bit sub-block scales/mins | 256 |
| Q5_K   | 5.5         | 5-bit quants (split ql/qh) + 6-bit sub-block scales/mins | 256 |
| Q6_K   | 6.5625      | 6-bit quants (split ql/qh) + int8 sub-block scales | 256 |
| Q8_0   | 8.5         | 8-bit values + FP16 per-block scale | 32 |
| Q8_K   | 9.125       | 8-bit values + float32 scale + int16 block sums | 256 |

### Memory Budget (bitnet-b1.58-2B-4T, 2048 context)

| Component | Size |
|-----------|------|
| GGUF buffer (weights + F16 embeddings) | ~620 MB |
| INT8 embedding cache (128K × 2560) | ~329 MB |
| KV cache (30 layers × 2048 × 640 × 4 × 2) | ~298 MB (~149 MB with `--kv16`) |
| RunState activations | ~3 MB |
| **Total** | **~1,250 MB** (~1,101 MB with `--kv16`) |

## Design Decisions

### Why not lookup tables (LUTs) for the ternary kernel?

T-MAC and bitnet.cpp use lookup tables to replace multiply-accumulate with table-indexed addition. The idea: precompute partial sums for each possible weight bit pattern, then look them up instead of doing arithmetic. We evaluated this approach and rejected it.

**The I2_S SDOT kernel doesn't need LUTs.** The ternary decode is 4 shifts + 3 masks + 4 subtracts per 64 values — branchless, no data-dependent control flow, no table lookup. ARM's `vdotq_s32` then does 16 int8×int8 multiply-accumulates in a single instruction (~3 cycles). A LUT replaces one fast hardware instruction with a memory load — trading L1 cache capacity for arithmetic savings on operations that are already free relative to memory stalls.

**The bottleneck is DRAM bandwidth, not decode arithmetic.** At ~52 tok/s the runtime sustains ~43 GB/s reading ~825 MB of weight data per token. The ternary decode costs ~6 cycles per 128 elements; the SDOT instructions cost ~8 cycles; the memory stalls cost everything else. A LUT can't reduce the bytes read from DRAM. It can only consume L1 cache that's currently holding weight prefetch buffers and KV cache tiles.

**Where LUTs would help (but don't apply here).** TQ1_0 base-3 encoding requires 3 multiplies + 1 shift + 1 narrow per 8 elements for trit extraction — genuinely expensive. A 256-entry byte→5-trit LUT (2 KB) would eliminate this. But the target model uses I2_S, not TQ1_0, so this optimization has zero impact on the actual workload.

### Why not llama.cpp / Ollama?

**llama.cpp is a general-purpose inference engine. bitnet.c is a ternary-specific one.**

llama.cpp supports dozens of quantization formats, model architectures, GPU backends (CUDA, Metal, Vulkan, SYCL), and serving modes. That generality costs ~200K lines of C++ and a build system that pulls in platform-specific SDKs. Ollama wraps llama.cpp with a Go service layer, adding another ~50K lines and a Docker-style distribution model. Both are excellent tools for running arbitrary GGUF models.

bitnet.c exists because BitNet's ternary weights ({-1, 0, +1}) make most of that machinery irrelevant:

- **No GPU needed.** Ternary matvec is memory-bandwidth-bound, not compute-bound. A single M1 Max CPU core sustains ~52 tok/s — there's no FLOPs deficit to offload. GPU dispatch overhead and PCIe transfers would add latency for zero throughput gain.
- **Minimal format support.** bitnet.c supports ~10 quant formats vs llama.cpp's dozens of format x operation x backend combinations.
- **No abstraction layers.** llama.cpp routes tensor operations through GGML's graph-based backend abstraction. bitnet.c calls the matvec kernel directly — the forward pass is a flat loop over layers with inline SIMD.
- **Embeddable.** The entire engine is ~8,000 lines of C11 with zero dependencies. It compiles to WASM and runs in a browser. Try that with llama.cpp's Metal backend.

Microsoft's own [BitNet inference framework](https://github.com/microsoft/BitNet) takes the opposite approach: it forks llama.cpp and patches in ternary kernel support. This inherits llama.cpp's full dependency tree (CMake, Python, conda) for a model that only needs addition and subtraction.

**When to use llama.cpp/Ollama instead:** if you need GPU inference, non-BitNet models, OpenAI-compatible API serving, or multi-model management. They're the right tools for general LLM deployment. bitnet.c is for when you want a single ternary model running as fast as possible with nothing else in the way.

### Why C

**The inner loop is 8 NEON intrinsics.** The I2_S SDOT matvec kernel — the function that dominates runtime on ternary models — is 40 lines of C with ARM intrinsics. There is no abstraction to manage, no object lifetime to track, no type system to appease. The code *is* the assembly, minus register allocation.

**Why not Rust?** Rust solves memory safety at the type system level, which is genuinely valuable at scale — large teams, deep dependency trees, long-lived codebases with contributor churn. For a small, focused inference engine with zero dependencies and one or two authors who understand every line, the tradeoffs don't pay off:

- *The hot path is NEON intrinsics.* The matvec kernel, logits computation, and attention scoring are all hand-written SIMD. These are `unsafe` by definition in Rust — you get the borrow checker's overhead without its guarantees where performance matters.
- *Zero-copy GGUF loading.* The model is mmap'd and weights are read directly from the mapped buffer as raw pointers into packed binary data. This is one `mmap` call in C. In Rust it's `unsafe` pointer arithmetic wrapped in lifetime-annotated structs, or a dependency on `memmap2` + `bytemuck` + `zerocopy`.
- *No dependencies to protect against.* This project links libc, libm, and nothing else. No crates, no supply chain, no transitive dependencies. Cargo's safety story is about managing a dependency graph — there is no graph here.
- *Build time.* Clean build: under 3 seconds. A comparable Rust project with `memmap2`, `half`, `rayon`: 20-40 seconds.

**Why not C++?** Everything C gives you here but with a language that actively fights simplicity. The GGUF parser is a flat struct with pointer arithmetic. The thread pool is pthreads + condvar. The transformer forward pass is a `for` loop over layers. In C++ someone would reach for `std::variant` for tensor types, `std::jthread` with `std::barrier`, and a template-based layer abstraction. All objectively worse for a focused inference engine.

**Why not Zig?** Zig's `comptime`, explicit allocators, and native SIMD vectors are genuinely appealing. For a new project not needing Emscripten WASM support, it would be a strong choice. But Zig's WASM target doesn't support the Emscripten features this project relies on (`EXPORTED_FUNCTIONS`, `MODULARIZE`, `SINGLE_FILE`), and the ecosystem isn't there yet for production inference workloads.

**Why not Python / Go / Java?** An inference engine's job is to read ~825 MB of weights from DRAM per token as fast as the memory bus allows, then do minimal arithmetic on the result. Managed runtimes add GC pauses, prevent manual memory layout, and can't express SIMD intrinsics. Python would need NumPy or ctypes back into C. Go can't do NEON. Java's `Vector` API is experimental and JNI overhead kills the inner loop. The right language for a DRAM-bandwidth-bound SIMD kernel is the one that compiles to the SIMD instructions.

## Known Limitations

### Multi-turn chat degrades after code-heavy responses

The 2B 1.58-bit model (~400 MB of effective weights) struggles with multi-turn conversations when earlier turns contain code blocks. Backtick-heavy formatting in the KV cache attention context disrupts the model's output quality on subsequent turns, often producing single-digit garbage responses.

**Works well:** short Q&A across multiple turns (e.g., "Hello" → "What is 2+2?" → "Capital of France?"), and non-code-to-code transitions (e.g., "What are primary colors?" → "Write hello world in Python").

**Breaks down:** code-in-turn-1 followed by any turn 2 (e.g., "Write hello world in C" → "Now in Python"). This reproduces identically with single-pass prompt encoding, confirming it is a model limitation — not an inference engine bug.

### Long responses degenerate into repetition

The model cannot reliably sustain coherent generation beyond ~50-100 tokens. Open-ended prompts that trigger verbose responses (numbered lists, detailed explanations) often degenerate into repetitive n-gram loops. A built-in loop detector stops generation when this is detected, but the degenerate tokens already written to the KV cache can affect subsequent turns.

Both limitations are inherent to the model's capacity (2B parameters at 1.58-bit quantization) and are not addressable at the inference engine level.

## Acknowledgments

This project stands on the shoulders of giants:

- **[llama2.c](https://github.com/karpathy/llama2.c)** by Andrej Karpathy — the inspiration for proving that a complete LLM inference engine can be beautifully simple. The modular architecture, mmap-based loading, and "make it work in pure C" philosophy all trace directly to Karpathy's pioneering work.

- **[BitNet](https://github.com/microsoft/BitNet)** by Microsoft Research — the breakthrough 1.58-bit quantization research ([paper](https://arxiv.org/abs/2402.17764)) that makes ternary-weight transformers practical. The official framework and model weights make this project possible.

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** / **[GGML](https://github.com/ggml-org/ggml)** by Georgi Gerganov — the GGUF file format, TQ1_0/TQ2_0 quantization implementations, and the broader ecosystem of efficient LLM inference.

## License

[MIT](LICENSE)
