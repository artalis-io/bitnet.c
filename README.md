# bitnet.c

A minimal, embeddable LLM inference engine in pure C11.

Loads GGUF models and runs autoregressive text generation on CPU. Supports 20+ quantization formats — from ternary (I2_S, TQ1/TQ2) through k-quants (Q2–Q8) and imatrix (IQ2–IQ4) to unquantized (F16, BF16, F32). Handles both standard transformer and hybrid SSM+Attention architectures (Gated DeltaNet). Inspired by Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

Zero dependencies beyond libc and libm, four SIMD backends, compiles to WASM, and fits in ~8,000 lines of modular C.

## Features

- **Pure C11** — no C++, no frameworks, no dependencies beyond libc and libm
- **GGUF model loading** — loads any GGUF file with supported tensor types
- **20+ quantization formats** — ternary, k-quants, imatrix codebook, and unquantized (see table below)
- **Full transformer forward pass** — RoPE, GQA, RMSNorm, sub-norms, tied/untied embeddings
- **Hybrid SSM + Attention** — Gated DeltaNet SSM layers (conv1d, SiLU, delta rule recurrence) alongside standard GQA attention layers
- **Flash GQA attention** — online softmax with KV-head grouping, single-pass over KV cache
- **Optional F16 KV cache** — `--kv16` halves attention DRAM bandwidth with minimal precision loss
- **4 SIMD backends** — ARM NEON/SDOT, AVX2, WASM SIMD128, scalar fallback (auto-selected at compile time)
- **Mixture of Experts (MoE)** — sparse MoE with top-K routing, batched expert dispatch, 3 I/O modes (mmap, pread+LRU cache, madvise)
- **Pthread thread pool** — persistent workers with atomic work-stealing dispatch
- **BPE tokenizer** — loaded directly from GGUF metadata
- **Sampling** — greedy (argmax), multinomial, and nucleus (top-p)
- **Native mmap** — zero-copy model loading on macOS/Linux
- **WASM build** — runs in the browser via Emscripten with Web Worker streaming

## Quantization Formats

### Ternary / Binary

| Format | bpw | Block | Packing |
|--------|-----|-------|---------|
| I2_S   | 2.0 | 128 | 2-bit interleaved + per-tensor scale |
| TQ1_0  | 1.69 | 256 | Base-3 (5 values/byte) + residual |
| TQ2_0  | 2.06 | 256 | 2-bit fields (4 values/byte) |

### K-Quants

| Format | bpw | Block | Packing |
|--------|-----|-------|---------|
| Q2_K | 2.63 | 256 | 2-bit quants + 4-bit sub-block scales/mins |
| Q3_K | 3.44 | 256 | 3-bit quants (split ql/qh) + 6-bit scales |
| Q4_0 | 4.50 | 32 | 4-bit nibbles + FP16 per-block scale |
| Q4_1 | 5.00 | 32 | 4-bit nibbles + FP16 scale + FP16 min |
| Q4_K | 4.50 | 256 | 4-bit quants + 6-bit sub-block scales/mins |
| Q5_K | 5.50 | 256 | 5-bit quants (split ql/qh) + 6-bit scales/mins |
| Q6_K | 6.56 | 256 | 6-bit quants (split ql/qh) + int8 scales |
| Q8_0 | 8.50 | 32 | 8-bit values + FP16 per-block scale |
| Q8_K | 9.13 | 256 | 8-bit values + float32 scale + int16 sums |

### Importance-Matrix (imatrix) Codebook

| Format | bpw | Block | Packing |
|--------|-----|-------|---------|
| IQ2_XXS | 2.06 | 256 | 2-bit codebook (256-entry grid) + packed signs/scales |
| IQ2_XS  | 2.31 | 256 | 2-bit codebook (512-entry grid) + explicit scales |
| IQ2_S   | 2.56 | 256 | 2-bit codebook (1024-entry grid) + scales |
| IQ3_XXS | 3.06 | 256 | 3-bit codebook (256-entry grid) + packed signs/scales |
| IQ3_S   | 3.56 | 256 | 3-bit codebook (512-entry grid) + separate signs/scales |
| IQ4_NL  | 4.50 | 32 | 4-bit non-linear (16-entry LUT) |
| IQ4_XS  | 4.25 | 256 | 4-bit non-linear + 6-bit sub-block scales |

### Unquantized

| Format | bpw | Packing |
|--------|-----|---------|
| F16  | 16 | IEEE 754 half-precision |
| BF16 | 16 | bfloat16 (truncated float) |
| F32  | 32 | IEEE 754 single-precision |

## SIMD Backends

| Backend | Platform | Notes |
|---------|----------|-------|
| ARM NEON/SDOT | Apple Silicon, ARMv8 | SDOT int8 matvec, native FP16 logits |
| AVX2 | x86-64 (Haswell+) | DPBUSD int8 matvec, F16C conversion |
| WASM SIMD128 | Browser (Emscripten) | 128-bit vector ops |
| Scalar | Any C11 compiler | Portable fallback |

Auto-selected at compile time based on target architecture.

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
  --pread         Force pread for MoE expert loading (lower RSS than mmap)
  --cache-mb <N>  Expert LRU cache budget in MB (default: 4096, 0 to disable)
  --madvise       madvise-guided mmap for MoE (experimental)
  --draft <path>  Draft model for speculative decoding (greedy, same tokenizer required)
  --draft-k <int> Draft tokens per iteration (default: 5)
  -t <int>        Number of threads (default: auto-detect)
```

### Chat Mode

`--chat` enters an interactive REPL with multi-turn conversation support. KV cache is reused across turns so context accumulates naturally. Type `/quit` or Ctrl-D to exit.

Chat mode defaults to `--temp 0.5 --topp 0.9 --repeat-penalty 1.1` for more natural conversation. Override with explicit flags.

### MoE Expert I/O Modes

For Mixture of Experts models (Qwen3-MoE, OLMoE, Mixtral), expert weights can be loaded in three modes:

- **mmap** (default): Direct memory-mapped access. Full model in RAM. Fastest throughput with cross-expert batched dispatch.
- **pread** (`--pread`): SSD streaming via pread syscalls with 2 prefetch threads and an LRU expert cache. Uses `--cache-mb` to control cache budget (default 4096 MB). Lower RSS at the cost of I/O latency.
- **madvise** (`--madvise`): Mmap with `MADV_WILLNEED` prefetch hints per expert. Experimental — syscall overhead currently negates the benefit.

```bash
# Default: mmap (all experts in RAM)
./bitnet models/Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello" -n 64

# Pread with 4GB expert cache (lower RSS)
./bitnet models/Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello" -n 64 --pread

# Pread with custom cache size
./bitnet models/Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello" -n 64 --pread --cache-mb 2048
```

## Getting a Model

Any GGUF model using supported weight types works. Download from HuggingFace:

```bash
mkdir -p model
pip install huggingface-hub

# Example: Qwen2.5-3B (Q4_K_M quantization)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  --include "qwen2.5-3b-instruct-q4_k_m.gguf" \
  --local-dir model/

# Example: BitNet ternary model
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \
  --include "bitnet-b1.58-2B-4T-TQ2_0.gguf" \
  --local-dir model/
```

Then run:

```bash
./bitnet model/qwen2.5-3b-instruct-q4_k_m.gguf -p "The capital of France is"
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
│   ├── iq_tables.h             # IQ codebook lookup tables (shared across backends)
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
│   │   ├── fp16.c              # FP16/BF16 ↔ FP32 conversion
│   │   └── {format}_{backend}.c # ~80 backend kernel files
│   ├── transformer/            # Per-backend transformer kernels
│   │   ├── rmsnorm_{neon,avx2,wasm,scalar}.c
│   │   ├── gqa_{neon,avx2,wasm,scalar}.c
│   │   └── logits_{neon,avx2,wasm,scalar}.c
│   ├── model.c                 # GGUF → Config/Weights mapping
│   ├── transformer.c           # Forward pass: layer loop, FFN, dispatch
│   ├── tokenizer.c             # BPE encode/decode from GGUF vocab
│   ├── sampler.c               # Argmax, multinomial, top-p sampling
│   ├── threadpool.c            # Thread pool with condvar dispatch
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

## Model Support

Tested models with generation quality and performance on Apple M1 Max (8 P-cores, 32 GB), release build, greedy decoding, 8 threads:

| Model | Size | Quant | Architecture | tok/s | Quality |
|-------|------|-------|--------------|-------|---------|
| bitnet-b1.58-2B-4T | 1.1 GB | I2_S (ternary) | Transformer | **29–41** | Factual, correct code |
| Qwen2.5-3B-Instruct | 1.7 GB | Q4_0 | Transformer | **23–25** | Coherent, instruct-following |
| Llama3-8B-1.58 | 3.3 GB | TQ1_0 (ternary) | Transformer | **8–9** | Basic completion |
| Qwen3.5-9B | 5.3 GB | Q4_K_M (mixed) | Hybrid SSM+Attention | **2.8–3.1** | Best quality, chain-of-thought |

Any GGUF model using supported weight types works — these are just the tested configurations.

### Hybrid SSM + Attention

bitnet.c supports hybrid architectures that mix SSM (state space model) layers with standard attention layers, such as Qwen3.5's Gated DeltaNet:

- **SSM layers**: Conv1d (kernel=4) → SiLU → L2-norm Q/K → delta rule recurrence → per-head RMSNorm → SiLU gate → output projection
- **Attention layers**: Standard GQA with RoPE, flash attention, KV cache
- **Mixed layout**: The model config specifies which layers use SSM vs attention

## Performance

Measured on Apple M1 Max (8 P-cores, 32 GB), PGO build, greedy decoding, 8 threads. llama.cpp b8320 via Homebrew.

| Model | Size | Quant | bitnet.c | llama.cpp CPU | llama.cpp Metal |
|-------|------|-------|----------|---------------|-----------------|
| bitnet-b1.58-2B-4T | 620 MB | I2_S (ternary) | **52 tok/s** | — | — |
| Qwen2.5-3B-Instruct | 1.7 GB | Q4_0 | **30 tok/s** | 40 tok/s | 84 tok/s |
| Llama3-8B-1.58 | 3.4 GB | TQ1_0 (ternary) | **14.5 tok/s** | 19 tok/s | — |

bitnet.c is a pure CPU engine with no GPU backend. On ternary models (TQ1_0) it reaches **76% of llama.cpp CPU** — close to parity. On standard quants (Q4_0) it reaches **75% of llama.cpp CPU** using multi-row interleaved kernels (4 output rows per pass, amortizing activation loads). llama.cpp does not support TQ1_0 on Metal.

## Design Decisions

### Why not llama.cpp / Ollama?

llama.cpp supports dozens of quantization formats, model architectures, GPU backends (CUDA, Metal, Vulkan, SYCL), and serving modes. That generality costs ~200K lines of C++ and a build system that pulls in platform-specific SDKs.

bitnet.c exists as the opposite tradeoff:

- **Embeddable.** ~8,000 lines of C11, zero dependencies. Compiles to WASM and runs in a browser. Link it into your app as a static library.
- **No GPU backend.** For models that fit in RAM, CPU inference with good SIMD kernels is fast enough and avoids GPU dispatch overhead, driver dependencies, and framework complexity.
- **No abstraction layers.** llama.cpp routes tensor operations through GGML's graph-based backend abstraction. bitnet.c calls the matvec kernel directly — the forward pass is a flat loop over layers with inline SIMD.
- **Fast builds.** Clean build under 3 seconds. No CMake, no Python, no package managers.

**When to use llama.cpp/Ollama instead:** GPU inference, models too large for CPU, OpenAI-compatible API serving, or multi-model management.

### Why C

The inner loop is SIMD intrinsics — the code *is* the assembly, minus register allocation. The entire forward pass is a `for` loop over layers. The GGUF parser is pointer arithmetic into a mmap'd buffer. C gives you exactly what's needed and nothing else.

Rust would wrap every mmap pointer in `unsafe`. C++ would template the tensor types. Go can't do NEON. Python would need ctypes back into C. For a DRAM-bandwidth-bound SIMD kernel, the right language compiles directly to the SIMD instructions.

## Known Limitations

- **Small ternary models degrade on multi-turn chat.** The 2B 1.58-bit BitNet model (~400 MB effective weights) struggles with multi-turn conversations, especially after code-heavy responses. This is a model capacity limitation, not an inference engine bug.
- **Long generation degenerates into repetition.** Small models cannot reliably sustain coherent generation beyond ~50-100 tokens. A built-in loop detector stops generation when this is detected.

## Acknowledgments

- **[llama2.c](https://github.com/karpathy/llama2.c)** by Andrej Karpathy — the inspiration for proving that a complete LLM inference engine can be beautifully simple.
- **[BitNet](https://github.com/microsoft/BitNet)** by Microsoft Research — the 1.58-bit quantization research ([paper](https://arxiv.org/abs/2402.17764)) that makes ternary-weight transformers practical.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** / **[GGML](https://github.com/ggml-org/ggml)** by Georgi Gerganov — the GGUF file format, quantization implementations, codebook tables, and the broader ecosystem.

## License

[MIT](LICENSE)
