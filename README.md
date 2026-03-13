# bitnet.c

A clean-room, pure C inference engine for [BitNet b1.58](https://arxiv.org/abs/2402.17764) transformer models.

Inspired by Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) — a beautifully minimal LLaMA inference implementation in a single C file — **bitnet.c** takes the same philosophy and applies it to Microsoft's [BitNet](https://github.com/microsoft/BitNet) architecture with its 1.58-bit ternary weights.

Where Microsoft's official BitNet inference framework depends on a modified llama.cpp fork (~100K+ lines of C++), bitnet.c delivers a complete inference pipeline in ~2,700 lines of modular, readable C.

## Features

- **Pure C11** — no C++, no frameworks, no dependencies beyond libc and libm
- **GGUF model loading** — compatible with existing BitNet GGUF files (e.g. `bitnet-b1.58-2B-4T`)
- **TQ1_0 & TQ2_0 dequantization** — native support for ternary quantization formats
- **Full transformer forward pass** — RoPE, GQA, RMSNorm, sub-norms, tied embeddings
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
│   ├── quant.h         # TQ1_0/TQ2_0 dequantization + ternary matmul
│   ├── model.h         # Config, Weights, model loading
│   ├── transformer.h   # Forward pass API
│   ├── tokenizer.h     # BPE tokenizer API
│   └── sampler.h       # Sampling strategies
├── src/
│   ├── platform.c      # mmap/fread/timing abstraction
│   ├── gguf.c          # GGUF binary format parser
│   ├── quant.c         # FP16 conversion, TQ1/TQ2 dequant, ternary matvec
│   ├── model.c         # GGUF → Config/Weights mapping
│   ├── transformer.c   # Forward pass: attention, FFN, sub-norms
│   ├── tokenizer.c     # BPE encode/decode from GGUF vocab
│   ├── sampler.c       # Argmax, multinomial, top-p sampling
│   └── main.c          # CLI entry point
├── test/
│   ├── test_gguf.c     # GGUF parser tests
│   ├── test_quant.c    # Dequantization + matvec tests
│   ├── test_transformer.c  # RMSNorm, softmax, RoPE tests
│   ├── test_tokenizer.c    # BPE encode/decode tests
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

**~46 tok/s** (256 tokens generated)

### Optimization History

| Optimization | tok/s | Speedup |
|---|---|---|
| Baseline (scalar C) | ~15.5 | 1.0x |
| + SDOT int8 accumulation + batch matvec | ~33 | 2.1x |
| + Arithmetic ternary decode + RoPE precompute | ~38 | 2.5x |
| + Pthread thread pool (replace OpenMP) | ~38 | 2.5x |
| + Arena allocator + native FP16 logits + prefetch | **~46** | **3.0x** |

### Key Optimizations

- **SDOT (vdotq_s32)** — ARM dot product instructions for I2_S ternary matvec. Quantize activations to int8 once, then use integer dot products instead of float FMA. 2x speedup.
- **Batch matvec** — group independent projections (QKV, gate+up) into a single thread pool dispatch, sharing the int8-quantized activation vector.
- **Pthread thread pool** — persistent worker threads with condvar dispatch (~2us), replacing OpenMP fork/join (~15us on macOS).
- **Native FP16 logits** — `-mcpu=apple-m1` enables `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` for native F16 FMA in the logits computation (embedding dot products). Apple clang's `-march=native` misses this.
- **Arena allocator** — all RunState buffers in a single contiguous allocation for better TLB coverage.

### Bandwidth Analysis

At ~46 tok/s the workload is **DRAM bandwidth-bound**, reading ~1.15 GB per token:

| Component | Data/Token |
|---|---|
| Layer weights (30 layers, I2_S) | 497 MB |
| Logits (F16 embeddings, 128K vocab) | 656 MB |

This sustains ~53 GB/s — near the M1 Max CPU bandwidth ceiling (~55-60 GB/s). Further gains require reducing data volume (e.g. INT8 output embeddings).

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
| KV cache (30 layers × 2048 × 640 × 4 × 2) | ~298 MB |
| RunState activations | ~3 MB |
| **Total** | **~921 MB** |

## Acknowledgments

This project stands on the shoulders of giants:

- **[llama2.c](https://github.com/karpathy/llama2.c)** by Andrej Karpathy — the inspiration for proving that a complete LLM inference engine can be beautifully simple. The modular architecture, mmap-based loading, and "make it work in pure C" philosophy all trace directly to Karpathy's pioneering work.

- **[BitNet](https://github.com/microsoft/BitNet)** by Microsoft Research — the breakthrough 1.58-bit quantization research ([paper](https://arxiv.org/abs/2402.17764)) that makes ternary-weight transformers practical. The official framework and model weights make this project possible.

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** / **[GGML](https://github.com/ggml-org/ggml)** by Georgi Gerganov — the GGUF file format, TQ1_0/TQ2_0 quantization implementations, and the broader ecosystem of efficient LLM inference.

## License

[MIT](LICENSE)
