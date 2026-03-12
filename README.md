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

Download a BitNet GGUF model. For example, `bitnet-b1.58-2B-4T`:

```bash
# Using huggingface-cli
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/

# Or download directly
wget https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/resolve/main/bitnet-b1.58-2B-4T-TQ1_0.gguf
```

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
