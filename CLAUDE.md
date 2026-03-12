# CLAUDE.md

Instructions for Claude Code when working on this project.

## Project Overview

bitnet.c is a pure C11 inference engine for BitNet b1.58 transformer models. It loads GGUF model files and performs autoregressive text generation. Inspired by Karpathy's llama2.c.

## Build

```bash
make          # build the main binary
make debug    # build with -DDEBUG -g -O0
make clean    # remove artifacts
make test     # run all unit tests
```

Individual test targets: `make test_gguf`, `make test_quant`, `make test_tokenizer`, `make test_transformer`.

## Architecture

Modules are organized in strict dependency order — each depends only on those above it:

1. `platform` — mmap/buffer abstraction, timing
2. `gguf` — GGUF v3 binary parser (standalone)
3. `quant` — TQ1_0/TQ2_0 dequantization, ternary matvec (standalone)
4. `model` — GGUF → Config/Weights mapping (depends on gguf + quant)
5. `tokenizer` — BPE tokenizer (depends on gguf)
6. `transformer` — forward pass (depends on model + quant)
7. `sampler` — sampling strategies (standalone)
8. `main` — CLI wiring

Headers live in `include/`, implementations in `src/`, tests in `test/`.

## Code Style

- C11, compiled with `-Wall -Wextra`
- No external dependencies (only libc + libm)
- Use `#ifdef __EMSCRIPTEN__` for WASM-specific code paths
- Use `#ifdef DEBUG` for debug-only logging
- Prefix public API functions with module name: `gguf_`, `model_`, `tokenizer_`, etc.
- Static helper functions are file-local
- Memory: caller allocates structs, modules fill them. Use `_init`/`_free` pairs.

## Key Types

- `MappedFile` — wraps mmap'd or malloc'd buffer
- `GGUFFile` — parsed GGUF header, KV pairs, tensor info
- `BlockTQ1` / `BlockTQ2` — quantized weight blocks (54 / 66 bytes per 256 elements)
- `QWeight` — ternary weight tensor descriptor (zero-copy into GGUF buffer)
- `Config` — model hyperparameters
- `Model` — config + weights + runtime state
- `Tokenizer` — BPE vocab + sorted index for encoding
- `Sampler` — sampling parameters + RNG state

## Testing

Tests use assert-based checks with synthetic data — no real model files needed for unit tests. Each test file is self-contained and can be compiled independently with its module dependencies.

`test_e2e.c` requires a real GGUF model file: `./test_e2e model.gguf`

## WASM

WASM build requires Emscripten. Run `./wasm/build.sh`. The API wrapper in `wasm/api.c` exports functions via `EMSCRIPTEN_KEEPALIVE`. The browser demo uses a Web Worker for non-blocking inference.

## Common Tasks

- **Add a new GGUF metadata key**: read it in `model_load()` in `src/model.c`
- **Add a new tensor type**: add block struct to `include/quant.h`, dequant function to `src/quant.c`
- **Modify the forward pass**: edit `transformer_forward()` in `src/transformer.c`
- **Add a new sampling strategy**: extend `sampler_sample()` in `src/sampler.c`
- **Export a new function to WASM**: add `EMSCRIPTEN_KEEPALIVE` wrapper in `wasm/api.c`, update `build.sh` exported functions list
