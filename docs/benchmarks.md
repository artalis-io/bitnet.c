# Benchmarks

Benchmark numbers in this repository are local checkpoints. They are useful for
detecting regressions and comparing implementation choices on the same machine;
they should not be treated as universal performance claims.

Important variables:

- model family and quant format
- prompt length and generated token count
- thread count
- backend placement: scalar, NEON, AVX2, WASM SIMD, Metal, WebGPU, llama.cpp CPU,
  or llama.cpp Metal
- mmap versus pread and page-cache state for MoE models
- `--maxseq`, KV mode, and whether prefill is enabled

## Reproducible Gates

```bash
make bench_llama_compare
make bench
./bench_kernels models/model.gguf --iters 100 --threads 4 --toks 32
```

GPU/coherence-adjacent checks:

```bash
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal

llama-server -m models/qwen2.5-3b-instruct-q4_0.gguf \
  -ngl 99 -fa on -c 512 -np 1 --host 127.0.0.1 --port 8027
python3 test/compare_llama_topk.py models/qwen2.5-3b-instruct-q4_0.gguf \
  --metal --llama-metal --flash --maxseq 512 \
  --gpu-max-storage-binding-mb 4096 \
  --top-k 10 --min-overlap 3 \
  --llama-server-url http://127.0.0.1:8027 \
  --benchmark

make fetch-wgpu
make BN_ENABLE_WEBGPU=1 test_gpu_wgpu
```

## Current Local Checkpoint

The active Metal acceptance gate is top-logit coherence plus token throughput
against llama.cpp served with Metal and flash attention enabled (`llama-server
-fa on -np 1`). The comparator defaults `--min-throughput-ratio` to `1.0`, so
`--benchmark` fails until bitnet.c is at least on par with the llama.cpp server
sample.

The latest local strict server-mode sample used `qwen2.5-3b-instruct-q4_0.gguf`,
`--maxseq 512`, `--flash`, deterministic sampling, 128 generated tokens, and
Qwen2.5 top-logit prompts:

| Engine | tok/s |
|---|---:|
| bitnet.c Metal | 91.58 |
| llama.cpp Metal `-fa on -np 1` | 102.20 |

Ratio: 0.896. Top-1 matched `8/8` prompts and mean top-10 overlap was `9.62`.
This is good coherence evidence, but it is not yet throughput parity.

An older 32-token `make bench_llama_compare` checkpoint measured median bitnet.c
at 39.30 tok/s versus median llama.cpp at 17.59 tok/s. Treat that as historical
only; the current acceptance bar is the top-k plus llama-server gate above.

## MoE Notes

MoE models are strongly affected by page-cache state and expert locality.

- `mmap` usually gives the best throughput when the model fits in RAM and the
  expert working set is warm.
- `--pread --cache-mb N` lowers RSS and can be preferable for serving larger
  sparse models.
- No-cache pread is a memory-saving fallback and is normally slower.

When comparing against llama.cpp, record whether llama.cpp is actually CPU-only
or whether it routes some work to Metal/GPU even when a CPU-looking flag is used.

## GPU Notes

Metal and WebGPU use the `BnGPUBackend` command contract. Backend performance is
limited by the availability and quality of native kernels for the selected quant
and op kind.

Current caveats:

- WebGPU runtime checks may skip on machines where wgpu-native reports no
  suitable adapter.
- Unsupported SSM or MoE blocks can fall back to CPU.
- Oversized bindings can force CPU logits fallback on constrained adapters.
- Native-layout Q4_0 and broader low-bit GPU kernels remain optimization work.

## Historical Context

Earlier measurements on an Apple M1 Max showed the CPU path reaching high memory
bandwidth utilization on BitNet ternary models and competitive throughput on
several dense and sparse quantized models. Keep old numbers in commit history;
current docs should prefer reproducible commands and recent gates over broad
claims.
