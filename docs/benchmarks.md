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

make BN_ENABLE_METAL=1 bench_llama_topk
make BN_ENABLE_METAL=1 bench_llama_topk_server
make BN_ENABLE_METAL=1 bench_kernels
./bench_kernels models/qwen2.5-3b-instruct-q4_0.gguf --metal --iters 20 --toks 8
./bench_kernels models/qwen2.5-3b-instruct-q4_0.gguf --metal \
  --metal-disable-q4-q8 --iters 20 --toks 8
./bench_kernels models/qwen2.5-3b-instruct-q4_0.gguf --metal \
  --metal-enable-q6-q8k --iters 20 --toks 8

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
| bitnet.c Metal | 96.55 |
| llama.cpp Metal `-fa on -np 1` | 117.71 |

Ratio: 0.820. Top-1 matched `8/8` prompts and mean top-10 overlap was `9.62`.
This is good coherence evidence, but it is not yet throughput parity.

Follow-up checks:

- A later self-contained `make BN_ENABLE_METAL=1 LLAMA_TOPK_PORT=8039
  bench_llama_topk_server` run reached top-1 `8/8`, mean top-10 overlap 9.62,
  bitnet.c 96.83 tok/s, and llama.cpp 97.39 tok/s. The ratio was 0.994, so
  the strict default `--min-throughput-ratio 1.0` gate still failed. Because
  the llama.cpp server number was much lower than the earlier 115-118 tok/s
  samples, treat this as a near-miss checkpoint rather than a parity claim.
- The comparator now supports `--bench-runs N` and uses median throughput for
  the ratio. The default Makefile gate uses `--bench-runs 3` to avoid accepting
  or rejecting parity on a single noisy throughput sample, and the managed
  helper chooses a free localhost port when `LLAMA_TOPK_PORT=0`. The latest
  target-level `make BN_ENABLE_METAL=1 bench_llama_topk_server` run measured
  bitnet.c samples `[94.92, 95.09, 95.37]` tok/s and llama.cpp samples
  `[115.96, 116.45, 115.61]` tok/s, giving medians 95.09 and 115.96
  respectively, ratio 0.820.
- CPU-only, same Qwen2.5 Q4_0 gate, bitnet.c `--flash` versus
  `llama-server -ngl 0 -fa on -c 512 -np 1`: bitnet.c measured
  43.24 tok/s, llama.cpp measured 45.42 tok/s, ratio 0.952. Mean top-10
  overlap was 9.50, but one prompt swapped the top two logits, so this is
  near-throughput parity rather than a clean top-1 parity result.
- Existing bitnet.c Metal `--kv16` is not an acceptance shortcut yet. On the
  same direct 64-token prompt, `--metal --flash --kv16` measured 42.60 tok/s,
  far below the FP32-KV Metal path. Do not enable it in the strict gate until
  the Metal KV16 path is made native and fast.
- llama.cpp build `8320` (`128142fe7`) uses a much more parallel Metal
  flash-attention implementation family (`kernel_flash_attn_ext*`), including
  head-size-specialized vector kernels and f16/quantized KV variants. bitnet.c's
  Metal flash shader now uses a bounded, short-context, one-head-per-threadgroup
  scores/softmax/combine fusion, but forcing it below the current short-context
  threshold still does not satisfy the strict 128-token server gate. With
  `--gpu-flash-min-kv 0`, the latest sample matched top-1 on `8/8` prompts with
  mean top-10 overlap `9.62`, but measured 95.27 tok/s versus llama.cpp
  118.88 tok/s (ratio 0.801). Keep the default threshold conservative until
  the Metal flash path is tiled/chunked enough to beat the non-flash
  scores/softmax/combine path at the acceptance length.
- `BN_GPU_PROFILE=4` now reports per-shape Metal timing for matvec-like ops.
  The latest diagnostic frame on Qwen2.5 Q4_0 attributes roughly 3.21ms/token
  to fused gate/up, 2.06ms/token to FFN down matvecs, 1.45ms/token to the
  Q6_K logits matvec, 0.73ms/token to stacked QKV split matvecs, and
  0.61ms/token to dense attention/output matvecs. Attention score/softmax/
  combine together remain under 0.75ms/token at this acceptance length, so the
  current gap is still FFN/logits dominated. A 16-thread-per-row native Q4 fused gate/up experiment
  regressed the 128-token direct run to 96.41 tok/s, so it was not kept.
  `--gpu-disable-fused-gateup` also regressed the same direct run to
  95.69 tok/s, confirming that the fused path is still preferable even though
  it is the hottest kernel family. `--metal-private-weights` measured
  93.82 tok/s and is not a parity lever on this M1 Max setup.
  `--metal-disable-barriers` matched top-1 on `8/8` prompts with mean top-10
  overlap 9.62, but measured 96.59 tok/s versus llama.cpp 117.34 tok/s
  (ratio 0.823), so explicit Metal barriers should stay enabled.
  `--metal-q4-prepared` exposed the existing prepared Q4_0 Metal upload layout
  to the main benchmark path, but it regressed the same 128-token direct run to
  77.04 tok/s, so the default packed Q4_0 layout remains the better gate path.
  `--metal-disable-q4-q8` measured 91.53 tok/s, so the Q4_0 x Q8 activation
  path should stay enabled by default. `--gpu-split-residual-rmsnorm` measured
  89.18 tok/s, so the fused residual RMSNorm path should also stay enabled.
  `--gpu-cpu-logits` measured 85.45 tok/s, so the GPU Q6_K logits path remains
  preferable despite being visible in the profile. A Q4_0 x Q8 shader
  experiment that accumulated each `char4` dot with integer arithmetic instead
  of the current `float4` dot path measured 94.42 tok/s, so the float4 dot path
  remains the better kernel variant on this setup. Q4_0 x Q8 threadgroup
  geometry sweeps also did not improve the gate: 32-row groups measured
  95.75 tok/s and 8-row groups measured 96.22 tok/s, so the current 16-row
  geometry remains the best tested setting. The opt-in `--metal-enable-q6-q8k`
  logits path now uses a parallel Metal Q8_K activation quantizer and a
  vectorized Q6_K x Q8_K matvec shaped like the default Q6_K float-vector
  shader. The corrected llama-server gate improved to 72.04 tok/s versus
  llama.cpp at 117.20 tok/s with top-1 `8/8` and mean top-10 overlap 9.75, but
  it remains slower than the default Q6_K logits path and should stay
  diagnostic-only. Reducing CPU threads with
  `-t 1` measured 91.62 tok/s, so the default thread setting remains better
  even for GPU decode. The default Q4_0 x Q8 activation policy applies to
  layers 0-2 on this 36-layer model; extending it to all layers with
  `--q4-q8-to-layer 35` measured 84.57 tok/s, and extending it to layers 0-4
  measured 94.42 tok/s, so the conservative first-three-layer policy remains
  the best tested setting. Restricting the same first-three-layer range to
  attention-only measured 94.73 tok/s, and FFN-only measured 92.19 tok/s, so
  applying Q4_0 x Q8 to both attention and FFN in those early layers remains
  preferable. `--q4-q8-disable-gateup` uses the native Q4 fused gate/up shader
  while leaving Q4_0 x Q8 enabled elsewhere; it improved the direct
  `bench_kernels` fused row from 338.5 us/call to 303.2 us/call, but the strict
  llama-server gate remained essentially flat at 94.91 tok/s versus llama.cpp
  115.37 tok/s (ratio 0.823, top-1 `8/8`, mean top-10 overlap 9.75), so it is
  diagnostic-only for now. `--q4-q8-disable-ffn-down` also stayed flat at
  94.90 tok/s versus llama.cpp 115.72 tok/s (ratio 0.820, top-1 `8/8`, mean
  top-10 overlap 9.88). Combining `--q4-q8-disable-gateup` with
  `--q4-q8-disable-ffn-down` measured 95.76 tok/s versus llama.cpp 115.40 tok/s
  (ratio 0.830, top-1 `8/8`, mean top-10 overlap 10.00), so native-FFN policy
  toggles are not enough to close the Metal gap. The next Q4_0 FFN kernel
  direction should be row-grouped matvec/gateup work that reuses each activation
  slice across multiple output rows, similar to llama.cpp's
  `mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>` structure; bitnet.c's current
  kernels mostly assign one output row to each 8-lane group. A first
  row-grouped two-row Q4_0 x Q8 matvec diagnostic was not kept: it measured
  94.50 tok/s versus llama.cpp 99.76 tok/s on that run, and the direct
  `bench_kernels` FFN `up`/`down` rows were slower than the default Q4/Q8
  shader. Lightweight `--gpu-profile 1` timing shows the warmed decode
  frames spend about 8.9-9.6ms in GPU execution, 0.1ms in encoding, and
  effectively 0.0ms in logits readback, so reducing full-logit readback is not
  expected to close the current Metal parity gap on this setup.
- `bench_kernels --metal` now uses GPU-resident weights for per-matrix matvec
  timing, including the quantized output/logits matrix, instead of accidentally
  measuring the CPU quant path. Use `./bench_kernels model.gguf --metal
  --metal-enable-q6-q8k` when iterating on the opt-in Q6_K x Q8_K logits
  diagnostic. A 20-iteration Qwen2.5 sample measured the default Metal Q6_K
  logits row at 1786.4 us/call for 151936 x 2048. The older scalar
  `--metal-enable-q6-q8k` shader measured 11898.2 us/call; the vectorized
  Q6_K x Q8_K shader reduced that to 5178.5 us/call. That is a material
  improvement, but still about 2.9x slower than the default Q6_K Metal logits
  kernel, so it remains diagnostic-only. The same default sample measured
  layer-0 Q4_0 `up` at 283.0 us/call, `gate` at 287.8 us/call, and `down` at
  294.3 us/call; small K/V projections are dominated by fixed dispatch
  overhead. Use this microbenchmark for shader iteration, then confirm
  candidates with the top-k llama-server gate.
- The same benchmark also reports `gateup*`, the graph-lowered production
  fused gate/up SiLU op on the stacked gate/up buffer. The first Metal sample
  measured layer-0 Q4_0 `gateup*` at 338.5 us/call for 22016 x 2048, giving
  the next FFN shader iterations a direct hotspot target instead of relying
  only on separate `gate` and `up` matvec timings.

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
