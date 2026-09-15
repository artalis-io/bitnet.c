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

Parity ratios require the same runtime axis on both sides. In particular, a
`bitnet_scalar` result cannot be divided by a native Apple llama.cpp result,
because that llama.cpp build uses NEON. The comparator rejects this combination.
Use matching `--bitnet-runtime scalar --llama-runtime scalar` labels together
with `--llama-bench-bin` pointing at an actual scalar llama.cpp build.

Prompt processing has a separate gate from token generation. For example, an
AVX2-only comparison at the accepted 85% floor is:

```bash
python3 test/compare_llama_topk.py model.gguf \
  --bitnet ./bitnet_avx2 \
  --bitnet-runtime avx2 --llama-runtime avx2 \
  --llama-bench-bin /path/to/avx2/llama-bench \
  --skip-topk --benchmark-prefill --bench-prompt-tokens 128 \
  --bench-runs 3 --min-prefill-throughput-ratio 0.85 -t 20
```

Use `--benchmark` and `--min-throughput-ratio` for decode throughput. Both
benchmark modes reject mismatched runtime axes and contaminated host load.

## Latest measured x86 checkpoint

Qwen3.5 sparse 122B-A10B MXFP4 passes on the `q5-no-unused-pack` artifacts:

| Backend | Token IDs | Prefill / llama | Decode / llama |
| --- | ---: | ---: | ---: |
| AVX2 | 192/192 | 32.27 / 35.86 tok/s (90.0%) | 5.30 / 5.72 tok/s (92.7%) |
| AVX512 | 192/192 | 37.16 / 40.48 tok/s (91.8%) | 5.04 / 5.58 tok/s (90.3%) |

Eight threads, default caches, context 512, flash off, pp128/tg64 speed
gates with three bitnet runs. Token parity uses F32 KV both; speed uses
F32 bitnet KV versus F16 llama KV. Artifact hashes, test coverage and logs
are in [the detailed checkpoint](#avoid-unused-x86-q5_k-preparation).
This does not establish completion of the full requested model/backend
matrix. Other sections below record distinct, often older artifacts and
configurations rather than a single fully revalidated release.

The [Qwen3 dense refresh](#qwen3-dense-current-artifact-refresh) retains
192/192 token parity on both x86 artifacts. AVX2's three- and five-run
speed gates pass. Its AVX512 prefill repeat failed narrowly; the newer
[Q6_K row-pair candidate](#q6_k-avx512-row-pair-prefill) retains 192/192
parity and passes two isolated five-run gates: prefill 89.8%/92.4%, decode
92.4%/93.4%. The AVX2 binary is unchanged. These are per-fixture checkpoints,
not a full-matrix acceptance of the latest AVX512 binary.

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
  --metal-disable-small-dense-native-quant --iters 20 --toks 8
./bench_kernels models/qwen2.5-3b-instruct-q4_0.gguf --metal \
  --metal-specialized-native-quant --iters 20 --toks 8

make fetch-wgpu
make BN_ENABLE_WEBGPU=1 test_gpu_wgpu
```

## Earlier Metal Checkpoint

This Metal checkpoint used top-logit coherence plus token throughput
against llama.cpp served with Metal and flash attention enabled (`llama-server
-fa on -np 1`). The comparator defaults `--min-throughput-ratio` to `1.0`, so
`--benchmark` fails until bitnet.c is at least on par with the llama.cpp server
sample.

The latest local strict server-mode sample used `qwen2.5-3b-instruct-q4_0.gguf`,
`--maxseq 512`, `--flash`, deterministic sampling, 128 generated tokens, and
Qwen2.5 top-logit prompts:

| Engine | tok/s |
|---|---:|
| bitnet.c Metal | 53.41 |
| llama.cpp Metal `-fa on -np 1` | 58.33 |

Ratio: 0.916. Top-1 matched `8/8` prompts and mean top-10 overlap was `9.62`.
This is good coherence evidence, but it is not yet throughput parity.

These August 2026 medians supersede the older absolute numbers below. The
large change in both engines' absolute throughput reinforces that acceptance
must use the same-run ratio rather than comparing results from different
llama.cpp builds or machine states.

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
  `--metal-native-quant-prepared` exposed the existing prepared Q4_0 Metal upload layout
  to the main benchmark path, but it regressed the same 128-token direct run to
  77.04 tok/s, so the default packed Q4_0 layout remains the better gate path.
  `--metal-disable-small-dense-native-quant` measured 91.53 tok/s, so the Q4_0 x Q8 activation
  path should stay enabled by default. `--gpu-split-residual-rmsnorm` measured
  89.18 tok/s, so the fused residual RMSNorm path should also stay enabled.
  `--gpu-cpu-logits` measured 85.45 tok/s, so the GPU Q6_K logits path remains
  preferable despite being visible in the profile. A Q4_0 x Q8 shader
  experiment that accumulated each `char4` dot with integer arithmetic instead
  of the current `float4` dot path measured 94.42 tok/s, so the float4 dot path
  remains the better kernel variant on this setup. Q4_0 x Q8 threadgroup
  geometry sweeps also did not improve the gate: 32-row groups measured
  95.75 tok/s and 8-row groups measured 96.22 tok/s, so the current 16-row
  geometry remains the best tested setting. The opt-in `--metal-specialized-native-quant`
  logits path now uses a parallel Metal Q8_K activation quantizer and a
  vectorized Q6_K x Q8_K matvec shaped like the default Q6_K float-vector
  shader. The corrected llama-server gate improved to 72.04 tok/s versus
  llama.cpp at 117.20 tok/s with top-1 `8/8` and mean top-10 overlap 9.75, but
  it remains slower than the default Q6_K logits path and should stay
  diagnostic-only. Reducing CPU threads with
  `-t 1` measured 91.62 tok/s, so the default thread setting remains better
  even for GPU decode. The default Q4_0 x Q8 activation policy applies to
  layers 0-2 on this 36-layer model; extending it to all layers with
  `--small-dense-native-quant-to-layer 35` measured 84.57 tok/s, and extending it to layers 0-4
  measured 94.42 tok/s, so the conservative first-three-layer policy remains
  the best tested setting. Restricting the same first-three-layer range to
  attention-only measured 94.73 tok/s, and FFN-only measured 92.19 tok/s, so
  applying Q4_0 x Q8 to both attention and FFN in those early layers remains
  preferable. `--small-dense-native-quant-disable-gateup` uses the native Q4 fused gate/up shader
  while leaving Q4_0 x Q8 enabled elsewhere; it improved the direct
  `bench_kernels` fused row from 338.5 us/call to 303.2 us/call, but the strict
  llama-server gate remained essentially flat at 94.91 tok/s versus llama.cpp
  115.37 tok/s (ratio 0.823, top-1 `8/8`, mean top-10 overlap 9.75), so it is
  diagnostic-only for now. `--small-dense-native-quant-disable-ffn-down` also stayed flat at
  94.90 tok/s versus llama.cpp 115.72 tok/s (ratio 0.820, top-1 `8/8`, mean
  top-10 overlap 9.88). Combining `--small-dense-native-quant-disable-gateup` with
  `--small-dense-native-quant-disable-ffn-down` measured 95.76 tok/s versus llama.cpp 115.40 tok/s
  (ratio 0.830, top-1 `8/8`, mean top-10 overlap 10.00), so native-FFN policy
  toggles are not enough to close the Metal gap. The next Q4_0 FFN kernel
  direction should be row-grouped matvec/gateup work that reuses each activation
  slice across multiple output rows, similar to llama.cpp's
  `mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>` structure; bitnet.c's current
  kernels mostly assign one output row to each 8-lane group. A first
  row-grouped two-row Q4_0 x Q8 matvec diagnostic was not kept: it measured
  94.50 tok/s versus llama.cpp 99.76 tok/s on that run, and the direct
  `bench_kernels` FFN `up`/`down` rows were slower than the default
  small-dense native-quant shader. Lightweight `--gpu-profile 1` timing shows
  the warmed decode frames spend about 8.9-9.6ms in GPU execution, 0.1ms in encoding, and
  effectively 0.0ms in logits readback, so reducing full-logit readback is not
  expected to close the current Metal parity gap on this setup.
- `bench_kernels --metal` now uses GPU-resident weights for per-matrix matvec
  timing, including the quantized output/logits matrix, instead of accidentally
  measuring the CPU quant path. Use `./bench_kernels model.gguf --metal
  --metal-specialized-native-quant` when iterating on the opt-in Q6_K x Q8_K logits
  diagnostic. A 20-iteration Qwen2.5 sample measured the default Metal Q6_K
  logits row at 1786.4 us/call for 151936 x 2048. The older scalar
  `--metal-specialized-native-quant` shader measured 11898.2 us/call; the vectorized
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
- Do not collect sparse samples by launching a new process for every sample.
  On hosts where the mapped model is a substantial fraction of RAM, alternating
  large BitNet and llama.cpp processes measures eviction order more than kernel
  throughput. `compare_llama_topk.py --benchmark --bench-runs N` now runs all
  BitNet repetitions under one model load and passes `-r N` to one
  `llama-bench` process. BitNet resets `BnSession` and sampler history between
  repetitions while retaining immutable model and backend-resident state.
- Use `--bench-warmup-runs N` for full unreported requests before measured
  samples. `--bitnet-bench-warmup-tokens N` instead advances within every
  request before its timed interval; these controls answer different questions
  and should be recorded separately.

When comparing against llama.cpp, record whether llama.cpp is actually CPU-only
or whether it routes some work to Metal/GPU even when a CPU-looking flag is used.

### ARM Native CPU Checkpoint

The following M1 Max samples use the native ARM/NEON runtime on both sides,
eight threads, one loaded model per engine, one full BitNet warmup request, and
two or three measured repetitions. Token counts differ because the 18-22 GB
models require shorter runs on this 32 GB host.

| Model | Shape | tg | BitNet tok/s | llama.cpp tok/s | Ratio |
|---|---|---:|---:|---:|---:|
| Qwen2.5 3B Q4_0 | dense | 32 | 55.19 | 47.68 | 1.158 |
| Qwen3 0.6B Q8_0 | dense | 4 | 182.82 | 107.57 | 1.700 |
| Qwen3 30B-A3B Q4_K_M | sparse | 16 | 29.68 | 9.18 | 3.233 |
| Qwen3.5 9B Q4_K_M | dense | 16 | 16.12 | 11.54 | 1.397 |
| Qwen3.5 35B-A3B Q4_K_M | sparse | 8 | 37.13 | 3.84 | 9.669 |
| Qwen3.6 27B Q4_K_XL | dense | 8 | 4.00 | 3.36 | 1.190 |
| Qwen3.6 35B-A3B Q4_K_M | sparse | 8 | 36.72 | 4.44 | 8.271 |
| Gemma4 E4B Q4_0 | dense | 16 | 32.09 | 25.09 | 1.279 |
| Gemma4 26B Q4_0 | sparse | 8 | 22.68 | 0.49 | 46.276 |

The large sparse ratios include warmed mmap expert locality and should not be
generalized to cold-start serving. The Gemma4 llama.cpp result is especially
slow and needs confirmation on a longer run before it is used as an acceptance
claim. No scalar ratio is reported: a matching scalar llama.cpp executable is
not installed on this machine.

### x86 CPU Checkpoint

On Qwen3 4B Q4_K_M with matching CPU-only llama.cpp builds, a 128-token
prefill and 16-token generation workload gives the following median ratios:

| Runtime | Threads | Flash | BitNet tg | llama.cpp tg | tg ratio | BitNet pp | llama.cpp pp | pp ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| AVX2 | 8 | off | 21.34 | 27.15 | 0.786 | 80.39 | 151.01 | 0.532 |
| AVX512 | 16 | off | 35.46 | 38.57 | 0.919 | 143.03 | 326.00 | 0.439 |
| AVX512 | 16 | on | 41.32 | 43.05 | 0.960 | 169.51 | 422.82 | 0.401 |

AVX512 decode is within the accepted range at the measured 16-thread setting,
but CPU prefill is not. Profiling the non-flash AVX512 path attributes about
790 ms of a 1506 ms run to FFN projections, 365 ms to QKV/output projections,
and 319 ms to attention. The earlier Q4_K AVX512 x16-row/four-token GEMM was
also tested in the active single and multi-matmul dispatches; it reduced
prefill from 143.03 to 138.90 tok/s and was not retained.

Interchanging the Q4_K x8 GEMM's row-group and four-token-panel loops, and a
follow-up eight-group two-dimensional cache tile, were neutral or slightly
slower. The tiled run measured 1006.5 ms versus a 996.1 ms baseline for 128
tokens at 16 threads, with projection subtotals unchanged within run variance.
Both layouts were reverted; the packed kernel is instruction-bound at this
shape rather than limited by repeated whole-matrix traversal.

Thread scaling confirms that this is a microkernel gap. On the same pp128
workload, bitnet.c measures 7381.6 ms at one thread (17.34 tok/s), while the
matching llama.cpp AVX512 build measures 35.19 tok/s. Bitnet.c improves from
985.5 ms at 16 threads to 752.7 ms at 32 threads (170.1 tok/s), but llama.cpp
reaches 416.36 tok/s at 32 threads; 64 bitnet.c threads regress to 893.3 ms.
Inspection of the matching x86 reference showed the missing reuse dimension:
its AVX512 Q4_K GEMM computes 16 prompt tokens by 16 output rows per tile,
whereas bitnet.c's current x8 kernel computes four tokens by eight rows and its
earlier experimental x16 kernel still computed only four tokens. This motivated
the true 16-token x 16-row quant-owned tile below while preserving the
established Q8_K activation and prepared-weight contracts.

That true 16-token by 16-row tile is now retained for AVX512 Q4_K prepared
matmul. It reuses each pair of packed eight-row weight groups across four Q8_K
activation panels, then sends a 1--3 token remainder through the established
x8 prepared matvec reduction. A focused 17-token test covers both paths. On
Qwen3 4B pp128 at 16 threads, the first profiled run fell from 996.1 to 892.5
ms: gate/up dropped from 262.7 to 174.9 ms and QKV from 131.9 to 112.5 ms.
Three subsequent runs measured 909.7, 802.8, and 889.1 ms (144.0 tok/s median,
up from the prior 135.81 tok/s checkpoint). Strict sampled-token parity remains
40/40 for Qwen3 4B and 24/24 for Qwen3.6 27B. This is a useful incremental
prefill gain, though 144.0 versus llama.cpp's 352.53 tok/s remains below the
accepted throughput ratio.

Disassembly later showed that the 16-token tile uses roughly 7.6 KiB of stack
and has extensive accumulator spills. Halving it to eight tokens reduced the
live accumulator set but lost more weight reuse than it saved: flash pp128 fell
from 192.2 to 169.5 tok/s, with QKV, output, and gate/up projections all
slower. The eight-token variant was removed. A useful follow-up must preserve
the sixteen-token weight reuse while shortening live ranges, for example by
separating the comparatively small min-correction pass.
That correction split was subsequently measured: reconstructing the identical
Q4_K min term from native metadata in a compact second pass passed the focused
correctness test, but the extra metadata traversal and scalar reductions
reduced GCC flash pp128 from 208.4 to 158.9 tok/s. It also increased static
stack traffic after inlining, so the split was removed.

Folding each Q4_K minimum correction into its primary FP32 accumulator takes
the complementary approach: it removes all 16 separate minimum accumulators
without adding a metadata pass. On a fresh GCC 13, 32-thread Qwen3 4B control,
`pp128` increased from 140.6 tok/s to 190.0 tok/s; a subsequent seven-run
median measured 184.8 tok/s. The compiled frame fell to about 6.4 KiB. This
regroups FP32 correction terms, so it was retained only after the focused
AVX512 quant test passed, Qwen3 4B remained exact at 128/128 sampled IDs, all
other dense Qwen gates and Gemma4 remained exact, and the known Qwen3 and
Qwen3.8 sparse boundaries stayed unchanged at 22/24 and 21/24 respectively.
An adjacent Clang 18 check measured 213.2 tok/s folded versus 219.3 tok/s with
separate correction accumulators. The small Clang cost does not offset the
default GCC gain, and folded Clang remains faster than folded GCC, so the
format-owned kernel keeps one compiler-independent arithmetic path.
A matched post-change GCC run measured 191.5 tok/s against 424.4 tok/s for the
current CPU-only AVX512 llama.cpp build at 32 threads with flash attention
disabled, a `0.451x` prefill ratio. A warmed bitnet.c profile attributes about
201 ms to Q6_K FFN down, 131 ms to Q4_K attention output, 121 ms to Q4_K
gate/up, 94 ms to QKV, and 73 ms to attention. Q6_K down is now the largest
individual stage; the overall prefill target remains open despite the Q4_K
improvement.

Compiler allocation is a material axis for this kernel. A warning-free Clang
18 AVX512 build reduced its stack frame to about 5.5 KiB and its static stack
references from 767 to 434. At 32 threads, the same flash pp128 workload rose
from 208.4 tok/s with GCC 13 to 254.3 tok/s with Clang (+22.0%); warmed QKV,
output, and gate/up totals were about 78, 109, and 95 ms. The Clang binary also
passed the full Qwen3 4B strict gate, 128/128 generated token IDs across eight
prompts, against the matching llama.cpp AVX512 oracle. This is a supported
build-time gain, but 254.3 tok/s still remains below llama.cpp's accepted
prefill range, so source-level kernel work remains necessary.

Selecting the existing expanded Q6_K prepared representation for both FFN down
and mixed QKV was also rejected. It allocated 861 MB for this model and left
down projection at 255.2 ms versus 253.3 ms with native blocks; total pp128 was
880.2 ms, within the widened-Q4 run variance. Avoiding Q6 nibble expansion is
insufficient by itself; that format needs a vectorized multi-row/multi-token
tile before the extra resident representation is justified.

Changing the native Q6_K prompt tile did not close that gap. Reducing its token
tile from 16 to 8 and 4 lowered weight reuse and moved pp128 from a fresh
812.8 ms baseline to 826.8 and 837.3 ms respectively. Doubling the output-row
tile from four to eight measured 826.6 ms over seven runs, also a regression.
All three variants were reverted; a useful Q6_K improvement requires a new
vectorized kernel rather than another loop-bound adjustment.

The existing llama-compatible Q6_Kx8 prepared layout and generic packed GEMM
were then exercised directly. They agreed with the native kernel within
0.000610352 on the focused synthetic test, but changed the reduction order and
measured 819.7 ms for pp128 with down projection still near 271 ms. This route
was also reverted: repacking alone provides no throughput benefit until the
packed kernel performs real SIMD work across its eight output columns.

An AVX512-VBMI prototype then gathered each byte coefficient across the eight
packed rows and accumulated the resulting sub-dots in vector lanes. It matched
the native kernel within 0.000549316, but the permute-and-broadcast overhead
left pp128 at 814.0 ms and Q6_K down projection near 270 ms. It was removed;
the packed kernel needs a byte transpose feeding VNNI rather than coefficient
gathers feeding 32-bit multiplies.

After the Q4_K accumulator improvement, the refreshed GCC profile attributes
about 201 ms of `pp128` to Q6_K FFN down, making it the largest stage. Folding
the eight AVX2 scale-correction lanes into the lower half of the AVX512 dot
accumulator removed one horizontal reduction without changing integer results,
but measured 190.7 tok/s versus 191.5 tok/s for the adjacent control; profiled
down time moved from about 201 to 206 ms. The widening and masked insert offset
the saved reduction, so the experiment was removed.
A second exact-integer formulation centered prepared Q6_K weights into signed
bytes, biased activations into unsigned bytes, and used `VPDPBUSD` with one
precomputed weighted-center correction per block. This avoids activation
bsums, but the required per-vector activation biasing regressed `pp128` to
185.2 tok/s and moved the isolated Q6_K down matrix from roughly 63 to 109
microseconds. The shared prepared representation and original unsigned-weight
kernel were restored.

For Q4_K tensors whose row count supports the x8 layout, preparation no longer
also allocates the legacy per-block scale and FP32 delta tables. All active x86
matvec and prompt GEMM routes consume the complete x8 representation directly;
non-x8 shapes retain the legacy tables as their fallback. On Qwen3 4B this
reduces reported Q4_K prepared storage from 1964 to 1683 MB without changing
pp128 throughput (157.4 tok/s in the measured run), and strict AVX512 sampled
token parity remains 40/40. A focused test asserts that x8 preparation does not
silently restore the redundant allocations.

The AVX512 FP32 batched-attention value reduction now gathers sixteen KV rows
per instruction while preserving the prior four-accumulator, 64-token lane
assignment. On the same Qwen3 4B pp128 profile, warmed attention time fell from
roughly 160--171 ms to 140--145 ms and end-to-end throughput rose from 156.8 to
161.2 tok/s. The eight-prompt dense gate remains 40/40, and focused coverage
exercises both the masked tail and circular KV-cache indexing. This is a small
incremental gain; CPU prefill remains below the accepted parity range.
An independent four-iteration recheck measured 156.4 tok/s, with warmed
attention varying from 153 to 175 ms, so 161.2 tok/s is a best observed result
rather than a stable throughput floor.

A native-layout AVX512 Q6_K prompt tile was also tested after that recheck. It
hoisted weight unpacking across sixteen tokens and used VNNI for four paired
64-byte chunks, but was neutral end to end (156.6 versus 156.4 tok/s) and moved
the warmed down projection from roughly 223--229 ms to 236--238 ms. It was
removed, leaving the established AVX2-width native Q6_K prompt kernel in the
AVX512 build.

Strict token comparison now also requires an explicitly selected llama.cpp
binary directory to contain `llama-completion`; it no longer falls back to a
PATH binary. With matching builds, Qwen3 4B passes all 40 generated IDs on
AVX2 and AVX512. The AVX512 fix was to remove an AVX512-only policy override
that forced Q4_K through the canonical single-row dot kernel. The native
prepared multi-row Q4_K route selects llama.cpp's token 11477 (`king`) instead
of token 883 (`man`) at the former fifth-token boundary and passes the complete
40-token gate without an environment override.

Several narrower diagnostics ruled out Q6_K as the root cause. Canonical Q6_K
projection routing and the native AVX512 Q6_K VNNI kernel both fix the former
boundary but reduce the full gate to 36/40. Fused-gate bypass, tied-logit
refinement, Q6_K prepared-cache bypass, individual projection roles, and
attention-only or FFN-only routing do not fix it. These diagnostic changes were
not retained.

With the retained native Q4_K policy, a stable 32-token, 16-thread decode run
measured 30.67 tok/s against llama.cpp's 34.86 tok/s, a 0.880 ratio. A separate
128-token prefill run measured 135.81 versus 352.53 tok/s, a 0.385 ratio, so
AVX512 decode now meets the accepted range while CPU prefill remains open.

The same production-sampler oracle passes the available dense AVX512 matrix:
Qwen3 4B Q4_K_M is 40/40, Qwen3.5 27B Q5_K_M is 40/40, Qwen3.6 27B Q4_K_M is
24/24, and Qwen3.8 27B Q4_K_XL is 24/24. Each result covers all eight standard
prompts with the matching CPU-only AVX512 llama.cpp build.

Qwen3 30B-A3B Q4_K_M initially matched 19/24 generated IDs and 7/8 first
tokens on AVX512. Keeping the small MoE router dot products and router softmax
at the established AVX2 reduction width raises this to 22/24 and 8/8 first
tokens; both changes are required to repair the original sky-prompt boundary.
Expert projections and activation remain AVX512-capable. An AVX2-width expert
SiLU experiment merely moved the last mismatch between prompts and was not
retained.

Using AVX2 GQA and attention softmax together raises the sparse result to
23/24, but regresses Qwen3 4B dense from 40/40 to 35/40, including the former
Q4_K fifth-token boundary. Either attention fallback alone still leaves the
sparse story boundary. Both attention changes were therefore rejected; the
remaining sparse result is 22/24 pending a fix that preserves the dense gate.
Forcing tokenwise prompt evaluation on both engines with `--no-prefill` also
scores 22/24: it repairs the story and HTTP boundaries but moves the two
third-token mismatches to the capital and arithmetic prompts. Matching llama.cpp
batch threads and context size does not change the default gate. Prefill
geometry is therefore numerically relevant, but disabling it is not a parity
fix and is not part of the retained sparse fixture.

One remaining Qwen3 sparse AVX512 boundary is specifically sensitive to
llama.cpp's `CPU_REPACK` backend. On the story prompt, bitnet.c ranks token
1030 (`had`) at 21.3114 and token 4829 (`wanted`) at 21.0983. The llama layer
probe with `--no-extra-bufts` also selects token 1030 (21.2709 versus
21.2056), matching bitnet.c. The probe's default extra-buffer selection
reproduces production `llama-completion`: token 4829 leads 21.2268 to 21.1495. Production
reports a 13.4 GiB `CPU_REPACK` buffer and dispatches Q4_K x8 repack kernels.
However, the complete `--no-repack` gate scores only 19/24: it repairs the
story prompt while regressing the arithmetic, HTTP, and sky prompts. Repacking
is therefore a useful differential axis, not a parity workaround or the sole
source of accumulated-state differences. `compare_llama.sh` exposes this axis
as `--llama-no-repack`.

At the story prompt's divergent third-token state, layer-output comparison
against production-repacked llama.cpp stays close through layer 4. Layer 5 is
the first material decode-time amplification: current-token Q/K/V projections
remain close, but attention scores differ before the MoE projection, and the
layer output sum separates by about 0.00138. Binary cache comparison localized
the larger prefix differences to prompt positions 2 and 6. A position-2 trace
then found the first material prefill separation in layer 4's Q4_K attention
output projection: attention, softmax, and KQV agree, while the output sum is
0.0671922 in bitnet.c versus 0.0677268 in llama.cpp and the following MoE
amplifies it. The four-position pattern is not a cache-layout defect: a direct
differential probe against llama.cpp's exported Q8_Kx4 packer matched every
byte. It reflects four-row repacked GEMM sensitivity to the small differences
accumulated by preceding layers. The remaining work is therefore exact
production-repack arithmetic alignment, not model-specific KV handling.
Matching llama.cpp's double-precision `sum_rows` literally for routed-expert
aggregation was tested and rejected: the complete sparse gate fell from 22/24
to 20/24, adding arithmetic and sky regressions. The selected-expert layout and
preceding router arithmetic make that isolated reduction change an incorrect
end-to-end substitute.

A caller-boundary prefill bisection also separated ordinary transformer
projections from routed-expert projections. Replaying only MoE gate/up and down
matrices through prepared single-row matvec fixes the AVX512 HTTP boundary but
does not fix its story boundary; AVX2 keeps its story match and still misses
HTTP. Restoring batched expert down while replaying only gate/up leaves the
AVX512 HTTP fix intact, localizing that sensitivity to low-occupancy Q4_K
gate/up work. This is diagnostic rather than a valid dispatch policy: routing
all prepared Q4_K batches with fewer than four rows through the exact-row
kernel changes the complete AVX512 gate from 22/24 to 21/24. It repairs story
and HTTP but introduces arithmetic and sky regressions. Both code experiments
were reverted. The prepared x8 implementation follows llama.cpp's production
`ggml_gemv_q4_K_8x8_q8_K` accumulation structure, so the evidence continues to
point to accumulated prefill state and graph geometry rather than a model-role
exception in quant dispatch.

Binary comparison subsequently pinned down both ends of that accumulation.
For the story prompt at position 2, layer 0 attention norm, Q, K, concatenated
attention value, and Q4_K output projection are bitwise identical to the
production-repacked AVX512 llama.cpp graph. The normalized MoE input is also
bitwise identical, but the following F32 router logits are the first differing
operation: 69 of 128 values differ, with a maximum error of 9.54e-7. Expert
selection is unchanged and the raw MoE output returns to ULP-scale error
(1.19e-7 maximum). Layer outputs then grow from 1.19e-7 at layer 0 to 2.38e-7
at layer 3.

At layer 4, attention norm and KQV still differ by only 1.19e-7 maximum, but
Q8_K activation quantization maps one of 4096 values differently (index 962 is
0 in bitnet.c and -1 in llama.cpp) and produces different scales in 12 of 16
blocks. The Q4_K output projection consequently reaches a 1.27e-4 maximum
error and sums to 0.0671910 versus 0.0677265. Forcing that projection from the
AVX512 x16 GEMM to the AVX2-width x8 GEMM produces bit-for-bit identical
bitnet.c output and leaves the full gate at 22/24, ruling out the x16 kernel.
The experiment was reverted. Matching llama.cpp's prompt-time single-vector
F32 router accumulation made all 128 layer-0 logits bitwise identical when fed
the same captured input, but worsened the complete AVX512 gate from 22/24 to
18/24 IDs. Pairing it with llama.cpp's native 16-lane AVX512 softmax improved
that result only to 20/24; native-width softmax without the router change also
gave 20/24 and lost the first sky token. Both inference experiments were
reverted. The llama observer used during that diagnosis was corrected as well:
`ffn_moe_topk` and the pre-reshape `ffn_moe_weights_norm` callback are both
laid out as `[K, tokens]`, which is ambiguous to extent-based axis inference
when the prompt length equals K. The corrected route indices agree, and the
selected weights differ only at ULP scale. The remaining alignment target is
therefore the earlier prefill state accumulation that reaches the layer-4
Q8_K rounding discontinuity, not an isolated router or softmax substitution.
With the corrected no-flash observer, a model loaded without llama.cpp's extra
buffer types first differs in the layer-0 Q4_K Q/K projection (V is Q6_K and
remains exact). For Q at position 2, 946/4096 output values match exactly, the
maximum difference is `3.73e-8`, and the mean absolute difference is
`4.09e-9`. Byte-level tests proved that bitnet.c's Q4_Kx8 weight repack and
Q8_Kx4 activation pack match llama.cpp exactly, and llama.cpp's exported
`ggml_gemm_q4_K_8x8_q8_K` reproduces the complete bitnet.c output on those
buffers. The probe now enables extra buffer types by default, as production
`llama-completion` does; this creates the 13.4 GiB `CPU_REPACK` buffer and makes the
graph Q projection match bitnet.c in all 4096 values. At prompt position 7 the
layer-0 Q/K/V projections, attention scores, softmax, KQV, and output
projection all match bit-for-bit as well.

The first production-repack difference is therefore the following batched F32
router operation: selected experts remain identical while logits and normalized
weights differ at ULP scale. Those small differences accumulate through four
layers and cross a Q8_K activation rounding boundary at layer 5. Replacing the
existing AVX2-width router reduction in the AVX512 build with llama.cpp's
standalone four-accumulator AVX512 vector-dot reduction was tested and rejected:
the complete gate fell from 22/24 to 20/24 IDs and lost the first sky token.
Production's batched F32 graph path is not equivalent to that standalone dot
helper, so the experiment was reverted.
`--no-extra-bufts` remains available only as an explicit differential axis for
the non-repacked CPU path.

Qwen3.6 35B-A3B Q8_0 passes the same eight-prompt sparse gate on both AVX2 and
AVX512: 24/24 generated IDs and 8/8 first tokens on each backend. This isolates
the remaining Qwen3 30B-A3B result from generic MoE routing: its experts use
Q4_K gate/up weights and an even split of Q4_K and Q6_K down tensors. Routing
those formats through the validated AVX512 VNNI kernels did not change either
remaining boundary, so that dispatch experiment was not retained.

The sharded Qwen3.5 122B-A10B MXFP4 fixture loads successfully through the
pread sparse path. Its standard AVX512 gate matches 22/24 generated IDs and all
8 first tokens; the only divergence follows the first sky token. That boundary
crosses over by backend: bitnet.c AVX512 emits `because` where llama.cpp AVX512
emits `due`, while bitnet.c AVX2 emits `due` where llama.cpp AVX2 emits
`because`. The expert MXFP4 kernel is scalar on both x86 builds, so this is a
near-tied accumulated-state boundary rather than a format loader failure.

Qwen3.8 Flash-Next UD-Q4_K_XL also loads correctly from all four shards. Its
standard AVX512 sparse gate matches 21/24 generated IDs and 7/8 first tokens;
the sole failing prompt diverges immediately (`was` versus llama.cpp's
`commemorated`). The same first token passes against the matching AVX2 build.
Forcing the complete SSM recurrence, GQA plus attention softmax, compiler
auto-vectorization, dense/MoE activation kernels, explicit F32 reductions, or
every Q8_0 matvec and batched-matmul path to AVX2 width does not change the
AVX512 token. Those diagnostics were reverted, leaving numerical layer
localization as the next step.

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

On the dual RTX PRO 6000 Blackwell host, Gemma4 31B Q4_K_S exposes the current
CUDA prefill boundary clearly. A 128-token no-logits prefill measured 113.6
tok/s with the quantized batch kernels. Enabling eager K-quant auxiliary caches
after keeping explicitly duplicated individual weights quant-only raised this
to 989--1141 tok/s without exhausting the 96 GB device; caching both canonical
and stacked copies exhausted memory during model upload. The F16-cache route is
diagnostic-only because its strict llama.cpp comparison matched 33/38 generated
IDs, versus 59/59 for the quantized route. The remaining production work is a
tiled Q4_K/Q5_K MMQ path: the current CUDA kernel computes one output row by
four prompt tokens, while llama.cpp stages tiles of 128 rows by up to 128 tokens
in shared memory on this device class.
A one-warp 16x16 WMMA diagnostic was not retained: applying scale/min
correction after every 32-value group reduced the same prefill to 94.0 tok/s.
A later 128-row by 16-token integer-WMMA diagnostic was also removed. It was
correct for Q4_K boundary tiles, but reduced prefill from 113.5 to 102.4 tok/s.
The quant registry keeps this asymmetric path Q4_K-specific; Q5_K uses its
separate deinterleaved K-quant capability and kernel family.
A 64-row by 16-token raw-MMA variant then removed the per-group barriers and
kept accumulators lane-local, but was slower again at 98.6 tok/s. Cooperative
runtime unpacking and per-output scale/min application remain too expensive;
the production MMQ path needs a backend-owned packed weight layout prepared at
upload time, not repeated conversion inside every matmul.
An opt-in upload-time packed Q4_K layout validates that direction. It stores
expanded integer values and half-precision scale/min products in each CUDA
buffer and frees them with the buffer; no model or runtime object owns the
backend representation. The 64-row by 16-token raw-MMA kernel then reached
129.3 tok/s (+13.9% over 113.5), with q/k falling from 154 to 137 ms and the
output projection from 128 to 88 ms per 60 layers. Strict generation remained
59/59 sampled token IDs. At this checkpoint the path was retained behind
`BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_MATMUL8`; the later optimized version is now
the default as described below.
When the packed representation is present, dense prefill now composes the
stacked MMQ with the existing activation kernel instead of selecting the older
row-wise fused gate/up kernel. Gate/up falls from 497 to 396 ms and total
prefill reaches 143.9 tok/s (+26.8% over the quantized baseline); the separate
activation costs only 1.2 ms per 60 layers.
The same backend-owned representation now supports Q5_K without casting between
quant block types: Q4_K and Q5_K share a scale-array decoder, then use
format-specific upload packers. On Gemma4, packed Q5_K reduced the 60-layer
down projection from about 202.7 to 192.7 ms and raised total prefill to 146.1
tok/s. A wide 130-row by 512-column by 17-token test covers nonzero Q5_K high
bits, scales, and minima through the packed boundary path. The full strict
comparison still matched 59/59 generated IDs and 8/8 first output-token IDs.
A 32-token version of the packed MMA tile was rejected: it regressed prefill to
124.0 tok/s, with the Q5_K down projection increasing to about 282 ms. Its
additional registers, shared memory, and dependency depth outweighed the extra
weight reuse on this device.
Keeping the 16-token tile but hoisting its panel-invariant A fragments and
row scale/min products is much more effective. It preserves the exact
accumulation order while avoiding the same shared-memory and coefficient loads
for both 8-token panels. Reusing each token's activation scale and Q8 sum across
its two row fragments, and loading scale/min only for the two distinct row
fragments, raises Gemma4 prefill to 229.2 tok/s. The q/k, value, output,
gate/up, and down projection totals are approximately 76, 28, 56, 255, and
120 ms per 60 layers. The strict gate remains 59/59 generated IDs and 8/8
first output-token IDs. Loading each row coefficient in one subgroup leader
and broadcasting it with `__shfl_sync` was slightly slower at 227.2 tok/s, so
ordinary coalesced loads remain in the retained kernel.
The backend-private Q8_1 representation now stores the exact signed sum of its
32 quantized bytes in the previously unused 16-bit auxiliary field. Computing
that sum once in each Q8_1 producer removes a serial 32-byte reduction repeated
by every output tile, raising prefill to 234.5 tok/s without changing the
correction arithmetic. With that bottleneck removed, increasing the row tile
from 64 to 128 and the block from four to eight warps becomes beneficial:
Gemma4 reaches 237.4 tok/s. The compiled `sm_120` kernel uses 62 registers and
37,888 bytes of shared memory with no spills, and strict parity remains 59/59
generated IDs and 8/8 first output-token IDs.
A 64-row by 512-column variant that staged two adjacent K blocks per barrier
was rejected at 146.9 tok/s. Its larger shared tile and 16-group dependency
chain outweighed the reduced barrier count. The 32-token tile instead keeps the
256-column K tile and uses 16 warps: paired warps share each 16-row group and
independently compute one 16-token half, retaining the proven per-warp
accumulator footprint. This raises Gemma4 prefill from 237.4 to 385.0 tok/s;
gate/up falls from about 244 to 123 ms per 60 layers. The compiled `sm_120`
kernel uses 61 registers and 43,008 bytes of shared memory with no spills.
For 64 or more prompt tokens, a 64-row by 64-token specialization assigns four
token warps to each 16-row group. It reaches 450.4 tok/s, with q/k, value,
output, gate/up, down, and attention totals of approximately 38.2, 16.6, 33.5,
102.3, 71.1, and 14.8 ms per 60 layers. A 32-row by 128-token alternative was
slower at 418.1 tok/s and was removed. Dispatch is therefore shape-driven:
128x16 below 32 tokens, 128x32 below 64, and 64x64 thereafter. Wide Q4_K and
Q5_K tests at 17, 33, and 65 tokens exercise every specialization and partial
final tiles. Before shared-stride padding, the `sm_120` 64x64 specialization
used 61 registers and 36,864 bytes of shared memory with no spills. The full
Gemma4 gate remains 59/59
generated IDs and 8/8 first output-token IDs.
After rebuilding the current CUDA source, the end-to-end comparison gate
measures 456.3 tok/s versus llama.cpp's 2403.92 tok/s for `pp128` (0.190x),
and 58.84 tok/s versus 66.23 tok/s for generation (0.888x). Decode is within
the accepted parity range; prefill remains the blocking throughput gap.
The packed path is now the CUDA default, with
`BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_MATMUL8=1` as a diagnostic override. Buffer
creation still skips the auxiliary layout when it cannot preserve the 4 GiB
device-memory reserve, falling back to the canonical quantized kernels. A
benchmark-harness correction also excludes `BnSession` destruction and
recreation between repeated prompt samples from the timed interval. With five
warmed `pp128` samples, the corrected gate measures 457.7 tok/s; the former
timing incorrectly charged roughly 60 ms of request setup per repetition and
reported 373.0 tok/s for the same kernels.
Nsight Systems attributes 73.3% of the warmed `pp128` GPU time to the packed
MMQ kernel, including about 1.76 ms per layer for the 43,008-row stacked
gate/up projection. Both staged byte matrices previously had an unpadded
256-byte row stride, which made lanes reading different rows alias the same
shared-memory banks. Padding each staged row by 16 bytes, following the same
bank-conflict avoidance used by llama.cpp's MMQ layout, raises the five-sample
Gemma4 result from 457.7 to 497.4 tok/s (8.7%). Arithmetic and dispatch remain
unchanged; the CUDA kernel suite passes and the production sampler gate remains
73/73 generated IDs with 8/8 first output-token IDs.
Vectorizing the cooperative staging copies from individual bytes to 32-bit
chunks raises the same five-sample run further to 615.6 tok/s. Widening the
fully aligned packed-weight copy again to 128-bit chunks reaches 670.4 tok/s;
the Q8_1 activation copy stays 32-bit because its payload begins four bytes
into a 36-byte block. A measured 128-bit activation-copy variant was flat at
669.8 tok/s and was removed. A 32-byte weight-copy variant assigning exactly
one segment to each of the 512 threads also regressed to 566.2 tok/s; its two
vector temporaries and less flexible scheduling outweighed the reduced loop
count. These copies populate the same padded shared
layout without changing any quantized values or accumulation order. A focused
dense profile confirms the improvement in the MMQ projections: q/k falls from
38.2 to 27.9 ms, gate/up from 102.0 to 71.6 ms, and down from 71.1 to 52.0 ms
per 60 layers. Total profiled layer time falls from about 281 to 208 ms. The
CUDA kernel suite and strict 73/73 production sampler gate both pass after the
change. Relative to the original 457.7 tok/s packed baseline this is a 46.5%
gain, though 670.4 tok/s remains only 0.279x the retained llama.cpp `pp128`
result.
On `sm_80` and newer, the aligned 16-byte weight transactions now use
`cp.async`; the kernel stages the ordinary Q8_1 activation bytes and metadata
while those transfers are in flight, then waits before the existing block
barrier. Lower CUDA architectures retain the synchronous copy. Gemma4 reaches
751.7 tok/s across five warmed samples, another 12.1% improvement and 64.2%
above the initial 457.7 tok/s packed baseline. The profiled q/k, value,
output, gate/up, and down totals are approximately 22.7, 9.3, 19.3, 60.6,
and 39.1 ms per 60 layers; total layer time is about 171 ms. A 4-byte async
activation copy is not available in the `sm_120` instruction form, so the
activation path remains synchronous pending a naturally aligned packed
staging representation. Boundary-shape CUDA tests pass and the strict Gemma4
gate remains 73/73 generated IDs and 8/8 first output-token IDs.
An MMQ-only 16-byte-aligned Q8_1 scratch layout was tested to make the
activation payload eligible for two 16-byte `cp.async` transfers per block.
It passed the CUDA boundary suite but measured 755.2 tok/s versus the retained
752.9 tok/s ten-sample run. The 0.3% difference is within run variance and does
not justify a second quantizer, 48-byte scratch blocks, or the extra backend
layout, so the experiment was removed.
Keeping the ordinary scratch layout but storing its scale and signed-sum
metadata in shared FP16/int16 form was also rejected. Although this halves the
metadata footprint and passes the CUDA boundary suite, converting FP16 in each
consumer warp reduced Gemma4 pp128 from 751.9 to 620.6 tok/s. The retained
kernel converts once during cooperative staging and shares FP32/int32 values.
Replacing expanded MMQ residency with uncapped eager FP16/cuBLAS caches fits
the 96 GB device for the benchmark and reaches 871.3 tok/s, but a generation
session then runs out of memory while allocating its 1 GB activation buffer.
With an 8 GiB cache reserve it runs but loses strict parity at 72/73 IDs. A
hybrid retaining MMQ while caching only tensors below 100 MiB reaches 791.7
tok/s with a conservative 40 GiB cache reserve; it additionally needs a 4 GiB
optional-layout reserve to leave room for session state and then matches only
63/73 IDs. The first visible divergence follows the first generated token for
`Once upon a time, there was a`. FP16 cache paths are therefore diagnostic
throughput options, not parity-preserving replacements for packed quantized
MMQ. These trials also show that independent per-buffer cache and stacked-
layout reserve checks do not constitute a model-wide optional-memory budget.
Using `cp.async`'s source-byte-count operand for native boundary zero fill was
also rejected at 717.3 tok/s. Clamping the source row and supplying the dynamic
copy size penalized every full tile, so the retained kernel keeps its ordinary
valid-row async instruction and explicit boundary zero fill. After restoring
that path and rebuilding the authoritative `sm_120` targets, a ten-sample
warmed run measured 752.9 tok/s; `test_cuda_backend`, `backend_matrix.sh`, and
`git diff --check` all passed.
Staging the row scale/min coefficients in shared memory to deduplicate their
loads across the four token warps was rejected at 396.5 tok/s. The additional
shared-memory footprint and cooperative load phase outweighed the cached
global reads, so the retained kernel continues to load those backend-packed
coefficients directly.
Covering the same 64x64 output tile with 256 threads by making each warp compute
four 8-token panels was also rejected at 318.2 tok/s. Doubling the per-warp
accumulator set reduced latency hiding more than it reduced redundant row
work; the retained 64-token specialization therefore uses 512 threads and two
panels per warp.
A separate 32-row by 64-token, 256-thread specialization preserved the existing
two-panel warp work while reducing static shared memory from 39,936 to 31,232
bytes. It nevertheless produced only 606.1--608.6 tok/s across three Gemma4
`pp128` runs, versus 742.5 tok/s for the adjacent profiled 64x64 checkpoint.
Nsight Systems attributed 64.9% of that checkpoint's GPU time to the packed
MMQ kernel, whose compiled 64x64 specialization uses 60 registers per thread
with no local-memory spills. The smaller block result rules out shared-memory
residency alone as the dominant gap, so the retained launch remains 64x64 with
512 threads.
Changing the 16-byte asynchronous weight copies from the L2-oriented
`cp.async.cg` policy to `cp.async.ca` was neutral-to-negative in an adjacent
ten-iteration comparison: 754.9 versus 756.9 tok/s for the restored `cg`
version. The two 64-token tiles do reuse weights at `pp128`, but retaining them
in L1 does not improve this access pattern, so the original cache policy stays.
Factoring the Q8_1 activation scale out of the scale and minimum correction
terms removes one FP32 multiply per output correction without changing the
quantized dot product or its K-block accumulation order. The retained form
measured 758.2--763.4 tok/s in four ten-iteration Gemma4 `pp128` runs, versus
756.9 tok/s for the adjacent unfactored control. It increases the compiled
64x64 specialization from 60 to 63 registers per thread but does not spill.
The production sampler remains exact at 73/73 generated IDs across all eight
Gemma4 prompts. A subsequent matched harness run measured 767.8 versus 2367.8
tok/s for prefill (`0.324x`) and 54.0 versus 64.5 tok/s for decode (`0.838x`).
The latter also shows that the earlier `0.905x` short decode result was not a
stable closure of the decode throughput gate.
Per-node Nsight graph tracing on a 32-token production decode attributes 30.4%
of GPU kernel time to Q4_K split projections and 13.1% to ordinary Q4_K
matvecs. Backend shape profiling further identifies the stacked
43,008x5,376 gate/up projection as 5.54 ms/token, while the exact-order
262,144x5,376 tied-logits projection costs 1.69 ms/token. Graph launch overhead
is already negligible at about 53 microseconds/token; the corresponding stream
synchronization waits about 19.6 ms/token, so the remaining decode gap is GPU
execution rather than host dispatch.
An authoritative `sm_120` recheck measured Gemma4 `pp128` at 773.4 tok/s over
three iterations. Three adjacent 32-token production decode runs measured
51.72--52.75 tok/s. Routing the 5,376-wide Q4_K split projections through the
existing four-warp row reduction produced 52.11--52.48 tok/s, which is neutral
within run variance while changing accumulation order. The large-width policy
experiment was therefore reverted; multi-warp row cooperation does not address
the remaining decode bottleneck by itself.
A complementary half-warp experiment assigned two output rows to each warp so
that both 16-lane groups traversed all 21 Q4_K blocks independently. Alternating
short production runs measured 44.94--46.21 tok/s for the two-row kernel versus
46.26--49.41 tok/s for the retained full-warp kernel. Halving the reduction
width increases row concurrency but does not improve this memory-bound shape;
the experimental kernel and its policy surface were removed.
Moving the Q8_1 activation-tile loads into a second `cp.async` group was also
rejected. The naturally aligned four-byte `cp.async.ca` path measured 763.0
tok/s at `pp128` and 46.7 tok/s decode, versus 770.2 tok/s and 47.7 tok/s for
the adjacent synchronous-load control. The existing asynchronous packed-weight
group already overlaps these loads; a second fine-grained group adds scheduling
cost without changing arithmetic, so synchronous activation staging remains.
Replacing exact-order logits with the existing warp-per-row reduction retained
73/73 Gemma4 token IDs but only raised an isolated three-run range from
51.9--52.5 to 53.5--54.0 tok/s, still below the parity floor and not enough to
justify violating the reference-order flag. It was removed. Forcing the
existing fused Q4_K gate/up kernel past the hybrid format's split preference
was also rejected at 51.6--52.6 tok/s; the retained split path measured
51.9--52.5 tok/s in the adjacent control.
The Q8_1 scratch already stores each 32-value signed sum, so an experiment used
that metadata for the Q4_K minimum correction instead of reconstructing it with
DP4A across four lanes. It passed the CUDA boundary suite but left the dominant
gate/up shape essentially unchanged at 92.07 versus 92.30 microseconds and
measured 52.3--52.7 tok/s end to end. The extra arithmetic was hidden by the
weight access path, so the original lane-local reduction and rounding order are
retained.
A 64-row by 128-token specialization was also rejected. It used 1024 threads,
kept the existing two-panel accumulator footprint, and read Q8_1 scale/sum
metadata from the cached activation buffer so its staged byte tiles fit the
48 KiB static shared-memory limit. Despite loading each weight tile only once
for all 128 tokens, Gemma4 `pp128` fell from 457.4 to 362.5 tok/s. The larger
block and repeated metadata reads cost more than the saved weight loads, so the
retained 64x64 kernel remains the best measured token-axis decomposition.
A closer 128-row by 128-token, 256-thread experiment used 72 KiB of opt-in
dynamic shared memory and gave every lane 64 outputs, matching llama.cpp's
high-level tile geometry and occupancy-one intent. It fell further to 223.3
tok/s. Large dimensions alone do not reproduce llama.cpp's transposed and
padded shared layouts or its fragment loading; those mechanisms must be
implemented together before another large-tile attempt is justified.
An intermediate attempt to keep Q4_K nibbles compact in the packed buffer and
unpack them while staging the existing MMA tile was rejected at 99.5 tok/s.
llama.cpp makes compact staging pay through its complete 128-row MMQ shared
layout; adding unpack work to this smaller tile only increases its critical
path. Upload-time expansion remains the measured choice until that larger
kernel structure is implemented.
A later compact backend block retained the predecoded scale/min products while
shrinking only Q4_K values from 256 expanded bytes to 128 packed bytes. It was
correct but still regressed the current 64x64 path from 457.1 to 414.4 tok/s;
storage reduction alone does not hide nibble expansion on the load path.
Likewise, widening the parity-preserving one-row DP4A kernel from four to 16
prompt tokens only improved a controlled 10-iteration run from 105.7 to 109.4
tok/s (3.5%), so token-width reuse alone is not a viable route to parity.
Selective F32 Q4_K caches capped at 300 MiB per tensor fit in 96 GB and reached
142.6 tok/s, but strict generation matched only 33/38 sampled IDs. Raising the
cap to 512 MiB exhausted memory during activation allocation, and a 2048 MiB
cap exhausted memory during weight upload. Dequantized SGEMM is therefore not a
correct or sufficient replacement for a native quantized batch kernel.
The next tensor-core attempt must amortize correction and synchronization over
a larger K tile and share one format-owned eligibility predicate across every
dispatch site.

The CUDA matrix now applies strict llama.cpp generation comparison to sharded
Qwen sparse models as well as single-file models. Previously the sharded path
returned after coherence or smoke testing, which left Qwen3.5 and Qwen3.8
sparse token parity untested even when `RUN_LLAMA_COMPARE=1`. The throughput
gate also defaults to a 128-token prompt, matching the representative `pp128`
baseline rather than the former 16-token micro-workload.

Strict comparison now traces token IDs directly from the production
`llama-completion` sampler with `test/libllama_token_trace.so`. The former
`llama_layer_probe` oracle constructed its own context and selected token 11
where the production CUDA CLI selected token 13, incorrectly reporting a
failure even though bitnet.c and the CLI emitted the same ten IDs. The corrected
Qwen3.8 dense gate passes that prompt 10/10. Gemma4 dense passes all eight
prompts and all 73 generated IDs in a ten-token run (some prompts terminate at
EOG).

The corrected oracle also exposes one real Qwen3.8 dense CUDA boundary case.
For `Once upon a time, there was a`, bitnet.c CUDA, bitnet.c CPU, and llama.cpp
CPU produce IDs `3777,95704,6725`, while llama.cpp CUDA produces
`3777,855,6725`. At the second step llama.cpp CUDA ranks token 855 at 16.3177
and token 95704 at 16.2707; bitnet.c CUDA ranks token 95704 at 16.6045 and token
855 at 15.8265, close to its CPU result. The full corrected Qwen3.8 gate is
therefore 71/80 IDs across eight prompts. The same three-token prompt passes
3/3 sampled IDs against both the AVX2 and AVX512 llama.cpp builds. Disabling
llama.cpp CUDA graphs does
not change the result. Disabling llama.cpp KV offload or reducing full model
offload from 66 to 65 layers restores token 95704; `--no-op-offload` does not.
Temporarily disabling all three llama.cpp fused DeepSeek V4 hyperconnection
operators (`pre`, `comb`, and `post`) also leaves token 855 unchanged, ruling
out that fusion as the source of the CUDA-only boundary. The layer-count result
is numerically sensitive rather than monotonic (63
layers flips back to token 855). On the bitnet.c side, independently disabling
the SSM graph, SSM/FFN fusion, 128-wide delta kernel, prepared SSM inputs,
prefill scan, stacked prefill, and streaming prefill leaves token 95704
unchanged. The difference is too broad for output-row refinement alone and is
localized to the GPU-resident attention/recurrent-state numerical path rather
than one optional bitnet.c fast path.

For tall Q6_K output projections, Metal now selects the quant-owned Q8_K
activation path by backend shape policy (`rows >= 65536`) without enabling the
same arithmetic for ordinary layer projections. On the M1 Max Qwen2.5 3B
fixture, an adjacent warmed `tg64` comparison measured 46.54--48.00 tok/s with
the promoted path versus 42.69--45.51 tok/s with
`BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT=1`. The strict Metal-to-Metal gate
matched all 8 generated IDs. The adjacent llama.cpp Metal control remained
faster at 58.16 +/- 0.57 tok/s, so this is a measured reduction of the open gap,
not Metal throughput completion.

An AVX2 Q4_K x8 prefill experiment folded the four minimum-correction
accumulators into the four primary FP accumulators, mirroring the profitable
AVX512 x16 change. On Qwen3 4B Q4_K_M at 128 prompt tokens and eight threads,
three adjacent seven-iteration runs averaged 86.0 tok/s folded versus 85.4
tok/s with the original separate accumulators. The 0.6% difference is within
run variance and does not justify changing floating-point association, so the
AVX2 implementation remains unchanged.

Compiler allocation is also material for the AVX2 x8 kernel. On the same
Qwen3 4B `pp128`, eight-thread workload, Clang 18 measured 118.9--119.2 tok/s
over three adjacent five-iteration runs, while the default GCC 13 control was
about 85.4 tok/s. The Clang function has a 2104-byte frame and 261 static stack
references versus GCC's 2848-byte frame and 434 references. Exact sampled-token
parity remained 128/128 and the AVX2 quant suite passed. A matched throughput
gate measured Clang bitnet.c at 104.61 tok/s versus llama.cpp AVX2 at 151.47
tok/s (`0.691x`), so this is substantial progress but not acceptance. GCC
register-renaming and loop-pressure flags only reached a median 88.1 tok/s and
were not added globally.

The AVX2 Q4_K x8 kernel now consumes its RHS and activation shuffle patterns
at the dot-product expressions instead of keeping both complete shuffled banks
live. The intrinsic operations and reduction order are unchanged. GCC static
stack references fell from 434 to 355, and three adjacent five-iteration
`pp128` runs improved from roughly 85.4 tok/s to 99.3 tok/s median. Clang was
neutral at 118.5 tok/s median. Both compiler builds passed the AVX2 quant suite
and Qwen3 strict generation remained exact at 128/128 sampled IDs. The
post-change matched GCC gate measured 96.96 tok/s versus llama.cpp's 147.80
tok/s, raising the checkpoint to `0.656x`; AVX2 prefill acceptance remains open.
Collapsing each pattern-one/pattern-two dot pair into a single source expression
was then rejected: GCC stack references rose from 355 to 360 and three runs
fell to a 97.6 tok/s median. The retained layout keeps the shuffle operations
just-in-time but exposes the two partial-sum banks separately to the scheduler.
With Clang 18 and CPU flash attention enabled on both sides, the matched Qwen3
4B gates reach the requested throughput floor: `pp128` is 135.59 versus 155.86
tok/s (`0.870x`), and `tg64` is 19.38 versus 21.40 tok/s (`0.906x`). This is not
yet an AVX2 acceptance configuration because the strict flash-on generation
gate matches all eight first IDs but only 89/128 full-prefix IDs. A shape-policy
experiment that used flash only for batched prefill also failed strict parity
(86/128 against llama.cpp flash and an early divergence against its non-flash
oracle) because prompt attention changes deeper-layer KV state. It was reverted.
The proven exact AVX2 configuration remains flash-off at 128/128 IDs, while the
proven 85% throughput configuration remains flash-on; closing that numerical
gap is still required.

The parity-preserving AVX2 non-flash attention path now exposes independent
head/token work units through the CPU backend vtable, matching the scheduling
granularity already available to the flash implementation while retaining the
same per-unit dot, softmax, and value-reduction arithmetic. With Clang 18, a
direct `pp128` run reached 116.9 tok/s and the matched gate improved from the
prior `0.691x` checkpoint to 111.53 versus 150.84 tok/s (`0.739x`). GCC measured
100.4 tok/s over seven iterations versus the prior 98.9, while AVX512 remained
neutral at 190.7 tok/s. Strict Qwen3 generation stayed exact at 128/128 AVX2
IDs and 40/40 AVX512 IDs.

The AVX2 FP32 value-combine loop now processes eight contiguous head dimensions
per vector while maintaining the original eight token-lane FMA histories and
the same final reduction tree for every scalar result. This removes the prior
dimension-major strided loads without changing attention arithmetic. Clang 18
`attn_cpu` fell from roughly 242 ms to 43 ms and direct `pp128` reached 138.9
tok/s. The adjacent non-flash acceptance gates now pass for Qwen3 4B: prefill
is 137.47 versus 150.20 tok/s (`0.915x`), decode is 18.96 versus 19.58 tok/s
(`0.969x`), and strict generation remains exact at 128/128 IDs. GCC direct
prefill also improved to 120.4 tok/s, though its matched `0.737x` ratio remains
below the floor. This establishes one accepted AVX2 dense configuration; the
remaining requested model matrix still requires per-model throughput gates.

The same Clang AVX2 binary was checked on Qwen3.5 27B Q5_K_M at eight
threads with flash attention disabled. Three-run gates measured `pp128`
at 13.25 versus 13.18 tok/s (`1.006x`) and `tg64` at 2.57 versus 2.59
tok/s (`0.992x`). Bitnet prefill samples were 13.32, 13.25, and 13.22;
decode samples were all 2.57 tok/s. These pass the speed floor, but this
configuration is not accepted: the production-sampler strict gate matched
39/40 IDs across eight five-token prompts, with all eight first IDs matching.
For `HTTP status code 404 means`, both engines emit
`that the requested resource`, then bitnet emits `is` and llama.cpp emits
`was`. The same 4/5-ID result persists at one thread and with
`--no-prefill` at eight threads (llama.cpp batch and ubatch both one).
Neither thread-count reduction nor sequential prompt processing resolves
this numerical boundary.

Reproduce the speed checkpoint with the Clang-built `bitnet_avx2`:

```bash
python3 test/compare_llama_topk.py \
  /data/models/gguf/qwen3_5/27b/q5_k_m/instruct/Qwen3.5-27B-Q5_K_M.gguf \
  --bitnet ./bitnet_avx2 --bitnet-runtime avx2 --llama-runtime avx2 \
  --llama-bench-bin /home/mark/artalis.io/tools/llama.cpp/build-avx2/bin/llama-bench \
  --skip-topk --benchmark-prefill --benchmark --llama-throughput bench \
  --bench-prompt-tokens 128 --bench-tokens 64 --bench-runs 3 \
  --min-prefill-throughput-ratio 0.85 --min-throughput-ratio 0.85 -t 8

LLAMA_BIN_DIR=/home/mark/artalis.io/tools/llama.cpp/build-avx2/bin \
BITNET=./bitnet_avx2 ./test/compare_llama.sh \
  /data/models/gguf/qwen3_5/27b/q5_k_m/instruct/Qwen3.5-27B-Q5_K_M.gguf \
  -n 5 --strict -t 8 --llama-flash-off
```

The strict gate explicitly uses F32 KV on both engines; the throughput
comparator above retains llama-bench's default KV types and bitnet's default
KV mode. Thus the speed checkpoint is not evidence of strict parity under
every benchmark cache setting.

The Qwen3.5 AVX2 HTTP boundary was subsequently localized using identical
prompt/generated token IDs. At layer 0, the Q6_K QKV projection and convolution
checkpoints agree, while the Q5_K gate projection differs because AVX2 defaults
to float-input Q5_K matvecs. Selecting Q8_K inputs with
`BN_AVX2_KQUANT_FLOAT=0` makes all 6144 gate-projection values exactly match
llama.cpp. The normal batched-prefill strict gate then passes both 40/40 and
128/128 sampled IDs across the eight prompts. Three-run speed gates with that
setting also pass: `pp128` is 13.22 versus 13.17 tok/s (`1.003x`), and `tg64`
is 2.62 versus 2.59 tok/s (`1.012x`). The same Clang AVX2 build, eight threads,
and flash-off settings apply. Quant policy now defaults Q5_K matvecs to this
Q8_K-input path; `BN_AVX2_KQUANT_FLOAT=1` retains the float diagnostic. This
change is format-owned and adds no model-family or runtime ownership branches.
After promotion, the cleanly rebuilt Clang AVX2 binary passes 128/128 again
without an environment override. The native clean build and AVX2 quant suite
compile without warnings, and the full `make test` suite passes.

An additional Qwen3.8 27B UD-Q4_K_XL check at eight threads matches 21/24
sampled IDs across eight three-token prompts. Only the year prompt diverges:
bitnet starts `was in the`, while llama.cpp starts `witnessed a significant`.
A separately compiled Clang AVX2 control containing the exact previous Q5_K
policy reproduces the same 0/3-ID year result, excluding the Q5_K default
change as its cause. This model's eight-thread acceptance remains open.

The Qwen3.8 year prompt has identical input IDs on both engines. Its failure
persists with GCC and at one thread; sequential prefill passes 3/3 IDs.
Rebuilding the AVX2 layer observer from current source is essential: the
stale September 4 executable disabled extra buffer types, while production
uses a 2503 MiB CPU_REPACK buffer. With the corrected observer, the first
visible separation at position 10 is layer 14's IQ3_S FFN down projection,
not layer 1's Q4_K projection. Bitnet previously evaluated IQ3_S with float
activations; it now has an AVX2 Q8_K-input dot kernel selected through quant
dispatch and quant capability metadata. Scalar, ARM, and WASM float kernels
are retained. The kernel exactly matches llama.cpp's exported AVX2
`ggml_vec_dot_iq3_s_q8_K` on 17 synthetic rows each at 1, 2, 17, and 68 blocks.
The captured 5120-value layer-14 projection has a maximum difference of
`1.34e-7` after this change, versus roughly `0.003` in the previously reported
first 16 values. The complete gate remains 21/24 IDs; the next visible error
growth is at layer 16's Q5_K SSM output projection, following small differences
in the preceding state. IQ3_S arithmetic alignment alone does not establish
Qwen3.8 acceptance or its speed gate.
The post-change AVX512 gate at eight threads passes 24/24 IDs across all
eight prompts against its matching production llama.cpp build. The clean
native build, Clang AVX2 quant suite, AVX512 build, and full unit suite pass
without compiler warnings. The local `test/llama_layer_probe_avx2` binary
has also been rebuilt from current source so subsequent traces use the
production repack setting.

The remaining IQ3_S projection difference was subsequently traced to
llama.cpp's IQ panel path, which is independent of `--no-repack` and selected
for batches of at least eight tokens with eight-row-aligned weights. The
existing quant-owned IQ panel decoder now also supports IQ3_S. Its captured
layer-14 output matches all 5120 llama.cpp graph values exactly; the standalone
dot still differs from that graph in 4471 values, despite matching the exported
standalone reference kernel. The panel allocation-failure fallback now selects
the correct IQ3_S, IQ3_XXS, or IQ4_XS dot kernel. A synthetic 16-row, three-block,
nine-token regression checks the panel's integer-superblock/FMA accumulation
against an independent dequantization-based reference.

The Clang AVX2 eight-thread gate still returns 21/24 sampled IDs. Last-prompt
position checkpoints now agree through layer 34, with the next visible
separation at layer 35 attention. Tracing earlier prompt position 4 instead
finds the first visible layer-output difference at layer 34: the Q4_K FFN-up
projection differs slightly before Q5_K FFN-down amplification. Thus the panel
fix advances numerical alignment but does not close Qwen3.8 AVX2 acceptance.
The clean native build and full test suite pass without warnings; the Clang
AVX2 quant suite also passes, including the new panel regression.
However, the rebuilt AVX512 gate regresses from 24/24 to 21/24 on the same
year prompt: bitnet now starts `witnessed a significant`, while its matching
llama.cpp AVX512 build starts `was in the`. Neither runtime is accepted by
this checkpoint. For the next AVX2 discrepancy, the complete layer-34 FFN
input at position 4 and its Q8_K x4 packing match exactly. Isolated Q4_K x8
GEMM outputs also match llama.cpp exactly, while the full-run projection's
first value matches the non-panel x8 matmul path. Routing remains under
investigation; no throughput acceptance is claimed for this change.

The routing investigation found an arena-sizing omission in
`prepared_stats_add_bytes`: `BN_PREPARED_WEIGHT_Q5_K_X8` was not included in
the size total, although preparation allocated those layouts. The temporary
full-run trace shows a null prepared weight for the layer-34 Q4_K FFN-up
projection after earlier allocations consume the undersized arena. Q5_K x8
bytes now contribute to the existing low-bit repack total. A mixed Q5_K/Q4_K
backend-layout regression checks the exact reported sum and verifies both
layouts are registered when using an arena of precisely that size. The clean
native build and full test suite pass without warnings after this correction.
With both fixes, the rebuilt Clang AVX2 eight-thread, flash-off standard gate
passes **24/24 sampled IDs across eight prompts**, including the year prompt.
The rebuilt AVX512 gate remains 21/24: its year prompt is still the only
failure, with bitnet choosing `witnessed a significant` and llama.cpp choosing
`was in the`. The former AVX512 24/24 checkpoint therefore does not apply to
the corrected revision. These changes retain quant/backend-layout ownership
and introduce no model-family special cases.

The corrected Clang AVX2 build's three-run, eight-thread speed checkpoint is
`tg64` **3.05 versus 2.92 tok/s (1.045x)** and `pp128` **12.92 versus 16.53
tok/s (0.781x)**. Bitnet decode samples are 3.05, 3.06, and 3.01 tok/s;
prefill samples are 12.92, 12.78, and 12.92 tok/s. Decode clears the floor,
but prefill fails 0.85, so Qwen3.8 AVX2 overall acceptance remains open despite
its standard token gate passing. The speed harness retains default KV settings,
unlike the strict token gate's matched F32 KV configuration.

```bash
python3 -u test/compare_llama_topk.py \
  /data/models/gguf/qwen3_8/27b/q4_k_xl/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --bitnet ./bitnet_avx2 --bitnet-runtime avx2 --llama-runtime avx2 \
  --llama-bench-bin /home/mark/artalis.io/tools/llama.cpp/build-avx2/bin/llama-bench \
  --skip-topk --benchmark-prefill --benchmark --llama-throughput bench \
  --bench-prompt-tokens 128 --bench-tokens 64 --bench-runs 3 \
  --min-prefill-throughput-ratio 0.85 --min-throughput-ratio 0.85 -t 8
```

The sequential identical-ID replay still has a near-tied final-token
divergence, so the accepted checkpoint requires normal batched prefill.
For reproducing this numerical trace, use IDs
`8957,2552,1970,220,19,15,19,3254,421,279,10897,4939` and observe position 11.
Re-encoding the displayed continuation is not equivalent: the current bitnet
tokenizer splits ` resource` into IDs `193245,322,341` rather than `4939`.
All eight original gate prompts tokenize identically to llama.cpp; this
separate tokenizer limitation does not explain the original HTTP gate failure.

### IQ4_NL batched AVX2 panel optimization

Profiling Qwen3.8 27B UD-Q4_K_XL `pp128` at eight threads found six IQ4_NL
projections consuming about 1.506 seconds. The batch path still decoded and
multiplied individual values in scalar loops for every token. It now decodes
an eight-row panel once per batch and uses AVX2 integer dots, retaining the
same per-block FMA accumulation order. The previous scalar implementation
remains the allocation-failure fallback. This stays entirely in quant code;
no model or backend ownership changes are needed.

An exact-reference test covers 16 rows, five blocks, and nine tokens with
signed codebook values, both nibbles, distinct scales, and multiple row groups.
The clean native build, full unit suite, Clang AVX2 quant suite, and AVX512
build pass without warnings. A first full-model profile reduces prompt time
from 10.121 to 8.545 seconds; formal throughput and token gates are recorded
separately below.

The longer Clang AVX2 gate (`-n 16 --strict -t 8 --llama-flash-off`) matches
123/128 sampled IDs. Only the sky prompt differs, after 11 shared IDs:
bitnet continues `The shorter wavelengths` while llama.cpp continues
`This phenomenon is`. A saved pre-IQ4_NL-optimization binary reproduces the
same 11/16-ID sky result, so this is a newly exposed longer-prefix boundary,
not a regression from vectorizing the panel. The shorter 24-ID checkpoint
is not sufficient to claim complete token parity. The AVX512 standard gate
remains 21/24 on its existing year-prompt mismatch.

The final three-run Clang AVX2, eight-thread speed gate passes both floors:
`pp128` is **15.02 versus 16.53 tok/s (0.909x)**, up from the previous
0.781x checkpoint, and `tg64` is **3.00 versus 2.99 tok/s (1.003x)**.
Bitnet prefill samples are 15.09, 15.02, and 14.93 tok/s; decode samples are
3.00, 3.01, and 3.00 tok/s. Reproduce with the Qwen3.8 throughput command
above. As before, throughput uses default KV settings while the token gate
explicitly matches F32 KV. The speed requirement is met for this configuration,
but the longer 123/128-ID gate leaves full Qwen3.8 AVX2 acceptance open.

### IQ4_NL decode routing and SSM projection reduction

The longer Qwen3.8 AVX2 sky-prompt failure exposed two independent decode
arithmetic differences. IQ4_NL single-vector and homogeneous multi-weight
calls consume float activations, whereas llama.cpp uses Q8_0 inputs.
A candidate quant-owned Q8-input kernel was added. Eight-row-compatible tensors
use integer block reductions followed by scalar FMA; other shapes retain
the standalone two-accumulator vector-dot order, including the separately
rounded odd-block tail. Tests exercise this kernel directly for both shapes
and odd block counts, independently of production routing.
Isolated comparisons against llama.cpp match all 17 rows at block counts
1, 2, 5, 160, and 544. Replaying the captured layer-1 IQ4_NL down projection
matches all 5,120 reference outputs exactly.

An AVX2-only SSM-output flag also forces a reference-dot reduction instead
of the normal format-selected kernel. An experimental removal made the
normal Q6_K kernel match all 5,120 outputs of the captured layer-1 projection;
the forced reduction did not. However, a better local trace did not establish
better end-to-end parity, so removal must be gated independently.

With both corrections, the recorded first 16 layer-output values match
llama.cpp through layers 0–2 at the first decode position of the sky prompt.
Layer 3 first differs by about 1.49e-8, despite matching head-0 attention
scores. This trace is diagnostic evidence, not a full-tensor parity gate.
The sky prompt still shares only 11 of 16 generated IDs. The combined
experimental revision scored 119/128, adding a Python-prompt divergence
after 12 IDs to the existing sky failure (previously 123/128). Consequently
the SSM override removal was backed out for isolation. The Python prompt
still scored 12/16 with IQ4_NL routing alone, isolating the regression to
that change. Both production routing changes were therefore backed out;
the candidate kernel and direct reference tests remain for further work.
No runtime switch or model-specific exception was added. The combined
experimental revision had passed the clean native build, full unit suite,
Clang AVX2 quant suite, and AVX512 build without warnings, but those checks
were insufficient to authorize the routing change. Its AVX512 standard
gate remained 21/24 with the same year-prompt mismatch.

After restoring production routing, the Python prompt returns to **16/16**
sampled-ID parity. The final clean native build, full unit suite, Clang AVX2
quant suite, and AVX512 build pass without warnings. The full 128-ID gate
was not rerun after restoration; its prior 123/128 result remains a
historical checkpoint, not a new measurement.

The retained revision's three-run Clang AVX2, eight-thread speed gate passes:
`pp128` is **14.93 versus 16.59 tok/s (0.900x)** and `tg64` is
**3.01 versus 2.93 tok/s (1.027x)**. Prefill samples are 14.89, 14.93,
and 15.07 tok/s; all three decode samples are 3.01 tok/s. This uses the
same default-KV throughput configuration described above, not the matched
F32 KV configuration of the token gate. Full token acceptance remains open.

### Longer sparse CPU acceptance checkpoint

The current Clang AVX2 Qwen3 30B-A3B Q4_K_M gate at eight threads,
`--pread --cache-mb 2048`, matched **77/128 sampled IDs** across the eight
standard prompts with 16 generated tokens each. All eight first IDs matched,
but capital, story, arithmetic, and sky prefixes were only 5/16, 4/16, 3/16,
and 1/16 respectively. Year, fox, HTTP, and Python matched 16/16 each.
The sky prompt also scored 1/16 with mmap loading, so switching off pread
does not repair that boundary. This is a token result, not a throughput
measurement; it does not establish sparse acceptance.

The adjacent three-run, eight-thread AVX2 speed baseline (default mmap and
cache settings) also misses acceptance: `tg64` is **19.73 versus 25.64 tok/s
(0.770x)** and `pp128` is **77.47 versus 149.69 tok/s (0.518x)**.
Decode samples are 19.72, 19.96, and 19.73; prefill samples are 77.47, 78.75,
and 73.70 tok/s. Throughput uses default KV settings, unlike the explicitly
matched F32 KV token gate.

A diagnostic run with `BN_CPU_PREPARED_CACHE_MB=32768` and
`BN_CPU_PREPARE_ALL_EXPERTS=1` raises decode to **24.07 versus 25.74 tok/s
(0.935x)**; samples are 24.08, 24.05, and 24.07. Prefill remains below the
floor at **79.46 versus 149.69 tok/s (0.531x)**, with samples 72.50, 79.46,
and 84.03. These environment settings are not new defaults or token-parity
fixes. The result identifies prepared-layout caching as a substantial decode
cost. Source inspection shows decode acquires layouts from the backend-owned
cache, while MoE prefill still prepares and frees local projection layouts
for each active expert. Reusing eligible cached layouts during prefill is a
specific next optimization to measure without changing quant arithmetic or
moving buffers onto model weights.
A separate startup check confirms the prepare-all path completed for 15,360
expert projections (5,339 ms) without reporting a preparation failure.

Both CPU family scripts now request 16 tokens per prompt for every `full`
case. Previously several full sparse/newer-model checks stopped at five
tokens. `CPU_PARITY_THREADS` (default 1) sets the shared thread count, with
family-specific `QWEN_CPU_PARITY_THREADS` and `GEMMA4_CPU_PARITY_THREADS`
overrides. The top-level CPU gate collects Gemma4 results even if Qwen
fails and returns nonzero if either family fails. Synthetic harness tests
in `make test` verify full and standard budgets, thread propagation and
validation, and failure collection without loading any GGUF.
The clean native build, full unit suite, and rebuilt Clang AVX2 executable
pass without compiler warnings; the shell syntax and synthetic harness
checks pass as well.

### Reusing prepared expert layouts during CPU prefill

MoE prefill now borrows existing eligible prepared projections through MoE cache
helpers, which retain ownership in `BnBackendModel`. Acquisitions are pinned
until gate/up/down computation completes. Cache-disabled, capacity-limited,
and cache-miss paths retain local preparation. Only stable mmap
addresses (including sharded mmap) can enter this address-keyed cache; pread
staging buffers are excluded. The prefill implementation keeps backend types
opaque, and quant selection and arithmetic are unchanged.
Lookups do not insert or evict entries on a miss. The initial inserting
variant reduced default-cache prefill from 77.47 to 72.88 tok/s, while a
32 GiB prewarmed cache improved from 79.46 to 90.48 tok/s. This motivated
lookup-only reuse: keep the warm benefit without populating the bounded
cache with every prefill expert. These are intermediate measurements;
the final lookup-only variant is measured separately below.

A synthetic nine-token, two-expert Q4_K batch matches local preparation
bit-for-bit with a cold full cache, warm full cache, and one-entry cache
pressure. Additional checks cover pin release, pread exclusion, and sharded
mmap eligibility. The clean native build, full unit suite, updated MoE tests,
architecture checks, and AVX2/AVX512 builds pass without warnings.
The saved before, initial-inserting, and final lookup-only Clang AVX2 binaries
produce identical complete gate
reports for Qwen3 sparse at eight threads with mmap: **77/128** IDs, with the
same four failing prompts. This preserves the known token baseline; it does
not resolve token parity.

The final lookup-only default-cache run measures `tg64` **19.76 versus
28.10 tok/s (0.703x)** and `pp128` **75.70 versus 143.87 tok/s (0.526x)**.
Bitnet decode samples are 19.74, 20.04, and 19.76; prefill samples are
72.26, 76.78, and 75.70 tok/s. Decode returns to the earlier 19.73 tok/s
baseline, while prefill remains slightly below the earlier 77.47 tok/s;
this is not evidence of a default-cache speedup. Reference throughput also
varied across adjacent runs, so retain the per-run ratios rather than mixing
control measurements from different runs.

With `BN_CPU_PREPARED_CACHE_MB=32768 BN_CPU_PREPARE_ALL_EXPERTS=1`, the
final lookup-only run reaches `pp128` **94.63 versus 148.89 tok/s (0.636x)**
and `tg64` **24.10 versus 26.61 tok/s (0.906x)**. Bitnet prefill samples
are 91.54, 95.08, and 94.63; decode samples are 24.10, 24.11, and 24.10.
This is about 19% more prefill throughput than the preceding warm-cache
baseline of 79.46 tok/s, but remains below the required prefill floor.
Default cache budgets are unchanged, and the larger-cache configuration is
explicit rather than a model-family-specific default.
The final 32 GiB prepare-all sampled-ID gate also reports **77/128**, with
an identical complete report to the cold mmap gate. Thus the real-model
cache-hit path preserves this measured token baseline as well as the
synthetic bit-for-bit batch comparisons. Both token and prefill acceptance
remain open.

### Shared Q4_K activation preparation for multi-matrix prefill

Profiling warmed Qwen3 sparse `pp128` found MoE gate/up projections taking
890.1 ms versus 168.3 ms for down projections. The generic prepared
multi-matrix entry point still invoked separate Q4_K matmuls, repeating
activation quantization, four-token panel packing, and thread dispatch.
Homogeneous AVX2-capable Q4_K matrices now share those steps through the
existing prepared-input multi-dispatch function. Each matrix retains its
previous kernel and reduction order; other formats and platforms retain
their existing routes. No model-family or projection-role branch was added.

An exact test compares shared and separate calls across token counts
1, 2, 3, 4, 5, and 9; 16-, 24-, and 15-row matrices; mixed prepared/unprepared
layouts; and inline/thread-pool execution. The clean native build, full unit
suite, Clang AVX2 quant suite, and AVX2/AVX512 builds pass without warnings.
A new profile reduces gate/up to 613.0 ms and total prompt time from
1422.2 to 1177.8 ms; down remains approximately unchanged at 172.7 ms.

With the explicit 32 GiB prepare-all cache, the three-run AVX2 gate measures
`pp128` **123.84 versus 149.34 tok/s (0.829x)** and `tg64` **24.26 versus
25.52 tok/s (0.951x)**. Prefill samples are 127.36, 123.84, and 112.27;
decode samples are 24.06, 25.57, and 24.26 tok/s. Prefill improves from the
preceding 94.63 tok/s checkpoint but still misses the 0.85 floor. These
throughput results use default KV settings, as in the preceding checkpoints.

With default mmap/cache settings, `pp128` improves from 75.70 to **96.31
versus 144.81 tok/s (0.665x)**. `tg64` is **19.90 versus 25.80 tok/s
(0.771x)**. Bitnet prefill samples are 104.07, 96.31, and 92.73; decode
samples are 19.90, 19.86, and 19.91. Both default-cache speed floors remain
open, despite the prefill improvement.

The final warmed AVX2 16-token-per-prompt gate remains **77/128**; its complete
report is identical to the preceding lookup-only implementation. Expanded
AVX512 coverage scores **100/128** with all eight first IDs matching: story
and HTTP stop matching after two IDs, while the other six prompts match all
16. These use the corresponding production llama.cpp SIMD binaries and
matched F32 KV with flash off. Neither backend has full sparse token parity;
the AVX512 result is expanded coverage, not a claim of improvement over a
previous 128-ID baseline.

### Avoiding duplicate Q8_K preparation for packed panels

The prepared Q4_K single- and multi-matrix paths now pack four-token panels
first. When all selected kernels consume those panels, only the remaining
one to three tokens receive canonical Q8_K preparation. Mixed/unprepared
matrices, disabled native batching, and failed panel allocation retain full
canonical preparation. The packed buffer is borrowed during multi-dispatch
and freed by its caller; the public prepared-input API is unchanged.
Eligibility checks include the actual prepared-layout kind, preventing a
kernel-internal fallback from reading omitted canonical rows.

Exact comparisons now include 8-, 16-, and 17-token batches, fully packed
pairs, mixed three-matrix groups, and controls with every canonical row
initialized. A focused MemorySanitizer run passes both normal execution and
fault injection that confirms at least three panel-allocation failures;
fallback single/multi outputs match bit-for-bit, with no uninitialized-read
report. The final clean native build, full suite, AVX2 quant tests, and both
SIMD builds pass without warnings.

Qwen3 dense Q4_K_M passes **128/128 sampled IDs on both AVX2 and AVX512**.
The warmed Qwen3 sparse reports remain unchanged at **77/128 AVX2** and
**100/128 AVX512**, including identical full gate reports to the previous
implementation. All token checks use matching production SIMD binaries,
eight threads, F32 KV, and flash off.

The explicit 32 GiB prepare-all AVX2 sparse speed gate now passes both
configured 0.85 floors: `pp128` is **126.25 versus 148.02 tok/s (0.853x)**
and `tg64` is **24.81 versus 25.47 tok/s (0.974x)**. Prefill samples are
130.32, 126.25, and 119.99; decode samples are 24.33, 24.81, and 26.96.
This is a measured median result for that cache configuration, using default
KV settings; it does not establish default-cache speed or sparse token
acceptance, and prefill is close to the threshold.

Default mmap/cache settings remain below the floors: `pp128` is **102.47
versus 148.21 tok/s (0.691x)** and `tg64` is **20.07 versus 25.68 tok/s
(0.782x)**. Bitnet prefill samples are 101.62, 102.47, and 106.37; decode
samples are 20.48, 19.93, and 20.07. This improves default prefill from the
previous 96.31 tok/s checkpoint without changing its cache budget.

### Router-only checkpoint (superseded; not accepted)

The initial x86 prefill experiment routed F32 logits through the existing
quant batched F32 implementation, then reuses the normal softmax/top-K code.
Single-token routing and non-x86 routing are unchanged. No model names or
backend storage were added to the routing implementation.

For Qwen3 30B-A3B Q4_K_M, AVX2, prompt `The color of the sky is`, position 5,
the production-repacked llama probe and bitnet router inputs were byte-identical.
The old router differed in 88/128 logits (maximum error `2.86102e-6`). The
batch path matches all 128 logits, all eight selected weights, and all 2048
layer-0 raw MoE output values exactly. Binary captures are
`/tmp/moe-router-enabled-{logits,raw}0.bin` and their
`/tmp/moe-router-ll-{logits,raw}0.bin` reference counterparts. The reference
input capture must select exact tensor name `ffn_norm-0`: the reshaped alias
has ambiguous token-axis inference and is not an interchangeable observer.

This local alignment was **not token acceptance**. The eight-prompt, 16-token,
thread-8 production gates regress from 77/128 to **73/128 on AVX2** and from
100/128 to **66/128 on AVX512** (first IDs 8/8 and 7/8 respectively).
Reports: `/tmp/moe-batched-router-enabled128.log` and
`/tmp/moe-batched-router-avx512-128.log`. Earlier throughput measurements
are not revalidated for this experimental state.

The next concrete discrepancy was inside layer-1 expert execution. In the
same AVX2 sky trace, routing weights match, but experts 22 and 92 differ
already at their SiLU outputs; the other six experts' first 16 activation
values match. Both differing experts receive exactly four prompt tokens.
Bitnet uses its packed four-token GEMM at this boundary, whereas llama's
repacked `MUL_MAT_ID` implementation (`ggml-cpu/repack.cpp`) calls GEMV for
every routed token, even when an expert receives four or more. The next
experiment must test this accumulation-order difference while retaining
batch input sharing and the dense GEMM behavior, then rerun the complete
token and throughput gates. The router change alone is not a finished fix.

Synthetic router coverage now checks batches of 1–5 tokens, non-vector-aligned
dimensions, inline and three-worker dispatch, preservation of raw logits,
weight normalization/scaling, and untouched output on tokenwise fallback.
The clean native build is warning-free and the full `make test`, including
the backend architecture matrix, passes. The new batch capability is owned
by the local MoE CPU ops table; exported orchestration has no ISA branches.

### Qwen3 sparse GEMV-order alignment

The quant layer now exposes `bn_quant_matmul_prepared_multi_gemv`: inputs
are batched but each token keeps the zero-flag `matvec_batch` arithmetic.
Packed Q4_K projections share canonical Q8_K input preparation and one
threadpool dispatch, using the existing batched x8 GEMV-order kernel rather
than the four-token GEMM. Unsupported formats/layouts and allocation
failures use tokenwise matvec dispatch. Dense matmul APIs are unchanged.
MoE selects this operation through its CPU ops policy, without inspecting
quant formats or attaching backend state to weights. Non-x86 MoE selection
is unchanged.

Together with the batched F32 router, this makes all 2048 layer-1 raw MoE
output values byte-identical to production-repacked llama.cpp in the AVX2
sky trace. The first 16 layer outputs agree through layers 0–46. AVX512
also needs native 16-lane router softmax and the four-accumulator, 64-element
single-token router dot. The latter matches all 128 layer-0 decode router
logits in the story trace at position 8.

Qwen3 30B-A3B Q4_K_M now passes **128/128 sampled IDs and 8/8 first IDs on
both AVX2 and AVX512**, at eight threads, matching F32 KV and flash off.
Reports are `/tmp/moe-gemv-order-avx2-128.log` and
`/tmp/moe-gemv-native-router-128.log`. The AVX512 intermediate totals were
85/128 with GEMV-order experts alone and 117/128 after native-width softmax;
the native-width decode dot closes the final capital-prompt mismatch.

With an explicit 32 GiB prepare-all cache, AVX2 `pp128` is **131.81 versus
140.64 tok/s (0.937x)** and `tg64` is **25.45 versus 26.32 tok/s (0.967x)**.
Bitnet prefill samples are 131.35, 135.98, and 131.81; decode samples are
26.12, 25.45, and 24.34. Both 0.85 speed gates pass. These throughput tests
use default benchmark KV settings (bitnet F32, llama F16), unlike the matched
F32 token gates, and do not establish default-cache performance.
Report: `/tmp/moe-gemv-order-warm-speed.log`.

The corresponding warmed AVX512 speed row remains below its prefill floor:
`pp128` is **113.01 versus 175.73 tok/s (0.643x)**, while `tg64` is
**24.17 versus 25.91 tok/s (0.933x)**. Prefill samples are 110.38, 114.52,
and 113.01; decode samples are 24.17, 24.18, and 24.17.
Report: `/tmp/moe-gemv-order-avx512-warm-speed.log`.

AVX2 default mmap/cache settings also remain below the speed floors:
`pp128` is **104.67 versus 148.42 tok/s (0.705x)** and `tg64` is
**21.01 versus 25.12 tok/s (0.836x)**. Prefill samples are 104.67, 96.17,
and 105.56; decode samples are 21.01, 21.98, and 20.76.
Report: `/tmp/moe-gemv-order-default-speed.log`. These are speed failures,
not exceptions to the original goal; the warmed AVX2 result must not be
presented as default-cache or AVX512 speed acceptance.

A sequential warmed 128-token stage profile points to CPU attention as the
next AVX512 target: **282.874 ms AVX512 versus 52.177 ms AVX2**. MoE gate/up
is comparable (291.0 versus 293.4 ms), and AVX512 down is faster (222.9
versus 267.7 ms). The AVX512 FP32 weighted-value implementation currently
gathers token-strided values separately for every output dimension, whereas
AVX2 processes contiguous output lanes. Any replacement must preserve the
AVX512 token reduction order and the newly passing sampled-ID gate.
Profiles: `/tmp/moe-gemv-profile-avx512.log`,
`/tmp/moe-gemv-profile-avx2.log`. These single profiles identify a candidate
bottleneck; they are not additional throughput acceptance measurements.

Exact synthetic checks cover 1–17 tokens, one to three Q4_K projections,
packed/mixed/unprepared layouts, inline/three-worker dispatch, and F32
fallback with up to five projections. Focused MemorySanitizer checks pass,
including independent failure injection for all three shared-input
allocations, with byte-exact fallback outputs. Source and log:
`/tmp/moe_gemv_msan.c`, `/tmp/moe-gemv-msan.log`.

Final verification passes: clean warning-free native build, full `make test`
(including architecture and multiline harness checks), Clang AVX2 quant
tests/build, and AVX512 build, quant tests, and compile check. Logs are
`/tmp/moe-gemv-final-{clean,tests,avx2,avx512}.log`.

The longer Qwen3.6 sparse Q8_0 AVX2 gate remains **111/128**, first IDs 8/8;
the earlier 24-token smoke result does not cover these later divergences.
Its capital continuation genuinely differs after the first three IDs.
The word summary previously read only the first output line and misleadingly
showed all prompt texts matching; it now flattens newlines, with a synthetic
multiline regression test. Strict sampled-ID acceptance is unchanged.
Report: `/tmp/moe-gemv-order-q36-avx2-128.log`.

### AVX512 attention and packed-row scheduling

FP32 batched attention now loads eight adjacent output dimensions instead
of gathering token-strided values independently for every dimension. Its
64 token accumulators and final reduction tree are unchanged. At eight
threads, the same warmed 128-token profile reduces CPU attention from
**282.874 to 51.560 ms**. The independent gather-reference test covers 132
combinations of head width (8–256), KV length (1–257), and wrapped/non-wrapped
storage, with two KV heads and an unaligned layer offset. Focused Clang
AddressSanitizer/UBSan/LeakSanitizer checks pass; leak checking required an
approved run outside the ptrace-based sandbox.

Packed Q4_K GEMV and GEMM dispatch now reports physical output rows to the
thread pool. Private quant adapters map those intervals to eight-/sixteen-row
kernel groups, rounding both boundaries upward so even split groups execute
exactly once. Previously, a 768-row GEMV exposed only 96 work units; the
pool's 32-unit minimum yielded three chunks per projection. No thread-pool
policy, public API, arithmetic kernel, model rule, or backend storage changed.
Tests poison outputs and compare threaded GEMV/GEMM against serial results,
including 33-row chunks that split both packed group widths and a token tail.

The warmed Qwen3 30B-A3B Q4_K_M progression is:

| Retained changes | Threads | bitnet pp128 | llama pp128 | Ratio |
|---|---:|---:|---:|---:|
| GEMV-order arithmetic, before this optimization | 8 | 113.01 | 175.73 | 0.643 |
| Contiguous attention loads | 8 | 139.81 | 176.49 | 0.792 |
| Contiguous attention loads | 16 | 138.01 | 227.68 | 0.606 |
| Also physical-row GEMV dispatch | 16 | 167.23 | 223.67 | 0.748 |
| Also physical-row GEMM dispatch | 16 | 185.86 | 229.99 | 0.808 |
| All changes, acceptance configuration | 8 | **158.22** | **176.54** | **0.896** |

The final eight-thread decode result is **26.12 versus 25.61 tok/s (1.020x)**,
so both 0.85 speed floors pass. Prefill samples are 158.22, 162.37, and
155.45; decode samples are 27.31, 24.97, and 26.12. This is the explicit
`BN_CPU_PREPARED_CACHE_MB=32768 BN_CPU_PREPARE_ALL_EXPERTS=1` configuration,
not a default-cache claim. Benchmark KV defaults remain bitnet F32 and
llama F16; token gates use matching F32 KV with flash off.
Report: `/tmp/avx512-attn-allrows-t8-speed.log`. Sixteen threads offer higher
absolute throughput but do not meet the relative prefill floor, so the
accepted speed configuration is eight threads.

The retained dispatch also preserves the warmed AVX2 speed pass: **138.42
versus 148.92 tok/s prefill (0.930x)** and **24.27 versus 25.92 tok/s decode
(0.936x)**. Samples are 139.10, 137.19, 138.42 for prefill and 24.86, 24.27,
24.02 for decode. Report: `/tmp/avx512-attn-final-avx2-warm-speed.log`.

Final Qwen3 dense 4B and sparse 30B-A3B gates each pass **128/128 IDs on both
AVX2 and AVX512**, eight threads, matching F32 KV, flash off. Reports:
`/tmp/avx512-attn-final-{avx2,avx512}-{dense,sparse}-128.log`.
Clean rebuilt binaries are byte-identical to the saved binaries used by
these gates. Clean native build, full tests, architecture checks, Clang
AVX2 build/quant tests, and AVX512 build/quant/compile checks pass without
warnings; logs are `/tmp/avx512-attn-final-{clean,tests,avx2,avx512}.log`.
The native-SIMD MemorySanitizer run also passes the split-row scheduling and
shared-input allocation-failure cases (`/tmp/avx512-attn-quant-msan.log`).

The newly measured Qwen3 dense 4B default-cache rows, at eight threads, are:

| Backend | Prefill bitnet / llama (tok/s) | Ratio | Decode bitnet / llama (tok/s) | Ratio |
|---|---:|---:|---:|---:|
| AVX2 | 143.50 / 151.66 | **0.946** | 19.28 / 21.46 | **0.898** |
| AVX512 | 176.75 / 243.23 | **0.727 — fail** | 18.91 / 21.18 | **0.893** |

AVX2 prefill samples are 143.50, 144.58, 143.39 and decode samples are
20.01, 19.28, 18.88. AVX512 prefill samples are 176.75, 177.09, 175.17;
decode samples are 18.91, 18.91, 18.92. The dense AVX512 prefill gap remains
open despite passing 128/128 tokens and the decode floor. Reports:
`/tmp/avx512-attn-final-{avx2,avx512}-dense-speed.log`.
The dense AVX512 stage profile now points beyond attention: FFN consumes
537.586 ms (gate/up 271.658, down 261.660), versus 51.797 ms for attention,
123.294 ms for QKV, and 114.452 ms for the attention output projection.
This single profile (`/tmp/avx512-attn-final-dense-profile.log`) identifies
the next matrix-kernel investigation; it is not a separate speed gate.

### Dense AVX512 compiler and Q6 tile experiments

The dense Qwen3 4B eight-thread investigation also measured Clang with the
existing explicit AVX512 flags and 16-token Q6 tile. Prefill reached
203.85 / 244.43 tok/s (**0.834**, still below the 0.85 floor); decode reached
19.41 / 20.98 (**0.925**). This is **not an accepted checkpoint**: Clang's
quant integration suite fails `test_q4_matmul_correctness` at line 1127
with both `-march=native` and the explicit AVX512 flags. No model token
gate was completed for that compiler candidate at this measurement. Logs:
`/tmp/dense-avx512-ffn-clang-speed.log` and
`/tmp/dense-avx512-clang-explicit-quant.log`.

A separate AVX512-only Q6 four-token tile experiment passed GCC's quant
integration suite but regressed the single-profile down-projection time
from 261.660 to 282.661 ms (total FFN 537.586 to 555.449 ms).
It was reverted; the 16-token tile and GCC configuration remain in use.
These profile observations are diagnostic, not throughput gates.
Logs: `/tmp/dense-avx512-gcc-tile4-{test,profile}.log`.

The Clang failure subsequently exposed a Q4_0 AVX512 correctness bug:
`q4_join_256_zero` used `_mm512_castsi256_si512`, which leaves the upper
256 bits undefined, before a full-width dot product and reduction.
It now uses explicit zero extension. The scalar/AVX2/AVX512 comparison
test covers all 1–9 row counts and 1–5 block counts, including partial
row groups and poisoned output tails. Clang's AVX512 kernel suite and
native quant integration suite both pass after the fix
(`/tmp/q4-zext-clang-tests.log`). Q8's superficially similar helper only
consumes the low-half result and was not changed. This repair stays
entirely within quant kernels; no model or backend ownership changed.
The subsequent clean GCC build and full `make test`, GCC AVX512 kernel
tests and compile checks, and Clang AVX512 application build all pass
without warnings (`/tmp/q4-zext-{clean,tests,gcc-avx512,clang-build}.log`).
The repaired Clang binary also passes dense Qwen3 4B strict token parity:
128/128 generated token IDs and 8/8 first-token IDs, eight threads,
F32 KV and flash attention off (`/tmp/q4-zext-clang-dense-128.log`).
No new throughput checkpoint is claimed after this repair.

### Q6 physical-row scheduling and repaired Clang throughput

The repaired generic Clang AVX512 binary initially measured Qwen3 4B
`pp128` / `tg64` at eight threads as 205.26 / 245.50 tok/s prefill
(0.836) and 19.32 / 22.03 decode (0.877), three runs. Sixteen threads
increased absolute throughput but lowered the ratios to 0.760 / 0.846.
Keeping eight threads and adding `-mtune=znver4` with the same explicit
ISA and floating-point flags did not help: 0.824 / 0.878. Neither the
tuning flag nor a thread-count change was adopted. Reports:
`/tmp/q4-zext-clang-dense{,-t16}-speed.log` and
`/tmp/dense-clang-znver4-speed.log`.

Q6_K batched matmul now schedules physical rows, as the packed Q4_K paths
already do. A private adapter maps chunk boundaries to whole four-row
kernel groups; the quant arithmetic and public API are unchanged. All
ordinary, prepared-input single, and prepared-input multi Q6 routes use
the adapter. For a 2560-row projection and eight total threads, this
changes 20 minimum-sized group chunks to 32 physical-row chunks.
The change is x86-local and does not alter ARM/Metal or model ownership.

The new exact regression test uses 2113 rows, 768 columns, 16 total
threads, and 2/5/16/17 tokens. It compares all three routes to a direct
serial kernel, including split four-row groups, the final one-row group,
and poisoned output guards. Clang native quant integration tests and a
targeted ASan/UBSan run pass; leak detection was disabled for the
sandboxed sanitizer invocation. Logs:
`/tmp/q6-rows-clang-quant.log`, `/tmp/q6-rows-sanitize.log`.

The first three-run throughput gate was noisy and **failed**: prefill
samples 211.47, 179.17, 199.22 tok/s, median 199.22 / 251.15 = 0.793;
decode 19.59 / 22.08 = 0.887. Three alternating old/new profiles then
showed down-projection / gate-up time ratios of 1.177/1.145/1.174 before
and 1.090/1.103/1.100 after; other stages also varied, so these are
diagnostic observations rather than an independent speed gate.

A subsequent five-run gate passed with tightly grouped bitnet prefill
samples 210.22, 212.98, 210.98, 210.63, 210.15 tok/s:
**210.63 / 234.12 = 0.900 prefill**, and
**19.28 / 20.15 = 0.957 decode**. Decode samples were
19.50, 19.01, 19.28, 20.13, 19.02. These are generic Clang AVX512,
eight-thread, default-cache results, with bitnet F32 KV and llama's
default F16 KV, flash off. Both the failed and passing reports are
retained: `/tmp/q6-rows-clang-dense-speed{,5}.log`.

The measured ISA binaries are built with `make CC=clang bitnet_avx2`
and `make CC=clang bitnet_avx512`, using the Makefile's default explicit
ISA flags (no `-mtune` override). The general Makefile compiler default
has not changed; these Clang measurements must not be attributed to a
default GCC AVX512 build. The clean native build/full suite, Clang AVX2
quant suite, GCC AVX512 quant suite/compile check, and final Clang AVX512
build passed without warnings (`/tmp/q6-rows-final-*.log`). The final
AVX512 binary is byte-identical to the preserved speed/token candidate.
All four strict Qwen3 gates pass 128/128 generated token IDs and 8/8
first-token IDs: dense 4B and sparse 30B-A3B, each on Clang AVX2 and
AVX512, eight threads, F32 KV on both sides, flash off. Reports:
`/tmp/q6-rows-{avx2,avx512}-{dense,sparse}-128.log`.

The final isolated five-run dense repeat **fails prefill acceptance**:
208.74 / 248.38 = **0.840**, despite stable bitnet samples
209.77, 208.60, 209.25, 207.39, 208.74. Decode passes at
19.10 / 20.17 = **0.947**, samples 19.82, 19.10, 19.16, 19.01, 18.99.
Report: `/tmp/q6-rows-final-dense-speed5.log`. The earlier 0.900 result
does not establish repeatable acceptance: llama's measured prefill
throughput changed materially between runs. Dense AVX512 speed remains
open. The local scheduling improvement and exact token gates are retained,
but no broader backend/family completion is implied.

The repaired Clang AVX512 sparse Qwen3 30B-A3B checkpoint passes at eight
threads with `BN_CPU_PREPARED_CACHE_MB=32768` and
`BN_CPU_PREPARE_ALL_EXPERTS=1`: prefill **170.19 / 168.82 = 1.008**,
decode **24.05 / 26.25 = 0.916**, three runs, `pp128`/`tg64`.
Prefill samples are 171.17, 166.60, 170.19 and decode samples are
24.79, 23.95, 24.05. This remains an explicitly warmed-cache result,
not a default-cache claim. Report: `/tmp/q6-rows-final-sparse-speed.log`.

The final Clang AVX2 dense eight-thread check passes prefill but fails
decode: **145.24 / 150.70 = 0.964 prefill**, and
**19.13 / 22.94 = 0.834 decode**. Prefill samples are
145.24, 145.47, 138.81; decode samples are 18.98, 20.22, 19.13.
Report: `/tmp/q6-rows-final-avx2-dense-speed.log`. This scheduling change
does not change decode arithmetic, but the current throughput evidence
does not support claiming both AVX2 dense speed floors pass.

The next dense AVX512 prefill investigation is the Q4_K x16 GEMM's
register pressure: the current generic Clang function has a 4608-byte
stack frame and 387 static stack-address references; the preserved GCC
control has a 6528-byte frame and 734 references. These disassembly counts
include the function's paths and are diagnostic, not dynamic spill counts
or a new performance gate.

### Q4_K AVX512 wider-token GEMM

The Q4_K x16-row GEMM now shares weight decoding over 32 tokens when
available, with a 16-token final region. The caller still rounds the
wide region to 16 tokens: the four-token remainder retains its existing
arithmetic, while every wide-region output retains the original
multiply/add/correction order. No quant layout, public API, model rule,
runtime policy, or ARM/Metal path changed.

Two alternatives were rejected first. Moving operand shuffles to their
use sites left Clang's 4608-byte frame and 387 static stack references
unchanged. An eight-token tile reduced these to 4096 bytes / 350 references
but slowed gate/up work in all three alternating profiles. The retained
32-token candidate has a 5568-byte frame / 352 references and improves
weight reuse; static stack size alone did not predict the useful choice.
The profile logs are `/tmp/q4-live-{before,tile8,control32,tile32}-profile-*.log`.

A temporary independent copy of the original 16-token implementation
matches the candidate bit-for-bit for 64 rows, three quant blocks per
row, and token counts 1/4/8/15/16/17/20/31/32/33/48/63/64/65.
The same comparison passes ASan/UBSan (sandboxed, leak detection disabled).
Logs: `/tmp/q4-live-exact.log`, `/tmp/q4-live-sanitize.log`.
The permanent integration test now checks 17/31/32/33/48/63/64/65-token
batches against independent 16-token calls, scalar tolerance, GEMV tails,
and poisoned unused output space.

The first isolated Qwen3 dense 4B five-run gate at eight threads passes:
prefill **216.62 / 247.03 = 0.877**, decode
**19.30 / 20.58 = 0.938**. Prefill samples are
216.07, 216.66, 185.94, 217.28, 216.62; decode samples are
19.79, 19.30, 19.81, 19.08, 19.07. This uses generic Clang AVX512,
default cache, `pp128` / `tg64`, bitnet F32 KV and llama default F16 KV,
flash off. Report: `/tmp/q4-live-tile32-speed5.log`.

The clean native build/full suite, Clang AVX2 quant suite, GCC AVX512
quant/compile checks, and final Clang AVX512 build/native quant suite
pass without warnings (`/tmp/q4-live-final-*.log`). The rebuilt binary
is byte-identical to the measured candidate. The standard Qwen3 dense
and sparse gates remain 128/128 generated IDs and 8/8 first IDs
(`/tmp/q4-live-{dense,sparse}-128.log`).

Longer prompts expose **pre-existing parity gaps**, so those short-prompt
passes do not establish general Qwen3 acceptance. Two repeated-context
prompts, generating 32 IDs each, match only **24/64 dense** (19/32 and
5/32) and **35/64 sparse** (32/32 and 3/32), although both first IDs match.
The preserved pre-change dense binary fails identically at 24/64;
direct before/after comparisons also match all 32 generated bitnet IDs
for each of the four model/prompt combinations. Thus the wider tile
does not introduce these observed failures. Reports:
`/tmp/q4-live-{dense,sparse}-long64.log`,
`/tmp/q4-live-before-dense-long64.log`, and
`/tmp/q4-live-{dense,sparse}-{before,after}-long{0,1}.log`.

The final isolated five-run dense speed repeat also passes, with prefill
**215.42 / 250.85 = 0.859** and decode
**19.08 / 20.73 = 0.920**. Prefill samples are tightly grouped:
215.38, 216.07, 215.05, 215.42, 216.11; decode samples are
19.21, 19.58, 19.04, 19.07, 19.08. Report:
`/tmp/q4-live-final-dense-speed5.log`. The dense AVX512 throughput floor
is met in both five-run checkpoints, but the newly exposed longer-prompt
token failures still prevent overall dense Qwen3 acceptance.

Both longer-prompt gates pass **64/64** with token-by-token prefill on
both engines (`--no-prefill`, mapped to llama `-b 1 -ub 1`). The prompts
contain 61 and 86 tokens. This narrows the observed dense and sparse
failures to the batched-prefill configuration; it is a diagnostic, not
acceptance by disabling batching or sacrificing throughput. Reports:
`/tmp/q4-live-{dense,sparse}-long64-tokenwise.log`.

The final warmed-cache sparse gate also passes: **169.27 / 167.21 = 1.012
prefill**, **27.46 / 26.27 = 1.045 decode**, three runs at eight threads,
`pp128` / `tg64`, `BN_CPU_PREPARED_CACHE_MB=32768` and
`BN_CPU_PREPARE_ALL_EXPERTS=1`. Samples are 169.27, 168.98, 172.93 for
prefill and 26.36, 29.19, 27.46 for decode. Report:
`/tmp/q4-live-final-sparse-speed.log`. The longer batched-prompt parity
failure still prevents claiming full sparse Qwen3 acceptance.

A concrete next arithmetic comparison is minimum correction handling:
llama's `ggml/src/ggml-cpu/arch/x86/repack.cpp` Q4_K wide GEMM keeps
`acc_min_rows` separate and subtracts at the final store (around lines
2436–2445); bitnet's wide kernel folds each correction into the main
accumulator. This pre-existing difference is a candidate cause of the
long-prompt failures, not yet a demonstrated root cause. The 32-token
experiment intentionally retained the prior bitnet arithmetic while
measuring weight-reuse effects.

### AVX512 batched-prefill arithmetic alignment

Two model-independent arithmetic differences were corrected after the
longer Qwen3 prompts exposed failures. First, the Q4_K wide GEMM now
accumulates minimum corrections separately and subtracts them once at
the final store, matching llama.cpp and the existing four-token kernel.
The strengthened regression test compares every full four-token panel
bit-for-bit across 17/31/32/33/48/63/64/65-token batches, with 32 output
rows, three quant blocks per row, GEMV tails, and poisoned unused output.
It fails before this repair and passes afterward. The change alone
improves sparse long-prompt parity from 35/64 to 46/64, while dense
remains 24/64; it is not the sole cause. Reports:
`/tmp/q4-min-regression-before.log`, `/tmp/q4-min-multiblock-test.log`,
`/tmp/q4-min-{dense,sparse}-long64.log`.

Next, a dense layer-0 trace at position 60 of the 61-token prompt showed
matching displayed Q/K/V projections and attention probabilities, then
different weighted value sums. The AVX512 value path had reproduced the
generic vector-dot order (four sixteen-lane accumulators). The matching
llama build enables LLAMAFILE SGEMM, whose FP32 kernel instead uses one
sixteen-lane accumulator per output. Bitnet now uses that order while
retaining contiguous eight-dimension value loads. The existing AVX2
eight-lane path, ARM/Metal paths, model rules, and ownership are unchanged.

The attention regression test now uses an independent strided-gather
reference with one accumulator. It fails before the change and passes
afterward for 132 cases: six head sizes, eleven KV lengths, and wrapped
or unwrapped caches. Targeted ASan/UBSan runs pass for both the Q4
multi-block panel test and attention tests (leak detection disabled in
the sandbox). Logs: `/tmp/attn-sgemm-regression-before.log`,
`/tmp/attn-sgemm-build-test.log`, `/tmp/q4-min-sanitize.log`,
`/tmp/attn-sgemm-sanitize.log`.

Together these repairs pass both longer batched-prompt AVX512 gates at
**64/64 generated IDs**, dense and sparse, with both first IDs matching.
The standard eight-prompt gates also pass **128/128** for each model.
These use eight threads, F32 KV on both engines, and flash off; batching
remains enabled. Reports: `/tmp/attn-sgemm-{dense,sparse}-{128,long64}.log`.
Clean native builds/full tests, Clang AVX2 quant tests, GCC AVX512
quant/compile checks, and final Clang AVX512 quant/transformer suites all
pass without warnings (`/tmp/attn-sgemm-final-*.log`). The final AVX512
binary is byte-identical to the preserved gate candidate.

The complete 4096-value layer-0 weighted attention output at position 60
is byte-identical to llama.cpp after the reduction fix (16,384-byte dumps
`/tmp/attn-sgemm-kqv-{bn,ll}.bin`). Clang AVX2 also passes the new dense
and sparse longer-prompt gates at 64/64 generated IDs each, with batching
enabled (`/tmp/attn-sgemm-avx2-{dense,sparse}-long64.log`).

The corrected Clang AVX512 dense five-run throughput gate passes at eight
threads: prefill **215.74 / 240.89 = 0.896**, decode
**19.05 / 20.31 = 0.938**. Prefill samples are
215.74, 186.05, 215.74, 214.44, 217.72; decode samples are
19.52, 19.09, 19.05, 19.05, 19.04. This uses default prepared-cache
settings, `pp128` / `tg64`, bitnet F32 KV versus llama's default F16 KV,
and flash off. Report: `/tmp/attn-sgemm-final-dense-speed5.log`.

Sparse Qwen3 30B-A3B also passes on the corrected Clang AVX512 binary:
**170.10 / 173.39 = 0.981 prefill**, **24.08 / 25.34 = 0.950 decode**,
three runs at eight threads with `pp128` / `tg64`. Prefill samples are
170.10, 164.27, 171.05; decode samples are 24.54, 24.08, 24.01.
This uses the explicit warmed expert-cache configuration
`BN_CPU_PREPARED_CACHE_MB=32768 BN_CPU_PREPARE_ALL_EXPERTS=1`, not the
default cache. Report: `/tmp/attn-sgemm-final-sparse-speed.log`.

An isolated five-run dense prefill repeat passes at
**216.80 / 248.65 = 0.872**, samples 217.80, 217.02, 215.52, 216.80,
189.38 (`/tmp/attn-sgemm-final-dense-pp-repeat.log`). Thus the corrected
batched path meets the 0.85 prefill floor in both recorded five-run gates.

For reproducing the added token checks, the first prompt is eight copies
of `The capital of France is Paris. ` followed by
`The capital of France is`; the second is six copies of
`Once upon a time, there was a small village beside a river. ` followed
by `One morning`. Pass each as `--prompt` to `test/compare_llama.sh`
with `-n 32 --strict -t 8 --llama-flash-off` and matching ISA binaries.
The repeat strings include their trailing spaces. These prompts produce
61 and 86 input IDs respectively with both Qwen3 fixtures.

### Hybrid-model parity refresh after AVX512 prefill alignment

The current Clang ISA binaries were rechecked with the standard eight prompts,
16 generated tokens each, eight threads, strict sampled token IDs, F32 KV on
both engines, and llama flash attention disabled:

| Fixture | Runtime | First output IDs | Generated prefix IDs | Strict gate |
| --- | --- | --- | --- | --- |
| Qwen3.6 35B-A3B Q8_0 abliterated | AVX2 | 8/8 | 111/128 | FAIL |
| Qwen3.8 27B UD-Q4_K_XL | AVX512 | 7/8 | 104/128 | FAIL |

Logs: `/tmp/matrix-refresh-q36-sparse-avx2-128.log` and
`/tmp/matrix-refresh-q38-dense-avx512-128.log`. Qwen3.6 diverges after three
capital-prompt tokens and twelve HTTP-prompt tokens. Qwen3.8 diverges on the
first year-prompt token and after twelve tokens on HTTP and sky prompts.
These are unresolved failures, not speed or parity acceptance checkpoints.

For the Qwen3.6 capital prompt, one-thread traces reproduce the fourth-token
decision difference: bitnet ID 32 versus llama ID 760, following common IDs
11751, 13, 198. The five-token prompt places this decision at position 7.
Full binary comparisons establish byte-identical layer-0 output (2048 floats)
and layer-1 SSM gate output (4096 floats). The displayed layer-1 normalized
FFN input and all eight router weights match, while the combined FFN output
has small differences. By layer 8 the SSM gate differences are amplified at
the output projection. This does not yet establish the root cause; inspect
layer-1 expert computation and combination before changing SSM arithmetic.
Trace files are `/tmp/q36-capital-{bn,ll}-pos7.txt`; binary comparisons use
`/tmp/q36-capital-{bn,ll}-l0.bin` and
`/tmp/q36-capital-{bn,ll}-gate1.bin`.

No inference implementation changed during this refresh. Baseline
`make -j8 test` passed (`/tmp/q36-refresh-baseline-tests.log`). Only the dense
Gemma4 31B fixture was found by the Gemma/sparse-name search under
`/data/models/gguf`; the sparse fixture location still needs confirmation.

### Native single-row MoE dot reduction

Corrected `moe_dot_row_avx2` in `src/moe_cpu_kernels.c`: AVX2 now uses the
native F32 dot's adjacent-lane final reduction; AVX512 uses four 16-lane
accumulators with 64-column steps and native reduction. The previous helper
used AVX2 accumulation on both ISAs and a different horizontal-sum tree.
This affects shared-expert gates and single-row router tails, not model
policy, quant layout, ARM, or Metal.

On the captured Qwen3.6 layer-1 shared-gate input, the old AVX2 dot produced
`-2.77486849` and sigmoid `0.0586974397`; native reduction produces
`-2.77486825` and `0.0586974546`, exactly matching llama. Replay:
`/tmp/q36_shared_replay.c`. After correction the complete 2048-float layer-1
output is byte-identical (`/tmp/q36-single-dot-{bn,ll}-l1.bin`). At position 7
the displayed layer-output prefixes now match through layer 20, versus
layer 0 before. The next observed drift occurs in layer 21's convolution
despite matching displayed current QKV projection values, pointing the
next investigation toward prior-token state/input differences.

The standard strict eight-prompt, 16-token, eight-thread Qwen3.6 gates remain
**FAIL**: AVX2 **108/128** (previously 111/128), AVX512 **112/128**. AVX2's
HTTP continuation now matches, while the year continuation newly differs;
the capital mismatch remains. Logs: `/tmp/q36-single-dot-avx2-128.log` and
`/tmp/q36-single-dot-avx512-128.log`. Retaining the native arithmetic fixes
the independently verified operation and layer mismatch, but these results
are not an end-to-end parity improvement claim or acceptance checkpoint.

`test_moe_native_single_dot` checks exact native dot and shared-gate results
at 11 dimensions from 1 through 2051, including unaligned inputs and tails.
It fails with the old native kernel (`/tmp/q36-single-dot-red.log`) and
passes with Clang native AVX512 and explicit AVX2 builds. Baseline tests,
warning-free `make clean && make -j8 bitnet`, and full `make -j8 test` pass.
Logs: `/tmp/q36-single-dot-{build,clean,fulltests,test-avx2}.log`.

The preserved pre-fix AVX512 binary independently scores 108/128 on the same
Qwen3.6 suite (`/tmp/q36-single-dot-before-avx512-128.log`), so AVX512 improves
to 112/128 while AVX2 regresses from 111/128 to 108/128. Qwen3 30B-A3B Q4_K_M
regression gates remain **128/128 on both AVX2 and AVX512**
(`/tmp/q36-single-dot-q3-{avx2,avx512}-128.log`). Throughput was not remeasured
in this change; no new speed acceptance is claimed.

### Qwen3.6 AVX2 parity: attention sigmoid alignment

The next capital-prompt mismatch was traced backward from position 7's SSM
state to full-attention layer 19 at position 6. The complete 4096-float
attention result before gating was byte-identical, but x86's approximate
vector sigmoid differed from llama's `expf` sigmoid. `cpu_backend.c` now
uses `x *= 1 / (1 + expf(-gate))` for this operation on x86, matching the
existing non-flash batched-prefill gate. ARM/Metal and model/quant policy
are unchanged. This also handles non-vector-multiple lengths safely.

The gated-attention branch now exposes pre-gate, Q/gate, post-gate and
softmax arrays through the existing optional debug machinery; the llama
layer probe recognizes the corresponding named stages. Diagnostic logs:
`/tmp/q36-state-{bn-pos4,bn-pos5,bn-pos6,ll-pos4,ll-decode}.txt` and
`/tmp/q36-attn-{bn,ll}-pos6.txt`. Pre-gate binary equality:
`/tmp/q36-attn-{bn,ll}-pregate.bin`. After the fix, full layer-19 output
also matches byte-for-byte (`/tmp/q36-attn-fixed-{bn,ll}-l19.bin`).

Strict eight-prompt gates, 16 tokens per prompt, eight threads, F32 KV on
both engines and flash off:

| Fixture/runtime | Previous | Current | Gate |
| --- | --- | --- | --- |
| Qwen3.6 35B-A3B Q8_0 AVX2 | 108/128 | **128/128** | PASS |
| Qwen3.6 35B-A3B Q8_0 AVX512 | 112/128 | 112/128 | FAIL |

Logs: `/tmp/q36-attn-sigmoid-{avx2,avx512}-128.log`. AVX512 retains the
HTTP and Python continuation mismatches. A targeted Qwen3.8 dense AVX2 sky
check remains 11/16 (`/tmp/q38-attn-sigmoid-sky-avx2.log`); this fix does not
resolve that separate failure.

`test_attention_sigmoid_gate_reference` fails with the approximate kernel
and passes after correction on Clang native AVX512 and explicit AVX2. It
checks exact results and untouched boundaries for unaligned inputs at
lengths 0, 1, 7, 8, 9, 15, 16, 17, 256 and 257. Baseline tests, warning-free
clean build and the full test suite pass. Logs:
`/tmp/q36-attn-sigmoid-{red,build,clean,tests}.log` and
`/tmp/q36-attn-test-avx2.log`.

Isolated AVX2 throughput, default cache policy, eight threads, pp128/tg64,
three measured runs and flash off: decode **14.11 / 14.22 tok/s (0.992)**,
prefill **69.70 / 124.55 tok/s (0.560)**. Bitnet samples are
decode `[14.11, 14.09, 14.13]`, prefill `[71.50, 69.28, 69.70]`.
The combined speed gate fails because prefill remains below 0.85. As in
the earlier CPU speed checkpoints, these use bitnet's default F32 KV and
llama's default F16 KV (strict token gates use F32 on both).
Log: `/tmp/q36-attn-sigmoid-avx2-speed.log`. Command:

```sh
python3 -u test/compare_llama_topk.py \
  /data/models/gguf/qwen3_6/35b_a3b/q8_0/abliterated/qwen3.6_35b_a3b_q8_0.gguf \
  --bitnet ./bitnet_avx2 --bitnet-runtime avx2 --llama-runtime avx2 \
  --llama-bench-bin /home/mark/artalis.io/tools/llama.cpp/build-avx2/bin/llama-bench \
  --skip-topk --benchmark-prefill --benchmark --llama-throughput bench \
  --bench-prompt-tokens 128 --bench-tokens 64 --bench-runs 3 \
  --min-prefill-throughput-ratio .85 --min-throughput-ratio .85 -t 8
```

### Q8_0 GEMV-order batch routing for MoE prefill

Profiling Qwen3.6 35B-A3B Q8_0 AVX2 with 128 prompt token IDs of 1 and eight
threads attributed 789 ms to routed gate/up and 528 ms to routed down
projections (`/tmp/q36-prefill-profile.log`). The GEMV-order batch API was
dispatching Q8_0 token by token, despite native Q8_0 matmul retaining the
same arithmetic on both x86 ISAs.

The quant registry now declares `BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL`
for Q8_0. The x86 GEMV-order batch dispatcher uses normal prepared matmul
for compatible matrices, retaining its policy and allocation fallback.
No MoE/model-specific dispatch or ARM/Metal path changed. Other quant
formats retain their existing GEMV-order routes.

Synthetic replay compares complete outputs exactly for 512x2048 and
2048x512 expert shapes plus a 7x96 tail shape, with 4, 16 and 64 tokens.
All nine cases are bit-exact on AVX2 and native AVX512. Isolated replay
speedups on the two expert shapes were 2.75–5.00x AVX2 and 2.26–4.07x
AVX512; these are kernel-level timings, not whole-model acceptance.
Sources/logs: `/tmp/q36_q8_batch_replay.c`,
`/tmp/q36-q8-batch-replay{,-avx512}.log`.

Permanent `test_q8_matmul_gemv_order` checks the capability, two matrices,
threaded and serial execution, unaligned outputs, untouched guards and
token counts 1, 2, 4, 7, 8, 9 and 17 against per-token `matvec_batch`.
Clang native AVX512 and explicit AVX2 quant tests pass. Baseline tests,
warning-free clean build and full `make test` pass:
`/tmp/q36-q8-route-{baseline,build,clean,tests,test-avx2}.log`.

Qwen3.6 strict 128-token gates remain unchanged: **128/128 AVX2 (PASS)**
and **112/128 AVX512 (FAIL)**, with eight threads, F32 KV on both engines
and flash off. Logs: `/tmp/q36-q8-route-{avx2,avx512}-128.log`.

Isolated AVX2 pp128/tg64, three measured runs, eight threads and default
cache policy: prefill **82.91 / 121.52 tok/s (0.682)** and decode
**14.08 / 14.77 tok/s (0.953)**. Prefill samples are
`[84.38, 79.24, 82.91]`, decode `[14.08, 14.09, 14.08]`.
Bitnet prefill improves about 19% over the preceding 69.70 tok/s checkpoint,
but the combined speed gate still fails its 0.85 prefill threshold.
KV defaults remain F32 bitnet/F16 llama for this speed comparison.
Log: `/tmp/q36-q8-route-avx2-speed.log`; use the preceding sigmoid-alignment
benchmark command with the current `bitnet_avx2` binary.

The follow-up diagnostic profile reports routed gate/up 516 ms, down
284 ms and total MoE 996 ms (`/tmp/q36-q8-route-profile.log`), down from
789/528/1500 ms before routing. Whole prompt time remains 1853 ms in that
single profiling run, so substantial work remains outside those two
projection timings as well. Use the isolated multi-run gate above for
throughput claims, not these individual instrumented samples.

### Qwen3.6 prefill thread-count and dispatch-cost diagnosis

No inference code changed in this diagnostic pass. Q8_0 prepared-weight
allocation is not a hidden x86 expense: its repacked size implementation
is gated to NEON, and its F32-scale auxiliary storage to relaxed WASM SIMD.
The x86 prefill acquisition guard therefore returns without preparing it.

Single instrumented pp128 runs at four and sixteen threads took 3085 ms
and 1419 ms respectively, versus the preceding eight-thread 1853 ms sample.
Logs: `/tmp/q36-prefill-t{4,16}-profile.log`. These are exploratory profiles,
not a substitute for a matched multi-run benchmark.

An isolated three-run, sixteen-thread AVX2 comparison (same model, default
cache, pp128/tg64, flash off) gives:

| Metric | bitnet tok/s | llama tok/s | Ratio |
| --- | --- | --- | --- |
| Prefill | 99.89 | 199.17 | 0.502 FAIL |
| Decode | 24.24 | 21.71 | 1.117 PASS |

Bitnet samples: prefill `[100.53, 99.89, 96.39]`, decode
`[24.24, 24.13, 24.29]`. Speed KV defaults are F32 bitnet/F16 llama.
Log: `/tmp/q36-q8-route-avx2-speed-t16.log`; reproduce using the preceding
benchmark command with `-t 16`. Strict token parity at sixteen threads
is also **128/128**, F32 KV on both, flash off:
`/tmp/q36-q8-route-avx2-t16-128.log`. Higher thread count improves absolute
speed but does not solve prefill parity against the matching reference.

The CPU SSM prefill loop makes four synchronized dispatches per token,
or 15,360 for 30 SSM layers and 128 tokens. An isolated empty-callback
replay of the same task sizes (8192, 16, 32, 32) costs approximately
57 ms with eight threads and 73 ms with sixteen. Source/log:
`/tmp/q36_dispatch_cost.c`, `/tmp/q36-dispatch-cost.log`. This measures
empty-dispatch overhead, not cache effects during real computation; it
does not explain the whole prefill gap. Next investigate SSM computation
and state locality, rather than assuming barrier removal alone suffices.

### SSM locality replay and Q8 projection investigation

A standalone recurrence replay compares token-major dispatch with one
worker processing all 128 tokens for each value head. It reuses the existing
delta kernel without changing its arithmetic: 16 key heads, 32 value heads,
128-dimensional heads, identical inputs and zero initial state. Every
output and final-state float matches byte-for-byte on both AVX2 and AVX512.
After the first run, head-major execution is approximately 1.45x faster at
eight threads and 1.8–2.0x at sixteen. Absolute AVX2 times are roughly
3.17 versus 2.17 ms per layer at eight threads, and 2.2 versus 1.1–1.3 ms
at sixteen. These are synthetic delta-only results, not whole-model gains.
Source: `/tmp/q36_ssm_locality.c`; logs:
`/tmp/q36-ssm-locality-{avx2,avx512}.log`. No production SSM scheduling
change was made: extrapolated delta savings alone are modest relative to
the measured whole-prefill gap.

A separate eight-thread AVX2 replay of SSM QKV/Z/output projection shapes
(8192x2048, 4096x2048, 2048x4096) with 128 tokens is also bit-exact between
per-token GEMV and batch matmul. Batch projection time is approximately
12 ms combined per layer, several times the delta recurrence cost.
Source/log: `/tmp/q36_ssm_projection_replay.c`,
`/tmp/q36-ssm-projection-replay.log`; warning-free compilation log:
`/tmp/q36-ssm-projection-build.log`.

Disassembly of the current AVX2 Q8 matmul shows an out-of-line
`bn_fp16_to_fp32` call inside the block loop and repeated accumulator
loads/stores on the stack. This identifies a concrete next kernel target:
test native FP16 conversion and fixed-size full-tile specialization before
introducing broader SSM orchestration changes. No inference code changed
during this diagnostic pass.

### AVX2 Q8 batch native half conversion

The Q8_0 AVX2 batch kernel now uses F16C half-to-float conversion when
compiled with that capability, retaining the existing software fallback.
Only weight-scale conversion changes; dot accumulation, FMA and reduction
order are unchanged. No model/runtime routing or ARM/Metal code changed.
Disassembly confirms the inner-loop conversion call is gone, although
dynamic-tile accumulator loads/stores remain a separate optimization target.

Two isolated alternating baseline/candidate replays of the preceding three
projection shapes remain bit-exact. Batch milliseconds per shape are
`[5.781, 3.459, 2.989]` versus `[5.230, 3.078, 2.707]`, then
`[5.788, 3.379, 3.001]` versus `[5.242, 3.090, 2.714]` (eight threads,
128 tokens). These roughly 9–11% time reductions are synthetic kernel
measurements, not whole-model speed acceptance. Baseline executable:
`/tmp/q36_ssm_projection_replay`; candidate: `/tmp/q8_convert_projection`.

The permanent GEMV-order batch regression now covers signed zero,
subnormal/normal boundaries, signed unit scales and maximum finite half
scales. Explicit Clang AVX2 tests, the warning-free clean default build,
and full `make test` pass. The AVX2 kernel also compiles without F16C.
Logs: `/tmp/q8-convert-{baseline,avx2-build,build,tests}.log`.
Qwen3.6 35B-A3B Q8_0 strict parity remains **128/128 AVX2** at eight
threads, F32 KV on both engines and flash off:
`/tmp/q8-convert-avx2-128.log`.

Isolated whole-model AVX2 pp128/tg64, three runs, eight threads and default
cache: prefill **96.65 / 124.23 tok/s (0.778, FAIL)** and decode
**14.09 / 14.31 tok/s (0.985, PASS)**. Bitnet samples are
`[94.33, 97.51, 96.65]` prefill and `[14.08, 14.09, 14.09]` decode.
Prefill is about 17% above the preceding 82.91 tok/s checkpoint but still
below the 0.85 acceptance threshold. Speed KV defaults remain F32 bitnet
and F16 llama; flash is off. Log: `/tmp/q8-convert-avx2-speed.log`.
Use the preceding eight-thread benchmark command with
`--bitnet /tmp/q8-convert-avx2` to reproduce. Fixed-size full-tile
specialization is the next kernel investigation, not part of this change.

### AVX2 Q8 full-tile register accumulation

The Q8 AVX2 batch kernel now separates constant eight-token full tiles
from the existing variable-width tail. Each token retains its original
block traversal, scaled FMA and final horizontal reduction. Clang
disassembly shows eight register-destination FMAs in the full-tile block
loop; the tail still uses its existing accumulator array. No routing,
model, backend-state or ARM/Metal behavior changes.

Projection replay remains exact for all outputs. Two alternating
conversion-only/full-tile replay pairs gave batch milliseconds
`[5.231, 3.084, 2.705]` / `[4.534, 2.463, 2.466]`, then
`[5.279, 3.076, 2.792]` / `[4.514, 2.459, 2.405]` for the three
preceding SSM projection shapes. The unchanged GEMV reference also ran
faster in the new executable, so these timings alone do not isolate the
tile's gain from executable-layout effects. Whole-model gates are the
acceptance evidence. Executables: `/tmp/q8_convert_projection` and
`/tmp/q8_tile_projection`.

The permanent exact GEMV-order regression now additionally covers token
counts 15, 16, 31, 32 and 33, including repeated full tiles and tails,
with serial/threaded execution and output guards. Explicit Clang AVX2
quant tests pass (`/tmp/q8-tile-avx2-build.log`), and compilation without
F16C also succeeds.

The warning-free clean default build and full test suite pass:
`/tmp/q8-tile-{baseline,build,tests}.log`. Qwen3.6 35B-A3B Q8_0
strict sampled-token parity remains **128/128 AVX2**, eight threads,
F32 KV on both engines, flash off: `/tmp/q8-tile-avx2-128.log`.

The isolated eight-thread, default-cache pp128/tg64 speed gate now passes:
prefill **112.04 / 124.12 tok/s (0.903)** and decode
**14.06 / 14.23 tok/s (0.988)**. Three bitnet samples are
`[114.45, 112.04, 112.02]` prefill and `[14.06, 14.26, 14.06]` decode.
An independent five-run prefill check also passes the 0.85 threshold:
**109.13 / 123.93 tok/s (0.881)**, bitnet samples
`[103.96, 107.69, 109.13, 112.96, 117.45]`.
Speed KV defaults remain F32 bitnet/F16 llama, with flash off.
Logs: `/tmp/q8-tile-avx2-speed.log`, `/tmp/q8-tile-avx2-pp-repeat.log`.
Reproduce with the preceding eight-thread benchmark command using
`--bitnet /tmp/q8-tile-avx2`; the independent repeat omits `--benchmark`
and uses `--bench-runs 5`.

Two longer batched-prefill prompts also match all 32 generated IDs each:
eight repetitions of `The capital of France is Paris. ` followed by
`The capital of France is`, and six repetitions of
`Once upon a time, there was a small village beside a river. ` followed
by `One morning`. Logs: `/tmp/q8-tile-avx2-long-{capital,story}.log`.
Together with the eight short prompts, this is **192/192 sampled IDs**
for this Qwen3.6 sparse AVX2 checkpoint, not acceptance of the outstanding
AVX512/CUDA or broader model matrix.

### Native AVX512 SSM convolution and output gate

The Qwen3.6 sparse Python prompt first showed a displayed layer-output
difference at layer 5, position 6 (last prompt token). Its complete
8192-value QKV projection was already byte-exact with llama.cpp:
`/tmp/q36-py512-{bn,ll}-qkv5.bin`. The next stage, convolution/SiLU,
was using AVX2 arithmetic because the AVX512 dispatcher selected narrow
kernels by gate type and tensor size.

The AVX512 SSM dispatch wrappers now call their native convolution and
output-gate implementations without those shape exceptions. No model
conditions, quant changes, backend-state changes or ARM/Metal changes are
introduced. AVX2 dispatch remains unchanged. Native convolution uses the
existing sequential FMA and native vector SiLU; the output gate uses its
existing native vector activation and normalization.

Convolution-only routing raised Qwen3.6 AVX512 from 112/128 to 120/128:
Python and HTTP passed, while sky diverged at token 9. Enabling native
output-gate arithmetic as well gives **128/128** on the eight short
prompts and **64/64** on the same two longer prompts used above, thus
**192/192** sampled IDs. All use eight threads, F32 KV on both engines,
flash off, batch prefill enabled. Logs:
`/tmp/q36-nativeconv-avx512-128.log`,
`/tmp/q36-nativeboth-avx512-128.log`,
`/tmp/q36-nativeboth-long-{capital,story}.log`.
The complete 8192-value layer-5 convolution output also matches exactly:
`/tmp/q36-py512-native-conv5.bin`, `/tmp/q36-py512-ll-conv5.bin`.

Qwen3.8 dense AVX512 improves from the preceding 104/128 checkpoint to
**120/128**, with the year prompt now passing. HTTP and sky still each
match only 12/16, so its token gate remains **FAIL**:
`/tmp/q38-nativeboth-avx512-128.log`. The earlier convolution-only
Qwen3.8 attempt is invalid because a clean build removed its executable
between prompts; the final gate uses a saved binary outside the build tree.

Permanent exact native-dispatch tests cover both gate types, convolution
dimensions 1, 15, 16, 17, 33, 8192 and 8208, and output-gate head sizes
1, 15, 16, 17 and 128 with 32/64 heads. They verify dispatch against the
existing native kernels, including convolution state. Existing scalar
tolerance tests continue to check the kernel calculations. Clang native
tests, explicit AVX2 SSM tests, ASan/UBSan, warning-free clean default
build and full `make test` pass. Logs:
`/tmp/q36-nativeboth-{ssm,ssm-avx2,sanitize,build,tests}.log`.
LeakSanitizer required execution outside the sandbox's ptrace environment.

Isolated Qwen3.6 AVX512 pp128/tg64, three runs, eight threads, default
cache: prefill **71.72 / 131.00 tok/s (0.547, FAIL)** and decode
**13.86 / 14.80 tok/s (0.936, PASS)**. Bitnet samples are
`[70.84, 71.72, 71.96]` prefill and `[13.86, 13.94, 13.82]` decode.
Speed KV defaults remain F32 bitnet/F16 llama; flash is off.
Log: `/tmp/q36-nativeboth-avx512-speed.log`. Reproduce with the preceding
pp128/tg64 command, using `/tmp/q36-nativeboth-avx512`, both runtime
selectors `avx512` and llama's `build-avx512/bin/llama-bench`.
Thus this checkpoint establishes token parity, not combined speed
acceptance. The separate AVX512 Q8 batch kernel still has out-of-line
half conversion and a dynamic four-row/eight-token accumulator tile;
the successful AVX2 batch optimizations have not yet been applied there.

### AVX512 Q8 fixed pair tile

The AVX512 Q8 batch kernel now handles full four-row/eight-token tiles
as two fixed two-row/eight-token tiles. Sixteen independent accumulators
leave register space for dot-product temporaries; Clang disassembly
confirms register-destination FMAs throughout the full-tile block loop.
Partial rows/tokens retain the existing bounded loop. Batch scale
conversion uses F16C when compiled in, with the software fallback retained.
Per-output block traversal, FMA and eight-lane horizontal reduction are
unchanged. Quant dispatch, model/runtime concerns, decode kernels and
ARM/Metal paths are untouched.

Conversion alone was not a win in this kernel: two isolated projection
replay pairs showed baseline batch milliseconds near `[7.95, 4.05, 4.02]`
and conversion-only near `[8.86, 4.51, 4.37]`. Adding the fixed pair tile
reduced the three SSM projection shapes from
`[8.045, 4.064, 3.923]` to `[4.911, 2.479, 2.403]`, then
`[7.988, 4.085, 3.946]` to `[4.824, 2.472, 2.366]` in an independent
alternating replay. All outputs are exact against per-token GEMV;
eight threads, 128 tokens. These roughly 39–40% batch-time reductions
are kernel-level measurements, not whole-model acceptance.
Executables: `/tmp/q8_wide_{baseline,convert,tile}`; source:
`/tmp/q36_ssm_projection_replay.c`.

The exact GEMV-order batch regression adds token counts 63, 64 and 65,
retaining half-scale boundary cases, two matrices, row tails, threaded
and serial execution, unaligned outputs and guards. Clang native quant
tests pass (`/tmp/q8-wide-tests-build.log`); the kernel also compiles
without F16C.

The warning-free clean default build and full `make test` pass:
`/tmp/q8-wide-{baseline,build,tests}.log`. A focused ASan/UBSan run of
the exact batch regression passes (`/tmp/q8-wide-sanitize.log`), with
leak detection disabled for the sandbox. Qwen3.6 sparse strict AVX512
parity remains **128/128** for the eight short prompts, eight threads,
F32 KV on both engines, flash off: `/tmp/q8-wide-avx512-128.log`.

Isolated whole-model AVX512 pp128/tg64, three runs, eight threads and
default cache: prefill **87.73 / 130.55 tok/s (0.672, FAIL)** and decode
**13.78 / 14.26 tok/s (0.966, PASS)**. Bitnet samples are
`[86.80, 87.73, 88.67]` prefill and `[13.72, 13.78, 14.01]` decode.
Prefill is about 22% above the preceding 71.72 tok/s checkpoint, but
still below the 0.85 acceptance threshold. Speed KV defaults remain
F32 bitnet/F16 llama, flash off. Log: `/tmp/q8-wide-avx512-speed.log`;
reproduce with the preceding AVX512 benchmark command and
`--bitnet /tmp/q8-wide-avx512`. Smaller expert batches still use the
variable-width loop and are the next kernel investigation; no speed
claim for that prospective optimization is implied.

### AVX512 Q8 physical-row scheduling

Small-batch replay found a scheduling contribution to the remaining Q8
prefill gap. A 512-row expert matrix was submitted as 128 four-row groups.
With eight workers and the pool's minimum chunk of 32, that offers only
four work chunks. The Q8 AVX512 batch adapter now submits physical rows
and rounds both chunk boundaries upward to four-row groups, matching the
existing Q4/Q6 scheduling convention. Every group remains assigned once;
the threadpool itself and kernel arithmetic are unchanged.

An exploratory isolated replay of 512x2048, 2048x512 and 8192x2048 with
2, 4, 8 and 16 tokens is exact against GEMV on AVX2 and AVX512. For the
512x2048 case, one before/after scheduling replay gives batch milliseconds
`[0.036, 0.054, 0.045, 0.082]` versus
`[0.030, 0.037, 0.032, 0.053]`. These very short measurements identify
the next experiment, not whole-model acceptance. Source:
`/tmp/q8_small_replay.c`; executables:
`/tmp/q8_small_{avx2,avx512,rows}`. No additional small-token tile change
was made in this pass.

The new permanent row-scheduling regression uses 2113 rows, 96 columns,
16 workers and token counts 2, 4, 8, 9 and 17, repeated three times.
The resulting 33-row chunks cut through four-row groups, and the final
group contains one valid row. All output floats must match serial GEMV
exactly, with unaligned output and untouched guards.
Clang native quant tests pass: `/tmp/q8-small-rows-tests-build.log`.

The clean default build is warning-free and the full suite passes:
`/tmp/q8-small-{baseline,rows-build,rows-tests}.log`. Focused ASan/UBSan
batch and row-scheduling tests pass with leak detection disabled for the
sandbox (`/tmp/q8-small-rows-sanitize.log`). Qwen3.6 sparse AVX512
strict parity remains **128/128** on the eight short prompts, eight
threads, F32 KV on both engines and flash off:
`/tmp/q8-small-rows-avx512-128.log`.

Isolated AVX512 pp128/tg64, three runs, eight threads, default cache:
prefill **98.71 / 131.39 tok/s (0.751, FAIL)** and decode
**13.77 / 14.93 tok/s (0.922, PASS)**. Bitnet samples are
`[98.70, 99.34, 98.71]` prefill and `[13.86, 13.77, 13.75]` decode.
Prefill improves about 13% over the preceding 87.73 tok/s checkpoint,
but remains below 0.85. Speed KV defaults are F32 bitnet/F16 llama,
flash off. Log: `/tmp/q8-small-rows-avx512-speed.log`; reproduce with
the preceding AVX512 benchmark command using
`--bitnet /tmp/q8-small-rows-avx512`. Fixed small-token kernels remain
an outstanding experiment after correcting this scheduling issue.

### AVX512 Q8 short-token specialization

Q8 AVX512 batches below eight tokens now use fixed 4/2/1-token pieces
per output row. A shared inline helper keeps the calculation in one
place; Clang specializes its constant counts into four, two and one
register accumulators. Full eight-token tiles and physical-row scheduling
remain unchanged. Each output retains its block traversal, scaled FMA
and eight-lane reduction; no model/runtime, decode or ARM/Metal path changes.

The small-batch replay remains exact for all twelve shape/token cases.
Exploratory batch timings for 2/4 tokens improve from 0.028/0.034 ms to
0.022/0.024 ms at 512x2048, and 0.022/0.027 ms to 0.017/0.022 ms at
2048x512. These short three-repetition measurements are not whole-model
acceptance. Source: `/tmp/q8_small_replay.c`; before/after executables:
`/tmp/q8_small_rows`, `/tmp/q8_short_replay`.

The permanent exact GEMV-order regression now covers every token count
1 through 17, plus the existing larger boundaries through 65, including
all combinations of short pieces, row tails, scale boundaries, threaded
execution and output guards. Clang native quant tests pass:
`/tmp/q8-short-clang-build.log`; compilation without F16C also passes.

The warning-free clean default build and full suite pass:
`/tmp/q8-short-{baseline,build,tests}.log`. Focused ASan/UBSan tests for
exact batch output and row scheduling pass with leak detection disabled
for the sandbox (`/tmp/q8-short-sanitize.log`). Qwen3.6 sparse AVX512
remains **192/192** sampled IDs: the eight short prompts plus the two
longer prompts described above, eight threads, F32 KV on both engines,
flash off. Logs: `/tmp/q8-short-avx512-128.log`,
`/tmp/q8-short-long-{capital,story}.log`.

Isolated AVX512 pp128/tg64, three runs, eight threads and default cache:
prefill **110.73 / 130.43 tok/s (0.849, FAIL)** and decode
**13.72 / 14.96 tok/s (0.917, PASS)**. Bitnet samples are
`[111.65, 110.73, 109.94]` prefill and `[13.67, 13.72, 13.76]` decode.
An independent five-run prefill repeat is lower:
**102.20 / 131.32 tok/s (0.778, FAIL)**, samples
`[99.46, 102.92, 101.57, 102.20, 112.98]`. Neither result establishes
the 0.85 gate; retain both rather than interpreting the near-threshold
first run as acceptance. Speed KV defaults remain F32 bitnet/F16 llama,
flash off. Logs: `/tmp/q8-short-avx512-speed.log`,
`/tmp/q8-short-avx512-pp-repeat.log`. Reproduce with the preceding AVX512
command using `/tmp/q8-short-avx512`; for the repeat omit `--benchmark`
and use `--bench-runs 5`.
Mixed-size tails still reread weights across their 4/2/1 pieces, leaving
direct fixed 3/5/6/7-token kernels as a further experiment.

### AVX512 Q8 direct remainder tiles

Q8 AVX512 short tiles now specialize all remainders 1–7 directly, using
the same inline helper with up to seven independent accumulators. A
3/5/6/7-token batch no longer rereads a row's weights in separate 4/2/1
pieces. Eight-token tiles, physical-row scheduling, accumulation order,
decode and ARM/Metal paths are unchanged.

A twenty-repetition isolated replay of 3/5/6/7 tokens remains exact for
all twelve cases. Batch milliseconds before/after direct remainders:

| Shape | Split 4/2/1 pieces | Direct remainder |
| --- | --- | --- |
| 512x2048 | 0.031, 0.035, 0.039, 0.050 | 0.021, 0.025, 0.026, 0.028 |
| 2048x512 | 0.023, 0.029, 0.032, 0.040 | 0.020, 0.022, 0.023, 0.025 |
| 8192x2048 | 0.378, 0.418, 0.430, 0.603 | 0.218, 0.259, 0.268, 0.287 |

Source: `/tmp/q8_mixed_replay.c`; executables:
`/tmp/q8_mixed_{base,direct}`. These kernel measurements do not establish
whole-model speed acceptance. Clang inlines the constant-count helpers
and retains register accumulators on the new paths.

The permanent 16-worker, 2113-row exact scheduling regression now also
covers 3, 5, 6, 7, 13, 14 and 15 tokens. The existing batch regression
already covers every count 1–17 and larger boundaries. Clang native
quant tests pass (`/tmp/q8-direct-clang-build.log`), and the kernel
compiles without F16C.

The clean default build is warning-free and the full suite passes:
`/tmp/q8-direct-{baseline,build,tests}.log`. Focused ASan/UBSan passes
with leak detection disabled for the sandbox (`/tmp/q8-direct-sanitize.log`).
Qwen3.6 sparse AVX512 again matches **192/192** sampled IDs on the eight
short and two longer prompts, eight threads, F32 KV on both engines,
flash off: `/tmp/q8-direct-avx512-128.log`,
`/tmp/q8-direct-long-{capital,story}.log`.

The isolated three-run AVX512 pp128/tg64 gate at eight threads and default
cache still fails prefill: **107.50 / 130.53 tok/s (0.824)**. Decode
passes: **13.76 / 14.14 tok/s (0.973)**. Bitnet samples are
`[98.98, 107.50, 110.75]` prefill and `[13.76, 13.77, 13.73]` decode.
Speed KV defaults remain F32 bitnet/F16 llama, flash off. The faster
tail replay has not established whole-model speed acceptance.
Log: `/tmp/q8-direct-avx512-speed.log`; reproduce with the preceding
AVX512 benchmark command using `/tmp/q8-direct-avx512`.

Sequential diagnostic profiles with 128 prompt token IDs of 1 show
similar total times on current AVX512 and the retained AVX2 checkpoint:
1283.9 versus 1298.5 ms. Total MoE is 742.7 versus 750.0 ms, with routed
gate/up 401.4 versus 372.9 ms and down 199.7 versus 210.1 ms. Logs:
`/tmp/q8-direct-profile{512,2}.log`. These single instrumented runs are
not acceptance measurements, and the profile still does not break out
SSM operations. They do not support attributing the remaining gap solely
to AVX512 SSM computation; routed MoE projection work remains substantial.

### Qwen3.8 AVX512 Q3_K prefill input alignment

The remaining HTTP divergence was traced back to prefill layer 0. SSM
output and FFN normalization matched; the first differing projection was
`blk.0.ffn_up.weight`, a Q3_K tensor (gate/down are IQ4_XS). The complete
5120-value FFN input is byte-exact with llama.cpp:
`/tmp/q38-http512-{bn,ll}-ffnin0.bin`.

Q3_K decode already quantized activations to Q8_K, but AVX512 prefill
selected a float-input batch kernel, and the quant registry still
declared float-input behavior on x86. The registry now reports Q3_K's
actual x86 input convention, and the incompatible AVX512 batch override
is removed. Prefill uses the existing Q8_K matvec fallback. This is a
format-level correction; model/runtime and ARM/Metal paths are unchanged.
The complete corrected 17408-value up projection matches llama.cpp:
`/tmp/q38-q3k-{bn,ll}-up0.bin`.

The existing exact Q3_K batch/matvec regression had excluded AVX512.
Enabling it reproduced the failure (`/tmp/q38-q3k-red.log`); after the
correction it passes. It now checks registry behavior, serial and
three-worker execution, token counts 1, 3, 8 and 11, and untouched
output tails. Clang native quant tests pass (`/tmp/q38-q3k-green.log`).
This correctness fix does not by itself establish speed acceptance.

The intermediate Q3_K-only full gate reached 116/128 tokens, versus the
prior 120/128 (`/tmp/q38-q3k-avx512-128.log`). Local prefill agreement did
not establish end-to-end parity. A subsequent IQ4_NL single-matvec/input
correction reached 120/128 on AVX512 and 118/128 on AVX2
(`/tmp/q38-native-inputs-avx{512,2}-128.log`); both failed the strict gate.

The next decode mismatch was isolated to layer 1's IQ4_NL down projection
at position 8 of the HTTP prompt. Its complete 17408-value input matched,
but the model uses `bn_quant_matvec_batch`, whose generic float-input
shortcut bypassed the corrected single-matvec dispatch. The x86 batch
route now shares Q8 activation quantization for IQ4_NL, and Q8_K for
Q3_K/IQ3_S, using their existing integer kernels. No model-specific
conditions or ARM/Metal changes were introduced.

After correcting the actual batch route, all 5120 output floats of that
IQ4_NL projection are byte-exact with llama.cpp (`/tmp/q38-batch-down.bin`
and `/tmp/q38-iq4nl-ll-out.bin`). Exact regression checks cover task counts
1, 2 and 25 (including fallback), serial/three-worker dispatch and untouched
output tails for IQ4_NL, Q3_K and IQ3_S. The new check failed before the
batch correction (`/tmp/q38-batch-red.log`); the corrected native and AVX2
quant suites pass. Focused ASan/UBSan tests pass with leak checking disabled
(`/tmp/q38-batch-sanitize.log`). Full token gates remain separate from this
projection-level evidence; no speed acceptance is claimed here.

The clean default build is warning-free and the full post-change test suite
passes (`/tmp/q38-batch-default-build.log`, `/tmp/q38-batch-tests.log`).
The corrected Qwen3.8 short-token gate passes **128/128 on AVX512**.
AVX2 improves to **123/128**, still failing on the sky prompt (11/16);
all other prompts match. Logs: `/tmp/q38-batch-avx{512,2}-128.log`.
Both use eight threads, F32 KV on both engines, and llama flash attention
disabled. Longer-prompt parity and throughput acceptance are not yet
established for this checkpoint.

The two longer-prompt gates subsequently pass 32/32 each
(`/tmp/q38-batch-long-{capital,story}.log`), bringing the AVX512 checkpoint
to **192/192 tokens**. Throughput remains a separate pending gate.

The isolated eight-thread speed gate then reports decode **3.02/2.92
tokens/s (1.034)** and prefill **16.13/19.68 tokens/s (0.819)**.
Decode passes; prefill fails the 0.85 threshold, so this checkpoint is
**not speed-accepted**. Three bitnet samples are decode 3.02/3.02/3.02
and prefill 16.03/16.16/16.13. The benchmark uses 128 prompt tokens,
64 decode tokens, default cache settings, F32 bitnet KV and F16 llama KV.
Log: `/tmp/q38-batch-avx512-speed.log`.

### Qwen3.8 AVX2 Q6_K reference-batch routing

The remaining AVX2 sky mismatch first appears at layer 1's SSM output
projection on the first decode step (position 6). The full 6144-value
gated input is byte-exact (`/tmp/q38-sky2-gate-{bn,ll}.bin`), and the
weight is Q6_K. A captured-input replay against all 5120 llama output
values finds zero differences for single matvec with or without
`REFERENCE_DOT`, and for ordinary batch matvec. Only flagged batch matvec
differs (4629 values, maximum error 4.76837e-7).

The x86 batch dispatcher selected the scalar-per-superblock reduction
kernel for flagged Q6_K, whereas single matvec selected the existing
four-row lane-wise accumulation kernel. The batch route now selects the
same four-row kernel and group count. Transformer policy, model rules,
kernel arithmetic and non-x86 routing are unchanged. The corrected model
projection is byte-exact (`/tmp/q38-q6route-out-bn.bin` versus
`/tmp/q38-sky2-out-ll.bin`).

An exact regression reproduces the old failure and passes after correction
on AVX2 and native AVX512 (`/tmp/q38-q6route-{red,green,native-tests}.log`).
It covers 133 rows (including a partial four-row group), three superblocks,
1/2/25 tasks, serial and three-worker dispatch, and untouched output tails.
Full-model parity is being regated; local projection agreement is not a
claim of end-to-end acceptance.

The Q6_K correction passes a warning-free clean build, full tests and the
focused ASan/UBSan check (leak checking disabled), but its complete AVX2
short gate **regresses to 113/128**, from 123/128. Year (10/16), sky
(11/16) and Python (12/16) fail (`/tmp/q38-q6route-avx2-128.log`). This
remains an intermediate, unaccepted checkpoint. The year trace is exact
in displayed values through prefill and decode layer 13, then first
differs at layer 14's Q4_K SSM output projection; its complete 6144-value
input matches (`/tmp/q38-year2-gate-{bn,ll}.bin`). Further routing analysis
is required rather than assuming the local Q6_K fix establishes parity.

### Qwen3.8 AVX2 Q4_K unpacked batch routing

Replaying the year prompt's layer 14 Q4_K projection against all 5120
llama outputs finds zero differences for single matvec, with or without
`REFERENCE_DOT`. Both batch routes differ in 4260 values (maximum error
3.57628e-7), despite a byte-exact 6144-value input. AVX2's unpacked batch
route now uses the same existing four-row kernel as single matvec, with
the corresponding four-row item count. Packed routing, AVX512 routing,
transformer policy, and non-x86 paths are unchanged.

The corrected model projection matches all 5120 floats
(`/tmp/q38-q4route-out-bn.bin` versus `/tmp/q38-year2-out-ll.bin`). The
new exact regression fails before the change and passes afterward
(`/tmp/q38-q4route-{red,green}.log`), checking ordinary and reference-dot
tasks, 133 rows, three superblocks, 1/2/25 tasks, serial/three-worker
execution and untouched output tails. Native AVX512 quant tests also
pass, as do focused AVX2 ASan/UBSan tests with leak checking disabled
(`/tmp/q38-q4route-{native-tests,sanitize}.log`). Full token gates are
pending; this remains a local correction, not an acceptance claim.

The clean default build is warning-free and all tests pass
(`/tmp/q38-q4route-default-build.log`, `/tmp/q38-q4route-tests.log`).
The Qwen3 dense AVX2 short-token regression gate passes 128/128
(`/tmp/q3dense-q4route-avx2-128.log`).

Qwen3.8 AVX2 now passes **128/128 short tokens plus 32/32 on each longer
prompt: 192/192 total**. This resolves year, sky and Python from the
intermediate Q6_K-only trial. Logs: `/tmp/q38-q4route-avx2-128.log` and
`/tmp/q38-q4route-long-{capital,story}.log`. The gates use eight threads,
F32 KV on both engines, and llama flash attention disabled.

The isolated AVX2 throughput gate also passes: **prefill 14.49/16.31
tokens/s (0.889)** and **decode 3.01/2.89 tokens/s (1.042)**. Bitnet's
three prefill samples are 14.49/14.45/14.55 and decode samples are
3.01/3.01/3.01. This is an accepted eight-thread checkpoint for this
Qwen3.8 dense model: 192/192 tokens and both speed ratios above 0.85.
The benchmark uses default cache settings, 128 prompt tokens, 64 decode
tokens, and F32 bitnet KV versus F16 llama KV. Log:
`/tmp/q38-q4route-avx2-speed.log`. This does not establish other model,
thread-count or backend checkpoints.

The subsequent current-code AVX512 diagnostic profile (`/tmp/q38-current-profile-all.log`)
uses 128 token IDs of value 1 and eight threads. Total prefill is 8150 ms;
pure-Q5_K prepared-input calls account for 1763 ms across 32 two-matrix
calls and 1475 ms across 42 single-matrix calls. These calls still select
`bn_quant_q5k_avx2_sdot_matmul_range` in the AVX512 prepared-input dispatcher.
Q3_K is not the main bottleneck: a preceding current-code run measures
only 102 ms across its three projections (`/tmp/q38-current-profile.log`).
This instrumentation lives in `/tmp`, not production code. Nested wrapper
timings must not be summed together; these are diagnostic observations,
not replacement throughput-gate results.

### Q5_K four-token weight reuse on AVX512

The Q5_K native-order batch entry previously called single-token GEMV for
every token, repeating weight decoding. In AVX512 builds it now decodes
one row/superblock for four tokens, maintaining separate lane-wise
accumulators and min corrections. Block FMA order and the final horizontal
reduction are unchanged; remaining 1–3 tokens use the existing GEMV path.
The optimization uses AVX512's larger register file while retaining AVX2
intrinsics. Non-AVX512 execution, decode, model policy and ARM/Metal paths
are unchanged.

An isolated eight-thread synthetic replay (128 tokens, mean of five runs)
remains byte-exact with GEMV and measures:

| Rows × columns | Previous ms | Four-token ms |
| --- | ---: | ---: |
| 17408 × 5120 | 37.146 | 24.309 |
| 5120 × 17408 | 35.199 | 24.366 |
| 5120 × 6144 | 13.657 | 8.575 |

Logs: `/tmp/q5-batch-{base,tile4}.log`. These are kernel measurements,
not full-model speed acceptance. The exact regression covers 137 rows,
three superblocks, token counts 1–17 and 31–33, serial/three-worker
execution and untouched output tails. Native and AVX2 quant suites pass,
as does focused ASan/UBSan with leak checking disabled
(`/tmp/q5-tile4-{native-tests,avx2-tests,sanitize}.log`). Full-model gates
are pending.

The clean default build is warning-free and all tests pass
(`/tmp/q5-tile4-default-build.log`, `/tmp/q5-tile4-tests.log`). Both longer
Qwen3.8 prompts retain 32/32 tokens each
(`/tmp/q5-tile4-q38-long-{capital,story}.log`). The short-token gate also
passes 128/128 (`/tmp/q5-tile4-q38-128.log`), preserving **192/192 total**.
Full-model throughput remains pending for this optimization.

The isolated Qwen3.8 AVX512 throughput gate subsequently passes: **prefill
21.16/19.63 tokens/s (1.078)** and **decode 3.02/2.91 tokens/s (1.038)**.
Bitnet samples are prefill 21.17/21.16/20.83 and decode 3.01/3.03/3.02.
Combined with 192/192 token parity, this is an **accepted eight-thread
Qwen3.8 dense AVX512 checkpoint**. The earlier 0.819 prefill failure is
superseded for this model/configuration. It uses default cache settings,
128 prompt tokens, 64 decode tokens and F32 bitnet KV versus F16 llama KV.
Log: `/tmp/q5-tile4-q38-speed.log`. No broader model/backend acceptance is
implied.

### Post-Q5_K-tile broader CPU gates

Qwen3.5 dense 27B Q5_K_M passes **128/128 short tokens on both AVX2 and
AVX512**, using the current retained binaries at eight threads, F32 KV on
both engines and flash attention disabled. Logs:
`/tmp/q5-tile4-q35-avx{2,512}-128.log`. Longer-prompt and current speed
gates are still pending; the Qwen3.8 speed result is not transferred to
this model.

The two longer Qwen3.5 prompts subsequently pass 32/32 each on both
backends, for **192/192 tokens per backend**. These longer runs use
`--maxseq 4096` on both engines. Logs:
`/tmp/q5-tile4-q35-avx{2,512}-long-{capital,story}.log`.
Current-model speed acceptance remains pending.

Initial Gemma4 dense CPU runs without an explicit context cap returned no
bitnet completion. A direct `--maxseq 4096` run succeeds, and both parity
gates have been restarted with that cap on both engines. The uncapped
failure's cause is not established by the first harness logs; these are
failed runs, not parity evidence. Capped logs:
`/tmp/q5-tile4-gemma4-avx{2,512}-ctx4096.log`.

The capped Gemma4 AVX2 short gate passes 128/128 and both longer prompts
pass 32/32 each: **192/192 total**, with speed still unverified.
The AVX512 short gate reports 124/124 matching prefixes but correctly
fails because generation lengths differ. A targeted year prompt shows
12 matching bitnet tokens versus 16 llama tokens; both repeat the word
"struggle" (`/tmp/gemma4-wide-year-length.log`). This is consistent with
the existing four-gram loop-abort policy in `generate.c`, not an observed
token-ID divergence. AVX512's two longer prompts pass 32/32 each.

The complete AVX512 short gate is being rerun with the existing
`BN_DISABLE_LOOP_ABORT=1` diagnostic switch. This aligns fixed-length
generation with llama.cpp without changing EOS handling, kernels or
normal production stopping policy. It is pending, not accepted, until
the rerun completes. Log: `/tmp/q5-tile4-gemma4-avx512-noloop-128.log`.

That rerun passes **128/128 tokens**, confirming the short-run length
failure was resolved by disabling the repetition abort. With the two
passing 32-token longer prompts, Gemma4 AVX512 has **192/192 token parity**
under the documented comparison settings. Its current speed gate remains
unverified. The comparator now labels unequal-length completions as
partial and prints per-prompt/aggregate token counts; strict pass criteria
are unchanged. A synthetic unequal-count fixture reproduces the formerly
opaque report and verifies the new diagnostics
(`/tmp/parity-count-{red,green}.log`).

The reporting change passes the shell syntax check, warning-free clean
default build and full test suite (`/tmp/parity-count-build.log`,
`/tmp/parity-count-tests.log`). Inference code and acceptance thresholds
are unchanged by this reporting change.

The isolated Qwen3.5 dense AVX512 speed gate passes: **prefill 19.35/14.36
tokens/s (1.348)** and **decode 2.63/2.59 tokens/s (1.015)**. Bitnet
prefill samples are 19.35/19.08/19.49; decode samples are 2.63/2.63/2.63.
Together with 192/192 token parity, this is an accepted eight-thread
checkpoint for this model/backend. Speed uses 128 prompt tokens, 64 decode
tokens, three bitnet runs, default caches, and F32 bitnet KV versus F16
llama KV. Log: `/tmp/q5-tile4-q35-avx512-speed.log`.

The matching isolated AVX2 speed gate also passes: **prefill 13.02/13.20
tokens/s (0.986)** and **decode 2.63/2.59 tokens/s (1.015)**. Bitnet
prefill samples are 13.02/12.97/13.08; decode samples are 2.63/2.63/2.63.
With the previously completed 192/192 token gate, Qwen3.5 dense AVX2 is
also accepted at eight threads under the same speed settings. Log:
`/tmp/q5-tile4-q35-avx2-speed.log`. This does not establish acceptance for
Qwen3.5 sparse, CUDA, or other model families.

Gemma4 dense AVX512 subsequently passes its isolated speed gate: **prefill
30.88/35.19 tokens/s (0.877)** and **decode 2.64/2.77 tokens/s (0.953)**.
Bitnet samples are prefill 30.88/29.52/31.02 and decode 2.64/2.65/2.64.
Together with the documented 192/192 token matches, this is an accepted
eight-thread checkpoint under the comparison settings: `--maxseq 4096`,
`BN_DISABLE_LOOP_ABORT=1`, flash off, default caches, pp128/tg64, and F32
bitnet KV versus F16 llama KV for speed (F32 both for parity). Disabling
the repetition abort does not establish default stopping-policy parity.
Log: `/tmp/q5-tile4-gemma4-avx512-speed.log`.

The matching Gemma4 dense AVX2 speed gate also passes: **prefill 18.51/19.00
tokens/s (0.974)** and **decode 2.64/2.90 tokens/s (0.910)**. Bitnet
prefill samples are 19.13/18.51/18.10; decode samples are 2.64/2.65/2.64.
This completes its eight-thread checkpoint with the previously documented
192/192 token matches. Speed uses the same context cap, repetition-abort
override, cache, KV and pp128/tg64 settings as AVX512 above. Log:
`/tmp/q5-tile4-gemma4-avx2-speed.log`. Gemma4 sparse coverage remains
outstanding; the local model inventory still contains only Gemma4 dense
31B. These CPU results do not establish CUDA acceptance.

Measured binary SHA-256 identities for these post-Q5-tile CPU gates:

```text
bitnet AVX512 /tmp/q5-tile4-avx512
578bce71ad23a55af4fc541b1f6848e332333722ebd621d1339cc94cd0e1edb2
bitnet AVX2 /tmp/q38-q4route-avx2
55a15175f4f91c5702498f0d547e3cf2c9c0457833d35b43dae133468fa607cd
llama.cpp build-avx512/bin/llama-bench
7648d1cc733d692131b28113eaf0c4fe7d772a00cf236d8b63cca18624d1591c
llama.cpp build-avx2/bin/llama-bench
80c632b6713e7c8ba30094d2d7d4832cb45a1cca82cf06763cb214cbeb8d0832
```

The llama.cpp source checkout reports HEAD
`3d3d7c81813067fc8c185da017e0af03b4269b1e`; the binary hashes above identify
the measured artifacts independently of subsequent source changes.

### Qwen3.6 dense F32 decode projection localization

The current saved CPU binaries fail the Qwen3.6 dense 27B Q4_K_M short
gate: **AVX2 100/128**, **AVX512 92/128**. Both match all eight first
output IDs and generate equal lengths. Runs use eight threads,
`--maxseq 4096`, F32 KV on both engines and flash off. Logs:
`/tmp/q5-tile4-q36dense-avx{2,512}-128.log`. No speed acceptance is implied.

AVX512's story prompt first diverges after three matching output tokens.
One-thread traces show an earlier arithmetic difference at layer 0 on
the first decode step (position 8): the SSM alpha/beta F32 projections.
The complete normalized input is byte-identical (5120 floats):
`/tmp/q36dense-story-norm-{bn,ll}.bin`. Layer-0 alpha and beta weights are
F32, each 48x5120. Replaying the captured input against alpha weights
with bitnet's existing one-accumulator AVX512 dot differs from llama.cpp
on **40/48 outputs**, maximum 1.43051147e-6. A four-accumulator dot with
the reference merge order matches **all 48 outputs exactly**.

Replay source/log: `/tmp/q36dense_projection_replay.c`,
`/tmp/q36dense-projection-replay.log`; full reference alpha:
`/tmp/q36dense-story-alpha-ll.bin`. Text traces:
`/tmp/q36dense-story-bn.txt`, `/tmp/q36dense-story-ll.log`.
The local reference's `llamafile_sgemm` declines single-token work, while
its generic F32 dot uses four SIMD accumulators. Multi-token prefill
uses a separate matrix path. This identifies a decode projection-order
mismatch, not yet a verified whole-model fix. Production code is
unchanged; the baseline full suite passes
(`/tmp/q36dense-f32-baseline.log`).

The Qwen3.5 sparse 122B-A10B MXFP4 current AVX512 smoke passes 8/8 on the
capital prompt but still fails the known sky boundary: 1/8, `blue because`
versus `blue due`, equal output lengths. Logs:
`/tmp/q5-tile4-q35sparse-avx512-{smoke,sky}.log`. These use eight threads,
`--maxseq 512`, F32 KV both and flash off. Sparse parity remains open.

### F32 x86 decode accumulation alignment

`src/quant/f32_avx2.c` now uses four independent SIMD accumulators for
single-token F32 dot products, with the reference merge and horizontal
reduction order on AVX2 and AVX512. Multi-token matrix accumulation is
unchanged. The scalar tail uses explicit `fmaf` to prevent GCC from
separating multiplication and addition where Clang fused them. This is
format-local: no model-family conditions, new API, backend ownership
changes, or ARM/Metal modifications were introduced.

The new exact regression covers 137 rows, unaligned inputs, column
boundaries 1/7/8/15/16/31/32/33/63/64/65/127/5120/5123, partial row ranges,
single-token matmul, batched matvec dispatch, and serial/three-worker
execution, with output guards. It fails before the change and passes
under Clang AVX2, Clang native AVX512, and GCC native after making the
tail explicitly fused. The existing multi-token prefill-order regression
is retained. Logs: `/tmp/f32-dot-{red,native-tests,avx2-tests}.log` and
`/tmp/f32-dot-gcc-{diagnose,green}.log`. ASan/UBSan passes with leak
detection disabled (`/tmp/f32-dot-sanitize.log`). The production path's
48 alpha outputs now match the captured reference exactly:
`/tmp/f32-dot-q36dense-alpha-bn.bin`.

Qwen3.6 dense's eight-thread short gates improve from AVX2 100/128 and
AVX512 92/128 to **128/128 on each**. Qwen3 sparse AVX512 also passes
128/128. These runs use `--maxseq 4096`, F32 KV both and flash off. Logs:
`/tmp/f32-dot-q36dense-avx{2,512}-128.log`,
`/tmp/f32-dot-q3sparse-avx512-128.log`.

Both longer Qwen3.6 prompts subsequently pass 32/32 on each backend,
completing **192/192 tokens per backend** under the same comparison
settings. Logs:
`/tmp/f32-dot-q36dense-avx{2,512}-long-{capital,story}.log`. Current-model
speed acceptance remains unverified. The final clean default build is
warning-free and the full test suite passes:
`/tmp/f32-dot-final-{build,tests}.log`.

Final Clang binaries are byte-identical before/after the explicit-tail
adjustment, so the completed short gates cover these exact artifacts:

```text
/tmp/f32-dot-final-avx2
9668846eb7660024b5b559e8f38395f05a31053a0dfe34ae980221d67d155a85
/tmp/f32-dot-final-avx512
a98393b820cad953b06a081392765ec6a567b741fa9dae4e431c29e9c680ef85
```

The isolated Qwen3.6 dense AVX512 speed gate subsequently passes:
**prefill 36.57/30.82 tokens/s (1.186)** and **decode 3.04/2.90 tokens/s
(1.048)**. Bitnet prefill samples are 36.57/36.67/36.52; decode samples
are 3.08/3.04/3.04. Combined with 192/192 token parity, this is an
accepted eight-thread checkpoint for this model/backend. Speed uses
pp128/tg64, three bitnet runs, default caches, flash off and F32 bitnet
KV versus F16 llama KV. Log: `/tmp/f32-dot-q36dense-avx512-speed.log`.
The matching isolated AVX2 speed gate also passes: **prefill 22.79/20.12
tokens/s (1.133)** and **decode 3.04/2.91 tokens/s (1.045)**. Bitnet
prefill samples are 22.84/22.35/22.79; decode samples are 3.05/3.04/3.03.
This completes the Qwen3.6 dense eight-thread AVX2 checkpoint with
192/192 token parity and the same speed settings as above. Log:
`/tmp/f32-dot-q36dense-avx2-speed.log`. Sparse-model and CUDA acceptance
are not implied by either dense CPU result.

### Sparse refresh after the F32 decode fix

Current Qwen3.5 122B-A10B MXFP4 sky-prompt checks pass **8/8 on AVX2**
but still fail **1/8 on AVX512**, with `blue because` versus `blue due`.
The expanded AVX2 gate is **112/128**, not accepted: the fox prompt has
only 2/16 token-ID prefix matches despite matching rendered words, and
the sum prompt has 14/16. Qwen3.8 Flash-Next AVX512 still fails its year
prompt on the first token (`was`, ID 557, versus `commemorated`, ID
77296), **0/8**. All runs use the current F32-fix binaries, eight
threads, `--maxseq 512`, F32 KV both and flash off. Logs:
`/tmp/f32-dot-q35sky-avx{2,512}.log`,
`/tmp/f32-dot-q35sparse-avx2-128.log`,
`/tmp/f32-dot-q38year-avx512.log`.

Qwen3.5 sky-prompt one-thread traces localize an earlier difference to
layer-0 MoE output on the first decode step (position 6). The complete
normalized expert input matches byte-for-byte (3072 floats), as do the
eight selected expert IDs and routing weights. The first selected expert
is 141. Its MXFP4 gate projection matches all 1024 reference outputs
using the existing scalar kernel. A two-accumulator/eight-lane replay
instead differs on 892/1024, maximum 7.45058e-8, so no MXFP4 kernel
change is justified by this capture. The displayed shared-expert
gate/up, activation, down projection and scalar gate also match in replay.

The decisive discrepancy is the expert **down** projection, which is
Q5_K despite the fixture's MXFP4 name. On the same activation,
single-matvec dispatch matches **all 3072 reference outputs exactly**,
but multi-input dispatch differs on **3072/3072**, maximum 9.57515e-6.
`matvec_multi.c` selects the old float-input Q5_K four-row kernel, while
single dispatch quantizes activations to Q8_K and uses the integer-dot
kernel. This establishes a multi-input route inconsistency; a corrected
implementation and whole-model re-gates are still pending.

Evidence: `/tmp/q35sky-current-{bn.txt,ll.log}`,
`/tmp/q35sky-norm-{bn,ll}.bin`, `/tmp/q35sky-gate-ll.bin`,
`/tmp/q35sky-down-ll.bin`, `/tmp/q35_mxfp4_replay.c`,
`/tmp/q35-mxfp4-replay.log`, `/tmp/q35_shared_replay.c`, and
`/tmp/q35-shared-replay.log`. No production inference code changed during
this localization.

### Q5_K multi-input activation routing

The x86 Q5_K multi-input path now quantizes each task's activation to
Q8_K and dispatches the existing integer-dot kernel, matching single
matvec. It retains a single thread-pool dispatch and forwards prepared
weight state. Explicit AVX2 float-dot diagnostics retain their existing
fallback; native AVX512 retains the integer route used by single matvec.
Mixed column counts use per-task fallback instead of a shared stride.
Only `src/quant/matvec_multi.c` and quant tests changed: no model-family
branches, ownership changes, new APIs, or ARM/Metal arithmetic changes.

The exact regression uses independent synthetic Q5_K matrices and inputs,
133–137 rows, three superblocks, task counts 1/2/8/24/25, serial and
three-worker dispatch, untouched-output guards and mixed column counts.
It fails on the old float-input route and passes under Clang AVX2 and
native AVX512. Logs: `/tmp/q5multi-red.log`,
`/tmp/q5multi-final-{avx2,native}-tests.log`. The corrected captured
expert replay matches all 3072 reference down-projection outputs through
both single and multi-input dispatch (`/tmp/q5multi-replay.log`).

The first complete Qwen3.5 sparse 122B-A10B run passes 128/128 short
tokens plus two 32/32 longer prompts on each backend, versus the prior
AVX2 112/128 short failure and AVX512 sky-prompt 1/8 failure. Logs:
`/tmp/q5multi-q35sparse-avx{2,512}-128.log`,
`/tmp/q5multi-q35sparse-avx{2,512}-long-{capital,story}.log`.
These checks use eight threads, `--maxseq 512`, F32 KV both and flash off.
Final-artifact reruns after the mixed-width safeguard also pass:
**192/192 tokens on AVX2 and AVX512**, with equal generation lengths.
Logs: `/tmp/q5multi-final-q35sparse-avx{2,512}-128.log` and
`/tmp/q5multi-final-q35sparse-avx{2,512}-long-{capital,story}.log`.
The final warning-free clean default build and full suite pass
(`/tmp/q5multi-final-{build,tests}.log`), as does final ASan/UBSan with
leak detection disabled (`/tmp/q5multi-final-sanitize.log`). Speed
remains unverified; this is not yet full sparse-model acceptance.

Final artifact SHA-256 identities:

```text
/tmp/q5multi-final-avx2
ce35ee7569e7897625511ab7a7929b38edbb1dfe8c1a5319c5bed5501489b476
/tmp/q5multi-final-avx512
c7f4f942f33f2be1deb320dff36bf5dcbf3102ae3ddc22a6d14589513f101c54
```

The Qwen3.5 sparse AVX512 isolated speed gate fails despite token parity:
**prefill 14.11/39.76 tokens/s (0.355)** and **decode 4.30/5.56 tokens/s
(0.773)**. Bitnet prefill samples are 14.11/14.13/14.03; decode samples
are 4.30/4.30/4.30. Settings are eight threads, default caches,
pp128/tg64, three bitnet runs, flash off and F32 bitnet KV versus F16
llama KV. Log: `/tmp/q5multi-q35sparse-avx512-speed.log`. This checkpoint
is not speed-accepted.

A subsequent isolated diagnostic with `BN_PREFILL_PROFILE=1` and 128
prompt token IDs of 1 reports 9467.1 ms prefill. MoE total is 8034.2 ms,
including routed gate/up 6638.2 ms, down 602.5 ms, shared experts
128.0 ms, routing 44.6 ms and weighted accumulation 39.6 ms. Gate/up
therefore accounts for about 70% of prefill wall time. The fixture uses
MXFP4 routed gate/up weights, making that format's projection path the
next optimization target; the Q5_K down correctness fix is retained.
This single instrumented run is diagnostic, not a speed acceptance
measurement. Log: `/tmp/q5multi-q35sparse-profile.log`. AVX2 speed has
not yet been measured on this checkpoint.

### MXFP4 x86 integer-block SIMD

The x86 MXFP4 kernel now unpacks nibbles with SIMD lookup, widens both
weights and activations to signed 16-bit values, and reduces the block's
32 integer products before scaling. The scalar FP32 accumulation order
between blocks is unchanged. Widening also handles activation -128
without signed-byte negation overflow. The new format-local file is
`src/quant/mxfp4_avx2.c`; x86 dispatch selects it, while scalar and NEON
implementations remain unchanged. No model-specific routing was added.

Exact scalar-versus-SIMD tests cover all exponent-byte values, all int8
activation values, 137 rows, 1/2/3/7/96/129 blocks, unaligned input,
partial row ranges, output guards, and serial/three-worker dispatch.
Clang AVX2/native tests, the warning-free clean default build, the full
suite, and ASan/UBSan (leak detection disabled) pass. Logs:
`/tmp/mxfp4-simd-{native-tests,avx2-tests,build,tests,sanitize}.log`.

An isolated single-thread 1024x3072 warm projection replay, 200 iterations
per trial, measures scalar 0.583/0.592/0.583 ms versus SIMD
0.168/0.167/0.167 ms (**3.48–3.55x**), with exact outputs. Source/log:
`/tmp/mxfp4_simd_bench.c`, `/tmp/mxfp4-simd-bench.log`. This kernel
measurement does not establish whole-model speed acceptance.

The new binaries retain **192/192 Qwen3.5 sparse tokens on both AVX2 and
AVX512**, eight short prompts plus two longer prompts, eight threads,
`--maxseq 512`, F32 KV both and flash off. Logs:
`/tmp/mxfp4-simd-q35sparse-avx{2,512}-128.log` and
`/tmp/mxfp4-simd-q35sparse-avx{2,512}-long-{capital,story}.log`.

The isolated AVX512 whole-model gate improves but still fails: **prefill
26.03/40.56 tokens/s (0.642)** and **decode 4.66/5.78 tokens/s (0.806)**.
Bitnet samples are prefill 26.03/25.87/26.41 and decode 4.67/4.65/4.66.
The preceding scalar checkpoint measured 14.11 prefill and 4.30 decode.
Settings remain eight threads, default caches, pp128/tg64, three bitnet
runs, flash off, F32 bitnet KV versus F16 llama KV. Log:
`/tmp/mxfp4-simd-q35sparse-avx512-speed.log`. Speed acceptance is not
established; the exact SIMD improvement is retained.

A fresh isolated diagnostic prefill profile reports 5180.9 ms total,
with routed gate/up reduced from 6638.2 to **2271.9 ms**. Down projections
take 640.2 ms and shared experts 128.9 ms; total MoE is 3716.6 ms.
Gate/up remains substantial, but is no longer 70% of prefill wall time.
Log: `/tmp/mxfp4-simd-q35sparse-profile.log`. These instrumented samples
are diagnostic, not acceptance measurements.

Measured artifact SHA-256 identities:

```text
/tmp/mxfp4-simd-avx2
4b4c3ac688eafcf5a8c584065df40bf2a9ebda8acab3c48d23e0f96a83411701
/tmp/mxfp4-simd-avx512
ee3e0cbf2cb694fd0ebb7289d6b9562d84ad2e9b62a232cd3ee38fbfa9e427e4
```

### MXFP4 x86 batched weight reuse

The format-local MXFP4 kernel now supports four-token tiles and explicit
one/two/three-token tails. It reuses unpacked weights while retaining the
scalar block-by-block FP32 sum independently for each token. Quant dispatch
owns activation quantization and allocation fallback. MXFP4 declares the
existing `BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL` capability so the existing
MoE GEMV-order batching route can use this kernel. No model-family branches,
model/backend ownership changes, or ARM/Metal changes are involved.

The exact regression covers 1/2/3/4/5/7/8/9/15/16/17 tokens, 137 rows,
all exponent bytes and signed int8 activation values (including -128),
unaligned input, partial row ranges, output guards, scalar-oracle comparisons,
public matmul and GEMV-order multi-matrix routing, and serial/three-worker
dispatch. Clang AVX2/native tests and ASan/UBSan with leak detection disabled
pass (`/tmp/mxfp4-batch-{avx2,native}-final-tests.log`,
`/tmp/mxfp4-batch-sanitize.log`).

An isolated 1024x3072 single-thread warm kernel benchmark reports
1.316–1.319x speedup for two tokens, 1.475–1.499x for four, 1.498–1.501x
for eight, and 1.459–1.460x for seventeen, with exact outputs. Source/log:
`/tmp/mxfp4_batch_bench.c`, `/tmp/mxfp4-batch-bench.log`. These are kernel
measurements, not whole-model acceptance results.

The warning-free clean default build and full test suite also pass
(`/tmp/mxfp4-batch-{clean,build,tests}.log`). Candidate artifact identities:

```text
/tmp/mxfp4-batch-avx2
cd25daba9a2722bdfbdd02c32eb3d1ee7a264641890170b96bdc23068b3e6dd6
/tmp/mxfp4-batch-avx512
7cf58e271f7bb74a6003cf4f0838d8ffbbe8c0d538bd0d6617bbcc268f5ed826
```

Whole-model token gates pass **192/192 on AVX2 and AVX512**: eight
16-token short prompts and two 32-token longer prompts, with equal
generation lengths. Settings are eight threads, context 512, F32 KV both,
flash off, and default stopping. Logs:
`/tmp/mxfp4-batch-q35sparse-avx{2,512}-{128,long}.log`.

The isolated AVX512 speed gate remains below acceptance: **prefill
30.72/40.10 tokens/s (0.766)** and **decode 4.67/5.61 tokens/s (0.832)**.
Bitnet samples are prefill 30.36/30.77/30.72 and decode 4.67/4.67/4.67.
The prior single-token SIMD checkpoint measured 26.03 prefill and 4.66
decode. Settings remain eight threads, default caches, pp128/tg64, three
bitnet runs, flash off, F32 bitnet KV versus F16 llama KV. Log:
`/tmp/mxfp4-batch-q35sparse-avx512-speed.log`. The batched optimization
is retained, but this is not sparse-model speed acceptance. AVX2 speed
has not yet been measured on this artifact.

A fresh isolated diagnostic with 128 prompt IDs of 1 reports 4499.0 ms
prefill and 2997.8 ms total MoE. Routed gate/up takes **1503.3 ms**
(previous single-token SIMD profile: 2271.9 ms), down 679.8 ms, shared
experts 134.5 ms, routing 48.2 ms, and weighted accumulation 38.7 ms.
Log: `/tmp/mxfp4-batch-q35sparse-profile.log`. Gate/up remains substantial,
but the remaining prefill time also needs profiling; the gate/up kernel
alone no longer explains the full gap. This instrumented run is diagnostic,
not a speed acceptance measurement.

### MXFP4 sparse AVX512 thread scaling

Isolated matching-thread gates on the unchanged batched MXFP4 artifact
show that additional threads improve absolute throughput, but do not
close the gap to llama.cpp. All use default caches, pp128/tg64, three
bitnet runs, flash off, F32 bitnet KV and F16 llama KV.

| Threads | Bitnet prefill | llama prefill | Ratio | Bitnet decode | llama decode | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 30.72 | 40.10 | 0.766 | 4.67 | 5.61 | 0.832 |
| 16 | 41.55 | 68.70 | 0.605 | 6.85 | 8.91 | 0.769 |
| 32 | 49.42 | 111.37 | 0.444 | 8.02 | 9.89 | 0.811 |

Throughputs are tokens/s. At sixteen threads, bitnet prefill samples are
42.11/40.82/41.55 and decode 6.87/6.84/6.85. At thirty-two threads,
prefill samples are 49.42/49.58/49.08 and decode 8.02/8.01/8.02.
Logs: `/tmp/mxfp4-batch-q35sparse-avx512-t{16,32}-speed.log`;
eight-thread baseline: `/tmp/mxfp4-batch-q35sparse-avx512-speed.log`.
Both new gates fail the 0.85 minimum. Token parity was established at
eight threads, not re-gated at sixteen/thirty-two. No new runtime default
is selected from these measurements.

A separate isolated 32-thread diagnostic with 128 prompt IDs of 1 takes
2781.8 ms prefill, including 2093.6 ms total MoE, gate/up 733.9 ms, down
565.2 ms, and shared experts 64.6 ms. The comparable eight-thread profile
reported 4499.0/2997.8/1503.3/679.8/134.5 ms respectively. Routed down
projections thus scale much less than gate/up in these diagnostic samples;
total MoE also includes time not covered by its individual projection
timers. Log: `/tmp/mxfp4-batch-q35sparse-t32-profile.log`. These profiles
do not replace the repeated uninstrumented gates above.

Code inspection identifies another candidate for measured optimization:
x86 MXFP4 `bn_quant_matvec_batch` lacks a shared-quantization/multi-task
dispatch route and falls back to individual matvec calls. This affects
batched routed expert projections and singleton-expert prefill groups.
It is a follow-up hypothesis, not a measured attribution of the scaling
gap; no dispatch implementation changed in this thread-scaling check.

### MXFP4 x86 multi-matrix dispatch

`bn_quant_matvec_batch` now quantizes a common MXFP4 input once and
dispatches the existing exact SIMD row kernels for all matrices together.
This x86-only quant-layer route handles up to 24 equal-width matrices;
unequal widths, larger batches, and forced-float requests retain their
existing fallback behavior. No dot-product arithmetic, model-family policy,
backend ownership, or ARM/Metal code changes.

Exact tests cover 0/1/2/8/16/24/25 tasks with independent weights, varying
row counts, all exponent bytes, unaligned input/scratch, output and scratch
guards, unequal-width fallback, forced-float fallback, and serial/three-worker
dispatch. Clang AVX2/native tests, the warning-free clean default build,
the full suite, and ASan/UBSan with leak detection disabled pass. Logs:
`/tmp/mxfp4-dispatch-{avx2,native}-tests.log`,
`/tmp/mxfp4-dispatch-{clean,build,tests,sanitize}.log`.

Candidate artifact identities:

```text
/tmp/mxfp4-dispatch-avx2
719cd01ddf9fc8cc31e0ae29847a67a7991ef6be4ab7cd95d941adc095350e1f
/tmp/mxfp4-dispatch-avx512
21c312d8a4065e440717de5e6d31e0e900541281f1c9a720af23cb5861454d03
```

Whole-model gates retain **192/192 token IDs on both AVX2 and AVX512**,
eight short prompts plus two longer prompts, with equal generation
lengths. Settings: eight threads, context 512, F32 KV both, flash off,
default stopping. Logs:
`/tmp/mxfp4-dispatch-q35sparse-avx{2,512}-{128,long}.log`.

The isolated eight-thread AVX512 gate remains below acceptance: **prefill
30.90/40.70 tokens/s (0.759)** and **decode 4.72/5.61 tokens/s (0.841)**.
Bitnet samples are prefill 30.88/31.09/30.90 and decode 4.75/4.72/4.72.
The prior checkpoint measured 30.72 prefill and 4.67 decode; the measured
change is small and is not evidence of a substantial whole-model speedup.
Settings: default caches, pp128/tg64, three bitnet runs, flash off, F32
bitnet KV versus F16 llama KV. Log:
`/tmp/mxfp4-dispatch-q35sparse-avx512-speed.log`.

At thirty-two threads the isolated gate measures **prefill 48.20/111.22
tokens/s (0.433)** and **decode 8.11/9.23 tokens/s (0.879)**. Bitnet samples
are prefill 49.46/48.20/47.63 and decode 8.11/8.16/8.09. The decode ratio
passes this measurement, but the preceding artifact measured 8.02/9.89;
the reference's lower rate accounts for much of the ratio change. Prefill
remains a clear failure, and its samples span a wider range than the
previous run. This does not establish whole-model speed acceptance or
32-thread token parity. Log:
`/tmp/mxfp4-dispatch-q35sparse-avx512-t32-speed.log`.

An isolated alternating old/new 32-thread pp128 check gives old prefill
times 2738.0/2557.1/2614.0 ms and new times 2646.3/2611.4/2630.5 ms.
The paired direction is inconsistent; medians differ by only 0.6%.
This does not establish a repeatable prefill regression or a meaningful
prefill speedup. The exact shared-quantization dispatch is retained, but
the remaining speed work must target larger costs. Logs:
`/tmp/mxfp4-dispatch-ab-{batch,dispatch}-{1,2,3}.log`.

The next candidate is Q5_K down-projection batching: the MoE GEMV-order
route falls back to per-token dispatch for this format, while an existing
batched integer kernel already has exact direct-kernel tests. Public-route
equivalence, prepared-weight behavior, and explicit float-policy fallback
must be checked before extending the capability declaration/routing.

### Q5_K GEMV-order prefill batching

The quant registry now declares Q5_K's existing x86 integer matmul as
GEMV-order preserving under the default runtime policy. This lets the
existing MoE GEMV-order route use its four-token native AVX512 kernel
(and one-dispatch AVX2 implementation) for down projections. The dispatcher
retains per-token fallback for explicit float-input or batch-disable
policies. No new kernel, model-family branch, backend ownership change,
or ARM/Metal execution change is introduced.

Public-route exact tests cover one/two independent matrices, 136/137 rows,
1/2/3/4/5/8/17/25/33 tokens, unaligned input, output guards, prepared Q5_K
x8 weights and null preparation, default/float/batch-disable policies,
and zero/three background workers. The oracle is the existing zero-flag
batched matvec route, not merely the single kernel. Clang AVX2/native
tests, warning-free clean default build, full suite, and ASan/UBSan with
leak detection disabled pass. Logs:
`/tmp/q5-gemv-batch-{avx2,native}-final-tests.log`,
`/tmp/q5-gemv-batch-{clean,build,tests,sanitize}.log`.

Candidate artifact identities:

```text
/tmp/q5-gemv-batch-avx2
0cafb564bcf7d12450672fdbb823e04bdb641fbb10aac430021a741a49307994
/tmp/q5-gemv-batch-avx512
87f45e41c9cf7f93f4b9d7d470a03bdaf7abc6efe92d2d175d9fb449f092245b
```

Whole-model gates retain **192/192 tokens on both AVX2 and AVX512**,
with equal generation lengths across eight short prompts and two longer
prompts. Settings: eight threads, context 512, F32 KV both, flash off,
default stopping. Logs:
`/tmp/q5-gemv-batch-q35sparse-avx{2,512}-{128,long}.log`.

The isolated eight-thread AVX512 gate improves prefill but remains below
acceptance: **prefill 32.94/40.46 tokens/s (0.814)** and **decode
4.68/5.54 tokens/s (0.845)**. Bitnet samples are prefill 32.94/33.37/32.61
and decode 4.68/4.68/4.68. The preceding dispatch checkpoint measured
30.90 prefill and 4.72 decode. Settings remain default caches, pp128/tg64,
three bitnet runs, flash off, F32 bitnet KV versus F16 llama KV. Log:
`/tmp/q5-gemv-batch-q35sparse-avx512-speed.log`. This is not full
sparse-model speed acceptance; AVX2 speed remains unmeasured here.

An isolated eight-thread diagnostic with 128 prompt IDs of 1 reports
4316.1 ms prefill, total MoE 2747.1 ms, routed gate/up 1498.8 ms, down
398.9 ms, and shared experts 136.6 ms. The previous batched MXFP4 profile
reported down 679.8 ms; the other large measured projections are similar.
Log: `/tmp/q5-gemv-batch-q35sparse-profile.log`. This instrumented run
is diagnostic, not a speed acceptance measurement.

Code inspection identifies potentially wasted Q5_K preparation as a
follow-up: `bn_quant_matvec_uses_prepared_weight` returns true for the
default x86 Q5_K route when an x8 layout can be constructed, but the
canonical single and batched kernels used by this route read the original
weights and do not consume the packed auxiliary buffer. MoE acquisition
can therefore prepare/cache unused weights. This policy should reflect
the selected kernel's actual needs; its cost has not yet been isolated.

### Avoid unused x86 Q5_K preparation

The quant preparation policy now returns false for public x86 Q5_K
dispatch: both its canonical integer kernels and explicit float-input
diagnostic read original weights. This prevents MoE execution and prefill
from acquiring/building unused x8 layouts. Explicit x8 preparation and
kernel APIs remain available; no model-family or backend ownership changes
are involved, and ARM/Metal execution is unchanged.

The added policy regression fails on the previous implementation
(`/tmp/q5-no-unused-pack-red.log`) and passes after the fix. It covers
null/zero-worker/three-worker pools, default/float/batch-disable policies,
and default/reference/native/forced-float task flags. Existing public-route
tests still construct an x8 layout explicitly and check exact outputs
with and without it. Clang AVX2/native tests, the warning-free clean
default build, full suite, and ASan/UBSan with leak detection disabled
pass. Logs: `/tmp/q5-no-unused-pack-{avx2,native}-tests.log`,
`/tmp/q5-no-unused-pack-{clean,build,tests,sanitize}.log`.

Candidate artifact identities:

```text
/tmp/q5-no-unused-pack-avx2
b96a52588393bd34f5e6b43199d8e058da8ed1732aa715759a55830b2ea83df2
/tmp/q5-no-unused-pack-avx512
ebf03516eec49af17cdb1b0df7d0bd821defe8bec5674b9f3d4e0e5227abfebd
```

Whole-model gates retain **192/192 tokens on each of AVX2 and AVX512**,
with equal generation lengths. Settings: eight short 16-token prompts,
two longer 32-token prompts, eight threads, context 512, F32 KV both,
flash off, default stopping. Logs:
`/tmp/q5-no-unused-pack-q35sparse-avx{2,512}-{128,long}.log`.

The isolated eight-thread AVX512 gate now **passes**: prefill
**37.16/40.48 tokens/s (0.918)** and decode **5.04/5.58 tokens/s (0.903)**.
Bitnet samples are prefill 36.83/37.69/37.16 and decode 5.02/5.04/5.04.
The preceding Q5_K batching checkpoint measured 32.94 prefill and 4.68
decode. Settings: default caches, context 512, pp128/tg64, three bitnet
runs, flash off, F32 bitnet KV versus F16 llama KV. Log:
`/tmp/q5-no-unused-pack-q35sparse-avx512-speed.log`. Together with the
192/192 token-ID gate, this accepts the Qwen3.5 sparse 122B-A10B MXFP4
fixture on AVX512 at this artifact/configuration, not the remaining model
or backend matrix.

The subsequent isolated AVX2 gate also **passes**: prefill
**32.27/35.86 tokens/s (0.900)** and decode **5.30/5.72 tokens/s (0.927)**.
Bitnet samples are prefill 31.71/32.27/32.59 and decode 5.26/5.30/5.30.
Settings match the eight-thread AVX512 gate above, using the AVX2
executables on both sides. Log:
`/tmp/q5-no-unused-pack-q35sparse-avx2-speed.log`. With 192/192 token
parity, this accepts the same fixture on AVX2 at this checkpoint too.

A fresh isolated eight-thread diagnostic reports 3779.9 ms prefill and
2257.6 ms total MoE, versus 4316.1/2747.1 ms before the policy fix. The
individual projection timings remain broadly similar (gate/up 1501.7 ms,
down 447.1 ms, shared 136.8 ms). MoE time outside the listed component
timers drops from about 590 to 55 ms, consistent with removing the
preparation work performed outside those timers. Reported RSS is essentially
unchanged (15.14 versus 15.13 GB); this is a compute/preparation saving,
not a demonstrated resident-memory reduction. Log:
`/tmp/q5-no-unused-pack-q35sparse-profile.log`. This profile is diagnostic,
not a replacement for the repeated speed gates.

### Qwen3 dense current-artifact refresh

The `q5-no-unused-pack` AVX2/AVX512 artifacts above retain **192/192
token IDs on each backend** for Qwen3-4B-Q4_K_M: eight 16-token short
prompts plus two 32-token longer prompts, equal generation lengths,
eight threads, context 4096, F32 KV both, flash off, default stopping.
Logs: `/tmp/q3dense-current-avx{2,512}-{128,long}.log`.

Sequential isolated three-run speed gates at eight threads, default
caches, context 512, pp128/tg64, flash off, F32 bitnet KV versus F16 llama
KV both pass:

| Backend | Bitnet prefill / llama | Ratio | Bitnet decode / llama | Ratio |
| --- | ---: | ---: | ---: | ---: |
| AVX2 | 145.52 / 149.70 tok/s | 0.972 | 19.11 / 21.83 tok/s | 0.875 |
| AVX512 | 215.13 / 239.36 tok/s | 0.899 | 19.29 / 21.69 tok/s | 0.889 |

AVX2 samples are prefill 145.52/144.27/146.29, decode 19.23/19.03/19.11.
AVX512 samples are prefill 186.21/216.69/215.13, decode 19.07/19.69/19.29.
Logs: `/tmp/q3dense-current-avx{2,512}-speed.log`. The old AVX2 failure
measured bitnet 19.13 versus llama 22.94 decode; the current ratio improves
without a material increase in bitnet's absolute decode rate. Reference
variation therefore matters, and this refresh is not a new decode
optimization claim. No production code changed during these measurements.

An isolated five-run AVX2 repeat also passes: **prefill 145.89/143.33
tokens/s (1.018)** and **decode 18.93/20.57 tokens/s (0.920)**. Bitnet
prefill samples are 147.99/145.89/144.10/144.88/151.09; decode samples
19.30/18.92/18.93/18.93/18.93. Log:
`/tmp/q3dense-current-avx2-speed5.log`. Both current gates pass, but the
reference-rate variation remains visible; do not describe this as a
material bitnet decode acceleration versus the historical failure.

The corresponding five-run AVX512 repeat **fails prefill narrowly**:
208.57/246.24 tokens/s (0.847); decode passes at 19.08/20.62 (0.925).
Bitnet prefill samples are 217.28/186.75/208.57/184.49/214.55, decode
19.10/19.08/19.08/19.55/19.06. Log:
`/tmp/q3dense-current-avx512-speed5.log`. The earlier three-run pass
does not resolve this repeat failure. The substantial prefill spread
needs investigation before calling the current AVX512 speed gate stable.

An affinity-controlled five-run diagnostic restricts both engines through
the parent comparator's `taskset -c 0,8,16,24,32,40,48,56`. Local `lscpu`
maps these to eight physical cores, one in each L3 domain, without SMT
siblings. Prefill becomes steadier: bitnet 217.21 versus llama 217.72
tokens/s (0.998), samples 218.84/219.44/217.21/215.52/214.91. Decode
rises substantially in both engines: bitnet 36.83 versus llama 43.53
(0.846), samples 36.86/36.82/36.86/36.83/36.74. This still narrowly
fails decode. Log: `/tmp/q3dense-current-avx512-spread-speed5.log`.
This demonstrates material placement sensitivity, not a kernel speedup
or unrestricted-affinity acceptance. No runtime default was changed.

A second matched CPU-set diagnostic uses
`taskset -c 0,1,16,17,32,33,48,49` (two physical cores in each of four
L3 domains). It also fails: prefill 192.80/222.52 tokens/s (0.866), decode
36.01/43.30 (0.832). Bitnet prefill samples are
215.09/192.80/135.05/220.01/152.20; decode samples
35.98/36.01/35.99/36.04/36.01. Log:
`/tmp/q3dense-current-avx512-fourccd-speed5.log`. Restricting the allowed
CPU set does not pin individual workers to specific cores and has not
eliminated the prefill variability. Neither affinity diagnostic establishes
an accepted configuration or affinity-specific token parity. Further
AVX512 performance work is required; do not replace the failed repeat
with the earlier passing sample.

### Q6_K AVX512 row-pair prefill

The `q6-rowpair` candidate adds a format-local two-row/four-token raw
Q6_K matmul path. Each 256-bit half of the AVX512 accumulator retains one
row's existing eight lanes, preserving block FMA and final reduction order.
Expanded layouts and irregular row/token counts retain the prior path.
There are no model-family branches, public API changes, or ARM/Metal changes.

Artifacts:

- `/tmp/q6-rowpair-avx512`, SHA256
  `7e9074f079f75e519023e0ad82943eef525029dad3276a9db844360b152cafd6`.
- `/tmp/q6-rowpair-avx2`, SHA256
  `b96a52588393bd34f5e6b43199d8e058da8ed1732aa715759a55830b2ea83df2`,
  byte-identical to the preceding `q5-no-unused-pack` AVX2 artifact.

Exact kernel tests cover 4/136/137 rows, 1/3/38 blocks, thirteen token
counts from 1 through 33, full signed scale/input byte ranges, unaligned
input, partial ranges, output guards, and serial/three-worker execution.
The oracle is the existing single-token four-row kernel. Existing prepared
layout and public-dispatch tests also pass. Native and AVX2 quant suites,
ASan/UBSan quant checks, a clean warning-free default build, and the full
test suite pass. Logs: `/tmp/q6-rowpair-{native,avx2}-final-tests.log`,
`/tmp/q6-rowpair-{clean,build,tests,sanitize}.log`.

AVX512 model checks retain **192/192 Qwen3 dense token IDs**, and short
regressions retain **128/128 each for Qwen3.6 and Qwen3.8 dense**. Settings:
eight threads, context 4096, F32 KV both, flash off, default stopping.
Logs: `/tmp/q6-rowpair-q3dense-{128,long}.log`,
`/tmp/q6-rowpair-q{36,38}dense-128.log`. These regressions do not refresh
the entire model/backend matrix.

An isolated single-thread microbenchmark uses the actual Qwen3-4B layer-0
Q6_K down projection (2560 rows, 9728 columns), 128 synthetic quantized
inputs, and exact output comparison. Three old/new timings are
41.737/33.809, 41.967/33.741, and 41.915/33.578 ms: approximately
1.24–1.25x faster for this kernel. Log: `/tmp/q6-row2-tile4-bench.log`.
This is not a whole-model speed claim.

Two sequential isolated five-run Qwen3 dense AVX512 speed gates pass with
eight threads, unrestricted affinity, default caches, context 512,
pp128/tg64, flash off, F32 bitnet KV versus F16 llama KV:

| Gate | Bitnet prefill / llama | Ratio | Bitnet decode / llama | Ratio |
| --- | ---: | ---: | ---: | ---: |
| First | 222.45 / 247.63 tok/s | 0.898 | 19.06 / 20.62 tok/s | 0.924 |
| Repeat | 221.99 / 240.28 tok/s | 0.924 | 19.05 / 20.39 tok/s | 0.934 |

First prefill samples: 221.84/222.65/222.45/221.99/223.46; repeat:
221.34/220.46/221.99/223.19/223.93. First decode samples:
19.06/19.30/19.04/19.06/19.03; repeat: 19.01/19.64/19.28/19.05/19.02.
Logs: `/tmp/q6-rowpair-q3dense-speed5.log` and
`/tmp/q6-rowpair-q3dense-speed5-repeat.log`. Compared with the preceding
artifact's failed five-run prefill gate, these two groups have higher,
tighter bitnet prefill rates. Decode is essentially unchanged. These
measurements accept this fixture/configuration at the 85% threshold;
they do not guarantee that placement variability is eliminated generally.

### Qwen3.8 Flash-Next Q5_1 localization (diagnostic only)

The current `q6-rowpair-avx512` artifact still fails the eight-token year
prompt immediately: bitnet token 557 (`was`) versus llama token 77296
(`commemorated`), 0/8 matching IDs. Settings: eight threads, context 512,
F32 KV both, flash off, default stopping. Log:
`/tmp/q6-rowpair-q38year.log`. The observer probe independently reproduces
llama's first token, with top logits 17.1869125 for 77296 and 17.0956936
for 557. Current layer traces:
`/tmp/q38year-current-{bn,ll}.trace`.

This fixture's expert gate/up weights are Q4_K, but its expert down
weights are **Q5_1**, with 640 input columns and 2560 output rows.
Bitnet's Q5_1 route currently uses float activations; llama's x86 dot uses
Q8_1 activations, including FP16-rounded scale and offset-sum metadata.
A replay of layer-0 expert 369 with the **same captured llama activation**
finds 2560/2560 differing output values, maximum absolute error
0.000666329, RMS 0.000190539. Log: `/tmp/q38year-q51-replay.log`;
diagnostic source: `/tmp/q38year-q51-replay.c`.

A temporary `/tmp/q51-oracle-avx512` binary substitutes only the Q5_1
scalar-range entry with calls to the local llama Q8_1 quantizer and dot
routine. This is an external-library diagnostic, **not a production fix or
an acceptable final implementation**. Layer-0 FFN first-16 maximum error
drops from 0.00019655 to approximately 8e-9, confirming this source of
numerical error. However, the year gate still fails 0/8, and later-layer
differences remain; Q5_1 alone is not the complete explanation. Logs:
`/tmp/q51-oracle-q38year.log`, `/tmp/q38year-oracle-bn.trace`.
The next implementation must provide native format-local Q5_1 behavior
and exact tests, while continuing to localize the remaining divergence.
No production code was changed for this diagnosis. The baseline full
suite passes: `/tmp/q51-baseline-tests.log`.

### Native x86 Q5_1 activation quantization

The `q51-native` artifacts replace x86 Q5_1 float-activation evaluation
with a native Q8_1 dot. Activation quants use nearest-even rounding;
scale and scaled integer sum are independently rounded to FP16. The
dot retains eight FMA lanes and the existing x86 horizontal reduction,
with a separate block-ordered offset accumulation. Public single-vector,
batch, multi-input, and matmul routes select the same implementation.
Registry metadata declares GEMV-order matmul support. No model-specific
branches, model/session ownership changes, external runtime dependency,
or ARM/Metal changes are introduced.

Artifacts:

- `/tmp/q51-native-avx2`, SHA256
  `f1129cf93873604c4524c8a5a0a89ea66f592ad65df072c5534a17b00e8fc18c`.
- `/tmp/q51-native-avx512`, SHA256
  `c729e305f68ce716d71df7f7156fa01a7489361ef5766f3f2403a717c0cefec1`.

The layer-0 expert replay above now matches all **2560/2560** values
exactly. An independent ctypes comparison against each matching local
llama x86 library matches **3288/3288 outputs per ISA**, covering
1/2/3/20/80/321 blocks, 137 rows, random inputs, zero blocks, rounding
ties, and small scales. Script/log: `/tmp/q51-reference-check.{py,log}`.
This comparison uses the external library only as a test oracle.

Synthetic unit tests independently unpack Q5_1, quantize activations,
and reproduce the declared lane order. They cover unaligned float input,
signed offsets, zero blocks and rounding ties, partial row ranges,
output guards, NULL/zero-worker/three-worker dispatch, and all public
routes including prepared multi-GEMV. AVX2 tests, the clean warning-free
default build, full suite, and ASan/UBSan checks pass. Logs:
`/tmp/q51-avx2-final-tests.log`, `/tmp/q51-native-{clean,build,fulltests,sanitize}.log`,
and `/tmp/q51-native-final-fulltests.log` (including the final multi-GEMV
test addition). ISA build log: `/tmp/q51-native-isa-build.log`.

The optional `BN_PREFILL_ALLOW_HYBRID_BATCH=1` diagnostic does not resolve
the AVX512 year prompt: still 0/8 tokens. Log:
`/tmp/q51-native-q38year-hybrid.log`. No default prefill policy was changed.
This native quant correction is not yet an accepted full-model checkpoint.

Fresh eight-prompt, up-to-16-token strict gates compare the previous
`q6-rowpair` binaries with `q51-native`. Settings: eight threads, context
512, F32 KV both, flash off, default stopping. All four gates **fail**:

| Artifact | ISA | First-token matches | Matched token prefixes | Emitted bitnet / llama |
| --- | --- | ---: | ---: | ---: |
| q6-rowpair | AVX2 | 8/8 | 107/120 | 120/120 |
| q51-native | AVX2 | 8/8 | 117/120 | 123/120 |
| q6-rowpair | AVX512 | 7/8 | 98/123 | 128/123 |
| q51-native | AVX512 | 6/8 | 82/122 | 122/123 |

Prefix denominators sum the shorter generation length per prompt, not
necessarily llama's total; unequal lengths remain strict failures. AVX2
improves the year prompt from 5/16 to 16/16, but the fox prompt now has
11 versus 8 tokens (7/8 matching prefix), and the sum prompt still has
14/16. AVX512 retains its immediate year failure and **introduces an HTTP
regression**: bitnet starts with token 271 while llama uses 421; the old
artifact matched all 16 tokens there. Its fox and sum prompts also fail.
Logs: `/tmp/q51-{baseline,native}-q38sparse-avx{2,512}-128.log`.

The native kernel is retained as work in progress because its declared
quant computation is independently bit-exact, not because overall token
parity improved on both ISAs. The remaining accumulated-state divergence,
including the new AVX512 HTTP boundary, must be resolved before acceptance.
No speed acceptance is claimed. A final scalar-forcing guard preserves
the prior scalar kernel selection under `BN_FORCE_SCALAR`; normal ISA
binary hashes above are unchanged after rebuilding. Final checks:
`/tmp/q51-native-{scalar-guard-tests,final-isa-build}.log`.

### Hyper-connection residual product rounding

The Qwen3.8 sparse HTTP regression after native Q5_1 was localized to
residual combination. The reference graph multiplies block output by
scatter weight, rounds that product, then adds the residual. Bitnet's
combined expression could contract those operations into an FMA.

Controlled AVX512 diagnostics on top of `q51-native` distinguish the
operations: changing only hyper-connection mixing order still fails the
eight-token HTTP prompt; changing only residual-combine contraction
restores 8/8. Changing both also restores HTTP but does not resolve the
year prompt. Adding SIMD low-rank SiLU to the combined diagnostic does
not resolve year either. Logs: `/tmp/hc-{mix,combine,order,both}-parity.log`.
These temporary binaries are diagnostics, not the final implementation.

The final implementation adds a generic scaled-residual-add callback to
the existing CPU backend table. Its x86 helper stages the product through
volatile request scratch before adding, preventing contraction without
compiler-specific pragmas. Other backends retain the original expression.
Decode and batch-prefill hyper-connection combination both use this
callback, with session-owned `hc_norm` scratch. No ISA condition remains
in orchestration; model/quant/backend ownership is unchanged.

Final artifacts:

- `/tmp/hc-backend-avx2`, SHA256
  `aab09ef64d1d843cb1759193339cdba1751a5b240c2c2783cc69cb0fa36398f6`.
- `/tmp/hc-backend-avx512`, SHA256
  `5dfda23282de8e2b8edf958ce2925bbfc189cc8f716db77c2c4249c198b6904c`.

Synthetic tests exercise cancellation that distinguishes FMA from a
rounded product plus add, unaligned arrays, guards, and dimensions
1/7/8/15/16/17/32/33, both directly and through four-stream combination.
The final clean build is warning-free; the full suite (including the
architecture matrix), explicit Clang AVX2 transformer suite, and native
Clang ASan/UBSan transformer suite pass. Logs:
`/tmp/hc-backend-{clean,build,fulltests,avx2-tests,sanitize,isa-build}.log`.

Final two-prompt gates use sixteen tokens each, eight threads, context
512, F32 KV both, flash off, default stopping. **HTTP is 16/16 on both
ISAs**. Year still fails: AVX2 is 3/16 (worse than `q51-native`'s 16/16),
AVX512 is 0/16. Thus the combined gates fail at 19/32 and 16/32,
respectively, with equal generation lengths. Logs:
`/tmp/hc-backend-avx{2,512}-boundaries.log`. This is a partial arithmetic
correction, not overall model acceptance or a speed checkpoint.

The preceding `hc-staged` implementation used the same product staging
but placed ISA selection in orchestration; it failed the architecture
check and was replaced by the backend callback above. Its eight-prompt
diagnostic gates failed at AVX2 101/120 matching prefixes (128/120 emitted)
and AVX512 102/121 (121/123 emitted). Logs:
`/tmp/hc-staged-q38sparse-avx{2,512}-128.log`. These wider results belong
to that intermediate artifact, not a fresh full gate of `hc-backend`.

The layer probe now observes hyper-connection norm, gate, mixed output,
and injection tensors. Reshaped mixed-output aliases are excluded to
avoid capturing the wrong token row. A position-10 layer-0 FFN mixed
capture contains exactly 2560 floats and selects the second occurrence.
The original `q51-native` differs from llama at 842 values (max 2.38419e-7);
the combined arithmetic-order diagnostic differs at 614 (max 1.78814e-7).
This confirms remaining upstream rounding differences even before later
layer accumulation. Files: `/tmp/hc-{ll,current,order}-mixed-pos10.bin`,
`/tmp/hc-ll-pos10.trace`; probe build log: `/tmp/hc-probe-final-build.log`.

### Hyper-connection replay: F32 prefill versus GEMV order

Further AVX512 localization uses the year prompt at **position 0, layer
0**, avoiding earlier-token state accumulation. The production artifact
remains `hc-backend-avx512`; the following overrides are temporary
diagnostics only.

With the **same captured llama normalized input**, native Q8 down/up
projection plus hyper-connection mixing matches all 10240 gate values
and all 2560 mixed outputs. Scalar and SIMD low-rank SiLU, and original
versus separately staged mixing, all agree for this particular capture.
Script/log: `/tmp/hcmix-replay.{c,log}`. This rules out mixing as the cause
for this input; it is not a universal equivalence claim.

The actual production SSM output is likewise **2560/2560 exact**, but
the following residual differs at 4708/10240 values (max 1.30385e-8), and
its FFN normalization differs at 4563/10240 (max 1.56462e-6). Captures:
`/tmp/hcmix-bn-{ssm_out,hc_after_attn,hc_ffn_norm}-pos0.bin` and corresponding
`/tmp/hcmix-ll-{linear_attn_out,hc_combine,norm}-pos0.bin`.

An identical-input replay of the **four F32 injection weights** isolates
the reduction order: GEMV differs in two rows by +/-1.90735e-6, whereas
the existing F32 matmul kernel matches all four exactly. That kernel
uses one SIMD accumulator, versus GEMV's four independent accumulators.
Script/log: `/tmp/hcinject-replay.{c,log}`. A diagnostic that calls the
matmul kernel directly for injection removes **all 4708 residual
differences**. This establishes the cause of this local discrepancy.

Calling the public matmul API with one token is **not** the same
experiment: it falls back to GEMV. The initial `hcinject-inject` attempt
was therefore ineffective and is not evidence for a changed injection
order. The corrected direct-kernel binary is `/tmp/hcinject-direct-avx512`.
An additional `/tmp/hcinject-all-avx512` diagnostic replaces every F32
GEMV kernel call with matrix-order evaluation. It makes the first
residual, all 10240 normalized values, and all 2560 mixed values exact.
Captures: `/tmp/hcinject-all-{hc_after_attn,hc_ffn_norm,hc_ffn_out}-pos0.bin`.

Neither the corrected injection-only override nor the all-F32 override
fixes the first year token: still 557 versus llama 77296. HTTP's first
token matches. Logs: `/tmp/hcinject-{direct,all}-first-token.log`. These
two-prompt checks request just one token and do not establish decode or
speed parity. The all-F32 trace then shows matching first-layer expert
IDs but slightly different router scores/weights:
`/tmp/hcinject-all-pos0.trace` versus `/tmp/hcmix-ll-norm-pos0.trace`.

This exposes a prefill/GEMV distinction that must be handled deliberately,
not by globally changing decode arithmetic. It also contradicts the
current blanket native-AVX512 `cpu_matmul_matches_matvec` declaration
for F32. That declaration controls prefill projection replay, so changing
it alone could replace the desired matrix order with GEMV; the metadata
and its consumer need a coordinated review. No production arithmetic or
metadata was changed in this localization turn. The baseline full suite
passes: `/tmp/hcinject-baseline-tests.log`.

A further temporary `hcinject-router-avx512` binary combines matrix-order
F32 projections with matrix-order F32 routing/shared-gate dots. Its
logged first-layer selected weights match all ten llama weights, and
both targeted first tokens now match, including the formerly failing
year token 77296. Log: `/tmp/hcinject-router-first-token.log`; trace:
`/tmp/hcinject-router-pos0.trace`.

Extending that diagnostic to sixteen tokens per prompt yields **year
16/16**, but **HTTP 14/16**, with equal output lengths: 30/32 overall,
still a strict failure. Log: `/tmp/hcinject-router-32.log`. Thus a blanket
change is not an accepted fix: prefill matrix arithmetic and subsequent
GEMV decode need to be treated separately. The existing batched MoE
router already uses F32 matmul when given multiple tokens; the remaining
work should use coherent prefill execution/dispatch rather than retain
these global diagnostic overrides. No speed acceptance is claimed.

### Hyper-connection batch-prefill localization

The next diagnostic keeps production decode unchanged and exercises genuine
CPU batch prefill. Generation normally enforces the architecture's
`PREFILL_DECODE_PARITY` policy before the hybrid-batch option can apply.
Temporary `/tmp/hcbatch-generate.c` bypasses that entry policy; this is
not a production policy change. A direct profile confirms `batch=1`,
`parity_cpu=0`, and eleven prompt tokens in
`/tmp/hcbatch-inject-pos0.err`.

The entry bypass alone still produces year token 557 rather than 77296.
Its separate HTTP wording, `An HTTP 404 error means`, passes one token;
that wording is not the standard HTTP boundary prompt. Log:
`/tmp/hcbatch-entry-first.log`.

The existing HC batch helper still called injection GEMV per token.
Temporary `/tmp/hcbatch-prefill.c` retains per-token normalization and
mixing, collects normalized inputs, and calls public quant matmul once
for all injection rows/tokens. With entry bypass, it yields year 0/16
and standard HTTP 16/16: a strict 16/32 failure, equal lengths.
Log: `/tmp/hcbatch-inject-32.log`.

The shared-expert gate also remained a per-token F32 dot. Temporary
`/tmp/hcbatch-moe.c` computes all shared-gate logits through existing
F32 matmul, then applies the existing sigmoid and accumulation.
Combining these three diagnostic source replacements gives **32/32
token IDs**, equal lengths, on the year and standard HTTP prompts at
sixteen tokens each. Decode sources and kernels are unchanged. Log:
`/tmp/hcbatch-combined-32.log`. This resolves both targeted boundaries,
unlike the earlier global matrix-order override's 30/32.

The first-layer, position-zero MoE raw-output text trace now agrees
with the previous matrix-order projection/router diagnostic, including
summary statistics and the first sixteen values. This is a text-trace
comparison, not a full-tensor equivalence assertion. Files:
`/tmp/hcbatch-combined-pos0.trace` and `/tmp/hcinject-router-pos0.trace`.
The new raw-output binary contains all 2560 floats (10240 bytes).
A fresh capture from the earlier global-override diagnostic,
`/tmp/hcbatch-global-reference-moe_raw-pos0.bin`, is byte-identical to
`/tmp/hcbatch-combined-moe_raw-pos0.bin` (both 10240 bytes, `cmp` exits
zero). This verifies all 2560 raw-output values against that diagnostic,
not against a newly captured llama tensor.
The requested HC-residual dump on the injection-only run was **not
produced**, since batch prefill lacks that dump point; it must not be
treated as a successful capture.

AVX512 diagnostic SHA256s:

- Entry bypass: `c3462821570432377d1509ac3cf5e4a0109d654c9fda72890481f2e314ecc695`.
- Plus batched injection: `150ee07d893da6f55a0348558ac54739f7b554b615240907303ff61c7d5ed0cc`.
- Plus batched shared gate: `179c561c8a22db7f31dde1de8070b0d3e4bf1d903a7e1518168b1de64d88cfcd`.

All three Clang builds have empty warning/error logs. The unchanged
production baseline passes `make test` in
`/tmp/hcbatch-baseline-tests.log`. Production ISA binaries retain the
`hc-backend` hashes above. No speed acceptance or production promotion
is claimed by these temporary diagnostics.

The analogous explicit-AVX2 build is warning-free, SHA256
`039d4d1640354023da6d6af7d421777cbd166616ac5069f0050b21e48c69e12b`.
Against the matching AVX2 llama executable, it passes HTTP 16/16 but
matches only the first six year tokens: **22/32**, equal lengths,
strict failure. Log: `/tmp/hcbatch-combined-avx2-32.log`. AVX2 llama's
year continuation differs from AVX512 llama's, so each gate uses its
own matching ISA reference rather than imposing one ISA's token stream.

The full eight-prompt, sixteen-token AVX512 diagnostic gate finishes at
**114/119 matching token prefixes**, with **119/123 emitted tokens**
(bitnet/llama), and all eight first tokens matching. Six prompts match
fully; fox matches 4/7 compared prefixes with lengths 7/11, and the
arithmetic prompt matches 14/16. This remains a **strict failure**;
the denominator must not hide the unequal output lengths. Log:
`/tmp/hcbatch-combined-128.log`. No inference jobs remain running from
this localization pass.

Inspection also clarifies the earlier metadata caveat: both current
x86 backend initializers select `PREFILL_PROJECTION_REPLAY_NONE`.
The inaccurate blanket F32 matmul/GEMV equivalence declaration is thus
latent here, not an active source of replay in these diagnostic runs.

### Control-token probe correction and shared-gate shape replay

The layer probe previously excluded non-EOG control tokens from greedy
generation. Raw `llama-completion` does not do this: on the fox prompt,
both engines emit `5388,13,271,248045` (including `<|im_start|>`) before
their fifth-token divergence. The old probe instead selected token 2
at the control-token boundary, invalidating later free-running trace
comparisons there. This does **not** invalidate the completion-based
strict gates or initial-prefill captures above.

`test/llama_layer_probe.cpp` now performs unmasked raw greedy selection,
offers `--decode-token-ids` to force a continuation after ordinary
prefill, validates forced IDs against the vocabulary, and provides
`--self-test` for control-shaped IDs, tie-breaking, and malformed ID
lists. Forced mode prints both predicted and fed token IDs, and reports
the prediction after the final forced token. It is mutually exclusive
with `--generate`. The final AVX512 probe has SHA256
`e43785e2546abd897ceb05d407096fdf094cd736829538f79e487d4f9839112d`.

On the fox prompt, corrected greedy generation agrees with matching
llama-completion on **6/6 tokens and all 60 top-logit entries exactly**,
despite probe/completion thread counts of 1/8. Logs:
`/tmp/hcbound-fox-fixed-greedy-ll.out` and
`/tmp/hcbound-fox-completion.err`. Forcing the shared four-token prefix
also predicts completion's next token 271:
`/tmp/hcbound-fox-forced-ll.out`. The old, incorrect probe output is
retained separately in `/tmp/hcbound-fox-ll.out`.

Further localization corrects the previous interpretation of the
shared-gate override. At layer 0, last fox prompt position 7, the HC
FFN input is **2560/2560 byte-identical** to llama:
`/tmp/hcbound-fox-{bn,ll}-ffn-pos7.bin`. With that identical input,
replaying the one-row F32 shared gate gives:

- llama/GEMV: -0.785474658, exactly equal;
- matrix-order kernel: -0.785474777, difference -1.19209e-7.

Replay source: `/tmp/hcbound-gate-replay.c`. The matching reference's
`tinyBLAS::matmul` accepts rows divisible by four and SIMD-aligned
columns, otherwise falling back to dot-product order. Hence the
one-row shared gate uses GEMV **even during batched prefill**. The
previous all-matrix shared-gate override's token improvement was a
compensating error, not evidence that that override is correct.

Retaining the original shared-gate GEMV (`hcbatch-inject`) makes the
entire layer-zero FFN output **2560/2560 byte-identical** to llama:
`/tmp/hcbound-native-gate-ffnout-pos7.bin` versus
`/tmp/hcbound-fox-ll-ffnout-pos7.bin`, both 10240 bytes, `cmp` exit zero.
Layer-one HC mixed-input text values still differ, so the remaining
prefill discrepancy is not resolved. This artifact still fails the
fox/arithmetic strict gate at **18/27**, equal emitted lengths 27/27:
fox 4/11, arithmetic 14/16. Log: `/tmp/hcbound-native-gate-32.log`.

A separate temporary `/tmp/hcbound-mix-cpu.c` disables FP contraction
within HC mixing and scales after summing gated streams, following the
reference graph's operation sequence. With batched injection and
original shared-gate GEMV, `/tmp/hcbound-mix-avx512` still fails at
**34/59**, equal lengths: fox 4/11, arithmetic 14/16, year 0/16, HTTP
16/16. It also leaves the layer-one mixed-input text discrepancy.
Log: `/tmp/hcbound-mix-64.log`; trace:
`/tmp/hcbound-mix-all-pos7.trace`; SHA256:
`3afe286255134313a64bca19530809429c29a2051c200e42d2ca3471574f9e43`.
This override is not promoted. The next boundary to examine is the HC
residual/injection between layers zero and one, rather than attributing
the remaining difference to the now-verified layer-zero FFN output.

Both worktree ISA probe binaries were refreshed from the corrected
source and pass self-tests. AVX2 probe SHA256:
`e50dad47038aa6773e51c318eed1796886554240d9bb4e534d3a1aea31b6d46f`.
Its warning-free build log is `/tmp/hcbound-probe-avx2-build.log`.
The saved older `/tmp/hc-layer-probe-avx512` remains unmodified for
historical comparisons; do not use its masked generation as the raw
completion reference. All jobs from this pass are terminal.

Only the diagnostic probe and documentation changed in this pass;
production inference code and ISA artifacts remain unchanged.
Baseline and final `make test` pass in
`/tmp/hcbound-{baseline,final}-tests.log`; the clean default build is
warning-free in `/tmp/hcbound-build.log`. Probe self-tests pass, with
external headers treated as system includes for the warning-free
`/tmp/hcbound-probe-validated-build.log` build. No speed acceptance is
claimed.

### PLE reduction correction and all-position residual audit

The next boundary capture separates the layer-zero FFN residual from
the **positional layer embedding (PLE)** applied before layer one's HC
mixer. The reference renames the post-FFN residual to `l_last`, so
requesting a second `hc_combine-0` occurrence produces no file. A
temporary `/tmp/hcres-layer-probe` recognizes the `l_last` alias;
`/tmp/hcres-boundary-prefill.c` and `/tmp/hcres-preple-prefill.c` add
matching temporary observation points. These are diagnostic artifacts,
not production observation or execution changes.

For the fox prompt at position seven, **all 10240 values match** after
attention and after FFN/before PLE:

- `/tmp/hcres-{bn,ll}-afterattn.bin` (40960 bytes each).
- `/tmp/hcres-bn-preple.bin` versus `/tmp/hcres-ll-last.bin`
  (40960 bytes each).

Both `cmp` checks exit zero, and all four FFN injection values match.
The earlier `/tmp/hcres-bn-afterffn.bin` capture is actually **after
PLE**, so comparing it directly with `l_last` crosses an additional
operation and is not evidence that HC combining is wrong.

PLE's gate computes an elementwise product followed by a reduction.
The reference rounds each product to F32, sums in double, then converts
the result back to F32 before scaling. Production previously used a
float dot accumulation. A diagnostic double reduction changes all four
position-seven gates to the exact reference values:
`0.0756018236,0.0551914759,0.105617262,0.0161359068`.
Trace: `/tmp/hcple-dot.trace` versus `/tmp/hcbound-fox-ll-all-pos7.trace`.

The verified reduction is now implemented as the generic internal
`bn_transformer_sum_products` operation in `transformer/math_backend.c`.
x86 uses separately rounded float products and a double sum;
non-x86 backends retain their original float reduction. The PLE
orchestrator calls that operation without ISA/model-name branches.
Tests distinguish float versus double accumulation and rounded versus
unrounded products, and cover zero size, unaligned inputs, guard values,
tail lengths, and the actual 2560-element dimension.

Validation passes: baseline `make test`, clean warning-free default
build, final full suite (`/tmp/hcple-{baseline,final}-tests.log`,
`/tmp/hcple-build.log`), Clang AVX2/AVX512 reduction tests, scalar
fallback, and ASan/UBSan. LeakSanitizer initially could not run under
the sandbox's ptrace; the approved unsandboxed rerun of
`/tmp/hcple-sum-test-sanitize` passes. The new math backend, combined
with the existing temporary batched-injection diagnostic, reproduces
all four matching gates (`/tmp/hcple-backend-batch.trace`).

Current production ISA artifacts and targeted sixteen-token gates:

- AVX2 `/tmp/hcple-backend-avx2`, SHA256
  `c8320ff1e6b7c45918afd5288aab057cad5437a7e8d371a44e6048b4d6b6c547`:
  year 16/16, HTTP 16/16, **32/32**, equal lengths. This improves the
  preceding `hc-backend` checkpoint's 19/32 on the same pair.
- AVX512 `/tmp/hcple-backend-avx512`, SHA256
  `0152582b45964245afa944735aa75d05453e434b4a2494352a461c0259fa6078`:
  year 0/16, HTTP 9/16, **9/32**, equal lengths. This regresses from
  16/32 and is **not accepted model parity**. A verified local
  arithmetic correction does not imply globally improved token parity
  while other errors remain.

Logs: `/tmp/hcple-backend-avx{2,512}-32.log`. Both worktree ISA binaries
match these artifacts. No speed acceptance is claimed.

Two further PLE issues remain diagnostic-only: historical-first
convolution accumulation, and grouping `residual + (gated + conv)`
rather than `(residual + gated) + conv`. A temporary combined variant
also uses native SIMD SiLU for the convolution output. It still fails
the fox/arithmetic/year/HTTP gate at **34/59** matching prefixes, with
**64/59 emitted tokens**. Log: `/tmp/hcple-simd-64.log`. Neither this
variant nor its convolution changes were promoted.

An all-position residual audit explains why matching the final prompt
row was insufficient. Full pre-PLE captures
`/tmp/hcple-{bn,ll}-preple-all.bin` contain eight rows of 10240 floats
(327680 bytes each). Rows 0–3 and 5–7 are byte-identical. **Only row 4
differs: 9033/10240 values, max difference 4.24683094e-6.** Utility:
`/tmp/hcple-compare.c`. That row can enter later PLE convolution history.
At row four, the HC-to-FFN input is nevertheless **2560/2560 exact**
(`/tmp/hcple-{bn,ll}-ffn-pos4.bin`), and all ten logged router weights
match. The first-layer FFN output differs, narrowing the next upstream
investigation to expert/shared-FFN execution at this position. Traces:
`/tmp/hcple-{bn,ll}-pos4.trace`. Thus both a real PLE reduction error and
a separate earlier FFN discrepancy exist; do not attribute all residual
differences to either one alone. All jobs from this pass are terminal.

### Routed SwiGLU policy and corrected PLE history

At fox prompt position four, the logged first sixteen down-projection
values match for nine routed experts; slot four, expert 119, differs.
This is a logged-prefix comparison, not a full-tensor assertion for
each expert. Reference capture: `/tmp/ffn4-ll.trace`. The identical
FFN input and routing weights isolate the discrepancy downstream of
routing. Qwen4exp's `MOE_REFERENCE_SILU` flag forced scalar SiLU on
x86, whereas the reference uses vectorized SwiGLU.

Removing only that flag in a temporary architecture source makes
**all 81920 pre-PLE floats byte-identical**, including the previously
bad position four: `/tmp/ffn4-native-preple-all.bin` versus
`/tmp/hcple-ll-preple-all.bin`, 327680 bytes each. The diagnostic still
uses forced batch entry and batched HC injection; this does not claim
that the production prefill route has become identical.

The policy correction is now in the architecture registry, with a
regression assertion in `test_transformer.c`; reference MoE attention
is retained. No quant kernel or model-name branch was added. The NEON
SwiGLU implementation already ignores this scalar-reference flag, so
its activation implementation is unchanged.

Production Clang artifacts and the same year/HTTP sixteen-token gate:

- AVX2 `/tmp/ffn4-policy-avx2`, SHA256
  `51737934602e8c8717233e62ba0483b5d7d71d6c8a14362636990b292e7854ad`:
  **32/32**, equal lengths, unchanged from the PLE-reduction checkpoint.
- AVX512 `/tmp/ffn4-policy-avx512`, SHA256
  `7660b3c7474ad20279c70bb3bf228c660bf18936540723767e85c448faac02d3`:
  year 0/16, HTTP 16/16, **16/32**, equal lengths; improves from 9/32
  but remains a strict failure.

Logs: `/tmp/ffn4-policy-avx{2,512}-32.log`. Worktree ISA binaries
match these artifacts. No throughput acceptance is claimed.

A second temporary candidate combines this policy correction with
the previous PLE convolution-order, residual-grouping, and vector-SiLU
diagnostics. Its layer-one post-PLE HC normalization is now
**10240/10240 byte-identical** at fox position seven:
`/tmp/ffn4-ple-native-norm.bin` versus `/tmp/hcple-ll-norm.bin`,
40960 bytes each. Candidate: `/tmp/ffn4-ple-native-avx512`;
trace: `/tmp/ffn4-ple-native.trace`. This verifies that the earlier
FFN discrepancy contaminated PLE history. The temporary PLE source
contains a direct ISA call and is not suitable for promotion as-is;
production changes must remain backend-owned and preserve other
backends. Exactness at this intermediate boundary does not establish
full-model token parity.

The additional production fox/arithmetic gate also fails both ISAs:
AVX2 has prefixes 4/8 and 14/16 (**18/24**, emitted **32/24**),
AVX512 has 8/9 and 14/16 (**22/25**, emitted **25/27**).
Logs: `/tmp/ffn4-policy-avx{2,512}-fox-arith.log`.
The combined diagnostic's four-prompt result is **28/57** matching
prefixes, with **57/59 emitted tokens**: fox 9/9 but unequal lengths,
arithmetic 3/16, year 0/16, HTTP 16/16. Thus it fixes the inspected
boundary without fixing the complete generation, and arithmetic
regresses versus the earlier temporary PLE candidate. Log:
`/tmp/ffn4-ple-native-64.log`; candidate SHA256:
`354d4ae0e06dbb93052175e63180e84a1253d2e0419ead47bf72a936705f6af0`.

Baseline and final `make test` pass, along with `make clean` followed
by a warning-free default `make bitnet`. Logs:
`/tmp/ffn4-{baseline,final}-tests.log`, `/tmp/ffn4-clean.log`,
`/tmp/ffn4-build.log`. Both production Clang ISA builds are warning-free
in `/tmp/ffn4-policy-avx{2,512}-build.log`; `git diff --check` passes.
All jobs from this pass are terminal. The next step is to express the
verified PLE arithmetic through generic backend operations, then
continue tracing beyond the now-exact layer-one boundary; the
diagnostic batch-entry override must not become a global policy bypass.

### Backend-owned PLE convolution and residual arithmetic

The verified temporary PLE arithmetic is now expressed through two
generic internal math operations: `bn_transformer_dilated_conv_silu`
and `bn_transformer_scaled_branch_add`. x86 convolution accumulates
oldest-first with separately rounded F32 products, then applies native
SiLU; its residual operation groups `x + (rounded(value * scale) + branch)`.
Other backends retain current-first convolution, scalar SiLU, and
sequential residual additions. ISA dispatch is confined to
`transformer/math_backend.c`; CPU orchestration contains no new ISA
or model-name conditions. History remains session-owned and read-only
to the convolution helper; orchestration still advances it.

Synthetic tests cover zero size, kernel sizes 1–4, dilation 1–3,
unaligned buffers, widths 1/7/8/15/16/17/32/33, in-place output, guard
values, unchanged history, accumulation cancellation, residual grouping,
and non-fused scaled products. Focused Clang AVX2, AVX512, and scalar
tests pass (`/tmp/ple-op-test-{avx2,avx512,scalar}`). ASan/UBSan passes
in `/tmp/ple-op-test-sanitize` after an approved unsandboxed rerun;
the initial LeakSanitizer run cannot work under sandbox ptrace.

Baseline and final full tests pass in `/tmp/ple-op-baseline.log` and
`/tmp/ple-op-final-tests.log`. The clean default build and both Clang
ISA builds are warning-free (`/tmp/ple-op-build.log`,
`/tmp/ple-op-avx{2,512}-build.log`). Production artifacts:

- AVX2 `/tmp/ple-op-avx2`, SHA256
  `ff0a127819260067eadc7f5c8e7e04b3da8921b454c1bb99d8c455a9137827e9`.
- AVX512 `/tmp/ple-op-avx512`, SHA256
  `c31ad1cbcb567460af27d837f1b96e376822236e395c121eaaedaecd053efd00`.

Both worktree ISA binaries match these artifacts. Native ARM/Metal
runtime validation is not performed on this x86 host; no new ARM/Metal
acceptance is claimed.

A diagnostic build uses the production math helpers plus the existing
temporary batch-entry/HC-injection overrides. At fox position seven,
all **10240 post-PLE HC normalization values** remain byte-identical:
`/tmp/ple-op-batch-norm.bin` versus `/tmp/hcple-ll-norm.bin`, 40960
bytes each. The subsequent layer-one FFN output is also **2560/2560
byte-identical**: `/tmp/ple-op-l1-ffn.bin` versus
`/tmp/ple-op-ll-l1-ffn.bin`, 10240 bytes each. These are complete selected
row comparisons, not all-position/all-layer parity. Diagnostic binary:
`/tmp/ple-op-batch-avx512`, SHA256
`220e017cbbb0d68fab1be1e9792c7dce7ff868df57a4b5c652457babcea754f7`.
No global prefill-policy override is promoted.

Production four-prompt gates (fox/arithmetic/year/HTTP, sixteen-token
cap, eight threads, F32 KV and flash off) still fail:

- AVX2: prefixes 5/6, 14/16, 16/16, 16/16; **51/54**, with **54/56
  emitted tokens**. The preceding policy-only checkpoint had 50/56
  prefix matches and 64/56 emitted tokens over these same prompts.
- AVX512: prefixes 4/8, 14/16, 0/16, 9/16; **27/56**, with **56/59
  emitted tokens**. This regresses from the policy-only checkpoint's
  38/57 prefix matches and 57/59 emitted tokens. Exact local arithmetic
  does not make the remaining production path token-exact.

Logs: `/tmp/ple-op-avx{2,512}-64.log`. Both strict gates exit one;
neither is accepted model parity and no new speed acceptance is claimed.
All jobs from this pass are terminal, and `git diff --check` passes.

### SSM sigmoid output-gate correction

The next all-layer diagnostic (`/tmp/ple-next-ll.trace` versus
`/tmp/ple-op-batch.trace`) first differs in the logged FFN-output prefix
at layer six. Its recurrent delta output is nevertheless **6144/6144
byte-identical**: `/tmp/ple6-bn-delta.bin` versus
`/tmp/ple6-ll-delta.bin`, 24576 bytes each. The following sigmoid gate
used x86 SIMD exponential approximations, whereas the reference's
`ggml_vec_sigmoid_f32` uses scalar `expf`. This differs from the native
vector SiLU operation and must not be conflated with it.

A temporary scalar-sigmoid candidate makes the complete gated output
at layer six **6144/6144 byte-identical**:
`/tmp/ssmgate-native-gate.bin` versus `/tmp/ple6-ll-gate.bin`,
24576 bytes each. Its FFN-output **first sixteen values** match through
layer 46, with the first remaining prefix discrepancy at layer 47.
This does not assert whole-tensor or all-position equality through
those layers. Candidate: `/tmp/ssmgate-batch-avx512`, SHA256
`8f4fba2cdba11842f196369897e70112ec2f8173db34714861915ab95a1c2e5d`.
It retains diagnostic batch-entry and HC-injection overrides.

The correction is promoted only in `transformer/ssm_avx2.c`, shared
by the AVX2 and AVX512 sigmoid paths; the native SiLU path and existing
architecture-selected gate kind are retained. There are no model-name
checks or quantization changes, and NEON/Metal implementations are
untouched. The new exact regression test fails against the preceding
kernel and passes against the correction. It covers widths
1/7/8/15/16/17/31/32/33/128/257, unaligned buffers, selected head ranges,
untouched surrounding values, and infinite sigmoid inputs.

Both explicit Clang SSM suites pass (`/tmp/ssmgate-suite-avx{2,512}`),
including existing SiLU, convolution, normalization, recurrent delta,
and dispatch checks. Focused ASan/UBSan passes in
`/tmp/ssmgate-test-sanitize` after an approved rerun outside sandbox
ptrace. Baseline full tests pass in `/tmp/ssmgate-baseline.log`.

The diagnostic four-prompt gate still fails at **36/56 prefix tokens**,
with **56/59 emitted tokens**: fox 4/8, arithmetic 16/16, year 0/16,
HTTP 16/16. Log: `/tmp/ssmgate-batch-64.log`. No speed acceptance is
claimed. The next observed shape difference is the final FFN:
llama selects output rows after the final attention projection, so
its final FFN/router operates on one row, whereas the diagnostic
bitnet prefill still processes the entire prompt matrix. Reference
code: `qwen4exp.cpp` lines 400–406. This can select different F32
GEMV/GEMM arithmetic and requires a runtime-level solution, not a
model-specific quantization override.

That next boundary is now directly verified. The final FFN input is
**2560/2560 byte-identical** (`/tmp/ssmgate-l47-input.bin` versus
`/tmp/ssmgate-ll-l47-input.bin`, 10240 bytes each). Replaying the F32
router on that input reproduces **512/512 reference logits with GEMV**,
but GEMM differs in **258/512**, max difference **1.90735e-6**.
Reference logits: `/tmp/ssmgate-ll-l47-router.bin`, 2048 bytes;
replay source/binary: `/tmp/ssmgate-router-replay{.c,}`. This identifies
a real final-row execution mismatch without changing quant kernels.
Do not use the existing probe's final-layer top-k/weight printout:
its special expert-row reader still indexes the original prompt
position after the reference has compacted to one output row. The
F32 input/logit captures used here select the valid compacted row.

Final validation passes: clean warning-free default build, full tests,
and `git diff --check` (`/tmp/ssmgate-build.log`,
`/tmp/ssmgate-final-tests.log`). Production Clang artifacts and the
four-prompt sixteen-token gate:

- AVX2 `/tmp/ssmgate-avx2`, SHA256
  `0820327e0d7a0cdf6f11f4fdf320b79f5372039be725addaa25aadd6476f4b6c`:
  **54/56 prefix tokens**, equal **56/56 emitted lengths**. Fox 8/8,
  arithmetic 14/16, year 16/16, HTTP 16/16. Still a strict failure.
- AVX512 `/tmp/ssmgate-avx512`, SHA256
  `0cce20b943db37e7e98965643ba2e49d4ea6983d372950371728f43c3995562f`:
  **18/59 prefix tokens**, **64/59 emitted lengths**. Fox 4/11,
  arithmetic 14/16, year 0/16, HTTP 0/16. Regresses from 27/56 and
  is not accepted parity.

Logs: `/tmp/ssmgate-avx{2,512}-64.log`; both ISA builds are warning-free
and worktree binaries match the saved artifacts. All jobs are terminal.
No speed acceptance is claimed; final-row-aware prefill execution and
the production sequential-prefill discrepancy remain unresolved.

### Last-output-row prefill diagnostic

A temporary `/tmp/lastrow-prefill.c` compacts the activation and HC
residual views to the final output row after the final attention/SSM
state updates and before the final FFN. It leaves `all_logits` requests
uncompacted. This is combined with the existing temporary batch-entry
and batched-HC-injection overrides; no production runtime policy is
changed in this pass. The diagnostic mutates the local token count and
therefore also changes profile reporting; a production implementation
must preserve original request metadata and explicitly test all-logit,
no-logit, and session-state behavior.

AVX512 candidate `/tmp/lastrow-batch-avx512`, SHA256
`f0a9cd681b1f79f8abbc4bebd723aa6e4670e848533ac0a9aa771d5a866952a3`,
has **512/512 exact final-router logits**:
`/tmp/lastrow-l47-router.bin` versus
`/tmp/ssmgate-ll-l47-router.bin`, 2048 bytes each. All 48 logged FFN
output prefixes match. More importantly, the **complete prefill logit
vector is byte-identical: 248320/248320 floats**.
Files: `/tmp/lastrow-bn-logits-all.bin` and
`/tmp/lastrow-ll-logits.bin`, 993280 bytes each. These results are for
the fox prompt, eight prompt tokens; they do not imply all-prompt or
decode-state parity.

The diagnostic four-prompt sixteen-token gates are:

- AVX2 `/tmp/lastrow-batch-avx2`, SHA256
  `8f2fe708b33d0f58c71fc68124ffe39c3ead8410dc89f9e56dc063065e3e09a5`:
  **56/56 token IDs**, equal lengths, strict pass.
- AVX512: **36/56 prefix tokens**, **56/59 emitted lengths**, strict
  failure; fox 4/8, arithmetic 16/16, year 0/16, HTTP 16/16.

Logs: `/tmp/lastrow-batch-avx2-64.log`, `/tmp/lastrow-batch-64.log`.
Both temporary builds are warning-free. Production binaries remain
the preceding `ssmgate` artifacts; no throughput acceptance is claimed.

The continuing AVX512 fox trace consumes the same four-token prefix
`5388,13,271,248045`. Logged raw FFN prefixes match at every layer at
decode positions eight and nine. At position ten, the first differing
FFN prefix is layer four; at position eleven it is layer five, where
current QKV prefixes match but convolution differs, consistent with
earlier history contamination. At position ten/layer four, SSM-output
and post-attention-HC normalization prefixes match, but the following
HC mix/FFN input differs. Traces: `/tmp/lastrow-decode-all.trace` and
`/tmp/lastrow-ll-decode11.trace`. Compare CPU decode `bitnet_moe_raw`
with reference `llama_ffn_out`, not decode `bitnet_lout`, which is a
different boundary.

Capture caveats: `BN_DUMP_LAYER_POS=-1` selects position minus one,
not all positions; omit it to capture all. `LLAMA_PROBE_DUMP_OCCURRENCE`
counts matches separately at each position, not across decode steps.
Generated-token probes without a dump/list request intentionally emit
only token IDs. Missing captures from those configurations are not
numerical mismatches.

The broader AVX2 eight-prompt gate (24-token cap) passes **176/176
sampled IDs with equal lengths**, `/tmp/lastrow-batch-avx2-192.log`.
This is the temporary last-row/batched-entry candidate, not the
production sequential-prefill route.

At decode position ten/layer four, the complete HC normalization input
is **10240/10240 exact**: `/tmp/lastrow-p10l4-norm.bin` versus
`/tmp/lastrow-ll-p10l4-norm.bin`, 40960 bytes each. The HC low-rank
branch still used scalar SiLU while the reference uses vector SiLU.
A second temporary candidate replaces that activation on AVX512,
making the complete HC-to-FFN mixed input **2560/2560 exact**:
`/tmp/lastrow-hcsilu-p10l4.bin` versus
`/tmp/lastrow-ll-p10l4-mixed.bin`, 10240 bytes each. This variant
contains a direct ISA call in a temporary CPU source and must not be
promoted as-is. Candidate `/tmp/lastrow-hcsilu-avx512`, SHA256
`6715b55df32c7214721b25a870b79c38aed186614863e06ef2dfc0c035a35aff`.

Its default fox gate improves to **9/9 matching prefix IDs**, but
emits **9/11 tokens** and therefore still fails strict parity:
`/tmp/lastrow-hcsilu-fox.log`. The generation loop's four-token
repetition detector is a possible early-stop cause, pending a run
with that explicit runtime option disabled. Initial temporary build
attempts failed because of a relative include and an incomplete source
copy; the corrected build is warning-free in
`/tmp/lastrow-hcsilu-build3.log`. No production source changed here.

The explicit `BN_DISABLE_LOOP_ABORT=1` rerun confirms the fox length
difference was the repetition detector: **11/11 sampled IDs and equal
lengths**, strict pass in `/tmp/lastrow-hcsilu-fox-noloop.log`. This
option is part of that result and must not be omitted when comparing
the candidate with the raw reference continuation.

An isolated AVX2 five-run pp128/tg64 benchmark attempt produces no
accepted throughput result: bitnet emits only 61 of the required 64
tokens and the harness rejects it (`/tmp/lastrow-avx2-speed.log`).
The positive token count reflects normal end-of-generation; loop abort
returns minus one. A separate matched pp128/tg32 workload is used for
the subsequent throughput check, not a truncated tg64 sample.

The isolated eight-thread, five-run **pp128/tg32** check fails both
85% thresholds for `/tmp/lastrow-batch-avx2`:

- Decode: **5.03 versus 6.62 tok/s**, ratio **0.760**; bitnet samples
  5.03, 5.01, 5.02, 5.03, 5.07 tok/s.
- Prefill: **17.62 versus 54.34 tok/s**, ratio **0.324**; bitnet samples
  16.92, 17.57, 17.75, 17.89, 17.62 tok/s.

Log: `/tmp/lastrow-avx2-speed32.log`, terminal exit one. The reference
is the matching AVX2 llama-bench with flash disabled; bitnet uses its
default F32 KV and the reference throughput harness uses its default
F16 KV. No other build/inference job ran during the measurement.
Thus the candidate passes the eight-prompt token gate but does not
meet either throughput requirement. All jobs from this pass are
terminal and `git diff --check` passes. Production sources/binaries
are unchanged; runtime integration, all-logit/session-state tests,
the remaining AVX512 matrix, and substantial speed work remain.

### MoE prefill preparation policy diagnostic

Profiling the AVX2 last-output-row candidate above at eight threads, pp128
(token ID 1 repeated), identifies 2846.8 ms of local expert preparation across
4549 expert visits. Total prompt time is 7497.9 ms. The existing
`BN_CPU_DISABLE_PREPARED_QWEIGHTS=1` setting previously had no effect on this
batch-prefill path, although decode already honors it.

A three-line runtime-policy guard now makes prefill honor the same setting;
default preparation behavior is unchanged. The temporary guarded candidate
`/tmp/lastrow-moe-policy-avx2` retains the diagnostic last-row/batched-entry
changes described above. With preparation explicitly disabled it passes
**176/176 sampled IDs, equal lengths, eight prompts**, against matching AVX2
llama-completion: `/tmp/moeprep-raw-avx2-192.log`.

One isolated profile with that setting reports **5376.4 ms** prompt time,
versus 7497.9 ms above (28% lower). Gate/up compute increases from about
346 to 1069 ms, but avoiding 2847 ms of preparation more than offsets it.
Logs: `/tmp/lastrow-moe-prof.err`, `/tmp/moeprep-guard-raw.err`. These are
single profiling samples, not a new accepted throughput gate; the overall
85% speed requirement remains unmet. The default runtime and production
generation-entry behavior have not been changed by this diagnostic.

The regression test warms the prepared cache, then verifies at an activation
observer that disabled preparation leaves every entry evictable. Checking an
empty cache was insufficient: prefill uses local layouts on cache misses.
The corrected test fails against the previous implementation at the pinned
cache assertion (`/tmp/moeprep-test-old2.log`).

The focused MoE suite passes with the fix in the native build and explicit
Clang AVX2/AVX512 builds (`/tmp/moeprep-test-new.log`,
`/tmp/moeprep-avx2-test.log`, `/tmp/moeprep-avx512-test.log`). Rebuilt production
binaries contain only the runtime guard, not the temporary batching/last-row
diagnostics:

- AVX2 SHA256: `48c17306feace937f0ba8cbaedd2d36d261e3fffc574049491663e9b3403fefd`.
- AVX512 SHA256: `d3d28f9ed563c309a4c12030d9e0280953346847e60ae3a3d8e5c3650f76c9de`.

Baseline `make test`, then `make clean && make bitnet` and the full `make test`
all pass. Default and explicit-ISA builds report no warnings. Logs:
`/tmp/moeprep-baseline.log`, `/tmp/moeprep-clean-build.log`,
`/tmp/moeprep-full-test.log`. `git diff --check` passes; all jobs are terminal.

### HC low-rank activation integration

The temporary last-output-row AVX512 candidate with native HC low-rank SiLU
now passes the full eight-prompt, 24-token-cap gate: **179/179 sampled IDs,
equal lengths**, against matching AVX512 llama-completion. The corresponding
AVX2 candidate passes **176/176 IDs, equal lengths**. Both runs explicitly set
`BN_DISABLE_LOOP_ABORT=1`; the different totals reflect each ISA reference's
fox continuation/EOG behavior. Logs: `/tmp/hcsilu-avx512-192-noloop.log` and
`/tmp/hcsilu-avx2-192-noloop.log`. These are correctness runs, not speed samples.

Production now calls `bn_transformer_scaled_silu` for HC low-rank activation.
The helper lives in `transformer/math_backend.c`: F32 scale first, native
AVX512/AVX2 SiLU and scalar tails, unchanged scalar arithmetic on other
backends. No ISA branching was added to model or transformer orchestration.
The temporary batching/generation-entry/last-row changes remain unintegrated.

The new exact-arithmetic test covers unaligned arrays, vector-width boundaries,
scalar tails, zero size, multiple scales, and buffer guards. It rejects the old
scalar operation (`/tmp/hcsilu-old-op-test.log`) and passes AVX2, AVX512, and
forced-scalar focused builds. ASan/UBSan passes outside the sandbox after the
initial LeakSanitizer ptrace restriction. Baseline tests, clean default build,
full parallel test suite, and explicit Clang ISA builds pass without warnings:
`/tmp/hcsilu-baseline.log`, `/tmp/hcsilu-clean-build.log`,
`/tmp/hcsilu-full-test.log`, `/tmp/hcsilu-production-avx{2,512}-build.log`.

Rebuilt production binary SHA256:

- AVX2: `ec8d70c3140364e153e81eecb235f5ddd04f4528c2c2fe53b2720edd912943c0`.
- AVX512: `54a6802baa78212037e35bf42005d21d49de00590538650ab334a251956afbb5`.

The separate eight-prompt production CLI gates still **fail**, with the same
explicit loop-abort setting and matching references:

- AVX2: **150/176 matching prefix IDs**, equal total lengths (176/176).
- AVX512: **106/179 matching prefix IDs**, emitted lengths 192/179; one
  prompt has unequal token counts.

Logs: `/tmp/hcsilu-production-avx2-192-noloop.log` and
`/tmp/hcsilu-production-avx512-192-noloop.log`, both terminal exit one.
The activation correction is not a production parity claim. The passing
diagnostic still requires batched HC injection and final-output-row selection;
production keeps its existing decode-style prefill policy. Integrating those
changes requires all-logit, position, KV/SSM/PLE state, and continuation tests,
without globally bypassing generation's policy. No new throughput gate was
run in this pass. All jobs are terminal and `git diff --check` passes.

### Production HC prefill and output-row integration

The passing prefill behavior is now integrated without the temporary global
generation-entry override. CPU backend capabilities select batched HC
injection and final-FFN output-row selection on AVX2/AVX512; other backends
retain their existing paths. Generation's parity fallback is bypassed only
when the model uses hyper-connections and the CPU backend supports their
batch path. Model architecture flags and quant kernels are unchanged.

HC normalization rows use prefill-owned arena storage and injection runs
through the existing prefill projection helper. Final-row selection occurs
after final attention/SSM and PLE updates, applies only to host execution,
and does not prune all-logit requests. Profiling retains the original prompt
length. The new synthetic test compares last-logit, all-logit, and no-logit
requests at a nonzero start position, checking exact KV/SSM/PLE history and
token-history agreement, unchanged caller-owned position, finite outputs,
and bit-exact next-token continuation. It covers dense and HC-MoE fixtures,
with either attention or SSM in the final layer; PLE is deliberately placed
in the final HC layer.

An initial direct quant call was rejected by the backend architecture check;
it was replaced by the existing prefill projection helper. The corrected
clean build and full test suite pass without warnings:
`/tmp/hcprefill-clean-build2.log`, `/tmp/hcprefill-full-test2.log`.
Before that routing-only correction, the integrated binaries passed Qwen3.8
sparse **176/176 AVX2 and 179/179 AVX512 IDs**, and Qwen3 dense **192/192 on
both ISAs**, all equal lengths. Logs: `/tmp/hcprefill-avx{2,512}-192.log` and
`/tmp/hcprefill-q3dense-avx{2,512}-192.log`. All token runs explicitly set
`BN_DISABLE_LOOP_ABORT=1`, eight prompts, 24-token cap, eight threads,
F32 KV, matching ISA llama-completion, and flash disabled.

Final production binary SHA256 (after the projection routing correction):

- AVX2: `0cdfc5d5a99a10c2a4318602f203bbff981d134f49ea06bb753a232a92340625`.
- AVX512: `21b267719989f32bb3b6cacae8cc15715120d54b14466c36328abb22254224ce`.

The exact final production binaries also pass Qwen3.8 sparse:
**176/176 AVX2 and 179/179 AVX512 sampled IDs, equal lengths**, with the
explicit settings above. Logs:
`/tmp/hcprefill-final-q38sparse-avx2-192.log` and
`/tmp/hcprefill-final-q38sparse-avx512-192.log`. The final synthetic state
ASan/UBSan rerun passes in `/tmp/hcprefill-state-sanitize-approved.log` after
rebuilding against the corrected projection routing. Its first sandboxed
attempt was blocked by LeakSanitizer's ptrace restriction, not a reported
memory defect.

The exact final binaries additionally pass **192/192 IDs with equal lengths**
on both AVX2 and AVX512 for Qwen3-4B-Q4_K_M and Qwen3.8-27B-UD-Q4_K_XL.
Logs: `/tmp/hcprefill-final-q3dense-avx{2,512}-192.log` and
`/tmp/hcprefill-final-q38dense-avx{2,512}-192.log`. Thus this pass has final
production token evidence for these two dense fixtures and Qwen3.8 sparse,
not a renewed acceptance of the entire model/backend matrix.

The isolated final-production AVX2 **pp128/tg32**, eight-thread, five-run
throughput gate still fails both 85% thresholds, using default preparation
and cache settings:

- Decode: **5.08 vs 7.05 tok/s**, ratio **0.721**; bitnet samples
  5.08, 5.10, 5.07, 5.10, 5.03.
- Prefill: **17.72 vs 56.55 tok/s**, ratio **0.313**; bitnet samples
  17.72, 17.68, 17.69, 17.77, 17.93.

Log: `/tmp/hcprefill-final-avx2-speed32.log`, terminal exit one. This uses
matching AVX2 llama-bench with flash off, bitnet F32 KV versus the reference
benchmark's default F16 KV, and explicit `BN_DISABLE_LOOP_ABORT=1`. No other
build or inference jobs ran during measurement. Command settings are
`--skip-topk --benchmark-prefill --benchmark --llama-throughput bench
--bench-prompt-tokens 128 --bench-tokens 32 --bench-runs 5
--min-prefill-throughput-ratio .85 --min-throughput-ratio .85 -t 8`.
The previous preparation profile remains the next optimization lead;
production parity is no longer dependent on temporary sources. All jobs in
this pass are terminal and `git diff --check` passes.

### Preparation-disabled production throughput follow-up

On the exact production AVX2 binary above, the isolated pp128/tg32,
eight-thread, five-run gate with `BN_CPU_DISABLE_PREPARED_QWEIGHTS=1`
and `BN_DISABLE_LOOP_ABORT=1` reports:

- Decode: **7.03 vs 7.04 tok/s**, ratio **0.999**; bitnet samples
  7.03, 7.03, 7.03, 7.03, 7.11. The decode threshold passes.
- Prefill: **25.32 vs 56.95 tok/s**, ratio **0.445**; bitnet samples
  24.66, 25.71, 25.33, 24.97, 25.32. The prefill threshold fails.

Log: `/tmp/hcprefill-raw-avx2-speed32.log`, terminal exit one because
prefill remains below 85%. Other settings match the preceding throughput
gate, including flash off and benchmark KV defaults. No other build or
inference ran concurrently. Production defaults are unchanged.

The next temporary candidate batches HC down/up projections as well as
injection, replacing per-token projection dispatch with matrix operations.
`/tmp/hcfullbatch-avx2`, SHA256
`87b4458851342afc3b6c4a49eb38066638d584edfd805d08dba178a0bb5477a1`, uses
`/tmp/hcfullbatch-prefill.c`. This remains diagnostic: it uses temporary
per-call allocations and a copied norm helper, which must not be promoted
without proper workspace ownership and backend routing.

The preparation-disabled final production AVX2 binary passes the matching
eight-prompt token gate: **176/176 IDs, equal counts**, in
`/tmp/hcprefill-final-raw-avx2-192.log`. This supplies correctness evidence
for the near-parity decode result under the explicit preparation setting.
The temporary fully batched HC candidate also passes **176/176 IDs, equal
counts**, in `/tmp/hcfullbatch-raw-avx2-192.log`, and passes the existing
synthetic all/last/no-logit state-and-continuation suite when compiled with
Clang AVX2 (`/tmp/hcfullbatch-state-avx2.log`). Both candidate builds are
warning-free. No production source changed in this follow-up.

The isolated five-run pp128 check for the temporary fully batched HC
candidate measures **30.26 vs 56.62 tok/s**, ratio **0.534**, still below
85%. Bitnet samples: 29.45, 30.26, 30.44, 29.41, 30.35. This is about
19.5% higher than the 25.32 tok/s preparation-disabled production result.
Log: `/tmp/hcfullbatch-raw-avx2-pp128.log`, terminal exit one. Settings remain
eight threads, preparation and loop-abort disabled, matching AVX2 reference,
flash off, default benchmark KV formats. Decode was not remeasured for this
prefill-only candidate. No other build or inference ran concurrently.
The gain supports integrating the batched HC operations with reusable
prefill workspace and shared normalization routing, but does not establish
overall speed parity. All jobs are terminal; `git diff --check` passes.

### Batched HC projection production integration

HC down/up batching now uses reusable prefill-arena normalization, low-rank,
and gate buffers, with checked incremental size accounting. Reference-order
RMSNorm is shared through `transformer/rmsnorm_backend.c` instead of copied
into prefill. Scalar tails and arithmetic order are unchanged; the existing
x86 HC-batch capability controls the new path. ARM/Metal dispatch and default
prepared-weight policy are unchanged.

Baseline tests, clean default build, full tests, and the backend architecture
check pass without warnings (`/tmp/hcintegrate-baseline.log`,
`/tmp/hcintegrate-clean-build.log`, `/tmp/hcintegrate-full-test.log`). The
new normalization contract test covers exact F32-square/double-sum order,
unaligned buffers, guards, tails, zero size, and in-place output. The synthetic
state test now exercises reference-order HC normalization too. Clang AVX512
state tests and the AVX2 ASan/UBSan state suite pass:
`/tmp/hcintegrate-state-avx512.log`,
`/tmp/hcintegrate-state-sanitize-approved.log`. The sanitizer's first sandboxed
run hit the known LeakSanitizer ptrace restriction; the approved rerun passes.

Production Qwen3.8 sparse gates pass **176/176 AVX2 and 179/179 AVX512 sampled
IDs, equal counts**, eight prompts with a 24-token cap, eight threads, matching
ISA llama-completion, F32 KV, flash off, and explicit
`BN_CPU_DISABLE_PREPARED_QWEIGHTS=1 BN_DISABLE_LOOP_ABORT=1`.
Logs: `/tmp/hcintegrate-avx{2,512}-192.log`.

Production SHA256:

- AVX2: `92a82b9f6d984f1828f3d282df66435b451e79460e11080e446040e1ef607eb8`.
- AVX512: `fd7bf96a6f0ab5e2f560f9315367fbc645aec6aafc168469f6049a159e82733f`.

A separate temporary Q5_1 diagnostic batches thread dispatch while calling
the existing dot kernel unchanged for each token. It uses
`/tmp/q51batch-matmul.c`; binary `/tmp/q51batch-avx2`, SHA256
`70602666096cca54dcd7b7139a7bfda6c37c666baa6800b1d51a164997c097b3`.
The Clang AVX2 quant suite passes, including the existing exact Q5_1 tests
(`/tmp/q51batch-test-avx2.log`). This quant change is not in production.

Isolated eight-thread, five-run pp128 measurements with preparation and
loop-abort disabled confirm **30.07 vs 56.61 tok/s (0.531)** for integrated
production HC prefill. Bitnet samples: 30.54, 30.07, 29.73, 30.04, 30.45.
Log: `/tmp/hcintegrate-avx2-pp128.log`, exit one. The dispatch-only Q5_1
diagnostic reaches **30.84 vs 56.57 (0.545)**, samples 31.59, 30.72, 31.19,
30.84, 30.73 (`/tmp/q51batch-avx2-pp128.log`, exit one). Its small gain does
not support dispatch overhead as the principal remaining bottleneck.

A second temporary kernel replaces Q5_1's out-of-line software half-to-float
conversions with F16C intrinsics. It retains the dispatch diagnostic and
passes the existing Clang AVX2 quant suite, including exact Q5_1 comparisons
(`/tmp/q51f16c-test-avx2.log`). Sources: `/tmp/q51f16c-kernel.c` and
`/tmp/q51batch-matmul.c`; binary `/tmp/q51f16c-avx2`, SHA256
`f6e3914ca25694c0917205638f9211b9bb72356fb3eb70179ab06f4074b3f056`.
This is not production and has not yet passed a full model token gate.

The F16C diagnostic's isolated five-run pp128 result is **34.99 vs 56.82
tok/s**, ratio **0.616**; samples 34.70, 34.99, 35.40, 35.45, 34.44.
Log: `/tmp/q51f16c-avx2-pp128.log`, terminal exit one. This is a clearer
improvement than dispatch batching alone but still misses the 85% target.
All three speed checks used the same explicit preparation/loop-abort
settings, matching AVX2 reference, flash off, default benchmark KV formats,
and no concurrent build/inference jobs. Before promotion, the conversion
candidate needs full-model token evidence, FP16 edge-case coverage, and a
proper non-F16C fallback. All jobs are terminal and `git diff --check` passes.

### Q5_1 native conversion and batched dispatch integration

Q5_1 now uses format-local hardware half-to-float conversion when F16C is
available, with the existing software conversion as its compile-time fallback.
Its native matmul dispatch calls the existing per-token dot arithmetic from
one row-dispatched batch, respecting the native-matmul disable policy. The
registry's existing GEMV-order matmul capability remains the routing contract;
no model-specific branches or prepared layouts were added. ARM/Metal and
other quant formats are unchanged.

Tests sweep all 65536 FP16 encodings in each affine metadata field. Non-NaN
results match the software-conversion reference bit-for-bit; NaNs retain NaN
classification. Exact batch tests cover NULL/serial/threaded pools, disabled
native matmul, unaligned input, partial-row writes, and guards. The Q5_1 object
compiled with `-mno-f16c` passes the current quant suite
(`/tmp/q51-no-f16c-test2.log`); this checks that module's fallback, not an
otherwise F16C-free build of all AVX2 kernels.

Baseline, clean default build, full test suite, backend architecture checks,
and ASan/UBSan quant tests pass without warnings. Logs:
`/tmp/q51integrate-baseline.log`, `/tmp/q51integrate-clean-build.log`,
`/tmp/q51integrate-full-test.log`, `/tmp/q51integrate-sanitize-approved.log`.
The sanitizer's sandboxed attempt encountered the known LeakSanitizer ptrace
restriction; its approved rerun passes. The pre-integration F16C candidate
also completed its full AVX2 token gate at **176/176 IDs, equal counts**
(`/tmp/q51f16c-avx2-192.log`).

Production binary SHA256:

- AVX2: `625ac7b90dfaf68cc40ae7c5521d699fea017bb1e6740ed9ad6589c2e87fa110`.
- AVX512: `b8652abf2a43da8f74b5ba515dedcf509e77fc0570a327bf9205d4137e67b6dc`.

The exact integrated binaries pass the Qwen3.8 sparse eight-prompt gates:
**176/176 AVX2 and 179/179 AVX512 sampled IDs, equal counts**, with a
24-token cap, eight threads, matching ISA llama-completion, F32 KV, flash
off, and explicit `BN_CPU_DISABLE_PREPARED_QWEIGHTS=1 BN_DISABLE_LOOP_ABORT=1`.
Logs: `/tmp/q51integrate-avx{2,512}-192.log`. Disassembly confirms native
half-to-float instructions in Q5_1 while the separately rounded float-to-half
input metadata calculation is unchanged.

The integrated AVX2 five-run pp128/tg32 gate, eight threads and the same
explicit preparation/loop-abort settings, reports:

- Decode: **7.05 vs 7.21 tok/s**, ratio **0.978**, threshold pass; bitnet
  samples 7.02, 7.02, 7.09, 7.05, 7.05.
- Prefill: **35.54 vs 56.42 tok/s**, ratio **0.630**, threshold fail; bitnet
  samples 35.54, 35.81, 35.87, 35.22, 34.55.

Log: `/tmp/q51integrate-avx2-speed32.log`, terminal exit one. The reference
is matching-ISA llama-bench with flash off; throughput retains bitnet's F32
KV and llama-bench's default F16 KV. No other build or inference ran during
the measurement. Overall speed parity remains unmet despite decode passing.

The matching isolated AVX512 gate reports:

- Decode: **6.91 vs 6.99 tok/s**, ratio **0.989**, threshold pass; bitnet
  samples 6.93, 6.91, 6.93, 6.88, 6.89.
- Prefill: **35.05 vs 60.51 tok/s**, ratio **0.579**, threshold fail; bitnet
  samples 35.94, 34.92, 35.55, 35.03, 35.05.

Log: `/tmp/q51integrate-avx512-speed32.log`, terminal exit one. Same pp128/tg32,
eight-thread, five-run and explicit runtime settings; reference ISA changes
to AVX512. It ran after the AVX2 benchmark, with no competing task jobs.
Both checked ISAs now meet the decode target with production token parity
for this fixture, while prefill and the broader matrix remain incomplete.

A subsequent isolated pp128 profile of current production AVX2 with
preparation disabled records 3942.4 ms prompt time and 2067.2 ms MoE time:
route 143.2, gate/up 1092.7, activation 11.0, down 637.4, accumulation 43.0,
shared 79.8 ms (`/tmp/q51integrate-profile.err`). This is a single diagnostic
sample, not another throughput gate. Gate/up is now the largest measured
MoE projection cost; the existing profile does not fully separate HC/SSM
costs outside MoE, so those must not be inferred from the residual alone.
All jobs are terminal and `git diff --check` passes.

### HC/SSM timing and rejected Q4_K conversion experiment

An AVX2 diagnostic copy of current prefill added timers around batched HC
mixing and the attention/SSM mixer branches. With the same Qwen3.8 Flash Next
fixture, preparation disabled, 128 prompt tokens, and eight threads,
`/tmp/hcmixer-profile.err` completed with prompt time 3987.0 ms:

- Batched HC mixing: 551.7 ms across 95 calls.
- SSM mixer branches: 1049.7 ms; attention mixer branches: 187.8 ms.
- Existing MoE total: 2071.6 ms, including gate/up 1104.6 ms and down 628.0 ms.

These are diagnostic wall-clock categories, not an acceptance benchmark or
a complete partition of prompt time. In particular, the SSM branch includes
its projections, not only the recurrent kernel. Profiling stayed in `/tmp`.

A separate temporary Q4_K AVX2 source replaced the four software half-to-float
conversion calls in unpacked SDOT paths with F16C. The AVX2 quant suite passed
(`/tmp/q4kf16c-test.log`). An isolated five-run pp128 comparison at eight threads,
with preparation and loop abort disabled, measured 35.60 tok/s against 56.85
tok/s for matching-ISA llama.cpp: ratio 0.626, below the 0.85 gate
(`/tmp/q4kf16c-speed.log`, exit 1). Samples were 35.60, 35.54, 36.00, 35.17,
and 35.86 tok/s. This is effectively unchanged from the previous production
median of 35.54; the experiment was not promoted or token-gated. Production
sources and binaries remain at the Q5_1 integration checkpoint. All diagnostic
and benchmark processes completed.

### SSM projection breakdown and Q8 tile experiment

Further temporary AVX2 instrumentation separated the 36 SSM layers' work
(`/tmp/ssmcost.err`, pp128, eight threads, preparation disabled). Input
projections took 465.7 ms, convolution dispatches 23.3 ms, Q/K normalization
7.3 ms, recurrent delta dispatches 334.7 ms, gate dispatches 43.7 ms, and output
projection 147.5 ms. Prompt time was 4005.0 ms. Dispatch timings include worker
coordination; this diagnostic is not a throughput gate. Input timing also
includes the associated debug-dump calls; scalar alpha/beta transforms and
residual combination are outside these categories.

GGUF inspection confirms the first SSM layer's large QKV, gate, and output
projections are Q8_0 (2560→10240, 2560→6144, and 6144→2560 respectively), while
alpha/beta projections are F32. A temporary Q8_0 AVX2 batch-kernel experiment
reduced the token tile from eight to four without changing per-output
arithmetic order. The full AVX2 quant suite passed, including bitwise batch
versus GEMV comparisons over token counts and threaded execution
(`/tmp/q8tile4-test.log`).

The isolated five-run pp128 gate at eight threads, preparation and loop abort
disabled, measured 35.34 tok/s versus matching-ISA llama.cpp's 56.76: ratio
0.623, below 0.85 (`/tmp/q8tile4-speed.log`, exit 1). Samples: 35.37, 35.14,
35.34, 35.47, 33.90. The four-token tile offered no gain over the production
35.54 median and was not promoted or model-token-gated. Production sources
and binaries remain unchanged. All processes completed. These measurements
favor investigating projection execution and MoE gate/up work, not assuming
the recurrent SSM kernel accounts for the full mixer cost.

### Unpacked Q4_K GEMV-order batch scheduling candidate

Inspection found that `bn_quant_matmul_prepared_multi_gemv` batches prepared
Q4_K weights but falls back to a dispatch per token for unpacked Q4_K.
A temporary `/tmp/rawq4batch-matmul.c` candidate quantizes all activation rows
once, shares them between projections, and dispatches physical output-row
ranges once. Its callback invokes the existing four-row GEMV kernel for each
token; no dot-product arithmetic changes. Prepared layouts, forced-float
policy, and AVX512 reference-dot overrides retain existing paths. This is
not yet production code; integration still needs a native-batch policy guard,
final fallback/overflow review, and normal production build/test validation.

The full quant suite passed for explicit Clang AVX2 and AVX512 builds, including
exact multi-GEMV tests against per-token calls with partial rows, multiple
matrices, mixed prepared layouts, and threading. Logs are
`/tmp/rawq4batch-test.log` and `/tmp/rawq4batch-test512.log`. The initial AVX512
test link omitted `-mavx512dq`; the corrected full-ISA test build passed.
Production baseline `make -j8 test` also passed (`/tmp/rawq4batch-baseline.log`).

Qwen3.8 Flash Next strict gates, eight prompts, 24-token cap, eight threads,
maxseq 512, matching ISA, F32 KV, flash off, preparation and loop abort disabled:

- AVX2: 176/176 sampled IDs, equal counts, PASS (`/tmp/rawq4batch-192.log`).
- AVX512: 179/179 sampled IDs, equal counts, PASS (`/tmp/rawq4batch-512-192.log`).

Isolated pp128, five-run benchmarks with the same runtime overrides:

- AVX2: 38.04 versus 56.37 tok/s, ratio 0.675. Samples 38.05, 38.64, 38.04,
  37.29, 37.56 (`/tmp/rawq4batch-speed.log`).
- AVX512: 38.79 versus 62.78 tok/s, ratio 0.618. Samples 38.38, 38.58, 38.79,
  39.54, 39.04 (`/tmp/rawq4batch-speed512.log`).

Both speed gates exited 1: still below 0.85, despite higher medians than the
previous production 35.54/35.05. Reference benchmark KV defaults to F16;
bitnet uses F32. Decode speed was not remeasured for this candidate.
Candidate SHA256s: AVX2
`8fbe8db7e165addaf88a3d2a8738fe296b6924e66a36ff70e63cfce9955c9f2d`;
AVX512 `ae833b5de44be9eeb0b874ef3db6b8d09c5f88477bcfc5d93bda4ed3bd9875f4`.
All processes completed; production binaries remain at the Q5_1 checkpoint.

### Unpacked Q4_K GEMV-order batching integration

The scheduling candidate above is now integrated in `src/quant/matmul.c`.
Homogeneous unpacked Q4_K batches share Q8_K activation quantization and one
physical-row dispatch, invoking the existing four-row GEMV kernel per token.
The new path honors the native-batch disable switch, forced-float override,
and AVX512 reference-dot override. Mixed/prepared layouts and non-x86 routes
are unchanged. Allocation failures retain per-token GEMV fallback; allocation
sizes are bounded by the quantized input byte count. There are no API changes,
model-specific conditions, persistent buffers, or backend ownership changes.

`test_q4k_unpacked_gemv_batch_policy` adds bitwise per-token comparisons with
one/two/four/five matrices, row counts 137/143/1/4/7, null and threaded pools,
native batching disabled, forced-float, and reference-dot policies. It checks
output sentinels and restores its environment. Existing prepared/mixed-layout
coverage also passes. Validation:

- Baseline `make -j8 test`: `/tmp/rawq4integrate-baseline.log`, exit 0.
- Clean default build and full tests: `/tmp/rawq4integrate-clean-build.log`
  and `/tmp/rawq4integrate-full-test.log`, exit 0, zero warnings.
- Explicit Clang AVX2/AVX512 quant suites:
  `/tmp/rawq4integrate-test2.log`, `/tmp/rawq4integrate-test512.log`, exit 0.
- AVX2 ASan/UBSan quant suite: `/tmp/rawq4integrate-sanitize-approved.log`,
  exit 0. Initial sandbox execution hit LeakSanitizer's ptrace restriction;
  the approved unsandboxed run completed successfully.

Final production binaries (`bitnet_avx2`/`bitnet_avx512`, also saved as
`/tmp/rawq4integrate-avx2` and `/tmp/rawq4integrate-avx512`) were independently
token-gated on Qwen3.8 Flash Next: eight prompts, 24-token cap, eight threads,
maxseq 512, matching ISA, F32 KV on both sides, flash off. With preparation
and loop abort disabled, AVX2 matched 176/176 sampled IDs and AVX512 179/179,
both with equal token counts and strict PASS. Logs:
`/tmp/rawq4integrate-avx2-192.log`, `/tmp/rawq4integrate-avx512-192.log`.

Isolated five-run pp128/tg32 measurements of those exact binaries, eight
threads, same runtime overrides, matching-ISA llama-bench, flash off:

| Backend | Decode bitnet/reference tok/s | Decode ratio | Prefill bitnet/reference tok/s | Prefill ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 7.06 / 7.18 | 0.983 PASS | 38.04 / 57.02 | 0.667 FAIL |
| AVX512 | 6.89 / 6.93 | 0.994 PASS | 38.69 / 62.88 | 0.615 FAIL |

Logs `/tmp/rawq4integrate-avx2-speed32.log` and
`/tmp/rawq4integrate-avx512-speed32.log` both exit 1 because prefill misses
0.85. AVX2 prefill samples: 37.06, 38.29, 38.04, 38.29, 37.13; AVX512:
39.26, 37.73, 37.81, 39.49, 38.69. Decode samples: AVX2 7.07, 7.11, 7.06,
7.06, 7.06; AVX512 6.88, 6.89, 6.89, 6.89, 6.89. Reference benchmark KV
defaults to F16, bitnet to F32. Prefill medians improve on the previous
production 35.54/35.05, but neither backend meets the prefill target.

SHA256: AVX2 `2fea066b20b863d18d763478cd79f74ddb2be7f924968ce93e280a37b524d465`;
AVX512 `214571925ffd5b521d95f84fdf3385587f753ca1b4f5a3bc9931217ea9be3d7f`.
All processes completed. Other model fixtures, default preparation settings,
and CUDA have not been re-gated by this integration; the full goal remains open.

### Reuse decoded Q4_K weights in GEMV-order batches

The unpacked callback now routes to the existing
`bn_quant_q4k_avx2_sdot_matmul_4row_range`, reusing decoded weights across
tokens instead of repeatedly invoking the single-token kernel. Both retain
the same per-output integer dot, scalar FMA accumulation, and final subtraction.
Only the quant callback changed; policy guards and all model/backend paths
remain as in the preceding integration. The policy test now uses nine tokens
to cover the eight-token tile and a tail. Existing tests cover other counts,
mixed layouts, partial row groups, and threading.

Two AVX2 candidates were measured at pp128/t8, five isolated runs, preparation
and loop abort disabled, matching-ISA llama-bench, flash off:

- Existing batch kernel: 39.37 versus 57.19 tok/s, ratio 0.688; samples
  37.84, 39.59, 39.52, 39.18, 39.37 (`/tmp/q4kreuse-speed.log`).
- A separate compact-float accumulator experiment: 38.43 versus 57.22,
  ratio 0.672; samples 39.19, 39.29, 37.55, 38.43, 38.31
  (`/tmp/q4compact-speed.log`). Quant tests passed, but this slower variant
  was rejected; no accumulator representation changed in production.

The existing-kernel candidate also measured AVX512 39.46 versus 62.54 tok/s,
ratio 0.631; samples 39.47, 38.16, 39.46, 39.40, 40.06
(`/tmp/q4kreuse-speed512.log`). All prefill gates exit 1, below 0.85. These
are modest gains over the prior production 38.04/38.69, not achievement of
prefill parity. Bitnet benchmark KV is F32; reference defaults to F16.

Strict eight-prompt, 24-token-cap gates on Qwen3.8 Flash Next at t8/maxseq512,
F32 KV on both sides, flash off, preparation and loop abort disabled passed:
AVX2 176/176 and AVX512 179/179 sampled IDs, equal counts. Logs are
`/tmp/q4kreuse-avx2-192.log` and `/tmp/q4kreuse-avx512-192.log`. Both explicit
ISA quant suites passed before and after integration. Final validation logs:

- `/tmp/q4reuseintegrate-baseline.log`: baseline full tests, exit 0.
- `/tmp/q4reuseintegrate-clean-build.log` and
  `/tmp/q4reuseintegrate-full-test.log`: clean build/full tests, exit 0,
  zero warnings.
- `/tmp/q4reuseintegrate-test-avx2.log` and
  `/tmp/q4reuseintegrate-test-avx512.log`: final ISA quant suites, exit 0.
- `/tmp/q4reuseintegrate-sanitize-approved.log`: AVX2 ASan/UBSan suite,
  exit 0 after approved execution outside the sandbox's LSAN ptrace restriction.

Final production builds were compared against the token-gated/speed-tested
candidates: `readelf -lW` program headers are identical, and `objcopy -O binary
--remove-section=.note.gnu.build-id` images compare byte-for-byte equal on
both ISAs. Full-file hashes differ because of build metadata/source filenames;
the allocated executable content is identical. This comparison, rather than
a second token run of the same code/data, links the gates to the final builds.
Production binaries are restored as `bitnet_avx2` and `bitnet_avx512`, with
copies `/tmp/q4reuseintegrate-avx2` and `/tmp/q4reuseintegrate-avx512`.

SHA256 final AVX2: `ee28cde4c1b998eea7c0c0c344be1b11d3bb68eba9557bb34ed5d0d625c98189`;
final AVX512: `916f210e4254d8396d7f3abb454944034a6d47769fa4f5c58f07b13bc3b6c482`.
Candidate AVX2: `746f6f31c641de47a4761de6e973fc7e1a60261a484b2120997466629fe07c48`;
candidate AVX512: `84523d2d08f4062e31930be6641b1f7ac1b892148651c9a93fa767f6aba6ff14`.
Decode speed was not remeasured in this step. All processes completed; broader
model/default-runtime/CUDA gates remain open.

### Q5_1 four-token weight-reuse candidate

A fresh profile of the Q4_K-reuse production AVX2 binary, preparation disabled,
pp128/t8, completed in `/tmp/q4reuse-current-profile.err`: prompt 3724.2 ms,
MoE gate/up 745.3 ms, down 696.8 ms, route 146.1 ms, shared 84.4 ms, MoE total
1787.5 ms. This diagnostic is not a throughput gate. It identifies the Q5_1
down projection as a substantial remaining cost after the Q4_K improvements.

Temporary `/tmp/q51reuse-kernel.c` adds a four-token tile to Q5_1 matmul.
Each token retains independently rounded Q8_1 d/s metadata and its original
eight FMA lanes plus scalar offset; decoded weight bytes and FP16 scales are
shared across the tile. The single-token kernel and all production sources
remain unchanged. This candidate still needs shared input-quantization code,
bounded scratch-memory handling for unusually wide inputs, and production
integration validation before promotion.

Full explicit AVX2 and AVX512 quant suites passed
(`/tmp/q51reuse-test.log`, `/tmp/q51reuse-test512.log`). Strict Qwen3.8 Flash
Next gates also passed on eight prompts, 24-token cap, t8/maxseq512, matching
ISA, F32 KV on both sides, flash off, preparation and loop abort disabled:
AVX2 176/176 sampled IDs and AVX512 179/179, with equal token counts
(`/tmp/q51reuse-avx2-192.log`, `/tmp/q51reuse-avx512-192.log`).

Isolated pp128/t8, five-run measurements with the same runtime overrides:

- AVX2 41.07 versus 56.05 tok/s, ratio 0.733; samples 41.40, 39.90, 41.79,
  41.07, 39.69 (`/tmp/q51reuse-speed.log`).
- AVX512 40.68 versus 62.04 tok/s, ratio 0.656; samples 40.68, 40.12, 42.24,
  40.62, 41.32 (`/tmp/q51reuse-speed512.log`).

Both gates exit 1 because prefill is below 0.85. Reference benchmark KV is
default F16; bitnet is F32. Decode speed was not remeasured. Candidate SHA256s:
AVX2 `a9c0fcdbac9a2b13c3c33fcae1651448867b6133e36aa44b078d4b60931c4039`;
AVX512 `afdbee9c3c4575df735f0f73805f1b42ebb1c7bfc772f6b4859037e2ae12536d`.
All processes completed. Production remains at Q4_K weight reuse, and the
full cross-model/backend goal remains open.

### Q5_1 four-token weight-reuse integration

The candidate above is integrated in `src/quant/q4_1_avx2.c`. Single-token
and batch paths share one Q8_1 input-quantization helper. The batch path reuses
decoded Q5_1 weights across four tokens, with independent eight-lane FMA and
offset accumulators. Tiled scratch arrays stay below 42 KiB per worker;
matrices wider than 256 blocks (8192 columns) retain the original single-token
scratch footprint. This is a bound on the new tile, not a new global limit
on all quant scratch. Empty ranges and nonpositive token counts return before
allocation, and token advancement uses the actual tile size. No public API,
model-specific condition, runtime policy, or backend ownership changes.

The exact Q5_1 suite now covers nine tokens and block counts 1/3/20/80/256/257,
including the tile's width boundary, wide fallback, full tiles and tails,
partial rows, zero-token calls, native-batch override, threaded execution,
and output guards. Validation completed:

- Baseline full tests: `/tmp/q51tile-baseline.log`, exit 0.
- Clean default build/full tests: `/tmp/q51tile-clean-build.log` and
  `/tmp/q51tile-full-test.log`, exit 0, zero warnings.
- Explicit AVX2/AVX512 quant suites: `/tmp/q51tile-test-avx2.log` and
  `/tmp/q51tile-test-avx512.log`, exit 0.
- AVX2 ASan/UBSan suite: `/tmp/q51tile-sanitize-approved.log`, exit 0 after
  approved execution outside the sandbox's LeakSanitizer ptrace restriction.
- Q5_1 module compiled with `-mno-f16c`, linked into the AVX2 test suite:
  `/tmp/q51tile-no-f16c-test.log`, exit 0. This checks the module fallback,
  not an engine-wide claim of operation without F16C.

The final integrated binaries were freshly strict-token-gated on Qwen3.8 Flash
Next: eight prompts, 24-token cap, t8/maxseq512, matching ISA, F32 KV on both
sides, flash off, preparation and loop abort disabled. AVX2 matched 176/176
and AVX512 179/179 sampled IDs, equal counts, both PASS
(`/tmp/q51tile-avx2-192.log`, `/tmp/q51tile-avx512-192.log`).

Final isolated five-run pp128/tg32 gates, same overrides and thread count:

| Backend | Decode bitnet/reference tok/s | Decode ratio | Prefill bitnet/reference tok/s | Prefill ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 7.04 / 7.04 | 1.000 PASS | 40.98 / 56.88 | 0.720 FAIL |
| AVX512 | 6.92 / 7.13 | 0.971 PASS | 41.30 / 62.68 | 0.659 FAIL |

Both `/tmp/q51tile-avx2-speed32.log` and `/tmp/q51tile-avx512-speed32.log`
exit 1 because prefill misses 0.85. Prefill samples: AVX2 41.64, 41.44, 40.75,
40.86, 40.98; AVX512 40.30, 41.86, 41.63, 39.78, 41.30. Decode samples:
AVX2 7.05, 7.08, 7.04, 7.04, 7.04; AVX512 6.92, 6.93, 6.92, 6.91, 6.91.
Reference benchmark KV is default F16; bitnet is F32. These are improvements
over prior prefill medians 39.37/39.46, not completion of prefill parity.

Production binaries are `bitnet_avx2` and `bitnet_avx512`, with copies at
`/tmp/q51tile-avx2` and `/tmp/q51tile-avx512`. SHA256 AVX2:
`93658d6f0be4a0d92ef13dff416943d912ea73f204fdaa598b202febd30d6cd5`;
AVX512 `dd9981b28da66fe078d187a7dcf1d90ac12cc2132214533a8794de7aecad0f69`.
All processes completed. Other model fixtures, default preparation settings,
and CUDA have not been re-gated by this step; the full goal remains open.

### Q8_0 AVX512VL and sixteen-token tiling candidate

Dispatch inspection confirmed AVX512 Q8_0 matmul uses its VNNI four-row
kernel, not `bn_quant_q8_avx2_matmul_range`. An initial temporary AVX2 tile
edit built under AVX512 was therefore not benchmarked or promoted: it did
not change the active route. The following experiments instead modify the
active AVX512 implementation only, leaving production unchanged.

Temporary `/tmp/q8vl-kernel.c` replaces 512-bit VNNI operations whose upper
result lanes are unused with equivalent 256-bit VNNI operations. Disassembly
confirms ZMM `vpdpbusd` becomes YMM `vpdpbusd`; eight FMA/reduction lanes remain
unchanged. A second candidate, `/tmp/q8vl16-kernel.c`, additionally replaces
the full-tile two-row/eight-token loop with one row/sixteen tokens, still
using sixteen accumulators. Remainders 1–7 retain specialized paths and 8–15
use the generic tile path. Both full AVX512 quant suites pass
(`/tmp/q8vl-test.log`, `/tmp/q8vl16-test.log`), including batch/GEMV bitwise
checks over token counts and partial rows.

Isolated Qwen3.8 Flash Next pp128/t8, five-run benchmarks, preparation and
loop abort disabled, matching-ISA llama-bench, flash off:

- Narrow VNNI only: 41.90 versus 62.84 tok/s, ratio 0.667; samples 40.97,
  42.19, 41.90, 41.90, 41.18 (`/tmp/q8vl-speed.log`).
- Narrow VNNI plus sixteen-token tile: 42.76 versus 62.67 tok/s, ratio 0.682;
  samples 43.53, 42.87, 42.76, 42.31, 42.51 (`/tmp/q8vl16-speed.log`).

Both gates exit 1, still below 0.85. Prior production prefill was 41.30 tok/s.
Reference benchmark KV defaults to F16, bitnet uses F32. Decode speed was not
remeasured. The combined candidate passed strict sampled-token parity:
179/179 IDs, equal counts, eight prompts, 24-token cap, t8/maxseq512, F32 KV
on both sides, flash off, same overrides (`/tmp/q8vl16-192.log`, exit 0).

Candidate SHA256s: narrow-only
`90da482f613bf3abd3a36fb0738df795c0fb701e4bf583c963389330de5aadbf`;
combined `f06aeecccb7381b0db3a3f086a1dd9ce84a6af0c4e928b0e4643d9367daeacb7`.
Before promotion, the new dot helper must preserve a 512-bit fallback for
builds without AVX512VL, the loop comments must describe the new layout, and
production tests/build/sanitizers and final-artifact gates must be validated.
All processes completed. Production remains at Q5_1 tiling; the broader
cross-model/backend goal remains open.

### Q8_0 AVX512VL/sixteen-token integration and Qwen3.6 refresh

The combined candidate is integrated in `src/quant/q8_avx512_vnni.c`.
A format-local helper uses 256-bit VNNI when AVX512VL is compiled in and
otherwise returns the lower eight lanes of the original 512-bit operation.
Full tiles process one row across sixteen tokens; short and generic remainder
paths retain their arithmetic. Tile advancement uses the actual remaining
count. Model/runtime APIs and non-AVX512 execution paths are unchanged.

The existing Q8 batch/GEMV regression now also compares each expected output
directly against the independent AVX2 four-row kernel, covering token counts,
partial rows, and finite half-scale boundaries. Validation:

- Baseline tests `/tmp/q8vl-integrate-baseline.log`, exit 0.
- Clean build `/tmp/q8vl-integrate-clean-build.log`, exit 0, zero warnings.
- Final full suite `/tmp/q8vl-integrate-full-test-final.log`, exit 0, zero
  warnings. An initial missing-field initializer warning in the added test
  was fixed before this final run.
- Explicit AVX512 quant suite `/tmp/q8vl-final-test-native.log`, exit 0.
- Q8 module compiled with `-mno-avx512vl`, linked into the AVX512 quant suite:
  `/tmp/q8vl-final-test-no-vl.log`, exit 0. Disassembly confirms ZMM VNNI in
  that module and YMM VNNI in the normal build. This is module fallback
  coverage, not an engine-wide no-VL claim.
- AVX512 ASan/UBSan `/tmp/q8vl-integrate-sanitize-approved.log`, exit 0 after
  approved execution outside the sandbox's LSAN ptrace restriction.

The final AVX512 binary passed a fresh strict Qwen3.8 Flash Next gate:
179/179 sampled IDs, equal counts, eight prompts, 24-token cap, t8/maxseq512,
matching ISA, F32 KV on both engines, flash off, preparation and loop abort
disabled (`/tmp/q8vl-integrate-192.log`). The rebuilt AVX2 binary is byte-for-byte
identical to the prior Q5_1 checkpoint; its gates were not redundantly rerun.

Final isolated five-run Qwen3.8 pp128/tg32 at t8 with the same overrides:
decode 6.94 versus 7.33 tok/s (0.947 PASS); prefill 42.33 versus 62.53
(0.677 FAIL). Prefill samples 43.54, 43.63, 42.33, 41.37, 42.32; decode
6.94, 6.94, 6.94, 6.96, 6.94. Log
`/tmp/q8vl-integrate-avx512-speed32.log` exits 1 because prefill misses 0.85.

The same final binary was then checked on Qwen3.6 35B-A3B Q8_0 abliterated,
with default preparation/cache settings, flash off, matching-ISA llama-bench,
pp128/tg64 and five isolated runs per configuration:

| Threads on both engines | Decode bitnet/reference tok/s | Decode ratio | Prefill bitnet/reference tok/s | Prefill ratio |
| --- | --- | --- | --- | --- |
| 8 | 13.78 / 14.06 | 0.980 PASS | 111.89 / 132.01 | 0.848 FAIL |
| 12 | 20.82 / 19.73 | 1.055 PASS | 125.81 / 174.30 | 0.722 FAIL |

Eight-thread prefill samples: 115.99, 107.29, 115.83, 110.43, 111.89; decode:
13.74, 13.77, 13.81, 13.78, 13.87. Twelve-thread prefill: 117.60, 123.12,
127.03, 125.81, 129.87; decode: 20.99, 20.95, 20.82, 20.79, 20.78.
Logs `/tmp/q8vl-q36-speed.log` and `/tmp/q8vl-q36-t12-speed.log` both exit 1.
Do not round the 0.848 result into acceptance. Twelve threads improve absolute
speed but not the prefill ratio. Qwen3.6 token parity was not refreshed in
this step. All speed gates use bitnet F32 KV versus reference default F16 KV.

Production binaries are restored, with copies `/tmp/q8vl-integrate-avx2`
and `/tmp/q8vl-integrate-avx512`. SHA256 AVX2:
`93658d6f0be4a0d92ef13dff416943d912ea73f204fdaa598b202febd30d6cd5`;
AVX512 `ddd5fe747179c23dfedb4448f29698c794f40d5b11cef8a2163f1c2c2d0d1aff`.
All processes completed. Qwen3.8/Qwen3.6 prefill and the broader requested
cross-model/backend gates remain open.

### Fine-grained SSM prefill scheduling prototype

Thread-pool inspection found that at eight threads a 48-item task exceeds
the `4 * threads` fine-grain cutoff and receives chunks of 32: only two
chunks can run concurrently. A broad temporary experiment changed the cutoff
to `threads * TP_CHUNK_MIN`. It passed the thread-pool tests and measured
AVX2 pp128/t8 at 43.93 versus 55.92 tok/s (0.786), samples 45.78, 43.80,
43.93, 43.93, 45.00 (`/tmp/tpfine-speed.log`, exit 1). This broad policy
change is diagnostic only and is not proposed for production.

Sequential, otherwise matched temporary profiles on current AVX2 code
(`/tmp/tpcost-base.err`, `/tmp/tpcost-fine.err`) localized the improvement:

| SSM dispatch category | Default ms | Fine-small-task ms |
| --- | --- | --- |
| Input projections | 474.8 | 418.6 |
| Convolution | 26.6 | 23.1 |
| Q/K normalization | 8.4 | 7.0 |
| Recurrent delta | 346.4 | 84.8 |
| Output gate | 44.8 | 18.8 |
| Output projection | 150.0 | 151.0 |

Prompt time was 3458.6 versus 3109.6 ms; MoE totals were essentially unchanged,
1532.2 versus 1527.7 ms. These single-run diagnostic timings include dispatch
coordination and are not independent acceptance gates.

A scoped prototype (`/tmp/ssmfine-threadpool.c`, `/tmp/ssmfine-prefill.c`)
adds an opt-in fine dispatch that publishes a per-dispatch flag through the
existing pool synchronization. Default dispatch resets the flag and keeps its
original policy. Only prefill SSM recurrent-delta and output-gate tasks opt in.
Both default-dispatch and fine-dispatch variants of the thread-pool suite pass
(`/tmp/ssmfine-test.log`, `/tmp/ssmfine-fine-test.log`). This remains temporary;
production integration needs a declared runtime API, policy-reset/range tests,
and backend-owned x86 opt-in so ARM/Metal scheduling remains unchanged.

Strict Qwen3.8 Flash Next token gates passed on eight prompts, 24-token cap,
t8/maxseq512, matching ISA, F32 KV on both sides, flash off, preparation and
loop abort disabled: AVX2 176/176, AVX512 179/179 sampled IDs, equal counts
(`/tmp/ssmfine-avx2-192.log`, `/tmp/ssmfine-avx512-192.log`).

Isolated five-run pp128/t8 speed gates with the same runtime overrides:

- Scoped AVX2: 43.79 versus 56.43 tok/s, ratio 0.776; samples 43.75, 43.51,
  43.79, 44.00, 44.12 (`/tmp/ssmfine-speed.log`).
- Scoped AVX512: 47.97 versus 61.63 tok/s, ratio 0.778; samples 46.79, 44.72,
  48.03, 48.47, 47.97 (`/tmp/ssmfine-speed512.log`).

Both gates exit 1: still below 0.85, despite improvements over production
40.98/42.33. Reference benchmark KV is default F16; bitnet is F32. Decode
speed and broader fixtures were not re-gated for the prototype. SHA256 AVX2:
`ed7b1cac8634929bf6cc0660d56e676ec521cd8c176510ec66d4be2b2385c68a`;
AVX512 `4525688826956d9815afa7169e54a2a210e903ea62d69dc2aa81996c694da071`.
All processes completed. Production remains at the Q8_0 AVX512VL integration;
the full cross-model/backend goal remains open.

### Backend-selected fine-grained SSM prefill integration

The scoped scheduling prototype is integrated. `bn_tp_dispatch_fine()` is
an opt-in runtime API: with a non-NULL pool each claimed range has one item;
the NULL-pool fallback remains serial over the whole range. Its per-dispatch
flag is published alongside task state before the generation release, and
normal dispatch explicitly resets it. Default chunking and decode call sites
are unchanged.

`BnPrefillCPUOps` selects the dispatcher for independent SSM recurrent-update
and output-gate heads. AVX2/AVX512 select fine dispatch; NEON, WASM, and scalar
select normal dispatch. This avoids model-specific conditions and preserves
the existing non-x86 scheduling policy. No model/session ownership changes.

New thread-pool tests alternate fine and normal dispatch across null, serial,
four-thread and eight-thread pools. Atomic per-item counters verify exact
coverage, sentinels verify bounds, recorded range widths verify fine behavior
and policy reset, and empty/multi-task calls are included. The transformer
test checks the CPU-backend dispatcher selection. Validation:

- Baseline `/tmp/ssmgrain-baseline.log`, exit 0.
- Clean build/full tests `/tmp/ssmgrain-clean-build.log` and
  `/tmp/ssmgrain-full-test.log`, exit 0, zero warnings.
- Thread-pool ASan/UBSan `/tmp/ssmgrain-threadpool-sanitize-approved.log`,
  exit 0.
- AVX2 synthetic Qwen/SSM/prefill state ASan/UBSan
  `/tmp/ssmgrain-state-sanitize-approved.log`, exit 0, including all-logit
  versus last-logit state consistency. Both sanitizer runs used approved
  execution outside the sandbox after LSAN's ptrace restriction.

Fresh strict gates on the integrated binaries passed for Qwen3.8 Flash Next:
AVX2 176/176 and AVX512 179/179 sampled IDs, equal counts, eight prompts,
24-token cap, t8/maxseq512, matching ISA, F32 KV on both engines, flash off,
preparation and loop abort disabled (`/tmp/ssmgrain-avx2-192.log`,
`/tmp/ssmgrain-avx512-192.log`).

Final isolated five-run pp128/tg32 gates with the same runtime overrides:

| Backend | Decode bitnet/reference tok/s | Decode ratio | Prefill bitnet/reference tok/s | Prefill ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 7.05 / 7.10 | 0.993 PASS | 44.04 / 56.80 | 0.775 FAIL |
| AVX512 | 6.91 / 7.01 | 0.986 PASS | 48.20 / 62.23 | 0.775 FAIL |

Logs `/tmp/ssmgrain-avx2-speed32.log` and `/tmp/ssmgrain-avx512-speed32.log`
both exit 1 because prefill is below 0.85. Prefill samples: AVX2 43.67, 44.92,
42.86, 44.04, 44.50; AVX512 48.43, 47.64, 48.92, 48.20, 47.23. Decode:
AVX2 7.28, 7.05, 7.05, 7.04, 7.04; AVX512 7.20, 6.90, 6.91, 6.91, 6.92.
Speed KV remains bitnet F32 versus reference default F16. Production prefill
improves from 40.98/42.33, but neither backend meets the prefill gate.

Production binaries are restored, with copies `/tmp/ssmgrain-avx2` and
`/tmp/ssmgrain-avx512`. SHA256 AVX2:
`bb1b87f49407ea4ead703bd0db2270c4a9c727161832efab0c716cc8c7228b43`;
AVX512 `ac1d64a8b435f09cfacb2b31808b261765449784b9c4de8d490b6d9ac241ae58`.
All processes completed. Other fixtures, default preparation settings, and
CUDA were not re-gated here; the full requested goal remains open.

### F32 batch scheduling diagnostic (not integrated)

Following the SSM-head integration, a temporary copy of `src/quant/matmul.c`
changed only the AVX2/AVX512 F32 batch branch from `bn_tp_dispatch()` to
`bn_tp_dispatch_fine()`. Production sources and binaries remain unchanged.
The baseline full suite passed (`/tmp/f32grain-baseline.log`, exit 0), and
both temporary Clang builds completed without warnings.

Qwen3.8 Flash Next strict gates passed on the temporary binaries:
AVX2 176/176 and AVX512 179/179 sampled IDs, equal token counts, eight prompts,
24-token cap, t8/maxseq512, matching ISA, F32 KV on both engines, flash off,
preparation and loop abort disabled. Logs:
`/tmp/f32grain-avx2-parity.log`, `/tmp/f32grain-avx512-parity.log` (exit 0).

Isolated five-run pp128/t8 measurements with those runtime overrides:

| Backend | bitnet / reference tok/s | Ratio | bitnet samples |
| --- | --- | --- | --- |
| AVX2 | 45.70 / 56.65 | 0.807 FAIL | 43.24, 46.05, 45.69, 46.19, 45.70 |
| AVX512 | 48.13 / 61.42 | 0.784 FAIL | 48.55, 47.86, 47.88, 48.13, 49.41 |

Logs `/tmp/f32grain-avx2-speed.log`, `/tmp/f32grain-avx512-speed.log`, both
exit 1. Matching-ISA llama-bench, flash off, bitnet F32 KV versus reference
default F16 KV. AVX2 improves over the previous 44.04 tok/s checkpoint;
AVX512 is essentially unchanged from 48.20. Neither passes 0.85. These are
separate runs, not an interleaved end-to-end A/B test; decode was not refreshed.

A standalone diagnostic (`/tmp/f32grain-bench.c`) alternates normal and fine
scheduling on the unchanged F32 kernel with an eight-thread pool. Each result
is compared bitwise against serial execution and checked for sentinel damage.
After four warm-up pairs, it averages 100 pairs (12 for 4096 rows):

| Rows × columns, tokens | AVX2 normal / fine ms | AVX512 normal / fine ms |
| --- | --- | --- |
| 48 × 2560, 128 | 1.0878 / 0.2162 | 0.5390 / 0.1348 |
| 512 × 2560, 128 | 2.2345 / 2.2206 | 1.0889 / 1.0831 |
| 4 × 10240, 128 | 0.1566 / 0.1567 | 0.0918 / 0.0915 |
| 48 × 2560, 2 | 0.0166 / 0.0049 | 0.0074 / 0.0049 |
| 137 × 33, 9 | 0.0013 / 0.0050 | 0.0048 / 0.0123 |
| 512 × 32, 2 | 0.0012 / 0.0103 | 0.0030 / 0.0232 |
| 4096 × 2560, 128 | 18.0266 / 17.7014 | 8.7331 / 8.5963 |

Both microbenchmarks passed every exactness/bounds check and exited 0
(`/tmp/f32grain-micro-avx2.log`, `/tmp/f32grain-micro-avx512.log`). They ran
sequentially after all model inference completed. These cache-resident timings
diagnose scheduling, not model throughput. Unconditional fine F32 dispatch is
not suitable: expensive short-row projections benefit, but tiny batches
regress substantially. A subsequent candidate should select dispatch by work
size in quant/runtime code, without model-family conditions. Other fixtures,
default preparation, and CUDA remain unverified by this experiment.

### Work-size-gated F32 batch scheduling integration

The unconditional F32 experiment above is replaced by a conservative quant
policy: fine dispatch only for parallel x86 batches with at least 2048 input
columns, at least 32768 products per output row across the batch, and at most
eight output rows per thread. Shape arithmetic uses widened multiplication and
division to avoid overflow. The policy lives in `src/quant/policy.c`; only the
existing x86 F32 batch branch calls it. Kernels, decode, model ownership, and
ARM/Metal scheduling are unchanged.

Boundary experiments rejected a cost-only 4096-product rule (137/512-row
small batches slowed down), then rejected a 32768-product rule without a
minimum column count (48-row, 128-column, 256-token batches slowed down).
The latter is consistent with adjacent-output cache-line contention; that
cause is an inference, not a hardware-counter measurement. The final scoped
rule avoids those cases. Alternating normal/scoped-dispatch microbenchmarks
on the unchanged kernel, t8, exact serial comparison and sentinels, measured:

| Rows × columns, tokens | AVX2 normal / scoped ms | AVX512 normal / scoped ms |
| --- | --- | --- |
| 48 × 2560, 128 | 1.1045 / 0.2189 | 0.5247 / 0.1551 |
| 48 × 2047, 17 (ineligible) | 0.1112 / 0.1109 | 0.0463 / 0.0455 |
| 48 × 2048, 16 | 0.0984 / 0.0243 | 0.0358 / 0.0204 |
| 48 × 2049, 16 | 0.0988 / 0.0238 | 0.0385 / 0.0190 |
| 48 × 128, 256 (ineligible) | 0.0445 / 0.0448 | 0.0431 / 0.0423 |

Full results `/tmp/f32cost-scopedmicro-avx2.log` and
`/tmp/f32cost-scopedmicro-avx512.log`, exit 0; source `/tmp/f32cost-bench.c`.
These are cache-resident scheduling diagnostics, not throughput acceptance.

Production tests cover policy boundaries, invalid dimensions, INT_MAX inputs,
and bitwise output equality against direct serial kernel execution across
12 shapes with null, serial, four-thread and eight-thread pools. Three repeats
per shape/pool reset output sentinels, including both eligible/ineligible
batches and scalar column tails. Validation:

- `/tmp/f32cost-baseline.log`: full baseline, exit 0.
- `/tmp/f32cost-clean-build.log`, `/tmp/f32cost-full-test.log`: clean build
  and full suite, exit 0, zero warnings.
- `/tmp/f32cost-test-avx2.log`, `/tmp/f32cost-test-avx512.log`: Clang ISA
  quant suites, exit 0, zero build warnings.
- `/tmp/f32cost-sanitize-approved.log`: AVX2 quant ASan/UBSan, exit 0;
  approved unsandboxed rerun after LSAN's ptrace restriction.

Fresh Qwen3.8 Flash Next strict token gates passed: AVX2 176/176 and AVX512
179/179, equal counts, eight prompts, 24-token cap, t8/maxseq512, matching ISA,
F32 KV on both engines, flash off, preparation and loop abort disabled.
Logs `/tmp/f32cost-avx2-parity.log`, `/tmp/f32cost-avx512-parity.log`, exit 0.

Final isolated five-run pp128/tg32, t8 gates with those runtime overrides:

| Backend | Decode bitnet / reference tok/s | Ratio | Prefill bitnet / reference tok/s | Ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 7.05 / 7.06 | 0.999 PASS | 44.20 / 56.38 | 0.784 FAIL |
| AVX512 | 6.90 / 7.17 | 0.962 PASS | 47.76 / 62.18 | 0.768 FAIL |

Logs `/tmp/f32cost-avx2-speed32.log` and `/tmp/f32cost-avx512-speed32.log`,
exit 1. Prefill samples: AVX2 44.12, 45.68, 45.97, 44.20, 44.12; AVX512
47.47, 47.76, 46.75, 48.14, 48.44. Decode samples: AVX2 7.05, 7.05, 7.08,
7.05, 7.04; AVX512 6.91, 6.90, 6.90, 6.91, 6.90. Speed KV remains bitnet
F32 versus reference default F16. These separate model runs do not establish
an end-to-end improvement over the prior 44.04/48.20 checkpoint; the scoped
kernel gain is the stronger evidence. Prefill acceptance remains unmet.

Restored worktree binaries match `/tmp/f32cost-avx2` and
`/tmp/f32cost-avx512`, SHA256 respectively:
`7422cd4f6173ab5478c52c423a5a3c8617c96dc1d3431e74de54c2459223235b`,
`22fbb3dfde4b7190e432d065994b5017d8c48e6fb07248b659b1093794511187`.
All jobs finished. The broader model/backend matrix, default preparation,
and CUDA were not refreshed here. Re-profile the integrated prefill path
before choosing the next optimization; the full goal remains open.

### Integrated-path profile and token-parallel HC prototype

Re-profiled the current F32-cost-gated checkpoint using temporary prefill
instrumentation that preserves both integrated dispatch changes. Three
sequential pp128/t8 samples per ISA, token ID 1 repeated 128 times, preparation
and loop abort disabled, `BN_PREFILL_PROFILE=1`, maxseq512, one-token cap:

| Measurement (ms) | AVX2 sample range | AVX512 sample range |
| --- | --- | --- |
| Prompt | 2818.8–3141.9 | 2666.5–2742.3 |
| MoE total | 1393.0–1551.7 | 1304.9–1347.9 |
| MoE gate/up | 669.8–744.3 | 656.6–676.7 |
| MoE down | 419.5–472.6 | 401.3–421.8 |
| SSM input projections | 376.2–418.9 | 308.4–318.8 |
| SSM recurrent update | 82.7–89.6 | 98.4–103.4 |
| SSM output projection | 134.2–153.2 | 126.0–131.4 |

Logs `/tmp/currentcost-{avx2,avx512}-{1,2,3}.err`, all exit 0. This is a
profiling experiment, not a speed gate; the AVX512 one-token request sampled
EOG and returned zero generated tokens. No decode-rate claim is made.

A finer HC profile on AVX2 measured norm 79.607, down projection 149.433,
scaled SiLU 1.163, up projection 101.047, gate/mix 220.681, and injection
14.695 ms (`/tmp/currenthc-avx2.err`). This identifies substantial serial
work outside the MoE and SSM timings.

A temporary prototype schedules independent HC tokens in the normalization
and gate/mix loops using fine dispatch. Each token retains the original
stream traversal, reductions, expf, and accumulation order. It changes only
the existing batched-HC branch; production code is not modified. In an AVX2
profile, norm drops to 16.487 ms and gate/mix to 35.292 ms
(`/tmp/hctokens-avx2.err`). These are separate diagnostic runs, not paired
end-to-end measurements.

Timer-free binaries `/tmp/hctokens-clean-avx2` and
`/tmp/hctokens-clean-avx512` passed fresh Qwen3.8 Flash Next strict gates:
176/176 and 179/179 sampled IDs, equal counts, eight prompts, 24-token cap,
t8/maxseq512, matching ISA, F32 KV both, flash off, preparation and loop abort
disabled. Logs `/tmp/hctokens-clean-{avx2,avx512}-parity.log`, exit 0.
The existing synthetic Qwen/SSM/HC prefill-state suite and its AVX2 ASan/UBSan
build passed (`/tmp/hctokens-test.log`,
`/tmp/hctokens-state-sanitize-approved.log`); LSAN required the approved
unsandboxed rerun.

Isolated five-run pp128/tg32/t8 gates on the timer-free prototypes:

| Backend | Decode bitnet / reference tok/s | Ratio | Prefill bitnet / reference tok/s | Ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 7.12 / 7.02 | 1.014 PASS | 48.37 / 56.03 | 0.863 PASS |
| AVX512 | 6.90 / 7.01 | 0.984 PASS | 50.93 / 62.41 | 0.816 FAIL |

Logs `/tmp/hctokens-clean-avx2-speed32.log` (exit 0) and
`/tmp/hctokens-clean-avx512-speed32.log` (exit 1). AVX2 prefill samples:
48.37, 46.95, 48.79, 49.43, 45.25; AVX512: 48.50, 50.43, 51.85, 51.29,
50.93. Speed KV remains bitnet F32 versus reference default F16. These are
explicit preparation-disabled prototype results, not default-setting or
whole-matrix acceptance. Prototype SHA256 values respectively:
`3306a7adc54e68aab3b9d4e807e817e272bab5b67a67f1094183e47d7dae22d3`,
`1cdea16ab991bea85fc11bd47a929a784a4439b47cfbec11a6f38fbb99e97b13`.

**Integration deferred for a newly exposed threaded test discrepancy.**
The existing synthetic HC state fixture uses no explicit worker pool. A
temporary test copy installs an owned seven-worker pool after model load.
Both the prototype and unchanged production prefill then fail its
all-logits/last-logit comparison (exit 134):
`/tmp/hctokens-threaded.log`, `/tmp/hctokens-control.log`. A detailed control
run identifies `use_hc=1`, `final_ssm=0`, logit 0: last `-0.509162784`, all
`-0.506452382`, difference `-0.00271040201`
(`/tmp/hctokens-control-detail.log`). Thus this failure also exists without
the proposed parallel loops; its cause is not yet established. The test is
not weakened, and no production integration is claimed. Investigate pooled
HC prefill equivalence before promotion, then re-run explicit-thread state
tests. Temporary sources: `/tmp/hctokens-clean-prefill.c` and
`/tmp/hctokens-test-qwen36.c`. All jobs completed; production binaries remain
the F32-cost-gated checkpoint. The full requested goal remains open.

### Threaded state-test discrepancy resolved: malformed synthetic Q norm

The preceding HC integration hold was traced to the synthetic fixture, not
the HC token-parallel change. A zero-worker pool passed; seven workers failed
with preparation disabled too. Repeated failures also occurred with HC off,
and activation logging could hide them. Serializing only the single-row HC
fallback did not fix the failure.

ThreadSanitizer reported overlapping writes in
`batched_attn_naive_avx2_one()` at the attention output copy
(`/tmp/hcthread-tsan-noaslr.log`). Initial TSAN attempts failed with unexpected
memory mappings, including the approved unsandboxed run. Running the test with
`setarch x86_64 -R` allowed TSAN to initialize; ASLR was disabled only for that
diagnostic process.

The actual context was `heads=2, headsize=128, tokens=4, outstride=128,
qstride=256, gated=0` (`/tmp/hcthread-shape.log`). The synthetic Qwen GGUF
incorrectly declared a shared Q-normalization tensor of length `q_dim=128`
instead of `head_size=64`. The loader uses that shared norm length for the
per-layer head size, so this malformed fixture produced overlapping head/token
outputs and incorrect gated-Q detection. Serial execution masked the shape
error. This diagnosis does not establish any corresponding defect in the
real-model fixtures, and no loader-validation change is made here.

`test/test_qwen36.c` now emits the correct shared-head norm length and asserts
the loaded attention head size and Q/output projection widths. Its existing
all-logits/last-logit/no-logits state and continuation checks now run with
null, zero-worker, and seven-worker pools. Assertions and tolerances are not
relaxed. With the corrected fixture:

- Production and HC prototype eight-thread controls pass:
  `/tmp/hcfixture-control.log`, `/tmp/hcfixture-proto.log`, exit 0.
- GCC AVX2 ThreadSanitizer passes both corrected eight-thread controls with
  no race reports: `/tmp/hcfixture-tsan-control.log`,
  `/tmp/hcfixture-tsan-proto.log`, exit 0.
- Baseline full suite `/tmp/hcfixture-baseline.log`, exit 0.
- Clean default build and full suite `/tmp/hcfixture-clean-build.log`,
  `/tmp/hcfixture-full-test.log`, exit 0, zero warnings.
- Final Clang AVX2/AVX512 tests with the permanent expanded pool coverage:
  `/tmp/hcfixture-final-avx2.log`, `/tmp/hcfixture-final-avx512.log`, exit 0,
  zero build warnings.

Only the regression fixture and documentation changed in production. Engine
sources and real GGUFs are unchanged, and the restored ISA binaries still
match the F32-cost checkpoint. All processes completed. The HC prototype is
now clear of this test discrepancy but remains unintegrated; promote it with
the corrected regression coverage and re-gate the final artifacts next.
The broader model/backend objective remains open.

### Token-parallel HC integration and final Qwen3.8 refresh

Integrated the HC prototype in `src/transformer/prefill.c`. Private task
contexts schedule independent complete tokens for grouped normalization and
gate/mix. The norm function is selected before dispatch, and each token keeps
the original stream-order arithmetic. Task contexts are stack-local and
activation buffers remain request-local. The existing CPU capability selects
batched HC only on AVX2/AVX512; single-token and non-batched backend fallbacks
remain unchanged. No model-family conditions, public API, or ownership changes.

The corrected synthetic state test now additionally compares prefill and
continuation logits bitwise across null, zero-worker, three-worker, and
seven-worker pools, for both final-attention/final-SSM variants and HC/MoE
versus non-HC/dense fixtures. Existing all-logits/last-logit/no-logits state
checks are retained. Final validation:

- Baseline `/tmp/hcintegrate-baseline.log`, exit 0.
- Clean default build and full suite `/tmp/hcintegrate-clean-build.log`,
  `/tmp/hcintegrate-full-test.log`, exit 0, zero warnings.
- Clang AVX2/AVX512 synthetic suites `/tmp/hcintegrate-test-avx2.log`,
  `/tmp/hcintegrate-test-avx512.log`, exit 0, zero build warnings.
- GCC AVX2 ThreadSanitizer `/tmp/hcintegrate-tsan.log`, exit 0, no race
  reports; per-process ASLR disabled with `setarch x86_64 -R`.
- AVX2 ASan/UBSan `/tmp/hcintegrate-asan-approved.log`, exit 0, after the
  approved unsandboxed rerun for LSAN's ptrace restriction.

Fresh strict token gates on the final ISA binaries, eight prompts, 24-token
cap, t8/maxseq512, matching ISA, F32 KV both engines, flash off, loop abort
disabled:

| Fixture | AVX2 sampled IDs | AVX512 sampled IDs | Preparation |
| --- | --- | --- | --- |
| Qwen3.8 Flash Next Q4_K_XL | 176/176 PASS | 179/179 PASS | disabled |
| Qwen3.8 27B Q4_K_XL dense | 192/192 PASS | 192/192 PASS | default |

All token counts match. Logs `/tmp/hcintegrate-{avx2,avx512}-parity.log` and
`/tmp/hcintegrate-dense-{avx2,avx512}-parity.log`, exit 0.

Final isolated five-run pp128/tg32/t8 speed gates, same preparation/loop
settings as above, matching-ISA llama-bench, flash off. Bitnet KV is F32 and
reference benchmark KV is default F16:

| Fixture/backend | Decode bitnet / reference tok/s | Ratio | Prefill bitnet / reference tok/s | Ratio |
| --- | --- | --- | --- | --- |
| Flash Next AVX2 | 7.09 / 6.96 | 1.019 PASS | 47.96 / 56.90 | 0.843 FAIL |
| Flash Next AVX512 | 6.93 / 7.06 | 0.982 PASS | 50.28 / 62.51 | 0.804 FAIL |
| 27B dense AVX2 | 3.03 / 2.85 | 1.063 PASS | 15.01 / 16.42 | 0.914 PASS |
| 27B dense AVX512 | 3.05 / 2.85 | 1.070 PASS | 21.82 / 19.69 | 1.108 PASS |

Logs `/tmp/hcintegrate-{avx2,avx512}-speed32.log`, exit 1, and
`/tmp/hcintegrate-dense-{avx2,avx512}-speed32.log`, exit 0. Prefill samples:

- Flash Next AVX2: 46.34, 48.31, 49.85, 47.71, 47.96.
- Flash Next AVX512: 50.28, 49.87, 51.89, 49.38, 51.48.
- Dense AVX2: 15.01, 15.16, 14.96, 15.01, 15.09.
- Dense AVX512: 21.80, 21.82, 22.37, 22.22, 21.38.

Decode samples respectively: 7.06, 7.09, 7.39, 7.21, 7.04; 7.00, 6.92,
6.93, 6.93, 6.93; 3.03, 3.03, 3.04, 3.03, 3.03; and five times 3.05.
The earlier AVX2 prototype's 0.863 prefill pass does not override the final
artifact's 0.843 failure. Both sparse prefill gates remain open despite gains
over the previous 44.20/47.76 tok/s checkpoint.

Restored worktree binaries match `/tmp/hcintegrate-avx2` and
`/tmp/hcintegrate-avx512`. SHA256 respectively:
`258b0376818075666065e0c188cd0b5a955e64c03dc252c92a9e9ab8f463ea55`,
`5c1576aca5806d3b716258ba44701a77da0e7cdbcfb806e7e79d0c825cffba6d`.
All processes completed. Only the Gemma4 31B dense GGUF is still present under
`/data/models/gguf/gemma4`; sparse Gemma4 remains unavailable locally. Other
fixtures, CUDA, and default sparse preparation were not refreshed by these
gates. The full requested objective remains open.

### Default sparse preparation cost and cache-only diagnostic

The current HC-integrated AVX2 checkpoint was profiled on Qwen3.8 Flash Next,
pp128 (token ID 1 repeated), t8/maxseq512, one-token cap, loop abort disabled.
Default preparation took 5316.9 ms prompt time versus 2635.1 ms with
preparation disabled. MoE total was 4010.7 versus 1386.1 ms, although prepared
gate/up execution was faster (361.4 versus 661.9 ms). Logs
`/tmp/prepcost-default.err`, `/tmp/prepcost-raw.err`, exit 0. These are separate
diagnostic runs, not a statistical throughput gate.

A temporary `src/moe_prefill.c` copy reuses cached prepared projections but
does not construct local packed copies on misses. Decode preparation and the
rest of the runtime are unchanged. The cache-only prototype passed the AVX2
MoE suite, including exact cold/warm comparisons and pin/release checks
(`/tmp/prepcache-test.log`, exit 0, zero build warnings). With default
preparation enabled elsewhere, profile prompt times were 2949.7 ms AVX2 and
2408.9 ms AVX512 (`/tmp/prepcache-profile-{avx2,avx512}.err`, exit 0).
The AVX512 one-token profile sampled EOG; it is not a decode benchmark.

Fresh strict gates on this prototype, default preparation/cache budget,
eight prompts, 24-token cap, t8/maxseq512, matching ISA, F32 KV both, flash
off and loop abort disabled: AVX2 176/176, AVX512 179/179 sampled IDs, equal
counts, PASS. Logs `/tmp/prepcache-{avx2,avx512}-parity.log`, exit 0.

Isolated five-run pp128/tg32/t8 gates with default preparation/cache budget:

| Backend | Decode bitnet / reference tok/s | Ratio | Prefill bitnet / reference tok/s | Ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 5.03 / 6.87 | 0.732 FAIL | 48.38 / 56.77 | 0.852 PASS |
| AVX512 | 5.46 / 6.84 | 0.798 FAIL | 50.56 / 62.22 | 0.813 FAIL |

Logs `/tmp/prepcache-{avx2,avx512}-speed32.log`, exit 1. Bitnet F32 KV versus
reference benchmark default F16 KV. Prefill samples: AVX2 47.45, 49.05, 49.56,
48.38, 47.09; AVX512 48.76, 48.22, 52.77, 50.56, 52.07. Decode samples:
AVX2 5.11, 5.03, 5.03, 5.05, 5.02; AVX512 5.45, 5.49, 5.49, 5.46, 5.46.
Thus cache-only prefill alone does not solve default-setting throughput.

A separate AVX2 decode-only five-run tg32/t8 experiment, same prototype and
`BN_CPU_PREPARED_CACHE_MB=32768` (no prepare-all override), reached 7.10 versus
7.09 tok/s, ratio 1.001 PASS. Samples 7.10, 7.13, 7.10, 7.10, 7.10
(`/tmp/prepcache-cache32-avx2-speed32.log`, exit 0). This isolates cache
capacity as a performance factor for this fixture. The 32 GiB configuration
was not separately token-gated and does not establish whole-matrix or
default-budget acceptance.

Follow-up strict gates explicitly used `BN_CPU_PREPARED_CACHE_MB=32768`
with the same cache-only prototype, no prepare-all override, default
preparation enabled, loop abort disabled, eight prompts/24-token cap,
t8/maxseq512, matching ISA, F32 KV both, and flash off. AVX2 passed 176/176
sampled IDs; AVX512 passed 179/179, with equal output counts on all eight
prompts. Logs `/tmp/prepcache-cache32-{avx2,avx512}-parity.log`, exit 0.
This closes the larger-cache token-check gap above for this prototype and
fixture, not production-policy or whole-matrix acceptance.

The subsequent isolated AVX512 decode-only five-run tg32/t8 gate with the
same 32 GiB setting passed: 6.96 versus 7.03 tok/s, ratio 0.990. Bitnet samples
6.96, 6.96, 7.03, 6.96, 6.96; log
`/tmp/prepcache-cache32-avx512-speed32.log`, exit 0. Reference benchmark uses
default F16 KV versus bitnet F32 KV. This compares with the default-budget
0.798 ratio above and supports cache capacity as a decode factor on both
ISAs. It does not close the AVX512 prefill gap, establish a suitable universal
cache budget, or promote the temporary cache-only prefill implementation.

To assess whether local packing should ever be retained, a standalone Q4_K
640×2560 projection benchmark compared raw GEMV-order batching against local
arena allocation/preparation plus packed batching, t8, seven measured pairs
after two warm-ups, alternating order. Every output compared bitwise equal.
Median milliseconds (packed total excludes arena destruction):

| Tokens per expert | AVX2 raw / packed total | AVX512 raw / packed total |
| --- | --- | --- |
| 1 | 0.0123 / 0.1596 | 0.0277 / 0.1652 |
| 2 | 0.0224 / 0.1752 | 0.0231 / 0.1780 |
| 4 | 0.0307 / 0.1838 | 0.0307 / 0.1781 |
| 8 | 0.0534 / 0.2079 | 0.0539 / 0.2009 |
| 16 | 0.1041 / 0.2479 | 0.1031 / 0.2458 |
| 32 | 0.2042 / 0.3264 | 0.2004 / 0.3186 |
| 64 | 0.4123 / 0.4771 | 0.3982 / 0.4422 |
| 128 | 0.7995 / 0.7644 | 0.7890 / 0.7137 |
| 256 | 1.6096 / 1.3749 | 1.5663 / 1.2642 |

Source `/tmp/prepbreak-bench.c`, logs `/tmp/prepbreak-{avx2,avx512}.log`,
exit 0; zero build warnings. These cache-resident synthetic measurements
suggest a break-even near 128 tokens for this shape, not a universal format
threshold. Large batches can benefit from local preparation, so unconditional
cache-only behavior is not promoted. A subsequent policy should account for
format and per-expert reuse in quant/runtime ownership, with broader shape
and cold-cache validation. Decode cache sizing remains a separate concern.

Temporary prototype source `/tmp/prepcache-moe-prefill.c`; binary SHA256:
AVX2 `fcbfe74f226ee9e37fcb619c02d7ca913229af87a45b1b82c7fd7814b57bf762`,
AVX512 `76b5ad04265ef2371d3b717f3008db5705269f9c3aa62bc206c71ab3e9dccccd`.
All processes completed. Production engine sources/binaries remain the
HC-integrated checkpoint; only this evidence was added to documentation.
Other fixtures, CUDA, and large-batch behavior are not accepted by these
experiments. The full goal remains open.

### Broader Q4_K local-packing break-even sweep

The temporary `/tmp/prepbreak-bench.c` now accepts row/column dimensions and
an optional cold-weight flag. Seven shapes were measured at t8 with token
counts 1, 2, 4, 8, 16, 32, 64, 128, 256, seven measured pairs after two
warm-ups, alternating raw/prepared order. Cold mode flushes every source
weight cache line with CLFLUSH followed by MFENCE before each timed path;
flush time is excluded. This isolates cold source weights, not a fully cold
process or model. Packed total includes arena allocation, packing and compute,
but excludes destruction, as in the earlier benchmark.

First tested token count where median packed total beat raw batching:

| Q4_K rows × cols | Warm AVX2 / AVX512 | Cold AVX2 / AVX512 |
| --- | --- | --- |
| 640 × 2560 | 128 / 128 | 256 / 256 |
| 768 × 2048 | 256 / 128 | 256 / 256 |
| 1536 × 2048 | 128 / 128 | 256 / 256 |
| 2048 × 1536 | 128 / 128 | 128 / 128 |
| 4096 × 2048 | 128 / 128 | 256 / 256 |
| 2048 × 4096 | 128 / 128 | 256 / 256 |
| 143 × 768 | None / None | None / None |

For cold 640×2560 at 128 tokens, raw/packed total were 0.8231/0.8879 ms
AVX2 and 0.8024/0.8513 ms AVX512; at 256 tokens, 1.6307/1.5258 and
1.5822/1.4038 ms respectively. The non-eight-row layout remained slower
even at 256 tokens: cold raw/packed total 0.2863/0.9977 ms AVX2,
0.2783/1.0092 ms AVX512. This supports a layout-aware reuse decision rather
than unconditional packing or a universal 128-token threshold. Other thread
counts, formats and shapes still need coverage before generalizing policy.

Every output pair compared bitwise equal across all 252 shape/token/ISA/
cache-state cases. Both ISA builds had zero warnings and all runs exited 0.
Warm logs `/tmp/prepshapes-{avx2,avx512}.log`; cold logs
`/tmp/prepcold-{avx2,avx512}.log`; corresponding build logs use
`/tmp/{prepshapes,prepcold}-build-{avx2,avx512}.log`. ISA runs were isolated
from each other and from compilation. Production engine code and binaries
were unchanged. These are kernel diagnostics, not replacement end-to-end
token or throughput gates.

### Transient Q4_K preparation policy integration

Quant policy now owns `bn_quant_batch_preparation_worthwhile`, a transient
GEMV-order batch packing cost decision separate from format eligibility.
Native x86 Q4_K batching skips local preparation below 128 tokens and for
non-eight-row layouts, within the existing raw-batch column bound. Larger
x8 batches retain prior behavior; this is not a claim that packing always
wins at 128 tokens. Diagnostic float/reference or disabled-native-batch
routes, other formats, and non-x86 behavior are unchanged. MoE prefill consults
the policy only after checking for a cached layout. Decode cache behavior,
budgets, model ownership and GPU state are unchanged.

Added quant boundary/invalid-input tests and expanded the MoE cold/warm cache
test to 9, 127 and 128 tokens. Baseline/full final `make -j8 test` passed;
`make clean && make -j8 bitnet` passed with zero warnings. Clang AVX2 and
AVX512 quant suites passed, and AVX2 MoE ASan/UBSan passed after an approved
rerun outside the sandbox (LeakSanitizer ptrace restriction). Logs
`/tmp/preppolicy-baseline.log`, `/tmp/preppolicy-clean-build.log`,
`/tmp/preppolicy-final-test.log`, `/tmp/preppolicy-quant-{avx2,avx512}.log`,
`/tmp/preppolicy-moe-sanitize-approved.log`, all exit 0.

Fresh default-preparation/default-cache strict Flash Next gates, eight
prompts/24-token cap, t8/maxseq512, matching ISA, F32 KV both, flash off,
loop abort disabled: AVX2 176/176 and AVX512 179/179 sampled IDs, equal counts,
PASS. Logs `/tmp/preppolicy-{avx2,avx512}-parity.log`, exit 0.

Isolated AVX512 five-run pp128/tg32/t8 default-setting gate: prefill
51.03/62.50 tok/s (0.817 FAIL), decode 5.51/6.96 (0.792 FAIL). Prefill samples
52.61, 52.48, 51.03, 50.96, 50.35; decode 5.51, 5.56, 5.50, 5.51, 5.52.
Log `/tmp/preppolicy-avx512-speed32.log`, exit 1. Bitnet F32 KV versus reference
benchmark default F16 KV. The policy removes unnecessary local packing but
does not close either default-setting throughput gap on this fixture.

The subsequent isolated AVX2 gate, identical settings, reached prefill
48.72/56.72 tok/s (0.859 PASS) but decode 5.11/7.06 (0.724 FAIL).
Prefill samples 48.85, 48.74, 47.56, 48.72, 47.12; decode samples
5.11, 5.07, 5.07, 5.12, 5.12. Log `/tmp/preppolicy-avx2-speed32.log`, exit 1
because decode remains below threshold. All validation/benchmark processes
completed; no concurrent compilation or inference affected these speed runs.

Production checkpoint SHA256 (also restored as worktree ISA binaries):
AVX2 `3170d9067aa7f2385ac66ef0c90e4c56e0a07d524715480de72dc7efc99ebaee`,
AVX512 `5cb1e6fdd6835743c54e40d3012c1dbbab6b9d9e50e776081ea3d903868b70c4`.
Other model/backend gates and the earlier 32 GiB configuration have not yet
been refreshed on these production binaries. The full goal remains open.

### Current-checkpoint Qwen3.6 sparse refresh

The transient-packing-policy production binaries above were re-gated on
`/data/models/gguf/qwen3_6/35b_a3b/q8_0/abliterated/qwen3.6_35b_a3b_q8_0.gguf`.
Default preparation/cache, loop abort disabled, eight prompts/24-token cap,
t8/maxseq512, matching ISA, F32 KV both, flash off: AVX2 and AVX512 each
passed 192/192 sampled IDs with equal output counts on all eight prompts.
Logs `/tmp/preppolicy-q36-{avx2,avx512}-parity.log`, exit 0.

Subsequent isolated five-run pp128/tg64/t8 speed gates, matching-ISA
llama-bench, default preparation/cache, bitnet F32 KV versus reference default
F16 KV:

| Backend | Decode bitnet/reference tok/s | Ratio | Prefill bitnet/reference tok/s | Ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 14.06 / 14.06 | 1.000 PASS | 118.30 / 123.07 | 0.961 PASS |
| AVX512 | 13.79 / 13.98 | 0.986 PASS | 113.59 / 130.62 | 0.870 PASS |

AVX2 prefill samples: 119.13, 112.33, 118.30, 111.89, 120.95; decode:
14.05, 14.07, 14.06, 14.06, 14.07. AVX512 prefill: 107.77, 113.59, 120.98,
116.05, 111.97; decode: 13.78, 13.79, 13.79, 13.80, 13.80.
Logs `/tmp/preppolicy-q36-{avx2,avx512}-speed.log`, both exit 0.
The AVX512 median now passes the 0.85 threshold; sample variation and changed
reference timings mean this is not evidence attributing the improvement to
one particular intervening implementation change. This accepts this fixture's
eight-thread CPU configuration on the current checkpoint, not twelve-thread,
CUDA, other quantizations, or whole-matrix acceptance. No engine code changed
during this refresh; all processes completed.

### Current-checkpoint Qwen3 sparse warmed-cache refresh

Current production binaries (SHA256 in the transient-packing integration
section) were re-gated on
`/data/models/gguf/qwen3/30b_a3b/q4_k_m/Qwen3-30B-A3B-Q4_K_M.gguf` with
`BN_CPU_PREPARED_CACHE_MB=32768 BN_CPU_PREPARE_ALL_EXPERTS=1` and loop abort
disabled. Eight prompts/24-token cap, t8/maxseq512, matching ISA, F32 KV both,
flash off: AVX2 and AVX512 each passed 192/192 sampled IDs, equal counts on all
eight prompts. Logs `/tmp/preppolicy-q3moe-{avx2,avx512}-parity.log`, exit 0.

Isolated five-run pp128/tg64/t8 speed gates with the same explicit warmed-cache
configuration, matching-ISA llama-bench, bitnet F32 KV versus reference default
F16 KV:

| Backend | Decode bitnet/reference tok/s | Ratio | Prefill bitnet/reference tok/s | Ratio |
| --- | --- | --- | --- | --- |
| AVX2 | 24.11 / 24.62 | 0.979 PASS | 145.60 / 146.34 | 0.995 PASS |
| AVX512 | 25.72 / 25.23 | 1.019 PASS | 171.70 / 175.47 | 0.978 PASS |

AVX2 prefill samples: 146.47, 143.90, 145.60, 148.80, 141.92; decode:
24.12, 24.12, 24.11, 24.11, 24.09. AVX512 prefill samples: 171.70, 167.17,
179.80, 175.61, 160.58; decode: 28.89, 25.78, 25.30, 25.08, 25.72.
Logs `/tmp/preppolicy-q3moe-{avx2,avx512}-speed.log`, both exit 0.
These accept this fixture's explicit warmed-cache eight-thread CPU
configuration on the current checkpoint, not the default cache budget or
startup latency (prepare-all cost is outside timed inference). No engine code
changed; all checks completed. CUDA and the remaining matrix are still open.

### Current CUDA Qwen3.8 dense single-device recheck

`make -j8 BN_ENABLE_CUDA=1 bitnet` completed with zero warnings; saved as
`/tmp/preppolicy-cuda`, SHA256
`fa734caff646530f821884948b6f6b9acb3ea8b2076af24382c8027272818eb8`.
The worktree `bitnet` is now CUDA-enabled; dedicated AVX2/AVX512 binaries
remain unchanged. Build log `/tmp/preppolicy-cuda-build.log`, exit 0.

Driver access required approved execution outside the sandbox. GPU 0 had
96682/97887 MiB occupied, so both engines were isolated to physical GPU 1
with `CUDA_VISIBLE_DEVICES=1 BN_CUDA_DEVICE=0`; no other workload was stopped.
Qwen3.8 dense 27B Q4_K_XL, production llama-completion sampled-ID oracle,
eight prompts/10-token cap, t8, F32 KV both, flash off, loop abort disabled:
the strict gate failed 61/80 prefix IDs with equal output counts.

The first invocation placed `--llama-cuda` after `--maxseq 512`; the harness
resets its llama argument array on backend selection, dropping the prior
context flag. A second complete gate put backend selection first, explicitly
confirming `-ngl 99 -c 512 -ctk f32 -ctv f32 -fa off -t 8`, and reproduced
61/80. Logs `/tmp/preppolicy-cuda-q38dense-parity.log` and
`/tmp/preppolicy-cuda-q38dense-ctx512-parity.log`, both exit 1. Context-flag
loss is therefore not the cause of these mismatches; harness argument-order
independence remains a separate follow-up.

`In the year 2020, the world` diverges at the first token: bitnet ID 557
(`was`) versus llama ID 30008 (`witnessed`). The previously known
`Once upon a time, there was a` divergence remains after one matching token.
All six other prompts match. Earlier 71/80 evidence is not an equivalent
older-binary single-device control, so this does not establish that an
intervening engine change caused the additional failing prompt. It establishes
the current unresolved CUDA baseline on GPU 1. No inference implementation
changed in this recheck, no throughput acceptance is claimed, and all runs
completed. The full goal remains open.

### CUDA prefill and partial-offload controls

On the same current CUDA binary and physical GPU 1, two failing Qwen3.8 dense
prompts were retested at ten tokens each, t8/context512, F32 KV both, flash off,
loop abort disabled, production sampled-ID oracle. No inference code changed.

* `--no-prefill` in the comparison harness disables bitnet batch prefill and
  also sets llama `-b 1 -ub 1`. This still fails 1/20 prefix IDs. Bitnet's
  first year-prompt token remains 557 (`was`), while llama changes from 30008
  (`witnessed`) to 579 (`'s`). Once-upon still diverges after one token.
  Log `/tmp/preppolicy-cuda-q38dense-noprefill-parity.log`, exit 1.
* Adding `--llama-batch 2048 --llama-ubatch 512` after `--no-prefill` restores
  the reference batch sizes while keeping bitnet serial prefill. The original
  two mismatches return unchanged (year 557 versus 30008; Once-upon prefix
  1/10), total 1/20. Log
  `/tmp/preppolicy-cuda-q38dense-bnserial-parity.log`, exit 1.
* Restoring normal bitnet prefill and limiting llama to `--llama-gpu-layers 65`
  makes both prompts pass all 20/20 sampled IDs with equal output counts.
  Log `/tmp/preppolicy-cuda-q38dense-ngl65-parity.log`, exit 0.

Thus disabling bitnet batching does not resolve the current failures, whereas
reference batching/offload placement changes the numerical outcome. Partial
offload is a localization control, not full-CUDA acceptance or justification
to weaken the reference gate. Comparing the graph placement/arithmetic across
the reference offload boundary remains necessary before assigning a kernel
cause. All diagnostic processes completed; the full goal remains open.

### CUDA offload-boundary placement identified

Read-only metadata inspection shows Qwen3.8 dense has `qwen35.block_count=65`
and `nextn_predict_layers=1`: 64 trunk blocks plus one MTP block. The current
reference loader (`llama-model.cpp`, `i_gpu_start = max(n_layer_all + 1 -
n_gpu_layers, 0)`) therefore leaves block 0 on CPU at `-ngl 65`; it does not
move the output head to CPU. A production llama-completion run on physical
GPU 1 with verbosity 5 confirmed block 0 CPU, blocks 1 and 2 CUDA0, output
layer 65 CUDA0, and `offloaded 65/66`. The recurrent-memory log also explicitly
places layer 0 on CPU, with a 3.12 MiB CPU RS buffer. Log
`/tmp/preppolicy-llama-ngl65-placement.log`, output
`/tmp/preppolicy-llama-ngl65-placement.out`, exit 0. Settings: year prompt,
one-token cap, t8/context512, F32 KV, flash off, greedy; no trace injection.

Block 0 is recurrent. Its inspected projections include Q5_K QKV
(5120→10240), Q8_0 alpha/beta (5120→48), Q8_0 SSM output (6144→5120),
IQ4_XS FFN gate/down and Q3_K FFN up. Together with the prior 20/20
partial-offload control, this selects block 0 projections/recurrent update as
the next useful tensor-comparison boundary. It does not prove a particular
operation is wrong or exclude downstream scheduling changes. In particular,
the output-head-only explanation is not supported by actual placement.
No implementation changed; the diagnostic process completed. Full-CUDA
token parity remains unresolved.

### Block-0 reference tensor comparison across CPU/CUDA placement

The default `make test/llama_layer_probe` linked installed `/usr/local/lib`
libraries whose hashes differ from the benchmark reference. Initial captures
`/tmp/cudal0-ll{65,99}.{out,err}` are not authoritative reference captures.
A separate `/tmp/cudal0-matched-probe` was compiled from the current probe
source with explicit headers, link path and rpath for
`/home/mark/artalis.io/tools/llama.cpp/build/bin`; runs also explicitly set
`LD_LIBRARY_PATH` to that directory.

Matched-reference captures on physical GPU 1 compared `--gpu-layers 65` and
99 for the year prompt, context512/F32 KV/flash off. This diagnostic probe
uses one CPU thread and an evaluation observer, so it does not replace the
production eight-thread sampled-token gate. At the last prompt position 10:

| Captured block-0 row | Elements | CPU-placement vs GPU-placement |
| --- | --- | --- |
| `attn_norm-0` | 5120 | Bitwise identical |
| `linear_attn_qkv_mixed-0` (Q5_K projection) | 10240 | max abs diff 0.1595935822; RMS diff 0.03504385363 |

The maximum QKV difference is at index 5365. Full-row dumps:
`/tmp/cudal0-{65,99}-{llama_attn_norm,llama_ssm_qkv}.bin` with corresponding
`.out/.err` logs. All four runs exited 0; elementwise metrics were computed
from little-endian F32 data with Python standard-library struct/math (NumPy
was unavailable). No summaries were substituted for the full-row comparison.

The matched probe's final ranking also reproduces the year-prompt flip:
CPU block 0 ranks 557 (`was`) at 16.392643 above 30008 (`witnessed`) at
16.3887844; GPU block 0 ranks 30008 at 16.453783 above 557 at 16.408514.
Summary logs `/tmp/cudal0-matched-ll{65,99}.{out,err}`, exit 0. This identifies
a difference already present at the QKV projection from identical captured
input, before the recurrent update. It does not yet compare bitnet's CUDA
projection or establish that this projection alone causes the final mismatch.
No inference arithmetic changed; all diagnostic processes completed.

### Bitnet CUDA QKV capture and execution-shape control

A temporary `/tmp/cudal0-gpu-fallback.c` copy adds two binary dump calls to
the existing GPU SSM comparison path, using its already captured normalized
projection input and raw QKV output. It was compiled and linked into
`/tmp/cudal0-bitnet`; no production source or inference arithmetic changed.
Build log `/tmp/cudal0-bn-build.log`, exit 0, zero warnings. Year prompt,
GPU 1, bitnet serial prefill, t8/context512, one-token cap, compare layer 0 /
position 10. Dumps `/tmp/cudal0-bn-{gpu_ssm_projection_input,gpu_ssm_qkv_raw}.bin`
and corresponding `.out/.err` logs; both runs exited 0.

Full-row comparisons against the matched-reference captures above:

| Bitnet CUDA row / reference | Max absolute difference | RMS difference |
| --- | --- | --- |
| Input / CPU or GPU placement | 0 (bitwise identical, 5120 elements) | 0 |
| QKV / CPU block 0, batched | 0.03637313843 | 0.008753269968 |
| QKV / GPU block 0, batched | 0.14070892334 | 0.03399332732 |

Because bitnet used serial prefill for these hooks, a matching llama CUDA
`--sequential` probe control was then captured. Its QKV row agrees with bitnet
to max abs 0.000003814697266, RMS 0.0000002262196630, across 10240 elements.
Dump `/tmp/cudal0-ll99-serial-qkv.bin`, matching `.out/.err` logs, exit 0.
All metrics use full F32 dumps, not printed prefixes.

Thus the large QKV difference is strongly associated with reference batch
versus single-token execution, while the two single-token CUDA projections
agree closely. This does not establish bitnet's normal batched QKV output,
prove an end-to-end cause, or resolve the previously observed serial-token
mismatch after later operations. Next comparisons should separate batched
projection arithmetic from subsequent recurrent/FFN differences. Diagnostic
observer and reference one-thread caveats still apply. All runs completed;
production binaries and the full goal remain unchanged.

### Q5_K batched minimum-correction difference quantified

Reference source inspection found a concrete batch arithmetic distinction.
On Blackwell, `ggml_cuda_should_use_mmvq` selects Q5_K MMVQ only for at most
five columns; the eleven-token year prompt instead selects supported MMQ
(the reference build has FORCE_CUBLAS off). Q5_K MMQ uses DS4 activation
metadata: FP16 scale and FP16 sum of the original 32 float activations.
Its minimum correction multiplies the latter by the weight minimum.
Single-token Q5_K MMVQ instead uses scaled sums of quantized activation
integers, as does bitnet's current `cuda_vec_dot_q5k_q8_1` helper.
Both layouts use 32-value scales for Q5_K; the 64-value D2S6 layout belongs
to Q2_K, not this projection.

A standalone standard-library diagnostic `/tmp/cudal0-min-correction.py`
reads the captured normalized input and this exact Q5_K tensor (relative
offset 1779773440 plus GGUF data offset 10996640), computes each minimum-
correction difference, and predicts its effect on all 10240 output rows.
It includes F32 arithmetic for the activation reduction and FP16 metadata
rounding; it does not emulate the complete MMQ reduction or all quantization
rounding differences. Comparing matched-reference batched minus serial QKV:

| Difference | Max absolute | RMS |
| --- | --- | --- |
| Observed batch minus serial | 0.1407089233 | 0.03399332815 |
| Predicted minimum-correction term | 0.1400204747 | 0.03398686313 |
| Residual after subtracting prediction | 0.007058535288 | 0.000730517815 |

Log `/tmp/cudal0-min-correction.log`, exit 0. The large RMS reduction supports
minimum-correction semantics as the dominant cause of this captured batch/
serial QKV difference. It is not a complete kernel equivalence proof or an
end-to-end parity fix. Remaining rounding/reduction error and other format
paths still matter. No production code changed; the next implementation
should keep batch correction semantics format/backend-owned and retain
single-token arithmetic. The full goal remains open.

### Temporary Q5_K batch-correction prototype

`/tmp/q5orig-gpu.cu` adds rounded original-activation sums to temporary CUDA
Q8 input metadata and uses them in the short-batch Q5_K dot kernels above five
tokens, matching the observed Blackwell MMVQ/MMQ boundary. Single-token dot
arithmetic is retained. This prototype expands private scratch metadata and
does not cover the separate large packed-MMQ path; it is not production-ready
or a universal device policy. It was linked as `/tmp/q5orig-bitnet`; build
log `/tmp/q5orig-build.log`, exit 0, zero warnings. Production sources/binaries
were not changed.

The two failing Qwen3.8 dense prompts still fail the ten-token strict gate
(1/20 matching prefix IDs, equal counts), same settings as the GPU-1 control.
Log `/tmp/q5orig-parity.log`, exit 1. Therefore this change alone does not
resolve either continuation.

Normal prefill entry profiling confirms batch=1 and eleven prompt tokens
(`/tmp/q5orig-entry.err`, exit 0). A host-buffer QKV dump hook was used for
the same block 0 / position 10 row in both the production control and the
prototype. The hook's CPU-oriented name does not establish CPU execution;
projection routing can use GPU kernels before returning the host buffer.
Direct comparison with matched-reference batched CUDA QKV:

| Binary | Max absolute error | RMS error |
| --- | --- | --- |
| Production control | 0.1407089233 | 0.03399333055 |
| Q5_K original-sum prototype | 0.007057189941 | 0.000730518789 |

Dumps `/tmp/q5orig-{control,default}-prefill-qkv.bin`, with corresponding
`/tmp/q5orig-{control,default}-prefill.{out,err}` logs, both exit 0. This is a
direct before/after full-row comparison, not merely the earlier predicted
correction. Residual projection error, other quantized projections and later
operations remain to investigate. No throughput or full-kernel-suite acceptance
is claimed for the temporary prototype. All processes completed; goal open.

### Q5_K batch scale-product rounding closes the year-prompt gap

On the captured normalized input, F32 division versus reciprocal-multiply
activation quantization produced identical integer values and FP16 scales.
Reference tensor-core MMQ additionally rounds each weight super-scale × group
scale/minimum product to FP16 before accumulation (`mmq-load-tiles.cuh`).
The single-token helper instead applies those factors in F32 later.

`/tmp/cudal0-mmq-weight-rounding.py` decodes this Q5_K tensor and predicts the
effect of those rounded products. Predicted correction RMS 0.000730507282;
remaining captured error after correction RMS 0.0000004762535575, max
0.000008467352018. Log `/tmp/cudal0-mmq-weight-rounding.log`, exit 0.

A second temporary CUDA source `/tmp/q5round-gpu.cu` combines original-sum
minimum correction with rounded scale/minimum products in the same short-
batch Q5_K helpers, retaining single-token arithmetic. Build/link
`/tmp/q5round-bitnet` completed without warnings
(`/tmp/q5round-build.log`, exit 0). Direct normal-prefill block-0 position-10
QKV comparison against matched-reference batched CUDA now gives max abs
0.000007629394531, RMS 0.0000004738277560. Dump
`/tmp/q5round-prefill-qkv.bin`, run logs `/tmp/q5round-prefill.{out,err}`,
exit 0. This validates the correction in actual CUDA execution, not only
the analytical predictor.

The subsequent strict two-prompt gate on physical GPU 1, context512/t8,
F32 KV both, flash off, ten-token cap, improves from 1/20 to 11/20 matching
prefix IDs with equal counts. The entire year-prompt continuation now passes
10/10; Once-upon still diverges after one matching token. Log
`/tmp/q5round-parity.log`, exit 1. Thus the two batch arithmetic corrections
close one observed full-CUDA token gap, not the remaining one or the full
matrix. Large packed-MMQ paths, private scratch-layout overhead, cross-format
tests and device-aware dispatch still need work before production integration.
No production source/binary changed; all processes completed; goal open.

### Q5_K prototype full 24-token CUDA control

The rounded-scale/original-sum prototype was expanded to the full eight-prompt
24-token strict Qwen3.8 dense gate, then the unchanged production CUDA binary
was run with identical settings on physical GPU 1: t8/context512, F32 KV both,
flash off, loop abort disabled, production llama-completion sampled-ID oracle.

| Prompt | Production matching prefix / 24 | Prototype matching prefix / 24 |
| --- | --- | --- |
| France | 24 | 24 |
| Year 2020 | 0 | 10 |
| Fox | 24 | 24 |
| Once upon | 1 | 1 |
| Arithmetic | 23 | 23 |
| HTTP | 16 | 16 |
| Sky | 12 | 12 |
| Python | 24 | 24 |
| Total | 124 / 192 FAIL | 134 / 192 FAIL |

Both engines produced equal counts on all prompts. Logs
`/tmp/q5round-control-full24-parity.log` and `/tmp/q5round-full24-parity.log`,
both exit 1. Prototype SHA256
`ea268bfda72f8dc499d80f60ee68e19ba35eaca4d24ab2e0975cd674b258b876`;
production control remains `fa734caff646530f821884948b6f6b9acb3ea8b2076af24382c8027272818eb8`.
Thus the later arithmetic/HTTP/sky failures are present in the control, not
new matching-prefix regressions from this prototype. The year-prompt benefit
is limited to ten matching tokens, not a longer-generation pass. No throughput
or general regression-suite acceptance is claimed. All runs completed;
production remains unchanged and full CUDA parity remains open.

### Combined short-batch Q4_K/Q5_K experiment regresses token coverage

Temporary `/tmp/q45round-gpu.cu` extends the Q5_K prototype's original-sum
minimum correction and FP16 scale-product rounding to the short-batch Q4_K
dot helpers, including shared-input four/eight-token variants. The one-token
helpers retain their prior default arithmetic. Build/link
`/tmp/q45round-bitnet` passed without warnings (`/tmp/q45round-build.log`).
SHA256 `56e5632477f924025e7ec9d6318e40498177604c4f7831976fbfa51cc98500de`.

The same full eight-prompt/24-token strict CUDA gate on GPU 1, t8/context512,
F32 KV both, flash off, loop abort disabled, falls from the Q5_K-only
prototype's 134/192 to 124/192 matching prefix IDs. Year returns to a first-
token mismatch (557 versus 30008), losing its ten-token matching prefix;
all other prompt prefix lengths match the preceding control table. Counts
remain equal. Log `/tmp/q45round-full24-parity.log`, exit 1.

This experiment is not promoted. Although reference Q4_K MMQ uses analogous
correction metadata, end-to-end behavior regressed and a direct same-input
Q4_K tensor comparison is required before concluding whether the extension,
its routing, or interactions with remaining differences are responsible.
No production source/binary changed; all processes completed. Full goal open.

### Same-input Q4_K projection validates the isolated correction

The first non-embedding Q4_K QKV projection is block 1, 5120→10240,
relative GGUF offset 2000661376. Matched-reference GPU-1 probe captures at
year-prompt position 10 saved `attn_norm-1` and `linear_attn_qkv_mixed-1`
(`/tmp/q4same-ll-{llama_attn_norm,llama_ssm_qkv}.bin`, corresponding `.out/.err`
logs, exit 0). Probe context512/F32 KV/flash-off/observer caveats still apply.

A standalone `/tmp/q4same.c` reads that exact weight tensor and captured
normalized input. It repeats the input across eleven independent token rows,
invokes the public CUDA matmul hook, and saves the last output row. The same
test was linked against production CUDA (`/tmp/q4same-control`) and the combined
Q4_K/Q5_K candidate (`/tmp/q4same-candidate`). Both ran on GPU 1 and exited 0,
without CPU fallback; returned GPU matmul success is asserted. Build commands
completed without warnings. Full-row comparison against the captured reference:

| Q4_K implementation | Elements | Max abs error | RMS error |
| --- | --- | --- | --- |
| Production | 10240 | 0.04676580429 | 0.005083599589 |
| Batch-corrected candidate | 10240 | 0.000001430511475 | 0.0000001117444344 |

Dumps `/tmp/q4same-{control,candidate}.bin`, logs
`/tmp/q4same-{control,candidate}.log`. Identical input isolates the projection
from upstream activation differences. The candidate substantially improves
this isolated Q4_K projection despite the previous end-to-end year-prompt
regression. That regression therefore cannot by itself disprove these batch
correction semantics; interactions with remaining differences require further
tensor-level investigation. No production source/binary changed and no
end-to-end or throughput acceptance is claimed. All processes completed.

### Combined candidate block-0 trace: next divergence is alpha/beta

The combined temporary Q4_K/Q5_K candidate completed a normal eleven-token
year-prompt prefill on GPU 1 with position-10 summaries enabled:
`/tmp/q45-block0-trace.{txt,out,err}`. Generation completed; no diagnostic
process remains. Comparing block 0 against `/tmp/cudal0-matched-ll99.out`
shows QKV and convolution remain closely aligned, while the next visible
projection differences are Q8_0 alpha/beta. For example alpha element 0 is
3.15519047 versus 3.14741468, and beta element 4 is 0.443292975 versus
0.434050947. The recurrent gate remains close, but the output projection
differs (element 0: 0.00409722328 versus 0.0051717679), feeding different
FFN-normalized inputs. These summary samples are localization evidence,
not full-tensor error bounds or a proof of the remaining root cause.

Source inspection confirms alpha/beta use `prefill_quant_matmul_multi`, which
can dispatch GPU batches; its host-side name does not establish CPU execution.
CUDA Q8_0 matmul also has distinct float-input and quantized-input routes.
The next isolation step is a same-input Q8_0 projection comparison with the
actual runtime route verified. No production implementation or binary changed;
the previous failing end-to-end gates remain authoritative.

### Same-input Q8_0 alpha projection identifies float-input mismatch

Standalone `/tmp/q8same.c` reads block-0 `ssm_alpha.weight` (Q8_0,
5120→48, relative offset 1948836032) and the previously matched reference
normalized input. Eleven identical rows isolate its batch projection.
`/tmp/q8same` links the production CUDA object and asserts successful direct
GPU matmul, with no wrapper fallback. Matched reference capture is
`/tmp/q8same-ll-alpha.{bin,out,err}`; GPU-1 standalone results are
`/tmp/q8same-{control,native}.{bin,log}`. All completed successfully.

The default CUDA result reproduces the alpha value seen in the complete
prefill trace. Enabling the existing `BN_CUDA_ENABLE_NATIVE_QUANT_MATMUL=1`
route changes only the tested runtime configuration, not production source.
Full 48-element comparisons (`/tmp/q8same-compare.py`, `.log`):

| Comparison | Max abs error | RMS error |
| --- | --- | --- |
| Default CUDA vs reference | 0.02513408661 | 0.008753629264 |
| Default CUDA vs float-input prediction | 0.0000009937385776 | 0.0000003314041874 |
| Reference vs FP16-scale Q8 input prediction | 0.001552314194 | 0.0005771244803 |
| Reference vs FP32-scale Q8 input prediction | 0.0000006134040440 | 0.0000002090537250 |
| Existing native-quant CUDA route vs reference | 0.0000009536743164 | 0.0000002762853875 |

Reference MMQ selects D4 (FP32 activation scales) for Q8_0. The optional
bitnet kernel already has that scale storage, whereas its default Q8_0
matmul consumes float activations directly. This establishes a concrete
arithmetic mismatch for this projection. It does not establish matching
rounding at all quantizer boundaries, serial/decode semantics, other batch
sizes, performance, or end-to-end acceptance.

The combined Q4_K/Q5_K prototype with this Q8_0 option was also tested with
the full eight-prompt, 24-token-cap strict CUDA gate:
`/tmp/q458native-full24-parity.log`, exit 1. Prefix counts remain
24/0/24/1/23/16/12/24 = **124/192**, with equal token counts for every prompt.
Both sides use context512/F32 KV/flash-off/t8 and full reference GPU offload.
This invocation mistakenly supplied `BN_NO_LOOP_ABORT` instead of
`BN_DISABLE_LOOP_ABORT`, so loop detection was not disabled; all 192 tokens
were nevertheless generated on both sides. Do not describe this run as
loop-abort-disabled. No throughput gate was run.

A subsequent normal prefill trace with the option enabled,
`/tmp/q458native-block0-trace.{txt,out,err}`, completed successfully and
confirms the end-to-end path actually uses the option. Alpha element 0 is
now 3.14741421 versus reference 3.14741468; beta element 4 is 0.434050798
versus 0.434050947. SSM output element 0 is now 0.0051725395 versus
0.0051717679. The next visible differences are in FFN gate/up projections
(gate element 0 -0.0950292051 versus -0.0938772932; up element 0
0.0200548675 versus 0.0211969018). Same-input IQ4_XS/Q3_K comparisons are
the next isolation step; these samples do not establish full-row errors.
Production binary SHA remains
`fa734caff646530f821884948b6f6b9acb3ea8b2076af24382c8027272818eb8`.
All diagnostic processes completed; no production implementation changed.

### Same-input FFN gate/up: activation quantization explains both differences

Matched-reference block-0 post-attention normalization was captured as
`/tmp/ffnsame-input.{bin,out,err}`. Standalone
`/tmp/ffnsame-{gate,up}.c` repeats that exact 5120-element input across eleven
rows and asserts successful direct production CUDA matmul, without wrapper
fallback. The IQ4_XS gate and Q3_K up tensors each have 17408 output rows,
relative offsets 1863168000 and 1910517760 respectively. GPU-1 results are
`/tmp/ffnsame-{gate,up}-bn.{bin,log}`; corresponding matched reference captures
are `/tmp/ffnsame-{gate,up}-ll.{bin,out,err}`. All runs completed successfully.

CPU diagnostic `/tmp/ffnsame-predict.c` dequantizes the same weights using
the production quant API, accumulates products in double, and compares two
inputs: original floats and 32-element Q8 quantization with FP32 scales.
The latter models the reference MMQ D4 input layout selected for both formats.
Predictions are `/tmp/ffnsame-{gate,up}-{float,quant}.bin`; comparisons and
script are `/tmp/ffnsame-compare.{log,py}`. Full-row results:

| Projection / comparison | Max abs error | RMS error |
| --- | --- | --- |
| IQ4_XS CUDA vs reference | 0.002915799618 | 0.0005550584454 |
| IQ4_XS CUDA vs float prediction | 0.0000001192092896 | 0.00000001009216299 |
| IQ4_XS reference vs quantized prediction | 0.0000004768371582 | 0.00000001761343202 |
| Q3_K CUDA vs reference | 0.002192772925 | 0.0005369825465 |
| Q3_K CUDA vs float prediction | 0.00000005960464478 | 0.000000009052584132 |
| Q3_K reference vs quantized prediction | 0.0000002384185791 | 0.00000001638652462 |

Thus the FFN differences persist with identical upstream inputs, and input
quantization explains both to near floating-point accumulation error. This
supports prototyping batch input-quantized execution for these formats;
it does not validate serial/decode dispatch thresholds, all rounding boundary
cases, end-to-end parity, or throughput. These are temporary diagnostics only;
no production implementation or binary changed, and all processes completed.

### Temporary FFN D4-input prototype validates on GPU

`/tmp/ffnproto-gpu.cu` extends the combined Q4_K/Q5_K diagnostic candidate
with a quantize/dequantize input kernel ahead of generic batch dot products.
It applies FP32-scale Q8 rounding to Q3_K batches above five tokens and
IQ4_XS batches above eight, following the inspected Blackwell reference
batch-path boundaries. Single-token arithmetic is unchanged. The prototype
uses a temporary CUDA allocation/free per selected projection: this is an
arithmetic experiment, not a production allocation policy or speed candidate.
It covers three generic matmul fallback launch sites, not every fused backend
route. No model-name checks or production source changes were introduced.

The temporary CUDA build and three links completed without warnings
(`/tmp/ffnproto-build.log`). An initially truncated source-tool capture caused
a failed compile; the source was reconstructed from bounded reads and its
diff against the prior candidate checked before the successful build.
Baseline `make test` completed with exit 0 and no compiler warnings in
`/tmp/ffnproto-baseline.log`.

Direct GPU-1 same-input projection tests both exited 0:
`/tmp/ffnproto-{gate,up}.{bin,log}` against the preceding reference dumps.
IQ4_XS gate max error is 4.768371582e-7, RMS 1.984374581e-8;
Q3_K up max error is 2.384185791e-7, RMS 1.827162077e-8.
Temporary full executable `/tmp/ffnproto-bitnet` SHA is
`5df6cc17604750f69add2f00383e3e61923fdd62cbce8d407f05e483860cd505`.
The production binary remains unchanged.

The complete strict gate with this executable and
`BN_CUDA_ENABLE_NATIVE_QUANT_MATMUL=1 BN_DISABLE_LOOP_ABORT=1` completed:
`/tmp/ffnproto-full24-parity.log`, exit 1. Reference full GPU offload,
context512/F32 KV/flash-off/t8, eight prompts and 24-token cap remain fixed.
Prefix counts are still 24/0/24/1/23/16/12/24 = **124/192**, with no
token-count mismatches. Isolated arithmetic agreement has not translated
into accepted end-to-end parity; no speed claim is made.

Normal position-10 year-prompt prefill trace
`/tmp/ffnproto-block0-trace.{txt,out,err}` subsequently completed and verifies
the FFN changes are reached. Gate element 0 is -0.0938538462 versus reference
-0.0938772932, up element 0 is 0.0212415587 versus 0.0211969018, and block-0
output element 0 is -0.0294829123 versus -0.0294566825. The previous combined
Q4_K/Q5_K-only block-0 output element was -0.0306624472. These summaries show
a substantially closer first-block boundary, but still not equality. Next
work should measure full-row boundary errors and track the remaining
divergence downstream rather than infer correctness from sampled elements.
All processes completed; production remains unchanged.

### Downstream localization: block-14 IQ3_S requires a different investigation

Full reference summaries `/tmp/ffnproto-ll-all.{out,err}` completed on GPU 1.
Comparison script/log `/tmp/ffnproto-boundaries.{py,log}` compares selected
position-10 block outputs against the prototype trace. First-16-element RMS
error grows from 2.08e-5 at block 0 to 0.0145 at block 13, then jumps to
0.0961 at block 14. These are sampled, not full-row, boundary errors.

Block 14 uses IQ3_S FFN down weights, shape 17408→5120, relative GGUF offset
4942689920. Matched reference input/output dumps are
`/tmp/iq3same-{ffn_swiglu,ffn_out}.{bin,out,err}`. Direct production CUDA
same-input eleven-row test `/tmp/iq3same.c` and `/tmp/iq3same-bn.{bin,log}`
completed with asserted GPU matmul success and no fallback. Full output
error against reference is max 0.475981541, RMS 0.0916988003.

Unlike preceding formats, activation quantization does not explain this:
`/tmp/iq3same-predict.c` float-input prediction matches CUDA with max
5.960464478e-8 and RMS 1.678633440e-8, whereas its FP32-scale Q8 input
prediction differs from reference by max 0.4776667804, RMS 0.0916883079.
The 512-entry codebook and block layout match inspected reference source.
An additional check calls `dequantize_row_iq3_s` from the exact reference
`build/bin/libggml-base.so`, not merely copied source. For row 0, the
library-dequantized float dot is -0.04951069276, production CUDA is
-0.04951068386, quantized-input prediction is -0.04873141274, and the
reference GPU graph output is -0.09681457281. Script/log:
`/tmp/iq3same-libcheck.{py,log}`.

This does not yet prove a reference CUDA bug or a bitnet decoding bug.
Next work must independently verify the reference GPU matmul and captured
graph input/output before choosing any arithmetic change. All processes
completed; no production source or binary changed.

### Independent reference IQ3_S graph confirms an oracle inconsistency

`/tmp/iq3same-ll-direct.cpp` constructs a standalone ggml CUDA multiply from
the exact GGUF IQ3_S tensor and captured input, repeats eleven input rows,
and links the same explicit reference `build/bin` libraries. GPU-1 run
`/tmp/iq3same-ll-direct.{bin,log}` completed successfully. Its output matches
the full-model captured `ffn_out-14` **bit-for-bit**, ruling out a capture-only
explanation. One- and eight-token runs (`/tmp/iq3same-ll-{serial,eight}.{bin,log}`)
agree within max 5.96e-8/RMS 9.59e-9, but both differ substantially from the
eleven-token result and expected decoded dot. All runs exited 0.

A one-block basis-vector test `/tmp/iq3same-basis.cpp` uses the first 256
weights, one output row, and 256 one-hot input rows. The resulting effective
GPU weights (`/tmp/iq3same-basis.{bin,log}`, exit 0) differ from the exact
reference CPU library dequantization by max 0.04023832083. For example,
positions 0–3 produce zeros rather than -0.0048103/-0.0016034/0.011224/-0.0048103;
positions 4–7 match the decoded weights. Many subsequent low-four positions
show similar discrepancies, but not all are zero. Simply masking the first
four weights of every group of eight does not explain the full projection:
`/tmp/iq3masked-predict.c` still differs by max 0.469137162/RMS 0.06635755687.
Do not adopt that mask as a correction.

These independent tests establish an inconsistency in the current reference
GPU oracle, not yet its cause. Reference build/source provenance and a clean
isolated CUDA rebuild should be checked before attempting to imitate the
IQ3_S result. No bitnet production arithmetic was changed. All diagnostic
processes completed; full goal acceptance remains outstanding.

### Fresh reference CUDA build reproduces IQ3_S inconsistency

Reference HEAD is `3d3d7c81813067fc8c185da017e0af03b4269b1e`; its only local
source edits are two unrelated `kqv_wo` debug callbacks in `llama-graph.cpp`.
The existing CUDA library was built September 2 with CUDA 13.2.78 for
`sm_120a`. A fresh build directory `/tmp/llama-iq3-clean.5FlBek` was configured
against the same sources with explicit CUDA 13.2 compiler, architecture120a,
Release, CUDA enabled, and NCCL disabled (single-device test). Existing
reference build artifacts and sources were not modified. Compiler-cache
writes required approved execution after a sandbox failure.

`cmake --build ... --target ggml-cuda -j8` completed with exit 0 and no compiler
warnings in `/tmp/iq3-clean-build-approved.log`; configure log is
`/tmp/iq3-clean-configure.log`. New CUDA library SHA:
`f407bf0f88a5ab2cbb207c8733fa5fc85713c131906de9f01fccc56924d5384c`.
`LD_LIBRARY_PATH` and `ldd` verified both CUDA/base libraries resolve inside
the new directory. Repeating the standalone basis and eleven-token multiply
with these libraries yielded outputs bit-for-bit identical to the original:
logs `/tmp/iq3-clean-{basis,direct}.log`, output `/tmp/iq3-clean-direct.bin`.
Original basis output is preserved at `/tmp/iq3same-basis-original.bin`.
This rules out stale original build artifacts as the explanation under the
tested compiler/configuration, not a compiler or source-level problem.

Compute Sanitizer `--tool initcheck --error-exitcode 99` on the fresh-library
basis test exited 0 and reported zero errors
(`/tmp/iq3-clean-initcheck.log`). That check alone does not rule out all shared
memory, race, arithmetic, or compiler issues. Further kernel-level isolation
is required before changing bitnet to match the IQ3_S oracle. All processes
completed; production bitnet and original reference artifacts remain unchanged.

### Reference IQ3_S memcheck detects invalid reads at the actual model shape

Small CUDA diagnostic `/tmp/iq3decode.cu` uses the reference codebook and
block declarations to test scalar decoding, packed sign intrinsics, and
packed two-integer index loading. All three agree bit-for-bit with the
reference CPU dequantizer for the captured block. It compiled and ran
successfully on GPU 1; this narrows the issue beyond those isolated operations.

Reference basis-test `racecheck` exited 0 with zero hazards
(`/tmp/iq3-clean-racecheck.log`), but **memcheck exited 99 with 514 errors**
(`/tmp/iq3-clean-memcheck.log`). It reports invalid four-byte global reads in
`mul_mat_q<(ggml_type)21,128,true>`. Thus passing initialization/race checks
must not be interpreted as memory safety. The planned device-optimization
experiment was paused before building any variant; its empty temporary
directory is `/tmp/iq3-noopt.UeS6lA`.

The same check on the actual 17408→5120, eleven-token isolated projection
also failed, exit 99, with **33980 errors**:
`/tmp/iq3-clean-direct-memcheck.log`. The first invalid global read is in
`mul_mat_q<(ggml_type)21,16,false>` at kernel offset `0x6010`; the basis
variant reports offset `0x15f0`. This establishes the memory fault is not
limited to the one-row diagnostic shape. Sanitized runs did not complete
valid numerical output; do not use their output files as parity evidence.

Next work should map the faulting instruction to the IQ3_S batch kernel
and resolve the reference fault in an isolated build before treating its
output as an arithmetic oracle. Bitnet must not emulate invalid reads.
No production code changed; all processes completed.

### Explicit IQ3_S packed-byte extraction fixes the isolated reference batch fault

Recompiling only the reference IQ3_S MMQ instantiation with line information
mapped the invalid read to `mmq-load-tiles.cuh:1400`, the first codebook lookup
through a byte pointer into local `int2 qs_packed`. Log
`/tmp/iq3line-memcheck.log` again exited 99. This source expression should
index at most 511, so the observed out-of-bounds access is inconsistent with
its intended semantics. This does not by itself distinguish a compiler bug
from all source-language/optimization concerns.

An isolated variant replaces those two byte-pointer reads with explicit
unsigned shifts/masks of `qs_packed.x` and `.y`, preserving the decoded byte
indices. Temporary sources `/tmp/iq3patch-{mmq,load,vec}.cuh`, translation unit
`/tmp/iq3patch.cu`, and object `/tmp/iq3patch.o` leave original reference sources
untouched. Only the IQ3_S batch object differs in the side library
`/tmp/iq3-explicit.Mop46K/libggml-cuda.so.0`; the earlier lineinfo-only library
is `/tmp/iq3-noopt.UeS6lA/libggml-cuda.so.0` (directory name notwithstanding,
device optimization was not disabled).

The actual-shape eleven-token reference projection now passes memcheck with
zero errors and exit 0 (`/tmp/iq3patch-memcheck.log`). Its full 5120-element
output `/tmp/iq3patch-direct.bin` matches the independent FP32-scale quantized
prediction with max error 9.126961231e-8 and RMS 2.070421937e-8. The one-block
basis test also passes memcheck, exit 0 (`/tmp/iq3patch-basis-memcheck.log`).
This validates the batch fix on those cases; the reference single-token
IQ3_S path remains separately uncorrected and must be investigated before
resuming full parity gates against this candidate. Bitnet production and
original reference build/source remain unchanged. All processes completed.

### Reference IQ3_S decode fault corrected in isolated library

The original reference single-token path also fails memcheck: exit 99,
14117 errors in `mul_mat_vec_q<21,1,false,false,false>`
(`/tmp/iq3-decode-control-memcheck.log`). The corresponding byte-pointer
lookups in `vec_dot_iq3_s_q8_1` were replaced with explicit unsigned extraction
from the packed integer pair, as for the validated batch variant. Only
temporary source copies and a rebuilt `mmvq.cu` object changed. Supporting
temporary header include paths avoid including duplicate original headers.

Combined side library `/tmp/iq3-both.SXqFoQ/libggml-cuda.so.0` SHA is
`ada851a572cadf3577ace0e7b51fcdce59c031f995c159a0dd21c722f0d8b9e6`.
The corrected single-token actual-shape test passes memcheck with zero errors
and exit 0 (`/tmp/iq3-both-serial-memcheck.log`). Output
`/tmp/iq3-both-serial.bin` matches independent FP16-scale Q8 input prediction
(`/tmp/iq3serial-predict.c`, `/tmp/iq3serial-gate-quant.bin`) with max error
2.980232239e-8 and RMS 8.672829866e-9. Eleven-token output
`/tmp/iq3-both-batch.bin` remains bit-for-bit equal to the validated batch-only
variant. Original reference and production bitnet sources/binaries remain
unchanged. This validates those IQ3_S cases, not the entire backend.

The complete eight-prompt, 24-token-cap CUDA gate using this corrected
reference and `/tmp/ffnproto-bitnet` completed with exit 1:
`/tmp/iq3fixed-ref-full24-parity.log`. Configuration remains full GPU offload,
context512/F32 KV/flash-off/t8, native Q8_0 batch input enabled, loop abort
disabled. `LLAMA_LIB_DIR` explicitly selects the combined side CUDA library,
fresh base library, then original reference build for remaining libraries;
`ldd` verified these resolutions. This is a distinct corrected-reference
experiment, not a passing gate against the original faulty binary. No
throughput result or broad backend acceptance is claimed. Prefix counts are
24/24/24/21/24/12/16/24 = **169/192**, all eight first tokens match, and token
counts match for every prompt. Remaining divergences are story, HTTP, and
sky; all diagnostic/gate processes completed.

### Bitnet IQ3_S batch-input prototype matches repaired reference projection

Temporary `/tmp/iq3bn-gpu.cu` extends the earlier FFN prototype by enabling
the same FP32-scale quantize/dequantize input step for IQ3_S batches above
eight tokens. Only that format predicate changed; decode arithmetic and
production sources are untouched. This retains per-call scratch allocation
overhead and is not a production performance design. CUDA build/log
`/tmp/iq3bn-build.log` and two executable links completed without warnings.

Direct same-input actual-shape GPU-1 test `/tmp/iq3bn-same` passes memcheck
with zero errors and exit 0 (`/tmp/iq3bn-same-memcheck.log`). Full output
`/tmp/iq3bn-same.bin` versus corrected reference `/tmp/iq3-both-batch.bin`
has max error 1.043081284e-7 and RMS 2.628695890e-8. Temporary full executable
`/tmp/iq3bn-bitnet` SHA:
`a8df1ceb9ee8bc19043404401e477d7849e4d1e6d4c7c3a9e5fbeaae5b229f2a`.
This is projection-level evidence, not an end-to-end acceptance claim.

The complete corrected-reference strict gate subsequently completed with
exit 1 (`/tmp/iq3bn-full24-parity.log`): **145/192** token-prefix matches,
7/8 first tokens, no token-count mismatches. Configuration is unchanged
from the preceding 169/192 corrected-reference run. The year prompt now
diverges at its first output (candidate 30008, reference 557); the remaining
prefix counts are unchanged. Thus this individually validated batch change
regresses the full gate by 24 tokens and is not promoted. Remaining numerical
differences need tensor-level isolation against the repaired reference;
projection accuracy alone does not establish full-model parity. No speed
gate was run. All processes completed and production remains unchanged.

### Repaired-reference trace isolates Q6_K input group-size mismatch

Fresh repaired-reference and current IQ3_S bitnet prototype traces completed:
`/tmp/iq3repair-ll-all.{out,err}` and `/tmp/iq3bn-all.{txt,out,err}`. Boundary
comparison `/tmp/iq3repair-boundaries.{py,log}` confirms the old block-14 jump
is gone: first-16 RMS is 0.01453 at block 13 and 0.01459 at block 14, rather
than the former 0.0961. Earlier differences still accumulate; these boundary
statistics remain sampled, not full-row errors.

Block-1 SSM output is Q6_K, shape 6144→5120, relative offset 2175694592.
Matched repaired-reference input/output captures
`/tmp/q6same-{ssm_gate,linear_attn_out}.{bin,out,err}` and standalone CUDA
`/tmp/q6same.c`, `/tmp/q6same-bn.{bin,log}` all completed successfully.
Identical-input full-row comparison gives max error 0.006151437759 and RMS
0.0003110413819. Independent dequantized-weight predictions show:

| Comparison | Max error | RMS error |
| --- | --- | --- |
| CUDA vs float-input prediction | 0.008475065231 | 0.0003232063429 |
| Reference vs 32-value FP32-scale input quantization | 2.384185791e-7 | 3.891978428e-9 |
| CUDA vs 256-value FP32-scale input quantization | 7.450580597e-9 | 1.308080875e-9 |

Prediction sources `/tmp/q6same-predict.c` and
`/tmp/q6group256-predict.c` explain the difference: this CUDA path uses Q8_K
input quantization in groups of 256, whereas reference batch MMQ uses D4
groups of 32. Next prototype should test matching Q6_K batch semantics;
single-token/reference thresholds and other dispatch paths need independent
validation. No production source or binary changed, and all processes ended.

### Q6_K 32-value batch-input prototype validates projection arithmetic

Temporary `/tmp/q6bn-gpu.cu` extends the IQ3_S candidate with 32-value
FP32-scale input quantization for Q6_K batches above seven tokens. It routes
the three host/device/batch matmul entry points through the diagnostic
quantize/dequantize helper ahead of existing dot kernels. Decode is unchanged;
this is not complete fused-path coverage or a production allocation design.
Build `/tmp/q6bn-build.log` and executable links completed without warnings.
Full executable `/tmp/q6bn-bitnet` SHA:
`b29a341f940908153e06957cb0f75538f61987d0dc505f586074a4c25848116c`.

Actual-shape same-input test `/tmp/q6bn-same` passes GPU-1 memcheck with zero
errors and exit 0 (`/tmp/q6bn-same-memcheck.log`). Output
`/tmp/q6bn-same.bin` versus reference `/tmp/q6same-linear_attn_out.bin` has
max error 2.384185791e-7 and RMS 4.267998008e-9, down from the preceding
0.006151437759/0.0003110413819. Production sources and binary remain unchanged.

The full corrected-reference gate completed with exit 1:
`/tmp/q6bn-full24-parity.log`, **172/192** prefix tokens, all eight first
tokens matching, no token-count mismatches. Per-prompt prefixes are
24/24/24/24/24/12/16/24. This improves the immediate IQ3_S-only candidate's
145/192 and the earlier corrected-reference 169/192, but HTTP and sky still
fail strict parity. Full GPU offload, context512/F32 KV/flash-off/t8, native
Q8_0 input enabled and loop abort disabled are unchanged. No performance
gate was run; diagnostic allocation overhead remains. All processes ended.

### IQ4_NL batch-input prototype validates the remaining block-1 down format

Block-1 FFN down uses IQ4_NL, shape 17408→5120, relative offset 2030152576.
Repaired-reference captures `/tmp/iq4nl-{ffn_swiglu,ffn_out}.{bin,out,err}`
and direct production CUDA `/tmp/iq4nl-same.c`, `/tmp/iq4nl-bn.{bin,log}`
completed successfully on GPU 1. Identical-input output differs by max
0.0007695453241, RMS 0.0001912118757. Independent predictions
(`/tmp/iq4nl-predict.c`) confirm CUDA matches float-input multiplication
(max 1.490116119e-8/RMS 3.370201316e-9), while reference matches 32-value
FP32-scale input quantization (max 1.862645149e-8/RMS 4.085517483e-9).

Temporary `/tmp/iq4nl-gpu.cu` adds IQ4_NL above eight tokens to the existing
diagnostic batch-input predicate; all prior prototype corrections remain.
CUDA build and links completed without warnings (`/tmp/iq4nl-build.log`).
The same-input candidate passes memcheck with zero errors and exit 0
(`/tmp/iq4nl-memcheck.log`); `/tmp/iq4nl-candidate.bin` matches reference with
max 2.980232239e-8 and RMS 5.130333780e-9. Full executable
`/tmp/iq4nl-bitnet` SHA:
`ac91f1980069951196e36e43e38b0bf2750c5150c7abf5c38da31c809db88ada`.
This remains an allocation-heavy arithmetic prototype; production unchanged.

The complete corrected-reference strict gate then failed, exit 1
(`/tmp/iq4nl-full24-parity.log`): **148/192** prefix tokens, 7/8 first tokens,
no token-count mismatches, prefixes 24/0/24/24/24/12/16/24. The year prompt
again flips to candidate 30008 versus reference 557; other prefixes match
the preceding Q6_K candidate. All gate settings are unchanged. This regresses
the previous 172/192 result and is not promoted. No speed gate was run.
Next work should replay identical token histories and compare logits/tensors
at divergence, rather than treat individual projection accuracy as sufficient
for model-level parity. All processes completed.

## Historical Context

Earlier measurements on an Apple M1 Max showed the CPU path reaching high memory
bandwidth utilization on BitNet ternary models and competitive throughput on
several dense and sparse quantized models. Keep old numbers in commit history;
current docs should prefer reproducible commands and recent gates over broad
claims.
