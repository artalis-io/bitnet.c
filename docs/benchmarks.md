# Benchmarks

Benchmark numbers in this repository are local checkpoints. They are useful for
detecting regressions and comparing implementation choices on the same machine;
they should not be treated as universal performance claims.

## Prefill diagnosis

Build `make bench_prefill` for CPU, or
`make BN_ENABLE_CUDA=1 bench_prefill` for optional CUDA. The default compiler
prefers `/usr/local/cuda/bin/nvcc`, then `nvcc` on PATH; `NVCC=...` still
overrides it. This is a warm-cache benchmark with FP16 KV. It resets session
state outside each measurement and performs one
untimed warmup. Its default repeated `hello` prompt is synthetic and can give
unrepresentative expert locality. Use a UTF-8 corpus containing at least the
requested number of tokens for representative measurements:

```bash
./bench_prefill model.gguf 128 5 16 --prompt-file /path/to/corpus.txt --with-logits
./bench_prefill model.gguf 128 5 16 --prompt-file /path/to/corpus.txt --with-logits --tokenwise
./bench_prefill model.gguf 128 1 16 --prompt-file /path/to/corpus.txt --with-logits --profile
BN_CUDA_DEVICE=0 ./bench_prefill model.gguf 128 5 8 --cuda --prompt-file /path/to/corpus.txt --with-logits
```

File mode tokenizes the complete text, then selects its first N tokens,
including BOS when the tokenizer requests it. It rejects insufficient text
rather than repeating it. Use identical corpus bytes for comparisons and
record their hash. Without `--with-logits`, both modes omit vocabulary
projection; with it, both compute logits only for the final token. These
contracts must match the reference benchmark. Use `--flash` explicitly when
comparing flash attention. Profiling perturbs execution; collect throughput
separately with profiling disabled.

CUDA mode fails if backend creation, weight upload or activation initialization
fails; it never silently reports CPU measurements as CUDA. It uses the
generation prefill wrapper so KV/SSM handoff follows the serving contract,
then reads a device activation to include completion of queued GPU work in
the timed interval. CPU mode retains the direct transformer path. The
`mode=batch` label identifies the requested mode; entry/stage profiles reveal
policy fallbacks. CUDA does not imply every operator is device-resident.
The benchmark opens split GGUF checkpoints through the shard-aware loader.
It does not yet configure a bounded GPU expert cache or implement an expert
placement policy; large MoE placement needs separate qualification.

`--profile` captures `BN_PREFILL_PROFILE=1` in the benchmark's CPU runtime,
including with one thread. The same environment variable enables host prefill
profiles in other callers that create a threadpool runtime. Summary records
identify the configured backend, selected `path`, requested `output`, token
count, dimensions, expert counts, and number of layers executed in fused
chains. `backend=cuda` does not mean every operator ran on CUDA. `path=decode`
or `decode_logits` identifies a sequence-policy fallback; each physical batch
in a microbatched request has its own record. Single-token entry uses forward
directly and does not emit a batched summary.

All timing fields are host wall milliseconds. `chain` includes complete fused
layer/model chains and is not counted as `qkv`. `attn_chain` measures fused
QKV, attention and output projection calls. `ssm` includes SSM projections,
recurrence and output for unfused SSM blocks. `moe` includes routing, expert
execution, and its output composition outside fused chains. `gateup`, `act`
and `down` are subdivisions of `ffn`, not additional totals. `total` includes
requested final logits; `other` is the remainder after non-overlapping stage
totals, including setup, auxiliary operations and cleanup. Small negative
remainders within printed rounding precision are possible. Backend calls
include their existing synchronization; these fields do not measure GPU
kernel duration, transfer bytes, overlap or individual synchronization events.
For MoE, the benchmark also prints existing per-session I/O/compute/cache
statistics after each timed run; those are not CUDA transfer counters.

### CUDA native-quant qualification

Q8 routed and dense projections must preserve their activation-scale and
reduction contracts across standalone, combined, and batched entry points.
The default reference path uses full-warp routed reductions and the existing
FP32-scale MMQ implementation above eight tokens. Small combined routed
batches retain activation quantization rather than switching to float-input
arithmetic below the MMA threshold. Combined routing follows the explicit
router-fusion policy rather than inferring it from expert weight format.

These experimental paths remain available only through explicit runtime flags:

- `BN_CUDA_ENABLE_NATIVE_QUANT_MMA=1`: alternative dense/routed Q8 MMA with
  FP16 activation scales.
- `BN_CUDA_ENABLE_MOE_NATIVE_QUANT_GATEUP_2ROW=1`: half-warp gate/up reduction.
- `BN_CUDA_ENABLE_MOE_NATIVE_QUANT_DOWN_2ROW=1`: half-warp down reduction.

They are not numerically qualified substitutes for the reference defaults.
Existing gate/up and down disable flags take precedence over their enable
flags. The stored independent CUDA output hashes remain unchanged; numerical
failures must not be resolved by regenerating expected values from this
implementation. Faster kernels need to preserve the same contracts before
they become defaults.

The independent F32 SSM projection/state failure is corrected. Input
normalization and alpha projection matched the reference bit-for-bit; the
first difference was the final decay exponential. F32 prefill used `expf`
while the reference library is built with `-use_fast_math` and uses the
intrinsic exponential. Decode and prefill now use the same explicit
softplus/decay helper. All 12 original state hashes remain unchanged and pass;
four independent cases at 64/129 tokens also pass, including the 5120-wide,
48-value-head model geometry. This does not by itself qualify the full model
or complete backend. The older checkpoint below records the prior failure.

The reference projection policy also applies at 64 tokens and above. The
previous size override silently stacked the F32 projection matrices despite
that policy, changing reduction geometry. Separate device projections now
preserve the reference contract for those larger batches.

The Q6_K eight-warp decode tail now retains unscaled block sums and folds
them with explicit fused multiply-add. Scaling each tail block before adding
it discarded the contraction used by the reference kernel. The parallel
execution remains enabled. The focused Q6 small-batch/decode reference gate
passes with its original hashes.

After these corrections, Qwen3.8 27B Q4_K_XL passes the strict eight-prompt
CUDA comparison: **192/192 sampled continuation token IDs**, all eight first
IDs and equal output counts. Settings are GPU 0, 24 generated tokens per
prompt, eight CPU threads, context 512, F32 KV and flash off on both engines,
with production llama.cpp sampler tracing and loop abort disabled. Log:
`/tmp/bitnet-goal-qwen38-full-parity.log`. This supersedes the 184/192 result
in the older checkpoint below. It does not yet establish long/chunked/resumed
prefill correctness, complete CUDA-suite success, or performance parity.

#### SSM reference generator

`test/reference_ssm_projection.c` builds a standalone GGML CUDA graph and
does not link this runtime. It computes weighted RMSNorm, independent F32
alpha/beta projections, softplus, sigmoid, and a gated-delta state update.
It reproduces all 16 fixture hashes with reference CUDA library SHA256
`e3330904452e38a9dedc307dedf83e182322a27d6b10dd1df457e02326082840`.
The graph uses zero Q/K/V and nonzero initial state to isolate projected
decay. It does not qualify nonzero beta sensitivity or quantized projections.

```bash
reference_root=/path/to/llama.cpp
cc -O2 -std=c11 -I"$reference_root/ggml/include" \
  test/reference_ssm_projection.c -L"$reference_root/build/bin" \
  -Wl,-rpath,"$reference_root/build/bin" -lggml -lggml-base -lggml-cuda \
  -o /tmp/reference_ssm_projection
/tmp/reference_ssm_projection 5120 48 129 /tmp/ssm-reference
```

The optional prefix writes independent normalization, alpha, softplus, gate
and decay tensors. State hashes are printed as fixture initializers. Preserve
the reference implementation and library fingerprint when comparing results.

#### Routed reduction reference

`test/reference_mxfp4_routed.c` builds independent GGML CUDA gate/up and
Q5_K down projections for the first routed fixture. It applies explicit
ordered fused and separately rounded reductions on the host to isolate the
contraction contract. Build it using the SSM generator command above,
substituting this source and adding `-lm`. With the same reference library,
it prints `fused=c7f6670bdcc72838 separate=df9207cd3d107e5d`.
The first value is the unchanged routed golden; the second reproduces the
prior runtime failure. This checks that fixture's contraction, not general
end-to-end model correctness. All direct MXFP4/Q5_K routed fixture hashes
pass after correcting the ordered dispatcher's reduction mode. Other
per-expert paths retain their existing reduction contract.

The routed MXFP4 graph now honors the reference activation flag through the
existing AVX2-reference SiLU arithmetic. Its intermediate FP32 activations
differ, but subsequent Q8 quantization can erase that difference. The test
therefore checks the intermediate change and cross-mode consistency instead
of requiring a different final hash. Existing independent output goldens
remain unchanged. Q4_K and Q5_K block-reference requests retain their
format-specific dispatcher. Routed MXFP4, Q5_K/Q6_K and Q4_K/Q5_1 focused
graph tests pass, including flag changes and graph replay.

CUDA graph reuse now includes operation arithmetic flags and routed geometry
in its layout key. Previously switching from ordered to block-reference
execution reused a graph with 100 nodes for a 45-node sequence. Decode
positions and route contents remain dynamic and do not force graph rebuilds.
The next full-suite failure was the dense Q4_K gate/up projection fixture at
width 2048 and 17 tokens. The stacked-input shortcut changed MMQ projection
geometry relative to the independent separate-projection graph. It now
shares quantized input while issuing separate batched projection views, so
both stacked and separate weight uploads retain reference geometry. The
focused dense gate/up suite passes with unchanged goldens. Complete
CUDA-suite qualification is still pending.

#### Reset and prefill continuation qualification

CUDA now implements the existing `reset_activations` backend hook. Session
reset previously cleared host KV/SSM state but left device state intact.
The hook waits for preceding work on the backend/default streams, clears
device KV and recurrent state, and preserves uploaded weights and allocated
working buffers. A synthetic regression covers clearing, idempotence and
preserving working storage.

Build and run the real-model execution-invariance check:

```bash
make BN_ENABLE_CUDA=1 test_prefill_continuation
BN_CUDA_DEVICE=0 ./test_prefill_continuation model.gguf docs/inference.md 256 64 --cuda
```

Arguments select initial prompt length and chunk size. The corpus must contain
at least prompt-length plus 64 tokens; it is never repeated. The test compares
whole and chunked prefill, eight decode predictions, a resumed 64-token corpus
suffix, and eight further predictions. It uses F32 KV, flash off and eight CPU
threads. After a mismatch it replays the same reference token history to
isolate numerical/state differences. It checks finite logits and exact greedy
IDs, reporting maximum/RMS logits differences. This is not an independent
llama.cpp reference comparison or concurrent-session qualification.

On Qwen3.8 27B Q4_K_XL/GPU 0, the whole-versus-whole reset control at 256
tokens is bit-identical for all 16 checked logits vectors
(`/tmp/bitnet-goal-reset-control.log`). The 256-token/64-token-chunk case
passes 16/16 greedy IDs, including resumed decode
(`/tmp/bitnet-goal-reset-chunk64.log`). Before the reset fix even the
whole-versus-whole control failed 15/16 with large logits differences.
The 1024-token/129-token-chunk case (including a partial final chunk) also
passes 16/16 IDs across initial and resumed decode
(`/tmp/bitnet-goal-reset-long-chunk129.log`). A 33-token/9-token-chunk case
also passes 16/16 IDs, crossing the small-batch projection boundary with a
partial final chunk (`/tmp/bitnet-goal-reset-short-chunk9.log`). These corpus
checks supplement,
but do not replace, independent model-reference comparisons. The standard
CPU suite passes after these changes; its optional model matrix still skips
without fixture paths configured. Historical CUDA repeated-run timing below
predates the reset fix and is not a qualification of correctly reset sessions.

A subsequent short-chunk regression isolated a second lifecycle bug: shared
scratch could grow after decode graph capture, leaving captured quantization
pointers stale. Shared scratch storage changes now contribute to the graph
cache key after reservation. Stable bindings keep replay enabled; a resized
buffer requires recapture. This is distinct from clearing persistent session
state. With replay enabled, the current CUDA 33/9, 256/64 and 1024/129 cases
each pass 16/16 IDs (logs `/tmp/bitnet-goal-storage-continuation-33-9.log`,
`/tmp/bitnet-goal-storage-continuation-256-64.log`, and
`/tmp/bitnet-goal-storage-continuation-1024-129.log`). The short case failed
with stale bindings and passed diagnostically with replay disabled before
this fix. A contended routed test also exposed activation transfers using the
private stream when execution explicitly selected the default stream.
Activation writes/readbacks now use the configured execution stream, avoiding
that cross-stream input race without adding a global synchronization. Reset,
routed MXFP4/Q5_K, Q5_K/Q6_K, Q4_K/Q5_1 graph/reference cases and dense gate/up
references pass after this correction (`/tmp/bitnet-goal-stream-focused.log`).
Compute Sanitizer completed the graph-enabled 33/9 model check with 16/16
IDs and **zero memory errors** (`/tmp/bitnet-goal-storage-memcheck.log`).
The CPU 33/9 corpus check remains failing at 15/16 IDs, with small
logits differences (`/tmp/bitnet-goal-final-cpu-continuation.log`); internal
CPU chunked execution is not yet qualified by this check.

The clean build has zero warnings and the standard CPU suite passes
(`/tmp/bitnet-goal-final-clean-build.log`,
`/tmp/bitnet-goal-final-cpu-suite.log`). Independent Qwen CUDA comparisons
retain 192/192 IDs on the eight short prompts and 24/24 IDs on a 4096-byte
corpus prefix (`/tmp/bitnet-goal-final-qwen38-parity.log`,
`/tmp/bitnet-goal-final-long-reference.log`). The latter uses context 2048;
prompt SHA256 is
`5ba8da8fc2ff5e856a2e5d689785a80fc4691a5352d11290bdea864aa35f37f5`.

The F32 decode-attention failure involved two issues. The test applied its
flash oracle to a separate KQ/softmax/V pipeline, and the split softmax used a
block-wide halving reduction instead of GGML's warp-first XOR reduction. The
latter changes results at 33 keys. The runtime now uses warp-first reduction;
all original flash hashes remain unchanged and pass. The split path matches
ten separately generated independent GGML split hashes, including both
33-key cases (`/tmp/bitnet-goal-split-qualified.log`). Full-suite qualification
is being rerun; this focused result does not establish complete backend
success.

`test/reference_decode_attention_split.c` is the standalone split oracle.
Build it using the SSM reference-generator command above, substituting this
source and adding `-lm`. Run it with valid-key count and seed, e.g.
`/tmp/reference_decode_attention_split 33 23`. It uses explicit F32 matmul
accumulation, a 256-position padded cache, and a masked softmax. It does not
link this runtime. Provenance: llama.cpp
`df18bee11e96557b07aa7b12db0976f3c7ac9165`, reference CUDA library SHA256
`e3330904452e38a9dedc307dedf83e182322a27d6b10dd1df457e02326082840`.
The ten emitted initializers are recorded in
`/tmp/bitnet-goal-split-independent-values.log`. The existing flash oracle
retains its original source/version and expected values; its distinct
normalization contract is not regenerated from the split implementation.

The clean softmax build has zero warnings and passes the standard CPU suite
(`/tmp/bitnet-goal-split-clean-build.log`,
`/tmp/bitnet-goal-split-cpu-suite.log`). The independent short-prompt gate
retains 192/192 IDs (`/tmp/bitnet-goal-split-qwen38-parity.log`). CUDA chunked
and resumed tests retain 16/16 IDs in each of the 33/9, 256/64 and 1024/129
cases (`/tmp/bitnet-goal-split-continuation-*.log`). The longer independent
prompt now fails at **21/24** IDs, choosing “layer” where the reference
chooses “operation” (`/tmp/bitnet-goal-split-long-reference.log`); the earlier
24/24 checkpoint is not a qualification of this newer math path. The CPU
33/9 check remains 15/16, with a 0.0140915-logit reference margin between the
two competing tokens (`/tmp/bitnet-goal-cpu-continuation-margin.log`). An
exact 33-token UTF-8 prefix was decoded and retokenized to verify identical
input IDs before comparing CPU engines. That independent CPU comparison
fails at 5/8 continuation IDs (`/tmp/bitnet-goal-cpu-prefix33-reference.log`),
choosing “High-Level Flow” instead of “High-level flow”. These failures
remain open; the strict criteria have not been relaxed.

The next complete-suite failure is FP16 windowed head256/GQA2 attention at
one key, one KV head and window 3
(`/tmp/bitnet-goal-split-cuda-suite.log`). Its plain dispatch excludes head
counts supported by the fused-RoPE reference vector path. The dispatch
correction passes the complete focused head256 reference/window replay
checks, including plain, in-place, fused-RoPE and captured replay modes
(`/tmp/bitnet-goal-head-dispatch-focused.log`). Its original golden values
remain unchanged.

The resulting clean build passes the **complete CPU suite, complete CUDA
backend suite, and CUDA diagnostic isolation** with zero build warnings:
`/tmp/bitnet-goal-head-clean-build.log`,
`/tmp/bitnet-goal-head-cpu-suite.log`, and
`/tmp/bitnet-goal-head-cuda-suite.log`. This closes the synthetic backend
suite failures; it does not close the open CPU corpus/long-prompt model
comparisons or the broader offloading, serving and vLLM performance goals.
The exact-prefix CPU tokenwise comparison passes 8/8 with both engines using
one-token batches (`/tmp/bitnet-goal-cpu-prefix33-tokenwise-reference.log`),
while the batched comparison above fails 5/8. That narrows the remaining CPU
investigation to prefill execution/contract differences. No production
batching has been disabled to obtain this result.

### Qwen3.8 27B CUDA checkpoint

Current local build: GPU 0, RTX PRO 6000 Blackwell Max-Q Workstation Edition,
driver 595.104.02, toolkit 13.4.92, GCC 13.3, SM120a. Same GGUF and
`docs/inference.md` corpus as the CPU checkpoint below; 128 prompt tokens,
eight host threads, FP16 KV, flash off, final-token logits included, profiling
disabled, warm cache, session reset excluded:

| Requested execution | Timed runs | Median ms | Tokens/s |
| --- | ---: | ---: | ---: |
| CUDA batch | 5 | 218.188 | 586.649 |
| CUDA tokenwise prompt control | 3 | 2266.818 | 56.467 |

The tokenwise control omits intermediate vocabulary projections and is not a
decode-throughput measurement. These numbers demonstrate existing batching
benefit, not a new kernel speedup or vLLM parity. Reproduce with the corpus
commands above and `BN_CUDA_DEVICE=0`. Binary SHA256:
`bitnet=f1b09ad135f3bccdcbffc20ae9b957909766753d4dc0d79c05ec15244baf35c9`,
`bench_prefill=c27c71be071228f8b4f0a08da22d4e531865fcf3abf2078866ed9f916c31b95c`.

The warning-free clean CUDA build and `make BN_ENABLE_CUDA=1 test` pass.
Default/calibrated routed Q8, dense Q8 MMQ, independent combined routing,
and diagnostic-isolation checks pass with unchanged reference hashes.
Compute Sanitizer memcheck on the routed-Q8 reference cases reports zero
errors. The complete CUDA kernel suite still fails the F32 SSM case above.

The real-model strict production-reference gate, eight prompts and a
24-token cap, F32 KV on both runtimes, context 512, flash off, eight threads,
matches all eight first token IDs and **184/192** continuation prefix tokens
with no token-count mismatches. The sky prompt diverges after 16/24 tokens;
the other seven prompts match 24/24. This is a **failed strict gate**.
A separate prompt made from the first 700 characters of the corpus passes
an eight-token strict check. Neither result establishes complete model or
long-context qualification. No vLLM performance comparison was run.

### Qwen3.8 27B prefill diagnosis checkpoint

On the Ryzen Threadripper 9980X with GCC 13.3, the native x86 build using
`Qwen3.8-27B-UD-Q4_K_XL.gguf` and the first 128 tokens of `docs/inference.md`
measured the following warm-cache, three-run medians. These runs include final
logits, FP16 KV, flash off, and no profiling:

| Threads | Prefill ms | Tokens/s |
| --- | ---: | ---: |
| 16 | 4969.113 | 25.759 |
| 32 | 3687.412 | 34.713 |

The corpus SHA256 is
`d9f0e92bb69cceb8853f99aa1200133fbbbb40bb8f16ca1988477789dd1b77bf`.
The benchmark binary SHA256 is
`08bb0e026299d681b3c6fcde280ecad2408679100a7d4ac403ed969ff3920486`
(base revision `e2fb6b6` plus the prefill observability changes).
This is a thread-count comparison, not a kernel speedup or vLLM parity claim.
A matching 32-thread tokenwise control measured 20069.139 ms (6.378 tokens/s)
in one timed run after warmup, versus 34.713 tokens/s for batched execution.
The single-run control demonstrates substantial batching benefit; it is not a
multi-run performance qualification.

A separate profiled 16-thread run reported `path=batch`, zero decode fallback,
and a 4913.563 ms total: dense FFN 3035.319 ms (61.8%), SSM 1480.841 ms
(30.1%), QKV 205.579 ms, attention output 112.847 ms, and attention itself
11.718 ms. Within FFN, gate/up took 1652.649 ms and down 1365.699 ms.
The unclassified remainder was only 4.456 ms. This identifies FFN and SSM
execution as the next CPU optimization targets at this prompt length; it
does not support attributing the gap to an absent batched prefill path.
Longer prompts require separate attention and recurrence measurements.
At 32 threads, a separate profile reported 3517.470 ms total, FFN 2053.690 ms,
and SSM 1183.988 ms; the same two stages remain dominant.

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

### Qualified Flash offload profile (2026-11-06)

Qwen3.8 Flash Next Q4_K_XL, 1047 real corpus tokens, F32 KV, no flash,
eight CPU threads, physical RTX PRO 6000 Blackwell GPU 1. Expert-cache budgets
are 2 GiB RAM and a conservative 4 GiB VRAM; base weights, PLE, KV and
activations are outside those budgets. The filesystem page cache was warm.
The four qualified flags are fused HC mean/post, dense prefix chaining and
GDN decode order. Capture covers initial prefill only; all 16 independent
initial/resumed decode IDs still match after tracing.

Nsight attributes 62.0 of 70.1 profiled host seconds to MoE. GPU kernel time
sums to about 7.8 seconds, led by Q8 batched projections (4.47 seconds),
routed K-quant (1.41 seconds) and Q5_1 (1.39 seconds). Initial prefill copies
about 89.5 GB host-to-device and 26.4 GB device-to-host and makes roughly
52,000 CUDA allocations. Wide HC intermediates contribute substantially to
these copies. CUDA API wall times include GPU waits and must not be added
to kernel times as independent costs.

The grouped expert loop loaded CPU projections and prepared CPU layouts
before trying the GPU, then the GPU path loaded those projections again.
CPU loading/preparation now happens only after GPU rejection. A regression
requires a GPU-cache hit to work with no host weights, and verifies exact
CPU fallback after a backend partially writes output and rejects execution.

A serialized paired process comparison, with identical harnesses and one
cold then one reset/warm prefix per version, gives:

| Phase | Before | After | Expert reads before/after |
| --- | ---: | ---: | ---: |
| Initial 1047 | 70.215 s | 67.290 s | 87.962 / 43.981 GB |
| Reset/warm 1047 | 23.180 s | 21.890 s | 87.962 / 43.981 GB |
| Resumed 64 | 6.277 s | 5.977 s | 47.951 / 23.975 GB |

This is one paired run, with all 16 reference IDs preserved, rather than a
statistical latency or vLLM parity qualification. The read reduction is exactly
50%; GPU uploads are unchanged. Both initial and reset/warm prefixes have
zero GPU cache hits and 14,055 misses, so the large cold/warm difference is
not GPU expert reuse. The GPU fetch helper also bypasses the configured RAM
LRU; its counters remain zero. RAM caching, allocation reuse and device-resident
HC/PLE/MoE intermediates remain follow-up work.

Evidence: `/tmp/bitnet-goal-qualified-flash-profile/initial1047.nsys-rep`,
its SQLite/CSV reports and `baseline-summary.json`, plus
`/tmp/bitnet-goal-moe-gpu-first-final/performance.json` and `qualification.json`.

### Linear Q8 schedule construction (2026-11-06)

The same profile exposed a larger cold-start cost inside the gathered Q8
expert path. Its stream-K schedule planner scanned every grid block for
every logical tile. For the 1047-token, 512-expert down projection this is
92,160 tiles with one block per tile: about 8.49 billion interval checks.
The original CUDA timeline has a 46.867-second GPU-idle gap ending at the
1,105,924-byte schedule upload. Expert-call instrumentation reports zero
rejections and 46.2 seconds inside gathered expert callbacks cold versus
3.8 seconds warm, despite identical expert-cache misses.

When grid size equals tile count, the partition assigns exactly the full
`[0, K_blocks)` interval to each tile. The planner now emits that interval
directly, preserving its contents and order while reducing construction from
quadratic to linear time. Split schedules keep their existing implementation.
No independent reference values were changed.

With both this change and deferred CPU expert preparation, the identical
serialized harness measures:

| Phase | Original baseline | Combined changes | Expert reads |
| --- | ---: | ---: | ---: |
| Initial 1047 | 70.215 s | 21.463 s | 87.962 / 43.981 GB |
| Reset/warm 1047 | 23.180 s | 21.534 s | 87.962 / 43.981 GB |
| Resumed 64 | 6.277 s | 5.532 s | 47.951 / 23.975 GB |

That is 3.27x faster initial prefill in this single paired measurement, with
all 16 independent initial/resumed IDs preserved. A second CUDA timeline
has exactly the same kernel-launch counts and transfer bytes as the original;
total kernel time remains 7.843 seconds. Its largest GPU-idle gap is 44.1 ms.
The improvement removes CPU planning work; it does not reduce PCIe traffic or
establish vLLM parity. The budgets, warm filesystem cache and single-pair
limits from the preceding checkpoint apply.

Evidence: `/tmp/bitnet-goal-mmq-plan-linear-final/performance.json`,
`trace-comparison.json`, `initial1047.nsys-rep` and `qualification.json`.

### Bounded RAM cache for GPU expert uploads (2026-11-06)

GPU expert uploads now consult the configured pread RAM cache. Decode admits
whole experts; batched prefill borrows existing hits without admission or LRU
promotion. Upload leases prevent eviction, and entries become visible only
after all three projections have loaded successfully. Disabled, undersized or
fully pinned caches fall back to session staging. GPU RAM-hit counters are
separate from CPU-cache and VRAM-cache counters. Cache metadata still requires
serialized execution; this does not qualify concurrent serving. Tokenwise
prefill using the decode API currently follows decode admission policy.

The same Flash workload and qualified flags as above were measured serially
with RAM caching disabled and an 8 GiB RAM budget. Each process ran two identical
requests, each including two 1047-token prefixes, eight decode steps, a 64-token
resumed prefix and eight more decode steps. Both configurations preserve all
32 independent IDs. The filesystem page cache was warm; no physical-disk cold
claim is made. The conservative 4 GiB VRAM expert cache had zero decode hits.

| RAM budget | Request | Mean decode | Median decode | Expert file-read bytes over 16 steps |
| --- | --- | ---: | ---: | ---: |
| Disabled | First | 290.949 ms | 294.891 ms | 24,068,096,000 |
| Disabled | Repeated | 274.251 ms | 273.961 ms | 24,068,096,000 |
| 8 GiB | First | 367.150 ms | 348.023 ms | 12,963,635,200 |
| 8 GiB | Repeated | 309.783 ms | 310.360 ms | 12,295,475,200 |

The repeated request saves 48.9% of expert reads, but mean decode is 13.0%
slower. A 2 GiB trial also saved reads while slowing decode. Use
`--cache-mb 0` for the cache-disabled control when tuning this workload. RAM caching is a
placement capability, not a demonstrated latency improvement on this hot-page-
cache workload. First-touch cost and working-set churn remain optimization
questions. A temporary pinned-upload wrapper was slower and was not added to
the runtime. These are single serialized process comparisons, not statistical
performance qualification or vLLM parity. RAM slab budgets exclude metadata,
base weights, PLE, KV, activations and the operating system's page cache.

The final clean candidate builds without warnings and passes the standard and
full CUDA suites, diagnostic isolation and backend-matrix checks. Qwen27 keeps
192/192 short and 24/24 long IDs; Flash keeps 179/179 short IDs and 16/16 on
each of eight matched prefill/resume schedules. All 170 initial/resumed state
captures match exactly: 72,327,168 float words. The separate cross-schedule
Flash long-prompt CLI mismatch remains unqualified. These checks do not establish
ARM hardware or concurrent-serving correctness for the new cache path.
The frozen executable is SHA-256
`79227cb618e02c086469e90c432f2e05c3e568b7a0405e7bc100a94606e969a5`.
Qualification: `/tmp/bitnet-goal-ram-direct-qualified/qualification.json`.

Evidence: `/tmp/bitnet-goal-ram-direct-final/repeat-exits.json`,
`repeat-summary.json`, `repeat-0.log`, `repeat-8192.log` and
`ram-2048-pinned.log`.

### GPU expert capacity experiment (2026-11-06)

A benchmark-only variant increases the conservative GPU expert cache from 128
to 1,024 slots, retaining the 32 MiB charge per expert (4 versus 32 GiB budget).
RAM caching is disabled. Hardware, four qualified flags, F32 KV, corpus and
repeated-request schedule match the preceding experiment. No runtime defaults
or arithmetic changed. Both configurations preserve all 32 independent IDs.

| GPU slots | Request | Mean decode | Median decode | GPU hits / calls | Expert-read bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 | First | 290.949 ms | 294.891 ms | 0 / 7,680 | 24,068,096,000 |
| 128 | Repeated | 274.251 ms | 273.961 ms | 0 / 7,680 | 24,068,096,000 |
| 1,024 | First | 192.899 ms | 192.582 ms | 2,645 / 7,680 | 15,784,755,200 |
| 1,024 | Repeated | 206.025 ms | 198.599 ms | 2,645 / 7,680 | 15,784,755,200 |

Repeated-request mean decode improves 24.9%, with 34.4% fewer expert reads.
This is a single serialized process per budget using the previous serialized
128-slot control, not a statistical latency or vLLM parity result. It isolates
GPU expert reuse; it does not establish an optimal cache size or remove prefill
transfer costs. Sampled total GPU usage during the larger-cache run was about
18 GiB, including other model/session allocations; this is not an allocator
peak or a measurement of expert buffers alone.

The CLI's `model_moe_entry_bytes()` currently estimates from the first MoE layer
and charges pread projection bytes differently from mmap projection bytes.
The benchmark uses an independent conservative slot charge. Audit packed/lazy
backend allocations and heterogeneous layer formats before treating the CLI's
budget accounting as equivalent or changing defaults.

Evidence: `/tmp/bitnet-goal-vram-reuse/summary.json`, `32g.log` and `repeat.c`.

### Expert-cache charge audit (2026-11-06)

The CLI previously sized expert slots from the first routed layer. CUDA's
backend did not supply the borrowed-layout cache-charge callback used there,
so packed MMQ and gate/up auxiliary layouts were omitted, while a separate
policy estimate covered some down-projection auxiliary storage.

`bn_gpu_moe_bridge_cache_entry_bytes()` now scans every routed layer, charges
raw bytes plus the backend's declared packed/lazy layout bound for each owned
projection, and takes the largest layer as the uniform slot charge. Fused
gate/up uploads use their combined shape. Mmap-native borrowed layouts retain
the backend's specific charge. Arithmetic overflow disables cache creation
rather than producing a wrapped budget. The CLI uses this shared helper.

This charge covers resident expert payloads and declared backend auxiliary
layouts. It excludes cache metadata, base weights, PLE, KV, activations,
allocator overhead, backend scratch and the incoming expert upload before LRU
eviction. It is not a total-process VRAM limit. Borrowed-layout discounts rely
on the backend's supported native layout being created successfully; an owned
fallback after allocation failure needs per-handle accounting for a strict
budget. CUDA currently uses owned expert uploads.

Synthetic coverage includes every projection's extra layout, a larger second
routed layer, ignored dense layers, pread versus mmap-native charges, absent
backend callbacks and overflow. The cache capacity experiment above remains a
separate benchmark with its explicit 32 MiB charge.

The final clean CUDA executable builds without warnings and passes standard,
full CUDA, diagnostic-isolation and backend-matrix checks. Qwen27 retains
192/192 short and 24/24 long IDs; Flash retains 179/179 short IDs and 16/16 on
each of eight matched schedules. All 72,327,168 words across 170 initial/resumed
state captures agree exactly with the preserved oracle. The known separate
cross-schedule Flash long CLI mismatch remains unqualified.

A real Flash replay using the new helper selects a 17,715,200-byte slot charge:
242 entries fit within 4 GiB. Two matched 9/9 requests, including resumed prefill,
preserve 32/32 independent IDs. This verifies the selected capacity and token
parity; it does not measure total live CUDA allocations or a strict memory peak.
The new helper also passes the WASM syntax check; no new ARM/Metal hardware
qualification is claimed.

Evidence: `/tmp/bitnet-goal-cache-accounting-final/qualification.json`,
`charged-replay.log`, `charged-replay-exit.json` and `wasm-syntax.log`.
Frozen executable SHA-256:
`04f781216f8f71dbc0d014c07bf7c110782ea1cebae86aca1a55ef26c1949a4b`.

### Separate batched-prefill GPU cache policy (2026-11-06)

Batched prefill now borrows GPU expert-cache hits without LRU promotion and
uploads misses into temporary buffers instead of admitting them. This mirrors
the RAM lookup-only policy and preserves decode's working set across prompt
scans. Decode retains normal admission and promotion; fully preloaded resident
experts remain reusable. Temporary tracking capacity is checked before uploads,
and partially created buffers are cleaned up on failure.

The regression test fills a two-entry GPU cache, borrows the oldest entry during
prefill with host weights unavailable, and checks that a later decode admission
still evicts that entry. A prefill miss neither evicts nor admits; its three
buffers survive until explicit release. Failures on each upload and missing or
exhausted temporary ownership leave the original cache and handle count intact.

The cache still requires serialized execution. Tokenwise prefill through the
decode API retains decode admission policy. Temporary expert allocations are
outside the resident expert-cache budget; this policy does not enforce a total
VRAM peak or qualify concurrent serving.

The protected clean candidate passes the standard/full CUDA suites, diagnostic
isolation, backend matrix and WASM syntax checks. Qwen27 retains 192/192 short
and 24/24 long IDs; Flash retains 179/179 short IDs and 16/16 on all eight matched
prefill/resume schedules. All 170 initial/resumed state captures remain exact
(72,327,168 words). The replay using the new 4 GiB charge passes 32/32 IDs.
These checks retain the separate cross-schedule Flash long CLI limitation.

Qualification: `/tmp/bitnet-goal-prefill-cache-final/qualification.json`.
Frozen executable SHA-256:
`47d3fd80fe5b92b7c8c900b1d68cdbf203c215d58f1b8780de3b0128f52c7b94`.

A serialized paired process comparison uses the same 1,024-slot conservative
GPU cache (32 GiB budget, 32 MiB charge), RAM cache disabled, physical GPU 1,
eight CPU threads, F32 KV and the four qualified Flash flags. Each process runs
two identical requests with two 1047-token prefixes, eight decode steps, a
64-token resumed prefix and eight more decode steps. The filesystem cache is
warm. Both processes preserve all 32 independent reference IDs.

| Phase | Admit/promote prefill | Lookup-only prefill | Expert-read bytes before / after |
| --- | ---: | ---: | ---: |
| First request, initial 1047 | 20.884 s | 21.176 s | 43,981,209,600 / 43,981,209,600 |
| Repeated request, reset/warm 1047 | 21.668 s | 20.662 s | 43,981,209,600 / 40,968,908,800 |
| Repeated request, resumed 64 | 5.264 s | 4.716 s | 23,855,206,400 / 20,989,542,400 |
| Repeated request, mean decode (16 steps) | 196.657 ms | 196.049 ms | 15,784,755,200 / 15,406,387,200 |

Repeated reset/warm prefill improves 4.6% and resumed prefill 10.4% in this
single pair; initial cold prefill is 1.4% slower and mean decode is nearly
unchanged. Repeated reset/warm prefill now borrows 964 cached experts versus
zero, and resumed prefill borrows 956 versus 37. This is evidence of working-set
preservation, not statistical latency qualification, a default-capacity tuning
result or vLLM parity. The 4 GiB charge replay above is a separate configuration;
these timings must not be attributed to its 242 slots. Whole-process peak VRAM,
concurrent serving and logical-phase policy for tokenwise prefill remain open.

Evidence: `/tmp/bitnet-goal-prefill-cache-final/performance.json`,
`performance-exits.json`, `before.log`, `after.log` and `repeat.c`.

### Tokenwise prefill upload policy (2026-11-06)

The session now carries a scoped `moe_cache_prefill` flag at the public generation
and transformer prefill entry points, including no-logits and all-logits calls.
The GPU expert bridge honors it when tokenwise execution reaches the ordinary
expert accessor. Prompt ingestion therefore borrows RAM/VRAM hits without LRU
promotion and uses temporary GPU uploads for misses even with `--no-prefill`,
single-token chunks or decode-based transformer fallback. The flag is saved and
restored on every return, including errors, and cleared by session reset.
Direct forward calls outside these entry points retain decode admission.

This changes cache admission rather than kernel selection, batching contracts or
checkpoint routing. The all-logits prefill API also applies this upload policy
when used for verification. CPU-only expert-cache admission is unchanged; shared
cache metadata still requires serialized execution. This is not a concurrent
serving or whole-process memory-peak qualification.

Focused tests cover restored outer scopes after empty/success and invalid-input
returns, session reset, and normal expert-accessor behavior under the session
flag. The GPU cache test still requires no admission, eviction or LRU promotion
while prefill owns temporary uploads, including upload-failure cleanup.

The protected clean candidate builds without warnings and passes standard/full
CUDA, diagnostic isolation, backend matrix and WASM syntax checks. Qwen27 retains
192/192 short and 24/24 long IDs; Flash retains 179/179 short IDs and 16/16 on
each of eight matched schedules. All 170 initial/resumed state captures remain
exact (72,327,168 words). The new-charge replay preserves 32/32 IDs.

A separately generated pinned llama.cpp oracle uses nine initial corpus tokens,
eight generated IDs, 64 resumed corpus tokens and eight more generated IDs,
with prefill chunk size one. Five candidate modes each preserve 16/16 IDs:
chunk-size-one generation ingestion, forced tokenwise ingestion, forced
no-logits prefixes, direct transformer last/no-logits calls and direct
single-token all-logits calls. Every mode also compares GPU cache entries,
hash slots and LRU/free-list endpoints before and after each ingestion phase;
all ten comparisons remain unchanged. A control linked with the previously
qualified bridge fails this invariant at the initial forced-tokenwise prefix,
while using the current session layout and wrappers. Reference IDs and oracle
metadata are preserved by hash.

This closes the tokenwise GPU-upload policy gap described in the preceding
checkpoint. It does not establish a performance improvement, CPU-only cache
policy, strict total-VRAM bound, concurrent serving or vLLM parity. The separate
cross-schedule Flash long CLI mismatch remains unqualified.

Evidence: `/tmp/bitnet-goal-tokenwise-phase-final/qualification.json`,
`phase-exits.json`, `phase-probe.c`, `phase-control-origin.json`,
`old-control.log`, and `/tmp/bitnet-goal-tokenwise-phase-oracle/result.json`.
Frozen executable SHA-256:
`597078429e76fc97b2d2967001cfc931f00a6679f4769aebe7687fed204baf8f`.

### Batch HC token launches without changing arithmetic (2026-11-06)

A new initial-prefill trace of the qualified Flash path retained 16/16 reference
IDs and exposed 198,933 fused HC mix/combine launches. Their batch callbacks
already upload complete arrays but launch once for every token. The kernels
now select token rows through `blockIdx.y`; callbacks tile at 65,535 rows and
retain each token's original block geometry, stream order and fused/separate
arithmetic. Graph/decode launches still use a single token row. Host/device
buffer lifetimes and transfers are unchanged.

A new independent CPU-reference test uses exact binary-fraction inputs and zero
gates for fused/unfused modes, irregular 257-by-259 shapes, and 65,537 token rows
crossing the grid-Y tile boundary. All 2,105,392 words agree exactly with guards
intact. Existing 72 fused-mean and 72 fused-post reference cases also pass. The
protected clean build has no warnings; standard/full CUDA, diagnostic isolation
and backend matrix pass. Qwen27 keeps 192/192 short and 24/24 long IDs; Flash
keeps 179/179 short IDs and 16/16 on all eight matched schedules. All 170 state
captures remain exact (72,327,168 words), and the new-charge replay keeps 32/32.
The separate cross-schedule Flash long CLI mismatch remains unqualified.

| Initial 1047-token trace | Before | Batched launches |
| --- | ---: | ---: |
| HC mix launches | 99,467 | 97 |
| HC combine launches | 99,466 | 96 |
| Combined HC kernel time | 236.526 ms | 8.977 ms |
| All kernel launches | 304,351 | 105,611 |
| All kernel time | 7.873 s | 7.604 s |
| Host-to-device bytes | 89,522,335,108 | 89,522,335,108 |
| Device-to-host bytes | 26,408,132,096 | 26,408,132,096 |

Every non-HC kernel-launch count and every transfer count/byte total is
unchanged. About 15.35 GB of DtoH copies have the 1047-by-10240 wide-HC shape;
shape alone does not identify which callback owns each transfer. Synchronous
`cudaMemcpy` API wall time remains about 10.8 seconds, and there are still about
52,000 allocations and frees. API times include waits and must not be added to
kernel times. Keeping wide intermediates on-device remains the next target.

Three untraced process pairs in alternating order measure initial prefill as:

| Version | Samples (seconds) | Median |
| --- | --- | ---: |
| Before | 21.333, 21.257, 21.045 | 21.257 s |
| Batched launches | 20.888, 21.165, 21.139 | 21.139 s |

The median difference is only 0.6%, with one pair slower: these samples do not
establish a substantial wall-time speedup despite the launch/GPU-time reduction.
Each process preserves all 16 independent initial/resumed IDs. Timing interposes
only the fixture's profiler start/stop window, with no Nsight during measurement.
A separate single paired process samples seven forward-plus-CPU-argmax/logging
intervals in each decode phase: mean 340.0 versus 334.7 ms, median 324.5 versus
321.9 ms. This excludes each phase's final forward and is not pure-kernel or
statistical decode-latency qualification.

All measurements use physical GPU 1, eight CPU threads, F32 KV, the four qualified
Flash flags, a warm filesystem cache, 2 GiB RAM cache and conservative 128-slot
4 GiB GPU cache. Base weights, PLE, KV, activations and temporary uploads are
outside expert-cache budgets. These results do not establish whole-process
memory peaks, concurrent throughput/tails, ARM/Metal qualification or vLLM parity.

Evidence: `/tmp/bitnet-goal-hc-launch-final/qualification.json`,
`trace-comparison.json`, `timings.json`, `decode-comparison.json`,
`timing-provenance.json` and `initial1047.nsys-rep`; baseline trace/provenance:
`/tmp/bitnet-goal-current-transfer-profile/`.
Frozen executable SHA-256:
`bc09b508cf51045f59a851c525cd743f2c8993363a67ed5f2870ec105e1dd234`.

### Keep HC pre-mix intermediates on-device (2026-11-06)

An optional backend HC prefill operation now composes grouped RMS normalization,
down projection, scaled SiLU, up projection, stream mixing and injection
projection. Residuals upload once; normalized rows, low-rank values and gates
remain in an aligned backend workspace. Only mixed activations and injection
values return to the host. The existing primitive kernels and projection
geometry remain unchanged. Both host outputs are staged before commit.

CUDA supports 4–2048 tokens, F32/native-Q8 down/up projections and F32 injection.
Unsupported shapes/types, single-token calls, output HC without injection, host
reference execution and debug dump paths retain the staged implementation.
This is one HC stage, not a fully device-resident layer or inference graph.
The workspace is temporary backend memory outside expert-cache budgets.

Eighteen fused/unfused coherence cases compare against the established staged
GPU API: F32 and Q8, irregular ranks, token counts 4/9/129/1047 and the model's
10240-wide, rank-320 shape. All 414,158 output words agree exactly, with guards
and unchanged outputs on the tested rejection path. Existing independent
Q8/HC references, standard/full CUDA, diagnostic isolation, backend matrix and
WASM syntax checks pass. The protected clean build has no warnings. Qwen27 keeps
192/192 short and 24/24 long IDs; Flash keeps 179/179 short IDs and 16/16 on all
eight matched schedules. All 170 initial/resumed state captures remain exact
(72,327,168 words); the new-charge replay keeps 32/32 IDs. The known separate
cross-schedule Flash long CLI mismatch remains unqualified.

A real Flash diagnostic records 95 successful initial HC batches and one
intentional single-token rejection, retaining all 16 initial/resumed IDs.
Compared with the preceding batched-launch checkpoint:

| Initial 1047-token CUDA trace | Staged HC | Device HC intermediates |
| --- | ---: | ---: |
| HtoD bytes | 89,522,335,108 | 72,971,359,108 |
| DtoH bytes | 26,408,132,096 | 18,005,328,896 |
| HtoD calls | 54,186 | 53,616 |
| DtoH calls | 21,640 | 21,260 |
| Kernel launches | 105,611 | 105,611 |
| Kernel time | 7.604 s | 7.599 s |

This removes 16.551 GB HtoD and 8.403 GB DtoH (24.954 GB combined), with every
kernel-launch count preserved. Copy reductions match the four wide and two
low-rank uploads plus two wide and two low-rank readbacks eliminated per eligible
HC batch. Synchronous-copy API wall time is still about 10.3 seconds, and roughly
51,800 allocations/frees remain. API wall times include waits; they are not
independent costs to add to GPU times.

Three untraced process pairs in AB/BA/AB order give:

| Version | Initial prefill samples (seconds) | Median |
| --- | --- | ---: |
| Staged HC | 20.569, 20.818, 21.033 | 20.818 s |
| Device HC intermediates | 19.602, 19.424, 19.646 | 19.602 s |

Median initial prefill improves 5.8% in these samples, with all three pairs
faster and all 16 independent IDs preserved per process. The same runs provide
42 decode-plus-next-argmax/logging intervals per version: mean 333.7 versus
333.3 ms, median 317.8 versus 322.6 ms. These exclude each phase's final forward
and do not establish a decode gain or regression. Prefill timing uses the
fixture's profiler window without Nsight; the CUDA traces are separate runs.

Configuration: physical GPU 1, eight CPU threads, F32 KV, four qualified Flash
flags, warm filesystem cache, 2 GiB RAM cache and a conservative 128-slot/4 GiB
GPU expert cache. Base weights, PLE, KV, activations and temporary workspaces
are outside expert-cache budgets. No statistical throughput/tail, whole-process
VRAM peak, concurrent-serving, ARM/Metal or vLLM parity qualification is claimed.
Remaining host/device boundaries include HC post-combine and MoE gathering/
reduction; transient expert allocation/upload churn also remains.

Evidence: `/tmp/bitnet-goal-hc-chain-final/qualification.json`,
`trace-comparison.json`, `timings.json`, `timing-provenance.json`,
`diagnostic.log`, `diagnostic-exit.json` and `initial1047.nsys-rep`.
Baseline executable/trace: `/tmp/bitnet-goal-hc-launch-final/`.
Frozen executable SHA-256:
`86fdf19aa2282a218b0dd27667e31913c964438256c9145b8b976ad1a03f243a`.

### Skip unused packed preparation for gathered experts (2026-11-06)

Explicit gathered-prefill uploads now concatenate split gate/up payloads into
quant-only CUDA buffers when the quant registry declares the gathered K-quant
path. Those kernels consume raw rows and do not use the packed MMQ copy built
by the ordinary stacked-buffer constructor. The GPU policy module owns this
allocation decision; the bridge composes that policy without format branches.

Ordinary decode, tokenwise prompt forward and resident preload retain their
packed construction. Existing cached buffers remain reusable during prefill.
Backends without the optional quant-only constructor use their established
stacked constructor. The focused regression checks byte-for-byte concatenation,
constructor selection across gathered/decode/tokenwise calls, cleanup on upload
failure, and fallback when the optional constructor is absent. No checkpoint
routing or arithmetic kernels are changed.

The corrected protected clean run passes the architecture check, standard/full
CUDA suites, diagnostic isolation, backend matrix and WASM syntax checks, with
zero build warnings. Qwen27 retains 192/192 short and 24/24 long IDs; Flash retains
179/179 short IDs and 16/16 on all eight matched schedules. All 170 state captures
remain exact (72,327,168 words), and the new-charge replay retains 32/32 IDs.
The separate cross-schedule Flash long CLI mismatch remains unqualified.

| Initial 1047-token trace | Packed temporary gate/up | Quant-only temporary gate/up |
| --- | ---: | ---: |
| Q4_K packing launches | 13,674 | 0 |
| Total kernel launches | 105,611 | 91,937 |
| CUDA allocations | 51,766 | 38,092 |
| CUDA frees | 51,759 | 38,085 |
| GPU kernel time | 7.599 s | 7.536 s |
| HtoD bytes | 72,971,359,108 | 72,971,359,108 |
| DtoH bytes | 18,005,328,896 | 18,005,328,896 |

Every remaining kernel-launch count and all transfer counts/bytes are identical.
Allocation/free API wall time drops from about 3.38 to 2.43 s; API times include
waits and must not be added to kernel times. This avoids auxiliary preparation,
not raw expert uploads or MoE gathering/reduction transfers.

Three untraced pairs in AB/BA/AB order preserve all 16 independent IDs per process:

| Version | Initial prefill samples (seconds) | Median |
| --- | --- | ---: |
| Packed temporary gate/up | 19.560, 19.562, 19.437 | 19.560 s |
| Quant-only temporary gate/up | 18.701, 18.651, 18.503 | 18.651 s |

Median initial prefill improves 4.6% in these samples. The same runs contain 42
forward-plus-next-argmax/logging intervals per version: mean 328.1 versus 333.9 ms,
median 316.4 versus 319.8 ms. Decode samples are slower, so this does not establish
an overall prefill-and-decode performance gain.

A follow-up repeated-request pair links the prior qualified bridge with the
current remaining objects as its control. Both processes reset the session and
retain model caches between two identical requests; both preserve 32/32 IDs.
The second request's prefill improves 19.385 to 18.561 s, but its 14 measured
decode intervals give mean 311.3 versus 325.0 ms (4.4% slower), median 313.5
versus 325.9 ms. The first request's mean is 332.7 versus 325.1 ms. These limited
samples do not isolate a cause or prove a statistical regression; warmed decode
remains an explicit follow-up. Ordinary decode's constructor selection is
unchanged and covered by the focused unit test.

A subsequent ABBA diagnostic uses two identical requests per process and pins
all eight CPU threads to physical cores 0–7 sharing L3. The control substitutes
only the previous qualified bridge object; all other objects and CUDA kernels
are shared. All four processes pass 32/32 independent predictions (128 total).
At every corresponding decode step, I/O bytes, RAM/VRAM cache hits and misses,
constructor counts, destructor counts and uploaded payload bytes are identical
across all four processes. No production source changes accompany this probe.

| Second request, direct forward | Packed control | Quant-only prefill |
| --- | ---: | ---: |
| Samples | 32 | 32 |
| Mean | 322.385 ms | 322.498 ms |
| Median | 324.611 ms | 324.725 ms |
| Mean expert I/O | 66.502 ms | 66.338 ms |
| Mean buffer construction | 166.923 ms | 166.901 ms |
| Mean buffer destruction | 64.196 ms | 64.400 ms |

Direct timing includes all eight forwards per phase and excludes CPU argmax
and logging. The mean change is +0.035%; the earlier 4.4% warm slowdown does not
repeat in this fixed-affinity control. Different affinity and timing boundaries
prevent combining these samples with the earlier unrestricted measurements or
attributing their difference to CPU migration. Constructor times include CPU
concatenation, allocation, upload, packing and synchronization; destructor times
can include GPU waits. Their approximately 231 ms combined duration identifies
buffer lifecycle work as the next investigation, without establishing pure
allocator cost or the benefit of reuse. Evidence and hashes:
`/tmp/bitnet-goal-decode-work/{results,summary,provenance}.json`.

A temporary allocation interposer then tests exact-size reuse without modifying
production code. It tracks legacy `cudaMalloc`/`cudaFree`, synchronizes the device
before retaining an allocation, and bounds inactive storage to 64 entries and
16 MiB. The experiment uses the same fixed-affinity ABBA layout, two identical
requests per process and the configuration below. The off control loads the same
interposer and tracks allocations without retaining them. The interposer covers
all intercepted legacy allocations, including scratch, so these results do not
establish the gain from a weight-only backend pool.

A first variant rejects admission when full. It improves first-request prefill
18.767 to 17.421 s but leaves repeated prefill (18.562 to 18.523 s) and warmed
decode (322.916 to 322.910 ms) effectively unchanged. Inactive allocations from
earlier phases occupy the pool; only 22,070 of 157,518 allocation requests reuse
storage. This is not the policy selected for further implementation.

The second variant evicts the oldest inactive allocations to admit current
shapes. Two independent enabled processes have identical allocation counters:
149,472 reuses and 8,046 underlying allocations (94.9% reuse), 6,847 evictions,
peak inactive storage 16,774,692 bytes, zero synchronization errors and zero
tracking overflows. All four ABBA processes pass 32/32 independent predictions
(128 total). Per-step expert read bytes, RAM/VRAM hits and misses, constructor
counts, destructor counts and uploaded payload bytes are identical across them.

| LRU allocation diagnostic | Reuse off | Reuse on | Change |
| --- | ---: | ---: | ---: |
| First prefill, mean of two processes | 18.891 s | 15.774 s | -16.5% |
| Second prefill, mean of two processes | 18.561 s | 15.657 s | -15.6% |
| Warm direct forward, 32 samples/version | 315.921 ms | 204.722 ms | -35.2% |
| Warm direct-forward median | 317.243 ms | 205.833 ms | |
| Mean expert I/O | 65.718 ms | 66.051 ms | |
| Mean buffer construction | 163.123 ms | 106.192 ms | |
| Mean buffer destruction | 62.464 ms | 7.858 ms | |

The result supports implementing backend-owned allocation reuse with explicit
ownership, bounded inactive storage, error cleanup and synchronization before
reuse. It is not production qualification or vLLM parity. The interposer retains
idle allocations until process exit and supports this single-visible-device
experiment; a library implementation must release its own pool at backend
destruction. The 16 MiB is separate from the expert-residency budget and does not
bound total VRAM. No claim about cold filesystem I/O, concurrent executors,
bit-exact recurrent state or other models follows from these token checks.
Sources, logs, input hashes and summaries:
`/tmp/bitnet-goal-allocation-probe/` and
`/tmp/bitnet-goal-allocation-lru-probe/`.

The standard baseline tests pass. The separately linked full CUDA suite also
passes with the LRU interposer enabled, including its independent reference-word
checks; diagnostic isolation passes separately. The CUDA suite records 170,730
allocation reuses, peak inactive storage exactly 16 MiB, zero synchronization
errors and zero tracking overflows. These checks support the prototype but do
not replace model/state qualification of the eventual backend-owned pool.
Test logs and hashes are retained in the latter directory's `checks.json`.

Configuration matches the preceding checkpoint: physical GPU 1, eight CPU
threads, F32 KV, four qualified Flash flags, warm filesystem cache, 2 GiB RAM
cache and conservative 128-slot/4 GiB GPU expert cache. Timing uses profiler
markers without Nsight; traces are separate runs. Decode intervals include CPU
argmax/logging and exclude each phase's final forward. No vLLM parity, concurrent
throughput/tail, strict total-VRAM or ARM/Metal qualification follows.

Evidence: `/tmp/bitnet-goal-raw-gateup-checked/qualification.json`,
`trace-comparison.json`, `timings.json`, `warm-comparison.json`, `warm.c`,
`initial1047.nsys-rep` and the clean/standard/backend-matrix logs.
Baseline: `/tmp/bitnet-goal-hc-chain-final/`.
Frozen executable SHA-256:
`7b063381b1076361e2f7b614eb5123aeb2b1239a4b23a7a0f024e5f67b794b2b`.

### Backend-owned CUDA weight allocation reuse (2026-11-06)

`BN_CUDA_ENABLE_WEIGHT_POOL=1` enables a backend-local inactive allocation pool.
Leave the variable unset to disable it; policy flags use presence semantics and
are captured when the backend is created. Raw weights and their owned packed,
FP16 and FP32 layouts share exact-size storage after destruction. Activation and
scratch allocations keep their existing paths. The pool retains at most 64
allocations and 16 MiB of requested inactive payload, evicting the oldest entries
to admit current shapes. This excludes driver allocation overhead and is separate
from the resident expert-cache budget; it does not bound total VRAM.

Each allocation completes a device fence before admission, preserving the old
destruction lifetime across default-stream preparation and nonblocking decode.
Failed fences never publish reusable storage. A failed weight allocation trims
inactive storage and retries once. Backend destruction releases the pool;
`bn_gpu_cuda_trim_weight_pool()` releases only inactive storage explicitly.
`bn_gpu_cuda_weight_pool_stats()` reports current payload/count, limits and
cumulative hits, misses, evictions and fence failures. Access shares the backend's
existing serialized-executor requirement. No handles move onto CPU model weights.

The protected clean CUDA build has zero warnings. Standard and full CUDA tests,
diagnostic isolation, backend matrix, AVX2/AVX512 syntax checks and AVX512 quant
tests pass. Focused tests verify new contents after reuse, entry/byte limits,
eviction, oversized-allocation bypass, trim and backend isolation. Separate fault
injection verifies failed-fence rejection, allocation-failure trim/retry and
failure propagation with an empty pool. The policy source also passes syntax
checking with `__EMSCRIPTEN__`; this is not a WASM runtime qualification.

With the pool enabled, Qwen27 retains 192/192 short and 24/24 long independent
IDs. Flash retains 179/179 short IDs, 16/16 on all eight matched prefill/resume
schedules and all 170 captured states bit-for-bit (72,327,168 words). The actual
242-slot cache-charge replay retains 32/32 IDs. The separate cross-schedule Flash
long CLI mismatch remains open; neither reference IDs nor reference math change.

Four serialized processes in off/on/on/off order each execute two identical
requests with session reset and retained model caches. Each passes 32/32 IDs,
for 128 total. All corresponding per-forward expert read bytes, RAM/VRAM hit and
miss counts, constructor/destructor counts and uploaded payload bytes match.
Configuration: physical GPU 1, eight threads pinned to physical cores 0–7 sharing
L3, F32 KV, the four qualified Flash flags, warm filesystem cache, 2 GiB RAM cache
and conservative 128-slot/4 GiB GPU expert cache. Timed runs use the same frozen
objects and replay binary, with the pool policy absent or present; no allocation
interposer or Nsight is loaded during timing. The marker timer is shared.

| Weight pool comparison | Off | On | Change |
| --- | ---: | ---: | ---: |
| First 1047-token prefill, two-process mean | 18.719 s | 15.825 s | -15.5% |
| Second prefill, two-process mean | 18.604 s | 15.550 s | -16.4% |
| Warm direct forward, 32 samples/version | 323.689 ms | 203.365 ms | -37.2% |
| Warm direct-forward median | 325.668 ms | 204.395 ms | |
| Mean expert I/O | 66.415 ms | 65.432 ms | |
| Mean buffer construction | 167.814 ms | 105.690 ms | |
| Mean buffer destruction | 64.696 ms | 7.678 ms | |

Direct timing includes all eight forwards per phase and excludes CPU argmax and
logging. Constructor/destructor wall times include uploads, preparation and
waits; they are not pure allocator costs. These are limited serialized samples,
not concurrent throughput, tail latency or vLLM parity measurements.

Separate first-prefill traces preserve all 91,937 kernel launches by name/count,
all transfer counts and bytes (HtoD 72,971,359,108; DtoH 18,005,328,896; DtoD
25,731,072), and all 32 predictions per traced process. CUDA allocations drop
38,092 to 9,606 and frees 38,085 to 9,594. Explicit device synchronizations rise
1 to 28,492 as they replace the lifetime fences implicit in frees. Allocation
plus free API wall time drops about 2.441 to 0.161 s; explicit device-sync API
time is about 0.022 s with reuse. Kernel time remains about 7.54 s. API times
include waits and must not be added to kernel times. This removes allocation
churn; raw expert uploads and host MoE gathering/reduction remain.

Checkpoint: `/tmp/bitnet-goal-weight-pool/qualification.json`, with source and
reference hashes, clean/test logs, `performance.json`, `timings.json`,
`trace-comparison.json`, fault-injection sources and both Nsight traces.
Frozen executable SHA-256:
`3d0e44324148f1a552338137c3fb5d21c7b269ee8f829e3b7ef2df47bf8f2956`.

### vLLM checkpoint and precision compatibility gate (2026-11-06)

The local vLLM runtime is available. Its pinned image is
`ghcr.io/local-inference-lab/vllm@sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`,
reporting vLLM `0.1.dev21830+g8244d1c44.cu134` and PyTorch
`2.14.0a0+4fdf77b940.nv26.8.63802676`. The base image rejects quantization
method `gguf`; its installed method registry contains no GGUF implementation.

The official [GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin/tree/e454affe0ca23c9dc8754ec3033270cb30622ed5)
at commit `e454affe0ca23c9dc8754ec3033270cb30622ed5` builds successfully against
that image's PyTorch and CUDA, targeting SM120a. The wheel hash is
`d6dd44400cc02ca7f3a0452f32589d2eed6588da10ff04858d78b551f9f828ef`.
The pinned llama.cpp `gguf-py` package supplies the Python GGUF reader; the
original vLLM image and model files are unchanged.

The probe uses the exact existing Qwen27 Q4_K_XL file, SHA-256
`3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
It contains F32, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_NL and IQ4_XS
tensors. It is not a uniform four-bit checkpoint. The other locally available
safetensors checkpoint mixes NVFP4, FP8 and BF16, with separate activation
quantization and global scales; comparing its runtime to Q4_K_XL would not
establish matched-quantization parity. BitNet currently lacks the NVFP4 format
and that mixed-precision activation/scaling contract.

For text-only loading, the probe extracts the existing HF `text_config` from
the local Qwen27 metadata and omits the safetensors quantization declaration,
since formats come from the GGUF. A one-line runtime adapter correction maps
`QWEN35_ARCHITECTURES['qwen3_5_text']` to vLLM's existing
`Qwen3_5ForCausalLM`. The plugin otherwise selects the conditional-generation
wrapper, which requires vision metadata/projector tensors even for this text
probe. The correction selects the native text implementation; it changes no
checkpoint values or model kernels. All failed attempts are retained.

| Proposed comparison | Verified status |
| --- | --- |
| Same GGUF, F32 activations/KV/state | Weights load, but both tested GDN prefill implementations reject F32 |
| Same GGUF, BF16 activations/KV and F32 recurrent state | vLLM generation works; BitNet has no matching BF16 activation/KV path yet |
| Native mixed NVFP4/FP8 safetensors versus Q4_K_XL | Different quantization and scaling contracts; no parity ratio is valid |

The F32 probe loads 17.21 GiB of GPU weights. Automatic GDN prefill selects
FlashInfer/CuteDSL and fails with `DSL kernel only supports fp16/bf16 inputs`.
Selecting the existing Triton GDN option also fails: `ChunkGatedDeltaRuleFunction
does not support float32`. These are observed limitations of this pinned
image/backend combination, not claims that the model or vLLM is unavailable.
No precision casts or assertions were changed to force the comparison through.
A separate selector probe confirms that explicit in-tree CuteDSL falls back to
Triton on SM120; the image's `b12x` GDN path requires BF16 and rejects F32 before
execution. The BF16 `b12x` selection succeeds, but its generation/performance is
not tested here. The observed BF16 vectors below come from explicit Triton GDN.

BF16 generation succeeds with the same GGUF using explicit Triton GDN prefill
and decode, automatic BF16 KV, F32 SSM state, maximum context 512, concurrency
one, disabled prefix caching and eager execution. This is a compatibility probe,
not a performance baseline: no latency ratio against BitNet's F32 path follows.
Normal generation produces eight IDs:
`[11751, 13, 198, 760, 6511, 314, 9564, 369]`.

A worker-side observer preserves all eight IDs and records 32 layer-0 input/output
buffers across initial prefill and the first decode forward: 799,296 BF16 words,
all finite. Eight observed modules cover normalization, GDN projections and
normalization, output projection and the dense FFN. Sampled layer-0/3 parameter
and cache metadata establish these distinctions:

| Value | Measured vLLM dtype |
| --- | --- |
| Captured activation inputs/outputs | BF16 |
| Attention KV at layer 3 | BF16 |
| Convolution history at layer 0 | BF16 |
| Recurrent SSM state at layer 0 | F32 |
| Layer-0 `A_log` parameter | F32 |
| Layer-0 `dt_bias` and norm parameters | BF16 |

TF32 is disabled in the observed model. FP16 KV storage in BitNet would not
match this contract: BF16 activation rounding, parameter precision and
convolution/KV storage must also be addressed. Loading BF16 weight tensors is
not itself a BF16 activation mode. The next comparison gate is an explicit
BitNet precision contract followed by native CUDA BF16 execution, validated
against these observed boundaries before timing equivalent workloads. Future
timing must use identical input IDs and matched output/graph policies; the
eight-token smoke test is not a full-model BF16 correctness qualification.

Evidence: `/tmp/bitnet-goal-vllm-parity/`, including `inventory.json`, the plugin
wheel/build log, F32 failure logs, successful BF16 generation and observer logs,
`precision-contract.json`, `captures/manifest.json`, `capture-validation.json`
and `probe-summary.json`. Worker hooks run only in the isolated local probe;
the successful capture explicitly shuts down its engine. BitNet's qualified
`3d0e4432…` executable and implementation sources remain unchanged.

### Native CUDA BF16 weighted normalization (2026-11-06)

The first typed BF16 backend operation is implemented:
`rmsnorm_bf16_batch`, with capability/dispatch facades in `gpu_backend.h`.
Inputs, outputs and normalization parameters use BF16 storage; reduction,
normalization and parameter-offset addition use F32. The optional offset selects
`gamma = float(w_BF16)` or `gamma = 1 + float(w_BF16)`. The effective gamma is
not rounded back to BF16 before multiplication. Parameter preparation and the
choice of offset remain caller/model-policy responsibilities; the CUDA kernel
contains no model-family branch and changes no CPU-visible model weights.

The host-entry callback stages BF16 data in backend-owned scratch, respects the
existing prefill/decode stream handoff, supports in-place output and commits
host output only after successful completion. Invalid shapes, parameter types,
short buffers, epsilon values and offset selectors reject without changing
output. Other backends leave this optional callback absent. This does not expose
a global BF16 inference option or change the existing F32 execution path.

| BF16 normalization check | Result |
| --- | --- |
| Independent synthetic cases | 44 |
| Distinct synthetic reference words | 50,952 |
| Separate/in-place comparisons | 101,904/101,904 bit-exact |
| Captured Qwen27 layer-0 words | 30,720/30,720 bit-exact in both output modes |
| Additional batch-size check | 65,537 rows |
| Failure injection | Partial readback preserves separate/alias output; retry succeeds |

Synthetic references come from the pinned image's
`GemmaRMSNorm.forward_native`, using Torch CUDA native IR with F32 reduction and
BF16 output. Tests require at most one BF16 ULP for possible reduction-order
differences; all current comparisons are exact. A separate strict boundary case
distinguishes F32 `1+w` from prematurely rounded BF16 gamma, so that allowance
cannot mask the parameter-precision error. Signed zero, input/output canaries,
unsupported capability, invalid calls and in-place execution are also checked.
The checked-in reference header is generated entirely from synthetic data and
requires no model, Docker or Python packages to run the unit test.

The real-model replay uses the preceding observer's unchanged first-normalization
inputs/outputs: five initial rows and the first decode row, each width 5120.
The source GGUF's F32 normalization vector is transformed to an offset parameter
and rounded to BF16 before upload. Both direct and in-place calls reproduce all
30,720 expected BF16 words. This qualifies that boundary, not an end-to-end BF16
model or its other captured operations.

The protected clean CUDA build is warning-free. Standard tests, the final full
CUDA suite, diagnostic isolation, backend matrix, AVX2/AVX512 checks and AVX512
quant tests pass. Existing F32 gates retain Qwen27 192/192 short and 24/24 long,
Flash 179/179 short, 16/16 on all eight matched prefill/resume schedules and all
170 captured states exactly (72,327,168 F32 words). The actual cache-charge
replay retains 32/32 IDs. The final full CUDA run includes the additional strict
parameter-rounding regression; production objects are unchanged by that test
addition. User-owned reference binaries are preserved and hash-checked.

Remaining BF16 work includes precision planning, matrix and residual/activation
operations, GDN execution and BF16 KV/convolution storage. The current global
inference contract remains F32, and the cross-schedule Flash long CLI mismatch
remains separately open. No whole-model BF16 correctness, throughput or matched
vLLM parity follows from this primitive. Backend access remains serialized.

Evidence: `/tmp/bitnet-goal-bf16-norm/qualification.json`, with the synthetic
generator and reference hashes, real-boundary replay, readback fault injection,
clean/test logs, original reference-ID hashes and 170-state comparison.
Frozen executable SHA-256:
`cfc31a055f4b63bf2b80bd24035dada97bdf8565d2b8c12f553c021f33d2d5bb`.

### Native CUDA BF16 matrix components (2026-11-06)

Added two optional backend operations: independent decoded BF16 weight layouts
for IQ4_NL/IQ4_XS and dense BF16 matrix multiplication. The quant registry owns
the decoded-layout declaration. Decoding performs the quantized format's FP32
scale/codebook arithmetic followed by one BF16 RNE conversion per weight.
The source handle is borrowed, remains unchanged and can be freed immediately
after layout preparation; the caller owns and budgets the new dense payload
at `2 * rows * cols` bytes. It is not an uncharged lazy attachment to raw
weights or a new MoE cache policy.

The GEMM callback accepts BF16 weights, activations and output, accumulates in
FP32 and disables reduced-precision reductions while preserving the previous
cuBLAS math mode. It supports host input/output aliasing, observes stream
handoffs and commits output only after staged readback succeeds. It currently
uses host entry/exit buffers rather than a resident BF16 model graph. Other
backends leave both optional callbacks absent; native BF16 GEMM is advertised
on CUDA SM80 or newer.

The pinned GGUF plugin's actual linear dispatch matters here. IQ formats are
excluded from `MMQ_QUANT_TYPES`: large batches enter `DEQUANT_TYPES`, invoke
CUDA BF16 weight dequantization and then Torch BF16 matrix multiplication.
Small IQ batches use A8 MMVQ (at most 16 rows, or 8 when output width exceeds
5120). K formats instead use A8 MMQ above their MMVQ threshold. Reading only
the lower-level Triton GEMM implementation gives a different precision contract;
those kernels are not selected by this default IQ linear dispatch. The existing
F32 path therefore cannot stand in for this BF16 operation simply by rounding
its output.

| BF16 matrix check | Result |
| --- | --- |
| Independent BF16 GEMM cases | 6; odd K, partial rows and batches through 129 |
| Distinct GEMM output words | 9,264 |
| Separate/in-place output checks | 18,528/18,528 bit-exact |
| Decoded IQ4 layouts | 8, covering both formats, signed scales and zero |
| Decoded BF16 words | 165,888; complete byte hashes match independent reference |
| Source ownership | Source destroyed before every decoded-layout readback |
| Weight allocation pool | Same focused tests pass with pool disabled and enabled |
| Combined decode + GEMM vs plugin linear dispatch | 24 cases, 115,992/115,992 bit-exact output words |
| Injected partial readback failure | Separate/alias output unchanged; retry succeeds |
| Checked-in reference regeneration | Byte-identical |

References use the previously pinned image, Torch and plugin wheel. Independent
Torch FP32 format decoding followed by BF16 conversion is checked against the
plugin's identity-matrix projection. Dyadic GEMM inputs give exact FP32 sums.
The generator is `test/reference_cuda_bf16_matrix.py`; ordinary tests use the
checked-in header and need no external model, Docker or Python packages. Layout
readback uses qualified BF16 RMSNorm on an all-ones input with zero epsilon,
so every scale is exactly one and signed-zero bits are retained.

The protected clean CUDA build has zero warnings. Standard/full CUDA tests,
diagnostic isolation, backend matrix and AVX2/AVX512 checks and quant tests pass.
F32 regressions retain Qwen27 192/192 short and 24/24 long, Flash 179/179 short,
16/16 on all eight matched prefill/resume schedules, 32/32 with the actual cache
charge, and all 72,327,168 words across 170 captured states exactly. User-owned
reference binaries are preserved and hash-checked.

These are backend primitives, not a complete BF16 inference mode. Precision
planning, remaining mixed GGUF formats, small-batch IQ/K-format A8 contracts,
resident intermediates, residual/activation operations, attention, GDN and BF16
KV/convolution storage remain. The cross-schedule Flash long CLI mismatch is
still separately open. No model-level BF16 correctness or matched vLLM
performance claim follows from this checkpoint.

Evidence: `/tmp/bitnet-goal-bf16-matmul/qualification.json` and
`precision-contract.json`, with reference generation, composite comparisons,
fault injection, clean/test logs, immutable reference-ID hashes and state replay.
Frozen executable SHA-256:
`cd80901848102ed3daa7e26993b8374feb28ad067c38ec825bda540c11b5de0f`.

### Explicit CUDA BF16 GGUF matrix dispatch (2026-11-06)

Added a backend-quant matrix plan and optional `matmul_quant_bf16` callback for
IQ4_NL/IQ4_XS. It preserves the pinned plugin's format/batch arithmetic:

| Condition | Execution | Additional decoded weight payload |
| --- | --- | --- |
| Output width <=5120 and tokens <=16 | A8 MMVQ | None |
| Output width >5120 and tokens <=8 | A8 MMVQ | None |
| Larger batches | Decoded BF16 GEMM, FP32 accumulation | `2 * rows * cols` bytes |

The planner uses registry-declared decode metadata, validates shapes/overflow
and rejects unsupported formats without changing its output plan. Large batches
require a matching caller-owned BF16 layout; the callback does not allocate or
admit a hidden weight copy. The raw handle identifies the original format.
The existing F32 dispatch and CPU-visible model weights remain unchanged.
`backend_quant.h` now declares its C linkage for CUDA/C++ consumers.

Small batches quantize BF16 activations directly into FP16-scale Q8 blocks on
the device, without an expanded F32 activation buffer. The dot kernel uses byte
permutations and DP4A with one warp per output row, preserving the reference
lane partition, scale multiplication order, FP32 reduction and final BF16 RNE
conversion. Large batches call the preceding native BF16 GEMM. Both paths stage
readback before committing host output and support in-place output. These are
host-entry operations, not a resident BF16 model execution graph.

| Dispatch qualification | Result |
| --- | --- |
| Independent synthetic cases | 104: 68 A8 MMVQ, 36 decoded GEMM |
| Distinct BF16 output words | 1,173,104 |
| Separate/in-place checks | 2,346,208 exact reference words |
| Thresholds | 8/9 and 16/17 tokens; output widths 5120/5121 |
| Input widths | 256, 768, 5120; includes plugin padding cases |
| Strict cuBLAS 13.7 replay | All outputs match that version's independent reference |
| Strict cuBLAS 13.10 replay | All outputs match that version's independent reference |
| Captured Qwen27 IQ4_XS gate/down projections | 135,168/135,168 BF16 words exact |
| Failure injection | Small/large, alias/separate outputs unchanged; retries succeed |
| Weight pool | Same focused suite passes with pooling enabled |
| Reference regeneration | Byte-identical combined header |

The larger-width cases exposed an environment difference: the pinned vLLM image
loads cuBLAS 13.7.0.27, while the host links 13.10.0.20. The same unchanged Torch
linear dispatch selects different FP32 reduction kernels for two IQ4_XS GEMMs
(output width 5121; 9 and 17 tokens). Their 15 differing output words include
cancellation-sensitive results with up to four BF16 ULPs of difference. This is
not an A8/decoded-layout dispatch error: using the image's unchanged libraries,
the native backend matches all original 13.7 reference hashes exactly. An
independent Torch run with the host's unchanged libraries matches the native
13.10 results exactly as well. The ordinary test retains both complete-output
hashes rather than adding an elementwise tolerance; separate strict replays
require the corresponding version's hash. Future matched performance/correctness
comparisons must pin the cuBLAS libraries as well as hardware, model, precision
and batch schedules. The library paths and SHA-256 hashes are recorded.

The real-model replay consumes unchanged captured BF16 inputs and outputs from
the preceding Qwen27 observer: layer-0 FFN gate and down projections for five
initial tokens and the first decode token. It reads the original GGUF weights
and uses the new small-batch callback. All 135,168 expected BF16 words match
without a tolerance. This qualifies those boundaries; the mixed Q3_K up
projection and whole BF16 forward path remain unimplemented.

Protected clean build has zero warnings. Standard/full CUDA, diagnostics,
backend matrix and AVX2/AVX512 checks and quant tests pass. Existing F32 gates
retain Qwen27 192/192 short and 24/24 long, Flash 179/179 short, 16/16 on all
eight matched schedules, 32/32 with the actual cache charge and all 72,327,168
words across 170 captured states exactly. User-owned reference binaries are
preserved and hash-checked.

Evidence: `/tmp/bitnet-goal-bf16-dispatch/qualification.json`, including both
independent reference manifests, strict library-specific replays, library hashes,
real-boundary replay, failure injection and model/state gates. Reproducible
reference sources are `test/reference_cuda_bf16_dispatch.{c,py}`; ordinary tests
need only the checked-in header. The original 13.7 reference values are retained.
Frozen executable SHA-256:
`fbf3ca0bbd5cea324b8f4a8a17e429496959f69fe9c458b645654cf882d88df5`.

Remaining work includes other mixed GGUF formats and their A8 contracts,
model-level precision planning, resident BF16 intermediates, residual/activation,
attention, GDN and BF16 KV/convolution storage. No full BF16 model mode or matched
vLLM throughput/latency parity is claimed. The cross-schedule Flash long CLI
mismatch and serving/concurrency qualification remain separately open.

### Native CUDA BF16 Q3_K matrix execution (2026-11-06)

Extended the explicit BF16 GGUF matrix plan and CUDA callback to Q3_K. Quant
registry metadata now distinguishes linear arithmetic from decoded-layout
support: IQ4 keeps its preceding profile, Q3_K declares A8 MMVQ/MMQ, and other
formats remain explicitly unsupported. Q3_K does not declare or allocate a
BF16 decoded-weight layout. Both Q3_K paths consume the original quantized
weight handle and BF16 activations/output, with FP16 Q8 activation scales and
FP32 accumulation. The existing F32 routing and kernels are unchanged.

| Q3_K condition | Execution | Additional decoded weight payload |
| --- | --- | --- |
| Output width <=5120 and tokens <=6 | A8 MMVQ | None |
| Output width >5120 and tokens <=2 | A8 MMVQ | None |
| Larger batches | Tiled A8 MMQ | None |

The small path uses one warp per output row and preserves the reference's
four-value lane dots and inner/outer FP32 sums. The 128-row by four-token MMQ
tile shares activation blocks across output rows and decodes each weight group
once for four tokens. Each pair of signed 16-value subblock dots is combined
as integers before FP32 scaling, matching the pinned plugin's MMQ arithmetic.
This differs from the existing F32-input path, whose reduction contract is
preserved. DP4A handles the packed integer dots; output uses BF16 RNE rounding.
No expanded F32 activation buffer or hidden weight materialization is needed.

The common symmetric BF16-to-Q8 quantizer now uses a flat grid. This removes
its 65,535-token grid-Y limit while retaining the IQ4 results exactly. Q3_K MMQ
also flattens its output tiles, handles partial rows/tokens without divergent
barriers, and stages host readback before committing separate or in-place
output. Large requests remain subject to shape/overflow checks and available
backend scratch memory; this is not model-level context or serving qualification.

| Q3_K qualification | Result |
| --- | --- |
| Independent synthetic cases | 61: 30 MMVQ, 31 MMQ |
| Distinct BF16 output words | 329,321 |
| Separate/in-place checks | 658,642 exact reference words |
| Dispatch thresholds | 2/3 and 6/7 tokens; output widths 5120/5121 |
| Input widths | 256, 768, 5120; includes reference activation padding |
| Long matrix batch | 65,537 tokens, exact; flat quantization/output grids |
| Captured Qwen27 FFN-up projection | 104,448/104,448 BF16 words exact |
| Injected partial readback failure | MMVQ/MMQ, alias/separate output unchanged; retry succeeds |
| Weight pool | Same focused reference suite passes with pooling enabled |
| Checked-in generator | Reproduces reference header byte-for-byte |

References use the unchanged pinned image, plugin CUDA extension and Torch
linear dispatcher. The first 60 reference cases are preserved when adding the
long-batch case. The reference quantizer itself cannot launch a 65,537-row grid;
that additional reference concatenates two unchanged MMQ calls of 32,768 and
32,769 tokens. Both retain MMQ dispatch and identical independent per-token
linear arithmetic. The runtime evaluates the whole batch in one callback.
Q3_K A8 execution does not use cuBLAS or accept a per-element numerical
allowance. Both digest fields in the shared test schema contain the same
independent reference hash. Reproducible sources are
`test/reference_cuda_bf16_q3.{c,py}`; ordinary tests need only the checked-in
reference header and synthetic C fixtures.

The real-model replay reads the original GGUF's layer-0 Q3_K FFN-up weights
and the unchanged observer's captured BF16 inputs/outputs. Five prefill tokens
select MMQ at output width 17408; the first decode token selects MMVQ. All
87,040 prefill and 17,408 decode output words match exactly. Together with the
preceding IQ4 gate/down boundary qualification, all three layer-0 FFN matrix
boundaries have independent BF16 reference coverage. This does not qualify their
residual/activation composition or a complete BF16 model forward pass.

The protected clean build has zero warnings. Standard/full CUDA tests,
diagnostic isolation, backend matrix and AVX2/AVX512 checks and quant tests pass.
The full CUDA suite retains preceding IQ4 matrix and normalization references.
F32 model gates retain Qwen27 192/192 short and 24/24 long, Flash 179/179 short,
16/16 on all eight matched schedules, 32/32 with the actual cache charge and
all 72,327,168 words across 170 captured states exactly. User-owned reference
binaries are preserved and hash-checked.

Evidence: `/tmp/bitnet-goal-bf16-q3/qualification.json`, with independent
reference generation, long-batch comparison, real-boundary replay, fault
injection and the model/state gates. Frozen executable SHA-256:
`b7ab0443a1f7100a89a880611030f7841ca2f9934b6be0204da0a77c2ccec7f0`.

Other mixed GGUF formats, model-level precision planning, resident BF16
residual/activation, attention, GDN and KV/convolution execution remain open.
No full BF16 model mode or measured vLLM throughput/latency parity is claimed.
The cross-schedule Flash long CLI mismatch and serving/concurrency qualification
remain separate open work.

### Resident CUDA BF16 SwiGLU FFN (2026-11-06)

Added optional `dense_ffn_bf16` for
`down(silu(gate(X)) * up(X))`, using the preceding qualified GGUF matrix
profiles. Gate, up, activation and down intermediates stay on the device.
The callback uploads the BF16 input once and reads back only the BF16 output;
it contains no normalization, residual addition or F32 fallback. It supports
in-place output and commits host output only after successful staged readback.
All weight/profile/shape checks run before compute begins.

The borrowed weight descriptor carries each original GGUF type, raw handle and
optional independently budgeted BF16 layout. Large-batch GEMM accepts a decoded
layout after the raw GPU payload has been released. A8 plans still require raw
weights, even if a compatible BF16 layout is present; a cache miss cannot change
the selected arithmetic. Tests cover both ownership modes, reject mismatched
format tags and reject a decoded layout substituted for required A8 execution.
No CPU-visible model weights are mutated and no weight copy is admitted lazily.

The matrix entry points now share validated device execution helpers. A8 gate
and up reuse one BF16-to-Q8 input quantization. The down projection prepares its
own activation input only when its plan requires A8. Two BF16 hidden vectors
reuse backend prefill scratch; the gated product overwrites the gate vector.
All scratch is reserved before launching the composed sequence. The active
hidden payload is `4 * tokens * hidden_dim` bytes, plus input/output/Q8 scratch;
these activation allocations are separate from weight-cache budgets.

The reference's activation precision is significant: CUDA `silu_and_mul` rounds
SiLU to BF16 before the multiplication, then rounds the product to BF16. In a
65,536-word independent probe, this matches every reference word; one final
rounding of an FP32 SiLU/product matches only 52,689 words. The native kernel
preserves both conversions. On the unchanged Qwen observer data, the resident
activation intermediate matches all 104,448 distinct captured BF16 words exactly
(208,896 comparisons including separate/in-place execution).

| Resident FFN qualification | Result |
| --- | --- |
| Independent synthetic FFNs | 160; four format combinations, two seeds, four geometries, five batch sizes |
| Formats | IQ4_XS/Q3_K/IQ4_NL, including entirely IQ4_XS FFNs |
| Distinct FFN output words | 798,720 |
| Separate/in-place output checks | 1,597,440 exact reference words |
| cuBLAS 13.7 and 13.10 reference runs | Identical full FFN outputs and headers for all covered cases |
| Source ownership | Every GEMM raw handle destroyed before ordinary test execution |
| Fresh decoded-only context | 8,448 exact output words, all raw GPU handles released |
| Captured Qwen27 FFN outputs | 30,720 distinct words exact; 61,440 separate/in-place comparisons |
| Captured activation intermediate | 104,448 distinct words exact in both output modes |
| Readback fault injection | A8/mixed GEMM, alias/separate output untouched; retry succeeds |
| Weight pool | Final 160-case suite passes with pooling enabled |
| Reference generation | Both independent library runs reproduce the checked-in header |

The real-model replay uses layer-0 weights from the original Qwen27 GGUF and
the unchanged BF16 observer input. It computes all three projections and the
activation, then compares against the captured down-projection output. Five
prefill tokens exercise IQ4 MMVQ plus Q3_K MMQ; one decode token uses MMVQ.
An external passive launch observer reads the fourth kernel's gated activation
and compares it to the original down-projection input. Observed and unobserved
calls preserve every expected output. No test hook is added to production code.
This validates the core FFN composition, not its normalization/residual boundary
or the whole BF16 model.

Host/device traffic was measured separately with capture disabled and weight
uploads excluded. The baseline is three host-entry BF16 matrix calls, with the
unchanged captured gated input supplied to down; it measures projection traffic
without attributing a CPU activation cost. It is not the existing F32 resident
FFN path or a vLLM latency benchmark.

| Qwen27 sample | Separate matrix calls H2D / D2H | Resident FFN H2D / D2H |
| --- | --- | --- |
| Five prefill tokens | 276,480 / 399,360 bytes; 3 / 3 calls | 51,200 / 51,200 bytes; 1 / 1 calls |
| One decode token | 55,296 / 79,872 bytes; 3 / 3 calls | 10,240 / 10,240 bytes; 1 / 1 calls |

This removes 84.8% of the measured tensor traffic in both samples. Each resident
call has six observed custom launches: two quantizers, three projections and
one activation. No explicit device/stream synchronization occurs between stages;
allocation occurs before compute and final readback is a host boundary. The
passive activation observer's extra synchronization/copy is excluded from the
traffic run. No throughput or latency parity claim follows from these counts.

Protected clean build is warning-free. Standard tests, the final full CUDA
suite including all 160 FFNs, final diagnostics, backend matrix and AVX2/AVX512
checks and quant tests pass. Preceding BF16 matrix, Q3_K and RMSNorm reference
tests pass after the device-helper refactor. F32 model gates retain Qwen27
192/192 short and 24/24 long, Flash 179/179 short, 16/16 on all eight matched
schedules, 32/32 with actual cache charging and all 72,327,168 words across
170 captured states exactly. User-owned reference binaries are preserved and
hash-checked. The extra final CUDA run includes the decoded-only IQ4 FFNs;
production objects did not change when those reference cases were added.

Evidence: `/tmp/bitnet-goal-bf16-ffn/qualification.json`, with both independent
FFN reference runs, activation-contract probe, source-ownership tests, transfer
counts, passive intermediate capture, fault injection and model/state gates.
Reference sources are `test/reference_cuda_bf16_ffn.{c,py}`; normal tests need
only checked-in data and C fixtures. Frozen executable SHA-256:
`d40cf678301d4fac6a6628865b546749552839dd96766b221cae6d687731744b`.

Remaining work includes other mixed GGUF matrix formats, model-level precision
planning, resident BF16 normalization/residual boundaries, attention, GDN and
KV/convolution storage. The F32 model execution contract remains the default.
A full BF16 model mode, measured matched-vLLM parity, serving/concurrency and
the separate cross-schedule Flash long CLI mismatch remain open.

### CUDA BF16 residual/RMSNorm/FFN composition (2026-11-06)

Optional `add_rmsnorm_bf16_batch` and `dense_ffn_bf16_add_norm` extend the
preceding FFN checkpoint. With BF16 inputs X/R, F32 sum S=X+R drives RMSNorm;
the second output is BF16(S). The norm parameter is BF16, with gamma formed
in F32 using the declared zero/one offset. Normalizing BF16(S) instead changes
8,381 of 30,720 captured Qwen27 norm words. Independent unchanged vLLM IR
reproduces every captured norm word using the F32 sum.

The composed callback returns FFN(norm(S)) and BF16(S), matching Qwen's
post-attention/MLP boundary; it does not add the FFN output to the residual.
It validates plans and reserves scratch before compute, uploads both inputs,
keeps normalized input and FFN intermediates resident, and reads both outputs
back together. Host outputs are committed only after successful readback.
Outputs must be disjoint but can independently alias either input, including
cross-input swapping. Other backends retain absent optional capabilities.
Existing F32 model execution is unchanged.

| Qualification | Exact result |
| --- | --- |
| Standalone add/RMSNorm | 38 cases, 1,480,784 comparisons across both outputs and four arrangements |
| Composed add/norm/FFN | 48 cases, 2,293,760 comparisons across both outputs and four arrangements |
| Norm geometries | Dimensions 1 through 5,120, warp/block boundaries, 65,537-token grid, gamma offsets 0/1 |
| FFN geometries | 256/256, 768/512, 256/5,376; batches 1/5/17/33; IQ4_XS with IQ4_XS or Q3_K up |
| Original Qwen27 layer-0 replay | Five prefill tokens and one decode token; both outputs exact in all four arrangements, for norm and composed FFN |
| Qwen replay comparisons | 491,520 exact BF16 words; 30,720 distinct positions in each logical vector |
| Independent reference regeneration | Checked-in Python reproduces checked-in header byte for byte |
| cuBLAS versions | Native tests pass against host 13.10 and pinned reference 13.7 |
| Pooling | All 86 cases pass with weight pooling enabled |
| Readback failures | 16 partial copies crossing into the second vector; both outputs/inputs unchanged, retries exact |

An initial composed case exposed one normalized BF16 word differing at a
rounding boundary, propagating to 91 FFN words. It reproduced with both cuBLAS
versions. An isolated kernel comparison fixes it by explicitly preserving the
F32 addition of epsilon after mean-square division. The new kernel uses
`__fadd_rn`; references and earlier norm kernels are unchanged. No tolerance
was introduced. GEMM raw weights are released before synthetic execution;
required A8 raw handles remain present.

Transfer counts use the original Qwen inputs, exclude weight uploads and
observer copies, and compare separate norm plus resident FFN to composition:

| Sample | Separate H2D / D2H | Composed H2D / D2H |
| --- | --- | --- |
| Five tokens | 153,600 / 153,600 bytes; 3 / 2 calls | 102,400 / 102,400 bytes; 2 / 1 calls |
| One token | 30,720 / 30,720 bytes; 3 / 2 calls | 20,480 / 20,480 bytes; 2 / 1 calls |

Composition removes one third of measured tensor traffic. Both paths issue
seven observed custom launches; neither explicitly synchronizes the stream or
device between stages. Final host readback is a synchronization boundary. Both
measured paths retain exact outputs. These counts establish no latency or
throughput improvement and no vLLM parity result.

The protected clean CUDA build has zero warnings. Standard/full CUDA,
diagnostic isolation, backend matrix, AVX2/AVX512 checks and quant tests pass.
Qwen27 retains 192/192 short and 24/24 long tokens; Flash retains 179/179 short,
16/16 on all eight matched schedules and 32/32 with actual cache charging.
All 72,327,168 words across 170 initial/resumed states remain exact. Preserved
user reference binaries are hash-checked.

Evidence: `/tmp/bitnet-goal-bf16-residual/qualification.json`. Independent
reference generator: `test/reference_cuda_bf16_residual.py`; synthetic expected
hashes: `test/cuda_bf16_residual_reference.h`. References use the preceding
pinned vLLM image/plugin, unchanged native IR, plugin linear and BF16 activation.
Frozen executable SHA-256:
`bccfdccd7e4fb1464382751f6c434de189c0912700c28e2f7214684ce1a66b9e`.

This qualifies one resident decoder boundary. Full BF16 model execution,
remaining matrix formats, attention/GDN/KV state, serving, matched vLLM
performance and the cross-schedule Flash long mismatch remain open.

### BF16 GDN linear reference topology (2026-11-06)

`test/reference_cuda_bf16_gdn_linear.py` now replays Qwen27 layer-0 QKV/Z,
beta/alpha and output projections directly from the original GGUF and saved
independent vLLM prefill/decode captures. All six outputs match exactly:
129,600 BF16 words. The pinned plugin's unchanged `GGUFHeadTilingLayout`
restores packed weight rows for V, Z, beta and alpha, and maps the output
projection input to GGUF order. Comparing raw GGUF rows to vLLM's logical
head order incorrectly produces widespread mismatches. Same-type merged
weights dispatch once using their combined row count; splitting a merged
projection can alter MMVQ/MMQ selection.

The tool saves transformed packed weights, activation inputs, complete
reference outputs and SHA-256 hashes for native-kernel replay. This establishes
the reference layout and arithmetic boundary; native BF16 Q5_K/Q8_0 linear
kernels remain unqualified. At five tokens the 16,384-row merged Q5_K input
projection uses A8 MMQ, while decode uses MMVQ. The Q8_0 96-row beta/alpha and
5,120-row output projections use MMVQ for both captured batch sizes.

The GGUF layer matrix inventory contains 191 Q5_K and 110 Q8_0 tensors, alongside
Q4_K, Q6_K, IQ3_S and the already covered IQ4/Q3_K formats; F32 convolution
matrices are counted separately. Q5_K and Q8_0 are the immediate typed BF16
linear gaps required to connect the first complete GDN layer. Existing F32
execution is unchanged. Evidence and full inventory:
`/tmp/bitnet-goal-bf16-model-linear/qualification.json`.

### CUDA BF16 Q8_0 A8 linear execution (2026-11-06)

Q8_0 now declares a typed BF16 linear contract in the quant registry. The shared
backend planner selects MMVQ for batches of at most six tokens, or two when
output rows exceed 5,120; larger batches select MMQ. Raw weights remain borrowed
backend handles. No BF16 decoded copy is allocated, and a cached layout cannot
replace the required A8 arithmetic. Existing F32 dispatch is unchanged.

The decode kernel assigns four lanes to each 32-value weight block and eight
blocks per warp. It preserves the reference's FP16 activation scale, F32 scale
product, integer partial dots, lane accumulation and XOR reduction. The prefill
kernel shares activation blocks across 128 rows and four tokens, computes full
32-value integer dots and accumulates them in block order. Its flattened tile
grid handles batches beyond 65,535 without a grid-Y failure. Both produce BF16
outputs and use the existing staged host commit/failure behavior.

| Q8_0 qualification | Result |
| --- | --- |
| Independent matrix cases | 61 |
| Distinct output words | 329,321 |
| Separate/in-place comparisons | 658,642 exact whole-output reference words |
| Output widths | 1, 33, 129, 5,120, 5,121 |
| Input widths | 256, 768, 5,120 |
| Coverage | Both dispatch thresholds, partial row/token tiles, positive/negative/zero weight scales |
| Long batch | 65,537 tokens; independent reference concatenates two unchanged MMQ calls to avoid its quantizer grid-Y limit |
| Original Qwen27 GDN projections | Beta/alpha and output, five prefill tokens and one decode token |
| Real-model replay | 31,296 distinct BF16 words; 62,592 exact separate/in-place comparisons |
| Pooling | Same 61 cases pass with weight pooling enabled |

The real replay consumes the preceding GDN tool's exported transformed packed
weights and inputs, so logical head ordering is held constant. This isolates
native arithmetic from layout conversion. Both real captured batch sizes use
MMVQ; synthetic cases independently qualify the MMQ branch. References use the
unchanged pinned vLLM plugin; Q8_0 A8 involves no cuBLAS call or tolerance.
The checked-in C producer and Python generator preserve complete raw weights,
reference outputs, manifests and hashes. Normal CUDA tests use only synthetic
fixtures and checked-in expected hashes.

A scoped external trace of each real separate/in-place call observes two custom
launches (input quantization and MMVQ), one H2D and one D2H, and no explicit
stream/device synchronization. Beta/alpha traffic is 51,200/960 bytes at five
tokens and 10,240/192 at one; output-projection traffic is 61,440/51,200 and
12,288/10,240. Weight uploads are excluded. These are transfer/launch counts,
not latency or throughput measurements.

Protected clean CUDA build is warning-free. Standard/full CUDA, diagnostic
isolation, backend matrix, AVX2/AVX512 and the retained model/state regression
gates pass. Evidence: `/tmp/bitnet-goal-bf16-q8/qualification.json`.
This is typed component support, not full-model BF16 execution or a throughput
claim. Q5_K remains the next projection format required by the first GDN layer;
remaining mixed formats, GDN/attention/KV precision, serving, measured vLLM
parity and the cross-schedule Flash long mismatch remain open.

### CUDA BF16 Q5_K affine A8 linear execution (2026-11-06)

Q5_K now declares typed BF16 A8 MMVQ/MMQ in the quant registry and shares the
six-token threshold (two above 5,120 output rows). Native kernels borrow raw
weights, keep decoded-weight bytes at zero and return BF16 with F32
accumulation. MMQ shares activation blocks across 128 rows and four tokens;
its flattened grid supports long batches. Existing F32 execution is unchanged.

The reference uses different affine corrections in the two branches. MMVQ
sums quantized lanes, weighted by their FP16 scale. MMQ uses the FP16-rounded
sum of original BF16 input values. A dedicated affine input quantizer supplies
that sum only for Q5 MMQ plans. Shared gate/up preparation enables it if either
projection needs it; symmetric formats ignore the sum field. Down preparation
selects its own contract. No format substitution or CPU fallback is introduced.

An initial matrix test exposed a BF16 rounding boundary at row width 33 and
five tokens. Isolated unchanged-reference comparisons showed that fusing the
scale/data product matches, while fusing the minimum product changes one of
165 output words. Both new kernels now use an explicit FMA after separately
rounding the minimum product. References were preserved and no tolerance added.

| Q5_K qualification | Result |
| --- | --- |
| Independent matrix cases | 61; 329,321 distinct output words |
| Separate/in-place matrix checks | 658,642 exact words |
| Matrix coverage | Signed/zero data and minimum scales, dispatch thresholds, partial tiles, widths 256/768/5,120, 65,537 tokens |
| Independent mixed FFNs | 90; Q5/Q8/Q5, Q8/Q5/Q8, Q5/IQ4_XS/Q3_K |
| FFN dimensions/hidden | 256/256, 768/512, 256/5,376; two seeds; batches 1/5/9/17/33 |
| Separate/in-place FFN checks | 998,400 exact words; 499,200 distinct outputs |
| Independent cuBLAS 13.7/13.10 FFN runs | Identical reference outputs and headers |
| Original Qwen QKV/gate projection | 98,304 distinct prefill/decode words; 196,608 exact comparisons |
| Entire first GDN layer projection set | Q5 input plus rerun Q8 beta/alpha/output: 129,600 distinct words; 259,200 comparisons |
| Pooling | All 61 matrix and 90 FFN cases pass with pooling enabled |
| Reference regeneration | Both checked-in Q5 reference headers reproduced exactly |

Real replays use the preceding tool's original transformed GGUF weights,
inputs and independent vLLM captures. Five tokens exercise merged Q5 MMQ;
one token exercises MMVQ. Separate and in-place outputs match exactly. The
normal synthetic test needs no model files and preserves all earlier matrix,
FFN and residual references. The 65,537-token reference uses two unchanged MMQ
calls to avoid its quantizer grid-Y limit; native execution uses one batch.

Scoped traces exclude weight uploads. Each real call issues two custom
launches, one H2D and one D2H, with no explicit stream/device synchronization.
Five-token input/output traffic is 51,200/163,840 bytes; one-token traffic is
10,240/32,768. Readback remains a host synchronization boundary. These counts
are not latency or throughput measurements.

Protected clean CUDA build is warning-free. Standard/full CUDA, diagnostic
isolation, backend matrix, AVX2/AVX512, model and saved-state gates pass. Evidence:
`/tmp/bitnet-goal-bf16-q5/qualification.json`. Frozen executable SHA-256:
`22e5c48e808f6d7fd3823ac06710855d09f8dbd692b197dbdb0ea44d2776c5a4`.

This qualifies the first GDN layer's linear projection formats, not its full
recurrence or model execution. Resident BF16 GDN, gated norm, attention and
KV/state precision, remaining mixed formats, serving, matched vLLM performance
and the separate cross-schedule Flash long mismatch remain open.

### CUDA BF16 tiled causal convolution with history (2026-11-06)

Added optional `causal_conv_silu_bf16_batch` for widths 2/3/4, without bias.
Weights are resident BF16 `[dim,width]`; input/output are `[tokens,dim]` and
caller-owned history is BF16 `[width-1,dim]`, oldest first. Output can alias
input; history must be disjoint from input/output. Combined staged readback
commits output and history only after success. Other backends retain an absent
optional capability. Existing F32 convolution and Q/K normalization are unchanged.

The unchanged reference Triton convolution rounds each BF16 product before
F32 accumulation. A 54,074-word probe matches every word using this sequence;
F32 fused products match only 37,497. SiLU consumes the F32 sum, then rounds to
BF16. The native kernel preserves those conversion boundaries explicitly.
Carried history stores raw BF16 input, preserving its bits independently of
activation. No scalar fallback, weight conversion or automatic placement occurs.

Prefill parallelizes across eight-token/channel tiles rather than a serial
loop over the entire prompt. Each tile reads its immutable raw input/history
window, and only the final token tile writes carried history. A flattened grid
avoids a grid-Y token limit. Short calls use 64 channel threads and longer ones
128. No inter-tile synchronization is required.

| Convolution qualification | Exact result |
| --- | --- |
| Independent fixtures | 146; cold/warm history, widths 2/3/4 |
| Channels | 1, 31, 32, 33, 255, 256, 257, 10,240 |
| Tokens | 1/5/17 plus 65,537-token cases |
| Distinct output/history words | 1,663,564 / 399,786 |
| Separate/in-place/chunked/decode comparisons | 7,991,240 exact words |
| Independent prefill/update equivalence | All 144 short fixtures agree in output and history |
| Real-weight replay | Qwen27 layer-0 weights and captured QKV inputs, five prefill tokens followed by one decode token |
| Real-weight output/history checks | 245,760 exact comparisons across both output modes |
| Readback failures | Eight partial copies into the history vector; output/input/history unchanged, retries exact |
| Pooling | All 146 cases pass with weight pooling enabled |

The real replay compares against independently executed convolution primitives,
not an observed model convolution output or the full GDN layer. Value-head rows
are restored using the pinned GGUF adapter's layout; F32 convolution weights
round to BF16 as in the reference model. The first call starts with zero history,
then decode consumes the carried prefill history. Synthetic references use
unchanged native vLLM prefill/update operations in the pinned image; no BitNet
output generates a golden value. Cache slot one avoids the reference's reserved
null slot. Reference generator: `test/reference_cuda_bf16_conv.py`.

Each real callback issues one custom kernel, two H2D copies and one combined
D2H copy, with no explicit stream/device synchronization. Five-token traffic
is 163,840 bytes each direction; one-token traffic is 81,920 bytes each direction,
including history. Weight uploads are excluded. Final readback remains a host
synchronization boundary; this callback does not keep history resident across
host entries or establish a full-model performance result.

A standalone kernel comparison uses real Qwen weights, synthetic BF16 inputs,
zero initial history and warmed data. Five CUDA-event samples time 100-node
CUDA graph replays after warmup; all other task tests finished before measurement
on GPU 0. The largest shape runs first to warm the device. Both extracted kernels
produce identical complete outputs/history for every timed shape.

| Tokens, 10,240 channels | Serial BF16 prototype median | Tiled median | Ratio |
| --- | --- | --- | --- |
| 1 | 2.27 us | 1.83 us | 1.24x |
| 5 | 3.92 us | 3.23 us | 1.21x |
| 512 | 189.64 us | 19.88 us | 9.54x |
| 1,047 | 382.50 us | 38.93 us | 9.82x |

This measures extracted device kernels against the serial BF16 prototype,
including graph-node scheduling. It excludes copies, allocations and the host
callback; warm-cache inputs/history remain immutable across replays. It is not
a vLLM comparison or full-model prefill/decode result. Raw samples, ranges,
source, device/process records and equality assertions are retained with the
qualification evidence.

Protected clean CUDA build is warning-free. Standard/full CUDA, diagnostics,
backend matrix, AVX2/AVX512, model and fresh saved-state regression gates pass.
Evidence: `/tmp/bitnet-goal-bf16-conv/qualification.json`. Frozen executable:
`ec2c0f51013bba9296666f1f30c231b08fa007ec9392c96ec019b05f41a70141`.

Q/K normalization, gating and the F32 recurrent-state contract remain before a
complete BF16 GDN layer can run. Other mixed formats, attention/KV precision,
serving, matched vLLM parity and the cross-schedule Flash long mismatch remain open.

### CUDA BF16 packed GDN decode/state contract (2026-11-06)

The optional `gdn_decode_bf16` host bridge implements the pinned vLLM packed
decode operator for K=V=128, logical repeated K heads and F32 `[V,K]` recurrent
state. Input QKV and beta/alpha rows are BF16. Natural-log A and bias are
resident F32 and BF16 respectively. The profile fixes additive L2 epsilon at
1e-6 and softplus threshold at 20; its caller supplies positive query scale.
It keeps normalized Q/K in FP32 through recurrence, decays state before the
K dot product and uses explicit FMA local dot trees for both update and output.
The private prototype also passes with automatic contraction disabled.

| Decode qualification | Exact result |
| --- | --- |
| Independent unmodified reference | 9 cases, 8 dependent steps each; K/V head counts 1/1, 2/6, 16/48 |
| Output/state comparisons | 65,387,520 across separate, input-alias and fresh host-state allocation modes |
| Reference regeneration | All exported bytes and embedded hashes identical |
| Captured and synthetic trajectories | 48 calls; 294,912 BF16 output and 37,748,736 F32 state words |
| Captured first decode | All 6,144 core-output words match the original model capture |
| Partial readback recovery | 48 injected failures; output/state/input unchanged; retries exact |
| Pooling | All independent cases also pass with bounded weight pooling enabled |

For the target 16/48-head shape, each host call launches one kernel, uploads
3,166,400 bytes in three copies and reads back 3,158,016 bytes in one staged
copy. No explicit synchronization calls were observed; the synchronous copies
still impose transfer waits. Resident parameter uploads are excluded from
these counts. Output and state commit together only after successful readback;
output may alias QKV input, while state and BA must remain disjoint.

These counts qualify the host bridge and its arithmetic. Device-resident state
composition and native chunk/WY prefill remain before full BF16 model execution;
no throughput, decode-latency or vLLM performance-parity result is claimed.
Evidence: `/tmp/bitnet-goal-bf16-gdn-decode/qualification.json`.

### CUDA BF16 GDN prefill Q/K and gate preparation (2026-11-06)

Added optional `gdn_prefill_prepare_bf16` with packed BF16 QKV and
`[beta,alpha]` input rows. It normalizes Q/K, copies V without changing its bits,
and produces F32 log-decay and beta. Resident parameters are natural-log A in
F32 and bias in BF16. Positive finite epsilon and nonnegative finite softplus
threshold are caller-supplied, preserving architecture ownership of policy.
No query scaling, recurrence, scalar fallback or lazy parameter conversion is
performed. Output QKV may alias input; the scalar vectors and packed BA storage
must be disjoint. All three output vectors remain unchanged on failure.

The kernel preserves the pinned prefill operator's precision, including additive
L2 epsilon, approximate square root followed by full division, and BF16 rounding
of normalized Q/K. Gate preparation uses `log(1+exp(...))`, not `log1p`, with
explicit F32 addition, exponential and threshold order. Decay and beta remain
F32. Half-warp reduction trees retain the reference's local grouping. Eight-token
head tiles use a flattened grid; packed BA addressing uses size_t arithmetic.
Internal padding aligns F32 readback vectors after odd-length BF16 output.

| Preparation qualification | Exact result |
| --- | --- |
| Independent cases | 178 |
| Default wrapper profile | 154 cases; all original reference hashes preserved |
| Additional policy cases | 24 unchanged-kernel runs; epsilon 1e-5/1e-3 and thresholds 0/8/20 |
| Q/K widths | 1 through 256, including partial/power-of-two boundaries |
| Target geometry | 16 K heads, 48 V heads, K/V width 128 |
| Tokens | 1/5/17/33 and 65,537 |
| Distinct BF16 QKV words | 16,629,634 |
| Distinct F32 words per gate vector | 150,702 |
| Separate/in-place/chunked comparisons | 38,078,936 raw BF16/F32 values exact |
| Pooling | All 178 cases pass with weight pooling enabled |
| Real-parameter replay | 124,032 exact QKV/log-decay/beta comparisons across both output modes |
| Failure recovery | 24 partial readbacks into both F32 vectors; all outputs and inputs unchanged; retry exact |

The unchanged reference wrapper fixes epsilon at 1e-6 and threshold at 20;
additional policy cases call its unchanged kernel with different constexpr
arguments. The reference currently fails to compile when the padded K and V
widths differ because a branch-local mask name has incompatible types. This
backend profile explicitly requires matching power-of-two buckets with widths
at most 256. No reference math was patched to bypass that limitation.

Real replay uses independent convolution results from the preceding checkpoint,
captured Qwen BA projections and actual GGUF parameters. As in the pinned
adapter, raw negative decay coefficients become natural logs on the CPU before
GPU loading; value-head rows are restored to logical order and bias rounds to
BF16. This checks the prefill primitive on five-token and single-token inputs.
It does not observe a full model preparation boundary directly, validate the
fused model decode recurrence or close the complete GDN layer.

Scoped transfer traces observe one custom kernel, two H2D copies and one combined
D2H copy, without explicit stream/device synchronization. Five-token input/output
traffic is 103,360/104,320 bytes; one-token traffic is 20,672/20,864 bytes. Weight
uploads are excluded. Host readback remains a synchronization boundary; no
resident-history, throughput or vLLM parity claim follows from these counts.

Protected clean CUDA build is warning-free. Standard/full CUDA, diagnostics,
backend matrix, AVX2/AVX512, model and fresh saved-state regressions pass.
Evidence: `/tmp/bitnet-goal-bf16-gdn-prep/qualification.json`. Generator:
`test/reference_cuda_bf16_gdn_prep.py`. Frozen executable:
`05c52a6d2328b3bde42f9d7df8e4e41c4ab66377198c3b67274e9a22aef6b4cc`.

Remaining work includes the F32 recurrent-state contract, gated normalization,
resident layer/model composition, other mixed formats, attention/KV precision,
serving, matched vLLM performance and the cross-schedule Flash long mismatch.

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

## CPU Q5_K prefill reduction investigation (2026-11-04)

On the Threadripper 9980X, the pinned llama.cpp build
`df18bee11e96557b07aa7b12db0976f3c7ac9165` selects its tiled CPU matrix
multiplication path for the 33-token Qwen3.8 27B prompt. The first layer's
5,120 normalized input floats match bit for bit between engines. Q8_K
activation scales, codes, and block sums also match independently.

Replaying the captured input through the current Q5_K batched kernel and
llama.cpp's exported legacy vector dot gives identical results to each
other, but only 2,215 of 10,240 outputs match the captured tiled projection
bit for bit (maximum difference 7.62939453e-6). The existing 16-lane VNNI
vector reduction does not resolve the difference.

A standalone arithmetic replay instead sums scaled integer dot products and
minimum corrections within each 256-element block, computes the corrected
block value with fused negative multiply-add, and accumulates each block
with `fmaf(block_value, activation_scale, output)`. This matches **all
10,240 captured outputs bit for bit**, maximum and RMS difference zero.
Using a separately rounded multiply and add for the final accumulation
matches only 5,326 outputs. This identifies the required tiled arithmetic;
it does not yet qualify a production kernel or fix model-level CPU parity.

Captures and replay are local diagnostic artifacts:
`/tmp/bitnet-goal-cpu33-norm.bin`,
`/tmp/bitnet-goal-ref33-qkv.bin`, and
`/tmp/bitnet-goal-q5-tiled-replay.c`. The reference probe was explicitly
linked to the pinned build's versioned libraries and run with that build's
`LD_LIBRARY_PATH`; the installed `/usr/local/lib` probe is not the oracle.
The subsequent format-local kernel entry point
`bn_quant_q5k_avx512_tiled_matmul_range` shares four-token weight decoding
with the existing kernel, performs integer superblock reduction, and keeps
the existing GEMV-order entry point intact. It reproduces all 10,240 captured
outputs exactly. It is not selected by production dispatch yet; prefill
policy integration and model-level qualification remain required.

`test/reference_cpu_q5k_tiled.c` independently generates synthetic graph
hashes using only pinned GGML CPU libraries. `test_quant` checks those hashes
for 8, 9, 17, and 33 tokens, 137 output rows and three superblocks, both
serially and through threaded row partitioning, and checks untouched output
tails. All quant integration tests pass, including existing GEMV-order tests.
The independent reference hashes were generated before executing the new
kernel tests; existing golden values are unchanged.
Validation after the kernel change: clean CUDA-enabled `bitnet` build with
zero warnings, full `make -j8 BN_ENABLE_CUDA=1 test` passed, and an AVX2-only
compile of the changed source passed with `-Wall -Wextra -Wshadow`.
The optional real-model matrix is not implied by the standard suite; model
prefill/decode qualification remains open. Logs are
`/tmp/bitnet-goal-q5-clean-build.log` and
`/tmp/bitnet-goal-q5-tiled-full-suite.log`.

The reference CPU library SHA-256 is
`e28660ece5030ec85229b5e03e57d8f0da55866b63943623a651e1adbd02716f`.
Reproduce the synthetic oracle with the pinned build (its CPU library must
have AVX512 VNNI enabled):

```sh
ref=/home/mark/artalis.io/tools/llama.cpp
cc -O3 -march=native -I"$ref/ggml/include" test/reference_cpu_q5k_tiled.c \
  "$ref/build/bin/libggml.so.0" "$ref/build/bin/libggml-base.so.0" \
  "$ref/build/bin/libggml-cpu.so.0" -lm -o /tmp/reference_cpu_q5k_tiled
LD_LIBRARY_PATH="$ref/build/bin" /tmp/reference_cpu_q5k_tiled
```


## CPU Q5_K integration and GDN normalization probe (2026-11-04)

An isolated executable `/tmp/bitnet-goal-tiled-q5-bitnet` selects the new
Q5_K tiled arithmetic at eight or more tokens and enables native Q5_K
prefill projections. The production dispatch and registry are unchanged.
For the same 33-token prompt, its complete layer-0 QKV output (10,240
floats) matches the independent capture byte for byte. Its post-convolution
SiLU output also matches all 10,240 floats byte for byte. Independent token
parity nevertheless remains **5/8**, identical to the baseline; projection
agreement alone is not model qualification.

With identical convolution inputs, the next discrepancy is Q/K normalization.
The pinned reference's `build_gdn_l2_norm` in `src/models/models.h` computes
`scale(rms_norm(x, eps/head_dim), 1/sqrt(head_dim))`. The current CPU SSM
kernel computes `x / max(sqrt(sum(x*x)), eps)`. Besides the different epsilon
placement, the reference performs two separately rounded output scales.

A second isolated executable `/tmp/bitnet-goal-gdn-q5-bitnet` combines the
Q5_K candidate with the reference RMS-and-scale sequence. The first 16
layer-0 Q-normalized values now match exactly. The first 16 values also
match at SSM delta, SSM gate, SSM output projection, and FFN input norm;
the next observed difference is the FFN gate projection. These are trace
samples, not claims of complete tensor equality for those later stages.
Token parity still remains **5/8**. Neither candidate is promoted.

Evidence: `/tmp/bitnet-goal-tiled-q5-prefix33-reference.log`,
`/tmp/bitnet-goal-tiled-q5-prefix33.trace`,
`/tmp/bitnet-goal-ref33-conv.bin`, `/tmp/bitnet-goal-tiled33-conv.bin`,
`/tmp/bitnet-goal-gdn-q5-prefix33-reference.log`, and
`/tmp/bitnet-goal-gdn-q5-prefix33.trace`. Reference captures use the same
explicitly pinned libraries described above. Next work must qualify the
remaining FFN matrix arithmetic and express GDN normalization through
architecture-owned semantics across CPU/GPU paths, preserving existing
independent test contracts rather than globally replacing generic L2 norm.

## CPU FFN tiled arithmetic qualification (2026-11-04)

Complete layer-0 FFN input captures from the isolated Q5_K/GDN candidate
and pinned reference match byte for byte (5,120 floats). The gate and down
weights are IQ4_XS; the up weights are Q3_K. Replaying that identical input
with per-superblock integer reduction, FP32 weight scaling, and fused
activation-scale accumulation matches every reference output for both the
IQ4_XS gate and Q3_K up projections (17,408 floats each, maximum/RMS zero).
Current legacy arithmetic matches only 3,651 IQ4_XS outputs and 3,770 Q3_K
outputs bit for bit, respectively. Activation Q8_K codes/scales/sums remain
identical to the independently exported reference quantizer.

An isolated IQ4_XS panel epilogue changes
`fma(integer_total, weight_scale * activation_scale, output)` to
`fma(integer_total * weight_scale, activation_scale, output)`. Combined with
the prior Q5_K and GDN candidates, the complete captured gate tensor now
matches all 17,408 values byte for byte. The formerly failing 33-token
prompt improves from **5/8 to 8/8** independent generated token IDs.

Broader checks prevent promotion: the eight-prompt CPU comparison gives
**177/192** prefix token matches, all eight first tokens match, with no
count mismatches. The story prompt matches 21/24 and the HTTP 404 prompt
12/24; all others match 24/24. This is a CPU result, separate from the prior
192/192 CUDA gate. Internal full-versus-chunked prefill (33 tokens, chunk 9,
64-token resume, eight predictions per phase) still gives **15/16**. Its
full-prefill step-5 prediction is now 11258, while chunked predicts 89583.
The six-token initial tail uses a different arithmetic route from the full
batch. No test threshold or independent expected values were relaxed.

Adding isolated Q3_K tiled prefill still leaves the story prompt at 21/24.
Thus exact layer-0 projection contracts do not close whole-model parity.
Production dispatch, registry, normalization, and IQ4_XS/Q3_K kernels remain
unchanged; these executables are diagnostic candidates.

Evidence: `/tmp/bitnet-goal-iq4xs-tiled-replay.c`,
`/tmp/bitnet-goal-q3k-tiled-replay.c`,
`/tmp/bitnet-goal-gdn-iq4-q5-ffn-gate.bin`,
`/tmp/bitnet-goal-gdn-iq4-q5-prefix33-reference.log`,
`/tmp/bitnet-goal-gdn-iq4-q5-cpu-short-reference.log`,
`/tmp/bitnet-goal-gdn-iq4-q5-continuation33-9.log`, and
`/tmp/bitnet-goal-all-tiled-story-reference.log`. All processes completed.
Candidate SHA-256 values:

- Q5_K/GDN/IQ4_XS: `9e228e96e5d936f6384c46e2a8f3dd11b3a5c79d3f88be1e3bad9dfde90df8dd`.
- With Q3_K: `bed6268836b18906cd15eed340ef71c6a79a0d93c76bfa95508569371805a632`.

Next qualification needs complete tensor/logit comparisons under identical
teacher-forced histories at the story/HTTP divergence, plus explicit,
consistent prefill arithmetic for short chunk tails before integration.

## CPU chunk-tail divergence localization (2026-11-04)

An isolated variant lowers the Q5_K tiled and IQ-panel batch floor from eight
to two tokens, keeping the preceding GDN normalization candidate. It still
fails the 33-token/chunk-9 continuation gate at **15/16**. Thus the remaining
failure cannot be attributed solely to the six-token tail selecting legacy
Q5_K/IQ4_XS arithmetic. The resumed phase still matches all eight predictions.

A full prompt-history trace compares every recorded stage at positions 0–32
between whole and chunked execution. Layer 0 has **zero differing recorded
summaries**, including their first-16 samples, at all 33 positions. These
summaries are diagnostic evidence, not proof of full tensor equality.
The first differing stage is layer 1 QKV at position 8; its recorded input
norm agrees. Layer 1 QKV differs exactly at positions **8, 17, 26, and 31**.
Its QKV and Z weight formats are Q4_K (layer 0 uses Q5_K).

Those positions coincide with different four-token panel/canonical-tail
assignments: each nine-token chunk has a one-token tail at 8, 17, or 26;
the final six-token chunk has a two-token tail at 31–32, whereas the whole
33-token batch has only position 32 in its tail. Position 32 QKV itself
agrees, while its convolution differs after consuming the earlier divergent
history. This localizes the next investigation to Q4_K packed-input versus
canonical-tail arithmetic; complete same-input captures are still needed
before claiming a kernel fix.

Evidence: `/tmp/bitnet-goal-short-tiled-continuation33-9.log`,
`/tmp/bitnet-goal-short-tiled-continuation-pos32.trace`, and
`/tmp/bitnet-goal-short-tiled-continuation-all.trace`. Both trace processes
completed with the same 15/16 failure. Production dispatch remains unchanged.

## Q4_K same-input oracle and tail candidates (2026-11-04)

Pinned-reference layer-1 normalization and QKV captures isolate Q4_K from
the rest of the model. Using that exact normalized input (5,120 floats),
the existing prepared Q4_K four-token matrix path matches **all 10,240
reference projection outputs bit for bit**. Q8_K scales, codes, and sums
also match independently. In contrast, the unprepared legacy vector/matrix
path matches 1,543 outputs, and the proposed fused superblock formula
matches only 799 (maximum difference 1.90734863e-6). The prepared result
establishes that this reference projection uses the repacked Q4_K arithmetic
contract; the Q5_K tiled contract cannot simply be generalized to Q4_K.
The replay uses an independently created prepared weight, with a 256 MiB
arena to hold its packed layout.

Three isolated diagnostics distinguish token agreement from qualification:

- Padding Q4_K float-input batches to a complete four-token panel gives
  16/16 continuation token matches, but still leaves substantial logit
  differences. It is not a sufficient fix.
- Repacking canonical tail inputs into a temporary four-token panel and
  executing the existing packed kernel gives 16/16 continuation matches
  and **zero initial-phase logit differences at all eight steps**. All
  1,456 recorded position-32 stage summaries agree. Its resumed logits
  still differ, and its independent 33-token prompt regresses to 5/8.
  The prototype is allocation-heavy and is rejected for promotion.
- A Q4_K superblock-formula candidate gives 16/16 continuation matches
  and 8/8 on that independent prompt, but its projection fails the direct
  same-input oracle above. Story/HTTP independent prefixes remain 33/48.
  This candidate is also unqualified, despite those narrow passes.

For the superblock candidate, a second continuation schedule (33-token
prompt, chunk 11, 64-token resumed suffix, eight predictions per phase)
passes **16/16 with zero maximum/RMS logit difference at every step in
both phases**. Chunk 9 passes 16/16 with zero initial-phase differences,
but resumed logits differ after its one-token suffix tail. The runtime's
`prefill_internal` explicitly dispatches a one-token prefill through
`bn_transformer_forward`, which uses decode matrix arithmetic. Consistent
multi-token arithmetic therefore does not by itself qualify arbitrary
chunk boundaries.

Evidence: `/tmp/bitnet-goal-q4-layer1-replay.c`,
`/tmp/bitnet-goal-ref33-layer1-{norm,qkv}.bin`,
`/tmp/bitnet-goal-pad-q4-continuation33-9.log`,
`/tmp/bitnet-goal-q4-packed-tail-continuation33-9.log`,
`/tmp/bitnet-goal-q4-packed-tail-prefix33-reference.log`,
`/tmp/bitnet-goal-q4-tiled-prefix33-reference.log`,
`/tmp/bitnet-goal-q4-tiled-continuation33-{9,11}.log`, and
`/tmp/bitnet-goal-q4-tiled-story-http-reference.log`. All processes completed;
production dispatch and golden values remain unchanged.

Next work must distinguish reference packed-GEMM versus GEMV contracts
explicitly and preserve the selected prefill contract across partial panels
and one-token continuation requests. A passing greedy prefix alone must not
promote a candidate whose same-input oracle contradicts that contract.

## Durable Q4_K repack reference contracts (2026-11-04)

`test/reference_cpu_q4k_repack.cpp` now reproduces the independent packed
CPU oracle without model files or bitnet objects. It explicitly allocates
Q4_K weights in GGML's `CPU_REPACK` buffer, uses three superblocks, and
executes batches of 1–9, 17, and 33 tokens. Run it for 136 rows (eight-row
runtime groups) and 128 rows (also eligible for sixteen-row runtime groups).
The pinned reference chooses `q4_K_8x8` for both fixtures.

The reference itself changes output rounding across batch schedules.
For 136 rows, comparing identical input prefixes to its 33-token execution:
one token changes 101 floats (maximum 7.62939453e-5), five tokens change 93
(maximum 3.43322754e-5), and nine tokens change 103 (maximum 4.57763672e-5).
Four and eight tokens match their full-batch prefixes exactly. These are
independent reference observations, not relaxed runtime tolerances.

`test_q4k_prepared_reference_panels` in `test_quant` preserves all **22
independently generated hashes** across both row layouts, serial and
threaded dispatch, and checks output sentinels. It also verifies directly
that the same first token differs between a one-token call and a full panel.
The current prepared runtime path passes these exact contracts, including
tails. Existing reference values are unchanged.

The quant registry consequently no longer declares Q4_K matrix arithmetic
universally identical to GEMV on AVX512. This is capability metadata
correction; it does not select a new kernel. CPU AVX512's normal prefill
projection replay policy is `NONE`, so this change does not replace its
current matrix arithmetic with per-token replay.

Reproduce against pinned llama.cpp commit
`df18bee11e96557b07aa7b12db0976f3c7ac9165` and the CPU library SHA above:

```sh
ref=/home/mark/artalis.io/tools/llama.cpp
c++ -O3 -march=native -std=c++17 -Wall -Wextra -Wshadow \
  -isystem "$ref/ggml/include" test/reference_cpu_q4k_repack.cpp \
  "$ref/build/bin/libggml.so.0" "$ref/build/bin/libggml-base.so.0" \
  "$ref/build/bin/libggml-cpu.so.0" -o /tmp/reference_cpu_q4k_repack
LD_LIBRARY_PATH="$ref/build/bin" /tmp/reference_cpu_q4k_repack 136
LD_LIBRARY_PATH="$ref/build/bin" /tmp/reference_cpu_q4k_repack 128
```

Validation completed after the metadata/test changes: clean CUDA-enabled
`bitnet` build with zero warnings, followed by full
`make -j8 BN_ENABLE_CUDA=1 test` passing. Optional external-model cases remain
skipped unless configured; this does not claim that the open Qwen model
qualification gates are resolved. Logs:
`/tmp/bitnet-goal-q4-oracle-clean-build.log` and
`/tmp/bitnet-goal-q4-oracle-full-suite.log`.

The oracle intentionally calls a pinned internal C++ repack entry point;
it is not a stable external GGML API. Local oracle outputs are
`/tmp/bitnet-goal-reference-cpu-q4k-repack.log` and
`/tmp/bitnet-goal-reference-cpu-q4k-repack128.log`.

## Independent chunked-prefill model probe (2026-11-04)

`test/llama_layer_probe.cpp` now accepts `--prefill-chunk N`. It submits
explicit chunks into one reference context, sets batch/microbatch capacity
to N, and keeps tensor observer positions relative to the active chunk.
`--sequential` remains separate and mutually exclusive. Nonpositive,
malformed, or overflowing chunk sizes fail before model load. Chunked
layer-input embedding dumps fail explicitly because the probe does not yet
aggregate them across calls; selected intermediate tensor and KV captures
remain available.

The pinned independent Qwen reference itself changes greedy predictions
under these schedules. For the same 33-token prompt and F32 KV, flash off,
context 113, eight predictions:

- Whole prefill: `5604,13,271,550,4962,11258,6195,271`.
- Chunk 9: `5604,13,271,550,4962,89583,21753,271`.
- Chunk 11: `5604,13,271,550,4962,89583,21753,271`.

These sequences are independently generated, not revised expected values
from bitnet output. The existing internal continuation check remains
unchanged and useful as a consistency diagnostic. Independent chunked
qualification must additionally compare matching execution schedules and
token histories, rather than interpret disagreement between two reference
schedules as proof of runtime failure.

A token-only diagnostic harness then reads each schedule's independently
generated IDs from a file and replays matching histories through the isolated
Q5_K/GDN/IQ4_XS runtime candidate (using the original Q4_K prepared kernels,
not the rejected Q4_K formula candidates). Both **chunk 9 and chunk 11 pass
8/8 initial-phase predictions**, including the divergent step 5 and its
following token. This is a matched-schedule independent result; it neither
changes the internal 15/16 invariance result nor qualifies the resumed phase
against an independent oracle. The broader CPU 177/192 failures remain open.
No new runtime default is enabled.

Diagnostic harness and results:
`/tmp/bitnet-goal-matched-chunk-tokens.c`,
`/tmp/bitnet-goal-reference-{chunk9,chunk11}-ids.txt`, and
`/tmp/bitnet-goal-candidate-matched-{chunk9,chunk11}.log`.

Probe self-tests pass. A real chunk-9 tensor capture at position 8 reports
`ne=5120,9,1,1` and writes exactly 5,120 normalization floats, validating row
selection at the first chunk boundary. Reference runs and captures:
`/tmp/bitnet-goal-reference-prefill-{whole,chunk9,chunk11}.log`,
`/tmp/bitnet-goal-reference-chunk9-pos8.log`, and
`/tmp/bitnet-goal-reference-chunk9-pos8-norm.bin`.
The source is compiled with the same explicitly pinned versioned libraries
and runtime library path as preceding authoritative probes.

Validation after the probe change: its self-tests pass, invalid/conflicting
chunk arguments are rejected before model loading, the clean CUDA-enabled
`bitnet` build has zero warnings, and full
`make -j8 BN_ENABLE_CUDA=1 test` passes. Logs:
`/tmp/bitnet-goal-chunk-probe-clean-build.log` and
`/tmp/bitnet-goal-chunk-probe-full-suite.log`. Optional external-model matrix
entries remain skipped unless configured; these checks do not qualify all
modern model gates.


## HTTP divergence and Q6_K prefill contract (2026-11-04)

For the isolated Q5_K/GDN/IQ4_XS candidate, the HTTP prompt matches the first
12 reference tokens, then chooses `like` (4650) instead of `web` (2817).
At prediction 12 the reference logits are 24.5190487 versus 24.4701996;
the candidate gives 24.5353565 versus 24.5517616. Both histories are identical
through this prediction. Matching tokenwise prefill instead passes **24/24**,
so the remaining discrepancy is specific to batched prefill history.
Adding the earlier isolated Q3_K correction alone still gives 12/24.

Complete tensor captures at position 19 establish that all 5,120 layer-0
normalized-input floats match byte for byte. Recorded first-16 samples
through layer-0 SSM/FFN also agree; those samples are not full-tensor proof.
Layer-1 delta output differs in 1,656/6,144 floats, maximum 7.45058060e-9;
its gate maximum difference is 2.23517418e-8. By layer 2 the delta maximum
is 2.11747829e-6 and gate maximum 0.000129260123, concentrated in later value
heads. The layer-2 Q6_K output projection matches **all 5,120 outputs exactly
on identical reference input** in decode. Replaying the actual differing
gate input produces maximum output difference 0.000312864780. This rules
out changing the decode projection to compensate for upstream state/input
differences.

The independent eight-token prefill capture exposes a distinct Q6_K matrix
contract: applying integer reduction within each superblock, FP32 weight
scaling, then fused activation-scale accumulation matches all 5,120 layer-2
outputs exactly. Existing GEMV-order arithmetic matches only 431 outputs
bit for bit (maximum 1.19209290e-7). A preliminary layer-0 replay was
inapplicable because that projection is Q8_0, not Q6_K; the authoritative
Q6_K replay explicitly selects layer 2 and asserts its format.

`bn_quant_q6k_avx2_tiled_matmul_range` now provides this arithmetic as a
separate internal entry point. It shares eight-token weight decoding with
the existing implementation. The existing GEMV-order entry point and all
production dispatch remain intact. `test/reference_cpu_q6k_tiled.c` generates
four independent GGML graph hashes (8, 9, 17, 33 tokens, 137 output rows,
three superblocks). `test_q6k_tiled_reference` verifies them with serial and
threaded execution and output sentinels. Both native AVX512-enabled and
explicit AVX2-only quant integration suites pass these exact hashes.

The checked-in entry point also reproduces all 5,120 real-model prefill
projection outputs exactly in a direct same-input replay
(`/tmp/bitnet-goal-http-prefill-q6-production-replay.c`). Validation completed:
clean CUDA-enabled build with zero warnings, explicit AVX2-only full quant
integration tests, and full `make -j8 BN_ENABLE_CUDA=1 test` all pass.
Logs: `/tmp/bitnet-goal-q6-clean-build.log`,
`/tmp/bitnet-goal-q6-avx2-quant.log`, and
`/tmp/bitnet-goal-q6-full-suite.log`. Optional external-model matrix entries
remain skipped unless configured; no broad model qualification is implied.

This is **not** whole-model promotion: the isolated combined Q6_K/Q3_K/
Q5_K/GDN/IQ-panel candidate regresses HTTP from 12/24 to **9/24** reference
prefix matches. Correct same-input projection arithmetic is necessary but
does not close the remaining model contracts. No reference golden was
changed and no candidate kernel is selected by default.

Evidence: `/tmp/bitnet-goal-http-{ref,candidate}24.log`,
`/tmp/bitnet-goal-http-tokenwise-reference.log`,
`/tmp/bitnet-goal-http-{ref,candidate}-pos19.trace`,
`/tmp/bitnet-goal-http-output-replay.c`,
`/tmp/bitnet-goal-http-prefill-q6-replay.c`,
`/tmp/bitnet-goal-reference-q6-tiled.log`,
`/tmp/bitnet-goal-q6-production-quant.log`,
`/tmp/bitnet-goal-q6-avx2-quant.log`, and
`/tmp/bitnet-goal-q6-tiled-http-reference.log`.

## Mixed-format CPU prefill: independent gates pass in isolation (2026-11-04)

Combining the qualified Q5_K, Q6_K, Q3_K, IQ-panel arithmetic and GDN
normalization sequences moves the HTTP final-prefill trace into agreement
through layer 44. The first recorded difference is layer 45 QKV. Its
complete normalized input matches the reference byte for byte (5,120
floats), but the fallback output matches only 697/10,240 floats; maximum
difference is 2.86102295e-6. Replaying that same input with the prepared Q4_K
matrix path matches **all 10,240 reference outputs exactly**.

This layer pairs Q4_K QKV with IQ4_XS Z. The shared prepared K-quant group
is ineligible for that combination. Its fallback calls independent host
projections, where Q4_K takes per-token GEMV and the native-prefill branch
discards prepared resources by passing NULL. This changes the arithmetic
contract because grouping is unavailable, despite a prepared Q4_K matrix
kernel already existing.

The isolated correction declares native prefill for Q4_K and resolves each
independent host projection's prepared resource through the existing backend
resource policy. It passes that resource to matrix execution. There are no
layer-number or model-file conditions. All **672 comparable recorded entries**
at the final HTTP prompt token then agree across all 64 layers, including
summary statistics and first-16 samples. This does not claim full-tensor
identity at every stage.

The combined candidate now passes independent CPU sampled-token gates:

| Workload | Result |
| --- | --- |
| Eight standard prompts, 24 predictions each, context cap 512 | **192/192** |
| Formerly failing HTTP and story prompts | **48/48** (also included above) |
| Longer 4,096-byte prompt, 24 predictions, context cap 2048 | **24/24** |
| 33-token corpus prefix, chunk 9, matched reference history | **8/8** |
| Same prefix, chunk 11, matched reference history | **8/8** |

CPU threads are 8, KV is F32, flash attention is off, and sampling is greedy
with repetition penalty 1. The long prompt SHA-256 remains
`5ba8da8fc2ff5e856a2e5d689785a80fc4691a5352d11290bdea864aa35f37f5`.
Chunk checks use independently generated schedule-specific histories, not
whole-versus-chunk equivalence. The resumed phase still needs an independent
matched-schedule oracle. None of these runs establishes performance parity.

Candidate executable `/tmp/bitnet-goal-mixed-bitnet` SHA-256:
`c79aee07abb6299201afb3bcc9a6539c6230f611fbde4179777e9c0fcc9a3927`.
Changed diagnostic sources are `/tmp/bitnet-goal-mixed-prefill.c` and
`/tmp/bitnet-goal-mixed-registry.c`, combined with the preceding isolated
format/GDN sources. The independent reference remains the same pinned
llama.cpp build and GGUF. All processes completed.

Evidence: `/tmp/bitnet-goal-q4-layer45-replay.c`,
`/tmp/bitnet-goal-http-{ref,q6-candidate}7-{norm,qkv}45.bin`,
`/tmp/bitnet-goal-http-mixed-pos7.trace`,
`/tmp/bitnet-goal-mixed-http-story-reference.log`,
`/tmp/bitnet-goal-mixed-cpu-eight-reference.log`,
`/tmp/bitnet-goal-mixed-cpu-long-reference.log`, and
`/tmp/bitnet-goal-mixed-matched-chunk{9,11}.log`.

**Production integration remains required.** The diagnostic candidate alters
ordinary quant dispatch directly; that approach must not replace the public
GEMV-order contract. Integrate reference-prefill order explicitly, preserve
prepared resources when mixed groups split, and put the scaled-RMS GDN
normalization rule under architecture ownership. Preserve legacy/default
operator reference values and qualify the corresponding CUDA normalization
path separately. CPU candidate gates passing do not qualify the current
production defaults or close the remaining CUDA long/resumed gates.

## Explicit CPU prefill API integration (2026-11-05)

`bn_quant_matmul_prefill_prepared` now exposes supported reference-tiled
execution without changing ordinary matrix or GEMV-order APIs. It accepts
the caller's prepared weight, validates matrix/block dimensions and size
arithmetic, and prepares request-local Q8_K inputs before dispatch. Invalid
inputs or tiled-input allocation failure return -1 before output writes.
Other formats and batches below the supported tile floor retain the normal
prepared-matmul path, including its existing fallback semantics.

The quant registry owns `BN_QUANT_CAP_CPU_TILED_REFERENCE_PREFILL` and
`bn_quant_format_cpu_prefill_uses_tiled(type, tokens)`. Current compiled
coverage is Q5_K and Q6_K at eight or more tokens on AVX512 VNNI, and Q6_K
on AVX2. Q5_K on AVX2, scalar/other backends, and other formats do not claim
this tiled route. They continue through the existing prepared matrix path;
this partial API coverage is not a model-level parity declaration.

Tests exercise the public entry point against the unchanged independent
Q5_K/Q6_K hashes, serial and threaded output-row partitions, and output
sentinels. The 22 Q4_K prepared reference cases pass through both APIs,
confirming that prepared layouts survive delegation. Short Q6_K batches
match existing matrix execution byte for byte. Invalid request tests check
that output remains untouched. Existing public GEMV-order tests still pass.

Real-model same-input replay through the public API also matches every
captured output across eight repeated input rows: 10,240 outputs per row for
Q5_K and 5,120 per row for Q6_K. The original independent captures are used;
no expected values are generated by this API.
Evidence: `/tmp/bitnet-goal-prefill-api-real-replay.c`,
`/tmp/bitnet-goal-prefill-api-quant.log`, and
`/tmp/bitnet-goal-prefill-api-avx2.log`.

Validation after these API changes: `make BN_ENABLE_CUDA=1 test_quant_avx2`
and full `make -j8 BN_ENABLE_CUDA=1 test` pass. The clean CUDA-enabled build
completes with zero warnings. Logs:
`/tmp/bitnet-goal-prefill-api-clean-build.log` and
`/tmp/bitnet-goal-prefill-api-full-suite.log`. Optional external-model matrix
entries remain skipped unless configured; synthetic suite success does not
promote the isolated model candidate.

Default transformer/generation dispatch is unchanged. Remaining integration
must route modern-model prefill explicitly through this interface, extend
shared-input/multi-projection execution and the remaining format contracts,
retain prepared resources for mixed groups, and declare GDN normalization
semantics in architecture policy. The isolated 192/192 CPU model candidate
is still separate from production defaults.

## Shared-input reference-prefill projections (2026-11-05)

`bn_quant_matmul_prefill_prepared_multi` extends the opt-in CPU entry point to
multiple projections of the same floating-point input. It validates every
weight/output and the shared column count before any output write. Supported
tiled Q5_K/Q6_K projections share one Q8_K quantization pass and one set of
activation buffers. Other formats retain independent prepared matrix
execution, so Q4_K layouts and original F32 inputs are preserved.

A bounded four-task stack array processes arbitrarily many projections in
blocking dispatch groups. The shared activation cache remains alive across
every group and is freed only after execution completes. A projection that
uses a delegated path does not force the other projections into that path.
The single-projection API now delegates to the same implementation.

Synthetic tests mix Q5_K, prepared Q4_K, F32, and two Q6_K projections with
different row counts. They cover one, three, four, and five projections;
8, 9, 17, and 33 tokens; serial and threaded execution; untouched output
sentinels; and invalid late-group widths, weights, and output pointers.
Supported tiled outputs match the existing independent hashes. Prepared
Q4_K matches its pinned AVX512 oracle, and delegated paths match their
existing matrix contract, including AVX2's non-tiled Q5_K route. No golden
values were changed.

After refactoring, real-model single-input replay still matches every
captured Q5_K/Q6_K output across eight repeated rows. Current evidence:
`/tmp/bitnet-goal-prefill-multi-quant.log`,
`/tmp/bitnet-goal-prefill-multi-avx2.log`, and
`/tmp/bitnet-goal-prefill-api-real-replay.c`.

Validation completed: clean CUDA-enabled `bitnet` build with zero warnings,
`make BN_ENABLE_CUDA=1 test_quant_avx2`, and full
`make -j8 BN_ENABLE_CUDA=1 test` all pass. Logs:
`/tmp/bitnet-goal-prefill-multi-clean-build.log` and
`/tmp/bitnet-goal-prefill-multi-full-suite.log`. Optional external-model
matrix cases remain skipped unless configured; this is API qualification,
not new model-default or performance qualification.

This API is not yet wired into default transformer dispatch. Remaining work
includes the other reference format kernels, prequantized projection groups,
prepared-resource retention in host fallbacks, and architecture-owned GDN
normalization before promoting the independently qualified CPU candidate.

## Q3_K reference-prefill qualification (2026-11-05)

The opt-in reference-prefill API now includes Q3_K on AVX2 and AVX512
builds. Its kernel shares decoded weights across four tokens, accumulates
each superblock in integer arithmetic, then applies the reference floating
point scaling order. Existing GEMV and default matrix entry points retain
their previous arithmetic.

`test/reference_cpu_q3k_tiled.c` generates independent GGML CPU hashes for
137 rows, three superblocks, and 8, 9, 10, 11, 17, and 33 tokens. The
10- and 11-token cases cover the two- and three-token tile tails. Direct
kernel and public API tests match all six hashes in serial and threaded
execution, with output sentinels checked. One- and seven-token API calls
retain the existing matrix contract. Mixed projection tests now include
Q3_K after the first four-task dispatch group, alongside Q5_K, prepared
Q4_K, F32, and Q6_K.

The oracle uses llama.cpp revision
`df18bee11e96557b07aa7b12db0976f3c7ac9165` and CPU library SHA-256
`e28660ece5030ec85229b5e03e57d8f0da55866b63943623a651e1adbd02716f`.
Real-model replay through the public API also matches all 17,408 captured
layer-zero Q3_K FFN-up outputs across eight repeated input rows. This
qualifies that projection's arithmetic, not eight independent prompts.
The same replay preserves exact Q5_K and Q6_K projection results.

Validation: clean CUDA-enabled `bitnet` build with zero warnings,
`make BN_ENABLE_CUDA=1 test_quant_avx2`, and full
`make -j8 BN_ENABLE_CUDA=1 test` pass. Logs are
`/tmp/bitnet-goal-q3-api-clean-build.log`,
`/tmp/bitnet-goal-q3-api-avx2.log`, and
`/tmp/bitnet-goal-q3-api-full-suite.log`. Independent oracle output is in
`/tmp/bitnet-goal-reference-q3-tiled.log`; real projection replay source is
`/tmp/bitnet-goal-q3-api-real-replay.c`.

Default transformer dispatch is unchanged. IQ-format reference arithmetic,
prequantized projection integration, and architecture-owned GDN semantics
remain required before promoting the complete CPU model candidate. This
checkpoint establishes no new model-default or performance result.

The next format gate has an independent generator in
`test/reference_cpu_iq_tiled.c`: IQ4_XS, IQ3_S, and IQ3_XXS each have six
GGML CPU hashes over the same odd-row and token-tail shapes. The generator
builds with `-std=c11 -Wall -Wextra -Werror` and runs successfully against
the pinned library above; captured output is
`/tmp/bitnet-goal-reference-iq-tiled.log`. These are reference fixtures for
the pending IQ kernel work, not evidence that the runtime matches them.

## IQ reference-prefill kernels (2026-11-05)

The opt-in CPU prefill API now routes IQ4_XS, IQ3_S, and IQ3_XXS to an
AVX2 kernel with reference block scaling. It computes each superblock's
integer total, multiplies by the weight scale, then fuses the activation
scale into the running sum. The existing matrix API keeps its original
combined-scale arithmetic. Both paths share the same integer dot helper.

The new kernel decodes eight weight rows once per four tokens. Its scratch
is a single stack-resident superblock panel, with bounded partial stores
for arbitrary physical row ranges. It performs no heap allocation and
cannot change arithmetic because a panel allocation failed. Registry
capabilities select this route for batches of at least eight tokens on
AVX2/AVX512; shorter calls retain existing prepared matrix behavior.

All 18 independent IQ hashes from `test/reference_cpu_iq_tiled.c` match
through direct dispatch, the single-projection API, and five shared-input
projections spanning two bounded task groups. Tests cover serial and
threaded execution, output sentinels, odd row counts, every four-token
tail, and unchanged one-/seven-token behavior. Native and AVX2 quant suites
pass without changing any existing golden values. Logs:
`/tmp/bitnet-goal-iq-api-quant.log` and
`/tmp/bitnet-goal-iq-api-avx2.log`.

Real-model replay matches all 17,408 captured layer-zero IQ4_XS FFN-gate
outputs across eight repeated input rows. Q3_K, Q5_K, and Q6_K captures
also remain exact. Source and output:
`/tmp/bitnet-goal-iq-api-real-replay.c` and
`/tmp/bitnet-goal-iq-api-real-replay.log`. This is projection qualification;
the repeated rows are not independent model histories.

The clean CUDA-enabled `bitnet` build passes with zero warnings
(`/tmp/bitnet-goal-iq-api-clean-build.log`). Full
`make -j8 BN_ENABLE_CUDA=1 test` also passes
(`/tmp/bitnet-goal-iq-api-full-suite.log`); optional external-model cases
remain unconfigured. This is the standard suite, not a new full CUDA
hardware or full-model qualification. Default transformer integration,
prequantized projection groups, and architecture-owned GDN semantics are
still pending. No model-default or performance claim follows from this
kernel checkpoint.

## Caller-prepared reference-prefill activations (2026-11-05)

`bn_quant_matmul_prefill_prepared_kquant_input_multi` accepts caller-owned
Q8_K values, scales, and block sums for shared-input projections. Supported
tiled kernels consume those buffers directly without allocating or
re-quantizing activations. Fully tiled groups accept null original-float
and scratch pointers. Delegated projections require those pointers and
retain their existing prepared matrix behavior; this includes Q4_K's
prepared panels and original F32 inputs. Delegated paths can still perform
their own input preparation, so this API does not claim to remove that
work for every format.

The float-input and prequantized APIs share group validation and bounded
four-task dispatch. Invalid group members, missing activation buffers,
incompatible widths, and missing delegated inputs are rejected before any
output write. Batches below each format's tiled floor use their established
matrix arithmetic and require original float inputs.

Tests extend existing independent Q3_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, and
IQ3_XXS hash gates to the prequantized API, including calls without float
inputs. Five-projection IQ groups span two dispatch groups. Mixed
Q5_K/prepared-Q4_K/F32/Q6_K/Q3_K tests exercise one, three, four, and five
projections, serial and threaded execution, output sentinels, and invalid
late members. One- and seven-token cases retain existing behavior. Native
and AVX2 quant suites pass; no existing golden values were changed.

Prequantized real-model replay also matches every captured Q5_K, Q6_K,
Q3_K, and IQ4_XS projection output across eight repeated rows, with null
float-input and scratch pointers. This checks direct use of the prepared
activations; it does not represent independent model histories. Source and
output are `/tmp/bitnet-goal-prequant-api-real-replay.c` and
`/tmp/bitnet-goal-prequant-api-real-replay.log`.

Validation: clean CUDA-enabled `bitnet` build with zero warnings,
`make BN_ENABLE_CUDA=1 test_quant_avx2`, and full
`make -j8 BN_ENABLE_CUDA=1 test` pass. Optional external-model matrix cases
remain unconfigured. Logs are
`/tmp/bitnet-goal-prequant-api-clean-build.log`,
`/tmp/bitnet-goal-prequant-api-avx2.log`, and
`/tmp/bitnet-goal-prequant-api-full-suite.log`.

This API is a prerequisite for model-path integration. It is not yet
selected by default transformer dispatch, and does not qualify full-model
correctness or establish a throughput improvement.

## Explicit GDN Q/K normalization semantics (2026-11-05)

`BnModelArchOps` now owns an explicit `BnSSMQKNormMode`, copied into
`BnConfig` during architecture setup. Qwen35/Qwen35MoE and Qwen4exp declare
RMS-scaled Q/K normalization; other registered families retain clamped L2.
Applying an absent architecture resets this field to clamped L2.

The SSM normalization context carries the mode. RMS-scaled execution uses
the double sum of individually rounded float squares, RMS epsilon divided
by head dimension, and two separately rounded output multiplies: the RMS
scale followed by `1/sqrt(head_dim)`. The portable implementation is shared
by scalar, AVX2, NEON, and WASM dispatch for this mode. Existing L2 arithmetic
is retained under mode zero. Production CPU/prefill/debug callers currently
select that existing mode explicitly; declaring architecture semantics does
not yet promote a model-default change.

`test/reference_cpu_gdn_norm.c` generates ten independent GGML CPU graph
hashes: dimensions 1, 7, 16, 128, and 256 at epsilons 1e-6 and 1e-5, with
zero, tiny, normal, and large finite heads in each case. Kernel tests match
every hash, compare both Q and K, check untouched buffer sentinels, and
exercise split head ranges. Native AVX512, explicit x86-64-v3 AVX2, and
scalar-only builds pass; this does not establish execution on ARM/WASM.
Existing L2 equivalence checks also pass. Reference output and focused logs:
`/tmp/bitnet-goal-reference-gdn-norm.log`,
`/tmp/bitnet-goal-gdn-policy-ssm.log`,
`/tmp/bitnet-goal-gdn-policy-avx2.log`, and
`/tmp/bitnet-goal-gdn-policy-scalar.log`.

The clean CUDA-enabled `bitnet` build is warning-free, and full
`make -j8 BN_ENABLE_CUDA=1 test` passes. Optional external-model matrix
cases remain unconfigured. Logs:
`/tmp/bitnet-goal-gdn-policy-clean-build.log` and
`/tmp/bitnet-goal-gdn-policy-full-suite.log`.

The remaining integration must select this architecture mode in model
execution, implement and qualify the corresponding GPU operation, and
re-run independent model histories alongside the qualified projection
kernels. No new model-default or performance result is claimed here.

## CUDA GDN RMS-scaled operation (2026-11-05)

The CUDA SSM Q/K normalization operation now accepts an explicit mode in
parameter `p[4]`. Mode zero preserves the existing clamped-L2 kernel; the
RMS-scaled mode uses a separate kernel with 256 threads below width 1024
and 1024 threads otherwise. It follows the CUDA reference's FP32/FMA partial
sums and warp-XOR reduction, RMS epsilon divided by head dimension, and
separate RMS/output-scale multiplies. Unsupported mode values fail execution.

`test/reference_cuda_gdn_norm.c` independently generates fourteen GGML CUDA
graph hashes for widths 1, 7, 16, 128, 256, 1024, and 1536 at two epsilons.
Each fixture includes zero, tiny, normal, and large finite input heads.
The reference is llama.cpp revision
`df18bee11e96557b07aa7b12db0976f3c7ac9165`, CUDA library SHA-256
`e3330904452e38a9dedc307dedf83e182322a27d6b10dd1df457e02326082840`.
These CUDA values are distinct from the CPU fixtures where their reduction
orders differ; the CPU expected values remain unchanged.

All fourteen hashes match in synchronous, stream, and CUDA graph execution.
Tests check separate Q/K buffers with nonzero offsets, untouched leading
and trailing sentinels, and RMS/L2/RMS mode switches within one allocation
generation. Existing L2 epsilon checks still pass. Focused command:
`/tmp/bitnet-goal-gdn-cuda-test --ssm-qk-rms-reference`, with
`BN_CUDA_DEVICE=0 CUDA_VISIBLE_DEVICES=0`. Logs:
`/tmp/bitnet-goal-reference-cuda-gdn-norm.log` and
`/tmp/bitnet-goal-gdn-cuda-focused.log`.

The clean CUDA-enabled `bitnet` build is warning-free and full
`make -j8 BN_ENABLE_CUDA=1 test` passes. Logs:
`/tmp/bitnet-goal-gdn-cuda-clean-build.log` and
`/tmp/bitnet-goal-gdn-cuda-full-suite.log`. Optional external-model matrix
cases remain unconfigured.

Model graph emission and the direct prefill helper still select legacy
normalization. The next integration must select the architecture mode
alongside CPU reference projection dispatch, then qualify independent
model histories. This checkpoint claims no new model-default behavior,
Metal/WebGPU mode support, or performance result.

## GDN model-path normalization integration (2026-11-05)

CPU decode/prefill and CUDA graph decode now select the architecture's Q/K
normalization mode. The fused CUDA SSM prefill interface carries that mode
explicitly, independently of FFN activation. The host-visible CUDA
convolution/normalization hook also accepts the mode, keeping that operation
on CUDA when RMS-scaled normalization is selected. Existing layer-state
fixtures explicitly select legacy L2 and retain their expected values.

`test/reference_cuda_gdn_convnorm.c` adds six independent strided GGML CUDA
fixtures for three prompt rows, two Q/K heads, widths 7/128/1024, and two
epsilons. Identity convolution taps make its output equal to the graph's
SiLU input. Tests check packed Q/K/V outputs, retained raw convolution
history, and output/state sentinels. This caught two subnormal normalized
outputs in the 1024-wide case: the reference produces negative zero while
ordinary round-to-nearest multiplication retained subnormals. Explicit
`mul.rn.ftz.f32` for the two RMS output multiplies fixes the mismatch without
changing legacy kernels or golden values. All six batched hashes and the
fourteen isolated normalization hashes pass. Logs:
`/tmp/bitnet-goal-reference-cuda-gdn-convnorm.log` and
`/tmp/bitnet-goal-model-norm-ftz-focused.log`.

Real projection replay provides another arithmetic check. The pinned GPU
model probe captures layer-zero convolution and normalized Q values for
"HTTP status code 404 means" at position 7. Replaying the same convolution
input through the new CUDA normalization operation matches 2,048/2,048
reference Q outputs exactly; legacy L2 matches 206/2,048 with maximum
difference 2.38418579e-7. Diagnostic source:
`/tmp/bitnet-goal-model-norm-real-op.c`; independent captures:
`/tmp/bitnet-goal-model-norm-llama_ssm_conv.bin` and
`/tmp/bitnet-goal-model-norm-llama_ssm_q_norm.bin`. This isolates one
operation; it does not prove the complete pipeline.

The first integrated CUDA model comparison matches 24/24 long-prompt tokens
(`/tmp/bitnet-goal-model-norm-cuda-long.log`), improving on the preceding
21/24 gate. Short prompts instead match 172/192 tokens with all eight first
IDs matching (`/tmp/bitnet-goal-model-norm-cuda-short.log`). Restoring the
CUDA convolution hook leaves that short result at 172/192
(`/tmp/bitnet-goal-model-norm-hook-cuda-short.log`). These model histories
precede the final subnormal-output correction and require requalification.
The short regression remained unresolved at this checkpoint; the following
batched-scan checkpoint identifies and corrects it without reverting the
reference arithmetic.

The standard suite passes before the final hook/FTZ changes. The earlier
full CUDA suites also pass, including preserved original goldens, in
`/tmp/bitnet-goal-gdn-cuda-hardware-suite.log` and
`/tmp/bitnet-goal-model-norm-cuda-hardware-suite.log`. The final clean build
passes with zero warnings
(`/tmp/bitnet-goal-model-norm-final-clean-build.log`). Final standard and
hardware suite results remain pending in
`/tmp/bitnet-goal-model-norm-final-full-suite.log` and
`/tmp/bitnet-goal-model-norm-ftz-hardware-suite.log`. The frozen current
model-comparison binary is `/tmp/bitnet-goal-model-norm-current-bitnet`, with
SHA-256 in `/tmp/bitnet-goal-model-norm-current-sha256.txt`; its short/long
requalification logs use the `bitnet-goal-model-norm-current-cuda-` prefix.
CPU reference-projection integration and independent short/long/chunked/
resumed model qualification remain open. No performance parity or modern
Metal/WebGPU qualification is claimed.

## Corrected CUDA batched-scan normalization (2026-11-05)

The missing model-path branch was the fused prefill scan: its per-token
fallback and graph decode selected RMS-scaled normalization, but the scan
still launched the legacy L2 kernel. Actual fused-prefill Q capture matched
only 206/2,048 reference elements, exactly the isolated old-L2 replay.
The scan now selects normalization by the same mode as the other paths.
Actual fused-prefill Q now matches all 2,048 reference elements exactly
(`/tmp/bitnet-goal-model-norm-scan-qkv.bin`), using the earlier independent
GPU model captures and identical prompt/position.

`test/reference_cuda_gdn_prefill_norm.c` adds four independent CUDA GDN
state hashes: 2/9 rows and epsilons 1e-6/1e-3. Nonzero retained convolution
history supplies Q/K/V for the first row, followed by zero projections;
gate is zero, beta is 0.5, and initial recurrent state is zero. This
isolates normalization selection in a full fused layer without requiring a
model. Both scan and serial prefill match all 16,384 expected state values
via bit hashes. Legacy mode must produce a different hash, preventing a
zero-input fixture from concealing a missed mode. Tests also verify output
residuals and sentinels. All earlier expected values remain unchanged.
Source and output:
`test/reference_cuda_gdn_prefill_norm.c` and
`/tmp/bitnet-goal-reference-cuda-gdn-prefill-norm.log`; focused runtime log:
`/tmp/bitnet-goal-model-norm-scan-focused.log`.

Independent model comparisons now pass 192/192 short tokens and 24/24 long
tokens, with matching first IDs and token counts. Logs:
`/tmp/bitnet-goal-model-norm-scan-cuda-short.log` and
`/tmp/bitnet-goal-model-norm-scan-cuda-long.log`. The compared binary is
`/tmp/bitnet-goal-model-norm-scan-bitnet`, SHA-256
`73cd5f4a0be2860ff8f8e44f0f359409e7d42f743e7133b6871c3f2fd981ad3b`.
Reference flags, precision, threads, context limits, and the 4,096-byte long
prompt match the previous model gates. Repetition penalty is one; reference
attention is non-flash with F32 KV.

Internal whole/chunked prefill, decode, 64-token resume, and decode checks
also pass 16/16 greedy IDs for schedules 33/9, 256/64, and 1024/129. Logs use
the `/tmp/bitnet-goal-model-norm-scan-continuation-` prefix. These are
execution-consistency diagnostics, not independent model histories or a
bitwise/logit-equivalence claim.

The clean CUDA-enabled build is warning-free, and full standard
`make -j8 BN_ENABLE_CUDA=1 test` passes. Logs:
`/tmp/bitnet-goal-model-norm-scan-clean-build.log` and
`/tmp/bitnet-goal-model-norm-scan-full-suite.log`. Optional model-matrix
cases remain unconfigured. The updated full CUDA hardware suite also passes,
including original reference hashes and the new nonzero-key fixtures:
`/tmp/bitnet-goal-model-norm-scan-hardware-suite.log`. A separately frozen
clean-build binary, `/tmp/bitnet-goal-model-norm-clean-bitnet`, also passes
192/192 short and 24/24 long tokens with the same comparison commands.
Its SHA-256 is
`3b0fd7aebc226d9eb4d59aeffc206a50590ff7b8b2985e2b8dd43c3637c02dee`;
logs use the `bitnet-goal-model-norm-clean-cuda-` prefix, and both commands
return zero in `/tmp/bitnet-goal-model-norm-clean-status.txt`.

CPU reference-projection integration and independent chunked/resumed model
qualification remain open. This checkpoint makes no performance comparison.

## CPU reference-prefill model integration (2026-11-05)

CPU requests with the architecture's reference-recurrent policy now select
the qualified reference-prefill matrix APIs. Other model policies and GPU
numerical paths retain their existing dispatch. The prefill backend owns
the quant API calls; orchestration selects the model contract and retrieves
prepared weights from backend-owned resources. Mixed groups and individual
host fallback projections keep those prepared layouts, including the Q4_K
projection previously lost when its companion format could not share the
prepared K-quant route.

Prepared K-quant groups reuse their activation cache through the new
prequantized API. Delegated matrix preparation uses the session's separate
`x_q` scratch, preserving the read-only cached Q8_K inputs until all blocking
dispatches finish. The float-input route shares preparation where its
existing callers already group projections. This integration does not
claim that every host projection has been co-scheduled into a shared group.

Projection helpers now return status through PLE, hyperconnection, single,
and grouped callers; new reference API failures abort prefill and release
its temporary arena. A singleton hyperconnection injection also receives
session scratch instead of a null pointer. The unchanged output-row/state
tests cover that valid tail path. `test_prefill_projection_failure` corrupts
the first layer's Z format after a successful control run and checks all
three prefill APIs: failure must be reported, logits stay untouched, and
position plus recurrent/convolution state remain unchanged before that
layer's convolution. This does not promise rollback after an arbitrary
failure later in a request.

The first integrated native AVX512 CPU binary passes 48/48 tokens on the
two formerly failing prompts, then 192/192 standard short tokens and
24/24 long tokens. CUDA regression comparisons also pass 192/192 and 24/24.
Those model comparisons use `/tmp/bitnet-goal-cpu-integration-bitnet`,
SHA-256 `44188e030fbacedf5086a04eb378dd2cb45396f249192058b08d122f6730ba98`,
before the direct quant calls were moved behind thin prefill-backend
adapters. All four command statuses are zero in
`/tmp/bitnet-goal-cpu-integration-status.txt`; logs use the
`bitnet-goal-cpu-integration-{cpu,cuda}-{short,long}.log` names.

The backend architecture check and full standard suite pass after that
boundary correction (`/tmp/bitnet-goal-cpu-integration-backend-matrix.log`
and `/tmp/bitnet-goal-cpu-integration-adapter-full-suite.log`). The CUDA
hardware suite passes with existing reference values preserved
(`/tmp/bitnet-goal-cpu-integration-cuda-hardware-suite.log`). Final clean
CUDA-enabled build passes with zero warnings
(`/tmp/bitnet-goal-cpu-integration-adapter-clean-build.log`). Optional
external-model matrix cases remain unconfigured.

The exact final clean binary is frozen as
`/tmp/bitnet-goal-cpu-integration-final-bitnet`, SHA-256
`d12a2294058aa2d4a1b075e13a95b35b3eefb98da914bdb0c6dbf887e080d591`.
Its four CPU/CUDA short/long model reruns all return zero: 192/192 short
and 24/24 long tokens on each backend. Status is recorded in
`/tmp/bitnet-goal-cpu-integration-final-status.txt`; logs use the
`bitnet-goal-cpu-integration-final-` prefix. No newer whole-model AVX2/NEON
qualification or performance claim follows from the native CPU gates.

## Independent chunked and resumed prefill (2026-11-05)

The paired reference harness passes all ten Qwen3.8 27B cases: CPU and CUDA
at prefix/chunk schedules 33/9, 33/64, 33/11, 256/64, and 1024/129. Each
case checks eight greedy IDs after initial prefill, commits all eight tokens,
submits a 64-token resumed input, then checks eight more IDs: 160/160 IDs
across the matrix. Chunk 64 submits the 33-token prefix and 64-token resume
in one batch each. This qualifies the tested schedules, not arbitrary contexts.

The runtime supplies input IDs only; a separately linked llama.cpp probe
produces expected output IDs. Both engines use eight threads, non-flash
attention, F32 KV on CPU and F16 KV on CUDA. Each engine uses the same chunk
schedule; whole and chunked CPU reference arithmetic can yield different
histories. The replay teacher-forces reference IDs after scoring each greedy
prediction, so any mismatch fails without changing subsequent history.
Resumed probe mode commits fixed-length phases even through EOS; legacy
probe generation retains its original stopping behavior.

Reproduce one case after separately building the probe against the intended
llama.cpp installation:

```sh
make BN_ENABLE_CUDA=1 test_prefill_reference
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin CUDA_VISIBLE_DEVICES=0 \
python3 test/compare_prefill_reference.py \
  --model /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --probe /path/to/independently-linked-llama-probe \
  --backend cuda --tokens 1024 --chunk 129 \
  --output-dir /tmp/qwen-resumed-cuda-1024-129
```

Use `--backend cpu` for CPU qualification. Artifacts include frozen corpus,
input IDs, independent output IDs, both engine logs, executable hashes,
commands, precision, and exit statuses. The harness rejects incomplete oracle
output, incorrect resume position, failed replay, and missing success markers.
Seven model-free protocol tests run as part of `make test`.

The qualified probe is `/tmp/bitnet-goal-resumed-reference-probe`, SHA-256
`09ba75801c51911023a4c617d5dc5bd78e6a773a20b2cf41abd8eb7e77570e32`,
linked to llama.cpp revision `df18bee11e96557b07aa7b12db0976f3c7ac9165`
in `/home/mark/artalis.io/tools/llama.cpp/build/bin`. The replay snapshot is
`/tmp/bitnet-goal-prefill-reference-runner`, SHA-256
`5a84c64e84ddca4e8707aa364579a223070604ef9297c9b4c2b30dd861ee821d`.
Reports and logs are under `/tmp/bitnet-goal-independent-prefill/`; aggregate
statuses are in `matrix-status.json`. All ten paired reports and runtime logs
were checked. The post-harness clean CUDA-enabled build has zero warnings;
full standard tests pass (`/tmp/bitnet-goal-resume-probe-clean-build.log` and
`/tmp/bitnet-goal-resume-probe-full-suite.log`). These are correctness results;
AVX2/NEON whole-model and vLLM performance qualification remain open.

## AVX2 reference revision and Q5_K prefill (2026-11-05)

The pre-existing llama.cpp `build-avx2/bin` reports revision `3d3d7c818`
(version 0.3.0), rather than the qualified native reference revision
`df18bee11e96557b07aa7b12db0976f3c7ac9165` (version 0.6.0). Its initial
short/long comparisons and resumed probe results do not qualify the current
runtime. In particular, a probe compiled with current headers must not be
linked to those older context-parameter libraries.

An isolated AVX2-only CPU reference was built from the qualified revision in
`/tmp/bitnet-goal-llama-avx2-reference`, with native and all AVX512 features
disabled, AVX2/FMA/F16C enabled, Release optimization, and CUDA disabled.
Configure arguments and source revision are recorded in
`/tmp/bitnet-goal-llama-avx2-reference-provenance.json`. The source's existing
five-line GLM architecture alias is retained; no reference source was changed
for this build. Build log:
`/tmp/bitnet-goal-llama-avx2-reference-build.log`.

Against the current reference, the pre-change runtime also fails the 33/9
independent resumed case at 15/16 IDs, confirming an actual AVX2 gap independent
of the old library mismatch. Artifacts:
`/tmp/bitnet-goal-current-prefill-avx2-before-q5/cpu-33-9/`.
The current independent Q5_K operator generator produces the same four batched
graph hashes on AVX2 as on AVX512, while the old reference produces GEMV-order
hashes. Reference hashes from the current build are recorded in
`/tmp/bitnet-goal-current-reference-q5-avx2-hashes.txt`.

The reference-prefill API now selects the shared four-token Q5_K tile on AVX2
as well as AVX512. Legacy GEMV and ordinary batch entry points keep their
existing accumulation order. The tiled API checks all four independent hashes
at 8, 9, 17, and 33 tokens with 137 rows, including threaded partitions and
float-input/prequantized public APIs. The AVX2 quant suite passes. Replaying
the same current-reference 33/9 history after the correction passes 16/16 IDs
(`/tmp/bitnet-goal-q5-avx2-33-9-replay.log`), versus 15/16 before it. The
full short gate now passes 192/192 IDs, and the long gate passes 24/24,
with zero token-count mismatches (`/tmp/bitnet-goal-q5-avx2-{short,long}.log`).
The qualified executable is `/tmp/bitnet-goal-q5-avx2-bitnet`, SHA-256
`1f6a2603f18d1989e8bee06555b5000486b7cc439c716274a4c4b598b1669090`.
All five paired schedules (33/9, 33/64, 33/11, 256/64, and 1024/129) pass
16/16 IDs each, totaling 80/80. Their reports and runtime logs were checked
in `/tmp/bitnet-goal-q5-independent-avx2/`. All seven command statuses are zero in
`/tmp/bitnet-goal-q5-avx2-status.json`. The post-change clean CUDA build has
zero warnings, full standard and AVX2 quant suites pass, and the full CUDA
hardware suite passes, including diagnostic isolation. Logs use
`/tmp/bitnet-goal-q5-avx2-final-{clean-build,full-suite}.log` and
`/tmp/bitnet-goal-q5-avx2-cuda-hardware-suite.log`. This is a correctness
correction, with no speed claim. The final native CPU/CUDA runtime snapshot
`/tmp/bitnet-goal-q5-final-native-bitnet`, SHA-256
`aaa8f6751161d78b0d0a489770c34e033a19ce67c359e0e4b19d0ee6fc1207e6`,
also passes all four short/long reruns: 192/192 and 24/24 on each backend.
Statuses are in `/tmp/bitnet-goal-q5-final-native-status.json`; logs use
`/tmp/bitnet-goal-q5-final-native-{cpu,cuda}-{short,long}.log`.

Build the optional AVX2 replay tool with `make test_prefill_reference_avx2`,
then supply `--runtime ./test_prefill_reference_avx2` to the paired Python
harness. Link its independent probe to the matching isolated reference and
set `LD_LIBRARY_PATH` to that build's `bin` directory.

Fresh probes support `--reference-info`, which reports the linked llama/GGML
versions and GGML commit without loading a model. Supply
`--reference-commit df18bee11e96557b07aa7b12db0976f3c7ac9165` to the paired
harness to reject a missing, duplicated, or mismatched revision before input
extraction or model execution. The report records the actual revision and
requested revision. Eleven model-free protocol tests now cover these checks;
the current AVX2 probe also passes its operator-free self-tests.
The real 33/9 paired run with revision pinning passes 16/16 IDs in
`/tmp/bitnet-goal-q5-pinned-avx2/cpu-33-9/`. Full standard tests pass with the
eleven protocol checks included (`/tmp/bitnet-goal-reference-pin-full-suite.log`).

## Scoped resident CUDA prefill trace (2026-11-05)

`bench_prefill --cuda-profile` brackets each measured CUDA run with profiler
start/stop calls. Model loading, weight upload, warmup, and session reset are
outside capture. A device synchronization after reset is also outside timing
and capture. Prefill's completion readback remains inside timing. Capture stops
on a failed run too; failures do not print a successful benchmark result.
`--cuda-profile` requires `--cuda`. `--kv32` selects F32 KV for matched-precision
comparisons; the benchmark's existing default remains F16 KV.

Reproduce a single measured trace:

```sh
make BN_ENABLE_CUDA=1 bench_prefill
CUDA_VISIBLE_DEVICES=0 BN_CUDA_DEVICE=0 nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/qwen-prefill-256 \
  ./bench_prefill /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf 256 1 8 \
    --cuda --cuda-profile --with-logits --prompt-file docs/inference.md --profile
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  --format csv --output /tmp/qwen-prefill-256-stats /tmp/qwen-prefill-256.nsys-rep
```

The first real-model capture uses 256 corpus tokens, one measured run after
warmup, eight threads, F16 KV, non-flash attention, and final-token logits.
It returns success and contains kernel and transfer events. Exact SQLite
transfer totals, excluding model upload, are:

| Direction | Copies | Bytes | Summed device copy duration |
| --- | ---: | ---: | ---: |
| Host to device | 135 | 453,027,408 | 11.205 ms |
| Device to host | 82 | 571,418,628 | 10.035 ms |
| Device to device | 16 | 83,886,080 | 0.038 ms |

The trace also has 200 blocking `cudaMemcpy` API calls totaling 326.701 ms,
33 async-copy API calls, and one explicit stream synchronization. Blocking
copy API time includes waiting for preceding kernels; it is not pure transfer
time. GPU copy durations and kernel durations can overlap and must not be
added as a critical-path estimate. The largest summed kernel durations are
packed K-quant MMQ (92.867 ms/191 launches) and the leading IQ4 codebook
ordered MMA specialization (77.688 ms/70 launches). Actual tiled kernels are
running, while intermediate host traffic remains substantial. Host profiling
reports 48 fused chain layers and zero decode-stage time; its `attn_cpu` label
is a host-stage label and does not establish CPU attention fallback.

The profiled wall time is 432.909 ms (591.348 tokens/s), recorded only as a
diagnostic with profiling overhead and concurrent CPU verification activity.
It is not an uncontended throughput baseline or a vLLM comparison. Artifacts:
`/tmp/bitnet-goal-prefill-capture-256.{nsys-rep,sqlite,log}`,
`/tmp/bitnet-goal-prefill-capture-256-stats_*.csv`, and the matching
`-manifest.json`/`-summary.json`. The captured executable is preserved as
`/tmp/bitnet-goal-profiled-prefill-f16`.

The final capture/precision-option clean build is warning-free, full standard
tests pass, and a real CUDA `--kv32` run reports F32 KV successfully. Logs:
`/tmp/bitnet-goal-capture-kv32-{clean-build,full-suite}.log` and
`/tmp/bitnet-goal-prefill-kv32-smoke.log`. The latter runs on physical GPU 1
while the reference initializes on GPU 0 and serves only as a functional
precision-option check. CPU-only C11 syntax checking and the capture flag's
required-CUDA CLI rejection also pass.

An isolated vLLM 0.33.0 / PyTorch 2.14.0+cu130 environment is installed in
`/tmp/bitnet-goal-vllm-env`. The official [GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin)
at revision `e454affe0ca23c9dc8754ec3033270cb30622ed5` builds for SM120.
Its two Qwen configuration imports require a compatibility adjustment to
import the classes from Transformers 5.20.2 instead of the removed vLLM
modules. Text-only `qwen3_5_text` must also map to the current core
`Qwen3_5ForCausalLM` class; the old conditional-generation mapping fails on a
missing `vision_config`. The isolated dependency patch is checked in as
`bench/patches/vllm_gguf_033_qwen_text.patch`; it applies to the plugin revision
above. No kernel or weight arithmetic is changed. Plugin import and GPU
detection pass. Version/patch provenance is recorded in
`/tmp/bitnet-goal-vllm-reference-environment.json`.

A real same-GGUF load attempt rejects F32 activations with explicit F16 KV
at vLLM configuration validation; that precision combination is unsupported
by this vLLM version. Log: `/tmp/bitnet-goal-vllm-qwen38-smoke.log`.
The next smoke attempt uses F32 activations, automatic F32 KV, and F32
convolution/recurrent state, matching the runtime's available F32 KV path.
It uses the same GGUF weights and shared input IDs, with a text-only HF
configuration derived from the local model config; the NVFP4 checkpoint's
weights are not substituted. The text-only architecture retry loads the same GGUF, but automatic GDN
selection reaches a FlashInfer SM120 DSL kernel that rejects F32 inputs
(`/tmp/bitnet-goal-vllm-qwen38-smoke-f32-text.log`). A subsequent attempt
selects the supported Triton GDN prefill and decode backends explicitly,
retaining F32 precision (`/tmp/bitnet-goal-vllm-qwen38-smoke-f32-triton.log`).
That attempt also loads the model but fails in `ChunkGatedDeltaRuleFunction`,
which explicitly rejects F32 activations. Both available GDN prefill backends
therefore reject the tested FP32 workload. The exact F32 comparison is
unqualified, rather than silently substituting a lower-precision reference.
The separate FP16 activation/F16 KV generation smoke passes and emits eight
IDs: `5604,13,271,550,4962,89583,21753,271`. Log and result are
`/tmp/bitnet-goal-vllm-qwen38-smoke-f16.log` and
`/tmp/bitnet-goal-vllm-qwen38-smoke-f16-result.json`. Its activation precision differs from the current
runtime, so it does not establish matched performance. No vLLM speed ratio
is established.

## Experimental mixed-format Q/K prefill view (2026-11-05)

The Qwen3.8 checkpoint uses different formats for Q and K in its attention
layers (for example layer 3 uses Q5_K/Q4_K). Its auxiliary prediction block
also disables optional copied weight stacks. These conditions prevented the
fused dense-layer path from seeing a Q/K stack even though both original
weights were resident.

`BN_CUDA_ENABLE_PREFILL_QK_PAIR=1` opts into a backend-owned borrowed Q/K view.
The view retains each child's original format, owns neither child, allocates
no device weights, and is consumed only by the dense-prefill hook. General
attention resource planning does not advertise it as a stacked weight. The
model/session/weight ownership split is preserved, and default dispatch stays
unchanged. Metadata-only views remain available with auxiliary prediction
blocks; a host upload regression test protects that distinction from copied
optional layouts.

Focused CUDA tests preserve independently generated K hashes while changing
Q's format, including gated Q, aliased output, guard values, and child use
after view destruction. Host resource-selection/auxiliary-upload tests pass.
The first clean CUDA build has zero warnings; full standard and CUDA hardware
suites pass, with original reference values preserved. Logs use the
`/tmp/bitnet-goal-qk-pair-` prefix.

Actual model profiling confirms 64 fused layers for the initial chunk instead
of 48; resumed chunks retain their existing dispatch. An earlier candidate
still inherited the auxiliary-block restriction and never selected the view;
its passing model comparisons do not qualify the active path. The active
candidate passes 192/192 short and 24/24 long CUDA IDs, but the independent
F16-KV schedule matrix is 79/80: 33/9, 33/11, 256/64, and 1024/129 pass 16/16,
while 33/64 misses the sixth initial greedy ID by a 0.00349 logit gap.
Expected IDs are unchanged. Statuses/logs are in
`/tmp/bitnet-goal-qk-pair-active-matrix/` and
`/tmp/bitnet-goal-qk-pair-active-cuda-{short,long}.log`.

Inspection found that dense prefill hard-coded fused RoPE rounding while
standalone Q/K preparation honors the model's separate-RoPE setting. The
borrowed-pair branch now forwards that existing setting; stacked legacy
layouts retain their current contract. The formerly failing 33/64 case now
passes 16/16 IDs with its reference IDs unchanged
(`/tmp/bitnet-goal-qk-pair-rope-independent-33-64.log`).

A new independent standalone CUDA RoPE generator
(`test/reference_cuda_qk_pair_rope.c`) produces hash `cbb2be23d5347d46`. The
paired dense layer matches every bit of that 1,056-value K result when the
model requests separate RoPE; the fused rounding mode must differ. Exact
identity projections and zero-epsilon normalization isolate the rotation
contract. These focused CUDA tests pass
(`/tmp/bitnet-goal-qk-pair-rope-cuda-reference.log`), preserving existing
reference values. The corrected path passes all five independent schedules
(80/80 IDs), the 192/192 short and 24/24 long CUDA token gates, and a further
full-batch 256/256 case (16/16 IDs). Statuses are recorded in
`/tmp/bitnet-goal-qk-pair-rope-matrix/status.json` and
`/tmp/bitnet-goal-qk-pair-rope-cuda-status.json`; all five runtime logs were
checked. The final clean build has zero warnings, and full standard/CUDA
suites pass, including the new fixture and CUDA diagnostic isolation. Logs:
`/tmp/bitnet-goal-qk-rope-final-{clean-build,full-suite,cuda-suite}.log`.
The feature remains opt-in.

## Same-binary Q/K view transfer comparison (2026-11-05)

After full-batch 256-token independent qualification passed, scoped Nsight
runs compare the same executable, GPU 0, corpus, 256-token prefix, final-token
logits, F16 KV, eight threads, one warm measured run, and non-flash attention.
Only `BN_CUDA_ENABLE_PREFILL_QK_PAIR` changes. Loading, upload, warmup, and
reset are excluded from both captures.

| Captured quantity | Control | Borrowed Q/K view |
| --- | ---: | ---: |
| Host-to-device bytes | 453,027,408 | 184,591,952 |
| Device-to-host bytes | 571,418,628 | 197,097,476 |
| Host/device copy count | 217 | 153 |
| Device-to-device bytes | 83,886,080 | 40,960 |
| Kernel launches | 1,584 | 1,790 |
| Summed kernel durations | 308.302 ms | 308.729 ms |
| Profiled wall time | 426.445 ms | 336.557 ms |

Host/device bytes fall 62.74%. Similar summed kernel duration alongside lower
wall time shows reduced orchestration overhead; blocking copy API durations
still include kernel waits and cannot be attributed wholly to PCIe transfer.
The profiled wall-time change is diagnostic, not an unprofiled speed gate or a
vLLM ratio. Unprofiled repeated runs remain next before deciding a default
policy. Short-prompt performance must be measured separately.

Artifacts are in `/tmp/bitnet-goal-qk-pair-rope-traces-256/`, including the
frozen executable/corpus, command manifest, both `.nsys-rep`/SQLite captures,
CSV reports, JSON summary, and the independent full-batch qualification logs.
This change removes an avoidable Q/K format constraint from resident-layer
execution; it does not yet make the entire hybrid prefill device-resident.

## Unprofiled Q/K view prefill measurements (2026-11-05)

Sequential same-binary runs after correctness jobs completed measure three
warm iterations per size and mode on GPU 0. Both modes use identical corpus
prefixes, F32 activations, F16 KV, final-token logits, non-flash attention,
eight threads, completed device work, and untimed session reset. Profiling is
disabled. The full-batch 1024-token independent check on the final clean replay
binary also passes 16/16 IDs before timing starts.

| Prefix tokens | Control median | Q/K view median | Median latency reduction |
| ---: | ---: | ---: | ---: |
| 33 | 175.649 ms | 165.848 ms | 5.58% |
| 256 | 418.505 ms | 324.255 ms | 22.52% |
| 1024 | 2024.091 ms | 1225.374 ms | 39.46% |

These are initial internal measurements, not statistical confidence bounds
or vLLM parity. The change remains opt-in through
`BN_CUDA_ENABLE_PREFILL_QK_PAIR=1`; the control omits that variable. Three-run
medians, means, token rates, commands, binary/corpus hashes, and independent
qualification artifacts are in
`/tmp/bitnet-goal-qk-pair-rope-unprofiled/{manifest,summary}.json` and the
per-case logs. This does not close the remaining backend, expert-offload,
serving, or matched-vLLM acceptance gates.

## ARM NEON 128 SSM functional check (2026-11-05)

An isolated GCC 13.3 AArch64 cross-toolchain and QEMU user emulator were
extracted under `/tmp/bitnet-goal-arm-tools/`; no system packages were
installed. The SSM suite builds warning-free with C11 and
`-march=armv8-a+simd`. Compiler macros confirm NEON enabled and dot-product
extensions absent, covering the baseline 128-bit NEON target.

The actual AArch64 executable passes under QEMU, including the independent
Q/K RMS-normalization hashes, convolution, legacy L2 normalization, delta
state update, NEON reduction order, and gates. The suite explicitly reports
`NEON vs scalar`; this is not an x86 fallback executable. Maximum delta
output/state differences are `5.96e-8` in these fixtures.

Compile command and logs:
`/tmp/bitnet-goal-neon-ssm-command.json`,
`/tmp/bitnet-goal-neon-ssm-build.log`, and
`/tmp/bitnet-goal-neon-ssm-test.log`. The executable is
`/tmp/bitnet-goal-neon-test-ssm`. Baseline NEON quant and transformer suites
also pass under Cortex-A53 emulation, built with `-Werror`; commands and
logs are in `/tmp/bitnet-goal-neon-unit-build/`. The separate
`armv8.2-a+dotprod` quant build also passes under QEMU with SDOT enabled;
all three cross-builds use `-Werror`.

The baseline build exposed unconditional SDOT intrinsics in nine quant
implementations. A widening integer NEON helper now provides identical
four-byte dot products and wrapping accumulation without requiring SDOT;
SDOT targets retain their hardware intrinsics. Synthetic tests cover all
lane broadcasts, signed-byte extremes, and accumulator overflow. The
activation Q8 reciprocal now uses `127 / amax`, preserving the existing
reference tie-rounding assertion. Baseline Q8 prefill now uses a four-token float-input NEON tile,
reusing each weight block while retaining decode’s per-block reduction
order. Exact tests cover 1–9 tokens, partial tiles, split row ranges,
signed-byte extremes, and output guards. These are functional checks;
no native ARM throughput improvement is claimed. The resulting source also
passes a clean CUDA-enabled native build without warnings, the complete
standard test target, and the AVX2 quant target; logs are
`/tmp/bitnet-goal-neon-q8-clean.log` and
`/tmp/bitnet-goal-neon-q8-native-tests.log`. Q4_K prepared-layout policy tests now account
for the layout being x86-only. Emulation establishes
functional coverage for the tested operators; native ARM performance and
whole-model qualification remain open.

## CUDA GQA6 cached-prefix attention qualification (2026-11-05)

`test/reference_cuda_prefix_attention.c` builds standalone GGML CUDA
flash-attention graphs with FP32 queries and FP16 K/V. It links only the
independent llama.cpp libraries at `df18bee11e96557b07aa7b12db0976f3c7ac9165`.
As a control, GQA16 with nine queries and a 32-token prefix reproduces the
existing `ee02ff3821594d63` golden exactly; original goldens are unchanged.

The new GQA6 fixtures use four KV heads and head size 256. Thirty-two cases
cover 9/33 queries, prefix positions 32/64/512/1024, two input magnitudes,
two attention scales, and unrestricted/257-token windows. The opt-in runtime
matches all 4,128,768 output float words by hash. Tests also cover padded
cache strides, invalid offsets/shapes, host output guards, input/output
aliasing, and unchanged cached prefix data.

Enable only for qualification with `BN_CUDA_ENABLE_GQA6_PREFIX=1`; default
eligibility remains unchanged. This qualifies the prepared attention
primitive, not dense-layer prefix fusion or whole-model execution. Those
integration and performance gates remain open.

The opt-in dense backend now also supports a borrowed Q/K pair at a nonzero
position using validated device cache spans and history-sized attention
scratch. A composition test passes exactly against qualified prepared
attention followed by the public output projection and residual. The test
uses exact +/-1 inputs, identity projections, nonzero FP16 history, padded
strides, unchanged cache checks, and output guards. F32 projection on this
shape uses the existing TF32 route; assuming an identity projection preserves
arbitrary FP32 attention bits initially failed, so the composition test
explicitly includes the public projection. No independent golden changed.
Focused success log: `/tmp/bitnet-goal-dense-prefix-focused.log`.

Transformer orchestration remains unchanged: its generic dense batch path
writes new K/V to host, and a request entering with an existing prefix must
establish device-cache coherence before this new backend route can be used.
Removing only the position guard would risk stale-prefix reads. Whole-model
integration must preserve that cache ownership contract.

An experimental integration now uses the existing within-request microbatch
coherence contract. `BN_CUDA_ENABLE_DENSE_PREFIX_CHAIN=1`, together with both
primitive opt-ins above, admits dense GQA6 attention only after all layer
resources and prefix spans pass preflight. Each completed block's host KV is
uploaded before the following block; public resumed requests retain their
existing path. No default policy changes.

Qwen3.8 27B at 1,024 input tokens passes 16/16 independent initial/resumed
decode checks. Profiles prove two 512-token blocks each execute all 64 layers
through the chain. Independent 513-token (singleton tail) and 1,057-token
(33-token tail) cases also pass 16/16. **The 521-token case fails one initial
phase decode token after its nine-token cached tail**, with candidate logit
gap `0.0216178894`. Both the existing-path control and an unchunked 521-token
resident-chain control pass. Thus this integration is not qualified for
activation by default; short cached-tail numerical behavior remains work.
The 32 primitive flash-attention goldens are unchanged and still pass. An additional
oracle 512+9 schedule changes four initial-phase generated IDs relative to
the oracle’s single 521-token batch. Replaying against that matched-schedule
oracle still reports two teacher-forced mismatches (initial step 3 and resumed
step 1); changing the oracle schedule does not resolve this failure. Both
reference artifacts are retained separately. The existing runtime
with external chunk size 512 also fails initial step 3 against the oracle’s
512+9 schedule (15/16 overall); the new resident path additionally misses
resumed step 1 (14/16). Thus there is both a broader chunk-schedule numerical
gap and an additional resident-path discrepancy to resolve.

Artifacts and frozen binaries: `/tmp/bitnet-goal-dense-prefix-chain-matrix/`;
the initial 1,024-token log is
`/tmp/bitnet-goal-dense-prefix-chain-1024.log`. The initial 2,048-token harness
attempt stopped before model execution because the corpus contained only
1,395 tokens; with a repeated frozen corpus, the 2,048-token case passes
16/16 and profiles four resident 512-token blocks.
The backend matrix caught a direct expert-count access in orchestration;
using the existing MoE route policy restored the module-boundary check.


Commands, individual oracle logs, and SHA256 provenance are saved under
`/tmp/bitnet-goal-gqa6-prefix-oracle/`. Focused hardware test:
`./test_cuda_backend --gqa6-prefix-reference`; its successful log is
`/tmp/bitnet-goal-gqa6-prefix-focused.log`. Fixtures are in
`test/cuda_gqa6_prefix_attention_reference.h` and are included in the full
CUDA test suite.

## Experimental non-flash cached-prefix GEMM (2026-11-05)

The nine-token tail failure is being investigated without changing reference
IDs. Disabling SSM scan, stacked SSM prefill, or fused SSM gate/up leaves the
same initial mismatch in the frozen resident-path candidate. The cached
attention path was using the independently qualified *flash* primitive
while this whole-model reference requests non-flash attention.

The existing GEMM attention implementation now supports rectangular
`queries x (prefix + queries)` score/probability matrices and prefix-offset
causal/window masks. Default zero-prefix callers preserve their dimensions
and policies. The experimental dense-chain flag selects GEMM for a non-flash
cached prefix. The initial implementation makes the original 521-token
whole-model replay pass 16/16, but is **not qualified**: its new independent
non-flash primitive test fails the first exact hash.

`reference_cuda_prefix_attention.c` now accepts an optional final `flash`
argument (default 1, preserving the original generator). Argument 0 builds
separate GGML score/softmax/value graphs. Thirty-two independent fixtures are
stored in `test/cuda_gqa6_prefix_gemm_reference.h`; they are distinct from the
unchanged flash goldens. `GGML_REFERENCE_DUMP_DIR` optionally exports score,
probability, and output tensors for diagnosis. Focused runtime command:
`./test_cuda_backend --gqa6-prefix-gemm-reference`.

Stage comparison on the first fixture found zero exact score words, maximum
absolute error `0.01290369`, RMS `0.0043252097`. Explicit pedantic FP32
accumulation left the isolated scores and failed output hash unchanged, so
TF32 inheritance alone does not explain this fixture. An independent
reconstruction showed runtime scores match FP32 queries (maximum error
`1.91e-6`), while oracle scores match FP16-rounded queries (maximum error
`1.91e-6`). Runtime versus oracle differs by about `7.89e-4` on the inspected
first query. The prefix score GEMM now consumes FP16 queries and cached FP16
keys with FP32 accumulation. That version still passes the original
521-token model replay (16/16), but changes the first failed primitive hash
to `da92d3a21fd98325`, rather than matching `43a33a7d36903a37`. The original
flash suite and dense prefix integration continue to pass.

A subsequent 256-key-padded candidate still fails the first primitive hash
(`ba7ec74ba01fa325`) and regresses the original 521-token model replay to
15/16. That candidate was reverted; artifacts remain under
`/tmp/bitnet-goal-prefix-gemm-padded-*`. The non-flash oracle currently pads
its key tensor to 256 while the runtime primitive uses only the logical key
span. Matching the oracle/model tensor geometry and reduction semantics
remains unresolved. No hashes or token references have been relaxed or
replaced. The retained query-conversion candidate passes the model replay
but is still unqualified against the new primitive test. Do not infer
completion from either 16/16 model result. Artifacts:
`/tmp/bitnet-goal-prefix-gemm-*` and
`/tmp/bitnet-goal-gqa6-prefix-gemm-oracle/`.

## Prefix value accumulation follow-up (2026-11-05)

The independent non-flash graph exposes a second precision difference:
runtime PV output was entirely FP16-rounded, whereas only 0.0524% of
reference output words were exactly representable as FP16. Reconstructing
PV from reference probabilities and values confirms FP16 probabilities with
FP32 accumulation/output (first-query maximum reconstruction error
`7.45e-9`; retaining FP32 probabilities gives `1.09e-5`).

The experimental prefix GEMM now uses FP16 Q/K for scores, FP16 probability
and value operands for PV, and FP32 accumulation/output. Probability and
output element sizes are independently specified in the batched pointer
setup. The non-batched prefix route follows the same type contract.
Zero-prefix GEMM behavior remains unchanged.

This version retains 16/16 on the original 521-token model replay. On the
first independent fixture, exact output words rise to 30.7002%, maximum
error is `2.2279099e-5`, and RMS is `1.2299558e-6`; **the exact hash still
fails** (`dcf46969ba84bdc0` versus `43a33a7d36903a37`). Converted-query score
comparison gives maximum error `4.77e-6`, RMS `1.32e-6`. These checks remain
strict; no golden or threshold has changed.

The pinned GGML `mmf.cuh` kernel distributes dot-product chunks among warps
and combines warp accumulators in order. cuBLAS does not guarantee that
same arithmetic order. Reproducing that reference tiled reduction for the
prefix score/value kernels remains the next numerical task. Actual llama
KV sizing also rounds to at least 256 keys, capped by cache capacity
(`llama_kv_cache::get_n_kv`), so padded geometry must be treated explicitly.
Stage logs and binaries are under `/tmp/bitnet-goal-prefix-gemm-f32pv-*`.
`BN_TEST_PREFIX_OUTPUT_PATH` optionally saves the first primitive result
from the test harness for comparison.

## Reference-dispatch prefix attention follow-up (2026-11-05)

The short-prefix score/value implementation now uses 16x8 F16-input,
F32-accumulator MMA tiles. Independent 64-element dot-product chunks are
combined in the reference MMF warp order. All first eight nine-query fixtures
pass exact output hashes; the first case also matches every score float word.

The 33-query oracle trace establishes the larger-batch dispatch directly:
`nvjet_sm120_sss_tf32_mma_64x24x64_4_16x24x64_tmaAB_alignCD4_ptrBatch_bz_TNNN`
for scores and
`nvjet_sm120_hhh_mma_64x40x64_7_16x40x64_tmaAB_alignCD4_ptrBatch_bz_TNNN`
for values. Artifacts: `/tmp/bitnet-goal-prefix-ref33-trace.nsys-rep` and
`/tmp/bitnet-goal-prefix-ref33-kernels.csv`. Earlier inference that this
larger-batch score graph directly used FP16 queries was incomplete: TF32 RNE
and FP16 happen to round these normal-range input values to the same mantissa
precision, but their kernels combine products differently.

The experimental larger-batch path therefore requests TF32 score GEMM and
packs V into the reference's transposed head-major layout. Its FP16 value
GEMM writes head-major FP16 output, then unpacks to the runtime's token-major
FP32 interface. Prefix scratch explicitly accommodates the padded key span;
only backend scratch padding is zeroed, and cache reads remain bounded by
the validated logical history. With this dispatch/layout, the first sixteen
fixtures pass exact hashes, including all nine- and 33-query short-prefix
cases. Original independent goldens remain unchanged.

The next failing case is nine queries over a 512-token prefix. Its output
is 98.6111% exact; FP16 probabilities are 99.9982% exact. Remaining observed
probability differences are at most `2.33e-10`. The reference softmax keeps a
lane-specific XOR reduction result, while the prior runtime broadcast lane
zero's denominator. The follow-up prefix softmax now uses lane-specific XOR
sums, the reference block width, and approximate reciprocal. **All 32
non-flash prefix fixtures now pass exact hashes (4,128,768 output words).**
The batched and non-batched larger-prefix routes use explicit TF32 scoring,
transposed head-major V, and head-major FP16 output before token-major
unpacking. Zero-prefix/default policy remains unchanged.

The frozen candidate also passes original 521-token, 513-token, 1,057-token,
and 2,048-token initial/resumed model replays, each 16/16. The separate
matched 512+9 oracle reference still misses one initial-phase token (15/16),
so complete chunk-schedule qualification remains open. Frozen binaries and
SHA256 manifest are in `/tmp/bitnet-goal-prefix-qualified-candidate/`;
model logs are in `/tmp/bitnet-goal-prefix-qualified-model/`. Full CUDA and
final native regressions passed; the clean CUDA-enabled build has zero
warnings, standard/AVX2 suites pass, and diagnostic isolation passes. The
original five F16 replay schedules plus five full-prompt/tail schedules now
pass 160/160 IDs in total. With the same opt-in flags, F32 KV CUDA short
checks pass 192/192 and the original frozen 4,096-byte long prompt passes
24/24 (SHA256 `5ba8da8fc2ff5e856a2e5d689785a80fc4691a5352d11290bdea864aa35f37f5`).
An initial long check used the complete document and stopped at EOS after
one ID in both runtimes; that 1/1 result is retained but is not counted as
the 24-token long qualification.

The explicit `BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM=1` diagnostic mode still fails
the new exact larger-batch primitive hash (case 8); that alternate cuBLAS
shape is not covered by the default-route qualification. Logs:
`/tmp/bitnet-goal-prefix-dispatch-*`,
`/tmp/bitnet-goal-prefix-wide-*`, and
`/tmp/bitnet-goal-prefix-softmax-*`.


## CUDA nonzero-state GDN reduction qualification (2026-11-05)

The matched 512+9 failure reported above is resolved in the opt-in dense
prefix candidate. A tail-only layer trace located the first material drift
in SSM layer 1, before attention layer 3. Its QKV and gate projections were
bit-exact, while the recurrent output already differed. The pinned GGML
CUDA GDN kernel assigns eight lanes to each state column; the runtime used
32. Their FP32 feedback reductions differ across timesteps.

The prefill kernel now supports both lane layouts. The experimental dense
prefix policy selects eight lanes (four columns per warp), with RMS-scaled
normalization required in fused model prefill. Flags-unset execution retains
the existing layout. No independent reference values were replaced.
`test/reference_cuda_gdn_state.c` generates nine independent nonzero-state
fixtures through GGML alone at reference revision
`df18bee11e96557b07aa7b12db0976f3c7ac9165`: 1/9/33 timesteps and key/value
head shapes 1/1, 2/6, and 16/48. All 2,703,360 final-state words match exact
hashes. A legacy-layout negative control fails the nine-timestep fixture,
confirming that the test detects the reduction-order defect.

Frozen binaries and hashes are in `/tmp/bitnet-goal-gdn-order-candidate/`.
The eleven model cases in `/tmp/bitnet-goal-gdn-order-model/status.json`
pass 176/176 initial/resumed token IDs. These include the matched 512+9
oracle and the original unbatched 521-token oracle as separate workloads;
the latter explicitly uses `BN_GPU_PREFILL_FULL_PROMPT=1`. The independent
histories differ and neither reference is substituted for the other.
All cases enable `BN_CUDA_ENABLE_PREFILL_QK_PAIR=1`,
`BN_CUDA_ENABLE_GQA6_PREFIX=1`, and
`BN_CUDA_ENABLE_DENSE_PREFIX_CHAIN=1`.

The full CUDA hardware suite and diagnostic isolation pass. The final clean
CUDA build has zero warnings, and standard plus AVX2 suites pass. Evidence:
`/tmp/bitnet-goal-gdn-order-full-cuda.log`,
`/tmp/bitnet-goal-gdn-order-final-clean.log`,
`/tmp/bitnet-goal-gdn-order-final-native-tests.log`, and
`/tmp/bitnet-goal-gdn-reference-order-negative-control.log`.
The same frozen candidate also passes F32 KV short checks (192/192 IDs)
and the frozen 4,096-byte long prompt (24/24 IDs), recorded under
`/tmp/bitnet-goal-gdn-order-final-model/`.
This closes the observed matched-schedule defect for this candidate;
it does not establish vLLM parity, native ARM model qualification, or exact
behavior of the alternate non-batched cuBLAS diagnostic route.

The same-binary warm prefill check for the corrected candidate records:

| Prompt tokens | Default ms | Q/K pair ms | Resident chain ms | Chain latency reduction |
| --- | --- | --- | --- | --- |
| 33 | 176.399 | 165.865 | 162.827 | 7.69% |
| 256 | 419.547 | 326.411 | 316.121 | 24.65% |
| 1024 | 2042.649 | 1236.043 | 1207.933 | 40.86% |

These are three-run medians on GPU 0, with the same Q4_K_XL checkpoint,
F32 activations, F16 KV, non-flash attention, final logits, eight CPU threads,
and session reset excluded from timing. Each process warms up before timing.
They are preliminary internal comparisons without randomized order or
confidence intervals; no vLLM parity is implied. At 1,024 tokens the chain
includes two 512-token microbatches. The corpus, binary hash, commands,
flags, logs, and summary are preserved in
`/tmp/bitnet-goal-gdn-order-prefill-performance/`.


## Scoped 1,024-token resident-prefix profile (2026-11-05)

A same-binary scoped capture compares the Q/K pair alone with the corrected
resident chain. Scope is one warm prefill with final logits, F32 activations,
F16 KV, non-flash attention, eight CPU threads, and the same frozen corpus.
CUDA profiler API boundaries exclude model upload, session reset, and warmup.

| Measurement | Q/K pair | Resident chain |
| --- | --- | --- |
| Host-to-device bytes | 738,259,544 | 121,712,464 |
| Device-to-host bytes | 785,348,612 | 135,251,972 |
| HtoD / DtoH operations | 87 / 66 | 111 / 68 |
| GPU copy time, HtoD + DtoH | 39.335 ms | 5.959 ms |
| Sum of kernel durations | 1,197.432 ms | 1,205.309 ms |
| Kernel instances | 1,790 | 3,594 |

Host/device bytes fall 83.13%, while the chain uses two 512-token
microbatches and approximately doubles launches. Kernel sums do not fall:
the leading K-quant tensor kernel accounts for 361.665 ms (30.0%), the
leading IQ4 tensor kernel for 307.917 ms (25.5%), and the ordered Q3_K
kernel for 112.696 ms (9.3%) across six launches. These kernels are the
next measured compute targets. The trace contains no kernel names matching
`gemv`; this observation alone does not prove absence of all host fallbacks.

Synchronous `cudaMemcpy` API duration is about 1.20 seconds in the chain,
but includes waiting for earlier kernels. Its duration must not be reported
as transfer-only cost or added to GPU kernel time. Actual copy engine
durations above come from CUPTI activities. Device-to-device copies are
reported separately in the saved summary and excluded from host/device bytes.

A separate host-profile run confirms two 512-token batch records per run,
each with `chain_layers=64`, `decode=0`, and `attn_cpu=0`. Its measured
prefill is 1,192.978 ms; that single instrumented run is diagnostic, not a
replacement for the three-run medians.

Reports, SQLite activities, executable/corpus hashes, commands, flags, and
summary are in `/tmp/bitnet-goal-gdn-order-traces-1024/`. Hardware-counter
profiling with Nsight Compute 2026.3.1 was attempted but rejected with
`ERR_NVGPUCTRPERM`; no occupancy or bandwidth-utilization measurements are
claimed. The older `/usr/bin/ncu` (2022.4.1) also failed CUDA setup and is not
used as performance evidence. These captures diagnose this runtime and do
not establish matched-precision vLLM parity.


## IQ4 block-layout and register-limit experiments (2026-11-05)

Two isolated source candidates tested scheduling changes to the dominant
packed IQ4_XS ordered tensor kernel. Neither is retained. Both used the
resident chain, the same frozen corpus and checkpoint, F32 activations,
F16 KV, final logits, eight CPU threads, and three warm timing runs on GPU 0.
Each comparison below has its own freshly measured frozen baseline.

| Experiment | Tokens | Baseline ms | Candidate ms | Latency change |
| --- | --- | --- | --- | --- |
| Two row tiles / 256 threads instead of four / 512 | 256 | 310.805 | 327.779 | +5.46% |
| Two row tiles / 256 threads instead of four / 512 | 1024 | 1186.644 | 1242.293 | +4.69% |
| Original layout, minimum two resident blocks requested | 256 | 311.763 | 309.070 | -0.86% |
| Original layout, minimum two resident blocks requested | 1024 | 1186.167 | 1183.592 | -0.22% |

The smaller-block candidate regresses both lengths. The requested
higher-occupancy candidate changes compiled resources from 128 registers
and zero stack bytes to 64 registers and 16 stack bytes per thread, retaining
27,136 bytes of static shared memory per block. These are compiler resource
reports, not measured occupancy. Its sub-one-percent timing differences are
insufficient evidence of an improvement in this small fixed-order sample.
Neither candidate proceeds to correctness qualification or production use;
the original CUDA source is restored byte-for-byte.

Source/binary hashes, commands, flags, logs, and summaries are retained in
`/tmp/bitnet-goal-iq4-rowtile2/` and
`/tmp/bitnet-goal-iq4-occupancy2/`. Baseline static resources are recorded in
`/tmp/bitnet-goal-iq4-baseline-resources.log`. The next IQ4 optimization must
address actual tile/data reuse or instruction cost rather than assume that
smaller blocks or a lower register limit improve throughput.

After restoration, the clean CUDA-enabled build has zero warnings and the
standard plus AVX2 suites pass. Logs:
`/tmp/bitnet-goal-iq4-tuning-final-clean.log` and
`/tmp/bitnet-goal-iq4-tuning-final-native-tests.log`. The four user-owned
reference binaries are restored and verified against their pre-clean hashes.


## Flash Next split-file CPU cache qualification (2026-11-05)

The local Qwen3.8 Flash Next Q4_K_XL checkpoint has four shards and GGUF
architecture `qwen4exp`: 48 blocks, dimension 2560, 512 experts, top-10
routing, and expert hidden size 640. The initial CPU reference check passes
174/174 generated IDs across eight prompts with up to 24 tokens each
(early EOS reduces the count). Reference revision is
`df18bee11e96557b07aa7b12db0976f3c7ac9165`. This baseline exposed that the
requested 2 GiB RAM expert cache was inactive: split-file pread keeps the
legacy single-file descriptor negative, and both cache creation and the
pipelined executor checked only that descriptor.

The fix makes pread eligibility recognize valid shard descriptors and puts
the appropriate descriptor on every asynchronous gate/up/down request.
Synchronous fallback reads use the same projection-to-shard resolution.
The existing bounded LRU and execution policy remain responsible for
placement and computation. Synthetic tests read gate/up/down from distinct
files, reject an out-of-range shard, and verify request recovery after an
invalid descriptor. Single-file descriptor resolution remains supported.

The fixed candidate also passes the full eight-prompt reference comparison
(174/174 generated IDs) with the active 2 GiB cache; log:
`/tmp/bitnet-goal-flash-next-sharded-cache/short-reference.log`.

Real-model replay preserves all eight independently checked IDs with cache
disabled, a 64 MiB budget, and a 2 GiB budget. Unlike the baseline, nonzero
budgets now create the cache and report activity:

| Requested slab budget | Actual slab MiB | Slots | Hits | Misses |
| --- | --- | --- | --- | --- |
| 64 MiB | 61 | 16 | 0 | 3360 |
| 2048 MiB | 2045 | 537 | 1067 | 2293 |

These are expert-data slab budgets; metadata, model, activations, and I/O
staging are separate allocations. OS page cache was warm and concurrent
correctness jobs ran, so these runs establish output preservation and active
bounded caching, not cold-disk performance or throughput. Artifacts and
binary/model provenance are in
`/tmp/bitnet-goal-flash-next-cache-correctness/` (baseline) and
`/tmp/bitnet-goal-flash-next-sharded-cache/` (fixed candidate).
The clean CUDA-enabled build has zero warnings, and standard plus AVX2
regressions pass; logs use `/tmp/bitnet-goal-sharded-pread-final-*`.

CUDA Flash Next is **not qualified**: the baseline upload estimates expanded
full expert residency at 318.6 GiB, attempts quant-only full residency, then
fails at layer 0 and falls back to CPU. The comparison harness rejects this
as requested-backend activation failure. That CPU execution must not be
counted as CUDA correctness or offloading evidence. Bounded CUDA expert
placement, long/resumed correctness, and separate cold/warm policy
qualification remain open. Logs:
`/tmp/bitnet-goal-flash-next-cpu-short.log` and
`/tmp/bitnet-goal-flash-next-cuda-baseline.log`.

A separate diagnostic run with `BN_GPU_DISABLE_MOE_ALL_LAYOUT=1` activates
CUDA and passes eight generated IDs against the independent CPU oracle.
Explicit GPU cache budgets of 1024 and 4096 MiB preserve those IDs with
`CUDA weights uploaded` present and no reported fallback marker. The 1024
MiB cache has 349 slots and reports 0 hits / 4920 misses; 4096 MiB reports
1907 hits / 3013 misses (38.8%). This establishes cache reuse in the larger
budget for this smoke workload. It does not qualify all prompts, cached
prefixes, concurrent requests, cold storage, or performance. Expert misses
still use pread staging; warm OS pages are not evidence of a dedicated
bounded RAM tier feeding GPU misses. The default full-layout attempt remains
unfixed. Commands, outputs, and assertions are in
`cuda-bounded-result.json` and `cuda-bounded4096-result.json` under the fixed
candidate artifact directory.


## Pread CUDA placement and Flash Next batch failure (2026-11-05)

Full-expert upload requires mapped projection spans. The placement decision
now checks that prerequisite before selecting whole-model expert residency;
pread models retain GPU base weights and use the existing bounded expert
cache. A synthetic regression fails before this correction at the full
expert upload, then passes for both single-file and split-file pread.
Positive controls confirm that single-file and split-file mappings still
create full-expert handles. Standard/AVX2 regressions and the clean
warning-free CUDA build pass. Logs use
`/tmp/bitnet-goal-pread-placement-{negative,focused,final-*}`.
The comparison harness now forwards `--gpu-cache-mb` so this workload can
pin the expert-cache budget explicitly.

With no `BN_GPU_DISABLE_MOE_ALL_LAYOUT` override, the corrected Flash Next
candidate activates CUDA and preserves the eight independently checked IDs
at GPU cache budgets of 1024 and 4096 MiB. The 4096 MiB run records 1907 hits
and 3013 misses. This fixes the observed upload/activation failure; **it does
not qualify Flash Next batched CUDA correctness**.

Broader gates expose a failure beyond the five-token France prompt. The
short matrix matches only 24/174 reference IDs (and has one token-count
mismatch); the frozen 1047-token long input matches 0/24. The initial short
attempt also encountered oracle exit 139; its standalone retry succeeds,
and the subsequent complete short comparison fails on runtime output,
so that first infrastructure failure is not used as correctness evidence.

The same long prompt passes 24/24 on CPU against separately saved llama.cpp
IDs. An 11-token diagnostic prompt produces NaN logits in CUDA batch mode,
while tokenwise CUDA starts with the expected token. A full per-position
stage trace locates the first NaN at `bitnet_moe_routed_sum`, layer 1,
position 1, before later SSM convolution/normalization propagates the error.
A last-token-only trace initially showed the downstream SSM failure; it was
insufficient to locate the origin. Setting
`BN_CUDA_DISABLE_MOE_FFN_BATCH=1` produces finite logits and the expected
first token on that diagnostic while the rest of prefill remains batched.
This isolates the split-expert callback; the switch is not retained as a
correctness or performance solution. The batch execution remains open and no
reference values are replaced. Artifacts, commands, frozen binaries,
reference IDs, and traces are under
`/tmp/bitnet-goal-flash-next-default-cuda/`.


## Split-expert stream and stacked-geometry corrections (2026-11-05)

The Flash Next NaN failure exposed a stream-ordering defect in
`cuda_moe_ffn_batch`: gather/activation/scatter used the default stream,
while quantized matmuls inherited a nonblocking stream selected by prior
graph execution. The callback now enters an ordered default-stream scope
and restores the prior execution/cuBLAS stream on exit. It remains batched.
An analytic Q4_K regression primes graph execution without host readback,
then checks 9/17/33-token FFNs (151,040 output words plus guards). Before the
fix, the 17-token case returns zero where `silu(0.5) * 0.5` is required.
Its 1e-3 tolerance accommodates the quantized-input arithmetic; it is not an
independent GGML hash fixture.

An additional callback comparison reuses the existing independent GGML
FFN goldens without replacing any values. It checks all 46 plain-FFN
fixtures (382,464 words), including stacked and separate storage, through
the split-expert interface after graph execution. This detects a second
defect at fixture 203: stacked Q5_K gate/up, dim 5120, hidden 17408, 11
tokens. The callback used the combined row count for reference partitioning.
It now uses separate backend row views and planar activation for the
reference-quant path, matching the already-qualified standalone FFN path.
The original expected hash `07f45353fdc58d5d` is preserved; the unfixed
callback produced `6ea6e3b6557c83da`.

Both focused regressions, the full CUDA hardware suite, standard/AVX2
regressions, CUDA diagnostic isolation, and the clean
warning-free CUDA build pass. Frozen binaries/source hashes are in
`/tmp/bitnet-goal-split-expert-geometry-candidate/`; full regression logs use
`/tmp/bitnet-goal-split-geometry-*`. The checked worktree CUDA/test sources
match that manifest exactly. Failure evidence is in
`/tmp/bitnet-goal-split-expert-stream-negative.log` and
`/tmp/bitnet-goal-split-expert-geometry-negative.log`.

**Flash Next model-level CUDA correctness remains open.** The 11-token
prompt no longer produces NaNs, but selects 77296 rather than 557. An
independent fully offloaded CUDA oracle confirms 557, both on one GPU and
split by layer over two GPUs (49/49 layers offloaded). The single-GPU run
uses the same checkpoint, F32 KV, non-flash attention, and no precision
substitution; log `/tmp/bitnet-goal-split-expert-reference-single-cuda.log`.
Disabling shared-expert fusion also disables the separate GPU shared FFN,
so its matching first token does not isolate a shared-kernel defect.

A private replay under `/tmp/bitnet-goal-split-expert-shared-replay/`
compares the captured eleven layer-0 inputs and original shared weights
against a direct, independently linked GGML CUDA graph. All 28,160 down
projection values and eleven gate logits match exactly. Public dense and
split-expert callbacks also agree. The first model drift instead occurs
before MoE, in the hyperconnection mixer.

The hyperconnection prefill code forced Q8 projections through CPU reference
arithmetic even with CUDA active. Using the existing backend-aware reference
policy makes the first HC gate projection match the CUDA oracle exactly.
The year prompt then matches the first five of eight generated IDs, but
still fails full continuation qualification. Candidate and comparison are
in `/tmp/bitnet-goal-hc-cuda-candidate/`.

The remaining HC mean drift is consistent with a separate arithmetic
contract: the pinned CUDA oracle enables fused DSV4 HC pre-gate, which
accumulates products with FMA; the existing runtime fixture describes an
unfused graph. A new independent generator,
`test/reference_cuda_hc_fused_mean.c`, records the fused contract without
replacing the unfused goldens. The experimental policy
`BN_CUDA_ENABLE_FUSED_HC_MEAN=1` remains off by default. Its batch callback
and graph operation pass 72 independent cases and 847,872 exact output-word
comparisons, with guard words checked. The legacy kernel fails the first
new fixture, as expected; its existing unfused goldens remain unchanged.
The fused model trace matches the oracle's sampled layer-0 HC mixed
statistics and values, but still generates only five of eight matching IDs.
The next drift is in SSM QKV, whose projection selection also forced CPU
arithmetic under CUDA. The backend-aware SSM projection change then matches all eight diagnostic
token IDs. The 1,047-token long-prompt comparison still fails at generated token six
(prefix 5/24). The eight-prompt short suite improves to 159/179 token-prefix
matches, with all eight first tokens matching and no token-count mismatch,
but still fails strict parity. Clean CUDA-enabled compilation has no warnings;
standard, AVX512 quant, full CUDA operator, and CUDA diagnostic suites pass.
Artifacts use `/tmp/bitnet-goal-hc-fused-*` and
`/tmp/bitnet-goal-hc-ssm-projection-*`. No Flash Next CUDA model correctness
or vLLM parity claim follows from these operator checks.

### GDN decode reduction qualification

`test/reference_cuda_gdn_decode.c` independently evaluates the pinned
GGML CUDA gated-delta graph with decimal-valued Q/K/V and nonzero incoming
state. Each hash includes all token outputs and the complete final state.
These inputs expose reduction-order differences even on the first token;
the older power-of-two fixture is retained unchanged.

The legacy decode kernel fails the first new case (hash
`d8957305576e9320`, reference `e7341587bdf784ca`). The opt-in
`BN_CUDA_ENABLE_GDN_DECODE_ORDER=1` uses four columns per warp and eight
lanes per column, matching the pinned oracle. It passes 18 plain/fused
alpha-beta cases and 5,603,840 exact output/state word comparisons, with
host and device guard words checked, including a nonzero state offset.
It covers the standalone graph delta and fused
alpha-beta/delta dispatch; the same templated kernel is used by the serial
SSM prefill fallback. Default decode retains its previous reduction order.

Enabling only the previously qualified prefill order leaves the Flash Next
short comparison at 159/179 token-prefix matches. The combined prefill/decode-order short comparison fails at 156/179
token-prefix matches, with all eight first tokens matching and one
token-count mismatch (192 runtime IDs versus 179 oracle IDs). The HTTP
continuation is restored, while other continuations still diverge.
This operator qualification does not qualify the full model or justify
promoting either arithmetic flag to default. The long comparison remains
at a 5/24 token prefix, with no token-count mismatch. Clean CUDA compilation
is warning-free; standard tests, AVX512 runtime/checks, and CUDA diagnostics
pass. Artifacts use
`/tmp/bitnet-goal-gdn-decode-*` and
`/tmp/bitnet-goal-flash-gdn4-{prefill,decode}-short.log`.
The full CUDA operator suite also passes. A subsequent model trace matches
sampled layer-0 HC mixed, QKV, Z, alpha and beta values/statistics, then
first differs at delta. Inspection finds a separate architecture-only guard
still forcing CPU delta/gate in prefill despite CUDA projection selection.
The prefill code now uses its backend-aware reference decision for that
callback dispatch as well; model qualification of this handoff uses
`/tmp/bitnet-goal-flash-gdn-backend-*` artifacts. The handoff makes sampled
layer-0 SSM gate and output statistics/values match the independent oracle.
The next sampled mismatch is HC post-combine, which still forces host
arithmetic; the pinned oracle enables fused DSV4 HC post. This is an
investigation target, not yet a proven post-combine fix. Whole-model parity
still fails: short 138/179 prefix matches (one count mismatch), long 5/24.
The latest incremental build is warning-free and standard tests pass.

### Fused HC post-combine qualification

`test/reference_cuda_hc_fused_post.c` evaluates the complete pinned GGML
CUDA scale/sigmoid/scale/DSV4-HC-post graph with deterministic rational
inputs. The new fixture preserves all existing unfused combine goldens.
The legacy path fails case zero (`d5a686808bb98ebe` versus oracle
`f75674c8ff57f8de`), confirming that the fixture detects this distinction.

The experimental `BN_CUDA_ENABLE_FUSED_HC_POST=1` selects an explicit FMA
for residual accumulation in both the host batch callback and public graph
operation. The prefill combine dispatcher now uses the existing
backend-aware reference policy, allowing CUDA to invoke this callback.
CPU and Metal retain host reference arithmetic. All 72 independent cases
pass, covering 5,087,232 exact output-word comparisons across the batch
callback and graph/capture/replay paths, with host guard words intact.
The independent generator reproduces all 72 saved hashes. Qualification artifacts
use `/tmp/bitnet-goal-hc-post-*`; model-level parity remains a separate gate.

**Observer limitation discovered during model qualification:** the broad
intermediate-tensor observer changes the independent model's generated
sequence (first ID 77296), while the same pinned probe with `--no-observer`
produces the normal CLI sequence beginning 557,39338,440,449. Consequently,
the preceding sampled stage matches establish agreement with the observed
graph only; they do not establish agreement with normal CLI inference.
Wildcard binary dumps may also be overwritten by reshaped aliases. Select
an exact tensor name and occurrence when comparing binary values.

The probe now supports `LLAMA_PROBE_OBSERVE_NAME` to request only one named
boundary from the backend. Any such run must first reproduce the complete
no-observer generated sequence before its tensors can guide model fixes.
Normal CLI qualification for the combined HC-post candidate still fails:
short 117/179 token-prefix matches, one count mismatch; long 5/24. Clean
CUDA compilation, standard tests, AVX512 checks, full CUDA operators and
CUDA diagnostics pass. Fused HC post stays off by default.

The restricted observer for `hc_mixed-0`, occurrence 2, preserves all eight
normal/no-observer generated IDs. Its layer-0 FFN input at prompt position
10 matches all 2,560 runtime words exactly. This boundary is therefore
qualified for the next FFN investigation, unlike the broad observer trace.
Evidence: `/tmp/bitnet-goal-hc-post-candidate/minimal-observer-qualification.json`;
probe `/tmp/bitnet-goal-minimal-boundary-reference-probe` compiled against
pinned primary libraries with C++17 and `-Wall -Wextra -Werror`.

### Routed Q4_K/Q5_1 prefill arithmetic gap

Restricted observers of `ffn_out-0`, `ffn_moe_out-0` and
`ffn_shexp_gated-0` preserve all eight normal-oracle IDs for the year
prompt. Runtime layer-0 FFN output differs from the qualified normal
boundary on all 2,560 words (maximum absolute difference 0.0007830523).
A private repeated-input replay of the isolated shared branch matches the
qualified shared boundary on all 2,560 words. This isolates a routed
arithmetic issue for further investigation; it does not prove integrated
shared-state correctness under every expert schedule.

The checkpoint's layer-0 expert gate/up are Q4_K; down is Q5_1. Existing
routed Q4_K/Q5_1 goldens cover NT=1,2,4,8 only. The pinned CUDA oracle
switches indexed projections to MMQ above eight tokens, while the offloaded
combined callback invokes dense projections on smaller gathered expert
groups. Its descriptor also omits total-expert/index geometry. Separately,
the ordered routed implementation selects Q5_1 MMVQ unconditionally,
including batches above the oracle's MMVQ boundary.

`test/reference_cuda_q451_prefill.c` now independently generates 16 cases
at NT=9,11,17,33 for both synthetic and model-sized projections. It uses
GGML indexed Q4_K gate/up, split SwiGLU and indexed Q5_1 down, then the
existing separate weighted-add reduction contract. Expected values are
preserved in `test/cuda_q451_prefill_reference.h` as pending qualification
targets; the passing legacy goldens remain unchanged. A private copy of
the existing runtime reference runner fails the first new NT=9 case.
All 16 new runtime hashes disagree (394,240 output words covered); the
independent oracle reproduces all hashes on the second GPU. The private
inventory runner disables hash asserts and retains an inherited PASSED
trailer, which is not a passing result; its qualification JSON records
16 failures. The original strict negative runner aborts at case zero.
The NT=9 invalid-route negative check also exposes a validation gap in
the larger-batch fallback. These are reproducible kernel/API gaps,
independent of model observer effects. Logs: `/tmp/bitnet-goal-q451-prefill-negative/`.

A private full-batch padding diagnostic changes but does not fix the
model FFN output (maximum absolute difference 0.0007269681). Padding was
not added to the runtime. The next implementation needs routed MMQ
arithmetic, including affine Q5_1 down and full geometry, while preserving
bounded expert residency and grouped execution. Artifacts:
`/tmp/bitnet-goal-ffn-branch-replay/`,
`/tmp/bitnet-goal-moe-group-padding-diagnostic/`, and
`/tmp/bitnet-goal-q451-prefill-oracle*`.

### Q4_K/Q5_1 ordered MMQ implementation

The ordered routed callback now selects reference MMQ arithmetic above
NT=8 for both Q4_K gate/up and Q5_1 down. The affine down capability is
declared in the quant registry and implemented in the format-local
`src/quant/q5_1_cuda.cuh`; the backend owns scratch and launch geometry.
The kernel preserves global expert/rank split-K partitions, reciprocal
activation scales, original activation sums and two FMA updates per block.
Existing NT<=8 MMVQ behavior remains covered by unchanged goldens.

The first stage capture matches the independent oracle on all 36,864
SwiGLU values and all 18,432 down-projection values. The original new
fixtures used separate weighted multiply/add, while the standalone
callback's existing contract uses fused accumulation. Both contracts are
now generated independently from the same oracle down projections; all
16 original separate hashes remain intact, with new fused counterparts.
With equal contracts, the legacy routed code still fails (case-zero hash
39fc3d28f27187b1 versus fused oracle c759c383be67417b). The corrected code
passes all 16 callback cases, 1,576,960 exact output-word comparisons with
alias/repeat checks, and invalid-route negative checks.

This qualifies the ordered callback, not the bounded offload model path.
The split-expert combined callback still invokes dense projection
arithmetic and accumulates in expert order. Its existing global expert
count and array index provide geometry; the earlier description that this
information was absent from the API was too broad. The next step is to
use that geometry and retain checkpoint route order without padding or
uploading all expert weights. Kernel timing and model parity remain open.
Artifacts use `/tmp/bitnet-goal-q451-mmq-*`. Clean CUDA compilation is
warning-free; standard, AVX512 syntax/runtime, full CUDA and CUDA diagnostic
suites pass. The four user-owned probe binaries are preserved byte-for-byte.

### Gathered Q4_K/Q5_1 offload path

The expert-batch callback now runs one borrowed Q4_K/Q5_1 expert using
full prompt/expert/rank MMVQ or MMQ geometry, rather than selecting dense
arithmetic from the local group count. It handles both independent and
stacked gate/up buffers, host aliases and invalid-plan rejection. The
quant registry owns format eligibility. A synthetic test uploads and
releases one expert at a time and checks the unchanged independent
references: 32 shapes, 1,914,880 exact output-word comparisons across
stacking and alias modes. No all-expert weight stack is allocated.

For this format combination, CUDA's combined split-expert callback now
defers to the gathered path and existing slot-ordered reduction. The
transformer policy skips its speculative all-active-expert preload, so
cache budgets remain effective without warming every expert before the
per-expert loop. The backend keeps a rejection guard for direct calls.

The actual Flash Next layer-0 FFN output now matches the restricted,
normal-generation-preserving oracle on every one of 2,560 words. The
short model gate still fails: 124/176 token-prefix matches, all eight first
IDs matching, one count mismatch (176 runtime IDs vs 179 oracle IDs).
The year diagnostic matches five of eight IDs. Checkpoint metadata also
shows Q5_K gate/up with Q8_0 down at layer 2, and Q4_K gate/up with Q8_0
down at layers 4,30,46,47. Those gathered arithmetic paths remain open;
qualifying Q4_K/Q5_1 alone does not qualify the model. Full CUDA and clean
standard/AVX512 checks pass. Artifacts use
`/tmp/bitnet-goal-q451-group-*`. The final dispatch-policy build and tests
also pass without warnings, along with CUDA diagnostics and AVX512 runtime
checks. The long-prompt gate still fails at a 5/24 token prefix with equal
token counts. All four user-owned probe binaries remain byte-identical.

### Mixed K-quant / Q8_0 gathered execution

Independent fixtures now cover Q4_K/Q8_0 and Q5_K/Q8_0 at NT=1,8,9,11,33,
including model-sized dimensions, sparse expert groups, stacked gate/up and
host aliases. All 40 cases pass: 2,793,472 exact output-word comparisons.
Q8_0 down uses FP16-scale MMVQ inputs or FP32-scale MMQ inputs according to
the full prompt size. Its borrowed-expert kernel reuses the backend-owned
global MMQ schedule. Quant/backend behavior helpers own format eligibility;
a raw predicate accidentally introduced in transformer policy was caught
by the architecture gate and removed. The fixed gate and clean standard /
AVX512 checks pass without warnings; latest narrow CUDA and diagnostics pass. The final complete CUDA suite
also passes, and all four user-owned probe binaries remain byte-identical.

The year diagnostic now matches 8/8 normal-oracle IDs. Full parity remains
open: short 146/175 token-prefix matches with one count mismatch (175
runtime IDs / 179 oracle IDs); long 5/24 with equal counts. One earlier
short-suite oracle process exited 139. Its log is preserved; the completed
retry uses an immutable private copy of the token tracer, isolating mapped
instrumentation from build cleanup/restoration. The crash cause is not
established. Future comparisons should set `LLAMA_TOKEN_TRACE` to an
immutable private copy when running concurrently with build cleanup.

The next isolated decode issue is Q5_1 expert projection selection. The
current K-quant/native-quant flag helpers produce flags=0 for this format.
A graph replay with those flags differs from the independent CUDA
single-column oracle in all eight cases. This is a contract mismatch:
unflagged legacy matvec consumes floating inputs, while the routed oracle
quantizes activations. Preserve the unflagged contract and introduce the
appropriate explicit selection for routed down projections. Independent
targets are in `test/cuda_q51_matvec_reference.h`, generated by
`test/reference_cuda_q51_matvec.c`; the strict negative replay is under
`/tmp/bitnet-goal-q51-matvec-negative/`. Mixed-prefill artifacts use
`/tmp/bitnet-goal-qk8-*`. This evidence does not qualify decode or the model.

### Q5_1 explicit decode activation selection (2026-11-05)

CUDA routed Q4_K/Q5_1 down projections now explicitly select quantized
activations through the backend policy adapter. Unflagged Q5_1 matvec keeps
its existing floating-input contract; Metal selection is unchanged. The
format-local kernel supports bias and nonzero output offsets.

Eight independent single-column fixtures cover 43,304 exact output words
across direct execution, graph capture/replay, lowered activation requests,
and fused bias/copy. Another 10,826 words preserve saved legacy outputs;
these compatibility values are not independent reference results. Original
independent hashes are unchanged. The complete CUDA suite, clean warning-free
CUDA build, standard tests, and AVX512 checks pass. The four user-owned probe
binaries were restored byte-identically after cleanup.

Model qualification remains incomplete. For the 11-token year prompt, the
first decode layer's combined FFN output differs from the restricted normal
oracle by at most 2.98023224e-8 (RMS 3.70640952e-9; 755/2560 values exact).
The oracle observer preserves all eight normal generated IDs. Despite this
local improvement, the candidate matches only the first five of those eight
IDs, versus eight for the prior candidate. This is evidence for the local
projection correction, not a whole-model correctness or performance claim.
Fresh complete short/long suites for this candidate remain pending.

Artifacts: `/tmp/bitnet-goal-q51-decode-candidate/`,
`/tmp/bitnet-goal-q51-decode-positive.log`,
`/tmp/bitnet-goal-q51-decode-full-cuda.log`,
`/tmp/bitnet-goal-q51-decode-clean-build.log`, and
`/tmp/bitnet-goal-q51-decode-final-tests.log`. Candidate SHA256:
`36dc9bab808e43d81d63ae374126fba36064a72fd7b7ede79325f9cd92a8c8a8`.

The first-decode boundary scan further localizes the remaining discrepancy.
Restricted observers at layers 1, 2, 3, 7, 15, 31, and 47 each preserve the
normal oracle's eight generated IDs. Comparing the first 16 FFN output
values gives maximum differences of 7.5e-9, 3.11e-6, 4.09e-4, 5.76e-4,
7.92e-3, 4.68e-3, and 2.53e-2 respectively. These are sampled boundary
comparisons, not full-tensor error bounds. Layer two's full 2560-value FFN
output has maximum error 1.2435e-5 and RMS 1.9857e-6. Its full FFN input
has maximum error 5.9605e-7 and RMS 1.1569e-7. A CPU float32 quantization
diagnostic finds identical Q8 integer values and FP16 scales for those two
inputs; CUDA fast-division equivalence is not yet verified. The next check
is identical-input Q5_K/Q8_0 projection arithmetic at this boundary, before
attributing the discrepancy to upstream state. Machine-readable evidence:
`decode-boundary-scan.json` and `layer2-input-diagnostic.json` in the
candidate artifact directory above.

### Identical-input layer-two expert isolation (2026-11-05)

Real layer-two experts 63 and 84 were tested on both captured FFN inputs.
Separate Q5_K gate/up projections match pinned GGML CUDA bit-for-bit over
5,120 values. Subsequent reference SwiGLU and Q8_0 down projections match
over another 12,800 values. For these two experts, changing between the two
captured inputs does not change any tested projection output.

The stacked fused Q5_K gate/up path does differ: only 141/640 and 140/640
SwiGLU values match exactly, with maximum errors 1.1921e-7 and 5.9605e-8.
Its current kernels neither accept the reference-SiLU request nor use the
same reduction order as the qualified separate projections. Disabling only
`BN_CUDA_DISABLE_MOE_GATEUP_SPLIT=1` restores all eight normal reference IDs
on the year diagnostic. This localizes a concrete fused-path discrepancy;
it does not qualify the full model or justify removing batching/fusion.
The next implementation should preserve fused execution while honoring
reference activation and projection reduction semantics, with independent
fixtures and existing unflagged outputs preserved.

Private diagnostic sources, raw expert slices, comparison JSON, logs, and
source hashes are under `/tmp/bitnet-goal-layer2-identical/`. The diagnostic
programs compile with warnings treated as errors. No runtime change was
made during this isolation step.

### Reference fused Q5_K correction (2026-11-05)

The Q5_K fused gate/up implementation now honors explicit reference SiLU
and reference activation requests using the independent MMVQ four-warp
fold/XOR order and reference division operation. Both projections and
SwiGLU remain in one fused kernel. Unflagged calls retain their existing
kernels. Routed MoE emission now propagates reference FFN activation to
its fused operation.

Twelve synthetic independent GGML CUDA cases cover 37/640 output rows,
256/512/2560 input columns, and two seeds. All 4,062 unique reference
values match exactly across direct execution and graph capture/replay
(12,186 comparisons), including both request flags and guarded nonzero
copy offsets. The old implementation fails the first case. Sources are
`test/reference_cuda_q5k_fused_gateup.c` and
`test/cuda_q5k_fused_gateup_reference.h`; previous fixtures are unchanged.

This fixes an independently demonstrated fused-operator discrepancy, but
does not fix the Flash Next model continuation failure. The year prompt
still matches only 5/8 reference IDs. Disabling all fused gate/up alone
also leaves 5/8; disabling stacked MoE gate/up gives 8/8. Furthermore,
identical-input split Q5_K projections plus SwiGLU for real experts 63/84
match all 1,280 values exactly. Thus the earlier stacked-path diagnostic
cannot be attributed solely to the fused Q5_K kernel. Remaining work must
check other stacked formats/paths and accumulated model state. Artifacts
and qualification manifest: `/tmp/bitnet-goal-q5k-fused-final/`;
real-weight split comparison:
`/tmp/bitnet-goal-layer2-identical/split-comparison.json`.

The final clean CUDA build is warning-free; standard tests, AVX512 checks,
complete CUDA suite, and diagnostic isolation pass. All four user-owned
probe binaries remain byte-identical after cleanup. The corrected fused
kernel additionally matches all 1,280 real-weight SwiGLU values for experts
63/84. These checks preserve existing fixtures and do not qualify model
continuation. Candidate and clean-build hashes, source hashes, and model
IDs are recorded separately in the qualification manifest.

### First full-attention prefill boundary (2026-11-05)

The year-prompt prefill now has two stronger exact boundaries: layer two's
complete FFN output and layer three's complete attention input each match
all 2,560 independent normal-oracle values bit-for-bit. Layer three is the
first full-attention block. Its attention output already differs before
the FFN: first-16 maximum error 0.000675775575; the gated attention input
to the output projection also differs (first-16 maximum 0.0002320186).
These sampled errors are not full-tensor bounds. Each restricted observer
used for these boundaries preserves all eight normal oracle IDs.

This supersedes the inference that layer-two expert arithmetic was the
primary model failure. Disabling stacked MoE gate/up changes the layer-zero
FFN output and later state; its improved eight-token prefix is insufficient
to establish that it is the correct path. Disabling only fusion does not
improve the prefix. Disabling FP16 native-quant matmul also leaves the
attention error and 5/8 diagnostic prefix unchanged. Next work should
isolate the first full-attention block's Q8_0 Q/K/V projections,
normalization, RoPE, attention, and gate, using the exact input boundary.

Artifacts and observer qualification:
`/tmp/bitnet-goal-q4k-split-investigation/qualification.json`. Some
restricted Qcur/Kcur/Vcur attempts emitted no tensor values; they provide
no projection-correctness evidence. The broad tensor-name inventory is
only used to locate nodes and is excluded from numeric qualification.
No runtime code changed during this isolation step.

### Full-attention precision contract isolation (2026-11-05)

The complete 12,288-value layer-three Q projection matches an independent
Q8_0 GGML CUDA replay exactly. That replay repeats the captured input across
11 columns to retain the normal prefill shape. The normal model's Q/K/V
projection rows independently match this replay over 12,288/512/512 values.
Private raw-tensor observers for Q, K, V, post-RoPE Q, softmax, and pregate
attention each emit real tensors and preserve all eight normal reference IDs.
The runtime's full post-RoPE K differs by at most 1.1325e-6 (228/512 exact).

The major remaining attention difference tracks GPU MMF input precision.
A CPU double diagnostic with TF32 input truncation gives maximum softmax
error 2.9863e-7 and maximum value-product error 4.3958e-7 (RMS 3.0963e-8)
against the raw normal reference tensors. Full-FP32 value products using
those same reference probabilities differ by up to 2.5144e-3. These are
precision-contract diagnostics, not exact independent kernel fixtures.

The CUDA backend already has short MMF256 attention kernels. Its prepared
attention adapter currently admits only the MMA route, while transformer
backend QKV-only preparation is restricted to shared-K/V, ungated plans.
The gated Flash Next plan therefore falls back to CPU attention. The next
change should connect the existing GPU MMF execution through prepared gated
prefill with independent tests, rather than round CPU attention inputs.
Both `--kv16` and `--flash` can restore the eight-token diagnostic while
leaving the observed attention tensor incorrect; neither is qualification.
No runtime code changed during this isolation. Sources, raw captures,
projection replay, observer IDs, and diagnostic measurements are under
`/tmp/bitnet-goal-attention-q8-replay/qualification.json`.

### Prepared gated CUDA MMF attention integration (2026-11-06)

Prepared CUDA attention now admits the existing MMF256 score/value kernels
for FP32 cache, head size 256, position zero, and 4..16 prompt tokens.
Reference-attention CUDA models request this prepared route without requiring
`--flash`. CPU and Metal selection retain their previous behavior. The
adapter retains separate Q/K normalization, partial RoPE, gate extraction,
cache-policy output, and rejection before output writes for unsupported
MMVF shapes. Explicit state preserves the fused versus separate RoPE
contraction order when a reference normalization pass clears weight pointers.

Independent pinned GGML CUDA fixtures qualify 24 numeric cases, including
4/6/8/9/11/16 tokens, GQA4/GQA12, padded gated input, no norm/per-head norm,
partial RoPE, full causal/window-three attention, aliases, and guards.
Attention output and returned F32 keys match all 857,088 unique reference
values across 1,714,176 comparisons. Eight two/three-token cases verify
safe rejection of the distinct MMVF contract. All 16 original targets from
the initial fixture set remain unchanged after expanding boundary coverage.
Existing independent fixtures are unchanged. Sources:
`test/reference_cuda_gated_mmf.c`, `test/cuda_gated_mmf_fixture.h`, and
`test/cuda_gated_mmf_reference.h`.

The complete year-prompt layer-three prefill FFN output now matches all
2,560 normal-oracle values exactly; the previous path differed after this
first full-attention block. This extends the earlier exact layer-two FFN
and layer-three attention-input boundaries. It does not qualify subsequent
layers or continuation. A follow-up restricted normal-oracle comparison
also matches the full layer-four FFN output (2,560/2,560) and the first 16
layer-seven FFN values. The layer-seven sample is not a full-tensor gate.
Both oracle observers preserve the normal eight IDs. Qwen27 preserves
192/192 short and 24/24 long CUDA sampled IDs. Flash Next still fails: short 129/176 prefix matches, with
176 runtime versus 179 oracle IDs and one count-mismatched prompt; long
5/24 with equal counts. The short metric is worse than the older 146/175
checkpoint, despite the corrected independently verified boundary. Keep
this regression visible while locating the next difference; do not switch
to a compensating path based only on an eight-token diagnostic.

The clean CUDA build is warning-free; standard, AVX512, the initial complete
CUDA suite, expanded complete CUDA suite, and diagnostic isolation pass.
Both full-suite logs are retained in the qualification manifest directory. All four user-owned
probe binaries remain byte-identical after cleanup. Model comparisons use
the immutable private token tracer. No throughput or vLLM-parity claim is
made. Longer/resumed FP32 prepared attention and the two/three-column MMVF
contract remain open. Frozen candidate, logs, reference captures, source
hashes, and qualification: `/tmp/bitnet-goal-gated-mmf-final/`.

### CUDA shared Q8 projection policy (2026-11-06)

Shared CUDA Q8 gate/up/down projections no longer combine an explicit
native quantized-activation request with the CPU AVX2 reference-accumulation
flag. That flag selected the CPU-contract kernel before native execution.
Backend policy now preserves the requested native CUDA projection. Metal
selection and explicit legacy primitives are unchanged; the CUDA kernel
source is byte-identical to the preceding prepared-attention checkpoint.

Ten independent pinned GGML CUDA projection/bias fixtures cover 37x64,
256x512, 2560x640, 2560x2560, and 640x2560 shapes with two seeds. Direct,
graph capture/replay, both shared projection policies, fused bias/copy,
and nonzero-offset guards pass 48,424 comparisons (24,212 unique reference
values). The old policy fails the first independent case. Existing primitive
fixtures are unchanged. Sources: `test/reference_cuda_shared_q8_matvec.c`
and `test/cuda_shared_q8_matvec_reference.h`.

The clean warning-free build, standard tests, complete CUDA suite, and
diagnostic isolation pass; four user-owned probe binaries remain unchanged.
The year diagnostic remains 5/8. First-decode layer-zero FFN exact values
increase from 755/2560 to 1139/2560, with unchanged maximum error 2.98e-8.
Whole-model gates still fail and the short prefix metric regresses again:
121/178 tokens, one count-mismatched prompt (178 runtime / 179 oracle IDs),
versus 129/176 in the preceding prepared-attention checkpoint. Long remains
5/24 with equal counts. These failures are retained, not treated as model
qualification. Artifacts: `/tmp/bitnet-goal-shared-q8-final/`.

A separate scan of the preceding prepared-attention candidate matches the
first 16 prefill FFN values at layers 15, 31, 39, 43, 45, and 46. The final
layer 47 differs across all 2,560 values (maximum 3.0483e-3). Each restricted
observer preserves the normal reference IDs. Critically, the reference
layer-46 FFN tensor has 11 rows while layer 47 has one: the reference drops
unrequested output rows before the final stateless FFN. This runtime's
existing final-row selection is guarded by `!prefill_gpu`, retaining eleven
rows on CUDA and changing the final projection's MMQ/MMVQ contract. Next
work should qualify backend-aware final output-row selection while retaining
all rows for all-logit requests and preserving already completed KV/SSM/PLE
state. Scan and tensor-shape evidence:
`/tmp/bitnet-goal-post-mmf-scan/prefill-scan.json`.

### CUDA final stateless FFN output-row selection (2026-11-06)

CUDA fallback prefill now selects only the last requested output row for the
final stateless FFN, after token/KV/SSM/PLE updates complete. All-logit requests
retain every row, and backend-managed fused chains keep their own selection.
CPU behavior and Metal/WebGPU admission are unchanged. Two one-row arithmetic
fallbacks also needed correction: CUDA-resident quant projections may now use
the backend matmul callback at one token, with CPU retry on rejection; HC mixing
retains CUDA norm/projection/activation arithmetic when cropping leaves one row.

The year prompt's complete final-layer FFN now matches all 2,560 independent
normal-oracle values exactly (previous maximum error 3.0483e-3). The HTTP
prompt also matches all 2,560 final FFN values, with eight prompt tokens and
selected position seven. Its restricted observer preserves the no-observer
reference IDs. The initial position-five attempt emitted no final-row tensor
and is excluded from numeric evidence. Despite the exact final FFN, HTTP
still disagrees on the first sampled token, so the next boundary is HC output
mixing, final normalization, and logits; this is not merely a later-decode issue.

The new `test_prefill_output_rows` real-model gate checks last-logit, all-logit,
and no-logit requests. On Qwen27 CPU and CUDA, all requested logits are finite,
nine host/device state component hashes agree, and a teacher-forced next step
has bit-identical full logits. The gate checks optional PLE state if present;
the executed dense Qwen27 model does not qualify Flash Next PLE state. This
is execution invariance, not an independent mathematical reference. Build with
`make BN_ENABLE_CUDA=1 test_prefill_output_rows` and run with a dense GGUF path
plus optional `--cuda`.

Clean build, standard tests, adapter retry tests, full CUDA suite, and diagnostic
isolation pass. A test-only shadow warning was corrected and the transformer
target rechecked without warnings. Four user-owned probe binaries remain
unchanged. Qwen27 retains 192/192 short and 24/24 long CUDA IDs. Flash Next
remains 121/178 short prefix IDs (178 runtime / 179 oracle; one count mismatch)
and 5/24 long with equal counts. No performance-parity or complete model
qualification claim is made. Frozen binaries, state checks, model failures,
reference tensor captures, and provenance: `/tmp/bitnet-goal-last-row-final/`.

### CUDA native-quant logits activation contract

The CUDA logits adapter now requests the native quantized-activation matrix
path with one input row when the quant registry declares support. The generic
CUDA matvec callback retains its existing floating-input contract; rejection
falls back through the existing adapter. CPU and Metal dispatch are unchanged.
This fixes the Q8_0 output head selecting the legacy floating-input projection.

The existing ten independent Q8 projection fixtures now also exercise the
actual logits adapter: 60,530 word comparisons across five paths, with the
reference hashes unchanged. The old adapter fails the added path's first
fixture; the corrected adapter passes. Unit tests cover CUDA acceptance,
CUDA rejection/retry, and Metal dispatch. Clean CUDA build has no warnings;
standard tests and the full CUDA backend suite pass. The four user-owned
probe binaries were restored and hash checked after cleanup.

The frozen, newly linked executable matches all eight normal-reference IDs
for `HTTP status code 404 means`. The year prompt still matches only the first
five of eight IDs. Thus the HTTP first-token defect is resolved, while decode
correctness remains open. Two earlier diagnostic runs copied an executable
before the link finished; those runs are explicitly excluded. Current artifacts
are in `/tmp/bitnet-goal-native-logits-final/`, including the independent negative
and positive fixture evidence in the adjacent `native-logits-*.log` files.
These checks do not establish performance parity or full Flash Next support.

The completed short gate improves from 121/178 to 145/178 prefix IDs, with
178 runtime versus 179 oracle tokens and one count mismatch. All eight first
IDs match. The long gate remains 5/24. Qwen27 regression gates retain 192/192
short and 24/24 long IDs. The artifact directory contains `qualification.json`
with executable/source hashes and complete gate counts.

At the year prompt's first divergent decode position (15), the final
normalized input to the output projection differs by 0.452323 maximum and
0.0725122 RMS across 2,560 values (zero bit-exact words). The restricted
`result_norm` observer preserves all eight no-observer reference IDs. The
runtime's `gpu_result_norm` capture preserves its ordinary eight IDs. This
places a remaining discrepancy before the logits projection; localize decode
layer/state boundaries next. The initial runtime attempt selected the CPU
logits tag and emitted no tensor, so it is excluded. Numeric evidence and
output IDs are retained in `year-pos15-boundary.json` in the artifact directory.

### CUDA hyperconnection Q8 projection policy

Decode boundary tracing localizes an arithmetic mismatch inside the layer 3
hyperconnection mixer. On the year prompt's first decode step (position 11),
the HC residual before layer 3 and after its attention both agree within
2.98e-8, and the attention projection agrees within 1.49e-7. Yet the mixer gate
differs by 2.19e-3 and its FFN input by 3.79e-4. The restricted observers retain
all eight normal reference IDs. Later layer 4 differences are downstream and
do not by themselves establish a recurrent-state defect.

The mixer requests CPU-reference Q8 accumulation when the small-dense native
policy is disabled. Its CUDA down/up projections now instead request the
native quantized-activation contract through a backend policy helper. The
existing CPU/Metal projection flags and independent reference values remain
unchanged. The failing old policy hashes `2520f2787d6c8993` where the pinned
CUDA oracle expects `dc834a56adc00bd5` in the first fixture. The corrected
policy passes 14 independent cases and 199,356 word comparisons across six
paths. Four new cases cover actual HC down/up shapes, 320x10240 and
10240x320, with two deterministic seeds. Regeneration preserves all ten
existing target rows exactly. Unit tests check backend selection and the
flags emitted by the actual HC graph builder.

The captured layer 3 FFN input error falls to 5.96e-7 maximum (1.43e-7 RMS).
The year prompt still diverges at output six, so this local correction does
not qualify the full model. HC normalization still selects host reference
ordering through the same small-dense policy; qualify that selection against
the pinned CUDA oracle next, preserving the existing operator fixtures.
Tracing artifacts and excluded empty captures are recorded in
`/tmp/bitnet-goal-decode-boundaries/trace-manifest.json`; build, negative/positive
fixture checks, and model gates are in `/tmp/bitnet-goal-hc-native/`.

Clean warning-free build, standard tests, and the full CUDA suite pass. The
completed Flash short gate is 132/179 prefix IDs, with 192 runtime versus
179 oracle tokens and one count mismatch; long remains 5/24. This is not a
whole-model improvement over 145/178. At position 15 the final normalized-state
error grows from 0.452323 to 1.21561 maximum (0.102889 RMS), despite the improved
first-decode mixer boundary. Preserve these failures while qualifying the
remaining normalization ordering; independent operator agreement does not
establish a correct full-model trajectory.
Qwen27 regression gates remain 192/192 short and 24/24 long CUDA IDs. The
artifact directory's `qualification.json` records all four model gates,
source/executable hashes, test outcomes, and preserved probe hashes.

### CUDA mixer normalization, reduction, and F32 injection

The HC graph builder now selects CUDA normalization and gated reduction
independently of the small-dense quant policy. Host-reference order remains
available to CPU/Metal. Revalidation against pinned GGML CUDA df18bee11
reproduces all 48 retained RMSNorm hashes unchanged. The old graph selector
fails those targets; the corrected selector passes 387,072 RMSNorm word
comparisons and 847,872 fused-mean comparisons. Graph tests check the emitted
normalization and reduction parameters. No GPU kernel or reference target was
changed to obtain those passes.

The F32 HC injection projection also requested AVX512 accumulation on CUDA.
It now follows the backend projection policy, preserving Metal's existing
flags. The old flags fail the independent F32 matrix fixture. Two additional
pinned CUDA targets cover the actual 4x10240 injection and 48x2560 recurrent
coefficient shapes; all existing targets are preserved. Tests exercise both
the native matrix path and the HC-selected graph flags, including output
guards. Source generators are `test/reference_cuda_hc_rmsnorm.c` and
`test/reference_cuda_hc_f32.c`.

The normalization/reduction-only intermediate candidate passes standard and
CUDA tests but regresses Flash Next to 118/179 short prefix IDs (192 runtime
versus 179 oracle tokens); long stays 5/24. Its frozen executable, source
hashes, and failed gates remain in `/tmp/bitnet-goal-hc-norm/`. Adding the F32
injection correction gives 143/177 short prefix IDs, with 177 runtime versus
179 oracle tokens and one count mismatch. This is still unqualified and
does not establish improvement over the earlier 145/178 candidate. Final
injection-candidate artifacts are in `/tmp/bitnet-goal-hc-inject/`.

A standalone pinned-GGML replay of the actual layer 3 HC weights reproduces
the captured reference normalized input (10,240 values), gate (10,240), and
mixed output (2,560) exactly. This validates a controlled same-input replay
for separating mixer execution from incoming-state differences. Model
weights and provenance, replay code, and comparisons are retained in
`/tmp/bitnet-goal-hc-replay/`. The first-decode model boundary still differs
by about 3.79e-4 after these policy corrections; fixture passes alone do not
qualify the complete trajectory.

Feeding the runtime's captured post-attention residual into that pinned
replay reproduces the runtime mixed output exactly, 2,560/2,560 values. The
reference and runtime incoming residuals differ by only 2.98e-8 maximum
(2.94e-9 RMS; 1,451/10,240 words identical), yet the two reference replays
produce the observed 3.79e-4 output difference. Thus the corrected mixer
matches the independent same-input graph; the remaining trajectory error
must be traced earlier than that incoming residual. The replay/capture
preserves the ordinary runtime IDs. See `same-input-check.json` for the
comparisons. Further mixer arithmetic changes are not justified by this
later boundary discrepancy alone.
The final injection candidate passes the warning-free clean build, standard
tests, and full CUDA suite. Qwen27 retains 192/192 short and 24/24 long CUDA
IDs; Flash Next remains 143/177 short and 5/24 long. User-owned probe hashes
are unchanged. `qualification.json` in the injection artifact directory
records all model counts, source/executable hashes, independent negative
controls, and same-input replay checks.

### CUDA unscaled routed-expert weighted sum

The first decode layer now provides a controlled localization: attention
input, recurrent output, and FFN input each match all 2,560 independent values
exactly. Before this correction, the FFN output matches 2,052/2,560 values,
with maximum error 7.45e-9. Those tiny differences later cross quantization
boundaries. Selected expert IDs and all ten routing weights agree, including
an unobserved router replay. All 25,600 selected expert projection outputs
also agree. Actual shared-expert projection, sigmoid, and final-combine
replays rule out those branches for this captured input.

The discrepancy is the weighted reduction. The pinned reference fuses
multiply, expert views, and ordered additions for 2..15 experts even without
an additional expert output scale. The runtime requested its existing FMA
kernel only when a separate scale was present. Backend policy now also
selects that kernel with unit output scale for eligible complete CUDA sums.
One/16-expert cases, partial accumulation, missing backend capability, and
Metal retain the existing selection. Existing explicit-scale handling and
the unflagged weighted-add primitive are preserved.

Using the same captured weights and expert outputs, the old reduction matches
1,681/2,560 values; the corrected reduction matches all 2,560. Integrated
layer 0 first-decode FFN output also becomes exactly equal. Forty new pinned
CUDA fixtures cover 1/2/10/15/16 experts, widths 37/2560, one/three tokens,
and two seeds, with direct execution, capture, replay, and output guards:
311,640 word comparisons. The old selector fails the first two-expert case;
the corrected selection passes with reference targets unchanged. Unit tests
check both selection boundaries and emitted graph flags. Generators and
targets are `test/reference_cuda_moe_weighted_sum.c` and
`test/cuda_moe_weighted_sum_reference.h`.

The initial stacked gate/up diagnostic did not represent the production
borrowed-weight path and is excluded. A corrected separate-projection replay
reproduces the ordinary runtime FFN output exactly. The first standalone
router graph exposed a reference lookahead crash at graph end; a trailing
CONT consumer avoids it without modifying reference source. Trace evidence
and exclusions are in `/tmp/bitnet-goal-shared-gate/trace-manifest.json`.
Candidate builds, fixture negative controls, and model gates are in
`/tmp/bitnet-goal-moe-fma/`. The year prompt still diverges later, so exact
agreement at this first layer does not qualify the model or establish parity
with vLLM.

The completed candidate passes a warning-free clean build, standard tests,
the full CUDA suite, and Qwen27's 192/192 short plus 24/24 long CUDA gates.
Flash Next remains unqualified: short 120/179 prefix IDs, runtime 192 versus
oracle 179 tokens with one count mismatch; long 5/24. This is not a whole-model
improvement over the preceding candidate. Layer 1 now receives an exact
10,240-value residual. Its post-PLE attention-mixer input differs by at most
2.38e-7 (950/2,560 exact), localizing the next trace to that transition. Both
restricted observers preserve the normal reference IDs. Probe hashes remain
unchanged. Full counts, corrected test-link recheck, source hashes, and tensor
boundaries are recorded in the candidate's `qualification.json`.

### CUDA decode positional-embedding path

After the routed-sum correction, layer 1 receives an exact 10,240-value
residual, but its attention-mixer input still differs. Decode positional
embedding was forced through host-exact arithmetic by the model's recurrent
policy even on CUDA. Prefill already used the GPU operator path. Backend
policy now selects that GPU path for CUDA decode as well, while retaining
the existing non-CUDA selector and fallback behavior. No PLE kernel or
reference target was changed. Unit tests check CUDA versus Metal selection.

On the first decode step of the year prompt (position 11), the old host path
matches only 744/10,240 gated values (maximum error 3.73e-9) and 1,207/10,240
convolution values (maximum 2.98e-8). The GPU path matches all 10,240 values
in each capture exactly. The following attention-mixer input improves from
950/2,560 exact values (maximum 2.38e-7) to 2,560/2,560. Restricted reference
observers preserve the normal eight reference IDs. The runtime's ordinary
eight IDs also remain unchanged, including the later divergence; this is a
qualified local correction, not whole-model qualification.

The first attempts requested GPU-only tags while the old policy executed the
host path, so they emitted no runtime tensors and are excluded. The reference
gate callback captures only one scalar due to its row selection; no complete
four-stream gate proof is claimed from that file. Full gated-value and conv
vectors are the numeric evidence. Captures and comparisons are in
`/tmp/bitnet-goal-decode-boundaries/ple-host-comparison.json` and
`ple-native-comparison.json`; the frozen candidate and build/model gates
are in `/tmp/bitnet-goal-ple-native/`.

The completed candidate passes the warning-free clean build, standard tests,
and full CUDA suite. Qwen27 retains 192/192 short and 24/24 long reference IDs.
Flash Next improves from 120/179 to 144/179 short prefix IDs, with runtime 192
versus oracle 179 tokens and one count mismatch; long remains 5/24. These
Flash checks retain the four opt-in flags for fused HC mean/post, resident
prefix chaining, and GDN decode order, with 2 GiB RAM / 4 GiB GPU expert caches
on SM120. No default-path or cross-device model qualification is implied.

Layer 1's recurrent output and following FFN input each match all 2,560
independent values exactly. Its FFN output is the next non-exact boundary:
1,683/2,560 values match, maximum error 3.73e-9. The next investigation is
therefore a controlled same-input layer 1 FFN replay. Complete model counts,
boundaries, source/executable hashes, and unchanged user-probe hashes are
retained in the candidate's `qualification.json`.

### CUDA shared-expert scalar gate and activation policy

The layer 1 FFN input is exact, but the old FFN output matches only
1,683/2,560 values (maximum error 3.73e-9). A controlled replay of the actual
input and weights isolates the shared scalar gate: pinned CUDA produces
`0x3e1fe700`, while the CPU-reference gate path requested by the CUDA caller
produces `0x3e1fe6ff`. Native CUDA gate arithmetic matches the reference.
Combining identical routed and shared projections with the old scalar gate
reproduces the ordinary runtime FFN output exactly; selecting native gate
math reproduces the reference FFN output exactly. Both captured reference
observers preserve all eight normal IDs.

Backend policy now keeps shared-expert scalar gating and SiLU on CUDA's
native arithmetic contract. Non-CUDA selection and explicit legacy operator
flags remain unchanged. The same-input shared SiLU replay also improves
from 356/640 exact intermediate values to 640/640; Q8 down projections for
this input were already exact under both choices. This distinction avoids
claiming that the intermediate SiLU discrepancy caused this particular final
FFN error. No kernel or existing reference target was changed.

Eighty-six new pinned-CUDA cases cover sigmoid/add/reset/complement and SiLU
with input widths 32/256/640/2560/8192, varied output widths, and two seeds.
Direct, capture, replay, and guard checks pass 331,062 word comparisons. The
old selector fails a sigmoid/add case; graph tests check the flags emitted
for CUDA and Metal. Generator and targets are
`test/reference_cuda_moe_shared_math.c` and
`test/cuda_moe_shared_math_reference.h`. The integrated layer 1 first-decode
FFN output now matches all 2,560 reference values exactly. The year prompt
still diverges later, so whole-model qualification remains open.

The protected clean build has zero warnings; standard tests and the full CUDA
suite pass. The frozen candidate passes Qwen27 short (192/192) and long
(24/24). Flash short improves to 159/179 prefix IDs, with 179 tokens from
each runtime and no count mismatches; Flash long remains 5/24. These are
correctness comparisons against pinned llama.cpp, not vLLM performance parity.

On the year prompt's first decode, layer 2 recurrent and FFN outputs and
layer 3 attention input each match all 2,560 values exactly. Layer 3 attention
output matches 223/2,560, maximum error 1.49e-7. This narrows the next
investigation to attention/cache execution; it does not yet establish its
cause. Full gates and boundary counts are in
`/tmp/bitnet-goal-shared-native/qualification.json`.

A subsequent cache dump from the normal reference graph (observer disabled,
all eight generated IDs preserved) matches every prompt key and value in
layer 3: 5,632/5,632 each, positions 0–10. The temporary diagnostic probe
only adds access to the hybrid memory's attention cache; pinned reference
libraries are unchanged. This excludes prompt-cache corruption at this
boundary, while the current decode query/key and attention computation
remain unqualified. Layouts, hashes, and counts are in
`/tmp/bitnet-goal-attention3-cache/manifest.json`.

Original layer 0 replay inputs were preserved separately. Layer 1 replay
weights, input provenance, gate bits, and full-combine comparisons are in
`/tmp/bitnet-goal-layer1-ffn/trace-manifest.json`; the frozen candidate and
build/model gates are in `/tmp/bitnet-goal-shared-native/`.

### CUDA reference-attention Q8 projection policy

The layer 3 first-decode attention input is exact, but the old Q8 value
projection matches only 121/512 values (maximum error 2.68e-7). A pinned
GGML replay using the actual weights and identical input matches all 512
normal-reference cache values. Runtime native Q8 projection also matches
all 512; the explicit CPU accumulation flag reproduces the old runtime
value vector exactly. The correction belongs to projection policy, not
RoPE or the prompt cache.

Reference attention preparation now selects native CUDA accumulation for
Q8 projections in a native Q8 reference group. The existing mixed-format
group contract is preserved: applying per-projection format flags broadly
regressed Qwen27 from 192/192 to 180/192 under identical settings, while
the prior frozen binary still passed 192/192. The final policy retains the
qualified mixed-format flags and has a regression assertion for them.
Normalization, attention arithmetic, explicit legacy operator flags, and
non-CUDA paths retain their contracts.

Four independently generated targets add actual Q/KV dimensions
512x2560 and 12288x2560 with two seeds; all 14 prior targets are unchanged.
Eighteen cases across projection adapters pass 529,434 word comparisons.
Restoring the old selector fails the added attention-policy execution path
on the first case. Graph tests inspect all three emitted projection flags.

Integrated native Q8 query, new key, and new value vectors are each bit-exact
(6144/6144, 512/512, 512/512). Prompt keys and values remain exact. The
attention weighted output still differs: 380/6144 exact, maximum 3.34e-6.
A separate pinned GGML attention graph using the identical captured Q/K/V
reproduces the normal reference weighted output 6144/6144 exactly. Thus the
remaining difference is within score/softmax/value reduction. The runtime
`gpu_attn_probs` dump is excluded because it is not the fused path's actual
probability intermediate. Replay source, inputs, hashes, and exclusions:
`/tmp/bitnet-goal-attention3-cache/replay-manifest.json`.

With identical captured inputs, runtime's separate score/softmax/value path
also matches all 6,144 reference outputs; the automatically selected fused
path reproduces the 380/6,144 runtime result exactly. This is evidence for
qualifying the F32 GQA12 vector dispatch next, rather than changing RoPE,
cache contents, or downstream projection math.

The final protected clean build has zero warnings, standard tests and the
full CUDA suite pass, and Qwen27 passes short 192/192 and long 24/24.
Flash short is 152/179 prefix IDs with no token-count mismatch; long remains
5/24. Operator corrections do not imply monotonically improving sampled
prefixes while later arithmetic remains different. Whole-model qualification
remains open. Final source hashes, gates, and scope:
`/tmp/bitnet-goal-qkv-native-final/qualification.json`.

### Fused CUDA non-flash F32 attention order

A fused F32 path now preserves the reference MMVF score, row-softmax, and
MMVF value-reduction order for 256-wide heads and up to 256 cached keys.
It is selected by the existing reference-order graph flag when the model
requires reference attention and flash math was not explicitly requested.
Older flash targets and explicit flash requests retain their contracts.
Longer reference contexts use the qualified separate operations while the
long fused path remains open. Fusions that would bypass the requested order
are excluded. All intermediate values remain on the device.

The independent pinned GGML generator `test/reference_cuda_gqa12_order.c`
covers GQA12/KV2 and GQA6/KV4, two seeds, zero/three-key windows, and lengths
1/3/12/32/33/127/256/257/513/1047/2048. The 88 targets in
`test/cuda_gqa12_order_reference.h` pass 1,622,016 bit-exact comparisons,
including direct execution, graph capture/replay, nonzero cache offsets,
padded strides, output guards, and unchanged-cache hashes. Restoring the
old attention selection fails case 4 (three keys). Graph tests cover CUDA
and Metal, explicit flash, F16 caches, and the 256/257 dispatch boundary.

The first implementation was correct but slower in the captured real
12-key replay: 31.2 versus 21.5 microseconds for separate operations.
Coalescing value reads across output features while retaining the same
reduction tree gives 19.3 versus 21.8 microseconds in 1,000 warmed adapter
executions, both producing all 6,144 reference values exactly. This scoped
wall-time measurement includes adapter readback; it is not end-to-end
serving or vLLM parity evidence. The initial slower version was replaced.

With weighted attention exact, the remaining layer 3 output error isolates
to Q8 output-projection accumulation. A same-input pinned GGML replay and
native CUDA projection each match the reference 2,560/2,560. Explicit CPU
accumulation reproduces the old runtime vector exactly (only 223/2,560
reference values exact). Output policy now selects native CUDA arithmetic
within its native Q8 query group and preserves qualified mixed-format
selection. Two new 2560x6144 targets retain all 18 previous Q8 targets;
the 20-case adapter suite passes 767,352 comparisons, and restoring the old
output selector fails its execution check.

The integrated layer 3 attention output is now 2,560/2,560 exact. Its FFN
output is the next measured non-exact boundary: 577/2,560, maximum 7.45e-9.
Same-input replay is required before assigning that error to an operator.
The next FFN input is also 2,560/2,560 exact. A same-input unobserved router
replay finds 512/512 logits and all ten expert IDs exact, but normalized
weights differ in all ten values, maximum 1.49e-8. This narrows the next
correction to route-weight computation rather than projection or selection;
a full-combine replay is still needed to assign the FFN error. Input,
weights, source, IDs, and hashes are preserved in
`/tmp/bitnet-goal-layer3-ffn/router-manifest.json`.

The final protected clean build has zero warnings; standard tests, backend
matrix, and full CUDA suite pass. Qwen27 remains 192/192 short and 24/24
long. Flash short matches 145 prefix IDs, with 175 runtime versus 179
reference tokens (one count mismatch); long remains 5/24. An incomplete
reference long run under concurrent diagnostic work is excluded; the solo
rerun supplies the long result. Correct isolated operators do not qualify
the whole model while later arithmetic differs. No vLLM parity claim is made.
Reference provenance and projection replay:
`/tmp/bitnet-goal-gqa12/reference-manifest.json`; final source hashes and gates:
`/tmp/bitnet-goal-attention-native-final/qualification.json`.

### CUDA native router normalization policy

The layer 3 FFN input and all 512 router logits are exact, but the old
separate-top-k request differs in every normalized weight (maximum 1.49e-8).
A controlled replay of the identical input and actual F32 router weights
shows native fused top-k matches all ten reference weights and expert IDs.
The separate request reproduces all ten old runtime weights exactly. The
explicit CPU accumulation request is a different contract and is not the
observed production path. These distinctions are preserved in
`/tmp/bitnet-goal-router-order/comparison.json`.

Backend policy now selects native CUDA fused routing for admitted power-of-two
expert counts 32–512 and active counts 1–16. It retains normalization flags,
output scale semantics, model anatomy, CPU/Metal selection, unsupported
shapes, and explicit legacy operator flags. Decode policy and emitted flag
checks cover the selection; the old architecture-level accessor remains
unchanged. CUDA prefill already uses the native route contract.

The new independent pinned GGML generator `test/reference_cuda_router_order.c`
covers 480 cases: five expert counts, four active counts, normalized and
unnormalized weights, two seeds, scales 1/0.75, tied probabilities, and
selected experts sharing reduction lanes. Targets are stored separately in
`test/cuda_router_order_reference.h`; earlier reference targets are unchanged.
Direct, captured, and replayed executions pass 20,880 weight/ID comparisons
with output guards. Restoring the old selector fails a tied-probability case
(experts 32, active 2), including checkpoint expert selection.

The integrated layer 3 FFN now matches all 2,560 reference values exactly,
with its input unchanged. The year prompt's first eight sampled IDs now also
match. This establishes the previously isolated FFN correction; whole-model
qualification still depends on the complete prompt and long-context gates.
The protected clean build has zero warnings; standard tests and the full
CUDA suite pass. Qwen27 remains 192/192 short and 24/24 long. Flash now
passes the full short gate: 179/179 sampled IDs with identical counts over
eight prompts. The CLI text comparison still displays special-token text
differently on one prompt; this is not a sampled-ID failure. The long gate
remains 5/24, so long-prefill state and subsequent decode are the next gates.
No whole-model or vLLM parity qualification is claimed.

The first long-context boundary probe compares recurrent layer 0 at position
1047, after the 1,047-token prefill. Its projected output matches all 2,560
values exactly. The exact-name reference observer preserves the eight IDs
from a separate normal graph run. This qualifies that output boundary only,
not all recurrent state or the remaining long-context layers. Continue with
the following FFN and later boundaries. Probe source, outputs, normal IDs,
and hashes: `/tmp/bitnet-goal-long-decode-boundaries/qualification.json`.

Reference source, inputs, weights, hashes, and counts are preserved in
`/tmp/bitnet-goal-router-order/reference-manifest.json`; final source hashes
and gates are in `/tmp/bitnet-goal-router-native-final/qualification.json`.

## Historical Context

Earlier measurements on an Apple M1 Max showed the CPU path reaching high memory
bandwidth utilization on BitNet ternary models and competitive throughput on
several dense and sparse quantized models. Keep old numbers in commit history;
current docs should prefer reproducible commands and recent gates over broad
claims.


### CUDA stacked Q8 prefill projection geometry

The long Flash Next discrepancy begins in prefill history, before the first
decode step. At prompt position 1038 the layer 0 SSM output, HC attention
combine, and FFN input are bit-exact. Its combined FFN output differs by at
most 2.98e-8, which enters layer 1 PLE history and later amplifies in recurrent
state. Swapping only captured PLE history in a same-input replay reproduces
both runtime and reference convolution outputs exactly; the current PLE
normalization and convolution are therefore not the correction target.

A same-input Q8 shared-expert replay isolates stacked gate/up geometry:
separate projections match all 2,560 reference outputs, while the stacked
projection reproduces the runtime output exactly. The allocation contains
two logical 640-row projections, but the old launch uses 1,280 rows for MMQ
partition selection. CUDA now retains stacked storage and projects two row
views using the logical row count. This applies to contiguous host batches
and the device shared-FFN helper for native quant formats; the specialized
routed host layout and single-token path retain their prior contracts.

`test/reference_cuda_q8_stacked_ffn.c` independently generates 52 pinned
GGML CUDA cases: dimensions 256/2560, hidden sizes 96/640, two seeds, and
batch sizes 1,2,3,4,8,9,16,17,33,129,257,513,1047. Separate and stacked host
FFNs and device grouped shared-FFNs pass 45,934,592 output-word checks with
output guards. These callbacks are not graph-capture tests. The old backend
fails the 2560x640, 129-token stacked case against unchanged reference
hashes. Actual model replay now matches 2,560/2,560 reference shared outputs
for both storage layouts, and the integrated position-1038 combined FFN
also matches all 2,560 values. Evidence is retained in
`/tmp/bitnet-goal-prefix-ffn-replay/`,
`/tmp/bitnet-goal-ple-long-history/`, and
`/tmp/bitnet-goal-q8-prefill-native-final/`.

The protected clean build has zero warnings; standard tests, the full CUDA
suite, and backend matrix pass. Qwen27 retains 192/192 short and 24/24 long;
Flash retains 179/179 short with identical stopping counts. A normal
unobserved runtime run matches 24/24 unobserved reference completion IDs
with matched 1,047-token microbatches. The reference itself changes IDs at
step 5 under its default 512-token microbatch schedule, retaining the
5/24 cross-schedule result. The diagnostic eight-token agreement is therefore
explained by reference batching, rather than observation. Both schedules
are retained; chunked/resumed and schedule-independent qualification remain
open. See `matched-schedule-comparison.json` and the two reference logs.
Layer 1 first-decode recurrent output remains numerically different
(max 0.000338674). No vLLM parity is claimed. Final sources and gates:
`/tmp/bitnet-goal-q8-prefill-native-final/qualification.json`.

The remaining PLE discrepancy is now independently isolated. With the Q8
correction, the captured position-1038 residual entering PLE is exact in all
10,240 values, but its first gated stream still differs. Pinned GGML on the
identical captured normalized key/query selects 512 reduction threads for
one token and 128 for 1,047 tokens on the 188-SM device. Both existing CUDA
kernel geometries reproduce their own GGML outputs exactly (four gates).
The 128-thread replay also matches all four actual model reference gates;
the currently requested 512-thread replay matches three, with one difference
of 7.45058e-9. The exact-name model observer preserves all eight normal
reference IDs. This isolates lost batch geometry in prefill orchestration,
rather than a remaining error in the signed-root kernel. That orchestration
correction is still open. Controlled sources, sparse full-batch reference
replays, and the actual-model comparison are in
`/tmp/bitnet-goal-prefix-ple-reduction/`.


### CUDA PLE prefill reduction geometry

Prefix PLE now propagates its logical token count through the projected
embedding adapter. The generic signed-root gate callback accepts that batch
count alongside its explicit reduction mode. Mode zero selects from the
full stream-row count and backend SM metadata; modes 32/128/512 preserve
explicit arithmetic contracts. Each call still processes one token, keeping
sequential convolution-history updates, while the reduction retains the
geometry of the logical batch. Single-token decode keeps explicit mode 512.
The selection has no model-name branch and no hardware-count constant in
transformer code. This corrects reduction ordering; it does not yet batch
all PLE transfers or remove per-token allocations.

`test/reference_cuda_signed_sqrt_batch.c` independently produces 28 pinned
GGML CUDA targets with widths 256/2560, four streams, two rational-input
seeds, and token counts 1,8,93,94,95,512,1047. The 93/94/95 cases bracket the
188-SM reference device's row-reduction selection boundary. Runtime tests
check first/middle/last token gates and full gated vectors, including the
32-thread small-width mode. Earlier signed-root reference targets remain
unchanged. Sources and controlled actual-model replays are retained in
`/tmp/bitnet-goal-prefix-ple-reduction/`.

All 28 new automatic-selection cases pass 473,424 output-word checks. Forcing
the old 512-thread selector fails the retained 95-token, width-256 target.
The earlier explicit-mode signed-root reference suite also passes unchanged.
Protected clean build has zero warnings; standard tests and backend matrix
pass. Final integration evidence is collected under
`/tmp/bitnet-goal-ple-batch-native-final/`.

The integrated 1,047-token-prefix history now matches all 92,160 saved PLE
values exactly, and the following layer 1 recurrent output matches all
2,560 reference values exactly. The captures exit successfully and use the
same four Flash correctness flags and bounded 2 GiB RAM / 4 GiB GPU expert
caches as the earlier checkpoint. Normal unobserved generation retains
24/24 reference completion IDs under matched 1,047-token microbatches;
the 512-token reference schedule remains a separate comparison.
The full CUDA suite passes, Qwen27 retains 192/192 short and 24/24 long, and
Flash retains 179/179 short. This qualifies the corrected boundaries and
these token workloads, not all chunk/resume schedules or vLLM performance.


### Chunked prefill entry admission

The bounded-cache F32-KV continuation harness exposes a separate entry bug.
`bn_prefill()` admits host-orchestrated batch fallback when native GPU
chaining is unavailable, while `bn_prefill_no_logits()` previously tested
only native chaining. Earlier chunks therefore silently ran sequentially
for HC models, changing both arithmetic order and performance relative to
the requested independent chunk schedule.

Controls use the same pinned no-observer reference, token inputs, F32 KV,
eight threads, 2 GiB RAM expert cache and 4 GiB GPU expert cache. The GPU
qualification harness conservatively charges each cache entry 32 MiB (128
slots); these cache budgets exclude base weights, PLE tables, KV and
activations. Before correction, a 257-token prefix in chunks of 129 matches
14/16 teacher-forced token checks, a 33-token prefix in chunks of 9 matches
8/16, and a single 9-token prefix followed by resumed chunks of 9 matches
15/16. Whole-prefix controls at 33 and 257 tokens each match 16/16. Each
case includes eight initial decode tokens, 64 resumed input tokens on the
specified chunk schedule, and eight more decode tokens. Failed checks feed
the supplied reference IDs onward; their later matches are conditional.
Replacing only intermediate no-logits calls with ordinary batched prefill
makes the failing 33/9 case pass all 16 unchanged reference checks.

Both APIs now use one batch-admission policy. The regression covers HC
fallback admission without native chaining, missing backend/config, and
explicit no-prefill. Existing independent token files are retained for the
post-correction replay. Pre-correction sources, binaries and logs are in
`/tmp/bitnet-goal-flash-chunked-f32/`; final evidence is collected in
`/tmp/bitnet-goal-prefill-entry-final/`.

Post-correction, the original no-logits harness passes 16/16 on 257/129,
33/9, 257/257, and 33/33 (prefix length / chunk size). The 9/9 case improves
its trajectory but retains one resumed mismatch (15/16, now resumed step 2).
Longer matched replays remain open: 1047/512 matches 14/16 (initial step 6
and resumed step 3 fail), and 1047/129 matches 15/16 (resumed step 3 fails).
The 1047/512 initial candidate gap is only 0.00152588; no tolerance is used
to turn sampled-ID failures into passes. These are independent teacher-forced
checks, not a claim of complete schedule qualification. All underlying
reference token files and failed logs are retained.

The entry correction passes a protected clean build with zero warnings,
standard tests, backend checks, and the full CUDA suite. CUDA math source
is unchanged from the PLE batch-geometry checkpoint. Qwen27 retains 192/192
short and 24/24 long; Flash retains its 179/179 short gate. The remaining
continuation discrepancies require boundary/state comparison before any
further numerical change.

The additional 1047/1047 whole-prefix plus resumed control passes 16/16.
The 9/9 all-logits control retains the same resumed step-2 mismatch as the
corrected no-logits API, isolating that remaining numerical failure from
the admission asymmetry. Final build/source hashes, retained failures,
controls and gates: `/tmp/bitnet-goal-prefill-entry-final/qualification.json`.


### Recurrent state handoff from GPU decode to host prefill

Read-only snapshots isolate the remaining 9/9 resumed failure. Immediately
before resumed prefill, the first three device recurrent states (786,432
F32 words each), their convolution histories (30,720 words each), and PLE
history (92,160 words) match the pinned no-observer reference exactly. The
host recurrent mirrors are stale: layer 0 differs by up to 2.11579. After
resumed prefill without a refresh, layer 0 convolution history is still
exact, but its recurrent state differs by up to 0.549348; slower-decaying
heads retain the error while faster heads forget it.

A controlled replay changes only the device-to-host import of recurrent and
convolution state before resumed prefill. All 16 unchanged independent IDs
then match, and all 2,543,616 captured post-resume words across those three
layers and PLE history are bit-exact. The reference snapshots preserve all
eight initial and all eight resumed normal IDs. No external model math or
reference targets are changed. Sources and negative/positive snapshots:
`/tmp/bitnet-goal-resume9-state/`,
`/tmp/bitnet-goal-resume9-handoff/`, and
`/tmp/bitnet-goal-resume9-refresh/`.

Production now records pending decode state in per-request backend session
metadata. Successful GPU forward, no-logits forward and argmax mark it;
CPU completion, successful state upload and session reset clear it. Before
host-orchestrated recurrent prefill, the backend adapter imports both state
buffers when needed. It clears the marker only after both reads succeed,
so a failed second read can retry both before host computation proceeds.
Consecutive host batches do not repeat this transfer, and native GPU
prefill does not request the host handoff. Unit tests cover independent
sessions, reset, complete transfer sizes/content, failed-read retry and
avoiding redundant reads. This is a decode-to-host-prefill handoff marker,
not a claim of complete multi-session GPU buffer scheduling.

The production handoff now passes the previously failing 9/9 and 1047/129
replays. Seven of eight retained schedules pass all 16 unchanged reference
checks. The 1047/512 case reaches 15/16: its initial step-6 mismatch remains,
while all eight resumed checks pass on the common reference history. The
normal production state snapshot reproduces all 2,543,616 captured words
bit-for-bit, without a manual refresh in the qualification harness.
Protected clean build has zero warnings; standard tests, full CUDA suite
and backend matrix pass. Qwen27 retains 192/192 short and 24/24 long; Flash
retains 179/179 short. Its separate default-reference long comparison stays
5/24 because the runtime and reference use different batch schedules.
Full source/build hashes and retained failures:
`/tmp/bitnet-goal-recurrent-handoff-final/qualification.json`.

The remaining matched 1047/512 failure is isolated beyond the corrected
handoff. Passive snapshots after initial prefill find the first three
recurrent states, their convolution histories and PLE history exact. The
first attention cache at layer 3 differs in all rows (301,826/536,064 words
exact, max 0.000284791), with later recurrent layers diverging downstream.
Host and device keys agree, excluding upload corruption at this boundary.
On identical captured raw K and gamma at position 0, independent pinned
GGML RMSNorm matches all 512 actual reference cache values. The CPU prefill
RMSNorm replay instead reproduces all 512 runtime values exactly (only 334
match the reference). This isolates a backend numerical-contract mismatch
in key normalization, without changing the CPU contract or any kernel.
The passive runtime hook preserves all 16 candidate/reference predictions;
the no-observer reference snapshot preserves all 16 normal IDs.
Evidence: `/tmp/bitnet-goal-long512-state/` and
`/tmp/bitnet-goal-long512-k-norm/`. Native CUDA prepared attention for the
large F32 batch is the next gate; whole-model and performance goals remain
open.

## Flash Next GGUF MoE reference qualification

The pinned GGUF plugin at `e454affe0ca23c9dc8754ec3033270cb30622ed5` needs
correctness fixes before its CUDA grouped MoE path can be a Flash Next baseline.
The unpatched kernel skips expert IDs above 255 even though this checkpoint has
512 experts. Its grouped token-offset lookup also indexes a one-element local
array by warp number. GPU memcheck independently finds reads beyond the actual
640-column Q5_1 down-weight tensor when the final weight tile is partial.
Unpatched grouped results and timings for this configuration are unsupported
comparisons; zeroed high experts reduce the work and produce incorrect outputs.

Apply these patches, in order, to a clean checkout of that plugin revision:

1. `bench/patches/vllm_gguf_moe_expert_count.patch` passes the actual tensor
   expert count through the grouped kernels and retains invalid-ID rejection.
2. `bench/patches/vllm_gguf_moe_metadata_index.patch` uses the current warp's
   first token offset when loading activation metadata.
3. `bench/patches/vllm_gguf_q51_partial_tile.patch` passes the valid block count
   to tile loaders and masks Q5_1 loads in the final partial tile. It covers
   both grouped MoE and ordinary quantized matrix multiplication.

The reference build uses image
`745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`,
`TORCH_CUDA_ARCH_LIST=12.0a`, and the plugin's original build flags. The original
wheel remains unchanged; reference containers install it and replace only its
CUDA extension with the corrected build. Record all three patch hashes and the
extension hash alongside benchmark results. The original wheel hash is
`d6dd44400cc02ca7f3a0452f32589d2eed6588da10ff04858d78b551f9f828ef`.

The corrections preserve quantizer, dot-product, activation and reduction
formulas. Capacity tests preserve all 24 previously valid vector/low-ID outputs
and correct eight skipped high-ID cases. Group-placement tests use identical
weights with varying input scales for Q4_K, Q5_K, Q5_1 and Q8_0. Complete FFN
captures cover layers 2, 3 and 47 at 1, 5, 17, 63, 64, 65, 128, 129 and 257
tokens, including real high expert IDs, shared experts and HC combination.
All 513 captured tensors are reproduced byte-for-byte after the bounded-load
fix; this preserves the independent values rather than regenerating goldens.
GPU memcheck passes the capacity/grouping tests and actual Q5_1 layer-3 replay
at 65 and 257 tokens. Use `--report-api-errors explicit` for these reference
checks: cuBLAS's implicit unsupported-fabric capability probes otherwise appear
as four API reports. Memory checking remains enabled, and the original reports
are retained with the evidence.

Reference probes are `test/reference_cuda_bf16_moe_capacity.py`,
`test/reference_cuda_bf16_moe_grouping.py`, and
`test/reference_cuda_bf16_flash_moe.py`. Evidence, original failures, corrected
build provenance and capture hashes are under `/tmp/bitnet-goal-flash-experts/`.
These qualify the comparison baseline and independent FFN values. Resident expert
execution, bounded placement and whole-model vLLM performance parity remain open.

## Native gathered expert BF16 FFN qualification (2026-11-08)

`test/reference_cuda_bf16_expert.c` replays the independently qualified Flash
Next captures above against optional SM120 resident CUDA projection and FFN
operations. It covers layers 2, 3 and 47 at all nine captured batch sizes,
including the 64/65 MoE dispatch boundary, 128/129 router boundary and expert
IDs above 255. The three actual format pairs are Q5_K/Q8_0, Q4_K/Q5_1 and
Q4_K/Q8_0 for gate-up/down respectively.

All 3,140 gathered expert groups match exactly: 9,420 isolated projections
compare 83,980,800 BF16 values; complete FFN activation/down outputs compare
69,984,000 more. The complete FFN uses original gathered input and raw expert
weights, without reference activations as inputs. The separate down-projection
test uses captured activations to isolate that operation. Original captures and
comparison-plugin patches are preserved. Full replay under GPU memcheck reports
zero errors and leaks. Full layer-3 batches 64 and 65 pass racecheck; selected
experts at layer-3 batches 64, 65, 257 and layer-2 batches 64, 65 pass both
racecheck and synccheck. A redundant all-expert batch-257 race run was deliberately
interrupted when narrowing this scope; its partial log is retained.

The API requires borrowed raw expert handles to remain valid through stream
completion, request-owned disjoint BF16 views and pre-reserved A8 scratch. Plan
validation checks all three projections before launching any FFN work. Guard
tests verify rejected plans preserve output/scratch sentinels and a valid retry
succeeds after reserving scratch. Quant capability metadata distinguishes this
MoE arithmetic from ordinary dense dispatch and decoded-weight GEMM.

Nsight captures of a warmed layer-3 expert at total batches 64 and 65 each show
six kernels on one stream: input quantization, gate and up projections, SiLU
product, activation quantization and down projection. Gate/up share quantized
input. Both traces contain no allocations, copies, memset or device fence during
submission; the test explicitly waits once after submission. These are structural
checks, not latency measurements or vLLM performance parity results.

The harness groups/gathers on the host and retains one expert's three weight
handles at a time. It does not establish GPU grouping performance, a total VRAM
bound, cache admission/eviction correctness or complete MoE execution. GPU
grouping, routing-weight BF16 rounding and reduction, shared-expert integration,
event-protected bounded placement and whole-model gates remain required.
Commands, binary/source hashes, numerical/sanitizer logs and traces are recorded
under `/tmp/bitnet-goal-resident-expert/`.
The protected clean CUDA build, standard suite, complete CUDA backend suite and
diagnostic isolation pass without compiler warnings. The final qualification
audit verifies all 513 oracle files remain unchanged and records current hashes.

## Resident GPU MoE grouping and routed composition (2026-11-08)

`test/reference_cuda_bf16_moe_group.c` checks optional SM120 grouping/gather and
weighted reduction against the same preserved Flash Next captures. Grouping uses
four kernels: reset, histogram, prefix scan and scatter/gather. It produces exclusive
expert offsets, grouped-slot and inverse maps; order within an expert is unspecified.
Reduction uses three kernels: inverse-bound validation, multiplication rounded to
BF16 in original slot order, then ordered F32 summation rounded to BF16. It consumes
the grouping status without host readback. Invalid device IDs set status mask 1;
invalid inverse bounds set mask 2. Invalid metadata suppresses downstream writes.

All 27 actual cases and 36 synthetic cases pass across three repeats, totaling
512,792,532 BF16 comparisons for gathered input, weighted slots and routed output.
The synthetic cases span 1/3/65/257 tokens, dimensions 1/37/2560, all routes to expert
511, distributed routing and concentrated IDs 0/256. They exercise duplicate IDs,
empty groups, bad device IDs and inverse bounds, metadata resets, valid retries,
request ownership, aliasing, alignment, extents and unchanged inputs. Host-plan
rejections preserve metadata/output sentinels. GPU memcheck reports zero errors or
leaks; racecheck and synccheck pass the complete 63-case replay.

`test/reference_cuda_bf16_expert.c ... --grouped` adds composition with the qualified
raw expert FFNs. It uses actual checkpoint weights and GPU-gathered input, then
compares native weighted slots and routed output with the independent captures.
All 27 cases match 61,585,920 BF16 values exactly; the full composition also passes
memcheck with zero errors or leaks. Captured down projections are not inputs to
this composition. Only 513 offsets plus one status word (2,056 bytes) return to the
host before expert staging. The harness explicitly completes every expert before
releasing its three borrowed weight handles. This establishes arithmetic and buffer
composition, not asynchronous cache lifetime correctness or a total VRAM bound.

Warm Nsight traces at layer 3/batch 257 show four grouping and three reduction
kernels respectively, each on one stream with no submission allocations, transfers,
memset or device-wide fence. Each capture has one explicit outer completion wait.
These traces are structural checks, not latency or vLLM parity measurements.

`make BN_ENABLE_CUDA=1 test_cuda_moe_group` runs the 36 synthetic cases without a
model or capture files. The complete CUDA test target chains this after the FP16
test and before the backend suite, keeping GPU probes sequential under `make -j`.
Unsupported optional precision profiles explicitly skip this test. Full commands,
logs, traces and hashes are under `/tmp/bitnet-goal-resident-group/`.

The current grouping materializes one BF16 input row per route. A production
prefill path still needs original-input quantization reuse, batched expert pointer
banks, stream/event-based bounded weight placement, shared experts and HC integration.
Default native model admission and whole-model performance qualification are unchanged.
The protected clean build and standard suite pass without warnings. The complete
CUDA suite, including the new synthetic target and diagnostic isolation, also
passes without warnings. The final audit preserves all 513 independent reference
files and records current source/binary hashes.

## Asynchronous bounded expert weight-bank qualification (2026-11-08)

The optional SM120 bank API reserves 1–4 fixed slots with 1–64 raw expert triples
per slot. Payload budgets apply separately to pinned host memory and device weight
memory, using the maximum supported pair (Q5_K gate/up, Q8_0 down). Actual supported
pairs use Q4_K/Q5_K gate/up and Q5_1/Q8_0 down. Each acquired lease reports actual
serialized gate/down sizes and per-expert stride. Matching reserve is idempotent;
resizing a live pool is rejected. A8 scratch and activation views are reserved
separately and are excluded from the bank budgets.

Acquisition returns BUSY while the slot's previous compute-completion event is
pending, without waiting or modifying the output lease. Generation and shape checks
reject stale/forged leases. Submission validates the complete bank before queuing
any upload, then uses one bulk H2D copy on a separate copy stream, an event dependency
on the compute stream, the expert FFNs and a completion event after their final
weight consumer. Thus pinned bytes and device weights remain protected through
compute completion, rather than only DMA completion. Request teardown completes
compute and copy streams before freeing the pool. Counters report reserved payload,
queued upload bytes, submitted banks/experts and busy acquisitions; busy counts are
poll counts, not transfer-stall durations.

`test/reference_cuda_bf16_expert.c ... --banked` streams all 27 preserved actual
cases through two banks of eight experts. `--banked1` repeats the same cases with
one expert per bank, stressing rapid reuse. Both configurations match 61,585,920
weighted/output BF16 values exactly, or 123,171,840 comparisons combined. Each run
processes 3,140 expert groups and uploads exactly 11,035,750,400 raw bytes. The
eight-expert run uses 408 bulk submissions; the single-expert run uses 3,140.
Reserved weight payload per memory tier is 63,897,600 bytes and 7,987,200 bytes
respectively. These figures bound bank payload, not total RAM/VRAM. The host checks
2,056 bytes of grouping metadata before staging each case; no activation readback
occurs between grouping, expert execution and resident reduction.

Both complete streaming replays pass memcheck with zero errors and leaks. The
model-free `test/test_cuda_moe_bank.c` covers exact/undersized budgets, invalid
geometry and formats, idempotent reserve, no resize, cancel/reacquire, stale leases,
whole-bank rejection before upload, insufficient scratch, view ownership/aliasing,
two-slot reuse, partial banks, changed format pairs, exact transfer counters and
teardown without an explicit completion wait. Its normal, memcheck, racecheck and
synccheck runs pass; actual layer-3/batch-65 matrix execution also passes synccheck.
The new `test_cuda_moe_bank` target follows the existing group test and precedes the
backend suite, retaining sequential GPU execution under `make -j`.

The layer-3/batch-257 trace streams 206 experts in 26 bulk copies totaling
632,832,000 bytes. It contains 1,236 FFN kernels (six per expert) on one compute
stream and H2D copies on a distinct stream. The trace records one compute-stream
wait and two event records per bank, no submission allocation, memset, event or
device synchronization, and one explicit outer completion wait. Union intersection
of copy/kernel intervals shows 986,753 ns of overlap in this single trace. This
demonstrates overlap; it is not a latency or vLLM parity benchmark.

The implementation still uses host group descriptors and launches per-expert FFNs.
It does not yet provide grouped matrix launches, original-input quantization reuse,
model-cache admission/eviction integration, cold/warm offload latency metrics or a
global serving memory budget. Shared experts, HC integration and whole-model
qualification remain open. Commands, original captures, binary/source hashes,
per-case counters, sanitizer logs and traces are under `/tmp/bitnet-goal-expert-bank/`.
The protected clean build, standard suite and complete CUDA backend chain with
diagnostic isolation pass without warnings. The final qualification audit verifies
unchanged independent captures, quant kernels and protected diagnostic binaries.

## Reusing original-input A8 quantization across expert banks (2026-11-08)

The optional `--prepared` and `--prepared1` reference modes prepare original BF16
token rows once, validate grouped slot bounds and map complete A8 blocks into
expert order. Gate/up projections then read the mapped blocks directly; down
projection inputs are still quantized from the computed expert activations.
The copied blocks preserve scale, original-input-sum metadata and quantized values.
Existing vector/matrix arithmetic and the total-batch 64/65 dispatch boundary are
unchanged. A zero preparation ticket retains legacy per-expert input quantization.

Setup explicitly reserves original/grouped A8 workspace with fixed capacity and
a separate payload budget. At 257 tokens and dimension 2560, this occupies
8,141,760 device bytes, separate from weight-bank payload and linear scratch.
Preparation validates request ownership, exact/disjoint views and capacity before
queuing work, then returns a request-local generation ticket. Subsequent banks
must match both the latest generation and prepared batch size. Repreparing is
ordered on the compute stream after previously submitted consumers. Grouping status
must be checked by the caller before bank submission; malformed slot maps set the
device status and suppress mapping rather than indexing outside input storage.

Both capacities (one and eight experts per bank) match all 27 actual cases exactly:
61,585,920 weighted/output BF16 comparisons each. The legacy eight-expert path
also matches all 27 cases on the current implementation. The model-free bank test
adds undersized/invalid workspace setup, idempotence/no resize, missing preparation,
bad views/maps, generation invalidation, different batch geometry and valid retries.
Both full prepared replays pass GPU memcheck with zero errors and leaks; lifecycle
memcheck/racecheck/synccheck and actual matrix-path synccheck also pass.

The layer-3/batch-257 trace has 206 expert groups, 26 bulk H2D copies and unchanged
632,832,000 raw upload bytes. Preparation contributes three kernels (map validation,
original-input quantization and A8 mapping); each FFN contributes five. Total
compute kernels are 1,033 versus the previous 1,236, with one original-input
quantization plus 206 down quantizations. The trace retains separate copy/compute
streams, no submission allocation or internal host fence, and one explicit outer
completion wait. Observed copy/kernel overlap is 1,486,245 ns in this single trace;
it is a structural check, not a throughput or vLLM parity benchmark.

The current grouping still materializes BF16 routed input for bank view validation,
and the prepared workspace materializes grouped A8 blocks. Grouped kernels that
index original A8 rows directly can remove those copies and reduce per-expert
launches. Model-cache prefill/decode policy, shared FFNs, HC integration, whole-model
gates and matched performance comparisons remain required. Commands, source/binary
hashes, preserved references and traces are under `/tmp/bitnet-goal-moe-input-reuse/`.
The protected clean build, standard suite and full CUDA suite with diagnostic
isolation pass without compiler warnings. The final audit verifies preserved
reference bytes, quant kernels and user diagnostic binaries, and records current
source/binary hashes.

## Grouped expert-bank CUDA launch qualification (2026-11-08)

`--grouped-bank` and `--grouped-bank1` in the expert reference probe select four
compute launches per bank: one gate/up projection grid, activation, down-input
quantization and down projection. The projection grids select expert and role,
with token/row tiling inside each expert. Launch width uses the largest active
row count in the current bank, and inactive tiles exit uniformly before shared
barriers. The original Q4_K/Q5_K/Q5_1/Q8_0 MMVQ/MMQ bodies, affine quantizer and
two-round BF16 SiLU product are extracted as shared device functions; source-body
hashes and independent numerical replay audit that extraction. Matrix/vector
selection still follows total MoE tokens through the 64/65 boundary.

Grouped setup explicitly budgets pinned descriptors, device descriptors and
grouped down A8 scratch. At capacity eight/two slots, descriptors occupy 192 bytes
per memory tier. At 257 tokens the grouped device workspace is 1,850,592 bytes,
including those descriptors; original/grouped input A8 storage remains 8,141,760
bytes separately. Raw weight-bank payload remains 63,897,600 bytes per tier.
These figures exclude public activation views, linear scratch and allocator/event
overhead. Each submission copies descriptors into stable pinned storage and queues
one descriptor upload alongside the raw weight upload; the existing ready/completed
events protect both descriptor and weight lifetimes. Validation rejection queues
neither upload.

Both grouped capacities match all 27 preserved actual cases exactly: 61,585,920
weighted/output BF16 values each. The prepared per-expert and legacy paths also
match all 27 cases on the extracted cores. Model-free matrix tests add 12 cases
over hidden widths 32/96/160 and all four gate/down format pairs. They use one-row
and five-row experts at separated offsets, exercise partial row/four-token tiles,
and verify zero-weight outputs plus untouched gap/suffix sentinels for gate, up
and down. Lifecycle tests cover separate workspace budgets, no resize, missing
preparation/workspace and invalid grouped flags. Both full grouped replays pass
memcheck with zero errors or leaks. Lifecycle/tail racecheck and synccheck, actual
layer-3/batch-65 racecheck, and layer-3/batch-257 plus layer-2/batch-65 synccheck pass.

The layer-3/batch-257 trace has 206 experts and 26 banks, with 107 compute kernels:
three preparation kernels and four per bank. The prepared per-expert path used
1,033 kernels for this case. Raw uploads remain 632,832,000 bytes; 26 descriptor
copies add 2,472 bytes, yielding 52 H2D copies. Event counts remain one wait and
two records per bank, with no submission allocations, memset, event/device fence
or internal host completion wait. The harness explicitly waits once at the end.

This trace observes zero copy/compute overlap. Kernel execution totals 7,120,357 ns,
H2D execution 11,234,604 ns, and the first-to-last device activity spans 45,583,314 ns.
The remaining span contains gaps consistent with the host staging/launch timeline;
these single-trace timings do not isolate their cause or establish throughput or
vLLM parity. Cache integration and matched end-to-end measurement are still needed.
Source hashes, preserved references, per-case accounting, sanitizer logs and the
trace are under `/tmp/bitnet-goal-grouped-bank/`.
The protected clean build, standard suite and full CUDA suite with diagnostic
isolation pass without compiler warnings. The final audit verifies independent
reference bytes, core-body extraction hashes and protected diagnostic binaries.


### Complete native Flash Next HC/FFN composition (2026-11-08)

The actual-weight composition probe now executes HC normalization/projections and
mixing, router/shared-gate projections, native top-k/grouping, bounded grouped
expert banks, ordered routed reduction, the shared Q8 FFN, gated addition, and HC
injection/combine. Only the original wide hidden state seeds neural computation;
expected intermediate activations and routing decisions are comparison outputs.
All 27 preserved cases (layers 3/2/47, token counts 1/5/17/63/64/65/128/129/257)
match 215,275,158 BF16 values and 65,610 routing values/indices exactly.

The original shared `SiluAndMul` uses one final BF16 rounding, whereas routed
experts use separate BF16 SiLU/product rounding. The previous shared two-round
assumption was incorrect. An explicit dense FFN activation profile preserves the
existing separate default and selects fused rounding for this shared branch.
Independent tests cover 524,288 inputs per profile. Accurate division also fixes
three BF16 sigmoid subnormal cases and 21 SiLU/product edge cases; HC scaled SiLU
and extreme-gate tests cover these boundaries. Reference captures remain intact.

The layer-3, 65-token trace has 165 expert groups in 21 banks, 506,880,000 raw
weight upload bytes, 1,980 descriptor bytes, one 2,080-byte metadata readback,
and two stream completion fences (metadata plus explicit outer completion).
Compute stays on one stream; copies use the separate transfer stream. No
allocation, memset, device-wide synchronization or event completion wait occurs
inside the captured execution. Operator setup is warmed, but weights still stream
on each replay: this is not a warm-cache or vLLM performance comparison.

The complete probe and model-free tests pass memory/synchronization checks;
model-free tests also pass race checks. Evidence and commands are preserved in
`/tmp/bitnet-goal-shared-hc/`. Production model/session integration, global cache
leases and placement budgets, short/long/chunked/resumed model gates, and matched
end-to-end vLLM measurements remain open. Audit modern dense-model activation
profile selection independently before changing its default or prior goldens.


### Production-owned Flash Next FFN layouts (2026-11-08)

The backend precision container now owns immutable BF16 router/shared-gate
projections and raw Q8 shared FFN weights for all 48 layers. Routed expert bytes
remain in CPU RAM/file storage; preparation copies only expert-map anatomy and
borrows any existing repacked CPU source. It does not upload all routed experts.
The shared FFN explicitly selects the independently qualified fused SiLU profile.

Preparation requires the same HC/PLE/GDN/QSA source container, validates every
layer before GPU uploads, rejects unsupported routing scales/weight formats and
expert geometry/offset overflow, and checks an aggregate GPU tensor budget.
Each raw F32 router upload is released after BF16 conversion. On any failure,
new allocations are released and the existing attention/HC components survive.
Layouts cannot be replaced while live and are released by backend model teardown.
Mock lifecycle tests inject all 15 upload and six decode failures across three
layers, check preflight failures without uploads, and verify exact budget edges.

Actual checkpoint preparation retains 376,750,080 FFN tensor bytes, with a
376,765,440-byte preparation peak. Including HC/PLE/GDN/QSA, declared payload is
7,132,468,352 bytes and peak is 7,132,483,712 bytes. These exclude allocator/context
overhead, CPU expert source, request activations and staging/cache budgets.
The full FFN composition probe's `--model-layout` mode borrows these production
router/shared handles. All 27 original cases still match 215,275,158 BF16 values
and 65,610 routing values/indices exactly; memcheck reports zero errors/leaks.
The clean build has zero warnings and the standard test suite passes.

Evidence is under `/tmp/bitnet-goal-flash-ffn-model/`. This qualifies model-owned
FFN preparation, not a production session FFN submit hook: the probe still drives
operator composition and expert-bank staging. Request orchestration, global
cache leases/budgets and cold/warm placement policies remain open, along with
whole-model gates and matched vLLM performance/serving acceptance.


### Production Flash Next FFN request and RAM-cache policy (2026-11-08)

`bn_transformer_bf16_flash_ffn_submit` now executes the full native FFN inside the
production HC transaction, using immutable backend-model weights and request-owned
scratch. Preparation reserves two cap-8 expert banks, input/down A8 workspace,
routing and activation buffers under explicit host/GPU budgets. No resident
allocation or host activation upload occurs during submit. One packed 2,080-byte
routing metadata read selects expert groups. BUSY returns preserve the cursor and
prepared-input generation, allowing caller retry without rerouting or internal
spinning. HC's raw/normalized wide states survive until injection/combine.
FFN positions commit only after successful outer completion; failures poison the
request while retaining old committed positions. Model admission remains unchanged.

The existing MoE source-lease API is exposed through `moe.h`, with const I/O
control access. Prefill borrows RAM-cache hits without promotion/admission;
decode admits misses. Source leases protect synchronous copies into pinned banks
and are released immediately after those copies. Bank completion events protect
the copied payload through GPU use. Pread misses can read directly into these
pinned slots, without another expert-sized host staging allocation. Shared cache
access across requests remains executor-serialized. This adds no GPU expert cache.

Both mmap and split-file pread pass all 27 preserved numerical cases: 215,275,158
BF16 values and 65,610 routing values/indices exactly per mode. Each mode streams
11,035,750,400 raw bytes in 408 banks across these cases. Forced BUSY tests verify
one routing read/input preparation per block, no premature HC combine/commit, and
healthy retry. Allocated resident payload plus backend workspace/bank accounting
matches declared GPU bytes for every case. Exact budget edges, partial-preparation
rollback, queued-work failure, outer-fence failure and interleaved request isolation
are tested. One persistent request also runs 5-token and 65-token prompt chunks
then one decode token, reaching position 71 with three input generations; these
are independent FFN fixture inputs, not a full-model continuation qualification.

With a fresh 64 MiB shared RAM-cache budget, a one-token decode has 10 misses and
30,720,000 file-read bytes; the next request has 10 hits and zero file-read bytes.
Subsequent prefill borrows all 10 hits. The initial 27 prefill cases leave the
empty cache unpopulated. Cold/warm here means RAM-cache state, not cold OS pages
or storage. GPU upload is still 30,720,000 bytes for each one-token replay.
Per-request raw bank payload is 63,897,600 bytes per host/device tier, with 192
bytes of descriptors per tier. Input A8 is 31,680 bytes/token, down A8 is 7,200,
and shared-linear A8 is 2,880; additional resident FFN tensors are 198,348
bytes/token plus 4,128 bytes. Host metadata read storage adds 2,080 bytes. These
exclude model weights, RAM-cache/source pages, allocator/context and C metadata.

Request diagnostics report source/file/cache bytes, bank uploads, metadata read
wall time, source fetch/copy wall time and BUSY-to-retry spans. The latter include
caller scheduling delay and are not isolated DMA stall measurements. Mmap/pread
request tests pass memcheck with zero errors/leaks; synchronization checks pass.
Clean build and standard tests pass without compiler warnings. Evidence is under
`/tmp/bitnet-goal-flash-ffn-request/`.

Whole-model short/long/chunked/resumed gates, production token embedding/output
and complete layer orchestration, global VRAM placement/leases, aggregate serving
admission, matched resident vLLM performance and continuous batching remain open.


### Flash Next vocabulary endpoints and explicit model driver (2026-11-08)

The checkpoint has separate Q8 token/output tables and an output HC mixer, with
no additional final norm. The pinned original model repeats input embeddings over
four HC streams. Native Q8 selected-row embedding now declares its quant capability
in the registry and preserves the original FP16 scale/product rounding before
BF16 conversion. Its independent oracle covers every FP16 scale bit pattern paired
with every signed 8-bit quant, plus token-count tails: 16,816,384 output words.
NaN payloads are canonicalized only for synthetic hash comparisons. Finite values,
infinities and signed zeros remain exact. Invalid IDs poison only their request;
peer requests and released/reinitialized containers remain healthy. The existing
Q4 embedding reference remains unchanged.

The immutable model container now owns both raw vocabulary tables under an
aggregate preparation budget, with rollback at either upload failure and no
decoded vocabulary expansion. Their payload is 1,350,860,800 bytes; total prepared
HC/PLE/GDN/QSA/FFN/vocabulary payload is 8,483,329,152 bytes. Token IDs, BF16 logits
and the embedding status word add 4 bytes/token + 496,640 + 4 GPU bytes to requests;
retained model-driver token copies add 4 host bytes/token. Existing scratch holds
embedding repetition and output HC projection stages. Declared GPU bytes match
tracked resident allocations plus opaque backend workspace for all endpoint cases.

Production token-submit and head-submit APIs now perform device-resident lookup,
four-stream repetition and output HC mixing followed by a last-row Q8 vocabulary
projection. Five actual-checkpoint cases (1/5/17/129/257 tokens) match 11,712,000
BF16 words and 409 token IDs exactly against fresh pinned references. Head inputs
come from preserved independent FFN captures; native tests execute the production
FFN before the head, preserving the original references. No extra output norm or
CPU activation bridge is introduced. The independent oracle was replayed under
memcheck into a fresh directory: all 35 data files, header and manifest match.

An explicit model begin/submit API connects token input, PLE at its configured
layer before attention, all 48 HC attention/FFN pairs and the output head. It
copies token IDs into prepared request storage and retains its layer/phase cursor
across expert-bank BUSY returns. The successful submit leaves head work queued;
outer completion commits all recurrent, PLE and FFN positions. Ordinary model
admission is unchanged pending independent whole-model qualification.

A smoke test executes one-token prefill and subsequent decode through the full
driver, with forced bank retries, stable copied token IDs, 48 then 96 accumulated
FFN blocks, finite logits across all 248,320 vocabulary entries and committed
positions 1 then 2. Argmax IDs are 11 and 11 for the initial token 116. These IDs
are recorded observations, not validated expected model tokens. Component accuracy
and finite full-model execution do not prove full-model token correctness.

Clean build/standard tests pass without compiler warnings. Native endpoint,
embedding and full-driver smoke tests pass memcheck with zero errors/leaks;
full-driver synchronization checks pass. Evidence is preserved under
`/tmp/bitnet-goal-flash-embedding/`. Independent whole-model short/long/chunked/
resumed token gates and common session/CLI admission remain open. Global VRAM
expert placement, aggregate serving budgets, matched vLLM performance and batching
acceptance also remain open.

The complete CUDA regression suite also passes on this checkpoint, including the
new Q8 embedding cases and existing Q4 embedding, MoE and diagnostic isolation tests.

Source audit found that the host vLLM checkout differs from the pinned image.
The relevant pinned model ordering was inspected directly, and the common HC
normalization/mix/combine math methods match the inspected host methods. Future
whole-model references must use the pinned image sources preserved in this evidence directory.


### Independent Flash whole-model references (2026-11-08)

The explicit Flash driver now matches independent original-operator references
through all 48 layers, rather than only producing finite smoke-test logits.
Reference inputs begin at checkpoint token embeddings and continue through PLE,
HC attention/FFN blocks and the output head. The pinned vLLM image supplies the
original neural operators; the already documented corrected GGUF extension
supplies raw expert/vocabulary operations. Core matrices use the qualified
decoded-BF16 profile. This is an operator correctness comparison, not a served
vLLM GGUF adapter or a performance benchmark.

Qualified schedules are one-token prefill plus decode, the five-token plain-text
prompt `The capital of France is` plus four greedy decode steps, that prompt split
2+2+1 plus four decode steps, 129-token prefill plus decode, and 1,025-token prefill
plus four decode steps. Comparisons are exact at every attention/FFN wide-hidden
boundary. Additional checks cover selected GDN intermediates/F32 states, QSA
projections/attention/indices, expert IDs/weights, PLE history, output HC mixing
and all vocabulary logits. Each schedule also runs independently through the
production begin/submit/complete driver with fresh request state. The 1,025-token
case compares 1,125,382,624 BF16 words and 6,064,248 F32/I32 words across the two
execution paths, with zero differences. Each chunk schedule retains its own
reference; cross-schedule floating-point identity is not assumed.

A real second-chunk cancellation exposed two original GDN output contractions.
Enumerating all 24 supported original output-kernel configurations reproduced both
behaviors using identical prepared inputs. Two-warp/BV32 configurations round the
scaled local term before a history FMA; the other inspected configurations round
scaled history before a local FMA. The resident plan now names that arithmetic
contract. Existing zero-initialized callers retain local FMA; Flash explicitly
selects history FMA. Future whole-model captures fix the supported original
BK64/BV32/two-warp/three-stage configuration so autotune timing cannot change the
reference contraction. No input-value or token-count heuristic selects the profile.

The new model-free cancellation fixture checks both contracts, F32 next-state
bits and immutable initial state. All 35 existing resumed GDN prefill calls still
pass, and the previous 68 independent GDN output/state files and all earlier whole
model reference files retain their recorded hashes. The fixture generator and
full-model generator are `test/reference_cuda_bf16_gdn_cancellation.py` and
`test/reference_cuda_bf16_flash_whole.py`; the comparison harness is
`test/reference_cuda_bf16_flash_ffn.c --model-reference`.

Evidence is preserved under `/tmp/bitnet-goal-flash-whole/`. Common session/CLI
admission, larger context coverage, global expert VRAM placement and serving
budgets, continuous batching, and matched vLLM performance acceptance remain open.


The final production sources build cleanly with zero compiler warnings; standard
and complete CUDA regression suites pass. Both output profiles pass the new
cancellation regression. Native cancellation and complete short chunked model
runs pass memory and synchronization checking with zero errors (memory checking
also reports zero leaked allocations). The independent original-operator oracle
passes memory checking and reproduces every preserved short chunked reference
file exactly, including all recorded source-weight hashes.


The long resumed whole-model gate also passes: 1,025 prompt tokens split into
seven 129-token chunks and a final 122-token chunk, followed by four decode steps.
Both execution paths agree exactly with their schedule's independent reference:
1,129,719,264 BF16 words and 11,569,272 F32/I32 words, zero differences. This closes
the 1K-context chunked/resumed accuracy gate for the explicit Flash driver;
larger-context and ordinary runtime admission gates remain open.


These whole-model comparisons use mmap-backed expert sources. Bounded pread
source-cache behavior retains its earlier component qualification; a full-model
pread comparison remains open. The tested contexts also do not reach QSA's
2,048-token selection/pruning boundary (512 selected groups, compression ratio 4).
That boundary and longer contexts need independent whole-model gates before
claiming general long-context admission.


The 1,025-token uninterrupted and eight-chunk native whole-model runs also pass
memory checking with zero errors and zero leaked allocations. The eight-chunk
run passes synchronization checking with zero errors. Final source, binary, log,
reference and preserved user-diagnostic hashes are recorded in
`/tmp/bitnet-goal-flash-whole/qualification.json`.


### Flash whole-model bounded pread cache qualification (2026-11-08)

The independent 48-layer Flash comparison harness now also supports
`FIRST_SHARD CAPTURES --model-reference-pread REF DUMP [CACHE_MB]`.
It selects shard-file pread explicitly, disables expert mmap access, derives each
cache projection stride from the maximum actual checkpoint projection size, and
verifies every layer fits. Source-weight files remain mapped for model/core
loading; this is a bounded expert-cache qualification, not a total process RSS
limit or a cold-storage benchmark.

All preserved schedules pass exactly through both layer-by-layer execution and a
fresh production driver: one-token and five-token prompts, 2+2+1 chunked prefill,
129-token prefill, 1,025-token prefill, and 1,025 tokens split 129*7+122. Each
continues through its original decode schedule. No expected values or numerical
profiles change. The existing mmap whole-model comparison and all 27 legacy FFN
request cases also pass.

The harness snapshots cache membership without promoting entries or retaining
leases. Prefill must preserve membership; decode must admit misses and account
for evictions. Every phase checks expert hit/miss counts, pread projection counts,
and `pread bytes == RAM miss bytes`, with RAM hit plus miss bytes equal to raw
GPU upload bytes. Request GPU/host reservations remain constant across phases;
raw bank reservations are 63,897,600 bytes in each host/device tier. This checks
RAM payload ownership and existing event-protected streaming banks; it does not
introduce a model-wide expert VRAM cache or aggregate serving-budget accounting.

| Expert payload budget | Allocated payload | Slots | Observed behavior |
| --- | ---: | ---: | --- |
| 64 MiB | 63,897,600 bytes | 16 | First decode admits 480 experts, retains 16 and evicts 464; later decode continues evicting. Prefill borrows without admission. |
| 2 GiB | 2,144,563,200 bytes | 537 | Retains the full 480-expert decode working set across fresh requests. |

The 2 GiB replay's first decode has 480 RAM misses and reads 1,504,256,000 bytes.
Repeating the same decode in a fresh request has 480 RAM hits, zero misses and
zero pread bytes, with identical logits. Its intervening prefill borrows 172
cached experts without changing the 480 cached entries. Both cold and warm
decodes still upload 1,504,256,000 bytes to the GPU: RAM hits are not VRAM hits.
This transfer remains a concrete target for the planned bounded VRAM placement.
Here cold/warm describes the expert RAM cache; filesystem-cache state is not
controlled.

Per-phase logs include raw uploads, source reads, cache hits/misses/evictions,
metadata read bytes, bank counts/retries, request reservations and source/read
wall times. The bank retry spans include caller scheduling delays, and metadata
read timing can include preceding stream work. Observer reads and sanitizer
instrumentation also affect these tests. These are correctness/accounting traces,
not isolated transfer-stall measurements or vLLM performance results.

The clean build and standard tests pass without compiler warnings. Native
1,025-token chunked pread and fully warm replay pass memory checking with zero
errors and zero leaked allocations; chunked pread passes synchronization checking.
Evidence and immutable-reference hashes are under
`/tmp/bitnet-goal-flash-pread-whole/`. Larger-context pruning, common session/CLI
admission, bounded expert VRAM residency, aggregate serving budgets, batching,
and matched vLLM prefill/decode performance remain open.


The complete CUDA regression suite and diagnostic isolation checks also pass on
the final rebuild. Final qualification, per-phase cache/transfer metrics,
source/binary/log hashes, unchanged independent-reference hashes and preserved
user diagnostic binaries are recorded in
`/tmp/bitnet-goal-flash-pread-whole/qualification.json` and `metrics.json`.
This closes the explicit Flash driver's full-model pread accuracy/accounting gate
through 1,025 tokens; ordinary CLI admission and performance gates remain open.


### Flash 4K pruning under declared execution profiles (2026-11-08)

The explicit 48-layer Flash driver now passes independent whole-model gates at
and beyond QSA pruning, in both mmap and bounded pread modes. The reference
contract names the existing native numerical/selection policies: four-warp
triangular inversion, the previously pinned history-FMA output configuration,
and exact score rank with lower-index ties and chronological QSA group emission.
This does not establish literal bit equality with an unmodified, arrival-ordered
vLLM execution, and does not yet enable ordinary Flash CLI admission.

The first uncontrolled 4,097-token capture exposed a GDN inversion configuration
difference. Native and a fresh original-operator replay agree exactly at every
captured recurrence stage, including F32 state, on identical prepared inputs.
Forcing the original two-warp inversion reproduces the uncontrolled layer-0
output/state exactly; four warps reproduces native exactly. The two profiles
differ at 270,007 of 25,171,968 BF16 output values and 59,665 F32 state values.
The failed literal native comparison and original capture remain preserved;
no production kernel or expected file was changed to hide this difference.

QSA also has an existing selection-policy distinction. In the controlled 4K
prefill's first sparse layer, all 2,046 pruned rows choose the same groups as the
original operation, but their emission order differs. This changes 241,678 BF16
attention values and 1,373,640 output-projection values on the same neural input.
The reference retains the original choices, F32 scores and original attention/
projection outputs separately. It verifies validity and score-optimality before
independently applying the declared chronological policy and calling original
attention. Canonical ranking uses NumPy; native uses radix threshold/compaction.
A separate verifier checks cutoff inequalities and tie-ID ordering without
reusing the generator's sorting algorithm.

`test/reference_cuda_bf16_flash_whole.py` adds explicit
`--gdn-inverse-warps {2,4,8}` and `--qsa-selection {original,canonical}` options.
The default inversion remains autotuned and selection remains original. New
controlled captures use `--gdn-inverse-warps 4 --qsa-selection canonical`, with
these settings recorded in their manifests. Diagnostic scores may contain
masked nonfinite cells outside the visible prefix; visible scores and every
neural output still undergo finite-value checks. All earlier references remain
immutable. A fresh controlled five-token/four-decode capture preserves every
previously recorded short neural value and source-weight hash exactly.

| Whole-model schedule | Compared BF16 words | Compared F32/I32 words |
| --- | ---: | ---: |
| 2,047-token prefill, eight decode steps across first pruning | 2,246,873,760 | 11,335,848 |
| 4,097-token prefill, four decode steps | 4,475,877,856 | 12,429,432 |
| 4,097-token prompt split 257*15+242, four decode steps | 4,485,170,656 | 24,225,912 |

Each row includes all attention/FFN layer boundaries and selected recurrent,
indexing, routing, PLE and head intermediates, followed by a fresh complete
production-driver replay. Every compared word matches exactly in each I/O mode.
Original ranking validity/score-optimality is independently checked on 49,248
pruned query rows across these three reference trajectories. All pruned rows
change order; two change membership only among exactly tied cutoff scores in
the chunked trajectory. These original alternatives are retained, not used as
interchangeable bit-exact expected model values.

The clean build and standard suite pass without compiler warnings. Evidence,
including uncontrolled and controlled captures, original inversion audits,
ranking checks and preserved failure artifacts, is under
`/tmp/bitnet-goal-flash-pruning/`. No production source changes in this step.
Common session/CLI integration, bounded expert VRAM residency, matched vLLM
performance, continuous batching, later models and backend qualification remain
open. Literal unmodified-vLLM 4K bit parity remains explicitly unqualified.


Final native memory checking passes for both uninterrupted 4,097-token and
sixteen-chunk pread trajectories through layer-by-layer and fresh production
driver execution: zero errors and zero leaked allocations. The chunked trajectory
also passes synchronization checking with zero errors. The complete CUDA suite
and diagnostic isolation pass on the final warning-free rebuild. All production
source hashes and four user diagnostic binaries are preserved. Final checks,
reference/source/binary/log hashes, ranking verification, original inversion
audits and the explicit literal-parity limitation are recorded in
`/tmp/bitnet-goal-flash-pruning/qualification.json`.

### Flash CLI transfer and placement baseline (2026-11-08)

The ordinary native BF16 CLI was measured on the dedicated RTX PRO 6000
Blackwell Max-Q (`GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05`), with pread,
a 2 GiB explicit RAM cache, default 1,025-token batches and temperature zero.
Short input is the five-token frozen prompt; long input is the frozen
4,097-token prompt, split 1,025/1,025/1,025/1,022. One warmup precedes the
reported runs. OS file-cache state is uncontrolled; these are not cold-disk
measurements or vLLM parity results.

Three unprofiled runs give 349.4–358.9 ms for the short prompt and
4,596.5–4,600.8 ms for the long prompt (about 891 prompt tokens/s). CLI reported
output rates are 7.48–7.58 tokens/s for eight short-case output tokens and
8.18–8.25 for five long-case output tokens. The reported numerator includes the
first token sampled from prefill, so these rates are not isolated decode
latencies and cannot establish the 5% vLLM gate.

Nsight CUDA traces separate preparation from embedding-launch windows:

| Operation | Streamed expert bytes | Active expert groups | Routing metadata reads |
| --- | ---: | ---: | ---: |
| Five-token prefill | 4,941,926,400 | 1,574 | 48 |
| First 1,025-token chunk | 18,959,257,600 | 6,055 | 48 |
| Next 1,025-token chunk | 14,189,363,200 | 4,529 | 48 |
| Third 1,025-token chunk | 14,290,329,600 | 4,560 | 48 |
| Final 1,022-token chunk | 13,963,468,800 | 4,452 | 48 |
| Each decode step | 1,504,256,000 | 480 | 48 |

Every routing read is 2,080 bytes; ordinary windows contain approximately
50–51 stream fences. Endpoint differences include the next token upload or
the final transaction fence. Long prefill selects grouped matrix kernels and
Q8 MMQ; decode selects grouped small-batch kernels and MMVQ. Batched prefill
uploads each active expert once per layer/chunk, rather than once per routed
token. The tracked GPU activity union occupies roughly 82–89% of long-prefill
windows and about 25% of decode windows; this is neither SM occupancy nor an
isolated request-latency measurement. Windows include caller work and between-run
restart, and profiler overhead is present.

An all-CUDA-API/memory trace observes zero device-heap or pinned-host allocation
events inside ten inference windows. Tracked peak device heap is 9,790,628,712
bytes and pinned host payload 67,046,592 bytes for the short, capacity-2,048
configuration. Both tracked heaps balance to zero on teardown; context static
storage is separate. A coverage-instrumented CLI independently records zero
CPU/legacy orchestration entries across 13 ordinary forward/prefill calls,
including three no-logits chunks. Both frozen token trajectories match.

`bench/analyze_flash_trace.py` reproduces trace extraction from Nsight SQLite
exports. The export includes all captured routing and bank transfer counts;
preparation is excluded, while final teardown is excluded at the completion
fence. Production sources are unchanged from the qualified session/CLI build.

#### Cache placement implications

`--trace-routing` on the independent whole-model oracle now retains IDs for all
48 FFN layers. The native comparator matches every captured routing word and
the existing model values: 32 phases, 4,491,225,056 BF16 and 16,716,198 F32/I32
words. All 1,180 prior neural files retain their values; original QSA diagnostic
choices remain explicitly nondeterministic. The new 376 route files are added
in a separate capture. The zero-cache placement replay exactly reproduces
measured expert counts and upload bytes for all eight phases.

The raw expert set is 77,017,907,200 bytes (71.73 GiB). Reserving the largest
3,993,600-byte slot for every expert instead needs 98,146,713,600 bytes
(91.41 GiB), which prevents full residency on this GPU after core/request state
is included. Compact size-class pools preserve full-residency feasibility.
The checkpoint has 43 layers with 3,072,000-byte experts, four with
3,584,000-byte experts and one with 3,993,600-byte experts.

`bench/simulate_flash_vram_cache.py` compares global fixed-slot LRU with compact
size-class pools, pins all current-layer hits, and uses separate prefill/decode
policies. Over the four captured decode steps, baseline upload is 6.017 GB:

| Simulated placement/prefill policy | Budget | Cold decode upload | Third replay decode upload |
| --- | ---: | ---: | ---: |
| Largest slots / borrow only | 2 GiB | 4.395 GB | 3.958 GB |
| Compact pools / borrow only | 1.5 GiB | 4.395 GB | 3.958 GB |
| Compact pools / borrow only | 2 GiB | 4.391 GB | 3.954 GB |
| Compact pools / retain prompt tail | 2 GiB | 4.101 GB | 4.108 GB |

Tail admission helps cold decode here but is not uniformly better on warm
replay. This model accounts for bank-to-cache device copies on tail admission.
Smaller largest-slot caches
thrash across the full layer cycle. These are placement simulations over a
four-step reference trajectory, not implemented GPU residency, general cache
hit-rate predictions or measured speedups. Long decode and broader prompts
remain necessary before choosing defaults.

The next implementation needs model-owned compact pools, event-protected
request leases, cached-weight grouped execution, and separate GPU/RAM hit,
miss and upload accounting. The legacy GPU expert cache has immediate handle
eviction and fixed entry accounting, so it cannot be attached unchanged to the
resident request path. Evidence, commands, source/binary hashes, coverage,
memory events and preserved captures are under `/tmp/bitnet-goal-flash-profile/`.


### Implemented Flash compact cache: local qualification (2026-11-08)

Explicit `--gpu-cache-mb 2048` enables compact model-owned expert pools in the
native CUDA BF16 Flash path. Default zero preserves the uncached control.
This uses the same headless RTX PRO 6000 Blackwell Max-Q UUID as the earlier
baseline. The 2 GiB budget reserves 2,147,328,000 bytes: 613 small, 57 medium
and 15 large raw expert slots, with no quantization or routing changes.
Prefill borrows without admission/promotion; decode admits into event-protected
LRU pools. Stats separate GPU misses from subsequent RAM hits/misses and disk
reads. Cache hits do not touch the RAM LRU. Miss fills currently perform a
host-to-bank upload followed by an equal-sized bank-to-cache device copy.

All 32 phases in the frozen all-layer routing reference match exactly:
4,491,225,056 BF16 words and 16,716,198 F32/I32 words, zero differences. Actual
hit/miss counts/bytes, resident payload, evictions and pressure bypasses match
`bench/simulate_flash_vram_cache.py` compact-pool/borrow replay. Initial decode
uploads 4,391,116,800 bytes over four steps; repeated warm decode uploads
3,954,483,200 versus 6,017,024,000 uncached (34.28% less). Warm prompt uploads
54,134,272,000 versus 61,402,419,200 bytes. These are the captured trajectory,
not general hit-rate predictions. Decode fill device copies equal its miss
bytes; prefill admission/device fill copies are zero.

Local timing uses the same 4,097-token prompt, default four-chunk schedule,
five generated tokens, one warmup and three measured runs per case. Sessions
restart, while shared source/GPU caches remain warm. Each pair holds the RAM
budget fixed. Values below are medians; raw samples and commands are retained.

| RAM cache | GPU cache | Prompt ms | CLI reported output tokens/s |
| --- | ---: | ---: | ---: |
| 2 GiB | 0 | 4611.7 | 8.17 |
| 2 GiB | 2 GiB | 4413.2 | 9.62 |
| 4 GiB | 0 | 4578.4 | 8.84 |
| 4 GiB | 2 GiB | 4343.7 | 9.78 |

The relative reported output-rate gains are 17.75% and 10.63%; prompt time falls
4.30% and 5.13%. CLI output rate counts the first token sampled from prefill,
so it is not isolated decode latency. Fixed case order, three samples and a
four-step repeated decode trajectory limit these measurements. They establish
local benefit, not full-residency performance or the 5% vLLM goal.

The independent numerical reference retains the documented GDN inverse-4 /
history-FMA / canonical-QSA qualification profile; it does not establish literal
unmodified-vLLM bit parity or invariance across different prefill chunk schedules.
Synthetic cache tests, entire CUDA/CPU suites, dense Qwen27 regression, bounded
pread/mmap, CLI/chat resets, preparation rollback and failed-session isolation
pass. Compute Sanitizer reports zero errors/leaks for cache unit and real-model
short execution and zero synchronization errors for long CLI execution.

A separate allocation-only run reserves all 77,017,907,200 expert bytes
(73,450 MiB) and returns the expected first token. Its pool starts empty and
prefill only borrows, so this is capacity evidence, not full preloading or a
fully resident benchmark. Full preload, larger prompt/decode sets, cache policy
selection and matched vLLM timing remain open. Current evidence is under
`/tmp/bitnet-goal-flash-vram/` and earlier reference/profile evidence is preserved.


The cached short CLI all-API/memory trace contains ten inference windows
(two restarted runs, five phases each). Nsight independently records
17,448,550,400 raw upload bytes, exactly matching cumulative production counters;
all 6,988 expert groups remain present. The extra pointer-table traffic is
55,904 bytes. Cache fills are counted as separate device copies. Routing still
requires 48 metadata reads per phase and roughly 50–52 stream fences; caching
has not removed this control-path cost. Trace activity durations include
profiler overhead and do not independently measure critical-path DMA stalls.

Captured runtime APIs and memory events show zero inference allocations.
Tracked device-heap peak is 11,937,956,840 bytes and pinned-host peak is
67,046,720 bytes; both return to zero at teardown. Relative to the earlier
uncached trace, these increase by exactly the 2,147,328,000-byte pool plus
128 device pointer-table bytes, and 128 pinned-host pointer-table bytes.
Static CUDA module allocations are tracked separately. These counters cover
captured API/memory events, not unobserved driver-internal allocations.
`bench/analyze_flash_trace.py --expert-cache` parses pointer/descriptor pairs
and device fills explicitly, including phases without raw expert uploads.
The earlier uncached capture also passes the revised parser.


### Flash fully resident expert execution (2026-11-08)

Enable explicit full preloading with:

```sh
./bitnet MODEL --cuda-bf16 --gpu-cache-mb 73450 --gpu-preload-experts \
  --pread --cache-mb 64 --maxseq 8192
```

The preload API rejects partial capacity, insufficient staging budget, nonempty
pools and any bound requests before reading sources. A single pinned buffer
holds the largest raw expert (3,993,600 bytes); every payload is uploaded and
completed before the buffer is reused. All expert metadata is published only
after success. Source-read failure leaves every key absent and permits retry;
CUDA transfer failure poisons the cache. Setup allocates/frees staging and a
temporary stream before request preparation. Model-owned GPU pool ownership and
request lifetime rules remain intact. Source orchestration stays above
backend_layout, using existing checked GGUF/pread leases without RAM admission.

Real-model preload reads/uploads exactly 77,017,907,200 bytes for 24,576 experts
(73,728 projections). One observed setup took 4,235.8 ms with the filesystem
cache already warm; this does not measure cold storage. Every long-reference
phase has all experts present before execution, including the initial prompt.
All 32 phases match 4,491,225,056 BF16 and 16,716,198 F32/I32 reference words
exactly, including all-layer routing IDs. Every phase records zero expert
misses, file reads, uploads, device fill copies and evictions. The 20-phase
short reference, failure recovery and peer isolation also pass. Independently
captured goldens and the explicit GDN inverse-4 / history-FMA / canonical-QSA
qualification limits are preserved. A separate 32-token continuation matches
uncached native token IDs; those additional tokens establish cache equivalence
rather than an expanded independent oracle.

A new local comparison fixes the RAM cache at 2 GiB, uses the same 4,097-token
prompt, default 1,025-token chunks and five output tokens, then restarts the
request for one warmup and three measured runs. Shared caches remain warm.
Medians on the designated RTX PRO 6000 Blackwell Max-Q are:

| Expert placement | GPU pool | Prompt ms | CLI reported output tokens/s |
| --- | ---: | ---: | ---: |
| Stream | 0 | 4656.0 | 8.14 |
| Lazy compact cache | 2 GiB | 4397.6 | 9.60 |
| Explicit full preload | 73,450 MiB | 3368.0 | 88.52 |

Preloading/model preparation are outside these prompt/output intervals and
logged separately. CLI output rate counts the first token sampled from
prefill; it is not isolated decode latency. Three samples, fixed case order,
a repeated four-step decode trajectory and concurrency one limit conclusions.
The fully resident path still needs prefill work and a matched vLLM engine
baseline before assessing the 5% parity criterion. The existing vLLM operator
oracle is a numerical reference rather than a served-engine timing baseline.

The short all-API/memory trace contains ten inference windows and all 6,988
expert groups. It records zero raw expert upload/fill bytes, zero large H2D
copies across all streams, and zero inference allocation calls/events.
Tracked device-heap peak is 86,808,536,040 bytes; pinned-host peak is
67,046,720 bytes. Both return to zero. Static CUDA module data is separate.
The preload staging allocation is released before the larger request banks
are allocated. These figures cover captured API/memory events and exclude
unobserved driver-internal allocation. Routing still reads 48 metadata blocks
per phase. The analyzer accepts explicit validated bank stream IDs for traces
where full residency removes the raw uploads previously used for discovery.
Pointer/descriptor sequences and total expert count are cross-checked; the
prior uncached and lazy-cache parser regressions pass.

Protected clean build and CPU-only build have zero warnings; standard/CUDA
suites, dense Qwen27 and Flash CLI checks pass. Compute Sanitizer reports zero
errors/leaks for synthetic cache/preload tests and full-resident real-model
short execution, and zero synchronization errors for the long CLI. All raw
commands/samples, checks, source/binary hashes and references are preserved in
`/tmp/bitnet-goal-flash-preload/`.


The long resident all-API/memory trace covers two restarted runs: eight prefill
windows (8,194 input tokens) and eight decode windows. All 43,032 expert groups
are present; expert uploads, fill copies and inference allocations remain zero.
Prefill emits 40,656 kernels across both runs. Summed GPU kernel duration is
6,803.91 ms; activity union is 6,800.84 ms within 6,913.58 ms of phase windows
(98.37%). This is activity coverage, not SM occupancy. Grouped Q4_K gate/up
accounts for 3,233.10 ms and Q5_1 down for 1,670.51 ms, together 72.07% of
summed prefill kernel time. These measured quantized matrix kernels are the
next prefill target. The eight decode windows contain 20,792 kernels and
100.26 ms summed kernel duration; their 122.10 ms windows include caller work
and restart, so they are not isolated decode latency measurements. Full kernel
aggregates and phase data are in `resident-long-summary.json` and
`resident-long-analysis.json` under the evidence directory.


### Residency scope for the parity benchmark (2026-11-08)

The Flash measurements above qualify complete **expert** residency. The
28,800,138,240-byte PLE embedding table remains host-owned; selected rows are
gathered into bounded upload leases in `src/transformer/bf16.c`. Zero expert
uploads therefore does not mean zero model-weight transfers. The checkpoint
contains 111,323,630,080 raw tensor bytes (5,505,584,640 core,
28,800,138,240 PLE, 77,017,907,200 experts), exceeding this GPU's capacity even
before decoded layouts and request state. The current Flash case does not
satisfy full-model GPU residency for the parity criterion. A matched Flash
comparison must account for the same PLE host placement explicitly.

The Qwen3.8-27B GGUF is `qwen35`, has 17,548,181,504 raw tensor bytes and no PLE
tensors. The pinned GGUF plugin includes a Qwen3.5 adapter, making this the next
fully resident engine-comparison candidate. Engine/config/precision and memory
qualification still need to be performed; the vLLM operator oracle is not an
engine benchmark. Inventory and source evidence are recorded in
`/tmp/bitnet-goal-flash-mma/residency-scope.json`.


### Grouped integer tensor-core prefill (2026-11-08)

Native grouped Q4_K/Q5_K gate/up and Q5_1 down matrix projections now use
integer tensor cores. Q4_K/Q5_K share the previously qualified dense affine
core; the dense wrapper preserves its geometry. Q5_1 retains FP16-rounded
scale/minimum products and sequential F32 accumulation over 32-value blocks,
including down widths not divisible by 256. Grouped token tiles are 8, 24 or
64 according to the largest group in the bank. The original total-token
MMVQ/MMQ precision selection is unchanged. Vector/decode arithmetic, Q8_0
matrix execution, routing, source/cache policy and memory budgets are preserved.
No persistent packed weights or inference allocations are introduced.

A new nonzero differential compares every output/gate/up word with the retained
per-expert DP4A path: 160 combinations of gate/down formats, hidden widths
32/96/160/640 and uneven group lengths 1/7/8/9/23/24/25/63/64/65. Active values
must be finite and nonzero, while inactive rows retain sentinels. The same cases
pass memory and synchronization checking. All 32 long independent reference
phases pass in full-expert-cache, lazy-cache and uncached execution, preserving
4,491,225,056 BF16 and 16,716,198 F32/I32 words exactly, including routing IDs.
The 32-phase real-model memory check reports zero errors/leaks; long CLI
synchronization checking, short references, CLI/reset/chat, cache/preload,
standard/CUDA suites and dense Qwen27 regression pass. Clean and CPU-only
builds have zero warnings. Reference hashes and precision-profile limits remain
unchanged. A 32-token continuation matches the saved pre-change binary.

The saved baseline binary hashes identically to the preceding qualified
preload binary. New before/after pairs use the same 2 GiB source RAM cache,
4,097-token prompt, default four-chunk schedule, five outputs, one warmup and
three measured runs. Shared caches remain warm across restarted sessions.
Medians on the designated headless GPU are:

| Placement | Matrix path | Prompt ms | CLI reported output tokens/s |
| --- | --- | ---: | ---: |
| All experts resident | DP4A baseline | 3336.1 | 88.43 |
| All experts resident | Integer MMA | 1709.4 | 88.24 |
| Stream from RAM | DP4A baseline | 4659.4 | 8.19 |
| Stream from RAM | Integer MMA | 4312.2 | 8.17 |

Resident-expert prompt time falls 48.76% (1.95x prefill throughput); streaming
prompt time falls 7.45%. This difference limits claims for offloaded workloads:
reducing matrix compute does not remove their source/transfer cost. Separate
short-prompt, 64-output samples report 73.05 versus 72.79 tokens/s, consistent
with unchanged decode kernels within this small sample. CLI output rate includes
the first prefill-sampled token and is not isolated decode latency. Fixed case
order, three samples and repeated prompts limit generalization. Flash's host
PLE table remains present; this is full expert residency, not full-model GPU
residency or a matched vLLM parity result.

The new long resident-expert trace emits 9,594 grouped MMA calls during eight
prefill windows and zero during eight decode windows. Summed prefill kernel
duration falls from 6,803.91 to 3,386.08 ms over two restarted prompts; kernel
count remains 40,656. Expert groups remain 43,032 across all 16 windows, with
zero expert uploads/fill copies and zero inference allocations. Tracked device
heap remains 86,971,359,720 peak bytes and pinned host 67,046,720; both return
to zero. GPU activity union covers 3,382.96 of 3,491.01 ms of prefill windows,
which include caller overhead and profiler effects, rather than isolated
request latency or SM occupancy.

Frozen routing counts independently reproduce all 50,633,920 MMA thread blocks
in that two-run trace. Of 25,316,960 launched blocks per prompt, 15,459,680
(61.06%) return uniformly because bank-max row padding exceeds an expert's
actual rows. Compact tile scheduling is a concrete next experiment. This
geometry model does not measure occupancy or predict its latency improvement;
no such compaction is implemented yet. Remaining matrix and Q8_0 kernels,
transfer/chunk policy, readbacks/fences and the matched Qwen27 engine benchmark
remain open. Evidence and reproducible scripts are under
`/tmp/bitnet-goal-flash-mma/`.

### Loaded-model Qwen27 engine comparison harness (2026-11-08)

`bench/bench_bf16_latency.c` measures native BF16 first-token and subsequent
single-request greedy token delivery. Build with the same CUDA/MKL flags as
`bitnet`, using target `bench_bf16_latency`, then run:

```sh
CUDA_VISIBLE_DEVICES=GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05 \
CUDA_DEVICE_ORDER=PCI_BUS_ID BN_CUDA_DEVICE=0 \
./bench_bf16_latency MODEL.gguf PROMPT.i32 64 3 1 8192 8192
```

`PROMPT.i32` contains little-endian signed 32-bit token IDs. The JSONL output
includes raw/decoded linear layouts, output IDs and every decode interval.
TTFT includes request restart, prefill, logits readback and greedy argmax;
decode is `(last_delivery - first_delivery) / 63`, excluding the first token
produced by prefill. Model load, layout/request allocation, tokenization and
JSON serialization are outside the intervals. CPU fallback is rejected.

`bench/bench_vllm_latency.py ENGINE.json CASES.json OUTPUT.json` implements the
corresponding cumulative-token delivery protocol for an already installed
vLLM environment. Engine arguments must specify single-request concurrency,
no prefix cache and no tokenizer. Cases contain `name`, `prompt_ids` and an
optional frozen `oracle8`. Placement and effective configuration are saved
separately. Inspect delivery batch sizes before interpreting inter-token
latency. Local `apply_model` inspection requires the pinned engine's trusted
serialization opt-in; the benchmark container is offline and isolated.

The pinned reference uses vLLM `8244d1c44`, GGUF plugin
`0.0.5+e454affe`, the existing GGUF extension metadata/index/Q5_1-tail fixes,
and `bench/patches/vllm_gguf_005_qwen_text.patch`. The last patch maps a text
config to `Qwen3_5ForCausalLM` without changing tensor arithmetic. It is
separate from the older 0.3.3 import-compatibility patch.

The reference image's FA2 binary has 76 SM80 images and PTX 9.4; a split-KV
variant fails to load on this host with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
`bench/patches/vllm_fa2_506341a_sm120.patch` builds native SM120 code from
vLLM's exact FA2 source pin `506341a143fcabd4bb79052a7605ada727d6b3f5`
(CUTLASS `62750a2b75c802660e4894434dc55e839f322277`). Build only
`_vllm_fa2_C` with `CMAKE_CUDA_ARCHITECTURES=120`, `CUDA_ARCHS=12.0` and
`NVCC_THREADS=2`, leaving the installed FA3 extension intact. The patch
changes architecture/build selection only. All 76 forward units are built;
ten BF16 paged-attention cases pass against an FP32 Torch reference with
PTX JIT disabled (largest absolute error 0.00390625). This loading/numerical
check alone does not establish engine token equivalence or performance parity.
Evidence, complete commands, source/binary hashes and failures are retained
in `/tmp/bitnet-goal-qwen27-engine/`.

An additional compatibility check located the image's forward-compatibility
libraries under `/usr/local/cuda/compat/lib.real`; its default search path
points at `/usr/local/cuda/compat/lib`. Explicitly placing `lib.real` first
allows the **original** FA2 binary to load all ten standalone cases. Their
110,592 BF16 words match the native SM120 rebuild exactly. This limited A/B
check does not cover complete engine behavior. Prefer the original reference
binary with the corrected process-local library path when establishing the
engine comparison; the native rebuild remains a separately recorded fallback.
No host driver or repository dependency was replaced.

The actual engine comparison now runs to completion with all 965 reference
parameters and all named buffers on CUDA (17,924,169,810 parameter bytes).
The unchanged Qwen27 GGUF is fully GPU-resident in both engines; native
profiling reports 34,655,021,572 peak tracked device-heap bytes, zero inference
allocations, no weight uploads and a maximum inference H2D copy of 4,100 bytes
(token IDs). Decode has 1,456 kernels, two stream fences and 496,648 D2H bytes
per step in this trace, including BF16 logits readback. The profile windows
include caller/restart overhead and are not isolated latency measurements.

**Performance parity remains unqualified because token equivalence fails.**
The original-FA2 reference with process-local CUDA compatibility libraries
produces stable outputs across one warmup and three measured runs per case.
All 64 outputs match native for the 5-token case and the synthetic 4,097-token
repeat case. The 129-token case diverges at output 3; the 1,025-token case at
output 22. A prior three-case, eight-output engine smoke test passed, showing
why that smoke alone is insufficient. The medium mismatch repeats when run
first in a fresh engine and through both `LLM.generate` and cumulative
`LLMEngine.step`, including eight-output requests. The SM120 FA2 rebuild also
reproduces the medium mismatch; its long case first diverges at output 44.
Standalone attention equality does not imply full-engine equality.

Diagnostic medians, in milliseconds (lower is faster):

| Prompt tokens | Native TTFT | vLLM TTFT | Native decode/token | vLLM decode/token | All 64 IDs match |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 5 | 49.74 | 50.35 | 15.54 | 18.12 | Yes |
| 129 | 219.41 | 370.79 | 15.62 | 18.03 | No |
| 1025 | 1213.12 | 2577.25 | 15.84 | 18.36 | No |
| 4097 (repeat) | 4679.62 | 10306.85 | 16.35 | 18.78 | Yes |

Both engines use BF16 activations, F32 SSM state, the same quantized checkpoint,
GPU UUID, explicit token IDs, 8,192 context/prefill capacity, concurrency one,
64 greedy outputs and ignored EOS. Reference compilation, FULL_AND_PIECEWISE
CUDA graphs, FlashAttention2, FlashInfer GDN prefill, CUDA GDN decode and
normal autotuning stay enabled. Reference KV allocation is 2 GiB with no host
KV offload or prefix caching. Each delivery contains exactly one output token.
Native runs precede reference runs; the small sample and fixed case order limit
performance conclusions. The original reference uses the image's CUDA 615.65.02
forward-compatible userspace libraries over host driver 595.104.02; native
uses the host driver. Complete commands and effective configurations are saved.

Native library/headers retain all 315 recorded production hashes from the
preceding qualified milestone, and frozen expected tokens are unchanged.
Baseline tests, protected clean build (zero warnings), standard tests, the
native benchmark's CPU guard/build and both compatibility-patch dry runs pass.
The previous production CUDA qualification remains separate from this
benchmark work. The next correctness task is an aligned engine-versus-native
logits/intermediate trace at the first medium divergence, including selected
GDN/linear execution plans and top-logit margins. Do not rewrite the frozen
expectations or report the diagnostic timing ratios as general parity.

### Qwen27 arithmetic-profile diagnosis (2026-11-08)

The medium mismatch is a near-tie: native's output-3 logits are 22.25 for ID 63
and 22.125 for ID 12; a reproducible default compiled engine produces 22.25
for both and chooses the lower ID, 12. Six requests in one engine preserve
that choice with/without capture, with/without logprobs, and with 8/64 outputs.
An earlier instrumented engine initialization produced ID 63 instead; agreement
on a short greedy prefix is therefore insufficient to qualify the numerical
contract. The frozen token expectations remain unchanged.

Replaying the native final hidden vectors through the reference vocabulary
head reproduces all 744,960 logits exactly across prefill and two aligned
decode steps. The observed differences are upstream of that projection.
Changing the worker's BF16 reduction permission before initialization, with a
fresh compilation cache, leaves all 15,360 captured hidden values and 744,960
logits identical for the default FlashInfer case. That permission is an
important precision setting to record, but does not explain this case.

A stronger correctness-only check now compares every layer of the entire
129-token prefill against an actual reference engine using eager execution,
FLA GDN prefill and FP32 BF16 reduction. All **169,743,360 BF16 words in 257
vectors match exactly**: input norms, post-attention norms, FFN outputs,
residuals and final norm across 64 layers. The first input norm also matches
separately through both eager and compiled reference norm calls (660,480 words).
This qualifies that explicit eager prefill profile; it does **not** qualify
compiled/default-engine equality, decode after this reference trace, or
performance parity. The default compiled performance baseline is retained.

There is a concrete compiled-operator difference. On a deterministic
129-by-17,408 BF16 gate/up batch, the original CUDA SiLU-and-multiply and
PyTorch-native calls agree, while the unmodified compiled call differs in
581,479 of 2,245,632 outputs (maximum absolute difference 0.03125). Generated
Triton code retains F32 through SiLU and multiplication. A separate 30,720-word
gated-norm fixture matches its original, native and compiled paths exactly;
its frozen reference bytes were revalidated rather than regenerated.

The native backend already exposes separate/fused SiLU arithmetic in its
resident FFN plan. A temporary diagnostic selecting FUSED changes the medium
prediction to the default compiled prefix, but its final hidden vectors still
differ (prefill RMS difference 0.05966; output-3 RMS difference 0.04975 against
the captured default engine). This is not a complete fix and is not enabled
in production. Further work must identify the remaining compiled arithmetic
boundaries instead of treating greedy-token agreement as sufficient evidence.

The benchmark now records worker-side reduction settings and per-layer GDN
prefill/decode/fusion selection. `--require-fp32-reduction` rejects a worker
permitting reduced BF16 reductions at inspection time; configure worker
precision before initialization/warmup, since this inspection cannot prove
which setting an existing CUDA graph captured. Native JSONL explicitly records
`bf16_gemm_reduction: fp32`. The worker audit is exercised on the real model
(all 48 GDN plans); the new native metadata and unchanged eight-token fixture
pass. Protected clean build, standard tests and CPU guard pass with zero build
warnings, and all 315 recorded production source/header hashes and four user
binaries remain unchanged. Evidence is under
`/tmp/bitnet-goal-qwen27-logits/`; all traced timings are diagnostic only.

### Compiled prefill boundaries and retained FP32 intermediates (2026-11-08)

A persistent diagnostic worker now captures existing generated-kernel inputs
and outputs without adding module hooks or recompiling the model. The initial
GC scan missed the launchers because vLLM had frozen about 1.6 million Python
objects; enumerating `PyCodeCache.modules` and `modules_no_attr` reaches the
live cached kernels. The successful capture wraps 253 existing launchers and
records 513 BF16 matrix snapshots across 129 normalization-related launches.
An additional first-block capture records activation operands. The same engine
produces the same eight tokens before capture, with capture, after a controlled
GDN-prefill dispatch change, and after restoring all launchers/backend methods.
These are correctness traces, not latency or throughput measurements.

The earliest observed difference is input normalization: one of 660,480 BF16
values differs at token 30, feature 4,345 (absolute difference 0.001953125).
Replaying the original generated first-norm function on its captured embedding
and weight inputs reproduces all 660,480 compiled outputs exactly with
`XBLOCK=1`, `R0_BLOCK=1024`, eight warps. The eager norm on those same inputs
matches native exactly. Ten original-kernel tile/warp configurations produce
one to three differences from the eager result; selected reduction geometry
is therefore part of a reproducible arithmetic profile.

Two larger intermediate-precision differences are now isolated on actual
checkpoint activations, not only synthetic inputs:

| Boundary | Original compiled output | Eager-style result on the same operands |
| --- | --- | --- |
| Layer-0 SiLU/product, 2,245,632 BF16 values | Exact replay with FP32 SiLU/product and a final BF16 store | 592,671 differences; max absolute error 0.0078125 |
| Residual after two additions, 660,480 BF16 values | Exact replay retaining the first sum in FP32 until the second addition | 135,284 differences when the first sum is rounded to BF16 |

The activation replay uses the unchanged original vLLM CUDA/native/compiled
methods; both the standalone compiled call and the FP32-intermediate formula
match the actual compiled graph output exactly. The residual analysis uses
captured operands and explicit FP32 additions with RTNE BF16 stores, matched
against the original graph's stored residual. Generated code also retains
FP32 between those additions. This explains why selecting the existing native
fused-SiLU option alone did not reproduce the compiled hidden vectors.

A live FLA prefill dispatch counterfactual keeps the outer compiled kernels,
weights and decode backend intact. At the first post-attention norm it reduces
differences from 455,728 to 86,384 values and RMS error from 0.00060764 to
0.00011413 against the eager/native trace, but does not restore full equality.
This is a diagnostic dispatch override, not a replacement performance baseline
or a qualification of a separately compiled FLA model.

`bench/bench_vllm_latency.py` now records cached kernel source SHA256 values,
selected tile/warp/stage configurations, FP-fusion settings and Inductor's
`emulate_precision_casts` policy. The real worker audit sees 254 cached kernels
(including a diagnostic activation replay) and verifies the first norm's exact
source/configuration; `emulate_precision_casts` is false. These cached/warmup
records describe available configurations, not per-request kernel use. The
separate captured launch manifest identifies actual execution.

The next implementation step is a supported backend arithmetic profile that
accounts for retained FP32 activation/residual intermediates and the selected
normalization reductions, qualified against original compiled outputs. Preserve
the existing eager profile and independent goldens. No production kernel or
runtime arithmetic has changed in this investigation, and general compiled
performance parity remains unqualified. Evidence and replay scripts are under
`/tmp/bitnet-goal-qwen27-kernels/`.


### Explicit retained-FP32 request profile (2026-11-08)

`bench_bf16_latency` accepts an optional final `eager` (default) or
`fp32-intermediates` argument and records `arithmetic_profile` in its JSON
configuration. The latter retains the first residual addition in FP32 through
the second addition and uses the existing fused FP32 SiLU/product kernel.
It retains the qualified contiguous FP32 normalization reduction and native
GDN/attention implementations. It does not claim equality with arbitrary
Inductor normalization tiles, or replace the original default compiled-vLLM
performance baseline. The model/CPU preparation math profile is independent.

The new three-input norm callback passes 38 independent PyTorch cases with
14,407,680 exact BF16 output words. Existing CUDA references and real-model
eager goldens are preserved. Real-model profile checks cover complete logit
replay across restart, observing residuals, request isolation, no-logit prefill,
matching chunk schedules, resume/decode, and invalidation after a fence failure.
Evidence, commands and logs: `/tmp/bitnet-goal-bf16-f32-profile/`.


On the designated RTX PRO 6000 Blackwell Max-Q, one warmup and three measured
64-output runs through the normal request path give these medians for the
explicit `fp32-intermediates` profile (milliseconds, TTFT includes restart,
prefill, host logits and greedy selection):

| Prompt tokens | TTFT | Mean decode step |
| --- | ---: | ---: |
| 5 | 49.99 | 15.67 |
| 129 | 222.29 | 15.75 |
| 1025 | 1225.57 | 15.98 |
| 4097 | 4733.90 | 16.52 |

All measured repeats within each case produce the same 64 output IDs. The
129-token case still selects token 63 at output 3, whereas the earlier default
compiled reference selects 12. These measurements therefore establish stable
execution and latency for this arithmetic profile; they do not close matched
compiled-vLLM numerical or performance parity. Kernel memcheck and synccheck
both pass with zero errors.

The new CUDA residual operation also replays all 660,480 stored residual words
from the original compiled 129-token graph exactly on its captured operands.
This gate checks the residual output only, not equivalence of normalization
tiles. A protected clean build and standard suite pass with zero warnings,
and all existing CLI frozen-reference gates pass. Flash Next's 4,097-token
prompt plus 32 greedy outputs matches the saved trajectory exactly with the
expert preload path; this regression checks output IDs, not every layer word.


### Compiled norm reduction isolation (2026-11-08)

A standalone CUDA replay of the selected first compiled normalization now
matches all 660,480 BF16 output words on the captured 129-token operands.
The per-thread four-value reduction uses `(a0+a2)+(a1+a3)`; changing it to a
sequential local sum restores the one-word discrepancy. On these operands,
separate square/add versus FMA and exact division versus reciprocal multiply
do not change that result; this does not establish their general equivalence.

Original cached cubins for the first, two-input and three-input normalizations
all replay their respective captured outputs exactly (1,981,440 words total).
The standalone two/three-input implementations still differ in at least three
BF16 values each across the tested reduction/contraction variants. These are
unresolved diagnostic results, with no production arithmetic changes or new
full-model parity claim. Evidence and CUDA replay source are under
`/tmp/bitnet-goal-compiled-norm/`.


### All three captured compiled norms reproduced (2026-11-08)

The remaining two/three-input standalone norm discrepancies are resolved on
the original captured operands: all three norms now match 1,981,440 BF16
output words exactly. The missing operation was fused mean scaling plus
epsilon: `fmaf(sum, 1.0f / 5120.0f, 1e-6f)`. Disassembly of both original
cached cubins confirms `FFMA` with reciprocal `0x394ccccd` and epsilon
`0x358637bd` before `MUFU.RSQ`. Separate rounded mean and epsilon addition
changes output bits even when the sum of squares is identical.

For the selected 512-thread add-norm kernels, the eight local accumulators
reduce in offset order 2, 4, 1. The 256-thread first norm uses the previously
qualified four-accumulator tree. Exact captured replay uses variant 0 for the
first norm and 288 for both add-norms in the standalone evidence harness.
Original cubin controls remain exact and unchanged; memory checking reports
zero errors. Instrumented PTX copies expose sums/scales for diagnosis only.
No production arithmetic changed in this follow-up. Independent synthetic and
other-shape qualification is required before exposing this reduction contract
in the runtime; full model equivalence and the remaining GDN differences are
still open. Evidence: `/tmp/bitnet-goal-compiled-norm/results9.jsonl`, original
SASS, source and qualification metadata in the same directory.


### Synthetic qualification of compiled normalization formulas (2026-11-08)

The three standalone formulas now match 144,629,760 BF16 values over 120
independent synthetic cases: eight input patterns and batch sizes 1, 5, 17,
129 and 1025, for each of the three norm operations. Patterns include exact
cancellation, small/large scale and wide exponent ranges. Original selected
cubins supply the reference outputs; no reference outputs were adjusted.
Captured replay still matches all 1,981,440 values. Memcheck and synccheck
report zero errors.

The expanded cases exposed two issues hidden by the captured operands. The
first norm also requires fused mean scaling plus epsilon. More importantly,
the persistent two-input norm combines local square pairs with FMA within
each tile before combining tiles, whereas the three-input norm accumulates
across tiles before local reduction. Treating these as the same arithmetic
left one BF16 mismatch in the 1025-token wide-range case. Keeping their
operation order distinct removes it. Current replay selections are variant
32 for the first norm and 288 for both add-norm entry points, whose kernel
branches implement different square/reduction orders.

This qualifies the formulas against those selected original configurations
at dimension 5120, epsilon 1e-6 and weight offset 1; it does not qualify other
compiler tiles or full-model parity. Production arithmetic is unchanged in
this follow-up. Next integrate a named reduction contract with independent
synthetic goldens and retain existing eager gates. Evidence/source:
`/tmp/bitnet-goal-compiled-norm/qualification.json` and `synthetic.cu`.


### Backend tiled normalization contract implemented (2026-11-08)

CUDA resident RMSNorm, add-RMSNorm and three-input add-RMSNorm now accept
`BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA`. This is an explicit fixed arithmetic
contract: dimension 5120, epsilon 1e-6, shared BF16 gamma, weight offset 1.
Other shapes/settings reject before work instead of silently selecting another
reduction. The first, persistent two-input and looped three-input kernels
retain their separately qualified square/reduction orders and fused mean plus
epsilon. Intermediates remain on device; no request allocations are added.
Existing default/contiguous reductions retain their original arithmetic.
Ordinary dense request initialization does not yet select the new contract.

The full CUDA suite passes, including 120 new independent original-cubin cases
with 144,629,760 exact norm words and 48,209,920 exact three-input residual
words. Seven invalid-plan variants are checked through all three callbacks
for every case. The repository generator reproduces the frozen header exactly;
memcheck and synccheck both report zero errors. Reference provenance, cubin
hashes and selected configurations are in
`test/cuda_compiled_norm_reference.json`; the generator is
`test/reference_cuda_compiled_norm.cu`. Existing reference values are preserved.
Evidence and command logs: `/tmp/bitnet-goal-compiled-runtime/`.

Next connect this reduction contract to an explicit dense request arithmetic
profile and qualify complete short/long/chunked/resumed reference traces.
Native GDN/attention equivalence and default compiled-vLLM parity remain open;
passing these operator gates alone does not close those requirements.

The protected clean build and standard tests also pass with zero warnings;
all four pre-existing probe binaries are preserved byte-for-byte. Frozen CLI
integration passes all 20 cases. Flash Next's 4097-token prompt and 32-output
trajectory matches the saved expert-preload baseline (output-ID regression,
not an all-layer comparison). On original captured three-input operands, the
production callback matches 660,480 norm and 660,480 residual words exactly.
The broader request-level compiled comparison remains the next gate.


### Explicit tiled-FP32 dense request profile (2026-11-08)

`BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32` now selects the qualified fixed
tiled normalization contract together with retained FP32 activation/residual
intermediates. Backend reduction capabilities are declared explicitly; missing
support rejects before request allocation. The profile is request-local,
survives allocation-free restart, and uses the ordinary forward/prefill paths.
The eager default and existing FP32-intermediate profile remain available.
`bench_bf16_latency` accepts `tiled-fp32` and records that label in JSON.

The real-model runtime gate passes for both explicit profiles: complete logit
replay across restart, observer checks, interleaved eager-session isolation,
matching chunk schedules, resume/decode, no-logit publication and failure
invalidation. The new profile adds no request allocations or CPU fallback.
Its full runtime memory check, full CUDA regression and protected clean/CLI
checks are still pending in `/tmp/bitnet-goal-tiled-request/` at this entry.

One warmup plus three measured 64-output runs yield median TTFT/decode-step
milliseconds: 5 tokens 49.66/15.56; 129 tokens 213.62/15.62; 1025 tokens
1209.56/15.88; 4097 tokens 4668.55/16.34. Within-case output IDs are stable.
Against the saved original compiled reference, short and 4097-token cases
match all 64 output IDs; the 129-token case still differs at output 3
(63 versus 12), and the 1025-token case first differs at output 51.
This is not a fresh paired vLLM timing run or full numerical parity claim.

Read-only inspection of the exact reference image identifies another concrete
GDN difference: FlashInfer SM120 casts its F32 KK accumulator to FP16 before
`CollectiveInverse`, which uses FP16 MMA operands and stores, while native
prefill retains F32 inverse profile4 before its BF16 conversion. A queued
first-layer capture will localize the remaining divergence. Source hashes and
original files are preserved under the same evidence directory; no GDN
arithmetic changed in this implementation.

Further read-only inspection of the pinned FlashInfer source identifies two
additional GDN arithmetic boundaries: alpha processing scans
`log2(exp(g) + 1e-10)`, and beta multiplies the F32 key-dot accumulator after
the dot product. Native prefill scans log decay directly and scales operands
before the correlation. These source-level differences require actual
operand/output comparison; they are not yet a proof of the first divergent
operator. An original-GDN capture wrapper is prepared under
`/tmp/bitnet-goal-qwen27-gdn/controller/` with syntax checks only. It has not
been launched; the existing serialized runtime memory-check/regression pipeline
remains active. No production kernel changes accompany this investigation.


### Tiled request qualification complete; GDN core isolated (2026-11-08)

All queued native checks now pass: full real-model runtime memcheck reports
zero errors, the full CUDA suite passes, protected clean build and standard
tests pass with zero warnings, all frozen CLI cases pass, and the Flash Next
4097-token/32-output trajectory remains exact. Four pre-existing probe
binaries retain their hashes. No request arithmetic defaults changed.

A controlled capture from the original default compiled reference now compares
layer0 directly with the native tiled profile. Input normalization (660,480
BF16 values), prepared Q/K/V, log decay, beta and the F32 initial state all
match bit-for-bit. The GDN output differs in 388,377 of 792,576 BF16 values
(max absolute error 0.078125); next state differs in 776,661 of 786,432 F32
values (max absolute error about 0.112315). Thus a remaining difference is
inside the GDN execution path, with matching actual operands. Instrumentation
preserves eight generated control IDs before/during capture and after fully
restoring the first-norm launcher and GDN dispatch bindings.

Direct replay of these captured operands identifies the actual default as
FlashInfer's `cp_delta_rule_dsl_sm120` context-parallel path. Default-auto and
forced-CP replay reproduce all 1,579,008 captured output/state values exactly.
Forcing the non-CP SM120 path instead changes 175,624 BF16 output values and
668,317 F32 state values. Non-CP is a diagnostic counterfactual, not an
acceptable replacement for this default compiled comparison. A native
compatible profile must qualify the CP phases and their arithmetic.

This fresh reference initialization selects token 63 at medium output3,
whereas the saved earlier benchmark selected 12; all three controls within
the fresh engine agree. Existing frozen fixtures and saved benchmark outputs
remain unchanged. The operator comparison relies on exact captured operands,
not cross-initialization greedy-ID agreement. Full-model equivalence,
matched performance parity and the broader serving/model/backend requirements
remain open. Evidence: `/tmp/bitnet-goal-qwen27-gdn/qualification.json`;
native qualification: `/tmp/bitnet-goal-tiled-request/qualification.json`.


### First native FlashInfer-CP stage reproduced (2026-11-08)

A stage-level capture of the unchanged original CP call now freezes signed
inverse tiles, local transfer/state, fixed-up state, final output and next
state. Copying these boundaries preserves all 1,579,008 original output/state
values. The 129-token call chooses CP chunk length 512. Its T precompute is
independent of decay: `T = -(inv(I + lower(beta_row * KKT)) beta)^T`.

A standalone CUDA prototype reproduces the T stage exactly: all 589,824 BF16
values from the captured prompt and 32,243,712 values from 40 independent
original-kernel synthetic cases. Batches span 1, 5, 17, 63, 64, 65, 129, 257,
513 and 1025, with tiny, signed and sparse keys and beta values including zero
and one. Original references are generated twice and checked for equality;
existing goldens remain unchanged. Memcheck and synccheck report zero errors.
The prototype uses BF16 tensor-core Gram products, FP16 inverse storage,
explicit F32-FMA diagonal8 elimination, FP16 MMA block composition and the
reference's FP16 rounding of64-level partial products before their addition.
A CPU diagnostic with rounded high-precision matmuls still differs in 592
captured BF16 values and is not used as a golden or implementation substitute.

No production GDN kernels changed. The remaining CP transfer/state, fixup and
final prefill phases must be implemented and qualified, including nonzero
initial state and multiple CP chunks, before exposing a complete compatible
profile. Full-model parity and the original wider goal remain open.
Source, references and qualification: `/tmp/bitnet-goal-qwen27-gdn/cuda-t/`,
`/tmp/bitnet-goal-qwen27-gdn/t-synthetic/` and `stages/`.


### Native CP transfer/state and fixup stages qualified (2026-11-08)

The standalone CUDA prototype now reproduces local transfer and state
precomputation exactly: 185,597,952 F32 words over 40 independent original
FlashInfer cases. The composed T/MN/fixup pipeline also matches all 92,798,976
F32 prefix-state values, totaling 278,396,928 exact F32 comparisons. Cases use
zero/nonzero F32 initial state, lengths 1, 17, 63, 64, 65, 129, 257, 321,
513 and 1025, and explicit CP lengths 64/512 (up to 17 chunks). All references
are generated twice and checked for equality. Memcheck and synccheck report
zero errors for both MN and the composed fixup suite. Captured real-model
transfer, local state and fixed state each match all 786,432 F32 values.

The MN implementation consumes the transposed view of the physical signed T
matrix, preserves the reference's BF16 intermediate stores, and keeps decay
placement distinct for transfer/state updates. For 48 heads, fixup selects
raw-F32 TF32 MMA with even/odd feature permutation inside each 8-wide block;
introducing an explicit conversion or using the old KKT's 16-wide permutation
would change this contract. Fixed-state index j stores the state after chunk j.

Final CP prefill remains unimplemented. Its final recurrent state must not be
replaced by the last fixed state: those reference tensors differ in 786,418
of 786,432 captured F32 values (max difference about0.08454). Native final
output/state, runtime integration and full model/performance/serving goals
remain open. Production GDN kernels are unchanged; all new kernels are still
standalone prototypes. Source and evidence are under
`/tmp/bitnet-goal-qwen27-gdn/cuda-mn/`, `cuda-fixup/`, and `mn-synthetic/`.


### Dense Qwen27 CP profile: scoped resident CUDA qualification (2026-11-08)

The complete SM120 CP GDN pipeline is now integrated behind the explicit
`BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32` request profile (`cp-fp32` in
`bench_bf16_latency`). It retains F32 activation/residual and attention-sigmoid
intermediates, the fixed tiled 5120-column normalization contract, F32 SSM
state, and explicit `expf` log-decay staging. Existing eager arithmetic remains
the compatibility default. Earlier standalone-only status entries above are
superseded by this integration.

This comparison uses the same Qwen3.8-27B UD-Q4_K_XL GGUF checkpoint, RTX PRO
6000 Blackwell Max-Q (SM120), one request, context capacity 8192, and a prefill
budget of 8192 tokens. Native preparation uses MKL VML HA. The original pinned
vLLM image is
`sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`,
with GGUF loading, BF16 compute/KV, F32 SSM, FA2, default compiled graphs and
prefix caching disabled. All learned reference parameters are CUDA resident.
The reference kernels are unchanged; the GGUF text-only architecture mapping
and validated plugin binary are the previously documented compatibility setup.

One warmup and three measured runs generate 64 greedy tokens per prompt.
TTFT includes request setup/restart, prefill, readback and sampling; decode is
the median of each run's mean over the following 63 token-delivery intervals.
Setup/model loading is excluded.

| Prompt tokens | Native TTFT ms | vLLM TTFT ms | Native decode ms | vLLM decode ms |
| --- | ---: | ---: | ---: | ---: |
| 5 | 52.372 | 50.051 | 15.851 | 18.014 |
| 129 | 221.615 | 369.617 | 15.880 | 17.977 |
| 1025 | 1290.407 | 2563.279 | 16.130 | 18.250 |
| 4097, repeated-token prompt | 4951.733 | 10240.937 | 16.561 | 18.725 |

Every measured token stream matches the fresh unmodified reference for all
64 outputs, including warmups. All 128 normalization stages of the captured
129-token prefill match exactly: 84,541,440 BF16 values. A separate five-sample
short-prompt repeat measures 52.279 ms, versus 53.445 ms for the saved previous
native executable. The short case is 4.64% slower in the three-sample table
and 4.45% slower in that repeat; the other measured prefill/decode cases are
faster than vLLM. This meets the 5% threshold for these single-request resident
CUDA workloads, with a narrow short-prompt margin.

The CP operator preserves 551,755,776 exact output/state comparisons over
56 independent original-FlashInfer cases. They include nonzero initial state,
partial tiles, multiple CP chunks and 511/512/513-token chunk boundaries.
The implementation skips unused MN/fixup work for a single CP chunk. Its
preparation writes the token-major query copy contiguously and uses 512
threads for independent copies/casts while retaining prefix math in warp zero.
Nsight records the expected 36 launches for a 129-token multiplier-domain CP
call, no allocation or host transfer, and one final device-to-device state
copy. Preparation time for its three local blocks falls from 132.064 to
49.568 microseconds. CPU/CUDA suites, request lifecycle tests, kernel memory,
initialization and synchronization checks, and real-model initialization
checking pass.

The original saved reference values remain unchanged. A fresh audit established
that the earlier layer-9 one-value mismatch came from reference configuration
variation: actual original gamma, all operands and both outputs match native
on 3,307,520 values. Source hashes and selected launch configurations accompany
the fresh trace. This profile does not promise bit identity with arbitrary
compiler/autotune selections.

Large native prompts also pass a 512-token schedule using both automatic
chunking and separate no-logits-prefix/one-token-tail calls, followed by 64
decode tokens. Those native schedules agree. The default vLLM 512-token-budget
run changes its own output for the unsplit 129-token control and disagrees
with native on that case and the 1025-token case. That external chunked
comparison remains unqualified pending an arithmetic-configuration and
one-token-tail dispatch audit. The 8192-token-budget result above does not
establish parity for this different schedule, other checkpoint/precision
formats, libc preparation, other GPUs, CPU backends, offloaded models or
concurrent serving. The wider model/backend/serving goal remains open.

Commands, inputs, exact comparisons, source/configuration hashes, profiler
reports and both reference snapshots are retained in
`/tmp/bitnet-goal-cp-layout/`, `/tmp/bitnet-goal-norm9/benchmark/` and
`/tmp/bitnet-goal-norm9/controller/`. The scheduling-specific follow-up is under
`/tmp/bitnet-goal-cp-layout/chunk-reference/` and `chunk-native/`.

### Qwen27: 512-token reference arithmetic and prompt-tail audit (2026-11-08)

The pinned original vLLM engine was audited with a 512-token batch budget,
chunked prefill enabled, and the same model, precision, device and cache
settings as the comparison above. Both audit controllers exited successfully.
The unsplit 129-token prompt produced the same eight tokens before, during
and after instrumentation: `[18065, 271, 63, 3431, 14905, 508, 63, 271]`.
These differ from the 8192-budget reference beginning at output index 2.

The full capture aligned 128 backbone normalization stages. The first
divergence occurs at layer 0 input normalization, before recurrent state:
all 660,480 embedding words and 5,120 weight words match the 8192-budget
reference exactly, but three of the 660,480 normalized BF16 words differ.
The stock 512-budget engine selected `R0_BLOCK=8192`, 16 warps for this
operator; the qualified 8192-budget engine selected `R0_BLOCK=1024`,
8 warps. Thus even this unsplit control has different reduction arithmetic.

An additional operator replay used the unchanged original three-input
normalization kernels on identical captured layer 10 operands and gamma.
The selected 512-budget configuration (`R0_BLOCK=1024`, 8 warps) reproduced
both captured outputs exactly. The original 8192-budget configuration
(`R0_BLOCK=4096`, 16 warps) preserved the residual output exactly and changed
four normalized BF16 words. This diagnostic measures configuration effects;
it does not replace either stock reference or its performance baseline.

An initial ordinal capture selected the ninth three-input norm call, which
is layer 10 under the 512-budget compilation and layer 9 under the
8192-budget compilation. The weight identity was checked against GGUF,
cross-layer comparisons were discarded, and the full stage capture was
used instead. Kernel names and call ordinals alone do not establish layer
identity across different compilations.

Live GDN metadata confirms the 1025-token prompt executes as
`512 prefill, 512 prefill, 1 prefill`, followed by decode. The 4097-token
prompt executes eight 512-token prefills and a one-token prefill, followed
by decode. This agrees with the original metadata builder's explicit
`treat_short_extends_as_decodes=False` policy and with native prompt-tail
dispatch. No native dispatch change is justified by this audit.

The stock 512-budget token comparison remains unqualified. Arithmetic
variation before recurrent state does not rule out a separate continuation
defect. The next correctness control must explicitly match the numerical
profile using unchanged original operators, first on the unsplit prompt
and then on chunked/resumed prompts. Production code, defaults and frozen
oracles were unchanged during this audit; existing suites were not rerun
for these external diagnostic scripts. Evidence, source hashes, controls,
metadata and comparisons are in `/tmp/bitnet-goal-chunk-audit/` and
`/tmp/bitnet-goal-chunk-full/qualification.json`. The full production goal
remains active.

### Qwen27: matched arithmetic for chunked/resumed correctness (2026-11-08)

A separate pinned original vLLM run fixed only the launch configurations of
its unchanged 5120-column RMSnorm JIT functions: one-input norms use
`R0_BLOCK=1024`, 8 warps; two-input norms use `8192`, 16 warps; three-input
norms use `4096`, 16 warps. The policy was installed before engine startup
so graph capture uses the same declared arithmetic. GDN, FA2, model weights,
BF16 activations, F32 SSM state, graph mode and 512-token scheduling were
retained. This is an additional correctness control, not a stock performance
measurement or a replacement for the preserved stock reference.

With that profile, the unsplit 129-token prompt matches native at all
128 backbone normalization stages: 84,541,440 BF16 words, zero differences.
Its 64 generated tokens agree before, during and after capture. The 5-token
prompt agrees in two runs; the 1025-token and 4097-token prompts each agree
in three runs, with 64 generated tokens per run. These schedules include
`512+512+1` and `8*512+1` prompt chunks. The previously measured native
explicit resume runs (prefix without logits, then one-token prompt tail)
match native automatic chunking and these independent reference streams.
All 11 controlled reference generations completed successfully and the
controller exited with status zero.

This qualifies those chunked and resumed paths under the explicit native
CP_FP32 / MKL VML HA numerical profile. It does not assert bit identity with
every stock compiler/autotune configuration, or performance parity for the
512-token schedule. The original stock 512-budget output differences and
all frozen references remain preserved. Evidence and configuration/source
hashes are in `/tmp/bitnet-goal-chunk-matched/qualification.json`.

### Qwen27 CP arithmetic exposed in the CLI (2026-11-08)

The production CLI now accepts `--cuda-bf16-profile eager|cp-fp32`.
`eager` remains the compatibility default. `cp-fp32` selects the qualified
request-local arithmetic, requires the supported dense hybrid model and
SM120 CUDA capabilities, and uses the existing required MKL VML preparation.
The selected profile survives restart and is reapplied after session reset.
Unsupported model/backend combinations fail explicitly; the Flash checkpoint
is rejected before GPU preparation when this dense-only profile is selected.

For an explicit 8192-token prompt chunk and context capacity:

```sh
./bitnet "$MODEL" --cuda-bf16 --cuda-bf16-profile cp-fp32 \
  --prefill-batch 8192 --maxseq 8192 --temp 0 --repeat-penalty 1 \
  -p "Explain expert offloading." -n 64
```

The new `test/test_cli_bf16_cp.py` uses the separate frozen independent
`test/bf16_cli_cp_reference.json`. It checks all 64 output tokens for each
of the four prompt lengths at both 8192- and 512-token chunk sizes: eight
streams, 512 tokens total. The 8192-budget tokens come from the preserved
stock engine; the 512-budget tokens come from the explicitly matched
normalization control above. It also checks invalid flags, explicit eager
selection, repeated benchmark/no-logits reset and chat reset. The existing
eager CLI fixture and test remain unchanged and pass.

The protected clean CUDA/MKL build has zero warnings; standard CPU tests,
the complete CUDA suite with diagnostic isolation, both CLI integration
suites and real-model request runtime tests pass. The latter cover all
three optional arithmetic profiles, isolation, restart, chunk/resume and
failure invalidation. All four user-owned llama probe binaries retain their
hashes. Commands, logs and results are in `/tmp/bitnet-goal-cp-cli/`.
This exposes the qualified path for normal use; the earlier performance
claim remains scoped to its measured resident workloads and configuration.

### Flash compact expert-grid experiment: no latency gain (2026-11-08)

A compact-grid prototype removed bank-max-row padding from the grouped
Q4_K/Q5_K gate/up and Q5_1 down MMA launches. It retained the original
8/24/64 token-tile selection and per-output arithmetic, mapping blocks through
the existing descriptors in expert/feature/token order. It added no buffer,
metadata transfer, kernel launch or inference allocation. Every one of the
9,857,280 useful block coordinates in the frozen 4097-token routing case
matched the original grid's useful coordinates. The prototype also passed
192 nonzero GPU differential cases against the retained per-expert DP4A path.

The experiment did not establish a useful performance gain. Fresh resident-
expert measurements use the same 4097-token prompt, host PLE placement,
2 GiB source RAM cache, five outputs, one warmup and three measured runs.
Median prefill was 1642.8 ms for the saved production binary and 1640.4 ms
for the prototype, a 0.146% difference. Fixed run order and three samples
limit interpretation of this small difference. These are local timings,
not matched vLLM results or isolated decode measurements.

Paired Nsight traces each contain two restarted prompts. The compact grid
reduces grouped-MMA block count from 50,633,920 to 19,714,560, retaining
all 9,594 grouped-MMA calls. Summed grouped-MMA duration is 1531.047 ms
before and 1532.642 ms after. Removing 61.06% uniformly returning blocks
therefore did not reduce measured MMA time. These sums include profiler
effects and do not measure SM occupancy or isolated request latency.
Tracked device heap and pinned host peaks are identical in both traces,
with zero inference allocations, expert uploads or cache-fill copies.

The production kernel sources were restored byte-for-byte to their saved
pre-experiment versions. The experimental binary and sources remain under
`/tmp/bitnet-goal-flash-compact/`. The retained test expansion adds eight
unsorted experts, disjoint row ranges with gaps, and groups spanning up to
five 64-token tiles, across all existing gate/down formats and hidden widths.
The final restored build has zero warnings; standard CPU tests, the full
CUDA suite with diagnostic isolation, all 192 differential cases, memcheck
and synccheck pass. No original golden values were changed.

The paired baseline trace identifies Q8 matrix projections as another
candidate: dense Q8 kernels consume 572.953 ms and grouped Q8 down kernels
217.909 ms across the two prompts, approximately 24% of summed prefill
kernel time. They retain the DP4A matrix path. An integer-MMA experiment
must preserve Q8's F32 scale product and sequential per-32-value FMA order;
Q5_1's FP16-rounded scale product is a different contract. No Q8 matrix
optimization is implemented or qualified by this experiment. Commands,
profiles, geometry models and verification are under
`/tmp/bitnet-goal-flash-compact/`.

### Q8 integer-MMA matrix execution (2026-11-08)

SM120 native BF16 Q8_0 matrix projections now use integer tensor cores for
dense linear operations and grouped expert down projections. The existing
total-batch activation quantization and MMVQ/MMQ precision selection are
preserved. Q8 retains its F32 product of FP16 scales and sequential per-32-value
FMA accumulation, followed by BF16 rounding. It does not adopt Q5_1's FP16
product rounding. Unaligned 34-byte weight blocks use safe loads. Token tiles
are 8/24/64. Vector/decode kernels, other-device dense dispatch and the
per-expert DP4A path remain available with their existing behavior.

The integer-only dot32 helper moved byte-for-byte into a shared private quant
header. No public API or quant capability declarations changed: Q8 BF16
linear/MoE support was already declared in the format registry. Model weights,
routing, cache admission, placements, budgets and request ownership are
preserved. There are no new persistent packed layouts or inference allocations.

The direct test checks all three MMA tiles against the retained DP4A kernel
and an independent scalar CPU recurrence in 72 cases. It includes token and
row tails, K=32/96/160/640/2560/5120, signed quants, zero, subnormal and extreme
finite positive/negative FP16 scales. Existing frozen plugin Q8 references
remain unchanged. All 192 grouped nonzero differential cases pass against
the unchanged per-expert dispatch. The whole-model long oracle passes all
32 phases in each of resident, lazy-cache and uncached modes: each checks
4,491,225,056 BF16 and 16,716,198 F32/I32 words with zero differences.

Fresh before/after Flash samples use the same 4097-token prompt, five outputs,
one warmup and three measured runs, a 2 GiB source cache and shared caches
retained across restarted sessions. Filesystem-cache state is not independently
controlled:

| Placement | Before prompt ms | After prompt ms | Change |
| --- | ---: | ---: | ---: |
| Resident experts, host PLE | 1641.7 | 1354.7 | -17.48% |
| Stream experts from RAM | 4232.1 | 4213.8 | -0.43% |

The streaming difference is within the sample spread; no streaming speedup is
established. A separate 64-output resident-expert control reports medians of
72.83 and 72.89 output tokens/s. This CLI rate includes the prefill-sampled
token and is not isolated decode latency. Fixed case order and three samples
limit generalization. These Flash cases retain host PLE placement and do not
establish full-model GPU residency or matched vLLM performance parity.

Paired profiles cover two restarted prompts. Dense Q8 matrix duration falls
from 576.970 to 138.135 ms across 1,152 calls; grouped Q8 down duration falls
from 221.085 to 67.564 ms across 554 calls. Projection call counts and the
other 9,594 grouped-MMA calls are retained. Summed kernel durations include
profiler effects and do not measure SM occupancy or isolated request latency.
Tracked device/pinned peaks, expert transfers and inference allocation counts
are unchanged; tracked device heap and pinned host allocations return to zero.

The clean CUDA/MKL build has zero warnings. Standard CPU tests, the complete
CUDA suite and diagnostic isolation pass. Direct and grouped tests pass
memcheck/synccheck with zero errors. Whole-model uncached memcheck passes all
32 phases and the oracle with zero errors. Both Qwen27 CLI suites and the
request-state/isolation/restart/failure tests pass on the final rebuild.

A fresh original Qwen27 vLLM timing run was also made because both the saved
baseline and new native short-prompt timings exceeded the older reference's
5% threshold. The old timing snapshot is preserved. With the fresh reference,
short native TTFT is 52.911 versus 50.500 ms, a 4.77% gap. Other measured
prefill and decode cases are faster. This remains the explicit CP_FP32 plus
MKL VML HA, fully resident, single-request, 8192-token budget/context scope.
The fresh stock run has stable medium/long streams that differ from the
preserved arithmetic-profile streams beginning at indices 2/21. Current
correctness qualification uses the preserved profile-matched references,
which both native binaries match for every measured 64-token stream; timing
and fresh-stock bit identity are recorded separately. Commands, both timing
snapshots, all stream comparisons, profiles and checks are under
`/tmp/bitnet-goal-q8-mma/`.
### Bounded Flash expert supply diagnostics (2026-11-09)

`bench_flash_offload` measures ordinary Flash prefill and greedy decode with
explicit GPU expert and managed RAM cache budgets. Build with the same CUDA/MKL
flags as `bitnet`, then run:

```sh
./bench_flash_offload MODEL.gguf PROMPT.i32 64 3 0 128 128 12288 8192
```

Arguments after the prompt are output count, measured requests, warmup requests,
prefill chunk, maximum sequence length, GPU expert cache MiB and RAM cache MiB.
The prompt contains little-endian I32 token IDs. JSON records include configuration,
tokens, individual decode times and per-phase counter deltas. Caches persist across
restarted requests; warmup records retain chronological run indices. Decode timing
includes forward plus greedy selection and excludes the diagnostic getter gaps.
TTFT includes request restart and prefill. CPU fallback is rejected.

On the designated SM120 GPU and Flash Next Q4_K_XL checkpoint, a five-token prompt
followed by 64 outputs produced these unprofiled mean decode times (63 steps):

| Managed RAM cache | Request 0, empty expert caches | Request 1 | Request 2 | Request 2 per-token p95 |
| --- | ---: | ---: | ---: | ---: |
| 0 MiB | 27.39 ms | 18.91 ms | 18.91 ms | 29.09 ms |
| 2048 MiB | 41.98 ms | 22.93 ms | 22.89 ms | 41.79 ms |
| 8192 MiB | 50.99 ms | 20.72 ms | 17.74 ms | 25.09 ms |

All nine 64-token streams match. The GPU expert cache reserves 12,883,353,600
bytes and starts empty in each process. Core tensors, request buffers and host PLE
storage are additional allocations. The OS filesystem cache is uncontrolled and
is outside the managed RAM budget: PRead bytes do not establish physical disk I/O.
This is one repeated trajectory, without concurrency, and establishes neither
general workload performance nor vLLM parity.

GPU decode hit fractions are identical across RAM budgets: 86.04% on request 0
and 95.13% on requests 1/2. Corresponding raw expert H2D payloads are
13,280,768,000 and 4,739,481,600 bytes. The 2 GiB RAM cache misses on every GPU
miss, including request 2; the zero-cache control reads directly into pinned
staging and avoids managed-cache fill/copy overhead. With 8 GiB, source PRead
bytes fall to 2,613,760,000 on request 1 and zero on request 2. Larger cache cold
latency is higher; these measurements do not identify its full cause. No default
cache or runtime policy changed.

An independent Nsight Systems capture validates upload bytes and D2D cache-fill
bytes against actual GPU copy activities in all 192 phases. Fully warm request 2
transfers 4,739,481,600 H2D bytes in 88.90 ms of copy activity (about 53.3 GB/s),
with 2.28 ms of D2D fill activity. Its CPU source-wrapper span totals 188.97 ms
with no source reads. These spans can overlap and cannot be summed to infer
latency or isolated transfer stalls. The wrapper includes reads, cache handling
and CPU staging. Metadata/retry spans also include dependencies and caller gaps.
`upload_bytes` covers raw expert-bank uploads, not all H2D traffic. Prefill
intentionally bypasses GPU expert admission, and its D2D fill bytes are zero.
With the GPU cache disabled, misses are derived from submitted experts and raw
uploads; the eight-output control also preserves tokens. No inference runtime
allocations were observed in the trace.

Protected clean CUDA/MKL compilation and benchmark compilation have zero warnings;
standard tests and real-model runs pass. Evidence, commands, logs, comparison JSON
and the trace are under `/tmp/bitnet-goal-offload-supply/`. The next policy work
should evaluate admission and CPU staging on the GPU-conditioned miss stream,
including cold costs, before changing cache defaults.

An external linker wrapper subsequently captured the actual host expert-acquire
sequence without changing library code. Replay reproduces every observed RAM LRU
hit and source byte count in all 192 phases. At 537 slots (2 GiB), FIFO and
resident-only LFU also have zero warm hits. Inserting at the LRU end yields only
77 hits of 1,474 misses on request 2. A bounded ghost list of twice the slot count
with second-touch admission never admits an expert at this capacity: a second
touch alone is insufficient when the reuse distance exceeds the ghost budget.

Exploratory full-history frequency admission, comparing a candidate against the
least-frequent resident and retaining residents on ties, yields 537 request-2
hits and reduces simulated source reads from 4.739 to 3.032 GB. It also reduces
request-0 admitted payload from 13.281 to 1.700 GB. At 2,151 slots the replayed
LRU again matches the measured 8 GiB result: 655/1,474 warm hits, then 1,474/1,474.
These are counterfactual counts on a fixed GPU-miss sequence, not measured latency
gains. The frequency experiment has no aging, uses full-history metadata over
checkpoint IDs, and does not establish behavior under changing or concurrent
requests. Admission can avoid cache-fill copies but still needs a hit copy into
pinned staging. Runtime policy remains unchanged. Capture code, trace, validated
replay and policy definitions are under `/tmp/bitnet-goal-cache-policy/`.

### Measured RAM admission prototype (2026-11-09)

An external linker wrapper implements frequency admission with saturating U16
counters, LRU tie breaking and strict candidate-frequency comparison. The aged
variant halves counters every 4,096 decode GPU-miss requests. It uses 270,336
bytes of fixed metadata for this 48-layer/512-expert checkpoint, plus the existing
2 GiB RAM slab. Its global state is confined to the standalone serial experiment;
it is not a library implementation or a concurrent cache design. Prefill still
borrows without admission or promotion. No production code or defaults changed.

Two comparisons, reversing variant order on the second pass, preserve every
64-token output stream and every GPU hit/miss/upload/fill counter. Independent
replay exactly reproduces all 192 phase hit/read counts for both unaged and aged
repeated-request experiments. Results below are mean decode ms per step; pairs
are the two process measurements, not confidence intervals:

| Workload and request | Existing 2 GiB LRU | Aged admission, 2 GiB | No managed RAM cache, separate control |
| --- | ---: | ---: | ---: |
| Repeated, request 0 | 43.29 / 42.71 | 32.75 / 33.82 | 27.36 |
| Repeated, request 2 | 22.86 / 22.89 | 19.07 / 18.87 | 19.12 |
| Alternating, request 0 | 42.27 / 42.91 | 32.62 / 32.62 | 27.66 |
| Alternating, request 4 | 26.50 / 26.91 | 22.10 / 21.81 | 20.90 |

Alternating requests swap the second/fourth valid token IDs on odd requests and
restore the original five-token prompt on even requests. This is a controlled
changing-routing test, not a representative serving workload. Aged repeated
request 2 has 507 RAM hits and reads 3.120 GB rather than LRU's 4.739 GB; alternating
request 4 has 244 hits and reads 5.946 GB rather than 6.744 GB. The unaged repeated
request reproduces the predicted 537 hits/3.032 GB and measures 18.51 ms at
request 2, but has no adaptation/aging qualification.

The admission experiment improves the undersized LRU control by about 17–19%
at the final warm requests, yet does not establish an improvement over direct
PRead into pinned staging. The no-cache control is faster cold and in the
alternating case despite reading more bytes. Read savings alone are therefore
insufficient grounds to integrate this policy. Pinned-staging copies, admission
costs and memory budgets still matter. The 8 GiB fully warm cache result remains
a separate larger-memory placement point. All tests are serial, filesystem cache
is uncontrolled, and isolated critical-path stalls remain unmeasured. Commands,
source, raw logs, exact replay and comparisons are under
`/tmp/bitnet-goal-cache-admission/`. Production integration of this frequency
prototype is deferred pending an advantage over the no-cache control.

### Serialized multi-request CUDA baseline (2026-11-09)

`bench_bf16_concurrency` keeps several backend-only Qwen27 sessions live on one
shared immutable model and admits queued requests when a session finishes. It
executes one decode per ready request, followed by one round-robin prefill chunk.
All requests arrive together. The implementation executes each session call
separately; it does not implement fused cross-request decode or production
continuous batching. This baseline measures what those changes must improve.

```sh
./bench_bf16_concurrency MODEL.gguf PROMPT.i32 64 8 4 128 2048 uniform
./bench_bf16_concurrency MODEL.gguf LONG_PROMPT.i32 64 8 4 128 2048 mixed
```

Arguments are output count, request count, maximum live sessions, prefill chunk,
maximum sequence length and optional `uniform`/`mixed` mode. In mixed mode odd
requests use five tokens while even requests retain the full prompt. Each request
also swaps valid prompt IDs deterministically to exercise different state. Before
timing, every request runs in isolation with the identical prompt, CP_FP32 profile,
MKL VML HA and chunk boundaries. Every interleaved output must match this reference.
This is an isolation/continuation regression check, not an independent model oracle.
CPU fallback is rejected.

Measured time excludes shared-model loading, tokenization, isolated reference
execution and JSON output; it includes request initialization, admission, decode,
prefill and session teardown. Per-request TTFT and completion include queue time
from the common arrival instant. Delivery timestamps expose inter-token latency
including interference from other requests. The slot limit bounds concurrently
allocated session contexts; it is not a byte-accounted KV allocator or paged KV.
`reserved_context_slots_peak` excludes recurrent states, workspace and weights.
The benchmark does not implement cancellation, HTTP or arrival-rate workloads.

On the designated SM120 GPU, eight requests with 129-token prompts, 64 outputs,
128-token chunks and 2048-token capacity show:

| Maximum live sessions | Aggregate output tokens/s | Per-token inter-delivery p95 |
| --- | ---: | ---: |
| 1 | 53.16 | 15.87 ms |
| 2 | 52.54 | 32.15 ms |
| 4 | 51.60 | 65.39 ms |
| 8 | 51.05 | 131.94 ms |

These single-sample cases retain identical per-request token streams across all
four concurrency limits. Throughput includes the prefill-sampled first output.
P95 pools the 504 decode delivery intervals; it is distinct from request-level
TTFT/completion tails. No concurrent vLLM parity claim follows from these results.
Evidence and commands are under `/tmp/bitnet-goal-serving-baseline/`.

Mixed cases alternate 1,025-token and five-token prompts. At four live sessions,
increasing prefill chunks reduces the number of pauses but lengthens the largest:

| Prefill chunk | Output tokens/s | Inter-delivery p95 | Inter-delivery p99 | Maximum interval |
| --- | ---: | ---: | ---: | ---: |
| 128 | 34.19 | 242.27 ms | 243.46 ms | 251.96 ms |
| 512 | 36.75 | 73.68 ms | 672.72 ms | 695.45 ms |

The lower p95 with 512-token chunks does not mean its longest pauses improved;
fewer prefill calls shift which intervals fall below that percentile. These are
single-sample observations, not a general optimal chunk policy. All eight cases
(uniform at 1/2/4/8 live; mixed at 1/2/4 live with chunk 128 and at four live with
chunk 512) match 4,096 generated tokens to isolated execution. Streams also remain
identical across concurrency limits with the same chunk schedule. Clean CUDA/MKL
build, standard tests, warning-free benchmark compilation, CPU unsupported-build
stub and argument validation pass. The four existing user probe binaries retain
their original hashes. No production runtime/kernel changes were required for
this baseline; production scheduling, KV allocation and batched execution remain
unfinished.

### Independent-sequence GDN decode batches (2026-11-09)

`BnGPUGDNDecodeResidentPlan.n_sequences` now supports packed sequence-major
one-token decode batches. Zero/one retains the original single-sequence kernel
specialization; larger batches launch independent sequences on the second grid
axis. Each sequence has separate F32 initial/next state and BF16 input/output;
geometry and immutable weights are shared. The original reduction, FMA and
rounding contract is preserved. All views remain owned by the supplied backend
request. This does not reinterpret a batch as a multi-token recurrence.

The new `--bf16-gdn-decode-batch-reference` CUDA test covers batches 1/2/3/8/16,
three head geometries including HK16/HV48, three input/state seeds and five
dependent launches. Sequences begin at different points of the preserved
eight-step unmodified-vLLM trajectory, checking distinct state and input rows.
All 408,672,000 output/state words match frozen reference hashes over 225
launches. Initial states remain unchanged, boundary guards survive, and invalid
counts, extents, aliases and ownership reject without publishing output/state.
Neither the independent reference header nor its generator was changed.

Nsight Systems confirms 180 grouped kernel launches covering 1,305 independent
sequence updates: 45 launches each at grid Y=2/3/8/16. Batch-one and reference
warmup calls retain the original specialization. This capture validates batching
structure; its host fixture reads/hashes do not establish serving latency or
throughput gains.

Protected clean CUDA/MKL build has zero warnings; standard tests, the complete
CUDA suite and diagnostic isolation pass. Batch memcheck and synccheck report
zero errors. The preserved dense 64-token CP_FP32/MKL stream and six interleaved
16-token streams remain exact after rebuilding. User probe binary hashes are
preserved. Commands, logs, unchanged-reference hashes, trace and qualification
are under `/tmp/bitnet-goal-gdn-batch/`.

This operator is not yet used by the serving runtime. Efficient integration
still needs validated access to individual sessions' state, shared-row
projections, sequence-aware attention and KV scheduling. Packing full recurrent
matrices into a temporary batch on every decode step would add substantial
traffic; the packed arithmetic contract alone does not solve that ownership and
placement problem. No concurrent vLLM performance gate is established here.

### Direct session-state GDN groups (2026-11-09)

The CUDA backend now has a bounded group coordinator that runs GDN directly
against separate participant requests' buffers. `BnGPUGDNDecodeGroupPlan` lists
one token per distinct live participant. State views are validated against that
participant's registry; the shared-row extension below also permits explicitly
coordinator-owned activations. Geometry and immutable weight handles are shared. The
coordinator owns preallocated pinned/device pointer descriptors and CUDA events,
while participants retain ownership of F32 recurrent state.
No full state matrices need to be gathered or scattered for this operator.

Setup queries/reserves descriptor payload for 1–256 participants. The budget
includes descriptor storage, host event table and container, excluding driver,
event and allocator/page overhead. Capacity is fixed; matching reserve is
idempotent. Execution validates the entire group before queueing work, records
producer-ready events, joins them on the coordinator stream, copies descriptors,
launches one kernel and orders subsequent participant work after its done event.
Pinned descriptor reuse can return `BN_GPU_GROUP_BUSY`; retry remains healthy.
There is no explicit host fence or allocation on normal submission. Partial
device submission failure fences the coordinator and invalidates participants.
Callers serialize access and retain immutable weights until pending work ends.
Output/next-state publication requires successful participant and coordinator fences,
including upstream logical error checks.

The `--bf16-gdn-decode-group-reference` test uses separate requests, async pinned
input uploads and queued input consumers. Batches 1/2/3/8/16 cover all nine frozen
GDN fixtures with staggered trajectory positions. All 408,672,000 output/state
values hash-match the unchanged independent references; previous state, guards
and rejection atomicity pass. Coordinator teardown while participant work is
pending also passes. A bounded queued-memset case observes descriptor BUSY in
the normal run and verifies recovery; sanitizer runs also accept the queue
draining before that observation. One duplicate retry-check update makes 226
kernel launches in total.

Independent Nsight tracing confirms 226 direct kernels covering 1,351 sequence
updates on 45 coordinator streams. Those streams copy exactly 54,040 descriptor
bytes across 226 H2D copies, with zero activation/state copies. No CUDA runtime
allocations occur between each case's first and last group kernel. Fixture setup
and participant upload/readback traffic are separate; these correctness captures
do not establish serving latency or throughput gains.

The final protected clean build and benchmark build have zero warnings. Standard
tests, the complete CUDA suite and diagnostic isolation pass. Memcheck and
synccheck report zero errors. The preserved dense 64-token stream and six
interleaved 16-token streams remain exact. User probe binaries and independent
reference files retain their hashes. Evidence, final logs and trace are under
`/tmp/bitnet-goal-gdn-direct/`; `initial/` contains checks before the final
busy-path cleanup and is not the final qualification.

Whole-model batched serving remains unfinished. Shared-row projections will
also need explicitly validated batch-owned activation rows or efficient row
gather/scatter, followed by sequence-aware convolution, RoPE, attention and KV
scheduling. State ownership checks must remain strict as that access expands.

### Shared activation rows with session-owned GDN state (2026-11-09)

GDN groups now accept BF16 X/BA/output rows owned either by their participant or
by the explicit coordinator. F32 initial/next states remain strictly participant
owned. Other request owners are rejected; the generic view resolver is unchanged.
This lets a future batched projection leave its result in coordinator storage
for GDN to consume directly. Every output/next-state write is checked against
every other row's inputs and writes before submission, preventing overlap through
a shared allocation. Read-only input sharing is permitted. Both coordinator and
participant fences remain necessary to detect upstream logical errors.

The shared-row test uses guarded packed activation buffers, independent request
states, async uploads and queued consumers. All 408,672,000 output/state values
match the unchanged frozen GDN hashes across batches 1/2/3/8/16. The original
participant-owned mode also passes, totaling 817,344,000 values in the two group
modes. Added rejection cases cover otherwise valid coordinator-owned F32 state,
overlapping output rows and a row writing another row's input. Numerical checks,
state immutability, guards and rejection atomicity pass.

Tracing confirms 226 indirect kernels and 54,040 bytes of pointer-descriptor
uploads. There are no D2D copies on coordinator streams and no CUDA runtime
allocations between each case's first/last group kernel. Unlike the earlier
participant-input fixture, this fixture also uploads packed X/BA and reads back
packed outputs on coordinator streams; those verification transfers are counted
separately from descriptors. This capture does not measure a serving speedup.

The clean build and full CUDA build have zero warnings. Standard tests, complete
CUDA suite, diagnostic isolation, memcheck and synccheck pass. Preserved dense
64-token and interleaved six-by-16-token streams remain exact. Reference files
and user probe hashes are unchanged. Commands, logs, trace and qualification are
under `/tmp/bitnet-goal-gdn-rows/`. Actual batched projection composition,
convolution history, per-sequence RoPE/attention and KV/admission scheduling
remain required for whole-model serving.

### Batched projection → GDN composition (2026-11-09)

`--bf16-projection-gdn-reference` now executes two real BF16 projections into
coordinator-owned packed X/BA rows, followed by grouped GDN using each request's
F32 state. Selector matrices reproduce eight controlled input steps through
GEMM; batches 1/2/3/8/16 begin at staggered trajectory positions. Produced X/BA
stay on the GPU between projection and recurrence. The test uses the existing
decoded BF16 GEMM and shared-state APIs; production kernels are unchanged.

This composition exposed a reference-contract difference. GEMM canonicalizes
negative zero in the synthetic inputs. All 72 BF16 output hashes remain equal
to the direct-input fixture, but five F32 continuation-state hashes change.
The original reference header/generator remain unchanged. A separate
`reference_cuda_bf16_projected_gdn.py` runs actual Torch BF16 selector GEMMs and
the unmodified pinned vLLM recurrent kernel. It verifies projection bits across
all tested batch sizes, exports eight dependent steps and produces the new
`cuda_bf16_projected_gdn_reference.h`. Native fixture warmup uses the independently
verified zero-sign contract; the composed rows themselves come from GPU GEMMs.
No production zero normalization or approximate comparison was introduced.

All 408,672,000 composed output/state values hash-match that independent
reference. Original participant/shared-input modes still pass their original
references. GPU tracing identifies 450 projection kernels: 90 singleton GEMV
and 360 batched small-N matrix kernels, feeding 226 grouped GDN launches. The
coordinator transfers 21,600 selector bytes and 54,040 pointer-descriptor bytes;
there are no coordinator D2D copies. No CUDA runtime allocations occur between
each case's first/last GDN kernel. Fixture initialization and correctness
readbacks are additional traffic, and verification fences make this a
correctness test rather than a serving performance benchmark.

The protected clean build, standard tests, complete CUDA suite and diagnostic
isolation pass with zero compilation warnings. Memcheck/synccheck report zero
errors. Preserved dense 64-token and six interleaved 16-token streams remain
exact; user probe hashes are preserved. Reference provenance includes the pinned
image, Torch version and unchanged vLLM kernel source hash. Evidence is under
`/tmp/bitnet-goal-projection-gdn/`, including the initial signed-zero mismatch.

This qualifies synthetic decoded-BF16 projection composition. Quantized
checkpoint projections, convolution history, per-sequence RoPE/attention,
KV/admission scheduling and whole-model batched serving remain unfinished.

### Real checkpoint quantized projection/GDN composition (2026-11-09)

`bench_checkpoint_gdn` is an optional real-tensor correctness gate. It loads the
independently replayed Qwen27 layer-0 Q5_K QKV/Z, Q8_0 BA/output, F32 log-decay
and BF16 bias parameters. Its default controlled graph is QKV/Z+BA projection → raw
GDN → output projection. The optional convolution capture adds independent
decode histories before GDN. Both modes exclude gated norm/head layout,
residuals and FFN; neither is a complete linear-attention block.

```sh
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=PATH bench_checkpoint_gdn
./bench_checkpoint_gdn REFERENCE_DIR LINEAR_CAPTURE_DIR GDN_PREP_DIR [CONV_CAPTURE_DIR]
```

The reference directory comes from `test/reference_cuda_checkpoint_gdn.py`
inside the pinned vLLM image with the e454affe GGUF plugin and qualified compatible
extension. Generation reads immutable captures and calls the original GGUF
projection and recurrent functions; expected values never come from native
output. A second run of the published generator reproduced all 248 binary files
exactly. Provenance records parameter, extension and GDN source hashes.

Batches 1/2/4/8 run five dependent steps from both zero and distinct nonzero
initial state. Native QKV/Z, BA, GDN output, F32 next state and final projection
are compared byte-for-byte after graph submission. All 122,126,400 values match,
including 117,964,800 F32 state values. Intermediates remain on the GPU between
the three quantized projections and grouped recurrence.

Nsight records 40 grouped GDN launches covering 150 independent sequence updates.
Dispatch retains Q5_K/Q8_0 vector kernels at batches 1/2, selects Q5_K pair-split
execution at batches 4/8, and Q8 integer MMA at batch 8. Coordinator H2D traffic
is 1,536,000 hidden-input bytes plus 6,000 pointer-descriptor bytes. There are no
coordinator D2D copies or CUDA runtime allocations between each case's first/last
GDN kernel. Setup weight/state uploads and correctness readbacks are additional
traffic. This fixture does not establish latency or throughput gains.

Protected clean build, standard tests, warning-free gate compilation, CPU
unsupported-build stub, native byte comparisons, memcheck and synccheck pass.
The published reference reproduces the original capture exactly and user probe
hashes are preserved. Production runtime source is unchanged from the earlier
qualified build; the full CUDA suite was not repeated for this gate-only addition.
Commands, reference captures, source provenance, trace and qualification are under
`/tmp/bitnet-goal-checkpoint-gdn/`. Whole-model batching still requires independent
convolution history, per-sequence RoPE/attention, KV allocation and scheduling.

### Independent-history grouped convolution (2026-11-09)

CUDA now supports one-token convolution groups with independent session-owned
BF16 histories and participant/coordinator-owned activation rows. Each group
launches one kernel across sequence rows. Histories, outputs, extents and owners
are validated before submission, including cross-row write aliases. The generic
view resolver remains strict. Existing convolution/GDN arithmetic is preserved.

Convolution and GDN coordinators use separate bounded descriptor workspaces.
Shared queue helpers implement ready/done event joins, pinned-copy backpressure,
error fencing and participant poisoning. Separate metadata permits immediate
convolution → GDN submission while earlier metadata remains in flight. Each
workspace still has bounded BUSY behavior on reuse; this is not an unbounded
multi-layer submission queue.

The new `--bf16-conv-group-reference` gate staggers request progress through the
unchanged five-token reference trajectories. Active row lists change while
histories occupy different positions. Batches 1/2/3/8/16, widths 2/3/4, eight
channel dimensions and zero/nonzero initial history pass 1,536 launches,
7,200 sequence updates and 13,992,300 frozen output/history words. Guards,
previous-history immutability, FIFO contents and rejection atomicity pass.

The optional checkpoint gate now accepts a convolution capture directory, and
`reference_cuda_checkpoint_gdn.py --conv CONV_CAPTURE_DIR` generates its
independent counterpart. Actual Qwen27 Q5_K/Q8_0 projections → convolution →
GDN → Q8_0 output projection match all 128,270,400 values byte-for-byte at batches
1/2/4/8 over five dependent steps, including both BF16 histories and F32 states.
The original graph without convolution still passes its 122,126,400 values.
Gated normalization/layout and residual/FFN layers remain outside these graphs.

The checkpoint trace confirms 40 convolution groups and 40 GDN groups covering
150 sequence updates. Coordinator H2D payload comprises 1,536,000 hidden-input
bytes and 12,000 descriptor bytes. Coordinator D2D copies and CUDA runtime
allocations between each case's first/last GDN kernel are zero. Fixture
readbacks, setup uploads and explicit validation fences are additional work;
this does not establish serving latency or throughput gains.

Protected clean compilation, standard tests, complete CUDA suite and diagnostic
isolation pass without warnings. Both the staggered convolution gate and real
checkpoint chain pass memcheck/synccheck with zero errors. Preserved dense and
interleaved token streams remain exact. Evidence is under
`/tmp/bitnet-goal-conv-group/`. Next integrate gated normalization/layout and
per-sequence RoPE/attention, then whole-model batching and KV/admission scheduling.

### Complete BF16 benchmark arithmetic policy

`bench_bf16_latency` also accepts `--policy POLICY.json` after the seven
required arguments, in place of the legacy arithmetic/mask arguments:

```sh
./bench_bf16_latency MODEL.gguf PROMPT.i32 64 3 1 8192 8192 --policy POLICY.json
```

The JSON object must contain exactly `arithmetic`, `ffn_arithmetic`,
`input_norm_reductions` (65 integers), and `post_norm_reductions` (64 integers).
Values use the enums in `src/transformer/bf16_internal.h` and
`include/gpu_backend.h`: arithmetic 0–4, FFN 0–2, reductions 0–6. Zero entries
inherit defaults. The reader rejects duplicate/unknown fields, incorrect array
lengths, invalid numbers, truncation, embedded NULs, trailing data, and files
of 16 KiB or more before reading the prompt or model. Runtime admission still
checks the selected policy against backend capabilities.

The `bf16_policy` JSONL record contains the complete `admitted` policy copied
from the runtime, including resolved defaults. It replaces the partial
`norm_policy` mask record; the legacy command-line arithmetic and mask syntax
remains supported. Keep this record with timing results: the base arithmetic
label alone cannot identify the selected FFN and normalization contracts.
The CPU wire-format checks run as `make test_bench_bf16_policy` and as part
of `make test`. Independent reference qualification and the remaining
performance gates are described in `docs/qwen38-cuda-correctness.md`.

`bench/policies/qwen38-cuda4097.json` records the explicit plan qualified
against `test/bf16_cuda4097_reference.json`. It is scoped to that independent
reference variant and is not a universal compiler policy or a production
default. The earlier cold-compiled reference uses a different input-norm plan.

Qualification of the policy-file path reproduces all 64 frozen tokens for
each of short/medium/long/repeat4097, both on the initial warmup and after
request restart (512 token comparisons). Each emitted admitted policy matches
the reviewed file exactly. A separate legacy/file replay verifies that default
entries are reported as the runtime's canonical choices. These checks validate
benchmark configuration and token replay; they do not capture fresh raw logits
or establish vLLM timing parity. Evidence: `/tmp/bitnet-goal-bench-policy`.

### Complete policy for batched decode diagnostics

`bench_bf16_batch` accepts `[uniform|mixed] [--policy POLICY.json]` after
its required arguments. It uses the same strict four-field policy format as
`bench_bf16_latency`. Every isolated reference request, failure/rejection
self-test request, serving request, and decode coordinator receives that
policy. Without a file it retains its existing CP_FP32 defaults. The emitted
`bf16_policy.admitted` record describes the coordinator's canonical plan;
`config.arithmetic` is its numeric enum value. Policy parsing occurs before
prompt/model reads.

This harness compares native batched greedy streams with native isolated
execution. Its batch-of-one self-test additionally checks complete logits,
rejection atomicity, and failure poisoning. A successful native token check
is not independent multi-request qualification: GEMM shapes can change
arithmetic, and the round-robin separate-prefill schedule differs from a
reference engine that packs all prompts into one prefill call.

The explicit 4097-reference policy passes native two-request 64-output checks
with uniform 129-token prompts and mixed 129/5-token prompts using 64-token
prefill chunks. Both retain the full-logit batch-of-one and failure-atomicity
self-tests. A four-request 64-output run diverges from isolated execution at
output 14; with legacy defaults, it diverges at output 57. Neither four-request
result is admitted as qualified serving performance. The unchanged kernels
and differing GEMM shapes require an independent schedule-matched comparison.
Evidence and emitted policies: `/tmp/bitnet-goal-batch-qualification`.


### Flash bank dependency reporting (2026-11-10)

`bench/analyze_flash_trace.py --dependencies` now adds a conservative exposed
bank-dependency lower bound, separate from copy activity durations and host
source spans. The existing bounded cold/warm capture has 18,789 matched waits;
all consumers are grouped expert kernels and positive intervals do not overlap.
The decode lower bounds average 3.748 ms cold and 1.335/1.354 ms on the two
repeated requests. These are not total source stalls or latency savings.
Six independent two-stream controls distinguish exposed waits, hidden work and
late CPU submission using CUDA event timings. Existing 192-phase metrics remain
unchanged. No runtime policy changes follow from this diagnostic. See
[method, calibration and scope](qwen38-flash-dependencies.md).


### Host expert-staging experiments (2026-11-10)

External linker-wrapper trials tested two/four-thread copies, AVX2 streaming
stores, upfront cache-page preparation and transparent huge-page advice. All
81 streams and 5,184 GPU cache phase records match controls. Two-thread copies
produce only about a 1% fully warm gain at 8 GiB; undersized-cache variants still
trail direct reads. Upfront page preparation worsens combined startup cost, and
huge-page cold results are inconsistent despite fewer faults. No runtime defaults
changed. See [results and rejected hypotheses](qwen38-flash-staging-experiments.md).
