# Bounded arrival runtime for Qwen3.8 27B

The library scheduler in `include/generate_batch.h` provides serialized greedy
continuous batching. Its CUDA adapter now reproduces all 134 independently
captured engine steps and all 784 full BF16 logit heads while using eight
pooled sessions for twelve requests. This establishes the controlled arrival
case's execution and KV payload bound. Three uninstrumented timing runs now
meet both performance floors for this controlled arrival case, with all
independent full-head references preserved.

The fixed wall-clock extension now supports an optional `mixed_token_budget`
while preserving the full prefill-only budget. Its measured cap8 configuration
reduces the previous tail spike, but p99 remains 5.12% above the matched vLLM
configuration. See [independent controls and results](qwen38-cuda-decode-aware.md).
The controlled-step qualification above and the fixed-clock result have distinct
scopes.

The subsequent [one-token prefill specialization](qwen38-cuda-single-token-prefill.md)
now passes the bounded twelve-request case: p99 is 69.41 versus 69.56 ms, with
all independent references preserved. Sustained serving qualification remains open.

## Library behavior

Initialize a zeroed `BnGenerateBatch` with immutable `BnGenerateBatchLimits`
and backend callbacks for acquiring a session, executing a cohort, returning a
session, and reading a monotonic clock. The runtime owns a fixed 64-record FIFO
queue and at most sixteen live leases. Prompts and output arrays remain caller
owned and borrowed until a terminal event or runtime destruction.

`bn_generate_batch_enqueue` accepts a new arrival, validates token IDs and array
ranges, and applies queue backpressure. `bn_generate_batch_step` admits queued
requests while leases are available, reserves one token per selected decode,
and divides the remaining token budget among bounded prefill chunks. Query
length ordering and recurrent execution roles come from `generate_batch_plan`.
The caller's input does not supply captured chunk sizes or roles.

Execution completes before greedy sampling. Every selected sample must be valid
before the runtime updates any output prefix or request progress. Token events
share one cohort timestamp and all occur before completed sessions return to
the pool. Cancellation works between steps. An execution or sampling failure
preserves previously published output prefixes, discards all live execution
states, and reports terminal failure for pending requests. The failed runtime
must be reinitialized. Calls are serialized; callbacks may inspect `info()` but
must not reenter mutating APIs.

The scheduler itself does not allocate per-step scratch. Backend admission can
allocate lazily within its budget. This API currently provides greedy sampling;
it does not implement stochastic sampling or a network server. The ordinary
`enqueue` timestamps worker submission. `enqueue_at` preserves an earlier
external arrival in the backend clock domain, including time spent waiting
while an engine step prevents enqueueing.

## Controlled workload and independent evidence

Four requests arrive at each of engine steps 0, 8, and 68. Their prompt lengths
are 129, 5, and 129 tokens respectively, with 64 outputs each. The live limit is
eight, the token budget is 256, the chunk bound is 64, and up to four prefills
share a step. Context capacity is 8192. These are controlled engine-step
arrivals, not a fixed wall-clock offered load. Both runtimes use the same GGUF
checkpoint, BF16 activations, F32 recurrent state, disabled CUDA graphs, and the
explicit captured arithmetic policy. Qualification is on CUDA SM120 with 188
SMs; it does not establish performance on other devices or model formats.

`bench/prepare_bf16_arrivals.py` converts the independent reference into BGA1.
It writes arrivals, prompts, limits, and greedy expectations. Expected outputs
are used only for validation after the timed execution. Captured query lengths,
positions, and execution roles are excluded from the workload. The library
planner and actual greedy outputs determine all subsequent execution inputs.

The CUDA adapter in `bench/bench_bf16_arrivals.c` calls the existing transformer
batch executor and caller-owned KV pool. Its actual generation matches all
134 matrices, 784 heads, and 194,682,880 BF16 words. This includes partial
prefill heads and late arrivals after session reuse. It creates eight sessions,
reuses four, discards none, and finishes with no outstanding leases.

Each slot charges 512 MiB of KV payload; eight retained slots charge exactly
4 GiB, including idle slots. This budget excludes model weights, SSM state,
activations, backend workspace, driver allocations, and allocator overhead.
It does not establish a total device-memory bound or paged/variable-capacity KV
scheduling.

The model-free runtime test also reproduces all 134 captured cohorts. It checks
queue bounds, cancellation, pool exhaustion, publication before reclamation,
and atomic failure both before and after a valid output prefix. ASan/UBSan,
the protected clean build with zero warnings, and the standard suite pass.
The log verifier rejects tampered geometry, identity, token, timestamp, and
pool-accounting records. The full CUDA suite and diagnostic isolation pass.
The changed Q8 kernel passes 72 CPU/DP4A differential cases, focused memcheck
and synccheck, and the unchanged pinned-plugin backend references under
memcheck. The actual runtime and all 784 heads remain exact; legacy batch-four
and batch-sixteen replays preserve all 256 and 1024 full heads respectively.

The 134-step whole-workload memcheck passed before the Q8 optimization. Runtime
ownership and admission code are unchanged; the changed kernel's bounds and
synchronization are checked by the focused tests above. See
[machine-readable evidence](qwen38-cuda-arrivals-runtime.json).

## Reproduction

Build with the project's qualified CUDA and MKL flags, then prepare the workload:

```sh
python3 bench/prepare_bf16_arrivals.py \
  test/bf16_cuda_staggered12_reference.json /tmp/arrivals.bin
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 \
  MKL_VML_ROOT=/tmp/bitnet-goal-bf16-vml/mkl bench_bf16_arrivals
mkdir -p /tmp/arrivals-logits
```

Set the designated CUDA device as required by the local qualification setup.
With `MODEL` set to the qualified GGUF checkpoint, run the diagnostic:

```sh
./bench_bf16_arrivals "$MODEL" /tmp/arrivals.bin \
  bench/policies/qwen38-cuda-staggered12-noncp.json /tmp/arrivals-logits \
  > /tmp/arrivals.jsonl
python3 bench/verify_bf16_arrivals.py /tmp/arrivals.jsonl \
  --reference test/bf16_cuda_staggered12_reference.json
python3 test/verify_bf16_mixed_reference.py /tmp/arrivals-logits \
  --reference test/bf16_cuda_staggered12_reference.json
```

Replace the output directory argument with `--timing` for buffered timing mode.
Diagnostic and profiler runs are excluded from performance results.

## Qualified timing result

Three-run medians for the controlled arrival case are:

| Metric | Native | Matched reference |
| --- | ---: | ---: |
| Initial three pure-prefill steps | 721.256 ms | 1388.994 ms |
| Mean decode step, 127 steps | 47.146 ms | 49.307 ms |
| Four-request decode steps, 69 steps | 37.459 ms | 37.125 ms |
| Eight-request decode steps, 58 steps | 58.670 ms | 63.801 ms |
| End-to-end output throughput | 101.870 tokens/s | 83.116 tokens/s |
| p95 time to first token | 721.255 ms | 1471.062 ms |
| p99 inter-token latency | 298.030 ms | 683.918 ms |

Initial pure-prefill throughput is 1.926x the reference; mean decode latency is
4.4% lower. Both decode cohort sizes individually meet the 5% latency floor.
The four mixed steps contain both prompt and generation work and are not
pure-prefill measurements. TTFT and ITL entries are medians of nearest-rank
percentiles over twelve and 756 samples per run respectively. Native step
durations include enqueue/admission and pool returns; reference per-engine-step
phase times exclude `add_request`, while its whole-loop elapsed includes it.
Both timings include their CPU metadata recording; neither reads GPU tensors
for observation. Final pool/model teardown is outside the native timer.
See [all runs and gates](qwen38-cuda-arrivals-performance.json).

The earlier baseline missed decode parity, averaging 78.645 ms for eight
requests versus 63.801 ms reference. Profiling isolated 60 Q8 matrix calls per
eight-request step. The SM120 small-batch kernel now stages eight consecutive
Q8 blocks per barrier pair, with a 64-token cooperative tile that zeroes inactive
rows. Integer dots, scale products, and ascending per-block F32 FMAs retain
the original precision contract. Larger Q8 matrix paths and other architectures
retain their existing dispatch.

In matching profiles, the 3,480 affected calls average 71.571 microseconds
instead of 396.301 microseconds. Transfer byte counts and synchronization call
counts are unchanged. The window from first token upload through last output
download transfers 6,677,056 bytes host to device, 389,370,552 device to host,
and 4,235,886,592 device to device. Initial preparation/admission is excluded;
later admissions are included. Of the device-to-host bytes, 389,365,760 are the
required output heads. The profile records 1,209 stream synchronizations and
200,704 stream event waits. Runtime API durations include implicit blocking
and are not isolated GPU stall measurements. The CUDA adapter installs a guard
that rejects CPU fallback. See [profile evidence](qwen38-q8-small-batch-profile.json).

The [fixed wall-clock case](qwen38-cuda-wall-clock.md) now preserves external
arrival queueing and has an independent compute reference. Its measured p99
inter-token latency remains worse than vLLM, so serving parity is still open.
Variable-capacity or paged KV scheduling, total device-memory budgeting, and
the remaining model/backend/offloading qualifications are also open. This
controlled case does not complete the full goal.
