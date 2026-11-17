# Request-aware batch admission and KV capacity

The generic batch runtime now supplies request metadata to an opt-in acquisition
callback. The CUDA arrival adapter uses it to select request-sized KV leases.
Independent full-head controls remain exact. The measured memory reduction comes
with a cold-admission tail tradeoff; sized mode remains opt-in. This budgets KV
payload, not all device memory.

The measurements below describe the original session-recreation implementation.
[Idle KV growth](qwen38-idle-kv-resize.md) subsequently preserves non-KV buffers
and qualifies the same controlled replay without the measured p99.9 regression.
The original evidence remains recorded here.

## API and ownership

`bn_generate_batch_init_request_aware` accepts a `BnGenerateBatchAcquireRequest`
callback and the existing backend context. The descriptor contains the request
ID, prompt length, maximum outputs and `required_capacity = prompt + outputs - 1`.
The final generated token is not another model input. Metadata is immutable and
borrowed only for the callback. The callback may acquire a suitable pooled lease;
its success, BUSY, failure, ownership and cleanup contracts match legacy acquire.

The original `BnGenerateBatchBackend` struct layout and initializer are unchanged.
The new constructor ignores its legacy acquire pointer, which may be NULL. Other
callbacks retain their contracts. FIFO admission, sample validation, publication,
reclamation, failure cleanup and cancellation use the same runtime code paths.
The generic module gains no GPU/model dependency.

The CUDA adapter selects sized admission with a final `--sized-kv` argument.
It queries each lease's actual backend capacity and buffers inventory snapshots
at every acquisition/release. The verifier checks capacity bounds, unique lease
lifetimes, active-slot immutability, idle reclamation, creation/reuse/eviction
counts, retained bytes and peak bytes. Idle allocation can exceed peak live
concurrency and is still charged. Oversized idle reuse retains its actual charge.

## Independent model and memory checks

Configured limits remain eight live slots, maximum context 8192, chunk 64, and a
4 GiB KV budget. The twelve-request control uses the previous full token budget;
the 48-request control retains mixed cap 8. Neither policy nor expected neural
values changed. Each request reserves its complete prompt/output trajectory.

| Case | Full heads checked | KV peak/retention | Created / reused / evicted |
| --- | ---: | ---: | ---: |
| Legacy fixed twelve requests | 784 | 4 GiB | 8 / 4 / 0 |
| Sized twelve requests | 784 | 96 MiB | 8 / 4 / 0 |
| Sized 48-request controlled replay | 4,132 | 512 MiB | 20 / 28 / 12 |
| Sized 48-request fixed-deadline diagnostic | 4,136 | 512 MiB | validated per transition |

The twelve-request pool retains four 256-token and four 128-token slots. The
48-request pool eventually retains eight 1024-token slots. The controlled replay
matches all 1,026,058,240 BF16 words; the fixed-deadline diagnostic matches
all 1,027,051,520 words. Both use the unchanged independent original model-math
controls from [the 48-request qualification](qwen38-cuda-load48.md), whose numerical
host-selection adapter is explicitly separate from original performance runs.
All six timing traces also match their controlled geometry and greedy streams.

## Matched before/after timing

Three paired runs use the identical 572-step controlled arrival schedule and
model/precision/context settings. One pair reverses order. Preparation of model
weights is outside the timer, while request admission, allocation/reclamation,
execution, sampling and pool returns are inside. Timing captures no full heads;
slot snapshots are host metadata and buffered until the end. This is a pool-mode
comparison, not a fresh vLLM or steady-state offered-load qualification.

| Three-run median | Fixed capacity | Sized capacity |
| --- | ---: | ---: |
| Output tokens/s | 69.39 | 69.52 |
| p99 inter-token latency, ms | 70.28 | 70.17 |
| p99.9 inter-token latency, ms | 95.90 | 113.61 |
| Intervals above 100 ms per run | 0 | 8 |
| Retained KV payload | 4 GiB | 512 MiB |

Throughput and p99 are essentially unchanged. **p99.9 is 18.5% higher.** The long
gaps coincide with admitting four newly allocated larger leases at calls 66 and 214
while four decodes are active. Those correlations do not isolate an allocation
or transfer stall duration. Tails use nearest rank over 3,024 intervals per run,
then a three-run median; p99.9 has a small effective sample. The extra creation
and reclamation cost needs further tuning rather than being hidden by p99 alone.
No default mode changed and no total-VRAM or all-tail parity claim is made.

## Validation and reproduction

Clean CUDA/MKL build passes with zero warnings and protected user binaries retain
their hashes. Standard tests and runtime ASan/UBSan pass. Unit tests cover metadata,
BUSY retries, legacy callback isolation, cancellation, fatal acquisition failure,
one-output capacity, and 1,024 sequential requests with 64-bit IDs through reused
queue records. Snapshot tests reject active-slot changes, false eviction counts,
bad charges/peaks, undersized leases, missing events and leaks; retained idle
inventory may exceed peak live concurrency.

Use the qualified build/device settings. With the existing controlled workload:

```sh
python3 bench/prepare_bf16_arrivals.py test/bf16_cuda_load48_572_reference.json \
  /tmp/capacity572.bin --mixed-budget 8
mkdir -p /tmp/capacity572
./bench_bf16_arrivals "$MODEL" /tmp/capacity572.bin \
  bench/policies/qwen38-cuda-decode-aware8-noncp.json /tmp/capacity572 --sized-kv \
  > /tmp/capacity572.jsonl
python3 bench/verify_bf16_arrivals.py /tmp/capacity572.jsonl \
  --reference test/bf16_cuda_load48_572_reference.json
python3 test/verify_bf16_mixed_reference.py /tmp/capacity572 \
  --reference test/bf16_cuda_load48_572_reference.json
```

The same option accepts fixed-deadline workloads; verify their actual geometry
against a matching independent control. Diagnostic timings are excluded from
performance medians. The existing fixed-capacity path remains selectable by
omitting the option. Source hashes, scopes, per-run metrics, memory checks and
long-gap localization are in [the machine-readable results](qwen38-request-capacity-admission.json).
External evidence: `/tmp/bitnet-goal-capacity-admission`.

Remaining work includes the cold-admission tail, full backend memory admission
beyond KV payload, steady-state serving and the remaining model/offload/backend
goal. No complete-goal claim follows from this integration.
