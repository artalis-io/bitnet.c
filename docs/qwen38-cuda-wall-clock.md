# Fixed wall-clock arrival qualification

The runtime now preserves external arrival timestamps through
`bn_generate_batch_enqueue_at`. The fixed offered-load case executes correctly
and improves throughput and mean decode latency, but its p99 inter-token
latency is substantially worse than the current vLLM comparison. Overall
serving parity is not qualified.

This page records the uncapped mixed-step baseline. The subsequent optional
cap8 result reduces p99 to 73.12 ms, versus 69.56 ms for the matched decode-aware
vLLM policy; it still narrowly misses a 5% tail target. See
[the capped result](qwen38-cuda-decode-aware.md).

The subsequent [one-token prefill specialization](qwen38-cuda-single-token-prefill.md)
now passes the bounded twelve-request case: p99 is 69.41 versus 69.56 ms, with
all independent references preserved. Sustained serving qualification remains open.

## Workload and correctness

Twelve requests arrive in groups of four at 100, 1100, and 4100 milliseconds.
Prompt lengths are 129, 5, and 129 tokens respectively; every request produces
64 outputs. Limits remain eight live sessions, four prefills, 256 scheduled
tokens, 64-token chunks, and 8192-token context capacity. Native KV payload
retention is 4 GiB, including idle slots; other device memory is excluded.

The worker records the offered deadline separately from the time it can
actually enqueue the request. TTFT begins at the offered deadline, so polling,
a busy GPU step, and slot queueing all count. The public API accepts only finite
arrival timestamps no later than the current backend clock, in that clock's
domain. FIFO order remains enqueue order. Legacy `enqueue` records the current
backend time. Unit tests cover preserved timestamps, delayed admission,
cancellation and failure events, invalid/future timestamps, and rejection
atomicity under ASan/UBSan.

BGA1 retains its previous byte-for-byte wire format. BGA2 contains fixed arrival
offsets and prompts, without continuation expectations. Its benchmark output
has `correct: null` until an independent comparison is supplied. The verifier
checks deadlines, worker and admission times, logical progress, physical roles,
fair token shares, output publication, and retained KV charges. Synthetic tests
reject altered submissions, early enqueueing, wrong roles, invalid tokens,
incorrect charges, and unsupported self-certification.

The native diagnostic's actual enqueue steps are 0, 11, and 60. An independent
original-runtime control injects the same prompts at those steps and derives
its own scheduling, greedy outputs, norm contracts, and recurrent dispatches.
All 132 matrices, 784 full heads, and 194,682,880 BF16 words match. The captured
arithmetic plan independently equals the admitted native plan. The three native
timing runs have exactly the same execution geometry and token streams as this
diagnostic. Their last group is enqueued at step 61; it waits for the same slots
and begins execution at the same point.

The compute reference is
[wall-clock control](../test/bf16_cuda_wall_clock132_reference.json).
[Tested control source](../test/reference_cuda_bf16_wall_clock_control.py) and
[tested original timing source](../test/reference_cuda_bf16_wall_clock.py) run
in the pinned reference image with the same model/plugin/configuration mounts
as the existing CUDA references. The control uses engine-step injections to
make full-head observation independent of wall-clock scheduling. Its timings
are not used for the offered-load comparison.

## Measured result and open tail issue

Both engines use the same fixed deadlines, GGUF checkpoint, BF16 activations,
F32 recurrent state, disabled CUDA graphs, and context/concurrency limits on
CUDA SM120 with 188 SMs. vLLM uses its existing round-robin prefill policy and
a 256-token budget. Native timing includes polling, enqueue/admission,
execution, sampling/publication, and pool returns; final teardown is excluded.
The original timing process performs a warmup before its three samples. Both
engines include CPU geometry recording and omit full-head observation during
timing.

Three-run medians are:

| Metric | Native | Reference |
| --- | ---: | ---: |
| Output throughput | 101.677 tokens/s | 83.065 tokens/s |
| Mean pure-decode step | 47.247 ms | 50.010 ms |
| Four-request decode step | 36.651 ms | 36.523 ms |
| Eight-request decode step | 58.679 ms | 63.714 ms |
| p95 time to first token | 1009.450 ms | 2843.450 ms |
| p99 inter-token latency | 285.957 ms | 71.918 ms |

Percentiles use nearest rank over twelve TTFT and 756 ITL observations per
run, followed by the median across three runs. Scheduling differs: native has
65 four-row and 60 eight-row decode steps; reference has 63 and 62. No aggregate
pure-prefill throughput claim is made for these differing schedules.

The free-running schedulers generate different greedy streams. Native's actual
schedule is independently bit-exact in the original runtime; cross-schedule
bit equality is not claimed. The earlier controlled case remains the comparison
with identical matrices and input geometry.

Native's worst intervals occur when four active decodes share a step with 252
prefill tokens. Two such steps affect enough token intervals to raise p99 near
286 milliseconds. Throughput and mean latency gains do not resolve this tail
regression. Next work is a smaller mixed token budget while retaining the full
prefill budget when no decode is ready. The pinned vLLM source exposes
`max_num_scheduled_tokens`, the `decode-aware` policy, and
`VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=1`; these provide an explicit reference
configuration for that experiment without changing model math.

All samples and measurement scopes are in
[wall-clock evidence](qwen38-cuda-wall-clock.json). This is one serialized
worker with deterministic offered load; network transport, larger sustained
loads, total-memory admission, and the remaining model/backend/offload goal
remain open.

## Reproduction

```sh
python3 bench/prepare_bf16_arrivals.py \
  test/bf16_cuda_staggered12_reference.json /tmp/clock.bin \
  --arrivals-ms 100 100 100 100 1100 1100 1100 1100 4100 4100 4100 4100
./bench_bf16_arrivals "$MODEL" /tmp/clock.bin \
  bench/policies/qwen38-cuda-staggered12-noncp.json --timing > /tmp/clock.jsonl
python3 bench/verify_bf16_wall_clock.py /tmp/clock.jsonl \
  --workload /tmp/clock.bin --reference test/bf16_cuda_wall_clock132_reference.json
```

Use the qualified CUDA/MKL build flags and designated device from the runtime
qualification. Actual schedules can change with timing. A new geometry needs a
matching independent control; the verifier rejects mismatched rows or outputs.
