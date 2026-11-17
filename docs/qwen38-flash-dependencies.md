# Flash expert upload dependencies

The Flash trace analyzer can now report exposed bank dependencies separately
from GPU copy duration and host source time. The existing cold/warm capture
contains 18,789 matched bank waits. Each first consumer is the expected grouped
expert kernel, with no overlapping positive intervals in this capture.

The checkpoint, 12 GiB GPU expert cache, 8 GiB managed RAM cache, five-token
prompt and three 64-output requests are the same as the bounded expert-supply
diagnostic in `docs/benchmarks.md`. Caches persist between requests. Filesystem
cache state is uncontrolled; these are not physical-disk measurements. This
analysis reuses that recorded trace rather than claiming a new runtime benchmark.

| Request | Exposed dependency per decode, ms | Positive waits / decode waits |
| --- | ---: | ---: |
| Cold expert caches | 3.748 | 2,378 / 6,048 |
| Repeated request 1 | 1.335 | 1,271 / 6,048 |
| Repeated request 2 | 1.354 | 1,287 / 6,048 |

Each decode average covers 63 steps. Prefill has 215 matched waits per request;
its lower bounds total 83.536, 32.009 and 32.146 ms respectively. Prefill bypasses
GPU expert-cache admission, so the warm cache does not eliminate its uploads.

## What the measurement means

For each compute-stream wait, the analyzer matches the exact event ID and record
generation to its producer stream. It finds the last producer GPU activity before
the event record, all preceding consumer-stream GPU work, and the first consumer's
host API return. The exposed interval is:

```
max(0, producer_activity_end - max(prior_consumer_activity_end, consumer_API_end))
```

At that point prior compute has finished and the consumer has been submitted,
but required producer work remains incomplete. The event's device timestamp is
unavailable in this capture, so preceding producer activity provides a readiness
lower bound. The report unions intervals to prevent duplicate counting when
multiple waits precede a consumer. It supports one captured process/device/context,
checks event associations and submission order, and requires grouped expert banks.

This is a timestamp-derived lower bound on exposed producer dependencies. It
excludes CPU reads and staging before consumer submission, event overhead and
other scheduling delays. It is not total source stall time, an isolated transfer
duration, or a counterfactual latency saving. Profiler overhead remains present.
Do not add this value to copy durations or source-wrapper spans: those overlap.
The remaining host-staging cost and opportunities for overlap require separate
qualification before changing RAM placement or runtime policy.

## Calibration and reproduction

A separate two-stream CUDA control exercises kernel and H2D producers, each
with an exposed dependency, a dependency hidden by prior compute, and a consumer
submitted late by the CPU. All six outputs are checked. For the two exposed
cases, the timestamp calculation gives 7.015695/8.296658 ms versus independent
CUDA event gaps of 7.016448/8.295200 ms, within the 0.1 ms calibration tolerance.
Hidden cases report zero; late submissions also report zero despite CUDA event
gaps of about 24 ms. This prevents interpreting CPU submission gaps as DMA waits.
The microsecond differences reflect timestamp resolution and instrumentation;
the controls do not prove an exact counterfactual speedup on a full model.

Build and profile [the calibration workload](../bench/qualify_cuda_dependencies.cu)
with the designated GPU and standard CUDA environment:

```sh
nvcc -O3 -arch=sm_120 bench/qualify_cuda_dependencies.cu -o /tmp/dependencies -ldl
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/dependencies /tmp/dependencies
nsys export --type sqlite --output /tmp/dependencies.sqlite /tmp/dependencies.nsys-rep
```

`extract_dependencies()` in `bench/cuda_dependencies.py` reads the export.
The NVTX ranges identify the six control cases; the executable prints independent
CUDA event gaps. The qualification evidence contains the checked per-case values.
For the recorded Flash capture:

```sh
python3 bench/analyze_flash_trace.py TRACE.sqlite --phases-per-run 64 \
  --prefill-batches 1 --expert-cache --bank-stream-id 15 --dependencies \
  --output /tmp/flash-dependencies.json
```

Omitting `--dependencies` retains the original report. All preexisting metrics
in all 192 phase windows are unchanged. The new option adds bank wait counts,
positive dependency counts and `bank_dependency_lower_bound_ms` per phase.
Explicit stream IDs belong to a capture and must be identified again for a new
capture. Source hashes, original database hash and results are in
[the machine-readable qualification](qwen38-flash-dependencies.json); raw evidence
is under `/tmp/bitnet-goal-offload-dependency`.

No runtime or cache policy changes follow from this diagnostic alone. Matched
offload performance, sustained serving, total backend-memory admission and the
remaining models/backends in the full goal still require work.
