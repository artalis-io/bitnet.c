# Flash host-staging experiments

Four external experiments tested host expert-cache staging without changing
inference kernels, routing or cache policy. None supports a new default. All
81 generated streams match the preserved control and all 5,184 phase-level GPU
hit/miss/upload/fill records remain exact. Host-cache counters also match between
variants with the same RAM budget. The runtime is unchanged.

The workload remains five prompt tokens, 64 outputs, three requests per process,
a 12 GiB GPU expert cache and either 0, 2 or 8 GiB managed RAM. Caches start empty
and persist between requests. Model/GPU preparation is outside request timing;
filesystem cache state is uncontrolled. These serial repeated requests do not
establish general workload performance, physical-disk behavior or vLLM parity.

## Parallel copies and streaming stores

A standalone linker wrapper targets large copies into actual pinned-bank address
ranges during inference. It does not interpose copies inside CUDA libraries.
Two/four total threads use persistent workers and a blocking join before upload.
An AVX2 streaming-store variant uses aligned stores and a store fence. Preparation
of workers is outside request timing. One repeat reverses the serial/two-thread
order. Four-thread and streaming-store cases are exploratory single processes.

| RAM and copy variant | Cold decode mean, ms | Final warm decode mean, ms |
| --- | ---: | ---: |
| 8 GiB, serial | 50.72 / 50.99 | 17.79 / 17.76 |
| 8 GiB, two threads | 49.74 / 49.62 | 17.63 / 17.56 |
| 8 GiB, four threads | 53.59 | 17.79 |
| 8 GiB, streaming stores, one thread | 49.83 | 17.88 |
| 8 GiB, streaming stores, two threads | 51.34 | 18.19 |
| 2 GiB, serial | 42.36 / 43.15 | 23.05 / 23.05 |
| 2 GiB, two threads | 41.85 / 41.03 | 22.55 / 22.56 |
| 2 GiB, streaming stores, one thread | 41.10 | 22.44 |
| Direct reads into pinned staging, no RAM cache | 28.00 / 28.59 | 19.12 / 19.45 |

Parallelism slightly reduces measured copy time, but the fully warm end-to-end
gain is about 1%. More threads or streaming stores do not consistently help.
The undersized 2 GiB variants still trail direct reads. These results do not
justify worker-management complexity or changing portable copy behavior.

Separate byte-copy checks cover threshold boundaries, unaligned destinations,
odd lengths, matching bytes and untouched prefix/suffix guards for one, two and
four threads, with and without streaming stores. All six ASan/UBSan checks pass.
The unwrapped benchmark and the serial instrumentation control agree closely.

## Preparing RAM pages before inference

The RAM-cache slab is allocated lazily; its pages are faulted in during cache
fills. An external allocation wrapper writes one byte per page, plus the last
byte, before model/GPU preparation. Cache contents remain unpublished and empty.
This tests page readiness rather than preloading experts.

For 8 GiB, page preparation incurs about 2.10 million minor faults and takes
1.15/1.28 seconds. Timed cold decode falls from 53.33/52.12 to 38.53/38.62 ms.
However, the first request plus this explicit preparation takes **4.03/4.17
seconds**, versus **3.85/3.78 seconds** without preparation. These totals exclude
the common model/GPU preparation cost; moving page faults earlier is not a free
startup speedup. Final warm decode remains about 17.76 ms.

At 2 GiB, page preparation takes 291 ms. The first request plus preparation takes
3.18 seconds versus 3.13 seconds without it. This also fails to improve combined
startup cost. The lower timed request latency must not be presented alone.

## Transparent huge-page hint

On this Linux host, transparent huge pages use `madvise` policy. Another external
wrapper advises only whole pages inside the allocated cache slab; it performs
no upfront page writes. The hint succeeds and costs less than 0.01 ms.

The 8 GiB cold-request minor-fault count falls from about 1.76 million to 75,000,
but cold decode is unstable: **60.92/41.58 ms** versus **51.83/52.98 ms** in the
two baseline processes. At 2 GiB, the hint measures 39.42 versus 42.65 ms cold,
with much smaller warm differences. Fewer faults do not establish a robust cold
latency improvement. These traces do not isolate the cause of the slow huge-page
run, so no compaction or allocation mechanism is claimed as proven.

The negative results narrow the next work to source-lease ownership, bounded
placement and overlap rather than more copy-loop tuning. No library source,
cache default or CLI option changed. Standard tests pass; runtime source hashes
are unchanged. Commands, standalone wrappers, raw logs, per-run results and
scope are recorded in [the machine-readable evidence](qwen38-flash-staging-experiments.json).
The full inference/performance/serving/model/backend goal remains active.
