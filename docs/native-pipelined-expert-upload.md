# Pipelined expert staging

Pinned expert uploads now accept a borrowed CPU projection source. CUDA copies
one projection into its own pinned offset and immediately enqueues H2D, allowing
the next CPU fill to overlap the preceding transfer. Already staged sources
skip the CPU copy. One final completion still precedes publication.

All descriptors are checked before allocation or copying. Staging ranges must
be disjoint; borrowed source spans must not overlap any part of the staging
allocation or overflow their address range. Failed transfers retain the existing
poison/abort and cleanup contract. Host leases survive through completion.
The same 13,315,584-byte pinned bundle overlaps the total staging budget;
there is no new pinning, device allocation, consumer event or mutable model state.
The option remains disabled by default (`--pinned-staging` is available in the
standalone native benchmark).

In the matched RAM-warm/48-slot VRAM-miss replay, median decode was
**154.366 ms with ordinary staging and 155.704 ms with pipelined staging**
(0.87% higher latency; ten samples each).
The previous serial-fill pinned replay was 159.536 ms; that historical
comparison is uncontrolled. Every matched current frame preserved cache,
source-read, allocation/refill and H2D counters. Warm decode still moves
3,195,740,160 expert bytes per token with 720 projection refills and 240 triplet
completions.

Warm bypass prefill measured 78.97
tokens/s with ordinary staging and 97.08
tokens/s with pinned staging (four 513-token chunks per configuration).
Page-cache/disk warmth, CPU placement and clocks were uncontrolled. These are
fixed replay samples, not serving throughput or vLLM parity. Main elapsed time
includes the CPU copies; `upload_time_ms` now also includes them inside the
backend callback and is not comparable with the parent's callback-only counter.

A separate source-exact Nsight run showed 2400/2400 host intervals between
successive triplet enqueues overlapping preceding triplet DMA; 0 intervals
were fully covered. Each such interval contains the next CPU fill and loop
overhead, so it is a proxy rather than a direct CPU memcpy measurement. Warm
stream synchronization count remains 413, transfer bytes and kernel launch
counts match the parent, and warm weight malloc/free calls remain zero.
Pinned host allocations/frees and the tracked device peak are unchanged.
Profiler timings are excluded from the performance samples.

All 12 targeted model gates passed, plus the protected clean product build,
standard tests, full CUDA suite and diagnostics with zero warnings. Tests cover
alias/overlap rejection before mutation, borrowed/mixed source creation and
refill against unchanged frozen FFN outputs, whole-model all-logit checks,
BUSY retries, ASAN/UBSAN on C paths, CUDA memcheck/initcheck, and fresh strict
Qwen parity. CUDA translation units and the legacy MoE CPU kernel object use
normal builds; they are not claimed as sanitizer-instrumented.
Main model runs verified 182 forwards with 364
exact published CPU/GPU checks; the separate trace verified another 14 forwards.
Four user reference binaries and independent source/oracle values were preserved.
The source fixture uses the existing explicit Diffbot lookup to Exllama Engram
math adapter, not original Diffbot full-runtime equivalence.

The full goal remains active: production admission/text support, broader model
and backend qualification, controlled offload behavior, serving/KV scheduling
and matched resident vLLM parity remain open.

Evidence: [native-pipelined-expert-upload.json](native-pipelined-expert-upload.json).
Prior implementation: [native-pinned-expert-upload.md](native-pinned-expert-upload.md).
