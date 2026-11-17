# Idle KV growth without rebuilding sessions

Request-sized CUDA KV admission now grows an idle session's KV allocations while
retaining its recurrent state buffers, activations and workspaces. The qualified
48-request controlled replay retains 512 MiB instead of fixed mode's 4 GiB, with
essentially unchanged throughput and p99 latency. The earlier cold-admission
p99.9 regression is absent in these three paired runs. Sized mode remains opt-in;
the budget covers KV payload, not total device memory.

## Ownership and failure behavior

The optional `resident_resize_discard` backend callback operates on a complete
owned allocation during serialized setup. It fences the request, frees the old
storage before allocating its replacement, and leaves contents unspecified.
Success changes the view generation, invalidating previous full views and slices.
Same-size calls are idempotent; validation failures leave storage unchanged.
Failure after releasing storage poisons the request, which must be destroyed.
CUDA bounds each request to 65,535 allocations and each allocation to 65,535
resizes, rejecting exhaustion without wrapping generations.

The pool restarts an idle session before replacing its 32 attention KV buffers.
It never resizes a leased session or migrates an active prefix. Actual retained
capacity remains charged, including idle slots. Other idle slots can be reclaimed
to satisfy the final KV budget; leased slots cannot. A partially failed growth
discards the session. Backends without the callback retain the previous session
recreation path. Fixed-capacity acquisition still reuses adequate idle slots.

## Correctness and allocation checks

The real-model test grows a 64-token session to 256 tokens without ordinary
resident allocations, replacing exactly 32 KV buffers and retaining session
identity. All 26 checked prefix output heads match a separate 512-token session
exactly. An active-prefix growth attempt is rejected without mutation. Injecting
failure after the first replacement verifies disposal and budget cleanup.

The unchanged twelve-request independent controls match all 784 heads in both
fixed and sized modes. The unchanged 48-request control matches 4,132 full heads,
or 1,026,058,240 BF16 values. Its inventory verifier confirms eight creations,
40 reuses, 12 idle resizes, zero evictions, and 512 MiB peak/retained KV payload.
The previous implementation created 20 sessions and evicted 12 for this workload.

The clean CUDA/MKL build has zero warnings; standard tests, the full CUDA suite
and diagnostic isolation pass. Focused resize tests cover F32/BF16/I32 storage,
grow/shrink cycles, invalid targets, stale full views and slices, pinned uploads,
and isolation of other allocations. CUDA memcheck and synccheck report zero
errors. Pool unit tests pass ASan/UBSan; snapshot tests validate resize counters
and reject unexplained capacity changes. Existing numerical references are unchanged.

## Controlled timing and profiler evidence

Three pairs run the same 572-step, 48-request workload with eight live slots,
chunk size 64, mixed budget 8, maximum context 8192 and a 4 GiB KV budget. One
pair reverses mode order. Admission, sampling and pool returns are timed; model
preparation is excluded. Full-head diagnostics are separate from timing runs.

| Three-run median | Fixed KV | Sized KV with idle growth |
| --- | ---: | ---: |
| Output tokens/s | 69.10 | 69.28 |
| p99 inter-token latency, ms | 70.50 | 70.56 |
| p99.9 inter-token latency, ms | 95.81 | 92.01 |
| Peak/retained KV payload | 4 GiB | 512 MiB |

Sized p99.9 ranges from 91.96 to 92.48 ms; no sized run has an interval above
100 ms. Fixed p99.9 ranges from 95.69 to 100.06 ms, with four intervals above
100 ms in one run. Each run has 3,024 intervals; tails use nearest rank followed
by a three-run median. This small finite sample does not establish sustained
serving tails. The preceding implementation's separate three-run median sized
p99.9 was 113.61 ms; its paired fixed control was 95.90 ms.

Nsight traces localize the two growth boundaries at calls 66 and 214. At each,
the old implementation executes 1,044 `cudaMalloc` and 1,068 `cudaFree` calls;
idle growth executes 128 of each. The measured host gaps fall from 55.52/47.16 ms
to 12.05/10.29 ms. These windows extend from the prior logits completion fence
to the current embedding launch and include CPU sampling and admission. They
contain profiler overhead and are not isolated transfer stalls or measurements
of latency savings. Allocation cost remains visible.

Use the reproduction commands in
[request-aware admission](qwen38-request-capacity-admission.md), retaining
`--sized-kv`. The fixed option remains available by omitting it. Per-run timing,
source hashes, inventories, scope and profile records are in
[the machine-readable results](qwen38-idle-kv-resize.json). External evidence is
under `/tmp/bitnet-goal-idle-kv`.

This qualification adds no new vLLM comparison. Total backend-memory admission,
sustained serving, wider contexts, expert supply critical-path stalls and the
remaining model/backend requirements remain open. The full goal is active.
