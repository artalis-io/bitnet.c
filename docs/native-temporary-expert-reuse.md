# Reusable native expert buffers

The CUDA native expert store now reuses idle temporary projection storage instead
of allocating and freeing it for every upload. Routing, expert grouping, arithmetic,
head geometry, upload completion and consumer-release fences are unchanged.
Backends without the optional refill callback keep the existing behavior.

The store owns a lazy triplet per temporary reservation slot. A live lease prevents
reuse. Shape/context/format/marker/byte-count validation precedes refill; a failed
transfer discards the affected triplet before any lease is published. Store close
refuses live leases and frees all retained storage.

`temporary_allocated_payload` includes idle and leased reusable storage, within
the existing temporary quota. `temporary.charged_payload` counts active reservations
and overlaps that storage; **do not add these values**. Retention raises the idle
footprint, so a released key no longer implies freed device storage. Backends
without refill support report zero reusable allocation and release immediately.

## Measured effect

Same GPU, checkpoint, source fixture and benchmark as
[native forward profiling](native-forward-profiling.md): DeepSeek V4.1 Flash1017
EXL3, explicit Diffbot row-storage to Exllama F16 Engram execution adapter, fresh513
+ resumed513 + five greedy tokens, f16acc disabled. This is not original Diffbot
whole-runtime equivalence or a new inference CLI admission.

| Configuration | Before, prefill tokens/s | After, prefill tokens/s | After, later decode median |
| --- | ---: | ---: | ---: |
| Window 1, bypass | 45.35 | 56.19 | 223.59 ms |
| Window 32, bypass | 58.02 | 78.88 | 223.50 ms |
| Window 32, decode retention | 62.49 | 83.52 | 21.58 ms |

Window-32 bypass prefill throughput improved **36.0%**. Both versions upload
exactly 102,090,582,528 expert bytes for the first chunk and 92,529,993,216 bytes
for the resumed chunk. Kernel-category counts and explicit request fences are
unchanged. Before/after results are three-pass observations; OS page-cache warmth
and clocks were not controlled. Later decode uses ten samples from passes two
and three of an identical fixed sequence, not a serving hit-rate estimate.

The first window-32 prefill creates **96** projection buffers and refills them
22,905 times. Nsight confirms **96 `cudaMalloc`, zero `cudaFree`** in its forward
interval, versus 23,001 of each before. The pool retains 426,098,688 payload bytes
within the existing 639,148,032-byte temporary budget. The traced device allocation
peak remains 28,110,765,417 bytes and returns to zero at teardown; untracked driver
and context overhead is excluded.

A separate 48-slot VRAM / 1024-slot RAM case also reproduced every source output.
Warm decode had 240 RAM hits and zero VRAM hits per token, read zero new source
bytes, and still uploaded 3,195,740,160 bytes per token. Its five-sample median
was 217.73 ms. RAM residency does not remove PCIe traffic;
decode-cache eviction still allocates/frees buffers in this implementation.

## Validation and remaining work

All 13 gates passed, including both head modes, late partial-group BUSY/retry,
ASan/UBSan, CUDA memcheck with full leak checks, initcheck and fresh strict Qwen
comparison (192/192 token IDs). Main gates checked 168 full forwards and 336 exact
GPU/CPU published results; supplementary tracing and RAM tests checked another
28 forwards. Synthetic tests cover pinned-slot isolation, stale leases, poisoned
refill rollback, capacity bounds and teardown. CUDA tests retain the original
independent EXL3 FFN expected values.

The required clean product build, standard tests, full CUDA suite and diagnostic
isolation passed with zero warnings. The four user reference binaries were
preserved. No commits were made.

This removes allocation churn, not upload publication fences. Transfer staging,
decode-cache replacement, larger candidate contexts, production model/tokenizer
admission, continuous batching/KV scheduling and matched vLLM prefill/decode
comparisons remain open. The full multi-model/backend goal remains active.

[Detailed metrics, source changes and artifact hashes](native-temporary-expert-reuse.json)
