# Decode cache storage reuse

The generic GPU expert cache now supports a replacement reservation that keeps
an unpinned victim's storage while removing its old key from publication. The
native expert store uses this path when the backend supports refill.

The pending slot owns and charges the retained buffers. They remain unavailable
to lookups while the caller refills them. Commit accepts only the exact borrowed
handle tuple. Abort destroys retained storage, including a partly overwritten
triplet. A layout change discards old storage before allocating replacements.
Pinned entries are skipped, stale tickets fail, and pending replacements prevent
close. The regular reservation API keeps its previous destroy-on-eviction behavior.

No routing, expert grouping, arithmetic, GPU kernel or head geometry changes.
Existing consumer-completion and upload-publication fences remain. Immutable
model anatomy/CPU weights are untouched; LRU/hash/lease transitions stay in the
cache module, with model-specific layout metadata in the backend expert store.

## Measured result

Same source fixture and GPU as [temporary expert reuse](native-temporary-expert-reuse.md):
DeepSeek V4.1 Flash1017 EXL3, explicit Diffbot row-storage to Exllama F16 Engram
adapter, fresh513 + resumed513 + five greedy tokens. This is not original Diffbot
whole-runtime equivalence. RAM budget: 13,635,158,016 bytes (1024 experts); decode
VRAM and temporary budgets: 639,148,032 bytes each (48 experts).

| Repeated decode with RAM hits and VRAM misses | Before | After |
| --- | ---: | ---: |
| Median latency | 217.73 ms | 153.97 ms |
| New projection buffers per token | 720 | 0 |
| Projection refills per token | 0 | 720 |
| Expert H2D bytes per token | 3,195,740,160 | 3,195,740,160 |

Observed latency fell **29.3%**. The first decode fills 48 slots
using 144 projection allocations, then refills the remaining 576 projections.
Subsequent misses reuse those slots. Every warm decode has 240 RAM hits, zero
VRAM hits and zero source-reader bytes; required GPU transfer bytes are unchanged.
Matched first-two-pass cache-hit/miss and I/O counts also agree with the prior
implementation, including cached hits during resumed prefill.

Before uses five warm samples; after uses ten. Input-history staging through
CPU-logit publication is timed; setup and verification are excluded. OS page-cache
warmth and clocks are uncontrolled. These are fixed-sequence replay observations,
not disk-cold behavior, general serving hit rates or vLLM parity.

## Resource and correctness evidence

The maximum decode charge stays at 639,148,032 bytes. Retained pending storage
is charged to its existing slot, not to an extra pool. The temporary pool remains
inside its own quota. A separate source-exact Nsight run confirms zero device
allocation/free calls during warm decode; tracked device allocations return to
zero at teardown. The small-VRAM trace peak is 19,402,373,481 bytes,
excluding untracked driver/context overhead. Profiler timings are not used above.

All 18 gates passed: source-exact baseline and miss-heavy workloads, both head
modes, forced BUSY/retry, ASan/UBSan, CUDA memcheck/full leaks, initcheck and fresh
Qwen compatibility (192/192 token IDs). Main gates checked 245 forwards and 490
exact GPU/CPU results; tracing checked another 14 forwards. Cache tests exercise
1000 replacements with a pinned anchor, pending ownership, discard, poison abort,
stale tickets and physical memory bounds. Native-store tests check exact refill
bytes, partial failure, unpublication and retry. Required clean/product/standard/
CUDA/diagnostic checks passed with zero warnings; four user reference binaries
were preserved.

Transfers and publication remain blocking. Warm offloading still performs 720
weight copies and publication waits per token. Pinned/batched staging, production
model/tokenizer admission, continuous batching/KV scheduling, longer candidate
contexts and matched vLLM prefill/decode measurements remain open. The full goal
is active.

[Detailed results and immutable artifact hashes](native-decode-cache-reuse.json)
