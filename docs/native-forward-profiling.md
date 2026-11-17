# Native CUDA forward profiling

`bench_native_forward` exercises the internal native forward API against a trusted,
immutable source fixture. It does not admit this checkpoint through the inference CLI.

Build with the same CUDA and MKL flags as the product, then run:

```sh
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=/tmp/bitnet-goal-bf16-vml/mkl bench_native_forward
./bench_native_forward MODEL_DIR REFERENCE_DIR VOCAB_JSON 32 3 0
./bench_native_forward MODEL_DIR REFERENCE_DIR VOCAB_JSON 32 3 0 --admit-decode --decode-cache-experts 1024 --ram-cache-experts 1024
```

Select the intended GPU with `CUDA_VISIBLE_DEVICES` and `BN_CUDA_DEVICE=0`.
The fixture needs `reference.json`, `host-reference.json`, token IDs and source
logits from `test/reference_native_forward_resume.py` and its independent host
conversion. `VOCAB_JSON` is the original immutable vocabulary JSON consumed by
the native vocabulary/map adapter, not a text BPE encoder. Frames must describe
one contiguous sequence. Numeric positional arguments are fused window, sequence
repetitions, and all-logits mode (0 = last token only; 1 = every token).

Every forward checks GPU F16 and published CPU F32 logits exactly. The timed
region includes input-history staging through CPU-logit publication. Model and
request preparation, reference loading and verification are outside the timer.
`--observe` adds callback counters without extra GPU operations or fences; use
separate runs for timing. Counters are zero/uncollected without that flag.

The phase option uses lookup-only prefill and admission for the native decode
profile (at most eight rows). Defaults are 48 expert triples each for RAM, decode
VRAM and temporary VRAM. RAM/decode cache capacities are independently configurable;
temporary capacity remains 48 triples. Quotas cover expert payloads, with separate
staging, core weights, activations, row caches, CUDA overhead and OS page cache.

## Qualified measurements (2026-11-13)

GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q, UUID
`GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05`.
DeepSeek V4.1 Flash1017 EXL3 checkpoint, explicit Diffbot row-storage to Exllama
F16 Engram execution adapter, f16acc disabled. This is not original Diffbot
whole-runtime equivalence. Workload: fresh513 + resumed513 + five greedy tokens.

| Configuration | Prefill tokens/s | Later-pass decode median |
| --- | ---: | ---: |
| Window 1, cache bypass | 45.35 | 295.31 ms |
| Window 32, cache bypass | 58.02 | 296.88 ms |
| Window 32, decode retention | 62.49 | 21.59 ms |

Prefill uses the median combined time for both 513-token chunks across three
sequences. Decode uses ten samples from passes two and three. The retained
configuration starts with empty RAM/VRAM expert caches; first-pass decode times
are 769.5, 549.8, 345.7, 354.4, 247.3 ms.
Later passes replay identical IDs with retained experts. These results establish
neither disk-cold performance, general request hit rates, serving tails nor vLLM
parity. OS page-cache warmth and clocks were not controlled.

Window 32 improved prefill throughput by 27.9%. For the first chunk,
fused calls fell from 7,578 to 297 and explicit request synchronization calls from
7,723 to 442; both configurations uploaded exactly 102,090,582,528 expert bytes.
The resumed chunk uploaded 92,529,993,216 bytes in either bypass configuration.

Decode retention has 13,635,158,016-byte RAM and decode-VRAM budgets each. The
replayed sequence retains 702 expert triples (9,347,539,968 VRAM payload bytes).
Every warm decode step hits all 240 requested experts and uploads zero expert
weight bytes. Input/metadata transfers still occur. This is a bounded working-set
result, not full-checkpoint GPU residency. Prefill borrows existing hits without
promotion or admission, preserving the retained decode working set.

## What the trace proves

A separate Nsight Systems capture also passed all 14 source comparisons.
Each forward's CUDA allocation count exactly equals its expert projection upload
count. The first prefill therefore makes 23,001 `cudaMalloc` and 23,001 `cudaFree`
calls despite zero request-level activation-allocation callbacks. Warm decode
makes neither call, but still has 173 CUDA stream synchronizations per token.
Boundary timings include waiting for GPU work; they are not all removable CPU
overhead. Nsight timings are excluded from the table.

The trace tracked a maximum of 28,110,765,417 device allocation bytes, with zero
remaining at teardown; untracked CUDA/context overhead is excluded. Per-forward
trace intervals start at embedding dispatch and stop before external logit
verification. Whole-process totals also include model/session preparation.

All 11 gates passed: timing/counter configurations, both head modes, ASan/UBSan,
CUDA memcheck with full leak checks, initcheck and fresh strict Qwen comparison
(192/192 token IDs). Required clean/product/standard/CUDA/diagnostic checks passed
with zero warnings; the four user reference binaries were preserved.

The next execution change should target reusable temporary expert buffers and
publication/completion boundaries, preserving source grouping and routing. Large
candidate contexts, native tokenizer/CLI admission, production serving policies,
matched vLLM performance and the full multi-model/backend goal remain open.

[Detailed results and immutable artifact hashes](native-forward-profiling.json)
