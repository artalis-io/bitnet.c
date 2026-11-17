# GLM paged FP8 indexer cache and resident prefill chain

The SM120 CUDA path now appends completed indexer pools into a paged cache and
gathers visible history without a host activation bridge. The production
`bn_transformer_bf16_indexer_cache` helper connects the qualified checkpoint
pre-indexer to these operations, with one global metadata/alias preflight.
Weighted scoring, selection/expansion and MLA attention remain open; the
ordinary whole-model GLM loader remains guarded.

## Layout, state and failure contract

Each physical page holds 64 indexer states. Its first 8,192 bytes contain
64 × 128 E4M3FN bytes; its final 256 bytes contain 64 F32 scale bit patterns.
The total is 8,448 bytes per page. Scales are planar, rather than interleaved
with each state's FP8 bytes. This qualifies the original F32-scale profile,
not packed MX-scale interpretation. Raw cache/value/output byte bases must be
four-byte aligned; all views have exact types/counts and request ownership.

`BnGPUIndexerFP8AppendResidentPlan` copies raw values/scales into I32 physical
slots. Negative and out-of-range slots are padding and leave cache bytes
unchanged; valid slots must be unique within one submission. Physical slot
and block zero are valid. A zero-row append is a no-op with empty inputs.
One warp copies each 128-byte value row using 32-bit words and copies the
scale bits without arithmetic, including NaN payloads and signed zero.

`BnGPUIndexerFP8GatherResidentPlan` maps logical rows through the page table
and emits contiguous FP8 bytes and F32 scales. Repeated readable page mappings
are allowed. Invalid pages zero their rows and set a sticky device error flag;
valid gathers do not clear an earlier error. The caller initializes status to
zero and fences/checks it before accepting outputs or state. This safe invalid
page behavior is an explicit native contract, outside the original reference's
valid-page comparison. A zero-row gather leaves outputs and status untouched.

The composed plan owns no buffers. The caller reserves the cache, page table,
slot descriptors, status and all pre-indexer scratch. Cache history before
`floor(position/4)` must already be present; new slots must match the new
logical pool range through the page table. The helper computes its own queries,
normalized keys, gates, head weights, compressed keys and next tail, appends
those new pools, then gathers the visible history. It performs no allocation,
host transfer, synchronization or logical position commit. A failure can
partially modify cache/next-tail buffers; callers discard that request.

This composition qualifies the contiguous-history prefill path. A future
paged decode scorer should consume the cache directly; the gather is not a
claim of an optimal full decode path. Cache buffers belong to request/backend
session state, while immutable model resources retain their earlier ownership.

## Original references and integration

`test/reference_cuda_glm_fp8_cache.py` calls the unchanged original GLM gated
pool writer and original CUDA indexer-cache gather. The gather's original
scale destination is a `[rows,4]` byte tensor, subsequently viewed as F32;
native replay preserves exactly those bits in its F32 output view.

Eighteen model-free cases cover 0/1/2/63/64/65/129/1,025/65,537 newly compressed
rows, two seeds, shuffled physical slots, reversed page mappings, invalid
append slots, untouched bytes and partial final pages. Native pooling produces
its own values/scales before the append. Full cache bytes, gathered values and
scales are checked against independent captures. Additional canaries cover
alignment, aliases, foreign owners, counts/overflow, invalid gather pages,
sticky status, empty operations and raw scale-bit transport.

`test/reference_glm_indexer_cache.py` composes the original writer/gather with
the preserved original pre-indexer captures and actual pool APE. Native replay
starts at hidden-state/normalized-q-rank inputs and computes every internal
pre-indexer and cache result itself, including warm tail/cache history. It
compares matching cold, chunked and resumed schedules, preserving their earlier
projection and LayerNorm dispatch semantics.

The optional `test_cuda_glm_indexer_cache` target builds the actual-checkpoint
replay. Its arguments are the first model shard, pre-indexer reference directory
and cache-chain reference directory. The model-free `test_cuda_glm_fp8_cache`
target is in the serial CUDA backend test chain; an optional captured-reference
directory enables direct byte comparisons in addition to fixture digests.

## Qualification scope

The [machine-readable record](glm53-cuda-fp8-cache.json) records final test
counts, source hashes, sanitizer results and the warmed execution trace.
This checkpoint does not qualify the remaining signed-weight FP8 score math,
causal masking per query row, pool top-k/expansion, MLA latent attention,
whole-model generation, GLM VRAM expert caching, other backend/model scope,
matched vLLM performance or serving tails. Those remain part of the full goal.

The installed original SM120 DeepGEMM scorer was also probed on preserved actual
pre-indexer data: valid scores reproduce across calls and include negative
values. That gives the next stage an available primary reference; it does not
qualify a native scorer. In particular, GLM cannot inherit Qwen's unweighted,
nonnegative-score contract.

Final qualification passed: 52,953,486 exact cache elements and 24,393,382
exact actual-checkpoint pre-indexer/cache elements. Independent reference
captures were byte-identical. The clean build had zero warnings; standard and
CUDA suites, dense/MoE regressions, host ASAN/UBSAN probes, and device memcheck
and synccheck passed. User probe binaries were preserved byte-for-byte.

The warmed 129-token submission at position 3 contains 16 GPU kernels,
including one append and one gather, no CUDA memory-copy activity or allocation
API calls, and one caller stream fence. Staging and status readback occur
outside the capture. CUPTI additionally records a context synchronization
without a matching runtime API call; the raw records are retained. This trace
qualifies this operation chain, not full-model throughput.
