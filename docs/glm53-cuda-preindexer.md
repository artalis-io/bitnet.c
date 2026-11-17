# GLM checkpoint pre-indexer projections and F32 LayerNorm

This records the pre-indexer checkpoint. The subsequent
[paged cache qualification](glm53-cuda-fp8-cache.md) adds append/gather and
resident cache-chain composition. Remaining-work statements below refer to
this earlier checkpoint.

The CUDA backend now composes GLM's actual block-three pre-indexer: separate
Q8 query/key/gate projections, F32 LayerNorm, raw-F32 head projection, query
Hadamard/FP8 conversion, ordered head scaling, and gated key pooling with
initial/next tail state. The caller supplies hidden states and the MLA
caller's normalized q-rank input. This is the complete pre-indexer at that
boundary; scores, selection/expansion, paged-cache binding and latent attention
remain open. The ordinary whole-model loader remains guarded.

`BnBackendBF16IndexerWeights` owns the seven backend resources and borrows CPU
sources only during preparation. Its query/key/gate weights retain independent
Q8 projection widths and original vector/matrix dispatch. Head weights,
LayerNorm gamma/bias and pool APE retain all source F32 bits. Preparation checks
anatomy, tensor types/shapes, every shard's payload bounds, finite F32 values
and GPU budget before uploading. A failed upload frees partial resources and
leaves the caller's destination unchanged. No host concatenation/staging
allocation is needed. Actual raw GPU tensor payload is **8,326,144 bytes**;
request scratch, activations and CUDA allocator/context overhead are separate.
GPU resources outlive borrowing requests, and the GPU outlives their teardown.

`bn_transformer_bf16_indexer` validates all fifteen active views together before
submission. Views have exact types/counts, belong to the request, and are
pairwise disjoint. The resource's physical layer must match the plan. Caller
reservation covers BLAS and A8 scratch for `max(dim,q_rank)*tokens`. No
activation leaves the GPU during this helper; it allocates, transfers, fences
and commits no logical position. The next tail is separate from the initial
state. Execution failure can partially submit work; callers discard that
request without accepting next state.

## LayerNorm arithmetic

The original compiled GLM operator uses full F32 gamma/bias and BF16 input/output
with width 128. It has distinct one-row and batched Welford layouts. One row
first combines adjacent features; a batch first combines features 64 apart.
Both use the original descending XOR tree and opposite final merge orientations
in alternating feature groups. The initial prototype differed by one word; inspection found both layout
and final merge-orientation differences. The native kernel now preserves
both layouts, FMA boundaries, variance division, approximate reciprocal square
root and final F32 affine FMA before BF16 materialization. The source/PTX audit
is preserved with the qualification evidence.

Independent fixtures cover 36 combinations: one through 65,537 rows, two
parameter/input seeds, and epsilons `1e-6` and `1e-5`. They include zeros,
signed zeros, constants, near-constant large-offset rows and small/large
amplitudes. Full F32 parameters are distinguishable from rounded BF16 copies.
The actual GLM resource binds the checkpoint's `1e-6` indexer epsilon.

## Checkpoint reference profile

`test/reference_glm_preindexer.py` uses unchanged original CUDA GGUF projection
operators, the original compiled GLM LayerNorm and head-scale leaves, and the
original Hadamard/FP8 and gated-pool operations. The original paged decode
operator supplies tail-state updates and independently checks pooled output
bytes. References record weight/source/extension hashes and assert CUDA GGUF
operators are enabled.

This is an explicit **raw GGUF Q8/F32 profile**. The default HF indexer merges
key/head parameters into BF16 storage before caching its head weights in F32.
That cast is a different profile. Preserving GGUF head-weight bits does not
establish default-HF or whole-vLLM equivalence. LayerNorm's own F32 gamma/bias
match its declared F32 parameter contract.

The independent reference derives each supplied q-rank input using the actual
Q8 MLA q-A projection and original RMSNorm with BF16 gamma. Native replay starts
at the public hidden-state/normalized-q-rank boundary; it does not claim native
MLA q-A execution. Native warmup computes its own indexer tail before resumed
phases. No expected internal pre-indexer projection, normalization or pool value
is injected into the candidate helper.

Twenty-four schedules cover cold and native three-token warmup, vector/matrix
batches across the 6/7-token dispatch boundary, token-by-token decode,
chunked/resumed execution and prefills through 1,025 tokens. Every stage is
compared directly against the matching reference schedule. The dispatch and
LayerNorm profiles can legitimately round differently across schedules; this
qualification preserves their reference semantics rather than asserting all
chunking choices have identical arithmetic.

## Reproduction and remaining work

The optional `test_cuda_glm_preindexer` target builds the checkpoint replay;
run it with the first model shard and a reference directory. The model-free
`test_cuda_glm_indexer` target now includes the LayerNorm fixtures in the serial
CUDA suite. Its optional second directory argument enables direct LayerNorm
file comparisons alongside the existing first directory for FP8/pool captures.
The reference generators are `test/reference_glm_preindexer.py` and
`test/reference_cuda_glm_layernorm.py`. Synthetic CPU tests cover source
ownership/bounds/budgets, upload rollback, global alias/count validation and
all eight submission-failure boundaries.

The final checkpoint replay passes **22,659,076 direct output-element
comparisons** across 24 cold/warm schedules and 62 phases. LayerNorm separately
passes **34,190,848 direct values** and the earlier FP8/pool core retains its
19,069,100 comparisons. Fresh original captures reproduce all prior values.
The protected clean build is warning-free; standard/full CUDA suites and
diagnostic isolation pass. Host ASan/UBSan covers the preparation/orchestration
code with synthetic resource/state probes. CUDA memcheck and synccheck report
zero errors for both model-free core/norm and actual-checkpoint replays.
The earlier complete MoE and dense-prefix regressions remain exact for
527,733,360 and 221,074,320 words respectively.

A warmed actual-weight trace captures a seven-token phase at position three,
after native three-token tail warmup and priming of that same seven-token plan.
It records 14 kernels: three tiled Q8 projections and their quantizers, query
FWHT, LayerNorm, input cast, F32 head GEMM/reduction, head scaling, pooling and
tail publication. There are no transfer or allocation APIs and one explicit
final host stream fence. Captured outputs match 61,986 elements; the replay's
89,026 count also includes the uncaptured three-token warmup. CUPTI additionally
records an unattributed final context synchronization. The trace excludes the
caller's q-A/RMS computation and later cache/scoring/attention, and establishes
no throughput or vLLM performance claim.

See the [machine-readable qualification](glm53-cuda-preindexer.json) for final
results and source hashes. Remaining GLM work includes paged FP8 side-cache
addressing, weighted scores, pool selection and tail expansion, MLA projection
and compressed-latent attention, whole-model binding/generation, long contexts,
VRAM expert caching and CPU/backend qualification. Matched vLLM performance and
serving acceptance remain part of the active full goal.
