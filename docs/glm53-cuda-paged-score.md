# GLM direct paged FP8 score and decode indexer

The SM120 paged scorer reads compressed FP8 keys and F32 scales directly from
the planar 64-row cache through the request's page table. It shares the FP8 MMA,
signed head weighting, per-key scaling and reduction order with the qualified
contiguous scorer. It does not allocate or fill a gathered history buffer.
Scores retain the per-query causal complete-pool mask, with future columns set
to negative infinity. A visible invalid physical page writes negative infinity
for that page and sets sticky status; callers must fence/check status before
publishing output or accepting state.

`bn_transformer_bf16_indexer_decode` composes native pre-indexer, newly completed
pool append, direct paged score and token selection. Short decode uses the
original contiguous causal-token packing and bypasses score submission. The
normal path has eleven operation stages and the short path ten. All view
metadata and aliases are checked before any submission. Scratch, cache, slots,
page table and outputs are reserved by the caller; the helper makes no
allocation, transfer, fence or position commit. Errors may leave partial
cache/tail/output writes, so discard the request. Query/head preparation is
still performed in short mode; this is not its final performance optimization.

The direct path removes one gather kernel and `132*n_keys` bytes of temporary
history per call. It still computes the visible score matrix and reads the
paged history. It is a single sequential request/chunk interface, with no
multi-request mapping or speculative tree semantics. The actual replay also
uses it for long warmup prefixes, then verifies resumed single-token calls.
This does not establish whole-model decode speed or vLLM parity.

`test/reference_cuda_glm_paged_score.py` captures the original paged operator
twice and compares it bit-for-bit with the original contiguous operator after
masking original padding. Eight cases cover empty pools, singleton/odd/even
query chunks, partial pages, reversed physical mappings and a position above
262K tokens. A second complete capture and fixture header are compared against
the first. Native tests use the original fixture digests and optional direct
byte comparisons, plus malformed-view/ownership/alignment guards, negative and
out-of-range page IDs, fail-closed outputs and sticky status.

The optional `test_cuda_glm_indexer_decode` target takes a model shard and the
pre-indexer, cache, score and selection reference directories. It starts from
x/normalized QR and carries its own native tail and cache between phases. It
allocates no gathered history and disables the gather and contiguous-score
callbacks. Upstream outputs and scores compare directly to preserved original
captures. Short token lists compare byte-for-byte; sparse selection compares
exact original pool/token membership with native ordering, grouping, tail and
padding checked separately. The documented deterministic selection order/tie
policy still requires downstream MLA/generation qualification.

The standard host mock suite covers alias rejection for all new views, missing
callbacks and every operation failure in normal and short modes, with gather
and contiguous scorer callbacks absent. Full GLM admission remains guarded;
MLA projection/cache/attention/output and generation, other backend/model
scope, expert placement, serving and matched vLLM performance remain open.

Final qualification passed. Native paged fixtures match 1,134,288 original
F32 values, and both independent original captures and fixture headers are
byte-identical. Ten additional original actual-checkpoint decode phases cover
10,255 scores and reproduce the preserved nonpaged values exactly. Native
composed replay matches 145,015,814 pre-indexer/cache elements and 11,548,687
causal scores across nine schedules and 23 phases. Selection checks retain the
previous qualification's exact short packing and untied sparse membership
scope; 42,074,208 I32 selection values are checked.

Clean build, standard/CUDA suites, historical prefill/score regressions, host
ASAN/UBSAN and unit/actual-chain memcheck/synccheck passed with zero compiler
warnings. User probe binaries were preserved byte-for-byte. The warmed
single-token submission at position4,097 contains 14 kernels, including a
direct paged scorer and selector, with no gather kernel, CUDA memory-copy
activity or allocation API calls. This phase completes no new pool, so no
append kernel is expected. One caller stream fence occurs inside capture;
staging/readback occur outside. Additional unattributed CUPTI context
synchronization records are retained. See the
[qualification record](glm53-cuda-paged-score.json).
