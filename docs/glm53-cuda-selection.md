# GLM pool selection and token expansion

The SM120 resident selector implements signed F32 top512 pool selection and
four-token expansion. It writes 512 diagnostic pool indices and 2,051 token
indices per query, with -1 padding and no count column. The native order is
chronological; equal cutoff scores prefer the lower pool index. Signed zeros
tie, while negative scores and subnormals retain their numeric order.

Two packing modes match the original GLM dispatch. Sparse mode expands pool
history into columns0..2047 and appends up to three incomplete-tail tokens in
columns2048..2050. Short-prefill mode requires the chunk's final position to
be at most 2,048 tokens, consumes no score view and emits contiguous causal
indices. The composed helper skips score submission in short mode while
updating the pre-indexer and cache. Query/head preparation is still performed;
this is not the final short-prefill performance optimization.

`bn_transformer_bf16_indexer_select` checks both output views against every
upstream view before submission. Normal mode composes twelve operation stages;
short mode uses eleven and does not require the score callback. Allocation,
staging, scratch reservation, caller fencing and status checks remain caller
responsibilities. The sticky cache status also reports nonfinite visible
scores: affected rows are filled with -1. Future columns are ignored. Failure
may partially modify cache/tail/output; discard the request before publishing
state. This is a gathered-history prefill path, not direct paged decode.

The native deterministic policy is explicit. The pinned original selector can
return different arrival orders. No raw reference indices were reordered or
replaced. For captured untied cutoff scores, tests compare pool and token
membership exactly, and separately verify native ordering, four-token groups,
tail locations and padding. Short-mode index lists compare byte-for-byte.
The captured long-prefix corpus has 4,105 over-budget query rows and zero
cutoff ties. Synthetic ties test the documented native policy against an
independent CPU full sort; they do not prove identical tied membership to the
original atomic selector. Downstream MLA/order/tie numerical and whole-model
generation qualification remain open, and GLM admission remains guarded.

`test/reference_glm_selection.py` preserves two raw original invocations for
every phase, including the original short causal fill and original sparse
pool expansion. The optional actual-checkpoint replay starts from x/normalized
QR and retains native pool/tail/cache history between calls. Original
activations are not injected into intermediate native boundaries. Existing
score-chain references are preserved unchanged.

`test_cuda_glm_select` is part of the serial model-free CUDA suite. It covers
signed/random/negative-only scores, ties, signed zeros, subnormals, extreme
finite values, empty/small pools, all tail positions, long histories, ignored
future NaNs, invalid visible scores, sticky status and rejected metadata.
The optional `test_cuda_glm_indexer_select` target takes a model shard and
pre-indexer, cache, score and selection reference directories. Host mock tests
cover all output aliases and every operation failure in both modes.

Final qualification passed. Model-free independent full-sort tests checked
2,768,040 I32 values under the documented native policy. Actual native replay
checked 42,074,208 selection values across 16,416 queries: 2,060 short queries
have 4,225,060 token indices byte-identical to the original; 14,356 sparse
queries have exact original pool/token membership with ordering and packing
checked separately. Upstream 146,404,112 pre-indexer/cache elements and
11,548,687 scored F32 elements also match directly. The historical score-chain
replay passed unchanged.

The clean production build and final test builds have zero warnings. Standard
and complete CUDA suites, host ASAN/UBSAN, and unit/actual-chain device memcheck
and synccheck passed. An initial test-helper compiler range warning was fixed
with explicit bounds and the qualification was rerun; its original log is
preserved. Production source hashes did not change during that correction.
User probe binaries were preserved byte-for-byte.

Warm sparse prefill at 4,097 tokens has 18 kernels; warm short prefill at 1,025
has 17 and no scorer kernel. Both captures contain no CUDA memory-copy activity
or allocation API calls and one caller stream fence. Staging and readback are
outside capture; raw additional CUPTI context synchronization records are
retained. See the [qualification record](glm53-cuda-selection.json). These
captures do not establish whole-model speed or vLLM parity.
