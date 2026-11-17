# GLM native pre-indexer, cache and causal scores

`bn_transformer_bf16_indexer_scores` composes the native pre-indexer, paged
FP8 cache append/gather and signed FP8 scorer. The caller supplies hidden states
and normalized MLA query-rank input. Query/key/gate projections, head weights,
LayerNorm, query FP8 conversion, pool/tail state, cache history and scores are
then computed by the native path. Reference activations are never injected at
an internal boundary during replay.

The composed plan checks score metadata, capacity arithmetic and aliases against
all 21 upstream views before submitting any work. This includes query-byte
alignment and availability of the score callback. Existing cache metadata
validation is shared with the cache-only helper. The score output is
F32[tokens, floor((position+tokens)/4)], with negative infinity in future pool
columns and an empty view when no complete pool exists.

All views and scratch are reserved by the caller. The chain allocates, copies,
fences and commits no position itself. Cache slots/page tables are scheduler
metadata. The sticky gather status must be initialized to zero and checked
after a caller fence before accepting state or publishing output. Execution
failure can leave cache/next-tail partially modified; discard the request.
The contiguous-history gather is a prefill interface, not the intended direct
paged decode scorer.

The optional `test_cuda_glm_indexer_scores` target takes the first checkpoint
shard, pre-indexer reference directory, cache reference directory and score
reference directory. It reads the captured schedules and supplies only x/QR
inputs. Its tail and paged history persist natively between phases. It compares
every pre-indexer output, full cache including untouched sentinel bytes,
gathered history, scales and causal scores directly against independent
original operators. The prior tail is also checked as immutable.

The custom schedule option in `test/reference_glm_preindexer.py` retains its
historical default schedules. `test/reference_glm_indexer_scores.py` scores the
original pre-indexer and persistent original cache captures. Each original
capture is repeated internally; two independent complete captures are also
compared byte-for-byte. Nine schedules cover 23 phases, zero complete pools,
2,049/4,097-token prefixes, chunks and resumed/single-token submissions after
long prefixes. Different schedules preserve their own original dispatch and
rounding semantics; this does not assert identical quantized activations across
all ways of partitioning a prompt.

The standard CPU mock suite tests global rejection against all 21 upstream
aliases, malformed views, absent callbacks, zero-key behavior and failure at
each of the eleven submission stages. Actual CUDA replay covers the arithmetic
and persistent history. Whole-model GLM admission remains guarded: selection,
pool/tail expansion, paged decode, latent MLA attention and generation remain
open. The full backend, expert placement, serving and vLLM parity goal is active.

Final qualification passed: 146,404,112 pre-indexer/cache output elements and
12,074,513 causal F32 scores match the original references exactly. All 418
files across the three reference stages are byte-identical across independent
complete captures. The clean build had zero warnings; standard and CUDA
suites, historical cache replay, actual dense/MoE regressions, host ASAN/UBSAN
and device memcheck/synccheck passed. User probe binaries were preserved.

The warmed 4,097-token combined submission contains 17 kernels, including one
cache append, one gather and one scorer. The captured chain has no CUDA memory
copies or allocation API calls and one caller stream fence. Staging, status
readback and comparisons occur outside capture. CUPTI also records a context
synchronization without a matching runtime API call; the raw records are
retained. This does not establish whole-model throughput or paged decode.
See the [machine-readable qualification record](glm53-cuda-indexer-scores.json).
