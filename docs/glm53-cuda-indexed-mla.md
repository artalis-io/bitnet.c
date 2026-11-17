# Resident indexed MLA and selection-order diagnostics

`bn_transformer_bf16_indexed_mla` now connects hidden input through MLA
preparation, latent KV writes, indexer preparation/cache/score/selection,
sparse attention and value/output projection. Normalized QR is computed once
and borrowed directly by the indexer. Query, latent cache, logical selection
and sticky status are linked exact views. Latent-token pages and pooled-indexer
pages retain separate tables and capacities.

The helper validates every subplan, the six shared-view links and all
cross-stage aliases before submitting work. It queues 20 operations on the
short path or 21 on the scored path, with no allocation, host bridge,
synchronization or position commit. Callers reserve all storage, preserve
history, fence and check the shared status before accepting position/tails.
Submission failure requires discarding partial results. Existing component
APIs retain their behavior through shared validation helpers.

The native composition matches 742,909,705 diagnostic-reference elements over
three schedules: `[1]`, `[1025,1024,1,3]` and `[4097,1,3]`. This includes native
QR, indexer intermediates, persistent caches, causal scores, selection,
attention and output projection. No original QR or other intermediate is
injected into the native operation. Decode and resumed prefill reuse both
caches and the compression tail. The replay disables the indexer's gather and
contiguous-score callbacks, ensuring the direct paged-score path is used.

The reference distinction is essential: original operators run on an explicit
chronological selected-set diagnostic for exact composition checks. Raw
original selections and outputs are archived separately and never reordered
or replaced. The default decode selector resolves to vLLM's persistent backend;
prefill uses its original per-row operator. Independent recapture preserves
all 98 deterministic diagnostic files byte-for-byte.

Selection order changes FP8 tile arithmetic even with identical selected
membership. Actual checkpoint weights and a 4,097-token input demonstrate:

| Output-projection comparison | Observed relative L2 difference |
| --- | --- |
| Original prefill repeated against its first run | 0.1984–0.1990% |
| Chronological order against that prefill run | 0.2076% |
| Original default decode repeated against its first run | 0–0.4818% |
| Chronological order against each decode run | 0.4573–0.4850% |

The prefill probe has no cutoff ties. The decode probe also has an untied
cutoff and unchanged membership, but produces 15 distinct index orders across
16 original runs. These observations establish original numerical variation;
they do not establish an acceptable error tolerance or token equivalence.
Native chronological ordering and lower-index tie handling are unchanged.
Full-model logits, tokens and tie effects remain unqualified.

Clean build and final test builds have zero warnings. Standard and complete
CUDA suites, component regressions, host ASAN/UBSAN, and device
memcheck/synccheck passed. Host mocks cover the shared links, cross-stage
aliases and every 20/21-operation submission failure. Four user probe binaries
remain byte-identical. Warm complete indexed MLA uses 32 kernels for
4,097-row prefill and 30 for decode, with no CUDA allocation or copy activity
and one caller stream fence each. Additional unattributed CUPTI context
synchronization records are retained.

See the [qualification record](glm53-cuda-indexed-mla.json). The comparison
retains the [raw-GGUF and matched-BLAS profile](glm53-cuda-mla-projection.md).
Whole GLM admission stays guarded. Complete mHC/MoE blocks, full-model
generation and order/tie qualification are next; throughput and latency parity
with vLLM remain unproven.
