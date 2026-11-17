# GLM NoPE CUDA sparse MLA attention

The native path now connects prepared latent queries and paged FP8 KV to sparse
attention and value/output projection. It consumes the existing selector's
2,051 logical entries, validates and translates them through a request-owned
page table, and pads the physical table to 2,176 columns. It preserves selection
order. Future, invalid logical, and invalid page indices are masked before the
attention gather and set a sticky device status. The caller must check status
and discard failed results before committing position.

`BnGPUMlaFP8AttentionResidentPlan` uses exact, disjoint resident views with
explicit workspace. `bn_transformer_bf16_mla_attention` preflights twelve views
and capabilities before attention, per-head value projection and raw output
projection. The SM120 NoPE profile has 64 heads, latent width 512, original key
width 256, and attention scale 0.0625. Queries/outputs are BF16; KV uses the
[qualified 656-byte packed cache](glm53-cuda-mla-cache.md).

Attention uses pinned FlashInfer CUDA templates through a private C ABI.
Batches through 64 rows use split decode/merge; larger batches use swapAB
prefill directly. This matches the qualified original default dispatch.
Kernel attributes are configured at backend creation. There is no Python,
Torch or TVM runtime, upstream mutable configuration cache, execution-time
allocation, transfer, synchronization, or fallback. Immutable model resources
remain backend-owned; cache, indices, status, activations and scratch belong
to the request.

Decode requires 2,236,928 partial-workspace bytes per row, at most 143,163,392
bytes for 64 rows. Batched prefill requires no split partials. Query, output,
index, LSE, cache and linear workspace reservations are separate. The adapter
builds as C++17 with the original CUDA math flags; other engine code retains
its existing language settings. Revision pins, fetch instructions and licenses
are in [third_party/flashinfer](../third_party/flashinfer/README.md).

The actual-checkpoint replay connects native input preparation, cache writes,
short-path selection, page mapping, attention, and value/output projection.
Eight batches through 1,025 tokens match 112,793,443 reference words, including
40,927,232 attention BF16 outputs, and 1,805,312 cache bytes. Hidden inputs are
synthetic, using the preserved original raw-GGUF projection profile. Value
GEMMs retain the [matched cuBLAS comparison profile](glm53-cuda-mla-projection.md);
this does not qualify the default HF merged loader.

A model-free regression is included in `make test_cuda_backend`. Twelve cases
cover rows 1/2/3/64/65/129 at positions 0 and 4096, including long prescribed sparse
sets and both dispatch branches. Its 17,301,504 attention BF16 outputs and
all LSE, physical indices and cache bytes match original operators. Invalid
metadata, unsafe indices, sticky status, ownership, aliases, empty prefill
scratch and submission failures are also checked. Long prescribed sets do
not qualify native overbudget top-k ordering or ties.

Clean build, standard and complete CUDA suites, projection/indexer regressions,
host ASAN/UBSAN, and device memcheck/synccheck passed. Final test builds have
zero warnings. Original operators run twice per capture; independent complete
recaptures preserve identical bytes. Four user probe binaries are unchanged.
Warm attention plus output projection uses six kernels for decode and five
for 129-row prefill, with no allocation/copy activity and one caller stream
fence each. Additional unattributed CUPTI context synchronization is retained.

See the [qualification record](glm53-cuda-mla-attention.json). Whole GLM
admission remains guarded. Native overbudget selection order/tie effects,
complete MLA mHC/MoE blocks, full chunked/resumed model execution and generation
remain unfinished. These component checks do not establish vLLM throughput or
latency parity.
