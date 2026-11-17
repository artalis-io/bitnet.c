# GLM resident FP8 indexer transforms and gated pool state

This records the transform/pool checkpoint. The subsequent
[checkpoint pre-indexer qualification](glm53-cuda-preindexer.md) adds F32
LayerNorm and actual projection/resource binding. Remaining-work statements
below describe this earlier checkpoint.

The SM120 CUDA backend now provides resident Hadamard/FP8 query conversion,
four-token gated key compression with explicit tail state, and F32 head-weight
scaling. These are the arithmetic and streaming-state core of GLM's sparse
indexer. They do not yet constitute a complete indexer or MLA attention layer.
Projection and F32 LayerNorm binding, paged FP8-cache addressing, scores, pool
selection/expansion, latent attention, and whole-model execution remain open.
The ordinary GLM loader remains guarded; the full modern-model goal is active.

## Arithmetic and storage

Each query row has 128 BF16 features. Seven ordered F32 Hadamard butterfly
stages precede multiplication by the F32 representation of `1/sqrt(128)` and
BF16 materialization. Quantization finds the row's absolute maximum, clamps it
to at least `1e-4`, computes `2^ceil(log2(absmax/448))`, and converts scaled
values to E4M3FN with round-to-nearest-even and finite saturation. Raw FP8 bytes
use I8 resident storage; scales use separate F32 views. This storage declaration
adds no integer or FP8 model-weight execution capability. The quant registry
declares one-byte I8 scalar storage with no loadable-model or CPU-kernel hooks.

GLM pools four consecutive keys using a separate softmax for each feature.
Scores add the slot's full F32 absolute-position bias to its BF16 gate value.
The maximum subtraction, approximate exponential, ordered denominator sum,
FMA accumulation and division preserve the pinned original operator's order.
The pooled result materializes as BF16 before the same Hadamard/FP8 conversion.
Head-weight scaling performs two ordered F32 multiplies. GLM's 32 heads of
width 128 give a factor of `1/64`; a second non-power-of-two factor exercises
rounding in the primitive reference. Inputs and intermediate arithmetic are
qualified for finite neural values; nonfinite/overflow recovery is not claimed.

`BnGPUFWHT128FP8ResidentPlan`, `BnGPUKPoolFP8ResidentPlan`, and
`BnGPUIndexerWeightScaleResidentPlan` expose these operations through the GPU
vtable. CUDA-private quantization arithmetic lives in `src/quant/`.

## Request state and submission

The caller supplies a read-only initial tail and disjoint next tail, each
BF16 `[2,4,128]`: normalized keys followed by gate scores, with slots indexed by
logical position modulo four. Unwritten slots survive. A pool completed across
a chunk boundary reads earlier slots from the initial tail and new slots from
current input. Completed pools are emitted in chronological order. The number
of emitted rows is `floor((position+tokens)/4)-floor(position/4)`; zero requires
empty output and scale views.

All active views have exact scalar types/counts, belong to one request, and
are disjoint. Visible validation precedes kernels; malformed metadata leaves
outputs and tail state unchanged. Buffers and immutable F32 APE must already
exist. Submission allocates, transfers, fences and commits no logical position.
The caller decides when to accept next state, reuse tail buffers, append cache
rows and publish outputs. CUDA execution failure poisons the request.

This profile covers sequential text with four-token pooling and no speculative
rows. It leaves persistent cache layout and serving scheduling to their owning
layers. It does not silently reuse Qwen's four-head BF16 mean-pool profile:
GLM has 32 query heads, gated pools and FP8 indexer values.

## Independent references and validation

`test/reference_cuda_glm_indexer.py` invokes the unchanged installed vLLM GLM
operators. Query conversion and pool outputs are repeated independently. The
reference additionally runs the original paged decode/tail-update operator and
checks that its cache bytes/scales equal original whole-pool compression.
This environment uses the original F32 cache-scale mode; the native core
returns scales separately and does not yet qualify a paged-cache writer.
Full source hashes and reference digests are recorded with the captures.

Sixteen query cases cover 1, 2, 31, 32, 33, 129, 1,025 and 65,537 rows, two
seeds, zeros, signed zeros, constants, impulses and amplitudes from `2^-30`
to `2^24`. Forty-two pool cases cover empty/completed pools, all starting
position residues, arbitrary populated initial tails, and resumed positions
through 1,025. Native replay uses whole submissions and chunks of 1/3/7/64,
checking complete output and final tail. Input/tail immutability and rejected
metadata canaries cover ownership, aliasing, counts, overflow and invalid
parameters. Expected values and existing fixtures are preserved.

Build/run `make BN_ENABLE_CUDA=1 test_cuda_glm_indexer` in the qualified CUDA
configuration. It is part of the serial CUDA backend test dependency chain and
uses model-free reference fixtures. Passing a captured-reference directory to
the executable additionally enables direct byte comparisons. The final replay
passes **19,069,100 output elements** over 226 schedules, by both direct bytes
and fixture digests. Separate original captures reproduce the same values.
The protected warning-free clean build, standard suite, complete CUDA suite
and diagnostics pass. CUDA memcheck and synccheck report zero errors. Earlier
complete MoE and dense-prefix replays remain exact for 527,733,360 and
221,074,320 words respectively.

A warmed trace covers one 32-head query transform/scaling and one pool
completion at position three, with projected inputs, APE and buffers prepared
before capture. It records four kernels, no transfer or allocation APIs and
one explicit final host stream fence. The original post-capture outputs match
5,313 elements exactly. CUPTI also records an unattributed final context
synchronization. A separate first-launch trace preserves the driver module
load; it is excluded from the warmed interval. This is a scoped dependency
trace, not a throughput or whole-indexer benchmark.

The machine-readable
[qualification record](glm53-cuda-indexer-core.json) supplies final counts,
source identities, sanitizer outcomes and the scoped resident trace.
