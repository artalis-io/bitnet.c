# GLM NoPE MLA latent KV cache

The CUDA path now writes normalized latent KV directly into request-owned
packed FP8 cache pages. `bn_transformer_bf16_mla_cache` connects the six existing
projection/normalization operations to the cache writer after checking all
nine views and backend capabilities. It submits seven operations and performs
no allocation, transfer, synchronization, or position commit. Model weights
remain immutable backend resources; callers own cache capacity and slot mapping.

The qualified profile matches the installed vLLM `fp8_ds_mla` writer:
64 rows per page, 656 bytes per row, 512 E4M3 bytes, four F32 scales for
128-feature groups, and 128 zero padding bytes for NoPE. Scales are powers of
two with a minimum derived from `max(absmax / 448, 1e-4)`. The clamp is applied
after division. Negative slots are padding; native writes also skip positive
out-of-capacity slots. Valid slots within a submission must be unique. Slot
zero is valid. Inputs and cache rows require 16-byte alignment.

The installed original writer requires the 656-byte allocation even though
NoPE attention needs only a 528-byte payload. The original 528-byte rejection
is preserved in the experiment artifacts. This implementation follows the
installed reference rather than assuming the current upstream writer's
accepted layouts. The [upstream cache source](https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu)
provides the algorithm; installed operator captures provide the qualification
values.

Two independent invocations capture every original result before native
comparison. Synthetic tests cover zero through 4,105 rows, tiny and large
finite values, signed zero, skipped slots, untouched page canaries, page
boundaries, ownership, extents, alignment and empty submissions. Actual
checkpoint projection replay covers eight batch sizes through 1,025 tokens.
Across batched, single-row and synchronized chunk writes, 32,873,472 cache
bytes match exactly. These chunk schedules split cache writes while keeping
projection GEMM geometry fixed; they do not qualify full chunked attention.
Actual projection comparison retains the matched cuBLAS profile documented
in [MLA projection qualification](glm53-cuda-mla-projection.md).

Clean compilation has zero warnings. Standard and complete CUDA suites,
projection/indexer regressions, host ASAN/UBSAN, and device memcheck/synccheck
passed. The warm 129-token projection/cache trace contains ten kernels, no
CUDA copies or allocation calls, and one caller stream fence. Additional
unattributed CUPTI context synchronization records are retained. Four user
probe binaries remain byte-identical.

See the [qualification record](glm53-cuda-mla-cache.json). Native attention,
index mapping, selection order/tie effects, complete MLA blocks and model
admission/generation remain unfinished. This is not a throughput or vLLM
parity claim.

A follow-up standalone adapter experiment using the original FlashInfer CUDA
kernel templates matches 40,927,232 BF16 attention outputs and all LSE values
for eight batches through 1,025 tokens. Decode uses the split/merge kernel;
larger batches use swapAB prefill. The original model reserves
`round_up(2048 + 4 - 1, 128) = 2176` sparse columns, with 125 padding entries
set to `-1`. A 2,051-column table is the semantic selection output, not the
physical attention launch layout; the installed reference rejects that width
at 65 tokens. See the [experimental record](glm53-cuda-mla-attention-probe.json).
The adapter is outside the runtime and uses shared vendor templates, so this
is launch/numerical integration evidence, not an independent attention kernel
or completed model support.
