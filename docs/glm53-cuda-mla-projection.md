# GLM NoPE MLA projection preparation

The native path prepares normalized query rank, normalized latent KV, projected
queries and absorbed latent queries from x. A second helper expands a supplied
latent attention context through the per-head value matrix and output
projection. Head projections use a single strided batched BF16 GEMM across
heads with F32 accumulation and token-major input/output. Latent attention,
cache updates and selection integration are not part of these helpers.

The immutable backend resource owns raw Q8 QA/QB/KVA/O plus BF16 norm and
per-head K/V layouts. K/V layouts use explicit identity-projection
materialization, matching the original generic quantized-weight getter applied
to the split GGUF tensors in their storage axes. The quant registry declares
this separate preparation kind. It preserves the original Q8_1 basis scale
rounded to FP16(1/127); it is not canonical direct dequantization. Existing
direct decode behavior is unchanged.

This distinction is measured, not assumed. For the actual K/V tensors, the
original direct dequantizer and original identity getter differ in 543,286
and 537,649 BF16 elements. The direct dequantizer matches a float16 intermediate
rounding path; the identity getter also differs from one-round canonical
BF16 conversion. The original identity outputs and the diagnostic probe scripts/results are preserved.
The native materialized weights are compared to the original identity getter
through complete per-head basis projections.

The preparation helper uses an explicit vector-eight RMSNorm mode for QR1536
and KV512 at eps1e-5: contiguous feature groups, increasing-distance warp
reductions, sequential warp aggregation and correctly rounded mean division.
Existing normalization modes retain their behavior. The layout follows the
[original RMSNorm source](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu)
and is qualified against the installed original operator, rather than relying
on the current source alone. The 1,025-token case exposed the difference from
the prior default reduction and from approximate mean division.

Exact BF16 batched-GEMM comparison also requires matching vendor libraries.
The preserved Torch reference uses cuBLAS130101. The system CUDA13.5 library
produces six differing BF16 values in the investigated three-token value
expansion; matching the reference cuBLAS removes those differences. Goldens
were not changed. Projection and trace qualification therefore uses
`LD_LIBRARY_PATH=/tmp/bitnet-goal-vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib`.
Builds and the complete general CUDA suite use the normal system toolchain
and libraries. This is an explicit comparison profile, not a claim of bitwise
portability across BLAS versions or equivalence to the default HF merged
loader. The paired library hashes are in the qualification record.

Resource planning accounts for 140,513,280 retained payload bytes and a
149,426,176-byte peak including serial raw conversion temporaries. Request
BLAS workspace and activation scratch are separate caller reservations. CPU
checkpoint sources stay immutable; failed uploads/conversions free partial
resources and leave the caller's output struct unchanged. GPU resources outlive
borrowing sessions. Preparation and output helpers check all metadata/aliases
before six and two operation submissions respectively, and allocate, transfer,
fence and commit no position themselves.

`test/reference_glm_mla_projection.py` uses original GGUF CUDA projections,
original RMSNorm, original identity weight getters and original torch.bmm.
Each capture is repeated; a second independent complete capture preserves the
same bytes. Eight token counts cover singleton and batched shapes through
1,025 tokens. Output projection begins at a supplied context, because latent
attention is not implemented here. The optional
`test_cuda_glm_mla_projection` target takes the first model shard and reference
directory. Host tests cover source ownership, exact budgets, all twelve
preparation failures and all projection-submission failures.

Whole GLM admission remains guarded. Next are normalized latent cache,
sparse attention with the original scale1/sqrt256, integration of native
selection and projection, order/tie numerical qualification and full-model
generation. The wider model/backend, expert placement, serving and vLLM parity
requirements remain open.

Final qualification passed: 16,777,216 prepared-weight BF16 values and
97,202,176 projection/input BF16 values match the preserved original captures
exactly under the matched BLAS profile. All 83 reference files are identical
across independent complete captures. Clean build and final test builds have
zero warnings. Standard and complete CUDA suites, historical native indexer
replay, host ASAN/UBSAN and device memcheck/synccheck passed. User probe
binaries were preserved byte-for-byte.

Warm 129-token preparation uses nine GPU kernels; output projection uses
three. Each capture has no CUDA memory-copy activity or allocation API calls
and one caller stream fence. All staging/readback is outside capture, and
additional unattributed CUPTI context synchronization records are retained.
See the [qualification record](glm53-cuda-mla-projection.json). These are
separate projection stages, not complete MLA attention or a throughput claim.
