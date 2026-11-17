# GLM dense decoder blocks on CUDA

The resident CUDA path now executes the checkpoint's token embedding and its
first three complete dense KDA decoder blocks. Each block includes attention
mHC pre, KDA projections/convolution/recurrence/output projection, fused
attention-post → FFN-pre, and the dense clamped SwiGLU FFN. The FFN output,
current residual, and F32 mixing coefficients stay deferred for the following
block's fused boundary.

This qualifies the SM120, BF16-activation, raw-GGUF-quantized recurrent profile.
It does **not** enable complete GLM generation. Block three is indexed MLA with
MoE; indexed MLA, routed/shared experts, whole-model resource binding and
execution, long-context qualification, and matched performance remain open.
The ordinary loader's incomplete-topology guard remains in place.

## Clamped activation arithmetic

The original CUDA `SiluAndMulWithClamp.forward_cuda` calls
`torch.ops._C.silu_and_mul_with_clamp`. For alpha one and beta zero, its BF16
contract is:

1. Read gate/up as F32. Clamp gate only above the limit; clamp up on both sides.
2. Compute SiLU in F32, then round **SiLU alone** to BF16.
3. Multiply the expanded rounded SiLU by F32 `(clamped_up + positive_zero)`.
4. Round the product to BF16.

Clamped intermediates themselves are not rounded to BF16. This distinction
matters for limits such as 1.234567 and 20.1. The positive-zero addition also
matters for output sign when the limit is zero. A fused F32 SiLU/product is a
different arithmetic contract, even though its formula looks equivalent.

`BnGPUClampedSwiGLUResidentPlan` exposes this explicit profile. Its CUDA hook is
currently advertised only on qualified SM120. It validates dimensions, finite
nonnegative limit, owner, exact BF16 extents, and non-overlapping input/output.
Execution queues one kernel, without allocation, transfers, or synchronization.

The model-free CUDA suite generates every finite positive and negative BF16 gate
code, pairs it with deterministic up values, and tests four clamp bounds. It
compares 262,144 output words through original-output byte hashes, checks input
preservation and offset canaries, and rejects invalid plans without modifying
the output. The generator calls the original operation twice; it does not use
native outputs as reference values.

## Resource ownership and submission

`BnBackendBF16DenseFFNWeights` owns merged Q8_0 gate/up and raw Q8_0 down buffers.
The binder consumes anatomy-owned tensor roles and per-layer ordinary FFN clamp,
validates shard payloads and shapes, and checks both budgets before uploading.
Gate precedes up in the merged projection. MoE layers and other weight formats
are rejected by this dense binder. Failure releases partial uploads and leaves
the destination unchanged. CPU model weights remain immutable.

One dense FFN has 160,432,128 GPU payload bytes and requires 106,954,752 bytes of
host concatenation storage during preparation. The three complete blocks plus
the raw Q8_0 embedding table retain 1,604,782,984 GPU weight payload bytes.
These figures exclude request buffers, mapped checkpoint pages and file cache,
and allocator/context overhead; they are not whole-model working-set budgets.

`bn_transformer_bf16_dense_ffn` queues two existing A8 linear operations and the
clamped activation. `bn_transformer_bf16_dense_kda_block` composes the complete
block and checks visible view metadata and aliasing across both sublayers before
submission. All scratch and state/history generations are request-owned and
preallocated. Reserve A8 scratch for `tokens * max(dim, heads*128, hidden_dim)`.
Ordinary and fused mHC use their respective setup-only split estimators; the
ordinary entry requires more split scratch than the small-batch fused entry.

Submission performs no state or position commit. On successful completion the
caller can exchange state/history generations. Backend failure can follow
partial submission; discard that request rather than treating the block as a
rollback transaction. Public metadata validation cannot establish the validity
of opaque backend allocations or weight handles.

## Independent checkpoint replays

All weights come from the actual six-shard
`GLM-5.3-Flash-UD-Q4_K_XL` checkpoint. The first three dense gate/up/down matrices
are Q8_0. Original GGUF matrix dispatch retains its width-dependent small-batch
thresholds; no weight decoding or BF16 weight substitution is used.

| Replay | Cases | Checked words | Coverage |
|---|---:|---:|---|
| Three dense FFNs | 27 | 22,609,920 | Gate/up, activated, down output; chunk/decode thresholds and batches 32, 33, 65 |
| Embedding → three complete blocks | 12 | 221,074,320 | Every selected boundary, F32 recurrent states, BF16 histories, deferred FFN state |
| Existing deferred attention regression | 12 | 58,644,448 | Previous incoming FFN state → KDA → following FFN pre |

The three-block replay uses actual token IDs, including vocabulary endpoints.
Each cold case starts from zero state; each resumed case computes its own
three-token warmup natively, then processes the next nine tokens. Both use
schedules `[9]`, `[2,7]`, `[3,3,3]`, `[7,1,1]`, `[6,3]`, and nine single-token
steps. Reference state is never injected after warmup. Original captures repeat
every case twice, and a separate full three-block capture produces identical
reference manifests. Actual replay comparisons are direct byte comparisons,
including F32 state and coefficients, rather than output hashes or tolerances.

These are recurrent KDA/eager embedding-expansion component comparisons, not
whole-vLLM generation comparisons. Compiled first-layer broadcast fusion and
optimized chunk-KDA prefill may have distinct arithmetic profiles. Nine-token
compositions do not establish long-context correctness or throughput parity.

## Reproduction

Build with `BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1` and the installed MKL VML root.
Run GPU commands serially on the selected inference device. Use the pinned
GLM-enabled vLLM/GGUF environment and a supported CUDA compiler for reference
capture. The optional targets build without running a model automatically:

```sh
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=/tmp/bitnet-goal-bf16-vml/mkl test_cuda_glm_dense_ffn test_cuda_glm_dense_prefix
python test/reference_glm_dense_ffn.py --model FIRST_SHARD.gguf --out /tmp/dense-ref
python test/reference_glm_dense_prefix.py --model FIRST_SHARD.gguf --out /tmp/prefix-ref
./test_cuda_glm_dense_ffn FIRST_SHARD.gguf /tmp/dense-ref
./test_cuda_glm_dense_prefix FIRST_SHARD.gguf /tmp/prefix-ref
./test_cuda_backend --clamped-swiglu-reference
```

See the accompanying machine-readable qualification record for validation and
trace results. The full modern-model/runtime goal remains active.

## Validation and residency trace

The protected clean build has zero warnings. The complete standard suite,
complete CUDA suite and diagnostic isolation pass. The prior deferred-attention
replay remains exact. Host ASan/UBSan checks cover the modified preparation and
orchestration code; CUDA memcheck covers activation, dense FFNs and the prefix,
and synccheck covers activation and the prefix, all with zero reported errors.
Four pre-existing user probe binaries are preserved byte-for-byte across clean.
No earlier reference fixtures or tolerances are changed.

A warm nine-token trace covers embedding, two resident expansion kernels and
all three blocks. It records 78 kernels on one stream: 18 A8 quantizations,
18 Q8 matrix kernels, nine splits, three convolutions, three KDA recurrent
kernels, three gated norms, three clamped activations, and the mHC/embedding/
expansion kernels. No allocation API is recorded. The sole transfer is the
four-byte embedding error-status read at the explicit final completion fence;
there are no activation or weight transfers. One explicit stream fence follows
all submissions. CUPTI also records an unattributed context synchronization
after the last kernel, with no inter-operation synchronization. Setup, split
queries, warmup, output comparisons and state/history commits are outside the
captured interval. This single observational trace establishes residency, not a
statistical prefill/decode performance result or vLLM parity.

The [routed-execution follow-up](glm53-cuda-routed-execution.md) qualifies bounded
Q4_K/Q5_K expert banks with supplied routes. GLM routing, shared experts and
complete MoE/model composition are not implied by that execution qualification.
