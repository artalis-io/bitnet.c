# Resident CUDA KDA recurrence

`kda_recurrent_bf16_resident` executes a contiguous request chunk with 128-wide,
equal query/key/value heads. It consumes packed convolved BF16 Q/K/V, raw BF16
gate and beta projections, immutable F32 decay parameters and bias, and an F32
initial recurrent state. It produces BF16 core outputs and a separate F32 next
state. Request views own all mutable storage; weights remain backend-owned.

Q/K normalization uses F32 sum-epsilon normalization with epsilon 1e-6. The
normalized vectors remain F32 through recurrence. Bounded per-key decay uses
the caller's negative lower bound; beta is sigmoided in F32. The core output
rounds to BF16 before the subsequent output RMS normalization/sigmoid gate.

Both raw `A_log` and stored `-exp(A_log)` parameters are explicit API modes.
Stored values are used directly, preserving GGUF parameter bits. No log/exp
round trip reconstructs an approximation of the original parameter. The caller
must supply finite weights/bias and nonpositive stored negative-exponential
values; runtime metadata validation does not scan immutable device tensors.

Each warp holds eight value rows in registers and loops over the supplied
tokens. Four contiguous keys belong to each lane. Q/K squared sums and recurrent
dot products intentionally use different arithmetic: the latter contract one
multiply into each local pair addition before the warp reduction. The existing
independently qualified GDN dot helper supplies this operation. Explicitly
rounding every product changed nine BF16 outputs in the GLM-sized prototype;
preserving contraction removed every captured state and output mismatch.

Execution allocates no scratch storage, copies no activations/state through host
memory, and introduces no internal fence. The caller commits the separate next
state after successful completion. Output and next-state views must be disjoint
from inputs and each other. Validation rejection queues no kernel. Execution
failure invalidates the resident request through the existing backend contract.

## Independent evidence

The fixture generator `test/reference_cuda_kda.py` calls the unchanged installed
GLM fused recurrent KDA operator. Every original capture is repeated exactly
before emitting expected BF16 byte hashes and every prefix's F32 word hash.
Sixteen cases cover T/H = 1/1, 5/2, 17/64 and 65/2, two seeds, and both parameter
storage modes. Token scales exercise zero, tiny, ordinary and large Q/K values;
initial state and immutable parameters deliberately retain F32 low bits.

For stored negative-exponential parameters, eager F32 addition/multiplication
first prepares scaled gate logits. The original operator then uses those F32
logits with unit amplitude and zero bias. This is an explicit reference input
transformation retaining the same rounded add/multiply boundaries; it does not
modify the original operator or use native runtime output as a reference.
Raw-log fixtures use the original fused gate directly.

The native suite checks full chunks, split chunks with committed next state,
and one-token updates against the same fixtures. It covers 396 launches and
95,931,904 F32 state/BF16 output words, including every recurrent prefix in the
one-token schedule. Invalid shapes, ranges, scales, storage modes, weights,
ownership, view types/extents and aliases reject without output/state mutation.
Initial state remains unchanged and writable output/state canaries survive.

An additional composition experiment consumes the preceding frozen GLM-chain
captures under `/tmp/bitnet-goal-glm-chain`. Actual native convolution output
feeds native KDA on the same resident stream; no reference intermediate is
substituted. T17/H64 and T65/H2 each pass full, split, and one-token schedules,
covering 26,701,824 convolution/core/history/state words with direct bytewise
comparisons. Every checked F32 recurrent state and BF16 core output matches the
unchanged original capture exactly.

Qualification metadata and source hashes are in `glm53-cuda-kda.json`.

The protected clean build has zero warnings. Standard and complete CUDA suites
pass, including CUDA diagnostic isolation. Compute Sanitizer reports zero errors
for the new operation. The compiled sources/fixtures remained unchanged through
qualification; four protected user-provided binaries retain their hashes.

## Remaining scope

This operation is a native resident recurrent execution path. It does not yet
implement a matrix-based parallel KDA prefill algorithm, establish vLLM latency
or throughput parity, register GLM model anatomy, or wire a whole model. Exact
comparisons were executed on the designated SM120 GPU. Other CUDA devices,
NEON, and Metal are not qualified by these results.

The complete chain still needs independently qualified output RMS normalization
with F32 gamma and sigmoid gating. CPU BF16 boundary differences also remain
open. Indexed MLA, resident mHC, model wiring, whole-model correctness,
offloading and serving/performance acceptance remain part of the full goal.
