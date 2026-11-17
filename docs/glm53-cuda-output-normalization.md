# GLM resident output RMS normalization

`gated_rmsnorm_f32_weight_bf16_resident` implements the GLM recurrent KDA output
normalizer with immutable F32 gamma, BF16 input and sigmoid gate, and BF16 output.
It accepts the explicit `BN_GPU_GATED_NORM_GLM_R2` profile: 128 columns, two rows
per warp, eight contiguous columns per lane, and a caller-supplied positive
epsilon. GLM uses epsilon 1e-5 at this boundary.

The profile preserves the original local square-pair reductions, approximate
square-root followed by F32 division, ordered normalization/weight/gate
multiplications, and final BF16 rounding. Gamma retains all F32 bits. The older
BF16-weight FLA and compiled SiLU profiles retain their arithmetic and capability
mask; they do not accept the new profile. The new entry point accepts only GLM_R2
with sigmoid activation and a `[1,128]` F32 gamma tensor.

Input and gate views remain read-only. Output must be disjoint from both. The
operation uses the resident request stream, without allocation, activation
copies through host memory, or an internal fence. Kernel layout is fixed rather
than inferred from row count or device occupancy.

## Independent normalizer references

`test/reference_cuda_glm_output_norm.py` calls the unchanged GLM
`rms_norm_gated` helper. It clones the input before every call because that
original helper writes its input in place. Original captures repeat exactly
before producing the committed byte-hash fixtures.

Twenty-two cases cover row counts 1/3/15/16/17/64/130/377/1088/6144/65537,
two deterministic seeds, epsilons 1e-6/1e-5, zero/tiny/ordinary/large inputs,
saturated and nonsaturated gates, and F32 gamma with deliberately non-BF16 low
bits. Rounding that gamma to BF16 changes 22,375 original output values.

The native tests qualify full row batches and odd-sized partitions against the
same fixtures: 622 launches and 37,576,704 output words. They also check both
inputs remain unchanged, output canaries, and rejection without output mutation
for invalid dimensions, epsilon, activation/profile, weight type/shape, view
ownership/type/extent, and aliases. These checks run in the full CUDA suite and
via `test_cuda_backend --glm-output-norm-reference`.

## Composition and remaining scope

The composition harness under `/tmp/bitnet-goal-glm-cuda-output` uploads raw
projected BF16 Q/K/V, initial convolution history, F32 recurrent state, raw gate
projections and immutable weights from the preceding frozen original captures.
It queues native F32-weight convolution, native KDA recurrence, and this output
normalizer on the same resident stream. No reference intermediate is supplied
between these operations. Host reads occur after the composed execution to
compare convolution values, histories, F32 state, BF16 core outputs, and final
BF16 outputs directly against the unchanged original captures.

T17/H64 and T65/H2 each pass full chunks, split chunks, one-token updates, and
prefill followed by one-token decode. All eight schedules match exactly over
39,207,168 convolution/core/output/history/state words. Compute Sanitizer reports
zero errors for both the normalizer suite and the mixed prefill/decode chain.

The protected clean build has zero warnings. Standard and full CUDA suites pass,
including diagnostic isolation. Compiled source/fixture hashes remain unchanged
through qualification, and four protected user binaries retain their hashes.

Two Nsight diagnostic captures of the T17/H64 pipeline show exactly three native
kernels on one stream and zero recorded memory-copy bytes or allocation API
calls within the capture. There is one explicit final harness stream sync and
an additional unattributed context-sync activity after the kernels finish; no
inter-operation synchronization is recorded. The first capture includes lazy
CUDA library loading. The warm capture first executes the same operations with
unchanged immutable inputs before capture, and measures kernel durations of
5.536/37.889/1.824 microseconds for convolution/KDA/output normalization. These
single diagnostic timings include profiler effects and establish no statistical
performance or vLLM parity claim.

This qualifies projected recurrent KDA arithmetic at the tested GLM shape. It
does not include the input/output projection GEMMs, model registration, indexed
MLA, mHC, MoE execution or a complete GLM model. The reference recurrence is the
original fused recurrent operator; the optimized matrix-based chunk-prefill
algorithm has a separate precision/execution contract and remains open. CPU
rounding differences, other GPU hardware, whole-model correctness, and measured
vLLM prefill/decode parity also remain open.

Metadata, source hashes and verification results are recorded in
`glm53-cuda-output-normalization.json`.
