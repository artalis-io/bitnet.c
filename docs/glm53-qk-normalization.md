# Direct F32 Q/K normalization for KDA

The SSM normalization interface now has an explicit `BN_SSM_QK_NORM_L2_SUM_EPS`
mode. It computes `x / sqrt(sum(x*x) + eps)` in F32, using a deterministic
zero-padded binary reduction and no intermediate BF16 store. The existing
clamped-L2 and RMS-plus-scaling modes retain their values and behavior. No
model family is automatically switched to this new mode.

The scalar, AVX2, NEON and WASM dispatchers route this mode through the shared
portable implementation. Its bounded scratch uses the padded head width, with
`BN_MAX_VLA_ELEMS` as the limit. Q/K buffers must be disjoint, head ranges valid,
and epsilon finite and positive. Invalid metadata is a no-op, consistent with
the internal range-function interface. The caller owns all activation storage.

## Why the distinction matters

The original GLM KDA recurrence normalizes raw convolution outputs inside its
kernel, in F32, and immediately consumes those normalized values. Storing them
to BF16 first changes this boundary. Adding epsilon to the sum also differs
from clamping the sum to epsilon, particularly for small vectors. The existing
RMS-plus-scaling operation is mathematically equivalent to direct normalization
but has a different floating-point operation sequence; its old profile remains
available unchanged.

Four cases were captured before implementation from unchanged original GPU
normalization and fused KDA recurrence functions. They include zero, tiny and
ordinary BF16-representable projection values, dimensions 1/3/128, GLM's
64-head shape, and sequences up to 65 tokens. Every original prefix state and
output is retained. Original repeated runs are exact; standalone normalization
also agrees with a float64 check.

Across scalar, AVX2 and AVX512 compilation configurations, **122,541,840** values
are checked with full and partitioned head ranges and continuation into fresh
caller-owned state. Maximum observed absolute differences are below 6e-8 for
normalized vectors, 1.8e-7 for prefix state and 1.5e-8 for output. The threshold
is 2e-6 absolute. These are numerical core checks, not full-model BF16-output
bit identity or new optimized SIMD kernels.

The standard SSM fixture checks the new mode through available backend routes,
head isolation, epsilon behavior and preservation of F32 low bits. Scalar SSM
ASan/UBSan passes. The clean build has zero warnings; the standard suite
and full CUDA regression/diagnostic suites pass. ARM/NEON and WASM routing code has not been executed on matching
hardware in this qualification.

Reproduce with the qualified reference environment and selected GPU:

```sh
python test/reference_ssm_kda_norm.py --output /tmp/kda-norm-reference
python test/verify_ssm_kda_norm.py --reference-dir /tmp/kda-norm-reference \
  --output-dir /tmp/kda-norm-verified --mode scalar --mode avx2 --mode avx512
```

Optional ISA modes require matching hardware; the default is portable scalar.
The generator writes a small header only with explicit `--fixture-header PATH`.
All captured arrays and the checked-in header regenerate exactly. Source hashes,
case shapes, errors and scope are in [the machine-readable results](glm53-qk-normalization.json).
External evidence is under `/tmp/bitnet-goal-glm-norm`.

The layer audit also confirms separate Q/K/V convolutions can be concatenated
by channel in the reference, and output processing is RMS normalization followed
by sigmoid gating with epsilon 1e-5. Existing operations cover these basic forms,
but their BF16 boundaries and complete GLM layer behavior still require independent
qualification. Model anatomy, indexed MLA, backend wiring and whole-model
short/long/chunked/resumed prefill/decode remain open. The full goal remains active.
