# GLM bounded KDA coefficients

The SSM module now constructs the bounded coordinate-wise log decay and scalar
beta used by GLM KDA. This connects projected gate inputs to the previously
qualified [KDA state step](glm53-kda-state-step.md). It does not register GLM as
a supported model or introduce a CUDA execution path.

The separate [portable mHC core](mhc-core.md) is now qualified as well. Its
projection, CUDA residency and complete layer integration remain open.

## Parameters and operation

`bn_transformer_ssm_kda_coefficients()` takes a `BnSSMKDACoeffCtx` and head range:

```
amplitude = exp(parameter)       // BN_SSM_KDA_A_LOG
          = -parameter          // BN_SSM_KDA_A_NEGATIVE_EXP
log_decay[k] = lower_bound / (1 + exp(-amplitude * (gate[k] + bias[k])))
beta = 1 / (1 + exp(-beta_raw))
```

Inputs and outputs are F32. Gate, bias and log decay have layout `[head][key]`;
the parameter and beta arrays have one value per head. Key dimensions range
from 1 to 256; the lower bound must be finite and negative. Each call writes
only its requested head range and allocates nothing. Caller/model/session
ownership is unchanged. Exact in-place gate/log-decay and raw-beta/beta aliases
are supported; other overlap is forbidden.

The complete selected range is validated before writing. Invalid metadata,
nonfinite inputs, overflowing gate-plus-bias, overflowing `exp(A_log)` and
positive stored negative-exponential parameters fail without changing outputs.
Underflow to zero amplitude and sigmoid saturation are valid. Invalid inputs
outside a selected head range are not read. Valid empty ranges permit null
buffers. No epsilon, clamp or routing change is added to the checkpoint formula.

## Actual checkpoint storage

All 34 KDA layers in the local GLM5.3 Q4_K_XL GGUF set were compared with the
local Hugging Face checkpoint's unquantized decay/bias tensors. GGUF `ssm_a`
follows **`-exp(A_log)`**, within 1.91e-6 of a NumPy F32 exponential; it is not
raw `A_log`. Every timestep-bias value matches exactly. This comparison covers
68 parameter arrays: 2,176 decay values and 278,528 bias values.

The explicit storage tag lets a future GGUF loader preserve its existing F32
values. Taking their logarithm and exponentiating again is unnecessary and can
change rounding. Model-family code still needs to select the correct tag when
the GLM layer path is integrated; no architecture alias is introduced here.

## Independent numerical checks

Five synthetic cases use the unchanged original GLM Triton safe-gate operation
and original GPU F32 sigmoid. They include saturation, tiny amplitudes, zero and
signed inputs, odd dimensions, GLM's 64-head/128-coordinate shape, dimensions up
to 256, sequences up to 65 tokens, and lower bounds -1, -5 and -12. Repeated
original runs are exact and also agree with a float64 sigmoid calculation.

A separate reference call fuses original gate construction, beta activation
and KDA recurrence for 17 tokens at the GLM head shape, with nonzero F32 initial
state. Every prefix state and output is retained. This checks the C coefficient
operation together with the C recurrence rather than comparing coefficients
alone.

The verifier passes **217,326,408** value comparisons across scalar, AVX2 and
AVX512 compilation configurations, both parameter representations, full and
partitioned head ranges, separate and in-place outputs, and resumed caller-owned
state. These are compilations of portable code, not new optimized SIMD kernels.
Maximum observed differences from the original are:

| Boundary | Maximum absolute difference |
| --- | ---: |
| Log decay | 1.91e-6 |
| Beta | 1.20e-7 |
| Coupled prefix state | 1.79e-7 |
| Coupled output | 2.24e-8 |

Thresholds are 5e-6 for log decay and coupled state/output, and 2e-7 for beta.
This is numerical core agreement; it does not establish full-model BF16 output
bit identity. Q/K normalization, projections and final output rounding remain
separate operations.

The standard SSM suite includes a small immutable captured fixture, both
representations, alias and head-isolation checks, saturation/underflow checks,
and rejection of bad later-head inputs before any output mutation. Scalar SSM
ASan/UBSan passes. The final clean build has zero warnings and the standard
suite, full CUDA suite and CUDA diagnostic isolation pass. ARM/NEON hardware is
not qualified by these x86 runs.

Reproduce with the qualified vLLM environment and explicitly selected GPU:

```sh
python test/reference_ssm_kda_coefficients.py --output /tmp/kda-coeff-reference
python test/verify_ssm_kda_coefficients.py --reference-dir /tmp/kda-coeff-reference \
  --output-dir /tmp/kda-coeff-verified --mode scalar --mode avx2 --mode avx512
```

Omit the optional ISA modes for the portable default. The capture tool writes a
small header only with explicit `--fixture-header PATH`; regeneration reproduces
all captured arrays and the checked-in fixture exactly. Existing model goldens
are unchanged. Source hashes, cases, per-build errors, storage comparisons and
scope are in [the machine-readable qualification](glm53-kda-coefficients.json).
External evidence is under `/tmp/bitnet-goal-glm-gate`.

Remaining GLM work includes mHC mixing, indexed MLA, convolution/output processing,
model/session/backend wiring and independent whole-model short/long/chunked/resumed
prefill and decode checks. The full serving, offloading, performance and other
model/backend goal remains active.
