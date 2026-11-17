# GLM5.3 KDA state update

The SSM module now provides a portable KDA recurrence with per-key-coordinate
decay and caller-owned F32 state. This is a numerical building block for GLM5.3,
not a claim that its loader, CUDA graph or full model is supported. The existing
Qwen GDN path and architecture selection remain unchanged.

[Bounded coefficient construction](glm53-kda-coefficients.md) is now qualified
as a separate portable operation, including the stored GGUF decay representation.
Complete GLM model and CUDA integration remain open.

## Operation and ownership

`bn_transformer_ssm_kda_step()` takes `BnSSMKDACtx` and a head range. Inputs are
already projected and normalized Q/K, projected V, activated scalar beta per
head, and log decay per key coordinate. For each head it computes:

```
D[v,k] = state[v,k] * exp(log_decay[k])
delta[v] = beta * (V[v] - sum_k D[v,k] * K[k])
state[v,k] = D[v,k] + delta[v] * K[k]
out[v] = sum_k state[v,k] * (Q[k] * q_scale)
```

State layout is `[head][value][key]`; key and value dimensions may differ and
range from 1 to 256. Each call updates only its head range. Input buffers are
read-only and may not overlap state/output. Metadata and missing-buffer failures
return before mutation; a valid empty range permits null buffers. The function
allocates nothing and introduces no global state. A future model session owns
the recurrent state, and backend-specific storage remains a separate concern.

The existing GDN operation decays a whole head by one scalar. Aliasing GLM to
that operation would apply different state evolution. This API keeps the two
operations explicit rather than changing existing Qwen arithmetic.

## Independent references and checks

Five cases are captured from the unchanged installed original GLM KDA Triton
implementation. Inputs are BF16-representable values lifted to F32; state and
outputs are F32. Gates include zero, -5 and intermediate coordinate-wise values;
beta includes zero and one. Cases include zero/nonzero initial state, odd key
dimension 3, dimension 1, 64 heads of dimension 128, and a 65-token sequence.
Every prefix state and output is retained. Two original runs match exactly and
the results also agree with a float64 recurrence within 4.8e-7 for state and
4.7e-8 for output.

The C implementation passes **120,699,708** value comparisons across scalar,
AVX2 and AVX512 compilation configurations, full-head and partitioned-head
execution, and continuation into freshly copied caller-owned state. These are
three compilations of the portable implementation, not three newly optimized
SIMD kernels. Maximum observed differences from the original are below 3e-7
for state and 8e-8 for output. The qualification threshold is 2e-6 absolute.
This establishes numerical core agreement, not original BF16-output bit parity.

The standard SSM test contains a small independently captured fixture, an exact
rectangular-state rank-one case, head isolation checks, invalid-metadata checks
and empty-range handling. Scalar SSM tests pass ASan/UBSan. The clean build has
zero warnings and the standard suite, full CUDA suite and CUDA diagnostic
isolation pass. ARM/NEON hardware has not been
executed for this new operation.

Generate new independent captures with the qualified vLLM Python environment and
an explicitly selected GPU:

```sh
python test/reference_ssm_kda.py --output /tmp/kda-reference
python test/verify_ssm_kda.py --reference-dir /tmp/kda-reference \
  --output-dir /tmp/kda-verified --mode scalar --mode avx2 --mode avx512
```

The verifier's default is the portable scalar compilation. Request the optional
x86 ISA modes only on hardware that supports them. It verifies immutable capture
hashes, every prefix output/state, full/partitioned head ranges, continuation and
read-only inputs, then writes libraries and results to the requested output folder.

The optional `--fixture-header PATH` writes the small case as a C header only
when explicitly requested. Regeneration reproduces every captured array and the
checked-in header exactly. Existing numerical references are unchanged. The
primary source hashes, case shapes, comparison errors and scopes are recorded
in [the machine-readable qualification](glm53-kda-state-step.json); external
evidence is under `/tmp/bitnet-goal-glm-kda`.

## Model audit and remaining integration

The local Q4_K_XL checkpoint has six GGUF shards and 1,412 unique tensors. Its
first shard contains metadata and no tensors; remaining shard tensor counts are
397, 337, 320, 341 and 17. GGUF architecture is `glm5next`, with 46 blocks
including one next-token prediction block; the text config has 45 main layers.
This model is locally available and requires its own architecture semantics.

The tensor inventory confirms separate Q/K/V convolutions, low-rank `ssm_f_*`
and `ssm_g_*` projections, per-coordinate timestep bias and per-head decay
parameters. Other required pieces include bounded gate construction, activation
and output normalization, mHC mixing with Sinkhorn normalization, indexed MLA
with pooled keys, MoE routing/clamping and model/session/backend wiring. Existing
hyper-connection routines implement a different mixing scheme and cannot be
assumed equivalent. Tensor inventories are under
`/tmp/bitnet-goal-model-target-audit`.

No architecture alias, loader registration, CLI support or performance claim
is introduced by this operation. Independent full-model short/long/chunked/resumed
prefill and decode remain required. The wider inference, serving, offloading,
DeepSeek and backend goal remains active.
