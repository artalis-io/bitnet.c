# Portable mHC coefficients and mixing

The transformer math module now provides mHC coefficient generation, input
collapse and post-layer residual mixing. These are building blocks for the
remaining GLM5.3 and DeepSeek work. They are separate from the existing
norm/down/up/injection hyper-connection implementation and introduce no model
alias or automatic path selection.

## Operations and ownership

`bn_transformer_mhc_coefficients()` accepts already projected, globally
RMS-normalized scores, three scales and a base vector. For H streams, vectors
have layout `[pre(H), post(H), comb(H*H)]`:

```
pre = sigmoid(pre_scores * scale[0] + pre_base) + pre_eps
post = sigmoid(post_scores * scale[1] + post_base) * post_mult
comb = row_softmax(comb_scores * scale[2] + comb_base) + sinkhorn_eps
comb = column_normalize(comb, denominator_epsilon=sinkhorn_eps)
repeat iterations-1 times:
    comb = row_normalize(comb, denominator_epsilon=sinkhorn_eps)
    comb = column_normalize(comb, denominator_epsilon=sinkhorn_eps)
```

Normalization keeps the reference's epsilon placement and fixed iteration count;
it does not replace these operations with an exact doubly stochastic projection.
The context supports 1–16 streams and 1–128 iterations. GLM uses four streams,
20 iterations, epsilons of 1e-6 and post multiplier 2. Projection and global RMS
normalization remain the caller/backend's responsibility.

`bn_transformer_mhc_mix_input()` collapses streams using the pre coefficients.
`bn_transformer_mhc_mix_output()` applies the post coefficients and combination
matrix. Matrix orientation is **source row, destination column**:

```
input[d] = sum_i pre[i] * residual[i,d]
output[j,d] = sum_i comb[i,j] * residual[i,d] + post[j] * branch[d]
```

All operations use F32 arrays, allocate nothing and own no model or session
state. Outputs must not overlap inputs. Metadata and finite-input validation
precede writes; coefficient generation uses bounded scratch and commits only
after normalization succeeds. A zero-width mix permits null buffers. Arithmetic
follows F32; final BF16 rounding is a separate operation. The existing model
paths remain unchanged.

## Independent qualification

Seven cases were captured from the unchanged original
vLLM CPU Torch mHC helpers, using BF16-representable residuals lifted to F32.
Prepared scores include the original projection/global RMS calculation; the
new C coefficient operation consumes those prepared scores. References cover
1, 2, 4, 8 and 16 streams, GLM's 4096 and DeepSeek's 5120 widths, up to 17 token rows,
and 1, 3, 20 and 128 normalization iterations. Original repeated runs are exact.
Coefficients also agree with a float64 calculation.

Across scalar, AVX2 and AVX512 compilation configurations, **1,436,049** values
are checked, including coupled collapse and post-layer outputs:

| Boundary | Maximum absolute difference |
| --- | ---: |
| Pre coefficients | 5.97e-8 |
| Post coefficients | 1.20e-7 |
| Combination matrix | 7.46e-8 |
| Collapsed input | 4.77e-7 |
| Mixed output | 4.77e-7 |

The threshold is 5e-6 absolute. These are three compilations of portable code,
not three optimized SIMD implementations. This establishes F32 core agreement,
not full-model or original BF16-output bit parity.

The standard suite now runs `test_mhc`. Its small immutable capture checks all
coefficient and mixing boundaries with guard elements. An exact non-symmetric
matrix case catches source/destination transposition. Failure checks include
bad metadata, late nonfinite inputs, coefficient overflow/normalization failure,
empty mixes and unchanged outputs after rejection. ASan/UBSan passes. The clean
build has zero warnings; the standard suite, full CUDA regression suite and
diagnostic isolation pass. ARM/NEON hardware has
not been executed for this new operation.

Reproduce with the qualified reference environment:

```sh
CUDA_VISIBLE_DEVICES='' python test/reference_mhc.py --output /tmp/mhc-reference
python test/verify_mhc.py --reference-dir /tmp/mhc-reference \
  --output-dir /tmp/mhc-verified --mode scalar --mode avx2 --mode avx512
```

The verifier defaults to scalar; optional ISA modes require matching hardware.
The generator writes a fixture header only with explicit `--fixture-header PATH`.
It reproduces every captured array and the checked-in header exactly. Existing
model reference values are unchanged. Source hashes, cases, per-build errors
and scope are in [the machine-readable results](mhc-core.json). External evidence
is under `/tmp/bitnet-goal-mhc`.

Complete model wiring, quantized projection/global RMS order, CUDA residency and
independent full-model short/long/chunked/resumed prefill/decode qualification
remain open. The full inference, offloading, serving, performance and backend
goal remains active.
