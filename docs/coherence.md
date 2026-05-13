# Coherence Tests

`test_coherence` validates cross-backend behavior using a real GGUF model.

## Build And Run

```bash
# CPU-only scalar/SIMD checks
make test_coherence
./test_coherence models/model.gguf

# Metal
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/model.gguf --metal

# WebGPU
make fetch-wgpu
make BN_ENABLE_WEBGPU=1 test_coherence
./test_coherence models/model.gguf --webgpu
```

`--gpu` remains accepted by the CLI/test harness as a compatibility spelling for
the available GPU backend in older scripts.

## Phases

| Phase | What it checks | Pass criteria |
|---|---|---|
| 1 | GPU/backend forward pass versus CPU greedy decode | First 3 of 5 tokens match. |
| 2 | Compile-time SIMD backend versus scalar matvec on layer 0 weights | `max_diff < 2.0` per weight. |
| 3 | Standalone GPU matvec versus CPU scalar on layer 0 `wq` when available | `max_diff < 2.0`. |

Some models skip a phase when layer 0 does not expose the required standard
attention tensor, for example SSM-first hybrids.

## Interpreting Results

Exact token equality is expected on many small and medium models. Larger models
can diverge after a few tokens from harmless FP32 reduction-order drift, even
when standalone matvec checks pass.

Forward-pass GPU reductions use compensated local accumulation in RMSNorm,
residual RMSNorm, per-head RMSNorm, softmax sums, GQA score dots, GQA combine,
and SSM delta paths. The final workgroup or simdgroup reduction order can still
vary by backend, but local compensation keeps the main accumulation paths closer
to the CPU reference before that final backend-specific reduction.

MoE and hybrid models may use CPU fallback for unsupported GPU blocks. That is
acceptable only when the fallback is deterministic and visible in tests or debug
output.

## Quant Coverage

The test is most useful when the selected model exercises the quant formats and
model blocks touched by the change. Keep at least one representative model for:

- ternary: `I2_S`, `TQ1_0`, `TQ2_0`
- legacy quants: `Q4_0`, `Q4_1`, `Q8_0`
- k-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`
- IQ formats: `IQ2_*`, `IQ3_*`, `IQ4_*`
- dense attention, MoE, and hybrid SSM/attention models
