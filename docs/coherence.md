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

The strict CPU matrices (`test/qwen_cpu_parity.sh` and
`test/gemma4_cpu_parity.sh`) run every selected model/backend with both default
prefill and tokenwise prefill, followed by greedy decode. Tokenwise mode passes
`--no-prefill` to bitnet and batch/microbatch size 1 to llama.cpp. Use
`CPU_PARITY_PREFILL_MODES=default` or `tokenwise` for a focused rerun; the
family-specific `QWEN_CPU_PARITY_PREFILL_MODES` and
`GEMMA4_CPU_PARITY_PREFILL_MODES` override that setting. The default is
`default,tokenwise`. Default prefill follows runtime policy, so these CLI checks
alone do not prove that every model executed batched kernels.

The CUDA matrices (`bench/qwen_cuda_matrix.sh` and
`bench/gemma4_cuda_matrix.sh`) use the same pair of modes when
`RUN_LLAMA_COMPARE=1`. `CUDA_PARITY_PREFILL_MODES` selects `default`,
`tokenwise`, or the default `default,tokenwise`. A failure in the first mode
does not skip the second; either failure makes the matrix fail.

For AVX comparisons, set `LLAMA_AVX2_BIN_DIR` and `LLAMA_AVX512_BIN_DIR` to
the corresponding reference builds. Reduction order can change greedy
continuations across ISAs and prefill modes; compare each configuration with
its matching llama.cpp configuration. Strict comparison requires sampled token
IDs, equal output counts, and successful exits from both inference processes.

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


CUDA hybrid MoE prefill starting at position zero uses 512-token physical batches when every layer and
cache span passes the prefix-attention preflight. This matches the pinned
llama.cpp default microbatch size. Current eligibility requires FP16 KV,
head256/GQA16 attention and the reference recurrent backend capability, within
2048 total keys. Prefix continuation is limited to internal batches whose KV
handoff succeeded. Public resumed requests and other paths retain their existing scheduling.

Set `BN_GPU_PREFILL_FULL_PROMPT=1` before creating the backend to retain one full
prompt batch for diagnostics. Unset the variable to restore microbatching;
backend boolean policies are presence-based, so a value of `0` also enables it.
When comparing this mode, set the reference
physical batch size explicitly (`--llama-ubatch 2048` in `test/compare_llama.sh`).
Batch scheduling and host-to-backend KV handoff live in the transformer runtime;
backend preflight checks resident cache capacity and RoPE resources without
changing request state.
