# Qwen3.5 MoE Performance Notes

Qwen3.5-style hybrid SSM/MoE models stress several parts of the engine at once:

- SSM layers
- attention layers
- routed experts
- shared experts
- large context metadata
- mixed quant and floating-point tensors

## Current Status

The CPU path is functional for tested Qwen3.5-style GGUFs. Metal/WebGPU can run
supported blocks and fall back to CPU for unsupported SSM or MoE pieces. The
fallback boundary must remain explicit and test-covered.

Latest local top-logit/throughput checks against `llama-server -fa on -np 1`
show that coherence is good, but backend performance is uneven:

| Model | Backend | Coherence | bitnet.c tok/s | llama.cpp tok/s | Ratio |
|---|---|---:|---:|---:|---:|
| `Qwen3.5-9B-Q4_K_M` | ARM NEON / CPU | 8/8 top-1, mean top-10 9.88 | 17.34 | 18.14 | 0.956 |
| `Qwen3.5-9B-Q4_K_M` | Metal | 8/8 top-1, mean top-10 9.75 | 15.28 | 32.80 | 0.466 |
| `Qwen3.5-35B-A3B-Q4_K_M` | ARM NEON / CPU | 8/8 top-1, mean top-10 9.62 | 3.91 | 4.34 | 0.901 |
| `Qwen3.5-35B-A3B-Q4_K_M` | Metal | 8/8 top-1, mean top-10 9.88 | 0.61 | 48.82 | 0.012 |

The CPU path is close enough to justify targeted tuning. The Metal MoE path is
not a kernel-micro-optimization problem yet; the first missing piece is clear
placement/fallback reporting and a whole-block GPU MoE plan that avoids
CPU/GPU ping-pong.

## Measurement Caveats

Performance varies heavily with:

- mmap versus pread
- expert cache size
- page-cache warmth
- whether the backend keeps SSM/MoE blocks on GPU or falls back to CPU
- short prompts where prefill dominates
- large context settings that inflate KV allocation

When comparing against llama.cpp, record the actual backend placement. Some
llama.cpp runs that look CPU-oriented can still route selected MoE work to Metal
or another GPU backend.

## Improvement Plan

CPU / ARM NEON:

- Add repeatable Qwen3/Qwen3.5 dense and MoE gates that run top-k coherence and
  throughput with `--ngl 0`, fixed `--maxseq`, and model-size-specific token
  budgets. The acceptance result should be top-1/top-k parity plus throughput
  ratio, not only "runs without crashing."
- Profile Q4_K_M dense and MoE separately. Dense Qwen3.5 is near parity, so
  start with Q4_K/Q5_K/Q6_K NEON matvec/logits costs before changing MoE code.
- For MoE, measure expert routing distribution, cache hit rate, mmap/pread
  behavior, active expert count, shared expert cost, and page faults. Optimize
  expert locality and batching from those measurements.
- Batch routed expert gate/up/down work when experts share quant type and
  shape, so activation quantization and thread dispatch are amortized across
  multiple expert matvecs.
- Sweep thread count, expert batch size, and `--cache-mb` for the 35B-A3B
  fixture. Encode the best default policy in MoE execution once the sweep is
  stable.

Metal:

- Add per-layer/per-op placement logs for Qwen3.5 dense and MoE: native Metal,
  repacked Metal, split/fused Metal, CPU fallback, and exact missing
  capability. This is required before interpreting low Metal tok/s.
- For dense Qwen3.5, profile Q4_K_M Metal rows with `bench_kernels --metal`
  and add native/repacked kernels only for rows that dominate end-to-end time.
- For MoE, move whole blocks rather than isolated matvecs: router logits,
  top-k routing, expert gate/up/down, shared experts, accumulation, residual,
  and norm should stay backend-resident when the block is placed on Metal.
- Add a Metal expert cache/upload policy with explicit capacity, LRU/working
  set reporting, and a clear fallback when a layer cannot fit.
- Avoid per-op CPU/GPU ping-pong. If an unsupported SSM/MoE feature forces CPU
  fallback, schedule a whole block on CPU unless measured evidence shows a
  finer split is faster.
- Capture llama.cpp placement logs for the same `-ngl 99` MoE runs. Throughput
  comparisons are only useful if both engines are actually placing the same
  kinds of work on Metal.

## Useful Flags

```bash
./bitnet models/qwen3.5.gguf --pread --cache-mb 4096 --maxseq 4096 -t 8
./bitnet models/qwen3.5.gguf --metal --maxseq 4096 -t 8
./test_coherence models/qwen3.5.gguf --metal
```

Use `--maxseq` deliberately. Some hybrid models advertise very large context
windows, and allocating full KV for the advertised limit can dominate memory.
