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

## Useful Flags

```bash
./bitnet models/qwen3.5.gguf --pread --cache-mb 4096 --maxseq 4096 -t 8
./bitnet models/qwen3.5.gguf --metal --maxseq 4096 -t 8
./test_coherence models/qwen3.5.gguf --metal
```

Use `--maxseq` deliberately. Some hybrid models advertise very large context
windows, and allocating full KV for the advertised limit can dominate memory.
