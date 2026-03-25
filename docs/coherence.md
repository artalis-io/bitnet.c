# GPU/CPU Coherence Test Results

Cross-backend validation: GPU forward pass vs CPU, SIMD (NEON SDOT) vs scalar matvec, GPU standalone matvec vs CPU scalar.

## Running

```bash
make BN_ENABLE_GPU=1 test_coherence
./test_coherence <model.gguf> --gpu    # all 3 phases
./test_coherence <model.gguf>          # CPU-only (Phase 2 only)
```

## Phases

| Phase | What | Pass criteria |
|-------|------|---------------|
| 1 | GPU vs CPU greedy decode (5 tokens) | First 3 tokens must match (FP32 drift allowed after) |
| 2 | Scalar kernel vs compile-time SIMD backend (layer 0 weights) | max_diff < 2.0 per weight |
| 3 | GPU standalone matvec vs CPU scalar (layer 0 wq) | max_diff < 2.0 |

## Results (M1 Max, 2025-03-25)

| Model | Weight types | Phase 1 (GPU=CPU) | Phase 2 (NEON SDOT vs scalar) | Phase 3 (GPU matvec) | Result |
|-------|-------------|-------------------|-------------------------------|---------------------|--------|
| bitnet-b1.58-2B-4T | I2_S | 5/5 match | 7/7 PASS (max 1.25) | PASS (0.48) | **PASS** |
| Llama3-8B-1.58-TQ1_0-F16 | TQ1_0 | 0/5 (GPU zeros) | 7/7 PASS (max 0.01) | PASS (0.01) | **FAIL** |
| olmoe-1b-7b-q2k | Q2_K/Q3_K | 5/5 match | 4/4 PASS (0.00) | PASS (0.00) | **PASS** |
| olmoe-1b-7b-q4_0 | Q4_0 | 5/5 match | 4/4 PASS (max 0.01) | PASS (0.01) | **PASS** |
| qwen2.5-3b-instruct-q4_0 | Q4_0 | 5/5 match | 7/7 PASS (max 0.02) | PASS (0.02) | **PASS** |
| Qwen3-0.6B-Q8_0 | Q8_0 | 5/5 match | 7/7 PASS (max 0.02) | PASS (0.01) | **PASS** |
| Qwen3-30B-A3B-Q4_K_M | Q4_K/Q6_K (MoE) | 5/5 match | 4/4 PASS (max 0.03) | PASS (0.01) | **PASS** |
| Qwen3.5-35B-A3B-Q4_K_M | Q4_K (SSM+Attn) | 5/5 match | SKIP (SSM layer 0) | SKIP | **PASS** |
| Qwen3.5-9B-Q4_K_M | Q4_K/Q6_K | 5/5 match | 3/3 PASS (max 0.03) | SKIP (SSM) | **PASS** |
| TinyMixtral-4x220M-Q4_K_M | Q4_K/Q6_K (MoE) | 5/5 match | 4/4 PASS (max 0.01) | PASS (0.01) | **PASS** |

## Known Issues

- **Llama3-8B TQ1_0**: GPU forward pass returns all-zero logits. TQ1_0 standalone matvec (Phase 2, 3) passes, so the issue is in the forward-pass wiring for TQ1_0 weights, not the shader itself. Needs investigation.
- **MoE models** (OLMoE, Qwen3-30B, TinyMixtral): GPU forward pass falls back to CPU for MoE layers (router_weight check in forward_gpu). Phase 1 compares CPU-via-GPU-fallback vs pure-CPU, so tokens match trivially. Phase 2/3 still validate the matvec kernels.
- **SSM hybrid models** (Qwen3.5): Layer 0 may be an SSM layer with no standard attention weights, causing Phase 2/3 SKIPs.
- **I2_S SDOT vs scalar**: max_diff up to ~1.3 for large cols (6912) due to INT8 x-quantization noise in the SDOT path. This is expected — the SDOT kernel quantizes x to INT8 for integer dot products, while scalar uses FP32 throughout.

## Quant Types Covered

| Type | Phase 2 tested | Phase 3 tested | Models |
|------|---------------|---------------|--------|
| I2_S | yes | yes | bitnet-2B |
| TQ1_0 | yes | yes | Llama3-8B |
| Q2_K | yes | yes | olmoe-q2k |
| Q3_K | yes | - | olmoe-q2k |
| Q4_0 | yes | yes | olmoe-q4_0, qwen2.5-3b |
| Q4_K | yes | yes | Qwen3-30B, Qwen3.5-9B, TinyMixtral |
| Q6_K | yes | - | Qwen3-30B, Qwen3.5-9B, TinyMixtral |
| Q8_0 | yes | yes | Qwen3-0.6B |
