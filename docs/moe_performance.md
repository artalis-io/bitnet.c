# MoE Compute Layer Optimization

Profiled with Qwen3-30B-A3B-Q4_K_M (48 layers, 128 experts, 8 active).
Bottleneck is in the compute layer, not SSD streaming.

## Results

| Metric | Before | After | Change |
|---|---|---|---|
| tok/s | 3.53 | 10.17 | **+188%** |
| compute_ms (32 tok) | ~7730 | ~1848 | -76% |
| route_ms (32 tok) | ~113 | ~75 | -34% |
| dispatches/token | 1,152 | 144 | -87% |

Measured on Apple M1, mmap mode, 8 threads. Average of 5 warm runs.

## Optimizations Applied

### 1. Q4_K batch matvec path (Phase 1)

Added generic float-x batch path to `bn_quant_matvec_batch()` covering all K-quant
types (Q4_K, Q5_K, Q6_K, Q3_K, Q2_K), Q8_K, BF16, Q4_1, and all IQ types.

These types use float x directly — no int8 quantization needed. The batch path
dispatches up to 16 tasks in a single `bn_tp_dispatch()` call.

`bn_quant_get_float_kernel(type)` maps quant type → platform-optimal kernel function.

### 2. Cross-expert batched dispatch (Phase 2)

Restructured `bn_moe_forward()` mmap path to batch all K experts' projections:

- **Gate+up**: 2K matvecs dispatched as single `bn_quant_matvec_batch()` call
  (K gate + K up into separate output buffers per expert)
- **Down**: K matvecs dispatched via `bn_tp_dispatch()` with per-expert ctx
  structs (each expert has different x after SwiGLU)

Dispatch count: **24/layer → 3/layer** (gate+up batch, SwiGLU, down batch).
With 48 layers: 1,152 → 144 dispatches/token.

New batch buffers in `BnMoEState`:
- `expert_hb_batch[K]` — K gate outputs [moe_hidden]
- `expert_hb2_batch[K]` — K up outputs [moe_hidden]
- `expert_down_batch[K]` — K down outputs [dim]
- Memory: ~393KB for K=8, dim=3072, moe_hidden=4608

pread path remains serial (single scratch buffer constraint).

### 3. Parallel SwiGLU activation (Phase 3)

SwiGLU dispatched as K tasks to thread pool, one per expert.
Each task processes moe_hidden elements of that expert's activation.

### 4. Vectorized router (Phase 4)

Router matvec uses SIMD (NEON 4-accumulator FMA / AVX2 2-accumulator FMA)
and dispatches n_experts dot products to the thread pool.

128 experts × dim → parallelized across threads with vectorized inner loop.

## Architecture Notes

- Q4_K uses float x directly (no SDOT, no int8 quantization). The "redundant x
  quantization" bottleneck only applies to Q4_0/Q8_0/I2_S models.
- `BN_MAX_MOE_K = 16` caps cross-expert batching. Models with K > 16 fall back
  to the serial path. No known model exceeds K=16.
- Mixed quant types within a layer are not possible — all experts in a layer
  share the same quant type.
