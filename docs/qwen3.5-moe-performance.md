# Qwen3.5-35B-A3B Q4_K_M Performance Analysis

Apple M1 Max (32 GB), 8 threads, 64 tokens generated. Hybrid SSM + MoE architecture: 40 layers (10 attention + 30 SSM), 256 experts/layer, K=8 active, shared expert per layer.

## Throughput by I/O Mode

| Mode | tok/s | RSS | MB/tok | Cache hit | pf_wait (ms) |
|------|-------|-----|--------|-----------|-------------|
| mmap (warm cache) | **5–7** | 20.1 GB | 580 | — | 0 |
| pread + 4 GB cache | ~4.5 | 11.3 GB | 157 | 72.9% | — |
| pread (no cache) | ~2.5 | 11.0 GB | 580 | — | — |

vs llama.cpp `-ngl 0` (same hardware): **6–8 tok/s** → bitnet.c is **~80%** of llama.cpp. Note: llama.cpp routes MoE expert dispatch (`MUL_MAT_ID`) to Metal GPU even with `-ngl 0` when batch_size >= 32 (always true for 256 experts). The comparison is pure CPU vs CPU+GPU. MoE numbers vary significantly with page cache state (21 GB model on 32 GB machine).

## Time Breakdown (mmap, per token ≈ 85ms)

| Phase | Total (ms) | Per-tok (ms) | % | Description |
|-------|-----------|-------------|---|-------------|
| **down** | 1819 | 28.4 | **59%** | Down projection matvec (batched multi-dispatch) |
| **gate+up** | 697 | 10.9 | **23%** | Gate+up batch matvec (cross-expert batched) |
| **shared** | 406 | 6.3 | **13%** | Shared expert FFN (gate+up batched + down) |
| route | 188 | 2.9 | 6% | Router matvec + softmax + top-K selection |
| swiglu | 161 | 2.5 | — | SwiGLU activation (element-wise) |
| norm | 4 | 0.1 | — | RMSNorm |
| accum | 6 | 0.1 | — | Weighted expert accumulation |

## Dispatch Batching

Gate+up projections are batched across all K=8 experts in a single `bn_quant_matvec_batch` dispatch (2K=16 tasks). Down projections use `bn_quant_matvec_multi` — each expert has a different post-SwiGLU input, but all K are pre-quantized independently and dispatched in a single thread pool wake/wait cycle. Shared expert gate+up are also batched (2 tasks, one dispatch). This reduces dispatches from ~480/token to ~80/token.

## Pread Cache Efficiency

With 256 experts (vs 128 for Qwen3-30B), the 4 GB cache holds ~2800 expert slots. At K=8 and 40 layers, each token accesses 320 experts. The 72.9% hit rate cuts I/O from 580 → 157 MB/tok, saving 9 GB RSS (20→11 GB) with only 7% speed loss vs mmap.

| Cache size | Expected hit rate | Notes |
|-----------|------------------|-------|
| 0 (off) | 0% | 580 MB/tok, full SSD streaming |
| 4 GB | 72.9% | Default, good balance |
| 6 GB | ~85% | Diminishing returns |
| 8 GB | ~90% | Near-ceiling for Zipf distribution |

## Optimization Opportunities

### 1. Down projection batching (highest impact, ~30% potential)

Pre-quantize each expert's post-SwiGLU hidden state independently, then batch all K=8 down projections in a single dispatch. This eliminates 7 of 8 dispatch overheads per layer (280 fewer dispatches/token). Requires K separate x_q buffers but saves significant thread synchronization cost.

### 2. Shared expert fusion (13% of compute)

The shared expert runs 4 separate operations (gate matvec, up matvec, SwiGLU, down matvec) with 4 dispatches per layer. A fused kernel or at minimum batching gate+up together would reduce overhead.

### 3. Route + I/O overlap (pread path)

Currently routing completes before expert loading starts. Starting expert prefetch during router softmax+top-K computation would hide ~3ms of I/O latency per layer.

### 4. Larger default cache for 256-expert models

Auto-detect expert count and scale cache budget: `cache_mb = min(8192, n_experts * 16)` for 256-expert models would default to 4 GB (current) but could be bumped.

## Model Configuration

```
Architecture: qwen35moe (hybrid SSM + attention + MoE)
dim: 2048, layers: 40 (10 attn + 30 SSM)
heads: 16, kv_heads: 2, head_size: 256
experts: 256, active: 8, expert_hidden: 512
shared_expert_hidden: 512 (with sigmoid gate)
ssm: conv_kernel=4, state_size=128, inner_size=4096
rope: partial (64 of 256 dims), theta=10M, MROPE sections=[11,11,10]
vocab: 248320, max_seq: 262144
quantization: Q4_K_M (gate/up: Q4_K, down: Q5_K)
file size: 21.0 GB
```
