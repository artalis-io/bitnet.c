# Benchmarks

Single-core and multi-core performance on Apple M1 Max (32 GB), macOS.

**Toolchain:** Apple Clang 17.0.0, Emscripten 4.0.1, Node.js 20.20.1

## Throughput (tok/s)

| Model | Format | Params | NEON 1T | NEON 8T | NEON 8T PGO | WASM 1T |
|-------|--------|--------|---------|---------|-------------|---------|
| BitNet b1.58 2B-4T | I2_S | 2B | 20.5 | 48.7 | 52.5 | 7.3 |
| Qwen2.5 3B Instruct | Q4_0 | 3B | 4.3 | 18.7 | 25.4 | 4.5 |
| Llama3 8B 1.58 | TQ1_0 | 8B | 3.2 | 10.4 | 14.5 | - |

PGO build trained on respective model (128 tokens). Q4_0 uses weight repacking (split scales/qs) for NEON SDOT. WASM is single-threaded (no SharedArrayBuffer in Node.js RAWFS mode).
Llama3 8B (3.3 GB) exceeds the WASM 4 GB address space with runtime allocations.

## MoE Throughput (tok/s)

Apple M1 Max 8T, 64-128 tokens generated.

### Qwen3-30B-A3B-Q4_K_M (17.7 GB, 128 experts/layer, K=8)

| Mode | tok/s | Hit rate | MB/tok | RSS |
|------|-------|----------|--------|-----|
| mmap | **19.04** | — | 1046 | 16.2 GB |
| pread + 4 GB cache | 11.88 | 86% | 144 | 10.3 GB |
| pread (no cache) | 6.54 | — | 1046 | 8.3 GB |

### Qwen3.5-35B-A3B-Q4_K_M (21.0 GB, 256 experts/layer, K=8, hybrid SSM+MoE)

| Mode | tok/s | Hit rate | MB/tok | RSS |
|------|-------|----------|--------|-----|
| mmap | **11.74** | — | 580 | 20.1 GB |
| pread + 4 GB cache | 9.67 | 73% | 157 | 11.3 GB |
| pread (no cache) | 5.83 | — | 580 | 11.0 GB |

Expert LRU cache (open-addressing hash + intrusive LRU list) stores full expert weights (gate+up+down) in a contiguous slab. Batched down projection dispatch (K tasks in one wake/wait cycle) and shared expert gate+up batching reduce dispatch overhead. `--cache-mb N` to configure (0 to disable, pread mode only).

**When to use which mode:**
- **mmap** (default): Use when the model fits comfortably in RAM. Best throughput, simplest.
- **pread + cache**: Use when the model exceeds available RAM, or you want to limit RSS.
- **madvise**: Experimental. Currently slower than both mmap and pread due to syscall overhead. Not recommended for production use.

## vs llama.cpp (b8320)

Measured with `llama-bench`, same hardware (M1 Max, 8 threads), warm page cache (`cat model.gguf > /dev/null` before measurement).

| Model | Format | bitnet.c | llama.cpp `-ngl 0` | Ratio | Notes |
|-------|--------|----------|-------------------|-------|-------|
| BitNet b1.58 2B-4T | I2_S | 52.5 | — | — | Ternary (no llama.cpp support) |
| Qwen2.5 3B Instruct | Q4_0 | 25.4 | 40.2 | 63% | Multi-row kernel gap |
| Llama3 8B 1.58 | TQ1_0 | 14.5 | 19.3 | 76% | Multi-row kernel gap |
| Qwen3-30B-A3B MoE | Q4_K_M | 8–10 | 10–12 | ~80% | llama.cpp uses Metal GPU for MoE |
| Qwen3.5-35B-A3B MoE | Q4_K_M | 5–7 | 6–8 | ~80% | llama.cpp uses Metal GPU for MoE |

### MoE performance by page cache state (Qwen3-30B, M1 Max 32 GB)

| Condition | bitnet.c | llama.cpp | Gap |
|-----------|----------|-----------|-----|
| Cold (SSD thrashing) | 5–7 tok/s | 7–8 tok/s | ~30% |
| Warm page cache | 8–10 tok/s | 10–12 tok/s | ~20% |
| Warm + bitnet.c runs first | 19–20 tok/s | 10–12 tok/s | bitnet 2x faster |

**Why llama.cpp leads on MoE:** llama.cpp with `-ngl 0` on Apple Silicon still routes MoE expert dispatch (`MUL_MAT_ID`) to Metal GPU when batch_size >= 32 (always true for 128+ experts). The "CPU" comparison for MoE models is actually pure CPU vs CPU+GPU. The ~20% warm-cache gap is entirely from Metal GPU providing higher sustained memory bandwidth for scattered expert weight reads (thousands of GPU threads hide memory latency vs 8 CPU threads that stall on cache misses).

The CPU Q4_K dot product kernel is equivalent between both engines — there is no kernel optimization that will close this gap. It requires either a Metal backend (against project identity) or reducing total expert weight reads (already optimized with batched dispatch, Q8_K integer accumulation, etc.).

**Dense models:** llama.cpp leads due to multi-row interleaved Q4_K/Q4_0 kernels that amortize activation loads across rows.

## Per-Kernel Bandwidth (GB/s)

### BitNet b1.58 2B-4T (I2_S)

| Kernel | Dims | NEON 1T | NEON 4T | WASM 1T | WASM/NEON |
|--------|------|---------|---------|---------|-----------|
| wq | 2560x2560 | 14.3 | 34.0 | 13.6 | 0.95x |
| wk | 640x2560 | 14.2 | 26.5 | 14.0 | 0.99x |
| wv | 640x2560 | 14.2 | 29.7 | 13.6 | 0.96x |
| wo | 2560x2560 | 14.1 | 30.7 | 13.8 | 0.98x |
| up | 6912x2560 | 12.1 | 33.3 | 13.7 | **1.13x** |
| down | 2560x6912 | 12.4 | 32.1 | 14.0 | **1.13x** |
| gate | 6912x2560 | 12.0 | 34.0 | 13.6 | **1.13x** |

WASM Relaxed SIMD SDOT achieves 95-113% of native NEON SDOT throughput on I2_S ternary matvec. The FFN kernels (up/down/gate) are slightly faster in WASM due to V8's superior instruction scheduling for large matrices.

### Qwen2.5 3B Instruct (Q4_0)

| Kernel | Dims | NEON 1T | NEON 4T | WASM 1T | WASM/NEON |
|--------|------|---------|---------|---------|-----------|
| wq | 2048x2048 | 8.3 | 20.7 | 9.1 | **1.09x** |
| wk | 256x2048 | 8.2 | 7.3 | 9.3 | **1.13x** |
| wv | 256x2048 | 8.3 | 19.0 | 9.4 | **1.13x** |
| wo | 2048x2048 | 8.2 | 21.7 | 9.2 | **1.12x** |
| up | 11008x2048 | 6.7 | 27.9 | 9.1 | **1.36x** |
| down | 2048x11008 | 8.1 | 29.0 | 9.4 | **1.16x** |
| gate | 11008x2048 | 8.0 | 28.3 | 9.1 | **1.13x** |

WASM Q4_0 SDOT kernels consistently outperform single-threaded NEON by 9-36%, likely due to V8's SIMD JIT optimizations and more efficient register allocation for the Q4 dequant+dot product loop.

### Llama3 8B 1.58 (TQ1_0) — native only

| Kernel | Dims | NEON 1T | NEON 4T |
|--------|------|---------|---------|
| wq | 4096x4096 | 5.0 | 17.1 |
| wk | 1024x4096 | 5.3 | 16.7 |
| wv | 1024x4096 | 5.3 | 15.7 |
| wo | 4096x4096 | 5.0 | 16.6 |
| up | 14336x4096 | 5.0 | 18.4 |
| down | 4096x14336 | 5.1 | 18.3 |
| gate | 14336x4096 | 5.1 | 18.3 |
| logits (F16) | 128256x4096 | 12.1 | 46.5 |

## Per-Kernel Latency (us/call)

### BitNet b1.58 2B-4T (I2_S)

| Kernel | NEON 1T | NEON 4T | WASM 1T |
|--------|---------|---------|---------|
| wq | 115 | 49 | 121 |
| wk | 30 | 16 | 30 |
| wv | 30 | 14 | 31 |
| wo | 117 | 54 | 119 |
| up | 365 | 133 | 325 |
| down | 358 | 139 | 318 |
| gate | 369 | 130 | 327 |

### Qwen2.5 3B Instruct (Q4_0)

| Kernel | NEON 1T | NEON 4T | WASM 1T |
|--------|---------|---------|---------|
| wq | 284 | 114 | 260 |
| wk | 37 | 41 | 33 |
| wv | 36 | 16 | 32 |
| wo | 287 | 109 | 257 |
| up | 1898 | 454 | 1397 |
| down | 1574 | 440 | 1361 |
| gate | 1582 | 448 | 1402 |

## Notes

**Backend details:**
- **ARM NEON + SDOT**: `vdotq_s32` for integer dot products (I2_S, TQ1, TQ2, Q4_0, Q8_0), `vmlaq_f32` FMA, native FP16 logits with `vfmaq_f16`
- **WASM Relaxed SIMD**: `i32x4.relaxed_dot_i8x16_i7x16_add` (SDOT equivalent), `f32x4.relaxed_madd` (FMA), vectorized F16 bit-manipulation for logits
- Multi-threading uses a persistent pthread pool with condvar dispatch (~2us overhead)

**WASM limitations:**
- Single-threaded only (no `SharedArrayBuffer` in Node.js RAWFS mode)
- 4 GB address space limit (wasm32) — models + runtime must fit in 4 GB
- Logits benchmark unreliable (V8 JIT eliminates unused results); tok/s numbers are authoritative

**Reproducing:**
```bash
# Native
make bench
./bench_kernels models/<model>.gguf --iters 100 --threads 4 --toks 32

# WASM (requires Emscripten + Node.js 20+)
bash bench/bench_wasm.sh
node --experimental-wasm-relaxed-simd bench/bench_wasm.js models/<model>.gguf --iters 100 --toks 32
```
