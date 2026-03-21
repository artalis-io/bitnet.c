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

Qwen3-30B-A3B-Q4_K_M (17.7 GB, 128 experts/layer, K=8), Apple M1 Max 8T, 128 tokens generated.

| Mode | tok/s | Hit rate | MB/tok | pf_wait (ms) | RSS |
|------|-------|----------|--------|-------------|-----|
| mmap | 14.13 | — | 1046 | 0 | 16.2 GB |
| pread + 4 GB cache | 11.88 | 86% | 144 | 1148 | 10.3 GB |
| pread + 2 GB cache | 9.18 | 63% | 388 | 3666 | 10.2 GB |
| pread (no cache) | 6.54 | — | 1046 | 8858 | 8.3 GB |

The expert LRU cache (open-addressing hash + intrusive LRU list) stores full expert weights (gate+up+down) in a contiguous slab. Default 4 GB budget → 1402 slots → 86% hit rate, cutting I/O from 1046 → 144 MB/tok. Cache hits are batched (cross-expert gate+up matvec dispatch like the mmap path). First miss I/O is overlapped with hit batch compute. `--cache-mb N` to configure (0 to disable, pread mode only).

**When to use which mode:**
- **mmap** (default): Use when the model fits comfortably in RAM. Best throughput, simplest.
- **pread + cache**: Use when the model exceeds available RAM, or you want to limit RSS. The 4 GB default cache gives 86% hit rate on Qwen3-30B with 10 GB RSS vs 16 GB for mmap.
- **madvise**: Experimental. Currently slower than both mmap and pread due to syscall overhead (1152 madvise calls/token). Not recommended for production use.

## vs llama.cpp (b8320)

Measured with `llama-bench`, same hardware (M1 Max, 8 threads). Both use `-p 0 -n 256` (pure generation, no prompt).

| Model | Format | bitnet.c (PGO) | llama.cpp CPU | llama.cpp Metal | CPU ratio |
|-------|--------|----------------|---------------|-----------------|-----------|
| BitNet b1.58 2B-4T | I2_S | 52.5 | — | — | — |
| Qwen2.5 3B Instruct | Q4_0 | 25.4 | 40.2 | 84.4 | 63% |
| Llama3 8B 1.58 | TQ1_0 | 14.5 | 19.3 | N/A | 76% |

llama.cpp's remaining CPU advantage comes from multi-row interleaved kernels (`block_q4_0x4` packing 4 rows together, using `vdotq_laneq_s32` to compute 4 output rows per pass and amortize activation loads). Both engines use weight-repacked NEON kernels for Q4_0 matvec. Metal is GPU offload (`-ngl 99`). TQ1_0 Metal is not implemented in llama.cpp b8320.

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
