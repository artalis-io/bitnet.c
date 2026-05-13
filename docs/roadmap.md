# Roadmap

Development roadmap for bitnet.c.

## Phase 1: Core Modules (Naive C, Correct) — Done

- [x] `platform` — mmap/buffer abstraction, timing
- [x] `gguf` — GGUF v3 binary format parser
- [x] `quant` — TQ1_0/TQ2_0 dequantization, ternary matvec
- [x] `model` — GGUF → config, architecture rules, and immutable weights
- [x] `transformer` — full forward pass (RoPE, GQA, sub-norms, tied embeddings)
- [x] `tokenizer` — BPE encode/decode from GGUF metadata
- [x] `sampler` — argmax, multinomial, top-p sampling
- [x] `main` — CLI entry point with argument parsing
- [x] Unit tests for all modules
- [x] Makefile (native build)

## Phase 2: WASM Build — Done

- [x] `wasm/api.c` — EMSCRIPTEN_KEEPALIVE wrapper functions
- [x] `wasm/build.sh` — Emscripten build script
- [x] `wasm/worker.js` — Web Worker with streaming token output
- [x] `wasm/index.html` — minimal browser demo

## Phase 3: Validation & Correctness

- [ ] Greedy decode comparison vs `llama-cli` reference output
- [ ] Bit-exact TQ1_0/TQ2_0 dequantization validation against llama.cpp
- [ ] Activation dump mode (`-DDEBUG`) with layer-by-layer checkpoints
- [ ] Test against multiple BitNet GGUF models (2B, 4B variants)
- [ ] Validate tokenizer output matches reference BPE implementation
- [ ] Edge cases: empty prompt, single token, max sequence length

## Phase 4: SIMD Optimization — Done

- [x] ARM NEON kernels for I2_S/TQ1_0/TQ2_0 ternary matvec
- [x] SDOT (vdotq_s32) int8 accumulation for I2_S — 2x speedup over float FMA
- [x] Arithmetic ternary decode `(bits - 1)` — 15% speedup over compare-based
- [x] Batch matvec dispatch (QKV, gate+up grouped)
- [x] Native FP16 logits path via `-mcpu=apple-m1`
- [x] x86 AVX2 kernels for all quant formats (I2_S, Q4_0, Q8_0, Q6_K, Q8_K, Q4_K, Q5_K, Q3_K)
- [x] AVX2 fine-tuning: FMA across all kernels, Q8_0 DPBUSD, TQ1/TQ2 AVX2 kernels, Q5_K/Q3_K vectorization
- [x] WASM SIMD128 kernels for all quant formats

## Phase 5: Memory & Performance — Done

- [x] Pthread thread pool (~2us condvar dispatch, replaces OMP fork/join)
- [x] Arena-backed session buffers for request-local state
- [x] RoPE frequency + cos/sin precomputation
- [x] Preallocated sampler candidates buffer (eliminates per-token malloc)
- [x] Prefetch hints in I2_S SDOT, TQ1_0, TQ2_0 kernels
- [x] KV cache quantization (F16 KV via --kv16)
- [x] Sliding window KV cache (ring buffer, continues past seq_len)
- [x] Batch inference (bn_transformer_prefill)
- [x] Profile-guided optimization (PGO build)
- [x] INT8 output embeddings (~52.5 tok/s)

## Phase 6: Modular Backend Architecture — Done

- [x] Split `quant.c` into per-format per-backend modules (`src/quant/`)
- [x] Split `transformer.c` into per-backend modules (`src/transformer/`)
- [x] Internal headers (`quant_internal.h`, `transformer_internal.h`) with context structs
- [x] Backend selection via Makefile variables (ARM: NEON+scalar, x86: AVX2+scalar)
- [x] AVX2 cross-compile syntax check (`make avx2-check`)

## Phase 7: Extended Quantization Formats — Done

- [x] Q4_0 (4-bit) with SDOT/DPBUSD integer matvec
- [x] Q8_0 (8-bit) with NEON/AVX2/WASM kernels
- [x] Q6_K (6-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q8_K (8-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q4_K (4-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q5_K (5-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q3_K (3-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Non-tied output weights (separate output projection matrix)

## Phase 8: Extended Format Coverage — Done

- [x] Q2_K (2-bit k-quant) with NEON/AVX2/WASM kernels
- [x] Q4_1 (4-bit with min) with NEON/AVX2/WASM kernels
- [x] BF16 weight type with NEON/AVX2/WASM kernels
- [x] IQ4_NL (4-bit non-linear codebook) with NEON/AVX2/WASM kernels
- [x] IQ4_XS (4-bit non-linear with sub-block scales) with NEON/AVX2/WASM kernels
- [x] IQ3_XXS (3-bit codebook) with NEON/AVX2/WASM kernels
- [x] IQ3_S (3-bit codebook with separate signs) with NEON/AVX2/WASM kernels
- [x] IQ2_XXS (2-bit codebook) with NEON/AVX2/WASM kernels
- [x] IQ2_XS (2-bit codebook with scales) with NEON/AVX2/WASM kernels
- [x] IQ2_S (2-bit codebook, 1024-entry grid) with NEON/AVX2/WASM kernels

## Performance Analysis (M1 Max, bitnet-b1.58-2B-4T)

### Current: ~52.5 tok/s (8 P-cores)

The workload is **DRAM bandwidth-bound**. Each token reads ~0.83 GB from memory:

| Component | Data Read | % of Total |
|---|---|---|
| 30x layer I2_S weights (Q/K/V/O + gate/up/down) | 497 MB | 60% |
| Logits (INT8 embedding x 128K vocab) | 328 MB | 40% |
| KV cache (pos-dependent) | ~18 MB | <2% |

M1 Max CPU aggregate DRAM bandwidth: ~55 GB/s (CPU-only; the 400 GB/s spec is GPU-inclusive).
At 52.5 tok/s x 0.83 GB = **~43 GB/s sustained** — 79% of max bandwidth.

### Optimization history

| Change | tok/s | Delta |
|---|---|---|
| Baseline (naive C) | ~15.5 | — |
| SDOT int8 accumulation + batch matvec | ~33 | +113% |
| Arithmetic ternary decode + RoPE precompute | ~38 | +15% |
| Pthread thread pool (replace OMP) | ~41 | +8% |
| Arena allocator + FP16 native logits + prefetch | ~46 | +12% |
| INT8 output embeddings + SDOT logits | ~52.5 | +14% |

### Multi-model benchmarks (M1 Max, 8 P-cores, PGO)

| Model | Format | Size | tok/s |
|---|---|---|---|
| bitnet-b1.58-2B-4T | I2_S | 1.1 GB | 52.5 |
| Qwen2.5-3B-Instruct | Q4_0 | 1.7 GB | 30.0 |
| Llama3-8B-1.58 | TQ1_0 | 3.4 GB | 14.5 |

### What would move the needle

Only **reducing data volume** helps at this point:

1. **Speculative decoding** — use a smaller draft model to reduce per-token cost.
2. **KV cache quantization to INT8** — further reduces attention data at long positions.
3. **Weight clustering / pruning** — reduce I2_S weight data below 497 MB.

## Phase 9: Concurrent Sessions & Prompt Cache — Done

- [x] BnModel/BnSession split (shared model + per-request mutable state)
- [x] `bn_session_create/free/reset` API
- [x] BnPromptCache (shared KV prefix with longest-prefix matching, FIFO eviction)
- [x] Thread-safe prompt cache with configurable byte budget
- [x] Prompt cache integrated in chat mode CLI

## Phase 10: SIMD Backend Parity — Done

- [x] AVX2 flash GQA (online softmax attention)
- [x] WASM flash GQA + scalar flash GQA
- [x] WASM I8 SDOT logits + `bn_quant_f16_rows_to_i8` for WASM
- [x] WASM SDOT for TQ1_0, TQ2_0, Q8_0, Q4_K, Q6_K
- [x] WASM `bn_quant_x_to_q8k` (Q8_K super-block quantization)
- [x] AVX2 Q4_K/Q6_K fused matmul (batch prefill)
- [x] VLA guards + SIMD alignment guards in all GQA backends

## Phase 11: GPU Compute Backend (WebGPU) — Done

- [x] `BnGPUBackend` vtable (matvec, matmul, matvec_batch, execute, init_activations)
- [x] GPU dispatch integration in `quant/dispatch.c` (`bn_quant_matvec_gpu`, `bn_quant_matvec_batch_gpu`)
- [x] 22 WGSL matvec shaders for all quant types (I2_S through IQ2_S + F16/F32)
- [x] 9 WGSL forward-pass shaders (rmsnorm, rope, gqa_scores, softmax, gqa_combine, silu_gate, relu2_gate, residual_add, bias_add)
- [x] `bn_model_upload_weights` / `bn_model_release_gpu` for GPU buffer management
- [x] Norm weight + bias + tied embedding GPU upload
- [x] Persistent scratch buffers + batched command encoding
- [x] Single-submit GPU-resident forward pass (one command buffer per token)
- [x] `--webgpu` CLI flag, `make BN_ENABLE_WEBGPU=1`, `make fetch-wgpu`
- [x] GPU validation benchmark (20/20 matvec pass, 3/3 matmul pass)
- [x] wgpu-native vendoring with SHA-256 verification
- See [docs/hull-integration.md](hull-integration.md) for the Hull integration design

### Library API — Done
- [x] SSE chunk formatter (`bn_format_sse_chunk`, `bn_format_sse_done`)
- [x] Logprobs API (`bn_logprobs_compute`)
- [x] Multi-turn chat formatting (`bn_chat_format_messages`)
- [x] Stop strings (`BnStopStrings`)
- [x] Allocator vtable (`BnAllocator`, compatible with Keel's `KlAllocator`)

## Future Work

### Transformer Architecture Redesign

The next major maintainability item is to split `src/transformer.c` into explicit planning and execution layers while preserving CPU correctness and Metal/WebGPU behavior. The goal is not a new math path first; it is to make model-family, quant-format, and backend decisions visible and testable before adding more SoTA model and quant coverage.

Target architecture:

```
GGUF/model load -> model anatomy + tensor roles
quant layer     -> format operations
backend layout  -> uploaded buffers and optional stacked/fused layouts
planner         -> layer/block execution plan
executor        -> CPU / Metal / WebGPU / CUDA
```

- [x] **Step 1: map and freeze behavior** — catalog every architecture branch, quant special case, and backend fast path in `transformer.c`; add lightweight route tests for QKV, FFN, MoE, SSM, KV cache, flash attention, and fallback behavior. See [transformer-behavior-map.md](transformer-behavior-map.md).
- [x] **Step 2: start the kernel capability layer** — replace direct backend capability bit checks with named internal predicates (`bn_transformer_gpu_can_*`) and cover them in `test_transformer`.
- [x] **Step 3: extract layer-shape planning** — introduce internal helpers for `is_attn`, `attn_idx`, `ssm_idx`, `q_dim`, `q_gated`, `q_wide`, per-layer `head_size`, `kv_dim`, `n_kv_heads`, `kv_mul`, and KV mode. These helpers should be synthetic-testable without loading a GGUF.
- [x] **Step 4: define per-block plan structs** — add small internal plan structs for attention, FFN, SSM, MoE, logits, and backend placement. Start with `BnAttentionPlan` carrying layer kind, KV mode, Q/K/V shape, norm/bias flags, and placement.
- [x] **Step 5: split CPU planning from CPU execution** — make `forward_single_layer` consume layer/block plans while keeping CPU math straightforward and reference-quality. Do not optimize or fuse CPU behavior during this step.
- [x] **Step 6: split GPU op emission by block** — move GPU construction out of `forward_gpu` into internal emitters such as `emit_gpu_attention_ops`, `emit_gpu_ffn_ops`, `emit_gpu_ssm_ops`, `emit_gpu_moe_ops`, and `emit_gpu_logits_ops`.
- [x] **Step 7: make fusion rules declarative** — represent QKV split, gate/up fusion, RoPE QK fusion, residual+norm, flash attention, Q4_K split, Q8 split, and Q5_K split as rule checks with explicit required tensor roles, quant format, shape compatibility, activation, and backend cap.
- [x] **Step 8: move backend buffer layout out of model loading** — keep `model.c` responsible for model anatomy and tensor roles; move QKV stacks, gate/up stacks, SSM stacks, fused bias buffers, and backend-specific upload choices into a backend layout layer.
- [x] **Step 9: extract architecture-specific model rules** — move Qwen/Gemma/DeepSeek/Nemotron-style shape, activation, norm, MRoPE, SSM, and shared-expert rules out of the main transformer loop into model-architecture helpers.
- [x] **Step 10: make backend placement explicit** — choose CPU, Metal, WebGPU, future CUDA, or CPU fallback per planned op/block. Fallback must be deterministic and visible in tests/debug output.
- [x] **Step 11: enforce parity gates** — require `make clean && make bitnet`, `make test`, coherence tests for touched GPU backends, and llama.cpp CPU/Metal benchmark comparisons before declaring the redesign complete.

Success criteria: adding a new quant should primarily touch `gguf`, `quant`, backend kernels, and capability registration; adding a backend should primarily implement `BnGPUBackend` and advertised caps; adding a model family should primarily touch model metadata and architecture helpers. `transformer.c` should stop accumulating backend/model/quant cross-product branches except for genuinely new execution primitives.

### Architecture Cohesion Follow-up

The transformer redesign introduced explicit plan structs, quant metadata,
backend-owned model/session state, and backend-private shader lowering. The
largest ownership leaks are gone. The remaining cleanup is mostly about turning
the current GPU command emission into a higher-level graph/value IR, making
fallback reasons first-class, and expanding architecture rules without adding
new cross-product branches.

Target ownership model:

```
GGUF parser       -> raw tensor metadata and bytes
model anatomy     -> architecture, tensor roles, layer/block semantics
model weights     -> immutable CPU-visible weights and CPU-side transforms
quant registry    -> format sizing, layout, CPU kernels, backend capabilities
backend model     -> uploaded buffers, packed/fused layouts, backend-owned weights
backend session   -> activation buffers, KV buffers, per-request backend state
op planner        -> backend-neutral layer/block execution plan
backend lowerer   -> CPU / Metal / WebGPU / CUDA / AVX512 concrete execution
```

- [x] **Step 12: split loaded weights from backend-resident weights** — introduce `BnBackendModel` and `BnBackendSession` so GPU handles, stacked QKV buffers, gate/up stacks, SSM stacks, norm uploads, fused bias buffers, and future CUDA state no longer live inside `BnModel`, `BnLayerWeights`, or `BnQWeight`. `BnModel` should remain shared, immutable, and backend-independent after load.
- [x] **Step 13: make quant formats table-driven** — add a `BnQuantFormatOps` registry that owns data sizing, block geometry, dequant support, CPU matvec/matmul hooks, repack support, native-layout support, split-matvec support, and backend capability registration. Adding a new quant should not require editing unrelated switch forests across `model`, `quant`, `transformer`, and GPU upload code.
- [x] **Step 14: replace concrete GPU shader IDs with backend-neutral op kinds** — define an intermediate operation IR (`MATVEC`, `RMSNORM`, `ROPE`, `ATTENTION`, `FFN`, `FUSED_GATE_UP_ACT`, `LOGITS`, etc.) and let each backend lower it to Metal/WebGPU/CUDA-specific kernels. Keep shader IDs and buffer indices private to backend implementations.
- [x] **Step 15: promote backend layout into a real module** — move inline layout/upload helpers out of `include/backend_layout.h` into `src/backend_layout.c`. This module should choose native vs repacked layouts, stacked tensors, fused-bias buffers, QKV/gate-up/SSM packing, and record deterministic fallback reasons for debug output and tests.
- [x] **Step 16: make model architecture rules pluggable** — replace one-off config flags such as `arch_gemma4` with a `BnModelArchOps` registry for architecture-specific config loading, tensor-role mapping, layer classification, activation/norm rules, MRoPE rules, SSM rules, MoE/shared-expert rules, and future Qwen/Gemma/DeepSeek/Nemotron variants.
- [x] **Step 17: shrink `transformer.c` into orchestration only** — split the remaining implementation into focused modules under `src/transformer/`: `plan.c`, `cpu.c`, `gpu.c`, `gpu_emit.c`, `kv.c`, `logits.c`, and `prefill.c`. The top-level transformer loop should select plans, execute blocks, and handle deterministic fallback, not encode backend/model/quant cross-product logic.
- [x] **Step 18: add architecture boundary tests** — add synthetic tests that prove model load does not allocate backend state, backend upload does not mutate model anatomy, quant registry entries advertise consistent capabilities, and CPU/GPU/CUDA/AVX512 placement decisions are visible before execution. These tests should not require model files.

Success criteria: model-family additions primarily register architecture rules and tensor-role mappings; quant additions primarily register `BnQuantFormatOps` and kernels; backend additions primarily implement layout lowering and execution; `BnModel` remains backend-neutral; fallback decisions are explainable in logs/tests; and `transformer.c` no longer needs direct knowledge of individual quant formats or backend shader IDs.

Current audit for Steps 12-18:

| Step | Current evidence | Remaining gap |
|---|---|---|
| 12 | `BnBackendModel` and `BnBackendSession` own GPU handles, qweight buffers, backend graph state, stacked QKV/gate-up/SSM buffers, norms, biases, and tied embeddings. `BnQWeight`, `BnLayerWeights`, and `BnWeights` no longer expose GPU handles. | CUDA-specific backend state is not implemented yet. |
| 13 | `BnQuantFormatOps` registry centralizes format names, block geometry, sizing, embedded-scale behavior, support status, CPU matvec/batch/matmul capability, generic CPU matvec/matmul hook entrypoints, GPU split capability, backend-neutral GPU split op-code selection, fused gate-up SiLU backend capability, CPU pre-Q8K activation reuse capability, CPU repack capability, GPU native-layout capability, and GPU repack-layout capability. Transformer planning records registry-selected split/native layout decisions instead of per-format Q8/Q5/Q4 flags, and CPU/MoE/GPU-emission paths query quant capabilities instead of hard-coding Q4_0/Q8_0/Q4_K/Q5_K split-shader choices or Q4_K/Q6_K pre-Q8K reuse. | Per-format CPU hook specialization and some backend lowering choices are still partly in dispatch/backend switch logic. |
| 14 | `BnGPUOpKind` gives each emitted GPU op a semantic kind and `BnGPUOpCode` adds a backend-neutral concrete-op layer. `BnGPUOp` no longer exposes `shader`, public headers no longer export `BN_GPU_SHADER_*` IDs or shader mapping helpers, quant registry and transformer planning expose split op codes only, and `src/transformer/gpu_emit.c` is op-code-only. Public graph references now use `BN_GPU_VALUE_*`; current Metal/WebGPU shader IDs, backend activation slot aliases, and op-code-to-shader lowering live in backend-private `src/gpu_shader.h` plus backend `execute` implementations. Tests verify op-code-to-kind behavior and the backend matrix rejects shader IDs in GPU emission. | `src/transformer/gpu_emit.c` still emits the backend command array directly. A future IR pass should model graph values and multi-output ops explicitly before backend command lowering. |
| 15 | `include/backend_layout.h` is declarations-only and `src/backend_layout.c` owns stacked/fused upload decisions plus deterministic fallback reasons covered by tests. | Native/repacked layout selection is still narrow and mostly GPU-upload focused. |
| 16 | `BnModelArchOps` registry covers architecture matching, explicit Qwen/BitNet/Gemma4 family entries, prefixes, activation, Gemma4 shape rules, SSM-layer classification, MoE config helpers, architecture flags, and tensor-role name/scale mapping for attention, SSM, dense FFN, MoE expert, and shared-expert roles. The old `arch_gemma4` config field is removed. | MRoPE/local-attention rules, DeepSeek/Nemotron rules, tokenizer-family rules, and backend placement constraints need fuller registry entries. |
| 17 | Planning moved to `src/transformer/plan.c`; QKV/logits/RMSNorm/attention/SSM/dense-FFN/MoE GPU emission moved to `src/transformer/gpu_emit.c`; GPU-resident forward graph orchestration moved to `src/transformer/gpu.c`; TurboQuant, FP16, and FP32 KV row/write helpers moved to `src/transformer/kv.c`; CPU logits orchestration moved to `src/transformer/logits.c`; batch prefill moved to `src/transformer/prefill.c`; CPU layer execution, attention execution, SSM block execution, dense-FFN block execution, FFN activation, residual add, RoPE application, and GQA backend dispatch moved to `src/transformer/cpu.c`; `transformer.c` now handles token bounds, embedding/RoPE setup, CPU-vs-GPU top-level routing, and logits timing. It is currently 144 lines, down from 2,699 before the split. | `transformer.c` is now near orchestration-only, but `src/transformer/gpu_emit.c` still builds backend command arrays directly instead of producing a higher-level graph-value IR first. |
| 18 | Synthetic tests cover backend model/session ownership, upload-not-mutating CPU weights, quant registry metadata, model-load-without-backend-state, model architecture registry behavior, CPU execution helper boundaries, CPU ISA placement including AVX-512, and Metal/WebGPU/CUDA placement visibility. | There is no CUDA implementation or real CUDA/AVX-512 benchmark matrix beyond advertised placement visibility. |

Latest local code gate before this docs-only cleanup: `make test`, `make clean`
followed by `make bitnet`, `make BN_ENABLE_METAL=1 test_coherence`,
`./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal`,
`make BN_ENABLE_WEBGPU=1 test_gpu_wgpu`, `make bench_llama_compare`, and
`./test/backend_matrix.sh` passed after the shader/buffer contract cleanup.
`make BN_ENABLE_WEBGPU=1 test_gpu_wgpu` compiled the WebGPU backend but skipped
runtime GPU checks on this machine because wgpu-native reported no suitable
adapter. The llama.cpp comparison gate runs three trials and reports median plus
mean. On `qwen2.5-3b-instruct-q4_0.gguf` with 32 tokens and 8 threads, that run
measured median bitnet.c at 39.30 tok/s versus median llama.cpp at 17.59 tok/s
(median ratio 2.2347; mean ratio 2.1997). This short 32-token benchmark remains
noisy and should be treated as a gate/check, not a durable performance claim.

### Backend Expansion Plan

The backend roadmap should follow the architecture cleanup rather than racing ahead of it. CUDA, AVX-512, Metal, WebGPU, WASM SIMD, and scalar CPU should share quant metadata, model-family rules, execution planning, and fallback reporting. Backend-specific code should own only layout lowering, kernel selection, memory residency, and execution.

- [ ] **CUDA backend** — add a `BnGPUBackend` implementation backed by CUDA streams, device buffers, graph capture where useful, and kernels for matvec, batched matvec, RMSNorm, RoPE, attention, FFN, logits, MoE routing, SSM, and KV-cache operations.
- [ ] **CUDA quant kernels** — implement CUDA kernels for the active GGUF formats first (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, `BF16`, `F16`, `F32`), then add IQ and ternary formats once the registry can advertise per-backend capability precisely.
- [ ] **CUDA memory policy** — make backend sessions own activation buffers, KV buffers, temporary reductions, graph scratch, and stream-local state so multiple sessions can share one immutable backend model safely.
- [ ] **AVX-512 backend** — add AVX-512 VNNI/BF16 kernels behind compile-time and runtime detection, with scalar/AVX2 fallback. Target `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, `BF16`, and logits first.
- [ ] **AVX-512 dispatch hygiene** — keep AVX-512 as a quant/backend implementation detail registered through `BnQuantFormatOps`, not as new architecture branches inside `transformer.c`.
- [ ] **Backend parity matrix** — maintain a test/benchmark matrix for scalar, NEON, AVX2, AVX-512, WASM SIMD, Metal, WebGPU, and CUDA showing which op kinds and quant formats are native, repacked, split, fused, or CPU fallback.
- [ ] **llama.cpp comparison gate** — for every new backend milestone, compare prompt processing and token generation against equivalent llama.cpp runs on the same model, quant, thread count, context length, and GPU/CPU placement.

### Quantization Coverage Plan

Quant support should be added through the quant registry and backend capability table. A format is not considered fully supported until sizing, dequant, CPU matvec, CPU batch path where relevant, backend upload/layout, native or repacked backend kernels, tests, and llama.cpp comparison coverage are all accounted for.

- [ ] **Complete remaining legacy GGUF formats** — add or finish `Q5_0`, `Q5_1`, and any still-missing legacy variants with scalar, NEON/AVX2/WASM, AVX-512 where applicable, and backend capability registration.
- [ ] **Finish IQ-family parity** — verify and fill gaps for `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, and `IQ4_XS` across CPU SIMD, WASM SIMD, and GPU fallback/native paths.
- [ ] **Add modern low-bit GPU-friendly formats** — evaluate `MXFP4`/`NVFP4`-style block floating formats, `FP8` (`E4M3`/`E5M2`) where GGUF/tooling support exists, and other SoTA OSS deployment formats before committing to kernels.
- [ ] **Native-layout kernels before repack proliferation** — prefer zero-copy native GGUF layouts when they are competitive, especially for `Q4_0`, `Q8_0`, and k-quants, and use repacking only when it gives a measured win.
- [ ] **Quant capability tests** — add synthetic tests that validate block size, data size, scale layout, native/repacked support, split support, backend support, and fallback reasons for every registered quant.
- [ ] **Quant benchmark fixtures** — keep a small benchmark suite that runs the same prompt through bitnet.c and llama.cpp for representative dense, MoE, and hybrid models across common quants.

### Model Family Support Plan

Model-family support should be data-driven through model architecture ops, tensor-role mapping, and planner-visible capabilities. The goal is dedicated support for current OSS model families without turning `model.c` or `transformer.c` into family-specific switchboards.

- [ ] **Qwen 3.5 / Qwen 3.6** — register architecture rules for config loading, tokenizer assumptions, GQA layout, RoPE/MRoPE behavior, activation, norm placement, MoE/shared-expert variants, and backend placement constraints.
- [ ] **Gemma 4** — finish pluggable Gemma-family rules for shape derivation, shared attention value/key behavior, local/global attention, altup-style blocks if present, and family-specific tensor naming.
- [ ] **DeepSeek v4 Flash** — add architecture rules for MLA/MoE-style routing if present in GGUF exports, shared experts, routed experts, activation/norm behavior, and memory-aware expert loading.
- [ ] **Nemotron 3 Super** — add architecture rules for tensor naming, block layout, activation/norm choices, attention variants, and quant/backend restrictions once public GGUF conventions are stable.
- [ ] **Model-family fixtures** — add synthetic config tests for every architecture rule and at least one real GGUF smoke/coherence test per supported family when model files are available.
- [ ] **Unsupported-feature reporting** — fail early with explicit messages when a model requires an op kind, tensor role, quant format, or backend capability that bitnet.c does not yet implement.

### Cross-Product Support Matrix

Do not add model, quant, or backend support as isolated one-offs. Each milestone should update a visible matrix that says which model families, quant formats, and backends are native, repacked, partially supported, or CPU fallback.

- [ ] **Model families x backends** — track Qwen 3.5, Qwen 3.6, Gemma 4, DeepSeek v4 Flash, Nemotron 3 Super, existing Qwen2/Qwen3, Llama-style dense models, BitNet ternary models, and MoE/hybrid families across scalar, NEON, AVX2, AVX-512, WASM SIMD, Metal, WebGPU, and CUDA.
- [ ] **Quant formats x backends** — track F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ-family formats, TQ1_0/TQ2_0/I2_S, and evaluated FP8/MXFP4/NVFP4-style formats across every backend.
- [ ] **Model families x quants** — record the recommended and tested quants for each family, including dense, MoE, SSM, MLA, local/global attention, and ternary/BitNet-style variants.
- [ ] **Fallback reasons** — every unsupported matrix cell should point to a concrete missing capability: tensor role, op kind, quant kernel, backend memory policy, architecture rule, tokenizer behavior, or validation fixture.
- [ ] **Benchmark parity rows** — for representative cells, keep bitnet.c vs llama.cpp prompt-processing and generation numbers with the same model, quant, thread count, context length, batch size, and GPU placement.

### Next Architecture Cleanup

The transformer file split is complete enough that the next work should focus on
the remaining leaks between planning, graph values, backend lowering, and
capability reporting.

- [ ] **Introduce a higher-level graph-value IR** — `gpu_emit.c` still builds the backend command array directly. Add an internal graph layer that models values, aliases, multi-output ops, and fallback boundaries before lowering to `BnGPUOp`.
- [ ] **Make fallback reasons first-class** — store the reason for CPU fallback or reduced GPU placement in plan/layout structs and expose it in tests/debug logs.
- [ ] **Finish native/repacked layout policy** — make backend layout choose native GGUF, repacked, split, stacked, or fused layouts through a table-driven policy instead of backend-local ad hoc decisions.
- [ ] **Move remaining backend switch logic behind registries** — quant and architecture registries should own capability declarations; backend lowerers should consume them without adding model-family branches.
- [ ] **Expand architecture registry coverage** — add fuller rules for MRoPE, local attention, tokenizer-family assumptions, DeepSeek/Nemotron-style tensor roles, and backend placement constraints.
- [ ] **Keep `transformer.c` orchestration-only** — new model, quant, and backend support should not grow the top-level transformer loop.

### Cohesion and Coupling Debt

The latest architecture review found that the largest ownership leaks are fixed,
but several internal seams still violate high-cohesion/low-coupling goals. These
items should be addressed before adding substantial new backend, model-family,
or quant-format surface area.

- [x] **Split `transformer_internal.h` into narrower internal boundaries** — it still includes model, quant, math, SIMD, GPU planning, CPU execution, KV/TurboQuant helpers, logits, plan structs, and RMSNorm backend symbols. Split into focused headers such as `transformer_plan_internal.h`, `transformer_kv_internal.h`, `transformer_cpu_internal.h`, and a tiny RMSNorm backend header.
- [x] **Split `quant_internal.h` by quant concern and backend** — it still centralizes all context structs, every scalar/NEON/AVX2/WASM kernel prototype, dispatcher helpers, and inline unpack helpers. Split into `quant_ctx.h`, `quant_kernels_scalar.h`, `quant_kernels_neon.h`, `quant_kernels_avx2.h`, `quant_kernels_wasm.h`, and `kquant_helpers.h` or equivalent.
- [x] **Hide model runtime/backend strategy behind opaque subobjects** — `BnModel` is smaller but still exposes mapped file lifetime, thread pool, weight arena, MoE I/O, expert fd, backend state, and TurboQuant state. Move these into opaque `BnModelRuntime` / `BnModelIO` style subobjects or make `BnModel` opaque to public consumers.
- [x] **Move CPU prepared-layout generation out of `model.c`** — Q4/Q8 repacking and related prepared-weight registration are backend/quant layout concerns, not canonical GGUF model loading. Move them under backend layout or quant preparation.
- [x] **Move MoE expert map loading out of `model.c`** — expert tensor offset/stride derivation and fused gate/up expert mapping are MoE loader concerns. Keep `model.c` focused on architecture config, tensor role lookup, and immutable weight references.
- [x] **Split `BnLayerWeights` into tagged substructures** — the current layer struct carries attention, SSM, dense FFN, MoE, and shared-expert fields at once. Introduce `BnAttentionWeights`, `BnSSMWeights`, `BnFFNWeights`, and `BnMoEWeights` or a similar tagged layout to reduce accidental cross-family coupling.
- [x] **Decompose `src/moe.c` by responsibility** — routing, expert I/O, LRU cache, madvise/pread, per-token execution, prefill batching, shared experts, and stats are still in one large module. Split into `moe_route`, `moe_io`, `moe_cache`, `moe_execute`, and `moe_prefill`.
- [x] **Split backend model and backend session headers** — implementation is split, but `backend_model.h` still declares both immutable backend model resources and per-session graph state APIs. Introduce `backend_session.h` so session graph users do not couple to model handle storage.
- [x] **Move backend quant shader-op mapping out of public inline helpers** — `backend_quant.h` maps tensor types directly to GPU shader IR op codes. Keep public/backend-quant helpers semantic and move concrete op-code selection into GPU emission or backend lowering.

### GPU Optimization

- [x] Improve GPU forward-pass precision where backend reductions diverge too early.
- [ ] Add FP16 KV cache support on GPU.
- [ ] Broaden GPU-native SSM and MoE execution to reduce CPU fallback.
- [ ] Finish native-layout Q4_0 and related low-bit matvec kernels.
- [ ] Add a graph-value IR before backend lowering so Metal/WebGPU/CUDA can share more planning logic.

### Extended Model Support
- [ ] LoRA adapter loading
- [ ] Dedicated Qwen 3.5 / Qwen 3.6 support through `BnModelArchOps`
- [ ] Dedicated Gemma 4 support through `BnModelArchOps`
- [ ] Dedicated DeepSeek v4 Flash support through `BnModelArchOps`
- [ ] Dedicated Nemotron 3 Super support through `BnModelArchOps`

### Developer Experience
- [x] Interactive mode (--chat REPL with sliding window)
- [ ] Token probability output mode (for debugging/research)
- [ ] JSON output mode (structured generation metadata)
- [ ] Model info dump command (`--info` to print config without inference)

### SIMD Backends
- [ ] AVX-512 VNNI — native `vpdpbusd`, 512-bit vectors (Ice Lake+, Zen 4+)

### Platform Expansion
- [ ] Windows support (VirtualAlloc instead of mmap)
- [ ] iOS/Android builds (static library)
- [ ] Python bindings (ctypes or cffi wrapper)
- [ ] Node.js native addon

## Non-Goals

- Full GGUF compatibility (only supported types listed above)
- Training or fine-tuning
- Multi-GPU / distributed inference
- Replacing llama.cpp for general LLM inference
- HTTP server (that's Hull's job)
