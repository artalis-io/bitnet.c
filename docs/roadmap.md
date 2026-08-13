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
adapter.

The current strict Metal acceptance gate is `test/compare_llama_topk.py` against
`llama-server -fa on -np 1`. It requires top-1/top-k coherence and, when
`--benchmark` is used, defaults `--min-throughput-ratio` to `1.0`. The latest
local Qwen2.5 sample matched top-1 on `8/8` prompts with mean top-10 overlap
`9.62`, but measured bitnet.c at 96.55 tok/s versus llama.cpp at 117.71 tok/s
(ratio 0.820), so this gate is intentionally still not satisfied. A later
self-contained gate rerun measured bitnet.c at 96.83 tok/s versus llama.cpp at
97.39 tok/s (ratio 0.994), still below the default `1.0` threshold and not a
parity claim because the llama.cpp baseline was much lower than prior samples. The older
32-token `make bench_llama_compare` result, median bitnet.c 39.30 tok/s versus
llama.cpp 17.59 tok/s, is historical and should not be used as the active
parity claim.

Use `make BN_ENABLE_METAL=1 bench_llama_topk_server` for the self-contained
gate: it starts `llama-server` with `-fa on -np 1`, runs the same comparator,
and shuts the server down afterwards. The default target asks the helper to
choose a free localhost port, uses `--bench-runs 3`, and compares median
throughput. The latest target-level run measured bitnet.c samples
`[94.92, 95.09, 95.37]` tok/s and llama.cpp samples
`[115.96, 116.45, 115.61]` tok/s, giving medians 95.09 and 115.96
respectively (ratio 0.820), so the median gate is also intentionally still not
satisfied.

A current thermally constrained self-contained rerun after the prepared-layout,
fused gate/up, and Q6_K activation-load improvements still matched top-1 on
8/8 prompts with mean top-10 overlap 9.62. Three 128-token samples measured
bitnet.c `[53.02, 54.57, 55.12]` tok/s and llama.cpp
`[57.92, 58.87, 54.20]` tok/s, giving medians 54.57 and 57.92 and a ratio of
0.942. This is materially closer than the earlier 0.820 gate but remains below
the required 1.0 parity threshold.
An explicit paired traversal of prepared Q4_0 gate/up weights was rejected.
It attempted to share Q8 activation loads while retaining independent gate and
up accumulation order, but a candidate/control end-to-end pair measured only
48.38 versus 47.38 tok/s and direct-kernel samples were dominated by a global
35-40% frequency swing also visible in unrelated `up` and `down` rows. The
normalized fused-kernel cost did not improve enough to justify the added
register pressure, so the original two traversals were restored. This was a
Metal-private shader experiment; quant and model policy were unchanged.
The prepared Q4_0/Q8 fused gate/up pipeline now uses a pipeline-scoped
16-row/128-thread launch instead of its former 32-row/256-thread geometry. The
matching shader stride changed with that concrete pipeline contract; the
type-wide Q4_0 tile helper and quant capability declarations are unchanged.
The focused Qwen2.5 strict comparison retained its existing 3/16 generated-ID
prefix. Candidate/control/confirmation 128-token runs measured 53.48, 48.96,
and 52.39 tok/s, a conservative 7.0% adjacent improvement. This is
Metal-private scheduling and does not introduce model-family selection into
the backend. A subsequent authoritative self-contained three-run gate measured
bitnet.c `[55.75, 55.53, 54.34]` tok/s and llama.cpp
`[61.51, 61.13, 61.22]` tok/s, giving medians 55.53 and 61.22 and a ratio of
0.907. Top-1 remained 8/8 with mean top-10 overlap 9.62. Dense Metal parity is
therefore still open despite the retained scheduling improvement.
Metal specialized native-quant decode is now opt-in rather than the default.
The existing quant registry still declares which formats and shapes support
the behavior, and Metal lowering still owns the concrete Q8_K activation
pipelines; only backend feature policy changed. The canonical
`--metal-specialized-native-quant` diagnostic remains available. With the
float-input default, the focused Qwen2.5 comparison improves from the prior
3/16 generated-ID prefix to 16/16. The authoritative dense gate measures
bitnet.c `[57.22, 57.32, 57.41]` tok/s versus llama.cpp
`[60.41, 58.79, 61.28]`, medians 57.32 and 60.41, ratio 0.949, with top-1
8/8 and mean top-10 overlap 9.62. Dense Metal parity remains open but is closer
than the preceding 0.907 gate.
The regular Metal Q4_0 repack now retains each GGUF scale as FP16 instead of
expanding it to FP32, while keeping the existing sequential nibble plane and
32-bit alignment. This reduces backend-resident traffic from 20 to 18 bytes
per block without changing the represented scale value. The layout change is
confined to Metal upload ownership and its six private consumers; quant
capabilities, model-family policy, transformer planning, and WebGPU layout are
unchanged. The focused strict comparison remains 16/16 generated token IDs and
full Metal coherence passes 20/20. Under a hotter confirmation run, the direct
kernel benchmark improved generic fused gate/up from 509.7 to 484.0 us (5.0%)
and FFN down from 479.5 to 463.5 us (3.3%), while unrelated Q6_K logits slowed,
supporting a Q4_0-specific bandwidth gain rather than a frequency artifact.
The managed three-run dense gate passed top-1 on 8/8 prompts with mean top-10
overlap 9.62 and measured bitnet.c at 52.21 tok/s versus llama.cpp build 9950
at 51.35 tok/s, ratio 1.017. Absolute rates were thermally depressed, so this
closes that matched gate but should be reconfirmed in the broader model matrix.
Reducing that exact prepared gate/up pipeline again to an 8-row/64-thread
launch was rejected. It preserved the generated continuation but measured
55.15 tok/s versus an adjacent retained 16-row/128-thread control at
56.13 tok/s, a 1.7% regression. The type-wide tile helper remained untouched.
A 24-row/192-thread intermediate point also preserved 16/16 generated IDs but
measured 57.98 tok/s versus the immediately preceding retained 16-row control
at 58.17 tok/s. It was removed; 16 rows remains the best measured concrete
pipeline geometry without changing shared quant tile policy.
Applying a 16-row/128-thread launch to the separate prepared Q4_0/Q8 matvec
pipeline was neutral and removed: candidate and adjacent restored 32-row
control measured 55.20 and 55.09 tok/s. This concrete-pipeline result was not
generalized into quant metadata or shared tile policy.

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
- [ ] **Metal flash-attention parity** — evolve the current bounded short-context Metal flash shader into a tiled/chunked implementation that keeps enough parallel work for the 128-token server gate and matches llama.cpp `-fa on` style f16-KV throughput behavior. Lowering `BN_GPU_FLASH_MIN_KV` is not sufficient until the flash path beats the non-flash scores/softmax/combine path at the acceptance length.
- [ ] **Metal FFN matvec parity** — redesign the Metal Q4_0 FFN hot path around the measured bottlenecks: fused gate/up, FFN down, and Q6_K logits. The current first-three-layer Q4_0 x Q8 policy is the best tested setting; `--small-dense-native-quant-disable-ffn-down` measured 94.90 tok/s versus llama.cpp 115.72 tok/s, and combining it with `--small-dense-native-quant-disable-gateup` measured 95.76 tok/s versus llama.cpp 115.40 tok/s, so native-FFN policy toggles are not enough. The next concrete kernel direction is a row-grouped Q4_0 matvec/gateup design that reuses each activation slice across multiple output rows, matching the structure of llama.cpp's `mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>` instead of the current one-output-row-per-8-lane-group design. A first two-row Q4_0 x Q8 matvec diagnostic was rejected because it regressed the direct FFN `up`/`down` rows and measured only 94.50 tok/s in the llama-server gate.
  A complete 18-byte native-GGUF storage diagnostic was also rejected after
  exercising regular matvec, stacked QKV, and fused gate/up together. The
  original 8-lane native shader measured 49.2 tok/s versus the 49.1 tok/s
  repacked baseline in the 32-token kernel benchmark. Matching llama.cpp's
  half-block-per-lane, four-row decomposition reduced the same benchmark to
  47.5 tok/s, with `up`/`down` at 379.5/390.4 us and fused gate/up at
  632.9 us. The experiment was removed rather than leaving a slower alternate
  storage contract in the backend.
  Sharing each Q4_0 fused gate/up activation-block sum across the four rows in
  a Metal SIMD group was exact but not faster. Two experimental runs measured
  51.34 and 51.10 tok/s; the immediately restored per-row control reached
  52.31 tok/s. The shuffle-based sharing was removed, and the earlier 47.32
  tok/s control is treated as run-state noise rather than a valid speedup.
  Full two-row reuse in the generic float-input fused gate/up kernel is retained.
  It computes two gate/up rows per eight-lane group, halves the threadgroup
  count, and reuses each activation block across both rows without changing the
  per-row dot accumulation order. Two-row runs measured 53.45 and 53.03 tok/s
  versus adjacent restored one-row controls at 52.31 and 52.80 tok/s; median
  throughput improved from 52.56 to 53.24 tok/s (1.3%) with identical observed
  128-token output.
  Extending the same kernel to four rows per eight-lane group regressed to
  48.98 tok/s and changed the continuation near the end of the 128-token run.
  The four-row geometry was removed; two rows is the accepted reuse/register
  pressure point on M1.
  Keeping the same two-row arithmetic but reducing the launch from 256 threads
  and 64 rows to 128 threads and 32 rows was unstable and slower overall. Runs
  measured 53.95 and 49.45 tok/s, with the slower run changing the late
  continuation; the restored 256-thread control measured 52.38 tok/s with the
  accepted output. The 128-thread geometry was removed.
  The intermediate 192-thread/48-row geometry was exact and stable but neutral:
  two runs measured 53.40 and 52.61 tok/s, overlapping the retained 256-thread
  distribution. It was removed rather than adding another launch contract
  without a robust gain.
  Explicitly loading each activation `float4` once and retaining all eight
  vectors in registers across both rows and gate/up dots preserved the observed
  continuation but regressed to 50.98 tok/s. The register-array implementation
  was removed; the helper-based two-row kernel lets the Metal compiler/cache
  handle repeated activation references more efficiently.
  Applying two-row reuse to the separate generic Q4_0 matvec was also rejected.
  An initial type-wide tile change incorrectly affected split shaders and
  produced invalid tokens; after scoping geometry to the concrete Q4_0 matvec
  pipeline, the kernel measured 52.94 tok/s, offered no gain over the retained
  dense path, and changed the late continuation. All matvec shader and geometry
  changes were removed. This reinforces that tile geometry is a backend shader
  contract, not a quant-format-wide policy.
  A four-row implementation for the separate opt-in prepared-F32 layout was
  exact but neutral as well. Paired 128-token runs measured 52.04 and 51.61
  tok/s versus adjacent default controls at 51.49 and 52.37 tok/s; the prepared
  average was slightly lower. The shader and its pipeline-specific 128-row
  launch geometry were fully removed, leaving the existing 32-row prepared
  diagnostic unchanged.
  The production repacked-Q4_0/Q8 fused gate/up shader now evaluates gate and
  up in one paired helper, loading each activation `char4` once while retaining
  independent accumulators and the original dot/add order. A low-thermal
  `--prefill-iters 0 --toks 0 --iters 200` alternating sequence measured the
  paired helper at 325.0 us, the restored original at 348.4 us, and the paired
  confirmation at 313.0 us. The conservative adjacent improvement is 6.7%.
  The focused Qwen2.5 output kept its existing 3/16 llama.cpp token prefix.
  Whole-model decode was thermally constrained to roughly 55 tok/s during the
  earlier comparison, so this is not yet evidence that the dense end-to-end
  parity gap is closed.
  Explicitly retaining each packed gate/up `uint` across both nibble halves
  was rejected. It reduced repeated source expressions but increased live
  register pressure, and the fused microbenchmark measured 456.3 us in a
  different run state. The compact paired helper was restored and later won
  the controlled alternating comparison above.
  Keeping gate/up as a `float2` accumulator through the block loop and SIMD
  reduction was rejected too. Component-wise arithmetic order was unchanged,
  but the fused row regressed to 488.6 us; separate scalar accumulators were
  restored for better compiler register scheduling.
  Replacing each four-value floating dot with four signed integer products and
  a single FP32 conversion was neutral. The fused microbenchmark measured
  336.2 us versus an adjacent 337.2 us floating-dot control, only 0.3% and
  below run variance. The integer helper was removed; the existing floating
  dot remains the simpler compiler contract.
  The backend-private four-row prepared Q4_0 layout now stores native nibble
  bytes. Its Metal upload path previously XORed every byte with `0x88`, while
  all five private consumers immediately XORed the bytes back before the
  unchanged `nibble - 8` decode. Removing both sides preserves the decoded
  values and eliminates one `uchar4` XOR per fragment. The FFN-down
  microbenchmark measured 330.4 us, an adjacent transformed-layout control at
  333.7 us, and a native-byte confirmation at 319.9 us; the conservative
  adjacent improvement is 4.1%. The focused Qwen2.5 output retained its
  existing 3/16 llama.cpp token prefix. This changes only a Metal-resident
  packed-layout contract and its matching shaders; GGUF quant semantics and
  model-family policy are unchanged.
  Sharing each prepared-Q8 activation `char4` across the four output-row
  subgroups with `simd_shuffle` was rejected. Although it reduced activation
  reads fourfold without changing row arithmetic, FFN down measured 332.2 us
  versus an adjacent direct-load control at 293.4 us, a 13.2% regression. The
  activation cache already serves these shared reads more cheaply than the
  added shuffle/extraction sequence, so direct loads were restored.
  Retaining the `0x88` upload transform and decoding its signed nibbles with
  arithmetic shifts was also rejected. It removed XOR-back, masks, and
  subtracts, but FFN down measured 330.7 us versus the adjacent accepted
  native-byte control at 293.4 us. Native-byte storage with vector
  mask/subtract decoding was restored across all five private consumers.
- [ ] **Metal Q6_K x Q8_K diagnostics** — the Q8_K activation quantizer is now parallelized, and `bench_kernels --metal` now reports the quantized tied-embedding logits row through GPU-resident weights. On Qwen2.5 Q4_0, default Metal Q6_K logits measured 1786.4 us/call, the older scalar `--metal-specialized-native-quant` shader measured 11898.2 us/call, and the vectorized Q6_K x Q8_K shader reduced that to 5178.5 us/call for the same 151936 x 2048 matrix. The opt-in path passes top-k coherence but only reaches 72.04 tok/s versus llama.cpp at 117.20 tok/s, so it remains diagnostic-only until it beats the default Q6_K logits path.
  The default Metal Q6_K shader now loads its four contiguous activation
  fragments as `float4` values while preserving the existing four-row reuse
  and component-wise accumulation order. A low-thermal
  `--prefill-iters 0 --toks 0 --iters 200` candidate/control/confirmation
  sequence measured 2497.4, 2634.1, and 2467.1 us/call respectively, a 5.2%
  conservative adjacent improvement. The focused Qwen2.5 output retained its
  existing 3/16 llama.cpp token prefix. This is Metal shader load scheduling;
  it does not change quant capabilities or model-family policy.
  Explicit `uchar4` loads for each row's Q6 low/high-bit fragments were
  rejected. Candidate runs measured 2529.8 and 2579.4 us/call; the intervening
  2740.4 us scalar-index control coincided with a broad slowdown in unrelated
  rows, while the accepted activation-load shader had already confirmed at
  2467.1 us. The weight-vector result is therefore thermally confounded and
  slower than the strongest valid control, so scalar byte indexing was
  restored.
  An eight-row/64-tile diagnostic appeared to reduce logits from an adjacent
  2792.4 us control to 1383.3 and 1423.9 us, but full coherence exposed a
  3.145 max error. The Q6 type-wide Metal tile helper also dispatches a
  separate specialized four-row pipeline; changing that shared helper to 64
  skipped half of the specialized pipeline's rows. Six-row and split-`float4`
  accumulator follow-ups reproduced the same error because the dispatch
  mismatch, not register pressure, was the cause. All changes were removed.
  Future row-reuse work must attach geometry to the selected concrete Metal
  pipeline rather than infer it from the quant type alone.
  The corrected experiment then targeted the selected specialized
  Q6_K/Q8_K pipeline directly and attached its 64-row launch only when that
  exact pipeline object was selected. Full coherence passed 20/20, proving the
  concrete contract was correct. Under a shorter low-drift comparison,
  however, eight-row logits measured 1777.7 us versus an adjacent four-row
  control at 1685.0 us, a 5.5% regression. The specialized shader and
  pipeline-scoped geometry change were removed; four-row reuse remains the
  accepted register-pressure point.
  The now-default float-input Q6_K pipeline was tested separately at
  16 rows/128 threads, with its shader and concrete pipeline launch changed
  together while opt-in specialized pipelines retained their own geometry.
  Strict Qwen2.5 parity passed 16/16, but throughput measured 57.90 tok/s
  versus the recent retained 32-row control at 58.17 tok/s. The candidate was
  removed; this was backend scheduling, not a quant or model-policy change.
  Matching llama.cpp b9950's two-row-per-SIMD-group Q6_K reuse was also tested
  on the default float-input pipeline with eight groups and a 16-row tile. It
  preserved 16/16 strict IDs but measured 57.06 tok/s versus the retained
  four-row path around 58.17 tok/s. Four-row activation reuse remains faster
  on M1 Max; the two-row shader and pipeline-scoped tile were removed.
- [ ] **Metal greedy logits readback reduction** — lower priority: lightweight profiling shows readback is effectively 0.0ms on the current M1 Max gate, but a GPU-side argmax/top-k path may still be useful on discrete-memory backends. Keep full-logit readback for logprobs, top-logit comparison, repetition penalties, and non-greedy sampling.

### Qwen 3 / Qwen 3.5 Backend Improvement Plan

The current model-family gate is top-logit coherence plus generation throughput
against `llama-server -fa on -np 1` with the same model, context, prompt, and
CPU/GPU placement. The local fixture set covers Qwen2.5 dense, Qwen3 dense,
Qwen3 MoE, Qwen3.5 dense, Qwen3.5 MoE, Qwen3.6 dense and sparse, and Gemma4
dense and sparse fixtures.

Latest local measurements:

| Model | Backend | Coherence | bitnet.c tok/s | llama.cpp tok/s | Ratio |
|---|---|---:|---:|---:|---:|
| `Qwen3-0.6B-Q8_0` | ARM NEON / CPU | 7/8 top-1, mean top-10 9.38 | 92.79 | 113.60 | 0.817 |
| `Qwen3-0.6B-Q8_0` | Metal | 8/8 top-1, mean top-10 10.00 | 209.16 | 174.08 | 1.202 |
| `Qwen3.5-9B-Q4_K_M` | ARM NEON / CPU | 8/8 top-1, mean top-10 9.88 | 17.34 | 18.14 | 0.956 |
| `Qwen3.5-9B-Q4_K_M` | Metal | 64/64 generated IDs across 8 prompts | 17.40 | 16.34 +/- 0.14 | 1.065 |
| `Qwen3-30B-A3B-Q4_K_M` | ARM NEON / CPU | 8/8 top-1, mean top-10 9.50 | 2.84 | 5.62 | 0.505 |
| `Qwen3-30B-A3B-Q4_K_M` | Metal | 8/8 top-1, mean top-10 9.38; CPU coherence 14 pass, 0 fail | 40.22 | 43.87 | 0.917 |
| `Qwen3.5-35B-A3B-Q4_K_M` | ARM NEON / CPU | 8/8 top-1, mean top-10 9.62 | 3.91 | 4.34 | 0.901 |
| `Qwen3.5-35B-A3B-Q4_K_M` | Metal | 32/32 generated IDs on focused prompt | 31.06 | 30.53 +/- 0.57 | 1.017 |

Current eight-thread CPU decode diagnostics against `llama-bench` (2026-08-08,
Apple M1 Max, prompt processing disabled, matched generation length) show that
the older CPU throughput rows above are no longer representative of the current
tree:

| Model | bitnet.c median tok/s | llama-bench tok/s | Ratio |
|---|---:|---:|---:|
| `Qwen3-0.6B-Q8_0` (`tg64`, 5 runs) | 136.08 | 134.50 | 1.012 |
| `Qwen3.5-9B-Q4_K_M` (`tg32`, 3 runs) | 16.60 | 11.71 | 1.418 |
| `Qwen3-30B-A3B-Q4_K_M` (`tg128`, 1 long run) | 21.93 | 14.53 | 1.509 |
| `Qwen3.5-35B-A3B-Q4_K_M` (`tg256`, 1 long run) | 20.22 | 9.32 | 2.170 |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` (`tg128`, 1 long run) | 8.42 | 8.11 | 1.038 |

An adjacent 2026-08-09 rerun after aligning the quant-owned ARM Q8_0 paired-lane
reduction measured `Qwen3-0.6B-Q8_0` at 138.01 tok/s versus llama.cpp build
9950 at 135.03 +/- 2.20 tok/s for `tg64`. The same process sequence measured
`Qwen3.6-35B-A3B-UD-Q4_K_M` at 29.06 tok/s versus 8.95 +/- 2.17 tok/s for
`tg128`; the sparse result remains strongly residency-sensitive and is recorded
as throughput evidence rather than a stable replacement baseline.

Matched scalar-versus-NEON spot checks from the same tree make the ARM SIMD
contribution explicit:

| Model | Tokens | scalar tok/s | ARM NEON tok/s | NEON/scalar | llama.cpp CPU tok/s |
|---|---:|---:|---:|---:|---:|
| `Qwen2.5-3B-Instruct-Q4_0` | 32 | 11.87 | 41.94 | 3.53x | 41.24 +/- 3.58 |
| `Qwen3-0.6B-Q8_0` | 32 | 16.95 | 126.23 | 7.45x | 110.77 +/- 11.41 |
| `Qwen3.5-9B-Q4_K_M` | 32 | 4.00 | 13.65 | 3.41x | 9.71 |
| `Qwen3-30B-A3B-Q4_K_M` | 16 | 4.67 | 10.57 | 2.26x | 6.75 +/- 2.69 |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | 8 | 2.58 | 6.58 | 2.55x | 4.44 +/- 1.48 |
| `Gemma4-E4B-Q4_0` | 16 | 6.62 | 26.22 | 3.96x | 21.59 +/- 3.40 |

An adjacent current-tree refresh on 2026-08-09 used separately compiled
`bitnet_scalar` and ARM NEON binaries with eight threads. The bitnet.c and
llama.cpp columns use the same generation length; sparse Qwen3.6 was rerun
after residency warmup and remains a short, high-variance throughput check:

| Model | Tokens | scalar tok/s | ARM NEON tok/s | NEON/scalar | llama.cpp CPU tok/s | NEON/llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-3B-Instruct-Q4_0` | 32 | 11.69 | 46.71 | 4.00x | 42.43 +/- 1.36 | 1.101 |
| `Qwen3-0.6B-Q8_0` | 32 | 16.89 | 119.61 | 7.08x | 112.52 +/- 1.81 | 1.063 |
| `Qwen3.5-9B-Q4_K_M` | 16 | 3.51 | 13.36 | 3.81x | 9.14 +/- 0.12 | 1.462 |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | 8 | 3.94 | 19.26 | 4.89x | 4.44 +/- 1.52 | 4.338 |
| `Gemma4-E4B-Q4_0` | 8 | 7.14 | 31.57 | 4.42x | 26.67 +/- 1.66 | 1.184 |

The dense rows establish current ARM NEON speed parity at these measured
lengths. The sparse Qwen3.6 row is not a replacement for the retained longer
gate because mmap residency dominates short runs. Scalar and NEON generated
the same text in every paired run, so the SIMD speedup did not change the
sampled trajectory on these focused prompts.

An isolated 2026-08-10 refresh after matching the quant-owned prepared Q4_0
ARM accumulation order to llama.cpp measured Qwen2.5 Q4_0 `tg64` at 48.45
tok/s on NEON and 10.80 tok/s scalar, versus llama.cpp build 9950 at
40.70 +/- 2.69 tok/s. Qwen3.5 9B Q4_K `tg32` measured 12.91 tok/s on NEON and
3.02 tok/s scalar, versus llama.cpp at 9.23 +/- 0.30 tok/s. Focused three-token
checks matched llama.cpp sampled IDs on both CPU implementations for both dense
models, and Qwen3.6 35B-A3B matched 3/3 on NEON. The conservative Metal build
also matched 3/3 IDs for dense Qwen2.5 and sparse Gemma4 E26B, while the
model-independent Q4_0 x Q8 and routed-Q4 graph tests remained bit-exact. A
subsequent sustained Metal sample was discarded because the full CLI reported
no Metal device and explicitly fell back to CPU even though the standalone
backend test still discovered the M1 Max; CPU fallback throughput must not be
reported as Metal evidence. The failure reproduced only for the redirected
launch used by that sample. Sequential unredirected runs showed explicit Metal
device, weight-upload, and forward-ready logs: dense Qwen2.5 `tg64` measured
58.51 tok/s versus llama.cpp Metal at 62.58 +/- 0.49 tok/s (0.935x), while
sparse Gemma4 E26B `tg8` measured 23.04 tok/s versus 41.57 +/- 8.90 tok/s.
These are valid current-tree measurements, but neither proves Metal speed
parity. The sparse profile attributed about 406 ms of eight-token MoE time to
the conservative CPU fallback; the resident routed capability remains
withdrawn because its end-to-end sampled IDs are incorrect.

A later 2026-08-10 Metal requalification restored explicit M1 Max device
discovery and fixed two backend/source-layout residency defects. Metal now
reports `recommendedMaxWorkingSetSize` and `currentAllocatedSize` through the
generic backend memory contract, so `model_gpu` rejects oversized optional
resident layouts before allocation. Resident expert borrowing also requires
an actual stable mmap projection base; forced-pread experts are copied and
charged at their real quantized byte size instead of borrowing recyclable CPU
cache slots at handle-only cost. With `--pread --gpu-cache-mb 1024`, sparse
Gemma4 keeps the qualified Metal backend alive instead of failing upload and
silently falling back wholesale to CPU. That conservative composition is
fragmented and measures only `6.25 tok/s`. The CPU-route/mmap-resident Metal
diagnostic reaches `13.73 tok/s` at `tg8` but retains only a `2/8` sampled-ID
prefix, so it remains unpromoted. The adjacent dense Qwen2.5 Metal eight-token
control remains healthy at `98.16 tok/s`. These decisions depend only on
backend memory capability and source-buffer lifetime, not quant-format or
model-family selectors.

A backend-only CPU-route/resident-expert composition is now available as an
opt-in diagnostic with `BN_GPU_DISABLE_ROUTED_MOE_DECODE=1` and
`BN_METAL_ENABLE_CPU_ROUTE_RESIDENT_MOE=1`. MoE continues to own routing,
quant eligibility remains in `backend_quant`, and Metal advertises only the
resident execution capability. Policy tests and the backend architecture
matrix pass, while the production default remains unchanged. Its first strict
wrapper run produced a 4/4 sampled-ID prefix, but that result is not accepted
as Metal evidence because stderr capture could have hidden a CPU fallback.
Subsequent explicit qualification was blocked when the host stopped exposing
a Metal device even to `test_metal_f32`; speed and end-to-end parity must be
rerun with visible device/upload/forward-ready logs before this composition can
be promoted.

The subsequent layer-boundary trace found that the resident routed-MoE exit
skipped the model-policy per-layer input adapter (or its planned output scale)
before advancing to the next layer. The transformer GPU orchestrator now
composes that already-planned adapter on the routed branch just as it does on
the ordinary branch; model policy still owns the adapter, quant code remains
unchanged, and Metal only advertises independent execution capabilities. On
Gemma4 E26B, the layer-1 input discrepancy fell from `458.21521` maximum
absolute error (`14.91397` RMS) to `0.0835473` (`0.00481493` RMS). The default
Metal path then matched the retained eight-token reference sequence exactly
and measured `37.09` and `38.36 tok/s`, versus the retained llama.cpp Metal
baseline of `32.64 +/- 0.51 tok/s`. The backend architecture matrix passes
after promotion of the qualifying Metal capabilities.

The apparent dense Qwen2.5 regression was a benchmark-oracle axis error. The
strict wrapper used its CPU-only llama.cpp sampled-ID probe even when the
requested baseline was llama.cpp Metal. CPU produced
`2889,324,290,11,358,614,264,3405`, while both Metal engines produced
`2889,436,3847,11,358,614,264,3491`. The wrapper now uses the sampled-ID probe
only for a CPU llama baseline and retokenizes the actual requested-backend
completion for Metal/CUDA. The corrected Metal-to-Metal gate passes `8/8`
IDs. An adjacent sustained `tg64` run measured bitnet.c at `52.70 tok/s`
versus llama.cpp Metal at `51.80 +/- 0.67 tok/s` (`1.017x`). Generic CLI
diagnostic selectors now expose the existing transformer binary-state, QKV,
GQA, attention, and FFN comparison policies without adding model or quant
selection to `main`.

The next backend-matched qualification kept correctness and throughput as
separate gates. Gemma4 E4B matched llama.cpp Metal for `8/8` generated IDs,
but its default graph rejects a prepared Q4_0 FFN-down buffer submitted to an
ordinary matvec (`2560 x 10240`) and falls back to CPU. A full-length `tg64`
run therefore measured only `3.50 tok/s`, versus llama.cpp Metal at `32.16
+/- 0.35 tok/s`. Routing the existing quant-only FFN-down handle through the
Metal block-Q8 path removed the rejection but retained only `1/5` strict IDs,
so that attempted promotion was reverted. Qwen3.5 9B Q4_K_M also remains
unqualified on Metal. The broad reference-attention capability had allowed
its unique-value attention requirement to consume a path qualified only for
shared-value attention; narrowing that semantic composition and withdrawing
Metal's unneeded whole-token fallback claim restores the generic per-attention
handoff. The previously failing prompt now passes `8/8` Metal-to-Metal IDs
without regressing Gemma4 E4B's `8/8` shared-value gate. Sustained `tg64`
measured `9.20` and `9.31 tok/s`, versus llama.cpp Metal at `13.24 +/- 0.13`
tok/s, so speed parity remains open. Moving only the attention WO projection
back to Metal regressed immediately to `0/8` and was fully reverted. These are
independent backend execution gaps; no combined model/quant/runtime selector
was introduced.

The recheck also exposed a build-configuration integrity issue. The old phony
`config-check` recipe could delete stale objects only after make had already
decided that those prerequisites were up to date, and test/build sequences
could therefore leave a binary whose object configuration did not match the
requested runtime. The Makefile now uses a content-sensitive `.build-config`
stamp as a prerequisite of every reusable C, Objective-C, and CUDA object.
An actual plain-CPU to `BN_ENABLE_METAL=1` switch rebuilt all objects, linked
the Metal backend, and produced explicit M1 Max device, upload, and
forward-ready logs. Focused GPU/transformer tests, the backend architecture
matrix, and `git diff --check` pass. Runtime logs remain mandatory benchmark
evidence; `--metal` alone is never accepted as proof of Metal execution.

The 2026-08-10 CPU matrix was extended to the previously missing local model
families. Qwen3.6 27B dense `tg8` measured 4.68 tok/s on ARM NEON and 1.33
tok/s scalar versus llama.cpp CPU at 3.63 +/- 0.09 tok/s; the focused strict
gate matched 3/3 sampled IDs. Qwen3.5 35B-A3B sparse `tg8` measured 9.60 tok/s
on ARM NEON and 5.34 tok/s scalar versus llama.cpp at 6.11 +/- 1.48 tok/s, and
also matched 3/3 sampled IDs. The short sparse run is residency-sensitive:
the NEON profile spent about 1.13 seconds waiting for prefetch, so it is a
focused parity check rather than a replacement for the retained long-run
baseline. A Gemma4 E26B CPU refresh from the same process sequence was
discarded because both bitnet processes exited before generation under memory
pressure; only llama.cpp completed, so there was no valid adjacent comparison.

An isolated 2026-08-10 Gemma4 E26B rerun completed after the large-model
processes had cleared. With `--pread --cache-mb 4096 -t 8`, bitnet ARM NEON
measured `15.22 tok/s` at `tg32`, versus llama.cpp CPU at `14.99 +/- 5.08
tok/s` (`1.015x`). The 2 GiB `tg8` controls measured `13.49 tok/s` on NEON and
`6.77 tok/s` scalar versus llama.cpp at `8.66 +/- 2.64 tok/s`; the longer row
is the stronger throughput result. The standard three-token NEON gate matches
all `8/8` first sampled IDs and `21/24` generated IDs. The remaining three-ID
failure is not hidden by decoded text: llama's frontend removes a leading
normal `:` token on the HTTP prompt, so both frontends display `"Not Found`
despite different sampled IDs. `compare_llama.sh --strict` now requires the
complete expected sampled-ID count when its llama probe is available and uses
decoded text only as the fallback oracle. The probe also receives the same
context limit and KV type as the comparison CLI. These are benchmark-oracle
corrections; no quant, model-family, or runtime-backend selector was added.

An isolated sequential recheck on 2026-08-09, after stopping all overlapping
large-model processes, measured Qwen2.5 dense at 31.40 tok/s on ARM NEON and
11.06 tok/s in the forced-scalar build versus `llama-bench -t 8 tg32` at
31.53 +/- 4.54 tok/s. The same short Qwen3.6 sparse check measured 7.65 tok/s
on ARM NEON and 3.58 tok/s scalar versus `llama-bench -t 8 tg8` at
5.02 +/- 1.99 tok/s. The short sparse comparison remains residency-sensitive;
the retained `tg128` row above is the stronger sparse throughput gate.

The dense run uses identical eight-thread generation commands and demonstrates
clear NEON speed parity at 1.41x llama.cpp. The short sparse llama results have
high residency variance and are not replacements for the longer amortized rows
above. In particular, the Qwen3.6 `tg8` llama result has a 33% relative standard
deviation; its `tg128` row remains the regression gate. On the focused sparse
France prompt, the first five of sixteen generated
IDs match llama.cpp before numerical drift changes the continuation. The
focused Qwen2.5 and Gemma4 runs each match the first three of sixteen or more
generated IDs before drifting; throughput parity must not be reported as deep
token parity for those fixtures.

The 2026-08-09 Qwen2.5 follow-up localized that dense continuation drift to the
Q4_0 fused gate/up kernel's scalar `expf` SiLU. Its gate/up projections matched
llama.cpp exactly, but llama.cpp's ARM CPU path applies its NEON exponential
polynomial. The fused Q4_0 kernel now uses the existing matching NEON primitive
for both four-row groups and unaligned worker-range edges. The model continues
to declare SiLU semantics while the quant fusion owns its numerical kernel; no
model name or backend selection was added to quant dispatch. Focused `Hello`
generation now passes 8/8 IDs on scalar CPU, ARM NEON, and Metal. An isolated
NEON 64-token run measured 37.59 tok/s versus `llama-bench -t 8 tg64` at
29.35 +/- 2.74 tok/s. This supersedes the earlier three-token Qwen2.5 prefix
for that focused prompt, but broader deep-parity coverage remains open.

These results establish CPU speed parity for the two dense fixtures at the
measured eight-thread setting, but not deep token parity. The Qwen3 Q8_0
four-token actual-ID gate matches the first token on 8/8 prompts and 27/32
generated IDs overall;
scalar matches 122/128 generated token IDs and NEON matches 107/128. Keep the
deep parity issue open and do not treat throughput parity as numerical parity.
For the sparse Qwen3 fixture, a matched five-token France prompt passes 5/5
generated token IDs. Short sparse runs are not stable throughput evidence:
first-touch mmap runs measured about 5.5 tok/s in both engines, while a repeated
bitnet.c run reached 31.59 tok/s. The 128-token row above amortizes first-touch
cost within each process and is the current CPU sparse speed comparison.
The Qwen3.5 sparse fixture shows the same residency effect: 8.73 tok/s on a
first 64-token run, 28.08 tok/s on the immediate repeat, and 20.22 tok/s over
256 tokens. Its matched five-token France sample also passes 5/5 token IDs.
The Qwen3.6 sparse long run is only 3.8% above llama.cpp, so it remains a close
regression guard; its matched five-token France sample passes 5/5 token IDs.
For the focused `Once upon a time, there was a` failure, the corrected Q8_0
kernel makes the current-position layer-1 recurrent QKV projection bit-exact.
The next mismatch was a generic AArch64 execution-contract defect: llama.cpp
applies its four-lane NEON exponential polynomial to recurrent and MoE SiLU,
while the corresponding bitnet NEON kernels still called scalar `expf`.
Using the existing matching NEON primitive makes captured layer-0/layer-1 SSM
convolution and output projections match llama.cpp exactly. Moving full-router
softmax exponential/reduction behind the MoE CPU backend boundary then matches
llama.cpp's AArch64 vector contract without putting NEON code in routing
semantics. The focused actual IDs become `3777,15019,58763,6725`, matching
llama.cpp 4/4, and the broader eight-prompt four-token gate improves from 27/32
to 30/32 generated IDs while retaining 8/8 first-token matches. The short sparse run reached
37.18 tok/s and a subsequent `tg128` run measured 25.76 tok/s; the latter is
still above the adjacent llama.cpp 8.95 +/- 2.17 result but remains
residency-sensitive. Deep parity remains open on the two remaining broader-gate
IDs. Both corrections are backend-owned NEON arithmetic;
no Qwen-specific quant or runtime selector was added.
Making specialized native-quant Metal decode opt-in also materially improves
the current Qwen3 sparse gate without a model-family exception. The new default
passes 16/16 strict generated IDs. Three sequential 128-token samples measure
`[42.52, 43.11, 42.95]` tok/s, median 42.95, versus a non-concurrent
`llama-bench` result of `43.60 +/- 0.30` tok/s, ratio 0.985. A concurrent
llama-server top-k run was invalidated after its second prompt because unified
memory pressure produced an all-zero bitnet top-logit dump; it is not used as
evidence. Sparse Metal speed parity is now close but remains below 1.0.
At the standard matched eight-thread setting, three bitnet.c samples measure
`[43.09, 42.83, 42.96]` tok/s, median 42.96, while `llama-bench -t 8` reports
`43.06 +/- 2.00` tok/s. The median ratio is 0.998 and the distributions are
statistically indistinguishable, so this establishes practical sparse speed
parity but not a literal >=1.0 median gate. Strict decoded parity remains
16/16. The dense Metal ratio remains the larger unresolved gap.

Immediate interpretation: dense Qwen3 and Qwen3.5 CPU speed parity is clean at
the measured eight-thread setting, while deep Qwen3 decoded-token parity remains
open. Dense Qwen3 Metal remains healthy. The routed
resident Metal implementation has removed the former Qwen3-30B-A3B CPU
fallback bottleneck: all 48 MoE layers report resident execution. The strict
Metal-versus-CPU coherence run passes 14 checks with no failures, and the
managed llama.cpp comparison matches top-1 on 8/8 prompts with mean top-10
overlap 9.38 and 26/32 generated token IDs. A two-row routed down-K-quant
kernel, selected through quant behavior, measured 42.41 and 41.31 tok/s around
a one-row 40.23 tok/s control. A fresh restored-default `tg128` gate measures
37.20 versus a matched llama.cpp rerun at 44.02 +/- 0.84 tok/s, ratio 0.845;
the older 40.22/43.87 pair is not currently reproducible, so sparse Metal speed
parity remains open. Level-4 profiling attributes 11.7 ms/token to the combined
routed FFN shaders, the largest GPU category. Unbounded or depth-eight command
queuing does not improve this fully resident single-graph path and was removed.
Metal per-shape profiling now includes the quant type in both its aggregation
key and output. On the sparse graph this identifies 48 Q4_K stacked projections
(`4608x2048`), 47 Q4_K output projections (`2048x4096`), 24 Q4_K small
projections, and 24 Q6_K small projections; Q4_K dominates the non-MoE
projection cost. A two-row Q4_K x Q8_K output matvec diagnostic measured only
34.58 tok/s and changed the continuation, so both its shader and 64-row launch
geometry were removed. Future projection work should retain the accepted
one-row accumulation order.
Explicitly materializing the default float-input Q4_K projection's four
activation regions with eight `float4` loads also preserved 16/16 strict token
IDs, but was unstable and ultimately slower. The confirmation measured
387.5 us for `4096x2048` wq and 432.8 us for `2048x4096` wo, versus an adjacent
scalar-load control at 375.4 and 245.4 us. The vector-load form was removed;
the compiler-managed scalar population keeps lower register pressure.
The typed profiler also reports 48 adjacent Q4_K/Q6_K activation-quantization
reuse candidates per sparse graph. Skipping all 48 reduces specialized
quantizer dispatches from 143 to 95 but measures 35.62 tok/s versus an adjacent
35.58 tok/s no-reuse control, so the execution cache was removed. Likewise,
replacing the Q4_K split shader's scalar packed-byte loop with four-wide loads
and dot products regressed the long gate to 34.47 tok/s and was removed. These
results rule out activation-quantizer dispatch count and superficial byte-loop
vectorization as the current sparse gap.
A later split-workgroup diagnostic was invalidated before acceptance because
it changed the float-input `q4k_matvec_split` shader while runtime selected the
separate Q8_K-activation split pipeline. The mismatched 128/192-thread results
are not performance evidence; all launch and shader changes were restored. A
corrected experiment then changed the selected `q4k_q8k_matvec_split` shader,
its concrete runtime tile, thread count, and grid divisor together. Strict
sparse parity passed 16/16 generated IDs, but the 16-row/128-thread candidate
measured 36.21 tok/s versus its adjacent restored 32-row/256-thread control at
36.99 tok/s, a 2.1% regression. The corrected candidate was removed too.
Level-4 Metal profiling now separates routed gate/up from routed down without
changing normal submission. Across 47 layers it measures Q4_K gate/up at
6.576 ms and Q6_K down at 4.665 ms, locating the larger routed cost in gate/up.
A profiling-only Metal route-history dispatch now records selected expert IDs
without a CPU synchronization inside the graph and simulates per-layer LRU
capacity at backend teardown. Over the 128-token Qwen3-30B gate it recorded
50,656 expert accesses: hit rates were 0.2%, 0.8%, 3.7%, 26.7%, 52.3%, and
76.9% for 1, 2, 4, 8, 16, and 32 slots per layer. Even 32 native-layout slots
would consume about 2.6 GB and its 23.1% miss rate would add roughly 293 MB per
token of source-read plus cache-write traffic around the existing 635 MB per
token gate/up weight read. A dynamic packed expert cache is therefore not the
next implementation target. Current llama.cpp Metal source instead confirms
that its native-layout Q4_K path uses FP32 activations, two rows per SIMD group,
four block partitions, and 16-bit masked arithmetic; that exact arithmetic is
the next bounded routed-kernel diagnostic.
That upstream-style arithmetic was ported as a backend-only opt-in and then
fully removed. The Q4_K gate/up slice alone reached 37.08 tok/s versus an
adjacent 35.06 tok/s Q8_K control, but reduced the focused llama.cpp token
prefix from the control's 16/16 to 5/16. Completing the slice with upstream's
two-row FP32 Q6_K expert reduction fell to 33.26 tok/s and diverged immediately
after `Paris.`. The environment policy, both pipelines, alternate bindings,
launch geometry, and diagnostic shader were removed. Partial or complete
llama.cpp arithmetic transplantation is therefore not sufficient inside this
fused routed graph; the accepted Q8_K gate/up plus two-row Q6_K down path keeps
the stronger token contract.
A two-byte-at-a-time integer Q4_K gate/up loop then regressed the long gate to
34.23 tok/s and was removed. The next routed design should address the Q4_K
gate/up row decomposition or backend-resident layout rather than the smaller
Q6_K down phase.
An axis-correct diagnostic declared routed-down quant-scratch use in the quant
registry and omitted Metal's extra internal barrier for Q6_K down. It measured
35.28 tok/s versus an adjacent 35.76 tok/s unconditional-barrier control, so
the runtime change, capability API, and tests were all removed. The accepted
path retains the conservative barrier.
Splitting the routed Q4_K gate/up dot across 16 lanes was also rejected: it
measured 34.75 tok/s versus 39.15 tok/s after restoring the accepted 8-lane
path (and 42.98 tok/s in an earlier 8-lane control). The experiment was removed;
no quant-type dispatch exception remains in the Metal runtime.
An 8-lane two-row gate/up variant was rejected as well. Row pairing alone
measured 39.50 tok/s over 128 tokens, and explicitly fusing both rows into one
activation-load loop reduced throughput to 38.67 tok/s, versus the accepted
single-row gate/up gate at 40.22 tok/s. Both variants were fully removed.
Assigning separate eight-lane subgroups to gate and up for the same routed row
was also rejected. It preserved each projection's accepted reduction order and
the observed 128-token continuation, but the required cross-subgroup handoff
reduced throughput to 33.54 tok/s. The shader, threadgroup scratch, and
pipeline-specific 16-row geometry were fully removed; the accepted fused
gate/up subgroup remains the better instruction-overlap tradeoff.
A backend shape-specialized copy of the routed Q4_K gate/up pipeline made the
2048-column block count and row stride compile-time constants without changing
the quant or model contracts. It measured 35.21 tok/s versus the adjacent
35.06 tok/s generic control and changed the late continuation. The duplicate
shader entry point, pipeline, and shape selector were removed as neutral.
Sharing each packed Q4_K byte between its low/high-nibble lane pair was also
rejected. Four-byte loads by the even lane plus SIMD-pair broadcasts preserved
the scalar accumulation sequence within each lane, but shuffle and extraction
cost reduced throughput to 31.38 tok/s and changed the continuation. The
accepted independent scalar-byte loads were restored.
A 128-thread routed-FFN scheduling diagnostic was also removed. Its initial
49.52-53.30 tok/s readings were invalid because hard-coded 256-thread row
strides skipped output ranges, which strict coherence caught at the first
token. After making row coverage complete, coherence returned to 14 pass and
0 fail but throughput was only 37.83 tok/s, below the 40.07 tok/s adjacent
256-thread control.
Two further Q6_K routed-down diagnostics were rejected and fully removed. A
four-row-per-SIMD-group float kernel, adapted from the standalone Q6_K matvec,
measured only 32.28 tok/s and changed the generated trajectory. Keeping the
two-row reduction but quantizing each routed intermediate to Q8_K stalled the
long gate before completion. The accepted routed Q6_K path therefore remains
the two-row float-activation kernel; the next design must reduce its measured
cost without changing accumulation order or adding model-family dispatch.
Metal's approximate `fast::exp` intrinsic was also tested only inside the
routed Q4_K gate/up SiLU implementation. It preserved the observed 128-token
continuation but measured 37.62 tok/s versus an adjacent precise-`exp` control
at 37.68 tok/s. With no measurable gain, the shader was restored; no model,
quant-capability, or dispatch policy change was introduced.
Explicitly staging the shared Q8_K activation, scales, and block sums in Metal
threadgroup memory was rejected too. It retained the accepted eight-lane row
decomposition but measured only 37.74 tok/s versus the adjacent 37.68 tok/s
control and changed the 128-token continuation. The staging helper and fixed
scratch-size assumption were removed rather than becoming a backend layout
contract without a material gain.
A fuller llama.cpp-inspired decomposition was also tested and removed. It
partitioned Q4_K blocks across all four eight-lane subsets of a SIMD group and
reused each activation slice across two gate/up rows, while retaining native
GGUF weights and the existing Q8_K activation contract. It regressed to 37.03
tok/s and changed the continuation versus the 37.68 tok/s accepted control.
The result indicates that llama.cpp's multi-row structure depends on its
packed float-input arithmetic as a whole; launch geometry alone is not a viable
optimization for this Q8_K routed kernel.
The complete float-input variant was then measured rather than inferred. It
removed the pre-gate Q8_K quantizer and used a block-parallel, two-row fused
Q4_K gate/up kernel over the original float activation. Throughput regressed to
36.77 tok/s and the continuation diverged immediately, versus the restored
Q8_K path at 37.78 tok/s. The float shader, alternate binding, launch geometry,
and quantizer omission were fully removed. Q8_K remains the best measured
activation representation for this routed Metal kernel; further work should
target packed Q4_K weight arithmetic or a genuinely different backend-resident
weight layout.
A packed integer-vector version of the accepted Q4_K/Q8_K inner loop was also
tested separately from the earlier float-dot diagnostic. Four Q4 and Q8 values
were loaded together and multiplied as `int4`, preserving the complete observed
128-token continuation, but throughput fell to 37.30 tok/s from the restored
37.78 tok/s. It was removed; on M1 the compiler's scalar integer loop remains
the faster exact implementation.
Forcing that scalar loop to fully unroll preserved the same continuation but
measured only 37.40 tok/s, so the pragma was removed as well. These exact
variants rule out simple inner-loop instruction shaping as the missing sparse
Metal speedup.
Compacting router-logit execution from one 256-thread group per expert to one
SIMD group per expert was rejected too. Although it removed shared-memory
reduction and processed eight experts per threadgroup, it reduced end-to-end
throughput to 34.05 tok/s and changed the continuation. The original
256-thread, one-expert reduction and launch geometry were restored.
Mapping routed gate/up rows and route slots onto a two-dimensional Metal grid
was also rejected. It removed the per-thread flattened-index division without
changing quant arithmetic or accumulation order, but measured 36.25 and 32.84
tok/s versus an adjacent restored flattened-grid control at 35.72 tok/s. The
second run's clear regression and absence of a repeatable gain do not justify
an alternate launch contract, so both shader indexing and runtime geometry
were restored.
Doubling the exact routed gate/up workgroup from 256 to 512 threads was rejected
as well. It preserved the generated continuation but measured only 31.78 tok/s
against the adjacent 256-thread control's 35.72 tok/s. The 32-row workgroup
mapping remains the accepted Metal schedule.
A 40-row/320-thread routed Q4_K gate/up schedule was also rejected. Its
runtime grid and shader row stride were changed together while routed down
retained an independent 256-thread launch, but strict decoded parity held only
2/16 generated IDs instead of the accepted 16/16. It was removed without using
throughput as acceptance evidence; no quant or model policy changed.
A routed-only Q8_K activation quantizer that emitted the eight 32-value block
sums consumed by Q4_K/Q5_K directly was rejected. After correcting the matching
per-slot scratch stride, it preserved 16/16 strict llama.cpp token IDs, but two
128-token runs reached only 30.65 and 29.80 tok/s. Saving one block-sum load and
integer add in each consumer did not offset doubling the quantizer's serial sum
loop. The private pipeline, compact layout, and all consumer changes were
removed; the generic sixteen-sum Q8_K scratch contract is restored.
A second compact-sum implementation retained the original sixteen parallel
16-value reductions and paired their results after one threadgroup barrier. It
also preserved 16/16 strict token IDs and recovered throughput to 35.63 and
36.04 tok/s, but did not beat the adjacent 35.72 tok/s control beyond run
variance. It too was removed rather than retaining a redundant routed-only
quantizer pipeline and scratch layout.
Returning Q4_K/Q5_K scale/min metadata from the routed shader as its native
`uchar2` width instead of `uint2` was also rejected. Although extraction was
integer-equivalent and accumulation was unchanged, Metal compiler code
generation regressed the 128-token gate to 30.33 tok/s and changed the long
continuation. The 32-bit metadata representation was restored.
A branchless `uint2` scale/min decoder was rejected too. It replaced the
lane-divergent low/high branch with common byte loads and `select`, preserving
the exact extraction and generated continuation. Runs of 36.25 and 32.56 tok/s
did not show a repeatable gain over the 35.72 tok/s control, and the second was
a material regression, so the original decoder was restored.
Explicitly expanding the fixed two-row pointer and accumulation loops in the
routed Q6_K down helper was neutral to slightly worse at 35.91 and 35.36 tok/s.
It preserved the generated continuation, but the compiler already optimizes
the compact two-row form; the expanded shader was removed.
The routed Q4_K gate/up shader now computes flattened route slot, output row,
and selected expert once per eight-lane row subgroup and broadcasts them with
`simd_shuffle`, instead of repeating uniform division and route lookup in all
eight lanes. It is backend-private, preserves the quant arithmetic and graph
contract, passes 16/16 strict llama.cpp token IDs, and measured 36.42 and 37.10
tok/s versus an adjacent restored control at 36.22 tok/s (about 1.5% median
gain). The equivalent Q4_0 entry point remains unchanged because it was not
part of this measured sparse gate.
Applying the same uniform-load idea to Q6_K down was rejected. Loading expert
ID and route weight once per 32-lane SIMD group and broadcasting them preserved
16/16 strict token IDs, but measured 36.37 and 35.78 tok/s, below the retained
Q4_K-only path's 36.52 tok/s final check. A level-4 phase comparison was
invalidated by a simultaneous roughly twofold slowdown across every GPU
category, so it was not used as acceptance evidence. The Q6_K broadcasts were
removed.
A complete 192-thread routed gate/up schedule was rejected after separating
its launch width from the following 256-thread routed-down contract. The first
diagnostic incorrectly reused 192 threads for down and strict parity caught the
resulting skipped rows immediately (0/16 IDs); with down coverage corrected,
strict parity returned to 16/16. The valid 192-thread runs measured 35.16 and
35.83 tok/s, below the accepted 256-thread path, so all alternate geometry and
temporary down-launch separation were removed.
Replacing routed Q4_K gate/up block indexing with explicit pointer induction
for gate, up, Q8 values, scales, and block sums preserved 16/16 strict token
IDs and measured 37.08 and 37.23 tok/s. An adjacent restored indexed control
reached 37.46 tok/s, so the compiler-generated addressing remains preferable
and the pointer variant was removed.
Coalescing each routed Q4_K projection's adjacent FP16 `(d, dmin)` metadata as
one `half2` load also preserved 16/16 strict token IDs, but measured only
30.32 tok/s versus an adjacent scalar-load control at 32.79 tok/s, a 7.5%
regression. The compiler schedules the independent scalar conversions better;
the `half2` diagnostic was removed.
Replacing the per-byte low/high-nibble ternary with a lane-invariant shift and
mask also preserved 16/16 strict token IDs, but measured 33.15 tok/s versus an
adjacent ternary control at 33.02 tok/s, only 0.4% and below run variance. The
shift form was removed rather than retaining neutral instruction shaping.
Marking immutable mmap-backed Metal weight views with untracked hazards also
preserved 16/16 strict token IDs, but measured 32.20 tok/s versus an adjacent
tracked-buffer control at 34.57 tok/s, a 6.9% regression. Both consolidated
model views and per-tensor borrowed fallbacks were restored to the default
tracked `MTLResourceStorageModeShared` policy. This was a backend resource
diagnostic only; no quant or model policy changed.
Attaching the committed mmap residency set to the Metal command queue was also
rejected. The SDK-supported queue lifecycle (`addResidencySet:` paired with
`removeResidencySet:`) preserved 16/16 strict sparse token IDs, but a matched
128-token Qwen3-30B-A3B run measured 34.10 tok/s versus the adjacent unattached
control at 36.47 tok/s, a 6.5% regression. The queue calls were removed while
the existing residency-set allocation, commit, request, and release behavior
was retained. This probe was confined to backend resource ownership; it did
not change quant dispatch or model-family policy.
Removing the existing per-graph `requestResidency` refresh was neutral too.
The candidate measured 43.21 tok/s versus an adjacent restored control at
42.99 tok/s, only 0.5% and below sparse run variance. The request remains part
of the backend residency lifecycle; quant and model policy were untouched.
A 128-thread top-k launch for 128-expert routing was exact but neutral. The
shader derived the active SIMD-group count from the generic expert count and
strict parity remained 16/16; end-to-end runs measured 37.47 and 37.56 tok/s
against an adjacent 37.46 tok/s 256-thread control. The shape-driven geometry
and reduction change were removed rather than retaining another backend launch
contract for no material gain.
A 32-row/128-thread launch for the generic dense Q4_0 fused gate/up pipeline
was rejected after preserving 16/16 strict Qwen2.5 token IDs. Its 128-token
decode measured 57.34 tok/s versus an adjacent restored 64-row/256-thread
control at 57.68 tok/s. The smaller backend launch geometry was fully removed;
quant capabilities and model-family policy were unchanged.
The opposite 80-row/320-thread generic fused gate/up launch was rejected too.
It preserved 16/16 strict Qwen2.5 token IDs but measured 57.64 tok/s versus an
adjacent restored 64-row/256-thread control at 58.12 tok/s. The wider concrete
Metal launch contract was fully removed without changing quant capabilities or
model-family policy.
Hoisting repeated Q4_0 affine-correction activation sums was tested in two
forms and rejected. A separate 64-block GPU prepass plus resource barrier
regressed the focused fused gate/up row from 199.0 us to 490.7 us. A
single-dispatch threadgroup-cache version was then rejected during
qualification because Metal activation-pipeline initialization failed on the
M1 Max. Both shader experiments were removed. That failure also exposed a
runtime ownership defect: Metal and WebGPU activation-init failure left the
uploaded backend attached and Metal subsequently crashed. `main` now releases
backend model state and destroys the failed backend before CPU fallback,
matching the existing CUDA lifecycle without moving ownership into model or
quant code.
A 40-row/320-thread launch for the separate generic Q4_0 matvec pipeline was
rejected before benchmarking because the strict Qwen2.5 gate failed at the
first generated token. The shader stride and exact pipeline launch override
were fully removed; no quant dispatch or model-family policy changed.
Qwen3.5 MoE remains below the acceptance threshold. Qwen3.5-9B now has an
accepted hybrid Metal trajectory: the model-owned reference-attention semantic
selects the backend-declared CPU attention handoff, while SSM and dense FFN
remain GPU-resident. The quant registry independently supplies reference
K-quant capability for the model-owned reference-recurrent projection request.
The standard strict comparison passes 8/8 prompts and 40/40 generated token IDs
against llama.cpp Metal. Three matched 32-token decodes on the restored default
Metal synchronization path measure 14.66, 14.95, and 15.01 tok/s (14.95 median)
versus `llama-bench` at 15.07 tok/s, or 99.2% of baseline. An experiment that
queued consecutive GPU-only graphs measured equivalently on this hybrid path
and regressed the fully resident sparse path; it was removed completely rather
than retaining an unproven runtime policy. The adjacent sparse Qwen3 Metal
coherence gate still passes 14 checks with no failures.
Earlier layer comparison
shows SSM layer 0 is close (`qkv` max error about 2e-6, projection about 1.8e-5),
while ordinary k-quant FFN accumulation reaches about 0.02 next-norm error by
the end of layer 0. The first attention layer's Q4_K Q/K projections are within
about 3e-6, but its Q6_K value projection differs by 0.154 max and 0.032 mean.
A prior composed CPU-attention/Metal-SSM+FFN diagnostic preserved only the
first token. The accepted path removes the redundant whole-model fallback
guard and uses the existing generic per-attention handoff policy; it contains
no Qwen-specific runtime exception.
Metal now honors the existing quant-owned `REFERENCE_KQUANT` IR flag for Q6_K
matvecs through a scalar-order backend kernel, and reference-attention emission
sets that flag on Q/K/V projections. On Qwen3.5-9B layer 3 this reduces the
Q6_K value projection from 0.154 max / 0.032 mean error to 7.2e-7 max /
8.9e-8 mean error. Exact scalar-order Q4_K/Q5_K expansion was rejected: the
CPU reference is the NEON accumulation contract, it did not improve layer-0
FFN parity or recover the first token, and the full diagnostic fell to
12.89 tok/s. The accepted Q6_K slice remains opt-in behind reference-attention
backend policy and does not weaken the default full-fallback correctness guard.
An exact-input SSM diagnostic now reruns the active CPU recurrence from the
same pre-layer state using Metal-produced Q/K/V, alpha, and beta values. On
Qwen3.5-9B layer 0 it matches the Metal state bit-for-bit (`max_abs=0`, mean
and RMS 0), proving the recurrence kernel is not the source of the observed
`0.00447` end-state drift. That drift is induced upstream: the post-conv/L2
QKV vectors differ by at most about `1.9e-6`, but the delta-rule update
amplifies those differences. A Q4_K x Q8_K integer block-order experiment was
also rejected and fully restored because adjacent standalone and end-to-end
controls were unchanged at reported precision. The next parity target is the
quant-declared projection contract feeding SSM, especially Q5_K QKV, without
placing model-family checks in Metal or backend checks in model policy.
A model-semantic / quant-capability / backend-pipeline Q5_K reference SSM
projection slice was evaluated and fully removed. Although its ownership
boundaries were correct, layer-0 post-conv/L2 QKV error worsened from
`1.91e-6` to `2.86e-6`, state max error worsened from `0.00447440` to
`0.00447488`, and the one-token diagnostic slowed. The adjacent restored
control returned to the original measurements. The next diagnostic must
capture raw pre-convolution QKV values on identical inputs; another projection
kernel is not justified by post-convolution evidence alone.
The SSM compare graph now snapshots its exact projection input and raw QKV
output into otherwise-unused attention activation slots before convolution.
On Qwen3.5-9B layer 0, CPU Q5_K versus Metal on that identical input differs
by `0.03832` max, `0.006146` mean, and `0.007826` RMS. This directly locates
the trajectory divergence in the Q5_K backend arithmetic contract. The later
convolution/L2 stage compresses the visible QKV difference to about `1.9e-6`,
while the recurrent update amplifies it into the observed state drift. Future
Q5_K work should be evaluated against this raw same-input metric before token
or throughput testing.
The quant-owned Q5_K specialized-native contract now uses Q8_K activations on
Metal, matching the active ARM NEON SDOT path without any model-family or SSM
condition in backend code. On the same Qwen3.5-9B layer-0 input, raw QKV error
drops from `0.03832` max / `0.006146` mean / `0.007826` RMS to `3.81e-6` max /
`8.02e-8` mean / `1.58e-7` RMS. Quant integration tests and the backend
architecture matrix pass. This resolves the raw Q5_K projection contract;
downstream SSM state and token parity remain separate open work.
Follow-up same-input diagnostics clear the adjacent operations individually:
the active CPU Q5_K batch path still agrees with captured Metal raw QKV within
`3.81e-6`, CPU convolution plus L2 normalization on the exact Metal raw QKV
agrees within `4.77e-7`, recurrence remains bit-exact on identical inputs, and
the Q5_K output projection agrees within `1.49e-8` on the exact gated input.
The ARM Q5_K disassembly uses a rounded negative correction followed by two
FMAs; the Metal Q5_K x Q8_K shader now states that order explicitly. Precise
shader compilation produced no measurable improvement and was removed. A
global Q4_K specialized-native preference at 4096 columns fixed its standalone
projection comparison but did not improve SSM state and slightly worsened later
comparisons, so that quant-policy experiment was also removed.
Despite the isolated agreement, the opt-in full Qwen3.5 Metal graph remains
incorrect: five greedy tokens across eight prompts match llama.cpp `0/40`, and
the representative final distribution overlaps only `2/10` logits. Keep the
default model-policy fallback in place. The remaining target is cumulative
cross-operation state/dataflow parity, not another model-specific quant branch.

ARM NEON / CPU plan:

- [x] **Requested-family CPU parity gate** — `make test_cpu_parity` builds
  `bitnet` and `bitnet_scalar`, then runs the Qwen and Gemma4 CPU parity gates
  in sequence. Use `make test_cpu_parity_required` for the final no-skips
  proof once the Qwen3.6 sparse, Gemma4 dense, and Gemma4 sparse fixtures are
  present; it checks llama.cpp tools, upstream fixture metadata, all requested
  fixture presence, and known download sizes before building.
  `make check-cpu-parity-fixtures` reports
  local fixture status, `make check-cpu-parity-remote-fixtures` verifies the
  current Hugging Face repos, selected filenames, GGUF architectures, and file
  sizes, and `make fetch-cpu-parity-fixtures` runs that remote check before
  fetching missing large fixtures.
- [x] **Qwen-family CPU parity gate** — `make test_qwen_cpu_parity` runs
  `test/qwen_cpu_parity.sh`, which compares NEON and scalar output against
  llama.cpp for discovered Qwen2.5, Qwen3, Qwen3.5, and Qwen3.6 fixtures. The
  standard level uses 5-token Qwen3/Qwen3.5 dense gates and at least 3-token
  gates for every other dense/MoE case. The deeper default is required because
  a Q8_0 reduction-order regression matched the first sparse Qwen3.6 token but
  diverged immediately afterward;
  `QWEN_CPU_PARITY_LEVEL=full` raises the token budget for deeper checks, and
  `QWEN_CPU_PARITY_CASES=qwen25,qwen3_dense` supports focused reruns. Set
  `QWEN_CPU_PARITY_BACKENDS` or shared `CPU_PARITY_BACKENDS` to select
  `neon,native,scalar,avx2,avx512`; AVX selections build runnable x86 parity
  binaries on x86_64 hosts.
- [x] **Gemma4 CPU parity gate wiring** — `make test_gemma4_cpu_parity` runs
  `test/gemma4_cpu_parity.sh`, which discovers dense E2B/E4B/31B and sparse
  26B-A4B Gemma4 GGUFs while excluding mmproj sidecars, then compares both
  NEON and scalar backends against llama.cpp with `--maxseq 512` by default.
  `GEMMA4_CPU_PARITY_BACKENDS` or shared `CPU_PARITY_BACKENDS` selects
  `neon,native,scalar,avx2,avx512`; the default is host-derived so ARM runs
  NEON/scalar and x86_64 runs scalar/AVX2/AVX512. The gate skips when fixtures are absent
  and fails missing fixtures when `REQUIRE_MODELS=1` is set. Both local Gemma4
  fixtures are now present, and standard coverage uses at least three tokens
  per prompt; the dense five-token sample passes while sparse
  deep-token parity remains open.
  The 2026-08-09 strict sampled-ID run passes dense Gemma on both NEON and
  scalar at 23/23 generated IDs across eight prompts. Sparse Gemma reaches
  23/24 on NEON (7/8 first IDs) and 21/24 on scalar (6/8 first IDs), confirming
  that the remaining failure is shared model/MoE semantics with numerical
  sensitivity rather than an ARM SIMD-only defect.
  Layer-0 route diagnostics now use ggml tensor byte strides in the llama.cpp
  probe. The corrected probe and bitnet both select experts
  `102,86,21,63,26,84,22,116`; their sorted weights and raw router logits also
  agree closely. The earlier apparent route-ID mismatch was a probe layout bug,
  not model execution. Neither llama-style FP16-table GEGLU alone nor combined
  with the Q4_0 x Q8_0 reference dot fixes the known third Hello token, so both
  experiments were rejected rather than made model- or platform-specific.
  Full-vector binary checkpoints localize the early accumulation without a
  semantic discontinuity: relative RMS error at layer output is 0.15% at layer
  0, 0.68% at layer 1, 1.17% at layer 5, and roughly 3-4% through layers
  10-29. Within layer 1 it progresses from 0.21% after attention RMSNorm to
  0.30%/0.23%/0.47% in raw Q/K/V and 0.51% in the combined attention value,
  reaching 0.59% after the attention residual. Existing reference-math and
  Q4-dot controls produce byte-identical layer-1 output here, ruling them out
  as corrective switches. Continue by comparing the exact reduction contract
  of the first attention projections rather than adding architecture-specific
  tolerances.
- [ ] **Resolve Gemma4 sparse deep token parity** — the dense E4B fixture
  measures 27.55 tok/s versus llama-bench at 21.76 tok/s and passes a matched
  5/5-token Hello sample. The sparse 26B-A4B fixture measures 23.04 tok/s on a
  warm eight-token sample versus llama-bench at 8.97 tok/s, but matches only
  2/5 generated Hello token IDs. Scalar and NEON produce the same divergent
  sequence, so investigate shared Gemma/MoE numerical semantics rather than
  adding a quant- or platform-specific correction. Longer raw generations can
  currently trigger the repeated-ngram loop abort and return `tokens=-1`; do
  not use those aborted runs as speed evidence.
  A matched Metal recheck on 2026-08-09 keeps the axes separate: dense E4B
  passes 5/5 generated IDs and measures 32.24 tok/s versus llama.cpp
  39.13 +/- 10.44 on the noisy `tg5` sample. Sparse 26B-A4B matches 4/5 IDs
  but measures 19.07 tok/s versus 45.66 +/- 13.40 at `tg8`. The sparse
  numerical divergence also occurs on scalar and NEON, so its correction
  belongs to Gemma/MoE model semantics. Metal profiling assigns the speed gap
  primarily to resident routed expert gate/up/down execution, which remains a
  backend runtime concern; do not encode either issue as a quant-format or
  platform-specific model exception.
  A Metal-private paired Q4_0 routed gate/up traversal was tested and removed.
  It preserved the 4/5 sparse token prefix and shared activation loads between
  the two projections without changing either accumulator's order, but `tg8`
  samples ranged from 17.60 to 21.58 tok/s around an adjacent 19.04 tok/s
  restored control. The result is too pressure-sensitive to justify retaining
  another shader path. The larger opportunity remains eliminating per-layer
  host orchestration and command submission through the generic resident graph,
  not adding a Gemma- or quant-conditioned runtime exception.
  Enabling Metal's existing reference-attention capability keeps dense E4B at
  5/5 IDs and raises sparse 26B-A4B from 19.07 to 25.98 tok/s, but reduces the
  sparse prefix from 4/5 to 2/5 IDs. Forcing the sparse FFN back to CPU leaves
  that 2/5 result unchanged, isolating the added drift to Metal attention rather
  than routed MoE. A position-1 CPU trace against the local llama.cpp layer
  probe agrees closely through layer-0 attention, router top-8 and normalized
  weights, routed output, dense branch, combined output, and final state. The
  first expert-set difference appears at layer 17 after gradual numerical drift.
  The next backend task is an exact-order Metal GQA scores/softmax/combine mode,
  selected by the model's declared reference-attention requirement composed
  with a backend capability. It must not be selected by Gemma name, MoE shape,
  or Q4_0 type.
  That diagnostic is now implemented and stage-isolated. Replacing scores,
  softmax, or value combine independently produces the same 2/8 resident
  trajectory, so GQA reduction order is not the source of the drift. A
  CPU-order Metal RMSNorm makes the layer-0 position-0 normalized activation
  bit-exact, but Q/K/V differences remain unchanged (Q max abs 0.032253), which
  rules out norm reduction order. A sequential-block Q4_0 Metal matvec likewise
  left the projection error unchanged and was removed. The remaining arithmetic
  distinction is quant-owned: CPU decode uses the Q4_0 x quantized-activation
  SDOT contract while the default Metal projection consumes float activations.
  Any further parity experiment must be declared by Q4_0 capabilities and then
  composed with backend support; it must not be selected by Gemma semantics.
  A current `BN_GPU_PROFILE=4` fallback run also shows roughly 155 Metal execute
  calls per decoded token, commonly one operation per call, because reference
  attention creates a host dependency at every layer. Encode/submission cost is
  about 0.3 ms for most calls. This explains much of the 19 tok/s fallback gap;
  eliminating it requires a coherent resident attention path or an explicitly
  asynchronous backend execution contract, not a model-family scheduling rule.
  The role-specific prepared-native-quant attention capability was then tested
  end to end without adding a model selector. Q4_0 Q projection max error at
  layer 0 fell from 0.032253 to 0.001083, but the eight-token continuation
  regressed from 4/8 to 0/8 llama.cpp IDs. Long unflushed graphs also exposed a
  shared activation-quant scratch dependency; an unconditional Metal resource
  barrier removed the NaN but did not restore token parity, and adjacent sparse
  throughput controls did not justify applying that barrier to every prepared
  dispatch. FP16-rounding the stored Q8 block scale to mirror CPU storage left
  Q/K/V metrics unchanged. The capability, barrier, and scale experiment were
  all removed. Prepared CPU-SDOT arithmetic is therefore not a valid substitute
  for the float Metal attention contract; shared Gemma/MoE semantic drift near
  the known layer-17 route transition remains the parity target.
  A deeper `Hello` trace then found that the shared Gemma GEGLU path rounded
  both its input and result through FP16 even though the architecture declares
  the FP32 tanh GELU approximation. Removing that artificial
  activation-boundary rounding improves scalar and ARM NEON from a 2/5
  generated-ID prefix to 3/5 while preserving dense E4B at 5/5. The redirected
  strict Metal harness also reaches 3/5, but terminal-paced Metal runs remain
  at 2/5; full intra-graph barriers do not remove that timing-sensitive
  difference. Metal repeatability is therefore a separate backend-runtime
  issue, not parity evidence. The fourth CPU decision remains close (`0.028`
  between the leading comma and llama.cpp's hyphen on the forced matched
  context), so deeper shared Gemma/MoE drift is still open. The activation
  correction lives in MoE model semantics; it adds no quant-format or backend
  selector.
  Metal's generic GEGLU shader and resident routed-MoE activation now implement
  that same declared FP32 tanh contract instead of a backend-private FP16
  variant. Dense Metal remains 5/5 and the redirected sparse gate remains 3/5;
  terminal pacing remains 2/5, so this semantic alignment is retained while
  command/resource repeatability stays open under backend ownership.
  A sampled-ID audit then found that the strict comparison's llama.cpp side
  retokenized decoded text instead of recording the IDs selected by the model.
  That is not invertible for this vocabulary: llama.cpp actually generates
  `3324,236772,616,236772,1048`, while retokenizing the identical visible text
  produces `3324,236772,45518,236772,45518`. The local layer probe now supports
  `--generate` and the strict gate consumes those actual sampled IDs when the
  probe is available. Under that stronger definition, sparse Gemma CPU parity
  is 2/5 IDs, not 3/5. At the first mismatch llama.cpp ranks `616` above
  `45518` by about 0.115 logit; layer-0 CPU checkpoints remain close and the
  difference grows gradually, so the unresolved correction remains shared
  model/MoE numerical semantics rather than a quant- or runtime-specific rule.
  The generic dense-residual GPU graph also exposed a backend activation-size
  defect: `HB/HB2` were allocated for `max(hidden_dim,
  expert_hidden_dim)`, but post-normalization may store a `dim`-element vector
  there. Sparse Gemma has `dim=2816` and `hidden_dim=2112`. A shared GPU sizing
  policy now allocates `max(dim, hidden_dim, expert_hidden_dim)` across Metal,
  WebGPU, and CUDA. With valid storage, the experimental Metal dense branch can
  be compared reliably, but it still differs from the CPU dense residual by up
  to 0.0998 at layer 0 and the combined state by 0.336. That compounds into an
  incorrect first token, so Metal does not advertise the dense-residual
  low-bit capability yet; the parity-preserving CPU semantic fallback remains
  selected through capability composition.
- [ ] **Resolve Qwen3 dense deep token parity** — the Q8_0 scalar path now
  pre-quantizes batch inputs, uses NEON-matching activation inverse arithmetic,
  and reduces Q8 blocks in four-block groups. The short first-token gate passes
  8/8 prompts, but the 16-token strict gate matches 122/128 scalar token IDs and
  107/128 NEON token IDs against llama.cpp. Diagnose shared numerical order and
  the additional NEON drift without introducing model-specific SIMD behavior.
  A quant-owned diagnostic changing Q8 block activations from symmetric
  `amax/127` scaling to signed-maximum `max/-128` scaling fixed the focused
  France prompt but reduced aggregate NEON parity to 103/128, so it was
  rejected and fully reverted. A second diagnostic kept NEON transformer math
  but substituted the existing float Q8 matvec; it reduced parity further to
  93/128. Scalar's 122/128 result therefore does not come from the Q8 dot
  kernel alone.
  A current operation-isolation pass found the remaining short NEON drift in
  backend RMSNorm arithmetic: layer inputs and norms begin identical, changing
  Q8_0 reduction order fixes dense output but breaks sparse Qwen3.6, and scalar
  GQA/activation substitutions do not fix the focused prompt. Using the stable
  scalar-order RMSNorm implementation from the NEON backend restores 40/40
  generated IDs across the standard five-token Qwen3 dense gate while retaining
  the sparse Qwen3.6 Metal regression prompt. Qwen3 decode remains 135.15 tok/s
  at eight threads, above the retained llama.cpp CPU baseline. This is a
  platform-runtime backend choice; no quant or model-family selector was added.
- [ ] **Resolve Qwen2.5 deep greedy-token parity** — the current Q4_0 fixture
  matches llama.cpp on all 8 first generated token IDs. A 2026-08-10 strict
  eight-token NEON refresh matches 46/64 actual sampled IDs across the standard
  prompt set. The llama probe now honors tokenizer BOS metadata, reads the
  public last-token logits API instead of a callback row, and uses FP16 KV when
  serving as the default llama-completion sampling oracle; layer diagnostics
  retain their explicit FP32-KV mode. Scalar and
  NEON diverge from llama.cpp at the same token on the focused France prompt,
  and the Q4_0 quantized-dot diagnostic produces the same 3/8 prefix, so this
  is not evidence for a scalar-only or Q4_0-kernel-specific defect. Matched
  non-flash layer probes are effectively identical through layer 0 and show
  gradual numerical drift rather than a discrete semantic break. At the first
  differing decision, llama.cpp ranks `country`, `capital`, and `city` within
  0.49 logits, while bitnet.c ranks the same three within 0.13 logits. Continue
  with shared numerical-order comparisons; do not add a model-specific tie
  override or move this policy into a quant/backend axis.
  The same refresh measured the 32-token `Once upon a time` control at 54.87
  tok/s on ARM NEON and 12.41 tok/s in the forced-scalar build; both produced
  the same continuation. `llama-bench -ngl 0 -t 8 tg32` measured 48.69 +/-
  1.51 tok/s, so the current dense CPU issue is deep numerical parity, not
  throughput. The comparison harness now expands an empty bitnet argument
  vector portably on macOS Bash 3.2 instead of aborting before generation and
  sets the same repeat penalty on both engines. Matching bitnet's KV storage
  with `--kv16` leaves the actual-ID result at 46/64 and worsens one decoded
  continuation, ruling out KV precision as the remaining shared drift source.
  On the exact first-divergence France context, both engines return the same
  top-10 token set; bitnet ranks `capital` first and `country` third, while
  llama.cpp ranks `country` first and `capital` second. Bitnet's logits are
  identical across one/eight threads, decode-only prefill, scalar, and NEON,
  further excluding those runtime axes.
  The matching FP32-KV sequential layer checkpoint now bounds the gradual
  drift more tightly: layer-output max absolute error is `6.02e-6` at layer 0,
  `6.02e-3` at layer 1, `1.58e-1` at layer 17, and `1.64` at layer 35. Within
  layer 1, attention RMSNorm remains within `5.48e-6`, the current K projection
  within `1.67e-6`, the combined attention value reaches `7.95e-4`, and the
  FFN output reaches `5.95e-3`. The scalar-order reference-math route produces
  the same attention checkpoint as NEON, further excluding an ARM-only GQA
  reduction defect. Binary transformer checkpoints are now independent of the
  optional text statistics observer, so future comparisons cannot silently
  skip their output when only `BN_DUMP_BINARY_*` is configured.
- [ ] **Profile CPU k-quant hot paths by model family** — collect per-op timing
  for Q8_0 dense, Q4_K_M dense, and Q4_K_M MoE. Break down attention QKV/O,
  FFN gate/up/down, routed expert gate/up/down, shared expert work, logits, and
  router computation. Do not optimize blindly from aggregate tok/s.
- [x] **Close Q4_K_M dense CPU speed gap** — the current eight-thread
  `Qwen3.5-9B-Q4_K_M` decode diagnostic measures 16.60 tok/s versus
  llama-bench at 11.71 tok/s. Preserve that result while investigating deep
  numerical parity in the quant and transformer-math owners; do not add a
  model-family branch to a SIMD kernel.
- [ ] **Make MoE CPU expert execution cache-aware** — for Qwen3/Qwen3.5 MoE,
  measure expert cache hit rate, mmap/pread behavior, routed expert count,
  shared expert cost, and page faults. Optimize expert locality and active
  expert batching before adding more SIMD variants.
- [ ] **Batch routed expert matvecs where possible** — group selected experts
  by quant type and shape so NEON kernels can amortize activation quantization
  and thread dispatch overhead across gate/up/down work. Preserve deterministic
  routing and top-k logits parity as the acceptance check.
- [ ] **Tune thread partitioning for sparse MoE** — large MoE CPU runs are
  sensitive to tiny per-expert jobs and thread wake overhead. Add a benchmark
  sweep for thread count, expert batch size, and pread cache size, then encode
  the best default policy in MoE execution rather than relying on CLI tuning.
- [x] **Add Qwen3.6 CPU fixture before claiming support** — the local
  `Qwen3.6-27B-UD-Q4_K_XL` fixture is discoverable through
  `BN_MODEL_QWEN36_DENSE` / `BN_MODEL_ROOT`, and scalar plus NEON one-token
  llama.cpp parity smoke checks pass on the current tree.
- [x] **Add Qwen3.6 sparse CPU fixture** — the local
  `Qwen3.6-35B-A3B-UD-Q4_K_M` fixture measures 8.42 tok/s versus llama-bench
  at 8.11 tok/s over 128 tokens and passes a matched 5/5-token sample.
- [x] **Add Gemma4 real-model fixtures** — the official E4B dense and 26B-A4B
  sparse Q4_0 fixtures are local and exercised. Fixture coverage is complete;
  sparse deep-token correctness is tracked separately above.
  A focused `Hello` diagnostic also excludes three tempting axis-local
  substitutes for the remaining 2/4 sampled-ID prefix: quant-owned reference
  Q4 dot order, the scalar transformer-math backend, and model-owned reference
  RMSNorm order all leave the strict prefix unchanged. The synthetic prepared
  Metal Q4_0 kernel agrees with the CPU prepared kernel within `9.7e-8`, so the
  shared CPU/Metal token choice is not evidence of a basic Metal repack defect.
  Metal resource barriers now clear only the hazards covered by each
  resource-specific barrier instead of discarding unrelated pending writes.
  With the dense-residual diagnostic enabled, the decode-only sparse Gemma4
  gate matches 8/8 sampled IDs against the llama.cpp FP32-KV probe. The strict
  harness now forces FP16 probe KV only when bitnet.c is also passed `--kv16`;
  its previous unconditional FP16 probe compared different runtime states.
  The parity-preserving `tg32` path measures 7.27 tok/s versus llama.cpp Metal
  at 32.64 +/- 0.51 tok/s. A quant-only attention-output resource removed 19
  prepared-layout rejections and reached 38.60 tok/s in a short profile, but
  alternated between the correct first ID and the known wrong `6491` trajectory
  across identical Metal runs, including with full barriers. That candidate
  was removed. Prepared FP32-input and prepared block-Q8 attention diagnostics
  both execute the full resident graph at roughly 64-77 tok/s but immediately
  choose the same wrong `6491` token. A layer-0 comparison keeps Q/K/V and GQA
  within a few micro-units before the attention output differs by about 0.35;
  a precise one-lane prepared-Q4 `wo` reduction matching the CPU NEON block
  order did not restore the trajectory and was also fully removed. A CPU
  prompt-to-Metal state handoff was also rejected: on a confirmed Apple M1 Max
  run, the CPU top-level prompt produced `79303` before either KV or SSM upload,
  rather than the reference `236764`. The parity path is therefore a
  backend-orchestrated graph with reference-attention fallback, not an
  interchangeable CPU prompt execution path; the handoff and its temporary
  runtime controls were fully removed. Whole-token routing now uses the
  backend-neutral `BN_GPU_CAP_REFERENCE_ATTENTION_TOKEN_FALLBACK` capability
  instead of relying on a prepared graph rejection. The optional backend
  `prepare_cpu_operations` hook lets Metal request residency and prefault its
  mapped model view before CPU orchestration. This path passes the strict 8/8
  sampled-ID gate and measures 6.38 tok/s at `tg32`, so it removes accidental
  control flow but does not close the 32.64 tok/s Metal gap. Enabling the
  backend-only CPU-route resident-MoE callback also retains the same 8/8 prefix
  after expert down-scale plumbing, but per-layer command submission reaches
  only 5.87 tok/s at `tg8`. Continue from the prepared attention-output
  placement and upstream cache-order boundary;
  do not encode this as a Gemma, Q4_0, or Metal-combined selector.
  The explicit reference-attention diagnostic now advertises the existing
  quant/backend `PREPARED_NATIVE_QUANT_ATTENTION` capability, allowing the
  quant-owned block-Q8 activation path to be measured independently of model
  family. At prompt position 4, layer-0 Q/K/V agree with CPU within `3.8e-6`,
  GQA within `4.3e-6`, and post-attention state within `1.5e-5`; layer 29
  post-attention and composed MoE state remain within about `1.7e-5`. GPU
  logits agree with CPU logits computed from the same final state within
  `1.2e-4`, yet the near-tied argmax still selects `6491` instead of `236764`.
  CPU-order global RMSNorm and final-layer CPU refinement do not restore the
  reference token. The remaining work is therefore accumulated full-state
  trajectory equivalence, not a basic Q4 projection, attention, or logits
  kernel defect.
  Continue from matched layer inputs and compare attention, routed MoE, and
  dense-residual operation ordering; do not create a Gemma + Q4_0 + Metal
  selector. An exact forced-context layer-0 trace (`2,9259,3324,236772`) now
  narrows this further: input attention RMSNorm agrees with llama.cpp to about
  `1e-6` in the displayed values, while the first visible difference is the
  Q4_0 Q/K/V projection at roughly `1e-3`; attention output, routed MoE, and
  dense MLP then drift gradually, with identical top-8 expert IDs. The pinned
  llama.cpp ARM source uses the same absolute-maximum Q8_0 activation scale,
  FP16 scale storage, and ties-to-even conversion as the quant-owned NEON
  implementation. Its remaining Q4_0 difference is the two-accumulator
  floating reduction structure, which the earlier native-layout experiment
  already showed worsens final token parity. A later quant-owned diagnostic
  reproduced that paired ARM reduction exactly and applied it without any
  model selector: sparse Gemma retained the same wrong focused token and dense
  Qwen2.5 retained the same `3/8` prefix, while bypassing the faster prepared
  layout. The diagnostic was removed. Do not revive that reduction as a global
  default without new aggregate evidence across dense and sparse fixtures.

Metal plan:

- [x] **Make Metal fallback reasons visible per layer/op** — print or record
  backend placement for Qwen3.5 dense and MoE models: native Metal, repacked
  Metal, split/fused Metal, CPU fallback, and the exact missing capability.
  Current profiling reports routed tensor types, resident-layer counts,
  route wait/copy time, native-quant dispatches, barriers, and per-op timing.
- [x] **Prioritize k-quant Metal kernels for Q4_K_M dense** — current matched
  `tg32` measurements put `Qwen3.5-9B-Q4_K_M` at 17.40 tok/s versus
  llama.cpp at 16.34 +/- 0.14 tok/s, with 64/64 generated IDs matching across
  eight prompts. The sparse `Qwen3.5-35B-A3B-Q4_K_M` fixture reaches
  31.06 tok/s versus 30.53 +/- 0.57 tok/s with 32/32 generated IDs matching.
  Quant-format kernels remain owned by `src/quant/`; model-family selection and
  runtime placement remain separate policy axes.
- [ ] **Keep dense Qwen3 Metal as a regression guard** — `Qwen3-0.6B-Q8_0`
  Metal is currently coherent and faster than llama.cpp. Add it to the Metal
  parity matrix so future Qwen3.5/MoE changes cannot regress the healthy Q8_0
  dense path.
- [ ] **Implement MoE GPU placement as a whole path** — Qwen3-30B-A3B now keeps
  all 48 routed layers resident, passes strict CPU/Metal coherence, matches
  llama.cpp top-1 on 8/8 prompts, and reaches 0.917x llama.cpp in the managed
  Metal gate. For Qwen3/Qwen3.5 MoE generally,
  do not move only one expert matvec to Metal. A useful Metal path needs router
  logits, top-k routing, expert gate/up/down, shared experts, residual/norm,
  and expert output accumulation resident on the backend with minimal readback.
- [ ] **Add Metal expert cache/upload policy** — large MoE models cannot assume
  all experts fit in fast GPU residency. Implement a backend-owned expert cache
  with explicit capacity, LRU/working-set reporting, async upload where
  possible, and clear fallback when a layer exceeds budget.
- [ ] **Avoid CPU/GPU ping-pong at fallback boundaries** — when SSM, MoE, or a
  quant format falls back to CPU, schedule whole blocks on CPU or whole blocks
  on Metal. Per-op fallback inside a layer should be treated as a bug unless a
  benchmark proves it is faster.
- [ ] **Compare against llama.cpp placement, not only tok/s** — collect
  llama.cpp logs for the same MoE runs to see which tensors/layers are actually
  on Metal. Matching `-ngl 99` is not enough if bitnet.c and llama.cpp place
  routed experts differently.
- [ ] **Qwen3.6 Metal fixture** — add a Qwen3.6 GGUF before any Metal support
  claim. Run both dense and MoE forms if available, then decide whether the
  limiting work is architecture rules, quant kernels, or MoE placement.
  The local dense `Qwen3.6-27B-UD-Q4_K_XL` fixture now runs on the generic
  Metal path at 4.71 tok/s for the focused eight-token generation versus
  llama.cpp build 9950 `tg16` at 4.32 +/- 0.07 tok/s. The decoded completion
  and first token agree, with a 2/8 retokenized ID prefix before early
  termination, so this establishes speed and short-output parity rather than
  deep eight-token parity. No Qwen3.6 condition was added to Metal; model
  semantics remain under `BnModelArchOps` and k-quant execution remains under
  quant/backend capability composition.
  The sparse `Qwen3.6-35B-A3B-UD-Q4_K_M` fixture is fully routed-resident on
  Metal. The earlier focused strict 32/32 claim was based on a stale comparison
  path and is withdrawn: for `Once upon a time, there was a`, llama.cpp emits
  actual IDs `3777,15019,58763,6725`, while scalar, NEON, reference-attention
  Metal, and fully resident Metal all emit `3777,95704,6725,8254`. Moving the
  CPU-attention boundary across layers 5 through 40 does not change that local
  sequence, which rules out Metal attention placement as its cause. The default
  reference-attention handoff measures 22.15 tok/s versus the retained
  llama.cpp result of 30.34 +/- 1.07 tok/s. Fully native
  attention reaches 29.63 tok/s but produces an incorrect first token, so it
  remains opt-in and is not parity evidence. Profiling attributes nearly all
  of the default gap to ten forced command-buffer completions per token; CPU
  attention math itself is below 1 ms per layer. Diagnosis found that ordinary
  Metal Q8_0 Q/K/V projections
  differed from the active ARM block-Q8 path by up to 0.116. Transformer
  emission now composes the model-owned reference-attention request with the
  quant-owned block-Q8 activation capability for Q, K, V, and output
  projections. On layer 3 this reduces Q/K/V differences to at most about
  3.8e-6 and the complete attention output from 0.0423 max error to 1.31e-6.
  The remaining native end-to-end divergence is amplification across multiple
  full-attention blocks. Backend capability reporting now distinguishes
  reference recurrent execution from reference full attention, while model
  requirements remain in `BnModelArchOps` and quant activation behavior remains
  in the quant registry. No model-family condition was added to transformer
  emission or Metal lowering.
  After aligning the shared AArch64 SSM SiLU and MoE router-softmax contracts,
  a real-device `BN_GPU_FORCE_GRAPH=1` check with backend-resident MoE and the
  model-requested CPU reference-attention handoff emits the focused llama.cpp
  IDs `3777,15019,58763,6725` exactly. A sustained run terminated at EOS after
  68 requested tokens and measured 29.37 tok/s; adjacent llama.cpp Metal
  `tg128` measured 30.89 +/- 0.02 tok/s, ratio 0.951. This is not a fully
  resident attention result: profiling shows ten CPU-attention handoffs per
  token. Enabling Metal's generic reference-attention capability remains a
  diagnostic only; a 2026-08-09 rerun sampled
  `225696,205405,159874,92657` instead of
  `3777,15019,58763,6725`. Reference-attention stage masks `0`, `1`, `2`, `4`,
  and `7` all produce the same wrong first ID, ruling out the score, softmax,
  and combine substitutions as the token-level source. A fallback-boundary
  sweep shows that even layer 3 as the only native attention block changes the
  first ID to `13`; keeping CPU reference attention from layer 3 restores
  `3777`. Direct prompt-cache comparison starts with a layer-3 key max error of
  about `0.14` and then grows to double-digit errors in later attention blocks.
  Local prepared-Q8 Q/K/V comparison can reduce a single layer to roughly
  `1e-6` error, but that is not sufficient for the recurrent prompt trajectory.
  The attempted paired-lane Metal Q8 reduction was removed because it did not
  restore IDs and reduced short-run speed. Earlier checks performed without device access were
  CPU fallbacks and are not Metal evidence. The remaining roughly 5% sparse
  Metal gap is still open and must be addressed inside backend attention
  correctness/scheduling, without a model-family or quant-format routing
  exception.
  A later adjacent sustained rerun measured 25.28 tok/s at eight threads versus
  llama.cpp Metal at 27.02 +/- 0.25 tok/s, ratio 0.936. A runtime-only thread
  sweep reached 26.73 tok/s over 64 tokens at ten threads, but fell to 22.86
  tok/s over 128 tokens after repeated large-model runs; four and six threads
  measured 23.64 and 21.69 tok/s. This is not stable evidence for changing the
  default. Detailed production-path profiling shows each CPU attention block
  spends roughly 0.35 ms in QKV/GQA/output math and about 2 ms waiting for the
  preceding three-layer recurrent Metal chunk. Unified-memory activation and
  KV writes are only a few microseconds, so removing those copies would not
  close the gap. Thread count remains a measured runtime tuning axis rather
  than a model or quant policy.
  Matching llama.cpp's paired-lane ARM Q8_0 reduction makes the first divergent
  recurrent QKV projection at layer 1 bit-exact and improves the focused Qwen3
  0.6B decode result, but the sparse sampled-token divergence remains later in
  the shared trajectory. A prior four-block Q8_0 accumulation-order candidate reduced the layer-39 local
  attention residual error from about 1.35e-3 to 1.43e-6 and retained 64/64
  strict IDs on the dense Qwen3 Q8_0 fixture. It nevertheless reduced the
  sparse default gate from 32/32 to 1/32 generated IDs because the same generic
  Q8 kernel also participates in the hybrid recurrent trajectory. The candidate
  was removed; local operation error is not accepted over contradictory
  end-to-end token evidence.
  A subsequent sampled-ID audit invalidated the retained 32/32 claim and first
  exposed `3777,95704,6725,8254` on that prompt. Boundary sweeps ruled out Metal
  attention placement, and forced checkpoints localized independent shared CPU
  numerical mismatches: Q8_0 paired-lane reduction, NEON SSM/MoE SiLU, and the
  full-router NEON softmax reduction. Those generic fixes now produce the
  llama.cpp sequence on CPU and Metal with resident MoE plus reference-attention
  CPU handoffs. The broader CPU gate is
  30/32 actual IDs, so the complete sparse sampled-ID gate remains open even
  though this focused regression is resolved.

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

### GPU Graph IR Cleanup

Goal: introduce an internal, backend-neutral GPU graph-value IR that models
transformer GPU work as semantic values and ops before lowering to backend
shader commands. `BnGPUOp` should become a backend-private command format, not
the structure transformer GPU emission builds directly.

- [x] **Add semantic graph IR types** — define graph values, aliases, multi-output ops, dependency metadata, and fallback reasons without shader IDs, fixed activation slots, or `p[8]` command parameters.
- [x] **Build graph helpers for simple ops** — add builder helpers for RMSNorm, copy, residual add, activation, and logits so emitters can append semantic ops and return value IDs.
- [x] **Lower graph IR to current shader commands** — add a compatibility lowerer from the graph IR to the existing `BnGPUOp` command array so behavior can stay unchanged during migration.
- [x] **Move GPU emitters onto the graph IR incrementally** — GPU emitters now build semantic graph IR before lowering to `BnGPUOp` for the current command surface (RMSNorm, logits, copy, residual add, activation, matvec, dense fused gate/up, QKV/SSM/MoE split projections, RoPE, flash attention, GQA scores/softmax/combine, SSM conv/L2 norm/alpha-beta/delta/gate, bias add, weighted add, residual RMSNorm, per-head RMSNorm, deinterleave Q).
- [x] **Make backend command ABI private** — `gpu_graph.h` now exposes the semantic graph IR instead of the shader command ABI, transformer GPU declarations moved out of `include/`, and project code keeps `gpu_shader_ir_internal.h` under `src/` for backend lowering/execution compatibility.
- [x] **Centralize lowered command submission** — transformer GPU orchestration now calls a submit helper for lowered command finalization/execution and named activation sync helpers for CPU fallback boundaries instead of invoking backend shader execution/read/write slots directly.
- [x] **Start persistent graph emission** — transformer GPU orchestration now owns a `BnTransformerGPUEmitContext` with a pending `BnGPUValueGraph`; orchestration-level RMSNorm/logits append semantic graph ops and lower them together only at legacy boundaries or submit.
- [x] **Route simple op emission through the persistent graph context** — copy, residual add, activation, matvec, and fused gate/up helpers now share context append logic, with legacy wrappers lowering through that context for compatibility during block-emitter migration.
- [x] **Route complex helper emission through the persistent graph context** — split matvec, RoPE, flash/GQA attention, SSM, utility, and legacy RMSNorm/logits wrappers now use context append/lower logic; `gpu_emit.c` no longer creates stack-local value graphs for helper lowering.
- [x] **Convert dense FFN block emission to the persistent graph context** — dense FFN now appends directly to `BnTransformerGPUEmitContext` from GPU orchestration.
- [x] **Convert QKV and attention block emission to the persistent graph context** — attention-layer Q/K/V projection, RoPE, flash/GQA attention, gated-Q activation, output projection, and residual RMSNorm now append directly to the orchestration-owned graph context.
- [x] **Convert SSM block emission to the persistent graph context** — SSM projection, conv/L2/alpha-beta/delta/gate, output projection, and residual RMSNorm now append directly to `BnTransformerGPUEmitContext`.
- [x] **Convert MoE block emission to the persistent graph context** — expert gate/up/down projections, expert weighted accumulation, shared-expert accumulation, residual add, and next RMSNorm now append directly to `BnTransformerGPUEmitContext`; transformer GPU orchestration no longer has legacy graph-lowering boundaries.
- [x] **Remove transformer emitter shim APIs** — transformer internals no longer expose `BnGPUOp *ops, int *n` block emitters; tests and orchestration use `BnTransformerGPUEmitContext` directly, with `BnGPUOp` retained only as the lowered backend command buffer.
- [x] **Deduplicate backend dependency logic** — Metal and WebGPU now share backend-private shader access-mask metadata for pass/barrier decisions instead of duplicating read/write switch tables.
- [x] **Reduce transformer GPU coupling** — block emitters now receive a compact `BnTransformerGPUEmitResources` for backend capabilities and handles instead of separate GPU/backend plumbing; MoE keeps explicit model/session inputs only where expert-cache bridge ownership still requires them.
- [x] **Narrow MoE GPU emission inputs** — expert-buffer resolution now produces `BnGPUMoEResources`; the MoE graph emitter consumes routed expert buffers and weights without taking the full model/session pair.
- [x] **Move MoE GPU resource resolution out of transformer emission** — the model/session-facing MoE bridge resolver now lives in `gpu_moe_bridge`, so `gpu_emit.c` stays focused on graph construction.
- [x] **Narrow dense FFN GPU emit resources** — dense FFN emission now takes pre-resolved backend handles and weight buffers via `BnTransformerGPUDenseFFNResources` instead of doing backend lookups through `BnTransformerGPUEmitResources`.
- [x] **Narrow QKV/attention/SSM GPU emit resources** — QKV, attention, and SSM block emitters now take pre-resolved resource structs for backend handles and weight buffers instead of doing backend lookups through `BnTransformerGPUEmitResources`.
- [x] **Narrow MoE shared-expert emit resources** — MoE graph emission now takes `BnTransformerGPUMoESharedResources` alongside `BnGPUMoEResources`, and the generic `BnTransformerGPUEmitResources` wrapper is gone from transformer GPU code.
- [x] **Move transformer GPU resource resolution out of orchestration** — QKV, attention, SSM, dense FFN, and MoE shared resource resolvers now live in `src/transformer/gpu_resources.c`, leaving `gpu.c` focused on control flow.
- [x] **Narrow remaining transformer GPU handle lookups** — output/logit, next-norm, initial-norm, and per-layer validation handle resolution now lives behind focused resource helpers, so `gpu.c` no longer reaches directly into backend model handles.
- [x] **Split transformer GPU orchestration fallback policy** — GPU-forward eligibility, request/model validation, top-level fallback reasons, and top-level resource preflight now live in `src/transformer/gpu_policy.c`; `gpu.c` keeps graph orchestration and CPU fallback sync boundaries.
- [x] **Consolidate transformer GPU CPU fallback blocks** — repeated GPU flush/read CPU block/write/re-normalize sequences for SSM and MoE fallback now live in `src/transformer/gpu_fallback.c`, while `gpu.c` keeps the fallback placement decisions.
- [x] **Move transformer GPU logits fallback mechanics out of orchestration** — the oversized-logits GPU flush/read/CPU matvec path now lives in `src/transformer/gpu_fallback.c`; `gpu.c` keeps only the size decision and result flow.
- [x] **Move transformer GPU operation-budget policy out of orchestration** — graph capacity sizing now lives in `src/transformer/gpu_policy.c`, so `gpu.c` no longer owns approximate per-block op-count constants.
- [x] **Move transformer GPU binding-limit policy out of orchestration** — the max storage binding default and oversized-logits decision now live in `src/transformer/gpu_policy.c`, so `gpu.c` only chooses between logits fallback and logits emit.
- [x] **Move transformer GPU fallback debug reporting out of orchestration** — fallback debug gating and one-shot stderr reporting now live in `src/transformer/gpu_policy.c`; `gpu.c` only reports the reason and unwinds the emit context.
- [x] **Move transformer GPU flush mechanics out of orchestration** — pending-op no-readback flush now lives in `bn_transformer_gpu_emit_context_flush`; `gpu.c` and CPU fallback helpers no longer duplicate execution checks.
- [x] **Move transformer GPU rejection cleanup out of orchestration** — fallback rejection cleanup now uses `bn_transformer_gpu_reject_forward`, removing the local `GPU_REJECT` macro from `gpu.c`.
- [x] **Move transformer GPU uncached MoE buffer cleanup out of orchestration** — uncached expert GPU handles are tracked in `BnGPUMoETemporaryBuffers` and released through the MoE bridge, so `gpu.c` no longer destroys MoE buffers directly.
- [x] **Hide lowered shader command storage from transformer-facing APIs** — `BnTransformerGPUEmitContext` now stores lowered command state opaquely, backend-session command-buffer access returns `void *`, and the GPU backend `execute` vtable no longer exposes `BnGPUOp`.
- [x] **Move shader lowering headers under `src/`** — graph lowering, quant-to-shader op selection, and the lowered shader command ABI now live outside public include space; `gpu.c` and CPU fallback helpers use semantic emit helpers instead of fixed shader slots.
- [x] **Move transformer GPU MoE fallback selection out of orchestration** — MoE FFN fallback placement now flows through `BnTransformerGPUMoEFFNFallbackPolicy`, with backend-matrix coverage preventing `gpu.c` from calling the raw CPU fallback predicate directly.

### Cohesion and Coupling Debt

The latest architecture review found that the largest ownership leaks are fixed,
but several internal seams still violate high-cohesion/low-coupling goals. These
items should be addressed before adding substantial new backend, model-family,
or quant-format surface area.

- [x] **Split `transformer_internal.h` into narrower internal boundaries** — it still includes model, quant, math, SIMD, GPU planning, CPU execution, KV/TurboQuant helpers, logits, plan structs, and RMSNorm backend symbols. Split into focused headers such as `transformer_plan_internal.h`, `transformer_kv_internal.h`, `transformer_cpu_internal.h`, and a tiny RMSNorm backend header.
- [x] **Split `quant_internal.h` by quant concern and backend** — it still centralizes all context structs, every scalar/NEON/AVX2/WASM kernel prototype, dispatcher helpers, and inline unpack helpers. Split into `quant_ctx.h`, `quant_kernels_scalar.h`, `quant_kernels_neon.h`, `quant_kernels_avx2.h`, `quant_kernels_wasm.h`, and `kquant_helpers.h` or equivalent.
- [x] **Hide model runtime/backend strategy behind opaque subobjects** — `BnModel` is smaller but still exposes mapped file lifetime, thread pool, weight arena, MoE I/O, expert fd, backend state, and TurboQuant state. Move these into opaque `BnModelRuntime` / `BnModelIO` style subobjects or make `BnModel` opaque to public consumers.
- [x] **Move CPU prepared-layout generation out of `model.c`** — native-quant repacking and related prepared-weight registration are backend/quant layout concerns, not canonical GGUF model loading. Move them under backend layout or quant preparation.
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
  - Metal reference-attention work now keeps prepared native-quant attention
    behind its own backend capability
    (`BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION`). The generic prepared-quant
    capability remains available to FFN/per-layer paths and no longer implies
    that attention projections are coherent. This removed a resident sparse
    Gemma NaN without coupling quant selection to the Gemma/reference-attention
    model policy. Metal's private `BN_METAL_REFERENCE_ATTENTION_STAGES` mask
    isolates score/softmax/combine shader substitutions for parity diagnostics.
    On sparse Gemma, masks `0`, `1`, `2`, and `4` all produced the same 2/8
    token prefix, ruling out those three GQA stages as the token-level source.
    A layer-0 QKV comparison instead measured max differences of 0.03225 (Q),
    0.00112 (K), and 0.01360 (V); precise Metal compilation did not change
    those values and was not retained.
  - Sparse Gemma decode currently keeps the complete MoE FFN on CPU because
    Metal does not declare `BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32`. Lifting
    that conservative composition gate activates GPU routing and routed Q4_0
    experts, but changes the first `Hello` token from 3324 to 6305. At layer 0,
    route indices match the CPU reference and routed gate/up activation error is
    below 0.00043, while the composed routed state reaches 0.368 max error. The
    missing parity-qualified work is a backend-resident dense residual branch
    with `ffn_post_norm_1`, `ffn_post_norm_2`, and final post-norm resources;
    the whole-FFN CPU gate remains in place until that capability is implemented
    and passes token parity.
    After the shared reference-activation fixes, a strict `Once upon a time`
    check matches 4/4 sampled token IDs on the conservative path. A 16-token
    M1 Max run measured 18.66 tok/s versus llama.cpp Metal `tg16` at
    40.06 +/- 9.04 tok/s. Runtime MoE accounting reported about 633 ms of CPU
    MoE work across those 16 tokens, so the parity-qualified dense-residual
    capability remains the dominant sparse speed gap.
    Re-testing the existing generic Metal dense-residual graph after the shared
    activation fixes improved it substantially: three focused prompts plus
    `Once upon a time` matched 4/4 IDs, and generation reached 24.64 tok/s
    (32% above the conservative path). `Hello` still diverged after 2/4 IDs,
    so Metal does not advertise the capability yet. A three-step coherence
    trace shows CPU and Metal both generate `upon-a` for that prompt while
    llama.cpp generates `upon-the`; route indices and weights agree at layer 0.
    The remaining `Hello` gap is therefore shared Gemma/MoE semantics rather
    than a backend-only routing failure.
    A later CPU-route/resident-expert diagnostic separates routing from backend
    execution without adding a model-family selector. With Metal advertising
    the dense-residual low-bit capability only for that opt-in run, the
    `Once upon a time` prefix again matched llama.cpp (`236764,3622`), CPU MoE
    time fell to zero, and the Metal expert cache recorded 571 hits and 869
    misses. Throughput was only 8.49 tok/s. Per-operation profiling identified
    the dominant cost as the existing CPU-attention policy: each of 30 layers
    forces a host/device handoff that flushes roughly 50 queued residual/MoE
    operations and costs about 10-15 ms. The next sparse optimization target is
    therefore a parity-qualified generic resident-attention composition, not a
    quant- or model-specific routed-kernel selector.
    Backend-owned alternate output-projection layout support subsequently let
    the existing reference-attention graph remain resident instead of rejecting
    its prepared `wo` handle. The selected-expert graph then measured 8.09
    tok/s over 16 cold tokens and 10.51 tok/s over a warmed 64-token run with an
    8 GB expert cache. It retained the strict short prefix but diverged later in
    the trajectory, so it is not deep-parity qualified. The aggregate resident
    expert layout reached 35.52 tok/s, much closer to llama.cpp, but reproduced
    the known first-token failure (`6491` instead of `236764`). An attempt to
    give the selected graph generic borrowed expert buffers produced zero/pad
    logits and was removed: expert pointers do not by themselves prove a
    backend-compatible borrowed layout. Future zero-copy work must use an
    explicit backend-owned aggregate-buffer view contract. Quant eligibility,
    model semantics, and runtime layout selection remain independent axes.
    A follow-up established why a plain aggregate subview is insufficient:
    Metal's repacked Q4_0 matrix stores the complete scale plane before the
    complete nibble plane, so one expert is not a contiguous byte range. A
    diagnostic repacked aggregate plus byte subviews produced pad/zero logits
    and was removed. Preloading all 3,840 independently repacked layer-expert
    entries preserved the 8/8 prefix and reached 100% cache hits, but memory
    pressure reduced throughput to 2.56 tok/s. Changing the native aggregate
    Q4_0 SIMD reduction order also left the wrong first token unchanged. The
    next layout must preserve per-expert repack boundaries inside a compact
    backend-owned expert array; raw views and full duplicate caches are both
    rejected directions.
    A Metal-private native-GGUF Q4_0 matvec path now provides an explicit
    zero-copy selected-expert layout contract. The backend validates block
    geometry and byte bounds before wrapping immutable mmap storage, while the
    generic MoE bridge only consumes the optional backend capability. The
    focused sparse Gemma prompt matches llama.cpp for 8/8 generated token IDs.
    A byte-locale 32-token comparison matches 25/30 retokenized llama.cpp IDs.
    The apparent early stop in the longer run is the generic repeated-ngram
    detector, and a CPU run follows the same post-token-25 trajectory as Metal.
    The remaining deep-parity defect therefore belongs to shared Gemma/model
    semantics rather than this Metal Q4_0 layout; the path is not yet deep-
    parity or speed qualified. Quant eligibility, model semantics, and backend
    runtime layout remain separate decisions, and the backend architecture
    matrix passes with this composition.
    Backend-reported cache charges now distinguish logical tensor bytes from
    resident handle bytes, and the generic cache independently caps capacity at
    the policy-provided total expert count. This raises the zero-copy sparse
    Gemma cache from 1,283 to all 3,840 possible layer/expert entries without a
    model or quant selector in cache code. A cold 16-token sample improved from
    24.57 to 28.79 tok/s with the same token trajectory; misses remain cold
    wrapper creation and no longer force later eviction.
    Eagerly preloading all zero-copy handles reached 100% cache hits and zero
    streamed expert bytes but reduced the same cold sample to 24.64 tok/s, so
    eager preload was not retained. Replacing the aggregate routed Q4_0
    activation-quantized dot with the selected path's float-input native dot
    also left its first token at `6491` instead of `236764`; matching the
    selected path's float4 block order and completing each expert-down SIMD
    reduction before weighted accumulation did not change that result either.
    Both shader changes were removed. The aggregate parity defect is therefore
    outside those quant arithmetic choices. A profiled short selected-expert
    run reached 36.74 tok/s. The frame counter reaches roughly 150 while
    prefilling the five-token prompt, which is about 30 route-dependent backend
    executions per token, not 155 executions per token. Reducing those
    synchronization points is the remaining runtime optimization; it must
    preserve the current backend layout and model-policy boundaries.
    A layer-0 position-4 aggregate trace further localizes its parity failure:
    CPU and GPU select the same eight experts with closely matching weights,
    per-expert mids stay below `1.2e-6` max error, and the routed branch is
    within `0.00169`, while the dense residual branch reaches `0.6174` max
    error and the combined layer state reaches `0.9847`. Disabling prepared
    small-dense native quantization leaves the wrong first token unchanged.
    The next correction therefore belongs to generic dense-residual backend
    composition, not routed expert addressing, quant selection, or model-family
    policy. Focused tests now cover native borrowed-layout support queries,
    backend cache charges, and total-entry cache caps.
    A later dense-residual trace found that the non-native FFN-down operation
    was initially handed Metal's prepared native-quant layout and rejected.
    Backend-owned alternate quant-layout plumbing now lets that graph execute,
    but the ordinary Metal Q4-down path then failed first-token parity on
    `Once upon a time`. The dense-residual capability remains unadvertised;
    the remaining acceleration work is a parity-qualified, quant-owned Q4
    reduction rather than a model-family or transformer routing exception.
  - Dense Gemma per-layer-input FFN composition now requires the independent
    backend capability `BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN` in
    addition to generic prepared native-quant support; native FFN-down has a
    still narrower `BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN` bit.
    Model architecture declares only the per-layer-input semantic and
    transformer policy composes it with these backend capabilities. Metal now
    claims the gate/up capability after the all-42-layer path matched 8/8
    sampled IDs with both batch and decode-only prefill. It does not claim the
    narrower down capability: native down first destabilizes the quantized
    per-layer adapter input at layer 23 and the full path fails after 1/8 IDs.
    The Metal Q8 activation kernel now preserves the quant contract's FP16
    round-trip for stored block scales and is compiled with safe math. A layer-0
    diagnostic then measured zero differing Q8 bytes, zero scale error, and zero
    FFN-down output error against CPU. Reference GELU lowering now also preserves
    the CPU contract's FP16 input and output rounding; this reduced the focused
    layer-0 per-layer-state maximum error to `7.629e-6`. The full multi-token
    native-down path still fails after 1/8 sampled IDs, so that local numerical
    result is not promoted into a backend capability. The remaining session
    drift is not a reason for a Gemma, Q4_0, or Metal branch in transformer
    policy. In a same-prompt 64-token M1 Max A/B, all-layer native gate/up with
    conservative down measured 23.30 tok/s versus 21.93 tok/s for the fully
    conservative path, a 6.2% gain. A warmed llama.cpp Metal `tg64` baseline
    measured 35.97 +/- 0.49 tok/s, so dense Metal remains a speed-parity gap.
    A backend-owned ordered-SIMD down reduction subsequently matched the ARM
    NEON sequential block/FMA contract and reached 24.66 tok/s on the same
    prompt. It passed `Hello` at 8/8 IDs but failed the other seven strict
    prompts after 1-2 IDs, demonstrating that CPU-exact reduction alone does
    not reproduce llama.cpp's Metal trajectory. The down capability therefore
    remains unadvertised.
    A later profile showed the conservative emitter rejecting ordinary Q4_0
    down operations backed by prepared-native buffers and preserving parity via
    fallback. Generalizing the backend-owned alternate quant layout removed the
    rejection, but the dense E4B `Once upon a time` gate fell to a `1/4`
    sampled-ID prefix. That routing experiment was reverted: layout correctness
    is necessary but does not qualify ordinary GPU down arithmetic. Separating
    the Q8-activation request from prepared-layout selection then exercised the
    intended non-prepared Q4_0 x Q8 Metal kernel, but the gate remained `1/4`;
    the mismatch is therefore in that kernel's reduction contract rather than
    accidental FP32 activation selection. That experiment was also reverted.
    A model-independent Metal graph test now exercises the non-prepared Q4_0 x
    Q8 path at the real 10,240-column down width. The fast kernel stays within
    `3.3e-7` of CPU, while a backend-owned sequential reference kernel matches
    CPU bit-for-bit. Enabling the reference kernel through the generic
    dense-residual capability restored dense E4B `4/4` strict token parity and
    measured 27.03 tok/s, but the same capability made sparse E26B diverge on
    its first token (`6491` versus llama.cpp `236764`). The capability was
    withdrawn instead of adding a dense/sparse model-family exception; it must
    pass the aggregate dense and sparse gate before Metal advertises it.
    A resident-path trace localized the sparse failure further. At layer 0,
    GPU and CPU select identical experts with route-weight differences below
    `1.5e-6`, and the routed expert sum differs by only about `0.0016` max.
    Comparing the complete CPU MoE computation against the GPU graph at prompt
    position 4 shows about `1.03` max layer-state error and `0.65` max error in
    the dense residual output; at layer 15 the corresponding local errors are
    `0.14` and `0.32`. Accelerating only layer 0 retains the correct first
    token, while accelerating layers 0-14 does not, confirming cumulative
    arithmetic drift rather than a routing or single-layer failure. The
    tokenwise coherence harness without the CLI's resident Metal setup does not
    cover this failure mode and must not be used alone to qualify the
    capability. Forcing private copies of the 13.8 GB expert fixture exceeded
    practical residency and produced invalid zero-token output, so mmap-backed
    resident execution remains the authoritative sparse gate.
    A model-independent two-expert Metal graph test now covers the complete
    native Q4_0 routed FFN contract (gate/up, GELU, down, and route weighting)
    without enabling the production capability. It is bit-exact against the
    corresponding CPU Q8-block contract, ruling out basic expert addressing,
    activation lowering, or
    kernel lowering as the source of the CLI divergence. Three backend-only
    follow-ups were rejected: sequential Q4/Q8 accumulation, float-input Q4,
    and deterministic route-ordinal resident slots all retained the wrong
    first sparse token (`6491`). Float input reached about 39.6 tok/s before a
    per-expert ordered-down variant regressed to 23.4 tok/s; neither earned the
    aggregate capability. Debug comparisons must refresh CPU routing before
    comparing selected expert slots, because GPU-only routing otherwise leaves
    request-local CPU route state stale and creates a false permutation signal.
    The same
    16-token conservative profile measured 29.14 tok/s on the current M1 Max
    run, still below the retained llama.cpp Metal baseline.
    CPU-routed resident MoE no longer implicitly advertises
    `BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32`: resident expert routing and the
    dense-residual graph are independent backend capabilities, and the latter
    has not passed the aggregate dense/sparse parity gate. Focused transformer,
    GPU-backend, and architecture-matrix tests pass with that promotion
    removed. A first hybrid E26B run under severe Metal residency pressure
    returned an invalid pad token, so it is not accepted as parity or speed
    evidence; the retained selected-expert path remains the authoritative
    sparse result until a clean hybrid run is available.
    A backend-private diagnostic capability now makes the withheld aggregate
    graph reproducible without changing production policy. At layer 0,
    position 4, the ordinary float-input Q4_0 dense down projection differed
    from CPU by `0.08081`; post-norm amplified that to `0.6545`. Reusing the
    quant-owned Q4_0 x Q8 activation contract made the raw down projection
    bit-exact and reduced the post-norm maximum to `6.1e-5`. The pre-down
    dense activation was already within `1.9e-5` max (`6.1e-7` RMS).
    The remaining aggregate error is the routed Q4_0 down/weight accumulation:
    its raw branch differs by about `0.00159`, which later normalization
    amplifies to about `0.345`. Uploading CPU-reference route weights from the
    same resident input did not reduce that error, ruling out router softmax
    precision. The next correction belongs to the backend routed-down
    reduction contract. The diagnostic capability, quant comparison, and
    route override remain opt-in and do not introduce model-family selection.
    A later end-to-end state trace corrected the local-comparison diagnosis.
    On the five-token `Once upon a time` prompt, the fast prepared Metal path
    differs from the full CPU logits-input state by `2801.57` max (`150.00`
    RMS), even though standalone projections and same-input operation checks
    pass. Prompt KV divergence is small in attention layers 0-1, then grows
    sharply from layer 2; GPU layer 0 followed by CPU layers 1-29 retains token
    `236764`, while GPU layers 0-1 followed by CPU selects `528`. Expert IDs
    agree throughout the non-perturbing route trace, so the next gate must
    compare full layer-boundary states rather than infer trajectory coherence
    from same-input kernel checks.
    The trace also exposed a backend-independent output bug: GPU readback
    logits skipped the model-declared final logit softcap. CPU and GPU now call
    the same logits-policy transform after quant refinement. This preserves the
    model/quant/runtime axes: model policy supplies the scalar, quant refinement
    remains format-owned, and transformer orchestration applies the transform
    without a backend or model-family branch. Focused transformer tests, the
    Metal build, `backend_matrix.sh`, and `git diff --check` pass.
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
