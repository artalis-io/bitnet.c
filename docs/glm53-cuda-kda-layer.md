# Checkpoint-bound GLM KDA attention sublayer

The backend can now prepare and execute the complete KDA attention sublayer
from the actual GLM-5.3-Flash GGUF checkpoint. Its entry is BF16 hidden states
after the outer input normalization; its exit is the BF16 attention output
projection. Outer normalization, mHC and residual composition are outside this
sublayer. Ordinary `model_load` still rejects GLM until the complete topology
has a dedicated binder and executor.

## Weight preparation and ownership

`bn_backend_layout_bf16_kda_init` consumes an immutable `BnModelAnatomy` layer
plan and the merged GGUF directory. It checks geometry, types, payload bounds
in the owning shard, finite F32 parameters, and nonpositive stored decay values
before GPU uploads. Shared anatomy value types now live in `model_anatomy.h`;
family planning APIs remain in `model_arch.h`.

The qualified profile has Q8_0 projections, 128-wide heads and gate ranks,
four convolution taps, and F32 convolution, decay, bias and output norm weights.
Preparation merges Q/K/V/beta/F_A/G_A raw rows in that order. Keeping them in one
projection preserves the original GGUF dispatch threshold: wide projections
switch from MMVQ to MMQ after two tokens, while smaller projections switch after
six. F_A and G_A remain separate BF16 activation materializations before F_B and
G_B; the low-rank pairs are not algebraically folded.

Eight backend-owned GPU buffers retain only raw checkpoint payloads. Separate
Q/K/V convolution weights become one backend-owned stack. Stored `-exp(A_log)`
values retain their F32 bits, without a log/exp round trip. CPU weights remain
unchanged. The caller frees borrowing requests before freeing these weights;
the GPU outlives both. Preparation is transactional: invalid inputs, budgets or
any upload failure leave the destination unchanged and release partial buffers.

Each tested layer retains 146,653,952 bytes of GPU tensor payload and uses one
reused 108,347,392-byte host staging allocation. CUDA quant-only uploads create
no auxiliary decoded caches. Budgets exclude mapped checkpoint pages and
allocator/context overhead; these numbers are not whole-model working sets.

## Resident execution

`bn_transformer_bf16_kda_layer` queues four Q8 linears, three column splits,
F32-weight convolution, bounded KDA recurrence, and F32-gamma RMS/sigmoid gating.
All intermediates, recurrent states and histories remain resident. Workspace
views are caller/request-owned, exact-sized and preallocated. A8 scratch must
be reserved for `tokens * max(model_dim, heads * 128)` before execution, including
the output projection's input width.

Visible view metadata, ownership and pairwise disjointness are checked before
submission. Opaque slot generations, device storage and reserved scratch remain
backend checks. Backend/device failure can follow partial submission: the caller
discards the failed request, with no rollback promise. Initial state/history are
read-only; the caller commits their next generations after successful completion.
This operation does not advance a model session's position by itself.

## Independent checkpoint qualification

The optional replay uses layers 0 and 44 from all six actual checkpoint shards.
For identical nine-token hidden inputs, it tests full prefill, splits `[2,7]`,
`[3,3,3]`, `[7,1,1]`, `[6,3]`, and nine one-token calls. Histories and recurrent
states start nonzero. Each schedule has its own independent reference because
Q8 MMVQ/MMQ arithmetic can differ across the dispatch boundary.

References call the original GGUF plugin matmul and original GLM/vLLM
convolution, fused recurrent KDA and gated RMS operators. Every schedule is
captured twice and requires exact repeated results. Inputs to raw GGUF matmul
are explicitly contiguous: its C++ quantizers do not consume arbitrary tensor
strides. One-token reference convolution updates its input, so that input is
cloned to preserve the original merged projection snapshot. An audit of this
capture correction confirmed that all weight hashes and all downstream
reference values stayed unchanged; only 22 saved projection snapshots changed.

The native chain consumes only original hidden inputs, history, state and
checkpoint weights. It compares merged projection, F_B/G_B outputs, convolution,
core output, gated output, final projection, F32 state and BF16 history after
every chunk. All 54,216,448 compared words match bit for bit across both layers
and all six schedules. CUDA memcheck reports zero errors. Synthetic tests cover
packing, budgets, shard bounds, invalid types/shapes/parameters, alias preflight,
every upload failure and every execution-step failure. The new preparation and
orchestration implementations also pass ASan/UBSan with leak detection enabled.

The final clean build has zero warnings; standard tests, the full CUDA backend
suite and CUDA diagnostic isolation pass. A warm nine-token layer trace records
14 kernels on one stream, zero memory-copy bytes and no allocation API calls.
There is one explicit final stream synchronization plus an unattributed
context-sync activity after the last kernel; no synchronization occurs between
layer operations. These are residency diagnostics, not statistical latency or
vLLM parity measurements.

To reproduce with the qualified GLM-enabled vLLM/GGUF-plugin environment:

```sh
python test/reference_glm_kda_layer.py --model /path/to/first-shard.gguf --out /tmp/kda-ref
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=/path/to/mkl test_cuda_glm_kda_layer
./test_cuda_glm_kda_layer /path/to/first-shard.gguf /tmp/kda-ref/reference
```

Select the intended GPU through `CUDA_VISIBLE_DEVICES` for capture and replay.
Source, checkpoint and output hashes plus verification results are recorded in
`glm53-cuda-kda-layer.json`.

This qualifies the fused-recurrent precision profile on the tested SM120 GPU.
It does not qualify optimized matrix-based `chunk_kda`, an arbitrary vLLM GGUF
whole-model wrapper, long-context layer behavior, CPU BF16 rounding parity,
indexed MLA, mHC, full GLM inference or vLLM performance parity. Those remain
part of the active full goal.
