# GLM raw-F32 routing and complete streamed CUDA MoE

The complete block-three MoE sublayer now composes F32 routing, biased sigmoid
selection, the shared expert, bounded routed-expert streaming, ordered reduction
and BF16 shared-output addition. The model anatomy supplies activation bounds
and routing metadata. This checkpoint uses 288 experts, top eight, a normalized
routing scale of 2.5, Q4_K gate/up, Q5_K down and one Q8_0 shared expert, with
clamp ten. This qualification is on SM120 CUDA with the designated GPU; it
does not admit these new operators on other GPU architectures or CPU/Metal.
The ordinary whole-model loader remains guarded.

`BnBackendBF16MoEWeights` owns backend router/bias/shared buffers and borrows the
immutable CPU expert map. Router weights retain their source F32 values. The
GPU resource occupies 31,458,432 payload bytes; shared preparation temporarily
uses 17,825,792 host bytes. Routed expert payloads are streamed rather than
uploaded at preparation. Ownership and finite source checks precede uploads;
failed preparation frees partial resources and leaves the destination empty.

`bn_transformer_bf16_moe_begin` validates all 19 views together, queues routing,
reads 1,172 bytes of grouping metadata, then queues the shared branch and input
quantization. `bn_transformer_bf16_moe_submit` acquires event-protected banks,
loads CPU expert payloads into pinned leases, submits grouped computation and
finishes reduction/addition. BUSY preserves continuation for the same work.
Execution or source-I/O errors poison that work; callers discard the request.
These helpers make no logical session-position commits. The CPU map/model must
outlive borrowers; CPU source/cache execution is serialized, while mutable
host/work state belongs to the request.

Prefill borrows existing RAM-cache entries without admission. Decode admits
misses. The qualification uses a 128 MiB RAM cache, enough for the first token's
eight selected experts. Cold decode loads 121,634,816 bytes; immediate warm
decode and decode after the larger prefills have eight hits and zero source
reads. Both still upload 121,634,816 bytes to the GPU. The GLM model-owned VRAM
expert-cache profile is not implemented by this checkpoint.

Two capacity-three bank slots reserve 122,683,392 payload bytes in each memory
domain. Activation/quantization scratch, descriptors, CUDA context and allocator
overhead, mapped source pages and OS cache are additional. The harness opens
all GGUF shards even in pread mode; this does not establish a whole-process RSS
bound or qualify an unmapped production model loader.

## Reference contract

The installed vLLM 0.33.0 source commit is `4ba787ec9`. Its grouped router uses
`0.5*tanh(0.5*x)+0.5`, biased selection with lower-ID ties, unbiased weights,
ordered F32 normalization and rounded multiplication by the route scale. Its
single-group implementation changes unbiased-score recovery above 1,024
tokens: smaller batches retain the original sigmoid; larger batches subtract
the bias from the rounded selection score. The native path preserves this
rounding boundary. F32 projection uses pedantic SGEMM with TF32 disabled.
This qualifies a raw-GGUF-F32 router profile, not an unspecified HF loader's
parameter casting behavior.

Two installed original-plugin limitations prevent a whole-plugin equivalence
claim. Its whole-MoE method omits the checkpoint clamp configuration. Also,
`csrc/gguf/moe.cuh` rejects matrix expert IDs above 255, silently dropping some
selected GLM experts. The original unsupported captures remain preserved.
The independent reference composes unchanged original operators, supplies the
checkpoint clamp, and partitions only the expert domain into `[0,256)` and
`[256,288)` for matrix calls. It remaps local IDs, preserves original token/slot
geometry and precision thresholds, and bit-selects output rows without an
arithmetic merge. Original routing and global IDs remain unchanged. Vector
calls use the original full 288-expert domain.

An audit of the original four cases confirms unchanged hashes for inputs,
routing and the shared branch, and exact preservation of every routed word for
IDs below 256. Changes occur only in previously unsupported high-expert rows.
Neither the native path nor original kernels were changed to omit those rows.
Earlier supplied-route goldens and tolerances remain unchanged. Any older
whole-plugin claim for matrix batches with more than 256 experts needs its own
expert-domain audit before being reused. The earlier Flash Next comparison
already documents and qualifies a separate corrected extension; its evidence
should retain those patch hashes. That corrected extension was audited here
using the full 288-expert domain: every captured word in all six GLM cases is
identical to the unchanged-operator expert-partition composition. This bounds
the earlier metadata-index concern for these cases without generalizing it to
other geometries or changing any golden.

## Reproducing the checkpoint

Build optional targets `test_cuda_glm_router` and `test_cuda_glm_moe_layer` with
the CUDA configuration. They take the first checkpoint shard and a reference
directory. The latter accepts `--pread-only` to omit its mmap execution mode.
`test/reference_glm_moe_layer.py --model FIRST_SHARD --out DIRECTORY` captures
router/shared cases; add `--experts` for the complete MoE cases. The reference
requires the original vLLM GGUF CUDA extension and records source/binary hashes.
`--experts --corrected-expert-domain` additionally permits the separately
qualified corrected extension, guarded by its exact binary hash, to repeat
the full-domain cross-check.
Tests are opt-in and require the actual checkpoint; ordinary unit tests use
synthetic data.

The router/shared reference covers token counts 1, 2, 6, 7, 9, 64, 65, 129,
1,024, 1,025 and 2,048. The complete MoE reference covers 1, 9, 65, 129, 1,024
and 1,025 with actual checkpoint-selected experts. Comparisons include F32
logits/weights, IDs, shared projection/activation/output, routed activation/up/
down, each weighted output and final sums. Native inverse maps restore the
reference slot order. Synthetic checks distinguish ties, finite sigmoid
underflow, nonfinite fallback and malformed metadata on both sides of the
1,024-token dispatch boundary. Resource checks cover budgets, source-map
mismatches, finite F32 values and upload rollback. The complete harness also
rejects aliased views, wrong layer/counts and invalid begin/submit transitions.
CPU-only state probes verify BUSY continuation, poisoned-work rejection, host
source-buffer restoration/cancellation on I/O failure and retained progress
when a later bank submission fails.

The final replay passes 527,733,360 exact mixed BF16/F32/I32 word comparisons
across six mmap and nine pread schedules. Router/shared replay separately
passes 46,182,720 exact words. The warning-free clean build, standard suite,
full CUDA suite and diagnostic isolation pass, as do the existing bank/cache,
165,888,000-word supplied-route and 221,074,320-word dense-prefix regressions.
Host ASan/UBSan and CUDA memcheck/synccheck report no errors for their scoped
resource/state and actual-checkpoint replays.

A warm single-token complete-sublayer trace records 31 kernels, 121,634,816
weight-upload bytes, 96 descriptor-upload bytes and the 1,172-byte metadata
read. No activation transfer or allocation API occurs in the captured interval.
Three copy-ready device waits protect the three bank submissions. Two explicit
host stream fences complete routing metadata and final output respectively.
CUPTI additionally labels the successful nonblocking event query as an event
synchronization and records an unattributed final context synchronization;
these records are retained in the evidence. The captured warm output passes
117,040 exact comparisons (351,120 including the two uncaptured warm-up steps).
This is an observational transfer/dependency trace, not a throughput benchmark.

Qualification results and source hashes are recorded in
[the machine-readable record](glm53-cuda-moe-layer.json). This is a complete MoE
sublayer checkpoint. Indexed MLA/indexer, whole-model resource and session
binding, long-context generation, GLM VRAM-cache policy, CPU qualification,
matched resident vLLM performance and serving tail latency remain open. The
full modern-model goal remains active.
