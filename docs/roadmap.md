# Roadmap

Development roadmap for bitnet.c.

## Current priorities: modern models and memory-constrained inference

This section sets development priority; the historical phases below record
earlier work and do not establish current model or performance qualification.

### Request-aware admission integrated with bounded KV pools (2026-11-10)

A separate request-aware batch initializer supplies ID, prompt/output lengths
and full trajectory capacity to backend acquisition without changing the legacy
backend struct ABI. The CUDA adapter opts into sized leases and emits every
slot transition; independent validation checks live ownership, idle eviction,
actual capacity charges, retained storage and peak bytes.

Both sized model controls remain exact: 134 cohorts/784 heads and 572 cohorts/
4,132 heads. A fixed-deadline sized diagnostic also matches 573 cohorts/4,136
heads. Retained KV falls from 4 GiB to 96 MiB for twelve requests and 512 MiB for
48 requests. Six controlled timing traces preserve their original geometries
and streams. Standard tests, runtime sanitizers, record recycling past 64 total
requests and legacy model gates pass; clean build has zero warnings.

The paired 48-request medians keep throughput/p99 essentially unchanged, but
p99.9 increases from 95.90 to 113.61 ms. Eight intervals exceed 100 ms per run,
coinciding with new larger-lease admission while decodes are active. Sized mode
remains opt-in; this is not full VRAM or all-tail parity qualification. See
[API, independent checks and tradeoff](qwen38-request-capacity-admission.md).
Next: reduce exposed cold-admission cost and account for non-KV storage, while
retaining the remaining Flash offload/model/backend scope of the full goal.

### Request-sized KV pool capability (2026-11-10)

The dense BF16 pool supports capacity buckets, best-fit idle reuse and bounded
idle reclamation. Active leases are never evicted. BUSY is mutation-free; actual
retained capacities remain charged, including oversized idle reuse. Healthy
evictions and poisoned discards are counted separately. The fixed-capacity API
and its 134-step/784-head independent runtime control remain exact.

Model-free ownership/pressure/failure tests and ASan/UBSan pass. A real-model
check admits two 64-token KV slots with 8 MiB payload inside a 32 MiB budget;
all thirteen prefix output matrices match fixed 512-capacity execution through
position 64. Boundary rejection queues no GPU work, and reuse reproduces the
initial output. Clean zero-warning build and standard tests pass. See
[scope and actual-allocation checks](qwen38-kv-capacity-pool.md).

This is a pool capability, not automatic batch admission or a total VRAM bound.
Next: pass request capacity into backend acquisition and qualify its use in
mixed serving, then extend memory admission beyond KV payload.

### Forty-eight offered requests and forty slot reuses (2026-11-10)

A larger finite overloaded arrival window now qualifies 48 requests with
5/129/513-token prompts, 64 outputs each, eight live slots, forty slot reuses
and 4 GiB retained KV payload. All actual timing geometries/streams match their
independent numerical controls: 573 cohorts / 4,136 full heads and 572 cohorts /
4,132 heads, over one billion BF16 words per control. Numerical controls use
original model/worker math with an explicit FIFO host-selection adapter;
performance comparisons use untouched original schedulers.

Three-run native throughput is 69.625 tokens/s versus 56.954 for original
round-robin256 and 28.674 for decode-aware8. Native p99 ITL is 69.974 ms versus
716.789 and 138.299 ms. Scheduler geometry differs, so this is complete-engine
behavior rather than a kernel-only speedup. The eleven-second arrival window
and subsequent drain do not establish steady-state serving or total VRAM bounds.

Reusable validation now covers up to 64 logical histories / 4096 steps without
relaxing the standalone BMS1 sixteen-request/256-step bound. Prompt-only workload
specifications and BGA4 controlled mixed-budget replay retain legacy wire bytes.
Clean zero-warning builds, standard and targeted tests, both large full-head
controls and legacy 134-cohort model gates pass. See
[scope, source epochs and reproduction](qwen38-cuda-load48.md).
Next: steady-state offered load and broader context/KV-device memory qualification;
wider Flash expert-offload comparisons and the remaining model/backend goal stay open.

### One-token recurrent prefill: matched case passes (2026-11-10)

An exact NONCP one-token inverse initializer skips unused Gram/inversion work.
Frozen CPU/CUDA references remain unchanged; full CUDA and diagnostic suites
pass, including 1.36 billion exact recurrent output/state words. Model controls
remain exact for 144 cohorts / 848 full heads, 143 cohorts / 844 heads, and the
legacy 134-cohort runtime. Clean zero-warning build and standard tests pass.

The twelve-request fixed-clock case now passes measured throughput, mean decode
and p99 margins: 92.638 versus 79.961 output tokens/s, 46.571 versus 50.128 ms
mean decode, and 69.412 versus 69.561 ms p99 ITL. Independent Nsight controls
show mixed-step inverse/Gram cost falling from 3.929 to 0.186 ms, with identical
transfer and synchronization counts. This qualifies the bounded matched case,
not sustained serving or the full modern-model goal. Next: larger offered loads,
context/concurrency coverage, and full KV/device memory accounting before wider Flash Next expert-offload qualification. See
[qualification and measurement scope](qwen38-cuda-single-token-prefill.md).

### Mixed prefill/decode latency budget (2026-11-10)

The planner now accepts an optional mixed-step token cap while keeping the full
budget for prefill-only work. With cap8, all active decodes run and selected
prefills continue progressing. Legacy cap0 and BGA1/BGA2 behavior stay qualified;
BGA3 adds the cap to fixed-clock workloads. Clean zero-warning CUDA/MKL build,
standard suites, targeted tests and planner ASan/UBSan pass.

Both observed schedules independently qualify: 143 runtime cohorts / 844 full
heads and 144 packed replay cohorts / 848 heads, with all three timing streams
matching their actual control. The original new compilation profile's norm
reductions are captured in an explicit policy; the old-profile failure is
retained as evidence. Fixed-clock medians: 92.273 versus 79.961 tokens/s,
46.443 versus 50.128 ms mean decode, and 73.122 versus 69.561 ms p99 ITL. Tail
latency is 5.12% higher and remains outside a 5% target. This twelve-request
case does not qualify sustained serving or complete the full goal. Next:
profile the short mixed cohorts, improve the remaining tail cost, and extend
load/KV-memory qualification. See
[results and reproduction](qwen38-cuda-decode-aware.md).

### Fixed offered load and arrival accounting (2026-11-10)

`generate_batch_enqueue_at` preserves arrivals recorded while a GPU step blocks
the serialized worker. BGA2 workloads carry fixed deadlines and omit expected
continuations. The new clock verifier distinguishes timing/progress validation
from independent numerical qualification. Clean builds, standard tests, unit
and ASan/UBSan checks pass; the legacy 134-step full-head replay stays exact.

For twelve requests offered at 100, 1100, and 4100 ms, native's actual 132-step
schedule matches all 784 full heads from an independent original-runtime
control. Three native timing traces match that compute geometry and outputs.
Fixed wall-clock medians are 101.677 versus 83.065 output tokens/s and 47.247
versus 50.010 ms mean decode, but p99 ITL is 285.957 versus 71.918 ms. Large
mixed prefill steps block active decodes. Overall serving parity remains open;
latency-aware mixed budgeting is the next experiment. See
[wall-clock scope and evidence](qwen38-cuda-wall-clock.md).

### Controlled staggered arrival runtime and CUDA parity (2026-11-10)

The library scheduler supplies FIFO admission, bounded mixed cohorts,
cancellation, backpressure, and token publication before pooled reclamation.
Twelve requests arrive at controlled engine steps 0, 8, and 68. Actual greedy
generation preserves all 784 full BF16 heads across 134 steps, using eight
created sessions and four reuses with 4 GiB of retained KV payload.

The initial eight-request decode gap is resolved by staging adjacent Q8 blocks
in a larger cooperative tile while retaining the original integer dots and
sequential F32 FMA order. Clean builds, standard/full CUDA/diagnostic tests,
focused Q8 memory/synchronization checks, real-model runtime checks, and legacy
batch-four/batch-sixteen full-vector regressions pass without rebasing goldens.
The runtime's earlier whole-workload memory check also passes.

Three-run medians reach 1.926x initial pure-prefill throughput, 47.146 ms mean
decode versus 49.307 ms reference, and 101.870 versus 83.116 output tokens/s.
Both four- and eight-request decode cohorts meet the 5% latency floor. p95 TTFT
is 721.255 versus 1471.062 ms; p99 ITL is 298.030 versus 683.918 ms. The four
mixed steps contain both prompt and generation work and are not pure-prefill
measurements. Profiling records the affected Q8 calls at 71.571 versus 396.301
microseconds, with transfer bytes and synchronization call counts preserved.

This qualifies the controlled arrival case, not fixed wall-clock offered load,
total device-memory budgeting, or the remaining model/backend/offload goal.
See [runtime scope and evidence](qwen38-cuda-arrivals-runtime.md),
[all timing samples](qwen38-cuda-arrivals-performance.json), and
[profile counters](qwen38-q8-small-batch-profile.json).

### Qwen3.8 27B runtime integration (2026-11-05)

Architecture-owned RMS-scaled Q/K normalization now reaches CPU decode and
prefill, CUDA graph decode, fused CUDA prefill, and the host-visible CUDA
convolution/normalization helper. Independent operator fixtures preserve
the separate CPU and CUDA arithmetic contracts. The CPU model path now
selects the reference-prefill projection APIs, retaining prepared layouts
and shared prequantized inputs with separate session scratch.

CUDA now passes 192/192 short reference tokens and 24/24 on the long prompt
(previously 21/24). An initial 172/192 short regression exposed a missed
batched-scan branch still using L2 during prefill while decode used the new
rule. Correcting that branch restores parity and matches all 2,048 captured
real Q-normalization outputs. Nonzero-key recurrent-state fixtures qualify
normalization selection in both scan and serial prefill.

The integrated CPU candidate also passes 192/192 short and 24/24 long
reference tokens on native AVX512, with CUDA regression gates unchanged.
Backend-adapter and projection-failure checks pass, and the final clean binary
passes all four CPU/CUDA short/long gates. Independent matched-schedule
chunked/resumed comparisons also pass 16/16 IDs in each of ten CPU/CUDA
cases: prefix/chunk 33/9, 33/64, 33/11, 256/64, and 1024/129.
AVX2 now also passes 192/192 short and 24/24 long tokens after correcting
Q5_K reference prefill accumulation. Its five matched chunked/resumed schedules
also pass 80/80 IDs against a separately built reference at the same revision.
NEON whole-model gates remain open. See `docs/benchmarks.md` for
provenance and limits; these are correctness gates, not vLLM parity.

### Qualified CUDA Q/K view experiment (2026-11-05)

An opt-in backend-owned Q/K view preserves heterogeneous quant formats and
allows fused attention prefill without copied weight stacks, including
checkpoints with auxiliary prediction blocks. Independent RoPE/operator
fixtures and all CUDA short/long/resumed gates pass after preserving the
model's separate-RoPE arithmetic. Full standard and CUDA hardware suites pass.

Scoped 256-token captures show 62.74% fewer host/device bytes. Initial
unprofiled three-run medians reduce latency by 5.58%, 22.52%, and 39.46% at
33, 256, and 1024 tokens respectively. See `docs/benchmarks.md` for exact
workloads, limitations, and artifacts. This remains an experiment while
broader backend correctness and matched-precision vLLM qualification are
open. It does not yet keep the whole hybrid prefill on device.

### CUDA cached-prefix continuation qualification (2026-11-05)

The opt-in resident prefix chain now passes eleven initial/resumed schedules
(176/176 IDs), including matched 512+9 and separately preserved unbatched
521-token reference histories. A nonzero-state GDN trace exposed a different
FP32 reduction order: the reference uses eight lanes per state column.
The experimental prefill path now matches that layout, with nine independent
state fixtures covering 2,703,360 exact words and a failing legacy-layout
negative control. Full CUDA, standard, AVX2, clean-build, and diagnostic
isolation gates pass. Default policy remains unchanged. See
`docs/benchmarks.md` for artifacts and remaining qualification limits.

### Flash Next bounded expert-cache integration (2026-11-05)

The four-shard `qwen4exp` checkpoint passes the initial CPU short reference
baseline (174/174 generated IDs; early EOS). Qualification exposed an
inactive RAM LRU: split-file pread was excluded by checks of the single-file
descriptor. Shard-aware eligibility and per-request projection descriptors
now activate the existing bounded cache. Cache-disabled, 64 MiB, and 2 GiB
model replays preserve the first eight reference IDs; the fixed 2 GiB path
also passes all 174/174 short reference IDs. Unit tests and clean
standard/AVX2 regressions pass. The mapped-source placement prerequisite now avoids the failed full-residency
attempt for pread storage. Default CUDA activates with explicit expert-cache
budgets and preserves the eight-token smoke output, but broader CUDA batch
gates fail: 24/174 short IDs and 0/24 long IDs. A trace locates NaNs in the
layer-1 routed MoE output; CPU long execution passes 24/24. Subsequent
stream-ordering and stacked-projection corrections remove the NaNs and
pass 46 unchanged independent FFN callback goldens. A near-tie model
mismatch remains in the integrated shared-expert path, confirmed by an
independent fully offloaded CUDA oracle. Flash Next CUDA
batch correctness and the dedicated RAM-to-GPU tier remain open. See `docs/benchmarks.md` for scope and artifacts.

### Flash Next prepared CUDA attention (2026-11-06)

The first full-attention prefill block now reaches backend-owned MMF256
execution for 4..16 FP32 prompt rows, including gated queries. Independent
fixtures qualify attention and returned keys, and the complete layer-three
FFN output matches the normal oracle on the year prompt. Qwen27 CUDA
short/long gates remain 192/192 and 24/24. Whole Flash Next CUDA correctness
is still open: short 129/176 prefix matches with one token-count mismatch,
long 5/24. The short metric regresses from the older checkpoint; local exact
boundaries do not establish model qualification. Next work must locate the
remaining layer/continuation discrepancy, qualify MMVF short shapes and
long/resumed FP32 GPU attention, and then measure end-to-end performance.
See `docs/benchmarks.md` for the independent fixtures and artifacts.

### Shared CUDA projections and final output rows (2026-11-06)

Shared Q8 projection policy now selects the independently qualified native
CUDA contract instead of the conflicting CPU-reference flag. Full tests
pass, but Flash Next remains unqualified: short 121/178 prefix tokens with
one count mismatch, long 5/24. The final prefill FFN also exposes a concrete
schedule difference: the reference crops to one requested output row, while
the runtime limits that optimization to CPU. Next qualify GPU final-row
selection, preserving all-logit requests and persistent session state,
before interpreting further greedy-prefix changes as model correctness.
See `docs/benchmarks.md` for independent projection fixtures and failures.

### CUDA final-row prefill integration (2026-11-06)

CUDA now follows final output-row selection after persistent layer state is
complete, and preserves native arithmetic in the resulting one-row projections
and HC mixer. Final FFN tensors match the independent normal oracle exactly
on the year and HTTP prompts. CPU/CUDA last/all/no-logit state invariance and
Qwen27 short/long gates pass. Flash Next whole-model gates remain 121/178 short
and 5/24 long. HTTP still disagrees on its first output despite an exact final
FFN; inspect HC output mixing, final normalization, and logits next. Retain
all-logit and resumed-session qualification while resolving that remaining gap.

### CUDA native-quant logits integration (2026-11-06)

The logits adapter now requests native quantized activations on CUDA through
the registry-supported matrix API, preserving the legacy matvec fallback.
Independent Q8 fixtures cover the adapter and preserve all reference hashes;
standard tests, the full CUDA suite, and a warning-free clean build pass.
The HTTP prompt now matches its full 24-token completion. Flash Next short
prefix matches improve from 121/178 to 145/178, with the existing token-count
mismatch still present. The year prompt still diverges at the sixth output
token, so decode state qualification remains the next gate. Qwen27 retains
192/192 short and 24/24 long CUDA tokens; Flash Next long remains 5/24.
See `docs/benchmarks.md` for provenance and limits.

### CUDA HC projection contract (2026-11-06)

The first decode trace isolates a CPU-reference Q8 projection request inside
the CUDA hyperconnection mixer. Backend policy now selects CUDA quantized
activations for its down/up projections. The old selection fails independent
fixtures; the new selection passes 14 cases including actual mixer shapes,
preserving all existing targets. The layer 3 FFN input maximum error drops
from 3.79e-4 to 5.96e-7. Clean build, standard tests, and the full CUDA suite
pass. Whole-model correctness remains open: the short gate is 132/179 prefix
IDs, with 192 runtime versus 179 oracle tokens. This is not an end-to-end
improvement over the prior 145/178 score. Qualify the mixer's normalization
ordering against CUDA next; it still depends on the small-dense host-reference
policy. See `docs/benchmarks.md` for the boundary trace and excluded captures.
Qwen27 retains 192/192 short and 24/24 long CUDA IDs; Flash Next long remains
5/24. The active goal remains open.

### CUDA mixer backend arithmetic and controlled replay (2026-11-06)

HC normalization, gated reduction, Q8 projections, and F32 injection now
select CUDA arithmetic independently of the small-dense quant policy.
Metal/CPU reference selection is preserved. All 48 retained norm targets
revalidate unchanged against pinned CUDA; graph-selector tests expose the
old norm and F32 injection failures. A replay of the actual layer 3 weights
matches both the independent reference capture and the runtime output
exactly when given their respective inputs. The remaining 3.79e-4 mixed
output difference is explained by an incoming residual difference of just
2.98e-8. Trace the first earlier decode divergence next; further changes to
the proven same-input mixer are not supported by this boundary.

Flash Next still fails: short 143/177 prefix IDs, runtime 177 versus oracle
179 tokens, and long 5/24. This is not full-model or performance qualification.
See `docs/benchmarks.md` for intermediate failures and controlled replay data.
The clean build, standard tests, and full CUDA suite pass; Qwen27 retains
192/192 short and 24/24 long CUDA tokens.

### CUDA unscaled expert reduction qualification (2026-11-06)

Selected expert outputs and routing weights already match independently at
the first decode layer, but the weighted reduction selected FMA only for
explicit expert output scales. CUDA now selects its existing FMA contract
for complete unscaled sums of 2..15 experts as well. Boundary and emitted-flag
tests preserve unsupported counts, partial sums, and other backends. Forty
independent direct/capture/replay cases pass; the first decode FFN output and
the entire residual entering layer 1 now match exactly. The next discrepancy
is after PLE and the layer 1 attention mixer, maximum 2.38e-7.

Clean build, standard tests, full CUDA suite, and Qwen27 short/long gates pass.
Flash Next remains unqualified at 120/179 short prefix IDs (192 runtime versus
179 oracle tokens) and 5/24 long. Preserve the failures and isolate that next
transition rather than treating local exactness as whole-model qualification.
See `docs/benchmarks.md` and `/tmp/bitnet-goal-moe-fma/qualification.json`.

### CUDA decode PLE backend selection (2026-11-06)

Decode positional embedding no longer inherits host-exact arithmetic from
the recurrent policy on CUDA. It selects the existing GPU PLE operators,
matching prefill's backend contract and preserving non-CUDA selection.
Captured gated and convolution vectors each match all 10,240 reference
values exactly; layer 1 attention input, recurrent output, and FFN input
also become exact. Its FFN output is the next non-exact boundary, maximum
3.73e-9. Use a same-input replay there next.

Clean build, standard tests, full CUDA suite, and Qwen27's short/long gates
pass. The experimental Flash path improves to 144/179 short prefix IDs with
one token-count mismatch, but long remains 5/24. Neither whole-model nor vLLM
parity qualification is complete. See `docs/benchmarks.md` and
`/tmp/bitnet-goal-ple-native/qualification.json` for scope and provenance.

### CUDA shared-expert native math checkpoint (2026-11-06)

A same-input replay isolates the next layer 1 FFN discrepancy to the shared
scalar gate's CPU arithmetic selection on CUDA. Backend policy now selects
native CUDA scalar gating and SiLU; explicit legacy operator flags and
non-CUDA selection remain intact. Eighty-six independent pinned-reference
cases pass 331,062 direct/capture/replay word comparisons. The old selector
fails the new fixture. Integrated layer 1 FFN output is now bit-exact.

Clean warning-free build, standard tests, full CUDA suite, and Qwen27 short
192/192 and long 24/24 gates pass. Experimental Flash short reaches 159/179
prefix IDs with no token-count mismatches; long remains 5/24. At first
decode, layer 2 recurrent and FFN outputs and layer 3 attention input are
exact; attention output differs by at most 1.49e-7. Cache versus attention
arithmetic is the next isolation step. Whole-model and vLLM qualification
remain open. Evidence: `/tmp/bitnet-goal-shared-native/qualification.json`.

### CUDA attention projection checkpoint (2026-11-06)

Same-input replay isolates CPU accumulation flags on native Q8 attention
projections. The corrected CUDA policy makes layer 3 decode query, new key,
and new value vectors bit-exact. Mixed-format Q/K/V group flags retain their
qualified contract; broad per-projection re-selection failed Qwen27's gate.
The final mixed-format-preserving candidate restores Qwen27 short 192/192
and long 24/24. Eighteen independent projection cases pass 529,434 word
comparisons, and the old selector fails the new execution check.

Attention weighted output still differs despite identical Q/K/V. A separate
pinned GGML replay reproduces all 6,144 reference outputs exactly. The current
F32-KV reference-order vector path admits GQA6, while Flash's GQA12 uses the
generic reduction. A controlled runtime replay confirms the separate
score/softmax/value path is exact, while the fused path reproduces the error.
Next qualify GQA12's vector contract independently before extending the
dispatch. Final clean build, standard tests, and full CUDA suite pass; Flash
short is 152/179 and long 5/24. Whole-model and vLLM qualification remain open.
See `docs/benchmarks.md`, `/tmp/bitnet-goal-attention3-cache/replay-manifest.json`,
and `/tmp/bitnet-goal-qkv-native-final/` for evidence and final gates.

### Fused non-flash CUDA attention checkpoint (2026-11-06)

A reference-order fused F32 path now preserves MMVF/softmax/MMVF arithmetic
for 256-wide heads and up to 256 keys. Longer reference contexts use the
qualified separate operations. Explicit flash and existing flash targets
retain their contracts. Coalesced value reads preserve the reduction tree;
a captured 12-key adapter replay measures 19.3 versus 21.8 microseconds for
separate operations, including readback. This is not vLLM parity evidence.
Eighty-eight independent cases pass 1,622,016 bit-exact comparisons.

The native Q8 output-projection selector also removes a proven CPU
accumulation override while preserving qualified mixed-format selection.
Twenty independent projection cases pass 767,352 comparisons. Layer 3's
attention output is now entirely exact. Its FFN input is exact, while its
output differs by at most 7.45e-9. Router logits and expert IDs are exact;
normalized weights differ by up to 1.49e-8 on identical input. Isolate that
route-weight operation next before assigning the full FFN error.

Final clean warning-free build, standard tests, backend matrix, and full
CUDA suite pass. Qwen27 short/long remain 192/192 and 24/24. Flash short has
145 prefix IDs with 175 runtime versus 179 reference tokens; long remains
5/24. Full-model and vLLM qualification remain open. Evidence:
`/tmp/bitnet-goal-attention-native-final/qualification.json` and
`/tmp/bitnet-goal-layer3-ffn/router-manifest.json`.

### CUDA router normalization checkpoint (2026-11-06)

A same-input replay proves the separate-top-k request caused the layer 3
FFN discrepancy: native CUDA routing matches all ten reference weights and
IDs, while the separate request reproduces the old runtime weights. Backend
policy now selects the qualified native contract for power-of-two counts
32–512 and active counts 1–16, preserving legacy operator flags, CPU/Metal,
unsupported shapes, normalization, and scaling. All 480 independent pinned
cases pass 20,880 direct/capture/replay weight/ID comparisons, including
rounded ties and shared normalization lanes. The old selector fails an
expert-selection case. Layer 3's FFN input and output are now bit-exact.

Clean warning-free build, standard tests, and full CUDA suite pass. Qwen27
remains 192/192 short and 24/24 long. Flash now passes short 179/179 with
identical stopping counts across eight prompts; its text display still
differs on special tokens. Long remains 5/24, requiring isolation of long
prefill state before further qualification. Whole-model and vLLM gates
remain open. The first decode output of recurrent layer 0 after the long
1,047-token prefill is bit-exact (2,560 values); the observer preserves normal
reference IDs. The following FFN and later boundaries remain to isolate.
Evidence: `/tmp/bitnet-goal-router-native-final/qualification.json` and
`/tmp/bitnet-goal-long-decode-boundaries/qualification.json`.

### CUDA stacked Q8 prefill checkpoint (2026-11-06)

The long-context failure is traced into layer 1 PLE history. Same-input
history swaps reproduce both convolution outputs, while layer 0 prefill's
shared FFN exposes an upstream stacked-storage reduction mismatch. CUDA now
uses two logical row views for stacked native-quant gate/up batches. The
independent GGML fixtures cover 52 cases and 45,934,592 output words across
host and device shared-FFN paths; the old backend fails a retained target.
The captured position-1038 combined FFN is now exact. Normal unobserved
runtime and reference completion agree on 24/24 long tokens at matched
1,047-token microbatches. The default 512-token reference schedule differs
at step 5, retaining the earlier 5/24 cross-schedule result. Both schedules
are retained; this is not schedule-independent qualification. Layer 1
recurrent output still differs numerically. Clean warning-free
build, standard/full CUDA tests, and backend checks pass. Qwen27 retains
192/192 short and 24/24 long; Flash retains 179/179 short. Further state and
full-model qualification remain open.
The next discrepancy is isolated to PLE reduction geometry: on identical
captured inputs, 128 threads match all four actual model gates, whereas the
current per-token 512-thread call misses one by 7.45e-9. Both kernels match
their own independently replayed GGML row geometries. Propagating full-batch
geometry through prefix PLE remains open.
Evidence: `/tmp/bitnet-goal-q8-prefill-native-final/`,
`/tmp/bitnet-goal-prefix-ple-reduction/`, and the stacked Q8 section in
`docs/benchmarks.md`.

### CUDA PLE batch-geometry checkpoint (2026-11-06)

Prefix PLE now passes the logical batch size to CUDA reduction selection.
The backend selects from the full row count and device SM metadata; explicit
32/128/512 contracts and single-token decode remain qualified separately.
All 28 new pinned-reference cases pass 473,424 output-word checks. The old
512-thread selector fails a retained target, while earlier explicit-mode
reference targets remain unchanged and pass.

After the 1,047-token prefix, all 92,160 saved PLE history values and all
2,560 values in the following recurrent output are now exact. Clean build
has zero warnings; standard tests, full CUDA tests, and backend checks pass.
Qwen27 retains 192/192 short and 24/24 long. Flash retains 179/179 short and
24/24 long under matched 1,047-token reference microbatches; the default
512-token reference comparison remains separate. Bounded-cache chunked and
resumed F32-KV qualification is the next gate. This does not yet remove
per-token PLE allocations/transfers or establish vLLM performance parity.
Evidence: `/tmp/bitnet-goal-ple-batch-native-final/` and
`/tmp/bitnet-goal-prefix-ple-reduction/`.

### Shared prefill-entry admission checkpoint (2026-11-06)

`bn_prefill_no_logits()` omitted the host-orchestrated batch fallback used
by `bn_prefill()`. HC chunks without logits silently dispatched sequentially.
A controlled ordinary-prefill substitution fixes the failing 33/9 replay
against unchanged independent IDs. Both APIs now share admission policy,
with regression coverage for fallback without native chaining and explicit
no-prefill. Clean warning-free build, standard/full CUDA tests and backend
checks pass. Qwen27 retains 192/192 short and 24/24 long; Flash retains
179/179 short.

Bounded-cache F32-KV replays now pass 16/16 on 257/129, 33/9, 257/257 and
33/33, plus the 1047/1047 whole-prefix/resume control. The 9/9 case still
has a resumed mismatch (15/16), while 1047/512
and 1047/129 reach 14/16 and 15/16. Failed checks continue with reference
IDs, so later matches are conditional. These results preserve the remaining
failures rather than qualify all chunk/resume schedules. Next: compare
recurrent/KV/PLE state at the failing continuation boundaries.
Evidence: `/tmp/bitnet-goal-prefill-entry-final/` and
`/tmp/bitnet-goal-flash-chunked-f32/`.

### Recurrent decode-to-prefill handoff checkpoint (2026-11-06)

Before resumed prefill, device state matches the independent reference but
host recurrent mirrors are stale. Refreshing only recurrent/convolution
state restores 16/16 IDs and 2,543,616 bit-exact captured words. Production
now tracks pending decode state per backend session and imports it before
host-orchestrated recurrent prefill. CPU completion, successful upload and
reset clear the marker; failed reads remain retryable. Tests cover isolation,
reset, failure retry, transfer contents and avoiding repeated host reads.

Clean warning-free build, standard/full CUDA suites and backend checks pass.
Qwen27 retains 192/192 short and 24/24 long; Flash retains 179/179 short.
Seven of eight matched prefill/resume schedules now pass 16/16; 1047/512
remains 15/16 due to an initial step-6 mismatch. All resumed checks agree on
common reference histories. A production snapshot reproduces all captured
recurrent/conv/PLE state words exactly on the small resumed case.

The remaining first discrepancy is layer 3 attention keys after initial
prefill: identical raw K/gamma replay under pinned CUDA RMSNorm matches all
512 reference values at position 0; CPU prefill RMSNorm matches all runtime
values. The earlier recurrent layers and PLE history are exact. Next:
qualify native CUDA prepared attention for large F32 batches, preserving CPU
contracts. Serving, remaining models and matched vLLM performance are open.
Evidence: `/tmp/bitnet-goal-recurrent-handoff-final/qualification.json`,
`/tmp/bitnet-goal-long512-state/`, and
`/tmp/bitnet-goal-long512-k-norm/`.

### Batched F32 attention and partial-row Q8 checkpoint (2026-11-06)

The head-256 F32 prepared-attention path now performs query/key normalization,
RoPE, causal/window softmax, value accumulation and query gating in CUDA.
It reads existing F32 prefix caches with the declared offset/stride. Small
prefix batches retain MMF arithmetic; larger batches use cuBLAS. Padded query
rows now reserve aligned GEMM workspace. This path is bounded to batches of
at least four tokens and at most 2048 history keys; other shapes retain the
existing admission and fallback behavior.

Independent pinned-GGML tests cover 16 attention cases and 33,917,952 exact
output/key words, including prefixes, shared/per-head norm weights, window
masks, aliasing and cache isolation. This closes the 1047/512 initial-prefill
failure. Qualification also exposed an earlier HC projection discrepancy:
for 320 output rows, Q8 MMQ selected a full-row token tile despite a partial
row tile. At 129 tokens, width 80 instead of 128 changes stream-K fixup
intervals for tokens 80-127. The backend planner now uses power-of-two token
widths for partial output rows. An independent 40-case, 4,822,272-word test
fails on the old planner and passes with the correction; existing reference
values are preserved.

All eight matched bounded-cache F32-KV prefill/resume schedules pass 16/16:
257/129, 33/9, 9/9, 257/257, 33/33, 1047/512, 1047/129 and 1047/1047.
For 1047/129, all 72,327,168 captured recurrent, convolution, PLE and active
attention-key words match exactly across initial and resumed prefill.
The caches budget 2 GiB RAM and 4 GiB VRAM for experts, excluding base weights,
KV and activations. Qwen27 retains 192/192 short and 24/24 long IDs; Flash
retains 179/179 short IDs. Clean warning-free build, standard/full CUDA
suites, diagnostic isolation and backend checks pass.

These are matched-schedule correctness results. The CLI comparison against
a different reference microbatch schedule remains separate and unqualified.
Host-orchestrated transfers, larger-context attention, expert-placement
performance, serving and matched vLLM performance remain open.
Evidence: `/tmp/bitnet-goal-f32-attention-q8-final/qualification.json`,
`/tmp/bitnet-goal-f32-hc129-replay/`, and
`/tmp/bitnet-goal-f32-resume129-boundaries/`.

### GPU expert prefill host-work checkpoint (2026-11-06)

Profiling the qualified Flash path found duplicate expert reads: the grouped
loop loaded and prepared CPU weights before a GPU attempt, which loaded them
again. CPU weights/layouts are now materialized only after GPU rejection.
Tests require cached GPU execution without host weights and exact CPU fallback
after a rejected partial write. All eight matched schedules, 72,327,168 captured
state words, Qwen27 short/long and the full standard/CUDA suites retain parity.

A serialized paired run halves expert-read bytes and reduces warm 1047-token
prefill from 23.180 to 21.890 seconds. This is a single-pair offload result,
not vLLM parity. Profiling also identifies substantial HC host/device copies,
per-expert allocation churn and a GPU fetch path that bypasses the configured
RAM LRU. See `docs/benchmarks.md` for scope and cold/warm measurements.
Evidence: `/tmp/bitnet-goal-moe-gpu-first-final/qualification.json` and
`/tmp/bitnet-goal-qualified-flash-profile/`.

### Linear Q8 stream-K planning checkpoint (2026-11-06)

The qualified Flash timeline contains a 46.867-second GPU-idle gap while the
host constructs a no-split Q8 schedule. Its 92,160 tiles each scan 92,160
blocks. When block count equals tile count, the backend now emits the identical
full K interval directly, making this case linear. Split arithmetic and
independent reference values are unchanged.

Together with deferred CPU expert preparation, the serialized 1047-token
measurement improves from 70.215 to 21.463 seconds, with expert reads halved.
All 16 benchmark IDs agree. CUDA traces preserve every kernel count and transfer
byte; total kernel time stays 7.843 seconds and the largest GPU-idle gap falls
to 44.1 ms. This is bounded offload evidence from one paired run, not vLLM
parity. See `docs/benchmarks.md` for scope and remaining costs.
Evidence: `/tmp/bitnet-goal-mmq-plan-linear-final/qualification.json` and
`trace-comparison.json`.

### GPU upload RAM placement checkpoint (2026-11-06)

GPU expert uploads now use the bounded pread RAM cache. Decode admits whole
experts; batched prefill consumes hits without changing the working set. Pinned
leases protect upload lifetimes; incomplete reads abort invisible reservations.
The cache remains serialized, and tokenwise prefill still uses decode admission.

A repeated Flash request with an 8 GiB RAM budget saves 48.9% of expert-read
bytes but raises mean decode from 274.3 to 309.8 ms against the disabled-cache
control on a warm filesystem cache. Both runs preserve all 32 independent IDs.
This establishes placement and observability, not a latency improvement. Keep
cache-disabled measurements and optimize allocation/upload/reuse before making
speed claims. Serving concurrency and vLLM parity remain open.
The clean candidate passes standard/full CUDA, diagnostic-isolation and backend
matrix gates; Qwen27 short/long, Flash short and all eight matched Flash schedules
retain parity. All 72,327,168 words across 170 initial/resumed state captures
match the independent oracle exactly. Cross-schedule Flash long CLI parity is
still open. Qualification: `/tmp/bitnet-goal-ram-direct-qualified/qualification.json`.
Evidence: `/tmp/bitnet-goal-ram-direct-final/repeat-summary.json`; see
`docs/benchmarks.md` for the configuration and measurement limits.

### GPU expert reuse experiment (2026-11-06)

With RAM caching disabled, increasing the conservative benchmark GPU cache from
128 to 1,024 entries lowers repeated-request mean decode from 274.3 to 206.0 ms
(24.9%) and expert reads by 34.4%; both runs preserve 32/32 reference IDs. This
single-process comparison motivates GPU reuse and allocation work, not a default
budget increase. Audit CLI cache charges across pread/mmap, backend auxiliary
allocations and every MoE layer before changing placement defaults. Prefill,
serving and vLLM parity remain open. Evidence:
`/tmp/bitnet-goal-vram-reuse/summary.json`; see `docs/benchmarks.md` for limits.

### GPU expert charge correction (2026-11-06)

CLI expert-cache sizing now uses the maximum across all routed layers, with
backend packed/lazy auxiliary layouts and fused gate/up shape accounted for.
The estimate lives beside expert upload policy in `gpu_moe_bridge`. Synthetic
tests cover different layer sizes, mmap-native versus owned uploads, missing
callbacks and overflow. The real Flash charge is 17,715,200 bytes, selecting
242 entries within 4 GiB; two matched 9/9 requests preserve 32/32 IDs.

The protected clean build and standard/full CUDA suites pass. Qwen27 short/long,
Flash short and all eight matched Flash schedules retain parity; all 170 state
captures (72,327,168 words) remain exact. The charge bounds resident payload
estimates, not total live CUDA allocations: incoming upload, scratch, allocator
overhead and other model/session allocations are separate. Per-handle accounting
and strict peak verification remain follow-up work. No latency improvement or
vLLM parity is claimed. Evidence:
`/tmp/bitnet-goal-cache-accounting-final/qualification.json`.

### Preserve GPU decode experts during batched prefill (2026-11-06)

Batched prefill now borrows GPU cache hits without LRU promotion and owns miss
uploads through temporary buffers, preserving decode's RAM/VRAM working sets.
Tests cover hit reuse without host weights, no admission or eviction on misses,
unchanged LRU eviction order, each upload failure, temporary ownership and cleanup.
The protected clean standard/full CUDA and backend checks pass; Qwen27 short/long,
Flash short and all eight matched schedules retain parity. All 170 state captures
(72,327,168 words) remain exact. The new-charge replay preserves 32/32 IDs.

With the same 1,024-slot conservative GPU cache and RAM cache disabled, both
serialized benchmark processes preserve 32/32 reference IDs. Repeated warm
1047-token prefill improves 21.668 to 20.662 s; resumed 64-token prefill improves
5.264 to 4.716 s. Repeated-request mean decode is nearly unchanged (196.7 versus
196.0 ms), and initial cold prefill is 1.4% slower. This is one paired process
measurement, not statistical or vLLM parity qualification. Tokenwise prefill still
uses decode admission, metadata remains serialized, and temporary buffers are
outside the resident cache budget. Evidence:
`/tmp/bitnet-goal-prefill-cache-final/qualification.json` and `performance.json`.
See `docs/benchmarks.md` for configuration, phase timings and limits.

### Carry prefill upload policy through tokenwise fallback (2026-11-06)

Public generation and transformer prefill entry points now scope a session-local
cache flag, including no-logits and all-logits calls. Ordinary GPU expert access
honors that scope, so single-token chunks, `--no-prefill` and tokenwise fallback
preserve decode's RAM/VRAM working sets. The prior flag is restored on every
return and session reset clears it. Direct forward calls outside prefill retain
decode admission; CPU-only cache policy remains unchanged.

Focused scope/reset/cache tests and the protected clean standard/full CUDA gates
pass. Qwen27 short/long, Flash short and all eight matched schedules retain
parity; 170 state captures (72,327,168 words) remain exact. Five additional
matched chunk-one/API modes preserve 80/80 independent IDs and unchanged GPU
cache entries/hash/LRU endpoints over ten ingestion phases. The old-bridge
control fails the cache invariant, confirming that the replay detects the bug.
The new-charge replay also preserves 32/32 IDs. This closes the GPU-upload
phase-policy gap, not serving concurrency, total VRAM peaks, CPU-only cache
policy, performance or vLLM parity. Evidence:
`/tmp/bitnet-goal-tokenwise-phase-final/qualification.json`; see
`docs/benchmarks.md` for oracle provenance and limits.

### Batch HC token launches (2026-11-06)

CUDA HC mix/combine callbacks now launch token rows through grid Y rather than
launching separately for every token, tiling at 65,535 rows. Each token keeps its
block geometry and arithmetic. A four-case fused/unfused CPU reference checks
2,105,392 words, including irregular shapes and the tile boundary; existing HC
reference hashes, full standard/CUDA, model and state gates retain parity.

The initial Flash trace reduces HC launches from 198,933 to 193 and HC kernel
time from 236.5 to 9.0 ms, with identical non-HC launch counts and transfer bytes.
Three paired untraced processes give median initial prefill of 21.257 versus
21.139 s: only a 0.6% difference, with one slower pair. Do not claim substantial
wall-time or vLLM parity from these samples. One decode sample gives mean
forward-plus-argmax/logging intervals of 340.0 versus 334.7 ms, not statistical
latency qualification. Wide host/device round trips and allocation churn remain
the next performance targets. Evidence:
`/tmp/bitnet-goal-hc-launch-final/qualification.json` and `trace-comparison.json`;
see `docs/benchmarks.md` for scope and measurement limits.

### Device-resident HC pre-mix intermediates (2026-11-06)

An optional CUDA prefill operation keeps grouped normalization, down/SiLU/up and
stream-mixing intermediates on-device, returning only mixed activations and
injection values. The original primitives and projection geometry are retained.
F32/native-Q8 projections support 4–2048 tokens; other shapes/types, single-token,
output HC and debug/reference paths keep the staged fallback. The real Flash
prefix uses 95 batches successfully and intentionally rejects one single-token
call. Local workspaces are outside expert-cache budgets.

All 414,158 words across 18 staged-versus-device coherence cases agree exactly.
Protected clean standard/full CUDA and model gates pass; all 170 state captures
(72,327,168 words) remain exact. Three untraced pairs improve median initial
1047-token prefill from 20.818 to 19.602 s (5.8%). Traces remove 24.954 GB of
host/device traffic with every kernel-launch count unchanged. Decode intervals
have nearly flat means and a slightly higher median; no decode performance or
vLLM parity claim follows. HC post-combine, MoE gather/reduction boundaries,
expert upload/allocation churn and serving remain open. Evidence:
`/tmp/bitnet-goal-hc-chain-final/qualification.json` and `trace-comparison.json`;
see `docs/benchmarks.md` for configuration and limits.

### Skip unused packed gathered-prefill weights (2026-11-06)

Explicit gathered-prefill split gate/up uploads now use CUDA quant-only buffers
where GPU policy and quant metadata permit. Gathered kernels read raw rows;
ordinary decode, tokenwise prompt forward and resident preload retain packed
construction. The focused test verifies payload bytes, constructor selection,
failure cleanup and optional-capability fallback. The architecture check and
protected clean standard/full CUDA/model/state gates pass; all 170 captures
(72,327,168 words) remain exact.

Traces remove 13,674 Q4_K packing kernels and 13,674 allocations/frees, while
preserving every compute-kernel count and transfer byte. Three paired processes
improve median initial prefill 19.560 to 18.651 s (4.6%). Decode samples are slower:
primary mean intervals 328.1 versus 333.9 ms; one repeated-request pair gives
warmed mean 311.3 versus 325.0 ms. Both repeated processes preserve 32/32 IDs,
but warmed decode remains a performance concern requiring follow-up. Do not
claim an overall decode/prefill or vLLM parity result. Expert allocation reuse,
GPU-resident MoE gather/reduction and serving remain open. Evidence:
`/tmp/bitnet-goal-raw-gateup-checked/qualification.json`, `trace-comparison.json`
and `warm-comparison.json`; see `docs/benchmarks.md` for scope and limits.

Fixed-affinity ABBA follow-up preserves 128/128 predictions and identical
per-step I/O, cache and buffer-lifecycle work. Across 32 warmed direct forwards
per version, means are 322.385 versus 322.498 ms (+0.035%). The earlier slowdown
does not repeat under these conditions; its cause remains unproven. About
231 ms per forward is inside buffer construction/destruction, including uploads,
preparation and waits, making bounded allocation reuse the next investigation.
Production binary remains the qualified `7b063381…` checkpoint. Diagnostic
evidence: `/tmp/bitnet-goal-decode-work/summary.json` and `provenance.json`.

A subsequent temporary allocation interposer establishes a useful reuse policy:
64 exact-size inactive allocations capped at 16 MiB, with oldest-entry eviction
and full device synchronization before retention. In fixed-affinity ABBA runs,
all 128 reference predictions and per-step expert work remain unchanged; warm
direct forward improves 315.921 to 204.722 ms (35.2%) and repeated prefill 18.561
to 15.657 s (15.6%). A no-eviction variant loses its benefit after the first
request, so admission must follow current shapes. These are diagnostic results
across all legacy allocations, not a qualified weight-only implementation.
Implement a backend-owned bounded pool next, then repeat model/state and timing
gates. Evidence: `/tmp/bitnet-goal-allocation-lru-probe/summary.json`.
Standard baseline tests, the full CUDA suite with the interposer enabled and
diagnostic isolation pass; synchronization errors and tracking overflows remain
zero. Production code and its qualified binary are unchanged.

### Backend-owned CUDA weight allocation reuse (2026-11-06)

Implemented opt-in `BN_CUDA_ENABLE_WEIGHT_POOL`: exact-size reuse of inactive raw
weights and owned packed/converted layouts, bounded to 64 entries and 16 MiB of
requested payload per backend. A device fence precedes admission; oldest-entry
eviction follows current shapes. Explicit trim and backend destruction release
inactive storage. Allocation failure trims and retries; failed fences reject
reuse. Counters and focused limit/content/isolation tests accompany the API.
The pool is separate from expert-residency budgets and requires serialized access.

Protected clean build is warning-free; standard/full CUDA, diagnostics, backend
matrix and AVX2/AVX512 checks pass. Injected fence and allocation failures pass.
Qwen27 retains 192/192 short and 24/24 long; Flash retains 179/179 short, all eight
matched schedules and all 72,327,168 captured state words exactly. The actual
242-slot cache-charge replay retains 32/32 IDs.

Fixed-affinity ABBA measurements preserve 128/128 predictions and identical
per-step expert work. Warm direct forward improves 323.689 to 203.365 ms (37.2%);
repeated prefill improves 18.604 to 15.550 s (16.4%). Separate traces preserve all
91,937 kernel launches and transfer bytes while reducing first-prefill CUDA
allocations 38,092 to 9,606. Pool remains opt-in; these are serialized offload
measurements, not fully resident vLLM parity. GPU-resident MoE gather/reduction,
default-policy qualification, cross-schedule Flash long behavior and serving
remain open. Evidence: `/tmp/bitnet-goal-weight-pool/qualification.json` and the
detailed benchmark section.

### Matched vLLM precision gate (2026-11-06)

Built and registered the official GGUF plugin against the available pinned
vLLM/CUDA image, then loaded the exact Qwen27 Q4_K_XL file. A text-only adapter
alias selects vLLM's existing causal-language-model class without modifying its
math. Both automatic FlashInfer and explicit Triton GDN prefill reject F32
activations in this image. BF16 generation succeeds on the same weights, but
BitNet's qualified F32 activation path is not a matched comparison.

An observer preserves all eight normal BF16 IDs and captures 32 layer-0 buffers
(799,296 BF16 words), plus layer-0/3 parameter and cache dtypes. Activations, KV
and convolution history are BF16; recurrent state and `A_log` are F32; `dt_bias`
and normalization parameters are BF16. This defines the next implementation
gate: explicit precision policy and native CUDA BF16 execution, qualified at
these boundaries before comparing throughput/latency. FP16 KV alone does not
meet it. The local native NVFP4/FP8 checkpoint also differs from Q4_K_XL and
cannot supply a matched-quantization ratio with the current runtime.

No vLLM parity or BF16 BitNet correctness claim follows. Existing BitNet F32
correctness and allocation-reuse qualification remain unchanged. Evidence:
`/tmp/bitnet-goal-vllm-parity/probe-summary.json`; detailed versions, setup,
unsupported comparisons and precision scope are recorded in `docs/benchmarks.md`.

### Native CUDA BF16 normalization (2026-11-06)

Implemented the first typed BF16 operation: optional backend weighted RMSNorm
with BF16 inputs/outputs/parameters, F32 reduction and F32 parameter-offset
addition. It supports in-place output and stages readback before committing host
results. No model-family branch or CPU model-weight mutation is introduced.
This is not a complete BF16 inference mode and does not change F32 dispatch.

All 44 synthetic cases match exactly (101,904 comparisons including aliasing).
The unchanged vLLM layer-0 normalization captures match all 30,720 BF16 words
for separate and in-place output. Tests also cover 65,537 rows, signed zero,
invalid calls, the effective-gamma rounding boundary and partial-readback failure
with successful retry. Protected clean build is warning-free; standard/full CUDA,
diagnostics, backend matrix and AVX2/AVX512 checks pass. Existing F32 model gates,
all eight matched schedules and 72,327,168 captured state words remain exact.

Next implement precision planning and the remaining native BF16 matrix,
residual/activation, GDN and KV/convolution paths before enabling or benchmarking
a full BF16 model. Serving and matched vLLM parity remain open. Evidence:
`/tmp/bitnet-goal-bf16-norm/qualification.json`; detailed scope and reference
provenance are in `docs/benchmarks.md`.

### Native CUDA BF16 matrix components (2026-11-06)

Implemented optional dense BF16 GEMM with FP32 accumulation and independent
BF16 decoded-weight handles for IQ4_NL/IQ4_XS. The quant registry declares the
layout contract; the caller owns and budgets its dense payload. Source handles
can be destroyed after preparation. Existing F32 model execution is unchanged.

The component tests check 18,528 GEMM output words exactly, including in-place
output, and complete hashes of 165,888 decoded BF16 words. The combined layout
and GEMM match all 115,992 outputs in 24 independent plugin linear-dispatch
cases. Pool-on/off and partial-readback failure/retry checks pass. The actual
reference large-IQ path uses CUDA weight dequantization followed by BF16 GEMM;
its small IQ and K-format paths still require separate A8 precision contracts.

Protected clean build is warning-free. Standard/full CUDA, diagnostics, backend
matrix and AVX2/AVX512 checks pass. Existing F32 model gates, all eight matched
prefill/resume schedules and 72,327,168 captured state words remain exact.

Next add typed precision planning, remaining mixed GGUF formats and A8 paths,
resident BF16 intermediates, residual/activation, attention, GDN and BF16
KV/convolution storage. A complete BF16 model mode, serving and matched vLLM
performance remain open. The cross-schedule Flash long CLI mismatch remains a
separate correctness issue. Evidence:
`/tmp/bitnet-goal-bf16-matmul/qualification.json`; scope and provenance are in
`docs/benchmarks.md`.

### Explicit BF16 GGUF matrix dispatch (2026-11-06)

Added an optional backend-quant plan and CUDA callback for IQ4_NL/IQ4_XS.
Small batches use direct BF16-to-A8 activation quantization and one-warp DP4A
MMVQ; larger batches require a separately budgeted decoded BF16 layout and
FP32-accumulating GEMM. Thresholds match the pinned plugin: up to 16 tokens at
output width <=5120, otherwise up to 8. Unsupported formats reject explicitly.
The existing F32 model path is unchanged.

All 104 synthetic cases pass exact whole-output checks for separate and in-place
execution (2,346,208 words), including threshold and padding boundaries. Strict
replays against each of cuBLAS 13.7 and 13.10 match its independent Torch
reference. Two large GEMMs choose different reduction algorithms across those
library versions; both original reference sets are retained without an
elementwise tolerance. Matched vLLM comparisons must pin cuBLAS too. Captured
Qwen27 layer-0 gate/down boundaries match all 135,168 BF16 words exactly.
Failure/retry and weight-pool checks pass.

Protected clean build is warning-free. Standard/full CUDA, diagnostics, backend
matrix and AVX2/AVX512 checks pass. Existing F32 model gates, all eight matched
prefill/resume schedules and 72,327,168 captured state words remain exact.

Next implement the remaining mixed-format matrix contracts (especially Q3_K),
model-level precision planning and resident BF16 residual/activation, attention,
GDN and KV/convolution execution. A full BF16 model mode, serving and measured
vLLM parity remain open. The cross-schedule Flash long CLI mismatch remains a
separate correctness issue. Evidence:
`/tmp/bitnet-goal-bf16-dispatch/qualification.json`; detailed scope and reference
provenance are in `docs/benchmarks.md`.

### Native BF16 Q3_K matrix execution (2026-11-06)

Extended the explicit matrix planner and CUDA callback to Q3_K. Registry
metadata separates linear arithmetic from decoded-layout support. Both Q3_K
paths use A8 activation quantization with BF16 input/output, FP16 scales and
FP32 accumulation; no decoded weight copy is needed. MMVQ thresholds match
the pinned plugin (6 tokens, or 2 above output width 5120). Larger batches use
a 128-row by four-token MMQ tile, sharing activations and decoded weight groups.
Existing F32 dispatch and reduction contracts are unchanged.

All 61 synthetic cases match 658,642 separate/in-place reference output words
exactly. The common BF16 quantizer and Q3_K output tiles use flat grids, including
a passing 65,537-token matrix batch. Its independent reference concatenates two
unchanged MMQ calls below the reference quantizer's grid-Y limit. Captured
Qwen27 FFN-up boundaries match all 104,448 BF16 words exactly (five prefill
and one decode token). Failure/retry, pooling and reproducible-reference checks
pass. All three layer-0 FFN matrix boundaries now have independent coverage;
their activation/residual composition and full BF16 model are not yet qualified.

Protected clean build is warning-free. Standard/full CUDA, diagnostics, backend
matrix and AVX2/AVX512 checks pass. Existing model gates, all eight matched
prefill/resume schedules and 72,327,168 captured state words remain exact.

Next implement the remaining mixed-format contracts, model-level precision
planning and resident BF16 residual/activation, attention, GDN and KV/convolution
execution. Serving and measured vLLM parity remain open, as does the separate
cross-schedule Flash long CLI mismatch. Evidence:
`/tmp/bitnet-goal-bf16-q3/qualification.json`; detailed scope and reference
provenance are in `docs/benchmarks.md`.

### Resident BF16 SwiGLU FFN (2026-11-06)

Added an optional CUDA callback composing gate/up, staged-BF16 SiLU/multiply
and down without intermediate host transfers. A8 gate/up share input
quantization; the gated product reuses the gate buffer. Weight descriptors keep
original format metadata separate from storage. GEMM can execute after raw GPU
weights have been freed, while A8 still requires raw weights and cannot silently
substitute a cached BF16 layout. Existing F32 model execution is unchanged.

All 160 synthetic FFNs match 1,597,440 separate/in-place output reference words
exactly. Entirely decoded IQ4 FFNs also pass in a fresh context after all raw
handles are released. The unchanged Qwen27 capture matches all 30,720 distinct
FFN output words and 104,448 activation-intermediate words in both output modes.
The activation probe confirms BF16 SiLU rounding before BF16 multiplication.
Failure/retry and pooling checks pass. The independent cuBLAS 13.7/13.10 runs
produce identical FFN references for these geometries.

Measured tensor traffic falls 84.8% versus composing three host-entry BF16
matrix calls: five-token Qwen FFN traffic drops from 675,840 to 102,400 bytes,
with one upload/readback and resident intermediates. This is a component traffic
measurement, not a latency benchmark or matched-vLLM parity claim.

Protected clean build is warning-free. Standard/final full CUDA, diagnostics,
backend matrix and AVX2/AVX512 checks pass. Existing F32 model gates, all eight
matched schedules and 72,327,168 captured state words remain exact.

Next connect the remaining mixed-format contracts and model-level precision
planning with resident BF16 normalization/residual, attention, GDN and KV/state
storage. The core FFN now composes correctly; a full BF16 model, serving and
measured vLLM parity remain open. The cross-schedule Flash long CLI mismatch
remains a separate correctness issue. Evidence:
`/tmp/bitnet-goal-bf16-ffn/qualification.json`; scope and provenance are in
`docs/benchmarks.md`.

### Resident BF16 residual/RMSNorm/FFN boundary (2026-11-06)

Added optional CUDA add/RMSNorm and composed add/RMSNorm/FFN callbacks. Both
return a BF16 carried residual; normalization consumes the unrounded F32 sum,
matching the unchanged vLLM IR. The composed operation keeps normalized input
and all FFN intermediates on device. It returns the down-projection output
and carried residual, matching the decoder boundary before the next residual
addition. Separate, in-place and cross-input aliases are supported; both host
outputs remain unchanged on failure.

All 38 norm cases and 48 composed FFNs match independent references exactly:
1,480,784 and 2,293,760 checks respectively across four output arrangements.
Qwen27 layer-0 prefill/decode replays match both returned vectors in all four
arrangements. Reference regeneration, cuBLAS 13.7/13.10, pooling and 16 injected
partial-readback failures pass. The new norm kernel preserves an explicit F32
rounding boundary between mean-square division and epsilon addition.

Measured tensor traffic is one third lower than separate add/RMSNorm and
resident FFN calls: 307,200 to 204,800 bytes for five tokens, 61,440 to 40,960
for one token. These are component transfer counts, not latency measurements.
Protected clean build, standard/full CUDA, diagnostics, backend matrix,
AVX2/AVX512 and existing model/state gates pass. Evidence:
`/tmp/bitnet-goal-bf16-residual/qualification.json`.

Full-model BF16 precision planning, remaining matrix formats, resident
attention/GDN/KV state, serving and measured vLLM parity remain open. The
separate cross-schedule Flash long mismatch remains open.

### BF16 GDN projection reference boundary (2026-11-06)

Added an independent GGUF-to-captured-vLLM projection replay. All 129,600 words
across layer-0 QKV/Z, beta/alpha and output projections match for prefill and
decode after applying the reference's GDN head-layout transforms. The tool
exports packed weights, inputs and expected outputs for native kernel tests.
The inventory identifies Q5_K (191 layer matrices) and Q8_0 (110) as immediate
format gaps. Next qualify typed BF16 A8 MMVQ/MMQ for those formats before
connecting the complete resident GDN layer. Evidence:
`/tmp/bitnet-goal-bf16-model-linear/qualification.json`. This is reference
qualification; it does not add native format support or change runtime dispatch.

### Typed CUDA BF16 Q8_0 linear execution (2026-11-06)

Added Q8_0 to the quant registry's explicit BF16 linear contracts. The planner
selects A8 MMVQ through six tokens (two above 5,120 output rows) and tiled A8
MMQ for larger batches. Both use FP16 activation scales and BF16 output with
F32 accumulation, preserving the independent plugin arithmetic. MMQ shares
activation blocks across a 128-row/four-token tile; a flat grid supports large
batch counts. No decoded weight copy, lazy placement or F32 fallback is used.

All 61 independent synthetic cases pass 658,642 exact separate/in-place word
comparisons, including threshold widths, partial tiles, signed/zero scales,
768/5,120 input widths and 65,537 tokens. Original Qwen layer-0 beta/alpha and
output projections match all 31,296 distinct prefill/decode words in both output
modes. This closes the Q8_0 matrix gap identified by the GDN reference replay;
Q5_K and complete resident BF16 GDN/state execution remain open. Evidence:
`/tmp/bitnet-goal-bf16-q8/qualification.json`.

### Typed CUDA BF16 Q5_K linear execution (2026-11-06)

Added Q5_K A8 MMVQ and tiled MMQ to the explicit BF16 quant contract. The
planner preserves the reference's six/two-token thresholds. MMQ receives the
FP16 original-activation sum required by its affine correction; MMVQ instead
uses quantized-lane sums. Shared FFN gate/up preparation requests the affine
sum whenever either projection needs it. Symmetric preparation is retained
for other plans. Both Q5 kernels explicitly preserve the reference's final
FMA order after separately rounding the minimum product.

All 61 independent matrix cases and 90 mixed Q5/Q8/IQ4/Q3 FFNs match exactly:
658,642 matrix and 998,400 FFN comparisons including in-place output. The FFN
references are identical with cuBLAS 13.7 and 13.10. Original Qwen QKV/gate
prefill/decode matches all 98,304 distinct words in both output modes. Rerunning
the Q8 projections on this binary brings the complete first-layer projection
set to 129,600 distinct captured words exact, with 259,200 comparisons.

Evidence: `/tmp/bitnet-goal-bf16-q5/qualification.json`. This closes the first
GDN layer's typed linear-format gap. The GDN recurrence, gated normalization,
state and complete BF16 execution remain to be connected; other mixed model
formats, serving and measured vLLM parity remain open.

### Typed CUDA BF16 causal convolution and history (2026-11-06)

Added an optional causal convolution/SiLU callback with resident BF16 weights
and caller-owned BF16 history. It preserves the unchanged vLLM Triton contract:
each product rounds to BF16 before F32 accumulation, then SiLU consumes the F32
sum. Direct raw-input windows parallelize prefill across eight-token/channel
tiles. The final tile carries the raw BF16 history, oldest first. The flattened
grid supports long batches. Existing F32 convolution and Q/K norms are unchanged.

All 146 independent cases pass 7,991,240 output/history comparisons, including
separate/in-place output, chunked/resumed calls, token-by-token update and a
65,537-token sequence. Independent prefill and update agree on every short
fixture. Real Qwen weights and captured projected inputs match the unchanged
reference convolution for five prefill tokens followed by decode; eight partial
readback failures preserve both host outputs/history and retry exactly.

In an isolated warm CUDA-graph kernel comparison, tiling lowers 512-token time
from 189.64 to 19.88 us and 1,047-token time from 382.50 to 38.93 us against the
serial BF16 prototype, with identical output/history. Copies and host entry
are excluded; this is not a model-level speed or vLLM parity result.

This establishes convolution precision and state handoff, not the GDN recurrence.
Q/K normalization, gating, F32 recurrent state and the complete resident BF16
layer remain to be connected. Evidence:
`/tmp/bitnet-goal-bf16-conv/qualification.json`.

### Typed CUDA BF16 GDN prefill preparation (2026-11-06)

Added optional fused Q/K normalization, V copy and gate preparation. Packed
QKV and `[beta,alpha]` input rows produce BF16 Q/K/V plus F32 log-decay and beta
in one kernel. Epsilon and softplus threshold are supplied by the caller;
architecture policy is not encoded in backend selection. The primitive expects
resident natural-log A parameters in F32 and BF16 bias, and does not scale Q or
update recurrent state. Outputs commit together after successful readback.

All 178 independent cases pass 38,078,936 raw BF16/F32 comparisons, including
in-place output, chunks, long batches and varied policy values. The original
154 default-profile references are preserved. Real Qwen parameters and captured
inputs match independent preparation results in both output modes; 24 partial
readbacks preserve all outputs/inputs and retry exactly. The reference fused
kernel requires matching padded K/V widths; the backend rejects unsupported
profiles explicitly. Qwen's 128/128 geometry is supported.

Evidence: `/tmp/bitnet-goal-bf16-gdn-prep/qualification.json`. This qualifies
prefill preparation, including a single-token call to that contract, not the
reference model's fused decode recurrence. The separate decode/state qualification
follows below; native chunk recurrence, gated normalization and complete resident
BF16 layer/model execution remain.

### Typed CUDA BF16 GDN decode/state update (2026-11-06)

Added optional `gdn_decode_bf16` for the pinned packed decode contract: BF16
QKV and `[beta,alpha]` rows, resident natural-log A in F32, BF16 bias and
caller-owned F32 `[V,K]` state. It supports 128/128 head widths with logical
repeated K heads, fixed additive epsilon 1e-6 and softplus threshold 20.
Query scale is caller-supplied. Unsupported geometry is rejected explicitly.
The model planner must select this profile only when its policy matches.

Decode retains FP32 normalized Q/K and explicit FMA dot trees. This differs
from the earlier prefill preparation primitive, which rounds normalized Q/K
to BF16. Nine independent eight-step cases pass 65,387,520 output/state
comparisons with separate output, input aliasing and fresh host state storage.
Real captured parameters plus synthetic trajectories pass 48 calls, including
all captured first-decode core outputs. Injected partial readbacks preserve
both outputs and all inputs, then retry exactly.

This is a host bridge for qualifying recurrent arithmetic. Device-resident
composition, native chunk/WY prefill, gated normalization and complete BF16
layer/model execution remain. Evidence:
`/tmp/bitnet-goal-bf16-gdn-decode/qualification.json`.

### Request-owned resident GPU resources (2026-11-06)

`BnBackendSession` now owns an optional backend resource container, separate
from immutable weights and shared activation slots. An owner identity prevents
lookup through a different backend; failed attachment leaves ownership with the
caller. Session reset and destruction detach the container and invoke its
cleanup callback before host state is reset. Backends must finish outstanding
work in that callback. Session tests cover isolation, rejected replacement,
reset, reattachment and exactly-once cleanup.

The private resident chunk/WY GDN core passes 124,368,240 intermediate and
68,247,552 resumed output/state comparisons. Its latest isolated graph medians
are 112.539 us at 257 tokens and 332.096 us at 1,047 tokens, still 2.29/2.38x
the pinned reference component. It is not wired into model execution yet.
Evidence: `/tmp/bitnet-goal-bf16-gdn-ldmatrix/investigation.json`.

The CUDA backend now exposes typed request-owned resident buffers, explicit
workspace reservation and a prepared BF16 chunk/WY GDN operator. Views validate
owner, type, extent, alignment and output aliases before queueing work. The
operator reads immutable previous state and writes separate next state; callers
commit only after successful completion. Its seven kernel launches allocate,
transfer and synchronize nothing. Independent references cover 1..1,047-token
and resumed schedules, guarded buffers and two-request cleanup. The initial
profile supports HK16/HV48/K128/V128 and the pinned inverse reduction profile4.
Model-level precision planning and resident preparation/projection composition
remain necessary before selecting it in normal inference.

Resident GDN preparation now writes planar BF16 Q/K/V plus F32 decay/beta in
one launch, directly consumable by the resident chunk operator. It shares the
host bridge kernel's arithmetic and validates request ownership, tensor extents,
types, scalar parameters and output aliases before submission. Expanded composed
fixtures exposed a latent Q/K reduction-order boundary: contiguous per-lane
columns now match the reference layout. Original preparation fixtures and frozen
independent preparation-to-prefill trajectories remain unchanged. Neither
operator is selected in normal model execution yet; resident projections,
convolution, gated normalization and model precision planning remain open.

Resident packed BF16 GDN decode now accepts immutable previous state and a
separate next-state view. It retains F32 normalized Q/K and shares the original
host bridge kernel, whose input/output state pointer still aliases. Request
ownership, dtype, extent and output aliases are checked before submission.
Four decode steps after each frozen independent prefill trajectory qualify the
GPU-state handoff without host feedback. Convolution/projections/gated norm and
normal model precision dispatch remain necessary before enabling a BF16 model
execution path.

Resident BF16 causal convolution now carries immutable previous history and a
separate next-history buffer on the request stream. It reuses the host bridge
kernel for widths2/3/4 and supports direct convolution→GDN prefill/decode
composition. Expanded decode trajectories exposed fast division zeroing tiny
SiLU tails for large denominators; full-range division now preserves the
reference values. Both convolution history and F32 GDN state stay on the GPU
through the chain. Projections, gated normalization and model dispatch remain
open before normal BF16 inference can select the resident path.

Resident BF16 gated RMSNorm, head-axis transpose and A8 linear projections now
run on the request stream with scratch reserved before execution. The 128-wide
normalization keeps intermediates F32 until its final BF16 store. Independent
synthetic checks cover 56 normalization calls and 251 resident projection cases.
Actual Qwen27 layer-0 prefill/decode captures match both input projections and
the connected normalization→head transpose→output projection chain: 203,328
BF16 words in total. The input QKV/Z weights are Q5_K (GGUF type13); beta/alpha
and output weights are Q8_0. The post-core chain uses four kernel launches with
no intermediate transfers, allocations or explicit synchronization. Complete
layer wiring, model precision dispatch and remaining attention/KV/FFN integration
remain open; this does not establish full-model BF16 correctness or speed parity.
Evidence: `/tmp/bitnet-goal-resident-gdn-norm/qualification.json`.

The complete Qwen35 GDN block now composes resident projections, a bit-preserving
column split, convolution, prefill/decode, gated normalization and output
projection. Original five-token prefill plus one decode match 1,997,376 boundary
and state words. Independent 257/1,047-token prefills, each followed by eight
decode steps, match another 81,067,776 words. Convolution history and F32 recurrent
state carry directly between GPU calls; audited execution uses 18 launches per
prefill and 11 per decode without intermediate host transfers or allocations.
This remains standalone block composition, excluding outer residual/FFN and
normal model dispatch. The actual model still needs Q4_K/Q6_K/IQ3_S typed linear
profiles, request-local decoded GEMM, embeddings and resident attention/KV.
Evidence: `/tmp/bitnet-goal-resident-gdn-layer/qualification.json`.

Q4_K and Q6_K BF16 linear profiles are now declared in the quant registry and
implemented for both A8 MMVQ and tiled MMQ, including request-resident execution.
Q4_K MMQ consumes the registry-declared original-input sum; Q6_K preserves its
32-lane MMVQ partition and two-group inner MMQ reduction. All 132 original
independent cases remain unchanged, with two added 65,537-token grid cases.
The 134 linear and 144 mixed Q4_K/Q6_K/Q5_K FFN fixtures pass, as do full checkpoint
attention-K, attention-output and 248,320-row vocabulary projections at batches
1/3/7/17. These checks cover 16,995,464 exact BF16 output words. The previous
connected GDN block, normal-path token/state/cache tests and sanitizers remain
qualified. Normal full-model BF16 execution still needs IQ3_S, request-local
decoded GEMM, embedding/attention/KV/FFN orchestration and precision planning.
Evidence: `/tmp/bitnet-goal-bf16-q4q6-production/qualification.json`.

IQ3_S BF16 decode/linear and request-owned decoded BF16 GEMM are now qualified.
The 52 independent linear fixtures and 120 mixed IQ3_S/Q4_K/Q5_K FFN fixtures
pass exactly. The actual layer-14 down projection matches all 89,128,960 decoded
BF16 weight words and host/resident projections at batches 1/8/16/17/129.
Generic resident linear now covers A8 and decoded GEMM while retaining the old
A8-only contract. Each GEMM request owns its stream, cuBLAS handle and explicit
workspace; tests cover interleaved requests, peer release/reuse and cleanup after
backend destruction. The qualified decoded-GEMM profile requires 16-byte view
starts: an unaligned destination selected a different cuBLAS reduction, so that
case is rejected before submission rather than relaxing expected values.
CUPTI runtime/driver callbacks observe no allocation/free/copy/synchronization
APIs inside resident linear submissions and identify SM120 BF16 split-K GEMM.
Cold module loads are recorded separately; total code-memory and cold-start
latency claims remain open. All previous token, state, cache-budget and GDN
regressions pass. Normal BF16 model execution still needs resident embedding,
normalization/residual/FFN and attention/KV integration plus model precision
planning. Evidence: `/tmp/bitnet-goal-bf16-iq3s-production/qualification.json`.

### Resident BF16 attention and KV append (2026-11-07)

CUDA now exposes an explicitly reserved request-owned BF16 attention operation
for head size 256. It appends token-major K/V to bounded device caches, executes
causal prefill or GQA decode with the reference split-KV policy, and optionally
applies the BF16 sigmoid gate. The caller owns the logical cache position and
commits it after a successful fence. Exact extents, ownership, alignment and
output/cache aliases are checked before submission.

The adapter instantiates unchanged FlashAttention 2 kernel math from the same
pinned source as the vLLM comparison, without Torch headers or runtime. Only
this third-party adapter requires C++17; the engine remains C11 and the main
CUDA translation unit remains C++11. CUDA builds fetch pinned sources into
ignored `.deps/`, with an offline `FA2_ROOT` override; CPU builds do not fetch
them. Dependency versions and redistribution notices are recorded under
`third_party/flash-attention/` and `third_party/cutlass/`.

Independent paged-cache vLLM outputs match all 84,230,144 native BF16 words over
31 steps in 11 persistent cache chains, covering 1..4,095-token prefill,
chunked/resumed prefill, decode, several GQA ratios and ungated MHA. Tests
also check untouched cache tails, guarded buffers, immutable inputs and invalid
descriptors. Captured Qwen layer-3 prefill and subsequent decode match the
attention and gated output exactly. The reference container must use its normal
NVIDIA entrypoint to enable its forward-compatible driver libraries; bypassing
that startup caused a standalone PTX-toolchain error, while the unchanged
reference succeeds with standard startup.

The protected clean build has zero warnings. Standard tests, the complete
CUDA suite, diagnostic isolation, backend matrix and AVX2/AVX512 checks pass.
Memory and race checks report zero errors/hazards. CUPTI observes no allocation,
free, copy or synchronization APIs inside any of the 31 attention submissions;
there are 2..4 kernel launches per submission, with reservation and explicit
fences outside that scope. Qwen retains 192/192 short and 24/24 long tokens;
Flash retains 179/179 short tokens, all eight resumed schedules, 72,327,168
exact state words and the charged 242-slot cache's 32/32 replay. Qualification:
`/tmp/bitnet-goal-bf16-attention/qualification.json`.

This operation consumes Q/K after normalization and RoPE. The preparation
integration is qualified in the next checkpoint below. Full-model precision/
layout planning, logits and normal request dispatch still need integration
before BF16 model execution or a matched end-to-end vLLM performance comparison
can be claimed. Evidence:
`/tmp/bitnet-goal-bf16-attention/`.

### Resident BF16 Q/K preparation and connected attention block (2026-11-07)

The head-256 CUDA preparation operation now performs per-head Q/K RMSNorm in
F32, rounds normalization to BF16, applies partial NeoX RoPE in F32, and writes
BF16 Q/K plus verbatim V and raw gate values. The caller supplies sequential
text positions, BF16 weights/cos-sin cache, and the architecture-owned norm
weight offset. Exact extents, types, ownership, alignment and output aliases
are validated before submission. The operator uses one kernel with no allocation,
free, copy or synchronization APIs inside submission.

Preparation accepts separate Q/gate, K and V projection buffers. Qwen's actual
layer-3 checkpoint uses Q5_K/Q4_K/Q5_K respectively, so combining all three into
a single quantized weight would violate its format contract. Separate resident
inputs preserve the registered linear profiles without an intermediate packing
copy. Tests submit all projections, preparation, KV append, attention/gate and
the Q6_K output projection before the first host reference read. KV remains
resident between prefill, decode and resumed prefill steps.

All 35,861,504 independent preparation words match exactly across 17 cases,
including wider-range inputs, signed-zero gamma, different RoPE dimensions and
positions through 8,191. The original twelve reference cases remain unchanged.
A shared-memory WAR hazard found by race checking was fixed by storing the final
sum separately from the immutable warp partials; all cases now have zero memory
errors and race hazards.

The connected attention block matches 56,217,600 checkpoint words across ten
steps: captured five-token prefill plus decode, and independent actual-weight
chains 129+1+17+1 and 1047+1+17+1 with synthetic normalized BF16 inputs. It uses
no native output as reference feedback. Memory checks pass on all three chains;
the captured short chain also passes race checking. CUPTI observes no allocation,
free, copy or synchronization APIs inside any connected submission. Preparation
uses one kernel; the block uses twelve kernels normally and thirteen for split-KV
decode. This includes separate activation quantization for the three projections;
shared QKV quantization is a future performance opportunity.

The protected clean build has zero warnings. Standard tests, full CUDA tests,
diagnostic isolation, backend matrix and AVX2/AVX512 checks pass. Qwen retains
192/192 short and 24/24 long tokens. Flash retains 179/179 short tokens, all eight
resumed schedules, all 72,327,168 state words, and the charged 242-slot cache's
32/32 replay. Existing resident GDN and embedding-to-layer-0 composition pass.
Evidence: `/tmp/bitnet-goal-bf16-qk-prep/qualification.json`.

This qualifies the connected attention component, not normal full-model BF16
execution. Backend-owned precision/layout preparation, all-layer orchestration,
final normalization/logits and normal request dispatch remain before whole-model
BF16 correctness and matched vLLM performance comparisons. Serving and the wider
model/offload acceptance criteria remain open.

### Complete BF16 model pilot and dispatch precision fixes (2026-11-07)

A disposable native pilot now executes all 64 Qwen27 layers from original token
IDs through final normalization and vocabulary projection. Five-token prefill
and seven autoregressive decode steps produce the same eight greedy tokens as
the unchanged pinned vLLM worker. All 276 captured checkpoints, including both
complete logit vectors, match 4,705,280 BF16 words exactly. Expected outputs never
feed native activations or state.

The all-layer comparison exposed two boundaries that component fixtures alone
did not catch. First, the reference's mixed-gamma RMSNorm uses a contiguous F32
mean with four accumulators, a batch-dependent reduction width, and a rounded
mean multiplication. A new explicit resident reduction profile implements that
contract for dimension 5120; the existing default reduction remains unchanged.
All 38,799,360 words in 52 independent cases match, with zero memory/race errors.

Second, GDN projection shard geometry affects kernel selection. In a uniform
quantized QKV/Z group, the reference can merge projections. In a mixed-format
group, it executes Q, K, V and Z shards separately; the smaller Q/K shards select
a different small-batch contract. Applying the combined-QKV plan to a mixed
group gave incorrect Q/K prefill results even though the standalone combined
matmul reference passed. The pilot now preserves logical shard dispatch. A new
resident F32/BF16 column join feeds convolution without a host packing copy;
10 guarded cases preserve 2,816,872 words, including arbitrary bit patterns.

CUPTI observes zero allocation, free, copy or synchronization APIs inside all
52 norm submissions, 10 joins and eight complete embedding-to-logits forward
submissions. The full pilot submits 1,698 kernels for five-token prefill and
1,362 per decode step. Token upload, final logit read/fence, host sampling,
initialization and diagnostic checkpoint reads are outside these scopes. These
are execution/transfer observations, not matched performance results.

The protected clean build has zero warnings. Standard tests, full CUDA tests,
diagnostic isolation, backend matrix and AVX2/AVX512 checks pass. Existing Qwen
192/192 short and 24/24 long, Flash 179/179 short, eight resumed schedules,
72,327,168 state words and the charged-cache 32/32 replay remain unchanged.
Evidence: `/tmp/bitnet-goal-bf16-model/qualification.json`.

This is still a private pilot, not normal CLI/library BF16 execution. It uses
the frozen reference rotary cache and disposable process-owned weight/state
storage. Backend-model precision/layout ownership, transactional loading,
request lifecycle/reset, native rotary-cache construction and normal dispatch
remain. The pilot's scalar CPU log conversion also differs from the reference
by one F32 ULP in 63 of 2,304 GDN coefficients; complete BF16-mode recurrent-state
agreement and longer/chunked/resumed model qualification remain unproven.
Serving and matched end-to-end vLLM performance parity remain open.

### Earlier isolated CPU correctness checkpoint (2026-11-04)

An isolated CPU candidate passes 192/192 standard reference tokens, 24/24
on the longer prompt, and 8/8 on each of two matched chunk schedules.
The final missing short-prompt correction preserves prepared matrix
arithmetic when mixed-format projections cannot share K-quant preparation.
See `docs/benchmarks.md` for provenance and limits.

That isolated candidate established the reference arithmetic now being
integrated above while retaining explicit GEMV-order APIs. Independent
resumed prefill and the remaining backend gates must be qualified before
further default acceleration. These are correctness results, not a vLLM
performance comparison.

### Support policy

| Tier | Scope | Commitment |
| --- | --- | --- |
| Optimization targets | Qwen3.8 27B, Qwen3.8 Flash Next, DeepSeek V4.1 Flash 1017, GLM5.3 Flash | Model semantics, regression gates, and separate prefill/decode performance qualification |
| Compatible models | Older Qwen and other models using maintained operators | Preserve correctness through shared execution; no new model-specific optimization obligation |
| Retirement candidates | Bespoke paths unused by targets or useful regression fixtures | Audit consumers and test coverage before removing; age alone is not a removal criterion |

CPU targets are AVX2, AVX512 and ARM NEON 128. CUDA is the primary GPU
performance target. Metal is secondary; existing WebGPU/WASM compatibility
does not expand the performance qualification matrix. Keep small historical
fixtures for inexpensive operator and architecture regressions.

### Execution order and acceptance gates

1. **Measure actual execution before changing kernels.** The host prefill
   profile must distinguish batched execution, decode fallback, fused chains,
   SSM, MoE, and requested logits. Benchmark realistic prompt prefixes as well
   as synthetic inputs. Record matrix shapes, selected kernels, backend
   transfers in bytes, and synchronization at the backend that owns them.
   Host stage timings are not GPU kernel event timings.
2. **Qualify Qwen3.8 27B end to end, starting with CUDA.** Fix model/quant
   correctness gaps first. Establish baselines at multiple prompt and context
   lengths, then improve the largest measured stage. Require actual tiled
   matrix multiplication and device-resident intermediate activations where
   supported. A batched API implemented as repeated GEMVs is insufficient.
   Follow with Qwen3.8 Flash Next as the first target MoE.
3. **Implement measured expert placement.** Separate prefill and decode
   placement decisions. Prefill groups token assignments by expert and reuses
   each fetched expert across its rows. Decode selects GPU residency,
   host-to-device expert transfer, or CPU expert computation from measured
   costs. Backend state owns resident/packed expert buffers; sessions own
   mutable routing and activations. Preserve native router semantics and
   expert count/top-k. Track miss bytes, transfer stalls, CPU work, and reuse;
   a cache-hit percentage alone is not an acceptance criterion.
4. **Extend target architecture coverage.** Implement DeepSeek and GLM
   semantics in model architecture policy, quant behavior in the quant
   registry, and hardware execution in backends. A model name in this list
   does not mean its checkpoint is loadable or qualified yet.
5. **Add serving scheduling above sessions.** Continuous batching, bounded
   chunked prefill, and KV allocation must follow sound single-request
   execution. Hull remains the HTTP serving owner. Scheduling and expert
   residency must account for each other's memory budget.

Compare resident execution against vLLM on identical hardware, checkpoint,
precision/quantization, context, prompt corpus, concurrency and output
contract. Track prefill throughput and time to first token separately from
decode inter-token latency and aggregate serving throughput. Report warm and
cold behavior separately. Offloaded runs have a separate fixed VRAM/RAM
budget gate; they do not establish parity with a fully GPU-resident run.
Publish correctness and performance evidence together. No vLLM parity is
claimed by this roadmap.

First implementation checkpoint: corrected host timing attribution and a
CPU/CUDA prefill benchmark with file prompts, final-logit measurements, and a
tokenwise control. Backend transfer/kernel instrumentation, full CUDA qualification,
expert placement, and serving scheduling remain open.

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
  The required target defaults both families to `full` (16 tokens per prompt).
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
  `QWEN_CPU_PARITY_LEVEL=full` now requests 16 tokens per prompt for every
  case, including sparse models and Qwen3.6/3.8; the former 5-token full
  checks missed longer-prefix failures. `CPU_PARITY_THREADS` sets the shared
  thread count (default 1), with `QWEN_CPU_PARITY_THREADS` taking precedence.
  The top-level matrix collects both families even when Qwen fails.
  Synthetic argument/exit-status tests run in `make test`.
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
  `GEMMA4_CPU_PARITY_LEVEL=full` now requests 16 tokens per prompt for both
  dense and sparse cases. `GEMMA4_CPU_PARITY_THREADS` overrides shared
  `CPU_PARITY_THREADS` for this family.
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

### Backend model layout ownership (2026-11-07)

`BnBackendModel` now accepts one opaque owned GPU layout container. Attachment
transfers ownership only on success, rejects replacement, and uses an owner
identity for lookup. GPU release and model destruction detach the container
before its cleanup callback, then destroy registered GPU buffers. This lets
precision-specific layouts borrow existing handles while keeping their own
allocations outside immutable model weights. Sessions must finish before model
teardown; this API does not extend weight lifetime automatically.

Synthetic lifecycle coverage verifies invalid attachment, owner isolation,
replacement rejection, cleanup ordering, repeated release, and destruction
after partial preparation without a bound GPU. The BF16 pilot has not yet been
migrated into this container; normal BF16 dispatch and the remaining full-model
qualification are still open.

### Backend-owned BF16 model preparation (2026-11-07)

The complete Qwen27 pilot now consumes `bn_backend_layout_prepare_bf16` instead
of allocating disposable process-owned weights. Preparation builds raw quantized
weights, optional decoded BF16 layouts, reordered GDN projections and auxiliary
parameters in one backend-owned container. Nothing is published until every
upload succeeds. Failure releases the partial container and preserves existing
backend ownership; model teardown destroys the container before registered GPU
handles. Request KV, convolution history and recurrent state remain separate.

The architecture registry now selects the hybrid offset-norm preparation
contract explicitly. The current preparer accepts only the qualified dense
5120/17408, 64-layer hybrid geometry and rejects other layouts. Quant registry
plans still select the linear math contract; mixed-format GDN Q/K and V shards
retain their logical dispatch geometry. This adds no default BF16 execution or
model-family dispatch inside CUDA kernels.

Both preparation bounds tested (5 and 129 tokens) preserve all 276 frozen
checkpoints: 4,705,280 exact BF16 words and eight greedy tokens. Actual execution
in both runs remains five-token prefill followed by seven decode steps; the
129-token preparation run only additionally qualifies decoded-weight allocation
and teardown. The two runs release all 830 and 910 tracked GPU buffers,
respectively. Full-model CUDA memory checking reports zero errors and zero
leaked allocations. Real upload failures at positions 1, 3, 14, 200 and 829 leave
zero live buffers; synthetic tests cover invalid metadata and decoded-layout
creation failure. References remain unchanged.

The pilot still owns its request metadata locally and borrows the frozen
reference rotary-cache data. Normal CLI/library BF16 dispatch, native rotary
cache construction, exact reference A_log conversion, complete BF16 recurrent
state comparisons and longer/chunked/resumed full-model qualification remain.
Serving, bounded expert offloading and matched performance goals remain open.
Evidence is under `/tmp/bitnet-goal-bf16-prepared/`.

Final qualification passes the protected clean build with zero warnings,
standard suite, full CUDA suite and diagnostic isolation. Existing reference
runs remain Qwen 192/192 short and 24/24 long, and Flash 179/179. The source,
reference, binary, checkpoint and cleanup assertions are recorded in
`/tmp/bitnet-goal-bf16-prepared/qualification.json`; no end-to-end performance
parity is claimed.

### Session-owned BF16 execution through forward/prefill (2026-11-07)

`src/transformer/bf16.c` now executes embedding, all 64 hybrid layers, final
normalization and vocabulary projection using the backend-owned BF16 layout.
A session-owned host descriptor holds views, cache capacity, position and
recurrent-buffer parity. GPU allocations remain in the independently owned
resident request. Reset/free detach both containers, finish GPU work, then free
host metadata without dereferencing expired model/backend storage.

After explicit BF16 model and request setup, the ordinary
`bn_transformer_forward` and `bn_transformer_prefill` APIs select this execution
path before preparing legacy state. Missing/failed requests reject; they cannot
fall through into CPU execution. Position/parity commit only after logits and
preceding GPU work complete. Invalid input queues no work. A submission,
transfer or observer error invalidates the request until reset and leaves the
caller's output logits unpublished. Argmax acceleration, no-logits and
all-position prefill entry points currently reject BF16 before changing state.

The repository integration fixture `test/test_bf16_runtime.c` checks observed
and ordinary API execution, two-session isolation, reset/reuse, existing-owner
preservation, partial reserve/allocation/upload failures, and failures during
submission, output read and diagnostics. Its unchanged primary captures match
276 checkpoints and 4,705,280 BF16 words; the normal API also matches both
complete logit vectors (496,640 words) without intermediate diagnostic fences.
Eight autoregressive tokens match in both modes, without reference feedback.

Build the optional integration fixture with
`make BN_ENABLE_CUDA=1 test_bf16_runtime`, then run
`./test_bf16_runtime MODEL.gguf COS_SIN.bin DUMP_DIR` or replace `DUMP_DIR` with
`--plain`. It requires the pinned Qwen27 GGUF and independent 262144x64 BF16
rotary-cache fixture; the dump directory must already exist. It is not part of
the model-free standard suite. ARM, x86, scalar and WASM source lists include
the new transformer implementation; WASM shell scripts pass syntax checks,
but no Emscripten toolchain is available here for a WASM link test.

This is explicit internal setup feeding existing library execution APIs. CLI
initialization and native rotary-cache generation remain unimplemented. The
ordinary session still allocates unused CPU mirrors; typed-only session memory
planning remains future work. Complete BF16 recurrent-state agreement, exact
A_log conversion and longer/chunked/resumed full-model qualification remain
open, as do expert offloading, serving and matched performance parity.
Evidence is under `/tmp/bitnet-goal-bf16-request/`.

Final checkpoint qualification passes the clean CUDA build with zero warnings,
standard suite, full CUDA suite, diagnostic isolation and backend matrix. The
complete request integration fixture passes memory checking with zero errors
and zero bytes leaked, including injected failure/reset paths. Existing
reference runs remain Qwen 192/192 short, 24/24 long, and Flash 179/179.
`/tmp/bitnet-goal-bf16-request/qualification.json` records the binary/source and
reference checks. These are correctness/lifecycle results, not performance
parity or full long-sequence BF16 qualification.

### Native model-owned BF16 rotary cache (2026-11-07)

BF16 preparation now builds its rotary cache directly on CUDA and owns it in
the immutable backend model layout. Request setup selects that cache when no
diagnostic override is supplied. Cache generation is transactional with the
other model weights; a failed cache build releases the entire unpublished
layout. Normal forward/prefill execution needs no captured rotary data.

The model-independent GPU operation uses F32 powers and reciprocal frequencies,
rounded F32 angles, precise sine/cosine and BF16 rounding. The initial profile
supports rotary dimension 64 and 1..262144 positions, packed [cos32|sin32]. The
architecture's existing model-policy adapters supply theta and rotary geometry.
No model-family selection occurs inside CUDA and the legacy F32 path is unchanged.

Reproducing the pinned vLLM/Torch recipe matches the original full cache's
16,777,216 words. The production native builder matches every one of those
words, recovered through actual Q/K preparation on its nonblocking request
stream. A fast-power prototype differs in 477,576 words; the precise recipe is
retained. Independent fixtures add three bases, 93 boundary-position cases,
small/tail cache lengths, invalid parameters, input immutability and guarded
consumer outputs: 12,224 exact reference words. The fixture header reproduces
byte-for-byte from its pinned Torch generator.

`test_bf16_runtime MODEL.gguf --native DUMP_DIR` now runs the complete model
using model-owned native rotary data; `--plain` may replace the dump directory
for normal API execution. The native run preserves all 276 checkpoints,
4,705,280 BF16 words and eight autoregressive tokens. The fixture also injects
cache-creation failure before publishing the model layout, in addition to its
request allocation, transfer, submission, reset and session-isolation checks.

CLI BF16 initialization is still absent. Longer/chunked/resumed BF16 model
qualification, complete F32 recurrent-state agreement and the known A_log
conversion differences remain open. Sessions still allocate unused CPU mirrors.
Expert offloading, serving and matched whole-model performance parity remain
open. Evidence is under `/tmp/bitnet-goal-bf16-rope/`.

Final qualification passes the protected clean build with zero warnings,
standard suite, full CUDA suite, diagnostic isolation and backend matrix.
Rotary memory/race checks report zero errors/hazards; the full native-cache
model fixture reports zero errors and zero bytes leaked. Existing reference
runs remain Qwen 192/192 short, 24/24 long, and Flash 179/179.
`/tmp/bitnet-goal-bf16-rope/qualification.json` records the final source,
binary, reference and output checks. These results do not establish
long-sequence BF16 state correctness or end-to-end performance parity.

### BF16 generation entry and larger-prefill diagnosis (2026-11-07)

Generation now selects typed BF16 prefill before the legacy GPU entry policy.
It validates the entire prompt and capacity before submitting any chunk, uses
the request's prepared token bound, and bypasses legacy CPU KV/SSM uploads.
Explicit tokenwise prefill uses the same request transaction. Prefix chunks
can execute without logits: after all layers, a successful fence commits
position/parity without final normalization, vocabulary projection or host
logit publication. A failed chunk/fence invalidates the request until reset.
Generation's external position advances only after a successful forward step.
Speculative verification still rejects BF16 before work because all-position
logits are not yet supported.

The integration fixture now exercises `bn_prefill` and `bn_generate`, including
eight independent expected callback tokens, no-logits/full-forward state
equivalence, tokenwise/direct equivalence, 5+1 chunking, whole-prompt invalid-ID
and capacity rejection, later-chunk failure, fence failure and external-position
handling. It retains model/request rollback, reset and session isolation checks.

A new pinned vLLM capture uses 129 natural-text tokens and preserves all layer
norm/output checkpoints. This larger native prefill is **not qualified**: the
final native run matches only 2/8 greedy tokens. Diagnosis found two distinct
causes. First, the CPU logarithm used to prepare `A_log` differs from the pinned
Torch result. Direct calls identify that result as MKL `vsLn` (2304/2304 words),
not the SLEEF implementation initially suspected. Native scalar `logf` differs
in 63 words; native CUDA `logf` differs in 28, SLEEF in 53. No replacement math
implementation or dependency has been adopted.

Second, mixed-format GDN projections must preserve separate logical Q2048 and
K2048 matrix shapes even when their quant dispatch cutoffs agree. Combining
those shapes changed decoded GEMM output at layer21. The preparer/executor now
keep Q and K separate and join their outputs on GPU. Uniform QKV groups retain
their combined layout. This changes no original CPU model weights.

A diagnostic-only binary substitutes reference-prepared `A_log` parameters to
isolate the remaining error. With separate Q/K shapes it matches all 516
captured checkpoints (171,555,840 BF16 words) and eight tokens. This is causal
evidence, **not native-engine qualification**; no expected activation or state
is injected, and the parameter lookup exists only under `/tmp`, outside engine
sources/builds. Independent reference values remain unchanged. The production
engine still uses native `logf` preparation and its 129-token gate remains
failing. CLI BF16 setup is intentionally not enabled before that issue is fixed.
Evidence is under `/tmp/bitnet-goal-bf16-generation/`.

The scoped five-token checkpoint passes the protected clean build with zero
warnings, standard/full CUDA suites, diagnostic isolation, backend matrix and
legacy Qwen/Flash regressions. Its expanded integration run has zero GPU memory
errors and zero bytes leaked; the 276 checkpoints and both normal-API logit
vectors remain exact. CUPTI observes no allocation/free/module-load APIs in
prepared generation prefill: normal logits use 1,776 kernels and 496,648 D2H
bytes; no-logits uses 1,773 kernels and 8 D2H status bytes. Both upload 20 token
bytes and perform two explicit synchronization calls. Initialization/startup
are outside these scopes; no latency claim is made.

`/tmp/bitnet-goal-bf16-generation/qualification.json` records these passes AND
`native129_qualified: false`. The failing larger-prefill gate is retained in
`native129-gate.json`; the reference-parameter diagnostic is separately marked
unqualified. The full runtime goal remains open.

### Explicit host math profile for BF16 preparation (2026-11-07)

`bn_backend_layout_prepare_bf16_with_math` selects host preparation arithmetic
explicitly. The existing wrapper retains libc `logf`; a build with
`BN_ENABLE_MKL_VML=1` also provides `BN_BACKEND_BF16_MATH_MKL_VML_HA`.
Unavailable or unknown profiles fail before GPU allocation. Prepared model
resources record the selected profile and actual library version. The vendor
path uses per-call high-accuracy mode with FTZ/DAZ disabled, sequential linkage,
and no process-wide VML mode or thread-setting calls. Backend ownership and
execution precision are unchanged.

For the qualified Linux x86-64 vendor version:

```sh
python3 patches/fetch-mkl-vml.py
make clean
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 bitnet
```

The fetcher verifies fixed SHA256 hashes for the standalone oneMKL 2021.1.1
runtime and headers and preserves their licenses under the dependency directory.
It installs no Python packages and requires no Torch library. An existing
installation can be selected with `MKL_VML_ROOT=/absolute/path`; that version
must be qualified separately. Clean between changes to build options. Preserve
local diagnostic binaries before cleaning, as required by the agent workflow.
This option supplies host parameter preparation, not a replacement CPU matrix
backend or a portable ARM math implementation.

The original 129-token independent reference remains unchanged. With the
standalone pinned library, all 2,304 prepared coefficients, 516 whole-model
checkpoints (171,555,840 BF16 words), and eight greedy tokens match exactly.
`/tmp/bitnet-goal-bf16-vml/native129-gate.json` records the actual native run;
no reference-parameter lookup is compiled or supplied. The libc profile retains
its previously recorded larger-prefill mismatch. CLI enablement still requires
longer, chunked, resumed, and complete state qualification. Full runtime and
performance acceptance criteria remain open.

The additional independent state capture uses the reference scheduler's actual
cache slots and preserves all 516 earlier activation checkpoints. Across the
129-token prefill and seven decode steps, all 301,989,888 F32 SSM state words and
11,796,480 BF16 convolution-history words match exactly. The ordinary run without
an observer matches eight tokens and both complete host-logit arrays (496,640
BF16 words). The optional profile passes clean warning-free CUDA compilation,
standard suites with and without the option, the full CUDA suite, backend
checks, and whole-model sanitizer checks with zero errors and zero leaked bytes.
State evidence is in `/tmp/bitnet-goal-bf16-vml/states/qualification.json`.
These results qualify this pinned host/profile and schedule; they do not yet
qualify arbitrary chunk schedules or the portable libc profile at 129 tokens.

The optional real-model generation/failure fixture can select the profile:

```sh
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 test_bf16_runtime
./test_bf16_runtime /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf --native --plain --mkl-vml
```

This fixture requires the qualified model; the ordinary inference CLI has no
BF16 preparation switch yet.

### Prefill operation identity and deterministic greedy ties (2026-11-07)

A real scheduler-driven `64 + 64 + 1` prefill exposed a distinction that token
count cannot encode: its final one-token prompt chunk still uses prefill GDN
arithmetic. The first two chunks were bit-exact; selecting recurrent decode
for the tail changed the F32 state. Session execution now carries the operation
explicitly through the prefill/forward entry points. Explicit `--no-prefill`
continues to use tokenwise forward operations. The integration fixture also
checks resumed prompt calls against the same chunk schedule.

The next divergence came from SIMD greedy selection when BF16 logits tied.
Horizontal reduction previously preferred the first SIMD lane, which could
select a larger token ID. NEON, AVX2, and AVX512 now reduce equal maxima by the
lowest token index. Exhaustive synthetic tied-pair cases cover vector lanes,
vector blocks, and a scalar tail, plus all-negative-infinity input. The old
AVX2 and AVX512 implementations fail 784 and 720 of those cases respectively;
scalar, AVX2, and AVX512 pass after the fix. No NEON hardware result is claimed.

With both fixes and the pinned math profile, the actual `bn_prefill` /
`bn_generate` consumer matches the independent vLLM chunk schedule at all 960
state/history checkpoints: 377,487,360 F32 SSM words, 14,745,600 BF16 history
words, and eight greedy tokens. Calling `bn_prefill(64, 0)` and then
`bn_prefill(65, 64)` before generation also matches every checkpoint. The
original failed runs remain under `/tmp/bitnet-goal-bf16-vml/chunk64/` with
`before-mode` and `before-tie` suffixes; reference captures were not changed.
Final build qualification is recorded separately under
`/tmp/bitnet-goal-bf16-operation/`. Longer-context, CLI, offloading, serving,
and matched-performance qualification remain unfinished.

The same final build also passes an independent **1,025-token** prefill followed
by seven decode steps. All 301,989,888 F32 state words, 11,796,480 BF16 history
words, both complete vocabulary logit arrays (496,640 BF16 words), and eight
generated tokens match the pinned reference. A separate ordinary run with no
intermediate observers reproduces both logit arrays and all eight tokens.
Evidence and frozen reference hashes are under
`/tmp/bitnet-goal-bf16-long1025/`. This extends the qualified prompt lengths to
1,025 tokens for this host/profile; broader contexts and CLI integration still
need qualification.

### Explicit resident BF16 CLI (2026-11-07)

`--cuda-bf16` selects the qualified resident BF16 path and implies CUDA. Build
with `BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1`; it prepares backend-owned weights
without the legacy F32 upload or CPU weight packing. `--prefill-batch N` bounds
prompt chunks (default 1025, capped by request capacity). For example:

```sh
./bitnet /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf --cuda-bf16 --maxseq 2048 \
    --prefill-batch 64 -p 'Hello' -n 32 --temp 0 --repeat-penalty 1
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 test_cli_bf16 MODEL=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

The mode requires the qualified dense layout and available vendor math. Device,
model preparation, request preparation, or inference failures return nonzero;
they never select CPU execution. Draft verification, alternate KV formats, and
legacy GPU math overrides are rejected. BF16 KV and attention are provided by
the resident path. CPU prompt-cache snapshots are bypassed in chat because they
do not contain backend-owned KV/SSM state; reset and context rollover recreate
the request while retaining immutable prepared weights. Repeated benchmarks do
the same. `--no-prefill` explicitly uses tokenwise forward operations.

The real-model CLI test uses frozen independent prompt IDs and expected tokens
with model/image/plugin/capture provenance in `test/bf16_cli_reference.json`.
It covers short/129/1025-token input, chunked input, benchmark and chat resets,
context rollover, no-logits prefill, capacity failures, unsupported options, and
missing devices. Reference-parity runs specify repetition penalty 1 explicitly;
the existing CLI sampling default is retained. Timing output from correctness
runs is not a performance-parity qualification.

Chat now commits the final sampled assistant token before appending its turn-end
marker. Generation intentionally leaves that token unforwarded; closing the turn
first had made request position disagree with token history. Callback position
tracking distinguishes an uncommitted token from one already forwarded before a
loop abort, and a sampled turn-end marker is not appended twice. A regression
uses the pinned independent tokenizer's 69-token chat prompt: two generated
tokens and one closing marker must leave position 72. The previous CLI reported
71; the corrected CLI reports 72. The CLI fixture also checks identical results
after explicit reset and each automatic context rollover.

### Resident BF16 integer tensor-core execution (2026-11-07)

Q4_K/Q5_K/Q6_K A8 prompt matrices now use backend-private 16-row integer MMA
tiles on SM80+: 24-token tiles for short matrices and 64-token tiles for larger
ones. The existing precision planner still selects MMVQ versus MMQ; this only
changes the implementation of each operation. Older devices retain DP4A.
Four-byte aligned Q4_K/Q5_K loads decode only the requested scale groups, and
matrix tiles copy quantized words into shared memory. Weights retain their
original quantized storage. Integer dots, affine two-group reductions, and
Q6_K's outer FMA order remain unchanged. No request-time global allocations or
intermediate host transfers are added.

The shared CUDA FP16 conversion now uses hardware conversion while explicitly
preserving the original NaN sign/payload. `make BN_ENABLE_CUDA=1
test_cuda_fp16` checks all 65,536 input bit patterns against an integer IEEE-754
oracle. The full CUDA target includes this test. CPU conversion and SIMD code
are unchanged.

On the qualified SM120 host/model/math profile, isolated warm CLI medians fall
from 8,249.85 to 1,219.2 ms for 1,025 prompt tokens, from 1,044.55 to 215.3 ms
for 129 tokens, and from 97.95 to 51.7 ms for five tokens. Long-context decode
falls from approximately 30.2 to 16.7 ms per forward step.

A separate callback-based benchmark uses three warmups and ten measured
requests. It includes the first CPU sample in backend-ready first-token time
and measures decode from first to last callback, avoiding rounded CLI rates.
The optimized vLLM comparison retains normal compilation/graphs and automatic
FlashInfer prefill/CUDA decode selection on the same headless GPU, GGUF,
BF16 activation/KV and F32 SSM configuration, context 2048, concurrency one,
eight greedy outputs, repetition penalty one, and disabled prefix caching.

| Prompt tokens | Native ready-to-first (ms) | vLLM scheduled-to-first (ms) | Native decode (ms/step) | vLLM decode (ms/step) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 51.85 | 49.02 | 15.83 | 18.07 |
| 129 | 215.82 | 376.02 | 16.10 | 18.18 |
| 1025 | 1214.66 | 2542.10 | 16.61 | 18.04 |

This is a backend-ready comparison, not full serving qualification. The
five-token first-token result remains 5.8% slower, outside the 5% gate. Native
request reset/preparation also adds about 30 ms: a separate short-case probe
attributes approximately 14.5 ms to reset, 15.6 ms to initialization, and
0.007 ms to the final readiness fence. Including setup raises short-case
first-token time to about 82 ms. Request allocation/reuse is the next latency
target. The optimized vLLM 129-token output differs from the retained canonical
eager/Triton numerical profile; both measurements and original goldens are
preserved. No all-workload vLLM parity claim is made.

The integrated build passes the standard suite, complete CUDA suite,
diagnostics, CLI regression, 63/64/65-token matrix boundary checks, and
sanitizer checks with zero errors or leaked allocations. The short observer
matches all 516 retained checkpoints (8,391,680 BF16 words) and eight tokens.
Full-state comparisons preserve all independent references for 129-token,
chunked, resumed, and 1,025-token prefill followed by decode. All checked F32
SSM, BF16 history, activation, and vocabulary logits words match. Legacy
comparisons retain 192/192 Qwen short, 24/24 Qwen long, and 179/179 Flash short
token checks. Scoped profiling confirms the new kernels and exactly 4,100
input bytes and 496,648 output/status bytes for the long prefill.

Evidence, commands, source hashes, callback measurements and profiles are under
`/tmp/bitnet-goal-bf16-fast-integrated/`; the initial optimized vLLM baseline is
under `/tmp/bitnet-goal-resident-baseline/`. Earlier qualified tile experiments
are retained separately under `/tmp/bitnet-goal-bf16-mma-integrated/`.

The new kernels compile independently for SM75/SM80/SM90, with the older
architecture fallback excluded from host selection. Whole-backend SM80/SM90
builds remain unqualified: the exact pre-change source already fails on shared
MoE helpers hidden behind the SM120 flag. A private guard fix exposes additional
existing kernels exceeding the 48 KB static shared-memory limit. Those backend
build issues, the remaining latency gates, bounded modern-model expert
placement, serving, and the broader runtime goal remain open.


### Backend-only sessions and device initialization (2026-11-07)

The explicit native CUDA BF16 path now creates a `bn_session_create_backend`
session. It retains host logits and token history, while backend session state
owns activations, KV, recurrent state, and request compute buffers. Ordinary
`bn_session_create` retains the existing CPU-compatible allocation. For the
qualified Qwen 27B configuration this reduces host session storage from
426,770,432 to 1,001,536 bytes. CPU execution and CPU prompt-cache entry points
reject backend-only sessions before accessing absent CPU buffers; reset/free
release outstanding backend work first.

The optional backend `resident_zero` operation initializes validated resident
views in request-stream order. Native BF16 requires this capability. CUDA uses
`cudaMemsetAsync` instead of uploading host zero arrays for the 96 recurrent
state/history buffers (153,944,064 bytes). Initialization failure at the first,
second, or last zero operation releases the request; numerical state and
reference checkpoints are unchanged. The production allocator remains the
existing synchronous allocator; private asynchronous allocation experiments
have not been adopted.

The standard and complete CUDA suites, native short/long/chunked/resumed
reference comparisons, CLI regressions, sanitizer gates, legacy Qwen/Flash
checks, and the real-model initialization/execution fault-injection test pass.
Evidence and setup-inclusive latency measurements are retained under
`/tmp/bitnet-goal-backend-zero/`. These changes do not complete the broader
expert-offloading, serving, or cross-model performance acceptance criteria.

The three-warmup/ten-sample rerun measures median setup of 13.8–15.4 ms.
Five-token ready-to-first is 52.84 ms and reset-to-first is 66.68 ms; decode
is 16.18 ms/step. At 129/1025 prompt tokens ready-to-first is 218.17/1216.03 ms
and decode is 16.15/16.51 ms/step. The five-token comparison remains outside
the 5% vLLM gate, and setup-inclusive serving parity is not qualified.
A scoped initialization Nsight capture confirms no host-to-device copies:
97 device memsets include the 96 state/history buffers plus backend status,
with three small device-to-host status copies and six stream synchronizations.
Allocation dominates initialization (268 cudaMalloc calls); pooling/reuse
remains an open latency target.


### Reusing healthy native requests (2026-11-07)

`bn_transformer_bf16_session_restart` restarts a healthy backend-only session
without allocating or releasing resident buffers. It clears the initial SSM
state/history bank, fences the request stream, resets recurrent parity and
position, and clears host token history. Old attention KV remains allocated;
attention only reads the new prefix up to the current position. Workspace,
weights, rotary storage, and configured capacities remain unchanged. Ordinary
session reset/free still releases request storage. Failed requests require full
reset/reinitialization, including after a partial restart clear or failed fence.
The CLI uses this path for native BF16 benchmark/chat resets and retains full
reinitialization for absent or failed requests.

The real-model runtime fixture compares every vocabulary logit for prefill and
seven decode steps across three restarts, including dirty odd/even recurrent
parity. It verifies no resident allocations or uploads during restart and keeps
a second live session unchanged. Clear failures at operations 1, 2, and 96,
and a failed completion fence invalidate the request without publishing a new
host position/history or logits. Evidence is under
`/tmp/bitnet-goal-backend-reuse/`; broader qualification results are recorded
there separately from this API description.


Protected clean/build and standard tests pass without warnings, as do the
complete CUDA suite, CLI reset/chat/rollover regressions, and legacy Qwen and
Flash checks. State/reference observers now execute a dirty 31-token prefix,
restart the same buffers, and compare 129-token, chunked, resumed, and
1,025-token prefill/decode with the preserved references. The short observer
similarly restarts after five tokens and matches all 516 checkpoints.

Three warmups and ten measured requests give 0.19–0.25 ms warm setup. Median
reset-inclusive first-token times are 54.98/225.99/1237.97 ms for 5/129/1025
prompt tokens; decode is 16.80/16.57/16.77 ms per step. Short ready-to-first is
54.79 ms. Short prefill remains outside the 5% vLLM gate. These timings include
no new arithmetic optimization and do not establish complete serving parity.
A scoped Nsight restart trace verifies zero allocations/frees, 96 device
memsets totaling 153,944,064 bytes, one four-byte device-to-host status read,
one stream fence, and no host-to-device transfers.

The complete real-model runtime fixture, including restart replay and all
injected failures, also passes CUDA memcheck with zero errors and leaked bytes.


### Short-prompt CUDA execution (2026-11-07)

Native BF16 Q4_K/Q5_K matrix operations now use a 16-row by eight-token tile
for prompts up to eight tokens on SM80 and newer devices. Four warps compute
independent two-group affine terms; warp zero adds those F32 terms in the
original pair/block order. Shared F32 intermediates preserve the existing
rounding contract. Larger prompts retain the existing tiles, and pre-SM80
selection retains DP4A. Quantization and MMVQ/MMQ precision thresholds are
unchanged. No request-time allocations, packed-weight growth, or intermediate
host transfers are added.

Q4_K/Q5_K/Q6_K and IQ4 native BF16 vector kernels process four independent rows
per block, retaining each warp's original arithmetic and guarding padded row
tails. Contiguous F32 normalization retains the original logical reduction
lanes/tree, while short batches use additional physical threads for independent
output elements. CPU, legacy model execution, and backend ownership are
unchanged.

The investigation preserved rejected loop-unrolling and 64-row tile candidates.
Interleaved measurements share one prepared model/session and alternate control
and candidate order to reduce drift seen between separate processes. Profiling
and experiment evidence is under `/tmp/bitnet-goal-short-prefill/`; production
qualification and fresh baseline evidence are under
`/tmp/bitnet-goal-short-integrated/`. Nsight Systems traces are available;
Nsight Compute hardware-counter access was denied by the installed driver.


The integrated build passes protected clean/build with zero warnings, the
standard and full CUDA suites, CLI regressions, all preserved short/long/
chunked/resumed state and activation comparisons, and legacy Qwen/Flash
checks. All 516 short checkpoints (8,391,680 BF16 words) remain exact. The
real-model initialization/restart/execution failure fixture passes normally
and under memcheck with zero errors or leaked bytes. Q4/Q6, Q5, and contiguous
F32 normalization race checks report zero hazards. The quant kernels compile
independently for SM75/SM80/SM90; earlier whole-backend SM80/SM90 build debt
remains outside this qualification.

A fresh optimized vLLM run uses the same pinned image/plugin, headless GPU,
GGUF, BF16 activation/KV and F32 SSM configuration, context 2048, concurrency
one, eight greedy outputs, and disabled prefix caching. It retains automatic
FlashInfer GDN prefill, CUDA GDN decode, compilation, and full/piecewise CUDA
graphs. Both engines use three warmups and ten measured requests.

| Prompt tokens | Native ready-to-first (ms) | vLLM scheduled-to-first (ms) | Native decode (ms/step) | vLLM decode (ms/step) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 51.91 | 49.56 | 16.70 | 18.18 |
| 129 | 231.91 | 381.45 | 16.55 | 18.31 |
| 1025 | 1279.46 | 2576.21 | 16.73 | 18.18 |

All three measured cases pass the backend-ready latency gate; the short case
is 4.75% slower, and decode is faster in every case. Including native reset
makes short first-token time 52.11 ms, 5.16% slower than vLLM's scheduled
metric, narrowly outside the gate. This is not full serving qualification.
The optimized 129-token vLLM output differs from both the canonical eager/
Triton sequence and the older optimized capture; its ten measured repeats
agree. All three output profiles are retained. Native execution continues to
match the unchanged canonical reference. No numerical equivalence claim is
made for the optimized 129-token profile.

The production short trace selects 111 Q5 and 61 Q4 pair-split matrix kernels.
It records 48.05 ms summed GPU kernel time, 20 bytes host-to-device and
496,648 bytes logits/status device-to-host, with zero global allocations.
Evidence and source hashes are under `/tmp/bitnet-goal-short-integrated/`,
including the fresh baseline in `reference/` and scope in `qualification.json`.
Expert placement, Flash Next native execution, continuous batching/KV
scheduling, other target architectures/models, and broader parity remain open.


### Native Flash Next normalization foundation (2026-11-07)

Resident BF16 normalization now supports contiguous widths from 2048 through
130560, divisible by four, with the pinned F32 mean reduction. At widths of
8192 and above Torch splits work across both block axes; the CUDA operation
preserves the X-then-Y reduction order. Wider reductions require a multi-CTA
reference algorithm and are rejected. Common widths 2560/4096/5120 retain
compile-time specialization. The existing 5120-wide golden fixtures are
unchanged.

Both resident RMSNorm plans add `weight_groups`: zero/one keeps shared gamma;
otherwise row i selects gamma[i % groups], with weights [groups, dim] and a
row count divisible by groups. Grouped gamma requires CONTIGUOUS_F32 reduction.
This supports Flash Next's four HC streams with separate per-stream affine
weights. Gamma remains immutable backend-owned storage; outputs remain
request-owned BF16 views. Invalid dimensions, group counts, short weights,
aliases, and foreign views reject without queueing or poisoning a healthy
request. Rebuild clients/objects using the extended resident-plan structures.

Independent fixtures from the pinned Torch/vLLM image cover 488 shared-width
cases (218,438,016 exact words) and 160 grouped cases (122,555,664 exact words),
including non-power-of-two widths/groups, batch/reduction boundaries, and
separate/fused residual forms. Offset, unfused cases also match vLLM's actual
GroupedGemmaRMSNorm module. The original 5120 fixture checks another 38,799,360
words unchanged. Reference generators and frozen hash headers live in test/;
captures, identities and commands are under `/tmp/bitnet-goal-flash-wide-norm/`.

Protected clean/build and standard/CUDA suites pass without warnings. CLI,
short/long/chunked/resumed model-state comparisons, the native runtime failure
fixture, and legacy Qwen/Flash checks pass. Dedicated wide/grouped memcheck and
racecheck report zero errors, leaks, or hazards. The normalization kernels
compile independently for SM75/SM80/SM90; whole-backend older-architecture
qualification remains open. A scoped five-token/four-stream normalization
trace records two kernels and one explicit fence, with zero allocations and
zero host/device transfer bytes inside the measured region.

The Qwen27 rerun measures ready-to-first 50.21/223.06/1225.62 ms and decode
15.94/15.61/15.85 ms per step for 5/129/1025 prompt tokens. Short reset-inclusive
time is 50.38 ms. These remain within the recent optimized vLLM comparison
gates; run-to-run variation is not attributed to a speedup from this API change.
Full serving/tail/concurrency qualification remains open.

Flash Next's checkpoint reports four HC streams, low rank 320, and PLE head
width 160. Its native BF16 layout remains disabled: typed HC composition, PLE,
sparse attention, expert execution and bounded placement still need integration
and qualification. Existing F32 cache/legacy support remains available. This
normalization work does not establish end-to-end native Flash Next support.

### Flash Next resident HC post-projection stages (2026-11-07)

The CUDA resident API now provides scaled SiLU, hyper-connection stream mixing,
and residual injection/combine. These preserve the reference BF16 rounding
boundaries and stream-mean reduction order for 2/4/8/16 streams. Inputs and
outputs remain backend/request-owned device views; invalid dimensions, aliases,
extents and request ownership reject before execution. The added backend vtable
entries require rebuilding consumers.

Independent fixtures exercise the pinned common GatedResidual implementation
with normalization and linear projections isolated. Across 96 geometries,
36,475,136 BF16 output words match exactly. This qualifies post-projection
operations, not projection arithmetic or a full HC block. Standard and CUDA
suites, CLI, short/long/chunked/resumed model references, native runtime failure
checks and legacy Qwen/Flash regressions pass. Dedicated memory and race
sanitizers report zero errors, leaks or hazards. Isolated kernel compilation
passes SM75/80/90; runtime entry points require SM80 or newer.

A five-token, 2560-wide, four-stream primitive trace records three kernels
(3.296 microseconds total GPU time), one explicit fence, no allocations and no
host/device copies inside the capture. It excludes projections, normalization,
setup and result reads. These timings do not establish full-model throughput.
Qwen27 ready-to-first medians for 5/129/1025 prompt tokens are
50.04/221.79/1220.99 ms, with decode 15.80/15.46/15.73 ms per step; run-to-run
variation is not attributed to HC primitives unused by that dense model.
Evidence is under `/tmp/bitnet-goal-flash-hc/`.

The actual Flash Next shards contain 77,017,907,200 bytes of expert tensors,
28,800,138,240 bytes in the IQ4_NL per-layer positional embedding table, and
5,505,584,640 other tensor bytes. Placement must budget this table separately
from experts, plus decoded layouts, workspaces and request state. HC down/up
projections are Q8_0, injection and normalization weights F32. Typed projection
semantics, complete HC composition, PLE, sparse attention, expert execution and
bounded placement remain open; native Flash Next stays disabled pending full
independent correctness qualification. Serving and broader vLLM parity remain
open.


### Flash Next checkpoint HC projection references (2026-11-07)

`test/reference_cuda_bf16_hc_checkpoint.py` captures complete, unchanged common
GatedResidual blocks with actual checkpoint weights and synthetic hidden states.
The explicit profile decodes GGUF tensors into BF16 parameters and disables
reduced-precision GEMM accumulation. It covers layers 0/2/47, attention/FFN
branches and 1/5/17/129 tokens, retaining normalization, all three projections,
scaled activation, mixed output and combined output. Hooks observe results;
they do not replace computations. All saved stage values are finite and a
second capture leaves existing binary outputs unchanged.

`test/reference_cuda_bf16_hc_checkpoint.c` compares the existing native BF16
GEMM entry point with each captured projection input/output: 72 comparisons,
9,634,368 BF16 words, zero differences. Evidence and the decoded-weight/source
identities are under `/tmp/bitnet-goal-flash-hc-projection/`. This supports a
decoded-BF16 HC projection contract. It does not qualify Q8 activation
arithmetic, resident HC composition or end-to-end Flash Next execution. The
pinned GGUF plugin used for dense comparisons has no Flash Next adapter;
these explicit checkpoint references must not be described as such an adapter.

### Resident BF16 HC composition (2026-11-07)

The quant registry now declares direct BF16 linear arithmetic. Its planner
selects BF16 GEMM with F32 accumulation for every token count, including decode;
Q8_0 and other quantized activation contracts retain their existing dispatch.
Callers supply an immutable backend-owned BF16 matrix in the decoded handle.
No conversion or weight admission occurs during the resident operation.

`test/reference_cuda_bf16_hc_resident.c` composes grouped normalization,
down projection, scaled SiLU, up projection, stream mixing, injection projection
and residual combine using request-owned intermediates. Actual checkpoint
weights and input/block tensors are uploaded before execution. No captured
intermediate is supplied to the native chain. Across the 24 checkpoint blocks,
all seven intermediate/final outputs match: 30,938,688 BF16 words exactly.
This qualifies isolated common HC composition; model loading, PLE, sparse
attention and expert integration remain necessary before native Flash Next
can be enabled. Evidence is under `/tmp/bitnet-goal-hc-resident/`.

Protected clean/build and standard/CUDA suites pass without warnings. CLI,
short/long/chunked/resumed independent state checks, the 8,391,680-word short
checkpoint trace, quantized/chunked sanitizers and legacy Qwen/Flash parity
checks pass. Direct BF16 matrix fixtures also exercise missing weights,
unaligned output rejection, A8-only rejection, guarded writes and healthy retry.
Dedicated complete-HC memcheck reports zero errors/leaks; racecheck reports
zero hazards. A five-token chain after three warm-ups records nine GPU kernels
for seven operations (including two cuBLAS split-K reductions), 33.920 us of
summed GPU kernel time, one explicit fence and zero allocations/copies inside
the capture. Workspace and weights are prepared beforehand; result reads occur
afterward. This isolated trace is not an end-to-end latency benchmark. The
initial trace with lazy cuBLAS module loading is retained separately.

### HC preparation from raw checkpoint weights (2026-11-07)

The registry declares Q8_0 and F32 decoded-BF16 layouts independently of matrix
arithmetic. CUDA preparation converts raw Q8_0 through exact F32 scale/product
and one BF16 RNE step; F32 tensors use one BF16 RNE conversion. Q8_0 linear
execution retains its existing A8 dispatch. Staged HC matrices explicitly use
the BF16 arithmetic descriptor.

`BnBackendBF16HCWeights` owns the normalization, down, up and injection layouts.
Initialization validates complete mix/combine shapes and embedded scales,
checks a caller-supplied peak tensor-payload budget before allocation, then
uploads/converts one tensor at a time and releases temporary raw handles.
Already-BF16 inputs reuse their upload once. Failure releases partial resources
and leaves the output unchanged; free is repeatable. The budget covers retained
plus temporary raw/decoded tensor payload, excluding allocator/context overhead.
It is a preparation primitive, not an aggregate model/expert placement policy.

The resident checkpoint probe can now open the original GGUF shards and invoke
production preparation. Captured decoded weights are used only for comparison,
never as inputs in this mode. The six sampled HC branches match 39,628,800
prepared BF16 weight values and 30,938,688 chain intermediate/final values
exactly. Each branch retains 13,209,600 bytes, with a preparation payload peak
of 16,609,280 bytes. The default CUDA fixture also checks 16,777,216 Q8 outputs
covering every finite FP16 scale crossed with every signed Q8 code (nonfinite
scale slots replaced by zero), plus 8,192 finite F32 input patterns. Ownership,
all upload/conversion failure points, budget rejection and invalid descriptors
have synthetic tests. Evidence is under `/tmp/bitnet-goal-hc-weight/`.

This prepares complete attention/FFN HC branches. Attaching them to the full
native Flash model container, its output mixer without injection, PLE, sparse
attention, recurrent gating and typed expert execution remain open. Native
Flash stays disabled pending complete independent model qualification.

The final protected clean build and standard/CUDA suites pass without warnings,
including architecture checks, all short/long/chunked/resumed references, CLI,
quantized/chunked sanitizers and legacy Qwen/Flash parity. All 24 raw tensor
hashes used in the sampled independent captures remain unchanged. Dedicated
raw-checkpoint HC memcheck/racecheck report zero errors, leaks or hazards; the
exhaustive conversion fixture also passes memcheck with zero errors/leaks.
The warmed five-token chain records nine kernels, 34.079 us summed GPU kernel
time, one explicit fence and zero allocations/copies inside the capture.
Preparation, workspace and input uploads, weight comparisons and result reads
are outside that region. No end-to-end performance improvement is claimed.

### Model-owned HC preparation and output mixer (2026-11-07)

The existing backend precision resource container now owns all attention/FFN HC
layouts plus the final mixer. Preparation validates every source and computes
aggregate retained/peak tensor payload before GPU allocation, then attaches the
container transactionally. A distinct HC getter exposes these immutable weights.
The runnable BF16 getter stays NULL for HC-only preparation; dense preparation
rejects a live HC container. GPU release frees both arrays and the output mixer.
The mixer has an explicit preparation entry point without injection and rejects
an injected source rather than inventing weights.

Synthetic ownership tests exercise all 38 upload/conversion failure points in
a two-layer collection, invalid late-layer input before allocation, aggregate
budget rejection/exact acceptance, repeated initialization, and backend release.
All standard tests and the separate public-mixer checks pass. The real Flash
checkpoint prepares 96 layer branches plus the output mixer with retained
payload 1,281,249,280 bytes and a sequential preparation peak of 1,284,730,880
bytes. These figures exclude allocator/context and request/cache overhead.

The independent common-HC generator adds an output-mixer mode using the actual
checkpoint tensors and unchanged mix implementation with combine disabled.
Repeated branch and mixer captures remain byte-identical. Native execution
borrows model-owned weights for the sampled six branches and final mixer:
46,192,640 BF16 weight values and 34,538,048 intermediate/final values match
exactly across 1/5/17/129-token prompts. Evidence and scope are under
`/tmp/bitnet-goal-hc-model/`. Full-model execution, PLE, sparse attention,
recurrent/expert integration, bounded expert placement and serving remain open;
HC-ready ownership does not enable native Flash inference.

The final clean build, standard/CUDA suites, CLI, all short/long/chunked/resumed
reference gates, quantized/chunked sanitizers and legacy parity pass without
warnings. Dedicated full-container memcheck reports zero errors/leaks and
racecheck zero hazards, including final-mixer execution and backend teardown.
All 27 sampled raw tensor identities match the independent captures. The
warmed sampled five-token branch trace records nine kernels, 33.760 us summed
GPU kernel time, one explicit fence and zero allocations/copies inside capture.
Complete model preparation and readback are excluded; no full-model latency or
throughput claim follows from this isolated trace.

### Batched PLE IDs and host row staging (2026-11-07)

A batch PLE hash helper accepts independent request contexts, irregular query
boundaries and empty requests. EOS cuts older history within each request.
Products wrap in unsigned 64-bit arithmetic; explicit signed remainder matches
the pinned Torch/Triton contract without signed-overflow undefined behavior.
All dimensions, IDs, head ranges and output aliases validate before writes.
Seventy-two independent pinned NVIDIA Triton cases cover n-grams 2/3/8, head
counts, prompt boundaries, EOS and signed overflow: 41,040 IDs match exactly.

The backend quant helper stages raw selected RAM-table rows without allocating
or dequantizing. It derives row bytes from registry metadata, rejects external
scale layouts, validates every ID before copying, and zero-pads to 16-byte
strides. IQ4_NL rows of width 160 occupy 90 raw bytes and 96 staging bytes. The
actual Flash checkpoint uses 16 heads, hence 1536 staging bytes per token. For
1/5/17/129-token batches, all 2,432 IDs and 233,472 padded bytes match independent
checkpoint captures. One reusable 198,144-byte staging allocation suffices for
the largest case; this does not bound OS page-cache residency or the whole
model's RAM usage. The 28.8 GB table remains host-mapped in this probe.

Address/undefined-behavior instrumentation found unaligned serialized PLE
metadata reads. Production loading now copies layer/multiplier/offset/vocabulary
values safely from byte-addressed arrays. Strict sanitizer runs cover production
Flash loading, actual selected-row staging, synthetic overflow cases and invalid
input guards without errors. Captured decoded BF16 rows are retained for the
next GPU-decoding check; native GPU row decoding/request integration is not yet
implemented. Native Flash execution remains disabled. Evidence and scope live
under `/tmp/bitnet-goal-ple-rows/`.

The final protected clean build, standard/CUDA suites, CLI, independent
short/long/chunked/resumed state gates, quantized/chunked sanitizers and legacy
Qwen/Flash parity pass without warnings. Both independent reference captures
repeat unchanged. GPU kernels and request interfaces did not change in this
step; staging sizes are byte counts, not measured transfer/latency results.

### Resident selected-row IQ4_NL decoding (2026-11-07)

The resident GPU API now decodes padded raw rows from a request-owned I32 byte
container directly into a request-owned BF16 output. The initial CUDA profile
uses the registry's IQ4_NL F32-dequantize/BF16-RNE contract, without weight
handles, per-call allocation or implicit readback. It validates exact extents,
row stride, format, owners, aliases and overflow before launch. The new vtable
entry requires rebuilding consumers; the protected clean build does so.

Fifty-two synthetic/frozen-reference cases check 592,128 values across row and
block boundaries, padded widths, duplicates and reversed selections. Nonzero
view offsets, guards, read-only inputs, invalid plan rejection and healthy
retry are covered. The actual-checkpoint host hash/gather/upload/decode probe
matches all 389,120 captured BF16 values for 1/5/17/129-token batches, together
with the unchanged IDs and packed bytes. It allocates GPU buffers once:
198,144 bytes for raw staging and 660,480 bytes for BF16 output, then reuses
prefix views. The 28.8 GB source table remains mapped on the host.

The new kernel compiles independently for SM75/80/90; its runtime entry point
requires SM80 or newer. This does not qualify the whole backend on older
architectures. Evidence is under `/tmp/bitnet-goal-ple-device/`. Bounded pinned
host staging, asynchronous transfer integration and the rest of PLE execution
remain open. Native Flash is still disabled pending complete qualification.

The protected clean build, standard/CUDA suites, CLI, short/long/chunked/resumed
state references, quantized/chunked sanitizers and legacy parity pass without
warnings. Dedicated synthetic and actual-checkpoint memcheck/racecheck report
zero errors, leaks or hazards. The new kernel uses 14 registers with zero stack
or local spills (compiler reports 1024 shared bytes).

A warmed five-token trace includes the real selected-row upload: one kernel
(1.184 us GPU time), 7680 bytes host-to-device, no device-to-host bytes, and no
allocations/frees inside capture. The current pageable-buffer transfer helper
fences after upload; a decode fence makes two explicit fences total. Host
hash/gather, setup and comparison reads are excluded. This exposes the remaining
transfer synchronization work and establishes no end-to-end latency speedup.

### Bounded pinned upload leases (2026-11-07)

Request-owned upload pools reserve one to eight fixed pinned slots under a
caller-supplied payload budget. Reservation is transactional and matching calls
are idempotent; live pools do not resize. Acquire returns a writable generation-
tagged lease, or BUSY while the slot's prior DMA remains incomplete. Submit
revokes the lease, queues H2D and records a completion event without a stream
fence. Cancel abandons an acquired lease. Stale/foreign/modified leases and
oversized destinations reject without queueing work. Teardown completes pending
work before destroying events and releasing pinned memory. Page, allocator and
event overhead are outside the payload budget. The new vtable entries require
a full consumer rebuild.

The actual PLE probe gathers directly into two pinned slots of 198,144 bytes
each (396,288 bytes total payload), reuses its fixed GPU input/output buffers,
and still matches all 389,120 independent BF16 values plus IDs and raw bytes.
Lifecycle tests cover budget/resize rejection, cancellation, stale generations,
request ownership, duplicate submission, event-driven reuse, and pending-copy
teardown. An 8 MiB test observed BUSY before DMA completion and verifies safe
buffer reuse. This establishes asynchronous submission, not concurrent GPU copy
and model compute. Native Flash request integration and the remaining PLE/model
operations are still open; native Flash remains disabled. Evidence is under
`/tmp/bitnet-goal-ple-upload/`.

The protected clean build, standard/CUDA suites, CLI, short/long/chunked/resumed
state references, quantized/chunked sanitizers and legacy parity pass without
warnings. Dedicated unit and actual-checkpoint memcheck/racecheck report zero
errors, leaks or hazards. A scoped lifecycle audit records four pinned
allocations/frees and four event creations/destructions, including pending-copy
teardown. Host writers must relinquish leases and stop before request teardown.

The warmed five-token pinned upload/decode trace records one kernel (1.184 us
GPU time), one 7680-byte H2D copy, no D2H and no allocations/frees. Submission
has no stream fence; one final decode fence replaces the prior two. Event query
and record calls remain visible. Setup, host hash/gather and comparison reads
are excluded. This proves the asynchronous boundary and resource lifetime, not
an end-to-end throughput or latency improvement.

### Resident PLE dilated convolution (2026-11-07)

A request-resident CUDA primitive now evaluates a contiguous request chunk's
PLE convolution, SiLU, inner residual, and outer residual in one kernel. It
preserves the reference BF16 materialization boundaries and produces a separate
chronological next-history buffer, allowing the caller to commit only after
successful completion. Kernel sizes 2–4 and dilations 1–8 are accepted; outputs
must be disjoint from inputs and each other. The primitive queues work on the
request stream without allocating, copying, or synchronizing internally.

Independent pinned NVIDIA reference captures cover 216 combinations of channel
widths 1/33/10240, kernel sizes 2/3/4, dilations 1/3/8, token counts 1/5/17/129,
and two input patterns. The CUDA test checks 68,054,976 exact BF16 output/history
values across whole-chunk and irregular resumed execution. Guard, invalid-plan,
and input-preservation checks also pass, as do CUDA memory and race checks.
The reference fixture explicitly selects an active state slot: slot zero is
reserved by the reference backend and silently omits the convolution.

This qualifies the convolution primitive, not the complete Flash PLE block.
The PLE gate/projections, native Flash orchestration, sparse attention, and
expert placement remain open. Evidence and immutable captures are under
`/tmp/bitnet-goal-ple-conv/`.

A protected clean build and the standard/CUDA suites pass without compiler
warnings. Real-model CLI, all four state/continuation cases, independent short
replay, chunked CUDA sanitizer checks, and older Qwen/Flash compatibility gates
also pass. A scoped warm 129-token, 10,240-channel trace contains one kernel,
zero copies or allocations, and only the probe's explicit completion fence.
This trace is a primitive execution check, not an end-to-end throughput result.

### Resident PLE gate and grouped normalization (2026-11-07)

The CUDA PLE gate keeps normalized key/query scratch, the gated value, and its
normalized convolution input in request-owned BF16 buffers. It queues two
existing grouped-normalization kernels, one batched dot/gate/value kernel, and
one grouped-normalization kernel. Shared values are broadcast across HC streams;
each stream uses its own immutable key/query/convolution gamma. The path
preserves eager BF16 materializations for normalized key/query, their products,
dot, scaled dot, signed square root, sigmoid, gated value, and final norm.
The qualified profiles are widths 2560/4096 and one/four streams. All buffers
and weights are validated before writes; writable views must be disjoint.
There is no internal allocation, transfer, or synchronization.

Both pinned NVIDIA fused and eager tensor references are retained for 36 cases
with 1/5/129 tokens, one/four streams, and general/negative/zero gates. Gate
outputs agree between both independent paths across all 13,478,400 values;
their normalized outputs differ at 44 values. The native path matches all
26,956,800 eager BF16 gate and normalized values exactly. This follows the
runtime's existing eager contiguous-F32 reduction contract; it does not claim
bitwise equivalence to the fused normalization variant. Guards, read-only
inputs, invalid-plan rejection before writes, and CUDA memory/race checks pass.

This is a resident gate primitive, not complete native Flash execution.
Actual-checkpoint key/value projections and a complete embedding/projection/
gate/convolution chain still need independent qualification and production
ownership/orchestration. Evidence is under `/tmp/bitnet-goal-ple-gate/`.

The protected clean build, standard/CUDA suites, real-model CLI, four state/
continuation cases, short replay, chunked sanitizer checks, and older Qwen/Flash
compatibility gates all pass without compiler warnings. A scoped warm gate
trace at 129 tokens, four streams, and width 2560 contains four kernels, zero
copies or allocations, and only the probe's explicit completion fence. This
is a primitive execution check; complete Flash/vLLM parity remains unproven.

### Actual-checkpoint resident PLE chain (2026-11-07)

An independent checkpoint capture and native comparison probe now compose PLE
hash IDs, selected IQ4_NL embedding rows, key/value projection, eager BF16 gate,
and dilated convolution. Both read the actual Flash Next GGUF weights; the
native chain computes every intermediate itself. Captures supply comparisons,
never replacement weights or intermediate results. Hidden activations and
initial history use deterministic synthetic inputs, so this is module-level
qualification with actual checkpoint weights, not a full-model replay.

The merged key/value projection is numerically significant. Independent
captures show 10/7/56/923 differing BF16 projection values for separate versus
merged GEMMs at 1/5/17/129 tokens, respectively; the 129-token chunk64/decode
profiles differ at 311/772 values. The native probe stacks raw Q8 key/value
weights, decodes the merged matrix once to BF16, performs one merged GEMM,
and splits the output on-device. It matches the merged reference exactly.

All six execution cases match 23,644,160 BF16 values across embedding, key,
value, gated value, normalized convolution input, final output, final history,
and unchanged hidden input. Cases include whole prompts of 1/5/17/129 tokens,
a 129-token prompt in 64-token chunks, and token-by-token execution. Each case
is compared against its own independently executed precision/shape profile;
this does not assert shape-independent floating-point rounding.

The explicit reference profile is decoded-BF16 checkpoint weights, merged
Torch F.linear, eager gate materializations, and pinned NVIDIA convolution.
It does not establish a Flash GGUF plugin or optimized fused-gate equivalence.
Two pinned host staging slots bound row-upload payload to 396,288 bytes;
the complete 28.8 GB embedding table is not uploaded. This bounds staging,
not the operating system's mmap page cache or the future expert-placement budget.

Backend-model-owned PLE preparation, request orchestration, sparse attention,
and expert placement are still required before native Flash can be enabled.
The new sources are `test/reference_cuda_bf16_ple_chain.py` and `.c`; evidence
and immutable captures are under `/tmp/bitnet-goal-ple-chain/`.

Protected clean build and standard/CUDA suites, all real-model correctness
regressions, and older Qwen/Flash compatibility gates pass without compiler
warnings. The complete chain also passes CUDA memory/leak and race checks.
A scoped warm 129-token whole-chunk trace contains nine kernels (including
cuBLAS split-K GEMM and reduction), one 198,144-byte H2D copy, no D2H copies or
allocations, and only the probe's explicit completion fence. This is a single
execution trace, not an end-to-end throughput or vLLM parity result.

The uninstrumented probe observed zero staging-slot waits for whole/chunk64
profiles and 30 explicit caller waits for its queued token-by-token fixture.
Wait counts vary with timing; the probe reports them rather than assuming
readiness. Production request scheduling must handle lease BUSY asynchronously
and avoid adopting this diagnostic caller's blocking retry policy.

### Backend-owned PLE weight preparation (2026-11-07)

PLE preparation now extends the existing backend-owned HC precision container.
It preserves the merged Q8 key/value projection, decodes it to one immutable
BF16 matrix, and stages grouped BF16 key/query/convolution gamma plus convolution
weights. Temporary raw GPU uploads are released after decoding. CPU model
weights remain unchanged; sessions borrow backend handles. The partial HC/PLE
container deliberately remains non-runnable until the rest of Flash is qualified.

Preparation checks the complete supported profile and both budgets before
allocating PLE buffers. The GPU budget covers aggregate HC/PLE preparation
payload; the separate host budget covers temporary raw concatenation. Failure
releases only new PLE allocations and preserves the existing HC container.
Tests exercise all ten raw-upload/decode failure points, insufficient budgets,
foreign source/GPU identities, duplicate preparation, and teardown.

Actual-checkpoint PLE retention is 65,679,360 bytes, with a 100,352,000-byte PLE
GPU preparation peak and 34,816,000-byte temporary host concatenation. Including
HC, retained GPU tensor payload is 1,346,928,640 bytes and preparation peaks at
1,381,601,280 bytes. These are payload budgets; allocator/context overhead and
mmap page cache are excluded. The six complete chain cases still match all
23,644,160 BF16 reference values using the production-owned weights.

Request buffers and chain orchestration still live in the diagnostic probe.
Production request orchestration and the remaining Flash execution paths are
next; this preparation API does not enable Flash inference or prove serving
performance parity. Evidence is under `/tmp/bitnet-goal-ple-owner/`.

The fresh clean build, final standard suite, full CUDA suite, real-model CLI,
all state/continuation cases, and older Qwen/Flash compatibility gates pass
without compiler warnings. ASan/UBSan with leak detection covers preparation
failure and teardown; it also exposed a missing cleanup in the existing GPU
policy test fixture, which is now fixed. The actual owned-weight chain passes
CUDA memory/leak and race checks. Its warm trace retains nine kernels, one
198,144-byte H2D upload, no D2H/allocations, and one explicit probe completion
fence. This remains module evidence, not full-model or serving parity.

### Production PLE request state and submission (2026-11-07)

The explicit BF16 runtime now owns PLE request buffers and orchestration in
`src/transformer/bf16.c`. Request preparation reserves reusable packed rows,
embedding/projection/gate buffers, two chronological history banks, pinned
staging slots, and GEMM workspace through the existing BackendSession resource
contracts. Upstream GPU work can write directly into the borrowed hidden view;
submission queues the full resident chain without a host activation bridge.
The partial operator still does not admit a complete Flash model.

Submission and completion are separate. BUSY returns before hashing or queuing
work, with no internal wait. Successful submission records pending state;
position, predecessor context, and history parity commit only after an explicit
successful completion fence. Pending requests reject another submission/reset.
Execution/completion failures invalidate the operator until ordinary session
release/reinitialization. Healthy reset imports explicit prefix state or uses
EOS context and zero history without reallocating request storage. The pinned
reference model's context initialization also fills EOS.

Budget preflight includes the backend's GEMM workspace through a new pure-host
size query. At capacity 129, GPU payload is 57,238,016 bytes: 23,683,584 bytes
of tensors/history plus 33,554,432 bytes of GEMM workspace. Host pinned/ID
payload is 404,544 bytes, including 396,288 bytes of staging. Shared weights,
fixed host metadata, allocator/event overhead and driver/context overhead are
excluded. Preparation rejects insufficient budgets before request creation;
all 13 buffer-allocation failure points release newly created resources.
Packed-row decoder admission is a backend-quant policy helper using registry
metadata, so transformer code does not inspect raw quant formats.

The actual-checkpoint probe now invokes the production request API. Whole,
chunk64 and token-by-token profiles run twice with reset/reuse, matching
47,288,320 independent BF16 values and 13,120 row IDs. It also checks invalid
input, BUSY without synchronization, pending-state rejection, failure without
position commit, recovery via session release, and two interleaved requests
with distinct contexts/history and reversed completion order. One request uses
an independent capture; the altered second request is checked against its
isolated native baseline. No complete-model or serving equivalence is claimed.

The protected clean build, standard/CUDA suites, real-model CLI and state/
continuation checks, older Qwen/Flash compatibility, and host ASan/UBSan suite
pass without compiler warnings. The production request probe also passes CUDA
memory/leak and race checks, including allocation rollback and interleaving.
A warm submit/complete trace contains nine kernels, one 198,144-byte H2D upload,
no D2H copies or allocations, and one explicit completion fence. Completion
is still synchronous when explicitly requested; a full serving scheduler and
nonblocking completion mechanism remain outside this operator checkpoint.
Evidence and source hashes are under `/tmp/bitnet-goal-ple-request/`.

### Actual Flash GDN block qualification (2026-11-07)

The pinned Flash model uses natural, non-interleaved GDN heads
(`gqa_interleaved_layout=False`) and sigmoid output gating. Its merged QKVZ
projection has output sizes 2048/2048/6144/6144, and its merged beta/alpha
projection uses 48/48 outputs. The existing dense Qwen27 adapter applies a
head permutation, SiLU gating, and output transpose; that adapter must not be
reused unchanged for Flash.

An independent actual-checkpoint block capture and native resident probe now
cover merged projections, causal convolution, GDN prefill/decode, sigmoid gated
RMS normalization, natural-order output projection, and all carried state.
The existing generic CUDA kernels match exactly without arithmetic changes:
106,012,096 BF16 values and 45,613,056 F32 state values over 58 phases. Six cases
include 1/5/129/1025-token prompts, 64/512-token chunking with carried state,
and eight subsequent decode steps per case. One-token prompt tails still use
prefill preparation; decode uses its distinct F32 Q/K normalization path.

A separate independent CPU audit checks natural-order coefficient preparation
for every one of the model's 36 GDN layers. All 1,728 F32 log coefficients match
the qualified per-call MKL VML HA profile exactly. Parameters and intermediates
are computed independently from GGUF in both paths; capture files are only
comparisons, never substituted native inputs. This is an explicit decoded-BF16
weight profile with synthetic normalized hidden inputs, not full-model replay
or an established Flash GGUF plugin loading contract.

Protected clean build, standard/CUDA suites, real-model state/continuation
regressions and older Qwen/Flash compatibility pass without compiler warnings.
The new block passes CUDA memory/leak and race checks. A warmed 129-token
prefill trace contains 16 kernels, zero copies or allocations, and one explicit
completion fence; it is not an end-to-end performance result.

Backend-owned Flash GDN preparation and request integration still need to
preserve these semantics. QSA/compressed attention and typed/grouped experts
remain required before enabling complete native Flash. New reference sources
are `test/reference_cuda_bf16_flash_gdn.py`, `.c`, and
`test/reference_cuda_bf16_flash_coefficients.py`; evidence and immutable captures
are under `/tmp/bitnet-goal-flash-gdn/`.

### Backend-owned Flash GDN cache (2026-11-07)

The precision container now stages every Flash GDN layer alongside existing HC
and PLE resources. Each valid layer owns merged decoded-BF16 QKVZ and beta/alpha
matrices, output projection, convolution/gated-norm/delta-bias weights, and
natural-order F32 log coefficients. The model descriptor retains sigmoid
activation, geometry, epsilon, and the qualified MKL VML HA math profile.
Non-SSM layer entries remain empty. Complete Flash admission remains disabled.

All layers and both payload budgets are checked before allocating new GDN
buffers. Preparation stages QKVZ, output, BA, then small parameters, releasing
raw GPU uploads after decoding. Failure rolls back only new GDN resources and
preserves HC/PLE. Tests exercise 26 upload/decode failure points across two
layers, insufficient GPU/host budgets, invalid late-layer data/coefficients,
foreign source identity, duplicate preparation and teardown. Builds without
the qualified math profile reject preparation without a silent libc fallback.

All 36 actual GDN layers retain 4,173,024,384 bytes, with a 4,189,162,080-byte
component preparation peak and 44,564,480-byte temporary host concatenation.
Including HC/PLE, retained GPU tensor payload is 5,519,953,024 bytes and maximum
preparation payload is 5,536,090,720 bytes. Shared allocator/context overhead,
requests, attention, embedding/logits and experts are outside these figures.
The owned-weight layer-zero block still matches all 106,012,096 BF16 values and
45,613,056 F32 state values from the independent captures. Other GDN layers are
staged and shape-checked; their full forward replay remains unqualified.

The production GDN request adapter remains next. It must use natural heads,
merged projections and sigmoid gating, extending existing request ownership
without enabling partial Flash. Evidence is under
`/tmp/bitnet-goal-flash-gdn-owner/`.

The protected clean build, standard/CUDA suites, real-model state and continuation
checks, and older Qwen/Flash compatibility gates pass without compiler warnings.
Host ASan/UBSan covers cache rollback/lifetime tests; the actual 36-layer cache
also passes CUDA memory/leak and race checks. The owned-weight block trace
remains 16 kernels, zero copies or allocations, and one explicit completion
fence. These are component results; full Flash request integration, attention,
experts and serving parity remain unfinished.

The CUDA GDN prefill workspace now has a pure host payload query. Admission and
reservation share the same checked sizing function, including chunk64 state
rounding: 13,461,696 bytes at 129 tokens and 96,209,088 bytes at 1,025 tokens.
This is shared operator workspace, separate from per-layer recurrent/history
banks, resident scratch tensors, GEMM workspace and driver/allocator overhead.
Boundary tests cover 63/64/65 tokens, invalid capacities and the maximum accepted
capacity without allocating that capacity. The combined Flash request adapter
and its single outer state commit remain unfinished.

Flash now has an explicit combined PLE/GDN operator request adapter. Preparation
checks payload budgets before GPU work and attaches host metadata only after
all allocations and initialization succeed. It owns double F32 recurrent state
and BF16 convolution history for all 36 GDN layers, plus shared scratch and one
GDN/GEMM workspace each. A 129-token capacity uses 272,903,232 bytes of resident
tensors, 47,016,128 bytes of workspace, and 404,544 bytes of pinned/ID host
payload. Shared model weights, fixed host metadata, driver/allocator/event
overhead and the remaining model components are excluded.

Batch begin selects components at their matching committed positions and
acquires PLE staging before queuing GPU work. BUSY leaves the transaction
unchanged. Each GDN layer may submit once; incomplete completion is rejected
without fencing. One outer completion fence precedes all selected state,
history, context and position commits. An execution/completion error poisons
the request without publishing next banks. Healthy restart clears all state
without reallocating; each layer keeps its own parity and position so partial
diagnostic execution cannot advance untouched layers. PLE-only requests retain
their existing API and behavior.

The production request replay matches 212,024,192 BF16 and 91,226,112 F32 state
values over two reset/reuse passes of six independent captured profiles, with
chunked prefill followed by eight decode steps. The synthetic hidden pattern
depends on phase, so these are per-profile reference matches, not identical
inputs across chunk sizes or full-model chunk equivalence. A combined batch
matches the independent PLE and GDN captures and commits with one fence.
Tests also cover all 171 allocation failure points, disjoint banks for 36
layers in one transaction, and two interleaved requests with reversed
completion order. Only layer-zero GDN has independent full operator math
references; all-layer transaction coverage does not establish all-layer model
correctness. Full Flash attention, experts, model execution and serving parity
remain unfinished; the partial adapter does not enable CLI admission. Evidence
is under `/tmp/bitnet-goal-flash-request/`.

The clean build, CUDA-enabled and CPU-only standard suites, complete CUDA
suite/diagnostics, CLI/state/continuation references and older-model regressions
pass without compiler warnings. Actual request host code passes ASan/UBSan
with leak detection; CUDA initialization requires ASan's `protect_shadow_gap=0`
in this environment (the initial failure and minimal reproducer are retained).
CUDA memory checking reports zero errors/leaked bytes and race checking zero
hazards. All-API Nsight tracing of a warmed 129-token GDN-only transaction
shows 16 kernels, no copies or allocations, and one explicit completion fence.
These establish operator/request behavior, not full-model serving performance.

The Flash QSA side branch now has a single-launch resident CUDA pre-indexer
for sequential text: merged four-head Q / one-head raw K, head dimension 128,
rotary dimension 64, compression ratio 4 and BF16 side storage. It writes a
separate next raw-key ring and appends only completed compressed groups,
preserving the initial ring for an outer state transaction. No allocation,
transfer, fence or position commit occurs inside submission. Gamma offsets,
pooling-to-BF16 and the pinned branch-specific BF16 rotary multiply/FMA
boundaries are explicit. Main attention, sparse selection and Flash request
integration remain separate unfinished work.

The actual layer-three GGUF projection plus pre-indexer matches 36,028,800
BF16 values across 265 phases and 12 profiles, including long 2053/4097 prompts,
whole/chunked prefill, crossing four-token groups and eight decode steps.
The pinned NVIDIA/Triton reference is preserved. GPU-constructed RoPE constants
are used; the earlier CPU-constructed cache differed in two values outside the
covered positions and is retained with its captures.

Identical-input chunk comparisons expose numerical batch-shape differences
in the independent projection GEMM itself. Native matches every shape exactly;
this does not imply bitwise chunk invariance. At 129 tokens, chunk sizes 64/7/1
change 5/9/11 projected values and 4/6/9 normalized Q values, while final
compressed keys agree. At 2053 tokens, 257-token chunks change 841 projected
values, 738 normalized Q values and 52 compressed values (maximum absolute
difference 0.03125). No Q row differs when its projected row is identical.
These differences are retained in the shape audit; full-model routing/token
consequences remain unqualified. Evidence is under
`/tmp/bitnet-goal-flash-qsa/` and `/tmp/bitnet-goal-flash-qsa-fused/`.

The final single-launch QSA path passes the protected warning-free clean build,
CUDA-enabled and CPU-only standard suites, complete CUDA diagnostics, dense
CLI/state/continuation references, combined PLE/GDN replay and older-model
compatibility checks. CUDA memory checking reports zero errors/leaked bytes;
race checking reports zero hazards. A warmed 129-token projection/pre-indexer
trace contains three kernels (two GEMM plus one fused QSA kernel), zero copies
or allocations and one explicit completion fence. No full-model latency or
serving-parity claim follows from these component results.

QSA scoring and packed selection now run on resident CUDA views. The scorer
uses BF16 tensor-core dots with F32 accumulation, ReLU and the pinned adjacent
head-pair reduction order. The selector uses an exact radix threshold and
chronological compaction; equal scores prefer lower group indices, with both
signed zeros treated equally. It emits up to 512 groups expanded to raw-token
indices, the causal partial-group tail, `-1` padding and a trailing valid count
in the 2,052-column buffer. Neither submission allocates, transfers, fences
or commits request positions. Main sparse attention and request integration
remain unfinished.

The pinned vLLM top-k operation has arrival-order-dependent output. Across
12 replays of the same 4,097-token prefill scores, 2,046 rows changed index
order and nine changed membership at tied scores; all initial and replayed
outputs passed exact score-optimality and index-validity checks. Those original
outputs are retained separately from the independent NumPy canonical ranking
oracle. Literal vLLM index-order equality is therefore not the native contract.
The effect of different tied memberships and accumulation order on full
attention/model outputs remains unqualified.

The actual checkpoint pipeline recomputes projection and pre-indexer outputs
before scoring and selection. It matches 2,635,822 visible reference F32 scores
and 13,424,184 canonical packed I32 values, while retaining the earlier exact
36,028,800 BF16 comparisons. Synthetic checks cover 200,406 tensor-core score
values at tile boundaries and 270,864 packed values for ties, unique ranks,
signed zeros and up to 65,536 groups. The initial F32 reduction-order and
signed-zero failures are retained alongside their corrections. Evidence is
under `/tmp/bitnet-goal-qsa-selection/` and
`/tmp/bitnet-goal-qsa-selection-qualified/`. These are operator results, not
full-model correctness or serving-parity claims.

The corrected scorer/selector passes the warning-free protected clean build,
CUDA-enabled and CPU-only standard suites, full CUDA diagnostics, dense
CLI/state/continuation references, combined PLE/GDN replay and older-model
regressions. CUDA memory checking reports zero errors/leaked bytes and race
checking zero hazards. The warmed 4,097-token chain traces four kernels
(GEMM, pre-indexer, scorer, selector), zero copies/allocations and one explicit
completion fence. `test/reference_cuda_qsa_canonical.py` independently verifies
the existing canonical files without overwriting mismatches. Full sparse
attention, model integration and serving-parity work remain unfinished.

The Flash GGUF normalization boundary required a correction before continuing
main sparse attention. The local Qwen4Exp converter stores zero-centered Gemma
norms as F32 `gamma + 1`, including HC, PLE, main Q/K and indexer Q/K norms.
Earlier explicit decoded-BF16 captures treated those stored scales as gamma;
the generic kernels then applied a second affine offset. Those historical
captures remain operator tests for their supplied parameters, and are preserved;
they do not establish the checkpoint normalization contract.

Architecture policy now declares this storage offset. Backend model preparation
recovers gamma in F32 before BF16 rounding, using a temporary norm array and
leaving shared CPU weights unchanged. Standalone HC initialization continues to
accept gamma directly. Ordinary SSM RMSNorm scales are excluded. Host tests check
near-one stored scales, immutable sources, budgets and partial-allocation cleanup.
Fresh independent captures and native comparisons are under
`/tmp/bitnet-goal-gamma-storage/`: 46,192,640 HC weight values, 34,538,048 HC
execution values, 23,644,160 PLE values and 36,028,800 QSA pre-indexer values match
exactly. The corrected indexer pipeline also matches 2,635,822 F32 scores and
13,424,184 canonical packed indices.

A new main sparse-attention reference captures 12 profiles and 265 phases,
including split-K partial outputs and LSE, prefill, chunking and eight decode
steps. It preserves original pinned selections and separately evaluates canonical
selections. Some tied/order-dependent selections change BF16 attention and output
projection values; literal full-model parity therefore remains unqualified.
This reference capture does not implement native main sparse attention, make
Flash runnable, or establish vLLM performance parity.

The corrected PLE references also pass request reset/reuse and session isolation:
47,288,320 BF16 values and 13,120 IDs across 12 cases. Combined PLE/GDN request
replay passes 212,024,192 BF16 values and 91,226,112 F32 state values using the
corrected PLE reference and the unchanged independent GDN reference. The request
probe accepts an explicit PLE capture directory to keep those profiles separate.
The standard suite, architecture gates and full CUDA diagnostics pass; HC and PLE
component memory checks report zero errors and zero leaked allocations.

A private CUDA QSA split-K merge stage is implemented in
`src/gpu_cuda_qsa_attention.cuh`, with a standalone diagnostic in
`test/reference_cuda_qsa_merge.cu`. It consumes F32 normalized partial outputs and
base-2 LSE, retains the distinct denominator butterfly and numerator pair trees,
rounds attention to BF16 before the F32 sigmoid gate, then stores BF16. Compiled
pinned code contracts each numerator pair with the even split product fused into
the rounded odd product. The initial separate-product implementation differed on
20 checkpoint values; explicit preservation of that FMA order fixes them.

The private merge matches 5,505,024 checkpoint BF16 values across all 256 captured
split-K phases and 147,456 synthetic values for 4, 8, 16 and 64 splits. Synthetic
cases include empty and single-populated splits, cancellation, large dynamic
ranges, underflow thresholds and extreme gates. The pinned kernel independently
replays all checkpoint partials exactly. Captures and failing candidates remain
under `/tmp/bitnet-goal-qsa-merge/`; memory/race checks cover the same final cases.
This qualifies the merge stage on the headless SM120 GPU. The paged QK/softmax/PV
stage, bounded resident API, full sparse-attention integration and performance
comparison remain unfinished. The merge is not connected to default inference.
The protected clean build, standard tests and complete CUDA diagnostics pass.
The standalone stage also compiles warning-free for SM80 and SM90; these compile
checks do not qualify runtime behavior or repair existing whole-backend build debt
on those architectures.

The private QSA core now implements paged tensor-core BF16 QK/PV tiles, online
softmax and split-K partial outputs/LSE, together with the previously qualified
merge. The core uses the reference's split counts and F32 probability-sum trees;
its execution uses four warps with query fragments retained in registers.
Normalization uses libdevice `log2f` for LSE. Replacing it with approximate `lg2`,
or changing the register-pair reduction tree, fails exact F32 comparisons.

The corrected checkpoint component matches 163,946,496 F32 partial values,
640,416 F32 LSE values and 54,343,680 BF16 gated outputs across 265 phases.
Nine additional synthetic paged cases cover all seven dispatch profiles,
including the 8- and 16-split configurations, two requests, permuted/shared pages,
invalid requests/pages/indices, empty rows and trailing-count boundaries. They
match another 13,541,376 F32 partial values, 52,896 F32 LSE values and 16,336,896
BF16 outputs. This tests explicit attention-stage inputs; it does not recompute
native projections, normalization or sparse selection as a complete model path.

Aligned 128-bit KV staging, shared score/probability layouts, register softmax
and query reuse improve the initial candidate by about 3.2–20.9x on those
synthetic component timings. A further KV swizzle did not establish a consistent
benefit and was not retained. Warm CUDA-graph timing still measures the native
component at roughly 1.6–3.2x the pinned stage's latency. This does not establish
the requested whole-model 5% parity target. These are initial 100-replay component
measurements, with setup and
comparison transfers excluded, rather than end-to-end performance results.
Captured native graphs have one kernel for single-split execution and two for
partial/merge execution, with no copy or allocation nodes. Evidence, failed
candidates and matched inputs are under `/tmp/bitnet-goal-qsa-partial/`.
The core remains private: bounded resident API, native QK/selection composition,
request integration and full-model/serving qualification are still required.

The protected clean build, standard suite, complete CUDA diagnostics and earlier
merge regression pass. Memory checks report zero errors/leaked allocations and
race checks zero hazards on both checkpoint and paged datasets. The standalone
core compiles warning-free for SM80/SM90; runtime qualification remains SM120.
The final core reports no local-memory or stack spills. These checks do not
qualify a resident ownership/budget API, full Flash execution or serving.

The paged QSA core now has an SM120 resident backend API with explicit one-time
kernel preparation, a pure workspace query, and request-owned typed views.
The caller supplies bounded scratch; submission neither allocates nor grows it.
The maximum scratch requirement is 25,264,128 bytes across supported token caps
up to 131,070. Read-only views may alias; output and scratch must be disjoint.
I32 metadata supports naturally aligned subviews, while BF16/F32 buffers require
16-byte alignment. Invalid plans are rejected before submission, and a valid
retry remains usable. Cache contents and request position are not mutated.

Through this API, all 274 retained checkpoint and paged cases match the same
177,487,872 F32 partials, 693,312 F32 LSE values and 70,680,576 BF16 outputs.
An independent workspace oracle checks 303 capacity queries. The in-tree CUDA
test also checks 16,939,008 BF16 outputs across 11 batch sizes against a
closed-form one-key attention result, reusing a single bounded workspace.
Memory checks cover both captured datasets and the synthetic unit; race checks
cover the paged dataset and unit. The previous private-core qualification also
race-checked the full checkpoint dataset. All report zero errors or hazards.
Nsight Systems confirms one prefill kernel or two decode partial/merge kernels
on the request stream, no copies or allocations, and only the caller's outer
completion fence. Evidence is under `/tmp/bitnet-goal-qsa-api/`.

This API remains separate from default model inference. Native projection,
normalization, selection, paged-cache append and output-projection composition
are still required for full Flash execution. Runtime qualification remains
SM120; standalone SM75/80/90 compilation does not qualify those whole backends.
The earlier component timings remain the performance evidence; this adapter
does not establish whole-model vLLM parity or serving qualification.

Resident QK preparation now accepts independent input row strides, with zero
preserving the existing contiguous layout. This allows Q/gate, K and V to be
overlapping read-only slices of Flash's 13,312-column merged projection instead
of requiring split copies. Input extents exclude padding after the final row;
stride, alignment, overflow and output-alias checks precede kernel submission.
The existing 17 independent contiguous cases still match 35,861,504 BF16 words.
Another 51 merged, padded-merged and independently padded layouts match
107,584,512 words, including invalid-plan rejection, healthy retry and unchanged
input/padding/canary checks.

The actual corrected Flash checkpoint captures also qualify this path across
265 phases: 117,744,640 Q/K/V/gate BF16 words match exactly. Recomputing the
merged projection natively from the retained decoded BF16 checkpoint weights
matches another 117,744,640 projected words and the same QK outputs. Norm inputs
use checkpoint gamma recovered in F32 before BF16 conversion. This qualifies
the decoded BF16 weight profile, not every original packed weight format.
Memory and race checks pass on the synthetic layouts, direct checkpoint path
and native projection/QK chain. Traces show one direct QK kernel or two kernels
for projection plus QK, on one request stream with zero copies/allocations and
only the outer completion fence, for both prefill and decode.
Evidence is under `/tmp/bitnet-goal-qk-stride/`.

Paged-cache append and composition with indexer, selection, attention and output
projection remain necessary before this becomes full Flash inference. These
correctness and execution traces are not new latency benchmarks; the full-model
prefill/decode and serving performance requirements remain open.
The protected clean CUDA build, standard suite, complete CUDA diagnostics and
backend architecture checks pass with the final strided implementation.

Paged K/V append now has a resident backend operation that copies both matrices
in one kernel, preserving raw BF16 bits. It accepts naturally aligned I32 slot
metadata and aligned BF16 rows, validates ownership/extents/aliases, and masks
negative or out-of-range slots. Physical slot zero is valid. The scheduler must
provide unique valid slots within each submission; later ordered submissions
may reuse them. Submission performs no allocation, host transfer, synchronization
or position commit. Ten shapes match an independent pinned Triton scatter
oracle and a CPU byte-copy reference, checking 36,000,384 cache words, unchanged
inputs/padding, read-only aliases, invalid plans and healthy retries.

A native layer diagnostic now composes main projection, QK preparation, indexer
projection/preparation, scoring, canonical selection, paged K/V append, sparse
attention and output projection. All twelve checkpoint schedules and 265 phases
match 437,987,840 BF16 values and 18,149,940 selected indices exactly, including
persistent cache contents across chunked prefill and subsequent decode. Position
and history-bank commit occur after the single outer completion fence. Main and
output projections use the retained decoded-BF16 Q8 checkpoint weights; indexer
projections use the BF16 checkpoint weights. The lower-ID tie/chronological
selection policy remains explicit and differs from the original pinned
atomic top-k ordering. Evidence is under `/tmp/bitnet-goal-qsa-append/`.

This composes production operators in a checkpoint diagnostic. Backend-model
weight preparation, request ownership/budget integration and full Flash model
admission still require implementation. It does not establish expert offloading,
whole-model vLLM parity or serving qualification.
Memory and race checks pass for the append unit and the complete native layer
chain. Traces show nine kernels for prefill and eleven for decode, including
cuBLAS's separate decode output-projection split-K reduction. Each submission
uses one request stream, one fused K/V append kernel, no copies or allocations,
and only the caller's outer completion fence. The protected clean build,
standard suite, full CUDA diagnostics and backend architecture checks pass.

Flash QSA now has immutable backend-owned weight preparation alongside HC, PLE
and GDN. It preflights all applicable layers and aggregate budgets before uploads,
prepares merged main/indexer projections and output projections, recovers norm
Gamma in F32 before BF16 rounding, and owns a shared rotary cache. Failed
preparation preserves existing components. Synthetic ownership tests cover
27 upload/decode failure points, budget boundaries, cleanup, and unchanged CPU
norms; the same tests pass under AddressSanitizer with leak detection.

The real checkpoint prepares all twelve QSA layers, including a 262,144-position
rotary cache. QSA tensor payload is 1,268,271,104 bytes; aggregate HC/PLE/GDN/QSA
payload and preparation peak are 6,788,224,128 bytes, with a 36,208,640-byte QSA
host staging bound. These figures exclude allocator/context/mmap overhead.
All 9,216 QSA norm values match the independent CPU conversion and the CPU norm
arrays remain unchanged. Layer 3 then replays all 265 retained phases using the
model-owned buffers, matching the same 437,987,840 BF16 values and 18,149,940
indices. Model-backed memory checks report zero errors/leaked GPU allocations;
the earlier full-chain race qualification covers the unchanged GPU operators.
Model-backed traces retain nine prefill/eleven decode kernels on one stream,
with no submission-time allocation/copies and only the outer completion fence.
Evidence is under `/tmp/bitnet-goal-qsa-model/`.

The remaining integration is QSA request state, cache/indexer ownership and
budget admission in the production Flash request API. Weight preparation still
does not admit the complete model; MoE/offloading, full-model correctness and
performance, and serving remain unfinished.

QSA now executes through the production Flash request API. Prepared model
weights enable QSA layers in newly prepared requests; existing GDN-only requests
retain their earlier path. An explicit context cap bounds per-layer K/V caches,
compressed indexer caches and history, plus shared activations, selection buffers
and partial workspace. GPU admission checks the full request payload before
allocation; host admission also covers temporary slot-map setup. The slot map
and page table are prepared once, so submission needs no metadata upload.

QSA and GDN use the same expected/submitted-layer transaction. One outer fence
precedes position/history-bank commits; incomplete transactions do not fence.
Execution or completion failure poisons the request without advancing committed
positions. Healthy restart clears state without allocating. PLE reset also
respects the request context cap. The production API matches all 265 checkpoint
phases and the same 437,987,840 BF16 values plus 18,149,940 indices. Additional
checks cover all 239 allocation failure points, exact budget boundaries, context
limits, peer-layer isolation, reversed completion of two distinct QSA requests,
and a five-token mixed GDN/QSA transaction against independent references.

ASAN/UBSAN and GPU memory checks pass for the replay/rollback paths. The legacy
GDN request still passes its original 171-point rollback and 212,024,192 BF16 /
91,226,112 F32 reference values; PLE retains 47,288,320 exact values and 13,120
row IDs. Production request traces retain nine prefill/eleven decode kernels
on one stream, no submission-time allocation or copies, and one outer fence.
Evidence is under `/tmp/bitnet-goal-qsa-request/`.

The Flash request owns attention state and execution, but is still an explicit
operator API. Full HC/FFN/MoE orchestration, expert placement and complete model
admission remain necessary before end-to-end correctness/performance and serving
qualification. These component checks do not establish whole-model vLLM parity.

The production Flash request now owns HC residual-stream flow as well. HC mix
normalizes and projects the wide stream directly into the attention input;
HC combine consumes the block result and advances the wide stream by swapping
existing buffer views. No new GPU scratch or preparation allocations are needed.
Block transactions track attention HC completion and caller-produced FFN block
consumption separately, so an unfinished HC branch cannot commit state. PLE apply
advances the same stream, while the existing raw operator APIs remain available.

A fresh independent reference composes the pinned common HC implementation,
actual layer-3 QSA and HC combine. All twelve schedules and 265 phases match
805,975,220 BF16 values and 18,149,940 indices exactly. FFN-side HC with explicit
external block results matches another 7,102,048 values across four batch sizes;
this checks residual wiring and does not implement MoE computation. PLE residual
advancement and injected execution failures are also checked. ASAN/UBSAN and GPU
memory checks pass. Raw QSA, GDN and PLE regressions remain exact. Traces show
18 prefill/19 decode kernels, including cuBLAS reductions, on one request stream
with no submission-time allocations/copies and one outer completion fence.
Evidence is under `/tmp/bitnet-goal-flash-hc-request/`.

Actual resident BF16 MoE remains next. The current checkpoint's layer-3 routed
gate/up weights are Q4_K and its down weights Q5_1. Q5_1 now declares an explicit
FP32-dequantize/BF16-RNE layout capability; direct raw BF16 linear dispatch is
still unsupported. The existing MoE bridge uses host F32 activations.
These paths need bounded expert placement and a resident block-output contract,
rather than applying the ordinary residual-add FFN path inside HC combine.
Full-model admission, expert-offload metrics, performance parity and serving
qualification remain open.

The Q5_1 layout preparation uses block-32 metadata, accepts the actual 640-column
expert down shape, and returns independently owned, caller-budgeted BF16 storage.
It does not allocate layouts for all experts. Independent gguf-py/Torch captures
check 3,384,640 decoded values exactly, including layer-3 experts 0 and 511, and
803,016 resident dense GEMM outputs at batch sizes 1, 5, 17 and 129. All actual
checkpoint GEMM outputs match exactly. Two synthetic cancellation outputs differ
by one BF16 step between FP32 reduction orders; the original Torch captures are
preserved and the FP64 midpoint audit is recorded. Layout comparisons stay exact;
GEMM comparisons allow one BF16 step. Truncated payloads and incomplete blocks
are rejected. The oracle and replay sources are
`test/reference_cuda_bf16_q51_layout.{py,c}`; evidence is under
`/tmp/bitnet-goal-flash-moe/`. The full inventory also records Q5_K gate/up at
layer 2 and Q8_0 down at layers 2, 4, 30, 46 and 47, so resident MoE planning must
use per-layer format metadata.

The clean CUDA build and standard suite pass without warnings; the complete
CUDA suite, diagnostic isolation and Q5_1 replay under GPU memcheck also pass
with zero memory errors or leaks. These checks qualify layout preparation and
its resident GEMM consumer, not complete resident MoE execution or performance.

The optional resident BF16 MoE top-k operation is now qualified on SM120 for
512 experts and top-10 routing. It consumes device logits and produces F32
weights plus I32 expert IDs and source-row mappings without allocations,
transfers or an internal fence. It preserves the pinned implementation's
small-batch sorted profile through 128 tokens and native warp profile above
128 tokens, including probability-based ranking, lower-ID ties, optional
renormalization and zeroing nonfinite probabilities.

Independent captures use the inherited `ReplicatedLinear` constructor, actual
checkpoint weight loader and original `fused_topk`. The loader converts F32
checkpoint gate storage to BF16 parameters under the BF16 model configuration;
F32 storage alone must not select F32 router arithmetic. All 38 cases match
193,440 routing values/indices exactly across repeated calls. Coupled resident
gate/shared-gate projections across layers 0, 2, 3 and 47 match 1,108,080 BF16
values and another 64,800 routing values/indices exactly. Tests cover the
128/129 boundary, ties, underflow and nonfinite rows, invalid views and request
ownership. The clean build/standard suite and full CUDA suite pass without
warnings; GPU memcheck, racecheck and synccheck report no errors, leaks or
hazards. Warm traces show one top-k kernel with no submission allocations or
copies and one explicit outer completion fence. This is not a latency/parity
benchmark. Sources are `test/reference_cuda_bf16_moe_router.{py,c}`; evidence is
under `/tmp/bitnet-goal-flash-moe-router/`.

Resident expert execution and placement remain open. The pinned GGUF MoE plugin
uses vector execution through 64 total tokens and expert-grouped matrix
execution above 64 for these formats, independently of the router's 128-token
boundary. A per-expert batch must retain total-batch geometry and the original
activation quantization, projection, activation, weighted BF16 rounding and
reduction contracts. Calling an ordinary dense operator with its per-expert
batch size can select different arithmetic. Q5_1 decoded layouts remain useful
explicit storage, but do not establish parity with the plugin's A8 MoE path.

The complete FFN oracle is now captured for all three actual expert-format
combinations, using layers 2, 3 and 47 at nine batch sizes spanning the 64/65
expert and 128/129 router boundaries. It includes HC mix, actual loaded gates,
routed projections/activation/weighted reduction, shared experts and HC post.
Its 513 tensors contain 232,071,318 BF16 values, 21,870 F32 routing weights and
43,740 I32 indices/source rows. All stages reproduce exactly on a second run.
These are independent reference values; the native resident expert FFN has not
yet been implemented or compared against them.

Qualifying this oracle required fixing the comparison plugin: its grouped
kernel skipped experts above 255, indexed a one-element token-offset array by
warp number, and read beyond partial Q5_1 weight tiles at the actual 640-column
down shape. Three explicit patches under `bench/patches/` preserve the original
quantizer/dot/activation/reduction formulas and add correct bounds/indexing.
The bounded-loader version preserves every captured value byte-for-byte and
passes GPU memcheck at layer-3 batches 65 and 257; capacity and grouping probes
also pass memory checking. Original failing captures and sanitizer reports are
retained. See `docs/benchmarks.md` for reference-build requirements and unsupported
unpatched comparisons. Evidence is under `/tmp/bitnet-goal-flash-experts/`.

### Flash Next native gathered expert FFN qualification (2026-11-08)

Optional SM120 resident operations now execute raw Q4_K, Q5_K, Q5_1 and Q8_0
expert projections and complete gathered-expert FFNs. Quant metadata declares
this MoE profile separately from ordinary dense BF16 projection support. The
plan retains the original MoE batch size: vector execution through 64 total
tokens, matrix execution above 64, even when an expert receives only one row.
Q5_1 uses original-input-sum A8 metadata, FP16 scale/offset products and bounded
partial-tile loads. The FFN quantizes input once for gate/up, applies the two
BF16 activation rounds, then quantizes and projects down.

The native replay matches all 27 qualified actual-checkpoint cases exactly:
3,140 expert groups, 9,420 isolated projections, 83,980,800 projection values
and 69,984,000 complete-FFN activation/output values. Complete FFNs consume the
original gathered input and raw weights; captured intermediate activations are
used only for isolated down-projection checks. Invalid plans are rejected before
submission, including malformed weights, insufficient reserved scratch, aliasing,
misalignment and cross-request views. Memory checking covers the complete replay;
race/synchronization checks cover both execution modes and actual format pairs.
Warm traces contain six kernels with no submission allocations or transfers.
The protected clean build, standard suite, full CUDA suite and diagnostic
isolation pass without compiler warnings.

This is an expert primitive qualification. The test gathers rows on the host;
GPU grouping/gather, weighted scatter/reduction, shared-expert composition,
bounded asynchronous weight leases and whole-model integration remain open.
Native model admission and performance qualification are unchanged. See
`docs/benchmarks.md` and `/tmp/bitnet-goal-resident-expert/` for evidence.

### Flash Next GPU expert grouping and routed reduction (2026-11-08)

Optional SM120 resident operations now group top-10 routes across 512 experts,
gather BF16 inputs and retain forward/inverse row maps on the GPU. Group offsets
are exclusive; row order within an expert is unspecified. The inverse map restores
original token/rank order before routing-weight multiplication. Each product
rounds to BF16 before the ordered F32 sum and final BF16 output. Device status
guards reject invalid IDs or inverse bounds without unsafe indexing; malformed
host plans are rejected before submission. Grouping resets metadata on every call.

The independent replay matches all 27 actual cases and 36 synthetic cases across
three repeats, checking 512,792,532 BF16 values. Synthetic cases cover concentrated
and duplicate routing, empty experts, IDs above 255, odd dimensions and malformed
device metadata with valid retries. A separate composition uses GPU-gathered input,
the qualified raw expert FFNs and resident weighted reduction. All 27 cases match
61,585,920 weighted/output values exactly without host activation staging.

The composition harness reads 2,056 bytes of offsets/status per case to stage one
expert's raw weights at a time, then explicitly completes each expert before
releasing weights. It is a correctness harness, not the asynchronous grouped bank
scheduler. Shared experts, HC composition, event-protected bounded cache placement
and whole-model gates remain open. Gathering currently replicates input rows per
route; quantizing original input once and grouped bank execution remain performance
work. See `/tmp/bitnet-goal-resident-group/` for commands, captures and qualification.
The clean build, standard suite and full CUDA suite with diagnostic isolation
pass without warnings; GPU memory, race and synchronization checks also pass.

### Bounded asynchronous expert weight banks (2026-11-08)

Optional SM120 request resources now reserve fixed pinned-RAM and GPU raw-weight
banks with independent payload budgets. Acquisition is nonblocking and uses the
previous bank's compute-completion event, protecting both DMA source bytes and
device weights through their last consumer. Generation checks reject stale leases.
Submission validates every expert, range, view and scratch requirement before
queuing a single bulk H2D copy on a separate stream, a compute-stream dependency,
FFNs and the bank-completion event. Request teardown drains both streams. No
submission allocation or per-expert host fence is required.

The streaming replay matches all 27 actual cases at capacities one and eight:
123,171,840 weighted/output BF16 comparisons are exact across the two runs. Each
configuration processes 3,140 expert groups and transfers 11,035,750,400 raw bytes;
capacity eight uses 408 bank submissions, versus 3,140 at capacity one. The
two-slot capacity-eight pool reserves 63,897,600 payload bytes each in pinned RAM
and GPU weight storage. Activation/workspace buffers and allocator/event overhead
are separate. Both full replays and the lifecycle test pass GPU memory checking
with zero errors or leaks. Lifecycle race/sync checks and actual matrix-path sync
checking pass. A streaming trace confirms DMA/compute overlap and no submission
allocations or internal host fence.

This qualifies asynchronous staging and weight lifetime, not the final grouped
matrix executor or model cache. Compute still launches six kernels per expert and
quantizes gathered input per expert. Original-input quantization reuse, grouped
matrix launches, model-cache prefill/decode policies, cold/warm stall accounting,
shared experts, HC composition and whole-model gates remain open. Evidence is
under `/tmp/bitnet-goal-expert-bank/`.
The protected clean build, standard suite and full CUDA suite with the new bank
lifecycle test and diagnostic isolation pass without compiler warnings. The final
audit preserves all 513 independent reference files and records current hashes.

### Original-input quantization reuse across expert banks (2026-11-08)

The optional prepared-bank path now quantizes original MoE token rows once and
maps complete A8 blocks into expert order. Gate/up projections reuse these blocks
across every bank; down projections retain their activation quantization. The
original-input and grouped A8 buffers have an explicit separate workspace budget.
Generation tickets reject stale preparation state or a different batch size before
any weight upload. Repreparation follows earlier consumers on the compute stream.

Both one- and eight-expert bank configurations match all 27 actual cases exactly,
checking 123,171,840 weighted/output BF16 values combined. The legacy bank path
also passes all 27 cases. Lifecycle checks cover workspace budgets, malformed
maps, request ownership, generation changes and batch-geometry rejection. Full
prepared replays pass GPU memory checking with zero errors or leaks; lifecycle
race/sync checks and actual matrix-path synchronization checks also pass.

The layer-3/batch-257 trace contains one original-input quantization and 206 down
quantizations. Its 206 expert FFNs plus preparation require 1,033 kernels, compared
with 1,236 before reuse. Raw upload bytes and bank payload budgets are unchanged;
the maximum tested 257-token preparation adds 8,141,760 workspace bytes. This is
operator qualification, not whole-model prefill parity. Grouped matrix launches,
removal of redundant BF16 gathering, cache-policy integration, shared experts and
HC/model qualification remain open. Evidence: `/tmp/bitnet-goal-moe-input-reuse/`.
The protected clean build, standard suite and complete CUDA backend chain with
diagnostic isolation pass without warnings. All 513 independent reference files
remain unchanged in the final qualification audit.

### Grouped expert-bank projection launches (2026-11-08)

The optional grouped-bank path now dispatches gate/up projection, activation,
down-input quantization and down projection once per bank. CUDA grids select the
expert, projection role and token/row tile. The existing qualified format bodies
are shared device functions, retaining the original per-row arithmetic and total
batch MMVQ/MMQ selection. Pinned/device descriptors and grouped down A8 scratch
have separate explicit budgets; descriptor lifetime follows the bank's compute
completion event. Whole-bank host validation still precedes every upload.

Both grouped capacities match all 27 actual cases exactly, with 123,171,840 BF16
weighted/output comparisons combined. Prepared per-expert and legacy paths also
pass all 27 cases after extraction of the shared device bodies. Twelve synthetic
matrix cases cover hidden widths 32/96/160, all four supported format pairs,
unequal row counts, incomplete four-token tiles, gaps and untouched output rows.
Both grouped replays pass GPU memcheck; lifecycle/tail and actual matrix race/sync
checks pass without errors, leaks or hazards.

For the same 206-expert example, compute launches fall from 1,033 prepared
per-expert kernels to 107 grouped kernels (four per bank plus three preparation
kernels). Descriptor uploads add 2,472 bytes to the unchanged 632,832,000 raw
weight bytes. The trace observes zero copy/compute overlap; device execution and
copy time leave substantial gaps in this streaming harness. These are structural
results, not whole-model throughput parity. Shared FFNs, HC/model integration,
cache policy and matched end-to-end measurements remain open. Evidence is under
`/tmp/bitnet-goal-grouped-bank/`.
The protected clean build, standard suite and complete CUDA backend chain with
diagnostic isolation pass without warnings. The final audit preserves all 513
reference files and verifies the extraction records for all ten device cores.


### Complete native Flash Next HC/FFN composition (2026-11-08)

The actual-weight composition probe now executes HC normalization/projections and
mixing, router/shared-gate projections, native top-k/grouping, bounded grouped
expert banks, ordered routed reduction, the shared Q8 FFN, gated addition, and HC
injection/combine. Only the original wide hidden state seeds neural computation;
expected intermediate activations and routing decisions are comparison outputs.
All 27 preserved cases (layers 3/2/47, token counts 1/5/17/63/64/65/128/129/257)
match 215,275,158 BF16 values and 65,610 routing values/indices exactly.

The original shared `SiluAndMul` uses one final BF16 rounding, whereas routed
experts use separate BF16 SiLU/product rounding. The previous shared two-round
assumption was incorrect. An explicit dense FFN activation profile preserves the
existing separate default and selects fused rounding for this shared branch.
Independent tests cover 524,288 inputs per profile. Accurate division also fixes
three BF16 sigmoid subnormal cases and 21 SiLU/product edge cases; HC scaled SiLU
and extreme-gate tests cover these boundaries. Reference captures remain intact.

The layer-3, 65-token trace has 165 expert groups in 21 banks, 506,880,000 raw
weight upload bytes, 1,980 descriptor bytes, one 2,080-byte metadata readback,
and two stream completion fences (metadata plus explicit outer completion).
Compute stays on one stream; copies use the separate transfer stream. No
allocation, memset, device-wide synchronization or event completion wait occurs
inside the captured execution. Operator setup is warmed, but weights still stream
on each replay: this is not a warm-cache or vLLM performance comparison.

The complete probe and model-free tests pass memory/synchronization checks;
model-free tests also pass race checks. Evidence and commands are preserved in
`/tmp/bitnet-goal-shared-hc/`. Production model/session integration, global cache
leases and placement budgets, short/long/chunked/resumed model gates, and matched
end-to-end vLLM measurements remain open. Audit modern dense-model activation
profile selection independently before changing its default or prior goldens.


### Production-owned Flash Next FFN layouts (2026-11-08)

The backend precision container now owns immutable BF16 router/shared-gate
projections and raw Q8 shared FFN weights for all 48 layers. Routed expert bytes
remain in CPU RAM/file storage; preparation copies only expert-map anatomy and
borrows any existing repacked CPU source. It does not upload all routed experts.
The shared FFN explicitly selects the independently qualified fused SiLU profile.

Preparation requires the same HC/PLE/GDN/QSA source container, validates every
layer before GPU uploads, rejects unsupported routing scales/weight formats and
expert geometry/offset overflow, and checks an aggregate GPU tensor budget.
Each raw F32 router upload is released after BF16 conversion. On any failure,
new allocations are released and the existing attention/HC components survive.
Layouts cannot be replaced while live and are released by backend model teardown.
Mock lifecycle tests inject all 15 upload and six decode failures across three
layers, check preflight failures without uploads, and verify exact budget edges.

Actual checkpoint preparation retains 376,750,080 FFN tensor bytes, with a
376,765,440-byte preparation peak. Including HC/PLE/GDN/QSA, declared payload is
7,132,468,352 bytes and peak is 7,132,483,712 bytes. These exclude allocator/context
overhead, CPU expert source, request activations and staging/cache budgets.
The full FFN composition probe's `--model-layout` mode borrows these production
router/shared handles. All 27 original cases still match 215,275,158 BF16 values
and 65,610 routing values/indices exactly; memcheck reports zero errors/leaks.
The clean build has zero warnings and the standard test suite passes.

Evidence is under `/tmp/bitnet-goal-flash-ffn-model/`. This qualifies model-owned
FFN preparation, not a production session FFN submit hook: the probe still drives
operator composition and expert-bank staging. Request orchestration, global
cache leases/budgets and cold/warm placement policies remain open, along with
whole-model gates and matched vLLM performance/serving acceptance.


### Production Flash Next FFN request and RAM-cache policy (2026-11-08)

`bn_transformer_bf16_flash_ffn_submit` now executes the full native FFN inside the
production HC transaction, using immutable backend-model weights and request-owned
scratch. Preparation reserves two cap-8 expert banks, input/down A8 workspace,
routing and activation buffers under explicit host/GPU budgets. No resident
allocation or host activation upload occurs during submit. One packed 2,080-byte
routing metadata read selects expert groups. BUSY returns preserve the cursor and
prepared-input generation, allowing caller retry without rerouting or internal
spinning. HC's raw/normalized wide states survive until injection/combine.
FFN positions commit only after successful outer completion; failures poison the
request while retaining old committed positions. Model admission remains unchanged.

The existing MoE source-lease API is exposed through `moe.h`, with const I/O
control access. Prefill borrows RAM-cache hits without promotion/admission;
decode admits misses. Source leases protect synchronous copies into pinned banks
and are released immediately after those copies. Bank completion events protect
the copied payload through GPU use. Pread misses can read directly into these
pinned slots, without another expert-sized host staging allocation. Shared cache
access across requests remains executor-serialized. This adds no GPU expert cache.

Both mmap and split-file pread pass all 27 preserved numerical cases: 215,275,158
BF16 values and 65,610 routing values/indices exactly per mode. Each mode streams
11,035,750,400 raw bytes in 408 banks across these cases. Forced BUSY tests verify
one routing read/input preparation per block, no premature HC combine/commit, and
healthy retry. Allocated resident payload plus backend workspace/bank accounting
matches declared GPU bytes for every case. Exact budget edges, partial-preparation
rollback, queued-work failure, outer-fence failure and interleaved request isolation
are tested. One persistent request also runs 5-token and 65-token prompt chunks
then one decode token, reaching position 71 with three input generations; these
are independent FFN fixture inputs, not a full-model continuation qualification.

With a fresh 64 MiB shared RAM-cache budget, a one-token decode has 10 misses and
30,720,000 file-read bytes; the next request has 10 hits and zero file-read bytes.
Subsequent prefill borrows all 10 hits. The initial 27 prefill cases leave the
empty cache unpopulated. Cold/warm here means RAM-cache state, not cold OS pages
or storage. GPU upload is still 30,720,000 bytes for each one-token replay.
Per-request raw bank payload is 63,897,600 bytes per host/device tier, with 192
bytes of descriptors per tier. Input A8 is 31,680 bytes/token, down A8 is 7,200,
and shared-linear A8 is 2,880; additional resident FFN tensors are 198,348
bytes/token plus 4,128 bytes. Host metadata read storage adds 2,080 bytes. These
exclude model weights, RAM-cache/source pages, allocator/context and C metadata.

Request diagnostics report source/file/cache bytes, bank uploads, metadata read
wall time, source fetch/copy wall time and BUSY-to-retry spans. The latter include
caller scheduling delay and are not isolated DMA stall measurements. Mmap/pread
request tests pass memcheck with zero errors/leaks; synchronization checks pass.
Clean build and standard tests pass without compiler warnings. Evidence is under
`/tmp/bitnet-goal-flash-ffn-request/`.

Whole-model short/long/chunked/resumed gates, production token embedding/output
and complete layer orchestration, global VRAM placement/leases, aggregate serving
admission, matched resident vLLM performance and continuous batching remain open.


### Flash Next vocabulary endpoints and explicit model driver (2026-11-08)

The checkpoint has separate Q8 token/output tables and an output HC mixer, with
no additional final norm. The pinned original model repeats input embeddings over
four HC streams. Native Q8 selected-row embedding now declares its quant capability
in the registry and preserves the original FP16 scale/product rounding before
BF16 conversion. Its independent oracle covers every FP16 scale bit pattern paired
with every signed 8-bit quant, plus token-count tails: 16,816,384 output words.
NaN payloads are canonicalized only for synthetic hash comparisons. Finite values,
infinities and signed zeros remain exact. Invalid IDs poison only their request;
peer requests and released/reinitialized containers remain healthy. The existing
Q4 embedding reference remains unchanged.

The immutable model container now owns both raw vocabulary tables under an
aggregate preparation budget, with rollback at either upload failure and no
decoded vocabulary expansion. Their payload is 1,350,860,800 bytes; total prepared
HC/PLE/GDN/QSA/FFN/vocabulary payload is 8,483,329,152 bytes. Token IDs, BF16 logits
and the embedding status word add 4 bytes/token + 496,640 + 4 GPU bytes to requests;
retained model-driver token copies add 4 host bytes/token. Existing scratch holds
embedding repetition and output HC projection stages. Declared GPU bytes match
tracked resident allocations plus opaque backend workspace for all endpoint cases.

Production token-submit and head-submit APIs now perform device-resident lookup,
four-stream repetition and output HC mixing followed by a last-row Q8 vocabulary
projection. Five actual-checkpoint cases (1/5/17/129/257 tokens) match 11,712,000
BF16 words and 409 token IDs exactly against fresh pinned references. Head inputs
come from preserved independent FFN captures; native tests execute the production
FFN before the head, preserving the original references. No extra output norm or
CPU activation bridge is introduced. The independent oracle was replayed under
memcheck into a fresh directory: all 35 data files, header and manifest match.

An explicit model begin/submit API connects token input, PLE at its configured
layer before attention, all 48 HC attention/FFN pairs and the output head. It
copies token IDs into prepared request storage and retains its layer/phase cursor
across expert-bank BUSY returns. The successful submit leaves head work queued;
outer completion commits all recurrent, PLE and FFN positions. Ordinary model
admission is unchanged pending independent whole-model qualification.

A smoke test executes one-token prefill and subsequent decode through the full
driver, with forced bank retries, stable copied token IDs, 48 then 96 accumulated
FFN blocks, finite logits across all 248,320 vocabulary entries and committed
positions 1 then 2. Argmax IDs are 11 and 11 for the initial token 116. These IDs
are recorded observations, not validated expected model tokens. Component accuracy
and finite full-model execution do not prove full-model token correctness.

Clean build/standard tests pass without compiler warnings. Native endpoint,
embedding and full-driver smoke tests pass memcheck with zero errors/leaks;
full-driver synchronization checks pass. Evidence is preserved under
`/tmp/bitnet-goal-flash-embedding/`. Independent whole-model short/long/chunked/
resumed token gates and common session/CLI admission remain open. Global VRAM
expert placement, aggregate serving budgets, matched vLLM performance and batching
acceptance also remain open.

The complete CUDA regression suite also passes on this checkpoint, including the
new Q8 embedding cases and existing Q4 embedding, MoE and diagnostic isolation tests.

Source audit found that the host vLLM checkout differs from the pinned image.
The relevant pinned model ordering was inspected directly, and the common HC
normalization/mix/combine math methods match the inspected host methods. Future
whole-model references must use the pinned image sources preserved in this evidence directory.


### Independent Flash whole-model references (2026-11-08)

The explicit Flash driver now matches independent original-operator references
through all 48 layers, rather than only producing finite smoke-test logits.
Reference inputs begin at checkpoint token embeddings and continue through PLE,
HC attention/FFN blocks and the output head. The pinned vLLM image supplies the
original neural operators; the already documented corrected GGUF extension
supplies raw expert/vocabulary operations. Core matrices use the qualified
decoded-BF16 profile. This is an operator correctness comparison, not a served
vLLM GGUF adapter or a performance benchmark.

Qualified schedules are one-token prefill plus decode, the five-token plain-text
prompt `The capital of France is` plus four greedy decode steps, that prompt split
2+2+1 plus four decode steps, 129-token prefill plus decode, and 1,025-token prefill
plus four decode steps. Comparisons are exact at every attention/FFN wide-hidden
boundary. Additional checks cover selected GDN intermediates/F32 states, QSA
projections/attention/indices, expert IDs/weights, PLE history, output HC mixing
and all vocabulary logits. Each schedule also runs independently through the
production begin/submit/complete driver with fresh request state. The 1,025-token
case compares 1,125,382,624 BF16 words and 6,064,248 F32/I32 words across the two
execution paths, with zero differences. Each chunk schedule retains its own
reference; cross-schedule floating-point identity is not assumed.

A real second-chunk cancellation exposed two original GDN output contractions.
Enumerating all 24 supported original output-kernel configurations reproduced both
behaviors using identical prepared inputs. Two-warp/BV32 configurations round the
scaled local term before a history FMA; the other inspected configurations round
scaled history before a local FMA. The resident plan now names that arithmetic
contract. Existing zero-initialized callers retain local FMA; Flash explicitly
selects history FMA. Future whole-model captures fix the supported original
BK64/BV32/two-warp/three-stage configuration so autotune timing cannot change the
reference contraction. No input-value or token-count heuristic selects the profile.

The new model-free cancellation fixture checks both contracts, F32 next-state
bits and immutable initial state. All 35 existing resumed GDN prefill calls still
pass, and the previous 68 independent GDN output/state files and all earlier whole
model reference files retain their recorded hashes. The fixture generator and
full-model generator are `test/reference_cuda_bf16_gdn_cancellation.py` and
`test/reference_cuda_bf16_flash_whole.py`; the comparison harness is
`test/reference_cuda_bf16_flash_ffn.c --model-reference`.

Evidence is preserved under `/tmp/bitnet-goal-flash-whole/`. Common session/CLI
admission, larger context coverage, global expert VRAM placement and serving
budgets, continuous batching, and matched vLLM performance acceptance remain open.


The final production sources build cleanly with zero compiler warnings; standard
and complete CUDA regression suites pass. Both output profiles pass the new
cancellation regression. Native cancellation and complete short chunked model
runs pass memory and synchronization checking with zero errors (memory checking
also reports zero leaked allocations). The independent original-operator oracle
passes memory checking and reproduces every preserved short chunked reference
file exactly, including all recorded source-weight hashes.


The long resumed whole-model gate also passes: 1,025 prompt tokens split into
seven 129-token chunks and a final 122-token chunk, followed by four decode steps.
Both execution paths agree exactly with their schedule's independent reference:
1,129,719,264 BF16 words and 11,569,272 F32/I32 words, zero differences. This closes
the 1K-context chunked/resumed accuracy gate for the explicit Flash driver;
larger-context and ordinary runtime admission gates remain open.


These whole-model comparisons use mmap-backed expert sources. Bounded pread
source-cache behavior retains its earlier component qualification; a full-model
pread comparison remains open. The tested contexts also do not reach QSA's
2,048-token selection/pruning boundary (512 selected groups, compression ratio 4).
That boundary and longer contexts need independent whole-model gates before
claiming general long-context admission.


The 1,025-token uninterrupted and eight-chunk native whole-model runs also pass
memory checking with zero errors and zero leaked allocations. The eight-chunk
run passes synchronization checking with zero errors. Final source, binary, log,
reference and preserved user-diagnostic hashes are recorded in
`/tmp/bitnet-goal-flash-whole/qualification.json`.


### Flash whole-model bounded pread cache qualification (2026-11-08)

The independent 48-layer Flash comparison harness now also supports
`FIRST_SHARD CAPTURES --model-reference-pread REF DUMP [CACHE_MB]`.
It selects shard-file pread explicitly, disables expert mmap access, derives each
cache projection stride from the maximum actual checkpoint projection size, and
verifies every layer fits. Source-weight files remain mapped for model/core
loading; this is a bounded expert-cache qualification, not a total process RSS
limit or a cold-storage benchmark.

All preserved schedules pass exactly through both layer-by-layer execution and a
fresh production driver: one-token and five-token prompts, 2+2+1 chunked prefill,
129-token prefill, 1,025-token prefill, and 1,025 tokens split 129*7+122. Each
continues through its original decode schedule. No expected values or numerical
profiles change. The existing mmap whole-model comparison and all 27 legacy FFN
request cases also pass.

The harness snapshots cache membership without promoting entries or retaining
leases. Prefill must preserve membership; decode must admit misses and account
for evictions. Every phase checks expert hit/miss counts, pread projection counts,
and `pread bytes == RAM miss bytes`, with RAM hit plus miss bytes equal to raw
GPU upload bytes. Request GPU/host reservations remain constant across phases;
raw bank reservations are 63,897,600 bytes in each host/device tier. This checks
RAM payload ownership and existing event-protected streaming banks; it does not
introduce a model-wide expert VRAM cache or aggregate serving-budget accounting.

| Expert payload budget | Allocated payload | Slots | Observed behavior |
| --- | ---: | ---: | --- |
| 64 MiB | 63,897,600 bytes | 16 | First decode admits 480 experts, retains 16 and evicts 464; later decode continues evicting. Prefill borrows without admission. |
| 2 GiB | 2,144,563,200 bytes | 537 | Retains the full 480-expert decode working set across fresh requests. |

The 2 GiB replay's first decode has 480 RAM misses and reads 1,504,256,000 bytes.
Repeating the same decode in a fresh request has 480 RAM hits, zero misses and
zero pread bytes, with identical logits. Its intervening prefill borrows 172
cached experts without changing the 480 cached entries. Both cold and warm
decodes still upload 1,504,256,000 bytes to the GPU: RAM hits are not VRAM hits.
This transfer remains a concrete target for the planned bounded VRAM placement.
Here cold/warm describes the expert RAM cache; filesystem-cache state is not
controlled.

Per-phase logs include raw uploads, source reads, cache hits/misses/evictions,
metadata read bytes, bank counts/retries, request reservations and source/read
wall times. The bank retry spans include caller scheduling delays, and metadata
read timing can include preceding stream work. Observer reads and sanitizer
instrumentation also affect these tests. These are correctness/accounting traces,
not isolated transfer-stall measurements or vLLM performance results.

The clean build and standard tests pass without compiler warnings. Native
1,025-token chunked pread and fully warm replay pass memory checking with zero
errors and zero leaked allocations; chunked pread passes synchronization checking.
Evidence and immutable-reference hashes are under
`/tmp/bitnet-goal-flash-pread-whole/`. Larger-context pruning, common session/CLI
admission, bounded expert VRAM residency, aggregate serving budgets, batching,
and matched vLLM prefill/decode performance remain open.


The complete CUDA regression suite and diagnostic isolation checks also pass on
the final rebuild. Final qualification, per-phase cache/transfer metrics,
source/binary/log hashes, unchanged independent-reference hashes and preserved
user diagnostic binaries are recorded in
`/tmp/bitnet-goal-flash-pread-whole/qualification.json` and `metrics.json`.
This closes the explicit Flash driver's full-model pread accuracy/accounting gate
through 1,025 tokens; ordinary CLI admission and performance gates remain open.


### Flash 4K pruning under declared execution profiles (2026-11-08)

The explicit 48-layer Flash driver now passes independent whole-model gates at
and beyond QSA pruning, in both mmap and bounded pread modes. The reference
contract names the existing native numerical/selection policies: four-warp
triangular inversion, the previously pinned history-FMA output configuration,
and exact score rank with lower-index ties and chronological QSA group emission.
This does not establish literal bit equality with an unmodified, arrival-ordered
vLLM execution, and does not yet enable ordinary Flash CLI admission.

The first uncontrolled 4,097-token capture exposed a GDN inversion configuration
difference. Native and a fresh original-operator replay agree exactly at every
captured recurrence stage, including F32 state, on identical prepared inputs.
Forcing the original two-warp inversion reproduces the uncontrolled layer-0
output/state exactly; four warps reproduces native exactly. The two profiles
differ at 270,007 of 25,171,968 BF16 output values and 59,665 F32 state values.
The failed literal native comparison and original capture remain preserved;
no production kernel or expected file was changed to hide this difference.

QSA also has an existing selection-policy distinction. In the controlled 4K
prefill's first sparse layer, all 2,046 pruned rows choose the same groups as the
original operation, but their emission order differs. This changes 241,678 BF16
attention values and 1,373,640 output-projection values on the same neural input.
The reference retains the original choices, F32 scores and original attention/
projection outputs separately. It verifies validity and score-optimality before
independently applying the declared chronological policy and calling original
attention. Canonical ranking uses NumPy; native uses radix threshold/compaction.
A separate verifier checks cutoff inequalities and tie-ID ordering without
reusing the generator's sorting algorithm.

`test/reference_cuda_bf16_flash_whole.py` adds explicit
`--gdn-inverse-warps {2,4,8}` and `--qsa-selection {original,canonical}` options.
The default inversion remains autotuned and selection remains original. New
controlled captures use `--gdn-inverse-warps 4 --qsa-selection canonical`, with
these settings recorded in their manifests. Diagnostic scores may contain
masked nonfinite cells outside the visible prefix; visible scores and every
neural output still undergo finite-value checks. All earlier references remain
immutable. A fresh controlled five-token/four-decode capture preserves every
previously recorded short neural value and source-weight hash exactly.

| Whole-model schedule | Compared BF16 words | Compared F32/I32 words |
| --- | ---: | ---: |
| 2,047-token prefill, eight decode steps across first pruning | 2,246,873,760 | 11,335,848 |
| 4,097-token prefill, four decode steps | 4,475,877,856 | 12,429,432 |
| 4,097-token prompt split 257*15+242, four decode steps | 4,485,170,656 | 24,225,912 |

Each row includes all attention/FFN layer boundaries and selected recurrent,
indexing, routing, PLE and head intermediates, followed by a fresh complete
production-driver replay. Every compared word matches exactly in each I/O mode.
Original ranking validity/score-optimality is independently checked on 49,248
pruned query rows across these three reference trajectories. All pruned rows
change order; two change membership only among exactly tied cutoff scores in
the chunked trajectory. These original alternatives are retained, not used as
interchangeable bit-exact expected model values.

The clean build and standard suite pass without compiler warnings. Evidence,
including uncontrolled and controlled captures, original inversion audits,
ranking checks and preserved failure artifacts, is under
`/tmp/bitnet-goal-flash-pruning/`. No production source changes in this step.
Common session/CLI integration, bounded expert VRAM residency, matched vLLM
performance, continuous batching, later models and backend qualification remain
open. Literal unmodified-vLLM 4K bit parity remains explicitly unqualified.


Final native memory checking passes for both uninterrupted 4,097-token and
sixteen-chunk pread trajectories through layer-by-layer and fresh production
driver execution: zero errors and zero leaked allocations. The chunked trajectory
also passes synchronization checking with zero errors. The complete CUDA suite
and diagnostic isolation pass on the final warning-free rebuild. All production
source hashes and four user diagnostic binaries are preserved. Final checks,
reference/source/binary/log hashes, ranking verification, original inversion
audits and the explicit literal-parity limitation are recorded in
`/tmp/bitnet-goal-flash-pruning/qualification.json`.

### Flash native BF16 session and CLI integration (2026-11-08)

The qualified Flash checkpoint now runs through ordinary backend-only
`BnSession`, prefill/decode, generation and `--cuda-bf16` CLI dispatch. The
architecture owns the Flash HC preparation contract. Full preparation builds
HC, PLE, GDN, QSA, shared/routed FFN metadata and vocabulary endpoints in a
private backend owner, then publishes the complete layout and GPU binding.
Allocation, budget or publication failure releases the staged buffers and leaves
the destination unchanged. Component-only layouts remain outside ordinary
execution. GPU layouts and handles remain in backend state.

Common position/capacity queries, healthy restart, no-logits prefill and host
logits publication now support complete Flash layouts. No-logits execution skips
the output HC mixer and vocabulary projection, preserves host logits and commits
the same recurrent transaction. BF16 logits readback staging is included in the
request host budget. Execution/readback failure poisons the request without
publishing logits or permitting CPU fallback; reset/reinitialization restores
the independent first-prefill values. Two interleaved short requests preserve
separate positions and produce identical reference logits. Chat reset, context
rollover and repeated benchmark reset also pass through the real CLI.

Frozen independent references pass exact comparisons through manual operators,
the complete driver, ordinary sessions, and restarted no-logits prefill followed
by decode. Coverage includes five prompt tokens, uninterrupted 4,097-token
prefill, sixteen chunks of 257/242 tokens, and the actual default schedule of
1,025/1,025/1,025/1,022 tokens, each followed by four decode steps. The four paths
pass 20, 20, 80 and 32 reference phases respectively. Ordinary host float logits
match exact BF16 expansion. Mmap and bounded 64 MiB pread CLI runs match their
schedule's expected tokens, including the default batch size with 8,192 context
capacity. All prior reference files retain their recorded hashes; the new
default-schedule capture uses the same source-weight hashes.

These references retain the explicit native contract: four-warp GDN inversion,
history-FMA output and score-optimal canonical QSA selection. Reference token
trajectories differ across chunk schedules; this qualification establishes
matched-schedule agreement, not cross-schedule invariance or literal unmodified
vLLM bit parity. The newly captured default schedule independently validates
24,600 pruned QSA rows; one changes membership only among tied cutoff scores.
Original choices remain in the capture diagnostics.

The model tensor payload remains 8,483,329,152 bytes. CLI preparation budgets
are 12 GiB GPU/64 MiB temporary host payload; requests have explicit 4 GiB
GPU/256 MiB host limits, separate from source expert storage and allocator
overhead. Expert banks retain their bounded two-slot ownership. The default
schedule's reference request uses 1,155,148,040 GPU bytes and 67,615,012 host
bytes at capacity 4,101. Real CLI qualification separately uses capacity 8,192.
Warm RAM hits still require expert upload; persistent expert VRAM residency
remains open.

The clean CUDA/MKL and CPU-only builds and standard tests pass without compiler
warnings. Dense Qwen runtime/CLI regressions pass. Runtime memory checking reports
zero errors and zero leaked allocations; default-batch CLI synchronization
checking reports zero errors. The reusable real-model CLI test is
`test/test_cli_flash_bf16.py`; it requires a frozen manifest SHA and matching
chunk schedule. Commands, reference/source/binary hashes, complete CUDA checks
and explicit qualification limits are recorded under
`/tmp/bitnet-goal-flash-runtime/`.

Next work is bounded expert VRAM placement and transfer/synchronization
measurement, followed by matched fully resident vLLM prefill/decode benchmarks.
The 5% performance target, continuous batching/KV scheduling/tail latency,
DeepSeek/GLM integration and additional backend hardware gates remain open.

### Flash offloading measurements and cache layout constraints (2026-11-08)

Ordinary CLI traces now establish the offloading baseline: each decode step
streams 1,504,256,000 expert bytes, reads 48 routing metadata blocks and crosses
approximately 50–51 stream fences. Long default-batch prefill uses grouped
matrix/MMQ kernels and streams 61,402,419,200 expert bytes over the 4,097-token
prompt. Three unprofiled prompt measurements are 4,596.5–4,600.8 ms. These are
local offloading measurements, not matched vLLM performance comparisons.

Independent coverage records zero CPU/legacy orchestration entries across 13
native public-API dispatches; CUDA memory tracing records zero device-heap or
pinned-host allocation events during inference. The independent oracle now
retains every layer's routing IDs. The native comparator passes all 32 phases,
including those new IDs, and preserves earlier neural values. Placement replay
matches measured zero-cache transfers and active expert counts exactly.

The measured layout rules out allocating a largest-expert slot for every key:
that requires 91.41 GiB before core/request state, versus 71.73 GiB for the raw
experts themselves. Compact size-class pools are needed to preserve a fully
resident Flash comparison on the 96 GB target. Partial-cache replay also shows
why small global LRU caches can thrash and why prefill tail admission is not a
uniform improvement. The simulator does not establish actual speedup or a
production default from its four-step decode trajectory.

Implement compact model-owned residency with event-protected request leases
next. Preserve original expert selection and quant arithmetic, retain the
borrow-only control, and distinguish GPU misses from subsequent RAM cache
lookups and file reads. Qualify numerical equivalence, eviction/failure/isolation,
longer decode trajectories, bounded payload/peak memory and real cold/warm
performance before enabling defaults. Existing legacy handle-cache eviction
does not satisfy the resident path's asynchronous lease or compact-layout needs.
Detailed data and reproducible analysis tools are in `docs/benchmarks.md` and
`/tmp/bitnet-goal-flash-profile/`.


### Flash compact VRAM expert cache qualified as opt-in (2026-11-08)

The native Flash path now implements model-owned raw expert pools with compact
size classes, grouped cached-weight execution and event-protected request leases.
Use explicit `--gpu-cache-mb N` with `--cuda-bf16`; the native default remains
zero. A 2 GiB budget reserves 2,147,328,000 payload bytes for 685 experts, using
all three checkpoint format classes. Prefill borrows existing entries without
admission or promotion; decode admits/promotes. GPU hits skip RAM-cache lookups
and source copies. Decode admission pressure bypasses to the streaming banks.
Core weights, request buffers, pinned banks and the expert pool have separate
budgets. Cache preparation failure rolls back before publishing model state.

All 32 independent reference phases preserve neural values and every layer's
routing IDs. Their hit/miss bytes, residency, evictions and pressure bypasses
match the frozen placement replay exactly. Tests cover nonzero distinct outputs,
all three raw format pairs, vector and matrix execution, poisoned hit sources,
mixed hits/misses, eviction, pinned pressure, stale leases, request teardown,
failed-session isolation, preparation rollback and zero-capacity budgets. CPU
and CUDA suites, dense-model regression, short/long Flash CLI/reset/chat paths,
CPU-only build, CUDA memory checking and synchronization checking pass.

On the captured four-step decode trajectory, warm upload falls from
6,017,024,000 to 3,954,483,200 bytes. Matched local CLI samples show 11–18% higher
reported output rate and 4–5% lower prompt time with the 2 GiB GPU cache,
depending on source RAM budget. These short trajectories do not establish a
production default or vLLM parity. Miss admission currently uploads to a bank
then copies into its pool; actual device copy bytes are reported separately.

Allocating all 77,017,907,200 raw expert bytes succeeds on the target GPU, but
full preloading is not implemented. Next qualify explicit full preloading and
longer offloaded trajectories, profile the remaining transfers/fences, then run
matched fully resident vLLM benchmarks. Continuous batching, KV scheduling,
tail latency, other modern model families and remaining backend hardware gates
stay open. Detailed commands, checks and hashes are recorded under
`/tmp/bitnet-goal-flash-vram/`; measurements are in `docs/benchmarks.md`.


### Flash full expert preloading qualified (2026-11-08)

`--cuda-bf16 --gpu-cache-mb 73450 --gpu-preload-experts` now uploads every
Flash routed expert before requests bind. The backend requires an empty pool
with capacity for all keys and zero bound requests. One 3,993,600-byte pinned
staging payload serves all 24,576 experts; entries become visible after every
upload completes. Source-read failure leaves the pool empty and retryable.
CPU source I/O stays in transformer orchestration and borrows RAM entries
without admission/promotion. Defaults and offloaded execution are unchanged.

The real checkpoint preloads 77,017,907,200 bytes across 73,728 source
projections. All 32 long and 20 short frozen reference phases pass with full
residency, zero misses, expert file reads, uploads, fill copies or evictions.
Routing IDs and neural values remain exact under the existing qualified
numerical profile. A 32-token continuation also matches uncached native
execution, extending cache-equivalence coverage beyond the independent short
oracle. Preparation rollback, failed-session isolation, CLI mmap/pread/reset/
chat/rollover, entire CUDA/standard suites, CPU-only build and dense Qwen27
regression pass. Memory checking passes for both synthetic preload failures
and the real fully resident model; long CLI synchronization checking passes.

The short resident CUDA memory trace confirms zero expert transfers and
inference allocations. Tracked device-heap peak is 86,808,536,040 bytes, with
complete heap/pinned teardown. Local long-prompt median is 3,368.0 ms resident
versus 4,656.0 ms streaming; reported output rate is 88.52 versus 8.14 tokens/s.
Preloading is excluded from these execution timings and separately logged.
These remain local single-request CLI metrics, with the first prefill-sampled
token included in output rate. Matched served-vLLM latency/throughput,
prefill acceleration, continuous batching/KV scheduling/tails, additional
model families and backend hardware gates remain open. Commands, traces,
hashes and qualification evidence are under `/tmp/bitnet-goal-flash-preload/`.


Resident long-prompt profiling now identifies grouped Q4_K gate/up and Q5_1
down as 72.07% of summed prefill kernel time. GPU activity covers 98.37% of
prefill windows, with no expert uploads. Prioritize those matrix kernels and
retain the independent quant arithmetic gates while establishing the matched
vLLM engine comparison. The local reference image and GGUF plugin loader/
adapter sources have been inventoried for that work; an operator oracle alone
does not satisfy the engine benchmark requirement.


### Residency scope for the parity benchmark (2026-11-08)

The Flash measurements above qualify complete **expert** residency. The
28,800,138,240-byte PLE embedding table remains host-owned; selected rows are
gathered into bounded upload leases in `src/transformer/bf16.c`. Zero expert
uploads therefore does not mean zero model-weight transfers. The checkpoint
contains 111,323,630,080 raw tensor bytes (5,505,584,640 core,
28,800,138,240 PLE, 77,017,907,200 experts), exceeding this GPU's capacity even
before decoded layouts and request state. The current Flash case does not
satisfy full-model GPU residency for the parity criterion. A matched Flash
comparison must account for the same PLE host placement explicitly.

The Qwen3.8-27B GGUF is `qwen35`, has 17,548,181,504 raw tensor bytes and no PLE
tensors. The pinned GGUF plugin includes a Qwen3.5 adapter, making this the next
fully resident engine-comparison candidate. Engine/config/precision and memory
qualification still need to be performed; the vLLM operator oracle is not an
engine benchmark. Inventory and source evidence are recorded in
`/tmp/bitnet-goal-flash-mma/residency-scope.json`.


### Grouped tensor-core prefill qualified (2026-11-08)

Grouped Q4_K/Q5_K gate/up and Q5_1 down now use integer MMA while preserving
the original activation quantization, FP16 scale products and F32 accumulation
order. All 160 nonzero DP4A differential cases, independent long/short reference
phases, full-expert/lazy/uncached modes, entire CPU/CUDA suites, dense Qwen27
regression and memory/synchronization checks pass. No new runtime allocations
or budget changes are introduced. Decode and Q8_0 matrix kernels are unchanged.

Matched local before/after measurements reduce resident-expert 4K prompt time
from 3,336.1 to 1,709.4 ms (48.76%), while streaming improves from 4,659.4 to
4,312.2 ms (7.45%). The smaller streaming gain leaves source/transfer policy
important. Profiling confirms grouped MMA during prefill only, halves summed
prefill kernel duration and retains allocation-free execution. Routing-derived
geometry matches measured launches and identifies 61.06% uniformly skipped
MMA blocks from uneven expert groups; compact tile scheduling is an evidence-
based follow-up experiment. Preserve the numerical reference gates.

The Qwen27 vLLM benchmark preparation now has a locally validated text config,
851 mapped backbone tensors and 48 GDN layout adapters; 15 MTP tensors are
intentionally excluded. GGUF's 65 total blocks include one predictor, matching
64 backbone layers. This is configuration/name-map validation rather than
engine execution. Run the actual fully GPU-resident engine next, verify
placement/precision/correctness and establish isolated matched prefill/decode
metrics. Flash measurements retain the host PLE residency limitation described
above. Continuous batching/KV/tails, broader offload workloads and remaining
model/backend integration are still open. Evidence is in
`/tmp/bitnet-goal-flash-mma/` and `docs/benchmarks.md`.

### Qwen27 engine benchmark runs; numerical comparison remains open (2026-11-08)

The actual pinned vLLM engine now loads the unchanged mixed-quant Qwen27 GGUF
and executes compiled/graph-enabled requests with fully GPU-resident weights.
Native and reference token-delivery harnesses separate TTFT from 63 subsequent
decode steps, retain output IDs and audit placement/configuration. Process-local
image CUDA compatibility libraries solve the reference FA2 PTX loading failure;
an exact-source SM120 rebuild is also recorded as an optional alternative.

This closes engine-execution and residency preparation, **not parity**. All
64 IDs match for the short and repeated-4K cases, but the 129-token case first
diverges at output 3, and the original-reference 1K case at output 22. The
medium mismatch persists as the first fresh-engine request and through both
reference APIs with eight outputs. Preserve the independent arithmetic gates
and expected tokens. Trace aligned logits/intermediates and selected execution
plans next, starting before the first medium divergence. Diagnostic latency
medians are in `docs/benchmarks.md`; they cannot qualify general parity until
the numerical difference is understood. Evidence is under
`/tmp/bitnet-goal-qwen27-engine/`. Native production source is unchanged from
the preceding qualified milestone. Serving, wider offload workloads and the
remaining models/backends remain open.

### Qwen27 eager prefill exact; compiled arithmetic still open (2026-11-08)

The complete 129-token eager/FLA reference prefill now matches native at all
257 observed vectors across 64 layers: 169,743,360 BF16 words, zero differences.
Reference head replay also matches all 744,960 logits for three native hidden
vectors. The default compiled engine's disputed greedy choice is a BF16 tie;
worker reduction permission, capture, logprobs and output count do not explain
that reproducible case. Frozen expected values and production execution remain
unchanged.

Compiled SiLU-and-multiply is demonstrably a different arithmetic profile on
an independent 2,245,632-element batch. Existing native FUSED activation alone
changes the greedy branch but does not restore hidden-vector equality. Do not
promote that partial adjustment or use the passing eager trace as the default
performance baseline. Next inspect generated prefill-kernel boundaries and
qualify the remaining compiled activation/residual/GDN arithmetic with original
reference values preserved. Benchmark metadata now exposes worker precision
and GDN plans; optional FP32-reduction checking is explicit. Full compiled
parity, serving/KV scheduling/tails, wider expert offload and remaining
models/backends are still open. See `docs/benchmarks.md` and
`/tmp/bitnet-goal-qwen27-logits/`.

### Compiled precision contract identified at live kernel boundaries (2026-11-08)

The actual compiled 129-token prefill can now be observed at existing generated
kernel buffers, with token controls before/after instrumentation and restored
backend methods. The first normalization differs from eager/native in one BF16
value; the selected original kernel/configuration replays every compiled value
exactly. Actual layer-0 activations also prove retained FP32 SiLU/product
intermediates, and captured residuals prove that the compiler retains the first
sum in FP32 through the second addition. Eager-style rounding changes 592,671
activation values and 135,284 residual values on those same operands. This is
why fused SiLU alone is insufficient.

Add and qualify a supported backend-owned arithmetic profile for these compiled
boundaries, distinct from the CPU preparation math-library setting. Keep the
existing eager profile and its exact reference gates. Original default compiled
vLLM remains the performance baseline; the diagnostic FLA dispatch override
and alternate reduction configurations do not replace it. The benchmark now
exports cached kernel source/configuration hashes and intermediate-cast policy
so nominal BF16 settings cannot hide these differences. Serving/KV scheduling,
concurrency/tails, wider offload workloads and remaining model/backend hardware
qualification remain part of the active goal. Evidence is in
`/tmp/bitnet-goal-qwen27-kernels/` and `docs/benchmarks.md`.


### Request-local retained FP32 intermediates (2026-11-08)

An explicit `FP32_INTERMEDIATES` BF16 request profile now retains both additions
in the three-input residual in FP32 and selects fused FP32 SiLU/product
intermediates. The existing eager profile remains the default. Profile selection
is immutable until request reset and survives allocation-free restart. Scratch
buffers are reused; the profile adds no request allocations or host arithmetic.
The backend callback validates separate input/output views and supports the
existing contiguous FP32 normalization reduction, including grouped weights.

Independent original PyTorch references qualify 38 cases / 14,407,680 exact
BF16 norm and residual words, including cancellation, grouped weights and wide
rows. The complete CUDA suite passes with previous goldens unchanged. The
Qwen27 real-model runtime gate passes full-logit replay after restart, observer
transparency, interleaved eager-session isolation, matching chunk schedules,
resumed prefill followed by decode, no-logit publication and failure poisoning.
This does not qualify arbitrary compiled vLLM normalization geometry or GDN
arithmetic. Default compiled-vLLM parity remains open. Evidence is under
`/tmp/bitnet-goal-bf16-f32-profile/`.

The new CUDA residual operation also replays all 660,480 stored residual words
from the original compiled 129-token graph exactly on its captured operands.
This gate checks the residual output only, not equivalence of normalization
tiles. A protected clean build and standard suite pass with zero warnings,
and all existing CLI frozen-reference gates pass. Flash Next's 4,097-token
prompt plus 32 greedy outputs matches the saved trajectory exactly with the
expert preload path; this regression checks output IDs, not every layer word.


### Compiled norm reduction isolation (2026-11-08)

A standalone CUDA replay of the selected first compiled normalization now
matches all 660,480 BF16 output words on the captured 129-token operands.
The per-thread four-value reduction uses `(a0+a2)+(a1+a3)`; changing it to a
sequential local sum restores the one-word discrepancy. On these operands,
separate square/add versus FMA and exact division versus reciprocal multiply
do not change that result; this does not establish their general equivalence.

Original cached cubins for the first, two-input and three-input normalizations
all replay their respective captured outputs exactly (1,981,440 words total).
The standalone two/three-input implementations still differ in at least three
BF16 values each across the tested reduction/contraction variants. These are
unresolved diagnostic results, with no production arithmetic changes or new
full-model parity claim. Evidence and CUDA replay source are under
`/tmp/bitnet-goal-compiled-norm/`.


### All three captured compiled norms reproduced (2026-11-08)

The remaining two/three-input standalone norm discrepancies are resolved on
the original captured operands: all three norms now match 1,981,440 BF16
output words exactly. The missing operation was fused mean scaling plus
epsilon: `fmaf(sum, 1.0f / 5120.0f, 1e-6f)`. Disassembly of both original
cached cubins confirms `FFMA` with reciprocal `0x394ccccd` and epsilon
`0x358637bd` before `MUFU.RSQ`. Separate rounded mean and epsilon addition
changes output bits even when the sum of squares is identical.

For the selected 512-thread add-norm kernels, the eight local accumulators
reduce in offset order 2, 4, 1. The 256-thread first norm uses the previously
qualified four-accumulator tree. Exact captured replay uses variant 0 for the
first norm and 288 for both add-norms in the standalone evidence harness.
Original cubin controls remain exact and unchanged; memory checking reports
zero errors. Instrumented PTX copies expose sums/scales for diagnosis only.
No production arithmetic changed in this follow-up. Independent synthetic and
other-shape qualification is required before exposing this reduction contract
in the runtime; full model equivalence and the remaining GDN differences are
still open. Evidence: `/tmp/bitnet-goal-compiled-norm/results9.jsonl`, original
SASS, source and qualification metadata in the same directory.


### Synthetic qualification of compiled normalization formulas (2026-11-08)

The three standalone formulas now match 144,629,760 BF16 values over 120
independent synthetic cases: eight input patterns and batch sizes 1, 5, 17,
129 and 1025, for each of the three norm operations. Patterns include exact
cancellation, small/large scale and wide exponent ranges. Original selected
cubins supply the reference outputs; no reference outputs were adjusted.
Captured replay still matches all 1,981,440 values. Memcheck and synccheck
report zero errors.

The expanded cases exposed two issues hidden by the captured operands. The
first norm also requires fused mean scaling plus epsilon. More importantly,
the persistent two-input norm combines local square pairs with FMA within
each tile before combining tiles, whereas the three-input norm accumulates
across tiles before local reduction. Treating these as the same arithmetic
left one BF16 mismatch in the 1025-token wide-range case. Keeping their
operation order distinct removes it. Current replay selections are variant
32 for the first norm and 288 for both add-norm entry points, whose kernel
branches implement different square/reduction orders.

This qualifies the formulas against those selected original configurations
at dimension 5120, epsilon 1e-6 and weight offset 1; it does not qualify other
compiler tiles or full-model parity. Production arithmetic is unchanged in
this follow-up. Next integrate a named reduction contract with independent
synthetic goldens and retain existing eager gates. Evidence/source:
`/tmp/bitnet-goal-compiled-norm/qualification.json` and `synthetic.cu`.


### Backend tiled normalization contract implemented (2026-11-08)

CUDA resident RMSNorm, add-RMSNorm and three-input add-RMSNorm now accept
`BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA`. This is an explicit fixed arithmetic
contract: dimension 5120, epsilon 1e-6, shared BF16 gamma, weight offset 1.
Other shapes/settings reject before work instead of silently selecting another
reduction. The first, persistent two-input and looped three-input kernels
retain their separately qualified square/reduction orders and fused mean plus
epsilon. Intermediates remain on device; no request allocations are added.
Existing default/contiguous reductions retain their original arithmetic.
Ordinary dense request initialization does not yet select the new contract.

The full CUDA suite passes, including 120 new independent original-cubin cases
with 144,629,760 exact norm words and 48,209,920 exact three-input residual
words. Seven invalid-plan variants are checked through all three callbacks
for every case. The repository generator reproduces the frozen header exactly;
memcheck and synccheck both report zero errors. Reference provenance, cubin
hashes and selected configurations are in
`test/cuda_compiled_norm_reference.json`; the generator is
`test/reference_cuda_compiled_norm.cu`. Existing reference values are preserved.
Evidence and command logs: `/tmp/bitnet-goal-compiled-runtime/`.

Next connect this reduction contract to an explicit dense request arithmetic
profile and qualify complete short/long/chunked/resumed reference traces.
Native GDN/attention equivalence and default compiled-vLLM parity remain open;
passing these operator gates alone does not close those requirements.

The protected clean build and standard tests also pass with zero warnings;
all four pre-existing probe binaries are preserved byte-for-byte. Frozen CLI
integration passes all 20 cases. Flash Next's 4097-token prompt and 32-output
trajectory matches the saved expert-preload baseline (output-ID regression,
not an all-layer comparison). On original captured three-input operands, the
production callback matches 660,480 norm and 660,480 residual words exactly.
The broader request-level compiled comparison remains the next gate.


### Explicit tiled-FP32 dense request profile (2026-11-08)

`BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32` now selects the qualified fixed
tiled normalization contract together with retained FP32 activation/residual
intermediates. Backend reduction capabilities are declared explicitly; missing
support rejects before request allocation. The profile is request-local,
survives allocation-free restart, and uses the ordinary forward/prefill paths.
The eager default and existing FP32-intermediate profile remain available.
`bench_bf16_latency` accepts `tiled-fp32` and records that label in JSON.

The real-model runtime gate passes for both explicit profiles: complete logit
replay across restart, observer checks, interleaved eager-session isolation,
matching chunk schedules, resume/decode, no-logit publication and failure
invalidation. The new profile adds no request allocations or CPU fallback.
Its full runtime memory check, full CUDA regression and protected clean/CLI
checks are still pending in `/tmp/bitnet-goal-tiled-request/` at this entry.

One warmup plus three measured 64-output runs yield median TTFT/decode-step
milliseconds: 5 tokens 49.66/15.56; 129 tokens 213.62/15.62; 1025 tokens
1209.56/15.88; 4097 tokens 4668.55/16.34. Within-case output IDs are stable.
Against the saved original compiled reference, short and 4097-token cases
match all 64 output IDs; the 129-token case still differs at output 3
(63 versus 12), and the 1025-token case first differs at output 51.
This is not a fresh paired vLLM timing run or full numerical parity claim.

Read-only inspection of the exact reference image identifies another concrete
GDN difference: FlashInfer SM120 casts its F32 KK accumulator to FP16 before
`CollectiveInverse`, which uses FP16 MMA operands and stores, while native
prefill retains F32 inverse profile4 before its BF16 conversion. A queued
first-layer capture will localize the remaining divergence. Source hashes and
original files are preserved under the same evidence directory; no GDN
arithmetic changed in this implementation.

Further read-only inspection of the pinned FlashInfer source identifies two
additional GDN arithmetic boundaries: alpha processing scans
`log2(exp(g) + 1e-10)`, and beta multiplies the F32 key-dot accumulator after
the dot product. Native prefill scans log decay directly and scales operands
before the correlation. These source-level differences require actual
operand/output comparison; they are not yet a proof of the first divergent
operator. An original-GDN capture wrapper is prepared under
`/tmp/bitnet-goal-qwen27-gdn/controller/` with syntax checks only. It has not
been launched; the existing serialized runtime memory-check/regression pipeline
remains active. No production kernel changes accompany this investigation.


### Tiled request qualification complete; GDN core isolated (2026-11-08)

All queued native checks now pass: full real-model runtime memcheck reports
zero errors, the full CUDA suite passes, protected clean build and standard
tests pass with zero warnings, all frozen CLI cases pass, and the Flash Next
4097-token/32-output trajectory remains exact. Four pre-existing probe
binaries retain their hashes. No request arithmetic defaults changed.

A controlled capture from the original default compiled reference now compares
layer0 directly with the native tiled profile. Input normalization (660,480
BF16 values), prepared Q/K/V, log decay, beta and the F32 initial state all
match bit-for-bit. The GDN output differs in 388,377 of 792,576 BF16 values
(max absolute error 0.078125); next state differs in 776,661 of 786,432 F32
values (max absolute error about 0.112315). Thus a remaining difference is
inside the GDN execution path, with matching actual operands. Instrumentation
preserves eight generated control IDs before/during capture and after fully
restoring the first-norm launcher and GDN dispatch bindings.

Direct replay of these captured operands identifies the actual default as
FlashInfer's `cp_delta_rule_dsl_sm120` context-parallel path. Default-auto and
forced-CP replay reproduce all 1,579,008 captured output/state values exactly.
Forcing the non-CP SM120 path instead changes 175,624 BF16 output values and
668,317 F32 state values. Non-CP is a diagnostic counterfactual, not an
acceptable replacement for this default compiled comparison. A native
compatible profile must qualify the CP phases and their arithmetic.

This fresh reference initialization selects token 63 at medium output3,
whereas the saved earlier benchmark selected 12; all three controls within
the fresh engine agree. Existing frozen fixtures and saved benchmark outputs
remain unchanged. The operator comparison relies on exact captured operands,
not cross-initialization greedy-ID agreement. Full-model equivalence,
matched performance parity and the broader serving/model/backend requirements
remain open. Evidence: `/tmp/bitnet-goal-qwen27-gdn/qualification.json`;
native qualification: `/tmp/bitnet-goal-tiled-request/qualification.json`.


### First native FlashInfer-CP stage reproduced (2026-11-08)

A stage-level capture of the unchanged original CP call now freezes signed
inverse tiles, local transfer/state, fixed-up state, final output and next
state. Copying these boundaries preserves all 1,579,008 original output/state
values. The 129-token call chooses CP chunk length 512. Its T precompute is
independent of decay: `T = -(inv(I + lower(beta_row * KKT)) beta)^T`.

A standalone CUDA prototype reproduces the T stage exactly: all 589,824 BF16
values from the captured prompt and 32,243,712 values from 40 independent
original-kernel synthetic cases. Batches span 1, 5, 17, 63, 64, 65, 129, 257,
513 and 1025, with tiny, signed and sparse keys and beta values including zero
and one. Original references are generated twice and checked for equality;
existing goldens remain unchanged. Memcheck and synccheck report zero errors.
The prototype uses BF16 tensor-core Gram products, FP16 inverse storage,
explicit F32-FMA diagonal8 elimination, FP16 MMA block composition and the
reference's FP16 rounding of64-level partial products before their addition.
A CPU diagnostic with rounded high-precision matmuls still differs in 592
captured BF16 values and is not used as a golden or implementation substitute.

No production GDN kernels changed. The remaining CP transfer/state, fixup and
final prefill phases must be implemented and qualified, including nonzero
initial state and multiple CP chunks, before exposing a complete compatible
profile. Full-model parity and the original wider goal remain open.
Source, references and qualification: `/tmp/bitnet-goal-qwen27-gdn/cuda-t/`,
`/tmp/bitnet-goal-qwen27-gdn/t-synthetic/` and `stages/`.


### Native CP transfer/state and fixup stages qualified (2026-11-08)

The standalone CUDA prototype now reproduces local transfer and state
precomputation exactly: 185,597,952 F32 words over 40 independent original
FlashInfer cases. The composed T/MN/fixup pipeline also matches all 92,798,976
F32 prefix-state values, totaling 278,396,928 exact F32 comparisons. Cases use
zero/nonzero F32 initial state, lengths 1, 17, 63, 64, 65, 129, 257, 321,
513 and 1025, and explicit CP lengths 64/512 (up to 17 chunks). All references
are generated twice and checked for equality. Memcheck and synccheck report
zero errors for both MN and the composed fixup suite. Captured real-model
transfer, local state and fixed state each match all 786,432 F32 values.

The MN implementation consumes the transposed view of the physical signed T
matrix, preserves the reference's BF16 intermediate stores, and keeps decay
placement distinct for transfer/state updates. For 48 heads, fixup selects
raw-F32 TF32 MMA with even/odd feature permutation inside each 8-wide block;
introducing an explicit conversion or using the old KKT's 16-wide permutation
would change this contract. Fixed-state index j stores the state after chunk j.

Final CP prefill remains unimplemented. Its final recurrent state must not be
replaced by the last fixed state: those reference tensors differ in 786,418
of 786,432 captured F32 values (max difference about0.08454). Native final
output/state, runtime integration and full model/performance/serving goals
remain open. Production GDN kernels are unchanged; all new kernels are still
standalone prototypes. Source and evidence are under
`/tmp/bitnet-goal-qwen27-gdn/cuda-mn/`, `cuda-fixup/`, and `mn-synthetic/`.

### Current Qwen27 CP execution and CLI qualification (2026-11-08)

The complete native CP GDN pipeline is integrated into the request-local
CP_FP32 arithmetic profile, including log-decay staging, transfer/state,
fixup, final prefill and F32 continuation state. Fifty-six independent
operator cases qualify 551,755,776 exact output/state words. The fixed
tiled normalization and compiled attention gate are qualified independently.

For the pinned fully resident Qwen27 checkpoint on the designated SM120
GPU, explicit CP_FP32 plus MKL VML HA passes the measured single-request
8192-budget vLLM latency gate. The latest short-prompt TTFT is 52.911 versus
50.500 ms (4.77%); remaining measured prefill and decode cases are faster.
All four 64-token streams agree with the preserved arithmetic-profile
reference. The fresh stock run's medium/long streams differ at indices 2/21;
current stock bit identity and profile-matched correctness are recorded
separately. Hardware, precision, context and workload limits are recorded in
`docs/benchmarks.md`.

The 512-token chunked and explicit-resume schedules also match 64-token
independent streams when the original reference normalization kernels use
the same declared reduction configurations. The unsplit control matches
84,541,440 normalization words exactly. Stock vLLM chooses different
arithmetic under its 512-token budget and retains different output streams;
both stock baselines and all original goldens are preserved.

The CLI exposes this profile through `--cuda-bf16-profile cp-fp32`, retaining
the eager default. Clean build, CPU/CUDA suites, request-state tests and
both real CLI suites pass. The new CLI fixture covers four prompt lengths
and both chunk schedules. Evidence is under `/tmp/bitnet-goal-cp-layout/`,
`/tmp/bitnet-goal-chunk-matched/` and `/tmp/bitnet-goal-cp-cli/`.

The full goal remains open. Flash expert placement/cache and grouped MMA
already have local qualification. Compacting uniformly returning expert-grid
blocks did not establish a latency gain, so the original grid was retained.
The expanded 192-case differential and memory/synchronization checks pass.
Paired profiles identified Q8 matrix projections at about 24% of summed
prefill kernel time. SM120 integer-MMA execution is now integrated with its
original F32 scale/FMA contract, retaining the independent per-expert DP4A
path. Whole-model resident/lazy/uncached oracles, operator differentials,
CPU/CUDA suites, sanitizers and dense CLI/request regressions pass. Resident-
expert Flash prefill falls from 1641.7 to 1354.7 ms in the measured 4K case;
streaming does not show a meaningful improvement. Evidence is under
`/tmp/bitnet-goal-flash-compact/` and `/tmp/bitnet-goal-q8-mma/`.
Offload policy and cold/
warm transfer accounting, matched placement comparisons, continuous batching,
KV scheduling, concurrent throughput/tail latency, DeepSeek/GLM execution
and additional backend hardware qualification still require work.
### Flash bounded-cache supply measurement (2026-11-09)

The reusable `bench_flash_offload` now reports cold/restarted request timing,
RAM/GPU cache hits, source reads and expert transfer bytes. With a 12 GiB GPU
expert budget on the measured short repeated Flash request, the 2 GiB RAM cache
thrashes on every GPU miss; direct PRead staging averages 18.91 ms per warm
decode step versus 22.89 ms with that cache. An 8 GiB RAM cache eventually
eliminates source reads and reaches 17.74 ms. All nine 64-token streams match.
Independent GPU traces validate copy bytes in all 192 phases. Timed copy
activities and CPU wrapper spans overlap; isolated critical-path transfer
stalls remain unqualified. No runtime cache defaults changed.

Next, evaluate RAM admission and CPU staging against the GPU-conditioned miss
stream, retaining cold/warm controls and bounded ownership. Serving, concurrent
throughput/tails, matched placement comparisons, DeepSeek/GLM and remaining
hardware qualification remain part of the full unfinished goal. Detailed scope
and reproduction are in `docs/benchmarks.md`; evidence is under
`/tmp/bitnet-goal-offload-supply/`.

The subsequent external frequency-admission experiment reduces source reads and
improves the undersized LRU control, with exact tokens/GPU counters and independent
phase replay. Direct PRead into pinned staging remains competitive on repeated
requests and faster on the controlled alternating sequence. Do not integrate
frequency admission solely on read-count improvements: it has not demonstrated
an advantage over that control. Evidence and reversed-order measurements are in
`/tmp/bitnet-goal-cache-admission/`. Runtime defaults remain unchanged. Further
offload work needs reduced staging cost or measured critical-path evidence;
continuous batching/KV scheduling and the rest of the full goal remain open.

### Concurrent-session execution baseline (2026-11-09)

`bench_bf16_concurrency` now exercises queued admission into bounded live-session
slots, chunked prefill continuation and interleaved decode on the real Qwen27
CUDA path. It compares each request with isolated execution using identical chunk
boundaries, and records queue-inclusive TTFT/completion plus every output delivery
time. The uniform 1/2/4/8-live cases preserve all streams. Aggregate throughput
stays near 51–53 tokens/s while per-token inter-delivery p95 grows from 15.87 to
131.94 ms. This is serialized execution of concurrent session lifetimes, not fused
cross-request batching or a production scheduler. Context slots are bounded;
byte-accounted or paged KV allocation is still missing. Hull remains HTTP owner.

The next execution work must batch projections across request rows while keeping
each request's F32 recurrent state, KV position and arithmetic profile separate.
The existing GDN decode plan represents one token/state, and attention represents
one contiguous causal sequence; concatenating request rows as one sequence would
be incorrect. A scheduler loop alone does not meet the serving/performance goal.
Use the mixed short/long cases to qualify prefill interference and delivery tails
as these operators and KV ownership evolve. Evidence is under
`/tmp/bitnet-goal-serving-baseline/`; the full goal remains unfinished.

The first batched recurrent primitive is now qualified: packed independent-
sequence GDN decode preserves separate F32 states and the original arithmetic.
Batches 1/2/3/8/16 match 408,672,000 frozen reference output/state words. GPU
tracing confirms one launch per batch, and the complete CPU/CUDA suites,
sanitizers and preserved model streams pass. Evidence is under
`/tmp/bitnet-goal-gdn-batch/`. This is operator qualification, not serving
integration or a throughput gain. Next add validated direct access to session-
owned recurrent state, avoiding full state packing each step, then combine
shared-row projections and sequence-aware attention with admission/KV scheduling.
The full model, serving, parity and backend scope remains open.

Direct multi-request GDN state access is now implemented and qualified. A
separate coordinator joins participant streams using events and accesses their
validated session-owned F32 buffers directly. Its 226 traced kernels use only
54,040 bytes of pointer-descriptor transfers, with no activation/state staging.
The unchanged independent reference fixtures, complete CPU/CUDA suites,
sanitizers, busy/retry and preserved model streams pass. Evidence is under
`/tmp/bitnet-goal-gdn-direct/`. This removes the full-state packing requirement
for GDN; whole-model batching still needs shared-row projection/activation
access, independent convolution histories, per-sequence RoPE/attention positions
and KV/admission scheduling. No serving throughput or vLLM parity gain is claimed
by this operator qualification.

Shared coordinator-owned activation rows are now accepted by GDN groups while
F32 state remains participant-owned. Cross-row write aliases and foreign state
ownership are rejected before submission. Shared-row and original group modes
pass their unchanged independent references, full suites and sanitizers; GPU
traces show no D2D staging. Evidence is under `/tmp/bitnet-goal-gdn-rows/`.
Next compose actual batched projections with this operator, then integrate
independent convolution history, RoPE positions, ragged attention and KV/admission
scheduling. These interfaces are not yet a whole-model batched serving runtime.

Actual decoded-BF16 projection → grouped GDN composition is now qualified with
device-resident intermediates and independent Torch/vLLM pipeline references.
The selector fixture exposed GEMM zero canonicalization changing five F32 state
hashes while preserving all BF16 output hashes; separate pipeline goldens retain
the original direct-input references. All 408,672,000 composed values, full
suites, sanitizers and preserved model streams pass. GPU traces distinguish
batched matrix kernels from singleton GEMV. Evidence is under
`/tmp/bitnet-goal-projection-gdn/`. Next qualify checkpoint-format projections
and integrate independent convolution history, per-sequence RoPE/attention and
KV/admission scheduling into actual whole-model batching. The full goal remains
open.

### Checkpoint-format multi-request composition (2026-11-09)

The optional `bench_checkpoint_gdn` gate now qualifies actual captured Qwen27
layer-0 Q5_K/Q8_0 projections through independent-state GDN and Q8_0 output
projection. Forty phases at batches 1/2/4/8 match 122,126,400 independent values
byte-for-byte, including carried F32 state. The published Torch/vLLM generator
reproduces all fixtures; memcheck/synccheck and standard checks pass. Trace
selection and transfers are recorded under `/tmp/bitnet-goal-checkpoint-gdn/`.
This controlled graph omits convolution, gating/layout and full-model residuals.
Next implement independent convolution history and per-sequence RoPE/attention,
then connect them to whole-model batching and KV/admission scheduling. Operator
correctness does not establish serving performance parity; the full goal remains
open.

### Grouped convolution histories (2026-11-09)

Independent one-token convolution histories are now implemented on CUDA, with
strict participant ownership, shared activation rows and cross-row alias checks.
Separate bounded convolution/GDN descriptor storage permits direct composition.
Staggered active requests match the existing convolution references, and the
real checkpoint projection → convolution → GDN → output graph matches
128,270,400 independent values. Full suites, sanitizers and preserved model
streams pass; evidence is under `/tmp/bitnet-goal-conv-group/`.

Next integrate gated normalization/head layout and per-sequence RoPE/attention,
then whole-model batch orchestration and KV/admission scheduling. Descriptor
reuse can apply bounded backpressure; measure multi-layer submission behavior
before choosing larger metadata queues. This operator work does not establish
serving performance parity or complete the full goal.


### Complete controlled GDN subblock (2026-11-09)

The optional checkpoint gate now composes convolution, GDN, gated RMSNorm,
head-layout transpose and output projection using captured Qwen27 layer-0
weights. Forty phases at batches 1/2/4/8 match 132,571,200 independent values
byte-for-byte. The reference uses the original vLLM/plugin operations, including
normalization and layout. Existing raw/conv fixtures remain unchanged and pass.
Clean CUDA build, standard tests, memcheck and synccheck pass; evidence and
kernel/transfer traces are under `/tmp/bitnet-goal-gdn-fullblock/`.

Inputs are supplied BF16 hidden states; outer input normalization, residuals,
FFN and whole-model orchestration are outside this gate. Intermediates remain
on-device, with split/transpose copy kernels. Trace checks find no runtime
allocations between the first and last GDN launch per case. Validation readbacks
are additional traffic; these measurements do not establish serving parity.

Next add independently positioned RoPE/attention with request-owned KV and
shared projection rows, then whole-model batch orchestration and bounded
KV/admission scheduling. The full goal remains open.


### Grouped QK preparation and independent RoPE positions (2026-11-09)

CUDA now prepares Q/K normalization, RoPE, V and attention gates for distinct
requests in one launch. Every sequence retains its own position, token count,
geometry and immutable weights; activations may belong to its request or an
explicit coordinator. Flattened token/head tiles avoid padding all requests to
the longest chunk. The singleton arithmetic remains shared with this path.

A bounded descriptor workspace joins participant streams and rejects cross-row
write/read aliases before submission. Busy metadata returns a retryable result.
Success uses no allocation or explicit host fence; partial submission failures
fence the coordinator and poison participants. Caller completion requires
successful coordinator and participant fences before committing logical state.

All 17 existing frozen vLLM QK/RoPE fixtures match across group sizes
1/2/3/8/16 and four ownership/packed layouts: 717,230,080 BF16 words. Clean build,
standard/CUDA suites, diagnostic isolation, memcheck and synccheck pass. The
complete controlled GDN checkpoint gate and preserved model token streams also
pass. Trace evidence under `/tmp/bitnet-goal-qk-group/` confirms 340 independent
preparations in 148 group launches and bounded descriptor uploads; validation
readbacks are additional traffic. These fixtures are not serving benchmarks.

Next add grouped attention over participant-owned KV caches, connect shared
projection rows through whole-model execution, then qualify continuous batching
and bounded KV/admission scheduling. Multi-layer descriptor backpressure still
needs measurement. This work does not establish concurrent vLLM parity; the
full goal remains open.


### Grouped attention with independent KV caches (2026-11-09)

CUDA now batches one-token attention decode across distinct live requests.
Cache pointers, logical positions and attention scratch remain participant-owned;
activation rows may belong to an explicit coordinator. KV append, FA2 forward,
split combine and gating use grouped launches. Each request retains the existing
singleton split count and reduction specialization. Upstream FA2 sources are
unchanged, and singleton and grouped paths share parameter and gate arithmetic.
The initial group profile supports head size 256 and at most 64 Q heads per KV head.

The bounded workspace validates every row and cross-row alias before submission,
joins producer/consumer streams, and returns retryable busy metadata without
submission. Success allocates nothing and uses no explicit host fence; partial
submission failure fences and poisons the group. Completion requires successful
coordinator and participant fences before committing logical progress.

Staggered groups of 1/2/3/8/16 with three ownership layouts and both gate profiles
match 5,071,872 frozen attention/gate words and 2,117,099,520 KV words. Initial
synthetic KV prefixes match the original oracle inputs; subsequent decode state
persists and a 17-token resumed prefill gap uses the existing participant path.
Clean build, standard/CUDA suites, diagnostics, memcheck and synccheck pass.
The controlled GDN checkpoint gate and preserved model token streams also pass.

Evidence under `/tmp/bitnet-goal-attention-group/` records 270 decode updates in
108 grouped forward launches, 96 combines and 36,720 descriptor upload bytes.
No runtime allocations occur between the first and last decode per fixture.
The new kernels report zero stack/local memory in the resource dump. Validation
readbacks are additional traffic; these fixtures do not establish serving speed.

Next compose actual checkpoint projections, grouped QK/RoPE, grouped attention
and output projection, then connect whole-model batching and bounded KV/admission
scheduling. Multi-layer descriptor backpressure and concurrent vLLM parity remain
unqualified. The full goal remains open.


### Checkpoint projections through grouped attention (2026-11-09)

The optional `bench_checkpoint_attention` gate now composes Qwen27 layer3
Q5_K/Q4_K/Q5_K projections, grouped QK/RoPE, grouped FA2 with request-owned KV,
BF16 sigmoid gating, and Q6_K output projection. Batches 1/2/4/8 across two
seeds and five steps match 670,658,560 independent BF16 values, including every
cache word. Requests start at different positions, including contexts beyond
4095, and carry their own cache updates. All graph operations are queued before
the first validation readback. The published reference uses the original pinned
vLLM/plugin operations and records checkpoint tensor and norm/cache hashes.

Clean build and standard tests pass without warnings; graph memcheck/synccheck
and C11 stub/usage checks pass. Runtime sources remain byte-identical to the
prior full CUDA/model-qualified turn. Evidence is under
`/tmp/bitnet-goal-checkpoint-attention/`. Trace checks confirm 40 grouped QK and
40 grouped attention launches, 1,536,000 input and 42,000 descriptor H2D bytes,
13,824,000 coordinator validation D2H bytes, and no runtime allocations during
the composed execution spans. Participant KV setup/readbacks are extra traffic.
Kernel selection records matrix and multivector paths for the different shapes;
it does not assume every small batch selects GEMM.

The gate supplies normalized BF16 hidden states and analytical KV prefixes;
outer input normalization, residuals, FFN and whole-model orchestration are
excluded. It covers the BF16 sigmoid profile and does not establish prefill or
serving performance. Next integrate the qualified GDN and attention subblocks
into whole-model batched decode, then continuous batching and bounded KV/admission
scheduling. The full goal remains open.


### Experimental whole-model decode batching (2026-11-09)

The dense hybrid BF16 runtime now has an explicit batch coordinator and executes
all 64 layers plus vocabulary projection for ready decode rows together. Each
participant retains its own F32 SSM state, convolution history, KV cache, position
and parity. Coordinator storage contains activations and workspaces, without
allocating redundant layer states. Singleton and batch paths share the layer
graph, projections, norm/residual math and SSM output tail. Input validation
precedes submission; logits and positions publish only after all completion
fences. Partial execution failure poisons the coordinator and every participant.

`bench_bf16_batch` admits bounded live sessions and batches ready decode rows,
with one round-robin prefill chunk at a time. Preparation reserves a bounded
64-slot pinned source queue per operator; the measured single-slot implementation
needed 383 retry fences in a two-request run, versus zero with these queues.
Existing primitive reservations retain their single-slot behavior. Whole-model
traces confirm 48 grouped SSM and 16 grouped attention layers per call, token-ID
and descriptor uploads only, and no CUDA runtime allocations inside decode spans.

Clean build, standard/CUDA suites, diagnostics, and preserved legacy model streams
pass. One/two-request greedy streams match isolated native execution, including
mixed lengths, session turnover and one-token prefill chunks. Batch-of-one full
logits match exactly. Invalid-cohort checks preserve state/output. An injected
FFN error after grouped SSM submission verifies poisoning and no partial logits;
whole-model memory/synchronization sanitizer runs pass, including that failure
path. Evidence is under `/tmp/bitnet-goal-model-batch/`.

Larger-cohort accuracy qualification remains open. Four/eight-request runs fail
the strict isolated-token check at output 4. Diagnostic row-wise body projections
restore the streams; no scalar fallback was added to the runtime. An independent
original-vLLM replay of 308 same-input prepared body matrices matches both native
batch and row-wise outputs exactly (4,044,288 BF16 values per mode), while the
reference modes themselves differ in 2,143,749 values. This establishes a real
batch-shape arithmetic difference, not full-model correctness: the replay does
not validate complete model layout, state and norm composition.

Next obtain an independent full-model batched oracle with explicit effective
batch shapes, compiler/capture settings and prefill boundaries. Keep the larger
isolated-token gate failing until an appropriate independent batch comparison
qualifies the result. Production continuous scheduling, byte-accounted KV
admission, batched prefill and concurrent vLLM performance parity remain open.
The full goal is active.


### Independent dense batching and bounded admission qualification (2026-11-09)

Independent original-runtime full-logit references now qualify singleton,
four-request, sixteen-request, chunked prefill, and bounded twenty-request
admission/refill cases for the resident Qwen3.8 27B GGUF/BF16 CUDA path. The
original frozen oracles are unchanged. Explicit arithmetic policies derive
normalization and CP/NONCP recurrence from actual original execution; cohort
attention splitting includes one-token prefill tails. No default arithmetic
promotion or other vLLM format/graph comparison follows from these results.

All measured cases meet the prefill/decode floors. The twenty-request burst
with sixteen live requests matches all 1280 complete BF16 output vectors and
all three timing streams. Prefill throughput is 2.272x the matched reference;
mean decode engine steps are 3.4% faster, and end-to-end throughput is 111.155
versus 83.185 tokens/s. Median run p99 inter-token latency is 93.932 versus
101.021 ms, after publishing ready tokens before reclaiming completed sessions.
The reclamation cost remains timed. Model-free evidence checks reject missing
logits, bad admission bounds, wrong streams, and disk-instrumented timing runs.

Portable references, policies, hashes, per-case limitations and metrics are in
`docs/qwen38-cuda-matched-cases.json` and `docs/qwen38-cuda-correctness.md`.
Clean builds, standard tests, CUDA diagnostics/memory checks and real-model
runtime gates pass for the production changes; subsequent benchmark/evidence
changes pass clean builds, standard tests and independent full-model replay.

This is a bounded uniform-wave benchmark, not a production serving scheduler
or byte-accounted KV pool. Mixed and staggered work, production admission/KV
scheduling, expert RAM/VRAM placement and cold/warm stalls, other backend and
model qualification remain open. The full goal remains active.

Mixed Qwen3.8 CUDA correctness and the bounded four-request serving geometry
are now qualified: all 260 replay heads and all 256 sampled serving heads match
independent full-vocabulary BF16 references. The combined scheduler entry point
preserves explicit recurrent roles and distinct mixed decode arithmetic.
Median remaining decode is within 2.74% of the pinned reference; the initial
mixed phase is 1.806x faster. These results do not cover staggered arrivals,
a production KV memory pool or mixed-request tail parity. See
`qwen38-cuda-mixed-correctness.json` and `qwen38-cuda-correctness.md`.

### Dense request KV admission and reuse

The dense BF16 runtime now has a caller-owned request-pool handle with immutable
capacity, chunk limit and arithmetic policy. Admission checks a KV payload byte
budget before allocating another slot. Idle slots remain charged; synchronized
restart recycles their state, while poisoned or failed-restart slots are freed
and uncharged. The pool does not budget SSM/activation/workspace/model memory
and does not implement paged KV or the full continuous scheduler.

The protected clean build and standard suite pass without warnings; ownership
and failure checks pass under address/undefined-behavior sanitizers. Loaded-model
allocation interception proves the KV charge; focused GPU memory checking reports
zero errors. A 20-request/16-slot replay matches all 1,280 complete reference
heads, reuses four slots and retains exactly 8 GiB of charged KV including idle
slots. The mixed entry point also matches all 256 sampled heads with pooling.

Three normal runs give median prefill 2934.878 ms versus 6612.085 ms, decode engine
step 66.242 ms versus 69.583 ms, and throughput 112.379 versus 83.185 tokens/s.
TTFT p95 is 8944.572 versus 12946.963 ms; inter-token p99 is 93.939 versus
101.021 ms. This qualifies the bounded uniform arrival waves, with persistent
pool destruction outside request timing and admission/restart/return inside.
Staggered arrivals, variable-capacity/total-request-memory admission, production
continuous scheduling, expert supply critical-path stalls and the remaining
model/backend scope keep the full goal open. See `qwen38-cuda-session-pool.md`
and `qwen38-cuda-session-pool.json`. Evidence:
`/tmp/bitnet-goal-dense-kv-pool-init`.


### Idle KV growth qualified (2026-11-10)

Request-sized CUDA admission now replaces only idle attention KV storage,
retaining session workspaces and recurrent buffers. Old view generations and
slices are rejected; partial failure discards the session. The unchanged
48-request independent control matches 4,132 full heads. Its pool creates eight
sessions, reuses 40, and performs 12 idle resizes without evictions, retaining
512 MiB instead of fixed mode’s 4 GiB of KV payload.

Three paired controlled timing runs give 69.28 versus 69.10 output tokens/s,
p99 70.56 versus 70.50 ms, and p99.9 92.01 versus 95.81 ms (sized/fixed).
No sized run has an inter-token gap above 100 ms. Clean build, standard/CUDA
suites, resize memcheck/synccheck and real-model growth/failure checks pass.
This remains opt-in KV-only admission, with no total-memory, sustained-serving
or new vLLM parity claim. Expert critical-path supply and the broader full goal
remain open. See [qualification](qwen38-idle-kv-resize.md).


### GLM5.3 portable KDA state primitive (2026-11-10)

The SSM module now has a coordinate-wise log-decay KDA step with caller-owned
F32 state, separate from Qwen's scalar-per-head GDN. Five unchanged original
GLM Triton captures and a float64 recurrence qualify 120,699,708 comparisons
across portable, AVX2 and AVX512 builds, including every prefix, head partitioning
and resumed caller storage. Maximum observed state/output differences are below
3e-7/8e-8. A small captured fixture runs in the standard SSM suite; an offline
verifier reproduces the larger matrix. Clean build, standard/CUDA suites,
diagnostic isolation and scalar SSM ASan/UBSan pass.

The local GLM Q4_K_XL six-shard set has 1,412 tensors and metadata for 45 main
layers plus one MTP. The first shard is metadata-only. Model support still requires
gate/convolution/output handling, mHC, indexed MLA and loader/session/backend
wiring; no architecture alias or full model/performance claim is introduced.
See [qualification and limits](glm53-kda-state-step.md). Full goal remains active.


### GLM bounded KDA coefficients and stored decay (2026-11-10)

Portable KDA coefficient construction now supports raw A_log and stored
negative exponentials explicitly. Across all 34 local GLM KDA layers, GGUF
ssm_a follows -exp(A_log) within F32 rounding and timestep biases match exactly.
The API preserves stored values without a logarithm/exponential round trip.
Five unchanged original gate/sigmoid captures and a fused gate/recurrence capture
qualify 217,326,408 comparisons across three compilation modes, two parameter
representations, head partitioning, in-place outputs and resumed caller state.
Clean build, standard/CUDA suites, diagnostic isolation and scalar SSM sanitizer
checks pass. No model alias, CUDA coefficient kernel or full-model performance
claim is introduced. mHC, indexed MLA and complete GLM integration remain open.
See [qualification](glm53-kda-coefficients.md). The full goal remains active.


### Portable mHC coefficients and residual mixing (2026-11-10)

The transformer math module now provides mHC pre/post coefficients, fixed-count
Sinkhorn normalization, weighted collapse and source-to-destination residual
mixing. Seven unchanged original CPU Torch helper captures qualify 1,436,049
values across scalar, AVX2 and AVX512 builds, including GLM's 4096 and
DeepSeek's 5120 widths. Maximum coefficient and coupled-mix differences are
below 5e-7. The standard suite includes an immutable fixture and non-symmetric
orientation guard; clean build, standard/CUDA regression suites and ASan/UBSan
pass. No CUDA mHC kernel, model alias or performance claim is added. Projection,
global RMS, BF16 rounding and full model wiring/qualification remain open.
See [qualification and limits](mhc-core.md). The full goal remains active.


### Direct F32 KDA Q/K normalization (2026-11-10)

An explicit sum-epsilon L2 mode now normalizes Q/K in F32 without a BF16
intermediate. Existing clamped and RMS-scaled modes retain their behavior; all
CPU dispatchers route the new mode through a shared bounded implementation.
Four independent original GPU normalization/recurrence captures qualify
122,541,840 values across scalar, AVX2 and AVX512 builds, head partitioning and
resumed caller state. Maximum prefix-state difference is below 1.8e-7.
Clean build, standard/CUDA regression suites, diagnostic isolation and scalar
SSM sanitizers pass. NEON/WASM routing has not been executed on matching hardware.
No GLM model alias or new CUDA normalization kernel is introduced. Complete
convolution/output rounding boundaries, indexed MLA and model wiring remain open.
See [qualification](glm53-qk-normalization.md). The full goal remains active.

### GLM F32-weight resident convolution

Added a separate CUDA operation preserving F32 convolution weights, ordered F32
FMA accumulation, BF16 activation/history and final output rounding. The older
BF16 product-rounding operation keeps its contract. Thirty unchanged-original
reference cases qualify full, split and one-token execution: 714 launches over
9,423,054 output/history words, including the 24,576-channel GLM shape. Original
captures reproduce exactly; clean/standard/full CUDA/diagnostic checks pass and
Compute Sanitizer reports zero errors. Complete KDA recurrence/output boundaries,
CPU rounding differences, indexed MLA and whole-model wiring remain open.
See [qualification](glm53-cuda-convolution.md). The full goal remains active.

### Native resident GLM KDA recurrence

Added native CUDA KDA with F32 normalized Q/K and state, bounded per-key decay,
raw-log or directly stored negative-exponential parameters, and a BF16 core
output boundary. The independently qualified contracted dot reduction preserves
original prefix states and outputs exactly. Sixteen original-reference cases
cover 396 full/split/one-token launches and 95,931,904 state/output words.
Actual native convolution feeding KDA also matches the earlier GLM captures
across six schedules and 26,701,824 convolution/core/history/state words.
Clean build, standard/full CUDA suites, diagnostic isolation and memcheck pass.
Output RMS sigmoid gating with F32 gamma, indexed MLA, resident mHC, model wiring,
whole-model correctness and performance acceptance remain open. Recurrent KDA
does not yet supply a matrix-based parallel prefill path or measured vLLM parity.
See [qualification](glm53-cuda-kda.md). The full goal remains active.

### GLM projected recurrent KDA output boundary

Added the explicit GLM_R2 resident RMS/sigmoid profile with full F32 gamma,
fixed row reduction and square-root/division arithmetic. Twenty-two unchanged
original-reference cases qualify 622 launches and 37,576,704 output words;
rounding gamma to BF16 changes 22,375 original values. The complete projected
native convolution/KDA/output-normalization chain matches 39,207,168 words
across eight full/split/one-token/prefill-then-decode schedules. Standard/full CUDA
and diagnostic suites pass, with zero memory-check errors in the normalizer and
mixed chain. CUDA traces show three native kernels on one stream with no recorded
activation copies between them. Projection GEMMs, indexed MLA, resident mHC,
model wiring and whole-model/performance acceptance remain open; the recurrent
path does not qualify optimized matrix-based chunk prefill or vLLM parity.
See [qualification](glm53-cuda-output-normalization.md). The full goal remains active.

### GLM architecture-owned main-model anatomy

Added explicit GLM tensor roles and a metadata/shape binding plan for all 45 main
blocks: 34 KDA, 11 indexed MLA, three dense FFNs followed by MoE, and distinct mHC
function/base/scale tensors. The plan binds all 1,383 main-model/global tensors
once and excludes only the 29 auxiliary MTP tensors. Per-layer SwiGLU clamp
arrays, latent versus projected MLA dimensions, and F32 indexer/router parameters
are preserved. Malformed and ambiguous layouts reject without mutating the plan.
The payload-free unit fixture and live six-shard C audit agree; sanitizer,
clean/standard/full CUDA and diagnostic checks pass. This is a binding-plan
milestone: the incompatible legacy GQA/GDN loader rejects the topology until a
dedicated binder/executor consumes it. Model execution, indexed MLA/mHC and full
correctness/performance acceptance remain open.
See [qualification](glm53-model-anatomy.md). The full goal remains active.


### Checkpoint-bound GLM KDA sublayer

Added transactional Q8_0/F32 weight preparation and complete resident KDA
attention execution from post-input-normalization hidden states through the
output projection. Actual checkpoint layers 0 and 44 match all 54,216,448
compared BF16/F32 words across six full/split/decode schedules. Budgets, shard
bounds, failure cleanup and view preflight have synthetic coverage; clean,
standard, full CUDA/diagnostic, ASan/UBSan and CUDA memcheck checks pass. The warm
trace records 14 kernels on one stream with no copies/allocations between
operations. Input/output GEMMs are now included; indexed MLA, mHC, full-model
binding, long contexts and measured vLLM parity remain open. Shared immutable
anatomy values moved to `model_anatomy.h`; architecture planning APIs retain
their existing ownership. See [qualification](glm53-cuda-kda-layer.md).
The full goal remains active.


### Resident mHC and materialized GLM attention transition

Added bounded backend-owned mHC weight preparation, SM120 split-K TF32/F32
coefficient kernels, BF16 collapse/input normalization and postmix. Actual
attention/FFN weights from blocks 0 and 44 pass 20 batch cases with 256,373,856
exact output/input comparisons. The resident mHC → normalized KDA → mHC
attention transition passes 56,428,288 exact comparisons across six prefill and
decode schedules, including state/history and residual output. The regular CUDA
suite has a model-free procedural reference with 30,296 exact comparisons and
canaries. Clean/standard/full CUDA, diagnostics, host ASan/UBSan and CUDA
memcheck/synccheck checks pass. The warm transition trace records 18 kernels on
one stream with no recorded copies or allocation calls inside the operation.
The ordinary GLM loader remains guarded: indexed MLA, FFNs, whole-model binding,
deferred/fused inter-layer mHC, long contexts and measured vLLM parity remain
open. See [qualification](glm53-cuda-mhc-attention.md). The full goal is active.


### Default deferred mHC/KDA boundaries

Implemented SM120's default unrounded-F32 fused residual projection for small
batches, its 4/5, 15/16 and 32/33 dispatch boundaries, and the materialized TF32
large-batch path. Native checkpoint replay passes 48,918,512 exact comparisons
across 36 cases; the default incoming-FFN-state → attention/KDA → next-FFN-input
chain passes 58,644,448 exact comparisons across six prefill/decode schedules.
The regular CUDA suite now covers all nine tested regimes with original-output
hashes, canaries and native incoming coefficient generation. Clean/standard/full
CUDA, diagnostics, host sanitizers and CUDA memory/synchronization checks pass.
The warm chain records 20 kernels, two fused projections, no recorded transfers
or allocation calls, and no inter-operation synchronization. This closes the
previous default-mHC arithmetic gap for the tested geometry; complete FFNs,
indexed MLA, full-model binding, CPU/long-context qualification and matched
vLLM parity remain open. See [qualification](glm53-cuda-deferred-mhc.md).
The full goal remains active.

### First three complete GLM dense decoder blocks

Added the resident clamped SwiGLU contract, bounded backend-owned Q8 dense FFN
weights and complete deferred mHC/KDA/dense-FFN block composition. Actual token
embedding through the first three decoder blocks passes 221,074,320 exact word
comparisons across cold and natively warmed chunk/decode schedules. Dense FFNs
separately pass 22,609,920 comparisons, including both matrix-dispatch thresholds
and batches 32, 33 and 65. The regular CUDA activation reference covers every
finite BF16 gate code at four clamp bounds, including non-BF16 bounds and signed
zeros. Clean/standard/full CUDA, diagnostics and host/CUDA sanitizers pass.
The warm three-block trace records 78 kernels, no allocation/activation/weight
transfers, a four-byte final status read, and no inter-operation synchronization.
The ordinary loader remains guarded. Indexed MLA, routed/shared MoE execution
and offload binding, whole-model execution, long contexts and matched vLLM
performance remain open. See [qualification](glm53-cuda-dense-prefix.md).
The full goal remains active.

### GLM routed execution through bounded expert banks

Generalized resident grouping, ordered reduction and bounded streaming banks for
288 experts/top-eight routes. Per-submission clamped activation preserves
per-layer metadata without duplicating bank pools; Q4_K gate/up plus actual
Q5_K down weights now execute through the tiled grouped path. The actual-weight
supplied-route replay passes 165,888,000 exact comparisons across vector/matrix
thresholds, three bank capacities and both per-expert/grouped execution modes.
Qwen bank/cache regressions, standard/full CUDA/diagnostics, dense-prefix replay
and CUDA memcheck/synccheck pass. A warm bank trace records 16 kernels, exact
weight/descriptor transfer accounting, event dependencies and no activation
transfers or allocation APIs. Router decisions, shared experts, GLM RAM/VRAM-cache
policy, indexed MLA and whole-model execution remain open. See
[qualification](glm53-cuda-routed-execution.md). The full goal remains active.


### GLM complete MoE with actual routing and RAM expert caching

Added source-F32 routing projection, biased sigmoid top-eight selection,
shared-expert preparation and complete resident MoE composition with bounded
expert-bank streaming. The reference explicitly composes original operators:
the installed whole-plugin method omits GLM clamping, and its matrix kernel
rejects expert IDs above 255. Expert-domain partitioning preserves original
token geometry and leaves all supported-domain reference words unchanged.
Actual mmap/pread execution, cold/warm decode and prefills through 1,025 tokens
pass 527,733,360 exact comparisons. Warm decode avoids CPU source reads and
still uploads routed weights. See [qualification](glm53-cuda-moe-layer.md).
GLM VRAM caching, indexed MLA/indexer, whole-model binding, CPU execution,
long-context generation and matched resident vLLM performance remain open.
The full goal remains active.

### GLM FP8 indexer transforms and gated pool state

Added resident Hadamard-128/E4M3FN query conversion, ordered F32 head-weight
scaling, and four-token gated key pooling with explicit initial/next tail
state. I8 provides raw-byte resident storage without model-weight execution
capabilities. Independent unchanged GLM operators cover large query batches,
all pool offsets, resumed state, and whole versus token/chunk submissions.
The core is part of the serial CUDA regression chain. See
[qualification](glm53-cuda-indexer-core.md). Projection/LayerNorm resource
binding, paged cache addressing, score/top-k/tail expansion and latent attention
remain open before complete GLM indexer/MLA or whole-model admission. The full
modern-model and matched vLLM performance/serving goals remain active.


### GLM checkpoint pre-indexer and F32 LayerNorm

Bound the actual Q8 query/key/gate and source-F32 head/norm/bias/pool tensors
into an immutable backend resource and composed the resident pre-indexer from
hidden states plus the MLA caller's normalized q-rank input. LayerNorm preserves
the original single-row and batched Welford layouts and affine FMA boundaries.
Actual cold/chunked/resumed schedules through 1,025 tokens pass 22,659,076 direct
output-element comparisons; model-free LayerNorm cases cover 34,190,848 values.
The raw-GGUF head profile is explicit and does not imply default-HF merged-BF16
or whole-vLLM equivalence. See [qualification](glm53-cuda-preindexer.md).
Paged side-cache addressing, score/selection/expansion, latent MLA attention,
whole-model execution and the full backend/performance/serving goal remain open.


### GLM paged FP8 indexer cache

Added exact planar F32-scale/FP8 cache append and page-table gather, and a
resident pre-indexer/cache composition with global metadata/alias preflight.
Model-free original-writer/gather captures cover 52,953,486 output elements;
actual checkpoint cold/chunked/resumed chains cover 24,393,382 direct elements.
The native chain computes its own pool/tail history. Invalid page status,
untouched bytes, physical slot zero and raw scale bits have explicit contracts.
See [qualification](glm53-cuda-fp8-cache.md). Signed-weight FP8 scoring,
causal masking, selection/expansion and MLA/whole-model execution remain open;
resident vLLM performance and serving acceptance remain part of the full goal.

### GLM signed FP8 causal score primitive

Added a resident SM120 FP8 MMA scorer preserving signed folded head weights,
per-key F32 scaling and original head reduction order. Causality is enforced
per query row and future scores are negative infinity. Independent captures
cover 350,090 synthetic and 5,506,816 actual-checkpoint F32 score elements,
including 2,049/4,097-token prefixes beyond the short-context bypass. Standard
and CUDA suites, device memcheck/synccheck and the single-kernel resident trace
passed. See [qualification](glm53-cuda-fp8-score.md). Native projection/cache/
scorer composition, paged decode, pool selection/expansion, MLA, whole-model
execution and the full backend/performance/serving requirements remain open.
Original top-k order variation is preserved in both raw captures; selected
membership was stable in the observed repeats.

### GLM native pre-indexer/cache/score composition

Composed native projections, FP8 conversion, signed head weights, pooled tail
state, paged cache append/gather and per-query causal scoring from hidden states
and normalized QR. Global score metadata/alias preflight precedes all eleven
submissions. Nine independent original-reference schedules cover 23 phases,
including 2,049/4,097-token prefixes, chunked/resumed histories and subsequent
single-token calls. Native history is retained between phases; 146,404,112
pre-indexer/cache elements and 12,074,513 scores match exactly. Standard/CUDA
suites, dense/MoE/cache regressions, host/device sanitizers and the 17-kernel
resident trace passed. See [qualification](glm53-cuda-indexer-scores.md).
Pool selection/expansion, direct paged decode, MLA, whole-model generation,
expert placement and the full backend/performance/serving goal remain open.

### GLM pool selection and expansion

Added signed F32 top512 pool selection and 2,051-column token expansion,
including the original short-prefill contiguous packing and score bypass.
Native ordering/tie policy is explicit and deterministic. Actual replay checks
exact untied pool/token membership for 14,356 sparse queries and byte-identical
token lists for 2,060 short queries; original raw order is preserved in both
reference captures. Independent CPU full-sort tests check native ties,
negative/subnormal scores and invalid-score status. Standard/CUDA suites,
host/device sanitizers, historical score replay and warm sparse/short traces
passed. See [qualification](glm53-cuda-selection.md). Downstream MLA/order/tie
numerics and generation remain unqualified; GLM admission remains guarded.
Direct paged decode, expert placement, other backends/models, serving and
matched vLLM performance requirements remain part of the full goal.

### GLM direct paged decode indexer

Added direct planar FP8/F32 cache scoring and native pre-indexer/append/score/
select composition without gathered history. Shared FP8 MMA arithmetic keeps
the qualified contiguous scorer intact. Original paged/nonpaged captures match
1,134,288 fixture values; ten original actual-checkpoint decode phases also
match preserved scores exactly. Native replay with gather and contiguous-score
callbacks disabled matches 145,015,814 upstream elements and 11,548,687 scores,
plus prior selection membership/packing checks. Standard/CUDA suites, host/
device sanitizers, historical prefill regressions and the no-gather resident
single-token trace passed. See [qualification](glm53-cuda-paged-score.md).
MLA projection/latent attention/output, order/tie numerical qualification,
whole-model admission/generation, expert placement, other backend/model scope,
serving and matched vLLM parity remain open.

### GLM NoPE MLA projections

Added immutable MLA raw/derived weight preparation, per-head batched BF16
linear projection, x-to-normalized-QR/KV/query preparation and latent-context
value/output projection. Explicit identity materialization preserves Q8_1
basis-scale rounding; canonical decode is unchanged. A vector-eight RMSNorm
mode preserves original reductions and correctly rounded mean division.
Original captures match 16,777,216 prepared-weight and 97,202,176 projection/
input BF16 values with matched reference cuBLAS; all 83 files reproduce across
independent captures. Standard/CUDA suites, host/device sanitizers and resident
projection traces passed. BLAS-version rounding differences are documented,
not hidden by replacing goldens. See [qualification](glm53-cuda-mla-projection.md).
Latent KV cache/sparse attention, selection order/tie numerics, integrated MLA
and whole-model generation/admission remain open, along with the full expert
placement, backend/model, serving and matched vLLM performance goal.
