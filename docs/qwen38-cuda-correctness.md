# Qwen3.8 27B CUDA correctness qualification

The resident BF16 path supports an explicit, request-local arithmetic and
normalization plan for layer inputs, post-attention outputs, and
the final hidden output. The plan is copied at request
and batch-coordinator creation, survives restart, and must agree across a decode
cohort. Unsupported reductions fail admission before allocation or execution.
The normal runtime defaults are unchanged.

## Current performance checkpoint

Three fully resident, four-request CUDA SM120 geometries meet both performance
floors. Each request produces 64 output tokens; sequence capacity is 8192.
Both engines use the same GGUF checkpoint, BF16 activations, F32 recurrent
state, and disabled CUDA graphs. These are three-run medians with the explicit
batch-aware attention policy enabled.

| Input tokens per request | Native prefill | Reference prefill | Throughput ratio | Native decode batch | Reference decode batch | Decode difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 129 | 585.410 ms | 1327.594 ms | 2.268x | 37.353 ms | 37.698 ms | -0.9% |
| 1025 | 4362.057 ms | 10290.435 ms | 2.359x | 38.811 ms | 38.688 ms | +0.3% |
| 2048 | 9010.810 ms | 20622.066 ms | 2.289x | 39.842 ms | 39.155 ms | +1.8% |

Every geometry matches all 256 complete independently captured BF16 logit
vectors, and every timing run matches all 256 output tokens. Legacy singleton
attention replay also remains exact. The protected clean build has zero
warnings; standard tests, the complete CUDA suite, diagnostic isolation,
focused attention memory checks, and real-model runtime tests pass. Existing
independent oracles are unchanged.

The quant kernels reuse each packed weight load for both nibble halves in
Q4_K/Q5_K/Q6_K tiles. Integer dots, scale values, and floating-point reduction
order are unchanged. Q6_K retains halfword loads for its two-byte row alignment.
Smaller affine launches retain an unbounded entry point; measured large SM120
grids use a separate six-block entry point.

At the 2049-key decode boundary, the previous singleton occupancy heuristic
selected 33 attention splits, while the original four-request execution selected
17. This changed the reduction tree and failed the full-vector comparison.
The explicit `attention_split_policy: 1` now selects splits using cohort size
and maximum key length, while each row retains its own actual KV bounds.
Policy zero preserves the legacy singleton behavior and its existing JSON
representation. Batch policy admission requires the backend capability; CUDA
currently advertises it on SM120. Equal head geometry and sufficient participant
scratch are validated before any grouped work is queued. BF16 batch-policy
requests reserve 128 split slots without increasing their KV capacity.

The new independent primitive fixture covers batch sizes 1, 2, 4, and 8,
uniform and ragged contexts through 4097 keys, and automatic splitting in the
unmodified reference FlashAttention entry point. Its hardware-specific
partition provenance is recorded in
[qwen38-attention-batch-reference.json](qwen38-attention-batch-reference.json).
Tests also check insufficient-scratch rejection, append bounds, and policy
copy, restart, and cohort agreement.

A separately captured single-request case with 129 input tokens also passes.
Its native three-run medians are 188.592 ms prefill and 16.349 ms decode,
versus 361.165 ms and 20.837 ms in the matched reference: 1.915x prefill
throughput and 21.5% lower decode latency. All 64 complete BF16 logit vectors
and all timing-run tokens match. The unchanged default four-request replay
also remains exact after extending the diagnostic tools to 1–16 requests.

The singleton reference selects context-parallel recurrent prefill, unlike
the four-request reference. The pinned wrapper's pure host threshold and a
separate GPU trace both confirm this choice; the trace contains all four CP
stages for each of the 48 recurrent layers. The singleton execution therefore
uses the existing explicit CP arithmetic policy with the independently derived
normalization plan. The original failed NONCP replay is preserved and excluded
from performance results. No default arithmetic selection changed. Evidence:
`/tmp/bitnet-goal-packed-b1` and `/tmp/bitnet-goal-packed-b1-profile`. Profile
measurements are excluded from the reported medians. The copied reference
program's hardcoded request-count label is corrected in derived timing metadata,
with the original samples and an explicit erratum retained.

A 16-request case with 129 input tokens and 64 outputs per request also passes:
2312.652 ms native prefill versus 5241.893 ms reference (2.267x throughput),
and 95.484 ms native decode batches versus 100.604 ms reference (5.1% lower
latency). All 1024 complete BF16 logit vectors and all timing-run tokens match.
The reference needs an 8 GiB cache budget to admit the complete cohort; its
original 2 GiB attempt admitted only ten requests and was rejected by the
capture's strict shape check. The successful capture proves 2064 initial
prefill tokens and 16 rows on every subsequent decode call. This measures
short contexts at concurrency 16, rather than sixteen simultaneous 8K contexts.
Evidence: `/tmp/bitnet-goal-packed-b16-cache8`; the excluded setup is retained
in `/tmp/bitnet-goal-packed-b16`.

A chunked four-request case with 4097 input tokens per request also passes.
Each prompt is submitted as 2048 + 2048 + 1 tokens, followed by 64 outputs.
The reference trace proves prefill cohorts of 8192, 8192, and 4 tokens, then
63 four-row decode calls. Native median prefill is 18224.502 ms versus
41732.391 ms reference (2.290x throughput); native decode batches are
40.470 ms versus 39.709 ms (1.9% higher latency). All 256 complete BF16 output
vectors and all three timing streams match.

This case exposed one-token prefill tails using singleton attention partitions.
Both preceding chunks' complete intermediate logits match independently, and
observing those heads leaves subsequent native logits unchanged. All-one-token
packed tails now use the existing batch-aware attention group when that policy
is explicitly selected. They retain prefill recurrent arithmetic, position
transactions, and prefill accounting. Longer or mixed-length prefill cohorts
retain their prefill attention path. Runtime tests cover that distinction,
head-suppressed logit preservation, and allocation-free tail submission. Full
CUDA, diagnostic isolation, attention memory checks, real-model runtime, and
legacy replay pass after the change. Evidence:
`/tmp/bitnet-goal-packed-chunk4097-rr/tail-promotion`.

Independent reference metadata and explicit policies for the added cases are
preserved in [qwen38-cuda-matched-cases.json](qwen38-cuda-matched-cases.json),
including hashes and provenance. Existing reference fixtures remain unchanged.
The replay diagnostic accepts `[REQUESTS [PREFILL_CHUNK]]` after its original
four positional arguments; omitted arguments retain four requests and one
prefill cohort. The chunk-aware verifier reads geometry from its supplied
independent reference and checks token budgets and call counts.

A bounded admission/refill case submits twenty 129-token requests with at most
sixteen live, then admits the remaining four after the first wave completes.
The independent trace proves 128 model calls, with 2064 and 516 prefill tokens
at calls 0 and 64. Every one of the 1280 complete BF16 output vectors matches,
covering 317,849,600 words; all three uninstrumented timing streams also match.
Timestamp checks independently establish a peak of sixteen live requests and
completion before refill. Three-run medians are:

| Metric | Native | Matched reference |
| --- | ---: | ---: |
| Prefill across both waves | 2910.326 ms | 6612.085 ms |
| Decode engine step, 63 calls at each cohort size | 67.231 ms | 69.583 ms |
| End-to-end output throughput | 111.155 tokens/s | 83.185 tokens/s |
| p95 time to first token, including admission wait | 9058.633 ms | 12946.963 ms |
| p99 inter-token latency | 93.932 ms | 101.021 ms |

Tail entries are medians of each run's nearest-rank percentiles, with twenty
TTFT samples and 1260 inter-token samples per run. All requests arrive as one
burst. The reference uses an 8 GiB cache budget and disabled CUDA graphs.
An initial benchmark result exposed final-token publication interleaved with
session reclamation: the thirteen worst intervals in every run occurred on
output 63. Publishing the complete ready cohort before freeing completed
sessions reduced median p99 from 114.779 ms to 93.932 ms. Reclamation remains
inside the timed decode phase and total elapsed time.

`bench_bf16_batch --dump-logits DIRECTORY` exports already-read BF16 heads only
for diagnostic runs with an external token reference. Its output explicitly
marks disk I/O; both timing verifiers reject such runs as performance evidence.
`test/verify_bf16_serving_reference.py` checks every matrix and request/step row
hash. `bench/verify_bf16_serving_batch.py --correctness-only` checks diagnostic
stream and admission evidence without reporting timings. Normal serving
verification also checks live counts, refill ordering, budgets, throughput,
and delivery timestamps. Twelve model-free negative/positive checks run in
`make test`. Evidence: `/tmp/bitnet-goal-serving-20x16/delivery-after-cohort`;
the preceding publication-order result remains preserved in its parent.

These results qualify one singleton, three uniform four-request geometries,
one uniform sixteen-request geometry, one chunked four-request geometry, and
one bounded uniform admission/refill case. Capacity 8192 does not mean each
request has an 8192-token measured context. Mixed or staggered serving,
production scheduling, byte-accounted KV pools, and expert offloading still
need qualification. Native phase wall times include host sampling and
request teardown where applicable; reference times use engine-step wall times.
Profile timings are excluded. Evidence is in
`/tmp/bitnet-goal-attention-batch-splits`,
`/tmp/bitnet-goal-packed-2048`, `/tmp/bitnet-goal-packed-1025`, and
`/tmp/bitnet-goal-packed-parity`. These results do not compare against other
vLLM checkpoint formats or graph-enabled configurations.

## Qualified reference and checkpoint

The current reference uses original vLLM compiled execution with FlashInfer
prefill and Triton GDN decode. The checkpoint is
`Qwen3.8-27B-UD-Q4_K_XL.gguf`, SHA-256
`3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
The pinned reference image is
`sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`.
Reference observations preserve the control run’s raw logits.

The compiled reference selects mixed persistent and looped RMS reductions by
layer. The looped 5120-wide two-input reduction uses 1024-column tiles and
four accumulators: square tile 1, FMA tile 0, then FMA tiles 2, 3, and 4.
Reversing the initial FMA matters even when most BF16 outputs agree.
Plain and three-input normalization retain their existing arithmetic.

For this captured reference compilation, the looped-layer mask is
`fb8db8daab89f9e9`. It was selected by comparing every normalization output on
original operands, not by fitting generated tokens. All 64 layers match;
5,248,000 BF16 values were checked per layer. The mask is an explicit benchmark
input, not an automatic model or prompt heuristic. A different reference
compilation needs its own qualification.

## Whole-model evidence

Starting from zero state, production prefill of 1,025 tokens followed by one
decode step matches all 146,155,520 compared values exactly: logits, hidden
output, F32 recurrent state, convolution history, and attention KV state.
No reference state was imported into the native execution.

The same explicit plan matches all 64 generated tokens for each of the short,
medium, long, and 4,097-token repeat prompts. The frozen operator fixture
includes original layer-0 edge rows and a layer-7 row that distinguishes the
initial FMA order. Its extractor verifies captured tensor hashes and reconstructs
residual inputs from independently captured block outputs.

## Matched 256-token chunk schedule

A separate reference with scheduler budget 256 records prefill chunks
`[256, 256, 256, 256, 1]` and the first decode at position 1025. This compilation
selects different normalization reductions, including for three-input residual
norms. The original unchunked plan therefore does not qualify this configuration.

The new persistent and looped contracts cover one-, two-, and three-input
normalization. Persistent three-input norms round each square before reduction;
the original compiled instructions contain no square FMA contractions. Both
contracts retain fused mean scaling/epsilon. The selected plan is:

| Scope | Persistent baseline | Looped mask |
|---|---|---|
| Initial layer input | Persistent | None |
| Post-attention layers 0..63 | Persistent | `1403407550526451` |
| Layer inputs 1..63 and final norm (index64) | Persistent | `0403407550526450` (bit `index-1`) |

All 169,082,880 normalization values in the captured first 256-token chunk
match the chosen reductions on original operands. Both reductions agree for
post-attention layer57 on that calibration input; persistent was selected.
The plan is an explicit arithmetic choice qualified on the workloads below,
not a promise to recover arbitrary compiler/autotuner choices.

Starting from zero state, the ordinary device-resident production path matches
all 146,155,520 compared values after the five prefill chunks and first decode.
No reference state or activation is imported into this execution. All four
64-token streams also match the chunked reference. The prior unchunked plan
retains its 146,155,520-value exact match after these changes.

Independent operator fixtures cover 83,763,200 normalization values plus
residual outputs. Their extractor launches original selected cubins and excludes
in-place kernels with ambiguous pointer counts. Fixture reproduction is
byte-identical. Separate recurrent-operator replays match 10,229,760 output/state
values, including four nonzero-state continuation calls.

The immutable API plan uses `input_norm_reductions[0..64]` and
`post_norm_reductions[0..63]`. `DEFAULT` entries inherit the arithmetic profile.
For this chunked configuration, initialize the initial input with
`BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT`; select that reduction or
`BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED` for each remaining scope using the
masks above. Pass the plan to `bn_transformer_bf16_request_init_with_policy`
with arithmetic `CP_FP32_GATED_R2`, maximum chunk256, and capacity8192.
Request and batch-coordinator admission require backend capabilities, copy the
plan, and reject incompatible cohorts before execution. Restart preserves it.

## Prefill resumed after decode

The original vLLM public streaming-input API provides an independent session
continuation reference. The qualified schedule is initial prefill129, seven
decode steps, resumed prefill64 at position136, and seven more decode steps.
Each phase samples eight outputs; the uncomputed last sample of the first phase
is discarded before appending new input, matching the reference API semantics.
Fresh input lists keep control and observed requests identical. Actual forward
metadata verifies the schedule and nonzero incoming state on resumed prefill.
All sixteen raw logits are identical between control and observed runs.

The ordinary native API follows the same schedule from zero state. All
92,088,320 compared values match around the first post-resume decode at
position200: logits, final hidden output, F32 recurrent state, convolution
history, and attention KV state. Both eight-token phases match. No reference
state or activation is imported. Compute Sanitizer reports zero errors.

`test/bf16_resume_reference.json` freezes only independently captured reference
hashes, shapes, input IDs, output tokens, checkpoint identity, and scope.
`extract_bf16_resume_reference.py` verifies source capture hashes and alignment;
it never reads native expected values. The comparison tool checks the actual
checkpoint hash, input IDs, phase order, and all259 payloads with exact sizes
and hashes. Missing coverage, corruption, truncation, and misalignment fail.
Fifteen synthetic protocol checks run in the standard `make test` suite.

To validate a saved native dump and its input/phase JSON log:

```sh
python3 test/compare_bf16_resume_reference.py \
  --model MODEL.gguf --dumps DUMP_DIR --log NATIVE_LOG
```

This command compares existing dumps; it does not execute inference. This
qualifies the stated resumed schedule with the explicit budget256 plan.
Independent multi-request execution, other checkpoint formats, and vLLM's
default FlashInfer/CUDA backend combination remain separate gates.

Against the separate default FlashInfer/CUDA reference, the medium and long
streams still diverge (first differences at generated tokens 15 and 50,
zero-based).

These results also do not establish the goal of performance within 5% of vLLM. Keep those
gates separate from this matched-backend correctness evidence.

Run the **unchunked** explicit plan with:

```sh
./bench_bf16_latency MODEL.gguf PROMPT.i32 64 1 0 8192 8192 \
  cp-fp32-gated-r2 fb8db8daab89f9e9
```

### Explicit persistent fused-square normalization

`BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES` adds an
explicit fixed-width reduction contract. It uses the persistent 8192 layout
and fused square pairs for three inputs, as well as the existing persistent
one/two-input arithmetic. It requires dim 5120, shared gamma, offset 1, and
eps 1e-6. Unsupported plans reject; default arithmetic is unchanged.

The original stock final-normalization cubin has an in-place three-input
signature. It is distinct from ordinary two-input normalization despite their
matching pointer counts. The independent extractor pins that cubin's SHA-256,
maps hidden/block/residual operands explicitly, and obtains normalized output
from the original kernel. Scalar CPU F32 additions with one final BF16 rounding
supply residual sums. The native API continues to require disjoint outputs.
The existing general extractor still excludes in-place kernels.

Forty cases cover eight seeds and 1/5/17/129/1025 rows: 48,209,920 independently
captured normalization values. The expanded general CUDA test also exercises
one/two-input equivalence, invalid plans, output guards, input preservation, and
residual hashes. All 165,478,400 normalization values match; operator memcheck
reports zero errors. A complete 1025-token prefill followed by decode also
passes memcheck with zero errors. Runtime tests check capability rejection before allocation,
policy copying, restart, and cohort isolation with the new contract.

A candidate policy derived from stock Nsight launch signatures and cached
cubins has also been tested through ordinary zero-state production execution.
It does **not** pass the complete model comparison against the separate
FlashInfer/CUDA graphs-disabled capture: the first recurrent-state difference
is at layer 16. This is not an accepted stock execution policy or a performance
parity result. Full intermediate captures are available for localization.

Stock control streams also differ between archived, fresh uninstrumented, and
profiled executions, despite matching recorded effective configuration and
extension hashes. Repetitions within each run agree. The cause remains
unproven; profiling, sampling ties, or a race has not been established as the
explanation. Fresh input IDs are recorded, but the archived benchmark did not
retain input-content hashes. Keep stock reproducibility separate from the
qualified explicit FlashInfer/Triton plans above.

The first intermediate difference is earlier than the recurrent-state mismatch:
two BF16 words in layer 1's FFN output, with identical FFN input. Read-only stage
observation preserves all 259 native payloads. Independent replay of the exact
checkpoint's gate/up projections matches all 17,843,200 values in each output.
The original compiled F32 SiLU cubin differs from the native activation in two
words. An SM120 prototype reproduces those two differences with `div.rn.f32`;
`div.full.f32`, as used by the original PTX, matches all 17,843,200 activation
values. A default-SM52 build hides this distinction, so future prototype and
production qualification must use the actual GPU compilation target. This is
a localized next arithmetic contract; it has not yet been admitted to the
production FFN policy or qualified through the whole model. The eager BF16
SiLU custom op has different rounding and cannot replace the compiled oracle.

### Compiled F32 FFN activation policy

`BN_GPU_BF16_FFN_SILU_COMPILED` now provides the original compiled SiLU
contract: libdevice exponential, F32 addition, PTX `div.full.f32`, F32 product,
and one final BF16 rounding. Existing `SEPARATE` and `FUSED` retain their
arithmetic. The backend advertises supported activation profiles explicitly.

`BnTransformerBF16Policy.ffn_arithmetic` selects the contract for a dense hybrid
request or decode coordinator. DEFAULT resolves to the existing base behavior;
retained-intermediate requests report canonical FUSED_RN. Explicit choices
require retained F32 intermediates and backend capability admission before
allocation. Requests copy the policy, restart preserves it, and decode cohorts
reject mismatched activation arithmetic before queueing work. Flash model
activation defaults are unchanged.

The independent compiled-cubin fixture covers all 65,536 BF16 gate encodings
with eight finite up values (524,288 combinations, canonical NaN payloads), plus
the two full-model edge pairs. It regenerates byte-for-byte from the original
cubin. Tests use the actual SM120 compilation target. A guarded complete
resident FFN checks another 296,960 outputs, input preservation, intermediate
values, rejected plans, and output boundaries. Existing activation fixtures
remain unchanged.

The matched FlashInfer/CUDA graphs-disabled budget 8192 plan is now:

- Arithmetic CP_FP32 (FLA gated norm), FFN COMPILED.
- Initial and ordinary input/post norms: TILED_5120_FMA (2).
- Input norms at layers 21/59/61: PERSISTENT (4); final input 64: PERSISTENT_FUSED_SQUARES (6).
- Post norms at layers 58/60: LOOPED (5).

This plan comes from independently replaying all 129 captured normalization
vectors (676,992,000 values) with checkpoint gamma and reconstructed retained
residuals. Some reduction choices have identical outputs on these operands;
matching a vector does not recover an arbitrary compiler's autotuner policy.
This is a matched explicit plan, distinct from the earlier stock-trace candidate.

Ordinary native execution from zero state now matches all 146,155,520 payload
values around the first decode after 1025 prompt tokens. All 257 full intermediate
stages match (1,348,736,000 BF16 values). Read-only observation preserves all 259
native payloads. The short, medium, and 1025-token cases match all 192 generated
tokens and all 192 complete raw-logit vectors. A matched whole-model sanitizer
run reports zero errors. Clean build, standard and complete CUDA suites,
diagnostic checks, and real-model runtime policy checks pass.

With this original 1025-token reference's plan, the 4097-token repeated prompt
matches all 64 generated tokens but differs in all 64 raw-logit vectors. The
separate 4097-token reference qualification below resolves that case for its
matching explicit plan. These results do
not resolve stock archived/fresh/profiled reproducibility, arbitrary compiler
bounds, concurrent execution, or the performance target. Defaults remain unchanged.

`test/bf16_cuda_reference.json` freezes only original reference hashes, shapes,
input IDs, tokens, engine configuration, and checkpoint identity: 259 canonical
state/KV/logit payloads, 257 intermediate stages, and 256 stream-logit vectors.
`test/extract_bf16_cuda_reference.py` validates original capture provenance and
canonicalizes original paged KV into the native comparison layout. It never
reads native expected values. This is reference extraction, not an inference
runner or a claim that every frozen stream already matches.

### Independent 4097-token reference qualification

A fresh original reference capture covers zero-state prefill at positions
0–4096 and the first decode at position 4097. Its own control and observed
streams match all 256 raw-logit vectors across short, medium, long, and repeated
4097-token prompts. The selected decode payload is stream index 193. Standard
attention uses FlashAttentionImpl (FA2); the FlashInfer/CUDA settings describe
GDN prefill and decode. CUDA graphs are disabled for this correctness capture.

This reference differs numerically from the frozen cold-compiled reference
despite identical recorded engine configuration, model, and AOT cache key.
It is preserved separately in `test/bf16_cuda4097_reference.json`; the original
`test/bf16_cuda_reference.json` remains byte-identical. The extractor accepts
`--case repeat4097`, checks the anchor and capture validation, and records
provenance. Default extraction still reproduces the original JSON exactly.
Wrong and unknown anchors are rejected.

The matching explicit native policy uses CP_FP32 and FFN COMPILED, with input
and post reductions set to 2 except input layers 24/28/38/61 set to 4, final
input 64 set to 6, and post layers 58/60 set to 5. Independent replay finds a
matching existing reduction contract for all 129 normalization vectors
(2,705,986,560 values). This is an operand-qualified plan, not a claim to recover
every physical compiler selection.

Ordinary native execution from zero state matches all 347,482,112 values in
259 state/KV/logit payloads and all 5,390,996,480 BF16 values in 257 complete
prefill stages. Read-only observation preserves all 259 native payload hashes.
The same explicit plan matches all 256 generated tokens and all 256 complete
raw-logit vectors across the four cases. Whole-model CUDA memcheck reports zero
errors. A protected clean build has zero warnings and the standard tests pass;
the four preserved reference binaries retain their hashes.

A further original-only Nsight capture maps all 258 observed normalization
launches to cached cubins and reduction contracts. Its first difference from
the fresh 4097-token reference is one BF16 word at layer 9's input norm; all
36 preceding full stages agree. Replaying that original traced cubin on the
same preceding reference operands reproduces the traced output exactly and
the single differing word against the other reference. This establishes a
concrete reduction-choice cause for that first divergence. Cache loading alone
does not explain all cross-run differences, and other causes remain unproven.

The evidence is under `/tmp/bitnet-goal-cuda4097`, including comparison reports,
the full explicit policy, original launch/cubin mappings, replay inputs, and
sanitizer output. These diagnostic captures do not measure performance parity.
Defaults remain unchanged. Concurrent and multi-step batched execution,
reproducible optimized-stock comparisons, serving, bounded expert offloading,
and the within-5% vLLM performance target remain open.

### Multi-request reference admission gate

The multi-process offline reference API does not guarantee that every supplied
prompt reaches the scheduler before its first step. A four-prompt candidate
capture produced first-logit shapes `[1,248320]` in control and `[2,248320]`
in observation; their streams differed and neither supplied the required
all-four first decode at position 129. That candidate is rejected. This is
a demonstrated schedule mismatch, not evidence of same-shape arithmetic
instability caused by observing tensors. Batch qualification must record
actual prefill/decode shapes and positions as well as final token streams.

The native batch diagnostic now accepts the same complete immutable policy
file as the latency harness and passes it to all requests and coordinators.
Two-request uniform and mixed/chunked cases pass their native isolated-token
checks under the explicit 4097-reference plan; four-request execution does
not. Independent multi-step batched acceptance remains open. Evidence lives
under `/tmp/bitnet-goal-batch-qualification`.

With `VLLM_ENABLE_V1_MULTIPROCESSING=0`, the pinned original engine queues
all four offline requests before stepping. The trace confirms one zero-state
prefill of 516 tokens (four 129-token sequences), followed by four-row decode
calls at positions 129–191. Control and observation match all 256 generated
tokens and all 256 raw-logit vectors. The first decode contributes 257 full
intermediate stages (5,263,360 BF16 values), plus original before/after
recurrent and attention-cache captures.

`test/extract_bf16_cuda_batch_reference.py` verifies the original capture,
per-row greedy outputs, hashes, shapes, zero-state flags, and schedule before
writing `test/bf16_cuda_batch4_reference.json`. The frozen data contains only
original reference values, never native expected outputs. Raw attention block
captures include storage outside the valid prefix; consumers must use the
recorded block tables and positions before comparing canonical KV. This is
a qualified independent oracle, not a passing native four-request gate or
a performance parity result.

### Four-request first-block localization

The first decode at position 129 now has a separate original block-0 replay.
Its inputs are the four captured original input-norm rows, original before-state
and convolution histories, and checkpoint projection/convolution/norm weights.
The original replay matches the full-model block output, next recurrent state,
and next history byte-for-byte. Native grouped execution then matches every
checked projection, convolution, recurrence, gated norm, head-layout, output,
state, and history value: 3,535,232 values. CUDA memcheck reports zero errors.
`test/bf16_cuda_batch4_block0_reference.json` freezes only original hashes and
provenance; evidence and replay sources are in `/tmp/bitnet-goal-b4-gdn-replay`.

This clears the captured first-block arithmetic subgraph. Importing original
before-state is a diagnostic and does not qualify native zero-state packed
prefill or subsequent batched decoding. The coordinator still supports one
decode row per request only. CP prefill's resident-view resolver deliberately
rejects views owned by another request; multi-sequence prefill needs explicit
sequence boundaries and state ownership, rather than passing participant
state through the existing single-request plan. Existing ownership checks
and production arithmetic remain unchanged.

### Explicit grouped CP-prefill state ownership

The CUDA backend now exposes `BnGPUGDNCPPrefillGroupPlan`: each sequence
names its state-owning resident request and an existing CP-prefill plan.
Q/K/V, decay, beta, and output views must belong to the coordinator; initial
and next states must belong to the named request. Participants must be distinct
and share the coordinator's backend/device. The existing single-request
CP-prefill operator still rejects foreign state.

The group validates every sequence and all cross-sequence write overlaps
before queueing work. A separately reserved event table orders participant
streams before coordinator execution and orders participant reuse afterward.
Execution uses the coordinator's existing bounded CP scratch, sized for the
largest individual sequence, and queues the independent sequence kernels in
order. It adds no host readback, host synchronization, or descriptor upload
on the successful execution path. Partial execution failure poisons the
coordinator and every participant. This is an ownership/ordering primitive,
not a claim that recurrence kernels execute concurrently.

New guarded tests reject malformed later rows, repeated state owners,
cross-row output aliasing, and in-place state, checking that all output/state
buffers remain unchanged. A group containing a 129-token sequence and a
one-token sequence matches each sequence's unchanged frozen original
FlashInfer output and F32 state hashes, including unused output tails and
input preservation. Two submissions are queued before readback to exercise
event reuse. The full CUDA suite and diagnostic isolation pass; the final
mixed-length focused suite checks 554,127,360 existing/new output and state
words, and CUDA memcheck reports zero errors. Protected clean build has zero
warnings, standard tests pass, and real-model runtime policy checks pass.
Evidence: `/tmp/bitnet-goal-cp-prefill-group`.

The transformer coordinator does not yet consume this primitive. Packed
convolution and attention boundaries, shared multi-sequence projections,
last-row logits gathering, and atomic position commits remain required
before native zero-state packed-prefill qualification. No default arithmetic
or performance-parity claim changes.

### Variable-length grouped convolution prefill

`BnGPUCausalConvPrefillGroupPlan` adds an explicit positive token count per
sequence to the existing convolution group plan. Its optional CUDA callback
uses the existing bounded convolution descriptor/event workspace and BUSY
contract. Histories stay with each distinct participant; activation slices
can belong to the coordinator. The complete group, view sizes, token counts,
and cross-sequence aliases are validated before queueing any work.

The legacy callback retains one-token semantics. Groups consisting entirely
of one-token rows use its existing fused decode kernel. Longer/mixed groups
queue the existing multi-token convolution kernel per sequence, with the
existing participant/coordinator event ordering and failure poisoning. The
shared bounded descriptor upload remains; this is not a zero-transfer or
concurrent-convolution-kernel claim. Arithmetic and weights are unchanged.

Guarded tests compare mixed 17/1-token sequences and 65537/1-token boundary
cases against unchanged independent original output/history hashes. They
cover zero/nonzero histories, kernels 2–4, separate sequence boundaries,
unchanged inputs, unused output tails, all-one fused execution, malformed
later rows, invalid/overflow counts, aliases, and repeated state owners. Two
submissions are queued before readback. The focused gate passes 1,996,744
new output/history values alongside existing decode-group checks.

Evidence is under `/tmp/bitnet-goal-conv-prefill-group`. Transformer packed
prefill is still not integrated: attention-prefill KV ownership, packed row
capacity, shared projections, last-row logits gathering, and atomic position
commit remain required.

The protected clean build has zero warnings and preserves all four reference
binary hashes. Standard tests, the complete CUDA backend suite and diagnostic
isolation, the grouped-convolution CUDA memcheck (zero errors), and real-model
runtime policy tests pass after this change.

### Attention prefill groups with request-owned KV

`BnGPUAttentionPrefillGroupPlan` and the optional CUDA
`attention_prefill_group_bf16_resident` callback accept one or more causal
tokens per sequence, including resumed prefill chunks. Each distinct request
retains its KV cache and FA2 scratch; activations can remain in coordinator
storage. Existing decode-group admission still requires one token per row.

The complete group and cross-sequence write aliases are validated before any
KV append. Existing bounded attention descriptors/events order participant
streams into the coordinator. Each sequence then runs the unchanged append,
FA2 forward, and optional gate arithmetic on the coordinator stream, using
its owner's scratch. Completion orders participant reuse; partial execution
failure invalidates all participants and the coordinator. The bounded
descriptor upload and BUSY contract remain. Sequence kernels run in order;
this is not a concurrent-attention-kernel or zero-transfer claim.

The existing independent frozen attention chains now run through both decode
and prefill groups. New modes include initial prefill, resumed chunks, mixed
lengths/positions, participant activations, coordinator activations, packed
coordinator arenas, and both supported gate contracts. Invalid later rows
are checked before KV/output mutation. Two identical prefill submissions
are queued before readback to test event ordering and owner-scratch reuse.
The focused gate passes 1,562,680,320 output/reference words and
4,234,199,040 persistent KV-cache words across all modes. Evidence:
`/tmp/bitnet-goal-attention-prefill-group`.

Q/K preparation already supports variable-length groups. Transformer packed
prefill still needs separate packed-row capacity, shared projections/FFN,
last-row gathering for the batched logits head, and atomic position commits.
These operator results do not qualify native zero-state four-request
execution or vLLM performance parity.

After this change, the protected clean build has zero warnings and preserves
all four reference binaries. Standard tests, the complete CUDA suite and
diagnostic isolation, the attention-group CUDA memcheck (zero errors), and
real-model runtime policy tests pass.

### Resident last-row selection for the logits head

`BnGPUGatherRowsResidentPlan` selects host-specified rows from a resident
BF16/F32 matrix into a disjoint resident output matrix owned by the same
request. It preserves every bit and permits repeated indices. The callback
validates dimensions, exact extents, types, ownership, disjointness, and every
index before queueing any copy. Host indices are borrowed only until return.

The CUDA implementation queues one device-to-device copy per output row,
without staging allocation, index upload, tensor readback, or explicit host
synchronization on success. This makes the copy count proportional to the
number of selected rows; it is not a performance-parity result. Packed
prefill will select each participating sequence's final token before the
shared logits projection, preserving its request-row GEMM shape.

Tests pass 344,408 bit-exact selected words: all BF16 encodings, F32 signed
zero/NaN payloads/infinities/subnormal patterns, permutation, duplicate rows,
more output rows than input rows, and 5120-column model-shaped rows. Invalid
later indices and malformed plans leave guarded outputs untouched. Two
submissions precede readback, and changing the host index array after enqueue
does not affect the selected rows. Source bits and output guards remain intact.
Evidence: `/tmp/bitnet-goal-row-gather`. The packed transformer coordinator
still needs to consume this and the grouped prefill operations.

After this change, the protected clean build has zero warnings and preserves
all four reference binary hashes. Standard tests, the full CUDA backend suite
and diagnostic isolation, row-gather CUDA memcheck (zero errors), and real-model
runtime policy checks pass.

### Packed transformer prefill integration (qualification pending)

`bn_transformer_bf16_batch_prepare_packed_with_policy` now prepares a dense
coordinator with separate request and packed-token limits. Existing preparation
APIs retain decode-only allocation and admission. Packed setup requires
CP_FP32/R2 and the grouped-prefill/gather capabilities before allocation;
activations use the packed-row limit while logits storage and decode admission
use the request limit. CP scratch is bounded and reserved separately.

`bn_transformer_bf16_batch_prefill` runs shared projections, FFNs, and norms
over the packed tokens, then sequence-local convolution, CP recurrence,
Q/K preparation, and attention through explicit state owners. Requested last
rows are gathered before the shared logits projection. Intermediate chunks
can omit outputs. Full-cohort validation precedes GPU work; outputs and
positions/parities commit only after coordinator and participant completion.
Execution/observer failure poisons the complete cohort. Packed observers use
position -1 and caller sequence order.

Real-model tests pass complete-logit equality for a single packed request
against ordinary execution, including chunk/resume followed by decode. They
also check invalid-later-row atomicity, packed-row versus request capacity,
optional output preservation, mixed lengths, same-geometry sequence-permutation
isolation, and injected post-SSM failure poisoning without partial publication.
The permutation check is metamorphic, not an independent model oracle.
Protected clean build has zero warnings; standard tests, the full CUDA suite
and diagnostic isolation, and ordinary real-model runtime tests pass. The
expanded real-model CUDA memory check also passes (`REQUEST_RUNTIME PASSED`,
zero reported memory errors).

Independent four-request acceptance does **not** pass yet. A zero-state
4x129-token prefill plus 63 batched decode steps completes, but all 256 raw-logit
vectors differ from `test/bf16_cuda_batch4_reference.json` under the existing
4097-reference policy. First token differences occur at outputs 29/0/23/3.
No native values replace that frozen oracle. A separate internally stable
original startup variant captures 257 prefill and 257 first-decode stages;
it differs numerically from the old frozen run and is kept separately.
Native stage/state comparison against that variant isolates the first block's
recurrence. A separate normalization calibration using only original operands
selects existing reduction contracts and makes layer-zero input normalization
exact. All QKV/Z/BA projections, convolution outputs and final histories,
and prepared Q/K/V/decay/beta then match the original bit for bit. Read-only
observation leaves all 256 native logit vectors unchanged.

The original FlashInfer dispatcher chooses non-CP recurrence for four sequences
(192 sequence-head pairs on the 188-SM device). Its default and forced non-CP
outputs and final states match the independent whole-model capture exactly.
The native CP recurrence instead matches original forced-CP execution exactly:
3,170,304 BF16 output values and 3,145,728 F32 state values. Between these two
original algorithms, 710,807 output values and 2,673,282 state values differ.
Thus single-sequence CP qualification does not qualify the four-request
upstream arithmetic contract. These diagnostic forced-CP values do not replace
the default-path oracle.

A standalone non-CP prototype includes decay before the inverse's FP16
rounding and omits the corresponding post-inverse decay multiplication. It
matches all those original default-path output and state values on the first
layer's four-request prefill. Eight additional independent original-runtime
cases (four sequences each, nonzero initial states, lengths 1, 17, 63, 64,
65, 129, 513, and 2049) match all 96,460,800 output/state values. Standalone
CUDA memcheck passes at lengths 1, 65, 513, and 2049 with zero errors.
A separately linked full-model diagnostic then matches all 257 prefill stages
and all 96 captured SSM state/history payloads from the independent original
startup variant. Its 256 generated tokens match, but only 241 of 256 complete
raw-logit vectors match; the first remaining difference is decode step 49,
request row 2. Production integration and complete raw-logit qualification
remain necessary. The old frozen four-request oracle stays unchanged.

A read-only first-decode capture also matches all 96 original SSM state/history
payloads, including every F32 state bit, without changing any native raw logits.
Independent first-decode RMS calibration uses checkpoint embedding rows
exported through the original plugin plus original gamma and stage operands;
all 129 normalization vectors match the current policy. Some contracts tie on
these four rows, so this does not establish their equivalence for later decode.
A new original run differs at 205 prefill stages despite being internally
stable; its late-decode capture is kept separately and cannot explain the
previous run's late drift. Further qualification must identify the actual
selected original kernel contracts within one reference run.
Evidence and pending process state: `/tmp/bitnet-goal-packed-prefill`.
The API is explicit and is not enabled as the default inference path.
No four-request correctness or performance-parity claim follows from this
integration.


### Original kernel tracing and independent packed diagnostic qualification

A separate original run records all 8,256 RMS launches (129 per model call),
including parameter names, source hashes, reduction blocks, warp counts, and
selected compiled-kernel hashes. All 63 decode calls keep the same contracts.
Its control and observed runs have identical complete raw logits. All 16,448
prefill/decode stage tensors are retained in the evidence directory. This
separate startup variant does not replace any frozen reference.

The matching diagnostic policy is derived from these launch contracts and
cached PTX, with all copies' floating-point instruction sequences checked for
consistency. It is not selected by fitting model logits. All 129 original
prefill normalization vectors independently match those contracts. An explicit
rounded-division mean experiment fails 128 of those checks and is rejected;
PTX text alone did not justify changing the native mean calculation.

The separately linked non-CP packed diagnostic matches all 256 complete raw
BF16 logit vectors and all 256 greedy tokens against this original run. A
read-only late capture additionally matches all 257 prefill and 257 decode-49
stage tensors, plus all 96 state/history payloads before decode 49 and all 96
after it: 998,023,168 values in total. F32 recurrent states are compared as
32-bit words. Observation leaves every native raw logit vector unchanged.

Evidence: `/tmp/bitnet-goal-packed-prefill/original-kernel-trace`,
`traced-native-logit-comparison.json`, `traced-late-comparison.json`, and
`traced-prefill-policy-proof.json`. The non-CP implementation remains a
separately linked diagnostic; production capability declarations, admission,
regression fixtures, integration checks, and memory checking are still needed.
No default-path or performance-parity promotion follows from this evidence.

### Explicit non-CP production recurrence (verification in progress)

`BnGPUGDNCPPrefillResidentPlan.recurrence` now selects CP (zero/default) or
NONCP. The latter reuses the bounded workspace and grouped state-owner/event
protocol, but runs one sequential 64-token-block sweep. Decay enters the
inverse before FP16 rounding and is omitted from its post-inverse weight
multiplication. Unknown recurrence values reject before GPU submission.
CUDA advertises both contracts on the qualified SM120 target only.

`BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32` (5) admits that capability explicitly
for singleton and packed prefill. Request restart and coordinator policy
matching retain the selected contract; decode keeps the existing native
recurrence. Other arithmetic profiles retain their previous behavior.
`bench/policies/qwen38-cuda-packed-noncp.json` records the independently traced
norm contracts without changing default inference selection.

Forty-eight new original-runtime synthetic fixtures cover zero/nonzero initial
states, multiplier/log decay, and lengths 1 through 8192. Existing CP fixtures
are unchanged. Singleton output/state guards and malformed-plan atomicity now
include invalid recurrence values. A mixed NONCP129/CP17 group verifies distinct
arithmetic, lengths, state owners, and event reuse. The combined CP/non-CP
primitive check passes 1,356,791,808 exact output/state words. The protected
clean build has zero warnings and standard tests pass. The complete CUDA suite
and diagnostic isolation pass, as do all real-model runtime tests, now including
NONCP singleton/packed chunk-resume, restart, failure poisoning, and capability
admission. Production replay matches all 256 complete independent logit vectors.
The combined primitive CUDA memory check reports zero errors; the expanded
whole-runtime memory check also passes with zero reported errors.

A reproducible loaded-model replay is built with `make BN_ENABLE_CUDA=1
BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=... reference_cuda_bf16_packed`. It takes
`MODEL.gguf PROMPT.i32 POLICY.json OUTPUT_DIR` and exports 64 four-row BF16
logit matrices. `python3 test/verify_bf16_packed_reference.py OUTPUT_DIR`
checks all 256 rows against the separate original-only
`test/bf16_cuda_packed_traced_reference.json`; it never regenerates expectations.
The original 1K, 4097-token, and four-request frozen oracle hashes remain
unchanged. Evidence: `/tmp/bitnet-goal-noncp-integration`.


The exact original capture program is retained as
`test/reference_cuda_bf16_packed_original.py`; its SHA-256 matches the frozen
fixture provenance. It runs inside the pinned vLLM/FlashInfer image and expects
the documented evidence, model/config, original plugin wheel/extension, and
prompt mounts. It is not part of unit tests and does not update native goldens.

### Packed serving benchmark

`bench_bf16_batch` accepts `--packed-reference TOKENS.i32` alongside its
existing `--policy` and `uniform`/`mixed` options. The reference file contains
request-major little-endian int32 token IDs, exactly `REQUESTS * OUTPUTS` words,
including the first prefill-sampled token. Use independently generated tokens
for the same prompts, arithmetic policy, and scheduling geometry. Streams must
match before timing results are reported. The default mode retains its
isolated-execution checks; packed mode reports an external-token reference.

Packed mode allocates a coordinator for `MAX_LIVE * PREFILL_CHUNK` tokens,
rejecting budgets above 8192 before model load. Each scheduler iteration runs
ready decode work first, then submits one cohort containing a bounded chunk
from each pending prompt. Intermediate chunks omit logits; each request's last
chunk supplies its first output. Request-local positions and recurrent/KV state
continue through the existing packed runtime transaction.

JSON output includes the packed token budget, prefill/decode batch counts,
processed prefill tokens, and separate prefill/decode phase wall times. These
phase times include host sampling and request teardown where applicable;
request allocation/admission also contributes to the reported elapsed time
and TTFT. They are not kernel-only measurements.

For the qualified four-request fixture, export `streams[*].tokens` from
`test/bf16_cuda_packed_traced_reference.json` in request-major int32 format.
Use 64 outputs, four requests/live slots, 129-token chunks, max sequence 8192,
uniform lengths, and `bench/policies/qwen38-cuda-packed-noncp.json`.
`python3 bench/verify_bf16_packed_batch.py RUN.jsonl` then checks the independent
256-token streams, exact policy, geometry, and batch counts before summarizing
throughput, TTFT, and inter-token p50/p95/p99 using nearest-rank percentiles.
This verifier does not establish vLLM performance parity.

The benchmark mode passes a protected clean build without warnings, standard
unit tests, CPU argument/reference/budget checks, and three GPU runs checking
all 256 independently supplied tokens. The scheduling helper also passes CPU
unit tests and AddressSanitizer/UndefinedBehaviorSanitizer. Mixed and chunked
serving schedules still need end-to-end qualification against matching
independent references. Evidence: `/tmp/bitnet-goal-packed-serving`.

### Matched packed performance checkpoint

A fresh run of the pinned original vLLM/FlashInfer runtime captures its actual
normalization launch contracts separately from timing. Native execution uses
that derived policy and matches all 256 complete BF16 logit vectors before
performance comparison. Capture hooks are removed before the three timed
reference runs. Both engines use the same GPU, GGUF weights, BF16 activations,
F32 recurrent state, four 129-token prompts, 64 output tokens per request,
8192-token context limit, and disabled CUDA graphs.

| Median phase measurement | Native | vLLM |
| --- | ---: | ---: |
| Prefill, 516 input tokens | 633.520 ms | 1327.594 ms |
| Decode batch latency, four requests | 47.939 ms | 37.698 ms |

Prefill throughput is 2.096 times the reference on this case. Decode latency
is 1.272 times the reference and fails the 1.05 parity ceiling. This single
geometry does not establish parity across contexts or concurrency levels.
Phase timing boundaries differ in host overhead: native includes sampling and
request teardown; reference uses engine-step wall time. Whole-request latency
and tail metrics remain separate serving measurements.

A separately instrumented native run profiles only the measured serving loop;
its slower timings are excluded from the table. Quantized matrix kernels
account for about 90% of decode GPU duration. Q5_K affine pair-split kernels
alone consume 1020.6 ms across 63 decode steps. This identifies small-batch
matrix execution as the first optimization target; CUDA API durations that
include GPU waits must not be counted as independent transfer overhead.
Evidence, policies, raw-vector checks, timing samples, and profiler output:
`/tmp/bitnet-goal-packed-parity`.

The first profile-driven change bounds the short-prompt Q4_K/Q5_K affine
pair-split kernel to six resident blocks for SM120 row counts at least 12288.
The arithmetic and reduction order are unchanged. The six-block launch is
restricted to the larger SM120 row grids. Synthetic comparisons showed
that the higher occupancy avoids an extra execution wave on the large row
shapes; a single-warp alternative was slower and was discarded.

A fresh protected clean build and standard tests pass with zero warnings.
The complete CUDA suite and diagnostic isolation pass. Frozen original Q4_K
and Q5_K cases additionally check repeated rows at 12287, 12288, and 17409 rows,
including masked row/token tiles. Both focused CUDA memory checks report zero
errors. Loaded-model replay still matches all 256 complete independent BF16
logit vectors, and all three new serving runs match their 256 expected tokens.
Median decode batch latency is 46.324 ms, compared with the earlier 47.939 ms
native baseline and 37.698 ms vLLM reference. This is a 3.4% native improvement
at this checkpoint; the remaining 22.9% reference gap fails the parity target.
Evidence: `/tmp/bitnet-goal-decode-tiles`. These repeated runs are sequential,
so an interleaved baseline comparison remains useful for controlling clock drift.

The next change vectorizes Q6_K tensor-core weight unpacking without changing
integer dots, scales, or floating-point reductions. Four values are unpacked
per lane using halfword loads, preserving the format's two-byte row alignment.
The repeated-row checks now include Q6_K at a 630-byte row stride, so alternate
rows exercise alignment that would make four-byte raw loads unsafe.
The protected clean build and standard tests pass without warnings; the full
CUDA suite, diagnostic isolation, and focused memory checks pass. The loaded
model still matches all 256 complete independent BF16 logit vectors, and all
three timing runs match their expected token streams.

Median native decode is now 43.343 ms per four-request batch, a cumulative
9.6% reduction from the initial 47.939 ms. Prefill is 620.718 ms. Decode remains
15.0% slower than the 37.698 ms reference, so parity is still unachieved.
These sequential sample medians are supplemented by three interleaved runs
of each native binary, in forward/reverse/forward order. Their median decode
latencies are 50.674 ms before either change, 47.287 ms after the Q4_K/Q5_K
change, and 43.898 ms with both changes: a 13.4% cumulative reduction under
those controls. All nine runs match the independent token streams. Historical
control binaries contain profiler start/stop calls outside the measured loop;
no profiler is attached during these timing runs. The different absolute
latencies demonstrate why hardware-matched timings still need repeated,
closely scheduled controls.
Evidence: `/tmp/bitnet-goal-decode-q6-vector`.

A subsequent profile found a compiler side effect in the first occupancy
change: even its one-block annotation raised register usage on small launches
from 123–126 to 148–152 registers per thread. The implementation now shares a
forced-inline arithmetic core between an unbounded entry point and a separate
six-block entry point. This removes that annotation from small launches while
retaining the measured large-grid path.

This correction passes the protected clean build and standard tests without
warnings, the complete CUDA suite and diagnostic isolation, both focused
memory checks, and the 256-vector independent loaded-model replay. Three new
verified timing runs give median decode 42.595 ms and prefill 621.847 ms. Decode
is 11.1% below the initial native median but remains 13.0% above the earlier
vLLM median; parity is still unachieved. Evidence:
`/tmp/bitnet-goal-decode-affine-occupancy`.

The follow-up native profile confirms that the unbounded small kernels return
to 126 registers for Q5_K and 123 for Q4_K. The bounded variants use 80.
Summed decode GPU duration is 2629.269 ms across the 63 steps, compared with
3000.639 ms in the initial profile; profile timings remain excluded from the
performance table. The profiled run also passes the independent token check.

A fresh profile of the pinned original runtime selected different normalization
autotuner contracts. Its independent capture is kept separately, with a new
policy derived from the actual launch configurations and PTX. Native replay
matches all 256 complete logit vectors for this policy too, before profiling.
Thus the following kernel comparison uses matching arithmetic contracts;
profiled timings do not replace the uninstrumented parity samples.

Across 63 decode steps, original/native summed kernel durations are
2293.892/2585.822 ms. Kernel interval unions equal the sums, so overlap does not
inflate these totals. The principal gaps are Q5_K matrix execution
568.810/781.642 ms, Q4_K matrix execution 172.847/252.422 ms, and Q6_K matrix
execution 115.174/223.427 ms. IQ4 totals are already close. Native launches
91,728 decode kernels versus 132,993 in the reference: reducing launch count
alone would not address the measured compute gap. Evidence:
`/tmp/bitnet-goal-packed-vllm-profile`.

### Mixed prefill/decode correctness

The schema-4 fixture `test/bf16_cuda_mixed_reference.json` preserves a fresh
independent capture of four concurrent prompts of lengths 129, 5, 129, 5,
with 64 outputs each, a 256-token scheduler budget and 64-token prefill chunks.
The pinned original V2 model runner binds internal request IDs to complete
controlled prompts and records every input ID, position and row order.
Native replay matches all 66 matrices: 260 complete vocabulary heads,
64,563,200 BF16 words. This includes four partial-prefill heads in addition
to the 256 sampled outputs. This is a correctness result; native mixed
serving performance remains unqualified.

`bn_transformer_bf16_batch_step` retains explicit per-row recurrent roles.
The original metadata function places a contiguous decode prefix before a
prefill tail; even a one-token generation row in that tail uses prefill
recurrence. The first three calls therefore use prefill, mixed, prefill
recurrence, followed by 63 decode calls. Logical prompt progress alone cannot
determine the recurrent operator. The first three calls include four logical
decode inputs, so their duration must not be reported as pure prefill time.

The original mixed decode operator has a distinct normalization contract:
128-thread reductions followed by approximate reciprocal square root. Pure
decode retains its existing packed square-root/division contract. An explicit
backend profile selects this behavior without changing default arithmetic.
Eight model-free cases independently exercise both original operators,
including varied exponents, across resident and grouped native execution.
The CUDA checks compare 60,235,776 BF16/F32 words, preserve canaries and inputs,
and reject unknown profiles before submission. Mixed runtime checks cover
invalid cohorts, unavailable capability, positions, counters and poisoning
after partial device writes. Full CUDA, memory, synchronization, model runtime
and legacy 256-head replay checks pass.

Reproduce the untimed replay after building with CUDA and MKL VML enabled:

```sh
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=/path/to/mkl reference_cuda_bf16_mixed
python3 test/prepare_bf16_mixed_reference.py /tmp/mixed-schedule.bin
mkdir -p /tmp/mixed-logits
./reference_cuda_bf16_mixed MODEL.gguf /tmp/mixed-schedule.bin bench/policies/qwen38-cuda-mixed-noncp.json /tmp/mixed-logits
python3 test/verify_bf16_mixed_reference.py /tmp/mixed-logits
```

Use the pinned model hash and SM120 arithmetic/capabilities recorded in the
fixture. The serializer validates input identity and state progression before
model execution. The verifier checks every matrix and sampled row hash;
diagnostic disk writes disqualify this replay as a performance measurement.
Evidence: `/tmp/bitnet-goal-mixed-scheduler-v2-bound/gdn-profile-promotion`.

The opt-in `bench_bf16_batch --fused-steps` path now submits these mixed cohorts
through the same entry point. Buffered step records expose exact request order,
positions, token counts and recurrent roles without printing during timing.
All 256 sampled full vocabulary heads match before timing. The schema-4
verifier checks all 66 step geometries, the immutable arithmetic policy,
logical/backend counters, token streams and delivery times. Diagnostic logit
I/O is accepted only in correctness mode, which emits no performance metrics.

Three uninstrumented runs on the pinned GPU give these medians:

| Measurement | Native | Original reference |
| --- | ---: | ---: |
| Initial three mixed steps | 442.446 ms | 798.983 ms |
| Remaining decode step | 37.387 ms | 36.390 ms |
| End-to-end throughput | 90.708 tokens/s | 82.784 tokens/s |
| Last request's first token | 466.832 ms | 799.656 ms |

Remaining decode is 2.74% slower, within the 5% target for this geometry.
The initial mixed phase is 1.806x faster; it includes four logical decode
inputs and is not a pure prefill throughput result. Native elapsed time includes
session admission and reclamation. Original reference per-request tail timings
were not captured, so there is no mixed-request tail-latency parity claim.
This qualification covers four requests arriving together; staggered admission,
a production KV pool and memory-pressure scheduling remain open.
Portable results and source hashes: `docs/qwen38-cuda-mixed-correctness.json`.
Evidence: `/tmp/bitnet-goal-mixed-serving-trace`.

```sh
./bench_bf16_batch MODEL.gguf PROMPT.i32 64 4 4 64 8192 mixed --policy bench/policies/qwen38-cuda-mixed-noncp.json --packed-reference TOKENS.i32 --fused-steps > mixed.jsonl
python3 bench/verify_bf16_mixed_batch.py mixed.jsonl
```

`PROMPT.i32` is stream 0's complete input from the schema-4 fixture;
`TOKENS.i32` contains the fixture's expected outputs in request-major order.
The benchmark constructs the other controlled prompts by the documented swaps.

The subsequent request-pool qualification adds retained KV byte accounting and
state reuse without changing GPU arithmetic. The 20-request replay still matches
all 1,280 complete heads, including four requests placed into restarted slots.
An 8 GiB KV payload limit charges all 16 allocated slots even after they become
idle. Caller-owned handle lifecycle, failure cleanup, the actual allocation count
and focused GPU memory checks pass. Full results and the separate non-KV memory
and scheduling limits are in `qwen38-cuda-session-pool.md` and
`qwen38-cuda-session-pool.json`.

## Controlled staggered compute qualification

A twelve-request reference introduces four requests at engine steps 0, 8, and
68, with at most eight live requests. Prompts contain 129, 5, and 129 tokens
respectively; each request produces 64 outputs. The reference uses a 256-token
step budget and a 64-token prefill chunk bound. This controls arrival steps;
it does not measure a fixed wall-clock offered load.

Native exact-input replay matches all 134 matrices, 784 full heads, and
194,682,880 BF16 words. Input IDs are bound to the original prompts and greedy
history. Actual recurrent dispatch and normalization contracts are captured
independently. Read-only observation leaves every reference matrix unchanged.
The initial exact-input replay prepares all twelve sessions. The subsequent
[arrival runtime](qwen38-cuda-arrivals-runtime.md) derives the schedule and
qualifies the same full heads with an eight-slot pool.

Two compute changes are required. When decoded gate and up matrices share a
quant format, the original runtime multiplies their stacked matrix in one GEMM.
Separate GEMMs can select a different reduction tree. Backend preparation now
owns one concatenated allocation and two borrowed row views; compiled FFN
execution uses one GEMM and the existing compiled SiLU arithmetic. Other
execution layouts retain their previous path. CPU weight ownership is unchanged.

The explicit batch attention policy now also uses the maximum query length to
select FA2 mode. When any row has multiple query tokens, even one-token rows use
causal nonsplit prefill mode. Query and KV bounds remain local to each request.
The singleton policy retains its existing selection. Independent primitive
references cover merged FFN geometry and seven ragged attention cohorts in both
policies, including one-token rows alongside 63-token rows.

The protected clean build has zero warnings. Standard tests, the full CUDA
suite and diagnostic isolation, focused memcheck and synccheck, and real-model
runtime tests pass. Existing batch-four, batch-sixteen, 4097-token chunked, and
four-request mixed full-vector references remain exact; their goldens were not
changed. See [staggered evidence](qwen38-cuda-staggered-correctness.json),
[portable reference](../test/bf16_cuda_staggered12_reference.json), and
[explicit policy](../bench/policies/qwen38-cuda-staggered12-noncp.json).

Replay the exact-input diagnostic with `test/prepare_bf16_mixed_reference.py`,
`reference_cuda_bf16_mixed`, and `test/verify_bf16_mixed_reference.py`, supplying
the portable reference and policy above. The subsequent
[bounded arrival runtime](qwen38-cuda-arrivals-runtime.md) derives its schedule
from arrivals and prompt progress and feeds its own greedy outputs into decode.
Its eight-slot execution preserves all 784 full heads.

The `generate_batch_plan` module provides allocation-free host planning with
explicit request, prefill, chunk, token, and context limits. It reserves decode
capacity before dividing the remaining budget, preserves admission order for
equal query lengths, and separates logical prompt progress from recurrent
execution roles. The `generate_batch` runtime adds FIFO admission, cancellation,
backpressure, failure cleanup, and cohort token publication before reclamation.
Model-free tests reproduce all 134 captured cohorts and pass ASan/UBSan.

A Q8 optimization resolves the initial eight-request decode gap. It stages
adjacent blocks together and uses a larger cooperative tile while preserving
the original integer dots and sequential F32 FMA order. Full CUDA/diagnostic
tests, focused Q8 memcheck/synccheck, real-model runtime checks, all 784 arrival
heads, and legacy batch-four/batch-sixteen full vectors pass without changing
goldens. The whole arrival-runtime memcheck passed before this kernel change;
focused checks cover the changed kernel's memory and synchronization behavior.

Three-run medians are 721.256 ms for the initial pure-prefill steps versus
1388.994 ms reference (1.926x throughput), and 47.146 ms mean decode versus
49.307 ms (4.4% lower latency). Both four- and eight-request decode cohorts
individually meet the 5% floor. Throughput is 101.870 versus 83.116 tokens/s.
p95 TTFT is 721.255 versus 1471.062 ms; p99 ITL is 298.030 versus 683.918 ms.
Scope, all samples, profile counters, and qualification gates are preserved in
[arrival performance](qwen38-cuda-arrivals-performance.json). These controlled
engine-step arrivals do not establish fixed wall-clock offered-load performance
or full-goal completion.
