# Qwen3.8 CUDA: 48 offered requests and repeated slot reuse

The finite 48-request case completes with exact independent numerical controls,
eight live sessions and forty session reuses. Native throughput and token-gap
tails improve over both measured original-runtime configurations. This is an
overloaded arrival window followed by a drain, not steady-state or network
serving qualification. Total device-memory budgeting remains open.

## Offered workload and measurements

The original comparison uses the pinned vLLM/GGUF environment from the prior
stages. The same designated SM120 GPU, model SHA, BF16 weights/activations, F32 SSM state,
8192 context cap and 4 GiB KV payload budget are used across engines. Forty-eight
unique synthetic token prompts arrive four at a time, from 100 through 11,100 ms.
Prompt lengths alternate by group between 129, 5 and 513 tokens; each class has
sixteen requests. Every request generates 64 tokens with greedy sampling and EOS
ignored: 3,072 outputs and 10,352 prompt tokens in total. Native uses eight live
slots, four prefill lanes, chunk64, token budget256 and mixed-step cap8.

The workload specification contains prompts and offered deadlines, without
expected continuations. It is independent of the generated references. The same
prompts, deadlines and output count are sent to each original-runtime setting:

- `decode-aware`, `max_num_scheduled_tokens=8`, with
  `VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=1`.
- `round-robin`, `max_num_scheduled_tokens=256`, with the same prefill chunk,
  lane count, live-slot limit and other engine settings.

Original performance runs use their original schedulers and model/worker math.
Native chooses prefill lanes in FIFO order; the original controller rotates
lanes and can promote requests near decode. These differences affect batching.
The comparison is complete-engine behavior, not an isolated kernel speedup.

| Three-run median | Native cap8 | Original decode-aware8 | Original round-robin256 |
| --- | ---: | ---: | ---: |
| Output tokens/s | 69.63 | 28.67 | 56.95 |
| Elapsed time, s | 44.12 | 107.14 | 53.94 |
| p95 TTFT from offered deadline, s | 30.56 | 90.17 | 38.90 |
| p95 inter-token latency, ms | 69.05 | 71.18 | 68.22 |
| p99 inter-token latency, ms | 69.97 | 138.30 | 716.79 |

The native runs have 573, 572 and 572 steps; original decode-aware runs have
1,538 each, while original round-robin runs have 417 each. All engines execute
13,376 input/decode tokens. A different batching policy is a major contributor
to the end-to-end result; selecting only the slower original configuration would
misrepresent it. Relative to the faster original configuration, native
throughput is 22.3% higher. The high TTFT values reflect offered demand above
service capacity and its resulting backlog. Do not treat this finite run as an
unbounded-load admission or steady-state latency guarantee.

Pure-decode means depend on cohort width. Native four-row decode averages
38.97 ms versus 38.20 ms for round-robin256; native eight-row decode averages
59.55 ms versus 66.40 ms for round-robin256 and 66.70 ms for decode-aware8.
The per-width counts and contexts differ across free-clock schedulers; the
artifact reports each run's distribution. Aggregate decode means mix widths and
are not a matched-context kernel claim. No aggregate pure-prefill throughput
claim is made for these mixed schedules.

Tails use nearest rank over 3,024 inter-token intervals per run, followed by a
three-run median. Timing includes polling, enqueue/admission, execution, sampling
and pool return; final teardown is excluded. Native buffers host metadata until
the end. Original timing includes its CPU geometry observer, without GPU tensor
capture. Full-head diagnostic writes are excluded from timing medians.

## Numerical control and its scheduler scope

To check native's actual compute geometry, the numerical control uses original
model/worker math, sampling and KV management with an explicitly labeled host
FIFO selection adapter. It ignores previous prefill service when ordering lanes
and disables nearest-decode promotion. This control is **not an unmodified
original scheduler** and is **never used in performance measurements**. It has
no expected token inputs and does not fit numerical outputs or arithmetic
policies. Its original output histories generate the new expected values.

The observed original norm configurations/PTX reproduce the already-selected
`qwen38-cuda-decode-aware8-noncp.json` policy without changes. Original pure-host
role metadata and every recurrent layer's NONCP dispatch are checked. Both
control histories are bound to complete prompts, positions and original greedy
outputs, including all partial-prefill heads:

| Control | Full heads | Exact BF16 words | Native evidence |
| --- | ---: | ---: | --- |
| 573 cohorts | 4,136 | 1,027,051,520 | Actual fixed-clock diagnostic |
| 572 cohorts | 4,132 | 1,026,058,240 | Deterministic controlled-step runtime replay |

All three unobserved native timing geometries and all 3,072 token IDs per run
match the corresponding control. Free-clock cross-engine token equality is not
claimed when schedules differ. The 572-step replay generates its own tokens;
expected continuations are compared only after execution. Both numerical
control programs are byte-identical; only input arrival-step metadata differs.
Source hashes and the FIFO adapter's original-source provenance are retained.

## Memory, protocol and validation

Every native 48-request run creates eight slots, reuses them forty times, discards none,
and ends with zero leases. Peak and retained KV payload are exactly 4 GiB,
including idle slots. Model weights, SSM/activation/workspace storage, driver
memory and allocator overhead are outside that KV figure. It is not a total
VRAM budget claim.

The trace validator now supports up to 64 logical requests and 4096 steps while
retaining the sixteen-live-row bound. The standalone BMS1 C replay keeps its
sixteen-request/256-step limit and identical frozen bytes. Tests cover larger
logical histories, late corruption, identity/role/position errors, and rejection
of oversized standalone replays. BGA1/2/3 bytes stay compatible. BGA4 extends
controlled-step inputs with a mixed budget, allowing deterministic runtime
replay through the existing bounded session pool.

Two protected clean builds, zero-warning CUDA/MKL compilation, standard tests
and targeted protocol/reference tests pass. The original 134-step runtime and
all its full heads remain exact. No inference-library, GPU-kernel, public-header
or existing golden-reference changes were needed for this qualification.
The benchmark C change adds only protocol parsing before the timer; the earlier
performance source snapshot is separately hashed. Existing full CUDA kernel
qualification remains recorded in the preceding
[one-token prefill stage](qwen38-cuda-single-token-prefill.md).

## Reproduction

Use the qualified CUDA/MKL build and designated device described by the prior
stages. With `MODEL` pointing to the qualified GGUF:

```sh
python3 bench/prepare_bf16_arrivals.py test/bf16_cuda_load48_workload.json /tmp/load48.bin
./bench_bf16_arrivals "$MODEL" /tmp/load48.bin \
  bench/policies/qwen38-cuda-decode-aware8-noncp.json --timing > /tmp/load48.jsonl
python3 bench/verify_bf16_wall_clock.py /tmp/load48.jsonl --workload /tmp/load48.bin \
  --reference test/bf16_cuda_load48_572_reference.json
```

Actual timing can produce another schedule. Use the matching 573-step control
when applicable; an unmatched geometry requires a new independent control.
For deterministic controlled replay and full-head verification:

```sh
python3 bench/prepare_bf16_arrivals.py test/bf16_cuda_load48_572_reference.json \
  /tmp/control572.bin --mixed-budget 8
mkdir -p /tmp/control572
./bench_bf16_arrivals "$MODEL" /tmp/control572.bin \
  bench/policies/qwen38-cuda-decode-aware8-noncp.json /tmp/control572 > /tmp/control572.jsonl
python3 bench/verify_bf16_arrivals.py /tmp/control572.jsonl \
  --reference test/bf16_cuda_load48_572_reference.json
python3 test/verify_bf16_mixed_reference.py /tmp/control572 \
  --reference test/bf16_cuda_load48_572_reference.json
```

`test/reference_cuda_bf16_fifo_control.py` preserves the
original-math control generator and its explicit adapter. Per-run metrics,
phase/width distributions, source epochs, memory scope and qualification flags
are in [the machine-readable results](qwen38-cuda-load48.json). External
commands, original performance sources, image digest, captures and exits are
under `/tmp/bitnet-goal-load48`.
