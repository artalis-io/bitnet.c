# Qwen3.8 CUDA mixed-step token cap

The optional `BnGenerateBatchLimits.mixed_token_budget` bounds steps containing
logical decode work. Zero retains the qualified legacy schedule. A positive cap
must be at least `max_requests` and no greater than `token_budget`. Decodes reserve
one token each; prefills share the remaining budget. Prefill-only steps retain the
full token budget. This prevents long prompt chunks from blocking active decodes
while ensuring selected prefills still make progress.

The subsequent [one-token prefill specialization](qwen38-cuda-single-token-prefill.md)
now passes the bounded twelve-request case: p99 is 69.41 versus 69.56 ms, with
all independent references preserved. Sustained serving qualification remains open.

## Measurement

Same designated SM120 GPU, model SHA, pinned vLLM image, BF16 arithmetic and F32
SSM state as [the original wall-clock case](qwen38-cuda-wall-clock.md). Twelve
requests arrive at fixed deadlines 100/1100/4100 ms in groups of four; prompts
129/5/129 tokens, 64 outputs each, eight live requests, four prefill lanes,
64-token chunks, 256-token bulk budget and 4 GiB KV payload budget. New mixed cap:
eight tokens. Original vLLM uses public `max_num_scheduled_tokens=8`,
`prefill_policy="decode-aware"`, and
`VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=1`. No scheduler or model arithmetic patch.
Three timed runs follow warmup. Both engines include deadline polling and
request enqueue in elapsed time; vLLM includes a CPU geometry observer, with no
GPU tensor observation. Final teardown is excluded. This is a serialized worker
benchmark, not sustained network serving or a standalone prefill throughput test.

| Three-run median | Native | vLLM |
| --- | ---: | ---: |
| Output tokens/s | 92.27 | 79.96 |
| Mean pure decode step, ms | 46.44 | 50.13 |
| Four-request decode, ms | 37.77 | 37.25 |
| Eight-request decode, ms | 58.75 | 64.11 |
| p99 inter-token latency, ms | 73.12 | 69.56 |

Throughput improves 15.4% and mean decode latency improves 7.4%. p99 inter-token
latency is **5.12% higher**, narrowly outside a 5% target. Overall serving parity
remains unqualified. Compared with the previous native 256-token mixed budget,
p99 drops from 285.96 to 73.12 ms and throughput drops from 101.68 to 92.27
tokens/s. Different normalization profiles and wall-clock schedules are recorded;
these are complete configuration results, not an isolated kernel speedup.
Nearest-rank tails use 756 inter-token intervals per run and then a three-run
median. The twelve-request sample does not establish sustained-load tails.

## Independent correctness

The original runtime's compilation configuration selected a different RMS norm
reduction profile. The previous profile failed comparison and remains in the
external evidence directory. The new explicit policy is derived from the original
selected kernel configurations and PTX, without fitting outputs:
`bench/policies/qwen38-cuda-decode-aware8-noncp.json`. No production arithmetic or
GPU kernel changed for this policy.

Fixed deadlines produced two actual native schedules. Both were regenerated with
original runtime math at their actual enqueue steps:

- 143 cohorts: the native runtime diagnostic matches all 844 full BF16 heads
  (209,582,080 words), including partial prompt heads.
- 144 cohorts: the packed replay matches all 848 heads (210,575,360 words).
  Both corresponding wall-clock timings match all geometry and greedy streams.

The third timing matches the 143-cohort control. All 768 sampled outputs per run
are independently checked. Original input histories and physical recurrent roles
are independently bound; NONCP prefill dispatch is observed at all recurrent
layers. Cross-engine free wall-clock schedules and token streams differ; equality
across those schedules is not claimed. Portable controls are
`test/bf16_cuda_decode_aware{143,144}_reference.json`.

## Validation and reproduction

Baseline and final standard suites pass. Protected clean CUDA/MKL build passes
with zero warnings; user llama diagnostic binaries retain their hashes. Planner
ASan/UBSan and targeted runtime/wire/verifier tests pass. The legacy 134-cohort
runtime geometry and full-head reference remain unchanged and pass with their
original policy. No new CUDA kernels were introduced by this change.

BGA1 and BGA2 bytes remain compatible. BGA3 extends BGA2 with a ninth header integer
for the mixed budget and carries no expected continuations. Logs retain
`correct:null`; the external verifier supplies independent numerical evidence.

```sh
python3 bench/prepare_bf16_arrivals.py \
  test/bf16_cuda_staggered12_reference.json /tmp/capped.bin --mixed-budget 8 \
  --arrivals-ms 100 100 100 100 1100 1100 1100 1100 4100 4100 4100 4100
./bench_bf16_arrivals "$MODEL" /tmp/capped.bin \
  bench/policies/qwen38-cuda-decode-aware8-noncp.json --timing > /tmp/capped.jsonl
python3 bench/verify_bf16_wall_clock.py /tmp/capped.jsonl --workload /tmp/capped.bin \
  --reference test/bf16_cuda_decode_aware144_reference.json
```

Timing changes can produce another geometry; the verifier rejects an unmatched
control. Use the 143-cohort control when appropriate, or obtain a new independent
control. Detailed per-run timings, phase counts, measurement scopes and source
hashes are in [the machine-readable evidence](qwen38-cuda-decode-aware.json).
External raw evidence: `/tmp/bitnet-goal-mixed-budget`.
