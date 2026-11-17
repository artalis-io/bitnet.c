# Exact one-token recurrent prefill specialization

The twelve-request fixed-clock case now passes its throughput, mean decode and
p99 inter-token margins against the matched vLLM configuration. This is a bounded
case qualification; sustained serving, wider context/concurrency, full memory
accounting and the complete modern-model goal remain open.

The subsequent [48-request finite-load qualification](qwen38-cuda-load48.md)
checks forty session reuses and both actual schedules against independent full
heads. It expands the offered-load coverage while leaving steady-state serving
and total memory accounting open.

## Change and independent validation

For NONCP recurrent prefill with exactly one token, the strictly lower triangular
matrix is empty. The inverse's only valid element is one. A CUDA initializer
writes the same padded BF16 inverse tensor: the first element per head is
`BF16(-beta * 1)`, all other elements are positive zero. This skips the unused
key Gram matrix and full triangular inversion. Longer prefills, CP recurrence,
and the remaining state/output operations keep their execution paths. No
allocation, synchronization or backend API change was introduced.

Clean CUDA/MKL build passes with zero warnings; protected user llama binaries
retain their hashes. Standard tests and the full CUDA suite plus diagnostic
isolation pass. Existing independent recurrent references check 1,356,791,808
exact output/state words, including one-token NONCP with zero/nonzero initial
states and both multiplier/log decay inputs. Golden references are unchanged.
A standalone original-versus-specialized inverse comparison checks all 196,608
padded words; memory checking reports zero errors. That standalone check is not
a claim that the entire model workload ran under memory checking.

Model-level proof also passes:

- Native 144-cohort diagnostic: all 848 heads / 210,575,360 BF16 words exact.
- 143-cohort packed regression: all 844 heads / 209,582,080 words exact.
- All three timing geometries and 768 generated tokens per run match the
  independently captured 144-cohort control.
- Legacy 134-cohort runtime schedule and full-head control remain exact.

Portable independent controls and the explicit norm policy are unchanged from
[the mixed-budget qualification](qwen38-cuda-decode-aware.md). The specialization
was not used to generate or alter expected values.

## Matched wall-clock measurement

Same designated SM120 GPU, pinned original runtime, model SHA, precision, memory
limits and offered workload as the mixed-budget case: twelve requests in groups
of four at 100/1100/4100 ms; 129/5/129-token prompts; 64 outputs each; eight live
requests; 4 GiB KV payload; max chunk64; four prefill lanes; bulk token budget256
and mixed cap8. vLLM uses `max_num_scheduled_tokens=8`,
`prefill_policy="decode-aware"`, and
`VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=1`.

| Three-run median | Native | vLLM |
| --- | ---: | ---: |
| Output tokens/s | 92.64 | 79.96 |
| Mean pure decode step, ms | 46.57 | 50.13 |
| Four-request decode, ms | 37.92 | 37.25 |
| Eight-request decode, ms | 58.86 | 64.11 |
| p95 inter-token latency, ms | 68.49 | 67.22 |
| p99 inter-token latency, ms | 69.41 | 69.56 |

Native throughput is 15.9% higher, mean decode latency is 7.1% lower and p99
inter-token latency is 0.2% lower. All three native p99 values are 69.3–69.5 ms.
The prior capped configuration had 73.12 ms p99 and narrowly missed a 5% margin.
The diagnostic writes full heads and has higher latency; it is excluded from
performance statistics. Tails use nearest rank over 756 intervals per run,
then a three-run median. Reference timings are the preserved matched original
runs, with a CPU geometry observer and no GPU tensor capture. Cross-engine
free-clock schedules and streams differ; native numerical proof uses its own
actual independent execution control. This is not a standalone prefill throughput
or sustained network-serving claim.

## Kernel and transfer evidence

Separate Nsight runs before and after both match the independent 143-cohort
control. Each has 17 short mixed steps. Per-step aggregate kernel time:

| Operation | Before | After |
| --- | ---: | ---: |
| Full inverse | 2.855 ms | removed for one-token rows |
| Key Gram matrix | 1.074 ms | removed for one-token rows |
| Exact inverse initializer | absent | 0.186 ms |

The initializer replaces 3,264 inverse and 3,264 Gram launches in those mixed
steps with 3,264 initializer launches. Other sequence lengths retain the original
kernels. Instrumented runs are excluded from timing medians.

From the first 1024-byte input HtoD through the last output DtoH (initial
preparation/admission excluded; later admissions included), counts are identical:

| Transfer | Calls | Bytes |
| --- | ---: | ---: |
| Host to device | 19,023 | 7,076,416 |
| Device to host | 1,428 | 419,169,300 |
| Device to device | 5,068 | 13,296,197,632 |

Both runs have 1,296 stream synchronizations and 216,064 event waits in that
window. Their host API durations are inclusive and are not isolated transfer
stall measurements. The CUDA adapter rejects CPU operations during execution;
these successful runs have no observed CPU fallback. KV budget figures cover
payload including idle retained slots, not all device memory.

Use the [mixed-budget reproduction](qwen38-cuda-decode-aware.md) with the current
build and the 144-cohort control for these timing traces. Timing can produce a
different schedule; verify it against a matching independent control. Detailed
per-run values, phase counts, source hashes, profiles and verification scope are
in [the machine-readable results](qwen38-cuda-single-token-prefill.json).
External evidence: `/tmp/bitnet-goal-single-inverse`.
