# Dense CUDA request pool

`BnTransformerBF16Pool` adds serialized admission and request reuse to the
qualified dense hybrid BF16 runtime. Callers own a zero-initialized handle and use `pool_init`/`pool_free`.
An initialized handle must not be copied. It owns backend-only sessions and borrows
the immutable model and GPU. The model and GPU must outlive the pool.
Capacity, chunk limit and arithmetic policy are copied at creation and remain
fixed. The API is in `src/transformer/bf16_internal.h`; implementation is in
`src/transformer/bf16_pool.c`.

A pool allocates slots lazily. `acquire` returns zero and a session on success,
one with a NULL output when exhausted, or minus one on invalid input or
preparation failure. Exhaustion submits no GPU work and performs no allocation.
Borrowed sessions must not be reset or freed by the caller.
Returning a healthy lease retains its allocations. The next acquisition clears
recurrent state and host history using the existing synchronized restart path;
KV above the new committed position stays inaccessible. Failed initialization
never charges a slot; failed restart and poisoned returned requests release
storage and remove its charge. Foreign and duplicate returns are rejected.
The caller serializes access, returns each lease once and retains no pointers
after returning it. Free the pool after its last use.

The budget counts raw KV payload for every allocated slot, including idle
slots. It charges full context capacity rather than the number of written
tokens. For the qualified model's 16 attention layers:

```
bytes/slot = capacity * 16 layers * 2 vectors * 1024 elements * 2 bytes
```

An 8,192-token slot costs 512 MiB; sixteen retained slots cost 8 GiB.
SSM state, convolution history, activations, backend workspace, model weights,
allocator overhead and driver memory are excluded. This is a KV reservation
budget, not a total VRAM limit or a paged KV allocator. The pool derives the
attention-layer count from the admitted model anatomy. `PoolInfo` exposes
current and peak charged bytes, allocated/leased slots and creation, reuse
and discard counters.

`bench_bf16_batch --kv-pool-mib N` opts into this lifetime model. Admission may
stop at the byte limit before the requested live-session limit. Returning a
completed request remains timed; final pool destruction happens after the
request workload, since idle allocations belong to the persistent pool.
Normal runs emit a `kv_pool` record marked `scope: kv_payload_only`.
The bounded-case verifiers check this accounting when enabled. Geometry and
complete-logit reference checks remain necessary: changing the admitted cohort
can change matrix arithmetic. Defaults retain the existing allocation path.

Model-free checks cover byte-limit rejection, retained idle charges, policy
ownership, foreign/duplicate returns, initialization/restart failures and
poison disposal. Address and undefined-behavior sanitizers pass. The loaded
model runtime test intercepts actual KV allocations, checks the charged 32 MiB
for a 512-token slot, verifies full-logit equality after reuse and checks poison
disposal. The protected clean build and standard suite pass without warnings.
The focused GPU memory check passes with zero errors. Across 20 requests and
16 retained slots, all 1,280 complete reference heads match (317,849,600 BF16
words). Four later requests reuse existing slots: created 16, reused 4,
discarded 0, outstanding leases 0. Both current and peak KV reservations stay
at 8 GiB after completion, including idle slots. The mixed entry point also
matches all 256 sampled reference heads with the pool enabled.

This component does not complete continuous scheduling. Staggered admission,
variable-capacity placement, a production scheduling interface, total request
memory budgets and mixed-request tail comparisons remain separate work.

Three normal runs without logit dumps give these medians:

| Measurement | Native pooled | Original reference |
| --- | ---: | ---: |
| Prefill phase | 2934.878 ms | 6612.085 ms |
| Decode engine step | 66.242 ms | 69.583 ms |
| End-to-end throughput | 112.379 tokens/s | 83.185 tokens/s |
| TTFT p95 | 8944.572 ms | 12946.963 ms |
| Inter-token p99 | 93.939 ms | 101.021 ms |

Decode averages 63 calls with 16 rows and 63 calls with four rows, rather than
one fixed cohort. Tail summaries use nearest-rank percentiles per run, then
the median of three runs (20 TTFT and 1,260 inter-token samples per run).
The reference uses its persistent 8 GiB cache. Native initial slot allocations
and subsequent restart/return costs remain inside request workload timing.
These measurements qualify this bounded geometry only.

Evidence: `/tmp/bitnet-goal-dense-kv-pool-init`.
Portable metrics, source hashes and limitations: `qwen38-cuda-session-pool.json`.
The earlier opaque heap-handle prototype and its measurements remain under
`/tmp/bitnet-goal-dense-kv-pool`; final results above use the caller-owned API.
The broad sanitizer run there was intentionally stopped before reaching the
new pool checks; the final focused pool sanitizer completed successfully.

```sh
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=/path/to/mkl bench_bf16_batch
./bench_bf16_batch MODEL.gguf PROMPT.i32 64 20 16 129 8192 --policy bench/policies/qwen38-cuda-serving20-noncp.json --packed-reference TOKENS.i32 --kv-pool-mib 8192 > pooled.jsonl
python3 bench/verify_bf16_serving_batch.py pooled.jsonl --reference test/bf16_cuda_serving20_reference.json
```

Inputs are stream 0's complete prompt and request-major output tokens from
the stated fixture. Use `--dump-logits DIRECTORY` only for the untimed word
replay, with `test/verify_bf16_serving_reference.py` and the serving verifier's
`--correctness-only` mode. The focused memory test is
`test_bf16_runtime MODEL.gguf --native --plain --mkl-vml --pool-only`.
