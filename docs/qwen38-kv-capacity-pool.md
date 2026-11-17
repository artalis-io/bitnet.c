# Request-sized dense KV pool

The dense BF16 pool now exposes `bn_transformer_bf16_pool_acquire_capacity`.
The existing fixed-capacity acquisition path remains available. This qualifies
a pool capability; the generic batch runtime still uses its existing acquisition
callback and does not yet automatically pass each request's required capacity.
This change budgets KV payload only, not total device memory.

The subsequent [request-aware admission integration](qwen38-request-capacity-admission.md)
now connects this API to the generic runtime through an opt-in constructor and
CUDA adapter. It preserves full model controls and reports the measured
cold-admission tail tradeoff. The pool-only scope above describes this stage.

The new path rounds the requested capacity up to a power of two, with a floor of
`max_tokens` and a clamp at the configured pool capacity. It first reuses the
smallest adequate idle slot. Oversized reuse retains the full actual allocation
charge. If no suitable idle slot exists, it checks whether active leases leave
enough slots and bytes before reclaiming anything. BUSY leaves all cache entries
unchanged and queues no allocation or GPU work. Feasible admission can reclaim
idle slots, largest first; it never moves, shrinks or evicts an active lease.

Failed preparation may occur after idle reclaim, but cannot leave a phantom
allocation or lease. Poisoned requests and failed restart are discarded and
uncharged at their actual capacity. Healthy eviction and failure discard have
separate counters. `slot_capacity[16]` and `slot_leased[16]` snapshots expose
retained allocations, including idle entries. `kv_bytes_per_slot` continues to
mean the configured full-capacity ceiling; `reserved_kv_bytes` is the actual sum.
Initialization still requires budget for at least one configured maximum slot.

## Validation

Model-free tests cover bucket rounding, the minimum token floor, non-power-of-two
maximum capacity, best-fit selection, slot and byte pressure, mutation-free BUSY
with idle cache entries, active-lease isolation, idle reclamation, failure cleanup,
oversized reuse charges and legacy interoperability. Unit tests and ASan/UBSan
pass. Protected clean CUDA/MKL build has zero warnings; the standard suite passes,
and protected user diagnostic binaries retain their hashes.

The real Qwen27 integration checks the following with the existing pool-only
model fixture:

| Quantity | Observed |
| --- | ---: |
| Configured maximum capacity | 512 tokens |
| Requested capacity | 33 tokens |
| Rounded allocation capacity | 64 tokens |
| Pool KV budget | 32 MiB |
| Simultaneous sized leases | 2 |
| Actual GPU KV allocation payload | 8 MiB total |

The resident allocation wrapper independently counts the two BF16 KV vectors
per attention layer. Thirteen full-vocabulary prefix outputs, through position
64 in matching chunks, equal fixed-512-capacity execution byte for byte. A
65th-token request is rejected without allocation, writes, reads, linear work
or synchronization, and position remains 64. Reuse allocates nothing, resets
state, and reproduces the initial prefix output exactly. Other session storage
and the separate reference fixture are outside the KV budget measurement.

The legacy controlled runtime also retains all 134 cohorts and 784 independently
captured full heads (194,682,880 BF16 words). No quantization, CUDA kernel or
existing expected-value changes were needed. Broader CUDA kernel qualification
remains in the preceding stages; this change uses their existing request
initialization/restart/free operations.

Reproduce the model check with the qualified build and designated GPU:

```sh
./test_bf16_runtime "$MODEL" --native --plain --mkl-vml --pool-only
```

Detailed source hashes, scope and checks are in
[the machine-readable evidence](qwen38-kv-capacity-pool.json). External logs are
under `/tmp/bitnet-goal-pool-sized`.

Next, carry each queued request's prompt-plus-output capacity into backend
acquisition, preserving the generic runtime's legacy callback contract. Then
qualify variable-capacity serving against the independent full-model controls
and account for non-KV device storage separately. Existing fixed-capacity
benchmarks are not automatically memory-optimized by this API addition.
