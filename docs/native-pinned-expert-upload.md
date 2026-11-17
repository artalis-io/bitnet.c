# Bounded pinned expert upload

Native expert stores can opt into a single pinned host staging bundle. CUDA
queues the gate, up and down projections on a model-owned nonblocking stream
and completes the triplet with one synchronization before cache publication.
Existing callers keep ordinary staging unless they explicitly provide a pinned
budget through `bn_gpu_moe_trellis_prepare_model_pinned` or the store API.
The standalone benchmark exposes `--pinned-staging`.

The pinned bundle is **13,315,584 bytes** in the qualified checkpoint. It
replaces ordinary staging within the same total staging budget. It does not
pin the RAM expert cache. Nsight records exactly one additional 13,315,584-byte
host allocation and matching free; total device allocation peak is unchanged.
Decode and temporary VRAM quotas remain separate;
retained temporary storage overlaps its existing quota. All three descriptors
are checked before allocation or copying. A failed batch drains attempted
copies, destroys private new handles and aborts potentially poisoned retained
buffers without publishing a cache entry. Host source leases remain held until
completion. The store remains serialized and model-owned.

With RAM warm and a 48-expert VRAM cache, decode median changed from
**153.069 ms to 159.536 ms** across 10 samples each
(4.22% higher latency). Every matched frame preserved cache,
source-read, allocation/refill and H2D counts. Warm decode still transfers
3,195,740,160 expert bytes per token: 720 projection copies now use 240 triplet
completion fences. A separate Nsight run confirms total stream synchronizations
fell from 893 to 413 per warm decode, with zero weight malloc/free calls and
identical frame H2D bytes. Profiler timing is excluded from these samples.

For warm bypass-prefill chunks, aggregate throughput was
78.42 tokens/s with ordinary
staging and 96.89 tokens/s with
pinned staging (four 513-token chunks per configuration). All corresponding
cache, source-read and upload counts matched. These small replay samples do
not establish serving throughput.

These are fixed replay measurements; OS page-cache/disk warmth and clocks were
not controlled. Forward timing includes the CPU copy into pinned staging;
`upload_time_ms` measures only the backend callback and excludes that copy.
The workload uses the existing explicit Diffbot lookup to Exllama Engram math
adapter, not original Diffbot full-runtime equivalence. It covers fresh513,
resumed513 and five self-fed greedy steps at context1031.

All 25 qualification gates passed: preserved default paths, pinned bypass and
both cache configurations, all-logit checks, BUSY retries, ASAN/UBSAN, CUDA
memcheck/initcheck, and fresh strict Qwen compatibility. The main runs verified
357 forwards with 714 exact published CPU/GPU
checks; a separate trace verified another 14 forwards. The protected clean build,
standard tests, full CUDA suite and diagnostics passed with zero warnings;
four user reference binaries were preserved.

The option remains disabled by default. Fewer synchronization calls did not
improve RAM-warm decode in this replay; the prefill gain does not establish a
general decode improvement.

Publication remains synchronous. Source I/O overlap, transfer/compute overlap,
production admission/serving, broader model/backend qualification and matched
vLLM parity remain open. The full goal is active and incomplete.

Machine-readable evidence: [native-pinned-expert-upload.json](native-pinned-expert-upload.json).
Prior storage-reuse evidence: [native-decode-cache-reuse.md](native-decode-cache-reuse.md).
