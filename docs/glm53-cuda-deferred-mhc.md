# Default deferred GLM mHC and KDA boundaries

The resident CUDA path now preserves the default GLM mHC post/pre fusion
contract, including the batch-size transitions at 4/5, 15/16 and 32/33 tokens.
The attention orchestrator can retain the KDA branch and coefficients for the
following fused boundary, and can consume an incoming deferred FFN branch.
Existing zero-initialized plans retain their materialized behavior.

This closes the previously documented default-mHC arithmetic gap for the
qualified SM120, four-stream, width-4,096 profile. It does not enable complete
GLM generation: indexed MLA, full FFNs/MoE, the whole-model binder/executor,
long contexts and measured vLLM parity remain open.

## Why the materialized shortcut differs

For up to 32 tokens, the original fused operator computes a new residual in
F32, projects that **unrounded F32 residual** through F32 weights and accumulates
its squared sum. It also stores a BF16 copy of the residual for collapse and
future residual use. Computing postmix, rounding it to BF16, then invoking the
ordinary pre operation changes this contract. Across the 36 actual-weight
reference cases, that shortcut changes 81,576 normalized BF16 values in the
small-batch cases; above 32 tokens the reference paths coincide.

The native implementation follows the original selection:

| Tokens | Projection tile | Splits | Threads | Projection/statistics input |
|---|---:|---:|---:|---|
| 1–4 | 2 | 16 | 256 | Unrounded F32 mixed residual |
| 5–15 | 2 | 8 | 128 | Unrounded F32 mixed residual |
| 16–32 | 6 | 8 | 128 | Unrounded F32 mixed residual |
| Above 32 | TF32 MMA | Explicit normal-pre splits | Existing TF32 kernel | BF16 materialized residual |

Small-batch projection retains the original per-feature/source-stream FMA order,
warp butterfly reduction and ascending cross-warp/split reductions. The
coefficient and collapse/normalization kernels retain their previously qualified
contracts. Above 32 tokens, postmix and normal pre execute on the same request
stream. The fused API validates the complete active input/output set before
submitting either path.

## API and ownership

`BnGPUMHCFusedResidentPlan` contains the next pre plan and four previous
request-owned views: residual, branch, post coefficients and combination matrix.
The next pre plan's residual view is an output in this operation. Every output
is disjoint from all inputs and other outputs. Weights remain immutable backend
resources; no handles are added to CPU model weights.

The setup-only fused split estimator selects the default profile. Execution
queues three kernels for small batches and four for the materialized large
batch path. It allocates no buffers, copies no activations to/from the host and
inserts no fences. Backend/device failure can follow partial submission; callers
discard the request rather than commit its outputs or recurrent state.

`BnTransformerBF16MHCKDAPlan` adds explicit `has_previous` and `deferred` flags.
With previous inputs, the orchestrator queues the fused boundary before KDA.
Deferred output retains the branch, current residual and coefficients instead
of materializing a final postmix. All active visible views are checked across
the entire composition before submission. Opaque allocation/weight validation
remains a backend responsibility. No session position is advanced by this API.
The original materialized mode remains covered by its checkpoint replay.

Weight preparation and its budgets are unchanged from the
[materialized mHC qualification](glm53-cuda-mhc-attention.md). The tested chain
retains KDA plus attention/FFN mHC weights, totaling 149,816,280 bytes of GPU
tensor payload. Request storage, mapped checkpoint pages and allocator/context
overhead are separate from that number.

## Independent qualification

Original vLLM operators captured 36 actual-weight cases, each run twice with
exact repeated results. They cover next stages `blk.0` FFN, `blk.1` attention,
`blk.44` FFN and `blk.44` attention, with their preceding attention/FFN
coefficients. Batch sizes are 1, 4, 5, 9, 15, 16, 32, 33 and 128. CUDA 13.5 is
explicitly selected for the original SM120 DeepGEMM/TileLang reference.

The native replay loads weights through the production GGUF reader/binder and
consumes only the original incoming residual, branch and coefficient state.
Every current residual, projection partial, squared sum, coefficient,
collapsed input and normalized input matches bit for bit. The replay checks
48,918,512 words, including input immutability and both norm modes. Invalid
plans preserve checked outputs. The committed generator also reproduces every
pre-existing 32-case reference boundary unchanged when adding the 9-token cases.

The coupled replay uses blocks 1 and 44 and covers all six nine-token schedules:
full prefill, `[2,7]`, `[3,3,3]`, `[7,1,1]`, `[6,3]`, and nine single-token calls.
It starts with original incoming deferred FFN state, computes attention pre
natively, feeds that resident normalized result through KDA, and executes the
following fused FFN pre boundary. Initial KDA state/history are nonzero. Incoming branch/residual/coefficient
state is supplied as an original-operator capture; this replay does not compute
the preceding FFN matrix output. All
58,644,448 compared words match exactly, including current residuals,
normalized attention input, KDA intermediates, F32 recurrent state, BF16 history,
and the following FFN's coefficients and normalized input. No reference
intermediate feeds the native chain.

A model-free procedural fixture exercises all nine batch regimes in the regular
CUDA suite. It hashes every output byte from the original operators, checks
buffer canaries and input immutability, and generates incoming coefficients
natively before the fused operation. Actual-checkpoint replay additionally
compares values directly. The fixture covers 7,248,128 output words, including
both norm modes, and reproduces byte for byte. No native output produced a
golden, and no existing reference tolerance or neural golden was changed.

The final clean build has zero warnings. Standard tests, the complete CUDA
suite including the new fixture, and diagnostic isolation pass. Actual-weight
fused/deferred replays pass CUDA memcheck; the procedural fixture also passes
memcheck and synccheck. Host ASan/UBSan with leak detection covers the updated
orchestration/preparation tests. Earlier materialized checkpoint replays remain
exact.

The warm nine-token chain records 20 kernels on one stream, including two fused
projections, with zero recorded copy bytes or allocation API calls. No
synchronization occurs between operations. The harness's final stream sync and
an additional unattributed context-sync activity after the last kernel are
recorded. These diagnostics establish residency, not statistical latency or
vLLM performance parity.

Build, sanitizer, reference and residency records are captured in
`glm53-cuda-deferred-mhc.json`.

## Reproduction

Select the intended GPU with `CUDA_VISIBLE_DEVICES` for all GPU commands. Set
`CUDA_HOME` and `PATH` to the supported compiler in the qualified reference
environment:

```sh
python test/reference_glm_mhc_fused.py --model /path/to/first-shard.gguf --out /tmp/fused-ref
python test/reference_glm_deferred_kda.py --model /path/to/first-shard.gguf --incoming-reference /tmp/fused-ref/reference --out /tmp/deferred-ref
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=/path/to/mkl test_cuda_glm_mhc_fused test_cuda_glm_deferred_kda
./test_cuda_glm_mhc_fused /path/to/first-shard.gguf /tmp/fused-ref/reference
./test_cuda_glm_deferred_kda /path/to/first-shard.gguf /tmp/deferred-ref/reference
```

`test/reference_cuda_mhc_fused.py --header PATH --metadata PATH` generates the
procedural fixture. `./test_cuda_backend --mhc-fused-reference` selects its
native replay after the test binary has been built.

The following FFN input is qualified here; its dense/expert matrix computation
is not included in this boundary replay. This does not prove complete decoder
or whole-model inference, CPU BF16 equivalence, other widths/hardware,
optimized `chunk_kda`, long-context GLM behavior or matched vLLM performance.
Those remain work under the full active goal.

The [dense decoder follow-up](glm53-cuda-dense-prefix.md) completes the first
three dense blocks and their token-embedding/state handoffs. Indexed MLA, MoE
and complete-model execution remain separate requirements.
