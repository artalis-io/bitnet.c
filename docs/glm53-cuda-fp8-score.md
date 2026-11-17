# GLM-5.3 Flash CUDA FP8 scorer

The resident SM120 scorer consumes gathered FP8 keys and per-key F32 scales,
FP8 queries with 32 heads of dimension 128, and signed F32 head weights with
query scales already folded. It preserves the original FP8 MMA and head
reduction arithmetic; negative valid scores are allowed. Other architectures
do not advertise this callback. This operation is an indexer component and
is not yet connected to full-model GLM execution.

`BnGPUIndexerFP8ScoreResidentPlan` describes one contiguous request chunk.
Its key count is exactly `floor((position+n_tokens)/4)`. Query row `i` sees only
keys below `floor((position+i+1)/4)`; later columns are written as negative
infinity, including rows with no complete pools. Zero complete keys require
empty key, scale and score views. Input and output views are exact, disjoint
and request-owned; FP8 byte inputs are four-byte aligned. Extent, overflow,
alias, alignment and ownership failures reject before submission.

Each warp computes 16 keys against the 32 query heads with native FP8
`m16n8k32` instructions. Head-pair accumulation, per-key scaling and the two
XOR shuffle reductions follow the original SM120 scorer. There is one kernel
submission and no allocation, transfer, fence or position commit in the
operation. This gathered-history prefill interface is not the eventual paged
decode implementation, and no throughput parity is claimed.

`test/reference_cuda_glm_score.py` captures independent original DeepGEMM
results twice, masking the original uninitialized padding before comparison.
The standard CUDA chain runs synthetic fixtures with signed weights, zero and
partial pools, warm positions and partial tiles. An optional reference directory
allows direct byte comparisons as well as fixture digests. Actual-checkpoint
replays consume preserved original pre-indexer outputs; they do not prove that
the native projection/cache/scorer composition has been integrated.

Additional original captures at 2,049 and 4,097 tokens cross the short-prefill
bypass. In two 4,097-token original top-k invocations, 2,046 rows changed pool
order while selected membership stayed identical. Both raw outputs are retained;
no canonicalized ordering has been substituted. Pool selection and expansion
remain separate unqualified native work, followed by MLA and full-model gates.

Final test and trace evidence is recorded in `glm53-cuda-fp8-score.json` once
qualification completes. The broader Qwen, DeepSeek, CPU, expert placement,
serving and matched vLLM performance requirements remain active.

Final qualification passed: 350,090 synthetic and 5,506,816 actual-checkpoint
F32 score elements matched bit-for-bit, including masked future columns.
Actual captures cover 1,025, 2,049 and 4,097 tokens. The clean build had zero
warnings; the standard and complete CUDA suites passed. Device memcheck and
synccheck reported zero errors for every synthetic and actual scorer replay.
Independent synthetic captures and their fixture header were byte-identical,
and the user probe binaries were preserved byte-for-byte.

The warmed 4,097-token scorer trace contains one kernel, no CUDA memory-copy
activity or allocation API calls, and one caller stream fence. All staging and
readback occur outside capture. Raw CUPTI synchronization records are retained,
including any context synchronization without a matching runtime API call.
See the [machine-readable qualification record](glm53-cuda-fp8-score.json).
