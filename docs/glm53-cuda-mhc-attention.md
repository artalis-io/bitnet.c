# Resident mHC and the materialized GLM attention transition

The CUDA backend now executes the four-stream mHC pre/post operations and the
complete materialized attention transition: expanded BF16 residual streams →
mHC input normalization → KDA attention → mHC residual output. Checkpoint-bound
replay uses attention and FFN mHC weights from GLM blocks 0 and 44, with the KDA
attention transition qualified separately across six prefill/decode schedules.

This does not enable ordinary GLM generation. The legacy loader guard remains
in place. Indexed MLA, complete FFNs, the dedicated whole-model binder/executor,
and the deferred/fused inter-layer mHC precision profile remain open.

## Arithmetic and resident execution

`BnGPUMHCPreResidentPlan` and `BnGPUMHCPostResidentPlan` describe explicit
request-owned resident views. Their current CUDA implementation is exposed on
SM120. It supports four streams and widths divisible by 1,024; numerical
qualification here covers width 4,096. Other widths and GPU architectures are
not claimed as independently qualified by this work.

The pre operation submits three kernels: split-K TF32 projection/squared sum,
F32 coefficient generation with Sinkhorn normalization, and stream collapse
with optional input normalization. Postmix is a fourth kernel. There is no
CPU fallback, allocation, host transfer or synchronization inside these calls.
The setup-only split estimator matches the original operator's SM-count/batch
rule. Explicit splits in the plan pin the arithmetic independently of later
batch changes; every split partial is reduced in ascending order.

The projection retains F32 weights. Before TF32 MMA, native loads round those
weights to TF32 using round-to-nearest, matching the reference TMA conversion.
BF16 activations are exact TF32 inputs. The Q8 checkpoint projection is decoded
to F32 at preparation, preserving its low bits; it is never rounded to BF16.
The source reference uses vLLM's SM120 DeepGEMM path with TileLang mHC kernels.
CUDA 13.5 is explicitly selected for that reference, avoiding the older `nvcc`
found earlier in the shell's default `PATH`.

Pre/post coefficients and the combination matrix stay F32. Combination matrix
rows identify source streams and columns identify destinations. The normalized
pre operation preserves the original fused contract: variance uses the
**unrounded F32 stream sum**, while its numerator uses the BF16-rounded stream
sum and BF16 gamma. It is not equivalent in rounding to applying a standalone
RMSNorm to the BF16 collapsed buffer. Postmix also preserves the reference
contraction: its first residual product rounds before the post/branch FMA;
subsequent source-stream products accumulate with FMA.

`bn_transformer_bf16_mhc_kda_attention` composes these operations with the
previously qualified KDA attention sublayer. It checks all visible view extents,
types, owners and cross-operation aliases before submission. Opaque allocation
generations and weight handles remain backend checks. Backend/device failure
can follow partial submission; callers discard the request rather than commit
its output/state/history. The function allocates no buffers, performs no host
copies or fences, and does not advance a model session's position.

## Weights, budgets and ownership

`bn_backend_layout_bf16_mhc_init` consumes an immutable anatomy plan and the
merged GGUF directory. It accepts Q8_0/F32 projection storage with F32 scale,
base and gamma. It checks geometry, each owning shard's payload bounds, decoded
finite values, and both budgets before uploading. Invalid metadata, malformed
parameters or any of the four upload failures leave the destination unchanged
and release partial GPU allocations. Norm gamma rounds once to BF16; source
CPU weights remain unchanged.

Each tested attention or FFN mHC layout owns four immutable GPU buffers totaling
1,581,164 bytes, with one reused 1,572,864-byte host staging allocation. Together
with the KDA attention weights, a tested attention transition retains
148,235,116 bytes of GPU tensor payload. These budgets exclude mapped checkpoint
pages, request buffers and allocator/context overhead. CUDA quant-only uploads
create no decoded auxiliary copies. Borrowing requests are released before
weights, and the GPU outlives both. The layouts are ready for ownership by a
future dedicated GLM backend model; they are not CPU model weight handles.

## Independent qualification

The original operator captures cover attention and FFN parameters from blocks
0 and 44, at batches 1, 9, 127, 128 and 1,024: 20 actual-weight cases, each
captured twice with exact repeated results. Native execution binds weights
through the production six-shard GGUF reader. Only original residual and branch
inputs enter the replay; reference intermediates never feed native execution.

All projection partials, squared sums, coefficients, collapsed/normalized inputs
and postmix outputs match bit for bit. The replay checks 150,778,976 output
words plus 105,594,880 input words for immutability, totaling 256,373,856 exact
comparisons. Both optional norm modes are exercised. Invalid plans preserve
checked outputs, and CUDA memcheck reports zero errors.

The pre-coefficient capture uses the original coefficient kernel with the same
projection splits as the regular pre operation. An earlier diagnostic capture
called the delayed wrapper, which selects a different split profile. Correcting
that capture changed only saved pre coefficients; all weights and every other
saved boundary remained unchanged. The committed generator reproduces all 20
final cases byte for byte. No native result generated a reference value.

The coupled attention replay starts with expanded residual streams, computes
mHC normalization natively, passes that resident result into KDA, and mixes the
native attention branch back into residual streams. Both blocks pass full
nine-token prefill, `[2,7]`, `[3,3,3]`, `[7,1,1]`, `[6,3]`, and nine one-token
schedules. Histories and recurrent states start nonzero. All 56,428,288 compared
words match exactly, including normalized input, KDA intermediates, F32 state,
BF16 history and residual output. Its CUDA memcheck also reports zero errors.
The earlier KDA-only checkpoint replay remains exact.

The regular CUDA suite additionally contains a model-free, procedural-input
fixture with original SM120 operator outputs. It checks 30,296 exact values,
input immutability, buffer canaries, invalid-plan rejection and both norm modes.
Its header reproduces byte for byte. Synthetic CPU tests cover Q8→F32 projection
preparation, gamma rounding, budgets, payload bounds, malformed parameters,
every upload failure and every attention-composition submission failure. New
preparation/orchestration code passes ASan/UBSan with leak detection enabled.

The final clean build has zero warnings. Standard tests, the full CUDA suite
including the new procedural case, and diagnostic isolation pass. The fixture
also passes CUDA memcheck and synccheck with zero errors. A warm nine-token
attention-transition trace records 18 kernels on one stream, zero recorded copy
bytes and no allocation API calls. There is one explicit final stream sync and
an additional unattributed context-sync activity after the last kernel, with no
synchronization between operations. These are residency diagnostics, not
statistical latency or vLLM parity measurements.

Source, reference and verification records are in
`glm53-cuda-mhc-attention.json`.

## Reproduction

Select the intended GPU through `CUDA_VISIBLE_DEVICES` for every GPU command.
Use the qualified GLM-enabled vLLM/GGUF plugin environment with a supported CUDA
compiler on `CUDA_HOME` and `PATH`:

```sh
python test/reference_glm_mhc.py --model /path/to/first-shard.gguf --out /tmp/mhc-ref
python test/reference_glm_kda_layer.py --model /path/to/first-shard.gguf --out /tmp/attention-ref --mhc-reference /tmp/mhc-ref/reference
make BN_ENABLE_CUDA=1 BN_ENABLE_MKL_VML=1 MKL_VML_ROOT=/path/to/mkl test_cuda_glm_mhc test_cuda_glm_mhc_kda
./test_cuda_glm_mhc /path/to/first-shard.gguf /tmp/mhc-ref/reference
./test_cuda_glm_mhc_kda /path/to/first-shard.gguf /tmp/attention-ref/reference /tmp/mhc-ref/reference
```

`test/reference_cuda_mhc.py --header PATH --metadata PATH` regenerates the
procedural fixture; `./test_cuda_backend --mhc-reference` selects its native
replay after the CUDA test binary is built.

This is a materialized attention-transition precision profile, including actual
input/output GEMMs and mHC. It does not establish equality with vLLM's default
deferred/fused inter-layer mHC path, optimized `chunk_kda`, a whole GGUF model
wrapper, CPU BF16 rounding, long-context GLM inference or vLLM performance
parity. Those remain acceptance work under the original active goal.


The subsequent [default deferred mHC qualification](glm53-cuda-deferred-mhc.md)
adds the original fused small-batch path and its KDA/FFN-entry composition. The
materialized precision profile and evidence above remain unchanged.
