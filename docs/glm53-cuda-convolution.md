# GLM F32-weight resident convolution

The CUDA backend exposes `causal_conv_f32_weight_bf16_resident` for causal
convolution with immutable F32 weights and BF16 input, history, and output.
It uses `BnGPUCausalConvResidentPlan`, kernels 2/3/4, chronological history
`[kernel-1,dim]`, and channel-major weights `[dim,kernel]`.

Each channel accumulates ordered F32 FMAs, applies SiLU to that F32 sum, and
rounds the output to BF16. Weights retain their F32 low bits. The previous
`causal_conv_bf16_resident` operation retains its BF16 product-rounding contract
and rejects F32 weights. Callers choose the operation explicitly.

Both operations tile over channels and eight-token windows. Input and initial
history remain immutable; the caller commits the separate next-history view
after successful completion. Writable views must be disjoint from inputs and
each other. The new operation reuses resident request storage without allocating,
transferring activations through host memory, or introducing an internal fence.
These properties follow from implementation inspection; this stage does not
claim a measured end-to-end transfer or latency improvement.

## Independent reference

`test/reference_cuda_f32_weight_conv.py` invokes the unchanged installed vLLM
causal convolution operators. Full prefill and repeated one-token updates must
agree before reference values are emitted. The original operator's source hash,
Torch version, fixture hashes, and tested source hashes are recorded in
`glm53-cuda-convolution.json`.

Thirty cases cover widths 1, 31, 257, 768, and GLM's concatenated Q/K/V width
24,576, token counts 1/5/17/65, kernels 2/3/4, and two deterministic seeds.
F32 weights deliberately contain low bits that BF16 conversion would discard.
The native test checks full prefill, split prefill with carried history, and
one-token updates against the same original-reference byte hashes: 714 launches
cover 9,423,054 output/history words. It also checks read-only input/history,
canaries, and validation rejection without output mutation.

The protected clean build has zero warnings. The standard suite, complete CUDA
suite, and CUDA diagnostic isolation pass. Compute Sanitizer reports zero
errors for the new resident operation. A second original-reference capture
reproduced all 61 data/metadata files and the generated header exactly.

## Scope

This qualifies a resident convolution operation on the designated Blackwell
GPU. It does not register GLM model anatomy, wire its whole transformer, qualify
native KDA recurrence or output normalization, or establish vLLM performance
parity. Independent CPU convolution and recurrent BF16 boundary differences
remain open; a native CUDA arithmetic match does not qualify the CPU path.

The preceding external full-chain experiment under
`/tmp/bitnet-goal-glm-chain` found four BF16 convolution mismatches at the GLM
shape and additional recurrent output rounding differences. Changing the CPU
exponential to `expf` or using FMA alone did not remove those four mismatches.
An isolated native CUDA probe matched all captured F32 convolution shadow
values and BF16 outputs, motivating this separate operation. The independent
full-chain captures remain unchanged.

The next resident KDA stage must retain F32 Q/K normalization and per-key
decay, F32 recurrent state, and the BF16 core-output boundary before RMS sigmoid
gating. Existing GDN preparation and recurrence have different contracts.
The original GLM forward path has no activation between `f_a`/`f_b` or
`g_a`/`g_b`, but it materializes projection outputs; algebraically folding those
pairs would remove intermediate rounding and requires separate qualification.
