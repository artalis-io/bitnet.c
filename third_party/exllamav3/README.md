The vector4 RMSNorm implementation in `src/gpu_cuda_native_norm.cuh` adapts
`exllamav3/exllamav3_ext/norm.cu` from commit
`bd1f519172406a9560bd5c90e61000691bb69a07` (Copyright 2025 Turboderp).
The upstream MIT license is retained in this directory.

The port keeps the upstream fast-math arithmetic local to this operator using
explicit PTX operations. It does not change global CUDA compiler flags.
Independent original-kernel captures and qualification are documented in
`docs/deepseek-v41-1017-native-rmsnorm.json`.

HC4 carry mixing and residual application in `src/gpu_cuda_hc4.cu` adapt
`hc_mix.cu` and `hc_lanes.cuh` from the same commit and license. Only this
translation unit uses `--use_fast_math`. Immutable F32 projection weights
are rounded to FP16 on read for at most 32 rows; traffic remains F32.

The deterministic router in `src/gpu_cuda_router.cu` and
`src/gpu_cuda_router_det.cuh` adapts `routing.cu`, `routing_gemm.cu`,
`det_gemm.cuh` and the selected warp reduction from `reduction.cuh`,
under the same upstream commit and MIT license. Router fast-math flags
are confined to its translation unit. Immutable derived layouts live
in backend-owned router handles, with request-owned execution workspace.

The folded EXL3 batch reconstruction and adaptive FP16 partial-accumulation
GEMMs in `src/gpu_cuda_exl3_fold.cu` and its private codec headers adapt
`quant/reconstruct.cu`, `quant/exl3_dq.cuh`, `quant/codebook.cuh`,
`quant/hadamard_inner.cuh`, `ptx.cuh`, `util.cuh`, and `hgemm_f16acc.cu`
from the same commit and license. Fast math is confined to that translation
unit. The batch wrapper passes borrowed immutable weight pointers as kernel
arguments; no device pointer-table allocation is needed during execution.
GEMM policy is chosen during explicit request setup, using source shape rules
for FP32 cuBLAS or FP16 partial accumulators flushed every 32 terms.
Qualification and library-version limits are recorded in
`docs/deepseek-v41-1017-native-folded-ffn.json`.

The fixed-order native MoE slot reduction in `src/gpu_cuda_native_moe.cu`
adapts `exl3_moe_gather_kernel` from `quant/exl3_moe.cu` at the same commit.
Its narrow assignment metadata is range-bounded; source F32 multiplication,
FMA accumulation and final add retain local fast-math semantics. CPU tier
planning in `src/moe_policy.c` follows `plan_groups`/`batch_cap` from
`modules/moe_batch_recon.py`, including the upstream two-slab cap heuristic.

The direct fused EXL3 expert tier in `src/gpu_cuda_exl3_fused.cu` and
`src/cuda_exl3_fused/` adapts the original `exl3_moe_kernel.cuh`,
`exl3_gemm_inner.cuh`, Hadamard, codebook, PTX and utility headers from the
same commit. Backend-private metadata binds borrowed immutable expert
handles; locks, scheduler and temporary slabs are request-owned. The port
uses cooperative launches and validates co-residency. Full-tier group width
is explicit because it changes split-K rounding; shrinking a cache window
must retain that width. Route weighting remains inside the output Hadamard.
Original normal/cooperative launches and independent bounded-window
composition are captured in the native fused-FFN qualification evidence.

The output Hadamard transpose adds a post-read warp synchronization before
shared scratch is reused. Racecheck reproduces the missing synchronization
in the unchanged original helper; the port keeps original arithmetic and
original reference captures while closing that warp-local write-after-read
hazard. Scheduler tickets and barrier senses retain their source lifecycle.
