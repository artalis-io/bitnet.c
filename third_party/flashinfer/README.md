The SM120 GLM NoPE sparse MLA adapter uses FlashInfer commit
`beabf5004585b03da60e927bbb803acbaa62fe36`, matching the independent reference
package 0.7.2.post1. Its CUTLASS and CCCL submodules are pinned respectively to
`b46b16d003484063bca4ed365e44095c4c6ed633` and
`16bd510c9b712e82b0ab6cbb630d8e29ba1f7116`.

CUDA builds fetch these sources into ignored `.deps/flashinfer` using
`patches/fetch-cuda-mla.sh`. An offline build may prepare that pinned checkout
or set `MLA_ROOT`. The fetcher rejects modified tracked sources and different
revisions. CPU-only builds do not fetch or compile the dependency.

The adapter instantiates only GLM53_NOPE, 64-head, 512-latent FP8 decode and
swapAB prefill kernels and the decode merge. It calls templates through a
private C ABI without Python, Torch, TVM, upstream host dispatch, or upstream
mutable configuration caches. Kernel attributes are set at backend creation;
workspace belongs to the request. The adapter uses a separate C++17 CUDA
translation unit; engine code remains C11 and the main CUDA unit remains C++11.

Vendor include diagnostics 177 (unused constants) and 549 (the host-pass return
of an unused NVFP4 helper) are scoped to vendor headers. No NVFP4 helper is
instantiated by this adapter, and attention arithmetic is unchanged. License
notices for FlashInfer, NVIDIA sparse MLA, CUTLASS and CCCL are retained here.
