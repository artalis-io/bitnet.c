CUDA attention uses FlashAttention 2 commit
`506341a143fcabd4bb79052a7605ada727d6b3f5` from
https://github.com/vllm-project/flash-attention and its pinned CUTLASS submodule
`62750a2b75c802660e4894434dc55e839f322277`. This is the dependency pinned by
the vLLM reference commit `8244d1c44`.

`make BN_ENABLE_CUDA=1` fetches these sources once into ignored `.deps/`.
For an offline build, prepare that directory with
`sh patches/fetch-cuda-attention.sh`, or set `FA2_ROOT` to an existing pinned
checkout. The script refuses a different revision or modified tracked files.
CPU-only builds do not fetch or compile this dependency.

Only the BF16, head-size-256 forward kernels are instantiated. Dropout, ALiBi,
soft caps, and local attention are excluded. The adapter includes the kernel
headers directly, without Torch headers, Torch runtime, upstream host dispatch,
or process-exiting error handlers. The CUDA adapter is compiled as C++17 with
the upstream CUDA math flags; the engine remains C11. The existing CUDA backend
continues to use its C++11 compilation flags.

CUDA 13.5 deprecates vector typedefs used by this pinned CUTLASS version.
Deprecation diagnostics are suppressed only for this third-party translation
unit. The kernel math is unmodified. Upstream BSD license notices are included
here and in `../cutlass/LICENSE` for redistribution with the binary.
