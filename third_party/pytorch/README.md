The 32-element unstable key/value sorting network in
`src/gpu_cuda_native_moe.cu` adapts `SortUtils.cuh` from PyTorch commit
`08187d9e0fba026dc8217405802ab5381dc88d90`. The BSD license is retained here.
It preserves the source MoE assignment order at small sort sizes; larger
assignment arrays use stable CUB sorting. Qualification must compare the
complete GPU argsort permutation, not only sorted expert IDs.
