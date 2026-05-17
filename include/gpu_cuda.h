#ifndef BN_GPU_CUDA_H
#define BN_GPU_CUDA_H

#ifdef BN_ENABLE_CUDA

#include "gpu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

BnGPUBackend *bn_gpu_cuda_create(void);
void bn_gpu_cuda_destroy(BnGPUBackend *gpu);

#ifdef __cplusplus
}
#endif

#endif // BN_ENABLE_CUDA

#endif // BN_GPU_CUDA_H
