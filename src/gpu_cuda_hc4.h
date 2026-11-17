#ifndef BN_GPU_CUDA_HC4_H
#define BN_GPU_CUDA_HC4_H
#include <cuda_runtime.h>
/* Validated internal launch interface; no allocation/copy/synchronization. */
int bn_cuda_hc4_mix_launch(cudaStream_t,const float*,const float*,const float*,const float*,float*,float*,float*,float*,const float*,float*,int,int,int,float,float,int);
int bn_cuda_hc4_apply_launch(cudaStream_t,float*,const void*,const float*,const float*,int,int,int);
#endif
