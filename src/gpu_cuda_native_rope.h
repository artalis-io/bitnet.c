#ifndef BN_GPU_CUDA_NATIVE_ROPE_H
#define BN_GPU_CUDA_NATIVE_ROPE_H
#include <cuda_runtime.h>
int bn_cuda_native_rope(cudaStream_t,void*,const void*,const void*,const void*,
    int,int,int,int,int,int,int,int,int,float);
int bn_cuda_native_frequency(cudaStream_t,float*,float*,int,int,float,float,float,float);
#endif
