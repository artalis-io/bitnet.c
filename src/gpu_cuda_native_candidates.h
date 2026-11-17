#ifndef BN_GPU_CUDA_NATIVE_CANDIDATES_H
#define BN_GPU_CUDA_NATIVE_CANDIDATES_H
#include <cuda_runtime.h>
int bn_cuda_native_candidates_begin(cudaStream_t,const void*,void*,void*,int,int,int,int,int,int);
int bn_cuda_native_candidates_end(cudaStream_t,const void*,const void*,void*,int,int,int);
#endif
