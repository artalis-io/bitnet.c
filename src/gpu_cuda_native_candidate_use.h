#ifndef BN_GPU_CUDA_NATIVE_CANDIDATE_USE_H
#define BN_GPU_CUDA_NATIVE_CANDIDATE_USE_H
#include <cuda_runtime.h>
int bn_cuda_native_candidate_gather(cudaStream_t,const void*,const void*,const void*,void*,void*,void*,void*,void*,void*,int,int,int,int,int);
int bn_cuda_native_candidate_mask(cudaStream_t,const void*,const void*,const void*,void*,int,int,int,int,int);
int bn_cuda_native_candidate_remap(cudaStream_t,const void*,const void*,void*,int);
#endif
