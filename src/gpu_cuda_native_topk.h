#ifndef BN_GPU_CUDA_NATIVE_TOPK_H
#define BN_GPU_CUDA_NATIVE_TOPK_H
#include <cuda_runtime.h>
int bn_cuda_native_index_topk(cudaStream_t,const void*,void*,void*,void*,void*,int,int,int,int,int,int,const void*active=nullptr);
#endif
