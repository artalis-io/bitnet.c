#ifndef BN_GPU_CUDA_NATIVE_INDEX_H
#define BN_GPU_CUDA_NATIVE_INDEX_H
#include <cuda_runtime.h>
int bn_cuda_native_index_score(cudaStream_t,const void*,const void*,const void*,const void*,void*,int,int,int,int,int,int,const void*active=nullptr);
#endif
