#ifndef BN_GPU_CUDA_COMPRESS_H
#define BN_GPU_CUDA_COMPRESS_H
#include <cuda_runtime.h>
int bn_cuda_compress(cudaStream_t,void**,int,int,int,int,int,int,int,int,float);
#endif
