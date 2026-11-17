#ifndef BN_GPU_CUDA_ROUTER_H
#define BN_GPU_CUDA_ROUTER_H
#include <cuda_runtime.h>
#include <stdint.h>
int bn_cuda_router_prepare_launch(cudaStream_t,const void*,void*,float*,int,int);
int bn_cuda_router_run_launch(cudaStream_t,const void*,const void*,const void*,const float*,const void*,void*,int32_t*,void*,void*,void*,float*,float*,int,int,int,int,float);
#endif
