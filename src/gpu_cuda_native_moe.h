#ifndef BN_GPU_CUDA_NATIVE_MOE_H
#define BN_GPU_CUDA_NATIVE_MOE_H
#include <cuda_runtime.h>
struct BnCudaNativeMoEGroup {int starts[16],counts[16];};
size_t bn_cuda_native_moe_sort_bytes(int assignments);
int bn_cuda_native_moe_index(cudaStream_t,void*,size_t,const void*,const void*,void*,void*,void*,void*,void*,void*,void*,void*,int,int);
int bn_cuda_native_moe_group(cudaStream_t,const BnCudaNativeMoEGroup*,const void*,const void*,void*,int,int,int,int,int);
int bn_cuda_native_moe_reduce(cudaStream_t,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,int,int,int,int,int);
#endif
