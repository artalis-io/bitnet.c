#ifndef BN_GPU_CUDA_EXL3_FOLD_H
#define BN_GPU_CUDA_EXL3_FOLD_H
#include <cuda_runtime.h>
struct BnCudaFoldPointers {const void *packed[16],*pre[16],*post[16];};
int bn_cuda_fold_probe(int sms,float*ratio);
int bn_cuda_fold_prepare();
int bn_cuda_fold_kind(int B,int M,int K,int N,int cc,int sms,int enabled);
int bn_cuda_fold_reconstruct(cudaStream_t,const BnCudaFoldPointers*,void*,int,int,int);
int bn_cuda_fold_gemm(cudaStream_t,int,const void*,const void*,void*,int,int,int,int,int);
#endif
