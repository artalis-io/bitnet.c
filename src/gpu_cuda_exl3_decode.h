#ifndef BN_GPU_CUDA_EXL3_DECODE_H
#define BN_GPU_CUDA_EXL3_DECODE_H
#include <cuda_runtime.h>
struct BnCudaDecodeEXL3Pointers {
    const void *packed[3][48], *pre[3][48], *post[3][48];
    int experts[48], count;
};
struct BnCudaDecodeEXL3IO {
    void *x,*indices,*weights,*tables,*indices64;
    void *had_g,*had_u,*gate,*up,*act,*down,*counters,*out;
};
int bn_cuda_exl3_decode_info(int R,int D,int H,int K,int *info);
int bn_cuda_exl3_decode_launch(cudaStream_t,const BnCudaDecodeEXL3Pointers*,
    const BnCudaDecodeEXL3IO*,int R,int D,int H,int E,int K,float limit);
#endif
