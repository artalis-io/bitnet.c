#ifndef BN_GPU_CUDA_NATIVE_ATTENTION_H
#define BN_GPU_CUDA_NATIVE_ATTENTION_H
#include <cuda_runtime.h>
#include <cuda_fp16.h>
struct BnCudaNativeLatentAttentionArgs {
    const half *q,*ring,*kv,*pc,*pr;
    const int *bt,*ids;
    const float *sinks,*freq;
    half *out;
    float *ml,*av;
    int R,T,rate,dense,W,pos,beg,floor,epp,S;
};
int bn_cuda_native_latent_attention_init();
int bn_cuda_native_latent_attention(cudaStream_t,const BnCudaNativeLatentAttentionArgs*);
#endif
