/* Source GPT-J pair-norm/rotation, exllamav3 bd1f5191 MIT. No ATen/graphs. */
#include "gpu_cuda_native_rope.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
namespace bn_cuda_native_rope_detail {
#include "gpu_cuda_native_rope_kernel.cuh"
}
int bn_cuda_native_rope(cudaStream_t stream,void*data,const void*freq,const void*weight,
    const void*positions,int R,int H,int D,int stride,int offset,int rotary,
    int rotate_offset,int pos,int bf16,float eps) {
    using namespace bn_cuda_native_rope_detail;
    half*q=(half*)data+offset;int thr=((D/2+31)/32)*32;
    int parallel=1024/thr;if(parallel>H)parallel=H;
    int groups=(H+parallel-1)/parallel;if(R>=32)groups=1;
#define ARGS q,q,nullptr,nullptr,(const float*)freq,1,R,H,0,D,stride,0,rotary,pos,nullptr,(const uint32_t*)positions,1.f,weight,nullptr,eps,0.f,false,0,0.f,0,1,1,rotate_offset
    if(bf16)rope_kernel<ROPESTYLE_GPTJ,true><<<dim3(R,1,groups),dim3(thr,parallel),0,stream>>>(ARGS);
    else rope_kernel<ROPESTYLE_GPTJ,false><<<dim3(R,1,groups),dim3(thr,parallel),0,stream>>>(ARGS);
#undef ARGS
    return cudaGetLastError()==cudaSuccess?0:-1;
}
