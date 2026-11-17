/* Candidate-source arithmetic from pinned exllamav3 bd1f5191 dsv4.py.
 * Device mechanics are backend-private; buffers belong to the request. */
#include "gpu_cuda_native_candidates.h"
#include <cuda_fp16.h>
#include <cstddef>
namespace bn_cuda_native_candidates_detail {
__global__ void maxima(const half*s,half*blk,unsigned char*drop,int R,int T,int stride,int bs,int pos,int rate) {
    int blocks=T/8+1;size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(size_t)R*blocks)return;
    int row=(int)(i/blocks),b=(int)(i%blocks);half best=__ushort_as_half(0xfc00);
    /* Source amax chooses the later payload on equal values, including -0. */
    for(int j=0;j<8;j++){int col=b*8+j;half v=col<T?s[(size_t)row*stride+col]:__ushort_as_half(0xfc00);
        if(!(__half2float(best)>__half2float(v)))best=v;}
    int newest=(pos+row+1)/rate-1;if(newest<0)newest=0;
    blk[(size_t)row*bs+b]=b==newest/8?__ushort_as_half(0x7c00):best;drop[i]=1;
}
__global__ void scatter(const half*blk,const int*indices,unsigned char*drop,int R,int blocks,int stride) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)R*2048)return;
    int row=(int)(i/2048),b=indices[i];
    if(b>=0&&b<blocks)drop[(size_t)row*blocks+b]=__half_as_ushort(blk[(size_t)row*stride+b])==0xfc00?1:0;
}
}
int bn_cuda_native_candidates_begin(cudaStream_t st,const void*s,void*blk,void*drop,int R,int T,int stride,int bs,int pos,int rate) {
    size_t n=(size_t)R*(T/8+1);
    bn_cuda_native_candidates_detail::maxima<<<(unsigned)((n+255)/256),256,0,st>>>((const half*)s,(half*)blk,(unsigned char*)drop,R,T,stride,bs,pos,rate);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
int bn_cuda_native_candidates_end(cudaStream_t st,const void*blk,const void*indices,void*drop,int R,int blocks,int stride) {
    size_t n=(size_t)R*2048;
    bn_cuda_native_candidates_detail::scatter<<<(unsigned)((n+255)/256),256,0,st>>>((const half*)blk,(const int*)indices,(unsigned char*)drop,R,blocks,stride);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
