#include "gpu_cuda_native_candidate_use.h"
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
namespace bn_cuda_native_candidate_use_detail {
template<int Threads>__device__ int scan(int v,int*shared) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int d=1;d<32;d*=2){int a=__shfl_up_sync(0xffffffff,v,d);if(lane>=d)v+=a;}
    if(lane==31)shared[warp]=v;__syncthreads();
    if(warp==0){int a=lane<Threads/32?shared[lane]:0;for(int d=1;d<32;d*=2){int b=__shfl_up_sync(0xffffffff,a,d);if(lane>=d)a+=b;}if(lane<Threads/32)shared[lane]=a;}
    __syncthreads();return v+(warp?shared[warp-1]:0);
}
__global__ void local_union(const unsigned char*drop,int*counts,int*offsets,int*tiles,int R,int T) {
    __shared__ int sums[8];int blocks=T/8+1,b=blockIdx.x*256+threadIdx.x,n=0;
    if(b<blocks){bool kept=false;for(int row=0;row<R;row++)kept|=drop[(size_t)row*blocks+b]==0;if(kept){n=T-b*8;if(n>8)n=8;}counts[b]=n;}
    int incl=scan<256>(n,sums);if(b<blocks)offsets[b]=incl-n;
    if(threadIdx.x==255)tiles[blockIdx.x]=incl;
}
__global__ void global_union(int*tiles,int*count,int N,int capacity) {
    __shared__ int sums[32];int t=threadIdx.x,v=t<N?tiles[t]:0,incl=scan<1024>(v,sums);
    if(t<N)tiles[t]=incl-v;if(t==N-1)*count=incl<=capacity?incl:0;
}
template<bool Vector>__global__ void gather(const half*keys,const int*bt,const int*counts,const int*offsets,const int*tiles,int*ids,half*out,int capacity,int epp,int blocks) {
    int lane=threadIdx.x%32,b=blockIdx.x*8+threadIdx.x/32;if(b>=blocks)return;
    int n=counts[b],start=offsets[b]+tiles[b/256];if(n<1||n>8||start<0||start>capacity-n)return;
    if(lane<n)ids[start+lane]=b*8+lane;
    if constexpr(Vector){for(int i=lane;i<n*16;i+=32){int logical=b*8+i/16;size_t physical=epp?(size_t)bt[logical/epp]*epp+logical%epp:(size_t)logical;*(uint4*)(out+(size_t)(start+i/16)*128+(i%16)*8)=*(const uint4*)(keys+physical*128+(i%16)*8);}}
    else {for(int i=lane;i<n*128;i+=32){int logical=b*8+i/128;size_t physical=epp?(size_t)bt[logical/epp]*epp+logical%epp:(size_t)logical;out[(size_t)(start+i/128)*128+i%128]=keys[physical*128+i%128];}}
}
__global__ void mask(const unsigned char*drop,const int*ids,const int*count,half*s,int R,int blocks,int stride,int pos,int rate) {
    int col=blockIdx.y*128+threadIdx.x,row=blockIdx.x,N=*count;if(col>=N||row>=R)return;
    int ent=ids[col];if(drop[(size_t)row*blocks+ent/8]||ent>=(pos+row+1)/rate)s[(size_t)row*stride+col]=__ushort_as_half(0xfc00);
}
__global__ void remap(const int*ids,const int*count,int*out,int R) {
    size_t i=(size_t)blockIdx.x*256+threadIdx.x;if(i>=(size_t)R*512)return;int j=out[i],N=*count;out[i]=j>=0&&j<N?ids[j]:-1;
}
}
int bn_cuda_native_candidate_gather(cudaStream_t st,const void*drop,const void*keys,const void*bt,void*counts,void*offsets,void*tiles,void*ids,void*count,void*out,int R,int T,int capacity,int epp,int nt) {
    using namespace bn_cuda_native_candidate_use_detail;
    local_union<<<nt,256,0,st>>>((const unsigned char*)drop,(int*)counts,(int*)offsets,(int*)tiles,R,T);if(cudaGetLastError()!=cudaSuccess)return -1;
    global_union<<<1,1024,0,st>>>((int*)tiles,(int*)count,nt,capacity);if(cudaGetLastError()!=cudaSuccess)return -1;
    int blocks=T/8+1;
    if((((uintptr_t)keys|(uintptr_t)out)&15u)==0)gather<true><<<(blocks+7)/8,256,0,st>>>((const half*)keys,(const int*)bt,(const int*)counts,(const int*)offsets,(const int*)tiles,(int*)ids,(half*)out,capacity,epp,blocks);
    else gather<false><<<(blocks+7)/8,256,0,st>>>((const half*)keys,(const int*)bt,(const int*)counts,(const int*)offsets,(const int*)tiles,(int*)ids,(half*)out,capacity,epp,blocks);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
int bn_cuda_native_candidate_mask(cudaStream_t st,const void*drop,const void*ids,const void*count,void*s,int R,int T,int stride,int pos,int rate) {
    int capacity=R*16384;if(capacity>T)capacity=T;
    bn_cuda_native_candidate_use_detail::mask<<<dim3(R,(capacity+127)/128),128,0,st>>>((const unsigned char*)drop,(const int*)ids,(const int*)count,(half*)s,R,T/8+1,stride,pos,rate);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
int bn_cuda_native_candidate_remap(cudaStream_t st,const void*ids,const void*count,void*out,int R) {
    bn_cuda_native_candidate_use_detail::remap<<<(R*512+255)/256,256,0,st>>>((const int*)ids,(const int*)count,(int*)out,R);return cudaGetLastError()==cudaSuccess?0:-1;
}
