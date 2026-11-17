/* Source arithmetic: exllamav3 bd1f5191 dsa_triton.py, MIT license. */
#include "gpu_cuda_native_index.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
namespace bn_cuda_native_index_detail {
__device__ __forceinline__ unsigned pair(const half*p){return (unsigned)__half_as_ushort(p[0])|((unsigned)__half_as_ushort(p[1])<<16);}
__device__ __forceinline__ void mma(float*z,const unsigned*a,const unsigned*b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":"+f"(z[0]),"+f"(z[1]),"+f"(z[2]),"+f"(z[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#endif
}
template<bool Few,bool Vector,bool Dynamic>__global__ void score(const half*q,const half*w,const half*k,const int*bt,half*out,
    int R,int T,int stride,int pos,int rate,int epp,const int*active) {
    constexpr int N=Few?64:128;__shared__ __align__(16) half kt[N][136];__shared__ float products[Few?32:1][Few?64:1];__shared__ __align__(16) half qt[32][136];__shared__ float wh[32];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,key_base=blockIdx.y*N;
    if constexpr(Dynamic){int n=*active;if(n<0||n>T||key_base>=n)return;T=n;}
    if(Vector){for(int t=threadIdx.x;t<N*16;t+=256){int n=key_base+t/16,d=t%16*8;size_t row=n;if(n<T&&epp)row=(size_t)bt[n/epp]*epp+n%epp;uint4 v=n<T?*(const uint4*)(k+row*128+d):make_uint4(0,0,0,0);*(uint4*)(&kt[t/16][d])=v;}}
    else {    for(int t=threadIdx.x;t<N*128;t+=256){int n=key_base+t/128,d=t%128;size_t row=n;
        if(n<T&&epp)row=(size_t)bt[n/epp]*epp+n%epp;
        kt[t/128][d]=n<T?k[row*128+d]:__float2half_rn(0.f);}}
    if(Few){if(Vector){for(int t=threadIdx.x;t<32*16;t+=256)*(uint4*)(&qt[t/16][t%16*8])=*(const uint4*)(q+((size_t)blockIdx.x*32+t/16)*128+t%16*8);}
        else {for(int t=threadIdx.x;t<32*128;t+=256)qt[t/128][t%128]=q[((size_t)blockIdx.x*32+t/128)*128+t%128];}if(threadIdx.x<32)wh[threadIdx.x]=__half2float(w[blockIdx.x*32+threadIdx.x]);}
    __syncthreads();
    if(Few) {
        int r=blockIdx.x;
        for(int base_h=0;base_h<32;base_h+=16) {
            float z[4]={0,0,0,0};
            #pragma unroll
            for(int off=0;off<128;off+=16){unsigned a[4],b[2];
                for(int i=0;i<4;i++)a[i]=pair(&qt[base_h+lane/4+(i%2)*8][off+2*(lane%4)+(i/2)*8]);
                for(int i=0;i<2;i++)b[i]=pair(&kt[warp*8+lane/4][off+2*(lane%4)+i*8]);
                mma(z,a,b);}
            for(int i=0;i<4;i++){int h=base_h+lane/4+(i/2)*8,n=warp*8+2*(lane%4)+i%2;products[h][n]=fmaxf(z[i],0.f);}
        }
        __syncthreads();
        if(threadIdx.x<N){int n=threadIdx.x,key=key_base+n;float p[8];
            /* Source ptxas contracts head0*w0 + head8*w8 into an FMA
             * with the second product rounded; likewise heads16/24. */
            for(int h=0;h<8;h++) {
                float a=fmaf(products[h][n],wh[h],products[h+8][n]*wh[h+8]);
                float b=fmaf(products[h+16][n],wh[h+16],products[h+24][n]*wh[h+24]);
                p[h]=a+b;
            }
            for(int d=4;d;d/=2)for(int h=0;h<d;h++)p[h]+=p[h+d];
            int bound=(pos+r+1)/rate;if(bound>T)bound=T;
            if(key<T)out[(size_t)r*stride+key]=__float2half_rn(key<bound?p[0]*(1.f/64):-INFINITY);}
    } else {
        int row_base=blockIdx.x*32+(warp/4)*16,nbase=(warp%4)*32;float acc[4][4]={};
        for(int h=0;h<32;h++) {
            if(Vector){for(int t=threadIdx.x;t<32*16;t+=256){int r=blockIdx.x*32+t/16;uint4 v=r<R?*(const uint4*)(q+((size_t)r*32+h)*128+t%16*8):make_uint4(0,0,0,0);*(uint4*)(&qt[t/16][t%16*8])=v;}}
            else {            for(int t=threadIdx.x;t<32*128;t+=256){int r=blockIdx.x*32+t/128;qt[t/128][t%128]=r<R?q[((size_t)r*32+h)*128+t%128]:__float2half_rn(0.f);}}
            if(threadIdx.x<32){int r=blockIdx.x*32+threadIdx.x;wh[threadIdx.x]=r<R?__half2float(w[r*32+h]):0.f;}
            __syncthreads();
            float z[4][4]={};
            #pragma unroll
            for(int off=0;off<128;off+=16){unsigned a[4];for(int i=0;i<4;i++){a[i]=pair(&qt[(warp/4)*16+lane/4+(i%2)*8][off+2*(lane%4)+(i/2)*8]);}
                for(int n=0;n<4;n++){unsigned b[2];for(int i=0;i<2;i++)b[i]=pair(&kt[nbase+n*8+lane/4][off+2*(lane%4)+i*8]);mma(z[n],a,b);}}
            for(int n=0;n<4;n++)for(int i=0;i<4;i++){acc[n][i]=fmaf(fmaxf(z[n][i],0.f),wh[(warp/4)*16+lane/4+(i/2)*8],acc[n][i]);}
            __syncthreads();
        }
        for(int n=0;n<4;n++)for(int i=0;i<4;i++){int r=row_base+lane/4+(i/2)*8,key=key_base+nbase+n*8+2*(lane%4)+i%2;
            if(r<R&&key<T){int bound=(pos+r+1)/rate;if(bound>T)bound=T;out[(size_t)r*stride+key]=__float2half_rn(key<bound?acc[n][i]*(1.f/64):-INFINITY);}}
    }
}
}
int bn_cuda_native_index_score(cudaStream_t s,const void*q,const void*w,const void*k,const void*bt,void*out,int R,int T,int stride,int pos,int rate,int epp,const void*active) {
    using namespace bn_cuda_native_index_detail;
    /* Preserve the public four-byte alignment contract for sliced views. */
    bool vector=(((uintptr_t)q|(uintptr_t)k)&15u)==0;
    dim3 grid(R<=4?R:(R+31)/32,R<=4?(T+63)/64:(T+127)/128);
#define BN_INDEX_LAUNCH(F,V,D) score<F,V,D><<<grid,256,0,s>>>((const half*)q,(const half*)w,(const half*)k,(const int*)bt,(half*)out,R,T,stride,pos,rate,epp,(const int*)active)
#define BN_INDEX_ROUTE(D) do {if(R<=4){if(vector){BN_INDEX_LAUNCH(true,true,D);}else{BN_INDEX_LAUNCH(true,false,D);}}else{if(vector){BN_INDEX_LAUNCH(false,true,D);}else{BN_INDEX_LAUNCH(false,false,D);}}}while(0)
    if(active){BN_INDEX_ROUTE(true);}else{BN_INDEX_ROUTE(false);}
#undef BN_INDEX_ROUTE
#undef BN_INDEX_LAUNCH
    return cudaGetLastError()==cudaSuccess?0:-1;
}
