/* Source arithmetic: exllamav3 bd1f5191 dsa_triton.py, MIT.
 * License: third_party/exllamav3/LICENSE. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include "gpu_cuda_native_attention.h"
namespace bn_cuda_native_attention_detail {
static __device__ __forceinline__ void mma(float*z,unsigned*a,unsigned*b){
#if __CUDA_ARCH__ >= 800
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":"+f"(z[0]),"+f"(z[1]),"+f"(z[2]),"+f"(z[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#else
 (void)z;(void)a;(void)b;
#endif
}

static __device__ __forceinline__ void load4(unsigned*a,const half*p) {
#if __CUDA_ARCH__ >= 800
    unsigned addr=(unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr));
#else
    (void)p;for(int i=0;i<4;i++)a[i]=0;
#endif
}
static __device__ __forceinline__ void load2(unsigned*b,const half*p) {
#if __CUDA_ARCH__ >= 800
    unsigned addr=(unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(b[0]),"=r"(b[1]):"r"(addr));
#else
    (void)p;b[0]=b[1]=0;
#endif
}
static __device__ __forceinline__ void load2_trans(unsigned*b,const half*p) {
#if __CUDA_ARCH__ >= 800
    unsigned addr=(unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(b[0]),"=r"(b[1]):"r"(addr));
#else
    (void)p;b[0]=b[1]=0;
#endif
}
static __device__ __forceinline__ float expn(float x){float y;x*=1.4426950408889634f;asm("ex2.approx.f32 %0,%1;":"=f"(y):"f"(x));return y;}
static __device__ __forceinline__ float divide(float a,float b){float y;asm("div.full.f32 %0,%1,%2;":"=f"(y):"f"(a),"f"(b));return y;}
using Args=BnCudaNativeLatentAttentionArgs;
template<int BH>struct Shared{half q[BH][584],k[32][584],p[BH][40];float sc[BH][40],m[BH],l[BH],alpha[BH];int valid[32],address[32],chunk[32];};
static __device__ __forceinline__ void emit_pair(Args p,int row,int h,int d,float a,float b,float c,float s,bool rotate) {
    if(rotate){float x=a*s,y=b*s;float oa=fmaf(a,c,-y),ob=fmaf(b,c,x);a=oa;b=ob;}
    size_t off=(size_t)(h/8)*p.R*4096+(size_t)row*4096+(h%8)*512+d;
    p.out[off]=__float2half_rn(a);p.out[off+1]=__float2half_rn(b);
}
static __device__ __forceinline__ void emit(Args p,int row,int h,int d,float a,float b) {
    float c=1.f,s=0.f;bool rotate=p.freq&&d>=448;
    if(rotate){float theta=p.freq[(d-448)/2]*(p.pos+row);c=cosf(theta);s=sinf(theta);}
    emit_pair(p,row,h,d,a,b,c,s,rotate);
}
template<int BH,bool Split>__global__ void attention(Args p){extern __shared__ unsigned char raw[];auto&sm=*(Shared<BH>*)raw;int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x/(64/BH),hb=blockIdx.x%(64/BH),split=blockIdx.y;
 // Every row and latent/RoPE split is eight-half aligned; preserve scalar views too.
 bool qvector=((uintptr_t)p.q&15u)==0;
 bool kvector=(((uintptr_t)p.ring|(uintptr_t)p.kv|(uintptr_t)p.pc|(uintptr_t)p.pr)&15u)==0;
 if(qvector) {
    for(int t=threadIdx.x;t<BH*72;t+=128) {
        int h=t/72,d=t%72*8;
        uint4 v=d>=448&&d<512?make_uint4(0,0,0,0):*(const uint4*)(p.q+((size_t)row*64+hb*BH+h)*512+(d<448?d:d-64));
        *(uint4*)(&sm.q[h][d])=v;
    }
 } else for(int t=threadIdx.x;t<BH*576;t+=128) {
    int h=t/576,d=t%576;
    sm.q[h][d]=d>=448&&d<512?__float2half(0):p.q[((size_t)row*64+hb*BH+h)*512+(d<448?d:d-64)];
 }
 for(int h=threadIdx.x;h<BH;h+=128){sm.m[h]=!Split&&p.sinks?p.sinks[hb*BH+h]:-INFINITY;sm.l[h]=!Split&&p.sinks?1.f:0.f;}
 float acc[BH/16][16][4]={};__syncthreads();int np=p.dense?min((p.pos+row+1)/p.rate,p.T):512,nw=p.W,n=np+nw,chunk=(n+p.S-1)/p.S,j0=Split?split*chunk:0,j1=Split?min(j0+chunk,n):n;
 #pragma unroll
 for(int phase=0;phase<2;phase++){int N=phase?32:16,start=phase?max(j0-nw,0):j0,end=phase?j1-nw:min(j1,nw);
  for(int at=start;at<end;at+=N){
   if(threadIdx.x<32) {
       int k=threadIdx.x,idx=at+k;
       if(phase) {
           int id=idx<end?(p.dense?idx:p.ids[(size_t)row*512+idx]):-1;
           sm.valid[k]=id>=0;
           // EPP divides256, so page addressing is a shift, never per-value division.
           int shift=p.epp?__ffs(p.epp)-1:0;
           sm.address[k]=id>=0?p.bt[id>>shift]*p.epp+(id&(p.epp-1)):0;
           sm.chunk[k]=0;
       } else {
           int abs=p.pos+row-idx;
           sm.valid[k]=k<N&&idx<end&&abs>=p.floor;
           sm.chunk[k]=abs>=p.pos;sm.address[k]=abs-(abs>=p.pos?p.pos:p.beg);
       }
   }
   __syncthreads();
   if(kvector) {
       for(int t=threadIdx.x;t<N*72;t+=128) {
           int k=t/72,d=t%72*8;uint4 v=make_uint4(0,0,0,0);
           if(sm.valid[k]&&!(d>=448&&d<512)) {
               int address=sm.address[k];
               const half*input=phase?(d<448?p.pc+(size_t)address*448+d:p.pr+(size_t)address*64+d-512):
                   (sm.chunk[k]?p.kv:p.ring)+(size_t)address*512+(d<448?d:d-64);
               v=*(const uint4*)input;
               if(!phase) {
                   // Source adds the inactive zero operand when merging ring/chunk.
                   unsigned*u=(unsigned*)&v;
                   #pragma unroll
                   for(int i=0;i<4;i++){if((u[i]&0xffffu)==0x8000u)u[i]&=0xffff0000u;if((u[i]>>16)==0x8000u)u[i]&=0x0000ffffu;}
               }
           }
           *(uint4*)(&sm.k[k][d])=v;
       }
   } else for(int t=threadIdx.x;t<N*576;t+=128) {
       int k=t/576,d=t%576;half v=__float2half(0);
       if(sm.valid[k]&&!(d>=448&&d<512)) {
           int address=sm.address[k];
           v=phase?(d<448?p.pc[(size_t)address*448+d]:p.pr[(size_t)address*64+d-512]):
               (sm.chunk[k]?p.kv:p.ring)[(size_t)address*512+(d<448?d:d-64)];
           if(!phase)v=__hadd(v,__float2half(0));
       }
       sm.k[k][d]=v;
   }
   __syncthreads();
   if(warp*8<N)
   #pragma unroll
   for(int m=0;m<BH/16;m++){float z[4]={};
   #pragma unroll
   for(int off=0;off<576;off+=16){unsigned a[4],b[2];
    load4(a,&sm.q[m*16+lane%16][off+(lane/16)*8]);
    load2(b,&sm.k[warp*8+lane%8][off+((lane/8)%2)*8]);
    mma(z,a,b);}for(int i=0;i<4;i++)sm.sc[m*16+lane/4+(i/2)*8][warp*8+2*(lane%4)+i%2]=z[i];}
   __syncthreads();
   // Four lanes per head cover each eight-key group in source reduction order.
   int sh=threadIdx.x/4,column=(threadIdx.x%4)*2;
   if(sh<BH) {
       float values[4][2],maximum=-INFINITY;
       #pragma unroll
       for(int group=0;group<4;group++) {
           #pragma unroll
           for(int pair=0;pair<2;pair++) {
               int k=group*8+column+pair;
               float v=k<N&&sm.valid[k]?sm.sc[sh][k]*0.04419417382415922f:-INFINITY;
               values[group][pair]=v;maximum=fmaxf(maximum,v);
           }
       }
       maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2));
       maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
       float previous=sm.m[sh],next=fmaxf(previous,maximum),me=next==-INFINITY?0.f:next;
       float alpha=previous==-INFINITY?0.f:expn(previous-me),sums[4];
       #pragma unroll
       for(int group=0;group<4;group++) {
           float prob[2];
           #pragma unroll
           for(int pair=0;pair<2;pair++) {
               int k=group*8+column+pair;
               prob[pair]=k<N&&sm.valid[k]?expn(values[group][pair]-me):0.f;
               if(k<N)sm.p[sh][k]=__float2half_rn(prob[pair]);
           }
           float sum=prob[0]+prob[1];
           sum+=__shfl_xor_sync(0xffffffff,sum,2);sum+=__shfl_xor_sync(0xffffffff,sum,1);
           sums[group]=sum;
       }
       float sum=(sums[0]+sums[2])+(sums[1]+sums[3]);
       __syncwarp();
       if(!column){sm.m[sh]=next;sm.l[sh]=fmaf(sm.l[sh],alpha,sum);sm.alpha[sh]=alpha;}
   }
   __syncthreads();
   #pragma unroll
   for(int m=0;m<BH/16;m++)
   #pragma unroll
   for(int nb=0;nb<16;nb++){int dbase=(nb*4+warp)*8;float*z=acc[m][nb];for(int i=0;i<4;i++)z[i]*=sm.alpha[m*16+lane/4+(i/2)*8];
   #pragma unroll
   for(int off=0;off<N;off+=16){unsigned a[4],b[2];
    load4(a,&sm.p[m*16+lane%16][off+(lane/16)*8]);
    load2_trans(b,&sm.k[off+lane%16][dbase<448?dbase:dbase+64]);
    mma(z,a,b);}}
   __syncthreads();
  }
 }
 float cosine[2]={1.f,1.f},sine[2]={0.f,0.f},denominator[BH/16][2];
 if(!Split) {
     if(p.freq) {
         #pragma unroll
         for(int n=0;n<2;n++) {
             int d=(n+14)*32+warp*8+2*(lane%4);
             float theta=p.freq[(d-448)/2]*(p.pos+row);
             cosine[n]=cosf(theta);sine[n]=sinf(theta);
         }
     }
     #pragma unroll
     for(int m=0;m<BH/16;m++) {
         #pragma unroll
         for(int hi=0;hi<2;hi++) {
             float l=sm.l[m*16+lane/4+hi*8];denominator[m][hi]=l==0.f?1.f:l;
         }
     }
 }
 #pragma unroll
 for(int m=0;m<BH/16;m++)
 #pragma unroll
 for(int nb=0;nb<16;nb++)
 #pragma unroll
 for(int hi=0;hi<2;hi++){int h=m*16+lane/4+hi*8,d=(nb*4+warp)*8+2*(lane%4);float a=acc[m][nb][hi*2],b=acc[m][nb][hi*2+1];if(Split){size_t base=((size_t)blockIdx.x*p.S+split)*BH+h;p.av[base*512+d]=a;p.av[base*512+d+1]=b;if(!nb&&warp==0&&lane%4==0){p.ml[base*2]=sm.m[h];p.ml[base*2+1]=sm.l[h];}}else{float den=denominator[m][hi];bool rotate=p.freq&&nb>=14;emit_pair(p,row,hb*BH+h,d,divide(a,den),divide(b,den),rotate?cosine[nb-14]:1.f,rotate?sine[nb-14]:0.f,rotate);}}
}
__global__ void combine(Args p) {
    int row=blockIdx.x/4,hb=blockIdx.x%4,h=threadIdx.x/8,lane=threadIdx.x%8;
    float m=p.sinks?p.sinks[hb*16+h]:-INFINITY,l=p.sinks?1.f:0.f,a[8]={},b[8]={};
    for(int split=0;split<p.S;split++) {
        size_t off=((size_t)blockIdx.x*p.S+split)*16+h;
        float ms=p.ml[off*2],ls=p.ml[off*2+1],mn=fmaxf(m,ms),me=mn==-INFINITY?0.f:mn;
        float alpha=m==-INFINITY?0.f:expn(m-me),beta=ms==-INFINITY?0.f:expn(ms-me);
        l=fmaf(l,alpha,ls*beta);
        #pragma unroll
        for(int slot=0;slot<8;slot++) {
            int d=blockIdx.y*128+lane*2+slot*16;
            a[slot]=fmaf(a[slot],alpha,p.av[off*512+d]*beta);
            b[slot]=fmaf(b[slot],alpha,p.av[off*512+d+1]*beta);
        }
        m=mn;
    }
    float den=l==0.f?1.f:l;
    #pragma unroll
    for(int slot=0;slot<8;slot++) {
        int d=blockIdx.y*128+lane*2+slot*16;
        emit(p,row,hb*16+h,d,divide(a[slot],den),divide(b[slot],den));
    }
}
}
int bn_cuda_native_latent_attention_init() {
    using namespace bn_cuda_native_attention_detail;
    cudaError_t a=cudaFuncSetAttribute(attention<16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<16>));
    cudaError_t b=cudaFuncSetAttribute(attention<32,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<32>));
    if(a!=cudaSuccess||b!=cudaSuccess){(void)cudaGetLastError();return -1;}return 0;
}
int bn_cuda_native_latent_attention(cudaStream_t stream,const BnCudaNativeLatentAttentionArgs*args) {
    using namespace bn_cuda_native_attention_detail;const Args&p=*args;
    if(p.S>1){attention<16,true><<<dim3(p.R*4,p.S),128,sizeof(Shared<16>),stream>>>(p);
        if(cudaGetLastError()!=cudaSuccess)return -1;combine<<<dim3(p.R*4,4),128,0,stream>>>(p);}
    else attention<32,false><<<p.R*2,128,sizeof(Shared<32>),stream>>>(p);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
