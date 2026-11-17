/* EXL3 slot arithmetic: exllamav3 bd1f519172406a9560bd5c90e61000691bb69a07,
 * MIT third_party/exllamav3/LICENSE. Small unstable sort comparator/network
 * adapted from PyTorch 08187d9e0fba026dc8217405802ab5381dc88d90 SortUtils.cuh,
 * BSD third_party/pytorch/LICENSE. Fast math local to this translation unit. */
#include "gpu_cuda_native_moe.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <stdint.h>
namespace {
__global__ void clear_counts(int*counts,int E){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<=E)counts[i]=0;}
__global__ void prepare(const int*ids,int*keys,int*values,int*counts,int A,int E){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<A){int e=ids[i];if(e<0||e>=E)e=E;keys[i]=e;values[i]=i;atomicAdd(counts+e,1);}}
__device__ void swap_pair(int*a,int*b,int*va,int*vb,bool*validA,bool*validB,bool dir){bool sw=((*a<*b)&&*validA)||!*validB;if(sw==dir){int t=*a;*a=*b;*b=t;t=*va;*va=*vb;*vb=t;bool v=*validA;*validA=*validB;*validB=v;}}
__global__ void small_sort(const int*keys,const int*values,int*out,int*order,int A){
 __shared__ int k[32],v[32];__shared__ bool valid[32];
 for(int j=0;j<2;j++){int i=threadIdx.x+j*16;valid[i]=i<A;k[i]=i<A?keys[i]:0;v[i]=i<A?values[i]:0;}
 for(unsigned size=2;size<32;size*=2){bool dir=(threadIdx.x&(size/2))!=0;for(unsigned stride=size/2;stride;stride/=2){__syncthreads();unsigned i=2*threadIdx.x-(threadIdx.x&(stride-1));swap_pair(k+i,k+i+stride,v+i,v+i+stride,valid+i,valid+i+stride,dir);}}
 for(unsigned stride=16;stride;stride/=2){__syncthreads();unsigned i=2*threadIdx.x-(threadIdx.x&(stride-1));swap_pair(k+i,k+i+stride,v+i,v+i+stride,valid+i,valid+i+stride,false);}
 __syncthreads();for(int j=0;j<2;j++){int i=threadIdx.x+j*16;if(i<A){out[i]=k[i];order[i]=v[i];}}
}
__global__ void finish(const int*order,const half*weights,const int*counts,int*starts,int*inv,half*sorted_weights,int A,int E){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i==0){int s=0;for(int e=0;e<=E;e++){starts[e]=s;s+=counts[e];}}if(i<A){int a=order[i];inv[a]=i;sorted_weights[i]=weights[a];}}
__global__ void group_rows(BnCudaNativeMoEGroup meta,const half*x,const int*order,half*out,int R,int K,int D,int M,int B){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)B*M*D/8;if(i>=n)return;int col=(int)(i%(D/8))*8;size_t row=i/(D/8);int b=(int)(row/M),m=(int)(row%M);int a=m<meta.counts[b]?order[meta.starts[b]+m]:-1;int token=a>=0&&a<R*K?a/K:-1;uint4 v={0,0,0,0};if(token>=0)v=*(const uint4*)(x+(size_t)token*D+col);*(uint4*)(out+i*8)=v;}
/* Same accumulation statements/order as upstream exl3_moe_gather_kernel.
 * Narrow metadata is exact in the admitted assignment/slot range. */
__global__ void reduce_slots(float*__restrict__ out,const float*__restrict__ scratch,const int*__restrict__ ids,const int*__restrict__ inv,const int*__restrict__ starts,const int*__restrict__ base,const int*__restrict__ kind,const half*__restrict__ weights,int D,int K,int E,int capacity,int A){
 __shared__ int slots[32];__shared__ float wts[32];__shared__ int nslots;int token=blockIdx.x;
 if(threadIdx.x<(unsigned)K){int a=token*K+threadIdx.x,e=ids[a],slot=-1;float wt=1;
  if(e>=0&&e<E){int type=kind[e];if(type){int pos=inv[a];if(pos>=0&&pos<A){int64_t s=(int64_t)base[e]+pos-starts[e];if(s>=0&&s<capacity){slot=(int)s;if(type==2)wt=__half2float(weights[pos]);}}}}
  slots[threadIdx.x]=slot;wts[threadIdx.x]=wt;}
 __syncthreads();if(threadIdx.x==0){int n=0;for(int k=0;k<K;k++)if(slots[k]>=0){slots[n]=slots[k];wts[n]=wts[k];n++;}nslots=n;}__syncthreads();int n=nslots;if(!n)return;
 for(int col=threadIdx.x;col<D;col+=blockDim.x){float sum=0;for(int k=0;k<n;k++)sum+=scratch[(size_t)slots[k]*D+col]*wts[k];out[(size_t)token*D+col]+=sum;}
}
}
size_t bn_cuda_native_moe_sort_bytes(int A){if(A<1)return 0;size_t bytes=0;int*ptr=NULL;if(cub::DeviceRadixSort::SortPairs(NULL,bytes,ptr,ptr,ptr,ptr,A)!=cudaSuccess)return 0;return bytes?bytes:1;}
int bn_cuda_native_moe_index(cudaStream_t stream,void*work,size_t bytes,const void*ids,const void*weights,void*keys,void*values,void*sorted,void*order,void*inv,void*sw,void*counts,void*starts,int A,int E){
 clear_counts<<<(E+256)/256,256,0,stream>>>((int*)counts,E);if(cudaGetLastError()!=cudaSuccess)return -1;
 prepare<<<(A+255)/256,256,0,stream>>>((const int*)ids,(int*)keys,(int*)values,(int*)counts,A,E);if(cudaGetLastError()!=cudaSuccess)return -1;
 if(A<=32)small_sort<<<1,16,0,stream>>>((const int*)keys,(const int*)values,(int*)sorted,(int*)order,A);
 else if(cub::DeviceRadixSort::SortPairs(work,bytes,(const int*)keys,(int*)sorted,(const int*)values,(int*)order,A,0,32,stream)!=cudaSuccess)return -1;
 if(cudaGetLastError()!=cudaSuccess)return -1;
 finish<<<(A+255)/256,256,0,stream>>>((const int*)order,(const half*)weights,(const int*)counts,(int*)starts,(int*)inv,(half*)sw,A,E);return cudaGetLastError()==cudaSuccess?0:-1;
}
int bn_cuda_native_moe_group(cudaStream_t stream,const BnCudaNativeMoEGroup*meta,const void*x,const void*order,void*out,int R,int K,int D,int M,int B){size_t n=(size_t)B*M*D/8;group_rows<<<(n+255)/256,256,0,stream>>>(*meta,(const half*)x,(const int*)order,(half*)out,R,K,D,M,B);return cudaGetLastError()==cudaSuccess?0:-1;}
int bn_cuda_native_moe_reduce(cudaStream_t stream,void*out,const void*scratch,const void*ids,const void*inv,const void*starts,const void*base,const void*kind,const void*w,int R,int D,int K,int E,int slots){reduce_slots<<<R,D<1024?D:1024,0,stream>>>((float*)out,(const float*)scratch,(const int*)ids,(const int*)inv,(const int*)starts,(const int*)base,(const int*)kind,(const half*)w,D,K,E,slots,R*K);return cudaGetLastError()==cudaSuccess?0:-1;}
