/* Q8 integer-MMA differential: retained DP4A and independent scalar CPU FMA. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <math.h>
#include "quant.h"
struct BnCudaBlockQ8_1 {uint16_t d;int16_t qsum;int8_t qs[32];};
static_assert(sizeof(BnBlockQ8_0)==34,"weight layout");
static_assert(sizeof(BnCudaBlockQ8_1)==36,"activation layout");
static __host__ __device__ float cuda_fp16_to_fp32(uint16_t u){__half_raw r;r.x=u;return __half2float((__half)r);}
static __device__ int cuda_dp4a_i32(int a,int b,int c){return __dp4a(a,b,c);}
#define OK(x) do{cudaError_t err=(x);if(err!=cudaSuccess){fprintf(stderr,"CUDA %s at%d\n",cudaGetErrorString(err),__LINE__);abort();}}while(0)
#include "quant/int8_mma_cuda.cuh"
static __device__ __forceinline__ void q8_bf16_mmq_core(__nv_bfloat16 *out, const BnBlockQ8_0 *weights, const BnCudaBlockQ8_1 *input, int rows, int cols, int n_tokens, int row, int token0, BnCudaBlockQ8_1 activation[4]) {
    int blocks = cols / 32;
    float acc[4] = {0};
    for (int b = 0; b < blocks; b++) {
        if (threadIdx.x < 4) {
            int token = token0 + threadIdx.x;
            if (token >= n_tokens) token = n_tokens - 1;
            activation[threadIdx.x] = input[(size_t)token * blocks + b];
        }
        __syncthreads();
        if (row < rows) {
            const BnBlockQ8_0 *w = weights + (size_t)row * blocks + b;
            int packed[8];
#pragma unroll
            for (int j = 0; j < 8; j++) memcpy(&packed[j], w->qs + j * 4, sizeof(int));
            float wd = cuda_fp16_to_fp32(w->d);
#pragma unroll
            for (int t = 0; t < 4; t++) {
                if (token0 + t >= n_tokens) continue;
                int dot = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    int v; memcpy(&v, activation[t].qs + j * 4, sizeof(v));
                    dot = cuda_dp4a_i32(packed[j], v, dot);
                }
                float scale = wd * cuda_fp16_to_fp32(activation[t].d);
                acc[t] = fmaf(scale, (float)dot, acc[t]);
            }
        }
        __syncthreads();
    }
    if (row < rows) {
#pragma unroll
        for (int t = 0; t < 4; t++)
            if (token0 + t < n_tokens)
                out[(size_t)(token0 + t) * rows + row] = __float2bfloat16_rn(acc[t]);
    }
}

static __global__ void q8_bf16_mmq_kernel(__nv_bfloat16 *out,
    const BnBlockQ8_0 *weights, const BnCudaBlockQ8_1 *input,
    int rows, int cols, int n_tokens) {
    __shared__ BnCudaBlockQ8_1 activation[4];
    int tiles = (n_tokens - 1) / 4 + 1;
    q8_bf16_mmq_core(out, weights, input, rows, cols, n_tokens, (blockIdx.x / tiles) * 128 + threadIdx.x, (blockIdx.x % tiles) * 4, activation);
}


#include "quant/q8_bf16_mma_cuda.cuh"
template<int T>static __global__ void run_mma(__nv_bfloat16*out,const BnBlockQ8_0*w,const BnCudaBlockQ8_1*x,int rows,int cols,int nt){int tiles=(nt+T-1)/T;cuda_bf16_q8_mma_core<T>(out,w,x,rows,cols,nt,(blockIdx.x/tiles)*16,(blockIdx.x%tiles)*T);}
static void launch(int tile,__nv_bfloat16*out,const BnBlockQ8_0*w,const BnCudaBlockQ8_1*x,int rows,int cols,int nt){
 if(!tile){q8_bf16_mmq_kernel<<<((rows+127)/128)*((nt+3)/4),128>>>(out,w,x,rows,cols,nt);return;}
 if(tile==640){cuda_bf16_q8_grouped_mma_kernel<64><<<((rows+15)/16)*((nt+63)/64),256>>>(out,w,x,rows,cols,nt);return;}
 unsigned blocks=((rows+15)/16)*((nt+tile-1)/tile);
 if(tile==8)run_mma<8><<<blocks,32>>>(out,w,x,rows,cols,nt);
 else if(tile==24)run_mma<24><<<blocks,96>>>(out,w,x,rows,cols,nt);
 else run_mma<64><<<blocks,256>>>(out,w,x,rows,cols,nt);
}
static void test(int rows,int cols,int nt,bool bench){
 std::vector<BnBlockQ8_0>w((size_t)rows*cols/32);std::vector<BnCudaBlockQ8_1>x((size_t)nt*cols/32);
 const uint16_t scales[]={0,1,0x3ff,0x1000,0x2401,0x3c01,0x7bff,0x9000,0x8001,0x83ff,0xbc01,0xfbff};
 for(size_t i=0;i<w.size();i++){w[i].d=scales[i%12];for(int j=0;j<32;j++)w[i].qs[j]=(int8_t)((i*13+j*17)%256-128);}
 for(size_t i=0;i<x.size();i++){x[i].d=scales[(i*5+3)%12];x[i].qsum=(int16_t)0x7bff;for(int j=0;j<32;j++)x[i].qs[j]=(int8_t)((i*29+j*7)%256-128);}
 BnBlockQ8_0*dw;BnCudaBlockQ8_1*dx;__nv_bfloat16*dy;size_t count=(size_t)rows*nt;
 OK(cudaMalloc(&dw,w.size()*sizeof(w[0])));OK(cudaMalloc(&dx,x.size()*sizeof(x[0])));OK(cudaMalloc(&dy,(count+32)*2));
 OK(cudaMemcpy(dw,w.data(),w.size()*sizeof(w[0]),cudaMemcpyHostToDevice));OK(cudaMemcpy(dx,x.data(),x.size()*sizeof(x[0]),cudaMemcpyHostToDevice));
 std::vector<uint16_t>ref(count),actual(count+32);std::vector<int>tiles={0,8,24,64,640};
 for(int tile:tiles){
  OK(cudaMemset(dy,0x5a,(count+32)*2));launch(tile,dy,dw,dx,rows,cols,nt);OK(cudaGetLastError());OK(cudaMemcpy(actual.data(),dy,(count+32)*2,cudaMemcpyDeviceToHost));
  for(size_t i=count;i<count+32;i++)assert(actual[i]==0x5a5a);
  if(!tile)memcpy(ref.data(),actual.data(),count*2);else for(size_t i=0;i<count;i++)if(ref[i]!=actual[i]){fprintf(stderr,"diff rows%d cols%d nt%d tile%d idx%zu old%u new%u\n",rows,cols,nt,tile,i,ref[i],actual[i]);abort();}
  if(!bench&&tile==8){
   for(int t=0;t<nt;t++)for(int r=0;r<rows;r++){
    float sum=0;for(int b=0;b<cols/32;b++){const auto&a=w[(size_t)r*(cols/32)+b];const auto&v=x[(size_t)t*(cols/32)+b];int dot=0;for(int k=0;k<32;k++)dot+=(int)a.qs[k]*(int)v.qs[k];float scale=cuda_fp16_to_fp32(a.d)*cuda_fp16_to_fp32(v.d);sum=fmaf(scale,(float)dot,sum);}
    uint32_t u;memcpy(&u,&sum,4);uint16_t bf=(uint16_t)((u+0x7fff+((u>>16)&1))>>16);assert(bf==actual[(size_t)t*rows+r]);
   }
  }
  if(bench){cudaEvent_t a,b;OK(cudaEventCreate(&a));OK(cudaEventCreate(&b));for(int i=0;i<5;i++)launch(tile,dy,dw,dx,rows,cols,nt);OK(cudaEventRecord(a));for(int i=0;i<30;i++)launch(tile,dy,dw,dx,rows,cols,nt);OK(cudaEventRecord(b));OK(cudaEventSynchronize(b));float ms;OK(cudaEventElapsedTime(&ms,a,b));printf("{\"rows\":%d,\"cols\":%d,\"tokens\":%d,\"tile\":%d,\"us\":%.3f}\n",rows,cols,nt,tile,ms*1000/30);OK(cudaEventDestroy(a));OK(cudaEventDestroy(b));}
 }
 OK(cudaFree(dw));OK(cudaFree(dx));OK(cudaFree(dy));
}
int main(int argc,char**argv){bool small=argc>1&&strcmp(argv[1],"--bench-small")==0;bool bench=small||(argc>1&&strcmp(argv[1],"--bench")==0);int cases=0;
 if(small){for(int rows:{96,1024,5120})for(int cols:{5120,6144}){test(rows,cols,8,true);cases++;}}
 else if(bench){for(int nt:{65,129,1025})for(int cols:{640,2560}){test(cols==640?2560:640,cols,nt,true);cases++;}}
 else{for(int nt:{1,7,8,9,23,24,25,63,64,65,129,257})for(int cols:{32,96,160,640,2560,5120}){test(33,cols,nt,false);cases++;}}
 fprintf(stderr,"CUDA Q8 MMA CPU/DP4A differential PASSED %d cases\n",cases);
}
