#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));std::exit(1);} }while(0)
__device__ unsigned long long ns(){unsigned long long v;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));return v;}
__global__ void delay_cycles(unsigned long long duration,int *out) {unsigned long long t=ns();while(ns()-t<duration){} if(out)*out=137;}
__global__ void consumer_kernel(const unsigned char *in,int *out) {*out=in[0];}
int main(){
 cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,0));cudaStream_t prod,cons;CHECK(cudaStreamCreateWithFlags(&prod,cudaStreamNonBlocking));CHECK(cudaStreamCreateWithFlags(&cons,cudaStreamNonBlocking));
 unsigned char *host,*dev;int *output;size_t bytes=128u*1024u*1024u;CHECK(cudaMallocHost(&host,bytes));for(size_t i=0;i<bytes;i++)host[i]=137;CHECK(cudaMalloc(&dev,bytes));CHECK(cudaMalloc(&output,sizeof(int)));CHECK(cudaMemcpyAsync(dev,host,bytes,cudaMemcpyHostToDevice,prod));CHECK(cudaStreamSynchronize(prod));
 delay_cycles<<<1,1,0,prod>>>(1000000ull,nullptr);consumer_kernel<<<1,1,0,cons>>>(dev,output);CHECK(cudaDeviceSynchronize());
 const char *names[]={"exposed-kernel","hidden-kernel","late-kernel","exposed-copy","hidden-copy","late-copy"};
 for(int i=0;i<6;i++){
  cudaEvent_t ready,prior_end,consumer_start;CHECK(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));CHECK(cudaEventCreate(&prior_end));CHECK(cudaEventCreate(&consumer_start));nvtxRangePushA(names[i]);
  int prior_ms=(i==1||i==4)?12:1;int prod_ms=i==1?1:8;
  delay_cycles<<<1,1,0,cons>>>(1000000ull*prior_ms,nullptr);CHECK(cudaEventRecord(prior_end,cons));
  if(i<3)delay_cycles<<<1,1,0,prod>>>(1000000ull*prod_ms,nullptr);
  else for(int k=0;k<4;k++)CHECK(cudaMemcpyAsync(dev,host,bytes,cudaMemcpyHostToDevice,prod));
  CHECK(cudaEventRecord(ready,prod));CHECK(cudaStreamWaitEvent(cons,ready,0));
  if(i==2||i==5)std::this_thread::sleep_for(std::chrono::milliseconds(25));
  CHECK(cudaEventRecord(consumer_start,cons));consumer_kernel<<<1,1,0,cons>>>(dev,output);CHECK(cudaGetLastError());CHECK(cudaStreamSynchronize(cons));int value=0;CHECK(cudaMemcpy(&value,output,sizeof(int),cudaMemcpyDeviceToHost));if(value!=137)return 2;
  float gap;CHECK(cudaEventElapsedTime(&gap,prior_end,consumer_start));std::printf("{\"case\":\"%s\",\"gpu_event_gap_ms\":%.6f,\"value\":%d}\n",names[i],gap,value);nvtxRangePop();CHECK(cudaEventDestroy(ready));CHECK(cudaEventDestroy(prior_end));CHECK(cudaEventDestroy(consumer_start));
 }
 CHECK(cudaFree(output));CHECK(cudaFree(dev));CHECK(cudaFreeHost(host));CHECK(cudaStreamDestroy(cons));CHECK(cudaStreamDestroy(prod));return 0;
}
