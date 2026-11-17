/* Original direct EXL3 fused expert arithmetic, exllamav3 bd1f5191, MIT.
 * Fast math local; backend/request own weights, launch policy and locks. */
#include "gpu_cuda_exl3_fused.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda/atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <tuple>
namespace bn_cuda_fused_detail {
#include "cuda_exl3_fused/quant/exl3_moe_kernel.cuh"
using Kernel=void(*)(EXL3_MOE_KERNEL_ARGS);
static Kernel kernel(int m,int n){if(m==16&&n==256)return exl3_moe_kernel<3,256,2,16>;if(m==16&&n==128)return exl3_moe_kernel<3,128,2,16>;if(m==32&&n==128)return exl3_moe_kernel<3,128,2,32>;if(m==64&&n==128)return exl3_moe_kernel<3,128,2,64>;return nullptr;}

/* grid.x is the original per-expert split-K width, grid.y stays zero.
 * Flatten only independent expert groups into grid.z. No inner operation
 * observes another request's group index, locks, temporary slabs or output. */
template<int M,int N>
__global__ __launch_bounds__(512)
void exl3_moe_joint_kernel(__grid_constant__ const BnCudaFusedEXL3Batch batch)
{
    int first=0,index=0;
    for(;index<batch.count;index++) {
        if((int)blockIdx.z<first+batch.items[index].groups)break;
        first+=batch.items[index].groups;
    }
    if(index==batch.count)return;
    const BnCudaFusedEXL3Item &p=batch.items[index];
    exl3_moe_body<3,N,2,M>(
        (const half*)p.x,(half*)p.state_g,(half*)p.state_u,
        (half*)p.intermediate_g,(half*)p.intermediate_u,nullptr,p.pointers,
        (const int*)p.counts,(const int*)p.order,(const half*)p.weights,
        p.dim,p.hidden,p.experts,p.topk,p.capacity,p.groups,p.limit,0,6,6,6,
        (int*)p.locks,(float*)p.slots,(const int*)p.base,p.lo,p.hi,
        p.tokens,p.slot_rows,(int)blockIdx.z-first,p.groups);
}
using JointKernel=void(*)(BnCudaFusedEXL3Batch);
static JointKernel joint_kernel(int m,int n){if(m==16&&n==256)return exl3_moe_joint_kernel<16,256>;if(m==16&&n==128)return exl3_moe_joint_kernel<16,128>;if(m==32&&n==128)return exl3_moe_joint_kernel<32,128>;if(m==64&&n==128)return exl3_moe_joint_kernel<64,128>;return nullptr;}
}
int bn_cuda_exl3_fused_prepare(int*smem,int*blocks,int*joint_blocks)
{
    int device=0;cudaDeviceProp p={};
    if(!smem||!blocks||!joint_blocks||cudaGetDevice(&device)!=cudaSuccess||
       cudaGetDeviceProperties(&p,device)!=cudaSuccess||!p.cooperativeLaunch)return -1;
    int bytes=p.sharedMemPerBlockOptin<SMEM_MAX?(int)p.sharedMemPerBlockOptin:SMEM_MAX;
    const int ms[]={16,16,32,64},ns[]={256,128,128,128};
    for(int i=0;i<4;i++) {
        auto f=bn_cuda_fused_detail::kernel(ms[i],ns[i]);
        auto j=bn_cuda_fused_detail::joint_kernel(ms[i],ns[i]);
        int nb=0,nj=0;
        if(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess||
           cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,f,512,bytes)!=cudaSuccess||nb<1||
           cudaFuncSetAttribute(j,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess||
           cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nj,j,512,bytes)!=cudaSuccess||nj<1)return -1;
        blocks[i]=nb;joint_blocks[i]=nj;
    }
    *smem=bytes;return 0;
}
size_t bn_cuda_exl3_fused_lock_bytes(){return (size_t)(MOE_SCHED_OFFSET+MOE_SCHED_INTS)*sizeof(int);}
int bn_cuda_exl3_fused_launch(cudaStream_t stream,const BnCudaFusedEXL3Item*p,int smem)
{
    if(!p)return -1;
    auto f=bn_cuda_fused_detail::kernel(p->tile_m,p->tile_n);if(!f)return -1;
    void*out=nullptr;int act=0,kg=6,ku=6,kd=6;
    void*args[]={(void*)&p->x,(void*)&p->state_g,(void*)&p->state_u,
        (void*)&p->intermediate_g,(void*)&p->intermediate_u,&out,(void*)&p->pointers,
        (void*)&p->counts,(void*)&p->order,(void*)&p->weights,(void*)&p->dim,
        (void*)&p->hidden,(void*)&p->experts,(void*)&p->topk,(void*)&p->capacity,
        (void*)&p->groups,(void*)&p->limit,&act,&kg,&ku,&kd,(void*)&p->locks,
        (void*)&p->slots,(void*)&p->base,(void*)&p->lo,(void*)&p->hi,
        (void*)&p->tokens,(void*)&p->slot_rows};
    return cudaLaunchCooperativeKernel((const void*)f,dim3(p->width,1,p->groups),
        dim3(512),args,smem,stream)==cudaSuccess?0:-1;
}
int bn_cuda_exl3_fused_group_launch(cudaStream_t stream,const BnCudaFusedEXL3Batch*p,int smem)
{
    if(!p||p->count<2||p->count>BN_CUDA_FUSED_EXL3_GROUP_MAX)return -1;
    const auto &first=p->items[0];auto f=bn_cuda_fused_detail::joint_kernel(first.tile_m,first.tile_n);
    if(!f)return -1;int groups=0;
    for(int i=0;i<p->count;i++) {
        const auto &v=p->items[i];
        if(v.tile_m!=first.tile_m||v.tile_n!=first.tile_n||v.width!=first.width||v.groups<1)return -1;
        groups+=v.groups;
    }
    void*args[]={(void*)p};
    return cudaLaunchCooperativeKernel((const void*)f,dim3(first.width,1,groups),
        dim3(512),args,smem,stream)==cudaSuccess?0:-1;
}
