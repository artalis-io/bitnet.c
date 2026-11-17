#ifndef BN_GPU_CUDA_EXL3_FUSED_VALIDATE_CUH
#define BN_GPU_CUDA_EXL3_FUSED_VALIDATE_CUH
#include "gpu_cuda_exl3_fused.h"
namespace bn_cuda_fused {
using bn_cuda_gdn::ResidentRequest;
static int shape(int m,int n){if(m==16&&n==256)return 0;if(m==16&&n==128)return 1;if(m==32&&n==128)return 2;if(m==64&&n==128)return 3;return -1;}
static size_t workspace_size(void*vctx){BnCudaCtx*ctx=(BnCudaCtx*)vctx;return ctx&&ctx->compute_capability>=800?bn_cuda_exl3_fused_lock_bytes():0;}
static int reserve(void*vctx,void*request)
{
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;ResidentRequest*r=(ResidentRequest*)request;
    if(!ctx||!r||r->backend_owner!=ctx||ctx->compute_capability<800||bn_cuda_gdn::set_device(r))return -1;
    if(r->fused_locks)return 0;
    int smem=0,blocks[4]={},joint_blocks[4]={};
    if(bn_cuda_exl3_fused_prepare(&smem,blocks,joint_blocks))return -1;
    void*locks=NULL;cudaEvent_t ready=NULL,done=NULL;
    cudaError_t e=cudaMalloc(&locks,bn_cuda_exl3_fused_lock_bytes());
    if(e==cudaSuccess)e=cudaMemsetAsync(locks,0,bn_cuda_exl3_fused_lock_bytes(),r->stream);
    if(e==cudaSuccess)e=cudaStreamSynchronize(r->stream);
    if(e==cudaSuccess)e=cudaEventCreateWithFlags(&ready,cudaEventDisableTiming);
    if(e==cudaSuccess)e=cudaEventCreateWithFlags(&done,cudaEventDisableTiming);
    if(e!=cudaSuccess) {
        if(ready)cudaEventDestroy(ready);
        if(done)cudaEventDestroy(done);
        cudaFree(locks);
        if(e==cudaErrorMemoryAllocation)(void)cudaGetLastError();
        return -1;
    }
    r->fused_smem=smem;memcpy(r->fused_blocks,blocks,sizeof(blocks));
    memcpy(r->fused_joint_blocks,joint_blocks,sizeof(joint_blocks));
    r->fused_locks=locks;r->fused_ready=ready;r->fused_done=done;return 0;
}
static int info(void*vctx,void*request,int m,int n,BnGPUFusedEXL3LaunchInfo*out)
{
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;ResidentRequest*r=(ResidentRequest*)request;int s=shape(m,n);
    if(!ctx||!r||!out||r->backend_owner!=ctx||!r->fused_locks||r->failed||s<0)return -1;
    BnGPUFusedEXL3LaunchInfo v={r->fused_smem,512,r->fused_blocks[s],ctx->multiprocessors,ctx->multiprocessors*r->fused_blocks[s]/8};
    if(v.max_groups>64)v.max_groups=64;
    *out=v;return 0;
}
static int validate(void*vctx,void*request,const BnGPUFusedEXL3ResidentPlan*p,BnCudaFusedEXL3Item*out)
{
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;ResidentRequest*r=(ResidentRequest*)request;
    BnGPUFusedEXL3Workspace ws;BnGPUFusedEXL3LaunchInfo launch;
    if(!p||!out||!ctx||!r||r->backend_owner!=ctx||r->failed||
       bn_gpu_native_moe_shape(p->tokens,p->topk,p->experts)||
       bn_gpu_fused_exl3_workspace(p->groups,p->capacity,p->dim,p->hidden,&ws)||
       info(vctx,request,p->tile_m,p->tile_n,&launch)||p->group_width<8||p->group_width>32||
       p->groups>launch.multiprocessors*launch.blocks_per_sm/p->group_width||
       p->resident_count<1||p->resident_count>32||p->groups>p->resident_count||p->slot_rows<1||
       (size_t)p->slot_rows>SIZE_MAX/(size_t)p->dim/4||!isfinite(p->act_limit)||p->act_limit<0||
       p->act_limit>65504||(p->tile_n==256&&(p->dim%256||p->hidden%256)))return -1;
    const BnQuantTrellisOps*ops=bn_quant_format_trellis_ops(p->format);
    if(!ops||ops->fused!=BN_QUANT_TRELLIS_FUSED_EXL3_MUL1_3_WEIGHTED_F32||p->codebook_marker!=ops->codebook_marker)return -1;
    BnCudaFusedEXL3Item item={};item.pointers.count=p->resident_count;
    for(int b=0;b<p->resident_count;b++) {
        int expert=p->expert_ids[b];if(expert<0||expert>=p->experts)return -1;
        for(int j=0;j<b;j++)if(p->expert_ids[j]==expert)return -1;
        item.pointers.experts[b]=expert;
        for(int pr=0;pr<3;pr++) {
            int K=pr==2?p->hidden:p->dim,N=pr==2?p->dim:p->hidden;
            size_t packed=(size_t)K*N/8*3,bytes=packed+((size_t)K+N)*2;
            const BnCudaBuffer*w=(const BnCudaBuffer*)p->weight[pr][b];
            if(!w||w->trellis_owner!=ctx||!w->data||w->type!=p->format||w->rows!=N||w->cols!=K||
               w->trellis_marker!=p->codebook_marker||w->size!=bytes||((uintptr_t)w->data&15u))return -1;
            item.pointers.packed[pr][b]=w->data;
            item.pointers.pre[pr][b]=(const unsigned char*)w->data+packed;
            item.pointers.post[pr][b]=(const unsigned char*)item.pointers.pre[pr][b]+(size_t)K*2;
        }
    }
    size_t A=(size_t)p->tokens*p->topk,E=(size_t)p->experts;
    BnGPUResidentView v[]={p->x,p->counts,p->order,p->weights,p->slot_base,p->state_g,p->state_u,p->intermediate_g,p->intermediate_u,p->slots};
    size_t count[]={(size_t)p->tokens*p->dim,E+1,A,A,E,ws.state_count,ws.state_count,ws.intermediate_count,ws.intermediate_count,(size_t)p->slot_rows*p->dim};
    size_t align[]={16,4,4,2,4,16,16,16,16,16};
    int type[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32};
    void*d[10];
    for(int i=0;i<10;i++) {
        if(v[i].count!=count[i]||v[i].type!=type[i])return -1;
        d[i]=bn_cuda_gdn::view_data(r,v[i]);if(!d[i]||((uintptr_t)d[i]&(align[i]-1)))return -1;
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(v[i],v[j]))return -1;
    }
    item.x=d[0];item.counts=d[1];item.order=d[2];item.weights=d[3];item.base=d[4];
    item.state_g=d[5];item.state_u=d[6];item.intermediate_g=d[7];item.intermediate_u=d[8];item.slots=d[9];item.locks=r->fused_locks;
    item.tokens=p->tokens;item.dim=p->dim;item.hidden=p->hidden;item.experts=p->experts;item.topk=p->topk;
    item.capacity=p->capacity;item.lo=p->tile_m==16?1:p->tile_m==32?17:33;
    item.hi=p->tile_m==16?16:p->tile_m==32?32:p->capacity;
    item.tile_m=p->tile_m;item.tile_n=p->tile_n;item.width=p->group_width;item.groups=p->groups;
    item.slot_rows=p->slot_rows;item.limit=p->act_limit;*out=item;return 0;
}
static int run(void*vctx,void*request,const BnGPUFusedEXL3ResidentPlan*p)
{
    BnCudaFusedEXL3Item item;ResidentRequest*r=(ResidentRequest*)request;
    if(validate(vctx,request,p,&item)||bn_cuda_gdn::set_device(r))return -1;
    int error=bn_cuda_exl3_fused_launch(r->stream,&item,r->fused_smem);
    if(error)r->failed=1;
    return error;
}
}
#include "gpu_cuda_exl3_fused_group.cuh"
#endif
