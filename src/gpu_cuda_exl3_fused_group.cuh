#ifndef BN_GPU_CUDA_EXL3_FUSED_GROUP_CUH
#define BN_GPU_CUDA_EXL3_FUSED_GROUP_CUH
namespace bn_cuda_fused {
static int group(void*vctx,void*request,const BnGPUFusedEXL3GroupPlan*p,BnGPUFusedEXL3GroupStats*out)
{
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;ResidentRequest*owner=(ResidentRequest*)request;
    if(!ctx||!owner||!p||!out||owner->backend_owner!=ctx||owner->failed||!owner->fused_done||
       !p->consumers||p->n_consumers<2||p->n_consumers>BN_GPU_FUSED_EXL3_MAX_CONSUMERS)return -1;
    ResidentRequest*requests[BN_GPU_FUSED_EXL3_MAX_CONSUMERS]={};
    BnCudaFusedEXL3Item items[BN_GPU_FUSED_EXL3_MAX_CONSUMERS];int present=0;
    for(int i=0;i<p->n_consumers;i++) {
        const auto &c=p->consumers[i];ResidentRequest*r=(ResidentRequest*)c.request;
        if(validate(vctx,c.request,c.plan,items+i)||!r->fused_ready||!r->fused_done||r->fused_smem!=owner->fused_smem)return -1;
        for(int j=0;j<i;j++)if(requests[j]==r)return -1;
        requests[i]=r;present|=r==owner;
    }
    if(!present||bn_cuda_gdn::set_device(owner))return -1;
    BnGPUFusedEXL3GroupStats stats={};int used[BN_GPU_FUSED_EXL3_MAX_CONSUMERS]={};
    for(int seed=0;seed<p->n_consumers;seed++) {
        if(used[seed])continue;
        const auto &first=items[seed];int s=shape(first.tile_m,first.tile_n);
        int capacity=ctx->multiprocessors*requests[seed]->fused_joint_blocks[s]/first.width;
        int max_consumers=capacity<1?1:capacity<BN_CUDA_FUSED_EXL3_GROUP_MAX?capacity:BN_CUDA_FUSED_EXL3_GROUP_MAX;
        BnCudaFusedEXL3Batch batch={};int indices[BN_CUDA_FUSED_EXL3_GROUP_MAX]={};
        for(int i=seed;i<p->n_consumers&&batch.count<max_consumers;i++) {
            const auto &v=items[i];
            if(used[i]||v.tile_m!=first.tile_m||v.tile_n!=first.tile_n||v.width!=first.width)continue;
            indices[batch.count]=i;batch.items[batch.count++]=v;used[i]=1;
        }
        int groups=batch.count;
        if(batch.count>1) {
            for(int i=0;i<batch.count;i++)batch.items[i].groups=1;
            bool progress=true;
            while(progress&&groups<capacity) {
                progress=false;
                for(int i=0;i<batch.count&&groups<capacity;i++)
                    if(batch.items[i].groups<items[indices[i]].groups){batch.items[i].groups++;groups++;progress=true;}
            }
        }
        for(int i=0;i<batch.count;i++) {
            ResidentRequest*r=requests[indices[i]];
            if(r!=owner&&(cudaEventRecord(r->fused_ready,r->stream)!=cudaSuccess||
                cudaStreamWaitEvent(owner->stream,r->fused_ready,0)!=cudaSuccess))goto failed;
        }
        if(batch.count>1) {
            if(bn_cuda_exl3_fused_group_launch(owner->stream,&batch,owner->fused_smem))goto failed;
            stats.joint_kernel_launches++;
        }else if(bn_cuda_exl3_fused_launch(owner->stream,&batch.items[0],owner->fused_smem))goto failed;
        stats.kernel_launches++;stats.consumers+=(uint64_t)batch.count;
        if(cudaEventRecord(owner->fused_done,owner->stream)!=cudaSuccess)goto failed;
        for(int i=0;i<batch.count;i++) {
            ResidentRequest*r=requests[indices[i]];
            if(r!=owner&&cudaStreamWaitEvent(r->stream,owner->fused_done,0)!=cudaSuccess)goto failed;
        }
    }
    *out=stats;return 0;
failed:
    owner->failed=1;
    for(int i=0;i<p->n_consumers;i++)requests[i]->failed=1;
    return -1;
}
}
#endif
