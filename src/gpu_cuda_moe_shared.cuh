#ifndef BN_GPU_CUDA_MOE_SHARED_CUH
#define BN_GPU_CUDA_MOE_SHARED_CUH
/* Included inside bn_cuda_moe_resident after the single-request bank path. */
struct SharedBankWork {
    ResidentRequest *request;
    BankState *bank;
    BankSlot *slot;
    void *data[4];
    int max_rows, active;
};
static int bank_submit_group(void *ctx_ptr,void *ptr,const BnGPUMoEBankGroupPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *owner=(ResidentRequest*)ptr;
    if(!ctx||!owner||!p||owner->backend_owner!=ctx||owner->failed||ctx->compute_capability!=1200||
       !p->consumers||p->n_consumers<1||p->n_consumers>16||p->n_experts<1||p->n_experts>64)return -1;
    BankSlot *source=bank_lease(owner,&p->lease);BankState *bank=(BankState*)owner->moe_banks;
    if(!source||(unsigned)p->n_experts>bank->config.experts_per_slot)return -1;
    if(bank->cache) {
        if(bank->cache->failed||!bank->cache_open||bank->cache_group_owner!=owner||
           p->cache_generation!=bank->cache_layer.generation||!p->cache_generation||
           p->lease.gate_type!=bank->cache->gate_type[bank->cache_layer.layer]||
           p->lease.down_type!=bank->cache->down_type[bank->cache_layer.layer]||
           !source->host_raw||!source->device_raw)return -1;
    }else if(p->cache_generation)return -1;
    const BnQuantFormatOps *gate=bn_quant_format_ops(p->lease.gate_type),*down=bn_quant_format_ops(p->lease.down_type);
    if(!gate||!down)return -1;
    SharedBankWork work[16]={};int used[64]={};size_t expert_calls=0,projections=0;
    JointGroupedBatch joint={};joint.count=p->n_consumers;
    for(int i=0;i<p->n_consumers;i++) {
        const BnGPUMoEBankConsumerPlan *c=p->consumers+i;SharedBankWork *w=work+i;
        w->request=(ResidentRequest*)c->request;ResidentRequest *r=w->request;
        if(!r||r->backend_owner!=ctx||r->failed)return -1;
        for(int j=0;j<i;j++)if(work[j].request==r)return -1;
        w->bank=(BankState*)r->moe_banks;BankState *b=w->bank;
        if(!b||b->cache!=bank->cache||b->config.dim!=bank->config.dim||b->config.hidden_dim!=bank->config.hidden_dim||
           b->config.k!=bank->config.k||b->config.n_experts!=bank->config.n_experts||
           c->workspace_slot>=b->config.slots||b->config.experts_per_slot<(unsigned)p->n_experts||
           !c->experts||c->total_tokens<1||c->total_tokens>INT_MAX/b->config.k||
           !c->prepared_input||c->prepared_input!=b->input_generation||c->total_tokens!=b->input_tokens||
           !b->grouped_input||!b->grouped_down||c->total_tokens>b->grouped_capacity||
           (c->clamped!=0&&c->clamped!=1)||(c->clamped&&(!isfinite(c->clamp_limit)||c->clamp_limit<0)))return -1;
        if(bank->cache&&(b->cache_group_owner!=owner||c->total_tokens!=b->cache_nt))return -1;
        w->slot=b->slots+c->workspace_slot;
        if((r==owner&&w->slot!=source)||!w->slot->host_items||!w->slot->device_items)return -1;
        int n=c->total_tokens*b->config.k,dim=b->config.dim,hidden=b->config.hidden_dim;
        if(dim>INT_MAX/n||hidden>INT_MAX/n)return -1;
        BnGPUResidentView views[]={c->x,c->gate,c->up,c->out};
        int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};
        size_t counts[]={(size_t)n*dim,(size_t)n*hidden,(size_t)n*hidden,(size_t)n*dim};
        if(!views_validate(r,views,types,counts,4,w->data))return -1;
        for(int e=0;e<p->n_experts;e++) {
            const BnGPUMoEBankItem *item=c->experts+e;
            if((unsigned)item->expert>=(unsigned)b->config.n_experts||item->n_rows<0||item->n_rows>c->total_tokens||
               item->first_row<0||item->first_row>n-item->n_rows||
               (i&&item->expert!=p->consumers[0].experts[e].expert))return -1;
            for(int j=0;j<e;j++) {
                const BnGPUMoEBankItem *other=c->experts+j;
                if(other->expert==item->expert||(other->n_rows&&item->n_rows&&
                   item->first_row<other->first_row+other->n_rows&&other->first_row<item->first_row+item->n_rows))return -1;
            }
            if(bank->cache&&(bank->cache_seen[item->expert]||
                item->first_row!=b->cache_offsets[item->expert]||
                item->n_rows!=b->cache_offsets[item->expert+1]-b->cache_offsets[item->expert]))return -1;
            if(item->n_rows){used[e]=1;w->active++;expert_calls++;}
            if(item->n_rows>w->max_rows)w->max_rows=item->n_rows;
        }
        if(!w->active)return -1;
    }
    for(int e=0;e<p->n_experts;e++)if(!used[e])return -1;
    if(bank->cache)for(int m=0;m<bank->cache_group_count;m++) {
        ResidentRequest *member=bank->cache_group_requests[m];BankState *mb=(BankState*)member->moe_banks;
        int active=0,present=0;
        for(int e=0;e<p->n_experts;e++){int expert=p->consumers[0].experts[e].expert;
            active|=mb->cache_offsets[expert+1]>mb->cache_offsets[expert];}
        for(int i=0;i<p->n_consumers;i++)present|=work[i].request==member;
        if(active&&!present)return -1;
    }
    if(set_device(owner))return -1;
    for(int i=0;i<p->n_consumers;i++) {
        SharedBankWork *w=work+i;
        if(w->slot==source)continue;
        if(w->slot->state==1)return BN_GPU_MOE_BANK_BUSY;
        if(w->slot->state==2) {
            cudaError_t ready=cudaEventQuery(w->slot->completed);
            if(ready==cudaErrorNotReady){cudaError_t last=cudaGetLastError();
                if(last!=cudaSuccess&&last!=cudaErrorNotReady)goto failed;
                return BN_GPU_MOE_BANK_BUSY;}
            if(ready!=cudaSuccess)goto failed;
        }
    }
    if(cudaGetLastError()!=cudaSuccess)goto failed;
    source->state=2;
    for(int i=0;i<p->n_consumers;i++) {
        work[i].slot->state=2;
        memcpy(work[i].slot->host_items,p->consumers[i].experts,(size_t)p->n_experts*sizeof(BnGPUMoEBankItem));
    }
    if(bank->cache) {
        if(expert_cache_upload_bank(bank,source,&p->lease,p->consumers[0].experts,p->n_experts))goto failed;
    }else if(cudaMemcpyAsync(source->device,source->host,(size_t)p->n_experts*p->lease.expert_bytes,
        cudaMemcpyHostToDevice,bank->copy_stream)!=cudaSuccess)goto failed;
    for(int i=0;i<p->n_consumers;i++) {
        SharedBankWork *w=work+i;
        if(cudaMemcpyAsync(w->slot->device_items,w->slot->host_items,(size_t)p->n_experts*sizeof(BnGPUMoEBankItem),
            cudaMemcpyHostToDevice,w->bank->copy_stream)!=cudaSuccess)goto failed;
    }
    if(cudaEventRecord(source->ready,bank->copy_stream)!=cudaSuccess)goto failed;
    if(cudaStreamWaitEvent(owner->stream,source->ready,0)!=cudaSuccess)goto failed;
    for(int i=0;i<p->n_consumers;i++) {
        SharedBankWork *w=work+i;ResidentRequest *r=w->request;
        if(w->slot!=source) {
            if(cudaEventRecord(w->slot->ready,w->bank->copy_stream)!=cudaSuccess||
               cudaStreamWaitEvent(r->stream,w->slot->ready,0)!=cudaSuccess||
               cudaEventRecord(w->slot->completed,r->stream)!=cudaSuccess||
               cudaStreamWaitEvent(owner->stream,w->slot->completed,0)!=cudaSuccess)goto failed;
        }
        const BnGPUMoEBankConsumerPlan *c=p->consumers+i;
        GroupedBatch &batch=joint.batch[i];batch.raw=(const unsigned char*)source->device;
        batch.raw_ptrs=bank->cache?source->device_raw:NULL;
        batch.stride=p->lease.expert_bytes;batch.gate_bytes=p->lease.gate_bytes;
        batch.items=w->slot->device_items;batch.input=w->bank->grouped_input;
        batch.down_input=w->bank->grouped_down;batch.gate=(__nv_bfloat16*)w->data[1];batch.up=(__nv_bfloat16*)w->data[2];
        batch.out=(__nv_bfloat16*)w->data[3];batch.dim=bank->config.dim;batch.hidden=bank->config.hidden_dim;
        batch.max_rows=w->max_rows;batch.clamped=c->clamped;batch.clamp_limit=c->clamp_limit;
        batch.experts=p->n_experts;batch.matrix=c->total_tokens>64;
        if(batch.max_rows>joint.max_rows)joint.max_rows=batch.max_rows;
    }
    if(joint_grouped_execute(owner,joint,gate->bf16_moe,down->bf16_moe,&projections))goto failed;
    /* Event waits capture the recorded state at the host API call. Re-record
     * source.ready only AFTER its copy wait; re-record each peer.completed
     * only AFTER the owner has captured its input-ready state. These events
     * then protect the actual joint outputs and raw-bank lifetime. Access to
     * bank submission/acquisition is caller-serialized, including across slots. */
    if(cudaEventRecord(source->ready,owner->stream)!=cudaSuccess)goto failed;
    for(int i=0;i<p->n_consumers;i++)if(work[i].slot!=source) {
        if(cudaStreamWaitEvent(work[i].request->stream,source->ready,0)!=cudaSuccess||
           cudaEventRecord(work[i].slot->completed,work[i].request->stream)!=cudaSuccess)goto failed;
    }
    for(int i=0;i<p->n_consumers;i++)if(work[i].slot!=source&&
        cudaStreamWaitEvent(owner->stream,work[i].slot->completed,0)!=cudaSuccess)goto failed;
    if(cudaEventRecord(source->completed,owner->stream)!=cudaSuccess)goto failed;
    bank->stats.joint_kernel_launches=saturated_add(bank->stats.joint_kernel_launches,projections+2);
    bank->stats.joint_projection_launches=saturated_add(bank->stats.joint_projection_launches,projections);
    bank->stats.joint_launches_avoided=saturated_add(bank->stats.joint_launches_avoided,
        (size_t)p->n_consumers*4-projections-2);
    if(bank->cache)for(int e=0;e<p->n_experts;e++) {
        int expert=p->consumers[0].experts[e].expert,slot=bank->cache_entries[expert];
        bank->cache_seen[expert]=1;if(slot>=0)bank->cache->entry[slot].valid=1;
    }
    else bank->stats.upload_bytes=saturated_add(bank->stats.upload_bytes,(size_t)p->n_experts*p->lease.expert_bytes);
    bank->stats.shared_uploads=saturated_add(bank->stats.shared_uploads,1);
    bank->stats.shared_consumers=saturated_add(bank->stats.shared_consumers,(size_t)p->n_consumers);
    {
        size_t saved=expert_calls-(size_t)p->n_experts;
        size_t bytes=saved&&p->lease.expert_bytes>SIZE_MAX/saved?SIZE_MAX:saved*p->lease.expert_bytes;
        bank->stats.shared_upload_bytes_avoided=saturated_add(bank->stats.shared_upload_bytes_avoided,bytes);
    }
    for(int i=0;i<p->n_consumers;i++) {
        BankState *b=work[i].bank;
        b->stats.submitted_banks=saturated_add(b->stats.submitted_banks,1);
        b->stats.submitted_experts=saturated_add(b->stats.submitted_experts,(size_t)work[i].active);
        b->stats.grouped_banks=saturated_add(b->stats.grouped_banks,1);
        b->stats.metadata_upload_bytes=saturated_add(b->stats.metadata_upload_bytes,(size_t)p->n_experts*sizeof(BnGPUMoEBankItem));
    }
    return 0;
failed:
    if(bank->cache)bank->cache->failed=1;
    /* Drain every borrowing stream before caller teardown can free the source
     * bank. This is an error boundary, never a steady-state host fence. */
    (void)cudaStreamSynchronize(bank->copy_stream);
    for(int i=0;i<p->n_consumers;i++) {
        (void)cudaStreamSynchronize(work[i].bank->copy_stream);
        (void)cudaStreamSynchronize(work[i].request->stream);work[i].request->failed=1;
    }
    (void)cudaStreamSynchronize(owner->stream);owner->failed=1;return -1;
}
#endif
