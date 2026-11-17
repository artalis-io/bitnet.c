#ifndef BN_GPU_CUDA_EXPERT_CACHE_BANK_CUH
#define BN_GPU_CUDA_EXPERT_CACHE_BANK_CUH
/* Placement/upload is shared by private and joint banks. Caller validates the
 * whole bank first and records copy/compute readiness after this returns. */
static int expert_cache_upload_bank(BankState *b,BankSlot *s,const BnGPUMoEBankLease *lease,
    const BnGPUMoEBankItem *items,int count) {
    for(int i=0;i<count;i++) {
        int expert=items[i].expert,hit=(b->cache_layer.hits[expert/64]>>(expert%64))&1;
        int slot=hit?b->cache_entries[expert]:expert_cache_admit(b,expert);if(slot==-2)return -1;
        if(hit&&expert_cache_decode_selection(b,expert)){expert_cache_unlink(b->cache,slot);expert_cache_touch(b->cache,slot);}
        unsigned char *raw=(unsigned char*)s->device+(size_t)i*lease->expert_bytes;
        if(!hit) {
            if(cudaMemcpyAsync(raw,(unsigned char*)s->host+(size_t)i*lease->expert_bytes,
                lease->expert_bytes,cudaMemcpyHostToDevice,b->copy_stream)!=cudaSuccess)return -1;
            b->stats.upload_bytes=saturated_add(b->stats.upload_bytes,lease->expert_bytes);
        }
        if(slot>=0) {
            unsigned char *cached=expert_cache_data(b->cache,slot);
            if(!hit) {
                if(cudaMemcpyAsync(cached,raw,lease->expert_bytes,cudaMemcpyDeviceToDevice,b->copy_stream)!=cudaSuccess)return -1;
                b->cache_stats.fill_copy_bytes=saturated_add(b->cache_stats.fill_copy_bytes,lease->expert_bytes);
            }
            raw=cached;
        }
        s->host_raw[i]=raw;
        for(int role=0;role<3;role++)s->weights[i*3+role].data=raw+(role==2?2:role)*lease->gate_bytes;
    }
    size_t metadata=(size_t)count*sizeof(unsigned char*);
    if(cudaMemcpyAsync(s->device_raw,s->host_raw,metadata,cudaMemcpyHostToDevice,b->copy_stream)!=cudaSuccess)return -1;
    b->stats.metadata_upload_bytes=saturated_add(b->stats.metadata_upload_bytes,metadata);return 0;
}
#endif
