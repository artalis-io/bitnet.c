#ifndef BN_GPU_CUDA_EXPERT_CACHE_GROUP_CUH
#define BN_GPU_CUDA_EXPERT_CACHE_GROUP_CUH
/* Included after private cache begin. One owner pins the selected union;
 * borrowed participant pointers are cleared before any participant teardown. */
static int expert_cache_begin_group(void *ptr,int layer,const BnGPUExpertCacheGroupRow *rows,
    int count,BnGPUExpertCacheLayer *out) {
    ResidentRequest *owner=(ResidentRequest*)ptr;BankState *b=owner?(BankState*)owner->moe_banks:NULL;
    ExpertCache *c=b?b->cache:NULL;BankState *members[16]={};int active[512]={};uint64_t decode[8]={};int found=0;
    if(!c||c->failed||owner->failed||!out||!rows||count<1||count>16||layer<0||layer>=c->layers||
       b->cache_open||b->cache_group_owner||b->config.n_experts!=c->experts||b->config.k<1||b->config.k>32||b->config.k>c->experts)return -1;
    for(int i=0;i<count;i++) {
        ResidentRequest *r=(ResidentRequest*)rows[i].request;const int *o=rows[i].offsets;int nt=rows[i].n_tokens;
        BankState *v=r?(BankState*)r->moe_banks:NULL;members[i]=v;
        if(!v||r->failed||r->backend_owner!=owner->backend_owner||v->cache!=c||v->cache_open||v->cache_group_owner||
           v->config.n_experts!=c->experts||v->config.k!=b->config.k||!o||nt<1||nt>v->grouped_capacity||nt>INT_MAX/v->config.k||
           (rows[i].prefill!=0&&rows[i].prefill!=1)||o[0]||o[c->experts]!=nt*v->config.k)return -1;
        for(int j=0;j<i;j++)if(rows[j].request==r)return -1;
        for(unsigned slot=0;slot<v->config.slots;slot++)if(v->slots[slot].state==1)return -1;
        for(int e=0;e<c->experts;e++) {
            if(o[e]<0||o[e+1]<o[e]||o[e+1]-o[e]>nt)return -1;
            if(o[e+1]>o[e]){active[e]=1;if(!rows[i].prefill)decode[e/64]|=UINT64_C(1)<<(e%64);}
        }
        found|=r==owner;
    }
    if(!found||set_device(owner))return -1;
    ExpertCacheLayerState *state=&c->layer[layer];if(state->open)return BN_GPU_MOE_BANK_BUSY;
    int ready=expert_cache_ready(c,layer);if(ready<0)return -1;if(!ready)return BN_GPU_MOE_BANK_BUSY;
    if(c->epoch==UINT64_MAX)return -1;
    BnGPUExpertCacheLayer ticket={};ticket.request=owner;ticket.layer=layer;ticket.generation=++c->epoch;
    state->owner=owner;state->epoch=ticket.generation;state->open=1;
    for(int i=0;i<count;i++) {
        BankState *v=members[i];v->cache_group_owner=owner;v->cache_nt=rows[i].n_tokens;
        memcpy(v->cache_offsets,rows[i].offsets,((size_t)c->experts+1)*sizeof(int));
        b->cache_group_requests[i]=(ResidentRequest*)rows[i].request;
    }
    b->cache_group_count=count;memcpy(b->cache_group_decode,decode,sizeof(decode));
    size_t bytes=c->pool[c->layer_pool[layer]].bytes;
    for(int e=0;e<c->experts;e++) {
        b->cache_entries[e]=-1;b->cache_seen[e]=!active[e];if(!active[e])continue;
        int slot=c->lookup[layer*c->experts+e];
        if(slot>=0&&c->entry[slot].valid) {
            c->entry[slot].pin=ticket.generation;b->cache_entries[e]=slot;ticket.hits[e/64]|=UINT64_C(1)<<(e%64);
            b->cache_stats.hits++;b->cache_stats.hit_bytes+=bytes;
        }else {b->cache_stats.misses++;b->cache_stats.miss_bytes+=bytes;}
    }
    b->cache_layer=ticket;b->cache_open=1;b->cache_prefill=0;*out=ticket;return 0;
}
static int expert_cache_decode_selection(const BankState *b,int expert) {
    return b->cache_group_owner?((b->cache_group_decode[expert/64]>>(expert%64))&1)!=0:!b->cache_prefill;
}
#endif
