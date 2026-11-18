#ifndef BN_GPU_CUDA_EXPERT_CACHE_CUH
#define BN_GPU_CUDA_EXPERT_CACHE_CUH
/* Included inside bn_cuda_moe_resident. Model-owned compact raw pools. The
 * caller serializes metadata operations; layer epochs and completion events
 * protect asynchronous consumers across request streams. */
struct ExpertCacheEntry {
    int layer, expert, pool, index, prev, next, valid;
    uint64_t pin, created;
};
struct ExpertCachePool {
    unsigned char *data;
    size_t bytes;
    int count, first, free_head, head, tail;
};
struct ExpertCacheLayerState {
    ResidentRequest *owner;
    cudaEvent_t completed;
    uint64_t epoch;
    int open, pending;
};
struct ExpertCache {
    BnCudaCtx *owner;
    int device, layers, experts, dim, hidden, pools, entries, failed, bindings;
    size_t reserved;
    uint64_t epoch;
    ExpertCachePool pool[4];
    ExpertCacheEntry *entry;
    int *lookup;
    int layer_pool[64], gate_type[64], down_type[64];
    ExpertCacheLayerState layer[64];
};

static void expert_cache_unlink(ExpertCache *c, int slot) {
    ExpertCacheEntry *e=&c->entry[slot]; ExpertCachePool *p=&c->pool[e->pool];
    if(e->prev>=0)c->entry[e->prev].next=e->next;else p->head=e->next;
    if(e->next>=0)c->entry[e->next].prev=e->prev;else p->tail=e->prev;
}
static void expert_cache_touch(ExpertCache *c,int slot) {
    ExpertCacheEntry *e=&c->entry[slot];ExpertCachePool *p=&c->pool[e->pool];
    e->prev=-1;e->next=p->head;
    if(p->head>=0)c->entry[p->head].prev=slot;else p->tail=slot;
    p->head=slot;
}
static void expert_cache_erase(ExpertCache *c,int slot) {
    ExpertCacheEntry *e=&c->entry[slot];ExpertCachePool *p=&c->pool[e->pool];
    c->lookup[e->layer*c->experts+e->expert]=-1;expert_cache_unlink(c,slot);
    e->layer=-1;e->valid=0;e->next=p->free_head;p->free_head=slot;
}
static void expert_cache_destroy(void *ptr) {
    ExpertCache *c=(ExpertCache*)ptr;if(!c)return;
    int previous=-1;(void)cudaGetDevice(&previous);
    if(cudaSetDevice(c->device)==cudaSuccess) {
        for(int l=0;l<c->layers;l++)if(c->layer[l].completed) {
            if(c->layer[l].pending)(void)cudaEventSynchronize(c->layer[l].completed);
            cudaEventDestroy(c->layer[l].completed);
        }
        for(int i=0;i<c->pools;i++)cudaFree(c->pool[i].data);
    }
    free(c->entry);free(c->lookup);free(c);
    if(previous>=0)(void)cudaSetDevice(previous);
}
static void *expert_cache_create(void *ctx_ptr,const BnGPUExpertCacheConfig *cfg) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;
    if(!ctx||!cfg||ctx->compute_capability!=1200||cfg->n_layers<=0||cfg->n_layers>64||
       cfg->n_experts<1||cfg->n_experts>512||!cfg->gate_types||!cfg->down_types||!cfg->device_budget_bytes||
       cfg->dim<=0||cfg->dim%256||cfg->hidden_dim<=0||cfg->hidden_dim%32)return NULL;
    size_t sizes[64],sum=0,gb,db;
    for(int l=0;l<cfg->n_layers;l++) {
        if(bank_pair_bytes(cfg->dim,cfg->hidden_dim,cfg->gate_types[l],cfg->down_types[l],&gb,&db,&sizes[l])||
           sizes[l]>SIZE_MAX-sum)return NULL;
        sum+=sizes[l];
    }
    ExpertCache *c=(ExpertCache*)calloc(1,sizeof(*c));if(!c)return NULL;
    c->owner=ctx;c->device=ctx->device;c->layers=cfg->n_layers;c->experts=cfg->n_experts;
    c->dim=cfg->dim;c->hidden=cfg->hidden_dim;
    size_t n=cfg->device_budget_bytes/sum;if(n>(size_t)c->experts)n=(size_t)c->experts;
    size_t left=cfg->device_budget_bytes-n*sum;
    for(int l=0;l<c->layers;l++) {
        int pool=0;while(pool<c->pools&&c->pool[pool].bytes!=sizes[l])pool++;
        if(pool==c->pools) {
            if(c->pools==4){expert_cache_destroy(c);return NULL;}
            c->pool[pool].bytes=sizes[l];c->pools++;
        }
        c->layer_pool[l]=pool;c->gate_type[l]=cfg->gate_types[l];c->down_type[l]=cfg->down_types[l];
        c->pool[pool].count+=(int)n;
        if(n<(size_t)c->experts&&left>=sizes[l]){c->pool[pool].count++;left-=sizes[l];}
    }
    for(int p=0;p<c->pools;p++) {
        ExpertCachePool *pool=&c->pool[p];pool->first=c->entries;
        pool->head=pool->tail=-1;pool->free_head=pool->count?pool->first:-1;
        c->entries+=pool->count;c->reserved+=pool->bytes*(size_t)pool->count;
    }
    c->entry=(ExpertCacheEntry*)calloc((size_t)(c->entries?c->entries:1),sizeof(*c->entry));
    c->lookup=(int*)malloc((size_t)c->layers*c->experts*sizeof(int));
    if(!c->entry||!c->lookup)goto fail;
    for(int i=0;i<c->layers*c->experts;i++)c->lookup[i]=-1;
    if(cudaSetDevice(c->device)!=cudaSuccess)goto fail;
    for(int p=0;p<c->pools;p++) {
        ExpertCachePool *pool=&c->pool[p];
        if(pool->count&&cudaMalloc((void**)&pool->data,pool->bytes*(size_t)pool->count)!=cudaSuccess)goto fail;
        for(int i=0;i<pool->count;i++) {
            ExpertCacheEntry *e=&c->entry[pool->first+i];e->layer=-1;e->pool=p;e->index=i;
            e->next=i+1<pool->count?pool->first+i+1:-1;
        }
    }
    for(int l=0;l<c->layers;l++)if(cudaEventCreateWithFlags(&c->layer[l].completed,cudaEventDisableTiming)!=cudaSuccess)goto fail;
    return c;
fail:
    expert_cache_destroy(c);(void)cudaGetLastError();return NULL;
}
static int expert_cache_memory(void *ptr,BnGPUExpertCacheMemory *out) {
    ExpertCache *c=(ExpertCache*)ptr;if(!c||!out)return -1;
    size_t entries=(size_t)(c->entries?c->entries:1);
    *out=BnGPUExpertCacheMemory{sizeof(*c)+entries*sizeof(*c->entry)+
        (size_t)c->layers*c->experts*sizeof(*c->lookup),c->reserved};return 0;
}
static int expert_cache_bind(void *ptr,void *cache,size_t host_budget,size_t device_budget) {
    ResidentRequest *r=(ResidentRequest*)ptr;ExpertCache *c=(ExpertCache*)cache;
    BankState *b=r?(BankState*)r->moe_banks:NULL;
    if(!b||!c||b->config.n_experts!=c->experts||b->config.k<1||b->config.k>32||b->config.k>c->experts||c->failed||c->owner!=r->backend_owner||c->device!=r->device||b->cache||
       !b->grouped_down||b->config.dim!=c->dim||b->config.hidden_dim!=c->hidden)return -1;
    size_t bytes=(size_t)b->config.experts_per_slot*sizeof(unsigned char*);
    if(bytes*b->config.slots>host_budget||bytes*b->config.slots>device_budget||set_device(r))return -1;
    const unsigned char **host[4]={},**device[4]={};
    for(unsigned i=0;i<b->config.slots;i++) {
        if(cudaMallocHost((void**)&host[i],bytes)!=cudaSuccess||cudaMalloc((void**)&device[i],bytes)!=cudaSuccess) {
            for(unsigned j=0;j<b->config.slots;j++) {
                if(host[j])cudaFreeHost(host[j]);if(device[j])cudaFree(device[j]);
            }
            (void)cudaGetLastError();return -1;
        }
    }
    for(unsigned i=0;i<b->config.slots;i++){b->slots[i].host_raw=host[i];b->slots[i].device_raw=device[i];}
    b->stats.reserved_group_host_bytes+=bytes*b->config.slots;
    b->stats.reserved_group_device_bytes+=bytes*b->config.slots;
    b->cache=c;c->bindings++;return 0;
}
static int expert_cache_preload(void *ptr,size_t host_budget,BnGPUExpertCacheRead read,
    void *source,BnGPUExpertCachePreloadInfo *out) {
    ExpertCache *c=(ExpertCache*)ptr;
    if(!c||c->failed||c->bindings||!read||!out||c->entries!=c->layers*c->experts)return -1;
    size_t largest=0;
    for(int p=0;p<c->pools;p++) {
        if(c->pool[p].head!=-1)return -1;
        if(c->pool[p].bytes>largest)largest=c->pool[p].bytes;
    }
    if(largest>host_budget)return -1;
    int previous=-1;(void)cudaGetDevice(&previous);
    if(cudaSetDevice(c->device)!=cudaSuccess)return -1;
    unsigned char *staging=NULL;cudaStream_t stream=NULL;
    int result=-1,used[4]={};
    BnGPUExpertCachePreloadInfo info={};
    if(cudaMallocHost((void**)&staging,largest)!=cudaSuccess||
       cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess)goto done;
    info.host_staging_bytes=largest;
    for(int l=0;l<c->layers;l++)for(int e=0;e<c->experts;e++) {
        int p=c->layer_pool[l];ExpertCachePool *pool=&c->pool[p];
        if(used[p]>=pool->count)goto done;
        if(read(source,l,e,staging,pool->bytes))goto done;
        if(cudaMemcpyAsync(pool->data+(size_t)used[p]*pool->bytes,staging,pool->bytes,
           cudaMemcpyHostToDevice,stream)!=cudaSuccess||cudaStreamSynchronize(stream)!=cudaSuccess) {
            c->failed=1;goto done;
        }
        used[p]++;info.upload_bytes+=pool->bytes;
    }
    /* Only metadata publication remains; all raw payloads have completed. */
    memset(used,0,sizeof(used));
    for(int l=0;l<c->layers;l++)for(int e=0;e<c->experts;e++) {
        int p=c->layer_pool[l],slot=c->pool[p].first+used[p]++;
        ExpertCacheEntry *entry=&c->entry[slot];
        entry->layer=l;entry->expert=e;entry->valid=1;
        c->lookup[l*c->experts+e]=slot;expert_cache_touch(c,slot);
        info.resident_experts++;info.resident_bytes+=c->pool[p].bytes;
    }
    for(int p=0;p<c->pools;p++)c->pool[p].free_head=-1;
    *out=info;result=0;
done:
    if(stream)cudaStreamDestroy(stream);
    if(staging)cudaFreeHost(staging);
    if(result)(void)cudaGetLastError();
    if(previous>=0)(void)cudaSetDevice(previous);
    return result;
}
static int expert_cache_ready(ExpertCache *c,int layer) {
    ExpertCacheLayerState *s=&c->layer[layer];
    if(!s->pending)return 1;
    cudaError_t e=cudaEventQuery(s->completed);
    if(e==cudaErrorNotReady)return 0;
    if(e!=cudaSuccess){c->failed=1;return -1;}
    s->pending=0;return 1;
}
static int expert_cache_begin(void *ptr,int layer,int prefill,const int *offsets,int nt,BnGPUExpertCacheLayer *out) {
    ResidentRequest *r=(ResidentRequest*)ptr;BankState *b=r?(BankState*)r->moe_banks:NULL;
    ExpertCache *c=b?b->cache:NULL;
    if(!c||b->config.n_experts!=c->experts||b->config.k<1||b->config.k>32||b->config.k>c->experts||c->failed||!out||!offsets||layer<0||layer>=c->layers||b->cache_open||b->cache_group_owner||
       (prefill!=0&&prefill!=1)||nt<=0||nt>b->grouped_capacity||nt>INT_MAX/b->config.k||offsets[0]||offsets[c->experts]!=nt*b->config.k||set_device(r))return -1;
    for(int e=0;e<c->experts;e++)if(offsets[e]<0||offsets[e+1]<offsets[e]||offsets[e+1]-offsets[e]>nt)return -1;
    ExpertCacheLayerState *state=&c->layer[layer];
    if(state->open)return BN_GPU_MOE_BANK_BUSY;
    int ready=expert_cache_ready(c,layer);if(ready<0)return -1;if(!ready)return BN_GPU_MOE_BANK_BUSY;
    if(c->epoch==UINT64_MAX)return -1;
    BnGPUExpertCacheLayer ticket={};ticket.request=r;ticket.layer=layer;ticket.generation=++c->epoch;
    state->owner=r;state->epoch=ticket.generation;state->open=1;
    size_t bytes=c->pool[c->layer_pool[layer]].bytes;
    for(int e=0;e<c->experts;e++) {
        int slot=offsets[e+1]>offsets[e]?c->lookup[layer*c->experts+e]:-1;
        b->cache_entries[e]=-1;b->cache_seen[e]=offsets[e+1]==offsets[e];
        if(offsets[e+1]==offsets[e])continue;
        if(slot>=0&&c->entry[slot].valid) {
            c->entry[slot].pin=ticket.generation;b->cache_entries[e]=slot;ticket.hits[e/64]|=UINT64_C(1)<<(e%64);
            b->cache_stats.hits++;b->cache_stats.hit_bytes+=bytes;
        } else {b->cache_stats.misses++;b->cache_stats.miss_bytes+=bytes;}
    }
    memcpy(b->cache_offsets,offsets,((size_t)c->experts+1)*sizeof(int));
    b->cache_layer=ticket;b->cache_open=1;b->cache_prefill=prefill;b->cache_nt=nt;*out=ticket;return 0;
}
#include "gpu_cuda_expert_cache_group.cuh"

/* -1 is intentional bypass, -2 is a CUDA failure. Never evict a pinned
 * current-layer hit, or any entry whose layer consumption is still pending. */
static int expert_cache_admit(BankState *b,int expert) {
    ExpertCache *c=b->cache;int layer=b->cache_layer.layer;
    ExpertCachePool *p=&c->pool[c->layer_pool[layer]];
    if(!expert_cache_decode_selection(b,expert))return -1;
    int slot=p->free_head;
    if(slot>=0)p->free_head=c->entry[slot].next;
    else {
        for(slot=p->tail;slot>=0;slot=c->entry[slot].prev) {
            ExpertCacheEntry *e=&c->entry[slot];ExpertCacheLayerState *s=&c->layer[e->layer];
            if(s->open&&e->pin==s->epoch)continue;
            int ready=expert_cache_ready(c,e->layer);if(ready<0)return -2;if(!ready)continue;
            break;
        }
        if(slot<0){b->cache_stats.bypasses++;return -1;}
        ExpertCacheEntry *old=&c->entry[slot];c->lookup[old->layer*c->experts+old->expert]=-1;
        expert_cache_unlink(c,slot);b->cache_stats.evictions++;
    }
    ExpertCacheEntry *e=&c->entry[slot];e->layer=layer;e->expert=expert;e->valid=0;
    e->pin=e->created=b->cache_layer.generation;c->lookup[layer*c->experts+expert]=slot;
    expert_cache_touch(c,slot);b->cache_entries[expert]=slot;return slot;
}
static unsigned char *expert_cache_data(ExpertCache *c,int slot) {
    ExpertCacheEntry *e=&c->entry[slot];return c->pool[e->pool].data+(size_t)e->index*c->pool[e->pool].bytes;
}
static int expert_cache_close(ResidentRequest *r,BankState *b,int cancel) {
    ExpertCache *c=b->cache;
    if(!b->cache_open)return cancel?0:-1;
    ExpertCacheLayerState *s=&c->layer[b->cache_layer.layer];
    if(!s->open||s->owner!=r||s->epoch!=b->cache_layer.generation)return -1;
    if(!cancel)for(int e=0;e<c->experts;e++)if(!b->cache_seen[e])return -1;
    if(cudaSetDevice(r->device)!=cudaSuccess||cudaEventRecord(s->completed,r->stream)!=cudaSuccess)c->failed=1;
    s->pending=!c->failed;s->open=0;s->owner=NULL;b->cache_open=0;
    for(unsigned i=0;i<b->config.slots;i++)if(b->slots[i].state==1)b->slots[i].state=0;
    for(int i=0;i<b->cache_group_count;i++) {
        BankState *peer=(BankState*)b->cache_group_requests[i]->moe_banks;
        peer->cache_group_owner=NULL;
        for(unsigned slot=0;slot<peer->config.slots;slot++)if(peer->slots[slot].state==1)peer->slots[slot].state=0;
        b->cache_group_requests[i]=NULL;
    }
    b->cache_group_count=0;
    if(cancel)for(int i=0;i<c->entries;i++)if(c->entry[i].layer>=0&&!c->entry[i].valid&&c->entry[i].created==s->epoch)
        expert_cache_erase(c,i);
    return c->failed?-1:0;
}
static int expert_cache_cancel(void *ptr) {
    ResidentRequest *r=(ResidentRequest*)ptr;BankState *b=r?(BankState*)r->moe_banks:NULL;
    if(b&&b->cache_group_owner){r=b->cache_group_owner;b=(BankState*)r->moe_banks;}
    return b&&b->cache?expert_cache_close(r,b,1):0;
}
static int expert_cache_end(void *ptr,const BnGPUExpertCacheLayer *ticket) {
    ResidentRequest *r=(ResidentRequest*)ptr;BankState *b=r?(BankState*)r->moe_banks:NULL;
    if(!b||!b->cache||!ticket||ticket->request!=r||ticket->layer!=b->cache_layer.layer||
       ticket->generation!=b->cache_layer.generation)return -1;
    return expert_cache_close(r,b,0);
}
static int expert_cache_stats(void *ptr,BnGPUExpertCacheStats *out) {
    ResidentRequest *r=(ResidentRequest*)ptr;BankState *b=r?(BankState*)r->moe_banks:NULL;
    if(!b||!out)return -1;*out=b->cache_stats;
    if(b->cache) {
        out->reserved_device_bytes=b->cache->reserved;
        for(int i=0;i<b->cache->entries;i++)if(b->cache->entry[i].layer>=0&&b->cache->entry[i].valid) {
            out->resident_experts++;out->resident_bytes+=b->cache->pool[b->cache->entry[i].pool].bytes;
        }
    }
    return 0;
}
#include "gpu_cuda_expert_cache_bank.cuh"
#endif
