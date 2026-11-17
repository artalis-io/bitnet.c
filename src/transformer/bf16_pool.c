#include "bf16_internal.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include <limits.h>
#include <stdlib.h>

typedef struct {
    const BnModel *model;
    BnGPUBackend *gpu;
    BnTransformerBF16Policy policy;
    int max_tokens,capacity;
    size_t kv_bytes_per_token;
    BnTransformerBF16PoolInfo info;
    struct {BnSession *session;int leased,capacity;} slots[16];
} BF16PoolState;

int bn_transformer_bf16_pool_init(BnTransformerBF16Pool *pool,const BnModel *model,
    BnGPUBackend *gpu,int max_slots,int max_tokens,int capacity,
    size_t kv_budget_bytes,const BnTransformerBF16Policy *policy) {
    if(!pool||pool->state||!model||!gpu||!policy||max_slots<1||max_slots>16||max_tokens<1||
       max_tokens>capacity||capacity<1||capacity>INT_MAX/1024||
       capacity>model->config.seq_len||model->config.n_layers!=64||
       model->config.bf16_layout!=BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM||
       !model->weights.layers||bn_model_gpu(model)!=gpu)return -1;
    size_t attention_layers=0;
    for(int l=0;l<64;l++) {
        if(model->weights.layers[l].block_kind==BN_LAYER_BLOCK_ATTENTION)attention_layers++;
        else if(model->weights.layers[l].block_kind!=BN_LAYER_BLOCK_SSM)return -1;
    }
    /* The admitted dense hybrid request allocates two BF16 vectors of 1024
     * elements per token per attention layer (bf16.c request initialization).
     * Charge full capacity, including currently inaccessible KV positions. */
    if(!attention_layers||(size_t)capacity>SIZE_MAX/4096/attention_layers)return -1;
    size_t bytes=(size_t)capacity*4096*attention_layers;
    if(kv_budget_bytes<bytes)return -1;
    BF16PoolState *p=calloc(1,sizeof(*p));if(!p)return -1;
    p->model=model;p->gpu=gpu;p->policy=*policy;p->max_tokens=max_tokens;p->capacity=capacity;
    p->kv_bytes_per_token=bytes/(size_t)capacity;
    p->info.max_slots=max_slots;p->info.kv_budget_bytes=kv_budget_bytes;p->info.kv_bytes_per_slot=bytes;
    pool->state=p;return 0;
}

static void remove_slot(BF16PoolState *p,int slot,int eviction) {
    bn_session_free(p->slots[slot].session,NULL);
    p->slots[slot].session=NULL;p->slots[slot].leased=0;
    p->info.allocated_slots--;p->info.reserved_kv_bytes-=(size_t)p->slots[slot].capacity*p->kv_bytes_per_token;
    p->slots[slot].capacity=0;
    if(eviction)p->info.evicted++;else p->info.discarded++;
}

static int acquire(BF16PoolState *p,int capacity,BnSession **out) {
    int best=-1;
    for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session&&!p->slots[i].leased&&
       p->slots[i].capacity>=capacity&&(best<0||p->slots[i].capacity<p->slots[best].capacity))best=i;
    if(best>=0) {
        if(bn_transformer_bf16_session_restart(p->model,p->slots[best].session)) {
            remove_slot(p,best,0);return -1;
        }
        p->slots[best].leased=1;p->info.leased_slots++;p->info.reused++;
        *out=p->slots[best].session;return 0;
    }
    size_t bytes=(size_t)capacity*p->kv_bytes_per_token,leased_bytes=0;
    for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].leased)
        leased_bytes+=(size_t)p->slots[i].capacity*p->kv_bytes_per_token;
    /* Prove feasibility before reclaiming anything. BUSY must not evict even
     * an idle cache entry when active leases prevent admission. */
    if(p->info.leased_slots==p->info.max_slots||bytes>p->info.kv_budget_bytes-leased_bytes)return 1;
    /* Reuse the non-KV request state when the backend supports destructive
     * storage resize. No adequate idle slot exists here, so this only grows KV. */
    int grow=-1;
    if(p->gpu->resident_resize_discard) {
        for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session&&!p->slots[i].leased&&
           (grow<0||p->slots[i].capacity>p->slots[grow].capacity))grow=i;
    }
    if(grow>=0) {
        size_t old=(size_t)p->slots[grow].capacity*p->kv_bytes_per_token;
        while(bytes>p->info.kv_budget_bytes-(p->info.reserved_kv_bytes-old)) {
            int idle=-1;
            for(int i=0;i<p->info.max_slots;i++)if(i!=grow&&p->slots[i].session&&!p->slots[i].leased&&
               (idle<0||p->slots[i].capacity>p->slots[idle].capacity))idle=i;
            if(idle<0)return -1;
            remove_slot(p,idle,1);
        }
        BnSession *s=p->slots[grow].session;
        if(bn_transformer_bf16_session_restart(p->model,s)||bn_transformer_bf16_resize_idle_kv(p->model,s,capacity)) {
            remove_slot(p,grow,0);return -1;
        }
        p->slots[grow].capacity=capacity;p->slots[grow].leased=1;
        p->info.reserved_kv_bytes=p->info.reserved_kv_bytes-old+bytes;
        if(p->info.reserved_kv_bytes>p->info.peak_kv_bytes)p->info.peak_kv_bytes=p->info.reserved_kv_bytes;
        p->info.leased_slots++;p->info.reused++;p->info.resized++;*out=s;return 0;
    }
    while(p->info.allocated_slots==p->info.max_slots||bytes>p->info.kv_budget_bytes-p->info.reserved_kv_bytes) {
        int largest=-1;
        for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session&&!p->slots[i].leased&&
           (largest<0||p->slots[i].capacity>p->slots[largest].capacity))largest=i;
        if(largest<0)return -1;
        remove_slot(p,largest,1);
    }
    int slot=0;while(p->slots[slot].session)slot++;
    BnSession *s=bn_session_create_backend(p->model,NULL);if(!s)return -1;
    if(bn_transformer_bf16_request_init_with_policy(p->model,p->gpu,bn_session_backend(s),
        p->max_tokens,capacity,NULL,0,&p->policy)) {
        bn_session_free(s,NULL);return -1;
    }
    p->slots[slot].session=s;p->slots[slot].leased=1;p->slots[slot].capacity=capacity;
    p->info.created++;p->info.allocated_slots++;p->info.leased_slots++;
    p->info.reserved_kv_bytes+=bytes;
    if(p->info.reserved_kv_bytes>p->info.peak_kv_bytes)p->info.peak_kv_bytes=p->info.reserved_kv_bytes;
    *out=s;return 0;
}

int bn_transformer_bf16_pool_acquire(BnTransformerBF16Pool *pool,BnSession **out) {
    BF16PoolState *p=pool?pool->state:NULL;
    if(!out)return -1;
    *out=NULL;if(!p)return -1;
    return acquire(p,p->capacity,out);
}

int bn_transformer_bf16_pool_acquire_capacity(BnTransformerBF16Pool *pool,int required_capacity,BnSession **out) {
    BF16PoolState *p=pool?pool->state:NULL;
    if(!out)return -1;
    *out=NULL;if(!p||required_capacity<1||required_capacity>p->capacity)return -1;
    int capacity=1,needed=required_capacity>p->max_tokens?required_capacity:p->max_tokens;
    while(capacity<needed) {
        if(capacity>p->capacity/2){capacity=p->capacity;break;}
        capacity*=2;
    }
    return acquire(p,capacity,out);
}

int bn_transformer_bf16_pool_release(BnTransformerBF16Pool *pool,BnSession *s) {
    BF16PoolState *p=pool?pool->state:NULL;
    if(!p||!s)return -1;
    for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session==s) {
        if(!p->slots[i].leased)return -1;
        p->info.leased_slots--;
        if(bn_transformer_bf16_position(bn_session_backend(s))<0)remove_slot(p,i,0);
        else p->slots[i].leased=0;
        return 0;
    }
    return -1;
}

int bn_transformer_bf16_pool_info(const BnTransformerBF16Pool *pool,BnTransformerBF16PoolInfo *out) {
    const BF16PoolState *p=pool?pool->state:NULL;
    if(!p||!out)return -1;
    *out=p->info;
    for(int i=0;i<16;i++){out->slot_capacity[i]=p->slots[i].capacity;out->slot_leased[i]=p->slots[i].leased;}
    return 0;
}

void bn_transformer_bf16_pool_free(BnTransformerBF16Pool *pool) {
    BF16PoolState *p=pool?pool->state:NULL;
    if(!p)return;
    for(int i=0;i<p->info.max_slots;i++)bn_session_free(p->slots[i].session,NULL);
    free(p);pool->state=NULL;
}
