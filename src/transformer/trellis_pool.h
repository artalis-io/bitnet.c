/* Exact-shape retention charges all idle and live native request payload. */
typedef struct {
    const BnTransformerTrellisPool *owner;
    const BnModel *model;
    BnTransformerTrellisRequestConfig ceiling;
    BnTransformerTrellisPoolInfo info;
    struct {BnSession *session;int leased;BnTransformerTrellisRequestConfig config;BnTransformerTrellisRequestPlan plan;} slots[16];
} TrellisPool;
static TrellisPool *trellis_pool_get(const BnTransformerTrellisPool *pool) {
    TrellisPool *p=pool?pool->state:NULL;return p&&p->owner==pool?p:NULL;
}
static void trellis_pool_remove(TrellisPool *p,int index,int eviction) {
    bn_session_free(p->slots[index].session,NULL);
    p->info.reserved_device-=p->slots[index].plan.device_bytes;
    p->info.reserved_host-=p->slots[index].plan.host_bytes;p->info.reserved_pinned-=p->slots[index].plan.pinned_bytes;
    p->info.allocated_slots--;if(p->slots[index].leased)p->info.leased_slots--;
    if(eviction)p->info.evicted++;else p->info.discarded++;
    memset(p->slots+index,0,sizeof(p->slots[index]));
}
int bn_transformer_trellis_pool_init(BnTransformerTrellisPool *pool,const BnModel *model,
    const BnTransformerTrellisRequestConfig *ceiling,int max_slots,size_t device,size_t host,size_t pinned) {
    BnTransformerTrellisRequestPlan plan;
    if(!pool||pool->state||max_slots<1||max_slots>16||bn_transformer_trellis_request_plan(model,ceiling,&plan))return -1;
    TrellisPool *p=calloc(1,sizeof(*p));if(!p)return -1;
    p->owner=pool;p->model=model;p->ceiling=*ceiling;p->info.max_slots=max_slots;
    p->info.device_budget=device;p->info.host_budget=host;p->info.pinned_budget=pinned;pool->state=p;return 0;
}
int bn_transformer_trellis_pool_acquire(BnTransformerTrellisPool *pool,int max_tokens,int required,BnSession **out) {
    TrellisPool *p=trellis_pool_get(pool);if(!out)return -1;*out=NULL;
    if(!p||max_tokens<1||max_tokens>p->ceiling.max_tokens||required<1||required>p->ceiling.maxseq)return -1;
    int needed=required>max_tokens?required:max_tokens;
    if(needed>INT_MAX-255)return -1;
    BnTransformerTrellisRequestConfig c=p->ceiling;c.max_tokens=max_tokens;c.maxseq=(needed+255)/256*256;
    if(c.maxseq>p->ceiling.maxseq)return -1;
    BnTransformerTrellisRequestPlan q;
    if(bn_transformer_trellis_request_plan(p->model,&c,&q)||q.device_bytes>p->info.device_budget||
       q.host_bytes>p->info.host_budget||q.pinned_bytes>p->info.pinned_budget)return -1;
    for(int j=0;j<p->info.max_slots;j++)if(p->slots[j].session&&!p->slots[j].leased&&
       p->slots[j].config.max_tokens==c.max_tokens&&p->slots[j].config.maxseq==c.maxseq) {
        p->slots[j].leased=1;p->info.leased_slots++;p->info.reused++;*out=p->slots[j].session;return 0;
    }
    size_t live_device=0,live_host=0,live_pinned=0;
    for(int j=0;j<p->info.max_slots;j++)if(p->slots[j].leased) {
        live_device+=p->slots[j].plan.device_bytes;live_host+=p->slots[j].plan.host_bytes;live_pinned+=p->slots[j].plan.pinned_bytes;
    }
    if(p->info.leased_slots==p->info.max_slots||q.device_bytes>p->info.device_budget-live_device||
       q.host_bytes>p->info.host_budget-live_host||q.pinned_bytes>p->info.pinned_budget-live_pinned)return 1;
    while(p->info.allocated_slots==p->info.max_slots||q.device_bytes>p->info.device_budget-p->info.reserved_device||
       q.host_bytes>p->info.host_budget-p->info.reserved_host||q.pinned_bytes>p->info.pinned_budget-p->info.reserved_pinned) {
        int idle=-1;
        for(int j=0;j<p->info.max_slots;j++)if(p->slots[j].session&&!p->slots[j].leased&&
           (idle<0||p->slots[j].plan.device_bytes>p->slots[idle].plan.device_bytes))idle=j;
        if(idle<0)return -1;
        trellis_pool_remove(p,idle,1);
    }
    int slot=0;while(p->slots[slot].session)slot++;
    BnSession *s=bn_session_create_backend(p->model,NULL);if(!s)return -1;
    if(bn_transformer_trellis_prepare_request(p->model,s,&c)){bn_session_free(s,NULL);return -1;}
    p->slots[slot].session=s;p->slots[slot].leased=1;p->slots[slot].config=c;p->slots[slot].plan=q;
    p->info.created++;p->info.allocated_slots++;p->info.leased_slots++;
    p->info.reserved_device+=q.device_bytes;p->info.reserved_host+=q.host_bytes;p->info.reserved_pinned+=q.pinned_bytes;
    if(p->info.reserved_device>p->info.peak_device)p->info.peak_device=p->info.reserved_device;
    if(p->info.reserved_host>p->info.peak_host)p->info.peak_host=p->info.reserved_host;
    if(p->info.reserved_pinned>p->info.peak_pinned)p->info.peak_pinned=p->info.reserved_pinned;
    *out=s;return 0;
}
int bn_transformer_trellis_pool_release(BnTransformerTrellisPool *pool,BnSession *s,int discard) {
    TrellisPool *p=trellis_pool_get(pool);if(!p||!s||(discard!=0&&discard!=1))return -1;
    for(int j=0;j<p->info.max_slots;j++)if(p->slots[j].session==s) {
        if(!p->slots[j].leased)return -1;
        p->info.released++;
        if(discard){trellis_pool_remove(p,j,0);return 0;}
        if(bn_transformer_trellis_session_restart(p->model,s)) {
            p->info.restart_failures++;trellis_pool_remove(p,j,0);return 0;
        }
        p->slots[j].leased=0;p->info.leased_slots--;return 0;
    }
    return -1;
}
int bn_transformer_trellis_pool_info(const BnTransformerTrellisPool *pool,BnTransformerTrellisPoolInfo *out) {
    TrellisPool *p=trellis_pool_get(pool);if(!p||!out)return -1;*out=p->info;
    for(int j=0;j<16;j++) {
        out->max_tokens[j]=p->slots[j].session?p->slots[j].config.max_tokens:0;
        out->capacity[j]=p->slots[j].session?p->slots[j].config.maxseq:0;out->leased[j]=p->slots[j].leased;
    }
    return 0;
}
void bn_transformer_trellis_pool_free(BnTransformerTrellisPool *pool) {
    TrellisPool *p=trellis_pool_get(pool);if(!p)return;
    for(int j=0;j<p->info.max_slots;j++)if(p->slots[j].session)bn_session_free(p->slots[j].session,NULL);
    free(p);pool->state=NULL;
}
