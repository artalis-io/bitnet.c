#include "session.h"
#include "session_internal.h"
/* Main-owned exact-shape pool. Native shape/payload plans remain transformer
 * owned; aggregate admission and generations use the shared scheduler ledger. */
static int native_cli_memory_quote(const BnTransformerTrellisRequestPlan *plan,BnGenerateBatchMemoryQuote *out) {
    size_t host=plan->host_bytes;
    if(native_cli_add(&host,plan->pinned_bytes)||
       native_cli_add(&host,sizeof(BnSession)+sizeof(BnSessionBackendState)+sizeof(SHArena)))return -1;
    BnGenerateBatchMemory payload={host,plan->device_bytes};*out=(BnGenerateBatchMemoryQuote){payload,payload};return 0;
}
typedef struct NativeCLIMemoryPool {
    const struct NativeCLIMemoryPool *owner;const BnModel *model;BnGenerateBatchMemoryLedger *memory;
    BnTransformerTrellisRequestConfig ceiling;BnTransformerTrellisPoolInfo info;int failed;
    struct {BnSession *session;int leased;BnTransformerTrellisRequestConfig config;
        BnTransformerTrellisRequestPlan plan;BnGenerateBatchMemoryTicket ticket;} slots[16];
} NativeCLIMemoryPool;
static int native_cli_pool_init(NativeCLIMemoryPool *p,const BnModel *m,
    const BnTransformerTrellisRequestConfig *ceiling,int slots,size_t device,size_t host,size_t pinned,
    BnGenerateBatchMemoryLedger *memory) {
    BnTransformerTrellisRequestPlan plan;
    if(!p||p->owner||!memory||!memory->initialized||memory->shared_preparing||slots<1||slots>16||
       bn_transformer_trellis_request_plan(m,ceiling,&plan))return -1;
    NativeCLIMemoryPool next={0};next.owner=p;next.model=m;next.memory=memory;next.ceiling=*ceiling;next.info.max_slots=slots;
    next.info.device_budget=device;next.info.host_budget=host;next.info.pinned_budget=pinned;*p=next;return 0;
}
static int native_cli_pool_remove(NativeCLIMemoryPool *p,int i,int eviction) {
    bn_session_free(p->slots[i].session,NULL);
    int status=bn_generate_batch_memory_drop(p->memory,p->slots[i].ticket);
    p->info.reserved_device-=p->slots[i].plan.device_bytes;p->info.reserved_host-=p->slots[i].plan.host_bytes;
    p->info.reserved_pinned-=p->slots[i].plan.pinned_bytes;p->info.allocated_slots--;
    if(p->slots[i].leased)p->info.leased_slots--;
    if(eviction)p->info.evicted++;else p->info.discarded++;
    memset(p->slots+i,0,sizeof(p->slots[i]));if(status)p->failed=1;return status;
}
static int native_cli_pool_acquire(NativeCLIMemoryPool *p,int max_tokens,int required,BnSession **out) {
    if(!out)return -1;
    *out=NULL;
    if(!p||p->owner!=p||p->failed||max_tokens<1||max_tokens>p->ceiling.max_tokens||
       required<1||required>p->ceiling.maxseq)return -1;
    int needed=required>max_tokens?required:max_tokens;if(needed>INT_MAX-255)return -1;
    BnTransformerTrellisRequestConfig c=p->ceiling;c.max_tokens=max_tokens;c.maxseq=(needed+255)/256*256;
    BnTransformerTrellisRequestPlan q;BnGenerateBatchMemoryQuote quote;
    if(c.maxseq>p->ceiling.maxseq||bn_transformer_trellis_request_plan(p->model,&c,&q)||native_cli_memory_quote(&q,&quote)||
       q.device_bytes>p->info.device_budget||q.host_bytes>p->info.host_budget||q.pinned_bytes>p->info.pinned_budget)return -1;
    for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session&&!p->slots[i].leased&&
       p->slots[i].config.max_tokens==c.max_tokens&&p->slots[i].config.maxseq==c.maxseq) {
        BnGenerateBatchMemoryTicket renewed;
        if(bn_generate_batch_memory_reuse(p->memory,p->slots[i].ticket,&renewed))return -1;
        p->slots[i].ticket=renewed;p->slots[i].leased=1;p->info.leased_slots++;p->info.reused++;*out=p->slots[i].session;return 0;
    }
    int possible=bn_generate_batch_memory_check_active(p->memory,quote);
    if(possible) {
        if(possible==1&&p->memory->busy_returns!=UINT64_MAX)p->memory->busy_returns++;
        return possible;
    }
    size_t device=0,host=0,pinned=0;
    for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].leased) {
        device+=p->slots[i].plan.device_bytes;host+=p->slots[i].plan.host_bytes;pinned+=p->slots[i].plan.pinned_bytes;
    }
    if(p->info.leased_slots==p->info.max_slots||q.device_bytes>p->info.device_budget-device||
       q.host_bytes>p->info.host_budget-host||q.pinned_bytes>p->info.pinned_budget-pinned) {
        if(p->memory->busy_returns!=UINT64_MAX)p->memory->busy_returns++;
        return 1;
    }
    BnGenerateBatchMemoryTicket ticket;
    for(;;) {
        int status=p->info.allocated_slots==p->info.max_slots||q.device_bytes>p->info.device_budget-p->info.reserved_device||
            q.host_bytes>p->info.host_budget-p->info.reserved_host||q.pinned_bytes>p->info.pinned_budget-p->info.reserved_pinned?1:
            bn_generate_batch_memory_reserve(p->memory,quote,&ticket);
        if(status<0)return -1;
        if(!status)break;
        int idle=-1;
        for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session&&!p->slots[i].leased&&
            (idle<0||p->slots[i].plan.device_bytes>p->slots[idle].plan.device_bytes))idle=i;
        if(idle<0)return 1;
        if(native_cli_pool_remove(p,idle,1))return -1;
    }
    BnSession *s=bn_session_create_backend(p->model,NULL);
    if(!s||bn_transformer_trellis_prepare_request(p->model,s,&c)||bn_generate_batch_memory_commit(p->memory,ticket)) {
        if(s)bn_session_free(s,NULL);
        if(bn_generate_batch_memory_drop(p->memory,ticket))p->failed=1;
        return -1;
    }
    int i=0;while(p->slots[i].session)i++;
    p->slots[i].session=s;p->slots[i].config=c;p->slots[i].plan=q;p->slots[i].ticket=ticket;p->slots[i].leased=1;
    p->info.created++;p->info.allocated_slots++;p->info.leased_slots++;
    p->info.reserved_device+=q.device_bytes;p->info.reserved_host+=q.host_bytes;p->info.reserved_pinned+=q.pinned_bytes;
    if(p->info.reserved_device>p->info.peak_device)p->info.peak_device=p->info.reserved_device;
    if(p->info.reserved_host>p->info.peak_host)p->info.peak_host=p->info.reserved_host;
    if(p->info.reserved_pinned>p->info.peak_pinned)p->info.peak_pinned=p->info.reserved_pinned;
    *out=s;return 0;
}
static int native_cli_pool_release(NativeCLIMemoryPool *p,BnSession *s,int discard) {
    if(!p||p->owner!=p||p->failed||!s||(discard!=0&&discard!=1))return -1;
    for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session==s) {
        if(!p->slots[i].leased)return -1;
        p->info.released++;
        if(discard)return native_cli_pool_remove(p,i,0);
        if(bn_transformer_trellis_session_restart(p->model,s)) {
            p->info.restart_failures++;return native_cli_pool_remove(p,i,0);
        }
        if(bn_generate_batch_memory_idle(p->memory,p->slots[i].ticket))return -1;
        p->slots[i].leased=0;p->info.leased_slots--;return 0;
    }
    return -1;
}
static int native_cli_pool_info(const NativeCLIMemoryPool *p,BnTransformerTrellisPoolInfo *out) {
    if(!p||p->owner!=p||p->failed||!out)return -1;
    *out=p->info;
    for(int i=0;i<16;i++) {
        out->max_tokens[i]=p->slots[i].session?p->slots[i].config.max_tokens:0;
        out->capacity[i]=p->slots[i].session?p->slots[i].config.maxseq:0;out->leased[i]=p->slots[i].leased;
    }
    return 0;
}
static void native_cli_pool_free(NativeCLIMemoryPool *p) {
    if(!p||p->owner!=p)return;
    for(int i=0;i<p->info.max_slots;i++)if(p->slots[i].session)(void)native_cli_pool_remove(p,i,0);
    p->owner=NULL;
}
static void native_cli_memory_report(const BnGenerateBatchMemoryLedger *m,const char *name) {
    fprintf(stderr,"%s scope=managed_serving_payload host_limit=%zu device_limit=%zu shared_host_upper=%zu shared_device_upper=%zu reserved_host=%zu reserved_device=%zu peak_reserved_host=%zu peak_reserved_device=%zu busy=%llu idle_retained=1\n",
        name,m->limit.host_bytes,m->limit.device_bytes,m->shared.host_bytes,m->shared.device_bytes,
        m->used.host_bytes,m->used.device_bytes,m->peak.host_bytes,m->peak.device_bytes,(unsigned long long)m->busy_returns);
}
