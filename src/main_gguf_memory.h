/* Main-owned payload admission and request storage. Transformer/backend plans
 * own shapes and allocation walks; this adapter owns reservations/lifetimes. */
#include "main_serve.h"
static size_t gguf_memory_model_host(const GGUFBatch *b) {
    return b->model.config.bf16_layout==BN_MODEL_BF16_LAYOUT_GLM_MHC?256u<<20:64u<<20;
}
static int gguf_memory_frontend(GGUFBatch *b,size_t decoder_bytes) {
    size_t host=sizeof(*b),prompts,outputs,limit,ram=0,transport=0;
    if(b->model.config.seq_len<1||b->model.config.seq_len==INT_MAX||
       native_cli_mul((size_t)b->options.serving_host_mb,1048576,&limit)||
       (!b->dense&&native_cli_mul((size_t)b->options.ram_mb,1048576,&ram))||
       native_cli_mul((size_t)b->model.config.seq_len+1,(size_t)b->input.count,&prompts)||
       native_cli_mul(prompts,sizeof(int),&prompts)||
       native_cli_mul((size_t)(b->options.serve&&b->options.tokens>b->model.config.seq_len?
            b->model.config.seq_len:b->options.tokens),(size_t)b->input.count,&outputs)||
       (b->options.serve&&serve_runtime_plan(SERVE_RECORD_BYTES,&transport))||native_cli_add(&host,transport)||
       native_cli_mul(outputs,sizeof(int32_t),&outputs)||
       native_cli_add(&host,prompts)||native_cli_add(&host,outputs)||native_cli_add(&host,decoder_bytes)||
       native_cli_add(&host,b->piece_capacity)||native_cli_add(&host,4u*BN_JSON_MAX_BYTES))return -1;
    /* Input/model/tokenizer loading precede this serving phase. Retained input
     * gets a conservative64MiB bound, not a claim about earlier load peaks. */
    b->frontend_host=host;
    return native_cli_add(&host,ram)||host>limit?-1:0;
}
static int gguf_memory_init(GGUFBatch *b) {
    size_t host=b->frontend_host,device,ram=0,gpu_cache,host_limit,device_limit;
    if(native_cli_mul((size_t)b->options.serving_host_mb,1048576,&host_limit)||
       (!b->dense&&native_cli_mul((size_t)b->options.ram_mb,1048576,&ram))||
       native_cli_mul((size_t)b->options.gpu_mb,1048576,&gpu_cache)||
       native_cli_add(&host,ram)||host>=host_limit)return -1;
    if(b->options.serving_device_mb) {
        if(native_cli_mul((size_t)b->options.serving_device_mb,1048576,&device_limit))return -1;
    }else {
        size_t free_bytes,total;
        if(!b->gpu->memory_info||b->gpu->memory_info(b->gpu->ctx,&free_bytes,&total))return -1;
        device_limit=free_bytes-free_bytes/10; /* Snapshot, not physical-memory guarantee. */
    }
    if(gpu_cache>=device_limit)return -1;
    b->model_host_budget=host_limit-host;b->model_device_budget=device_limit-gpu_cache;
    BnGenerateBatchMemoryQuote shared;
    if(b->dense) {
        if(bn_backend_layout_bf16_memory_plan(b->gpu,&b->model.config,&b->model.weights,b->options.budget,
            BN_BACKEND_BF16_MATH_MKL_VML_HA,&b->model_quote)||
           native_cli_mul((size_t)b->options.kv_mb,1048576,&b->kv_limit))return -1;
        shared=(BnGenerateBatchMemoryQuote){{host,b->model_quote.device_bytes},{host,b->model_quote.device_peak_bytes}};
        if(native_cli_add(&shared.retained.host_bytes,b->model_quote.host_bytes)||
           native_cli_add(&shared.peak.host_bytes,b->model_quote.host_peak_bytes))return -1;
    }else {
        if(b->model_host_budget>gguf_memory_model_host(b))b->model_host_budget=gguf_memory_model_host(b);
        if(b->model_device_budget>((size_t)12<<30))b->model_device_budget=(size_t)12<<30;
        device=b->model_device_budget;
        if(native_cli_add(&device,gpu_cache)||native_cli_add(&host,b->model_host_budget))return -1;
        BnGenerateBatchMemory fixed={host,device};shared=(BnGenerateBatchMemoryQuote){fixed,fixed};
    }
    return bn_generate_batch_memory_init(&b->memory,(BnGenerateBatchMemory){host_limit,device_limit},shared);
}
static int gguf_memory_coordinator(GGUFBatch *b) {
    if(bn_transformer_bf16_batch_memory_plan(&b->model,b->gpu,b->options.live,b->options.budget,
        &b->policy,&b->coordinator_quote)||
       native_cli_add(&b->coordinator_quote.host_bytes,sizeof(BnSessionBackendState))||
       native_cli_add(&b->coordinator_quote.host_peak_bytes,sizeof(BnSessionBackendState)))return -1;
    BnGenerateBatchMemoryQuote shared={{b->frontend_host,b->model_quote.device_bytes},
        {b->frontend_host,b->model_quote.device_bytes}};
    if(native_cli_add(&shared.retained.host_bytes,b->model_quote.host_bytes)||
       native_cli_add(&shared.retained.host_bytes,b->coordinator_quote.host_bytes)||
       native_cli_add(&shared.retained.device_bytes,b->coordinator_quote.device_bytes)||
       native_cli_add(&shared.peak.host_bytes,b->model_quote.host_bytes)||
       native_cli_add(&shared.peak.host_bytes,b->coordinator_quote.host_peak_bytes)||
       native_cli_add(&shared.peak.device_bytes,b->coordinator_quote.device_bytes))return -1;
    return bn_generate_batch_memory_shared_prepare_as(&b->memory,shared);
}
static int gguf_memory_capacity(const GGUFBatch *b,int required) {
    if(required<1||required>b->model.config.seq_len)return -1;
    if(!b->dense)return b->model.config.seq_len;
    int capacity=1,needed=required>b->chunk?required:b->chunk;
    while(capacity<needed) {
        if(capacity>b->model.config.seq_len/2){capacity=b->model.config.seq_len;break;}
        capacity*=2;
    }
    return capacity;
}
static int gguf_memory_quote(GGUFBatch *b,int capacity,BnGenerateBatchMemoryQuote *out,
    BnTransformerBF16RequestMemory *backend) {
    size_t device,host,session_host;
    int status=b->dense?bn_transformer_bf16_dense_memory_plan(&b->model,b->gpu,b->chunk,capacity,&b->policy,backend):
        bn_backend_layout_bf16_glm_model(bn_model_backend(&b->model))?
        bn_transformer_bf16_glm_memory_plan(&b->model,b->chunk,capacity,backend):
        bn_transformer_bf16_flash_memory_plan(&b->model,b->gpu,b->chunk,capacity,backend);
    if(status||native_cli_mul((size_t)b->options.device_mb,1048576,&device)||
       native_cli_mul((size_t)b->options.host_mb,1048576,&host)||backend->device_bytes>device||
       backend->host_peak_bytes>host||native_cli_mul((size_t)b->model.config.vocab_size,sizeof(float),&session_host)||
       native_cli_add(&session_host,(size_t)b->model.config.seq_len*sizeof(int32_t))||
       native_cli_add(&session_host,2*SH_ARENA_ALIGN+sizeof(BnSession)+sizeof(BnSessionBackendState)+sizeof(SHArena)))return -1;
    *out=(BnGenerateBatchMemoryQuote){{backend->host_bytes,backend->device_bytes},{backend->host_peak_bytes,backend->device_bytes}};
    return native_cli_add(&out->retained.host_bytes,session_host)||native_cli_add(&out->peak.host_bytes,session_host)?-1:0;
}
static int gguf_memory_ready(GGUFBatch *b) {
    BnGenerateBatchMemory retained;
    if(b->dense)retained=b->memory.shared;
    else {
        const BnBackendBF16GLMModel *glm=bn_backend_layout_bf16_glm_model(bn_model_backend(&b->model));
        const BnBackendBF16FlashIOModel *flash=bn_backend_layout_bf16_flash_model(bn_model_backend(&b->model));
        const BnBackendBF16FlashFFNModel *ffn=bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(&b->model));
        retained=(BnGenerateBatchMemory){b->frontend_host,glm?glm->bytes:flash?flash->model_bytes:0};
        size_t ram,gpu_cache,cache_host=glm?glm->expert_cache_host_bytes:ffn?ffn->expert_cache_host_bytes:0;
        if(!retained.device_bytes||native_cli_mul((size_t)b->options.ram_mb,1048576,&ram)||
           native_cli_mul((size_t)b->options.gpu_mb,1048576,&gpu_cache)||
           native_cli_add(&retained.host_bytes,ram)||native_cli_add(&retained.host_bytes,cache_host)||
           native_cli_add(&retained.device_bytes,gpu_cache))return -1;
    }
    int capacity=gguf_memory_capacity(b,b->options.serve?1:b->jobs[0].n+b->options.tokens-1);
    if(bn_generate_batch_memory_shared_commit_as(&b->memory,retained)||capacity<1||
       gguf_memory_quote(b,capacity,&b->request_quote,&b->backend_quote)||
       (b->dense&&bn_transformer_bf16_dense_kv_bytes(&b->model,capacity)>b->kv_limit))return -1;
    BnGenerateBatchMemoryTicket ticket;
    if(bn_generate_batch_memory_reserve(&b->memory,b->request_quote,&ticket))return -1;
    return bn_generate_batch_memory_drop(&b->memory,ticket);
}
static int gguf_memory_drop(GGUFBatch *b,int slot,int eviction) {
    BnSession *s=b->memory_slots[slot].session;
    if(!s)return -1;
    bn_session_free(s,NULL);
    if(b->dense)b->kv_reserved-=bn_transformer_bf16_dense_kv_bytes(&b->model,b->memory_slots[slot].capacity);
    b->memory_slots[slot].session=NULL;b->memory_slots[slot].leased=0;b->memory_slots[slot].capacity=0;
    if(eviction)b->evicted++;
    return bn_generate_batch_memory_drop(&b->memory,b->memory_slots[slot].ticket);
}
static int gguf_memory_largest_idle(const GGUFBatch *b,int except) {
    int best=-1;
    for(int i=0;i<16;i++)if(i!=except&&b->memory_slots[i].session&&!b->memory_slots[i].leased&&
        (best<0||b->memory_slots[i].capacity>b->memory_slots[best].capacity))best=i;
    return best;
}
static int gguf_memory_acquire(GGUFBatch *b,int required,BnSession **out) {
    *out=NULL;int capacity=gguf_memory_capacity(b,required),best=-1;
    if(capacity<1)return -1;
    for(int i=0;i<16;i++)if(b->memory_slots[i].session&&!b->memory_slots[i].leased&&
        b->memory_slots[i].capacity>=capacity&&(best<0||b->memory_slots[i].capacity<b->memory_slots[best].capacity))best=i;
    if(best>=0) {
        BnGenerateBatchMemoryTicket renewed;
        if(bn_generate_batch_memory_reuse(&b->memory,b->memory_slots[best].ticket,&renewed))return -1;
        b->memory_slots[best].ticket=renewed;
        if(bn_transformer_bf16_session_restart(&b->model,b->memory_slots[best].session)) {
            b->discarded++;(void)gguf_memory_drop(b,best,0);return -1;
        }
        b->memory_slots[best].leased=1;b->reused++;*out=b->memory_slots[best].session;return 0;
    }
    BnGenerateBatchMemoryQuote quote;BnTransformerBF16RequestMemory backend;
    if(gguf_memory_quote(b,capacity,&quote,&backend))return -1;
    int feasible=bn_generate_batch_memory_check_active(&b->memory,quote);
    if(feasible) {
        if(feasible==1&&b->memory.busy_returns!=UINT64_MAX)b->memory.busy_returns++;
        return feasible;
    }
    size_t kv=b->dense?bn_transformer_bf16_dense_kv_bytes(&b->model,capacity):0,active_kv=0;int active=0;
    if(b->dense&&(!kv||kv>b->kv_limit))return -1;
    for(int i=0;i<16;i++)if(b->memory_slots[i].leased) {
        active++;if(b->dense)active_kv+=bn_transformer_bf16_dense_kv_bytes(&b->model,b->memory_slots[i].capacity);
    }
    if(active>=b->options.live||(b->dense&&kv>b->kv_limit-active_kv)) {
        if(b->memory.busy_returns!=UINT64_MAX)b->memory.busy_returns++;
        return 1;
    }
    int grow=b->dense&&b->gpu->resident_resize_discard?gguf_memory_largest_idle(b,-1):-1;
    if(grow>=0) {
        size_t old_kv=bn_transformer_bf16_dense_kv_bytes(&b->model,b->memory_slots[grow].capacity);
        BnGenerateBatchMemoryTicket renewed;int status;
        for(;;) {
            status=kv>b->kv_limit-(b->kv_reserved-old_kv)?1:
                bn_generate_batch_memory_replace_idle(&b->memory,b->memory_slots[grow].ticket,quote,&renewed);
            if(status!=1)break;
            int idle=gguf_memory_largest_idle(b,grow);if(idle<0)break;
            if(gguf_memory_drop(b,idle,1))return -1;
        }
        if(status<0)return -1;
        if(!status) {
            b->memory_slots[grow].ticket=renewed;BnSession *s=b->memory_slots[grow].session;
            if(bn_transformer_bf16_session_restart(&b->model,s)||bn_transformer_bf16_resize_idle_kv(&b->model,s,capacity)||
                bn_generate_batch_memory_commit(&b->memory,renewed)) {
                b->discarded++;(void)gguf_memory_drop(b,grow,0);return -1;
            }
            b->kv_reserved=b->kv_reserved-old_kv+kv;
            if(b->kv_reserved>b->kv_peak)b->kv_peak=b->kv_reserved;
            b->memory_slots[grow].capacity=capacity;b->memory_slots[grow].leased=1;
            b->request_quote=quote;b->backend_quote=backend;b->reused++;b->resized++;*out=s;return 0;
        }
        /* Tight budget: destroy idle old storage before fresh preparation.
         * Feasibility above guarantees this cannot evict a live lease. */
        if(gguf_memory_drop(b,grow,1))return -1;
    }
    BnGenerateBatchMemoryTicket ticket;
    for(;;) {
        int slots=0;for(int i=0;i<16;i++)slots+=b->memory_slots[i].session!=NULL;
        int status=(slots>=b->options.live||(b->dense&&kv>b->kv_limit-b->kv_reserved))?1:
            bn_generate_batch_memory_reserve(&b->memory,quote,&ticket);
        if(status<0)return -1;
        if(!status)break;
        int idle=gguf_memory_largest_idle(b,-1);if(idle<0)return 1;
        if(gguf_memory_drop(b,idle,1))return -1;
    }
    BnSession *s=bn_session_create_backend(&b->model,NULL);size_t device,host;
    if(!s||native_cli_mul((size_t)b->options.device_mb,1048576,&device)||
       native_cli_mul((size_t)b->options.host_mb,1048576,&host))goto failed;
    if(b->dense) {
        if(bn_transformer_bf16_request_init_with_policy(&b->model,b->gpu,bn_session_backend(s),b->chunk,capacity,NULL,0,&b->policy))goto failed;
    }else if(bn_backend_layout_bf16_glm_model(bn_model_backend(&b->model))) {
        if(bn_transformer_bf16_glm_prepare_context(&b->model,s,b->chunk,capacity,device,host))goto failed;
        BnTransformerBF16GLMRequestInfo info;
        if(bn_transformer_bf16_glm_info(bn_session_backend(s),&info)||info.device_bytes!=backend.device_bytes||info.host_bytes!=backend.host_bytes)goto failed;
    }else {
        if(bn_transformer_bf16_flash_prepare_context(&b->model,b->gpu,bn_session_backend(s),b->chunk,capacity,device,host))goto failed;
        BnTransformerBF16FlashInfo info;
        if(bn_transformer_bf16_flash_info(bn_session_backend(s),&info)||info.gpu_bytes!=backend.device_bytes||
           info.host_bytes!=backend.host_bytes||info.preparation_host_peak_bytes!=backend.host_peak_bytes)goto failed;
    }
    if(bn_generate_batch_memory_commit(&b->memory,ticket))goto failed;
    b->memory_slots[ticket.slot].session=s;b->memory_slots[ticket.slot].ticket=ticket;
    b->memory_slots[ticket.slot].leased=1;b->memory_slots[ticket.slot].capacity=capacity;
    b->kv_reserved+=kv;if(b->kv_reserved>b->kv_peak)b->kv_peak=b->kv_reserved;
    b->request_quote=quote;b->backend_quote=backend;b->created++;*out=s;return 0;
failed:
    if(s)bn_session_free(s,NULL);
    (void)bn_generate_batch_memory_drop(&b->memory,ticket);return -1;
}
static int gguf_memory_release(GGUFBatch *b,BnSession *s,int discard) {
    for(int i=0;i<16;i++)if(b->memory_slots[i].session==s) {
        if(!b->memory_slots[i].leased)return -1;
        if(discard||bn_transformer_bf16_position(bn_session_backend(s))<0) {
            b->discarded++;return gguf_memory_drop(b,i,0);
        }
        if(bn_generate_batch_memory_idle(&b->memory,b->memory_slots[i].ticket))return -1;
        b->memory_slots[i].leased=0;return 0;
    }
    return -1;
}
static void gguf_memory_free(GGUFBatch *b) {
    for(int i=0;i<16;i++)if(b->memory_slots[i].session&&gguf_memory_drop(b,i,0))b->failed=1;
}
static void gguf_memory_report(const GGUFBatch *b) {
    int slots=0,idle=0;for(int i=0;i<16;i++)if(b->memory_slots[i].session){slots++;idle+=!b->memory_slots[i].leased;}
    fprintf(stderr,"cuda_batch_memory scope=managed_serving_payload host_limit=%zu device_limit=%zu shared_host_upper=%zu shared_device_upper=%zu reserved_host=%zu reserved_device=%zu peak_reserved_host=%zu peak_reserved_device=%zu busy=%llu idle_retained=1 slots=%d idle=%d request_host=%zu request_host_peak=%zu request_device=%zu variable_capacity=%d\n",
        b->memory.limit.host_bytes,b->memory.limit.device_bytes,b->memory.shared.host_bytes,b->memory.shared.device_bytes,
        b->memory.used.host_bytes,b->memory.used.device_bytes,b->memory.peak.host_bytes,b->memory.peak.device_bytes,(unsigned long long)b->memory.busy_returns,
        slots,idle,b->request_quote.retained.host_bytes,b->request_quote.peak.host_bytes,b->request_quote.retained.device_bytes,b->dense);
}
