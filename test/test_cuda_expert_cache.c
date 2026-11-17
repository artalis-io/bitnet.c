#include "gpu_backend.h"
#include "gpu_cuda.h"
#include "backend_session.h"
#include "quant.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

typedef struct {
    BnBackendSession *session;
    void *request;
    int nt;
    BnGPUMoEBankResidentPlan bank;
    BnGPUMoEBankInputPlan input;
} Request;
static BnGPUResidentView alloc(BnGPUBackend *g,void *r,int type,size_t n) {
    BnGPUResidentView v;assert(!g->resident_alloc(r,type,n,&v));return v;
}
static void init(Request *r,BnGPUBackend *g,int nt,void *cache) {
    memset(r,0,sizeof(*r));r->nt=nt;r->session=bn_backend_session_create();assert(r->session);
    r->request=bn_gpu_backend_resident_request(g,r->session);assert(r->request);
    BnGPUMoEBankConfig c={2,8,256,32,319488,319488,0,0};
    assert(!g->moe_bank_reserve(g->ctx,r->request,&c));
    assert(!g->moe_bank_input_reserve(g->ctx,r->request,nt,(size_t)nt*256*22));
    assert(!g->moe_bank_grouped_reserve(g->ctx,r->request,nt,192,(size_t)nt*10*36+192));
    assert(!g->linear_a8_bf16_reserve(g->ctx,r->request,(size_t)nt*256));
    if(cache)assert(!g->expert_cache_bind(r->request,cache,128,128));
    r->input.n_tokens=nt;r->input.x=alloc(g,r->request,BN_GGUF_TENSOR_BF16,(size_t)nt*256);
    r->input.slots=alloc(g,r->request,BN_GGUF_TENSOR_I32,(size_t)nt*10);
    r->input.status=alloc(g,r->request,BN_GGUF_TENSOR_I32,1);
    uint16_t *x=malloc((size_t)nt*256*2);int32_t *slots=malloc((size_t)nt*10*4);assert(x&&slots);
    for(int t=0;t<nt;t++)for(int j=0;j<256;j++)x[t*256+j]=(uint16_t)(0x3f80+t%7);
    for(int j=0;j<10;j++)for(int t=0;t<nt;t++)slots[j*nt+t]=t*10+j;
    int32_t zero=0;
    assert(!g->resident_write(r->request,r->input.x,x));
    assert(!g->resident_write(r->request,r->input.slots,slots));
    assert(!g->resident_write(r->request,r->input.status,&zero));free(x);free(slots);
    r->bank.total_tokens=nt;r->bank.grouped=1;
    r->bank.x=alloc(g,r->request,BN_GGUF_TENSOR_BF16,(size_t)nt*10*256);
    r->bank.gate=alloc(g,r->request,BN_GGUF_TENSOR_BF16,(size_t)nt*10*32);
    r->bank.up=alloc(g,r->request,BN_GGUF_TENSOR_BF16,(size_t)nt*10*32);
    r->bank.out=alloc(g,r->request,BN_GGUF_TENSOR_BF16,(size_t)nt*10*256);
}
static void offsets(int *out,int base,int nt) {
    out[0]=0;for(int i=0;i<512;i++)out[i+1]=out[i]+(i>=base&&i<base+10?nt:0);
}
static BnGPUExpertCacheLayer begin(Request *r,BnGPUBackend *g,int layer,int prefill,int base,int hits) {
    int off[513];offsets(off,base,r->nt);BnGPUExpertCacheLayer ticket;int status;
    while((status=g->expert_cache_begin(r->request,layer,prefill,off,r->nt,&ticket))==BN_GPU_MOE_BANK_BUSY) { thrd_yield(); }
    assert(!status);
    int count=0;for(int i=0;i<512;i++)count+=(int)((ticket.hits[i/64]>>(i%64))&1);
    assert(count==hits);return ticket;
}
static void fill(void *dst,size_t bytes,int type,int expert) {
    const BnQuantFormatOps *f=bn_quant_format_ops(type);assert(f&&bytes%f->bytes_per_block==0);
    memset(dst,0x11+expert%7,bytes);
    for(size_t i=0;i<bytes;i+=f->bytes_per_block) {
        uint16_t d=0x1000,zero=0; /* Keep activation sums finite in the FP16 A8 header. */memcpy((unsigned char*)dst+i,&d,2);
        if(type!=BN_GGUF_TENSOR_Q8_0)memcpy((unsigned char*)dst+i+2,&zero,2);
    }
}
static void execute(Request *r,BnGPUBackend *g,int gt,int dt,int base,const BnGPUExpertCacheLayer *ticket) {
    assert(!g->moe_bank_prepare_input(g->ctx,r->request,&r->input,&r->bank.prepared_input));
    for(int j=0;j<10;) {
        BnGPUMoEBankLease lease;int status;
        while((status=g->moe_bank_acquire(r->request,(unsigned)(j/8)%2,gt,dt,&lease))==BN_GPU_MOE_BANK_BUSY) { thrd_yield(); }
    assert(!status);
        int count=10-j;if(count>8)count=8;
        memset(lease.data,0,lease.expert_bytes*(size_t)count); /* Hits must ignore this poison. */
        BnGPUMoEBankItem items[8];
        for(int i=0;i<count;i++) {
            int expert=base+j+i;
            if(!ticket||!((ticket->hits[expert/64]>>(expert%64))&1)) {
                unsigned char *raw=(unsigned char*)lease.data+(size_t)i*lease.expert_bytes;
                fill(raw,lease.gate_bytes,gt,expert);fill(raw+lease.gate_bytes,lease.gate_bytes,gt,expert);
                fill(raw+lease.gate_bytes*2,lease.down_bytes,dt,expert);
            }
            items[i]=(BnGPUMoEBankItem){expert,(j+i)*r->nt,r->nt};
        }
        r->bank.lease=lease;r->bank.experts=items;r->bank.n_experts=count;
        assert(!g->moe_bank_submit(g->ctx,r->request,&r->bank));j+=count;
    }
}
static void run(Request *r,Request *control,BnGPUBackend *g,int gt,int dt,int layer,int prefill,int base,int hits) {
    execute(control,g,gt,dt,base,NULL);
    BnGPUExpertCacheStats before,after;BnGPUMoEBankStats b0,b1;
    assert(!g->expert_cache_stats(r->request,&before)&&!g->moe_bank_stats(r->request,&b0));
    BnGPUExpertCacheLayer ticket=begin(r,g,layer,prefill,base,hits),bad=ticket;bad.generation++;
    assert(g->expert_cache_end(r->request,&bad)==-1);
    execute(r,g,gt,dt,base,&ticket);assert(!g->expert_cache_end(r->request,&ticket));
    assert(g->expert_cache_end(r->request,&ticket)==-1);
    size_t bytes=(size_t)r->nt*10*256*2;uint16_t *actual=malloc(bytes),*expected=malloc(bytes);assert(actual&&expected);
    assert(!g->resident_read(r->request,r->bank.out,actual)&&!g->resident_read(control->request,control->bank.out,expected));
    for(size_t i=0;i<bytes/2;i++)assert((actual[i]&0x7f80)!=0x7f80);
    assert(!memcmp(actual,expected,bytes));
    if(!memcmp(actual,actual+(size_t)r->nt*256,(size_t)r->nt*256*2)) {
        fprintf(stderr,"CACHE_FIXTURE_IDENTICAL gt%d dt%d nt%d layer%d base%d hits%d first=%04x second=%04x\n",gt,dt,r->nt,layer,base,hits,actual[0],actual[(size_t)r->nt*256]);
    }
    assert(memcmp(actual,actual+(size_t)r->nt*256,(size_t)r->nt*256*2));
    free(actual);free(expected);
    assert(!g->expert_cache_stats(r->request,&after)&&!g->moe_bank_stats(r->request,&b1));
    assert(after.hits-before.hits==(size_t)hits&&after.misses-before.misses==(size_t)(10-hits));
    assert(b1.upload_bytes-b0.upload_bytes==after.miss_bytes-before.miss_bytes);
    assert(after.resident_bytes<=after.reserved_device_bytes);
}
static void cases(BnGPUBackend *g,int gt,int dt,int nt) {
    Request control,r,peer;init(&control,g,nt,NULL);
    BnGPUMoEBankLease lease;assert(!g->moe_bank_acquire(control.request,0,gt,dt,&lease));
    size_t expert_bytes=lease.expert_bytes;assert(!g->moe_bank_cancel(control.request,&lease));
    int gate[2]={gt,gt},down[2]={dt,dt};
    BnGPUExpertCacheConfig cfg={2,512,256,32,gate,down,10*expert_bytes};
    void *cache=g->expert_cache_create(g->ctx,&cfg);assert(cache);
    init(&r,g,nt,cache);init(&peer,g,nt,cache);
    run(&r,&control,g,gt,dt,0,0,0,0);
    run(&r,&control,g,gt,dt,0,0,0,10);
    run(&peer,&control,g,gt,dt,0,0,0,10);
    run(&peer,&control,g,gt,dt,1,0,20,0); /* Same pool, different layer: evict. */
    run(&r,&control,g,gt,dt,0,0,0,0);
    run(&r,&control,g,gt,dt,0,0,5,5);
    run(&peer,&control,g,gt,dt,1,1,40,0); /* Prefill never admits or promotes. */
    run(&r,&control,g,gt,dt,0,0,5,10);
    BnGPUExpertCacheLayer hold=begin(&r,g,0,0,5,10),sentinel,unchanged;
    memset(&sentinel,0x5a,sizeof(sentinel));unchanged=sentinel;int off[513];offsets(off,5,nt);
    assert(g->expert_cache_begin(peer.request,0,0,off,nt,&sentinel)==BN_GPU_MOE_BANK_BUSY&&!memcmp(&sentinel,&unchanged,sizeof(sentinel)));
    BnGPUExpertCacheStats before,after;assert(!g->expert_cache_stats(peer.request,&before));
    run(&peer,&control,g,gt,dt,1,0,30,0); /* All slots pinned by another request: bypass. */
    assert(!g->expert_cache_stats(peer.request,&after)&&after.bypasses-before.bypasses==10);
    assert(g->expert_cache_end(r.request,&hold)==-1); /* Not all selected experts submitted. */
    assert(!g->moe_bank_acquire(r.request,0,gt,dt,&r.bank.lease));
    assert(!g->expert_cache_cancel(r.request));
    assert(g->moe_bank_cancel(r.request,&r.bank.lease)==-1); /* Cancel invalidates outstanding leases. */
    run(&r,&control,g,gt,dt,0,0,5,10);
    Request abandoned;init(&abandoned,g,nt,cache);(void)begin(&abandoned,g,0,0,5,10);
    bn_backend_session_free(abandoned.session); /* Open-layer teardown must release the model lease. */
    run(&r,&control,g,gt,dt,0,0,5,10);
    bn_backend_session_free(peer.session);bn_backend_session_free(r.session);bn_backend_session_free(control.session);
    g->expert_cache_destroy(cache);
}
typedef struct {
    int gate[3],down[3],calls,fail_at;
    size_t bytes;
} PreloadSource;
static int preload_read(void *ptr,int layer,int expert,void *dst,size_t bytes) {
    PreloadSource *source=ptr;
    assert(layer==source->calls/512&&expert==source->calls%512);
    source->calls++;
    if(source->fail_at==source->calls)return -1;
    const BnQuantFormatOps *gate=bn_quant_format_ops(source->gate[layer]);
    const BnQuantFormatOps *down=bn_quant_format_ops(source->down[layer]);
    size_t gb=8192/(size_t)gate->block_elems*gate->bytes_per_block;
    size_t db=8192/(size_t)down->block_elems*down->bytes_per_block;
    assert(bytes==gb*2+db);
    fill(dst,gb,source->gate[layer],expert);
    fill((unsigned char*)dst+gb,gb,source->gate[layer],expert);
    fill((unsigned char*)dst+gb*2,db,source->down[layer],expert);
    source->bytes+=bytes;return 0;
}
static void preload_cases(BnGPUBackend *g) {
    PreloadSource source={{BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q4_K},
        {BN_GGUF_TENSOR_Q5_1,BN_GGUF_TENSOR_Q8_0,BN_GGUF_TENSOR_Q8_0},0,513,0};
    BnGPUExpertCacheConfig cfg={3,512,256,32,source.gate,source.down,27262976};
    void *cache=g->expert_cache_create(g->ctx,&cfg);assert(cache&&g->expert_cache_preload);
    BnGPUExpertCachePreloadInfo out,sentinel;memset(&out,0x5a,sizeof(out));sentinel=out;
    assert(g->expert_cache_preload(cache,19967,preload_read,&source,&out)==-1);
    assert(!source.calls&&!memcmp(&out,&sentinel,sizeof(out)));
    Request r;init(&r,g,1,cache);
    assert(g->expert_cache_preload(cache,19968,preload_read,&source,&out)==-1&&!source.calls);
    bn_backend_session_free(r.session);
    assert(g->expert_cache_preload(cache,19968,preload_read,&source,&out)==-1&&source.calls==513);
    assert(!memcmp(&out,&sentinel,sizeof(out)));
    init(&r,g,1,cache);BnGPUExpertCacheStats stats;
    assert(!g->expert_cache_stats(r.request,&stats)&&!stats.resident_bytes&&!stats.resident_experts);
    bn_backend_session_free(r.session);
    source.calls=source.fail_at=0;source.bytes=0;
    assert(!g->expert_cache_preload(cache,19968,preload_read,&source,&out));
    assert(source.calls==1536&&source.bytes==27262976);
    assert(out.upload_bytes==source.bytes&&out.resident_bytes==source.bytes&&out.resident_experts==1536&&out.host_staging_bytes==19968);
    assert(g->expert_cache_preload(cache,19968,preload_read,&source,&out)==-1&&source.calls==1536);
    for(int matrix=0;matrix<2;matrix++) {
        Request control;init(&r,g,matrix?65:1,cache);init(&control,g,matrix?65:1,NULL);
        for(int layer=0;layer<3;layer++) {
            run(&r,&control,g,source.gate[layer],source.down[layer],layer,1,0,10);
            run(&r,&control,g,source.gate[layer],source.down[layer],layer,0,502,10);
        }
        assert(!g->expert_cache_stats(r.request,&stats)&&stats.resident_experts==1536&&stats.resident_bytes==27262976);
        assert(!stats.misses&&!stats.evictions&&!stats.bypasses&&!stats.fill_copy_bytes);
        bn_backend_session_free(r.session);bn_backend_session_free(control.session);
    }
    g->expert_cache_destroy(cache);
    cfg.device_budget_bytes--;
    cache=g->expert_cache_create(g->ctx,&cfg);assert(cache);source.calls=0;
    assert(g->expert_cache_preload(cache,19968,preload_read,&source,&out)==-1&&!source.calls);
    g->expert_cache_destroy(cache);
}
#include "cuda_expert_cache_group_checks.h"
int main(void) {
    BnGPUBackend *g=bn_gpu_cuda_create();assert(g);
    if(!g->expert_cache_create) {
        puts("CUDA expert cache skipped: optional precision profile unavailable");
        bn_gpu_cuda_destroy(g);return 0;
    }
    int gate[3]={BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q4_K};
    int down[3]={BN_GGUF_TENSOR_Q5_1,BN_GGUF_TENSOR_Q8_0,BN_GGUF_TENSOR_Q8_0};
    BnGPUExpertCacheConfig cfg={3,512,256,32,gate,down,27262976},bad;
#define REJECT(change) do {bad=cfg;change;assert(!g->expert_cache_create(g->ctx,&bad));}while(0)
    REJECT(bad.n_layers=65);REJECT(bad.n_layers=0);REJECT(bad.n_experts=511);REJECT(bad.dim=255);
    REJECT(bad.hidden_dim=33);REJECT(bad.gate_types=NULL);REJECT(bad.device_budget_bytes=0);
#undef REJECT
    void *cache=g->expert_cache_create(g->ctx,&cfg);assert(cache);
    Request r;init(&r,g,1,cache);BnGPUExpertCacheStats stats;
    assert(!g->expert_cache_stats(r.request,&stats)&&stats.reserved_device_bytes==27262976);
    int off[513];offsets(off,0,1);off[512]--;
    BnGPUExpertCacheLayer ticket,sentinel;memset(&ticket,0x5a,sizeof(ticket));sentinel=ticket;
    assert(g->expert_cache_begin(r.request,0,0,off,1,&ticket)==-1&&!memcmp(&ticket,&sentinel,sizeof(ticket)));
    offsets(off,0,1);assert(g->expert_cache_begin(r.request,0,2,off,1,&ticket)==-1);
    assert(g->expert_cache_begin(r.request,3,0,off,1,&ticket)==-1);
    assert(g->expert_cache_bind(r.request,cache,128,128)==-1);
    assert(!g->expert_cache_stats(r.request,&stats)&&!stats.hits&&!stats.misses);
    bn_backend_session_free(r.session);g->expert_cache_destroy(cache);
    cfg.device_budget_bytes=1;cache=g->expert_cache_create(g->ctx,&cfg);assert(cache);
    init(&r,g,1,cache);assert(!g->expert_cache_stats(r.request,&stats)&&!stats.reserved_device_bytes);
    bn_backend_session_free(r.session);g->expert_cache_destroy(cache);
    cache_group_checks(g);
    preload_cases(g);
    for(int i=0;i<3;i++)for(int matrix=0;matrix<2;matrix++)cases(g,gate[i],down[i],matrix?65:1);
    bn_gpu_cuda_destroy(g);puts("CUDA compact expert cache PASSED: formats, matrix/vector, hit/miss/evict, pins, stale leases and teardown");return 0;
}
