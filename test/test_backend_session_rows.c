/* Reuse and run the lower-layer synthetic cache fixture, then exercise its
 * request-staging consumer with a deterministic backend and injected errors. */
#define main row_cache_base_main
#include "test_backend_model_rows.c"
#undef main
#include "backend_session.h"
#include "gpu_backend.h"
#ifndef __EMSCRIPTEN__
typedef struct {unsigned char *data;size_t count;int type;} StageBuffer;
typedef struct {unsigned char *data;int acquired;uint64_t generation;} StagePin;
typedef struct {
    StageBuffer buffers[12];StagePin pins[8];size_t cap;unsigned slots,nb;
    int reserve_calls,alloc_calls,acquires,cancels,submits,decodes;
    int reserve_fail,alloc_fail,submit_fail,decode_fail,busy_slot,acquire_fail,cancel_fail,bad_lease,alias_alloc;
    BnGPUQuantRowsBF16ResidentPlan last;
} StageRequest;
static void request_init(StageRequest *r) {
    memset(r,0,sizeof(*r));r->busy_slot=r->acquire_fail=r->cancel_fail=-1;
}
static void request_free(StageRequest *r) {
    for(unsigned i=0;i<r->nb;i++)free(r->buffers[i].data);
    for(unsigned i=0;i<r->slots;i++)free(r->pins[i].data);
    r->nb=r->slots=0;
}
static size_t width(int type){return type==BN_GGUF_TENSOR_I32?4:2;}
static unsigned char *pointer(StageRequest *r,BnGPUResidentView v) {
    assert(v.request==r&&v.slot<r->nb);StageBuffer *b=r->buffers+v.slot;
    assert(v.type==b->type&&v.offset<=b->count&&v.count<=b->count-v.offset);
    return b->data+v.offset*width(v.type);
}
static int reserve_pins(void *request,unsigned slots,size_t cap,size_t budget) {
    StageRequest *r=request;r->reserve_calls++;
    if(r->reserve_fail||slots>8||slots*cap>budget)return -1;
    assert(!r->slots);r->slots=slots;r->cap=cap;
    for(unsigned i=0;i<slots;i++){r->pins[i].data=malloc(cap);assert(r->pins[i].data);}
    return 0;
}
static int allocate_buffer(void *request,int type,size_t n,BnGPUResidentView *out) {
    StageRequest *r=request;r->alloc_calls++;if(r->alloc_calls==r->alloc_fail)return -1;
    assert(r->nb<12);unsigned slot=r->nb++;StageBuffer *b=r->buffers+slot;
    b->type=type;b->count=n+34;b->data=malloc(b->count*width(type));assert(b->data);
    memset(b->data,0xa5,b->count*width(type));*out=(BnGPUResidentView){r,17,n,slot,type};
    if(r->alloc_calls==r->alias_alloc)out->slot=0;
    return 0;
}
static int acquire_pin(void *request,unsigned slot,BnGPUHostUploadLease *out) {
    StageRequest *r=request;r->acquires++;assert(slot<r->slots);
    if((int)slot==r->busy_slot)return BN_GPU_UPLOAD_BUSY;
    if((int)slot==r->acquire_fail)return -1;
    StagePin *p=r->pins+slot;assert(!p->acquired);p->acquired=1;p->generation++;
    *out=(BnGPUHostUploadLease){r,p->data,r->cap,p->generation,slot};
    if(r->bad_lease)out->capacity--;
    return 0;
}
static int cancel_pin(void *request,const BnGPUHostUploadLease *lease) {
    StageRequest *r=request;r->cancels++;if((int)lease->slot==r->cancel_fail)return -1;
    assert(lease->request==r&&lease->slot<r->slots);r->pins[lease->slot].acquired=0;return 0;
}
static int submit_pin(void *request,const BnGPUHostUploadLease *lease,BnGPUResidentView out) {
    StageRequest *r=request;r->submits++;if(r->submits==r->submit_fail)return -1;
    StagePin *p=r->pins+lease->slot;assert(p->acquired&&p->generation==lease->generation);
    assert(out.count*width(out.type)<=r->cap);memcpy(pointer(r,out),p->data,out.count*width(out.type));p->acquired=0;return 0;
}
static int decode_rows(void *ctx,void *request,const BnGPUQuantRowsBF16ResidentPlan *p) {
    assert(ctx);StageRequest *r=request;r->decodes++;if(r->decode_fail)return -1;r->last=*p;
    const BnQuantRowScaledOps *ops=bn_quant_format_row_scaled_ops(p->type);assert(ops);
    return ops->decode_region_bf16(pointer(r,p->packed),p->packed.count*4,pointer(r,p->scales),p->scales.count*2,
        p->rows,p->cols,0,0,p->rows,p->cols,(uint16_t *)pointer(r,p->out),p->out.count);
}
static BnGPUBackend backend(void *ctx) {
    BnGPUBackend g={0};g.ctx=ctx;g.resident_alloc=allocate_buffer;g.resident_upload_reserve=reserve_pins;
    g.resident_upload_acquire=acquire_pin;g.resident_upload_cancel=cancel_pin;g.resident_upload_submit=submit_pin;
    g.quant_rows_bf16_resident=decode_rows;return g;
}
static BnBackendRowStageOptions options(void) {
    BnBackendRowStageOptions o={BN_QUANT_FORMAT_INT4_FP16S32,COLS,8,2,0,0,0};BnBackendRowStageWorkspace w;
    assert(!bn_backend_session_row_stage_plan(&o,&w));o.host_budget=w.host_bytes;o.pinned_budget=w.pinned_bytes;o.device_budget=w.device_bytes;
    assert(w.pinned_bytes==1024&&w.device_bytes==2624);return o;
}
static void check_guards(StageRequest *r) {
    for(unsigned j=0;j<r->nb;j++) {
        StageBuffer *b=r->buffers+j;size_t w=width(b->type);
        for(size_t i=0;i<17*w;i++)assert(b->data[i]==0xa5&&b->data[(b->count-17)*w+i]==0xa5);
    }
}
static void check_output(StageRequest *r,BnGPUResidentView out,const uint64_t *ids,size_t n) {
    unsigned char w[WB],s[SB];uint16_t expected_row[COLS];
    for(size_t i=0;i<n;i++) {
        expected(ids[i],w,s);assert(!bn_quant_int4_fp16s32_decode_region_bf16(w,WB,s,SB,1,COLS,0,0,1,COLS,expected_row,COLS));
        assert(!memcmp(pointer(r,out)+i*COLS*2,expected_row,COLS*2));
    }
    check_guards(r);
}
static void init_faults(BnGPUBackend *g,const BnBackendRowStageOptions *o) {
    for(int fail=0;fail<=7;fail++) {
        StageRequest r;request_init(&r);r.reserve_fail=fail==0;r.alloc_fail=fail;
        if(fail==7){r.alloc_fail=0;r.alias_alloc=4;}
        BnBackendRowStage stage={0};assert(bn_backend_session_row_stage_init(&stage,g,&r,o)==-1&&!stage.state);request_free(&r);
    }
    StageRequest r;request_init(&r);BnBackendRowStage stage={0};BnBackendRowStageOptions bad=*o;
    bad.device_budget--;assert(bn_backend_session_row_stage_init(&stage,g,&r,&bad)==-1&&!r.reserve_calls);
    bad=*o;bad.host_budget--;assert(bn_backend_session_row_stage_init(&stage,g,&r,&bad)==-1&&!r.reserve_calls);
    bad=*o;bad.pinned_budget--;assert(bn_backend_session_row_stage_init(&stage,g,&r,&bad)==-1&&!r.reserve_calls);
#ifdef BN_TEST_ROW_ALLOC
    deny_allocations=1;denied_allocations=0;assert(bn_backend_session_row_stage_init(&stage,g,&r,o)==-1&&!stage.state);
    deny_allocations=0;assert(denied_allocations==1&&!r.reserve_calls);
#endif
}
static void session_cases(BnGPUBackend *g,BnBackendRowCache *cache,const BnBackendRowStageOptions *o) {
    StageRequest r;request_init(&r);BnBackendRowStage stage={0};assert(!bn_backend_session_row_stage_init(&stage,g,&r,o));
    void *saved=stage.state;assert(!bn_backend_session_row_stage_init(&stage,g,&r,o)&&stage.state==saved&&r.alloc_calls==6&&r.reserve_calls==1);
    const uint64_t ids[]={1,2,3,1,2,3,1,2};BnGPUResidentView v[2],old;uint64_t epochs[2];
#ifdef BN_TEST_ROW_ALLOC
    deny_allocations=1;denied_allocations=0;
#endif
    assert(!bn_backend_session_row_stage_submit(&stage,cache,0,ids,8,v,epochs));
    assert(!bn_backend_session_row_stage_submit(&stage,cache,1,ids+2,2,v+1,epochs+1));
#ifdef BN_TEST_ROW_ALLOC
    deny_allocations=0;assert(!denied_allocations);
#endif
    assert(epochs[0]==1&&epochs[1]==1&&v[0].slot!=v[1].slot);check_output(&r,v[0],ids,8);check_output(&r,v[1],ids+2,2);
    old=v[0];assert(bn_backend_session_row_stage_current(&stage,0,1,&old));
    assert(!bn_backend_session_row_stage_submit(&stage,cache,0,ids+4,1,v,epochs));
    assert(epochs[0]==2&&!bn_backend_session_row_stage_current(&stage,0,1,&old));
    assert(bn_backend_session_row_stage_current(&stage,1,1,v+1));check_output(&r,v[1],ids+2,2);
    BnGPUResidentView unchanged=v[0];uint64_t epoch=99;int calls=r.acquires;
    uint64_t invalid[]={0,ROWS};
    assert(bn_backend_session_row_stage_submit(&stage,cache,0,invalid,2,v,&epoch)==-1&&r.acquires==calls&&epoch==99);
    assert(!memcmp(v,&unchanged,sizeof(*v)));
    assert(!bn_backend_session_row_stage_submit(&stage,NULL,0,NULL,0,NULL,NULL)&&r.acquires==calls);
    r.busy_slot=1;BnBackendRowCacheStats before,after;assert(!bn_backend_model_row_cache_stats(cache,&before));
    assert(bn_backend_session_row_stage_submit(&stage,cache,0,ids,2,v,&epoch)==BN_GPU_UPLOAD_BUSY);
    assert(!r.pins[0].acquired&&epoch==99&&!memcmp(v,&unchanged,sizeof(*v)));
    assert(!bn_backend_model_row_cache_stats(cache,&after)&&after.misses==before.misses&&after.hits==before.hits);
    r.busy_slot=-1;assert(!bn_backend_session_row_stage_submit(&stage,cache,0,ids,2,v,&epoch));
    assert(epoch==3&&bn_backend_session_row_stage_current(&stage,0,epoch,v));check_output(&r,*v,ids,2);
    StageRequest other;request_init(&other);BnBackendRowStage second={0};assert(!bn_backend_session_row_stage_init(&second,g,&other,o));
    BnGPUResidentView foreign;uint64_t fe;assert(!bn_backend_session_row_stage_submit(&second,cache,0,ids,2,&foreign,&fe));
    assert(!bn_backend_session_row_stage_current(&stage,0,epoch,&foreign));request_free(&other);bn_backend_session_row_stage_free(&second);
    request_free(&r);bn_backend_session_row_stage_free(&stage);bn_backend_session_row_stage_free(&stage);
}
static void backend_faults(BnGPUBackend *g,BnBackendRowCache *cache,const BnBackendRowStageOptions *o) {
    for(int fail=0;fail<7;fail++) {
        StageRequest r;request_init(&r);BnBackendRowStage stage={0};assert(!bn_backend_session_row_stage_init(&stage,g,&r,o));
        uint64_t ids[]={0,1};BnGPUResidentView out;uint64_t epoch;
        assert(!bn_backend_session_row_stage_submit(&stage,cache,1,ids,2,&out,&epoch));BnGPUResidentView saved=out;uint64_t original=epoch;
        int calls=r.submits;
        if(fail==0)r.submit_fail=r.submits+1;
        if(fail==1)r.submit_fail=r.submits+2;
        if(fail==2)r.decode_fail=1;
        if(fail==3)r.acquire_fail=1;
        if(fail==4){r.busy_slot=1;r.cancel_fail=0;}
        if(fail==5)r.bad_lease=1;
        if(fail==6)r.acquire_fail=0;
        assert(bn_backend_session_row_stage_submit(&stage,cache,0,ids,2,&out,&epoch)==-1);
        assert(epoch==original&&!memcmp(&saved,&out,sizeof(out))&&!bn_backend_session_row_stage_current(&stage,1,original,&saved));
        BnBackendRowStageInfo info;assert(!bn_backend_session_row_stage_info(&stage,&info)&&info.failed&&info.batches==1);
        if(fail>=3)assert(r.submits==calls);
        calls=r.acquires;assert(bn_backend_session_row_stage_submit(&stage,cache,1,ids,2,&out,&epoch)==-1&&r.acquires==calls);
        request_free(&r);bn_backend_session_row_stage_free(&stage);
    }
}
static void source_faults(BnGPUBackend *g,const BnBackendRowStageOptions *o,int invalid_scale) {
    char dir[]="/tmp/bn-row-stage-fault-XXXXXX",path[512];assert(mkdtemp(dir));fixture(dir,1);
    if(invalid_scale) {
        snprintf(path,sizeof(path),"%s/s.safetensors",dir);FILE *f=fopen(path,"r+b");assert(f);
        assert(!fseek(f,-SB,SEEK_END)&&fputc(invalid_scale==2?0xff:0,f)!=EOF&&fputc(invalid_scale==2?0x6f:0x80,f)!=EOF&&!fclose(f));
    }
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    size_t tensors[]={(size_t)bn_safetensors_index_find(&index,"packed"),(size_t)bn_safetensors_index_find(&index,"scales")};
    BnTrellisRowSource source={0};assert(!bn_trellis_row_source_init(&source,&index,BN_QUANT_FORMAT_INT4_FP16S32,tensors));
    BnBackendRowCache cache={0};assert(!bn_backend_model_row_cache_init(&cache,&source,4096,8));
    StageRequest r;request_init(&r);BnBackendRowStage stage={0};assert(!bn_backend_session_row_stage_init(&stage,g,&r,o));
    uint64_t id=0,epoch;BnGPUResidentView out;assert(!bn_backend_session_row_stage_submit(&stage,&cache,0,&id,1,&out,&epoch));
    BnGPUResidentView saved=out;uint64_t original=epoch;int calls=r.submits;
    if(!invalid_scale){snprintf(path,sizeof(path),"%s/w.safetensors",dir);FILE *f=fopen(path,"wb");assert(f&&!fclose(f));}
    id=ROWS-1;assert((invalid_scale==2?bn_backend_session_row_stage_submit_f16(&stage,&cache,0,&id,1,&out,&epoch):bn_backend_session_row_stage_submit(&stage,&cache,0,&id,1,&out,&epoch))==-1);
    assert(epoch==original&&!memcmp(&out,&saved,sizeof(out))&&r.submits==calls&&!r.pins[0].acquired&&!r.pins[1].acquired);
    assert(bn_backend_session_row_stage_current(&stage,0,original,&saved));
    BnBackendRowStageInfo info;assert(!bn_backend_session_row_stage_info(&stage,&info)&&!info.failed&&info.batches==1);
    id=0;assert(!bn_backend_session_row_stage_submit(&stage,&cache,0,&id,1,&out,&epoch)&&epoch==2);check_output(&r,out,&id,1);
    request_free(&r);bn_backend_session_row_stage_free(&stage);bn_backend_model_row_cache_free(&cache);
    bn_trellis_row_source_free(&source);bn_safetensors_index_free(&index);
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!unlink(path));
    snprintf(path,sizeof(path),"%s/w.safetensors",dir);assert(!unlink(path));snprintf(path,sizeof(path),"%s/s.safetensors",dir);assert(!unlink(path)&&!rmdir(dir));
}
static void metadata_upload(BnGPUBackend*g,BnBackendRowCache*cache,const BnBackendRowStageOptions*o) {
    for(int fault=0;fault<6;fault++) {
        StageRequest r;request_init(&r);BnBackendRowStage stage={0};
        assert(!bn_backend_session_row_stage_init(&stage,g,&r,o));BnBackendRowStageInfo info;
        assert(!bn_backend_session_row_stage_info(&stage,&info));size_t count=info.workspace.slot_bytes/4;
        int32_t*values=malloc((count+1)*sizeof(*values));assert(values);
        for(size_t i=0;i<=count;i++)values[i]=i%2?INT32_MAX:INT32_MIN;
        BnGPUResidentView dst;assert(!allocate_buffer(&r,BN_GGUF_TENSOR_I32,count,&dst));
        assert(bn_backend_session_row_stage_upload_i32(&stage,NULL,count,&dst)==-1);
        assert(bn_backend_session_row_stage_upload_i32(&stage,values,0,&dst)==-1);
        assert(bn_backend_session_row_stage_upload_i32(&stage,values,count+1,&dst)==-1);
        BnGPUResidentView alias={&r,17,1,0,BN_GGUF_TENSOR_I32};
        assert(bn_backend_session_row_stage_upload_i32(&stage,values,1,&alias)==-1&&!r.acquires);
        if(fault==1)r.busy_slot=0;
        if(fault==2)r.acquire_fail=0;
        if(fault==3)r.bad_lease=1;
        if(fault==4)r.submit_fail=1;
        if(fault==5){r.bad_lease=1;r.cancel_fail=0;}
        int status=bn_backend_session_row_stage_upload_i32(&stage,values,count,&dst);
        assert(status==(fault==0?0:fault==1?BN_GPU_UPLOAD_BUSY:-1));
        assert(!bn_backend_session_row_stage_info(&stage,&info)&&!info.batches&&!info.rows&&!info.epochs[0]);
        if(fault==1){assert(info.busy_returns==1&&!r.submits);r.busy_slot=-1;assert(!bn_backend_session_row_stage_upload_i32(&stage,values,count,&dst));}
        if(fault<2) {
            assert(!info.failed&&!memcmp(pointer(&r,dst),values,count*4));
            uint64_t id=0,epoch;BnGPUResidentView out;
            assert(!bn_backend_session_row_stage_submit(&stage,cache,0,&id,1,&out,&epoch)&&epoch==1);check_output(&r,out,&id,1);
        }else assert(info.failed&&bn_backend_session_row_stage_upload_i32(&stage,values,count,&dst)==-1);
        check_guards(&r);free(values);request_free(&r);bn_backend_session_row_stage_free(&stage);
    }
}
int main(void) {
    assert(!row_cache_base_main());char dir[]="/tmp/bn-row-stage-XXXXXX";assert(mkdtemp(dir));fixture(dir,1);
    char path[512];snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    size_t ids[]={(size_t)bn_safetensors_index_find(&index,"packed"),(size_t)bn_safetensors_index_find(&index,"scales")};BnTrellisRowSource source={0};
    assert(!bn_trellis_row_source_init(&source,&index,BN_QUANT_FORMAT_INT4_FP16S32,ids));BnBackendRowCache cache={0};assert(!bn_backend_model_row_cache_init(&cache,&source,4096,8));
    int ctx=1;BnGPUBackend g=backend(&ctx);BnBackendRowStageOptions o=options();init_faults(&g,&o);session_cases(&g,&cache,&o);backend_faults(&g,&cache,&o);
    source_faults(&g,&o,0);source_faults(&g,&o,1);source_faults(&g,&o,2);
    metadata_upload(&g,&cache,&o);
    bn_backend_model_row_cache_free(&cache);bn_trellis_row_source_free(&source);bn_safetensors_index_free(&index);
    assert(!unlink(path));snprintf(path,sizeof(path),"%s/w.safetensors",dir);assert(!unlink(path));snprintf(path,sizeof(path),"%s/s.safetensors",dir);assert(!unlink(path)&&!rmdir(dir));
    puts("Backend request row staging: budgets, banks, epochs, busy, partial-failure poisoning PASSED");return 0;
}
#else
int main(void){return row_cache_base_main();}
#endif
