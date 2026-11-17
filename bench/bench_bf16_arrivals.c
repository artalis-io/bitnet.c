/* Controlled engine-step arrivals through the library scheduler and KV pool.
 * Diagnostic mode writes all full heads; --timing buffers metadata until done. */
#include "generate_batch.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "backend_layout.h"
#include "gpu_cuda.h"
#include "platform.h"
#include "bf16_policy_io.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include <time.h>

typedef struct {int arrival,n,enqueue_step,lease_capacity;BnSession *lease;int32_t *prompt,*expected,*output;double *delivery,submitted,admitted,enqueued;} Request;
typedef struct {int count;uint64_t ids[16];int n[16],pos[16],role[16];double duration_ms;} Step;
typedef struct {int id,acquire;BnTransformerBF16PoolInfo info;} PoolEvent;
typedef struct {
    BnModel *model;BnGPUBackend *gpu;BnBackendSession *coordinator;
    BnTransformerBF16Pool pool;BnGenerateBatch runtime;
    Request requests[64];Step steps[4096];int calls,completed,outputs,wall_clock;
    const char *directory;unsigned char *raw;
    double start;
    int nr,sized_kv,pool_events;PoolEvent events[128];
} Context;
static int integer(FILE *f) {
    unsigned char b[4];assert(fread(b,1,4,f)==4);
    uint32_t n=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;
    assert(n<=INT32_MAX);return (int)n;
}
static int reject_cpu(void *p){(void)p;return -1;}
static double now(void *p){(void)p;return bn_platform_time_ms();}
static int acquire(void *ptr,BnGenerateBatchLease *out) {
    Context *c=ptr;BnSession *s=NULL;int rc=bn_transformer_bf16_pool_acquire(&c->pool,&s);
    if(!rc)*out=(BnGenerateBatchLease){s,s->state.logits,(size_t)c->model->config.vocab_size};
    return rc;
}
static void pool_event(Context *c,int id,int acquiring) {
    assert(c->pool_events<128);PoolEvent *e=&c->events[c->pool_events++];
    e->id=id;e->acquire=acquiring;assert(!bn_transformer_bf16_pool_info(&c->pool,&e->info));
}
static int acquire_request(void *ptr,const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out) {
    Context *c=ptr;assert(request&&request->id<(uint64_t)c->nr);
    BnSession *s=NULL;int rc=bn_transformer_bf16_pool_acquire_capacity(&c->pool,request->required_capacity,&s);
    if(!rc) {
        Request *r=&c->requests[request->id];assert(!r->lease);r->lease=s;
        r->lease_capacity=bn_transformer_bf16_capacity(bn_session_backend(s));
        assert(r->lease_capacity>=request->required_capacity);
        *out=(BnGenerateBatchLease){s,s->state.logits,(size_t)c->model->config.vocab_size};
        pool_event(c,(int)request->id,1);
    }
    return rc;
}
static void release(void *ptr,BnGenerateBatchLease lease,int discard) {
    Context *c=ptr;BnSession *s=lease.handle;
    if(discard)bn_backend_session_release_gpu_resources(bn_session_backend(s));
    assert(!bn_transformer_bf16_pool_release(&c->pool,s));
    if(c->sized_kv) {
        int id=0;while(id<c->nr&&c->requests[id].lease!=s)id++;
        assert(id<c->nr);c->requests[id].lease=NULL;pool_event(c,id,0);
    }
}
static int execute(void *ptr,const BnGenerateBatchExecutionRow *rows,int count) {
    Context *c=ptr;if(c->calls>=4096)return -1;
    BnTransformerBF16StepRow native[16];Step *trace=&c->steps[c->calls];trace->count=count;
    for(int i=0;i<count;i++) {
        const BnGenerateBatchExecutionRow *r=&rows[i];
        native[i]=(BnTransformerBF16StepRow){{bn_session_backend(r->lease.handle),r->tokens,r->n_tokens,r->position,
            r->lease.logits,r->lease.logits_count},r->decode_role?BN_TRANSFORMER_BF16_STEP_DECODE:BN_TRANSFORMER_BF16_STEP_PREFILL};
        trace->ids[i]=r->id;trace->n[i]=r->n_tokens;trace->pos[i]=r->position;trace->role[i]=r->decode_role;
    }
    if(bn_transformer_bf16_batch_step(c->model,c->gpu,c->coordinator,native,count,NULL,NULL))return -1;
    if(c->directory) {
        char path[4096];int n=snprintf(path,sizeof(path),"%s/logits-%d.bin",c->directory,c->calls);
        if(n<1||(size_t)n>=sizeof(path))return -1;
        FILE *f=fopen(path,"wb");if(!f)return -1;int bad=0;
        for(int i=0;i<count;i++) {
            size_t vocab=rows[i].lease.logits_count;
            for(size_t j=0;j<vocab;j++) {
                uint32_t word;memcpy(&word,rows[i].lease.logits+j,4);
                if(word&65535u)bad=1;
                c->raw[2*j]=(unsigned char)(word>>16);c->raw[2*j+1]=(unsigned char)(word>>24);
            }
            if(fwrite(c->raw,2,vocab,f)!=vocab)bad=1;
        }
        if(fclose(f))bad=1;
        if(bad)return -1;
    }
    c->calls++;return 0;
}
static void notify(void *ptr,const BnGenerateBatchEvent *event) {
    Context *c=ptr;assert(event->id<64);Request *r=&c->requests[event->id];
    if(event->kind==BN_GENERATE_BATCH_ADMITTED){r->submitted=event->submitted_ms-c->start;r->admitted=event->admitted_ms-c->start;}
    else if(event->kind==BN_GENERATE_BATCH_TOKEN) {
        assert(event->output_index>=0&&event->output_index<c->outputs);
        r->delivery[event->output_index]=event->timestamp_ms-c->start;
        if(event->complete)c->completed++;
    }
}
int main(int argc,char **argv) {
    int sized_kv=0,reclaim_prefill=0,valid_flags=argc>=5&&argc<=7;
    for(int i=5;i<argc;i++) {
        if(!strcmp(argv[i],"--sized-kv")&&!sized_kv)sized_kv=1;
        else if(!strcmp(argv[i],"--reclaim-prefill")&&!reclaim_prefill)reclaim_prefill=1;
        else valid_flags=0;
    }
    if(!valid_flags){fprintf(stderr,"usage: %s MODEL.gguf WORKLOAD.bin POLICY.json OUTPUT_DIR|--timing [--sized-kv] [--reclaim-prefill]\n",argv[0]);return 2;}
    Context *c=calloc(1,sizeof(*c));assert(c);
    c->sized_kv=sized_kv;
    BnTransformerBF16Policy policy;assert(!bench_policy_read(argv[3],&policy));
    FILE *f=fopen(argv[2],"rb");assert(f);char magic[4];assert(fread(magic,1,4,f)==4);
    assert(!memcmp(magic,"BGA1",4)||!memcmp(magic,"BGA2",4)||!memcmp(magic,"BGA3",4)||!memcmp(magic,"BGA4",4));
    c->wall_clock=!memcmp(magic,"BGA2",4)||!memcmp(magic,"BGA3",4);
    int nr=integer(f);c->nr=nr;c->outputs=integer(f);
    BnGenerateBatchLimits limits={0};limits.max_requests=integer(f);limits.max_prefills=integer(f);
    limits.max_chunk=integer(f);limits.token_budget=integer(f);limits.capacity=integer(f);int kv_mib=integer(f);
    if(!memcmp(magic,"BGA3",4)||!memcmp(magic,"BGA4",4))limits.mixed_token_budget=integer(f);
    BnGenerateBatchPlan empty;assert(!bn_generate_batch_plan(&limits,NULL,0,&empty));
    assert(nr>=1&&nr<=64&&c->outputs>=1&&c->outputs<=256&&limits.capacity<=8192&&kv_mib>=1&&(size_t)kv_mib<=SIZE_MAX/1048576);
    for(int i=0;i<nr;i++) {
        Request *r=&c->requests[i];r->arrival=integer(f);r->n=integer(f);
        assert((!i||r->arrival>=c->requests[i-1].arrival)&&r->arrival<=(c->wall_clock?3600000:4096)&&r->n>=1&&r->n<=limits.capacity&&c->outputs-1<=limits.capacity-r->n);
        r->prompt=malloc((size_t)r->n*4);
        if(!c->wall_clock)r->expected=malloc((size_t)c->outputs*4);
        r->output=malloc((size_t)c->outputs*4);r->delivery=calloc((size_t)c->outputs,sizeof(double));
        assert(r->prompt&&(c->wall_clock||r->expected)&&r->output&&r->delivery);
        for(int j=0;j<r->n;j++)r->prompt[j]=integer(f);
        if(!c->wall_clock)for(int j=0;j<c->outputs;j++)r->expected[j]=integer(f);
    }
    assert(fgetc(f)==EOF&&!ferror(f));fclose(f);
    BnGGUFFile *file=bn_gguf_open_file(argv[1]);assert(file);BnModel model={0};c->model=&model;
    assert(!bn_model_load_with_cpu_preparation(&model,file,limits.capacity,0,0,0));
    c->gpu=bn_gpu_cuda_create();assert(c->gpu);
    assert(!bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&model),c->gpu,&model.config,&model.weights,limits.token_budget,BN_BACKEND_BF16_MATH_MKL_VML_HA));
    c->gpu->prepare_cpu_operations=reject_cpu;
    c->coordinator=bn_backend_session_create();assert(c->coordinator);
    assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(&model,c->gpu,c->coordinator,limits.max_requests,limits.token_budget,&policy));
    assert(!bn_transformer_bf16_pool_init(&c->pool,&model,c->gpu,limits.max_requests,limits.max_chunk,limits.capacity,(size_t)kv_mib*1048576,&policy));
    if(strcmp(argv[4],"--timing")){c->directory=argv[4];c->raw=malloc((size_t)model.config.vocab_size*2);assert(c->raw);}
    BnGenerateBatchBackend backend={c,acquire,execute,release,now};
    if(c->sized_kv)assert(!bn_generate_batch_init_request_aware(&c->runtime,&limits,model.config.vocab_size,&backend,acquire_request,notify,c));
    else assert(!bn_generate_batch_init(&c->runtime,&limits,model.config.vocab_size,&backend,notify,c));
    if(reclaim_prefill)assert(!bn_generate_batch_set_prefill_policy(&c->runtime,BN_GENERATE_BATCH_PREFILL_RECLAIM));
    c->start=now(c);int next=0;
    for(int step=0;c->completed<nr;step++) {
        if(!c->wall_clock)assert(step<8192);
        double begin=now(c);
        while(next<nr&&(c->wall_clock?c->start+c->requests[next].arrival<=begin:c->requests[next].arrival<=step)) {
            Request *r=&c->requests[next];r->enqueue_step=c->calls;r->enqueued=now(c)-c->start;
            int accepted=c->wall_clock?bn_generate_batch_enqueue_at(&c->runtime,(uint64_t)next,r->prompt,r->n,r->output,c->outputs,c->start+r->arrival):
                bn_generate_batch_enqueue(&c->runtime,(uint64_t)next,r->prompt,r->n,r->output,c->outputs);
            assert(!accepted);next++;
        }
        int call=c->calls,result=bn_generate_batch_step(&c->runtime);assert(result>=0);
        if(c->calls>call)c->steps[call].duration_ms=now(c)-begin;
        if(c->wall_clock&&!result) {
            BnGenerateBatchInfo idle;assert(!bn_generate_batch_info(&c->runtime,&idle));
            assert(!idle.live&&!idle.queued&&next<nr);
            struct timespec delay={0,10000000};int slept=thrd_sleep(&delay,NULL);assert(slept==0||slept==-1);
        }
    }
    double elapsed=now(c)-c->start;
    BnGenerateBatchInfo info;BnTransformerBF16PoolInfo pool;BnTransformerBF16BatchInfo batch_info;
    assert(!bn_generate_batch_info(&c->runtime,&info)&&!info.failed&&!info.live&&!info.queued&&info.completed==(uint64_t)nr);
    assert(!bn_transformer_bf16_pool_info(&c->pool,&pool)&&!pool.leased_slots&&!pool.discarded);
    assert(!bn_transformer_bf16_batch_info(c->coordinator,&batch_info)&&!batch_info.failed);
    int correct=1,valid=1;
    for(int i=0;i<nr;i++)for(int j=0;j<c->outputs;j++) {
        int token=c->requests[i].output[j];if(token<0||token>=model.config.vocab_size)valid=0;
        if(!c->wall_clock&&token!=c->requests[i].expected[j])correct=0;
    }
    printf("{\"kind\":\"prefill_policy\",\"policy\":\"%s\"}\n",reclaim_prefill?"reclaim":"fixed");
    printf("{\"kind\":\"arrival_mode\",\"mode\":\"%s\",\"reference_checked\":%s,\"outputs_valid\":%s}\n",c->wall_clock?"wall_clock":"engine_step",c->wall_clock?"false":"true",valid?"true":"false");
    printf("{\"kind\":\"summary\",\"diagnostic\":%s,\"correct\":%s,\"requests\":%d,\"outputs_each\":%d,\"steps\":%d,\"peak_live\":%d,\"elapsed_ms\":%.6f,\"tokens_per_second\":%.6f,\"prefill_calls\":%llu,\"mixed_calls\":%llu,\"decode_calls\":%llu}\n",c->directory?"true":"false",c->wall_clock?"null":correct?"true":"false",nr,c->outputs,c->calls,info.peak_live,elapsed,1000.0*nr*c->outputs/elapsed,(unsigned long long)batch_info.prefill_calls,(unsigned long long)batch_info.mixed_calls,(unsigned long long)batch_info.decode_calls);
    printf("{\"kind\":\"kv_pool\",\"scope\":\"kv_payload_only\",\"created\":%llu,\"reused\":%llu,\"discarded\":%llu,\"leased_slots\":%d,\"allocated_slots\":%d,\"budget_bytes\":%zu,\"reserved_kv_bytes\":%zu,\"peak_kv_bytes\":%zu,\"kv_bytes_per_slot\":%zu}\n",(unsigned long long)pool.created,(unsigned long long)pool.reused,(unsigned long long)pool.discarded,pool.leased_slots,pool.allocated_slots,pool.kv_budget_bytes,pool.reserved_kv_bytes,pool.peak_kv_bytes,pool.kv_bytes_per_slot);
    if(c->sized_kv) {
        puts("{\"kind\":\"kv_admission\",\"mode\":\"request_capacity\"}");
        for(int i=0;i<c->pool_events;i++) {
            PoolEvent *e=&c->events[i];BnTransformerBF16PoolInfo *v=&e->info;
            printf("{\"kind\":\"kv_state\",\"event\":%d,\"request\":%d,\"action\":\"%s\",\"created\":%llu,\"reused\":%llu,\"evicted\":%llu,\"resized\":%llu,\"discarded\":%llu,\"reserved_kv_bytes\":%zu,\"peak_kv_bytes\":%zu,\"capacity\":[",i,e->id,e->acquire?"acquire":"release",(unsigned long long)v->created,(unsigned long long)v->reused,(unsigned long long)v->evicted,(unsigned long long)v->resized,(unsigned long long)v->discarded,v->reserved_kv_bytes,v->peak_kv_bytes);
            for(int j=0;j<16;j++)printf("%s%d",j?",":"",v->slot_capacity[j]);
            fputs("],\"leased\":[",stdout);for(int j=0;j<16;j++)printf("%s%d",j?",":"",v->slot_leased[j]);puts("]}");
        }
        for(int i=0;i<nr;i++)printf("{\"kind\":\"kv_lease\",\"id\":%d,\"required_capacity\":%d,\"lease_capacity\":%d}\n",i,c->requests[i].n+c->outputs-1,c->requests[i].lease_capacity);
    }
    for(int i=0;i<c->calls;i++) {
        Step *s=&c->steps[i];printf("{\"kind\":\"step\",\"call\":%d,\"duration_ms\":%.6f,\"rows\":[",i,s->duration_ms);
        for(int j=0;j<s->count;j++)printf("%s[%llu,%d,%d,%d]",j?",":"",(unsigned long long)s->ids[j],s->n[j],s->pos[j],s->role[j]);
        puts("]}");
    }
    for(int i=0;i<nr;i++) {
        Request *r=&c->requests[i];
        printf("{\"kind\":\"arrival\",\"id\":%d,\"offered_ms\":%.6f,\"worker_enqueue_ms\":%.6f,\"enqueue_step\":%d}\n",i,c->wall_clock?(double)r->arrival:r->submitted,r->enqueued,r->enqueue_step);
        printf("{\"kind\":\"request\",\"id\":%d,\"arrival_step\":%d,\"submitted_ms\":%.6f,\"admitted_ms\":%.6f,\"tokens\":[",i,c->wall_clock?r->enqueue_step:r->arrival,r->submitted,r->admitted);
        
        for(int j=0;j<c->outputs;j++)printf("%s%d",j?",":"",r->output[j]);
        fputs("],\"delivery_ms\":[",stdout);for(int j=0;j<c->outputs;j++)printf("%s%.6f",j?",":"",r->delivery[j]);puts("]}");
    }
    bn_generate_batch_free(&c->runtime);bn_transformer_bf16_pool_free(&c->pool);bn_backend_session_free(c->coordinator);
    bn_model_free(&model);bn_gpu_cuda_destroy(c->gpu);bn_gguf_free(file);
    for(int i=0;i<nr;i++){free(c->requests[i].prompt);free(c->requests[i].expected);free(c->requests[i].output);free(c->requests[i].delivery);}
    free(c->raw);free(c);return correct&&valid?0:1;
}
