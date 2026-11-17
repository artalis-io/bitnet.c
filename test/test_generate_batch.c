#include "generate_batch.h"
#include "generate_batch_plan_reference.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {int allocated,used,position;float logits[5];} Lease;
typedef struct {
    Lease leases[8];BnGenerateBatch *batch;
    int limit,created,reused,released,discarded,executes,trace;
    int check_policy_reentry,check_abort_reentry;
    int busy_acquire,fail_acquire,fail_execute,bad_logits;
    int admitted,tokens,completed,cancelled,failed,releases_before_execute;
    uint64_t expected_published;
    double clock;
    double submitted_ms[2],admitted_ms[2],first_delivery_ms[2];
    BnGenerateBatchEvent last_event;
    BnGenerateBatchAdmission admissions[16];int admission_calls;
    uint64_t next_admission;
} Probe;
static int acquire_request(void *context,const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out);
static int acquire(void *context,BnGenerateBatchLease *out) {
    Probe *p=context;if(p->fail_acquire)return -1;if(p->busy_acquire)return 1;
    for(int i=0;i<p->limit;i++)if(!p->leases[i].used) {
        Lease *l=&p->leases[i];if(l->allocated)p->reused++;else{l->allocated=1;p->created++;}
        l->used=1;l->position=0;*out=(BnGenerateBatchLease){l,l->logits,5};return 0;
    }
    return 1;
}
static int acquire_request(void *context,const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out) {
    Probe *p=context;assert(p->admission_calls<16);
    p->admissions[p->admission_calls++]=*request;
    return acquire(context,out);
}
static int forbidden_acquire(void *context,BnGenerateBatchLease *out) {
    (void)context;(void)out;assert(0);return -1;
}
static int acquire_stream(void *context,const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out) {
    Probe *p=context;assert(request->id==p->next_admission++);
    assert(request->prompt_tokens==1&&request->max_outputs==1&&request->required_capacity==1);
    p->admission_calls++;return acquire(context,out);
}
static int execute(void *context,const BnGenerateBatchExecutionRow *rows,int count) {
    Probe *p=context;p->releases_before_execute=p->released;
    if(p->check_policy_reentry)assert(bn_generate_batch_set_prefill_policy(p->batch,BN_GENERATE_BATCH_PREFILL_FIXED)==-1);
    BnGenerateBatchInfo info;assert(!bn_generate_batch_info(p->batch,&info));p->expected_published=info.output_tokens;
    for(int i=0;i<count;i++)p->expected_published+=(uint64_t)rows[i].emit_token;
    const BnGenerateBatchTrace *ref=p->trace?&generate_batch_trace[p->executes]:NULL;
    if(ref)assert(count==ref->count);
    for(int i=0;i<count;i++) {
        const BnGenerateBatchExecutionRow *r=&rows[i];Lease *l=r->lease.handle;
        assert(l->used&&l->position==r->position&&r->n_tokens>=1);
        if(ref)assert(r->id==(uint64_t)ref->id[i]&&r->n_tokens==ref->tokens[i]&&r->position==ref->position[i]&&r->decode_role==ref->role[i]);
        for(int j=0;j<r->n_tokens;j++)assert(r->tokens[j]==(int32_t)(r->id%5));
        for(int j=0;j<5;j++)l->logits[j]=j==(int)(r->id%5)?2.0f:0.0f;
        if(p->bad_logits&&i==count-1)l->logits[0]=NAN;
        l->position+=r->n_tokens;
    }
    p->executes++;return p->fail_execute?-1:0;
}
static void release(void *context,BnGenerateBatchLease lease,int discard) {
    Probe *p=context;Lease *l=lease.handle;assert(l&&l->used&&lease.logits==l->logits);
    l->used=0;p->released++;if(discard){l->allocated=0;p->discarded++;}
}
static double time_ms(void *context){return ++((Probe *)context)->clock;}
static void notify(void *context,const BnGenerateBatchEvent *e) {
    Probe *p=context;BnGenerateBatchInfo info;assert(!bn_generate_batch_info(p->batch,&info));
    if(p->check_abort_reentry)assert(bn_generate_batch_abort(p->batch)==-1);
    p->last_event=*e;
    assert(e->timestamp_ms>=e->submitted_ms);
    if(e->kind==BN_GENERATE_BATCH_ADMITTED){
        p->admitted++;assert(e->admitted_ms>=e->submitted_ms);
        if(e->id<2){p->submitted_ms[e->id]=e->submitted_ms;p->admitted_ms[e->id]=e->admitted_ms;}
    }
    else if(e->kind==BN_GENERATE_BATCH_TOKEN) {
        assert(p->released==p->releases_before_execute); /* whole cohort before reclamation */
        assert(e->token==(int)(e->id%5)&&e->output_index>=0);
        assert(info.output_tokens==p->expected_published);
        if(e->id<2&&!e->output_index)p->first_delivery_ms[e->id]=e->timestamp_ms;
        p->tokens++;p->completed+=e->complete;
    } else if(e->kind==BN_GENERATE_BATCH_CANCELLED)p->cancelled++;
    else {assert(e->kind==BN_GENERATE_BATCH_FAILED);p->failed++;}
}
static void init(BnGenerateBatch *batch,Probe *p,int live) {
    p->batch=batch;p->limit=live;
    BnGenerateBatchBackend backend={p,acquire,execute,release,time_ms};
    BnGenerateBatchLimits limits={live,live<4?live:4,64,256,8192,0};
    assert(!bn_generate_batch_init(batch,&limits,5,&backend,notify,p));
}
static void staggered(void) {
    BnGenerateBatch batch={0};Probe p={0};init(&batch,&p,8);p.trace=1;
    int32_t prompt[12][129],output[12][64];memset(output,0x5a,sizeof(output));
    for(int i=0;i<12;i++)for(int j=0;j<129;j++)prompt[i][j]=i%5;
    for(int step=0;step<134;step++) {
        int group=step==0?0:step==8?1:step==68?2:-1;
        if(group>=0)for(int i=4*group;i<4*group+4;i++)assert(!bn_generate_batch_enqueue(&batch,(uint64_t)i,prompt[i],group==1?5:129,output[i],64));
        assert(bn_generate_batch_step(&batch)==1);
    }
    BnGenerateBatchInfo info;assert(!bn_generate_batch_info(&batch,&info));
    assert(info.steps==134&&info.submitted==12&&info.completed==12&&info.output_tokens==768);
    assert(info.prefill_tokens==1052&&info.decode_tokens==756&&info.peak_live==8);
    assert(!info.queued&&!info.live&&!info.failed&&info.acquired==12&&info.released==12);
    assert(p.created==8&&p.reused==4&&p.released==12&&!p.discarded&&p.tokens==768&&p.completed==12);
    for(int i=0;i<12;i++)for(int j=0;j<64;j++)assert(output[i][j]==i%5);
    assert(!bn_generate_batch_step(&batch));bn_generate_batch_free(&batch);assert(!batch.state);
    bn_generate_batch_free(&batch);
    puts("Generate batch runtime: 134 independent cohorts, eight leases and four reuses PASSED");
}
static void failure_atomicity(int execute_failure,int prefix) {
    BnGenerateBatch batch={0};Probe p={0};init(&batch,&p,2);
    int32_t prompt[3][2]={{0,0},{1,1},{2,2}},output[3][3];memset(output,0x5a,sizeof(output));
    for(int i=0;i<3;i++)assert(!bn_generate_batch_enqueue(&batch,(uint64_t)i,prompt[i],2,output[i],3));
    if(prefix)assert(bn_generate_batch_step(&batch)==1);
    p.fail_execute=execute_failure;p.bad_logits=!execute_failure;
    assert(bn_generate_batch_step(&batch)==-1);
    BnGenerateBatchInfo info;assert(!bn_generate_batch_info(&batch,&info));
    assert(info.failed&&!info.live&&!info.queued&&info.output_tokens==(uint64_t)(prefix?2:0)&&info.steps==(uint64_t)prefix);
    assert(p.discarded==2&&p.failed==3&&p.tokens==(prefix?2:0));
    for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
        if(prefix&&i<2&&j==0)assert(output[i][j]==i);
        else {unsigned char *bytes=(unsigned char *)&output[i][j];for(size_t k=0;k<sizeof(output[i][j]);k++)assert(bytes[k]==0x5a);}
    }
    assert(bn_generate_batch_step(&batch)==-1);
    assert(bn_generate_batch_enqueue(&batch,9,prompt[0],2,output[0],3)==-1);
    bn_generate_batch_free(&batch);assert(p.released==2);
}
static void cancellation_and_exhaustion(void) {
    BnGenerateBatch batch={0};Probe p={0};init(&batch,&p,2);
    int32_t prompt[3][2]={{0,0},{1,1},{2,2}},out[3][3];
    for(int i=0;i<3;i++)assert(!bn_generate_batch_enqueue(&batch,(uint64_t)i,prompt[i],2,out[i],3));
    p.busy_acquire=1;assert(!bn_generate_batch_step(&batch));
    BnGenerateBatchInfo info;assert(!bn_generate_batch_info(&batch,&info)&&info.queued==3&&!info.live&&!p.executes);
    assert(!bn_generate_batch_cancel(&batch,2)&&p.cancelled==1);
    p.busy_acquire=0;assert(bn_generate_batch_step(&batch)==1);
    assert(!bn_generate_batch_cancel(&batch,0)&&p.cancelled==2&&p.released==1);
    assert(bn_generate_batch_cancel(&batch,0)==-1);
    assert(bn_generate_batch_step(&batch)==1&&bn_generate_batch_step(&batch)==1);
    assert(!bn_generate_batch_info(&batch,&info)&&info.completed==1&&info.cancelled==2&&!info.live&&!info.queued);
    bn_generate_batch_free(&batch);assert(p.released==2&&!p.discarded);
    memset(&p,0,sizeof(p));init(&batch,&p,2);
    assert(!bn_generate_batch_enqueue(&batch,0,prompt[0],2,out[0],3));p.fail_acquire=1;
    assert(bn_generate_batch_step(&batch)==-1&&p.failed==1&&!p.released);
    bn_generate_batch_free(&batch);
}
static void rejection_and_queue_bound(void) {
    BnGenerateBatch batch={0};Probe p={0};init(&batch,&p,2);
    int32_t prompt[2]={0,0},out[65][2];
    assert(!bn_generate_batch_enqueue(&batch,0,prompt,2,out[0],2));
    assert(bn_generate_batch_enqueue(&batch,0,prompt,2,out[1],2)==-1);
    assert(bn_generate_batch_enqueue(&batch,1,prompt,2,out[0],2)==-1);
    assert(bn_generate_batch_enqueue(&batch,1,prompt,2,prompt,2)==-1);
    int32_t invalid[2]={-1,0};assert(bn_generate_batch_enqueue(&batch,1,invalid,2,out[1],2)==-1);
    for(int i=1;i<64;i++)assert(!bn_generate_batch_enqueue(&batch,(uint64_t)i,prompt,2,out[i],2));
    assert(bn_generate_batch_enqueue(&batch,64,prompt,2,out[64],2)==1);
    BnGenerateBatchInfo info;assert(!bn_generate_batch_info(&batch,&info)&&info.submitted==64&&info.queued==64);
    assert(!bn_generate_batch_cancel(&batch,4));
    assert(!bn_generate_batch_enqueue(&batch,64,prompt,2,out[64],2));
    bn_generate_batch_free(&batch);assert(!p.released);
}
static void recorded_arrivals(void) {
    BnGenerateBatch batch={0};Probe p={0};init(&batch,&p,1);p.clock=1000;
    int32_t prompt[3][2]={{0,0},{1,1},{2,2}},out[3][2];memset(out,0x5a,sizeof(out));
    assert(!bn_generate_batch_enqueue_at(&batch,0,prompt[0],2,out[0],2,100));
    /* External timestamps account for delay; they do not reorder FIFO input. */
    assert(!bn_generate_batch_enqueue_at(&batch,1,prompt[1],2,out[1],2,90));
    BnGenerateBatchInfo before,after;assert(!bn_generate_batch_info(&batch,&before));
    double invalid[]={NAN,INFINITY,-INFINITY,p.clock+1000};
    for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++)
        assert(bn_generate_batch_enqueue_at(&batch,2,prompt[2],2,out[2],2,invalid[i])==-1);
    p.clock=NAN;
    assert(bn_generate_batch_enqueue(&batch,2,prompt[2],2,out[2],2)==-1);
    assert(!bn_generate_batch_info(&batch,&after));
    assert(before.submitted==after.submitted&&before.queued==after.queued&&!after.live&&!after.failed);
    for(size_t i=0;i<sizeof(out);i++)assert(((unsigned char *)out)[i]==0x5a);
    p.clock=1500;p.busy_acquire=1;assert(!bn_generate_batch_step(&batch));
    p.busy_acquire=0;
    for(int i=0;i<4;i++)assert(bn_generate_batch_step(&batch)==1);
    assert(p.completed==2&&p.submitted_ms[0]==100&&p.submitted_ms[1]==90);
    assert(p.first_delivery_ms[0]-p.submitted_ms[0]>=1400);
    assert(p.admitted_ms[1]>p.first_delivery_ms[0]);
    assert(!bn_generate_batch_enqueue_at(&batch,3,prompt[0],2,out[2],2,50));
    assert(!bn_generate_batch_cancel(&batch,3));
    assert(p.last_event.kind==BN_GENERATE_BATCH_CANCELLED&&p.last_event.submitted_ms==50);
    assert(!bn_generate_batch_enqueue_at(&batch,4,prompt[0],2,out[2],2,60));
    p.fail_acquire=1;assert(bn_generate_batch_step(&batch)==-1);
    assert(p.last_event.kind==BN_GENERATE_BATCH_FAILED&&p.last_event.submitted_ms==60);
    bn_generate_batch_free(&batch);
    puts("Generate batch recorded arrival queueing and rejection PASSED");
}
static void request_admission(void) {
    BnGenerateBatch batch={0};Probe p={0};p.batch=&batch;p.limit=1;
    BnGenerateBatchLimits limits={1,1,64,256,8192,0};
    BnGenerateBatchBackend backend={&p,forbidden_acquire,execute,release,time_ms};
    assert(bn_generate_batch_init_request_aware(&batch,&limits,5,&backend,NULL,notify,&p)==-1&&!batch.state);
    assert(!bn_generate_batch_init_request_aware(&batch,&limits,5,&backend,acquire_request,notify,&p));
    assert(!p.admission_calls);
    int32_t prompt[3][3]={{0,0,0},{1,1,1},{2,2,2}},out[3][3];
    assert(!bn_generate_batch_enqueue(&batch,0,prompt[0],2,out[0],3));
    assert(!bn_generate_batch_enqueue(&batch,1,prompt[1],1,out[1],1));
    assert(!bn_generate_batch_enqueue(&batch,2,prompt[2],3,out[2],2));
    p.busy_acquire=1;assert(!bn_generate_batch_step(&batch)&&p.admission_calls==1);
    p.busy_acquire=0;for(int i=0;i<6;i++)assert(bn_generate_batch_step(&batch)==1);
    assert(!bn_generate_batch_step(&batch)&&p.admission_calls==4&&p.completed==3);
    const uint64_t ids[]={0,0,1,2};const int capacities[]={4,4,1,4};
    for(int i=0;i<4;i++) {
        const BnGenerateBatchAdmission *r=&p.admissions[i];
        assert(r->id==ids[i]&&r->required_capacity==capacities[i]);
        assert(r->required_capacity==r->prompt_tokens+r->max_outputs-1);
    }
    bn_generate_batch_free(&batch);
    backend.acquire=NULL;
    assert(bn_generate_batch_init(&batch,&limits,5,&backend,notify,&p)==-1&&!batch.state);
    assert(!bn_generate_batch_init_request_aware(&batch,&limits,5,&backend,acquire_request,notify,&p));
    assert(!bn_generate_batch_enqueue(&batch,3,prompt[0],2,out[0],2));
    assert(!bn_generate_batch_cancel(&batch,3)&&p.admission_calls==4);
    assert(!bn_generate_batch_enqueue(&batch,0,prompt[0],2,out[0],2));
    p.fail_acquire=1;assert(bn_generate_batch_step(&batch)==-1&&p.admission_calls==5&&p.failed==1);
    assert(p.admissions[4].required_capacity==3);
    bn_generate_batch_free(&batch);
}
static void request_record_recycling(void) {
    BnGenerateBatch batch={0};Probe p={0};p.batch=&batch;p.limit=1;
    p.next_admission=UINT64_C(0x100000000);
    BnGenerateBatchLimits limits={1,1,64,256,8192,0};
    BnGenerateBatchBackend backend={&p,NULL,execute,release,time_ms};
    assert(!bn_generate_batch_init_request_aware(&batch,&limits,5,&backend,acquire_stream,notify,&p));
    for(int i=0;i<1024;i++) {
        uint64_t id=UINT64_C(0x100000000)+(uint64_t)i;
        int32_t prompt=(int32_t)(id%5),out=-1;
        assert(!bn_generate_batch_enqueue(&batch,id,&prompt,1,&out,1));
        assert(bn_generate_batch_step(&batch)==1&&out==prompt);
    }
    BnGenerateBatchInfo info;assert(!bn_generate_batch_info(&batch,&info));
    assert(info.submitted==1024&&info.completed==1024&&info.acquired==1024&&info.released==1024);
    assert(!info.live&&!info.queued&&!info.failed&&p.admission_calls==1024&&p.created==1&&p.reused==1023);
    bn_generate_batch_free(&batch);
}
static void prefill_policy_lifetime(void) {
    BnGenerateBatch batch={0};Probe p={0};
    assert(bn_generate_batch_set_prefill_policy(NULL,BN_GENERATE_BATCH_PREFILL_RECLAIM)==-1);
    assert(bn_generate_batch_set_prefill_policy(&batch,BN_GENERATE_BATCH_PREFILL_RECLAIM)==-1);
    init(&batch,&p,4);
    assert(!bn_generate_batch_set_prefill_policy(&batch,BN_GENERATE_BATCH_PREFILL_RECLAIM));
    assert(bn_generate_batch_set_prefill_policy(&batch,(BnGenerateBatchPrefillPolicy)-1)==-1);
    assert(!p.created&&!p.executes&&!p.clock&&!p.tokens&&!p.admitted);
    int32_t prompt[4][129],out[4][2];
    for(int i=0;i<4;i++)for(int j=0;j<129;j++)prompt[i][j]=i;
    for(int i=0;i<4;i++)assert(!bn_generate_batch_enqueue(&batch,(uint64_t)i,prompt[i],i<2?5:129,out[i],2));
    assert(bn_generate_batch_set_prefill_policy(&batch,BN_GENERATE_BATCH_PREFILL_FIXED)==-1);
    while(bn_generate_batch_step(&batch)>0){}
    assert(p.completed==4&&p.tokens==8);
    assert(bn_generate_batch_set_prefill_policy(&batch,BN_GENERATE_BATCH_PREFILL_FIXED)==-1);
    bn_generate_batch_free(&batch);memset(&p,0,sizeof(p));init(&batch,&p,1);
    assert(!bn_generate_batch_enqueue(&batch,0,prompt[0],5,out[0],2));
    assert(!bn_generate_batch_cancel(&batch,0));
    assert(bn_generate_batch_set_prefill_policy(&batch,BN_GENERATE_BATCH_PREFILL_RECLAIM)==-1);
    bn_generate_batch_free(&batch);
    memset(&p,0,sizeof(p));p.batch=&batch;p.limit=2;p.check_policy_reentry=1;
    BnGenerateBatchLimits limits={2,2,64,16,8192,0};
    BnGenerateBatchBackend backend={&p,acquire,execute,release,time_ms};
    assert(!bn_generate_batch_init(&batch,&limits,5,&backend,notify,&p));
    assert(!bn_generate_batch_set_prefill_policy(&batch,BN_GENERATE_BATCH_PREFILL_RECLAIM));
    assert(!bn_generate_batch_enqueue(&batch,0,prompt[0],2,out[0],2));
    assert(!bn_generate_batch_enqueue(&batch,1,prompt[1],65,out[1],2));
    assert(bn_generate_batch_step(&batch)==1);
    assert(p.leases[0].position==2&&p.leases[1].position==14);
    BnGenerateBatchInfo info;assert(!bn_generate_batch_info(&batch,&info));
    assert(info.prefill_tokens==16&&info.output_tokens==1&&out[0][0]==0);
    p.fail_execute=1;assert(bn_generate_batch_step(&batch)==-1);
    assert(!bn_generate_batch_info(&batch,&info)&&info.failed&&info.prefill_tokens==16&&info.output_tokens==1);
    assert(p.discarded==2&&p.failed==2&&out[0][0]==0);
    assert(bn_generate_batch_set_prefill_policy(&batch,BN_GENERATE_BATCH_PREFILL_FIXED)==-1);
    bn_generate_batch_free(&batch);
    puts("Prefill policy configuration, reclaimed execution and atomic failure PASSED");
}
static void external_abort(void) {
    BnGenerateBatch empty={0};assert(bn_generate_batch_abort(NULL)==-1&&bn_generate_batch_abort(&empty)==-1);
    for(int prefix=0;prefix<2;prefix++) {
        BnGenerateBatch batch={0};Probe p={0};init(&batch,&p,2);p.check_abort_reentry=1;
        int32_t prompt[3][2]={{0,0},{1,1},{2,2}},out[3][3];memset(out,0x5a,sizeof(out));
        for(int i=0;i<3;i++)assert(!bn_generate_batch_enqueue(&batch,(uint64_t)i,prompt[i],2,out[i],3));
        if(prefix)assert(bn_generate_batch_step(&batch)==1);
        assert(!bn_generate_batch_abort(&batch));BnGenerateBatchInfo info;
        assert(!bn_generate_batch_info(&batch,&info)&&info.failed&&!info.live&&!info.queued);
        assert(info.steps==(uint64_t)prefix&&info.output_tokens==(uint64_t)(2*prefix));
        assert(p.executes==prefix&&p.failed==3&&p.discarded==2*prefix&&p.released==2*prefix&&p.tokens==2*prefix);
        for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
            if(prefix&&i<2&&!j)assert(out[i][j]==i);
            else for(size_t k=0;k<sizeof(out[i][j]);k++)assert(((unsigned char*)&out[i][j])[k]==0x5a);
        }
        assert(bn_generate_batch_abort(&batch)==-1&&bn_generate_batch_step(&batch)==-1);
        assert(bn_generate_batch_enqueue(&batch,9,prompt[0],2,out[0],3)==-1);
        bn_generate_batch_free(&batch);assert(p.released==2*prefix);
    }
    Probe p={0};init(&empty,&p,1);assert(!bn_generate_batch_abort(&empty)&&!p.failed&&!p.executes);
    bn_generate_batch_free(&empty);
    puts("External abort: queued/live terminal failure, preserved prefixes, fenced discard and callback reentry rejection PASSED");
}
int main(void) {
    external_abort();
    prefill_policy_lifetime();staggered();failure_atomicity(0,0);failure_atomicity(1,0);failure_atomicity(0,1);failure_atomicity(1,1);cancellation_and_exhaustion();rejection_and_queue_bound();recorded_arrivals();
    request_admission();request_record_recycling();puts("Generate batch runtime PASSED");return 0;
}
