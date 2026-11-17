#include "generate_batch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int occupied;
    uint64_t id;
    const int32_t *prompt;
    int32_t *outputs;
    BnGenerateBatchCandidate progress;
    BnGenerateBatchLease lease;
    double submitted_ms,admitted_ms;
} BatchRequest;
typedef struct {
    BnGenerateBatchLimits limits;
    BnGenerateBatchBackend backend;
    BnGenerateBatchAcquireRequest acquire_request;
    BnGenerateBatchNotify notify;
    void *notify_context;
    int vocab_size,busy;
    BnGenerateBatchPrefillPolicy prefill_policy;
    BnGenerateBatchInfo info;
    BatchRequest requests[BN_GENERATE_BATCH_QUEUE_CAPACITY];
    int queued[BN_GENERATE_BATCH_QUEUE_CAPACITY],live[BN_GENERATE_BATCH_MAX_REQUESTS];
} BatchState;

static int error(void) {fprintf(stderr,"generate batch: invalid request or runtime state\n");return -1;}
static int overlaps(const void *a,size_t an,const void *b,size_t bn) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y?y-x<an:x-y<bn;
}
static void event(BatchState *s,BatchRequest *r,BnGenerateBatchEventKind kind,
    int token,int index,int complete,double now) {
    if(s->notify) {
        BnGenerateBatchEvent e={kind,r->id,token,index,complete,r->submitted_ms,r->admitted_ms,now};
        s->notify(s->notify_context,&e);
    }
}
static void release(BatchState *s,BatchRequest *r,int discard) {
    s->backend.release(s->backend.context,r->lease,discard);
    memset(&r->lease,0,sizeof(r->lease));s->info.released++;
}
static int fail(BatchState *s) {
    s->info.failed=1;
    for(int i=0;i<s->info.live;i++)release(s,&s->requests[s->live[i]],1);
    s->info.live=0;s->info.queued=0;
    double now=s->backend.time_ms(s->backend.context);
    for(int i=0;i<BN_GENERATE_BATCH_QUEUE_CAPACITY;i++)if(s->requests[i].occupied) {
        event(s,&s->requests[i],BN_GENERATE_BATCH_FAILED,-1,-1,1,now);
        s->requests[i].occupied=0;
    }
    s->busy=0;return -1;
}
static int init(BnGenerateBatch *batch,const BnGenerateBatchLimits *limits,
    int vocab_size,const BnGenerateBatchBackend *backend,BnGenerateBatchAcquireRequest acquire_request,
    BnGenerateBatchNotify notify,void *context) {
    BnGenerateBatchPlan plan;
    if(!batch||batch->state||!backend||(!acquire_request&&!backend->acquire)||!backend->execute||
       !backend->release||!backend->time_ms||vocab_size<1||
       (size_t)vocab_size>SIZE_MAX/sizeof(float)||bn_generate_batch_plan(limits,NULL,0,&plan)<0)return error();
    BatchState *s=calloc(1,sizeof(*s));if(!s)return error();
    s->limits=*limits;s->vocab_size=vocab_size;
    s->backend.context=backend->context;s->backend.execute=backend->execute;
    s->backend.release=backend->release;s->backend.time_ms=backend->time_ms;
    s->backend.acquire=acquire_request?NULL:backend->acquire;
    s->acquire_request=acquire_request;
    s->notify=notify;s->notify_context=context;batch->state=s;return 0;
}
int bn_generate_batch_init(BnGenerateBatch *batch,const BnGenerateBatchLimits *limits,
    int vocab_size,const BnGenerateBatchBackend *backend,BnGenerateBatchNotify notify,void *context) {
    return init(batch,limits,vocab_size,backend,NULL,notify,context);
}
int bn_generate_batch_init_request_aware(BnGenerateBatch *batch,const BnGenerateBatchLimits *limits,
    int vocab_size,const BnGenerateBatchBackend *backend,BnGenerateBatchAcquireRequest acquire_request,
    BnGenerateBatchNotify notify,void *context) {
    if(!acquire_request)return error();
    return init(batch,limits,vocab_size,backend,acquire_request,notify,context);
}
int bn_generate_batch_set_prefill_policy(BnGenerateBatch *batch,
    BnGenerateBatchPrefillPolicy policy) {
    BatchState *s=batch?batch->state:NULL;
    if(!s||s->busy||s->info.failed||s->info.submitted||
       (policy!=BN_GENERATE_BATCH_PREFILL_FIXED&&policy!=BN_GENERATE_BATCH_PREFILL_RECLAIM))return error();
    s->prefill_policy=policy;return 0;
}
static int enqueue(BnGenerateBatch *batch,uint64_t id,
    const int32_t *prompt,int n,int32_t *outputs,int max_outputs,
    int recorded,double submitted_ms) {
    BatchState *s=batch?batch->state:NULL;
    if(!s||s->busy||s->info.failed||!prompt||!outputs||n<1||max_outputs<1||
       (recorded&&!isfinite(submitted_ms))||
       n>s->limits.capacity||max_outputs-1>s->limits.capacity-n||
       (size_t)n>SIZE_MAX/sizeof(*prompt)||(size_t)max_outputs>SIZE_MAX/sizeof(*outputs))return error();
    size_t pn=(size_t)n*sizeof(*prompt),on=(size_t)max_outputs*sizeof(*outputs);
    if((uintptr_t)prompt>UINTPTR_MAX-pn||(uintptr_t)outputs>UINTPTR_MAX-on||overlaps(prompt,pn,outputs,on))return error();
    int slot=-1;
    for(int i=0;i<BN_GENERATE_BATCH_QUEUE_CAPACITY;i++) {
        BatchRequest *r=&s->requests[i];
        if(!r->occupied){if(slot<0)slot=i;continue;}
        if(r->id==id||overlaps(outputs,on,r->outputs,(size_t)r->progress.max_outputs*sizeof(*outputs))||
           overlaps(outputs,on,r->prompt,(size_t)r->progress.prompt_tokens*sizeof(*prompt))||
           overlaps(prompt,pn,r->outputs,(size_t)r->progress.max_outputs*sizeof(*outputs)))return error();
    }
    for(int i=0;i<n;i++)if(prompt[i]<0||prompt[i]>=s->vocab_size)return error();
    if(slot<0)return 1;
    s->busy=1;
    double now=s->backend.time_ms(s->backend.context);
    if(!isfinite(now)||(recorded&&submitted_ms>now)){s->busy=0;return error();}
    BatchRequest r={0};r.occupied=1;r.id=id;r.prompt=prompt;r.outputs=outputs;
    r.progress=(BnGenerateBatchCandidate){n,0,0,max_outputs};
    r.submitted_ms=recorded?submitted_ms:now;r.admitted_ms=-1;
    s->requests[slot]=r;s->queued[s->info.queued++]=slot;s->info.submitted++;s->busy=0;return 0;
}
int bn_generate_batch_enqueue(BnGenerateBatch *batch,uint64_t id,
    const int32_t *prompt,int n,int32_t *outputs,int max_outputs) {
    return enqueue(batch,id,prompt,n,outputs,max_outputs,0,0);
}
int bn_generate_batch_enqueue_at(BnGenerateBatch *batch,uint64_t id,
    const int32_t *prompt,int n,int32_t *outputs,int max_outputs,double submitted_ms) {
    return enqueue(batch,id,prompt,n,outputs,max_outputs,1,submitted_ms);
}
int bn_generate_batch_step(BnGenerateBatch *batch) {
    BatchState *s=batch?batch->state:NULL;
    if(!s||s->busy||s->info.failed)return error();
    s->busy=1;
    while(s->info.queued&&s->info.live<s->limits.max_requests) {
        BatchRequest *pending=&s->requests[s->queued[0]];
        BnGenerateBatchAdmission request={pending->id,pending->progress.prompt_tokens,
            pending->progress.max_outputs,pending->progress.prompt_tokens+(pending->progress.max_outputs-1)};
        BnGenerateBatchLease lease={0};
        int rc=s->acquire_request?s->acquire_request(s->backend.context,&request,&lease):
            s->backend.acquire(s->backend.context,&lease);
        if(rc==1&&!lease.handle&&!lease.logits&&!lease.logits_count)break;
        if(rc||!lease.handle||!lease.logits||lease.logits_count!=(size_t)s->vocab_size) {
            if(lease.handle)s->backend.release(s->backend.context,lease,1);
            return fail(s);
        }
        for(int i=0;i<s->info.live;i++) {
            BnGenerateBatchLease old=s->requests[s->live[i]].lease;
            if(old.handle==lease.handle)return fail(s);
            if(overlaps(old.logits,(size_t)s->vocab_size*sizeof(float),lease.logits,(size_t)s->vocab_size*sizeof(float))) {
                s->backend.release(s->backend.context,lease,1);return fail(s);
            }
        }
        int slot=s->queued[0];memmove(s->queued,s->queued+1,(size_t)(--s->info.queued)*sizeof(*s->queued));
        BatchRequest *r=&s->requests[slot];r->lease=lease;r->admitted_ms=s->backend.time_ms(s->backend.context);
        s->live[s->info.live++]=slot;s->info.acquired++;
        if(s->info.live>s->info.peak_live)s->info.peak_live=s->info.live;
        event(s,r,BN_GENERATE_BATCH_ADMITTED,-1,-1,0,r->admitted_ms);
    }
    BnGenerateBatchCandidate candidates[BN_GENERATE_BATCH_MAX_REQUESTS];
    for(int i=0;i<s->info.live;i++)candidates[i]=s->requests[s->live[i]].progress;
    BnGenerateBatchPlan plan;int planned=bn_generate_batch_plan_with_prefill_policy(
        &s->limits,candidates,s->info.live,s->prefill_policy,&plan);
    if(planned<0)return fail(s);
    if(!planned){s->busy=0;return 0;}
    BnGenerateBatchExecutionRow rows[BN_GENERATE_BATCH_MAX_REQUESTS];
    int selected[BN_GENERATE_BATCH_MAX_REQUESTS];
    for(int i=0;i<plan.count;i++) {
        BnGenerateBatchRow *p=&plan.rows[i];BatchRequest *r=&s->requests[s->live[p->candidate]];
        rows[i]=(BnGenerateBatchExecutionRow){r->id,r->lease,
            p->is_prefill?r->prompt+r->progress.prefilled:r->outputs+r->progress.generated-1,
            p->n_tokens,p->position,p->is_prefill,p->decode_role,p->emit_token};
    }
    if(s->backend.execute(s->backend.context,rows,plan.count))return fail(s);
    for(int i=0;i<plan.count;i++)if(plan.rows[i].emit_token) {
        float best=-INFINITY;int token=-1;
        for(int j=0;j<s->vocab_size;j++) {
            float x=rows[i].lease.logits[j];
            if(isnan(x)||x==INFINITY)return fail(s);
            if(x>best){best=x;token=j;}
        }
        if(token<0||!isfinite(best))return fail(s);
        selected[i]=token;
    }
    double now=s->backend.time_ms(s->backend.context);
    for(int i=0;i<plan.count;i++) {
        BnGenerateBatchRow *p=&plan.rows[i];BatchRequest *r=&s->requests[s->live[p->candidate]];
        if(p->is_prefill){r->progress.prefilled+=p->n_tokens;s->info.prefill_tokens+=(uint64_t)p->n_tokens;}
        else s->info.decode_tokens++;
        if(p->emit_token) {
            r->outputs[r->progress.generated++]=selected[i];s->info.output_tokens++;
            if(r->progress.generated==r->progress.max_outputs)s->info.completed++;
        }
    }
    s->info.steps++;
    for(int i=0;i<plan.count;i++)if(plan.rows[i].emit_token) {
        BatchRequest *r=&s->requests[s->live[plan.rows[i].candidate]];
        event(s,r,BN_GENERATE_BATCH_TOKEN,selected[i],r->progress.generated-1,r->progress.generated==r->progress.max_outputs,now);
    }
    int kept=0;
    for(int i=0;i<s->info.live;i++) {
        int slot=s->live[i];BatchRequest *r=&s->requests[slot];
        if(r->progress.generated==r->progress.max_outputs){release(s,r,0);r->occupied=0;}
        else s->live[kept++]=slot;
    }
    s->info.live=kept;s->busy=0;return 1;
}
int bn_generate_batch_cancel(BnGenerateBatch *batch,uint64_t id) {
    BatchState *s=batch?batch->state:NULL;
    if(!s||s->busy||s->info.failed)return error();
    int slot=-1;
    for(int i=0;i<BN_GENERATE_BATCH_QUEUE_CAPACITY;i++)if(s->requests[i].occupied&&s->requests[i].id==id){slot=i;break;}
    if(slot<0)return error();
    s->busy=1;
    for(int i=0;i<s->info.queued;i++)if(s->queued[i]==slot) {
        memmove(s->queued+i,s->queued+i+1,(size_t)(--s->info.queued-i)*sizeof(*s->queued));break;
    }
    for(int i=0;i<s->info.live;i++)if(s->live[i]==slot) {
        release(s,&s->requests[slot],0);
        memmove(s->live+i,s->live+i+1,(size_t)(--s->info.live-i)*sizeof(*s->live));break;
    }
    s->info.cancelled++;
    event(s,&s->requests[slot],BN_GENERATE_BATCH_CANCELLED,-1,-1,1,s->backend.time_ms(s->backend.context));
    s->requests[slot].occupied=0;s->busy=0;return 0;
}
int bn_generate_batch_abort(BnGenerateBatch *batch) {
    BatchState *s=batch?batch->state:NULL;
    if(!s||s->busy||s->info.failed)return error();
    s->busy=1;(void)fail(s);return 0;
}
int bn_generate_batch_info(const BnGenerateBatch *batch,BnGenerateBatchInfo *out) {
    BatchState *s=batch?batch->state:NULL;
    if(!s||!out)return error();
    *out=s->info;return 0;
}
void bn_generate_batch_free(BnGenerateBatch *batch) {
    BatchState *s=batch?batch->state:NULL;if(!s)return;
    if(s->busy){(void)error();return;}
    s->busy=1;
    for(int i=0;i<s->info.live;i++)release(s,&s->requests[s->live[i]],s->info.failed);
    free(s);batch->state=NULL;
}
