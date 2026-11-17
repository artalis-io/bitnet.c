/* Serialized product control loop shared by all model adapters. Model-owned
 * callbacks only encode/preflight into already bounded request storage. */
#ifndef BN_MAIN_SERVE_H
#define BN_MAIN_SERVE_H
#include "main_serve_io.h"
#include "generate_batch.h"
#include <signal.h>
#include <stdio.h>

typedef struct {const int32_t *prompt;int prompt_tokens;int32_t *outputs;} ServePrepared;
typedef int (*ServePrepare)(void *context,int slot,const ServeCommand *command,ServePrepared *out);
typedef struct {int occupied,terminal;uint64_t id,generation;} ServeRequest;
typedef struct ServeRuntime {
    const struct ServeRuntime *owner;ServeIngress ingress;
    BnGenerateBatch *scheduler;ServePrepare prepare;void *context;
    double (*clock)(void*);void *clock_context;int *adapter_failed;
    ServeRequest requests[BN_GENERATE_BATCH_QUEUE_CAPACITY];
    unsigned char *record;size_t record_limit;int max_outputs,started;
    uint64_t generations,accepted,rejected;
} ServeRuntime;

/* Ingress + worker copy + immutable JSON string payload upper bound. The JSON
 * parser separately limits nodes to16. Its opaque bookkeeping, pthread/stdio
 * internals and thread stack remain outside this managed payload quote. */
static inline int serve_runtime_plan(size_t limit,size_t *out) {
    size_t bytes;
    if(!out||serve_ingress_plan(limit,&bytes))return -1;
    size_t extra=sizeof(ServeRuntime)-sizeof(ServeIngress);
    if(limit>SIZE_MAX/2-1||extra>SIZE_MAX-2*(limit+1))return -1;
    extra+=2*(limit+1);if(bytes>SIZE_MAX-extra)return -1;
    *out=bytes+extra;return 0;
}
static inline int serve_runtime_init(ServeRuntime *s,BnGenerateBatch *scheduler,
    ServePrepare prepare,void *context,double (*clock)(void*),void *clock_context,
    int *adapter_failed,size_t limit,int max_outputs) {
    size_t bytes;BnGenerateBatchInfo info;
    if(!s||s->owner||!scheduler||!prepare||!clock||!adapter_failed||max_outputs<1||
       serve_runtime_plan(limit,&bytes)||bn_generate_batch_info(scheduler,&info)||
       info.failed||info.submitted)return -1;
    unsigned char *record=malloc(limit+1);if(!record)return -1;
    s->owner=s;s->scheduler=scheduler;s->prepare=prepare;s->context=context;s->clock=clock;
    s->clock_context=clock_context;s->adapter_failed=adapter_failed;s->record=record;
    s->record_limit=limit;s->max_outputs=max_outputs;return 0;
}
static inline void serve_runtime_reap(ServeRuntime *s) {
    for(int i=0;i<BN_GENERATE_BATCH_QUEUE_CAPACITY;i++)if(s->requests[i].terminal)
        memset(s->requests+i,0,sizeof(s->requests[i]));
}
/* Notification mapping only. Reclamation happens after step/cancel returns,
 * because token notification precedes execution-lease reclamation. */
static inline int serve_runtime_event(ServeRuntime *s,const BnGenerateBatchEvent *e,uint64_t *id) {
    if(!s||s->owner!=s||!e||!id||e->id>=BN_GENERATE_BATCH_QUEUE_CAPACITY||!s->requests[e->id].occupied)return -1;
    ServeRequest *request=s->requests+e->id;*id=request->id;
    if(e->kind==BN_GENERATE_BATCH_FAILED||e->kind==BN_GENERATE_BATCH_CANCELLED||
       (e->kind==BN_GENERATE_BATCH_TOKEN&&e->complete))request->terminal=1;
    return 0;
}
/* NULL preserves static-burst event bytes. Only the worker writes stdout. */
static inline int serve_event_end(ServeRuntime *s,const BnGenerateBatchEvent *e) {
    if(s&&(!e||e->id>=BN_GENERATE_BATCH_QUEUE_CAPACITY||!s->requests[e->id].occupied))return -1;
    if(s&&fprintf(stdout,",\"generation\":%llu,\"submitted_ms\":%.6f,\"admitted_ms\":%.6f,\"timestamp_ms\":%.6f",
        (unsigned long long)s->requests[e->id].generation,e->submitted_ms,e->admitted_ms,e->timestamp_ms)<0)return -1;
    return puts("}")==EOF?-1:0;
}
static inline int serve_trace_name(ServeRuntime *s,uint64_t slot,size_t frame,int logits,char *name,size_t bytes) {
    int n;
    if(s) {
        if(s->owner!=s||slot>=BN_GENERATE_BATCH_QUEUE_CAPACITY||!s->requests[slot].occupied)return -1;
        const ServeRequest *request=s->requests+slot;
        n=snprintf(name,bytes,"request-%llu-generation-%llu-forward-%zu.%s",(unsigned long long)request->id,
            (unsigned long long)request->generation,frame,logits?"f32le":"i32le");
    }else n=snprintf(name,bytes,"request-%llu-forward-%zu.%s",(unsigned long long)slot,frame,logits?"f32le":"i32le");
    return n<0||(size_t)n>=bytes?-1:0;
}
static inline int serve_runtime_reject(ServeRuntime *s,const char *reason,int has_id,uint64_t id) {
    serve_ingress_add(&s->rejected,1);
    if(fprintf(stdout,"{\"kind\":\"rejected\",\"reason\":\"%s\"",reason)<0||
       (has_id&&fprintf(stdout,",\"request\":%llu",(unsigned long long)id)<0)||puts("}")==EOF||fflush(stdout))return -1;
    return 0;
}
static inline int serve_runtime_command(ServeRuntime *s,const ServeRecord *record,int *closing) {
    if(record->oversized)return serve_runtime_reject(s,"record_too_large",0,0);
    BnJSON document={0};ServeCommand command;
    if(serve_command_parse(&document,s->record,record->bytes,s->record_limit,s->max_outputs,&command))
        return serve_runtime_reject(s,"invalid_command",0,0);
    int status=0;
    if(command.kind==SERVE_SHUTDOWN) {
        *closing=1;
        status=serve_ingress_stop(&s->ingress)||puts("{\"kind\":\"draining\"}")==EOF||fflush(stdout)?-1:0;
    }else {
        int existing=-1,free_slot=-1;
        for(int i=0;i<BN_GENERATE_BATCH_QUEUE_CAPACITY;i++) {
            if(s->requests[i].occupied&&s->requests[i].id==command.id)existing=i;
            if(!s->requests[i].occupied&&free_slot<0)free_slot=i;
        }
        if(command.kind==SERVE_CANCEL) {
            if(existing<0)status=serve_runtime_reject(s,"not_pending",1,command.id);
            else {status=bn_generate_batch_cancel(s->scheduler,(uint64_t)existing);serve_runtime_reap(s);}
        }else if(existing>=0)status=serve_runtime_reject(s,"duplicate_id",1,command.id);
        else if(free_slot<0)status=serve_runtime_reject(s,"queue_full",1,command.id);
        else if(s->generations==UINT64_MAX)status=-1;
        else {
            ServePrepared prepared={0};ServeRequest *request=s->requests+free_slot;
            *request=(ServeRequest){1,0,command.id,s->generations+1};
            int prepared_status=s->prepare(s->context,free_slot,&command,&prepared);
            if(prepared_status) {
                memset(request,0,sizeof(*request));
                status=prepared_status<0?-1:serve_runtime_reject(s,"request_limits",1,command.id);
            }else {
                status=bn_generate_batch_enqueue_at(s->scheduler,(uint64_t)free_slot,prepared.prompt,
                    prepared.prompt_tokens,prepared.outputs,command.max_outputs,record->received_ms);
                if(status) {
                    memset(request,0,sizeof(*request));
                    if(status==1)status=serve_runtime_reject(s,"queue_full",1,command.id);
                }else {
                    s->generations++;serve_ingress_add(&s->accepted,1);
                    if(fprintf(stdout,"{\"kind\":\"accepted\",\"request\":%llu,\"generation\":%llu,\"submitted_ms\":%.6f}\n",
                        (unsigned long long)command.id,(unsigned long long)request->generation,record->received_ms)<0||fflush(stdout))status=-1;
                }
            }
        }
    }
    bn_json_free(&document);return status;
}
/* Drain at most64 control records between GPU steps, keeping work fair under
 * input flood. EOF/shutdown drain accepted work; IO/backend failures abort it. */
static inline int serve_runtime_run(ServeRuntime *s,int input) {
    if(!s||s->owner!=s||s->started||input<0)return -1;
    s->started=1;
    if(signal(SIGPIPE,SIG_IGN)==SIG_ERR||serve_ingress_init(&s->ingress,input,s->record_limit,s->clock,s->clock_context))goto fail;
    if(fprintf(stdout,"{\"kind\":\"ready\",\"protocol\":1,\"queue_capacity\":%d,\"record_bytes\":%zu,\"max_outputs\":%d,\"clock\":\"monotonic_ms\"}\n",
        BN_GENERATE_BATCH_QUEUE_CAPACITY,s->record_limit,s->max_outputs)<0||fflush(stdout))goto fail;
    int closing=0;
    for(;;) {
        serve_runtime_reap(s);BnGenerateBatchInfo info;
        if(*s->adapter_failed||bn_generate_batch_info(s->scheduler,&info)||info.failed)goto fail;
        if(!closing)for(int i=0;i<BN_GENERATE_BATCH_QUEUE_CAPACITY;i++) {
            ServeRecord record;int wait=!i&&!info.live&&!info.queued;
            int result=serve_ingress_pop(&s->ingress,s->record,s->record_limit+1,&record,wait);
            if(result<0)goto fail;
            if(result==2){closing=1;break;}
            if(!result)break;
            if(serve_runtime_command(s,&record,&closing)||*s->adapter_failed)goto fail;
            if(closing)break;
        }
        if(bn_generate_batch_info(s->scheduler,&info)||info.failed)goto fail;
        if(info.live||info.queued) {
            if(bn_generate_batch_step(s->scheduler)!=1||*s->adapter_failed)goto fail;
        }else if(closing) {
            if(fprintf(stdout,"{\"kind\":\"stopped\",\"accepted\":%llu,\"completed\":%llu,\"cancelled\":%llu,\"rejected\":%llu}\n",
                (unsigned long long)s->accepted,(unsigned long long)info.completed,
                (unsigned long long)info.cancelled,(unsigned long long)s->rejected)<0||fflush(stdout))goto fail;
            serve_ingress_free(&s->ingress);return 0;
        }
    }
fail:
    serve_ingress_free(&s->ingress);
    BnGenerateBatchInfo info;
    if(!bn_generate_batch_info(s->scheduler,&info)&&!info.failed)(void)bn_generate_batch_abort(s->scheduler);
    return -1;
}
static inline void serve_runtime_free(ServeRuntime *s) {
    if(!s||s->owner!=s)return;
    serve_ingress_free(&s->ingress);free(s->record);memset(s,0,sizeof(*s));
}
#endif
