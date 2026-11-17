/* Main-owned bounded ingress. The reader never touches model/request state.
 * One serialized worker consumes records; the caller retains the input fd. */
#ifndef BN_MAIN_SERVE_IO_H
#define BN_MAIN_SERVE_IO_H
#include "json.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SERVE_INGRESS_CAPACITY 8
#define SERVE_RECORD_BYTES (1024u * 1024u)
typedef struct {size_t bytes;int oversized;double received_ms;} ServeRecord;
typedef struct {
    size_t queued,peak;uint64_t records,bytes,oversized;
    int ended,failed,waiting_input,waiting_space;
} ServeIngressInfo;
typedef struct ServeIngress {
    const struct ServeIngress *owner;
    pthread_mutex_t mutex;pthread_cond_t changed;pthread_t thread;
    int input,wake[2],started,stopped;size_t limit,head,tail;
    unsigned char *payload,read_buffer[4096];ServeRecord records[SERVE_INGRESS_CAPACITY];
    ServeIngressInfo info;double (*clock)(void*);void *clock_context;
} ServeIngress;

/* Fixed ingress allocation only. Worker copy/parser payload is separate. */
static inline int serve_ingress_plan(size_t limit,size_t *out) {
    if(!out||!limit||limit>BN_JSON_MAX_BYTES||limit>SIZE_MAX/SERVE_INGRESS_CAPACITY-1)return -1;
    size_t bytes=(limit+1)*SERVE_INGRESS_CAPACITY;
    if(bytes>SIZE_MAX-sizeof(ServeIngress))return -1;
    *out=bytes+sizeof(ServeIngress);return 0;
}
static inline void serve_ingress_add(uint64_t *value,uint64_t n) {
    *value=n>UINT64_MAX-*value?UINT64_MAX:*value+n;
}
static inline int serve_ingress_room(ServeIngress *s) {
    pthread_mutex_lock(&s->mutex);
    while(!s->stopped&&s->info.queued==SERVE_INGRESS_CAPACITY) {
        s->info.waiting_space=1;pthread_cond_broadcast(&s->changed);
        pthread_cond_wait(&s->changed,&s->mutex);
    }
    s->info.waiting_space=0;int stopped=s->stopped;
    pthread_mutex_unlock(&s->mutex);return !stopped;
}
static inline void serve_ingress_publish(ServeIngress *s,size_t bytes,int oversized,double at) {
    unsigned char *data=s->payload+s->tail*(s->limit+1);
    if(oversized)bytes=0;
    data[bytes]=0;
    pthread_mutex_lock(&s->mutex);
    s->records[s->tail]=(ServeRecord){bytes,oversized,at};s->tail=(s->tail+1)%SERVE_INGRESS_CAPACITY;
    s->info.queued++;if(s->info.queued>s->info.peak)s->info.peak=s->info.queued;
    serve_ingress_add(&s->info.records,1);if(oversized)serve_ingress_add(&s->info.oversized,1);
    pthread_cond_broadcast(&s->changed);pthread_mutex_unlock(&s->mutex);
}
static inline void *serve_ingress_reader(void *context) {
    ServeIngress *s=context;size_t used=0;int oversized=0,failed=0;double at=0;
    if(!serve_ingress_room(s))goto done;
    for(;;) {
        struct pollfd fds[2]={{s->input,POLLIN,0},{s->wake[0],POLLIN,0}};
        pthread_mutex_lock(&s->mutex);s->info.waiting_input=1;
        pthread_cond_broadcast(&s->changed);pthread_mutex_unlock(&s->mutex);
        int ready;do{ready=poll(fds,2,-1);}while(ready<0&&errno==EINTR);
        pthread_mutex_lock(&s->mutex);s->info.waiting_input=0;int stopped=s->stopped;
        pthread_mutex_unlock(&s->mutex);
        if(stopped||fds[1].revents)break;
        if(ready<0||(fds[0].revents&(POLLERR|POLLNVAL))){failed=1;break;}
        ssize_t n;do{n=read(s->input,s->read_buffer,sizeof(s->read_buffer));}while(n<0&&errno==EINTR);
        if(n<0){failed=1;break;}
        if(!n) {
            if(used||oversized)serve_ingress_publish(s,used,oversized,at);
            break;
        }
        at=s->clock(s->clock_context);if(!isfinite(at)){failed=1;break;}
        pthread_mutex_lock(&s->mutex);serve_ingress_add(&s->info.bytes,(uint64_t)n);pthread_mutex_unlock(&s->mutex);
        for(ssize_t i=0;i<n;i++) {
            unsigned char c=s->read_buffer[i];
            if(c=='\n') {
                serve_ingress_publish(s,used,oversized,at);used=0;oversized=0;
                if(!serve_ingress_room(s))goto done;
            }else if(used<s->limit)s->payload[s->tail*(s->limit+1)+used++]=c;
            else oversized=1;
        }
    }
done:
    pthread_mutex_lock(&s->mutex);s->info.ended=1;s->info.failed=failed;
    s->info.waiting_input=s->info.waiting_space=0;
    pthread_cond_broadcast(&s->changed);pthread_mutex_unlock(&s->mutex);return NULL;
}
/* Caller owns a zeroed handle and exclusive reading of input until free.
 * clock must be thread-safe and in the scheduler's monotonic clock domain. */
static inline int serve_ingress_init(ServeIngress *s,int input,size_t limit,
    double (*clock)(void*),void *clock_context) {
    size_t bytes;
    if(!s||s->owner||input<0||!clock||serve_ingress_plan(limit,&bytes))return -1;
    unsigned char *payload=malloc(bytes-sizeof(*s));if(!payload)return -1;
    if(pthread_mutex_init(&s->mutex,NULL)){free(payload);return -1;}
    if(pthread_cond_init(&s->changed,NULL)){pthread_mutex_destroy(&s->mutex);free(payload);return -1;}
    int wake[2];
    if(pipe(wake))goto fail;
    if(fcntl(wake[0],F_SETFD,FD_CLOEXEC)<0||fcntl(wake[1],F_SETFD,FD_CLOEXEC)<0) {
        close(wake[0]);close(wake[1]);goto fail;
    }
    s->owner=s;s->input=input;s->limit=limit;s->payload=payload;s->wake[0]=wake[0];s->wake[1]=wake[1];
    s->clock=clock;s->clock_context=clock_context;
    if(pthread_create(&s->thread,NULL,serve_ingress_reader,s)) {
        close(wake[0]);close(wake[1]);goto fail;
    }
    s->started=1;return 0;
fail:
    pthread_cond_destroy(&s->changed);pthread_mutex_destroy(&s->mutex);free(payload);memset(s,0,sizeof(*s));return -1;
}
/* 1 copies a record, 0 currently empty, 2 clean EOF/stop, -1 IO/argument
 * failure. Queued records drain before EOF/error. Empty results change no out. */
static inline int serve_ingress_pop(ServeIngress *s,unsigned char *data,size_t capacity,
    ServeRecord *out,int wait) {
    if(!s||s->owner!=s||!data||!out||capacity<=s->limit||(wait!=0&&wait!=1))return -1;
    pthread_mutex_lock(&s->mutex);
    while(wait&&!s->info.queued&&!s->info.ended&&!s->stopped)pthread_cond_wait(&s->changed,&s->mutex);
    int status=0;
    if(s->info.queued) {
        ServeRecord record=s->records[s->head];
        memcpy(data,s->payload+s->head*(s->limit+1),record.bytes+1);*out=record;
        s->head=(s->head+1)%SERVE_INGRESS_CAPACITY;s->info.queued--;
        pthread_cond_broadcast(&s->changed);status=1;
    }else if(s->info.failed)status=-1;
    else if(s->info.ended||s->stopped)status=2;
    pthread_mutex_unlock(&s->mutex);return status;
}
static inline int serve_ingress_info(ServeIngress *s,ServeIngressInfo *out) {
    if(!s||s->owner!=s||!out)return -1;
    pthread_mutex_lock(&s->mutex);*out=s->info;pthread_mutex_unlock(&s->mutex);return 0;
}
static inline int serve_ingress_stop(ServeIngress *s) {
    if(!s||s->owner!=s)return -1;
    pthread_mutex_lock(&s->mutex);int first=!s->stopped;s->stopped=1;
    pthread_cond_broadcast(&s->changed);pthread_mutex_unlock(&s->mutex);
    if(first) {unsigned char byte=1;ssize_t n;do{n=write(s->wake[1],&byte,1);}while(n<0&&errno==EINTR);if(n!=1)return -1;}
    return 0;
}
static inline void serve_ingress_free(ServeIngress *s) {
    if(!s||s->owner!=s)return;
    (void)serve_ingress_stop(s);if(s->started)pthread_join(s->thread,NULL);
    close(s->wake[0]);close(s->wake[1]);pthread_cond_destroy(&s->changed);pthread_mutex_destroy(&s->mutex);
    free(s->payload);memset(s,0,sizeof(*s));
}

typedef enum {SERVE_SUBMIT,SERVE_CANCEL,SERVE_SHUTDOWN} ServeCommandKind;
typedef struct {ServeCommandKind kind;uint64_t id;const char *prompt;int max_outputs;} ServeCommand;
/* The parsed prompt borrows document; encode/copy before freeing it. No
 * scheduling/backend operations, and failed parsing preserves command. */
static inline int serve_command_parse(BnJSON *document,const void *data,size_t bytes,
    size_t limit,int max_outputs,ServeCommand *command) {
    if(!document||document->state||!command||!limit||limit>BN_JSON_MAX_BYTES||max_outputs<1)return -1;
    BnJSONLimits limits={limit,16};
    if(bn_json_init_limits(document,data,bytes,&limits))return -1;
    const BnJSONValue *root=bn_json_root(document),*op=bn_json_get(root,"op");
    ServeCommand next={0};uint64_t outputs;const char *name=bn_json_text(op);
    if(bn_json_type(root)!=BN_JSON_OBJECT||bn_json_type(op)!=BN_JSON_STRING||!name)goto fail;
    if(!strcmp(name,"submit")) {
        next.kind=SERVE_SUBMIT;
        const BnJSONValue *prompt=bn_json_get(root,"prompt"),*n=bn_json_get(root,"max_outputs");
        if(bn_json_u64(bn_json_get(root,"id"),&next.id)||bn_json_type(prompt)!=BN_JSON_STRING||
           !(next.prompt=bn_json_text(prompt))||!*next.prompt)goto fail;
        next.max_outputs=max_outputs;
        if(n) {
            if(bn_json_u64(n,&outputs)||!outputs||outputs>(uint64_t)max_outputs)goto fail;
            next.max_outputs=(int)outputs;
        }
    }else if(!strcmp(name,"cancel")) {
        next.kind=SERVE_CANCEL;if(bn_json_u64(bn_json_get(root,"id"),&next.id))goto fail;
    }else if(!strcmp(name,"shutdown"))next.kind=SERVE_SHUTDOWN;
    else goto fail;
    for(size_t i=0;i<bn_json_size(root);i++) {
        const char *key=NULL;if(!bn_json_member(root,i,&key))goto fail;
        if(!strcmp(key,"op"))continue;
        if(next.kind!=SERVE_SHUTDOWN&&!strcmp(key,"id"))continue;
        if(next.kind==SERVE_SUBMIT&&(!strcmp(key,"prompt")||!strcmp(key,"max_outputs")))continue;
        goto fail;
    }
    *command=next;return 0;
fail:
    bn_json_free(document);return -1;
}
#endif
