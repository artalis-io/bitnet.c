#include "../src/main_serve.h"
#include <assert.h>
#include <sys/wait.h>
#include <time.h>

typedef struct {int used,allocated,slot,position;float logits[3];} Lease;
typedef struct {int32_t prompt,outputs[4];Lease *lease;} Job;
typedef struct {
    BnGenerateBatch batch;ServeRuntime runtime;Job jobs[64];Lease leases[2];
    int failed,mode,gate,progress,executes,discarded,reused,max_group;
} Model;
static void put(int fd,const void *data,size_t bytes) {
    const unsigned char *p=data;
    while(bytes){ssize_t n=write(fd,p,bytes);if(n<0&&errno==EINTR)continue;assert(n>0);p+=n;bytes-=(size_t)n;}
}
static void byte(int fd) {char c;ssize_t n;do{n=read(fd,&c,1);}while(n<0&&errno==EINTR);assert(n==1);}
static double now(void *context) {
    (void)context;struct timespec t;assert(!clock_gettime(CLOCK_MONOTONIC,&t));return t.tv_sec*1000.0+t.tv_nsec/1e6;
}
static int prepare(void *context,int slot,const ServeCommand *command,ServePrepared *out) {
    Model *m=context;
    if(strlen(command->prompt)!=1||command->prompt[0]<'0'||command->prompt[0]>'2')return 1;
    assert(!m->jobs[slot].lease);m->jobs[slot].prompt=command->prompt[0]-'0';
    *out=(ServePrepared){&m->jobs[slot].prompt,1,m->jobs[slot].outputs};return 0;
}
static int acquire(void *context,const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out) {
    Model *m=context;assert(request->id<64&&!m->jobs[request->id].lease);
    for(int i=0;i<2;i++)if(!m->leases[i].used) {
        Lease *l=m->leases+i;if(l->allocated)m->reused++;
        l->used=l->allocated=1;l->position=0;l->slot=(int)request->id;m->jobs[request->id].lease=l;
        *out=(BnGenerateBatchLease){l,l->logits,3};return 0;
    }
    return 1;
}
static int execute(void *context,const BnGenerateBatchExecutionRow *rows,int count) {
    Model *m=context;m->executes++;if(count>m->max_group)m->max_group=count;
    int block=(m->mode==0||m->mode==1)?m->executes==1:m->mode==2&&m->executes==2;
    if(block) {
        put(m->progress,"X",1);byte(m->gate);
        size_t target=m->mode==0?7:m->mode==1?2:1;
        struct timespec until;assert(!clock_gettime(CLOCK_REALTIME,&until));until.tv_sec+=10;
        ServeIngress *s=&m->runtime.ingress;pthread_mutex_lock(&s->mutex);
        while(s->info.records<target)assert(!pthread_cond_timedwait(&s->changed,&s->mutex,&until));
        assert(s->info.queued>=target-(m->mode==0?2:1));
        pthread_mutex_unlock(&s->mutex);put(m->progress,"R",1);
    }
    for(int i=0;i<count;i++) {
        const BnGenerateBatchExecutionRow *r=rows+i;Lease *l=r->lease.handle;
        assert(l->used&&l->slot==(int)r->id&&r->position==l->position);
        for(int j=0;j<r->n_tokens;j++)assert(r->tokens[j]==m->jobs[r->id].prompt);
        for(int j=0;j<3;j++)l->logits[j]=j==m->jobs[r->id].prompt?2.0f:0.0f;
        l->position+=r->n_tokens;
    }
    return 0;
}
static void release(void *context,BnGenerateBatchLease lease,int discard) {
    Model *m=context;Lease *l=lease.handle;assert(l->used&&m->runtime.requests[l->slot].occupied);
    assert(m->jobs[l->slot].lease==l);m->jobs[l->slot].lease=NULL;l->used=0;m->discarded+=discard;
}
static void notify(void *context,const BnGenerateBatchEvent *event) {
    Model *m=context;if(m->failed)return;uint64_t id;
    assert(!serve_runtime_event(&m->runtime,event,&id));
    const char *kind=event->kind==BN_GENERATE_BATCH_ADMITTED?"admitted":event->kind==BN_GENERATE_BATCH_TOKEN?"token":
        event->kind==BN_GENERATE_BATCH_CANCELLED?"cancelled":"failed";
    if(printf("{\"kind\":\"%s\",\"request\":%llu,\"index\":%d,\"token\":%d",kind,(unsigned long long)id,event->output_index,event->token)<0||
       serve_event_end(&m->runtime,event))m->failed=1;
    if(!m->failed&&event->kind==BN_GENERATE_BATCH_TOKEN&&event->complete) {
        assert(m->jobs[event->id].lease);
        if(printf("{\"kind\":\"completed\",\"request\":%llu",(unsigned long long)id)<0||serve_event_end(&m->runtime,event))m->failed=1;
    }
    if(fflush(stdout))m->failed=1;
}
static void child(int input,int output,int gate,int progress,int mode) {
    alarm(20);assert(dup2(output,STDOUT_FILENO)==STDOUT_FILENO);close(output);
    Model m={0};m.mode=mode;m.gate=gate;m.progress=progress;
    BnGenerateBatchLimits limits={2,2,4,8,128,0};BnGenerateBatchBackend backend={&m,NULL,execute,release,now};
    assert(!bn_generate_batch_init_request_aware(&m.batch,&limits,3,&backend,acquire,notify,&m));
    assert(!serve_runtime_init(&m.runtime,&m.batch,prepare,&m,now,NULL,&m.failed,256,4));
    if(mode==3){put(progress,"X",1);byte(gate);}
    int result=serve_runtime_run(&m.runtime,input);BnGenerateBatchInfo info;assert(!bn_generate_batch_info(&m.batch,&info));
    if(mode==0)assert(!result&&info.submitted==84&&info.completed==82&&info.cancelled==2&&m.runtime.rejected==3&&m.reused>=80&&m.max_group==2&&!m.discarded);
    else if(mode==1)assert(!result&&info.submitted==1&&info.completed==1&&!m.discarded);
    else assert(result==-1&&info.failed&&m.discarded==(mode==2?1:0));
    bn_generate_batch_free(&m.batch);serve_runtime_free(&m.runtime);close(input);close(gate);close(progress);exit(0);
}
typedef struct {pid_t pid;int input,output,gate,progress;} Process;
static Process start(int mode) {
    int input[2],output[2],gate[2],progress[2];assert(!pipe(input)&&!pipe(output)&&!pipe(gate)&&!pipe(progress));
    pid_t pid=fork();assert(pid>=0);
    if(!pid) {
        close(input[1]);close(output[0]);close(gate[1]);close(progress[0]);
        child(input[0],output[1],gate[0],progress[1],mode);
    }
    close(input[0]);close(output[1]);close(gate[0]);close(progress[1]);return (Process){pid,input[1],output[0],gate[1],progress[0]};
}
typedef struct {char kind[24],reason[24];uint64_t id,generation,accepted,completed,cancelled,rejected;int index,token;double submitted,timestamp;} Event;
static Event event(Process *process) {
    char line[512];size_t used=0;
    for(;;){assert(used+1<sizeof(line));ssize_t n=read(process->output,line+used,1);if(n<0&&errno==EINTR)continue;assert(n==1);if(line[used++]=='\n')break;}
    BnJSON d={0};assert(!bn_json_init(&d,line,used));const BnJSONValue *root=bn_json_root(&d);Event e={0};
    const char *s=bn_json_text(bn_json_get(root,"kind"));assert(s&&strlen(s)<sizeof(e.kind));strcpy(e.kind,s);
    s=bn_json_text(bn_json_get(root,"reason"));if(s){assert(strlen(s)<sizeof(e.reason));strcpy(e.reason,s);}
    (void)bn_json_u64(bn_json_get(root,"request"),&e.id);(void)bn_json_u64(bn_json_get(root,"generation"),&e.generation);
    (void)bn_json_u64(bn_json_get(root,"accepted"),&e.accepted);(void)bn_json_u64(bn_json_get(root,"completed"),&e.completed);
    (void)bn_json_u64(bn_json_get(root,"cancelled"),&e.cancelled);(void)bn_json_u64(bn_json_get(root,"rejected"),&e.rejected);
    s=bn_json_text(bn_json_get(root,"index"));if(s)e.index=atoi(s);s=bn_json_text(bn_json_get(root,"token"));if(s)e.token=atoi(s);
    s=bn_json_text(bn_json_get(root,"submitted_ms"));if(s)e.submitted=strtod(s,NULL);
    s=bn_json_text(bn_json_get(root,"timestamp_ms"));if(s)e.timestamp=strtod(s,NULL);
    bn_json_free(&d);return e;
}
static void send(Process *p,const char *s){put(p->input,s,strlen(s));}
static void finish(Process *p) {
    close(p->input);if(p->output>=0)close(p->output);close(p->gate);close(p->progress);
    int status;assert(waitpid(p->pid,&status,0)==p->pid&&WIFEXITED(status)&&!WEXITSTATUS(status));
}
static void arrivals_and_recycling(void) {
    Process p=start(0);Event e=event(&p);assert(!strcmp(e.kind,"ready"));
    send(&p,"{\"op\":\"submit\",\"id\":10,\"prompt\":\"0\"}\n{\"op\":\"submit\",\"id\":10,\"prompt\":\"0\"}\n");byte(p.progress);
    send(&p,"{\"op\":\"submit\",\"id\":20,\"prompt\":\"1\",\"max_outputs\":2}\n{\"op\":\"submit\",\"id\":30,\"prompt\":\"2\",\"max_outputs\":2}\n{\"op\":\"submit\",\"id\":40,\"prompt\":\"2\",\"max_outputs\":2}\n{\"op\":\"cancel\",\"id\":30}\n{\"op\":\"cancel\",\"id\":10}\n");put(p.gate,"g",1);byte(p.progress);
    int completed=0,cancelled=0,prefix=0,duplicates=0;double first=0,arrival=0;
    while(completed<2) {
        e=event(&p);
        if(!strcmp(e.kind,"token")) {
            assert(e.timestamp>=e.submitted);
            if(e.id==10){assert(!e.index&&!e.token);prefix++;first=e.timestamp;}
            else assert((e.id==20&&e.token==1)||(e.id==40&&e.token==2));
        }else if(!strcmp(e.kind,"accepted")&&e.id==20)arrival=e.submitted;
        else if(!strcmp(e.kind,"cancelled")){assert(e.id==10||e.id==30);cancelled++;}
        else if(!strcmp(e.kind,"completed")){assert(e.id==20||e.id==40);completed++;}
        else if(!strcmp(e.kind,"rejected")){assert(!strcmp(e.reason,"duplicate_id"));duplicates++;}
    }
    assert(prefix==1&&cancelled==2&&duplicates==1&&arrival>0&&arrival<=first);
    for(int i=0;i<80;i++) {
        uint64_t id=!i?10:i==79?UINT64_MAX:(uint64_t)1000+i;char line[128];
        snprintf(line,sizeof(line),"{\"op\":\"submit\",\"id\":%llu,\"prompt\":\"2\",\"max_outputs\":1}\n",(unsigned long long)id);send(&p,line);
        do{e=event(&p);assert(e.id==id);if(!strcmp(e.kind,"accepted"))assert(e.generation==(uint64_t)i+5);
           if(!strcmp(e.kind,"token"))assert(!e.index&&e.token==2);}while(strcmp(e.kind,"completed"));
    }
    send(&p,"{\"op\":\"submit\",\"id\":90,\"prompt\":\"bad\"}\n{\"op\":\"cancel\",\"id\":999999}\n{\"op\":\"shutdown\"}\n");
    e=event(&p);assert(!strcmp(e.kind,"rejected")&&!strcmp(e.reason,"request_limits"));
    e=event(&p);assert(!strcmp(e.kind,"rejected")&&!strcmp(e.reason,"not_pending"));
    e=event(&p);assert(!strcmp(e.kind,"draining"));e=event(&p);
    assert(!strcmp(e.kind,"stopped")&&e.accepted==84&&e.completed==82&&e.cancelled==2&&e.rejected==3);finish(&p);
}
static void drain(void) {
    Process p=start(1);Event e=event(&p);assert(!strcmp(e.kind,"ready"));
    send(&p,"{\"op\":\"submit\",\"id\":1,\"prompt\":\"1\"}\n");byte(p.progress);
    send(&p,"{\"op\":\"shutdown\"}\n");put(p.gate,"g",1);byte(p.progress);
    int tokens=0,draining=0;
    do{e=event(&p);if(!strcmp(e.kind,"token")){assert(e.token==1);tokens++;}
       if(!strcmp(e.kind,"draining"))draining++;}while(strcmp(e.kind,"stopped"));
    assert(tokens==4&&draining==1&&e.accepted==1&&e.completed==1);finish(&p);
}
static void output_failure(void) {
    Process p=start(2);Event e=event(&p);assert(!strcmp(e.kind,"ready"));
    send(&p,"{\"op\":\"submit\",\"id\":1,\"prompt\":\"0\"}\n");
    do{e=event(&p);}while(strcmp(e.kind,"token"));assert(!e.index);byte(p.progress);
    close(p.output);p.output=-1;put(p.gate,"g",1);byte(p.progress);finish(&p);
    p=start(3);byte(p.progress);close(p.output);p.output=-1;put(p.gate,"g",1);finish(&p);
}
int main(void) {
    alarm(30);assert(signal(SIGPIPE,SIG_IGN)!=SIG_ERR);
    arrivals_and_recycling();drain();output_failure();alarm(0);
    puts("Serving control: arrivals during execution, queued/live cancellation,84 accepted requests, recycling, shutdown and broken-output abort PASSED");return 0;
}
