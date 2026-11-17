/* Streaming request queue. GPU ownership stays in backend sessions/leases. */
typedef struct NativeScheduleState NativeScheduleState;
typedef struct {
    NativeScheduleState *scheduler;
    int occupied,planned;
    uint64_t id;
    const unsigned char *text;size_t bytes;
    BnGenerateNativeTextOptions options;
    BnGenerateNativeTextCallbacks callbacks;
    int predict;
    double submitted,admitted,first_token,first_byte;
    BnGenerateNativeTextCharge quote;
    BnGenerateNativeTextLease lease;
    BnGenerateNativeTextRequest request;
} NativeScheduleRequest;
struct NativeScheduleState {
    const BnGenerateNativeTextSchedule *owner;
    const BnModel *model;
    BnGenerateNativeTextScheduleLimits limits;
    BnGenerateNativeTextScheduleBackend backend;
    bn_native_text_event_callback notify;void *context;
    BnGenerateNativeTextScheduleInfo info;
    NativeScheduleRequest requests[BN_NATIVE_TEXT_SCHEDULE_QUEUE];
    int queue[BN_NATIVE_TEXT_SCHEDULE_QUEUE],live[BN_NATIVE_TEXT_SCHEDULE_LIVE];
    int cursor,busy,clock_fault;
    double last_ms;
};
static NativeScheduleState *native_schedule_get(const BnGenerateNativeTextSchedule *schedule) {
    NativeScheduleState *s=schedule?schedule->state:NULL;
    return s&&s->owner==schedule?s:NULL;
}
static int native_schedule_error(void) {
    fprintf(stderr,"native text schedule: invalid request or runtime state\n");return -1;
}
static int native_schedule_now(NativeScheduleState *s,double *out) {
    double t=s->backend.time_ms(s->backend.context);
    if(!isfinite(t)||t<0||t<s->last_ms){s->clock_fault=1;return -1;}
    s->last_ms=t;*out=t;return 0;
}
static void native_schedule_event(NativeScheduleState *s,NativeScheduleRequest *r,
    BnGenerateNativeTextEventKind kind,int generated,double now) {
    if(s->notify) {
        BnGenerateNativeTextEvent e={kind,r->id,generated,r->submitted,r->admitted,
            r->first_token,r->first_byte,now};s->notify(&e,s->context);
    }
}
static void native_schedule_remove(int *items,int *count,int position) {
    memmove(items+position,items+position+1,(size_t)(*count-position-1)*sizeof(*items));(*count)--;
}
static void native_schedule_terminal(NativeScheduleState *s,int slot,
    BnGenerateNativeTextEventKind kind,double now,int notify) {
    NativeScheduleRequest *r=s->requests+slot;int generated=r->request.generated;
    if(r->occupied==2) {
        bn_generate_native_text_request_free(&r->request);
        s->backend.release(s->backend.context,r->lease,kind!=BN_NATIVE_TEXT_COMPLETED);
        s->info.released++;
        s->info.device_bytes-=r->quote.device_bytes;s->info.host_bytes-=r->quote.host_bytes;
        s->info.pinned_bytes-=r->quote.pinned_bytes;
        int pos=0;while(s->live[pos]!=slot)pos++;
        native_schedule_remove(s->live,&s->info.live,pos);
        if(s->cursor>pos)s->cursor--;
        if(s->cursor>=s->info.live)s->cursor=0;
    }else {
        int pos=0;while(s->queue[pos]!=slot)pos++;
        native_schedule_remove(s->queue,&s->info.queued,pos);
    }
    r->occupied=0;
    if(notify&&!s->info.poisoned&&native_schedule_now(s,&now))kind=BN_NATIVE_TEXT_REQUEST_FAILED;
    if(kind==BN_NATIVE_TEXT_COMPLETED)s->info.completed++;
    else if(kind==BN_NATIVE_TEXT_REQUEST_CANCELLED)s->info.cancelled++;
    else s->info.failed++;
    if(notify)native_schedule_event(s,r,kind,generated,now);
    *r=(NativeScheduleRequest){0};
}
static int native_schedule_fail(NativeScheduleState *s) {
    s->info.poisoned=1;
    while(s->info.live)native_schedule_terminal(s,s->live[0],BN_NATIVE_TEXT_REQUEST_FAILED,s->last_ms,1);
    while(s->info.queued)native_schedule_terminal(s,s->queue[0],BN_NATIVE_TEXT_REQUEST_FAILED,s->last_ms,1);
    s->busy=0;return -1;
}
static int native_schedule_token(const unsigned char *piece,size_t bytes,int id,void *ptr) {
    NativeScheduleRequest *r=ptr;NativeScheduleState *s=r->scheduler;double now=s->last_ms;
    (void)native_schedule_now(s,&now);
    if(id>=0){s->info.output_tokens++;if(r->first_token<0)r->first_token=now;}
    if(bytes&&r->first_byte<0)r->first_byte=now;
    int stop=r->callbacks.token?r->callbacks.token(piece,bytes,id,r->callbacks.user_data):0;
    return stop||s->clock_fault;
}
static void native_schedule_forward(const int *tokens,int count,int position,
    const float *logits,int vocab,void *ptr) {
    NativeScheduleRequest *r=ptr;r->scheduler->info.forwards++;
    if(r->callbacks.forward)r->callbacks.forward(tokens,count,position,logits,vocab,r->callbacks.user_data);
}
int bn_generate_native_text_schedule_init(BnGenerateNativeTextSchedule *schedule,const BnModel *model,
    const BnGenerateNativeTextScheduleLimits *limits,const BnGenerateNativeTextScheduleBackend *backend,
    bn_native_text_event_callback notify,void *context) {
    if(!schedule||schedule->state||!model||!limits||limits->max_live<1||
       limits->max_live>BN_NATIVE_TEXT_SCHEDULE_LIVE||!backend||!backend->plan||!backend->acquire||
       !backend->release||!backend->time_ms)return native_schedule_error();
    NativeScheduleState *s=calloc(1,sizeof(*s));if(!s)return native_schedule_error();
    s->owner=schedule;s->model=model;s->limits=*limits;s->backend=*backend;
    s->notify=notify;s->context=context;schedule->state=s;return 0;
}
static int native_schedule_enqueue(BnGenerateNativeTextSchedule *schedule,uint64_t id,
    const unsigned char *text,size_t bytes,const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks,int predict,int recorded,double submitted) {
    NativeScheduleState *s=native_schedule_get(schedule);
    if(!s||s->busy||s->info.poisoned||!text||!bytes||!options||
       (predict!=0&&predict!=1)||options->prefill_tokens<1||options->max_new_tokens<0||
       (options->skip_special!=0&&options->skip_special!=1)||
       (options->admit_decode!=0&&options->admit_decode!=1)||
       (recorded&&(!isfinite(submitted)||submitted<0))||
       !native_text_disjoint(text,bytes,s,sizeof(*s))||
       !native_text_disjoint(text,bytes,schedule,sizeof(*schedule)))return native_schedule_error();
    int slot=-1;
    for(int i=0;i<BN_NATIVE_TEXT_SCHEDULE_QUEUE;i++) {
        NativeScheduleRequest *r=s->requests+i;
        if(!r->occupied){if(slot<0)slot=i;continue;}
        if(r->id==id)return native_schedule_error();
        if(r->occupied==2&&(!native_text_disjoint(text,bytes,r->lease.workspace.tokens,
           r->lease.workspace.token_capacity*sizeof(int))||!native_text_disjoint(text,bytes,
           r->lease.workspace.piece,r->lease.workspace.piece_capacity)))return native_schedule_error();
    }
    if(slot<0)return 1;
    s->busy=1;double now;
    if(native_schedule_now(s,&now))return native_schedule_fail(s);
    if(recorded&&submitted>now){s->busy=0;return native_schedule_error();}
    NativeScheduleRequest *r=s->requests+slot;
    *r=(NativeScheduleRequest){.scheduler=s,.occupied=1,.id=id,.text=text,.bytes=bytes,
        .options=*options,.predict=predict,.submitted=recorded?submitted:now,
        .admitted=-1,.first_token=-1,.first_byte=-1};
    if(callbacks)r->callbacks=*callbacks;
    s->queue[s->info.queued++]=slot;s->info.submitted++;s->busy=0;return 0;
}
int bn_generate_native_text_schedule_enqueue(BnGenerateNativeTextSchedule *schedule,uint64_t id,
    const unsigned char *text,size_t bytes,const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks,int predict) {
    return native_schedule_enqueue(schedule,id,text,bytes,options,callbacks,predict,0,0);
}
int bn_generate_native_text_schedule_enqueue_at(BnGenerateNativeTextSchedule *schedule,uint64_t id,
    const unsigned char *text,size_t bytes,const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks,int predict,double submitted_ms) {
    return native_schedule_enqueue(schedule,id,text,bytes,options,callbacks,predict,1,submitted_ms);
}
static int native_schedule_quote_zero(const BnGenerateNativeTextCharge *q) {
    return !q->max_tokens&&!q->capacity&&!q->device_bytes&&!q->host_bytes&&!q->pinned_bytes;
}
static int native_schedule_lease_zero(const BnGenerateNativeTextLease *l) {
    const BnGenerateNativeTextWorkspace *w=&l->workspace;
    return !l->session&&!w->encoder&&!w->decoder&&!w->scratch&&!w->tokens&&!w->token_capacity&&!w->piece&&!w->piece_capacity;
}
static int native_schedule_duplicate(NativeScheduleState *s,const BnGenerateNativeTextLease *l) {
    for(int i=0;i<s->info.live;i++)if(s->requests[s->live[i]].lease.session==l->session)return 1;
    return 0;
}
static int native_schedule_workspace(NativeScheduleState *s,const BnGenerateNativeTextLease *l) {
    const BnGenerateNativeTextWorkspace *w=&l->workspace;
    if(!l->session||!w->tokens||!w->token_capacity||w->token_capacity>SIZE_MAX/sizeof(int)||
       !w->piece||w->piece_capacity<6||!w->scratch||!w->encoder||!w->decoder)return -1;
    size_t n=w->token_capacity*sizeof(int);
    if(!native_text_disjoint(w->tokens,n,s,sizeof(*s))||
       !native_text_disjoint(w->piece,w->piece_capacity,s,sizeof(*s)))return -1;
    for(int i=0;i<BN_NATIVE_TEXT_SCHEDULE_QUEUE;i++) {
        NativeScheduleRequest *r=s->requests+i;if(!r->occupied)continue;
        if(!native_text_disjoint(w->tokens,n,r->text,r->bytes)||
           !native_text_disjoint(w->piece,w->piece_capacity,r->text,r->bytes))return -1;
        if(r->occupied!=2)continue;
        const BnGenerateNativeTextWorkspace *v=&r->lease.workspace;
        if(w->scratch==v->scratch||w->scratch->state==v->scratch->state||
           !native_text_disjoint(w->tokens,n,v->tokens,v->token_capacity*sizeof(int))||
           !native_text_disjoint(w->tokens,n,v->piece,v->piece_capacity)||
           !native_text_disjoint(w->piece,w->piece_capacity,v->tokens,v->token_capacity*sizeof(int))||
           !native_text_disjoint(w->piece,w->piece_capacity,v->piece,v->piece_capacity))return -1;
    }
    return 0;
}
int bn_generate_native_text_schedule_step(BnGenerateNativeTextSchedule *schedule,BnGenerateNativeTextScheduleStep *receipt) {
    NativeScheduleState *s=native_schedule_get(schedule);
    if(!s||s->busy||s->info.poisoned||(receipt&&(!native_text_disjoint(receipt,sizeof(*receipt),s,sizeof(*s))||
       !native_text_disjoint(receipt,sizeof(*receipt),schedule,sizeof(*schedule)))))return native_schedule_error();
    s->busy=1;if(receipt)*receipt=(BnGenerateNativeTextScheduleStep){0};
    double now;if(native_schedule_now(s,&now))return native_schedule_fail(s);
    int work=0;
    while(s->info.queued&&s->info.live<s->limits.max_live) {
        if(s->clock_fault)return native_schedule_fail(s);
        int slot=s->queue[0];NativeScheduleRequest *r=s->requests+slot;
        if(!r->planned) {
            BnGenerateNativeTextCharge q={0};
            int rc=s->backend.plan(s->backend.context,r->id,r->bytes,&r->options,r->predict,&q);
            if(rc<-1||rc>1||(rc&&!native_schedule_quote_zero(&q)))return native_schedule_fail(s);
            if(rc==1)break;
            if(rc||q.max_tokens<1||q.capacity<q.max_tokens||q.device_bytes>s->limits.device_budget||
               q.host_bytes>s->limits.host_budget||q.pinned_bytes>s->limits.pinned_budget) {
                native_schedule_terminal(s,slot,BN_NATIVE_TEXT_REQUEST_FAILED,now,1);work=1;continue;
            }
            r->quote=q;r->planned=1;
        }
        BnGenerateNativeTextCharge *q=&r->quote;
        if(q->device_bytes>s->limits.device_budget-s->info.device_bytes||
           q->host_bytes>s->limits.host_budget-s->info.host_bytes||
           q->pinned_bytes>s->limits.pinned_budget-s->info.pinned_bytes)break;
        BnGenerateNativeTextLease lease={0};int rc=s->backend.acquire(s->backend.context,r->id,q,&lease);
        int duplicate=lease.session&&native_schedule_duplicate(s,&lease);
        if(rc<-1||rc>1||(rc&&!native_schedule_lease_zero(&lease))||(!rc&&(duplicate||native_schedule_workspace(s,&lease)))) {
            if(lease.session&&!duplicate)s->backend.release(s->backend.context,lease,1);
            return native_schedule_fail(s);
        }
        if(rc==1)break;
        if(rc){native_schedule_terminal(s,slot,BN_NATIVE_TEXT_REQUEST_FAILED,now,1);work=1;continue;}
        r->lease=lease;r->occupied=2;s->info.acquired++;
        native_schedule_remove(s->queue,&s->info.queued,0);s->live[s->info.live++]=slot;
        s->info.device_bytes+=q->device_bytes;s->info.host_bytes+=q->host_bytes;s->info.pinned_bytes+=q->pinned_bytes;
        if(s->info.live>s->info.peak_live)s->info.peak_live=s->info.live;
        if(s->info.device_bytes>s->info.peak_device_bytes)s->info.peak_device_bytes=s->info.device_bytes;
        if(s->info.host_bytes>s->info.peak_host_bytes)s->info.peak_host_bytes=s->info.host_bytes;
        if(s->info.pinned_bytes>s->info.peak_pinned_bytes)s->info.peak_pinned_bytes=s->info.pinned_bytes;
        BnGenerateNativeTextCallbacks cb={native_schedule_token,native_schedule_forward,r};
        int init=bn_generate_native_text_request_init(&r->request,s->model,lease.session,&r->lease.workspace,
            r->text,r->bytes,&r->options,&cb,r->predict);
        size_t commits=(size_t)(r->options.max_new_tokens-(r->predict&&r->options.max_new_tokens>0));
        size_t prompt=r->request.prompt_tokens;
        if(native_schedule_now(s,&now))return native_schedule_fail(s);
        if(init||prompt+commits>(size_t)q->capacity||
           (prompt<(size_t)r->options.prefill_tokens?prompt:(size_t)r->options.prefill_tokens)>(size_t)q->max_tokens) {
            native_schedule_terminal(s,slot,BN_NATIVE_TEXT_REQUEST_FAILED,now,1);work=1;continue;
        }
        r->admitted=now;s->info.admitted++;native_schedule_event(s,r,BN_NATIVE_TEXT_ADMITTED,0,now);work=1;
    }
    if(s->clock_fault)return native_schedule_fail(s);
    if(!s->info.live){s->busy=0;return work;}
    int slot=s->live[s->cursor];s->cursor=(s->cursor+1)%s->info.live;
    NativeScheduleRequest *r=s->requests+slot;uint64_t id=r->id;
    BnGenerateNativeTextStatus status=bn_generate_native_text_request_step(&r->request);
    if(receipt)*receipt=(BnGenerateNativeTextScheduleStep){1,id,status};
    s->info.turns++;if(status==BN_NATIVE_TEXT_BUSY)s->info.busy_returns++;
    if(s->clock_fault||native_schedule_now(s,&now))return native_schedule_fail(s);
    if(status==BN_NATIVE_TEXT_DONE)native_schedule_terminal(s,slot,BN_NATIVE_TEXT_COMPLETED,now,1);
    else if(status!=BN_NATIVE_TEXT_PROGRESS&&status!=BN_NATIVE_TEXT_BUSY)
        native_schedule_terminal(s,slot,BN_NATIVE_TEXT_REQUEST_FAILED,now,1);
    if(s->clock_fault)return native_schedule_fail(s);
    s->busy=0;return 1;
}
int bn_generate_native_text_schedule_cancel(BnGenerateNativeTextSchedule *schedule,uint64_t id) {
    NativeScheduleState *s=native_schedule_get(schedule);
    if(!s||s->busy||s->info.poisoned)return native_schedule_error();
    int slot=-1;for(int i=0;i<BN_NATIVE_TEXT_SCHEDULE_QUEUE;i++)
        if(s->requests[i].occupied&&s->requests[i].id==id){slot=i;break;}
    if(slot<0)return native_schedule_error();
    s->busy=1;double now;if(native_schedule_now(s,&now))return native_schedule_fail(s);
    native_schedule_terminal(s,slot,BN_NATIVE_TEXT_REQUEST_CANCELLED,now,1);
    if(s->clock_fault)return native_schedule_fail(s);
    s->busy=0;return 0;
}
int bn_generate_native_text_schedule_info(const BnGenerateNativeTextSchedule *schedule,BnGenerateNativeTextScheduleInfo *out) {
    NativeScheduleState *s=native_schedule_get(schedule);if(!s||!out)return native_schedule_error();
    *out=s->info;return 0;
}
void bn_generate_native_text_schedule_free(BnGenerateNativeTextSchedule *schedule) {
    NativeScheduleState *s=native_schedule_get(schedule);if(!s||s->busy)return;
    s->busy=1;
    while(s->info.live)native_schedule_terminal(s,s->live[0],BN_NATIVE_TEXT_REQUEST_CANCELLED,s->last_ms,0);
    while(s->info.queued)native_schedule_terminal(s,s->queue[0],BN_NATIVE_TEXT_REQUEST_CANCELLED,s->last_ms,0);
    free(s);schedule->state=NULL;
}
