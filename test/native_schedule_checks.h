/* Scheduler ownership/flow tests use the same independent byte-prefix gold. */
typedef struct {
    const unsigned char *text;size_t bytes;
    const int *prompt;int prompt_count,predict,stop,fail_forward;
    BnGenerateNativeTextOptions options;
    Control *control;
    int admitted,completed,cancelled,failed;
    unsigned char output[128];size_t output_bytes;
    BnGenerateNativeTextEvent terminal;
} ScheduleJob;
typedef struct {
    BnGenerateNativeTextSchedule schedule;
    Control controls[2];int used[2],active,peak,acquires,releases,discards,pending_discards;
    int busy_plan,busy_acquire,fail_plan,fail_acquire,duplicate,check_reentry,bad_clock;
    int plan_calls,events;
    size_t device,host,pinned;
    double clock;
    ScheduleJob jobs[128];
    BnGenerateNativeTextWorkspace workspace[2];
    BnTokenizerEncodeScratch scratch[2];int tokens[2][64];unsigned char pieces[2][128];
} SchedulePool;
static void schedule_check_reentry(SchedulePool *p) {
    BnGenerateNativeTextScheduleInfo info;assert(!bn_generate_native_text_schedule_info(&p->schedule,&info));
    BnGenerateNativeTextScheduleStep receipt={0};
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==-1);
    assert(bn_generate_native_text_schedule_cancel(&p->schedule,0)==-1);
    bn_generate_native_text_schedule_free(&p->schedule);assert(p->schedule.state);
}
static int schedule_plan(void *ptr,uint64_t id,size_t bytes,const BnGenerateNativeTextOptions *options,
    int predict,BnGenerateNativeTextCharge *out) {
    SchedulePool *p=ptr;ScheduleJob *j=p->jobs+id;assert(id<128&&j->bytes==bytes&&j->predict==predict);
    assert(options->prefill_tokens==j->options.prefill_tokens);p->plan_calls++;
    if(p->busy_plan)return 1;
    if(p->fail_plan)return -1;
    *out=(BnGenerateNativeTextCharge){64,128,p->device,p->host,p->pinned};return 0;
}
static int schedule_acquire(void *ptr,uint64_t id,const BnGenerateNativeTextCharge *quote,
    BnGenerateNativeTextLease *out) {
    SchedulePool *p=ptr;assert(id<128&&quote->capacity==128);ScheduleJob *j=p->jobs+id;
    if(p->busy_acquire)return 1;
    if(p->fail_acquire)return -1;
    if(p->duplicate&&p->active){*out=(BnGenerateNativeTextLease){&p->controls[0].session,p->workspace[0]};return 0;}
    int i=0;while(i<2&&p->used[i])i++;assert(i<2);
    Control *c=p->controls+i;control_init(c,j->prompt,j->prompt_count);p->controls[0].peer=p->controls+1;
    c->predict=j->predict;c->busy_each=2;c->stop_at=j->stop;c->fail_forward=j->fail_forward;
    j->control=c;p->used[i]=1;p->active++;p->acquires++;if(p->active>p->peak)p->peak=p->active;
    *out=(BnGenerateNativeTextLease){&c->session,p->workspace[i]};return 0;
}
static void schedule_release(void *ptr,BnGenerateNativeTextLease lease,int discard) {
    SchedulePool *p=ptr;int i=lease.session==&p->controls[0].session?0:1;
    assert(lease.session==&p->controls[i].session&&p->used[i]);
    BnGenerateNativeTextScheduleInfo info;assert(!bn_generate_native_text_schedule_info(&p->schedule,&info));
    assert(info.live==p->active&&info.device_bytes==(size_t)p->active*p->device);
    if(p->check_reentry){p->check_reentry=0;schedule_check_reentry(p);}
    p->pending_discards+=discard&&p->controls[i].pending>0;
    p->controls[i].pending=0;p->controls[i].session.pos=0;
    p->used[i]=0;p->active--;p->releases++;p->discards+=discard;
}
static double schedule_clock(void *ptr) {SchedulePool *p=ptr;return p->bad_clock?NAN:++p->clock;}
static int schedule_token(const unsigned char *piece,size_t bytes,int id,void *ptr) {
    ScheduleJob *j=ptr;assert(bytes<=sizeof(j->output)-j->output_bytes);
    if(bytes)memcpy(j->output+j->output_bytes,piece,bytes);
    j->output_bytes+=bytes;
    return control_token(piece,bytes,id,j->control);
}
static void schedule_observe(const int *tokens,int n,int position,const float *logits,int vocab,void *ptr) {
    ScheduleJob *j=ptr;control_observe(tokens,n,position,logits,vocab,j->control);
}
static void schedule_event(const BnGenerateNativeTextEvent *event,void *ptr) {
    SchedulePool *p=ptr;assert(event->id<128);ScheduleJob *j=p->jobs+event->id;
    BnGenerateNativeTextScheduleInfo info;assert(!bn_generate_native_text_schedule_info(&p->schedule,&info));
    assert(event->timestamp_ms>=event->submitted_ms);p->events++;
    if(event->kind==BN_NATIVE_TEXT_ADMITTED){j->admitted++;assert(event->admitted_ms>=event->submitted_ms);return;}
    assert(info.live==p->active&&info.device_bytes==(size_t)p->active*p->device);
    j->terminal=*event;
    if(event->kind==BN_NATIVE_TEXT_COMPLETED) {
        j->completed++;assert(event->generated==(j->stop?j->stop:j->options.max_new_tokens));
        int found=0;
        for(size_t k=0;k<sizeof(native_text_control_cases)/sizeof(native_text_control_cases[0]);k++) {
            if(native_text_control_cases[k].skip==j->options.skip_special&&native_text_control_cases[k].limit==event->generated) {
                assert(j->output_bytes==native_text_control_cases[k].count&&
                    !memcmp(j->output,native_text_control_cases[k].bytes,j->output_bytes));found=1;
            }
        }
        assert(found);
        if(event->generated>=3)assert(event->first_byte_ms>event->first_token_ms&&event->first_token_ms>=event->admitted_ms);
        if(!event->generated)assert(event->first_byte_ms==-1&&event->first_token_ms==-1);
    }else if(event->kind==BN_NATIVE_TEXT_REQUEST_CANCELLED)j->cancelled++;
    else {assert(event->kind==BN_NATIVE_TEXT_REQUEST_FAILED);j->failed++;}
}
static SchedulePool *schedule_pool(const BnTokenizerEncoder *encoder,const BnTokenizerDecoder *decoder,
    int max_live,size_t device,size_t host,size_t pinned) {
    SchedulePool *p=calloc(1,sizeof(*p));assert(p);p->device=1000;p->host=100;p->pinned=10;
    static const unsigned char text0[]="abc123def456",text1[]={0,'a',0};
    for(int id=0;id<128;id++) {
        ScheduleJob *j=p->jobs+id;j->text=id&1?text1:text0;j->bytes=id&1?sizeof(text1):sizeof(text0)-1;
        j->predict=id&1;j->options=(BnGenerateNativeTextOptions){id&1?1:3,6,1,id&1};
        for(size_t k=0;k<sizeof(native_encode_cases)/sizeof(native_encode_cases[0]);k++)
            if(native_encode_cases[k].bytes==j->bytes&&!memcmp(native_encode_cases[k].text,j->text,j->bytes)) {
                j->prompt=native_encode_cases[k].ids;j->prompt_count=(int)native_encode_cases[k].count;break;
            }
        assert(j->prompt&&j->prompt_count>0);
    }
    size_t bytes;assert(!bn_tokenizer_encode_scratch_plan(128,&bytes));
    for(int i=0;i<2;i++) {
        assert(!bn_tokenizer_encode_scratch_init(p->scratch+i,128,bytes));
        p->workspace[i]=(BnGenerateNativeTextWorkspace){encoder,decoder,p->scratch+i,p->tokens[i],64,p->pieces[i],128};
        control_init(p->controls+i,p->jobs[i].prompt,p->jobs[i].prompt_count);
    }
    p->controls[0].peer=p->controls+1;
    BnGenerateNativeTextScheduleBackend backend={p,schedule_plan,schedule_acquire,schedule_release,schedule_clock};
    BnGenerateNativeTextScheduleLimits limits={max_live,device,host,pinned};
    assert(!bn_generate_native_text_schedule_init(&p->schedule,&p->controls[0].model,&limits,&backend,schedule_event,p));
    assert(!p->events&&!p->acquires&&!p->plan_calls&&!p->clock);return p;
}
static int schedule_submit(SchedulePool *p,uint64_t id,double submitted) {
    ScheduleJob *j=p->jobs+id;BnGenerateNativeTextCallbacks cb={schedule_token,schedule_observe,j};
    return bn_generate_native_text_schedule_enqueue_at(&p->schedule,id,j->text,j->bytes,&j->options,&cb,j->predict,submitted);
}
static void schedule_drain(SchedulePool *p) {
    BnGenerateNativeTextScheduleInfo info;int turns=0;
#ifdef BN_TEST_DENY_HEAP
    control_hot=1;control_heap_ops=0;
#endif
    do {
        BnGenerateNativeTextScheduleStep receipt;
        assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)>=0);
        assert(!bn_generate_native_text_schedule_info(&p->schedule,&info));assert(++turns<512);
    }while(info.live||info.queued);
#ifdef BN_TEST_DENY_HEAP
    control_hot=0;assert(!control_heap_ops);
#endif
}
static void schedule_destroy(SchedulePool *p) {
    bn_generate_native_text_schedule_free(&p->schedule);assert(!p->active&&!p->schedule.state);
    for(int i=0;i<2;i++)bn_tokenizer_encode_scratch_free(p->scratch+i);
    free(p);
}
static void native_schedule_checks(const BnTokenizerEncoder *encoder,const BnTokenizerDecoder *decoder) {
    BnGenerateNativeTextScheduleInfo info;
    /* Each of the three payload budgets independently prevents a second lease.
     * Request3 arrives after work started; reclaimed space admits it normally. */
    for(int dimension=0;dimension<3;dimension++) {
        SchedulePool *p=schedule_pool(encoder,decoder,2,dimension==0?1999:2000,dimension==1?199:200,dimension==2?19:20);
        assert(!schedule_submit(p,0,0)&&!schedule_submit(p,1,0));
        BnGenerateNativeTextScheduleStep receipt;
        assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1&&receipt.id==0&&receipt.status==BN_NATIVE_TEXT_BUSY);
        assert(p->acquires==1);assert(!schedule_submit(p,2,0));schedule_drain(p);
        assert(p->peak==1&&p->acquires==3&&p->releases==3&&!p->discards);
        assert(!bn_generate_native_text_schedule_info(&p->schedule,&info));
        assert(info.completed==3&&!info.queued&&!info.live&&!info.device_bytes&&!info.host_bytes&&!info.pinned_bytes);
        assert(info.busy_returns>0&&info.output_tokens==18);schedule_destroy(p);
    }
    SchedulePool *p=schedule_pool(encoder,decoder,2,2000,200,20);
    assert(!schedule_submit(p,0,0)&&!schedule_submit(p,1,0)&&!schedule_submit(p,2,0));
    BnGenerateNativeTextScheduleStep receipt;
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1&&receipt.id==0&&receipt.status==BN_NATIVE_TEXT_BUSY);
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1&&receipt.id==1&&receipt.status==BN_NATIVE_TEXT_BUSY);
    p->check_reentry=1;
    assert(!bn_generate_native_text_schedule_cancel(&p->schedule,0)&&p->pending_discards==1&&p->jobs[0].cancelled);
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1&&receipt.id==1);
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1&&receipt.id==2);
    schedule_drain(p);assert(p->jobs[1].completed&&p->jobs[2].completed&&p->discards==1);
    BnGenerateNativeTextSchedule copy=p->schedule;
    assert(bn_generate_native_text_schedule_info(&copy,&info)==-1);
    bn_generate_native_text_schedule_free(&copy);assert(p->schedule.state);
    schedule_destroy(p);
    /* A partial decode failure preserves its prefix; the peer still completes. */
    p=schedule_pool(encoder,decoder,2,2000,200,20);
    p->jobs[0].fail_forward=(p->jobs[0].prompt_count+2)/3+2;
    assert(!schedule_submit(p,0,0)&&!schedule_submit(p,1,0));schedule_drain(p);
    assert(p->jobs[0].failed&&p->jobs[0].terminal.generated==1&&!p->controls[0].flushes&&p->jobs[1].completed&&p->discards==1);
    schedule_destroy(p);
    /* Invalid ownership poisons all requests and returns a duplicate only once. */
    p=schedule_pool(encoder,decoder,2,2000,200,20);p->duplicate=1;
    assert(!schedule_submit(p,0,0)&&!schedule_submit(p,1,0));
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==-1&&p->releases==1&&!p->active);
    assert(!bn_generate_native_text_schedule_info(&p->schedule,&info)&&info.poisoned&&info.failed==2&&!info.live&&!info.queued);
    schedule_destroy(p);
    /* A regressing/broken clock cannot leave charged leases behind. */
    p=schedule_pool(encoder,decoder,2,2000,200,20);
    assert(!schedule_submit(p,0,0)&&!schedule_submit(p,1,0));
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1);p->bad_clock=1;
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==-1&&!p->active&&p->discards==2);
    schedule_destroy(p);
    /* Permanent oversize fails without acquiring; transient exhaustion queues. */
    p=schedule_pool(encoder,decoder,2,999,200,20);assert(!schedule_submit(p,0,0));
    assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1&&!receipt.attempted&&p->jobs[0].failed&&!p->acquires);
    schedule_destroy(p);
    for(int mode=0;mode<2;mode++) {
        p=schedule_pool(encoder,decoder,2,2000,200,20);p->busy_plan=mode==0;p->busy_acquire=mode==1;
        assert(!schedule_submit(p,0,0));assert(!bn_generate_native_text_schedule_step(&p->schedule,&receipt)&&!receipt.attempted&&!p->acquires);
        p->busy_plan=p->busy_acquire=0;schedule_drain(p);assert(p->jobs[0].completed);schedule_destroy(p);
    }
    for(int failure=0;failure<2;failure++) {
        p=schedule_pool(encoder,decoder,2,2000,200,20);
        if(failure)p->fail_acquire=1;else p->fail_plan=1;
        assert(!schedule_submit(p,0,0));
        assert(bn_generate_native_text_schedule_step(&p->schedule,&receipt)==1&&!receipt.attempted);
        assert(p->jobs[0].failed&&!p->acquires);
        assert(!bn_generate_native_text_schedule_info(&p->schedule,&info)&&!info.poisoned);
        schedule_destroy(p);
    }
    static const unsigned char bad_utf8[]={0xc0,0x80};
    p=schedule_pool(encoder,decoder,2,2000,200,20);
    p->jobs[0].text=bad_utf8;p->jobs[0].bytes=sizeof(bad_utf8);
    assert(!schedule_submit(p,0,0)&&!schedule_submit(p,1,0));schedule_drain(p);
    assert(p->jobs[0].failed&&p->jobs[1].completed&&p->discards==1);schedule_destroy(p);
    p=schedule_pool(encoder,decoder,2,2000,200,20);
    assert(schedule_submit(p,0,INFINITY)==-1&&schedule_submit(p,0,p->clock+100)==-1&&!p->events);
    assert(!schedule_submit(p,0,0));p->bad_clock=1;
    assert(schedule_submit(p,1,0)==-1&&p->jobs[0].failed);schedule_destroy(p);
    p=schedule_pool(encoder,decoder,2,2000,200,20);p->jobs[0].options.max_new_tokens=0;
    assert(!schedule_submit(p,0,0));schedule_drain(p);assert(p->jobs[0].completed);schedule_destroy(p);
    p=schedule_pool(encoder,decoder,2,2000,200,20);
    for(uint64_t id=0;id<64;id++)assert(!schedule_submit(p,id,0));
    assert(schedule_submit(p,64,0)==1&&schedule_submit(p,0,0)==-1);
    assert(!bn_generate_native_text_schedule_cancel(&p->schedule,7)&&!schedule_submit(p,64,0));
    int events=p->events;bn_generate_native_text_schedule_free(&p->schedule);assert(p->events==events&&!p->acquires);
    schedule_destroy(p);
    puts("Native scheduler: three budgets, continuous arrivals, FIFO/round-robin/BUSY, fenced cancellation, isolated failures, poison cleanup, timestamps and bounded queue PASSED");
}
