/* Native serving benchmark adapter. Uses the public scheduler and retained
 * transformer pool. Oracle files never supply actual execution token IDs. */
#include "native_schedule_metrics.h"
#include <threads.h>
typedef struct NativeServing NativeServing;
typedef struct {
    NativeServing *owner;
    ProfileNativeText text;
    BnTextTiming timing;
    BnGenerateNativeTextOptions options;
    BnGenerateNativeTextCharge quote;
    BnGenerateNativeTextEvent terminal;
    BnSession *session;
    size_t scratch_bytes;
    int id,prefill,required,capacity,end_position,ended,admitted,started,acquire_attempts;
    double acquire_ms,cleanup_ms;
} NativeServingJob;
struct NativeServing {
    const BnModel *model;
    BnTransformerTrellisPool pool;
    BnGenerateNativeTextSchedule schedule;
    NativeServingJob *jobs;
    int count,timed,terminals;
};
static size_t profile_serving_add(size_t a,size_t b){assert(b<=SIZE_MAX-a);return a+b;}
static size_t profile_serving_scale(size_t a,int b){assert(b>0&&a<=SIZE_MAX/(size_t)b);return a*(size_t)b;}
static int profile_serving_plan(void *ptr,uint64_t id,size_t bytes,
    const BnGenerateNativeTextOptions *options,int predict,BnGenerateNativeTextCharge *out) {
    NativeServing *s=ptr;if(id>=(uint64_t)s->count)return -1;NativeServingJob *j=s->jobs+id;
    if(bytes!=j->text.prompt_bytes||predict!=j->text.predict||
       options->prefill_tokens!=j->options.prefill_tokens||options->max_new_tokens!=j->options.max_new_tokens||
       options->admit_decode!=j->options.admit_decode||options->skip_special!=j->options.skip_special)return -1;
    *out=j->quote;return 0;
}
static int profile_serving_acquire(void *ptr,uint64_t id,const BnGenerateNativeTextCharge *q,
    BnGenerateNativeTextLease *out) {
    NativeServing *s=ptr;assert(id<(uint64_t)s->count);NativeServingJob *j=s->jobs+id;
    assert(!j->session&&q->max_tokens==j->quote.max_tokens&&q->capacity==j->quote.capacity&&
        q->device_bytes==j->quote.device_bytes&&q->host_bytes==j->quote.host_bytes&&q->pinned_bytes==j->quote.pinned_bytes);
    double begin=bn_platform_time_ms();BnSession *lease=NULL;
    int result=bn_transformer_trellis_pool_acquire(&s->pool,j->prefill,j->required,&lease);
    j->acquire_ms+=bn_platform_time_ms()-begin;j->acquire_attempts++;
    if(result)return result;
    assert(lease);j->session=lease;*out=(BnGenerateNativeTextLease){lease,j->text.workspace};return 0;
}
static void profile_serving_release(void *ptr,BnGenerateNativeTextLease lease,int discard) {
    NativeServing *s=ptr;int id=0;while(id<s->count&&s->jobs[id].session!=lease.session)id++;
    assert(id<s->count);NativeServingJob *j=s->jobs+id;j->end_position=lease.session->pos;
    double begin=bn_platform_time_ms();assert(!bn_transformer_trellis_pool_release(&s->pool,lease.session,discard));
    j->cleanup_ms+=bn_platform_time_ms()-begin;j->session=NULL;
}
static double profile_serving_clock(void *ptr){(void)ptr;return bn_platform_time_ms();}
static void profile_serving_forward(const int *tokens,int rows,int pos,const float *logits,int vocab,void *ptr) {
    NativeServingJob *j=ptr;native_text_timing_forward(tokens,rows,pos,logits,vocab,&j->timing);
    if(!j->owner->timed)profile_text_forward(tokens,rows,pos,logits,vocab,&j->text);
}
static int profile_serving_token(const unsigned char *piece,size_t bytes,int id,void *ptr) {
    NativeServingJob *j=ptr;int stop=native_text_timing_token(piece,bytes,id,&j->timing);
    if(!j->owner->timed)stop|=profile_text_token(piece,bytes,id,&j->text);
    return stop;
}
static void profile_serving_event(const BnGenerateNativeTextEvent *event,void *ptr) {
    NativeServing *s=ptr;assert(event->id<(uint64_t)s->count);NativeServingJob *j=s->jobs+event->id;
    if(event->kind==BN_NATIVE_TEXT_ADMITTED){assert(!j->admitted);j->admitted=1;return;}
    assert(!j->ended&&!j->session&&j->admitted&&event->admitted_ms>=event->submitted_ms&&
        event->timestamp_ms>=event->admitted_ms&&!j->timing.bad);
    if(event->kind==BN_NATIVE_TEXT_COMPLETED) {
        assert(event->generated==j->text.generated&&!native_text_timing_finish(&j->timing,event->generated));
        assert(j->end_position==j->timing.position);
    }else {
        assert(event->kind==BN_NATIVE_TEXT_REQUEST_CANCELLED&&!event->generated&&
            j->timing.frame_count==1&&!j->timing.token_count&&!j->timing.byte_count&&!j->timing.flushed);
        j->timing.end=event->timestamp_ms;
    }
    j->terminal=*event;j->ended=1;s->terminals++;
}
static void profile_serving_job_init(NativeServing *s,NativeServingJob *j,ProfileNativeText *base,
    int id,int references,int prefill,int admit,const BnTransformerTrellisRequestConfig *ceiling,
    size_t *workspaces,size_t *recording) {
    j->owner=s;j->id=id;j->prefill=prefill;ProfileNativeText *t=&j->text;
    /* Borrow immutable source owners; never copy initialized JSON/codec handles. */
    t->directory=base->directory;t->frames=base->frames;t->hostframes=base->hostframes;
    t->prompt=base->prompt;t->prompt_bytes=base->prompt_bytes;t->prompt_tokens=base->prompt_tokens;
    t->expected_ids=base->expected_ids;t->vocabulary=base->vocabulary;t->generated=base->generated;
    t->predict=base->predict;t->request_index=id%references;t->request_count=references;
    profile_json(&t->reference,t->directory,"text-reference.json",0);
    t->skip=native_text_workload_skip(id,references);assert(t->skip>=0);
    t->output=bn_json_at(bn_json_get(bn_json_root(&t->reference),"outputs"),(size_t)t->skip);assert(t->output);
    t->prefills=(t->prompt_tokens+(size_t)prefill-1)/(size_t)prefill;
    assert(!bn_tokenizer_encode_scratch_plan(t->prompt_bytes,&j->scratch_bytes)&&
        !bn_tokenizer_encode_scratch_init(&t->scratch,t->prompt_bytes,j->scratch_bytes));
    t->workspace=(BnGenerateNativeTextWorkspace){&base->encoder,&base->decoder,&t->scratch,
        malloc(t->prompt_tokens*sizeof(int)),t->prompt_tokens,malloc(65536),65536};
    t->joined=malloc(65536);assert(t->workspace.tokens&&t->workspace.piece&&t->joined);
    memset(t->workspace.tokens,0xa5,t->prompt_tokens*sizeof(int));
    j->options=(BnGenerateNativeTextOptions){prefill,t->generated,admit,t->skip};
    int commits=t->generated-(t->predict&&t->generated>0);
    assert(t->prompt_tokens<=(size_t)(INT_MAX-255-commits));
    j->required=(int)t->prompt_tokens+commits;int needed=j->required>prefill?j->required:prefill;
    j->capacity=(needed+255)/256*256;
    BnTransformerTrellisRequestConfig config=*ceiling;config.max_tokens=prefill;config.maxseq=j->capacity;
    BnTransformerTrellisRequestPlan plan;assert(!bn_transformer_trellis_request_plan(s->model,&config,&plan));
    size_t workspace=profile_serving_add(j->scratch_bytes,t->prompt_tokens*sizeof(int));
    workspace=profile_serving_add(workspace,t->workspace.piece_capacity);*workspaces=profile_serving_add(*workspaces,workspace);
    j->quote=(BnGenerateNativeTextCharge){prefill,j->capacity,plan.device_bytes,
        profile_serving_add(plan.host_bytes,workspace),plan.pinned_bytes};
    size_t nf=bn_json_size(t->frames),nt=(size_t)t->generated+1,ni=t->prompt_tokens+(size_t)t->generated;
    j->timing=(BnTextTiming){.prompt=(int)t->prompt_tokens,.max_new=t->generated,.vocabulary=t->vocabulary,
        .predict=t->predict,.frame_capacity=nf,.token_capacity=nt,.input_capacity=ni,.byte_capacity=65536,
        .frames=calloc(nf,sizeof(BnTextTimingFrame)),.tokens=calloc(nt,sizeof(BnTextTimingToken)),
        .inputs=malloc(ni*sizeof(int)),.bytes=t->joined};
    assert(j->timing.frames&&j->timing.tokens&&j->timing.inputs);
    size_t records=profile_serving_add(nf*sizeof(BnTextTimingFrame),nt*sizeof(BnTextTimingToken));
    records=profile_serving_add(records,ni*sizeof(int));records=profile_serving_add(records,65536);
    *recording=profile_serving_add(*recording,records);
}
static void profile_serving_cancel_verify(NativeServingJob *j) {
    ProfileNativeText *t=&j->text;BnTextTiming *r=&j->timing;assert(r->frame_count==1&&!r->bad);
    for(size_t i=0;i<t->prompt_tokens;i++)assert((uint32_t)t->workspace.tokens[i]==t->expected_ids[i]);
    const BnJSONValue *frame=bn_json_at(t->frames,0);int rows=profile_number(frame,"rows");
    assert(r->frames[0].position==0&&r->frames[0].rows==rows&&r->input_count==(size_t)rows&&j->end_position==rows);
    uint64_t *gold=profile_load(t->directory,frame,"tokens",(size_t)rows*8);
    for(int i=0;i<rows;i++)assert(gold[i]==(uint64_t)r->inputs[i]);
    free(gold);
}
static void profile_serving_distribution(const char *name,const double *values,size_t count) {
    BnNativeServingDistribution d;
    if(!count){printf("\"%s\":null",name);return;}
    assert(!native_serving_distribution(values,count,&d));
    printf("\"%s\":{\"n\":%zu,\"min\":%.6f,\"p50\":%.6f,\"p95\":%.6f,\"p99\":%.6f,\"max\":%.6f}",
        name,d.count,d.minimum,d.p50,d.p95,d.p99,d.maximum);
}
static void profile_text_schedule(const BnModel *model,ProfileNativeText *base,int references,
    const int *prefills,int requests,int max_tokens,int maxseq,int window,int pool_slots,int live,
    int timed,int arrival_ms,int cancel_id,int admit) {
    NativeServing s={.model=model,.count=requests,.timed=timed};assert(requests<=BN_NATIVE_SERVING_REQUESTS&&maxseq<=INT_MAX-255);
    BnTransformerTrellisRequestConfig ceiling={max_tokens,(maxseq+255)/256*256,window,0,0};
    BnTransformerTrellisRequestPlan plan;assert(!bn_transformer_trellis_request_plan(model,&ceiling,&plan));
    assert(!bn_transformer_trellis_pool_init(&s.pool,model,&ceiling,pool_slots,
        profile_serving_scale(plan.device_bytes,pool_slots),profile_serving_scale(plan.host_bytes,pool_slots),
        profile_serving_scale(plan.pinned_bytes,pool_slots)));
    s.jobs=calloc((size_t)requests,sizeof(*s.jobs));assert(s.jobs);
    size_t workspaces=0,recording=0,host_quote=0;
    for(int id=0;id<requests;id++) {
        int index=id%references;profile_serving_job_init(&s,s.jobs+id,base+index,id,references,
            prefills[index],admit,&ceiling,&workspaces,&recording);
        if(s.jobs[id].quote.host_bytes>host_quote)host_quote=s.jobs[id].quote.host_bytes;
    }
    BnGenerateNativeTextScheduleBackend backend={&s,profile_serving_plan,profile_serving_acquire,
        profile_serving_release,profile_serving_clock};
    BnGenerateNativeTextScheduleLimits limits={live,profile_serving_scale(plan.device_bytes,live),
        profile_serving_scale(host_quote,live),profile_serving_scale(plan.pinned_bytes,live)};
    assert(!bn_generate_native_text_schedule_init(&s.schedule,model,&limits,&backend,profile_serving_event,&s));
    BnGPUMoETrellisStats before,after;assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(model),&before));
    /* Preparation above and oracle verification below are outside the window.
     * Arrival backlog, all acquire/encode/forward/release work are inside. */
    double start=bn_platform_time_ms();int next=0,cancelled=0;
    while(s.terminals<requests) {
        double now=bn_platform_time_ms();
        while(next<requests&&now>=start+(double)next*arrival_ms) {
            NativeServingJob *j=s.jobs+next;BnGenerateNativeTextCallbacks cb={profile_serving_token,profile_serving_forward,j};
            if(!j->started){assert(!native_text_timing_start(&j->timing));j->timing.start=start+(double)next*arrival_ms;j->started=1;}
            int rc=bn_generate_native_text_schedule_enqueue_at(&s.schedule,(uint64_t)next,j->text.prompt,
                j->text.prompt_bytes,&j->options,&cb,j->text.predict,start+(double)next*arrival_ms);
            assert(rc>=0);if(rc==1)break;next++;
        }
        BnGenerateNativeTextScheduleStep receipt;int rc=bn_generate_native_text_schedule_step(&s.schedule,&receipt);assert(rc>=0);
        if(cancel_id>=0&&!cancelled&&receipt.attempted&&receipt.id==(uint64_t)cancel_id&&s.jobs[cancel_id].timing.frame_count) {
            assert(!s.jobs[cancel_id].ended&&!bn_generate_native_text_schedule_cancel(&s.schedule,(uint64_t)cancel_id));cancelled=1;
        }
        if(!rc){struct timespec pause={0,1000000};thrd_sleep(&pause,NULL);}
    }
    double end=bn_platform_time_ms();assert(end>start&&next==requests&&cancelled==(cancel_id>=0));
    assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(model),&after)&&!after.live_leases);
    BnGenerateNativeTextScheduleInfo schedule;BnTransformerTrellisPoolInfo pool;
    assert(!bn_generate_native_text_schedule_info(&s.schedule,&schedule)&&!bn_transformer_trellis_pool_info(&s.pool,&pool));
    assert(!schedule.live&&!schedule.queued&&!schedule.poisoned&&!schedule.failed&&!schedule.device_bytes&&
        !schedule.host_bytes&&!schedule.pinned_bytes&&!pool.leased_slots&&!pool.restart_failures);
    double requests_ms[BN_NATIVE_SERVING_REQUESTS],tokens_ms[BN_NATIVE_SERVING_REQUESTS],bytes_ms[BN_NATIVE_SERVING_REQUESTS];
    size_t completed=0,first_tokens=0,first_bytes=0;uint64_t generated=0;
    for(int id=0;id<requests;id++) {
        NativeServingJob *j=s.jobs+id;ProfileNativeText *t=&j->text;BnGenerateNativeTextEvent *e=&j->terminal;
        assert(j->ended&&!j->session);
        if(e->kind==BN_NATIVE_TEXT_COMPLETED) {
            profile_text_latency_verify(t,&j->timing,e->generated,j->end_position);
            if(!timed)profile_text_verify(t,e->generated,j->end_position,id);
            requests_ms[completed++]=e->timestamp_ms-e->submitted_ms;generated+=(uint64_t)e->generated;
            if(e->first_token_ms>=0)tokens_ms[first_tokens++]=e->first_token_ms-e->submitted_ms;
            if(e->first_byte_ms>=0)bytes_ms[first_bytes++]=e->first_byte_ms-e->submitted_ms;
        }else profile_serving_cancel_verify(j);
        printf("{\"kind\":\"native_serving_request\",\"id\":%d,\"request_index\":%d,\"cancelled\":%d,"
            "\"prediction_first\":%d,\"capacity\":%d,\"prefill_tokens\":%d,\"generated\":%d,\"source_checked\":true,"
            "\"submitted_ms\":%.6f,\"admitted_ms\":%.6f,\"terminal_ms\":%.6f,\"first_token_ms\":%.6f,\"first_byte_ms\":%.6f,"
            "\"request_ms\":%.6f,\"admission_delay_ms\":%.6f,\"acquire_ms\":%.6f,\"acquire_attempts\":%d,\"cleanup_ms\":%.6f,\"prompt_tokens\":%zu,\"position\":%d,\"forwards\":%zu,\"input_id_checks\":%zu,\"generated_ids\":[",
            id,t->request_index,e->kind==BN_NATIVE_TEXT_REQUEST_CANCELLED,t->predict,j->capacity,j->prefill,e->generated,
            e->submitted_ms-start,e->admitted_ms-start,e->timestamp_ms-start,e->first_token_ms<0?-1:e->first_token_ms-start,
            e->first_byte_ms<0?-1:e->first_byte_ms-start,e->timestamp_ms-e->submitted_ms,e->admitted_ms-e->submitted_ms,
            j->acquire_ms,j->acquire_attempts,j->cleanup_ms,t->prompt_tokens,j->end_position,j->timing.frame_count,j->timing.input_count);
        for(int token=0;token<j->timing.generated;token++)printf("%s%d",token?",":"",j->timing.tokens[token].id);
        printf("],\"output_hex\":\"");
        for(size_t byte=0;byte<j->timing.byte_count;byte++)printf("%02x",j->timing.bytes[byte]);
        puts("\"}");
    }
    printf("{\"kind\":\"native_serving_summary\",\"timing_valid\":%s,\"execution\":\"serialized_interleaved_forward_turns\","
        "\"quantile\":\"nearest_rank_completed_only\",\"requests\":%d,\"completed\":%zu,\"cancelled\":%d,\"max_live\":%d,\"peak_live\":%d,"
        "\"arrival_ms\":%d,\"window_ms\":%.6f,\"completed_tokens\":%llu,\"requests_per_s\":%.6f,\"tokens_per_s\":%.6f,"
        "\"pool_slots\":%d,\"pool_allocated\":%d,\"created\":%llu,\"reused\":%llu,\"evicted\":%llu,\"discarded\":%llu,"
        "\"pool_device\":%zu,\"pool_peak_device\":%zu,\"active_peak_device\":%zu,\"pool_host\":%zu,\"pool_pinned\":%zu,"
        "\"prepared_workspace_payload\":%zu,\"recording_payload\":%zu,\"expert_upload_bytes\":%zu,\"expert_upload_ms\":%.6f,"
        "\"ram_misses\":%zu,\"ram_miss_bytes\":%zu,\"source_read_bytes\":%zu,\"source_read_ms\":%.6f,",
        timed?"true":"false",requests,completed,cancelled,live,schedule.peak_live,arrival_ms,end-start,
        (unsigned long long)generated,1000*(double)completed/(end-start),1000*(double)generated/(end-start),
        pool_slots,pool.allocated_slots,(unsigned long long)pool.created,(unsigned long long)pool.reused,
        (unsigned long long)pool.evicted,(unsigned long long)pool.discarded,pool.reserved_device,pool.peak_device,
        schedule.peak_device_bytes,pool.reserved_host,pool.reserved_pinned,workspaces,recording,
        after.upload_bytes-before.upload_bytes,after.upload_time_ms-before.upload_time_ms,
        after.host.gpu_host_cache_misses-before.host.gpu_host_cache_misses,
        after.host.gpu_host_cache_miss_bytes-before.host.gpu_host_cache_miss_bytes,
        after.host.io_bytes-before.host.io_bytes,after.host.io_time_ms-before.host.io_time_ms);
    printf("\"released\":%llu,\"pool_leased\":%d,\"restart_failures\":%llu,\"busy_returns\":%llu,"
        "\"turns\":%llu,\"forwards\":%llu,\"output_tokens\":%llu,"
        "\"pool_budget_device\":%zu,\"pool_budget_host\":%zu,\"pool_budget_pinned\":%zu,"
        "\"pool_peak_host\":%zu,\"pool_peak_pinned\":%zu,\"active_peak_host\":%zu,\"active_peak_pinned\":%zu,"
        "\"active_budget_device\":%zu,\"active_budget_host\":%zu,\"active_budget_pinned\":%zu,",
        (unsigned long long)pool.released,pool.leased_slots,(unsigned long long)pool.restart_failures,
        (unsigned long long)schedule.busy_returns,(unsigned long long)schedule.turns,
        (unsigned long long)schedule.forwards,(unsigned long long)schedule.output_tokens,
        pool.device_budget,pool.host_budget,pool.pinned_budget,pool.peak_host,pool.peak_pinned,
        schedule.peak_host_bytes,schedule.peak_pinned_bytes,limits.device_budget,limits.host_budget,limits.pinned_budget);
    profile_serving_distribution("request_ms",requests_ms,completed);putchar(',');
    profile_serving_distribution("first_token_ms",tokens_ms,first_tokens);putchar(',');
    profile_serving_distribution("first_byte_ms",bytes_ms,first_bytes);puts("}");
    bn_generate_native_text_schedule_free(&s.schedule);bn_transformer_trellis_pool_free(&s.pool);
    for(int id=0;id<requests;id++) {
        NativeServingJob *j=s.jobs+id;ProfileNativeText *t=&j->text;
        bn_tokenizer_encode_scratch_free(&t->scratch);bn_json_free(&t->reference);
        free(t->workspace.tokens);free(t->workspace.piece);free(t->joined);
        free(j->timing.frames);free(j->timing.tokens);free(j->timing.inputs);
    }
    free(s.jobs);
}
