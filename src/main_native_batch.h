/* Product adapter: shared scheduler owns admission/chunks/publication; backend
 * session pool owns execution resources. No model-family rules or references. */
#include "generate_batch.h"
typedef struct {
    int *tokens;int count;int32_t *outputs;
    BnSession *session;BnTokenizerDecodeStream stream;size_t frames,bytes;
} NativeCLIBatchJob;
struct NativeCLIBatch {
    NativeCLIBatchInput input;NativeCLIBatchJob jobs[NATIVE_CLI_BATCH_REQUESTS];
    NativeCLIResources *resources;const NativeCLIOptions *options;
    BnGenerateBatch scheduler;NativeCLIMemoryPool pool;
    BnTransformerTrellisRequestConfig ceiling;BnTransformerTrellisRequestPlan quote;
    size_t pool_device,pool_host,pool_pinned;int32_t *outputs;int output_limit;ServeRuntime *serve;
    FILE *trace;int failed,vocab;uint64_t group_calls,single_forwards,retries;
    BnTransformerTrellisGroupStats stats;
};
static int native_cli_batch_read(NativeCLIResources *r,const NativeCLIOptions *o) {
    r->batch=calloc(1,sizeof(*r->batch));
    if(!r->batch)return -1;
    r->batch->output_limit=o->serve&&o->tokens>o->maxseq?o->maxseq:o->tokens;
    if(o->serve){r->batch->input.count=BN_GENERATE_BATCH_QUEUE_CAPACITY;return 0;}
    return native_cli_batch_input_init(&r->batch->input,r->text,r->text_bytes);
}
static int native_cli_batch_host(const NativeCLIResources *r,const NativeCLIOptions *o,size_t *extra) {
    size_t outputs,transport=0;
    if(native_cli_mul((size_t)r->batch->input.count,(size_t)r->batch->output_limit,&outputs)||
       native_cli_mul(outputs,sizeof(int32_t),&outputs)||native_cli_add(extra,outputs)||
       native_cli_add(extra,sizeof(*r->batch))||(o->serve&&serve_runtime_plan(SERVE_RECORD_BYTES,&transport))||
       native_cli_add(extra,transport))return -1;
    return 0;
}
static int native_cli_batch_encode(NativeCLIResources *r,const NativeCLIOptions *o) {
    NativeCLIBatch *b=r->batch;size_t used=0,total;
    if(o->maxseq>bn_model_native_anatomy(&r->model)->context||
       native_cli_mul((size_t)b->input.count,(size_t)b->output_limit,&total))return -1;
    if(o->serve) {
        b->outputs=calloc(total,sizeof(*b->outputs));if(!b->outputs)return -1;
        for(int i=0;i<b->input.count;i++) {
            b->jobs[i].tokens=r->tokens+(size_t)i*((size_t)o->maxseq+1);
            b->jobs[i].outputs=b->outputs+(size_t)i*b->output_limit;
        }
        return 0;
    }
    r->count=0;
    for(int i=0;i<b->input.count;i++) {
        NativeCLIBatchJob *j=b->jobs+i;size_t count=0;const char *text=b->input.prompts[i];
        if(used>=r->text_bytes||bn_tokenizer_encoder_encode(&r->encoder,&r->scratch,
           (const unsigned char*)text,strlen(text),r->tokens+used,r->text_bytes-used,&count)||
           native_cli_context(count,o->tokens,1,o->maxseq)||
           bn_tokenizer_decode_stream_init(&j->stream,&r->decoder,!o->keep_special))return -1;
        j->tokens=r->tokens+used;j->count=(int)count;
        used+=count;if(count>r->count)r->count=count;
    }
    /* Validate every prompt/context before allocating any generated-output
     * slab, including very large invalid -n values with no aggregate cap. */
    b->outputs=calloc(total,sizeof(*b->outputs));if(!b->outputs)return -1;
    for(int i=0;i<b->input.count;i++)b->jobs[i].outputs=b->outputs+(size_t)i*o->tokens;
    return 0;
}
static int native_cli_batch_plan(NativeCLIPlan *p,NativeCLIResources *r,const NativeCLIOptions *o) {
    NativeCLIBatch *b=r->batch;b->resources=r;b->options=o;
    b->vocab=bn_model_native_anatomy(&r->model)->vocab;
    b->ceiling=(BnTransformerTrellisRequestConfig){p->f.max_tokens,o->maxseq,o->window,0,0};
    BnGenerateBatchMemoryQuote request_quote;
    if(bn_transformer_trellis_request_plan(&r->model,&b->ceiling,&b->quote)||
       native_cli_memory_quote(&b->quote,&request_quote)||
       native_cli_mul(b->quote.device_bytes,(size_t)o->batch_live,&b->pool_device)||
       native_cli_mul(b->quote.host_bytes,(size_t)o->batch_live,&b->pool_host)||
       native_cli_mul(b->quote.pinned_bytes,(size_t)o->batch_live,&b->pool_pinned)||
       native_cli_batch_host(r,o,&p->shared.host_bytes))return -1;
    p->device=p->shared.device_bytes;p->host=p->shared.host_bytes;
    size_t requests_host;
    if(native_cli_mul(request_quote.peak.host_bytes,(size_t)o->batch_live,&requests_host)||
       native_cli_add(&p->host,requests_host)||native_cli_add(&p->device,b->pool_device))return -1;
    return 0;
}
static int native_cli_batch_admit_requests(NativeCLIBatch *b) {
    if(b->options->serve) {
        BnTransformerTrellisRequestConfig c=b->ceiling;c.max_tokens=1;c.maxseq=256;
        BnTransformerTrellisRequestPlan plan;BnGenerateBatchMemoryQuote quote;
        return bn_transformer_trellis_request_plan(&b->resources->model,&c,&plan)||native_cli_memory_quote(&plan,&quote)||
            bn_generate_batch_memory_check_shared(&b->resources->memory,quote)?-1:0;
    }
    for(int i=0;i<b->input.count;i++) {
    NativeCLIBatchJob *j=b->jobs+i;
    int max=j->count<b->ceiling.max_tokens?j->count:b->ceiling.max_tokens;
    int needed=j->count+b->options->tokens-1;if(needed<max)needed=max;
    if(needed>INT_MAX-255)return -1;
    BnTransformerTrellisRequestConfig config=b->ceiling;config.max_tokens=max;config.maxseq=(needed+255)/256*256;
    BnTransformerTrellisRequestPlan plan;BnGenerateBatchMemoryQuote quote;
    if(bn_transformer_trellis_request_plan(&b->resources->model,&config,&plan)||
       native_cli_memory_quote(&plan,&quote)||bn_generate_batch_memory_check_shared(&b->resources->memory,quote))return -1;
    }
    return 0;
}
static double native_cli_batch_now(void *context) {(void)context;return bn_platform_time_ms();}
static int native_cli_batch_acquire(void *context,const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out) {
    NativeCLIBatch *b=context;
    if(request->id>=(uint64_t)b->input.count)return -1;
    NativeCLIBatchJob *j=b->jobs+request->id;BnSession *s=NULL;
    int max=j->count<b->ceiling.max_tokens?j->count:b->ceiling.max_tokens;
    int status=native_cli_pool_acquire(&b->pool,max,request->required_capacity,&s);
    if(status)return status;
    j->session=s;*out=(BnGenerateBatchLease){s,s->state.logits,(size_t)b->vocab};return 0;
}
static void native_cli_batch_release(void *context,BnGenerateBatchLease lease,int discard) {
    NativeCLIBatch *b=context;
    if(native_cli_pool_release(&b->pool,lease.handle,discard))b->failed=1;
    for(int i=0;i<b->input.count;i++)if(b->jobs[i].session==lease.handle)b->jobs[i].session=NULL;
}
static int native_cli_batch_trace(NativeCLIBatch *b,const BnGenerateBatchExecutionRow *row) {
    if(!b->trace)return 0;
    NativeCLIBatchJob *j=b->jobs+row->id;char name[128],path[4096];
    for(int kind=0;kind<2;kind++) {
        if(serve_trace_name(b->serve,row->id,j->frames,kind,name,sizeof(name))||native_cli_path(path,sizeof(path),b->options->trace_directory,name)||
           native_cli_dump(path,kind?(const void*)row->lease.logits:(const void*)row->tokens,
               kind?row->lease.logits_count:(size_t)row->n_tokens,kind))return -1;
    }
    uint64_t id=b->serve?b->serve->requests[row->id].id:row->id;
    if(fprintf(b->trace,"{\"kind\":\"forward\",\"request\":%llu,\"index\":%zu,\"position\":%d,\"rows\":%d,\"vocabulary\":%zu",
       (unsigned long long)id,j->frames,row->position,row->n_tokens,row->lease.logits_count)<0||
       (b->serve&&fprintf(b->trace,",\"generation\":%llu,\"slot\":%llu",(unsigned long long)b->serve->requests[row->id].generation,(unsigned long long)row->id)<0)||
       fputs("}\n",b->trace)==EOF)return -1;
    j->frames++;return 0;
}
static int native_cli_batch_execute(void *context,const BnGenerateBatchExecutionRow *rows,int count) {
    NativeCLIBatch *b=context;const BnModel *m=&b->resources->model;
    if(b->failed||count<1||count>BN_GENERATE_BATCH_MAX_REQUESTS)return -1;
    for(int i=0;i<count;i++) {
        const BnGenerateBatchExecutionRow *v=rows+i;BnSession *s=v->lease.handle;
        if(v->id>=(uint64_t)b->input.count||s!=b->jobs[v->id].session||s->pos!=v->position||
           bn_session_native_history_stage(s,m,(const int*)v->tokens,v->n_tokens))return -1;
    }
    /* Uniform policies can share expert leases. Policy0 includes all rows
     * when decode admission is disabled; otherwise prefill/decode split. */
    for(int policy=0;policy<=(b->options->admit?1:0);policy++) {
        BnSession *sessions[16];BnTransformerTrellisForwardResult result[16];int n=0;
        for(int i=0;i<count;i++)if((rows[i].is_prefill?0:b->options->admit)==policy)sessions[n++]=rows[i].lease.handle;
        if(!n)continue;
        int status=BN_GPU_UPLOAD_BUSY,attempt=0;BnTransformerTrellisGroupStats stats={0};
        /* A serialized product step must finish its cohort. Permanent or
         * excessive backpressure fails/discards, rather than spinning forever. */
        while(status==BN_GPU_UPLOAD_BUSY&&attempt++<64) {
            status=n>1?bn_transformer_trellis_forward_group(m,sessions,n,policy,result,&stats):
                bn_transformer_trellis_forward_run(m,sessions[0],policy,result);
            if(status==BN_GPU_UPLOAD_BUSY)b->retries++;
        }
        if(status) {
            if(status==BN_GPU_UPLOAD_BUSY)fprintf(stderr,"native_batch_backpressure_limit attempts=64 participants=%d\n",n);
            if(status==BN_GPU_UPLOAD_BUSY&&n>1)(void)bn_transformer_trellis_forward_group_cancel(m,sessions,n);
            return -1;
        }
        if(n>1) {
            b->group_calls++;b->stats.calls+=stats.calls;b->stats.shared_layers+=stats.shared_layers;
            b->stats.individual_layers+=stats.individual_layers;b->stats.expert_borrows+=stats.expert_borrows;
            b->stats.fused_launches+=stats.fused_launches;b->stats.folded_launches+=stats.folded_launches;b->stats.waits+=stats.waits;
            b->stats.joint_fused_launches+=stats.joint_fused_launches;
            b->stats.fused_launches_avoided+=stats.fused_launches_avoided;
            b->stats.individual_fused_launches+=stats.individual_fused_launches;
            b->stats.lookahead_borrows+=stats.lookahead_borrows;
            b->stats.lookahead_busy_stops+=stats.lookahead_busy_stops;
            b->stats.decode_layers+=stats.decode_layers;b->stats.decode_cohorts+=stats.decode_cohorts;
            b->stats.decode_submissions+=stats.decode_submissions;b->stats.decode_expert_reuses+=stats.decode_expert_reuses;
        }else b->single_forwards++;
        for(int i=0;i<n;i++)if(!bn_transformer_trellis_forward_current(m,sessions[i],result+i))return -1;
    }
    for(int i=0;i<count;i++)if(native_cli_batch_trace(b,rows+i))return -1;
    return 0;
}
static void native_cli_batch_notify(void *context,const BnGenerateBatchEvent *event) {
    NativeCLIBatch *b=context;if(b->failed)return;
    uint64_t wire=event->id;
    if(b->serve&&serve_runtime_event(b->serve,event,&wire)){b->failed=1;return;}
    unsigned long long id=(unsigned long long)wire;
    if(event->id>=(uint64_t)b->input.count){b->failed=1;return;}
    NativeCLIBatchJob *j=b->jobs+event->id;NativeCLIResources *r=b->resources;size_t bytes=0;
    if(event->kind==BN_GENERATE_BATCH_ADMITTED) {
        if(printf("{\"kind\":\"admitted\",\"request\":%llu,\"prompt_tokens\":%d",id,j->count)<0||serve_event_end(b->serve,event))b->failed=1;
    }else if(event->kind==BN_GENERATE_BATCH_TOKEN) {
        if(bn_tokenizer_decode_stream_push(&r->decoder,&j->stream,&event->token,1,0,r->piece,r->piece_bytes,&bytes)||
           printf("{\"kind\":\"token\",\"request\":%llu,\"index\":%d,\"token\":%d,\"text\":",id,event->output_index,event->token)<0||
           native_cli_batch_json_text(stdout,r->piece,bytes)||serve_event_end(b->serve,event)||native_cli_add(&j->bytes,bytes))b->failed=1;
        if(event->complete&&!b->failed) {
            if(bn_tokenizer_decode_stream_push(&r->decoder,&j->stream,NULL,0,1,r->piece,r->piece_bytes,&bytes)||
               printf("{\"kind\":\"completed\",\"request\":%llu,\"generated\":%d,\"position\":%d,\"text\":",id,event->output_index+1,j->session->pos)<0||
               native_cli_batch_json_text(stdout,r->piece,bytes)||serve_event_end(b->serve,event)||native_cli_add(&j->bytes,bytes))b->failed=1;
        }
    }else {
        const char *kind=event->kind==BN_GENERATE_BATCH_CANCELLED?"cancelled":"failed";
        if(printf("{\"kind\":\"%s\",\"request\":%llu",kind,id)<0)b->failed=1;
        if(b->serve&&!b->failed&&(bn_tokenizer_decode_stream_push(&r->decoder,&j->stream,NULL,0,1,r->piece,r->piece_bytes,&bytes)||
           fputs(",\"text\":",stdout)==EOF||native_cli_batch_json_text(stdout,r->piece,bytes)))b->failed=1;
        if(!b->failed&&serve_event_end(b->serve,event))b->failed=1;
    }
    if(fflush(stdout))b->failed=1;
}
static int native_cli_batch_prepare_serving(void *context,int slot,const ServeCommand *command,ServePrepared *out) {
    NativeCLIBatch *b=context;NativeCLIResources *r=b->resources;NativeCLIBatchJob *j=b->jobs+slot;size_t count=0;
    if(j->session)return -1;
    if(bn_tokenizer_encoder_encode(&r->encoder,&r->scratch,(const unsigned char*)command->prompt,strlen(command->prompt),
        j->tokens,(size_t)b->options->maxseq+1,&count)||native_cli_context(count,command->max_outputs,1,b->options->maxseq))return 1;
    BnTransformerTrellisRequestConfig c=b->ceiling;c.max_tokens=count<(size_t)c.max_tokens?(int)count:c.max_tokens;
    int needed=(int)count+command->max_outputs-1;if(needed>INT_MAX-255)return 1;c.maxseq=(needed+255)/256*256;
    BnTransformerTrellisRequestPlan plan;BnGenerateBatchMemoryQuote quote;
    if(bn_transformer_trellis_request_plan(&r->model,&c,&plan)||native_cli_memory_quote(&plan,&quote)||
       plan.device_bytes>b->pool_device||plan.host_bytes>b->pool_host||plan.pinned_bytes>b->pool_pinned||
       bn_generate_batch_memory_check_shared(&r->memory,quote))return 1;
    bn_tokenizer_decode_stream_free(&j->stream);
    if(bn_tokenizer_decode_stream_init(&j->stream,&r->decoder,!b->options->keep_special))return -1;
    j->count=(int)count;j->frames=j->bytes=0;*out=(ServePrepared){(const int32_t*)j->tokens,j->count,j->outputs};return 0;
}
static int native_cli_batch_run(NativeCLIResources *r,const NativeCLIOptions *o,FILE *trace) {
    NativeCLIBatch *b=r->batch;b->trace=trace;
    BnGenerateBatchLimits limits={o->batch_live,o->batch_live,b->ceiling.max_tokens,o->batch_budget,o->maxseq,o->mixed_budget};
    BnGenerateBatchBackend backend={b,NULL,native_cli_batch_execute,native_cli_batch_release,native_cli_batch_now};
    if(native_cli_pool_init(&b->pool,&r->model,&b->ceiling,o->batch_live,b->pool_device,b->pool_host,b->pool_pinned,&r->memory)||
       bn_generate_batch_init_request_aware(&b->scheduler,&limits,b->vocab,&backend,
           native_cli_batch_acquire,native_cli_batch_notify,b))return -1;
    if(o->serve) {
        b->serve=calloc(1,sizeof(*b->serve));
        if(!b->serve||serve_runtime_init(b->serve,&b->scheduler,native_cli_batch_prepare_serving,b,native_cli_batch_now,b,&b->failed,
            SERVE_RECORD_BYTES,b->output_limit)||serve_runtime_run(b->serve,STDIN_FILENO))return -1;
    }else {
    for(int i=0;i<b->input.count;i++)if(bn_generate_batch_enqueue(&b->scheduler,(uint64_t)i,
        (const int32_t*)b->jobs[i].tokens,b->jobs[i].count,b->jobs[i].outputs,o->tokens))return -1;
    BnGenerateBatchInfo info;
    for(;;) {
        if(bn_generate_batch_info(&b->scheduler,&info))return -1;
        if(!info.live&&!info.queued)break;
        if(bn_generate_batch_step(&b->scheduler)!=1||b->failed)return -1;
    }
    }
    BnGenerateBatchInfo info;if(bn_generate_batch_info(&b->scheduler,&info))return -1;
    BnTransformerTrellisPoolInfo pool;
    if(info.failed||(o->serve?info.completed+info.cancelled!=info.submitted:info.completed!=(uint64_t)b->input.count)||
       native_cli_pool_info(&b->pool,&pool)||pool.leased_slots)return -1;
    fprintf(stderr,"native_batch_result completed=%llu steps=%llu group_calls=%llu single_forwards=%llu shared_layer_steps=%llu fallback_layer_steps=%llu shared_expert_borrows=%llu busy_retries=%llu created=%llu reused=%llu retained_device_payload=%zu peak_device_payload=%zu peak_live=%d\n",
        (unsigned long long)info.completed,(unsigned long long)info.steps,(unsigned long long)b->group_calls,
        (unsigned long long)b->single_forwards,(unsigned long long)b->stats.shared_layers,(unsigned long long)b->stats.individual_layers,
        (unsigned long long)b->stats.expert_borrows,(unsigned long long)b->retries,(unsigned long long)pool.created,
        (unsigned long long)pool.reused,pool.reserved_device,pool.peak_device,info.peak_live);
    fprintf(stderr,"native_batch_decode coordinated_layer_steps=%llu lease_cohorts=%llu private_kernel_submissions=%llu reused_expert_bindings=%llu\n",
        (unsigned long long)b->stats.decode_layers,(unsigned long long)b->stats.decode_cohorts,
        (unsigned long long)b->stats.decode_submissions,(unsigned long long)b->stats.decode_expert_reuses);
    fprintf(stderr,"native_batch_compute fused_kernel_launches=%llu joint_fused_kernel_launches=%llu avoided_fused_launches=%llu individual_fused_kernel_launches=%llu folded_ffn_calls=%llu\n",
        (unsigned long long)b->stats.fused_launches,(unsigned long long)b->stats.joint_fused_launches,
        (unsigned long long)b->stats.fused_launches_avoided,(unsigned long long)b->stats.individual_fused_launches,
        (unsigned long long)b->stats.folded_launches);
    fprintf(stderr,"native_batch_lookahead queued_borrows=%llu busy_stops=%llu\n",
        (unsigned long long)b->stats.lookahead_borrows,(unsigned long long)b->stats.lookahead_busy_stops);
    native_cli_memory_report(&r->memory,"native_batch_memory");
    BnGPUMoETrellisStats experts;
    if(bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(&r->model),&experts)||experts.live_leases)return -1;
    fprintf(stderr,"native_batch_experts upload_bytes=%zu upload_ms=%.6f ram_miss_bytes=%zu source_read_bytes=%zu source_read_ms=%.6f\n",
        experts.upload_bytes,experts.upload_time_ms,experts.host.gpu_host_cache_miss_bytes,experts.host.io_bytes,experts.host.io_time_ms);
    fprintf(stderr,"native_batch_pool retained_host_payload=%zu peak_host_payload=%zu retained_pinned_payload=%zu peak_pinned_payload=%zu evicted=%llu discarded=%llu restart_failures=%llu slots=%d idle=%d retained_metadata_payload=%zu\n",
        pool.reserved_host,pool.peak_host,pool.reserved_pinned,pool.peak_pinned,(unsigned long long)pool.evicted,
        (unsigned long long)pool.discarded,(unsigned long long)pool.restart_failures,pool.allocated_slots,pool.allocated_slots-pool.leased_slots,
        (size_t)pool.allocated_slots*(sizeof(BnSession)+sizeof(BnSessionBackendState)+sizeof(SHArena)));
    for(int i=0;i<16;i++)if(b->pool.slots[i].session) {
        const BnTransformerTrellisRequestPlan *q=&b->pool.slots[i].plan;
        fprintf(stderr,"native_batch_slot slot=%d max_tokens=%d capacity=%d host_payload=%zu pinned_payload=%zu device_payload=%zu\n",
            i,b->pool.slots[i].config.max_tokens,b->pool.slots[i].config.maxseq,q->host_bytes,q->pinned_bytes,q->device_bytes);
    }
    return b->failed?-1:0;
}
static void native_cli_batch_free(NativeCLIBatch *b) {
    if(!b)return;
    bn_generate_batch_free(&b->scheduler);native_cli_pool_free(&b->pool);
    if(b->serve){serve_runtime_free(b->serve);free(b->serve);}
    bn_json_free(&b->input.json);free(b->outputs);free(b);
}
