#include "native_text_timing.h"
/* Model/codec/session preparation and reference verification are outside the
 * interval. Callbacks record only clock readings, input IDs and output bytes;
 * no logit copies, file IO, heap calls or additional GPU work/fences. */
static void profile_text_latency_verify(ProfileNativeText *t,const BnTextTiming *r,int generated,int position) {
    size_t nf=bn_json_size(t->frames),nt=(size_t)t->generated+1;
    size_t omitted=t->predict&&t->generated>0;
    assert(generated==t->generated&&r->frame_count==nf-omitted&&r->token_count==nt&&position==r->position);
    for(size_t i=0;i<t->prompt_tokens;i++)assert((uint32_t)t->workspace.tokens[i]==t->expected_ids[i]);
    for(size_t i=0;i<r->frame_count;i++) {
        const BnJSONValue *frame=bn_json_at(t->frames,i);BnTextTimingFrame *got=r->frames+i;
        assert(got->position==profile_number(frame,"position")&&got->rows==profile_number(frame,"rows"));
        uint64_t *gold=profile_load(t->directory,frame,"tokens",(size_t)got->rows*8);
        for(int j=0;j<got->rows;j++)assert(gold[j]==(uint64_t)r->inputs[got->offset+(size_t)j]);
        free(gold);
    }
    const BnJSONValue *ids=bn_json_get(bn_json_root(&t->reference),"generated_ids");size_t offset=0;
    for(size_t i=0;i<nt;i++) {
        if(i<(size_t)t->generated) {
            uint64_t id;assert(!bn_json_u64(bn_json_at(ids,i),&id)&&r->tokens[i].id==(int)id);
        }else assert(r->tokens[i].id==-1);
        const BnJSONValue *chunk=bn_json_at(bn_json_get(t->output,"chunks"),i);size_t bytes;
        void *gold=profile_file(t->directory,bn_json_text(bn_json_get(chunk,"file")),&bytes);
        assert(bytes==r->tokens[i].bytes&&bytes<=r->byte_count-offset&&!memcmp(gold,r->bytes+offset,bytes));
        offset+=bytes;free(gold);
    }
    size_t bytes;void *gold=profile_file(t->directory,bn_json_text(bn_json_get(t->output,"full")),&bytes);
    assert(bytes==r->byte_count&&offset==bytes&&!memcmp(gold,r->bytes,bytes));free(gold);
}
static void profile_text_latency(ProfileNativeText *t,const BnModel *model,BnSession *session,
    int run,int prefill,int context,int admit,int ram_admit) {
    size_t nf=bn_json_size(t->frames),nt=(size_t)t->generated+1;
    assert(nf&&nf<=SIZE_MAX/sizeof(BnTextTimingFrame)&&nt<=SIZE_MAX/sizeof(BnTextTimingToken));
    assert(t->prompt_tokens<=(size_t)(INT_MAX-t->generated));
    size_t ni=t->prompt_tokens+(size_t)t->generated;
    BnTextTiming r={.prompt=(int)t->prompt_tokens,.max_new=t->generated,.vocabulary=t->vocabulary,
        .predict=t->predict,.frame_capacity=nf,.token_capacity=nt,.input_capacity=ni,
        .byte_capacity=65536,.bytes=t->joined};
    r.frames=calloc(nf,sizeof(*r.frames));r.tokens=calloc(nt,sizeof(*r.tokens));r.inputs=malloc(ni*sizeof(int));
    assert(r.frames&&r.tokens&&r.inputs);
    t->skip=native_text_workload_skip(run,t->request_count?t->request_count:1);assert(t->skip>=0);
    t->output=bn_json_at(bn_json_get(bn_json_root(&t->reference),"outputs"),(size_t)t->skip);
    int skip;assert(t->output&&!bn_json_bool(bn_json_get(t->output,"skip_special"),&skip)&&skip==t->skip);
    BnGenerateNativeTextOptions options={prefill,t->generated,admit,t->skip};
    BnGenerateNativeTextCallbacks callbacks={native_text_timing_token,native_text_timing_forward,&r};
    BnGPUMoETrellisStats before,after;
    assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(model),&before));
    memset(t->workspace.tokens,0xa5,t->prompt_tokens*sizeof(int));
    int status=native_text_timing_start(&r);assert(!status);
    int generated=t->predict?
        bn_generate_native_text_predict(model,session,&t->workspace,t->prompt,t->prompt_bytes,&options,&callbacks):
        bn_generate_native_text(model,session,&t->workspace,t->prompt,t->prompt_bytes,&options,&callbacks);
    status=native_text_timing_finish(&r,generated);assert(!status);
    assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(model),&after)&&!after.live_leases);
    /* All source comparisons below happen after the timer has stopped. */
    profile_text_latency_verify(t,&r,generated,session->pos);
    printf("{\"kind\":\"native_text_timing\",");
    if(t->request_count>1)printf("\"request_index\":%d,\"request_count\":%d,",t->request_index,t->request_count);
    printf("\"run\":%d,\"prediction_first\":%d,\"skip_special\":%d,"
        "\"prompt_tokens\":%zu,\"prefill_tokens\":%d,\"context_capacity\":%d,\"generated\":%d,\"forwards\":%zu,"
        "\"input_id_checks\":%zu,\"position\":%d,\"request_ms\":%.6f,\"prefill_ms\":%.6f,"
        "\"first_token_ms\":%.6f,\"first_byte_ms\":%.6f,\"post_prefill_first_token_ms\":%.6f,"
        "\"post_prefill_first_byte_ms\":%.6f,\"generation_tail_ms\":%.6f,\"admit_decode\":%d,"
        "\"initial_expert_uploads\":%zu,\"initial_decode_charged\":%zu,\"expert_upload_bytes\":%zu,"
        "\"expert_upload_ms\":%.6f,\"ram_hits\":%zu,\"ram_misses\":%zu,\"ram_miss_bytes\":%zu,"
        "\"decode_hits\":%zu,\"decode_misses\":%zu,\"ram_admit_misses\":%d,"
        "\"source_read_calls\":%d,\"source_read_bytes\":%zu,\"source_read_ms\":%.6f,\"output_bytes\":%zu,\"generated_ids\":[",
        run,t->predict,t->skip,t->prompt_tokens,prefill,context,generated,r.frame_count,r.input_count,r.position,
        r.end-r.start,r.prefill-r.start,r.first_token<0?-1:r.first_token-r.start,r.first_byte<0?-1:r.first_byte-r.start,
        r.first_token<0?-1:r.first_token-r.prefill,r.first_byte<0?-1:r.first_byte-r.prefill,r.end-r.prefill,admit,
        before.upload_count,before.decode.charged_payload,after.upload_bytes-before.upload_bytes,
        after.upload_time_ms-before.upload_time_ms,after.host.gpu_host_cache_hits-before.host.gpu_host_cache_hits,
        after.host.gpu_host_cache_misses-before.host.gpu_host_cache_misses,after.host.gpu_host_cache_miss_bytes-before.host.gpu_host_cache_miss_bytes,
        after.decode.hits-before.decode.hits,after.decode.misses-before.decode.misses,ram_admit,
        after.host.io_count-before.host.io_count,after.host.io_bytes-before.host.io_bytes,after.host.io_time_ms-before.host.io_time_ms,r.byte_count);
    for(int i=0;i<generated;i++)printf("%s%d",i?",":"",r.tokens[i].id);
    printf("],\"token_callback_ms\":[");
    for(size_t i=0;i<r.token_count;i++)printf("%s%.6f",i?",":"",r.tokens[i].time-r.start);
    printf("],\"output_hex\":\"");for(size_t i=0;i<r.byte_count;i++)printf("%02x",r.bytes[i]);
    printf("\"}\n");fflush(stdout);
    free(r.inputs);free(r.tokens);free(r.frames);
}
