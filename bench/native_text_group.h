/* Explicit static-cohort diagnostic for the grouped transformer API. Actual
 * raw encoding and greedy logits drive inference; source data only asserts. */
static int profile_group_greedy(const float *logits,int vocabulary) {
    int best=0;for(int i=0;i<vocabulary;i++){assert(isfinite(logits[i]));if(logits[i]>logits[best])best=i;}return best;
}
static void profile_text_group(const BnModel *model,ProfileNativeText *base,int references,const int *prefills,
    int requests,int max_tokens,int maxseq,int window,int pool_slots,int width,int timed,int admit) {
    NativeServing s={.model=model,.count=requests,.timed=timed};assert(width<=pool_slots&&requests<=BN_NATIVE_SERVING_REQUESTS);
    BnTransformerTrellisRequestConfig ceiling={max_tokens,(maxseq+255)/256*256,window,0,0};
    BnTransformerTrellisRequestPlan plan;assert(!bn_transformer_trellis_request_plan(model,&ceiling,&plan));
    assert(!bn_transformer_trellis_pool_init(&s.pool,model,&ceiling,pool_slots,
        profile_serving_scale(plan.device_bytes,pool_slots),profile_serving_scale(plan.host_bytes,pool_slots),profile_serving_scale(plan.pinned_bytes,pool_slots)));
    s.jobs=calloc((size_t)requests,sizeof(*s.jobs));assert(s.jobs);size_t workspaces=0,recording=0;
    for(int id=0;id<requests;id++)profile_serving_job_init(&s,s.jobs+id,base+id%references,id,references,prefills[id%references],admit,&ceiling,&workspaces,&recording);
    BnGPUMoETrellisStats before,after;assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(model),&before));
    BnTransformerTrellisGroupStats totals={0};uint64_t singles=0,group_calls=0;
    double start=bn_platform_time_ms();
    for(int first=0;first<requests;first+=width) {
        int n=requests-first;if(n>width)n=width;
        BnTokenizerDecodeStream streams[16]={{0}};int prefilled[16]={0},generated[16]={0},done[16]={0},selected[16]={0};
        double admitted[16];int completed=0;
        for(int i=0;i<n;i++) {
            NativeServingJob *j=s.jobs+first+i;ProfileNativeText *t=&j->text;
            assert(!native_text_timing_start(&j->timing));j->timing.start=start;
            BnGenerateNativeTextLease lease={0};assert(!profile_serving_acquire(&s,(uint64_t)j->id,&j->quote,&lease));
            size_t encoded=0;assert(!bn_tokenizer_encoder_encode(t->workspace.encoder,t->workspace.scratch,t->prompt,t->prompt_bytes,
                t->workspace.tokens,t->workspace.token_capacity,&encoded)&&encoded==t->prompt_tokens);
            assert(!bn_tokenizer_decode_stream_init(streams+i,t->workspace.decoder,t->skip));
            j->admitted=1;admitted[i]=bn_platform_time_ms();
        }
        while(completed<n) {
            BnSession *members[16];BnTransformerTrellisForwardResult outputs[16];
            const int *inputs[16];int rows[16],positions[16],owners[16],policies[16],count=0,uniform=1;
            for(int i=0;i<n;i++)if(!done[i]) {
                NativeServingJob *j=s.jobs+first+i;ProfileNativeText *t=&j->text;
                int prompt=prefilled[i]<(int)t->prompt_tokens;
                rows[count]=prompt?(int)t->prompt_tokens-prefilled[i]:1;
                if(rows[count]>j->prefill)rows[count]=j->prefill;
                if(!prompt)selected[i]=profile_group_greedy(j->session->state.logits,t->vocabulary);
                inputs[count]=prompt?t->workspace.tokens+prefilled[i]:selected+i;
                members[count]=j->session;positions[count]=j->session->pos;owners[count]=i;policies[count]=prompt?0:admit;
                if(count&&policies[count]!=policies[0])uniform=0;
                assert(!bn_session_native_history_stage(j->session,model,inputs[count],rows[count]));count++;
            }
            assert(count);
            if(count>1&&uniform) {
                int status;BnTransformerTrellisGroupStats stats={0};
                do {status=bn_transformer_trellis_forward_group(model,members,count,policies[0],outputs,&stats);}while(status==BN_GPU_UPLOAD_BUSY);
                assert(!status);group_calls++;
                totals.calls+=stats.calls;totals.shared_layers+=stats.shared_layers;totals.individual_layers+=stats.individual_layers;
                totals.expert_borrows+=stats.expert_borrows;totals.fused_launches+=stats.fused_launches;totals.folded_launches+=stats.folded_launches;totals.waits+=stats.waits;
                totals.joint_fused_launches+=stats.joint_fused_launches;
                totals.fused_launches_avoided+=stats.fused_launches_avoided;
                totals.individual_fused_launches+=stats.individual_fused_launches;
                totals.lookahead_borrows+=stats.lookahead_borrows;
                totals.lookahead_busy_stops+=stats.lookahead_busy_stops;
                totals.decode_layers+=stats.decode_layers;totals.decode_cohorts+=stats.decode_cohorts;
                totals.decode_submissions+=stats.decode_submissions;totals.decode_expert_reuses+=stats.decode_expert_reuses;
            }else {
                for(int k=0;k<count;k++) {
                    int status;do {status=bn_transformer_trellis_forward_run(model,members[k],policies[k],outputs+k);}while(status==BN_GPU_UPLOAD_BUSY);
                    assert(!status);singles++;
                }
            }
            for(int k=0;k<count;k++) {
                int i=owners[k];NativeServingJob *j=s.jobs+first+i;ProfileNativeText *t=&j->text;
                assert(bn_transformer_trellis_forward_current(model,j->session,outputs+k));
                profile_serving_forward(inputs[k],rows[k],positions[k],outputs[k].host_logits,t->vocabulary,j);
                if(prefilled[i]<(int)t->prompt_tokens)prefilled[i]+=rows[k];
                else {
                    size_t bytes=0;assert(!bn_tokenizer_decode_stream_push(t->workspace.decoder,streams+i,selected+i,1,0,
                        t->workspace.piece,t->workspace.piece_capacity,&bytes));
                    assert(!profile_serving_token(t->workspace.piece,bytes,selected[i],j));generated[i]++;
                }
                if(prefilled[i]==(int)t->prompt_tokens&&generated[i]==t->generated) {
                    size_t bytes=0;assert(!bn_tokenizer_decode_stream_push(t->workspace.decoder,streams+i,NULL,0,1,
                        t->workspace.piece,t->workspace.piece_capacity,&bytes));
                    assert(!profile_serving_token(t->workspace.piece,bytes,-1,j));
                    BnGenerateNativeTextLease lease={j->session,t->workspace};profile_serving_release(&s,lease,0);
                    BnGenerateNativeTextEvent event={BN_NATIVE_TEXT_COMPLETED,(uint64_t)j->id,generated[i],start,admitted[i],
                        j->timing.first_token,j->timing.first_byte,bn_platform_time_ms()};
                    profile_serving_event(&event,&s);done[i]=1;completed++;
                }
            }
        }
    }
    double end=bn_platform_time_ms();assert(end>start&&!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(model),&after)&&!after.live_leases);
    BnTransformerTrellisPoolInfo pool;assert(!bn_transformer_trellis_pool_info(&s.pool,&pool)&&!pool.leased_slots&&!pool.restart_failures);
    uint64_t tokens=0;
    for(int id=0;id<requests;id++) {
        NativeServingJob *j=s.jobs+id;ProfileNativeText *t=&j->text;assert(j->ended);
        profile_text_latency_verify(t,&j->timing,j->terminal.generated,j->end_position);
        if(!timed)profile_text_verify(t,j->terminal.generated,j->end_position,id);
        tokens+=(uint64_t)j->terminal.generated;
        printf("{\"kind\":\"native_forward_group_request\",\"id\":%d,\"position\":%d,\"forwards\":%zu,\"generated_ids\":[",
            id,j->end_position,j->timing.frame_count);
        for(int i=0;i<j->timing.generated;i++)printf("%s%d",i?",":"",j->timing.tokens[i].id);
        printf("],\"output_hex\":\"");for(size_t i=0;i<j->timing.byte_count;i++)printf("%02x",j->timing.bytes[i]);puts("\"}");
    }
    printf("{\"kind\":\"native_forward_group_summary\",\"timing_valid\":%s,\"cohort_width\":%d,\"completed\":%d,"
        "\"window_ms\":%.6f,\"tokens\":%llu,\"group_calls\":%llu,\"single_forwards\":%llu,\"attempts\":%llu,"
        "\"shared_layers\":%llu,\"individual_layers\":%llu,\"shared_expert_borrows\":%llu,\"fused_launches\":%llu,\"folded_launches\":%llu,\"waits\":%llu,"
        "\"joint_fused_launches\":%llu,\"fused_launches_avoided\":%llu,\"individual_fused_launches\":%llu,"
        "\"lookahead_borrows\":%llu,\"lookahead_busy_stops\":%llu,"
        "\"decode_layers\":%llu,\"decode_cohorts\":%llu,\"private_decode_submissions\":%llu,\"reused_decode_bindings\":%llu,"
        "\"created\":%llu,\"reused\":%llu,\"discarded\":%llu,\"retained_device\":%zu,\"peak_device\":%zu,"
        "\"expert_upload_bytes\":%zu,\"expert_upload_ms\":%.6f,\"ram_miss_bytes\":%zu,\"source_read_bytes\":%zu,\"source_read_ms\":%.6f}\n",
        timed?"true":"false",width,requests,end-start,(unsigned long long)tokens,(unsigned long long)group_calls,
        (unsigned long long)singles,(unsigned long long)totals.calls,(unsigned long long)totals.shared_layers,
        (unsigned long long)totals.individual_layers,(unsigned long long)totals.expert_borrows,(unsigned long long)totals.fused_launches,
        (unsigned long long)totals.folded_launches,(unsigned long long)totals.waits,
        (unsigned long long)totals.joint_fused_launches,(unsigned long long)totals.fused_launches_avoided,
        (unsigned long long)totals.individual_fused_launches,
        (unsigned long long)totals.lookahead_borrows,(unsigned long long)totals.lookahead_busy_stops,
        (unsigned long long)totals.decode_layers,(unsigned long long)totals.decode_cohorts,
        (unsigned long long)totals.decode_submissions,(unsigned long long)totals.decode_expert_reuses,
        (unsigned long long)pool.created,
        (unsigned long long)pool.reused,(unsigned long long)pool.discarded,pool.reserved_device,pool.peak_device,
        after.upload_bytes-before.upload_bytes,after.upload_time_ms-before.upload_time_ms,
        after.host.gpu_host_cache_miss_bytes-before.host.gpu_host_cache_miss_bytes,after.host.io_bytes-before.host.io_bytes,
        after.host.io_time_ms-before.host.io_time_ms);
    bn_transformer_trellis_pool_free(&s.pool);
    for(int id=0;id<requests;id++) {
        NativeServingJob *j=s.jobs+id;ProfileNativeText *t=&j->text;
        bn_tokenizer_encode_scratch_free(&t->scratch);bn_json_free(&t->reference);free(t->workspace.tokens);free(t->workspace.piece);free(t->joined);
        free(j->timing.frames);free(j->timing.tokens);free(j->timing.inputs);
    }
    free(s.jobs);
}
