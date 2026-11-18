#include "main_gguf_batch_options.h"
#ifdef BN_ENABLE_CUDA
#include "main_bf16_policy.h"
#include "generate_batch.h"
#include "main_serve.h"
typedef struct {int *prompt,n,position;int32_t *output;BnSession *session;BnTokenizerDecodeStream stream;size_t frames;} GGUFBatchJob;
typedef struct {
    BnModel model;BnGGUFFile *file;BnGPUBackend *gpu;BnTokenizer tokenizer;BnTokenizerDecoder decoder;
    BnTokenizerGGUFScratch encoder;ServeRuntime *serve;
    NativeCLIBatchInput input;GGUFBatchOptions options;
    BnGenerateBatch scheduler;BnBackendSession *coordinator;
    BnTransformerBF16Policy policy;GGUFBatchJob jobs[64];
    unsigned char *piece;size_t piece_capacity;int32_t *outputs;FILE *trace;
    int dense,failed,chunk;uint64_t packed_calls,fallback_rows,created,released,shared_calls;
    BnTransformerBF16SharedInfo shared;
    BnGenerateBatchMemoryLedger memory;
    BnGenerateBatchMemoryQuote request_quote;
    BnTransformerBF16RequestMemory backend_quote;
    struct {BnSession *session;BnGenerateBatchMemoryTicket ticket;int leased,capacity;} memory_slots[16];
    size_t frontend_host,model_host_budget,model_device_budget;uint64_t reused;
    BnBackendBF16MemoryPlan model_quote;
    BnTransformerBF16RequestMemory coordinator_quote;
    size_t kv_reserved,kv_peak,kv_limit;uint64_t evicted,resized,discarded;
} GGUFBatch;
#include "main_gguf_memory.h"
static double gguf_batch_clock(void *context){(void)context;return bn_platform_time_ms();}
static int gguf_batch_acquire(void *context,const BnGenerateBatchAdmission *request,BnGenerateBatchLease *out) {
    GGUFBatch *b=context;if(request->id>=(uint64_t)b->input.count)return -1;
    GGUFBatchJob *j=b->jobs+request->id;BnSession *s=NULL;int status;
    status=gguf_memory_acquire(b,request->required_capacity,&s);
    if(status)return status;
    j->session=s;j->position=0;
    *out=(BnGenerateBatchLease){s,s->state.logits,(size_t)b->model.config.vocab_size};return 0;
}
static void gguf_batch_release(void *context,BnGenerateBatchLease lease,int discard) {
    GGUFBatch *b=context;BnSession *s=lease.handle;
    if(gguf_memory_release(b,s,discard))b->failed=1;
    b->released++;
    for(int i=0;i<b->input.count;i++)if(b->jobs[i].session==s)b->jobs[i].session=NULL;
}
static int gguf_batch_trace(GGUFBatch *b,const BnGenerateBatchExecutionRow *r) {
    if(!b->trace)return 0;
    GGUFBatchJob *j=b->jobs+r->id;char name[128],path[4096];
    for(int kind=0;kind<2;kind++) {
        if(serve_trace_name(b->serve,r->id,j->frames,kind,name,sizeof(name))||
           native_cli_path(path,sizeof(path),b->options.trace,name)||native_cli_dump(path,kind?(const void*)r->lease.logits:(const void*)r->tokens,
           kind?r->lease.logits_count:(size_t)r->n_tokens,kind))return -1;
    }
    uint64_t id=b->serve?b->serve->requests[r->id].id:r->id;
    if(fprintf(b->trace,"{\"kind\":\"forward\",\"request\":%llu,\"index\":%zu,\"position\":%d,\"rows\":%d,\"decode_role\":%d,\"vocabulary\":%zu",
       (unsigned long long)id,j->frames,r->position,r->n_tokens,r->decode_role,r->lease.logits_count)<0||
       (b->serve&&fprintf(b->trace,",\"generation\":%llu,\"slot\":%llu",(unsigned long long)b->serve->requests[r->id].generation,(unsigned long long)r->id)<0)||
       fputs("}\n",b->trace)==EOF)return -1;
    j->frames++;return 0;
}
static int gguf_batch_execute(void *context,const BnGenerateBatchExecutionRow *rows,int count) {
    GGUFBatch *b=context;if(b->failed||count<1||count>16)return -1;
    for(int i=0;i<count;i++)if(rows[i].id>=(uint64_t)b->input.count||
        rows[i].lease.handle!=b->jobs[rows[i].id].session||rows[i].position!=b->jobs[rows[i].id].position)return -1;
    if(b->dense) {
        BnTransformerBF16StepRow native[16];
        for(int i=0;i<count;i++)native[i]=(BnTransformerBF16StepRow){
            {bn_session_backend(rows[i].lease.handle),rows[i].tokens,rows[i].n_tokens,rows[i].position,rows[i].lease.logits,rows[i].lease.logits_count},
            rows[i].decode_role?BN_TRANSFORMER_BF16_STEP_DECODE:BN_TRANSFORMER_BF16_STEP_PREFILL};
        if(bn_transformer_bf16_batch_step(&b->model,b->gpu,b->coordinator,native,count,NULL,NULL))return -1;
        b->packed_calls++;
    }else if(count>1&&(bn_backend_layout_bf16_glm_model(bn_model_backend(&b->model))||
        bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(&b->model)))) {
        BnTransformerBF16SessionRow cohort[16];BnTransformerBF16SharedInfo info;
        for(int i=0;i<count;i++)cohort[i]=(BnTransformerBF16SessionRow){rows[i].lease.handle,
            (const int*)rows[i].tokens,rows[i].n_tokens,rows[i].position,rows[i].is_prefill};
        int status=bn_backend_layout_bf16_glm_model(bn_model_backend(&b->model))?
            bn_transformer_bf16_glm_forward_group_session(&b->model,cohort,count,&info):
            bn_transformer_bf16_flash_forward_group_session(&b->model,cohort,count,&info);
        if(status)return -1;
        b->shared_calls++;b->shared.shared_layers+=info.shared_layers;b->shared.shared_banks+=info.shared_banks;
        b->shared.shared_consumers+=info.shared_consumers;b->shared.shared_upload_bytes_avoided+=info.shared_upload_bytes_avoided;
        b->shared.busy_fences+=info.busy_fences;
        b->shared.weight_upload_bytes+=info.weight_upload_bytes;
        b->shared.cache_hits+=info.cache_hits;b->shared.cache_misses+=info.cache_misses;
        b->shared.cache_miss_bytes+=info.cache_miss_bytes;b->shared.cache_evictions+=info.cache_evictions;
        b->shared.cache_bypasses+=info.cache_bypasses;
        b->shared.cache_hit_bytes+=info.cache_hit_bytes;b->shared.cache_fill_copy_bytes+=info.cache_fill_copy_bytes;
        if(info.cache_reserved_device_bytes>b->shared.cache_reserved_device_bytes)b->shared.cache_reserved_device_bytes=info.cache_reserved_device_bytes;
        b->shared.joint_kernel_launches+=info.joint_kernel_launches;
        b->shared.joint_projection_launches+=info.joint_projection_launches;
        b->shared.joint_launches_avoided+=info.joint_launches_avoided;
    }else for(int i=0;i<count;i++) {
        if(!bn_transformer_bf16_forward_session(&b->model,rows[i].lease.handle,(const int*)rows[i].tokens,
           rows[i].n_tokens,rows[i].position,rows[i].is_prefill))return -1;
        b->fallback_rows++;
    }
    for(int i=0;i<count;i++) {
        b->jobs[rows[i].id].position=rows[i].position+rows[i].n_tokens;
        if(gguf_batch_trace(b,rows+i))return -1;
    }
    return 0;
}
static void gguf_batch_event(void *context,const BnGenerateBatchEvent *e) {
    GGUFBatch *b=context;if(b->failed)return;
    if(e->id>=(uint64_t)b->input.count){b->failed=1;return;}
    uint64_t wire=e->id;if(b->serve&&serve_runtime_event(b->serve,e,&wire)){b->failed=1;return;}
    GGUFBatchJob *j=b->jobs+e->id;unsigned long long id=(unsigned long long)wire;size_t bytes=0;
    if(e->kind==BN_GENERATE_BATCH_ADMITTED) {
        if(printf("{\"kind\":\"admitted\",\"request\":%llu,\"prompt_tokens\":%d",id,j->n)<0||serve_event_end(b->serve,e))b->failed=1;
    }else if(e->kind==BN_GENERATE_BATCH_TOKEN) {
        if(bn_tokenizer_decode_stream_push(&b->decoder,&j->stream,&e->token,1,0,b->piece,b->piece_capacity,&bytes)||
           printf("{\"kind\":\"token\",\"request\":%llu,\"index\":%d,\"token\":%d,\"text\":",id,e->output_index,e->token)<0||
           native_cli_batch_json_text(stdout,b->piece,bytes)||serve_event_end(b->serve,e))b->failed=1;
        if(e->complete&&!b->failed) {
            if(bn_tokenizer_decode_stream_push(&b->decoder,&j->stream,NULL,0,1,b->piece,b->piece_capacity,&bytes)||
               printf("{\"kind\":\"completed\",\"request\":%llu,\"generated\":%d,\"position\":%d,\"text\":",id,e->output_index+1,j->position)<0||
               native_cli_batch_json_text(stdout,b->piece,bytes)||serve_event_end(b->serve,e))b->failed=1;
        }
    }else {
        if(printf("{\"kind\":\"%s\",\"request\":%llu",e->kind==BN_GENERATE_BATCH_CANCELLED?"cancelled":"failed",id)<0)b->failed=1;
        if(b->serve&&!b->failed&&(bn_tokenizer_decode_stream_push(&b->decoder,&j->stream,NULL,0,1,b->piece,b->piece_capacity,&bytes)||
           fputs(",\"text\":",stdout)==EOF||native_cli_batch_json_text(stdout,b->piece,bytes)))b->failed=1;
        if(!b->failed&&serve_event_end(b->serve,e))b->failed=1;
    }
    if(fflush(stdout))b->failed=1;
}
static int gguf_batch_prepare_serving(void *context,int slot,const ServeCommand *command,ServePrepared *out) {
    GGUFBatch *b=context;GGUFBatchJob *j=b->jobs+slot;
    if(j->session)return -1;
    int n=bn_tokenizer_gguf_encode(&b->tokenizer,&b->encoder,command->prompt,b->options.bos&&b->tokenizer.add_bos,
        j->prompt,b->model.config.seq_len+1);
    if(n<1||native_cli_context((size_t)n,command->max_outputs,1,b->model.config.seq_len))return 1;
    int capacity=gguf_memory_capacity(b,n+command->max_outputs-1);
    BnGenerateBatchMemoryQuote quote;BnTransformerBF16RequestMemory backend;
    if(capacity<1||gguf_memory_quote(b,capacity,&quote,&backend)||
       (b->dense&&bn_transformer_bf16_dense_kv_bytes(&b->model,capacity)>b->kv_limit)||
       bn_generate_batch_memory_check_shared(&b->memory,quote))return 1;
    bn_tokenizer_decode_stream_free(&j->stream);
    if(bn_tokenizer_decode_stream_init(&j->stream,&b->decoder,0))return -1;
    j->n=n;j->position=0;j->frames=0;*out=(ServePrepared){(const int32_t*)j->prompt,n,j->output};return 0;
}
static void gguf_batch_free(GGUFBatch *b) {
    bn_generate_batch_free(&b->scheduler);
    if(b->serve){serve_runtime_free(b->serve);free(b->serve);}
    if(b->coordinator)bn_backend_session_free(b->coordinator);
    gguf_memory_free(b);
    for(int i=0;i<64;i++)free(b->jobs[i].prompt);
    if(b->trace&&fclose(b->trace))b->failed=1;
    bn_tokenizer_decoder_free(&b->decoder);bn_tokenizer_gguf_scratch_free(&b->encoder);bn_tokenizer_free(&b->tokenizer);
    bn_model_moe_io_shutdown(&b->model);
    bn_moe_cache_free(bn_model_moe_cache(&b->model));bn_model_set_moe_cache(&b->model,NULL);
    bn_model_free(&b->model);if(b->gpu)bn_gpu_cuda_destroy(b->gpu);if(b->file)bn_gguf_free(b->file);
    bn_json_free(&b->input.json);free(b->piece);free(b->outputs);
}
static int gguf_batch_run(const GGUFBatchOptions *options) {
    GGUFBatch b={0};b.options=*options;const char *stage="input";int status=1;unsigned char *raw=NULL;size_t raw_bytes;
#define GB_REQUIRE(X,S) do{stage=S;if(!(X))goto done;}while(0)
    if(options->serve)b.input.count=BN_GENERATE_BATCH_QUEUE_CAPACITY;
    else {
    GB_REQUIRE(!strcmp(options->file,"-")?!native_cli_read(stdin,BN_JSON_MAX_BYTES,&raw,&raw_bytes):
        !native_cli_read_path(options->file,BN_JSON_MAX_BYTES,&raw,&raw_bytes),"batch input read");
    GB_REQUIRE(!native_cli_batch_input_init(&b.input,raw,raw_bytes),"batch prompt array");free(raw);raw=NULL;
    }
    b.file=bn_gguf_open_file(options->model);GB_REQUIRE(b.file,"GGUF model");
    BnModelBF16LayoutKind expected=!strcmp(options->profile,"dense-bf16")?BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM:
        !strcmp(options->profile,"flash-bf16")?BN_MODEL_BF16_LAYOUT_FLASH_HC:BN_MODEL_BF16_LAYOUT_GLM_MHC;
    GB_REQUIRE(bn_model_gguf_bf16_layout(b.file)==expected,"profile/model topology mismatch");
    b.dense=expected==BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM;
    GB_REQUIRE(!(expected==BN_MODEL_BF16_LAYOUT_GLM_MHC?bn_model_load_topology(&b.model,b.file,options->maxseq):
        bn_model_load_with_cpu_preparation(&b.model,b.file,options->maxseq,0,0,0)),"model load");
    GB_REQUIRE(!bn_tokenizer_init(&b.tokenizer,b.file)&&b.tokenizer.vocab_size==b.model.config.vocab_size,"GGUF tokenizer/vocabulary");
    size_t decoder_bytes,encoder_bytes=0,codec_bytes;
    GB_REQUIRE(!options->serve||!bn_tokenizer_gguf_scratch_plan(&b.tokenizer,SERVE_RECORD_BYTES,&encoder_bytes),"serving tokenizer workspace");
    GB_REQUIRE(!bn_tokenizer_decoder_plan(b.tokenizer.vocab_size,&decoder_bytes)&&
        !native_cli_piece_bound((size_t)b.tokenizer.max_token_length,&b.piece_capacity)&&
        (codec_bytes=decoder_bytes,!native_cli_add(&codec_bytes,encoder_bytes))&&!gguf_memory_frontend(&b,codec_bytes)&&
        !bn_tokenizer_decoder_init_gguf(&b.decoder,&b.tokenizer,b.file,decoder_bytes),"ByteLevel decoder/serving host preflight");
    GB_REQUIRE(!options->serve||!bn_tokenizer_gguf_scratch_init(&b.encoder,&b.tokenizer,SERVE_RECORD_BYTES,encoder_bytes),"serving tokenizer allocation");
    b.piece=malloc(b.piece_capacity);GB_REQUIRE(b.piece,"output buffer");
    GB_REQUIRE(b.model.config.seq_len>0&&b.model.config.seq_len<INT_MAX,"model context");
    if(options->serve&&b.options.tokens>b.model.config.seq_len)b.options.tokens=b.model.config.seq_len;
    for(int i=0;i<b.input.count;i++) {
        GGUFBatchJob *j=b.jobs+i;size_t bytes;
        GB_REQUIRE(!native_cli_mul((size_t)b.model.config.seq_len+1,sizeof(int),&bytes)&&(j->prompt=malloc(bytes)),"prompt buffer");
        if(!options->serve)j->n=bn_tokenizer_encode(&b.tokenizer,b.input.prompts[i],options->bos&&b.tokenizer.add_bos,j->prompt,b.model.config.seq_len+1);
        GB_REQUIRE((options->serve||(j->n>0&&!native_cli_context((size_t)j->n,options->tokens,1,b.model.config.seq_len)))&&
            !bn_tokenizer_decode_stream_init(&j->stream,&b.decoder,0),"prompt encoding/context");
    }
    size_t outputs;
    GB_REQUIRE(!native_cli_mul((size_t)b.input.count,(size_t)b.options.tokens,&outputs)&&
        !native_cli_mul(outputs,sizeof(int32_t),&outputs)&&(b.outputs=calloc(1,outputs)),"output arrays");
    for(int i=0;i<b.input.count;i++)b.jobs[i].output=b.outputs+(size_t)i*b.options.tokens;
    b.chunk=options->prefill;if(b.chunk>options->budget)b.chunk=options->budget;
    if(b.chunk>b.model.config.seq_len)b.chunk=b.model.config.seq_len;
    b.policy.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32;b.policy.attention_split_policy=BN_GPU_ATTENTION_SPLIT_BATCH;
    if(options->policy) {
        GB_REQUIRE(!native_cli_read_path(options->policy,16384,&raw,&raw_bytes)&&!gguf_batch_policy_parse(raw,raw_bytes,&b.policy),"arithmetic policy");free(raw);raw=NULL;
    }
    GB_REQUIRE((b.gpu=bn_gpu_cuda_create()),"CUDA device");
    GB_REQUIRE(!gguf_memory_init(&b),"aggregate serving payload budget");
    if(bn_model_uses_moe(&b.model)) {
        const BnMappedFile *mf=bn_gguf_primary_file(b.file);
        if(b.file->n_shards>1)GB_REQUIRE(b.file->owned_maps&&!bn_model_set_moe_shard_files(&b.model,b.file->owned_maps,b.file->n_shards),"expert shard readers");
        else {GB_REQUIRE(mf&&mf->fd>=0,"expert file reader");bn_model_set_moe_fd(&b.model,mf->fd);}
        if(options->ram_mb) {
            BnMoEProjectionBufferLayout layout=bn_moe_model_projection_buffer_layout(&b.model);size_t budget;
            GB_REQUIRE(!native_cli_mul((size_t)options->ram_mb,1048576,&budget),"RAM cache bound");
            BnMoECache *cache=bn_moe_cache_create(budget,layout.gate_bytes,layout.up_bytes,layout.down_bytes);
            GB_REQUIRE(cache,"RAM cache");bn_model_set_moe_cache(&b.model,cache);
        }
    }
    bn_model_set_thread_pool(&b.model,bn_tp_create(options->threads),1);
    size_t gpu_cache;
    GB_REQUIRE(!native_cli_mul((size_t)options->gpu_mb,1048576,&gpu_cache),"GPU cache bound");
    if(b.dense)GB_REQUIRE(!bn_backend_layout_prepare_bf16_with_limits(bn_model_backend(&b.model),b.gpu,&b.model.config,&b.model.weights,options->budget,BN_BACKEND_BF16_MATH_MKL_VML_HA,b.model_device_budget,b.model_host_budget),"dense resident model formats");
    else if(expected==BN_MODEL_BF16_LAYOUT_FLASH_HC)GB_REQUIRE(!bn_backend_layout_prepare_bf16_flash_model_with_cache(bn_model_backend(&b.model),b.gpu,&b.model.config,&b.model.weights,b.model_device_budget,b.model_host_budget,gpu_cache),"Flash model formats");
    else GB_REQUIRE(!bn_model_prepare_bf16_topology_with_cache(&b.model,b.gpu,b.model_device_budget,b.model_host_budget,gpu_cache),"GLM recurrent model, cache and payload budgets");
    if(b.dense) {
        GB_REQUIRE(!gguf_memory_coordinator(&b),"dense coordinator payload quote/budget");
        b.coordinator=bn_backend_session_create();GB_REQUIRE(b.coordinator,"coordinator");
        GB_REQUIRE(!bn_transformer_bf16_batch_prepare_packed_with_policy(&b.model,b.gpu,b.coordinator,options->live,options->budget,&b.policy),"dense packed execution");
    }
    GB_REQUIRE(!gguf_memory_ready(&b),"request quote/aggregate serving budget");
    if(options->trace) {
        char path[4096];GB_REQUIRE(!native_cli_path(path,sizeof(path),options->trace,"events.jsonl")&&(b.trace=fopen(path,"wbx")),"exclusive trace directory");
    }
    BnGenerateBatchLimits limits={options->live,options->live,b.chunk,options->budget,b.model.config.seq_len,options->mixed_budget};
    BnGenerateBatchBackend backend={&b,NULL,gguf_batch_execute,gguf_batch_release,gguf_batch_clock};
    GB_REQUIRE(!bn_generate_batch_init_request_aware(&b.scheduler,&limits,b.model.config.vocab_size,&backend,gguf_batch_acquire,gguf_batch_event,&b),"scheduler");
    if(options->serve) {
        b.serve=calloc(1,sizeof(*b.serve));
        GB_REQUIRE(b.serve&&!serve_runtime_init(b.serve,&b.scheduler,gguf_batch_prepare_serving,&b,gguf_batch_clock,&b,&b.failed,
            SERVE_RECORD_BYTES,b.options.tokens)&&!serve_runtime_run(b.serve,STDIN_FILENO),"external serving");
    }else {
    for(int i=0;i<b.input.count;i++)GB_REQUIRE(!bn_generate_batch_enqueue(&b.scheduler,(uint64_t)i,(const int32_t*)b.jobs[i].prompt,b.jobs[i].n,b.jobs[i].output,options->tokens),"enqueue");
    BnGenerateBatchInfo info;
    for(;;) {
        GB_REQUIRE(!bn_generate_batch_info(&b.scheduler,&info),"scheduler info");if(!info.live&&!info.queued)break;
        GB_REQUIRE(bn_generate_batch_step(&b.scheduler)==1&&!b.failed,"generation/output");
    }
    }
    BnGenerateBatchInfo info;
    GB_REQUIRE(!bn_generate_batch_info(&b.scheduler,&info)&&!info.failed&&
        (options->serve?info.completed+info.cancelled==info.submitted:info.completed==(uint64_t)b.input.count),"completion");
    fprintf(stderr,"cuda_batch_result profile=%s completed=%llu steps=%llu packed_calls=%llu fallback_rows=%llu released=%llu peak_live=%d\n",options->profile,
        (unsigned long long)info.completed,(unsigned long long)info.steps,(unsigned long long)b.packed_calls,(unsigned long long)b.fallback_rows,(unsigned long long)b.released,info.peak_live);
    if(b.shared_calls)fprintf(stderr,"cuda_batch_shared calls=%llu coordinated_moe_layers=%llu uploads=%llu consumers=%llu upload_bytes_avoided=%zu busy_fences=%zu joint_ffn_kernels=1 kernel_launches=%llu projection_launches=%llu launches_avoided=%llu\n",
        (unsigned long long)b.shared_calls,(unsigned long long)b.shared.shared_layers,(unsigned long long)b.shared.shared_banks,
        (unsigned long long)b.shared.shared_consumers,b.shared.shared_upload_bytes_avoided,b.shared.busy_fences,
        (unsigned long long)b.shared.joint_kernel_launches,(unsigned long long)b.shared.joint_projection_launches,
        (unsigned long long)b.shared.joint_launches_avoided);
    if(b.shared_calls)fprintf(stderr,"cuda_batch_shared_cache scope=cohort_unique_experts weight_upload_bytes=%zu hits=%zu misses=%zu hit_bytes=%zu fill_copy_bytes=%zu reserved_device_bytes=%zu miss_bytes=%zu evictions=%zu bypasses=%zu\n",
        b.shared.weight_upload_bytes,b.shared.cache_hits,b.shared.cache_misses,b.shared.cache_hit_bytes,
        b.shared.cache_fill_copy_bytes,b.shared.cache_reserved_device_bytes,
        b.shared.cache_miss_bytes,b.shared.cache_evictions,b.shared.cache_bypasses);
    fprintf(stderr,"cuda_batch_requests pooling=budgeted created=%llu reused=%llu request_device_cap_mb=%d request_host_cap_mb=%d\n",
        (unsigned long long)b.created,(unsigned long long)b.reused,options->device_mb,options->host_mb);
    if(b.dense)fprintf(stderr,"cuda_batch_pool scope=KV_payload_subset created=%llu reused=%llu reserved_kv=%zu peak_kv=%zu budget_kv=%zu evicted=%llu resized=%llu discarded=%llu\n",
        (unsigned long long)b.created,(unsigned long long)b.reused,b.kv_reserved,b.kv_peak,b.kv_limit,
        (unsigned long long)b.evicted,(unsigned long long)b.resized,(unsigned long long)b.discarded);
    gguf_memory_report(&b);
    status=0;
done:
    if(status)fprintf(stderr,"CUDA batch failed: %s\n",stage);
    free(raw);gguf_batch_free(&b);
    if(!status&&b.failed){fputs("CUDA batch failed: trace/output cleanup\n",stderr);status=1;}
    return status;
#undef GB_REQUIRE
}
#endif
static int gguf_batch_main(int argc,char **argv) {
    GGUFBatchOptions options;int parsed=gguf_batch_parse(argc,argv,&options);
    if(parsed){gguf_batch_usage(parsed>0?stdout:stderr,argv[0]);return parsed>0?0:2;}
#ifdef BN_ENABLE_CUDA
    if(!bn_backend_layout_bf16_math_available(BN_BACKEND_BF16_MATH_MKL_VML_HA)){fputs("CUDA batch requires MKL VML.\n",stderr);return 1;}
    return gguf_batch_run(&options);
#else
    fputs("CUDA batch requires BN_ENABLE_CUDA=1.\n",stderr);return 1;
#endif
}
