#include "main_native_options.h"
#ifdef BN_ENABLE_CUDA
#include "main_native_io.h"
#include "transformer_internal.h"
#include "safetensors.h"
#include "quant.h"
#include "main_native_batch_input.h"
#include "main_native_memory.h"
#include "main_serve.h"
typedef struct NativeCLIBatch NativeCLIBatch;
typedef struct NativeCLIResources {
    BnModel model;BnSafetensorsIndex index;BnJSON tokenizer;
    BnTokenizerEncoder encoder;BnTokenizerDecoder decoder;BnTokenizerEncodeScratch scratch;
    BnTokenizerVocab vocabulary;BnTokenizerMap map;
    BnGPUBackend *gpu;BnSession *session;
    unsigned char *text,*piece;int *tokens;
    size_t text_bytes,token_capacity,count,piece_bytes,ep_bytes,dp_bytes,scratch_bytes,vocab_bytes,map_bytes;
    NativeCLIBatch *batch;
    BnGenerateBatchMemoryLedger memory;BnGenerateBatchMemoryTicket session_ticket;int session_reserved;
} NativeCLIResources;
typedef struct {
    BnBackendNativeCoreStats core;BnBackendNativeParameterStats parameters;BnBackendNativeRouterStats routers;
    BnTransformerTrellisOptions f;BnTransformerTrellisKVOptions kv;
    BnTransformerTrellisSWAOptions swa;BnTransformerTrellisIndexOptions index;
    BnTransformerTrellisAttentionOptions attention;BnTransformerTrellisProducerOptions producer;
    BnTransformerTrellisRowsOptions rows;BnTransformerTrellisEngramOptions engram;
    BnTransformerTrellisHCOptions hc;BnTransformerTrellisForwardOptions forward;
    size_t history,entry,ram,decode,temporary,host,device;
    BnGenerateBatchMemory shared;BnTransformerTrellisRequestPlan request;
} NativeCLIPlan;
static int native_cli_model_json(BnJSON *doc,const char *path,size_t nodes) {
    unsigned char *data=NULL;size_t n;
    if(native_cli_read_path(path,BN_JSON_MAX_BYTES,&data,&n))return -1;
    BnJSONLimits limits={BN_JSON_MAX_BYTES,nodes};int status=bn_json_init_limits(doc,data,n,&limits);free(data);return status;
}
static int native_cli_tokenizers(NativeCLIResources *r,const char *path,size_t host_limit) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(&r->model);
    BnTokenizerEncoderPlan ep;BnTokenizerMapSpec spec;BnTokenizerMapWorkspace mp;
    if(!a||native_cli_model_json(&r->tokenizer,path,1000000)||
        bn_tokenizer_encoder_plan(&r->tokenizer,a->vocab,&ep)||
        bn_tokenizer_decoder_plan(a->vocab,&r->dp_bytes)||
        bn_tokenizer_vocab_plan(a->vocab,&r->vocab_bytes))return -1;
    r->ep_bytes=ep.host_bytes;size_t early=0,token_bytes,map_copy;
    if(native_cli_mul(r->token_capacity,sizeof(int),&token_bytes)||
        native_cli_mul((size_t)a->vocab,sizeof(uint32_t),&map_copy))return -1;
    const size_t base[]={r->ep_bytes,r->dp_bytes,r->vocab_bytes,r->text_bytes,r->scratch_bytes,token_bytes,map_copy};
    for(size_t i=0;i<sizeof(base)/sizeof(base[0]);i++)if(native_cli_add(&early,base[i]))return -1;
    if((host_limit&&early>host_limit)||
        bn_tokenizer_vocab_init_json(&r->vocabulary,&r->tokenizer,a->vocab,r->vocab_bytes)||
        bn_model_native_engram_map_spec(&r->model,&spec)||
        bn_tokenizer_map_plan(&r->vocabulary,&spec,&mp))return -1;
    size_t longest=0;
    for(int i=0;i<a->vocab;i++) {
        const char *raw=bn_tokenizer_vocab_raw(&r->vocabulary,i);if(!raw)return -1;
        size_t n=strlen(raw);if(n>longest)longest=n;
    }
    r->ep_bytes=ep.host_bytes;r->map_bytes=mp.host_bytes;
    if(native_cli_piece_bound(longest,&r->piece_bytes)||native_cli_add(&early,mp.host_bytes)||
        native_cli_add(&early,r->piece_bytes)||(host_limit&&early>host_limit)||
        bn_tokenizer_encoder_init_json(&r->encoder,&r->tokenizer,a->vocab,ep.host_bytes)||
        bn_tokenizer_decoder_init_json(&r->decoder,&r->tokenizer,a->vocab,r->dp_bytes)||
        bn_tokenizer_map_init(&r->map,&r->vocabulary,&spec,mp.host_bytes)||
        bn_model_prepare_native_token_map(&r->model,bn_tokenizer_map_ids(&r->map),
            (size_t)spec.vocab_size,(size_t)spec.vocab_size*sizeof(uint32_t)))return -1;
    bn_tokenizer_map_free(&r->map);bn_tokenizer_vocab_free(&r->vocabulary);return 0;
}
static int native_cli_shape(NativeCLIPlan *p,NativeCLIResources *r,const NativeCLIOptions *o) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(&r->model);BnModelNativeHead head;int row_heads=0;
    if(!a||a->experts.format!=BN_QUANT_FORMAT_EXL3_MUL1_3||a->engram_format!=BN_QUANT_FORMAT_INT4_FP16S32||
        bn_model_native_head(&r->model,&head)||head.profile!=BN_NATIVE_HEAD_F16_LINEAR_F16)return -1;
    for(int l=0;l<a->experts.layers;l++)if(a->layers[l].engram_rows) {
        BnModelNativeEngramMath math;
        if(bn_model_native_engram_math(&r->model,l,&math)||math.profile!=BN_NATIVE_ENGRAM_MATH_BF16_LOOKUP_F16||math.heads<1)return -1;
        if(math.heads>row_heads)row_heads=math.heads;
    }
    int batch=r->count<(size_t)o->prefill?(int)r->count:o->prefill;
    int cold;
    if(!row_heads||batch<1||batch>INT_MAX/row_heads||o->maxseq>a->context||
        native_cli_cold_experts(batch,a->experts.active_experts,a->experts.experts,o->window,&cold)||
        o->temporary_experts<cold)return -1;
    const BnMoEExpertReader *reader=bn_model_trellis_expert_reader(&r->model);
    if(!reader)return -1;
    for(int i=0;i<3;i++)if(native_cli_add(&p->entry,reader->bytes[i]))return -1;
    if(!p->entry||native_cli_mul((size_t)o->ram_mb,1024*1024,&p->ram)||
        native_cli_mul((size_t)o->gpu_mb,1024*1024,&p->decode)||
        native_cli_mul((size_t)o->temporary_experts,p->entry,&p->temporary)||
        (p->ram&&p->ram<p->entry)||(p->decode&&p->decode<p->entry))return -1;
    p->f=(BnTransformerTrellisOptions){batch,o->window,0,0,0};p->kv=(BnTransformerTrellisKVOptions){o->maxseq,0,0};
    p->rows=(BnTransformerTrellisRowsOptions){batch*row_heads,2,0,0,0};
    if(bn_backend_layout_native_core_plan(&r->index,a,&p->core)||
        bn_backend_layout_native_parameter_plan(&r->index,a,&p->parameters)||
        bn_backend_layout_native_router_plan(&r->index,a,&p->routers))return -1;
    return 0;
}
static int native_cli_plan(NativeCLIPlan *p,NativeCLIResources *r) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(&r->model);
    BnTransformerTrellisWorkspace f;BnTransformerTrellisKVWorkspace kv;BnTransformerTrellisSWAWorkspace sw;
    BnTransformerTrellisIndexWorkspace ix;BnTransformerTrellisAttentionWorkspace at;BnTransformerTrellisProducerWorkspace pr;
    BnTransformerTrellisRowsWorkspace rw;BnTransformerTrellisEngramWorkspace e;BnTransformerTrellisHCWorkspace h;
    BnTransformerTrellisForwardWorkspace fw;
    if(bn_transformer_trellis_session_plan(&r->model,&p->f,&f)||
        bn_transformer_trellis_kv_session_plan(&r->model,&p->kv,&kv)||
        bn_transformer_trellis_swa_session_plan(&r->model,&sw)||
        bn_transformer_trellis_index_session_plan(&r->model,&p->f,&p->kv,&ix)||
        bn_transformer_trellis_attention_session_plan(&r->model,&p->f,&p->kv,&at)||
        bn_transformer_trellis_producer_session_plan(&r->model,&p->f,&p->kv,&pr)||
        bn_transformer_trellis_rows_session_plan(&r->model,&p->rows,&rw)||
        bn_transformer_trellis_engram_session_plan(&r->model,&p->f,&p->rows,&e)||
        bn_transformer_trellis_hc_session_plan(&r->model,&p->f,&h)||
        bn_transformer_trellis_forward_session_plan(&r->model,&p->f,&p->forward,&fw))return -1;
    p->history=e.history_bytes;p->f.device_budget=f.device_bytes;p->f.host_budget=f.host_bytes;
    p->kv.device_budget=kv.device_bytes;p->kv.host_budget=kv.host_bytes;
    p->swa=(BnTransformerTrellisSWAOptions){sw.device_bytes,sw.host_bytes};
    p->index=(BnTransformerTrellisIndexOptions){ix.device_bytes,ix.host_bytes};
    p->attention=(BnTransformerTrellisAttentionOptions){at.device_bytes,at.host_bytes};
    p->producer=(BnTransformerTrellisProducerOptions){pr.device_bytes,pr.host_bytes};
    p->rows.host_budget=rw.host_bytes;p->rows.pinned_budget=rw.pinned_bytes;p->rows.device_budget=rw.device_bytes;
    p->engram=(BnTransformerTrellisEngramOptions){e.device_bytes,e.host_bytes};p->hc=(BnTransformerTrellisHCOptions){h.device_bytes,h.host_bytes};
    p->forward=(BnTransformerTrellisForwardOptions){0,fw.device_bytes,fw.host_bytes};
    const size_t devices[]={p->routers.device_peak_bytes,p->decode,p->temporary,f.device_bytes,kv.device_bytes,sw.device_bytes,
        ix.device_bytes,at.device_bytes,pr.device_bytes,rw.device_bytes,e.device_bytes,h.device_bytes,fw.device_bytes};
    for(size_t i=0;i<sizeof(devices)/sizeof(devices[0]);i++)if(native_cli_add(&p->device,devices[i]))return -1;
    size_t token_bytes,session_bytes,map_bytes;
    if(native_cli_mul(r->text_bytes,sizeof(int),&token_bytes)||
        native_cli_mul((size_t)a->context+(size_t)a->vocab,sizeof(uint32_t),&session_bytes)||
        native_cli_mul((size_t)a->vocab,sizeof(uint32_t),&map_bytes))return -1;
    const size_t hosts[]={p->routers.host_staging_bytes,p->ram,p->entry,32*1024*1024,r->text_bytes,token_bytes,
        r->ep_bytes,r->dp_bytes,r->scratch_bytes,r->vocab_bytes,r->map_bytes,r->piece_bytes,session_bytes,map_bytes,
        f.host_bytes,kv.host_bytes,sw.host_bytes,ix.host_bytes,at.host_bytes,pr.host_bytes,rw.host_bytes,rw.pinned_bytes,
        e.host_bytes,e.history_bytes,h.host_bytes,fw.host_bytes};
    for(size_t i=0;i<sizeof(hosts)/sizeof(hosts[0]);i++)if(native_cli_add(&p->host,hosts[i]))return -1;
    BnTransformerTrellisRequestConfig config={p->f.max_tokens,p->kv.maxseq,p->f.fused_window,0,0};
    size_t private_host;
    if(bn_transformer_trellis_request_plan(&r->model,&config,&p->request))return -1;
    private_host=p->request.host_bytes;
    if(native_cli_add(&private_host,p->request.pinned_bytes)||private_host<2*SH_ARENA_ALIGN)return -1;
    private_host-=2*SH_ARENA_ALIGN;
    if(p->host<private_host||p->device<p->request.device_bytes)return -1;
    p->shared=(BnGenerateBatchMemory){p->host-private_host,p->device-p->request.device_bytes};
    BnGenerateBatchMemoryQuote quote;
    if(native_cli_memory_quote(&p->request,&quote)||native_cli_add(&p->host,2*SH_ARENA_ALIGN+
       sizeof(BnSession)+sizeof(BnSessionBackendState)+sizeof(SHArena)))return -1;
    return 0;
}
typedef struct {const NativeCLIOptions *options;FILE *events;size_t frames,bytes;int failed;} NativeCLIOutput;
static void native_cli_observe(const int *tokens,int count,int position,const float *logits,int vocabulary,void *user) {
    NativeCLIOutput *out=user;if(!out->events||out->failed)return;
    char name[96],path[4096];
    snprintf(name,sizeof(name),"forward-%zu.i32le",out->frames);
    if(native_cli_path(path,sizeof(path),out->options->trace_directory,name)||native_cli_dump(path,tokens,(size_t)count,0))out->failed=1;
    snprintf(name,sizeof(name),"forward-%zu.f32le",out->frames);
    if(!out->failed&&(native_cli_path(path,sizeof(path),out->options->trace_directory,name)||native_cli_dump(path,logits,(size_t)vocabulary,1)))out->failed=1;
    if(!out->failed&&fprintf(out->events,"{\"kind\":\"forward\",\"index\":%zu,\"position\":%d,\"rows\":%d,\"vocabulary\":%d}\n",out->frames,position,count,vocabulary)<0)out->failed=1;
    out->frames++;
}
static int native_cli_output(const unsigned char *piece,size_t bytes,int token,void *user) {
    NativeCLIOutput *out=user;if(out->failed)return 1;
    if((bytes&&fwrite(piece,1,bytes,stdout)!=bytes)||fflush(stdout)){out->failed=1;return 1;}
    if(native_cli_add(&out->bytes,bytes)){out->failed=1;return 1;}
    if(out->options->token_ids&&token>=0)fprintf(stderr,"token_id=%d\n",token);
    if(out->events&&fprintf(out->events,"{\"kind\":\"token\",\"id\":%d,\"bytes\":%zu}\n",token,bytes)<0)out->failed=1;
    return out->failed;
}
#include "main_native_batch.h"
static int native_cli_run(const NativeCLIOptions *o) {
    NativeCLIResources r={0};NativeCLIPlan p={0};NativeCLIOutput output={o,NULL,0,0,0};
    size_t token_bytes=0,preliminary=0,host_limit=0,device_limit=0;
    char path[4096];const char *stage="model metadata";int status=1;unsigned char *config=NULL;size_t config_bytes=0;
#define NATIVE_REQUIRE(EXPR,STAGE) do{stage=STAGE;if(!(EXPR))goto done;}while(0)
    NATIVE_REQUIRE(!native_cli_mul((size_t)o->host_mb,1024*1024,&host_limit)&&
        !native_cli_mul((size_t)o->device_mb,1024*1024,&device_limit),"payload cap overflow");
    if(native_cli_batch_mode(o)&&!o->host_budget_set&&!o->host_mb)host_limit=(size_t)16<<30;
    NATIVE_REQUIRE(!native_cli_path(path,sizeof(path),o->directory,"model.safetensors.index.json")&&
        !bn_safetensors_index_init(&r.index,path),"model index");
    NATIVE_REQUIRE(!native_cli_path(path,sizeof(path),o->directory,"config.json")&&
        !native_cli_read_path(path,BN_JSON_MAX_BYTES,&config,&config_bytes)&&
        !bn_model_load_trellis_config(&r.model,&r.index,config,config_bytes)&&
        !bn_model_prepare_native_anatomy(&r.model)&&!bn_model_prepare_native_numerics(&r.model)&&
        !bn_model_prepare_native_rows(&r.model),"native profile/anatomy");free(config);config=NULL;
    if(o->serve) {
        NATIVE_REQUIRE(o->maxseq<=bn_model_native_anatomy(&r.model)->context&&
            !native_cli_mul((size_t)o->maxseq+1,BN_GENERATE_BATCH_QUEUE_CAPACITY,&r.token_capacity),"serving context/storage");
        r.text_bytes=SERVE_RECORD_BYTES;r.count=(size_t)(o->prefill<o->maxseq?o->prefill:o->maxseq);
    }else if(o->prompt) {
        r.text_bytes=strlen(o->prompt);NATIVE_REQUIRE(r.text_bytes&&r.text_bytes<=BN_JSON_MAX_BYTES,"prompt length");
        r.text=malloc(r.text_bytes);NATIVE_REQUIRE(r.text,"prompt allocation");memcpy(r.text,o->prompt,r.text_bytes);
    }else if(!strcmp(o->batch_file?o->batch_file:o->prompt_file,"-")) {
        NATIVE_REQUIRE(!native_cli_read(stdin,BN_JSON_MAX_BYTES,&r.text,&r.text_bytes)&&r.text_bytes,"stdin prompt");
    }else NATIVE_REQUIRE(!native_cli_read_path(o->batch_file?o->batch_file:o->prompt_file,BN_JSON_MAX_BYTES,&r.text,&r.text_bytes)&&r.text_bytes,"prompt file");
    if(!o->serve)r.token_capacity=r.text_bytes;
    if(native_cli_batch_mode(o))NATIVE_REQUIRE(!native_cli_batch_read(&r,o),"batch prompt array");
    NATIVE_REQUIRE(!native_cli_mul(r.token_capacity,sizeof(int),&token_bytes)&&
        !bn_tokenizer_encode_scratch_plan(r.text_bytes,&r.scratch_bytes),"request size");
    NATIVE_REQUIRE(!native_cli_path(path,sizeof(path),o->directory,"tokenizer.json")&&
        !native_cli_tokenizers(&r,path,host_limit),"tokenizer/profile or host payload cap");
    const size_t base[]={r.ep_bytes,r.dp_bytes,r.text_bytes,token_bytes,r.piece_bytes,r.scratch_bytes,r.vocab_bytes,r.map_bytes};
    for(size_t i=0;i<sizeof(base)/sizeof(base[0]);i++)NATIVE_REQUIRE(!native_cli_add(&preliminary,base[i]),"host payload overflow");
    if(r.batch)NATIVE_REQUIRE(!native_cli_batch_host(&r,o,&preliminary),"batch host payload overflow");
    NATIVE_REQUIRE(!host_limit||preliminary<=host_limit,"host payload cap before request allocation");
    r.tokens=malloc(token_bytes);r.piece=malloc(r.piece_bytes);
    NATIVE_REQUIRE(r.tokens&&r.piece&&!bn_tokenizer_encode_scratch_init(&r.scratch,r.text_bytes,r.scratch_bytes),"request buffers");
    if(r.batch)NATIVE_REQUIRE(!native_cli_batch_encode(&r,o),"batch prompt encoding/context");
    else {
        NATIVE_REQUIRE(!bn_tokenizer_encoder_encode(&r.encoder,&r.scratch,r.text,r.text_bytes,r.tokens,r.text_bytes,&r.count),"prompt encoding");
        NATIVE_REQUIRE(!native_cli_context(r.count,o->tokens,o->predict,o->maxseq),"prompt plus generation exceeds maxseq");
    }
    NATIVE_REQUIRE(!native_cli_shape(&p,&r,o),"native profile/shape");
    size_t initial_host=preliminary,map_copy;
    NATIVE_REQUIRE(!native_cli_mul((size_t)bn_model_native_anatomy(&r.model)->vocab,sizeof(uint32_t),&map_copy)&&
        !native_cli_add(&initial_host,p.routers.host_staging_bytes)&&!native_cli_add(&initial_host,32*1024*1024)&&
        !native_cli_add(&initial_host,map_copy)&&(!host_limit||initial_host<=host_limit)&&
        (!device_limit||p.routers.device_peak_bytes<=device_limit),"resident weight payload cap");
    if(o->trace_directory) {
        NATIVE_REQUIRE(!native_cli_path(path,sizeof(path),o->trace_directory,"events.jsonl")&&
            (output.events=fopen(path,"wbx"))!=NULL,"exclusive trace directory");
    }
    fprintf(stderr,"native_text_weights device_payload_peak_bytes=%zu\n",p.routers.device_peak_bytes);
    NATIVE_REQUIRE((r.gpu=bn_gpu_cuda_create())!=NULL,"CUDA device");
    if(native_cli_batch_mode(o)&&!o->device_budget_set&&!o->device_mb) {
        size_t free_bytes,total;
        NATIVE_REQUIRE(r.gpu->memory_info&&!r.gpu->memory_info(r.gpu->ctx,&free_bytes,&total),"CUDA memory snapshot");
        device_limit=free_bytes-free_bytes/10;
    }
    BnGenerateBatchMemory initial={initial_host,p.routers.device_peak_bytes};
    NATIVE_REQUIRE(!bn_generate_batch_memory_init(&r.memory,
        (BnGenerateBatchMemory){host_limit?host_limit:SIZE_MAX,device_limit?device_limit:SIZE_MAX},
        (BnGenerateBatchMemoryQuote){initial,initial}),"shared model payload admission");
    NATIVE_REQUIRE(!bn_model_prepare_native_core(&r.model,r.gpu,p.core.device_peak_bytes,p.core.host_staging_bytes)&&
        !bn_model_prepare_native_parameters(&r.model,p.parameters.device_peak_bytes,p.parameters.host_staging_bytes)&&
        !bn_model_prepare_native_routers(&r.model,p.routers.device_peak_bytes,p.routers.host_staging_bytes),"resident weights");
    NATIVE_REQUIRE(!bn_model_prepare_native_row_caches(&r.model,8*1024*1024,p.rows.max_rows,32*1024*1024),"row caches");
    NATIVE_REQUIRE(!native_cli_plan(&p,&r),"native request plan");
    if(r.batch)NATIVE_REQUIRE(!native_cli_batch_plan(&p,&r,o),"batch request plan");
    fprintf(stderr,"native_text_plan prompt_tokens=%zu prefill_tokens=%d maxseq=%d host_payload_peak_bytes=%zu device_payload_peak_bytes=%zu scope=%s\n",r.count,p.f.max_tokens,o->maxseq,p.host,p.device,r.batch?"requested_concurrency_ceiling":"request_payload_ceiling");
    BnGenerateBatchMemoryQuote shared={p.shared,p.shared},request_quote;
    NATIVE_REQUIRE(!bn_generate_batch_memory_shared_prepare_as(&r.memory,shared),"shared cache payload admission");
    if(r.batch)NATIVE_REQUIRE(!native_cli_batch_admit_requests(r.batch),"aggregate payload cap (batch request admission)");
    else NATIVE_REQUIRE(!native_cli_memory_quote(&p.request,&request_quote)&&
        !bn_generate_batch_memory_check_shared(&r.memory,request_quote),"aggregate payload cap (native request admission)");
    BnGPUMoETrellisPolicy placement={.ram_admit_misses=o->ram_admit,
        .ram_pages=o->ram_system_pages?BN_MOE_CACHE_PAGES_SYSTEM:BN_MOE_CACHE_PAGES_HUGE_HINT};
    NATIVE_REQUIRE(!bn_gpu_moe_trellis_prepare_model_policy(&r.model,p.ram,p.decode,p.temporary,p.entry,o->pinned?p.entry:0,&placement),"expert placement");
    BnGPUMoETrellisStats placement_stats;
    NATIVE_REQUIRE(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(&r.model),&placement_stats),"expert page policy");
    fprintf(stderr,"native_text_ram_pages policy=%s cache_enabled=%d\n",
        placement_stats.ram_pages==BN_MOE_CACHE_PAGES_SYSTEM?"system":"huge",p.ram!=0);
    NATIVE_REQUIRE(!bn_generate_batch_memory_shared_commit(&r.memory),"shared payload commit");
    if(r.batch) {
        NATIVE_REQUIRE(!native_cli_batch_run(&r,o,output.events),"batch generation/output");
        status=0;goto done;
    }
    NATIVE_REQUIRE(!bn_generate_batch_memory_reserve(&r.memory,request_quote,&r.session_ticket),"native request payload reservation");
    r.session_reserved=1;
    BnTransformerTrellisRequestConfig request_config={p.f.max_tokens,o->maxseq,o->window,0,0};
    NATIVE_REQUIRE((r.session=bn_session_create_backend(&r.model,NULL))!=NULL&&
        !bn_transformer_trellis_prepare_request(&r.model,r.session,&request_config)&&
        !bn_generate_batch_memory_commit(&r.memory,r.session_ticket),"native session");
    native_cli_memory_report(&r.memory,"native_text_memory");
    if(o->prepare) {
        NATIVE_REQUIRE(printf("{\"profile\":\"exl3-int4-f16\",\"prompt_bytes\":%zu,\"prompt_tokens\":%zu,\"prefill_tokens\":%d,\"maxseq\":%d,\"host_payload_peak_bytes\":%zu,\"device_payload_peak_bytes\":%zu,\"prepared\":true,\"excludes\":\"JSON/index/allocator/driver/library overhead and OS file cache\"}\n",
            r.text_bytes,r.count,p.f.max_tokens,o->maxseq,p.host,p.device)>0&&!fflush(stdout),"prepare output");
        status=0;goto done;
    }
    BnGenerateNativeTextWorkspace w={&r.encoder,&r.decoder,&r.scratch,r.tokens,r.text_bytes,r.piece,r.piece_bytes};
    BnGenerateNativeTextOptions g={o->prefill,o->tokens,o->admit,!o->keep_special};
    BnGenerateNativeTextCallbacks cb={native_cli_output,output.events?native_cli_observe:NULL,&output};
    int generated=o->predict?
        bn_generate_native_text_predict(&r.model,r.session,&w,r.text,r.text_bytes,&g,&cb):
        bn_generate_native_text(&r.model,r.session,&w,r.text,r.text_bytes,&g,&cb);
    NATIVE_REQUIRE(generated>=0&&!output.failed,"generation/output");
    BnGPUMoETrellisStats stats;
    NATIVE_REQUIRE(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(&r.model),&stats)&&!stats.live_leases,"completed expert placement");
    fprintf(stderr,"native_text_result generated=%d position=%d output_bytes=%zu ram_hits=%zu ram_misses=%zu upload_bytes=%zu\n",
        generated,r.session->pos,output.bytes,stats.host.gpu_host_cache_hits,stats.host.gpu_host_cache_misses,stats.upload_bytes);
    status=0;
done:
    if(output.events&&fclose(output.events)){status=1;stage="trace output";}
    if(status)fprintf(stderr,"native text failed: %s\n",stage);
    free(config);native_cli_batch_free(r.batch);if(r.session)bn_session_free(r.session,NULL);
    if(r.session_reserved&&bn_generate_batch_memory_drop(&r.memory,r.session_ticket))status=1;
    bn_model_free(&r.model);
    if(r.gpu)bn_gpu_cuda_destroy(r.gpu);
    bn_safetensors_index_free(&r.index);
    bn_tokenizer_map_free(&r.map);bn_tokenizer_vocab_free(&r.vocabulary);
    bn_tokenizer_encode_scratch_free(&r.scratch);bn_tokenizer_decoder_free(&r.decoder);bn_tokenizer_encoder_free(&r.encoder);
    bn_json_free(&r.tokenizer);free(r.tokens);free(r.piece);free(r.text);return status;
#undef NATIVE_REQUIRE
}
#endif
static int native_cli_main(int argc,char **argv) {
    NativeCLIOptions options;int parsed=native_cli_parse(argc,argv,&options);
    if(parsed){native_cli_usage(parsed>0?stdout:stderr,argv[0]);return parsed>0?0:2;}
#ifdef BN_ENABLE_CUDA
    return native_cli_run(&options);
#else
    fputs("Native text requires a CUDA build.\n",stderr);return 1;
#endif
}
