/* Explicit native CUDA profile. Saved source IDs and logits are required.
 * Timed region: history staging through completed CPU-logit publication.
 * Verification and model/session preparation are outside the timed region.
 * Optional callback counters add no device work or completion boundaries.
 */
#include "backend_layout.h"
#include "gpu_cuda.h"
#include "gpu_moe_bridge.h"
#include "json.h"
#include "generate.h"
#include "model.h"
#include "platform.h"
#include "quant.h"
#include "safetensors.h"
#include "session.h"
#include "session_internal.h"
#include "tokenizer.h"
#include "transformer_internal.h"
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "native_text_workload.h"

static int observing;
static struct {
    size_t allocs, writes, write_bytes, reads, read_bytes, submits, submit_bytes;
    size_t syncs, fused, fused_experts, folded, decode, linear, ffn;
    double sync_ms, read_ms;
} observed;
static BnGPUBackend original;

static size_t profile_bytes(BnGPUResidentView v) {
    size_t width=bn_quant_format_resident_scalar_bytes(v.type);
    assert(width && v.count<=SIZE_MAX/width);
    return v.count*width;
}
static int profile_alloc(void *r,int type,size_t count,BnGPUResidentView *out) {
    if(observing)observed.allocs++;
    return original.resident_alloc(r,type,count,out);
}
static int profile_write(void *r,BnGPUResidentView v,const void *host) {
    if(observing){observed.writes++;observed.write_bytes+=profile_bytes(v);}
    return original.resident_write(r,v,host);
}
static int profile_read(void *r,BnGPUResidentView v,void *host) {
    if(!observing)return original.resident_read(r,v,host);
    double start=bn_platform_time_ms();
    int rc=original.resident_read(r,v,host);
    observed.reads++;observed.read_bytes+=profile_bytes(v);
    observed.read_ms+=bn_platform_time_ms()-start;
    return rc;
}
static int profile_submit(void *r,const BnGPUHostUploadLease *lease,BnGPUResidentView v) {
    if(observing){observed.submits++;observed.submit_bytes+=profile_bytes(v);}
    return original.resident_upload_submit(r,lease,v);
}
static int profile_sync(void *r) {
    if(!observing)return original.resident_synchronize(r);
    double start=bn_platform_time_ms();
    int rc=original.resident_synchronize(r);
    observed.syncs++;observed.sync_ms+=bn_platform_time_ms()-start;
    return rc;
}
#define PROFILE_OP(NAME,FIELD,TYPE,COUNTER) \
static int NAME(void *c,void *r,const TYPE *plan) { \
    if(observing)observed.COUNTER++; \
    return original.FIELD(c,r,plan); \
}
PROFILE_OP(profile_folded,trellis_folded_ffn_resident,BnGPUFoldedEXL3FFNResidentPlan,folded)
PROFILE_OP(profile_decode,trellis_decode_resident,BnGPUDecodeEXL3ResidentPlan,decode)
PROFILE_OP(profile_linear,linear_f16_resident,BnGPUF16LinearResidentPlan,linear)
PROFILE_OP(profile_ffn,ffn_f16_resident,BnGPUF16FFNResidentPlan,ffn)
static int profile_fused(void *c,void *r,const BnGPUFusedEXL3ResidentPlan *plan) {
    if(observing){observed.fused++;observed.fused_experts+=(size_t)plan->resident_count;}
    return original.trellis_fused_resident(c,r,plan);
}
static void profile_observe(BnGPUBackend *g) {
    original=*g;
    g->resident_alloc=profile_alloc;g->resident_write=profile_write;
    g->resident_read=profile_read;g->resident_upload_submit=profile_submit;
    g->resident_synchronize=profile_sync;g->trellis_fused_resident=profile_fused;
    g->trellis_folded_ffn_resident=profile_folded;g->trellis_decode_resident=profile_decode;
    g->linear_f16_resident=profile_linear;g->ffn_f16_resident=profile_ffn;
}
static void *profile_file(const char *dir,const char *name,size_t *bytes) {
    char path[4096];int length=name[0]=='/'?snprintf(path,sizeof(path),"%s",name):snprintf(path,sizeof(path),"%s/%s",dir,name);
    assert(length>0&&(size_t)length<sizeof(path));
    FILE *file=fopen(path,"rb");assert(file&&!fseek(file,0,SEEK_END));
    long end=ftell(file);assert(end>=0&&!fseek(file,0,SEEK_SET));
    *bytes=(size_t)end;void *blob=malloc(*bytes?*bytes:1);assert(blob);
    assert(fread(blob,1,*bytes,file)==*bytes&&!fclose(file));return blob;
}
static void profile_json(BnJSON *doc,const char *dir,const char *name,int vocab) {
    size_t bytes;void *blob=profile_file(dir,name,&bytes);
    BnJSONLimits limits={vocab?bytes:BN_JSON_MAX_BYTES,vocab?1000000:BN_JSON_MAX_NODES};
    assert(!bn_json_init_limits(doc,blob,bytes,&limits));free(blob);
}
static int profile_number(const BnJSONValue *v,const char *key) {
    uint64_t n;assert(!bn_json_u64(bn_json_get(v,key),&n)&&n<=INT_MAX);return (int)n;
}
static void *profile_load(const char *dir,const BnJSONValue *frame,const char *key,size_t count) {
    size_t bytes;const char *name=bn_json_text(bn_json_get(frame,key));assert(name);
    void *blob=profile_file(dir,name,&bytes);assert(bytes==count);return blob;
}
static int profile_arg(const char *text,int lo,int hi) {
    char *end;errno=0;long n=strtol(text,&end,10);
    assert(!errno&&*text&&!*end&&n>=lo&&n<=hi);return (int)n;
}
static void profile_map_setup(BnModel *m,const char *path) {
    BnJSON doc={0};profile_json(&doc,".",path,1);
    BnTokenizerMapSpec spec;assert(!bn_model_native_engram_map_spec(m,&spec));
    BnTokenizerVocab vocab={0};size_t bytes;BnTokenizerMapWorkspace work;
    assert(!bn_tokenizer_vocab_plan(spec.vocab_size,&bytes));
    assert(!bn_tokenizer_vocab_init_json(&vocab,&doc,spec.vocab_size,bytes));
    assert(!bn_tokenizer_map_plan(&vocab,&spec,&work));BnTokenizerMap map={0};
    assert(!bn_tokenizer_map_init(&map,&vocab,&spec,work.host_bytes));
    assert(!bn_model_prepare_native_token_map(m,bn_tokenizer_map_ids(&map),
        (size_t)spec.vocab_size,(size_t)spec.vocab_size*sizeof(uint32_t)));
    bn_tokenizer_map_free(&map);bn_tokenizer_vocab_free(&vocab);bn_json_free(&doc);
}

#include "native_text_profile.h"
#include "native_text_latency.h"
#include "native_text_schedule.h"
#include "native_text_group.h"

int main(int argc,char **argv) {
    if(argc<7){fprintf(stderr,"Usage: %s model-dir reference-dir vocab-json fused-window runs all-logits [--text | --predict-text] [--text-timing] [--session-pool N] [--schedule-live N] [--forward-group N] [--arrival-ms N] [--schedule-cancel ID] [--request-reference DIR ...] [--ram-admit-misses] [--ram-page-policy huge|system] [--observe] [--pinned-staging] [--admit-decode] [--decode-cache-experts N] [--ram-cache-experts N]\n",argv[0]);return 2;}
    const char *modeldir=argv[1],*block_dir=argv[2];
    const char *reference_dirs[BN_TEXT_WORKLOAD_MAX_REQUESTS]={block_dir};
    int request_count=1;
    int window=profile_arg(argv[4],1,32),runs=profile_arg(argv[5],1,100);
    int all_logits=profile_arg(argv[6],0,1),observe=0,admit_decode=0,pinned_staging=0;
    int decode_entries=48,ram_entries=48,text_mode=0,text_timing=0,ram_admit=0,pool_slots=0,schedule_live=0,arrival_ms=0,cancel_id=-1,forward_group=0;
    int ram_system_pages=0,page_policy_set=0;
    for(int i=7;i<argc;i++){
        if(!strcmp(argv[i],"--text"))text_mode=1;
        else if(!strcmp(argv[i],"--predict-text"))text_mode=2;
        else if(!strcmp(argv[i],"--text-timing"))text_timing=1;
        else if(!strcmp(argv[i],"--session-pool")&&i+1<argc)
            pool_slots=profile_arg(argv[++i],1,16);
        else if(!strcmp(argv[i],"--forward-group")&&i+1<argc)
            forward_group=profile_arg(argv[++i],1,16);
        else if(!strcmp(argv[i],"--schedule-live")&&i+1<argc)
            schedule_live=profile_arg(argv[++i],1,16);
        else if(!strcmp(argv[i],"--arrival-ms")&&i+1<argc)
            arrival_ms=profile_arg(argv[++i],0,60000);
        else if(!strcmp(argv[i],"--schedule-cancel")&&i+1<argc)
            cancel_id=profile_arg(argv[++i],0,99);
        else if(!strcmp(argv[i],"--ram-admit-misses"))ram_admit=1;
        else if(!strcmp(argv[i],"--ram-page-policy")&&i+1<argc) {
            const char *value=argv[++i];
            if(page_policy_set||(strcmp(value,"huge")&&strcmp(value,"system"))) {
                fprintf(stderr,"Invalid or duplicate RAM page policy\n");return 2;
            }
            page_policy_set=1;ram_system_pages=!strcmp(value,"system");
        }
        else if(!strcmp(argv[i],"--observe"))observe=1;
        else if(!strcmp(argv[i],"--pinned-staging"))pinned_staging=1;
        else if(!strcmp(argv[i],"--request-reference")&&i+1<argc&&
                *argv[i+1]&&request_count<BN_TEXT_WORKLOAD_MAX_REQUESTS)
            reference_dirs[request_count++]=argv[++i];
        else if(!strcmp(argv[i],"--admit-decode"))admit_decode=1;
        else if(!strcmp(argv[i],"--decode-cache-experts")&&i+1<argc)
            decode_entries=profile_arg(argv[++i],48,4096);
        else if(!strcmp(argv[i],"--ram-cache-experts")&&i+1<argc)
            ram_entries=profile_arg(argv[++i],1,65536);
        else {fprintf(stderr,"Unknown or incomplete option: %s\n",argv[i]);return 2;}
    }
    if(text_mode&&all_logits){fprintf(stderr,"--text requires last-row-only logits (all-logits=0)\n");return 2;}
    if(text_timing&&(!text_mode||observe)){fprintf(stderr,"--text-timing requires --text/--predict-text and excludes --observe\n");return 2;}
    if(request_count>1&&!text_mode){fprintf(stderr,"--request-reference requires --text/--predict-text\n");return 2;}
    if(pool_slots&&!text_mode){fprintf(stderr,"--session-pool requires --text/--predict-text\n");return 2;}
    if(schedule_live&&(!text_mode||!pool_slots||observe)){
        fprintf(stderr,"--schedule-live requires --text/--predict-text and --session-pool, excludes --observe\n");return 2;
    }
    if((arrival_ms||cancel_id>=0)&&!schedule_live){fprintf(stderr,"arrival/cancel options require --schedule-live\n");return 2;}
    if(cancel_id>=runs){fprintf(stderr,"--schedule-cancel must name a request below runs\n");return 2;}
    if(forward_group&&(text_mode!=1||pool_slots<forward_group||schedule_live||observe||arrival_ms||cancel_id>=0)){
        fprintf(stderr,"--forward-group requires --text and sufficient --session-pool, excludes scheduler/arrival/cancel/observe\n");return 2;
    }
    BnJSON request_docs[BN_TEXT_WORKLOAD_MAX_REQUESTS]={0};
    BnJSON request_hostdocs[BN_TEXT_WORKLOAD_MAX_REQUESTS]={0};
    const BnJSONValue *request_frames[BN_TEXT_WORKLOAD_MAX_REQUESTS];
    const BnJSONValue *request_hostframes[BN_TEXT_WORKLOAD_MAX_REQUESTS];
    int request_prefill[BN_TEXT_WORKLOAD_MAX_REQUESTS];
    int max_tokens=1,maxseq=0;
    for(int request=0;request<request_count;request++) {
        profile_json(&request_docs[request],reference_dirs[request],"reference.json",0);
        profile_json(&request_hostdocs[request],reference_dirs[request],"host-reference.json",0);
        request_frames[request]=bn_json_get(bn_json_root(&request_docs[request]),"frames");
        request_hostframes[request]=bn_json_get(bn_json_root(&request_hostdocs[request]),"frames");
        BnTextWorkloadShape shape={0};int context;
        for(size_t i=0;i<bn_json_size(request_frames[request]);i++) {
            const BnJSONValue *frame=bn_json_at(request_frames[request],i);
            assert(!native_text_workload_add(&shape,profile_number(frame,"rows"),
                                             profile_number(frame,"position")));
        }
        assert(!native_text_workload_finish(&shape,&request_prefill[request],&context));
        if(request_prefill[request]>max_tokens)max_tokens=request_prefill[request];
        if(context>maxseq)maxseq=context;
    }
    const BnJSONValue *frames=request_frames[0],*hostframes=request_hostframes[0];
    size_t n;void *blob;
 char path[1024];snprintf(path,sizeof(path),"%s/model.safetensors.index.json",modeldir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));blob=profile_file(modeldir,"config.json",&n);BnModel m={0};assert(!bn_model_load_trellis_config(&m,&index,blob,n)&&!bn_model_prepare_native_anatomy(&m)&&!bn_model_prepare_native_numerics(&m)&&!bn_model_prepare_native_rows(&m));free(blob);profile_map_setup(&m,argv[3]);
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g);if(observe)profile_observe(g);const BnModelNativeAnatomy*a=bn_model_native_anatomy(&m);BnBackendNativeCoreStats core;BnBackendNativeParameterStats params;BnBackendNativeRouterStats routers;
 assert(!bn_backend_layout_native_core_plan(&index,a,&core)&&!bn_model_prepare_native_core(&m,g,core.device_peak_bytes,core.host_staging_bytes));
 assert(!bn_backend_layout_native_parameter_plan(&index,a,&params)&&!bn_model_prepare_native_parameters(&m,params.device_peak_bytes,params.host_staging_bytes));
 assert(!bn_backend_layout_native_router_plan(&index,a,&routers)&&!bn_model_prepare_native_routers(&m,routers.device_peak_bytes,routers.host_staging_bytes));
 int row_heads=0;
 for(int layer=0;layer<a->experts.layers;layer++)if(a->layers[layer].engram_rows){
     BnModelNativeEngramMath math;
     assert(!bn_model_native_engram_math(&m,layer,&math)&&math.heads>0);
     if(math.heads>row_heads)row_heads=math.heads;
 }
 assert(row_heads>0&&max_tokens<=INT_MAX/row_heads);
 assert(!bn_model_prepare_native_row_caches(&m,8*1024*1024,max_tokens*row_heads,32*1024*1024));const BnMoEExpertReader*reader=bn_model_trellis_expert_reader(&m);size_t entry=reader->bytes[0]+reader->bytes[1]+reader->bytes[2];BnGPUMoETrellisPolicy placement={.ram_admit_misses=ram_admit,.ram_pages=ram_system_pages?BN_MOE_CACHE_PAGES_SYSTEM:BN_MOE_CACHE_PAGES_HUGE_HINT};assert(entry<=SIZE_MAX/(size_t)ram_entries&&entry<=SIZE_MAX/4096&&!bn_gpu_moe_trellis_prepare_model_policy(&m,entry*(size_t)ram_entries,entry*(size_t)decode_entries,entry*48,entry,pinned_staging?entry:0,&placement));
 BnGPUMoETrellisStats placement_stats;assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(&m),&placement_stats));
 fprintf(stderr,"native_bench_ram_pages policy=%s\n",placement_stats.ram_pages==BN_MOE_CACHE_PAGES_SYSTEM?"system":"huge");
 BnTransformerTrellisOptions f={max_tokens,window,0,0,0};BnTransformerTrellisWorkspace fw;BnTransformerTrellisKVOptions kv={maxseq,0,0};BnTransformerTrellisKVWorkspace kw;BnTransformerTrellisSWAWorkspace sw;BnTransformerTrellisIndexWorkspace iw;BnTransformerTrellisAttentionWorkspace aw;BnTransformerTrellisProducerWorkspace pw;
 assert(!bn_transformer_trellis_session_plan(&m,&f,&fw)&&!bn_transformer_trellis_kv_session_plan(&m,&kv,&kw)&&!bn_transformer_trellis_swa_session_plan(&m,&sw)&&!bn_transformer_trellis_index_session_plan(&m,&f,&kv,&iw)&&!bn_transformer_trellis_attention_session_plan(&m,&f,&kv,&aw)&&!bn_transformer_trellis_producer_session_plan(&m,&f,&kv,&pw));
 f.device_budget=fw.device_bytes;f.host_budget=fw.host_bytes;kv.device_budget=kw.device_bytes;kv.host_budget=kw.host_bytes;BnTransformerTrellisSWAOptions so={sw.device_bytes,sw.host_bytes};BnTransformerTrellisIndexOptions io={iw.device_bytes,iw.host_bytes};BnTransformerTrellisAttentionOptions ao={aw.device_bytes,aw.host_bytes};BnTransformerTrellisProducerOptions po={pw.device_bytes,pw.host_bytes};
 BnTransformerTrellisRowsOptions rows={max_tokens*row_heads,2,0,0,0};BnTransformerTrellisRowsWorkspace rw;assert(!bn_transformer_trellis_rows_session_plan(&m,&rows,&rw));rows.host_budget=rw.host_bytes;rows.pinned_budget=rw.pinned_bytes;rows.device_budget=rw.device_bytes;
 BnTransformerTrellisEngramWorkspace ew;BnTransformerTrellisHCWorkspace hw;assert(!bn_transformer_trellis_engram_session_plan(&m,&f,&rows,&ew)&&!bn_transformer_trellis_hc_session_plan(&m,&f,&hw));BnTransformerTrellisEngramOptions e={ew.device_bytes,ew.host_bytes};BnTransformerTrellisHCOptions h={hw.device_bytes,hw.host_bytes};
 BnSession *session=bn_session_create_backend(&m,NULL);assert(session&&session->backend_only);
 assert(!bn_session_native_history_prepare(session,&m,max_tokens,ew.history_bytes));
 BnTransformerTrellisForwardOptions vo={all_logits,0,0};
 BnTransformerTrellisForwardWorkspace vw;
 assert(!bn_transformer_trellis_forward_session_plan(&m,&f,&vo,&vw));
 vo.device_budget=vw.device_bytes;vo.host_budget=vw.host_bytes;
#define FULL_PREPARE() bn_transformer_trellis_prepare_session_forward(&m,session,&f,&kv,&so,&io,&ao,&po,&rows,&e,&h,&vo)

    fprintf(stderr,"native profile window%d max_tokens%d maxseq%d observe%d request_ffn_device%zu temporary_payload%zu decode_payload%zu ram_payload%zu admit_decode%d\n",window,max_tokens,maxseq,observe,fw.device_bytes,entry*48,entry*(size_t)decode_entries,entry*(size_t)ram_entries,admit_decode);
    int *tokens=malloc((size_t)max_tokens*sizeof(*tokens));assert(tokens);
    if(text_mode) {
        ProfileNativeText text[BN_TEXT_WORKLOAD_MAX_REQUESTS]={0};
        for(int request=0;request<request_count;request++) {
            profile_text_init(&text[request],reference_dirs[request],argv[3],a->vocab,
                              request_frames[request],request_hostframes[request]);
            text[request].predict=text_mode==2;
            text[request].request_index=request;text[request].request_count=request_count;
        }
        if(forward_group) {
            profile_text_group(&m,text,request_count,request_prefill,runs,max_tokens,maxseq,window,
                pool_slots,forward_group,text_timing,admit_decode);
            for(int request=0;request<request_count;request++)profile_text_free(&text[request]);
            free(tokens);goto profile_cleanup;
        }
        if(schedule_live) {
            profile_text_schedule(&m,text,request_count,request_prefill,runs,max_tokens,maxseq,window,
                pool_slots,schedule_live,text_timing,arrival_ms,cancel_id,admit_decode);
            for(int request=0;request<request_count;request++)profile_text_free(&text[request]);
            free(tokens);goto profile_cleanup;
        }
        BnTransformerTrellisPool pool={0};
        if(pool_slots) {
            assert(maxseq<=INT_MAX-255);
            BnTransformerTrellisRequestConfig ceiling={max_tokens,(maxseq+255)/256*256,window,0,all_logits};
            BnTransformerTrellisRequestPlan plan;
            assert(!bn_transformer_trellis_request_plan(&m,&ceiling,&plan));
            assert(plan.device_bytes<=SIZE_MAX/(size_t)pool_slots&&
                   plan.host_bytes<=SIZE_MAX/(size_t)pool_slots&&plan.pinned_bytes<=SIZE_MAX/(size_t)pool_slots);
            assert(!bn_transformer_trellis_pool_init(&pool,&m,&ceiling,pool_slots,
                plan.device_bytes*(size_t)pool_slots,plan.host_bytes*(size_t)pool_slots,
                plan.pinned_bytes*(size_t)pool_slots));
        }
        for(int run=0;run<runs;run++) {
            int request=native_text_workload_pick(run,request_count);assert(request>=0);
            BnSession *lease=session;int lease_capacity=maxseq;
            if(pool_slots) {
                size_t needed=text[request].prompt_tokens+(size_t)text[request].generated;
                if(text[request].predict&&text[request].generated)needed--;
                assert(needed<=(size_t)(INT_MAX-255));
                lease_capacity=((int)needed+255)/256*256;
                assert(!bn_transformer_trellis_pool_acquire(&pool,
                    request_prefill[request],(int)needed,&lease));
            }else {bn_session_reset(session,&m);assert(!FULL_PREPARE());}
            if(text_timing)profile_text_latency(&text[request],&m,lease,run,
                request_prefill[request],lease_capacity,admit_decode,ram_admit);
            else profile_text_execute(&text[request],&m,lease,run,
                request_prefill[request],admit_decode);
            if(pool_slots)assert(!bn_transformer_trellis_pool_release(&pool,lease,0));
        }
        if(pool_slots) {
            BnTransformerTrellisPoolInfo info;
            assert(!bn_transformer_trellis_pool_info(&pool,&info));
            printf("{\"kind\":\"native_session_pool\",\"timing_scope\":\"generation_only_excludes_acquire_release\","
                   "\"slots\":%d,\"allocated\":%d,\"leased\":%d,\"created\":%llu,\"reused\":%llu,"
                   "\"released\":%llu,\"evicted\":%llu,\"discarded\":%llu,\"restart_failures\":%llu,"
                   "\"reserved_device\":%zu,\"reserved_host\":%zu,\"reserved_pinned\":%zu}\n",
                pool_slots,info.allocated_slots,info.leased_slots,(unsigned long long)info.created,
                (unsigned long long)info.reused,(unsigned long long)info.released,(unsigned long long)info.evicted,
                (unsigned long long)info.discarded,(unsigned long long)info.restart_failures,
                info.reserved_device,info.reserved_host,info.reserved_pinned);
            bn_transformer_trellis_pool_free(&pool);
        }
        for(int request=0;request<request_count;request++)profile_text_free(&text[request]);
        free(tokens);goto profile_cleanup;
    }
    size_t verified=0,verified_bytes=0,greedy=0;
    for(int run=0;run<runs;run++){
        bn_session_reset(session,&m);assert(!FULL_PREPARE());
        for(size_t fi=0;fi<bn_json_size(frames);fi++){
            const BnJSONValue *frame=bn_json_at(frames,fi);
            const char *name=bn_json_text(bn_json_get(frame,"name"));assert(name);
            int R=profile_number(frame,"rows"),pos=profile_number(frame,"position"),generated=0;
            assert(session->pos==pos);
            const BnJSONValue *value=bn_json_get(frame,"generated");
            if(value)assert(!bn_json_bool(value,&generated));
            uint64_t *raw=profile_load(block_dir,frame,"tokens",(size_t)R*8);
            for(int i=0;i<R;i++){assert(raw[i]<(uint64_t)a->vocab);tokens[i]=(int)raw[i];}free(raw);
            if(generated){int top=0;for(int i=1;i<a->vocab;i++)if(session->state.logits[i]>session->state.logits[top])top=i;assert(R==1&&top==tokens[0]);greedy++;}
            const char *key=all_logits?"logits":"last_logits";
            size_t gpu_bytes=(size_t)(all_logits?R:1)*a->vocab*2;
            void *gpu_gold=profile_load(block_dir,frame,key,gpu_bytes),*got=malloc(gpu_bytes);
            void *host_gold=profile_load(block_dir,bn_json_get(hostframes,name),key,a->vocab*4);
            assert(got);BnGPUMoETrellisStats before,after;
            assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(&m),&before));
            memset(&observed,0,sizeof(observed));observing=observe;
            double start=bn_platform_time_ms();
            assert(!bn_session_native_history_stage(session,&m,tokens,R));
            BnTransformerTrellisForwardResult out;int status,busy=0;
            int admit=admit_decode&&R<=8;
            do{status=bn_transformer_trellis_forward_run(&m,session,admit,&out);if(status==BN_GPU_UPLOAD_BUSY)busy++;}while(status==BN_GPU_UPLOAD_BUSY);
            double elapsed=bn_platform_time_ms()-start;observing=0;
            assert(!status&&session->pos==pos+R&&bn_transformer_trellis_forward_current(&m,session,&out));
            assert(!bn_gpu_moe_trellis_stats(bn_gpu_moe_trellis_model_store(&m),&after)&&!after.live_leases);
            assert(out.logits.count*2==gpu_bytes&&!g->resident_read(out.request,out.logits,got));
            assert(!memcmp(gpu_gold,got,gpu_bytes)&&!memcmp(host_gold,session->state.logits,a->vocab*4));
            verified+=2;verified_bytes+=gpu_bytes+a->vocab*4;
            printf("{\"run\":%d,"
                "\"frame\":\"%s\","
                "\"rows\":%d,"
                "\"position\":%d,"
                "\"window\":%d,"
                "\"all_logits\":%d,"
                "\"observed\":%d,"
                "\"elapsed_ms\":%.6f,"
                "\"busy\":%d,"
                "\"admit\":%d,"
                "\"decode_cache_bytes\":%zu,"
                "\"ram_cache_bytes\":%zu,"
                "\"expert_uploads\":%zu,"
                "\"expert_upload_bytes\":%zu,"
                "\"expert_upload_ms\":%.6f,"
                "\"decode_hits\":%zu,"
                "\"decode_misses\":%zu,"
                "\"ram_hits\":%zu,"
                "\"ram_misses\":%zu,"
                "\"ram_miss_bytes\":%zu,"
                "\"decode_charged\":%zu,"
                "\"temporary_charged\":%zu,"
                "\"temporary_allocated\":%zu,"
                "\"temporary_peak\":%zu,"
                "\"buffer_creates\":%zu,"
                "\"buffer_refills\":%zu,"
                "\"expert_read_calls\":%d,"
                "\"expert_read_bytes\":%zu,"
                "\"expert_read_ms\":%.6f,"
                "\"resident_allocs\":%zu,"
                "\"write_calls\":%zu,"
                "\"write_bytes\":%zu,"
                "\"read_calls\":%zu,"
                "\"read_bytes\":%zu,"
                "\"read_ms\":%.6f,"
                "\"upload_submit_calls\":%zu,"
                "\"upload_submit_bytes\":%zu,"
                "\"sync_calls\":%zu,"
                "\"sync_ms\":%.6f,"
                "\"fused_calls\":%zu,"
                "\"fused_experts\":%zu,"
                "\"folded_calls\":%zu,"
                "\"decode_calls\":%zu,"
                "\"linear_calls\":%zu,"
                "\"ffn_calls\":%zu,\"pinned_staging_bytes\":%zu,\"upload_batches\":%zu}\n",
                run,name,R,pos,window,all_logits,observe,elapsed,busy,admit,entry*(size_t)decode_entries,entry*(size_t)ram_entries,after.upload_count-before.upload_count,after.upload_bytes-before.upload_bytes,after.upload_time_ms-before.upload_time_ms,after.decode.hits-before.decode.hits,after.decode.misses-before.decode.misses,after.host.gpu_host_cache_hits-before.host.gpu_host_cache_hits,after.host.gpu_host_cache_misses-before.host.gpu_host_cache_misses,after.host.gpu_host_cache_miss_bytes-before.host.gpu_host_cache_miss_bytes,after.decode.charged_payload,after.temporary.charged_payload,after.temporary_allocated_payload,after.temporary_peak_payload,after.buffer_create_count-before.buffer_create_count,after.buffer_reuse_count-before.buffer_reuse_count,after.host.io_count-before.host.io_count,after.host.io_bytes-before.host.io_bytes,after.host.io_time_ms-before.host.io_time_ms,observed.allocs,observed.writes,observed.write_bytes,observed.reads,observed.read_bytes,observed.read_ms,observed.submits,observed.submit_bytes,observed.syncs,observed.sync_ms,observed.fused,observed.fused_experts,observed.folded,observed.decode,observed.linear,observed.ffn,after.pinned_staging_payload,after.upload_batches-before.upload_batches);
            fflush(stdout);free(gpu_gold);free(host_gold);free(got);
        }
    }
    fprintf(stderr,"VERIFIED checks%zu bytes%zu self_fed%zu\n",verified,verified_bytes,greedy);
    free(tokens);
profile_cleanup:
    bn_session_free(session,NULL);bn_model_free(&m);bn_safetensors_index_free(&index);
    bn_gpu_cuda_destroy(g);
    for(int request=0;request<request_count;request++) {
        bn_json_free(&request_hostdocs[request]);bn_json_free(&request_docs[request]);
    }
    return 0;
#undef FULL_PREPARE
}
