/* Explicit token-to-logit GLM execution/lifecycle. The first four connected
 * blocks are compared with independent original captures. Remaining layers
 * exercise native integration/replay, not whole-model oracle equivalence. */
#include "backend_layout.h"
#include "backend_model.h"
#include "backend_session.h"
#include "gpu_cuda.h"
#include "model_arch.h"
#include "model_internal.h"
#include "moe.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
static int (*real_acquire)(void*,unsigned,int,int,BnGPUMoEBankLease*);
static int (*real_fused)(void*,void*,const BnGPUMHCFusedResidentPlan*);
static int (*real_post)(void*,void*,const BnGPUMHCPostResidentPlan*);
static int (*real_pre)(void*,void*,const BnGPUMHCPreResidentPlan*);
static int (*real_alloc)(void*,int,size_t,BnGPUResidentView*);
static int (*real_sync)(void*);
static int (*real_select)(void*,void*,const BnGPUIndexerSelectResidentPlan*);
static const char *replay_dir,*replay_prefix;
static int index_diagnostics;
static BnGPUBackend *gpu;
static unsigned acquire_calls,forced_busy,fused_calls,pre_calls,alloc_calls,fail_alloc;
static int force_busy,fail_sync,check_prefix;
static const char *dense_dir,*mla_dir;
static unsigned frame_number,capture_frames=2;
static int chunks[8191],nchunks,capture_states,max_tokens=9,capacity=64,prompt_tokens;
extern int cudaProfilerStart(void);
extern int cudaProfilerStop(void);
static void schedule_init(void){
    const char *p=getenv("BN_GLM_REQUEST_SCHEDULE");
    if(!p){chunks[0]=getenv("BN_GLM_REQUEST_FAST")?1:9;nchunks=1;prompt_tokens=chunks[0];return;}
    int total=0;
    for(;;){char*end;long n=strtol(p,&end,10);assert(end!=p&&n>=1&&n<=8191&&nchunks<8191);
        chunks[nchunks++]=(int)n;total+=(int)n;assert(total<=8191);
        if(!*end)break;
        assert(*end==','&&end[1]);p=end+1;
    }
    prompt_tokens=total;capacity=((total+1+63)/64)*64;
    for(int j=0;j<nchunks;++j)if(chunks[j]>max_tokens)max_tokens=chunks[j];
}
static const char *capture_dir;
static void save_view(void*r,int layer,const char*name,BnGPUResidentView v){
    size_t width=v.type==BN_GGUF_TENSOR_I8?1:v.type==BN_GGUF_TENSOR_BF16?2:4;
    size_t bytes=v.count*width;void*data=malloc(bytes);assert(data&&!gpu->resident_read(r,v,data));
    char path[2048];snprintf(path,sizeof(path),"%s/%u-layer%d-%s.bin",capture_dir,frame_number-1,layer,name);
    FILE*f=fopen(path,"wb");assert(f&&fwrite(data,1,bytes,f)==bytes);fclose(f);free(data);
}
static void capture(void*r,int layer,BnGPUResidentView residual,BnGPUResidentView branch,BnGPUResidentView post,BnGPUResidentView comb){
    if(!capture_dir||frame_number>capture_frames)return;
    BnGPUResidentView views[]={residual,branch,post,comb};const char*names[]={"residual","out","post","comb"};
    for(int j=0;j<4;++j)save_view(r,layer,names[j],views[j]);
}
static int post(void*c,void*r,const BnGPUMHCPostResidentPlan*p){capture(r,44,p->residual,p->branch,p->post,p->comb);return real_post(c,r,p);}
static void *read_file(const char *dir,const char *name,size_t bytes){
    char path[2048];snprintf(path,sizeof(path),"%s/%s.bin",dir,name);FILE*f=fopen(path,"rb");assert(f);
    void *p=malloc(bytes);assert(p&&fread(p,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return p;
}
static void compare(void *request,BnGPUResidentView v,const char *dir,const char *name){
    size_t bytes=v.count*bn_quant_format_resident_scalar_bytes(v.type);void *a=malloc(bytes),*b=read_file(dir,name,bytes);assert(a);
    assert(!gpu->resident_read(request,v,a));if(memcmp(a,b,bytes)){fprintf(stderr,"connected prefix mismatch: %s/%s\n",dir,name);abort();}free(a);free(b);
}
static int acquire(void *r,unsigned s,int g,int d,BnGPUMoEBankLease *out){
    acquire_calls++;if(force_busy&&(acquire_calls&1)){forced_busy++;return BN_GPU_MOE_BANK_BUSY;}return real_acquire(r,s,g,d,out);
}
static int alloc_probe(void*r,int t,size_t n,BnGPUResidentView*out){alloc_calls++;return alloc_calls==fail_alloc?-1:real_alloc(r,t,n,out);}
static int sync_probe(void*r){return fail_sync?-1:real_sync(r);}
static int pre(void*c,void*r,const BnGPUMHCPreResidentPlan*p){pre_calls++;return real_pre(c,r,p);}
static int fused(void*c,void*r,const BnGPUMHCFusedResidentPlan*p){
    fused_calls++;
    if(!(fused_calls&1))capture(r,(int)fused_calls/2-1,p->previous_residual,p->previous_branch,p->previous_post,p->previous_comb);
    if(check_prefix&&fused_calls==6){compare(r,p->previous_residual,dense_dir,"0-layer2-ffn_residual");compare(r,p->previous_branch,dense_dir,"0-layer2-out");compare(r,p->previous_post,dense_dir,"0-layer2-ffn_post");compare(r,p->previous_comb,dense_dir,"0-layer2-ffn_comb");}
    if(check_prefix&&fused_calls==7){compare(r,p->previous_residual,mla_dir,"0-attn_residual");compare(r,p->previous_branch,mla_dir,"0-attn_out");compare(r,p->previous_post,mla_dir,"0-attn_post");compare(r,p->previous_comb,mla_dir,"0-attn_comb");}
    if(check_prefix&&fused_calls==8){compare(r,p->previous_residual,mla_dir,"0-ffn_residual");compare(r,p->previous_branch,mla_dir,"0-out");compare(r,p->previous_post,mla_dir,"0-ffn_post");compare(r,p->previous_comb,mla_dir,"0-ffn_comb");}
    return real_fused(c,r,p);
}
/* Test-only original PERMUTATION replay. Scores and deterministic pool
 * membership are checked before replacing ONLY expanded selection metadata.
 * No activation/weight/state is injected. Automatic execution stays unchanged. */
static int select_probe(void*c,void*r,const BnGPUIndexerSelectResidentPlan*p){
    int result=real_select(c,r,p);if(result)return result;
    int layer=(int)fused_calls/2;assert(layer%4==3);
    if(replay_dir){unsigned phase=(frame_number-1)%capture_frames;char name[512];
        if(p->scores.count){snprintf(name,sizeof(name),"%s-%u/layer%d-scores",replay_prefix,phase,layer);compare(r,p->scores,replay_dir,name);}
        snprintf(name,sizeof(name),"%s-%u/layer%d-pools",replay_prefix,phase,layer);compare(r,p->pools,replay_dir,name);
        if(capture_dir&&frame_number<=capture_frames)save_view(r,layer,"native_selection",p->out);
        snprintf(name,sizeof(name),"%s-%u/layer%d-selection",replay_prefix,phase,layer);
        void*indices=read_file(replay_dir,name,p->out.count*4);assert(!gpu->resident_write(r,p->out,indices));free(indices);
    }
    if(index_diagnostics&&capture_dir&&frame_number<=capture_frames){
        if(p->scores.count)save_view(r,layer,"scores",p->scores);
        save_view(r,layer,"pools",p->pools);save_view(r,layer,"selection",p->out);
    }
    return 0;
}
static int frame(BnBackendSession *s,const int32_t*ids,int n,int pos,int prefill,uint16_t*out){
    frame_number++;pre_calls=fused_calls=acquire_calls=forced_busy=0;BnTransformerBF16GLMRequestInfo q;
    int trace=getenv("BN_GLM_REQUEST_TRACE")&&frame_number==capture_frames;
    if(trace){assert(!capture_dir);assert(!cudaProfilerStart());}
    assert(!bn_transformer_bf16_glm_begin(s,ids,n,pos,prefill));
    BnTransformerBF16GLMLayerStateViews state_view;
    assert(bn_transformer_bf16_glm_layer_views(s,0,&state_view)==-1);assert(bn_transformer_bf16_glm_complete(s)==-1);
    assert(bn_transformer_bf16_glm_restart(s)==-1);assert(bn_transformer_bf16_glm_begin(s,ids,n,pos,prefill)==-1);
    int status;size_t retries=0;while((status=bn_transformer_bf16_glm_submit(s))==BN_GPU_MOE_BANK_BUSY){
        assert(!bn_transformer_bf16_glm_info(s,&q)&&q.active&&!q.failed&&q.position==pos);retries++;thrd_yield();
    }
    if(status){assert(!bn_transformer_bf16_glm_info(s,&q));fprintf(stderr,"frame failed layer%d pos%d nt%d\n",q.layer,pos,n);abort();}
    assert(pre_calls==1&&fused_calls==89);assert(!bn_transformer_bf16_glm_info(s,&q)&&q.position==pos&&q.active&&q.head_queued&&q.layer==45);
    assert(!gpu->resident_read(q.request,q.logits,out));assert(bn_transformer_bf16_glm_submit(s)==-1);
    int best=0;float maximum=-INFINITY;for(int j=0;j<154880;++j){uint32_t bits=(uint32_t)out[j]<<16;float v;memcpy(&v,&bits,4);assert(isfinite(v));if(v>maximum){maximum=v;best=j;}}
    assert(!bn_transformer_bf16_glm_complete(s));assert(!bn_transformer_bf16_glm_info(s,&q)&&q.position==pos+n&&!q.active&&!q.failed);
    if(trace)assert(!cudaProfilerStop());
    for(int l=0;l<45;++l){BnTransformerBF16GLMLayerStateViews v;
        assert(!bn_transformer_bf16_glm_layer_views(s,l,&v)&&v.layer==l&&v.position==pos+n);
        if(l%4==3){assert(!v.state.count&&!v.history.count&&v.latent.count==(size_t)((capacity+63)/64)*64*656&&v.index.count==(size_t)(((capacity+3)/4+63)/64)*8448&&v.tail.count==1024);}
        else{assert(v.state.count==1048576&&v.history.count==73728&&!v.latent.count&&!v.index.count&&!v.tail.count);}
        if(capture_dir&&capture_states&&frame_number<=capture_frames){
            if(l%4==3){save_view(q.request,l,"latent",v.latent);save_view(q.request,l,"index",v.index);save_view(q.request,l,"tail",v.tail);}
            else{save_view(q.request,l,"history",v.history);save_view(q.request,l,"state",v.state);}
        }
    }
    if(force_busy)assert(forced_busy&&retries>=forced_busy);
    printf("GLM full frame pos%d nt%d greedy%d logit%.6g retries%zu forced%u IObytes%zu\n",pos,n,best,maximum,retries,forced_busy,q.source.io_bytes);fflush(stdout);return best;
}
int main(int argc,char**argv){
    assert(argc==5);schedule_init();capture_frames=(unsigned)nchunks+1;capture_states=getenv("BN_GLM_REQUEST_STATES")!=NULL;BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModel model={0};assert(!bn_model_load_topology(&model,f,capacity));
    const BnModelAnatomy*a=bn_model_anatomy(&model);size_t nmaps=0;
    const BnMoEExpertMap*maps=bn_model_topology_experts(&model,&nmaps);assert(a&&maps&&nmaps==45);
    gpu=bn_gpu_cuda_create();assert(gpu);BnBackendModel*b=bn_model_backend(&model);assert(b);
    assert(!bn_model_prepare_bf16_topology(&model,gpu,(size_t)12*1024*1024*1024,256u*1024*1024));
    assert(!bn_model_set_moe_shard_files(&model,f->owned_maps,f->n_shards));
    /* A maximum-capacity cache accepts all heterogeneous layer profiles. Its
     * shared payload is deliberately separate from request admission. */
    size_t gate=0,down=0;const BnBackendBF16GLMModel*w=bn_backend_layout_bf16_glm_model(b);
    for(int l=3;l<45;++l){if(w->layers[l].moe.gate_bytes>gate)gate=w->layers[l].moe.gate_bytes;if(w->layers[l].moe.down_bytes>down)down=w->layers[l].moe.down_bytes;}
    BnMoEIO *io=bn_model_moe_io(&model);io->cache=bn_moe_cache_create(128u*1024*1024,gate,gate,down);assert(io->cache);
    real_acquire=gpu->moe_bank_acquire;real_pre=gpu->mhc_pre_bf16_resident;real_fused=gpu->mhc_fused_post_pre_bf16_resident;
    real_select=gpu->indexer_select_resident;index_diagnostics=getenv("BN_GLM_REQUEST_INDEX_DIAGNOSTICS")!=NULL;
    replay_dir=getenv("BN_GLM_REQUEST_REPLAY_DIRECTORY");replay_prefix=getenv("BN_GLM_REQUEST_REPLAY_PREFIX");if(!replay_prefix)replay_prefix="first";
    assert(!strcmp(replay_prefix,"first")||!strcmp(replay_prefix,"repeat"));
    gpu->indexer_select_resident=select_probe;
    real_post=gpu->mhc_post_bf16_resident;capture_dir=getenv("BN_GLM_REQUEST_CAPTURE");gpu->mhc_post_bf16_resident=post;
    real_alloc=gpu->resident_alloc;real_sync=gpu->resident_synchronize;gpu->resident_alloc=alloc_probe;gpu->resident_synchronize=sync_probe;
    BnBackendSession*s=bn_backend_session_create();assert(s);size_t db=(size_t)16*1024*1024*1024,hb=256u*1024*1024;
    assert(bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,1,hb)==-1&&!alloc_calls&&!bn_backend_session_has_gpu_resources(s));
    assert(bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,db,1)==-1&&!alloc_calls&&!bn_backend_session_has_gpu_resources(s));
    fail_alloc=17;assert(bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,db,hb)==-1&&alloc_calls==17&&!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
    alloc_calls=0;fail_alloc=0;assert(!bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,db,hb));unsigned allocations=alloc_calls;
    BnTransformerBF16GLMRequestInfo q;assert(!bn_transformer_bf16_glm_info(s,&q));size_t exact_device=q.device_bytes,exact_host=q.host_bytes;
    printf("GLM prepared device%zu host%zu allocations%u\n",exact_device,exact_host,allocations);fflush(stdout);
    assert(bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,db,hb)==-1);
    bn_backend_session_free(s);s=bn_backend_session_create();assert(s);alloc_calls=0;
    assert(bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,exact_device-1,exact_host)==-1&&!alloc_calls&&!bn_backend_session_has_gpu_resources(s));
    assert(bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,exact_device,exact_host-1)==-1&&!alloc_calls&&!bn_backend_session_has_gpu_resources(s));
    assert(!bn_transformer_bf16_glm_prepare(s,b,io,max_tokens,capacity,exact_device,exact_host));alloc_calls=0;
    gpu->moe_bank_acquire=acquire;gpu->mhc_pre_bf16_resident=pre;gpu->mhc_fused_post_pre_bf16_resident=fused;
    int32_t *fixture=read_file(argv[2],"ids",12*4),*ids=malloc((size_t)(prompt_tokens<12?12:prompt_tokens)*4);assert(ids);
    memcpy(ids,fixture,12*4);free(fixture);
    for(int j=12;j<prompt_tokens;++j)ids[j]=(int32_t)(((uint64_t)j*7919+(uint64_t)j*j+713)%154880);
    int nt=chunks[0];char dd[2048],md[2048];
    snprintf(dd,sizeof(dd),"%s/0-%d",argv[2],nt==1?5:0);snprintf(md,sizeof(md),"%s/0-%d",argv[3],nt==1?5:0);dense_dir=dd;mla_dir=md;
    int32_t bad=-1;assert(bn_transformer_bf16_glm_begin(s,&bad,1,0,1)==-1);bad=154880;assert(bn_transformer_bf16_glm_begin(s,&bad,1,0,1)==-1);
    assert(bn_transformer_bf16_glm_begin(s,ids,max_tokens+1,0,1)==-1&&bn_transformer_bf16_glm_begin(s,ids,1,1,1)==-1);
    BnTransformerBF16GLMLayerStateViews v;
    assert(bn_transformer_bf16_glm_layer_views(NULL,0,&v)==-1);
    assert(bn_transformer_bf16_glm_layer_views(s,-1,&v)==-1&&bn_transformer_bf16_glm_layer_views(s,45,&v)==-1&&bn_transformer_bf16_glm_layer_views(s,0,NULL)==-1);
    uint16_t *logits=malloc(154880*2),*gold=malloc((size_t)capture_frames*154880*2);assert(logits&&gold);
    for(int replay=0;replay<2;++replay){force_busy=replay;int pos=0;int32_t token=0;
        for(int phase=0;phase<=nchunks;++phase){int prefill=phase<nchunks,t=prefill?chunks[phase]:1;
            check_prefix=prefill&&phase==0&&nchunks==1&&(chunks[0]==1||chunks[0]==9);
            token=frame(s,prefill?ids+pos:&token,t,pos,prefill,logits);pos+=t;
            uint16_t*expected=gold+(size_t)phase*154880;
            if(replay)assert(!memcmp(logits,expected,154880*2));else memcpy(expected,logits,154880*2);
        }
        assert(!alloc_calls);assert(!bn_transformer_bf16_glm_restart(s));assert(!bn_transformer_bf16_glm_info(s,&q)&&!q.position&&!q.parity);
    }
    /* A failed final fence preserves all logical positions/generations and
     * prevents restart or reuse. The teardown's real fence drains queued work. */
    force_busy=0;check_prefix=0;replay_dir=NULL;index_diagnostics=0;assert(!bn_transformer_bf16_glm_begin(s,ids,1,0,1));int status;
    while((status=bn_transformer_bf16_glm_submit(s))==BN_GPU_MOE_BANK_BUSY){thrd_yield();}
    assert(!status);
    fail_sync=1;assert(bn_transformer_bf16_glm_complete(s)==-1);fail_sync=0;assert(!bn_transformer_bf16_glm_info(s,&q)&&q.failed&&!q.position&&!q.parity);
    assert(bn_transformer_bf16_glm_restart(s)==-1&&bn_transformer_bf16_glm_begin(s,ids,1,0,1)==-1&&bn_transformer_bf16_glm_layer_views(s,0,&v)==-1);
    bn_backend_session_free(s);gpu->moe_bank_acquire=real_acquire;gpu->mhc_pre_bf16_resident=real_pre;gpu->mhc_fused_post_pre_bf16_resident=real_fused;
    gpu->indexer_select_resident=real_select;gpu->mhc_post_bf16_resident=real_post;gpu->resident_alloc=real_alloc;gpu->resident_synchronize=real_sync;
    FILE*out=fopen(argv[4],"wb");assert(out&&fwrite(gold,2,(size_t)capture_frames*154880,out)==(size_t)capture_frames*154880);fclose(out);
    free(logits);free(gold);free(ids);bn_moe_cache_free(io->cache);io->cache=NULL;bn_model_free(&model);bn_gpu_cuda_destroy(gpu);bn_gguf_free(f);
    puts("GLM whole-request integration/replay/lifecycle PASSED (whole-model independent oracle remains pending)");return 0;
}
