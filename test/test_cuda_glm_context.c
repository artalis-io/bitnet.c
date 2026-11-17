/* Actual-checkpoint qualification of shared BnSession/prefill/generate dispatch.
 * Capture callbacks observe native outputs; they never inject activations. */
#include "model_internal.h"
#include "model_arch.h"
#include "session_internal.h"
#include "transformer.h"
#include "generate.h"
#include "sampler.h"
#include "tokenizer.h"
#include "moe.h"
#include "gpu_cuda.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static BnGPUBackend *gpu;
static BnModel *model;
static BnSession *session;
static const char *capture_dir;
static int phase=-1,fused_calls,post_calls,alloc_calls,fail_alloc,fail_read,fail_sync;
static unsigned acquire_calls,forced_busy;
static int force_busy,index_diagnostics;
static int frame_sizes[8192],frame_positions[8192],head_frames[8192],state_frames[8192];
static int (*real_alloc)(void*,int,size_t,BnGPUResidentView*);
static int (*real_read)(void*,BnGPUResidentView,void*);
static int (*real_sync)(void*);
static int (*real_pre)(void*,void*,const BnGPUMHCPreResidentPlan*);
static int (*real_fused)(void*,void*,const BnGPUMHCFusedResidentPlan*);
static int (*real_post)(void*,void*,const BnGPUMHCPostResidentPlan*);
static int (*real_select)(void*,void*,const BnGPUIndexerSelectResidentPlan*);
static int (*real_acquire)(void*,unsigned,int,int,BnGPUMoEBankLease*);
static void save(const char *name,const void *data,size_t bytes) {
    if(!capture_dir)return;
    char path[2048];snprintf(path,sizeof(path),"%s/%s.bin",capture_dir,name);
    FILE *f=fopen(path,"wb");assert(f&&fwrite(data,1,bytes,f)==bytes);assert(!fclose(f));
}
static void view(void *request,int layer,const char *role,BnGPUResidentView v) {
    if(!capture_dir||!v.count)return;
    size_t bytes=v.count*bn_quant_format_resident_scalar_bytes(v.type);
    void *data=malloc(bytes);assert(data&&!gpu->resident_read(request,v,data));
    char name[128];snprintf(name,sizeof(name),"%d-layer%d-%s",phase,layer,role);
    save(name,data,bytes);free(data);
}
static void tuple(void *r,int layer,BnGPUResidentView residual,BnGPUResidentView branch,
    BnGPUResidentView post,BnGPUResidentView comb) {
    view(r,layer,"residual",residual);view(r,layer,"out",branch);
    view(r,layer,"post",post);view(r,layer,"comb",comb);
}
static int allocate(void *r,int type,size_t count,BnGPUResidentView *out) {
    return ++alloc_calls==fail_alloc?-1:real_alloc(r,type,count,out);
}
static int readback(void *r,BnGPUResidentView v,void *out) {
    if(fail_read&&v.type==BN_GGUF_TENSOR_BF16&&v.count==154880)return -1;
    return real_read(r,v,out);
}
static int synchronize(void *r) {return fail_sync?-1:real_sync(r);}
static int acquire(void *r,unsigned slot,int gate,int down,BnGPUMoEBankLease *out) {
    if(force_busy&&++acquire_calls%2){forced_busy++;return BN_GPU_MOE_BANK_BUSY;}
    return real_acquire(r,slot,gate,down,out);
}
static int pre(void *c,void *r,const BnGPUMHCPreResidentPlan *p) {
    phase++;fused_calls=post_calls=0;
    if(capture_dir){assert(phase<8192);frame_sizes[phase]=p->n_tokens;
        frame_positions[phase]=bn_transformer_bf16_position(bn_session_backend(session));}
    return real_pre(c,r,p);
}
static int fused(void *c,void *r,const BnGPUMHCFusedResidentPlan *p) {
    if(!(++fused_calls&1))tuple(r,fused_calls/2-1,p->previous_residual,
        p->previous_branch,p->previous_post,p->previous_comb);
    return real_fused(c,r,p);
}
static int post(void *c,void *r,const BnGPUMHCPostResidentPlan *p) {
    post_calls++;if(capture_dir)head_frames[phase]=1;tuple(r,44,p->residual,p->branch,p->post,p->comb);return real_post(c,r,p);
}
static int select_indices(void *c,void *r,const BnGPUIndexerSelectResidentPlan *p) {
    int result=real_select(c,r,p);if(result)return result;
    if(!index_diagnostics)return 0;
    int layer=fused_calls/2;view(r,layer,"scores",p->scores);
    view(r,layer,"pools",p->pools);view(r,layer,"selection",p->out);return 0;
}
static void states(void) {
    if(capture_dir)state_frames[phase]=1;
    BnBackendSession *b=bn_session_backend(session);
    BnTransformerBF16GLMRequestInfo q;assert(!bn_transformer_bf16_glm_info(b,&q));
    for(int l=0;l<45;++l){BnTransformerBF16GLMLayerStateViews v;
        assert(!bn_transformer_bf16_glm_layer_views(b,l,&v)&&v.position==session->pos);
        view(q.request,l,"history",v.history);view(q.request,l,"state",v.state);
        view(q.request,l,"latent",v.latent);view(q.request,l,"index",v.index);view(q.request,l,"tail",v.tail);
    }
}
static int head(void) {
    int best=0;float *logits=session->state.logits;
    uint16_t *raw=malloc(154880*2);assert(raw);
    for(int j=0;j<154880;++j){uint32_t bits;assert(isfinite(logits[j]));
        memcpy(&bits,&logits[j],4);assert(!(bits&65535));raw[j]=(uint16_t)(bits>>16);
        if(logits[j]>logits[best])best=j;
    }
    char name[64];snprintf(name,sizeof(name),"%d-logits",phase);save(name,raw,154880*2);free(raw);
    assert(fused_calls==89&&post_calls==1);states();return best;
}
static void committed(int position,const int *ids,int nt,int start) {
    BnBackendSession *b=bn_session_backend(session);
    assert(session->pos==position&&bn_transformer_bf16_position(b)==position);
    for(int j=0;j<nt;++j)assert(session->state.token_history[start+j]==ids[j]);
}
static int emitted[2],nemitted;
static int collect(const char *piece,int token,void *unused) {
    (void)unused;assert(piece&&nemitted<2);emitted[nemitted++]=token;return 0;
}
static void state_equivalence(const int *ids,int capacity) {
    /* No-logits execution must preserve host output while producing exactly
     * the same committed state and next logits as the full-head path. */
    size_t sizes[45][3];void *copies[45][3]={{0}};
    BnBackendSession *b=bn_session_backend(session);
    assert(!bn_transformer_bf16_session_restart(model,session));
    assert(bn_transformer_prefill(model,session,ids,3,0));
    assert(bn_transformer_forward(model,session,ids[3],3));
    size_t head_bytes=154880*4;void *expected=malloc(head_bytes);assert(expected);
    memcpy(expected,session->state.logits,head_bytes);
    BnTransformerBF16GLMRequestInfo q;assert(!bn_transformer_bf16_glm_info(b,&q));
    for(int l=0;l<45;++l){BnTransformerBF16GLMLayerStateViews v;
        assert(!bn_transformer_bf16_glm_layer_views(b,l,&v));
        BnGPUResidentView a[3]={v.history.count?v.history:v.latent,v.state.count?v.state:v.index,v.tail};
        for(int j=0;j<3;++j){sizes[l][j]=a[j].count*bn_quant_format_resident_scalar_bytes(a[j].type);
            if(sizes[l][j]){copies[l][j]=malloc(sizes[l][j]);assert(copies[l][j]&&!gpu->resident_read(q.request,a[j],copies[l][j]));}}
    }
    assert(!bn_transformer_bf16_session_restart(model,session));
    for(int j=0;j<154880;++j)session->state.logits[j]=123.25f;
    post_calls=0;assert(!bn_prefill_no_logits(model,session,ids,3,0,0));
    assert(!post_calls);committed(3,ids,3,0);
    for(int j=0;j<154880;++j)assert(session->state.logits[j]==123.25f);
    assert(bn_transformer_forward(model,session,ids[3],3));
    assert(!memcmp(expected,session->state.logits,head_bytes));free(expected);
    assert(!bn_transformer_bf16_glm_info(b,&q));
    for(int l=0;l<45;++l){BnTransformerBF16GLMLayerStateViews v;
        assert(!bn_transformer_bf16_glm_layer_views(b,l,&v));
        BnGPUResidentView a[3]={v.history.count?v.history:v.latent,v.state.count?v.state:v.index,v.tail};
        for(int j=0;j<3;++j)if(sizes[l][j]){void *actual=malloc(sizes[l][j]);assert(actual&&!gpu->resident_read(q.request,a[j],actual));
            assert(!memcmp(actual,copies[l][j],sizes[l][j]));free(actual);free(copies[l][j]);}
    }
    assert(bn_transformer_bf16_capacity(b)==capacity);
}
int main(int argc,char **argv) {
    assert(argc==3);capture_dir=argv[2];index_diagnostics=getenv("BN_GLM_CONTEXT_INDEX_DIAGNOSTICS")!=NULL;
    int chunks[8191],nchunks=0,total=0,batch=3;
    const char *text=getenv("BN_GLM_CONTEXT_SCHEDULE");if(!text)text="3,6";
    for(;;){char *end;long n=strtol(text,&end,10);assert(end!=text&&n>=1&&n<=8191&&nchunks<8191);
        chunks[nchunks++]=(int)n;total+=(int)n;assert(total<=8191);
        if(n>batch)batch=(int)n;
        if(!*end)break;
        assert(*end==','&&end[1]);text=end+1;
    }
    if(getenv("BN_GLM_CONTEXT_BATCH"))batch=atoi(getenv("BN_GLM_CONTEXT_BATCH"));
    assert(batch>=3);
    int capacity=((total+1+63)/64)*64;
    BnGGUFFile *file=bn_gguf_open_file(argv[1]);assert(file);
    BnModel owner={0};model=&owner;assert(!bn_model_load_topology(model,file,capacity));
    gpu=bn_gpu_cuda_create();assert(gpu&&!bn_model_prepare_bf16_topology(model,gpu,(size_t)12*1024*1024*1024,256u*1024*1024));
    assert(!bn_model_set_moe_shard_files(model,file->owned_maps,file->n_shards));
    const BnBackendBF16GLMModel *weights=bn_backend_layout_bf16_glm_model(bn_model_backend(model));
    size_t gate=0,down=0;for(int l=3;l<45;++l){if(weights->layers[l].moe.gate_bytes>gate)gate=weights->layers[l].moe.gate_bytes;
        if(weights->layers[l].moe.down_bytes>down)down=weights->layers[l].moe.down_bytes;}
    BnMoEIO *io=bn_model_moe_io(model);io->cache=bn_moe_cache_create(128u*1024*1024,gate,gate,down);assert(io->cache);
    real_alloc=gpu->resident_alloc;real_read=gpu->resident_read;real_sync=gpu->resident_synchronize;
    real_pre=gpu->mhc_pre_bf16_resident;real_fused=gpu->mhc_fused_post_pre_bf16_resident;real_post=gpu->mhc_post_bf16_resident;
    real_select=gpu->indexer_select_resident;real_acquire=gpu->moe_bank_acquire;
    gpu->resident_alloc=allocate;gpu->resident_read=readback;gpu->resident_synchronize=synchronize;
    gpu->mhc_pre_bf16_resident=pre;gpu->mhc_fused_post_pre_bf16_resident=fused;gpu->mhc_post_bf16_resident=post;
    gpu->indexer_select_resident=select_indices;gpu->moe_bank_acquire=acquire;
    session=bn_session_create_backend(model,NULL);assert(session);
    BnBackendSession *backend=bn_session_backend(session);size_t db=(size_t)16*1024*1024*1024,hb=256u*1024*1024;
    assert(!bn_transformer_bf16_max_tokens(backend)&&bn_transformer_bf16_position(backend)==-1);
    BnModel copy=*model;assert(bn_transformer_bf16_glm_prepare_context(&copy,session,batch,capacity,db,hb)==-1&&!alloc_calls);
    BnModel tiny={0};tiny.config.vocab_size=1;tiny.config.seq_len=4;
    BnSession *wrong=bn_session_create_backend(&tiny,NULL);assert(wrong);
    assert(bn_transformer_bf16_glm_prepare_context(model,wrong,batch,capacity,db,hb)==-1&&!alloc_calls);bn_session_free(wrong,NULL);
    assert(bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,1,hb)==-1&&!alloc_calls);
    assert(bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,db,1)==-1&&!alloc_calls);
    fail_alloc=17;assert(bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,db,hb)==-1&&alloc_calls==17);
    assert(!bn_backend_session_has_gpu_resources(backend)&&!bn_backend_session_has_execution_resources(backend));
    fail_alloc=alloc_calls=0;assert(!bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,db,hb));
    BnTransformerBF16GLMRequestInfo info;assert(!bn_transformer_bf16_glm_info(backend,&info));
    size_t exact_device=info.device_bytes,exact_host=info.host_bytes;
    bn_session_free(session,NULL);session=bn_session_create_backend(model,NULL);assert(session);backend=bn_session_backend(session);alloc_calls=0;
    assert(bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,exact_device-1,exact_host)==-1&&!alloc_calls);
    assert(bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,exact_device,exact_host-1)==-1&&!alloc_calls);
    assert(!bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,exact_device,exact_host));alloc_calls=0;
    assert(bn_transformer_bf16_max_tokens(backend)==batch&&bn_transformer_bf16_capacity(backend)==capacity);
    int count=total+1;if(count<batch+1)count=batch+1;if(count<4)count=4;
    int *ids=malloc((size_t)count*sizeof(*ids));assert(ids);
    int fixture[]={0,154879,1,713,32767,100000,42,8192,12345,17,154878,256};
    for(int j=0;j<count;++j)ids[j]=j<12?fixture[j]:(int)(((uint64_t)j*7919+(uint64_t)j*j+713)%154880);
    int old=ids[batch];ids[batch]=154880;assert(!bn_prefill(model,session,ids,batch+1,0,0));ids[batch]=old;
    assert(session->pos==0&&phase==-1&&bn_transformer_bf16_position(backend)==0);
    assert(!bn_transformer_forward(&copy,session,ids[0],0)&&phase==-1);
    assert(!bn_prefill_no_logits(model,session,NULL,0,0,0)&&phase==-1);
    int *greedy=malloc((size_t)(nchunks+1)*sizeof(*greedy));assert(greedy);
    BnTokenizer tok;assert(!bn_tokenizer_init(&tok,file)&&tok.vocab_size==154880);
    BnSampler sampler;assert(!bn_sampler_init(&sampler,154880,0.0f,.9f,17));
    nemitted=0;int invalid_pos=0;
    assert(bn_generate(&copy,session,&tok,&sampler,1,&invalid_pos,collect,NULL,NULL,NULL)==-2);
    invalid_pos=1;
    assert(bn_generate(model,session,&tok,&sampler,1,&invalid_pos,collect,NULL,NULL,NULL)==-2);
    assert(!nemitted&&phase==-1&&!session->pos);
    bn_session_reset(session,&copy);
    assert(bn_transformer_bf16_max_tokens(backend)==batch&&bn_backend_session_has_gpu_resources(backend));
    int qualified_frames=0;
    for(int replay=0;replay<2;++replay){force_busy=replay;acquire_calls=forced_busy=0;phase=-1;
        if(replay)capture_dir=NULL;
        int pos=0;
        for(int chunk=0;chunk<nchunks;++chunk){assert(bn_prefill(model,session,ids+pos,chunks[chunk],pos,0));
            committed(pos+chunks[chunk],ids+pos,chunks[chunk],pos);pos+=chunks[chunk];
            int token=head();if(!replay)greedy[chunk]=token;else assert(token==greedy[chunk]);}
        nemitted=0;assert(bn_generate(model,session,&tok,&sampler,2,&pos,collect,NULL,NULL,NULL)==2&&nemitted==2);
        assert(pos==total+1&&emitted[0]==greedy[nchunks-1]);
        int token=head();assert(token==emitted[1]);if(!replay)greedy[nchunks]=token;else assert(token==greedy[nchunks]);
        committed(total+1,&emitted[0],1,total);if(replay)assert(forced_busy);
        if(!replay)qualified_frames=phase+1;
        assert(!alloc_calls&&!bn_transformer_bf16_session_restart(model,session));
        assert(!session->pos&&bn_transformer_bf16_position(backend)==0);
    }
    /* Capture is disabled for lifecycle stress beyond the qualified sequence. */
    capture_dir=NULL;force_busy=0;state_equivalence(ids,capacity);assert(!alloc_calls);
    for(int kind=0;kind<2;++kind){assert(!bn_transformer_bf16_session_restart(model,session));
        for(int j=0;j<154880;++j)session->state.logits[j]=123.25f;
        fail_read=kind==0;fail_sync=kind==1;
        assert(!bn_transformer_forward(model,session,ids[3],0));fail_read=fail_sync=0;
        assert(!bn_transformer_bf16_glm_info(backend,&info)&&info.failed&&!info.position);
        assert(!session->pos&&!session->state.token_history[0]&&bn_transformer_bf16_position(backend)==-1);
        for(int j=0;j<154880;++j)assert(session->state.logits[j]==123.25f);
        assert(bn_transformer_bf16_session_restart(model,session)==-1);
        bn_session_reset(session,model);
        assert(!bn_transformer_bf16_glm_prepare_context(model,session,batch,capacity,exact_device,exact_host));
    }
    char path[2048];snprintf(path,sizeof(path),"%s/manifest.json",argv[2]);
    FILE *meta=fopen(path,"w");assert(meta);
    fprintf(meta,"{\"prompt_tokens\":%d,\"batch\":%d,\"capacity\":%d,\"device_bytes\":%zu,\"host_bytes\":%zu,\"generation\":[%d,%d],\"frames\":[",total,batch,capacity,exact_device,exact_host,emitted[0],emitted[1]);
    for(int i=0;i<qualified_frames;++i)fprintf(meta,"%s{\"nt\":%d,\"position\":%d,\"head\":%d,\"state\":%d}",i?",":"",frame_sizes[i],frame_positions[i],head_frames[i],state_frames[i]);
    fprintf(meta,"]}\n");assert(!fclose(meta));
    printf("GLM common session/generate PASSED: prompt%d frames%d batch%d device%zu host%zu\n",total,qualified_frames,batch,exact_device,exact_host);
    bn_sampler_free(&sampler);bn_tokenizer_free(&tok);free(ids);free(greedy);bn_session_free(session,NULL);
    gpu->resident_alloc=real_alloc;gpu->resident_read=real_read;gpu->resident_synchronize=real_sync;
    gpu->mhc_pre_bf16_resident=real_pre;gpu->mhc_fused_post_pre_bf16_resident=real_fused;gpu->mhc_post_bf16_resident=real_post;
    gpu->indexer_select_resident=real_select;gpu->moe_bank_acquire=real_acquire;
    bn_moe_cache_free(io->cache);io->cache=NULL;bn_model_free(model);bn_gpu_cuda_destroy(gpu);bn_gguf_free(file);return 0;
}
