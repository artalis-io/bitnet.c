/* Actual Flash layer3 native projection -> indexer/selection -> paged attention
 * -> output projection, with one outer fence per phase and persistent caches.
 * Usage: probe CASE_INDEX CAPTURE_DIRECTORY [PROFILE_PHASE] [--model FIRST_SHARD].
 * Decoded BF16 weights
 * and deterministic canonical selection; this is a layer, not full-model proof. */
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include "gguf.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "model_internal.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
static BnGPUResidentView slice(BnGPUResidentView v,size_t off,size_t count){return bn_gpu_resident_view_slice(v,off,count);}
static void *load(const char *path,size_t *bytes) {
    FILE *f=fopen(path,"rb");assert(f&&!fseek(f,0,SEEK_END));long n=ftell(f);
    assert(n>0&&!fseek(f,0,SEEK_SET));*bytes=(size_t)n;void *v=malloc(*bytes);
    assert(v&&fread(v,1,*bytes,f)==*bytes&&fgetc(f)==EOF);fclose(f);return v;
}
static void path(char *out,size_t n,const char *dir,const char *name,const char *suffix) {
    int len=snprintf(out,n,"%s/%s%s",dir,name,suffix);assert(len>0&&(size_t)len<n);
}
static BnGPUResidentView alloc(BnGPUBackend *g,void *r,int type,size_t count,int zero) {
    BnGPUResidentView v;assert(!g->resident_alloc(r,type,count,&v));if(zero)assert(!g->resident_zero(r,v));return v;
}
static void *weight(BnGPUBackend *g,const char *dir,const char **names,int n,int rows,int cols) {
    size_t total=(size_t)rows*cols*2,offset=0;unsigned char *stack=malloc(total);assert(stack);
    for(int i=0;i<n;i++) {
        char file[4096];path(file,sizeof(file),dir,names[i],".bf16");size_t bytes;void *v=load(file,&bytes);
        assert(bytes<=total-offset);memcpy(stack+offset,v,bytes);offset+=bytes;free(v);
    }
    assert(offset==total);void *w=g->buffer_create_quant_only(g->ctx,stack,total,BN_GGUF_TENSOR_BF16,rows,cols);
    free(stack);assert(w);return w;
}
static void *one_weight(BnGPUBackend *g,const char *dir,const char *name,int rows,int cols){return weight(g,dir,&name,1,rows,cols);}
static size_t compare(BnGPUBackend *g,void *r,BnGPUResidentView v,const char *dir,
    const char *prefix,const char *stage,int width) {
    char file[4096],suffix[128];int n=snprintf(suffix,sizeof(suffix),"-%s.bin",stage);assert(n>0&&(size_t)n<sizeof(suffix));
    path(file,sizeof(file),dir,prefix,suffix);size_t bytes;void *gold=load(file,&bytes);assert(bytes==v.count*(size_t)width);
    void *actual=malloc(bytes);assert(actual&&!g->resident_read(r,v,actual));size_t differences=0;
    for(size_t i=0;i<v.count;i++)if(memcmp((char*)gold+i*width,(char*)actual+i*width,(size_t)width)) {
        if(differences<2)fprintf(stderr,"%s %s word%zu differs\n",prefix,stage,i);
        differences++;
    }
    if(differences)fprintf(stderr,"%s %s differences%zu\n",prefix,stage,differences);
    assert(!differences);free(gold);free(actual);return v.count;
}
typedef struct {void *main,*index,*out,*qn,*kn,*iqn,*ikn,*cs;} Weights;
typedef struct {
    BnBackendSession *session;void *request;int total,cap,blocks,groups,position,parity;
    BnGPUResidentView x,projected,q,k,v,gate,ip,iq,history[2],compressed,scores,indices;
    BnGPUResidentView kc,vc,slots,table,req,attention,out,workspace;
} Request;
static void init(Request *s,BnGPUBackend *g,int total,int cap) {
    memset(s,0,sizeof(*s));s->total=total;s->cap=cap;s->blocks=(total+8+15)/16+1;s->groups=(total+8)/4;
    s->session=bn_backend_session_create();assert(s->session);s->request=bn_gpu_backend_resident_request(g,s->session);assert(s->request);
    void *r=s->request;assert(!g->linear_bf16_reserve(g->ctx,r,0));
    BnGPUResidentView *views[]={&s->x,&s->projected,&s->q,&s->k,&s->v,&s->gate,&s->ip,&s->iq,&s->attention,&s->out};
    const size_t widths[]={2560,13312,6144,512,512,6144,640,512,6144,2560};
    for(int i=0;i<10;i++)*views[i]=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*widths[i],0);
    for(int i=0;i<2;i++)s->history[i]=alloc(g,r,BN_GGUF_TENSOR_BF16,512,1);
    s->compressed=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)s->groups*128,1);
    s->kc=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)s->blocks*8192,1);
    s->vc=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)s->blocks*8192,1);
    s->scores=alloc(g,r,BN_GGUF_TENSOR_F32,(size_t)cap*s->groups,0);
    s->indices=alloc(g,r,BN_GGUF_TENSOR_I32,(size_t)cap*2052,0);
    s->slots=alloc(g,r,BN_GGUF_TENSOR_I32,(size_t)cap+1,0);
    s->req=alloc(g,r,BN_GGUF_TENSOR_I32,(size_t)cap,1);
    s->table=alloc(g,r,BN_GGUF_TENSOR_I32,(size_t)s->blocks-1,0);
    int *table=malloc(((size_t)s->blocks-1)*4);assert(table);
    for(int i=0;i<s->blocks-1;i++)table[i]=i+1;
    assert(!g->resident_write(r,s->table,table));free(table);
    size_t bytes;assert(!g->qsa_attention_bf16_workspace_size(g->ctx,cap,&bytes));
    s->workspace=alloc(g,r,BN_GGUF_TENSOR_F32,bytes/4,0);
}
static void linear(BnGPUBackend *g,void *r,void *weight_ptr,int rows,int cols,int nt,
    BnGPUResidentView x,BnGPUResidentView out) {
    BnGPUBF16LinearResidentPlan p={0};p.n_tokens=nt;p.rows=rows;p.cols=cols;
    p.weight.type=BN_GGUF_TENSOR_BF16;p.weight.decoded=weight_ptr;p.x=x;p.out=out;
    assert(!g->linear_bf16_resident(g->ctx,r,&p));
}
static void submit(BnGPUBackend *g,Request *s,const Weights *w,int nt,int prefill) {
    void *r=s->request;size_t n=(size_t)nt;int pos=s->position;
    BnGPUResidentView x=slice(s->x,0,n*2560),merged=slice(s->projected,0,n*13312);
    linear(g,r,w->main,13312,2560,nt,x,merged);
    BnGPUQKPrepResidentPlan q={0};q.n_tokens=nt;q.n_heads=24;q.n_kv_heads=2;q.head_size=256;
    q.rotary_dim=64;q.position=pos;q.eps=1e-6f;q.weight_offset=1;q.q_weight=w->qn;q.k_weight=w->kn;q.cos_sin=w->cs;
    q.q_gate_row_stride=q.k_in_row_stride=q.v_in_row_stride=13312;
    q.q_gate=slice(merged,0,(n-1)*13312+12288);q.k_in=slice(merged,12288,(n-1)*13312+512);q.v_in=slice(merged,12800,(n-1)*13312+512);
    q.q=slice(s->q,0,n*6144);q.k=slice(s->k,0,n*512);q.v=slice(s->v,0,n*512);q.gate=slice(s->gate,0,n*6144);
    assert(!g->qk_prep_bf16_resident(g->ctx,r,&q));
    BnGPUResidentView ip=slice(s->ip,0,n*640);linear(g,r,w->index,640,2560,nt,x,ip);
    BnGPUQSAIndexerResidentPlan ix={0};ix.n_tokens=nt;ix.position=pos;ix.eps=1e-6f;
    ix.q_weight=w->iqn;ix.k_weight=w->ikn;ix.cos_sin=w->cs;ix.projected=ip;
    ix.initial_history=s->history[s->parity];ix.next_history=s->history[1-s->parity];ix.q=slice(s->iq,0,n*512);
    int groups=(pos+nt)/4-pos/4;
    if(groups)ix.compressed=slice(s->compressed,(size_t)(pos/4)*128,(size_t)groups*128);
    assert(!g->qsa_indexer_bf16_resident(g->ctx,r,&ix));
    BnGPUQSAScoreResidentPlan sc={nt,pos,s->groups,ix.q,s->compressed,slice(s->scores,0,n*s->groups)};
    assert(!g->qsa_score_bf16_resident(g->ctx,r,&sc));
    BnGPUQSASelectResidentPlan se={nt,pos,s->groups,sc.scores,slice(s->indices,0,n*2052)};
    assert(!g->qsa_select_resident(g->ctx,r,&se));
    BnGPUPagedKVAppendResidentPlan ap={nt,s->blocks,16,512,q.k,q.v,slice(s->slots,1,n),s->kc,s->vc};
    assert(!g->paged_kv_append_bf16_resident(g->ctx,r,&ap));
    BnGPUQSAAttentionResidentPlan a={0};a.n_tokens=nt;a.use_prefill_config=prefill;a.num_cache_blocks=s->blocks;
    a.num_requests=1;a.table_width=s->blocks-1;a.q=q.q;a.k_cache=s->kc;a.v_cache=s->vc;a.indices=se.out;
    a.block_table=s->table;a.token_to_req=slice(s->req,0,n);a.gate=q.gate;a.out=slice(s->attention,0,n*6144);a.workspace=s->workspace;
    assert(!g->qsa_attention_bf16_resident(g->ctx,r,&a));
    linear(g,r,w->out,2560,6144,nt,a.out,slice(s->out,0,n*2560));
}
static uint16_t rounded_bf16(float x) {
    uint32_t u;memcpy(&u,&x,sizeof(u));return(uint16_t)((u+0x7fff+((u>>16)&1))>>16);
}
static uint64_t model_norm_hash(const BnModel *m) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(int l=0;l<m->config.n_layers;l++)if(m->weights.layers[l].block_kind==BN_LAYER_BLOCK_ATTENTION) {
        const BnLayerWeights *w=&m->weights.layers[l];
        const float *norms[]={w->attn.q_norm,w->attn.k_norm,w->indexer.q_norm,w->indexer.k_norm};
        for(int i=0;i<4;i++) {
            assert(norms[i]);const unsigned char *raw=(const unsigned char*)norms[i];
            size_t bytes=(i<2?256u:128u)*sizeof(float);
            for(size_t j=0;j<bytes;j++)h=(h^raw[j])*UINT64_C(1099511628211);
        }
    }
    return h;
}
static size_t verify_model_norms(BnGPUBackend *g,const BnModel *m,const BnBackendBF16FlashQSAModel *qsa) {
    size_t words=0;
    float offset=bn_model_backend_policy_zero_centered_norm_storage_offset(&m->config);
    for(int l=0;l<qsa->n_layers;l++)if(qsa->layers[l].valid) {
        const BnLayerWeights *source=&m->weights.layers[l];
        const float *cpu[]={source->attn.q_norm,source->attn.k_norm,source->indexer.q_norm,source->indexer.k_norm};
        const BnBackendBF16FlashQSAWeights *w=&qsa->layers[l];
        void *gpu[]={w->q_norm,w->k_norm,w->index_q_norm,w->index_k_norm};
        for(int i=0;i<4;i++) {
            int count=i<2?256:128;uint16_t ones[256],actual[256];float before[256];
            memcpy(before,cpu[i],(size_t)count*sizeof(float));
            for(int j=0;j<count;j++)ones[j]=0x3f80;
            assert(!bn_gpu_backend_rmsnorm_bf16_batch(g,actual,gpu[i],ones,1,count,0,0));
            for(int j=0;j<count;j++)assert(actual[j]==rounded_bf16(cpu[i][j]-offset));
            assert(!memcmp(before,cpu[i],(size_t)count*sizeof(float)));words+=(size_t)count;
        }
    }
    printf("QSA_MODEL_GAMMA_PASSED bf16_words%zu\n",words);return words;
}

int main(int argc,char **argv) {
    assert(argc>=3&&argc<=6);int model_mode=argc>=5;
    assert(!model_mode||!strcmp(argv[argc-2],"--model"));
    int profile=(argc==4||argc==6)?atoi(argv[3]):-1,profiled=0;
    BnGPUBackend *g=bn_gpu_cuda_create();assert(g&&g->paged_kv_append_bf16_resident&&g->qsa_attention_bf16_prepare);
    assert(!g->qsa_attention_bf16_prepare(g->ctx));Weights w={0};
    BnModel model={0};uint64_t cpu_norm_hash=0;BnGGUFFile *file=NULL;const BnBackendBF16FlashQSAModel *qsa=NULL;
    if(model_mode) {
        file=bn_gguf_open_file(argv[argc-1]);assert(file);
        assert(!bn_model_load_with_cpu_preparation(&model,file,262144,0,0,0));
        cpu_norm_hash=model_norm_hash(&model);
        BnBackendModel *owner=bn_model_backend(&model);assert(owner);
        const size_t budget=(size_t)12*1024*1024*1024,host=64u*1024*1024;
        assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&model.config,&model.weights,budget));
        assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&model.config,&model.weights,budget,host));
        assert(!bn_backend_layout_prepare_bf16_flash_gdn_model(owner,g,&model.config,&model.weights,budget,host));
        assert(!bn_backend_layout_prepare_bf16_flash_qsa_model(owner,g,&model.config,&model.weights,budget,host));
        qsa=bn_backend_layout_bf16_flash_qsa_model(owner);
        assert(qsa&&qsa->n_layers==48&&qsa->n_qsa_layers==12&&qsa->layers[3].valid&&qsa->rotary_capacity==262144);
        assert(!bn_backend_layout_bf16(owner)); /* Partial layouts do not admit a model. */
        assert(model_norm_hash(&model)==cpu_norm_hash);
        verify_model_norms(g,&model,qsa);
        const BnBackendBF16FlashQSAWeights *layer=&qsa->layers[3];
        w.main=layer->qkv.decoded;w.index=layer->index.decoded;w.out=layer->out.decoded;
        w.qn=layer->q_norm;w.kn=layer->k_norm;w.iqn=layer->index_q_norm;w.ikn=layer->index_k_norm;w.cs=qsa->rotary_cache;
        printf("QSA_MODEL_LAYOUT retained%zu peak%zu model%zu model_peak%zu host%zu layers%d\n",
            qsa->bytes,qsa->peak_bytes,qsa->model_bytes,qsa->model_peak_bytes,qsa->host_staging_bytes,qsa->n_qsa_layers);
    } else {
    const char *names[]={"attn_q.weight","attn_k.weight","attn_v.weight"},*index[]={"indexer.q_proj.weight","indexer.k_proj.weight"};
    w.main=weight(g,argv[2],names,3,13312,2560);w.index=weight(g,argv[2],index,2,640,2560);
    w.out=one_weight(g,argv[2],"attn_output.weight",2560,6144);
    w.qn=one_weight(g,argv[2],"attn_q_norm.weight",1,256);w.kn=one_weight(g,argv[2],"attn_k_norm.weight",1,256);
    w.iqn=one_weight(g,argv[2],"indexer.q_norm.weight",1,128);w.ikn=one_weight(g,argv[2],"indexer.k_norm.weight",1,128);
    /* Retained corrected capture's rotary cache is 8192 x 64 BF16. */
    w.cs=one_weight(g,argv[2],"cos-sin",8192,64);
    }
    FILE *f=fopen(argv[1],"r");assert(f);Request s={0};int ci,old=-1,total,cap,pos,nt,prefill,phases=0,cases=0;
    size_t bf16_words=0,index_words=0;char prefix[256];
    while(fscanf(f,"%d %d %d %d %d %d %255s",&ci,&total,&cap,&pos,&nt,&prefill,prefix)==7) {
        if(ci!=old){if(s.session)bn_backend_session_free(s.session);init(&s,g,total,cap);old=ci;cases++;}
        assert(nt>0&&nt<=s.cap&&pos==s.position&&total==s.total&&cap==s.cap);
        char input_file[4096];path(input_file,sizeof(input_file),argv[2],prefix,"-input.bin");size_t bytes;void *raw=load(input_file,&bytes);assert(bytes==(size_t)nt*2560*2);
        assert(!g->resident_write(s.request,slice(s.x,0,(size_t)nt*2560),raw));free(raw);
        int *slots=malloc((size_t)nt*4);assert(slots);for(int i=0;i<nt;i++)slots[i]=16+pos+i;
        assert(!g->resident_write(s.request,slice(s.slots,1,(size_t)nt),slots));free(slots);
        if(phases==profile) {
            submit(g,&s,&w,nt,prefill);assert(!g->resident_synchronize(s.request));
#ifdef BN_CAPTURE_CUDA_PROFILER
            assert(cudaProfilerStart()==cudaSuccess);
#else
            assert(!"build with BN_CAPTURE_CUDA_PROFILER");
#endif
        }
        submit(g,&s,&w,nt,prefill);assert(!g->resident_synchronize(s.request));
#ifdef BN_CAPTURE_CUDA_PROFILER
        if(phases==profile){assert(cudaProfilerStop()==cudaSuccess);profiled++;}
#endif
        /* Position and history bank commit only after the outer completion. */
        s.position+=nt;s.parity=1-s.parity;
        BnGPUResidentView views[]={s.x,s.projected,s.q,s.k,s.v,s.gate,s.iq,s.attention,s.out};
        const size_t widths[]={2560,13312,6144,512,512,6144,512,6144,2560};
        const char *stages[]={"input","projected","q","k","v","gate","index_q","attention","out"};
        for(int i=0;i<9;i++)bf16_words+=compare(g,s.request,slice(views[i],0,(size_t)nt*widths[i]),argv[2],prefix,stages[i],2);
        index_words+=compare(g,s.request,slice(s.indices,0,(size_t)nt*2052),argv[2],prefix,"indices",4);
        bf16_words+=compare(g,s.request,slice(s.kc,8192,(size_t)s.position*512),argv[2],prefix,"key_cache",2);
        bf16_words+=compare(g,s.request,slice(s.vc,8192,(size_t)s.position*512),argv[2],prefix,"value_cache",2);
        printf("phase%d %s tokens%d position%d native chain exact\n",phases,prefix,nt,pos);fflush(stdout);phases++;
    }
    assert(feof(f)&&phases>0&&(profile<0||profiled==1));fclose(f);bn_backend_session_free(s.session);
    if(model_mode) {assert(model_norm_hash(&model)==cpu_norm_hash);bn_model_free(&model);bn_gguf_free(file);}
    else {
        void *weights[]={w.main,w.index,w.out,w.qn,w.kn,w.iqn,w.ikn,w.cs};
        for(int i=0;i<8;i++)g->buffer_destroy(g->ctx,weights[i]);
    }
    bn_gpu_cuda_destroy(g);printf("QSA_NATIVE_CHAIN_PASSED cases%d phases%d bf16_words%zu index_words%zu\n",cases,phases,bf16_words,index_words);return 0;
}
