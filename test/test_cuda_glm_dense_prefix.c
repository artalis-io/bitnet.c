/* Actual tokens -> three dense decoder blocks, cold and native warmup. */
#include "gpu_cuda.h"
#include "model_arch.h"
#include "backend_layout.h"
#include "backend_session.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *load(const char*dir,const char*name,size_t bytes) {
    char path[1024];snprintf(path,sizeof(path),"%s/%s.bin",dir,name);FILE*f=fopen(path,"rb");assert(f);void*p=malloc(bytes);assert(p);
    assert(fread(p,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return p;
}
static BnGPUResidentView allocv(BnGPUBackend*g,void*r,int type,size_t count,const void*data) {
    BnGPUResidentView v={0};assert(!g->resident_alloc(r,type,count,&v));if(data)assert(!g->resident_write(r,v,data));return v;
}
static BnGPUResidentView slice(BnGPUResidentView v,size_t off,size_t n){return bn_gpu_resident_view_slice(v,off,n);}
static size_t check(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,int c,int phase,const char*name) {
    char file[256];(void)c;snprintf(file,sizeof(file),"%d-%s",phase,name);size_t word=v.type==BN_GGUF_TENSOR_F32?4:2;
    void*expected=load(dir,file,v.count*word),*actual=malloc(v.count*word);assert(actual);assert(!g->resident_read(r,v,actual));
    size_t bad=0;for(size_t i=0;i<v.count;++i)if(memcmp((char*)actual+i*word,(char*)expected+i*word,word)) {
        if(bad<3)fprintf(stderr,"case%d phase%d %s word%zu mismatch\n",c,phase,name,i);
        bad++;
    }
    if(bad){fprintf(stderr,"%zu / %zu mismatches\n",bad,v.count);exit(3);}free(actual);free(expected);return v.count;
}
static BnGPUMHCPreResidentPlan mhc_workspace(BnGPUBackend*g,void*r,const BnBackendBF16MHCWeights*w) {
    BnGPUMHCPreResidentPlan p={.dim=w->dim,.streams=4,.iterations=w->iterations,
        .rms_eps=w->rms_eps,.pre_eps=w->hc_eps,.sinkhorn_eps=w->hc_eps,.post_mult=2,.norm_eps=w->norm_eps,
        .fn=w->fn,.scale=w->scale,.base=w->base,.norm=w->norm};
    p.residual=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*4*w->dim,NULL);p.dot=allocv(g,r,BN_GGUF_TENSOR_F32,9*64*24,NULL);p.sqrsum=allocv(g,r,BN_GGUF_TENSOR_F32,9*64,NULL);
    p.pre=allocv(g,r,BN_GGUF_TENSOR_F32,9*4,NULL);p.post=allocv(g,r,BN_GGUF_TENSOR_F32,9*4,NULL);p.comb=allocv(g,r,BN_GGUF_TENSOR_F32,9*64,NULL);
    p.collapsed=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*w->dim,NULL);p.out=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*w->dim,NULL);return p;
}
static BnGPUMHCPreResidentPlan mhc_take(BnGPUMHCPreResidentPlan p,int t,int sp) {
    p.n_tokens=t;p.n_splits=sp;size_t n=t;
    p.residual=slice(p.residual,0,n*4*p.dim);p.dot=slice(p.dot,0,n*sp*24);p.sqrsum=slice(p.sqrsum,0,n*sp);
    p.pre=slice(p.pre,0,n*4);p.post=slice(p.post,0,n*4);p.comb=slice(p.comb,0,n*16);p.collapsed=slice(p.collapsed,0,n*p.dim);p.out=slice(p.out,0,n*p.dim);return p;
}
typedef struct {
    BnBackendBF16KDAWeights kda;
    BnBackendBF16MHCWeights attn,ffn;
    BnBackendBF16DenseFFNWeights dense;
    BnTransformerBF16DenseKDABlockPlan plan;
    BnGPUResidentView work[12];
    size_t sizes[12];
} Block;
int main(int argc,char**argv) {
    if(argc!=3){fprintf(stderr,"usage: %s first-shard.gguf prefix-reference-directory\n",argv[0]);return 2;}
    BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));BnGPUBackend*g=bn_gpu_cuda_create();assert(g);
    int ei=bn_gguf_find_tensor(f,"token_embd.weight");assert(ei>=0);BnGGUFTensorInfo*et=&f->tensors[ei];
    assert(et->type==BN_GGUF_TENSOR_Q8_0&&et->n_dims==2&&et->dims[0]==4096&&et->dims[1]==154880);
    size_t eb;assert(bn_gguf_tensor_size(et->type,et->dims[0]*et->dims[1],&eb)&&eb==674037760);
    void*eraw=bn_gguf_tensor_data(f,ei);assert(eraw);void*embedding=bn_gpu_backend_create_quant_only_buffer(g,eraw,eb,et->type,154880,4096);assert(embedding);
    Block b[3]={0};size_t weights=eb,words=0;
    for(int l=0;l<3;++l){Block*w=&b[l];assert(!bn_backend_layout_bf16_kda_init(&w->kda,g,f,a,l,146653952,108347392));
        assert(!bn_backend_layout_bf16_mhc_init(&w->attn,g,f,a,l,0,1581164,1572864));
        assert(!bn_backend_layout_bf16_mhc_init(&w->ffn,g,f,a,l,1,1581164,1572864));
        assert(!bn_backend_layout_bf16_dense_ffn_init(&w->dense,g,f,a,l,160432128,106954752));
        weights+=w->kda.bytes+w->attn.bytes+w->ffn.bytes+w->dense.bytes;}
    assert(weights==1604782984);
    int schedules[6][9]={{9},{2,7},{3,3,3},{7,1,1},{6,3},{1,1,1,1,1,1,1,1,1}},lengths[]={1,2,3,3,2,9};
    int32_t*ids=load(argv[2],"ids",12*4);
    for(int resumed=0;resumed<2;++resumed)for(int c=0;c<6;++c){
        char dir[1024];snprintf(dir,sizeof(dir),"%s/%d-%d",argv[2],resumed,c);
        BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);
        assert(!g->linear_a8_bf16_reserve(g->ctx,r,9u*12288));assert(!g->embedding_bf16_reserve(g->ctx,r));
        BnGPUResidentView iv=allocv(g,r,BN_GGUF_TENSOR_I32,12,ids),ev=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*4096,NULL),twice=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*8192,NULL);
        for(int l=0;l<3;++l){Block*w=&b[l];BnTransformerBF16DenseKDABlockPlan*p=&w->plan;*p=(BnTransformerBF16DenseKDABlockPlan){0};p->weights=&w->dense;p->attention.weights=&w->kda;p->attention.deferred=1;p->attention.has_previous=l>0;
            p->attention.mhc=mhc_workspace(g,r,&w->attn);p->ffn=mhc_workspace(g,r,&w->ffn);
            p->attention.initial_history=allocv(g,r,BN_GGUF_TENSOR_BF16,73728,NULL);p->attention.next_history=allocv(g,r,BN_GGUF_TENSOR_BF16,73728,NULL);
            p->attention.initial_state=allocv(g,r,BN_GGUF_TENSOR_F32,1048576,NULL);p->attention.next_state=allocv(g,r,BN_GGUF_TENSOR_F32,1048576,NULL);
            assert(!g->resident_zero(r,p->attention.initial_history));assert(!g->resident_zero(r,p->attention.initial_state));
            p->attention.branch_out=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*4096,NULL);
            p->gate_up=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*24576,NULL);p->activated=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*12288,NULL);p->out=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*4096,NULL);
            size_t sizes[]={9u*24896,9u*24576,9u*320,9u*256,9u*64,9u*128,9u*128,9u*8192,9u*8192,9u*24576,9u*8192,9u*8192};
            for(int i=0;i<12;++i){w->sizes[i]=sizes[i];w->work[i]=allocv(g,r,BN_GGUF_TENSOR_BF16,sizes[i],NULL);}
        }
        int off=0;for(int phase=0;phase<lengths[c]+resumed;++phase){
            int t=resumed&&phase==0?3:schedules[c][phase-resumed];
            BnGPUResidentView e=slice(ev,0,(size_t)t*4096),two=slice(twice,0,(size_t)t*8192),expanded=slice(b[0].plan.attention.mhc.residual,0,(size_t)t*16384);
            BnGPUEmbeddingResidentPlan ep={.n_tokens=t,.dim=4096,.vocab_size=154880,.weight=embedding,.indices=slice(iv,off,t),.out=e};assert(!g->embedding_bf16_resident(g->ctx,r,&ep));
            BnGPUJoinColumnsResidentPlan j1={.rows=t,.cols=8192,.left_cols=4096,.left=e,.right=e,.x=two},j2={.rows=t,.cols=16384,.left_cols=8192,.left=two,.right=two,.x=expanded};
            assert(!g->join_columns_resident(g->ctx,r,&j1));assert(!g->join_columns_resident(g->ctx,r,&j2));
            BnTransformerBF16DenseKDABlockPlan plans[3];
            for(int l=0;l<3;++l){Block*w=&b[l];BnTransformerBF16DenseKDABlockPlan*p=&plans[l];*p=w->plan;
                int asp=l?g->mhc_fused_bf16_split_count(g->ctx,t,4096):g->mhc_pre_bf16_split_count(g->ctx,t,4096),fsp=g->mhc_fused_bf16_split_count(g->ctx,t,4096);
                p->attention.mhc=mhc_take(p->attention.mhc,t,asp);p->ffn=mhc_take(p->ffn,t,fsp);
                BnGPUResidentView v[12];for(int i=0;i<12;++i)v[i]=slice(w->work[i],0,w->sizes[i]/9*t);
                p->attention.kda=(BnTransformerBF16KDAWorkspace){v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11]};
                p->attention.branch_out=slice(p->attention.branch_out,0,(size_t)t*4096);p->gate_up=slice(p->gate_up,0,(size_t)t*24576);p->activated=slice(p->activated,0,(size_t)t*12288);p->out=slice(p->out,0,(size_t)t*4096);
                if(l){p->attention.previous_residual=plans[l-1].ffn.residual;p->attention.previous_branch=plans[l-1].out;p->attention.previous_post=plans[l-1].ffn.post;p->attention.previous_comb=plans[l-1].ffn.comb;}
                int status=bn_transformer_bf16_dense_kda_block(r,p);if(status)fprintf(stderr,"block failure l=%d t=%d asp=%d fsp=%d\n",l,t,asp,fsp);assert(!status);
            }
            assert(!g->resident_synchronize(r));words+=check(g,r,e,dir,c,phase,"embedding");words+=check(g,r,expanded,dir,c,phase,"expanded");
            for(int l=0;l<3;++l){BnTransformerBF16DenseKDABlockPlan*p=&plans[l];BnTransformerBF16MHCKDAPlan*q=&p->attention;BnTransformerBF16KDAWorkspace*s=&q->kda;
                BnGPUResidentView vs[]={q->mhc.out,q->mhc.residual,q->mhc.post,q->mhc.comb,s->projected,s->g1,s->g2,s->conv,s->core,s->gated,q->branch_out,q->next_state,q->next_history,p->ffn.residual,p->ffn.post,p->ffn.comb,p->ffn.out,p->gate_up,p->activated,p->out};
                const char*names[]={"input","entry_residual","entry_post","entry_comb","projected","g1","g2","conv","core","gated","attn_out","state","history","ffn_residual","ffn_post","ffn_comb","ffn_input","gate_up","activated","out"};
                for(int i=0;i<20;++i){char name[128];snprintf(name,sizeof(name),"layer%d-%s",l,names[i]);words+=check(g,r,vs[i],dir,c,phase,name);}
                BnTransformerBF16MHCKDAPlan*base=&b[l].plan.attention;BnGPUResidentView tmp=base->initial_history;base->initial_history=base->next_history;base->next_history=tmp;tmp=base->initial_state;base->initial_state=base->next_state;base->next_state=tmp;
            }
            off+=t;
        }
        printf("resumed%d case%d first three decoder blocks PASSED\n",resumed,c);fflush(stdout);bn_backend_session_free(session);
    }
    printf("PASSED %zu exact three-block words, %zu GPU weight payload bytes\n",words,weights);
    for(int l=0;l<3;++l){bn_backend_layout_bf16_dense_ffn_free(&b[l].dense);bn_backend_layout_bf16_mhc_free(&b[l].attn);bn_backend_layout_bf16_mhc_free(&b[l].ffn);bn_backend_layout_bf16_kda_free(&b[l].kda);}
    free(ids);bn_gpu_backend_destroy_buffer(g,embedding);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;
}
