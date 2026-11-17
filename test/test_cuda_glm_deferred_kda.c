/* Default deferred FFN post -> KDA attention -> following FFN pre replay.
 * Only original incoming state enters native execution; all intermediates are
 * produced natively from actual checkpoint bindings and compared afterwards. */
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
    char file[128];snprintf(file,sizeof(file),"%d-%d-%s",c,phase,name);size_t word=v.type==BN_GGUF_TENSOR_BF16?2:4;
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
    p.residual=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*4*w->dim,NULL);p.dot=allocv(g,r,BN_GGUF_TENSOR_F32,9*16*24,NULL);p.sqrsum=allocv(g,r,BN_GGUF_TENSOR_F32,9*16,NULL);
    p.pre=allocv(g,r,BN_GGUF_TENSOR_F32,9*4,NULL);p.post=allocv(g,r,BN_GGUF_TENSOR_F32,9*4,NULL);p.comb=allocv(g,r,BN_GGUF_TENSOR_F32,9*16,NULL);
    p.collapsed=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*w->dim,NULL);p.out=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*w->dim,NULL);return p;
}
static BnGPUMHCPreResidentPlan mhc_take(BnGPUMHCPreResidentPlan p,int t,int sp) {
    p.n_tokens=t;p.n_splits=sp;size_t n=t;
    p.residual=slice(p.residual,0,n*4*p.dim);p.dot=slice(p.dot,0,n*sp*24);p.sqrsum=slice(p.sqrsum,0,n*sp);
    p.pre=slice(p.pre,0,n*4);p.post=slice(p.post,0,n*4);p.comb=slice(p.comb,0,n*16);p.collapsed=slice(p.collapsed,0,n*p.dim);p.out=slice(p.out,0,n*p.dim);return p;
}
int main(int argc,char**argv) {
    if(argc!=3){fprintf(stderr,"usage: %s first-shard.gguf deferred-reference-directory\n",argv[0]);return 2;}
    BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));BnGPUBackend*g=bn_gpu_cuda_create();assert(g);
    int schedules[6][9]={{9},{2,7},{3,3,3},{7,1,1},{6,3},{1,1,1,1,1,1,1,1,1}},lengths[]={1,2,3,3,2,9};size_t words=0;
    for(int li=0;li<2;++li) {
        int layer=li?44:1;char dir[1024];snprintf(dir,sizeof(dir),"%s/layer%d",argv[2],layer);
        BnBackendBF16KDAWeights kw={0};BnBackendBF16MHCWeights aw={0},fw={0};
        assert(!bn_backend_layout_bf16_kda_init(&kw,g,f,a,layer,512u*1024*1024,256u*1024*1024));
        assert(!bn_backend_layout_bf16_mhc_init(&aw,g,f,a,layer,0,4u*1024*1024,2u*1024*1024));assert(!bn_backend_layout_bf16_mhc_init(&fw,g,f,a,layer,1,4u*1024*1024,2u*1024*1024));
        int dim=kw.dim,p=kw.heads*128,merged=3*p+kw.heads+256;size_t history=(size_t)9*p,state=(size_t)p*128;
        void *oldres=load(dir,"old_residual",9u*4*dim*2),*oldbranch=load(dir,"old_branch",9u*dim*2),*oldpost=load(dir,"old_post",9*4*4),*oldcomb=load(dir,"old_comb",9*16*4),*h0=load(dir,"history0",history*2),*s0=load(dir,"initial",state*4);
        for(int c=0;c<6;++c) {
            BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);assert(!g->linear_a8_bf16_reserve(g->ctx,r,9u*p));
            BnGPUResidentView pr=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*4*dim,oldres),pb=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*dim,oldbranch),pp=allocv(g,r,BN_GGUF_TENSOR_F32,9*4,oldpost),pc=allocv(g,r,BN_GGUF_TENSOR_F32,9*16,oldcomb);
            BnGPUResidentView ih=allocv(g,r,BN_GGUF_TENSOR_BF16,history,h0),nh=allocv(g,r,BN_GGUF_TENSOR_BF16,history,NULL),is=allocv(g,r,BN_GGUF_TENSOR_F32,state,s0),ns=allocv(g,r,BN_GGUF_TENSOR_F32,state,NULL),branch=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*dim,NULL);
            BnGPUMHCPreResidentPlan attn=mhc_workspace(g,r,&aw),ffn=mhc_workspace(g,r,&fw);
            size_t sizes[]={9u*merged,9u*3*p,9u*(kw.heads+256),9u*256,9u*kw.heads,9u*128,9u*128,9u*p,9u*p,9u*3*p,9u*p,9u*p};BnGPUResidentView workspace[12];for(int i=0;i<12;++i)workspace[i]=allocv(g,r,BN_GGUF_TENSOR_BF16,sizes[i],NULL);
            int off=0;for(int phase=0;phase<lengths[c];++phase) {
                int t=schedules[c][phase],sp=g->mhc_fused_bf16_split_count(g->ctx,t,dim);BnGPUResidentView v[12];for(int i=0;i<12;++i)v[i]=slice(workspace[i],0,sizes[i]/9*t);
                BnTransformerBF16KDAWorkspace s={v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11]};
                BnTransformerBF16MHCKDAPlan plan={.weights=&kw,.mhc=mhc_take(attn,t,sp),.kda=s,.initial_history=ih,.next_history=nh,.initial_state=is,.next_state=ns,.branch_out=slice(branch,0,(size_t)t*dim),.deferred=1,.has_previous=1,
                    .previous_residual=slice(pr,(size_t)off*4*dim,(size_t)t*4*dim),.previous_branch=slice(pb,(size_t)off*dim,(size_t)t*dim),.previous_post=slice(pp,(size_t)off*4,(size_t)t*4),.previous_comb=slice(pc,(size_t)off*16,(size_t)t*16)};
                assert(!bn_transformer_bf16_mhc_kda_attention(r,&plan));
                BnGPUMHCFusedResidentPlan next={.next=mhc_take(ffn,t,sp),.previous_residual=plan.mhc.residual,.previous_branch=plan.branch_out,.previous_post=plan.mhc.post,.previous_comb=plan.mhc.comb};
                assert(!g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&next));assert(!g->resident_synchronize(r));
                BnGPUResidentView outputs[]={plan.mhc.out,plan.mhc.residual,plan.mhc.post,plan.mhc.comb,s.projected,s.g1,s.g2,s.conv,s.core,s.gated,plan.branch_out,ns,nh,next.next.residual,next.next.post,next.next.comb,next.next.out};
                const char*names[]={"input","entry_residual","entry_post","entry_comb","projected","g1","g2","conv","core","gated","out","state","history","ffn_residual","ffn_post","ffn_comb","ffn_input"};
                for(int i=0;i<17;++i)words+=check(g,r,outputs[i],dir,c,phase,names[i]);
                BnGPUResidentView tmp=ih;ih=nh;nh=tmp;tmp=is;is=ns;ns=tmp;off+=t;
            }
            printf("layer%d case%d default deferred attention/FFN entry PASSED\n",layer,c);fflush(stdout);bn_backend_session_free(session);
        }
        free(oldres);free(oldbranch);free(oldpost);free(oldcomb);free(h0);free(s0);bn_backend_layout_bf16_mhc_free(&aw);bn_backend_layout_bf16_mhc_free(&fw);bn_backend_layout_bf16_kda_free(&kw);
    }
    printf("PASSED %zu exact default deferred attention words\n",words);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;
}
