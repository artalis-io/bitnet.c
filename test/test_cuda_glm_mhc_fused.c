/* Optional actual-checkpoint resident mHC replay. No reference intermediate is
 * fed to native execution: weights bind through the production GGUF reader. */
#include "gpu_cuda.h"
#include "model_arch.h"
#include "backend_layout.h"
#include "backend_session.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *load(const char*dir,const char*name,size_t bytes) {
    char path[1024];snprintf(path,sizeof(path),"%s/%s.bin",dir,name);FILE*f=fopen(path,"rb");assert(f);
    void*p=malloc(bytes);assert(p);assert(fread(p,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return p;
}
static BnGPUResidentView allocv(BnGPUBackend*g,void*r,int type,size_t count,const void*data) {
    BnGPUResidentView v={0};assert(!g->resident_alloc(r,type,count,&v));
    if(data)assert(!g->resident_write(r,v,data));else assert(!g->resident_zero(r,v));return v;
}
static size_t check(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*name) {
    size_t word=v.type==BN_GGUF_TENSOR_F32?4:2,bytes=v.count*word;
    void*expected=load(dir,name,bytes),*actual=malloc(bytes);assert(actual);assert(!g->resident_read(r,v,actual));
    size_t bad=0;for(size_t i=0;i<v.count;++i)if(memcmp((char*)actual+i*word,(char*)expected+i*word,word)) {
        if(bad<3)fprintf(stderr,"%s/%s mismatch at %zu\n",dir,name,i);
        bad++;
    }
    if(bad){fprintf(stderr,"%zu / %zu mismatches\n",bad,v.count);exit(3);}
    free(expected);free(actual);return v.count;
}
static void zero_check(BnGPUBackend*g,void*r,BnGPUResidentView v) {
    size_t bytes=v.count*(v.type==BN_GGUF_TENSOR_F32?4:2);unsigned char*data=malloc(bytes);assert(data);
    assert(!g->resident_read(r,v,data));for(size_t i=0;i<bytes;++i)assert(!data[i]);free(data);
}
static void reject_fused(BnGPUBackend*g,void*r,const BnGPUMHCFusedResidentPlan *p) {
    BnGPUMHCFusedResidentPlan bad=*p;bad.next.out.count--;
    assert(g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&bad)==-1);
    bad=*p;bad.previous_comb=p->next.comb;
    assert(g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&bad)==-1);
    bad=*p;bad.previous_residual=p->next.residual;
    assert(g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&bad)==-1);
    bad=*p;bad.previous_post.request=&bad;
    assert(g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&bad)==-1);
    bad=*p;bad.next.n_splits++;
    assert(g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&bad)==-1);
    BnGPUResidentView v[]={p->next.residual,p->next.dot,p->next.sqrsum,p->next.pre,p->next.post,p->next.comb,p->next.collapsed,p->next.out};
    for(int i=0;i<8;++i)zero_check(g,r,v[i]);
}
int main(int argc,char**argv) {
    if(argc!=3){fprintf(stderr,"usage: %s first-shard.gguf reference-directory\n",argv[0]);return 2;}
    BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));
    BnGPUBackend*g=bn_gpu_cuda_create();assert(g&&g->mhc_fused_post_pre_bf16_resident);
    int sizes[]={1,4,5,9,15,16,32,33,128};int layers[]={0,1,44,44},branches[]={1,0,1,0};size_t words=0;
    for(int ci=0;ci<4;++ci) {
        int layer=layers[ci],ffn=branches[ci];BnBackendBF16MHCWeights w={0};
        assert(!bn_backend_layout_bf16_mhc_init(&w,g,f,a,layer,ffn,4u*1024*1024,2u*1024*1024));
        printf("layer%d %s: %zu GPU / %zu host staging bytes\n",layer,ffn?"ffn":"attn",w.bytes,w.host_staging_bytes);
        for(int c=0;c<9;++c) {
            int nt=sizes[c],dim=w.dim,sp=g->mhc_fused_bf16_split_count(g->ctx,nt,dim);assert(sp>0);
            size_t n=(size_t)nt;char dir[1024];snprintf(dir,sizeof(dir),"%s/%d-%s-%d",argv[2],layer,ffn?"ffn":"attn",nt);
            void*res=load(dir,"old_residual",n*4*dim*2),*branch=load(dir,"branch",n*dim*2);
            BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);
            BnGPUMHCPreResidentPlan p={0};p.n_tokens=nt;p.dim=dim;p.streams=4;p.n_splits=sp;p.iterations=w.iterations;
            p.rms_eps=w.rms_eps;p.pre_eps=p.sinkhorn_eps=w.hc_eps;p.post_mult=2;p.norm_eps=w.norm_eps;
            p.fn=w.fn;p.scale=w.scale;p.base=w.base;p.norm=w.norm;
            p.residual=allocv(g,r,BN_GGUF_TENSOR_BF16,n*4*dim,NULL);
            p.dot=allocv(g,r,BN_GGUF_TENSOR_F32,n*sp*24,NULL);p.sqrsum=allocv(g,r,BN_GGUF_TENSOR_F32,n*sp,NULL);
            p.pre=allocv(g,r,BN_GGUF_TENSOR_F32,n*4,NULL);p.post=allocv(g,r,BN_GGUF_TENSOR_F32,n*4,NULL);p.comb=allocv(g,r,BN_GGUF_TENSOR_F32,n*16,NULL);
            p.collapsed=allocv(g,r,BN_GGUF_TENSOR_BF16,n*dim,NULL);p.out=allocv(g,r,BN_GGUF_TENSOR_BF16,n*dim,NULL);
            void *old_post=load(dir,"old_post",n*4*4),*old_comb=load(dir,"old_comb",n*16*4);
            BnGPUMHCFusedResidentPlan fused={.next=p,
                .previous_residual=allocv(g,r,BN_GGUF_TENSOR_BF16,n*4*dim,res),
                .previous_branch=allocv(g,r,BN_GGUF_TENSOR_BF16,n*dim,branch),
                .previous_post=allocv(g,r,BN_GGUF_TENSOR_F32,n*4,old_post),
                .previous_comb=allocv(g,r,BN_GGUF_TENSOR_F32,n*16,old_comb)};
            reject_fused(g,r,&fused);
            assert(!g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&fused));assert(!g->resident_synchronize(r));
            BnGPUResidentView v[]={p.residual,p.dot,p.sqrsum,p.pre,p.post,p.comb,p.collapsed,p.out};
            const char*names[]={"residual","dot","sq","pre","post","comb","collapsed","normalized"};
            for(int i=0;i<8;++i)words+=check(g,r,v[i],dir,names[i]);
            words+=check(g,r,fused.previous_residual,dir,"old_residual");words+=check(g,r,fused.previous_branch,dir,"branch");
            words+=check(g,r,fused.previous_post,dir,"old_post");words+=check(g,r,fused.previous_comb,dir,"old_comb");
            fused.next.norm=NULL;assert(!g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&fused));assert(!g->resident_synchronize(r));words+=check(g,r,p.out,dir,"collapsed");
            printf("%d-%s-%d PASSED every intermediate; invalid plans preserved outputs\n",layer,ffn?"ffn":"attn",nt);fflush(stdout);
            bn_backend_session_free(session);free(res);free(branch);free(old_post);free(old_comb);
        }
        bn_backend_layout_bf16_mhc_free(&w);
    }
    printf("PASSED %zu exact default fused mHC words\n",words);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;
}
