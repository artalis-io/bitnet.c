#include "gpu_backend.h"
#include "gpu_cuda.h"
#include "backend_session.h"
#include "quant.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_glm_indexer_reference.h"
#include "cuda_glm_layernorm_reference.h"

static uint16_t bf16(float x) {
    uint32_t bits;memcpy(&bits,&x,4);bits+=0x7fff+((bits>>16)&1);return (uint16_t)(bits>>16);
}
static uint64_t hash(const void *data,size_t n) {
    const unsigned char *p=data;uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;++i)h=(h^p[i])*UINT64_C(1099511628211);
    return h;
}
static BnGPUResidentView alloc(BnGPUBackend *g,void *r,int type,size_t n,const void *data) {
    BnGPUResidentView v={0};if(!n)return v;
    assert(!g->resident_alloc(r,type,n,&v));if(data)assert(!g->resident_write(r,v,data));return v;
}
static size_t check(BnGPUBackend *g,void *r,BnGPUResidentView v,size_t width,uint64_t expected) {
    size_t bytes=v.count*width;void *data=malloc(bytes?bytes:1);assert(data);
    if(bytes)assert(!g->resident_read(r,v,data));
    uint64_t actual=hash(data,bytes);
    if(actual!=expected){fprintf(stderr,"indexer mismatch count%zu got%016llx expected%016llx\n",v.count,(unsigned long long)actual,(unsigned long long)expected);abort();}
    free(data);return v.count;
}
static void compare_file(BnGPUBackend *g,void *r,BnGPUResidentView v,size_t width,
    const char *directory,const char *prefix,const char *suffix) {
    if(!directory)return;
    char path[1024];int len=snprintf(path,sizeof(path),"%s/%s%s.bin",directory,prefix,suffix);
    assert(len>0&&(size_t)len<sizeof(path));FILE *f=fopen(path,"rb");assert(f);
    size_t bytes=v.count*width;void *expected=malloc(bytes?bytes:1),*actual=malloc(bytes?bytes:1);assert(expected&&actual);
    assert(fread(expected,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);
    if(bytes)assert(!g->resident_read(r,v,actual));
    if(memcmp(expected,actual,bytes)){fprintf(stderr,"direct reference mismatch: %s\n",path);abort();}
    free(expected);free(actual);
}
static void query_input(uint16_t *x,int rows,int seed) {
    int exponents[]={-30,-16,-4,0,4,12,24,0};
    for(int row=0;row<rows;++row)for(int d=0;d<128;++d) {
        size_t i=(size_t)row*128+d;
        float v=(float)((int)((i*37+i/13+(size_t)seed*11)%509)-254)/128;
        v=ldexpf(v,exponents[(row+seed)%8]);
        if(row==0)v=0;
        if(row==1)v=1;
        if(row==2)v=d==1?1.f:0.f;
        if(row==3)v=-0.f;
        x[i]=bf16(v);
    }
}
static uint16_t pool_input(int logical,int d,int seed,int gate) {
    size_t i=(size_t)logical*128+d;
    float v=gate?(float)((int)((i*13+i/17+(size_t)seed*3)%251)-125)/(seed?2:16):
        (float)((int)((i*37+i/13+(size_t)seed*11)%509)-254)/128;
    return bf16(v);
}
static size_t queries(BnGPUBackend *g,const char *directory) {
    size_t words=0;
    for(size_t ci=0;ci<sizeof(glm_query_reference)/sizeof(glm_query_reference[0]);++ci) {
        BnGLMQueryReference c=glm_query_reference[ci];size_t n=(size_t)c.rows;
        uint16_t *x=malloc(n*128*2);float *weights=malloc(n*4);assert(x&&weights);query_input(x,c.rows,c.seed);
        for(size_t i=0;i<n;++i)weights[i]=(float)((int)((i*17+(size_t)c.seed*13)%251)-125)/128+0x1p-17f;
        BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);
        BnGPUFWHT128FP8ResidentPlan p={.rows=c.rows,.x=alloc(g,r,BN_GGUF_TENSOR_BF16,n*128,x),
            .out=alloc(g,r,BN_GGUF_TENSOR_I8,n*128,NULL),.scales=alloc(g,r,BN_GGUF_TENSOR_F32,n,NULL)};
        assert(!g->fwht128_fp8_resident(g->ctx,r,&p));words+=check(g,r,p.out,1,c.out);words+=check(g,r,p.scales,4,c.scale);
        BnGPUIndexerWeightScaleResidentPlan w={.rows=c.rows,.factor=c.factor,
            .weights=alloc(g,r,BN_GGUF_TENSOR_F32,n,weights),.query_scales=p.scales,.out=alloc(g,r,BN_GGUF_TENSOR_F32,n,NULL)};
        assert(!g->indexer_weight_scale_resident(g->ctx,r,&w));words+=check(g,r,w.out,4,c.weighted);
        char prefix[80];snprintf(prefix,sizeof(prefix),"q-%d-%d",c.rows,c.seed);
        compare_file(g,r,p.out,1,directory,prefix,"");compare_file(g,r,p.scales,4,directory,prefix,"-scale");compare_file(g,r,w.out,4,directory,prefix,"-weighted");
        free(x);free(weights);bn_backend_session_free(s);
    }
    return words;
}
static size_t pools(BnGPUBackend *g,const char *directory) {
    size_t words=0;int chunks[]={INT_MAX,1,3,7,64};
    for(size_t ci=0;ci<sizeof(glm_pool_reference)/sizeof(glm_pool_reference[0]);++ci) {
        BnGLMPoolReference c=glm_pool_reference[ci];size_t n=(size_t)c.tokens;
        uint16_t *key=malloc(n*128*2),*gate=malloc(n*128*2),tail[1024]={0};float ape[512];assert(key&&gate);
        for(int t=0;t<c.tokens;++t)for(int d=0;d<128;++d){key[(size_t)t*128+d]=pool_input(c.position+t,d,c.seed,0);gate[(size_t)t*128+d]=pool_input(c.position+t,d,c.seed,1);}
        for(int t=c.position>4?c.position-4:0;t<c.position;++t)for(int d=0;d<128;++d){tail[(t%4)*128+d]=pool_input(t,d,c.seed,0);tail[512+(t%4)*128+d]=pool_input(t,d,c.seed,1);}
        for(int i=0;i<512;++i)ape[i]=(float)((i*19+c.seed*13)%127-63)/32+0x1p-17f;
        void *aw=bn_gpu_backend_create_quant_only_buffer(g,ape,sizeof(ape),BN_GGUF_TENSOR_F32,4,128);assert(aw);
        for(int mode=0;mode<5;++mode) {
            BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);
            BnGPUResidentView kv=alloc(g,r,BN_GGUF_TENSOR_BF16,n*128,key),gv=alloc(g,r,BN_GGUF_TENSOR_BF16,n*128,gate);
            BnGPUResidentView current=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,tail),next=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,NULL);
            BnGPUResidentView starting=current;
            BnGPUResidentView out=alloc(g,r,BN_GGUF_TENSOR_I8,(size_t)c.groups*128,NULL),scale=alloc(g,r,BN_GGUF_TENSOR_F32,c.groups,NULL);
            for(int done=0;done<c.tokens;) {
                int take=c.tokens-done;if(take>chunks[mode])take=chunks[mode];int pos=c.position+done;
                size_t before=(size_t)(pos/4-c.position/4),count=(pos+take)/4-pos/4;
                BnGPUKPoolFP8ResidentPlan p={.n_tokens=take,.position=pos,.ape=aw,.key=bn_gpu_resident_view_slice(kv,(size_t)done*128,(size_t)take*128),
                    .gate=bn_gpu_resident_view_slice(gv,(size_t)done*128,(size_t)take*128),.initial_tail=current,.next_tail=next};
                if(count){p.out=bn_gpu_resident_view_slice(out,before*128,count*128);p.scales=bn_gpu_resident_view_slice(scale,before,count);}
                assert(!g->kpool_fp8_resident(g->ctx,r,&p));BnGPUResidentView swap=current;current=next;next=swap;done+=take;
            }
            words+=check(g,r,out,1,c.out);words+=check(g,r,scale,4,c.scale);words+=check(g,r,current,2,c.tail);
            char prefix[80];snprintf(prefix,sizeof(prefix),"pool-%d-%d-%d",c.position,c.tokens,c.seed);
            compare_file(g,r,out,1,directory,prefix,"");compare_file(g,r,scale,4,directory,prefix,"-scale");compare_file(g,r,current,2,directory,prefix,"-tail");
            if(mode==0){check(g,r,starting,2,hash(tail,sizeof(tail)));check(g,r,kv,2,hash(key,n*128*2));check(g,r,gv,2,hash(gate,n*128*2));}
            bn_backend_session_free(s);
        }
        bn_gpu_backend_destroy_buffer(g,aw);free(key);free(gate);
    }
    return words;
}
static size_t layernorms(BnGPUBackend *g,const char *directory) {
    size_t words=0;
    for(size_t ci=0;ci<sizeof(glm_layernorm_reference)/sizeof(glm_layernorm_reference[0]);++ci) {
        BnGLMLayerNormReference c=glm_layernorm_reference[ci];size_t n=(size_t)c.rows*128;
        uint16_t *x=malloc(n*2);float w[128],b[128];assert(x);
        for(size_t i=0;i<n;++i){size_t row=i/128;float value=(float)((int)((i*37+i/13+(size_t)c.seed*11)%509)-254)/128;
            int mode=(int)((row+c.seed)%8);if(mode==0)value=0;if(mode==1)value=1;if(mode==2)value=4096.f+value*32;
            if(mode==3)value*=0x1p-20f;
            if(mode==4)value*=0x1p16f;
            if(mode==5)value=-0.f;
            x[i]=bf16(value);}
        for(int j=0;j<128;++j){w[j]=(float)((j*7+c.seed*3)%31-15)/16+1+0x1p-17f;b[j]=(float)((j*11+c.seed*5)%61-30)/128+0x1p-19f;}
        void *ww=bn_gpu_backend_create_quant_only_buffer(g,w,sizeof(w),BN_GGUF_TENSOR_F32,1,128);
        void *bb=bn_gpu_backend_create_quant_only_buffer(g,b,sizeof(b),BN_GGUF_TENSOR_F32,1,128);assert(ww&&bb);
        BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);
        BnGPULayerNorm128F32BF16ResidentPlan p={.n_tokens=c.rows,.eps=c.eps,.weight=ww,.bias=bb,.x=alloc(g,r,BN_GGUF_TENSOR_BF16,n,x),.out=alloc(g,r,BN_GGUF_TENSOR_BF16,n,NULL)};
        assert(!g->layernorm128_f32_bf16_resident(g->ctx,r,&p));words+=check(g,r,p.out,2,c.hash);
        char prefix[80];snprintf(prefix,sizeof(prefix),"norm-%d-%d-%d",c.rows,c.seed,c.eps_index);compare_file(g,r,p.out,2,directory,prefix,"");
        check(g,r,p.x,2,hash(x,n*2));bn_backend_session_free(s);bn_gpu_backend_destroy_buffer(g,ww);bn_gpu_backend_destroy_buffer(g,bb);free(x);
    }
    return words;
}
static void guards(BnGPUBackend *g) {
    BnBackendSession *s=bn_backend_session_create(),*other=bn_backend_session_create();assert(s&&other);
    void *r=bn_gpu_backend_resident_request(g,s),*peer=bn_gpu_backend_resident_request(g,other);assert(r&&peer);
    unsigned char canary[128];memset(canary,0x5a,sizeof(canary));float scale_canary=17.f;
    uint16_t zero[1024]={0},next_canary[1024],next_actual[1024];float ape[512]={0};
    for(int i=0;i<1024;++i)next_canary[i]=0x3e80;
    BnGPUFWHT128FP8ResidentPlan p={.rows=1,.x=alloc(g,r,BN_GGUF_TENSOR_BF16,128,zero),.out=alloc(g,r,BN_GGUF_TENSOR_I8,128,canary),.scales=alloc(g,r,BN_GGUF_TENSOR_F32,1,&scale_canary)};
    BnGPUFWHT128FP8ResidentPlan bad=p;
#define REJECT_Q(change) do{bad=p;change;assert(g->fwht128_fp8_resident(g->ctx,r,&bad)==-1);}while(0)
    REJECT_Q(bad.rows=0);REJECT_Q(bad.rows=INT_MAX);REJECT_Q(bad.out.count--);REJECT_Q(bad.x.count--);REJECT_Q(bad.scales.count++);
    REJECT_Q(bad.out=p.x);REJECT_Q(bad.x.request=peer);REJECT_Q(bad.x.offset=SIZE_MAX);
#undef REJECT_Q
    assert(!g->resident_read(r,p.out,canary));for(int i=0;i<128;++i)assert(canary[i]==0x5a);
    assert(!g->resident_read(r,p.scales,&scale_canary)&&scale_canary==17.f);
    void *aw=bn_gpu_backend_create_quant_only_buffer(g,ape,sizeof(ape),BN_GGUF_TENSOR_F32,4,128);assert(aw);
    BnGPUKPoolFP8ResidentPlan k={.n_tokens=1,.position=3,.ape=aw,.key=p.x,.gate=alloc(g,r,BN_GGUF_TENSOR_BF16,128,zero),
        .initial_tail=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,zero),.next_tail=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,next_canary),.out=p.out,.scales=p.scales};
    BnGPUKPoolFP8ResidentPlan kb;
#define REJECT_K(change) do{kb=k;change;assert(g->kpool_fp8_resident(g->ctx,r,&kb)==-1);}while(0)
    REJECT_K(kb.position=-1);REJECT_K(kb.position=INT_MAX);REJECT_K(kb.n_tokens=0);REJECT_K(kb.ape=NULL);
    REJECT_K(kb.next_tail=kb.initial_tail);REJECT_K(kb.gate=kb.key);REJECT_K(kb.out.count--);REJECT_K(kb.scales.count++);REJECT_K(kb.next_tail.request=peer);
    REJECT_K(kb.position=0); /* no completed pool requires empty outputs */
#undef REJECT_K
    assert(!g->resident_read(r,k.next_tail,next_actual)&&!memcmp(next_canary,next_actual,sizeof(next_actual)));
    assert(!g->resident_read(r,k.scales,&scale_canary)&&scale_canary==17.f);
    assert(!g->resident_read(r,p.out,canary));for(int i=0;i<128;++i)assert(canary[i]==0x5a);
    BnGPUIndexerWeightScaleResidentPlan w={.rows=1,.factor=.015625f,.weights=p.scales,.query_scales=alloc(g,r,BN_GGUF_TENSOR_F32,1,&scale_canary),.out=alloc(g,r,BN_GGUF_TENSOR_F32,1,&scale_canary)},wb;
#define REJECT_W(change) do{wb=w;change;assert(g->indexer_weight_scale_resident(g->ctx,r,&wb)==-1);}while(0)
    REJECT_W(wb.factor=NAN);REJECT_W(wb.factor=0);REJECT_W(wb.rows=0);REJECT_W(wb.out=wb.weights);REJECT_W(wb.query_scales=wb.weights);REJECT_W(wb.out.request=peer);
#undef REJECT_W
    assert(!g->resident_read(r,w.out,&scale_canary)&&scale_canary==17.f);
    void *nw=bn_gpu_backend_create_quant_only_buffer(g,ape,512,BN_GGUF_TENSOR_F32,1,128);
    void *nb=bn_gpu_backend_create_quant_only_buffer(g,ape,512,BN_GGUF_TENSOR_F32,1,128);assert(nw&&nb);
    BnGPULayerNorm128F32BF16ResidentPlan ln={.n_tokens=1,.eps=1e-6f,.weight=nw,.bias=nb,.x=p.x,.out=alloc(g,r,BN_GGUF_TENSOR_BF16,128,next_canary)},lb;
#define REJECT_L(change) do{lb=ln;change;assert(g->layernorm128_f32_bf16_resident(g->ctx,r,&lb)==-1);}while(0)
    REJECT_L(lb.n_tokens=0);REJECT_L(lb.n_tokens=INT_MAX);REJECT_L(lb.eps=NAN);REJECT_L(lb.eps=0);REJECT_L(lb.weight=NULL);REJECT_L(lb.bias=NULL);
    REJECT_L(lb.weight=aw);REJECT_L(lb.bias=aw);REJECT_L(lb.out=lb.x);REJECT_L(lb.x.count--);REJECT_L(lb.out.request=peer);
#undef REJECT_L
    check(g,r,ln.out,2,hash(next_canary,128*2));bn_gpu_backend_destroy_buffer(g,nw);bn_gpu_backend_destroy_buffer(g,nb);
    bn_gpu_backend_destroy_buffer(g,aw);bn_backend_session_free(other);bn_backend_session_free(s);
}
int main(int argc,char **argv) {
    assert(argc>=1&&argc<=3);const char *directory=argc>=2?argv[1]:NULL;const char *norm_directory=argc==3?argv[2]:NULL;
    BnGPUBackend *g=bn_gpu_cuda_create();assert(g&&g->fwht128_fp8_resident&&g->kpool_fp8_resident&&g->indexer_weight_scale_resident);
    guards(g);assert(g->layernorm128_f32_bf16_resident);size_t norm_words=layernorms(g,norm_directory);
    printf("CUDA GLM F32 LayerNorm PASSED: %zu reference elements (%s)\n",norm_words,norm_directory?"direct bytes and fixture digests":"fixture digests");
    size_t words=queries(g,directory)+pools(g,directory);bn_gpu_cuda_destroy(g);
    printf("CUDA GLM FP8 query, gated pool/tail and F32 head weights PASSED: %zu reference elements (%s)\n",words,directory?"direct bytes and fixture digests":"fixture digests");return 0;
}
