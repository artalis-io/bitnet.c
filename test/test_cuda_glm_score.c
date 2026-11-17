#define main glm_indexer_parent_main
#include "test_cuda_glm_indexer.c"
#undef main
#include "cuda_glm_score_reference.h"
static unsigned char score_byte(size_t i,int seed) {
    return (unsigned char)((i*37+i/13+(size_t)seed*11)%127+((i*17+(size_t)seed)%3==0?128:0));
}
static size_t score_cases(BnGPUBackend *g,const char *directory) {
    size_t total=0;
    for(size_t ci=0;ci<sizeof(glm_score_reference)/sizeof(glm_score_reference[0]);++ci) {
        BnGLMScoreReference c=glm_score_reference[ci];size_t n=(size_t)c.tokens,k=(size_t)(c.position+c.tokens)/4;
        unsigned char *q=malloc(n*4096),*key=malloc(k*128+1);float *scale=malloc(k*4+1),*weight=malloc(n*32*4);assert(q&&key&&scale&&weight);
        for(size_t i=0;i<n*4096;++i)q[i]=score_byte(i,c.seed);
        for(size_t i=0;i<k*128;++i)key[i]=score_byte(i,c.seed+17);
        for(size_t i=0;i<k;++i)scale[i]=ldexpf(1.f,(int)(i%13)-6);
        for(size_t i=0;i<n*32;++i)weight[i]=(float)((int)((i*17+(size_t)c.seed*13)%251)-125)/128+0x1p-17f;
        BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);
        BnGPUIndexerFP8ScoreResidentPlan p={.n_tokens=c.tokens,.position=c.position,.n_keys=(int)k,
            .q=alloc(g,r,BN_GGUF_TENSOR_I8,n*4096,q),.keys=alloc(g,r,BN_GGUF_TENSOR_I8,k*128,key),
            .scales=alloc(g,r,BN_GGUF_TENSOR_F32,k,scale),.weights=alloc(g,r,BN_GGUF_TENSOR_F32,n*32,weight),
            .scores=alloc(g,r,BN_GGUF_TENSOR_F32,n*k,NULL)};
        assert(!g->indexer_fp8_score_resident(g->ctx,r,&p));total+=check(g,r,p.scores,4,c.hash);
        char prefix[40];snprintf(prefix,sizeof(prefix),"%zu",ci);compare_file(g,r,p.scores,4,directory,prefix,"");
        BnBackendSession *other=bn_backend_session_create();assert(other);void *peer=bn_gpu_backend_resident_request(g,other);assert(peer);
        BnGPUIndexerFP8ScoreResidentPlan bad;
#define BAD(change) do{bad=p;change;assert(g->indexer_fp8_score_resident(g->ctx,r,&bad)==-1);}while(0)
        BAD(bad.n_tokens=0);BAD(bad.position=-1);BAD(bad.position=INT_MAX);BAD(bad.n_keys++);
        BAD(bad.n_tokens=INT_MAX);BAD(bad.q.request=peer);BAD(bad.q.offset=SIZE_MAX);
        BAD(bad.q.count--);BAD(bad.weights.type=BN_GGUF_TENSOR_I32);BAD(bad.weights=p.q);
        if(k){BAD(bad.scales.count--);BAD(bad.scores=p.scales);}
        BnGPUResidentView unaligned=alloc(g,r,BN_GGUF_TENSOR_I8,n*4096+1,NULL);
        BAD(bad.q=unaligned;bad.q.offset++;bad.q.count--);
        assert(check(g,r,p.scores,4,c.hash)==n*k); /* rejection leaves outputs intact */
#undef BAD
        bn_backend_session_free(other);bn_backend_session_free(s);free(q);free(key);free(scale);free(weight);
    }
    return total;
}
int main(int argc,char **argv) {
    assert(argc==1||argc==2);BnGPUBackend *g=bn_gpu_cuda_create();assert(g&&g->indexer_fp8_score_resident);
    size_t n=score_cases(g,argc==2?argv[1]:NULL);bn_gpu_cuda_destroy(g);
    printf("GLM FP8 signed causal scores PASSED: %zu exact F32 elements\n",n);return 0;
}
