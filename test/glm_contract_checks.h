#include "cuda_glm_contract_reference.h"
static uint64_t glm_contract_hash(const void*data,size_t bytes){
    const unsigned char*b=data;uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;++i){h^=b[i];h*=UINT64_C(1099511628211);}return h;
}
static void run_glm_contract_reference(BnGPUBackend*g) {
    assert(g->stream_mean_bf16_resident&&g->rmsnorm_bf16_resident);
    assert(g->resident_plain_rmsnorm_reductions&(1u<<BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096));
    uint16_t gamma[4096];for(int i=0;i<4096;++i)gamma[i]=(uint16_t)((0x3f00+(i*13)%256)|(i%5==0?0x8000:0));
    void*w=bn_gpu_backend_create_quant_only_buffer(g,gamma,sizeof(gamma),BN_GGUF_TENSOR_BF16,1,4096);assert(w);size_t words=0;
    for(size_t ci=0;ci<sizeof(cuda_glm_contract_refs)/sizeof(cuda_glm_contract_refs[0]);++ci){
        const BnCudaGLMContractReference*c=&cuda_glm_contract_refs[ci];size_t n=(size_t)c->tokens*4096;
        uint16_t*x=malloc(n*4*2),*actual=malloc(n*2);assert(x&&actual);
        static const uint16_t tiny[]={0,0x8000,1,0x8001,0x7f,0x807f,0x80,0x8080};
        static const uint16_t patterns[6][4]={{0x6400,0x3f80,0xe400,0x3f80},{0x6400,0xe400,0x3f80,0x3f80},{0x3f80,0x6400,0x3f80,0xe400},{0x8000,0x8000,0x8000,0x8000},{0,0,0,0},{1,1,1,1}};
        for(size_t i=0;i<n*4;++i){uint32_t z=(uint32_t)i+7193u;z^=z<<13;z^=z>>17;z^=z<<5;
            x[i]=c->mode==0?(uint16_t)((z&0x807f)|(((z>>8)%50+100)<<7)):c->mode==2?tiny[z%8]:patterns[(i/16384+i%4096)%6][i/4096%4];}
        assert(glm_contract_hash(x,n*8)==c->input);
        BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
        BnGPUResidentView in,mean,norm;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,n*4,&in)&&!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,n,&mean)&&!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,n,&norm));
        assert(!g->resident_write(r,in,x));BnGPUStreamMeanResidentPlan p={c->tokens,4096,4,in,mean};
        assert(!g->stream_mean_bf16_resident(g->ctx,r,&p)&&!g->resident_read(r,mean,actual));
        uint64_t mh=glm_contract_hash(actual,n*2);if(mh!=c->mean){fprintf(stderr,"GLM mean case%zu T%d mode%d hash%llu expected%llu\n",ci,c->tokens,c->mode,(unsigned long long)mh,(unsigned long long)c->mean);abort();}
        BnGPURMSNormResidentPlan np={.n_tokens=c->tokens,.dim=4096,.eps=1e-5f,.reduction=BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096,.weight=w,.x=mean,.out=norm};
        assert(!g->rmsnorm_bf16_resident(g->ctx,r,&np)&&!g->resident_read(r,norm,actual));
        uint64_t nh=glm_contract_hash(actual,n*2);if(nh!=c->norm){fprintf(stderr,"GLM final norm case%zu T%d mode%d hash%llu expected%llu\n",ci,c->tokens,c->mode,(unsigned long long)nh,(unsigned long long)c->norm);abort();}
        BnGPUStreamMeanResidentPlan bad=p;bad.streams=8;assert(g->stream_mean_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.dim=4095;assert(g->stream_mean_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.x.count--;assert(g->stream_mean_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.out=in;bad.out.count=n;assert(g->stream_mean_bf16_resident(g->ctx,r,&bad)==-1);
        BnGPURMSNormResidentPlan bn=np;bn.weight_offset=1;assert(g->rmsnorm_bf16_resident(g->ctx,r,&bn)==-1);
        bn=np;bn.eps=1e-6f;assert(g->rmsnorm_bf16_resident(g->ctx,r,&bn)==-1);
        assert(!g->resident_read(r,in,x)&&glm_contract_hash(x,n*8)==c->input);
        free(x);free(actual);bn_backend_session_free(s);words+=n*2;
    }
    bn_gpu_backend_destroy_buffer(g,w);printf("CUDA GLM stream mean/final RMS reference PASSED: %zu exact reference elements\n",words);
}
