/* Synthetic cases distinguish finite sigmoid underflow from invalid routing.
 * Exact expected values follow the qualified grouped-topk profile. */
#include <math.h>
static void glm_router_edge_checks(BnGPUBackend *g) {
    float bias[288]={0};
    void *b=bn_gpu_backend_create_quant_only_buffer(g,bias,sizeof(bias),BN_GGUF_TENSOR_F32,1,288);
    assert(b);
    for(int large=0;large<2;++large) {
        int nt=large?1025:1;size_t n=(size_t)nt;
        BnBackendSession *s=bn_backend_session_create();assert(s);
        void *r=bn_gpu_backend_resident_request(g,s);assert(r);
        float *logits=malloc(n*288*sizeof(float)),*weights=malloc(n*8*sizeof(float));
        int32_t *ids=malloc(n*8*sizeof(int32_t));assert(logits&&weights&&ids);
        BnGPUResidentView lv=allocv(g,r,BN_GGUF_TENSOR_F32,n*288,NULL);
        BnGPUResidentView wv=allocv(g,r,BN_GGUF_TENSOR_F32,n*8,NULL);
        BnGPUResidentView iv=allocv(g,r,BN_GGUF_TENSOR_I32,n*8,NULL);
        BnGPUBiasedSigmoidF32ResidentPlan p={.n_tokens=nt,.n_experts=288,.k=8,.route_scale=2.5f,.bias=b,.logits=lv,.weights=wv,.indices=iv};
        /* Bad metadata must leave existing output bytes untouched. */
        for(size_t i=0;i<n*8;++i)weights[i]=17.f;
        assert(!g->resident_write(r,wv,weights));
        BnGPUBiasedSigmoidF32ResidentPlan bad=p;bad.k=9;
        assert(g->moe_biased_sigmoid_f32_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.weights=lv;assert(g->moe_biased_sigmoid_f32_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.route_scale=NAN;assert(g->moe_biased_sigmoid_f32_resident(g->ctx,r,&bad)==-1);
        assert(!g->resident_read(r,wv,weights));for(size_t i=0;i<n*8;++i)assert(weights[i]==17.f);
        for(int profile=0;profile<4;++profile) {
            float value=profile==0?0.f:profile==1?-100.f:profile==2?NAN:INFINITY;
            float expected=profile==0?.3125f:profile==1?0.f:.125f;
            for(size_t i=0;i<n*288;++i)logits[i]=value;
            assert(!g->resident_write(r,lv,logits));
            assert(!g->moe_biased_sigmoid_f32_resident(g->ctx,r,&p));
            assert(!g->resident_read(r,wv,weights));assert(!g->resident_read(r,iv,ids));
            for(size_t i=0;i<n*8;++i){assert(ids[i]==(int)(i%8));assert(!memcmp(&weights[i],&expected,sizeof(float)));}
        }
        free(logits);free(weights);free(ids);bn_backend_session_free(s);
    }
    bn_gpu_backend_destroy_buffer(g,b);
    puts("Synthetic router ties, finite underflow, invalid fallback and metadata PASSED");
}
