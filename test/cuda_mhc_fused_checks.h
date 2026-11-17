#include "cuda_mhc_fused_reference.h"
static uint64_t mhc_fused_hash(const unsigned char *data,size_t bytes) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;++i)h=(h^data[i])*UINT64_C(1099511628211);
    return h;
}
static void run_mhc_fused_reference_case(BnGPUBackend*g) {
    if(!g->mhc_fused_post_pre_bf16_resident){puts("CUDA default fused mHC reference skipped (profile unavailable)");return;}
    enum {dim=4096,wide=16384};size_t words=0;
    float *fn=malloc((size_t)24*wide*4);uint16_t *norm=malloc(dim*2);assert(fn&&norm);
    for(int row=0;row<24;++row)for(int k=0;k<wide;++k)fn[(size_t)row*wide+k]=(float)((row*65537+k*31337+19)%65521-32760)*0x1p-20f;
    for(int i=0;i<dim;++i)norm[i]=cuda_test_bf16_rne(1+(float)((i*11)%33-16)*0x1p-7f);
    float scale[]={.3f,-.7f,1.2f},base[24];for(int i=0;i<24;++i)base[i]=(float)((i*7)%31-15)*0x1p-4f;
    void *fw=g->buffer_create_quant_only(g->ctx,fn,(size_t)24*wide*4,BN_GGUF_TENSOR_F32,24,wide),
        *nw=g->buffer_create_quant_only(g->ctx,norm,dim*2,BN_GGUF_TENSOR_BF16,1,dim),
        *sw=g->buffer_create_quant_only(g->ctx,scale,sizeof(scale),BN_GGUF_TENSOR_F32,1,3),
        *bw=g->buffer_create_quant_only(g->ctx,base,sizeof(base),BN_GGUF_TENSOR_F32,1,24);assert(fw&&nw&&sw&&bw);
    for(size_t ci=0;ci<sizeof(cuda_mhc_fused_refs)/sizeof(cuda_mhc_fused_refs[0]);++ci) {
        const BnCudaMHCFusedReference*ref=&cuda_mhc_fused_refs[ci];int nt=ref->tokens;size_t n=nt;
        uint16_t *res=malloc((n*wide+2)*2),*branch=malloc((n*dim+2)*2);assert(res&&branch);
        res[0]=res[n*wide+1]=branch[0]=branch[n*dim+1]=0x5a5a;
        for(size_t i=0;i<n*wide;++i)res[i+1]=cuda_test_bf16_rne((float)((int)((i*17+13)%257)-128)*0x1p-5f);
        for(size_t i=0;i<n*dim;++i)branch[i+1]=cuda_test_bf16_rne((float)((int)((i*29+3)%257)-128)*0x1p-6f);
        BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);
        BnGPUResidentView rw=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_BF16,n*wide+2,res),br=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_BF16,n*dim+2,branch);
        BnGPUMHCPreResidentPlan old={.n_tokens=nt,.dim=dim,.streams=4,.n_splits=64,.iterations=20,
            .rms_eps=1e-5f,.pre_eps=1e-6f,.sinkhorn_eps=1e-6f,.post_mult=2,.norm_eps=1e-5f,
            .fn=fw,.norm=NULL,.scale=sw,.base=bw,.residual=bn_gpu_resident_view_slice(rw,1,n*wide)};
        old.dot=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_F32,n*64*24,NULL);old.sqrsum=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_F32,n*64,NULL);
        old.pre=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_F32,n*4,NULL);old.post=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_F32,n*4,NULL);old.comb=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_F32,n*16,NULL);
        old.collapsed=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_BF16,n*dim,NULL);old.out=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_BF16,n*dim,NULL);
        BnGPUMHCFusedResidentPlan p={.next=old,.previous_residual=old.residual,.previous_branch=bn_gpu_resident_view_slice(br,1,n*dim),.previous_post=old.post,.previous_comb=old.comb};
        p.next.n_splits=ref->splits;p.next.norm=nw;
        size_t counts[]={n*wide,n*ref->splits*24,n*ref->splits,n*4,n*4,n*16,n*dim,n*dim};
        int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};
        BnGPUResidentView whole[8],v[8];for(int i=0;i<8;++i){size_t bytes=(counts[i]+2)*(types[i]==BN_GGUF_TENSOR_F32?4:2);void*s=malloc(bytes);assert(s);memset(s,0x5a,bytes);whole[i]=cuda_test_resident_alloc(g,r,types[i],counts[i]+2,s);v[i]=bn_gpu_resident_view_slice(whole[i],1,counts[i]);free(s);}
        p.next.residual=v[0];p.next.dot=v[1];p.next.sqrsum=v[2];p.next.pre=v[3];p.next.post=v[4];p.next.comb=v[5];p.next.collapsed=v[6];p.next.out=v[7];
        assert(!g->mhc_pre_bf16_resident(g->ctx,r,&old));
        BnGPUMHCFusedResidentPlan bad=p;bad.next.out.count--;assert(g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.previous_residual=p.next.residual;assert(g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&bad)==-1);
        for(int i=0;i<8;++i){size_t bytes=(counts[i]+2)*(types[i]==BN_GGUF_TENSOR_F32?4:2);unsigned char*s=malloc(bytes);assert(s);assert(!g->resident_read(r,whole[i],s));for(size_t k=0;k<bytes;++k)assert(s[k]==0x5a);free(s);}
        assert(!g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&p));assert(!g->resident_synchronize(r));
        for(int i=0;i<8;++i){size_t word=types[i]==BN_GGUF_TENSOR_F32?4:2;unsigned char*s=malloc((counts[i]+2)*word);assert(s);assert(!g->resident_read(r,whole[i],s));for(size_t k=0;k<word;++k)assert(s[k]==0x5a&&s[(counts[i]+1)*word+k]==0x5a);assert(mhc_fused_hash(s+word,counts[i]*word)==ref->hash[i]);free(s);words+=counts[i];}
        p.next.norm=NULL;assert(!g->mhc_fused_post_pre_bf16_resident(g->ctx,r,&p));unsigned char*s=malloc(n*wide*2+4);assert(s);assert(!g->resident_read(r,p.next.out,s));assert(mhc_fused_hash(s,n*dim*2)==ref->hash[6]);words+=n*dim;
        assert(!g->resident_read(r,rw,s));assert(!memcmp(s,res,(n*wide+2)*2));assert(!g->resident_read(r,br,s));assert(!memcmp(s,branch,(n*dim+2)*2));free(s);
        bn_backend_session_free(session);free(res);free(branch);
    }
    g->buffer_destroy(g->ctx,fw);g->buffer_destroy(g->ctx,nw);g->buffer_destroy(g->ctx,sw);g->buffer_destroy(g->ctx,bw);free(fn);free(norm);
    printf("CUDA default fused mHC original reference PASSED: %zu exact-byte hash words, canaries and dispatch boundaries\n",words);
}
