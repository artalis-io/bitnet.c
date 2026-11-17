#include "cuda_mhc_reference.h"
static void run_mhc_reference_case(BnGPUBackend *gpu) {
    if(!gpu->mhc_pre_bf16_resident){printf("CUDA mHC SM120 reference skipped (profile unavailable)\n");return;}
    enum {dim=4096,wide=16384,splits=64};
    uint16_t *res=malloc((wide+2)*2),*branch=malloc((dim+2)*2),*norm=malloc(dim*2);
    float *fn=malloc((size_t)24*wide*4);assert(res&&branch&&norm&&fn);
    res[0]=res[wide+1]=branch[0]=branch[dim+1]=0x5a5a;
    for(int i=0;i<wide;++i)res[i+1]=cuda_test_bf16_rne((float)((i*17+13)%257-128)*0x1p-5f);
    for(int i=0;i<dim;++i) {
        branch[i+1]=cuda_test_bf16_rne((float)((i*29+3)%257-128)*0x1p-6f);
        norm[i]=cuda_test_bf16_rne(1.0f+(float)((i*11)%33-16)*0x1p-7f);
    }
    for(int r=0;r<24;++r)for(int k=0;k<wide;++k)
        fn[(size_t)r*wide+k]=(float)((r*65537+k*31337+19)%65521-32760)*0x1p-20f;
    float scale[]={0.3f,-0.7f,1.2f},base[24];for(int i=0;i<24;++i)base[i]=(float)((i*7)%31-15)*0x1p-4f;
    void *fw=gpu->buffer_create_quant_only(gpu->ctx,fn,(size_t)24*wide*4,BN_GGUF_TENSOR_F32,24,wide);
    void *sw=gpu->buffer_create_quant_only(gpu->ctx,scale,sizeof(scale),BN_GGUF_TENSOR_F32,1,3);
    void *bw=gpu->buffer_create_quant_only(gpu->ctx,base,sizeof(base),BN_GGUF_TENSOR_F32,1,24);
    void *nw=gpu->buffer_create_quant_only(gpu->ctx,norm,dim*2,BN_GGUF_TENSOR_BF16,1,dim);assert(fw&&sw&&bw&&nw);
    BnBackendSession *session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(gpu,session);assert(r);
    BnGPUResidentView rw=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_BF16,wide+2,res);
    BnGPUResidentView br=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_BF16,dim+2,branch);
    size_t counts[]={splits*24,splits,4,4,16,dim,dim,wide};
    int types[]={BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};
    BnGPUResidentView whole[8],v[8];
    for(int i=0;i<8;++i){size_t word=types[i]==BN_GGUF_TENSOR_F32?4:2;void*sentinel=malloc((counts[i]+2)*word);assert(sentinel);memset(sentinel,0x5a,(counts[i]+2)*word);whole[i]=cuda_test_resident_alloc(gpu,r,types[i],counts[i]+2,sentinel);v[i]=bn_gpu_resident_view_slice(whole[i],1,counts[i]);free(sentinel);}
    BnGPUMHCPreResidentPlan p={.n_tokens=1,.dim=dim,.streams=4,.n_splits=splits,.iterations=20,
        .rms_eps=1e-5f,.pre_eps=1e-6f,.sinkhorn_eps=1e-6f,.post_mult=2,.norm_eps=1e-5f,
        .fn=fw,.scale=sw,.base=bw,.norm=nw,.residual=bn_gpu_resident_view_slice(rw,1,wide),
        .dot=v[0],.sqrsum=v[1],.pre=v[2],.post=v[3],.comb=v[4],.collapsed=v[5],.out=v[6]};
    BnGPUMHCPreResidentPlan bad=p;bad.out.count--;assert(gpu->mhc_pre_bf16_resident(gpu->ctx,r,&bad)==-1);
    bad=p;bad.dot=bad.sqrsum;assert(gpu->mhc_pre_bf16_resident(gpu->ctx,r,&bad)==-1);
    for(int i=0;i<8;++i){size_t bytes=(counts[i]+2)*(types[i]==BN_GGUF_TENSOR_F32?4:2);unsigned char*got=malloc(bytes);assert(got);assert(!gpu->resident_read(r,whole[i],got));for(size_t j=0;j<bytes;++j)assert(got[j]==0x5a);free(got);}
    BnGPUMHCPostResidentPlan q={.n_tokens=1,.dim=dim,.streams=4,.residual=p.residual,
        .branch=bn_gpu_resident_view_slice(br,1,dim),.post=p.post,.comb=p.comb,.out=v[7]};
    assert(!gpu->mhc_pre_bf16_resident(gpu->ctx,r,&p));assert(!gpu->mhc_post_bf16_resident(gpu->ctx,r,&q));assert(!gpu->resident_synchronize(r));
    const void *expected[]={cuda_mhc_ref_dot,cuda_mhc_ref_sq,cuda_mhc_ref_pre,cuda_mhc_ref_post,cuda_mhc_ref_comb,cuda_mhc_ref_collapsed,cuda_mhc_ref_normalized,cuda_mhc_ref_out};
    size_t words=0;for(int i=0;i<8;++i){size_t word=types[i]==BN_GGUF_TENSOR_F32?4:2;unsigned char*got=malloc((counts[i]+2)*word);assert(got);assert(!gpu->resident_read(r,whole[i],got));for(size_t j=0;j<word;++j){assert(got[j]==0x5a&&got[(counts[i]+1)*word+j]==0x5a);}assert(!memcmp(got+word,expected[i],counts[i]*word));free(got);words+=counts[i];}
    p.norm=NULL;assert(!gpu->mhc_pre_bf16_resident(gpu->ctx,r,&p));uint16_t *got=malloc((wide+2)*2);assert(got);assert(!gpu->resident_read(r,p.out,got));assert(!memcmp(got,cuda_mhc_ref_collapsed,dim*2));words+=dim;
    assert(!gpu->resident_read(r,rw,got));assert(!memcmp(got,res,(wide+2)*2));assert(!gpu->resident_read(r,br,got));assert(!memcmp(got,branch,(dim+2)*2));free(got);
    bn_backend_session_free(session);gpu->buffer_destroy(gpu->ctx,fw);gpu->buffer_destroy(gpu->ctx,sw);gpu->buffer_destroy(gpu->ctx,bw);gpu->buffer_destroy(gpu->ctx,nw);
    free(res);free(branch);free(norm);free(fn);printf("CUDA mHC original SM120 reference PASSED: %zu exact words and canaries\n",words);
}
