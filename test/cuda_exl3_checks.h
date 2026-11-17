/* Model-free complete projection fixtures captured from original exllamav3
 * kernels + its cuBLAS fallback arguments, commit bd1f519172406a9560bd5c90e61000691bb69a07.
 * Native output was not used to generate expected hashes. */
static uint32_t exl3_test_step(uint32_t *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}
static uint64_t exl3_test_hash(const uint16_t *a, size_t count) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; ++i) {
        h = (h ^ (a[i] & 255u)) * UINT64_C(1099511628211);
        h = (h ^ (a[i] >> 8)) * UINT64_C(1099511628211);
    }
    return h;
}
static void run_exl3_projection_reference(BnGPUBackend *gpu) {
    assert(gpu->trellis_projection_f16_workspace_size &&
           gpu->trellis_projection_f16_reserve && gpu->trellis_projection_f16_resident);
    assert(gpu->trellis_projection_f16_workspace_size(gpu->ctx) == 16u*1024u*1024u);
    assert(!gpu->trellis_projection_f16_workspace_size(NULL));
    enum { K=256, N=384, RAW=K*N*3/8 };
    unsigned char packed[RAW]; uint16_t suh[K],svh[N]; uint32_t state=20261116;
    for (int i=0;i<RAW/2;++i) {
        uint16_t w=(uint16_t)exl3_test_step(&state);
        packed[i*2]=(unsigned char)w;packed[i*2+1]=(unsigned char)(w>>8);
    }
    for (int i=0;i<K;++i) suh[i]=bn_fp32_to_fp16(((int)(exl3_test_step(&state)%4097)-2048)/1024.0f);
    for (int i=0;i<N;++i) svh[i]=bn_fp32_to_fp16(((int)(exl3_test_step(&state)%4097)-2048)/1024.0f);
    const int tokens[]={1,7,64,256};
    const uint64_t expected[]={UINT64_C(0x643b7f99c292b077),UINT64_C(0x043db321f32dad75),
        UINT64_C(0x43c1840e5c1d1b65),UINT64_C(0x2bb9bb9f40576189)};
    for (int case_id=0;case_id<4;++case_id) {
        int nt=tokens[case_id];size_t nx=(size_t)nt*K,ny=(size_t)nt*N;
        uint16_t *x=malloc(nx*2),*actual=malloc((ny+16)*2),*guard=malloc((size_t)K*N*2);
        assert(x&&actual&&guard);
        for(size_t i=0;i<nx;i++)x[i]=bn_fp32_to_fp16(((int)(exl3_test_step(&state)%8193)-4096)/2048.0f);
        for(size_t i=0;i<ny+16;i++)actual[i]=0xa55a;
        for(size_t i=0;i<(size_t)K*N;i++)guard[i]=0xa55a;
        void *request=gpu->resident_request_create(gpu->ctx);assert(request);
        BnGPUTrellisProjectionF16ResidentPlan p={0};
        p.format=BN_QUANT_FORMAT_EXL3_MUL1_3;p.codebook_marker=UINT32_C(0x83dcd12d);
        p.k=K;p.n=N;p.n_tokens=nt;
        p.packed=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_I8,RAW,packed);
        p.suh=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_F16,K,suh);
        p.svh=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_F16,N,svh);
        p.x=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_F16,nx,x);
        p.xh=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_F16,nx,NULL);
        p.decoded=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_F16,K*N,guard);
        p.gemm=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_F16,ny,NULL);
        BnGPUResidentView storage=cuda_test_resident_alloc(gpu,request,BN_GGUF_TENSOR_F16,ny+16,actual);
        p.out=bn_gpu_resident_view_slice(storage,8,ny);
        assert(gpu->trellis_projection_f16_resident(gpu->ctx,request,&p)==-1);
        assert(!gpu->trellis_projection_f16_reserve(gpu->ctx,request));
        assert(!gpu->trellis_projection_f16_reserve(gpu->ctx,request));
        BnGPUTrellisProjectionF16ResidentPlan bad=p;
#define EXL3_BAD(field,value) do { bad=p;bad.field=(value);assert(gpu->trellis_projection_f16_resident(gpu->ctx,request,&bad)==-1); } while(0)
        EXL3_BAD(k,0);EXL3_BAD(n,N-1);EXL3_BAD(n_tokens,INT_MAX);
        EXL3_BAD(format,BN_GGUF_TENSOR_Q3_K);EXL3_BAD(codebook_marker,UINT32_C(0xcbac1fed));
        EXL3_BAD(packed.count,RAW-1);EXL3_BAD(decoded.count,K*N-1);
        EXL3_BAD(out.type,BN_GGUF_TENSOR_BF16);EXL3_BAD(xh,p.x);
        EXL3_BAD(out,p.gemm);EXL3_BAD(out,bn_gpu_resident_view_slice(storage,9,ny));
        EXL3_BAD(out.slot,p.out.slot^UINT32_C(0x10000));
        void *peer=gpu->resident_request_create(gpu->ctx);assert(peer);
        BnGPUResidentView foreign=cuda_test_resident_alloc(gpu,peer,BN_GGUF_TENSOR_F16,ny,NULL);
        EXL3_BAD(out,foreign);
        assert(gpu->trellis_projection_f16_resident(gpu->ctx,peer,&p)==-1);
        assert(gpu->trellis_projection_f16_reserve(NULL,request)==-1);
        assert(gpu->trellis_projection_f16_resident(gpu->ctx,request,NULL)==-1);
#undef EXL3_BAD
        assert(!gpu->resident_read(request,storage,actual));
        for(size_t i=0;i<ny+16;i++)assert(actual[i]==0xa55a);
        assert(!gpu->resident_read(request,p.decoded,guard));
        for(size_t i=0;i<(size_t)K*N;i++)assert(guard[i]==0xa55a);
        for(int repeat=0;repeat<2;repeat++) {
            assert(!gpu->trellis_projection_f16_resident(gpu->ctx,request,&p));
            assert(!gpu->resident_read(request,storage,actual));
            assert(exl3_test_hash(actual+8,ny)==expected[case_id]);
            for(int i=0;i<8;i++)assert(actual[i]==0xa55a&&actual[ny+8+i]==0xa55a);
        }
        assert(!gpu->resident_read(request,p.x,guard));assert(!memcmp(guard,x,nx*2));
        gpu->resident_request_destroy(peer);gpu->resident_request_destroy(request);
        free(guard);free(actual);free(x);
    }
    puts("CUDA EXL3 complete FP16 projection original references PASSED");
}
