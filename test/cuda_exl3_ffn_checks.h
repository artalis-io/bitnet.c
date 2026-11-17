/* Frozen complete DQ expert outputs from unchanged original reconstruction,
 * FP16 activation, FP16/F32 Hadamard kernels + original cuBLAS fallback,
 * exllamav3 bd1f519172406a9560bd5c90e61000691bb69a07, SM120.
 * Synthetic RNG seed20261117, independent capture before native FFN code. */
#include "backend_model.h"
typedef struct { BnGPUBackend *gpu; void *weights[3]; } Exl3SharedWeights;
static void exl3_shared_weights_free(void *ptr) {
    Exl3SharedWeights *w=ptr;
    for(int i=0;i<3;i++)w->gpu->buffer_destroy(w->gpu->ctx,w->weights[i]);
    free(w);
}
static uint64_t exl3_ffn_hash(const float *a,size_t count) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<count;i++) {
        uint32_t bits;memcpy(&bits,a+i,4);
        for(int j=0;j<4;j++)h=(h^((bits>>(j*8))&255u))*UINT64_C(1099511628211);
    }
    return h;
}
static void run_exl3_ffn_reference(BnGPUBackend *gpu) {
    assert(gpu->trellis_ffn_f16_resident);
    enum { DIM=256,HIDDEN=384,KN=DIM*HIDDEN,RAW=KN*3/8 };
    unsigned char raw[3][RAW];uint16_t su[3][HIDDEN],sv[3][HIDDEN];
    const int ks[]={DIM,DIM,HIDDEN},ns[]={HIDDEN,HIDDEN,DIM};
    uint32_t state=20261117;
    for(int role=0;role<3;role++) {
        for(int i=0;i<RAW/2;i++){uint16_t w=(uint16_t)exl3_test_step(&state);raw[role][2*i]=(unsigned char)w;raw[role][2*i+1]=(unsigned char)(w>>8);}
        for(int i=0;i<ks[role];i++)su[role][i]=bn_fp32_to_fp16(((int)(exl3_test_step(&state)%4097)-2048)/1024.0f);
        for(int i=0;i<ns[role];i++)sv[role][i]=bn_fp32_to_fp16(((int)(exl3_test_step(&state)%4097)-2048)/1024.0f);
    }
    assert(gpu->trellis_buffer_create&&gpu->trellis_buffer_write);
    BnCudaWeightPoolStats pool_before,pool_after;
    assert(!bn_gpu_cuda_weight_pool_stats(gpu,&pool_before));
    Exl3SharedWeights *shared=calloc(1,sizeof(*shared));assert(shared);shared->gpu=gpu;
    for(int role=0;role<3;role++) {
        size_t bytes=RAW+((size_t)ks[role]+ns[role])*2;
        unsigned char *bundle=malloc(bytes);assert(bundle);
        memcpy(bundle,raw[role],RAW);memcpy(bundle+RAW,su[role],(size_t)ks[role]*2);
        memcpy(bundle+RAW+ks[role]*2,sv[role],(size_t)ns[role]*2);
        assert(!gpu->trellis_buffer_create(gpu->ctx,bundle,bytes-1,BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0x83dcd12d)));
        assert(!gpu->trellis_buffer_create(gpu->ctx,bundle,bytes,BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0xcbac1fed)));
        assert(!gpu->trellis_buffer_create(gpu->ctx,bundle,bytes,BN_GGUF_TENSOR_Q3_K,ks[role],ns[role],UINT32_C(0x83dcd12d)));
        shared->weights[role]=gpu->trellis_buffer_create(gpu->ctx,bundle,bytes,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0x83dcd12d));
        assert(shared->weights[role]);
        void *weight=shared->weights[role];
        assert(gpu->trellis_buffer_write(gpu->ctx,weight,bundle,bytes-1,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0x83dcd12d))==-1);
        assert(gpu->trellis_buffer_write(gpu->ctx,weight,bundle,bytes,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ns[role],ks[role],UINT32_C(0x83dcd12d))==-1);
        assert(gpu->trellis_buffer_write(gpu->ctx,weight,bundle,bytes,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0xcbac1fed))==-1);
        assert(gpu->trellis_buffer_write(gpu->ctx,weight,bundle,bytes,
            BN_GGUF_TENSOR_Q3_K,ks[role],ns[role],UINT32_C(0x83dcd12d))==-1);
        memset(bundle,0,bytes);
        assert(!gpu->trellis_buffer_write(gpu->ctx,weight,bundle,bytes,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0x83dcd12d)));
        memcpy(bundle,raw[role],RAW);memcpy(bundle+RAW,su[role],(size_t)ks[role]*2);
        memcpy(bundle+RAW+ks[role]*2,sv[role],(size_t)ns[role]*2);
        assert(!gpu->trellis_buffer_write(gpu->ctx,weight,bundle,bytes,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0x83dcd12d)));
        free(bundle);
    }
    /* Exercise the same frozen FFN outputs after triplet creation and refill.
     * Upload completion must publish weights to independent consumer streams. */
    assert(gpu->trellis_upload_create&&gpu->trellis_upload_batch&&gpu->trellis_upload_destroy);
    size_t bundle_bytes=RAW+((size_t)DIM+HIDDEN)*2;
    void *pinned=NULL;
    void *upload=gpu->trellis_upload_create(gpu->ctx,3*bundle_bytes,&pinned);
    assert(upload&&pinned);
    BnGPUTrellisUploadItem items[3];
    for(int role=0;role<3;role++) {
        unsigned char *bundle=(unsigned char*)pinned+(size_t)role*bundle_bytes;
        memcpy(bundle,raw[role],RAW);memcpy(bundle+RAW,su[role],(size_t)ks[role]*2);
        memcpy(bundle+RAW+ks[role]*2,sv[role],(size_t)ns[role]*2);
        items[role]=(BnGPUTrellisUploadItem){NULL,(size_t)role*bundle_bytes,bundle_bytes,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ks[role],ns[role],UINT32_C(0x83dcd12d),NULL};
    }
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,0)==-1);
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,4)==-1);
    items[2].offset=3*bundle_bytes;
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,3)==-1);
    assert(!items[0].buffer&&!items[1].buffer&&!items[2].buffer);
    items[2].offset=2*bundle_bytes;items[2].marker=0;
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,3)==-1);
    assert(!items[0].buffer&&!items[1].buffer&&!items[2].buffer);
    items[2].marker=UINT32_C(0x83dcd12d);
    unsigned char *host_projection_bytes=malloc(3*bundle_bytes);assert(host_projection_bytes);
    memcpy(host_projection_bytes,pinned,3*bundle_bytes);
    assert(memcmp(host_projection_bytes,host_projection_bytes+bundle_bytes,bundle_bytes));
    items[0].source=host_projection_bytes+bundle_bytes;items[2].offset=bundle_bytes;
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,3)==-1);
    assert(!items[0].buffer&&!items[1].buffer&&!items[2].buffer);
    assert(!memcmp(host_projection_bytes,pinned,3*bundle_bytes));
    items[2].offset=2*bundle_bytes;items[2].source=(unsigned char*)pinned+bundle_bytes;
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,3)==-1);
    assert(!memcmp(host_projection_bytes,pinned,3*bundle_bytes));
    items[2].source=(const void*)(uintptr_t)(UINTPTR_MAX-bundle_bytes+1);
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,3)==-1);
    assert(!memcmp(host_projection_bytes,pinned,3*bundle_bytes));
    for(int role=0;role<3;role++)items[role].source=host_projection_bytes+(size_t)role*bundle_bytes;
    memset(pinned,0xcc,3*bundle_bytes);
    assert(!gpu->trellis_upload_batch(gpu->ctx,upload,items,3));
    assert(!memcmp(host_projection_bytes,pinned,3*bundle_bytes));
    for(int role=0;role<3;role++) {
        gpu->buffer_destroy(gpu->ctx,shared->weights[role]);
        shared->weights[role]=items[role].buffer;
    }
    void *second=items[1].buffer;items[1].buffer=items[0].buffer;
    assert(gpu->trellis_upload_batch(gpu->ctx,upload,items,3)==-1);
    assert(items[1].buffer==items[0].buffer);items[1].buffer=second;
    for(int role=0;role<3;role++)items[role].source=NULL;
    memset(pinned,0,3*bundle_bytes);
    assert(!gpu->trellis_upload_batch(gpu->ctx,upload,items,3));
    for(int role=0;role<3;role++) {
        unsigned char *bundle=(unsigned char*)pinned+(size_t)role*bundle_bytes;
        memcpy(bundle,raw[role],RAW);memcpy(bundle+RAW,su[role],(size_t)ks[role]*2);
        memcpy(bundle+RAW+ks[role]*2,sv[role],(size_t)ns[role]*2);
    }
    for(int role=1;role<3;role++) {
        items[role].source=host_projection_bytes+(size_t)role*bundle_bytes;
        memset((unsigned char*)pinned+(size_t)role*bundle_bytes,0xcc,bundle_bytes);
    }
    assert(!gpu->trellis_upload_batch(gpu->ctx,upload,items,3));
    assert(!memcmp(host_projection_bytes,pinned,3*bundle_bytes));free(host_projection_bytes);
    gpu->trellis_upload_destroy(gpu->ctx,upload);
    BnBackendModel *owner=bn_backend_model_create();assert(owner);
    bn_backend_model_bind_gpu(owner,gpu);
    assert(!bn_backend_model_attach_gpu_resources(owner,shared,shared,exl3_shared_weights_free));
    assert(bn_backend_model_gpu_resources(owner,shared)==shared);
    const int tokens[]={1,7,64,256};
    const uint64_t expected[4][2]={
        {UINT64_C(0x332367c3a81e8153),UINT64_C(0x2cf1aff737b77535)},
        {UINT64_C(0x0493df811fe49665),UINT64_C(0x3d1efce64aaf2d28)},
        {UINT64_C(0x73d542bf26b4d530),UINT64_C(0x1b92ceebb297536c)},
        {UINT64_C(0xd7080d3c3805deed),UINT64_C(0x5b8ad84036cacb9f)} };
    for(int tc=0;tc<4;tc++) {
        int nt=tokens[tc];size_t nx=(size_t)nt*DIM,nh=(size_t)nt*HIDDEN;
        uint16_t *x=malloc(nx*2),*guard=malloc((KN+16)*2);float*out=malloc((nx+16)*4);
        assert(x&&guard&&out);
        for(size_t i=0;i<nx;i++)x[i]=bn_fp32_to_fp16(((int)(exl3_test_step(&state)%8193)-4096)/2048.0f);
        for(int i=0;i<KN+16;i++)guard[i]=0xa55a;
        for(size_t i=0;i<nx+16;i++)out[i]=12345.0f;
        void*r=gpu->resident_request_create(gpu->ctx);assert(r);
        BnGPUTrellisFFNResidentPlan p={0};
        BnGPUTrellisProjectionF16ResidentPlan *ps[]={&p.gate,&p.up,&p.down};
        BnGPUResidentView xv=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,nx,x);
        BnGPUResidentView xh=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,nh+16,guard);
        BnGPUResidentView decoded=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,KN+16,guard);
        BnGPUResidentView gemm_h=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,nh+16,guard);
        BnGPUResidentView gemm_f=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F32,nx+16,out);
        BnGPUResidentView output=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F32,nx+16,out);
        for(int role=0;role<3;role++) {
            BnGPUTrellisProjectionF16ResidentPlan *q=ps[role];
            q->format=BN_QUANT_FORMAT_EXL3_MUL1_3;q->codebook_marker=UINT32_C(0x83dcd12d);
            q->k=ks[role];q->n=ns[role];q->n_tokens=nt;
            q->packed=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_I8,RAW,raw[role]);
            q->suh=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,ks[role],su[role]);
            q->svh=cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,ns[role],sv[role]);
            q->x=role<2?xv:cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,nh,guard);
            q->xh=bn_gpu_resident_view_slice(xh,8,(size_t)nt*ks[role]);
            q->decoded=bn_gpu_resident_view_slice(decoded,8,KN);
            q->gemm=bn_gpu_resident_view_slice(role<2?gemm_h:gemm_f,8,(size_t)nt*ns[role]);
            q->out=role<2?cuda_test_resident_alloc(gpu,r,BN_GGUF_TENSOR_F16,nh,guard):bn_gpu_resident_view_slice(output,8,nx);
        }
        assert(gpu->trellis_ffn_f16_resident(gpu->ctx,r,&p)==-1);
        assert(!gpu->trellis_projection_f16_reserve(gpu->ctx,r));
        BnGPUTrellisFFNResidentPlan bad;
#define FFN_BAD(field,value) do {bad=p;bad.field=(value);assert(gpu->trellis_ffn_f16_resident(gpu->ctx,r,&bad)==-1);}while(0)
        FFN_BAD(down.codebook_marker,UINT32_C(0xcbac1fed));FFN_BAD(down.gemm.type,BN_GGUF_TENSOR_BF16);
        FFN_BAD(up.n_tokens,nt+1);FFN_BAD(activation_profile,(BnGPUTrellisFFNActivationProfile)1);
        FFN_BAD(act_limit,-1);FFN_BAD(act_limit,NAN);FFN_BAD(act_limit,INFINITY);FFN_BAD(act_limit,65505);
        FFN_BAD(up.x,p.up.xh);FFN_BAD(up.gemm,p.gate.out);FFN_BAD(down.xh,p.up.svh);
        FFN_BAD(up.out,p.gate.out);FFN_BAD(down.x,p.gate.out);
        FFN_BAD(down.out.slot,p.down.out.slot^UINT32_C(0x10000));
        assert(gpu->trellis_ffn_f16_resident(gpu->ctx,r,NULL)==-1);
#undef FFN_BAD
        assert(!gpu->resident_read(r,p.gate.out,guard));for(size_t i=0;i<nh;i++)assert(guard[i]==0xa55a);
        assert(!gpu->resident_read(r,p.up.out,guard));for(size_t i=0;i<nh;i++)assert(guard[i]==0xa55a);
        assert(!gpu->resident_read(r,p.down.x,guard));for(size_t i=0;i<nh;i++)assert(guard[i]==0xa55a);
        assert(!gpu->resident_read(r,output,out));for(size_t i=0;i<nx+16;i++)assert(out[i]==12345.0f);
        for(int clip=0;clip<2;clip++)for(int repeat=0;repeat<2;repeat++) {
            p.act_limit=clip?10.0f:0.0f;assert(!gpu->trellis_ffn_f16_resident(gpu->ctx,r,&p));
            assert(!gpu->resident_read(r,output,out));assert(exl3_ffn_hash(out+8,nx)==expected[tc][clip]);
            for(int i=0;i<8;i++)assert(out[i]==12345.0f&&out[nx+8+i]==12345.0f);
        }
        BnGPUTrellisFFNResidentPlan borrowed=p;
        BnGPUTrellisProjectionF16ResidentPlan *bp[]={&borrowed.gate,&borrowed.up,&borrowed.down};
        for(int role=0;role<3;role++) {
            bp[role]->weight=shared->weights[role];
            /* A weight handle together with old resident weights is ambiguous. */
            if(role==0)assert(gpu->trellis_ffn_f16_resident(gpu->ctx,r,&borrowed)==-1);
            bp[role]->packed=(BnGPUResidentView){0};bp[role]->suh=(BnGPUResidentView){0};bp[role]->svh=(BnGPUResidentView){0};
        }
        bad=borrowed;bad.gate.weight=shared->weights[2];assert(gpu->trellis_ffn_f16_resident(gpu->ctx,r,&bad)==-1);
        if(tc==0) {
            BnGPUBackend *foreign=bn_gpu_cuda_create();assert(foreign);
            size_t bytes=RAW+((size_t)ks[0]+ns[0])*2;
            unsigned char *bundle=malloc(bytes);assert(bundle);
            memcpy(bundle,raw[0],RAW);memcpy(bundle+RAW,su[0],ks[0]*2);
            memcpy(bundle+RAW+ks[0]*2,sv[0],ns[0]*2);
            void *other=foreign->trellis_buffer_create(foreign->ctx,bundle,bytes,
                BN_QUANT_FORMAT_EXL3_MUL1_3,ks[0],ns[0],UINT32_C(0x83dcd12d));assert(other);
            assert(gpu->trellis_buffer_write(gpu->ctx,other,bundle,bytes,
                BN_QUANT_FORMAT_EXL3_MUL1_3,ks[0],ns[0],UINT32_C(0x83dcd12d))==-1);
            assert(foreign->trellis_buffer_write(foreign->ctx,shared->weights[0],bundle,bytes,
                BN_QUANT_FORMAT_EXL3_MUL1_3,ks[0],ns[0],UINT32_C(0x83dcd12d))==-1);
            bad=borrowed;bad.gate.weight=other;
            assert(gpu->trellis_ffn_f16_resident(gpu->ctx,r,&bad)==-1);
            foreign->buffer_destroy(foreign->ctx,other);bn_gpu_cuda_destroy(foreign);free(bundle);
        }
        void *peer_r=gpu->resident_request_create(gpu->ctx);assert(peer_r);
        BnGPUTrellisFFNResidentPlan peer=borrowed;
        BnGPUTrellisProjectionF16ResidentPlan *pp[]={&peer.gate,&peer.up,&peer.down};
        BnGPUResidentView peer_x=cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F16,nx,x);
        BnGPUResidentView peer_xh=cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F16,nh,NULL);
        BnGPUResidentView peer_w=cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F16,KN,NULL);
        BnGPUResidentView peer_gh=cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F16,nh,NULL);
        BnGPUResidentView peer_gf=cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F32,nx,NULL);
        BnGPUResidentView peer_a=cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F16,nh,NULL);
        for(size_t i=0;i<nx+16;i++)out[i]=12345.0f;
        BnGPUResidentView peer_out=cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F32,nx+16,out);
        for(int role=0;role<3;role++) {
            pp[role]->x=role<2?peer_x:peer_a;
            pp[role]->xh=bn_gpu_resident_view_slice(peer_xh,0,(size_t)nt*ks[role]);
            pp[role]->decoded=peer_w;
            pp[role]->gemm=role<2?peer_gh:peer_gf;
            pp[role]->out=role<2?cuda_test_resident_alloc(gpu,peer_r,BN_GGUF_TENSOR_F16,nh,NULL):bn_gpu_resident_view_slice(peer_out,8,nx);
        }
        assert(!gpu->trellis_projection_f16_reserve(gpu->ctx,peer_r));
        borrowed.act_limit=10;peer.act_limit=0;
        assert(!gpu->trellis_ffn_f16_resident(gpu->ctx,r,&borrowed));
        assert(!gpu->trellis_ffn_f16_resident(gpu->ctx,peer_r,&peer));
        assert(!gpu->resident_read(r,output,out));assert(exl3_ffn_hash(out+8,nx)==expected[tc][1]);
        assert(!gpu->resident_read(peer_r,peer_out,out));assert(exl3_ffn_hash(out+8,nx)==expected[tc][0]);
        for(int i=0;i<8;i++)assert(out[i]==12345.0f&&out[nx+8+i]==12345.0f);
        peer.act_limit=10;assert(!gpu->trellis_ffn_f16_resident(gpu->ctx,peer_r,&peer));
        assert(!gpu->resident_read(peer_r,peer_out,out));assert(exl3_ffn_hash(out+8,nx)==expected[tc][1]);
        assert(!gpu->resident_read(r,output,out));assert(exl3_ffn_hash(out+8,nx)==expected[tc][1]);
        /* Consume a fresh upload on an already reserved non-blocking request,
         * with no intervening allocation or copy to hide upload ordering. */
        size_t cold_bytes=RAW+((size_t)ks[0]+ns[0])*2;
        unsigned char *cold_bundle=malloc(cold_bytes);assert(cold_bundle);
        memcpy(cold_bundle,raw[0],RAW);memcpy(cold_bundle+RAW,su[0],ks[0]*2);
        memcpy(cold_bundle+RAW+ks[0]*2,sv[0],ns[0]*2);
        void *cold= gpu->trellis_buffer_create(gpu->ctx,cold_bundle,cold_bytes,
            BN_QUANT_FORMAT_EXL3_MUL1_3,ks[0],ns[0],UINT32_C(0x83dcd12d));assert(cold);
        borrowed.gate.weight=cold;
        assert(!gpu->trellis_ffn_f16_resident(gpu->ctx,r,&borrowed));
        assert(!gpu->resident_read(r,output,out));assert(exl3_ffn_hash(out+8,nx)==expected[tc][1]);
        gpu->buffer_destroy(gpu->ctx,cold);free(cold_bundle);
        gpu->resident_request_destroy(peer_r);
        assert(!gpu->resident_read(r,decoded,guard));for(int i=0;i<8;i++)assert(guard[i]==0xa55a&&guard[KN+8+i]==0xa55a);
        assert(!gpu->resident_read(r,xv,guard));assert(!memcmp(guard,x,nx*2));
        gpu->resident_request_destroy(r);free(out);free(guard);free(x);
    }
    bn_backend_model_free(owner);
    assert(!bn_gpu_cuda_weight_pool_stats(gpu,&pool_after));
    assert(pool_after.idle_bytes==pool_before.idle_bytes && pool_after.idle_entries==pool_before.idle_entries);
    puts("CUDA EXL3 expert FFN original references + model-owned shared weights PASSED");
}
