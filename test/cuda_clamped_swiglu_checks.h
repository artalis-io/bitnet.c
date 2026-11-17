#include "cuda_clamped_swiglu_reference.h"
static void run_clamped_swiglu_reference_case(BnGPUBackend*g) {
    if(!g->clamped_swiglu_bf16_resident){puts("CUDA clamped SwiGLU reference skipped (profile unavailable)");return;}
    enum {h=32768};uint16_t*x=malloc((2*h+2)*2),*y=malloc((h+2)*2);assert(x&&y);
    for(size_t ci=0;ci<sizeof(cuda_clamped_swiglu_refs)/sizeof(cuda_clamped_swiglu_refs[0]);++ci){
        const BnCudaClampedSwiGLUReference*ref=&cuda_clamped_swiglu_refs[ci];
        x[0]=x[2*h+1]=0x5a5a;for(int i=0;i<h+2;++i)y[i]=0x5a5a;
        for(int i=0;i<h;++i){uint16_t bits=(uint16_t)(i+ref->sign*32768);x[i+1]=(bits&0x7f80)==0x7f80?0:bits;
            x[h+i+1]=cuda_test_bf16_rne((float)((i*37)%251-125)/8);}
        BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
        BnGPUResidentView xv=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_BF16,2*h+2,x),yv=cuda_test_resident_alloc(g,r,BN_GGUF_TENSOR_BF16,h+2,y);
        BnGPUClampedSwiGLUResidentPlan p={.n_tokens=1,.hidden_dim=h,.limit=ref->limit,.gate_up=bn_gpu_resident_view_slice(xv,1,2*h),.out=bn_gpu_resident_view_slice(yv,1,h)},bad=p;
        bad.limit=NAN;assert(g->clamped_swiglu_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.limit=-1;assert(g->clamped_swiglu_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.out=bn_gpu_resident_view_slice(xv,1,h);assert(g->clamped_swiglu_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.gate_up.count--;assert(g->clamped_swiglu_bf16_resident(g->ctx,r,&bad)==-1);
        bad=p;bad.out.type=BN_GGUF_TENSOR_F32;assert(g->clamped_swiglu_bf16_resident(g->ctx,r,&bad)==-1);
        assert(!g->resident_read(r,yv,y));for(int i=0;i<h+2;++i)assert(y[i]==0x5a5a);
        assert(!g->clamped_swiglu_bf16_resident(g->ctx,r,&p));assert(!g->resident_read(r,yv,y));
        assert(y[0]==0x5a5a&&y[h+1]==0x5a5a&&bf16_output_hash(y+1,h)==ref->hash);
        assert(!g->resident_read(r,xv,x));assert(x[0]==0x5a5a&&x[2*h+1]==0x5a5a);
        for(int i=0;i<h;++i){uint16_t bits=(uint16_t)(i+ref->sign*32768);assert(x[i+1]==((bits&0x7f80)==0x7f80?0:bits));assert(x[h+i+1]==cuda_test_bf16_rne((float)((i*37)%251-125)/8));}
        bn_backend_session_free(s);
    }
    free(x);free(y);puts("CUDA clamped SwiGLU finite BF16 gates, clamp rounding, signed zeros and canaries PASSED (262144 output words)");
}
