#include "cuda_native_frequency_reference.h"
static void run_native_frequency_reference(BnGPUBackend*g) {
    assert(g->rope_frequency_f32_resident);size_t calls=0,words=0;
    for(size_t i=0;i<sizeof(native_frequency_reference)/sizeof(*native_frequency_reference);i++) {
        const NativeFrequencyReference*f=&native_frequency_reference[i];size_t n=(size_t)f->dim/2;float start,scale;memcpy(&start,&f->start,4);memcpy(&scale,&f->scale,4);
        uint32_t*negative=malloc(n*4);assert(negative);for(size_t j=0;j<n;j++)negative[j]=f->words[j]^UINT32_C(0x80000000);
        void*r[2];BnGPUResidentView roots[2][2];BnGPUF32RopeFrequencyResidentPlan p[2];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView out=native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,n,NULL,&roots[q][0]),neg=native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,n,NULL,&roots[q][1]);
            p[q]=(BnGPUF32RopeFrequencyResidentPlan){f->dim,f->yarn,f->base,f->factor,start,scale,out,neg};BnGPUF32RopeFrequencyResidentPlan bad;
#define FREQ_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->rope_frequency_f32_resident(g->ctx,r[q],&bad)==-1);}while(0)
            FREQ_BAD(rotary_dim,1);FREQ_BAD(rotary_dim,2049);FREQ_BAD(yarn,2);FREQ_BAD(base,1);FREQ_BAD(base,NAN);FREQ_BAD(factor,0);FREQ_BAD(ramp_start,-1);FREQ_BAD(ramp_scale,0);FREQ_BAD(out.type,BN_GGUF_TENSOR_F16);FREQ_BAD(negative.count,n+1);FREQ_BAD(negative,p[q].out);FREQ_BAD(negative,bn_gpu_resident_view_slice(roots[q][0],17,n));
#undef FREQ_BAD
            for(int j=0;j<2;j++)native_router_guard_check(g,r[q],roots[q][j],n,BN_GGUF_TENSOR_F32,folded_test_unwritten(n*4));
        }
        BnGPUF32RopeFrequencyResidentPlan bad=p[0];bad.negative=p[1].negative;assert(g->rope_frequency_f32_resident(g->ctx,r[0],&bad)==-1);
        for(int rep=0;rep<2;rep++)for(int q=0;q<2;q++) {
            assert(!g->rope_frequency_f32_resident(g->ctx,r[q],p+q));native_router_guard_check(g,r[q],roots[q][0],n,BN_GGUF_TENSOR_F32,native_norm_hash(f->words,n*4));native_router_guard_check(g,r[q],roots[q][1],n,BN_GGUF_TENSOR_F32,native_norm_hash(negative,n*4));calls++;words+=n*2;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}free(negative);
    }
    printf("CUDA source F32 frequency tables/negation/guards PASSED: %zu calls, %zu words\n",calls,words);
}
