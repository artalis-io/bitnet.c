#include "cuda_native_latent_keys_reference.h"
static void run_native_latent_keys_reference(BnGPUBackend*g) {
    assert(g->latent_keys_f16_resident);F16FFNSource source={128,128,13,0};size_t tile=32*128*2;
    BnGPUF16RowsPlan wp={128,128,128*128*2+tile,tile,&source,f16_ffn_read};void*w=g->buffer_create_f16_rows(g->ctx,&wp);assert(w);size_t calls=0;
    for(size_t c=0;c<sizeof(native_latent_keys_fixtures)/sizeof(*native_latent_keys_fixtures);c++) {
        int R=native_latent_keys_fixtures[c].R,e0=native_latent_keys_fixtures[c].entry0;size_t n=(size_t)R*128;
        uint16_t*x=malloc(n*2);assert(x);for(size_t i=0;i<n;i++)x[i]=bn_fp32_to_fp16((float)((int)((i*73+(size_t)e0*17+19)%1021)-510)/1024);
        void*gamma=native_qkv_gamma(g,128,native_latent_keys_fixtures[c].bf16);void*r[2];BnGPUF16LatentKeysResidentPlan p[2];BnGPUResidentView storage[2][5];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[5];
            for(int j=0;j<3;j++)v[j]=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,n,j==0?x:NULL,&storage[q][j]);
            v[3]=native_router_guard(g,r[q],BN_GGUF_TENSOR_I32,R,NULL,&storage[q][3]);v[4]=native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,32,native_rope_freq1,&storage[q][4]);
            p[q]=(BnGPUF16LatentKeysResidentPlan){e0,2,{w,128,128,R,v[0],v[1]},
                {R,128,1,1e-6f,0,1,gamma,v[1],v[2]},
                {R,1,64,128,64,64,0,0,1e-6f,NULL,v[2],v[4],v[3]},
                {R,1,64,128,64,64,0,0,1e-6f,NULL,v[0],v[4],v[3]}};
            assert(g->latent_keys_f16_resident(g->ctx,r[q],p+q)==-1);assert(!g->linear_f16_reserve(g->ctx,r[q]));
            BnGPUF16LatentKeysResidentPlan bad;
#define KEYS_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->latent_keys_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            KEYS_BAD(entry0,-1);KEYS_BAD(entry0,INT_MAX);KEYS_BAD(rate,0);KEYS_BAD(rate,-1);KEYS_BAD(project.n_tokens,INT_MIN);KEYS_BAD(norm.weight,NULL);KEYS_BAD(norm.dim,124);KEYS_BAD(norm.constant_scale,2);KEYS_BAD(norm.out,p[q].norm.x);KEYS_BAD(keys_rope.positions,(BnGPUResidentView){0});KEYS_BAD(latents_rope.rotate_offset,INT_MAX);KEYS_BAD(latents_rope.norm,gamma);KEYS_BAD(latents_rope.inv_freq,p[q].project.x);KEYS_BAD(latents_rope.head_stride,INT_MIN);KEYS_BAD(keys_rope.pos0,1);KEYS_BAD(keys_rope.x,p[q].project.x);
#undef KEYS_BAD
            native_router_guard_check(g,r[q],storage[q][0],n,BN_GGUF_TENSOR_F16,native_norm_hash(x,n*2));
            for(int j=1;j<3;j++)native_router_guard_check(g,r[q],storage[q][j],n,BN_GGUF_TENSOR_F16,folded_test_unwritten(n*2));
            native_router_guard_check(g,r[q],storage[q][3],R,BN_GGUF_TENSOR_I32,folded_test_unwritten((size_t)R*4));
        }
        BnGPUF16LatentKeysResidentPlan bad=p[0];bad.latents_rope.positions=p[1].latents_rope.positions;assert(g->latent_keys_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int q=0;q<2;q++)for(int repeat=0;repeat<2;repeat++) {
            assert(!g->resident_write(r[q],p[q].project.x,x)&&!g->latent_keys_f16_resident(g->ctx,r[q],p+q));
            for(int j=0;j<3;j++)native_router_guard_check(g,r[q],storage[q][j],n,BN_GGUF_TENSOR_F16,native_latent_keys_fixtures[c].hash[j==0?2:j==1?0:1]);
            native_router_guard_check(g,r[q],storage[q][3],R,BN_GGUF_TENSOR_I32,native_latent_keys_fixtures[c].hash[3]);
            native_router_guard_check(g,r[q],storage[q][4],32,BN_GGUF_TENSOR_F32,native_norm_hash(native_rope_freq1,128));calls++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}g->buffer_destroy(g->ctx,gamma);free(x);
    }
    g->buffer_destroy(g->ctx,w);printf("CUDA source latent index keys reference PASSED: %zu calls\n",calls);
}
