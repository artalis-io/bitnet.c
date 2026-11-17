#include "cuda_native_paged_latents_reference.h"
static void run_native_paged_latents_reference(BnGPUBackend*g) {
    assert(g->paged_latent_publish_f16_resident);size_t calls=0;
    for(size_t c=0;c<sizeof(native_paged_latents_fixtures)/sizeof(*native_paged_latents_fixtures);c++) {
        int R=native_paged_latents_fixtures[c].R,D=native_paged_latents_fixtures[c].D,RD=native_paged_latents_fixtures[c].RD,I=native_paged_latents_fixtures[c].I;
        int epp=256/native_paged_latents_fixtures[c].rate;size_t cap=(size_t)8*epp;
        uint16_t*x=malloc((size_t)R*D*2),*k=malloc((size_t)R*I*2);assert(x&&k);int32_t bt[8];
        for(size_t i=0;i<(size_t)R*D;i++)x[i]=(uint16_t)((i*73+19)%65536-32768);
        for(size_t i=0;i<(size_t)R*I;i++)k[i]=(uint16_t)((i*37+41)%65536-32768);
        for(int i=0;i<8;i++)bt[i]=native_paged_latents_fixtures[c].perm?(i*5+3)%8:i;
        void*r[2];BnGPUPagedLatentPublishResidentPlan p[2];BnGPUResidentView storage[2][6];size_t counts[]={(size_t)R*D,(size_t)R*I,8,cap*(D-RD),cap*RD,cap*I};
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[6];
            for(int j=0;j<6;j++)v[j]=native_router_guard(g,r[q],j==2?BN_GGUF_TENSOR_I32:BN_GGUF_TENSOR_F16,counts[j],j==0?(const void*)x:j==1?(const void*)k:j==2?(const void*)bt:NULL,&storage[q][j]);
            p[q]=(BnGPUPagedLatentPublishResidentPlan){R,native_paged_latents_fixtures[c].entry0,epp,8,D,RD,I,v[0],v[1],v[2],v[3],v[4],v[5]};
            BnGPUPagedLatentPublishResidentPlan bad;
#define POOL_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->paged_latent_publish_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            POOL_BAD(n_rows,0);POOL_BAD(entry0,-1);POOL_BAD(entry0,INT_MAX);POOL_BAD(entries_per_page,0);POOL_BAD(pages,INT_MAX);POOL_BAD(rotary_dim,D);POOL_BAD(index_dim,INT_MAX);POOL_BAD(pool_idx.count,p[q].pool_idx.count+1);POOL_BAD(block_table.count,7);POOL_BAD(pool_idx,p[q].pool_c);POOL_BAD(latents,p[q].pool_c);
#undef POOL_BAD
            for(int j=3;j<6;j++)native_router_guard_check(g,r[q],storage[q][j],counts[j],BN_GGUF_TENSOR_F16,folded_test_unwritten(counts[j]*2));
        }
        BnGPUPagedLatentPublishResidentPlan bad=p[0];bad.block_table=p[1].block_table;assert(g->paged_latent_publish_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int q=0;q<2;q++) {
            assert(!g->paged_latent_publish_f16_resident(g->ctx,r[q],p+q));
            for(int j=3;j<6;j++)native_router_guard_check(g,r[q],storage[q][j],counts[j],BN_GGUF_TENSOR_F16,native_paged_latents_fixtures[c].hash[j-3]);
            native_router_guard_check(g,r[q],storage[q][0],counts[0],BN_GGUF_TENSOR_F16,native_norm_hash(x,counts[0]*2));
            native_router_guard_check(g,r[q],storage[q][1],counts[1],BN_GGUF_TENSOR_F16,native_norm_hash(k,counts[1]*2));
            native_router_guard_check(g,r[q],storage[q][2],8,BN_GGUF_TENSOR_I32,native_norm_hash(bt,32));calls++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}free(x);free(k);
    }
    printf("CUDA source paged latent publication PASSED: %zu calls\n",calls);
}
