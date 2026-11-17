#include "cuda_native_swa_ring_reference.h"
static void run_native_swa_ring_reference(BnGPUBackend*g) {
    assert(g->swa_ring_update_f16_resident);size_t calls=0;
    for(int config=0;config<4;config++) {
        size_t first=(size_t)config*10;int B=native_swa_ring_fixtures[first].B,D=native_swa_ring_fixtures[first].D;
        void*r[2];BnGPUResidentView ring[2],scratch[2],ring_root[2],scratch_root[2];
        for(int q=0;q<2;q++){r[q]=g->resident_request_create(g->ctx);assert(r[q]);ring[q]=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,(size_t)B*D,NULL,&ring_root[q]);scratch[q]=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,(size_t)B*D,NULL,&scratch_root[q]);}
        for(size_t f=first;f<first+10;f++) {
            int R=native_swa_ring_fixtures[f].R,pos=native_swa_ring_fixtures[f].pos;size_t n=(size_t)R*D;
            uint16_t*x=malloc(n*2);assert(x);for(int row=0;row<R;row++)for(int c=0;c<D;c++)x[(size_t)row*D+c]=(uint16_t)((((size_t)(pos+row)*31+c*73+19)%65536)-32768);
            assert(native_norm_hash(x,n*2)==native_swa_ring_fixtures[f].hash[0]);
            BnGPUSWARingUpdateResidentPlan p[2];BnGPUResidentView root[2];
            for(int q=0;q<2;q++) {
                BnGPUResidentView kv=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,n,x,&root[q]);
                p[q]=(BnGPUSWARingUpdateResidentPlan){R,pos,native_swa_ring_fixtures[f].before,native_swa_ring_fixtures[f].window,B,D,kv,ring[q],scratch[q]};
                BnGPUSWARingUpdateInfo info;assert(!bn_gpu_swa_ring_update_info(R,pos,p[q].window_beg,p[q].window,B,&info)&&info.next_beg==native_swa_ring_fixtures[f].after&&(int)info.mode==native_swa_ring_fixtures[f].mode);
                BnGPUSWARingUpdateResidentPlan bad;
#define RING_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->swa_ring_update_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
                RING_BAD(n_tokens,0);RING_BAD(pos0,INT_MAX);RING_BAD(window_beg,p[q].window_beg+1);RING_BAD(window,0);RING_BAD(dim,INT_MAX);RING_BAD(ring_rows,B-1);RING_BAD(kv.count,n-1);RING_BAD(scratch,ring[q]);RING_BAD(scratch.count,(size_t)B*D-1);RING_BAD(ring.type,BN_GGUF_TENSOR_I32);
#undef RING_BAD
                uint64_t pre=f==first?folded_test_unwritten((size_t)B*D*2):native_swa_ring_fixtures[f-1].hash[1];
                native_router_guard_check(g,r[q],ring_root[q],(size_t)B*D,BN_GGUF_TENSOR_F16,pre);
            }
            BnGPUSWARingUpdateResidentPlan bad=p[0];bad.scratch=scratch[1];assert(g->swa_ring_update_f16_resident(g->ctx,r[0],&bad)==-1);
            for(int q=0;q<2;q++) {
                assert(!g->swa_ring_update_f16_resident(g->ctx,r[q],p+q));
                native_router_guard_check(g,r[q],root[q],n,BN_GGUF_TENSOR_F16,native_swa_ring_fixtures[f].hash[0]);
                native_router_guard_check(g,r[q],ring_root[q],(size_t)B*D,BN_GGUF_TENSOR_F16,native_swa_ring_fixtures[f].hash[1]);
                native_router_guard_check(g,r[q],scratch_root[q],(size_t)B*D,BN_GGUF_TENSOR_F16,native_swa_ring_fixtures[f].hash[2]);calls++;
            }
            free(x);
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}
    }
    printf("CUDA source sliding-window ring PASSED: %zu calls\n",calls);
}
