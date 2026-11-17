#include "cuda_f32_add_reference.h"
static void run_f32_add_reference(BnGPUBackend *g) {
    const size_t sizes[]={1,7,31,32,33,255,256,257,8193};
    const size_t fixtures=sizeof(f32_add_source)/sizeof(f32_add_source[0]);
    size_t compared=0;
    assert(g->add_f32_resident);
    for(size_t c=0;c<sizeof(sizes)/sizeof(sizes[0]);c++) {
        size_t n=sizes[c];uint32_t *host[3];
        for(int j=0;j<3;j++){host[j]=malloc((n+16)*4);assert(host[j]);for(size_t i=0;i<n+16;i++)host[j][i]=UINT32_C(0xa5a5a5a5);}
        for(size_t i=0;i<n;i++){host[0][8+i]=f32_add_source[i%fixtures][0];host[1][8+i]=f32_add_source[i%fixtures][1];}
        void *requests[2];BnGPUResidentView stores[2][3],views[2][3];BnGPUAddF32ResidentPlan plans[2];
        for(int q=0;q<2;q++) {
            requests[q]=g->resident_request_create(g->ctx);assert(requests[q]);
            for(int j=0;j<3;j++) {
                stores[q][j]=cuda_test_resident_alloc(g,requests[q],BN_GGUF_TENSOR_F32,n+16,host[j]);
                views[q][j]=bn_gpu_resident_view_slice(stores[q][j],8,n);
            }
            plans[q]=(BnGPUAddF32ResidentPlan){n,views[q][0],views[q][1],views[q][2]};
            BnGPUAddF32ResidentPlan bad;
#define F32_ADD_BAD(F,V) do{bad=plans[q];bad.F=(V);assert(g->add_f32_resident(g->ctx,requests[q],&bad)==-1);}while(0)
            F32_ADD_BAD(count,0);F32_ADD_BAD(count,SIZE_MAX);F32_ADD_BAD(count,(size_t)INT_MAX+1);
            F32_ADD_BAD(x.type,BN_GGUF_TENSOR_F16);F32_ADD_BAD(y.count,n+1);
            F32_ADD_BAD(out.count,n+1);F32_ADD_BAD(out,views[q][0]);F32_ADD_BAD(out,views[q][1]);
            F32_ADD_BAD(x.offset,SIZE_MAX);F32_ADD_BAD(y.slot,UINT32_MAX);
            if(n>1){F32_ADD_BAD(out,bn_gpu_resident_view_slice(stores[q][0],9,n));}
#undef F32_ADD_BAD
            assert(g->add_f32_resident(NULL,requests[q],&plans[q])==-1);
            assert(g->add_f32_resident(g->ctx,NULL,&plans[q])==-1);
            assert(g->add_f32_resident(g->ctx,requests[q],NULL)==-1);
        }
        BnGPUAddF32ResidentPlan bad=plans[0];bad.y=views[1][1];
        assert(g->add_f32_resident(g->ctx,requests[0],&bad)==-1);
        bad=plans[0];bad.out=views[1][2];assert(g->add_f32_resident(g->ctx,requests[0],&bad)==-1);
        uint32_t *got=malloc((n+16)*4);assert(got);
        for(int q=0;q<2;q++)for(int j=0;j<3;j++) {
            assert(!g->resident_read(requests[q],stores[q][j],got));assert(!memcmp(got,host[j],(n+16)*4));
        }
        for(int repeat=0;repeat<2;repeat++)for(int q=0;q<2;q++)assert(!g->add_f32_resident(g->ctx,requests[q],&plans[q]));
        for(int alias=0;alias<2;alias++) {
            if(alias)for(int q=0;q<2;q++){plans[q].y=plans[q].x;assert(!g->add_f32_resident(g->ctx,requests[q],&plans[q]));}
            for(int q=0;q<2;q++) {
                assert(!g->resident_read(requests[q],stores[q][2],got));
                for(size_t i=0;i<8;i++)assert(got[i]==UINT32_C(0xa5a5a5a5)&&got[n+8+i]==UINT32_C(0xa5a5a5a5));
                for(size_t i=0;i<n;i++)assert(got[8+i]==f32_add_source[i%fixtures][alias?3:2]);
                compared+=n;
            }
        }
        for(int q=0;q<2;q++) {
            for(int j=0;j<2;j++){assert(!g->resident_read(requests[q],stores[q][j],got));assert(!memcmp(got,host[j],(n+16)*4));}
            g->resident_request_destroy(requests[q]);
        }
        free(got);for(int j=0;j<3;j++)free(host[j]);
    }
    printf("CUDA F32 add source edges/guards/rejection/two-request isolation PASSED: %zu values\n",compared);
}
