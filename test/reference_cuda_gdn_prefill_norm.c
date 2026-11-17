/* Independent nonzero-Q/K/V CUDA GDN state fixture for fused prefill.
 * llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165; CUDA SHA-256
 * e3330904452e38a9dedc307dedf83e182322a27d6b10dd1df457e02326082840.
 * Initial convolution history supplies the first row; zero projected inputs
 * and taps {1,0} make later convolution rows zero. Gate=0 and beta=0.5.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    const int counts[] = {2, 9};
    const float epsilons[] = {1e-6f, 1e-3f};
    enum { hd=128, width=3*hd, ns=hd*hd };
    ggml_backend_t backend=ggml_backend_cuda_init(0);assert(backend);
    for (unsigned e=0;e<2;e++) for(unsigned c=0;c<2;c++) {
        int nt=counts[c];
        struct ggml_init_params params={8*1024*1024,NULL,true};
        struct ggml_context*ctx=ggml_init(params);assert(ctx);
        struct ggml_tensor*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,nt);
        struct ggml_tensor*s=ggml_silu(ctx,x);
        struct ggml_tensor*q=ggml_view_4d(ctx,s,hd,1,nt,1,hd*4,width*4,width*nt*4,0);
        struct ggml_tensor*k=ggml_view_4d(ctx,s,hd,1,nt,1,hd*4,width*4,width*nt*4,hd*4);
        struct ggml_tensor*v=ggml_cont(ctx,ggml_view_4d(ctx,s,hd,1,nt,1,hd*4,width*4,width*nt*4,2*hd*4));
        q=ggml_scale(ctx,ggml_rms_norm(ctx,q,epsilons[e]/hd),1.0f/sqrtf((float)hd));
        k=ggml_scale(ctx,ggml_rms_norm(ctx,k,epsilons[e]/hd),1.0f/sqrtf((float)hd));
        struct ggml_tensor*g=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,1,nt,1);
        struct ggml_tensor*b=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,1,nt,1);
        struct ggml_tensor*state=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,hd,hd,1,1);
        struct ggml_tensor*out=ggml_gated_delta_net(ctx,q,k,v,g,b,state,1);
        struct ggml_cgraph*graph=ggml_new_graph_custom(ctx,64,false);
        ggml_build_forward_expand(graph,out);
        ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);
        float*raw=calloc((size_t)width*nt,4);float*zeros=calloc(ns,4);float beta[9];float result[ns];assert(raw&&zeros);
        for(int i=0;i<width;i++)raw[i]=sinf((float)(i+1)*0.13f)*(i<2*hd?0.001f:0.125f);
        for(int i=0;i<nt;i++)beta[i]=0.5f;
        ggml_backend_tensor_set(x,raw,0,(size_t)width*nt*4);
        ggml_backend_tensor_set(g,zeros,0,(size_t)nt*4);
        ggml_backend_tensor_set(b,beta,0,(size_t)nt*4);
        ggml_backend_tensor_set(state,zeros,0,ns*4);
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out,result,(size_t)nt*hd*4,ns*4);
        uint64_t hash=UINT64_C(14695981039346656037);
        for(int i=0;i<ns;i++){uint32_t bits;memcpy(&bits,result+i,4);hash=(hash^bits)*UINT64_C(1099511628211);}
        printf("{%d, %.9g, UINT64_C(0x%016llx)},\n",nt,epsilons[e],(unsigned long long)hash);
        free(zeros);free(raw);ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);return 0;
}
