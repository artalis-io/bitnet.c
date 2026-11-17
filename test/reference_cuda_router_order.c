/* Independent pinned GGML df18bee11 CUDA router oracle. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void) {
    const int sizes[]={32,64,128,256,512},ks[]={1,2,10,16};
    ggml_backend_t backend=ggml_backend_cuda_init(0);assert(backend);
    for(size_t z=0;z<5;z++)for(size_t j=0;j<4;j++)for(int norm=0;norm<2;norm++)
    for(int seed=0;seed<=23;seed+=23)for(int mode=0;mode<3;mode++)for(int scaled=0;scaled<2;scaled++) {
        int experts=sizes[z],k=ks[j];float *data=calloc((size_t)experts*32,4);float input[32]={1.0f};assert(data);
        for(int e=0;e<experts;e++) {
            float value=(float)((e*37+seed*11)%257-128)/37.0f;
            if(mode==1)value=0.25f+(float)((e*19+seed)%97)*0x1p-26f;
            if(mode==2)value=e%32==0?2.0f-(float)(e/32)*0.015625f:(float)((e*13+seed)%101-50)/97.0f;
            data[(size_t)e*32]=value;
        }
        struct ggml_init_params params={1024*1024,NULL,true};struct ggml_context*c=ggml_init(params);assert(c);
        struct ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,32,1),*w=ggml_new_tensor_2d(c,GGML_TYPE_F32,32,experts);
        struct ggml_tensor*logits=ggml_mul_mat(c,w,x),*probs=ggml_soft_max(c,logits),*ids=ggml_argsort_top_k(c,probs,k);
        struct ggml_tensor*weights=ggml_get_rows(c,ggml_reshape_3d(c,probs,1,experts,1),ids);weights=ggml_reshape_2d(c,weights,k,1);
        if(norm){struct ggml_tensor*sum=ggml_clamp(c,ggml_sum_rows(c,weights),6.103515625e-5f,INFINITY);weights=ggml_div(c,weights,sum);}
        weights=ggml_reshape_3d(c,weights,1,k,1);
        if(scaled)weights=ggml_scale(c,weights,0.75f);
        struct ggml_cgraph*g=ggml_new_graph_custom(c,64,false);ggml_build_forward_expand(g,ggml_cont(c,weights));
        ggml_backend_buffer_t b=ggml_backend_alloc_ctx_tensors(c,backend);assert(b);ggml_backend_tensor_set(x,input,0,sizeof(input));ggml_backend_tensor_set(w,data,0,(size_t)experts*32*4);
        assert(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS);float out[16];int32_t out_ids[16];ggml_backend_tensor_get(weights,out,0,(size_t)k*4);ggml_backend_tensor_get(ids,out_ids,0,(size_t)k*4);
        uint64_t wh=UINT64_C(14695981039346656037),ih=wh;for(int i=0;i<k;i++){uint32_t bits;memcpy(&bits,out+i,4);wh=(wh^bits)*UINT64_C(1099511628211);ih=(ih^(uint32_t)out_ids[i])*UINT64_C(1099511628211);}
        printf("{%d,%d,%d,%d,%d,%d,UINT64_C(0x%016llx),UINT64_C(0x%016llx)},\n",experts,k,norm,seed,mode,scaled,(unsigned long long)wh,(unsigned long long)ih);
        ggml_backend_buffer_free(b);ggml_free(c);free(data);
    }
    ggml_backend_free(backend);return 0;
}
