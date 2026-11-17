/* Independent non-flash F32 attention oracle: pinned GGML df18bee11 CUDA.
 * No runtime headers or kernels. Inputs are exactly specified rational values. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    const int lengths[]={1,3,12,32,33,127,256,257,513,1047,2048};
    ggml_backend_t b=ggml_backend_cuda_init(0);assert(b);
    for(int group=0;group<2;group++)for(size_t z=0;z<sizeof(lengths)/sizeof(lengths[0]);z++)
    for(int seed=0;seed<=23;seed+=23)for(int window=0;window<=3;window+=3) {
        const int nk=group?4:2,gqa=group?6:12,nh=nk*gqa,n=lengths[z],pad=(n+255)/256*256;
        float *q=malloc((size_t)nh*256*4),*k=calloc((size_t)pad*nk*256,4),*v=calloc((size_t)pad*nk*256,4),*mask=malloc((size_t)pad*4),*out=malloc((size_t)nh*256*4);assert(q&&k&&v&&mask&&out);
        for(int i=0;i<nh*256;i++)q[i]=(float)((i*17+seed*13)%251-125)/61.0f;
        for(int t=0;t<n;t++)for(int h=0;h<nk;h++)for(int j=0;j<256;j++) {
            size_t i=((size_t)t*nk+h)*256+j;
            k[i]=(float)((int)((i*11+seed*7)%257)-128)/59.0f;
            v[i]=(float)((int)((i*19+seed*3)%263)-131)/127.0f;
        }
        for(int i=0;i<pad;i++)mask[i]=i<n&&(!window||i>=n-window)?0.0f:-__builtin_inff();
        struct ggml_init_params params={1024*1024,NULL,true};struct ggml_context*c=ggml_init(params);assert(c);
        struct ggml_tensor*qt=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,nh,1),*kt=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,nk,pad),*vt=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,nk,pad),*mt=ggml_new_tensor_2d(c,GGML_TYPE_F32,pad,1);
        struct ggml_tensor*qp=ggml_permute(c,qt,0,2,1,3),*kp=ggml_permute(c,kt,0,2,1,3),*vp=ggml_permute(c,vt,0,2,1,3);
        struct ggml_tensor*sc=ggml_mul_mat(c,kp,qp);assert(ggml_prec_set_acc(sc,GGML_PREC_F32));
        struct ggml_tensor*pr=ggml_soft_max_ext(c,sc,mt,0.0625f,0.0f);
        struct ggml_tensor*result=ggml_mul_mat(c,ggml_cont(c,ggml_transpose(c,vp)),pr);result=ggml_cont(c,ggml_permute(c,result,0,2,1,3));
        struct ggml_cgraph*g=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(g,result);ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,b);assert(buf);
        ggml_backend_tensor_set(qt,q,0,(size_t)nh*256*4);ggml_backend_tensor_set(kt,k,0,(size_t)pad*nk*256*4);ggml_backend_tensor_set(vt,v,0,(size_t)pad*nk*256*4);ggml_backend_tensor_set(mt,mask,0,(size_t)pad*4);
        assert(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(result,out,0,(size_t)nh*256*4);
        uint64_t hash=UINT64_C(14695981039346656037);for(int i=0;i<nh*256;i++){union{float f;uint32_t u;}x={out[i]};hash=(hash^x.u)*UINT64_C(1099511628211);}
        printf("{%d,%d,%d,%d,%d,UINT64_C(0x%016llx)},\n",nk,gqa,n,seed,window,(unsigned long long)hash);
        ggml_backend_buffer_free(buf);ggml_free(c);free(q);free(k);free(v);free(mask);free(out);
    }
    ggml_backend_free(b);return 0;
}
