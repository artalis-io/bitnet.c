/* Independent pinned GGML CUDA df18bee11 unscaled MoE weighted reduction.
 * Match the checkpoint graph: MUL, all expert VIEWs, then ordered ADDs. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    const int dims[]={37,2560},counts[]={1,2,10,15,16},tokens[]={1,3};
    ggml_backend_t backend=ggml_backend_cuda_init(0);assert(backend);
    for(int di=0;di<2;di++)for(int ki=0;ki<5;ki++)for(int ti=0;ti<2;ti++)for(int seed=0;seed<=23;seed+=23) {
        int dim=dims[di],k=counts[ki],nt=tokens[ti];size_t n=(size_t)dim*k*nt;
        float*x=malloc(n*4),*w=malloc((size_t)k*nt*4),*out=malloc((size_t)dim*nt*4);assert(x&&w&&out);
        for(int t=0;t<nt;t++)for(int s=0;s<k;s++){
            w[t*k+s]=(float)((t*3+s*5+seed)%23+1)/29.0f;
            for(int d=0;d<dim;d++)x[((size_t)t*k+s)*dim+d]=(float)((d*37+s*19+t*11+seed)%251-125)/259.0f;
        }
        struct ggml_init_params ip={4u*1024u*1024u,NULL,true};struct ggml_context*c=ggml_init(ip);assert(c);
        struct ggml_tensor*tx=ggml_new_tensor_3d(c,GGML_TYPE_F32,dim,k,nt),*tw=ggml_new_tensor_3d(c,GGML_TYPE_F32,1,k,nt);
        struct ggml_tensor*weighted=ggml_mul(c,tx,tw),*views[16]={0};struct ggml_cgraph*g=ggml_new_graph_custom(c,128,false);ggml_build_forward_expand(g,weighted);
        for(int s=0;s<k;s++){views[s]=ggml_view_2d(c,weighted,dim,nt,weighted->nb[2],s*weighted->nb[1]);ggml_build_forward_expand(g,views[s]);}
        struct ggml_tensor*y=views[0];for(int s=1;s<k;s++){y=ggml_add(c,y,views[s]);ggml_build_forward_expand(g,y);}y=ggml_cont(c,y);ggml_build_forward_expand(g,y);
        ggml_backend_buffer_t b=ggml_backend_alloc_ctx_tensors(c,backend);assert(b);ggml_backend_tensor_set(tx,x,0,n*4);ggml_backend_tensor_set(tw,w,0,(size_t)k*nt*4);
        assert(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(y,out,0,(size_t)dim*nt*4);
        uint64_t hash=UINT64_C(14695981039346656037);for(int d=0;d<dim*nt;d++){uint32_t bits;memcpy(&bits,out+d,4);hash=(hash^bits)*UINT64_C(1099511628211);}
        printf("{%d,%d,%d,%d,UINT64_C(0x%016llx)},\n",dim,k,nt,seed,(unsigned long long)hash);
        ggml_backend_buffer_free(b);ggml_free(c);free(x);free(w);free(out);
    }
    ggml_backend_free(backend);return 0;
}
