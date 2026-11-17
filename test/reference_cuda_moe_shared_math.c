/* Independent pinned GGML CUDA df18bee11 shared-expert sigmoid/add and SiLU.
 * Link only reference GGML libraries; deterministic rational inputs. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    const int dims[]={32,256,640,2560,8192},rows_set[]={37,2560},act_rows[]={37,640,2560};
    ggml_backend_t backend=ggml_backend_cuda_init(0);assert(backend);
    for(int mode=0;mode<2;mode++)for(int di=0;di<(mode?1:5);di++)
    for(int ri=0;ri<(mode?3:2);ri++)for(int seed=0;seed<=23;seed+=23)
    for(int reset=0;reset<(mode?1:2);reset++)for(int complement=0;complement<(mode?1:2);complement++) {
        int rows=mode?act_rows[ri]:rows_set[ri],dim=mode?rows:dims[di];
        float*x=malloc((size_t)dim*4),*w=malloc((size_t)dim*4),*v=malloc((size_t)rows*4),*prior=malloc((size_t)rows*4),*out=malloc((size_t)rows*4);assert(x&&w&&v&&prior&&out);
        for(int d=0;d<dim;d++){x[d]=(float)((d*37+seed)%251-125)/(mode?33.0f:259.0f);w[d]=(float)((d*19+seed*3)%257-128)/1319.0f;}
        for(int d=0;d<rows;d++){v[d]=(float)((d*13+seed)%257-128)/67.0f;prior[d]=(float)((d*11+seed)%239-119)/157.0f;}
        struct ggml_init_params ip={4u*1024u*1024u,NULL,true};struct ggml_context*c=ggml_init(ip);assert(c);
        struct ggml_tensor*tx=ggml_new_tensor_2d(c,GGML_TYPE_F32,dim,1),*tw=NULL,*tv=ggml_new_tensor_2d(c,GGML_TYPE_F32,rows,1),*tp=NULL,*one=NULL,*y;
        if(mode)y=ggml_swiglu_split(c,tx,tv);
        else {
            tw=ggml_new_tensor_2d(c,GGML_TYPE_F32,dim,1);struct ggml_tensor*g=ggml_sigmoid(c,ggml_mul_mat(c,tw,tx));
            if(complement){one=ggml_new_tensor_1d(c,GGML_TYPE_F32,1);g=ggml_sub(c,one,g);}
            y=ggml_mul(c,tv,g);if(!reset){tp=ggml_new_tensor_2d(c,GGML_TYPE_F32,rows,1);y=ggml_add(c,tp,y);}
        }
        struct ggml_cgraph*graph=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(graph,y);
        ggml_backend_buffer_t b=ggml_backend_alloc_ctx_tensors(c,backend);assert(b);ggml_backend_tensor_set(tx,x,0,(size_t)dim*4);ggml_backend_tensor_set(tv,v,0,(size_t)rows*4);
        if(tw)ggml_backend_tensor_set(tw,w,0,(size_t)dim*4);
        if(tp)ggml_backend_tensor_set(tp,prior,0,(size_t)rows*4);
        if(one){float f=1;ggml_backend_tensor_set(one,&f,0,4);}
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(y,out,0,(size_t)rows*4);
        uint64_t hash=UINT64_C(14695981039346656037);for(int d=0;d<rows;d++){uint32_t bits;memcpy(&bits,out+d,4);hash=(hash^bits)*UINT64_C(1099511628211);}
        printf("{%d,%d,%d,%d,%d,%d,UINT64_C(0x%016llx)},\n",mode,dim,rows,seed,reset,complement,(unsigned long long)hash);
        ggml_backend_buffer_free(b);ggml_free(c);free(x);free(w);free(v);free(prior);free(out);
    }
    ggml_backend_free(backend);return 0;
}
