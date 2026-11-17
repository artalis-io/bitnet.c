/* Independent GGML CUDA df18bee11 Q5_K fused gate/up oracle.
 * Build against pinned GGML libraries only; no bitnet runtime linked. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "cuda_qk8_fixture.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    const int columns[]={256,512,2560}, rows_set[]={37,640};
    ggml_backend_t backend=ggml_backend_cuda_init(0);assert(backend);
    for(int c=0;c<3;c++)for(int r=0;r<2;r++)for(int seed=0;seed<=23;seed+=23) {
        int cols=columns[c],rows=rows_set[r];size_t bytes=(size_t)cols*rows/256*176;
        void *gate=malloc(bytes),*up=malloc(bytes);float *x=malloc((size_t)cols*4),*out=malloc((size_t)rows*4);
        assert(gate&&up&&x&&out);qk8_fixture_gate(gate,(size_t)cols*rows,1,seed);qk8_fixture_gate(up,(size_t)cols*rows,1,seed+19);
        for(int i=0;i<cols;i++)x[i]=(float)((i*37+11+seed)%251-125)/259.0f;
        struct ggml_init_params ip={16u*1024u*1024u,NULL,true};struct ggml_context *ctx=ggml_init(ip);assert(ctx);
        struct ggml_tensor *g=ggml_new_tensor_3d(ctx,GGML_TYPE_Q5_K,cols,rows,1),*u=ggml_new_tensor_3d(ctx,GGML_TYPE_Q5_K,cols,rows,1),*in=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,cols,1,1),*ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,1,1);
        struct ggml_tensor *y=ggml_swiglu_split(ctx,ggml_mul_mat_id(ctx,g,in,ids),ggml_mul_mat_id(ctx,u,in,ids));
        struct ggml_cgraph *graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);
        ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);int zero=0;
        ggml_backend_tensor_set(g,gate,0,bytes);ggml_backend_tensor_set(u,up,0,bytes);ggml_backend_tensor_set(in,x,0,(size_t)cols*4);ggml_backend_tensor_set(ids,&zero,0,4);
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(y,out,0,(size_t)rows*4);
        uint64_t hash=UINT64_C(14695981039346656037);for(int i=0;i<rows;i++){uint32_t bits;memcpy(&bits,out+i,4);hash=(hash^bits)*UINT64_C(1099511628211);}
        printf("{%d,%d,%d,UINT64_C(0x%016llx)},\n",rows,cols,seed,(unsigned long long)hash);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);free(gate);free(up);free(x);free(out);
    }
    ggml_backend_free(backend);return 0;
}
