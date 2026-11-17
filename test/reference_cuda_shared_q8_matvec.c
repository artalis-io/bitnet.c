/* Independent GGML df18bee11 CUDA Q8_0 single-column projection oracle.
 * Link pinned GGML libraries only. Inputs avoid host transcendental math. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    const int shapes[][2]={{37,64},{256,512},{2560,640},{2560,2560},{640,2560},
        {320,10240},{10240,320},{512,2560},{12288,2560},{2560,6144}};
    ggml_backend_t backend=ggml_backend_cuda_init(0);assert(backend);
    for(size_t shape=0;shape<sizeof(shapes)/sizeof(shapes[0]);shape++)for(int seed=0;seed<=23;seed+=23) {
        int rows=shapes[shape][0],cols=shapes[shape][1];
        size_t nb=(size_t)rows*cols/32,bytes=nb*34;
        uint8_t *weight=malloc(bytes);float *x=malloc((size_t)cols*sizeof(float));
        float *out=malloc((size_t)rows*sizeof(float));assert(weight && x && out);
        for(size_t b=0;b<nb;b++) {
            uint8_t *w=weight+b*34;
            uint16_t d=(uint16_t)(0x1800+((b+seed)%17)*37);
            memcpy(w,&d,2);
            for(int j=0;j<32;j++)w[2+j]=(uint8_t)((int)((b*19+j*23+seed)%255)-127);
        }
        for(int i=0;i<cols;i++)x[i]=(float)((i*37+11+seed)%251-125)/259.0f;
        struct ggml_init_params params={16u*1024u*1024u,NULL,true};
        struct ggml_context *ctx=ggml_init(params);assert(ctx);
        struct ggml_tensor *w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,cols,rows);
        struct ggml_tensor *in=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,1);
        struct ggml_tensor *result=ggml_mul_mat(ctx,w,in);
        struct ggml_cgraph *graph=ggml_new_graph_custom(ctx,16,false);
        ggml_build_forward_expand(graph,result);
        ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);
        ggml_backend_tensor_set(w,weight,0,bytes);
        ggml_backend_tensor_set(in,x,0,(size_t)cols*sizeof(float));
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(result,out,0,(size_t)rows*sizeof(float));
        uint64_t hash=UINT64_C(14695981039346656037);
        for(int i=0;i<rows;i++){uint32_t bits;memcpy(&bits,out+i,4);hash=(hash^bits)*UINT64_C(1099511628211);}
        uint64_t biased=UINT64_C(14695981039346656037);
        for(int i=0;i<rows;i++) {
            float value=out[i]+(float)((i*7+seed)%17-8)*0.015625f;
            uint32_t bits;memcpy(&bits,&value,4);
            biased=(biased^bits)*UINT64_C(1099511628211);
        }
        printf("{%d,%d,%d,UINT64_C(0x%016llx),UINT64_C(0x%016llx)},\n",rows,cols,seed,
            (unsigned long long)hash,(unsigned long long)biased);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);free(weight);free(x);free(out);
    }
    ggml_backend_free(backend);return 0;
}
