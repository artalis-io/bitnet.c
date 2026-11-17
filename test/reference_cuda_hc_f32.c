/* Independent pinned GGML CUDA df18bee11 F32 MMVF oracle for HC injection
 * and recurrent coefficients. Uses the retained F32 matrix fixture inputs. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const int shapes[][2] = {{10240,4},{2560,48}};
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    for (size_t ci=0; ci<sizeof(shapes)/sizeof(shapes[0]); ci++) {
        int cols=shapes[ci][0], rows=shapes[ci][1];
        float *w=malloc((size_t)cols*rows*sizeof(float));
        float *x=malloc((size_t)cols*sizeof(float));
        float *out=malloc((size_t)rows*sizeof(float));
        assert(w && x && out);
        for (int r=0; r<rows; r++) for (int d=0; d<cols; d++)
            w[(size_t)r*cols+d]=(float)((r*13+d*17)%257-128)/257.0f;
        for (int d=0; d<cols; d++) x[d]=(float)((d*37)%251-125)/259.0f;
        struct ggml_init_params params={16u*1024u*1024u,NULL,true};
        struct ggml_context *ctx=ggml_init(params); assert(ctx);
        struct ggml_tensor *tw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,rows);
        struct ggml_tensor *tx=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,1);
        struct ggml_tensor *ty=ggml_mul_mat(ctx,tw,tx);
        struct ggml_cgraph *graph=ggml_new_graph_custom(ctx,16,false);
        ggml_build_forward_expand(graph,ty);
        ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);
        ggml_backend_tensor_set(tw,w,0,(size_t)cols*rows*sizeof(float));
        ggml_backend_tensor_set(tx,x,0,(size_t)cols*sizeof(float));
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(ty,out,0,(size_t)rows*sizeof(float));
        uint64_t hash=UINT64_C(14695981039346656037);
        for (int r=0; r<rows; r++) {uint32_t bits;memcpy(&bits,out+r,4);hash=(hash^bits)*UINT64_C(1099511628211);}
        printf("{%d,%d,1,UINT64_C(0x%016llx)},\n",cols,rows,(unsigned long long)hash);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);free(w);free(x);free(out);
    }
    ggml_backend_free(backend);return 0;
}
