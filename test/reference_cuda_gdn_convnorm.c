/* Independent CUDA SiLU + strided GDN RMS normalization graph.
 * llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165; CUDA library SHA-256
 * e3330904452e38a9dedc307dedf83e182322a27d6b10dd1df457e02326082840.
 * Identity last convolution tap makes its output equal to this SiLU input.
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
    const int dims[] = {7, 128, 1024};
    const float epsilons[] = {1e-6f, 1e-5f};
    ggml_backend_t backend = ggml_backend_cuda_init(0); assert(backend);
    for (unsigned e = 0; e < 2; e++) for (unsigned c = 0; c < 3; c++) {
        int hd = dims[c], heads = 2, nt = 3, width = 5*hd, count = nt*width;
        float *input = malloc((size_t)count*4), *output = malloc((size_t)count*4);
        float *qout = malloc((size_t)nt*heads*hd*4), *kout = malloc((size_t)nt*heads*hd*4);
        assert(input && output && qout && kout);
        for (int t = 0; t < nt; t++) for (int i = 0; i < width; i++)
            input[t*width+i] = sinf((float)(t*width+i+1)*0.13f) *
                (t == 0 ? 1e-8f : t == 1 ? 1.0f : 1e4f);
        struct ggml_init_params params = {4*1024*1024, NULL, true};
        struct ggml_context *ctx = ggml_init(params); assert(ctx);
        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, nt);
        struct ggml_tensor *s = ggml_silu(ctx, x);
        struct ggml_tensor *q = ggml_view_3d(ctx, s, hd, heads, nt, hd*4, width*4, 0);
        struct ggml_tensor *k = ggml_view_3d(ctx, s, hd, heads, nt, hd*4, width*4, heads*hd*4);
        q = ggml_scale(ctx, ggml_rms_norm(ctx, q, epsilons[e]/hd), 1.0f/sqrtf((float)hd));
        k = ggml_scale(ctx, ggml_rms_norm(ctx, k, epsilons[e]/hd), 1.0f/sqrtf((float)hd));
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, q); ggml_build_forward_expand(graph, k);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(buffer);
        ggml_backend_tensor_set(x, input, 0, (size_t)count*4);
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(s, output, 0, (size_t)count*4);
        ggml_backend_tensor_get(q, qout, 0, (size_t)nt*heads*hd*4);
        ggml_backend_tensor_get(k, kout, 0, (size_t)nt*heads*hd*4);
        for (int t = 0; t < nt; t++) {
            memcpy(output+t*width, qout+t*heads*hd, (size_t)heads*hd*4);
            memcpy(output+t*width+heads*hd, kout+t*heads*hd, (size_t)heads*hd*4);
        }
        uint64_t hash = UINT64_C(14695981039346656037);
        for (int i = 0; i < count; i++) {
            uint32_t bits; memcpy(&bits, output+i, sizeof(bits));
            hash = (hash ^ bits)*UINT64_C(1099511628211);
        }
        printf("{%d, %.9g, UINT64_C(0x%016llx)},\n",hd,epsilons[e],(unsigned long long)hash);
        free(kout);free(qout);free(output);free(input);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);return 0;
}
