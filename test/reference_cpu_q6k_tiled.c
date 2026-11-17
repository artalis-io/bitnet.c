/* Independent CPU tiled Q6_K oracle. Links only GGML, never bitnet.c.
 * Build against the pinned llama.cpp headers and libggml{,-base,-cpu}.so.0.
 * Run with that build's LD_LIBRARY_PATH. AVX512 VNNI arithmetic contract.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    enum { rows = 137, nb = 3, cols = nb * 256, tokens = 33 };
    uint8_t weights[rows * nb * 210];
    float input[tokens * cols], output[tokens * rows];
    assert(ggml_type_size(GGML_TYPE_Q6_K) == 210);
    for (int b = 0; b < rows * nb; b++) {
        uint8_t *p = weights + b * 210;
        ggml_fp16_t d = ggml_fp32_to_fp16(0.002f * (b % 19 + 1));
        for (int i=0;i<192;i++) p[i]=(uint8_t)(b*31+i*37);
        for (int i=0;i<16;i++) p[192+i]=(uint8_t)(b*17+i*23);
        memcpy(p+208,&d,2);
    }
    for (int i = 0; i < tokens * cols; i++) input[i] = sinf((float)i * 0.13f);
    ggml_backend_t backend = ggml_backend_cpu_init(); assert(backend);
    ggml_backend_cpu_set_n_threads(backend, 4);
    const int counts[] = {8, 9, 17, 33};
    for (unsigned c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
        struct ggml_init_params params = {4*1024*1024, NULL, true};
        struct ggml_context *ctx = ggml_init(params); assert(ctx);
        struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, cols, rows);
        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, counts[c]);
        struct ggml_tensor *y = ggml_mul_mat(ctx, w, x);
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(buffer);
        ggml_backend_tensor_set(w, weights, 0, sizeof(weights));
        ggml_backend_tensor_set(x, input, 0, (size_t)counts[c]*cols*sizeof(float));
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y, output, 0, (size_t)counts[c]*rows*sizeof(float));
        uint64_t hash = UINT64_C(14695981039346656037);
        for (int i = 0; i < counts[c]*rows; i++) {
            uint32_t bits; memcpy(&bits, output+i, 4);
            hash = (hash ^ bits) * UINT64_C(1099511628211);
        }
        printf("{%d, UINT64_C(0x%016llx)},\n", counts[c], (unsigned long long)hash);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
