/* Independent CPU tiled IQ-format oracle. Links only GGML, never bitnet.c.
 * Build against the pinned llama.cpp headers and libggml{,-base,-cpu}.so.0.
 * Run with that build's LD_LIBRARY_PATH. AVX512 VNNI arithmetic contract.
 * Reference: llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165,
 * libggml-cpu.so.0 SHA-256:
 * e28660ece5030ec85229b5e03e57d8f0da55866b63943623a651e1adbd02716f.
 * Odd row count exercises the final partial row panel. Counts 9/10/11
 * exercise every four-token tail. Uses ordinary CPU buffers, no repacking.
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
static void run(enum ggml_type type, size_t block_size) {
    enum { rows = 137, nb = 3, cols = nb * 256, tokens = 33 };
    uint8_t *weights = malloc(rows * nb * block_size); assert(weights);
    float input[tokens * cols], output[tokens * rows];
    assert(ggml_type_size(type) == block_size);
    printf("type=%s\n", ggml_type_name(type));
    for (int b = 0; b < rows * nb; b++) {
        uint8_t *p = weights + b * block_size;
        ggml_fp16_t d = ggml_fp32_to_fp16(0.002f * (b % 19 + 1));
        for (size_t i = 2; i < block_size; i++) p[i] = (uint8_t)(b * 31 + i * 37);
        memcpy(p, &d, 2);
    }
    for (int i = 0; i < tokens * cols; i++) input[i] = sinf((float)i * 0.13f);
    ggml_backend_t backend = ggml_backend_cpu_init(); assert(backend);
    ggml_backend_cpu_set_n_threads(backend, 4);
    const int counts[] = {8, 9, 10, 11, 17, 33};
    for (unsigned c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
        struct ggml_init_params params = {4*1024*1024, NULL, true};
        struct ggml_context *ctx = ggml_init(params); assert(ctx);
        struct ggml_tensor *w = ggml_new_tensor_2d(ctx, type, cols, rows);
        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, counts[c]);
        struct ggml_tensor *y = ggml_mul_mat(ctx, w, x);
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(buffer);
        ggml_backend_tensor_set(w, weights, 0, rows * nb * block_size);
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
    free(weights);
}
int main(void) {
    run(GGML_TYPE_IQ4_XS, 136);
    run(GGML_TYPE_IQ3_S, 110);
    run(GGML_TYPE_IQ3_XXS, 98);
    return 0;
}
