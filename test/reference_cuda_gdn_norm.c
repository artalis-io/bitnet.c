/* Independent GGML CUDA GDN Q/K normalization oracle; never links bitnet.c.
 * llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165, CUDA device buffers.
 * Build with its ggml/include and versioned libggml{,-base,-cuda}.so libraries.
 * libggml-cuda SHA-256:
 * e3330904452e38a9dedc307dedf83e182322a27d6b10dd1df457e02326082840.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    const int dims[] = {1, 7, 16, 128, 256, 1024, 1536};
    const float epsilons[] = {1e-6f, 1e-5f};
    ggml_backend_t backend = ggml_backend_cuda_init(0); assert(backend);
    for (unsigned e = 0; e < 2; e++) for (unsigned c = 0; c < 7; c++) {
        int hd = dims[c];
        float input[4 * 1536], output[4 * 1536];
        const float scales[] = {0.0f, 1e-8f, 1.0f, 1e10f};
        for (int h = 0; h < 4; h++) for (int i = 0; i < hd; i++)
            input[h * hd + i] = sinf((float)(h * hd + i + 1) * 0.13f) * scales[h];
        struct ggml_init_params params = {4*1024*1024, NULL, true};
        struct ggml_context *ctx = ggml_init(params); assert(ctx);
        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, 4);
        struct ggml_tensor *y = ggml_scale(ctx,
            ggml_rms_norm(ctx, x, epsilons[e] / (float)hd), 1.0f / sqrtf((float)hd));
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(buffer);
        ggml_backend_tensor_set(x, input, 0, (size_t)4*hd*sizeof(float));
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y, output, 0, (size_t)4*hd*sizeof(float));
        uint64_t hash = UINT64_C(14695981039346656037);
        for (int i = 0; i < 4 * hd; i++) {
            uint32_t bits; memcpy(&bits, output + i, sizeof(bits));
            hash = (hash ^ bits) * UINT64_C(1099511628211);
        }
        printf("{%d, %.9g, UINT64_C(0x%016llx)},\n", hd, epsilons[e], (unsigned long long)hash);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
