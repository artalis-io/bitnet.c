/* Independent standalone CUDA RoPE oracle for the borrowed Q/K prefill view.
 * Link only the pinned GGML libraries, never bitnet.c. Normalization/projection
 * inputs are exact: +/-2, identity weights, zero epsilon give +/-1 before the
 * per-head scale. A separate graph applies RoPE so normalization cannot fuse
 * into it. Base one gives exactly representable unit frequencies. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { hd = 32, nt = 33 };
    float keys[nt * hd], output[nt * hd];
    int32_t positions[nt];
    for (int t = 0; t < nt; t++) {
        positions[t] = t;
        for (int i = 0; i < hd; i++) {
            float gamma = (float)(i + 3) / 17.0f;
            keys[t * hd + i] = ((i + t) & 1) ? -gamma : gamma;
        }
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    struct ggml_init_params params = {1024 * 1024, NULL, true};
    struct ggml_context *ctx = ggml_init(params);
    assert(ctx);
    struct ggml_tensor *k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, nt);
    struct ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
    struct ggml_tensor *result = ggml_rope_ext(ctx, k, pos, NULL, hd,
        GGML_ROPE_TYPE_NEOX, 64, 1.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    ggml_backend_tensor_set(k, keys, 0, sizeof(keys));
    ggml_backend_tensor_set(pos, positions, 0, sizeof(positions));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(result, output, 0, sizeof(output));
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < sizeof(output) / sizeof(output[0]); i++) {
        uint32_t bits;
        memcpy(&bits, output + i, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    printf("standalone_cuda_rope_hash=0x%016llx\n", (unsigned long long)hash);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
