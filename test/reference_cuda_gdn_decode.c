/* Independent decimal-input GDN decode oracle; link only GGML CUDA libraries
 * from llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165, SM120a.
 * Deliberately nonzero Q/K/V at every timestep exercise reduction order.
 * State layout is [key, value, head], matching GGML's transposed storage.
 * Each hash covers all token outputs followed by the complete final state. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const int counts[] = {1, 4, 9};
    const int key_heads[] = {1, 2, 16}, value_heads[] = {1, 6, 48};
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    for (int shape = 0; shape < 3; shape++) for (int c = 0; c < 3; c++) {
        int nt = counts[c], nk = key_heads[shape], nv = value_heads[shape];
        size_t nq = (size_t)nt * nk * 128, ny = (size_t)nt * nv * 128;
        size_t ns = (size_t)nv * 128 * 128, na = (size_t)nt * nv;
        float *q = malloc(nq * 4), *k = malloc(nq * 4), *v = malloc(ny * 4);
        float *g = malloc(na * 4), *b = malloc(na * 4), *state = malloc(ns * 4);
        float *result = malloc((ny + ns) * 4);
        assert(q && k && v && g && b && state && result);
        for (size_t i = 0; i < nq; i++) {
            q[i] = ((int)((i * 37 + 3) % 31) - 15) / 259.0f;
            k[i] = ((int)((i * 19 + 7) % 29) - 14) / 257.0f;
        }
        for (size_t i = 0; i < ny; i++)
            v[i] = ((int)((i * 13 + 11) % 23) - 11) / 131.0f;
        for (size_t i = 0; i < ns; i++)
            state[i] = ((int)((i * 29 + 5) % 101) - 50) / 1021.0f;
        for (size_t i = 0; i < na; i++) { g[i] = -0.6931471805599453f; b[i] = 0.5f; }
        struct ggml_init_params params = {16u * 1024u * 1024u, NULL, true};
        struct ggml_context *ctx = ggml_init(params);
        assert(ctx);
        struct ggml_tensor *qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, nk, nt, 1);
        struct ggml_tensor *kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, nk, nt, 1);
        struct ggml_tensor *vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, nv, nt, 1);
        struct ggml_tensor *gt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, nv, nt, 1);
        struct ggml_tensor *bt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, nv, nt, 1);
        struct ggml_tensor *st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, nv, 1);
        struct ggml_tensor *out = ggml_gated_delta_net(ctx, qt, kt, vt, gt, bt, st, 1);
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert(buffer);
        ggml_backend_tensor_set(qt, q, 0, nq * 4); ggml_backend_tensor_set(kt, k, 0, nq * 4);
        ggml_backend_tensor_set(vt, v, 0, ny * 4); ggml_backend_tensor_set(gt, g, 0, na * 4);
        ggml_backend_tensor_set(bt, b, 0, na * 4); ggml_backend_tensor_set(st, state, 0, ns * 4);
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, result, 0, (ny + ns) * 4);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (size_t i = 0; i < ny + ns; i++) {
            uint32_t bits; memcpy(&bits, result + i, 4);
            hash = (hash ^ bits) * UINT64_C(1099511628211);
        }
        printf("{%d, %d, %d, UINT64_C(0x%016llx)},\n", nt, nk, nv, (unsigned long long)hash);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        free(q); free(k); free(v); free(g); free(b); free(state); free(result);
    }
    ggml_backend_free(backend);
    return 0;
}
