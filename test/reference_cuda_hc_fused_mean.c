/* Independent fused HC pre-gate oracle.
 * Reference: llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165,
 * build/bin GGML CUDA (SM120a, -use_fast_math). Link only those libraries:
 * cc -O2 -std=c11 -Wall -Wextra -Werror -I$REF/ggml/include \
 *   test/reference_cuda_hc_fused_mean.c -L$REF/build/bin \
 *   -lggml -lggml-base -lggml-cuda -Wl,-rpath,$REF/build/bin -o hc_oracle
 * CUDA_VISIBLE_DEVICES=0 ./hc_oracle
 * Values are deterministic rational inputs, not host transcendental results.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    const int dims[] = {64, 320, 2560}, counts[] = {1, 11};
    const float magnitudes[] = {0.01f, 1.0f, 4.0f};
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    for (int d = 0; d < 3; d++) for (int streams = 2; streams <= 4; streams += 2)
    for (int n = 0; n < 2; n++) for (int seed = 0; seed <= 23; seed += 23)
    for (int m = 0; m < 3; m++) {
        int dim = dims[d], nt = counts[n];
        size_t count = (size_t)dim * streams * nt, outputs = (size_t)dim * nt;
        float *x = malloc(count * sizeof(float)), *gate = malloc(count * sizeof(float));
        float *out = malloc(outputs * sizeof(float));
        assert(x && gate && out);
        for (size_t i = 0; i < count; i++) {
            x[i] = (float)((int)((i * 37 + seed * 11) % 251) - 125) / 259.0f * magnitudes[m];
            gate[i] = (float)((int)((i * 19 + seed * 13) % 257) - 128) / 33.0f * magnitudes[m];
        }
        struct ggml_init_params params = {16u * 1024u * 1024u, NULL, true};
        struct ggml_context *ctx = ggml_init(params);
        assert(ctx);
        struct ggml_tensor *tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, streams, nt);
        struct ggml_tensor *tg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, streams, nt);
        struct ggml_tensor *ty = ggml_dsv4_hc_pre_gated(ctx, tx, tg, 1.0f / streams);
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, ty);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert(buffer);
        ggml_backend_tensor_set(tx, x, 0, count * sizeof(float));
        ggml_backend_tensor_set(tg, gate, 0, count * sizeof(float));
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(ty, out, 0, outputs * sizeof(float));
        uint64_t hash = UINT64_C(14695981039346656037);
        for (size_t i = 0; i < outputs; i++) {
            uint32_t bits; memcpy(&bits, out + i, sizeof(bits));
            hash = (hash ^ bits) * UINT64_C(1099511628211);
        }
        printf("{%d,%d,%d,%d,%.9ef,UINT64_C(0x%016llx)},\n",
            dim, streams, nt, seed, (double)magnitudes[m], (unsigned long long)hash);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        free(x); free(gate); free(out);
    }
    ggml_backend_free(backend);
    return 0;
}
