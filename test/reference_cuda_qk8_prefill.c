/* Independent prefill oracle for Q4_K or Q5_K gate/up + Q8_0 down.
 * llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165, GGML CUDA SM120a.
 * Link pinned GGML libraries only, plus -lm. No bitnet runtime is linked.
 * Preserves existing <=8-token MMVQ goldens; these cases cross into MMQ.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "cuda_mxfp4_fixture.h"
#include "cuda_qk8_fixture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t hash_words(const float *x, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, x + i, sizeof(bits));
        h = (h ^ bits) * UINT64_C(1099511628211);
    }
    return h;
}

int main(void) {
    const int dims[] = {256, 2560}, hidden_dims[] = {512, 640};
    const int token_counts[] = {1, 8, 9, 11, 33};
    const int experts = 16, k = 8;
    (void)mxfp4_fixture_weights;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    for (int five = 0; five < 2; five++)
    for (int shape = 0; shape < 2; shape++)
    for (int seed = 0; seed <= 23; seed += 23)
    for (int c = 0; c < 5; c++) {
        int dim = dims[shape], hidden = hidden_dims[shape], nt = token_counts[c];
        size_t elements = (size_t)experts * hidden * dim;
        size_t gate_bytes = elements / 256 * (five ? 176 : 144), down_bytes = elements / 32 * 34;
        size_t outputs = (size_t)nt * dim, projected_count = outputs * k;
        uint8_t *gate = malloc(gate_bytes), *up = malloc(gate_bytes);
        uint8_t *down = malloc(down_bytes);
        float *input = malloc(outputs * sizeof(float));
        float *projected = malloc(projected_count * sizeof(float));
        float *sum = malloc(outputs * sizeof(float));
        float *fused = malloc(outputs * sizeof(float));
        int32_t *ids = malloc((size_t)nt * k * sizeof(int32_t));
        assert(gate && up && down && input && projected && sum && fused && ids);
        qk8_fixture_gate(gate,elements,five,seed);
        qk8_fixture_gate(up,elements,five,seed+19);
        qk8_fixture_down(down,elements,seed+37);
        mxfp4_fixture_input(input, (int)outputs, seed);
        for (int t = 0; t < nt; t++) for (int j = 0; j < k; j++)
            ids[t * k + j] = (t * 7 + j * 3) % experts;
        struct ggml_init_params params = {64u * 1024u * 1024u, NULL, true};
        struct ggml_context *ctx = ggml_init(params);
        assert(ctx);
        struct ggml_tensor *g = ggml_new_tensor_3d(ctx, five ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K, dim, hidden, experts);
        struct ggml_tensor *u = ggml_new_tensor_3d(ctx, five ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K, dim, hidden, experts);
        struct ggml_tensor *d = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, hidden, dim, experts);
        struct ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 1, nt);
        struct ggml_tensor *routes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, nt);
        struct ggml_tensor *gates = ggml_mul_mat_id(ctx, g, x, routes);
        struct ggml_tensor *ups = ggml_mul_mat_id(ctx, u, x, routes);
        struct ggml_tensor *mid = ggml_swiglu_split(ctx, gates, ups);
        struct ggml_tensor *raw = ggml_mul_mat_id(ctx, d, mid, routes);
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, raw);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert(buffer);
        ggml_backend_tensor_set(g, gate, 0, gate_bytes);
        ggml_backend_tensor_set(u, up, 0, gate_bytes);
        ggml_backend_tensor_set(d, down, 0, down_bytes);
        ggml_backend_tensor_set(x, input, 0, outputs * sizeof(float));
        ggml_backend_tensor_set(routes, ids, 0, (size_t)nt * k * sizeof(int32_t));
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(raw, projected, 0, projected_count * sizeof(float));
        for (int t = 0; t < nt; t++) for (int i = 0; i < dim; i++) {
            float value = 0.0f, fused_value = 0.0f;
            for (int j = 0; j < k; j++) {
                /* Separate weighted multiply/add, matching existing fixtures. */
                volatile float weighted = projected[((size_t)t * k + j) * dim + i] *
                                          ((float)(j + 1) / 36.0f);
                value = value + weighted;
                fused_value = fmaf(projected[((size_t)t * k + j) * dim + i],
                                   (float)(j + 1) / 36.0f, fused_value);
            }
            sum[(size_t)t * dim + i] = value;
            fused[(size_t)t * dim + i] = fused_value;
        }
        printf("{%d,%d,%d,%d,%d,UINT64_C(0x%016llx)}, /* down=%016llx fused=%016llx gate=%d */\n",
            dim, hidden, experts, nt, seed,
            (unsigned long long)hash_words(sum, outputs),
            (unsigned long long)hash_words(projected, projected_count),
            (unsigned long long)hash_words(fused, outputs), five ? 13 : 12);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        free(gate); free(up); free(down); free(input); free(projected); free(sum); free(fused); free(ids);
    }
    ggml_backend_free(backend);
    return 0;
}
