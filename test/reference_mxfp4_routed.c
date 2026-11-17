/* Independent GGML CUDA projections for the first MXFP4/Q5_K routed fixture.
 * Does not link bitnet.c. Build like reference_ssm_projection.c, adding -lm.
 * Prints ordered fused and separate reduction hashes from the same projections.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "cuda_mxfp4_fixture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t reference_hash(const float *values, int count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, values + i, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    return hash;
}

int main(void) {
    enum { dim = 256, hidden = 512, experts = 16, k = 8 };
    size_t elements = (size_t)experts * hidden * dim;
    size_t gate_bytes = elements / 32 * 17;
    size_t down_bytes = elements / 256 * 176;
    uint8_t *gate = malloc(gate_bytes), *up = malloc(gate_bytes);
    uint8_t *down = malloc(down_bytes);
    float input[dim], projected[k * dim], fused[dim], separate[dim];
    int32_t ids[k];
    assert(gate && up && down);
    mxfp4_fixture_weights(gate, elements, 0);
    mxfp4_fixture_weights(up, elements, 19);
    for (size_t b = 0; b < elements / 32; b++) {
        gate[b * 17] -= 8;
        up[b * 17] -= 8;
    }
    const int seed = 37;
    for (size_t b = 0; b < elements / 256; b++) {
        uint8_t *w = down + b * 176;
        uint16_t scale = (uint16_t)(0x1800 + ((b + seed) % 17) * 37);
        uint16_t minimum = (uint16_t)(0x1400 + ((b * 3 + seed) % 19) * 29);
        memcpy(w, &scale, 2);
        memcpy(w + 2, &minimum, 2);
        for (int j = 0; j < 12; j++) w[4 + j] = (uint8_t)(b * 7 + j * 13 + seed);
        for (int j = 0; j < 32; j++) w[16 + j] = (uint8_t)(b * 11 + j * 17 + seed);
        for (int j = 0; j < 128; j++) w[48 + j] = (uint8_t)(b * 19 + j * 23 + seed);
    }
    mxfp4_fixture_input(input, dim, 0);
    for (int j = 0; j < k; j++) ids[j] = j * 3 % experts;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    struct ggml_init_params params = {64u * 1024u * 1024u, NULL, true};
    struct ggml_context *ctx = ggml_init(params);
    assert(ctx);
    struct ggml_tensor *g = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, dim, hidden, experts);
    struct ggml_tensor *u = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, dim, hidden, experts);
    struct ggml_tensor *d = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_K, hidden, dim, experts);
    struct ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 1, 1);
    struct ggml_tensor *routes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
    struct ggml_tensor *gates = ggml_mul_mat_id(ctx, g, x, routes);
    struct ggml_tensor *ups = ggml_mul_mat_id(ctx, u, x, routes);
    struct ggml_tensor *mid = ggml_mul(ctx, ggml_silu(ctx, gates), ups);
    struct ggml_tensor *raw = ggml_mul_mat_id(ctx, d, mid, routes);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, raw);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    ggml_backend_tensor_set(g, gate, 0, gate_bytes);
    ggml_backend_tensor_set(u, up, 0, gate_bytes);
    ggml_backend_tensor_set(d, down, 0, down_bytes);
    ggml_backend_tensor_set(x, input, 0, sizeof(input));
    ggml_backend_tensor_set(routes, ids, 0, sizeof(ids));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(raw, projected, 0, sizeof(projected));

    for (int i = 0; i < dim; i++) {
        float a = 0.0f, b = 0.0f;
        for (int j = 0; j < k; j++) {
            float weight = (float)(j + 1) / 36.0f;
            float value = projected[j * dim + i];
            a = fmaf(value, weight, a);
            /* Volatile enforces the intermediate rounding independently of
             * the host compiler's default contraction settings. */
            volatile float weighted = value * weight;
            b = b + weighted;
        }
        fused[i] = a;
        separate[i] = b;
    }
    printf("fused=%016llx separate=%016llx\n",
           (unsigned long long)reference_hash(fused, dim),
           (unsigned long long)reference_hash(separate, dim));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    free(down); free(up); free(gate);
    return 0;
}
