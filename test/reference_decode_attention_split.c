/* Independent GGML CUDA split-attention oracle; does not link bitnet.c.
 * Usage: reference_decode_attention_split [valid_keys seed]
 * Build like reference_ssm_projection.c, adding -lm. See docs/benchmarks.md.
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

static uint64_t reference_hash(const float *data, size_t count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, data + i, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    enum { d = 256, heads = 24, kv_heads = 4, padded = 256 };
    int valid = argc > 1 ? atoi(argv[1]) : 1;
    int seed = argc > 2 ? atoi(argv[2]) : 0;
    assert(argc <= 3 && valid > 0 && valid <= padded && (seed == 0 || seed == 23));
    size_t nq = (size_t)d * heads, nc = (size_t)d * padded * kv_heads;
    float *q = malloc(nq * sizeof(float)), *k = calloc(nc, sizeof(float));
    float *v = calloc(nc, sizeof(float)), *out = malloc(nq * sizeof(float));
    ggml_fp16_t *mask = malloc((size_t)padded * 32 * sizeof(*mask));
    assert(q && k && v && out && mask);
    float mag = seed ? 3.0f : 1.0f;
    for (size_t i = 0; i < nq; i++)
        q[i] = sinf((float)(i * 17 + 3 + seed)) * (1.31f * mag);
    for (int h = 0; h < kv_heads; h++)
        for (int t = 0; t < valid; t++)
            for (int j = 0; j < d; j++) {
                size_t i = ((size_t)h * padded + t) * d + j;
                k[i] = sinf((float)(i * 11 + 9 + seed)) * (1.7f * mag);
                v[i] = cosf((float)(i * 7 + 5 + seed)) * (0.73f * mag);
            }
    for (int i = 0; i < padded * 32; i++)
        mask[i] = ggml_fp32_to_fp16(i % padded < valid ? 0.0f : -INFINITY);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    struct ggml_init_params params = {64u * 1024u * 1024u, NULL, true};
    struct ggml_context *ctx = ggml_init(params);
    assert(ctx);
    struct ggml_tensor *qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, heads, 1);
    struct ggml_tensor *kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, padded, kv_heads, 1);
    struct ggml_tensor *vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, padded, kv_heads, 1);
    struct ggml_tensor *mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded, 32);
    struct ggml_tensor *scores = ggml_mul_mat(ctx, kt, qt);
    assert(ggml_prec_set_acc(scores, GGML_PREC_F32));
    struct ggml_tensor *probabilities = ggml_soft_max_ext(
        ctx, scores, mt, 1.0f / sqrtf((float)d), 0.0f);
    struct ggml_tensor *values = ggml_cont(ctx, ggml_transpose(ctx, vt));
    struct ggml_tensor *result = ggml_mul_mat(ctx, values, probabilities);
    assert(ggml_prec_set_acc(result, GGML_PREC_F32));
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    ggml_backend_tensor_set(qt, q, 0, nq * sizeof(float));
    ggml_backend_tensor_set(kt, k, 0, nc * sizeof(float));
    ggml_backend_tensor_set(vt, v, 0, nc * sizeof(float));
    ggml_backend_tensor_set(mt, mask, 0, (size_t)padded * 32 * sizeof(*mask));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(result, out, 0, nq * sizeof(float));
    printf("{%d, %d, UINT64_C(0x%016llx)},\n", valid, seed,
           (unsigned long long)reference_hash(out, nq));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    free(mask); free(out); free(v); free(k); free(q);
    return 0;
}
