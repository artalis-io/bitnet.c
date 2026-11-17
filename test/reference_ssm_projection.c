/* Independent GGML CUDA graph for the SSM projection/state fixtures.
 * This tool does not link bitnet.c. Build/run instructions and reference
 * library provenance are in docs/benchmarks.md (SSM reference generator).
 * Usage: reference_ssm_projection [dim heads tokens [dump_prefix]]
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <assert.h>
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

static void reference_set(struct ggml_tensor *tensor, const float *data) {
    ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
}

static void reference_dump(const char *prefix, const char *stage,
                           struct ggml_tensor *tensor) {
    if (!prefix) return;
    char path[1024];
    int size = snprintf(path, sizeof(path), "%s-%s.bin", prefix, stage);
    assert(size > 0 && (size_t)size < sizeof(path));
    size_t bytes = ggml_nbytes(tensor);
    void *data = malloc(bytes);
    assert(data);
    ggml_backend_tensor_get(tensor, data, 0, bytes);
    FILE *file = fopen(path, "wb");
    assert(file && fwrite(data, 1, bytes, file) == bytes);
    assert(fclose(file) == 0);
    free(data);
}

int main(int argc, char **argv) {
    int dim = argc > 1 ? atoi(argv[1]) : 256;
    int heads = argc > 2 ? atoi(argv[2]) : 4;
    int tokens = argc > 3 ? atoi(argv[3]) : 2;
    const char *prefix = argc > 4 ? argv[4] : NULL;
    assert(dim > 0 && dim <= 8192 && heads > 0 && heads <= 64 &&
           tokens > 0 && tokens <= 4096 && argc <= 5);
    const int head_size = 128;
    size_t nw = (size_t)dim * heads;
    size_t ns = (size_t)head_size * head_size * heads;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    struct ggml_init_params params = {64u * 1024u * 1024u, NULL, true};
    struct ggml_context *ctx = ggml_init(params);
    assert(ctx);
    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tokens);
    struct ggml_tensor *aw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, heads);
    struct ggml_tensor *bw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, heads);
    struct ggml_tensor *dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    struct ggml_tensor *norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_size, 1, tokens, 1);
    struct ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_size, 1, tokens, 1);
    struct ggml_tensor *v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_size, heads, tokens, 1);
    struct ggml_tensor *state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                   head_size, head_size, heads, 1);
    struct ggml_tensor *xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), norm);
    struct ggml_tensor *alpha = ggml_mul_mat(ctx, aw, xn);
    struct ggml_tensor *softplus = ggml_softplus(ctx, ggml_add(ctx, alpha, dt));
    struct ggml_tensor *gate = ggml_mul(ctx, softplus, a);
    struct ggml_tensor *beta = ggml_sigmoid(ctx, ggml_mul_mat(ctx, bw, xn));
    struct ggml_tensor *out = ggml_gated_delta_net(ctx, q, k, v,
        ggml_reshape_4d(ctx, gate, 1, heads, tokens, 1),
        ggml_reshape_4d(ctx, beta, 1, heads, tokens, 1), state, 1);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out);
    struct ggml_tensor *decay = ggml_exp(ctx, gate);
    ggml_build_forward_expand(graph, decay);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);

    float *xx = malloc((size_t)dim * tokens * sizeof(float));
    float *aa = malloc(nw * sizeof(float)), *bb = malloc(nw * sizeof(float));
    float *bias = malloc((size_t)heads * sizeof(float));
    float *al = malloc((size_t)heads * sizeof(float));
    float *ss = malloc(ns * sizeof(float)), *ones = malloc((size_t)dim * sizeof(float));
    float *zeros = calloc((size_t)head_size * heads * tokens, sizeof(float));
    float *result = malloc(ns * sizeof(float));
    assert(xx && aa && bb && bias && al && ss && ones && zeros && result);
    for (int t = 0; t < tokens; t++)
        for (int d = 0; d < dim; d++)
            xx[(size_t)t * dim + d] = (float)((d * 37 + t * 19) % 251 - 125) / 259.0f;
    for (int r = 0; r < heads; r++) {
        bias[r] = (float)(r % 7 - 3) / 64.0f;
        al[r] = -0.1f - (float)(r % 13) / 100.0f;
        for (int d = 0; d < dim; d++) {
            aa[(size_t)r * dim + d] = (float)((r * 13 + d * 17) % 257 - 128) / 257.0f;
            bb[(size_t)r * dim + d] = (float)((r * 11 + d * 19) % 251 - 125) / 259.0f;
        }
    }
    for (size_t i = 0; i < ns; i++)
        ss[i] = (float)((int)((i * 13) % 257) - 128) / 4096.0f;
    for (int i = 0; i < dim; i++) ones[i] = 1;
    reference_set(x, xx); reference_set(aw, aa); reference_set(bw, bb);
    reference_set(dt, bias); reference_set(a, al); reference_set(norm, ones);
    reference_set(state, ss); reference_set(q, zeros);
    reference_set(k, zeros); reference_set(v, zeros);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    reference_dump(prefix, "norm", xn); reference_dump(prefix, "alpha", alpha);
    reference_dump(prefix, "softplus", softplus); reference_dump(prefix, "gate", gate);
    reference_dump(prefix, "decay", decay);
    ggml_backend_tensor_get(out, result, (size_t)tokens * head_size * heads * sizeof(float),
                            ns * sizeof(float));
    printf("{%d, %d, %d, UINT64_C(0x%016llx)},\n", dim, heads, tokens,
           (unsigned long long)reference_hash(result, ns));
    free(result); free(zeros); free(ones); free(ss); free(al); free(bias);
    free(bb); free(aa); free(xx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
