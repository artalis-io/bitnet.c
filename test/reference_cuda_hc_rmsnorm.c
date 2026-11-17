/* Revalidate the retained HC RMSNorm inputs with pinned GGML CUDA df18bee11.
 * Link only the reference GGML libraries. Existing target hashes are not used
 * in this computation; dimensions, seeds, and input magnitudes are reused. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "cuda_hc_rmsnorm_reference.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    size_t cases = sizeof(cuda_hc_rmsnorm_reference) /
                   sizeof(cuda_hc_rmsnorm_reference[0]);
    for (size_t ci = 0; ci < cases; ci++) {
        const BnCudaHCRMSNormReference *r = &cuda_hc_rmsnorm_reference[ci];
        size_t count = (size_t)r->dim * r->streams;
        float *x = malloc(count * sizeof(float));
        float *w = malloc(count * sizeof(float));
        float *out = malloc(count * sizeof(float));
        assert(x && w && out);
        for (size_t i = 0; i < count; i++) {
            x[i] = sinf((float)(i * 13 + r->seed)) * r->magnitude;
            w[i] = cosf((float)(i * 7 + r->seed + 3)) * 1.5f;
        }
        struct ggml_init_params params = {16u*1024u*1024u, NULL, true};
        struct ggml_context *ctx = ggml_init(params);
        assert(ctx);
        struct ggml_tensor *tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r->dim, r->streams);
        struct ggml_tensor *tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r->dim, r->streams);
        struct ggml_tensor *ty = ggml_mul(ctx, ggml_rms_norm(ctx, tx, 1e-6f), tw);
        struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, ty);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert(buffer);
        ggml_backend_tensor_set(tx, x, 0, count*sizeof(float));
        ggml_backend_tensor_set(tw, w, 0, count*sizeof(float));
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(ty, out, 0, count*sizeof(float));
        uint64_t hash = UINT64_C(14695981039346656037);
        for (size_t i = 0; i < count; i++) {
            uint32_t bits; memcpy(&bits, out+i, sizeof(bits));
            hash = (hash ^ bits) * UINT64_C(1099511628211);
        }
        printf("%zu %016llx\n", ci, (unsigned long long)hash);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        free(x); free(w); free(out);
    }
    ggml_backend_free(backend);
    return 0;
}
