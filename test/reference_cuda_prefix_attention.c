/* Independent cached-prefix flash-attention oracle. Link pinned GGML CUDA
 * libraries only. Usage: reference_cuda_prefix_attention nt pos nk gqa seed
 * scale_is_one window [flash=1]. Output is token-major, matching runtime buffers. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    assert(argc == 8 || argc == 9);
    const int flash = argc == 9 ? atoi(argv[8]) : 1;
    assert(flash == 0 || flash == 1);
    const int nt = atoi(argv[1]), pos = atoi(argv[2]), nk = atoi(argv[3]);
    const int gqa = atoi(argv[4]), seed = atoi(argv[5]);
    const int unit = atoi(argv[6]), window = atoi(argv[7]);
    assert(nt > 1 && pos > 0 && nt + pos <= 2048 && nk > 0 && nk <= 4);
    assert((gqa == 6 || gqa == 16) && (seed == 0 || seed == 23));
    assert((unit == 0 || unit == 1) && window >= 0);
    const int hd = 256, nh = nk * gqa, keys = pos + nt;
    const int padded = (keys + 255) / 256 * 256;
    const int queries = (nt + 31) / 32 * 32;
    const size_t nq = (size_t)nt * nh * hd, nc = (size_t)padded * nk * hd;
    float *q = malloc(nq * sizeof(float)), *out = malloc(nq * sizeof(float));
    ggml_fp16_t *k = calloc(nc, sizeof(*k)), *v = calloc(nc, sizeof(*v));
    ggml_fp16_t *mask = malloc((size_t)padded * queries * sizeof(*mask));
    assert(q && out && k && v && mask);
    const float mag = seed ? 3.0f : 1.0f;
    for (size_t i = 0; i < nq; i++)
        q[i] = sinf((float)(i * 17 + 3 + seed)) * (1.31f * mag);
    for (size_t i = 0; i < (size_t)keys * nk * hd; i++) {
        k[i] = ggml_fp32_to_fp16(sinf((float)(i * 11 + 9 + seed)) * (1.7f * mag));
        v[i] = ggml_fp32_to_fp16(cosf((float)(i * 7 + 5 + seed)) * (0.73f * mag));
    }
    for (int t = 0; t < queries; t++)
        for (int j = 0; j < padded; j++) {
            int visible = t < nt && j <= pos + t &&
                (!window || j > pos + t - window);
            mask[(size_t)t * padded + j] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    struct ggml_init_params params = {16u * 1024u * 1024u, NULL, true};
    struct ggml_context *ctx = ggml_init(params);
    assert(ctx);
    struct ggml_tensor *qt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, nt);
    struct ggml_tensor *kt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, nk, padded);
    struct ggml_tensor *vt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, nk, padded);
    struct ggml_tensor *mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded, queries);
    struct ggml_tensor *qp = ggml_permute(ctx, qt, 0, 2, 1, 3);
    struct ggml_tensor *kp = ggml_permute(ctx, kt, 0, 2, 1, 3);
    struct ggml_tensor *vp = ggml_permute(ctx, vt, 0, 2, 1, 3);
    struct ggml_tensor *result, *scores = NULL, *probabilities = NULL;
    if (flash) {
        result = ggml_flash_attn_ext(ctx, qp, kp, vp, mt,
                                    unit ? 1.0f : 0.0625f, 0.0f, 0.0f);
    } else {
        scores = ggml_mul_mat(ctx, kp, qp);
        assert(ggml_prec_set_acc(scores, GGML_PREC_F32));
        probabilities = ggml_soft_max_ext(ctx, scores, mt,
                                    unit ? 1.0f : 0.0625f, 0.0f);
        result = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, vp)),
                             probabilities);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
    }
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    ggml_backend_tensor_set(qt, q, 0, nq * sizeof(float));
    ggml_backend_tensor_set(kt, k, 0, nc * sizeof(*k));
    ggml_backend_tensor_set(vt, v, 0, nc * sizeof(*v));
    ggml_backend_tensor_set(mt, mask, 0, (size_t)padded * queries * sizeof(*mask));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(result, out, 0, nq * sizeof(float));
    const char *dump_dir = getenv("GGML_REFERENCE_DUMP_DIR");
    if (dump_dir) {
        struct ggml_tensor *tensors[] = {scores, probabilities, result};
        const char *names[] = {"scores", "probabilities", "output"};
        for (int i = 0; i < 3; i++) {
            if (!tensors[i]) continue;
            size_t bytes = ggml_nbytes(tensors[i]);
            void *data = malloc(bytes);
            char path[1024];
            int len = snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, names[i]);
            assert(data && len > 0 && (size_t)len < sizeof(path));
            ggml_backend_tensor_get(tensors[i], data, 0, bytes);
            FILE *f = fopen(path, "wb");
            assert(f && fwrite(data, 1, bytes, f) == bytes);
            assert(fclose(f) == 0);
            free(data);
        }
    }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < nq; i++) {
        uint32_t bits;
        memcpy(&bits, out + i, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    printf("{%d, %d, %d, %d, %d, %d, %d, UINT64_C(0x%016llx)},\n",
        nt, pos, nk, gqa, seed, unit, window, (unsigned long long)hash);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx); ggml_backend_free(backend);
    free(q); free(out); free(k); free(v); free(mask);
    return 0;
}
