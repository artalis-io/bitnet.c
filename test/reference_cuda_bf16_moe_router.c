/* Replay pinned actual Flash router captures. Usage: probe CAPTURE_DIRECTORY.
 * Exact BF16 logits input, F32 probability outputs and I32 IDs/source rows.
 * No expected values enter the production operation. */
#include "gpu_cuda.h"
#include "backend_session.h"
#include "gguf.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif

static void *load(const char *dir, const char *name, const char *stage, size_t bytes) {
    char file[4096];
    int n = snprintf(file, sizeof(file), "%s/%s-%s.bin", dir, name, stage);
    assert(n > 0 && (size_t)n < sizeof(file));
    FILE *f = fopen(file, "rb");
    void *raw = malloc(bytes);
    assert(f && raw && fread(raw, 1, bytes, f) == bytes && fgetc(f) == EOF);
    fclose(f);
    return raw;
}

static BnGPUResidentView alloc(BnGPUBackend *g, void *r, int type, size_t n, const void *data) {
    BnGPUResidentView v;
    assert(!g->resident_alloc(r, type, n, &v));
    if (data) assert(!g->resident_write(r, v, data));
    return v;
}

static void *weight(BnGPUBackend *g, const char *dir, int layer, const char *name, int rows) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/layer%d-%s.bf16", dir, layer, name);
    assert(n > 0 && (size_t)n < sizeof(path));
    size_t bytes = (size_t)rows * 2560 * 2;
    void *data = malloc(bytes);
    FILE *f = fopen(path, "rb");
    assert(data && f && fread(data, 1, bytes, f) == bytes && fgetc(f) == EOF);
    fclose(f);
    void *w = g->buffer_create_quant_only(g->ctx, data, bytes, BN_GGUF_TENSOR_BF16, rows, 2560);
    assert(w); free(data); return w;
}

static size_t compare(BnGPUBackend *g, void *r, BnGPUResidentView v,
    const char *dir, const char *name, const char *stage, int width) {
    size_t bytes = v.count * (size_t)width;
    void *expected = load(dir, name, stage, bytes), *actual = malloc(bytes);
    assert(actual && !g->resident_read(r, v, actual));
    size_t differences = 0;
    for (size_t i = 0; i < v.count; i++) {
        if (memcmp((char *)actual + i * width, (char *)expected + i * width, (size_t)width)) {
            if (differences < 3) {
                uint32_t a = 0, b = 0;
                memcpy(&a, (char *)actual + i * width, (size_t)width);
                memcpy(&b, (char *)expected + i * width, (size_t)width);
                fprintf(stderr, "%s %s word%zu actual%08x expected%08x\n", name, stage, i, a, b);
            }
            differences++;
        }
    }
    if (differences) fprintf(stderr, "%s %s differences%zu/%zu\n", name, stage, differences, v.count);
    free(expected); free(actual);
    return differences;
}

static void guards(BnGPUBackend *g, void *r, const BnGPUMoETopKBF16ResidentPlan *valid) {
    BnGPUMoETopKBF16ResidentPlan p;
#define REJECT(statement) do { p = *valid; statement; assert(g->moe_topk_bf16_resident(g->ctx, r, &p) == -1); } while (0)
    REJECT(p.n_tokens = 0);
    REJECT(p.n_experts = 511);
    REJECT(p.k = 11);
    REJECT(p.renormalize = 2);
    REJECT(p.weights.count--);
    REJECT(p.indices.type = BN_GGUF_TENSOR_F32);
    REJECT(p.source_rows = p.indices);
    REJECT(p.logits.offset++);
    BnBackendSession *peer = bn_backend_session_create();
    assert(peer);
    void *other = bn_gpu_backend_resident_request(g, peer);
    assert(other);
    REJECT(p.weights = alloc(g, other, BN_GGUF_TENSOR_F32, p.weights.count, NULL));
    bn_backend_session_free(peer);
#undef REJECT
}

int main(int argc, char **argv) {
    assert(argc == 2 || (argc == 4 && !strcmp(argv[2], "--profile")));
    const char *profile = argc == 4 ? argv[3] : NULL;
    char file[4096];
    int len = snprintf(file, sizeof(file), "%s/cases.txt", argv[1]);
    assert(len > 0 && (size_t)len < sizeof(file));
    FILE *f = fopen(file, "r"); assert(f);
    BnGPUBackend *g = bn_gpu_cuda_create(); assert(g && g->moe_topk_bf16_resident);
    char name[128]; int nt, norm, checkpoint, layer, cases = 0;
    size_t words = 0, differences = 0, projection_words = 0, coupled_words = 0;
    while (fscanf(f, "%127s %d %d %d %d", name, &nt, &norm, &checkpoint, &layer) == 5) {
        if (profile && strcmp(profile, name)) continue;
        BnBackendSession *s = bn_backend_session_create(); assert(s);
        void *r = bn_gpu_backend_resident_request(g, s); assert(r);
        size_t n = (size_t)nt * 10;
        uint16_t *input = load(argv[1], name, "logits", (size_t)nt * 512 * 2);
        uint32_t *sentinel = malloc((n + 8) * 4); assert(sentinel);
        for (size_t i = 0; i < n + 8; i++) sentinel[i] = 0x5a5a5a5a;
        BnGPUMoETopKBF16ResidentPlan p = {0};
        p.n_tokens = nt; p.n_experts = 512; p.k = 10; p.renormalize = norm;
        p.logits = alloc(g, r, BN_GGUF_TENSOR_BF16, (size_t)nt * 512, input);
        BnGPUResidentView backing[3];
        for (int i = 0; i < 3; i++) backing[i] = alloc(g, r, i ? BN_GGUF_TENSOR_I32 : BN_GGUF_TENSOR_F32, n + 8, sentinel);
        p.weights = bn_gpu_resident_view_slice(backing[0], 4, n);
        p.indices = bn_gpu_resident_view_slice(backing[1], 4, n);
        p.source_rows = bn_gpu_resident_view_slice(backing[2], 4, n);
        guards(g, r, &p);
        for (int i = 0; i < 3; i++) {
            assert(!g->resident_read(r, backing[i], sentinel));
            for (size_t j = 0; j < n + 8; j++) assert(sentinel[j] == 0x5a5a5a5a);
        }
        for (int repeat = 0; repeat < 2; repeat++) {
#ifdef BN_CAPTURE_CUDA_PROFILER
            if (profile && repeat) assert(cudaProfilerStart() == cudaSuccess);
#endif
            assert(!g->moe_topk_bf16_resident(g->ctx, r, &p));
            assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
            if (profile && repeat) assert(cudaProfilerStop() == cudaSuccess);
#endif
            differences += compare(g, r, p.weights, argv[1], name, "weights", 4);
            differences += compare(g, r, p.indices, argv[1], name, "ids", 4);
            differences += compare(g, r, p.source_rows, argv[1], name, "source", 4);
            for (int i = 0; i < 3; i++) {
                assert(!g->resident_read(r, backing[i], sentinel));
                for (int j = 0; j < 4; j++) assert(sentinel[j] == 0x5a5a5a5a && sentinel[n + 4 + j] == 0x5a5a5a5a);
            }
            words += n * 3;
        }
        if (checkpoint && !profile) {
            void *gate = weight(g, argv[1], layer, "gate", 512);
            void *shared = weight(g, argv[1], layer, "shared", 1);
            uint16_t *x = load(argv[1], name, "input", (size_t)nt * 2560 * 2);
            BnGPUResidentView xv = alloc(g, r, BN_GGUF_TENSOR_BF16, (size_t)nt * 2560, x);
            BnGPUResidentView shared_out = alloc(g, r, BN_GGUF_TENSOR_BF16, (size_t)nt, NULL);
            assert(!g->linear_bf16_reserve(g->ctx, r, 0));
            BnGPUBF16LinearResidentPlan linear = {0};
            linear.n_tokens = nt; linear.rows = 512; linear.cols = 2560;
            linear.weight.type = BN_GGUF_TENSOR_BF16; linear.weight.decoded = gate;
            linear.x = xv; linear.out = p.logits;
            assert(!g->linear_bf16_resident(g->ctx, r, &linear));
            assert(!g->moe_topk_bf16_resident(g->ctx, r, &p));
            linear.rows = 1; linear.weight.decoded = shared; linear.out = shared_out;
            assert(!g->linear_bf16_resident(g->ctx, r, &linear));
            assert(!g->resident_synchronize(r));
            differences += compare(g, r, p.logits, argv[1], name, "logits", 2);
            differences += compare(g, r, shared_out, argv[1], name, "shared-logits", 2);
            differences += compare(g, r, p.weights, argv[1], name, "weights", 4);
            differences += compare(g, r, p.indices, argv[1], name, "ids", 4);
            differences += compare(g, r, p.source_rows, argv[1], name, "source", 4);
            projection_words += (size_t)nt * 513; coupled_words += n * 3;
            g->buffer_destroy(g->ctx, gate); g->buffer_destroy(g->ctx, shared); free(x);
        }
        free(input); free(sentinel); bn_backend_session_free(s);
        printf("%s checked\n", name); fflush(stdout); cases++;
    }
    assert(feof(f) && cases == (profile ? 1 : 38)); fclose(f); bn_gpu_cuda_destroy(g);
    printf("MOE_ROUTER_REPLAY cases%d words%zu projection_words%zu coupled_words%zu differences%zu\n",
        cases, words, projection_words, coupled_words, differences);
    assert(!differences);
    return 0;
}
