/* Device grouping/gather and BF16 weighted ordered reduction. Replays the
 * independent actual Flash FFN captures without rewriting any reference. */
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include "gguf.h"
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif

static void *load(const char *dir, const char *prefix, const char *suffix, size_t bytes) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s-%s.bin", dir, prefix, suffix);
    assert(n > 0 && (size_t)n < sizeof(path));
    FILE *f = fopen(path, "rb");
    void *p = malloc(bytes);
    assert(f && p && fread(p, 1, bytes, f) == bytes && fgetc(f) == EOF);
    fclose(f);
    return p;
}

static BnGPUResidentView alloc(BnGPUBackend *g, void *r, int type, size_t count) {
    BnGPUResidentView v;
    assert(!g->resident_alloc(r, type, count, &v));
    return v;
}

static uint16_t bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float f32(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void group_guards(BnGPUBackend *g, void *r, BnGPUMoEGroupBF16ResidentPlan p) {
    BnGPUMoEGroupBF16ResidentPlan q;
#define REJECT(change) do { q = p; change; assert(g->moe_group_bf16_resident(g->ctx, r, &q) == -1); } while (0)
    REJECT(q.n_tokens = 0);
    REJECT(q.n_tokens = INT_MAX);
    REJECT(q.dim = INT_MAX);
    REJECT(q.n_experts = 511);
    REJECT(q.k = 9);
    REJECT(q.indices.type = BN_GGUF_TENSOR_F32);
    REJECT(q.indices.count--);
    REJECT(q.x.count--);
    REJECT(q.offsets.count--);
    REJECT(q.cursors.count--);
    REJECT(q.slots.count--);
    REJECT(q.inverse.count--);
    REJECT(q.gathered.count--);
    REJECT(q.status.count--);
    REJECT(q.cursors = q.offsets; q.cursors.count = 512);
    REJECT(q.inverse = q.slots);
    REJECT(q.gathered = q.x; q.gathered.count *= 10);
    REJECT(q.slots.offset++);
    BnBackendSession *other = bn_backend_session_create();
    void *peer = bn_gpu_backend_resident_request(g, other);
    assert(peer);
    REJECT(q.status = alloc(g, peer, BN_GGUF_TENSOR_I32, 1));
    assert(g->moe_group_bf16_resident(g->ctx, peer, &p) == -1);
    bn_backend_session_free(other);
#undef REJECT
}

static void reduce_guards(BnGPUBackend *g, void *r, BnGPUMoEReduceBF16ResidentPlan p) {
    BnGPUMoEReduceBF16ResidentPlan q;
#define REJECT(change) do { q = p; change; assert(g->moe_reduce_bf16_resident(g->ctx, r, &q) == -1); } while (0)
    REJECT(q.n_tokens = 0);
    REJECT(q.n_tokens = INT_MAX);
    REJECT(q.dim = INT_MAX);
    REJECT(q.k = 9);
    REJECT(q.grouped.type = BN_GGUF_TENSOR_F32);
    REJECT(q.grouped.count--);
    REJECT(q.inverse.count--);
    REJECT(q.weights.count--);
    REJECT(q.weighted.count--);
    REJECT(q.out.count--);
    REJECT(q.status.count--);
    REJECT(q.weighted = q.grouped);
    REJECT(q.out = q.grouped; q.out.count /= 10);
    REJECT(q.inverse.offset++);
    BnBackendSession *other = bn_backend_session_create();
    void *peer = bn_gpu_backend_resident_request(g, other);
    assert(peer);
    REJECT(q.status = alloc(g, peer, BN_GGUF_TENSOR_I32, 1));
    assert(g->moe_reduce_bf16_resident(g->ctx, peer, &p) == -1);
    bn_backend_session_free(other);
#undef REJECT
}

static size_t replay(BnGPUBackend *g, int nt, int dim, const int32_t *ids,
    const uint16_t *x, const float *weights, const uint16_t *down,
    const uint16_t *weighted, const uint16_t *out, int adversarial, int profile) {
    size_t n = (size_t)nt * 10, nx = (size_t)nt * dim, total = n * dim;
    BnBackendSession *s = bn_backend_session_create(); assert(s);
    void *r = bn_gpu_backend_resident_request(g, s); assert(r);
    BnGPUMoEGroupBF16ResidentPlan p = {0};
    p.n_tokens = nt; p.k = 10; p.n_experts = 512; p.dim = dim;
    p.indices = alloc(g, r, BN_GGUF_TENSOR_I32, n);
    p.x = alloc(g, r, BN_GGUF_TENSOR_BF16, nx);
    p.offsets = alloc(g, r, BN_GGUF_TENSOR_I32, 513);
    p.cursors = alloc(g, r, BN_GGUF_TENSOR_I32, 512);
    p.slots = alloc(g, r, BN_GGUF_TENSOR_I32, n);
    p.inverse = alloc(g, r, BN_GGUF_TENSOR_I32, n);
    p.gathered = alloc(g, r, BN_GGUF_TENSOR_BF16, total);
    p.status = alloc(g, r, BN_GGUF_TENSOR_I32, 1);
    BnGPUMoEReduceBF16ResidentPlan q = {0};
    q.n_tokens = nt; q.k = 10; q.dim = dim;
    q.grouped = alloc(g, r, BN_GGUF_TENSOR_BF16, total);
    q.inverse = p.inverse; q.status = p.status;
    q.weights = alloc(g, r, BN_GGUF_TENSOR_F32, n);
    q.weighted = alloc(g, r, BN_GGUF_TENSOR_BF16, total);
    q.out = alloc(g, r, BN_GGUF_TENSOR_BF16, nx);
    int32_t *slots = malloc(n * 4), *inverse = malloc(n * 4), offsets[513], cursors[512], status;
    uint16_t *actual = malloc(total * 2), *grouped = malloc(total * 2);
    assert(slots && inverse && actual && grouped);
    assert(!g->resident_write(r, p.indices, ids) && !g->resident_write(r, p.x, x));
    assert(!g->resident_write(r, q.weights, weights));
    for (size_t j = 0; j < total; j++) actual[j] = 0x5a5a;
    assert(!g->resident_write(r, q.weighted, actual) && !g->resident_write(r, q.out, actual));
    status = 77;
    for (int j = 0; j < 513; j++) offsets[j] = -7;
    for (int j = 0; j < 512; j++) cursors[j] = -7;
    for (size_t j = 0; j < n; j++) slots[j] = inverse[j] = -7;
    assert(!g->resident_write(r, p.status, &status) && !g->resident_write(r, p.offsets, offsets));
    assert(!g->resident_write(r, p.cursors, cursors) && !g->resident_write(r, p.slots, slots));
    assert(!g->resident_write(r, p.inverse, inverse) && !g->resident_write(r, p.gathered, actual));
    group_guards(g, r, p); reduce_guards(g, r, q);
    assert(!g->resident_read(r, q.out, grouped));
    assert(!memcmp(actual, grouped, nx * 2));
    assert(!g->resident_read(r, p.status, &status) && status == 77);
    assert(!g->resident_read(r, p.offsets, offsets) && !g->resident_read(r, p.cursors, cursors));
    for (int j = 0; j < 513; j++) assert(offsets[j] == -7);
    for (int j = 0; j < 512; j++) assert(cursors[j] == -7);
    assert(!g->resident_read(r, p.slots, slots) && !g->resident_read(r, p.inverse, inverse));
    for (size_t j = 0; j < n; j++) assert(slots[j] == -7 && inverse[j] == -7);
    assert(!g->resident_read(r, p.gathered, grouped) && !memcmp(actual, grouped, total * 2));
    size_t words = 0;
    for (int repeat = 0; repeat < 3; repeat++) {
        assert(!g->moe_group_bf16_resident(g->ctx, r, &p));
        assert(!g->resident_read(r, p.status, &status) && !status);
        assert(!g->resident_read(r, p.offsets, offsets) && !g->resident_read(r, p.cursors, cursors));
        assert(!g->resident_read(r, p.slots, slots) && !g->resident_read(r, p.inverse, inverse));
        assert(!g->resident_read(r, p.gathered, actual));
        assert(offsets[0] == 0 && offsets[512] == (int)n);
        for (int e = 0; e < 512; e++) {
            int count = 0;
            for (size_t j = 0; j < n; j++) count += ids[j] == e;
            assert(offsets[e + 1] == offsets[e] + count && cursors[e] == offsets[e + 1]);
            for (int row = offsets[e]; row < offsets[e + 1]; row++) {
                int slot = slots[row];
                assert(slot >= 0 && slot < (int)n && ids[slot] == e && inverse[slot] == row);
                assert(!memcmp(actual + (size_t)row * dim, x + (size_t)(slot / 10) * dim, (size_t)dim * 2));
                memcpy(grouped + (size_t)row * dim, down + (size_t)slot * dim, (size_t)dim * 2);
            }
        }
        assert(!g->resident_write(r, q.grouped, grouped));
        assert(!g->moe_reduce_bf16_resident(g->ctx, r, &q));
        assert(!g->resident_read(r, q.status, &status) && !status);
        assert(!g->resident_read(r, q.weighted, actual));
        if (memcmp(actual, weighted, total * 2)) {
            for (size_t j = 0; j < total; j++) if (actual[j] != weighted[j]) {
                fprintf(stderr, "weighted nt%d dim%d word%zu got%04x expected%04x\n", nt, dim, j, actual[j], weighted[j]); break;
            }
            assert(0);
        }
        assert(!g->resident_read(r, q.out, actual));
        if (memcmp(actual, out, nx * 2)) {
            for (size_t j = 0; j < nx; j++) if (actual[j] != out[j]) {
                fprintf(stderr, "sum nt%d dim%d word%zu got%04x expected%04x\n", nt, dim, j, actual[j], out[j]); break;
            }
            assert(0);
        }
        words += total * 2 + nx;
    }
    assert(!g->resident_read(r, p.x, actual) && !memcmp(actual, x, nx * 2));
    assert(!g->resident_read(r, p.indices, slots) && !memcmp(slots, ids, n * 4));
    if (profile == 2) {
        assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
        assert(cudaProfilerStart() == cudaSuccess);
#endif
        assert(!g->moe_reduce_bf16_resident(g->ctx, r, &q));
        assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
        assert(cudaProfilerStop() == cudaSuccess);
#endif
        assert(!g->resident_read(r, q.out, actual) && !memcmp(actual, out, nx * 2));
    } else if (profile == 1) {
        assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
        assert(cudaProfilerStart() == cudaSuccess);
#endif
        assert(!g->moe_group_bf16_resident(g->ctx, r, &p));
        assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
        assert(cudaProfilerStop() == cudaSuccess);
#endif
        /* The old grouped down no longer matches an unspecified new grouping;
         * regenerate its permutation outside the profiled range. */
        assert(!g->resident_read(r, p.slots, slots));
        for (size_t row = 0; row < n; row++) memcpy(grouped + row * dim,
            down + (size_t)slots[row] * dim, (size_t)dim * 2);
        assert(!g->resident_write(r, q.grouped, grouped));
        assert(!g->moe_reduce_bf16_resident(g->ctx, r, &q));
        assert(!g->resident_read(r, q.out, actual) && !memcmp(actual, out, nx * 2));
    }
    if (adversarial) {
        int32_t *bad = malloc(n * 4); assert(bad);
        for (int which = 0; which < 2; which++) {
            memcpy(bad, ids, n * 4); bad[n - 1] = which ? 512 : -1;
            for (size_t j = 0; j < n; j++) slots[j] = inverse[j] = -7;
            for (size_t j = 0; j < total; j++) actual[j] = 0x5a5a;
            assert(!g->resident_write(r, p.indices, bad));
            assert(!g->resident_write(r, p.slots, slots) && !g->resident_write(r, p.inverse, inverse));
            assert(!g->resident_write(r, p.gathered, actual));
            assert(!g->resident_write(r, q.weighted, actual) && !g->resident_write(r, q.out, actual));
            assert(!g->moe_group_bf16_resident(g->ctx, r, &p));
            assert(!g->resident_read(r, p.status, &status) && status == 1);
            assert(!g->resident_read(r, p.slots, bad) && !memcmp(bad, slots, n * 4));
            assert(!g->resident_read(r, p.inverse, bad) && !memcmp(bad, inverse, n * 4));
            assert(!g->resident_read(r, p.gathered, grouped) && !memcmp(grouped, actual, total * 2));
            assert(!g->moe_reduce_bf16_resident(g->ctx, r, &q));
            assert(!g->resident_read(r, q.out, grouped) && !memcmp(grouped, actual, nx * 2));
            assert(!g->resident_read(r, q.weighted, grouped) && !memcmp(grouped, actual, total * 2));
        }
        assert(!g->resident_write(r, p.indices, ids));
        assert(!g->moe_group_bf16_resident(g->ctx, r, &p));
        assert(!g->resident_read(r, p.status, &status) && !status);
        assert(!g->resident_read(r, p.inverse, inverse));
        for (int which = 0; which < 2; which++) {
            memcpy(bad, inverse, n * 4); bad[0] = which ? (int)n : -1;
            status = 0;
            assert(!g->resident_write(r, p.status, &status) && !g->resident_write(r, p.inverse, bad));
            assert(!g->moe_reduce_bf16_resident(g->ctx, r, &q));
            assert(!g->resident_read(r, p.status, &status) && status == 2);
            assert(!g->resident_read(r, q.out, grouped) && !memcmp(grouped, actual, nx * 2));
            assert(!g->resident_read(r, q.weighted, grouped) && !memcmp(grouped, actual, total * 2));
        }
        assert(!g->moe_group_bf16_resident(g->ctx, r, &p));
        assert(!g->resident_read(r, p.status, &status) && !status);
        assert(!g->resident_read(r, p.slots, slots));
        for (size_t row = 0; row < n; row++) memcpy(grouped + row * dim,
            down + (size_t)slots[row] * dim, (size_t)dim * 2);
        assert(!g->resident_write(r, q.grouped, grouped));
        assert(!g->moe_reduce_bf16_resident(g->ctx, r, &q));
        assert(!g->resident_read(r, q.out, actual) && !memcmp(actual, out, nx * 2));
        free(bad);
    }
    free(slots); free(inverse); free(actual); free(grouped);
    bn_backend_session_free(s);
    return words;
}

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 4 || (argc == 5 &&
        (!strcmp(argv[4], "--profile") || !strcmp(argv[4], "--profile-reduce"))));
    int profile = argc == 5 ? (!strcmp(argv[4], "--profile") ? 1 : 2) : 0;
    int synthetic_only = argc == 2 && !strcmp(argv[1], "--synthetic");
    BnGPUBackend *g = bn_gpu_cuda_create();
    assert(g);
    if (synthetic_only && (!g->moe_group_bf16_resident || !g->moe_reduce_bf16_resident)) {
        puts("CUDA resident MoE grouping skipped: optional precision profile unavailable");
        bn_gpu_cuda_destroy(g);
        return 0;
    }
    assert(g->moe_group_bf16_resident && g->moe_reduce_bf16_resident);
    int only_layer = argc >= 4 ? atoi(argv[2]) : -1, only_nt = argc >= 4 ? atoi(argv[3]) : -1;
    size_t words = 0; int cases = 0;
    const int layers[] = {2, 3, 47}, tokens[] = {1, 5, 17, 63, 64, 65, 128, 129, 257};
    for (int li = 0; li < 3; li++) for (int ti = 0; ti < 9; ti++) {
        int layer = layers[li], nt = tokens[ti];
        if (synthetic_only || (only_layer >= 0 && (layer != only_layer || nt != only_nt))) continue;
        size_t n = (size_t)nt * 10, nx = (size_t)nt * 2560, total = n * 2560;
        char prefix[128]; snprintf(prefix, sizeof(prefix), "layer%d-%d", layer, nt);
        int32_t *ids = load(argv[1], prefix, "ids", n * 4);
        uint16_t *x = load(argv[1], prefix, "input", nx * 2);
        float *w = load(argv[1], prefix, "weights", n * 4);
        uint16_t *d = load(argv[1], prefix, "routed-down", total * 2);
        uint16_t *weighted = load(argv[1], prefix, "routed-weighted", total * 2);
        uint16_t *out = load(argv[1], prefix, "routed", nx * 2);
        words += replay(g, nt, 2560, ids, x, w, d, weighted, out, 0, profile);
        free(ids); free(x); free(w); free(d); free(weighted); free(out);
        cases++; printf("%s exact\n", prefix); fflush(stdout);
    }
    assert(cases == (synthetic_only ? 0 : only_layer >= 0 ? 1 : 27));
    int synthetic = 0;
    if (only_layer < 0) {
        const int dims[] = {1, 37, 2560}, nts[] = {1, 3, 65, 257};
        for (int di = 0; di < 3; di++) for (int ti = 0; ti < 4; ti++) for (int mode = 0; mode < 3; mode++) {
            int nt = nts[ti], dim = dims[di];
            size_t n = (size_t)nt * 10, nx = (size_t)nt * dim, total = n * dim;
            int32_t *ids = malloc(n * 4);
            uint16_t *x = malloc(nx * 2), *d = malloc(total * 2), *weighted = malloc(total * 2), *out = malloc(nx * 2);
            float *w = malloc(n * 4); assert(ids && x && d && weighted && out && w);
            for (size_t j = 0; j < nx; j++) x[j] = bf16((float)((int)(j * 31 % 509) - 254) / 127);
            for (size_t j = 0; j < n; j++) {
                ids[j] = mode == 0 ? 511 : mode == 1 ? (int)(j * 37 % 512) : (j % 3 ? 0 : 256);
                w[j] = (float)((int)(j * 19 % 29) - 14) / 31;
                for (int col = 0; col < dim; col++) {
                    size_t i = j * dim + col;
                    d[i] = bf16((float)((int)(i * 37 % 251) - 125) / 17);
                    weighted[i] = bf16(f32(d[i]) * w[j]);
                }
            }
            for (size_t j = 0; j < nx; j++) {
                float sum = 0;
                for (int rank = 0; rank < 10; rank++) sum += f32(weighted[(j / dim * 10 + rank) * dim + j % dim]);
                out[j] = bf16(sum);
            }
            words += replay(g, nt, dim, ids, x, w, d, weighted, out, 1, 0);
            free(ids); free(x); free(w); free(d); free(weighted); free(out); synthetic++;
        }
    }
    bn_gpu_cuda_destroy(g);
    printf("RESIDENT_MOE_GROUP cases%d synthetic%d bf16_words%zu differences0\n", cases, synthetic, words);
    return 0;
}
