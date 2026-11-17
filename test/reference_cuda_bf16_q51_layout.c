/* Compare every decoded Q5_1 weight and resident GEMM output against the
 * independent gguf-py/Torch captures from reference_cuda_bf16_q51_layout.py.
 * Layout words must match exactly. GEMM may differ by one BF16 step because
 * cuBLAS and Torch choose different FP32 reduction orders near rounding ties.
 * Usage: probe CAPTURE_DIRECTORY [OBSERVATION_DIRECTORY]. Observation mode
 * writes actual GEMM results separately and is not a qualification pass. */
#include "gpu_cuda.h"
#include "backend_session.h"
#include "gguf.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *load(const char *directory, const char *name,
                  const char *suffix, size_t bytes) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s-%s.bin", directory, name, suffix);
    assert(n > 0 && (size_t)n < sizeof(path));
    FILE *f = fopen(path, "rb");
    void *data = malloc(bytes);
    assert(f && data && fread(data, 1, bytes, f) == bytes && fgetc(f) == EOF);
    fclose(f);
    return data;
}

static void exact(const uint16_t *actual, const uint16_t *expected, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (actual[i] != expected[i]) {
            fprintf(stderr, "word %zu: %04x != %04x\n", i, actual[i], expected[i]);
            abort();
        }
    }
}

static size_t matrix_compare(const uint16_t *actual, const uint16_t *expected, size_t n) {
    size_t differences = 0;
    for (size_t i = 0; i < n; i++) {
        int distance = (int)actual[i] - (int)expected[i];
        assert(distance >= -1 && distance <= 1);
        differences += distance != 0;
    }
    return differences;
}

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 3);
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/cases.txt", argv[1]);
    assert(n > 0 && (size_t)n < sizeof(path));
    FILE *cases = fopen(path, "r");
    assert(cases);
    BnGPUBackend *g = bn_gpu_cuda_create();
    assert(g && bn_gpu_backend_can_bf16_decode(g));
    /* Reject incomplete blocks and truncated payloads before decoding. */
    unsigned char truncated[24] = {0};
    const int invalid_cols[] = {31, 64};
    for (size_t j = 0; j < sizeof(invalid_cols) / sizeof(invalid_cols[0]); j++) {
        void *bad = g->buffer_create_quant_only(g->ctx, truncated,
            sizeof(truncated), BN_GGUF_TENSOR_Q5_1, 1, invalid_cols[j]);
        assert(bad && !bn_gpu_backend_buffer_create_bf16_decoded(g, bad));
        g->buffer_destroy(g->ctx, bad);
    }
    size_t decoded_words = 0, output_words = 0, matrix_differences = 0;
    int rows, cols, count = 0;
    char name[128];
    while (fscanf(cases, "%127s %d %d", name, &rows, &cols) == 3) {
        assert(rows > 0 && cols > 0 && !(cols % 32));
        size_t nw = (size_t)rows * cols, bytes = nw / 32 * 24;
        void *raw = load(argv[1], name, "raw", bytes);
        void *source = g->buffer_create_quant_only(g->ctx, raw, bytes,
            BN_GGUF_TENSOR_Q5_1, rows, cols);
        assert(source);
        void *decoded = bn_gpu_backend_buffer_create_bf16_decoded(g, source);
        assert(decoded);
        g->buffer_destroy(g->ctx, source);
        free(raw);
        uint16_t *gold = load(argv[1], name, "decoded", nw * 2);
        uint16_t *ones = malloc(nw * 2), *out = malloc((nw + 2) * 2);
        assert(ones && out);
        for (size_t i = 0; i < nw; i++) ones[i] = 0x3f80;
        out[0] = out[nw + 1] = 0x5a5a;
        /* Norm(ones, eps=0) exposes every decoded coefficient exactly. */
        assert(!bn_gpu_backend_rmsnorm_bf16_batch(g, out + 1, decoded,
            ones, 1, (int)nw, 0.0f, 0));
        assert(out[0] == 0x5a5a && out[nw + 1] == 0x5a5a);
        printf("%s decoded\n", name); fflush(stdout);
        exact(out + 1, gold, nw);
        decoded_words += nw;
        free(ones); free(out); free(gold);
        const int batches[] = {1, 5, 17, 129};
        for (size_t j = 0; j < sizeof(batches) / sizeof(batches[0]); j++) {
            int nt = batches[j];
            size_t nx = (size_t)nt * cols, ny = (size_t)nt * rows;
            char suffix[32];
            snprintf(suffix, sizeof(suffix), "x-%d", nt);
            uint16_t *x = load(argv[1], name, suffix, nx * 2);
            snprintf(suffix, sizeof(suffix), "y-%d", nt);
            uint16_t *y = load(argv[1], name, suffix, ny * 2);
            uint16_t *actual = malloc((ny + 16) * 2);
            assert(actual);
            for (size_t i = 0; i < ny + 16; i++) actual[i] = 0x5a5a;
            BnBackendSession *s = bn_backend_session_create();
            assert(s);
            void *r = bn_gpu_backend_resident_request(g, s);
            assert(r && !g->linear_bf16_reserve(g->ctx, r, 0));
            BnGPUResidentView xv, storage;
            assert(!g->resident_alloc(r, BN_GGUF_TENSOR_BF16, nx, &xv));
            assert(!g->resident_alloc(r, BN_GGUF_TENSOR_BF16, ny + 16, &storage));
            assert(!g->resident_write(r, xv, x));
            assert(!g->resident_write(r, storage, actual));
            BnGPUBF16LinearResidentPlan plan = {0};
            plan.n_tokens = nt; plan.rows = rows; plan.cols = cols;
            plan.weight.type = BN_GGUF_TENSOR_BF16;
            plan.weight.decoded = decoded;
            plan.x = xv; plan.out = bn_gpu_resident_view_slice(storage, 8, ny);
            assert(!g->linear_bf16_resident(g->ctx, r, &plan));
            assert(!g->resident_read(r, storage, actual));
            printf("%s GEMM nt%d\n", name, nt); fflush(stdout);
            if (argc == 3) {
                int len = snprintf(path, sizeof(path), "%s/%s-y-%d.bin", argv[2], name, nt);
                assert(len > 0 && (size_t)len < sizeof(path));
                FILE *f = fopen(path, "wb");
                assert(f && fwrite(actual + 8, 2, ny, f) == ny);
                fclose(f);
            } else {
                matrix_differences += matrix_compare(actual + 8, y, ny);
            }
            for (size_t i = 0; i < 8; i++)
                assert(actual[i] == 0x5a5a && actual[ny + 8 + i] == 0x5a5a);
            output_words += ny;
            bn_backend_session_free(s);
            free(x); free(y); free(actual);
        }
        g->buffer_destroy(g->ctx, decoded);
        count++;
        printf("%s checked\n", name); fflush(stdout);
    }
    assert(feof(cases) && count == 5);
    fclose(cases);
    bn_gpu_cuda_destroy(g);
    printf("Q51_BF16_LAYOUT_%s cases%d decoded_words%zu output_words%zu matrix_ulp1_words%zu\n",
        argc == 3 ? "OBSERVED" : "PASSED",
        count, decoded_words, output_words, matrix_differences);
    return 0;
}
