/* Complete resident common-HC checkpoint comparison.
 * Generate captures with reference_cuda_bf16_hc_checkpoint.py first.
 * Link with test_cuda_backend objects/libraries; run:
 *   probe CAPTURE_DIRECTORY [CHECKPOINT_FIRST_SHARD]
 * With a checkpoint, production preparation consumes raw mapped tensors;
 * capture weights are read only for independent output comparisons.
 * Actual checkpoint weights, synthetic inputs; no whole-model claim. */
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "model_arch.h"
#include "gguf.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif

static uint16_t *read_words(const char *base, const char *name, size_t count) {
    char path[2048];
    int len = snprintf(path, sizeof(path), "%s/%s", base, name);
    assert(len > 0 && (size_t)len < sizeof(path));
    FILE *file = fopen(path, "rb");
    assert(file);
    uint16_t *data = malloc(count * sizeof(*data));
    assert(data && fread(data, sizeof(*data), count, file) == count);
    assert(fgetc(file) == EOF);
    fclose(file);
    return data;
}

static void *weight(BnGPUBackend *gpu, const char *base, int layer,
                    const char *role, const char *suffix, int rows, int cols) {
    char name[256];
    if (layer < 0) snprintf(name, sizeof(name), "output_hc_%s.weight.bf16", suffix);
    else snprintf(name, sizeof(name), "blk.%d.hc_%s_%s.weight.bf16", layer, role, suffix);
    uint16_t *data = read_words(base, name, (size_t)rows * cols);
    void *handle = gpu->buffer_create_quant_only(gpu->ctx, data,
        (size_t)rows * cols * sizeof(*data), BN_GGUF_TENSOR_BF16, rows, cols);
    free(data);
    assert(handle);
    return handle;
}

static size_t verify_weight(BnGPUBackend *gpu, const char *base, int layer,
                            const char *role, const char *suffix, void *handle, int count) {
    char name[256];
    if (layer < 0) snprintf(name, sizeof(name), "output_hc_%s.weight.bf16", suffix);
    else snprintf(name, sizeof(name), "blk.%d.hc_%s_%s.weight.bf16", layer, role, suffix);
    uint16_t *expected = read_words(base, name, (size_t)count);
    uint16_t *ones = malloc((size_t)count * 2), *actual = malloc((size_t)count * 2);
    assert(ones && actual);
    for (int i = 0; i < count; i++) ones[i] = 0x3f80;
    /* Unit input/zero epsilon exposes gamma unchanged through a public API. */
    assert(!bn_gpu_backend_rmsnorm_bf16_batch(gpu, actual, handle, ones, 1, count, 0, 0));
    size_t differences = 0;
    for (int i = 0; i < count; i++) if (actual[i] != expected[i]) differences++;
    printf("CHECKPOINT_WEIGHT %s differences%zu/%d\n", name, differences, count);
    free(ones); free(actual); free(expected); assert(!differences);
    return (size_t)count;
}

static BnQWeight checkpoint_weight(BnGGUFFile *file, int layer, const char *role,
                                   const char *suffix) {
    char name[256];
    if (layer < 0) snprintf(name, sizeof(name), "output_hc_%s.weight", suffix);
    else snprintf(name, sizeof(name), "blk.%d.hc_%s_%s.weight", layer, role, suffix);
    int index = bn_gguf_find_tensor(file, name); assert(index >= 0);
    const BnGGUFTensorInfo *t = &file->tensors[index];
    assert(t->n_dims >= 1 && t->n_dims <= 2 && t->dims[0] <= INT_MAX &&
        (t->n_dims == 1 || t->dims[1] <= INT_MAX));
    BnQWeight w = {bn_gguf_tensor_data(file, index), (int)t->type,
        t->n_dims == 1 ? 1 : (int)t->dims[1], (int)t->dims[0], 1.0f};
    assert(w.data); return w;
}

static BnGPUResidentView alloc_view(BnGPUBackend *gpu, void *request, size_t n) {
    BnGPUResidentView view;
    assert(!gpu->resident_alloc(request, BN_GGUF_TENSOR_BF16, n, &view));
    return view;
}

static void linear(BnGPUBackend *gpu, void *request, void *handle,
                   int nt, int rows, int cols, BnGPUResidentView x,
                   BnGPUResidentView out) {
    BnGPUBF16LinearResidentPlan plan = {0};
    plan.n_tokens = nt; plan.rows = rows; plan.cols = cols;
    plan.weight.type = BN_GGUF_TENSOR_BF16;
    plan.weight.decoded = handle;
    plan.x = x; plan.out = out;
    assert(!gpu->linear_bf16_resident(gpu->ctx, request, &plan));
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3 && argc != 4) { fprintf(stderr, "usage: %s captures [checkpoint.gguf [output-captures]]\n", argv[0]); return 2; }
    BnGGUFFile *file = argc >= 3 ? bn_gguf_open_file(argv[2]) : NULL;
    assert(argc < 3 || file);
    BnGPUBackend *gpu = bn_gpu_cuda_create();
    assert(gpu);
    BnBackendModel *backend = NULL;
    const BnBackendBF16HCModel *hc = NULL;
    if (file) {
        BnConfig c = {0}; bn_model_arch_apply_config(&c, bn_model_arch_ops_for("qwen4exp"));
        c.n_layers = (int)bn_gguf_get_u32(file, "qwen4exp.block_count");
        c.dim = (int)bn_gguf_get_u32(file, "qwen4exp.embedding_length");
        c.hyper_connection_count = (int)bn_gguf_get_u32(file, "qwen4exp.hyper_connection.count");
        c.hyper_connection_rank = (int)bn_gguf_get_u32(file, "qwen4exp.hyper_connection.low_rank");
        assert(c.n_layers == 48 && c.dim == 2560 && c.hyper_connection_count == 4 && c.hyper_connection_rank == 320);
        BnWeights w = {0};
        w.layers = calloc((size_t)c.n_layers, sizeof(*w.layers)); assert(w.layers);
        for (int l = 0; l <= c.n_layers; l++) for (int branch = 0; branch < (l == c.n_layers ? 1 : 2); branch++) {
            const char *role = branch ? "ffn" : "attn";
            int layer = l == c.n_layers ? -1 : l;
            BnHyperConnectionWeights *raw = layer < 0 ? &w.hc_output :
                (branch ? &w.layers[l].hc_ffn : &w.layers[l].hc_attn);
            BnQWeight gamma = checkpoint_weight(file, layer, role, "norm");
            assert(gamma.type == BN_GGUF_TENSOR_F32 && gamma.cols == 10240);
            raw->norm = (float *)gamma.data;
            raw->down = checkpoint_weight(file, layer, role, "down");
            raw->up = checkpoint_weight(file, layer, role, "up");
            if (layer >= 0) raw->inject = checkpoint_weight(file, layer, role, "inject");
        }
        backend = bn_backend_model_create(); assert(backend);
        assert(!bn_backend_layout_prepare_bf16_hc_model(backend, gpu, &c, &w, SIZE_MAX));
        hc = bn_backend_layout_bf16_hc_model(backend); assert(hc && !bn_backend_layout_bf16(backend));
        assert(hc->n_layers == 48 && !hc->output->inject.decoded);
        printf("HC_MODEL_LAYOUT retained%zu peak%zu\n", hc->bytes, hc->peak_bytes);
        free(w.layers);
    }
    const int layers[] = {0, 2, 47}, tokens[] = {1, 5, 17, 129};
    const char *roles[] = {"attn", "ffn"};
    const char *stages[] = {"input", "block", "norm", "down", "scaled", "up", "mixed", "inject", "combined"};
    size_t words = 0, differences = 0, weight_words = 0;
    for (int li = 0; li < (argc == 4 ? 4 : 3); li++) for (int ri = 0; ri < (li == 3 ? 1 : 2); ri++) {
        int layer = li == 3 ? -1 : layers[li]; const char *role = li == 3 ? "output" : roles[ri];
        const char *captures = li == 3 ? argv[3] : argv[1];
        int combine = li != 3;
        void *norm, *down, *up, *inject;
        if (file) {
            const BnBackendBF16HCWeights *layout = layer < 0 ? hc->output :
                (ri ? &hc->ffn[layer] : &hc->attention[layer]);
            norm = layout->norm; down = layout->down.decoded;
            up = layout->up.decoded; inject = layout->inject.decoded;
            weight_words += verify_weight(gpu, captures, layer, role, "norm", norm, 10240);
            weight_words += verify_weight(gpu, captures, layer, role, "down", down, 320*10240);
            weight_words += verify_weight(gpu, captures, layer, role, "up", up, 320*10240);
            if (combine) weight_words += verify_weight(gpu, captures, layer, role, "inject", inject, 4*10240);
            printf("CHECKPOINT_LAYOUT layer%d %s retained%zu peak%zu\n", layer, role, layout->bytes, layout->peak_bytes);
        } else {
            norm = weight(gpu, argv[1], layer, role, "norm", 1, 10240);
            down = weight(gpu, argv[1], layer, role, "down", 320, 10240);
            up = weight(gpu, argv[1], layer, role, "up", 10240, 320);
            inject = weight(gpu, argv[1], layer, role, "inject", 4, 10240);
        }
        for (int ti = 0; ti < 4; ti++) {
            int nt = tokens[ti];
            BnBackendSession *session = bn_backend_session_create();
            assert(session);
            void *request = bn_gpu_backend_resident_request(gpu, session);
            assert(request && !gpu->linear_bf16_reserve(gpu->ctx, request, 0));
            size_t counts[] = {(size_t)nt*10240, (size_t)nt*2560,
                (size_t)nt*10240, (size_t)nt*320, (size_t)nt*320,
                (size_t)nt*10240, (size_t)nt*2560, (size_t)nt*4, (size_t)nt*10240};
            BnGPUResidentView v[9];
            for (int i = 0; i < 9; i++) v[i] = alloc_view(gpu, request, counts[i]);
            for (int i = 0; i < 2; i++) {
                char name[256]; snprintf(name, sizeof(name), "%d-%s-%d-%s.bin", layer, role, nt, stages[i]);
                uint16_t *data = read_words(captures, name, counts[i]);
                assert(!gpu->resident_write(request, v[i], data)); free(data);
            }
            int repeats = 1;
#ifdef BN_CAPTURE_CUDA_PROFILER
            if (li == 0 && ri == 0 && ti == 1) repeats = 4;
#endif
            for (int repeat = 0; repeat < repeats; repeat++) {
#ifdef BN_CAPTURE_CUDA_PROFILER
            if (repeats == 4 && repeat == 3)
                assert(cudaProfilerStart() == cudaSuccess);
#endif
            BnGPURMSNormResidentPlan n = {0};
            n.n_tokens = nt*4; n.dim = 2560; n.weight_offset = 1;
            n.reduction = BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32;
            n.eps = 1e-6f; n.weight = norm; n.weight_groups = 4;
            n.x = v[0]; n.out = v[2];
            assert(!gpu->rmsnorm_bf16_resident(gpu->ctx, request, &n));
            linear(gpu, request, down, nt, 320, 10240, v[2], v[3]);
            BnGPUScaledSiluResidentPlan s = {0};
            s.rows = nt; s.cols = 320; s.divisor = 4; s.x = v[3]; s.out = v[4];
            assert(!gpu->scaled_silu_bf16_resident(gpu->ctx, request, &s));
            linear(gpu, request, up, nt, 10240, 320, v[4], v[5]);
            BnGPUHyperMixResidentPlan m = {0};
            m.n_tokens = nt; m.dim = 2560; m.streams = 4;
            m.normalized = v[2]; m.gate = v[5]; m.out = v[6];
            assert(!gpu->hyper_connection_mix_bf16_resident(gpu->ctx, request, &m));
            if (combine) {
            linear(gpu, request, inject, nt, 4, 10240, v[2], v[7]);
            BnGPUHyperCombineResidentPlan c = {0};
            c.n_tokens = nt; c.dim = 2560; c.streams = 4;
            c.residual = v[0]; c.block_out = v[1]; c.inject = v[7]; c.out = v[8];
            assert(!gpu->hyper_connection_combine_bf16_resident(gpu->ctx, request, &c));
            }
            assert(!gpu->resident_synchronize(request));
#ifdef BN_CAPTURE_CUDA_PROFILER
            if (repeats == 4 && repeat == 3)
                assert(cudaProfilerStop() == cudaSuccess);
#endif
            }
            for (int i = 2; i < (combine ? 9 : 7); i++) {
                char name[256]; snprintf(name, sizeof(name), "%d-%s-%d-%s.bin", layer, role, nt, stages[i]);
                uint16_t *expected = read_words(captures, name, counts[i]);
                uint16_t *actual = malloc(counts[i]*2); assert(actual);
                assert(!gpu->resident_read(request, v[i], actual));
                size_t diff = 0;
                for (size_t j = 0; j < counts[i]; j++) if (actual[j] != expected[j]) diff++;
                printf("%s differences %zu/%zu\n", name, diff, counts[i]); fflush(stdout);
                words += counts[i]; differences += diff; free(actual); free(expected);
            }
            bn_backend_session_free(session);
        }
        if (!file) {
            gpu->buffer_destroy(gpu->ctx, norm); gpu->buffer_destroy(gpu->ctx, down);
            gpu->buffer_destroy(gpu->ctx, up); gpu->buffer_destroy(gpu->ctx, inject);
        }
    }
    bn_backend_model_free(backend);
    bn_gguf_free(file);
    bn_gpu_cuda_destroy(gpu);
    printf("HC_CHECKPOINT_WEIGHTS exact_words%zu\n", weight_words);
    printf("HC_RESIDENT_CHECKPOINT words%zu differences%zu\n", words, differences);
    return differences ? 1 : 0;
}
