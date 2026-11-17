/* Native expert projection/FFN and routed streaming replay against independent
 * Flash captures. Usage: FIRST_SHARD CAPTURES [LAYER TOTAL_TOKENS [EXPERT]].
 * Isolated down projections use captured activation to isolate arithmetic.
 * Complete FFNs consume original input; --grouped retains per-expert waits,
 * --banked/--banked1 stream actual raw weights through bounded async banks. */
#include "gpu_cuda.h"
#include "backend_session.h"
#include "gguf.h"
#include "quant.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif

static void *load(const char *dir, const char *prefix, const char *stage, size_t bytes) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s-%s.bin", dir, prefix, stage);
    assert(n > 0 && (size_t)n < sizeof(path));
    FILE *f = fopen(path, "rb"); void *data = malloc(bytes);
    assert(f && data && fread(data, 1, bytes, f) == bytes && fgetc(f) == EOF);
    fclose(f); return data;
}

static BnGPUResidentView alloc(BnGPUBackend *g, void *r, size_t count) {
    BnGPUResidentView v;
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_BF16, count, &v)); return v;
}

static void guards(BnGPUBackend *g, void *r, const BnGPUMoEProjectBF16ResidentPlan *valid) {
    BnGPUMoEProjectBF16ResidentPlan p;
#define REJECT(statement) do { p = *valid; statement; assert(g->moe_project_bf16_resident(g->ctx, r, &p) == -1); } while (0)
    REJECT(p.n_tokens = 0);
    REJECT(p.total_tokens = p.n_tokens - 1);
    REJECT(p.rows++);
    REJECT(p.cols--);
    REJECT(p.type = BN_GGUF_TENSOR_Q6_K);
    REJECT(p.weight = NULL);
    REJECT(p.x.count--);
    REJECT(p.out.count--);
    REJECT(p.out = p.x);
    REJECT(p.out.offset++);
    BnBackendSession *peer = bn_backend_session_create(); assert(peer);
    void *other = bn_gpu_backend_resident_request(g, peer); assert(other);
    REJECT(p.out = alloc(g, other, p.out.count));
    p = *valid;
    p.x = alloc(g, other, valid->x.count);
    p.out = alloc(g, other, valid->out.count);
    assert(g->moe_project_bf16_resident(g->ctx, other, &p) == -1);
    assert(!g->linear_a8_bf16_reserve(g->ctx, other, 32));
    assert(g->moe_project_bf16_resident(g->ctx, other, &p) == -1);
    assert(!g->linear_a8_bf16_reserve(g->ctx, other, valid->x.count));
    bn_backend_session_free(peer);
#undef REJECT
}

/* Composition checks read back only offsets/status for host weight staging.
 * Gathered inputs, expert intermediates and weighted outputs stay resident.
 * The bank variant overlaps staged DMA with compute and protects slot reuse;
 * grouped matrix launches and original-input quantization reuse remain open. */
#include "cuda_bf16_expert_bank_replay.h"

static size_t grouped_ffn(BnGPUBackend *g, BnGGUFFile *f, const int *indices,
    const unsigned char **weights, const size_t *weight_bytes, const char *dir,
    const char *prefix, int nt, const int32_t *ids, const uint16_t *input, int bank_capacity, int profile, int reuse, int grouped) {
    size_t n = (size_t)nt * 10, nx = (size_t)nt * 2560, total = n * 2560;
    BnBackendSession *s = bn_backend_session_create(); assert(s);
    void *r = bn_gpu_backend_resident_request(g, s); assert(r);
    BnGPUMoEGroupBF16ResidentPlan p = {0};
    p.n_tokens = nt; p.k = 10; p.n_experts = 512; p.dim = 2560;
    p.x = alloc(g, r, nx); p.gathered = alloc(g, r, total);
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, n, &p.indices));
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, 513, &p.offsets));
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, 512, &p.cursors));
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, n, &p.slots));
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, n, &p.inverse));
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, 1, &p.status));
    BnGPUResidentView gate = alloc(g, r, n * 640), up = alloc(g, r, n * 640);
    BnGPUMoEReduceBF16ResidentPlan q = {0};
    q.n_tokens = nt; q.k = 10; q.dim = 2560;
    q.inverse = p.inverse; q.status = p.status;
    q.grouped = alloc(g, r, total); q.weighted = alloc(g, r, total); q.out = alloc(g, r, nx);
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_F32, n, &q.weights));
    float *routing = load(dir, prefix, "weights", n * 4);
    uint16_t *expected_weighted = load(dir, prefix, "routed-weighted", total * 2);
    uint16_t *expected_out = load(dir, prefix, "routed", nx * 2);
    uint16_t *actual = malloc(total * 2); assert(actual);
    assert(!g->resident_write(r, p.indices, ids) && !g->resident_write(r, p.x, input));
    assert(!g->resident_write(r, q.weights, routing));
    assert(!g->linear_a8_bf16_reserve(g->ctx, r, nx));
    assert(!g->moe_group_bf16_resident(g->ctx, r, &p));
    int32_t offsets[513], status;
    assert(!g->resident_read(r, p.offsets, offsets));
    assert(!g->resident_read(r, p.status, &status) && !status);
    assert(offsets[0] == 0 && offsets[512] == (int)n);
    int groups = 0;
    BnGPUMoEBankStats stats = {0};
    if (bank_capacity) groups = banked_ffn(g, r, f, indices, weights, weight_bytes, nt, offsets,
        p.gathered, gate, up, q.grouped, p.x, p.slots, p.status, bank_capacity, profile, reuse, grouped, &stats);
    else for (int expert = 0; expert < 512; expert++) {
        int first = offsets[expert], ng = offsets[expert + 1] - first;
        if (!ng) continue;
        assert(ng > 0 && ng <= nt); groups++;
        void *raw[3];
        for (int role = 0; role < 3; role++) {
            raw[role] = g->buffer_create_quant_only(g->ctx,
                weights[role] + (size_t)expert * weight_bytes[role], weight_bytes[role],
                (int)f->tensors[indices[role]].type, role == 2 ? 2560 : 640, role == 2 ? 640 : 2560);
            assert(raw[role]);
        }
        BnGPUMoEFFNBF16ResidentPlan full = {0};
        full.total_tokens = nt; full.ffn.n_tokens = ng; full.ffn.dim = 2560; full.ffn.hidden_dim = 640;
        full.ffn.weights.gate.raw = raw[0]; full.ffn.weights.gate.type = (int)f->tensors[indices[0]].type;
        full.ffn.weights.up.raw = raw[1]; full.ffn.weights.up.type = (int)f->tensors[indices[1]].type;
        full.ffn.weights.down.raw = raw[2]; full.ffn.weights.down.type = (int)f->tensors[indices[2]].type;
        full.ffn.x = bn_gpu_resident_view_slice(p.gathered, (size_t)first * 2560, (size_t)ng * 2560);
        full.ffn.gate = bn_gpu_resident_view_slice(gate, (size_t)first * 640, (size_t)ng * 640);
        full.ffn.up = bn_gpu_resident_view_slice(up, (size_t)first * 640, (size_t)ng * 640);
        full.ffn.out = bn_gpu_resident_view_slice(q.grouped, (size_t)first * 2560, (size_t)ng * 2560);
        assert(!g->moe_ffn_bf16_resident(g->ctx, r, &full));
        assert(!g->resident_synchronize(r));
        for (int role = 0; role < 3; role++) g->buffer_destroy(g->ctx, raw[role]);
    }
    assert(!g->moe_reduce_bf16_resident(g->ctx, r, &q));
    assert(!g->resident_read(r, q.status, &status) && !status);
    assert(!g->resident_read(r, q.weighted, actual) && !memcmp(actual, expected_weighted, total * 2));
    assert(!g->resident_read(r, q.out, actual) && !memcmp(actual, expected_out, nx * 2));
    if (bank_capacity) printf("%s banked cap%d groups%d banks%zu upload%zu reserved%zu busy%zu words%zu differences0 reuse%d input_bytes%zu grouped%d meta_bytes%zu group_host%zu group_device%zu\n",
        prefix, bank_capacity, groups, stats.submitted_banks, stats.upload_bytes, stats.reserved_device_bytes,
        stats.busy_acquires, total + nx, reuse, stats.reserved_input_bytes, grouped, stats.metadata_upload_bytes,
        stats.reserved_group_host_bytes, stats.reserved_group_device_bytes);
    else printf("%s native-routed groups%d words%zu differences0\n", prefix, groups, total + nx);
    fflush(stdout);
    free(routing); free(expected_weighted); free(expected_out); free(actual);
    bn_backend_session_free(s);
    return total + nx;
}

int main(int argc, char **argv) {
    int group_launch = argc > 3 && (!strcmp(argv[argc - 1], "--grouped-bank") ||
        !strcmp(argv[argc - 1], "--grouped-bank1") || !strcmp(argv[argc - 1], "--grouped-bank-profile"));
    int reuse = argc > 3 && (!strcmp(argv[argc - 1], "--prepared") ||
        !strcmp(argv[argc - 1], "--prepared1") || !strcmp(argv[argc - 1], "--prepared-profile") || group_launch);
    int bank_capacity = argc > 3 && (!strcmp(argv[argc - 1], "--banked") ||
        !strcmp(argv[argc - 1], "--banked-profile") || !strcmp(argv[argc - 1], "--prepared") ||
        !strcmp(argv[argc - 1], "--prepared-profile") || !strcmp(argv[argc - 1], "--grouped-bank") ||
        !strcmp(argv[argc - 1], "--grouped-bank-profile")) ? 8 :
        argc > 3 && (!strcmp(argv[argc - 1], "--banked1") || !strcmp(argv[argc - 1], "--prepared1") || !strcmp(argv[argc - 1], "--grouped-bank1")) ? 1 : 0;
    int bank_profile = argc > 3 && (!strcmp(argv[argc - 1], "--banked-profile") || !strcmp(argv[argc - 1], "--prepared-profile") || !strcmp(argv[argc - 1], "--grouped-bank-profile"));
    int grouped = bank_capacity || (argc > 3 && !strcmp(argv[argc - 1], "--grouped"));
    if (grouped) argc--;
    assert(argc == 3 || argc == 5 || argc == 6 || (argc == 7 && !strcmp(argv[6], "--profile")));
    int profile = argc == 7;
    int only_layer = argc >= 5 ? atoi(argv[3]) : -1;
    int only_nt = argc >= 5 ? atoi(argv[4]) : -1;
    int only_expert = argc >= 6 ? atoi(argv[5]) : -1;
    assert(!grouped || (only_expert < 0 && !profile));
    BnGGUFFile *f = bn_gguf_open_file(argv[1]); assert(f);
    BnGPUBackend *g = bn_gpu_cuda_create(); assert(g && g->moe_project_bf16_resident && g->moe_ffn_bf16_resident);
    assert(!grouped || (g->moe_group_bf16_resident && g->moe_reduce_bf16_resident));
    assert(!bank_capacity || (g->moe_bank_reserve && g->moe_bank_acquire && g->moe_bank_submit && g->moe_bank_stats));
    const int layers[] = {3, 2, 47}, tokens[] = {1, 5, 17, 63, 64, 65, 128, 129, 257};
    size_t checked = 0, different = 0, projections = 0, groups = 0, ffn_words = 0;
    int cases = 0;
    for (size_t li = 0; li < 3; li++) {
        int layer = layers[li]; if (only_layer >= 0 && layer != only_layer) continue;
        int indices[3]; const char *roles[] = {"gate", "up", "down"};
        size_t weight_bytes[3]; const unsigned char *weights[3];
        for (int role = 0; role < 3; role++) {
            char name[128]; snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", layer, roles[role]);
            indices[role] = bn_gguf_find_tensor(f, name); assert(indices[role] >= 0);
            const BnGGUFTensorInfo *t = &f->tensors[indices[role]];
            assert(t->n_dims == 3 && t->dims[2] == 512 && t->dims[0] == (uint64_t)(role == 2 ? 640 : 2560) && t->dims[1] == (uint64_t)(role == 2 ? 2560 : 640));
            assert(bn_gguf_tensor_size(t->type, t->dims[0] * t->dims[1], &weight_bytes[role]));
            weights[role] = bn_gguf_tensor_data(f, indices[role]); assert(weights[role]);
        }
        for (size_t ti = 0; ti < 9; ti++) {
            int nt = tokens[ti]; if (only_nt >= 0 && nt != only_nt) continue;
            char prefix[128]; snprintf(prefix, sizeof(prefix), "layer%d-%d", layer, nt);
            int32_t *ids = load(argv[2], prefix, "ids", (size_t)nt * 10 * 4);
            uint16_t *input = load(argv[2], prefix, "input", (size_t)nt * 2560 * 2);
            uint16_t *act = load(argv[2], prefix, "routed-activation", (size_t)nt * 10 * 640 * 2);
            uint16_t *gateup = load(argv[2], prefix, "routed-gateup", (size_t)nt * 10 * 1280 * 2);
            uint16_t *down = load(argv[2], prefix, "routed-down", (size_t)nt * 10 * 2560 * 2);
            if (grouped) {
                checked += grouped_ffn(g, f, indices, weights, weight_bytes, argv[2], prefix, nt, ids, input,
                    bank_capacity, bank_profile, reuse, group_launch);
                free(ids); free(input); free(act); free(gateup); free(down); cases++;
                continue;
            }
            size_t capacity = (size_t)nt * 2560;
            uint16_t *x = malloc(capacity * 2), *actual = malloc((capacity + 16) * 2);
            int *slots = malloc((size_t)nt * 10 * sizeof(*slots)); assert(x && actual && slots);
            BnBackendSession *s = bn_backend_session_create(); assert(s);
            void *r = bn_gpu_backend_resident_request(g, s); assert(r);
            BnGPUResidentView xv = alloc(g, r, capacity), storage = alloc(g, r, capacity + 16);
            BnGPUResidentView fg = alloc(g, r, (size_t)nt * 640), fu = alloc(g, r, (size_t)nt * 640);
            assert(!g->linear_a8_bf16_reserve(g->ctx, r, capacity));
            size_t case_diff = 0, case_groups = 0;
            for (int expert = 0; expert < 512; expert++) {
                if (only_expert >= 0 && expert != only_expert) continue;
                int ng = 0;
                for (int slot = 0; slot < nt * 10; slot++) if (ids[slot] == expert) slots[ng++] = slot;
                if (!ng) continue;
                assert(ng <= nt); groups++; case_groups++;
                void *expert_weights[3];
                for (int role = 0; role < 3; role++) {
                    int cols = role == 2 ? 640 : 2560, rows = role == 2 ? 2560 : 640;
                    int type = (int)f->tensors[indices[role]].type;
                    void *w = g->buffer_create_quant_only(g->ctx,
                        weights[role] + (size_t)expert * weight_bytes[role], weight_bytes[role], type, rows, cols);
                    assert(w);
                    expert_weights[role] = w;
                    for (int j = 0; j < ng; j++) {
                        const uint16_t *src = role == 2 ? act + (size_t)slots[j] * 640 : input + (size_t)(slots[j] / 10) * 2560;
                        memcpy(x + (size_t)j * cols, src, (size_t)cols * 2);
                    }
                    size_t ny = (size_t)ng * rows;
                    for (size_t j = 0; j < ny + 16; j++) actual[j] = 0x5a5a;
                    BnGPUMoEProjectBF16ResidentPlan p = {0};
                    p.n_tokens = ng; p.total_tokens = nt; p.rows = rows; p.cols = cols; p.type = type; p.weight = w;
                    p.x = bn_gpu_resident_view_slice(xv, 0, (size_t)ng * cols);
                    p.out = bn_gpu_resident_view_slice(storage, 8, ny);
                    BnGPUResidentView used = bn_gpu_resident_view_slice(storage, 0, ny + 16);
                    assert(!g->resident_write(r, p.x, x) && !g->resident_write(r, used, actual));
                    if (case_groups == 1) {
                        guards(g, r, &p);
                        assert(!g->resident_read(r, used, actual));
                        for (size_t j = 0; j < ny + 16; j++) assert(actual[j] == 0x5a5a);
                    }
                    assert(!g->moe_project_bf16_resident(g->ctx, r, &p));
                    assert(!g->resident_read(r, used, actual));
                    for (int j = 0; j < 8; j++) assert(actual[j] == 0x5a5a && actual[ny + 8 + j] == 0x5a5a);
                    size_t d = 0;
                    for (int j = 0; j < ng; j++) for (int col = 0; col < rows; col++) {
                        uint16_t expected = role == 2 ? down[(size_t)slots[j] * 2560 + col] : gateup[(size_t)slots[j] * 1280 + role * 640 + col];
                        uint16_t got = actual[8 + (size_t)j * rows + col];
                        if (got != expected) {
                            if (different + case_diff + d < 4) fprintf(stderr, "%s expert%d %s group%d row%d actual%04x expected%04x\n", prefix, expert, roles[role], ng, col, got, expected);
                            d++;
                        }
                    }
                    case_diff += d; checked += ny; projections++;
                }
                for (int j = 0; j < ng; j++) memcpy(x + (size_t)j * 2560,
                    input + (size_t)(slots[j] / 10) * 2560, 2560 * 2);
                size_t ny = (size_t)ng * 2560, nh = (size_t)ng * 640;
                for (size_t j = 0; j < ny + 16; j++) actual[j] = 0x5a5a;
                BnGPUMoEFFNBF16ResidentPlan full = {0};
                full.total_tokens = nt;
                full.ffn.n_tokens = ng; full.ffn.dim = 2560; full.ffn.hidden_dim = 640;
                full.ffn.weights.gate.raw = expert_weights[0]; full.ffn.weights.gate.type = (int)f->tensors[indices[0]].type;
                full.ffn.weights.up.raw = expert_weights[1]; full.ffn.weights.up.type = (int)f->tensors[indices[1]].type;
                full.ffn.weights.down.raw = expert_weights[2]; full.ffn.weights.down.type = (int)f->tensors[indices[2]].type;
                full.ffn.x = bn_gpu_resident_view_slice(xv, 0, (size_t)ng * 2560);
                full.ffn.gate = bn_gpu_resident_view_slice(fg, 0, nh);
                full.ffn.up = bn_gpu_resident_view_slice(fu, 0, nh);
                full.ffn.out = bn_gpu_resident_view_slice(storage, 8, ny);
                BnGPUResidentView used = bn_gpu_resident_view_slice(storage, 0, ny + 16);
                assert(!g->resident_write(r, full.ffn.x, x) && !g->resident_write(r, used, actual));
                assert(!g->resident_write(r, full.ffn.gate, actual) && !g->resident_write(r, full.ffn.up, actual));
                if (case_groups == 1) {
                    BnGPUMoEFFNBF16ResidentPlan bad = full;
                    bad.ffn.weights.down.raw = NULL;
                    assert(g->moe_ffn_bf16_resident(g->ctx, r, &bad) == -1);
                    bad = full; bad.ffn.up = bad.ffn.gate;
                    assert(g->moe_ffn_bf16_resident(g->ctx, r, &bad) == -1);
                    bad = full; bad.total_tokens = ng - 1;
                    assert(g->moe_ffn_bf16_resident(g->ctx, r, &bad) == -1);
                    bad = full; bad.ffn.activation_profile = BN_GPU_BF16_FFN_SILU_FUSED;
                    assert(g->moe_ffn_bf16_resident(g->ctx, r, &bad) == -1);
                    assert(!g->resident_read(r, full.ffn.gate, actual));
                    for (size_t j = 0; j < nh; j++) assert(actual[j] == 0x5a5a);
                    assert(!g->resident_read(r, full.ffn.up, actual));
                    for (size_t j = 0; j < nh; j++) assert(actual[j] == 0x5a5a);
                    assert(!g->resident_read(r, used, actual));
                    for (size_t j = 0; j < ny + 16; j++) assert(actual[j] == 0x5a5a);
                }
                assert(!g->moe_ffn_bf16_resident(g->ctx, r, &full));
                if (profile) {
                    assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
                    assert(cudaProfilerStart() == cudaSuccess);
#endif
                    assert(!g->moe_ffn_bf16_resident(g->ctx, r, &full));
                    assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
                    assert(cudaProfilerStop() == cudaSuccess);
#endif
                }
                assert(!g->resident_read(r, used, actual));
                for (int j = 0; j < 8; j++) assert(actual[j] == 0x5a5a && actual[ny + 8 + j] == 0x5a5a);
                for (int j = 0; j < ng; j++) for (int col = 0; col < 2560; col++) {
                    if (actual[8 + (size_t)j * 2560 + col] != down[(size_t)slots[j] * 2560 + col]) case_diff++;
                }
                assert(!g->resident_read(r, full.ffn.gate, actual));
                for (int j = 0; j < ng; j++) for (int col = 0; col < 640; col++) {
                    if (actual[(size_t)j * 640 + col] != act[(size_t)slots[j] * 640 + col]) case_diff++;
                }
                ffn_words += ny + nh;
                for (int role = 0; role < 3; role++) g->buffer_destroy(g->ctx, expert_weights[role]);
            }
            different += case_diff; assert(case_groups > 0);
            printf("%s groups%zu differences%zu\n", prefix, case_groups, case_diff); fflush(stdout); cases++;
            bn_backend_session_free(s);
            free(ids); free(input); free(act); free(gateup); free(down); free(x); free(actual); free(slots);
        }
    }
    bn_gpu_cuda_destroy(g); bn_gguf_free(f);
    if (bank_capacity) printf("RESIDENT_MOE_BANK cases%d cap%d bf16_words%zu differences%zu reuse%d grouped%d\n", cases, bank_capacity, checked, different, reuse, group_launch);
    else if (grouped) printf("RESIDENT_MOE_ROUTED cases%d bf16_words%zu differences%zu\n", cases, checked, different);
    else printf("RESIDENT_EXPERT_PROJECT cases%d groups%zu projections%zu bf16_words%zu ffn_words%zu differences%zu\n", cases, groups, projections, checked, ffn_words, different);
    fflush(stdout);
    assert(cases == (only_layer >= 0 ? 1 : 27) && !different);
    return 0;
}
