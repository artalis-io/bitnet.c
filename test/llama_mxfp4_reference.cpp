/* Independent dense MXFP4 CUDA oracle. Link against the pinned llama.cpp GGML
 * libraries, not bitnet's backend. Example (from the repository root):
 * g++ -O2 -std=c++17 -I/path/to/llama.cpp/ggml/include \
 *   test/llama_mxfp4_reference.cpp -L/path/to/llama.cpp/build/bin \
 *   -Wl,-rpath,/path/to/llama.cpp/build/bin -lggml -lggml-base -lggml-cuda \
 *   -o /tmp/llama_mxfp4_reference
 * CUDA_VISIBLE_DEVICES=1 /tmp/llama_mxfp4_reference
 * Baseline: 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120, 188 SMs.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "cuda_mxfp4_fixture.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend);
    int cases = 0;
    const int shapes[][2] = {{1,32}, {17,96}, {1024,3072}, {3072,1024},
                            {4,32}, {20,96}, {128,544}, {132,1056}};
    for (int group = 0; group < 3; group++) {
        for (int seed : {0, 23}) {
            for (int shape = 0; shape < (group == 2 ? 2 : 4); shape++) {
                int index = group == 0 ? shape :
                    group == 1 ? (shape < 2 ? shape + 4 : shape) : shape + 6;
                int rows = shapes[index][0], cols = shapes[index][1];
                int fixture_seed = seed + (group ? 100 : 0);
                for (int tokens : {1,2,4,8,9,16,17,33}) {
                    std::vector<uint8_t> weights((size_t)rows * cols / 32 * 17);
                    std::vector<float> input((size_t)cols * tokens), output((size_t)rows * tokens);
                    mxfp4_fixture_weights(weights.data(), (size_t)rows * cols, fixture_seed);
                    mxfp4_fixture_input(input.data(), input.size(), fixture_seed);
                    ggml_context *ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
                    assert(ctx);
                    ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, cols, rows);
                    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
                    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
                    ggml_cgraph *graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, y);
                    ggml_backend_buffer_t memory = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    assert(memory && ggml_nbytes(w) == weights.size());
                    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
                    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
                    uint64_t hash = UINT64_C(14695981039346656037);
                    for (float value : output) {
                        assert(std::isfinite(value));
                        uint32_t word;
                        memcpy(&word, &value, sizeof(word));
                        hash = (hash ^ word) * UINT64_C(1099511628211);
                    }
                    printf("    {BN_GGUF_TENSOR_MXFP4, %d, %d, %d, %d, UINT64_C(0x%016llx)},\n",
                        rows, cols, tokens, fixture_seed, (unsigned long long)hash);
                    fflush(stdout);
                    cases++;
                    ggml_backend_buffer_free(memory);
                    ggml_free(ctx);
                }
            }
        }
    }
    assert(cases == 160);
    ggml_backend_free(backend);
}
