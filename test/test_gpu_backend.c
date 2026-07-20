#include "gpu_backend.h"
#include "gpu_policy.h"
#include "backend_quant.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "quant.h"
#include "transformer_cpu_backend_internal.h"
#include "model.h"
#include "model_arch.h"
#include "gguf.h"
#include "sh_arena.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#if defined(BN_FORCE_SCALAR)
#define TEST_EXPECT_NATIVE_QUANT_ACTIVATION 0
#elif (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || \
    defined(__AVX2__) || defined(__wasm_relaxed_simd__)
#define TEST_EXPECT_NATIVE_QUANT_ACTIVATION 1
#else
#define TEST_EXPECT_NATIVE_QUANT_ACTIVATION 0
#endif

// --- Mock GPU backend ---
// Copies data on buffer_create, uses CPU scalar matvec through the vtable.

static void *mock_create(void *ctx, const void *data, size_t size,
                         int type, int rows, int cols) {
    (void)ctx; (void)type; (void)rows; (void)cols;
    void *copy = malloc(size);
    if (copy) memcpy(copy, data, size);
    return copy;
}

static void mock_destroy(void *ctx, void *buffer) {
    (void)ctx;
    free(buffer);
}

static int mock_matvec(void *ctx, float *out, void *W_buf, const float *x,
                       int rows, int cols, int type) {
    (void)ctx;
    BnQWeight W = {0};
    W.data = W_buf;
    W.type = type;
    W.rows = rows;
    W.cols = cols;
    W.scale = 1.0f;
    int max_dim = cols > rows ? cols : rows;
    int8_t *scratch = calloc(max_dim, 1);
    bn_quant_matvec(out, &W, x, scratch, NULL);
    free(scratch);
    return 0;
}

static BnGPUBackend mock_gpu = {
    .buffer_create = mock_create,
    .buffer_destroy = mock_destroy,
    .matvec = mock_matvec,
    .matmul = NULL,
    .ctx = NULL,
};

typedef struct {
    int destroys;
} DestroyCounter;

static void *mock_counted_create(void *ctx, const void *data, size_t size,
                                 int type, int rows, int cols) {
    (void)ctx; (void)type; (void)rows; (void)cols;
    void *copy = malloc(size);
    if (copy && data) memcpy(copy, data, size);
    return copy;
}

static void mock_counted_destroy(void *ctx, void *buffer) {
    DestroyCounter *counter = (DestroyCounter *)ctx;
    counter->destroys++;
    free(buffer);
}

// --- Helper: create I2_S weight data ---
// I2_S: 4 values per byte (2-bit ternary), all +1 encoding (value 2).
// Layout: interleaved bytes, per-tensor scale at end.
static uint8_t *make_i2s_data(int rows, int cols, float scale) {
    size_t nelements = (size_t)rows * cols;
    size_t data_size = nelements / 4 + 4;
    uint8_t *data = calloc(1, data_size);
    // Encode all +1: each 2-bit value = 2, so byte = 0xAA (10 10 10 10)
    for (size_t i = 0; i < nelements / 4; i++)
        data[i] = 0xAA;
    // Per-tensor scale stored as float at offset nelements/4
    memcpy(data + nelements / 4, &scale, sizeof(float));
    return data;
}

// --- Test 1: GPU upload weights ---
static void test_gpu_upload_weights(void) {
    printf("test_gpu_upload_weights... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    model.config.n_layers = 1;
    model.config.dim = 128;
    model.config.hidden_dim = 256;
    assert(bn_model_backend(&model) == NULL);
    assert(bn_model_gpu(&model) == NULL);

    model.weights.layers = calloc(1, sizeof(BnLayerWeights));
    assert(model.weights.layers);
    float attn_norm[128];
    float ffn_norm[128];
    for (int i = 0; i < 128; i++) {
        attn_norm[i] = 1.0f;
        ffn_norm[i] = 1.0f;
    }
    model.weights.layers[0].norm.attn_norm = attn_norm;
    model.weights.layers[0].norm.ffn_norm = ffn_norm;

    // Create a simple I2_S weight for wq
    float scale = 1.0f;
    uint8_t *wq_data = make_i2s_data(128, 128, scale);
    model.weights.layers[0].attn.wq.data = wq_data;
    model.weights.layers[0].attn.wq.type = BN_GGUF_TENSOR_I2_S;
    model.weights.layers[0].attn.wq.rows = 128;
    model.weights.layers[0].attn.wq.cols = 128;
    model.weights.layers[0].attn.wq.scale = scale;

    // Create ffn_up weight
    uint8_t *up_data = make_i2s_data(256, 128, scale);
    model.weights.layers[0].ffn.ffn_up.data = up_data;
    model.weights.layers[0].ffn.ffn_up.type = BN_GGUF_TENSOR_I2_S;
    model.weights.layers[0].ffn.ffn_up.rows = 256;
    model.weights.layers[0].ffn.ffn_up.cols = 128;
    model.weights.layers[0].ffn.ffn_up.scale = scale;

    const void *wq_cpu_data = model.weights.layers[0].attn.wq.data;
    const void *up_cpu_data = model.weights.layers[0].ffn.ffn_up.data;
    int wq_type = model.weights.layers[0].attn.wq.type;
    int up_type = model.weights.layers[0].ffn.ffn_up.type;
    int wq_rows = model.weights.layers[0].attn.wq.rows;
    int wq_cols = model.weights.layers[0].attn.wq.cols;
    int up_rows = model.weights.layers[0].ffn.ffn_up.rows;
    int up_cols = model.weights.layers[0].ffn.ffn_up.cols;

    int rc = bn_model_upload_weights(&model, &mock_gpu);
    assert(rc == 0);
    assert(bn_model_gpu(&model) == &mock_gpu);
    assert(bn_model_backend(&model) != NULL);
    assert(model.weights.layers[0].attn.wq.data == wq_cpu_data);
    assert(model.weights.layers[0].ffn.ffn_up.data == up_cpu_data);
    assert(model.weights.layers[0].attn.wq.type == wq_type);
    assert(model.weights.layers[0].ffn.ffn_up.type == up_type);
    assert(model.weights.layers[0].attn.wq.rows == wq_rows);
    assert(model.weights.layers[0].attn.wq.cols == wq_cols);
    assert(model.weights.layers[0].ffn.ffn_up.rows == up_rows);
    assert(model.weights.layers[0].ffn.ffn_up.cols == up_cols);
    bn_model_set_gpu_disabled(&model, 1);
    assert(bn_model_gpu(&model) == NULL);
    bn_model_set_gpu_disabled(&model, 0);
    assert(bn_model_gpu(&model) == &mock_gpu);
    assert(bn_backend_model_handle(bn_model_backend(&model), 0,
                                   BN_BACKEND_HANDLE_ATTN_NORM) != NULL);
    assert(bn_backend_model_handle(bn_model_backend(&model), 0,
                                   BN_BACKEND_HANDLE_FFN_NORM) != NULL);
    assert(bn_backend_model_qweight_buf(bn_model_backend(&model),
                                        &model.weights.layers[0].attn.wq) != NULL);
    assert(bn_backend_model_qweight_buf(bn_model_backend(&model),
                                        &model.weights.layers[0].ffn.ffn_up) != NULL);
    assert(bn_backend_model_handle(bn_model_backend(&model), 0,
                                   BN_BACKEND_HANDLE_QKV_STACKED) == NULL);
    assert(bn_backend_model_qweight_buf(bn_model_backend(&model),
                                        &model.weights.layers[0].attn.wk) == NULL);

    bn_model_release_gpu(&model);
    assert(bn_model_backend(&model) != NULL);
    assert(bn_model_gpu(&model) == NULL);
    bn_model_free(&model);
    free(wq_data);
    free(up_data);

    printf("PASSED\n");
}

// --- Test 2: GPU matvec ---
static void test_gpu_matvec(void) {
    printf("test_gpu_matvec... ");

    // Create I2_S weight: 1 row x 128 cols, all +1
    float scale = 1.0f;
    uint8_t *data = make_i2s_data(1, 128, scale);
    BnQWeight W = {0};
    W.data = data;
    W.type = BN_GGUF_TENSOR_I2_S;
    W.rows = 1;
    W.cols = 128;
    W.scale = scale;

    // Upload to mock GPU
    size_t sz = bn_qweight_data_size(&W);
    assert(sz > 0);
    void *W_buf = mock_gpu.buffer_create(mock_gpu.ctx, W.data, sz,
                                         W.type, W.rows, W.cols);
    assert(W_buf != NULL);

    // Input: all 1.0
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = 1.0f;

    // GPU path
    float out_gpu = 0;
    int8_t scratch[128];
    bn_backend_quant_matvec_gpu(&out_gpu, &W, x, scratch, NULL, &mock_gpu);
    float out_gpu_buf = 0;
    bn_backend_quant_matvec_gpu_buf(&out_gpu_buf, &W, W_buf, x, scratch, NULL,
                            &mock_gpu);

    // CPU path
    float out_cpu = 0;
    bn_backend_quant_matvec_gpu(&out_cpu, &W, x, scratch, NULL, &mock_gpu);

    assert(fabsf(out_gpu - out_cpu) < 1e-3f);
    assert(fabsf(out_gpu_buf - out_cpu) < 1e-3f);

    mock_gpu.buffer_destroy(mock_gpu.ctx, W_buf);
    free(data);

    printf("PASSED\n");
}

// --- Test 3: GPU fallback ---
static void test_gpu_fallback(void) {
    printf("test_gpu_fallback... ");

    float scale = 1.0f;
    uint8_t *data = make_i2s_data(1, 128, scale);
    BnQWeight W = {0};
    W.data = data;
    W.type = BN_GGUF_TENSOR_I2_S;
    W.rows = 1;
    W.cols = 128;
    W.scale = scale;
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = 1.0f;

    float out = 0;
    int8_t scratch[128];
    bn_backend_quant_matvec_gpu(&out, &W, x, scratch, NULL, &mock_gpu);

    // All +1 weights dot all 1.0 inputs = 128 * scale
    assert(fabsf(out - 128.0f) < 1.0f);

    free(data);
    printf("PASSED\n");
}

// --- Test 4: GPU release ---
static void test_gpu_release(void) {
    printf("test_gpu_release... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    model.config.n_layers = 1;

    model.weights.layers = calloc(1, sizeof(BnLayerWeights));
    assert(model.weights.layers);

    float scale = 1.0f;
    uint8_t *data = make_i2s_data(128, 128, scale);
    model.weights.layers[0].attn.wq.data = data;
    model.weights.layers[0].attn.wq.type = BN_GGUF_TENSOR_I2_S;
    model.weights.layers[0].attn.wq.rows = 128;
    model.weights.layers[0].attn.wq.cols = 128;
    model.weights.layers[0].attn.wq.scale = scale;

    int rc = bn_model_upload_weights(&model, &mock_gpu);
    assert(rc == 0);
    assert(bn_backend_model_qweight_buf(bn_model_backend(&model),
                                        &model.weights.layers[0].attn.wq) != NULL);

    bn_model_release_gpu(&model);
    assert(bn_model_gpu(&model) == NULL);
    assert(bn_model_backend(&model) != NULL);

    // Safe to call again
    bn_model_release_gpu(&model);
    bn_model_free(&model);
    free(data);

    printf("PASSED\n");
}

static void test_backend_model_release_owns_buffers(void) {
    printf("test_backend_model_release_owns_buffers... ");

    DestroyCounter counter = {0};
    BnGPUBackend gpu = {0};
    gpu.buffer_create = mock_counted_create;
    gpu.buffer_destroy = mock_counted_destroy;
    gpu.ctx = &counter;

    uint8_t bytes[4] = {1, 2, 3, 4};
    void *shared = gpu.buffer_create(gpu.ctx, bytes, sizeof(bytes), -1, 1, 4);
    void *norm = gpu.buffer_create(gpu.ctx, bytes, sizeof(bytes), -1, 1, 4);
    assert(shared && norm);

    BnQWeight weight = {0};
    weight.data = bytes;
    weight.type = BN_GGUF_TENSOR_Q4_0;
    weight.rows = 1;
    weight.cols = 32;

    BnBackendModel *backend = bn_backend_model_create();
    assert(backend != NULL);
    bn_backend_model_bind_gpu(backend, &gpu);
    assert(bn_backend_model_register_qweight(backend, &weight, shared) == 0);
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_QKV_STACKED,
                                            shared) == 0);
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_ATTN_NORM,
                                            norm) == 0);

    bn_backend_model_release_gpu(backend);
    assert(counter.destroys == 2);
    assert(bn_backend_model_raw_gpu(backend) == NULL);
    assert(bn_backend_model_qweight_buf(backend, &weight) == NULL);
    assert(bn_backend_model_handle(backend, 0,
                                   BN_BACKEND_HANDLE_ATTN_NORM) == NULL);

    bn_backend_model_free(backend);
    assert(counter.destroys == 2);

    printf("PASSED\n");
}

static BnGGUFKeyValue test_make_u32_kv(char *key, uint32_t value) {
    BnGGUFKeyValue kv = {0};
    kv.key = key;
    kv.type = BN_GGUF_TYPE_UINT32;
    kv.value.u32 = value;
    return kv;
}

static BnGGUFKeyValue test_make_u32_array_kv(char *key, uint32_t *values,
                                             uint64_t count) {
    BnGGUFKeyValue kv = {0};
    kv.key = key;
    kv.type = BN_GGUF_TYPE_ARRAY;
    kv.value.arr.elem_type = BN_GGUF_TYPE_UINT32;
    kv.value.arr.n = count;
    kv.value.arr.data = values;
    return kv;
}

static BnGGUFKeyValue test_make_bool_array_kv(char *key, uint8_t *values,
                                              uint64_t count) {
    BnGGUFKeyValue kv = {0};
    kv.key = key;
    kv.type = BN_GGUF_TYPE_ARRAY;
    kv.value.arr.elem_type = BN_GGUF_TYPE_BOOL;
    kv.value.arr.n = count;
    kv.value.arr.data = values;
    return kv;
}

static BnGGUFKeyValue test_make_f32_kv(char *key, float value) {
    BnGGUFKeyValue kv = {0};
    kv.key = key;
    kv.type = BN_GGUF_TYPE_FLOAT32;
    kv.value.f32 = value;
    return kv;
}

static BnGGUFKeyValue test_make_str_kv(char *key, char *value) {
    BnGGUFKeyValue kv = {0};
    kv.key = key;
    kv.type = BN_GGUF_TYPE_STRING;
    kv.value.str.str = value;
    kv.value.str.len = strlen(value);
    return kv;
}

static void test_gpu_policy_helpers(void) {
    printf("test_gpu_policy_helpers... ");

    BnGPUBackend gpu = mock_gpu;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.buffer_create_quant_only = mock_create;
    gpu.buffer_create_f16_cache = mock_create;
    gpu.buffer_create_kquant_f32_cache = mock_create;

    BnConfig layers = {0};
    assert(bn_gpu_policy_attention_layer_count(NULL) == 0);
    assert(bn_gpu_policy_ssm_layer_count(NULL) == 0);
    assert(!bn_gpu_policy_uses_hybrid_ssm(NULL));
    assert(!bn_gpu_policy_uses_hybrid_moe(NULL));
    layers.n_layers = 12;
    assert(bn_gpu_policy_attention_layer_count(&layers) == 12);
    assert(bn_gpu_policy_ssm_layer_count(&layers) == 0);
    assert(!bn_gpu_policy_uses_hybrid_ssm(&layers));
    assert(!bn_gpu_policy_uses_hybrid_moe(&layers));
    layers.full_attn_interval = 4;
    assert(bn_gpu_policy_attention_layer_count(&layers) == 3);
    assert(bn_gpu_policy_ssm_layer_count(&layers) == 9);
    assert(!bn_gpu_policy_uses_hybrid_ssm(&layers));
    assert(!bn_gpu_policy_uses_hybrid_moe(&layers));
    layers.ssm_inner_size = 128;
    assert(bn_gpu_policy_uses_hybrid_ssm(&layers));
    assert(!bn_gpu_policy_uses_moe(&layers));
    BnModel dense_model = {0};
    dense_model.config = layers;
    assert(!bn_model_uses_moe(&dense_model));
    assert(!bn_gpu_policy_moe_router_diff2_upload_enabled(&layers));
    assert(!bn_gpu_policy_moe_f16_aux_cache_auto_enabled(&layers));
    assert(bn_gpu_policy_moe_route_all_active_two(2, 2));
    assert(!bn_gpu_policy_moe_route_all_active_two(4, 2));
    assert(!bn_gpu_policy_moe_route_all_active_two(2, 1));
    assert(bn_gpu_policy_moe_route_expanded_topk(4, 2));
    assert(bn_gpu_policy_moe_route_expanded_topk(2, 4));
    assert(!bn_gpu_policy_moe_route_expanded_topk(2, 2));
    assert(bn_gpu_policy_moe_route_all_active_two_large_hidden(2, 2,
                                                               4096));
    assert(!bn_gpu_policy_moe_route_all_active_two_large_hidden(2, 2,
                                                                2048));
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE");
    assert(bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
        1, 1, BN_GGUF_TENSOR_Q6_K, 4096, 2, 2, 1, 1, 1));
    setenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
        1, 1, BN_GGUF_TENSOR_Q6_K, 4096, 2, 2, 1, 1, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE");
    assert(!bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
        2, 1, BN_GGUF_TENSOR_Q6_K, 4096, 2, 2, 1, 1, 1));
    assert(!bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
        1, 1, BN_GGUF_TENSOR_Q6_K, 2048, 2, 2, 1, 1, 1));
    assert(!bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
        1, 1, BN_GGUF_TENSOR_Q6_K, 4096, 4, 2, 1, 1, 1));
    assert(!bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
        1, 1, BN_GGUF_TENSOR_Q6_K, 4096, 2, 2, 1, 0, 1));

    BnConfig moe = {0};
    moe.dim = 2048;
    moe.hidden_dim = 512;
    moe.n_experts = 2;
    moe.n_experts_active = 2;
    moe.moe_intermediate_size = 4096;
    assert(bn_gpu_policy_uses_moe(&moe));
    BnModel moe_model = {0};
    moe_model.config = moe;
    assert(bn_model_uses_moe(&moe_model));
    assert(!bn_gpu_policy_uses_hybrid_moe(&moe));
    assert(bn_gpu_policy_moe_router_diff2_upload_enabled(&moe));
    assert(bn_gpu_policy_moe_f16_aux_cache_auto_enabled(&moe));
    moe.full_attn_interval = 4;
    assert(bn_gpu_policy_uses_hybrid_moe(&moe));
    moe.full_attn_interval = 0;
    moe.dim = 2049;
    assert(!bn_gpu_policy_moe_router_diff2_upload_enabled(&moe));
    assert(bn_gpu_policy_moe_f16_aux_cache_auto_enabled(&moe));
    moe.n_experts = 4;
    moe.n_experts_active = 2;
    assert(!bn_gpu_policy_moe_router_diff2_upload_enabled(&moe));
    assert(bn_gpu_policy_moe_f16_aux_cache_auto_enabled(&moe));

    uint32_t dense_kv_heads[2] = {8, 4};
    uint8_t dense_sliding[2] = {1, 0};
    BnGGUFKeyValue dense_kvs[6];
    dense_kvs[0] = test_make_str_kv("general.architecture", "gemma4");
    dense_kvs[1] = test_make_u32_kv("gemma4.expert_count", 0);
    dense_kvs[2] = test_make_u32_kv("gemma4.context_length", 8192);
    dense_kvs[3] = test_make_f32_kv("gemma4.rope.freq_base", 1000000.0f);
    dense_kvs[4] = test_make_u32_array_kv(
        "gemma4.attention.head_count_kv", dense_kv_heads, 2);
    dense_kvs[5] = test_make_bool_array_kv(
        "gemma4.attention.sliding_window_pattern", dense_sliding, 2);
    BnGGUFFile dense_gf = {0};
    dense_gf.n_kv = 6;
    dense_gf.kvs = dense_kvs;
    assert(bn_model_arch_gguf_u32(&dense_gf, "context_length") == 8192);
    assert(bn_model_arch_gguf_arr_n(
        &dense_gf, "attention.head_count_kv") == 2);
    assert(bn_model_arch_gguf_u32_or_i32_array(
        &dense_gf, "attention.head_count_kv", 1) == 4);
    assert(bn_model_arch_gguf_bool_array(
        &dense_gf, "attention.sliding_window_pattern", 0));
    assert(!bn_model_arch_gguf_bool_array(
        &dense_gf, "attention.sliding_window_pattern", 1));
    assert(bn_model_arch_gguf_f32(&dense_gf, "rope.freq_base") ==
           1000000.0f);
    assert(bn_gpu_policy_auto_caps_gguf_sequence(
        1, 0, 0, &dense_gf, 4096));
    assert(bn_gpu_policy_auto_caps_gguf_sequence(
        0, 1, 0, &dense_gf, 4096));
    assert(!bn_gpu_policy_auto_caps_gguf_sequence(
        0, 0, 1, &dense_gf, 4096));

    BnGGUFKeyValue moe_kvs[3];
    moe_kvs[0] = test_make_str_kv("general.architecture", "qwen35moe");
    moe_kvs[1] = test_make_u32_kv("qwen35moe.expert_count", 4);
    moe_kvs[2] = test_make_u32_kv("qwen35moe.context_length", 8192);
    BnGGUFFile moe_gf = {0};
    moe_gf.n_kv = 3;
    moe_gf.kvs = moe_kvs;
    assert(bn_gpu_policy_auto_caps_gguf_sequence(
        0, 0, 1, &moe_gf, 4096));
    moe_kvs[2].value.u32 = 4096;
    assert(!bn_gpu_policy_auto_caps_gguf_sequence(
        0, 0, 1, &moe_gf, 4096));

    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN");
    assert(bn_gpu_policy_moe_resident_routed_ffn_enabled(1));
    assert(!bn_gpu_policy_moe_resident_routed_ffn_enabled(0));
    setenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN", "1", 1);
    assert(!bn_gpu_policy_moe_resident_routed_ffn_enabled(1));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN");
    assert(bn_gpu_policy_moe_resident_routed_ffn_quant_eligible(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K));
    assert(bn_gpu_policy_moe_resident_routed_ffn_quant_eligible(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0));
    assert(!bn_gpu_policy_moe_resident_routed_ffn_quant_eligible(
        BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32));

    unsetenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE");
    assert(!bn_gpu_policy_moe_all_f16_cache_forced());
    assert(!bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q8_0, 1));
    setenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_all_f16_cache_forced());
    assert(bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q4_0, 0));
    setenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q8_0, 1));
    unsetenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE");

    unsetenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE");
    assert(!bn_gpu_policy_moe_gateup_f16_cache_enabled(1));
    setenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_gateup_f16_cache_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_gateup_f16_cache_enabled(1));
    unsetenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE");

    unsetenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE");
    assert(!bn_gpu_policy_partial_moe_f16_cache_enabled(1));
    setenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_partial_moe_f16_cache_enabled(1));
    assert(!bn_gpu_policy_partial_moe_f16_cache_enabled(0));
    unsetenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE");

    unsetenv("BN_CUDA_DEBUG_MOE_FIT");
    unsetenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE");
    assert(!bn_gpu_policy_moe_residency_fit_debug_enabled());
    assert(bn_gpu_policy_individual_upload_quant_only_enabled(&gpu));
    setenv("BN_CUDA_DEBUG_MOE_FIT", "1", 1);
    setenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_residency_fit_debug_enabled());
    assert(!bn_gpu_policy_individual_upload_quant_only_enabled(&gpu));
    unsetenv("BN_CUDA_DEBUG_MOE_FIT");
    unsetenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_individual_upload_quant_only_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_CUDA;

    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");
    assert(!bn_gpu_policy_moe_lazy_aux_cache_enabled());
    setenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_lazy_aux_cache_enabled());
    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");

    unsetenv("BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM");
    unsetenv("BN_CUDA_DEBUG_PREFILL_GEMM");
    assert(bn_gpu_policy_cuda_prefill_batched_gemm_enabled());
    assert(!bn_gpu_policy_cuda_prefill_gemm_debug_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM", "1", 1);
    setenv("BN_CUDA_DEBUG_PREFILL_GEMM", "1", 1);
    assert(!bn_gpu_policy_cuda_prefill_batched_gemm_enabled());
    assert(bn_gpu_policy_cuda_prefill_gemm_debug_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM");
    unsetenv("BN_CUDA_DEBUG_PREFILL_GEMM");

    unsetenv("BN_CUDA_CUBLAS_GEMM_ALGO");
    assert(bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(24) == 24);
    setenv("BN_CUDA_CUBLAS_GEMM_ALGO", "-1", 1);
    assert(bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(24) == -1);
    setenv("BN_CUDA_CUBLAS_GEMM_ALGO", "0", 1);
    assert(bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(24) == 0);
    setenv("BN_CUDA_CUBLAS_GEMM_ALGO", "23", 1);
    assert(bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(24) == 23);
    setenv("BN_CUDA_CUBLAS_GEMM_ALGO", "24", 1);
    assert(bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(24) == 24);
    setenv("BN_CUDA_CUBLAS_GEMM_ALGO", "not-an-int", 1);
    assert(bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(24) == 24);
    unsetenv("BN_CUDA_CUBLAS_GEMM_ALGO");

    unsetenv("BN_CUDA_DEBUG_READBACK");
    assert(!bn_gpu_policy_cuda_readback_debug_enabled());
    setenv("BN_CUDA_DEBUG_READBACK", "1", 1);
    assert(bn_gpu_policy_cuda_readback_debug_enabled());
    unsetenv("BN_CUDA_DEBUG_READBACK");

    unsetenv("BN_CUDA_DEBUG_CUBLAS_CACHE");
    assert(!bn_gpu_policy_cuda_cublas_cache_debug_enabled());
    assert(bn_gpu_policy_cuda_cublas_cache_reserve_mb_or_default(4096) ==
           4096);
    assert(bn_gpu_policy_cuda_cublas_workspace_mb_or_default(32) == 32);
    setenv("BN_CUDA_DEBUG_CUBLAS_CACHE", "1", 1);
    setenv("BN_CUDA_CUBLAS_CACHE_RESERVE_MB", "128", 1);
    setenv("BN_CUDA_CUBLAS_WORKSPACE_MB", "64", 1);
    assert(bn_gpu_policy_cuda_cublas_cache_debug_enabled());
    assert(bn_gpu_policy_cuda_cublas_cache_reserve_mb_or_default(4096) ==
           128);
    assert(bn_gpu_policy_cuda_cublas_workspace_mb_or_default(32) == 64);
    unsetenv("BN_CUDA_DEBUG_CUBLAS_CACHE");
    unsetenv("BN_CUDA_CUBLAS_CACHE_RESERVE_MB");
    unsetenv("BN_CUDA_CUBLAS_WORKSPACE_MB");

    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP_F16_OUT");
    assert(bn_gpu_policy_cuda_moe_cublas_gateup_f16_out_enabled());
    setenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP_F16_OUT", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_gateup_f16_out_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP_F16_OUT");

    unsetenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_VARIABLE");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED_VARIABLE");
    assert(!bn_gpu_policy_cuda_moe_cublas_grouped_variable_enabled());
    setenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_VARIABLE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_cublas_grouped_variable_enabled());
    setenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED_VARIABLE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_grouped_variable_enabled());
    unsetenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_VARIABLE");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED_VARIABLE");

    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GROUPED");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GROUPED");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED");
    unsetenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_SMALL");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GATEUP");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GATEUP");
    unsetenv("BN_CUDA_ENABLE_MOE_CUBLAS_GATEUP");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_ALL2_FIXED");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_SORT");
    assert(bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        1, 0, 1, 1, 1, 4, 2, 512));
    assert(!bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        1, 0, 1, 1, 0, 4, 2, 512));
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GROUPED", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        1, 0, 1, 1, 1, 4, 2, 512));
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GROUPED");
    setenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GROUPED", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        1, 0, 1, 1, 1, 4, 2, 512));
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GROUPED");
    assert(!bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        0, 1, 1, 1, 1, 4, 2, 128));
    setenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_SMALL", "1", 1);
    assert(bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        0, 1, 1, 1, 1, 4, 2, 128));
    unsetenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_SMALL");
    assert(bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        0, 1, 1, 1, 1, 2, 2, 4));
    setenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
        0, 1, 1, 1, 1, 2, 2, 4));
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED");
    assert(bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
        0, 1, 0, 1, 1, 0, 2));
    assert(!bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
        1, 1, 0, 1, 1, 0, 2));
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GATEUP", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
        0, 1, 0, 1, 1, 0, 2));
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CUBLAS_GATEUP");
    setenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GATEUP", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
        0, 1, 0, 1, 1, 0, 2));
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GATEUP");
    assert(!bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
        0, 0, 1, 1, 1, 0, 2));
    setenv("BN_CUDA_ENABLE_MOE_CUBLAS_GATEUP", "1", 1);
    assert(bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
        0, 0, 1, 1, 1, 0, 2));
    setenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
        0, 0, 1, 1, 1, 0, 2));
    unsetenv("BN_CUDA_ENABLE_MOE_CUBLAS_GATEUP");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP");
    assert(bn_gpu_policy_cuda_moe_cublas_all_active_two_fixed_enabled(1, 2, 2));
    assert(!bn_gpu_policy_cuda_moe_cublas_all_active_two_fixed_enabled(1, 4, 2));
    setenv("BN_CUDA_DISABLE_MOE_CUBLAS_ALL2_FIXED", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_cublas_all_active_two_fixed_enabled(1, 2, 2));
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_ALL2_FIXED");
    assert(bn_gpu_policy_cuda_moe_sorted_slots_enabled(1, 0, 2, 0, 1, 0));
    assert(!bn_gpu_policy_cuda_moe_sorted_slots_enabled(1, 0, 1, 0, 1, 0));
    assert(!bn_gpu_policy_cuda_moe_sorted_slots_enabled(1, 0, 2, 1, 1, 0));
    setenv("BN_CUDA_ENABLE_MOE_ROUTE_SORT", "1", 1);
    assert(bn_gpu_policy_cuda_moe_sorted_slots_enabled(0, 1, 2, 0, 0, 0));
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_SORT");

    unsetenv("BN_CUDA_PROFILE_MOE_PREFILL_INTERNAL");
    unsetenv("BN_CUDA_PROFILE_MOE_PREFILL_EVERY");
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_DIRECT_RESID_OUT");
    unsetenv("BN_CUDA_ENABLE_MOE_BATCH_FUSED_ROUTE_TOPK");
    unsetenv("BN_CUDA_DISABLE_MOE_BATCH_FUSED_ROUTE_TOPK");
    assert(!bn_gpu_policy_cuda_moe_prefill_internal_profile_enabled());
    assert(bn_gpu_policy_cuda_moe_prefill_profile_every_or_default(48) ==
           48);
    setenv("BN_CUDA_PROFILE_MOE_PREFILL_INTERNAL", "1", 1);
    setenv("BN_CUDA_PROFILE_MOE_PREFILL_EVERY", "9", 1);
    assert(bn_gpu_policy_cuda_moe_prefill_internal_profile_enabled());
    assert(bn_gpu_policy_cuda_moe_prefill_profile_every_or_default(48) ==
           9);
    setenv("BN_CUDA_PROFILE_MOE_PREFILL_EVERY", "0", 1);
    assert(bn_gpu_policy_cuda_moe_prefill_profile_every_or_default(48) ==
           48);
    unsetenv("BN_CUDA_PROFILE_MOE_PREFILL_INTERNAL");
    unsetenv("BN_CUDA_PROFILE_MOE_PREFILL_EVERY");
    assert(bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
        1, 0, 0, 1));
    assert(!bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
        0, 0, 0, 1));
    assert(!bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
        1, 1, 0, 1));
    assert(!bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
        1, 0, 1, 1));
    assert(!bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
        1, 0, 0, 0));
    setenv("BN_CUDA_DISABLE_MOE_PREFILL_DIRECT_RESID_OUT", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
        1, 0, 0, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_DIRECT_RESID_OUT");
    assert(!bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(256));
    setenv("BN_CUDA_ENABLE_MOE_BATCH_FUSED_ROUTE_TOPK", "1", 1);
    assert(bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(256));
    assert(!bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(257));
    setenv("BN_CUDA_DISABLE_MOE_BATCH_FUSED_ROUTE_TOPK", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(256));
    unsetenv("BN_CUDA_ENABLE_MOE_BATCH_FUSED_ROUTE_TOPK");
    unsetenv("BN_CUDA_DISABLE_MOE_BATCH_FUSED_ROUTE_TOPK");

    unsetenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST");
    unsetenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY");
    unsetenv("BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED");
    unsetenv("BN_CUDA_DEBUG_MOE_CUBLAS_GATEUP");
    assert(!bn_gpu_policy_cuda_moe_route_dist_profile_enabled());
    assert(bn_gpu_policy_cuda_moe_route_dist_profile_every_or_default(48) ==
           48);
    setenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST", "1", 1);
    setenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY", "7", 1);
    assert(bn_gpu_policy_cuda_moe_route_dist_profile_enabled());
    assert(bn_gpu_policy_cuda_moe_route_dist_profile_every_or_default(48) ==
           7);
    setenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY", "0", 1);
    assert(bn_gpu_policy_cuda_moe_route_dist_profile_every_or_default(48) ==
           48);
    setenv("BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED", "1", 1);
    setenv("BN_CUDA_DEBUG_MOE_CUBLAS_GATEUP", "1", 1);
    assert(bn_gpu_policy_cuda_moe_cublas_grouped_debug_enabled());
    assert(bn_gpu_policy_cuda_moe_cublas_gateup_debug_enabled());
    unsetenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST");
    unsetenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY");
    unsetenv("BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED");
    unsetenv("BN_CUDA_DEBUG_MOE_CUBLAS_GATEUP");

    unsetenv("BN_CUDA_DISABLE_MOE_FFN_BATCH");
    assert(bn_gpu_policy_cuda_moe_ffn_batch_enabled());
    setenv("BN_CUDA_DISABLE_MOE_FFN_BATCH", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_ffn_batch_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_FFN_BATCH");

    unsetenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_INTERNAL");
    unsetenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_EVERY");
    assert(!bn_gpu_policy_cuda_moe_ffn_batch_profile_enabled());
    assert(bn_gpu_policy_cuda_moe_ffn_batch_profile_every_or_default(24) ==
           24);
    setenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_INTERNAL", "1", 1);
    setenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_EVERY", "5", 1);
    assert(bn_gpu_policy_cuda_moe_ffn_batch_profile_enabled());
    assert(bn_gpu_policy_cuda_moe_ffn_batch_profile_every_or_default(24) ==
           5);
    setenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_EVERY", "-1", 1);
    assert(bn_gpu_policy_cuda_moe_ffn_batch_profile_every_or_default(24) ==
           24);
    unsetenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_INTERNAL");
    unsetenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_EVERY");

    unsetenv("BN_CUDA_ENABLE_DENSE_FFN");
    assert(!bn_gpu_policy_cuda_dense_ffn_enabled());
    setenv("BN_CUDA_ENABLE_DENSE_FFN", "1", 1);
    assert(bn_gpu_policy_cuda_dense_ffn_enabled());
    unsetenv("BN_CUDA_ENABLE_DENSE_FFN");

    unsetenv("BN_CUDA_DISABLE_DENSE_FFN_BATCH");
    assert(bn_gpu_policy_cuda_dense_ffn_batch_enabled());
    setenv("BN_CUDA_DISABLE_DENSE_FFN_BATCH", "1", 1);
    assert(!bn_gpu_policy_cuda_dense_ffn_batch_enabled());
    unsetenv("BN_CUDA_DISABLE_DENSE_FFN_BATCH");

    unsetenv("BN_CUDA_ENABLE_LOGITS_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_LOGITS_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_CUBLAS_LOGITS");
    unsetenv("BN_CUDA_ENABLE_F32_LOGITS_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F32_LOGITS_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_LOGITS_MATVEC");
    assert(!bn_gpu_policy_logits_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    assert(!bn_gpu_policy_cuda_cublas_logits_enabled());
    assert(!bn_gpu_policy_cuda_f32_logits_matvec_enabled());
    assert(!bn_gpu_policy_cuda_f16_logits_matvec_enabled());
    setenv("BN_CUDA_ENABLE_LOGITS_KQUANT_F32_CACHE", "1", 1);
    setenv("BN_CUDA_ENABLE_CUBLAS_LOGITS", "1", 1);
    setenv("BN_CUDA_ENABLE_F32_LOGITS_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_F16_LOGITS_MATVEC", "1", 1);
    assert(bn_gpu_policy_logits_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    unsetenv("BN_CUDA_ENABLE_LOGITS_KQUANT_F32_CACHE");
    setenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_logits_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    assert(bn_gpu_policy_cuda_cublas_logits_enabled());
    assert(bn_gpu_policy_cuda_f32_logits_matvec_enabled());
    assert(bn_gpu_policy_cuda_f16_logits_matvec_enabled());
    assert(!bn_gpu_policy_logits_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_logits_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    setenv("BN_CUDA_DISABLE_F32_LOGITS_MATVEC", "1", 1);
    assert(!bn_gpu_policy_cuda_f32_logits_matvec_enabled());
    setenv("BN_CUDA_DISABLE_LOGITS_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_logits_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    unsetenv("BN_CUDA_DISABLE_LOGITS_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_logits_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    unsetenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_LOGITS_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_CUBLAS_LOGITS");
    unsetenv("BN_CUDA_ENABLE_F32_LOGITS_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F32_LOGITS_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_LOGITS_MATVEC");

    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    assert(!bn_gpu_policy_backend_kquant_logits_refine_enabled(&gpu, 0));
    assert(bn_gpu_policy_backend_kquant_logits_refine_enabled(&gpu, 1));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_gpu_policy_backend_kquant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;
    assert(bn_gpu_policy_backend_kquant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_gpu_policy_kquant_logits_refine_top_or_default(64) == 64);
    setenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_backend_kquant_logits_refine_enabled(&gpu, 0));
    setenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE", "1", 1);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_gpu_policy_backend_kquant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_backend_kquant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    setenv("BN_GPU_Q6_Q8K_REFINE_TOP", "11", 1);
    assert(bn_gpu_policy_kquant_logits_refine_top_or_default(64) == 11);
    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");

    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_Q8_REFINE_TOP");
    assert(!bn_gpu_policy_backend_native_quant_logits_refine_enabled(&gpu, 0));
    assert(bn_gpu_policy_backend_native_quant_logits_refine_enabled(&gpu, 1));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_gpu_policy_backend_native_quant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;
    assert(bn_gpu_policy_backend_native_quant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_gpu_policy_native_quant_logits_refine_top_or_default(16) == 16);
    setenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_backend_native_quant_logits_refine_enabled(&gpu, 0));
    setenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE", "1", 1);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_gpu_policy_backend_native_quant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_backend_native_quant_logits_refine_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    setenv("BN_GPU_Q8_REFINE_TOP", "5", 1);
    assert(bn_gpu_policy_native_quant_logits_refine_top_or_default(16) == 5);
    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_Q8_REFINE_TOP");

    unsetenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE");
    assert(!bn_gpu_policy_logits_f16_cache_enabled(&gpu));
    setenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_logits_f16_cache_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_logits_f16_cache_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE");

    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_enabled(&gpu));
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_forced());
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_default_for_cols(
        1024));
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_default_for_cols(1025));
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 1024, 0));
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 1025, 0));
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q4_K, 2048, 0));
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 2048, 1));
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 2048, 2) ==
           32768 * sizeof(float));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_forced());
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_default_for_cols(1));
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_requires_full_buffer(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_requires_full_buffer(
        BN_GGUF_TENSOR_Q4_K));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    setenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_down_kquant_f32_cache_forced());
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_enabled(&gpu));
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 2048, 0));
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 2048, 2));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_down_kquant_f32_cache_enabled(&gpu));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");

    assert(bn_gpu_policy_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_gpu_policy_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q8_0, 1));
    assert(bn_gpu_policy_moe_prefers_quant_only(
        &gpu, BN_GGUF_TENSOR_Q8_0));
    assert(!bn_gpu_policy_moe_prefers_quant_only(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_moe_prefers_quant_only(
        &gpu, BN_GGUF_TENSOR_Q8_0));
    gpu.kind = BN_GPU_BACKEND_CUDA;

    unsetenv("BN_CUDA_LAYOUT_RESERVE_MB");
    unsetenv("BN_CUDA_MOE_FULL_RESERVE_MB");
    assert(bn_gpu_policy_layout_reserve_bytes() ==
           512u * 1024u * 1024u);
    assert(bn_gpu_policy_moe_full_reserve_bytes() ==
           512u * 1024u * 1024u);
    setenv("BN_CUDA_LAYOUT_RESERVE_MB", "7", 1);
    setenv("BN_CUDA_MOE_FULL_RESERVE_MB", "9", 1);
    assert(bn_gpu_policy_layout_reserve_bytes() ==
           7u * 1024u * 1024u);
    assert(bn_gpu_policy_moe_full_reserve_bytes() ==
           9u * 1024u * 1024u);
    setenv("BN_CUDA_LAYOUT_RESERVE_MB", "bad", 1);
    assert(bn_gpu_policy_layout_reserve_bytes() ==
           512u * 1024u * 1024u);
    setenv("BN_CUDA_MOE_FULL_RESERVE_MB",
           "18446744073709551615", 1);
    assert(bn_gpu_policy_moe_full_reserve_bytes() == SIZE_MAX);
    unsetenv("BN_CUDA_LAYOUT_RESERVE_MB");
    unsetenv("BN_CUDA_MOE_FULL_RESERVE_MB");

    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");
    assert(!bn_gpu_policy_moe_down_small_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_down_small_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    setenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_down_small_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_down_small_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_down_small_kquant_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");

    unsetenv("BN_CUDA_DISABLE_MATVEC");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0");
    unsetenv("BN_CUDA_DISABLE_Q5_0");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q4_K");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q5_K");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q6_K");
    unsetenv("BN_CUDA_DISABLE_PREPARED_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_K");
    assert(!bn_gpu_policy_matvec_disabled());
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q5_0));
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q4_K));
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q8_K));
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q4_0));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q8_0));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q5_0));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q4_K));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q5_K));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q6_K));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q8_K));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q4_0));
    assert(!bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_I2_S));
    setenv("BN_CUDA_DISABLE_Q4_K", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5_K", "1", 1);
    setenv("BN_CUDA_DISABLE_Q6_K", "1", 1);
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q4_K));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q5_K));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q6_K));
    unsetenv("BN_CUDA_DISABLE_Q4_K");
    unsetenv("BN_CUDA_DISABLE_Q5_K");
    unsetenv("BN_CUDA_DISABLE_Q6_K");
    setenv("BN_CUDA_DISABLE_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5_0", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_PREPARED_NATIVE_QUANT_MATVEC", "1", 1);
    assert(bn_gpu_policy_matvec_disabled());
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q8_0));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q5_0));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q4_K));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q5_K));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q6_K));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q8_K));
    assert(!bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q4_0));
    assert(!bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q5_0));
    assert(!bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q4_K));
    assert(!bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q8_K));
    assert(bn_gpu_policy_matvec_type_supported(BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CUDA_DISABLE_MATVEC");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0");
    unsetenv("BN_CUDA_DISABLE_Q5_0");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q4_K");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q5_K");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q6_K");
    unsetenv("BN_CUDA_DISABLE_PREPARED_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_K");
    setenv("BN_CUDA_DISABLE_Q8_0", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_K", "1", 1);
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q8_0));
    assert(bn_gpu_policy_matvec_type_disabled(BN_GGUF_TENSOR_Q8_K));
    unsetenv("BN_CUDA_DISABLE_Q8_0");
    unsetenv("BN_CUDA_DISABLE_Q8_K");

    unsetenv("BN_CUDA_DISABLE_MATMUL_BATCH");
    unsetenv("BN_CUDA_DISABLE_MATVEC_BATCH");
    assert(bn_gpu_policy_matmul_batch_enabled());
    assert(bn_gpu_policy_matvec_batch_enabled());
    setenv("BN_CUDA_DISABLE_MATMUL_BATCH", "1", 1);
    setenv("BN_CUDA_DISABLE_MATVEC_BATCH", "1", 1);
    assert(!bn_gpu_policy_matmul_batch_enabled());
    assert(!bn_gpu_policy_matvec_batch_enabled());
    unsetenv("BN_CUDA_DISABLE_MATMUL_BATCH");
    unsetenv("BN_CUDA_DISABLE_MATVEC_BATCH");

    unsetenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
    unsetenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");
    assert(bn_gpu_policy_small_kquant_native_enabled(0));
    assert(!bn_gpu_policy_small_kquant_native_enabled(1));
    assert(!bn_gpu_policy_small_kquant_native_disabled());
    setenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE", "1", 1);
    assert(bn_gpu_policy_small_kquant_native_enabled(1));
    setenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE", "1", 1);
    assert(bn_gpu_policy_small_kquant_native_enabled(1));
    assert(bn_gpu_policy_small_kquant_native_disabled());
    unsetenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
    assert(!bn_gpu_policy_small_kquant_native_enabled(0));
    unsetenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");

    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    unsetenv("BN_GPU_PREFILL_MATMUL");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");
    assert(!bn_gpu_policy_prefill_matmul_disabled());
    assert(!bn_gpu_policy_prefill_matmul_enabled());
    assert(!bn_gpu_policy_prefill_direct_kv_disabled());
    assert(!bn_gpu_policy_prefill_direct_kv_with_cpu_fallback_enabled());
    setenv("BN_GPU_DISABLE_PREFILL_MATMUL", "1", 1);
    setenv("BN_GPU_PREFILL_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV", "1", 1);
    setenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK", "1", 1);
    assert(bn_gpu_policy_prefill_matmul_disabled());
    assert(bn_gpu_policy_prefill_matmul_enabled());
    assert(bn_gpu_policy_prefill_direct_kv_disabled());
    assert(bn_gpu_policy_prefill_direct_kv_with_cpu_fallback_enabled());
    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    unsetenv("BN_GPU_PREFILL_MATMUL");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");

    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    unsetenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_LAYER");
    unsetenv("BN_GPU_CPU_FFN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER");
    assert(!bn_gpu_policy_cpu_decode_fallback_requested());
    assert(bn_gpu_policy_cpu_fallback_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_cpu_fallback_from_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_cpu_attention_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_cpu_attention_from_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_cpu_ffn_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_cpu_ffn_from_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_cpu_ffn_down_from_layer_or_default(-1) == -1);
    setenv("BN_GPU_CPU_FALLBACK_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    assert(bn_gpu_policy_cpu_fallback_layer_or_default(-1) == 1);
    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    setenv("BN_GPU_CPU_FALLBACK_FROM_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    assert(bn_gpu_policy_cpu_fallback_from_layer_or_default(-1) == 1);
    unsetenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
    setenv("BN_GPU_CPU_ATTN_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    assert(bn_gpu_policy_cpu_attention_layer_or_default(-1) == 1);
    unsetenv("BN_GPU_CPU_ATTN_LAYER");
    setenv("BN_GPU_CPU_ATTN_FROM_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    assert(bn_gpu_policy_cpu_attention_from_layer_or_default(-1) == 1);
    unsetenv("BN_GPU_CPU_ATTN_FROM_LAYER");
    setenv("BN_GPU_CPU_FFN_LAYER", "2", 1);
    setenv("BN_GPU_CPU_FFN_FROM_LAYER", "3", 1);
    setenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER", "4", 1);
    assert(bn_gpu_policy_cpu_ffn_layer_or_default(-1) == 2);
    assert(bn_gpu_policy_cpu_ffn_from_layer_or_default(-1) == 3);
    assert(bn_gpu_policy_cpu_ffn_down_from_layer_or_default(-1) == 4);
    unsetenv("BN_GPU_CPU_FFN_LAYER");
    unsetenv("BN_GPU_CPU_FFN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER");

    unsetenv("BN_CUDA_DISABLE_SSM_GRAPH");
    assert(!bn_gpu_policy_ssm_graph_disabled());
    setenv("BN_CUDA_DISABLE_SSM_GRAPH", "1", 1);
    assert(bn_gpu_policy_ssm_graph_disabled());
    unsetenv("BN_CUDA_DISABLE_SSM_GRAPH");

    unsetenv("BN_CUDA_DISABLE_QKV_MIXED_FUSE");
    unsetenv("BN_CUDA_DISABLE_QKV_KCACHE_FUSE");
    unsetenv("BN_CUDA_ENABLE_QKV_KPAIR_OPT");
    unsetenv("BN_CUDA_DISABLE_Q5_GATEUP_WARP");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_GATEUP_WARP");
    unsetenv("BN_CUDA_DISABLE_Q8_GATEUP_WARP");
    unsetenv("BN_CUDA_ENABLE_GRAPH_EXEC");
    unsetenv("BN_CUDA_ENABLE_UNSAFE_MOE_FFN");
    unsetenv("BN_CUDA_MOE_GRAPH_MAX_EXPERTS");
    unsetenv("BN_CUDA_DISABLE_GRAPH_EXEC");
    unsetenv("BN_CUDA_ENABLE_MOE_FFN");
    assert(!bn_gpu_policy_cuda_qkv_mixed_fuse_disabled());
    assert(bn_gpu_policy_cuda_qkv_key_cache_fuse_enabled());
    assert(!bn_gpu_policy_cuda_qkv_kpair_opt_enabled());
    assert(!bn_gpu_policy_cuda_legacy_block_gateup_warp_disabled());
    assert(!bn_gpu_policy_cuda_native_quant_gateup_warp_disabled());
    assert(!bn_gpu_policy_cuda_graph_exec_requested());
    assert(bn_gpu_policy_cuda_moe_graph_max_experts_or_default(128) == 128);
    assert(bn_gpu_policy_cuda_decode_graph_default_enabled(0, 0));
    assert(!bn_gpu_policy_cuda_decode_graph_default_enabled(1, 0));
    assert(bn_gpu_policy_cuda_decode_graph_default_enabled(1, 1));
    setenv("BN_CUDA_DISABLE_QKV_MIXED_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_QKV_KCACHE_FUSE", "1", 1);
    setenv("BN_CUDA_ENABLE_QKV_KPAIR_OPT", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5_GATEUP_WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_GATEUP_WARP", "1", 1);
    setenv("BN_CUDA_ENABLE_GRAPH_EXEC", "1", 1);
    setenv("BN_CUDA_MOE_GRAPH_MAX_EXPERTS", "7", 1);
    assert(bn_gpu_policy_cuda_qkv_mixed_fuse_disabled());
    assert(!bn_gpu_policy_cuda_qkv_key_cache_fuse_enabled());
    assert(bn_gpu_policy_cuda_qkv_kpair_opt_enabled());
    assert(bn_gpu_policy_cuda_legacy_block_gateup_warp_disabled());
    assert(bn_gpu_policy_cuda_native_quant_gateup_warp_disabled());
    assert(bn_gpu_policy_cuda_graph_exec_requested());
    assert(bn_gpu_policy_cuda_moe_graph_max_experts_or_default(128) == 7);
    unsetenv("BN_CUDA_ENABLE_GRAPH_EXEC");
    setenv("BN_CUDA_ENABLE_UNSAFE_MOE_FFN", "1", 1);
    assert(bn_gpu_policy_cuda_graph_exec_requested());
    setenv("BN_CUDA_MOE_GRAPH_MAX_EXPERTS", "0", 1);
    assert(bn_gpu_policy_cuda_moe_graph_max_experts_or_default(128) == 128);
    setenv("BN_CUDA_DISABLE_GRAPH_EXEC", "1", 1);
    assert(!bn_gpu_policy_cuda_decode_graph_default_enabled(0, 0));
    unsetenv("BN_CUDA_DISABLE_GRAPH_EXEC");
    setenv("BN_CUDA_ENABLE_MOE_FFN", "1", 1);
    assert(!bn_gpu_policy_cuda_decode_graph_default_enabled(0, 0));
    unsetenv("BN_CUDA_DISABLE_QKV_MIXED_FUSE");
    unsetenv("BN_CUDA_DISABLE_QKV_KCACHE_FUSE");
    unsetenv("BN_CUDA_ENABLE_QKV_KPAIR_OPT");
    unsetenv("BN_CUDA_DISABLE_Q5_GATEUP_WARP");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_GATEUP_WARP");
    unsetenv("BN_CUDA_DISABLE_Q8_GATEUP_WARP");
    unsetenv("BN_CUDA_ENABLE_UNSAFE_MOE_FFN");
    unsetenv("BN_CUDA_MOE_GRAPH_MAX_EXPERTS");
    unsetenv("BN_CUDA_ENABLE_MOE_FFN");

    unsetenv("BN_GPU_MAX_STORAGE_BINDING_MB");
    assert(bn_gpu_policy_max_storage_binding_bytes(0) ==
           128u * 1024u * 1024u);
    assert(bn_gpu_policy_max_storage_binding_bytes(7u * 1024u * 1024u) ==
           7u * 1024u * 1024u);
    setenv("BN_GPU_MAX_STORAGE_BINDING_MB", "3", 1);
    assert(bn_gpu_policy_max_storage_binding_bytes(7u * 1024u * 1024u) ==
           3u * 1024u * 1024u);
    setenv("BN_GPU_MAX_STORAGE_BINDING_MB", "-1", 1);
    assert(bn_gpu_policy_max_storage_binding_bytes(7u * 1024u * 1024u) ==
           7u * 1024u * 1024u);
    setenv("BN_GPU_MAX_STORAGE_BINDING_MB", "bad", 1);
    assert(bn_gpu_policy_max_storage_binding_bytes(7u * 1024u * 1024u) == 0);
    unsetenv("BN_GPU_MAX_STORAGE_BINDING_MB");

    unsetenv("BN_CUDA_DISABLE_CUBLAS_MATMUL");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_CUBLAS_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16");
    unsetenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    unsetenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_ENABLE_Q8_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_Q8_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_Q8_PREQ");
    unsetenv("BN_CUDA_DISABLE_Q8_PREQ");
    unsetenv("BN_CUDA_DISABLE_Q8_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_ENABLE_Q8_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_DISABLE_Q8_PREQ_LOGITS");
    unsetenv("BN_CUDA_ENABLE_Q8_PREQ_LOGITS");
    unsetenv("BN_CUDA_DISABLE_Q8K_INPUT_CACHE");
    unsetenv("BN_CUDA_DISABLE_PREPARED_KQUANT_INPUT_CACHE");
    assert(bn_gpu_policy_cuda_cublas_matmul_enabled());
    assert(bn_gpu_policy_cuda_down_kquant_cublas_f16_cache_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_matmul_enabled());
    assert(bn_gpu_policy_cuda_f16_native_quant_matmul_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_logits_disabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(0));
    assert(bn_gpu_policy_prepared_kquant_input_cache_enabled());
    setenv("BN_CUDA_ENABLE_NATIVE_QUANT_MATMUL", "1", 1);
    setenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT", "1", 1);
    setenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS", "1", 1);
    assert(bn_gpu_policy_cuda_native_quant_matmul_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(0));
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS", "1", 1);
    setenv("BN_CUDA_DISABLE_PREPARED_KQUANT_INPUT_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_native_quant_matmul_enabled());
    assert(!bn_gpu_policy_cuda_f16_native_quant_matmul_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_logits_disabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(1));
    assert(!bn_gpu_policy_prepared_kquant_input_cache_enabled());
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_DISABLE_PREPARED_KQUANT_INPUT_CACHE");
    setenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_0_PREPARED_INPUT_SPLIT", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_PREPARED_INPUT_LOGITS", "1", 1);
    assert(bn_gpu_policy_cuda_native_quant_matmul_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(0));
    setenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_0_PREPARED_INPUT_SPLIT", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_PREPARED_INPUT_LOGITS", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8K_INPUT_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_native_quant_matmul_enabled());
    assert(!bn_gpu_policy_cuda_f16_native_quant_matmul_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_logits_disabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(1));
    assert(!bn_gpu_policy_prepared_kquant_input_cache_enabled());
    unsetenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATMUL");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_ENABLE_Q8_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_Q8_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_Q8_PREQ");
    unsetenv("BN_CUDA_DISABLE_Q8_PREQ");
    unsetenv("BN_CUDA_DISABLE_Q8_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_ENABLE_Q8_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_LOGITS");
    unsetenv("BN_CUDA_DISABLE_Q8_PREQ_LOGITS");
    unsetenv("BN_CUDA_ENABLE_Q8_PREQ_LOGITS");
    unsetenv("BN_CUDA_DISABLE_Q8K_INPUT_CACHE");
    unsetenv("BN_CUDA_DISABLE_PREPARED_KQUANT_INPUT_CACHE");
    setenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_PREQ", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_PREQ_LOGITS", "1", 1);
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(0));
    setenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_PREQ", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_PREQ_LOGITS", "1", 1);
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled());
    assert(bn_gpu_policy_cuda_native_quant_prepared_input_logits_disabled());
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_ENABLE_Q8_PREQ");
    unsetenv("BN_CUDA_DISABLE_Q8_PREQ");
    unsetenv("BN_CUDA_DISABLE_Q8_PREQ_LOGITS");
    unsetenv("BN_CUDA_ENABLE_Q8_PREQ_LOGITS");
    unsetenv("BN_CUDA_FORCE_ASYMMETRIC_KQUANT_QUANT_MATMUL");
    unsetenv("BN_CUDA_FORCE_Q4K_QUANT_MATMUL");
    unsetenv("BN_CUDA_FORCE_DOWN_KQUANT_QUANT_MATMUL");
    unsetenv("BN_CUDA_FORCE_Q6K_QUANT_MATMUL");
    assert(!bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q8_0, 1));
    assert(bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q8_0, 0));
    assert(!bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q4_K, 1));
    assert(!bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q6_K, 1));
    setenv("BN_CUDA_FORCE_ASYMMETRIC_KQUANT_QUANT_MATMUL", "1", 1);
    assert(bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q4_K, 1));
    assert(!bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q5_K, 1));
    unsetenv("BN_CUDA_FORCE_ASYMMETRIC_KQUANT_QUANT_MATMUL");
    setenv("BN_CUDA_FORCE_Q4K_QUANT_MATMUL", "1", 1);
    assert(bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q4_K, 1));
    setenv("BN_CUDA_FORCE_DOWN_KQUANT_QUANT_MATMUL", "1", 1);
    assert(bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q6_K, 1));
    unsetenv("BN_CUDA_FORCE_DOWN_KQUANT_QUANT_MATMUL");
    setenv("BN_CUDA_FORCE_Q6K_QUANT_MATMUL", "1", 1);
    assert(bn_gpu_policy_cuda_force_quant_matmul_for_type(
        BN_GGUF_TENSOR_Q6_K, 1));
    unsetenv("BN_CUDA_FORCE_ASYMMETRIC_KQUANT_QUANT_MATMUL");
    unsetenv("BN_CUDA_FORCE_Q4K_QUANT_MATMUL");
    unsetenv("BN_CUDA_FORCE_DOWN_KQUANT_QUANT_MATMUL");
    unsetenv("BN_CUDA_FORCE_Q6K_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_Q6K_4WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_4WARP_1536_8960");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_4WARP_5120");
    unsetenv("BN_CUDA_ENABLE_Q6K_4WARP_5120");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_5WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q6K_5WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_3WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q6K_3WARP_2560_9728");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_2WARP_LONG");
    unsetenv("BN_CUDA_ENABLE_Q6K_2WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_2WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_Q6K_2WARP_LONG");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_1024_2560");
    unsetenv("BN_CUDA_ENABLE_Q6K_MATVEC4_1024_2560");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_512_2048");
    unsetenv("BN_CUDA_ENABLE_Q6K_MATVEC4_512_2048");
    assert(bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(1536, 8960));
    assert(bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(2560, 8192));
    assert(!bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(2560, 6144));
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_4WARP_5120", "1", 1);
    assert(bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(2560, 6144));
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_1536_8960", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(1536, 8960));
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_LONG", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(2560, 8192));
    assert(bn_gpu_policy_cuda_down_kquant_5warp_exact_enabled(1536, 8960));
    assert(bn_gpu_policy_cuda_down_kquant_5warp_exact_enabled(2560, 9728));
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_1536_8960", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_2560_9728", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_5warp_exact_enabled(1536, 8960));
    assert(!bn_gpu_policy_cuda_down_kquant_5warp_exact_enabled(2560, 9728));
    assert(bn_gpu_policy_cuda_down_kquant_3warp_exact_enabled(1536, 8960));
    assert(bn_gpu_policy_cuda_down_kquant_3warp_exact_enabled(2560, 9728));
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_1536_8960", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_2560_9728", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_3warp_exact_enabled(1536, 8960));
    assert(!bn_gpu_policy_cuda_down_kquant_3warp_exact_enabled(2560, 9728));
    assert(!bn_gpu_policy_cuda_down_kquant_2warp_long_enabled(2560, 8192));
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_2WARP_LONG", "1", 1);
    assert(bn_gpu_policy_cuda_down_kquant_2warp_long_enabled(2560, 8192));
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_2WARP_LONG", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_2warp_long_enabled(2560, 8192));
    assert(bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(1024, 2560));
    assert(bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(512, 2048));
    assert(!bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(768, 2048));
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_1024_2560", "1", 1);
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_512_2048", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(1024, 2560));
    assert(!bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(512, 2048));
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_4WARP_5120");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_2560_9728");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_2WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_2WARP_LONG");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_1024_2560");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_512_2048");
    setenv("BN_CUDA_ENABLE_Q6K_4WARP_5120", "1", 1);
    assert(bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(2560, 6144));
    setenv("BN_CUDA_DISABLE_Q6K_4WARP_1536_8960", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(1536, 8960));
    setenv("BN_CUDA_DISABLE_Q6K_4WARP_LONG", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(2560, 8192));
    setenv("BN_CUDA_DISABLE_Q6K_5WARP_1536_8960", "1", 1);
    setenv("BN_CUDA_DISABLE_Q6K_5WARP_2560_9728", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_5warp_exact_enabled(1536, 8960));
    assert(!bn_gpu_policy_cuda_down_kquant_5warp_exact_enabled(2560, 9728));
    setenv("BN_CUDA_DISABLE_Q6K_3WARP_1536_8960", "1", 1);
    setenv("BN_CUDA_DISABLE_Q6K_3WARP_2560_9728", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_3warp_exact_enabled(1536, 8960));
    assert(!bn_gpu_policy_cuda_down_kquant_3warp_exact_enabled(2560, 9728));
    setenv("BN_CUDA_ENABLE_Q6K_2WARP_LONG", "1", 1);
    setenv("BN_CUDA_DISABLE_Q6K_2WARP_LONG", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_2warp_long_enabled(2560, 8192));
    setenv("BN_CUDA_ENABLE_Q6K_MATVEC4_1024_2560", "1", 1);
    setenv("BN_CUDA_ENABLE_Q6K_MATVEC4_512_2048", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(1024, 2560));
    assert(!bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(512, 2048));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    assert(bn_gpu_policy_cuda_moe_down_quant_path_preferred(
        1, BN_GGUF_TENSOR_Q6_K, 1024, 4, 2));
    assert(bn_gpu_policy_cuda_moe_down_quant_path_preferred(
        1, BN_GGUF_TENSOR_Q6_K, 768, 2, 4));
    assert(!bn_gpu_policy_cuda_moe_down_quant_path_preferred(
        0, BN_GGUF_TENSOR_Q6_K, 1024, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_quant_path_preferred(
        1, BN_GGUF_TENSOR_Q6_K, 2048, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_quant_path_preferred(
        1, BN_GGUF_TENSOR_Q4_K, 1024, 4, 2));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_quant_path_preferred(
        1, BN_GGUF_TENSOR_Q6_K, 1024, 4, 2));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    setenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_quant_path_preferred(
        1, BN_GGUF_TENSOR_Q6_K, 1024, 4, 2));
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    assert(bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q6_K, 1, 0, 4096, 1024, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q6_K, 0, 0, 4096, 1024, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q6_K, 1, 1, 4096, 1024, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q4_K, 1, 0, 4096, 1024, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q6_K, 1, 0, 2048, 4096, 2, 2));
    assert(bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q6_K, 1, 0, 4096, 4096, 2, 2));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q6_K, 1, 0, 4096, 1024, 4, 2));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
        1, BN_GGUF_TENSOR_Q6_K, 1, 0, 4096, 1024, 4, 2));
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_HALFWARP");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_HALFWARP");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SPLIT4");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SPLIT4");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SCATTER_16ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_FLOAT_PATH");
    unsetenv("BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_PATH");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_PAIR2");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_4ROW");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM");
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_PAIR4_SUM");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_4ROW_SUM");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_K8_8ROW_SUM");
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_FIXED");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_EXACT_2048_768");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM_4ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F16_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN");
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN");
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW");
    assert(bn_gpu_policy_cuda_moe_down_4row_enabled(1024));
    assert(!bn_gpu_policy_cuda_moe_down_4row_enabled(1025));
    assert(bn_gpu_policy_cuda_moe_down_8row_enabled(1024));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_4row_enabled(1024));
    assert(!bn_gpu_policy_cuda_moe_down_8row_enabled(1024));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW");
    assert(!bn_gpu_policy_cuda_moe_down_halfwarp_enabled(
        BN_GGUF_TENSOR_Q6_K, 1, 2, 2));
    assert(bn_gpu_policy_cuda_moe_down_halfwarp_enabled(
        BN_GGUF_TENSOR_Q6_K, 1, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_halfwarp_enabled(
        BN_GGUF_TENSOR_Q6_K, 0, 2, 2));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_HALFWARP", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_halfwarp_enabled(
        BN_GGUF_TENSOR_Q6_K, 0, 4, 2));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_HALFWARP", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_halfwarp_enabled(
        BN_GGUF_TENSOR_Q6_K, 1, 4, 2));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_HALFWARP");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_HALFWARP");
    assert(!bn_gpu_policy_cuda_moe_down_split4_enabled(
        BN_GGUF_TENSOR_Q6_K, 1, 4, 2));
    assert(!bn_gpu_policy_cuda_moe_down_split4_enabled(
        BN_GGUF_TENSOR_Q6_K, 0, 4, 2));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_SPLIT4", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_split4_enabled(
        BN_GGUF_TENSOR_Q6_K, 0, 4, 2));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_SPLIT4", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_split4_enabled(
        BN_GGUF_TENSOR_Q6_K, 0, 4, 2));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SPLIT4");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SPLIT4");
    assert(bn_gpu_policy_cuda_moe_down_scatter_enabled(
        BN_GGUF_TENSOR_Q6_K, 0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_scatter_enabled(
        BN_GGUF_TENSOR_Q4_K, 0, 0));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_scatter_enabled(
        BN_GGUF_TENSOR_Q6_K, 0, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER");
    assert(!bn_gpu_policy_cuda_moe_down_scatter_16row_enabled(1, 1024));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_SCATTER_16ROW", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_scatter_16row_enabled(1, 768));
    assert(!bn_gpu_policy_cuda_moe_down_scatter_16row_enabled(0, 768));
    assert(bn_gpu_policy_cuda_moe_down_float_path_enabled());
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_FLOAT_PATH", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_float_path_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_FLOAT_PATH");
    setenv("BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_float_path_enabled());
    unsetenv("BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN");
    assert(bn_gpu_policy_cuda_moe_down_pair_path_enabled(0, 0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_pair_path_enabled(1, 0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_pair_path_enabled(0, 1, 0));
    assert(!bn_gpu_policy_cuda_moe_down_pair_path_enabled(0, 0, 1));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_PATH", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_pair_path_enabled(0, 0, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_PATH");
    setenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_pair_path_enabled(0, 0, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN");
    assert(bn_gpu_policy_cuda_moe_down_prefers_f32_cache(1, 4096, 0, 0));
    assert(bn_gpu_policy_cuda_moe_down_prefers_f32_cache(1, 4096, 1, 1));
    assert(!bn_gpu_policy_cuda_moe_down_prefers_f32_cache(0, 4096, 0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_prefers_f32_cache(1, 2048, 0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_prefers_f32_cache(1, 4096, 1, 0));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prefers_f32_cache(1, 4096, 0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_enabled(1, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prefers_f32_cache(1, 4096, 0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_enabled(1, 0));
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    assert(bn_gpu_policy_cuda_moe_down_f32_pair2_enabled(2, 2));
    assert(!bn_gpu_policy_cuda_moe_down_f32_pair2_enabled(4, 2));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_PAIR2", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f32_pair2_enabled(2, 2));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_PAIR2");
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f32_pair2_enabled(2, 2));
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2");
    assert(bn_gpu_policy_cuda_moe_down_f32_pair2_4row_enabled());
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_4ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f32_pair2_4row_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_4ROW");
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f32_pair2_4row_enabled());
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW");
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(1));
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM", "1", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(1));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(0));
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM", "1", 1);
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(1));
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM");
    setenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM", "1", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM", "1", 1);
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(1));
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM");
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled(1));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled(0));
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_PAIR4_SUM", "1", 1);
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled(1));
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_PAIR4_SUM");
    setenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM", "1", 1);
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled(1));
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM");
    assert(bn_gpu_policy_cuda_moe_down_prepared_k8_4row_sum_enabled(
        0, 8, 1024));
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_4row_sum_enabled(
        1, 8, 1024));
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_4row_sum_enabled(
        0, 9, 1024));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_4ROW_SUM", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_4row_sum_enabled(
        0, 8, 1024));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_4ROW_SUM");
    setenv("BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_4row_sum_enabled(
        0, 8, 1024));
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM");
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_8row_sum_enabled(
        1, 1024));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_K8_8ROW_SUM", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_prepared_k8_8row_sum_enabled(
        1, 1024));
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_8row_sum_enabled(
        0, 1024));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_K8_8ROW_SUM");
    setenv("BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_prepared_k8_8row_sum_enabled(
        1, 1024));
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM");
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_fixed_enabled(1));
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_FIXED", "1", 1);
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_fixed_enabled(1));
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_FIXED");
    setenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED", "1", 1);
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_fixed_enabled(1));
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED");
    assert(bn_gpu_policy_cuda_moe_down_resid_rmsnorm_fuse_enabled());
    setenv("BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_resid_rmsnorm_fuse_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE");
    assert(bn_gpu_policy_cuda_moe_down_prepared_k8_exact_2048_768_enabled(
        2048, 768, 8));
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_exact_2048_768_enabled(
        2048, 1024, 8));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_EXACT_2048_768", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_exact_2048_768_enabled(
        2048, 768, 8));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_EXACT_2048_768");
    setenv("BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_k8_exact_2048_768_enabled(
        2048, 768, 8));
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768");
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_accum_4row_enabled());
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM_4ROW", "1", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_accum_4row_enabled());
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM_4ROW");
    setenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW", "1", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_accum_4row_enabled());
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW");
    assert(bn_gpu_policy_cuda_moe_down_prepared_pair_4row_enabled());
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_4ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_pair_4row_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_4ROW");
    setenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_pair_4row_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW");
    assert(bn_gpu_policy_cuda_moe_down_f32_cache_enabled(1, 0));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_enabled(0, 0));
    assert(!bn_gpu_policy_cuda_moe_down_f32_cache_enabled(1, 1));
    assert(!bn_gpu_policy_cuda_moe_down_f16_cache_enabled(1));
    setenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_f16_cache_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F16_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f16_cache_enabled(1));
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F16_CACHE");
    setenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_f16_cache_enabled(1));
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_f16_cache_enabled(1));
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE");
    assert(bn_gpu_policy_cuda_moe_down_aux_f32_cache_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_down_aux_f32_cache_enabled(0));
    setenv("BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_aux_f32_cache_enabled(1));
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_aux_f32_cache_enabled(1));
    unsetenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");
    assert(!bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(2, 2, 4096));
    setenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(2, 2, 4096));
    assert(!bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(4, 2, 4096));
    assert(!bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(2, 2, 2048));
    setenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(2, 2, 4096));
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_PAIR_DOWN");
    setenv("BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(2, 2, 4096));
    setenv("BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(2, 2, 4096));
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN");
    assert(!bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(1024));
    setenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(1024));
    assert(!bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(1025));
    setenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(1024));
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOWN_8ROW");
    setenv("BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(1024));
    setenv("BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(1024));
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW");
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_DOT");
    unsetenv("BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_8ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_GATEUP_SPLIT");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(1, 4096, 0));
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(2, 2048, 1));
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(2, 2048, 0));
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(2, 4096, 1));
    setenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_DOT", "1", 1);
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(2, 4096, 0));
    setenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(1, 2048, 1));
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_DOT");
    setenv("BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP", "1", 1);
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(2, 4096, 0));
    unsetenv("BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP");
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_8row_enabled(2048));
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_8row_enabled(2049));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_8ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_8row_enabled(2048));
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_8ROW");
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_split_enabled(2048, 2));
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_split_enabled(2049, 4));
    setenv("BN_CUDA_ENABLE_MOE_GATEUP_SPLIT", "1", 1);
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_split_enabled(2048, 4));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_split_enabled(2048, 4));
    unsetenv("BN_CUDA_ENABLE_MOE_GATEUP_SPLIT");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_BATCH");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1");
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_GATEUP_2ROW");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_NATIVE_QUANT_DOWN_4ROW");
    unsetenv("BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_DOWN_2ROW");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW");
    assert(bn_gpu_policy_cuda_moe_block_prepared_batch_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_block_prepared_batch_enabled(0));
    assert(bn_gpu_policy_cuda_moe_gateup_block_2row_enabled(1024));
    assert(!bn_gpu_policy_cuda_moe_gateup_block_2row_enabled(1025));
    assert(!bn_gpu_policy_cuda_moe_down_block_4row_enabled(1024));
    assert(bn_gpu_policy_cuda_moe_down_block_2row_enabled(1024));
    setenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_BATCH", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_GATEUP_2ROW", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_NATIVE_QUANT_DOWN_4ROW", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_DOWN_2ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_block_prepared_batch_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_gateup_block_2row_enabled(1024));
    assert(bn_gpu_policy_cuda_moe_down_block_4row_enabled(1024));
    assert(!bn_gpu_policy_cuda_moe_down_block_2row_enabled(1024));
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_BATCH");
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_GATEUP_2ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_NATIVE_QUANT_DOWN_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_DOWN_2ROW");
    setenv("BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_block_prepared_batch_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_gateup_block_2row_enabled(1024));
    assert(bn_gpu_policy_cuda_moe_down_block_4row_enabled(1024));
    assert(!bn_gpu_policy_cuda_moe_down_block_2row_enabled(1024));
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW");
    unsetenv("BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_Q6K_4WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_4WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_4WARP_1536_8960");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_4WARP_5120");
    unsetenv("BN_CUDA_ENABLE_Q6K_4WARP_5120");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_5WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_5WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q6K_5WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_3WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_3WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q6K_3WARP_2560_9728");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_2WARP_LONG");
    unsetenv("BN_CUDA_ENABLE_Q6K_2WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_2WARP_LONG");
    unsetenv("BN_CUDA_DISABLE_Q6K_2WARP_LONG");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_1024_2560");
    unsetenv("BN_CUDA_ENABLE_Q6K_MATVEC4_1024_2560");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATVEC4_512_2048");
    unsetenv("BN_CUDA_ENABLE_Q6K_MATVEC4_512_2048");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_HALFWARP");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_HALFWARP");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SPLIT4");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SPLIT4");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_SCATTER_16ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_FLOAT_PATH");
    unsetenv("BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_PATH");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_PAIR2");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_4ROW");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM");
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_PAIR4_SUM");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_4ROW_SUM");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_K8_8ROW_SUM");
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_FIXED");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_K8_EXACT_2048_768");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOWN_ACCUM_4ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_PAIR_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F16_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_SMALL_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN");
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW");
    unsetenv("BN_CUDA_ENABLE_LOGITS_CACHE");
    assert(!bn_gpu_policy_decode_logits_cache_enabled(0));
    setenv("BN_CUDA_ENABLE_LOGITS_CACHE", "1", 1);
    assert(bn_gpu_policy_decode_logits_cache_enabled(0));
    assert(!bn_gpu_policy_decode_logits_cache_enabled(1));
    unsetenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE");
    assert(!bn_gpu_policy_moe_decode_cache_enabled());
    setenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_decode_cache_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");
    assert(!bn_gpu_policy_moe_decode_cache_disabled());
    setenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_moe_decode_cache_disabled());
    unsetenv("BN_CUDA_DISABLE_DECODE_CACHE");
    assert(!bn_gpu_policy_decode_cache_disabled());
    setenv("BN_CUDA_DISABLE_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_decode_cache_disabled());
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE");
    assert(!bn_gpu_policy_native_quant_decode_cache_disabled());
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_native_quant_decode_cache_disabled());
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_DECODE_CACHE");
    setenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_native_quant_decode_cache_disabled());
    unsetenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_LOGITS_ARGMAX");
    assert(!bn_gpu_policy_logits_argmax_disabled());
    setenv("BN_CUDA_DISABLE_LOGITS_ARGMAX", "1", 1);
    assert(bn_gpu_policy_logits_argmax_disabled());
    unsetenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
    assert(!bn_gpu_policy_dense_logits_argmax_enabled());
    setenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX", "1", 1);
    assert(bn_gpu_policy_dense_logits_argmax_enabled());
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX");
    assert(!bn_gpu_policy_moe_logits_mmvq_argmax_enabled());
    setenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX", "1", 1);
    assert(bn_gpu_policy_moe_logits_mmvq_argmax_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
    assert(!bn_gpu_policy_moe_logits_mmvq_argmax_disabled());
    setenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX", "1", 1);
    assert(bn_gpu_policy_moe_logits_mmvq_argmax_disabled());
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX");
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
    assert(bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(50000,
                                                                  1536));
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(49999,
                                                                   1536));
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(50000,
                                                                   2048));
    setenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX", "1", 1);
    assert(bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(50000,
                                                                  2048));
    setenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(50000,
                                                                   1536));
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536");
    assert(bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_enabled(1, 151936,
                                                                  1536));
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_enabled(1, 151936,
                                                                   2048));
    setenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_enabled(1, 151936,
                                                                   1536));
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_1WARP16_1536");
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_1warp16_1536_enabled(1));
    setenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_1WARP16_1536", "1", 1);
    assert(bn_gpu_policy_cuda_moe_logits_mmvq_1warp16_1536_enabled(1));
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536_UNROLL");
    assert(bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_unroll_enabled(1,
                                                                         0));
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_unroll_enabled(1,
                                                                          1));
    setenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536_UNROLL", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_unroll_enabled(1,
                                                                          0));
    unsetenv("BN_CUDA_DISABLE_ARGMAX_FAST");
    assert(bn_gpu_policy_cuda_argmax_fast_enabled());
    setenv("BN_CUDA_DISABLE_ARGMAX_FAST", "1", 1);
    assert(!bn_gpu_policy_cuda_argmax_fast_enabled());
    unsetenv("BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY");
    assert(!bn_gpu_policy_cuda_optimistic_argmax_penalty_enabled());
    setenv("BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY", "1", 1);
    assert(bn_gpu_policy_cuda_optimistic_argmax_penalty_enabled());
    unsetenv("BN_CUDA_ENABLE_LEGACY_BLOCK_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_Q5_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_LEGACY_BLOCK_WARP");
    unsetenv("BN_CUDA_ENABLE_Q5_WARP");
    assert(!bn_gpu_policy_cuda_legacy_block_matvec4_enabled());
    assert(!bn_gpu_policy_cuda_legacy_block_warp_enabled());
    unsetenv("BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_Q5K_DEINT_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_4WARP");
    unsetenv("BN_CUDA_DISABLE_Q5K_4WARP");
    unsetenv("BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_SPLIT_4WARP");
    unsetenv("BN_CUDA_ENABLE_Q5K_SPLIT_4WARP");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_GATEUP_2WARP");
    unsetenv("BN_CUDA_DISABLE_Q5K_GATEUP_2WARP");
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_pair_matvec_enabled());
    assert(bn_gpu_policy_cuda_deinterleaved_kquant_4warp_enabled(8192));
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_4warp_enabled(8193));
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_split_4warp_enabled(8192));
    assert(bn_gpu_policy_cuda_deinterleaved_kquant_gateup_2warp_enabled());
    setenv("BN_CUDA_ENABLE_LEGACY_BLOCK_MATVEC4", "1", 1);
    setenv("BN_CUDA_ENABLE_LEGACY_BLOCK_WARP", "1", 1);
    setenv("BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_PAIR_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_SPLIT_4WARP", "1", 1);
    assert(bn_gpu_policy_cuda_deinterleaved_kquant_split_4warp_enabled(8192));
    setenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_4WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_GATEUP_2WARP", "1", 1);
    assert(bn_gpu_policy_cuda_legacy_block_matvec4_enabled());
    assert(bn_gpu_policy_cuda_legacy_block_warp_enabled());
    assert(bn_gpu_policy_cuda_deinterleaved_kquant_pair_matvec_enabled());
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_4warp_enabled(8192));
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_split_4warp_enabled(8192));
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_gateup_2warp_enabled());
    unsetenv("BN_CUDA_ENABLE_LEGACY_BLOCK_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_LEGACY_BLOCK_WARP");
    unsetenv("BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_DEINTERLEAVED_KQUANT_SPLIT_4WARP");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_4WARP");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_GATEUP_2WARP");
    setenv("BN_CUDA_ENABLE_Q5_MATVEC4", "1", 1);
    setenv("BN_CUDA_ENABLE_Q5_WARP", "1", 1);
    setenv("BN_CUDA_ENABLE_Q5K_DEINT_PAIR_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_Q5K_SPLIT_4WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5K_4WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5K_GATEUP_2WARP", "1", 1);
    assert(bn_gpu_policy_cuda_legacy_block_matvec4_enabled());
    assert(bn_gpu_policy_cuda_legacy_block_warp_enabled());
    assert(bn_gpu_policy_cuda_deinterleaved_kquant_pair_matvec_enabled());
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_4warp_enabled(8192));
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_split_4warp_enabled(8192));
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_gateup_2warp_enabled());
    unsetenv("BN_CUDA_ENABLE_Q5_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_Q5_WARP");
    unsetenv("BN_CUDA_ENABLE_Q5K_DEINT_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_Q5K_SPLIT_4WARP");
    unsetenv("BN_CUDA_DISABLE_Q5K_4WARP");
    unsetenv("BN_CUDA_DISABLE_Q5K_GATEUP_2WARP");
    unsetenv("BN_CUDA_DISABLE_SYMMETRIC_KQUANT_DOT");
    unsetenv("BN_CUDA_DISABLE_Q4K_DOT");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_DOT");
    unsetenv("BN_CUDA_DISABLE_Q5K_DOT");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP");
    unsetenv("BN_CUDA_DISABLE_Q4K_4WARP");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q4K_4WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q4K_4WARP_2560_9728");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_OUT_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_QKV_MIXED_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q4K_QKV_MIXED_FUSE");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_QK_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_QK_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_4WARP_2048");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_4WARP_2048");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_5WARP_2560");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_5WARP_2560");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE_1792");
    unsetenv("BN_CUDA_ENABLE_Q4K_SPLIT_VALUE_FUSE_1792");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_VALUE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_Q8_1_FAST");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_Q8_1_FAST");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_QWARP4");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_5WARP_2560");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_5WARP_2560");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_2WARP");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_2WARP");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_WARP");
    unsetenv("BN_CUDA_DISABLE_Q8_WARP");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_SSM_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0_SSM_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0_SSM_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_SSM_PREQ");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_SSM_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_Q8_MIXED_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_Q8_MIXED_PREQ");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_MIXED_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_DOT_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT");
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOT_INPUT");
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_DOT_INPUT");
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_FUSED_TOPK");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_FUSED_TOPK");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_4WARP");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_2WARP");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP_TOPK");
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_DECODE");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_Q8X");
    unsetenv("BN_CUDA_DISABLE_MOE_ALL2_FAST");
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOT");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_PROFILE_MOE_INTERNAL");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_FIXED");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_ALL2_FIXED");
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_4ROW");
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW");
    unsetenv("BN_CUDA_ENABLE_F16_NATIVE_QUANT_SSM_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_Q8_0_SSM_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_SSM_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_SSM_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_Q8_0_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_Q5K_MATVEC");
    assert(bn_gpu_policy_cuda_symmetric_kquant_dot_enabled());
    assert(bn_gpu_policy_cuda_deinterleaved_kquant_dot_enabled());
    setenv("BN_CUDA_DISABLE_Q4K_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5K_DOT", "1", 1);
    assert(!bn_gpu_policy_cuda_symmetric_kquant_dot_enabled());
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_dot_enabled());
    unsetenv("BN_CUDA_DISABLE_Q4K_DOT");
    unsetenv("BN_CUDA_DISABLE_Q5K_DOT");
    setenv("BN_CUDA_DISABLE_Q4K_4WARP", "1", 1);
    setenv("BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE", "1", 1);
    setenv("BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4", "1", 1);
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_4warp_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_out_residual_rmsnorm_fuse_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_k_rope_cache_fuse_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_qwarp4_enabled(4096));
    unsetenv("BN_CUDA_DISABLE_Q4K_4WARP");
    unsetenv("BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4");
    assert(bn_gpu_policy_cuda_asymmetric_kquant_4warp_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_4warp_shape_enabled(1536, 8960));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_4warp_shape_enabled(2560, 9728));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_4warp_shape_enabled(3072, 9728));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_out_residual_rmsnorm_fuse_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_qkv_mixed_fuse_enabled(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_qkv_mixed_fuse_enabled(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_k_rope_cache_fuse_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_qk_rope_cache_fuse_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_4warp_enabled(2048));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_4warp_enabled(2560));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_5warp_enabled(2560));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_5warp_enabled(2048));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_value_rows(4608, 2048) == 512);
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_value_rows(2304, 2048) == 256);
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_value_rows(1792, 1536) == 0);
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_value_fuse_enabled(256));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_value_fuse_enabled(0));
    assert(!bn_gpu_policy_kquant_gateup_prepared_path_enabled(0));
    assert(bn_gpu_policy_kquant_gateup_prepared_path_enabled(1));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_gateup_qwarp4_enabled(4096));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_qwarp4_enabled(4097));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_gateup_5warp_enabled(1, 2560));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_5warp_enabled(0, 2560));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_gateup_2warp_enabled(1, 5120));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_2warp_enabled(1, 5121));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_gateup_4warp_enabled(1, 8192));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_4warp_enabled(1, 8193));
    assert(!bn_gpu_policy_cuda_native_quant_warp_disabled());
    assert(bn_gpu_policy_cuda_native_quant_ssm_matvec_enabled());
    assert(bn_gpu_policy_cuda_native_quant_ssm_prepared_input_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_K, 2048));
    assert(bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(2048, 1));
    assert(!bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(2049, 1));
    assert(!bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(2048, 0));
    assert(!bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    assert(!bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 1));
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_SSM_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_NATIVE_QUANT_MIXED_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_DOT_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT", "1", 1);
    assert(!bn_gpu_policy_cuda_native_quant_ssm_prepared_input_enabled());
    assert(bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_K, 2048));
    assert(!bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(2048, 1));
    assert(!bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT");
    assert(bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_SSM_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_NATIVE_QUANT_MIXED_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_DOT_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_BLOCK_PREPARED_INPUT");
    assert(!bn_gpu_policy_cuda_moe_router_fused_topk_enabled(256, 0));
    assert(!bn_gpu_policy_cuda_moe_router_warp_disabled(0));
    assert(bn_gpu_policy_cuda_moe_router_warp_disabled(1));
    assert(bn_gpu_policy_cuda_moe_router_4warp_enabled(2048));
    assert(!bn_gpu_policy_cuda_moe_router_4warp_enabled(2047));
    assert(bn_gpu_policy_cuda_moe_router_2warp_enabled(2048));
    assert(!bn_gpu_policy_cuda_moe_router_2warp_enabled(2047));
    assert(bn_gpu_policy_cuda_moe_router_warp_topk_enabled(256));
    assert(!bn_gpu_policy_cuda_moe_router_warp_topk_enabled(257));
    assert(bn_gpu_policy_cuda_moe_block_prepared_decode_enabled());
    assert(bn_gpu_policy_cuda_moe_all_active_two_fast_enabled(0));
    assert(!bn_gpu_policy_cuda_moe_all_active_two_fast_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_prepared_dot_enabled(0, 0, 0, 2048,
                                                       2048));
    assert(bn_gpu_policy_cuda_moe_prepared_dot_enabled(0, 0, 0, 2049,
                                                      2049));
    assert(!bn_gpu_policy_cuda_moe_internal_profile_enabled(1));
    assert(bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(1, 1));
    assert(!bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(0, 1));
    assert(!bn_gpu_policy_cuda_moe_gateup_prepared_4row_disabled());
    assert(!bn_gpu_policy_cuda_f16_native_quant_ssm_matvec_enabled());
    assert(!bn_gpu_policy_cuda_f16_native_quant_matvec_enabled());
    assert(!bn_gpu_policy_cuda_f16_packed_kquant_matvec_enabled());
    setenv("BN_CUDA_DISABLE_SYMMETRIC_KQUANT_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_1536_8960", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_2560_9728", "1", 1);
    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_OUT_RESID_RMSNORM_FUSE", "1", 1);
    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_QKV_MIXED_FUSE", "1", 1);
    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_QK_ROPE_CACHE_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_4WARP_2048", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_5WARP_2560", "1", 1);
    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE_1792", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_Q8_1_FAST", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_QWARP4", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_5WARP_2560", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_2WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_SSM_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_0_SSM_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_MIXED_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_ROUTER_FUSED_TOPK", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_FUSED_TOPK", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_4WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_2WARP", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP_TOPK", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_DECODE", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ALL2_FAST", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_DOT_INPUT", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOT", "1", 1);
    setenv("BN_CUDA_PROFILE_MOE_INTERNAL", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_FIXED", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_4ROW", "1", 1);
    setenv("BN_CUDA_ENABLE_F16_NATIVE_QUANT_SSM_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_SSM_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_F16_NATIVE_QUANT_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_F16_Q5K_MATVEC", "1", 1);
    assert(!bn_gpu_policy_cuda_symmetric_kquant_dot_enabled());
    assert(!bn_gpu_policy_cuda_deinterleaved_kquant_dot_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_4warp_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_4warp_shape_enabled(1536, 8960));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_4warp_shape_enabled(2560, 9728));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_out_residual_rmsnorm_fuse_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_qkv_mixed_fuse_enabled(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_k_rope_cache_fuse_enabled());
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE", "1", 1);
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_k_rope_cache_fuse_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_qk_rope_cache_fuse_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_4warp_enabled(2048));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_5warp_enabled(2560));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_split_value_rows(1792, 1536) == 256);
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_split_value_fuse_enabled(256));
    assert(bn_gpu_policy_kquant_gateup_prepared_path_enabled(0));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_qwarp4_enabled(4096));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_5warp_enabled(1, 2560));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_gateup_2warp_enabled(1, 5120));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_gateup_4warp_enabled(1, 8192));
    assert(bn_gpu_policy_cuda_native_quant_warp_disabled());
    assert(!bn_gpu_policy_cuda_native_quant_ssm_matvec_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_ssm_prepared_input_enabled());
    assert(bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_K, 2048));
    assert(!bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_K, 2049));
    assert(!bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(2048, 1));
    assert(!bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_DOT_INPUT");
    assert(bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    assert(!bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 1));
    unsetenv("BN_CUDA_DISABLE_Q8_0_SSM_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_Q8_MIXED_PREPARED_INPUT");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREPARED_INPUT");
    assert(bn_gpu_policy_cuda_native_quant_ssm_prepared_input_enabled());
    assert(!bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_K, 2048));
    assert(bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(2048, 1));
    assert(!bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    setenv("BN_CUDA_DISABLE_Q8_0_SSM_PREQ", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_MIXED_PREQ", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT", "1", 1);
    assert(!bn_gpu_policy_cuda_native_quant_ssm_prepared_input_enabled());
    assert(bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_K, 2048));
    assert(!bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(2048, 1));
    assert(!bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT");
    assert(bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(2048, 1, 0));
    unsetenv("BN_CUDA_DISABLE_Q8_0_SSM_PREQ");
    unsetenv("BN_CUDA_ENABLE_Q8_MIXED_PREQ");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT");
    assert(!bn_gpu_policy_cuda_moe_router_fused_topk_enabled(256, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_FUSED_TOPK");
    assert(bn_gpu_policy_cuda_moe_router_fused_topk_enabled(256, 0));
    assert(!bn_gpu_policy_cuda_moe_router_fused_topk_enabled(256, 1));
    assert(bn_gpu_policy_cuda_moe_router_warp_disabled(0));
    assert(!bn_gpu_policy_cuda_moe_router_4warp_enabled(2048));
    assert(!bn_gpu_policy_cuda_moe_router_2warp_enabled(2048));
    assert(!bn_gpu_policy_cuda_moe_router_warp_topk_enabled(256));
    assert(!bn_gpu_policy_cuda_moe_block_prepared_decode_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_NATIVE_QUANT_BLOCK_DECODE");
    setenv("BN_CUDA_DISABLE_Q8_MOE_Q8X", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_block_prepared_decode_enabled());
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_Q8X");
    assert(!bn_gpu_policy_cuda_moe_all_active_two_fast_enabled(0));
    assert(!bn_gpu_policy_cuda_moe_prepared_dot_enabled(1, 1, 1, 2048,
                                                       2048));
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_DOT");
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_DOT_INPUT", "1", 1);
    assert(bn_gpu_policy_cuda_moe_prepared_dot_enabled(0, 0, 1, 2048,
                                                      2048));
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_DOT_INPUT");
    setenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOT_INPUT", "1", 1);
    assert(bn_gpu_policy_cuda_moe_prepared_dot_enabled(0, 0, 0, 2048,
                                                      2048));
    unsetenv("BN_CUDA_ENABLE_MOE_ASYMMETRIC_KQUANT_DOT_INPUT");
    setenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2", "1", 1);
    assert(bn_gpu_policy_cuda_moe_prepared_dot_enabled(0, 0, 1, 2048,
                                                      2048));
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2");
    setenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT", "1", 1);
    assert(bn_gpu_policy_cuda_moe_prepared_dot_enabled(0, 0, 0, 2048,
                                                      2048));
    unsetenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT");
    setenv("BN_CUDA_DISABLE_MOE_Q4K_Q8K_DOT", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_prepared_dot_enabled(1, 1, 1, 2048,
                                                       2048));
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_Q8K_DOT");
    assert(bn_gpu_policy_cuda_moe_internal_profile_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_internal_profile_enabled(0));
    assert(!bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(1, 1));
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_4row_disabled());
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_MOE_ASYMMETRIC_KQUANT_FIXED");
    assert(!bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(1, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_ASYMMETRIC_KQUANT_GATEUP_4ROW");
    setenv("BN_CUDA_DISABLE_MOE_Q4K_ALL2_FIXED", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(1, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_ALL2_FIXED");
    assert(bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(1, 1));
    setenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW", "1", 1);
    assert(bn_gpu_policy_cuda_moe_gateup_prepared_4row_disabled());
    unsetenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW");
    assert(!bn_gpu_policy_cuda_f16_native_quant_ssm_matvec_enabled());
    assert(!bn_gpu_policy_cuda_f16_native_quant_matvec_enabled());
    assert(bn_gpu_policy_cuda_f16_packed_kquant_matvec_enabled());
    unsetenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_SSM_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_NATIVE_QUANT_MATVEC");
    assert(bn_gpu_policy_cuda_f16_native_quant_ssm_matvec_enabled());
    assert(bn_gpu_policy_cuda_f16_native_quant_matvec_enabled());
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_DOT");
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_NATIVE_DOT");
    unsetenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_DISABLE_SYMMETRIC_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q4K_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_MATVEC4");
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_MATMUL8");
    unsetenv("BN_CUDA_ENABLE_Q4K_MATMUL8");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH");
    unsetenv("BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH");
    unsetenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH");
    assert(bn_gpu_policy_kquant_dot_enabled());
    assert(!bn_gpu_policy_kquant_dot_forced());
    assert(bn_gpu_policy_cuda_symmetric_kquant_pair_matvec_enabled());
    assert(!bn_gpu_policy_kquant_matvec4_enabled(8192));
    assert(bn_gpu_policy_kquant_matvec4_enabled(16384));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_matmul8_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_sharedx_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_batch_sharedx_enabled());
    setenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT", "1", 1);
    setenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_Q4K_PAIR_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_Q4K_MATMUL8", "1", 1);
    assert(!bn_gpu_policy_kquant_dot_enabled());
    assert(bn_gpu_policy_kquant_dot_forced());
    assert(!bn_gpu_policy_cuda_symmetric_kquant_pair_matvec_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_matmul8_enabled());
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_DISABLE_Q4K_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_Q4K_MATMUL8");
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_DOT", "1", 1);
    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_NATIVE_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_SYMMETRIC_KQUANT_PAIR_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_MATVEC4", "1", 1);
    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_MATMUL8", "1", 1);
    setenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH", "1", 1);
    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH", "1", 1);
    assert(!bn_gpu_policy_kquant_dot_enabled());
    assert(bn_gpu_policy_kquant_dot_forced());
    assert(!bn_gpu_policy_cuda_symmetric_kquant_pair_matvec_enabled());
    assert(!bn_gpu_policy_kquant_matvec4_enabled(16384));
    assert(bn_gpu_policy_cuda_asymmetric_kquant_matmul8_enabled());
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_sharedx_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_batch_sharedx_enabled());
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_DOT");
    unsetenv("BN_CUDA_DISABLE_Q6K_DOT");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_DOT");
    unsetenv("BN_CUDA_ENABLE_Q6K_DOT");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_WARP");
    unsetenv("BN_CUDA_ENABLE_Q6K_WARP");
    unsetenv("BN_CUDA_ENABLE_MIXED_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_Q6K_Q4K_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_MIXED_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q6K_Q4K_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT");
    unsetenv("BN_CUDA_ENABLE_Q6K_Q8_1_DOT");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_PREPARED_DOT");
    unsetenv("BN_CUDA_DISABLE_Q6K_Q8_1_DOT");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT_ALL");
    unsetenv("BN_CUDA_ENABLE_Q6K_Q8_1_ALL");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_512_2048");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_512_2048");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_2560_9728");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_LOGITS_1536");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_LOGITS_SMALL");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_LOGITS_SMALL");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_1536");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_1536");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q6K_DOWN_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_F16_DOWN_KQUANT_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_Q6K_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_DOWN_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_Q6K_MATVEC");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATMUL8");
    unsetenv("BN_CUDA_ENABLE_Q6K_MATMUL8");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATMUL4");
    unsetenv("BN_CUDA_DISABLE_Q6K_MATMUL4");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC4");
    unsetenv("BN_CUDA_DISABLE_Q6K_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_BATCH_WARP");
    unsetenv("BN_CUDA_ENABLE_Q6K_BATCH_WARP");
    assert(bn_gpu_policy_cuda_down_kquant_dot_enabled());
    assert(!bn_gpu_policy_cuda_down_kquant_dot_forced());
    assert(!bn_gpu_policy_cuda_down_kquant_warp_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_pair_matvec_enabled(4096));
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_pair_matvec_enabled(5120));
    assert(!bn_gpu_policy_cuda_down_kquant_prepared_dot_enabled(1));
    assert(bn_gpu_policy_cuda_down_kquant_mmvq_enabled(5120, 4096, 0, 0));
    assert(bn_gpu_policy_cuda_down_kquant_mmvq_enabled(512, 2048, 0, 0));
    assert(bn_gpu_policy_cuda_down_kquant_mmvq_enabled(1536, 8960, 0, 0));
    assert(bn_gpu_policy_cuda_down_kquant_mmvq_enabled(2560, 9728, 0, 0));
    assert(bn_gpu_policy_cuda_down_kquant_mmvq_enabled(50000, 1536, 1, 0));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_enabled(5120, 4096, 0, 1));
    assert(bn_gpu_policy_cuda_down_kquant_mmvq_2warp_logits_enabled(50000, 1536, 1));
    assert(bn_gpu_policy_cuda_down_kquant_mmvq_2warp_logits_enabled(50000, 2048, 1));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_2warp_logits_enabled(50000, 1536, 0));
    assert(!bn_gpu_policy_cuda_down_kquant_residual_rmsnorm_fuse_enabled());
    assert(bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(2048, 8192, 0));
    assert(!bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(2048, 8192, 1));
    assert(!bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(2048, 4096, 0));
    assert(!bn_gpu_policy_cuda_down_kquant_matmul8_enabled());
    assert(bn_gpu_policy_cuda_down_kquant_matmul4_enabled());
    assert(bn_gpu_policy_cuda_down_kquant_matvec4_enabled());
    assert(!bn_gpu_policy_cuda_down_kquant_batch_warp_enabled());
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_DOT", "1", 1);
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_DOT", "1", 1);
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_WARP", "1", 1);
    setenv("BN_CUDA_ENABLE_MIXED_KQUANT_PAIR_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_512_2048", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_1536_8960", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2560_9728", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_LOGITS_1536", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_LOGITS_SMALL", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_1536", "1", 1);
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_RESID_RMSNORM_FUSE", "1", 1);
    setenv("BN_CUDA_ENABLE_F16_DOWN_KQUANT_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATMUL8", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATMUL4", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC4", "1", 1);
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_BATCH_WARP", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_dot_enabled());
    assert(bn_gpu_policy_cuda_down_kquant_dot_forced());
    assert(bn_gpu_policy_cuda_down_kquant_warp_enabled());
    assert(bn_gpu_policy_cuda_asymmetric_kquant_pair_matvec_enabled(5120));
    assert(bn_gpu_policy_cuda_down_kquant_prepared_dot_enabled(1));
    assert(!bn_gpu_policy_cuda_down_kquant_prepared_dot_enabled(0));
    setenv("BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT_ALL", "1", 1);
    assert(bn_gpu_policy_cuda_down_kquant_prepared_dot_enabled(0));
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_PREPARED_DOT", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_prepared_dot_enabled(1));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_enabled(512, 2048, 0, 0));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_enabled(1536, 8960, 0, 0));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_enabled(2560, 9728, 0, 0));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_enabled(50000, 1536, 1, 0));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_2warp_logits_enabled(50000, 1536, 1));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_2warp_logits_enabled(50000, 2048, 1));
    assert(bn_gpu_policy_cuda_down_kquant_residual_rmsnorm_fuse_enabled());
    assert(bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(2048, 4096, 0));
    assert(bn_gpu_policy_cuda_down_kquant_matmul8_enabled());
    assert(!bn_gpu_policy_cuda_down_kquant_matmul4_enabled());
    assert(!bn_gpu_policy_cuda_down_kquant_matvec4_enabled());
    assert(bn_gpu_policy_cuda_down_kquant_batch_warp_enabled());
    setenv("BN_CUDA_DISABLE_MIXED_KQUANT_PAIR_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_Q6K_MMVQ", "1", 1);
    assert(!bn_gpu_policy_cuda_asymmetric_kquant_pair_matvec_enabled(4096));
    assert(!bn_gpu_policy_cuda_down_kquant_mmvq_enabled(5120, 4096, 0, 0));
    assert(bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(2048, 4096, 0));
    unsetenv("BN_CUDA_ENABLE_F16_DOWN_KQUANT_MATVEC");
    assert(!bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(2048, 4096, 0));
    setenv("BN_CUDA_ENABLE_F16_Q5K_MATVEC", "1", 1);
    setenv("BN_CUDA_ENABLE_Q6K_DOT", "1", 1);
    assert(bn_gpu_policy_cuda_f16_packed_kquant_matvec_enabled());
    assert(bn_gpu_policy_cuda_down_kquant_dot_forced());
    unsetenv("BN_CUDA_ENABLE_F16_Q5K_MATVEC");
    unsetenv("BN_CUDA_ENABLE_Q6K_DOT");
    unsetenv("BN_CUDA_DISABLE_FUSE_BIAS");
    unsetenv("BN_CUDA_DISABLE_ROPE_FLASH_FUSE");
    unsetenv("BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE");
    unsetenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FLASH_FUSE");
    unsetenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FUSE");
    unsetenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_RMSNORM_FUSE");
    unsetenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_FUSE");
    assert(bn_gpu_policy_cuda_fuse_bias_enabled());
    assert(bn_gpu_policy_cuda_rope_flash_fuse_enabled());
    assert(!bn_gpu_policy_cuda_bias_rope_flash_fuse_enabled());
    assert(bn_gpu_policy_cuda_qk_norm_rope_flash_fuse_enabled());
    assert(bn_gpu_policy_cuda_qk_norm_rope_fuse_enabled());
    assert(bn_gpu_policy_cuda_weighted_add_sigmoid_residual_rmsnorm_fuse_enabled());
    assert(bn_gpu_policy_cuda_weighted_add_sigmoid_residual_fuse_enabled());
    setenv("BN_CUDA_DISABLE_FUSE_BIAS", "1", 1);
    setenv("BN_CUDA_DISABLE_ROPE_FLASH_FUSE", "1", 1);
    setenv("BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FLASH_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_RMSNORM_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_FUSE", "1", 1);
    assert(!bn_gpu_policy_cuda_fuse_bias_enabled());
    assert(!bn_gpu_policy_cuda_rope_flash_fuse_enabled());
    assert(bn_gpu_policy_cuda_bias_rope_flash_fuse_enabled());
    assert(!bn_gpu_policy_cuda_qk_norm_rope_flash_fuse_enabled());
    assert(!bn_gpu_policy_cuda_qk_norm_rope_fuse_enabled());
    assert(!bn_gpu_policy_cuda_weighted_add_sigmoid_residual_rmsnorm_fuse_enabled());
    assert(!bn_gpu_policy_cuda_weighted_add_sigmoid_residual_fuse_enabled());
    unsetenv("BN_CUDA_DEBUG_NAN_VERBOSE");
    unsetenv("BN_CUDA_DISABLE_STREAM_EXEC");
    unsetenv("BN_CUDA_PROFILE");
    unsetenv("BN_CUDA_PROFILE_EVERY");
    unsetenv("BN_CUDA_PROFILE_WALL");
    unsetenv("BN_CUDA_PROFILE_WALL_DETAIL");
    unsetenv("BN_CUDA_PROFILE_WALL_EVERY");
    unsetenv("BN_CUDA_PROFILE_SHAPES");
    unsetenv("BN_CUDA_DEVICE");
    unsetenv("BN_CUDA_DEBUG_EXEC_FAIL");
    unsetenv("BN_CUDA_DEBUG_SYNC_EACH_OP");
    unsetenv("BN_CUDA_DEBUG_NAN");
    unsetenv("BN_CUDA_DUMP_OPS");
    unsetenv("BN_CUDA_DUMP_OPS_EVERY");
    unsetenv("BN_CUDA_DUMP_OPS_LIMIT");
    assert(!bn_gpu_policy_cuda_nan_verbose_debug_enabled());
    assert(bn_gpu_policy_cuda_stream_exec_enabled());
    assert(!bn_gpu_policy_cuda_profile_enabled());
    assert(bn_gpu_policy_cuda_profile_every_or_default(1) == 1);
    assert(!bn_gpu_policy_cuda_wall_profile_enabled());
    assert(bn_gpu_policy_cuda_wall_profile_detail_limit_or_default(0) == 0);
    assert(bn_gpu_policy_cuda_wall_profile_every_or_default(16) == 16);
    assert(!bn_gpu_policy_cuda_profile_shapes_enabled());
    assert(bn_gpu_policy_cuda_device_selector() == NULL);
    assert(!bn_gpu_policy_cuda_exec_fail_debug_enabled());
    assert(!bn_gpu_policy_cuda_sync_each_op_debug_enabled());
    assert(!bn_gpu_policy_cuda_nan_debug_enabled());
    assert(!bn_gpu_policy_cuda_dump_ops_enabled());
    assert(!bn_gpu_policy_cuda_dump_ops_every_enabled());
    assert(bn_gpu_policy_cuda_dump_ops_limit_or_default(256) == 256);
    setenv("BN_CUDA_DEBUG_NAN_VERBOSE", "1", 1);
    setenv("BN_CUDA_DISABLE_STREAM_EXEC", "1", 1);
    setenv("BN_CUDA_PROFILE", "1", 1);
    setenv("BN_CUDA_PROFILE_EVERY", "7", 1);
    setenv("BN_CUDA_PROFILE_WALL", "1", 1);
    setenv("BN_CUDA_PROFILE_WALL_DETAIL", "3", 1);
    setenv("BN_CUDA_PROFILE_WALL_EVERY", "11", 1);
    setenv("BN_CUDA_PROFILE_SHAPES", "1", 1);
    setenv("BN_CUDA_DEVICE", "auto", 1);
    setenv("BN_CUDA_DEBUG_EXEC_FAIL", "1", 1);
    setenv("BN_CUDA_DEBUG_SYNC_EACH_OP", "1", 1);
    setenv("BN_CUDA_DEBUG_NAN", "1", 1);
    setenv("BN_CUDA_DUMP_OPS", "1", 1);
    setenv("BN_CUDA_DUMP_OPS_EVERY", "1", 1);
    setenv("BN_CUDA_DUMP_OPS_LIMIT", "64", 1);
    assert(bn_gpu_policy_cuda_nan_verbose_debug_enabled());
    assert(!bn_gpu_policy_cuda_stream_exec_enabled());
    assert(bn_gpu_policy_cuda_profile_enabled());
    assert(bn_gpu_policy_cuda_profile_every_or_default(1) == 7);
    setenv("BN_CUDA_PROFILE_EVERY", "0", 1);
    assert(bn_gpu_policy_cuda_profile_every_or_default(1) == 1);
    assert(bn_gpu_policy_cuda_wall_profile_enabled());
    assert(bn_gpu_policy_cuda_wall_profile_detail_limit_or_default(0) == 3);
    assert(bn_gpu_policy_cuda_wall_profile_every_or_default(16) == 11);
    setenv("BN_CUDA_PROFILE_WALL_EVERY", "-4", 1);
    assert(bn_gpu_policy_cuda_wall_profile_every_or_default(16) == 16);
    assert(bn_gpu_policy_cuda_profile_shapes_enabled());
    assert(strcmp(bn_gpu_policy_cuda_device_selector(), "auto") == 0);
    assert(bn_gpu_policy_cuda_exec_fail_debug_enabled());
    assert(bn_gpu_policy_cuda_sync_each_op_debug_enabled());
    assert(bn_gpu_policy_cuda_nan_debug_enabled());
    assert(bn_gpu_policy_cuda_dump_ops_enabled());
    assert(bn_gpu_policy_cuda_dump_ops_every_enabled());
    assert(bn_gpu_policy_cuda_dump_ops_limit_or_default(256) == 64);
    unsetenv("BN_CUDA_ENABLE_LOGITS_CACHE");
    unsetenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_LOGITS_ARGMAX");
    unsetenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX");
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536");
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_1WARP16_1536");
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536_UNROLL");
    unsetenv("BN_CUDA_DISABLE_ARGMAX_FAST");
    unsetenv("BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY");
    unsetenv("BN_CUDA_ENABLE_LEGACY_BLOCK_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_Q5_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_LEGACY_BLOCK_WARP");
    unsetenv("BN_CUDA_ENABLE_Q5_WARP");
    unsetenv("BN_CUDA_DISABLE_SYMMETRIC_KQUANT_DOT");
    unsetenv("BN_CUDA_DISABLE_Q4K_DOT");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_DOT");
    unsetenv("BN_CUDA_DISABLE_Q5K_DOT");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP");
    unsetenv("BN_CUDA_DISABLE_Q4K_4WARP");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q4K_4WARP_1536_8960");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q4K_4WARP_2560_9728");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_OUT_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_QKV_MIXED_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q4K_QKV_MIXED_FUSE");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_QK_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_QK_ROPE_CACHE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_4WARP_2048");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_4WARP_2048");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_5WARP_2560");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_5WARP_2560");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE_1792");
    unsetenv("BN_CUDA_ENABLE_Q4K_SPLIT_VALUE_FUSE_1792");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SPLIT_VALUE_FUSE");
    unsetenv("BN_CUDA_DISABLE_Q4K_SPLIT_VALUE_FUSE");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_Q8_1_FAST");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_Q8_1_FAST");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_QWARP4");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_5WARP_2560");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_5WARP_2560");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_GATEUP_2WARP");
    unsetenv("BN_CUDA_DISABLE_Q4K_GATEUP_2WARP");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_WARP");
    unsetenv("BN_CUDA_DISABLE_Q8_WARP");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_DOT");
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_NATIVE_DOT");
    unsetenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_DISABLE_SYMMETRIC_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q4K_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_NATIVE_MATVEC4");
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_MATMUL8");
    unsetenv("BN_CUDA_ENABLE_Q4K_MATMUL8");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH");
    unsetenv("BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH");
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_SHAREDX_BATCH");
    unsetenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_DOT");
    unsetenv("BN_CUDA_DISABLE_Q6K_DOT");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_DOT");
    unsetenv("BN_CUDA_ENABLE_Q6K_DOT");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_WARP");
    unsetenv("BN_CUDA_ENABLE_Q6K_WARP");
    unsetenv("BN_CUDA_ENABLE_MIXED_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_Q6K_Q4K_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_MIXED_KQUANT_PAIR_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q6K_Q4K_PAIR_MATVEC");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT");
    unsetenv("BN_CUDA_ENABLE_Q6K_Q8_1_DOT");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_PREPARED_DOT");
    unsetenv("BN_CUDA_DISABLE_Q6K_Q8_1_DOT");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_PREPARED_DOT_ALL");
    unsetenv("BN_CUDA_ENABLE_Q6K_Q8_1_ALL");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_512_2048");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_512_2048");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_1536_8960");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_1536_8960");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2560_9728");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_2560_9728");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_LOGITS_1536");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_LOGITS_SMALL");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_LOGITS_SMALL");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MMVQ_2WARP_1536");
    unsetenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_1536");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_Q6K_DOWN_RESID_RMSNORM_FUSE");
    unsetenv("BN_CUDA_ENABLE_F16_DOWN_KQUANT_MATVEC");
    unsetenv("BN_CUDA_ENABLE_F16_Q6K_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_DOWN_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_F16_Q6K_MATVEC");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_MATMUL8");
    unsetenv("BN_CUDA_ENABLE_Q6K_MATMUL8");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATMUL4");
    unsetenv("BN_CUDA_DISABLE_Q6K_MATMUL4");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC4");
    unsetenv("BN_CUDA_DISABLE_Q6K_MATVEC4");
    unsetenv("BN_CUDA_ENABLE_DOWN_KQUANT_BATCH_WARP");
    unsetenv("BN_CUDA_ENABLE_Q6K_BATCH_WARP");
    unsetenv("BN_CUDA_DISABLE_FUSE_BIAS");
    unsetenv("BN_CUDA_DISABLE_ROPE_FLASH_FUSE");
    unsetenv("BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE");
    unsetenv("BN_CUDA_DEBUG_NAN_VERBOSE");
    unsetenv("BN_CUDA_DISABLE_STREAM_EXEC");
    unsetenv("BN_CUDA_PROFILE");
    unsetenv("BN_CUDA_PROFILE_EVERY");
    unsetenv("BN_CUDA_PROFILE_WALL");
    unsetenv("BN_CUDA_PROFILE_WALL_DETAIL");
    unsetenv("BN_CUDA_PROFILE_WALL_EVERY");
    unsetenv("BN_CUDA_DEBUG_EXEC_FAIL");
    unsetenv("BN_CUDA_DEBUG_SYNC_EACH_OP");
    unsetenv("BN_CUDA_DEBUG_NAN");
    unsetenv("BN_CUDA_DUMP_OPS");
    unsetenv("BN_CUDA_DUMP_OPS_EVERY");
    unsetenv("BN_CUDA_DUMP_OPS_LIMIT");
    unsetenv("BN_CUDA_DISABLE_PREFILL_MOE_LAYER");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_LAYER");
    unsetenv("BN_CUDA_DEBUG_PREFILL_DENSE_LAYER");
    unsetenv("BN_CUDA_PREFILL_DENSE_PROFILE");
    unsetenv("BN_CUDA_PREFILL_DENSE_PROFILE_EVERY");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    unsetenv("BN_CUDA_DISABLE_PREFILL_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH");
    unsetenv("BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH");
    unsetenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH");
    unsetenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH");
    unsetenv("BN_CUDA_SSM_PROFILE");
    unsetenv("BN_CUDA_DISABLE_SSM_STACKED_PREFILL");
    unsetenv("BN_CUDA_DISABLE_SSM_STREAM_PREFILL");
    unsetenv("BN_CUDA_DISABLE_SSM_PREFILL_INPUT_ALIAS");
    unsetenv("BN_CUDA_DISABLE_SSM_F32_AB_PREFILL");
    unsetenv("BN_CUDA_DISABLE_SSM_PREFILL_SCAN");
    unsetenv("BN_CUDA_DISABLE_SSM_DELTA_128_WARP");
    unsetenv("BN_CUDA_SSM_FFN_PROFILE");
    unsetenv("BN_CUDA_ENABLE_SSM_FFN_GATEUP_F16_OUT");
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_GATEUP_F16_OUT");
    unsetenv("BN_CUDA_ENABLE_PACKED_KQUANT_FUSED_GATEUP");
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_SHARED_KQUANT_NATIVE_DOT");
    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE");
    assert(!bn_gpu_policy_cuda_prefill_moe_layer_disabled());
    assert(!bn_gpu_policy_cuda_prefill_dense_layer_disabled());
    assert(!bn_gpu_policy_cuda_prefill_dense_debug_enabled());
    assert(!bn_gpu_policy_cuda_prefill_dense_profile_enabled());
    assert(bn_gpu_policy_cuda_prefill_dense_profile_every_or_default(36) ==
           36);
    assert(!bn_gpu_policy_cuda_prefill_ssm_layer_disabled());
    assert(!bn_gpu_policy_prefill_ssm_layer_disabled());
    assert(bn_gpu_policy_cuda_prefill_fused_asymmetric_kquant_gateup_batch_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_fused_asymmetric_kquant_gateup_batch_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_profile_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_stacked_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_stream_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_input_alias_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_f32_ab_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_scan_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_delta_128_warp_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_ffn_profile_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_enabled());
    assert(!bn_gpu_policy_backend_opt_in_fused_gateup_enabled());
    assert(bn_gpu_policy_fused_gateup_silu_allowed(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_gpu_policy_fused_gateup_silu_allowed(
        &gpu, BN_GGUF_TENSOR_Q5_K));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_gpu_policy_fused_gateup_silu_allowed(
        &gpu, BN_GGUF_TENSOR_Q5_K));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_gpu_policy_shared_kquant_dot_enabled());
    assert(bn_gpu_policy_shared_expert_gate_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH", "1", 1);
    setenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH", "1", 1);
    assert(!bn_gpu_policy_cuda_prefill_fused_asymmetric_kquant_gateup_batch_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_fused_asymmetric_kquant_gateup_batch_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH");
    unsetenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH");
    setenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP", "1", 1);
    setenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT", "1", 1);
    assert(bn_gpu_policy_backend_opt_in_fused_gateup_enabled());
    assert(!bn_gpu_policy_shared_kquant_dot_enabled());
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");
    setenv("BN_CUDA_DISABLE_PREFILL_MOE_LAYER", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_DENSE_LAYER", "1", 1);
    setenv("BN_CUDA_DEBUG_PREFILL_DENSE_LAYER", "1", 1);
    setenv("BN_CUDA_PREFILL_DENSE_PROFILE", "1", 1);
    setenv("BN_CUDA_PREFILL_DENSE_PROFILE_EVERY", "6", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH",
           "1", 1);
    setenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH",
           "1", 1);
    setenv("BN_CUDA_SSM_PROFILE", "1", 1);
    setenv("BN_CUDA_DISABLE_SSM_STACKED_PREFILL", "1", 1);
    setenv("BN_CUDA_DISABLE_SSM_STREAM_PREFILL", "1", 1);
    setenv("BN_CUDA_DISABLE_SSM_PREFILL_INPUT_ALIAS", "1", 1);
    setenv("BN_CUDA_DISABLE_SSM_F32_AB_PREFILL", "1", 1);
    setenv("BN_CUDA_DISABLE_SSM_PREFILL_SCAN", "1", 1);
    setenv("BN_CUDA_DISABLE_SSM_DELTA_128_WARP", "1", 1);
    setenv("BN_CUDA_SSM_FFN_PROFILE", "1", 1);
    setenv("BN_CUDA_ENABLE_SSM_FFN_GATEUP_F16_OUT", "1", 1);
    setenv("BN_CUDA_ENABLE_PACKED_KQUANT_FUSED_GATEUP", "1", 1);
    setenv("BN_CUDA_DISABLE_SHARED_KQUANT_NATIVE_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE", "1", 1);
    setenv("BN_GPU_DISABLE_FUSED_GATEUP", "1", 1);
    assert(bn_gpu_policy_cuda_prefill_moe_layer_disabled());
    assert(bn_gpu_policy_cuda_prefill_dense_layer_disabled());
    assert(bn_gpu_policy_cuda_prefill_dense_debug_enabled());
    assert(bn_gpu_policy_cuda_prefill_dense_profile_enabled());
    assert(bn_gpu_policy_cuda_prefill_dense_profile_every_or_default(36) ==
           6);
    setenv("BN_CUDA_PREFILL_DENSE_PROFILE_EVERY", "0", 1);
    assert(bn_gpu_policy_cuda_prefill_dense_profile_every_or_default(36) ==
           36);
    assert(bn_gpu_policy_cuda_prefill_ssm_layer_disabled());
    assert(bn_gpu_policy_prefill_ssm_layer_disabled());
    assert(!bn_gpu_policy_cuda_prefill_fused_asymmetric_kquant_gateup_batch_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_fused_asymmetric_kquant_gateup_batch_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH",
           "1", 1);
    assert(!bn_gpu_policy_cuda_prefill_ssm_fused_asymmetric_kquant_gateup_batch_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_profile_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_stacked_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_stream_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_input_alias_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_f32_ab_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_scan_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_delta_128_warp_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_ffn_profile_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_enabled());
    setenv("BN_CUDA_DISABLE_SSM_FFN_GATEUP_F16_OUT", "1", 1);
    assert(!bn_gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_enabled());
    assert(bn_gpu_policy_backend_opt_in_fused_gateup_enabled());
    assert(!bn_gpu_policy_fused_gateup_silu_allowed(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    unsetenv("BN_GPU_DISABLE_FUSED_GATEUP");
    assert(bn_gpu_policy_fused_gateup_silu_allowed(
        &gpu, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_gpu_policy_shared_kquant_dot_enabled());
    assert(!bn_gpu_policy_shared_expert_gate_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_MOE_LAYER");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_LAYER");
    unsetenv("BN_CUDA_DEBUG_PREFILL_DENSE_LAYER");
    unsetenv("BN_CUDA_PREFILL_DENSE_PROFILE");
    unsetenv("BN_CUDA_PREFILL_DENSE_PROFILE_EVERY");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    unsetenv("BN_CUDA_DISABLE_PREFILL_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH");
    unsetenv("BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH");
    unsetenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH");
    unsetenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_ASYMMETRIC_KQUANT_GATEUP_BATCH");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH");
    unsetenv("BN_CUDA_SSM_PROFILE");
    unsetenv("BN_CUDA_DISABLE_SSM_STACKED_PREFILL");
    unsetenv("BN_CUDA_DISABLE_SSM_STREAM_PREFILL");
    unsetenv("BN_CUDA_DISABLE_SSM_PREFILL_INPUT_ALIAS");
    unsetenv("BN_CUDA_DISABLE_SSM_F32_AB_PREFILL");
    unsetenv("BN_CUDA_DISABLE_SSM_PREFILL_SCAN");
    unsetenv("BN_CUDA_DISABLE_SSM_DELTA_128_WARP");
    unsetenv("BN_CUDA_SSM_FFN_PROFILE");
    unsetenv("BN_CUDA_ENABLE_SSM_FFN_GATEUP_F16_OUT");
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_GATEUP_F16_OUT");
    unsetenv("BN_CUDA_ENABLE_PACKED_KQUANT_FUSED_GATEUP");
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
    unsetenv("BN_GPU_DISABLE_FUSED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_SHARED_KQUANT_NATIVE_DOT");
    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE");
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    unsetenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN");
    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    unsetenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN");
    unsetenv("BN_CUDA_ENABLE_MOE_PREFILL");
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    unsetenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN");
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX");
    unsetenv("BN_GPU_DISABLE_FUSED_GATEUP");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_GATEUP");
    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_FFN_DOWN");
    unsetenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN");
    unsetenv("BN_GPU_DISABLE_QKV_SPLIT");
    unsetenv("BN_GPU_DEBUG_QKV_SPLIT");
    unsetenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT");
    unsetenv("BN_GPU_DISABLE_SSM_AB_STACK");
    unsetenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM");
    unsetenv("BN_GPU_DEBUG_FALLBACK");
    unsetenv("BN_GPU_FORCE_GRAPH");
    unsetenv("BN_GPU_FLASH_MIN_KV");
    unsetenv("BN_GPU_FLASH_MAX_KV");
    unsetenv("BN_GPU_CPU_LOGITS");
    unsetenv("BN_GPU_COMPARE_LOGITS");
    unsetenv("BN_GPU_DEBUG_ARGMAX_COMPARE");
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_NATIVE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_NATIVE_FFN_DOWN");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN");
    unsetenv("BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_PREFILL");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_GRAPH");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_FAST_GRAPH");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_ROUTE");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_ALL2_FAST");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOT_PREPARED_INPUT_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_DOT_PREPARED_INPUT_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_ROUTE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREQUANT");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_ALL_ACTIVE_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_ALL2_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_LAYERS");
    unsetenv("BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_DOWN_SKIP_EPS");
    unsetenv("BN_CUDA_ALL2_Q4Q6_DOWN_SKIP_EPS");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_FAST_MOE_GRAPH");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_MOE_ALL2_FAST");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_MOE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_ROUTE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_ROUTE_Q8_1_PREQUANT");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_ALL2_DOWN");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN");
    unsetenv("BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN");
    unsetenv("BN_CUDA_QWEN2MOE_DOWN_SKIP_EPS");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_MOE_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_ATTN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_TO_LAYER");
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER");
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_LAYER");
    unsetenv("BN_GPU_COMPARE_MOE_POS");
    unsetenv("BN_GPU_COMPARE_MOE_INPUT_NORM");
    unsetenv("BN_GPU_COMPARE_MOE_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_ROUTE");
    unsetenv("BN_GPU_COMPARE_MOE_RAW");
    unsetenv("BN_GPU_COMPARE_MOE_MID");
    unsetenv("BN_GPU_COMPARE_MOE_PARTS");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_MID");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_DOWN");
    unsetenv("BN_GPU_COMPARE_MOE_NORM");
    unsetenv("BN_GPU_COMPARE_ATTENTION_LAYER");
    unsetenv("BN_GPU_COMPARE_ATTENTION_POS");
    unsetenv("BN_GPU_COMPARE_GQA_LAYER");
    unsetenv("BN_GPU_COMPARE_GQA_POS");
    unsetenv("BN_GPU_COMPARE_QKV_LAYER");
    unsetenv("BN_GPU_COMPARE_QKV_POS");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_POS");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_POS");
    unsetenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");
    unsetenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN");
    unsetenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN");
    unsetenv("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN_WO");
    unsetenv("BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO");
    assert(!bn_gpu_policy_cuda_prefill_attention_min_tokens_configured());
    assert(!bn_gpu_policy_prefill_attention_min_tokens_configured());
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(16) ==
           16);
    assert(bn_gpu_policy_prefill_attention_min_tokens_or_default(16) == 16);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "9", 1);
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_configured());
    assert(bn_gpu_policy_prefill_attention_min_tokens_configured());
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(16) ==
           9);
    assert(bn_gpu_policy_prefill_attention_min_tokens_or_default(16) == 9);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "0", 1);
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(16) ==
           16);
    assert(bn_gpu_policy_prefill_attention_min_tokens_or_default(16) == 16);
    assert(bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
               256) == 256);
    assert(!bn_gpu_policy_cuda_prefill_gemm_attention_enabled(255, 512));
    assert(bn_gpu_policy_cuda_prefill_gemm_attention_enabled(256, 512));
    assert(!bn_gpu_policy_cuda_prefill_gemm_attention_enabled(513, 512));
    setenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN", "1", 1);
    assert(bn_gpu_policy_cuda_prefill_gemm_attention_enabled(1, 512));
    assert(!bn_gpu_policy_cuda_prefill_gemm_attention_enabled(513, 512));
    unsetenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN");
    setenv("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", "64", 1);
    assert(bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
               256) == 64);
    assert(!bn_gpu_policy_cuda_prefill_gemm_attention_enabled(63, 512));
    assert(bn_gpu_policy_cuda_prefill_gemm_attention_enabled(64, 512));
    setenv("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", "0", 1);
    assert(bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
               256) == 256);
    setenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN", "1", 1);
    assert(!bn_gpu_policy_cuda_prefill_gemm_attention_enabled(256, 512));
    unsetenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN");
    unsetenv("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS");
    assert(bn_gpu_policy_cuda_prefill_attention_wo_enabled());
    assert(bn_gpu_policy_cuda_prefill_qkv_attention_wo_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_ATTN_WO", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO", "1", 1);
    assert(!bn_gpu_policy_cuda_prefill_attention_wo_enabled());
    assert(!bn_gpu_policy_cuda_prefill_qkv_attention_wo_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN_WO");
    unsetenv("BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO");
    assert(bn_gpu_policy_prefill_dense_chain_enabled());
    assert(bn_gpu_policy_prefill_hybrid_chain_enabled());
    assert(bn_gpu_policy_cuda_prefill_attention_enabled());
    assert(bn_gpu_policy_prefill_attention_enabled());
    assert(bn_gpu_policy_prefill_ssm_run_chain_enabled());
    assert(bn_gpu_policy_prefill_ssm_ffn_fuse_allowed());
    assert(!bn_gpu_policy_prefill_moe_chain_debug_enabled());
    assert(!bn_gpu_policy_prefill_hybrid_chain_debug_enabled());
    assert(!bn_gpu_policy_moe_prefill_enabled());
    assert(!bn_gpu_policy_moe_prefill_min_tokens_configured());
    assert(bn_gpu_policy_moe_prefill_min_tokens_or_default(1) == 1);
    assert(bn_gpu_policy_moe_cache_prefill_enabled());
    assert(bn_gpu_policy_moe_prefill_shared_fuse_enabled());
    assert(bn_gpu_policy_moe_route_batch_enabled());
    assert(!bn_gpu_policy_moe_route_batch_debug_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_ATTN", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN", "1", 1);
    setenv("BN_CUDA_DISABLE_SSM_FFN_FUSE", "1", 1);
    setenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN", "1", 1);
    setenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_PREFILL", "1", 1);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "7", 1);
    setenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_BATCH", "1", 1);
    setenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH", "1", 1);
    assert(!bn_gpu_policy_prefill_dense_chain_enabled());
    assert(!bn_gpu_policy_prefill_hybrid_chain_enabled());
    assert(!bn_gpu_policy_cuda_prefill_attention_enabled());
    assert(!bn_gpu_policy_prefill_attention_enabled());
    assert(!bn_gpu_policy_prefill_ssm_run_chain_enabled());
    assert(!bn_gpu_policy_prefill_ssm_ffn_fuse_allowed());
    assert(bn_gpu_policy_prefill_moe_chain_debug_enabled());
    assert(bn_gpu_policy_prefill_hybrid_chain_debug_enabled());
    assert(bn_gpu_policy_moe_prefill_enabled());
    assert(bn_gpu_policy_moe_prefill_min_tokens_configured());
    assert(bn_gpu_policy_moe_prefill_min_tokens_or_default(1) == 7);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "0", 1);
    assert(bn_gpu_policy_moe_prefill_min_tokens_or_default(1) == 1);
    assert(!bn_gpu_policy_moe_cache_prefill_enabled());
    assert(!bn_gpu_policy_moe_prefill_shared_fuse_enabled());
    assert(!bn_gpu_policy_moe_route_batch_enabled());
    assert(bn_gpu_policy_moe_route_batch_debug_enabled());
    assert(!bn_gpu_policy_large_hybrid_attention_enabled());
    assert(!bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled());
    assert(!bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled());
    assert(!bn_gpu_policy_large_hybrid_cpu_attention_safe_forced());
    assert(!bn_gpu_policy_large_hybrid_prefill_enabled());
    assert(!bn_gpu_policy_large_hybrid_prefill_chain_enabled());
    assert(!bn_gpu_policy_large_hybrid_prefill_disabled());
    assert(!bn_gpu_policy_large_hybrid_argmax_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_BATCH");
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN", "1", 1);
    setenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX", "1", 1);
    assert(bn_gpu_policy_large_hybrid_attention_enabled());
    assert(bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled());
    assert(bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_large_hybrid_cpu_attention_safe_forced());
    assert(bn_gpu_policy_large_hybrid_prefill_enabled());
    assert(bn_gpu_policy_large_hybrid_prefill_chain_enabled());
    assert(bn_gpu_policy_large_hybrid_prefill_disabled());
    assert(bn_gpu_policy_large_hybrid_argmax_enabled());
    assert(bn_gpu_policy_fused_gateup_enabled());
    assert(bn_gpu_policy_small_dense_exact_native_fused_gateup_enabled());
    assert(bn_gpu_policy_gateup_split_enabled());
    assert(bn_gpu_policy_small_dense_exact_native_ffn_down_enabled());
    assert(bn_gpu_policy_qkv_split_enabled());
    assert(!bn_gpu_policy_qkv_split_debug_enabled());
    assert(bn_gpu_policy_ssm_qkvz_split_enabled());
    assert(bn_gpu_policy_ssm_ab_stack_enabled());
    assert(!bn_gpu_policy_split_residual_rmsnorm_enabled());
    assert(!bn_gpu_policy_debug_fallback_enabled());
    assert(!bn_gpu_policy_force_graph_enabled());
    assert(bn_gpu_policy_flash_min_kv_or_default(0) == 0);
    assert(bn_gpu_policy_backend_flash_default_enabled(&gpu));
    assert(bn_gpu_policy_backend_flash_max_kv_or_default(&gpu, 0) == 2048);
    assert(bn_gpu_policy_backend_large_graph_native_enabled(&gpu));
    assert(bn_gpu_policy_backend_small_dense_native_enabled(&gpu));
    assert(bn_gpu_policy_backend_all_active_two_kquant_moe_supported(&gpu));
    assert(bn_gpu_policy_backend_cpu_attention_fallback_supported(&gpu));
    assert(bn_gpu_policy_backend_small_dense_exact_native_supported(&gpu));
    assert(bn_gpu_policy_backend_prefill_decode_fallback_supported(&gpu));
    assert(bn_gpu_policy_backend_prefill_chain_supported(&gpu));
    assert(bn_gpu_policy_backend_matvec_fallback_supported(&gpu));
    assert(bn_gpu_policy_backend_dense_batch_prefill_shape_supported(&gpu));
    assert(bn_gpu_policy_backend_lazy_moe_aux_cache_supported(&gpu));
    assert(bn_gpu_policy_backend_native_quant_logits_refine_default_supported(
        &gpu));
    assert(
        bn_gpu_policy_backend_all_active_two_kquant_moe_logits_refine_default_supported(
            &gpu));
    assert(bn_gpu_policy_backend_decode_graph_cache_supported(&gpu));
    assert(bn_gpu_policy_backend_moe_exact_attention_supported(&gpu));
    assert(bn_gpu_policy_backend_ssm_graph_supported(&gpu));
    assert(bn_gpu_policy_backend_large_hybrid_argmax_supported(&gpu));
    assert(bn_gpu_policy_backend_all_active_two_moe_direct_route_supported(
        &gpu));
    assert(bn_gpu_policy_backend_resident_moe_ffn_supported(&gpu));
    assert(bn_gpu_policy_backend_moe_gateup_split_supported(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_backend_flash_default_enabled(&gpu));
    assert(bn_gpu_policy_backend_flash_max_kv_or_default(&gpu, 0) == 0);
    assert(!bn_gpu_policy_backend_large_graph_native_enabled(&gpu));
    assert(!bn_gpu_policy_backend_small_dense_native_enabled(&gpu));
    assert(!bn_gpu_policy_backend_all_active_two_kquant_moe_supported(&gpu));
    assert(!bn_gpu_policy_backend_cpu_attention_fallback_supported(&gpu));
    assert(!bn_gpu_policy_backend_small_dense_exact_native_supported(&gpu));
    assert(!bn_gpu_policy_backend_prefill_decode_fallback_supported(&gpu));
    assert(!bn_gpu_policy_backend_prefill_chain_supported(&gpu));
    assert(!bn_gpu_policy_backend_matvec_fallback_supported(&gpu));
    assert(!bn_gpu_policy_backend_dense_batch_prefill_shape_supported(&gpu));
    assert(!bn_gpu_policy_backend_lazy_moe_aux_cache_supported(&gpu));
    assert(!bn_gpu_policy_backend_native_quant_logits_refine_default_supported(
        &gpu));
    assert(
        !bn_gpu_policy_backend_all_active_two_kquant_moe_logits_refine_default_supported(
            &gpu));
    assert(!bn_gpu_policy_backend_decode_graph_cache_supported(&gpu));
    assert(!bn_gpu_policy_backend_moe_exact_attention_supported(&gpu));
    assert(!bn_gpu_policy_backend_ssm_graph_supported(&gpu));
    assert(!bn_gpu_policy_backend_large_hybrid_argmax_supported(&gpu));
    assert(!bn_gpu_policy_backend_all_active_two_moe_direct_route_supported(
        &gpu));
    assert(!bn_gpu_policy_backend_resident_moe_ffn_supported(&gpu));
    assert(!bn_gpu_policy_backend_moe_gateup_split_supported(&gpu));
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;
    assert(!bn_gpu_policy_backend_flash_default_enabled(&gpu));
    assert(bn_gpu_policy_backend_flash_max_kv_or_default(&gpu, 7) == 7);
    assert(!bn_gpu_policy_backend_prefill_chain_supported(&gpu));
    assert(!bn_gpu_policy_backend_resident_moe_ffn_supported(&gpu));
    assert(!bn_gpu_policy_backend_flash_default_enabled(NULL));
    assert(bn_gpu_policy_backend_flash_max_kv_or_default(NULL, 11) == 11);
    assert(!bn_gpu_policy_backend_prefill_chain_supported(NULL));
    assert(!bn_gpu_policy_backend_resident_moe_ffn_supported(NULL));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_gpu_policy_cpu_logits_enabled());
    assert(!bn_gpu_policy_compare_logits_enabled());
    assert(!bn_gpu_policy_debug_argmax_compare_enabled());
    assert(!bn_gpu_policy_moe_ffn_disabled());
    assert(bn_gpu_policy_moe_router_topk_enabled(1));
    assert(!bn_gpu_policy_moe_router_topk_enabled(0));
    assert(bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(1));
    assert(!bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(0));
    assert(!bn_gpu_policy_moe_router_gpu_enabled());
    assert(bn_gpu_policy_moe_router_diff2_enabled());
    assert(bn_gpu_policy_moe_routed_ffn_batch_enabled());
    assert(bn_gpu_policy_moe_routed_ffn_batch_allowed(0));
    assert(!bn_gpu_policy_moe_routed_ffn_batch_allowed(1));
    assert(!bn_gpu_policy_moe_cpu_actual_override_enabled());
    assert(!bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled());
    assert(!bn_gpu_policy_small_dense_exact_native_disabled());
    assert(!bn_gpu_policy_small_dense_exact_native_ffn_down_requested());
    assert(!bn_gpu_policy_small_dense_prefill_disabled());
    assert(!bn_gpu_policy_native_quant_logits_refine_requested());
    assert(!bn_gpu_policy_native_quant_logits_refine_disabled());
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_NATIVE", "1", 1);
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_NATIVE_FFN_DOWN", "1", 1);
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_small_dense_exact_native_disabled());
    assert(bn_gpu_policy_small_dense_exact_native_ffn_down_requested());
    assert(bn_gpu_policy_native_quant_logits_refine_requested());
    assert(bn_gpu_policy_native_quant_logits_refine_disabled());
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_NATIVE");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_NATIVE_FFN_DOWN");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8", "1", 1);
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL", "1", 1);
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_small_dense_exact_native_disabled());
    assert(bn_gpu_policy_small_dense_exact_native_ffn_down_requested());
    assert(bn_gpu_policy_small_dense_prefill_disabled());
    assert(bn_gpu_policy_native_quant_logits_refine_requested());
    assert(bn_gpu_policy_native_quant_logits_refine_disabled());
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8", "1", 1);
    setenv("BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_PREFILL", "1", 1);
    setenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE", "1", 1);
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_small_dense_exact_native_disabled());
    assert(bn_gpu_policy_small_dense_exact_native_ffn_down_requested());
    assert(bn_gpu_policy_small_dense_prefill_disabled());
    assert(bn_gpu_policy_native_quant_logits_refine_requested());
    assert(bn_gpu_policy_native_quant_logits_refine_disabled());
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8");
    unsetenv("BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_PREFILL");
    unsetenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
    assert(!bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_fast_graph_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_cublas_decode_enabled());
    assert(bn_gpu_policy_cuda_moe_cublas_decode_enabled());
    assert(!bn_gpu_policy_cuda_moe_cublas_decode_debug_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_dot_prepared_input_default_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_route_dot_prepared_input_default_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled());
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_BLOCK_PREPARED_INPUT",
           "1", 1);
    assert(bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled());
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_BLOCK_PREPARED_INPUT");
    assert(!bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layers_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(4));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_all_active_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_enabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_default_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_layer_selected(4));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_default_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(0.25f) ==
           0.25f);
    assert(!bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_exact_attention_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_requested());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_disabled());
    assert(!bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled());
    int route_from_layer = 99;
    int route_to_layer = 99;
    bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(&route_from_layer,
                                                  &route_to_layer);
    assert(route_from_layer == -1);
    assert(route_to_layer == -1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_GRAPH", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CUBLAS_DECODE", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_ROUTE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOT_PREPARED_INPUT_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_DOT_PREPARED_INPUT_DEFAULT", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_LOGITS_REFINE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_MOE_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_ATTN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ROUTE_RESIDENT",
           "1", 1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE", "1", 1);
    setenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_FROM_LAYER", "2", 1);
    setenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_TO_LAYER", "4", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_graph_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cublas_decode_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_dot_prepared_input_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_route_dot_prepared_input_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_attention_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_requested());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled());
    bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(&route_from_layer,
                                                  &route_to_layer);
    assert(route_from_layer == 2);
    assert(route_to_layer == 4);
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_GRAPH");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_ROUTE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOT_PREPARED_INPUT_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_DOT_PREPARED_INPUT_DEFAULT");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_MOE_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_ATTN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_TO_LAYER");
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS",
           "1", 1);
    setenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS", "4-5",
           1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT", "1",
           1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT", "1",
           1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_ALL_ACTIVE_DOWN",
           "1", 1);
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_CACHE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN_DEFAULT",
           "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN", "1",
           1);
    setenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_LAYERS",
           "6", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_DEFAULT",
           "1", 1);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN",
           "1", 1);
    setenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_DOWN_SKIP_EPS", "0.125", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layers_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(4));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(6));
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_all_active_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_layer_selected(6));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_layer_selected(4));
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(0.25f) >
           0.124f);
    assert(bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(0.25f) <
           0.126f);
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ORDERED_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_ALL_ACTIVE_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FLOAT_4ROW_DOWN");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_LAYERS");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_F32_EXACT_4ROW_DOWN");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_DOWN_SKIP_EPS");
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_FAST_GRAPH", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_CUBLAS_DECODE", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE", "1", 1);
    setenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_ALL2_FAST", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q8K_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_ROUTE_Q8K_DEFAULT", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREPARED_INPUT", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS", "1", 1);
    setenv("BN_CUDA_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS", "3-5", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_ALL2_DOWN", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_CACHE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN", "1", 1);
    setenv("BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS", "6", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN", "1", 1);
    setenv("BN_CUDA_ALL2_Q4Q6_DOWN_SKIP_EPS", "0.125", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE", "1", 1);
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE", "1", 1);
    setenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER", "3", 1);
    setenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER", "5", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_graph_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cublas_decode_enabled());
    assert(!bn_gpu_policy_cuda_moe_cublas_decode_enabled());
    assert(bn_gpu_policy_cuda_moe_cublas_decode_debug_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_dot_prepared_input_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_route_dot_prepared_input_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layers_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(4));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(6));
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_all_active_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_layer_selected(6));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_layer_selected(4));
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(0.25f) >
           0.124f);
    assert(bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(0.25f) <
           0.126f);
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_attention_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_requested());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled());
    bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(&route_from_layer,
                                                  &route_to_layer);
    assert(route_from_layer == 3);
    assert(route_to_layer == 5);
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_GRAPH");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_FAST_GRAPH");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_ROUTE");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_ALL2_FAST");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_DOT_PREPARED_INPUT_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_ROUTE_DOT_PREPARED_INPUT_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_ROUTE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREPARED_INPUT");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREQUANT");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_FAST_PREPARED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ALL2_Q4Q6_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_ALL2_DOWN");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_Q6K_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_FLOAT_4ROW_DOWN");
    unsetenv("BN_CUDA_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_LAYERS");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_Q6K_F32_EXACT_4ROW_DOWN");
    unsetenv("BN_CUDA_ALL2_Q4Q6_DOWN_SKIP_EPS");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER");
    assert(!bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled());
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREQUANT", "1", 1);
    assert(bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled());
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_ROUTE_Q8_1_PREQUANT");
    setenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_FAST_MOE_GRAPH", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_MOE_CUBLAS_DECODE", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_MOE_ALL2_FAST", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_MOE_Q8K_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_ROUTE_Q8K_DEFAULT", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_ROUTE_Q8_1_PREQUANT", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_Q8K_GATEUP", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_FAST_Q8K_GATEUP", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_PAIR_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS", "1", 1);
    setenv("BN_CUDA_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS", "7-8", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_ORDERED_DOWN", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_ORDERED_DOWN", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_ALL2_DOWN", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_CACHE", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN", "1", 1);
    setenv("BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS", "9", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN", "1", 1);
    setenv("BN_CUDA_QWEN2MOE_DOWN_SKIP_EPS", "0.0625", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT", "1", 1);
    setenv("BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE", "1", 1);
    setenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE", "1", 1);
    setenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER", "4", 1);
    setenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER", "6", 1);
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_graph_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cublas_decode_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_dot_prepared_input_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_route_dot_prepared_input_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layers_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(7));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(9));
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_all_active_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_enabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_layer_selected(9));
    assert(!bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_layer_selected(7));
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_default_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_down_f32_exact_4row_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(0.25f) >
           0.062f);
    assert(bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(0.25f) <
           0.063f);
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_attention_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_requested());
    assert(bn_gpu_policy_all_active_two_kquant_moe_exact_gpu_route_disabled());
    assert(bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled());
    bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(&route_from_layer,
                                                  &route_to_layer);
    assert(route_from_layer == 4);
    assert(route_to_layer == 6);
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_FAST_MOE_GRAPH");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_MOE_ALL2_FAST");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_MOE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_ROUTE_Q8K_DEFAULT");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_ROUTE_Q8_1_PREQUANT");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_FAST_Q8K_GATEUP");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_PAIR_DOWN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_QWEN2MOE_Q6K_PAIR_DOWN_F32_LAYERS");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_ORDERED_DOWN");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_ALL2_DOWN");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_Q6K_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_FLOAT_4ROW_DOWN");
    unsetenv("BN_CUDA_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_LAYERS");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN_DEFAULT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6K_F32_EXACT_4ROW_DOWN");
    unsetenv("BN_CUDA_QWEN2MOE_DOWN_SKIP_EPS");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_GPU_ROUTE");
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
    unsetenv("BN_GPU_COMPARE_ATTENTION_LAYER");
    unsetenv("BN_GPU_COMPARE_ATTENTION_POS");
    unsetenv("BN_GPU_COMPARE_GQA_LAYER");
    unsetenv("BN_GPU_COMPARE_GQA_POS");
    unsetenv("BN_GPU_COMPARE_QKV_LAYER");
    unsetenv("BN_GPU_COMPARE_QKV_POS");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_POS");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_POS");
    assert(!bn_gpu_policy_moe_compare_layer_selected(3, 7));
    assert(!bn_gpu_policy_moe_compare_input_norm_enabled());
    assert(!bn_gpu_policy_moe_compare_actual_enabled());
    assert(!bn_gpu_policy_moe_compare_route_enabled());
    assert(!bn_gpu_policy_moe_compare_raw_enabled());
    assert(!bn_gpu_policy_moe_compare_mid_enabled());
    assert(!bn_gpu_policy_moe_compare_parts_enabled());
    assert(!bn_gpu_policy_moe_compare_shared_mid_enabled());
    assert(!bn_gpu_policy_moe_compare_shared_down_enabled());
    assert(!bn_gpu_policy_moe_compare_norm_enabled());
    assert(bn_gpu_policy_compare_attention_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_attention_pos_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_gqa_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_gqa_pos_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_qkv_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_qkv_pos_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_ffn_down_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_ffn_down_pos_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_ffn_state_layer_or_default(-1) == -1);
    assert(bn_gpu_policy_compare_ffn_state_pos_or_default(-1) == -1);
    assert(!bn_gpu_policy_moe_shared_cpu_fallback_enabled(1));
    assert(bn_gpu_policy_moe_gateup_split_enabled(1));
    assert(!bn_gpu_policy_moe_gateup_split_enabled(0));
    assert(!bn_gpu_policy_moe_route_profile_enabled());
    assert(bn_gpu_policy_moe_route_profile_every_or_default(28) == 28);
    setenv("BN_GPU_DISABLE_FUSED_GATEUP", "1", 1);
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_GATEUP", "1", 1);
    setenv("BN_GPU_DISABLE_GATEUP_SPLIT", "1", 1);
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_FFN_DOWN", "1", 1);
    setenv("BN_GPU_DISABLE_QKV_SPLIT", "1", 1);
    setenv("BN_GPU_DEBUG_QKV_SPLIT", "1", 1);
    setenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT", "1", 1);
    setenv("BN_GPU_DISABLE_SSM_AB_STACK", "1", 1);
    setenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM", "1", 1);
    setenv("BN_GPU_DEBUG_FALLBACK", "1", 1);
    setenv("BN_GPU_FORCE_GRAPH", "1", 1);
    setenv("BN_GPU_FLASH_MIN_KV", "32", 1);
    setenv("BN_GPU_FLASH_MAX_KV", "1024", 1);
    setenv("BN_GPU_CPU_LOGITS", "1", 1);
    setenv("BN_GPU_COMPARE_LOGITS", "1", 1);
    setenv("BN_GPU_DEBUG_ARGMAX_COMPARE", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_FFN", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK", "1", 1);
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE", "1", 1);
    setenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_LAYER", "3", 1);
    setenv("BN_GPU_COMPARE_MOE_INPUT_NORM", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ACTUAL", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ROUTE", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_RAW", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_PARTS", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_DOWN", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_NORM", "1", 1);
    setenv("BN_GPU_COMPARE_ATTENTION_LAYER", "10", 1);
    setenv("BN_GPU_COMPARE_ATTENTION_POS", "11", 1);
    setenv("BN_GPU_COMPARE_GQA_LAYER", "12", 1);
    setenv("BN_GPU_COMPARE_GQA_POS", "13", 1);
    setenv("BN_GPU_COMPARE_QKV_LAYER", "14", 1);
    setenv("BN_GPU_COMPARE_QKV_POS", "15", 1);
    setenv("BN_GPU_COMPARE_FFN_DOWN_LAYER", "16", 1);
    setenv("BN_GPU_COMPARE_FFN_DOWN_POS", "17", 1);
    setenv("BN_GPU_COMPARE_FFN_STATE_LAYER", "18", 1);
    setenv("BN_GPU_COMPARE_FFN_STATE_POS", "19", 1);
    setenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    setenv("BN_GPU_MOE_ROUTE_PROFILE", "1", 1);
    setenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY", "5", 1);
    assert(!bn_gpu_policy_fused_gateup_enabled());
    assert(!bn_gpu_policy_small_dense_exact_native_fused_gateup_enabled());
    assert(!bn_gpu_policy_gateup_split_enabled());
    assert(!bn_gpu_policy_small_dense_exact_native_ffn_down_requested());
    assert(!bn_gpu_policy_qkv_split_enabled());
    assert(bn_gpu_policy_qkv_split_debug_enabled());
    assert(!bn_gpu_policy_ssm_qkvz_split_enabled());
    assert(!bn_gpu_policy_ssm_ab_stack_enabled());
    assert(bn_gpu_policy_split_residual_rmsnorm_enabled());
    assert(bn_gpu_policy_debug_fallback_enabled());
    assert(bn_gpu_policy_force_graph_enabled());
    assert(bn_gpu_policy_flash_min_kv_or_default(0) == 32);
    assert(bn_gpu_policy_backend_flash_max_kv_or_default(&gpu, 0) == 1024);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_gpu_policy_backend_flash_max_kv_or_default(&gpu, 0) == 1024);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_gpu_policy_cpu_logits_enabled());
    assert(bn_gpu_policy_compare_logits_enabled());
    assert(bn_gpu_policy_debug_argmax_compare_enabled());
    assert(bn_gpu_policy_moe_ffn_disabled());
    assert(!bn_gpu_policy_moe_router_topk_enabled(1));
    assert(!bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(1));
    assert(bn_gpu_policy_moe_router_gpu_enabled());
    assert(!bn_gpu_policy_moe_router_diff2_enabled());
    assert(!bn_gpu_policy_moe_routed_ffn_batch_enabled());
    assert(bn_gpu_policy_moe_routed_ffn_batch_allowed(1));
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU", "1", 1);
    assert(!bn_gpu_policy_moe_router_gpu_enabled());
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH", "1", 1);
    assert(!bn_gpu_policy_moe_routed_ffn_batch_allowed(0));
    assert(!bn_gpu_policy_moe_routed_ffn_batch_allowed(1));
    assert(bn_gpu_policy_moe_cpu_actual_override_enabled());
    assert(bn_gpu_policy_moe_compare_layer_selected(3, 7));
    assert(!bn_gpu_policy_moe_compare_layer_selected(4, 7));
    setenv("BN_GPU_COMPARE_MOE_POS", "7", 1);
    assert(bn_gpu_policy_moe_compare_layer_selected(3, 7));
    assert(!bn_gpu_policy_moe_compare_layer_selected(3, 8));
    assert(bn_gpu_policy_moe_compare_input_norm_enabled());
    assert(bn_gpu_policy_moe_compare_actual_enabled());
    assert(bn_gpu_policy_moe_compare_route_enabled());
    assert(bn_gpu_policy_moe_compare_raw_enabled());
    assert(bn_gpu_policy_moe_compare_mid_enabled());
    assert(bn_gpu_policy_moe_compare_parts_enabled());
    assert(bn_gpu_policy_moe_compare_shared_mid_enabled());
    assert(bn_gpu_policy_moe_compare_shared_down_enabled());
    assert(bn_gpu_policy_moe_compare_norm_enabled());
    assert(bn_gpu_policy_compare_attention_layer_or_default(-1) == 10);
    assert(bn_gpu_policy_compare_attention_pos_or_default(-1) == 11);
    assert(bn_gpu_policy_compare_gqa_layer_or_default(-1) == 12);
    assert(bn_gpu_policy_compare_gqa_pos_or_default(-1) == 13);
    assert(bn_gpu_policy_compare_qkv_layer_or_default(-1) == 14);
    assert(bn_gpu_policy_compare_qkv_pos_or_default(-1) == 15);
    assert(bn_gpu_policy_compare_ffn_down_layer_or_default(-1) == 16);
    assert(bn_gpu_policy_compare_ffn_down_pos_or_default(-1) == 17);
    assert(bn_gpu_policy_compare_ffn_state_layer_or_default(-1) == 18);
    assert(bn_gpu_policy_compare_ffn_state_pos_or_default(-1) == 19);
    assert(!bn_gpu_policy_moe_shared_cpu_fallback_enabled(0));
    assert(bn_gpu_policy_moe_shared_cpu_fallback_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    assert(!bn_gpu_policy_moe_shared_cpu_fallback_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT", "1", 1);
    assert(!bn_gpu_policy_moe_gateup_split_enabled(1));
    assert(bn_gpu_policy_moe_route_profile_enabled());
    assert(bn_gpu_policy_moe_route_profile_every_or_default(28) == 5);
    setenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY", "0", 1);
    assert(bn_gpu_policy_moe_route_profile_every_or_default(28) == 28);
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN");
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX");
    unsetenv("BN_GPU_DISABLE_FUSED_GATEUP");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_GATEUP");
    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_FFN_DOWN");
    unsetenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN");
    unsetenv("BN_GPU_DISABLE_QKV_SPLIT");
    unsetenv("BN_GPU_DEBUG_QKV_SPLIT");
    unsetenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT");
    unsetenv("BN_GPU_DISABLE_SSM_AB_STACK");
    unsetenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM");
    unsetenv("BN_GPU_DEBUG_FALLBACK");
    unsetenv("BN_GPU_FORCE_GRAPH");
    unsetenv("BN_GPU_FLASH_MIN_KV");
    unsetenv("BN_GPU_FLASH_MAX_KV");
    unsetenv("BN_GPU_CPU_LOGITS");
    unsetenv("BN_GPU_COMPARE_LOGITS");
    unsetenv("BN_GPU_DEBUG_ARGMAX_COMPARE");
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT");
    setenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    assert(!bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(1));
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_LAYER");
    unsetenv("BN_GPU_COMPARE_MOE_POS");
    unsetenv("BN_GPU_COMPARE_MOE_INPUT_NORM");
    unsetenv("BN_GPU_COMPARE_MOE_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_ROUTE");
    unsetenv("BN_GPU_COMPARE_MOE_RAW");
    unsetenv("BN_GPU_COMPARE_MOE_MID");
    unsetenv("BN_GPU_COMPARE_MOE_PARTS");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_MID");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_DOWN");
    unsetenv("BN_GPU_COMPARE_MOE_NORM");
    unsetenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    unsetenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN");
    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    unsetenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN");
    unsetenv("BN_CUDA_ENABLE_MOE_PREFILL");
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    unsetenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN");
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX");
    unsetenv("BN_GPU_DISABLE_FUSED_GATEUP");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_GATEUP");
    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_DISABLE_FFN_DOWN");
    unsetenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN");
    unsetenv("BN_GPU_DISABLE_QKV_SPLIT");
    unsetenv("BN_GPU_DEBUG_QKV_SPLIT");
    unsetenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT");
    unsetenv("BN_GPU_DISABLE_SSM_AB_STACK");
    unsetenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM");
    unsetenv("BN_GPU_DEBUG_FALLBACK");
    unsetenv("BN_GPU_FORCE_GRAPH");
    unsetenv("BN_GPU_FLASH_MIN_KV");
    unsetenv("BN_GPU_FLASH_MAX_KV");
    unsetenv("BN_GPU_CPU_LOGITS");
    unsetenv("BN_GPU_COMPARE_LOGITS");
    unsetenv("BN_GPU_DEBUG_ARGMAX_COMPARE");
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_LAYER");
    unsetenv("BN_GPU_COMPARE_MOE_POS");
    unsetenv("BN_GPU_COMPARE_MOE_INPUT_NORM");
    unsetenv("BN_GPU_COMPARE_MOE_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_ROUTE");
    unsetenv("BN_GPU_COMPARE_MOE_RAW");
    unsetenv("BN_GPU_COMPARE_MOE_MID");
    unsetenv("BN_GPU_COMPARE_MOE_PARTS");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_MID");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_DOWN");
    unsetenv("BN_GPU_COMPARE_MOE_NORM");
    unsetenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");
    unsetenv("BN_GPU_MOE_CACHE_RESERVE_MB");
    assert(bn_gpu_policy_backend_placement(&gpu) == BN_BACKEND_CUDA);
    assert(bn_gpu_policy_cuda_cublas_cache_max_mb(128, 0) == 128);
    assert(bn_gpu_policy_cuda_cublas_cache_max_mb(128, 1) == 512);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q8_0, 0, 0) == 128);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q4_K, 0, 0) == 512);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q6_K, 0, 1) == 0);
    assert(bn_gpu_policy_cuda_down_kquant_f16_cache_adds_f32_down_cache());
    assert(bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16) == 128 * sizeof(uint16_t));
    assert(bn_gpu_policy_moe_down_aux_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16) == 128 * sizeof(uint16_t));
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 16) == 0);
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 32) == 256 * sizeof(uint16_t));
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 32) == 256 * sizeof(uint16_t));
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q4_K, 8, 16) == 0);
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q4_K, 8, 32) == 256 * sizeof(uint16_t));
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_I2_S, 8, 32) == 0);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_supported(
        BN_GGUF_TENSOR_Q4_K, 32));
    assert(bn_gpu_policy_cuda_cublas_aux_cache_supported(
        BN_GGUF_TENSOR_Q6_K, 32));
    assert(!bn_gpu_policy_cuda_cublas_aux_cache_supported(
        BN_GGUF_TENSOR_Q4_0, 32));
    assert(!bn_gpu_policy_cuda_cublas_aux_cache_supported(
        BN_GGUF_TENSOR_IQ4_XS, 32));
    assert(!bn_gpu_policy_cuda_cublas_aux_cache_supported(
        BN_GGUF_TENSOR_Q4_K, 31));
    assert(!bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q4_K, 8, 16));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_gpu_policy_backend_placement(&gpu) == BN_BACKEND_METAL);
    assert(!bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16));
    gpu.kind = BN_GPU_BACKEND_WEBGPU;
    assert(bn_gpu_policy_backend_placement(&gpu) == BN_BACKEND_WEBGPU);
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;
    assert(bn_gpu_policy_backend_placement(&gpu) == BN_BACKEND_GPU_UNKNOWN);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    setenv("BN_CUDA_CUBLAS_CACHE_MAX_MB", "7", 1);
    assert(bn_gpu_policy_cuda_cublas_cache_max_mb(128, 1) == 7);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q6_K, 1, 0) == 7);
    setenv("BN_CUDA_DISABLE_CUBLAS_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_CUBLAS_F16_CACHE", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_F16_KQUANT_F32_DOWN_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_cublas_matmul_enabled());
    assert(!bn_gpu_policy_cuda_cublas_aux_cache_supported(
        BN_GGUF_TENSOR_Q4_K, 32));
    assert(!bn_gpu_policy_cuda_down_kquant_cublas_f16_cache_enabled());
    assert(!bn_gpu_policy_cuda_down_kquant_f16_cache_adds_f32_down_cache());
    assert(!bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16));
    assert(!bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 16));
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_CUBLAS_F16_CACHE");
    setenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_cublas_f16_cache_enabled());
    unsetenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16");
    setenv("BN_CUDA_DISABLE_DOWN_KQUANT_CUBLAS_F16_CACHE", "1", 1);
    unsetenv("BN_CUDA_DISABLE_MOE_F16_KQUANT_F32_DOWN_CACHE");
    setenv("BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_f16_cache_adds_f32_down_cache());
    unsetenv("BN_CUDA_DISABLE_CUBLAS_MATMUL");
    unsetenv("BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE");
    assert(bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16) == 128 * sizeof(float));
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 32) == 256 * sizeof(float));
    assert(bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 32) == 256 * sizeof(float));
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_CUBLAS_F16_CACHE");
    setenv("BN_CUDA_CUBLAS_CACHE_MAX_MB", "1", 1);
    assert(!bn_gpu_policy_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 2048, 1024));
    unsetenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    setenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q6_K, 1, 0) == 0);
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_f16_cache_adds_f32_down_cache());
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_down_kquant_f16_cache_adds_f32_down_cache());
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    assert(bn_gpu_policy_moe_auto_resident_enabled());
    assert(!bn_gpu_policy_duplicate_moe_cache_enabled());
    setenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT", "1", 1);
    setenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_auto_resident_enabled());
    assert(bn_gpu_policy_duplicate_moe_cache_enabled());
    setenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE", "1", 1);
    assert(!bn_gpu_policy_duplicate_moe_cache_enabled());
    unsetenv("BN_METAL_ENABLE_MMAP_ZERO_COPY");
    assert(bn_gpu_policy_moe_cache_reserve_bytes() ==
           4096ull * 1024ull * 1024ull);
    setenv("BN_GPU_MOE_CACHE_RESERVE_MB", "123", 1);
    assert(bn_gpu_policy_moe_cache_reserve_bytes() ==
           123ull * 1024ull * 1024ull);
    setenv("BN_GPU_MOE_CACHE_RESERVE_MB", "bad", 1);
    assert(bn_gpu_policy_moe_cache_reserve_bytes() == 0);
    setenv("BN_GPU_MOE_CACHE_RESERVE_MB", "0", 1);
    assert(bn_gpu_policy_moe_cache_reserve_bytes() == 0);
    setenv("BN_GPU_MOE_CACHE_RESERVE_MB", "-1", 1);
    assert(bn_gpu_policy_moe_cache_reserve_bytes() == 0);
    setenv("BN_GPU_MOE_CACHE_RESERVE_MB", "18446744073709551615", 1);
    assert(bn_gpu_policy_moe_cache_reserve_bytes() == SIZE_MAX);
    unsetenv("BN_GPU_MOE_CACHE_RESERVE_MB");
    assert(!bn_gpu_policy_metal_mmap_zero_copy_enabled());
    setenv("BN_METAL_ENABLE_MMAP_ZERO_COPY", "1", 1);
    assert(bn_gpu_policy_metal_mmap_zero_copy_enabled());
    unsetenv("BN_METAL_SHARED_WEIGHTS");
    assert(!bn_gpu_policy_metal_shared_weights_enabled());
    setenv("BN_METAL_SHARED_WEIGHTS", "1", 1);
    assert(bn_gpu_policy_metal_shared_weights_enabled());
    unsetenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    assert(!bn_gpu_policy_metal_specialized_native_quant_enabled());
    assert(!bn_gpu_policy_specialized_native_quant_decode_path_enabled());
    assert(!bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
        BN_GGUF_TENSOR_Q6_K, 256, 1, 1));
    setenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT", "1", 1);
    assert(bn_gpu_policy_metal_specialized_native_quant_enabled());
    assert(bn_gpu_policy_specialized_native_quant_decode_path_enabled());
    assert(bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
        BN_GGUF_TENSOR_Q6_K, 256, 1, 1));
    assert(!bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
        BN_GGUF_TENSOR_Q4_K, 256, 1, 1));
    assert(!bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
        BN_GGUF_TENSOR_Q6_K, 128, 1, 1));
    assert(!bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
        BN_GGUF_TENSOR_Q6_K, 256, 0, 1));
    assert(!bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
        BN_GGUF_TENSOR_Q6_K, 256, 1, 0));
    unsetenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    setenv("BN_METAL_ENABLE_Q6_Q8K", "1", 1);
    assert(bn_gpu_policy_metal_specialized_native_quant_enabled());
    assert(bn_gpu_policy_specialized_native_quant_decode_path_enabled());
    unsetenv("BN_METAL_Q8_BARRIERS");
    assert(!bn_gpu_policy_metal_native_quant_barriers_enabled());
    setenv("BN_METAL_Q8_BARRIERS", "1", 1);
    assert(bn_gpu_policy_metal_native_quant_barriers_enabled());
    unsetenv("BN_METAL_CPU_ORDER_RMSNORM");
    assert(!bn_gpu_policy_metal_cpu_order_rmsnorm_enabled());
    setenv("BN_METAL_CPU_ORDER_RMSNORM", "1", 1);
    assert(bn_gpu_policy_metal_cpu_order_rmsnorm_enabled());
    unsetenv("BN_METAL_FULL_BARRIERS");
    unsetenv("BN_METAL_ENABLE_BARRIERS");
    unsetenv("BN_METAL_DISABLE_BARRIERS");
    assert(!bn_gpu_policy_metal_full_barriers_enabled());
    assert(!bn_gpu_policy_metal_barriers_enabled());
    assert(bn_gpu_policy_metal_barriers_disabled());
    bn_gpu_policy_apply_metal_barrier_disable_override();
    assert(bn_gpu_policy_metal_barriers_disabled());
    unsetenv("BN_METAL_DISABLE_BARRIERS");
    setenv("BN_METAL_ENABLE_BARRIERS", "1", 1);
    assert(!bn_gpu_policy_metal_full_barriers_enabled());
    assert(bn_gpu_policy_metal_barriers_enabled());
    assert(!bn_gpu_policy_metal_barriers_disabled());
    setenv("BN_METAL_FULL_BARRIERS", "1", 1);
    assert(bn_gpu_policy_metal_full_barriers_enabled());
    assert(bn_gpu_policy_metal_barriers_enabled());
    setenv("BN_METAL_DISABLE_BARRIERS", "1", 1);
    assert(bn_gpu_policy_metal_barriers_disabled());
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TO_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TAIL_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY");
    unsetenv("BN_METAL_DISABLE_SMALL_DENSE_EXACT_NATIVE_DEFAULT");
    unsetenv("BN_GPU_Q4_Q8");
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    unsetenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT");
    unsetenv("BN_METAL_Q4_PREPARED");
    unsetenv("BN_METAL_PRIVATE_WEIGHTS");
    assert(!bn_gpu_policy_metal_small_dense_exact_native_enabled());
    assert(!bn_gpu_policy_small_dense_exact_native_attn_only_enabled());
    assert(!bn_gpu_policy_small_dense_exact_native_ffn_only_enabled());
    assert(bn_gpu_policy_small_dense_exact_native_from_layer_or_default(40) ==
           -1);
    assert(bn_gpu_policy_small_dense_exact_native_to_layer_or_default(40, 0) ==
           -1);
    assert(!bn_gpu_policy_metal_native_quant_prepared_enabled());
    assert(!bn_gpu_policy_small_dense_native_quant_prepared_layer_default_enabled());
    assert(!bn_gpu_policy_metal_native_quant_prepared_upload_enabled());
    assert(bn_gpu_policy_webgpu_repacked_buffer_supported(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_gpu_policy_webgpu_repacked_buffer_supported(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_gpu_policy_webgpu_repacked_bias_supported(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_gpu_policy_webgpu_repacked_bias_supported(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_gpu_policy_metal_repacked_buffer_supported(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_gpu_policy_metal_repacked_buffer_supported(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_gpu_policy_metal_repacked_buffer_type(
        BN_GGUF_TENSOR_Q4_0) == BN_GGUF_TENSOR_Q4_0);
    assert(bn_gpu_policy_metal_repacked_buffer_type(
        BN_GGUF_TENSOR_Q8_0) == -1);
    assert(!bn_gpu_policy_metal_prepared_stacked_upload_blocked(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_gpu_policy_metal_exact_native_matvec_supported(
        BN_GGUF_TENSOR_Q4_0, 0, 0, 1, 1, 0));
    bn_gpu_policy_metal_apply_small_dense_exact_native_default();
    assert(bn_gpu_policy_metal_small_dense_exact_native_enabled());
    assert(bn_gpu_policy_metal_exact_native_matvec_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 0, 1, 1, 0));
    assert(bn_gpu_policy_metal_exact_native_matvec_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 1, 1, 0, 1));
    assert(bn_gpu_policy_metal_exact_native_graph_path_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 0, 0, 1, 1));
    assert(bn_gpu_policy_metal_exact_native_graph_path_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 1, 1, 1, 1));
    assert(!bn_gpu_policy_metal_exact_native_graph_path_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 1, 0, 1, 1));
    assert(!bn_gpu_policy_metal_exact_native_graph_path_supported(
        BN_GGUF_TENSOR_Q6_K, 1, 0, 0, 1, 1));
    assert(!bn_gpu_policy_metal_exact_native_graph_path_supported(
        BN_GGUF_TENSOR_Q4_0, 0, 0, 0, 1, 1));
    assert(!bn_gpu_policy_metal_exact_native_graph_path_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 0, 0, 0, 1));
    assert(!bn_gpu_policy_metal_exact_native_graph_path_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 0, 0, 1, 0));
    assert(!bn_gpu_policy_metal_exact_native_matvec_supported(
        BN_GGUF_TENSOR_Q6_K, 1, 0, 1, 1, 0));
    assert(!bn_gpu_policy_metal_exact_native_matvec_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 0, 0, 1, 0));
    assert(!bn_gpu_policy_metal_exact_native_matvec_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 0, 1, 0, 0));
    assert(!bn_gpu_policy_metal_exact_native_matvec_supported(
        BN_GGUF_TENSOR_Q4_0, 1, 1, 1, 0, 0));
    assert(getenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER") != NULL);
    assert(bn_gpu_policy_small_dense_exact_native_from_layer_or_default(40) ==
           0);
    assert(bn_gpu_policy_small_dense_exact_native_to_layer_or_default(40, 0) ==
           6);
    assert(!bn_gpu_policy_small_dense_exact_native_attn_only_enabled());
    assert(bn_gpu_policy_small_dense_exact_native_ffn_only_enabled());
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY");
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER", "10", 1);
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TO_LAYER", "20", 1);
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY", "1", 1);
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY", "1", 1);
    assert(bn_gpu_policy_small_dense_exact_native_from_layer_or_default(40) ==
           10);
    assert(bn_gpu_policy_small_dense_exact_native_to_layer_or_default(40, 0) ==
           20);
    assert(bn_gpu_policy_small_dense_exact_native_attn_only_enabled());
    assert(bn_gpu_policy_small_dense_exact_native_ffn_only_enabled());
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TO_LAYER");
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TAIL_NATIVE", "4", 1);
    assert(bn_gpu_policy_small_dense_exact_native_to_layer_or_default(40, 0) ==
           35);
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TAIL_NATIVE", "100", 1);
    assert(bn_gpu_policy_small_dense_exact_native_to_layer_or_default(40, 0) ==
           -1);
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TAIL_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY");
    setenv("BN_GPU_Q4_Q8", "1", 1);
    setenv("BN_GPU_Q4_Q8_FROM_LAYER", "12", 1);
    setenv("BN_GPU_Q4_Q8_TO_LAYER", "18", 1);
    assert(bn_gpu_policy_small_dense_exact_native_from_layer_or_default(40) ==
           12);
    assert(bn_gpu_policy_small_dense_exact_native_to_layer_or_default(40, 0) ==
           18);
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    bn_gpu_policy_apply_native_quant_prepared_override();
    assert(bn_gpu_policy_metal_native_quant_prepared_enabled());
    unsetenv("BN_METAL_Q4_PREPARED");
    setenv("BN_METAL_Q4_PREPARED", "1", 1);
    assert(bn_gpu_policy_metal_native_quant_prepared_enabled());
    assert(bn_gpu_policy_small_dense_native_quant_prepared_layer_default_enabled());
    assert(bn_gpu_policy_metal_native_quant_prepared_upload_enabled());
    assert(bn_gpu_policy_metal_prepared_stacked_upload_blocked(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_gpu_policy_metal_prepared_stacked_upload_blocked(
        BN_GGUF_TENSOR_Q8_0));
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER", "1", 1);
    assert(!bn_gpu_policy_metal_native_quant_prepared_upload_enabled());
    assert(!bn_gpu_policy_metal_prepared_stacked_upload_blocked(
        BN_GGUF_TENSOR_Q4_0));
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER", "0", 1);
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY", "1", 1);
    assert(!bn_gpu_policy_metal_native_quant_prepared_upload_enabled());
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY");
    setenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY", "1", 1);
    assert(!bn_gpu_policy_metal_native_quant_prepared_upload_enabled());
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TO_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_TAIL_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY");
    unsetenv("BN_GPU_Q4_Q8");
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    setenv("BN_METAL_DISABLE_SMALL_DENSE_EXACT_NATIVE_DEFAULT", "1", 1);
    bn_gpu_policy_metal_apply_small_dense_exact_native_default();
    assert(!bn_gpu_policy_metal_small_dense_exact_native_enabled());
    unsetenv("BN_METAL_DISABLE_SMALL_DENSE_EXACT_NATIVE_DEFAULT");
    setenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT", "1", 1);
    bn_gpu_policy_metal_apply_small_dense_exact_native_default();
    assert(!bn_gpu_policy_metal_small_dense_exact_native_enabled());
    unsetenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT");
    bn_gpu_policy_apply_metal_small_dense_exact_native_default_disable_override();
    bn_gpu_policy_metal_apply_small_dense_exact_native_default();
    assert(!bn_gpu_policy_metal_small_dense_exact_native_enabled());
    bn_gpu_policy_apply_metal_private_weights_override();
    assert(getenv("BN_METAL_PRIVATE_WEIGHTS") != NULL);
    unsetenv("BN_METAL_PRIVATE_WEIGHTS");
    unsetenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    bn_gpu_policy_apply_specialized_native_quant_decode_override();
    assert(bn_gpu_policy_specialized_native_quant_decode_path_enabled());
    assert(getenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT") != NULL);
    assert(getenv("BN_METAL_ENABLE_Q6_Q8K") == NULL);
    unsetenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    unsetenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT");
    unsetenv("BN_GPU_DEBUG_ARGMAX");
    assert(!bn_gpu_policy_argmax_debug_enabled());
    setenv("BN_GPU_DEBUG_ARGMAX", "1", 1);
    assert(bn_gpu_policy_argmax_debug_enabled());
    unsetenv("BN_GPU_PROFILE");
    assert(bn_gpu_policy_profile_level() == 0);
    setenv("BN_GPU_PROFILE", "4", 1);
    assert(bn_gpu_policy_profile_level() == 4);
    unsetenv("BN_CUDA_DISABLE_CUBLAS_MATMUL");
    unsetenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16");
    unsetenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    unsetenv("BN_CUDA_ENABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DOWN_KQUANT_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE");
    unsetenv("BN_CUDA_DISABLE_MATVEC");
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0");
    unsetenv("BN_CUDA_DISABLE_Q5_0");
    unsetenv("BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q4_K");
    unsetenv("BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q5_K");
    unsetenv("BN_CUDA_DISABLE_DOWN_KQUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q6_K");
    unsetenv("BN_CUDA_DISABLE_PREPARED_NATIVE_QUANT_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_K");
    unsetenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREPARED_INPUT_SPLIT");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_METAL_ENABLE_MMAP_ZERO_COPY");
    unsetenv("BN_METAL_SHARED_WEIGHTS");
    unsetenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    unsetenv("BN_METAL_Q8_BARRIERS");
    unsetenv("BN_METAL_CPU_ORDER_RMSNORM");
    unsetenv("BN_METAL_FULL_BARRIERS");
    unsetenv("BN_METAL_ENABLE_BARRIERS");
    unsetenv("BN_METAL_DISABLE_BARRIERS");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FROM_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_ATTN_ONLY");
    unsetenv("BN_GPU_SMALL_DENSE_EXACT_NATIVE_FFN_ONLY");
    unsetenv("BN_METAL_DISABLE_SMALL_DENSE_EXACT_NATIVE_DEFAULT");
    unsetenv("BN_GPU_Q4_Q8");
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    unsetenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT");
    unsetenv("BN_METAL_Q4_PREPARED");
    unsetenv("BN_GPU_DEBUG_ARGMAX");
    unsetenv("BN_GPU_PROFILE");

    printf("PASSED\n");
}

// --- Test 5: GPU batch matvec ---
static void test_gpu_batch(void) {
    printf("test_gpu_batch... ");

    float scale = 1.0f;
    // W1: 1 x 128, all +1
    uint8_t *data1 = make_i2s_data(1, 128, scale);
    BnQWeight W1 = {0};
    W1.data = data1;
    W1.type = BN_GGUF_TENSOR_I2_S;
    W1.rows = 1;
    W1.cols = 128;
    W1.scale = scale;

    // W2: 1 x 128, all +1
    uint8_t *data2 = make_i2s_data(1, 128, scale);
    BnQWeight W2 = {0};
    W2.data = data2;
    W2.type = BN_GGUF_TENSOR_I2_S;
    W2.rows = 1;
    W2.cols = 128;
    W2.scale = scale;

    // Upload to mock GPU
    size_t sz = bn_qweight_data_size(&W1);
    void *W1_buf = mock_gpu.buffer_create(mock_gpu.ctx, W1.data, sz, W1.type, W1.rows, W1.cols);
    void *W2_buf = mock_gpu.buffer_create(mock_gpu.ctx, W2.data, sz, W2.type, W2.rows, W2.cols);
    assert(W1_buf && W2_buf);

    float x[128];
    for (int i = 0; i < 128; i++) x[i] = 1.0f;

    float out1_gpu = 0, out2_gpu = 0;
    int8_t scratch[128];
    BnMatvecTask tasks[2] = {
         { &out1_gpu, &W1, NULL, 0 },
         { &out2_gpu, &W2, NULL, 0 },
    };
    bn_backend_quant_matvec_batch_gpu(tasks, 2, x, scratch, NULL, &mock_gpu);
    float out1_gpu_buf = 0, out2_gpu_buf = 0;
    BnMatvecTask buf_tasks[2] = {
         { &out1_gpu_buf, &W1, NULL, 0 },
         { &out2_gpu_buf, &W2, NULL, 0 },
    };
    const void *bufs[2] = { W1_buf, W2_buf };
    bn_backend_quant_matvec_batch_gpu_buf(buf_tasks, bufs, 2, x, scratch, NULL,
                                  &mock_gpu);

    // CPU reference
    float out1_cpu = 0, out2_cpu = 0;
    BnQWeight W1c = W1;
    BnQWeight W2c = W2;
    BnMatvecTask cpu_tasks[2] = {
         { &out1_cpu, &W1c, NULL, 0 },
         { &out2_cpu, &W2c, NULL, 0 },
    };
    bn_quant_matvec_batch(cpu_tasks, 2, x, scratch, NULL);

    assert(fabsf(out1_gpu - out1_cpu) < 1e-3f);
    assert(fabsf(out2_gpu - out2_cpu) < 1e-3f);
    assert(fabsf(out1_gpu_buf - out1_cpu) < 1e-3f);
    assert(fabsf(out2_gpu_buf - out2_cpu) < 1e-3f);

    mock_gpu.buffer_destroy(mock_gpu.ctx, W1_buf);
    mock_gpu.buffer_destroy(mock_gpu.ctx, W2_buf);
    free(data1);
    free(data2);

    printf("PASSED\n");
}

// --- Test 6: bn_qweight_data_size ---
static void test_data_size(void) {
    printf("test_data_size... ");

    BnQWeight w = {0};
    w.data = (void*)1;  // non-NULL sentinel

    w.type = BN_GGUF_TENSOR_I2_S; w.rows = 1; w.cols = 128;
    assert(bn_qweight_data_size(&w) == 128 / 4 + 4);

    w.type = BN_GGUF_TENSOR_Q4_0; w.rows = 1; w.cols = 32;
    assert(bn_qweight_data_size(&w) == 18);

    w.type = BN_GGUF_TENSOR_Q6_K; w.rows = 1; w.cols = 256;
    assert(bn_qweight_data_size(&w) == 210);

    w.type = BN_GGUF_TENSOR_F16; w.rows = 1; w.cols = 100;
    assert(bn_qweight_data_size(&w) == 200);

    // NULL data returns 0
    w.data = NULL;
    assert(bn_qweight_data_size(&w) == 0);

    printf("PASSED\n");
}

static void test_quant_registry(void) {
    printf("test_quant_registry... ");

    const BnQuantFormatOps *q4 = bn_quant_format_ops(BN_GGUF_TENSOR_Q4_0);
    assert(q4 != NULL);
    assert(strcmp(q4->name, "Q4_0") == 0);
    assert(q4->layout == BN_QUANT_LAYOUT_BLOCK32);
    assert(q4->block_elems == 32);
    assert(q4->bytes_per_block == 18);
    assert(bn_quant_format_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q4_0,
                                   BN_QUANT_CAP_LOADABLE |
                                   BN_QUANT_CAP_CPU_MATVEC |
                                   BN_QUANT_CAP_CPU_BATCH |
                                   BN_QUANT_CAP_CPU_MATMUL));
    assert(bn_quant_format_has_cpu_matvec(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_has_cpu_batch(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_has_cpu_matmul(BN_GGUF_TENSOR_Q4_0));
    assert(q4->matvec == bn_quant_matvec);
    assert(q4->matmul == bn_quant_matmul);
    assert(bn_quant_format_matvec(BN_GGUF_TENSOR_Q4_0) == bn_quant_matvec);
    assert(bn_quant_format_matmul(BN_GGUF_TENSOR_Q4_0) == bn_quant_matmul);
    assert(bn_quant_format_uses_embedded_scale(BN_GGUF_TENSOR_Q4_0));
    assert(bn_backend_quant_can_gpu_split(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_can_gpu_native(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_can_gpu_repack(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_can_cpu_repack(BN_GGUF_TENSOR_Q4_0));
    assert(bn_backend_quant_dense_graph_supported(BN_GGUF_TENSOR_F32));
    assert(bn_backend_quant_dense_graph_supported(BN_GGUF_TENSOR_F16));
    assert(bn_backend_quant_dense_graph_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_backend_quant_dense_graph_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_supports_gpu_dense_graph(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU);
    assert(bn_quant_format_gpu_allows_gateup_split_activation(BN_GGUF_TENSOR_Q4_0, 1));
    assert(bn_quant_format_data_size(BN_GGUF_TENSOR_Q4_0, 1, 32) == 18);

    const BnQuantFormatOps *i2s = bn_quant_format_ops(BN_GGUF_TENSOR_I2_S);
    assert(i2s != NULL);
    assert(i2s->layout == BN_QUANT_LAYOUT_I2S);
    assert(bn_quant_format_supported(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_format_has_cpu_matvec(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_format_has_cpu_batch(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_format_has_cpu_matmul(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_uses_embedded_scale(BN_GGUF_TENSOR_I2_S));
    assert(!bn_backend_quant_can_gpu_split(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_can_gpu_native(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_can_gpu_repack(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_can_cpu_repack(BN_GGUF_TENSOR_I2_S));
    assert(!bn_backend_quant_dense_graph_supported(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_supports_gpu_dense_graph(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_I2_S) == 0);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_I2_S) == 0);
    assert(bn_quant_format_data_size(BN_GGUF_TENSOR_I2_S, 1, 128) == 36);

    assert(bn_quant_format_supported(BN_GGUF_TENSOR_Q5_1));
    assert(bn_quant_format_supported(BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_has_cpu_matvec(BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_has_cpu_batch(BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_has_cpu_matmul(BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_matvec(BN_GGUF_TENSOR_Q5_0) == bn_quant_matvec);
    assert(bn_quant_format_matmul(BN_GGUF_TENSOR_Q5_0) == bn_quant_matmul);
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q5_0) ==
           BN_GPU_CAP_MIDBIT_BLOCK32_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q5_0) ==
           BN_GPU_CAP_MIDBIT_BLOCK32_FUSED_GATEUP_SILU);
    assert(!bn_quant_format_has_cap(99999, BN_QUANT_CAP_LOADABLE));
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q8_0) ==
           BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q5_K) ==
           BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q8_0) ==
           BN_GPU_CAP_NATIVE_QUANT_FUSED_GATEUP_SILU);
    assert(bn_backend_quant_dense_graph_native_quant_supported(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_dense_graph_native_quant_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_backend_quant_dense_graph_native_quant_supported(BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_supports_native_quant_logits_refine(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_supports_native_quant_logits_refine(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_gpu_dense_graph_native_quant(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_native_quant_logits_refine(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_uses_f16_logits_path(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_uses_f16_logits_path(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_logits_i8_cache(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_supports_logits_i8_cache(BN_GGUF_TENSOR_F32));
    assert(bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_tied_logits_uses_quant_path(BN_GGUF_TENSOR_F32));
    assert(bn_quant_format_tied_logits_uses_f16_path(BN_GGUF_TENSOR_F16));
    assert(!bn_quant_format_tied_logits_uses_f16_path(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_tied_logits_i8_weight_type() ==
           BN_GGUF_TENSOR_Q8_0);
    assert(bn_quant_format_tied_logits_f16_weight_type() ==
           BN_GGUF_TENSOR_F16);
    assert(bn_quant_format_tied_logits_f32_weight_type() ==
           BN_GGUF_TENSOR_F32);
    unsetenv("BN_CPU_TIED_KQUANT_REFINE_TOP");
    unsetenv("BN_CPU_TIED_Q6K_REFINE_TOP");
    unsetenv("BN_CPU_TIED_KQUANT_HYBRID_TOP");
    unsetenv("BN_CPU_TIED_Q6K_HYBRID_TOP");
    assert(bn_backend_quant_cpu_tied_kquant_refine_top() == 0);
    assert(bn_backend_quant_cpu_tied_kquant_hybrid_top() == 0);
    setenv("BN_CPU_TIED_KQUANT_REFINE_TOP", "1", 1);
    setenv("BN_CPU_TIED_KQUANT_HYBRID_TOP", "2", 1);
    assert(bn_backend_quant_cpu_tied_kquant_refine_top() == 1);
    assert(bn_backend_quant_cpu_tied_kquant_hybrid_top() == 2);
    setenv("BN_CPU_TIED_KQUANT_REFINE_TOP", "0", 1);
    setenv("BN_CPU_TIED_KQUANT_HYBRID_TOP", "1", 1);
    assert(bn_backend_quant_cpu_tied_kquant_refine_top() == 0);
    assert(bn_backend_quant_cpu_tied_kquant_hybrid_top() == 0);
    setenv("BN_CPU_TIED_KQUANT_REFINE_TOP", "256", 1);
    setenv("BN_CPU_TIED_KQUANT_HYBRID_TOP", "256", 1);
    assert(bn_backend_quant_cpu_tied_kquant_refine_top() == 128);
    assert(bn_backend_quant_cpu_tied_kquant_hybrid_top() == 128);
    unsetenv("BN_CPU_TIED_KQUANT_REFINE_TOP");
    unsetenv("BN_CPU_TIED_KQUANT_HYBRID_TOP");
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "1", 1);
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "2", 1);
    assert(bn_backend_quant_cpu_tied_kquant_refine_top() == 1);
    assert(bn_backend_quant_cpu_tied_kquant_hybrid_top() == 2);
    unsetenv("BN_CPU_TIED_Q6K_REFINE_TOP");
    unsetenv("BN_CPU_TIED_Q6K_HYBRID_TOP");
    assert(bn_quant_format_gpu_float_buffer_type() ==
           BN_GGUF_TENSOR_F32);
    assert(bn_quant_format_dense_f32_type() == BN_GGUF_TENSOR_F32);
    assert(bn_quant_format_is_f32(BN_GGUF_TENSOR_F32));
    assert(!bn_quant_format_is_f32(BN_GGUF_TENSOR_F16));
    assert(bn_quant_format_can_convert_dense_to_f32(BN_GGUF_TENSOR_F16));
    assert(bn_quant_format_can_convert_dense_to_f32(BN_GGUF_TENSOR_BF16));
    assert(!bn_quant_format_can_convert_dense_to_f32(BN_GGUF_TENSOR_Q4_0));
    uint16_t f16_src[2] = { bn_fp32_to_fp16(1.5f), bn_fp32_to_fp16(-2.0f) };
    float f32_dst[2] = { 0.0f, 0.0f };
    assert(bn_quant_format_convert_dense_to_f32(
               BN_GGUF_TENSOR_F16, f16_src, f32_dst, 2) == 0);
    assert(fabsf(f32_dst[0] - 1.5f) < 1e-3f);
    assert(fabsf(f32_dst[1] + 2.0f) < 1e-3f);
    assert(bn_quant_format_convert_dense_to_f32(
               BN_GGUF_TENSOR_Q4_0, f16_src, f32_dst, 2) == -1);
    assert(bn_transformer_cpu_has_native_quant_activation() ==
           TEST_EXPECT_NATIVE_QUANT_ACTIVATION);
    assert(bn_quant_format_gpu_requires_exact_silu(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_requires_exact_silu(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_supports_prepared_kquant(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_requires_float_kquant_fallback(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_is_float_kquant_fallback_candidate(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_can_gpu_split(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU);
    assert(bn_quant_format_gpu_matvec_kquant_dot_flag(BN_GGUF_TENSOR_Q4_K, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT);
    assert(bn_quant_format_gpu_matvec_kquant_dot_flag(BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_quant_format_gpu_matvec_kquant_dot_flag(BN_GGUF_TENSOR_Q8_0, 1) == 0);
    assert(bn_quant_format_supports_moe_routed_kquant_gateup(BN_GGUF_TENSOR_Q4_K,
                                          BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_supports_moe_routed_kquant_gateup(BN_GGUF_TENSOR_Q4_K,
                                           BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_moe_routed_kquant_gateup(BN_GGUF_TENSOR_Q4_K,
                                                     BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_moe_routed_kquant_gateup(BN_GGUF_TENSOR_Q4_K,
                                                      BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_cpu_fused_kquant_gateup_silu(BN_GGUF_TENSOR_Q4_0,
                                                     BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_supports_cpu_fused_kquant_gateup_silu(BN_GGUF_TENSOR_Q4_0,
                                                      BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_cpu_fused_kquant_gateup_silu(BN_GGUF_TENSOR_Q4_0,
                                                         BN_GGUF_TENSOR_Q4_0));
    assert(!bn_backend_quant_cpu_fused_kquant_gateup_silu(BN_GGUF_TENSOR_Q4_0,
                                                          BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_same_quant_format_pair_stackable(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_same_quant_format_pair_stackable(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_moe_asymmetric_kquant_down_route(BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_quant_format_supports_moe_asymmetric_kquant_down_route(BN_GGUF_TENSOR_Q4_K,
                                               BN_GGUF_TENSOR_Q4_K,
                                               BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_supports_moe_asymmetric_kquant_down_route(BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_backend_quant_moe_routed_asymmetric_kquant(BN_GGUF_TENSOR_Q4_K,
                                                         BN_GGUF_TENSOR_Q4_K,
                                                         BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_moe_routed_asymmetric_kquant(BN_GGUF_TENSOR_Q4_K,
                                                         BN_GGUF_TENSOR_Q4_K,
                                                         BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_moe_routed_asymmetric_kquant(BN_GGUF_TENSOR_Q4_K,
                                                          BN_GGUF_TENSOR_Q8_0,
                                                          BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_moe_native_quant_route(BN_GGUF_TENSOR_Q8_0,
                                                 BN_GGUF_TENSOR_Q8_0,
                                                 BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_supports_moe_native_quant_route(BN_GGUF_TENSOR_Q8_0,
                                                  BN_GGUF_TENSOR_Q8_0,
                                                  BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_moe_routed_native_quant(BN_GGUF_TENSOR_Q8_0,
                                          BN_GGUF_TENSOR_Q8_0,
                                          BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_moe_routed_native_quant(BN_GGUF_TENSOR_Q8_0,
                                           BN_GGUF_TENSOR_Q8_0,
                                           BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_moe_routed_op_uses_native_quant(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_moe_routed_op_uses_native_quant(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_moe_routed_op_uses_asymmetric_kquant(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_moe_routed_op_uses_asymmetric_kquant(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_allows_gateup_split_activation(BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_quant_format_gpu_allows_gateup_split_activation(BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_supports_prepared_kquant(BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_moe_down_uses_down_kquant(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_moe_down_uses_down_kquant(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_moe_down_uses_asymmetric_kquant(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_moe_down_uses_asymmetric_kquant(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_moe_down_uses_graph_kquant(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_moe_down_uses_graph_kquant(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_moe_down_uses_graph_kquant(BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_gpu_graph_gateup_needs_prepared_input_scratch(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_gpu_graph_gateup_needs_prepared_input_scratch(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_gpu_graph_gateup_needs_prepared_input_scratch(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_gpu_graph_gateup_needs_prepared_input_scratch(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_gpu_graph_matvec_needs_prepared_input_scratch(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_gpu_graph_matvec_needs_prepared_input_scratch(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_gpu_graph_matvec_down_kquant_needs_dot_scratch(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_gpu_graph_matvec_down_kquant_needs_dot_scratch(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_gpu_graph_matvec_asymmetric_kquant_needs_dot_scratch(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_gpu_graph_matvec_asymmetric_kquant_needs_dot_scratch(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_deinterleaved_kquant_pair_matvec(
        BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_deinterleaved_kquant_pair_matvec(
        BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_asymmetric_kquant_pair_matvec(
        BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_pair_matvec(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_symmetric_kquant_pair_matvec(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_symmetric_kquant_pair_matvec(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_native_quant_small_state_matvec_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_small_state_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_native_quant_f16_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_f16_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_f16_float_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q3_K));
    assert(bn_backend_quant_f16_float_cache_matvec_candidate(
        BN_GGUF_TENSOR_IQ3_XXS));
    assert(bn_backend_quant_f16_float_cache_matvec_candidate(
        BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_backend_quant_f16_float_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_packed_kquant_f16_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_packed_kquant_f16_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_down_kquant_f16_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_down_kquant_f16_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_kquant_logits_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_kquant_logits_cache_matvec_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_legacy_block_matvec_candidate(
        BN_GGUF_TENSOR_Q5_0));
    assert(!bn_backend_quant_legacy_block_matvec_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_down_kquant_dot_matvec_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_down_kquant_dot_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_down_kquant_warp_matvec_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_down_kquant_warp_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_asymmetric_kquant_dot_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_dot_matvec_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_asymmetric_kquant_prepared_input_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_prepared_input_matvec_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_deinterleaved_kquant_prepared_input_matvec_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_deinterleaved_kquant_prepared_input_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_native_quant_prepared_input_matvec_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_prepared_input_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_prepared_native_quant_matvec_candidate(
        BN_GGUF_TENSOR_Q8_K));
    assert(!bn_backend_quant_prepared_native_quant_matvec_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_native_quant_warp_matvec_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_warp_matvec_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_asymmetric_kquant_dot_matmul_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_dot_matmul_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_asymmetric_kquant_prepared_input_matmul_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_prepared_input_matmul_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_deinterleaved_kquant_prepared_input_matmul_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_deinterleaved_kquant_prepared_input_matmul_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_down_kquant_dot_matmul_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_down_kquant_dot_matmul_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_native_quant_matmul_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_matmul_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_legacy_block_matmul_candidate(
        BN_GGUF_TENSOR_Q5_0));
    assert(!bn_backend_quant_legacy_block_matmul_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_legacy_block_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q5_0));
    assert(!bn_backend_quant_legacy_block_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_native_quant_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q5_0));
    assert(bn_backend_quant_asymmetric_kquant_dot_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_dot_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_asymmetric_kquant_prepared_input_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_prepared_input_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_deinterleaved_kquant_prepared_input_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_deinterleaved_kquant_prepared_input_fused_gateup_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_matvec_allows_fused_bias(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_matvec_allows_fused_bias(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_split_allows_fused_bias(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_split_allows_fused_bias(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_asymmetric_kquant_dot_split_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_dot_split_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_asymmetric_kquant_prepared_input_split_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_prepared_input_split_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_deinterleaved_kquant_prepared_input_split_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_deinterleaved_kquant_prepared_input_split_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_native_quant_split_candidate(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_split_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_split_value_4warp_dot_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_split_value_4warp_dot_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_split_value_mmvq_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_split_value_mmvq_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_split_value_fuse_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_split_value_fuse_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_split_value_fuse_candidate(
        BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_legacy_block_pair_matmul(
        BN_GGUF_TENSOR_Q5_0, BN_GGUF_TENSOR_Q5_0));
    assert(!bn_backend_quant_legacy_block_pair_matmul(
        BN_GGUF_TENSOR_Q5_0, BN_GGUF_TENSOR_Q8_0));
    assert(bn_backend_quant_native_quant_pair_matmul(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_native_quant_pair_matmul(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q5_0));
    assert(bn_backend_quant_asymmetric_kquant_pair_matmul(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_asymmetric_kquant_pair_matmul(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_deinterleaved_kquant_pair_matmul(
        BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_deinterleaved_kquant_pair_matmul(
        BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_kquant_logits_argmax_candidate(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_kquant_logits_argmax_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_moe_route_all_active_two(2, 2));
    assert(!bn_backend_quant_moe_route_all_active_two(2, 1));
    assert(!bn_backend_quant_moe_route_all_active_two(4, 2));
    assert(bn_backend_quant_moe_all_active_two_kquant_shape(2, 2,
                                                BN_GGUF_TENSOR_Q6_K,
                                                4096, 2048));
    assert(!bn_backend_quant_moe_all_active_two_kquant_shape(2, 2,
                                                 BN_GGUF_TENSOR_Q4_K,
                                                 4096, 2048));
    assert(bn_backend_quant_moe_all_active_two_kquant_routed_op(BN_GGUF_TENSOR_Q4_K,
                                                    2, 2,
                                                    BN_GGUF_TENSOR_Q6_K,
                                                    4096, 2048));
    assert(!bn_backend_quant_moe_all_active_two_kquant_routed_op(BN_GGUF_TENSOR_Q8_0,
                                                     2, 2,
                                                     BN_GGUF_TENSOR_Q6_K,
                                                     4096, 2048));
    assert(!bn_backend_quant_moe_all_active_two_kquant_routed_op(BN_GGUF_TENSOR_Q4_K,
                                                     2, 2,
                                                     BN_GGUF_TENSOR_Q6_K,
                                                     4096, 2049));
    assert(bn_backend_quant_moe_all_active_two_graph_kquant_shape(2, 2,
                                                    BN_GGUF_TENSOR_Q4_K,
                                                    4096, 2048));
    assert(!bn_backend_quant_moe_all_active_two_graph_kquant_shape(2, 2,
                                                     BN_GGUF_TENSOR_Q4_K,
                                                     4095, 2048));
    assert(bn_backend_quant_requires_float_kquant_fallback(BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_supports_kquant_logits_refine(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_supports_kquant_logits_refine(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_kquant_logits_refine(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_logits_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_logits_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_down_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_moe_down_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_down_cublas_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_moe_down_cublas_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_down_cublas_cache_elem_bytes(
        BN_GGUF_TENSOR_Q6_K, 1) == (int)sizeof(uint16_t));
    assert(bn_quant_format_moe_down_cublas_cache_elem_bytes(
        BN_GGUF_TENSOR_Q6_K, 0) == (int)sizeof(float));
    assert(bn_quant_format_moe_down_cublas_cache_elem_bytes(
        BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_quant_format_moe_down_small_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_moe_down_small_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_gpu_matvec_kquant_dot_flag(
               BN_GGUF_TENSOR_Q4_K, 1) == BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT);
    assert(bn_backend_quant_gpu_matvec_kquant_dot_flag(
               BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_backend_quant_gpu_matvec_kquant_dot_flag(
               BN_GGUF_TENSOR_Q8_0, 1) == 0);
    assert(bn_backend_quant_gpu_matvec_exact_kquant_flag(
               BN_GGUF_TENSOR_Q6_K, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_EXACT_KQUANT);
    assert(bn_backend_quant_gpu_matvec_exact_kquant_flag(
               BN_GGUF_TENSOR_Q6_K, 0) == 0);
    assert(bn_backend_quant_gpu_matvec_exact_kquant_flag(
               BN_GGUF_TENSOR_Q4_K, 1) == 0);
    assert(bn_quant_format_gpu_matvec_exact_kquant_flag(BN_GGUF_TENSOR_Q6_K, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_EXACT_KQUANT);
    assert(bn_quant_format_gpu_matvec_exact_kquant_flag(BN_GGUF_TENSOR_Q6_K, 0) == 0);
    assert(bn_quant_format_gpu_matvec_exact_kquant_flag(BN_GGUF_TENSOR_Q4_K, 1) == 0);
    assert(bn_quant_format_supports_prepared_kquant(BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_requires_float_kquant_fallback(BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_lazy_moe_aux_cache_candidate(BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_lazy_moe_aux_cache_candidate(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_lazy_moe_aux_cache_candidate(BN_GGUF_TENSOR_Q5_K));
    float lazy_aux_tmp[BN_QK_K];
    BnBlockIQ3XXS lazy_aux_iq3 = {0};
    assert(bn_backend_quant_lazy_moe_aux_cache_dequant_block(
        BN_GGUF_TENSOR_IQ3_XXS, &lazy_aux_iq3, 0, lazy_aux_tmp) == 0);
    assert(bn_backend_quant_lazy_moe_aux_cache_dequant_block(
        BN_GGUF_TENSOR_F32, &lazy_aux_iq3, 0, lazy_aux_tmp) == -1);
    assert(bn_backend_quant_dense_graph_supported(BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_dense_graph_supported(BN_GGUF_TENSOR_Q8_K));
    int dense_dummy = 1;
    BnConfig dense_c = {0};
    BnLayerWeights dense_layers[1] = {0};
    BnWeights dense_w = {0};
    dense_c.n_layers = 1;
    dense_w.layers = dense_layers;
    dense_w.emb_type = BN_GGUF_TENSOR_Q4_0;
    dense_layers[0].attn.wq.data = &dense_dummy;
    dense_layers[0].attn.wq.type = BN_GGUF_TENSOR_Q4_0;
    assert(bn_backend_quant_dense_graph_model_supported(
        &dense_w, &dense_c, 0));
    assert(!bn_backend_quant_dense_graph_model_supported(
        &dense_w, &dense_c, 1));
    dense_w.emb_type = BN_GGUF_TENSOR_Q8_0;
    dense_layers[0].attn.wq.type = BN_GGUF_TENSOR_Q8_0;
    assert(bn_backend_quant_dense_graph_model_supported(
        &dense_w, &dense_c, 1));
    dense_w.output_weight.data = &dense_dummy;
    dense_w.output_weight.type = BN_GGUF_TENSOR_Q4_0;
    assert(!bn_backend_quant_dense_graph_model_supported(
        &dense_w, &dense_c, 1));
    assert(bn_backend_quant_dense_graph_model_supported(
        &dense_w, &dense_c, 0));
    assert(!bn_backend_quant_dense_graph_model_supported(NULL, &dense_c, 0));
    assert(!bn_backend_quant_dense_graph_model_supported(&dense_w, NULL, 0));
    assert(!bn_quant_format_can_gpu_native(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_can_gpu_repack(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_can_cpu_repack(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q5_K) ==
           BN_GPU_CAP_DEINTERLEAVED_KQUANT_FUSED_GATEUP_SILU);
    assert(bn_quant_format_gpu_fused_gateup_requires_backend_opt_in(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_gpu_fused_gateup_requires_backend_opt_in(BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_requires_float_kquant_fallback(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_moe_all_f16_cache_supported(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_moe_all_f16_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_moe_all_f16_cache_supported(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q8_0, 1));
    assert(bn_quant_format_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_quant_format_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_quant_format_lazy_moe_aux_cache_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_lazy_moe_aux_cache_candidate(
        BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_lazy_moe_aux_cache_candidate(
        BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_moe_prefers_quant_only(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_moe_prefers_quant_only(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_format_aux_cache_can_use_f16(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_aux_cache_can_use_f16(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_aux_cache_uses_f32(BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_quant_format_aux_cache_uses_f32(BN_GGUF_TENSOR_Q6_K, 1));
    assert(!bn_quant_format_aux_cache_uses_f32(BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_aux_cache_prefers_large_budget(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_aux_cache_prefers_large_budget(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_logits_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_moe_all_f16_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_down_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_moe_down_cublas_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_moe_down_cublas_cache_elem_bytes(
        BN_GGUF_TENSOR_Q6_K, 0) == (int)sizeof(float));
    assert(bn_quant_format_moe_down_small_kquant_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_quant_format_moe_prefers_quant_only(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_aux_cache_supported(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_aux_cache_can_use_f16(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_aux_cache_uses_f32(BN_GGUF_TENSOR_Q6_K, 0));
    assert(bn_quant_format_aux_cache_prefers_large_budget(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_aux_cache_force_asymmetric_kquant_f32(
        BN_GGUF_TENSOR_Q4_K, 1));
    assert(!bn_backend_quant_aux_cache_force_asymmetric_kquant_f32(
        BN_GGUF_TENSOR_Q6_K, 1));
    assert(!bn_backend_quant_aux_cache_force_asymmetric_kquant_f32(
        BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_backend_quant_aux_cache_down_kquant_can_use_f16(
        BN_GGUF_TENSOR_Q6_K, 1, 1));
    assert(bn_backend_quant_aux_cache_down_kquant_can_use_f16(
        BN_GGUF_TENSOR_Q6_K, 0, 0));
    assert(!bn_backend_quant_aux_cache_down_kquant_can_use_f16(
        BN_GGUF_TENSOR_Q6_K, 0, 1));
    assert(!bn_backend_quant_aux_cache_down_kquant_can_use_f16(
        BN_GGUF_TENSOR_Q4_K, 1, 1));
    assert(bn_backend_quant_aux_cache_add_down_kquant_f32(
        BN_GGUF_TENSOR_Q6_K, 1));
    assert(!bn_backend_quant_aux_cache_add_down_kquant_f32(
        BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_backend_quant_aux_cache_add_down_kquant_f32(
        BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_backend_quant_aux_cache_f32_storage(
        BN_GGUF_TENSOR_Q4_K, 1, 0));
    assert(bn_backend_quant_aux_cache_f32_storage(
        BN_GGUF_TENSOR_Q6_K, 0, 0));
    assert(!bn_backend_quant_aux_cache_f32_storage(
        BN_GGUF_TENSOR_Q6_K, 0, 1));
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_BF16, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_DENSE_BFLOAT_TO_F16);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q8_0, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_NATIVE_QUANT_TO_F16);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q5_0, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_LEGACY_BLOCK_TO_F16);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q3_K, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_COMPACT_KQUANT_TO_F16);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q4_K, 1, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_ASYMMETRIC_KQUANT_TO_F32);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q4_K, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_ASYMMETRIC_KQUANT_TO_F16);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q5_K, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_DEINTERLEAVED_KQUANT_TO_F16);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q6_K, 0, 1) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_DOWN_KQUANT_TO_F16);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_Q6_K, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_DOWN_KQUANT_TO_F32);
    assert(bn_backend_quant_aux_cache_dequant_route(
               BN_GGUF_TENSOR_F32, 0, 0) ==
           BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_NONE);
    assert(bn_quant_format_supports_moe_native_quant_route(BN_GGUF_TENSOR_Q8_0,
                                         BN_GGUF_TENSOR_Q8_0,
                                         BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_supports_moe_native_quant_route(BN_GGUF_TENSOR_Q8_0,
                                          BN_GGUF_TENSOR_Q8_0,
                                          BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_data_size(BN_GGUF_TENSOR_Q5_0, 1, 32) == 22);
    assert(bn_quant_format_data_size(99999, 1, 32) == 0);
    assert(bn_quant_format_data_size(BN_GGUF_TENSOR_Q4_K, 1, 128) == 0);

    printf("PASSED\n");
}

static void test_backend_layout_reasons(void) {
    printf("test_backend_layout_reasons... ");

    uint8_t data_a[18] = {0};
    uint8_t data_b[18] = {0};
    BnQWeight a = {0};
    BnQWeight b = {0};
    a.data = data_a;
    b.data = data_b;
    a.type = BN_GGUF_TENSOR_Q4_0;
    b.type = BN_GGUF_TENSOR_Q4_0;
    a.rows = 1;
    b.rows = 1;
    a.cols = 32;
    b.cols = 32;

    assert(bn_backend_layout_stackable_reason(&a, &b) == BN_BACKEND_LAYOUT_OK);
    assert(bn_backend_layout_stackable(&a, &b));
    assert(bn_backend_layout_stackable3(&a, &b, &a));
    assert(bn_backend_layout_stacked2_reason(&mock_gpu, &a, &b) ==
           BN_BACKEND_LAYOUT_OK);
    assert(strcmp(bn_backend_layout_reason_string(BN_BACKEND_LAYOUT_OK), "ok") == 0);
    assert(bn_backend_layout_stacked2_reason(NULL, &a, &b) ==
           BN_BACKEND_LAYOUT_NO_GPU);

    BnGPUBackend no_create = mock_gpu;
    no_create.buffer_create = NULL;
    assert(bn_backend_layout_stacked2_reason(&no_create, &a, &b) ==
           BN_BACKEND_LAYOUT_NO_BUFFER_CREATE);

    b.data = NULL;
    assert(bn_backend_layout_stackable_reason(&a, &b) == BN_BACKEND_LAYOUT_MISSING_WEIGHT);
    assert(bn_backend_layout_stacked2_reason(&mock_gpu, &a, &b) ==
           BN_BACKEND_LAYOUT_MISSING_WEIGHT);
    assert(!bn_backend_layout_stackable(&a, &b));
    b.data = data_b;

    b.type = BN_GGUF_TENSOR_Q8_0;
    assert(bn_backend_layout_stackable_reason(&a, &b) == BN_BACKEND_LAYOUT_TYPE_MISMATCH);
    b.type = BN_GGUF_TENSOR_Q4_0;

    b.cols = 64;
    assert(bn_backend_layout_stackable_reason(&a, &b) == BN_BACKEND_LAYOUT_COL_MISMATCH);
    assert(!bn_backend_layout_stackable3(&a, &b, &a));
    b.cols = 32;

    a.type = BN_GGUF_TENSOR_I2_S;
    assert(bn_backend_layout_stackable_reason(&a, &b) ==
           BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE);
    assert(strcmp(bn_backend_layout_reason_string(
               BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE),
           "embedded_scale_not_stackable") == 0);
    assert(strcmp(bn_backend_layout_reason_string((BnBackendLayoutReason)999), "unknown") == 0);
    a.type = BN_GGUF_TENSOR_Q4_0;

    float bias[1] = {0.0f};
    assert(bn_backend_layout_biased_qweight_reason(&mock_gpu, &a, bias) ==
           BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED);

    assert(bn_backend_layout_stacked3_qkv_reason(&mock_gpu, &a, &a, &a,
                                                 NULL, NULL, NULL,
                                                 0, 0, 0) ==
           BN_BACKEND_LAYOUT_OK);
    assert(bn_backend_layout_stacked3_qkv_reason(&mock_gpu, &a, &a, &a,
                                                 bias, NULL, NULL,
                                                 0, 0, 0) ==
           BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED);

    printf("PASSED\n");
}

static void test_backend_layout_prepared_qweights(void) {
    printf("test_backend_layout_prepared_qweights... ");

    BnBlockQ4_0 q4_blocks[4];
    memset(q4_blocks, 0, sizeof(q4_blocks));
    for (int i = 0; i < 4; i++) {
        q4_blocks[i].d = bn_fp32_to_fp16(1.0f);
        memset(q4_blocks[i].qs, 0x88, sizeof(q4_blocks[i].qs));
    }

    BnConfig config = {0};
    config.n_layers = 1;
    BnWeights weights = {0};
    BnLayerWeights layer = {0};
    weights.layers = &layer;
    layer.attn.wq.data = q4_blocks;
    layer.attn.wq.type = BN_GGUF_TENSOR_Q4_0;
    layer.attn.wq.rows = 4;
    layer.attn.wq.cols = 32;

    BnBackendLayoutPreparedStats stats = {0};
    size_t bytes = bn_backend_layout_prepared_qweights_size(&config, &weights, &stats);

#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__wasm_relaxed_simd__)
    assert(bytes > 0);
    assert(stats.lowbit_repack_bytes == bytes);
    assert(stats.kquant_scale_table_bytes == 0);
    assert(stats.expanded_kquant_weight_bytes == 0);
    assert(stats.f32_scale_table_bytes == 0);

    SHArena *arena = sh_arena_create(bytes + 4 * SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnBackendModel *backend = bn_backend_model_create();
    assert(backend != NULL);

    BnBackendLayoutPreparedStats built = {0};
    bn_backend_layout_prepare_qweights(backend, &config, &weights, arena, &built);
    assert(built.lowbit_repack_bytes == stats.lowbit_repack_bytes);
    assert(built.kquant_scale_table_bytes == 0);
    assert(built.expanded_kquant_weight_bytes == 0);
    assert(built.f32_scale_table_bytes == 0);
    const BnPreparedWeight *prepared =
        bn_backend_model_prepared_qweight(backend, &layer.attn.wq);
    assert(prepared != NULL);
    assert(prepared->kind == BN_PREPARED_WEIGHT_Q4_0_REPACK);
    assert(prepared->qs != NULL);
#ifdef __wasm_relaxed_simd__
    assert(prepared->f32_scales != NULL);
#else
    assert(prepared->scales != NULL);
#endif

    bn_backend_model_free(backend);
    sh_arena_free(arena);
#else
    assert(bytes == 0);
    assert(stats.lowbit_repack_bytes == 0);
    assert(stats.kquant_scale_table_bytes == 0);
    assert(stats.expanded_kquant_weight_bytes == 0);
    assert(stats.f32_scale_table_bytes == 0);
#endif

    BnBlockQ4K q4k_blocks[1];
    memset(q4k_blocks, 0, sizeof(q4k_blocks));
    q4k_blocks[0].d = bn_fp32_to_fp16(1.0f);
    q4k_blocks[0].dmin = bn_fp32_to_fp16(0.5f);
    for (int i = 0; i < 12; i++)
        q4k_blocks[0].scales[i] = (uint8_t)(i + 1);

    BnLayerWeights q4k_layer = {0};
    BnWeights q4k_weights = {0};
    q4k_weights.layers = &q4k_layer;
    q4k_layer.attn.wq.data = q4k_blocks;
    q4k_layer.attn.wq.type = BN_GGUF_TENSOR_Q4_K;
    q4k_layer.attn.wq.rows = 1;
    q4k_layer.attn.wq.cols = BN_QK_K;

    BnBackendLayoutPreparedStats q4k_stats = {0};
    size_t q4k_bytes =
        bn_backend_layout_prepared_qweights_size(&config, &q4k_weights,
                                                 &q4k_stats);
#if defined(__AVX2__)
    assert(q4k_bytes > 0);
    assert(q4k_stats.lowbit_repack_bytes == 0);
    assert(q4k_stats.kquant_scale_table_bytes == q4k_bytes);
    assert(q4k_stats.expanded_kquant_weight_bytes == 0);
    assert(q4k_stats.f32_scale_table_bytes == 0);

    SHArena *q4k_arena = sh_arena_create(q4k_bytes + 4 * SH_ARENA_ALIGN);
    assert(q4k_arena != NULL);
    BnBackendModel *q4k_backend = bn_backend_model_create();
    assert(q4k_backend != NULL);
    BnBackendLayoutPreparedStats q4k_built = {0};
    bn_backend_layout_prepare_qweights(q4k_backend, &config, &q4k_weights,
                                       q4k_arena, &q4k_built);
    assert(q4k_built.lowbit_repack_bytes == 0);
    assert(q4k_built.kquant_scale_table_bytes == q4k_stats.kquant_scale_table_bytes);
    assert(q4k_built.expanded_kquant_weight_bytes == 0);
    assert(q4k_built.f32_scale_table_bytes == 0);
    const BnPreparedWeight *q4k_prepared =
        bn_backend_model_prepared_qweight(q4k_backend, &q4k_layer.attn.wq);
    assert(q4k_prepared != NULL);
    assert(q4k_prepared->kind == BN_PREPARED_WEIGHT_Q4_K_SCALES);
    assert(q4k_prepared->qs != NULL);
    assert(q4k_prepared->f32_scales != NULL);
    bn_backend_model_free(q4k_backend);
    sh_arena_free(q4k_arena);
#else
    assert(q4k_bytes == 0);
    assert(q4k_stats.lowbit_repack_bytes == 0);
    assert(q4k_stats.kquant_scale_table_bytes == 0);
    assert(q4k_stats.expanded_kquant_weight_bytes == 0);
    assert(q4k_stats.f32_scale_table_bytes == 0);
#endif

    printf("PASSED\n");
}

int main(void) {
    test_data_size();
    test_quant_registry();
    test_gpu_policy_helpers();
    test_backend_layout_reasons();
    test_backend_layout_prepared_qweights();
    test_gpu_upload_weights();
    test_gpu_matvec();
    test_gpu_fallback();
    test_gpu_release();
    test_backend_model_release_owns_buffers();
    test_gpu_batch();
    printf("All GPU backend tests PASSED\n");
    return 0;
}
