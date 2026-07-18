#include "gpu_backend.h"
#include "gpu_policy.h"
#include "backend_quant.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "quant.h"
#include "transformer_cpu_backend_internal.h"
#include "model.h"
#include "gguf.h"
#include "sh_arena.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#if defined(BN_FORCE_SCALAR)
#define TEST_EXPECT_NATIVE_Q8X 0
#elif (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || \
    defined(__AVX2__) || defined(__wasm_relaxed_simd__)
#define TEST_EXPECT_NATIVE_Q8X 1
#else
#define TEST_EXPECT_NATIVE_Q8X 0
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

static void test_gpu_policy_helpers(void) {
    printf("test_gpu_policy_helpers... ");

    BnGPUBackend gpu = mock_gpu;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.buffer_create_f16_cache = mock_create;
    gpu.buffer_create_q6_f32_cache = mock_create;

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
    assert(!bn_gpu_policy_moe_router_diff2_upload_enabled(&layers));
    assert(!bn_gpu_policy_cuda_moe_f16_aux_cache_auto_enabled(&layers));

    BnConfig moe = {0};
    moe.dim = 2048;
    moe.hidden_dim = 512;
    moe.n_experts = 2;
    moe.n_experts_active = 2;
    moe.moe_intermediate_size = 4096;
    assert(bn_gpu_policy_uses_moe(&moe));
    assert(!bn_gpu_policy_uses_hybrid_moe(&moe));
    assert(bn_gpu_policy_moe_router_diff2_upload_enabled(&moe));
    assert(bn_gpu_policy_cuda_moe_f16_aux_cache_auto_enabled(&moe));
    moe.full_attn_interval = 4;
    assert(bn_gpu_policy_uses_hybrid_moe(&moe));
    moe.full_attn_interval = 0;
    moe.dim = 2049;
    assert(!bn_gpu_policy_moe_router_diff2_upload_enabled(&moe));
    assert(bn_gpu_policy_cuda_moe_f16_aux_cache_auto_enabled(&moe));
    moe.n_experts = 4;
    moe.n_experts_active = 2;
    assert(!bn_gpu_policy_moe_router_diff2_upload_enabled(&moe));
    assert(bn_gpu_policy_cuda_moe_f16_aux_cache_auto_enabled(&moe));

    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN");
    assert(bn_gpu_policy_cuda_moe_routed_ffn_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_routed_ffn_enabled(0));
    setenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_routed_ffn_enabled(1));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN");

    unsetenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE");
    assert(!bn_gpu_policy_cuda_moe_all_f16_cache_forced());
    assert(!bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q8_0, 1));
    setenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_all_f16_cache_forced());
    assert(bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q4_0, 0));
    setenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
        &gpu, BN_GGUF_TENSOR_Q8_0, 1));
    unsetenv("BN_CUDA_ENABLE_MOE_ALL_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_ALL_F16_CACHE");

    unsetenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE");
    assert(!bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(1));
    setenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(1));
    unsetenv("BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE");

    unsetenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE");
    assert(!bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(1));
    setenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(1));
    assert(!bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(0));
    unsetenv("BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE");

    unsetenv("BN_CUDA_DEBUG_MOE_FIT");
    unsetenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE");
    assert(!bn_gpu_policy_cuda_moe_fit_debug_enabled());
    assert(!bn_gpu_policy_cuda_keep_individual_f16_cache_enabled());
    setenv("BN_CUDA_DEBUG_MOE_FIT", "1", 1);
    setenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_fit_debug_enabled());
    assert(bn_gpu_policy_cuda_keep_individual_f16_cache_enabled());
    unsetenv("BN_CUDA_DEBUG_MOE_FIT");
    unsetenv("BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE");

    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");
    assert(!bn_gpu_policy_cuda_moe_lazy_aux_cache_enabled());
    setenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_lazy_aux_cache_enabled());
    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");

    unsetenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE");
    assert(!bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    setenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    assert(!bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    setenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q6_K));
    unsetenv("BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE");

    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    assert(!bn_gpu_policy_q6_logits_refine_enabled(1, 0));
    assert(bn_gpu_policy_q6_logits_refine_enabled(0, 0));
    assert(bn_gpu_policy_q6_logits_refine_enabled(1, 1));
    assert(bn_gpu_policy_q6_logits_refine_top_or_default(64) == 64);
    setenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_q6_logits_refine_enabled(1, 0));
    setenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_q6_logits_refine_enabled(0, 0));
    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    assert(!bn_gpu_policy_q6_logits_refine_enabled(0, 0));
    setenv("BN_GPU_Q6_Q8K_REFINE_TOP", "11", 1);
    assert(bn_gpu_policy_q6_logits_refine_top_or_default(64) == 11);
    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");

    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_Q8_REFINE_TOP");
    assert(!bn_gpu_policy_q8_logits_refine_enabled(1, 0));
    assert(bn_gpu_policy_q8_logits_refine_enabled(0, 0));
    assert(bn_gpu_policy_q8_logits_refine_enabled(1, 1));
    assert(bn_gpu_policy_q8_logits_refine_top_or_default(16) == 16);
    setenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_q8_logits_refine_enabled(1, 0));
    setenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_gpu_policy_q8_logits_refine_enabled(0, 0));
    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    assert(!bn_gpu_policy_q8_logits_refine_enabled(0, 0));
    setenv("BN_GPU_Q8_REFINE_TOP", "5", 1);
    assert(bn_gpu_policy_q8_logits_refine_top_or_default(16) == 5);
    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_Q8_REFINE_TOP");

    unsetenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE");
    assert(!bn_gpu_policy_cuda_logits_f16_cache_enabled(&gpu));
    setenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_logits_f16_cache_enabled(&gpu));
    unsetenv("BN_CUDA_ENABLE_LOGITS_F16_CACHE");

    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    assert(bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(&gpu));
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced());
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(1024));
    assert(bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(1025));
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 1024, 0));
    assert(bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 1025, 0));
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q4_K, 2048, 0));
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 2048, 1));
    assert(bn_gpu_policy_cuda_moe_down_q6_f32_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 2048, 2) ==
           32768 * sizeof(float));
    setenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced());
    assert(bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(1));
    assert(bn_gpu_policy_cuda_moe_down_q6_f32_cache_requires_full_buffer(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_requires_full_buffer(
        BN_GGUF_TENSOR_Q4_K));
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(&gpu));
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
        &gpu, BN_GGUF_TENSOR_Q6_K, 2048, 0));
    assert(!bn_gpu_policy_cuda_moe_down_q6_f32_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 2048, 2));
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");

    assert(bn_gpu_policy_cuda_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_gpu_policy_cuda_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q8_0, 1));
    assert(bn_gpu_policy_cuda_moe_prefers_quant_only(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_gpu_policy_cuda_moe_prefers_quant_only(
        BN_GGUF_TENSOR_Q4_K));

    unsetenv("BN_CUDA_LAYOUT_RESERVE_MB");
    unsetenv("BN_CUDA_MOE_FULL_RESERVE_MB");
    assert(bn_gpu_policy_cuda_layout_reserve_bytes() ==
           512u * 1024u * 1024u);
    assert(bn_gpu_policy_cuda_moe_full_reserve_bytes() ==
           512u * 1024u * 1024u);
    setenv("BN_CUDA_LAYOUT_RESERVE_MB", "7", 1);
    setenv("BN_CUDA_MOE_FULL_RESERVE_MB", "9", 1);
    assert(bn_gpu_policy_cuda_layout_reserve_bytes() ==
           7u * 1024u * 1024u);
    assert(bn_gpu_policy_cuda_moe_full_reserve_bytes() ==
           9u * 1024u * 1024u);
    setenv("BN_CUDA_LAYOUT_RESERVE_MB", "bad", 1);
    assert(bn_gpu_policy_cuda_layout_reserve_bytes() ==
           512u * 1024u * 1024u);
    setenv("BN_CUDA_MOE_FULL_RESERVE_MB",
           "18446744073709551615", 1);
    assert(bn_gpu_policy_cuda_moe_full_reserve_bytes() == SIZE_MAX);
    unsetenv("BN_CUDA_LAYOUT_RESERVE_MB");
    unsetenv("BN_CUDA_MOE_FULL_RESERVE_MB");

    unsetenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");
    assert(!bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    setenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    setenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
        &gpu, BN_GGUF_TENSOR_Q4_K));
    unsetenv("BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE");

    unsetenv("BN_CUDA_DISABLE_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0");
    unsetenv("BN_CUDA_DISABLE_Q5_0");
    unsetenv("BN_CUDA_DISABLE_Q4_K");
    unsetenv("BN_CUDA_DISABLE_Q5_K");
    unsetenv("BN_CUDA_DISABLE_Q6_K");
    unsetenv("BN_CUDA_DISABLE_Q8_K");
    assert(!bn_gpu_policy_cuda_matvec_disabled());
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q5_0));
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q4_K));
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q8_K));
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q4_0));
    setenv("BN_CUDA_DISABLE_MATVEC", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_0", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5_0", "1", 1);
    setenv("BN_CUDA_DISABLE_Q4_K", "1", 1);
    setenv("BN_CUDA_DISABLE_Q5_K", "1", 1);
    setenv("BN_CUDA_DISABLE_Q6_K", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_K", "1", 1);
    assert(bn_gpu_policy_cuda_matvec_disabled());
    assert(bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q8_0));
    assert(bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q5_0));
    assert(bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q4_K));
    assert(bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q5_K));
    assert(bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q6_K));
    assert(bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q8_K));
    assert(!bn_gpu_policy_cuda_matvec_type_disabled(BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CUDA_DISABLE_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0");
    unsetenv("BN_CUDA_DISABLE_Q5_0");
    unsetenv("BN_CUDA_DISABLE_Q4_K");
    unsetenv("BN_CUDA_DISABLE_Q5_K");
    unsetenv("BN_CUDA_DISABLE_Q6_K");
    unsetenv("BN_CUDA_DISABLE_Q8_K");

    unsetenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
    unsetenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");
    assert(bn_gpu_policy_cuda_small_kquant_native_enabled(0));
    assert(!bn_gpu_policy_cuda_small_kquant_native_enabled(1));
    assert(!bn_gpu_policy_cuda_small_kquant_native_disabled());
    setenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE", "1", 1);
    assert(bn_gpu_policy_cuda_small_kquant_native_enabled(1));
    setenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE", "1", 1);
    assert(bn_gpu_policy_cuda_small_kquant_native_enabled(1));
    assert(bn_gpu_policy_cuda_small_kquant_native_disabled());
    unsetenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
    assert(!bn_gpu_policy_cuda_small_kquant_native_enabled(0));
    unsetenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");

    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    unsetenv("BN_GPU_PREFILL_MATMUL");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");
    assert(!bn_gpu_policy_prefill_matmul_disabled());
    assert(!bn_gpu_policy_prefill_matmul_enabled());
    assert(!bn_gpu_policy_cuda_prefill_direct_kv_disabled());
    assert(!bn_gpu_policy_cuda_prefill_direct_kv_with_cpu_fallback_enabled());
    setenv("BN_GPU_DISABLE_PREFILL_MATMUL", "1", 1);
    setenv("BN_GPU_PREFILL_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV", "1", 1);
    setenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK", "1", 1);
    assert(bn_gpu_policy_prefill_matmul_disabled());
    assert(bn_gpu_policy_prefill_matmul_enabled());
    assert(bn_gpu_policy_cuda_prefill_direct_kv_disabled());
    assert(bn_gpu_policy_cuda_prefill_direct_kv_with_cpu_fallback_enabled());
    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    unsetenv("BN_GPU_PREFILL_MATMUL");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");

    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    unsetenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_FROM_LAYER");
    assert(!bn_gpu_policy_cpu_decode_fallback_requested());
    setenv("BN_GPU_CPU_FALLBACK_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    setenv("BN_GPU_CPU_FALLBACK_FROM_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    unsetenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
    setenv("BN_GPU_CPU_ATTN_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    unsetenv("BN_GPU_CPU_ATTN_LAYER");
    setenv("BN_GPU_CPU_ATTN_FROM_LAYER", "1", 1);
    assert(bn_gpu_policy_cpu_decode_fallback_requested());
    unsetenv("BN_GPU_CPU_ATTN_FROM_LAYER");

    unsetenv("BN_CUDA_DISABLE_SSM_GRAPH");
    assert(!bn_gpu_policy_cuda_ssm_graph_disabled());
    setenv("BN_CUDA_DISABLE_SSM_GRAPH", "1", 1);
    assert(bn_gpu_policy_cuda_ssm_graph_disabled());
    unsetenv("BN_CUDA_DISABLE_SSM_GRAPH");

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
    unsetenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16");
    unsetenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    unsetenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
    assert(bn_gpu_policy_cuda_cublas_matmul_enabled());
    assert(bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled());
    assert(!bn_gpu_policy_cuda_q8_0_quant_matmul_enabled());
    assert(bn_gpu_policy_cuda_f16_q8_0_matmul_enabled());
    assert(!bn_gpu_policy_cuda_q8_0_preq_split_enabled());
    setenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL", "1", 1);
    setenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT", "1", 1);
    assert(bn_gpu_policy_cuda_q8_0_quant_matmul_enabled());
    assert(bn_gpu_policy_cuda_q8_0_preq_split_enabled());
    setenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT", "1", 1);
    assert(!bn_gpu_policy_cuda_q8_0_quant_matmul_enabled());
    assert(!bn_gpu_policy_cuda_f16_q8_0_matmul_enabled());
    assert(!bn_gpu_policy_cuda_q8_0_preq_split_enabled());
    unsetenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_ENABLE_LOGITS_CACHE");
    assert(!bn_gpu_policy_cuda_decode_logits_cache_enabled(0));
    setenv("BN_CUDA_ENABLE_LOGITS_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_decode_logits_cache_enabled(0));
    assert(!bn_gpu_policy_cuda_decode_logits_cache_enabled(1));
    unsetenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE");
    assert(!bn_gpu_policy_cuda_moe_decode_cache_enabled());
    setenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_decode_cache_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");
    assert(!bn_gpu_policy_cuda_moe_decode_cache_disabled());
    setenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_moe_decode_cache_disabled());
    unsetenv("BN_CUDA_DISABLE_DECODE_CACHE");
    assert(!bn_gpu_policy_cuda_decode_cache_disabled());
    setenv("BN_CUDA_DISABLE_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_decode_cache_disabled());
    unsetenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE");
    assert(!bn_gpu_policy_cuda_q4_q8_decode_cache_disabled());
    setenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_q4_q8_decode_cache_disabled());
    unsetenv("BN_CUDA_DISABLE_LOGITS_ARGMAX");
    assert(!bn_gpu_policy_cuda_logits_argmax_disabled());
    setenv("BN_CUDA_DISABLE_LOGITS_ARGMAX", "1", 1);
    assert(bn_gpu_policy_cuda_logits_argmax_disabled());
    unsetenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
    assert(!bn_gpu_policy_cuda_dense_logits_argmax_enabled());
    setenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX", "1", 1);
    assert(bn_gpu_policy_cuda_dense_logits_argmax_enabled());
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX");
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled());
    setenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX", "1", 1);
    assert(bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
    assert(!bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled());
    setenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX", "1", 1);
    assert(bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled());
    unsetenv("BN_CUDA_ENABLE_LOGITS_CACHE");
    unsetenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_LOGITS_ARGMAX");
    unsetenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX");
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");
    unsetenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE");
    assert(!bn_gpu_policy_cuda_prefill_ssm_layer_disabled());
    assert(!bn_gpu_policy_cuda_q5k_fused_gateup_enabled());
    assert(bn_gpu_policy_cuda_shared_q4_q8_dot_enabled());
    assert(bn_gpu_policy_cuda_shared_expert_gate_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    setenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP", "1", 1);
    setenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT", "1", 1);
    setenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE", "1", 1);
    assert(bn_gpu_policy_cuda_prefill_ssm_layer_disabled());
    assert(bn_gpu_policy_cuda_q5k_fused_gateup_enabled());
    assert(!bn_gpu_policy_cuda_shared_q4_q8_dot_enabled());
    assert(!bn_gpu_policy_cuda_shared_expert_gate_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
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
    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
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
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
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
    assert(!bn_gpu_policy_cuda_prefill_attention_min_tokens_configured());
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(16) ==
           16);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "9", 1);
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_configured());
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(16) ==
           9);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "0", 1);
    assert(bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(16) ==
           16);
    assert(bn_gpu_policy_cuda_prefill_dense_chain_enabled());
    assert(bn_gpu_policy_cuda_prefill_hybrid_chain_enabled());
    assert(bn_gpu_policy_cuda_prefill_attention_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_run_chain_enabled());
    assert(bn_gpu_policy_cuda_prefill_ssm_ffn_fuse_allowed());
    assert(!bn_gpu_policy_cuda_prefill_moe_chain_debug_enabled());
    assert(!bn_gpu_policy_cuda_prefill_hybrid_chain_debug_enabled());
    assert(!bn_gpu_policy_cuda_moe_prefill_enabled());
    assert(!bn_gpu_policy_cuda_moe_prefill_min_tokens_configured());
    assert(bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(1) == 1);
    assert(bn_gpu_policy_cuda_moe_cache_prefill_enabled());
    assert(bn_gpu_policy_cuda_moe_prefill_shared_fuse_enabled());
    assert(!bn_gpu_policy_cuda_moe_route_batch_debug_enabled());
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
    setenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH", "1", 1);
    assert(!bn_gpu_policy_cuda_prefill_dense_chain_enabled());
    assert(!bn_gpu_policy_cuda_prefill_hybrid_chain_enabled());
    assert(!bn_gpu_policy_cuda_prefill_attention_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_run_chain_enabled());
    assert(!bn_gpu_policy_cuda_prefill_ssm_ffn_fuse_allowed());
    assert(bn_gpu_policy_cuda_prefill_moe_chain_debug_enabled());
    assert(bn_gpu_policy_cuda_prefill_hybrid_chain_debug_enabled());
    assert(bn_gpu_policy_cuda_moe_prefill_enabled());
    assert(bn_gpu_policy_cuda_moe_prefill_min_tokens_configured());
    assert(bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(1) == 7);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "0", 1);
    assert(bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(1) == 1);
    assert(!bn_gpu_policy_cuda_moe_cache_prefill_enabled());
    assert(!bn_gpu_policy_cuda_moe_prefill_shared_fuse_enabled());
    assert(bn_gpu_policy_cuda_moe_route_batch_debug_enabled());
    assert(!bn_gpu_policy_cuda_large_hybrid_attention_enabled());
    assert(!bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_enabled());
    assert(!bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_disabled());
    assert(!bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_forced());
    assert(!bn_gpu_policy_cuda_large_hybrid_prefill_enabled());
    assert(!bn_gpu_policy_cuda_large_hybrid_prefill_chain_enabled());
    assert(!bn_gpu_policy_cuda_large_hybrid_prefill_disabled());
    assert(!bn_gpu_policy_cuda_large_hybrid_argmax_enabled());
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN", "1", 1);
    setenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL", "1", 1);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX", "1", 1);
    assert(bn_gpu_policy_cuda_large_hybrid_attention_enabled());
    assert(bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_enabled());
    assert(bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_disabled());
    assert(bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_forced());
    assert(bn_gpu_policy_cuda_large_hybrid_prefill_enabled());
    assert(bn_gpu_policy_cuda_large_hybrid_prefill_chain_enabled());
    assert(bn_gpu_policy_cuda_large_hybrid_prefill_disabled());
    assert(bn_gpu_policy_cuda_large_hybrid_argmax_enabled());
    assert(bn_gpu_policy_fused_gateup_enabled());
    assert(bn_gpu_policy_q4_q8_fused_gateup_enabled());
    assert(bn_gpu_policy_gateup_split_enabled());
    assert(bn_gpu_policy_q4_q8_ffn_down_enabled());
    assert(bn_gpu_policy_qkv_split_enabled());
    assert(!bn_gpu_policy_qkv_split_debug_enabled());
    assert(bn_gpu_policy_ssm_qkvz_split_enabled());
    assert(bn_gpu_policy_ssm_ab_stack_enabled());
    assert(!bn_gpu_policy_split_residual_rmsnorm_enabled());
    assert(!bn_gpu_policy_debug_fallback_enabled());
    assert(!bn_gpu_policy_force_graph_enabled());
    assert(bn_gpu_policy_flash_min_kv_or_default(0) == 0);
    assert(bn_gpu_policy_flash_max_kv_or_default(1, 0) == 2048);
    assert(bn_gpu_policy_flash_max_kv_or_default(0, 0) == 0);
    assert(!bn_gpu_policy_cpu_logits_enabled());
    assert(!bn_gpu_policy_compare_logits_enabled());
    assert(!bn_gpu_policy_debug_argmax_compare_enabled());
    assert(!bn_gpu_policy_cuda_moe_ffn_disabled());
    assert(bn_gpu_policy_cuda_moe_router_topk_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_router_topk_enabled(0));
    assert(bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled(1));
    assert(!bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled(0));
    assert(!bn_gpu_policy_cuda_moe_router_gpu_enabled());
    assert(bn_gpu_policy_cuda_moe_router_diff2_enabled());
    assert(bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(0));
    assert(!bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(1));
    assert(!bn_gpu_policy_cuda_moe_cpu_actual_override_enabled());
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
    assert(!bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled(1));
    assert(bn_gpu_policy_cuda_moe_gateup_split_enabled(1));
    assert(!bn_gpu_policy_cuda_moe_gateup_split_enabled(0));
    assert(!bn_gpu_policy_moe_route_profile_enabled());
    assert(bn_gpu_policy_moe_route_profile_every_or_default(28) == 28);
    setenv("BN_GPU_DISABLE_FUSED_GATEUP", "1", 1);
    setenv("BN_GPU_Q4_Q8_DISABLE_GATEUP", "1", 1);
    setenv("BN_GPU_DISABLE_GATEUP_SPLIT", "1", 1);
    setenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN", "1", 1);
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
    setenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    setenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2", "1", 1);
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
    setenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    setenv("BN_GPU_MOE_ROUTE_PROFILE", "1", 1);
    setenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY", "5", 1);
    assert(!bn_gpu_policy_fused_gateup_enabled());
    assert(!bn_gpu_policy_q4_q8_fused_gateup_enabled());
    assert(!bn_gpu_policy_gateup_split_enabled());
    assert(!bn_gpu_policy_q4_q8_ffn_down_enabled());
    assert(!bn_gpu_policy_qkv_split_enabled());
    assert(bn_gpu_policy_qkv_split_debug_enabled());
    assert(!bn_gpu_policy_ssm_qkvz_split_enabled());
    assert(!bn_gpu_policy_ssm_ab_stack_enabled());
    assert(bn_gpu_policy_split_residual_rmsnorm_enabled());
    assert(bn_gpu_policy_debug_fallback_enabled());
    assert(bn_gpu_policy_force_graph_enabled());
    assert(bn_gpu_policy_flash_min_kv_or_default(0) == 32);
    assert(bn_gpu_policy_flash_max_kv_or_default(1, 0) == 1024);
    assert(bn_gpu_policy_flash_max_kv_or_default(0, 0) == 1024);
    assert(bn_gpu_policy_cpu_logits_enabled());
    assert(bn_gpu_policy_compare_logits_enabled());
    assert(bn_gpu_policy_debug_argmax_compare_enabled());
    assert(bn_gpu_policy_cuda_moe_ffn_disabled());
    assert(!bn_gpu_policy_cuda_moe_router_topk_enabled(1));
    assert(!bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled(1));
    assert(bn_gpu_policy_cuda_moe_router_gpu_enabled());
    assert(!bn_gpu_policy_cuda_moe_router_diff2_enabled());
    assert(bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(1));
    setenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_router_gpu_enabled());
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(0));
    assert(!bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(1));
    assert(bn_gpu_policy_cuda_moe_cpu_actual_override_enabled());
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
    assert(!bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled(0));
    assert(bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT", "1", 1);
    assert(!bn_gpu_policy_cuda_moe_gateup_split_enabled(1));
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
    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
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
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
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
    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
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
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
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
    assert(bn_gpu_policy_backend_is_cuda(&gpu));
    assert(bn_gpu_policy_cuda_cublas_cache_max_mb(128, 0) == 128);
    assert(bn_gpu_policy_cuda_cublas_cache_max_mb(128, 1) == 512);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q8_0, 0, 0) == 128);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q4_K, 0, 0) == 512);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q6_K, 0, 1) == 0);
    assert(bn_gpu_policy_cuda_q6k_f16_cache_adds_f32_down_cache());
    assert(bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16) == 128 * sizeof(uint16_t));
    assert(bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 16) == 0);
    assert(bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 32) == 256 * sizeof(uint16_t));
    assert(bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_Q4_K, 8, 16) == 0);
    assert(bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_Q4_K, 8, 32) == 256 * sizeof(uint16_t));
    assert(bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_I2_S, 8, 32) == 0);
    assert(!bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q4_K, 8, 16));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_gpu_policy_backend_is_cuda(&gpu));
    assert(!bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    setenv("BN_CUDA_CUBLAS_CACHE_MAX_MB", "7", 1);
    assert(bn_gpu_policy_cuda_cublas_cache_max_mb(128, 1) == 7);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q6_K, 1, 0) == 7);
    setenv("BN_CUDA_DISABLE_CUBLAS_MATMUL", "1", 1);
    setenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16", "1", 1);
    setenv("BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_cublas_matmul_enabled());
    assert(!bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled());
    assert(!bn_gpu_policy_cuda_q6k_f16_cache_adds_f32_down_cache());
    assert(!bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16));
    assert(!bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 16));
    unsetenv("BN_CUDA_DISABLE_CUBLAS_MATMUL");
    unsetenv("BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE");
    assert(bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
        &gpu, BN_GGUF_TENSOR_Q6_K, 8, 16) == 128 * sizeof(float));
    assert(bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 8, 32) == 256 * sizeof(float));
    setenv("BN_CUDA_CUBLAS_CACHE_MAX_MB", "1", 1);
    assert(!bn_gpu_policy_cuda_aux_cache_bytes(
        BN_GGUF_TENSOR_Q6_K, 2048, 1024));
    unsetenv("BN_CUDA_CUBLAS_CACHE_MAX_MB");
    setenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        BN_GGUF_TENSOR_Q6_K, 1, 0) == 0);
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    setenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_q6k_f16_cache_adds_f32_down_cache());
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    assert(bn_gpu_policy_moe_auto_resident_enabled());
    assert(!bn_gpu_policy_cuda_duplicate_moe_cache_enabled());
    setenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT", "1", 1);
    setenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE", "1", 1);
    assert(!bn_gpu_policy_moe_auto_resident_enabled());
    assert(bn_gpu_policy_cuda_duplicate_moe_cache_enabled());
    setenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE", "1", 1);
    assert(!bn_gpu_policy_cuda_duplicate_moe_cache_enabled());
    unsetenv("BN_METAL_ENABLE_MMAP_ZERO_COPY");
    assert(!bn_gpu_policy_metal_mmap_zero_copy_enabled());
    setenv("BN_METAL_ENABLE_MMAP_ZERO_COPY", "1", 1);
    assert(bn_gpu_policy_metal_mmap_zero_copy_enabled());
    unsetenv("BN_METAL_SHARED_WEIGHTS");
    assert(!bn_gpu_policy_metal_shared_weights_enabled());
    setenv("BN_METAL_SHARED_WEIGHTS", "1", 1);
    assert(bn_gpu_policy_metal_shared_weights_enabled());
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    assert(!bn_gpu_policy_metal_q6_q8k_enabled());
    setenv("BN_METAL_ENABLE_Q6_Q8K", "1", 1);
    assert(bn_gpu_policy_metal_q6_q8k_enabled());
    unsetenv("BN_METAL_Q8_BARRIERS");
    assert(!bn_gpu_policy_metal_q8_barriers_enabled());
    setenv("BN_METAL_Q8_BARRIERS", "1", 1);
    assert(bn_gpu_policy_metal_q8_barriers_enabled());
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
    setenv("BN_METAL_ENABLE_BARRIERS", "1", 1);
    assert(!bn_gpu_policy_metal_full_barriers_enabled());
    assert(bn_gpu_policy_metal_barriers_enabled());
    assert(!bn_gpu_policy_metal_barriers_disabled());
    setenv("BN_METAL_FULL_BARRIERS", "1", 1);
    assert(bn_gpu_policy_metal_full_barriers_enabled());
    assert(bn_gpu_policy_metal_barriers_enabled());
    setenv("BN_METAL_DISABLE_BARRIERS", "1", 1);
    assert(bn_gpu_policy_metal_barriers_disabled());
    unsetenv("BN_GPU_Q4_Q8");
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    unsetenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT");
    unsetenv("BN_METAL_Q4_PREPARED");
    assert(!bn_gpu_policy_metal_q4_q8_enabled());
    assert(!bn_gpu_policy_q4_q8_attn_only_enabled());
    assert(!bn_gpu_policy_q4_q8_ffn_only_enabled());
    assert(bn_gpu_policy_q4_q8_from_layer_or_default(40) == -1);
    assert(bn_gpu_policy_q4_q8_to_layer_or_default(40, 0) == -1);
    assert(!bn_gpu_policy_metal_q4_prepared_enabled());
    assert(!bn_gpu_policy_metal_q4_prepared_upload_enabled());
    bn_gpu_policy_metal_apply_q4_q8_default();
    assert(bn_gpu_policy_metal_q4_q8_enabled());
    assert(getenv("BN_GPU_Q4_Q8_FROM_LAYER") != NULL);
    assert(bn_gpu_policy_q4_q8_from_layer_or_default(40) == 0);
    assert(bn_gpu_policy_q4_q8_to_layer_or_default(40, 0) == 6);
    assert(!bn_gpu_policy_q4_q8_attn_only_enabled());
    assert(bn_gpu_policy_q4_q8_ffn_only_enabled());
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    setenv("BN_GPU_Q4_Q8_FROM_LAYER", "10", 1);
    setenv("BN_GPU_Q4_Q8_TO_LAYER", "20", 1);
    setenv("BN_GPU_Q4_Q8_ATTN_ONLY", "1", 1);
    setenv("BN_GPU_Q4_Q8_FFN_ONLY", "1", 1);
    assert(bn_gpu_policy_q4_q8_from_layer_or_default(40) == 10);
    assert(bn_gpu_policy_q4_q8_to_layer_or_default(40, 0) == 20);
    assert(bn_gpu_policy_q4_q8_attn_only_enabled());
    assert(bn_gpu_policy_q4_q8_ffn_only_enabled());
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    setenv("BN_GPU_Q4_Q8_TAIL_NATIVE", "4", 1);
    assert(bn_gpu_policy_q4_q8_to_layer_or_default(40, 0) == 35);
    setenv("BN_GPU_Q4_Q8_TAIL_NATIVE", "100", 1);
    assert(bn_gpu_policy_q4_q8_to_layer_or_default(40, 0) == -1);
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    setenv("BN_METAL_Q4_PREPARED", "1", 1);
    assert(bn_gpu_policy_metal_q4_prepared_enabled());
    assert(bn_gpu_policy_metal_q4_prepared_upload_enabled());
    setenv("BN_GPU_Q4_Q8_FROM_LAYER", "1", 1);
    assert(!bn_gpu_policy_metal_q4_prepared_upload_enabled());
    setenv("BN_GPU_Q4_Q8_FROM_LAYER", "0", 1);
    setenv("BN_GPU_Q4_Q8_ATTN_ONLY", "1", 1);
    assert(!bn_gpu_policy_metal_q4_prepared_upload_enabled());
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    setenv("BN_GPU_Q4_Q8_FFN_ONLY", "1", 1);
    assert(!bn_gpu_policy_metal_q4_prepared_upload_enabled());
    unsetenv("BN_GPU_Q4_Q8");
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    setenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT", "1", 1);
    bn_gpu_policy_metal_apply_q4_q8_default();
    assert(!bn_gpu_policy_metal_q4_q8_enabled());
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
    unsetenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE");
    unsetenv("BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE");
    unsetenv("BN_CUDA_DISABLE_MATVEC");
    unsetenv("BN_CUDA_DISABLE_Q8_0");
    unsetenv("BN_CUDA_DISABLE_Q5_0");
    unsetenv("BN_CUDA_DISABLE_Q4_K");
    unsetenv("BN_CUDA_DISABLE_Q5_K");
    unsetenv("BN_CUDA_DISABLE_Q6_K");
    unsetenv("BN_CUDA_DISABLE_Q8_K");
    unsetenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL");
    unsetenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL");
    unsetenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT");
    unsetenv("BN_GPU_MOE_DISABLE_AUTO_RESIDENT");
    unsetenv("BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE");
    unsetenv("BN_METAL_ENABLE_MMAP_ZERO_COPY");
    unsetenv("BN_METAL_SHARED_WEIGHTS");
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    unsetenv("BN_METAL_Q8_BARRIERS");
    unsetenv("BN_METAL_CPU_ORDER_RMSNORM");
    unsetenv("BN_METAL_FULL_BARRIERS");
    unsetenv("BN_METAL_ENABLE_BARRIERS");
    unsetenv("BN_METAL_DISABLE_BARRIERS");
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
    assert(bn_backend_quant_cuda_small_dense_supported(BN_GGUF_TENSOR_F32));
    assert(bn_backend_quant_cuda_small_dense_supported(BN_GGUF_TENSOR_F16));
    assert(bn_backend_quant_cuda_small_dense_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_supports_gpu_small_dense(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CAP_Q4_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CAP_Q4_FUSED_GATEUP_SILU);
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
    assert(!bn_backend_quant_cuda_small_dense_supported(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_supports_gpu_small_dense(BN_GGUF_TENSOR_I2_S));
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
           BN_GPU_CAP_Q5_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q5_0) ==
           BN_GPU_CAP_Q5_FUSED_GATEUP_SILU);
    assert(!bn_quant_format_has_cap(99999, BN_QUANT_CAP_LOADABLE));
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q8_0) ==
           BN_GPU_CAP_Q8_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q5_K) ==
           BN_GPU_CAP_Q5K_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q8_0) ==
           BN_GPU_CAP_Q8_FUSED_GATEUP_SILU);
    assert(bn_backend_quant_cuda_small_dense_q8_supported(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_cuda_small_dense_q8_supported(BN_GGUF_TENSOR_Q4_0));
    assert(bn_backend_quant_supports_q8_logits_refine(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_backend_quant_supports_q8_logits_refine(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_gpu_small_dense_q8(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_q8_logits_refine(BN_GGUF_TENSOR_Q8_0));
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
    unsetenv("BN_CPU_TIED_Q6K_REFINE_TOP");
    unsetenv("BN_CPU_TIED_Q6K_HYBRID_TOP");
    assert(bn_backend_quant_cpu_tied_q6k_refine_top() == 0);
    assert(bn_backend_quant_cpu_tied_q6k_hybrid_top() == 0);
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "1", 1);
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "2", 1);
    assert(bn_backend_quant_cpu_tied_q6k_refine_top() == 1);
    assert(bn_backend_quant_cpu_tied_q6k_hybrid_top() == 2);
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "0", 1);
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "1", 1);
    assert(bn_backend_quant_cpu_tied_q6k_refine_top() == 0);
    assert(bn_backend_quant_cpu_tied_q6k_hybrid_top() == 0);
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "256", 1);
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "256", 1);
    assert(bn_backend_quant_cpu_tied_q6k_refine_top() == 128);
    assert(bn_backend_quant_cpu_tied_q6k_hybrid_top() == 128);
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
    assert(bn_transformer_cpu_has_native_q8x_quant() ==
           TEST_EXPECT_NATIVE_Q8X);
    assert(bn_quant_format_gpu_requires_exact_silu(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_requires_exact_silu(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_can_preq8k(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_is_kquant_float_fallback_candidate(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_is_float_kquant_fallback_candidate(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_can_gpu_split(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_gpu_split_cap(BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_CAP_Q4K_MATVEC_SPLIT);
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_CAP_Q4_FUSED_GATEUP_SILU);
    assert(bn_quant_format_gpu_matvec_q8k_dot_flag(BN_GGUF_TENSOR_Q4_K, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_Q8K_DOT);
    assert(bn_quant_format_gpu_matvec_q8k_dot_flag(BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_quant_format_gpu_matvec_q8k_dot_flag(BN_GGUF_TENSOR_Q8_0, 1) == 0);
    assert(bn_quant_format_supports_moe_q4_gateup(BN_GGUF_TENSOR_Q4_K,
                                          BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_supports_moe_q4_gateup(BN_GGUF_TENSOR_Q4_K,
                                           BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_cpu_fused_q4_gateup_silu(BN_GGUF_TENSOR_Q4_0,
                                                     BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_supports_cpu_fused_q4_gateup_silu(BN_GGUF_TENSOR_Q4_0,
                                                      BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_pair_same_format(BN_GGUF_TENSOR_Q4_K,
                                                     BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_pair_same_format(BN_GGUF_TENSOR_Q4_K,
                                                      BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_supports_moe_q4_down_route(BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_quant_format_supports_moe_q4_down_route(BN_GGUF_TENSOR_Q4_K,
                                               BN_GGUF_TENSOR_Q4_K,
                                               BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_supports_moe_q4_down_route(BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q4_K,
                                              BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_quant_format_supports_moe_q8_route(BN_GGUF_TENSOR_Q8_0,
                                                 BN_GGUF_TENSOR_Q8_0,
                                                 BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_supports_moe_q8_route(BN_GGUF_TENSOR_Q8_0,
                                                  BN_GGUF_TENSOR_Q8_0,
                                                  BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_gpu_allows_gateup_split_activation(BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_quant_format_gpu_allows_gateup_split_activation(BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_can_preq8k(BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_is_kquant_float_fallback_candidate(BN_GGUF_TENSOR_Q6_K));
    assert(bn_backend_quant_supports_q6k_logits_refine(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_backend_quant_supports_q6k_logits_refine(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_q6_logits_refine(BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_cuda_logits_q6_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_cuda_logits_q6_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cuda_moe_down_q6_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_cuda_moe_down_q6_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cuda_moe_down_cublas_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_cuda_moe_down_cublas_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes(
        BN_GGUF_TENSOR_Q6_K, 1) == (int)sizeof(uint16_t));
    assert(bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes(
        BN_GGUF_TENSOR_Q6_K, 0) == (int)sizeof(float));
    assert(bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes(
        BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_quant_format_cuda_moe_down_q4_f32_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_cuda_moe_down_q4_f32_cache_supported(
        BN_GGUF_TENSOR_Q6_K));
    assert(bn_quant_format_gpu_matvec_exact_q6k_flag(BN_GGUF_TENSOR_Q6_K, 1) ==
           BN_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K);
    assert(bn_quant_format_gpu_matvec_exact_q6k_flag(BN_GGUF_TENSOR_Q6_K, 0) == 0);
    assert(bn_quant_format_gpu_matvec_exact_q6k_flag(BN_GGUF_TENSOR_Q4_K, 1) == 0);
    assert(bn_quant_format_can_preq8k(BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_is_kquant_float_fallback_candidate(BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_cuda_small_dense_supported(BN_GGUF_TENSOR_Q5_K));
    assert(bn_backend_quant_cuda_small_dense_supported(BN_GGUF_TENSOR_Q8_K));
    assert(!bn_quant_format_can_gpu_native(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_can_gpu_repack(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_can_cpu_repack(BN_GGUF_TENSOR_Q5_K));
    assert(bn_quant_format_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q5_K) ==
           BN_GPU_CAP_Q5K_FUSED_GATEUP_SILU);
    assert(bn_quant_format_gpu_fused_gateup_requires_cuda_opt_in(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_gpu_fused_gateup_requires_cuda_opt_in(BN_GGUF_TENSOR_Q4_K));
    assert(!bn_backend_quant_is_kquant_float_fallback_candidate(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_cuda_moe_all_f16_cache_supported(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_cuda_moe_all_f16_cache_supported(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_cuda_moe_all_f16_cache_supported(
        BN_GGUF_TENSOR_Q4_0));
    assert(!bn_quant_format_cuda_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q8_0, 1));
    assert(bn_quant_format_cuda_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_quant_format_cuda_moe_quant_only_after_cache(
        BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_quant_format_cuda_lazy_moe_aux_cache_candidate(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cuda_lazy_moe_aux_cache_candidate(
        BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_cuda_lazy_moe_aux_cache_candidate(
        BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_cuda_moe_prefers_quant_only(
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_cuda_moe_prefers_quant_only(
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cuda_aux_cache_supported(BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_cuda_aux_cache_supported(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cuda_aux_cache_supported(BN_GGUF_TENSOR_IQ4_XS));
    assert(!bn_quant_format_cuda_aux_cache_supported(BN_GGUF_TENSOR_I2_S));
    assert(bn_quant_format_cuda_aux_cache_can_use_f16(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_cuda_aux_cache_can_use_f16(BN_GGUF_TENSOR_Q4_K));
    assert(bn_quant_format_cuda_aux_cache_uses_f32(BN_GGUF_TENSOR_Q6_K, 0));
    assert(!bn_quant_format_cuda_aux_cache_uses_f32(BN_GGUF_TENSOR_Q6_K, 1));
    assert(!bn_quant_format_cuda_aux_cache_uses_f32(BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_cuda_aux_cache_prefers_large_budget(
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_quant_format_cuda_aux_cache_prefers_large_budget(
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_quant_format_supports_moe_q8_route(BN_GGUF_TENSOR_Q8_0,
                                         BN_GGUF_TENSOR_Q8_0,
                                         BN_GGUF_TENSOR_Q8_0));
    assert(!bn_quant_format_supports_moe_q8_route(BN_GGUF_TENSOR_Q8_0,
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
    assert(stats.q4_repack_bytes == bytes);
    assert(stats.q4k_scale_bytes == 0);
    assert(stats.q6k_weight_bytes == 0);
    assert(stats.q8_scale_bytes == 0);

    SHArena *arena = sh_arena_create(bytes + 4 * SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnBackendModel *backend = bn_backend_model_create();
    assert(backend != NULL);

    BnBackendLayoutPreparedStats built = {0};
    bn_backend_layout_prepare_qweights(backend, &config, &weights, arena, &built);
    assert(built.q4_repack_bytes == stats.q4_repack_bytes);
    assert(built.q4k_scale_bytes == 0);
    assert(built.q6k_weight_bytes == 0);
    assert(built.q8_scale_bytes == 0);
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
    assert(stats.q4_repack_bytes == 0);
    assert(stats.q4k_scale_bytes == 0);
    assert(stats.q6k_weight_bytes == 0);
    assert(stats.q8_scale_bytes == 0);
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
    assert(q4k_stats.q4_repack_bytes == 0);
    assert(q4k_stats.q4k_scale_bytes == q4k_bytes);
    assert(q4k_stats.q6k_weight_bytes == 0);
    assert(q4k_stats.q8_scale_bytes == 0);

    SHArena *q4k_arena = sh_arena_create(q4k_bytes + 4 * SH_ARENA_ALIGN);
    assert(q4k_arena != NULL);
    BnBackendModel *q4k_backend = bn_backend_model_create();
    assert(q4k_backend != NULL);
    BnBackendLayoutPreparedStats q4k_built = {0};
    bn_backend_layout_prepare_qweights(q4k_backend, &config, &q4k_weights,
                                       q4k_arena, &q4k_built);
    assert(q4k_built.q4_repack_bytes == 0);
    assert(q4k_built.q4k_scale_bytes == q4k_stats.q4k_scale_bytes);
    assert(q4k_built.q6k_weight_bytes == 0);
    assert(q4k_built.q8_scale_bytes == 0);
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
    assert(q4k_stats.q4_repack_bytes == 0);
    assert(q4k_stats.q4k_scale_bytes == 0);
    assert(q4k_stats.q6k_weight_bytes == 0);
    assert(q4k_stats.q8_scale_bytes == 0);
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
