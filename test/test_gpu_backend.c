#include "gpu_backend.h"
#include "quant.h"
#include "model.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

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

    model.weights.layers = calloc(1, sizeof(BnLayerWeights));
    assert(model.weights.layers);

    // Create a simple I2_S weight for wq
    float scale = 1.0f;
    uint8_t *wq_data = make_i2s_data(128, 128, scale);
    model.weights.layers[0].wq.data = wq_data;
    model.weights.layers[0].wq.type = BN_GGUF_TENSOR_I2_S;
    model.weights.layers[0].wq.rows = 128;
    model.weights.layers[0].wq.cols = 128;
    model.weights.layers[0].wq.scale = scale;

    // Create ffn_up weight
    uint8_t *up_data = make_i2s_data(256, 128, scale);
    model.weights.layers[0].ffn_up.data = up_data;
    model.weights.layers[0].ffn_up.type = BN_GGUF_TENSOR_I2_S;
    model.weights.layers[0].ffn_up.rows = 256;
    model.weights.layers[0].ffn_up.cols = 128;
    model.weights.layers[0].ffn_up.scale = scale;

    int rc = bn_model_upload_weights(&model, &mock_gpu);
    assert(rc == 0);
    assert(model.gpu == &mock_gpu);
    assert(model.weights.layers[0].wq.gpu_buf != NULL);
    assert(model.weights.layers[0].ffn_up.gpu_buf != NULL);
    // Weights with no data should have NULL gpu_buf
    assert(model.weights.layers[0].wk.gpu_buf == NULL);

    bn_model_release_gpu(&model);
    free(model.weights.layers);
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
    W.gpu_buf = mock_gpu.buffer_create(mock_gpu.ctx, W.data, sz,
                                        W.type, W.rows, W.cols);
    assert(W.gpu_buf != NULL);

    // Input: all 1.0
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = 1.0f;

    // GPU path
    float out_gpu = 0;
    int8_t scratch[128];
    bn_quant_matvec_gpu(&out_gpu, &W, x, scratch, NULL, &mock_gpu);

    // CPU path
    float out_cpu = 0;
    void *saved_buf = W.gpu_buf;
    W.gpu_buf = NULL;  // force CPU
    bn_quant_matvec_gpu(&out_cpu, &W, x, scratch, NULL, &mock_gpu);

    assert(fabsf(out_gpu - out_cpu) < 1e-3f);

    mock_gpu.buffer_destroy(mock_gpu.ctx, saved_buf);
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
    W.gpu_buf = NULL;  // no GPU buffer -> should fall back to CPU

    float x[128];
    for (int i = 0; i < 128; i++) x[i] = 1.0f;

    float out = 0;
    int8_t scratch[128];
    bn_quant_matvec_gpu(&out, &W, x, scratch, NULL, &mock_gpu);

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
    model.weights.layers[0].wq.data = data;
    model.weights.layers[0].wq.type = BN_GGUF_TENSOR_I2_S;
    model.weights.layers[0].wq.rows = 128;
    model.weights.layers[0].wq.cols = 128;
    model.weights.layers[0].wq.scale = scale;

    int rc = bn_model_upload_weights(&model, &mock_gpu);
    assert(rc == 0);
    assert(model.weights.layers[0].wq.gpu_buf != NULL);

    bn_model_release_gpu(&model);
    assert(model.weights.layers[0].wq.gpu_buf == NULL);
    assert(model.gpu == NULL);

    // Safe to call again
    bn_model_release_gpu(&model);

    free(model.weights.layers);
    free(data);

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
    W1.gpu_buf = mock_gpu.buffer_create(mock_gpu.ctx, W1.data, sz, W1.type, W1.rows, W1.cols);
    W2.gpu_buf = mock_gpu.buffer_create(mock_gpu.ctx, W2.data, sz, W2.type, W2.rows, W2.cols);
    assert(W1.gpu_buf && W2.gpu_buf);

    float x[128];
    for (int i = 0; i < 128; i++) x[i] = 1.0f;

    float out1_gpu = 0, out2_gpu = 0;
    int8_t scratch[128];
    BnMatvecTask tasks[2] = {
        { &out1_gpu, &W1 },
        { &out2_gpu, &W2 },
    };
    bn_quant_matvec_batch_gpu(tasks, 2, x, scratch, NULL, &mock_gpu);

    // CPU reference
    float out1_cpu = 0, out2_cpu = 0;
    BnQWeight W1c = W1; W1c.gpu_buf = NULL;
    BnQWeight W2c = W2; W2c.gpu_buf = NULL;
    BnMatvecTask cpu_tasks[2] = {
        { &out1_cpu, &W1c },
        { &out2_cpu, &W2c },
    };
    bn_quant_matvec_batch(cpu_tasks, 2, x, scratch, NULL);

    assert(fabsf(out1_gpu - out1_cpu) < 1e-3f);
    assert(fabsf(out2_gpu - out2_cpu) < 1e-3f);

    mock_gpu.buffer_destroy(mock_gpu.ctx, W1.gpu_buf);
    mock_gpu.buffer_destroy(mock_gpu.ctx, W2.gpu_buf);
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

int main(void) {
    test_data_size();
    test_gpu_upload_weights();
    test_gpu_matvec();
    test_gpu_fallback();
    test_gpu_release();
    test_gpu_batch();
    printf("All GPU backend tests PASSED\n");
    return 0;
}
