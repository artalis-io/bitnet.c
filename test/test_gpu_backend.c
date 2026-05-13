#include "gpu_backend.h"
#include "backend_quant.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "quant.h"
#include "model.h"
#include "gguf.h"
#include "sh_arena.h"
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
         { &out1_gpu, &W1, NULL },
         { &out2_gpu, &W2, NULL },
    };
    bn_backend_quant_matvec_batch_gpu(tasks, 2, x, scratch, NULL, &mock_gpu);
    float out1_gpu_buf = 0, out2_gpu_buf = 0;
    BnMatvecTask buf_tasks[2] = {
         { &out1_gpu_buf, &W1, NULL },
         { &out2_gpu_buf, &W2, NULL },
    };
    const void *bufs[2] = { W1_buf, W2_buf };
    bn_backend_quant_matvec_batch_gpu_buf(buf_tasks, bufs, 2, x, scratch, NULL,
                                  &mock_gpu);

    // CPU reference
    float out1_cpu = 0, out2_cpu = 0;
    BnQWeight W1c = W1;
    BnQWeight W2c = W2;
    BnMatvecTask cpu_tasks[2] = {
         { &out1_cpu, &W1c, NULL },
         { &out2_cpu, &W2c, NULL },
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
    assert(bn_backend_quant_can_gpu_native(BN_GGUF_TENSOR_Q4_0));
    assert(bn_backend_quant_can_gpu_repack(BN_GGUF_TENSOR_Q4_0));
    assert(bn_quant_format_can_cpu_repack(BN_GGUF_TENSOR_Q4_0));
    assert(bn_backend_quant_gpu_split_cap(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CAP_Q4_MATVEC_SPLIT);
    assert(bn_backend_quant_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CAP_Q4_FUSED_GATEUP_SILU);
    assert(bn_backend_quant_can_gpu_gateup_split_activation(BN_GGUF_TENSOR_Q4_0, 1));
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
    assert(!bn_backend_quant_can_gpu_native(BN_GGUF_TENSOR_I2_S));
    assert(!bn_backend_quant_can_gpu_repack(BN_GGUF_TENSOR_I2_S));
    assert(!bn_quant_format_can_cpu_repack(BN_GGUF_TENSOR_I2_S));
    assert(bn_backend_quant_gpu_split_cap(BN_GGUF_TENSOR_I2_S) == 0);
    assert(bn_backend_quant_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_I2_S) == 0);
    assert(bn_quant_format_data_size(BN_GGUF_TENSOR_I2_S, 1, 128) == 36);

    assert(bn_quant_format_supported(BN_GGUF_TENSOR_Q5_1));
    assert(!bn_quant_format_supported(BN_GGUF_TENSOR_Q5_0));
    assert(!bn_quant_format_has_cpu_matvec(BN_GGUF_TENSOR_Q5_0));
    assert(bn_quant_format_matvec(BN_GGUF_TENSOR_Q5_0) == NULL);
    assert(bn_quant_format_matmul(BN_GGUF_TENSOR_Q5_0) == NULL);
    assert(!bn_quant_format_has_cap(99999, BN_QUANT_CAP_LOADABLE));
    assert(bn_backend_quant_gpu_split_cap(BN_GGUF_TENSOR_Q8_0) ==
           BN_GPU_CAP_Q8_MATVEC_SPLIT);
    assert(bn_backend_quant_gpu_split_cap(BN_GGUF_TENSOR_Q5_K) ==
           BN_GPU_CAP_Q5K_MATVEC_SPLIT);
    assert(bn_backend_quant_gpu_fused_gateup_silu_cap(BN_GGUF_TENSOR_Q8_0) == 0);
    assert(bn_quant_format_can_preq8k(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_can_gpu_split(BN_GGUF_TENSOR_Q4_K));
    assert(bn_backend_quant_gpu_split_cap(BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_CAP_Q4K_MATVEC_SPLIT);
    assert(!bn_backend_quant_can_gpu_gateup_split_activation(BN_GGUF_TENSOR_Q4_K, 1));
    assert(bn_backend_quant_can_gpu_gateup_split_activation(BN_GGUF_TENSOR_Q4_K, 0));
    assert(bn_quant_format_can_preq8k(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_quant_format_can_preq8k(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_can_gpu_native(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_backend_quant_can_gpu_repack(BN_GGUF_TENSOR_Q5_K));
    assert(!bn_quant_format_can_cpu_repack(BN_GGUF_TENSOR_Q5_K));
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
    assert(bn_backend_layout_stackable_reason(&a, &b) == BN_BACKEND_LAYOUT_I2S_NOT_STACKABLE);
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
    assert(stats.q8_scale_bytes == 0);

    SHArena *arena = sh_arena_create(bytes + 4 * SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnBackendModel *backend = bn_backend_model_create();
    assert(backend != NULL);

    BnBackendLayoutPreparedStats built = {0};
    bn_backend_layout_prepare_qweights(backend, &config, &weights, arena, &built);
    assert(built.q4_repack_bytes == stats.q4_repack_bytes);
    assert(built.q8_scale_bytes == 0);
    const BnPreparedWeight *prepared =
        bn_backend_model_prepared_qweight(backend, &layer.attn.wq);
    assert(prepared != NULL);
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
    assert(stats.q8_scale_bytes == 0);
#endif

    printf("PASSED\n");
}

int main(void) {
    test_data_size();
    test_quant_registry();
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
