#include "quant.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// --- Integration test: dispatch routing ---
// Verifies that bn_quant_matvec dispatches correctly for each format.

static void test_dispatch_routing(void) {
    printf("test_dispatch_routing... ");

    // TQ2_0: all +1, dot with all-ones = 256
    BnBlockTQ2 *tq2 = (BnBlockTQ2 *)calloc(1, sizeof(BnBlockTQ2));
    for (int i = 0; i < 64; i++) tq2->qs[i] = 0xAA;
    tq2->d = 0x3C00;
    BnQWeight W_tq2 = { tq2, BN_GGUF_TENSOR_TQ2_0, 1, 256, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;
    float out;
    int8_t x_q[256];

    bn_quant_matvec(&out, &W_tq2, x, x_q, NULL);
    assert(fabsf(out - 256.0f) < 1e-3f);

    // Q8_0: all qs=1, scale=1 → dot = 32
    BnBlockQ8_0 *q8 = (BnBlockQ8_0 *)calloc(1, sizeof(BnBlockQ8_0));
    q8->d = 0x3C00;
    for (int i = 0; i < 32; i++) q8->qs[i] = 1;
    BnQWeight W_q8 = { q8, BN_GGUF_TENSOR_Q8_0, 1, 32, 1.0f };

    float x32[32];
    for (int i = 0; i < 32; i++) x32[i] = 1.0f;
    int8_t x_q32[32];

    bn_quant_matvec(&out, &W_q8, x32, x_q32, NULL);
    assert(fabsf(out - 32.0f) < 0.1f);

    // Q6_K: all scales=1, all ql/qh=0 → quant=-32, dot = 256*(-32)*1 = -8192
    BnBlockQ6K *q6k = (BnBlockQ6K *)calloc(1, sizeof(BnBlockQ6K));
    q6k->d = 0x3C00;
    for (int i = 0; i < 16; i++) q6k->scales[i] = 1;
    BnQWeight W_q6k = { q6k, BN_GGUF_TENSOR_Q6_K, 1, 256, 1.0f };

    bn_quant_matvec(&out, &W_q6k, x, x_q, NULL);
    assert(fabsf(out - (-8192.0f)) < 1.0f);

    free(tq2);
    free(q8);
    free(q6k);
    printf("PASSED\n");
}

// --- Integration test: batch matvec ---

static void test_matvec_batch(void) {
    printf("test_matvec_batch... ");

    BnBlockTQ2 *blocks1 = (BnBlockTQ2 *)calloc(2, sizeof(BnBlockTQ2));
    BnBlockTQ2 *blocks2 = (BnBlockTQ2 *)calloc(2, sizeof(BnBlockTQ2));

    // Matrix 1: all +1
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < 64; i++) blocks1[r].qs[i] = 0xAA;
        blocks1[r].d = 0x3C00;
    }

    // Matrix 2: all -1
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < 64; i++) blocks2[r].qs[i] = 0x00;
        blocks2[r].d = 0x3C00;
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_TQ2_0, 2, 256, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_TQ2_0, 2, 256, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    // Reference: individual calls
    float ref1[2], ref2[2];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    // Batch call
    float out1[2], out2[2];
    int8_t x_q[256];
    BnMatvecTask tasks[2] = {
        { out1, &W1 },
        { out2, &W2 },
    };
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < 2; i++) {
        assert(fabsf(out1[i] - ref1[i]) < 1e-3f);
        assert(fabsf(out2[i] - ref2[i]) < 1e-3f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

// --- Integration test: threaded matvec ---

static void test_matvec_threaded(void) {
    printf("test_matvec_threaded... ");

    int rows = 8, cols = 256;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data = (uint8_t *)calloc(data_size, 1);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < row_bytes; b++) {
            data[r * row_bytes + b] = 0x25;
        }
    }

    float tensor_scale = 0.5f;
    memcpy(data + (size_t)rows * row_bytes, &tensor_scale, sizeof(float));

    BnQWeight W = { data, BN_GGUF_TENSOR_I2_S, rows, cols, tensor_scale };

    float x[256];
    for (int i = 0; i < cols; i++) x[i] = 0.1f * (i % 13) - 0.6f;

    // Serial reference
    float ref[8];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref, &W, x, x_q_ref, NULL);

    // Threaded
    BnThreadPool *pool = bn_tp_create(3);
    float out[8];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, pool);

    for (int i = 0; i < rows; i++) {
        float err = fabsf(out[i] - ref[i]);
        float mag = fabsf(ref[i]) + 1e-6f;
        assert(err / mag < 0.02f);
    }

    bn_tp_free(pool);
    free(data);
    printf("PASSED\n");
}

// --- Integration test: matmul vs N individual matvecs ---
// Verifies that bn_quant_matmul produces identical results to calling
// bn_quant_matvec N times with different x vectors.
static void test_matmul_correctness(void) {
    printf("test_matmul_correctness... ");

    int rows = 4, cols = 256, n_tokens = 3;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data = (uint8_t *)calloc(data_size, 1);

    // Fill with deterministic ternary pattern
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < row_bytes; b++)
            data[r * row_bytes + b] = (uint8_t)((r * 17 + b * 31) & 0xFF);

    float tensor_scale = 0.25f;
    memcpy(data + (size_t)rows * row_bytes, &tensor_scale, sizeof(float));
    BnQWeight W = { data, BN_GGUF_TENSOR_I2_S, rows, cols, tensor_scale };

    // Create N different x vectors
    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < cols; i++)
            X[t * cols + i] = 0.1f * ((t * 7 + i * 3) % 19) - 0.9f;

    // Reference: N individual matvec calls
    float *ref = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    int8_t x_q[256];
    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(ref + t * rows, &W, X + t * cols, x_q, NULL);

    // Matmul: single call for all N tokens
    float *out = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    bn_quant_matmul(out, &W, X, n_tokens, x_q, NULL);

    // Compare
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float diff = fabsf(out[t * rows + r] - ref[t * rows + r]);
            float mag = fabsf(ref[t * rows + r]) + 1e-6f;
            assert(diff / mag < 0.01f);
        }
    }

    free(data); free(X); free(ref); free(out);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Quant Integration Tests ===\n");
    test_dispatch_routing();
    test_matvec_batch();
    test_matvec_threaded();
    test_matmul_correctness();
    printf("All quant integration tests passed!\n");
    return 0;
}
