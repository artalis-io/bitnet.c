#include "quant.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static uint16_t test_fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float ref_dot_f32(const float *w, const float *x, int cols) {
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) sum += w[i] * x[i];
    return sum;
}

static float ref_dot_f16(const uint16_t *w, const float *x, int cols) {
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) sum += bn_fp16_to_fp32(w[i]) * x[i];
    return sum;
}

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

    // Q5_K: all scales=1, mins=0, q=1 → dot = 256
    BnBlockQ5K *q5k = (BnBlockQ5K *)calloc(1, sizeof(BnBlockQ5K));
    q5k->d = 0x3C00;
    q5k->dmin = 0;
    for (int i = 0; i < 4; i++) q5k->scales[i] = 1;
    for (int i = 8; i < 12; i++) q5k->scales[i] = 1;
    for (int i = 0; i < 128; i++) q5k->qs[i] = 0x11;
    BnQWeight W_q5k = { q5k, BN_GGUF_TENSOR_Q5_K, 1, 256, 1.0f };

    bn_quant_matvec(&out, &W_q5k, x, x_q, NULL);
    assert(fabsf(out - 256.0f) < 1.0f);

    free(tq2);
    free(q8);
    free(q6k);
    free(q5k);
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
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
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

static void test_q4_matmul_correctness(void) {
    printf("test_q4_matmul_correctness... ");

    int rows = 7, cols = 96, n_tokens = 5;
    int n_bpr = cols / 32;
    BnBlockQ4_0 *blocks = (BnBlockQ4_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4_0));
    assert(blocks);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ4_0 *blk = &blocks[(size_t)r * n_bpr + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(1 + ((r + b) % 5)));
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (uint8_t)((r * 3 + b * 5 + i * 7) & 0x0F);
                uint8_t hi = (uint8_t)((r * 11 + b * 13 + i * 17 + 3) & 0x0F);
                blk->qs[i] = (uint8_t)(lo | (hi << 4));
            }
        }
    }

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };
    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    float *ref = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out_prepared = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    int8_t *x_q = (int8_t *)malloc((size_t)cols);
    int n_groups = rows / 4;
    BnPreparedWeight prepared = { 0 };
    prepared.qs = (uint8_t *)calloc((size_t)n_groups * n_bpr * 64, 1);
    prepared.scales = (uint16_t *)calloc((size_t)n_groups * n_bpr * 4, sizeof(uint16_t));
    assert(X && ref && out && out_prepared && x_q && prepared.qs && prepared.scales);

    for (int g = 0; g < n_groups; g++) {
        for (int b = 0; b < n_bpr; b++) {
            size_t gb = (size_t)g * n_bpr + b;
            for (int r = 0; r < 4; r++) {
                size_t src = (size_t)(g * 4 + r) * n_bpr + b;
                prepared.scales[gb * 4 + r] = blocks[src].d;
            }
            uint8_t *dst = prepared.qs + gb * 64;
            for (int ng = 0; ng < 4; ng++) {
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) * n_bpr + b;
                    const uint8_t *qs = blocks[src].qs + ng * 4;
                    uint8_t *dp = dst + ng * 16 + r * 4;
                    for (int j = 0; j < 4; j++)
                        dp[j] = qs[j] ^ 0x88;
                }
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < cols; i++) {
            int v = (t * 19 + i * 23) % 41;
            X[(size_t)t * cols + i] = 0.075f * (float)(v - 20);
        }
    }

    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(ref + (size_t)t * rows, &W, X + (size_t)t * cols, x_q, NULL);

    bn_quant_matmul(out, &W, X, n_tokens, x_q, NULL);
    bn_quant_matmul_prepared(out_prepared, &W, &prepared, X, n_tokens, x_q, NULL);

    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float diff = fabsf(out[(size_t)t * rows + r] - ref[(size_t)t * rows + r]);
            float mag = fabsf(ref[(size_t)t * rows + r]) + 1e-6f;
            assert(diff / mag < 0.01f || diff < 1e-4f);
            diff = fabsf(out_prepared[(size_t)t * rows + r] - ref[(size_t)t * rows + r]);
            assert(diff / mag < 0.01f || diff < 1e-4f);
        }
    }

    free(blocks); free(X); free(ref); free(out); free(out_prepared);
    free(x_q); free(prepared.qs); free(prepared.scales);
    printf("PASSED\n");
}

static void test_q5k_matmul_correctness(void) {
    printf("test_q5k_matmul_correctness... ");

    int rows = 2, cols = 256, n_tokens = 3;
    BnBlockQ5K *blocks = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    for (int r = 0; r < rows; r++) {
        blocks[r].d = 0x3C00;
        for (int i = 0; i < 4; i++) blocks[r].scales[i] = 1;
        for (int i = 8; i < 12; i++) blocks[r].scales[i] = 1;
        memset(blocks[r].qs, r == 0 ? 0x11 : 0x22, sizeof(blocks[r].qs));
    }
    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < cols; i++)
            X[t * cols + i] = 0.05f * ((t * 11 + i * 5) % 23) - 0.5f;

    float ref[6];
    int8_t x_q[256];
    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(ref + t * rows, &W, X + t * cols, x_q, NULL);

    float out[6];
    bn_quant_matmul(out, &W, X, n_tokens, x_q, NULL);

    for (int i = 0; i < rows * n_tokens; i++) {
        float diff = fabsf(out[i] - ref[i]);
        float mag = fabsf(ref[i]) + 1e-6f;
        assert(diff / mag < 0.02f);
    }

    free(blocks);
    free(X);
    printf("PASSED\n");
}

static void test_q5k_matvec_multi_correctness(void) {
    printf("test_q5k_matvec_multi_correctness... ");

    int rows = 3, cols = 256, n_tasks = 2;
    BnBlockQ5K *blocks1 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    BnBlockQ5K *blocks2 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    for (int r = 0; r < rows; r++) {
        blocks1[r].d = 0x3C00;
        blocks2[r].d = 0x3C00;
        for (int i = 0; i < 4; i++) {
            blocks1[r].scales[i] = 1;
            blocks2[r].scales[i] = 1;
        }
        for (int i = 8; i < 12; i++) {
            blocks1[r].scales[i] = 1;
            blocks2[r].scales[i] = 1;
        }
        memset(blocks1[r].qs, 0x11 + r, sizeof(blocks1[r].qs));
        memset(blocks2[r].qs, 0x22 + r, sizeof(blocks2[r].qs));
    }
    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float X1[256], X2[256];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.04f * ((i * 7) % 29) - 0.5f;
        X2[i] = 0.03f * ((i * 5 + 3) % 31) - 0.4f;
    }

    float ref1[3], ref2[3], out1[3], out2[3];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 256];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        float mag1 = fabsf(ref1[i]) + 1e-6f;
        float mag2 = fabsf(ref2[i]) + 1e-6f;
        assert(fabsf(out1[i] - ref1[i]) / mag1 < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / mag2 < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_q5k_matvec_batch_correctness(void) {
    printf("test_q5k_matvec_batch_correctness... ");

    int rows = 7, cols = 256;
    BnBlockQ5K *blocks1 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    BnBlockQ5K *blocks2 = (BnBlockQ5K *)calloc((size_t)rows, sizeof(BnBlockQ5K));
    for (int r = 0; r < rows; r++) {
        blocks1[r].d = 0x3C00;
        blocks2[r].d = 0x3C00;
        for (int i = 0; i < 4; i++) {
            blocks1[r].scales[i] = (uint8_t)(1 + ((r + i) % 3));
            blocks2[r].scales[i] = (uint8_t)(1 + ((2 * r + i) % 3));
        }
        for (int i = 8; i < 12; i++) {
            blocks1[r].scales[i] = (uint8_t)(1 + ((r + i) % 3));
            blocks2[r].scales[i] = (uint8_t)(1 + ((2 * r + i) % 3));
        }
        for (int i = 0; i < 128; i++) {
            blocks1[r].qs[i] = (uint8_t)(r * 19 + i * 7);
            blocks2[r].qs[i] = (uint8_t)(r * 23 + i * 5 + 1);
        }
        for (int i = 0; i < 32; i++) {
            blocks1[r].qh[i] = (uint8_t)(r * 11 + i * 3);
            blocks2[r].qh[i] = (uint8_t)(r * 13 + i * 5 + 2);
        }
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < cols; i++)
        x[i] = 0.04f * ((i * 7 + 5) % 31) - 0.6f;

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    int8_t x_q[256];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_i2s_matvec_multi_correctness(void) {
    printf("test_i2s_matvec_multi_correctness... ");

    int rows = 7, cols = 256, n_tasks = 3;
    int row_bytes = cols / 4;
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data1 = (uint8_t *)calloc(data_size, 1);
    uint8_t *data2 = (uint8_t *)calloc(data_size, 1);
    uint8_t *data3 = (uint8_t *)calloc(data_size, 1);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < row_bytes; b++) {
            data1[r * row_bytes + b] = (uint8_t)(r * 13 + b * 7);
            data2[r * row_bytes + b] = (uint8_t)(r * 11 + b * 5 + 3);
            data3[r * row_bytes + b] = (uint8_t)(r * 17 + b * 9 + 1);
        }
    }

    float scale = 0.25f;
    memcpy(data1 + (size_t)rows * row_bytes, &scale, sizeof(float));
    memcpy(data2 + (size_t)rows * row_bytes, &scale, sizeof(float));
    memcpy(data3 + (size_t)rows * row_bytes, &scale, sizeof(float));

    BnQWeight W1 = { data1, BN_GGUF_TENSOR_I2_S, rows, cols, scale };
    BnQWeight W2 = { data2, BN_GGUF_TENSOR_I2_S, rows, cols, scale };
    BnQWeight W3 = { data3, BN_GGUF_TENSOR_I2_S, rows, cols, scale };

    float X1[256], X2[256], X3[256];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.03f * ((i * 7) % 31) - 0.45f;
        X2[i] = 0.04f * ((i * 5 + 2) % 29) - 0.55f;
        X3[i] = 0.05f * ((i * 3 + 4) % 23) - 0.50f;
    }

    float ref1[7], ref2[7], ref3[7];
    float out1[7], out2[7], out3[7];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);
    bn_quant_matvec(ref3, &W3, X3, x_q_ref, NULL);

    BnMatvecMultiTask tasks[3] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
         { out3, &W3, X3, NULL },
    };
    int8_t x_q_bufs[3 * 256];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out3[i] - ref3[i]) / (fabsf(ref3[i]) + 1e-6f) < 0.02f);
    }

    free(data1);
    free(data2);
    free(data3);
    printf("PASSED\n");
}

static void test_q4_matvec_multi_correctness(void) {
    printf("test_q4_matvec_multi_correctness... ");

    int rows = 7, cols = 64, n_tasks = 2;
    int n_bpr = cols / 32;
    BnBlockQ4_0 *blocks1 = (BnBlockQ4_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4_0));
    BnBlockQ4_0 *blocks2 = (BnBlockQ4_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4_0));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ4_0 *a = &blocks1[r * n_bpr + b];
            BnBlockQ4_0 *c = &blocks2[r * n_bpr + b];
            a->d = 0x3C00;
            c->d = 0x3C00;
            for (int i = 0; i < 16; i++) {
                a->qs[i] = (uint8_t)(r * 19 + b * 11 + i * 7);
                c->qs[i] = (uint8_t)(r * 23 + b * 13 + i * 5 + 1);
            }
        }
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };

    float X1[64], X2[64];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.05f * ((i * 7) % 17) - 0.40f;
        X2[i] = 0.04f * ((i * 5 + 3) % 19) - 0.35f;
    }

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[64];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 64];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_q8_matvec_batch_correctness(void) {
    printf("test_q8_matvec_batch_correctness... ");

    int rows = 7, cols = 64;
    int n_bpr = cols / 32;
    BnBlockQ8_0 *blocks1 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));
    BnBlockQ8_0 *blocks2 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ8_0 *a = &blocks1[r * n_bpr + b];
            BnBlockQ8_0 *c = &blocks2[r * n_bpr + b];
            a->d = 0x3C00;
            c->d = 0x3C00;
            for (int i = 0; i < 32; i++) {
                a->qs[i] = (int8_t)(((r * 17 + b * 11 + i * 5) % 31) - 15);
                c->qs[i] = (int8_t)(((r * 13 + b * 7 + i * 3) % 29) - 14);
            }
        }
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float x[64];
    for (int i = 0; i < cols; i++)
        x[i] = 0.04f * ((i * 7 + 3) % 23) - 0.45f;

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[64];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    int8_t x_q[64];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_q8_matvec_multi_correctness(void) {
    printf("test_q8_matvec_multi_correctness... ");

    int rows = 7, cols = 64, n_tasks = 2;
    int n_bpr = cols / 32;
    BnBlockQ8_0 *blocks1 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));
    BnBlockQ8_0 *blocks2 = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ8_0));

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ8_0 *a = &blocks1[r * n_bpr + b];
            BnBlockQ8_0 *c = &blocks2[r * n_bpr + b];
            a->d = 0x3C00;
            c->d = 0x3C00;
            for (int i = 0; i < 32; i++) {
                a->qs[i] = (int8_t)(((r * 19 + b * 5 + i * 7) % 37) - 18);
                c->qs[i] = (int8_t)(((r * 11 + b * 13 + i * 3) % 35) - 17);
            }
        }
    }

    BnQWeight W1 = { blocks1, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };
    BnQWeight W2 = { blocks2, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float X1[64], X2[64];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.04f * ((i * 5 + 1) % 23) - 0.42f;
        X2[i] = 0.03f * ((i * 7 + 4) % 29) - 0.38f;
    }

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[64];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 64];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) / (fabsf(ref1[i]) + 1e-6f) < 0.02f);
        assert(fabsf(out2[i] - ref2[i]) / (fabsf(ref2[i]) + 1e-6f) < 0.02f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

static void test_bf16_matvec_batch_correctness(void) {
    printf("test_bf16_matvec_batch_correctness... ");

    int rows = 7, cols = 33;
    uint16_t *data1 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));
    uint16_t *data2 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v1 = 0.0625f * ((r * 17 + c * 5) % 23) - 0.625f;
            float v2 = 0.03125f * ((r * 11 + c * 7 + 3) % 29) - 0.4375f;
            data1[(size_t)r * cols + c] = test_fp32_to_bf16(v1);
            data2[(size_t)r * cols + c] = test_fp32_to_bf16(v2);
        }
    }

    BnQWeight W1 = { data1, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };
    BnQWeight W2 = { data2, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };

    float x[33];
    for (int i = 0; i < cols; i++)
        x[i] = 0.05f * ((i * 7 + 2) % 19) - 0.45f;

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[33];
    bn_quant_matvec(ref1, &W1, x, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out1, &W1, NULL, 0 },
         { out2, &W2, NULL, 0 },
    };
    int8_t x_q[33];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) < 1e-4f);
        assert(fabsf(out2[i] - ref2[i]) < 1e-4f);
    }

    free(data1);
    free(data2);
    printf("PASSED\n");
}

static void test_unquantized_matvec_correctness(void) {
    printf("test_unquantized_matvec_correctness... ");

    int rows = 5, cols = 37;
    float *f32_data = (float *)calloc((size_t)rows * cols, sizeof(float));
    uint16_t *f16_data = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));
    assert(f32_data && f16_data);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v = 0.02f * ((r * 13 + c * 7) % 31) - 0.27f;
            f32_data[(size_t)r * cols + c] = v;
            f16_data[(size_t)r * cols + c] = bn_fp32_to_fp16(v);
        }
    }

    float x[37];
    for (int i = 0; i < cols; i++)
        x[i] = 0.03f * ((i * 11 + 5) % 23) - 0.31f;

    BnQWeight W32 = { f32_data, BN_GGUF_TENSOR_F32, rows, cols, 1.0f };
    BnQWeight W16 = { f16_data, BN_GGUF_TENSOR_F16, rows, cols, 1.0f };
    float out32[5], out16[5];
    int8_t x_q[37];

    bn_quant_matvec(out32, &W32, x, x_q, NULL);
    bn_quant_matvec(out16, &W16, x, x_q, NULL);

    for (int r = 0; r < rows; r++) {
        assert(fabsf(out32[r] - ref_dot_f32(f32_data + (size_t)r * cols, x, cols)) < 1e-5f);
        assert(fabsf(out16[r] - ref_dot_f16(f16_data + (size_t)r * cols, x, cols)) < 1e-4f);
    }

    free(f32_data);
    free(f16_data);
    printf("PASSED\n");
}

static void test_bf16_matvec_multi_correctness(void) {
    printf("test_bf16_matvec_multi_correctness... ");

    int rows = 7, cols = 33, n_tasks = 2;
    uint16_t *data1 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));
    uint16_t *data2 = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v1 = 0.03125f * ((r * 13 + c * 3) % 31) - 0.5f;
            float v2 = 0.0625f * ((r * 19 + c * 5 + 1) % 17) - 0.375f;
            data1[(size_t)r * cols + c] = test_fp32_to_bf16(v1);
            data2[(size_t)r * cols + c] = test_fp32_to_bf16(v2);
        }
    }

    BnQWeight W1 = { data1, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };
    BnQWeight W2 = { data2, BN_GGUF_TENSOR_BF16, rows, cols, 1.0f };

    float X1[33], X2[33];
    for (int i = 0; i < cols; i++) {
        X1[i] = 0.04f * ((i * 5 + 1) % 23) - 0.42f;
        X2[i] = 0.03f * ((i * 7 + 4) % 29) - 0.38f;
    }

    float ref1[7], ref2[7], out1[7], out2[7];
    int8_t x_q_ref[33];
    bn_quant_matvec(ref1, &W1, X1, x_q_ref, NULL);
    bn_quant_matvec(ref2, &W2, X2, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out1, &W1, X1, NULL },
         { out2, &W2, X2, NULL },
    };
    int8_t x_q_bufs[2 * 33];
    bn_quant_matvec_multi(tasks, n_tasks, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out1[i] - ref1[i]) < 1e-4f);
        assert(fabsf(out2[i] - ref2[i]) < 1e-4f);
    }

    free(data1);
    free(data2);
    printf("PASSED\n");
}

static void test_mixed_kquant_matvec_batch_correctness(void) {
    printf("test_mixed_kquant_matvec_batch_correctness... ");

    int rows = 5, cols = 256;
    BnBlockQ4K *q4 = (BnBlockQ4K *)calloc((size_t)rows, sizeof(BnBlockQ4K));
    BnBlockQ6K *q6 = (BnBlockQ6K *)calloc((size_t)rows, sizeof(BnBlockQ6K));

    for (int r = 0; r < rows; r++) {
        q4[r].d = 0x3C00;
        q4[r].dmin = 0;
        for (int i = 0; i < 12; i++) q4[r].scales[i] = (uint8_t)((r * 7 + i * 3) & 0x3f);
        for (int i = 0; i < 128; i++) q4[r].qs[i] = (uint8_t)(r * 17 + i * 11);

        q6[r].d = 0x3C00;
        for (int i = 0; i < 16; i++) q6[r].scales[i] = (int8_t)((r * 5 + i * 3) % 17 - 8);
        for (int i = 0; i < 128; i++) q6[r].ql[i] = (uint8_t)(r * 13 + i * 5);
        for (int i = 0; i < 64; i++) q6[r].qh[i] = (uint8_t)(r * 19 + i * 7);
    }

    BnQWeight W4 = { .data = q4, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W6 = { .data = q6, .type = BN_GGUF_TENSOR_Q6_K, .rows = rows, .cols = cols, .scale = 1.0f };

    float x[256];
    for (int i = 0; i < cols; i++)
        x[i] = 0.03125f * ((i * 11 + 3) % 37) - 0.5f;

    float ref4[5], ref6[5], out4[5], out6[5];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref4, &W4, x, x_q_ref, NULL);
    bn_quant_matvec(ref6, &W6, x, x_q_ref, NULL);

    BnMatvecTask tasks[2] = {
         { out4, &W4, NULL, 0 },
         { out6, &W6, NULL, 0 },
    };
    int8_t x_q[256];
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out4[i] - ref4[i]) < 1e-4f);
        assert(fabsf(out6[i] - ref6[i]) < 1e-4f);
    }

    free(q4);
    free(q6);
    printf("PASSED\n");
}

static void test_mixed_kquant_matvec_multi_correctness(void) {
    printf("test_mixed_kquant_matvec_multi_correctness... ");

    int rows = 5, cols = 256;
    BnBlockQ4K *q4 = (BnBlockQ4K *)calloc((size_t)rows, sizeof(BnBlockQ4K));
    BnBlockQ6K *q6 = (BnBlockQ6K *)calloc((size_t)rows, sizeof(BnBlockQ6K));

    for (int r = 0; r < rows; r++) {
        q4[r].d = 0x3C00;
        q4[r].dmin = 0;
        for (int i = 0; i < 12; i++) q4[r].scales[i] = (uint8_t)((r * 9 + i * 5) & 0x3f);
        for (int i = 0; i < 128; i++) q4[r].qs[i] = (uint8_t)(r * 23 + i * 3);

        q6[r].d = 0x3C00;
        for (int i = 0; i < 16; i++) q6[r].scales[i] = (int8_t)((r * 7 + i * 5) % 19 - 9);
        for (int i = 0; i < 128; i++) q6[r].ql[i] = (uint8_t)(r * 11 + i * 13);
        for (int i = 0; i < 64; i++) q6[r].qh[i] = (uint8_t)(r * 17 + i * 7);
    }

    BnQWeight W4 = { .data = q4, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W6 = { .data = q6, .type = BN_GGUF_TENSOR_Q6_K, .rows = rows, .cols = cols, .scale = 1.0f };

    float x4[256], x6[256];
    for (int i = 0; i < cols; i++) {
        x4[i] = 0.025f * ((i * 7 + 5) % 41) - 0.45f;
        x6[i] = 0.02f * ((i * 13 + 2) % 43) - 0.42f;
    }

    float ref4[5], ref6[5], out4[5], out6[5];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref4, &W4, x4, x_q_ref, NULL);
    bn_quant_matvec(ref6, &W6, x6, x_q_ref, NULL);

    BnMatvecMultiTask tasks[2] = {
         { out4, &W4, x4, NULL },
         { out6, &W6, x6, NULL },
    };
    int8_t x_q_bufs[2 * 256];
    bn_quant_matvec_multi(tasks, 2, x_q_bufs, NULL);

    for (int i = 0; i < rows; i++) {
        assert(fabsf(out4[i] - ref4[i]) < 1e-4f);
        assert(fabsf(out6[i] - ref6[i]) < 1e-4f);
    }

    free(q4);
    free(q6);
    printf("PASSED\n");
}

static void fill_q4k_blocks(BnBlockQ4K *q4, int rows, int n_bpr, int seed) {
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ4K *blk = &q4[(size_t)r * n_bpr + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(1 + ((r + b + seed) % 5)));
            blk->dmin = bn_fp32_to_fp16(0.015625f * (float)(1 + ((2 * r + b + seed) % 7)));
            for (int i = 0; i < 12; i++)
                blk->scales[i] = (uint8_t)((r * 13 + b * 17 + i * 5 + seed) & 0x3f);
            for (int i = 0; i < 128; i++)
                blk->qs[i] = (uint8_t)(r * 19 + b * 23 + i * 7 + seed);
        }
    }
}

static void fill_q6k_blocks(BnBlockQ6K *q6, int rows, int n_bpr, int seed) {
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_bpr; b++) {
            BnBlockQ6K *blk = &q6[(size_t)r * n_bpr + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(1 + ((r + 2 * b + seed) % 5)));
            for (int i = 0; i < 16; i++)
                blk->scales[i] = (int8_t)((r * 7 + b * 11 + i * 3 + seed) % 31 - 15);
            for (int i = 0; i < 128; i++)
                blk->ql[i] = (uint8_t)(r * 11 + b * 13 + i * 5 + seed);
            for (int i = 0; i < 64; i++)
                blk->qh[i] = (uint8_t)(r * 17 + b * 19 + i * 9 + seed);
        }
    }
}

static void test_kquant_preq8k_matmul_correctness(void) {
    printf("test_kquant_preq8k_matmul_correctness... ");

    int rows = 7, cols = 512, n_tokens = 5;
    int n_bpr = cols / BN_QK_K;
    BnBlockQ4K *q4a = (BnBlockQ4K *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4K));
    BnBlockQ4K *q4b = (BnBlockQ4K *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ4K));
    BnBlockQ6K *q6 = (BnBlockQ6K *)calloc((size_t)rows * n_bpr, sizeof(BnBlockQ6K));
    assert(q4a && q4b && q6);

    fill_q4k_blocks(q4a, rows, n_bpr, 3);
    fill_q4k_blocks(q4b, rows, n_bpr, 29);
    fill_q6k_blocks(q6, rows, n_bpr, 7);

    BnQWeight W4a = { .data = q4a, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W4b = { .data = q4b, .type = BN_GGUF_TENSOR_Q4_K, .rows = rows, .cols = cols, .scale = 1.0f };
    BnQWeight W6 = { .data = q6, .type = BN_GGUF_TENSOR_Q6_K, .rows = rows, .cols = cols, .scale = 1.0f };

    float *X = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    int8_t *x_q = (int8_t *)malloc((size_t)n_tokens * cols);
    float *x_d = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
    int16_t *x_bsums = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
    float *ref4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *ref6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out4b = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    float *out6 = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
    assert(X && x_q && x_d && x_bsums && ref4 && ref6 && out4 && out4b && out6);

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < cols; i++)
            X[(size_t)t * cols + i] = 0.0275f * (float)((t * 17 + i * 11 + 5) % 43) - 0.55f;
        bn_quant_x_to_q8k(X + (size_t)t * cols,
                          x_q + (size_t)t * cols,
                          x_d + (size_t)t * n_bpr,
                          x_bsums + (size_t)t * n_bpr * 16, cols);
        bn_quant_matvec(ref4 + (size_t)t * rows, &W4a, X + (size_t)t * cols,
                        x_q + (size_t)t * cols, NULL);
        bn_quant_matvec(ref6 + (size_t)t * rows, &W6, X + (size_t)t * cols,
                        x_q + (size_t)t * cols, NULL);
    }

    bn_quant_matmul_preq8k(out4, &W4a, n_tokens, x_q, x_d, x_bsums, X, NULL);
    bn_quant_matmul_preq8k(out6, &W6, n_tokens, x_q, x_d, x_bsums, X, NULL);

    float *multi_out[3] = { out4, out4b, out6 };
    const BnQWeight *multi_w[3] = { &W4a, &W4b, &W6 };
    bn_quant_matmul_preq8k_multi(multi_out, multi_w, 3, n_tokens,
                                 x_q, x_d, x_bsums, X, NULL);

    for (int i = 0; i < rows * n_tokens; i++) {
        assert(fabsf(out4[i] - ref4[i]) < 1e-3f);
        assert(fabsf(out6[i] - ref6[i]) < 1e-3f);
    }

    free(q4a); free(q4b); free(q6);
    free(X); free(x_q); free(x_d); free(x_bsums);
    free(ref4); free(ref6); free(out4); free(out4b); free(out6);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Quant Integration Tests ===\n");
    test_dispatch_routing();
    test_matvec_batch();
    test_matvec_threaded();
    test_matmul_correctness();
    test_q4_matmul_correctness();
    test_q5k_matmul_correctness();
    test_q5k_matvec_multi_correctness();
    test_q5k_matvec_batch_correctness();
    test_i2s_matvec_multi_correctness();
    test_q4_matvec_multi_correctness();
    test_q8_matvec_batch_correctness();
    test_q8_matvec_multi_correctness();
    test_unquantized_matvec_correctness();
    test_bf16_matvec_batch_correctness();
    test_bf16_matvec_multi_correctness();
    test_mixed_kquant_matvec_batch_correctness();
    test_mixed_kquant_matvec_multi_correctness();
    test_kquant_preq8k_matmul_correctness();
    printf("All quant integration tests passed!\n");
    return 0;
}
