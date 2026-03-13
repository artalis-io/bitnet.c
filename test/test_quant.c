#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// --- Test FP16 conversion ---

static void test_fp16_conversion(void) {
    printf("test_fp16_conversion... ");

    // Test zero
    assert(fp16_to_fp32(0x0000) == 0.0f);

    // Test 1.0 (FP16: 0x3C00)
    float one = fp16_to_fp32(0x3C00);
    assert(fabsf(one - 1.0f) < 1e-6f);

    // Test -1.0 (FP16: 0xBC00)
    float neg_one = fp16_to_fp32(0xBC00);
    assert(fabsf(neg_one - (-1.0f)) < 1e-6f);

    // Test 0.5 (FP16: 0x3800)
    float half = fp16_to_fp32(0x3800);
    assert(fabsf(half - 0.5f) < 1e-6f);

    // Round-trip test
    float test_vals[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 0.001f};
    for (int i = 0; i < 7; i++) {
        uint16_t h = fp32_to_fp16(test_vals[i]);
        float back = fp16_to_fp32(h);
        float err = fabsf(back - test_vals[i]);
        // FP16 has limited precision, allow some error for small values
        assert(err < 0.01f || (test_vals[i] != 0 && err / fabsf(test_vals[i]) < 0.01f));
    }

    printf("PASSED\n");
}

// --- Test TQ2_0 dequantization ---

static void test_tq2_dequant(void) {
    printf("test_tq2_dequant... ");

    BlockTQ2 block;
    memset(&block, 0, sizeof(block));

    // Set scale to 1.0 in FP16
    block.d = 0x3C00;

    // Pack known pattern: first byte = 0b10_01_00_10 = value pattern [-1, 0, -1, +1]
    // 2-bit fields: bits[1:0]=2(+1), bits[3:2]=0(-1), bits[5:4]=1(0), bits[7:6]=2(+1)
    // So byte value: 2 | (0 << 2) | (1 << 4) | (2 << 6) = 2 + 0 + 16 + 128 = 146 = 0x92
    block.qs[0] = 0x92;

    // Fill rest with all-1 pattern (value 0): 0b01_01_01_01 = 0x55
    for (int i = 1; i < 64; i++) block.qs[i] = 0x55;

    float out[256];
    dequant_tq2_block(&block, out);

    // Check first 4 values from byte 0 (output order: l=0 first 32 bytes, then l=1, etc.)
    // l=0, m=0: (qs[0] >> 0) & 3 = 2 -> value = (2-1)*1.0 = +1.0
    assert(fabsf(out[0] - 1.0f) < 1e-6f);
    // l=0, m=1: (qs[1] >> 0) & 3 = 1 -> value = (1-1)*1.0 = 0.0
    assert(fabsf(out[1] - 0.0f) < 1e-6f);

    // l=1, m=0 (index 32): (qs[0] >> 2) & 3 = 0 -> value = (0-1)*1.0 = -1.0
    assert(fabsf(out[32] - (-1.0f)) < 1e-6f);

    // l=2, m=0 (index 64): (qs[0] >> 4) & 3 = 1 -> value = (1-1)*1.0 = 0.0
    assert(fabsf(out[64] - 0.0f) < 1e-6f);

    // l=3, m=0 (index 96): (qs[0] >> 6) & 3 = 2 -> value = (2-1)*1.0 = +1.0
    assert(fabsf(out[96] - 1.0f) < 1e-6f);

    printf("PASSED\n");
}

// --- Test TQ1_0 dequantization ---

static void test_tq1_dequant(void) {
    printf("test_tq1_dequant... ");

    BlockTQ1 block;
    memset(&block, 0, sizeof(block));

    // Set scale to 1.0 in FP16
    block.d = 0x3C00;

    // For TQ1_0, encoding 5 values {-1,0,1} -> {0,1,2} as base-3:
    // All zeros -> all values = -1 (mapped: 0,0,0,0,0 = base3 number 0)
    // Encoded: q = 0, scaled = (0 * 256 + 242) / 243 = 0
    block.qs[0] = 0;

    // All ones (value=0): {1,1,1,1,1} = 1*81+1*27+1*9+1*3+1 = 121
    // Encoded: (121 * 256 + 242) / 243 = 127
    block.qs[1] = 127;

    float out[256];
    dequant_tq1_block(&block, out);

    // With qs[0]=0: all 5 values should be -1.0
    // n=0,m=0: q = 0 * pow3[0] = 0*1 = 0, xi = (0*3)>>8 = 0, value = (0-1)*1.0 = -1.0
    assert(fabsf(out[0] - (-1.0f)) < 1e-6f);

    printf("PASSED\n");
}

// --- Test ternary matvec ---

static void test_ternary_matvec(void) {
    printf("test_ternary_matvec... ");

    // Create a small TQ2_0 matrix: 2 rows x 256 cols
    int n_blocks = 2;  // 2 rows, each 1 block of 256
    BlockTQ2 *blocks = (BlockTQ2 *)calloc(n_blocks, sizeof(BlockTQ2));

    // Row 0: all +1 (qs value 2 in every 2-bit field)
    // 2 in each field: 0b10_10_10_10 = 0xAA
    for (int i = 0; i < 64; i++) blocks[0].qs[i] = 0xAA;
    blocks[0].d = 0x3C00;  // scale = 1.0

    // Row 1: all 0 (qs value 1 in every 2-bit field)
    // 1 in each field: 0b01_01_01_01 = 0x55
    for (int i = 0; i < 64; i++) blocks[1].qs[i] = 0x55;
    blocks[1].d = 0x3C00;

    QWeight W = { blocks, 35, 2, 256, 1.0f };

    // Input: all 1.0
    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[256];
    ternary_matvec(out, &W, x, x_q, NULL);

    // Row 0: all +1 -> sum of 256 ones * 1.0 = 256.0 * scale(1.0)
    assert(fabsf(out[0] - 256.0f) < 1e-3f);

    // Row 1: all 0 -> 0.0
    assert(fabsf(out[1] - 0.0f) < 1e-3f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test I2_S matvec: compare single vs batch (catches SDOT quantization error) ---

static void test_i2s_matvec(void) {
    printf("test_i2s_matvec... ");

    // Create I2_S weight matrix: 4 rows x 256 cols
    // Each row needs 256/4 = 64 packed bytes + 4 bytes for float scale at end
    int rows = 4, cols = 256;
    int row_bytes = cols / 4;  // 64
    size_t data_size = (size_t)rows * row_bytes + sizeof(float);
    uint8_t *data = (uint8_t *)calloc(data_size, 1);

    // Fill with a known pattern: alternating -1, +1, 0, 0
    // Encoding: 0=-1, 1=0, 2=+1. Pack 4 per byte: bits 7-6, 5-4, 3-2, 1-0
    // subrow0=-1(0), subrow1=+1(2), subrow2=0(1), subrow3=0(1)
    // byte = (0<<6) | (2<<4) | (1<<2) | (1<<0) = 0x25
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < row_bytes; b++) {
            data[r * row_bytes + b] = 0x25;
        }
    }

    // Set per-tensor scale at end of packed data
    float tensor_scale = 0.5f;
    memcpy(data + (size_t)rows * row_bytes, &tensor_scale, sizeof(float));

    QWeight W = { data, 36, rows, cols, tensor_scale };

    // Input: ramp 0.1, 0.2, ..., with some variation
    float x[256];
    for (int i = 0; i < cols; i++) x[i] = 0.1f * (i % 17) - 0.8f;

    // Reference: single-call matvec
    float ref[4];
    int8_t x_q_ref[256];
    ternary_matvec(ref, &W, x, x_q_ref, NULL);

    // Batch call
    float out1[4], out2[4];
    int8_t x_q[256];
    memset(out1, 0, sizeof(out1));
    memset(out2, 0, sizeof(out2));

    // Split into 2 batch tasks (2 rows each) via separate QWeights
    size_t half_data = (size_t)2 * row_bytes;
    QWeight W1 = { data, 36, 2, cols, tensor_scale };
    QWeight W2 = { data + half_data, 36, 2, cols, tensor_scale };

    MatvecTask tasks[2] = {
        { out1, &W1 },
        { out2, &W2 },
    };
    ternary_matvec_batch(tasks, 2, x, x_q, NULL);

    // Compare with 2% relative tolerance (int8 quantization error)
    for (int i = 0; i < 2; i++) {
        float err = fabsf(out1[i] - ref[i]);
        float mag = fabsf(ref[i]) + 1e-6f;
        assert(err / mag < 0.02f);
    }
    for (int i = 0; i < 2; i++) {
        float err = fabsf(out2[i] - ref[2 + i]);
        float mag = fabsf(ref[2 + i]) + 1e-6f;
        assert(err / mag < 0.02f);
    }

    free(data);
    printf("PASSED\n");
}

// --- Test batch TQ2_0 matvec: verify batch produces same results as individual calls ---

static void test_matvec_batch(void) {
    printf("test_matvec_batch... ");

    // Create two TQ2_0 weight matrices: 2 rows x 256 cols each
    int n_blocks = 2;
    BlockTQ2 *blocks1 = (BlockTQ2 *)calloc(n_blocks, sizeof(BlockTQ2));
    BlockTQ2 *blocks2 = (BlockTQ2 *)calloc(n_blocks, sizeof(BlockTQ2));

    // Matrix 1: all +1
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < 64; i++) blocks1[r].qs[i] = 0xAA;
        blocks1[r].d = 0x3C00;
    }

    // Matrix 2: all -1 (qs value 0 in every field)
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < 64; i++) blocks2[r].qs[i] = 0x00;
        blocks2[r].d = 0x3C00;
    }

    QWeight W1 = { blocks1, 35, 2, 256, 1.0f };
    QWeight W2 = { blocks2, 35, 2, 256, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    // Reference: individual calls
    float ref1[2], ref2[2];
    int8_t x_q_ref[256];
    ternary_matvec(ref1, &W1, x, x_q_ref, NULL);
    ternary_matvec(ref2, &W2, x, x_q_ref, NULL);

    // Batch call
    float out1[2], out2[2];
    int8_t x_q[256];
    MatvecTask tasks[2] = {
        { out1, &W1 },
        { out2, &W2 },
    };
    ternary_matvec_batch(tasks, 2, x, x_q, NULL);

    for (int i = 0; i < 2; i++) {
        assert(fabsf(out1[i] - ref1[i]) < 1e-3f);
        assert(fabsf(out2[i] - ref2[i]) < 1e-3f);
    }

    free(blocks1);
    free(blocks2);
    printf("PASSED\n");
}

// --- Test threaded matvec: compare threaded vs serial output ---

static void test_matvec_threaded(void) {
    printf("test_matvec_threaded... ");

    // Create I2_S weight matrix: 8 rows x 256 cols
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

    QWeight W = { data, 36, rows, cols, tensor_scale };

    float x[256];
    for (int i = 0; i < cols; i++) x[i] = 0.1f * (i % 13) - 0.6f;

    // Serial reference
    float ref[8];
    int8_t x_q_ref[256];
    ternary_matvec(ref, &W, x, x_q_ref, NULL);

    // Threaded
    ThreadPool *pool = tp_create(3);
    float out[8];
    int8_t x_q[256];
    ternary_matvec(out, &W, x, x_q, pool);

    for (int i = 0; i < rows; i++) {
        float err = fabsf(out[i] - ref[i]);
        float mag = fabsf(ref[i]) + 1e-6f;
        assert(err / mag < 0.02f);
    }

    tp_free(pool);
    free(data);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Quant Tests ===\n");
    test_fp16_conversion();
    test_tq2_dequant();
    test_tq1_dequant();
    test_ternary_matvec();
    test_i2s_matvec();
    test_matvec_batch();
    test_matvec_threaded();
    printf("All quant tests passed!\n");
    return 0;
}
