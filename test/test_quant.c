#include "quant.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// --- Test FP16 conversion ---

static void test_fp16_conversion(void) {
    printf("test_fp16_conversion... ");

    // Test zero
    assert(bn_fp16_to_fp32(0x0000) == 0.0f);

    // Test 1.0 (FP16: 0x3C00)
    float one = bn_fp16_to_fp32(0x3C00);
    assert(fabsf(one - 1.0f) < 1e-6f);

    // Test -1.0 (FP16: 0xBC00)
    float neg_one = bn_fp16_to_fp32(0xBC00);
    assert(fabsf(neg_one - (-1.0f)) < 1e-6f);

    // Test 0.5 (FP16: 0x3800)
    float half = bn_fp16_to_fp32(0x3800);
    assert(fabsf(half - 0.5f) < 1e-6f);

    // Round-trip test
    float test_vals[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 0.001f};
    for (int i = 0; i < 7; i++) {
        uint16_t h = bn_fp32_to_fp16(test_vals[i]);
        float back = bn_fp16_to_fp32(h);
        float err = fabsf(back - test_vals[i]);
        // FP16 has limited precision, allow some error for small values
        assert(err < 0.01f || (test_vals[i] != 0 && err / fabsf(test_vals[i]) < 0.01f));
    }

    printf("PASSED\n");
}

// --- Test TQ2_0 dequantization ---

static void test_tq2_dequant(void) {
    printf("test_tq2_dequant... ");

    BnBlockTQ2 block;
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
    bn_quant_dequant_tq2(&block, out);

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

    BnBlockTQ1 block;
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
    bn_quant_dequant_tq1(&block, out);

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
    BnBlockTQ2 *blocks = (BnBlockTQ2 *)calloc(n_blocks, sizeof(BnBlockTQ2));

    // Row 0: all +1 (qs value 2 in every 2-bit field)
    // 2 in each field: 0b10_10_10_10 = 0xAA
    for (int i = 0; i < 64; i++) blocks[0].qs[i] = 0xAA;
    blocks[0].d = 0x3C00;  // scale = 1.0

    // Row 1: all 0 (qs value 1 in every 2-bit field)
    // 1 in each field: 0b01_01_01_01 = 0x55
    for (int i = 0; i < 64; i++) blocks[1].qs[i] = 0x55;
    blocks[1].d = 0x3C00;

    BnQWeight W = { blocks, 35, 2, 256, 1.0f };

    // Input: all 1.0
    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, NULL);

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

    BnQWeight W = { data, 36, rows, cols, tensor_scale };

    // Input: ramp 0.1, 0.2, ..., with some variation
    float x[256];
    for (int i = 0; i < cols; i++) x[i] = 0.1f * (i % 17) - 0.8f;

    // Reference: single-call matvec
    float ref[4];
    int8_t x_q_ref[256];
    bn_quant_matvec(ref, &W, x, x_q_ref, NULL);

    // Batch call
    float out1[4], out2[4];
    int8_t x_q[256];
    memset(out1, 0, sizeof(out1));
    memset(out2, 0, sizeof(out2));

    // Split into 2 batch tasks (2 rows each) via separate BnQWeights
    size_t half_data = (size_t)2 * row_bytes;
    BnQWeight W1 = { data, 36, 2, cols, tensor_scale };
    BnQWeight W2 = { data + half_data, 36, 2, cols, tensor_scale };

    BnMatvecTask tasks[2] = {
        { out1, &W1 },
        { out2, &W2 },
    };
    bn_quant_matvec_batch(tasks, 2, x, x_q, NULL);

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

// --- Test Q8_0 matvec ---

static void test_q8_matvec(void) {
    printf("test_q8_matvec... ");

    // 2 rows × 32 cols (1 block per row)
    int rows = 2, cols = 32;
    BnBlockQ8_0 *blocks = (BnBlockQ8_0 *)calloc(rows, sizeof(BnBlockQ8_0));

    // Row 0: scale=1.0, all qs=1 → dot with all-ones x = 32
    blocks[0].d = 0x3C00;  // FP16 1.0
    for (int i = 0; i < 32; i++) blocks[0].qs[i] = 1;

    // Row 1: scale=0.5 (FP16 0x3800), alternating +2/-2
    blocks[1].d = 0x3800;  // FP16 0.5
    for (int i = 0; i < 32; i++) blocks[1].qs[i] = (i % 2 == 0) ? 2 : -2;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float x[32];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[32];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Row 0: 32 * 1 * 1.0 = 32.0
    assert(fabsf(out[0] - 32.0f) < 0.1f);

    // Row 1: (16*2 - 16*2) * 0.5 = 0.0
    assert(fabsf(out[1] - 0.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test Q4_0 matvec ---

static void test_q4_matvec(void) {
    printf("test_q4_matvec... ");

    // 2 rows × 32 cols (1 block per row)
    int rows = 2, cols = 32;
    BnBlockQ4_0 *blocks = (BnBlockQ4_0 *)calloc(rows, sizeof(BnBlockQ4_0));

    // Row 0: scale=1.0, all nibbles = 10 → dequant = 10-8 = +2
    // qs byte: lo=0xA, hi=0xA → 0xAA
    blocks[0].d = 0x3C00;  // FP16 1.0
    for (int i = 0; i < 16; i++) blocks[0].qs[i] = 0xAA;

    // Row 1: scale=0.5, lo=12(=+4), hi=4(=-4)
    // qs byte: lo=0xC, hi=0x4 → 0x4C
    blocks[1].d = 0x3800;  // FP16 0.5
    for (int i = 0; i < 16; i++) blocks[1].qs[i] = 0x4C;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };

    float x[32];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[32];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Row 0: 32 * 2.0 * 1.0 = 64.0
    assert(fabsf(out[0] - 64.0f) < 0.1f);

    // Row 1: (16*4 + 16*(-4)) * 0.5 = 0.0
    assert(fabsf(out[1] - 0.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test Q8_0 matvec with multiple blocks and varying input ---

static void test_q8_matvec_multiblock(void) {
    printf("test_q8_matvec_multiblock... ");

    // 2 rows × 64 cols (2 blocks per row)
    int rows = 2, cols = 64;
    int n_blocks = rows * 2;
    BnBlockQ8_0 *blocks = (BnBlockQ8_0 *)calloc(n_blocks, sizeof(BnBlockQ8_0));

    // Row 0, block 0: scale=1.0, qs=[3, 3, ..., 3]
    blocks[0].d = 0x3C00;
    for (int i = 0; i < 32; i++) blocks[0].qs[i] = 3;

    // Row 0, block 1: scale=2.0 (FP16 0x4000), qs=[1, 1, ..., 1]
    blocks[1].d = 0x4000;
    for (int i = 0; i < 32; i++) blocks[1].qs[i] = 1;

    // Row 1, block 0: scale=1.0, qs=[-1, -1, ..., -1]
    blocks[2].d = 0x3C00;
    for (int i = 0; i < 32; i++) blocks[2].qs[i] = -1;

    // Row 1, block 1: scale=1.0, qs=[2, 2, ..., 2]
    blocks[3].d = 0x3C00;
    for (int i = 0; i < 32; i++) blocks[3].qs[i] = 2;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };

    float x[64];
    for (int i = 0; i < 64; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[64];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Row 0: 32*3*1.0 + 32*1*2.0 = 96 + 64 = 160.0
    assert(fabsf(out[0] - 160.0f) < 0.1f);

    // Row 1: 32*(-1)*1.0 + 32*2*1.0 = -32 + 64 = 32.0
    assert(fabsf(out[1] - 32.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test batch TQ2_0 matvec: verify batch produces same results as individual calls ---

static void test_matvec_batch(void) {
    printf("test_matvec_batch... ");

    // Create two TQ2_0 weight matrices: 2 rows x 256 cols each
    int n_blocks = 2;
    BnBlockTQ2 *blocks1 = (BnBlockTQ2 *)calloc(n_blocks, sizeof(BnBlockTQ2));
    BnBlockTQ2 *blocks2 = (BnBlockTQ2 *)calloc(n_blocks, sizeof(BnBlockTQ2));

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

    BnQWeight W1 = { blocks1, 35, 2, 256, 1.0f };
    BnQWeight W2 = { blocks2, 35, 2, 256, 1.0f };

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

    BnQWeight W = { data, 36, rows, cols, tensor_scale };

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

// --- Test Q6_K dequantization ---

static void test_q6k_dequant(void) {
    printf("test_q6k_dequant... ");

    BnBlockQ6K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;  // FP16 1.0

    // Set all scales to 1
    for (int i = 0; i < 16; i++) block.scales[i] = 1;

    // Set all ql bytes to 0 and all qh bytes to 0
    // This gives 6-bit quant = 0 for all elements, dequant = (0 - 32) * 1.0 * 1 = -32
    float out[256];
    bn_quant_dequant_q6k(&block, out);
    for (int i = 0; i < 256; i++) {
        assert(fabsf(out[i] - (-32.0f)) < 0.01f);
    }

    // Test with known values: ql[0] = 0x35 (lo=5, hi=3), qh[0] = 0xC9 (bits: 01 10 00 11)
    // scale[0]=2, d=1.0
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;  // FP16 1.0
    for (int i = 0; i < 16; i++) block.scales[i] = 2;
    block.ql[0] = 0x35;  // lo nibble=5, hi nibble=3
    block.ql[32] = 0x72; // lo nibble=2, hi nibble=7
    block.qh[0] = 0xC9;  // bits 0-1=01, bits 2-3=10, bits 4-5=00, bits 6-7=11

    bn_quant_dequant_q6k(&block, out);

    // Element 0: q = (5 | (01 << 4)) - 32 = (5 | 16) - 32 = 21 - 32 = -11, val = 1.0 * 2 * -11 = -22
    assert(fabsf(out[0] - (-22.0f)) < 0.01f);
    // Element 32: q = (2 | (10 << 4)) - 32 = (2 | 32) - 32 = 34 - 32 = 2, val = 1.0 * 2 * 2 = 4
    assert(fabsf(out[32] - 4.0f) < 0.01f);
    // Element 64: q = (3 | (00 << 4)) - 32 = 3 - 32 = -29, val = 1.0 * 2 * -29 = -58
    assert(fabsf(out[64] - (-58.0f)) < 0.01f);
    // Element 96: q = (7 | (11 << 4)) - 32 = (7 | 48) - 32 = 55 - 32 = 23, val = 1.0 * 2 * 23 = 46
    assert(fabsf(out[96] - 46.0f) < 0.01f);

    printf("PASSED\n");
}

// --- Test Q6_K matvec ---

static void test_q6k_matvec(void) {
    printf("test_q6k_matvec... ");

    // 2 rows × 256 cols (1 block per row)
    int rows = 2, cols = 256;
    BnBlockQ6K *blocks = (BnBlockQ6K *)calloc(rows, sizeof(BnBlockQ6K));

    // Row 0: d=1.0, all scales=1, all ql=0, all qh=0 → all quants = -32
    blocks[0].d = 0x3C00;
    for (int i = 0; i < 16; i++) blocks[0].scales[i] = 1;

    // Row 1: d=0.5, all scales=2, all ql=0x88 (lo=8,hi=8), qh=0 → quant=(8-32)=-24 and (8-32)=-24
    blocks[1].d = 0x3800;  // FP16 0.5
    for (int i = 0; i < 16; i++) blocks[1].scales[i] = 2;
    for (int i = 0; i < 128; i++) blocks[1].ql[i] = 0x88;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Row 0: 256 * (-32) * 1.0 * 1 = -8192
    assert(fabsf(out[0] - (-8192.0f)) < 1.0f);

    // Row 1: 256 * (-24) * 0.5 * 2 = -6144
    assert(fabsf(out[1] - (-6144.0f)) < 1.0f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test Q8_K dequantization ---

static void test_q8k_dequant(void) {
    printf("test_q8k_dequant... ");

    BnBlockQ8K block;
    memset(&block, 0, sizeof(block));
    block.d = 0.5f;  // float32 scale (NOT FP16!)

    // Set some known qs values
    block.qs[0] = 10;
    block.qs[1] = -20;
    block.qs[127] = 127;
    block.qs[255] = -128;

    float out[256];
    bn_quant_dequant_q8k(&block, out);

    // val = d * qs[i]
    assert(fabsf(out[0] - (0.5f * 10)) < 1e-4f);
    assert(fabsf(out[1] - (0.5f * -20)) < 1e-4f);
    assert(fabsf(out[127] - (0.5f * 127)) < 1e-4f);
    assert(fabsf(out[255] - (0.5f * -128)) < 1e-4f);
    // Zero elements should be zero
    assert(fabsf(out[100]) < 1e-6f);

    printf("PASSED\n");
}

// --- Test Q8_K matvec ---

static void test_q8k_matvec(void) {
    printf("test_q8k_matvec... ");

    // 2 rows × 256 cols (1 block per row)
    int rows = 2, cols = 256;
    BnBlockQ8K *blocks = (BnBlockQ8K *)calloc(rows, sizeof(BnBlockQ8K));

    // Row 0: d=1.0, all qs=1 → dot with all-ones x = 256
    blocks[0].d = 1.0f;
    for (int i = 0; i < 256; i++) blocks[0].qs[i] = 1;

    // Row 1: d=0.5, alternating +2/-2 → sum = 0
    blocks[1].d = 0.5f;
    for (int i = 0; i < 256; i++) blocks[1].qs[i] = (i % 2 == 0) ? 2 : -2;

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Row 0: 256 * 1 * 1.0 = 256.0
    assert(fabsf(out[0] - 256.0f) < 0.1f);
    // Row 1: (128*2 - 128*2) * 0.5 = 0.0
    assert(fabsf(out[1] - 0.0f) < 0.1f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test Q4_K dequantization ---

static void test_q4k_dequant(void) {
    printf("test_q4k_dequant... ");

    BnBlockQ4K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;     // FP16 1.0
    block.dmin = 0x3800;  // FP16 0.5

    // Set scales for sub-block 0: sc=2, m=1 (j<4 path)
    // scales[0] = (sc & 63) = 2, scales[4] = (m & 63) = 1
    block.scales[0] = 2;
    block.scales[4] = 1;

    // Set qs[0] = 0x53: lo nibble = 3, hi nibble = 5
    block.qs[0] = 0x53;

    float out[256];
    bn_quant_dequant_q4k(&block, out);

    // Element 0: lo nibble = 3, sc=2, m=1
    // val = d * sc * nibble - dmin * m = 1.0 * 2 * 3 - 0.5 * 1 = 5.5
    assert(fabsf(out[0] - 5.5f) < 0.01f);

    // Element 32: hi nibble = 5, sub-block 1 (scales[1]/scales[5] both 0)
    // val = d * 0 * 5 - dmin * 0 = 0
    assert(fabsf(out[32]) < 0.01f);

    printf("PASSED\n");
}

// --- Test Q4_K matvec ---

static void test_q4k_matvec(void) {
    printf("test_q4k_matvec... ");

    // 2 rows × 256 cols (1 block per row)
    int rows = 2, cols = 256;
    BnBlockQ4K *blocks = (BnBlockQ4K *)calloc(rows, sizeof(BnBlockQ4K));

    // Row 0: d=1.0, dmin=0, all scales sc=1 m=0, all qs nibbles = 2
    // Each byte: lo=2, hi=2 → 0x22
    blocks[0].d = 0x3C00;
    blocks[0].dmin = 0x0000;
    for (int j = 0; j < 8; j++) {
        // Set sc=1 for all 8 sub-blocks
        if (j < 4) {
            blocks[0].scales[j] = 1;      // sc
            blocks[0].scales[j + 4] = 0;  // m
        } else {
            // j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
            // Set scales[j+4] lo nibble = 1 for sc, hi nibble = 0 for m
            blocks[0].scales[j + 4] = 1;
        }
    }
    for (int i = 0; i < 128; i++) blocks[0].qs[i] = 0x22;

    // Row 1: d=0.5, dmin=0.25, sc=1, m=1, all nibbles = 0
    blocks[1].d = 0x3800;     // FP16 0.5
    blocks[1].dmin = 0x3400;  // FP16 0.25
    for (int j = 0; j < 4; j++) {
        blocks[1].scales[j] = 1;      // sc
        blocks[1].scales[j + 4] = 1;  // m
    }
    // qs all zeros (nibbles = 0)

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Row 0: each element = 1.0 * 1 * 2 - 0 = 2.0, sum = 256 * 2.0 = 512
    // But only sub-blocks 0-3 have sc=1, sub-blocks 4-7 have sc from j>=4 path
    // For j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4) = (1 & 0xF) | 0 = 1
    // So all sub-blocks have sc=1, each elem = 2, sum = 512
    assert(fabsf(out[0] - 512.0f) < 1.0f);

    // Row 1: sub-blocks 0-3 have sc=1, m=1; sub-blocks 4-7 have sc=0, m=0
    // For sub-blocks 0-3 (128 elements): val = 0.5*1*0 - 0.25*1 = -0.25
    // For sub-blocks 4-7 (128 elements): val = 0.5*0*0 - 0.25*0 = 0
    // Sum = 128 * (-0.25) + 128 * 0 = -32
    assert(fabsf(out[1] - (-32.0f)) < 1.0f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test Q5_K dequantization ---

static void test_q5k_dequant(void) {
    printf("test_q5k_dequant... ");

    BnBlockQ5K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;     // FP16 1.0
    block.dmin = 0x3800;  // FP16 0.5

    // Sub-block 0: sc=1, m=0
    block.scales[0] = 1;  // sc for sub-block 0
    block.scales[4] = 0;  // m for sub-block 0

    // Element 0: qs lo nibble = 7, qh bit 0 = 1 → q5 = 7 | 16 = 23
    block.qs[0] = 0x07;  // lo nibble = 7
    block.qh[0] = 0x01;  // bit 0 set → element 0 has high bit

    float out[256];
    bn_quant_dequant_q5k(&block, out);

    // Element 0: q5=23, val = d * sc * q5 - dmin * m = 1.0 * 1 * 23 - 0.5 * 0 = 23.0
    assert(fabsf(out[0] - 23.0f) < 0.01f);

    // Element 1: qs[1] = 0 (lo=0), qh bit 1 (qh[0] & 2 = 0) → q5 = 0
    // val = 1.0 * 1 * 0 - 0 = 0
    assert(fabsf(out[1]) < 0.01f);

    printf("PASSED\n");
}

// --- Test Q5_K matvec ---

static void test_q5k_matvec(void) {
    printf("test_q5k_matvec... ");

    // 2 rows × 256 cols
    int rows = 2, cols = 256;
    BnBlockQ5K *blocks = (BnBlockQ5K *)calloc(rows, sizeof(BnBlockQ5K));

    // Row 0: d=1.0, dmin=0, sc=1, m=0 for sub-blocks 0-3, qs=0, qh=0 → all q5=0
    blocks[0].d = 0x3C00;
    blocks[0].dmin = 0x0000;
    for (int j = 0; j < 4; j++) blocks[0].scales[j] = 1;

    // Row 1: d=0.5, dmin=0, sc=2, all qs lo=3, qh=0 → q5=3
    blocks[1].d = 0x3800;
    blocks[1].dmin = 0x0000;
    for (int j = 0; j < 4; j++) blocks[1].scales[j] = 2;
    for (int i = 0; i < 128; i++) blocks[1].qs[i] = 0x33;  // lo=3, hi=3

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    // Row 0: all q5=0, sc=1, d=1 → val=0 for all → sum=0
    assert(fabsf(out[0]) < 1.0f);

    // Row 1: sub-blocks 0-3 have sc=2, q5=3, d=0.5 → val=0.5*2*3=3.0
    // Sub-blocks 4-7 have sc=0 → val=0
    // Sum = 128 * 3.0 + 128 * 0 = 384
    assert(fabsf(out[1] - 384.0f) < 1.0f);

    free(blocks);
    printf("PASSED\n");
}

// --- Test Q3_K dequantization ---

static void test_q3k_dequant(void) {
    printf("test_q3k_dequant... ");

    BnBlockQ3K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;  // FP16 1.0

    // Set all scales to 32 (→ signed scale = 32-32 = 0)
    // With scale 0, all outputs should be 0 regardless of quant values
    // Pack scales: all bytes to encode scale=32 for all 16 sub-blocks
    // After q3k_unpack_scales: each byte should be 32
    // aux[0] = 0x20202020 (each byte=32=0x20, low nibble=0, upper=2)
    // Actually the packing is complex. Let's just set scales to make
    // the unpacked result be 32 (→ signed 0). With all zeros output is trivially 0.

    // Instead, test with a simple case: set scale[0] to known value
    // Use direct bytes: scales[0..11]
    // After unpacking, scales[0] should give a known 6-bit value.
    // The simplest test: all scale bytes = 0 → all 6-bit scales = 0
    // signed scale = 0 - 32 = -32
    // With qs all 0 and hmask all 0:
    //   2bit = 0, hmask=0 → q3 = 0 - 4 = -4
    //   val = d * (0 - 32) * (-4) = 1.0 * -32 * -4 = 128
    float out[256];
    bn_quant_dequant_q3k(&block, out);

    // All scales = 0 → signed scale = -32
    // All qs = 0 → 2-bit = 0, all hmask = 0 → q3 = 0 - 4 = -4
    // val = 1.0 * -32 * -4 = 128
    assert(fabsf(out[0] - 128.0f) < 0.01f);
    assert(fabsf(out[100] - 128.0f) < 0.01f);
    assert(fabsf(out[255] - 128.0f) < 0.01f);

    // Test with hmask bit set: q3 = 0 - 0 = 0 → val = -32 * 0 = 0
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;
    block.hmask[0] = 0xFF;  // All 8 bits set
    bn_quant_dequant_q3k(&block, out);
    // Element 0 reads hm[0] & 1 = 1, so q3 = 0 - 0 = 0, val = -32 * 0 = 0
    assert(fabsf(out[0]) < 0.01f);

    printf("PASSED\n");
}

// --- Test Q3_K matvec ---

static void test_q3k_matvec(void) {
    printf("test_q3k_matvec... ");

    // 2 rows × 256 cols
    int rows = 2, cols = 256;
    BnBlockQ3K *blocks = (BnBlockQ3K *)calloc(rows, sizeof(BnBlockQ3K));

    // Row 0: d=1.0, all scales=0 (→ signed -32), all qs=0, all hmask=0xFF
    // hmask all set → q3 = 0 - 0 = 0 → val = -32 * 0 = 0 → dot = 0
    blocks[0].d = 0x3C00;
    memset(blocks[0].hmask, 0xFF, 32);

    // Row 1: d=0.5, all scales=0 (→ signed -32), all qs=0, hmask=0
    // hmask=0 → q3 = 0 - 4 = -4, val = 0.5 * -32 * -4 = 64
    // dot with all-1 x = 256 * 64 = 16384
    blocks[1].d = 0x3800;  // FP16 0.5

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q3_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    int8_t x_q[256];
    bn_quant_matvec(out, &W, x, x_q, NULL);

    assert(fabsf(out[0]) < 1.0f);
    assert(fabsf(out[1] - 16384.0f) < 10.0f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Quant Tests ===\n");
    test_fp16_conversion();
    test_tq2_dequant();
    test_tq1_dequant();
    test_ternary_matvec();
    test_i2s_matvec();
    test_q8_matvec();
    test_q4_matvec();
    test_q8_matvec_multiblock();
    test_matvec_batch();
    test_matvec_threaded();
    test_q6k_dequant();
    test_q6k_matvec();
    test_q8k_dequant();
    test_q8k_matvec();
    test_q4k_dequant();
    test_q4k_matvec();
    test_q5k_dequant();
    test_q5k_matvec();
    test_q3k_dequant();
    test_q3k_matvec();
    printf("All quant tests passed!\n");
    return 0;
}
