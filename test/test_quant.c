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
    // l=0, m=0: (qs[0] >> 0) & 3 = 2 → value = (2-1)*1.0 = +1.0
    assert(fabsf(out[0] - 1.0f) < 1e-6f);
    // l=0, m=1: (qs[1] >> 0) & 3 = 1 → value = (1-1)*1.0 = 0.0
    assert(fabsf(out[1] - 0.0f) < 1e-6f);

    // l=1, m=0 (index 32): (qs[0] >> 2) & 3 = 0 → value = (0-1)*1.0 = -1.0
    assert(fabsf(out[32] - (-1.0f)) < 1e-6f);

    // l=2, m=0 (index 64): (qs[0] >> 4) & 3 = 1 → value = (1-1)*1.0 = 0.0
    assert(fabsf(out[64] - 0.0f) < 1e-6f);

    // l=3, m=0 (index 96): (qs[0] >> 6) & 3 = 2 → value = (2-1)*1.0 = +1.0
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

    // For TQ1_0, encoding 5 values {-1,0,1} → {0,1,2} as base-3:
    // All zeros → all values = -1 (mapped: 0,0,0,0,0 = base3 number 0)
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

    // Create a small TQ2_0 matrix: 2 rows × 256 cols
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
    ternary_matvec(out, &W, x);

    // Row 0: all +1 → sum of 256 ones * 1.0 = 256.0 * scale(1.0)
    assert(fabsf(out[0] - 256.0f) < 1e-3f);

    // Row 1: all 0 → 0.0
    assert(fabsf(out[1] - 0.0f) < 1e-3f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Quant Tests ===\n");
    test_fp16_conversion();
    test_tq2_dequant();
    test_tq1_dequant();
    test_ternary_matvec();
    printf("All quant tests passed!\n");
    return 0;
}
