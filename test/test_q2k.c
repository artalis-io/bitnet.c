#include "quant_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __ARM_NEON
#define Q2K_RANGE bn_quant_q2k_neon_range
#elif defined(__AVX2__)
#define Q2K_RANGE bn_quant_q2k_avx2_range
#else
#define Q2K_RANGE bn_quant_q2k_scalar_range
#endif

static void test_q2k_dequant(void) {
    printf("test_q2k_dequant... ");

    // All zeros: qs=0, scales=0 → all outputs = -dmin * (scale_byte >> 4)
    // With scales=0, output = d*0*(q&0xF) - dmin*0*(q>>4) = 0
    BnBlockQ2K block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;    // 1.0 in FP16
    block.dmin = 0x0000; // 0.0 in FP16

    float out[256];
    bn_quant_dequant_q2k(&block, out);

    // d=1.0, dmin=0.0, scales all 0, qs all 0 → all outputs = 0
    for (int i = 0; i < 256; i++)
        assert(fabsf(out[i]) < 0.01f);

    // Test with known values: d=1.0, dmin=0.5
    // scales[0] = 0x21 → scale=1, min=2 → dl=1.0, ml=1.0
    // qs[0] = 0x03 → q2=3 at shift=0 → val = 1.0*3 - 1.0 = 2.0
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;    // 1.0
    block.dmin = 0x3800;  // 0.5
    block.scales[0] = 0x21; // low nibble=1 (scale), high nibble=2 (min)
    block.qs[0] = 0x03;     // bits [1:0] = 3

    bn_quant_dequant_q2k(&block, out);
    // dl = 1.0 * 1 = 1.0, ml = 0.5 * 2 = 1.0
    // q2 = 3 → val = 1.0 * 3 - 1.0 = 2.0
    assert(fabsf(out[0] - 2.0f) < 0.01f);

    printf("PASSED\n");
}

static void test_q2k_matvec(void) {
    printf("test_q2k_matvec... ");

    int rows = 2, cols = 256;
    BnBlockQ2K *blocks = (BnBlockQ2K *)calloc(rows, sizeof(BnBlockQ2K));

    // Row 0: all zeros, d=1.0, dmin=0.0, scales=0 → dot = 0
    blocks[0].d = 0x3C00;
    blocks[0].dmin = 0x0000;

    // Row 1: d=0.5, dmin=0.0, scales[all]=0x01 (scale=1, min=0)
    // qs all 0xFF → each 2-bit field = 3, val = 0.5*1*3 = 1.5
    // dot with x=ones: 256 * 1.5 = 384
    blocks[1].d = 0x3800;    // 0.5
    blocks[1].dmin = 0x0000;
    memset(blocks[1].scales, 0x01, 16);  // scale=1, min=0
    memset(blocks[1].qs, 0xFF, 64);      // all 2-bit values = 3

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q2_K, rows, cols, 1.0f };

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;

    float out[2];
    BnQ2KCtx ctx = { out, &W, x };
    Q2K_RANGE(&ctx, 0, rows);

    assert(fabsf(out[0]) < 1.0f);
    assert(fabsf(out[1] - 384.0f) < 1.0f);

    free(blocks);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Q2_K Tests ===\n");
    test_q2k_dequant();
    test_q2k_matvec();
    printf("All Q2_K tests passed!\n");
    return 0;
}
