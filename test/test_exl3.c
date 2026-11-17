/* Frozen expected hashes: upstream exllamav3 CUDA reconstruction at
   bd1f519172406a9560bd5c90e61000691bb69a07, SM120, 2026-11-11.
   Scalar/half2 all 65536 entries agree. Tile fixtures are generated below,
   independently decoded by the unchanged upstream kernel before this port.
   Hash byte order is little endian regardless of host byte order. */
#include "quant.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

static uint64_t hash_word(uint64_t h, uint16_t w) {
    h = (h ^ (w & 255u)) * UINT64_C(1099511628211);
    return (h ^ (w >> 8)) * UINT64_C(1099511628211);
}

static uint32_t step(uint32_t *state) {
    *state ^= *state << 13; *state ^= *state >> 17; *state ^= *state << 5;
    return *state;
}

static void test_hadamard(void) {
    const uint64_t expected[3] = { UINT64_C(0xf0b28fb5f5583bc3),
        UINT64_C(0xbf350a793a2de53e), UINT64_C(0xc650ace65354606a) };
    for (int mode = 0; mode < 3; ++mode) {
        uint64_t hash = UINT64_C(14695981039346656037);
        uint32_t state = 20261113, scale_state = 20261114;
        for (int block = 0; block < 384; ++block) {
            uint16_t x[128], scales[128], out[128], inplace[128];
            for (int i = 0; i < 128; ++i) {
                uint32_t v = step(&state);
                float f = block < 128 ? (i == block ? 1.0f : 0.0f) :
                    ((int)(v % 8193) - 4096) / 1024.0f;
                x[i] = bn_fp32_to_fp16(f);
                scales[i] = block < 256 ? 0x3c00 : bn_fp32_to_fp16(
                    ((int)(step(&scale_state) % 4097) - 2048) / 1024.0f);
            }
            const uint16_t *pre = mode == 1 ? scales : NULL;
            const uint16_t *post = mode == 2 ? scales : NULL;
            assert(!bn_quant_exl3_hadamard128_fp16(x, out, pre, post));
            memcpy(inplace, x, sizeof(x));
            assert(!bn_quant_exl3_hadamard128_fp16(inplace, inplace, pre, post));
            assert(!memcmp(inplace, out, sizeof(out)));
            for (int i = 0; i < 128; ++i) hash = hash_word(hash, out[i]);
        }
        assert(hash == expected[mode]);
    }
    uint16_t in[128] = {0}, out[128], saved[128];
    memset(out, 0xa5, sizeof(out)); memcpy(saved, out, sizeof(out));
    assert(bn_quant_exl3_hadamard128_fp16(NULL, out, NULL, NULL) == -1);
    assert(bn_quant_exl3_hadamard128_fp16(in, NULL, NULL, NULL) == -1);
    assert(bn_quant_exl3_hadamard128_fp16(in, out, in, in) == -1);
    assert(!memcmp(out, saved, sizeof(out)));
}

static void test_matrix(void) {
    unsigned char *packed = malloc(64 * 768);
    uint16_t *out = malloc(1024 * 128 * sizeof(*out));
    assert(packed && out);
    uint32_t state = 20261111;
    for (int batch = 0; batch < 64; ++batch) {
        for (int i = 0; i < 384; ++i) {
            uint32_t v = step(&state);
            uint16_t w = batch == 0 ? 0 : batch == 1 ? 65535 :
                         batch == 2 ? (uint16_t)(i * 257) : (uint16_t)v;
            size_t off = (size_t)batch * 768 + (size_t)i * 2;
            packed[off] = (unsigned char)w;
            packed[off + 1] = (unsigned char)(w >> 8);
        }
    }
    assert(!bn_quant_exl3_mul1_decode_slice(packed, 64*768, 1024, 128,
                                            0, 128, out, 1024*128));
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < 1024*128; ++i) hash = hash_word(hash, out[i]);
    assert(hash == UINT64_C(0xfbaf6b6f7a5320e8));
    assert(!bn_quant_exl3_mul1_decode_slice(packed, 64*768, 1024, 128,
                                            16, 64, out, 1024*64));
    hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < 1024*64; ++i) hash = hash_word(hash, out[i]);
    assert(hash == UINT64_C(0x0a644334801ee154));
    const struct { int k, n, start, count; size_t bytes, elems; } bad[] = {
        {0,128,0,128,64*768,1024*128}, {-16,128,0,128,64*768,1024*128},
        {1023,128,0,128,64*768,1024*128}, {1024,127,0,128,64*768,1024*128},
        {1024,128,-16,128,64*768,1024*128}, {1024,128,1,16,64*768,1024*128},
        {1024,128,16,128,64*768,1024*128}, {1024,128,0,0,64*768,1024*128},
        {1024,128,0,127,64*768,1024*128}, {1024,128,0,128,64*768-1,1024*128},
        {1024,128,0,128,64*768+1,1024*128}, {1024,128,0,128,64*768,1024*128-1},
        {INT_MAX-15,INT_MAX-15,0,INT_MAX-15,SIZE_MAX,0}
    };
    memset(out, 0xa5, 1024*128*sizeof(*out));
    for (size_t i = 0; i < sizeof(bad)/sizeof(bad[0]); ++i)
        assert(bn_quant_exl3_mul1_decode_slice(packed, bad[i].bytes, bad[i].k,
            bad[i].n, bad[i].start, bad[i].count, out, bad[i].elems) == -1);
    assert(bn_quant_exl3_mul1_decode_slice(NULL,64*768,1024,128,0,128,out,1024*128)==-1);
    assert(bn_quant_exl3_mul1_decode_slice(packed,64*768,1024,128,0,128,NULL,1024*128)==-1);
    for (int i = 0; i < 1024*128; ++i) assert(out[i] == 0xa5a5);
    free(out); free(packed);
}

int main(void) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < 65536; ++i)
        hash = hash_word(hash, bn_quant_exl3_mul1_codebook((uint16_t)i));
    assert(hash == UINT64_C(0x55e1c748ed9f0cd7));
    uint32_t state = 20261111;
    uint16_t out[256];
    unsigned char packed[769];
    hash = UINT64_C(14695981039346656037);
    for (int batch = 0; batch < 64; ++batch) {
        for (int i = 0; i < 384; ++i) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            uint16_t w = batch == 0 ? 0 : batch == 1 ? 65535 :
                         batch == 2 ? (uint16_t)(i * 257) : (uint16_t)state;
            packed[1 + i * 2] = (unsigned char)w;
            packed[2 + i * 2] = (unsigned char)(w >> 8);
        }
        for (int tile = 0; tile < 8; ++tile) {
            memset(out, 0xa5, sizeof(out));
            assert(bn_quant_exl3_mul1_decode_tile(packed + 1 + tile * 96, out) == 0);
            for (int i = 0; i < 256; ++i) hash = hash_word(hash, out[i]);
        }
    }
    assert(hash == UINT64_C(0xe81183ee3114d1d0));
    assert(bn_quant_exl3_mul1_decode_tile(NULL, out) == -1);
    assert(bn_quant_exl3_mul1_decode_tile(packed, NULL) == -1);
    test_hadamard();
    test_matrix();
    puts("EXL3 codebook, tiles, matrix slices and FP16 Hadamard fixtures PASSED");
    return 0;
}
