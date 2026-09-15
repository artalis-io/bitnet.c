#ifndef BN_CUDA_MXFP4_FIXTURE_H
#define BN_CUDA_MXFP4_FIXTURE_H
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Seed >= 100 selects blockwise activation scales and zero blocks. The same
 * fixture is shared with the independent llama.cpp reference generator. */
static void mxfp4_fixture_weights(void *data, size_t elements, int seed) {
    uint8_t *w = (uint8_t *)data;
    assert(elements % 32 == 0);
    seed %= 100;
    for (size_t b = 0; b < elements / 32; b++) {
        w[b * 17] = (uint8_t)(115 + (b * 13 + (unsigned)seed) % 17);
        for (int j = 0; j < 16; j++)
            w[b * 17 + 1 + j] = (uint8_t)(((b * 7 + j * 3 + seed) & 15) |
                (((b * 11 + j * 5 + seed + 1) & 15) << 4));
    }
}

static void mxfp4_fixture_input(float *x, size_t count, int seed) {
    int expanded = seed >= 100;
    seed %= 100;
    for (size_t i = 0; i < count; i++) {
        int block = (int)(i / 32);
        float value = (float)((int)((i * 37 + (unsigned)seed * 11) % 257) - 128) / 32.0f;
        x[i] = expanded ? (block % 11 == 0 ? 0.0f :
            ldexpf(value, (block * 7 + seed) % 25 - 12)) : value;
    }
}
#endif
