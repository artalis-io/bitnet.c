#ifndef BN_CUDA_GATED_MMF_FIXTURE_H
#define BN_CUDA_GATED_MMF_FIXTURE_H
#include <stddef.h>
/* Rational synthetic inputs shared with the independent GGML generator. */
static inline void gated_mmf_fixture(float *raw, float *key, float *value,
    float *wq, float *wk, int nt, int nh, int nk, int seed) {
    int stride = 2 * nh * 256 + 7;
    for (int t = 0; t < nt; t++) {
        for (int i = 0; i < stride; i++) raw[(size_t)t * stride + i] = 12345.0f;
        for (int h = 0; h < nh; h++) for (int d = 0; d < 256; d++) {
            int i = h * 256 + d;
            raw[(size_t)t * stride + h * 512 + d] =
                (float)((i * 17 + t * 13 + seed) % 251 - 125) / 79.0f;
            raw[(size_t)t * stride + h * 512 + 256 + d] =
                (float)((i * 7 + t * 19 + seed) % 97 - 48) / 17.0f;
        }
    }
    for (size_t i = 0; i < (size_t)nt * nk * 256; i++) {
        key[i] = (float)((i * 11 + seed) % 241) / 97.0f - 1.25f;
        value[i] = (float)((i * 23 + seed) % 239) / 89.0f - 1.375f;
    }
    for (int i = 0; i < (seed ? nh : 1) * 256; i++)
        wq[i] = 0.5f + (float)(i % 17) / 16.0f;
    for (int i = 0; i < (seed ? nk : 1) * 256; i++)
        wk[i] = 0.75f + (float)(i % 13) / 32.0f;
}
#endif
