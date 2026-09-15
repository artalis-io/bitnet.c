#ifndef BN_CUDA_KQUANT_REFERENCE_H
#define BN_CUDA_KQUANT_REFERENCE_H
#include "quant.h"
#include "gguf.h"
#include <stdint.h>
#include <stddef.h>
static size_t kquant_fixture_block_size(int type) {
    return type == 12 ? sizeof(BnBlockQ4K) : type == 13 ? sizeof(BnBlockQ5K) : sizeof(BnBlockQ6K);
}
static void kquant_fixture_weights(void *data, int type, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        uint16_t d = (uint16_t)(0x2400 + (b % 17) * 37);
        uint16_t m = (uint16_t)(0x2000 + (b % 11) * 19);
        if (type == 14) {
            BnBlockQ6K *w = (BnBlockQ6K *)data + b;
            w->d = d;
            for (int i = 0; i < 16; i++) w->scales[i] = (int8_t)((int)((b * 7 + i * 13) % 255) - 127);
            for (int i = 0; i < 128; i++) w->ql[i] = (uint8_t)(b * 29 + i * 13);
            for (int i = 0; i < 64; i++) w->qh[i] = (uint8_t)(b * 17 + i * 37);
        } else {
            uint8_t *scales, *qs;
            if (type == 12) {
                BnBlockQ4K *w = (BnBlockQ4K *)data + b;
                w->d = d; w->dmin = m; scales = w->scales; qs = w->qs;
            } else {
                BnBlockQ5K *w = (BnBlockQ5K *)data + b;
                w->d = d; w->dmin = m; scales = w->scales; qs = w->qs;
                for (int i = 0; i < 32; i++) w->qh[i] = (uint8_t)(b * 17 + i * 37);
            }
            for (int i = 0; i < 12; i++) scales[i] = (uint8_t)(b * 17 + i * 11);
            for (int i = 0; i < 128; i++) qs[i] = (uint8_t)(b * 29 + i * 13);
        }
    }
}
static void kquant_fixture_input(float *x, size_t count) {
    for (size_t i = 0; i < count; i++) x[i] = (float)((int)((i * 37) % 257) - 128) / 256.0f;
}
/* Independent GGML CUDA F32-output hashes, llama.cpp
 * 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120. */
typedef struct {
    int type, rows, cols, tokens;
    uint64_t hash;
} BnCudaKQuantReferenceCase;
static const BnCudaKQuantReferenceCase cuda_kquant_reference[] = {
    {12,127,256,8,UINT64_C(0x41986fb95bd54ddd)},
    {12,128,768,17,UINT64_C(0x2fef5822e30c59a7)},
    {12,129,5120,33,UINT64_C(0x3b39642e08f34ed2)},
    {12,128,5120,51,UINT64_C(0xb9f5d35bcc771ff6)},
    {12,129,768,129,UINT64_C(0x7adf59ec237e7b79)},
    {12,24064,256,51,UINT64_C(0x0b0ea15e68bc44fd)},
    {12,24192,256,129,UINT64_C(0xd141c792ed8498ab)},
    {12,129,5376,6,UINT64_C(0xa5e4d6a7235cbb19)},
    {12,129,5376,7,UINT64_C(0xf40e5d5eca8dcb6b)},
    {12,129,5376,21,UINT64_C(0x8ec75291a6120de3)},
    {12,5376,21504,21,UINT64_C(0x7671756a1601f504)},
    {12,8192,5376,27,UINT64_C(0x460fd02af77e6261)},
    {12,5376,16384,21,UINT64_C(0x4c8899072c0a9037)},
    {13,127,256,8,UINT64_C(0x250612ce082a50d3)},
    {13,128,768,17,UINT64_C(0xe84817d29a05dac7)},
    {13,129,5120,33,UINT64_C(0xbc094a9b4796e926)},
    {13,128,5120,51,UINT64_C(0x0b2708fb06b29b98)},
    {13,129,768,129,UINT64_C(0x6d5bf37205173596)},
    {13,24064,256,51,UINT64_C(0x3b6cec277c70d196)},
    {13,24192,256,129,UINT64_C(0xf77d309709c41dd6)},
    {13,129,5376,6,UINT64_C(0x4449e9e8362d46ee)},
    {13,129,5376,7,UINT64_C(0x5776a3920f2c3aa5)},
    {13,129,5376,21,UINT64_C(0xa808991329c5d6b7)},
    {13,5376,21504,21,UINT64_C(0x559e6f382f16ce53)},
    {13,8192,5376,27,UINT64_C(0x9b4e8dd54b8d46e0)},
    {13,5376,16384,21,UINT64_C(0x8ce8353dbb52d3f6)},
    {14,127,256,8,UINT64_C(0x3115d0b00237c576)},
    {14,128,768,17,UINT64_C(0x7e93deb157fd52a5)},
    {14,129,5120,33,UINT64_C(0xb629b0655349ec9e)},
    {14,128,5120,51,UINT64_C(0xbf1044e6c5c733a3)},
    {14,129,768,129,UINT64_C(0x12227c7e1a4634fa)},
    {14,24064,256,51,UINT64_C(0x5286234da2f50167)},
    {14,24192,256,129,UINT64_C(0x2b6ec4bb9a785b2a)},
    {14,129,5376,21,UINT64_C(0xc700b7f0acfdab7f)},
    {14,5376,21504,21,UINT64_C(0x98e0a18a350418ff)},
    {14,8192,5376,27,UINT64_C(0xcef0f6c5df6ce05a)},
    {14,5376,16384,21,UINT64_C(0x77f9284d36f4799a)},
};
#endif
