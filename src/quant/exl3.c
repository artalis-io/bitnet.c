/*
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted scalar wire decoding from exllamav3 quant/{codebook,exl3_dq}.cuh
and reconstruct.cu, plus quant/hadamard_inner.cuh, commit bd1f519172406a9560bd5c90e61000691bb69a07.
*/
#include "quant.h"

uint16_t bn_quant_exl3_mul1_codebook(uint16_t index) {
    uint32_t x = (uint32_t)index * UINT32_C(0x83dcd12d);
    unsigned sum = (x & 255u) + ((x >> 8) & 255u) +
                   ((x >> 16) & 255u) + (x >> 24);
    /* The original half FMA has exact value numerator * 2^-18.
       Both numerator and scaling are exactly representable in FP32;
       perform the single final round-to-nearest-even conversion to FP16. */
    int numerator = 1774 * (1024 + (int)sum) - 2721792;
    return bn_fp32_to_fp16((float)numerator * 0x1p-18f);
}

static uint32_t read_le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int bn_quant_exl3_mul1_decode_tile(const void *packed, uint16_t *out) {
    if (!packed || !out) return -1;
    const unsigned char *bytes = packed;
    uint32_t words[24];
    uint16_t lanes[32][8];
    for (int i = 0; i < 24; ++i) words[i] = read_le32(bytes + i * 4);
    for (int lane = 0; lane < 32; ++lane) {
        int end_first = (lane * 8 + 257) * 3;
        int first = (end_first - 16) / 32;
        int last = (end_first + 20) / 32;
        int shift = (last + 1) * 32 - (end_first + 21);
        uint64_t merged = ((uint64_t)words[first % 24] << 32) |
                          words[last % 24];
        uint32_t high = (uint32_t)(merged >> shift);
        uint32_t low = (uint32_t)(merged >> (shift + 12));
        for (int j = 0; j < 4; ++j) {
            lanes[lane][j] = bn_quant_exl3_mul1_codebook(
                (uint16_t)(low >> ((3 - j) * 3)));
            lanes[lane][j + 4] = bn_quant_exl3_mul1_codebook(
                (uint16_t)(high >> ((3 - j) * 3)));
        }
    }
    /* Undo the original tensor-core fragment permutation, retaining the
       rotated weight basis. The suh/svh sidecars belong to projection ops. */
    for (int lane = 0; lane < 32; ++lane) {
        if (lane & 4) continue;
        int r = (lane % 4) * 2, c = (lane / 8) * 2;
        for (int j = 0; j < 8; ++j) {
            int row = r + (j % 2) + 8 * ((j / 2) % 2);
            int col = c + 8 * (j / 4);
            out[row * 16 + col] = lanes[lane][j];
            out[row * 16 + col + 1] = lanes[lane + 4][j];
        }
    }
    return 0;
}

int bn_quant_exl3_mul1_decode_slice(const void *packed, size_t packed_bytes,
                                    int k, int n, int start, int count,
                                    uint16_t *out, size_t out_elems) {
    if (!packed || !out || k <= 0 || n <= 0 || start < 0 || count <= 0 ||
        k % 16 || n % 16 || start % 16 || count % 16 ||
        start > n || count > n - start) return -1;
    size_t kt = (size_t)k / 16, nt = (size_t)n / 16;
    if (kt > SIZE_MAX / nt || kt * nt > SIZE_MAX / 96 ||
        packed_bytes != kt * nt * 96 ||
        (size_t)k > SIZE_MAX / (size_t)count ||
        (size_t)k * (size_t)count > SIZE_MAX / sizeof(*out) ||
        out_elems < (size_t)k * (size_t)count) return -1;
    const unsigned char *bytes = packed;
    uint16_t tile[256];
    for (size_t row = 0; row < kt; ++row) {
        for (int col = 0; col < count / 16; ++col) {
            size_t off = (row * nt + (size_t)(start / 16 + col)) * 96;
            bn_quant_exl3_mul1_decode_tile(bytes + off, tile);
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c)
                    out[(row * 16 + (size_t)r) * (size_t)count +
                        (size_t)col * 16 + (size_t)c] = tile[r * 16 + c];
        }
    }
    return 0;
}

int bn_quant_exl3_hadamard128_fp16(const uint16_t *input, uint16_t *out,
                                  const uint16_t *pre, const uint16_t *post) {
    if (!input || !out || (pre && post)) return -1;
    float h[32][4], next[32][4];
    for (int lane = 0; lane < 32; ++lane) {
        float v[4];
        for (int j = 0; j < 4; ++j) {
            int i = lane * 4 + j;
            uint16_t x = input[i];
            if (pre) x = bn_fp32_to_fp16(bn_fp16_to_fp32(x) *
                                        bn_fp16_to_fp32(pre[i]));
            v[j] = bn_fp16_to_fp32(x);
        }
        float s0 = v[0] + v[1], d0 = v[0] - v[1];
        float s1 = v[2] + v[3], d1 = v[2] - v[3];
        h[lane][0] = s0 + s1; h[lane][1] = d0 + d1;
        h[lane][2] = s0 - s1; h[lane][3] = d0 - d1;
    }
    /* Preserve the upstream warp-XOR addition order, using FP32 throughout
       the butterfly. In particular do not round each butterfly to FP16. */
    for (int mask = 1; mask < 32; mask <<= 1) {
        for (int lane = 0; lane < 32; ++lane)
            for (int j = 0; j < 4; ++j)
                next[lane][j] = ((lane & mask) ? -h[lane][j] : h[lane][j]) +
                                 h[lane ^ mask][j];
        for (int lane = 0; lane < 32; ++lane)
            for (int j = 0; j < 4; ++j) h[lane][j] = next[lane][j];
    }
    for (int lane = 0; lane < 32; ++lane) {
        for (int j = 0; j < 4; ++j) {
            int i = lane * 4 + j;
            uint16_t x = bn_fp32_to_fp16(h[lane][j] * 0.088388347648f);
            if (post) x = bn_fp32_to_fp16(bn_fp16_to_fp32(x) *
                                         bn_fp16_to_fp32(post[i]));
            out[i] = x;
        }
    }
    return 0;
}
