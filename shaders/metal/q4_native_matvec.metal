#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE matvec — reads original GGUF format (18 bytes/block)
// 32 rows/tile, 8 threads/row, simd_shuffle_xor reduction.
//
// GGUF Q4_0 block: [FP16 scale (2B)][16 packed nibble bytes]
// qs[j] = (elem[j] & 0xF) | (elem[j+16] << 4)
//
// Pre-scaling trick (from llama.cpp): read nibbles as uint16, mask with
// 0x000F/0x00F0/0x0F00/0xF000 to extract 4 nibbles without shifting.
// Pre-divide input vector by 1/16/256/4096 to compensate for bit positions.
// Bias correction: d * (sumy * -8 + acc) per block.
//
// Each thread processes 4 uint16 words = 8 nibble bytes = 16 elements per block
// (first half + second half interleaved in the same bytes).
// 2 "halves" per block: bytes 0-7 cover elements {0-7, 16-23},
// bytes 8-15 cover elements {8-15, 24-31}.
// Each thread handles ONE half (8 bytes) and processes 16 of 32 elements.
// With 8 threads per row at stride 8 blocks, this covers all blocks.
//
// Dispatch: (ceil(rows/32), n_tokens, 1)

kernel void q4_native_matvec(
    device const char  *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant uint      *p       [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * 32 : wid.x * 32;
    uint token = (extra > 0) ? 0 : wid.y;

    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;  // blocks per row
    uint x_base = token * cols;

    float acc = 0.0f;

    if (global_row < rows) {
        device const char *row_data = weights + (size_t)global_row * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *block = row_data + (size_t)b * 18;
            float d = float(*(device const half *)block);

            // Read packed nibbles as uint16 (4 words = 8 bytes = 16 elements)
            // First half: bytes 0-7 → elements {0-7} (lower nibbles) + {16-23} (upper nibbles)
            device const ushort *qs0 = (device const ushort *)(block + 2);
            // Second half: bytes 8-15 → elements {8-15} + {24-31}
            device const ushort *qs1 = (device const ushort *)(block + 2 + 8);

            uint eb = x_base + b * 32;

            // First half: elements {0-7, 16-23}
            float sumy0 = 0.0f;
            float a0[4] = {0.f, 0.f, 0.f, 0.f};
            for (ushort i = 0; i < 8; i += 2) {
                float y0 = x[eb + i];
                float y1 = x[eb + i + 1];
                float y16 = x[eb + i + 16];
                float y17 = x[eb + i + 17];
                sumy0 += y0 + y1 + y16 + y17;
                ushort w = qs0[i / 2];
                a0[0] += y0          * float(w & 0x000Fu);
                a0[1] += (y1 / 256)  * float(w & 0x0F00u);
                a0[2] += (y16 / 16)  * float(w & 0x00F0u);
                a0[3] += (y17 / 4096)* float(w & 0xF000u);
            }

            // Second half: elements {8-15, 24-31}
            float sumy1 = 0.0f;
            float a1[4] = {0.f, 0.f, 0.f, 0.f};
            for (ushort i = 0; i < 8; i += 2) {
                float y8 = x[eb + 8 + i];
                float y9 = x[eb + 8 + i + 1];
                float y24 = x[eb + 24 + i];
                float y25 = x[eb + 24 + i + 1];
                sumy1 += y8 + y9 + y24 + y25;
                ushort w = qs1[i / 2];
                a1[0] += y8          * float(w & 0x000Fu);
                a1[1] += (y9 / 256)  * float(w & 0x0F00u);
                a1[2] += (y24 / 16)  * float(w & 0x00F0u);
                a1[3] += (y25 / 4096)* float(w & 0xF000u);
            }

            acc += d * ((sumy0 + sumy1) * (-8.f)
                        + a0[0] + a0[1] + a0[2] + a0[3]
                        + a1[0] + a1[1] + a1[2] + a1[3]);
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = acc;
}
