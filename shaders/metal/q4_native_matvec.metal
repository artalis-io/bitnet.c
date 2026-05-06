#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE matvec — reads original GGUF format (18 bytes/block)
// 32 rows/tile, 8 threads/row, simd_shuffle_xor reduction.
// Pre-scaling trick with float4 vectorized x loads.
//
// GGUF Q4_0 block: [FP16 scale (2B)][16 packed nibble bytes]
// qs[j] = (elem[j] & 0xF) | (elem[j+16] << 4)
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

    uint nb = cols >> 5;
    uint x_base = token * cols;

    float acc = 0.0f;

    if (global_row < rows) {
        device const char *row_data = weights + (size_t)global_row * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *block = row_data + (size_t)b * 18;
            float d = float(*(device const half *)block);

            // float4 vectorized x loads (8 loads = 32 elements)
            device const float4 *xp = (device const float4 *)(x + x_base + b * 32);
            float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            float4 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];

            // Compute sum of all 32 x values for bias correction
            float sumy = (x0.x+x0.y+x0.z+x0.w) + (x1.x+x1.y+x1.z+x1.w)
                       + (x2.x+x2.y+x2.z+x2.w) + (x3.x+x3.y+x3.z+x3.w)
                       + (x4.x+x4.y+x4.z+x4.w) + (x5.x+x5.y+x5.z+x5.w)
                       + (x6.x+x6.y+x6.z+x6.w) + (x7.x+x7.y+x7.z+x7.w);

            // Pre-scaling: read uint16 words, mask nibbles, multiply by pre-scaled x
            // Layout: qs[j] = elem[j](lower) | elem[j+16](upper)
            // uint16 at byte 2k: bits 0-3=elem[2k], 4-7=elem[2k+16], 8-11=elem[2k+1], 12-15=elem[2k+1+16]
            device const ushort *qs = (device const ushort *)(block + 2);

            float sum = 0.0f;

            // Process 8 uint16 words = 16 nibble bytes = 32 elements
            // Word 0: elements {0, 16, 1, 17} via masks 0x000F, 0x00F0, 0x0F00, 0xF000
            sum += x0.x          * float(qs[0] & 0x000Fu);
            sum += (x0.y * 0.00390625f)  * float(qs[0] & 0x0F00u);
            sum += (x4.x * 0.0625f)   * float(qs[0] & 0x00F0u);
            sum += (x4.y * 0.000244140625f) * float(qs[0] & 0xF000u);

            // Word 1: elements {2, 18, 3, 19}
            sum += x0.z          * float(qs[1] & 0x000Fu);
            sum += (x0.w * 0.00390625f)  * float(qs[1] & 0x0F00u);
            sum += (x4.z * 0.0625f)   * float(qs[1] & 0x00F0u);
            sum += (x4.w * 0.000244140625f) * float(qs[1] & 0xF000u);

            // Word 2: elements {4, 20, 5, 21}
            sum += x1.x          * float(qs[2] & 0x000Fu);
            sum += (x1.y * 0.00390625f)  * float(qs[2] & 0x0F00u);
            sum += (x5.x * 0.0625f)   * float(qs[2] & 0x00F0u);
            sum += (x5.y * 0.000244140625f) * float(qs[2] & 0xF000u);

            // Word 3: elements {6, 22, 7, 23}
            sum += x1.z          * float(qs[3] & 0x000Fu);
            sum += (x1.w * 0.00390625f)  * float(qs[3] & 0x0F00u);
            sum += (x5.z * 0.0625f)   * float(qs[3] & 0x00F0u);
            sum += (x5.w * 0.000244140625f) * float(qs[3] & 0xF000u);

            // Word 4: elements {8, 24, 9, 25}
            sum += x2.x          * float(qs[4] & 0x000Fu);
            sum += (x2.y * 0.00390625f)  * float(qs[4] & 0x0F00u);
            sum += (x6.x * 0.0625f)   * float(qs[4] & 0x00F0u);
            sum += (x6.y * 0.000244140625f) * float(qs[4] & 0xF000u);

            // Word 5: elements {10, 26, 11, 27}
            sum += x2.z          * float(qs[5] & 0x000Fu);
            sum += (x2.w * 0.00390625f)  * float(qs[5] & 0x0F00u);
            sum += (x6.z * 0.0625f)   * float(qs[5] & 0x00F0u);
            sum += (x6.w * 0.000244140625f) * float(qs[5] & 0xF000u);

            // Word 6: elements {12, 28, 13, 29}
            sum += x3.x          * float(qs[6] & 0x000Fu);
            sum += (x3.y * 0.00390625f)  * float(qs[6] & 0x0F00u);
            sum += (x7.x * 0.0625f)   * float(qs[6] & 0x00F0u);
            sum += (x7.y * 0.000244140625f) * float(qs[6] & 0xF000u);

            // Word 7: elements {14, 30, 15, 31}
            sum += x3.z          * float(qs[7] & 0x000Fu);
            sum += (x3.w * 0.00390625f)  * float(qs[7] & 0x0F00u);
            sum += (x7.z * 0.0625f)   * float(qs[7] & 0x00F0u);
            sum += (x7.w * 0.000244140625f) * float(qs[7] & 0xF000u);

            acc += d * (sumy * (-8.f) + sum);
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        uint bias_offset = p[4];
        if (bias_offset > 0)
            acc += as_type<float>(((device const uint *)weights)[bias_offset + global_row]);
        out[out_offset + token * rows + global_row] = acc;
    }
}
