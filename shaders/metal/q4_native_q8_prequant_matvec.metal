#include <metal_stdlib>
using namespace metal;

// Q4_0 repacked matvec using prequantized Q8 activation blocks.

#define TILE_ROWS 16u

#define DQ4(w, sh) char4( \
    char(int(((w) >> (sh))       & 0xF) - 8), \
    char(int(((w) >> ((sh) + 4)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 8)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 12))& 0xF) - 8))

static inline float dot_char4(char4 a, char4 b) {
    return dot(float4(a), float4(b));
}

static inline float q4_q8_dot(uint w0, uint w1, uint w2, uint w3,
                              device const char4 *xq) {
    float acc = 0.0f;
    acc += dot_char4(DQ4(w0,  0), xq[0]);
    acc += dot_char4(DQ4(w0, 16), xq[1]);
    acc += dot_char4(DQ4(w1,  0), xq[2]);
    acc += dot_char4(DQ4(w1, 16), xq[3]);
    acc += dot_char4(DQ4(w2,  0), xq[4]);
    acc += dot_char4(DQ4(w2, 16), xq[5]);
    acc += dot_char4(DQ4(w3,  0), xq[6]);
    acc += dot_char4(DQ4(w3, 16), xq[7]);
    return acc;
}

kernel void q4_native_q8_prequant_matvec(
    device const uint  *weights  [[buffer(0)]],
    device const char  *x_q      [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out      [[buffer(3)]],
    constant uint      *p        [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * TILE_ROWS : wid.x * TILE_ROWS;
    uint token = (extra > 0) ? 0 : wid.y;

    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = rows * blocks_per_row;
    uint x_base = token * cols;
    uint scale_base = token * blocks_per_row;

    float acc = 0.0f;

    if (global_row < rows) {
        uint row_block_base = global_row * blocks_per_row;
        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            uint block_idx = row_block_base + b;
            float d = as_type<float>(weights[block_idx]);
            float dx = x_scales[scale_base + b];
            uint nib_base = total_blocks + block_idx * 4;
            device const char4 *xqb = (device const char4 *)(x_q + x_base + b * 32);
            float idot = q4_q8_dot(weights[nib_base], weights[nib_base + 1],
                                   weights[nib_base + 2], weights[nib_base + 3],
                                   xqb);
            acc += d * dx * idot;
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        uint bias_offset = p[4];
        if (bias_offset > 0)
            acc += as_type<float>(weights[bias_offset + global_row]);
        out[out_offset + token * rows + global_row] = acc;
    }
}

#undef DQ4
#undef TILE_ROWS
