#include <metal_stdlib>
using namespace metal;

// Q4_0 repacked split matvec using prequantized Q8 activation blocks.

#define DQ4(w, sh) char4( \
    char(int(((w) >> (sh))       & 0xF) - 8), \
    char(int(((w) >> ((sh) + 4)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 8)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 12))& 0xF) - 8))

static inline int q4_q8_dot(uint w0, uint w1, uint w2, uint w3,
                            device const char4 *xq) {
    #define DOT4(a, b) (int((a).x) * int((b).x) + int((a).y) * int((b).y) + \
                       int((a).z) * int((b).z) + int((a).w) * int((b).w))
    int acc = 0;
    acc += DOT4(DQ4(w0,  0), xq[0]);
    acc += DOT4(DQ4(w0, 16), xq[1]);
    acc += DOT4(DQ4(w1,  0), xq[2]);
    acc += DOT4(DQ4(w1, 16), xq[3]);
    acc += DOT4(DQ4(w2,  0), xq[4]);
    acc += DOT4(DQ4(w2, 16), xq[5]);
    acc += DOT4(DQ4(w3,  0), xq[6]);
    acc += DOT4(DQ4(w3, 16), xq[7]);
    #undef DOT4
    return acc;
}

kernel void q4_matvec_split_q8_prequant(
    device const uint  *weights  [[buffer(0)]],
    device const char  *x_q      [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out0     [[buffer(3)]],
    device float       *out1     [[buffer(4)]],
    device float       *out2     [[buffer(5)]],
    constant uint      *p        [[buffer(6)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1];
    uint split1 = p[2], split2 = p[3];
    uint bias_offset = p[4];
    uint off1 = p[6], off2 = p[7];

    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = rows * blocks_per_row;
    float acc = 0.0f;

    if (global_row < rows) {
        uint row_block_base = global_row * blocks_per_row;
        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            uint block_idx = row_block_base + b;
            float d = as_type<float>(weights[block_idx]);
            float dx = x_scales[b];
            if (dx == 0.0f)
                continue;
            uint nib_base = total_blocks + block_idx * 4;
            device const char4 *xqb = (device const char4 *)(x_q + b * 32);
            int idot = q4_q8_dot(weights[nib_base], weights[nib_base + 1],
                                 weights[nib_base + 2], weights[nib_base + 3],
                                 xqb);
            acc += d * dx * float(idot);
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        if (bias_offset > 0)
            acc += as_type<float>(weights[bias_offset + global_row]);
        if (split2 > 0 && global_row >= split2) {
            out2[off2 + global_row - split2] = acc;
        } else if (global_row >= split1) {
            out1[off1 + global_row - split1] = acc;
        } else {
            out0[global_row] = acc;
        }
    }
}

#undef DQ4
