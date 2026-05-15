#include <metal_stdlib>
using namespace metal;

// Q4_0 repacked split matvec for stacked projection buffers.
// Buffer layout: [f32 scales: n_blocks][nibble u32s: n_blocks * 4][optional f32 bias].

#define UQ4(w, sh) float4( \
    float(((w) >> (sh))        & 0xF), \
    float(((w) >> ((sh) + 4))  & 0xF), \
    float(((w) >> ((sh) + 8))  & 0xF), \
    float(((w) >> ((sh) + 12)) & 0xF))

static inline float q4_block_dot(uint w0, uint w1, uint w2, uint w3,
                                 float s, device const float4 *xp) {
    float4 sx0 = xp[0] + xp[1] + xp[2] + xp[3];
    float4 sx1 = xp[4] + xp[5] + xp[6] + xp[7];
    float4 sx = sx0 + sx1;
    float sumx = (sx.x + sx.y) + (sx.z + sx.w);
    float acc = 0.0f;
    acc += dot(UQ4(w0,  0), xp[0]);
    acc += dot(UQ4(w0, 16), xp[1]);
    acc += dot(UQ4(w1,  0), xp[2]);
    acc += dot(UQ4(w1, 16), xp[3]);
    acc += dot(UQ4(w2,  0), xp[4]);
    acc += dot(UQ4(w2, 16), xp[5]);
    acc += dot(UQ4(w3,  0), xp[6]);
    acc += dot(UQ4(w3, 16), xp[7]);
    return s * (acc - 8.0f * sumx);
}

kernel void q4_matvec_split(device const uint  *weights [[buffer(0)]],
                            device const float *x       [[buffer(1)]],
                            device float       *out0    [[buffer(2)]],
                            device float       *out1    [[buffer(3)]],
                            device float       *out2    [[buffer(4)]],
                            constant uint      *p       [[buffer(5)]],
                            uint3 wid [[threadgroup_position_in_grid]],
                            uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1];
    uint split1 = p[2], split2 = p[3];
    uint bias_offset = p[4];
    uint off0 = p[5], off1 = p[6], off2 = p[7];

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
            float s = as_type<float>(weights[block_idx]);
            uint nib_base = total_blocks + block_idx * 4;
            uint w0 = weights[nib_base];
            uint w1 = weights[nib_base + 1];
            uint w2 = weights[nib_base + 2];
            uint w3 = weights[nib_base + 3];

            device const float4 *xp = (device const float4 *)(x + b * 32);
            acc += q4_block_dot(w0, w1, w2, w3, s, xp);
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
            out0[off0 + global_row] = acc;
        }
    }
}

#undef UQ4
