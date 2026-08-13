#include <metal_stdlib>
using namespace metal;

static inline float4 q4_low(device const uchar *q) {
    return float4(q[0] & 15, q[1] & 15, q[2] & 15, q[3] & 15);
}

static inline float4 q4_high(device const uchar *q) {
    return float4(q[0] >> 4, q[1] >> 4, q[2] >> 4, q[3] >> 4);
}

static inline float q4_native_block_dot(device const uchar *block,
                                        device const float4 *xp) {
    float4 sx0 = xp[0] + xp[1] + xp[2] + xp[3];
    float4 sx1 = xp[4] + xp[5] + xp[6] + xp[7];
    float4 sx = sx0 + sx1;
    float sumx = (sx.x + sx.y) + (sx.z + sx.w);
    device const uchar *q = block + 2;
    float acc = 0.0f;
    acc += dot(q4_low(q + 0), xp[0]);
    acc += dot(q4_low(q + 4), xp[1]);
    acc += dot(q4_low(q + 8), xp[2]);
    acc += dot(q4_low(q + 12), xp[3]);
    acc += dot(q4_high(q + 0), xp[4]);
    acc += dot(q4_high(q + 4), xp[5]);
    acc += dot(q4_high(q + 8), xp[6]);
    acc += dot(q4_high(q + 12), xp[7]);
    return float(*(device const half *)block) * (acc - 8.0f * sumx);
}

kernel void q4_native_matvec(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_start = extra > 0 ? (wid.x + wid.y * extra) * 32 : wid.x * 32;
    uint token = extra > 0 ? 0 : wid.y;
    uint row_lane = lid.x & 7;
    uint global_row = tile_start + (lid.x >> 3);
    uint blocks_per_row = cols >> 5;
    uint row_bytes = blocks_per_row * 18;
    float acc = 0.0f;
    if (global_row < rows) {
        device const uchar *row = weights + global_row * row_bytes;
        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            device const float4 *xp =
                (device const float4 *)(x + token * cols + b * 32);
            acc += q4_native_block_dot(row + b * 18, xp);
        }
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    if (row_lane == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = acc;
}
